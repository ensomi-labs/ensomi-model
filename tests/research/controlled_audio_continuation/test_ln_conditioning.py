"""LN-count control expressiveness and native probability reconstruction."""
from dataclasses import asdict,replace
import math

import numpy as np
import pytest
import torch

from ensomi_model.research.bounded_typed_continuation.contract import ROW_ACTIONS
from ensomi_model.research.controlled_audio_continuation.allocation import LnAmountFeedback
from ensomi_model.research.controlled_audio_continuation.generation import ControlledSession
from ensomi_model.research.controlled_audio_continuation.model import ControlledAudioModel,load_model
from ensomi_model.research.controlled_audio_continuation.sampling import replay_row_scores
from ensomi_model.research.joint_audio_continuation.intervals import IntervalExample
from ensomi_model.research.planned_audio_continuation.intervals import collate_interval,score_interval
from ensomi_model.research.typed_audio_continuation.controls import ControlSchedule,ControlSpan
from planned_audio_continuation.test_distribution import chart,config

DEVICES=['cpu',pytest.param('mps',marks=pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS unavailable'))]


def setup(mode='reference_tilt'):
    torch.set_num_threads(1)
    return ControlledAudioModel(replace(config(),lookahead=16,bounded_head=True,
        condition_full_holds=True,minimum_action_gap_ms=60),style_names=('tech',),ln_conditioning=mode).eval()


def inputs(net,device):
    c=chart([0,100,250,400],[(1,0,0,0),(0,2,0,0),(0,0,0,1),(0,3,0,0)],500)
    x=collate_interval(IntervalExample(c,0,501),net.config,device,recovery=net.recovery).inputs
    return (net.temporal.boundary[None],x.base.row_exact[:1],x.base.row_legal[:1],x.base.occupancy[:1],
            x.row_preview[:1],x.consequence_local[:1],x.consequence_timing[:1]),x.response_allowed[:1]


def condition(net,rho,device):
    cc=ControlSchedule((ControlSpan(0,501,stars=3,ln_fraction=rho),),net.style_names)
    return torch.tensor(cc.at([0],encoding=net.control_encoding),device=device)


@pytest.mark.parametrize('device',DEVICES)
def test_actual_LN_condition_can_change_contextual_type_odds_without_moving_count_ownership(device):
    net=setup().to(device)
    with torch.no_grad():
        for p in net.parameters():p.zero_()
        first,last=net.composition.readout[0],net.composition.readout[-1]
        first.weight[0,0]=1.
        control_start=first.in_features-net.row_control.in_features
        first.weight[0,control_start+1]=3.
        last.weight[:,0]=net.composition.longs
    args,support=inputs(net,device)
    actions=torch.tensor(ROW_ACTIONS,device=device)
    heads=((actions==1)|(actions==2)).sum(-1);longs=(actions==2).sum(-1);releases=(actions==3).sum(-1)
    def run(mode,a,rho):
        net.ln_conditioning=mode
        audio=torch.zeros(1,net.config.conditioned_audio_width,device=device);audio[0,0]=a
        return net.planned_row_log_probs(audio,*args,control=condition(net,rho,device),response_allowed=support)[0]
    def odds(q):
        return q[(heads==1)&(longs==1)&(releases==0)].logsumexp(0)-q[(heads==1)&(longs==0)&(releases==0)].logsumexp(0)
    effects={}
    for mode in ('reference_tilt','contextual_tilt','direct'):
        effects[mode]=torch.stack([odds(run(mode,a,.8))-odds(run(mode,a,.2)) for a in (-1.,1.)])
    torch.testing.assert_close(effects['reference_tilt'],torch.full((2,),2*math.log(4),device=device),atol=3e-5,rtol=1e-5)
    assert abs(float((effects['contextual_tilt'][1]-effects['contextual_tilt'][0]).detach()))>.5
    torch.testing.assert_close(effects['direct'],effects['contextual_tilt']-2*math.log(4),atol=3e-5,rtol=1e-5)
    gradient=torch.autograd.grad(effects['contextual_tilt'].sum(),net.composition.readout[0].weight)[0]
    assert torch.isfinite(gradient).all() and gradient[:,control_start+1].abs().sum()>0
    direct_gradient=torch.autograd.grad(effects['direct'].sum(),net.composition.readout[0].weight)[0]
    assert torch.isfinite(direct_gradient).all() and direct_gradient[:,control_start+1].abs().sum()>0
    for h in range(1,5):
        before=run('reference_tilt',1.,.8)[(heads==h)&(releases==0)].logsumexp(0)
        after=run('contextual_tilt',1.,.8)[(heads==h)&(releases==0)].logsumexp(0)
        torch.testing.assert_close(before,after,atol=3e-5,rtol=1e-5)


def test_unknown_or_reference_requests_and_checkpoint_defaults_preserve_the_old_law(tmp_path):
    torch.manual_seed(6106);net=setup()
    with torch.no_grad():
        net.composition.readout[-1].weight.normal_(std=.1)
        net.row_consequence.output.weight.normal_(std=.1)
    args,support=inputs(net,'cpu');audio=torch.randn(1,net.config.conditioned_audio_width)
    for rho in (None,net.ln_reference):
        values=[]
        for mode in ('reference_tilt','contextual_tilt','direct'):
            net.ln_conditioning=mode
            values.append(net.planned_row_log_probs(audio,*args,control=condition(net,rho,'cpu'),response_allowed=support))
        torch.testing.assert_close(values[0],values[1],atol=0,rtol=0)
        # The encoded reference fraction is float32, so the old analytic
        # logit shift can retain roundoff where direct uses exact zero.
        torch.testing.assert_close(values[0],values[2],atol=3e-6,rtol=1e-6)
    for mode in ('reference_tilt','contextual_tilt','direct'):
        net.ln_conditioning=mode;options=net.probability_options()
        assert ('ln_conditioning' in options)==(mode!='reference_tilt')
        path=tmp_path/(mode+'.pt')
        torch.save(dict(format='controlled-audio/v1',model_config=asdict(net.config),probability_options=options,model=net.state_dict()),path)
        restored=load_model(path)
        assert restored.ln_conditioning==mode
        expected=net.planned_row_log_probs(audio,*args,control=condition(net,.7,'cpu'),response_allowed=support)
        actual=restored.planned_row_log_probs(audio,*args,control=condition(restored,.7,'cpu'),response_allowed=support)
        torch.testing.assert_close(expected,actual,atol=0,rtol=0)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('feedback',[False,True])
@pytest.mark.parametrize('mode',['contextual_tilt','direct'])
def test_conditioned_native_rows_match_current_weight_replay_across_scopes(device,feedback,mode):
    torch.manual_seed(783);net=setup(mode).to(device)
    with torch.no_grad():net.composition.readout[-1].weight.normal_(std=.08)
    controls=ControlSchedule((ControlSpan(0,1001,stars=3,ln_fraction=.8),
        ControlSpan(350,650,ln_fraction=.2),ControlSpan(500,800,style={'tech':1.})),net.style_names)
    policy=LnAmountFeedback() if feedback else None
    mel=np.random.default_rng(70).normal(size=(100,128)).astype(np.float32);recorded=[]
    class Recorder(ControlledSession):
        def prefer_rows(self,*args):
            q=super().prefer_rows(*args);recorded.append(q.clone());return q
    with torch.inference_mode():
        session=Recorder(net,mel,1000,controls,seed=84,ln_feedback=policy,
            head_times=(0,100,200,300,450,550,650,750,900))
        for end in (333,600,777,1000):session.publish_to(end)
    generated=replace(chart([r.time_ms for r in session.rows],[r.actions for r in session.rows],1000),mel=mel)
    encoded=net.encode_audio(torch.from_numpy(mel)[None].to(device))
    def probabilities(c):
        rescored=[]
        for i in range(IntervalExample(c,0,137).count):
            example=IntervalExample(c,i,137)
            batch=collate_interval(example,net.config,device,recovery=net.recovery)
            raw=score_interval(net,batch.inputs,None,controls=controls,encoded_full=encoded)
            rescored.append(replay_row_scores(raw.row[:len(batch.row_index)],example,controls,ln_feedback=policy))
        return torch.cat(rescored)
    q=probabilities(generated)
    torch.testing.assert_close(q,torch.stack(recorded),atol=3e-5,rtol=3e-6)
    reflected=replace(chart(generated.source.rows['time'],
        generated.source.rows['actions'][:,::-1],1000),mel=mel)
    reverse=[ROW_ACTIONS.index(row[::-1]) for row in ROW_ACTIONS]
    torch.testing.assert_close(q,probabilities(reflected)[:,reverse],atol=3e-5,rtol=3e-6)
    loss=-q[torch.arange(len(q)),torch.tensor([ROW_ACTIONS.index(r.actions) for r in session.rows])].sum()
    gradient=torch.autograd.grad(loss,net.composition.readout[0].weight)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0


@pytest.mark.parametrize('device',DEVICES)
def test_direct_counts_use_neural_condition_without_an_analytic_amount_prior(device):
    net=setup('direct').to(device)
    with torch.no_grad():
        for p in net.parameters():p.zero_()
    args,support=inputs(net,device);audio=torch.zeros(1,net.config.conditioned_audio_width,device=device)
    values=[]
    for rho in (0.,.2,.8,1.,None):
        values.append(net.planned_row_log_probs(audio,*args,control=condition(net,rho,device),response_allowed=support))
    for value in values[1:]:torch.testing.assert_close(values[0],value,atol=0,rtol=0)
    net.ln_reference=.7
    changed=net.planned_row_log_probs(audio,*args,control=condition(net,.8,device),response_allowed=support)
    torch.testing.assert_close(values[0],changed,atol=0,rtol=0)
    shifted=net.planned_row_log_probs(audio,*args,control=condition(net,.8,device),
                                    response_allowed=support,ln_shift=1.)
    actions=torch.tensor(ROW_ACTIONS,device=device)
    one_head=torch.isin(actions,torch.tensor([1,2],device=device)).sum(-1)==1
    ln=(actions==2).sum(-1)==1
    def odds(q):return q[0,one_head&ln].logsumexp(0)-q[0,one_head&~ln].logsumexp(0)
    torch.testing.assert_close(odds(shifted)-odds(values[0]),torch.tensor(1.,device=device),atol=3e-6,rtol=1e-6)
