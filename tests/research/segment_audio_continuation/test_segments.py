import numpy as np
import pytest
import torch

from ensomi_model.research.bounded_typed_continuation.contract import ROW_ACTIONS
from ensomi_model.research.controlled_audio_continuation.joint_trace import score_joint_trace
from ensomi_model.research.controlled_audio_continuation.trace import score_row_trace
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
from ensomi_model.research.oracle_time_continuation.replay import ExactReplayState,commit
from ensomi_model.research.segment_audio_continuation.model import (
    BirthOrigins,SegmentConfig,initialize,configure_materializer,load_model,with_continuous_context,
)
from ensomi_model.research.segment_audio_continuation.segments import (
    collate_segment,marginal_log_probability,score_segment,next_boundary,plan_condition,prefix_observation,
    training_intervals,
)
from ensomi_model.research.controlled_audio_continuation.conditional_views import StyleWithoutLn
from ensomi_model.research.segment_audio_continuation.generation import SegmentSession
from ensomi_model.research.typed_audio_continuation.controls import ControlSchedule,ControlSpan
from controlled_audio_continuation.test_joint_release import model as parent_model
from planned_audio_continuation.test_distribution import chart


def model(*,continuous=False,states=2,reset=True):
    parent=parent_model(hold_audio_width=4)
    net=initialize(parent,segment_config=SegmentConfig(states=states,span_ms=300,
        local_levels=1,hidden=24,code_width=8,audio_queries=2,continuous_context=continuous,
        reset_local_history=reset))[0].eval()
    if continuous:
        with torch.no_grad():net.context_projection.weight.normal_(std=.05)
    return net


def test_mixture_selects_a_whole_sequence_instead_of_a_mode_per_row():
    conditional=torch.tensor([2*np.log(.9),2*np.log(.1)],dtype=torch.float64,requires_grad=True)
    likelihood,posterior=marginal_log_probability(torch.zeros(2),conditional)
    torch.testing.assert_close(likelihood.exp(),torch.tensor(.41,dtype=torch.float64))
    gradient=torch.autograd.grad(likelihood,conditional)[0]
    torch.testing.assert_close(gradient,torch.tensor([81/82,1/82],dtype=torch.float64))
    torch.testing.assert_close(posterior.exp(),gradient)
    assert abs(float(likelihood.detach().exp())-.25)>.1


def test_hidden_LN_scope_is_not_revealed_as_a_private_plan_boundary():
    controls=ControlSchedule((ControlSpan(0,8001,stars=4),ControlSpan(0,4250,ln_fraction=.2),
        ControlSpan(3000,5000,style={'tech':0.})),('tech',))
    assert next_boundary(4000,8000,controls,4000)==4250
    assert next_boundary(4000,8000,StyleWithoutLn(controls),4000)==5000


def test_style_selected_crop_cannot_train_an_unlabelled_piece_with_expert_weight():
    controls=ControlSchedule((ControlSpan(0,8001,stars=4),ControlSpan(0,4500,ln_fraction=.2),
        ControlSpan(3000,5000,style={'tech':0.})),('tech',))
    assert training_intervals(controls,0,8000,8000,4000,require_style=True)==(
        (3000,4000),(4000,4500),(4500,5000))
    hidden=StyleWithoutLn(controls)
    assert training_intervals(hidden,0,8000,8000,4000,require_style=True)==((3000,4000),(4000,5000))
    assert training_intervals(hidden,0,8000,8000,4000)==((0,3000),(3000,4000),(4000,5000),(5000,8000))


@pytest.mark.parametrize('device',['cpu',pytest.param('mps',marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(),reason='MPS unavailable'))])
def test_actual_open_segment_trains_prior_and_joint_decoder_without_H_gradients(device):
    torch.manual_seed(280930)
    net=model().to(device);configure_materializer(net)
    source=chart([0,100,250,400,650,900],[(2,0,0,0),(0,1,0,0),(3,0,0,0),
        (0,0,2,0),(1,0,0,0),(0,0,3,0)],1000)
    rows=tuple(source.source.row(i) for i in range(len(source.source.rows)))
    heads=tuple(r.time_ms for r in rows if any(a in (1,2) for a in r.actions))
    controls=ControlSchedule((ControlSpan(0,1001,stars=4,ln_fraction=.4),),net.style_names)
    encoded=net.encode_audio(torch.from_numpy(source.mel)[None].to(device)).detach()
    trace=collate_segment(rows,heads,300,700,1000,net,device=device)
    assert trace.row.history_boundary==1 and trace.release.times[-1]==699
    result=score_segment(net,trace,heads,controls,encoded)
    assert torch.isfinite(result['log_probability'])
    (-result['log_probability']).backward()
    for root in ('decoder','plan_prior','plan_codes','birth_encoder','release_log_scale'):
        assert sum(float(p.grad.abs().sum()) for n,p in net.named_parameters()
            if n.split('.')[0]==root and p.grad is not None)>0,root
    for root in ('head_base','audio_input','prior_temporal'):
        assert all(p.grad is None for n,p in net.named_parameters() if n.split('.')[0]==root)


@pytest.mark.parametrize('continuous,reset',[(False,True),(True,True),(True,False)])
def test_sampling_and_exact_conditional_scoring_agree_across_plan_boundaries(continuous,reset):
    torch.manual_seed(280931);net=model(continuous=continuous,reset=reset)
    with torch.no_grad():net.temporal.boundary[1].fill_(.3)
    mel=np.random.default_rng(51).normal(size=(101,128)).astype(np.float32)
    controls=ControlSchedule((ControlSpan(0,1001,stars=4,ln_fraction=.4),),net.style_names)
    heads=(0,120,250,430,550,750,900);rowscores={};clocks={}
    class Recorder(SegmentSession):
        def prefer_rows(self,q,legal,now):
            value=super().prefer_rows(q,legal,now);rowscores[now]=value.clone();return value
        def release_clock_logits(self,*args):
            values=super().release_clock_logits(*args)
            native,valid=args[1],args[-1]
            for t,v in zip(native.cpu().numpy().reshape(-1)[valid.reshape(-1)],
                           values.detach().cpu().numpy()[valid.reshape(-1)]):
                clocks[(self.replay.row_count,int(t))]=float(v)
            return values
    session=Recorder(net,mel,1000,controls,seed=91,head_times=heads)
    for end in (137,303,511,800,1000):session.publish_to(end)
    assert len(session.plan_events)==4 and not any(session.replay.occupancy)
    encoded=net.encode_audio(torch.from_numpy(mel)[None])
    times=np.array([r.time_ms for r in session.rows]);saw_open=False
    for p in session.plan_events:
        a,b=p['start_ms'],p['end_ms']
        trace=collate_segment(session.rows,heads,a,b,1000,net)
        replay,history=prefix_observation(net,trace.row.rows,a)
        _,context=plan_condition(net,encoded,replay,history,heads,controls,a,b,1000)
        policy=net.bind(p['code'],BirthOrigins.from_rows(trace.row.rows),context)
        scores=score_joint_trace(policy,trace,controls,encoded,recovery_preference=None)
        expected=torch.tensor([float(rowscores[r.time_ms][ROW_ACTIONS.index(r.actions)])
            for r in trace.row.rows if a<=r.time_ms<b],dtype=torch.float64)
        torch.testing.assert_close(scores.row,expected,atol=3e-5,rtol=3e-6)
        if trace.release is not None:
            r=trace.release
            logits=torch.tensor([clocks[(int(np.searchsorted(times,t,side='left')),int(t))]
                for t in r.times.numpy()],dtype=torch.float64)
            expected=torch.where(trace.event,-torch.nn.functional.softplus(-logits),-torch.nn.functional.softplus(logits))
            expected=torch.where(trace.forced,0.,expected)
            torch.testing.assert_close(scores.release,expected,atol=3e-5,rtol=3e-6)
        state=ExactReplayState()
        for r in trace.row.rows:state=commit(state,r)
        saw_open |= any(state.occupancy) and b<=1000
        if continuous and b-a>50:
            prefix=collate_segment(session.rows,heads,a,b-20,1000,net)
            rescored=score_segment(net,prefix,heads,controls,encoded,plan_end_ms=b)
            bound=net.bind(p['code'],BirthOrigins.from_rows(prefix.row.rows),context)
            expected=score_joint_trace(bound,prefix,controls,encoded,
                ln_feedback=None,recovery_preference=None).log_probability
            torch.testing.assert_close(rescored['conditional'][p['code']],expected)
    assert saw_open


@pytest.mark.parametrize('continuous',[False,True])
def test_fork_and_control_cut_preserve_committed_facts_and_other_branch(continuous):
    torch.manual_seed(280932);net=model(continuous=continuous)
    controls=ControlSchedule((ControlSpan(0,1001,stars=4),),net.style_names)
    session=SegmentSession(net,np.zeros((101,128),np.float32),1000,controls,seed=15)
    session.publish_to(100);fork=session.fork(retry_seed=75)
    context=session.plan_context
    assert fork.plan_context is context
    before=(tuple(session.rows),session.replay,session.plan_end,session.plan_rng.get_state().clone())
    fork.update_controls(ControlSpan(200,450,stars=3,ln_fraction=.1))
    assert fork.plan_end==200 and session.plan_end==300
    fork.publish_to(470)
    assert tuple(session.rows)==before[0] and session.replay==before[1]
    assert session.plan_end==before[2] and torch.equal(session.plan_rng.get_state(),before[3])
    assert session.plan_context is context
    if continuous:assert fork.plan_context is not context
    starts=[p['start_ms'] for p in fork.plan_events if p['kind']=='selected']
    assert 200 in starts and 450 in starts


def test_complete_row_reflection_compact_queries_and_checkpoint(tmp_path):
    torch.manual_seed(280933);net=model()
    x=torch.randn(5,2,net.config.hidden);code=net.plan_codes.weight[1]
    q=net.decoder(x,code)
    mirror=torch.tensor([ROW_ACTIONS.index(tuple(reversed(a))) for a in ROW_ACTIONS])
    torch.testing.assert_close(net.decoder(x.flip(1),code)[:,mirror],q)
    subset=torch.tensor([i for i,a in enumerate(ROW_ACTIONS) if not any(x in (1,2) for x in a)])
    torch.testing.assert_close(net.decoder(x,code,subset),q[:,subset],atol=2e-6,rtol=2e-6)
    path=tmp_path/'segment.pt';torch.save(net.checkpoint(),path);loaded=load_model(path)
    assert loaded.segment_config==net.segment_config
    for k,v in net.state_dict().items():torch.testing.assert_close(v,loaded.state_dict()[k],rtol=0,atol=0)


@pytest.mark.parametrize('continuous',[False,True])
def test_birth_row_context_uses_each_hands_relative_columns(continuous):
    torch.manual_seed(280934);net=model(continuous=continuous)
    controls=ControlSchedule((ControlSpan(0,1001,stars=4),),net.style_names)
    encoded=net.encode_audio(torch.zeros(1,101,128))
    rows=(CompleteRow(0,(2,0,1,0)),CompleteRow(120,(0,1,0,0)))
    reflected=tuple(CompleteRow(r.time_ms,tuple(reversed(r.actions))) for r in rows)
    outputs=[]
    for value in (rows,reflected):
        trace=collate_segment(value,(0,120),120,121,1000,net)
        replay,history=prefix_observation(net,value,120)
        _,context=plan_condition(net,encoded,replay,history,(0,120),controls,120,121,1000)
        policy=net.bind(1,BirthOrigins.from_rows(value),context)
        outputs.append(score_row_trace(policy,trace.row,controls,encoded,
            ln_feedback=None,recovery_preference=None))
    mirror=torch.tensor([ROW_ACTIONS.index(tuple(reversed(a))) for a in ROW_ACTIONS])
    torch.testing.assert_close(outputs[0],outputs[1][:,mirror],atol=3e-6,rtol=3e-6)


def test_zero_context_upgrade_preserves_native_actor_and_reload(tmp_path):
    torch.manual_seed(281901);parent=model(states=1);net=with_continuous_context(parent).eval()
    assert not net.context_projection.weight.count_nonzero()
    for name,value in parent.state_dict().items():torch.testing.assert_close(value,net.state_dict()[name],rtol=0,atol=0)
    mel=np.zeros((101,128),np.float32)
    controls=ControlSchedule((ControlSpan(0,1001,stars=4.),),net.style_names)
    sessions=[SegmentSession(m,mel,1000,controls,seed=121,head_times=(0,150,300,600,800)) for m in (parent,net)]
    for s in sessions:s.publish_to(1000)
    assert sessions[0].rows==sessions[1].rows
    path=tmp_path/'context.pt';torch.save(net.checkpoint(),path);loaded=load_model(path)
    assert loaded.segment_config.continuous_context
    for name,value in net.state_dict().items():torch.testing.assert_close(value,loaded.state_dict()[name],rtol=0,atol=0)
    old=parent.checkpoint();old['segment_config'].pop('continuous_context')
    torch.save(old,tmp_path/'old.pt');assert not load_model(tmp_path/'old.pt').segment_config.continuous_context


@pytest.mark.parametrize('device',['cpu',pytest.param('mps',marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(),reason='MPS unavailable'))])
def test_single_code_restores_future_context_effect_and_joint_likelihood_gradients(device):
    torch.manual_seed(281902);net=model(continuous=True,states=1).to(device);configure_materializer(net)
    source=chart([0,100,250,400,650,900],[(2,0,0,0),(0,1,0,0),(3,0,0,0),
        (0,0,2,0),(1,0,0,0),(0,0,3,0)],1000)
    rows=tuple(source.source.row(i) for i in range(len(source.source.rows)))
    heads=(0,100,400,650);controls=ControlSchedule((ControlSpan(0,1001,stars=4.),),net.style_names)
    encoded=net.encode_audio(torch.from_numpy(source.mel)[None].to(device)).detach()
    trace=collate_segment(rows,heads,300,700,1000,net,device=device)
    (-score_segment(net,trace,heads,controls,encoded)['log_probability']).backward()
    for root in ('context_projection','plan_prior','decoder','release_log_scale'):
        assert sum(float(p.grad.abs().sum()) for n,p in net.named_parameters()
            if n.split('.')[0]==root and p.grad is not None)>0,root
    assert all(p.grad is None for p in net.prior_temporal.parameters())
    replay,history=prefix_observation(net,rows,400)
    changed=encoded.clone();changed[:,55:70]+=3
    # Hold local query/audio/history fixed; intervene only on the future plan input.
    local=collate_segment(rows,heads,400,401,1000,net,device=device)
    scores=[];contexts=[]
    for audio in (encoded,changed):
        prior,context=plan_condition(net,audio,replay,history,heads,controls,400,700,1000)
        assert prior.item()==0
        contexts.append(context)
        scores.append(score_row_trace(net.bind(0,BirthOrigins.from_rows(rows),context),local.row,
            controls,encoded,ln_feedback=None,recovery_preference=None))
    assert not torch.allclose(contexts[0],contexts[1])
    legal=torch.isfinite(scores[0]);assert (scores[0][legal]-scores[1][legal]).abs().max()>1e-6
