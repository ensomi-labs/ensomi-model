from dataclasses import replace

import torch
import pytest

from ensomi_model.research.segment_audio_continuation.bootstrap import fresh_model,source_log_probability
from ensomi_model.research.segment_audio_continuation.model import SegmentConfig
from ensomi_model.research.joint_audio_continuation.intervals import IntervalExample
from ensomi_model.research.typed_audio_continuation.controls import ControlSchedule,ControlSpan
from planned_audio_continuation.test_distribution import config,chart


def model(seed=281930,*,bounded=True):
    return fresh_model(torch.zeros(128),torch.ones(128),seed=seed,
        config=replace(config(),lookahead=16,bounded_head=bounded,condition_full_holds=True,minimum_action_gap_ms=20),
        segment_config=SegmentConfig(states=1,span_ms=4000,local_levels=1,
            hidden=24,code_width=8,audio_queries=2,continuous_context=True))


@pytest.mark.parametrize('bounded',[False,True])
def test_fresh_joint_learning_reaches_audio_H_and_complete_row_owners(bounded):
    net=model(bounded=bounded);net.train()
    c=chart([0,100,250,400,650,900],[(2,0,0,0),(0,1,0,0),(3,0,0,0),
        (0,0,2,0),(1,0,0,0),(0,0,3,0)],1000)
    controls=ControlSchedule((ControlSpan(0,1001,stars=4.,ln_fraction=.5),),net.style_names)
    opt=torch.optim.AdamW(net.parameters(),lr=1e-3)
    # Zero residual projections open on the first update. Re-encode full audio
    # with current weights before checking gradients through the global branch.
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        encoded=net.encode_audio(torch.from_numpy(c.mel)[None])
        score=source_log_probability(net,IntervalExample(c,0,1001),controls,encoded)
        assert score['head_events']==4 and score['row_events']==6
        assert torch.isfinite(score['joint'])
        (-score['joint']).backward()
        if step==0:opt.step()
    for root in ('audio_input','context','head_base' if bounded else 'timing','head_temporal','decoder','release_log_scale','row_control'):
        assert sum(float(p.grad.abs().sum()) for n,p in net.named_parameters()
            if n.split('.')[0]==root and p.grad is not None)>0,root


def test_H_likelihood_does_not_read_a_future_LN_tail_or_action_columns():
    net=model().eval()
    controls=ControlSchedule((ControlSpan(0,1001,stars=4.),),net.style_names)
    cases=[chart([0,100,tail,400,900],[(2,0,0,0),(0,1,0,0),(3,0,0,0),
        (0,0,2,0),(0,0,3,0)],1000) for tail in (250,300)]
    scores=[]
    for c in cases:
        encoded=net.encode_audio(torch.from_numpy(c.mel)[None])
        scores.append(source_log_probability(net,IntervalExample(c,0,1001),controls,encoded))
    torch.testing.assert_close(scores[0]['head'],scores[1]['head'],atol=1e-6,rtol=1e-7)
    assert not torch.allclose(scores[0]['materialization'],scores[1]['materialization'])


def test_fresh_seed_reproducibility_does_not_copy_a_trained_actor():
    a,b,c=model(31),model(31),model(32)
    for n,p in a.state_dict().items():torch.testing.assert_close(p,b.state_dict()[n],rtol=0,atol=0)
    assert not torch.equal(a.audio_input.weight,c.audio_input.weight)
    assert not torch.equal(a.decoder.output.weight,c.decoder.output.weight)
    assert a.segment_config.continuous_context and not a.context_projection.weight.count_nonzero()
