from dataclasses import replace

import numpy as np
import pytest
import torch

from ensomi_model.research.controlled_audio_continuation.joint_trace import collate_joint_trace, score_joint_trace
from ensomi_model.research.controlled_audio_continuation.sampling import replay_row_scores
from ensomi_model.research.joint_audio_continuation.intervals import IntervalExample
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
from ensomi_model.research.planned_audio_continuation.intervals import collate_interval, interval_losses, score_interval
from ensomi_model.research.typed_audio_continuation.controls import ControlSchedule, ControlSpan
from ensomi_model.research.typed_audio_continuation.program import Recovery
from controlled_audio_continuation.test_joint_release import model
from planned_audio_continuation.test_distribution import chart


@pytest.mark.parametrize('device',['cpu',pytest.param('mps',marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(),reason='MPS unavailable'))])
def test_partial_joint_trace_matches_source_clock_factors_and_gradients(device):
    torch.manual_seed(3701)
    net=model(hold_audio_width=4).to(device).eval()
    source=chart([0,100,250,400,650,900],[(2,0,0,0),(0,1,0,0),(3,0,0,0),
        (0,0,2,0),(1,0,0,0),(0,0,3,0)],1000)
    rows=tuple(source.source.row(i) for i in range(len(source.source.rows)))
    heads=tuple(r.time_ms for r in rows if any(a in (1,2) for a in r.actions))
    controls=ControlSchedule((ControlSpan(0,1001,stars=4,ln_fraction=.4),),net.style_names)
    encoded=net.encode_audio(torch.from_numpy(source.mel)[None].to(device))
    example=IntervalExample(source,0,701)
    batch=collate_interval(example,net.config,device,recovery=net.recovery,release_policy='r1_joint')
    expected=score_interval(net,batch.inputs,None,controls=controls,encoded_full=encoded)
    q=replay_row_scores(expected.row[:len(batch.row_index)],example,controls,ln_feedback=None)
    expected_lp=-interval_losses(expected,batch)[1].cpu().double()+q[torch.arange(len(batch.row_index)),batch.row_index.cpu()].sum()
    trace=collate_joint_trace(rows,heads,0,701,1000,net,device=device)
    actual=score_joint_trace(net,trace,controls,encoded)
    assert trace.row.rows[-1].time_ms==650 and trace.release.times[-1]==700
    torch.testing.assert_close(actual.log_probability,expected_lp.cpu().double(),atol=3e-4,rtol=3e-5)
    parameters=(net.release_log_scale,net.composition.readout[-1].weight)
    grads=[torch.autograd.grad(x,parameters,retain_graph=True) for x in (actual.log_probability,expected_lp)]
    for a,b in zip(*grads):torch.testing.assert_close(a,b,atol=3e-4,rtol=3e-5)
    assert abs(float(grads[0][0]))>0


def test_all_possible_release_or_survival_futures_have_unit_probability():
    torch.manual_seed(3702)
    net=model().eval();net.recovery=Recovery(1,1,1)
    net.config=replace(net.config,minimum_action_gap_ms=1)
    controls=ControlSchedule((ControlSpan(0,8,stars=4),),net.style_names)
    encoded=net.encode_audio(torch.zeros(1,3,128))
    probabilities=[]
    # The four possibilities on [2,5): release at 2/3/4 or remain genuinely open.
    for end in (2,3,4,None):
        rows=(CompleteRow(1,(2,0,0,0)),)
        if end is not None:rows+=(CompleteRow(end,(3,0,0,0)),)
        trace=collate_joint_trace(rows,(1,),2,5,7,net)
        value=score_joint_trace(net,trace,controls,encoded,recovery_preference=None)
        probabilities.append(value.log_probability.exp())
        if end is None:assert not len(value.row) and len(value.release)==3
    total=torch.stack(probabilities).sum()
    torch.testing.assert_close(total,torch.tensor(1.,dtype=torch.float64),atol=2e-7,rtol=2e-7)
    derivative=torch.autograd.grad(total,net.release_log_scale)[0]
    assert abs(float(derivative))<2e-7


def test_joint_likelihood_is_additive_across_no_row_and_control_boundaries():
    torch.manual_seed(3703)
    net=model().eval()
    rows=(CompleteRow(0,(2,2,2,2)),CompleteRow(75,(3,0,0,0)),
          CompleteRow(100,(1,0,0,0)),CompleteRow(200,(0,3,3,3)))
    controls=ControlSchedule((ControlSpan(0,301,stars=4),ControlSpan(127,193,ln_fraction=.2)),net.style_names)
    encoded=net.encode_audio(torch.zeros(1,31,128));heads=(0,100)
    full=score_joint_trace(net,collate_joint_trace(rows,heads,0,301,300,net),controls,encoded)
    assert len(full.release)>0
    pieces=[]
    for a,b in zip((0,73,76,127,193,225),(73,76,127,193,225,301)):
        trace=collate_joint_trace(rows,heads,a,b,300,net)
        pieces.append(score_joint_trace(net,trace,controls,encoded).log_probability)
    torch.testing.assert_close(sum(pieces),full.log_probability,atol=2e-5,rtol=2e-6)
