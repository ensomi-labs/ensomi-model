"""Missing content is an observation condition, never a fabricated physical prefix."""
from dataclasses import replace

import torch
import pytest

from ensomi_model.research.controlled_audio_continuation.model import ControlledAudioModel
from ensomi_model.research.joint_audio_continuation.intervals import IntervalExample
from ensomi_model.research.planned_audio_continuation.intervals import collate_interval,score_interval
from ensomi_model.research.typed_audio_continuation.controls import ControlSchedule,ControlSpan
from controlled_audio_continuation.test_sampling import source
from planned_audio_continuation.test_distribution import config


DEVICES=['cpu',pytest.param('mps',marks=pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS unavailable'))]


def fixture(device):
    torch.set_num_threads(1);torch.manual_seed(728)
    net=ControlledAudioModel(replace(config(),lookahead=16,bounded_head=True,
        condition_full_holds=True,minimum_action_gap_ms=60),layout_modulation=True).to(device).eval()
    c=source();batch=collate_interval(IntervalExample(c,0,1001),net.config,device,recovery=net.recovery)
    controls=ControlSchedule((ControlSpan(0,1001,stars=3,ln_fraction=.4),))
    return net,c,batch,controls


@pytest.mark.parametrize('device',DEVICES)
def test_missing_content_cannot_leak_hidden_values_but_preserves_exact_state_and_BOS(device):
    net,c,batch,controls=fixture(device);x=batch.inputs;b=x.base;n=len(b.row_times)
    audio=torch.randn(n,net.config.conditioned_audio_width,device=device)
    first=torch.randn(n,2,net.config.hidden,device=device,requires_grad=True)
    changed=first.detach()+torch.randn_like(first)*2
    cc=audio.new_tensor(controls.at(b.row_times.cpu().numpy(),encoding=net.control_encoding))
    def run(history,visible=None):
        return net.planned_row_log_probs(audio,history,b.row_exact,b.row_legal,b.occupancy,
            x.row_preview,x.consequence_local,x.consequence_timing,control=cc,
            response_allowed=x.response_allowed,history_visible=visible)
    all_visible=torch.ones(n,dtype=torch.bool,device=device)
    missing=~all_visible;non_BOS=~b.row_exact[:,0,-1].bool()
    assert non_BOS.any() and (~non_BOS).any()
    ordinary=run(first);explicit=run(first,all_visible)
    torch.testing.assert_close(ordinary,explicit,atol=0,rtol=0)
    masked=run(first,missing);other=run(changed,missing)
    torch.testing.assert_close(masked[non_BOS],other[non_BOS],atol=0,rtol=0)
    torch.testing.assert_close(masked[~non_BOS],ordinary[~non_BOS],atol=0,rtol=0)
    expected=run(torch.where(non_BOS[:,None,None],net.temporal.boundary[1][None,None],first))
    torch.testing.assert_close(masked,expected,atol=0,rtol=0)
    assert torch.equal(torch.isfinite(masked),torch.isfinite(ordinary))
    value=masked[non_BOS][torch.isfinite(masked[non_BOS])].sum()
    gh,gb=torch.autograd.grad(value,(first,net.temporal.boundary))
    assert torch.equal(gh,torch.zeros_like(gh))
    assert gb[1].abs().sum()>0 and gb[0].abs().sum()==0


@pytest.mark.parametrize('device',DEVICES)
def test_visibility_changes_only_row_observation_under_factual_interval_scoring(device):
    net,c,batch,controls=fixture(device)
    encoded=net.encode_audio(torch.as_tensor(c.mel,device=device)[None])
    def hidden(kind,times,indices):
        return dict(history_visible=torch.zeros(len(times),dtype=torch.bool,device=device)) if kind=='row' else {}
    regular=score_interval(net,batch.inputs,None,controls=controls,encoded_full=encoded)
    masked=score_interval(net,batch.inputs,None,controls=controls,encoded_full=encoded,history_options=hidden)
    torch.testing.assert_close(regular.head,masked.head,atol=0,rtol=0)
    torch.testing.assert_close(regular.release,masked.release,atol=0,rtol=0)
    valid=torch.isfinite(regular.row)
    assert torch.equal(valid,torch.isfinite(masked.row))
    assert (regular.row[valid]-masked.row[valid]).abs().max()>1e-4
