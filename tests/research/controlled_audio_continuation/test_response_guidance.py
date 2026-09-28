import numpy as np
import torch

from ensomi_model.research.controlled_audio_continuation.generation import ControlledSession
from ensomi_model.research.controlled_audio_continuation.response_guidance import ResponseGuidedSession
from ensomi_model.research.controlled_audio_continuation.joint_release import release_logits
from ensomi_model.research.player_response.action_response import ActionEnvelope,ActionResponseState,KINDS,TAUS_MS
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
from ensomi_model.research.typed_audio_continuation.controls import ControlSchedule,ControlSpan
from controlled_audio_continuation.test_joint_release import model,fixture


def reference():
    return ActionEnvelope((2.,6.),tuple(np.full((2,len(TAUS_MS),len(KINDS)),1.).tolist()),
        'fixture',(4000.,),((1.,),(1.,)))


def test_zero_guidance_preserves_native_output_and_forks_own_response_state():
    torch.manual_seed(24)
    net=model().eval()
    controls=ControlSchedule((ControlSpan(0,1201,stars=4,ln_fraction=.5),),net.style_names)
    mel=np.random.default_rng(24).normal(size=(121,128)).astype(np.float32)
    direct=ControlledSession(net,mel,1200,controls,seed=9,ln_feedback=None)
    guided=ResponseGuidedSession(net,mel,1200,controls,seed=9,ln_feedback=None,
        response_reference=reference(),response_strength=0.)
    direct.publish_to(1200);guided.publish_to(400)
    before=guided.response_state
    branch=guided.fork()
    branch.publish_to(1000)
    assert guided.response_state is before and guided.coverage == 400
    np.testing.assert_array_equal(guided.response_state.values,before.values)
    guided.publish_to(1200)
    assert guided.rows == direct.rows
    assert guided.response_state.replay == guided.replay
    rebuilt=ActionResponseState.from_rows(guided.rows,1200)
    np.testing.assert_allclose(rebuilt.values,guided.response_state.values,atol=1e-12)


def test_release_cost_changes_event_odds_before_the_subset_is_fixed():
    net,queries,context,encoded,controls=fixture()
    plain=release_logits(net,queries,context,encoded,controls,preference=None)
    def cost(times,actions):
        return np.broadcast_to((actions==3).sum(-1)*3.,(len(times),len(actions)))
    guided=release_logits(net,queries,context,encoded,controls,preference=None,candidate_cost=cost)
    assert bool((guided < plain-2.9).all())


def test_active_guidance_reads_the_forked_prefix_and_does_not_rewrite_H():
    torch.manual_seed(24)
    net=model().eval()
    controls=ControlSchedule((ControlSpan(0,1201,stars=4,ln_fraction=.5),),net.style_names)
    mel=np.random.default_rng(24).normal(size=(121,128)).astype(np.float32)
    source=ResponseGuidedSession(net,mel,1200,controls,seed=9,ln_feedback=None,
        response_reference=reference(),response_strength=4.)
    source.publish_to(400)
    branch=source.fork()
    branch.publish_to(1000)
    actions=((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))
    assert not np.allclose(source.response_cost([1100],actions),branch.response_cost([1100],actions))
    assert source.response_state.time_ms == 400 and branch.response_state.time_ms == 1000
    source.publish_to(1200)
    direct=ControlledSession(net,mel,1200,controls,seed=9,ln_feedback=None)
    direct.publish_to(1200)
    H=lambda s:[r.time_ms for r in s.rows if any(a in (1,2) for a in r.actions)]
    assert H(source)==H(direct)
