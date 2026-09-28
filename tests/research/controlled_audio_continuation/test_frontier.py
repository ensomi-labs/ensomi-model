import numpy as np
import torch
import pytest

from ensomi_model.research.controlled_audio_continuation.frontier import (
    ResponsePlanner, FrontierPlanning, NoAcceptableContinuation,
)
from ensomi_model.research.controlled_audio_continuation.generation import ControlledSession
from ensomi_model.research.player_response.envelope import AttackEnvelope
from ensomi_model.research.player_response.state import CommittedPlayState
from ensomi_model.research.player_response.action_response import ActionEnvelope,ActionResponseState,KINDS,TAUS_MS
from ensomi_model.research.typed_audio_continuation.controls import ControlSchedule, ControlSpan
from controlled_audio_continuation.test_ownership import model


def sessions():
    torch.set_num_threads(1)
    torch.manual_seed(272)
    net = model().eval()
    controls = ControlSchedule((ControlSpan(0, 6001, stars=4, ln_fraction=.6),), net.style_names)
    mel = np.random.default_rng(270).normal(size=(600, 128)).astype(np.float32)
    heads = tuple(range(50, 6000, 130))
    return [ControlledSession(net, mel, 6000, controls, seed=181, head_times=heads) for _ in range(2)]


def test_zero_response_preserves_native_trajectory_despite_private_lookahead():
    direct, start = sessions()
    reference = AttackEnvelope((1000.,), (2., 6.), ((1000.,), (1000.,)), 'loose-fixture')
    planner = ResponsePlanner(start, reference, seed=71)
    planner.publish_to(2000)
    prefix = tuple(planner.session.rows)
    assert all(r.time_ms <= 2000 for r in prefix)
    assert start.coverage == -1 and start.rows == []
    assert planner.state == CommittedPlayState.from_rows(prefix, 2000)
    planner.publish_to(6000)
    direct.publish_to(6000)
    assert planner.session.rows == direct.rows
    assert tuple(planner.session.rows[:len(prefix)]) == prefix
    assert not any(planner.state.replay.occupancy)
    assert all(len(d['proposals']) == 1 for d in planner.decisions)


def test_all_failed_candidates_preserve_published_history_and_do_not_commit_the_least_bad():
    _, start = sessions()
    reference = AttackEnvelope((1000., 4000.), (2., 6.), ((.5, .5), (.5, .5)), 'strict-fixture')
    planner = ResponsePlanner(start, reference, seed=83, config=FrontierPlanning(maximum_candidates=3))
    before=planner.state
    with pytest.raises(NoAcceptableContinuation):
        planner.publish_to(2000)
    decision = planner.decisions[0]
    assert len(decision['proposals']) == 3
    assert decision['selected'] is None
    assert all(p['excess_seconds'] > 0 and not p['accepted'] for p in decision['proposals'])
    assert planner.state is before and planner.session is start
    assert planner.state.replay == planner.session.replay
    assert start.rows == [] and start.coverage == -1
    assert decision['forecast_end_ms'] == 4000


def test_action_reference_can_publish_from_bos_without_an_unrequested_sentinel_scope():
    _,start=sessions()
    envelope=ActionEnvelope((2.,6.),tuple(np.full((2,len(TAUS_MS),len(KINDS)),1000.).tolist()),
        'loose-actions',(4000.,),((1.,),(1.,)))
    planner=ResponsePlanner(start,envelope,seed=11)
    planner.publish_to(2000)
    assert isinstance(planner.state,ActionResponseState)
    assert planner.session.coverage == planner.state.time_ms == 2000
    assert planner.decisions[0]['proposals'][0]['fully_scored']
    assert planner.decisions[0]['proposals'][0]['accepted']
    assert planner.decisions[0]['proposals'][0]['ranges'][0]['reference_horizon_ms'] == 4000
    assert planner.decisions[0]['proposals'][0]['ranges'][0]['work_limit'] == 1
    assert planner.state.replay == planner.session.replay


def test_longer_rolling_budget_rejects_a_future_that_passes_each_forecast():
    from copy import copy
    from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
    from ensomi_model.research.player_response.action_response import source_work
    chart=(CompleteRow(100,(1,0,0,0)),CompleteRow(4100,(1,0,0,0)))
    base=ActionEnvelope((2.,6.),tuple(np.ones((2,len(TAUS_MS),len(KINDS))).tolist()),'fixture')
    work=source_work([r.time_ms for r in chart],[r.actions for r in chart],base,4.)
    budget=float((max(work)+sum(work))/2)
    reference=ActionEnvelope(base.stars,base.maximum,'rolling-fixture',(4000.,8000.),
        ((budget,budget),(budget,budget)))
    class Session:
        duration_ms=6000
        controls=ControlSchedule((ControlSpan(0,6001,stars=4.),))
        def __init__(self):self.coverage=-1;self.rows=[]
        def fork(self,**options):return copy(self)
        def publish_to(self,end):
            self.coverage=end
            self.rows=[r for r in chart if r.time_ms<=end]
    planner=ResponsePlanner(Session(),reference,seed=1,config=FrontierPlanning(maximum_candidates=2))
    planner.publish_to(2000)
    committed=planner.session;state=planner.state;ledger=planner.work_events
    with pytest.raises(NoAcceptableContinuation):planner.publish_to(4000)
    assert planner.session is committed and planner.state is state and planner.work_events==ledger
    assert planner.session.coverage==2000 and planner.session.rows==list(chart[:1])
    for p in planner.decisions[-1]['proposals']:
        assert p['added_work']<budget
        assert p['rolling_work']['ranges'][0]['windows'][0]['accepted']
        assert not p['rolling_work']['ranges'][0]['windows'][1]['accepted']
