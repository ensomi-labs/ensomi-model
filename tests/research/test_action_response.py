import numpy as np
import pytest

from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
from ensomi_model.research.player_response.action_response import (
    ActionResponseState, ActionEnvelope, KINDS, SLICES, TAUS_MS,
    transition_impulses, source_peaks, action_response,
    source_work, window_work_maxima, recovery_potential,
    candidate_work,
)
from ensomi_model.research.player_response.state import CommittedPlayState
from ensomi_model.research.player_response.envelope import AttackEnvelope,sustained_response


def tap(t, lane=0):
    return CompleteRow(t,tuple(int(c == lane) for c in range(4)))


def reference(value=1.):
    return ActionEnvelope((2.,6.),tuple(np.full((2,len(TAUS_MS),len(KINDS)),value).tolist()),'fixture')


def test_incremental_state_matches_vector_calibration_and_split_silence():
    rows=[CompleteRow(0,(2,0,0,0)),tap(100,1),CompleteRow(400,(3,1,0,0)),
          tap(450),tap(1000),tap(1050),tap(16000,3),tap(20000,2)]
    state=ActionResponseState();peak=np.zeros_like(state.values);impulses=[]
    for row in rows:
        before=state.advance(row.time_ms).values
        state=state.observe(row)
        impulses.append((state.values-before)*np.asarray(TAUS_MS)[:,None]/1000)
        peak=np.maximum(peak,state.values)
    np.testing.assert_allclose(np.array(impulses)[:,0],transition_impulses(
        [r.time_ms for r in rows],[r.actions for r in rows]),atol=1e-12)
    np.testing.assert_allclose(source_peaks([r.time_ms for r in rows],[r.actions for r in rows]),
        np.stack([peak[:,s].max(-1) for s in SLICES],-1),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(state.advance(30000).values,state.advance(24000).advance(30000).values)


def test_old_attack_score_cannot_see_release_recovery_but_action_response_can():
    # Identical LN press/next attack; only the preceding real release moves.
    charts=[[CompleteRow(0,(2,0,0,0)),CompleteRow(r,(3,0,0,0)),tap(500)]
            for r in (200,475)]
    old=AttackEnvelope((1000.,),(2.,6.),((10.,),(10.,)),'loose')
    old_reports=[sustained_response(CommittedPlayState(),c,1500,old,[(-1,1500,4.)])[1] for c in charts]
    assert old_reports[0] == old_reports[1]
    states=[ActionResponseState.from_rows(c,500) for c in charts]
    assert states[1].values[0,8] > states[0].values[0,8]*10
    limit=np.full((2,len(TAUS_MS),len(KINDS)),1000.)
    limit[:,:,2]=2.
    env=ActionEnvelope((2.,6.),tuple(limit.tolist()),'RH-fixture')
    reports=[action_response(ActionResponseState(),c,1500,env,[(-1,1500,4.)])[1] for c in charts]
    assert reports[0]['acceptable'] and not reports[1]['acceptable']
    assert reports[1]['ranges'][0]['excess_by_kind']['RH_speed'] > 0


def test_one_interruption_does_not_clear_accumulated_finger_load():
    crowded=[tap(i*125) for i in range(16)]
    spread=[tap(i*125,i%4) for i in range(16)]
    a,b=[ActionResponseState.from_rows(rows,1900) for rows in (crowded,spread)]
    a_next=a.observe(tap(2000,1))
    assert a_next.values[2,4] > b.values[2,4]*2
    assert a_next.values[2,4] == pytest.approx(a.values[2,4]*np.exp(-100/4000))


def test_open_hold_retains_capacity_and_partner_role_through_silence():
    held=ActionResponseState.from_rows([CompleteRow(0,(2,0,0,0))],2000)
    assert held.replay.occupancy == (True,False,False,False)
    next_state=held.observe(tap(2100,1))
    assert next_state.values[0,25] == 4
    assert next_state.replay.open_ln_start_ms[0] == 0


def test_mirror_and_horizon_partition_preserve_the_same_response():
    rows=[tap(100),tap(200),CompleteRow(300,(2,0,0,0)),CompleteRow(400,(3,0,0,0)),tap(425)]
    env=reference()
    full,report=action_response(ActionResponseState(),rows,1500,env,[(-1,1500,4.)])
    prefix,left=action_response(ActionResponseState(),rows[:3],350,env,[(-1,350,4.)])
    _,right=action_response(prefix,rows[3:],1500,env,[(350,1500,4.)])
    mirror=[CompleteRow(r.time_ms,r.actions[::-1]) for r in rows]
    _,reflected=action_response(ActionResponseState(),mirror,1500,env,[(-1,1500,4.)])
    assert report['excess_seconds'] == pytest.approx(left['excess_seconds']+right['excess_seconds'])
    assert report['excess_seconds'] == pytest.approx(reflected['excess_seconds'])
    assert full.replay.occupancy == (False,)*4


def test_missing_request_is_unscored_not_a_safe_zero():
    _,report=action_response(ActionResponseState(),[tap(100)],500,reference(),[(-1,500,None)])
    assert not report['fully_scored'] and not report['acceptable']


def test_added_work_matches_online_updates_and_the_recovery_energy_identity():
    rows=[tap(100),tap(200),tap(300),tap(450)]
    env=reference()
    terminal,report=action_response(ActionResponseState(),rows,700,env,[(-1,700,4.)])
    work=source_work([r.time_ms for r in rows],[r.actions for r in rows],env,4.)
    assert sum(work) == pytest.approx(report['added_work'])
    remaining=recovery_potential(terminal.values,env.limits(4.)).sum(-1).mean()
    assert report['added_work'] == pytest.approx(report['excess_seconds']+remaining)
    assert window_work_maxima([r.time_ms for r in rows],work,(2000,))[0] == pytest.approx(sum(work))


def test_recovery_does_not_charge_the_future_for_already_committed_overload():
    initial=ActionResponseState.from_rows([tap(i*50) for i in range(20)],1000)
    env=ActionEnvelope((2.,6.),reference().maximum,'zero-addition', (4000.,), ((0.,),(0.,)))
    _,report=action_response(initial,(),2000,env,[(1000,2000,4.)])
    assert report['excess_seconds'] > 0
    assert report['added_work'] == 0 and report['acceptable']


def test_boundary_action_is_assessed_under_the_new_control_request():
    maxima=np.stack((np.full((len(TAUS_MS),len(KINDS)),.1),
                     np.full((len(TAUS_MS),len(KINDS)),1000.)))
    env=ActionEnvelope((2.,6.),tuple(maxima.tolist()),'scopes',(4000.,),((0.,),(0.,)))
    _,report=action_response(ActionResponseState(),[tap(500)],1000,env,[(-1,500,2.),(500,1000,6.)])
    assert all(r['added_work'] == 0 for r in report['ranges'])
    assert report['acceptable']


def test_hypothetical_row_work_matches_exact_state_transitions_including_wait():
    from ensomi_model.research.bounded_typed_continuation.contract import ROW_ACTIONS
    from ensomi_model.research.joint_audio_continuation.state import legal_rows
    state=ActionResponseState.from_rows([CompleteRow(0,(2,0,0,0)),tap(50,1),
        CompleteRow(100,(3,0,0,0)),tap(125),CompleteRow(200,(0,0,2,0)),tap(250,3)],250)
    times=[300,475]
    env=reference()
    values=candidate_work(state,times,ROW_ACTIONS,env.limits(4.))
    assert (values >= 0).all() and np.array_equal(values[:,0],np.zeros(2))
    legal=legal_rows([state.replay],[False])[0]
    for i,t in enumerate(times):
        before=state.advance(t)
        for j,actions in enumerate(ROW_ACTIONS):
            if not legal[j]:continue
            after=state.observe(CompleteRow(t,actions))
            work=(recovery_potential(after.values,env.limits(4.))-
                  recovery_potential(before.values,env.limits(4.))).sum(-1).mean()
            assert values[i,j] == pytest.approx(work,abs=1e-10)
