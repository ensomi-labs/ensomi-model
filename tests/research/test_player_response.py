import numpy as np
import pytest

from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
from ensomi_model.research.player_response.state import CommittedPlayState, observe_continuation
from ensomi_model.research.scoped_style_modeling.dataset import ContractError


def tap(time, lane):
    return CompleteRow(time, tuple(int(c == lane) for c in range(4)))


def test_same_exact_replay_has_different_sustained_load_for_same_future():
    shared = [tap(3100+100*c, c) for c in range(4)]
    jack = [tap(i*125, 0) for i in range(24)]+shared
    flow = [tap(i*125, i % 4) for i in range(24)]+shared
    a, b = [CommittedPlayState.from_rows(rows, 3400) for rows in (jack, flow)]
    assert a.replay == b.replay
    future = [tap(3500, 0), tap(3600, 0)]
    pa, pb = [observe_continuation(s, future, 4000, windows_ms=(4000,)) for s in (a, b)]
    assert pa.peak['attack_hz'][0, 0] == 27/4
    assert pb.peak['attack_hz'][0, 0] == 9/4
    assert a.time_ms == b.time_ms == 3400


def test_empty_future_preserves_holding_after_attack_rate_recovers():
    held = CommittedPlayState().observe(CompleteRow(0, (2, 0, 0, 0)))
    free = CommittedPlayState().observe(tap(0, 0))
    ph, pf = [observe_continuation(s, (), 3000, windows_ms=(1000,)) for s in (held, free)]
    np.testing.assert_array_equal(ph.terminal['attack_hz'], pf.terminal['attack_hz'])
    assert ph.held_ms[0] == 3000 and pf.held_ms[0] == 0
    assert ph.terminal['held_fraction'][0, 0] == 1
    assert ph.state_at_end.replay.open_ln_start_ms[0] == 0


def test_horizon_after_last_row_changes_exposure_and_terminal_response():
    state = CommittedPlayState().advance(0)
    short = observe_continuation(state, [tap(100, 0)], 500, windows_ms=(500,))
    long = observe_continuation(state, [tap(100, 0)], 1000, windows_ms=(500,))
    assert short.terminal['attack_hz'][0, 0] == 2
    assert long.terminal['attack_hz'][0, 0] == 0
    assert short.attack_exposure[0, 0] == pytest.approx(.8)
    assert long.attack_exposure[0, 0] == pytest.approx(1.)


def test_time_advance_composes_and_old_ln_origin_survives_retention():
    state = CommittedPlayState().observe(CompleteRow(0, (2, 0, 0, 0)))
    direct = state.advance(41000)
    split = state.advance(10000).advance(30000).advance(41000)
    assert direct == split
    assert direct.rows == () and direct.replay.open_ln_start_ms[0] == 0
    closed = direct.observe(CompleteRow(42000, (3, 0, 0, 0)))
    assert closed.window_facts((16000,))['held_fraction'][0, 0] == 1
    assert closed.replay.open_ln_start_ms[0] is None
    with pytest.raises(ContractError, match='row/no-row'):
        direct.observe(tap(40000, 1))


def test_recent_coordination_order_is_not_collapsed_into_lane_counts():
    a = [tap(i*100, lane) for i, lane in enumerate([0, 0, 1, 1])]
    b = [tap(i*100, lane) for i, lane in enumerate([0, 1, 0, 1])]
    sa, sb = [CommittedPlayState.from_rows(rows, 300) for rows in (a, b)]
    np.testing.assert_array_equal(sa.window_facts((1000,))['attack_hz'], sb.window_facts((1000,))['attack_hz'])
    assert sa.rows != sb.rows
    pa, pb = [observe_continuation(s, [tap(400, 0)], 800) for s in (sa, sb)]
    assert pa.events[0].previous_head_groups != pb.events[0].previous_head_groups


def test_response_mirrors_and_uses_real_proposed_ln_endpoints():
    prefix = [CompleteRow(0, (2, 0, 0, 0)), tap(200, 1)]
    future = [CompleteRow(500, (3, 0, 1, 0)), tap(900, 3)]
    mirror = lambda rows: [CompleteRow(r.time_ms, r.actions[::-1]) for r in rows]
    state = CommittedPlayState.from_rows(prefix, 200)
    flipped = CommittedPlayState.from_rows(mirror(prefix), 200)
    a, b = [observe_continuation(s, y, 1500) for s, y in ((state, future), (flipped, mirror(future)))]
    for name in a.peak:
        np.testing.assert_array_equal(a.peak[name], b.peak[name][:, ::-1])
        np.testing.assert_array_equal(a.terminal[name], b.terminal[name][:, ::-1])
    np.testing.assert_array_equal(a.held_ms, b.held_ms[::-1])
    assert a.held_ms[0] == 300
    assert a.events[0].entering_holds == (True, False, False, False)
    assert not any(a.events[0].resulting_holds)
