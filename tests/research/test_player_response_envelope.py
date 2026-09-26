import numpy as np
import pytest

from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
from ensomi_model.research.player_response.envelope import (
    AttackEnvelope, fit_attack_envelope, sustained_response,
)
from ensomi_model.research.player_response.state import CommittedPlayState


def tap(t, lane=0):
    return CompleteRow(t, tuple(int(c == lane) for c in range(4)))


def envelope():
    return AttackEnvelope((1000.,), (2., 4.), ((1.,), (2.,)), 'fixture')


def test_event_expiry_integral_matches_manual_area_and_horizon_partition():
    state = CommittedPlayState().advance(0)
    future = [tap(100), tap(200)]
    _, full = sustained_response(state, future, 1500, envelope(), [(0, 1500, 2)])
    prefix, a = sustained_response(state, future, 750, envelope(), [(0, 750, 2)])
    _, b = sustained_response(prefix.state_at_end, (), 1500, envelope(), [(750, 1500, 2)])
    # Two attacks occupy the one-second window from 200 through 1100 ms.
    assert full['excess_seconds'] == pytest.approx(.9)
    assert a['excess_seconds']+b['excess_seconds'] == pytest.approx(.9)


def test_control_changes_selection_reference_without_resetting_physical_history():
    state = CommittedPlayState().advance(0)
    future = [tap(100), tap(200)]
    plain, all_low = sustained_response(state, future, 1500, envelope(), [(0, 1500, 2)])
    changed, split = sustained_response(state, future, 1500, envelope(), [(0, 500, 2), (500, 1500, 4)])
    assert plain.state_at_end == changed.state_at_end
    np.testing.assert_array_equal(plain.peak['attack_hz'], changed.peak['attack_hz'])
    assert all_low['excess_seconds'] == pytest.approx(.9)
    assert split['excess_seconds'] == pytest.approx(.3)
    assert split['ranges'][1]['peak_ratio'] == 1.


def test_loaded_history_changes_same_future_and_release_is_not_an_attack():
    loaded = CommittedPlayState.from_rows([tap(100), tap(200)], 300)
    spread = CommittedPlayState.from_rows([tap(100, 1), tap(200, 2)], 300)
    responses = [sustained_response(s, [tap(400)], 1200, envelope(), [(300, 1200, 2)])[1]
                 for s in (loaded, spread)]
    assert responses[0]['excess_seconds'] > responses[1]['excess_seconds'] == 0
    held = CommittedPlayState().observe(CompleteRow(0, (2, 0, 0, 0)))
    observations, report = sustained_response(held, [CompleteRow(500, (3, 0, 0, 0))],
                                              1200, envelope(), [(0, 1200, 2)])
    assert report['excess_seconds'] == 0
    assert observations.held_ms[0] == 500
    assert observations.release_exposure[0, 0] > 0


def test_fitting_equalizes_song_groups_and_reports_monotonic_adjustment():
    stars = [2.]*11+[4., 4.]
    groups = ['a']*9+['b', 'c', 'd', 'e']
    values = [[10.]]*9+[[2.], [3.], [1.], [2.]]
    fitted, report = fit_attack_envelope(stars, groups, values, windows_ms=(1000,),
        knots=(2., 4.), quantile=.5, reference='train-fixture')
    assert report['raw_hz'] == [[3.], [1.]]
    assert fitted.maximum_hz == ((3.,), (3.,))
    assert report['bands'][0]['groups'] == 3
