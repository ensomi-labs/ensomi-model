import json

import numpy as np
import pytest

from ensomi_model.research.bounded_typed_continuation.contract import ROW_ACTIONS
from ensomi_model.research.gameplay_evaluation.release_calibration import release_calibration
from ensomi_model.research.gameplay_evaluation.temporal import ChartTrace, Scope
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow


def log_laws(*laws):
    result = np.full((len(laws), len(ROW_ACTIONS)), -np.inf)
    for i, law in enumerate(laws):
        for row, probability in law.items():
            result[i, ROW_ACTIONS.index(row)] = np.log(probability)
    return result


def observe(rows, clocks, laws, start, end, *, origin='synthetic'):
    return release_calibration(ChartTrace(rows, end), Scope('test', start, end),
                               clocks, laws, origin=origin)


def test_joint_release_calibration_distinguishes_equal_finger_marginals():
    rows = [CompleteRow(0, (2, 2, 0, 0)), CompleteRow(100, (3, 3, 1, 0))]
    laws = [log_laws({(3, 3, 1, 0): .5, (0, 0, 1, 0): .5}),
            log_laws({(3, 0, 1, 0): .5, (0, 3, 1, 0): .5})]
    a, b = [observe(rows, [100], p, 50, 101) for p in laws]
    assert a['summary']['H']['marginal'] == b['summary']['H']['marginal']
    assert a['summary']['H']['pair']['mean_brier'] == .25
    assert b['summary']['H']['pair']['mean_brier'] == 1.
    assert a['summary']['H']['subset_mean_brier'] == .5
    assert b['summary']['H']['subset_mean_brier'] == 1.5
    assert a['unsupported_observed_rows'] == 0
    assert b['unsupported_observed_rows'] == 1
    assert a['pair_risks'][0]['same_origin']
    assert a['pair_risks'][0]['independent_probability'] == .25


def test_true_origins_censoring_and_no_future_tail_access():
    rows = [CompleteRow(0, (2, 0, 0, 0)), CompleteRow(100, (0, 1, 0, 0)),
            CompleteRow(200, (0, 0, 1, 0)), CompleteRow(500, (3, 0, 0, 0))]
    laws = log_laws({(0, 1, 0, 0): .8, (3, 1, 0, 0): .2},
                    {(0, 0, 1, 0): .75, (3, 0, 1, 0): .25})
    scope = Scope('private', 50, 300)
    private = release_calibration(ChartTrace(rows[:3], 300), scope, [100, 200], laws, origin='reference')
    full = release_calibration(ChartTrace(rows, 501), scope, [100, 200], laws, origin='reference')
    assert private == full
    hold = private['holds'][0]
    assert hold['start_ms'] == 0 and hold['incoming'] and hold['censored']
    assert hold['release_ms'] is None
    assert hold['H_continue_opportunities'] == 2
    assert hold['H_continue_survival_product'] == pytest.approx(.6)
    assert hold['H_continue_surprisal_sum'] == pytest.approx(-np.log(.6))
    assert [r['age_ms'] for r in private['marginal_risks']] == [100, 200]
    quiet = observe(rows[:1], [], np.empty((0, 256)), 50, 90)
    assert quiet['holds'][0]['H_continue_survival_product'] is None
    assert quiet['summary']['H']['marginal']['mean_brier'] is None


def test_conditional_release_events_distinguish_time_choice_from_mark_choice():
    rows = [CompleteRow(0, (2, 2, 0, 0)), CompleteRow(100, (3, 0, 0, 0)),
            CompleteRow(200, (0, 3, 0, 0))]
    laws = log_laws({(3, 0, 0, 0): .5, (0, 3, 0, 0): .5}, {(0, 3, 0, 0): 1.})
    report = observe(rows, [100, 200], laws, 50, 201, origin='generated')
    assert report['summary']['R']['required_release_rows'] == 2
    assert report['summary']['R']['observed_releases_without_continue_option'] == 1
    assert all(r['can_continue'] for r in report['marginal_risks'][:2])
    assert not report['marginal_risks'][-1]['can_continue']
    assert report['summary']['H']['marginal']['count'] == 0
    assert report['origin'] == 'generated'
    assert 'Self-calibration' in report['interpretation']


def test_adjacent_scope_statistics_add_without_resetting_LN_context():
    rows = [CompleteRow(0, (2, 2, 0, 0)), CompleteRow(100, (0, 0, 1, 0)),
            CompleteRow(200, (3, 0, 0, 1)), CompleteRow(300, (0, 3, 0, 0))]
    laws = log_laws({(0, 0, 1, 0): .7, (3, 3, 1, 0): .3},
                    {(3, 0, 0, 1): .6, (0, 3, 0, 1): .4}, {(0, 3, 0, 0): 1.})
    whole = observe(rows, [100, 200, 300], laws, 50, 301)
    left = observe(rows[:2], [100], laws[:1], 50, 200)
    right = observe(rows, [200, 300], laws[1:], 200, 301)
    for role in ('H', 'R'):
        for kind in ('marginal', 'pair'):
            for key in ('count', 'predicted_sum', 'observed_sum', 'brier_sum'):
                assert whole['summary'][role][kind][key] == pytest.approx(
                    left['summary'][role][kind][key]+right['summary'][role][kind][key])
    assert right['marginal_risks'][0]['age_ms'] == 200
    assert right['marginal_risks'][0]['previous_H_hz'] == 2


def test_reflection_preserves_aggregates_and_relabels_columns():
    rows = [CompleteRow(0, (2, 0, 2, 0)), CompleteRow(100, (3, 1, 0, 0)),
            CompleteRow(200, (0, 0, 3, 0))]
    laws = log_laws({(3, 1, 0, 0): .7, (0, 1, 3, 0): .3}, {(0, 0, 3, 0): 1.})
    reverse = [ROW_ACTIONS.index(row[::-1]) for row in ROW_ACTIONS]
    a = observe(rows, [100, 200], laws, 50, 201)
    b = observe([CompleteRow(r.time_ms, r.actions[::-1]) for r in rows],
                [100, 200], laws[:, reverse], 50, 201)
    assert a['summary'] == b['summary']
    assert a['pair_risks'][0]['same_hand'] is False
    left = sorted((r['time_ms'], 3-r['column'], r['probability']) for r in a['marginal_risks'])
    right = sorted((r['time_ms'], r['column'], r['probability']) for r in b['marginal_risks'])
    assert left == right


def test_certain_wrong_release_stays_serializable_and_visible():
    rows = [CompleteRow(0, (2, 0, 0, 0)), CompleteRow(100, (0, 1, 0, 0))]
    report = observe(rows, [100], log_laws({(3, 1, 0, 0): 1.}), 50, 101)
    assert report['unsupported_observed_rows'] == 1
    assert report['holds'][0]['H_continue_survival_product'] == 0
    assert report['holds'][0]['H_continue_surprisal_sum'] is None
    assert report['holds'][0]['certain_premature_H_hazards'] == 1
    json.dumps(report, allow_nan=False)


def test_query_alignment_and_physical_role_support_are_checked():
    rows = [CompleteRow(0, (2, 0, 0, 0)), CompleteRow(100, (0, 1, 0, 0))]
    valid = log_laws({(0, 1, 0, 0): 1.})
    with pytest.raises(ValueError, match='align'):
        observe(rows, [99], valid, 50, 101)
    with pytest.raises(ValueError, match='normalized'):
        observe(rows, [100], valid+np.log(2), 50, 101)
    with pytest.raises(ValueError, match='entering LN'):
        observe(rows, [100], log_laws({(1, 0, 0, 0): 1.}), 50, 101)
    with pytest.raises(ValueError, match='event role'):
        observe(rows, [100], log_laws({(3, 0, 0, 0): 1.}), 50, 101)
    with pytest.raises(ValueError, match='origin'):
        observe(rows, [100], valid, 50, 101, origin='unknown')


def test_new_LN_in_an_atomic_handoff_is_not_an_entering_release_risk():
    rows = [CompleteRow(0, (2, 0, 0, 0)), CompleteRow(100, (3, 2, 0, 0))]
    report = observe(rows, [100], log_laws({(3, 2, 0, 0): 1.}), 50, 101)
    assert len(report['marginal_risks']) == 1
    assert report['marginal_risks'][0]['column'] == 0
    assert not report['pair_risks']
    new_hold = next(h for h in report['holds'] if h['column'] == 1)
    assert new_hold['censored'] and new_hold['start_ms'] == 100
    assert new_hold['H_continue_opportunities'] == 0


def test_small_normalization_roundoff_is_reported_before_correction():
    rows = [CompleteRow(0, (2, 0, 0, 0)), CompleteRow(100, (3, 0, 0, 0))]
    laws = log_laws({(3, 0, 0, 0): 1.})+np.log1p(1e-6)
    report = observe(rows, [100], laws, 50, 101)
    assert report['maximum_normalization_error'] == pytest.approx(1e-6)
    assert report['marginal_risks'][0]['probability'] == 1.
