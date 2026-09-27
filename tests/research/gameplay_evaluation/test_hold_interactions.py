import pytest

from ensomi_model.research.gameplay_evaluation.hold_interactions import hold_interactions
from ensomi_model.research.gameplay_evaluation.hold_relations import hold_relations
from ensomi_model.research.gameplay_evaluation.report import evaluate_scopes
from ensomi_model.research.gameplay_evaluation.temporal import ChartTrace, Scope
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow


def swapped_tails(swap):
    heads = {0: ((0, 2), (3, 2)), 200: ((1, 1),), 400: ((2, 1),),
             600: ((1, 2),), 800: ((0, 2),), 1000: ((2, 2),),
             1200: ((0, 1),), 1400: ((2, 1),), 1600: ((1, 1),)}
    rows = {t: [0] * 4 for t in (*heads, 900, 1100, 1700)}
    for t, values in heads.items():
        for lane, action in values:
            rows[t][lane] = action
    for t, lane in ((200 if swap else 600, 0), (1200 if swap else 800, 1),
                    (900, 0), (1100, 2), (1700, 3)):
        assert not rows[t][lane]
        rows[t][lane] = 3
    return [CompleteRow(t, tuple(actions)) for t, actions in sorted(rows.items())]


def test_equal_amount_duration_span_and_total_occupation_can_hide_different_LN_roles():
    scope = Scope('same_marginals', 0, 1701)
    traces = [ChartTrace(swapped_tails(s), 1701) for s in (False, True)]
    facts = [t.scope_report(scope) for t in traces]
    for name in ('H', 'heads', 'LN_heads', 'LN_head_fraction', 'any_held_fraction', 'mean_held_columns'):
        assert facts[0][name] == pytest.approx(facts[1][name])
    timing = [hold_relations(t, scope) for t in traces]
    assert timing[0]['duration_ms'] == timing[1]['duration_ms']
    assert timing[0]['H_span'] == timing[1]['H_span']
    a, b = [hold_interactions(t, scope) for t in traces]
    assert sum(a['pair_counts']['hold_tap'].values()) == 7
    assert sum(b['pair_counts']['hold_tap'].values()) == 5
    assert sum(a['pair_counts']['hold_LN_start'].values()) == 3
    assert sum(b['pair_counts']['hold_LN_start'].values()) == 5
    assert a['tap_fraction_with_continuing_hold'] == b['tap_fraction_with_continuing_hold'] == 1


def test_scope_counts_are_additive_and_keep_incoming_hold_origins():
    trace = ChartTrace(swapped_tails(True), 1701)
    whole = hold_interactions(trace, Scope('whole', 0, 1701))
    left = hold_interactions(trace, Scope('left', 0, 800))
    right = hold_interactions(trace, Scope('right', 800, 1701))
    for k, n in whole['counts'].items():
        assert n == left['counts'][k] + right['counts'][k]
    for kind, relations in whole['pair_counts'].items():
        for relation, n in relations.items():
            assert n == left['pair_counts'][kind][relation] + right['pair_counts'][kind][relation]
    assert right['incoming_holds'] == 2
    assert any(w['started_before_scope'] and w['start_ms'] == 0 for w in right['largest_hold_contexts'])


def test_private_horizon_never_reads_later_tails_or_closes_holds():
    rows = [CompleteRow(0, (2, 0, 0, 0)), CompleteRow(100, (0, 1, 0, 0)),
            CompleteRow(200, (0, 0, 1, 0)), CompleteRow(400, (3, 0, 0, 0))]
    scope = Scope('private', 50, 300)
    private = hold_interactions(ChartTrace(rows[:3], 300), scope)
    complete = hold_interactions(ChartTrace(rows, 401), scope)
    assert private == complete
    assert private['incoming_holds'] == private['holds_without_observed_release'] == 1
    assert private['largest_hold_contexts'][0]['release_ms_in_scope'] is None
    quiet = hold_interactions(ChartTrace(rows[:1], 300), Scope('quiet', 50, 300))
    assert quiet['involved_holds'] == 1 and quiet['tap_fraction_with_continuing_hold'] is None
    assert not quiet['largest_hold_contexts']


def test_staggered_co_start_releases_and_atomic_handoffs_keep_their_roles():
    rows = [CompleteRow(0, (2, 2, 0, 0)), CompleteRow(100, (3, 0, 1, 0)),
            CompleteRow(200, (0, 3, 0, 0))]
    scope = Scope('stagger', 0, 201)
    report = evaluate_scopes(ChartTrace(rows, 201), [scope])['scopes'][0]['LN_interactions']
    assert report['pair_counts']['co_start_hold_release'] == dict(same_hand=1, opposite_hand=0)
    assert report['pair_counts']['hold_tap'] == dict(same_hand=0, opposite_hand=1)
    assert report['pair_counts']['release_tap'] == dict(same_hand=0, opposite_hand=1)
    mirror = [CompleteRow(r.time_ms, r.actions[::-1]) for r in rows]
    reflected = hold_interactions(ChartTrace(mirror, 201), scope)
    for name in ('counts', 'pair_counts', 'pair_rates_per_second', 'observed_companion_head_rows_per_hold'):
        assert report[name] == reflected[name]
    together = [rows[0], CompleteRow(100, (3, 3, 1, 0))]
    same_tail = hold_interactions(ChartTrace(together, 201), scope)
    assert sum(same_tail['pair_counts']['co_start_hold_release'].values()) == 0
    assert sum(same_tail['pair_counts']['hold_tap'].values()) == 0
