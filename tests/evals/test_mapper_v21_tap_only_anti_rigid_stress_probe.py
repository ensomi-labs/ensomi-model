from __future__ import annotations

from pulsefield_model.evals.mapper_v21_tap_only_anti_rigid_stress_probe import (
    aggregate_results,
    case_matches_variant,
    decision_from_aggregate,
)


def test_case_matches_variant_uses_legal_and_metric_signature() -> None:
    row = _case(case_id="case", baseline_rigid=1.0, tap_rigid=1.0, hard_rigid=0.7, tap_legal=True)

    assert case_matches_variant(row, "baseline") is True
    assert case_matches_variant(row, "hard_block") is False


def test_decision_passes_when_tap_only_fixes_case05_and_keeps_rigidity_signal() -> None:
    rows = [
        _case(case_id="05_usao_knight_rider_kuo_kyoka_dnm_s_normal", baseline_rigid=0.67, tap_rigid=0.67, hard_rigid=0.55),
        _case(case_id="04", baseline_rigid=0.91, tap_rigid=0.70, hard_rigid=0.58),
        _case(case_id="29", baseline_rigid=1.0, tap_rigid=0.80, hard_rigid=0.73),
        _case(case_id="31", baseline_rigid=1.0, tap_rigid=1.0, hard_rigid=0.67),
    ]
    aggregate = aggregate_results(rows)

    decision = decision_from_aggregate(aggregate, rows)

    assert aggregate["tap_only_vs_baseline"]["rigid_improved_count"] == 2
    assert decision["route"] == "TEST_NEXT"


def test_decision_kills_when_case05_still_starves() -> None:
    rows = [
        _case(
            case_id="05_usao_knight_rider_kuo_kyoka_dnm_s_normal",
            baseline_rigid=0.67,
            tap_rigid=0.55,
            hard_rigid=0.55,
            tap_starved=True,
        )
    ]
    aggregate = aggregate_results(rows)

    decision = decision_from_aggregate(aggregate, rows)

    assert decision["route"] == "KILL"


def _case(
    *,
    case_id: str,
    baseline_rigid: float,
    tap_rigid: float,
    hard_rigid: float,
    tap_legal: bool = True,
    tap_starved: bool = False,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "baseline_legal": True,
        "hard_block_legal": True,
        "tap_only_legal": tap_legal,
        "v3_legal": True,
        "baseline_metrics": _metrics(rigid=baseline_rigid, starved=False),
        "hard_block_metrics": _metrics(rigid=hard_rigid, starved=False),
        "tap_only_metrics": _metrics(rigid=tap_rigid, starved=tap_starved),
        "v3_metrics": _metrics(rigid=0.75, starved=False),
        "tap_only_vs_baseline": {
            "dominant_spacing_improved": tap_rigid < baseline_rigid,
            "new_starved": tap_starved,
        },
        "tap_only_vs_hard_block": {
            "dominant_spacing_improved": tap_rigid < hard_rigid,
            "new_starved": tap_starved,
        },
        "transform": {"candidate_count": 1, "blocked_count": 1},
    }


def _metrics(*, rigid: float, starved: bool) -> dict[str, object]:
    return {
        "generated_event_count": 10,
        "event_count_ratio": 1.0,
        "dominant_spacing_ratio": rigid,
        "starved": starved,
        "second_window_event_share": 0.0 if starved else 0.5,
        "timing_match_100ms": {"f1": 0.75},
        "first_12_generated_times": [0, 160],
        "last_12_generated_times": [800, 960],
    }
