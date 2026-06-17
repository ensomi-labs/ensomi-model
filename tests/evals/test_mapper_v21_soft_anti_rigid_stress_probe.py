from __future__ import annotations

import pytest

from pulsefield_model.evals.mapper_v21_soft_anti_rigid_stress_probe import (
    aggregate_results,
    decision_from_aggregate,
    select_case_results,
)


def test_select_case_results_preserves_requested_order() -> None:
    selected = select_case_results(
        {"case_results": [{"case_id": "b"}, {"case_id": "a"}]},
        case_ids=("a", "b"),
    )

    assert [row["case_id"] for row in selected] == ["a", "b"]


def test_select_case_results_raises_for_missing_case() -> None:
    with pytest.raises(ValueError, match="missing requested case ids"):
        select_case_results({"case_results": [{"case_id": "a"}]}, case_ids=("missing",))


def test_decision_passes_when_soft_is_legal_not_starved_and_keeps_f1() -> None:
    aggregate = aggregate_results(
        [
            _case(baseline_rigid=1.0, hard_rigid=0.7, soft_rigid=0.8, baseline_f1=0.7, hard_f1=0.68, soft_f1=0.68),
            _case(baseline_rigid=1.0, hard_rigid=0.7, soft_rigid=0.8, baseline_f1=0.7, hard_f1=0.68, soft_f1=0.68),
            _case(baseline_rigid=1.0, hard_rigid=0.7, soft_rigid=0.8, baseline_f1=0.7, hard_f1=0.68, soft_f1=0.68),
            _case(baseline_rigid=0.6, hard_rigid=0.5, soft_rigid=0.6, baseline_f1=0.7, hard_f1=0.68, soft_f1=0.68),
        ]
    )

    decision = decision_from_aggregate(aggregate)

    assert aggregate["soft_vs_baseline"]["rigid_improved_count"] == 3
    assert decision["route"] == "TEST_NEXT"


def test_decision_kills_when_soft_adds_starvation() -> None:
    aggregate = aggregate_results(
        [
            _case(
                baseline_rigid=1.0,
                hard_rigid=0.7,
                soft_rigid=0.8,
                baseline_f1=0.7,
                hard_f1=0.68,
                soft_f1=0.68,
                soft_starved=True,
            )
        ]
    )

    decision = decision_from_aggregate(aggregate)

    assert aggregate["soft_vs_baseline"]["new_starved_count"] == 1
    assert decision["route"] == "KILL"


def test_decision_kills_when_soft_is_illegal() -> None:
    aggregate = aggregate_results(
        [
            _case(
                baseline_rigid=1.0,
                hard_rigid=0.7,
                soft_rigid=0.8,
                baseline_f1=0.7,
                hard_f1=0.68,
                soft_f1=0.68,
                soft_legal=False,
            )
        ]
    )

    decision = decision_from_aggregate(aggregate)

    assert aggregate["soft"]["all_legal"] is False
    assert decision["route"] == "KILL"


def _case(
    *,
    baseline_rigid: float,
    hard_rigid: float,
    soft_rigid: float,
    baseline_f1: float,
    hard_f1: float,
    soft_f1: float,
    soft_starved: bool = False,
    soft_legal: bool = True,
) -> dict[str, object]:
    return {
        "baseline_legal": True,
        "hard_block_legal": True,
        "soft_legal": soft_legal,
        "v3_legal": True,
        "baseline_metrics": _metrics(rigid=baseline_rigid, f1=baseline_f1, starved=False),
        "hard_block_metrics": _metrics(rigid=hard_rigid, f1=hard_f1, starved=False),
        "soft_metrics": _metrics(rigid=soft_rigid, f1=soft_f1, starved=soft_starved),
        "v3_metrics": _metrics(rigid=0.75, f1=0.65, starved=False),
        "soft_vs_baseline": {
            "dominant_spacing_improved": soft_rigid < baseline_rigid,
            "f1_delta": soft_f1 - baseline_f1,
            "new_starved": soft_starved,
        },
        "soft_vs_hard_block": {
            "dominant_spacing_improved": soft_rigid < hard_rigid,
            "f1_delta": soft_f1 - hard_f1,
            "new_starved": soft_starved,
        },
        "soft_vs_v3": {
            "dominant_spacing_improved": soft_rigid < 0.75,
            "f1_delta": soft_f1 - 0.65,
            "new_starved": soft_starved,
        },
        "transform": {"blocked_count": 4},
    }


def _metrics(*, rigid: float, f1: float, starved: bool) -> dict[str, object]:
    return {
        "dominant_spacing_ratio": rigid,
        "event_count_ratio": 1.0,
        "starved": starved,
        "timing_match_100ms": {"f1": f1},
    }
