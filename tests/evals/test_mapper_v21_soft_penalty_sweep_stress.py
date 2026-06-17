from __future__ import annotations

import pytest

from pulsefield_model.evals.mapper_v21_soft_penalty_sweep_stress import (
    case_matches_baseline,
    case_matches_hard_block,
    decision_from_penalty_results,
    select_best_penalty,
    summarize_penalty_result,
    validate_penalties,
)


def test_validate_penalties_requires_unique_finite_non_negative_values() -> None:
    assert validate_penalties((0.5, 1.0)) == (0.5, 1.0)

    with pytest.raises(ValueError, match="unique"):
        validate_penalties((1.0, 1.0))
    with pytest.raises(ValueError, match="non-negative"):
        validate_penalties((-1.0,))


def test_case_identity_detects_hard_block_and_baseline_matches() -> None:
    row = _case(soft_rigid=0.7, hard_rigid=0.7, baseline_rigid=1.0, soft_legal=True, hard_legal=True)

    assert case_matches_hard_block(row) is True
    assert case_matches_baseline(row) is False


def test_summarize_penalty_result_marks_primary_pass_when_safe_and_distinct() -> None:
    summary = {
        "config": {"penalty": 1.0},
        "output_dir": "tmp/penalty_1_0",
        "decision": {"route": "TEST_NEXT"},
        "aggregate": {
            "case_count": 4,
            "soft": {"all_legal": True, "legal_count": 4, "starved_count": 0},
            "soft_vs_baseline": {
                "new_starved_count": 0,
                "rigid_improved_count": 2,
                "mean_f1_delta": -0.01,
                "mean_dominant_spacing_ratio_delta": -0.1,
            },
        },
        "case_results": [
            _case(soft_rigid=0.8, hard_rigid=0.7, baseline_rigid=1.0),
            _case(soft_rigid=0.8, hard_rigid=0.7, baseline_rigid=1.0),
            _case(soft_rigid=0.6, hard_rigid=0.6, baseline_rigid=0.6),
            _case(soft_rigid=0.6, hard_rigid=0.6, baseline_rigid=0.6),
        ],
    }

    result = summarize_penalty_result(summary)

    assert result["primary_pass"] is True
    assert result["differs_from_hard_count"] == 2
    assert result["differs_from_baseline_count"] == 2


def test_decision_selects_best_passing_penalty() -> None:
    decision = decision_from_penalty_results(
        [
            _penalty_result(penalty=0.5, passing=True, rigid=2, f1=-0.02),
            _penalty_result(penalty=1.0, passing=True, rigid=3, f1=-0.03),
            _penalty_result(penalty=2.0, passing=False, rigid=4, f1=0.0),
        ]
    )

    assert decision["route"] == "TEST_NEXT"
    assert decision["selected_penalty"] == 1.0
    assert decision["passing_penalty_count"] == 2


def test_decision_kills_when_no_penalty_passes() -> None:
    decision = decision_from_penalty_results([_penalty_result(penalty=0.5, passing=False, rigid=1, f1=0.0)])

    assert decision["route"] == "KILL"
    assert decision["selected_penalty"] is None


def test_select_best_penalty_prefers_higher_rigidity_then_f1_then_lower_penalty() -> None:
    selected = select_best_penalty(
        [
            _penalty_result(penalty=0.5, passing=True, rigid=2, f1=-0.01),
            _penalty_result(penalty=1.0, passing=True, rigid=2, f1=-0.01),
        ]
    )

    assert selected["penalty"] == 0.5


def _penalty_result(*, penalty: float, passing: bool, rigid: int, f1: float) -> dict[str, object]:
    return {
        "penalty": penalty,
        "primary_pass": passing,
        "rigid_improved_count": rigid,
        "mean_f1_delta_vs_baseline": f1,
        "differs_from_hard_count": 1,
    }


def _case(
    *,
    soft_rigid: float,
    hard_rigid: float,
    baseline_rigid: float,
    soft_legal: bool = True,
    hard_legal: bool = True,
) -> dict[str, object]:
    return {
        "soft_legal": soft_legal,
        "hard_block_legal": hard_legal,
        "baseline_legal": True,
        "soft_metrics": _metrics(rigid=soft_rigid),
        "hard_block_metrics": _metrics(rigid=hard_rigid),
        "baseline_metrics": _metrics(rigid=baseline_rigid),
    }


def _metrics(*, rigid: float) -> dict[str, object]:
    return {
        "generated_event_count": 10,
        "event_count_ratio": 1.0,
        "dominant_spacing_ratio": rigid,
        "starved": False,
        "second_window_event_share": 0.5,
        "timing_match_100ms": {"f1": 0.75},
        "first_12_generated_times": [0, 160],
        "last_12_generated_times": [800, 960],
    }
