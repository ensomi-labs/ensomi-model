from __future__ import annotations

import pytest

from pulsefield_model.evals.mapper_v3_generated_prefix_event_budget_calibration_audit import (
    decision_from_metrics,
    primary_trace_metrics,
    run_generated_prefix_event_budget_calibration_audit,
)


def test_primary_trace_metrics_count_rank_near_deficits() -> None:
    summary = {
        "aggregate": {
            "failure_class_counts": {"time_shift_repetition": 2},
            "secondary_flag_counts": {"event_underselection": 2, "boundary_drift": 2},
        },
        "case_results": [
            _trace_case("a", rank=2, candidates=10, generated_second=0.05, reference_second=0.55),
            _trace_case("b", rank=6, candidates=12, generated_second=0.30, reference_second=0.40),
        ],
    }

    metrics = primary_trace_metrics(summary)

    assert metrics["case_count"] == 2
    assert metrics["rank_near_case_count"] == 1
    assert metrics["rank_near_candidate_count_total"] == 10
    assert metrics["second_window_deficit_case_count"] == 1
    assert metrics["mean_second_window_share_deficit"] == pytest.approx(0.30)


def test_decision_selects_selective_completion_budget_signal() -> None:
    decision = decision_from_metrics(
        {
            "required_artifacts_present": True,
            "primary_trace": {
                "rank_near_case_count": 6,
                "second_window_deficit_case_count": 6,
            },
            "spacing_escape": {
                "primary_recoverable_by_broad_release": True,
                "sentinel_overproduction_risk": True,
            },
            "failed_branch_checks": {
                "scalar_or_decode_repairs_failed": True,
            },
        }
    )

    assert decision["route"] == "TEST_SELECTIVE_COMPLETION_BUDGET_SIGNAL"


def test_decision_mutates_to_grammar_when_deficits_are_not_rank_near() -> None:
    decision = decision_from_metrics(
        {
            "required_artifacts_present": True,
            "primary_trace": {
                "rank_near_case_count": 2,
                "second_window_deficit_case_count": 5,
            },
            "spacing_escape": {
                "primary_recoverable_by_broad_release": True,
                "sentinel_overproduction_risk": True,
            },
            "failed_branch_checks": {
                "scalar_or_decode_repairs_failed": True,
            },
        }
    )

    assert decision["route"] == "MUTATE_TO_GRAMMAR_LEVEL_CONTINUATION_TARGET"


def test_run_audit_on_committed_artifacts(tmp_path) -> None:
    summary_path = tmp_path / "summary.json"
    report_path = tmp_path / "report.md"

    summary = run_generated_prefix_event_budget_calibration_audit(
        summary_output_path=summary_path,
        report_output_path=report_path,
    )

    assert summary["decision"]["route"] == "TEST_SELECTIVE_COMPLETION_BUDGET_SIGNAL"
    assert summary["metrics"]["guard_results"]["artifact_only"] is True
    assert summary["metrics"]["guard_results"]["no_training"] is True
    assert summary["metrics"]["primary_trace"]["rank_near_case_count"] == 6
    assert summary["metrics"]["primary_trace"]["second_window_deficit_case_count"] == 6
    assert summary["metrics"]["spacing_escape"]["sentinel_overproduction_risk"] is True
    assert summary_path.exists()
    assert report_path.exists()


def _trace_case(
    case_id: str,
    *,
    rank: int,
    candidates: int,
    generated_second: float,
    reference_second: float,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "baseline_projection": {
            "event_count_ratio": 0.5,
            "second_window_event_share": generated_second,
        },
        "classification": {
            "failure_class": "time_shift_repetition",
            "candidate_failures": {
                "event_underselection": {
                    "concrete": True,
                    "best_event_rank": rank,
                    "candidate_count": candidates,
                    "best_event_margin_vs_argmax": -0.5,
                },
                "boundary_drift": {
                    "concrete": True,
                    "second_window_event_share": generated_second,
                    "reference_second_window_event_share": reference_second,
                },
            },
        },
    }
