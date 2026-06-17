from __future__ import annotations

from pulsefield_model.evals.mapper_v2_1_anti_rigid_spacing_guard import (
    aggregate_results,
    decision_from_aggregate,
    generated_metrics,
    timing_match,
)


def test_timing_match_counts_one_to_one_matches_with_tolerance() -> None:
    timing = timing_match([100, 210, 400], [120, 300, 405], tolerance_ms=30)

    assert timing["matched_generated"] == 2
    assert timing["matched_reference"] == 2
    assert round(timing["precision"], 6) == round(2 / 3, 6)
    assert round(timing["recall"], 6) == round(2 / 3, 6)


def test_generated_metrics_marks_supported_second_window_starvation() -> None:
    metrics = generated_metrics(
        generated_times=[160, 320, 480, 640],
        reference_times=[100, 200, 8_100, 8_300],
        chart_end_ms=16_000,
    )

    assert metrics["starved"] is True
    assert metrics["second_window_event_share"] == 0.0
    assert metrics["reference_second_window_event_share"] == 0.5
    assert metrics["dominant_spacing_ms"] == 160
    assert metrics["dominant_spacing_ratio"] == 1.0


def test_decision_passes_when_rigidity_improves_without_guards_regressing() -> None:
    aggregate = aggregate_results(
        [
            _case(rigid_delta=-0.2, f1_delta=-0.01, blocked=3),
            _case(rigid_delta=-0.1, f1_delta=0.02, blocked=2),
            _case(rigid_delta=0.0, f1_delta=0.0, blocked=1),
        ]
    )

    decision = decision_from_aggregate(aggregate)

    assert aggregate["rigid_improved_count"] == 2
    assert aggregate["blocked_count"] == 6
    assert decision["route"] == "TEST_NEXT"


def test_decision_mutates_when_guard_never_activates() -> None:
    aggregate = aggregate_results([_case(rigid_delta=0.0, f1_delta=0.0, blocked=0)])

    decision = decision_from_aggregate(aggregate)

    assert decision["route"] == "MUTATE"
    assert "never activated" in decision["reason"]


def _case(*, rigid_delta: float, f1_delta: float, blocked: int) -> dict[str, object]:
    baseline_rigid = 1.0
    baseline_f1 = 0.75
    candidate_rigid = baseline_rigid + float(rigid_delta)
    candidate_f1 = baseline_f1 + float(f1_delta)
    return {
        "legal": True,
        "candidate_metrics": {
            "dominant_spacing_ratio": candidate_rigid,
            "starved": False,
            "timing_match_100ms": {"f1": candidate_f1},
        },
        "baseline_metrics": {
            "dominant_spacing_ratio": baseline_rigid,
            "starved": False,
            "timing_match_100ms": {"f1": baseline_f1},
        },
        "comparison": {
            "dominant_spacing_improved": candidate_rigid < baseline_rigid,
        },
        "transform": {"blocked_count": blocked},
    }
