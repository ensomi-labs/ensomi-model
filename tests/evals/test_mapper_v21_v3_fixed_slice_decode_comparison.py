from __future__ import annotations

from pulsefield_model.evals.mapper_v21_v3_fixed_slice_decode_comparison import (
    aggregate_results,
    decision_from_aggregate,
    illegal_cases,
    metrics_from_v3_row,
    rollout_status,
)


def test_metrics_from_v3_row_recomputes_starvation_from_reference_support() -> None:
    metrics = metrics_from_v3_row(
        {
            "generated_event_count": 4,
            "reference_event_count": 4,
            "event_count_ratio": 1.0,
            "second_window_event_count": 0,
            "second_window_event_share": 0.0,
            "boundary_event_count": 0,
            "boundary_event_ratio": 0.0,
            "duplicate_or_nonincreasing_spacing_ratio": 0.0,
            "dominant_spacing_ms": 160,
            "dominant_spacing_ratio": 1.0,
            "timing_match_100ms": {"f1": 0.5},
        },
        reference_times=[100, 200, 8_100, 8_300],
        chart_end_ms=16_000,
    )

    assert metrics["reference_second_window_event_share"] == 0.5
    assert metrics["starved"] is True


def test_decision_passes_when_guard_generalizes_and_beats_v3_starvation() -> None:
    aggregate = aggregate_results(
        [
            _case(
                baseline_rigid=1.0,
                guard_rigid=0.78,
                v3_rigid=0.75,
                baseline_f1=0.75,
                guard_f1=0.74,
                v3_f1=0.70,
                v3_starved=index < 12,
                blocks=10,
            )
            for index in range(32)
        ]
    )

    decision = decision_from_aggregate(aggregate)

    assert aggregate["guard_vs_baseline"]["rigid_improved_count"] == 32
    assert aggregate["v3"]["starved_count"] == 12
    assert aggregate["v21_guard"]["starved_count"] == 0
    assert decision["route"] == "TEST_NEXT"


def test_decision_kills_when_guard_does_not_generalize() -> None:
    aggregate = aggregate_results(
        [
            _case(
                baseline_rigid=1.0,
                guard_rigid=0.9 if index < 4 else 1.0,
                v3_rigid=0.75,
                baseline_f1=0.75,
                guard_f1=0.74,
                v3_f1=0.70,
                v3_starved=index < 12,
                blocks=2,
            )
            for index in range(32)
        ]
    )

    decision = decision_from_aggregate(aggregate)

    assert aggregate["guard_vs_baseline"]["rigid_improved_count"] == 4
    assert decision["route"] == "KILL"


def test_illegal_cases_reports_failed_variant_rollout_status() -> None:
    row = _case(
        baseline_rigid=1.0,
        guard_rigid=0.8,
        v3_rigid=0.75,
        baseline_f1=0.75,
        guard_f1=0.74,
        v3_f1=0.70,
        v3_starved=False,
        blocks=4,
    )
    row["case_id"] = "case_a"
    row["difficulty"] = 3.0
    row["v21_guard_legal"] = False
    row["v21_guard_rollout"] = {
        "completed": False,
        "dead_end": True,
        "max_tokens_exceeded": False,
        "terminal_ms": 7_990,
        "token_count": 98,
        "timepoint_count": 32,
    }

    failures = illegal_cases([row])

    assert failures == [
        {
            "case_id": "case_a",
            "variant": "v2.1 guard",
            "difficulty": 3.0,
            "difficulty_band": None,
            "completed": False,
            "dead_end": True,
            "max_tokens_exceeded": False,
            "terminal_ms": 7_990,
            "token_count": 98,
            "timepoint_count": 32,
            "summary_path": None,
        }
    ]


def test_rollout_status_uses_last_window_terminal_ms() -> None:
    status = rollout_status(
        {
            "rollout": {
                "completed": False,
                "dead_end": True,
                "max_tokens_exceeded": False,
                "window_count": 1,
                "token_count": 98,
                "timepoint_count": 32,
                "windows": [{"terminal_ms": 7_990}],
            }
        }
    )

    assert status["terminal_ms"] == 7_990
    assert status["dead_end"] is True


def _case(
    *,
    baseline_rigid: float,
    guard_rigid: float,
    v3_rigid: float,
    baseline_f1: float,
    guard_f1: float,
    v3_f1: float,
    v3_starved: bool,
    blocks: int,
) -> dict[str, object]:
    return {
        "v21_baseline_legal": True,
        "v21_guard_legal": True,
        "v3_legal": True,
        "v21_baseline_metrics": _metrics(rigid=baseline_rigid, f1=baseline_f1, starved=False),
        "v21_guard_metrics": _metrics(rigid=guard_rigid, f1=guard_f1, starved=False),
        "v3_metrics": _metrics(rigid=v3_rigid, f1=v3_f1, starved=v3_starved),
        "guard_vs_baseline": {
            "dominant_spacing_improved": guard_rigid < baseline_rigid,
            "f1_delta": guard_f1 - baseline_f1,
            "new_starved": False,
        },
        "guard_vs_v3": {
            "dominant_spacing_improved": guard_rigid < v3_rigid,
            "f1_delta": guard_f1 - v3_f1,
        },
        "transform": {"blocked_count": blocks},
    }


def _metrics(*, rigid: float, f1: float, starved: bool) -> dict[str, object]:
    return {
        "dominant_spacing_ratio": rigid,
        "event_count_ratio": 1.0,
        "second_window_event_share": 0.5 if not starved else 0.0,
        "boundary_event_ratio": 0.0,
        "duplicate_or_nonincreasing_spacing_ratio": 0.0,
        "starved": starved,
        "timing_match_100ms": {"f1": f1},
    }
