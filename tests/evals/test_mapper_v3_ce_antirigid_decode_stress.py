from __future__ import annotations

from pulsefield_model.evals.mapper_v3_ce_antirigid_decode_stress import (
    aggregate_results,
    decision_from_aggregate,
    full32_decision_from_aggregate,
    gate_results_from_aggregate,
    select_all_runs,
    select_rigid_stress_runs,
)


def test_select_rigid_stress_runs_filters_and_sorts_case_rows() -> None:
    summary = {
        "runs": [
            {"case_id": "b", "case_index": 2, "dominant_spacing_ratio": 0.97},
            {"case_id": "a", "case_index": 1, "dominant_spacing_ratio": 0.94},
            {"case_id": "c", "case_index": 0, "dominant_spacing_ratio": 0.95},
        ]
    }

    rows = select_rigid_stress_runs(summary)

    assert [row["case_id"] for row in rows] == ["c", "b"]


def test_select_all_runs_keeps_non_rigid_rows_and_sorts() -> None:
    summary = {
        "runs": [
            {"case_id": "b", "case_index": 2, "dominant_spacing_ratio": 0.97},
            {"case_id": "a", "case_index": 1, "dominant_spacing_ratio": 0.10},
        ]
    }

    rows = select_all_runs(summary)

    assert [row["case_id"] for row in rows] == ["a", "b"]


def test_decision_from_aggregate_passes_when_rigid_drops_without_starvation_or_f1_regression() -> None:
    decision = decision_from_aggregate(
        {
            "dead_end_count": 0,
            "max_token_count": 0,
            "rigid_case_reduction": 4,
            "starved_case_delta": 0,
            "mean_f1_delta": -0.01,
        }
    )

    assert decision["route"] == "TEST_NEXT"


def test_decision_from_aggregate_kills_dead_end_or_starvation_regression() -> None:
    dead_end = decision_from_aggregate(
        {
            "dead_end_count": 1,
            "max_token_count": 0,
            "rigid_case_reduction": 4,
            "starved_case_delta": 0,
            "mean_f1_delta": 0.0,
        }
    )
    starved = decision_from_aggregate(
        {
            "dead_end_count": 0,
            "max_token_count": 0,
            "rigid_case_reduction": 4,
            "starved_case_delta": 1,
            "mean_f1_delta": 0.0,
        }
    )

    assert dead_end["route"] == "KILL"
    assert starved["route"] == "KILL"


def test_full32_decision_requires_original_gate_and_ce_rigid_reduction() -> None:
    decision = full32_decision_from_aggregate(
        {
            "dead_end_count": 0,
            "max_token_count": 0,
            "rigid_case_reduction": 5,
            "starved_case_delta": 1,
            "mean_f1_delta": -0.01,
            "candidate": {
                "all_legal": True,
                "starved_count": 6,
                "rigid_case_count": 7,
                "mean_f1_100ms": 0.64,
                "median_event_count_ratio": 1.1,
                "max_boundary_event_ratio": 0.05,
            },
        }
    )

    assert decision["route"] == "TEST_NEXT"


def test_full32_decision_kills_when_original_rigid_gate_fails() -> None:
    decision = full32_decision_from_aggregate(
        {
            "dead_end_count": 0,
            "max_token_count": 0,
            "rigid_case_reduction": 3,
            "starved_case_delta": 0,
            "mean_f1_delta": 0.0,
            "candidate": {
                "all_legal": True,
                "starved_count": 4,
                "rigid_case_count": 8,
                "mean_f1_100ms": 0.70,
                "median_event_count_ratio": 1.0,
                "max_boundary_event_ratio": 0.01,
            },
        }
    )

    assert decision["route"] == "KILL"


def test_full32_gate_results_reports_event_ratio_failure() -> None:
    guards = gate_results_from_aggregate(
        {
            "dead_end_count": 0,
            "max_token_count": 0,
            "rigid_case_reduction": 11,
            "starved_case_delta": -4,
            "mean_f1_delta": 0.01,
            "candidate": {
                "case_count": 32,
                "all_legal": True,
                "starved_count": 1,
                "rigid_case_count": 0,
                "mean_f1_100ms": 0.70,
                "median_event_count_ratio": 1.41,
                "max_boundary_event_ratio": 0.05,
            },
        },
        all_cases=True,
    )

    assert guards["median_event_count_ratio_in_range"] is False
    assert guards["rigid_no_worse_than_original_baseline"] is True


def test_aggregate_results_counts_rigid_starved_and_transform_blocks() -> None:
    rows = [
        _case("a", base_rigid=1.0, cand_rigid=0.7, base_starved=False, cand_starved=False, blocks=3),
        _case("b", base_rigid=0.99, cand_rigid=0.96, base_starved=False, cand_starved=False, blocks=2),
    ]

    aggregate = aggregate_results(rows)

    assert aggregate["baseline"]["rigid_case_count"] == 2
    assert aggregate["candidate"]["rigid_case_count"] == 1
    assert aggregate["rigid_case_reduction"] == 1
    assert aggregate["total_transform_blocks"] == 5
    assert aggregate["starved_case_delta"] == 0


def _case(
    case_id: str,
    *,
    base_rigid: float,
    cand_rigid: float,
    base_starved: bool,
    cand_starved: bool,
    blocks: int,
) -> dict[str, object]:
    baseline = _metrics(base_rigid, base_starved)
    candidate = _metrics(cand_rigid, cand_starved)
    return {
        "case_id": case_id,
        "baseline_legal": True,
        "candidate_legal": True,
        "baseline_metrics": baseline,
        "candidate_metrics": candidate,
        "delta_vs_baseline": {
            "dominant_spacing_improved": cand_rigid < base_rigid,
            "new_starved": cand_starved and not base_starved,
        },
        "candidate_rollout": {"dead_end": False, "max_tokens_exceeded": False},
        "transform": {"blocked_count": blocks, "candidate_count": blocks},
    }


def _metrics(rigid: float, starved: bool) -> dict[str, object]:
    return {
        "dominant_spacing_ratio": rigid,
        "starved": starved,
        "event_count_ratio": 1.0,
        "second_window_event_share": 0.4,
        "boundary_event_ratio": 0.0,
        "timing_match_100ms": {"f1": 0.5},
    }
