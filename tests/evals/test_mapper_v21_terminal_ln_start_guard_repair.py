from __future__ import annotations

from pulsefield_model.evals.mapper_v21_terminal_ln_start_guard_repair import (
    CASE_04_ID,
    CASE_05_ID,
    aggregate_results,
    decision_from_aggregate,
)


def test_decision_passes_stress_gate_when_primary_repairs_are_legal_and_f1_holds() -> None:
    rows = [
        _case(CASE_04_ID, "baseline_guard", f1_delta=-0.01),
        _case(CASE_04_ID, "anti_rigid_guard", f1_delta=-0.01, blocks=3),
        _case(CASE_05_ID, "baseline_guard", f1_delta=-0.01),
        _case(CASE_05_ID, "anti_rigid_guard", f1_delta=-0.01, blocks=4),
    ]

    aggregate = aggregate_results(rows)
    decision = decision_from_aggregate(aggregate, rows, full32=False)

    assert aggregate["all_candidate_legal"] is True
    assert aggregate["new_starved_count"] == 0
    assert decision["route"] == "TEST_NEXT"
    assert decision["next_step"] == "Run the optional full32 widening gate before changing defaults."


def test_decision_kills_when_primary_case04_baseline_still_illegal() -> None:
    rows = [
        _case(CASE_04_ID, "baseline_guard", legal=False),
        _case(CASE_04_ID, "anti_rigid_guard", blocks=3),
        _case(CASE_05_ID, "baseline_guard"),
        _case(CASE_05_ID, "anti_rigid_guard", blocks=4),
    ]

    aggregate = aggregate_results(rows)
    decision = decision_from_aggregate(aggregate, rows, full32=False)

    assert decision["route"] == "KILL"
    assert decision["primary_case04_baseline_legal"] is False


def test_decision_kills_when_primary_case05_anti_rigid_still_illegal() -> None:
    rows = [
        _case(CASE_04_ID, "baseline_guard"),
        _case(CASE_04_ID, "anti_rigid_guard", blocks=3),
        _case(CASE_05_ID, "baseline_guard"),
        _case(CASE_05_ID, "anti_rigid_guard", legal=False, blocks=4),
    ]

    aggregate = aggregate_results(rows)
    decision = decision_from_aggregate(aggregate, rows, full32=False)

    assert decision["route"] == "KILL"
    assert decision["primary_case05_anti_rigid_legal"] is False


def test_decision_kills_new_starvation_even_when_legal() -> None:
    rows = [
        _case(CASE_04_ID, "baseline_guard"),
        _case(CASE_04_ID, "anti_rigid_guard", blocks=3),
        _case(CASE_05_ID, "baseline_guard"),
        _case(CASE_05_ID, "anti_rigid_guard", blocks=4, new_starved=True, starved=True),
    ]

    aggregate = aggregate_results(rows)
    decision = decision_from_aggregate(aggregate, rows, full32=False)

    assert aggregate["new_starved_count"] == 1
    assert decision["route"] == "KILL"


def test_decision_kills_mean_f1_regression_beyond_gate() -> None:
    rows = [
        _case(CASE_04_ID, "baseline_guard", f1_delta=-0.04),
        _case(CASE_04_ID, "anti_rigid_guard", f1_delta=-0.04, blocks=3),
        _case(CASE_05_ID, "baseline_guard", f1_delta=-0.04),
        _case(CASE_05_ID, "anti_rigid_guard", f1_delta=-0.04, blocks=4),
    ]

    aggregate = aggregate_results(rows)
    decision = decision_from_aggregate(aggregate, rows, full32=False)

    assert aggregate["mean_f1_delta"] == -0.04
    assert decision["route"] == "KILL"


def _case(
    case_id: str,
    mode: str,
    *,
    legal: bool = True,
    f1_delta: float = 0.0,
    starved: bool = False,
    new_starved: bool = False,
    blocks: int = 0,
) -> dict[str, object]:
    previous_f1 = 0.7
    candidate_f1 = previous_f1 + f1_delta
    return {
        "case_id": case_id,
        "mode": mode,
        "candidate_legal": legal,
        "candidate_rollout": {
            "completed": legal,
            "dead_end": not legal,
            "max_tokens_exceeded": False,
            "token_count": 64,
            "terminal_ms": 16_000,
        },
        "candidate_metrics": _metrics(f1=candidate_f1, starved=starved),
        "previous_metrics": _metrics(f1=previous_f1, starved=False),
        "candidate_vs_previous": {
            "f1_delta": f1_delta,
            "dominant_spacing_ratio_delta": -0.1,
            "event_count_ratio_delta": 0.0,
            "new_starved": new_starved,
        },
        "transform": {"blocked_count": blocks},
    }


def _metrics(*, f1: float, starved: bool) -> dict[str, object]:
    return {
        "dominant_spacing_ratio": 0.8,
        "event_count_ratio": 1.0,
        "starved": starved,
        "timing_match_100ms": {"f1": f1},
    }
