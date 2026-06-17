from __future__ import annotations

from pulsefield_model.evals.mapper_v3_delta_event_auxiliary_full4k_label_coverage import (
    _accumulate,
    _empty_aggregate,
    _finalize_metrics,
    decision_from_checks,
    guard_checks,
)


def test_guard_checks_route_to_bit_proxy_when_full4k_coverage_passes() -> None:
    checks = guard_checks(metrics=_passing_metrics(), context=_passing_context())
    decision = decision_from_checks(checks)

    assert all(checks.values())
    assert decision["route"] == "TEST_DELTA_EVENT_FULL4K_BIT_PROXY_OR_FACTOR_TARGET_CARD"


def test_decision_mutates_range_when_end_gap_exceeds_head_range() -> None:
    checks = {key: True for key in guard_checks(metrics=_passing_metrics(), context=_passing_context())}
    checks["end_gap_out_of_range_zero"] = False

    decision = decision_from_checks(checks)

    assert decision["route"] == "MUTATE_DELTA_EVENT_HEAD_RANGE_OR_BUCKETING"
    assert "end_gap_out_of_range_zero" in decision["reason"]


def test_decision_mutates_runtime_when_scope_is_partial() -> None:
    checks = {key: True for key in guard_checks(metrics=_passing_metrics(), context=_passing_context())}
    checks["full_scope"] = False

    decision = decision_from_checks(checks)

    assert decision["route"] == "MUTATE_FULL4K_LABEL_AUDIT_RUNTIME"
    assert "full_scope" in decision["reason"]


def test_aggregate_counts_labels_and_end_gap_distribution() -> None:
    aggregate = _empty_aggregate()
    _accumulate(
        aggregate,
        {
            "v3_token_count": 5,
            "time_shift_token_count": 2,
            "event_token_count": 2,
            "eos_count": 1,
            "event_label_count": 2,
            "signature_label_count": 2,
            "end_gap_label_count": 1,
            "out_of_range_delta_count": 0,
            "out_of_range_end_gap_count": 0,
            "negative_delta_count": 0,
            "negative_end_gap_count": 0,
            "event_deltas": [80, 160],
            "event_signatures": ["T...", ".T.."],
            "end_gap_ms": 760,
        },
    )

    metrics = _finalize_metrics(
        aggregate,
        candidate_window_count=1,
        source_window_count=1,
        skipped_stride_window_count=0,
        last_source_index=0,
        errors=[],
        unsupported_errors=[],
        limit=None,
    )

    assert metrics["audited_window_count"] == 1
    assert metrics["event_label_count"] == 2
    assert metrics["signature_label_count"] == 2
    assert metrics["end_gap_label_count"] == 1
    assert metrics["max_event_delta_ms"] == 160
    assert metrics["max_end_gap_ms"] == 760
    assert metrics["unique_event_delta_count"] == 2
    assert metrics["unique_end_gap_count"] == 1


def _passing_context() -> dict:
    return {
        "v3_full_dataset_audit": {
            "decision": {"route": "TEST_NEXT", "positive_gate_passed": True},
            "dataset": {"audit_window_count": 10},
        },
        "fixed_slice_label_coverage": {"route": "TEST_DELTA_EVENT_AUXILIARY_TINY_REAL_DATA_TRAINING"},
        "tiny_real_data_training": {"route": "TEST_DELTA_EVENT_AUXILIARY_FULL_SLICE_OR_FULL4K_LABEL_COVERAGE"},
    }


def _passing_metrics() -> dict:
    return {
        "limit": None,
        "candidate_window_count": 10,
        "audited_window_count": 10,
        "unsupported_window_count": 0,
        "label_error_count": 0,
        "negative_delta_count": 0,
        "negative_end_gap_count": 0,
        "out_of_range_delta_count": 0,
        "out_of_range_end_gap_count": 0,
        "event_label_count": 20,
        "signature_label_count": 20,
        "event_token_count": 20,
        "end_gap_label_count": 10,
    }
