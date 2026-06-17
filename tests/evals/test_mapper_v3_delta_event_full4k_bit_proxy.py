from __future__ import annotations

from pulsefield_model.evals.mapper_v3_delta_event_full4k_bit_proxy import (
    _accumulate_proxy,
    _empty_counters,
    decision_from_checks,
    finalize_proxy_metrics,
    guard_checks,
)
from pulsefield_model.evals.target_grammar_v3_delta_event_proxy_audit import delta_event_proxy_from_v3_tokens
from pulsefield_model.models.mapper.v3.vocab import MapperV3Vocab


def test_finalize_proxy_metrics_counts_rows_bits_and_reconstruction() -> None:
    vocab = MapperV3Vocab()
    tokens = [
        vocab.time_shift_token_id(80),
        vocab.event_token_id_from_signature("T..."),
        vocab.time_shift_token_id(80),
        vocab.time_shift_token_id(80),
        vocab.event_token_id_from_signature(".T.."),
    ]
    proxy = delta_event_proxy_from_v3_tokens(
        tokens,
        vocab=vocab,
        write_start_ms=0,
        write_end_ms=1000,
        chart_end_ms=1000,
        is_full_chart_end=False,
    )
    counters = _empty_counters()

    _accumulate_proxy(counters, token_ids=tokens, proxy=proxy, vocab=vocab)
    metrics = finalize_proxy_metrics(
        counters,
        candidate_window_count=1,
        source_window_count=1,
        skipped_stride_window_count=0,
        unsupported_window_count=0,
        label_error_count=0,
        last_source_index=0,
        limit=None,
        vocab=vocab,
    )

    assert metrics["audited_window_count"] == 1
    assert metrics["v3_token_count"] == 5
    assert metrics["proxy_row_count"] == 3
    assert metrics["event_row_count"] == 2
    assert metrics["end_row_count"] == 1
    assert metrics["event_reconstruction_mismatches"] == 0
    assert metrics["end_reconstruction_mismatches"] == 0
    assert metrics["unique"]["event_deltas"] == 2
    assert metrics["unique"]["event_signatures"] == 2
    assert metrics["unique"]["end_gaps"] == 1


def test_guard_checks_route_to_flat_or_factorized_when_proxy_passes() -> None:
    checks = guard_checks(metrics=_passing_metrics(), context=_passing_context())
    decision = decision_from_checks(checks, _passing_metrics())

    assert all(checks.values())
    assert decision["route"] == "TEST_DELTA_EVENT_FLAT_OR_FACTORIZED_TARGET_CARD"


def test_decision_routes_to_factorized_when_flat_pairs_are_not_tractable() -> None:
    metrics = _passing_metrics()
    metrics["flat_pair_tractable"] = False
    metrics["unique"]["flat_delta_event_pairs"] = 4097
    checks = guard_checks(metrics=metrics, context=_passing_context())

    decision = decision_from_checks(checks, metrics)

    assert all(checks.values())
    assert decision["route"] == "TEST_DELTA_EVENT_FACTOR_TARGET_MODEL_CARD"


def test_decision_mutates_when_factorized_bits_regress() -> None:
    checks = {key: True for key in guard_checks(metrics=_passing_metrics(), context=_passing_context())}
    checks["factorized_bits_not_worse_than_v3"] = False

    decision = decision_from_checks(checks, _passing_metrics())

    assert decision["route"] == "MUTATE_DELTA_EVENT_BUCKETING"
    assert "regressed" in decision["reason"]


def test_decision_kills_when_reconstruction_fails() -> None:
    checks = {key: True for key in guard_checks(metrics=_passing_metrics(), context=_passing_context())}
    checks["event_reconstruction_zero"] = False

    decision = decision_from_checks(checks, _passing_metrics())

    assert decision["route"] == "KILL_DELTA_EVENT_FACTOR_TARGET"
    assert "event_reconstruction_zero" in decision["reason"]


def _passing_context() -> dict:
    return {
        "v3_full_dataset_audit": {
            "decision": {"route": "TEST_NEXT", "positive_gate_passed": True},
            "dataset": {"audit_window_count": 10},
        },
        "full4k_label_coverage": {
            "route": "TEST_DELTA_EVENT_FULL4K_BIT_PROXY_OR_FACTOR_TARGET_CARD",
            "metrics": {"audited_window_count": 10, "event_label_count": 20},
        },
    }


def _passing_metrics() -> dict:
    return {
        "limit": None,
        "audited_window_count": 10,
        "label_error_count": 0,
        "event_reconstruction_mismatches": 0,
        "end_reconstruction_mismatches": 0,
        "sequence_length_ratio_vs_v3": 0.5,
        "proxy_factorized_total_bit_ratio_vs_v3": 0.8,
        "proxy_row_count": 30,
        "v3_token_count": 60,
        "event_row_count": 20,
        "flat_pair_tractable": True,
        "unique": {"flat_delta_event_pairs": 12},
    }
