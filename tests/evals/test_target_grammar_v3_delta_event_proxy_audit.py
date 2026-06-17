from __future__ import annotations

from collections import Counter

import pytest

from pulsefield_model.evals.target_grammar_v3_delta_event_proxy_audit import (
    DeltaEventProxyWindow,
    decision_from_checks,
    delta_event_proxy_from_v3_tokens,
    guard_checks,
    proxy_metrics,
    unigram_total_bits,
)
from pulsefield_model.models.mapper.v3.vocab import MapperV3Vocab


def _time_shift_tokens(vocab: MapperV3Vocab, delta_ms: int) -> list[int]:
    return [vocab.time_shift_token_id(piece) for piece in vocab.decompose_time_shift_delta(delta_ms)]


def test_delta_event_proxy_reconstructs_event_times_signatures_and_end_gap() -> None:
    vocab = MapperV3Vocab()
    token_ids = [
        *_time_shift_tokens(vocab, 160),
        vocab.event_token_id_from_signature("T..."),
        *_time_shift_tokens(vocab, 80),
        vocab.event_token_id_from_signature(".T.."),
        *_time_shift_tokens(vocab, 760),
    ]

    proxy = delta_event_proxy_from_v3_tokens(
        token_ids,
        vocab=vocab,
        write_start_ms=0,
        write_end_ms=1000,
        chart_end_ms=1000,
        is_full_chart_end=False,
    )

    assert [row.delta_ms for row in proxy.rows] == [160, 80]
    assert [row.signature for row in proxy.rows] == ["T...", ".T.."]
    assert proxy.end_gap_ms == 760
    assert proxy.reconstructed_event_times == (160, 240)
    assert proxy.reconstructed_event_signatures == ("T...", ".T..")
    assert proxy.reconstructed_target_end_ms == 1000
    assert proxy.event_mismatch_count == 0
    assert proxy.end_mismatch_count == 0


def test_proxy_metrics_charge_factorized_kind_delta_signature_and_end_gap_bits() -> None:
    vocab = MapperV3Vocab()
    first = DeltaEventProxyWindow(
        rows=(),
        end_gap_ms=1000,
        write_start_ms=0,
        target_end_ms=1000,
        original_event_times=(),
        original_event_signatures=(),
        reconstructed_event_times=(),
        reconstructed_event_signatures=(),
        reconstructed_target_end_ms=1000,
    )
    second = delta_event_proxy_from_v3_tokens(
        [
            *_time_shift_tokens(vocab, 160),
            vocab.event_token_id_from_signature("T..."),
            *_time_shift_tokens(vocab, 840),
        ],
        vocab=vocab,
        write_start_ms=0,
        write_end_ms=1000,
        chart_end_ms=1000,
        is_full_chart_end=False,
    )

    v3_tokens = [
        *_time_shift_tokens(vocab, 1000),
        *_time_shift_tokens(vocab, 160),
        vocab.event_token_id_from_signature("T..."),
        *_time_shift_tokens(vocab, 840),
    ]
    metrics = proxy_metrics(
        [first, second],
        v3_tokens=v3_tokens,
        vocab=vocab,
    )

    assert metrics["window_count"] == 2
    assert metrics["proxy_row_count"] == 3
    assert metrics["event_reconstruction_mismatches"] == 0
    assert metrics["unique"]["event_deltas"] == 1
    assert metrics["unique"]["end_gaps"] == 2
    assert metrics["sequence_length_ratio_vs_v3"] == pytest.approx(3 / len(v3_tokens))


def test_decision_routes_positive_when_proxy_is_shorter_reconstructive_and_not_worse_bits() -> None:
    metrics = {
        "event_reconstruction_mismatches": 0,
        "end_reconstruction_mismatches": 0,
        "sequence_length_ratio_vs_v3": 0.7,
        "proxy_factorized_total_bit_ratio_vs_v3": 0.9,
        "flat_pair_tractable": True,
    }

    checks = guard_checks(metrics)
    decision = decision_from_checks(checks, metrics)

    assert decision["route"] == "TEST_DELTA_EVENT_PROXY_MODEL_HEAD"


def test_decision_mutates_when_factorized_bits_regress() -> None:
    metrics = {
        "event_reconstruction_mismatches": 0,
        "end_reconstruction_mismatches": 0,
        "sequence_length_ratio_vs_v3": 0.7,
        "proxy_factorized_total_bit_ratio_vs_v3": 1.1,
        "flat_pair_tractable": True,
    }

    checks = guard_checks(metrics)
    decision = decision_from_checks(checks, metrics)

    assert decision["route"] == "MUTATE_DELTA_EVENT_BUCKETING"


def test_unigram_total_bits_is_zero_for_degenerate_stream() -> None:
    assert unigram_total_bits(Counter({"only": 4})) == pytest.approx(0.0)
