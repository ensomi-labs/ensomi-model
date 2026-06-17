from __future__ import annotations

from pulsefield_model.evals.mapper_v3_delta_event_auxiliary_fixed_slice_label_coverage import (
    decision_from_checks,
    guard_checks,
    window_label_coverage,
)
from pulsefield_model.models.mapper.v3.vocab import MapperV3Vocab


def _time_shift_tokens(vocab: MapperV3Vocab, delta_ms: int) -> list[int]:
    return [vocab.time_shift_token_id(piece) for piece in vocab.decompose_time_shift_delta(delta_ms)]


def test_window_label_coverage_derives_event_deltas_and_end_gap() -> None:
    vocab = MapperV3Vocab()
    tokens = [
        *_time_shift_tokens(vocab, 80),
        vocab.event_token_id_from_signature("T..."),
        *_time_shift_tokens(vocab, 160),
        vocab.event_token_id_from_signature(".T.."),
    ]
    current_ms = [0, 80, 80, 180, 240]

    coverage = window_label_coverage(
        token_ids=tokens,
        current_ms=current_ms,
        write_start_ms=0,
        write_end_ms=1000,
        chart_end_ms=1000,
        is_full_chart_end=False,
        vocab=vocab,
        delta_max_ms=8000,
        end_gap_max_ms=8000,
    )

    assert coverage["event_token_count"] == 2
    assert coverage["event_label_count"] == 2
    assert coverage["signature_label_count"] == 2
    assert coverage["end_gap_label_count"] == 1
    assert coverage["event_deltas"] == [80, 160]
    assert coverage["event_signatures"] == ["T...", ".T.."]
    assert coverage["end_gap_ms"] == 760
    assert coverage["out_of_range_delta_count"] == 0
    assert coverage["out_of_range_end_gap_count"] == 0


def test_window_label_coverage_reports_range_misses_without_throwing() -> None:
    vocab = MapperV3Vocab()
    tokens = [
        *_time_shift_tokens(vocab, 80),
        vocab.event_token_id_from_signature("T..."),
        *_time_shift_tokens(vocab, 160),
        vocab.event_token_id_from_signature(".T.."),
    ]
    current_ms = [0, 80, 80, 180, 240]

    coverage = window_label_coverage(
        token_ids=tokens,
        current_ms=current_ms,
        write_start_ms=0,
        write_end_ms=1000,
        chart_end_ms=1000,
        is_full_chart_end=False,
        vocab=vocab,
        delta_max_ms=100,
        end_gap_max_ms=100,
    )

    assert coverage["out_of_range_delta_count"] == 1
    assert coverage["out_of_range_end_gap_count"] == 1


def test_guard_checks_and_decision_route_to_tiny_real_data_training_when_counts_match() -> None:
    metrics = {
        "audited_window_count": 2,
        "label_error_count": 0,
        "negative_delta_count": 0,
        "negative_end_gap_count": 0,
        "out_of_range_delta_count": 0,
        "out_of_range_end_gap_count": 0,
        "event_label_count": 5,
        "signature_label_count": 5,
        "event_token_count": 5,
        "end_gap_label_count": 2,
    }

    checks = guard_checks(metrics)
    decision = decision_from_checks(checks)

    assert all(checks.values())
    assert decision["route"] == "TEST_DELTA_EVENT_AUXILIARY_TINY_REAL_DATA_TRAINING"


def test_decision_mutates_head_range_when_range_check_fails() -> None:
    checks = {key: True for key in guard_checks(
        {
            "audited_window_count": 1,
            "label_error_count": 0,
            "negative_delta_count": 0,
            "negative_end_gap_count": 0,
            "out_of_range_delta_count": 0,
            "out_of_range_end_gap_count": 0,
            "event_label_count": 1,
            "signature_label_count": 1,
            "event_token_count": 1,
            "end_gap_label_count": 1,
        }
    )}
    checks["delta_out_of_range_zero"] = False

    decision = decision_from_checks(checks)

    assert decision["route"] == "MUTATE_DELTA_EVENT_AUXILIARY_HEAD_RANGE"
    assert "delta_out_of_range_zero" in decision["reason"]
