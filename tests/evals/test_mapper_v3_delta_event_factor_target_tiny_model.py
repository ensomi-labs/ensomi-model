from __future__ import annotations

import pytest

from pulsefield_model.evals.mapper_v3_delta_event_factor_target_tiny_model import (
    BOS_INPUT_KIND,
    END_KIND,
    EVENT_KIND,
    IGNORE_INDEX,
    decision_from_checks,
    factorized_window_from_v3_tokens,
    guard_checks,
    run_delta_event_factor_target_tiny_model,
)
from pulsefield_model.models.mapper.v3.vocab import MapperV3Vocab


def _time_shift_tokens(vocab: MapperV3Vocab, delta_ms: int) -> list[int]:
    return [vocab.time_shift_token_id(piece) for piece in vocab.decompose_time_shift_delta(delta_ms)]


def test_factorized_window_builds_shifted_rows_and_reconstructs() -> None:
    vocab = MapperV3Vocab()
    tokens = [
        *_time_shift_tokens(vocab, 80),
        vocab.event_token_id_from_signature("T..."),
        *_time_shift_tokens(vocab, 160),
        vocab.event_token_id_from_signature(".T.."),
    ]

    window = factorized_window_from_v3_tokens(
        tokens,
        vocab=vocab,
        write_start_ms=0,
        write_end_ms=1000,
        chart_end_ms=1000,
        is_full_chart_end=False,
    )

    assert window.target_kind == (EVENT_KIND, EVENT_KIND, END_KIND)
    assert window.target_delta == (8, 16, IGNORE_INDEX)
    assert window.target_end_gap == (IGNORE_INDEX, IGNORE_INDEX, 76)
    assert window.input_kind == (BOS_INPUT_KIND, EVENT_KIND, EVENT_KIND)
    assert window.input_delta == (0, 8, 16)
    assert window.input_kind[-1] != END_KIND
    assert window.event_count == 2
    assert window.end_count == 1
    assert window.reconstruction_mismatches == 0


def test_guard_checks_route_to_production_plumbing_when_tiny_model_passes() -> None:
    checks = guard_checks(
        dataset=_passing_dataset(),
        training=_passing_training(),
        context=_passing_context(),
        max_total_loss_ratio=0.75,
    )
    decision = decision_from_checks(checks)

    assert all(checks.values())
    assert decision["route"] == "TEST_DELTA_EVENT_FACTOR_TARGET_PRODUCTION_PLUMBING_CARD"


def test_decision_mutates_when_loss_does_not_decrease_enough() -> None:
    checks = {key: True for key in guard_checks(
        dataset=_passing_dataset(),
        training=_passing_training(),
        context=_passing_context(),
        max_total_loss_ratio=0.75,
    )}
    checks["total_loss_decreased"] = False

    decision = decision_from_checks(checks)

    assert decision["route"] == "MUTATE_DELTA_EVENT_FACTOR_TARGET_MODEL_SCALE_OR_LR"
    assert "total_loss_decreased" in decision["reason"]


def test_decision_kills_when_reconstruction_fails() -> None:
    checks = {key: True for key in guard_checks(
        dataset=_passing_dataset(),
        training=_passing_training(),
        context=_passing_context(),
        max_total_loss_ratio=0.75,
    )}
    checks["reconstruction_mismatches_zero"] = False

    decision = decision_from_checks(checks)

    assert decision["route"] == "KILL_DELTA_EVENT_FACTOR_TARGET_MODEL_SURFACE"
    assert "reconstruction_mismatches_zero" in decision["reason"]


def test_gate_rejects_nonpositive_window_limit(tmp_path) -> None:
    with pytest.raises(ValueError, match="window_limit must be positive"):
        run_delta_event_factor_target_tiny_model(
            index_path=tmp_path / "missing.parquet",
            control_teacher_cache_dir=tmp_path / "cache",
            summary_output_path=tmp_path / "summary.json",
            report_output_path=tmp_path / "report.md",
            window_limit=0,
        )


def _passing_context() -> dict:
    return {
        "full4k_bit_proxy": {
            "route": "TEST_DELTA_EVENT_FACTOR_TARGET_MODEL_CARD",
            "row_ratio": 0.403157,
            "bit_ratio": 0.792753,
            "flat_pair_tractable": False,
        }
    }


def _passing_dataset() -> dict:
    return {
        "consumed_window_count": 4,
        "row_count": 20,
        "event_row_count": 16,
        "end_row_count": 4,
        "reconstruction_mismatch_count": 0,
    }


def _passing_training() -> dict:
    return {
        "all_losses_finite": True,
        "total_loss_ratio": 0.5,
        "kind_loss_ratio": 0.5,
        "delta_loss_ratio": 0.5,
        "signature_loss_ratio": 0.5,
        "end_gap_loss_ratio": 0.5,
        "first_step_gradients": {
            "kind_head_grad_abs": 1.0,
            "delta_head_grad_abs": 1.0,
            "signature_head_grad_abs": 1.0,
            "end_gap_head_grad_abs": 1.0,
            "kind_embedding_grad_abs": 1.0,
        },
    }
