from __future__ import annotations

import pytest

from pulsefield_model.evals.mapper_v3_delta_event_auxiliary_tiny_real_data_training import (
    decision_from_checks,
    guard_checks,
    run_delta_event_auxiliary_tiny_real_data_training,
)


def test_guard_checks_route_to_full_slice_or_full4k_when_real_training_passes() -> None:
    checks = guard_checks(
        training=_passing_training(),
        dataset={"consumed_window_count": 8, "batch_count": 2},
        context=_passing_context(),
        max_auxiliary_loss_ratio=0.75,
        max_token_loss_ratio=1.10,
    )
    decision = decision_from_checks(checks)

    assert all(checks.values())
    assert decision["route"] == "TEST_DELTA_EVENT_AUXILIARY_FULL_SLICE_OR_FULL4K_LABEL_COVERAGE"


def test_decision_mutates_when_token_loss_regresses() -> None:
    checks = {key: True for key in guard_checks(
        training=_passing_training(),
        dataset={"consumed_window_count": 8, "batch_count": 2},
        context=_passing_context(),
        max_auxiliary_loss_ratio=0.75,
        max_token_loss_ratio=1.10,
    )}
    checks["token_loss_not_materially_worse"] = False

    decision = decision_from_checks(checks)

    assert decision["route"] == "MUTATE_DELTA_EVENT_AUXILIARY_WEIGHTING"
    assert "token_loss_not_materially_worse" in decision["reason"]


def test_decision_kills_when_real_batch_labels_are_missing() -> None:
    checks = {key: True for key in guard_checks(
        training=_passing_training(),
        dataset={"consumed_window_count": 8, "batch_count": 2},
        context=_passing_context(),
        max_auxiliary_loss_ratio=0.75,
        max_token_loss_ratio=1.10,
    )}
    checks["every_batch_has_end_gap_labels"] = False

    decision = decision_from_checks(checks)

    assert decision["route"] == "KILL_DELTA_EVENT_AUXILIARY_REAL_DATA_TRAINING"
    assert "every_batch_has_end_gap_labels" in decision["reason"]


def test_gate_rejects_nonpositive_window_limit(tmp_path) -> None:
    with pytest.raises(ValueError, match="window_limit must be positive"):
        run_delta_event_auxiliary_tiny_real_data_training(
            index_path=tmp_path / "missing.parquet",
            control_teacher_cache_dir=tmp_path / "cache",
            summary_output_path=tmp_path / "summary.json",
            report_output_path=tmp_path / "report.md",
            window_limit=0,
        )


def _passing_context() -> dict:
    return {
        "fixed_slice_label_coverage": {"route": "TEST_DELTA_EVENT_AUXILIARY_TINY_REAL_DATA_TRAINING"},
        "synthetic_tiny_training": {"route": "TEST_DELTA_EVENT_AUXILIARY_FIXED_SLICE_GATE"},
    }


def _passing_training() -> dict:
    return {
        "min_event_label_count_per_batch": 4,
        "min_signature_label_count_per_batch": 4,
        "min_end_gap_label_count_per_batch": 1,
        "all_losses_finite": True,
        "initial_auxiliary_loss": 5.0,
        "final_auxiliary_loss": 2.5,
        "auxiliary_loss_ratio": 0.5,
        "total_loss_ratio": 0.8,
        "token_loss_ratio": 0.95,
        "first_step_gradients": {
            "delta_head_grad_abs": 1.0,
            "signature_head_grad_abs": 1.0,
            "end_gap_head_grad_abs": 1.0,
            "token_embedding_grad_abs": 1.0,
        },
    }
