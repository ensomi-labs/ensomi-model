from __future__ import annotations

import pytest

from pulsefield_model.evals.mapper_v3_delta_event_factor_target_model_loss_plumbing import (
    decision_from_checks,
    guard_checks,
    run_delta_event_factor_target_model_loss_plumbing,
)


def test_guard_checks_route_to_training_smoke_when_model_loss_plumbing_passes() -> None:
    checks = guard_checks(
        row_metrics=_passing_row_metrics(),
        default_off=_passing_default_off(),
        enabled=_passing_enabled(),
        context=_passing_context(),
    )
    decision = decision_from_checks(checks)

    assert all(checks.values())
    assert decision["route"] == "TEST_DELTA_EVENT_FACTOR_TARGET_TRAINING_SMOKE_CARD"


def test_decision_kills_when_default_off_regresses() -> None:
    checks = {key: True for key in guard_checks(
        row_metrics=_passing_row_metrics(),
        default_off=_passing_default_off(),
        enabled=_passing_enabled(),
        context=_passing_context(),
    )}
    checks["default_off_outputs_absent"] = False

    decision = decision_from_checks(checks)

    assert decision["route"] == "KILL_FACTOR_TARGET_MODEL_LOSS_DEFAULT_REGRESSION"
    assert "default-off" in decision["reason"]


def test_decision_mutates_when_gradient_is_missing() -> None:
    checks = {key: True for key in guard_checks(
        row_metrics=_passing_row_metrics(),
        default_off=_passing_default_off(),
        enabled=_passing_enabled(),
        context=_passing_context(),
    )}
    checks["delta_head_gradient_nonzero"] = False

    decision = decision_from_checks(checks)

    assert decision["route"] == "MUTATE_FACTOR_TARGET_MODEL_LOSS_PLUMBING"
    assert "delta_head_gradient_nonzero" in decision["reason"]


def test_gate_rejects_nonpositive_window_limit(tmp_path) -> None:
    with pytest.raises(ValueError, match="window_limit must be positive"):
        run_delta_event_factor_target_model_loss_plumbing(
            index_path=tmp_path / "missing.parquet",
            control_teacher_cache_dir=tmp_path / "cache",
            summary_output_path=tmp_path / "summary.json",
            report_output_path=tmp_path / "report.md",
            window_limit=0,
        )


def _passing_row_metrics() -> dict:
    return {
        "audited_window_count": 32,
        "row_count": 1479,
        "event_row_count": 1447,
        "end_row_count": 32,
        "row_mask_valid_count": 1479,
        "reconstruction_mismatch_count": 0,
    }


def _passing_default_off() -> dict:
    return {
        "batch_has_factor_target": False,
        "factor_outputs_absent": True,
        "lambda_metric": 0.0,
    }


def _passing_enabled() -> dict:
    return {
        "batch_size": 4,
        "row_count": 128,
        "event_row_count": 124,
        "outputs_align_rows": True,
        "loss_finite": True,
        "factor_loss": 1.0,
        "kind_label_count": 128,
        "delta_label_count": 124,
        "signature_label_count": 124,
        "end_gap_label_count": 4,
        "gradient_abs": {
            "kind_head": 1.0,
            "delta_head": 1.0,
            "signature_head": 1.0,
            "end_gap_head": 1.0,
        },
    }


def _passing_context() -> dict:
    return {
        "data_contract": {
            "route": "TEST_DELTA_EVENT_FACTOR_TARGET_MODEL_LOSS_PLUMBING_CARD",
            "row_count": 1479,
            "event_row_count": 1447,
            "end_row_count": 32,
        },
        "tiny_model": {
            "route": "TEST_DELTA_EVENT_FACTOR_TARGET_PRODUCTION_PLUMBING_CARD",
        },
    }
