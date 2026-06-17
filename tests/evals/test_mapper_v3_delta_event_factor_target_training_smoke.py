from __future__ import annotations

import pytest

from pulsefield_model.evals.mapper_v3_delta_event_factor_target_training_smoke import (
    _loss_total_check,
    decision_from_checks,
    guard_checks,
    run_delta_event_factor_target_training_smoke,
)


def test_guard_checks_route_to_longer_training_when_smoke_passes() -> None:
    checks = guard_checks(
        metrics=_passing_metrics(),
        loss_total_check={"matches": True},
        context=_passing_context(),
    )
    decision = decision_from_checks(checks)

    assert all(checks.values())
    assert decision["route"] == "TEST_DELTA_EVENT_FACTOR_TARGET_LONGER_TRAINING_CARD"


def test_decision_routes_metric_accounting_failure_separately() -> None:
    checks = {key: True for key in guard_checks(
        metrics=_passing_metrics(),
        loss_total_check={"matches": True},
        context=_passing_context(),
    )}
    checks["loss_total_recomputed_matches"] = False

    decision = decision_from_checks(checks)

    assert decision["route"] == "MUTATE_FACTOR_TARGET_TRAINING_METRIC_ACCOUNTING"
    assert "loss_total_recomputed_matches" in decision["reason"]


def test_decision_mutates_when_training_metrics_are_missing() -> None:
    checks = {key: True for key in guard_checks(
        metrics=_passing_metrics(),
        loss_total_check={"matches": True},
        context=_passing_context(),
    )}
    checks["final_eval_factor_loss_finite"] = False

    decision = decision_from_checks(checks)

    assert decision["route"] == "MUTATE_FACTOR_TARGET_TRAINING_SMOKE"
    assert "final_eval_factor_loss_finite" in decision["reason"]


def test_loss_total_check_includes_factor_target_weight() -> None:
    metrics = {
        "loss/total": 4.0,
        "loss/token": 1.0,
        "loss/delta_event_factor_target": 6.0,
    }
    loss_config = {"lambda_delta_event_factor_target": 0.5}

    check = _loss_total_check(metrics, loss_config)

    assert check["matches"]
    assert check["recomputed"] == 4.0


def test_gate_rejects_nonpositive_max_steps(tmp_path) -> None:
    with pytest.raises(ValueError, match="max_steps must be positive"):
        run_delta_event_factor_target_training_smoke(
            dataset_root=tmp_path / "dataset",
            index_path=tmp_path / "missing.parquet",
            control_teacher_cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "run",
            summary_output_path=tmp_path / "summary.json",
            report_output_path=tmp_path / "report.md",
            max_steps=0,
        )


def _passing_metrics() -> dict:
    metric_row = {
        "loss/total": 4.0,
        "loss/token": 1.0,
        "loss/delta_event_factor_target": 12.0,
        "phase/lambda_delta_event_factor_target": 0.25,
        "delta_event_factor/kind_label_count": 10.0,
        "delta_event_factor/delta_label_count": 9.0,
        "delta_event_factor/signature_label_count": 9.0,
        "delta_event_factor/end_gap_label_count": 1.0,
    }
    return {
        "completed_steps": 2,
        "max_steps": 2,
        "is_complete": True,
        "report_exists": True,
        "checkpoint_exists": True,
        "dataset_include_delta_event_factor_target": True,
        "model_use_delta_event_factor_target": True,
        "loss_lambda_delta_event_factor_target": 0.25,
        "last_train_metrics": dict(metric_row),
        "final_train_metrics": dict(metric_row),
        "final_eval_metrics": dict(metric_row),
    }


def _passing_context() -> dict:
    return {
        "model_loss_plumbing": {
            "route": "TEST_DELTA_EVENT_FACTOR_TARGET_TRAINING_SMOKE_CARD",
        }
    }
