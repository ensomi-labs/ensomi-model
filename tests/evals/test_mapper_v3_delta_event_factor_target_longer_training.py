from __future__ import annotations

import pytest

from pulsefield_model.evals.mapper_v3_delta_event_factor_target_longer_training import (
    _stability_metrics,
    decision_from_checks,
    guard_checks,
    run_delta_event_factor_target_longer_training,
)


def test_guard_checks_route_to_full4k_training_when_longer_run_is_stable(tmp_path) -> None:
    report = _report(tmp_path, final_factor=18.0)
    (tmp_path / "checkpoint.pt").write_bytes(b"fake")
    (tmp_path / "report.json").write_text("{}", encoding="utf-8")
    stability = _stability_metrics(
        report,
        checkpoint_path=tmp_path / "checkpoint.pt",
        max_factor_loss_ratio=1.15,
    )
    checks = guard_checks(
        stability=stability,
        loss_total_check={"matches": True},
        context=_passing_context(),
    )
    decision = decision_from_checks(checks)

    assert all(checks.values())
    assert decision["route"] == "TEST_DELTA_EVENT_FACTOR_TARGET_FULL4K_TRAINING_CARD"


def test_decision_mutates_lr_when_factor_loss_ratio_exceeds_limit() -> None:
    checks = {key: True for key in guard_checks(
        stability=_passing_stability(),
        loss_total_check={"matches": True},
        context=_passing_context(),
    )}
    checks["factor_loss_ratio_within_limit"] = False

    decision = decision_from_checks(checks)

    assert decision["route"] == "MUTATE_FACTOR_TARGET_LR_OR_LOSS_WEIGHT"
    assert "factor_loss_ratio_within_limit" in decision["reason"]


def test_decision_mutates_plumbing_when_eval_losses_are_nonfinite() -> None:
    checks = {key: True for key in guard_checks(
        stability=_passing_stability(),
        loss_total_check={"matches": True},
        context=_passing_context(),
    )}
    checks["all_eval_losses_finite"] = False

    decision = decision_from_checks(checks)

    assert decision["route"] == "MUTATE_FACTOR_TARGET_LONGER_TRAINING_PLUMBING"
    assert "all_eval_losses_finite" in decision["reason"]


def test_stability_metrics_collect_eval_points_and_ratio(tmp_path) -> None:
    report = _report(tmp_path, final_factor=18.0)
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"fake")
    (tmp_path / "report.json").write_text("{}", encoding="utf-8")

    stability = _stability_metrics(report, checkpoint_path=checkpoint_path, max_factor_loss_ratio=1.15)

    assert stability["eval_steps"] == [1, 8, 16]
    assert stability["eval_point_count"] == 3
    assert stability["all_eval_losses_finite"]
    assert stability["factor_loss_ratio"] == pytest.approx(0.9)
    assert stability["factor_loss_ratio_within_limit"]


def test_gate_rejects_nonpositive_max_steps(tmp_path) -> None:
    with pytest.raises(ValueError, match="max_steps must be positive"):
        run_delta_event_factor_target_longer_training(
            dataset_root=tmp_path / "dataset",
            index_path=tmp_path / "missing.parquet",
            control_teacher_cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "run",
            summary_output_path=tmp_path / "summary.json",
            report_output_path=tmp_path / "report.md",
            max_steps=0,
        )


def _passing_context() -> dict:
    return {
        "training_smoke": {
            "route": "TEST_DELTA_EVENT_FACTOR_TARGET_LONGER_TRAINING_CARD",
        }
    }


def _passing_stability() -> dict:
    metric_row = _metrics(20.0)
    return {
        "completed_steps": 16,
        "max_steps": 16,
        "is_complete": True,
        "report_exists": True,
        "checkpoint_exists": True,
        "dataset_include_delta_event_factor_target": True,
        "model_use_delta_event_factor_target": True,
        "loss_lambda_delta_event_factor_target": 0.25,
        "eval_point_count": 3,
        "all_eval_losses_finite": True,
        "factor_loss_ratio_within_limit": True,
        "final_eval_metrics": dict(metric_row),
        "final_train_metrics": dict(metric_row),
    }


def _report(tmp_path, *, final_factor: float) -> dict:
    return {
        "completed_steps": 16,
        "max_steps": 16,
        "is_complete": True,
        "dataset": {"include_delta_event_factor_target": True},
        "model_config": {"use_delta_event_factor_target": True},
        "loss_config": {"lambda_delta_event_factor_target": 0.25},
        "history": [
            {"step": 1, "eval": _metrics(20.0)},
            {"step": 8, "eval": _metrics(19.0)},
            {"step": 16, "eval": _metrics(final_factor)},
        ],
        "final_train_metrics": _metrics(17.5),
        "final_eval_metrics": _metrics(final_factor),
    }


def _metrics(factor_loss: float) -> dict:
    return {
        "loss/total": 1.0 + 0.25 * float(factor_loss),
        "loss/token": 1.0,
        "loss/delta_event_factor_target": float(factor_loss),
        "phase/lambda_delta_event_factor_target": 0.25,
        "delta_event_factor/kind_label_count": 10.0,
        "delta_event_factor/delta_label_count": 9.0,
        "delta_event_factor/signature_label_count": 9.0,
        "delta_event_factor/end_gap_label_count": 1.0,
    }
