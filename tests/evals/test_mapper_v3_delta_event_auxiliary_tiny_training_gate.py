from __future__ import annotations

import json
from pathlib import Path

import pytest

from pulsefield_model.evals.mapper_v3_delta_event_auxiliary_tiny_training_gate import (
    decision_from_checks,
    run_delta_event_auxiliary_tiny_training_gate,
)


def test_gate_routes_to_fixed_slice_when_synthetic_overfit_passes(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    report_path = tmp_path / "report.md"

    summary = run_delta_event_auxiliary_tiny_training_gate(
        summary_output_path=summary_path,
        report_output_path=report_path,
    )

    assert summary["decision"]["route"] == "TEST_DELTA_EVENT_AUXILIARY_FIXED_SLICE_GATE"
    assert summary["checks"]["default_off_auxiliary_logits_absent"] is True
    assert summary["checks"]["auxiliary_loss_decreased"] is True
    assert summary["training"]["event_label_count"] > 0
    assert summary["training"]["end_gap_label_count"] > 0
    assert summary["training"]["auxiliary_loss_ratio"] <= summary["config"]["max_aux_loss_ratio"]
    assert json.loads(summary_path.read_text(encoding="utf-8"))["decision"]["route"] == summary["decision"]["route"]
    assert "Route: `TEST_DELTA_EVENT_AUXILIARY_FIXED_SLICE_GATE`" in report_path.read_text(encoding="utf-8")


def test_decision_mutates_when_auxiliary_loss_does_not_decrease() -> None:
    checks = _passing_checks()
    checks["auxiliary_loss_decreased"] = False

    decision = decision_from_checks(checks)

    assert decision["route"] == "MUTATE_DELTA_EVENT_AUXILIARY_TRAINING"
    assert "auxiliary_loss_decreased" in decision["reason"]


def test_decision_kills_when_hard_gradient_check_fails() -> None:
    checks = _passing_checks()
    checks["delta_head_gradient_nonzero"] = False

    decision = decision_from_checks(checks)

    assert decision["route"] == "KILL_DELTA_EVENT_AUXILIARY_TRAINING"
    assert "delta_head_gradient_nonzero" in decision["reason"]


def test_gate_rejects_nonpositive_steps(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="steps must be positive"):
        run_delta_event_auxiliary_tiny_training_gate(
            summary_output_path=tmp_path / "summary.json",
            report_output_path=tmp_path / "report.md",
            steps=0,
        )


def _passing_checks() -> dict[str, bool]:
    return {
        "default_off_auxiliary_logits_absent": True,
        "default_off_lambda_zero": True,
        "event_labels_positive": True,
        "end_gap_labels_positive": True,
        "all_losses_finite": True,
        "initial_auxiliary_loss_positive": True,
        "final_auxiliary_loss_finite": True,
        "auxiliary_loss_decreased": True,
        "total_loss_decreased": True,
        "token_loss_not_worse": True,
        "delta_head_gradient_nonzero": True,
        "signature_head_gradient_nonzero": True,
        "end_gap_head_gradient_nonzero": True,
        "shared_decoder_gradient_nonzero": True,
        "no_rollout": True,
        "no_tokenizer_dataset_default_change": True,
        "no_c3_backreference_or_future_lookup": True,
    }
