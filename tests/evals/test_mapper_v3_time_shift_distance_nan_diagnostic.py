from __future__ import annotations

import json
from pathlib import Path

import torch

from pulsefield_model.evals.mapper_v3_time_shift_distance_nan_diagnostic import (
    analyze_time_shift_distance_surfaces,
    diagnostic_decision,
    report_markdown,
    write_report,
    write_summary_json,
)
from pulsefield_model.models.mapper.shared.loss import time_shift_distance_loss
from pulsefield_model.models.mapper.v3 import MapperV3Vocab


def test_analyze_surfaces_counts_all_nonfinite_masked_time_shift_rows() -> None:
    vocab = MapperV3Vocab()
    target_token = vocab.time_shift_token_id(80)
    target = torch.tensor([[target_token]], dtype=torch.long)
    mask = torch.ones_like(target, dtype=torch.bool)
    logits_final = torch.zeros((1, 1, vocab.size), dtype=torch.float32)
    logits_final[:, :, list(vocab.time_shift_token_ids)] = -torch.inf
    pre_mask = torch.zeros_like(logits_final)
    pre_mask[0, 0, target_token] = 5.0

    summary = analyze_time_shift_distance_surfaces(
        logits_final=logits_final,
        pre_mask_logits=pre_mask,
        target_tokens=target,
        target_mask=mask,
        vocab=vocab,
        current_ms=torch.tensor([[120]], dtype=torch.long),
    )
    masked_loss = time_shift_distance_loss(
        logits_final=logits_final,
        target_tokens=target,
        vocab=vocab,
        target_mask=mask,
    )
    pre_mask_loss = time_shift_distance_loss(
        logits_final=pre_mask,
        target_tokens=target,
        vocab=vocab,
        target_mask=mask,
    )

    assert summary["target_time_shift_rows"] == 1
    assert summary["all_nonfinite_masked_rows"] == 1
    assert summary["total_rows"] == 1
    assert summary["valid_rows"] == 1
    assert summary["all_nonfinite_masked_rows_any"] == 1
    assert summary["all_nonfinite_masked_valid_rows"] == 1
    assert summary["all_nonfinite_masked_invalid_rows"] == 0
    assert summary["all_nonfinite_masked_valid_non_target_rows"] == 0
    assert summary["finite_pre_mask_candidate_rows"] == 1
    assert summary["gold_masked_finite_rows"] == 0
    assert summary["gold_pre_mask_finite_rows"] == 1
    assert torch.isnan(masked_loss)
    assert torch.isfinite(pre_mask_loss)
    assert summary["examples"][0]["target_token"] == "TS_80"
    assert summary["examples"][0]["current_ms"] == 120


def test_analyze_surfaces_counts_all_nonfinite_non_target_rows() -> None:
    vocab = MapperV3Vocab()
    target_token = vocab.time_shift_token_id(80)
    event_token = vocab.event_token_id_from_signature("T...")
    target = torch.tensor([[target_token, event_token]], dtype=torch.long)
    mask = torch.ones_like(target, dtype=torch.bool)
    logits_final = torch.zeros((1, 2, vocab.size), dtype=torch.float32)
    logits_final[:, :, list(vocab.time_shift_token_ids)] = -torch.inf
    logits_final[0, 0, target_token] = 5.0
    pre_mask = torch.zeros_like(logits_final)
    pre_mask[0, 0, target_token] = 5.0

    summary = analyze_time_shift_distance_surfaces(
        logits_final=logits_final,
        pre_mask_logits=pre_mask,
        target_tokens=target,
        target_mask=mask,
        vocab=vocab,
    )
    masked_loss = time_shift_distance_loss(
        logits_final=logits_final,
        target_tokens=target,
        vocab=vocab,
        target_mask=mask,
    )

    assert summary["target_time_shift_rows"] == 1
    assert summary["all_nonfinite_masked_rows"] == 0
    assert summary["all_nonfinite_masked_rows_any"] == 1
    assert summary["all_nonfinite_masked_valid_rows"] == 1
    assert summary["all_nonfinite_masked_valid_non_target_rows"] == 1
    assert summary["all_nonfinite_masked_invalid_rows"] == 0
    assert summary["finite_masked_candidate_rows"] == 1
    assert torch.isfinite(masked_loss)


def test_analyze_surfaces_reports_no_target_time_shift_rows() -> None:
    vocab = MapperV3Vocab()
    target = torch.tensor([[vocab.event_token_id_from_signature("T...")]], dtype=torch.long)
    mask = torch.ones_like(target, dtype=torch.bool)
    logits = torch.zeros((1, 1, vocab.size), dtype=torch.float32)

    summary = analyze_time_shift_distance_surfaces(
        logits_final=logits,
        pre_mask_logits=logits.clone(),
        target_tokens=target,
        target_mask=mask,
        vocab=vocab,
    )

    assert summary["target_time_shift_rows"] == 0
    assert summary["all_nonfinite_masked_rows"] == 0
    assert summary["examples"] == []


def test_decision_routes_to_premask_repair_when_masked_rows_and_premask_gradient_pass() -> None:
    decision = diagnostic_decision(
        {
            "target_time_shift_rows": 4,
            "all_nonfinite_masked_rows": 4,
            "pre_mask_loss_all_finite": True,
            "pre_mask_gradient_all_finite": True,
            "pre_mask_gradient_any_nonzero": True,
            "masked_loss_any_nonfinite": True,
        }
    )

    assert decision["route"] == "TEST_PREMASK_TIME_SHIFT_DISTANCE_REPAIR"


def test_decision_routes_to_row_filtered_repair_when_non_target_rows_create_nan() -> None:
    decision = diagnostic_decision(
        {
            "target_time_shift_rows": 4,
            "all_nonfinite_masked_rows": 0,
            "all_nonfinite_masked_rows_any": 3,
            "row_filtered_masked_loss_all_finite": True,
            "row_filtered_masked_gradient_all_finite": True,
            "row_filtered_masked_gradient_any_nonzero": True,
            "pre_mask_loss_all_finite": True,
            "pre_mask_gradient_all_finite": True,
            "pre_mask_gradient_any_nonzero": True,
            "masked_loss_any_nonfinite": True,
        }
    )

    assert decision["route"] == "TEST_ROW_FILTERED_TIME_SHIFT_DISTANCE_REPAIR"


def test_decision_routes_to_pass_when_row_filtered_repair_makes_masked_loss_finite() -> None:
    decision = diagnostic_decision(
        {
            "target_time_shift_rows": 4,
            "all_nonfinite_masked_rows": 0,
            "all_nonfinite_masked_rows_any": 3,
            "row_filtered_masked_loss_all_finite": True,
            "row_filtered_masked_gradient_all_finite": True,
            "row_filtered_masked_gradient_any_nonzero": True,
            "masked_loss_any_nonfinite": False,
            "masked_loss_all_finite": True,
        }
    )

    assert decision["route"] == "PASS_ROW_FILTERED_TIME_SHIFT_DISTANCE_REPAIR"


def test_decision_mutates_when_slice_has_no_target_rows() -> None:
    decision = diagnostic_decision(
        {
            "target_time_shift_rows": 0,
            "all_nonfinite_masked_rows": 0,
            "pre_mask_loss_all_finite": False,
            "pre_mask_gradient_all_finite": False,
            "pre_mask_gradient_any_nonzero": False,
            "masked_loss_any_nonfinite": False,
        }
    )

    assert decision["route"] == "MUTATE_TIME_SHIFT_DISTANCE_DIAGNOSTIC"


def test_decision_kills_when_premask_candidate_is_not_finite() -> None:
    decision = diagnostic_decision(
        {
            "target_time_shift_rows": 4,
            "all_nonfinite_masked_rows": 4,
            "pre_mask_loss_all_finite": False,
            "pre_mask_gradient_all_finite": False,
            "pre_mask_gradient_any_nonzero": False,
            "masked_loss_any_nonfinite": True,
        }
    )

    assert decision["route"] == "KILL_TIME_SHIFT_DISTANCE_DISTANCE_LOSS"


def test_writers_emit_report_and_summary(tmp_path: Path) -> None:
    summary = {
        "decision": {"route": "TEST_PREMASK_TIME_SHIFT_DISTANCE_REPAIR", "reason": "ok", "next_step": "repair"},
        "aggregate": {
            "target_time_shift_rows": 2,
            "all_nonfinite_masked_rows": 2,
            "all_nonfinite_masked_rows_any": 2,
            "total_rows": 2,
            "valid_rows": 2,
            "all_nonfinite_masked_row_share": 1.0,
            "all_nonfinite_masked_any_row_share": 1.0,
            "all_nonfinite_masked_valid_rows": 2,
            "all_nonfinite_masked_valid_non_target_rows": 0,
            "all_nonfinite_masked_invalid_rows": 0,
            "finite_masked_candidate_share": 0.0,
            "finite_pre_mask_candidate_share": 1.0,
            "gold_masked_finite_share": 0.0,
            "masked_loss_values": [float("nan")],
            "masked_loss_all_finite": False,
            "pre_mask_loss_values": [0.01],
            "skip_masked_loss_values": [0.0],
            "row_filtered_masked_loss_values": [0.02],
            "pre_mask_gradient_abs_sums": [0.1],
            "pre_mask_gradient_all_finite": True,
            "pre_mask_gradient_any_nonzero": True,
            "row_filtered_masked_gradient_abs_sums": [0.2],
            "row_filtered_masked_gradient_all_finite": True,
            "row_filtered_masked_gradient_any_nonzero": True,
        },
        "examples": [],
        "interpretation": "diagnostic",
    }
    summary_path = tmp_path / "summary.json"
    report_path = tmp_path / "report.md"

    write_summary_json(summary, summary_path)
    write_report(summary, report_path)

    assert json.loads(summary_path.read_text(encoding="utf-8"))["decision"]["route"] == "TEST_PREMASK_TIME_SHIFT_DISTANCE_REPAIR"
    markdown = report_markdown(summary)
    assert "Route: `TEST_PREMASK_TIME_SHIFT_DISTANCE_REPAIR`" in markdown
    assert "target time-shift rows" in report_path.read_text(encoding="utf-8")
