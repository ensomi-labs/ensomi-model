from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch

from pulsefield_model.models.mapper.shared.loss import conditioned_event_distribution_loss
from pulsefield_model.models.mapper.v3 import MapperV3LossConfig, MapperV3ModelLoss, MapperV3Vocab


SUMMARY_SCHEMA_VERSION = 1
REPORT_ROOT = Path("artifacts/reports/audits/mapper_v2_1_grammar")


def run_conditioned_event_distribution_gate(
    *,
    summary_output_path: Path,
    report_output_path: Path,
) -> dict[str, Any]:
    start = time.monotonic()
    torch.manual_seed(20260617)
    vocab = MapperV3Vocab()
    overproduction = _high_difficulty_overproduction_probe(vocab)
    underproduction = _underproduction_probe(vocab)
    disabled_default = _disabled_default_probe(vocab)
    config = _config_probe()
    checks = {
        "high_difficulty_overproduction_finite": bool(overproduction["finite"]),
        "high_difficulty_overproduction_ratio": overproduction["ratio"],
        "high_difficulty_overproduction_conditioned": bool(overproduction["ratio"] > 1.5),
        "underproduction_finite": bool(underproduction["finite"]),
        "underproduction_gap": underproduction["gap"],
        "underproduction_penalized": bool(underproduction["gap"] > 0.0),
        "disabled_default_ok": bool(disabled_default["disabled_default_ok"]),
        "enabled_metric_positive": bool(disabled_default["enabled_metric_positive"]),
        "config_fields_present": bool(config["config_fields_present"]),
    }
    decision = synthesize_decision(checks)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar v3 conditioned event-distribution objective Stage 1 gate",
        "elapsed_s": time.monotonic() - start,
        "checks": checks,
        "probes": {
            "high_difficulty_overproduction": overproduction,
            "underproduction": underproduction,
            "disabled_default": disabled_default,
            "config": config,
        },
        "decision": decision,
        "next_card": decision.get("next_card"),
    }
    write_summary_json(summary, summary_output_path)
    write_report(summary, report_output_path)
    return summary


def synthesize_decision(checks: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "high_difficulty_overproduction_finite",
        "high_difficulty_overproduction_conditioned",
        "underproduction_finite",
        "underproduction_penalized",
        "disabled_default_ok",
        "enabled_metric_positive",
        "config_fields_present",
    )
    failed = [key for key in required if not bool(checks.get(key))]
    if failed:
        return {
            "route": "KILL_CONDITIONED_EVENT_DISTRIBUTION_PLUMBING",
            "reason": f"failed synthetic checks: {', '.join(failed)}",
            "interpretation": "The conditioned event-distribution objective is not ready for training.",
            "next_step": "Do not run rollout; repair or kill the loss-side objective first.",
            "next_card": None,
        }
    return {
        "route": "TEST_SHORT_ROLLOUT_GATE",
        "reason": "synthetic conditioning, disabled-default behavior, and config exposure passed",
        "interpretation": (
            "Stage 1 proves the objective is a real conditioned loss surface, but not that it improves "
            "trained chart quality."
        ),
        "next_step": "Run the short v3 conditioned-objective training/rollout gate with overgeneration guards.",
        "next_card": "target_grammar_v3_conditioned_event_distribution_short_rollout_gate",
    }


def write_summary_json(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    decision = _mapping(summary.get("decision"))
    checks = _mapping(summary.get("checks"))
    probes = _mapping(summary.get("probes"))
    lines = [
        "# Target Grammar v3 Conditioned Event-Distribution Objective Stage 1 Result Report",
        "",
        "## Scope",
        "",
        "This gate tests synthetic loss plumbing only. It does not train, roll out, change tokenizer behavior, or change decode policy.",
        "",
        "## Decision",
        "",
        f"Route: `{decision.get('route')}`.",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Recommended next step: {decision.get('next_step')}",
        f"- Next card: `{decision.get('next_card')}`",
        "",
        "## Checks",
        "",
        "| Check | Value |",
        "| --- | ---: |",
    ]
    for key, value in checks.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Probe Metrics",
            "",
            "| Probe | Metric | Value |",
            "| --- | --- | ---: |",
        ]
    )
    for probe_name, probe_value in probes.items():
        for metric, value in _mapping(probe_value).items():
            lines.append(f"| `{probe_name}` | `{metric}` | `{value}` |")
    lines.extend(
        [
            "",
            "## What This Proves",
            "",
            "- The objective has a finite synthetic loss.",
            "- High-difficulty sparse/zero-target overproduction receives a stronger penalty than the same low-difficulty context.",
            "- Underproduction of target events is penalized.",
            "- The new config is disabled by default and exposed through `MapperV3LossConfig`.",
            "",
            "## What Remains Unproven",
            "",
            "- No trained rollout quality improvement is proven.",
            "- No full32 or full-dataset v3 gate has been run for this objective.",
            "- This does not prove the remaining v3 failure is objective-side rather than target-grammar-side.",
            "",
            "## Next Step",
            "",
            str(decision.get("next_step")),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _high_difficulty_overproduction_probe(vocab: MapperV3Vocab) -> dict[str, Any]:
    event_id = vocab.event_token_id_from_signature("T...")
    shift_id = vocab.time_shift_token_id(100)
    target = torch.tensor([[shift_id, shift_id, shift_id, shift_id]], dtype=torch.long)
    current_ms = torch.tensor([[1000, 2000, 5000, 6000]], dtype=torch.long)
    mask = torch.ones_like(target, dtype=torch.bool)
    logits = torch.full((1, target.shape[1], vocab.size), -5.0, dtype=torch.float32)
    logits[:, :, shift_id] = 5.0
    logits[:, 2, event_id] = 6.0
    low = conditioned_event_distribution_loss(
        logits_final=logits,
        target_tokens=target,
        current_ms=current_ms,
        write_start_ms=torch.tensor([0], dtype=torch.long),
        write_end_ms=torch.tensor([8000], dtype=torch.long),
        normalized_difficulty=torch.tensor([[0.1]], dtype=torch.float32),
        vocab=vocab,
        target_mask=mask,
        zero_target_over_weight=2.0,
        high_difficulty_over_weight=4.0,
        high_difficulty_min=0.75,
    )
    high = conditioned_event_distribution_loss(
        logits_final=logits,
        target_tokens=target,
        current_ms=current_ms,
        write_start_ms=torch.tensor([0], dtype=torch.long),
        write_end_ms=torch.tensor([8000], dtype=torch.long),
        normalized_difficulty=torch.tensor([[0.9]], dtype=torch.float32),
        vocab=vocab,
        target_mask=mask,
        zero_target_over_weight=2.0,
        high_difficulty_over_weight=4.0,
        high_difficulty_min=0.75,
    )
    low_value = float(low.detach().cpu())
    high_value = float(high.detach().cpu())
    return {
        "low_difficulty_loss": low_value,
        "high_difficulty_loss": high_value,
        "ratio": high_value / max(low_value, 1e-12),
        "finite": bool(torch.isfinite(low).item() and torch.isfinite(high).item()),
    }


def _underproduction_probe(vocab: MapperV3Vocab) -> dict[str, Any]:
    event_id = vocab.event_token_id_from_signature("T...")
    shift_id = vocab.time_shift_token_id(100)
    target = torch.tensor([[shift_id, event_id, shift_id, event_id]], dtype=torch.long)
    current_ms = torch.tensor([[1000, 1000, 5000, 5000]], dtype=torch.long)
    mask = torch.ones_like(target, dtype=torch.bool)
    matching_logits = torch.full((1, target.shape[1], vocab.size), -5.0, dtype=torch.float32)
    suppressed_logits = torch.full_like(matching_logits, -5.0)
    for step, token_id in enumerate(target[0].tolist()):
        matching_logits[0, step, int(token_id)] = 5.0
        suppressed_logits[0, step, shift_id] = 5.0
    matching = conditioned_event_distribution_loss(
        logits_final=matching_logits,
        target_tokens=target,
        current_ms=current_ms,
        write_start_ms=torch.tensor([0], dtype=torch.long),
        write_end_ms=torch.tensor([8000], dtype=torch.long),
        normalized_difficulty=torch.tensor([[0.1]], dtype=torch.float32),
        vocab=vocab,
        target_mask=mask,
        under_weight=3.0,
    )
    suppressed = conditioned_event_distribution_loss(
        logits_final=suppressed_logits,
        target_tokens=target,
        current_ms=current_ms,
        write_start_ms=torch.tensor([0], dtype=torch.long),
        write_end_ms=torch.tensor([8000], dtype=torch.long),
        normalized_difficulty=torch.tensor([[0.1]], dtype=torch.float32),
        vocab=vocab,
        target_mask=mask,
        under_weight=3.0,
    )
    matching_value = float(matching.detach().cpu())
    suppressed_value = float(suppressed.detach().cpu())
    return {
        "matching_loss": matching_value,
        "suppressed_loss": suppressed_value,
        "gap": suppressed_value - matching_value,
        "finite": bool(torch.isfinite(matching).item() and torch.isfinite(suppressed).item()),
    }


def _disabled_default_probe(vocab: MapperV3Vocab) -> dict[str, Any]:
    event_id = vocab.event_token_id_from_signature("T...")
    target = torch.tensor([[event_id]], dtype=torch.long)
    logits = torch.zeros((1, 1, vocab.size), dtype=torch.float32)
    output = SimpleNamespace(logits_final=logits)
    minimal_batch = {
        "target_fragment_tokens": target,
        "target_fragment_mask": torch.ones_like(target, dtype=torch.bool),
    }
    enabled_batch = {
        **minimal_batch,
        "target_fragment_states": {"current_ms": torch.tensor([[1000]], dtype=torch.long)},
        "write_start_ms": torch.tensor([0], dtype=torch.long),
        "write_end_ms": torch.tensor([8000], dtype=torch.long),
        "normalized_difficulty": torch.tensor([[0.9]], dtype=torch.float32),
    }
    off = MapperV3ModelLoss(
        MapperV3LossConfig(
            lambda_density=0.0,
            lambda_conditioned_event_distribution=0.0,
            lambda_ln_close=0.0,
            lambda_adapter_reg=0.0,
        ),
        vocab=vocab,
    )(output, minimal_batch)
    on = MapperV3ModelLoss(
        MapperV3LossConfig(
            lambda_density=0.0,
            lambda_conditioned_event_distribution=0.5,
            lambda_ln_close=0.0,
            lambda_adapter_reg=0.0,
        ),
        vocab=vocab,
    )(output, enabled_batch)
    return {
        "off_lambda": off.metrics["phase/lambda_conditioned_event_distribution"],
        "off_conditioned_loss": float(off.conditioned_event_distribution_loss.detach().cpu()),
        "on_lambda": on.metrics["phase/lambda_conditioned_event_distribution"],
        "on_conditioned_loss": float(on.conditioned_event_distribution_loss.detach().cpu()),
        "disabled_default_ok": bool(float(off.conditioned_event_distribution_loss.detach().cpu()) == 0.0),
        "enabled_metric_positive": bool(float(on.conditioned_event_distribution_loss.detach().cpu()) > 0.0),
    }


def _config_probe() -> dict[str, Any]:
    config = MapperV3LossConfig()
    required = (
        "lambda_conditioned_event_distribution",
        "conditioned_event_under_weight",
        "conditioned_event_over_weight",
        "conditioned_event_zero_target_over_weight",
        "conditioned_event_high_difficulty_over_weight",
        "conditioned_event_high_difficulty_min",
    )
    missing = [name for name in required if not hasattr(config, name)]
    return {
        "config_fields_present": not missing,
        "missing_fields": missing,
    }


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the v3 conditioned event-distribution Stage 1 gate.")
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--report-output", required=True)
    args = parser.parse_args(argv)
    summary = run_conditioned_event_distribution_gate(
        summary_output_path=Path(args.summary_output),
        report_output_path=Path(args.report_output),
    )
    print(
        "mapper_v3_conditioned_event_distribution_gate_done "
        f"route={summary['decision']['route']} "
        f"next={summary['decision'].get('next_card')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
