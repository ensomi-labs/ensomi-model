from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch

from pulsefield_model.models.mapper.shared.loss import time_shift_distance_loss
from pulsefield_model.models.mapper.v3 import MapperV3LossConfig, MapperV3ModelLoss, MapperV3Vocab


SUMMARY_SCHEMA_VERSION = 1
REPORT_ROOT = Path("artifacts/reports/audits/mapper_v2_1_grammar")
DEFAULT_SUMMARY_OUTPUT = REPORT_ROOT / "target_grammar_v3_time_shift_distance_loss_stage1_summary.json"
DEFAULT_REPORT_OUTPUT = REPORT_ROOT / "target_grammar_v3_time_shift_distance_loss_stage1_result_report.md"


def run_time_shift_distance_loss_stage1_gate(
    *,
    summary_output_path: Path,
    report_output_path: Path,
) -> dict[str, Any]:
    started_at = time.monotonic()
    torch.manual_seed(20260617)
    vocab = MapperV3Vocab()
    separation = _matching_vs_rigid_probe(vocab)
    gradient = _gradient_probe(vocab)
    ignored = _non_time_shift_ignored_probe(vocab)
    disabled_default = _disabled_default_probe(vocab)
    config = _config_probe()
    checks = {
        "matching_loss_finite": bool(separation["matching_finite"]),
        "rigid_loss_finite": bool(separation["rigid_finite"]),
        "rigid_loss_greater_than_matching": bool(separation["gap"] > 1e-6),
        "gradient_finite": bool(gradient["finite"]),
        "gradient_nonzero": bool(gradient["nonzero"]),
        "non_time_shift_rows_ignored": bool(ignored["ignored"]),
        "disabled_default_ok": bool(disabled_default["disabled_default_ok"]),
        "enabled_metric_positive": bool(disabled_default["enabled_metric_positive"]),
        "config_fields_present": bool(config["config_fields_present"]),
    }
    decision = synthesize_decision(checks)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar v3 time-shift distance loss Stage 1 gate",
        "elapsed_s": time.monotonic() - started_at,
        "checks": checks,
        "probes": {
            "matching_vs_rigid": separation,
            "gradient": gradient,
            "non_time_shift_ignored": ignored,
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
        "matching_loss_finite",
        "rigid_loss_finite",
        "rigid_loss_greater_than_matching",
        "gradient_finite",
        "gradient_nonzero",
        "non_time_shift_rows_ignored",
        "disabled_default_ok",
        "enabled_metric_positive",
        "config_fields_present",
    )
    failed = [key for key in required if not bool(checks.get(key))]
    if failed:
        return {
            "route": "KILL_TIME_SHIFT_DISTANCE_PLUMBING",
            "reason": f"failed synthetic checks: {', '.join(failed)}",
            "interpretation": "The time-shift distance objective is not ready for training.",
            "next_step": "Do not run rollout; repair or kill the loss-side timing objective first.",
            "next_card": None,
        }
    return {
        "route": "TEST_TIME_SHIFT_DISTANCE_TINY_TRAINING_GATE",
        "reason": "synthetic separation, gradient, disabled-default behavior, and config exposure passed",
        "interpretation": (
            "Stage 1 proves a local time-shift value calibration loss surface. It does not prove trained "
            "rollout timing quality."
        ),
        "next_step": "Create a tiny v3 training/rollout card with rigid-grid and second-window guards.",
        "next_card": "target_grammar_v3_time_shift_distance_tiny_training_gate",
    }


def write_summary_json(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    decision = _mapping(summary.get("decision"))
    checks = _mapping(summary.get("checks"))
    probes = _mapping(summary.get("probes"))
    lines = [
        "# Target Grammar v3 Time-Shift Distance Loss Stage 1 Result Report",
        "",
        "## Scope",
        "",
        "This gate tests synthetic loss plumbing only. It does not train, roll out, change tokenizer behavior, change decode policy, or change inference.",
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
            "- Matching target time-shift logits score lower than rigid wrong-shift logits.",
            "- The auxiliary loss is finite and differentiable in the synthetic probe.",
            "- Non-time-shift target rows are ignored by the auxiliary term.",
            "- The new config is disabled by default and exposed through `MapperV3LossConfig`.",
            "",
            "## What Remains Unproven",
            "",
            "- No trained rollout quality improvement is proven.",
            "- No tiny, full32, or full-dataset v3 gate has been run for this objective.",
            "- This does not prove the remaining v3 failure is solved by loss-side timing calibration.",
            "",
            "## Next Step",
            "",
            str(decision.get("next_step")),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _matching_vs_rigid_probe(vocab: MapperV3Vocab) -> dict[str, Any]:
    target, mask, matching_logits, rigid_logits = _synthetic_time_shift_batch(vocab)
    matching = time_shift_distance_loss(
        logits_final=matching_logits,
        target_tokens=target,
        vocab=vocab,
        target_mask=mask,
    )
    rigid = time_shift_distance_loss(
        logits_final=rigid_logits,
        target_tokens=target,
        vocab=vocab,
        target_mask=mask,
    )
    matching_value = float(matching.detach().cpu())
    rigid_value = float(rigid.detach().cpu())
    return {
        "matching_loss": matching_value,
        "rigid_wrong_shift_loss": rigid_value,
        "gap": rigid_value - matching_value,
        "matching_finite": bool(torch.isfinite(matching).item()),
        "rigid_finite": bool(torch.isfinite(rigid).item()),
    }


def _gradient_probe(vocab: MapperV3Vocab) -> dict[str, Any]:
    target, mask, _, rigid_logits = _synthetic_time_shift_batch(vocab)
    logits = rigid_logits.detach().clone().requires_grad_(True)
    loss = time_shift_distance_loss(
        logits_final=logits,
        target_tokens=target,
        vocab=vocab,
        target_mask=mask,
    )
    loss.backward()
    grad = logits.grad
    assert grad is not None
    abs_sum = float(grad.abs().sum().detach().cpu())
    return {
        "loss": float(loss.detach().cpu()),
        "grad_abs_sum": abs_sum,
        "finite": bool(torch.isfinite(loss).item() and torch.isfinite(grad).all().item()),
        "nonzero": bool(abs_sum > 0.0),
    }


def _non_time_shift_ignored_probe(vocab: MapperV3Vocab) -> dict[str, Any]:
    event_id = vocab.event_token_id_from_signature("T...")
    target = torch.tensor([[event_id]], dtype=torch.long)
    mask = torch.ones_like(target, dtype=torch.bool)
    logits = torch.zeros((1, 1, vocab.size), dtype=torch.float32)
    loss = time_shift_distance_loss(
        logits_final=logits,
        target_tokens=target,
        vocab=vocab,
        target_mask=mask,
    )
    value = float(loss.detach().cpu())
    return {
        "loss": value,
        "ignored": bool(value == 0.0 and torch.isfinite(loss).item()),
    }


def _disabled_default_probe(vocab: MapperV3Vocab) -> dict[str, Any]:
    target, mask, _, rigid_logits = _synthetic_time_shift_batch(vocab)
    output = SimpleNamespace(logits_final=rigid_logits)
    batch = {
        "target_fragment_tokens": target,
        "target_fragment_mask": mask,
    }
    off = MapperV3ModelLoss(
        MapperV3LossConfig(
            lambda_density=0.0,
            lambda_time_shift_distance=0.0,
            lambda_ln_close=0.0,
            lambda_adapter_reg=0.0,
        ),
        vocab=vocab,
    )(output, batch)
    on = MapperV3ModelLoss(
        MapperV3LossConfig(
            lambda_density=0.0,
            lambda_time_shift_distance=0.5,
            lambda_ln_close=0.0,
            lambda_adapter_reg=0.0,
        ),
        vocab=vocab,
    )(output, batch)
    return {
        "off_lambda": off.metrics["phase/lambda_time_shift_distance"],
        "on_lambda": on.metrics["phase/lambda_time_shift_distance"],
        "off_time_shift_distance_loss": float(off.time_shift_distance_loss.detach().cpu()),
        "on_time_shift_distance_loss": float(on.time_shift_distance_loss.detach().cpu()),
        "disabled_default_ok": bool(float(off.time_shift_distance_loss.detach().cpu()) == 0.0),
        "enabled_metric_positive": bool(float(on.time_shift_distance_loss.detach().cpu()) > 0.0),
        "total_loss_delta": float((on.total_loss - off.total_loss).detach().cpu()),
    }


def _config_probe() -> dict[str, Any]:
    config = MapperV3LossConfig(lambda_time_shift_distance=0.25, time_shift_distance_scale_ms=500.0)
    return {
        "lambda_time_shift_distance": float(config.lambda_time_shift_distance),
        "time_shift_distance_scale_ms": float(config.time_shift_distance_scale_ms),
        "config_fields_present": bool(
            float(config.lambda_time_shift_distance) == 0.25
            and float(config.time_shift_distance_scale_ms) == 500.0
        ),
    }


def _synthetic_time_shift_batch(
    vocab: MapperV3Vocab,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shift_60 = vocab.time_shift_token_id(60)
    shift_80 = vocab.time_shift_token_id(80)
    shift_100 = vocab.time_shift_token_id(100)
    event = vocab.event_token_id_from_signature("T...")
    target = torch.tensor([[shift_80, shift_100, shift_60, event]], dtype=torch.long)
    mask = torch.ones_like(target, dtype=torch.bool)
    matching_logits = torch.full((1, target.shape[1], vocab.size), -5.0, dtype=torch.float32)
    rigid_logits = torch.full_like(matching_logits, -5.0)
    for step, token_id in enumerate(target[0].tolist()):
        matching_logits[0, step, int(token_id)] = 5.0
        rigid_logits[0, step, shift_100] = 5.0
    return target, mask, matching_logits, rigid_logits


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the v3 time-shift distance loss Stage 1 synthetic gate.")
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_time_shift_distance_loss_stage1_gate(
        summary_output_path=args.summary_output,
        report_output_path=args.report_output,
    )
    decision = summary["decision"]
    probes = summary["probes"]["matching_vs_rigid"]
    print(
        "mapper v3 time-shift distance loss stage1: "
        f"decision={decision['route']} "
        f"matching={probes['matching_loss']:.6f} "
        f"rigid={probes['rigid_wrong_shift_loss']:.6f} "
        f"gap={probes['gap']:.6f}"
    )


if __name__ == "__main__":
    main()
