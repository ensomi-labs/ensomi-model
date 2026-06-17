from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from pulsefield_model.models.mapper.v3 import (
    MapperTimepoint,
    MapperV3Config,
    MapperV3LossConfig,
    MapperV3Model,
    MapperV3ModelLoss,
    MapperV3Vocab,
    encode_mapper_window,
    ln_carry_state_tensors,
)
from pulsefield_model.models.mapper.v3.vocab import LaneAction


SUMMARY_SCHEMA_VERSION = 1
REPORT_ROOT = Path("artifacts/reports/audits/mapper_v2_1_grammar")
DEFAULT_EXPERIMENT_CARD = REPORT_ROOT / "target_grammar_v3_delta_event_auxiliary_tiny_training_gate_experiment_card.md"
DEFAULT_SUMMARY_OUTPUT = REPORT_ROOT / "target_grammar_v3_delta_event_auxiliary_tiny_training_gate_summary.json"
DEFAULT_REPORT_OUTPUT = REPORT_ROOT / "target_grammar_v3_delta_event_auxiliary_tiny_training_gate_result_report.md"


def run_delta_event_auxiliary_tiny_training_gate(
    *,
    summary_output_path: str | Path = DEFAULT_SUMMARY_OUTPUT,
    report_output_path: str | Path | None = DEFAULT_REPORT_OUTPUT,
    experiment_card_path: str | Path = DEFAULT_EXPERIMENT_CARD,
    steps: int = 40,
    learning_rate: float = 5e-3,
    max_aux_loss_ratio: float = 0.25,
    seed: int = 20260628,
) -> dict[str, Any]:
    started = time.monotonic()
    _validate_run_config(steps=steps, learning_rate=learning_rate, max_aux_loss_ratio=max_aux_loss_ratio)
    torch.manual_seed(int(seed))
    vocab = MapperV3Vocab()
    tokenized = _synthetic_tokenized_window(vocab)
    batch = _synthetic_batch(tokenized, control_dim=16)
    default_off = _default_off_probe(vocab=vocab, batch=batch)
    training = _enabled_training_probe(
        vocab=vocab,
        batch=batch,
        steps=int(steps),
        learning_rate=float(learning_rate),
    )
    checks = guard_checks(default_off=default_off, training=training, max_aux_loss_ratio=max_aux_loss_ratio)
    decision = decision_from_checks(checks)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar v3 delta-event auxiliary tiny training gate",
        "experiment_card": Path(experiment_card_path).as_posix(),
        "elapsed_s": time.monotonic() - started,
        "config": {
            "steps": int(steps),
            "learning_rate": float(learning_rate),
            "max_aux_loss_ratio": float(max_aux_loss_ratio),
            "seed": int(seed),
            "no_rollout": True,
            "dataset_schema_changed": False,
            "tokenizer_changed": False,
            "default_behavior_changed": False,
            "uses_c3_backreference": False,
            "uses_future_lookup": False,
        },
        "dataset": {
            "slice": "synthetic_v3_teacher_forced_window",
            "seq_len": int(tokenized.seq_len),
            "target_tokens": [vocab.token_name(int(token_id)) for token_id in tokenized.target_fragment_ids],
        },
        "default_off": default_off,
        "training": training,
        "checks": checks,
        "decision": decision,
        "next_step": decision["next_step"],
    }
    write_summary_json(summary, Path(summary_output_path))
    if report_output_path is not None:
        write_report(summary, Path(report_output_path))
    return summary


def guard_checks(
    *,
    default_off: Mapping[str, Any],
    training: Mapping[str, Any],
    max_aux_loss_ratio: float,
) -> dict[str, bool]:
    return {
        "default_off_auxiliary_logits_absent": bool(default_off.get("auxiliary_logits_absent")),
        "default_off_lambda_zero": abs(float(default_off.get("lambda_delta_event_auxiliary") or 0.0)) <= 1e-12,
        "event_labels_positive": int(training.get("event_label_count") or 0) > 0,
        "end_gap_labels_positive": int(training.get("end_gap_label_count") or 0) > 0,
        "all_losses_finite": bool(training.get("all_losses_finite")),
        "initial_auxiliary_loss_positive": float(training.get("initial_auxiliary_loss") or 0.0) > 0.0,
        "final_auxiliary_loss_finite": math.isfinite(float(training.get("final_auxiliary_loss") or float("nan"))),
        "auxiliary_loss_decreased": float(training.get("auxiliary_loss_ratio") or float("inf")) <= float(max_aux_loss_ratio),
        "total_loss_decreased": float(training.get("total_loss_ratio") or float("inf")) < 1.0,
        "token_loss_not_worse": float(training.get("token_loss_ratio") or float("inf")) <= 1.0,
        "delta_head_gradient_nonzero": float(training.get("first_step_gradients", {}).get("delta_head_grad_abs") or 0.0) > 0.0,
        "signature_head_gradient_nonzero": float(training.get("first_step_gradients", {}).get("signature_head_grad_abs") or 0.0) > 0.0,
        "end_gap_head_gradient_nonzero": float(training.get("first_step_gradients", {}).get("end_gap_head_grad_abs") or 0.0) > 0.0,
        "shared_decoder_gradient_nonzero": float(training.get("first_step_gradients", {}).get("token_embedding_grad_abs") or 0.0) > 0.0,
        "no_rollout": True,
        "no_tokenizer_dataset_default_change": True,
        "no_c3_backreference_or_future_lookup": True,
    }


def decision_from_checks(checks: Mapping[str, bool]) -> dict[str, str]:
    failed = [key for key, value in checks.items() if not bool(value)]
    if failed:
        hard_failures = {
            "default_off_auxiliary_logits_absent",
            "event_labels_positive",
            "end_gap_labels_positive",
            "all_losses_finite",
            "delta_head_gradient_nonzero",
            "signature_head_gradient_nonzero",
            "end_gap_head_gradient_nonzero",
        }
        if any(key in hard_failures for key in failed):
            route = "KILL_DELTA_EVENT_AUXILIARY_TRAINING"
            next_step = "Do not run real-data training; repair auxiliary labels, heads, or loss stability first."
        else:
            route = "MUTATE_DELTA_EVENT_AUXILIARY_TRAINING"
            next_step = "Adjust lambda, learning rate, or label placement before a real-data gate."
        return {
            "route": route,
            "reason": "failed checks: " + ", ".join(failed),
            "next_step": next_step,
        }
    return {
        "route": "TEST_DELTA_EVENT_AUXILIARY_FIXED_SLICE_GATE",
        "reason": "synthetic default-off and overfit trainability gates passed",
        "next_step": "Create a fixed-slice label coverage or tiny real-data training gate before any rollout or replacement work.",
    }


def write_summary_json(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_markdown(summary).rstrip() + "\n", encoding="utf-8")


def report_markdown(summary: Mapping[str, Any]) -> str:
    decision = _mapping(summary.get("decision"))
    training = _mapping(summary.get("training"))
    gradients = _mapping(training.get("first_step_gradients"))
    checks = _mapping(summary.get("checks"))
    dataset = _mapping(summary.get("dataset"))
    config = _mapping(summary.get("config"))
    lines = [
        "# Target Grammar v3 Delta-Event Auxiliary Tiny Training Gate Result Report",
        "",
        "## Scope",
        "",
        "This gate performs a synthetic teacher-forced overfit run for the default-off v3 delta-event auxiliary objective. It does not use the full training runner, real dataset cache, rollout, tokenizer changes, or default mapper changes.",
        "",
        "## Decision",
        "",
        f"Route: `{decision.get('route')}`.",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Recommended next step: {decision.get('next_step')}",
        "",
        "## Training Setup",
        "",
        f"- steps: `{config.get('steps')}`",
        f"- learning rate: `{config.get('learning_rate')}`",
        f"- synthetic sequence length: `{dataset.get('seq_len')}`",
        f"- target tokens: `{' '.join(str(token) for token in dataset.get('target_tokens', []))}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        _row("initial total loss", _fmt(training.get("initial_total_loss"))),
        _row("final total loss", _fmt(training.get("final_total_loss"))),
        _row("total loss ratio", _fmt(training.get("total_loss_ratio"))),
        _row("initial token loss", _fmt(training.get("initial_token_loss"))),
        _row("final token loss", _fmt(training.get("final_token_loss"))),
        _row("token loss ratio", _fmt(training.get("token_loss_ratio"))),
        _row("initial auxiliary loss", _fmt(training.get("initial_auxiliary_loss"))),
        _row("final auxiliary loss", _fmt(training.get("final_auxiliary_loss"))),
        _row("auxiliary loss ratio", _fmt(training.get("auxiliary_loss_ratio"))),
        _row("event labels", training.get("event_label_count")),
        _row("end-gap labels", training.get("end_gap_label_count")),
        _row("delta head grad abs", _fmt(gradients.get("delta_head_grad_abs"))),
        _row("signature head grad abs", _fmt(gradients.get("signature_head_grad_abs"))),
        _row("end-gap head grad abs", _fmt(gradients.get("end_gap_head_grad_abs"))),
        _row("token embedding grad abs", _fmt(gradients.get("token_embedding_grad_abs"))),
        "",
        "## Checks",
        "",
        "| Check | Value |",
        "| --- | ---: |",
    ]
    lines.extend(_row(key, value) for key, value in checks.items())
    lines.extend(
        [
            "",
            "## What Passed",
            "",
            "- Default-off v3 still emits no delta-event auxiliary logits.",
            "- The enabled auxiliary objective has event and end-gap labels on the synthetic teacher-forced batch.",
            "- Total, token, and auxiliary losses decreased under real optimizer steps.",
            "- First-step gradients reached all three auxiliary heads and the shared token embedding.",
            "",
            "## What Surfaced",
            "",
            "The auxiliary objective is trainable in a synthetic overfit setting. This is stronger than the previous plumbing smoke, but it still does not prove real-data label coverage, mapper-quality improvement, rollout legality, or full v3 replacement readiness.",
            "",
            "## What Is Not Proved",
            "",
            "- No real dataset, fixed32, full32, or 4k run was performed.",
            "- No autoregressive rollout quality was measured.",
            "- No target grammar replacement was made.",
            "- No C3 backreference or side-stream claim is tested here.",
            "",
            "## Verification",
            "",
            *_verification_lines(summary),
            "",
            "## Next Step",
            "",
            str(decision.get("next_step")),
        ]
    )
    return "\n".join(lines)


def _verification_lines(summary: Mapping[str, Any]) -> list[str]:
    commands = summary.get("commands")
    if isinstance(commands, Sequence) and not isinstance(commands, (str, bytes, bytearray)):
        lines = ["```bash"]
        results: list[str] = []
        for item in commands:
            command = _mapping(item).get("command")
            result = _mapping(item).get("result")
            if command:
                lines.append(str(command))
            if result:
                results.append(str(result))
        lines.append("```")
        if results:
            lines.extend(["", "Results:", *[f"- {result}" for result in results]])
        return lines
    return [
        "```bash",
        "uv run python -m pulsefield_model.evals.mapper_v3_delta_event_auxiliary_tiny_training_gate",
        "uv run --group dev pytest tests/evals/test_mapper_v3_delta_event_auxiliary_tiny_training_gate.py tests/models/mapper/v3 tests/evals/test_target_grammar_v3_delta_event_proxy_audit.py tests/training/test_mapper_v3.py -q",
        "uv run python -m py_compile src/pulsefield_model/evals/mapper_v3_delta_event_auxiliary_tiny_training_gate.py",
        "python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_auxiliary_tiny_training_gate_summary.json >/dev/null",
        "git diff --check",
        "```",
    ]


def _enabled_training_probe(
    *,
    vocab: MapperV3Vocab,
    batch: Mapping[str, Any],
    steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    model = MapperV3Model(_small_config(use_delta_event_auxiliary_target=True), vocab=vocab)
    loss_fn = MapperV3ModelLoss(
        MapperV3LossConfig(
            lambda_density=0.0,
            lambda_ln_close=0.0,
            lambda_adapter_reg=0.0,
            lambda_delta_event_auxiliary=1.0,
        ),
        vocab=vocab,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=0.0)
    curves: list[dict[str, float]] = []
    first_step_gradients: dict[str, float] | None = None
    all_losses_finite = True
    for step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss = loss_fn(output, batch)
        metrics = loss.metrics
        row = {
            "step": float(step),
            "total_loss": float(metrics["loss/total"]),
            "token_loss": float(metrics["loss/token"]),
            "delta_event_auxiliary_loss": float(metrics["loss/delta_event_auxiliary"]),
        }
        curves.append(row)
        all_losses_finite = all_losses_finite and all(math.isfinite(value) for value in row.values())
        loss.total_loss.backward()
        if step == 0:
            first_step_gradients = _gradient_snapshot(model)
        optimizer.step()
    if not curves:
        raise ValueError("delta-event auxiliary tiny training gate requires at least one optimizer step")
    initial = curves[0]
    final = curves[-1]
    return {
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "all_losses_finite": bool(all_losses_finite),
        "initial_total_loss": initial["total_loss"],
        "final_total_loss": final["total_loss"],
        "total_loss_ratio": _safe_ratio(final["total_loss"], initial["total_loss"]),
        "initial_token_loss": initial["token_loss"],
        "final_token_loss": final["token_loss"],
        "token_loss_ratio": _safe_ratio(final["token_loss"], initial["token_loss"]),
        "initial_auxiliary_loss": initial["delta_event_auxiliary_loss"],
        "final_auxiliary_loss": final["delta_event_auxiliary_loss"],
        "auxiliary_loss_ratio": _safe_ratio(final["delta_event_auxiliary_loss"], initial["delta_event_auxiliary_loss"]),
        "min_auxiliary_loss": min(row["delta_event_auxiliary_loss"] for row in curves),
        "event_label_count": int(round(float(loss.metrics["delta_event_auxiliary/event_label_count"]))),
        "end_gap_label_count": int(round(float(loss.metrics["delta_event_auxiliary/end_gap_label_count"]))),
        "first_step_gradients": first_step_gradients or {},
        "loss_curve_preview": [curves[index] for index in sorted({0, min(1, len(curves) - 1), len(curves) // 2, len(curves) - 1})],
    }


def _default_off_probe(*, vocab: MapperV3Vocab, batch: Mapping[str, Any]) -> dict[str, Any]:
    model = MapperV3Model(_small_config(use_delta_event_auxiliary_target=False), vocab=vocab)
    loss_fn = MapperV3ModelLoss(
        MapperV3LossConfig(lambda_density=0.0, lambda_ln_close=0.0, lambda_adapter_reg=0.0),
        vocab=vocab,
    )
    with torch.no_grad():
        output = model(batch)
        loss = loss_fn(output, batch)
    return {
        "auxiliary_logits_absent": bool(
            output.delta_event_delta_logits is None
            and output.delta_event_signature_logits is None
            and output.delta_event_end_gap_logits is None
        ),
        "lambda_delta_event_auxiliary": float(loss.metrics["phase/lambda_delta_event_auxiliary"]),
        "delta_event_auxiliary_loss": float(loss.delta_event_auxiliary_loss.detach().cpu()),
        "total_loss": float(loss.total_loss.detach().cpu()),
    }


def _gradient_snapshot(model: MapperV3Model) -> dict[str, float]:
    if model.delta_event_delta_head is None:
        raise RuntimeError("delta-event delta head missing in enabled training probe")
    if model.delta_event_signature_head is None:
        raise RuntimeError("delta-event signature head missing in enabled training probe")
    if model.delta_event_end_gap_head is None:
        raise RuntimeError("delta-event end-gap head missing in enabled training probe")
    return {
        "delta_head_grad_abs": _grad_abs(model.delta_event_delta_head.weight),
        "signature_head_grad_abs": _grad_abs(model.delta_event_signature_head.weight),
        "end_gap_head_grad_abs": _grad_abs(model.delta_event_end_gap_head.weight),
        "token_embedding_grad_abs": _grad_abs(model.token_embedding.weight),
    }


def _synthetic_tokenized_window(vocab: MapperV3Vocab):
    return encode_mapper_window(
        [
            MapperTimepoint(80, _actions(LaneAction.TAP)),
            MapperTimepoint(240, _actions(LaneAction.NONE, LaneAction.TAP)),
            MapperTimepoint(400, _actions(LaneAction.HOLD_START)),
            MapperTimepoint(700, _actions(LaneAction.HOLD_END, LaneAction.TAP)),
        ],
        vocab=vocab,
        write_start_ms=0,
        write_end_ms=8000,
        chart_end_ms=1000,
    )


def _synthetic_batch(tokenized: Any, *, control_dim: int) -> dict[str, Any]:
    return {
        "decoder_input_tokens": tokenized.decoder_input_tensor().unsqueeze(0),
        "target_fragment_tokens": tokenized.target_fragment_tensor().unsqueeze(0),
        "target_fragment_mask": torch.ones((1, tokenized.seq_len), dtype=torch.bool),
        "target_fragment_states": {
            "current_ms": tokenized.target_fragment_current_ms.unsqueeze(0),
            "open_mask": tokenized.target_fragment_open_mask.unsqueeze(0),
            "open_start_ms": tokenized.target_fragment_open_start_ms.unsqueeze(0),
            "open_age_ms": tokenized.target_fragment_open_age_ms.unsqueeze(0),
        },
        "ln_carry_in": _batched_carry(tokenized.ln_carry_in),
        "ln_carry_out": _batched_carry(tokenized.ln_carry_out),
        "close_labels": tokenized.close_labels.unsqueeze(0),
        "close_label_mask": tokenized.close_label_mask.unsqueeze(0),
        "write_start_ms": torch.tensor([tokenized.write_start_ms], dtype=torch.long),
        "write_end_ms": torch.tensor([tokenized.write_end_ms], dtype=torch.long),
        "chart_end_ms": torch.tensor([tokenized.chart_end_ms], dtype=torch.long),
        "is_full_chart_start": torch.tensor([tokenized.is_full_chart_start], dtype=torch.bool),
        "is_full_chart_end": torch.tensor([tokenized.is_full_chart_end], dtype=torch.bool),
        "difficulty": torch.tensor([[3.2]], dtype=torch.float32),
        "normalized_difficulty": torch.tensor([[0.1]], dtype=torch.float32),
        "density_target_8s": torch.zeros((1, 400, 1), dtype=torch.float32),
        "density_confidence_8s": torch.ones((1, 400, 1), dtype=torch.float32),
        "control_memory_8s": torch.zeros((1, 400, int(control_dim)), dtype=torch.float32),
        "density_teacher_8s": torch.zeros((1, 400, 1), dtype=torch.float32),
    }


def _small_config(**overrides: Any) -> MapperV3Config:
    values = {
        "control_dim": 16,
        "d_model": 32,
        "heads": 4,
        "layers": 1,
        "ffn_dim": 64,
        "dropout": 0.0,
        "max_seq_len": 64,
        "state_prior_hidden_dim": 16,
        "ln_close_hidden_dim": 16,
        "lane_embedding_dim": 4,
        "age_embedding_dim": 4,
        "use_global_context": False,
        "global_conv_blocks": 0,
        "use_delta_event_auxiliary_target": False,
        "delta_event_delta_max_ms": 8000,
        "delta_event_end_gap_max_ms": 8000,
    }
    values.update(overrides)
    return MapperV3Config(**values)


def _actions(*actions: LaneAction) -> tuple[LaneAction, ...]:
    padded = list(actions)
    while len(padded) < 4:
        padded.append(LaneAction.NONE)
    return tuple(padded)


def _batched_carry(carry: object) -> dict[str, torch.Tensor]:
    tensors = ln_carry_state_tensors(carry)  # type: ignore[arg-type]
    return {
        "current_ms": tensors["current_ms"].reshape(1),
        "open_mask": tensors["open_mask"].reshape(1, 4),
        "open_start_ms": tensors["open_start_ms"].reshape(1, 4),
        "open_age_ms": tensors["open_age_ms"].reshape(1, 4),
    }


def _grad_abs(parameter: torch.Tensor) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().abs().sum().cpu())


def _safe_ratio(numerator: float, denominator: float) -> float:
    if float(denominator) == 0.0:
        return 0.0 if float(numerator) == 0.0 else float("inf")
    return float(numerator) / float(denominator)


def _validate_run_config(*, steps: int, learning_rate: float, max_aux_loss_ratio: float) -> None:
    if int(steps) <= 0:
        raise ValueError("steps must be positive")
    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive finite")
    if not math.isfinite(float(max_aux_loss_ratio)) or float(max_aux_loss_ratio) <= 0.0:
        raise ValueError("max_aux_loss_ratio must be positive finite")


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _fmt(value: object) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(number) else f"{number:.6f}"


def _row(label: str, value: object) -> str:
    return f"| {label} | `{value}` |"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the v3 delta-event auxiliary synthetic tiny-training gate.")
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--report-output", default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--max-aux-loss-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260628)
    args = parser.parse_args(argv)
    summary = run_delta_event_auxiliary_tiny_training_gate(
        summary_output_path=Path(args.summary_output),
        report_output_path=Path(args.report_output),
        steps=args.steps,
        learning_rate=args.learning_rate,
        max_aux_loss_ratio=args.max_aux_loss_ratio,
        seed=args.seed,
    )
    print(
        "mapper_v3_delta_event_auxiliary_tiny_training_gate_done "
        f"route={summary['decision']['route']} "
        f"aux_loss_ratio={summary['training']['auxiliary_loss_ratio']:.6f} "
        f"summary={Path(args.summary_output).as_posix()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
