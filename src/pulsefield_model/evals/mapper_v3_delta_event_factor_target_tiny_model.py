from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from pulsefield_model.data.mapper_sparse_windows_v3 import MapperV3WindowDataset
from pulsefield_model.evals.target_grammar_v3_delta_event_proxy_audit import delta_event_proxy_from_v3_tokens
from pulsefield_model.models.mapper.v3.vocab import MapperV3Vocab


SUMMARY_SCHEMA_VERSION = 1
REPORT_ROOT = Path("artifacts/reports/audits/mapper_v2_1_grammar")
DEFAULT_INDEX_PATH = Path("artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet")
DEFAULT_CONTROL_TEACHER_CACHE_DIR = Path("artifacts/cache/stage2_mapper_v2_1/control_teacher_d384_l3_stride16_step002000")
DEFAULT_EXPERIMENT_CARD = REPORT_ROOT / "target_grammar_v3_delta_event_factor_target_tiny_model_experiment_card.md"
DEFAULT_SUMMARY_OUTPUT = REPORT_ROOT / "target_grammar_v3_delta_event_factor_target_tiny_model_summary.json"
DEFAULT_REPORT_OUTPUT = REPORT_ROOT / "target_grammar_v3_delta_event_factor_target_tiny_model_result_report.md"
DEFAULT_FULL4K_BIT_PROXY_SUMMARY = REPORT_ROOT / "target_grammar_v3_delta_event_full4k_bit_proxy_summary.json"

EVENT_KIND = 0
END_KIND = 1
BOS_INPUT_KIND = 2
IGNORE_INDEX = -100


@dataclass(frozen=True)
class FactorizedWindow:
    input_kind: tuple[int, ...]
    input_delta: tuple[int, ...]
    input_signature: tuple[int, ...]
    input_end_gap: tuple[int, ...]
    target_kind: tuple[int, ...]
    target_delta: tuple[int, ...]
    target_signature: tuple[int, ...]
    target_end_gap: tuple[int, ...]
    event_count: int
    end_count: int
    reconstruction_mismatches: int

    @property
    def row_count(self) -> int:
        return len(self.target_kind)


class TinyFactorizedDeltaEventModel(nn.Module):
    def __init__(
        self,
        *,
        control_dim: int,
        signature_classes: int,
        delta_classes: int = 801,
        end_gap_classes: int = 801,
        d_model: int = 64,
        heads: int = 4,
        layers: int = 2,
        max_rows: int = 256,
    ) -> None:
        super().__init__()
        self.kind_embedding = nn.Embedding(3, d_model)
        self.delta_embedding = nn.Embedding(delta_classes, d_model)
        self.signature_embedding = nn.Embedding(signature_classes, d_model)
        self.end_gap_embedding = nn.Embedding(end_gap_classes, d_model)
        self.position_embedding = nn.Embedding(max_rows, d_model)
        self.control_projection = nn.Linear(control_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.kind_head = nn.Linear(d_model, 2)
        self.delta_head = nn.Linear(d_model, delta_classes)
        self.signature_head = nn.Linear(d_model, signature_classes)
        self.end_gap_head = nn.Linear(d_model, end_gap_classes)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        input_kind = batch["input_kind"].to(dtype=torch.long)
        input_delta = batch["input_delta"].to(dtype=torch.long)
        input_signature = batch["input_signature"].to(dtype=torch.long)
        input_end_gap = batch["input_end_gap"].to(dtype=torch.long)
        mask = batch["row_mask"].to(dtype=torch.bool)
        control_context = batch["control_context"].to(dtype=torch.float32)
        batch_size, steps = input_kind.shape
        positions = torch.arange(steps, device=input_kind.device, dtype=torch.long).reshape(1, -1)
        hidden = (
            self.kind_embedding(input_kind)
            + self.delta_embedding(input_delta)
            + self.signature_embedding(input_signature)
            + self.end_gap_embedding(input_end_gap)
            + self.position_embedding(positions.expand(batch_size, -1))
            + self.control_projection(control_context).unsqueeze(1)
        )
        causal_mask = torch.triu(torch.ones((steps, steps), dtype=torch.bool, device=input_kind.device), diagonal=1)
        hidden = self.decoder(hidden, mask=causal_mask, src_key_padding_mask=~mask)
        hidden = self.norm(hidden)
        return {
            "kind_logits": self.kind_head(hidden),
            "delta_logits": self.delta_head(hidden),
            "signature_logits": self.signature_head(hidden),
            "end_gap_logits": self.end_gap_head(hidden),
        }


def run_delta_event_factor_target_tiny_model(
    *,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    control_teacher_cache_dir: str | Path = DEFAULT_CONTROL_TEACHER_CACHE_DIR,
    summary_output_path: str | Path = DEFAULT_SUMMARY_OUTPUT,
    report_output_path: str | Path | None = DEFAULT_REPORT_OUTPUT,
    experiment_card_path: str | Path = DEFAULT_EXPERIMENT_CARD,
    full4k_bit_proxy_summary_path: str | Path = DEFAULT_FULL4K_BIT_PROXY_SUMMARY,
    window_limit: int = 32,
    batch_size: int = 4,
    steps: int = 80,
    learning_rate: float = 3e-3,
    max_total_loss_ratio: float = 0.75,
    seed: int = 20260630,
    device_name: str = "cpu",
    max_cached_timepoint_maps: int = 16,
) -> dict[str, Any]:
    started = time.monotonic()
    _validate_run_config(
        window_limit=window_limit,
        batch_size=batch_size,
        steps=steps,
        learning_rate=learning_rate,
        max_total_loss_ratio=max_total_loss_ratio,
    )
    torch.manual_seed(int(seed))
    device = _resolve_device(device_name)
    vocab = MapperV3Vocab()
    dataset = MapperV3WindowDataset(
        index_path=index_path,
        vocab=vocab,
        control_teacher_cache_dir=control_teacher_cache_dir,
        require_control_teacher_cache=True,
        include_full_song_context=False,
        max_cached_timepoint_maps=int(max_cached_timepoint_maps),
        progress=False,
    )
    batches, dataset_metrics = _load_factorized_batches(
        dataset,
        vocab=vocab,
        window_limit=int(window_limit),
        batch_size=int(batch_size),
    )
    training = _train_tiny_model(
        batches=batches,
        vocab=vocab,
        device=device,
        steps=int(steps),
        learning_rate=float(learning_rate),
        seed=int(seed),
    )
    context = _load_context(Path(full4k_bit_proxy_summary_path))
    checks = guard_checks(
        dataset=dataset_metrics,
        training=training,
        context=context,
        max_total_loss_ratio=float(max_total_loss_ratio),
    )
    decision = decision_from_checks(checks)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar v3 delta-event factor target tiny model",
        "experiment_card": Path(experiment_card_path).as_posix(),
        "elapsed_s": time.monotonic() - started,
        "index_path": Path(index_path).as_posix(),
        "control_teacher_cache_dir": Path(control_teacher_cache_dir).as_posix(),
        "config": {
            "window_limit": int(window_limit),
            "batch_size": int(batch_size),
            "steps": int(steps),
            "learning_rate": float(learning_rate),
            "max_total_loss_ratio": float(max_total_loss_ratio),
            "seed": int(seed),
            "device": str(device),
            "max_cached_timepoint_maps": int(max_cached_timepoint_maps),
            "no_production_tokenizer_change": True,
            "no_dataset_schema_change": True,
            "no_mapper_default_change": True,
            "no_training_runner_change": True,
            "no_rollout": True,
            "uses_c3_backreference": False,
            "uses_future_lookup": False,
        },
        "dataset": {
            "source_window_count": len(dataset.records),
            "filter_report": dataset.filter_report.__dict__,
            **dataset_metrics,
        },
        "context": context,
        "training": training,
        "checks": checks,
        "decision": decision,
        "next_step": decision["next_step"],
    }
    write_summary_json(summary, Path(summary_output_path))
    if report_output_path is not None:
        write_report(summary, Path(report_output_path))
    return summary


def factorized_window_from_v3_tokens(
    token_ids: Sequence[int],
    *,
    vocab: MapperV3Vocab,
    write_start_ms: int,
    write_end_ms: int,
    chart_end_ms: int,
    is_full_chart_end: bool,
    delta_classes: int = 801,
    end_gap_classes: int = 801,
) -> FactorizedWindow:
    proxy = delta_event_proxy_from_v3_tokens(
        token_ids,
        vocab=vocab,
        write_start_ms=write_start_ms,
        write_end_ms=write_end_ms,
        chart_end_ms=chart_end_ms,
        is_full_chart_end=is_full_chart_end,
    )
    signature_index_by_value = {vocab.event_signature(token_id): index for index, token_id in enumerate(vocab.event_token_ids)}
    target_kind: list[int] = []
    target_delta: list[int] = []
    target_signature: list[int] = []
    target_end_gap: list[int] = []
    for row in proxy.rows:
        delta_label = _time_grid_label(int(row.delta_ms), delta_classes, name="event delta")
        target_kind.append(EVENT_KIND)
        target_delta.append(delta_label)
        target_signature.append(int(signature_index_by_value[str(row.signature)]))
        target_end_gap.append(IGNORE_INDEX)
    end_gap_label = _time_grid_label(int(proxy.end_gap_ms), end_gap_classes, name="end gap")
    target_kind.append(END_KIND)
    target_delta.append(IGNORE_INDEX)
    target_signature.append(IGNORE_INDEX)
    target_end_gap.append(end_gap_label)

    input_kind: list[int] = [BOS_INPUT_KIND]
    input_delta: list[int] = [0]
    input_signature: list[int] = [0]
    input_end_gap: list[int] = [0]
    for index in range(len(target_kind) - 1):
        input_kind.append(target_kind[index])
        input_delta.append(target_delta[index] if target_delta[index] != IGNORE_INDEX else 0)
        input_signature.append(target_signature[index] if target_signature[index] != IGNORE_INDEX else 0)
        input_end_gap.append(target_end_gap[index] if target_end_gap[index] != IGNORE_INDEX else 0)
    return FactorizedWindow(
        input_kind=tuple(input_kind),
        input_delta=tuple(input_delta),
        input_signature=tuple(input_signature),
        input_end_gap=tuple(input_end_gap),
        target_kind=tuple(target_kind),
        target_delta=tuple(target_delta),
        target_signature=tuple(target_signature),
        target_end_gap=tuple(target_end_gap),
        event_count=len(proxy.rows),
        end_count=1,
        reconstruction_mismatches=int(proxy.event_mismatch_count) + int(proxy.end_mismatch_count),
    )


def guard_checks(
    *,
    dataset: Mapping[str, Any],
    training: Mapping[str, Any],
    context: Mapping[str, Any],
    max_total_loss_ratio: float,
) -> dict[str, bool]:
    gradients = _mapping(training.get("first_step_gradients"))
    return {
        "no_production_tokenizer_or_dataset_change": True,
        "no_mapper_default_or_runner_change": True,
        "no_rollout": True,
        "no_c3_backreference_or_future_lookup": True,
        "full4k_bit_proxy_route_positive": _mapping(context.get("full4k_bit_proxy")).get("route")
        == "TEST_DELTA_EVENT_FACTOR_TARGET_MODEL_CARD",
        "consumed_windows_positive": int(dataset.get("consumed_window_count") or 0) > 0,
        "row_count_positive": int(dataset.get("row_count") or 0) > 0,
        "event_rows_positive": int(dataset.get("event_row_count") or 0) > 0,
        "end_rows_match_windows": int(dataset.get("end_row_count") or -1) == int(dataset.get("consumed_window_count") or -2),
        "reconstruction_mismatches_zero": int(dataset.get("reconstruction_mismatch_count") or 0) == 0,
        "all_losses_finite": bool(training.get("all_losses_finite")),
        "total_loss_decreased": float(training.get("total_loss_ratio") or math.inf) < float(max_total_loss_ratio),
        "kind_loss_decreased": float(training.get("kind_loss_ratio") or math.inf) < 1.0,
        "delta_loss_decreased": float(training.get("delta_loss_ratio") or math.inf) < 1.0,
        "signature_loss_decreased": float(training.get("signature_loss_ratio") or math.inf) < 1.0,
        "end_gap_loss_decreased": float(training.get("end_gap_loss_ratio") or math.inf) < 1.0,
        "kind_head_gradient_nonzero": float(gradients.get("kind_head_grad_abs") or 0.0) > 0.0,
        "delta_head_gradient_nonzero": float(gradients.get("delta_head_grad_abs") or 0.0) > 0.0,
        "signature_head_gradient_nonzero": float(gradients.get("signature_head_grad_abs") or 0.0) > 0.0,
        "end_gap_head_gradient_nonzero": float(gradients.get("end_gap_head_grad_abs") or 0.0) > 0.0,
        "shared_embedding_gradient_nonzero": float(gradients.get("kind_embedding_grad_abs") or 0.0) > 0.0,
    }


def decision_from_checks(checks: Mapping[str, bool]) -> dict[str, str]:
    failed = [key for key, passed in checks.items() if not bool(passed)]
    if not failed:
        return {
            "route": "TEST_DELTA_EVENT_FACTOR_TARGET_PRODUCTION_PLUMBING_CARD",
            "reason": "factorized delta-event target rows reconstructed and trained in a tiny real-data teacher-forced model",
            "next_step": "Create a bounded production-plumbing card for factorized v3 target rows before full training.",
        }
    hard_failures = {
        "full4k_bit_proxy_route_positive",
        "event_rows_positive",
        "end_rows_match_windows",
        "reconstruction_mismatches_zero",
        "all_losses_finite",
        "kind_head_gradient_nonzero",
        "delta_head_gradient_nonzero",
        "signature_head_gradient_nonzero",
        "end_gap_head_gradient_nonzero",
    }
    if any(key in hard_failures for key in failed):
        return {
            "route": "KILL_DELTA_EVENT_FACTOR_TARGET_MODEL_SURFACE",
            "reason": "factorized target model failed hard checks: " + ", ".join(failed),
            "next_step": "Do not productionize factorized target rows until labels, reconstruction, or gradients are repaired.",
        }
    return {
        "route": "MUTATE_DELTA_EVENT_FACTOR_TARGET_MODEL_SCALE_OR_LR",
        "reason": "factorized target model surface exists but training checks failed: " + ", ".join(failed),
        "next_step": "Tune tiny model scale, learning rate, or loss weights before production plumbing.",
    }


def write_summary_json(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_markdown(summary).rstrip() + "\n", encoding="utf-8")


def report_markdown(summary: Mapping[str, Any]) -> str:
    decision = _mapping(summary.get("decision"))
    dataset = _mapping(summary.get("dataset"))
    training = _mapping(summary.get("training"))
    checks = _mapping(summary.get("checks"))
    gradients = _mapping(training.get("first_step_gradients"))
    lines = [
        "# Target Grammar v3 Delta-Event Factor Target Tiny Model Result Report",
        "",
        "## Scope",
        "",
        "This research-only gate trains a tiny causal decoder on factorized delta-event rows derived from current v3 target fragments. It does not change production tokenizer behavior, dataset schema, mapper defaults, training runner, inference, or rollout.",
        "",
        "## Decision",
        "",
        f"Route: `{decision.get('route')}`.",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Recommended next step: {decision.get('next_step')}",
        "",
        "## Dataset",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        _row("source windows", dataset.get("source_window_count")),
        _row("consumed windows", dataset.get("consumed_window_count")),
        _row("row count", dataset.get("row_count")),
        _row("event rows", dataset.get("event_row_count")),
        _row("end rows", dataset.get("end_row_count")),
        _row("max row len", dataset.get("max_row_len")),
        _row("reconstruction mismatches", dataset.get("reconstruction_mismatch_count")),
        "",
        "## Training Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        _row("steps", training.get("steps")),
        _row("initial total loss", _fmt(training.get("initial_total_loss"))),
        _row("final total loss", _fmt(training.get("final_total_loss"))),
        _row("total loss ratio", _fmt(training.get("total_loss_ratio"))),
        _row("kind loss ratio", _fmt(training.get("kind_loss_ratio"))),
        _row("delta loss ratio", _fmt(training.get("delta_loss_ratio"))),
        _row("signature loss ratio", _fmt(training.get("signature_loss_ratio"))),
        _row("end-gap loss ratio", _fmt(training.get("end_gap_loss_ratio"))),
        _row("kind head grad abs", _fmt(gradients.get("kind_head_grad_abs"))),
        _row("delta head grad abs", _fmt(gradients.get("delta_head_grad_abs"))),
        _row("signature head grad abs", _fmt(gradients.get("signature_head_grad_abs"))),
        _row("end-gap head grad abs", _fmt(gradients.get("end_gap_head_grad_abs"))),
        _row("kind embedding grad abs", _fmt(gradients.get("kind_embedding_grad_abs"))),
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
            _what_passed(str(decision.get("route"))),
            "",
            "## What Surfaced",
            "",
            _what_surfaced(str(decision.get("route"))),
            "",
            "## What Is Not Proved",
            "",
            "- This does not prove production mapper quality.",
            "- This does not prove autoregressive rollout quality.",
            "- This does not yet integrate factorized rows into the production dataset/model/training runner.",
            "- This does not complete v3 replacement.",
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


def _load_factorized_batches(
    dataset: MapperV3WindowDataset,
    *,
    vocab: MapperV3Vocab,
    window_limit: int,
    batch_size: int,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any]]:
    consumed = min(int(window_limit), len(dataset))
    if consumed <= 0:
        raise ValueError("factor target tiny model requires at least one consumed window")
    windows: list[tuple[FactorizedWindow, torch.Tensor]] = []
    row_count = 0
    event_rows = 0
    end_rows = 0
    max_row_len = 0
    reconstruction_mismatches = 0
    for index in range(consumed):
        sample = dataset[index]
        factorized = factorized_window_from_v3_tokens(
            [int(token_id) for token_id in sample["target_fragment_tokens"].tolist()],
            vocab=vocab,
            write_start_ms=int(sample["write_start_ms"].item()),
            write_end_ms=int(sample["write_end_ms"].item()),
            chart_end_ms=int(sample["chart_end_ms"].item()),
            is_full_chart_end=bool(sample["is_full_chart_end"].item()),
        )
        control_context = sample["control_memory_8s"].to(dtype=torch.float32).mean(dim=0)
        windows.append((factorized, control_context))
        row_count += factorized.row_count
        event_rows += factorized.event_count
        end_rows += factorized.end_count
        max_row_len = max(max_row_len, factorized.row_count)
        reconstruction_mismatches += factorized.reconstruction_mismatches
    batches = []
    for start in range(0, consumed, int(batch_size)):
        batches.append(_collate_factorized_windows(windows[start : start + int(batch_size)]))
    return batches, {
        "consumed_window_count": int(consumed),
        "batch_count": len(batches),
        "row_count": int(row_count),
        "event_row_count": int(event_rows),
        "end_row_count": int(end_rows),
        "max_row_len": int(max_row_len),
        "reconstruction_mismatch_count": int(reconstruction_mismatches),
    }


def _collate_factorized_windows(items: Sequence[tuple[FactorizedWindow, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not items:
        raise ValueError("cannot collate empty factorized batch")
    batch_size = len(items)
    max_rows = max(window.row_count for window, _ in items)
    input_kind = torch.zeros((batch_size, max_rows), dtype=torch.long)
    input_delta = torch.zeros((batch_size, max_rows), dtype=torch.long)
    input_signature = torch.zeros((batch_size, max_rows), dtype=torch.long)
    input_end_gap = torch.zeros((batch_size, max_rows), dtype=torch.long)
    target_kind = torch.full((batch_size, max_rows), IGNORE_INDEX, dtype=torch.long)
    target_delta = torch.full((batch_size, max_rows), IGNORE_INDEX, dtype=torch.long)
    target_signature = torch.full((batch_size, max_rows), IGNORE_INDEX, dtype=torch.long)
    target_end_gap = torch.full((batch_size, max_rows), IGNORE_INDEX, dtype=torch.long)
    row_mask = torch.zeros((batch_size, max_rows), dtype=torch.bool)
    for batch_index, (window, _) in enumerate(items):
        length = window.row_count
        input_kind[batch_index, :length] = torch.tensor(window.input_kind, dtype=torch.long)
        input_delta[batch_index, :length] = torch.tensor(window.input_delta, dtype=torch.long)
        input_signature[batch_index, :length] = torch.tensor(window.input_signature, dtype=torch.long)
        input_end_gap[batch_index, :length] = torch.tensor(window.input_end_gap, dtype=torch.long)
        target_kind[batch_index, :length] = torch.tensor(window.target_kind, dtype=torch.long)
        target_delta[batch_index, :length] = torch.tensor(window.target_delta, dtype=torch.long)
        target_signature[batch_index, :length] = torch.tensor(window.target_signature, dtype=torch.long)
        target_end_gap[batch_index, :length] = torch.tensor(window.target_end_gap, dtype=torch.long)
        row_mask[batch_index, :length] = True
    return {
        "input_kind": input_kind,
        "input_delta": input_delta,
        "input_signature": input_signature,
        "input_end_gap": input_end_gap,
        "target_kind": target_kind,
        "target_delta": target_delta,
        "target_signature": target_signature,
        "target_end_gap": target_end_gap,
        "row_mask": row_mask,
        "control_context": torch.stack([context for _, context in items]),
    }


def _train_tiny_model(
    *,
    batches: Sequence[Mapping[str, torch.Tensor]],
    vocab: MapperV3Vocab,
    device: torch.device,
    steps: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    if not batches:
        raise ValueError("training requires at least one factorized batch")
    torch.manual_seed(int(seed))
    control_dim = int(batches[0]["control_context"].shape[-1])
    max_rows = max(int(batch["row_mask"].shape[1]) for batch in batches)
    model = TinyFactorizedDeltaEventModel(
        control_dim=control_dim,
        signature_classes=len(vocab.event_token_ids),
        max_rows=max(16, max_rows),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=0.0)
    initial = _evaluate_batches(model, batches, device=device)
    curves: list[dict[str, float]] = []
    first_step_gradients: dict[str, float] | None = None
    all_losses_finite = bool(initial["all_losses_finite"])
    for step in range(int(steps)):
        batch = _batch_to_device(batches[step % len(batches)], device=device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss, metrics = factorized_loss(output, batch)
        row = {
            "step": float(step),
            "total_loss": float(metrics["loss/total"]),
            "kind_loss": float(metrics["loss/kind"]),
            "delta_loss": float(metrics["loss/delta"]),
            "signature_loss": float(metrics["loss/signature"]),
            "end_gap_loss": float(metrics["loss/end_gap"]),
        }
        all_losses_finite = all_losses_finite and all(math.isfinite(value) for value in row.values())
        if not all_losses_finite:
            curves.append(row)
            break
        loss.backward()
        if step == 0:
            first_step_gradients = _gradient_snapshot(model)
        optimizer.step()
        curves.append(row)
    final = _evaluate_batches(model, batches, device=device)
    all_losses_finite = all_losses_finite and bool(final["all_losses_finite"])
    return {
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "all_losses_finite": bool(all_losses_finite),
        "initial_total_loss": initial["mean_total_loss"],
        "final_total_loss": final["mean_total_loss"],
        "total_loss_ratio": _safe_ratio(final["mean_total_loss"], initial["mean_total_loss"]),
        "initial_kind_loss": initial["mean_kind_loss"],
        "final_kind_loss": final["mean_kind_loss"],
        "kind_loss_ratio": _safe_ratio(final["mean_kind_loss"], initial["mean_kind_loss"]),
        "initial_delta_loss": initial["mean_delta_loss"],
        "final_delta_loss": final["mean_delta_loss"],
        "delta_loss_ratio": _safe_ratio(final["mean_delta_loss"], initial["mean_delta_loss"]),
        "initial_signature_loss": initial["mean_signature_loss"],
        "final_signature_loss": final["mean_signature_loss"],
        "signature_loss_ratio": _safe_ratio(final["mean_signature_loss"], initial["mean_signature_loss"]),
        "initial_end_gap_loss": initial["mean_end_gap_loss"],
        "final_end_gap_loss": final["mean_end_gap_loss"],
        "end_gap_loss_ratio": _safe_ratio(final["mean_end_gap_loss"], initial["mean_end_gap_loss"]),
        "first_step_gradients": first_step_gradients or {},
        "loss_curve_preview": _curve_preview(curves),
    }


def factorized_loss(output: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    kind_loss = F.cross_entropy(output["kind_logits"][batch["row_mask"]], batch["target_kind"][batch["row_mask"]])
    delta_loss, delta_count = _masked_cross_entropy(output["delta_logits"], batch["target_delta"])
    signature_loss, signature_count = _masked_cross_entropy(output["signature_logits"], batch["target_signature"])
    end_gap_loss, end_gap_count = _masked_cross_entropy(output["end_gap_logits"], batch["target_end_gap"])
    total = kind_loss + delta_loss + signature_loss + end_gap_loss
    metrics = {
        "loss/total": float(total.detach().cpu()),
        "loss/kind": float(kind_loss.detach().cpu()),
        "loss/delta": float(delta_loss.detach().cpu()),
        "loss/signature": float(signature_loss.detach().cpu()),
        "loss/end_gap": float(end_gap_loss.detach().cpu()),
        "labels/delta": float(delta_count),
        "labels/signature": float(signature_count),
        "labels/end_gap": float(end_gap_count),
    }
    return total, metrics


@torch.inference_mode()
def _evaluate_batches(
    model: TinyFactorizedDeltaEventModel,
    batches: Sequence[Mapping[str, torch.Tensor]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    rows: list[dict[str, float]] = []
    all_losses_finite = True
    for batch_raw in batches:
        batch = _batch_to_device(batch_raw, device=device)
        output = model(batch)
        _, metrics = factorized_loss(output, batch)
        rows.append(metrics)
        all_losses_finite = all_losses_finite and all(math.isfinite(float(value)) for value in metrics.values())
    return {
        "all_losses_finite": bool(all_losses_finite),
        "mean_total_loss": _mean([row["loss/total"] for row in rows]),
        "mean_kind_loss": _mean([row["loss/kind"] for row in rows]),
        "mean_delta_loss": _mean([row["loss/delta"] for row in rows]),
        "mean_signature_loss": _mean([row["loss/signature"] for row in rows]),
        "mean_end_gap_loss": _mean([row["loss/end_gap"] for row in rows]),
    }


def _masked_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, int]:
    valid = target.ne(IGNORE_INDEX)
    count = int(valid.sum().detach().cpu())
    if count <= 0:
        return logits.reshape(-1)[:0].sum(), 0
    return F.cross_entropy(logits[valid], target.to(device=logits.device, dtype=torch.long)[valid]), count


def _gradient_snapshot(model: TinyFactorizedDeltaEventModel) -> dict[str, float]:
    return {
        "kind_head_grad_abs": _grad_abs(model.kind_head.weight),
        "delta_head_grad_abs": _grad_abs(model.delta_head.weight),
        "signature_head_grad_abs": _grad_abs(model.signature_head.weight),
        "end_gap_head_grad_abs": _grad_abs(model.end_gap_head.weight),
        "kind_embedding_grad_abs": _grad_abs(model.kind_embedding.weight),
    }


def _load_context(path: Path) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if path.exists():
        data = _read_json(path)
        context["full4k_bit_proxy"] = {
            "path": path.as_posix(),
            "route": str(_mapping(data.get("decision")).get("route") or ""),
            "row_ratio": _mapping(data.get("metrics")).get("sequence_length_ratio_vs_v3"),
            "bit_ratio": _mapping(data.get("metrics")).get("proxy_factorized_total_bit_ratio_vs_v3"),
            "flat_pair_tractable": _mapping(data.get("metrics")).get("flat_pair_tractable"),
        }
    return context


def _time_grid_label(value_ms: int, class_count: int, *, name: str) -> int:
    value = int(value_ms)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    if value % 10 != 0:
        raise ValueError(f"{name} must be on the 10ms grid, got {value}")
    label = value // 10
    if label >= int(class_count):
        raise ValueError(f"{name} {value}ms exceeds factor target range {(int(class_count) - 1) * 10}ms")
    return int(label)


def _validate_run_config(
    *,
    window_limit: int,
    batch_size: int,
    steps: int,
    learning_rate: float,
    max_total_loss_ratio: float,
) -> None:
    if int(window_limit) <= 0:
        raise ValueError("window_limit must be positive")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(steps) <= 0:
        raise ValueError("steps must be positive")
    _require_positive_finite(learning_rate, "learning_rate")
    _require_positive_finite(max_total_loss_ratio, "max_total_loss_ratio")


def _require_positive_finite(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite numeric")
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive")


def _batch_to_device(value: Any, *, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, Mapping):
        return {key: _batch_to_device(item, device=device) for key, item in value.items()}
    return value


def _resolve_device(device_name: str) -> torch.device:
    name = str(device_name)
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


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
        "uv run python -m pulsefield_model.evals.mapper_v3_delta_event_factor_target_tiny_model",
        "uv run --group dev pytest tests/evals/test_mapper_v3_delta_event_factor_target_tiny_model.py tests/evals/test_mapper_v3_delta_event_full4k_bit_proxy.py tests/evals/test_mapper_v3_delta_event_auxiliary_full4k_label_coverage.py -q",
        "uv run python -m py_compile src/pulsefield_model/evals/mapper_v3_delta_event_factor_target_tiny_model.py",
        "python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_tiny_model_summary.json >/dev/null",
        "git diff --check",
        "```",
    ]


def _what_passed(route: str) -> str:
    if route == "TEST_DELTA_EVENT_FACTOR_TARGET_PRODUCTION_PLUMBING_CARD":
        return (
            "- Factorized rows reconstructed exactly from current v3 fragments.\n"
            "- Shifted previous-row teacher forcing was sufficient for a tiny causal model to optimize.\n"
            "- Kind, delta, signature, and end-gap heads all received gradients.\n"
            "- No production tokenizer/default/dataset/training/rollout change was made."
        )
    return "- The gate produced an explicit route; inspect failed checks before production plumbing."


def _what_surfaced(route: str) -> str:
    if route == "TEST_DELTA_EVENT_FACTOR_TARGET_PRODUCTION_PLUMBING_CARD":
        return "The factorized target surface is viable enough for a production-plumbing card, but still not replacement-ready."
    if route.startswith("MUTATE"):
        return "The factorized target surface exists, but tiny training needs model-scale, learning-rate, or loss-weight mutation."
    return "The factorized target surface failed a hard gate and should not be productionized yet."


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if float(denominator) == 0.0:
        return 0.0 if float(numerator) == 0.0 else float("inf")
    return float(numerator) / float(denominator)


def _grad_abs(parameter: torch.Tensor) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().abs().sum().cpu())


def _curve_preview(curves: Sequence[Mapping[str, float]]) -> list[dict[str, float]]:
    if not curves:
        return []
    indexes = sorted({0, min(1, len(curves) - 1), len(curves) // 2, len(curves) - 1})
    return [dict(curves[index]) for index in indexes]


def _fmt(value: object) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(number) else f"{number:.6f}"


def _row(label: str, value: object) -> str:
    return f"| {label} | `{value}` |"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the v3 delta-event factor-target tiny model gate.")
    parser.add_argument("--index-path", default=DEFAULT_INDEX_PATH)
    parser.add_argument("--control-teacher-cache-dir", default=DEFAULT_CONTROL_TEACHER_CACHE_DIR)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--report-output", default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--window-limit", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--max-total-loss-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-cached-timepoint-maps", type=int, default=16)
    args = parser.parse_args(argv)
    summary = run_delta_event_factor_target_tiny_model(
        index_path=Path(args.index_path),
        control_teacher_cache_dir=Path(args.control_teacher_cache_dir),
        summary_output_path=Path(args.summary_output),
        report_output_path=Path(args.report_output) if args.report_output else None,
        window_limit=args.window_limit,
        batch_size=args.batch_size,
        steps=args.steps,
        learning_rate=args.learning_rate,
        max_total_loss_ratio=args.max_total_loss_ratio,
        seed=args.seed,
        device_name=args.device,
        max_cached_timepoint_maps=args.max_cached_timepoint_maps,
    )
    print(
        "mapper_v3_delta_event_factor_target_tiny_model_done "
        f"route={summary['decision']['route']} "
        f"total_loss_ratio={summary['training']['total_loss_ratio']:.6f} "
        f"summary={Path(args.summary_output).as_posix()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
