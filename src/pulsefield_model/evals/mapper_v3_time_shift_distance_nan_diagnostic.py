from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from pulsefield_model.data.control_windows import DEFAULT_MAX_CACHED_MAPS
from pulsefield_model.data.mapper_sparse_windows_v3 import MapperV3WindowDataset, collate_mapper_v3_windows
from pulsefield_model.models.control import ControlDemoGlobalEncoderConfig
from pulsefield_model.models.mapper.shared.loss import time_shift_distance_loss
from pulsefield_model.models.mapper.v3 import MapperV3Config, MapperV3Model, MapperV3Vocab
from pulsefield_model.training.common import _set_deterministic_seed, select_torch_device
from pulsefield_model.training.mapper_common import _move_mapper_batch_tensors
from pulsefield_model.training.mapper_v3 import load_run_config


SUMMARY_SCHEMA_VERSION = 1
REPORT_ROOT = Path("artifacts/reports/audits/mapper_v2_1_grammar")
DEFAULT_SUMMARY_OUTPUT = REPORT_ROOT / "target_grammar_v3_time_shift_distance_masked_logit_nan_diagnostic_summary.json"
DEFAULT_REPORT_OUTPUT = REPORT_ROOT / "target_grammar_v3_time_shift_distance_masked_logit_nan_diagnostic_result_report.md"


def run_time_shift_distance_nan_diagnostic(
    *,
    config_path: str | Path,
    batch_limit: int = 1,
    summary_output: str | Path = DEFAULT_SUMMARY_OUTPUT,
    report_output: str | Path | None = DEFAULT_REPORT_OUTPUT,
    device_name: str | None = None,
    max_examples: int = 8,
) -> dict[str, Any]:
    batch_limit = int(batch_limit)
    if batch_limit <= 0:
        raise ValueError("batch_limit must be positive")
    max_examples = int(max_examples)
    if max_examples < 0:
        raise ValueError("max_examples must be non-negative")

    config = load_run_config(config_path)
    seed = int(config.get("seed", 1337))
    _set_deterministic_seed(seed)
    resolved_device_name = str(device_name or config.get("device", "auto"))
    device = select_torch_device(resolved_device_name)
    dataset = _build_dataset(config)
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 2)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        collate_fn=collate_mapper_v3_windows,
    )
    control_model_config = ControlDemoGlobalEncoderConfig(**dict(config.get("control_model", {})))
    model_values = dict(config.get("model", {}))
    model_values.setdefault("control_dim", control_model_config.d_model)
    model = MapperV3Model(MapperV3Config(**model_values)).to(device)
    model.train()

    vocab = model.vocab
    aggregate = empty_surface_aggregate()
    examples: list[dict[str, Any]] = []
    masked_losses: list[float] = []
    pre_mask_losses: list[float] = []
    skip_masked_losses: list[float] = []
    grad_abs_sums: list[float] = []
    grad_finite_values: list[bool] = []
    row_filtered_grad_abs_sums: list[float] = []
    row_filtered_grad_finite_values: list[bool] = []
    row_filtered_masked_losses: list[float] = []
    batch_summaries: list[dict[str, Any]] = []

    for batch_index, raw_batch in enumerate(loader):
        if batch_index >= batch_limit:
            break
        batch = _move_mapper_batch_tensors(raw_batch, device)
        output = model(batch)
        surfaces = build_time_shift_surfaces(output)
        target_tokens = batch["target_fragment_tokens"].to(device=device, dtype=torch.long)
        target_mask = batch["target_fragment_mask"].to(device=device, dtype=torch.bool)
        current_ms = _nested_tensor(batch, "target_fragment_states", "current_ms")
        batch_summary = analyze_time_shift_distance_surfaces(
            logits_final=surfaces["masked_logits"],
            pre_mask_logits=surfaces["pre_mask_logits"],
            target_tokens=target_tokens,
            target_mask=target_mask,
            vocab=vocab,
            current_ms=current_ms,
            max_examples=max(0, max_examples - len(examples)),
        )
        batch_summary["batch_index"] = int(batch_index)
        batch_summaries.append(batch_summary)
        _merge_surface_aggregate(aggregate, batch_summary)
        examples.extend(batch_summary.get("examples", []))

        masked_loss = _loss_value(
            surfaces["masked_logits"],
            target_tokens=target_tokens,
            target_mask=target_mask,
            vocab=vocab,
        )
        pre_loss, grad_abs_sum, grad_finite = _loss_and_gradient_probe(
            surfaces["pre_mask_logits"],
            target_tokens=target_tokens,
            target_mask=target_mask,
            vocab=vocab,
        )
        row_filtered_loss, row_filtered_grad_abs_sum, row_filtered_grad_finite = (
            _row_filtered_loss_and_gradient_probe(
                surfaces["masked_logits"],
                target_tokens=target_tokens,
                target_mask=target_mask,
                vocab=vocab,
            )
        )
        skip_loss = _loss_value(
            surfaces["masked_logits"],
            target_tokens=target_tokens,
            target_mask=target_mask & batch_summary["_finite_masked_time_shift_candidate_mask"],
            vocab=vocab,
        )
        masked_losses.append(masked_loss)
        pre_mask_losses.append(pre_loss)
        skip_masked_losses.append(skip_loss)
        grad_abs_sums.append(grad_abs_sum)
        grad_finite_values.append(grad_finite)
        row_filtered_masked_losses.append(row_filtered_loss)
        row_filtered_grad_abs_sums.append(row_filtered_grad_abs_sum)
        row_filtered_grad_finite_values.append(row_filtered_grad_finite)

    aggregate["all_nonfinite_masked_row_share"] = _safe_ratio(
        aggregate["all_nonfinite_masked_rows"],
        aggregate["target_time_shift_rows"],
    )
    aggregate["all_nonfinite_masked_any_row_share"] = _safe_ratio(
        aggregate["all_nonfinite_masked_rows_any"],
        aggregate["total_rows"],
    )
    aggregate["all_nonfinite_masked_valid_row_share"] = _safe_ratio(
        aggregate["all_nonfinite_masked_valid_rows"],
        aggregate["valid_rows"],
    )
    aggregate["gold_masked_finite_share"] = _safe_ratio(
        aggregate["gold_masked_finite_rows"],
        aggregate["target_time_shift_rows"],
    )
    aggregate["finite_pre_mask_candidate_share"] = _safe_ratio(
        aggregate["finite_pre_mask_candidate_rows"],
        aggregate["target_time_shift_rows"],
    )
    aggregate["finite_masked_candidate_share"] = _safe_ratio(
        aggregate["finite_masked_candidate_rows"],
        aggregate["target_time_shift_rows"],
    )
    aggregate["masked_loss_values"] = masked_losses
    aggregate["pre_mask_loss_values"] = pre_mask_losses
    aggregate["skip_masked_loss_values"] = skip_masked_losses
    aggregate["pre_mask_gradient_abs_sums"] = grad_abs_sums
    aggregate["pre_mask_gradient_all_finite"] = all(grad_finite_values) if grad_finite_values else False
    aggregate["pre_mask_gradient_any_nonzero"] = any(value > 0.0 for value in grad_abs_sums)
    aggregate["row_filtered_masked_loss_values"] = row_filtered_masked_losses
    aggregate["row_filtered_masked_loss_all_finite"] = bool(row_filtered_masked_losses) and all(
        math.isfinite(value) for value in row_filtered_masked_losses
    )
    aggregate["row_filtered_masked_gradient_abs_sums"] = row_filtered_grad_abs_sums
    aggregate["row_filtered_masked_gradient_all_finite"] = (
        all(row_filtered_grad_finite_values) if row_filtered_grad_finite_values else False
    )
    aggregate["row_filtered_masked_gradient_any_nonzero"] = any(value > 0.0 for value in row_filtered_grad_abs_sums)
    aggregate["masked_loss_any_nonfinite"] = any(not math.isfinite(value) for value in masked_losses)
    aggregate["pre_mask_loss_all_finite"] = bool(pre_mask_losses) and all(math.isfinite(value) for value in pre_mask_losses)
    aggregate["skip_masked_loss_all_finite"] = bool(skip_masked_losses) and all(
        math.isfinite(value) for value in skip_masked_losses
    )

    decision = diagnostic_decision(aggregate)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar v3 time-shift distance masked-logit NaN diagnostic",
        "decision": decision,
        "config": {
            "config_path": Path(config_path).as_posix(),
            "batch_limit": int(batch_limit),
            "device": str(device),
            "seed": int(seed),
            "max_examples": int(max_examples),
        },
        "aggregate": _public_aggregate(aggregate),
        "batch_summaries": [_public_batch_summary(row) for row in batch_summaries],
        "examples": examples[:max_examples],
        "interpretation": _interpretation(decision["route"]),
        "next_step": decision["next_step"],
    }
    write_summary_json(summary, summary_output)
    if report_output is not None:
        write_report(summary, report_output)
    return summary


def build_time_shift_surfaces(output: Any) -> dict[str, torch.Tensor]:
    base_logits = _require_tensor_attr(output, "base_logits")
    state_prior_bias = _require_tensor_attr(output, "state_prior_bias")
    ln_close_event_bias = _require_tensor_attr(output, "ln_close_event_bias")
    ln_close_time_shift_bias = _require_tensor_attr(output, "ln_close_time_shift_bias")
    logits_final = _require_tensor_attr(output, "logits_final")
    return {
        "masked_logits": logits_final,
        "pre_mask_logits": base_logits + state_prior_bias + ln_close_event_bias + ln_close_time_shift_bias,
    }


def analyze_time_shift_distance_surfaces(
    *,
    logits_final: torch.Tensor,
    pre_mask_logits: torch.Tensor,
    target_tokens: torch.Tensor,
    target_mask: torch.Tensor,
    vocab: MapperV3Vocab,
    current_ms: torch.Tensor | None = None,
    max_examples: int = 8,
) -> dict[str, Any]:
    if tuple(logits_final.shape) != tuple(pre_mask_logits.shape):
        raise ValueError("logits_final and pre_mask_logits must have the same shape")
    if tuple(target_tokens.shape) != tuple(logits_final.shape[:2]):
        raise ValueError("target_tokens must match logits shape prefix")
    if tuple(target_mask.shape) != tuple(target_tokens.shape):
        raise ValueError("target_mask must match target_tokens shape")
    shift_ids = torch.tensor(vocab.time_shift_token_ids, dtype=torch.long, device=logits_final.device)
    shift_id_set = torch.zeros((vocab.size,), dtype=torch.bool, device=logits_final.device)
    shift_id_set[shift_ids] = True
    target_tokens = target_tokens.to(device=logits_final.device, dtype=torch.long)
    target_mask = target_mask.to(device=logits_final.device, dtype=torch.bool)
    target_shift_mask = target_mask & shift_id_set[target_tokens.clamp(0, vocab.size - 1)]

    masked_shift_logits = logits_final.index_select(dim=-1, index=shift_ids)
    pre_mask_shift_logits = pre_mask_logits.index_select(dim=-1, index=shift_ids)
    masked_finite = torch.isfinite(masked_shift_logits)
    pre_mask_finite = torch.isfinite(pre_mask_shift_logits)
    finite_masked_candidate = masked_finite.any(dim=-1)
    finite_pre_mask_candidate = pre_mask_finite.any(dim=-1)
    all_nonfinite_masked_rows_any = ~finite_masked_candidate
    gold_masked = torch.gather(logits_final, dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
    gold_pre_mask = torch.gather(pre_mask_logits, dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)

    all_nonfinite_masked_rows = target_shift_mask & ~finite_masked_candidate
    examples = _surface_examples(
        target_tokens=target_tokens,
        target_shift_mask=target_shift_mask,
        all_nonfinite_masked_rows=all_nonfinite_masked_rows,
        finite_masked_count=masked_finite.sum(dim=-1),
        finite_pre_mask_count=pre_mask_finite.sum(dim=-1),
        gold_masked=gold_masked,
        gold_pre_mask=gold_pre_mask,
        vocab=vocab,
        current_ms=current_ms,
        max_examples=max_examples,
    )
    return {
        "total_rows": int(target_tokens.numel()),
        "valid_rows": _count(target_mask),
        "target_time_shift_rows": _count(target_shift_mask),
        "all_nonfinite_masked_rows": _count(all_nonfinite_masked_rows),
        "all_nonfinite_masked_rows_any": _count(all_nonfinite_masked_rows_any),
        "all_nonfinite_masked_valid_rows": _count(target_mask & all_nonfinite_masked_rows_any),
        "all_nonfinite_masked_invalid_rows": _count((~target_mask) & all_nonfinite_masked_rows_any),
        "all_nonfinite_masked_valid_non_target_rows": _count(
            target_mask & (~target_shift_mask) & all_nonfinite_masked_rows_any
        ),
        "finite_masked_candidate_rows": _count(target_shift_mask & finite_masked_candidate),
        "finite_pre_mask_candidate_rows": _count(target_shift_mask & finite_pre_mask_candidate),
        "gold_masked_finite_rows": _count(target_shift_mask & torch.isfinite(gold_masked)),
        "gold_pre_mask_finite_rows": _count(target_shift_mask & torch.isfinite(gold_pre_mask)),
        "examples": examples,
        "_finite_masked_time_shift_candidate_mask": finite_masked_candidate.detach(),
    }


def diagnostic_decision(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    target_rows = int(aggregate.get("target_time_shift_rows") or 0)
    all_nonfinite = int(aggregate.get("all_nonfinite_masked_rows") or 0)
    all_nonfinite_any = int(aggregate.get("all_nonfinite_masked_rows_any") or 0)
    pre_loss_ok = bool(aggregate.get("pre_mask_loss_all_finite"))
    grad_ok = bool(aggregate.get("pre_mask_gradient_all_finite"))
    grad_nonzero = bool(aggregate.get("pre_mask_gradient_any_nonzero"))
    row_filtered_loss_ok = bool(aggregate.get("row_filtered_masked_loss_all_finite"))
    row_filtered_grad_ok = bool(aggregate.get("row_filtered_masked_gradient_all_finite"))
    row_filtered_grad_nonzero = bool(aggregate.get("row_filtered_masked_gradient_any_nonzero"))
    masked_loss_nonfinite = bool(aggregate.get("masked_loss_any_nonfinite"))
    if target_rows <= 0:
        route = "MUTATE_TIME_SHIFT_DISTANCE_DIAGNOSTIC"
        reason = "no target time-shift rows were observed on this diagnostic slice"
        next_step = "Increase batch_limit or choose a slice with target time-shift rows before changing the objective."
    elif (
        masked_loss_nonfinite
        and all_nonfinite_any > 0
        and all_nonfinite == 0
        and row_filtered_loss_ok
        and row_filtered_grad_ok
        and row_filtered_grad_nonzero
    ):
        route = "TEST_ROW_FILTERED_TIME_SHIFT_DISTANCE_REPAIR"
        reason = (
            "masked loss is non-finite because non-target or padded rows are softmaxed before masking; "
            "row-filtered masked candidate loss has finite nonzero gradients"
        )
        next_step = "Create/execute a bounded repair card that computes time-shift distance only on target time-shift rows."
    elif all_nonfinite > 0 and pre_loss_ok and grad_ok and grad_nonzero:
        route = "TEST_PREMASK_TIME_SHIFT_DISTANCE_REPAIR"
        reason = "masked time-shift rows reproduce the NaN path and pre-mask candidate loss has finite nonzero gradients"
        next_step = "Create/execute a bounded repair card that feeds finite pre-mask logits to the auxiliary only."
    elif all_nonfinite <= 0 and masked_loss_nonfinite:
        route = "KILL_TIME_SHIFT_DISTANCE_DISTANCE_LOSS"
        reason = "the masked loss is non-finite but all-nonfinite masked time-shift rows were not observed"
        next_step = "Stop this objective and inspect a different NaN source before training again."
    elif all_nonfinite <= 0:
        route = "MUTATE_TIME_SHIFT_DISTANCE_DIAGNOSTIC"
        reason = "target time-shift rows were observed but the masked-logit all-nonfinite hypothesis was not confirmed"
        next_step = "Broaden the diagnostic slice or inspect model-parameter NaN onset."
    else:
        route = "KILL_TIME_SHIFT_DISTANCE_DISTANCE_LOSS"
        reason = "masked rows were found but the pre-mask candidate loss or gradient was not finite and nonzero"
        next_step = "Do not continue this auxiliary without a different objective formulation."
    return {"route": route, "reason": reason, "next_step": next_step}


def write_summary_json(summary: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(summary: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_markdown(summary), encoding="utf-8")


def report_markdown(summary: Mapping[str, Any]) -> str:
    decision = _mapping(summary.get("decision"))
    aggregate = _mapping(summary.get("aggregate"))
    examples = [row for row in summary.get("examples", []) if isinstance(row, Mapping)]
    lines = [
        "# Target Grammar v3 Time-Shift Distance Masked-Logit NaN Diagnostic Result Report",
        "",
        "## Scope",
        "",
        "This diagnostic inspects real v3 training batches without optimizer steps. It compares grammar-masked `logits_final` time-shift rows against a finite pre-mask candidate surface.",
        "",
        "## Decision",
        "",
        f"Route: `{decision.get('route')}`.",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Next step: {decision.get('next_step')}",
        "",
        "## Metrics",
        "",
        f"- total rows: `{aggregate.get('total_rows')}`",
        f"- valid rows: `{aggregate.get('valid_rows')}`",
        f"- target time-shift rows: `{aggregate.get('target_time_shift_rows')}`",
        f"- target all-nonfinite masked rows: `{aggregate.get('all_nonfinite_masked_rows')}`",
        f"- target all-nonfinite masked share: `{_fmt_ratio(aggregate.get('all_nonfinite_masked_row_share'))}`",
        f"- any-row all-nonfinite masked rows: `{aggregate.get('all_nonfinite_masked_rows_any')}`",
        f"- any-row all-nonfinite masked share: `{_fmt_ratio(aggregate.get('all_nonfinite_masked_any_row_share'))}`",
        f"- valid all-nonfinite masked rows: `{aggregate.get('all_nonfinite_masked_valid_rows')}`",
        f"- valid non-target all-nonfinite masked rows: `{aggregate.get('all_nonfinite_masked_valid_non_target_rows')}`",
        f"- invalid all-nonfinite masked rows: `{aggregate.get('all_nonfinite_masked_invalid_rows')}`",
        f"- finite masked candidate share: `{_fmt_ratio(aggregate.get('finite_masked_candidate_share'))}`",
        f"- finite pre-mask candidate share: `{_fmt_ratio(aggregate.get('finite_pre_mask_candidate_share'))}`",
        f"- gold masked finite share: `{_fmt_ratio(aggregate.get('gold_masked_finite_share'))}`",
        f"- masked loss values: `{aggregate.get('masked_loss_values')}`",
        f"- pre-mask loss values: `{aggregate.get('pre_mask_loss_values')}`",
        f"- skip-masked loss values: `{aggregate.get('skip_masked_loss_values')}`",
        f"- row-filtered masked loss values: `{aggregate.get('row_filtered_masked_loss_values')}`",
        f"- pre-mask gradient abs sums: `{aggregate.get('pre_mask_gradient_abs_sums')}`",
        f"- pre-mask gradient all finite: `{aggregate.get('pre_mask_gradient_all_finite')}`",
        f"- pre-mask gradient any nonzero: `{aggregate.get('pre_mask_gradient_any_nonzero')}`",
        f"- row-filtered masked gradient abs sums: `{aggregate.get('row_filtered_masked_gradient_abs_sums')}`",
        f"- row-filtered masked gradient all finite: `{aggregate.get('row_filtered_masked_gradient_all_finite')}`",
        f"- row-filtered masked gradient any nonzero: `{aggregate.get('row_filtered_masked_gradient_any_nonzero')}`",
        "",
        "## Examples",
        "",
        "| Batch | Step | ms | Target | Masked finite TS | Pre-mask finite TS | Gold masked finite | Gold pre-mask finite |",
        "| ---: | ---: | ---: | --- | ---: | ---: | --- | --- |",
    ]
    if examples:
        for row in examples:
            lines.append(
                "| {batch} | {step} | {ms} | `{target}` | {masked} | {pre} | {gold_masked} | {gold_pre} |".format(
                    batch=row.get("batch_index"),
                    step=row.get("step"),
                    ms=row.get("current_ms"),
                    target=row.get("target_token"),
                    masked=row.get("finite_masked_time_shift_count"),
                    pre=row.get("finite_pre_mask_time_shift_count"),
                    gold_masked=row.get("gold_masked_finite"),
                    gold_pre=row.get("gold_pre_mask_finite"),
                )
            )
    else:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(summary.get("interpretation")),
            "",
        ]
    )
    return "\n".join(lines)


def empty_surface_aggregate() -> dict[str, Any]:
    return {
        "batch_count": 0,
        "total_rows": 0,
        "valid_rows": 0,
        "target_time_shift_rows": 0,
        "all_nonfinite_masked_rows": 0,
        "all_nonfinite_masked_rows_any": 0,
        "all_nonfinite_masked_valid_rows": 0,
        "all_nonfinite_masked_invalid_rows": 0,
        "all_nonfinite_masked_valid_non_target_rows": 0,
        "finite_masked_candidate_rows": 0,
        "finite_pre_mask_candidate_rows": 0,
        "gold_masked_finite_rows": 0,
        "gold_pre_mask_finite_rows": 0,
    }


def _merge_surface_aggregate(aggregate: dict[str, Any], batch_summary: Mapping[str, Any]) -> None:
    aggregate["batch_count"] += 1
    for key in (
        "total_rows",
        "valid_rows",
        "target_time_shift_rows",
        "all_nonfinite_masked_rows",
        "all_nonfinite_masked_rows_any",
        "all_nonfinite_masked_valid_rows",
        "all_nonfinite_masked_invalid_rows",
        "all_nonfinite_masked_valid_non_target_rows",
        "finite_masked_candidate_rows",
        "finite_pre_mask_candidate_rows",
        "gold_masked_finite_rows",
        "gold_pre_mask_finite_rows",
    ):
        aggregate[key] += int(batch_summary.get(key) or 0)


def _public_aggregate(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in aggregate.items()
        if not str(key).startswith("_")
    }


def _public_batch_summary(batch_summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in batch_summary.items()
        if not str(key).startswith("_")
    }


def _build_dataset(config: Mapping[str, Any]) -> MapperV3WindowDataset:
    dataset_kwargs: dict[str, Any] = {
        "dataset_root": Path(config.get("dataset_root", "dataset")),
        "include_full_song_context": bool(config.get("include_full_song_context", True)),
        "max_cached_maps": DEFAULT_MAX_CACHED_MAPS if config.get("max_cached_maps") is None else int(config["max_cached_maps"]),
        "progress": bool(config.get("dataset_progress", False)),
    }
    if config.get("index_path") is not None:
        dataset_kwargs["index_path"] = Path(str(config["index_path"]))
    if config.get("control_v3_timeseries_path") is not None:
        dataset_kwargs["control_v3_timeseries_path"] = Path(str(config["control_v3_timeseries_path"]))
    if config.get("mapper_record_cache_path") is not None:
        dataset_kwargs["mapper_record_cache_path"] = Path(str(config["mapper_record_cache_path"]))
    if config.get("control_teacher_cache_dir") is not None:
        dataset_kwargs["control_teacher_cache_dir"] = Path(str(config["control_teacher_cache_dir"]))
        dataset_kwargs["require_control_teacher_cache"] = bool(config.get("require_control_teacher_cache", False))
    return MapperV3WindowDataset(**dataset_kwargs)


def _loss_value(
    logits: torch.Tensor,
    *,
    target_tokens: torch.Tensor,
    target_mask: torch.Tensor,
    vocab: MapperV3Vocab,
) -> float:
    loss = time_shift_distance_loss(
        logits_final=logits,
        target_tokens=target_tokens,
        vocab=vocab,
        target_mask=target_mask,
    )
    return float(loss.detach().cpu())


def _loss_and_gradient_probe(
    logits: torch.Tensor,
    *,
    target_tokens: torch.Tensor,
    target_mask: torch.Tensor,
    vocab: MapperV3Vocab,
) -> tuple[float, float, bool]:
    probe_logits = logits.detach().clone().requires_grad_(True)
    loss = time_shift_distance_loss(
        logits_final=probe_logits,
        target_tokens=target_tokens,
        vocab=vocab,
        target_mask=target_mask,
    )
    loss.backward()
    grad = probe_logits.grad
    if grad is None:
        return float(loss.detach().cpu()), 0.0, False
    return (
        float(loss.detach().cpu()),
        float(grad.detach().abs().sum().cpu()),
        bool(torch.isfinite(grad).all().item()),
    )


def _row_filtered_loss_and_gradient_probe(
    logits: torch.Tensor,
    *,
    target_tokens: torch.Tensor,
    target_mask: torch.Tensor,
    vocab: MapperV3Vocab,
) -> tuple[float, float, bool]:
    shift_ids = torch.tensor(vocab.time_shift_token_ids, dtype=torch.long, device=logits.device)
    shift_id_set = torch.zeros((vocab.size,), dtype=torch.bool, device=logits.device)
    shift_id_set[shift_ids] = True
    target_tokens = target_tokens.to(device=logits.device, dtype=torch.long)
    target_mask = target_mask.to(device=logits.device, dtype=torch.bool)
    target_shift_mask = target_mask & shift_id_set[target_tokens.clamp(0, vocab.size - 1)]
    if not bool(target_shift_mask.any()):
        zero = logits.reshape(-1)[:0].sum() * 0.0
        return float(zero.detach().cpu()), 0.0, True

    selected_logits = logits[target_shift_mask].detach().clone().reshape(1, -1, logits.shape[-1]).requires_grad_(True)
    selected_targets = target_tokens[target_shift_mask].reshape(1, -1)
    selected_mask = torch.ones_like(selected_targets, dtype=torch.bool, device=logits.device)
    loss = time_shift_distance_loss(
        logits_final=selected_logits,
        target_tokens=selected_targets,
        vocab=vocab,
        target_mask=selected_mask,
    )
    loss.backward()
    grad = selected_logits.grad
    if grad is None:
        return float(loss.detach().cpu()), 0.0, False
    return (
        float(loss.detach().cpu()),
        float(grad.detach().abs().sum().cpu()),
        bool(torch.isfinite(grad).all().item()),
    )


def _surface_examples(
    *,
    target_tokens: torch.Tensor,
    target_shift_mask: torch.Tensor,
    all_nonfinite_masked_rows: torch.Tensor,
    finite_masked_count: torch.Tensor,
    finite_pre_mask_count: torch.Tensor,
    gold_masked: torch.Tensor,
    gold_pre_mask: torch.Tensor,
    vocab: MapperV3Vocab,
    current_ms: torch.Tensor | None,
    max_examples: int,
) -> list[dict[str, Any]]:
    if max_examples <= 0:
        return []
    preferred = torch.nonzero(all_nonfinite_masked_rows, as_tuple=False)
    if int(preferred.shape[0]) == 0:
        preferred = torch.nonzero(target_shift_mask, as_tuple=False)
    examples: list[dict[str, Any]] = []
    current = None if current_ms is None else current_ms.detach().to(device=target_tokens.device)
    for row in preferred[:max_examples].tolist():
        batch_index, step = int(row[0]), int(row[1])
        token_id = int(target_tokens[batch_index, step].item())
        examples.append(
            {
                "batch_index": batch_index,
                "step": step,
                "current_ms": None if current is None else int(current[batch_index, step].item()),
                "target_token_id": token_id,
                "target_token": vocab.token_name(token_id),
                "finite_masked_time_shift_count": int(finite_masked_count[batch_index, step].item()),
                "finite_pre_mask_time_shift_count": int(finite_pre_mask_count[batch_index, step].item()),
                "gold_masked_finite": bool(torch.isfinite(gold_masked[batch_index, step]).item()),
                "gold_pre_mask_finite": bool(torch.isfinite(gold_pre_mask[batch_index, step]).item()),
            }
        )
    return examples


def _nested_tensor(batch: Mapping[str, Any], parent: str, key: str) -> torch.Tensor | None:
    value = batch.get(parent)
    if not isinstance(value, Mapping):
        return None
    nested = value.get(key)
    return nested if isinstance(nested, torch.Tensor) else None


def _require_tensor_attr(output: Any, name: str) -> torch.Tensor:
    value = getattr(output, name, None)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"model output missing tensor {name!r}")
    return value


def _count(mask: torch.Tensor) -> int:
    return int(mask.to(dtype=torch.bool).sum().detach().cpu())


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if float(denominator) == 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _fmt_ratio(value: object) -> str:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(numeric):
        return "nan"
    return f"{100.0 * numeric:.2f}%"


def _interpretation(route: str) -> str:
    if route == "TEST_ROW_FILTERED_TIME_SHIFT_DISTANCE_REPAIR":
        return (
            "The real-batch diagnostic supports a row-filtering repair rather than a tokenizer or global "
            "pre-mask surface change. Target time-shift rows have finite masked candidates, but other rows can "
            "be all -inf after grammar masking and make the full softmax non-finite before the target mask is "
            "applied."
        )
    if route == "TEST_PREMASK_TIME_SHIFT_DISTANCE_REPAIR":
        return (
            "The real-batch diagnostic supports the masked-logit NaN hypothesis. The next test should route "
            "only the time-shift distance auxiliary through a finite pre-mask logit surface."
        )
    if route == "MUTATE_TIME_SHIFT_DISTANCE_DIAGNOSTIC":
        return (
            "The diagnostic was inconclusive for the current slice. Broaden the diagnostic before training or "
            "changing the objective."
        )
    return (
        "The diagnostic does not support a simple pre-mask repair. Stop the current distance objective unless a "
        "new bounded card defines a different formulation."
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Diagnose v3 time-shift distance NaNs on real batches.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-limit", type=int, default=1)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT.as_posix())
    parser.add_argument("--report-output", default=DEFAULT_REPORT_OUTPUT.as_posix())
    parser.add_argument("--device")
    parser.add_argument("--max-examples", type=int, default=8)
    args = parser.parse_args(argv)
    summary = run_time_shift_distance_nan_diagnostic(
        config_path=args.config,
        batch_limit=args.batch_limit,
        summary_output=args.summary_output,
        report_output=args.report_output,
        device_name=args.device,
        max_examples=args.max_examples,
    )
    print(
        "mapper_v3_time_shift_distance_nan_diagnostic_done "
        f"route={summary['decision']['route']} "
        f"target_rows={summary['aggregate']['target_time_shift_rows']} "
        f"all_nonfinite={summary['aggregate']['all_nonfinite_masked_rows']} "
        f"all_nonfinite_any={summary['aggregate'].get('all_nonfinite_masked_rows_any')} "
        f"summary={Path(args.summary_output).as_posix()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
