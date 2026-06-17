from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from pulsefield_model.data.mapper_sparse_windows_v3 import MapperV3WindowDataset, collate_mapper_v3_windows
from pulsefield_model.models.mapper.v3 import MapperV3Config, MapperV3LossConfig, MapperV3Model, MapperV3ModelLoss
from pulsefield_model.models.mapper.v3.vocab import MapperV3Vocab


SUMMARY_SCHEMA_VERSION = 1
REPORT_ROOT = Path("artifacts/reports/audits/mapper_v2_1_grammar")
DEFAULT_INDEX_PATH = Path("artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet")
DEFAULT_CONTROL_TEACHER_CACHE_DIR = Path("artifacts/cache/stage2_mapper_v2_1/control_teacher_d384_l3_stride16_step002000")
DEFAULT_EXPERIMENT_CARD = REPORT_ROOT / "target_grammar_v3_delta_event_auxiliary_tiny_real_data_training_experiment_card.md"
DEFAULT_SUMMARY_OUTPUT = REPORT_ROOT / "target_grammar_v3_delta_event_auxiliary_tiny_real_data_training_summary.json"
DEFAULT_REPORT_OUTPUT = REPORT_ROOT / "target_grammar_v3_delta_event_auxiliary_tiny_real_data_training_result_report.md"
DEFAULT_LABEL_COVERAGE_SUMMARY_PATH = REPORT_ROOT / "target_grammar_v3_delta_event_auxiliary_fixed_slice_label_coverage_summary.json"
DEFAULT_SYNTHETIC_TRAINING_SUMMARY_PATH = REPORT_ROOT / "target_grammar_v3_delta_event_auxiliary_tiny_training_gate_summary.json"


def run_delta_event_auxiliary_tiny_real_data_training(
    *,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    control_teacher_cache_dir: str | Path = DEFAULT_CONTROL_TEACHER_CACHE_DIR,
    summary_output_path: str | Path = DEFAULT_SUMMARY_OUTPUT,
    report_output_path: str | Path | None = DEFAULT_REPORT_OUTPUT,
    experiment_card_path: str | Path = DEFAULT_EXPERIMENT_CARD,
    label_coverage_summary_path: str | Path = DEFAULT_LABEL_COVERAGE_SUMMARY_PATH,
    synthetic_training_summary_path: str | Path = DEFAULT_SYNTHETIC_TRAINING_SUMMARY_PATH,
    window_limit: int = 32,
    batch_size: int = 4,
    steps: int = 32,
    learning_rate: float = 3e-3,
    lambda_delta_event_auxiliary: float = 1.0,
    max_auxiliary_loss_ratio: float = 0.75,
    max_token_loss_ratio: float = 1.10,
    seed: int = 20260629,
    device_name: str = "cpu",
    max_cached_timepoint_maps: int = 16,
) -> dict[str, Any]:
    started = time.monotonic()
    _validate_run_config(
        window_limit=window_limit,
        batch_size=batch_size,
        steps=steps,
        learning_rate=learning_rate,
        lambda_delta_event_auxiliary=lambda_delta_event_auxiliary,
        max_auxiliary_loss_ratio=max_auxiliary_loss_ratio,
        max_token_loss_ratio=max_token_loss_ratio,
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
    batches, dataset_metrics = _load_real_batches(
        dataset,
        vocab=vocab,
        window_limit=int(window_limit),
        batch_size=int(batch_size),
    )
    training = _train_enabled_auxiliary_probe(
        batches=batches,
        vocab=vocab,
        device=device,
        steps=int(steps),
        learning_rate=float(learning_rate),
        lambda_delta_event_auxiliary=float(lambda_delta_event_auxiliary),
        max_seq_len=int(dataset_metrics["max_seq_len"]),
        seed=int(seed),
    )
    context = _load_context(
        label_coverage_summary_path=Path(label_coverage_summary_path),
        synthetic_training_summary_path=Path(synthetic_training_summary_path),
    )
    checks = guard_checks(
        training=training,
        dataset=dataset_metrics,
        context=context,
        max_auxiliary_loss_ratio=float(max_auxiliary_loss_ratio),
        max_token_loss_ratio=float(max_token_loss_ratio),
    )
    decision = decision_from_checks(checks)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar v3 delta-event auxiliary tiny real-data training",
        "experiment_card": Path(experiment_card_path).as_posix(),
        "elapsed_s": time.monotonic() - started,
        "index_path": Path(index_path).as_posix(),
        "control_teacher_cache_dir": Path(control_teacher_cache_dir).as_posix(),
        "config": {
            "window_limit": int(window_limit),
            "batch_size": int(batch_size),
            "steps": int(steps),
            "learning_rate": float(learning_rate),
            "lambda_delta_event_auxiliary": float(lambda_delta_event_auxiliary),
            "max_auxiliary_loss_ratio": float(max_auxiliary_loss_ratio),
            "max_token_loss_ratio": float(max_token_loss_ratio),
            "seed": int(seed),
            "device": str(device),
            "max_cached_timepoint_maps": int(max_cached_timepoint_maps),
            "no_rollout": True,
            "tokenizer_changed": False,
            "dataset_schema_changed": False,
            "default_behavior_changed": False,
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


def guard_checks(
    *,
    training: Mapping[str, Any],
    dataset: Mapping[str, Any],
    context: Mapping[str, Any],
    max_auxiliary_loss_ratio: float,
    max_token_loss_ratio: float,
) -> dict[str, bool]:
    first_step_gradients = _mapping(training.get("first_step_gradients"))
    return {
        "no_rollout": True,
        "no_tokenizer_dataset_default_change": True,
        "no_c3_backreference_or_future_lookup": True,
        "label_coverage_route_passed": _mapping(context.get("fixed_slice_label_coverage")).get("route")
        == "TEST_DELTA_EVENT_AUXILIARY_TINY_REAL_DATA_TRAINING",
        "synthetic_training_route_passed": _mapping(context.get("synthetic_tiny_training")).get("route")
        == "TEST_DELTA_EVENT_AUXILIARY_FIXED_SLICE_GATE",
        "consumed_windows_positive": int(dataset.get("consumed_window_count") or 0) > 0,
        "real_batches_positive": int(dataset.get("batch_count") or 0) > 0,
        "every_batch_has_event_labels": int(training.get("min_event_label_count_per_batch") or 0) > 0,
        "every_batch_has_signature_labels": int(training.get("min_signature_label_count_per_batch") or 0) > 0,
        "every_batch_has_end_gap_labels": int(training.get("min_end_gap_label_count_per_batch") or 0) > 0,
        "all_losses_finite": bool(training.get("all_losses_finite")),
        "initial_auxiliary_loss_positive": float(training.get("initial_auxiliary_loss") or 0.0) > 0.0,
        "final_auxiliary_loss_finite": math.isfinite(float(training.get("final_auxiliary_loss") or float("nan"))),
        "auxiliary_loss_decreased": float(training.get("auxiliary_loss_ratio") or float("inf"))
        <= float(max_auxiliary_loss_ratio),
        "total_loss_decreased": float(training.get("total_loss_ratio") or float("inf")) < 1.0,
        "token_loss_not_materially_worse": float(training.get("token_loss_ratio") or float("inf"))
        <= float(max_token_loss_ratio),
        "delta_head_gradient_nonzero": float(first_step_gradients.get("delta_head_grad_abs") or 0.0) > 0.0,
        "signature_head_gradient_nonzero": float(first_step_gradients.get("signature_head_grad_abs") or 0.0) > 0.0,
        "end_gap_head_gradient_nonzero": float(first_step_gradients.get("end_gap_head_grad_abs") or 0.0) > 0.0,
        "shared_decoder_gradient_nonzero": float(first_step_gradients.get("token_embedding_grad_abs") or 0.0) > 0.0,
    }


def decision_from_checks(checks: Mapping[str, bool]) -> dict[str, str]:
    failed = [key for key, passed in checks.items() if not bool(passed)]
    if not failed:
        return {
            "route": "TEST_DELTA_EVENT_AUXILIARY_FULL_SLICE_OR_FULL4K_LABEL_COVERAGE",
            "reason": "real fixed-slice tiny training gate passed for the enabled delta-event auxiliary objective",
            "next_step": "Stress delta-event label coverage on a larger slice or full4k, with explicit end-gap range analysis before any default change.",
        }
    kill_failures = {
        "label_coverage_route_passed",
        "synthetic_training_route_passed",
        "every_batch_has_event_labels",
        "every_batch_has_signature_labels",
        "every_batch_has_end_gap_labels",
        "all_losses_finite",
        "delta_head_gradient_nonzero",
        "signature_head_gradient_nonzero",
        "end_gap_head_gradient_nonzero",
    }
    weighting_failures = {
        "auxiliary_loss_decreased",
        "total_loss_decreased",
        "token_loss_not_materially_worse",
    }
    if any(key in kill_failures for key in failed):
        return {
            "route": "KILL_DELTA_EVENT_AUXILIARY_REAL_DATA_TRAINING",
            "reason": "failed hard real-data checks: " + ", ".join(failed),
            "next_step": "Do not spend rollout runtime; repair label plumbing, gradients, or loss stability first.",
        }
    if any(key in weighting_failures for key in failed):
        return {
            "route": "MUTATE_DELTA_EVENT_AUXILIARY_WEIGHTING",
            "reason": "failed optimization/weighting checks: " + ", ".join(failed),
            "next_step": "Adjust lambda, end-gap weight, learning rate, or model scale before larger training.",
        }
    return {
        "route": "MUTATE_DELTA_EVENT_AUXILIARY_REAL_DATA_GATE",
        "reason": "failed soft real-data checks: " + ", ".join(failed),
        "next_step": "Tighten the real-data gate setup before scaling.",
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
    dataset = _mapping(summary.get("dataset"))
    gradients = _mapping(training.get("first_step_gradients"))
    checks = _mapping(summary.get("checks"))
    config = _mapping(summary.get("config"))
    lines = [
        "# Target Grammar v3 Delta-Event Auxiliary Tiny Real-Data Training Result Report",
        "",
        "## Scope",
        "",
        "This gate trains a small v3 model on real fixed-slice teacher-forced batches with the delta-event auxiliary objective enabled. It does not roll out, change tokenizer behavior, change dataset schema, change mapper defaults, or use C3/future context.",
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
        _row("batch count", dataset.get("batch_count")),
        _row("batch size", config.get("batch_size")),
        _row("max seq len", dataset.get("max_seq_len")),
        _row("valid target tokens", dataset.get("valid_target_token_count")),
        "",
        "## Training Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        _row("steps", training.get("steps")),
        _row("learning rate", training.get("learning_rate")),
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
        _row("signature labels", training.get("signature_label_count")),
        _row("end-gap labels", training.get("end_gap_label_count")),
        _row("min event labels per batch", training.get("min_event_label_count_per_batch")),
        _row("min end-gap labels per batch", training.get("min_end_gap_label_count_per_batch")),
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
            _what_passed(str(decision.get("route"))),
            "",
            "## What Surfaced",
            "",
            _what_surfaced(str(decision.get("route"))),
            "",
            "## What Is Not Proved",
            "",
            "- This does not prove autoregressive rollout quality.",
            "- This does not prove full4k label coverage.",
            "- This does not make v3 target replacement ready.",
            "- This does not compare against a full v2.1/v3 trained pipeline.",
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


def _load_real_batches(
    dataset: MapperV3WindowDataset,
    *,
    vocab: MapperV3Vocab,
    window_limit: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    consumed = min(int(window_limit), len(dataset))
    if consumed <= 0:
        raise ValueError("tiny real-data training requires at least one consumed window")
    batches: list[dict[str, Any]] = []
    max_seq_len = 0
    valid_target_token_count = 0
    for start in range(0, consumed, int(batch_size)):
        samples = [dataset[index] for index in range(start, min(start + int(batch_size), consumed))]
        batch = collate_mapper_v3_windows(samples, pad_id=vocab.pad_id)
        seq_len = int(batch["target_fragment_tokens"].shape[1])
        max_seq_len = max(max_seq_len, seq_len)
        valid_target_token_count += int(batch["target_fragment_mask"].to(dtype=torch.bool).sum().item())
        batches.append(batch)
    return batches, {
        "consumed_window_count": int(consumed),
        "batch_count": len(batches),
        "max_seq_len": int(max_seq_len),
        "valid_target_token_count": int(valid_target_token_count),
    }


def _train_enabled_auxiliary_probe(
    *,
    batches: Sequence[Mapping[str, Any]],
    vocab: MapperV3Vocab,
    device: torch.device,
    steps: int,
    learning_rate: float,
    lambda_delta_event_auxiliary: float,
    max_seq_len: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    control_dim = _control_dim_from_batches(batches)
    model = MapperV3Model(
        _small_real_data_config(
            control_dim=control_dim,
            max_seq_len=max(64, int(max_seq_len)),
            use_delta_event_auxiliary_target=True,
        ),
        vocab=vocab,
    ).to(device)
    loss_fn = MapperV3ModelLoss(
        MapperV3LossConfig(
            lambda_density=0.0,
            lambda_ln_close=0.0,
            lambda_adapter_reg=0.0,
            lambda_delta_event_auxiliary=float(lambda_delta_event_auxiliary),
        ),
        vocab=vocab,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=0.0)
    initial = _evaluate_batches(model, loss_fn, batches, device=device)
    train_curve: list[dict[str, float]] = []
    first_step_gradients: dict[str, float] | None = None
    all_losses_finite = bool(initial["all_losses_finite"])
    for step in range(int(steps)):
        batch = _batch_to_device(batches[step % len(batches)], device=device)
        model.train()
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
        all_losses_finite = all_losses_finite and all(math.isfinite(value) for value in row.values())
        if not all_losses_finite:
            train_curve.append(row)
            break
        loss.total_loss.backward()
        if step == 0:
            first_step_gradients = _gradient_snapshot(model)
        optimizer.step()
        train_curve.append(row)
    final = _evaluate_batches(model, loss_fn, batches, device=device)
    all_losses_finite = all_losses_finite and bool(final["all_losses_finite"])
    return {
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "lambda_delta_event_auxiliary": float(lambda_delta_event_auxiliary),
        "all_losses_finite": bool(all_losses_finite),
        "initial_total_loss": initial["mean_total_loss"],
        "final_total_loss": final["mean_total_loss"],
        "total_loss_ratio": _safe_ratio(final["mean_total_loss"], initial["mean_total_loss"]),
        "initial_token_loss": initial["mean_token_loss"],
        "final_token_loss": final["mean_token_loss"],
        "token_loss_ratio": _safe_ratio(final["mean_token_loss"], initial["mean_token_loss"]),
        "initial_auxiliary_loss": initial["mean_delta_event_auxiliary_loss"],
        "final_auxiliary_loss": final["mean_delta_event_auxiliary_loss"],
        "auxiliary_loss_ratio": _safe_ratio(
            final["mean_delta_event_auxiliary_loss"],
            initial["mean_delta_event_auxiliary_loss"],
        ),
        "event_label_count": int(initial["event_label_count"]),
        "signature_label_count": int(initial["signature_label_count"]),
        "end_gap_label_count": int(initial["end_gap_label_count"]),
        "min_event_label_count_per_batch": int(initial["min_event_label_count_per_batch"]),
        "min_signature_label_count_per_batch": int(initial["min_signature_label_count_per_batch"]),
        "min_end_gap_label_count_per_batch": int(initial["min_end_gap_label_count_per_batch"]),
        "first_step_gradients": first_step_gradients or {},
        "loss_curve_preview": _curve_preview(train_curve),
    }


@torch.inference_mode()
def _evaluate_batches(
    model: MapperV3Model,
    loss_fn: MapperV3ModelLoss,
    batches: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    if not batches:
        raise ValueError("evaluation requires at least one real batch")
    model.eval()
    totals: list[float] = []
    tokens: list[float] = []
    auxiliaries: list[float] = []
    event_counts: list[int] = []
    signature_counts: list[int] = []
    end_gap_counts: list[int] = []
    all_losses_finite = True
    for batch_raw in batches:
        batch = _batch_to_device(batch_raw, device=device)
        output = model(batch)
        loss = loss_fn(output, batch)
        metrics = loss.metrics
        total_loss = float(metrics["loss/total"])
        token_loss = float(metrics["loss/token"])
        aux_loss = float(metrics["loss/delta_event_auxiliary"])
        totals.append(total_loss)
        tokens.append(token_loss)
        auxiliaries.append(aux_loss)
        event_counts.append(int(round(float(metrics["delta_event_auxiliary/event_label_count"]))))
        signature_counts.append(int(round(float(metrics["delta_event_auxiliary/signature_label_count"]))))
        end_gap_counts.append(int(round(float(metrics["delta_event_auxiliary/end_gap_label_count"]))))
        all_losses_finite = all_losses_finite and all(math.isfinite(value) for value in (total_loss, token_loss, aux_loss))
    return {
        "all_losses_finite": bool(all_losses_finite),
        "mean_total_loss": _mean(totals),
        "mean_token_loss": _mean(tokens),
        "mean_delta_event_auxiliary_loss": _mean(auxiliaries),
        "event_label_count": int(sum(event_counts)),
        "signature_label_count": int(sum(signature_counts)),
        "end_gap_label_count": int(sum(end_gap_counts)),
        "min_event_label_count_per_batch": min(event_counts),
        "min_signature_label_count_per_batch": min(signature_counts),
        "min_end_gap_label_count_per_batch": min(end_gap_counts),
    }


def _small_real_data_config(**overrides: Any) -> MapperV3Config:
    values = {
        "control_dim": 384,
        "d_model": 64,
        "heads": 4,
        "layers": 1,
        "ffn_dim": 128,
        "dropout": 0.0,
        "max_seq_len": 1024,
        "state_prior_hidden_dim": 32,
        "ln_close_hidden_dim": 32,
        "lane_embedding_dim": 4,
        "age_embedding_dim": 8,
        "use_global_context": False,
        "global_conv_blocks": 0,
        "use_delta_event_auxiliary_target": False,
        "delta_event_delta_max_ms": 8000,
        "delta_event_end_gap_max_ms": 8000,
    }
    values.update(overrides)
    return MapperV3Config(**values)


def _load_context(
    *,
    label_coverage_summary_path: Path,
    synthetic_training_summary_path: Path,
) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if label_coverage_summary_path.exists():
        coverage = _read_json(label_coverage_summary_path)
        context["fixed_slice_label_coverage"] = {
            "path": label_coverage_summary_path.as_posix(),
            "route": str(_mapping(coverage.get("decision")).get("route") or ""),
            "event_label_count": _mapping(coverage.get("metrics")).get("event_label_count"),
            "end_gap_label_count": _mapping(coverage.get("metrics")).get("end_gap_label_count"),
            "max_end_gap_ms": _mapping(coverage.get("metrics")).get("max_end_gap_ms"),
        }
    if synthetic_training_summary_path.exists():
        synthetic = _read_json(synthetic_training_summary_path)
        context["synthetic_tiny_training"] = {
            "path": synthetic_training_summary_path.as_posix(),
            "route": str(_mapping(synthetic.get("decision")).get("route") or ""),
            "auxiliary_loss_ratio": _mapping(synthetic.get("training")).get("auxiliary_loss_ratio"),
        }
    return context


def _gradient_snapshot(model: MapperV3Model) -> dict[str, float]:
    if model.delta_event_delta_head is None:
        raise RuntimeError("delta-event delta head missing in enabled real-data training probe")
    if model.delta_event_signature_head is None:
        raise RuntimeError("delta-event signature head missing in enabled real-data training probe")
    if model.delta_event_end_gap_head is None:
        raise RuntimeError("delta-event end-gap head missing in enabled real-data training probe")
    return {
        "delta_head_grad_abs": _grad_abs(model.delta_event_delta_head.weight),
        "signature_head_grad_abs": _grad_abs(model.delta_event_signature_head.weight),
        "end_gap_head_grad_abs": _grad_abs(model.delta_event_end_gap_head.weight),
        "token_embedding_grad_abs": _grad_abs(model.token_embedding.weight),
    }


def _batch_to_device(value: Any, *, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, Mapping):
        return {key: _batch_to_device(item, device=device) for key, item in value.items()}
    return value


def _control_dim_from_batches(batches: Sequence[Mapping[str, Any]]) -> int:
    if not batches:
        raise ValueError("control_dim requires at least one batch")
    control_memory = batches[0].get("control_memory_8s")
    if not isinstance(control_memory, torch.Tensor) or control_memory.ndim != 3:
        raise ValueError("real-data gate requires control_memory_8s tensors from the control-teacher cache")
    return int(control_memory.shape[-1])


def _resolve_device(device_name: str) -> torch.device:
    name = str(device_name)
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _validate_run_config(
    *,
    window_limit: int,
    batch_size: int,
    steps: int,
    learning_rate: float,
    lambda_delta_event_auxiliary: float,
    max_auxiliary_loss_ratio: float,
    max_token_loss_ratio: float,
) -> None:
    if int(window_limit) <= 0:
        raise ValueError("window_limit must be positive")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(steps) <= 0:
        raise ValueError("steps must be positive")
    _require_positive_finite(learning_rate, "learning_rate")
    _require_positive_finite(lambda_delta_event_auxiliary, "lambda_delta_event_auxiliary")
    _require_positive_finite(max_auxiliary_loss_ratio, "max_auxiliary_loss_ratio")
    _require_positive_finite(max_token_loss_ratio, "max_token_loss_ratio")


def _require_positive_finite(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite numeric")
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive")


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
        "uv run python -m pulsefield_model.evals.mapper_v3_delta_event_auxiliary_tiny_real_data_training",
        "uv run --group dev pytest tests/evals/test_mapper_v3_delta_event_auxiliary_tiny_real_data_training.py tests/evals/test_mapper_v3_delta_event_auxiliary_fixed_slice_label_coverage.py tests/evals/test_mapper_v3_delta_event_auxiliary_tiny_training_gate.py tests/models/mapper/v3/test_model.py tests/training/test_mapper_v3.py -q",
        "uv run python -m py_compile src/pulsefield_model/evals/mapper_v3_delta_event_auxiliary_tiny_real_data_training.py",
        "python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_auxiliary_tiny_real_data_training_summary.json >/dev/null",
        "git diff --check",
        "```",
    ]


def _curve_preview(curves: Sequence[Mapping[str, float]]) -> list[dict[str, float]]:
    if not curves:
        return []
    indexes = sorted({0, min(1, len(curves) - 1), len(curves) // 2, len(curves) - 1})
    return [dict(curves[index]) for index in indexes]


def _what_passed(route: str) -> str:
    if route == "TEST_DELTA_EVENT_AUXILIARY_FULL_SLICE_OR_FULL4K_LABEL_COVERAGE":
        return (
            "- Real fixed-slice batches carried event, signature, and end-gap labels.\n"
            "- The enabled auxiliary objective optimized under real optimizer steps.\n"
            "- Gradients reached all three auxiliary heads and the shared token embedding.\n"
            "- No tokenizer, dataset schema, default mapper, rollout, C3 replay, or future-lookup change was used."
        )
    return "- The gate produced an explicit route; see failed checks for the limiting evidence."


def _what_surfaced(route: str) -> str:
    if route == "TEST_DELTA_EVENT_AUXILIARY_FULL_SLICE_OR_FULL4K_LABEL_COVERAGE":
        return (
            "The delta-event auxiliary objective is now trainable on real fixed-slice teacher-forced batches. "
            "The remaining structural issue is coverage scale: the prior fixed-slice coverage audit hit the "
            "current `8000ms` end-gap boundary, so a larger/full4k range stress is required before defaults or "
            "replacement work."
        )
    if route == "MUTATE_DELTA_EVENT_AUXILIARY_WEIGHTING":
        return "Real-batch labels and gradients are present, but the current weighting or learning setup did not pass optimization guards."
    return "The auxiliary path failed a hard real-data gate; scaling to rollout would hide the cause."


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


def _grad_abs(parameter: torch.Tensor) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().abs().sum().cpu())


def _safe_ratio(numerator: float, denominator: float) -> float:
    if float(denominator) == 0.0:
        return 0.0 if float(numerator) == 0.0 else float("inf")
    return float(numerator) / float(denominator)


def _fmt(value: object) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(number) else f"{number:.6f}"


def _row(label: str, value: object) -> str:
    return f"| {label} | `{value}` |"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the v3 delta-event auxiliary tiny real-data training gate.")
    parser.add_argument("--index-path", default=DEFAULT_INDEX_PATH)
    parser.add_argument("--control-teacher-cache-dir", default=DEFAULT_CONTROL_TEACHER_CACHE_DIR)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--report-output", default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--window-limit", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--lambda-delta-event-auxiliary", type=float, default=1.0)
    parser.add_argument("--max-auxiliary-loss-ratio", type=float, default=0.75)
    parser.add_argument("--max-token-loss-ratio", type=float, default=1.10)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-cached-timepoint-maps", type=int, default=16)
    args = parser.parse_args(argv)
    summary = run_delta_event_auxiliary_tiny_real_data_training(
        index_path=Path(args.index_path),
        control_teacher_cache_dir=Path(args.control_teacher_cache_dir),
        summary_output_path=Path(args.summary_output),
        report_output_path=Path(args.report_output) if args.report_output else None,
        window_limit=args.window_limit,
        batch_size=args.batch_size,
        steps=args.steps,
        learning_rate=args.learning_rate,
        lambda_delta_event_auxiliary=args.lambda_delta_event_auxiliary,
        max_auxiliary_loss_ratio=args.max_auxiliary_loss_ratio,
        max_token_loss_ratio=args.max_token_loss_ratio,
        seed=args.seed,
        device_name=args.device,
        max_cached_timepoint_maps=args.max_cached_timepoint_maps,
    )
    print(
        "mapper_v3_delta_event_auxiliary_tiny_real_data_training_done "
        f"route={summary['decision']['route']} "
        f"aux_loss_ratio={summary['training']['auxiliary_loss_ratio']:.6f} "
        f"token_loss_ratio={summary['training']['token_loss_ratio']:.6f} "
        f"summary={Path(args.summary_output).as_posix()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
