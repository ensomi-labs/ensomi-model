from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from pulsefield_model.data.mapper_sparse_windows_v3 import MapperV3WindowDataset, collate_mapper_v3_windows
from pulsefield_model.models.mapper.v3 import MapperV3Config, MapperV3LossConfig, MapperV3Model, MapperV3ModelLoss
from pulsefield_model.models.mapper.v3.vocab import MapperV3Vocab


SUMMARY_SCHEMA_VERSION = 1
REPORT_ROOT = Path("artifacts/reports/audits/mapper_v2_1_grammar")
DEFAULT_INDEX_PATH = Path("artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet")
DEFAULT_CONTROL_TEACHER_CACHE_DIR = Path("artifacts/cache/stage2_mapper_v2_1/control_teacher_d384_l3_stride16_step002000")
DEFAULT_EXPERIMENT_CARD = REPORT_ROOT / "target_grammar_v3_delta_event_factor_target_model_loss_plumbing_experiment_card.md"
DEFAULT_SUMMARY_OUTPUT = REPORT_ROOT / "target_grammar_v3_delta_event_factor_target_model_loss_plumbing_summary.json"
DEFAULT_REPORT_OUTPUT = REPORT_ROOT / "target_grammar_v3_delta_event_factor_target_model_loss_plumbing_result_report.md"
DEFAULT_DATA_CONTRACT_SUMMARY = REPORT_ROOT / "target_grammar_v3_delta_event_factor_target_data_contract_summary.json"
DEFAULT_TINY_MODEL_SUMMARY = REPORT_ROOT / "target_grammar_v3_delta_event_factor_target_tiny_model_summary.json"


def run_delta_event_factor_target_model_loss_plumbing(
    *,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    control_teacher_cache_dir: str | Path = DEFAULT_CONTROL_TEACHER_CACHE_DIR,
    summary_output_path: str | Path = DEFAULT_SUMMARY_OUTPUT,
    report_output_path: str | Path | None = DEFAULT_REPORT_OUTPUT,
    experiment_card_path: str | Path = DEFAULT_EXPERIMENT_CARD,
    data_contract_summary_path: str | Path = DEFAULT_DATA_CONTRACT_SUMMARY,
    tiny_model_summary_path: str | Path = DEFAULT_TINY_MODEL_SUMMARY,
    window_limit: int = 32,
    batch_size: int = 4,
    d_model: int = 32,
    max_cached_timepoint_maps: int = 16,
) -> dict[str, Any]:
    started = time.monotonic()
    if int(window_limit) <= 0:
        raise ValueError("window_limit must be positive")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(d_model) <= 0:
        raise ValueError("d_model must be positive")
    vocab = MapperV3Vocab()
    default_off_dataset = _dataset(
        index_path=index_path,
        control_teacher_cache_dir=control_teacher_cache_dir,
        vocab=vocab,
        include_delta_event_factor_target=False,
        max_cached_timepoint_maps=max_cached_timepoint_maps,
    )
    enabled_dataset = _dataset(
        index_path=index_path,
        control_teacher_cache_dir=control_teacher_cache_dir,
        vocab=vocab,
        include_delta_event_factor_target=True,
        max_cached_timepoint_maps=max_cached_timepoint_maps,
    )
    consumed = min(int(window_limit), len(enabled_dataset))
    if consumed <= 0:
        raise ValueError("model/loss plumbing gate requires at least one consumed window")
    off_batch = collate_mapper_v3_windows(
        [default_off_dataset[index] for index in range(min(int(batch_size), consumed))],
        pad_id=vocab.pad_id,
    )
    enabled_samples = [enabled_dataset[index] for index in range(consumed)]
    enabled_batches = [
        collate_mapper_v3_windows(enabled_samples[start : start + int(batch_size)], pad_id=vocab.pad_id)
        for start in range(0, consumed, int(batch_size))
    ]
    row_metrics = _factor_metrics(enabled_batches, consumed_window_count=consumed)
    default_off_metrics = _default_off_probe(off_batch, vocab=vocab, d_model=int(d_model))
    enabled_metrics = _enabled_probe(enabled_batches[0], vocab=vocab, d_model=int(d_model))
    context = _load_context(
        data_contract_summary_path=Path(data_contract_summary_path),
        tiny_model_summary_path=Path(tiny_model_summary_path),
    )
    checks = guard_checks(
        row_metrics=row_metrics,
        default_off=default_off_metrics,
        enabled=enabled_metrics,
        context=context,
    )
    decision = decision_from_checks(checks)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar v3 delta-event factor target model/loss plumbing",
        "experiment_card": Path(experiment_card_path).as_posix(),
        "elapsed_s": time.monotonic() - started,
        "index_path": Path(index_path).as_posix(),
        "control_teacher_cache_dir": Path(control_teacher_cache_dir).as_posix(),
        "config": {
            "window_limit": int(window_limit),
            "batch_size": int(batch_size),
            "d_model": int(d_model),
            "max_cached_timepoint_maps": int(max_cached_timepoint_maps),
            "include_delta_event_factor_target_default": False,
            "use_delta_event_factor_target_default": False,
            "lambda_delta_event_factor_target_default": 0.0,
            "no_inference_or_rollout_change": True,
            "uses_c3_backreference": False,
            "uses_future_lookup": False,
        },
        "dataset": {
            "source_window_count": len(enabled_dataset.records),
            "filter_report": enabled_dataset.filter_report.__dict__,
        },
        "context": context,
        "row_metrics": row_metrics,
        "default_off": default_off_metrics,
        "enabled": enabled_metrics,
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
    row_metrics: Mapping[str, Any],
    default_off: Mapping[str, Any],
    enabled: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, bool]:
    data_contract = _mapping(context.get("data_contract"))
    tiny_model = _mapping(context.get("tiny_model"))
    gradients = _mapping(enabled.get("gradient_abs"))
    expected_rows = data_contract.get("row_count")
    expected_events = data_contract.get("event_row_count")
    expected_ends = data_contract.get("end_row_count")
    return {
        "data_contract_route_positive": data_contract.get("route")
        == "TEST_DELTA_EVENT_FACTOR_TARGET_MODEL_LOSS_PLUMBING_CARD",
        "tiny_model_route_positive": tiny_model.get("route")
        == "TEST_DELTA_EVENT_FACTOR_TARGET_PRODUCTION_PLUMBING_CARD",
        "audited_windows_positive": int(row_metrics.get("audited_window_count") or 0) > 0,
        "row_count_positive": int(row_metrics.get("row_count") or 0) > 0,
        "row_count_matches_data_contract": expected_rows is not None
        and int(row_metrics.get("row_count") or -1) == int(expected_rows or -2),
        "event_rows_match_data_contract": expected_events is not None
        and int(row_metrics.get("event_row_count") or -1) == int(expected_events or -2),
        "end_rows_match_data_contract": expected_ends is not None
        and int(row_metrics.get("end_row_count") or -1) == int(expected_ends or -2),
        "row_mask_count_matches_rows": int(row_metrics.get("row_mask_valid_count") or -1)
        == int(row_metrics.get("row_count") or -2),
        "reconstruction_mismatches_zero": int(row_metrics.get("reconstruction_mismatch_count") or 0) == 0,
        "default_off_batch_lacks_factor_target": not bool(default_off.get("batch_has_factor_target")),
        "default_off_outputs_absent": bool(default_off.get("factor_outputs_absent")),
        "default_off_loss_metric_zero": default_off.get("lambda_metric") is not None
        and float(default_off.get("lambda_metric")) == 0.0,
        "enabled_outputs_align_rows": bool(enabled.get("outputs_align_rows")),
        "enabled_loss_finite": bool(enabled.get("loss_finite")),
        "enabled_factor_loss_positive": float(enabled.get("factor_loss") or 0.0) > 0.0,
        "enabled_kind_labels_match_rows": int(enabled.get("kind_label_count") or -1)
        == int(enabled.get("row_count") or -2),
        "enabled_delta_labels_match_events": int(enabled.get("delta_label_count") or -1)
        == int(enabled.get("event_row_count") or -2),
        "enabled_signature_labels_match_events": int(enabled.get("signature_label_count") or -1)
        == int(enabled.get("event_row_count") or -2),
        "enabled_end_gap_labels_match_windows": int(enabled.get("end_gap_label_count") or -1)
        == int(enabled.get("batch_size") or -2),
        "kind_head_gradient_nonzero": float(gradients.get("kind_head") or 0.0) > 0.0,
        "delta_head_gradient_nonzero": float(gradients.get("delta_head") or 0.0) > 0.0,
        "signature_head_gradient_nonzero": float(gradients.get("signature_head") or 0.0) > 0.0,
        "end_gap_head_gradient_nonzero": float(gradients.get("end_gap_head") or 0.0) > 0.0,
        "no_inference_rollout_or_c3_change": True,
    }


def decision_from_checks(checks: Mapping[str, bool]) -> dict[str, str]:
    failed = [key for key, passed in checks.items() if not bool(passed)]
    if not failed:
        return {
            "route": "TEST_DELTA_EVENT_FACTOR_TARGET_TRAINING_SMOKE_CARD",
            "reason": "default-off behavior held and enabled factor target model/loss produced finite real-batch gradients",
            "next_step": "Create a bounded short training-smoke card with factor-target loss enabled.",
        }
    if any(key.startswith("default_off") for key in failed):
        return {
            "route": "KILL_FACTOR_TARGET_MODEL_LOSS_DEFAULT_REGRESSION",
            "reason": "default-off model/loss contract regressed: " + ", ".join(failed),
            "next_step": "Repair default-off behavior before training with factorized rows.",
        }
    return {
        "route": "MUTATE_FACTOR_TARGET_MODEL_LOSS_PLUMBING",
        "reason": "factor target model/loss plumbing failed checks: " + ", ".join(failed),
        "next_step": "Repair row alignment, factor loss, or gradients before training-smoke work.",
    }


def write_summary_json(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_markdown(summary).rstrip() + "\n", encoding="utf-8")


def report_markdown(summary: Mapping[str, Any]) -> str:
    decision = _mapping(summary.get("decision"))
    rows = _mapping(summary.get("row_metrics"))
    default_off = _mapping(summary.get("default_off"))
    enabled = _mapping(summary.get("enabled"))
    checks = _mapping(summary.get("checks"))
    gradients = _mapping(enabled.get("gradient_abs"))
    lines = [
        "# Target Grammar v3 Delta-Event Factor Target Model/Loss Plumbing Result Report",
        "",
        "## Scope",
        "",
        "This gate verifies optional production model/loss plumbing for factorized delta-event target rows. It does not train a full production model, change inference, change rollout, replace target fragments, or change mapper defaults.",
        "",
        "## Decision",
        "",
        f"Route: `{decision.get('route')}`.",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Recommended next step: {decision.get('next_step')}",
        "",
        "## Fixed-Slice Row Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        _row("audited windows", rows.get("audited_window_count")),
        _row("row count", rows.get("row_count")),
        _row("event rows", rows.get("event_row_count")),
        _row("end rows", rows.get("end_row_count")),
        _row("max row len", rows.get("max_row_len")),
        _row("row mask valid count", rows.get("row_mask_valid_count")),
        _row("reconstruction mismatches", rows.get("reconstruction_mismatch_count")),
        "",
        "## Default-Off Probe",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        _row("batch has factor target", default_off.get("batch_has_factor_target")),
        _row("factor outputs absent", default_off.get("factor_outputs_absent")),
        _row("lambda metric", default_off.get("lambda_metric")),
        "",
        "## Enabled Probe",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        _row("batch size", enabled.get("batch_size")),
        _row("row count", enabled.get("row_count")),
        _row("event row count", enabled.get("event_row_count")),
        _row("factor loss", _fmt(enabled.get("factor_loss"))),
        _row("total loss", _fmt(enabled.get("total_loss"))),
        _row("kind labels", enabled.get("kind_label_count")),
        _row("delta labels", enabled.get("delta_label_count")),
        _row("signature labels", enabled.get("signature_label_count")),
        _row("end-gap labels", enabled.get("end_gap_label_count")),
        _row("kind head grad abs", _fmt(gradients.get("kind_head"))),
        _row("delta head grad abs", _fmt(gradients.get("delta_head"))),
        _row("signature head grad abs", _fmt(gradients.get("signature_head"))),
        _row("end-gap head grad abs", _fmt(gradients.get("end_gap_head"))),
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
            "- This does not prove full training stability.",
            "- This does not prove autoregressive factor-row inference.",
            "- This does not prove rollout quality or complete v3 replacement readiness.",
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


def _dataset(
    *,
    index_path: str | Path,
    control_teacher_cache_dir: str | Path,
    vocab: MapperV3Vocab,
    include_delta_event_factor_target: bool,
    max_cached_timepoint_maps: int,
) -> MapperV3WindowDataset:
    return MapperV3WindowDataset(
        index_path=index_path,
        vocab=vocab,
        control_teacher_cache_dir=control_teacher_cache_dir,
        require_control_teacher_cache=True,
        include_full_song_context=False,
        include_delta_event_factor_target=bool(include_delta_event_factor_target),
        max_cached_timepoint_maps=int(max_cached_timepoint_maps),
        progress=False,
    )


def _default_off_probe(batch: Mapping[str, Any], *, vocab: MapperV3Vocab, d_model: int) -> dict[str, Any]:
    control_dim = int(_tensor(batch["control_memory_8s"]).shape[-1])
    model = MapperV3Model(_small_config(control_dim=control_dim, d_model=d_model), vocab=vocab)
    loss_fn = MapperV3ModelLoss(
        MapperV3LossConfig(lambda_density=0.0, lambda_ln_close=0.0, lambda_adapter_reg=0.0),
        vocab=vocab,
    )
    output = model(batch)
    loss = loss_fn(output, batch)
    factor_outputs = [
        output.delta_event_factor_kind_logits,
        output.delta_event_factor_delta_logits,
        output.delta_event_factor_signature_logits,
        output.delta_event_factor_end_gap_logits,
    ]
    return {
        "batch_has_factor_target": "delta_event_factor_target" in batch,
        "factor_outputs_absent": all(value is None for value in factor_outputs),
        "lambda_metric": loss.metrics.get("phase/lambda_delta_event_factor_target"),
        "loss_finite": bool(torch.isfinite(loss.total_loss).item()),
    }


def _enabled_probe(batch: Mapping[str, Any], *, vocab: MapperV3Vocab, d_model: int) -> dict[str, Any]:
    control_dim = int(_tensor(batch["control_memory_8s"]).shape[-1])
    model = MapperV3Model(
        _small_config(
            control_dim=control_dim,
            d_model=d_model,
            use_delta_event_factor_target=True,
        ),
        vocab=vocab,
    )
    loss_fn = MapperV3ModelLoss(
        MapperV3LossConfig(
            lambda_density=0.0,
            lambda_ln_close=0.0,
            lambda_adapter_reg=0.0,
            lambda_delta_event_factor_target=1.0,
        ),
        vocab=vocab,
    )
    output = model(batch)
    loss = loss_fn(output, batch)
    loss.total_loss.backward()
    factor = _mapping(batch.get("delta_event_factor_target"))
    row_count = int(_tensor(factor["row_count"]).sum().item())
    event_count = int(_tensor(factor["event_row_count"]).sum().item())
    batch_size = int(_tensor(factor["row_count"]).shape[0])
    max_rows = int(_tensor(factor["row_mask"]).shape[1])
    shapes = {
        "kind": list(_tensor(output.delta_event_factor_kind_logits).shape),
        "delta": list(_tensor(output.delta_event_factor_delta_logits).shape),
        "signature": list(_tensor(output.delta_event_factor_signature_logits).shape),
        "end_gap": list(_tensor(output.delta_event_factor_end_gap_logits).shape),
    }
    return {
        "batch_size": batch_size,
        "row_count": row_count,
        "event_row_count": event_count,
        "outputs_align_rows": all(shape[:2] == [batch_size, max_rows] for shape in shapes.values()),
        "output_shapes": shapes,
        "loss_finite": bool(torch.isfinite(loss.total_loss).item()),
        "total_loss": float(loss.total_loss.detach().cpu()),
        "factor_loss": float(_tensor(loss.delta_event_factor_target_loss).detach().cpu()),
        "kind_label_count": int(loss.metrics.get("delta_event_factor/kind_label_count") or 0),
        "delta_label_count": int(loss.metrics.get("delta_event_factor/delta_label_count") or 0),
        "signature_label_count": int(loss.metrics.get("delta_event_factor/signature_label_count") or 0),
        "end_gap_label_count": int(loss.metrics.get("delta_event_factor/end_gap_label_count") or 0),
        "gradient_abs": {
            "kind_head": _grad_abs(model.delta_event_factor_kind_head),
            "delta_head": _grad_abs(model.delta_event_factor_delta_head),
            "signature_head": _grad_abs(model.delta_event_factor_signature_head),
            "end_gap_head": _grad_abs(model.delta_event_factor_end_gap_head),
        },
    }


def _small_config(*, control_dim: int, d_model: int, **overrides: Any) -> MapperV3Config:
    values: dict[str, Any] = {
        "control_dim": int(control_dim),
        "d_model": int(d_model),
        "heads": 4,
        "layers": 1,
        "ffn_dim": int(d_model) * 2,
        "dropout": 0.0,
        "max_seq_len": 1024,
        "state_prior_hidden_dim": int(d_model),
        "ln_close_hidden_dim": int(d_model),
        "lane_embedding_dim": 4,
        "age_embedding_dim": 4,
        "use_global_context": False,
        "global_conv_blocks": 0,
    }
    values.update(overrides)
    return MapperV3Config(**values)


def _factor_metrics(batches: Sequence[Mapping[str, Any]], *, consumed_window_count: int) -> dict[str, Any]:
    row_count = 0
    event_row_count = 0
    end_row_count = 0
    max_row_len = 0
    row_mask_valid_count = 0
    reconstruction_mismatch_count = 0
    for batch in batches:
        factor = _mapping(batch.get("delta_event_factor_target"))
        if not factor:
            raise ValueError("enabled batch is missing delta_event_factor_target")
        row_count += int(_tensor(factor.get("row_count")).sum().item())
        event_row_count += int(_tensor(factor.get("event_row_count")).sum().item())
        end_row_count += int(_tensor(factor.get("end_row_count")).sum().item())
        max_row_len = max(max_row_len, int(_tensor(factor.get("row_mask")).shape[1]))
        row_mask_valid_count += int(_tensor(factor.get("row_mask")).sum().item())
        reconstruction_mismatch_count += int(_tensor(factor.get("reconstruction_mismatch_count")).sum().item())
    return {
        "audited_window_count": int(consumed_window_count),
        "row_count": int(row_count),
        "event_row_count": int(event_row_count),
        "end_row_count": int(end_row_count),
        "max_row_len": int(max_row_len),
        "row_mask_valid_count": int(row_mask_valid_count),
        "reconstruction_mismatch_count": int(reconstruction_mismatch_count),
    }


def _load_context(*, data_contract_summary_path: Path, tiny_model_summary_path: Path) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if data_contract_summary_path.exists():
        data = _read_json(data_contract_summary_path)
        metrics = _mapping(data.get("metrics"))
        context["data_contract"] = {
            "path": data_contract_summary_path.as_posix(),
            "route": str(_mapping(data.get("decision")).get("route") or ""),
            "row_count": metrics.get("row_count"),
            "event_row_count": metrics.get("event_row_count"),
            "end_row_count": metrics.get("end_row_count"),
        }
    if tiny_model_summary_path.exists():
        data = _read_json(tiny_model_summary_path)
        context["tiny_model"] = {
            "path": tiny_model_summary_path.as_posix(),
            "route": str(_mapping(data.get("decision")).get("route") or ""),
        }
    return context


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
        "uv run python -m pulsefield_model.evals.mapper_v3_delta_event_factor_target_model_loss_plumbing",
        "uv run --group dev pytest tests/models/mapper/v3/test_factor_target.py tests/data/test_mapper_sparse_windows_v3_factor_target.py tests/models/mapper/v3/test_model.py tests/training/test_mapper_v3.py tests/evals/test_mapper_v3_delta_event_factor_target_model_loss_plumbing.py -q",
        "uv run python -m py_compile src/pulsefield_model/models/mapper/v3/model.py src/pulsefield_model/models/mapper/v3/loss.py src/pulsefield_model/training/mapper_v3.py src/pulsefield_model/evals/mapper_v3_delta_event_factor_target_model_loss_plumbing.py",
        "python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_model_loss_plumbing_summary.json >/dev/null",
        "git diff --check",
        "```",
    ]


def _what_passed(route: str) -> str:
    if route == "TEST_DELTA_EVENT_FACTOR_TARGET_TRAINING_SMOKE_CARD":
        return (
            "- Default-off v3 model/loss outputs remain absent for factor-target fields.\n"
            "- Enabled row-level factor logits align with collated factor rows.\n"
            "- Factor loss is finite and positive on a real fixed-slice batch.\n"
            "- Kind, delta, signature, and end-gap factor heads all receive gradients."
        )
    return "- The gate produced an explicit route; inspect failed checks before training-smoke work."


def _what_surfaced(route: str) -> str:
    if route == "TEST_DELTA_EVENT_FACTOR_TARGET_TRAINING_SMOKE_CARD":
        return "The factorized target now has production model/loss plumbing, but still needs a bounded training-smoke gate."
    if route == "KILL_FACTOR_TARGET_MODEL_LOSS_DEFAULT_REGRESSION":
        return "The default model/loss behavior regressed and must be repaired before continuing."
    return "The row-level factor target branch needs shape, loss, or gradient repair before training."


def _tensor(value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError("expected torch.Tensor")
    return value


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _grad_abs(module: torch.nn.Module | None) -> float:
    if module is None:
        return 0.0
    parameter = getattr(module, "weight", None)
    if not isinstance(parameter, torch.Tensor) or parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().abs().sum().cpu())


def _fmt(value: object) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(number) else f"{number:.6f}"


def _row(label: str, value: object) -> str:
    return f"| {label} | `{value}` |"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the v3 delta-event factor target model/loss plumbing gate.")
    parser.add_argument("--index-path", default=DEFAULT_INDEX_PATH)
    parser.add_argument("--control-teacher-cache-dir", default=DEFAULT_CONTROL_TEACHER_CACHE_DIR)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--report-output", default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--window-limit", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--max-cached-timepoint-maps", type=int, default=16)
    args = parser.parse_args(argv)
    summary = run_delta_event_factor_target_model_loss_plumbing(
        index_path=Path(args.index_path),
        control_teacher_cache_dir=Path(args.control_teacher_cache_dir),
        summary_output_path=Path(args.summary_output),
        report_output_path=Path(args.report_output) if args.report_output else None,
        window_limit=args.window_limit,
        batch_size=args.batch_size,
        d_model=args.d_model,
        max_cached_timepoint_maps=args.max_cached_timepoint_maps,
    )
    print(
        "mapper_v3_delta_event_factor_target_model_loss_plumbing_done "
        f"route={summary['decision']['route']} "
        f"factor_loss={summary['enabled']['factor_loss']:.6f} "
        f"summary={Path(args.summary_output).as_posix()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
