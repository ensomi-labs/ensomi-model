from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pulsefield_model.data.control_windows import DEFAULT_DATASET_ROOT, ControlWindowDataset
from pulsefield_model.data.mapper_sparse_windows_v3 import is_mapper_v3_window_start_allowed
from pulsefield_model.evals.mapper_v3_delta_event_auxiliary_fixed_slice_label_coverage import window_label_coverage
from pulsefield_model.models.mapper.v3.tokenizer import (
    MAPPER_WRITE_MS,
    UnsupportedMapperActionError,
    encode_mapper_window,
    hitobjects_to_mapper_timepoints,
    mapper_chart_end_ms,
)
from pulsefield_model.models.mapper.v3.vocab import MapperV3Vocab
from pulsefield_model.osu_core.hitobjects import parse_mania_hit_objects


SUMMARY_SCHEMA_VERSION = 1
REPORT_ROOT = Path("artifacts/reports/audits/mapper_v2_1_grammar")
DEFAULT_INDEX_PATH = Path("artifacts/indexes/stage2_control_windows_4k_2to6_dense_local_bpm_norm_unique_le3.parquet")
DEFAULT_EXPERIMENT_CARD = REPORT_ROOT / "target_grammar_v3_delta_event_auxiliary_full4k_label_coverage_experiment_card.md"
DEFAULT_SUMMARY_OUTPUT = REPORT_ROOT / "target_grammar_v3_delta_event_auxiliary_full4k_label_coverage_summary.json"
DEFAULT_REPORT_OUTPUT = REPORT_ROOT / "target_grammar_v3_delta_event_auxiliary_full4k_label_coverage_result_report.md"
DEFAULT_V3_FULL_DATASET_SUMMARY_PATH = REPORT_ROOT / "target_grammar_v3_full_dataset_audit_summary.json"
DEFAULT_FIXED_SLICE_SUMMARY_PATH = REPORT_ROOT / "target_grammar_v3_delta_event_auxiliary_fixed_slice_label_coverage_summary.json"
DEFAULT_REAL_DATA_TRAINING_SUMMARY_PATH = REPORT_ROOT / "target_grammar_v3_delta_event_auxiliary_tiny_real_data_training_summary.json"


def run_delta_event_auxiliary_full4k_label_coverage(
    *,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    summary_output_path: str | Path = DEFAULT_SUMMARY_OUTPUT,
    report_output_path: str | Path | None = DEFAULT_REPORT_OUTPUT,
    experiment_card_path: str | Path = DEFAULT_EXPERIMENT_CARD,
    v3_full_dataset_summary_path: str | Path = DEFAULT_V3_FULL_DATASET_SUMMARY_PATH,
    fixed_slice_summary_path: str | Path = DEFAULT_FIXED_SLICE_SUMMARY_PATH,
    real_data_training_summary_path: str | Path = DEFAULT_REAL_DATA_TRAINING_SUMMARY_PATH,
    delta_max_ms: int = 8000,
    end_gap_max_ms: int = 8000,
    limit: int | None = None,
    max_cached_timepoint_maps: int = 64,
    progress_interval: int = 0,
    dataset_progress: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    _validate_range(delta_max_ms, name="delta_max_ms")
    _validate_range(end_gap_max_ms, name="end_gap_max_ms")
    if limit is not None and int(limit) <= 0:
        raise ValueError("limit must be positive or omitted for full scope")
    if int(max_cached_timepoint_maps) < 0:
        raise ValueError("max_cached_timepoint_maps must be non-negative")
    if int(progress_interval) < 0:
        raise ValueError("progress_interval must be non-negative")

    vocab = MapperV3Vocab()
    dataset = ControlWindowDataset(
        index_path=index_path,
        dataset_root=dataset_root,
        progress=bool(dataset_progress),
    )
    aggregate = _empty_aggregate()
    timepoints_cache: OrderedDict[str, tuple[Any, ...]] = OrderedDict()
    errors: list[dict[str, Any]] = []
    unsupported_errors: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    candidate_window_count = 0
    skipped_stride_window_count = 0
    last_source_index = -1
    for source_index, record in enumerate(dataset.records):
        last_source_index = int(source_index)
        if not is_mapper_v3_window_start_allowed(record):
            skipped_stride_window_count += 1
            continue
        if limit is not None and candidate_window_count >= int(limit):
            break
        candidate_window_count += 1
        try:
            tokenized = _tokenize_record_once(
                record,
                vocab=vocab,
                timepoints_cache=timepoints_cache,
                max_cached_timepoint_maps=int(max_cached_timepoint_maps),
            )
            coverage = window_label_coverage(
                token_ids=tokenized.target_fragment_ids,
                current_ms=tuple(int(value) for value in tokenized.target_fragment_current_ms.tolist()),
                write_start_ms=int(tokenized.write_start_ms),
                write_end_ms=int(tokenized.write_end_ms),
                chart_end_ms=int(tokenized.chart_end_ms),
                is_full_chart_end=bool(tokenized.is_full_chart_end),
                vocab=vocab,
                delta_max_ms=int(delta_max_ms),
                end_gap_max_ms=int(end_gap_max_ms),
            )
        except UnsupportedMapperActionError as exc:
            unsupported_errors.append(
                {
                    "source_index": int(source_index),
                    "candidate_window_index": int(candidate_window_count - 1),
                    "beatmap_path": record.beatmap_path.as_posix(),
                    "target_start_ms": int(record.target_start_ms),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            _maybe_print_progress(
                progress_interval=progress_interval,
                aggregate=aggregate,
                candidate_window_count=candidate_window_count,
                source_index=source_index,
                source_window_count=len(dataset.records),
                error_count=len(errors),
                started=started,
            )
            continue
        except Exception as exc:  # noqa: BLE001 - full-scope report needs concrete per-record failures.
            errors.append(
                {
                    "source_index": int(source_index),
                    "candidate_window_index": int(candidate_window_count - 1),
                    "beatmap_path": record.beatmap_path.as_posix(),
                    "target_start_ms": int(record.target_start_ms),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            _maybe_print_progress(
                progress_interval=progress_interval,
                aggregate=aggregate,
                candidate_window_count=candidate_window_count,
                source_index=source_index,
                source_window_count=len(dataset.records),
                error_count=len(errors),
                started=started,
            )
            continue
        _accumulate(aggregate, coverage)
        if len(examples) < 16 and _coverage_has_failure(coverage):
            examples.append(
                {
                    "source_index": int(source_index),
                    "candidate_window_index": int(candidate_window_count - 1),
                    "beatmap_path": record.beatmap_path.as_posix(),
                    "target_start_ms": int(record.target_start_ms),
                    "coverage": coverage,
                }
            )
        _maybe_print_progress(
            progress_interval=progress_interval,
            aggregate=aggregate,
            candidate_window_count=candidate_window_count,
            source_index=source_index,
            source_window_count=len(dataset.records),
            error_count=len(errors),
            started=started,
        )

    metrics = _finalize_metrics(
        aggregate,
        candidate_window_count=candidate_window_count,
        source_window_count=len(dataset.records),
        skipped_stride_window_count=skipped_stride_window_count,
        last_source_index=last_source_index,
        errors=errors,
        unsupported_errors=unsupported_errors,
        limit=limit,
    )
    context = _load_context(
        v3_full_dataset_summary_path=Path(v3_full_dataset_summary_path),
        fixed_slice_summary_path=Path(fixed_slice_summary_path),
        real_data_training_summary_path=Path(real_data_training_summary_path),
    )
    checks = guard_checks(metrics=metrics, context=context)
    decision = decision_from_checks(checks)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar v3 delta-event auxiliary full4k label coverage",
        "experiment_card": Path(experiment_card_path).as_posix(),
        "elapsed_s": time.monotonic() - started,
        "index_path": Path(index_path).as_posix(),
        "dataset_root": Path(dataset_root).as_posix(),
        "config": {
            "delta_max_ms": int(delta_max_ms),
            "end_gap_max_ms": int(end_gap_max_ms),
            "limit": None if limit is None else int(limit),
            "max_cached_timepoint_maps": int(max_cached_timepoint_maps),
            "progress_interval": int(progress_interval),
            "dataset_progress": bool(dataset_progress),
            "no_training": True,
            "no_rollout": True,
            "tokenizer_changed": False,
            "dataset_schema_changed": False,
            "default_behavior_changed": False,
            "uses_c3_backreference": False,
            "uses_future_lookup": False,
        },
        "dataset": {
            "control_source_row_count": int(getattr(dataset, "source_row_count", len(dataset.records))),
            "control_filtered_row_count": int(getattr(dataset, "filtered_row_count", len(dataset.records))),
            "control_source_map_count": int(getattr(dataset, "source_map_count", 0)),
            "control_filtered_map_count": int(getattr(dataset, "filtered_map_count", 0)),
        },
        "context": context,
        "metrics": metrics,
        "checks": checks,
        "decision": decision,
        "next_step": decision["next_step"],
        "examples": examples,
        "errors": errors[:32],
        "unsupported_examples": unsupported_errors[:32],
    }
    write_summary_json(summary, Path(summary_output_path))
    if report_output_path is not None:
        write_report(summary, Path(report_output_path))
    return summary


def guard_checks(*, metrics: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "no_training": True,
        "no_rollout": True,
        "no_tokenizer_dataset_default_change": True,
        "no_c3_backreference_or_future_lookup": True,
        "full_scope": metrics.get("limit") is None,
        "full_v3_representation_audit_positive": bool(
            _mapping(_mapping(context.get("v3_full_dataset_audit")).get("decision")).get("positive_gate_passed")
        )
        or _mapping(_mapping(context.get("v3_full_dataset_audit")).get("decision")).get("route") == "TEST_NEXT",
        "fixed_slice_label_coverage_positive": _mapping(context.get("fixed_slice_label_coverage")).get("route")
        == "TEST_DELTA_EVENT_AUXILIARY_TINY_REAL_DATA_TRAINING",
        "tiny_real_data_training_positive": _mapping(context.get("tiny_real_data_training")).get("route")
        == "TEST_DELTA_EVENT_AUXILIARY_FULL_SLICE_OR_FULL4K_LABEL_COVERAGE",
        "audited_windows_match_full_v3_audit": _expected_full_v3_audit_window_count(context) is not None
        and int(metrics.get("audited_window_count") or -1) == int(_expected_full_v3_audit_window_count(context) or -2),
        "candidate_windows_positive": int(metrics.get("candidate_window_count") or 0) > 0,
        "audited_windows_positive": int(metrics.get("audited_window_count") or 0) > 0,
        "label_errors_zero": int(metrics.get("label_error_count") or 0) == 0,
        "negative_deltas_zero": int(metrics.get("negative_delta_count") or 0) == 0,
        "negative_end_gaps_zero": int(metrics.get("negative_end_gap_count") or 0) == 0,
        "delta_out_of_range_zero": int(metrics.get("out_of_range_delta_count") or 0) == 0,
        "end_gap_out_of_range_zero": int(metrics.get("out_of_range_end_gap_count") or 0) == 0,
        "event_labels_match_event_tokens": int(metrics.get("event_label_count") or -1)
        == int(metrics.get("event_token_count") or -2),
        "signature_labels_match_event_tokens": int(metrics.get("signature_label_count") or -1)
        == int(metrics.get("event_token_count") or -2),
        "end_gap_labels_match_audited_windows": int(metrics.get("end_gap_label_count") or -1)
        == int(metrics.get("audited_window_count") or -2),
        "event_labels_positive": int(metrics.get("event_label_count") or 0) > 0,
        "end_gap_labels_positive": int(metrics.get("end_gap_label_count") or 0) > 0,
    }


def decision_from_checks(checks: Mapping[str, bool]) -> dict[str, str]:
    failed = [key for key, passed in checks.items() if not bool(passed)]
    if not failed:
        return {
            "route": "TEST_DELTA_EVENT_FULL4K_BIT_PROXY_OR_FACTOR_TARGET_CARD",
            "reason": "full4k delta-event auxiliary label coverage passed under current head ranges",
            "next_step": "Create a full4k delta-event bit-proxy or factor-target card before changing mapper defaults.",
        }
    runtime_failures = {
        "full_scope",
        "candidate_windows_positive",
        "audited_windows_positive",
        "audited_windows_match_full_v3_audit",
    }
    range_failures = {"delta_out_of_range_zero", "end_gap_out_of_range_zero"}
    context_failures = {
        "full_v3_representation_audit_positive",
        "fixed_slice_label_coverage_positive",
        "tiny_real_data_training_positive",
    }
    if any(key in runtime_failures for key in failed):
        return {
            "route": "MUTATE_FULL4K_LABEL_AUDIT_RUNTIME",
            "reason": "full4k label audit did not complete full-scope evidence: " + ", ".join(failed),
            "next_step": "Fix audit runtime/completeness before interpreting label coverage.",
        }
    if any(key in range_failures for key in failed):
        return {
            "route": "MUTATE_DELTA_EVENT_HEAD_RANGE_OR_BUCKETING",
            "reason": "full4k label range checks failed: " + ", ".join(failed),
            "next_step": "Mutate delta/end-gap range or bucket semantics before more training.",
        }
    if any(key in context_failures for key in failed):
        return {
            "route": "KILL_DELTA_EVENT_AUXILIARY_FULL4K_LABEL_SURFACE",
            "reason": "required prior evidence is missing or negative: " + ", ".join(failed),
            "next_step": "Repair the prior gate chain before interpreting full4k label coverage.",
        }
    return {
        "route": "KILL_DELTA_EVENT_AUXILIARY_FULL4K_LABEL_SURFACE",
        "reason": "full4k label semantics failed hard checks: " + ", ".join(failed),
        "next_step": "Repair label extraction semantics before training or factor-target work.",
    }


def write_summary_json(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_markdown(summary).rstrip() + "\n", encoding="utf-8")


def report_markdown(summary: Mapping[str, Any]) -> str:
    decision = _mapping(summary.get("decision"))
    metrics = _mapping(summary.get("metrics"))
    checks = _mapping(summary.get("checks"))
    config = _mapping(summary.get("config"))
    lines = [
        "# Target Grammar v3 Delta-Event Auxiliary Full4K Label Coverage Result Report",
        "",
        "## Scope",
        "",
        "This one-pass full4k audit derives delta-event auxiliary labels from current v3 target fragments. It does not train, roll out, change tokenizer behavior, change dataset schema, change mapper defaults, or use C3/future context.",
        "",
        "## Decision",
        "",
        f"Route: `{decision.get('route')}`.",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Recommended next step: {decision.get('next_step')}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        _row("source windows", metrics.get("source_window_count")),
        _row("candidate windows", metrics.get("candidate_window_count")),
        _row("audited windows", metrics.get("audited_window_count")),
        _row("unsupported skipped windows", metrics.get("unsupported_window_count")),
        _row("label error count", metrics.get("label_error_count")),
        _row("v3 token count", metrics.get("v3_token_count")),
        _row("time-shift token count", metrics.get("time_shift_token_count")),
        _row("event token count", metrics.get("event_token_count")),
        _row("event label count", metrics.get("event_label_count")),
        _row("signature label count", metrics.get("signature_label_count")),
        _row("end-gap label count", metrics.get("end_gap_label_count")),
        _row("out-of-range delta count", metrics.get("out_of_range_delta_count")),
        _row("out-of-range end-gap count", metrics.get("out_of_range_end_gap_count")),
        _row("negative delta count", metrics.get("negative_delta_count")),
        _row("negative end-gap count", metrics.get("negative_end_gap_count")),
        _row("max event delta ms", metrics.get("max_event_delta_ms")),
        _row("max end-gap ms", metrics.get("max_end_gap_ms")),
        _row("unique event deltas", metrics.get("unique_event_delta_count")),
        _row("unique end gaps", metrics.get("unique_end_gap_count")),
        "",
        "## Checks",
        "",
        "| Check | Value |",
        "| --- | ---: |",
    ]
    lines.extend(_row(key, value) for key, value in checks.items())
    lines.extend(["", "## Top Event Deltas", "", "| delta ms | count |", "| ---: | ---: |"])
    lines.extend(_counter_rows(metrics.get("top_event_deltas")))
    lines.extend(["", "## Top End Gaps", "", "| end gap ms | count |", "| ---: | ---: |"])
    lines.extend(_counter_rows(metrics.get("top_end_gaps")))
    lines.extend(["", "## What Passed", "", _what_passed(str(decision.get("route")))])
    lines.extend(
        [
            "",
            "## What Surfaced",
            "",
            _what_surfaced(str(decision.get("route")), metrics=metrics, config=config),
            "",
            "## What Is Not Proved",
            "",
            "- This does not prove trained mapper quality.",
            "- This does not prove autoregressive rollout quality.",
            "- This does not by itself make delta-event factorization the generated target representation.",
            "- This does not replace the full v2.1/v3 training and inference comparison.",
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


def _tokenize_record_once(
    record: Any,
    *,
    vocab: MapperV3Vocab,
    timepoints_cache: OrderedDict[str, tuple[Any, ...]],
    max_cached_timepoint_maps: int,
):
    write_start_ms = int(record.target_start_ms)
    write_end_ms = write_start_ms + MAPPER_WRITE_MS
    timepoints = _load_timepoints(
        record.beatmap_path,
        timepoints_cache=timepoints_cache,
        max_cached_timepoint_maps=max_cached_timepoint_maps,
    )
    chart_end_ms = mapper_chart_end_ms(timepoints)
    if chart_end_ms < write_start_ms:
        raise UnsupportedMapperActionError(
            f"mapper v3 write window starts after chart_end_ms: {write_start_ms} > {chart_end_ms}",
        )
    return encode_mapper_window(
        timepoints,
        vocab=vocab,
        write_start_ms=write_start_ms,
        write_end_ms=write_end_ms,
        chart_start_ms=0,
        chart_end_ms=chart_end_ms,
    )


def _load_timepoints(
    beatmap_path: Path,
    *,
    timepoints_cache: OrderedDict[str, tuple[Any, ...]],
    max_cached_timepoint_maps: int,
) -> tuple[Any, ...]:
    key = beatmap_path.as_posix()
    cached = timepoints_cache.get(key)
    if cached is not None:
        timepoints_cache.move_to_end(key)
        return cached
    timepoints = tuple(hitobjects_to_mapper_timepoints(parse_mania_hit_objects(beatmap_path, expected_key_count=4)))
    if int(max_cached_timepoint_maps) > 0:
        timepoints_cache[key] = timepoints
        while len(timepoints_cache) > int(max_cached_timepoint_maps):
            timepoints_cache.popitem(last=False)
    return timepoints


def _empty_aggregate() -> dict[str, Any]:
    return {
        "audited_window_count": 0,
        "v3_token_count": 0,
        "time_shift_token_count": 0,
        "event_token_count": 0,
        "eos_count": 0,
        "event_label_count": 0,
        "signature_label_count": 0,
        "end_gap_label_count": 0,
        "out_of_range_delta_count": 0,
        "out_of_range_end_gap_count": 0,
        "negative_delta_count": 0,
        "negative_end_gap_count": 0,
        "event_delta_counts": Counter(),
        "end_gap_counts": Counter(),
        "event_signature_counts": Counter(),
    }


def _accumulate(aggregate: dict[str, Any], coverage: Mapping[str, Any]) -> None:
    aggregate["audited_window_count"] += 1
    for key in (
        "v3_token_count",
        "time_shift_token_count",
        "event_token_count",
        "eos_count",
        "event_label_count",
        "signature_label_count",
        "end_gap_label_count",
        "out_of_range_delta_count",
        "out_of_range_end_gap_count",
        "negative_delta_count",
        "negative_end_gap_count",
    ):
        aggregate[key] += int(coverage.get(key) or 0)
    aggregate["event_delta_counts"].update(int(value) for value in _sequence(coverage.get("event_deltas")))
    aggregate["end_gap_counts"].update([int(coverage.get("end_gap_ms") or 0)])
    aggregate["event_signature_counts"].update(str(value) for value in _sequence(coverage.get("event_signatures")))


def _finalize_metrics(
    aggregate: Mapping[str, Any],
    *,
    candidate_window_count: int,
    source_window_count: int,
    skipped_stride_window_count: int,
    last_source_index: int,
    errors: Sequence[Mapping[str, Any]],
    unsupported_errors: Sequence[Mapping[str, Any]],
    limit: int | None,
) -> dict[str, Any]:
    event_deltas = _counter(aggregate.get("event_delta_counts"))
    end_gaps = _counter(aggregate.get("end_gap_counts"))
    signatures = _counter(aggregate.get("event_signature_counts"))
    metrics = {
        "limit": None if limit is None else int(limit),
        "source_window_count": int(source_window_count),
        "last_source_index": int(last_source_index),
        "candidate_window_count": int(candidate_window_count),
        "audited_window_count": int(aggregate["audited_window_count"]),
        "skipped_stride_window_count": int(skipped_stride_window_count),
        "unsupported_window_count": len(unsupported_errors),
        "label_error_count": len(errors),
        "v3_token_count": int(aggregate["v3_token_count"]),
        "time_shift_token_count": int(aggregate["time_shift_token_count"]),
        "event_token_count": int(aggregate["event_token_count"]),
        "eos_count": int(aggregate["eos_count"]),
        "event_label_count": int(aggregate["event_label_count"]),
        "signature_label_count": int(aggregate["signature_label_count"]),
        "end_gap_label_count": int(aggregate["end_gap_label_count"]),
        "out_of_range_delta_count": int(aggregate["out_of_range_delta_count"]),
        "out_of_range_end_gap_count": int(aggregate["out_of_range_end_gap_count"]),
        "negative_delta_count": int(aggregate["negative_delta_count"]),
        "negative_end_gap_count": int(aggregate["negative_end_gap_count"]),
        "max_event_delta_ms": max(event_deltas) if event_deltas else 0,
        "max_end_gap_ms": max(end_gaps) if end_gaps else 0,
        "unique_event_delta_count": len(event_deltas),
        "unique_end_gap_count": len(end_gaps),
        "unique_event_signature_count": len(signatures),
        "zero_delta_event_count": int(event_deltas.get(0, 0)),
        "top_event_deltas": _top_counter(event_deltas),
        "top_end_gaps": _top_counter(end_gaps),
        "top_event_signatures": _top_counter(signatures),
    }
    return metrics


def _load_context(
    *,
    v3_full_dataset_summary_path: Path,
    fixed_slice_summary_path: Path,
    real_data_training_summary_path: Path,
) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if v3_full_dataset_summary_path.exists():
        full = _read_json(v3_full_dataset_summary_path)
        context["v3_full_dataset_audit"] = {
            "path": v3_full_dataset_summary_path.as_posix(),
            "decision": _mapping(full.get("decision")),
            "audit": _mapping(full.get("audit")),
            "comparison": _mapping(full.get("comparison")),
            "dataset": _mapping(full.get("dataset")),
        }
    if fixed_slice_summary_path.exists():
        fixed = _read_json(fixed_slice_summary_path)
        context["fixed_slice_label_coverage"] = {
            "path": fixed_slice_summary_path.as_posix(),
            "route": str(_mapping(fixed.get("decision")).get("route") or ""),
            "max_end_gap_ms": _mapping(fixed.get("metrics")).get("max_end_gap_ms"),
            "event_label_count": _mapping(fixed.get("metrics")).get("event_label_count"),
            "end_gap_label_count": _mapping(fixed.get("metrics")).get("end_gap_label_count"),
        }
    if real_data_training_summary_path.exists():
        training = _read_json(real_data_training_summary_path)
        context["tiny_real_data_training"] = {
            "path": real_data_training_summary_path.as_posix(),
            "route": str(_mapping(training.get("decision")).get("route") or ""),
            "auxiliary_loss_ratio": _mapping(training.get("training")).get("auxiliary_loss_ratio"),
            "token_loss_ratio": _mapping(training.get("training")).get("token_loss_ratio"),
        }
    return context


def _expected_full_v3_audit_window_count(context: Mapping[str, Any]) -> int | None:
    dataset = _mapping(_mapping(context.get("v3_full_dataset_audit")).get("dataset"))
    value = dataset.get("audit_window_count")
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _maybe_print_progress(
    *,
    progress_interval: int,
    aggregate: Mapping[str, Any],
    candidate_window_count: int,
    source_index: int,
    source_window_count: int,
    error_count: int,
    started: float,
) -> None:
    audited = int(aggregate.get("audited_window_count") or 0)
    if int(progress_interval) <= 0 or audited <= 0 or audited % int(progress_interval) != 0:
        return
    elapsed_s = time.monotonic() - started
    print(
        "mapper_v3_delta_event_auxiliary_full4k_label_coverage_progress "
        f"audited_windows={audited} candidate_windows={candidate_window_count} "
        f"source_index={source_index + 1}/{source_window_count} errors={error_count} "
        f"max_event_delta_ms={_max_counter_key(aggregate.get('event_delta_counts'))} "
        f"max_end_gap_ms={_max_counter_key(aggregate.get('end_gap_counts'))} "
        f"elapsed_s={elapsed_s:.1f}",
        flush=True,
    )


def _coverage_has_failure(coverage: Mapping[str, Any]) -> bool:
    return any(
        int(coverage.get(key) or 0) > 0
        for key in (
            "out_of_range_delta_count",
            "out_of_range_end_gap_count",
            "negative_delta_count",
            "negative_end_gap_count",
        )
    )


def _what_passed(route: str) -> str:
    if route == "TEST_DELTA_EVENT_FULL4K_BIT_PROXY_OR_FACTOR_TARGET_CARD":
        return (
            "- Full4k v3 fragments produced event-delta, event-signature, and end-gap labels under the current head ranges.\n"
            "- Event/signature label counts matched event-token count.\n"
            "- End-gap label count matched audited windows.\n"
            "- No training, rollout, tokenizer/default change, C3 backreference, or future lookup was used."
        )
    return "- The audit produced an explicit route; inspect failed checks and examples before scaling."


def _what_surfaced(route: str, *, metrics: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    max_end_gap = int(metrics.get("max_end_gap_ms") or 0)
    end_gap_max = int(config.get("end_gap_max_ms") or 0)
    if route == "TEST_DELTA_EVENT_FULL4K_BIT_PROXY_OR_FACTOR_TARGET_CARD":
        if end_gap_max > 0 and max_end_gap >= end_gap_max:
            return (
                f"Full4k coverage still reaches the current end-gap boundary at `{max_end_gap}ms` but does not exceed it. "
                "This supports a follow-up full4k bit-proxy/factor-target card, while keeping the boundary visible as a range-risk."
            )
        return (
            f"Full4k coverage did not reach the configured end-gap boundary: max end-gap `{max_end_gap}ms` "
            f"under max `{end_gap_max}ms`."
        )
    if route == "MUTATE_DELTA_EVENT_HEAD_RANGE_OR_BUCKETING":
        return "The label semantics are present, but the current delta/end-gap head ranges are not sufficient at full4k scope."
    if route == "MUTATE_FULL4K_LABEL_AUDIT_RUNTIME":
        return "The audit did not produce full-scope evidence; partial coverage should not be used for replacement decisions."
    return "The full4k label surface failed hard checks and should be repaired before training or factor-target work."


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
        "uv run python -m pulsefield_model.evals.mapper_v3_delta_event_auxiliary_full4k_label_coverage --progress-interval 5000",
        "uv run --group dev pytest tests/evals/test_mapper_v3_delta_event_auxiliary_full4k_label_coverage.py tests/evals/test_mapper_v3_delta_event_auxiliary_fixed_slice_label_coverage.py tests/evals/test_mapper_v3_delta_event_auxiliary_tiny_real_data_training.py tests/models/mapper/v3/test_model.py -q",
        "uv run python -m py_compile src/pulsefield_model/evals/mapper_v3_delta_event_auxiliary_full4k_label_coverage.py",
        "python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_auxiliary_full4k_label_coverage_summary.json >/dev/null",
        "git diff --check",
        "```",
    ]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_range(value: int, *, name: str) -> None:
    if int(value) <= 0 or int(value) % 10 != 0:
        raise ValueError(f"{name} must be a positive 10ms-grid value")


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else ()


def _counter(value: object) -> Counter[Any]:
    return value if isinstance(value, Counter) else Counter()


def _top_counter(counter: Counter[Any], *, limit: int = 12) -> list[dict[str, Any]]:
    return [{"value": value, "count": int(count)} for value, count in counter.most_common(limit)]


def _counter_rows(value: object) -> list[str]:
    rows: list[str] = []
    for item in _sequence(value):
        row = _mapping(item)
        rows.append(f"| `{row.get('value')}` | `{row.get('count')}` |")
    return rows


def _max_counter_key(value: object) -> int:
    counter = _counter(value)
    return max(counter) if counter else 0


def _row(label: str, value: object) -> str:
    return f"| {label} | `{value}` |"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit full4k v3 delta-event auxiliary label coverage.")
    parser.add_argument("--index-path", default=DEFAULT_INDEX_PATH)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--report-output", default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--delta-max-ms", type=int, default=8000)
    parser.add_argument("--end-gap-max-ms", type=int, default=8000)
    parser.add_argument("--limit", type=_parse_limit)
    parser.add_argument("--max-cached-timepoint-maps", type=int, default=64)
    parser.add_argument("--progress-interval", type=int, default=0)
    parser.add_argument("--dataset-progress", action="store_true")
    args = parser.parse_args(argv)
    summary = run_delta_event_auxiliary_full4k_label_coverage(
        index_path=Path(args.index_path),
        dataset_root=Path(args.dataset_root),
        summary_output_path=Path(args.summary_output),
        report_output_path=Path(args.report_output) if args.report_output else None,
        delta_max_ms=args.delta_max_ms,
        end_gap_max_ms=args.end_gap_max_ms,
        limit=args.limit,
        max_cached_timepoint_maps=args.max_cached_timepoint_maps,
        progress_interval=args.progress_interval,
        dataset_progress=args.dataset_progress,
    )
    print(
        "mapper_v3_delta_event_auxiliary_full4k_label_coverage_done "
        f"route={summary['decision']['route']} "
        f"audited_windows={summary['metrics']['audited_window_count']} "
        f"label_errors={summary['metrics']['label_error_count']} "
        f"max_end_gap_ms={summary['metrics']['max_end_gap_ms']} "
        f"summary={Path(args.summary_output).as_posix()}",
        flush=True,
    )


def _parse_limit(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text == "all":
        return None
    try:
        parsed = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("limit must be a positive integer or 'all'") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("limit must be a positive integer or 'all'")
    return parsed


if __name__ == "__main__":
    main()
