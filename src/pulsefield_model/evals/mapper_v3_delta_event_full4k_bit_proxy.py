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
from pulsefield_model.evals.target_grammar_v3_delta_event_proxy_audit import (
    FLAT_PAIR_TRACTABLE_LIMIT,
    delta_event_proxy_from_v3_tokens,
    unigram_total_bits,
)
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
DEFAULT_EXPERIMENT_CARD = REPORT_ROOT / "target_grammar_v3_delta_event_full4k_bit_proxy_experiment_card.md"
DEFAULT_SUMMARY_OUTPUT = REPORT_ROOT / "target_grammar_v3_delta_event_full4k_bit_proxy_summary.json"
DEFAULT_REPORT_OUTPUT = REPORT_ROOT / "target_grammar_v3_delta_event_full4k_bit_proxy_result_report.md"
DEFAULT_V3_FULL_DATASET_SUMMARY_PATH = REPORT_ROOT / "target_grammar_v3_full_dataset_audit_summary.json"
DEFAULT_FULL4K_LABEL_COVERAGE_SUMMARY_PATH = (
    REPORT_ROOT / "target_grammar_v3_delta_event_auxiliary_full4k_label_coverage_summary.json"
)
DEFAULT_BOUNDED_PROXY_SUMMARY_PATH = REPORT_ROOT / "target_grammar_v3_delta_event_proxy_audit_summary.json"


def run_delta_event_full4k_bit_proxy(
    *,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    summary_output_path: str | Path = DEFAULT_SUMMARY_OUTPUT,
    report_output_path: str | Path | None = DEFAULT_REPORT_OUTPUT,
    experiment_card_path: str | Path = DEFAULT_EXPERIMENT_CARD,
    v3_full_dataset_summary_path: str | Path = DEFAULT_V3_FULL_DATASET_SUMMARY_PATH,
    full4k_label_coverage_summary_path: str | Path = DEFAULT_FULL4K_LABEL_COVERAGE_SUMMARY_PATH,
    bounded_proxy_summary_path: str | Path = DEFAULT_BOUNDED_PROXY_SUMMARY_PATH,
    limit: int | None = None,
    max_cached_timepoint_maps: int = 64,
    progress_interval: int = 0,
    dataset_progress: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
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
    counters = _empty_counters()
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
            token_ids = tuple(int(token_id) for token_id in tokenized.target_fragment_ids)
            proxy = delta_event_proxy_from_v3_tokens(
                token_ids,
                vocab=vocab,
                write_start_ms=int(tokenized.write_start_ms),
                write_end_ms=int(tokenized.write_end_ms),
                chart_end_ms=int(tokenized.chart_end_ms),
                is_full_chart_end=bool(tokenized.is_full_chart_end),
            )
        except UnsupportedMapperActionError as exc:
            unsupported_errors.append(_error_record(source_index, candidate_window_count - 1, record, exc))
            _maybe_print_progress(
                progress_interval=progress_interval,
                counters=counters,
                candidate_window_count=candidate_window_count,
                source_index=source_index,
                source_window_count=len(dataset.records),
                error_count=len(errors),
                started=started,
            )
            continue
        except Exception as exc:  # noqa: BLE001 - full-scope report needs concrete per-record failures.
            error = _error_record(source_index, candidate_window_count - 1, record, exc)
            errors.append(error)
            if len(examples) < 16:
                examples.append(error)
            _maybe_print_progress(
                progress_interval=progress_interval,
                counters=counters,
                candidate_window_count=candidate_window_count,
                source_index=source_index,
                source_window_count=len(dataset.records),
                error_count=len(errors),
                started=started,
            )
            continue
        _accumulate_proxy(counters, token_ids=token_ids, proxy=proxy, vocab=vocab)
        _maybe_print_progress(
            progress_interval=progress_interval,
            counters=counters,
            candidate_window_count=candidate_window_count,
            source_index=source_index,
            source_window_count=len(dataset.records),
            error_count=len(errors),
            started=started,
        )

    metrics = finalize_proxy_metrics(
        counters,
        candidate_window_count=candidate_window_count,
        source_window_count=len(dataset.records),
        skipped_stride_window_count=skipped_stride_window_count,
        unsupported_window_count=len(unsupported_errors),
        label_error_count=len(errors),
        last_source_index=last_source_index,
        limit=limit,
        vocab=vocab,
    )
    context = _load_context(
        v3_full_dataset_summary_path=Path(v3_full_dataset_summary_path),
        full4k_label_coverage_summary_path=Path(full4k_label_coverage_summary_path),
        bounded_proxy_summary_path=Path(bounded_proxy_summary_path),
    )
    checks = guard_checks(metrics=metrics, context=context)
    decision = decision_from_checks(checks, metrics)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar v3 delta-event full4k bit proxy",
        "experiment_card": Path(experiment_card_path).as_posix(),
        "elapsed_s": time.monotonic() - started,
        "index_path": Path(index_path).as_posix(),
        "dataset_root": Path(dataset_root).as_posix(),
        "config": {
            "limit": None if limit is None else int(limit),
            "max_cached_timepoint_maps": int(max_cached_timepoint_maps),
            "progress_interval": int(progress_interval),
            "dataset_progress": bool(dataset_progress),
            "flat_pair_tractable_limit": int(FLAT_PAIR_TRACTABLE_LIMIT),
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
        "errors": errors[:32],
        "unsupported_examples": unsupported_errors[:32],
        "examples": examples,
    }
    write_summary_json(summary, Path(summary_output_path))
    if report_output_path is not None:
        write_report(summary, Path(report_output_path))
    return summary


def guard_checks(*, metrics: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, bool]:
    expected_windows = _expected_full_v3_audit_window_count(context)
    expected_label_windows = _expected_label_coverage_window_count(context)
    expected_label_events = _expected_label_coverage_event_count(context)
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
        "full4k_label_coverage_positive": _mapping(context.get("full4k_label_coverage")).get("route")
        == "TEST_DELTA_EVENT_FULL4K_BIT_PROXY_OR_FACTOR_TARGET_CARD",
        "audited_windows_match_full_v3_audit": expected_windows is not None
        and int(metrics.get("audited_window_count") or -1) == int(expected_windows),
        "audited_windows_match_label_coverage": expected_label_windows is not None
        and int(metrics.get("audited_window_count") or -1) == int(expected_label_windows),
        "event_rows_match_label_coverage": expected_label_events is not None
        and int(metrics.get("event_row_count") or -1) == int(expected_label_events),
        "label_errors_zero": int(metrics.get("label_error_count") or 0) == 0,
        "event_reconstruction_zero": int(metrics.get("event_reconstruction_mismatches") or 0) == 0,
        "end_reconstruction_zero": int(metrics.get("end_reconstruction_mismatches") or 0) == 0,
        "sequence_shorter_than_v3": (_float(metrics.get("sequence_length_ratio_vs_v3")) or math.inf) < 1.0,
        "factorized_bits_not_worse_than_v3": (_float(metrics.get("proxy_factorized_total_bit_ratio_vs_v3")) or math.inf)
        <= 1.0,
        "proxy_rows_positive": int(metrics.get("proxy_row_count") or 0) > 0,
        "v3_tokens_positive": int(metrics.get("v3_token_count") or 0) > 0,
    }


def decision_from_checks(checks: Mapping[str, bool], metrics: Mapping[str, Any]) -> dict[str, str]:
    failed = [key for key, passed in checks.items() if not bool(passed)]
    if not failed:
        flat_pair_tractable = bool(metrics.get("flat_pair_tractable"))
        if flat_pair_tractable:
            return {
                "route": "TEST_DELTA_EVENT_FLAT_OR_FACTORIZED_TARGET_CARD",
                "reason": (
                    "full4k delta-event proxy passed reconstruction, sequence, factorized-bit, and flat-pair tractability gates "
                    f"(row ratio={_fmt(metrics.get('sequence_length_ratio_vs_v3'))}, "
                    f"bit ratio={_fmt(metrics.get('proxy_factorized_total_bit_ratio_vs_v3'))})"
                ),
                "next_step": "Create a bounded target/model card comparing flat delta-event rows versus factorized heads.",
            }
        return {
            "route": "TEST_DELTA_EVENT_FACTOR_TARGET_MODEL_CARD",
            "reason": (
                "full4k delta-event factorized proxy passed reconstruction, sequence, and bit gates; "
                "flat delta-event pairs are too numerous for the flat-token variant"
            ),
            "next_step": "Create a bounded factorized target/model card; keep flat delta-event rows as rejected baseline.",
        }
    runtime_failures = {
        "full_scope",
        "audited_windows_match_full_v3_audit",
        "audited_windows_match_label_coverage",
        "event_rows_match_label_coverage",
        "proxy_rows_positive",
        "v3_tokens_positive",
    }
    reconstruction_failures = {"event_reconstruction_zero", "end_reconstruction_zero", "label_errors_zero"}
    if any(key in runtime_failures for key in failed):
        return {
            "route": "MUTATE_FULL4K_PROXY_AUDIT_RUNTIME",
            "reason": "full4k proxy audit did not complete comparable full-scope evidence: " + ", ".join(failed),
            "next_step": "Fix audit completeness before interpreting delta-event proxy compression.",
        }
    if any(key in reconstruction_failures for key in failed):
        return {
            "route": "KILL_DELTA_EVENT_FACTOR_TARGET",
            "reason": "delta-event proxy failed reconstruction or hard label checks: " + ", ".join(failed),
            "next_step": "Do not implement factorized target; repair conversion semantics first.",
        }
    if "sequence_shorter_than_v3" in failed:
        return {
            "route": "KILL_DELTA_EVENT_PROXY_SEQUENCE",
            "reason": "delta-event proxy did not reduce sequence rows versus current v3",
            "next_step": "Stop this factorization path; return to simpler v3/v2.1 grammar hardening.",
        }
    if "factorized_bits_not_worse_than_v3" in failed:
        return {
            "route": "MUTATE_DELTA_EVENT_BUCKETING",
            "reason": "full4k delta-event proxy reduced rows but regressed the factorized unigram bit proxy",
            "next_step": "Try delta/end-gap bucketing or stop before target/model implementation.",
        }
    return {
        "route": "MUTATE_DELTA_EVENT_FULL4K_PROXY_CONTEXT",
        "reason": "full4k proxy audit failed context guards: " + ", ".join(failed),
        "next_step": "Repair the prior-evidence chain before target/model work.",
    }


def finalize_proxy_metrics(
    counters: Mapping[str, Any],
    *,
    candidate_window_count: int,
    source_window_count: int,
    skipped_stride_window_count: int,
    unsupported_window_count: int,
    label_error_count: int,
    last_source_index: int,
    limit: int | None,
    vocab: MapperV3Vocab,
) -> dict[str, Any]:
    v3_token_counts = _counter(counters.get("v3_token_counts"))
    kind_counts = _counter(counters.get("kind_counts"))
    event_delta_counts = _counter(counters.get("event_delta_counts"))
    end_gap_counts = _counter(counters.get("end_gap_counts"))
    signature_counts = _counter(counters.get("signature_counts"))
    flat_pair_counts = _counter(counters.get("flat_pair_counts"))
    v3_total_bits = unigram_total_bits(v3_token_counts)
    proxy_total_bits = (
        unigram_total_bits(kind_counts)
        + unigram_total_bits(event_delta_counts)
        + unigram_total_bits(signature_counts)
        + unigram_total_bits(end_gap_counts)
    )
    flat_total_bits = unigram_total_bits(kind_counts) + unigram_total_bits(flat_pair_counts) + unigram_total_bits(end_gap_counts)
    v3_token_count = int(sum(v3_token_counts.values()))
    event_rows = int(counters.get("event_row_count") or 0)
    end_rows = int(counters.get("end_row_count") or 0)
    proxy_rows = event_rows + end_rows
    return {
        "limit": None if limit is None else int(limit),
        "source_window_count": int(source_window_count),
        "last_source_index": int(last_source_index),
        "candidate_window_count": int(candidate_window_count),
        "audited_window_count": int(counters.get("audited_window_count") or 0),
        "skipped_stride_window_count": int(skipped_stride_window_count),
        "unsupported_window_count": int(unsupported_window_count),
        "label_error_count": int(label_error_count),
        "v3_token_count": int(v3_token_count),
        "proxy_row_count": int(proxy_rows),
        "event_row_count": int(event_rows),
        "end_row_count": int(end_rows),
        "sequence_length_ratio_vs_v3": _safe_ratio(proxy_rows, v3_token_count),
        "sequence_length_delta_vs_v3": int(proxy_rows - v3_token_count),
        "event_reconstruction_mismatches": int(counters.get("event_reconstruction_mismatches") or 0),
        "end_reconstruction_mismatches": int(counters.get("end_reconstruction_mismatches") or 0),
        "v3_unigram_total_bits": float(v3_total_bits),
        "v3_unigram_bits_per_token": _safe_ratio(v3_total_bits, v3_token_count),
        "proxy_factorized_total_bits": float(proxy_total_bits),
        "proxy_factorized_bits_per_row": _safe_ratio(proxy_total_bits, proxy_rows),
        "proxy_factorized_total_bit_ratio_vs_v3": _safe_ratio(proxy_total_bits, v3_total_bits),
        "flat_pair_total_bits": float(flat_total_bits),
        "flat_pair_bits_per_row": _safe_ratio(flat_total_bits, proxy_rows),
        "flat_pair_total_bit_ratio_vs_v3": _safe_ratio(flat_total_bits, v3_total_bits),
        "unique": {
            "v3_tokens": len(v3_token_counts),
            "kinds": len(kind_counts),
            "event_deltas": len(event_delta_counts),
            "end_gaps": len(end_gap_counts),
            "event_signatures": len(signature_counts),
            "flat_delta_event_pairs": len(flat_pair_counts),
        },
        "flat_pair_tractable": len(flat_pair_counts) <= FLAT_PAIR_TRACTABLE_LIMIT,
        "zero_delta_event_count": int(event_delta_counts.get(0, 0)),
        "max_event_delta_ms": max(event_delta_counts) if event_delta_counts else 0,
        "max_end_gap_ms": max(end_gap_counts) if end_gap_counts else 0,
        "top_event_deltas": _top_counter(event_delta_counts),
        "top_end_gaps": _top_counter(end_gap_counts),
        "top_event_signatures": _top_counter(signature_counts),
        "top_flat_pairs": _top_counter(flat_pair_counts),
        "special_token_counts": {
            "bos": int(v3_token_counts.get(int(vocab.bos_id), 0)),
            "eos": int(v3_token_counts.get(int(vocab.eos_id), 0)),
            "time_shift": sum(count for token_id, count in v3_token_counts.items() if vocab.is_time_shift_token(int(token_id))),
            "event": sum(count for token_id, count in v3_token_counts.items() if vocab.is_event_token(int(token_id))),
        },
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
    unique = _mapping(metrics.get("unique"))
    special = _mapping(metrics.get("special_token_counts"))
    lines = [
        "# Target Grammar v3 Delta-Event Full4K Bit Proxy Result Report",
        "",
        "## Scope",
        "",
        "This full4k representation audit converts current v3 target fragments into a factorized delta-event proxy and computes sequence/bit proxies. It does not train, roll out, change tokenizer behavior, change dataset schema, change mapper defaults, or use C3/future context.",
        "",
        "## Decision",
        "",
        f"Route: `{decision.get('route')}`.",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Recommended next step: {decision.get('next_step')}",
        "",
        "## Guard Results",
        "",
        "| Guard | Value |",
        "| --- | ---: |",
    ]
    lines.extend(_row(key, value) for key, value in checks.items())
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            _row("audited windows", metrics.get("audited_window_count")),
            _row("candidate windows", metrics.get("candidate_window_count")),
            _row("unsupported skipped windows", metrics.get("unsupported_window_count")),
            _row("v3 token count", metrics.get("v3_token_count")),
            _row("proxy row count", metrics.get("proxy_row_count")),
            _row("sequence ratio vs v3", _fmt(metrics.get("sequence_length_ratio_vs_v3"))),
            _row("v3 unigram total bits", _fmt(metrics.get("v3_unigram_total_bits"))),
            _row("proxy factorized total bits", _fmt(metrics.get("proxy_factorized_total_bits"))),
            _row("proxy bit ratio vs v3", _fmt(metrics.get("proxy_factorized_total_bit_ratio_vs_v3"))),
            _row("flat pair bit ratio vs v3", _fmt(metrics.get("flat_pair_total_bit_ratio_vs_v3"))),
            _row("event reconstruction mismatches", metrics.get("event_reconstruction_mismatches")),
            _row("end reconstruction mismatches", metrics.get("end_reconstruction_mismatches")),
            _row("unique event deltas", unique.get("event_deltas")),
            _row("unique end gaps", unique.get("end_gaps")),
            _row("unique event signatures", unique.get("event_signatures")),
            _row("unique flat delta-event pairs", unique.get("flat_delta_event_pairs")),
            _row("flat pair tractable", metrics.get("flat_pair_tractable")),
            _row("v3 time-shift token count", special.get("time_shift")),
            _row("v3 event token count", special.get("event")),
            _row("max event delta ms", metrics.get("max_event_delta_ms")),
            _row("max end-gap ms", metrics.get("max_end_gap_ms")),
            "",
            "## Top Event Deltas",
            "",
            "| delta ms | count |",
            "| ---: | ---: |",
        ]
    )
    lines.extend(_counter_rows(metrics.get("top_event_deltas")))
    lines.extend(["", "## Top End Gaps", "", "| end gap ms | count |", "| ---: | ---: |"])
    lines.extend(_counter_rows(metrics.get("top_end_gaps")))
    lines.extend(
        [
            "",
            "## What Passed",
            "",
            _what_passed(str(decision.get("route")), metrics=metrics, unique=unique),
            "",
            "## What Surfaced",
            "",
            _what_surfaced(str(decision.get("route")), metrics=metrics, unique=unique),
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


def _accumulate_proxy(
    counters: dict[str, Any],
    *,
    token_ids: Sequence[int],
    proxy: Any,
    vocab: MapperV3Vocab,
) -> None:
    counters["audited_window_count"] += 1
    counters["event_reconstruction_mismatches"] += int(proxy.event_mismatch_count)
    counters["end_reconstruction_mismatches"] += int(proxy.end_mismatch_count)
    counters["v3_token_counts"].update(int(token_id) for token_id in token_ids)
    for row in proxy.rows:
        delta_ms = int(row.delta_ms)
        signature = str(row.signature)
        counters["kind_counts"]["EVENT"] += 1
        counters["event_delta_counts"][delta_ms] += 1
        counters["signature_counts"][signature] += 1
        counters["flat_pair_counts"][f"{delta_ms}:{signature}"] += 1
        counters["event_row_count"] += 1
    counters["kind_counts"]["END"] += 1
    counters["end_gap_counts"][int(proxy.end_gap_ms)] += 1
    counters["end_row_count"] += 1
    # Keep vocab referenced in the call site contract; the streaming counts above are token-id based.
    _ = vocab


def _empty_counters() -> dict[str, Any]:
    return {
        "audited_window_count": 0,
        "event_row_count": 0,
        "end_row_count": 0,
        "event_reconstruction_mismatches": 0,
        "end_reconstruction_mismatches": 0,
        "v3_token_counts": Counter(),
        "kind_counts": Counter(),
        "event_delta_counts": Counter(),
        "end_gap_counts": Counter(),
        "signature_counts": Counter(),
        "flat_pair_counts": Counter(),
    }


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


def _load_context(
    *,
    v3_full_dataset_summary_path: Path,
    full4k_label_coverage_summary_path: Path,
    bounded_proxy_summary_path: Path,
) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if v3_full_dataset_summary_path.exists():
        full = _read_json(v3_full_dataset_summary_path)
        context["v3_full_dataset_audit"] = {
            "path": v3_full_dataset_summary_path.as_posix(),
            "decision": _mapping(full.get("decision")),
            "dataset": _mapping(full.get("dataset")),
            "comparison": _mapping(full.get("comparison")),
        }
    if full4k_label_coverage_summary_path.exists():
        coverage = _read_json(full4k_label_coverage_summary_path)
        context["full4k_label_coverage"] = {
            "path": full4k_label_coverage_summary_path.as_posix(),
            "route": str(_mapping(coverage.get("decision")).get("route") or ""),
            "metrics": _mapping(coverage.get("metrics")),
        }
    if bounded_proxy_summary_path.exists():
        proxy = _read_json(bounded_proxy_summary_path)
        context["bounded_proxy"] = {
            "path": bounded_proxy_summary_path.as_posix(),
            "route": str(_mapping(proxy.get("decision")).get("route") or ""),
            "row_ratio": _mapping(proxy.get("metrics")).get("sequence_length_ratio_vs_v3"),
            "bit_ratio": _mapping(proxy.get("metrics")).get("proxy_factorized_total_bit_ratio_vs_v3"),
        }
    return context


def _expected_full_v3_audit_window_count(context: Mapping[str, Any]) -> int | None:
    dataset = _mapping(_mapping(context.get("v3_full_dataset_audit")).get("dataset"))
    return _positive_int_or_none(dataset.get("audit_window_count"))


def _expected_label_coverage_window_count(context: Mapping[str, Any]) -> int | None:
    metrics = _mapping(_mapping(context.get("full4k_label_coverage")).get("metrics"))
    return _positive_int_or_none(metrics.get("audited_window_count"))


def _expected_label_coverage_event_count(context: Mapping[str, Any]) -> int | None:
    metrics = _mapping(_mapping(context.get("full4k_label_coverage")).get("metrics"))
    return _positive_int_or_none(metrics.get("event_label_count"))


def _positive_int_or_none(value: object) -> int | None:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _maybe_print_progress(
    *,
    progress_interval: int,
    counters: Mapping[str, Any],
    candidate_window_count: int,
    source_index: int,
    source_window_count: int,
    error_count: int,
    started: float,
) -> None:
    audited = int(counters.get("audited_window_count") or 0)
    if int(progress_interval) <= 0 or audited <= 0 or audited % int(progress_interval) != 0:
        return
    elapsed_s = time.monotonic() - started
    print(
        "mapper_v3_delta_event_full4k_bit_proxy_progress "
        f"audited_windows={audited} candidate_windows={candidate_window_count} "
        f"source_index={source_index + 1}/{source_window_count} errors={error_count} "
        f"event_rows={counters.get('event_row_count')} "
        f"unique_flat_pairs={len(_counter(counters.get('flat_pair_counts')))} "
        f"elapsed_s={elapsed_s:.1f}",
        flush=True,
    )


def _error_record(source_index: int, candidate_window_index: int, record: Any, exc: Exception) -> dict[str, Any]:
    return {
        "source_index": int(source_index),
        "candidate_window_index": int(candidate_window_index),
        "beatmap_path": record.beatmap_path.as_posix(),
        "target_start_ms": int(record.target_start_ms),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _what_passed(route: str, *, metrics: Mapping[str, Any], unique: Mapping[str, Any]) -> str:
    if route in {"TEST_DELTA_EVENT_FLAT_OR_FACTORIZED_TARGET_CARD", "TEST_DELTA_EVENT_FACTOR_TARGET_MODEL_CARD"}:
        return (
            "- Full4k delta-event proxy reconstruction was exact.\n"
            "- Proxy rows were fewer than current v3 tokens.\n"
            "- Factorized proxy bits were not worse than current v3 under the full-scope self-unigram proxy.\n"
            "- No training, rollout, tokenizer/default change, C3 backreference, or future lookup was used."
        )
    return "- The audit produced an explicit route; inspect failed checks and examples before target/model work."


def _what_surfaced(route: str, *, metrics: Mapping[str, Any], unique: Mapping[str, Any]) -> str:
    flat_pairs = int(unique.get("flat_delta_event_pairs") or 0)
    if route == "TEST_DELTA_EVENT_FLAT_OR_FACTORIZED_TARGET_CARD":
        return (
            f"Full4k supports a delta-event target follow-up: row ratio `{_fmt(metrics.get('sequence_length_ratio_vs_v3'))}`, "
            f"factorized bit ratio `{_fmt(metrics.get('proxy_factorized_total_bit_ratio_vs_v3'))}`, and "
            f"`{flat_pairs}` flat delta/event pairs under the tractability guard."
        )
    if route == "TEST_DELTA_EVENT_FACTOR_TARGET_MODEL_CARD":
        return (
            f"Full4k supports the factorized delta-event route, but `{flat_pairs}` flat delta/event pairs exceed the "
            f"`{FLAT_PAIR_TRACTABLE_LIMIT}` guard, so the flat-token variant should remain a rejected baseline."
        )
    if route == "MUTATE_DELTA_EVENT_BUCKETING":
        return "The row reduction survived, but full4k factorized bits regressed; bucketing should be tested before model work."
    if route.startswith("KILL"):
        return "The proxy failed a hard representation gate and should not be implemented as a target grammar."
    return "The audit did not produce comparable full-scope evidence."


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
        "uv run python -m pulsefield_model.evals.mapper_v3_delta_event_full4k_bit_proxy --progress-interval 5000",
        "uv run --group dev pytest tests/evals/test_mapper_v3_delta_event_full4k_bit_proxy.py tests/evals/test_mapper_v3_delta_event_auxiliary_full4k_label_coverage.py tests/evals/test_target_grammar_v3_delta_event_proxy_audit.py tests/models/mapper/v3/test_model.py -q",
        "uv run python -m py_compile src/pulsefield_model/evals/mapper_v3_delta_event_full4k_bit_proxy.py",
        "python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_full4k_bit_proxy_summary.json >/dev/null",
        "git diff --check",
        "```",
    ]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _counter(value: object) -> Counter[Any]:
    return value if isinstance(value, Counter) else Counter()


def _float(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _safe_ratio(numerator: float | int, denominator: float | int) -> float:
    return float(numerator) / float(denominator) if float(denominator) else 0.0


def _fmt(value: object) -> str:
    number = _float(value)
    return "n/a" if number is None else f"{number:.6f}"


def _top_counter(counter: Counter[Any], *, limit: int = 12) -> list[dict[str, Any]]:
    return [{"value": value, "count": int(count)} for value, count in counter.most_common(limit)]


def _counter_rows(value: object) -> list[str]:
    rows: list[str] = []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            row = _mapping(item)
            rows.append(f"| `{row.get('value')}` | `{row.get('count')}` |")
    return rows


def _row(label: str, value: object) -> str:
    return f"| {label} | `{value}` |"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit full4k v3 delta-event bit proxy.")
    parser.add_argument("--index-path", default=DEFAULT_INDEX_PATH)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--report-output", default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--limit", type=_parse_limit)
    parser.add_argument("--max-cached-timepoint-maps", type=int, default=64)
    parser.add_argument("--progress-interval", type=int, default=0)
    parser.add_argument("--dataset-progress", action="store_true")
    args = parser.parse_args(argv)
    summary = run_delta_event_full4k_bit_proxy(
        index_path=Path(args.index_path),
        dataset_root=Path(args.dataset_root),
        summary_output_path=Path(args.summary_output),
        report_output_path=Path(args.report_output) if args.report_output else None,
        limit=args.limit,
        max_cached_timepoint_maps=args.max_cached_timepoint_maps,
        progress_interval=args.progress_interval,
        dataset_progress=args.dataset_progress,
    )
    print(
        "mapper_v3_delta_event_full4k_bit_proxy_done "
        f"route={summary['decision']['route']} "
        f"row_ratio={summary['metrics']['sequence_length_ratio_vs_v3']:.6f} "
        f"bit_ratio={summary['metrics']['proxy_factorized_total_bit_ratio_vs_v3']:.6f} "
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
