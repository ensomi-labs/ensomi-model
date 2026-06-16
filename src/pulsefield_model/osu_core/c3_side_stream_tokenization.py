from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

import pandas as pd

from pulsefield_model.osu_core.beat_chunk_pattern_audit import (
    DEFAULT_BEAT_CHUNK_CACHE_PATH,
    DEFAULT_MOTIF_MAX_N,
    DEFAULT_MOTIF_MIN_N,
    DEFAULT_SMOOTHING_ALPHA,
)
from pulsefield_model.osu_core.beat_chunk_tokenizer_refinement_audit import DEFAULT_RARE_MAX_COUNT
from pulsefield_model.osu_core.beat_representation_audit import DEFAULT_DATASET_ROOT
from pulsefield_model.osu_core.context_adaptive_fallback_codec_audit import (
    DEFAULT_LZ_MAX_SPAN,
    DEFAULT_LZ_WINDOW_FALLBACKS,
    _CostPlan,
    _FallbackRecord,
    _LzPolicy,
    _baseline_tokens,
    _build_lz_hardening_plan,
    _collect_fallback_records,
    _dataset_summary,
    _decode_motif_tokens,
    _encode_sequence,
    _fit_motif_model,
    _iter_rows,
    _lz_candidate_options_by_record,
    _mirror_mask_4,
    _mirror_order_signature_4,
    _read_chunk_cache,
)
from pulsefield_model.osu_core.duration_ln_tokenization_audit import (
    DEFAULT_MOTIF_VOCAB_SIZE,
    _normalize_json,
)
from pulsefield_model.models.mapper.v2_1.tokenizer import MAPPER_WRITE_MS


SCHEMA_VERSION: Final[int] = 1
DEFAULT_REPORT_PATH: Final[Path] = Path(
    "artifacts/reports/audits/context_adaptive_fallback_codec/c3_side_stream_pipeline_report.json",
)
DEFAULT_RESULT_LOG_PATH: Final[Path] = Path(
    "artifacts/reports/audits/context_adaptive_fallback_codec/c3_side_stream_pipeline_result_log.md",
)
DEFAULT_MAPPER_SIDECAR_PATH: Final[Path] = Path(
    "artifacts/cache/c3_mapper_window_sidecar/c3_mapper_window_sidecar_le3.json",
)
DEFAULT_MAPPER_SIDECAR_REPORT_PATH: Final[Path] = Path(
    "artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_window_sidecar_report.json",
)
DEFAULT_MAPPER_SIDECAR_RESULT_LOG_PATH: Final[Path] = Path(
    "artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_window_sidecar_result_log.md",
)
SELECTED_C3_VARIANT: Final[str] = "a2_skeleton_residual_all_fallback_w256"
MAPPER_SIDECAR_CONTRACT: Final[str] = "r0_delta_main_plus_c3_fallback_side_stream_v1"
MAPPER_SIDECAR_PAD_ID: Final[int] = 0
DEFAULT_MAPPER_SIDECAR_MAX_TOKENS: Final[int] = 256
DEFAULT_BOUNDARY_RISK_MARGIN_MS: Final[int] = 2_000
FALLBACK_PLACEHOLDER_TOKEN: Final[str] = "F"
RAW_PREFIX: Final[str] = "RAW|"
REF_PREFIX: Final[str] = "REF|"
RES_PREFIX: Final[str] = "RES|"


@dataclass(frozen=True, slots=True)
class _AtomPayload:
    delta: int
    tap: int
    start: int
    end: int
    order_signature: str


@dataclass(frozen=True, slots=True)
class _SideDecodeResult:
    decoded_by_record_id: dict[int, str]
    side_token_count: int
    raw_token_count: int
    ref_token_count: int
    residual_token_count: int
    reference_error_count: int
    mismatch_count: int
    examples: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _TracedSideToken:
    token: str
    record: _FallbackRecord
    kind: str


@dataclass(frozen=True, slots=True)
class _WindowAnchor:
    beatmap_path: str
    raw_beatmap_path: str
    shard: str
    split: str
    source_row_index: int
    beatmap_set_id: int
    beatmap_id: int
    window_start_ms: int
    chunk_sort_ms: float
    near_boundary: bool


def audit_c3_side_stream_tokenization(
    *,
    chunk_cache_path: str | Path = DEFAULT_BEAT_CHUNK_CACHE_PATH,
    report_path: str | Path | None = DEFAULT_REPORT_PATH,
    result_log_path: str | Path | None = DEFAULT_RESULT_LOG_PATH,
    motif_vocab_size: int = DEFAULT_MOTIF_VOCAB_SIZE,
    motif_min_n: int = DEFAULT_MOTIF_MIN_N,
    motif_max_n: int = DEFAULT_MOTIF_MAX_N,
    smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
    rare_max_count: int = DEFAULT_RARE_MAX_COUNT,
    lz_window_fallbacks: int = DEFAULT_LZ_WINDOW_FALLBACKS,
    lz_max_span: int = DEFAULT_LZ_MAX_SPAN,
    limit_chunks: int | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    """Validate P0 artifact-only C3 fallback side-stream tokenization."""

    del rare_max_count
    started_at = time.perf_counter()
    chunk_cache_path = Path(chunk_cache_path)
    report_path = None if report_path is None else Path(report_path)
    result_log_path = None if result_log_path is None else Path(result_log_path)
    _validate_positive(motif_vocab_size, "motif_vocab_size")
    _validate_positive(motif_min_n, "motif_min_n")
    _validate_positive(motif_max_n, "motif_max_n")
    _validate_positive(lz_window_fallbacks, "lz_window_fallbacks")
    _validate_positive(lz_max_span, "lz_max_span")
    if motif_max_n < motif_min_n:
        raise ValueError("motif_max_n must be >= motif_min_n")

    chunk_df = _read_chunk_cache(chunk_cache_path, limit_chunks=limit_chunks)
    baseline_model = _fit_motif_model(
        chunk_df,
        name="r0_delta",
        token_mapper=_baseline_tokens,
        motif_vocab_size=motif_vocab_size,
        motif_min_n=motif_min_n,
        motif_max_n=motif_max_n,
        smoothing_alpha=smoothing_alpha,
        top_motif_limit=20,
    )
    records, record_summary = _collect_fallback_records(chunk_df, baseline_model=baseline_model)
    plan = _build_selected_skeleton_residual_plan(
        records,
        window_fallbacks=lz_window_fallbacks,
        max_span=lz_max_span,
    )
    side_decode = _encode_decode_side_stream(records, plan)
    main_guard = _main_stream_reconstruction_guard(
        chunk_df,
        baseline_model=baseline_model,
        records=records,
        side_decoded_by_record_id=side_decode.decoded_by_record_id,
    )
    span_stats = _span_stats(plan)
    token_stats = {
        "baseline_encoded_token_count": main_guard["baseline_encoded_token_count"],
        "main_stream_token_count": main_guard["main_stream_token_count"],
        "fallback_placeholder_count": main_guard["fallback_placeholder_count"],
        "side_stream_token_count": side_decode.side_token_count,
        "side_stream_raw_token_count": side_decode.raw_token_count,
        "side_stream_ref_token_count": side_decode.ref_token_count,
        "side_stream_residual_token_count": side_decode.residual_token_count,
        "combined_main_side_token_count": main_guard["main_stream_token_count"] + side_decode.side_token_count,
        "combined_to_baseline_token_ratio": (
            float(main_guard["main_stream_token_count"] + side_decode.side_token_count)
            / float(main_guard["baseline_encoded_token_count"])
            if main_guard["baseline_encoded_token_count"]
            else 0.0
        ),
    }
    reconstruction_guard = {
        "pass": (
            side_decode.reference_error_count == 0
            and side_decode.mismatch_count == 0
            and int(main_guard["full_token_stream_mismatch_count"]) == 0
            and int(main_guard["group_signature_mismatch_count"]) == 0
            and int(main_guard["missing_side_payload_count"]) == 0
        ),
        "side_stream_reference_error_count": side_decode.reference_error_count,
        "side_stream_payload_mismatch_count": side_decode.mismatch_count,
        "full_token_stream_mismatch_count": main_guard["full_token_stream_mismatch_count"],
        "group_signature_mismatch_count": main_guard["group_signature_mismatch_count"],
        "mapper_timepoint_compatible_mismatch_count": main_guard["group_signature_mismatch_count"],
        "missing_side_payload_count": main_guard["missing_side_payload_count"],
        "examples": [*side_decode.examples, *main_guard["examples"]][:10],
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "C3 side-stream pipeline tokenization P0",
        "command": command,
        "code_commit": _git_stdout("rev-parse", "HEAD"),
        "code_dirty": bool(_git_stdout("status", "--short")),
        "chunk_cache_path": chunk_cache_path.as_posix(),
        "report_path": None if report_path is None else report_path.as_posix(),
        "result_log_path": None if result_log_path is None else result_log_path.as_posix(),
        "limited": limit_chunks is not None,
        "limit_chunks": limit_chunks,
        "elapsed_s": time.perf_counter() - started_at,
        "config": {
            "motif_vocab_size": motif_vocab_size,
            "motif_min_n": motif_min_n,
            "motif_max_n": motif_max_n,
            "smoothing_alpha": smoothing_alpha,
            "lz_window_fallbacks": lz_window_fallbacks,
            "lz_max_span": lz_max_span,
            "selected_c3_variant": SELECTED_C3_VARIANT,
            "token_contract": (
                "main stream keeps motif/control tokens and replaces fallback atoms with F placeholders; "
                "C3 side stream writes RAW, REF, and skeleton residual tokens in fallback-substream order."
            ),
        },
        "dataset": _dataset_summary(chunk_df),
        "fallback_record_summary": record_summary,
        "baseline_model": {
            "learned_motif_count": baseline_model.learned_motif_count,
            "dictionary_cost_bits": baseline_model.dictionary_bits,
            "atom_vocab_size": len(baseline_model.atom_counter),
            "train_encoded_vocab_size": len(baseline_model.train_counter),
        },
        "selected_plan": {
            "variant": plan.name,
            "span_count": len(plan.lz_spans),
            "selected_fallback_literal_count": sum(span.length for span in plan.lz_spans),
            "metadata": plan.metadata,
        },
        "span_stats": span_stats,
        "token_stats": token_stats,
        "reconstruction_guard": reconstruction_guard,
        "pass_criteria": {
            "p0_roundtrip_pass": reconstruction_guard["pass"],
            "full_cache_pass": limit_chunks is None,
            "selected_variant": plan.name,
            "mapper_defaults_unchanged": True,
            "research_pass": bool(reconstruction_guard["pass"]),
        },
        "recommendation": _recommendation(reconstruction_guard, limit_chunks=limit_chunks),
    }
    report = _normalize_json(report)
    if report_path is not None:
        _write_json(report_path, report)
    if result_log_path is not None:
        _write_result_log(result_log_path, report)
    return report


def audit_c3_mapper_window_sidecar(
    *,
    chunk_cache_path: str | Path = DEFAULT_BEAT_CHUNK_CACHE_PATH,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    sidecar_path: str | Path | None = DEFAULT_MAPPER_SIDECAR_PATH,
    report_path: str | Path | None = DEFAULT_MAPPER_SIDECAR_REPORT_PATH,
    result_log_path: str | Path | None = DEFAULT_MAPPER_SIDECAR_RESULT_LOG_PATH,
    motif_vocab_size: int = DEFAULT_MOTIF_VOCAB_SIZE,
    motif_min_n: int = DEFAULT_MOTIF_MIN_N,
    motif_max_n: int = DEFAULT_MOTIF_MAX_N,
    smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
    rare_max_count: int = DEFAULT_RARE_MAX_COUNT,
    lz_window_fallbacks: int = DEFAULT_LZ_WINDOW_FALLBACKS,
    lz_max_span: int = DEFAULT_LZ_MAX_SPAN,
    mapper_window_ms: int = MAPPER_WRITE_MS,
    sidecar_max_tokens: int = DEFAULT_MAPPER_SIDECAR_MAX_TOKENS,
    boundary_risk_margin_ms: int = DEFAULT_BOUNDARY_RISK_MARGIN_MS,
    limit_chunks: int | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    """Build a mapper-window C3 side-stream token sidecar from the P0 C3 plan."""

    del rare_max_count
    started_at = time.perf_counter()
    chunk_cache_path = Path(chunk_cache_path)
    dataset_root = Path(dataset_root)
    sidecar_path = None if sidecar_path is None else Path(sidecar_path)
    report_path = None if report_path is None else Path(report_path)
    result_log_path = None if result_log_path is None else Path(result_log_path)
    _validate_positive(motif_vocab_size, "motif_vocab_size")
    _validate_positive(motif_min_n, "motif_min_n")
    _validate_positive(motif_max_n, "motif_max_n")
    _validate_positive(lz_window_fallbacks, "lz_window_fallbacks")
    _validate_positive(lz_max_span, "lz_max_span")
    _validate_positive(mapper_window_ms, "mapper_window_ms")
    _validate_positive(sidecar_max_tokens, "sidecar_max_tokens")
    _validate_positive(boundary_risk_margin_ms, "boundary_risk_margin_ms")
    if motif_max_n < motif_min_n:
        raise ValueError("motif_max_n must be >= motif_min_n")

    chunk_df = _read_chunk_cache_for_mapper_sidecar(chunk_cache_path, limit_chunks=limit_chunks)
    baseline_model = _fit_motif_model(
        chunk_df,
        name="r0_delta",
        token_mapper=_baseline_tokens,
        motif_vocab_size=motif_vocab_size,
        motif_min_n=motif_min_n,
        motif_max_n=motif_max_n,
        smoothing_alpha=smoothing_alpha,
        top_motif_limit=20,
    )
    records, record_summary = _collect_fallback_records(chunk_df, baseline_model=baseline_model)
    plan = _build_selected_skeleton_residual_plan(
        records,
        window_fallbacks=lz_window_fallbacks,
        max_span=lz_max_span,
    )
    traced = _trace_side_stream_tokens(records, plan)
    side_decode = _decode_traced_side_stream(records, traced)
    main_guard = _main_stream_reconstruction_guard(
        chunk_df,
        baseline_model=baseline_model,
        records=records,
        side_decoded_by_record_id=side_decode.decoded_by_record_id,
    )
    anchors = _chunk_window_anchors(
        chunk_df,
        dataset_root=dataset_root,
        mapper_window_ms=mapper_window_ms,
        boundary_risk_margin_ms=boundary_risk_margin_ms,
    )
    sidecar, sidecar_stats = _build_mapper_window_sidecar_payload(
        chunk_df,
        traced,
        anchors=anchors,
        mapper_window_ms=mapper_window_ms,
        sidecar_max_tokens=sidecar_max_tokens,
    )
    loader_guard = {"checked": False}
    if sidecar_path is not None:
        _write_sidecar_json(sidecar_path, sidecar)
        loader_guard = _loader_guard(sidecar_path, sidecar)
    span_window_stats = _span_window_stats(plan, records, anchors=anchors)
    reconstruction_guard = {
        "pass": (
            side_decode.reference_error_count == 0
            and side_decode.mismatch_count == 0
            and int(main_guard["full_token_stream_mismatch_count"]) == 0
            and int(main_guard["group_signature_mismatch_count"]) == 0
            and int(main_guard["missing_side_payload_count"]) == 0
        ),
        "side_stream_reference_error_count": side_decode.reference_error_count,
        "side_stream_payload_mismatch_count": side_decode.mismatch_count,
        "full_token_stream_mismatch_count": main_guard["full_token_stream_mismatch_count"],
        "group_signature_mismatch_count": main_guard["group_signature_mismatch_count"],
        "missing_side_payload_count": main_guard["missing_side_payload_count"],
        "examples": [*side_decode.examples, *main_guard["examples"]][:10],
    }
    pass_criteria = {
        "p3_sidecar_generation_pass": (
            bool(reconstruction_guard["pass"])
            and bool(sidecar_stats["token_preservation_pass"])
            and int(sidecar_stats["missing_anchor_token_count"]) == 0
            and (not loader_guard.get("checked") or bool(loader_guard.get("pass")))
        ),
        "token_preservation_pass": sidecar_stats["token_preservation_pass"],
        "loader_guard_pass": loader_guard.get("pass") if loader_guard.get("checked") else None,
        "reconstruction_pass": reconstruction_guard["pass"],
        "full_cache_pass": limit_chunks is None,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "C3 mapper-window sidecar generation P3",
        "command": command,
        "code_commit": _git_stdout("rev-parse", "HEAD"),
        "code_dirty": bool(_git_stdout("status", "--short")),
        "chunk_cache_path": chunk_cache_path.as_posix(),
        "dataset_root": dataset_root.as_posix(),
        "sidecar_path": None if sidecar_path is None else sidecar_path.as_posix(),
        "report_path": None if report_path is None else report_path.as_posix(),
        "result_log_path": None if result_log_path is None else result_log_path.as_posix(),
        "limited": limit_chunks is not None,
        "limit_chunks": limit_chunks,
        "elapsed_s": time.perf_counter() - started_at,
        "config": {
            "motif_vocab_size": motif_vocab_size,
            "motif_min_n": motif_min_n,
            "motif_max_n": motif_max_n,
            "smoothing_alpha": smoothing_alpha,
            "lz_window_fallbacks": lz_window_fallbacks,
            "lz_max_span": lz_max_span,
            "selected_c3_variant": SELECTED_C3_VARIANT,
            "mapper_window_ms": mapper_window_ms,
            "sidecar_max_tokens": sidecar_max_tokens,
            "boundary_risk_margin_ms": boundary_risk_margin_ms,
            "path_contract": "beatmap_path is resolved as dataset_root/shard/cache_beatmap_path",
            "timing_contract": "window assignment uses chunk_sort_ms, the earliest snapped event time in the chunk",
        },
        "dataset": _dataset_summary(chunk_df),
        "fallback_record_summary": record_summary,
        "selected_plan": {
            "variant": plan.name,
            "span_count": len(plan.lz_spans),
            "selected_fallback_literal_count": sum(span.length for span in plan.lz_spans),
            "metadata": plan.metadata,
        },
        "sidecar_stats": sidecar_stats,
        "span_window_stats": span_window_stats,
        "reconstruction_guard": reconstruction_guard,
        "loader_guard": loader_guard,
        "pass_criteria": pass_criteria,
        "recommendation": _mapper_sidecar_recommendation(
            pass_criteria,
            sidecar_stats=sidecar_stats,
            span_window_stats=span_window_stats,
            limit_chunks=limit_chunks,
        ),
    }
    report = _normalize_json(report)
    if report_path is not None:
        _write_json(report_path, report)
    if result_log_path is not None:
        _write_mapper_sidecar_result_log(result_log_path, report)
    return report


def _build_selected_skeleton_residual_plan(
    records: Sequence[_FallbackRecord],
    *,
    window_fallbacks: int,
    max_span: int,
) -> _CostPlan:
    policy = _LzPolicy(
        name=SELECTED_C3_VARIANT,
        label="P0 selected hardened C3 skeleton-residual fallback side stream",
        allowed_match_types=("skeleton",),
        activation="all",
        window_fallbacks=int(window_fallbacks),
        pointer_code="fixed_width",
    )
    options = _lz_candidate_options_by_record(records, policy=policy, max_span=max_span)
    return _build_lz_hardening_plan(
        records,
        options,
        policy=policy,
        max_span=max_span,
        pointer_model=None,
        table_model_cost_bits=0.0,
    )


def _read_chunk_cache_for_mapper_sidecar(path: Path, *, limit_chunks: int | None) -> pd.DataFrame:
    frame = _read_chunk_cache(path, limit_chunks=limit_chunks)
    for column, default in (("shard", ""), ("chunk_sort_ms", 0.0)):
        if column in frame.columns:
            continue
        try:
            extra = pd.read_parquet(path, columns=[column])
            if limit_chunks is not None:
                extra = extra.head(int(limit_chunks)).copy()
            frame[column] = extra[column].to_numpy()
        except (KeyError, ValueError, OSError):
            frame[column] = default
    return frame


def _encode_decode_side_stream(records: Sequence[_FallbackRecord], plan: _CostPlan) -> _SideDecodeResult:
    source_records = _records_by_source(records)
    spans_by_source: dict[int, dict[int, Any]] = defaultdict(dict)
    for span in plan.lz_spans:
        spans_by_source[int(span.source_row_index)][int(span.start_record_id)] = span

    decoded_by_record_id: dict[int, str] = {}
    side_token_count = 0
    raw_token_count = 0
    ref_token_count = 0
    residual_token_count = 0
    reference_error_count = 0
    mismatch_count = 0
    examples: list[dict[str, Any]] = []

    for source, ordered_records in source_records.items():
        side_tokens = _encode_side_stream_for_source(ordered_records, spans_by_source.get(source, {}))
        side_token_count += len(side_tokens)
        raw_token_count += sum(1 for token in side_tokens if token.startswith(RAW_PREFIX))
        ref_token_count += sum(1 for token in side_tokens if token.startswith(REF_PREFIX))
        residual_token_count += sum(1 for token in side_tokens if token.startswith(RES_PREFIX))
        decoded, errors = _decode_side_stream_for_source(ordered_records, side_tokens)
        decoded_by_record_id.update(decoded)
        reference_error_count += len(errors)
        examples.extend(errors[: max(0, 10 - len(examples))])
        for record in ordered_records:
            actual = decoded.get(record.id)
            if actual == record.token:
                continue
            mismatch_count += 1
            if len(examples) < 10:
                examples.append(
                    {
                        "guard": "side_payload",
                        "source_row_index": source,
                        "record_id": record.id,
                        "expected": record.token,
                        "decoded": actual,
                    }
                )

    return _SideDecodeResult(
        decoded_by_record_id=decoded_by_record_id,
        side_token_count=side_token_count,
        raw_token_count=raw_token_count,
        ref_token_count=ref_token_count,
        residual_token_count=residual_token_count,
        reference_error_count=reference_error_count,
        mismatch_count=mismatch_count,
        examples=tuple(examples[:10]),
    )


def _encode_side_stream_for_source(
    ordered_records: Sequence[_FallbackRecord],
    spans_by_start_record_id: Mapping[int, Any],
) -> list[str]:
    tokens: list[str] = []
    id_to_index = {record.id: index for index, record in enumerate(ordered_records)}
    index = 0
    while index < len(ordered_records):
        record = ordered_records[index]
        span = spans_by_start_record_id.get(record.id)
        if span is None:
            tokens.append(f"{RAW_PREFIX}{record.token}")
            index += 1
            continue
        previous_index = id_to_index[int(span.previous_record_id)]
        distance = index - previous_index
        tokens.append(f"{REF_PREFIX}{span.match_type}|{distance}|{span.length}")
        if span.match_type == "skeleton":
            for item in ordered_records[index : index + int(span.length)]:
                tokens.append(f"{RES_PREFIX}{item.token}")
        index += int(span.length)
    return tokens


def _trace_side_stream_tokens(records: Sequence[_FallbackRecord], plan: _CostPlan) -> dict[int, list[_TracedSideToken]]:
    source_records = _records_by_source(records)
    spans_by_source: dict[int, dict[int, Any]] = defaultdict(dict)
    for span in plan.lz_spans:
        spans_by_source[int(span.source_row_index)][int(span.start_record_id)] = span
    return {
        source: _encode_traced_side_stream_for_source(ordered_records, spans_by_source.get(source, {}))
        for source, ordered_records in sorted(source_records.items())
    }


def _encode_traced_side_stream_for_source(
    ordered_records: Sequence[_FallbackRecord],
    spans_by_start_record_id: Mapping[int, Any],
) -> list[_TracedSideToken]:
    traced: list[_TracedSideToken] = []
    id_to_index = {record.id: index for index, record in enumerate(ordered_records)}
    index = 0
    while index < len(ordered_records):
        record = ordered_records[index]
        span = spans_by_start_record_id.get(record.id)
        if span is None:
            traced.append(_TracedSideToken(f"{RAW_PREFIX}{record.token}", record, "raw"))
            index += 1
            continue
        previous_index = id_to_index[int(span.previous_record_id)]
        distance = index - previous_index
        traced.append(_TracedSideToken(f"{REF_PREFIX}{span.match_type}|{distance}|{span.length}", record, "ref"))
        if span.match_type == "skeleton":
            for item in ordered_records[index : index + int(span.length)]:
                traced.append(_TracedSideToken(f"{RES_PREFIX}{item.token}", item, "residual"))
        index += int(span.length)
    return traced


def _decode_traced_side_stream(
    records: Sequence[_FallbackRecord],
    traced_by_source: Mapping[int, Sequence[_TracedSideToken]],
) -> _SideDecodeResult:
    decoded_by_record_id: dict[int, str] = {}
    side_token_count = 0
    raw_token_count = 0
    ref_token_count = 0
    residual_token_count = 0
    reference_error_count = 0
    mismatch_count = 0
    examples: list[dict[str, Any]] = []
    source_records = _records_by_source(records)
    for source, ordered_records in source_records.items():
        traced = traced_by_source.get(source, ())
        side_tokens = [item.token for item in traced]
        side_token_count += len(side_tokens)
        raw_token_count += sum(1 for token in side_tokens if token.startswith(RAW_PREFIX))
        ref_token_count += sum(1 for token in side_tokens if token.startswith(REF_PREFIX))
        residual_token_count += sum(1 for token in side_tokens if token.startswith(RES_PREFIX))
        decoded, errors = _decode_side_stream_for_source(ordered_records, side_tokens)
        decoded_by_record_id.update(decoded)
        reference_error_count += len(errors)
        examples.extend(errors[: max(0, 10 - len(examples))])
        for record in ordered_records:
            actual = decoded.get(record.id)
            if actual == record.token:
                continue
            mismatch_count += 1
            if len(examples) < 10:
                examples.append(
                    {
                        "guard": "side_payload",
                        "source_row_index": source,
                        "record_id": record.id,
                        "expected": record.token,
                        "decoded": actual,
                    }
                )
    return _SideDecodeResult(
        decoded_by_record_id=decoded_by_record_id,
        side_token_count=side_token_count,
        raw_token_count=raw_token_count,
        ref_token_count=ref_token_count,
        residual_token_count=residual_token_count,
        reference_error_count=reference_error_count,
        mismatch_count=mismatch_count,
        examples=tuple(examples[:10]),
    )


def _decode_side_stream_for_source(
    ordered_records: Sequence[_FallbackRecord],
    side_tokens: Sequence[str],
) -> tuple[dict[int, str], list[dict[str, Any]]]:
    decoded: dict[int, str] = {}
    history: list[str] = []
    errors: list[dict[str, Any]] = []
    record_index = 0
    token_index = 0
    while record_index < len(ordered_records) and token_index < len(side_tokens):
        token = side_tokens[token_index]
        token_index += 1
        if token.startswith(RAW_PREFIX):
            payload = token[len(RAW_PREFIX) :]
            decoded[ordered_records[record_index].id] = payload
            history.append(payload)
            record_index += 1
            continue
        if not token.startswith(REF_PREFIX):
            errors.append({"guard": "side_stream", "reason": "unknown_token", "token": token})
            break
        match_type, distance, length = _parse_ref_token(token)
        previous_index = len(history) - distance
        if previous_index < 0 or previous_index + length > len(history):
            errors.append(
                {
                    "guard": "side_stream",
                    "reason": "reference_not_prior",
                    "record_index": record_index,
                    "distance": distance,
                    "length": length,
                    "history_count": len(history),
                }
            )
            break
        for offset in range(length):
            previous = history[previous_index + offset]
            if match_type == "exact":
                payload = previous
            elif match_type == "mirror":
                payload = _mirror_atom_token(previous)
            elif match_type == "skeleton":
                if token_index >= len(side_tokens) or not side_tokens[token_index].startswith(RES_PREFIX):
                    errors.append({"guard": "side_stream", "reason": "missing_skeleton_residual", "record_index": record_index})
                    return decoded, errors
                payload = side_tokens[token_index][len(RES_PREFIX) :]
                token_index += 1
                if _skeleton_signature(payload) != _skeleton_signature(previous):
                    errors.append(
                        {
                            "guard": "side_stream",
                            "reason": "skeleton_residual_mismatch",
                            "record_index": record_index,
                            "previous": previous,
                            "payload": payload,
                        }
                    )
            else:
                errors.append({"guard": "side_stream", "reason": "unknown_ref_type", "match_type": match_type})
                return decoded, errors
            if record_index >= len(ordered_records):
                errors.append({"guard": "side_stream", "reason": "ref_overruns_records"})
                return decoded, errors
            decoded[ordered_records[record_index].id] = payload
            history.append(payload)
            record_index += 1
    if record_index != len(ordered_records):
        errors.append(
            {
                "guard": "side_stream",
                "reason": "record_count_mismatch",
                "decoded_record_count": record_index,
                "expected_record_count": len(ordered_records),
            }
        )
    if token_index != len(side_tokens):
        errors.append(
            {
                "guard": "side_stream",
                "reason": "unused_side_tokens",
                "used_token_count": token_index,
                "side_token_count": len(side_tokens),
            }
        )
    return decoded, errors


def _parse_ref_token(token: str) -> tuple[str, int, int]:
    parts = token[len(REF_PREFIX) :].split("|")
    if len(parts) != 3:
        raise ValueError(f"invalid C3 REF token: {token!r}")
    match_type, distance, length = parts
    return match_type, int(distance), int(length)


def _main_stream_reconstruction_guard(
    chunk_df: pd.DataFrame,
    *,
    baseline_model: Any,
    records: Sequence[_FallbackRecord],
    side_decoded_by_record_id: Mapping[int, str],
) -> dict[str, Any]:
    id_to_motif = {token_id: motif for motif, token_id in baseline_model.vocab.motif_to_id.items()}
    records_by_chunk = _records_by_chunk(records)
    full_token_stream_mismatch_count = 0
    group_signature_mismatch_count = 0
    missing_side_payload_count = 0
    checked_chunk_count = 0
    baseline_encoded_token_count = 0
    main_stream_token_count = 0
    fallback_placeholder_count = 0
    examples: list[dict[str, Any]] = []

    ordered = chunk_df.sort_values(["source_row_index", "segment_id", "start_beat_units", "chunk_index"])
    for row in ordered.itertuples(index=False):
        expected_tokens = _baseline_tokens(row)
        encoded_tokens, _covered = _encode_sequence(expected_tokens, baseline_model.trie)
        chunk_key = (int(row.source_row_index), int(row.segment_id), int(row.chunk_index))
        fallback_records = deque(records_by_chunk.get(chunk_key, ()))
        main_tokens: list[str] = []
        decoded_tokens: list[str] = []
        for token in encoded_tokens:
            baseline_encoded_token_count += 1
            if _is_fallback_atom_token(token):
                main_tokens.append(FALLBACK_PLACEHOLDER_TOKEN)
                fallback_placeholder_count += 1
                if not fallback_records:
                    missing_side_payload_count += 1
                    decoded_tokens.append(token)
                    continue
                record = fallback_records.popleft()
                payload = side_decoded_by_record_id.get(record.id)
                if payload is None:
                    missing_side_payload_count += 1
                    payload = token
                decoded_tokens.append(payload)
                continue
            main_tokens.append(token)
            decoded_tokens.extend(_decode_motif_tokens([token], id_to_motif))
        main_stream_token_count += len(main_tokens)
        checked_chunk_count += 1
        if list(fallback_records):
            missing_side_payload_count += len(fallback_records)
        if decoded_tokens != expected_tokens:
            full_token_stream_mismatch_count += 1
            if len(examples) < 10:
                examples.append(
                    {
                        "guard": "main_stream",
                        "source_row_index": int(row.source_row_index),
                        "chunk_index": int(row.chunk_index),
                        "expected": expected_tokens[:20],
                        "decoded": decoded_tokens[:20],
                        "main_tokens": main_tokens[:20],
                    }
                )
        reconstructed_signature = _signature_from_delta_tokens(decoded_tokens)
        expected_signature = str(row.raw_signature or "")
        if reconstructed_signature != expected_signature:
            group_signature_mismatch_count += 1
            if len(examples) < 10:
                examples.append(
                    {
                        "guard": "group_signature",
                        "source_row_index": int(row.source_row_index),
                        "chunk_index": int(row.chunk_index),
                        "expected": expected_signature,
                        "decoded": reconstructed_signature,
                    }
                )
    return {
        "checked_chunk_count": checked_chunk_count,
        "baseline_encoded_token_count": baseline_encoded_token_count,
        "main_stream_token_count": main_stream_token_count,
        "fallback_placeholder_count": fallback_placeholder_count,
        "full_token_stream_mismatch_count": full_token_stream_mismatch_count,
        "group_signature_mismatch_count": group_signature_mismatch_count,
        "missing_side_payload_count": missing_side_payload_count,
        "examples": examples,
    }


def _records_by_source(records: Sequence[_FallbackRecord]) -> dict[int, list[_FallbackRecord]]:
    grouped: dict[int, list[_FallbackRecord]] = defaultdict(list)
    for record in records:
        grouped[int(record.source_row_index)].append(record)
    for values in grouped.values():
        values.sort(key=lambda item: (item.segment_id, item.absolute_units, item.chunk_index, item.group_index))
    return grouped


def _records_by_chunk(records: Sequence[_FallbackRecord]) -> dict[tuple[int, int, int], list[_FallbackRecord]]:
    grouped: dict[tuple[int, int, int], list[_FallbackRecord]] = defaultdict(list)
    for record in records:
        grouped[(int(record.source_row_index), int(record.segment_id), int(record.chunk_index))].append(record)
    for values in grouped.values():
        values.sort(key=lambda item: item.group_index)
    return grouped


def _is_fallback_atom_token(token: str) -> bool:
    return str(token).startswith("D:")


def _signature_from_delta_tokens(tokens: Sequence[str]) -> str:
    offset = 0
    atoms: list[str] = []
    for token in tokens:
        value = str(token)
        if value.startswith("C"):
            continue
        payload = _parse_atom_token(value)
        offset += payload.delta
        atom = f"A:{offset}:{payload.tap}:{payload.start}:{payload.end}"
        if payload.order_signature != ".":
            atom = f"{atom}:O{payload.order_signature}"
        atoms.append(atom)
    return ";".join(atoms)


def _parse_atom_token(token: str) -> _AtomPayload:
    value = str(token)
    if ":O" in value:
        base, order = value.split(":O", 1)
    else:
        base, order = value, "."
    parts = base.split(":")
    if len(parts) != 5 or parts[0] != "D":
        raise ValueError(f"invalid r0_delta atom token: {token!r}")
    return _AtomPayload(
        delta=int(parts[1]),
        tap=int(parts[2]),
        start=int(parts[3]),
        end=int(parts[4]),
        order_signature=order or ".",
    )


def _mirror_atom_token(token: str) -> str:
    payload = _parse_atom_token(token)
    base = "D:{delta}:{tap}:{start}:{end}".format(
        delta=payload.delta,
        tap=_mirror_mask_4(payload.tap),
        start=_mirror_mask_4(payload.start),
        end=_mirror_mask_4(payload.end),
    )
    order = _mirror_order_signature_4(payload.order_signature)
    return base if order == "." else f"{base}:O{order}"


def _skeleton_signature(token: str) -> tuple[int, str, int, int, int, str]:
    payload = _parse_atom_token(token)
    group_class = _group_class(payload.tap, payload.start, payload.end)
    return (
        payload.delta,
        group_class,
        payload.tap.bit_count(),
        payload.start.bit_count(),
        payload.end.bit_count(),
        "." if payload.order_signature == "." else "order",
    )


def _group_class(tap: int, start: int, end: int) -> str:
    parts: list[str] = []
    if tap:
        parts.append("tap")
    if start:
        parts.append("ln_start")
    if end:
        parts.append("ln_end")
    return "_".join(parts) if parts else "empty"


def _span_stats(plan: _CostPlan) -> dict[str, Any]:
    spans = plan.lz_spans
    selected_fallbacks = sum(span.length for span in spans)
    match_counter = Counter(span.match_type for span in spans)
    return {
        "span_count": len(spans),
        "selected_fallback_literal_count": selected_fallbacks,
        "noncontiguous_main_stream_span_count": sum(1 for span in spans if not span.full_group_contiguous),
        "target_cross_chunk_span_count": sum(1 for span in spans if span.target_crosses_chunk),
        "reference_cross_chunk_span_count": sum(1 for span in spans if span.reference_crosses_chunk),
        "match_type_usage": dict(match_counter.most_common()),
        "mean_span_length": float(selected_fallbacks) / float(len(spans)) if spans else 0.0,
    }


def _chunk_window_anchors(
    chunk_df: pd.DataFrame,
    *,
    dataset_root: Path,
    mapper_window_ms: int,
    boundary_risk_margin_ms: int,
) -> dict[tuple[int, int, int], _WindowAnchor]:
    anchors: dict[tuple[int, int, int], _WindowAnchor] = {}
    for row in chunk_df.itertuples(index=False):
        chunk_sort_ms = _row_float(row, "chunk_sort_ms", default=0.0)
        window_start_ms = _window_start_ms(chunk_sort_ms, mapper_window_ms=mapper_window_ms)
        modulo_ms = float(chunk_sort_ms) - float(window_start_ms)
        anchors[(int(row.source_row_index), int(row.segment_id), int(row.chunk_index))] = _WindowAnchor(
            beatmap_path=_resolved_mapper_beatmap_path(row, dataset_root=dataset_root),
            raw_beatmap_path=str(getattr(row, "beatmap_path", "")),
            shard=str(getattr(row, "shard", "")),
            split=str(getattr(row, "split", "")),
            source_row_index=int(row.source_row_index),
            beatmap_set_id=int(getattr(row, "beatmap_set_id", 0) or 0),
            beatmap_id=int(getattr(row, "beatmap_id", 0) or 0),
            window_start_ms=window_start_ms,
            chunk_sort_ms=float(chunk_sort_ms),
            near_boundary=modulo_ms >= float(max(0, mapper_window_ms - boundary_risk_margin_ms)),
        )
    return anchors


def _build_mapper_window_sidecar_payload(
    chunk_df: pd.DataFrame,
    traced_by_source: Mapping[int, Sequence[_TracedSideToken]],
    *,
    anchors: Mapping[tuple[int, int, int], _WindowAnchor],
    mapper_window_ms: int,
    sidecar_max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    token_to_id: dict[str, int] = {}
    windows: dict[tuple[str, int], dict[str, Any]] = {}
    for anchor in anchors.values():
        key = (anchor.beatmap_path, anchor.window_start_ms)
        windows.setdefault(
            key,
            {
                "beatmap_path": anchor.beatmap_path,
                "raw_beatmap_path": anchor.raw_beatmap_path,
                "shard": anchor.shard,
                "source_row_index": anchor.source_row_index,
                "beatmap_set_id": anchor.beatmap_set_id,
                "beatmap_id": anchor.beatmap_id,
                "split": anchor.split,
                "window_start_ms": anchor.window_start_ms,
                "token_ids": [],
                "boundary_risk_token_count": 0,
            },
        )

    traced_side_stream_token_count = 0
    missing_anchor_token_count = 0
    boundary_risk_token_count = 0
    token_kind_counter: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for source, traced_tokens in sorted(traced_by_source.items()):
        for traced in traced_tokens:
            traced_side_stream_token_count += 1
            token_kind_counter[traced.kind] += 1
            anchor = anchors.get((int(traced.record.source_row_index), int(traced.record.segment_id), int(traced.record.chunk_index)))
            if anchor is None:
                missing_anchor_token_count += 1
                if len(examples) < 10:
                    examples.append(
                        {
                            "reason": "missing_chunk_anchor",
                            "source_row_index": source,
                            "segment_id": traced.record.segment_id,
                            "chunk_index": traced.record.chunk_index,
                            "record_id": traced.record.id,
                        }
                    )
                continue
            token_id = token_to_id.setdefault(traced.token, len(token_to_id) + 1)
            window = windows[(anchor.beatmap_path, anchor.window_start_ms)]
            window["token_ids"].append(token_id)
            if anchor.near_boundary:
                window["boundary_risk_token_count"] += 1
                boundary_risk_token_count += 1

    sidecar_rows = []
    lengths: list[int] = []
    nonzero_lengths: list[int] = []
    for _key, row in sorted(windows.items(), key=lambda item: (item[1]["beatmap_path"], item[1]["window_start_ms"])):
        token_ids = list(row["token_ids"])
        token_count = len(token_ids)
        lengths.append(token_count)
        if token_count:
            nonzero_lengths.append(token_count)
        sidecar_rows.append(
            {
                "beatmap_path": row["beatmap_path"],
                "window_start_ms": int(row["window_start_ms"]),
                "token_ids": token_ids,
                "source_row_index": int(row["source_row_index"]),
                "split": row["split"],
                "raw_beatmap_path": row["raw_beatmap_path"],
                "shard": row["shard"],
                "beatmap_set_id": int(row["beatmap_set_id"]),
                "beatmap_id": int(row["beatmap_id"]),
                "boundary_risk_token_count": int(row["boundary_risk_token_count"]),
            }
        )
    sidecar_token_count = sum(lengths)
    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "contract": MAPPER_SIDECAR_CONTRACT,
        "token_pad_id": MAPPER_SIDECAR_PAD_ID,
        "token_id_base": 1,
        "mapper_window_ms": int(mapper_window_ms),
        "token_vocab": {token: token_id for token, token_id in sorted(token_to_id.items(), key=lambda item: item[1])},
        "windows": sidecar_rows,
    }
    stats = {
        "window_count": len(sidecar_rows),
        "window_with_tokens_count": len(nonzero_lengths),
        "window_with_tokens_rate": float(len(nonzero_lengths)) / float(len(sidecar_rows)) if sidecar_rows else 0.0,
        "traced_side_stream_token_count": traced_side_stream_token_count,
        "sidecar_token_count": sidecar_token_count,
        "token_preservation_pass": sidecar_token_count == traced_side_stream_token_count,
        "missing_anchor_token_count": missing_anchor_token_count,
        "token_vocab_size": len(token_to_id),
        "token_kind_counts": dict(token_kind_counter),
        "tokens_per_window": _distribution(lengths),
        "tokens_per_nonzero_window": _distribution(nonzero_lengths),
        "cap_sweep": _cap_sweep(lengths, caps=(64, 128, sidecar_max_tokens, 512, 1024)),
        "boundary_risk_token_count": boundary_risk_token_count,
        "boundary_risk_token_rate": (
            float(boundary_risk_token_count) / float(sidecar_token_count) if sidecar_token_count else 0.0
        ),
        "examples": examples,
    }
    return sidecar, stats


def _span_window_stats(
    plan: _CostPlan,
    records: Sequence[_FallbackRecord],
    *,
    anchors: Mapping[tuple[int, int, int], _WindowAnchor],
) -> dict[str, Any]:
    source_records = _records_by_source(records)
    by_source_index: dict[int, dict[int, _FallbackRecord]] = {
        source: {index: record for index, record in enumerate(values)}
        for source, values in source_records.items()
    }
    target_cross_window = 0
    reference_cross_window = 0
    target_reference_same_window = 0
    target_reference_window_delta: Counter[int] = Counter()
    missing_anchor_spans = 0
    for span in plan.lz_spans:
        records_by_index = by_source_index.get(int(span.source_row_index), {})
        target_windows = _span_windows(
            records_by_index,
            int(span.start_index),
            int(span.length),
            anchors=anchors,
        )
        reference_windows = _span_windows(
            records_by_index,
            int(span.previous_index),
            int(span.length),
            anchors=anchors,
        )
        if not target_windows or not reference_windows:
            missing_anchor_spans += 1
            continue
        if len(target_windows) > 1:
            target_cross_window += 1
        if len(reference_windows) > 1:
            reference_cross_window += 1
        target_start = min(target_windows)
        reference_start = min(reference_windows)
        delta = int(target_start - reference_start)
        target_reference_window_delta[delta] += 1
        if delta == 0:
            target_reference_same_window += 1
    span_count = len(plan.lz_spans)
    return {
        "span_count": span_count,
        "target_cross_mapper_window_span_count": target_cross_window,
        "target_cross_mapper_window_span_rate": float(target_cross_window) / float(span_count) if span_count else 0.0,
        "reference_cross_mapper_window_span_count": reference_cross_window,
        "reference_cross_mapper_window_span_rate": float(reference_cross_window) / float(span_count) if span_count else 0.0,
        "target_reference_same_window_span_count": target_reference_same_window,
        "target_reference_same_window_span_rate": float(target_reference_same_window) / float(span_count) if span_count else 0.0,
        "missing_anchor_span_count": missing_anchor_spans,
        "target_reference_window_delta_top": [
            {"delta_ms": int(delta), "count": int(count)}
            for delta, count in target_reference_window_delta.most_common(10)
        ],
    }


def _span_windows(
    records_by_index: Mapping[int, _FallbackRecord],
    start_index: int,
    length: int,
    *,
    anchors: Mapping[tuple[int, int, int], _WindowAnchor],
) -> set[int]:
    windows: set[int] = set()
    for index in range(start_index, start_index + length):
        record = records_by_index.get(index)
        if record is None:
            continue
        anchor = anchors.get((int(record.source_row_index), int(record.segment_id), int(record.chunk_index)))
        if anchor is not None:
            windows.add(int(anchor.window_start_ms))
    return windows


def _recommendation(reconstruction_guard: Mapping[str, Any], *, limit_chunks: int | None) -> str:
    if not reconstruction_guard.get("pass"):
        return "MUTATE: P0 side-stream artifact failed lossless reconstruction; do not move to mapper-side fields."
    if limit_chunks is not None:
        return "TEST_FULL: P0 side-stream artifact passed on a limited slice; run full-cache before mapper-side fields."
    return "TEST_NEXT: P0 side-stream artifact passed; implement optional mapper dataset shadow fields behind a disabled-by-default flag."


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(_normalize_json(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _write_result_log(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    guard = report.get("reconstruction_guard", {})
    token_stats = report.get("token_stats", {})
    span_stats = report.get("span_stats", {})
    pass_criteria = report.get("pass_criteria", {})
    lines = [
        "# C3 Side-Stream Pipeline Tokenization Result Log",
        "",
        "## Summary",
        "",
        f"- Recommendation: {report.get('recommendation')}",
        f"- P0 roundtrip pass: {pass_criteria.get('p0_roundtrip_pass') if isinstance(pass_criteria, Mapping) else None}",
        f"- Full cache: {pass_criteria.get('full_cache_pass') if isinstance(pass_criteria, Mapping) else None}",
        f"- Runtime seconds: {_fmt_float(report.get('elapsed_s'))}",
        f"- Limited: {report.get('limited')}",
        f"- Code dirty: {report.get('code_dirty')}",
        f"- Reconstruction pass: {guard.get('pass') if isinstance(guard, Mapping) else None}",
        f"- Side reference errors: {guard.get('side_stream_reference_error_count') if isinstance(guard, Mapping) else None}",
        f"- Side payload mismatches: {guard.get('side_stream_payload_mismatch_count') if isinstance(guard, Mapping) else None}",
        f"- Full token-stream mismatches: {guard.get('full_token_stream_mismatch_count') if isinstance(guard, Mapping) else None}",
        f"- Group signature mismatches: {guard.get('group_signature_mismatch_count') if isinstance(guard, Mapping) else None}",
        f"- Baseline encoded tokens: {token_stats.get('baseline_encoded_token_count') if isinstance(token_stats, Mapping) else None}",
        f"- Main stream tokens: {token_stats.get('main_stream_token_count') if isinstance(token_stats, Mapping) else None}",
        f"- Side stream tokens: {token_stats.get('side_stream_token_count') if isinstance(token_stats, Mapping) else None}",
        f"- Combined/main baseline ratio: {token_stats.get('combined_to_baseline_token_ratio') if isinstance(token_stats, Mapping) else None}",
        f"- Selected spans: {span_stats.get('span_count') if isinstance(span_stats, Mapping) else None}",
        f"- Noncontiguous spans: {span_stats.get('noncontiguous_main_stream_span_count') if isinstance(span_stats, Mapping) else None}",
        f"- Target cross-chunk spans: {span_stats.get('target_cross_chunk_span_count') if isinstance(span_stats, Mapping) else None}",
        "",
        "## Interpretation",
        "",
        str(report.get("recommendation")),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_mapper_sidecar_result_log(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_stats = report.get("sidecar_stats", {})
    span_stats = report.get("span_window_stats", {})
    guard = report.get("reconstruction_guard", {})
    loader_guard = report.get("loader_guard", {})
    pass_criteria = report.get("pass_criteria", {})
    tokens_per_window = sidecar_stats.get("tokens_per_window", {}) if isinstance(sidecar_stats, Mapping) else {}
    cap_sweep = sidecar_stats.get("cap_sweep", {}) if isinstance(sidecar_stats, Mapping) else {}
    lines = [
        "# C3 Mapper-Window Sidecar Generation Result Log",
        "",
        "## Summary",
        "",
        f"- Recommendation: {report.get('recommendation')}",
        f"- P3 sidecar generation pass: {pass_criteria.get('p3_sidecar_generation_pass') if isinstance(pass_criteria, Mapping) else None}",
        f"- Full cache: {pass_criteria.get('full_cache_pass') if isinstance(pass_criteria, Mapping) else None}",
        f"- Runtime seconds: {_fmt_float(report.get('elapsed_s'))}",
        f"- Limited: {report.get('limited')}",
        f"- Code dirty: {report.get('code_dirty')}",
        f"- Reconstruction pass: {guard.get('pass') if isinstance(guard, Mapping) else None}",
        f"- Loader guard pass: {loader_guard.get('pass') if isinstance(loader_guard, Mapping) else None}",
        f"- Sidecar token preservation pass: {sidecar_stats.get('token_preservation_pass') if isinstance(sidecar_stats, Mapping) else None}",
        f"- Sidecar tokens: {sidecar_stats.get('sidecar_token_count') if isinstance(sidecar_stats, Mapping) else None}",
        f"- Traced side-stream tokens: {sidecar_stats.get('traced_side_stream_token_count') if isinstance(sidecar_stats, Mapping) else None}",
        f"- Window count: {sidecar_stats.get('window_count') if isinstance(sidecar_stats, Mapping) else None}",
        f"- Windows with tokens: {sidecar_stats.get('window_with_tokens_count') if isinstance(sidecar_stats, Mapping) else None}",
        f"- Token vocab size: {sidecar_stats.get('token_vocab_size') if isinstance(sidecar_stats, Mapping) else None}",
        f"- Tokens/window p95: {tokens_per_window.get('p95') if isinstance(tokens_per_window, Mapping) else None}",
        f"- Tokens/window p99: {tokens_per_window.get('p99') if isinstance(tokens_per_window, Mapping) else None}",
        f"- Tokens/window max: {tokens_per_window.get('max') if isinstance(tokens_per_window, Mapping) else None}",
        f"- Cap sweep: {cap_sweep if isinstance(cap_sweep, Mapping) else None}",
        f"- Boundary-risk token rate: {sidecar_stats.get('boundary_risk_token_rate') if isinstance(sidecar_stats, Mapping) else None}",
        f"- Target cross-window span rate: {span_stats.get('target_cross_mapper_window_span_rate') if isinstance(span_stats, Mapping) else None}",
        "",
        "## Interpretation",
        "",
        str(report.get("recommendation")),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_sidecar_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(_normalize_json(payload), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _loader_guard(sidecar_path: Path | None, payload: Mapping[str, Any]) -> dict[str, Any]:
    if sidecar_path is None or not sidecar_path.exists():
        return {"checked": False}
    try:
        from pulsefield_model.data.mapper_sparse_windows_v2_1 import load_c3_side_stream_token_sidecar

        loaded = load_c3_side_stream_token_sidecar(sidecar_path)
    except Exception as exc:  # noqa: BLE001 - report loader incompatibility.
        return {
            "checked": True,
            "pass": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:500],
        }
    expected_windows = payload.get("windows", [])
    expected_window_count = len(expected_windows) if isinstance(expected_windows, Sequence) else 0
    expected_token_count = (
        sum(len(row.get("token_ids", ())) for row in expected_windows if isinstance(row, Mapping))
        if isinstance(expected_windows, Sequence)
        else 0
    )
    loaded_window_count = sum(len(windows) for windows in loaded.values())
    loaded_token_count = sum(len(tokens) for windows in loaded.values() for tokens in windows.values())
    return {
        "checked": True,
        "pass": loaded_window_count == expected_window_count and loaded_token_count == expected_token_count,
        "loaded_beatmap_count": len(loaded),
        "loaded_window_count": loaded_window_count,
        "expected_window_count": expected_window_count,
        "loaded_token_count": loaded_token_count,
        "expected_token_count": expected_token_count,
    }


def _mapper_sidecar_recommendation(
    pass_criteria: Mapping[str, Any],
    *,
    sidecar_stats: Mapping[str, Any],
    span_window_stats: Mapping[str, Any],
    limit_chunks: int | None,
) -> str:
    if not pass_criteria.get("p3_sidecar_generation_pass"):
        return "MUTATE: mapper-window C3 sidecar generation failed preservation, reconstruction, or loader gates."
    if limit_chunks is not None:
        return "TEST_FULL: mapper-window C3 sidecar passed on a limited slice; run full-cache generation."
    cap_sweep = sidecar_stats.get("cap_sweep")
    default_cap = cap_sweep.get(str(DEFAULT_MAPPER_SIDECAR_MAX_TOKENS), {}) if isinstance(cap_sweep, Mapping) else {}
    trunc_rate = float(default_cap.get("truncated_window_rate", 0.0) or 0.0) if isinstance(default_cap, Mapping) else 0.0
    boundary_rate = float(sidecar_stats.get("boundary_risk_token_rate", 0.0) or 0.0)
    target_cross_rate = float(span_window_stats.get("target_cross_mapper_window_span_rate", 0.0) or 0.0)
    if trunc_rate > 0.05:
        return "MUTATE: full-cache C3 sidecar works, but default cap truncation is high; run a cap/packing sweep before model conditioning."
    if boundary_rate > 0.10 or target_cross_rate > 0.10:
        return "MUTATE: full-cache C3 sidecar works, but chunk-sort window anchoring risk is high; compare against exact per-group timing."
    return "TEST_NEXT: full-cache C3 mapper-window sidecar is tractable; design a disabled-by-default model-conditioning probe."


def _resolved_mapper_beatmap_path(row: object, *, dataset_root: Path) -> str:
    raw_path = Path(str(getattr(row, "beatmap_path", "")))
    if raw_path.is_absolute():
        return raw_path.as_posix()
    shard = str(getattr(row, "shard", ""))
    if shard:
        return (dataset_root / shard / raw_path).as_posix()
    return (dataset_root / raw_path).as_posix()


def _window_start_ms(chunk_sort_ms: float, *, mapper_window_ms: int) -> int:
    safe_ms = max(float(chunk_sort_ms), 0.0)
    return int(safe_ms // int(mapper_window_ms)) * int(mapper_window_ms)


def _row_float(row: object, field: str, *, default: float) -> float:
    value = getattr(row, field, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _distribution(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0,
        }
    series = pd.Series([int(value) for value in values], dtype="int64")
    return {
        "count": int(series.shape[0]),
        "mean": float(series.mean()),
        "p50": float(series.quantile(0.50)),
        "p90": float(series.quantile(0.90)),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
        "max": int(series.max()),
    }


def _cap_sweep(values: Sequence[int], *, caps: Sequence[int]) -> dict[str, Any]:
    lengths = [int(value) for value in values]
    window_count = len(lengths)
    result: dict[str, Any] = {}
    for cap in sorted({int(value) for value in caps if int(value) > 0}):
        truncated = [length for length in lengths if length > cap]
        overflow_tokens = sum(max(0, length - cap) for length in lengths)
        result[str(cap)] = {
            "truncated_window_count": len(truncated),
            "truncated_window_rate": float(len(truncated)) / float(window_count) if window_count else 0.0,
            "overflow_token_count": int(overflow_tokens),
            "overflow_token_rate": float(overflow_tokens) / float(sum(lengths)) if sum(lengths) else 0.0,
            "max_overflow_tokens": max((length - cap for length in truncated), default=0),
        }
    return result


def _fmt_float(value: object) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _validate_positive(value: int, name: str) -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _git_stdout(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[3],
        )
    except OSError:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _command(args: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in ["python", "-m", "pulsefield_model.osu_core.c3_side_stream_tokenization", *args])


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate C3 fallback side-stream tokenization artifact.")
    parser.add_argument("--mapper-sidecar", action="store_true", help="build mapper-window C3 sidecar instead of P0 audit")
    parser.add_argument("--chunk-cache-path", type=Path, default=DEFAULT_BEAT_CHUNK_CACHE_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--sidecar-path", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--result-log-path", type=Path, default=None)
    parser.add_argument("--motif-vocab-size", type=int, default=DEFAULT_MOTIF_VOCAB_SIZE)
    parser.add_argument("--motif-min-n", type=int, default=DEFAULT_MOTIF_MIN_N)
    parser.add_argument("--motif-max-n", type=int, default=DEFAULT_MOTIF_MAX_N)
    parser.add_argument("--smoothing-alpha", type=float, default=DEFAULT_SMOOTHING_ALPHA)
    parser.add_argument("--rare-max-count", type=int, default=DEFAULT_RARE_MAX_COUNT)
    parser.add_argument("--lz-window-fallbacks", type=int, default=DEFAULT_LZ_WINDOW_FALLBACKS)
    parser.add_argument("--lz-max-span", type=int, default=DEFAULT_LZ_MAX_SPAN)
    parser.add_argument("--mapper-window-ms", type=int, default=MAPPER_WRITE_MS)
    parser.add_argument("--sidecar-max-tokens", type=int, default=DEFAULT_MAPPER_SIDECAR_MAX_TOKENS)
    parser.add_argument("--boundary-risk-margin-ms", type=int, default=DEFAULT_BOUNDARY_RISK_MARGIN_MS)
    parser.add_argument("--limit-chunks", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.mapper_sidecar:
        report = audit_c3_mapper_window_sidecar(
            chunk_cache_path=args.chunk_cache_path,
            dataset_root=args.dataset_root,
            sidecar_path=args.sidecar_path or DEFAULT_MAPPER_SIDECAR_PATH,
            report_path=args.report_path or DEFAULT_MAPPER_SIDECAR_REPORT_PATH,
            result_log_path=args.result_log_path or DEFAULT_MAPPER_SIDECAR_RESULT_LOG_PATH,
            motif_vocab_size=args.motif_vocab_size,
            motif_min_n=args.motif_min_n,
            motif_max_n=args.motif_max_n,
            smoothing_alpha=args.smoothing_alpha,
            rare_max_count=args.rare_max_count,
            lz_window_fallbacks=args.lz_window_fallbacks,
            lz_max_span=args.lz_max_span,
            mapper_window_ms=args.mapper_window_ms,
            sidecar_max_tokens=args.sidecar_max_tokens,
            boundary_risk_margin_ms=args.boundary_risk_margin_ms,
            limit_chunks=args.limit_chunks,
            command=_command(sys.argv[1:]),
        )
        print(
            "c3_mapper_window_sidecar "
            f"pass={report.get('pass_criteria', {}).get('p3_sidecar_generation_pass')} "
            f"full_cache={report.get('pass_criteria', {}).get('full_cache_pass')} "
            f"tokens={report.get('sidecar_stats', {}).get('sidecar_token_count')} "
            f"windows={report.get('sidecar_stats', {}).get('window_count')} "
            f"recommendation={report.get('recommendation')} "
            f"sidecar={report.get('sidecar_path')} "
            f"report={report.get('report_path')}"
        )
        return 0
    report = audit_c3_side_stream_tokenization(
        chunk_cache_path=args.chunk_cache_path,
        report_path=args.report_path or DEFAULT_REPORT_PATH,
        result_log_path=args.result_log_path or DEFAULT_RESULT_LOG_PATH,
        motif_vocab_size=args.motif_vocab_size,
        motif_min_n=args.motif_min_n,
        motif_max_n=args.motif_max_n,
        smoothing_alpha=args.smoothing_alpha,
        rare_max_count=args.rare_max_count,
        lz_window_fallbacks=args.lz_window_fallbacks,
        lz_max_span=args.lz_max_span,
        limit_chunks=args.limit_chunks,
        command=_command(sys.argv[1:]),
    )
    print(
        "c3_side_stream_tokenization "
        f"p0_roundtrip={report.get('pass_criteria', {}).get('p0_roundtrip_pass')} "
        f"full_cache={report.get('pass_criteria', {}).get('full_cache_pass')} "
        f"recommendation={report.get('recommendation')} "
        f"report={report.get('report_path')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
