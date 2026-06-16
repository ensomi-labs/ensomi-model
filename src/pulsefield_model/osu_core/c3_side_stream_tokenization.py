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


SCHEMA_VERSION: Final[int] = 1
DEFAULT_REPORT_PATH: Final[Path] = Path(
    "artifacts/reports/audits/context_adaptive_fallback_codec/c3_side_stream_pipeline_report.json",
)
DEFAULT_RESULT_LOG_PATH: Final[Path] = Path(
    "artifacts/reports/audits/context_adaptive_fallback_codec/c3_side_stream_pipeline_result_log.md",
)
SELECTED_C3_VARIANT: Final[str] = "a2_skeleton_residual_all_fallback_w256"
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
    parser.add_argument("--chunk-cache-path", type=Path, default=DEFAULT_BEAT_CHUNK_CACHE_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--result-log-path", type=Path, default=DEFAULT_RESULT_LOG_PATH)
    parser.add_argument("--motif-vocab-size", type=int, default=DEFAULT_MOTIF_VOCAB_SIZE)
    parser.add_argument("--motif-min-n", type=int, default=DEFAULT_MOTIF_MIN_N)
    parser.add_argument("--motif-max-n", type=int, default=DEFAULT_MOTIF_MAX_N)
    parser.add_argument("--smoothing-alpha", type=float, default=DEFAULT_SMOOTHING_ALPHA)
    parser.add_argument("--rare-max-count", type=int, default=DEFAULT_RARE_MAX_COUNT)
    parser.add_argument("--lz-window-fallbacks", type=int, default=DEFAULT_LZ_WINDOW_FALLBACKS)
    parser.add_argument("--lz-max-span", type=int, default=DEFAULT_LZ_MAX_SPAN)
    parser.add_argument("--limit-chunks", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = audit_c3_side_stream_tokenization(
        chunk_cache_path=args.chunk_cache_path,
        report_path=args.report_path,
        result_log_path=args.result_log_path,
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
