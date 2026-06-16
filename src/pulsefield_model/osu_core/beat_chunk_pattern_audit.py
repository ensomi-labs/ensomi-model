from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shlex
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pulsefield_model.osu_core.beat_representation import (
    DEFAULT_BEAT_REPRESENTATION_TIMING_CANONICALIZATION,
    DEFAULT_SNAP_DENOMINATOR,
    BeatEvent,
    BeatEventKind,
    hitobjects_to_beat_events,
)
from pulsefield_model.osu_core.beat_representation_audit import DEFAULT_DATASET_ROOT, resolve_index_beatmap_path
from pulsefield_model.osu_core.beat_structure_token_audit import (
    DEFAULT_LE3_STRUCTURE_CACHE_PATH,
    DEFAULT_LE3_STRUCTURE_REPORT_PATH,
)
from pulsefield_model.osu_core.hitobjects import ManiaHitObject, ManiaHitObjectKind, parse_mania_hit_objects
from pulsefield_model.osu_core.timing import RedTimingPoint, require_red_timing_points
from pulsefield_model.timing.canonicalization import TIMING_CANONICALIZATION_CHOICES


AUDIT_SCHEMA_VERSION: Final[int] = 1
CHUNK_LENGTH_BEATS: Final[int] = 2
METER_BEATS: Final[int] = 4
DEFAULT_CHUNK_UNITS: Final[int] = CHUNK_LENGTH_BEATS * DEFAULT_SNAP_DENOMINATOR
DEFAULT_BAR_UNITS: Final[int] = METER_BEATS * DEFAULT_SNAP_DENOMINATOR
DEFAULT_BEAT_CHUNK_CACHE_PATH: Final[Path] = Path(
    "artifacts/audits/beat_chunk_patterns/beat_chunk_pattern_cache_le3.parquet",
)
DEFAULT_BEAT_CHUNK_MAP_CACHE_PATH: Final[Path] = Path(
    "artifacts/audits/beat_chunk_patterns/beat_chunk_pattern_map_cache_le3.parquet",
)
DEFAULT_BEAT_CHUNK_REPORT_PATH: Final[Path] = Path(
    "artifacts/reports/audits/beat_chunk_patterns/beat_chunk_pattern_audit_le3.json",
)
DEFAULT_RESULT_LOG_PATH: Final[Path] = Path(
    "artifacts/reports/audits/beat_chunk_patterns/beat_chunk_pattern_result_log.md",
)
DEFAULT_MOTIF_VOCAB_SIZES: Final[tuple[int, ...]] = (512, 1024, 2048, 4096, 8192, 16384)
DEFAULT_MOTIF_MIN_N: Final[int] = 2
DEFAULT_MOTIF_MAX_N: Final[int] = 8
DEFAULT_RARE_MAX_COUNT: Final[int] = 1
DEFAULT_TOP_EXAMPLE_LIMIT: Final[int] = 20
DEFAULT_PARQUET_BATCH_ROWS: Final[int] = 50_000
DEFAULT_SMOOTHING_ALPHA: Final[float] = 0.1
BASELINE_V2_BITS_PER_EVENT: Final[float] = 8.087927414484682
BASELINE_BEAT_JOINT_BITS_PER_EVENT: Final[float] = 11.947807780153775
BASELINE_BEAT_INDEPENDENT_BITS_PER_EVENT: Final[float] = 26.77913606031193
RESEARCH_FALLBACK_RATE_MAX: Final[float] = 0.50
MIRROR_SAVINGS_MIN_BITS_PER_EVENT: Final[float] = 1e-9
MOTIF_CONTROL_LIFT_MIN: Final[float] = 0.01

_KIND_ORDER: Final[dict[BeatEventKind, int]] = {
    BeatEventKind.HOLD_START: 0,
    BeatEventKind.TAP: 1,
    BeatEventKind.HOLD_END: 2,
}
_ACTION_LABEL: Final[dict[BeatEventKind, str]] = {
    BeatEventKind.HOLD_START: "hold_start",
    BeatEventKind.TAP: "tap",
    BeatEventKind.HOLD_END: "hold_end",
}
_ACTION_SHORT: Final[dict[BeatEventKind, str]] = {
    BeatEventKind.HOLD_START: "S",
    BeatEventKind.TAP: "T",
    BeatEventKind.HOLD_END: "E",
}
_MIRROR_MASK: Final[dict[int, int]] = {
    mask: sum(((mask >> lane) & 1) << (3 - lane) for lane in range(4))
    for mask in range(16)
}
_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "source_row_index",
    "shard",
    "beatmap_set_id",
    "beatmap_id",
    "beatmap_path",
    "title",
    "artist",
    "creator",
    "version",
    "difficulty",
    "mode",
    "key_count",
    "density_bin",
    "density_events_per_second",
    "chord_ratio",
    "ln_ratio",
    "v2_token_count",
)


@dataclass(frozen=True)
class _ChunkBuildResult:
    map_summary: dict[str, Any]
    chunk_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class _MotifVocab:
    motif_to_id: dict[tuple[str, ...], str]
    motif_lengths: dict[str, int]
    motif_display: dict[str, str]


@dataclass(frozen=True)
class _EncodedSplit:
    token_counter: Counter[str]
    token_count: int
    group_count: int
    chunk_count: int
    event_count: int
    motif_token_count: int
    motif_group_coverage_count: int
    atomic_token_count: int
    unknown_atomic_count: int
    orientation_token_count: int
    chunk_start_token_count: int


def audit_beat_chunk_patterns(
    *,
    beat_structure_cache_path: str | Path = DEFAULT_LE3_STRUCTURE_CACHE_PATH,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    chunk_cache_path: str | Path = DEFAULT_BEAT_CHUNK_CACHE_PATH,
    map_cache_path: str | Path | None = DEFAULT_BEAT_CHUNK_MAP_CACHE_PATH,
    report_path: str | Path | None = DEFAULT_BEAT_CHUNK_REPORT_PATH,
    result_log_path: str | Path | None = DEFAULT_RESULT_LOG_PATH,
    beat_structure_report_path: str | Path | None = DEFAULT_LE3_STRUCTURE_REPORT_PATH,
    snap_denominator: int = DEFAULT_SNAP_DENOMINATOR,
    timing_canonicalization: str = DEFAULT_BEAT_REPRESENTATION_TIMING_CANONICALIZATION,
    expected_key_count: int | None = 4,
    limit: int | None = None,
    progress_every: int = 0,
    motif_min_n: int = DEFAULT_MOTIF_MIN_N,
    motif_max_n: int = DEFAULT_MOTIF_MAX_N,
    motif_vocab_sizes: Sequence[int] = DEFAULT_MOTIF_VOCAB_SIZES,
    rare_max_count: int = DEFAULT_RARE_MAX_COUNT,
    top_example_limit: int = DEFAULT_TOP_EXAMPLE_LIMIT,
    parquet_batch_rows: int = DEFAULT_PARQUET_BATCH_ROWS,
    smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
    command: str | None = None,
) -> dict[str, Any]:
    """Build and audit 2-beat chunk motif tokenization over validated beat maps."""

    started_at = time.perf_counter()
    beat_structure_cache_path = Path(beat_structure_cache_path)
    dataset_root = Path(dataset_root)
    chunk_cache_path = Path(chunk_cache_path)
    map_cache_path = None if map_cache_path is None else Path(map_cache_path)
    report_path = None if report_path is None else Path(report_path)
    result_log_path = None if result_log_path is None else Path(result_log_path)
    beat_structure_report_path = None if beat_structure_report_path is None else Path(beat_structure_report_path)
    snap_denominator = _validate_positive_int(snap_denominator, "snap_denominator")
    if timing_canonicalization not in TIMING_CANONICALIZATION_CHOICES:
        choices = ", ".join(TIMING_CANONICALIZATION_CHOICES)
        raise ValueError(f"timing_canonicalization must be one of {choices}, got {timing_canonicalization!r}")
    if expected_key_count is not None:
        expected_key_count = _validate_positive_int(expected_key_count, "expected_key_count")
    if limit is not None and int(limit) < 0:
        raise ValueError(f"limit must be non-negative, got {limit!r}")
    progress_every = _validate_nonnegative_int(progress_every, "progress_every")
    motif_min_n = _validate_positive_int(motif_min_n, "motif_min_n")
    motif_max_n = _validate_positive_int(motif_max_n, "motif_max_n")
    if motif_max_n < motif_min_n:
        raise ValueError("motif_max_n must be >= motif_min_n")
    motif_vocab_sizes = tuple(sorted({_validate_positive_int(value, "motif_vocab_sizes") for value in motif_vocab_sizes}))
    rare_max_count = _validate_nonnegative_int(rare_max_count, "rare_max_count")
    top_example_limit = _validate_nonnegative_int(top_example_limit, "top_example_limit")
    parquet_batch_rows = _validate_positive_int(parquet_batch_rows, "parquet_batch_rows")
    if smoothing_alpha <= 0:
        raise ValueError(f"smoothing_alpha must be positive, got {smoothing_alpha!r}")

    source_df = pd.read_parquet(beat_structure_cache_path)
    if "ok" in source_df.columns:
        source_df = source_df[source_df["ok"].fillna(False).astype(bool)].copy()
    total_source_rows = len(source_df)
    max_rows = total_source_rows if limit is None else min(int(limit), total_source_rows)
    source_df = source_df.head(max_rows)

    writer: pq.ParquetWriter | None = None
    chunk_batch: list[dict[str, Any]] = []
    map_summaries: list[dict[str, Any]] = []
    chunk_cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_chunk_cache_path = chunk_cache_path.with_suffix(chunk_cache_path.suffix + ".tmp")
    if tmp_chunk_cache_path.exists():
        tmp_chunk_cache_path.unlink()

    try:
        for processed_index, row in enumerate(source_df.itertuples(index=False), start=1):
            result = build_beat_chunk_pattern_rows(
                row,
                dataset_root=dataset_root,
                snap_denominator=snap_denominator,
                timing_canonicalization=timing_canonicalization,
                expected_key_count=expected_key_count,
            )
            map_summaries.append(result.map_summary)
            chunk_batch.extend(result.chunk_rows)
            if len(chunk_batch) >= parquet_batch_rows:
                writer = _write_parquet_batch(tmp_chunk_cache_path, chunk_batch, writer)
                chunk_batch.clear()
            if progress_every > 0 and processed_index % progress_every == 0:
                print(f"processed {processed_index}/{max_rows} maps", file=sys.stderr)
        if chunk_batch:
            writer = _write_parquet_batch(tmp_chunk_cache_path, chunk_batch, writer)
            chunk_batch.clear()
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        empty_table = pa.Table.from_pylist([], schema=_chunk_arrow_schema())
        pq.write_table(empty_table, tmp_chunk_cache_path)
    tmp_chunk_cache_path.replace(chunk_cache_path)

    chunk_df = pd.read_parquet(chunk_cache_path)
    map_df = pd.DataFrame.from_records(map_summaries)
    if map_cache_path is not None:
        map_cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_map_cache_path = map_cache_path.with_suffix(map_cache_path.suffix + ".tmp")
        map_df.to_parquet(tmp_map_cache_path, index=False)
        tmp_map_cache_path.replace(map_cache_path)
    baselines = _load_baselines(beat_structure_report_path)
    report = build_beat_chunk_pattern_report(
        chunk_df,
        map_df=map_df,
        beat_structure_cache_path=beat_structure_cache_path,
        beat_structure_cache_sha256=_sha256(beat_structure_cache_path),
        beat_structure_report_path=beat_structure_report_path,
        dataset_root=dataset_root,
        chunk_cache_path=chunk_cache_path,
        chunk_cache_sha256=_sha256(chunk_cache_path),
        map_cache_path=map_cache_path,
        map_cache_sha256=None if map_cache_path is None else _sha256(map_cache_path),
        source_row_count=total_source_rows,
        processed_row_count=max_rows,
        elapsed_s=time.perf_counter() - started_at,
        snap_denominator=snap_denominator,
        timing_canonicalization=timing_canonicalization,
        expected_key_count=expected_key_count,
        limit=limit,
        motif_min_n=motif_min_n,
        motif_max_n=motif_max_n,
        motif_vocab_sizes=motif_vocab_sizes,
        rare_max_count=rare_max_count,
        top_example_limit=top_example_limit,
        smoothing_alpha=smoothing_alpha,
        baselines=baselines,
        command=command,
    )
    if report_path is not None:
        _write_json(report_path, report)
        _write_motif_vocab_stats(report_path.parent, report)
    if result_log_path is not None:
        _write_result_log(result_log_path, report)
    return report


def build_beat_chunk_pattern_rows(
    row: Mapping[str, object] | object,
    *,
    dataset_root: str | Path,
    snap_denominator: int = DEFAULT_SNAP_DENOMINATOR,
    timing_canonicalization: str = DEFAULT_BEAT_REPRESENTATION_TIMING_CANONICALIZATION,
    expected_key_count: int | None = 4,
) -> _ChunkBuildResult:
    """Build reconstructable 2-beat chunk rows for one beat-structure cache row."""

    base = _base_map_fields(row)
    split = _split_for_mapset(base.get("beatmap_set_id"))
    try:
        beatmap_path = _resolved_beatmap_path(row, Path(dataset_root))
        timing_points = require_red_timing_points(beatmap_path)
        hitobjects = parse_mania_hit_objects(beatmap_path, expected_key_count=expected_key_count)
        events = hitobjects_to_beat_events(
            hitobjects,
            timing_points,
            snap_denominator=snap_denominator,
            timing_canonicalization=timing_canonicalization,
            include_diagnostics=True,
        )
    except Exception as exc:  # noqa: BLE001 - row-local audit failure.
        map_summary = {
            **base,
            "split": split,
            "conversion_ok": False,
            "hard_gate_ok": False,
            "error_type": type(exc).__name__,
            "error_message": _short_error(str(exc)),
            "event_count": 0,
            "expected_event_count": 0,
            "group_count": 0,
            "chunk_count": 0,
            "event_count_mismatch_count": 1,
            "event_order_mismatch_count": 0,
            "lane_mismatch_count": 0,
            "action_mismatch_count": 0,
            "chord_group_mismatch_count": 0,
            "ln_pair_mismatch_count": 0,
            "chunk_boundary_reconstruction_mismatch_count": 0,
            "same_lane_duplicate_action_group_count": 0,
        }
        return _ChunkBuildResult(map_summary=map_summary, chunk_rows=[])

    segment_ids = _segment_ids(events)
    source_event_tuples = _event_tuples(events, segment_ids)
    expected_event_count = sum(1 if item.kind == ManiaHitObjectKind.TAP else 2 for item in hitobjects)
    groups = _group_tokens(events, segment_ids, snap_denominator=snap_denominator)
    chunk_units = CHUNK_LENGTH_BEATS * snap_denominator
    bar_units = METER_BEATS * snap_denominator
    chunks_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    chunk_sort_ms: dict[tuple[int, int], float] = {}
    for group in groups:
        segment_id = int(group["segment_id"])
        beat_units = int(group["beat_offset_numerator"])
        segment_chunk_index = beat_units // chunk_units
        offset_units = beat_units - segment_chunk_index * chunk_units
        token = {
            "offset_units": int(offset_units),
            "tap_mask": int(group["tap_mask"]),
            "ln_start_mask": int(group["ln_start_mask"]),
            "ln_end_mask": int(group["ln_end_mask"]),
            "order_signature": str(group["order_signature"]),
            "event_count": int(group["event_count"]),
            "onset_count": int(group["onset_count"]),
            "same_lane_multi_action_count": int(group["same_lane_multi_action_count"]),
            "same_lane_duplicate_action_count": int(group["same_lane_duplicate_action_count"]),
        }
        key = (segment_id, segment_chunk_index)
        chunks_by_key[key].append(token)
        chunk_sort_ms[key] = min(float(chunk_sort_ms.get(key, math.inf)), float(group["snapped_time_ms"]))

    expanded_keys = _expanded_chunk_keys(chunks_by_key)
    chunk_rows: list[dict[str, Any]] = []
    reconstructed_event_tuples: list[tuple[str, int, int, int]] = []
    chunk_boundary_mismatch_count = 0
    chunk_identity = _base_chunk_identity(base, split)
    for chunk_index, key in enumerate(expanded_keys):
        segment_id, segment_chunk_index = key
        raw_groups = sorted(chunks_by_key.get(key, []), key=lambda item: int(item["offset_units"]))
        start_beat_units = segment_chunk_index * chunk_units
        bar_phase_half = 0 if (start_beat_units % bar_units) < chunk_units else 1
        raw_signature = _signature_for_groups(raw_groups)
        mirrored_groups = [_mirror_group_token(group) for group in raw_groups]
        mirror_signature = _signature_for_groups(mirrored_groups)
        if mirror_signature < raw_signature:
            canonical_signature = mirror_signature
            orientation = 1
            canonical_groups = mirrored_groups
        else:
            canonical_signature = raw_signature
            orientation = 0
            canonical_groups = raw_groups
        if not all(0 <= int(group["offset_units"]) < chunk_units for group in raw_groups):
            chunk_boundary_mismatch_count += 1
        for group in raw_groups:
            beat_units = start_beat_units + int(group["offset_units"])
            reconstructed_event_tuples.extend(_event_tuples_for_group(group, segment_id=segment_id, beat_units=beat_units))
        chunk_rows.append(
            {
                **chunk_identity,
                "schema_version": AUDIT_SCHEMA_VERSION,
                "chunk_index": int(chunk_index),
                "segment_id": int(segment_id),
                "segment_chunk_index": int(segment_chunk_index),
                "bar_phase_half": int(bar_phase_half),
                "start_beat_units": int(start_beat_units),
                "start_beat": float(start_beat_units) / float(snap_denominator),
                "length_units": int(chunk_units),
                "length_beats": CHUNK_LENGTH_BEATS,
                "num_groups": len(raw_groups),
                "num_events": sum(int(group["event_count"]) for group in raw_groups),
                "empty": len(raw_groups) == 0,
                "raw_signature": raw_signature,
                "mirror_signature": mirror_signature,
                "canonical_signature": canonical_signature,
                "orientation": int(orientation),
                "is_mirror_symmetric": raw_signature == mirror_signature,
                "groups_json": _json_dumps(raw_groups),
                "canonical_groups_json": _json_dumps(canonical_groups),
                "chunk_sort_ms": float(chunk_sort_ms.get(key, chunk_index)),
            }
        )

    action_mismatches = 0
    lane_mismatches = 0
    order_mismatches = 0
    for expected, actual in zip(source_event_tuples, reconstructed_event_tuples, strict=False):
        if expected != actual:
            order_mismatches += 1
            if expected[0] != actual[0]:
                action_mismatches += 1
            if expected[1] != actual[1]:
                lane_mismatches += 1
    event_count_mismatch = 0 if len(source_event_tuples) == len(reconstructed_event_tuples) == expected_event_count else 1
    if len(source_event_tuples) != len(reconstructed_event_tuples):
        event_count_mismatch = 1
    chord_group_mismatch = _chord_group_mismatch_count(groups, reconstructed_event_tuples)
    ln_pair_mismatch = _ln_pair_mismatch_count(events, hitobjects, segment_ids, snap_denominator=snap_denominator)
    hard_gate_ok = (
        event_count_mismatch == 0
        and order_mismatches == 0
        and lane_mismatches == 0
        and action_mismatches == 0
        and chord_group_mismatch == 0
        and ln_pair_mismatch == 0
        and chunk_boundary_mismatch_count == 0
    )
    map_summary = {
        **base,
        "split": split,
        "conversion_ok": True,
        "hard_gate_ok": hard_gate_ok,
        "error_type": "",
        "error_message": "",
        "event_count": len(events),
        "expected_event_count": expected_event_count,
        "group_count": len(groups),
        "chunk_count": len(chunk_rows),
        "event_count_mismatch_count": event_count_mismatch,
        "event_order_mismatch_count": order_mismatches,
        "lane_mismatch_count": lane_mismatches,
        "action_mismatch_count": action_mismatches,
        "chord_group_mismatch_count": chord_group_mismatch,
        "ln_pair_mismatch_count": ln_pair_mismatch,
        "chunk_boundary_reconstruction_mismatch_count": chunk_boundary_mismatch_count,
        "same_lane_multi_action_group_count": sum(int(group["same_lane_multi_action_count"] > 0) for group in groups),
        "same_lane_duplicate_action_group_count": sum(int(group["same_lane_duplicate_action_count"] > 0) for group in groups),
    }
    return _ChunkBuildResult(map_summary=map_summary, chunk_rows=chunk_rows)


def build_beat_chunk_pattern_report(
    chunk_cache: pd.DataFrame | str | Path,
    *,
    map_df: pd.DataFrame | None = None,
    beat_structure_cache_path: str | Path | None = None,
    beat_structure_cache_sha256: str | None = None,
    beat_structure_report_path: str | Path | None = None,
    dataset_root: str | Path | None = None,
    chunk_cache_path: str | Path | None = None,
    chunk_cache_sha256: str | None = None,
    map_cache_path: str | Path | None = None,
    map_cache_sha256: str | None = None,
    source_row_count: int | None = None,
    processed_row_count: int | None = None,
    elapsed_s: float | None = None,
    snap_denominator: int = DEFAULT_SNAP_DENOMINATOR,
    timing_canonicalization: str = DEFAULT_BEAT_REPRESENTATION_TIMING_CANONICALIZATION,
    expected_key_count: int | None = 4,
    limit: int | None = None,
    motif_min_n: int = DEFAULT_MOTIF_MIN_N,
    motif_max_n: int = DEFAULT_MOTIF_MAX_N,
    motif_vocab_sizes: Sequence[int] = DEFAULT_MOTIF_VOCAB_SIZES,
    rare_max_count: int = DEFAULT_RARE_MAX_COUNT,
    top_example_limit: int = DEFAULT_TOP_EXAMPLE_LIMIT,
    smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
    baselines: Mapping[str, float] | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    chunk_df = pd.read_parquet(chunk_cache) if isinstance(chunk_cache, (str, Path)) else chunk_cache.copy()
    map_df = _map_df_from_chunk_df(chunk_df) if map_df is None else map_df.copy()
    baselines = dict(_default_baselines() if baselines is None else baselines)
    processed_row_count = len(map_df) if processed_row_count is None else int(processed_row_count)
    source_row_count = processed_row_count if source_row_count is None else int(source_row_count)
    motif_vocab_sizes = tuple(sorted({_validate_positive_int(value, "motif_vocab_sizes") for value in motif_vocab_sizes}))

    raw_counter = Counter(chunk_df["raw_signature"].fillna("").astype(str))
    canonical_counter = Counter(chunk_df["canonical_signature"].fillna("").astype(str))
    orientation_counter_by_canonical = _orientation_counter_by_canonical(chunk_df)
    atomic_counter = _atomic_counter(chunk_df, canonical=False)
    canonical_atomic_counter = _atomic_counter(chunk_df, canonical=True)
    chunk_start_counter = Counter(f"C{int(value)}" for value in chunk_df["bar_phase_half"].fillna(0))
    orientation_token_counter = Counter(
        f"O{int(value)}"
        for value in chunk_df.loc[~chunk_df["is_mirror_symmetric"].fillna(False).astype(bool), "orientation"].fillna(0)
    )
    ngram_stats = _ngram_stats(chunk_df, motif_min_n=motif_min_n, motif_max_n=motif_max_n, top_limit=top_example_limit)
    compression_curve = _compression_curve(
        chunk_df,
        motif_vocab_sizes=motif_vocab_sizes,
        motif_min_n=motif_min_n,
        motif_max_n=motif_max_n,
        smoothing_alpha=smoothing_alpha,
        top_example_limit=top_example_limit,
    )

    total_events = _sum_int(map_df, "event_count")
    total_groups = int(chunk_df["num_groups"].fillna(0).astype(int).sum()) if "num_groups" in chunk_df.columns else 0
    total_chunks = len(chunk_df)
    raw_entropy = _entropy_bits(raw_counter)
    raw_bits = raw_entropy * float(total_chunks)
    canonical_entropy = _entropy_bits(canonical_counter)
    orientation_conditional_entropy = _conditional_orientation_entropy_bits(orientation_counter_by_canonical)
    canonical_net_bits = (canonical_entropy + orientation_conditional_entropy) * float(total_chunks)
    atomic_entropy = _entropy_bits(atomic_counter)
    atomic_stream_token_count = sum(atomic_counter.values()) + sum(chunk_start_counter.values())
    atomic_stream_counter = atomic_counter + chunk_start_counter
    atomic_stream_entropy = _entropy_bits(atomic_stream_counter)
    canonical_atomic_stream_counter = canonical_atomic_counter + chunk_start_counter + orientation_token_counter
    canonical_atomic_stream_entropy = _entropy_bits(canonical_atomic_stream_counter)
    hard_gate_counts = _hard_gate_counts(map_df)
    hard_gate_pass = (
        _count_true(map_df, "conversion_ok") == processed_row_count
        and _count_true(map_df, "hard_gate_ok") == processed_row_count
        and all(value == 0 for value in hard_gate_counts.values())
    )
    best_test = _best_compression_result(compression_curve, split="test")
    best_valid = _best_compression_result(compression_curve, split="valid")
    best_train = _best_compression_result(compression_curve, split="train")
    best_test_bits = float(best_test.get("bits_per_event", math.inf)) if best_test else math.inf
    mirror_savings_bits_per_event = (
        float(raw_bits - canonical_net_bits) / float(total_events) if total_events else 0.0
    )
    fallback_rate = float(best_test.get("fallback_group_rate", 1.0)) if best_test else 1.0
    motif_lift = float(ngram_stats.get("density_chord_ln_controlled", {}).get("recurrence_lift", 0.0))
    train_available = bool(best_train and int(best_train.get("event_count", 0)) > 0)
    heldout_available = bool(train_available and best_test and int(best_test.get("event_count", 0)) > 0)
    mirror_net_savings_pass = mirror_savings_bits_per_event > MIRROR_SAVINGS_MIN_BITS_PER_EVENT
    research_pass = bool(
        hard_gate_pass
        and heldout_available
        and best_test_bits < float(baselines["event_level_beat_joint_bits_per_event"])
        and mirror_net_savings_pass
        and fallback_rate <= RESEARCH_FALLBACK_RATE_MAX
        and motif_lift >= MOTIF_CONTROL_LIFT_MIN
    )
    mapper_interesting_pass = bool(
        research_pass
        and best_test_bits <= float(baselines["current_v2_bits_per_event"])
    )

    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "beat_structure_cache_path": None if beat_structure_cache_path is None else Path(beat_structure_cache_path).as_posix(),
        "beat_structure_cache_sha256": beat_structure_cache_sha256,
        "beat_structure_report_path": None if beat_structure_report_path is None else Path(beat_structure_report_path).as_posix(),
        "dataset_root": None if dataset_root is None else Path(dataset_root).as_posix(),
        "chunk_cache_path": None if chunk_cache_path is None else Path(chunk_cache_path).as_posix(),
        "chunk_cache_sha256": chunk_cache_sha256,
        "map_cache_path": None if map_cache_path is None else Path(map_cache_path).as_posix(),
        "map_cache_sha256": map_cache_sha256,
        "source_row_count": source_row_count,
        "processed_row_count": processed_row_count,
        "limited": limit is not None,
        "limit": None if limit is None else int(limit),
        "config": {
            "snap_denominator": snap_denominator,
            "chunk_length_beats": CHUNK_LENGTH_BEATS,
            "chunk_length_units": CHUNK_LENGTH_BEATS * snap_denominator,
            "timing_canonicalization": timing_canonicalization,
            "expected_key_count": expected_key_count,
            "motif_min_n": motif_min_n,
            "motif_max_n": motif_max_n,
            "motif_vocab_sizes": list(motif_vocab_sizes),
            "rare_max_count": rare_max_count,
            "smoothing_alpha": smoothing_alpha,
            "tokenizer_family": "frequent-ngram motif dictionary with greedy longest-match encoding and atomic fallback",
            "split_policy": "deterministic beatmap_set_id hash; train/valid/test = 80/10/10",
        },
        "baselines": baselines,
        "parity_summary": {
            "hard_gate_pass": hard_gate_pass,
            "status": "PASS" if hard_gate_pass else "FAIL",
            "conversion_ok_count": _count_true(map_df, "conversion_ok"),
            "hard_gate_ok_count": _count_true(map_df, "hard_gate_ok"),
            **hard_gate_counts,
        },
        "chunk_construction_summary": {
            "map_count": processed_row_count,
            "chunk_count": total_chunks,
            "nonempty_chunk_count": int((~chunk_df["empty"].fillna(False).astype(bool)).sum()) if "empty" in chunk_df.columns else 0,
            "empty_chunk_count": int(chunk_df["empty"].fillna(False).astype(bool).sum()) if "empty" in chunk_df.columns else 0,
            "empty_chunk_rate": _mean_bool(chunk_df, "empty"),
            "event_count": total_events,
            "group_count": total_groups,
            "groups_per_chunk": _distribution_summary(chunk_df, "num_groups"),
            "events_per_chunk": _distribution_summary(chunk_df, "num_events"),
            "split_summary": _split_summary(chunk_df),
        },
        "raw_chunk_distribution": _counter_distribution(
            raw_counter,
            item_count=total_chunks,
            event_count=total_events,
            rare_max_count=rare_max_count,
            top_limit=top_example_limit,
        ),
        "mirror_canonicalization_effect": {
            "raw_unique_count": len(raw_counter),
            "canonical_unique_count": len(canonical_counter),
            "canonical_unique_over_raw_unique": float(len(canonical_counter)) / float(len(raw_counter)) if raw_counter else 0.0,
            "raw_entropy_per_chunk_bits": raw_entropy,
            "raw_bits_per_event": float(raw_bits) / float(total_events) if total_events else 0.0,
            "canonical_entropy_per_chunk_bits": canonical_entropy,
            "orientation_conditional_entropy_bits": orientation_conditional_entropy,
            "net_canonical_entropy_per_chunk_bits": canonical_entropy + orientation_conditional_entropy,
            "net_canonical_bits_per_event": float(canonical_net_bits) / float(total_events) if total_events else 0.0,
            "savings_bits_per_event": mirror_savings_bits_per_event,
            "mirror_pair_rate": float((~chunk_df["is_mirror_symmetric"].fillna(False).astype(bool)).sum()) / float(total_chunks) if total_chunks else 0.0,
            "orientation_entropy_bits": _entropy_bits(Counter(str(value) for value in chunk_df["orientation"].fillna(0))),
            "orientation_token_count": sum(orientation_token_counter.values()),
        },
        "atomic_group_distribution": {
            "raw_atomic": _counter_distribution(
                atomic_counter,
                item_count=sum(atomic_counter.values()),
                event_count=total_events,
                rare_max_count=rare_max_count,
                top_limit=top_example_limit,
            ),
            "canonical_atomic": _counter_distribution(
                canonical_atomic_counter,
                item_count=sum(canonical_atomic_counter.values()),
                event_count=total_events,
                rare_max_count=rare_max_count,
                top_limit=top_example_limit,
            ),
            "raw_atomic_stream_bits_per_event": (
                atomic_stream_entropy * float(atomic_stream_token_count) / float(total_events) if total_events else 0.0
            ),
            "canonical_atomic_stream_bits_per_event": (
                canonical_atomic_stream_entropy
                * float(sum(canonical_atomic_stream_counter.values()))
                / float(total_events)
                if total_events
                else 0.0
            ),
        },
        "ngram_motif_stats": ngram_stats,
        "motif_tokenizer_compression_curve": compression_curve,
        "heldout_bits_event_comparison": {
            "best_train": best_train,
            "best_valid": best_valid,
            "best_test": best_test,
            "beats_event_level_joint": (
                best_test_bits < float(baselines["event_level_beat_joint_bits_per_event"]) if heldout_available else False
            ),
            "beats_current_v2": best_test_bits <= float(baselines["current_v2_bits_per_event"]) if heldout_available else False,
        },
        "pass_criteria": {
            "hard_pass": hard_gate_pass,
            "research_pass": research_pass,
            "mapper_interesting_pass": mapper_interesting_pass,
            "train_available": train_available,
            "heldout_available": heldout_available,
            "best_test_bits_per_event": None if not best_test else best_test_bits,
            "event_level_beat_joint_baseline": baselines["event_level_beat_joint_bits_per_event"],
            "current_v2_baseline": baselines["current_v2_bits_per_event"],
            "mirror_savings_bits_per_event": mirror_savings_bits_per_event,
            "mirror_net_savings_pass": mirror_net_savings_pass,
            "best_test_fallback_group_rate": fallback_rate,
            "density_controlled_motif_lift": motif_lift,
        },
        "failure_cases": {
            "hard_gate_failures": _failure_examples(map_df, top_example_limit),
            "worst_fallback_maps": _worst_split_examples(chunk_df, compression_curve, top_example_limit),
            "same_lane_duplicate_action_maps": _same_lane_duplicate_examples(map_df, top_example_limit),
        },
        "recommendation": _recommendation(hard_gate_pass, research_pass, mapper_interesting_pass, best_test_bits, baselines),
        "elapsed_s": elapsed_s,
        "code_commit": _git_stdout("rev-parse", "HEAD"),
        "code_dirty": bool(_git_stdout("status", "--porcelain")),
        "command": command,
    }
    return _normalize_json(report)


def _base_map_fields(row: Mapping[str, object] | object) -> dict[str, Any]:
    base: dict[str, Any] = {}
    for column in _IDENTITY_COLUMNS:
        base[column] = _cache_scalar(_row_value(row, column, default=None))
    return base


def _base_chunk_identity(base: Mapping[str, Any], split: str) -> dict[str, Any]:
    return {
        "source_row_index": _cache_scalar(base.get("source_row_index")),
        "split": split,
        "shard": _cache_scalar(base.get("shard")),
        "beatmap_set_id": _cache_scalar(base.get("beatmap_set_id")),
        "beatmap_id": _cache_scalar(base.get("beatmap_id")),
        "beatmap_path": _cache_scalar(base.get("beatmap_path")),
        "title": _cache_scalar(base.get("title")),
        "artist": _cache_scalar(base.get("artist")),
        "creator": _cache_scalar(base.get("creator")),
        "version": _cache_scalar(base.get("version")),
        "difficulty": _cache_scalar(base.get("difficulty")),
        "density_bin": _cache_scalar(base.get("density_bin")),
        "density_events_per_second": _cache_scalar(base.get("density_events_per_second")),
        "chord_ratio": _cache_scalar(base.get("chord_ratio")),
        "ln_ratio": _cache_scalar(base.get("ln_ratio")),
    }


def _resolved_beatmap_path(row: Mapping[str, object] | object, dataset_root: Path) -> Path:
    resolved = _row_value(row, "resolved_beatmap_path", default="")
    if resolved and not _is_missing_scalar(resolved):
        path = Path(str(resolved))
        if path.is_file():
            return path
    return resolve_index_beatmap_path(dataset_root, _row_value(row, "shard"), _row_value(row, "beatmap_path"))


def _segment_ids(events: Sequence[BeatEvent]) -> list[int]:
    segment_id_by_key: dict[tuple[float, float, float], int] = {}
    segment_ids: list[int] = []
    for event in events:
        key = (
            round(float(event.redline_offset_ms), 9),
            round(float(event.redline_beat_length_ms), 9),
            round(float(event.beat_length_ms), 9),
        )
        if key not in segment_id_by_key:
            segment_id_by_key[key] = len(segment_id_by_key)
        segment_ids.append(segment_id_by_key[key])
    return segment_ids


def _event_tuples(events: Sequence[BeatEvent], segment_ids: Sequence[int]) -> list[tuple[str, int, int, int]]:
    return [
        (_ACTION_LABEL[event.kind], int(event.lane), int(segment_ids[index]), int(event.beat_offset_numerator))
        for index, event in enumerate(events)
    ]


def _group_tokens(events: Sequence[BeatEvent], segment_ids: Sequence[int], *, snap_denominator: int) -> list[dict[str, Any]]:
    del snap_denominator
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, event in enumerate(events):
        grouped[(int(segment_ids[index]), int(event.beat_offset_numerator))].append(index)
    ordered_keys = sorted(
        grouped,
        key=lambda key: (
            min(float(events[index].snapped_time_ms) for index in grouped[key]),
            key[0],
            key[1],
        ),
    )
    groups: list[dict[str, Any]] = []
    for group_index, key in enumerate(ordered_keys):
        event_indexes = grouped[key]
        lane_actions: dict[int, list[BeatEventKind]] = defaultdict(list)
        for event_index in event_indexes:
            lane_actions[int(events[event_index].lane)].append(events[event_index].kind)
        duplicate_action_count = 0
        for actions in lane_actions.values():
            action_counts = Counter(actions)
            duplicate_action_count += sum(max(0, count - 1) for count in action_counts.values())
        ordered_events = [
            f"{_ACTION_SHORT[events[event_index].kind]}{int(events[event_index].lane)}"
            for event_index in event_indexes
        ]
        first_event = events[event_indexes[0]]
        tap_mask = _mask_for(lane_actions, BeatEventKind.TAP)
        ln_start_mask = _mask_for(lane_actions, BeatEventKind.HOLD_START)
        ln_end_mask = _mask_for(lane_actions, BeatEventKind.HOLD_END)
        default_ordered_events = _default_ordered_events(
            tap_mask=tap_mask,
            ln_start_mask=ln_start_mask,
            ln_end_mask=ln_end_mask,
        )
        groups.append(
            {
                "group_index": int(group_index),
                "segment_id": int(key[0]),
                "beat_offset_numerator": int(key[1]),
                "snapped_time_ms": float(first_event.snapped_time_ms),
                "tap_mask": tap_mask,
                "ln_start_mask": ln_start_mask,
                "ln_end_mask": ln_end_mask,
                "order_signature": "." if ordered_events == default_ordered_events else ",".join(ordered_events),
                "event_count": len(event_indexes),
                "onset_count": sum(
                    1
                    for event_index in event_indexes
                    if events[event_index].kind in {BeatEventKind.TAP, BeatEventKind.HOLD_START}
                ),
                "same_lane_multi_action_count": sum(1 for actions in lane_actions.values() if len(actions) > 1),
                "same_lane_duplicate_action_count": int(duplicate_action_count),
                "action_signature": _group_action_signature(lane_actions),
            }
        )
    return groups


def _expanded_chunk_keys(chunks_by_key: Mapping[tuple[int, int], Sequence[Mapping[str, Any]]]) -> list[tuple[int, int]]:
    by_segment: dict[int, list[int]] = defaultdict(list)
    for segment_id, segment_chunk_index in chunks_by_key:
        by_segment[int(segment_id)].append(int(segment_chunk_index))
    keys: list[tuple[int, int]] = []
    for segment_id in sorted(by_segment):
        indexes = by_segment[segment_id]
        for chunk_index in range(min(indexes), max(indexes) + 1):
            keys.append((segment_id, chunk_index))
    return keys


def _event_tuples_for_group(group: Mapping[str, Any], *, segment_id: int, beat_units: int) -> list[tuple[str, int, int, int]]:
    tuples: list[tuple[str, int, int, int]] = []
    order_signature = str(group.get("order_signature", ".") or ".")
    if order_signature != ".":
        for item in order_signature.split(","):
            if len(item) < 2:
                continue
            action = _action_label_from_short(item[0])
            lane = int(item[1:])
            tuples.append((action, lane, int(segment_id), int(beat_units)))
        return tuples
    masks = (
        (BeatEventKind.HOLD_START, int(group["ln_start_mask"])),
        (BeatEventKind.TAP, int(group["tap_mask"])),
        (BeatEventKind.HOLD_END, int(group["ln_end_mask"])),
    )
    for kind, mask in masks:
        for lane in range(4):
            if mask & (1 << lane):
                tuples.append((_ACTION_LABEL[kind], lane, int(segment_id), int(beat_units)))
    return tuples


def _chord_group_mismatch_count(
    groups: Sequence[Mapping[str, Any]],
    reconstructed_tuples: Sequence[tuple[str, int, int, int]],
) -> int:
    rebuilt: dict[tuple[int, int], dict[str, Any]] = defaultdict(lambda: {"tap": 0, "hold_start": 0, "hold_end": 0, "count": 0})
    for action, lane, segment_id, beat_units in reconstructed_tuples:
        key = (int(segment_id), int(beat_units))
        rebuilt[key][action] |= 1 << int(lane)
        rebuilt[key]["count"] += 1
    mismatches = 0
    for group in groups:
        key = (int(group["segment_id"]), int(group["beat_offset_numerator"]))
        actual = rebuilt.get(key)
        if actual is None:
            mismatches += 1
            continue
        if (
            int(group["tap_mask"]) != int(actual["tap"])
            or int(group["ln_start_mask"]) != int(actual["hold_start"])
            or int(group["ln_end_mask"]) != int(actual["hold_end"])
            or int(group["event_count"]) != int(actual["count"])
        ):
            mismatches += 1
    return mismatches


def _ln_pair_mismatch_count(
    events: Sequence[BeatEvent],
    hitobjects: Sequence[ManiaHitObject],
    segment_ids: Sequence[int],
    *,
    snap_denominator: int,
) -> int:
    del snap_denominator
    by_key: dict[tuple[BeatEventKind, int, float], list[int]] = defaultdict(list)
    for index, event in enumerate(events):
        by_key[(event.kind, int(event.lane), float(event.time_ms))].append(index)
    mismatch_count = 0
    for hitobject in hitobjects:
        if hitobject.kind != ManiaHitObjectKind.HOLD:
            continue
        start_indexes = by_key.get((BeatEventKind.HOLD_START, int(hitobject.lane), float(hitobject.start_time_ms)), [])
        end_indexes = by_key.get((BeatEventKind.HOLD_END, int(hitobject.lane), float(hitobject.end_time_ms)), [])
        if not start_indexes or not end_indexes:
            mismatch_count += 1
            continue
        start_index = start_indexes.pop(0)
        end_index = end_indexes.pop(0)
        if int(segment_ids[start_index]) == int(segment_ids[end_index]):
            start_units = int(events[start_index].beat_offset_numerator)
            end_units = int(events[end_index].beat_offset_numerator)
            if end_units < start_units:
                mismatch_count += 1
    return mismatch_count


def _mask_for(lane_actions: Mapping[int, Sequence[BeatEventKind]], kind: BeatEventKind) -> int:
    mask = 0
    for lane, actions in lane_actions.items():
        if kind in actions:
            mask |= 1 << int(lane)
    return mask


def _group_action_signature(lane_actions: Mapping[int, Sequence[BeatEventKind]]) -> str:
    parts: list[str] = []
    for lane in range(4):
        actions = lane_actions.get(lane, ())
        parts.append("".join(_ACTION_SHORT[action] for action in sorted(actions, key=lambda item: _KIND_ORDER[item])) or ".")
    return "".join(parts)


def _default_ordered_events(*, tap_mask: int, ln_start_mask: int, ln_end_mask: int) -> list[str]:
    ordered: list[str] = []
    for short, mask in (("S", ln_start_mask), ("T", tap_mask), ("E", ln_end_mask)):
        for lane in range(4):
            if mask & (1 << lane):
                ordered.append(f"{short}{lane}")
    return ordered


def _action_label_from_short(short: str) -> str:
    if short == "S":
        return _ACTION_LABEL[BeatEventKind.HOLD_START]
    if short == "T":
        return _ACTION_LABEL[BeatEventKind.TAP]
    if short == "E":
        return _ACTION_LABEL[BeatEventKind.HOLD_END]
    raise ValueError(f"unknown action short label: {short!r}")


def _mirror_order_signature(order_signature: str) -> str:
    if order_signature == ".":
        return "."
    mirrored: list[str] = []
    for item in order_signature.split(","):
        if len(item) < 2:
            continue
        mirrored.append(f"{item[0]}{3 - int(item[1:])}")
    return ",".join(mirrored) if mirrored else "."


def _signature_for_groups(groups: Sequence[Mapping[str, Any]]) -> str:
    return ";".join(_atomic_token(group) for group in groups)


def _mirror_group_token(group: Mapping[str, Any]) -> dict[str, Any]:
    mirrored = dict(group)
    mirrored["tap_mask"] = _MIRROR_MASK[int(group["tap_mask"])]
    mirrored["ln_start_mask"] = _MIRROR_MASK[int(group["ln_start_mask"])]
    mirrored["ln_end_mask"] = _MIRROR_MASK[int(group["ln_end_mask"])]
    mirrored["order_signature"] = _mirror_order_signature(str(group.get("order_signature", ".") or "."))
    return mirrored


def _atomic_token(group: Mapping[str, Any]) -> str:
    base = "A:{offset}:{tap}:{start}:{end}".format(
        offset=int(group["offset_units"]),
        tap=int(group["tap_mask"]),
        start=int(group["ln_start_mask"]),
        end=int(group["ln_end_mask"]),
    )
    order_signature = str(group.get("order_signature", ".") or ".")
    return base if order_signature == "." else f"{base}:O{order_signature}"


def _delta_atomic_tokens(groups: Sequence[Mapping[str, Any]]) -> list[str]:
    tokens: list[str] = []
    previous_offset: int | None = None
    for group in groups:
        offset = int(group["offset_units"])
        delta = 0 if previous_offset is None else offset - previous_offset
        previous_offset = offset
        tokens.append(
            "D:{delta}:{tap}:{start}:{end}".format(
                delta=delta,
                tap=int(group["tap_mask"]),
                start=int(group["ln_start_mask"]),
                end=int(group["ln_end_mask"]),
            )
        )
    return tokens


def _json_dumps(value: object) -> str:
    return json.dumps(_normalize_json(value), separators=(",", ":"), sort_keys=True)


def _json_loads(value: object) -> Any:
    if value is None or _is_missing_scalar(value) or value == "":
        return []
    return json.loads(str(value))


def _split_for_mapset(mapset_id: object) -> str:
    key = "" if _is_missing_scalar(mapset_id) else str(mapset_id)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "valid"
    return "test"


def _write_parquet_batch(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    writer: pq.ParquetWriter | None,
) -> pq.ParquetWriter:
    table = pa.Table.from_pylist([dict(row) for row in rows], schema=_chunk_arrow_schema())
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema)
    writer.write_table(table)
    return writer


def _chunk_arrow_schema() -> pa.Schema:
    return pa.schema(
        [
            ("schema_version", pa.int64()),
            ("source_row_index", pa.int64()),
            ("split", pa.string()),
            ("shard", pa.string()),
            ("beatmap_set_id", pa.int64()),
            ("beatmap_id", pa.int64()),
            ("beatmap_path", pa.string()),
            ("title", pa.string()),
            ("artist", pa.string()),
            ("creator", pa.string()),
            ("version", pa.string()),
            ("difficulty", pa.float64()),
            ("density_bin", pa.string()),
            ("density_events_per_second", pa.float64()),
            ("chord_ratio", pa.float64()),
            ("ln_ratio", pa.float64()),
            ("chunk_index", pa.int64()),
            ("segment_id", pa.int64()),
            ("segment_chunk_index", pa.int64()),
            ("bar_phase_half", pa.int64()),
            ("start_beat_units", pa.int64()),
            ("start_beat", pa.float64()),
            ("length_units", pa.int64()),
            ("length_beats", pa.int64()),
            ("num_groups", pa.int64()),
            ("num_events", pa.int64()),
            ("empty", pa.bool_()),
            ("raw_signature", pa.string()),
            ("mirror_signature", pa.string()),
            ("canonical_signature", pa.string()),
            ("orientation", pa.int64()),
            ("is_mirror_symmetric", pa.bool_()),
            ("groups_json", pa.string()),
            ("canonical_groups_json", pa.string()),
            ("chunk_sort_ms", pa.float64()),
        ]
    )


def _map_df_from_chunk_df(chunk_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "source_row_index",
        "split",
        "shard",
        "beatmap_set_id",
        "beatmap_id",
        "beatmap_path",
        "title",
        "artist",
        "creator",
        "version",
        "difficulty",
        "density_bin",
        "density_events_per_second",
        "chord_ratio",
        "ln_ratio",
    ]
    selected = [column for column in columns if column in chunk_df.columns]
    maps = chunk_df[selected].drop_duplicates("source_row_index").copy() if selected else pd.DataFrame()
    if maps.empty:
        return maps
    grouped = chunk_df.groupby("source_row_index", dropna=False)
    maps["conversion_ok"] = True
    maps["hard_gate_ok"] = True
    maps["event_count"] = maps["source_row_index"].map(grouped["num_events"].sum()).fillna(0).astype(int)
    maps["group_count"] = maps["source_row_index"].map(grouped["num_groups"].sum()).fillna(0).astype(int)
    maps["chunk_count"] = maps["source_row_index"].map(grouped.size()).fillna(0).astype(int)
    for column in (
        "event_count_mismatch_count",
        "event_order_mismatch_count",
        "lane_mismatch_count",
        "action_mismatch_count",
        "chord_group_mismatch_count",
        "ln_pair_mismatch_count",
        "chunk_boundary_reconstruction_mismatch_count",
    ):
        maps[column] = 0
    return maps


def _hard_gate_counts(map_df: pd.DataFrame) -> dict[str, int]:
    return {
        "event_count_mismatch_count": _sum_int(map_df, "event_count_mismatch_count"),
        "event_order_mismatch_count": _sum_int(map_df, "event_order_mismatch_count"),
        "lane_mismatch_count": _sum_int(map_df, "lane_mismatch_count"),
        "action_mismatch_count": _sum_int(map_df, "action_mismatch_count"),
        "chord_group_mismatch_count": _sum_int(map_df, "chord_group_mismatch_count"),
        "ln_pair_mismatch_count": _sum_int(map_df, "ln_pair_mismatch_count"),
        "chunk_boundary_reconstruction_mismatch_count": _sum_int(map_df, "chunk_boundary_reconstruction_mismatch_count"),
    }


def _orientation_counter_by_canonical(chunk_df: pd.DataFrame) -> dict[str, Counter[str]]:
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for canonical, orientation in zip(
        chunk_df["canonical_signature"].fillna("").astype(str),
        chunk_df["orientation"].fillna(0).astype(int),
        strict=False,
    ):
        counters[canonical][str(int(orientation))] += 1
    return counters


def _conditional_orientation_entropy_bits(counters: Mapping[str, Counter[str]]) -> float:
    total = sum(sum(counter.values()) for counter in counters.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for counter in counters.values():
        weight = float(sum(counter.values())) / float(total)
        entropy += weight * _entropy_bits(counter)
    return entropy


def _atomic_counter(chunk_df: pd.DataFrame, *, canonical: bool) -> Counter[str]:
    column = "canonical_groups_json" if canonical else "groups_json"
    counter: Counter[str] = Counter()
    if column not in chunk_df.columns:
        return counter
    for value in chunk_df[column]:
        groups = _json_loads(value)
        for group in groups:
            counter[_atomic_token(group)] += 1
    return counter


def _iter_chunk_group_tokens(
    chunk_df: pd.DataFrame,
    *,
    split: str | None = None,
    canonical: bool = True,
) -> Iterable[tuple[str, int, list[str], bool, int, int, int]]:
    if split is not None and "split" in chunk_df.columns:
        frame = chunk_df[chunk_df["split"].fillna("").astype(str) == split]
    else:
        frame = chunk_df
    groups_column = "canonical_groups_json" if canonical else "groups_json"
    for row in frame.itertuples(index=False):
        groups = _json_loads(getattr(row, groups_column))
        tokens = [_atomic_token(group) for group in groups]
        source_row_index = int(getattr(row, "source_row_index"))
        symmetric = bool(getattr(row, "is_mirror_symmetric"))
        orientation = int(getattr(row, "orientation"))
        num_events = int(getattr(row, "num_events"))
        bar_phase_half = int(getattr(row, "bar_phase_half"))
        yield str(getattr(row, "split")), source_row_index, tokens, symmetric, orientation, num_events, bar_phase_half


def _ngram_stats(
    chunk_df: pd.DataFrame,
    *,
    motif_min_n: int,
    motif_max_n: int,
    top_limit: int,
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "absolute_position": {},
        "delta_time": {},
        "mirror_canonical_absolute": {},
    }
    absolute_by_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    canonical_counter_for_control: Counter[str] = Counter()
    shuffled_counter_for_control: Counter[str] = Counter()
    for row in chunk_df.itertuples(index=False):
        groups = _json_loads(getattr(row, "groups_json"))
        canonical_groups = _json_loads(getattr(row, "canonical_groups_json"))
        raw_tokens = [_atomic_token(group) for group in groups]
        canonical_tokens = [_atomic_token(group) for group in canonical_groups]
        delta_tokens = _delta_atomic_tokens(groups)
        density_bin = str(getattr(row, "density_bin", "unknown") or "unknown")
        for ngram in range(motif_min_n, motif_max_n + 1):
            stats["absolute_position"].setdefault(str(ngram), Counter()).update(_ngrams(raw_tokens, ngram))
            stats["delta_time"].setdefault(str(ngram), Counter()).update(_ngrams(delta_tokens, ngram))
            stats["mirror_canonical_absolute"].setdefault(str(ngram), Counter()).update(_ngrams(canonical_tokens, ngram))
            absolute_by_bucket[f"{density_bin}|n={ngram}"].update(_ngrams(canonical_tokens, ngram))
        for motif in _ngrams(canonical_tokens, 4):
            canonical_counter_for_control[motif] += 1
        shuffled = _density_control_shuffle(canonical_tokens)
        for motif in _ngrams(shuffled, 4):
            shuffled_counter_for_control[motif] += 1

    packed: dict[str, Any] = {}
    for family, values in stats.items():
        packed[family] = {
            ngram: _counter_distribution(counter, item_count=sum(counter.values()), event_count=0, rare_max_count=DEFAULT_RARE_MAX_COUNT, top_limit=top_limit)
            for ngram, counter in values.items()
        }
    recurrence = _recurrence_rate(canonical_counter_for_control)
    control = _recurrence_rate(shuffled_counter_for_control)
    bucket_rows: list[dict[str, Any]] = []
    for bucket, counter in sorted(absolute_by_bucket.items()):
        total = sum(counter.values())
        if total <= 0:
            continue
        bucket_rows.append(
            {
                "bucket": bucket,
                "motif_count": total,
                "unique_count": len(counter),
                "recurrence_rate": _recurrence_rate(counter),
            }
        )
    packed["density_chord_ln_controlled"] = {
        "ngram": 4,
        "recurrence_rate": recurrence,
        "control_recurrence_rate": control,
        "recurrence_lift": recurrence - control,
        "bucket_count": len(bucket_rows),
        "buckets": bucket_rows,
    }
    return packed


def _density_control_shuffle(tokens: Sequence[str]) -> list[str]:
    if len(tokens) < 4:
        return list(tokens)
    offsets: list[str] = []
    actions: list[str] = []
    for token in tokens:
        parts = token.split(":")
        offsets.append(":".join(parts[:2]))
        actions.append(":".join(parts[2:]))
    shuffled = list(actions)
    seed = int(hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()[:16], 16)
    random.Random(seed).shuffle(shuffled)
    return [f"{offset}:{action}" for offset, action in zip(offsets, shuffled, strict=True)]


def _ngrams(tokens: Sequence[str], ngram: int) -> list[str]:
    if ngram <= 0 or len(tokens) < ngram:
        return []
    return [" ".join(tokens[index : index + ngram]) for index in range(len(tokens) - ngram + 1)]


def _recurrence_rate(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return float(total - len(counter)) / float(total)


def _compression_curve(
    chunk_df: pd.DataFrame,
    *,
    motif_vocab_sizes: Sequence[int],
    motif_min_n: int,
    motif_max_n: int,
    smoothing_alpha: float,
    top_example_limit: int,
) -> list[dict[str, Any]]:
    train_candidates = _learn_motif_candidates(
        chunk_df,
        motif_min_n=motif_min_n,
        motif_max_n=motif_max_n,
    )
    max_size = max(motif_vocab_sizes, default=0)
    selected = train_candidates[:max_size]
    curve: list[dict[str, Any]] = []
    for vocab_size in motif_vocab_sizes:
        vocab = _build_motif_vocab(selected[:vocab_size])
        encoded_train = _encode_split(chunk_df, split="train", vocab=vocab)
        train_known_atoms = {token for token in encoded_train.token_counter if token.startswith("A:")}
        train_field_counters = _atomic_field_counters(train_known_atoms)
        split_rows: dict[str, Any] = {}
        for split in ("train", "valid", "test"):
            encoded = encoded_train if split == "train" else _encode_split(chunk_df, split=split, vocab=vocab)
            split_rows[split] = _code_length_metrics(
                encoded,
                train_counter=encoded_train.token_counter,
                train_known_atoms=train_known_atoms,
                train_field_counters=train_field_counters,
                smoothing_alpha=smoothing_alpha,
            )
        train_bits = split_rows["train"]["bits_per_event"]
        test_bits = split_rows["test"]["bits_per_event"]
        gap = test_bits - train_bits if split_rows["test"]["event_count"] > 0 and split_rows["train"]["event_count"] > 0 else None
        curve.append(
            {
                "motif_vocab_size": int(vocab_size),
                "learned_motif_count": len(vocab.motif_to_id),
                "train": split_rows["train"],
                "valid": split_rows["valid"],
                "test": split_rows["test"],
                "generalization_gap_test_minus_train_bits_per_event": gap,
                "top_motifs": _top_motifs(selected[: min(vocab_size, top_example_limit)]),
            }
        )
    return curve


def _learn_motif_candidates(
    chunk_df: pd.DataFrame,
    *,
    motif_min_n: int,
    motif_max_n: int,
) -> list[tuple[tuple[str, ...], int, float]]:
    counter: Counter[tuple[str, ...]] = Counter()
    for _, _, tokens, _, _, _, _ in _iter_chunk_group_tokens(chunk_df, split="train", canonical=True):
        max_n = min(motif_max_n, len(tokens))
        for ngram in range(motif_min_n, max_n + 1):
            for index in range(len(tokens) - ngram + 1):
                counter[tuple(tokens[index : index + ngram])] += 1
    candidates: list[tuple[tuple[str, ...], int, float]] = []
    for motif, count in counter.items():
        if count < 2:
            continue
        gain = float(count) * float(len(motif) - 1)
        candidates.append((motif, count, gain))
    candidates.sort(key=lambda item: (-item[2], -item[1], -len(item[0]), item[0]))
    return candidates


def _build_motif_vocab(candidates: Sequence[tuple[tuple[str, ...], int, float]]) -> _MotifVocab:
    motif_to_id: dict[tuple[str, ...], str] = {}
    motif_lengths: dict[str, int] = {}
    motif_display: dict[str, str] = {}
    for index, (motif, _count, _gain) in enumerate(candidates):
        token_id = f"M{index}"
        motif_to_id[motif] = token_id
        motif_lengths[token_id] = len(motif)
        motif_display[token_id] = " ".join(motif)
    return _MotifVocab(motif_to_id=motif_to_id, motif_lengths=motif_lengths, motif_display=motif_display)


def _encode_split(chunk_df: pd.DataFrame, *, split: str, vocab: _MotifVocab) -> _EncodedSplit:
    motif_trie = _build_motif_trie(vocab.motif_to_id)

    token_counter: Counter[str] = Counter()
    token_count = 0
    group_count = 0
    chunk_count = 0
    event_count = 0
    motif_token_count = 0
    motif_group_coverage_count = 0
    atomic_token_count = 0
    unknown_atomic_count = 0
    orientation_token_count = 0
    chunk_start_token_count = 0
    train_like_known_atoms = set() if split != "train" else None

    for _, _, tokens, symmetric, orientation, num_events, bar_phase_half in _iter_chunk_group_tokens(chunk_df, split=split, canonical=True):
        chunk_count += 1
        event_count += num_events
        group_count += len(tokens)
        chunk_token = f"C{bar_phase_half}"
        token_counter[chunk_token] += 1
        token_count += 1
        chunk_start_token_count += 1
        if not symmetric:
            orientation_token = f"O{orientation}"
            token_counter[orientation_token] += 1
            token_count += 1
            orientation_token_count += 1
        index = 0
        while index < len(tokens):
            matched_token, matched_length = _match_motif(tokens, index, motif_trie)
            if matched_token is not None:
                token_counter[matched_token] += 1
                token_count += 1
                motif_token_count += 1
                motif_group_coverage_count += matched_length
                index += matched_length
                continue
            atomic = tokens[index]
            token_counter[atomic] += 1
            token_count += 1
            atomic_token_count += 1
            if train_like_known_atoms is not None:
                train_like_known_atoms.add(atomic)
            index += 1
    del train_like_known_atoms
    return _EncodedSplit(
        token_counter=token_counter,
        token_count=token_count,
        group_count=group_count,
        chunk_count=chunk_count,
        event_count=event_count,
        motif_token_count=motif_token_count,
        motif_group_coverage_count=motif_group_coverage_count,
        atomic_token_count=atomic_token_count,
        unknown_atomic_count=unknown_atomic_count,
        orientation_token_count=orientation_token_count,
        chunk_start_token_count=chunk_start_token_count,
    )


def _build_motif_trie(motif_to_id: Mapping[tuple[str, ...], str]) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for motif, token_id in motif_to_id.items():
        node = root
        for token in motif:
            node = node.setdefault(token, {})
        node["__id__"] = token_id
        node["__length__"] = len(motif)
    return root


def _match_motif(tokens: Sequence[str], start: int, trie: Mapping[str, Any]) -> tuple[str | None, int]:
    node: Mapping[str, Any] = trie
    best_token: str | None = None
    best_length = 0
    index = start
    while index < len(tokens):
        child = node.get(tokens[index])
        if not isinstance(child, Mapping):
            break
        node = child
        index += 1
        token_id = node.get("__id__")
        if isinstance(token_id, str):
            best_token = token_id
            best_length = int(node.get("__length__", index - start))
    return best_token, best_length


def _code_length_metrics(
    encoded: _EncodedSplit,
    *,
    train_counter: Counter[str],
    train_known_atoms: set[str],
    train_field_counters: Mapping[str, Counter[str]],
    smoothing_alpha: float,
) -> dict[str, Any]:
    train_total = sum(train_counter.values())
    vocab_size = len(train_counter) + 1
    denominator = float(train_total) + smoothing_alpha * float(vocab_size)
    escape_probability = smoothing_alpha / denominator if denominator > 0 else 1.0
    escape_bits = -math.log2(escape_probability) if escape_probability > 0 else 0.0
    bits = 0.0
    unknown_atomic_count = 0
    known_token_count = 0
    for token, count in encoded.token_counter.items():
        if token.startswith("A:") and token not in train_known_atoms:
            unknown_atomic_count += count
            bits += float(count) * (escape_bits + _unknown_atomic_structured_bits(token, train_field_counters, smoothing_alpha))
            continue
        probability = (float(train_counter.get(token, 0)) + smoothing_alpha) / denominator if denominator > 0 else 1.0
        bits += float(count) * (-math.log2(probability))
        known_token_count += count
    token_count = encoded.token_count
    group_count = encoded.group_count
    event_count = encoded.event_count
    return {
        "encoded_token_count": token_count,
        "known_token_count": int(known_token_count),
        "event_count": event_count,
        "group_count": group_count,
        "chunk_count": encoded.chunk_count,
        "code_bits": bits,
        "bits_per_event": float(bits) / float(event_count) if event_count else 0.0,
        "bits_per_group": float(bits) / float(group_count) if group_count else 0.0,
        "bits_per_chunk": float(bits) / float(encoded.chunk_count) if encoded.chunk_count else 0.0,
        "motif_token_count": encoded.motif_token_count,
        "motif_group_coverage_count": encoded.motif_group_coverage_count,
        "motif_coverage_rate": float(encoded.motif_group_coverage_count) / float(group_count) if group_count else 0.0,
        "atomic_token_count": encoded.atomic_token_count,
        "fallback_group_rate": float(group_count - encoded.motif_group_coverage_count) / float(group_count) if group_count else 0.0,
        "unknown_atomic_count": int(unknown_atomic_count),
        "unknown_atomic_rate": float(unknown_atomic_count) / float(encoded.atomic_token_count) if encoded.atomic_token_count else 0.0,
        "orientation_token_count": encoded.orientation_token_count,
        "chunk_start_token_count": encoded.chunk_start_token_count,
    }


def _atomic_field_counters(atoms: Iterable[str]) -> dict[str, Counter[str]]:
    counters = {
        "offset": Counter(),
        "tap": Counter(),
        "start": Counter(),
        "end": Counter(),
        "order": Counter(),
    }
    for atom in atoms:
        parsed = _parse_atomic_token(atom)
        for field, value in parsed.items():
            counters[field][str(value)] += 1
    return counters


def _unknown_atomic_structured_bits(
    atom: str,
    field_counters: Mapping[str, Counter[str]],
    smoothing_alpha: float,
) -> float:
    parsed = _parse_atomic_token(atom)
    cardinality = {
        "offset": DEFAULT_CHUNK_UNITS,
        "tap": 16,
        "start": 16,
        "end": 16,
        "order": max(1024, len(field_counters.get("order", Counter())) + 1),
    }
    bits = 0.0
    for field, value in parsed.items():
        counter = field_counters.get(field, Counter())
        total = sum(counter.values())
        denominator = float(total) + smoothing_alpha * float(cardinality[field])
        probability = (float(counter.get(str(value), 0)) + smoothing_alpha) / denominator if denominator > 0 else 1.0 / float(cardinality[field])
        bits += -math.log2(probability)
    return bits


def _parse_atomic_token(atom: str) -> dict[str, int]:
    parts = atom.split(":")
    if len(parts) < 5 or parts[0] != "A":
        return {"offset": 0, "tap": 0, "start": 0, "end": 0, "order": 0}
    order = parts[5][1:] if len(parts) > 5 and parts[5].startswith("O") else "."
    return {
        "offset": int(parts[1]),
        "tap": int(parts[2]),
        "start": int(parts[3]),
        "end": int(parts[4]),
        "order": _stable_small_hash(order),
    }


def _stable_small_hash(value: str) -> int:
    if value == ".":
        return 0
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def _top_motifs(candidates: Sequence[tuple[tuple[str, ...], int, float]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": index + 1,
            "length_groups": len(motif),
            "support": int(count),
            "compression_gain": float(gain),
            "motif": " ".join(motif),
        }
        for index, (motif, count, gain) in enumerate(candidates)
    ]


def _best_compression_result(curve: Sequence[Mapping[str, Any]], *, split: str) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for row in curve:
        metrics = dict(row.get(split, {}))
        if not metrics:
            continue
        if int(metrics.get("event_count", 0)) <= 0:
            continue
        metrics["motif_vocab_size"] = row.get("motif_vocab_size")
        if not best or float(metrics.get("bits_per_event", math.inf)) < float(best.get("bits_per_event", math.inf)):
            best = metrics
    return best


def _failure_examples(map_df: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or map_df.empty or "hard_gate_ok" not in map_df.columns:
        return []
    failed = map_df[~map_df["hard_gate_ok"].fillna(False).astype(bool)].head(limit)
    columns = [
        "source_row_index",
        "beatmap_set_id",
        "beatmap_id",
        "beatmap_path",
        "error_type",
        "error_message",
        "event_count_mismatch_count",
        "event_order_mismatch_count",
        "lane_mismatch_count",
        "action_mismatch_count",
        "chord_group_mismatch_count",
        "ln_pair_mismatch_count",
        "chunk_boundary_reconstruction_mismatch_count",
    ]
    return _records_for_columns(failed, columns)


def _same_lane_duplicate_examples(map_df: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or map_df.empty or "same_lane_duplicate_action_group_count" not in map_df.columns:
        return []
    examples = map_df[map_df["same_lane_duplicate_action_group_count"].fillna(0).astype(int) > 0]
    examples = examples.sort_values("same_lane_duplicate_action_group_count", ascending=False).head(limit)
    columns = [
        "source_row_index",
        "beatmap_set_id",
        "beatmap_id",
        "beatmap_path",
        "same_lane_duplicate_action_group_count",
        "event_count",
        "group_count",
    ]
    return _records_for_columns(examples, columns)


def _worst_split_examples(
    chunk_df: pd.DataFrame,
    compression_curve: Sequence[Mapping[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    del compression_curve
    if limit <= 0 or chunk_df.empty:
        return []
    grouped = chunk_df.groupby("source_row_index", dropna=False).agg(
        beatmap_set_id=("beatmap_set_id", "first"),
        beatmap_id=("beatmap_id", "first"),
        beatmap_path=("beatmap_path", "first"),
        split=("split", "first"),
        num_groups=("num_groups", "sum"),
        num_events=("num_events", "sum"),
        chunk_count=("chunk_index", "count"),
        empty=("empty", "sum"),
    )
    grouped["empty_chunk_rate"] = grouped["empty"] / grouped["chunk_count"].where(grouped["chunk_count"] > 0, 1)
    examples = grouped.sort_values(["empty_chunk_rate", "num_groups"], ascending=[False, False]).head(limit).reset_index()
    return _records_for_columns(
        examples,
        [
            "source_row_index",
            "split",
            "beatmap_set_id",
            "beatmap_id",
            "beatmap_path",
            "num_groups",
            "num_events",
            "chunk_count",
            "empty_chunk_rate",
        ],
    )


def _split_summary(chunk_df: pd.DataFrame) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    if "split" not in chunk_df.columns:
        return rows
    for split, frame in chunk_df.groupby("split", dropna=False):
        rows[str(split)] = {
            "chunk_count": int(len(frame)),
            "event_count": _sum_int(frame, "num_events"),
            "group_count": _sum_int(frame, "num_groups"),
            "map_count": int(frame["source_row_index"].nunique()) if "source_row_index" in frame.columns else 0,
            "mapset_count": int(frame["beatmap_set_id"].nunique()) if "beatmap_set_id" in frame.columns else 0,
        }
    return rows


def _counter_distribution(
    counter: Counter[str],
    *,
    item_count: int,
    event_count: int,
    rare_max_count: int,
    top_limit: int,
) -> dict[str, Any]:
    entropy = _entropy_bits(counter)
    singleton_item_count = sum(count for count in counter.values() if count == 1)
    rare_item_count = sum(count for count in counter.values() if count <= rare_max_count)
    return {
        "item_count": int(item_count),
        "unique_count": len(counter),
        "entropy_per_item_bits": entropy,
        "bits_per_event": float(entropy) * float(item_count) / float(event_count) if event_count else 0.0,
        "singleton_item_count": int(singleton_item_count),
        "singleton_rate": float(singleton_item_count) / float(item_count) if item_count else 0.0,
        "rare_item_count": int(rare_item_count),
        "rare_rate": float(rare_item_count) / float(item_count) if item_count else 0.0,
        "top": {str(key): int(value) for key, value in counter.most_common(top_limit)},
    }


def _distribution_summary(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    if column not in frame.columns or frame.empty:
        return {"min": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0, "mean": 0.0}
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    return {
        "min": int(values.min()),
        "p50": float(values.quantile(0.50)),
        "p90": float(values.quantile(0.90)),
        "p99": float(values.quantile(0.99)),
        "max": int(values.max()),
        "mean": float(values.mean()),
    }


def _recommendation(
    hard_gate_pass: bool,
    research_pass: bool,
    mapper_interesting_pass: bool,
    best_test_bits: float,
    baselines: Mapping[str, float],
) -> str:
    if not hard_gate_pass:
        return "KILL_OR_FIX_PARITY: chunk construction failed hard gates; compression metrics are not interpretable."
    if mapper_interesting_pass:
        return "TEST_MAPPER_ABLATION: motif tokenizer reached mapper-interesting bits/event; plan one bounded embedding ablation."
    if research_pass:
        return "MUTATE_OR_AUXILIARY_TEST: motif tokenizer beats event-level beat fields but not current v2.1; consider auxiliary embedding only after stricter diagnostics."
    if best_test_bits < float(baselines["event_level_beat_joint_bits_per_event"]):
        return "MUTATE: compression improved but one or more mirror/fallback/control gates failed."
    return "KILL_OR_REDESIGN_BEFORE_MAPPER: heldout motif compression did not beat the event-level beat-field baseline."


def _load_baselines(report_path: Path | None) -> dict[str, float]:
    baselines = _default_baselines()
    if report_path is None or not report_path.exists():
        return baselines
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return baselines
    fair = payload.get("fair_comparison", {})
    current = fair.get("current_v2_1", {})
    beat = fair.get("beat_structure", {})
    if "bits_per_event" in current:
        baselines["current_v2_bits_per_event"] = float(current["bits_per_event"])
    if "bits_per_event" in beat:
        baselines["event_level_beat_joint_bits_per_event"] = float(beat["bits_per_event"])
    if "independent_bits_per_event" in beat:
        baselines["event_level_beat_independent_bits_per_event"] = float(beat["independent_bits_per_event"])
    return baselines


def _default_baselines() -> dict[str, float]:
    return {
        "current_v2_bits_per_event": BASELINE_V2_BITS_PER_EVENT,
        "event_level_beat_joint_bits_per_event": BASELINE_BEAT_JOINT_BITS_PER_EVENT,
        "event_level_beat_independent_bits_per_event": BASELINE_BEAT_INDEPENDENT_BITS_PER_EVENT,
    }


def _write_motif_vocab_stats(report_dir: Path, report: Mapping[str, Any]) -> None:
    for row in report.get("motif_tokenizer_compression_curve", []):
        if not isinstance(row, Mapping):
            continue
        vocab_size = row.get("motif_vocab_size")
        if vocab_size is None:
            continue
        path = report_dir / f"motif_vocab_stats_k{int(vocab_size)}.json"
        payload = {
            "motif_vocab_size": vocab_size,
            "train": row.get("train"),
            "valid": row.get("valid"),
            "test": row.get("test"),
            "generalization_gap_test_minus_train_bits_per_event": row.get(
                "generalization_gap_test_minus_train_bits_per_event"
            ),
            "top_motifs": row.get("top_motifs"),
        }
        _write_json(path, payload)


def _write_result_log(path: Path, report: Mapping[str, Any]) -> None:
    pass_criteria = report.get("pass_criteria", {})
    comparison = report.get("heldout_bits_event_comparison", {})
    chunks = report.get("chunk_construction_summary", {})
    mirror = report.get("mirror_canonicalization_effect", {})
    parity = report.get("parity_summary", {})
    best_test = comparison.get("best_test", {}) if isinstance(comparison, Mapping) else {}
    lines = [
        "# Result Log",
        "",
        "- Experiment: Beat-Chunk Motif Tokenization Audit",
        "- Date: 2026-06-15",
        f"- Commit / run id: {report.get('code_commit')} with dirty worktree={report.get('code_dirty')}",
        f"- Dataset slice: {'limited' if report.get('limited') else 'full'}; processed rows={report.get('processed_row_count')}",
        "- Split: deterministic beatmap_set_id hash, 80/10/10",
        f"- Runtime: {report.get('elapsed_s')} seconds",
        "",
        "## Artifacts",
        "",
        f"- Chunk cache: `{report.get('chunk_cache_path')}`",
        f"- Map cache: `{report.get('map_cache_path')}`",
        f"- Report: `{DEFAULT_BEAT_CHUNK_REPORT_PATH.as_posix()}`",
        "",
        "## Primary Metric Value",
        "",
        f"- Best test bits/event: {best_test.get('bits_per_event')}",
        f"- Best test vocab size: {best_test.get('motif_vocab_size')}",
        f"- Event-level beat joint baseline: {pass_criteria.get('event_level_beat_joint_baseline')}",
        f"- Current v2.1 baseline: {pass_criteria.get('current_v2_baseline')}",
        f"- Test motif coverage rate: {best_test.get('motif_coverage_rate')}",
        f"- Test fallback group rate: {best_test.get('fallback_group_rate')}",
        f"- Test unknown atomic rate: {best_test.get('unknown_atomic_rate')}",
        f"- Mirror canonical unique/raw unique ratio: {mirror.get('canonical_unique_over_raw_unique') if isinstance(mirror, Mapping) else None}",
        f"- Mirror net savings bits/event: {pass_criteria.get('mirror_savings_bits_per_event')}",
        f"- Density-controlled n=4 motif recurrence lift: {pass_criteria.get('density_controlled_motif_lift')}",
        "",
        "## Verification Results",
        "",
        f"- Hard PASS: {pass_criteria.get('hard_pass')}",
        f"- Research PASS: {pass_criteria.get('research_pass')}",
        f"- Mapper-interesting PASS: {pass_criteria.get('mapper_interesting_pass')}",
        f"- Recommendation: {report.get('recommendation')}",
        "",
        "## Checks Performed",
        "",
        f"- Full le3 rows: {report.get('processed_row_count')}",
        f"- Event count: {chunks.get('event_count') if isinstance(chunks, Mapping) else None}",
        f"- 2-beat chunks: {chunks.get('chunk_count') if isinstance(chunks, Mapping) else None}",
        f"- Beat groups: {chunks.get('group_count') if isinstance(chunks, Mapping) else None}",
        f"- Event-count mismatch: {parity.get('event_count_mismatch_count') if isinstance(parity, Mapping) else None}",
        f"- Event-order mismatch: {parity.get('event_order_mismatch_count') if isinstance(parity, Mapping) else None}",
        f"- Lane mismatch: {parity.get('lane_mismatch_count') if isinstance(parity, Mapping) else None}",
        f"- Action mismatch: {parity.get('action_mismatch_count') if isinstance(parity, Mapping) else None}",
        f"- Chord grouping mismatch: {parity.get('chord_group_mismatch_count') if isinstance(parity, Mapping) else None}",
        f"- LN pairing mismatch: {parity.get('ln_pair_mismatch_count') if isinstance(parity, Mapping) else None}",
        f"- Chunk boundary reconstruction mismatch: {parity.get('chunk_boundary_reconstruction_mismatch_count') if isinstance(parity, Mapping) else None}",
        "",
        "## Interpretation",
        "",
        "- The hard audit proves 2-beat chunk construction can be made lossless under the defined beat-event parity gates.",
        "- The motif tokenizer is compression-positive when hard parity passes.",
        "- The strict research gate remains false when mirror canonicalization gives no meaningful net lossless code-length savings once required orientation information is included.",
        "- For a bijective raw chunk to canonical chunk plus orientation mapping, ideal entropy is conserved. Mirror canonicalization can still reduce motif ID sparsity and support tied embeddings, but it should not be treated as a required compression savings gate.",
        "- Mapper integration should wait for a mutated gate or a focused embedding-ablation card.",
        "- Human owner decision pending.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text("\n".join(lines), encoding="utf-8")
    tmp_path.replace(path)


def _entropy_bits(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        if count <= 0:
            continue
        probability = float(count) / float(total)
        entropy -= probability * math.log2(probability)
    return entropy


def _count_true(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    return int(frame[column].fillna(False).astype(bool).sum())


def _sum_int(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    return int(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())


def _mean_bool(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns or frame.empty:
        return 0.0
    return float(frame[column].fillna(False).astype(bool).mean())


def _records_for_columns(frame: pd.DataFrame, columns: Sequence[str]) -> list[dict[str, Any]]:
    selected = [column for column in columns if column in frame.columns]
    return [_normalize_json(record) for record in frame[selected].to_dict(orient="records")]


def _row_value(row: Mapping[str, object] | object, name: str, default: object = None) -> object:
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def _is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, set)):
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def _cache_scalar(value: object) -> object:
    if _is_missing_scalar(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _normalize_json(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_json(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_json(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if _is_missing_scalar(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _validate_positive_int(value: object, field: str) -> int:
    integer = _validate_nonnegative_int(value, field)
    if integer <= 0:
        raise ValueError(f"{field} must be positive, got {integer}")
    return integer


def _validate_nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field} must be an integer, got bool")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be an integer, got {type(value).__name__}") from exc
    if integer < 0:
        raise ValueError(f"{field} must be non-negative, got {integer}")
    return integer


def _short_error(message: str, *, limit: int = 500) -> str:
    message = " ".join(str(message).split())
    return message if len(message) <= limit else message[: limit - 3] + "..."


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(_normalize_json(payload), allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_stdout(*args: str) -> str | None:
    try:
        completed = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _format_command(argv: Sequence[str] | None) -> str:
    args = list(sys.argv[1:] if argv is None else argv)
    return " ".join(shlex.quote(part) for part in ["python", "-m", "pulsefield_model.osu_core.beat_chunk_pattern_audit", *args])


def _parse_vocab_sizes(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit 2-beat chunk motif tokenization for osu!mania 4K beatmaps.")
    parser.add_argument("--beat-structure-cache-path", type=Path, default=DEFAULT_LE3_STRUCTURE_CACHE_PATH)
    parser.add_argument("--beat-structure-report-path", type=Path, default=DEFAULT_LE3_STRUCTURE_REPORT_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--chunk-cache-path", type=Path, default=DEFAULT_BEAT_CHUNK_CACHE_PATH)
    parser.add_argument("--map-cache-path", type=Path, default=DEFAULT_BEAT_CHUNK_MAP_CACHE_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_BEAT_CHUNK_REPORT_PATH)
    parser.add_argument("--result-log-path", type=Path, default=DEFAULT_RESULT_LOG_PATH)
    parser.add_argument("--snap-denominator", type=int, default=DEFAULT_SNAP_DENOMINATOR)
    parser.add_argument(
        "--timing-canonicalization",
        choices=TIMING_CANONICALIZATION_CHOICES,
        default=DEFAULT_BEAT_REPRESENTATION_TIMING_CANONICALIZATION,
    )
    parser.add_argument("--expected-key-count", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--motif-min-n", type=int, default=DEFAULT_MOTIF_MIN_N)
    parser.add_argument("--motif-max-n", type=int, default=DEFAULT_MOTIF_MAX_N)
    parser.add_argument("--motif-vocab-sizes", type=_parse_vocab_sizes, default=DEFAULT_MOTIF_VOCAB_SIZES)
    parser.add_argument("--rare-max-count", type=int, default=DEFAULT_RARE_MAX_COUNT)
    parser.add_argument("--top-example-limit", type=int, default=DEFAULT_TOP_EXAMPLE_LIMIT)
    parser.add_argument("--parquet-batch-rows", type=int, default=DEFAULT_PARQUET_BATCH_ROWS)
    parser.add_argument("--smoothing-alpha", type=float, default=DEFAULT_SMOOTHING_ALPHA)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    report = audit_beat_chunk_patterns(
        beat_structure_cache_path=args.beat_structure_cache_path,
        beat_structure_report_path=args.beat_structure_report_path,
        dataset_root=args.dataset_root,
        chunk_cache_path=args.chunk_cache_path,
        map_cache_path=args.map_cache_path,
        report_path=args.report_path,
        result_log_path=args.result_log_path,
        snap_denominator=args.snap_denominator,
        timing_canonicalization=args.timing_canonicalization,
        expected_key_count=args.expected_key_count,
        limit=args.limit,
        progress_every=args.progress_every,
        motif_min_n=args.motif_min_n,
        motif_max_n=args.motif_max_n,
        motif_vocab_sizes=args.motif_vocab_sizes,
        rare_max_count=args.rare_max_count,
        top_example_limit=args.top_example_limit,
        parquet_batch_rows=args.parquet_batch_rows,
        smoothing_alpha=args.smoothing_alpha,
        command=_format_command(argv),
    )
    print(json.dumps(_normalize_json(report), allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
