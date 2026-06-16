from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np
import pandas as pd

from pulsefield_model.osu_core.beat_representation import (
    DEFAULT_BEAT_REPRESENTATION_TIMING_CANONICALIZATION,
    DEFAULT_SNAP_DENOMINATOR,
    DEFAULT_SNAP_DIAGNOSTIC_SUBDIVISIONS,
    BeatEvent,
    BeatEventKind,
    hitobjects_to_beat_events,
)
from pulsefield_model.osu_core.hitobjects import ManiaHitObject, ManiaHitObjectKind, parse_mania_hit_objects
from pulsefield_model.osu_core.timing import RedTimingPoint, require_red_timing_points
from pulsefield_model.timing.canonicalization import TIMING_CANONICALIZATION_CHOICES


AUDIT_SCHEMA_VERSION: Final[int] = 1
DEFAULT_LE3_INDEX_PATH: Final[Path] = Path(
    "artifacts/indexes/beatmap_index_4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3.parquet",
)
DEFAULT_LE3_AUDIT_CACHE_PATH: Final[Path] = Path(
    "artifacts/audits/beat_representation/"
    "beat_representation_audit_4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3.parquet",
)
DEFAULT_LE3_AUDIT_REPORT_PATH: Final[Path] = Path(
    "artifacts/reports/audits/beat_representation/"
    "beat_representation_audit_4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3.json",
)
DEFAULT_DATASET_ROOT: Final[Path] = Path("dataset")
DEFAULT_TOP_EXAMPLE_LIMIT: Final[int] = 20
SNAP_ERROR_BOUND_EPS_BEATS: Final[float] = 1e-9
_REQUIRED_INDEX_COLUMNS: Final[frozenset[str]] = frozenset(("shard", "beatmap_path"))
_OPTIONAL_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "beatmap_set_id",
    "beatmap_id",
    "title",
    "artist",
    "creator",
    "version",
    "difficulty",
    "mode",
    "key_count",
)


@dataclass(frozen=True)
class HitObjectLineScan:
    raw_hitobject_line_count: int = 0
    malformed_hitobject_line_count: int = 0
    unsupported_hitobject_type_count: int = 0
    out_of_range_x_count: int = 0
    nonfinite_hitobject_time_count: int = 0
    negative_hold_duration_count: int = 0
    zero_hold_duration_count: int = 0


def audit_beat_representation_index(
    *,
    source_index_path: str | Path = DEFAULT_LE3_INDEX_PATH,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    cache_path: str | Path = DEFAULT_LE3_AUDIT_CACHE_PATH,
    report_path: str | Path | None = DEFAULT_LE3_AUDIT_REPORT_PATH,
    snap_denominator: int = DEFAULT_SNAP_DENOMINATOR,
    timing_canonicalization: str = DEFAULT_BEAT_REPRESENTATION_TIMING_CANONICALIZATION,
    expected_key_count: int | None = 4,
    diagnostic_subdivisions: Sequence[int] = DEFAULT_SNAP_DIAGNOSTIC_SUBDIVISIONS,
    limit: int | None = None,
    progress_every: int = 0,
    top_example_limit: int = DEFAULT_TOP_EXAMPLE_LIMIT,
    command: str | None = None,
) -> dict[str, Any]:
    """Audit beatmap-to-beat-event conversion for every row in an index.

    The cache is one row per source index row. Per-event data is intentionally
    reduced to summary metrics so the artifact remains useful for full-index
    reruns without becoming a second event dataset.
    """

    started_at = time.perf_counter()
    source_index_path = Path(source_index_path)
    dataset_root = Path(dataset_root)
    cache_path = Path(cache_path)
    report_path = None if report_path is None else Path(report_path)
    snap_denominator = _validate_positive_int(snap_denominator, "snap_denominator")
    diagnostic_subdivisions = _validate_diagnostic_subdivisions(diagnostic_subdivisions)
    if timing_canonicalization not in TIMING_CANONICALIZATION_CHOICES:
        choices = ", ".join(TIMING_CANONICALIZATION_CHOICES)
        raise ValueError(f"timing_canonicalization must be one of {choices}, got {timing_canonicalization!r}")
    if expected_key_count is not None:
        expected_key_count = _validate_positive_int(expected_key_count, "expected_key_count")
    if limit is not None and int(limit) < 0:
        raise ValueError(f"limit must be non-negative, got {limit!r}")
    if progress_every < 0:
        raise ValueError(f"progress_every must be non-negative, got {progress_every!r}")
    top_example_limit = _validate_nonnegative_int(top_example_limit, "top_example_limit")

    source_df = pd.read_parquet(source_index_path)
    _require_index_columns(source_df, source_index_path)
    duplicate_path_counts = _duplicate_path_counts(source_df)
    rows: list[dict[str, Any]] = []
    total_rows = len(source_df)
    max_rows = total_rows if limit is None else min(int(limit), total_rows)

    for source_row_index, row in enumerate(source_df.itertuples(index=False), start=0):
        if source_row_index >= max_rows:
            break
        rows.append(
            audit_beat_representation_row(
                row,
                source_row_index=source_row_index,
                dataset_root=dataset_root,
                duplicate_source_path_count=duplicate_path_counts.get(_source_path_key(row), 1),
                snap_denominator=snap_denominator,
                timing_canonicalization=timing_canonicalization,
                expected_key_count=expected_key_count,
                diagnostic_subdivisions=diagnostic_subdivisions,
            ),
        )
        if progress_every > 0 and (source_row_index + 1) % progress_every == 0:
            print(f"processed {source_row_index + 1}/{max_rows} maps", file=sys.stderr)

    cache_df = pd.DataFrame.from_records(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_cache_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    cache_df.to_parquet(tmp_cache_path, index=False)
    tmp_cache_path.replace(cache_path)

    report = build_beat_representation_audit_report(
        cache_df,
        source_index_path=source_index_path,
        source_index_sha256=_sha256(source_index_path),
        dataset_root=dataset_root,
        cache_path=cache_path,
        cache_sha256=_sha256(cache_path),
        source_row_count=total_rows,
        elapsed_s=time.perf_counter() - started_at,
        snap_denominator=snap_denominator,
        timing_canonicalization=timing_canonicalization,
        expected_key_count=expected_key_count,
        diagnostic_subdivisions=diagnostic_subdivisions,
        limit=limit,
        top_example_limit=top_example_limit,
        command=command,
    )
    if report_path is not None:
        _write_json(report_path, report)
    return report


def audit_beat_representation_row(
    row: Mapping[str, object] | object,
    *,
    source_row_index: int,
    dataset_root: Path,
    duplicate_source_path_count: int,
    snap_denominator: int,
    timing_canonicalization: str,
    expected_key_count: int | None,
    diagnostic_subdivisions: Sequence[int],
) -> dict[str, Any]:
    base = _base_audit_row(row, source_row_index=source_row_index)
    base["duplicate_source_path_count"] = int(duplicate_source_path_count)
    invariant_errors = _index_invariant_errors(
        row,
        duplicate_source_path_count=duplicate_source_path_count,
        expected_key_count=expected_key_count,
    )
    base["index_invariant_errors"] = "|".join(invariant_errors)
    base["index_invariant_ok"] = not invariant_errors
    try:
        beatmap_path = resolve_index_beatmap_path(dataset_root, _row_value(row, "shard"), _row_value(row, "beatmap_path"))
        base["resolved_beatmap_path"] = beatmap_path.as_posix()
        if not beatmap_path.is_file():
            raise FileNotFoundError(f"beatmap_path does not exist: {beatmap_path}")
    except Exception as exc:  # noqa: BLE001 - audit records row-local path failures.
        base.update(_failure_fields("path_error", str(exc)))
        return _finalize_status(base)

    try:
        scan = scan_hitobject_lines(beatmap_path)
    except Exception as exc:  # noqa: BLE001 - audit records row-local source failures.
        base.update(_failure_fields(type(exc).__name__, str(exc)))
        return _finalize_status(base)
    _add_scan_metrics(base, scan)

    try:
        timing_points = require_red_timing_points(beatmap_path)
        hitobjects = parse_mania_hit_objects(beatmap_path, expected_key_count=expected_key_count)
        events = hitobjects_to_beat_events(
            hitobjects,
            timing_points,
            snap_denominator=snap_denominator,
            timing_canonicalization=timing_canonicalization,
            include_diagnostics=True,
            diagnostic_subdivisions=diagnostic_subdivisions,
        )
    except Exception as exc:  # noqa: BLE001 - full-index audit must not fail fast.
        base.update(_failure_fields(type(exc).__name__, str(exc)))
        return _finalize_status(base)

    base["path_ok"] = True
    base["conversion_ok"] = True
    _add_timing_metrics(base, timing_points)
    _add_hitobject_metrics(base, hitobjects)
    _add_event_metrics(base, events, hitobjects=hitobjects, timing_points=timing_points, snap_denominator=snap_denominator)
    return _finalize_status(base)


def build_beat_representation_audit_report(
    cache: pd.DataFrame | str | Path,
    *,
    source_index_path: str | Path | None = None,
    source_index_sha256: str | None = None,
    dataset_root: str | Path | None = None,
    cache_path: str | Path | None = None,
    cache_sha256: str | None = None,
    source_row_count: int | None = None,
    elapsed_s: float | None = None,
    snap_denominator: int | None = None,
    timing_canonicalization: str | None = None,
    expected_key_count: int | None = None,
    diagnostic_subdivisions: Sequence[int] | None = None,
    limit: int | None = None,
    top_example_limit: int = DEFAULT_TOP_EXAMPLE_LIMIT,
    command: str | None = None,
) -> dict[str, Any]:
    cache_df = pd.read_parquet(cache) if isinstance(cache, (str, Path)) else cache.copy()
    if source_row_count is None:
        source_row_count = len(cache_df)
    processed_row_count = len(cache_df)
    conversion_success_count = _count_true(cache_df, "conversion_ok")
    conversion_failure_count = processed_row_count - conversion_success_count
    ok_count = _count_true(cache_df, "ok")
    semantic_guard_success_count = _count_true(cache_df[cache_df.get("conversion_ok", False).astype(bool)], "semantic_guard_ok")
    total_events = _sum_int(cache_df, "event_count")
    threshold_counts = {
        "le_2ms": _sum_int(cache_df, "snap_error_le_2ms_event_count"),
        "le_5ms": _sum_int(cache_df, "snap_error_le_5ms_event_count"),
        "le_10ms": _sum_int(cache_df, "snap_error_le_10ms_event_count"),
        "gt_10ms": _sum_int(cache_df, "snap_error_gt_10ms_event_count"),
    }

    return _normalize_json(
        {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "source_index_path": None if source_index_path is None else Path(source_index_path).as_posix(),
            "source_index_sha256": source_index_sha256,
            "dataset_root": None if dataset_root is None else Path(dataset_root).as_posix(),
            "cache_path": None if cache_path is None else Path(cache_path).as_posix(),
            "cache_sha256": cache_sha256,
            "source_row_count": int(source_row_count),
            "processed_row_count": int(processed_row_count),
            "limited": limit is not None,
            "limit": None if limit is None else int(limit),
            "config": {
                "snap_denominator": snap_denominator,
                "timing_canonicalization": timing_canonicalization,
                "expected_key_count": expected_key_count,
                "diagnostic_subdivisions": None if diagnostic_subdivisions is None else list(diagnostic_subdivisions),
                "snap_error_bound_eps_beats": SNAP_ERROR_BOUND_EPS_BEATS,
            },
            "counts": {
                "ok_count": ok_count,
                "not_ok_count": processed_row_count - ok_count,
                "conversion_success_count": conversion_success_count,
                "conversion_failure_count": conversion_failure_count,
                "path_failure_count": processed_row_count - _count_true(cache_df, "path_ok"),
                "index_invariant_failure_count": processed_row_count - _count_true(cache_df, "index_invariant_ok"),
                "semantic_guard_failure_count": conversion_success_count - semantic_guard_success_count,
                "event_count_match_failure_count": _count_positive(cache_df, "event_count_mismatch_count"),
                "snap_bound_violation_row_count": _count_positive(cache_df, "snap_error_bound_violation_count"),
                "snapped_hold_inversion_row_count": _count_positive(cache_df, "snapped_hold_inversion_count"),
                "negative_hold_duration_row_count": _count_positive(cache_df, "negative_hold_duration_count"),
                "duplicate_red_timing_offset_row_count": _count_positive(cache_df, "duplicate_red_timing_offset_count"),
                "conflicting_duplicate_red_timing_offset_row_count": _count_positive(
                    cache_df,
                    "conflicting_duplicate_red_timing_offset_count",
                ),
            },
            "failure_counts_by_error_type": _value_counts(cache_df, "error_type", exclude_empty=True),
            "index_invariant_failure_counts": _split_counts(cache_df, "index_invariant_errors"),
            "event_totals": {
                "hitobject_count": _sum_int(cache_df, "hitobject_count"),
                "tap_count": _sum_int(cache_df, "tap_count"),
                "hold_count": _sum_int(cache_df, "hold_count"),
                "event_count": total_events,
                "hold_start_event_count": _sum_int(cache_df, "hold_start_event_count"),
                "hold_end_event_count": _sum_int(cache_df, "hold_end_event_count"),
            },
            "snap_error_ms": {
                "max_row_max_abs": _max_float(cache_df, "max_abs_snap_error_ms"),
                "p50_row_max_abs": _percentile_float(cache_df, "max_abs_snap_error_ms", 50),
                "p90_row_max_abs": _percentile_float(cache_df, "max_abs_snap_error_ms", 90),
                "p95_row_max_abs": _percentile_float(cache_df, "max_abs_snap_error_ms", 95),
                "p99_row_max_abs": _percentile_float(cache_df, "max_abs_snap_error_ms", 99),
                "max_row_p99_abs": _max_float(cache_df, "p99_abs_snap_error_ms"),
                "threshold_event_counts": threshold_counts,
                "threshold_event_rates": {
                    key: (float(value) / float(total_events) if total_events else 0.0)
                    for key, value in threshold_counts.items()
                },
            },
            "snap_error_beats": {
                "max_row_max_abs": _max_float(cache_df, "max_abs_snap_error_beats"),
                "p99_row_max_abs": _percentile_float(cache_df, "max_abs_snap_error_beats", 99),
                "max_row_p99_abs": _max_float(cache_df, "p99_abs_snap_error_beats"),
                "snap_error_bound_beats": _max_float(cache_df, "snap_error_bound_beats"),
                "snap_error_bound_violation_count": _sum_int(cache_df, "snap_error_bound_violation_count"),
            },
            "worst_snap_error_examples": _top_examples(cache_df, "max_abs_snap_error_ms", top_example_limit),
            "failure_examples": _failure_examples(cache_df, top_example_limit),
            "guard_failure_examples": _guard_failure_examples(cache_df, top_example_limit),
            "elapsed_s": elapsed_s,
            "code_commit": _git_stdout("rev-parse", "HEAD"),
            "code_dirty": bool(_git_stdout("status", "--porcelain")),
            "command": command,
        }
    )


def resolve_index_beatmap_path(dataset_root: str | Path, shard: object, beatmap_path: object) -> Path:
    dataset_root = Path(dataset_root)
    shard_text = _required_text(shard, field="shard")
    shard_path = Path(shard_text)
    if shard_path.is_absolute():
        raise ValueError(f"shard must be relative to dataset_root, got absolute path: {shard_text!r}")
    if len(shard_path.parts) != 1 or any(part in {"", ".", ".."} for part in shard_path.parts):
        raise ValueError(f"shard must be a single relative path component: {shard_text!r}")

    beatmap_text = _required_text(beatmap_path, field="beatmap_path")
    relative_path = Path(beatmap_text)
    if relative_path.is_absolute():
        raise ValueError(f"beatmap_path must be relative to shard root, got absolute path: {beatmap_text!r}")
    if any(part == ".." for part in relative_path.parts):
        raise ValueError(f"beatmap_path must not contain parent traversal: {beatmap_text!r}")

    root = dataset_root.resolve(strict=False)
    resolved = (dataset_root / shard_path / relative_path).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"beatmap_path must stay under dataset_root {root}: {beatmap_text!r}") from exc
    return dataset_root / shard_path / relative_path


def scan_hitobject_lines(beatmap_path: str | Path) -> HitObjectLineScan:
    section: str | None = None
    raw_hitobject_line_count = 0
    malformed_hitobject_line_count = 0
    unsupported_hitobject_type_count = 0
    out_of_range_x_count = 0
    nonfinite_hitobject_time_count = 0
    negative_hold_duration_count = 0
    zero_hold_duration_count = 0

    with Path(beatmap_path).open("r", encoding="utf-8-sig", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("//"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                continue
            if section != "HitObjects":
                continue

            raw_hitobject_line_count += 1
            parts = line.split(",")
            if len(parts) < 5:
                malformed_hitobject_line_count += 1
                continue
            try:
                x = int(parts[0])
                start_time_ms = float(parts[2])
                object_type = int(parts[3])
            except ValueError:
                malformed_hitobject_line_count += 1
                continue

            if x < 0 or x >= 512:
                out_of_range_x_count += 1
            if not math.isfinite(start_time_ms):
                nonfinite_hitobject_time_count += 1

            is_hold = (object_type & 128) != 0
            is_tap = (object_type & 1) != 0
            if not is_hold and not is_tap:
                unsupported_hitobject_type_count += 1
            if not is_hold:
                continue
            if len(parts) < 6:
                malformed_hitobject_line_count += 1
                continue
            try:
                end_time_ms = float(parts[5].split(":", 1)[0])
            except ValueError:
                malformed_hitobject_line_count += 1
                continue
            if not math.isfinite(end_time_ms):
                nonfinite_hitobject_time_count += 1
                continue
            if math.isfinite(start_time_ms):
                if end_time_ms < start_time_ms:
                    negative_hold_duration_count += 1
                elif end_time_ms == start_time_ms:
                    zero_hold_duration_count += 1

    return HitObjectLineScan(
        raw_hitobject_line_count=raw_hitobject_line_count,
        malformed_hitobject_line_count=malformed_hitobject_line_count,
        unsupported_hitobject_type_count=unsupported_hitobject_type_count,
        out_of_range_x_count=out_of_range_x_count,
        nonfinite_hitobject_time_count=nonfinite_hitobject_time_count,
        negative_hold_duration_count=negative_hold_duration_count,
        zero_hold_duration_count=zero_hold_duration_count,
    )


def _base_audit_row(row: Mapping[str, object] | object, *, source_row_index: int) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "source_row_index": int(source_row_index),
        "resolved_beatmap_path": "",
        "duplicate_source_path_count": 1,
        "index_invariant_ok": True,
        "index_invariant_errors": "",
        "path_ok": False,
        "conversion_ok": False,
        "semantic_guard_ok": False,
        "ok": False,
        "error_type": "",
        "error_message": "",
        "red_timing_count": 0,
        "duplicate_red_timing_offset_count": 0,
        "conflicting_duplicate_red_timing_offset_count": 0,
        "raw_hitobject_line_count": 0,
        "malformed_hitobject_line_count": 0,
        "unsupported_hitobject_type_count": 0,
        "out_of_range_x_count": 0,
        "nonfinite_hitobject_time_count": 0,
        "negative_hold_duration_count": 0,
        "zero_hold_duration_count": 0,
        "hitobject_count": 0,
        "tap_count": 0,
        "hold_count": 0,
        "event_count": 0,
        "expected_event_count": 0,
        "event_count_match": False,
        "event_count_mismatch_count": 0,
        "tap_event_count": 0,
        "hold_start_event_count": 0,
        "hold_end_event_count": 0,
        "event_before_first_redline_count": 0,
        "snapped_hold_inversion_count": 0,
        "max_abs_snap_error_ms": 0.0,
        "p50_abs_snap_error_ms": 0.0,
        "p90_abs_snap_error_ms": 0.0,
        "p95_abs_snap_error_ms": 0.0,
        "p99_abs_snap_error_ms": 0.0,
        "max_abs_snap_error_beats": 0.0,
        "p99_abs_snap_error_beats": 0.0,
        "snap_error_bound_beats": 0.0,
        "snap_error_bound_violation_count": 0,
        "snap_error_le_2ms_event_count": 0,
        "snap_error_le_5ms_event_count": 0,
        "snap_error_le_10ms_event_count": 0,
        "snap_error_gt_10ms_event_count": 0,
        "max_snap_error_kind": "",
        "max_snap_error_lane": -1,
        "max_snap_error_time_ms": 0.0,
        "max_snap_error_snapped_time_ms": 0.0,
        "max_snap_error_ms": 0.0,
        "max_snap_error_abs_ms": 0.0,
        "max_snap_error_abs_beats": 0.0,
        "max_snap_error_redline_bpm": 0.0,
        "max_snap_error_bpm": 0.0,
        "max_snap_error_beat_offset_numerator": 0,
        "max_snap_error_beat_offset_denominator": 0,
    }
    for column in _OPTIONAL_IDENTITY_COLUMNS:
        base[column] = _row_value(row, column, default=None)
    base["shard"] = _row_value(row, "shard", default="")
    base["beatmap_path"] = _row_value(row, "beatmap_path", default="")
    return {key: _cache_scalar(value) for key, value in base.items()}


def _add_scan_metrics(base: dict[str, Any], scan: HitObjectLineScan) -> None:
    base.update(
        {
            "raw_hitobject_line_count": scan.raw_hitobject_line_count,
            "malformed_hitobject_line_count": scan.malformed_hitobject_line_count,
            "unsupported_hitobject_type_count": scan.unsupported_hitobject_type_count,
            "out_of_range_x_count": scan.out_of_range_x_count,
            "nonfinite_hitobject_time_count": scan.nonfinite_hitobject_time_count,
            "negative_hold_duration_count": scan.negative_hold_duration_count,
            "zero_hold_duration_count": scan.zero_hold_duration_count,
        }
    )


def _add_timing_metrics(base: dict[str, Any], timing_points: Sequence[RedTimingPoint]) -> None:
    offsets: dict[float, set[float]] = {}
    for point in timing_points:
        offsets.setdefault(float(point.offset_ms), set()).add(float(point.beat_length_ms))
    duplicate_offsets = {offset: values for offset, values in offsets.items() if len(values) >= 1}
    base["red_timing_count"] = len(timing_points)
    base["duplicate_red_timing_offset_count"] = sum(1 for offset in offsets if sum(point.offset_ms == offset for point in timing_points) > 1)
    base["conflicting_duplicate_red_timing_offset_count"] = sum(
        1
        for offset, values in duplicate_offsets.items()
        if len(values) > 1 and sum(point.offset_ms == offset for point in timing_points) > 1
    )


def _add_hitobject_metrics(base: dict[str, Any], hitobjects: Sequence[ManiaHitObject]) -> None:
    tap_count = sum(1 for hitobject in hitobjects if hitobject.kind == ManiaHitObjectKind.TAP)
    hold_count = sum(1 for hitobject in hitobjects if hitobject.kind == ManiaHitObjectKind.HOLD)
    base["hitobject_count"] = len(hitobjects)
    base["tap_count"] = tap_count
    base["hold_count"] = hold_count
    base["expected_event_count"] = tap_count + 2 * hold_count


def _add_event_metrics(
    base: dict[str, Any],
    events: Sequence[BeatEvent],
    *,
    hitobjects: Sequence[ManiaHitObject],
    timing_points: Sequence[RedTimingPoint],
    snap_denominator: int,
) -> None:
    event_count = len(events)
    expected_event_count = int(base["expected_event_count"])
    abs_snap_error_ms = np.asarray([abs(event.snap_error_ms) for event in events], dtype=np.float64)
    abs_snap_error_beats = np.asarray(
        [
            event.diagnostics.abs_snap_error_beats
            if event.diagnostics is not None
            else abs(event.snap_error_ms) / event.beat_length_ms
            for event in events
        ],
        dtype=np.float64,
    )
    snap_error_bound_beats = 0.5 / float(snap_denominator) + SNAP_ERROR_BOUND_EPS_BEATS

    base["event_count"] = event_count
    base["event_count_match"] = event_count == expected_event_count
    base["event_count_mismatch_count"] = 0 if event_count == expected_event_count else 1
    base["tap_event_count"] = sum(1 for event in events if event.kind == BeatEventKind.TAP)
    base["hold_start_event_count"] = sum(1 for event in events if event.kind == BeatEventKind.HOLD_START)
    base["hold_end_event_count"] = sum(1 for event in events if event.kind == BeatEventKind.HOLD_END)
    base["event_before_first_redline_count"] = _event_before_first_redline_count(events, timing_points)
    base["snapped_hold_inversion_count"] = _snapped_hold_inversion_count(events, hitobjects)
    base["snap_error_bound_beats"] = snap_error_bound_beats
    base["snap_error_bound_violation_count"] = int(np.sum(abs_snap_error_beats > snap_error_bound_beats))
    if event_count == 0:
        return

    base["max_abs_snap_error_ms"] = float(np.max(abs_snap_error_ms))
    base["p50_abs_snap_error_ms"] = float(np.percentile(abs_snap_error_ms, 50))
    base["p90_abs_snap_error_ms"] = float(np.percentile(abs_snap_error_ms, 90))
    base["p95_abs_snap_error_ms"] = float(np.percentile(abs_snap_error_ms, 95))
    base["p99_abs_snap_error_ms"] = float(np.percentile(abs_snap_error_ms, 99))
    base["max_abs_snap_error_beats"] = float(np.max(abs_snap_error_beats))
    base["p99_abs_snap_error_beats"] = float(np.percentile(abs_snap_error_beats, 99))
    base["snap_error_le_2ms_event_count"] = int(np.sum(abs_snap_error_ms <= 2.0))
    base["snap_error_le_5ms_event_count"] = int(np.sum(abs_snap_error_ms <= 5.0))
    base["snap_error_le_10ms_event_count"] = int(np.sum(abs_snap_error_ms <= 10.0))
    base["snap_error_gt_10ms_event_count"] = int(np.sum(abs_snap_error_ms > 10.0))

    worst_index = int(np.argmax(abs_snap_error_ms))
    worst = events[worst_index]
    base["max_snap_error_kind"] = worst.kind.value
    base["max_snap_error_lane"] = worst.lane
    base["max_snap_error_time_ms"] = worst.time_ms
    base["max_snap_error_snapped_time_ms"] = worst.snapped_time_ms
    base["max_snap_error_ms"] = worst.snap_error_ms
    base["max_snap_error_abs_ms"] = float(abs_snap_error_ms[worst_index])
    base["max_snap_error_abs_beats"] = float(abs_snap_error_beats[worst_index])
    base["max_snap_error_redline_bpm"] = worst.redline_bpm
    base["max_snap_error_bpm"] = worst.bpm
    base["max_snap_error_beat_offset_numerator"] = worst.beat_offset_numerator
    base["max_snap_error_beat_offset_denominator"] = worst.beat_offset_denominator


def _finalize_status(base: dict[str, Any]) -> dict[str, Any]:
    semantic_guard_ok = (
        bool(base["conversion_ok"])
        and bool(base["event_count_match"])
        and int(base["malformed_hitobject_line_count"]) == 0
        and int(base["unsupported_hitobject_type_count"]) == 0
        and int(base["nonfinite_hitobject_time_count"]) == 0
        and int(base["negative_hold_duration_count"]) == 0
        and int(base["snapped_hold_inversion_count"]) == 0
        and int(base["snap_error_bound_violation_count"]) == 0
    )
    base["semantic_guard_ok"] = semantic_guard_ok
    base["ok"] = (
        bool(base["index_invariant_ok"])
        and bool(base["path_ok"])
        and bool(base["conversion_ok"])
        and bool(base["semantic_guard_ok"])
    )
    return {key: _cache_scalar(value) for key, value in base.items()}


def _failure_fields(error_type: str, error_message: str) -> dict[str, Any]:
    return {
        "path_ok": False if error_type == "path_error" else True,
        "conversion_ok": False,
        "error_type": error_type,
        "error_message": _short_error(error_message),
    }


def _snapped_hold_inversion_count(events: Sequence[BeatEvent], hitobjects: Sequence[ManiaHitObject]) -> int:
    by_key: dict[tuple[BeatEventKind, int, float], list[BeatEvent]] = {}
    for event in events:
        by_key.setdefault((event.kind, event.lane, event.time_ms), []).append(event)

    inversions = 0
    for hitobject in hitobjects:
        if hitobject.kind != ManiaHitObjectKind.HOLD:
            continue
        if not (
            math.isfinite(hitobject.start_time_ms)
            and math.isfinite(hitobject.end_time_ms)
            and hitobject.end_time_ms > hitobject.start_time_ms
        ):
            continue
        start_events = by_key.get((BeatEventKind.HOLD_START, hitobject.lane, hitobject.start_time_ms), [])
        end_events = by_key.get((BeatEventKind.HOLD_END, hitobject.lane, hitobject.end_time_ms), [])
        if not start_events or not end_events:
            continue
        start_event = start_events.pop(0)
        end_event = end_events.pop(0)
        if end_event.snapped_time_ms <= start_event.snapped_time_ms:
            inversions += 1
    return inversions


def _event_before_first_redline_count(events: Sequence[BeatEvent], timing_points: Sequence[RedTimingPoint]) -> int:
    if not timing_points:
        return 0
    first_offset_ms = min(float(point.offset_ms) for point in timing_points)
    return sum(1 for event in events if event.time_ms < first_offset_ms)


def _index_invariant_errors(
    row: Mapping[str, object] | object,
    *,
    duplicate_source_path_count: int,
    expected_key_count: int | None,
) -> list[str]:
    errors: list[str] = []
    if duplicate_source_path_count > 1:
        errors.append("duplicate_source_path")
    if _has_row_value(row, "mode") and not _is_missing_scalar(_row_value(row, "mode")):
        try:
            if int(_row_value(row, "mode")) != 3:
                errors.append("mode_not_mania")
        except (TypeError, ValueError):
            errors.append("mode_not_integer")
    if expected_key_count is not None and _has_row_value(row, "key_count") and not _is_missing_scalar(_row_value(row, "key_count")):
        try:
            if int(_row_value(row, "key_count")) != expected_key_count:
                errors.append("key_count_mismatch")
        except (TypeError, ValueError):
            errors.append("key_count_not_integer")
    if _has_row_value(row, "difficulty") and not _is_missing_scalar(_row_value(row, "difficulty")):
        try:
            difficulty = float(_row_value(row, "difficulty"))
        except (TypeError, ValueError):
            errors.append("difficulty_not_finite")
        else:
            if not math.isfinite(difficulty):
                errors.append("difficulty_not_finite")
    return errors


def _duplicate_path_counts(index_df: pd.DataFrame) -> dict[tuple[str, str], int]:
    counts: Counter[tuple[str, str]] = Counter()
    for row in index_df.itertuples(index=False):
        counts[_source_path_key(row)] += 1
    return dict(counts)


def _source_path_key(row: Mapping[str, object] | object) -> tuple[str, str]:
    return str(_row_value(row, "shard", default="")), str(_row_value(row, "beatmap_path", default=""))


def _top_examples(cache_df: pd.DataFrame, sort_column: str, limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or sort_column not in cache_df.columns:
        return []
    successful = cache_df[cache_df.get("conversion_ok", False).astype(bool)].copy()
    if successful.empty:
        return []
    top = successful.sort_values(sort_column, ascending=False).head(limit)
    columns = [
        "source_row_index",
        "shard",
        "beatmap_set_id",
        "beatmap_id",
        "beatmap_path",
        "title",
        "artist",
        "version",
        "difficulty",
        "max_abs_snap_error_ms",
        "p99_abs_snap_error_ms",
        "max_abs_snap_error_beats",
        "max_snap_error_kind",
        "max_snap_error_lane",
        "max_snap_error_time_ms",
        "max_snap_error_snapped_time_ms",
        "max_snap_error_ms",
        "max_snap_error_redline_bpm",
        "max_snap_error_bpm",
        "max_snap_error_beat_offset_numerator",
        "max_snap_error_beat_offset_denominator",
    ]
    return _records_for_columns(top, columns)


def _failure_examples(cache_df: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or "conversion_ok" not in cache_df.columns:
        return []
    failed = cache_df[~cache_df["conversion_ok"].astype(bool)].head(limit)
    columns = [
        "source_row_index",
        "shard",
        "beatmap_set_id",
        "beatmap_id",
        "beatmap_path",
        "error_type",
        "error_message",
        "index_invariant_errors",
    ]
    return _records_for_columns(failed, columns)


def _guard_failure_examples(cache_df: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or "ok" not in cache_df.columns:
        return []
    failed = cache_df[~cache_df["ok"].astype(bool)].head(limit)
    columns = [
        "source_row_index",
        "shard",
        "beatmap_set_id",
        "beatmap_id",
        "beatmap_path",
        "error_type",
        "index_invariant_errors",
        "event_count_mismatch_count",
        "negative_hold_duration_count",
        "snapped_hold_inversion_count",
        "snap_error_bound_violation_count",
        "duplicate_red_timing_offset_count",
        "conflicting_duplicate_red_timing_offset_count",
    ]
    return _records_for_columns(failed, columns)


def _records_for_columns(frame: pd.DataFrame, columns: Sequence[str]) -> list[dict[str, Any]]:
    selected = [column for column in columns if column in frame.columns]
    return [_normalize_json(record) for record in frame[selected].to_dict(orient="records")]


def _require_index_columns(index_df: pd.DataFrame, index_path: Path) -> None:
    missing = sorted(_REQUIRED_INDEX_COLUMNS.difference(index_df.columns))
    if missing:
        raise ValueError(f"{index_path} is missing required column(s): {missing}")


def _count_true(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    return int(frame[column].fillna(False).astype(bool).sum())


def _count_positive(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    return int((values > 0).sum())


def _sum_int(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    return int(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())


def _max_float(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns or frame.empty:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return 0.0
    return float(values.max())


def _percentile_float(frame: pd.DataFrame, column: str, percentile: float) -> float:
    if column not in frame.columns or frame.empty:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def _value_counts(frame: pd.DataFrame, column: str, *, exclude_empty: bool) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    values = frame[column].fillna("").astype(str)
    if exclude_empty:
        values = values[values != ""]
    return {str(key): int(value) for key, value in values.value_counts().sort_index().items()}


def _split_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    counts: Counter[str] = Counter()
    for value in frame[column].fillna("").astype(str):
        for part in value.split("|"):
            if part:
                counts[part] += 1
    return dict(sorted(counts.items()))


def _validate_diagnostic_subdivisions(subdivisions: Sequence[int]) -> tuple[int, ...]:
    if not subdivisions:
        raise ValueError("diagnostic_subdivisions must contain at least one subdivision")
    return tuple(sorted({_validate_positive_int(value, "diagnostic_subdivision") for value in subdivisions}))


def _parse_diagnostic_subdivisions(value: str) -> tuple[int, ...]:
    try:
        return _validate_diagnostic_subdivisions(tuple(int(part.strip()) for part in value.split(",") if part.strip()))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _required_text(value: object, *, field: str) -> str:
    if _is_missing_scalar(value):
        raise ValueError(f"{field} must be a non-empty relative path")
    text = str(value)
    if not text:
        raise ValueError(f"{field} must be a non-empty relative path")
    return text


def _has_row_value(row: Mapping[str, object] | object, name: str) -> bool:
    if isinstance(row, Mapping):
        return name in row
    return hasattr(row, name)


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
    return " ".join(shlex.quote(part) for part in ["python", "-m", "pulsefield_model.osu_core.beat_representation_audit", *args])


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit osu!mania beatmap to beat-based representation conversion.")
    parser.add_argument("--source-index-path", type=Path, default=DEFAULT_LE3_INDEX_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_LE3_AUDIT_CACHE_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_LE3_AUDIT_REPORT_PATH)
    parser.add_argument("--snap-denominator", type=int, default=DEFAULT_SNAP_DENOMINATOR)
    parser.add_argument(
        "--timing-canonicalization",
        choices=TIMING_CANONICALIZATION_CHOICES,
        default=DEFAULT_BEAT_REPRESENTATION_TIMING_CANONICALIZATION,
    )
    parser.add_argument("--expected-key-count", type=int, default=4)
    parser.add_argument("--any-key-count", action="store_true")
    parser.add_argument(
        "--diagnostic-subdivisions",
        type=_parse_diagnostic_subdivisions,
        default=DEFAULT_SNAP_DIAGNOSTIC_SUBDIVISIONS,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--top-example-limit", type=int, default=DEFAULT_TOP_EXAMPLE_LIMIT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    report = audit_beat_representation_index(
        source_index_path=args.source_index_path,
        dataset_root=args.dataset_root,
        cache_path=args.cache_path,
        report_path=args.report_path,
        snap_denominator=args.snap_denominator,
        timing_canonicalization=args.timing_canonicalization,
        expected_key_count=None if args.any_key_count else args.expected_key_count,
        diagnostic_subdivisions=args.diagnostic_subdivisions,
        limit=args.limit,
        progress_every=args.progress_every,
        top_example_limit=args.top_example_limit,
        command=_format_command(argv),
    )
    print(f"processed {report['processed_row_count']}/{report['source_row_count']} rows")
    print(f"ok {report['counts']['ok_count']}/{report['processed_row_count']} rows")
    print(f"conversion_success {report['counts']['conversion_success_count']}/{report['processed_row_count']} rows")
    print(f"cache_path {args.cache_path}")
    print(f"report_path {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
