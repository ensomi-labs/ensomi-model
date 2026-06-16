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
from typing import Any, Final, Mapping, Sequence

import numpy as np
import pandas as pd

from pulsefield_model.models.mapper.v2_1.tokenizer import (
    encode_full_chart_tokens,
    hitobjects_to_mapper_timepoints,
    mapper_chart_end_ms,
)
from pulsefield_model.models.mapper.v2_1.vocab import MapperV21Vocab
from pulsefield_model.osu_core.beat_representation import (
    DEFAULT_BEAT_REPRESENTATION_TIMING_CANONICALIZATION,
    DEFAULT_SNAP_DENOMINATOR,
    BeatEvent,
    BeatEventKind,
    hitobjects_to_beat_events,
)
from pulsefield_model.osu_core.beat_representation_audit import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_LE3_INDEX_PATH,
    resolve_index_beatmap_path,
)
from pulsefield_model.osu_core.hitobjects import ManiaHitObject, ManiaHitObjectKind, parse_mania_hit_objects
from pulsefield_model.osu_core.timing import RedTimingPoint, require_red_timing_points
from pulsefield_model.timing.canonicalization import TIMING_CANONICALIZATION_CHOICES


AUDIT_SCHEMA_VERSION: Final[int] = 1
DEFAULT_LE3_STRUCTURE_CACHE_PATH: Final[Path] = Path(
    "artifacts/audits/beat_structure/"
    "beat_structure_token_audit_4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3.parquet",
)
DEFAULT_LE3_STRUCTURE_REPORT_PATH: Final[Path] = Path(
    "artifacts/reports/audits/beat_structure/"
    "beat_structure_token_audit_4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3.json",
)
DEFAULT_TOP_EXAMPLE_LIMIT: Final[int] = 20
DEFAULT_RARE_MAX_COUNT: Final[int] = 1
DEFAULT_MOTIF_NGRAM: Final[int] = 4
SNAP_ERROR_BOUND_EPS_BEATS: Final[float] = 1e-9
SEQUENCE_LENGTH_BLOWUP_RATIO: Final[float] = 1.25
MOTIF_LIFT_MIN_SIGNAL: Final[float] = 0.01
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
_KIND_ORDER: Final[dict[BeatEventKind, int]] = {
    BeatEventKind.HOLD_START: 0,
    BeatEventKind.TAP: 1,
    BeatEventKind.HOLD_END: 2,
}
_ACTION_LABEL: Final[dict[BeatEventKind, str]] = {
    BeatEventKind.TAP: "tap",
    BeatEventKind.HOLD_START: "hold_start",
    BeatEventKind.HOLD_END: "hold_end",
}
_ACTION_SHORT: Final[dict[BeatEventKind, str]] = {
    BeatEventKind.TAP: "T",
    BeatEventKind.HOLD_START: "S",
    BeatEventKind.HOLD_END: "E",
}
_KNOWN_CANONICALIZATION_EXCEPTIONS: Final[frozenset[tuple[str, str, int]]] = frozenset(
    {
        (
            "0",
            "79839/Danny Baranowsky - The Battle of Lil' Slugger "
            "(Ch 1 Boss Extended Cut) (250bpm) (Staiain) [Insane].osu",
            222593,
        ),
    }
)


@dataclass(frozen=True)
class _BeatStructureAnalysis:
    event_records: list[dict[str, Any]]
    group_records: list[dict[str, Any]]
    reconstructed_event_tuples: list[tuple[str, int, int, int]]
    source_event_tuples: list[tuple[str, int, int, int]]
    field_counters: dict[str, Counter[str]]
    group_counters: dict[str, Counter[str]]
    joint_counter: Counter[str]
    v2_token_counter: Counter[str]
    v2_ts_counter: Counter[str]
    v2_lane_action_counter: Counter[str]
    v2_tokenization_ok: bool
    v2_error_type: str
    v2_error_message: str
    v2_token_count: int
    v2_time_shift_token_count: int
    v2_lane_action_token_count: int
    v2_special_token_count: int
    ln_pairing_mismatch_count: int
    snapped_ln_nonpositive_count: int
    snapped_ln_near_zero_count: int
    ln_cross_timing_segment_count: int
    same_lane_multi_action_group_count: int
    raw_time_collision_group_count: int
    snapped_order_inversion_count: int
    snap_error_bound_violation_count: int
    max_abs_snap_error_ms: float
    max_abs_snap_error_beats: float
    chord_group_count: int
    chord_event_count: int
    density_events_per_second: float
    chord_ratio: float
    ln_ratio: float
    motif_ngram_total_count: int
    motif_ngram_unique_count: int
    motif_ngram_repeated_count: int
    motif_recurrence_rate: float
    motif_control_recurrence_rate: float
    motif_recurrence_lift: float


def audit_beat_structure_token_index(
    *,
    source_index_path: str | Path = DEFAULT_LE3_INDEX_PATH,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    cache_path: str | Path = DEFAULT_LE3_STRUCTURE_CACHE_PATH,
    report_path: str | Path | None = DEFAULT_LE3_STRUCTURE_REPORT_PATH,
    snap_denominator: int = DEFAULT_SNAP_DENOMINATOR,
    timing_canonicalization: str = DEFAULT_BEAT_REPRESENTATION_TIMING_CANONICALIZATION,
    expected_key_count: int | None = 4,
    limit: int | None = None,
    progress_every: int = 0,
    rare_max_count: int = DEFAULT_RARE_MAX_COUNT,
    motif_ngram: int = DEFAULT_MOTIF_NGRAM,
    top_example_limit: int = DEFAULT_TOP_EXAMPLE_LIMIT,
    command: str | None = None,
) -> dict[str, Any]:
    """Audit a factorized beat-structure chart representation against v2.1 tokens."""

    started_at = time.perf_counter()
    source_index_path = Path(source_index_path)
    dataset_root = Path(dataset_root)
    cache_path = Path(cache_path)
    report_path = None if report_path is None else Path(report_path)
    snap_denominator = _validate_positive_int(snap_denominator, "snap_denominator")
    if timing_canonicalization not in TIMING_CANONICALIZATION_CHOICES:
        choices = ", ".join(TIMING_CANONICALIZATION_CHOICES)
        raise ValueError(f"timing_canonicalization must be one of {choices}, got {timing_canonicalization!r}")
    if expected_key_count is not None:
        expected_key_count = _validate_positive_int(expected_key_count, "expected_key_count")
    if limit is not None and int(limit) < 0:
        raise ValueError(f"limit must be non-negative, got {limit!r}")
    progress_every = _validate_nonnegative_int(progress_every, "progress_every")
    rare_max_count = _validate_nonnegative_int(rare_max_count, "rare_max_count")
    motif_ngram = _validate_positive_int(motif_ngram, "motif_ngram")
    top_example_limit = _validate_nonnegative_int(top_example_limit, "top_example_limit")

    source_df = pd.read_parquet(source_index_path)
    _require_index_columns(source_df, source_index_path)
    total_rows = len(source_df)
    max_rows = total_rows if limit is None else min(int(limit), total_rows)
    rows: list[dict[str, Any]] = []
    vocab = MapperV21Vocab()

    for source_row_index, row in enumerate(source_df.itertuples(index=False), start=0):
        if source_row_index >= max_rows:
            break
        rows.append(
            audit_beat_structure_token_row(
                row,
                source_row_index=source_row_index,
                dataset_root=dataset_root,
                snap_denominator=snap_denominator,
                timing_canonicalization=timing_canonicalization,
                expected_key_count=expected_key_count,
                rare_max_count=rare_max_count,
                motif_ngram=motif_ngram,
                vocab=vocab,
            )
        )
        if progress_every > 0 and (source_row_index + 1) % progress_every == 0:
            print(f"processed {source_row_index + 1}/{max_rows} maps", file=sys.stderr)

    cache_df = pd.DataFrame.from_records(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_cache_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    cache_df.to_parquet(tmp_cache_path, index=False)
    tmp_cache_path.replace(cache_path)

    report = build_beat_structure_token_audit_report(
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
        rare_max_count=rare_max_count,
        motif_ngram=motif_ngram,
        limit=limit,
        top_example_limit=top_example_limit,
        command=command,
    )
    if report_path is not None:
        _write_json(report_path, report)
    return report


def audit_beat_structure_token_row(
    row: Mapping[str, object] | object,
    *,
    source_row_index: int,
    dataset_root: Path,
    snap_denominator: int,
    timing_canonicalization: str,
    expected_key_count: int | None,
    rare_max_count: int = DEFAULT_RARE_MAX_COUNT,
    motif_ngram: int = DEFAULT_MOTIF_NGRAM,
    vocab: MapperV21Vocab | None = None,
) -> dict[str, Any]:
    base = _base_audit_row(row, source_row_index=source_row_index)
    vocab = MapperV21Vocab() if vocab is None else vocab
    try:
        beatmap_path = resolve_index_beatmap_path(dataset_root, _row_value(row, "shard"), _row_value(row, "beatmap_path"))
        base["resolved_beatmap_path"] = beatmap_path.as_posix()
        if not beatmap_path.is_file():
            raise FileNotFoundError(f"beatmap_path does not exist: {beatmap_path}")
    except Exception as exc:  # noqa: BLE001 - audit records row-local path failures.
        base.update(_failure_fields("path_error", str(exc)))
        return _finalize_row(base)

    try:
        timing_points = require_red_timing_points(beatmap_path)
        hitobjects = parse_mania_hit_objects(beatmap_path, expected_key_count=expected_key_count)
        events = hitobjects_to_beat_events(
            hitobjects,
            timing_points,
            snap_denominator=snap_denominator,
            timing_canonicalization=timing_canonicalization,
            include_diagnostics=True,
        )
        analysis = _analyze_structure(
            events,
            hitobjects=hitobjects,
            timing_points=timing_points,
            snap_denominator=snap_denominator,
            motif_ngram=motif_ngram,
            vocab=vocab,
        )
    except Exception as exc:  # noqa: BLE001 - full-index audit must not fail fast.
        base.update(_failure_fields(type(exc).__name__, str(exc)))
        return _finalize_row(base)

    base["path_ok"] = True
    base["conversion_ok"] = True
    _add_analysis_metrics(
        base,
        analysis,
        hitobjects=hitobjects,
        events=events,
        timing_points=timing_points,
        rare_max_count=rare_max_count,
    )
    return _finalize_row(base)


def build_beat_structure_token_audit_report(
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
    rare_max_count: int = DEFAULT_RARE_MAX_COUNT,
    motif_ngram: int = DEFAULT_MOTIF_NGRAM,
    limit: int | None = None,
    top_example_limit: int = DEFAULT_TOP_EXAMPLE_LIMIT,
    command: str | None = None,
) -> dict[str, Any]:
    cache_df = pd.read_parquet(cache) if isinstance(cache, (str, Path)) else cache.copy()
    if source_row_count is None:
        source_row_count = len(cache_df)
    processed_row_count = len(cache_df)
    total_events = _sum_int(cache_df, "event_count")
    total_records = _sum_int(cache_df, "beat_record_count")
    total_groups = _sum_int(cache_df, "beat_group_count")
    total_v2_tokens = _sum_int(cache_df, "v2_token_count")
    v2_ok_df = cache_df[cache_df.get("v2_tokenization_ok", False).astype(bool)] if "v2_tokenization_ok" in cache_df.columns else cache_df
    v2_comparable_event_count = _sum_int(v2_ok_df, "event_count")

    field_counters = {
        "beat_delta_units": _aggregate_json_counters(cache_df, "beat_delta_units_counter_json"),
        "bar_phase_units": _aggregate_json_counters(cache_df, "bar_phase_units_counter_json"),
        "beat_phase": _aggregate_json_counters(cache_df, "beat_phase_counter_json"),
        "subdivision": _aggregate_json_counters(cache_df, "subdivision_counter_json"),
        "lane": _aggregate_json_counters(cache_df, "lane_counter_json"),
        "action_type": _aggregate_json_counters(cache_df, "action_type_counter_json"),
        "group_lane_mask": _aggregate_json_counters(cache_df, "group_lane_mask_counter_json"),
        "group_action_signature": _aggregate_json_counters(cache_df, "group_action_signature_counter_json"),
        "ln_span_bucket": _aggregate_json_counters(cache_df, "ln_span_bucket_counter_json"),
    }
    group_counters = {
        "group_delta_units": _aggregate_json_counters(cache_df, "group_delta_units_counter_json"),
        "group_onset_count": _aggregate_json_counters(cache_df, "group_onset_count_counter_json"),
        "group_action_signature": _aggregate_json_counters(cache_df, "group_record_action_signature_counter_json"),
    }
    beat_joint_counter = _aggregate_json_counters(cache_df, "beat_joint_counter_json")
    v2_token_counter = _aggregate_json_counters(cache_df, "v2_token_counter_json")
    v2_ts_counter = _aggregate_json_counters(cache_df, "v2_ts_counter_json")
    v2_lane_action_counter = _aggregate_json_counters(cache_df, "v2_lane_action_counter_json")

    v2_token_entropy = _entropy_bits(v2_token_counter)
    beat_joint_entropy = _entropy_bits(beat_joint_counter)
    beat_field_entropies = {name: _entropy_bits(counter) for name, counter in field_counters.items()}
    beat_independent_bits_per_record = sum(beat_field_entropies.values())
    v2_bits_per_event = _bits_per_event(v2_token_entropy, total_v2_tokens, total_events)
    beat_joint_bits_per_event = _bits_per_event(beat_joint_entropy, total_records, total_events)
    beat_independent_bits_per_event = _bits_per_event(beat_independent_bits_per_record, total_records, total_events)
    sequence_length_ratio = float(total_records) / float(total_v2_tokens) if total_v2_tokens else 0.0
    motif_recurrence_rate = _weighted_rate(cache_df, "motif_ngram_repeated_count", "motif_ngram_total_count")
    motif_control_recurrence_rate = _weighted_column_average(cache_df, "motif_control_recurrence_rate", "motif_ngram_total_count")
    motif_recurrence_lift = motif_recurrence_rate - motif_control_recurrence_rate

    hard_gate_counts = {
        "event_count_mismatch_count": _sum_int(cache_df, "event_count_mismatch_count"),
        "event_order_mismatch_count": _sum_int(cache_df, "event_order_mismatch_count"),
        "lane_identity_mismatch_count": _sum_int(cache_df, "lane_identity_mismatch_count"),
        "action_type_mismatch_count": _sum_int(cache_df, "action_type_mismatch_count"),
        "chord_grouping_mismatch_count": _sum_int(cache_df, "chord_grouping_mismatch_count"),
        "ln_pairing_mismatch_count": _sum_int(cache_df, "ln_pairing_mismatch_count"),
        "snap_error_bound_violation_count": _sum_int(cache_df, "snap_error_bound_violation_count"),
        "unflagged_snapped_ln_nonpositive_count": _sum_int(cache_df, "unflagged_snapped_ln_nonpositive_count"),
    }
    hard_gate_alias_counts = {
        "lane_mismatch_count": hard_gate_counts["lane_identity_mismatch_count"],
        "action_mismatch_count": hard_gate_counts["action_type_mismatch_count"],
        "chord_group_mismatch_count": hard_gate_counts["chord_grouping_mismatch_count"],
        "ln_pair_mismatch_count": hard_gate_counts["ln_pairing_mismatch_count"],
        "snap_bound_violation_count": hard_gate_counts["snap_error_bound_violation_count"],
        "collapsed_ln_unflagged_count": hard_gate_counts["unflagged_snapped_ln_nonpositive_count"],
    }
    hard_gate_pass = (
        _count_true(cache_df, "path_ok") == processed_row_count
        and _count_true(cache_df, "conversion_ok") == processed_row_count
        and all(value == 0 for value in hard_gate_counts.values())
    )
    pass_criteria = {
        "sequence_length_ratio": sequence_length_ratio,
        "sequence_length_pass": sequence_length_ratio <= SEQUENCE_LENGTH_BLOWUP_RATIO if total_v2_tokens else False,
        "beat_joint_bits_per_event": beat_joint_bits_per_event,
        "beat_independent_bits_per_event": beat_independent_bits_per_event,
        "v2_token_bits_per_event": v2_bits_per_event,
        "joint_bits_lte_v2_pass": beat_joint_bits_per_event <= v2_bits_per_event if total_events else False,
        "motif_recurrence_rate": motif_recurrence_rate,
        "motif_control_recurrence_rate": motif_control_recurrence_rate,
        "motif_recurrence_lift": motif_recurrence_lift,
        "motif_lift_pass": motif_recurrence_lift >= MOTIF_LIFT_MIN_SIGNAL,
        "v2_tokenization_complete_pass": _count_true(cache_df, "v2_tokenization_ok") == processed_row_count,
        "known_canonicalization_exception_count": _sum_int(cache_df, "known_canonicalization_exception_count"),
        "hard_gate_pass": hard_gate_pass,
    }
    pass_criteria["overall_research_signal_pass"] = bool(
        hard_gate_pass
        and pass_criteria["sequence_length_pass"]
        and (
            pass_criteria["joint_bits_lte_v2_pass"]
            or beat_independent_bits_per_event <= v2_bits_per_event
        )
        and pass_criteria["motif_lift_pass"]
    )
    v2_total_bits = v2_token_entropy * float(total_v2_tokens)
    beat_joint_total_bits = beat_joint_entropy * float(total_records)
    beat_independent_total_bits = beat_independent_bits_per_record * float(total_records)

    report = {
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
            "rare_max_count": rare_max_count,
            "motif_ngram": motif_ngram,
            "sequence_length_blowup_ratio": SEQUENCE_LENGTH_BLOWUP_RATIO,
            "motif_lift_min_signal": MOTIF_LIFT_MIN_SIGNAL,
            "snap_error_bound_eps_beats": SNAP_ERROR_BOUND_EPS_BEATS,
            "known_canonicalization_exceptions": [
                {"shard": shard, "beatmap_path": path, "beatmap_id": beatmap_id}
                for shard, path, beatmap_id in sorted(_KNOWN_CANONICALIZATION_EXCEPTIONS)
            ],
        },
        "counts": {
            "ok_count": _count_true(cache_df, "ok"),
            "not_ok_count": processed_row_count - _count_true(cache_df, "ok"),
            "path_failure_count": processed_row_count - _count_true(cache_df, "path_ok"),
            "conversion_success_count": _count_true(cache_df, "conversion_ok"),
            "conversion_failure_count": processed_row_count - _count_true(cache_df, "conversion_ok"),
            "hard_gate_ok_count": _count_true(cache_df, "hard_gate_ok"),
            "hard_gate_failure_count": processed_row_count - _count_true(cache_df, "hard_gate_ok"),
            "v2_tokenization_success_count": _count_true(cache_df, "v2_tokenization_ok"),
            "v2_tokenization_failure_count": processed_row_count - _count_true(cache_df, "v2_tokenization_ok"),
        },
        "event_totals": {
            "hitobject_count": _sum_int(cache_df, "hitobject_count"),
            "tap_count": _sum_int(cache_df, "tap_count"),
            "hold_count": _sum_int(cache_df, "hold_count"),
            "event_count": total_events,
            "beat_record_count": total_records,
            "beat_group_count": total_groups,
            "v2_token_count": total_v2_tokens,
            "v2_time_shift_token_count": _sum_int(cache_df, "v2_time_shift_token_count"),
            "v2_lane_action_token_count": _sum_int(cache_df, "v2_lane_action_token_count"),
            "chord_group_count": _sum_int(cache_df, "chord_group_count"),
            "chord_event_count": _sum_int(cache_df, "chord_event_count"),
        },
        "hard_gates": {
            "pass": hard_gate_pass,
            "status": "PASS" if hard_gate_pass else "FAIL",
            **hard_gate_counts,
            **hard_gate_alias_counts,
            "known_canonicalization_exception_count": _sum_int(cache_df, "known_canonicalization_exception_count"),
            "known_5ms_ln_canonicalization_exception_count": _sum_int(cache_df, "known_5ms_ln_canonicalization_exception_count"),
            "snapped_ln_nonpositive_count": _sum_int(cache_df, "snapped_ln_nonpositive_count"),
            "snapped_ln_near_zero_count": _sum_int(cache_df, "snapped_ln_near_zero_count"),
        },
        "pass_criteria": pass_criteria,
        "fair_comparison": {
            "current_v2_1": {
                "event_count": total_events,
                "comparable_event_count": v2_comparable_event_count,
                "encoded_item_count": total_v2_tokens,
                "entropy_per_item_bits": v2_token_entropy,
                "entropy_bits": v2_total_bits,
                "bits_per_event": float(v2_total_bits) / float(total_events) if total_events else 0.0,
                "bits_per_comparable_event": float(v2_total_bits) / float(v2_comparable_event_count) if v2_comparable_event_count else 0.0,
                "rare_rate": _rare_rate(v2_token_counter, rare_max_count=rare_max_count),
                "tokenization_failure_count": processed_row_count - _count_true(cache_df, "v2_tokenization_ok"),
                "special_token_policy": "BOS/EOS included; PAD absent",
            },
            "beat_structure": {
                "event_count": total_events,
                "encoded_item_count": total_records,
                "joint_entropy_per_record_bits": beat_joint_entropy,
                "independent_entropy_per_record_bits": beat_independent_bits_per_record,
                "entropy_bits": beat_joint_total_bits,
                "independent_entropy_bits": beat_independent_total_bits,
                "bits_per_event": float(beat_joint_total_bits) / float(total_events) if total_events else 0.0,
                "independent_bits_per_event": float(beat_independent_total_bits) / float(total_events) if total_events else 0.0,
                "rare_joint_rate": _rare_rate(beat_joint_counter, rare_max_count=rare_max_count),
                "record_policy": "one factorized event record per audited beat event; chord fields repeated per event",
            },
        },
        "entropy": {
            "v2_token_entropy_bits": v2_token_entropy,
            "beat_joint_entropy_bits": beat_joint_entropy,
            "beat_field_entropy_bits": beat_field_entropies,
            "beat_independent_bits_per_record": beat_independent_bits_per_record,
            "v2_token_bits_per_event": v2_bits_per_event,
            "beat_joint_bits_per_event": beat_joint_bits_per_event,
            "beat_independent_bits_per_event": beat_independent_bits_per_event,
        },
        "rare_rates": {
            "rare_max_count": rare_max_count,
            "v2_token_rare_rate": _rare_rate(v2_token_counter, rare_max_count=rare_max_count),
            "v2_ts_rare_rate": _rare_rate(v2_ts_counter, rare_max_count=rare_max_count),
            "v2_lane_action_rare_rate": _rare_rate(v2_lane_action_counter, rare_max_count=rare_max_count),
            "beat_joint_rare_rate": _rare_rate(beat_joint_counter, rare_max_count=rare_max_count),
            "beat_field_rare_rates": {
                name: _rare_rate(counter, rare_max_count=rare_max_count)
                for name, counter in field_counters.items()
            },
        },
        "cardinality": {
            "v2_token_unique_count": len(v2_token_counter),
            "v2_ts_unique_count": len(v2_ts_counter),
            "v2_lane_action_unique_count": len(v2_lane_action_counter),
            "beat_joint_unique_count": len(beat_joint_counter),
            "beat_field_unique_counts": {name: len(counter) for name, counter in field_counters.items()},
            "beat_group_field_unique_counts": {name: len(counter) for name, counter in group_counters.items()},
        },
        "motif": {
            "ngram": motif_ngram,
            "total_count": _sum_int(cache_df, "motif_ngram_total_count"),
            "unique_count_sum_by_map": _sum_int(cache_df, "motif_ngram_unique_count"),
            "repeated_count": _sum_int(cache_df, "motif_ngram_repeated_count"),
            "recurrence_rate": motif_recurrence_rate,
            "control_recurrence_rate": motif_control_recurrence_rate,
            "recurrence_lift": motif_recurrence_lift,
            "density_bin_summary": _density_bin_summary(cache_df),
        },
        "motif_recurrence": {
            "density_controlled": {
                "bucket_count": len(_density_bin_summary(cache_df)),
                "motif_count": _sum_int(cache_df, "motif_ngram_total_count"),
                "recurrence_rate": motif_recurrence_rate,
                "control_recurrence_rate": motif_control_recurrence_rate,
                "lift": motif_recurrence_lift,
                "lift_mean": motif_recurrence_lift,
                "buckets": _density_bin_summary(cache_df),
            }
        },
        "distributions": {
            "v2_token_top": _top_counter(v2_token_counter),
            "v2_ts_top": _top_counter(v2_ts_counter),
            "v2_lane_action_top": _top_counter(v2_lane_action_counter),
            "beat_joint_top": _top_counter(beat_joint_counter),
            "beat_fields_top": {name: _top_counter(counter) for name, counter in field_counters.items()},
            "beat_group_fields_top": {name: _top_counter(counter) for name, counter in group_counters.items()},
        },
        "failure_counts_by_error_type": _value_counts(cache_df, "error_type", exclude_empty=True),
        "v2_failure_counts_by_error_type": _value_counts(cache_df, "v2_error_type", exclude_empty=True),
        "v2_failure_examples": _v2_failure_examples(cache_df, top_example_limit),
        "hard_gate_failure_examples": _hard_gate_failure_examples(cache_df, top_example_limit),
        "worst_joint_rare_examples": _top_examples(cache_df, "beat_joint_rare_rate", top_example_limit),
        "worst_motif_control_examples": _top_examples(cache_df, "motif_recurrence_lift", top_example_limit, ascending=True),
        "ln_outlier_examples": _top_examples(cache_df, "max_ln_span_units", top_example_limit),
        "elapsed_s": elapsed_s,
        "code_commit": _git_stdout("rev-parse", "HEAD"),
        "code_dirty": bool(_git_stdout("status", "--porcelain")),
        "command": command,
    }
    return _normalize_json(report)


def _base_audit_row(row: Mapping[str, object] | object, *, source_row_index: int) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "source_row_index": int(source_row_index),
        "resolved_beatmap_path": "",
        "path_ok": False,
        "conversion_ok": False,
        "hard_gate_ok": False,
        "ok": False,
        "error_type": "",
        "error_message": "",
        "v2_tokenization_ok": False,
        "v2_error_type": "",
        "v2_error_message": "",
        "hitobject_count": 0,
        "tap_count": 0,
        "hold_count": 0,
        "event_count": 0,
        "expected_event_count": 0,
        "beat_record_count": 0,
        "beat_group_count": 0,
        "events_per_record": 1.0,
        "records_per_v2_token": 0.0,
        "v2_token_count": 0,
        "v2_time_shift_token_count": 0,
        "v2_lane_action_token_count": 0,
        "v2_special_token_count": 0,
        "red_timing_count": 0,
        "timing_segment_count": 0,
        "event_count_mismatch_count": 0,
        "event_order_mismatch_count": 0,
        "lane_identity_mismatch_count": 0,
        "action_type_mismatch_count": 0,
        "action_mismatch_count": 0,
        "chord_grouping_mismatch_count": 0,
        "chord_group_mismatch_count": 0,
        "ln_pairing_mismatch_count": 0,
        "ln_pair_mismatch_count": 0,
        "snap_error_bound_violation_count": 0,
        "snap_bound_violation_count": 0,
        "snapped_ln_nonpositive_count": 0,
        "snapped_ln_near_zero_count": 0,
        "known_canonicalization_exception_count": 0,
        "known_5ms_ln_canonicalization_exception_count": 0,
        "unflagged_snapped_ln_nonpositive_count": 0,
        "collapsed_ln_unflagged_count": 0,
        "ln_cross_timing_segment_count": 0,
        "same_lane_multi_action_group_count": 0,
        "raw_time_collision_group_count": 0,
        "snapped_order_inversion_count": 0,
        "max_abs_snap_error_ms": 0.0,
        "max_abs_snap_error_beats": 0.0,
        "chord_group_count": 0,
        "chord_event_count": 0,
        "density_events_per_second": 0.0,
        "chord_ratio": 0.0,
        "ln_ratio": 0.0,
        "motif_ngram_total_count": 0,
        "motif_ngram_unique_count": 0,
        "motif_ngram_repeated_count": 0,
        "motif_recurrence_rate": 0.0,
        "motif_control_recurrence_rate": 0.0,
        "motif_recurrence_lift": 0.0,
        "density_bin": "empty",
        "beat_local_joint_entropy_bits": 0.0,
        "v2_local_token_entropy_bits": 0.0,
        "beat_local_joint_bits_per_event": 0.0,
        "v2_local_bits_per_event": 0.0,
        "beat_joint_unique_count": 0,
        "beat_joint_rare_rate": 0.0,
        "max_ln_span_units": 0,
        "beat_delta_units_counter_json": "{}",
        "bar_phase_units_counter_json": "{}",
        "beat_phase_counter_json": "{}",
        "subdivision_counter_json": "{}",
        "lane_counter_json": "{}",
        "action_type_counter_json": "{}",
        "group_lane_mask_counter_json": "{}",
        "group_action_signature_counter_json": "{}",
        "ln_span_bucket_counter_json": "{}",
        "group_delta_units_counter_json": "{}",
        "group_onset_count_counter_json": "{}",
        "group_record_action_signature_counter_json": "{}",
        "beat_joint_counter_json": "{}",
        "v2_token_counter_json": "{}",
        "v2_ts_counter_json": "{}",
        "v2_lane_action_counter_json": "{}",
    }
    for column in _OPTIONAL_IDENTITY_COLUMNS:
        base[column] = _row_value(row, column, default=None)
    base["shard"] = _row_value(row, "shard", default="")
    base["beatmap_path"] = _row_value(row, "beatmap_path", default="")
    return {key: _cache_scalar(value) for key, value in base.items()}


def _analyze_structure(
    events: Sequence[BeatEvent],
    *,
    hitobjects: Sequence[ManiaHitObject],
    timing_points: Sequence[RedTimingPoint],
    snap_denominator: int,
    motif_ngram: int,
    vocab: MapperV21Vocab,
) -> _BeatStructureAnalysis:
    event_list = list(events)
    segment_ids = _segment_ids(event_list)
    source_event_tuples = _event_tuples(event_list, segment_ids)
    ln_pairs = _ln_pairs(event_list, hitobjects, segment_ids, snap_denominator=snap_denominator)
    group_records, group_by_key = _group_records(event_list, segment_ids, snap_denominator=snap_denominator)
    event_records = _event_records(
        event_list,
        segment_ids,
        group_by_key=group_by_key,
        ln_span_units_by_start_index=ln_pairs["span_units_by_start_index"],
        snap_denominator=snap_denominator,
    )
    reconstructed_event_tuples = [
        (str(record["action_type"]), int(record["lane"]), int(record["segment_id"]), int(record["beat_offset_numerator"]))
        for record in event_records
    ]
    field_counters = _field_counters(event_records)
    group_counters = _group_counters(group_records)
    joint_counter = Counter(str(record["joint_key"]) for record in event_records)
    v2_metrics = _v2_metrics(hitobjects, vocab=vocab)
    motif_metrics = _motif_metrics(group_records, motif_ngram=motif_ngram)
    snap_errors_beats = [
        (
            float(event.diagnostics.abs_snap_error_beats)
            if event.diagnostics is not None
            else abs(float(event.snap_error_ms)) / float(event.beat_length_ms)
        )
        for event in event_list
    ]
    snap_error_bound_beats = 0.5 / float(snap_denominator) + SNAP_ERROR_BOUND_EPS_BEATS
    chart_duration_s = _chart_duration_seconds(event_list)
    onset_event_count = sum(1 for event in event_list if event.kind in {BeatEventKind.TAP, BeatEventKind.HOLD_START})
    chord_event_count = sum(int(record["event_count"]) for record in group_records if int(record["onset_count"]) >= 2)
    chord_group_count = sum(1 for record in group_records if int(record["onset_count"]) >= 2)

    return _BeatStructureAnalysis(
        event_records=event_records,
        group_records=group_records,
        reconstructed_event_tuples=reconstructed_event_tuples,
        source_event_tuples=source_event_tuples,
        field_counters=field_counters,
        group_counters=group_counters,
        joint_counter=joint_counter,
        v2_token_counter=v2_metrics["token_counter"],
        v2_ts_counter=v2_metrics["ts_counter"],
        v2_lane_action_counter=v2_metrics["lane_action_counter"],
        v2_tokenization_ok=bool(v2_metrics["ok"]),
        v2_error_type=str(v2_metrics["error_type"]),
        v2_error_message=str(v2_metrics["error_message"]),
        v2_token_count=int(v2_metrics["token_count"]),
        v2_time_shift_token_count=int(v2_metrics["time_shift_token_count"]),
        v2_lane_action_token_count=int(v2_metrics["lane_action_token_count"]),
        v2_special_token_count=int(v2_metrics["special_token_count"]),
        ln_pairing_mismatch_count=int(ln_pairs["mismatch_count"]),
        snapped_ln_nonpositive_count=int(ln_pairs["nonpositive_count"]),
        snapped_ln_near_zero_count=int(ln_pairs["near_zero_count"]),
        ln_cross_timing_segment_count=int(ln_pairs["cross_timing_segment_count"]),
        same_lane_multi_action_group_count=sum(int(record["same_lane_multi_action_count"] > 0) for record in group_records),
        raw_time_collision_group_count=sum(int(record["raw_time_count"] > 1) for record in group_records),
        snapped_order_inversion_count=_snapped_order_inversion_count(event_list),
        snap_error_bound_violation_count=sum(error > snap_error_bound_beats for error in snap_errors_beats),
        max_abs_snap_error_ms=max((abs(float(event.snap_error_ms)) for event in event_list), default=0.0),
        max_abs_snap_error_beats=max(snap_errors_beats, default=0.0),
        chord_group_count=chord_group_count,
        chord_event_count=chord_event_count,
        density_events_per_second=float(onset_event_count) / chart_duration_s if chart_duration_s > 0 else 0.0,
        chord_ratio=float(chord_group_count) / float(len(group_records)) if group_records else 0.0,
        ln_ratio=float(sum(1 for hitobject in hitobjects if hitobject.kind == ManiaHitObjectKind.HOLD)) / float(len(hitobjects)) if hitobjects else 0.0,
        motif_ngram_total_count=int(motif_metrics["total_count"]),
        motif_ngram_unique_count=int(motif_metrics["unique_count"]),
        motif_ngram_repeated_count=int(motif_metrics["repeated_count"]),
        motif_recurrence_rate=float(motif_metrics["recurrence_rate"]),
        motif_control_recurrence_rate=float(motif_metrics["control_recurrence_rate"]),
        motif_recurrence_lift=float(motif_metrics["recurrence_lift"]),
    )


def _add_analysis_metrics(
    base: dict[str, Any],
    analysis: _BeatStructureAnalysis,
    *,
    hitobjects: Sequence[ManiaHitObject],
    events: Sequence[BeatEvent],
    timing_points: Sequence[RedTimingPoint],
    rare_max_count: int,
) -> None:
    event_count = len(events)
    expected_event_count = sum(1 if item.kind == ManiaHitObjectKind.TAP else 2 for item in hitobjects)
    action_mismatches = 0
    lane_mismatches = 0
    order_mismatches = 0
    for expected, actual in zip(analysis.source_event_tuples, analysis.reconstructed_event_tuples, strict=False):
        if expected != actual:
            order_mismatches += 1
            if expected[0] != actual[0]:
                action_mismatches += 1
            if expected[1] != actual[1]:
                lane_mismatches += 1
    event_count_mismatch = 0 if event_count == expected_event_count == len(analysis.reconstructed_event_tuples) else 1
    if len(analysis.source_event_tuples) != len(analysis.reconstructed_event_tuples):
        event_count_mismatch = 1
    chord_grouping_mismatch = _chord_grouping_mismatch_count(analysis.group_records, analysis.event_records)
    known_exception_allowed = 1 if _is_known_canonicalization_exception(base) else 0
    known_exception_count = min(known_exception_allowed, analysis.snapped_ln_nonpositive_count)
    unflagged_nonpositive = max(0, analysis.snapped_ln_nonpositive_count - known_exception_count)
    hard_gate_ok = (
        event_count_mismatch == 0
        and order_mismatches == 0
        and lane_mismatches == 0
        and action_mismatches == 0
        and chord_grouping_mismatch == 0
        and analysis.ln_pairing_mismatch_count == 0
        and analysis.snap_error_bound_violation_count == 0
        and unflagged_nonpositive == 0
    )
    total_records = len(analysis.event_records)
    local_joint_entropy = _entropy_bits(analysis.joint_counter)
    local_v2_entropy = _entropy_bits(analysis.v2_token_counter)
    max_ln_span_units = max(
        (int(key) for key in analysis.field_counters["ln_span_units_raw"] if key not in {"none", "end"} and str(key).lstrip("-").isdigit()),
        default=0,
    )
    base.update(
        {
            "hard_gate_ok": hard_gate_ok,
            "hitobject_count": len(hitobjects),
            "tap_count": sum(1 for item in hitobjects if item.kind == ManiaHitObjectKind.TAP),
            "hold_count": sum(1 for item in hitobjects if item.kind == ManiaHitObjectKind.HOLD),
            "event_count": event_count,
            "expected_event_count": expected_event_count,
            "beat_record_count": total_records,
            "beat_group_count": len(analysis.group_records),
            "events_per_record": float(event_count) / float(total_records) if total_records else 0.0,
            "records_per_v2_token": float(total_records) / float(analysis.v2_token_count) if analysis.v2_token_count else 0.0,
            "red_timing_count": len(timing_points),
            "timing_segment_count": len({(event.redline_offset_ms, event.redline_beat_length_ms, event.beat_length_ms) for event in events}),
            "v2_tokenization_ok": analysis.v2_tokenization_ok,
            "v2_error_type": analysis.v2_error_type,
            "v2_error_message": _short_error(analysis.v2_error_message),
            "v2_token_count": analysis.v2_token_count,
            "v2_time_shift_token_count": analysis.v2_time_shift_token_count,
            "v2_lane_action_token_count": analysis.v2_lane_action_token_count,
            "v2_special_token_count": analysis.v2_special_token_count,
            "event_count_mismatch_count": event_count_mismatch,
            "event_order_mismatch_count": order_mismatches,
            "lane_identity_mismatch_count": lane_mismatches,
            "lane_mismatch_count": lane_mismatches,
            "action_type_mismatch_count": action_mismatches,
            "action_mismatch_count": action_mismatches,
            "chord_grouping_mismatch_count": chord_grouping_mismatch,
            "chord_group_mismatch_count": chord_grouping_mismatch,
            "ln_pairing_mismatch_count": analysis.ln_pairing_mismatch_count,
            "ln_pair_mismatch_count": analysis.ln_pairing_mismatch_count,
            "snap_error_bound_violation_count": analysis.snap_error_bound_violation_count,
            "snap_bound_violation_count": analysis.snap_error_bound_violation_count,
            "snapped_ln_nonpositive_count": analysis.snapped_ln_nonpositive_count,
            "snapped_ln_near_zero_count": analysis.snapped_ln_near_zero_count,
            "known_canonicalization_exception_count": known_exception_count,
            "known_5ms_ln_canonicalization_exception_count": known_exception_count,
            "unflagged_snapped_ln_nonpositive_count": unflagged_nonpositive,
            "collapsed_ln_unflagged_count": unflagged_nonpositive,
            "ln_cross_timing_segment_count": analysis.ln_cross_timing_segment_count,
            "same_lane_multi_action_group_count": analysis.same_lane_multi_action_group_count,
            "raw_time_collision_group_count": analysis.raw_time_collision_group_count,
            "snapped_order_inversion_count": analysis.snapped_order_inversion_count,
            "max_abs_snap_error_ms": analysis.max_abs_snap_error_ms,
            "max_abs_snap_error_beats": analysis.max_abs_snap_error_beats,
            "chord_group_count": analysis.chord_group_count,
            "chord_event_count": analysis.chord_event_count,
            "density_events_per_second": analysis.density_events_per_second,
            "chord_ratio": analysis.chord_ratio,
            "ln_ratio": analysis.ln_ratio,
            "density_bin": _density_bin(analysis.density_events_per_second, analysis.chord_ratio, analysis.ln_ratio),
            "motif_ngram_total_count": analysis.motif_ngram_total_count,
            "motif_ngram_unique_count": analysis.motif_ngram_unique_count,
            "motif_ngram_repeated_count": analysis.motif_ngram_repeated_count,
            "motif_recurrence_rate": analysis.motif_recurrence_rate,
            "motif_control_recurrence_rate": analysis.motif_control_recurrence_rate,
            "motif_recurrence_lift": analysis.motif_recurrence_lift,
            "beat_local_joint_entropy_bits": local_joint_entropy,
            "v2_local_token_entropy_bits": local_v2_entropy,
            "beat_local_joint_bits_per_event": _bits_per_event(local_joint_entropy, total_records, event_count),
            "v2_local_bits_per_event": _bits_per_event(local_v2_entropy, analysis.v2_token_count, event_count),
            "beat_joint_unique_count": len(analysis.joint_counter),
            "beat_joint_rare_rate": _rare_rate(analysis.joint_counter, rare_max_count=rare_max_count),
            "max_ln_span_units": max_ln_span_units,
            "beat_delta_units_counter_json": _counter_json(analysis.field_counters["beat_delta_units"]),
            "bar_phase_units_counter_json": _counter_json(analysis.field_counters["bar_phase_units"]),
            "beat_phase_counter_json": _counter_json(analysis.field_counters["beat_phase"]),
            "subdivision_counter_json": _counter_json(analysis.field_counters["subdivision"]),
            "lane_counter_json": _counter_json(analysis.field_counters["lane"]),
            "action_type_counter_json": _counter_json(analysis.field_counters["action_type"]),
            "group_lane_mask_counter_json": _counter_json(analysis.field_counters["group_lane_mask"]),
            "group_action_signature_counter_json": _counter_json(analysis.field_counters["group_action_signature"]),
            "ln_span_bucket_counter_json": _counter_json(analysis.field_counters["ln_span_bucket"]),
            "group_delta_units_counter_json": _counter_json(analysis.group_counters["group_delta_units"]),
            "group_onset_count_counter_json": _counter_json(analysis.group_counters["onset_count"]),
            "group_record_action_signature_counter_json": _counter_json(analysis.group_counters["action_signature"]),
            "beat_joint_counter_json": _counter_json(analysis.joint_counter),
            "v2_token_counter_json": _counter_json(analysis.v2_token_counter),
            "v2_ts_counter_json": _counter_json(analysis.v2_ts_counter),
            "v2_lane_action_counter_json": _counter_json(analysis.v2_lane_action_counter),
        }
    )


def _finalize_row(base: dict[str, Any]) -> dict[str, Any]:
    base["ok"] = bool(base["path_ok"]) and bool(base["conversion_ok"]) and bool(base["hard_gate_ok"])
    return {key: _cache_scalar(value) for key, value in base.items()}


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


def _group_records(
    events: Sequence[BeatEvent],
    segment_ids: Sequence[int],
    *,
    snap_denominator: int,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], dict[str, Any]]]:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, event in enumerate(events):
        groups[(int(segment_ids[index]), int(event.beat_offset_numerator))].append(index)

    ordered_keys = sorted(
        groups,
        key=lambda key: (
            min(float(events[index].snapped_time_ms) for index in groups[key]),
            key[0],
            key[1],
        ),
    )
    records: list[dict[str, Any]] = []
    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    previous_key: tuple[int, int] | None = None
    for group_index, key in enumerate(ordered_keys):
        event_indexes = groups[key]
        lane_actions: dict[int, list[BeatEventKind]] = defaultdict(list)
        raw_times = {float(events[index].time_ms) for index in event_indexes}
        for event_index in event_indexes:
            event = events[event_index]
            lane_actions[int(event.lane)].append(event.kind)
        tap_mask = _mask_for(lane_actions, BeatEventKind.TAP)
        hold_start_mask = _mask_for(lane_actions, BeatEventKind.HOLD_START)
        hold_end_mask = _mask_for(lane_actions, BeatEventKind.HOLD_END)
        lane_mask = tap_mask | hold_start_mask | hold_end_mask
        onset_count = sum(
            1
            for event_index in event_indexes
            if events[event_index].kind in {BeatEventKind.TAP, BeatEventKind.HOLD_START}
        )
        same_lane_multi_action_count = sum(1 for actions in lane_actions.values() if len(actions) > 1)
        delta_units = 0 if previous_key is None else _group_delta_units(events, groups[previous_key][0], event_indexes[0], snap_denominator)
        first_event = events[event_indexes[0]]
        meter = _meter_for_event(first_event)
        bar_phase_units = int(first_event.beat_offset_numerator) % (meter * snap_denominator)
        record = {
            "group_index": group_index,
            "segment_id": key[0],
            "beat_offset_numerator": key[1],
            "group_delta_units": int(delta_units),
            "bar_phase_units": bar_phase_units,
            "beat_phase": (int(first_event.beat_offset_numerator) // snap_denominator) % meter,
            "subdivision": int(first_event.beat_offset_numerator) % snap_denominator,
            "tap_mask": tap_mask,
            "hold_start_mask": hold_start_mask,
            "hold_end_mask": hold_end_mask,
            "lane_mask": lane_mask,
            "action_signature": _group_action_signature(lane_actions),
            "event_count": len(event_indexes),
            "onset_count": onset_count,
            "same_lane_multi_action_count": same_lane_multi_action_count,
            "raw_time_count": len(raw_times),
        }
        records.append(record)
        by_key[key] = record
        previous_key = key
    return records, by_key


def _event_records(
    events: Sequence[BeatEvent],
    segment_ids: Sequence[int],
    *,
    group_by_key: Mapping[tuple[int, int], Mapping[str, Any]],
    ln_span_units_by_start_index: Mapping[int, int],
    snap_denominator: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    previous_event: BeatEvent | None = None
    previous_segment_id: int | None = None
    previous_beat_offset_numerator: int | None = None
    for index, event in enumerate(events):
        segment_id = int(segment_ids[index])
        group = group_by_key[(segment_id, int(event.beat_offset_numerator))]
        meter = _meter_for_event(event)
        if previous_event is None or previous_segment_id is None or previous_beat_offset_numerator is None:
            beat_delta_units = 0
        elif segment_id == previous_segment_id:
            beat_delta_units = int(event.beat_offset_numerator) - int(previous_beat_offset_numerator)
        else:
            beat_delta_units = _event_delta_units(previous_event, event, snap_denominator)
        if event.kind == BeatEventKind.HOLD_START:
            ln_span_units: str | int = int(ln_span_units_by_start_index.get(index, 0))
            ln_span_bucket = _ln_span_bucket(int(ln_span_units))
        elif event.kind == BeatEventKind.HOLD_END:
            ln_span_units = "end"
            ln_span_bucket = "end"
        else:
            ln_span_units = "none"
            ln_span_bucket = "none"
        joint_key = "|".join(
            (
                str(beat_delta_units),
                str(int(event.beat_offset_numerator) % (meter * snap_denominator)),
                str(event.lane),
                _ACTION_LABEL[event.kind],
                str(group["lane_mask"]),
                str(ln_span_bucket),
            )
        )
        records.append(
            {
                "event_index": index,
                "segment_id": segment_id,
                "beat_offset_numerator": int(event.beat_offset_numerator),
                "beat_delta_units": int(beat_delta_units),
                "bar_phase_units": int(event.beat_offset_numerator) % (meter * snap_denominator),
                "beat_phase": (int(event.beat_offset_numerator) // snap_denominator) % meter,
                "subdivision": int(event.beat_offset_numerator) % snap_denominator,
                "lane": int(event.lane),
                "action_type": _ACTION_LABEL[event.kind],
                "group_lane_mask": int(group["lane_mask"]),
                "group_action_signature": str(group["action_signature"]),
                "ln_span_units": ln_span_units,
                "ln_span_bucket": ln_span_bucket,
                "joint_key": joint_key,
            }
        )
        previous_event = event
        previous_segment_id = segment_id
        previous_beat_offset_numerator = int(event.beat_offset_numerator)
    return records


def _ln_pairs(
    events: Sequence[BeatEvent],
    hitobjects: Sequence[ManiaHitObject],
    segment_ids: Sequence[int],
    *,
    snap_denominator: int,
) -> dict[str, Any]:
    by_key: dict[tuple[BeatEventKind, int, float], list[int]] = defaultdict(list)
    for index, event in enumerate(events):
        by_key[(event.kind, int(event.lane), float(event.time_ms))].append(index)
    span_units_by_start_index: dict[int, int] = {}
    mismatch_count = 0
    nonpositive_count = 0
    near_zero_count = 0
    cross_timing_segment_count = 0
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
        start_event = events[start_index]
        end_event = events[end_index]
        if int(segment_ids[start_index]) == int(segment_ids[end_index]):
            span_units = int(end_event.beat_offset_numerator) - int(start_event.beat_offset_numerator)
        else:
            cross_timing_segment_count += 1
            span_units = _event_delta_units(start_event, end_event, snap_denominator)
        span_units_by_start_index[start_index] = span_units
        if span_units <= 0 or float(end_event.snapped_time_ms) <= float(start_event.snapped_time_ms):
            nonpositive_count += 1
        if 0 < span_units <= 1:
            near_zero_count += 1
    return {
        "span_units_by_start_index": span_units_by_start_index,
        "mismatch_count": mismatch_count,
        "nonpositive_count": nonpositive_count,
        "near_zero_count": near_zero_count,
        "cross_timing_segment_count": cross_timing_segment_count,
    }


def _field_counters(records: Sequence[Mapping[str, Any]]) -> dict[str, Counter[str]]:
    fields = (
        "beat_delta_units",
        "bar_phase_units",
        "beat_phase",
        "subdivision",
        "lane",
        "action_type",
        "group_lane_mask",
        "group_action_signature",
        "ln_span_bucket",
        "ln_span_units_raw",
    )
    counters = {field: Counter() for field in fields}
    for record in records:
        for field in fields:
            source_field = "ln_span_units" if field == "ln_span_units_raw" else field
            counters[field][str(record[source_field])] += 1
    return counters


def _group_counters(records: Sequence[Mapping[str, Any]]) -> dict[str, Counter[str]]:
    counters = {
        "group_delta_units": Counter(),
        "onset_count": Counter(),
        "action_signature": Counter(),
    }
    for record in records:
        counters["group_delta_units"][str(record["group_delta_units"])] += 1
        counters["onset_count"][str(record["onset_count"])] += 1
        counters["action_signature"][str(record["action_signature"])] += 1
    return counters


def _v2_metrics(hitobjects: Sequence[ManiaHitObject], *, vocab: MapperV21Vocab) -> dict[str, Any]:
    token_counter: Counter[str] = Counter()
    ts_counter: Counter[str] = Counter()
    lane_action_counter: Counter[str] = Counter()
    try:
        timepoints = hitobjects_to_mapper_timepoints(hitobjects)
        chart_end_ms = mapper_chart_end_ms(timepoints, chart_start_ms=0)
        token_ids = encode_full_chart_tokens(timepoints, vocab=vocab, chart_start_ms=0, chart_end_ms=chart_end_ms)
    except Exception as exc:  # noqa: BLE001 - baseline failures are audit data.
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "token_counter": token_counter,
            "ts_counter": ts_counter,
            "lane_action_counter": lane_action_counter,
            "token_count": 0,
            "time_shift_token_count": 0,
            "lane_action_token_count": 0,
            "special_token_count": 0,
        }
    time_shift_count = 0
    lane_action_count = 0
    special_count = 0
    for token_id in token_ids:
        name = vocab.token_name(int(token_id))
        token_counter[name] += 1
        if vocab.is_time_shift_token(int(token_id)):
            time_shift_count += 1
            ts_counter[str(vocab.time_shift_value(int(token_id)))] += 1
        elif vocab.is_lane_action_token(int(token_id)):
            lane_action_count += 1
            lane_action_counter[name] += 1
        else:
            special_count += 1
    return {
        "ok": True,
        "error_type": "",
        "error_message": "",
        "token_counter": token_counter,
        "ts_counter": ts_counter,
        "lane_action_counter": lane_action_counter,
        "token_count": len(token_ids),
        "time_shift_token_count": time_shift_count,
        "lane_action_token_count": lane_action_count,
        "special_token_count": special_count,
    }


def _motif_metrics(group_records: Sequence[Mapping[str, Any]], *, motif_ngram: int) -> dict[str, float | int]:
    signatures = [
        f"{record['group_delta_units']}:{record['tap_mask']}:{record['hold_start_mask']}:{record['hold_end_mask']}"
        for record in group_records
    ]
    counter = _ngram_counter(signatures, motif_ngram)
    total = sum(counter.values())
    unique = len(counter)
    repeated = total - unique
    recurrence_rate = float(repeated) / float(total) if total else 0.0
    control_signatures = _control_signatures(signatures)
    control_counter = _ngram_counter(control_signatures, motif_ngram)
    control_total = sum(control_counter.values())
    control_repeated = control_total - len(control_counter)
    control_rate = float(control_repeated) / float(control_total) if control_total else 0.0
    return {
        "total_count": total,
        "unique_count": unique,
        "repeated_count": repeated,
        "recurrence_rate": recurrence_rate,
        "control_recurrence_rate": control_rate,
        "recurrence_lift": recurrence_rate - control_rate,
    }


def _ngram_counter(items: Sequence[str], ngram: int) -> Counter[str]:
    if ngram <= 0 or len(items) < ngram:
        return Counter()
    return Counter(" ".join(items[index : index + ngram]) for index in range(0, len(items) - ngram + 1))


def _control_signatures(signatures: Sequence[str]) -> list[str]:
    if len(signatures) < 4:
        return list(signatures)
    deltas: list[str] = []
    actions: list[str] = []
    for signature in signatures:
        parts = signature.split(":", 1)
        deltas.append(parts[0])
        actions.append(parts[1] if len(parts) > 1 else "")
    shuffled = list(actions)
    seed = int(hashlib.sha256("\n".join(signatures[:64]).encode("utf-8")).hexdigest()[:16], 16)
    random.Random(seed).shuffle(shuffled)
    return [f"{delta}:{action}" for delta, action in zip(deltas, shuffled, strict=True)]


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


def _meter_for_event(event: BeatEvent) -> int:
    # BeatEvent does not carry meter yet; current le3 maps are overwhelmingly 4/4.
    # Keeping this helper narrow makes a future meter-carrying event easy to adopt.
    del event
    return 4


def _group_delta_units(events: Sequence[BeatEvent], previous_index: int, current_index: int, snap_denominator: int) -> int:
    return _event_delta_units(events[previous_index], events[current_index], snap_denominator)


def _event_delta_units(previous: BeatEvent, current: BeatEvent, snap_denominator: int) -> int:
    if (
        round(float(previous.redline_offset_ms), 9) == round(float(current.redline_offset_ms), 9)
        and round(float(previous.beat_length_ms), 9) == round(float(current.beat_length_ms), 9)
    ):
        return int(current.beat_offset_numerator) - int(previous.beat_offset_numerator)
    return int(round((float(current.snapped_time_ms) - float(previous.snapped_time_ms)) / float(current.beat_length_ms) * snap_denominator))


def _ln_span_bucket(span_units: int) -> str:
    if span_units <= 0:
        return "nonpositive"
    if span_units <= 2:
        return "1-2"
    if span_units <= 6:
        return "3-6"
    if span_units <= 12:
        return "7-12"
    if span_units <= 24:
        return "13-24"
    if span_units <= 48:
        return "25-48"
    if span_units <= 96:
        return "49-96"
    if span_units <= 192:
        return "97-192"
    return "gt_192"


def _snapped_order_inversion_count(events: Sequence[BeatEvent]) -> int:
    count = 0
    previous = -math.inf
    for event in events:
        current = float(event.snapped_time_ms)
        if current < previous:
            count += 1
        previous = current
    return count


def _chart_duration_seconds(events: Sequence[BeatEvent]) -> float:
    if not events:
        return 0.0
    start = min(float(event.snapped_time_ms) for event in events)
    end = max(float(event.snapped_time_ms) for event in events)
    return max(0.0, (end - start) / 1000.0)


def _chord_grouping_mismatch_count(
    group_records: Sequence[Mapping[str, Any]],
    event_records: Sequence[Mapping[str, Any]],
) -> int:
    rebuilt: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in event_records:
        rebuilt[(int(record["segment_id"]), int(record["beat_offset_numerator"]))].append(record)
    mismatches = 0
    for group in group_records:
        key = (int(group["segment_id"]), int(group["beat_offset_numerator"]))
        records = rebuilt.get(key, [])
        lane_mask = 0
        actions: dict[int, list[BeatEventKind]] = defaultdict(list)
        for record in records:
            lane = int(record["lane"])
            lane_mask |= 1 << lane
            actions[lane].append(BeatEventKind(str(record["action_type"])))
        if lane_mask != int(group["lane_mask"]) or _group_action_signature(actions) != str(group["action_signature"]):
            mismatches += 1
    return mismatches


def _is_known_canonicalization_exception(base: Mapping[str, Any]) -> bool:
    try:
        beatmap_id = int(base.get("beatmap_id", -1))
    except (TypeError, ValueError):
        beatmap_id = -1
    key = (str(base.get("shard", "")), str(base.get("beatmap_path", "")), beatmap_id)
    return key in _KNOWN_CANONICALIZATION_EXCEPTIONS


def _failure_fields(error_type: str, error_message: str) -> dict[str, Any]:
    return {
        "path_ok": False if error_type == "path_error" else True,
        "conversion_ok": False,
        "hard_gate_ok": False,
        "error_type": error_type,
        "error_message": _short_error(error_message),
    }


def _require_index_columns(index_df: pd.DataFrame, index_path: Path) -> None:
    missing = sorted(_REQUIRED_INDEX_COLUMNS.difference(index_df.columns))
    if missing:
        raise ValueError(f"{index_path} is missing required column(s): {missing}")


def _aggregate_json_counters(frame: pd.DataFrame, column: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    if column not in frame.columns:
        return counter
    for value in frame[column].fillna("{}"):
        if not value:
            continue
        payload = json.loads(str(value))
        counter.update({str(key): int(count) for key, count in payload.items()})
    return counter


def _counter_json(counter: Counter[str]) -> str:
    return json.dumps({str(key): int(value) for key, value in sorted(counter.items())}, separators=(",", ":"), sort_keys=True)


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


def _bits_per_event(entropy_bits: float, encoded_item_count: int, event_count: int) -> float:
    if event_count <= 0:
        return 0.0
    return float(entropy_bits) * float(encoded_item_count) / float(event_count)


def _rare_rate(counter: Counter[str], *, rare_max_count: int) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    rare = sum(count for count in counter.values() if count <= rare_max_count)
    return float(rare) / float(total)


def _weighted_rate(frame: pd.DataFrame, numerator_column: str, denominator_column: str) -> float:
    numerator = _sum_int(frame, numerator_column)
    denominator = _sum_int(frame, denominator_column)
    return float(numerator) / float(denominator) if denominator else 0.0


def _weighted_column_average(frame: pd.DataFrame, value_column: str, weight_column: str) -> float:
    if value_column not in frame.columns or weight_column not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame[value_column], errors="coerce").fillna(0.0)
    weights = pd.to_numeric(frame[weight_column], errors="coerce").fillna(0.0)
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return 0.0
    return float((values * weights).sum()) / weight_sum


def _density_bin(density: float, chord_ratio: float, ln_ratio: float) -> str:
    density_label = "density_low" if density < 3.0 else "density_mid" if density < 7.0 else "density_high"
    chord_label = "chord_low" if chord_ratio < 0.15 else "chord_mid" if chord_ratio < 0.35 else "chord_high"
    ln_label = "ln_low" if ln_ratio < 0.15 else "ln_mid" if ln_ratio < 0.35 else "ln_high"
    return f"{density_label}|{chord_label}|{ln_label}"


def _density_bin_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if "density_bin" not in frame.columns:
        return []
    rows: list[dict[str, Any]] = []
    for density_bin, group in frame.groupby("density_bin", dropna=False):
        total = _sum_int(group, "motif_ngram_total_count")
        rows.append(
            {
                "density_bin": str(density_bin),
                "map_count": int(len(group)),
                "event_count": _sum_int(group, "event_count"),
                "motif_total_count": total,
                "motif_recurrence_rate": _weighted_rate(group, "motif_ngram_repeated_count", "motif_ngram_total_count"),
                "motif_control_recurrence_rate": _weighted_column_average(group, "motif_control_recurrence_rate", "motif_ngram_total_count"),
                "mean_density_events_per_second": _mean_float(group, "density_events_per_second"),
                "mean_chord_ratio": _mean_float(group, "chord_ratio"),
                "mean_ln_ratio": _mean_float(group, "ln_ratio"),
            }
        )
    return sorted(rows, key=lambda item: item["density_bin"])


def _top_counter(counter: Counter[str], limit: int = 20) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.most_common(limit)}


def _top_examples(cache_df: pd.DataFrame, sort_column: str, limit: int, *, ascending: bool = False) -> list[dict[str, Any]]:
    if limit <= 0 or sort_column not in cache_df.columns:
        return []
    successful = cache_df[cache_df.get("conversion_ok", False).astype(bool)].copy()
    if successful.empty:
        return []
    top = successful.sort_values(sort_column, ascending=ascending).head(limit)
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
        sort_column,
        "event_count",
        "beat_record_count",
        "beat_group_count",
        "v2_token_count",
        "density_bin",
        "density_events_per_second",
        "chord_ratio",
        "ln_ratio",
    ]
    return _records_for_columns(top, columns)


def _hard_gate_failure_examples(cache_df: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or "hard_gate_ok" not in cache_df.columns:
        return []
    failed = cache_df[~cache_df["hard_gate_ok"].astype(bool)].head(limit)
    columns = [
        "source_row_index",
        "shard",
        "beatmap_set_id",
        "beatmap_id",
        "beatmap_path",
        "error_type",
        "error_message",
        "event_count_mismatch_count",
        "event_order_mismatch_count",
        "lane_identity_mismatch_count",
        "action_type_mismatch_count",
        "chord_grouping_mismatch_count",
        "ln_pairing_mismatch_count",
        "snap_error_bound_violation_count",
        "snapped_ln_nonpositive_count",
        "known_canonicalization_exception_count",
        "unflagged_snapped_ln_nonpositive_count",
    ]
    return _records_for_columns(failed, columns)


def _v2_failure_examples(cache_df: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or "v2_tokenization_ok" not in cache_df.columns:
        return []
    failed = cache_df[~cache_df["v2_tokenization_ok"].astype(bool)].head(limit)
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
        "event_count",
        "beat_record_count",
        "beat_group_count",
        "v2_error_type",
        "v2_error_message",
    ]
    return _records_for_columns(failed, columns)


def _records_for_columns(frame: pd.DataFrame, columns: Sequence[str]) -> list[dict[str, Any]]:
    selected = [column for column in columns if column in frame.columns]
    return [_normalize_json(record) for record in frame[selected].to_dict(orient="records")]


def _count_true(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    return int(frame[column].fillna(False).astype(bool).sum())


def _sum_int(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    return int(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())


def _mean_float(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns or frame.empty:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else 0.0


def _value_counts(frame: pd.DataFrame, column: str, *, exclude_empty: bool) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    values = frame[column].fillna("").astype(str)
    if exclude_empty:
        values = values[values != ""]
    return {str(key): int(value) for key, value in values.value_counts().sort_index().items()}


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
    return " ".join(shlex.quote(part) for part in ["python", "-m", "pulsefield_model.osu_core.beat_structure_token_audit", *args])


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit factorized beat-structure tokens for osu!mania 4K beatmaps.")
    parser.add_argument("--source-index-path", type=Path, default=DEFAULT_LE3_INDEX_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_LE3_STRUCTURE_CACHE_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_LE3_STRUCTURE_REPORT_PATH)
    parser.add_argument("--snap-denominator", type=int, default=DEFAULT_SNAP_DENOMINATOR)
    parser.add_argument(
        "--timing-canonicalization",
        choices=TIMING_CANONICALIZATION_CHOICES,
        default=DEFAULT_BEAT_REPRESENTATION_TIMING_CANONICALIZATION,
    )
    parser.add_argument("--expected-key-count", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--rare-max-count", type=int, default=DEFAULT_RARE_MAX_COUNT)
    parser.add_argument("--motif-ngram", type=int, default=DEFAULT_MOTIF_NGRAM)
    parser.add_argument("--top-example-limit", type=int, default=DEFAULT_TOP_EXAMPLE_LIMIT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    report = audit_beat_structure_token_index(
        source_index_path=args.source_index_path,
        dataset_root=args.dataset_root,
        cache_path=args.cache_path,
        report_path=args.report_path,
        snap_denominator=args.snap_denominator,
        timing_canonicalization=args.timing_canonicalization,
        expected_key_count=args.expected_key_count,
        limit=args.limit,
        progress_every=args.progress_every,
        rare_max_count=args.rare_max_count,
        motif_ngram=args.motif_ngram,
        top_example_limit=args.top_example_limit,
        command=_format_command(argv),
    )
    print(json.dumps(_normalize_json(report), allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
