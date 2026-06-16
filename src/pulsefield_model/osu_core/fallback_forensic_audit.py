from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final, Iterable, Mapping, Sequence

import pandas as pd

from pulsefield_model.osu_core.beat_chunk_pattern_audit import (
    DEFAULT_BEAT_CHUNK_CACHE_PATH,
    DEFAULT_MOTIF_MAX_N,
    DEFAULT_MOTIF_MIN_N,
    DEFAULT_SMOOTHING_ALPHA,
)
from pulsefield_model.osu_core.beat_chunk_tokenizer_refinement_audit import DEFAULT_RARE_MAX_COUNT
from pulsefield_model.osu_core.duration_ln_tokenization_audit import (
    CONTROL_TOKENS,
    DEFAULT_MOTIF_VOCAB_SIZE,
    _EncodedStats,
    _MotifVocab,
    _build_motif_trie,
    _build_motif_vocab,
    _dictionary_cost_bits_generic,
    _encode_sequence,
    _make_model_token_bits,
    _rank_motifs,
    _score_encoded_generic,
    _split_report,
)


SCHEMA_VERSION: Final[int] = 1
DEFAULT_REPORT_PATH: Final[Path] = Path(
    "artifacts/reports/audits/occupancy_factorized_tokenization/fallback_forensic_report.json",
)
DEFAULT_RESULT_LOG_PATH: Final[Path] = Path(
    "artifacts/reports/audits/occupancy_factorized_tokenization/fallback_forensic_result_log.md",
)
DEFAULT_TABLES_CSV_PATH: Final[Path] = Path(
    "artifacts/reports/audits/occupancy_factorized_tokenization/fallback_forensic_tables.csv",
)
EXPECTED_R0_TEST_CHARGED_BITS_PER_EVENT: Final[float] = 4.421714207191301
CHUNK_UNITS: Final[int] = 96
BOUNDARY_NEAR_UNITS: Final[int] = 12


@dataclass(frozen=True)
class _FittedMotifModel:
    name: str
    vocab: _MotifVocab
    trie: Mapping[str, Any]
    atom_counter: Counter[str]
    dictionary_bits: float
    token_bits: Callable[[str], float]
    train_counter: Counter[str]
    learned_motif_count: int
    top_motifs: list[dict[str, Any]]


@dataclass
class _BucketAgg:
    table: str
    bucket: str
    split: str
    group_count: int = 0
    event_count: int = 0
    fallback_group_count: int = 0
    fallback_event_count: int = 0
    lane_erased_recovered_fallback_count: int = 0
    skeleton_recovered_fallback_count: int = 0
    mapsets: set[int] | None = None
    examples: list[dict[str, Any]] | None = None

    def add(
        self,
        *,
        beatmap_set_id: int,
        event_count: int,
        fallback: bool,
        lane_erased_recovered: bool,
        skeleton_recovered: bool,
        example: dict[str, Any] | None,
    ) -> None:
        if self.mapsets is None:
            self.mapsets = set()
        if self.examples is None:
            self.examples = []
        self.group_count += 1
        self.event_count += int(event_count)
        self.mapsets.add(int(beatmap_set_id))
        if not fallback:
            return
        self.fallback_group_count += 1
        self.fallback_event_count += int(event_count)
        self.lane_erased_recovered_fallback_count += int(lane_erased_recovered)
        self.skeleton_recovered_fallback_count += int(skeleton_recovered)
        if example is not None and len(self.examples) < 5:
            self.examples.append(example)


def audit_fallback_forensics(
    *,
    chunk_cache_path: str | Path = DEFAULT_BEAT_CHUNK_CACHE_PATH,
    report_path: str | Path | None = DEFAULT_REPORT_PATH,
    result_log_path: str | Path | None = DEFAULT_RESULT_LOG_PATH,
    tables_csv_path: str | Path | None = DEFAULT_TABLES_CSV_PATH,
    motif_vocab_size: int = DEFAULT_MOTIF_VOCAB_SIZE,
    motif_min_n: int = DEFAULT_MOTIF_MIN_N,
    motif_max_n: int = DEFAULT_MOTIF_MAX_N,
    smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
    rare_max_count: int = DEFAULT_RARE_MAX_COUNT,
    top_motif_limit: int = 20,
    limit_chunks: int | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    """Run cache-only fallback forensics for the baseline motif-delta tokenizer."""

    started_at = time.perf_counter()
    chunk_cache_path = Path(chunk_cache_path)
    report_path = None if report_path is None else Path(report_path)
    result_log_path = None if result_log_path is None else Path(result_log_path)
    tables_csv_path = None if tables_csv_path is None else Path(tables_csv_path)
    motif_vocab_size = _positive_int(motif_vocab_size, "motif_vocab_size")
    motif_min_n = _positive_int(motif_min_n, "motif_min_n")
    motif_max_n = _positive_int(motif_max_n, "motif_max_n")
    if motif_max_n < motif_min_n:
        raise ValueError("motif_max_n must be >= motif_min_n")
    if smoothing_alpha <= 0:
        raise ValueError(f"smoothing_alpha must be positive, got {smoothing_alpha!r}")
    rare_max_count = _nonnegative_int(rare_max_count, "rare_max_count")

    chunk_df = _read_chunk_cache(chunk_cache_path, limit_chunks=limit_chunks)
    baseline_model = _fit_motif_model(
        chunk_df,
        name="r0_delta",
        token_mapper=_baseline_tokens,
        motif_vocab_size=motif_vocab_size,
        motif_min_n=motif_min_n,
        motif_max_n=motif_max_n,
        smoothing_alpha=smoothing_alpha,
        top_motif_limit=top_motif_limit,
    )
    lane_model = _fit_motif_model(
        chunk_df,
        name="lane_erased",
        token_mapper=_lane_erased_tokens,
        motif_vocab_size=motif_vocab_size,
        motif_min_n=motif_min_n,
        motif_max_n=motif_max_n,
        smoothing_alpha=smoothing_alpha,
        top_motif_limit=top_motif_limit,
    )
    skeleton_model = _fit_motif_model(
        chunk_df,
        name="skeleton",
        token_mapper=_skeleton_tokens,
        motif_vocab_size=motif_vocab_size,
        motif_min_n=motif_min_n,
        motif_max_n=motif_max_n,
        smoothing_alpha=smoothing_alpha,
        top_motif_limit=top_motif_limit,
    )

    baseline_splits = {
        split: _score_baseline_split(
            chunk_df,
            split=split,
            model=baseline_model,
            dictionary_bits=baseline_model.dictionary_bits,
            rare_max_count=rare_max_count,
        )
        for split in ("train", "valid", "test")
    }
    forensic = _forensic_tables(
        chunk_df,
        baseline_model=baseline_model,
        lane_model=lane_model,
        skeleton_model=skeleton_model,
    )
    table_rows = forensic["tables"]
    summary = _summarize_forensics(table_rows)
    pass_criteria = _pass_criteria(
        baseline_splits=baseline_splits,
        invalid_transition_count=forensic["invalid_transition_count"],
        segment_state_reset_count=forensic["segment_state_reset_count"],
    )
    recommendation = _recommendation(summary)

    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "Fallback Forensic Audit",
        "command": command,
        "code_commit": _git_stdout("rev-parse", "HEAD"),
        "code_dirty": bool(_git_stdout("status", "--short")),
        "chunk_cache_path": chunk_cache_path.as_posix(),
        "report_path": None if report_path is None else report_path.as_posix(),
        "result_log_path": None if result_log_path is None else result_log_path.as_posix(),
        "tables_csv_path": None if tables_csv_path is None else tables_csv_path.as_posix(),
        "limited": limit_chunks is not None,
        "limit_chunks": limit_chunks,
        "elapsed_s": time.perf_counter() - started_at,
        "config": {
            "motif_vocab_size": motif_vocab_size,
            "motif_min_n": motif_min_n,
            "motif_max_n": motif_max_n,
            "smoothing_alpha": smoothing_alpha,
            "rare_max_count": rare_max_count,
            "boundary_near_units": BOUNDARY_NEAR_UNITS,
            "chunk_units": CHUNK_UNITS,
        },
        "dataset": _dataset_summary(chunk_df),
        "models": {
            "r0_delta": _model_summary(baseline_model),
            "lane_erased": _model_summary(lane_model),
            "skeleton": _model_summary(skeleton_model),
        },
        "baseline_splits": baseline_splits,
        "pass_criteria": pass_criteria,
        "forensic_summary": summary,
        "invalid_transition_count": forensic["invalid_transition_count"],
        "invalid_transition_examples": forensic["invalid_transition_examples"],
        "segment_state_reset_count": forensic["segment_state_reset_count"],
        "tables": table_rows,
        "recommendation": recommendation,
    }
    if report_path is not None:
        _write_json(report_path, report)
    if tables_csv_path is not None:
        tables_csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(table_rows).to_csv(tables_csv_path, index=False)
    if result_log_path is not None:
        _write_result_log(result_log_path, report)
    return report


def _fit_motif_model(
    chunk_df: pd.DataFrame,
    *,
    name: str,
    token_mapper: Callable[[Any], list[str]],
    motif_vocab_size: int,
    motif_min_n: int,
    motif_max_n: int,
    smoothing_alpha: float,
    top_motif_limit: int,
) -> _FittedMotifModel:
    atom_counter: Counter[str] = Counter()
    candidate_counter: Counter[tuple[str, ...]] = Counter()
    for row in _iter_rows(chunk_df, split="train"):
        tokens = token_mapper(row)
        atom_counter.update(tokens)
        max_n = min(motif_max_n, len(tokens))
        for ngram in range(motif_min_n, max_n + 1):
            for index in range(len(tokens) - ngram + 1):
                candidate_counter[tuple(tokens[index : index + ngram])] += 1
    candidates = _rank_motifs(candidate_counter)
    vocab = _build_motif_vocab(candidates[:motif_vocab_size])
    trie = _build_motif_trie(vocab.motif_to_id)
    train_counter: Counter[str] = Counter()
    for row in _iter_rows(chunk_df, split="train"):
        encoded, _covered = _encode_sequence(token_mapper(row), trie)
        train_counter.update(encoded)
    dictionary_bits = _dictionary_cost_bits_generic(
        vocab,
        atom_counter,
        smoothing_alpha=smoothing_alpha,
        motif_max_n=motif_max_n,
    )
    model_universe = set(train_counter) | set(atom_counter) | set(vocab.motif_lengths) | CONTROL_TOKENS
    token_bits = _make_model_token_bits(
        train_counter,
        atom_counter,
        model_universe,
        smoothing_alpha=smoothing_alpha,
    )
    return _FittedMotifModel(
        name=name,
        vocab=vocab,
        trie=trie,
        atom_counter=atom_counter,
        dictionary_bits=dictionary_bits,
        token_bits=token_bits,
        train_counter=train_counter,
        learned_motif_count=len(vocab.motif_to_id),
        top_motifs=_top_motifs(candidates[:top_motif_limit]),
    )


def _score_baseline_split(
    chunk_df: pd.DataFrame,
    *,
    split: str,
    model: _FittedMotifModel,
    dictionary_bits: float,
    rare_max_count: int,
) -> dict[str, Any]:
    stats = _EncodedStats()
    for row in _iter_rows(chunk_df, split=split):
        tokens = _baseline_tokens(row)
        encoded, covered_indexes = _encode_sequence(tokens, model.trie)
        stats.chunk_count += 1
        stats.event_count += int(row.num_events)
        stats.group_count += int(row.num_groups)
        stats.mapset_events[int(row.beatmap_set_id)] += int(row.num_events)
        stats.motif_group_coverage_count += sum(1 for index in covered_indexes if index > 0)
        for token in encoded:
            stats.token_counter[token] += 1
            stats.token_count += 1
            if token.startswith("M"):
                stats.motif_token_count += 1
            else:
                stats.atom_token_count += 1
    _score_encoded_generic(
        stats,
        token_bits=model.token_bits,
        dictionary_bits=dictionary_bits,
        rare_max_count=rare_max_count,
    )
    return _split_report(stats, dictionary_bits=dictionary_bits)


def _forensic_tables(
    chunk_df: pd.DataFrame,
    *,
    baseline_model: _FittedMotifModel,
    lane_model: _FittedMotifModel,
    skeleton_model: _FittedMotifModel,
) -> dict[str, Any]:
    aggs: dict[tuple[str, str, str], _BucketAgg] = {}
    invalid_transition_count = 0
    segment_state_reset_count = 0
    invalid_examples: list[dict[str, Any]] = []
    state_by_source_segment: dict[tuple[int, int], int] = {}
    ordered = chunk_df.sort_values(["source_row_index", "segment_id", "start_beat_units", "chunk_index"])
    for row in ordered.itertuples(index=False):
        source = int(row.source_row_index)
        segment = int(row.segment_id)
        state_key = (source, segment)
        state = state_by_source_segment.get(state_key, 0)
        baseline_tokens = _baseline_tokens(row)
        lane_tokens = _lane_erased_tokens(row)
        skeleton_tokens = _skeleton_tokens(row)
        _encoded, baseline_covered = _encode_sequence(baseline_tokens, baseline_model.trie)
        _lane_encoded, lane_covered = _encode_sequence(lane_tokens, lane_model.trie)
        _skel_encoded, skeleton_covered = _encode_sequence(skeleton_tokens, skeleton_model.trie)
        groups = json.loads(str(row.groups_json or "[]"))
        chunk_start_state = state
        for index, group in enumerate(groups, start=1):
            tap = int(group.get("tap_mask", 0))
            start = int(group.get("ln_start_mask", 0))
            end = int(group.get("ln_end_mask", 0))
            invalid_mask = (end & ~state) | (start & state)
            if invalid_mask:
                invalid_transition_count += invalid_mask.bit_count()
                if len(invalid_examples) < 10:
                    invalid_examples.append(
                        {
                            "source_row_index": source,
                            "beatmap_set_id": int(row.beatmap_set_id),
                            "chunk_index": int(row.chunk_index),
                            "offset_units": int(group.get("offset_units", 0)),
                            "pre_hold_mask": state,
                            "tap_mask": tap,
                            "ln_start_mask": start,
                            "ln_end_mask": end,
                            "invalid_mask": invalid_mask,
                        }
                    )
            fallback = index not in baseline_covered
            lane_recovered = index in lane_covered
            skeleton_recovered = index in skeleton_covered
            event_count = max(1, int(group.get("event_count", 1) or 1))
            example = (
                {
                    "source_row_index": source,
                    "beatmap_set_id": int(row.beatmap_set_id),
                    "chunk_index": int(row.chunk_index),
                    "offset_units": int(group.get("offset_units", 0)),
                    "pre_hold_mask": state,
                    "token": baseline_tokens[index],
                    "lane_erased_token": lane_tokens[index],
                    "skeleton_token": skeleton_tokens[index],
                }
                if fallback
                else None
            )
            feature_values = _feature_values(
                group,
                pre_hold_mask=state,
                chunk_start_hold_mask=chunk_start_state,
            )
            for table, bucket in feature_values:
                _add_bucket(
                    aggs,
                    split=str(row.split),
                    table=table,
                    bucket=bucket,
                    beatmap_set_id=int(row.beatmap_set_id),
                    event_count=event_count,
                    fallback=fallback,
                    lane_erased_recovered=lane_recovered,
                    skeleton_recovered=skeleton_recovered,
                    example=example,
                )
            state = (state | start) & ~end
        if state:
            _add_bucket(
                aggs,
                split=str(row.split),
                table="chunk_exit_hold_count",
                bucket=_count_bucket(state.bit_count()),
                beatmap_set_id=int(row.beatmap_set_id),
                event_count=0,
                fallback=False,
                lane_erased_recovered=False,
                skeleton_recovered=False,
                example=None,
            )
        state_by_source_segment[state_key] = state
    for state in state_by_source_segment.values():
        if state:
            segment_state_reset_count += state.bit_count()
    return {
        "tables": [_bucket_row(agg) for agg in sorted(aggs.values(), key=lambda item: (item.split, item.table, item.bucket))],
        "invalid_transition_count": invalid_transition_count,
        "invalid_transition_examples": invalid_examples,
        "segment_state_reset_count": segment_state_reset_count,
    }


def _feature_values(
    group: Mapping[str, Any],
    *,
    pre_hold_mask: int,
    chunk_start_hold_mask: int,
) -> list[tuple[str, str]]:
    tap = int(group.get("tap_mask", 0))
    start = int(group.get("ln_start_mask", 0))
    end = int(group.get("ln_end_mask", 0))
    offset = int(group.get("offset_units", 0))
    active_mask = tap | start | end
    values = [
        ("overall", "all"),
        ("group_class", _group_class(tap, start, end)),
        ("boundary_bucket", _boundary_bucket(offset)),
        ("active_hold_count", _count_bucket(pre_hold_mask.bit_count())),
        ("chunk_start_hold_count", _count_bucket(chunk_start_hold_mask.bit_count())),
        ("chord_cardinality", _count_bucket(active_mask.bit_count())),
        ("tap_count", _count_bucket(tap.bit_count())),
        ("ln_start_count", _count_bucket(start.bit_count())),
        ("ln_end_count", _count_bucket(end.bit_count())),
        ("offset_units", str(offset)),
        ("group_class_x_active", f"{_group_class(tap, start, end)}|active_{_count_bucket(pre_hold_mask.bit_count())}"),
        ("group_class_x_boundary", f"{_group_class(tap, start, end)}|{_boundary_bucket(offset)}"),
    ]
    if pre_hold_mask:
        values.append(("active_hold_mask", str(pre_hold_mask)))
    return values


def _add_bucket(
    aggs: dict[tuple[str, str, str], _BucketAgg],
    *,
    split: str,
    table: str,
    bucket: str,
    beatmap_set_id: int,
    event_count: int,
    fallback: bool,
    lane_erased_recovered: bool,
    skeleton_recovered: bool,
    example: dict[str, Any] | None,
) -> None:
    key = (split, table, bucket)
    agg = aggs.setdefault(key, _BucketAgg(table=table, bucket=bucket, split=split))
    agg.add(
        beatmap_set_id=beatmap_set_id,
        event_count=event_count,
        fallback=fallback,
        lane_erased_recovered=lane_erased_recovered,
        skeleton_recovered=skeleton_recovered,
        example=example,
    )


def _bucket_row(agg: _BucketAgg) -> dict[str, Any]:
    mapsets = agg.mapsets or set()
    fallback = agg.fallback_group_count
    return {
        "split": agg.split,
        "table": agg.table,
        "bucket": agg.bucket,
        "group_count": int(agg.group_count),
        "event_count": int(agg.event_count),
        "fallback_group_count": int(fallback),
        "fallback_group_rate": float(fallback) / float(agg.group_count) if agg.group_count else 0.0,
        "fallback_event_count": int(agg.fallback_event_count),
        "fallback_event_rate": float(agg.fallback_event_count) / float(agg.event_count) if agg.event_count else 0.0,
        "lane_erased_recovered_fallback_count": int(agg.lane_erased_recovered_fallback_count),
        "lane_erased_recovered_fallback_rate": (
            float(agg.lane_erased_recovered_fallback_count) / float(fallback) if fallback else 0.0
        ),
        "skeleton_recovered_fallback_count": int(agg.skeleton_recovered_fallback_count),
        "skeleton_recovered_fallback_rate": (
            float(agg.skeleton_recovered_fallback_count) / float(fallback) if fallback else 0.0
        ),
        "mapset_count": len(mapsets),
        "examples": json.dumps(agg.examples or [], separators=(",", ":"), sort_keys=True),
    }


def _summarize_forensics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key = {(str(row["split"]), str(row["table"]), str(row["bucket"])): row for row in rows}
    summary: dict[str, Any] = {}
    for split in ("valid", "test"):
        overall = by_key.get((split, "overall", "all"), {})
        fallback_count = int(overall.get("fallback_group_count", 0) or 0)
        active_rows = [
            row
            for row in rows
            if row.get("split") == split and row.get("table") == "active_hold_count" and row.get("bucket") != "0"
        ]
        boundary_rows = [
            row
            for row in rows
            if row.get("split") == split
            and row.get("table") == "boundary_bucket"
            and row.get("bucket") in {"exact_start", "near_start_le12", "near_end_le12"}
        ]
        chord_ln_rows = [
            row
            for row in rows
            if row.get("split") == split
            and row.get("table") == "group_class"
            and "ln" in str(row.get("bucket"))
            and "tap" in str(row.get("bucket"))
        ]
        active_fallback = sum(int(row.get("fallback_group_count", 0) or 0) for row in active_rows)
        boundary_fallback = sum(int(row.get("fallback_group_count", 0) or 0) for row in boundary_rows)
        chord_ln_fallback = sum(int(row.get("fallback_group_count", 0) or 0) for row in chord_ln_rows)
        lane_recovered = int(overall.get("lane_erased_recovered_fallback_count", 0) or 0)
        skeleton_recovered = int(overall.get("skeleton_recovered_fallback_count", 0) or 0)
        summary[split] = {
            "fallback_group_count": fallback_count,
            "fallback_group_rate": overall.get("fallback_group_rate"),
            "lane_erased_recovered_fallback_rate": float(lane_recovered) / float(fallback_count) if fallback_count else 0.0,
            "skeleton_recovered_fallback_rate": float(skeleton_recovered) / float(fallback_count) if fallback_count else 0.0,
            "active_hold_fallback_share": float(active_fallback) / float(fallback_count) if fallback_count else 0.0,
            "boundary_near_fallback_share": float(boundary_fallback) / float(fallback_count) if fallback_count else 0.0,
            "chord_ln_mixed_fallback_share": float(chord_ln_fallback) / float(fallback_count) if fallback_count else 0.0,
            "top_group_classes": _top_rows(rows, split=split, table="group_class", count_key="fallback_group_count", limit=8),
            "top_boundary_buckets": _top_rows(rows, split=split, table="boundary_bucket", count_key="fallback_group_count", limit=8),
            "top_active_hold_buckets": _top_rows(rows, split=split, table="active_hold_count", count_key="fallback_group_count", limit=8),
        }
    return summary


def _recommendation(summary: Mapping[str, Any]) -> str:
    test = summary.get("test", {}) if isinstance(summary, Mapping) else {}
    lane = float(test.get("lane_erased_recovered_fallback_rate", 0.0) or 0.0)
    skeleton = float(test.get("skeleton_recovered_fallback_rate", 0.0) or 0.0)
    active = float(test.get("active_hold_fallback_share", 0.0) or 0.0)
    boundary = float(test.get("boundary_near_fallback_share", 0.0) or 0.0)
    chord_ln = float(test.get("chord_ln_mixed_fallback_share", 0.0) or 0.0)
    if skeleton >= 0.35 or lane >= 0.35 or chord_ln >= 0.35:
        return "TEST_E3_E4: fallback is recoverable after factorization/wildcard masking or concentrated in chord/LN mixed groups."
    if active >= 0.35:
        return "TEST_E2: fallback is strongly tied to active hold state."
    if boundary >= 0.35:
        return "TEST_E5: fallback is strongly tied to chunk boundaries."
    return "MUTATE: fallback is diffuse under current probes; avoid larger tokenizer variants until a sharper cause is found."


def _pass_criteria(
    *,
    baseline_splits: Mapping[str, Mapping[str, Any]],
    invalid_transition_count: int,
    segment_state_reset_count: int,
) -> dict[str, Any]:
    test = baseline_splits.get("test", {})
    observed = float(test.get("bits_per_event_with_dictionary", math.inf))
    baseline_consistency = abs(observed - EXPECTED_R0_TEST_CHARGED_BITS_PER_EVENT) < 0.05
    return {
        "baseline_consistency_pass": baseline_consistency,
        "expected_r0_test_charged_bits_per_event": EXPECTED_R0_TEST_CHARGED_BITS_PER_EVENT,
        "observed_r0_test_charged_bits_per_event": observed,
        "active_hold_trace_clean_pass": int(invalid_transition_count) == 0 and int(segment_state_reset_count) == 0,
        "invalid_transition_count": int(invalid_transition_count),
        "segment_state_reset_count": int(segment_state_reset_count),
        "research_pass": bool(baseline_consistency),
    }


def _baseline_tokens(row: Any) -> list[str]:
    return [f"C{int(row.bar_phase_half)}", *_group_tokens(row, prefix="D", transform="raw")]


def _lane_erased_tokens(row: Any) -> list[str]:
    return [f"C{int(row.bar_phase_half)}", *_group_tokens(row, prefix="L", transform="lane_erased")]


def _skeleton_tokens(row: Any) -> list[str]:
    return [f"C{int(row.bar_phase_half)}", *_group_tokens(row, prefix="K", transform="skeleton")]


def _group_tokens(row: Any, *, prefix: str, transform: str) -> list[str]:
    groups = json.loads(str(row.groups_json or "[]"))
    tokens: list[str] = []
    previous_offset = 0
    for group in groups:
        offset = int(group.get("offset_units", 0))
        delta = offset - previous_offset
        previous_offset = offset
        tap = int(group.get("tap_mask", 0))
        start = int(group.get("ln_start_mask", 0))
        end = int(group.get("ln_end_mask", 0))
        order_signature = str(group.get("order_signature", ".") or ".")
        if transform == "raw":
            token = f"{prefix}:{delta}:{tap}:{start}:{end}"
        elif transform == "lane_erased":
            token = f"{prefix}:{delta}:T{tap.bit_count()}:S{start.bit_count()}:E{end.bit_count()}"
        elif transform == "skeleton":
            active_count = (tap | start | end).bit_count()
            token = f"{prefix}:{delta}:{_group_class(tap, start, end)}:A{_count_bucket(active_count)}"
        else:
            raise ValueError(f"unknown token transform: {transform}")
        tokens.append(token if order_signature == "." else f"{token}:O{order_signature}")
    return tokens


def _group_class(tap: int, start: int, end: int) -> str:
    parts: list[str] = []
    if tap:
        parts.append("tap")
    if start:
        parts.append("ln_start")
    if end:
        parts.append("ln_end")
    return "_".join(parts) if parts else "empty"


def _boundary_bucket(offset_units: int) -> str:
    offset_units = int(offset_units)
    if offset_units == 0:
        return "exact_start"
    if offset_units <= BOUNDARY_NEAR_UNITS:
        return "near_start_le12"
    if CHUNK_UNITS - offset_units <= BOUNDARY_NEAR_UNITS:
        return "near_end_le12"
    return "middle"


def _count_bucket(count: int) -> str:
    count = int(count)
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    return "3plus"


def _top_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    table: str,
    count_key: str,
    limit: int,
) -> list[dict[str, Any]]:
    selected = [row for row in rows if row.get("split") == split and row.get("table") == table]
    selected.sort(key=lambda row: (-int(row.get(count_key, 0) or 0), str(row.get("bucket"))))
    output = []
    for row in selected[:limit]:
        output.append(
            {
                "bucket": row.get("bucket"),
                "group_count": row.get("group_count"),
                "fallback_group_count": row.get("fallback_group_count"),
                "fallback_group_rate": row.get("fallback_group_rate"),
                "lane_erased_recovered_fallback_rate": row.get("lane_erased_recovered_fallback_rate"),
                "skeleton_recovered_fallback_rate": row.get("skeleton_recovered_fallback_rate"),
                "mapset_count": row.get("mapset_count"),
            }
        )
    return output


def _top_motifs(candidates: Sequence[tuple[tuple[str, ...], int, float]]) -> list[dict[str, Any]]:
    rows = []
    for index, (motif, count, gain) in enumerate(candidates):
        rows.append(
            {
                "rank": index + 1,
                "length": len(motif),
                "support": int(count),
                "gain": float(gain),
                "motif": " ".join(motif),
            }
        )
    return rows


def _model_summary(model: _FittedMotifModel) -> dict[str, Any]:
    return {
        "name": model.name,
        "learned_motif_count": model.learned_motif_count,
        "atom_vocab_size": len(model.atom_counter),
        "train_encoded_vocab_size": len(model.train_counter),
        "dictionary_cost_bits": model.dictionary_bits,
        "top_motifs": model.top_motifs,
    }


def _dataset_summary(chunk_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "chunk_count": int(len(chunk_df)),
        "event_count": int(pd.to_numeric(chunk_df["num_events"], errors="coerce").fillna(0).sum()),
        "group_count": int(pd.to_numeric(chunk_df["num_groups"], errors="coerce").fillna(0).sum()),
        "map_count": int(chunk_df["source_row_index"].nunique()),
        "mapset_count": int(chunk_df["beatmap_set_id"].nunique()),
        "split_summary": {
            str(split): {
                "chunk_count": int(len(frame)),
                "event_count": int(pd.to_numeric(frame["num_events"], errors="coerce").fillna(0).sum()),
                "group_count": int(pd.to_numeric(frame["num_groups"], errors="coerce").fillna(0).sum()),
                "map_count": int(frame["source_row_index"].nunique()),
                "mapset_count": int(frame["beatmap_set_id"].nunique()),
            }
            for split, frame in chunk_df.groupby("split", dropna=False)
        },
    }


def _read_chunk_cache(path: Path, *, limit_chunks: int | None) -> pd.DataFrame:
    columns = [
        "source_row_index",
        "split",
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
        "chunk_index",
        "segment_id",
        "segment_chunk_index",
        "bar_phase_half",
        "start_beat_units",
        "num_groups",
        "num_events",
        "raw_signature",
        "groups_json",
    ]
    frame = pd.read_parquet(path, columns=columns)
    if limit_chunks is not None:
        frame = frame.head(int(limit_chunks)).copy()
    return frame


def _iter_rows(chunk_df: pd.DataFrame, *, split: str | None = None) -> Iterable[Any]:
    frame = chunk_df
    if split is not None:
        frame = frame[frame["split"].fillna("").astype(str) == split]
    yield from frame.itertuples(index=False)


def _write_result_log(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pass_criteria = report.get("pass_criteria", {})
    summary = report.get("forensic_summary", {})
    lines = [
        "# Fallback Forensic Audit Result Log",
        "",
        "## Summary",
        "",
        f"- Research pass: {pass_criteria.get('research_pass')}",
        f"- Recommendation: {report.get('recommendation')}",
        f"- Runtime seconds: {_fmt_float(report.get('elapsed_s'))}",
        f"- Observed r0 test charged bits/event: {_fmt_float(pass_criteria.get('observed_r0_test_charged_bits_per_event'))}",
        f"- Baseline consistency pass: {pass_criteria.get('baseline_consistency_pass')}",
        f"- Active-hold trace clean pass: {pass_criteria.get('active_hold_trace_clean_pass')}",
        f"- Invalid transition count: {report.get('invalid_transition_count')}",
        f"- Segment end active hold count: {report.get('segment_state_reset_count')}",
        "",
        "## Test Forensic Summary",
        "",
    ]
    test = summary.get("test", {}) if isinstance(summary, Mapping) else {}
    for key in (
        "fallback_group_count",
        "fallback_group_rate",
        "lane_erased_recovered_fallback_rate",
        "skeleton_recovered_fallback_rate",
        "active_hold_fallback_share",
        "boundary_near_fallback_share",
        "chord_ln_mixed_fallback_share",
    ):
        lines.append(f"- {key}: {_fmt_float(test.get(key))}")
    lines.extend(["", "## Top Test Group Classes", "", "| bucket | fallback groups | fallback rate | lane-erased recovered | skeleton recovered |", "|---|---:|---:|---:|---:|"])
    for row in test.get("top_group_classes", []) if isinstance(test, Mapping) else []:
        lines.append(
            "| {bucket} | {fallback} | {rate} | {lane} | {skeleton} |".format(
                bucket=row.get("bucket"),
                fallback=row.get("fallback_group_count"),
                rate=_fmt_float(row.get("fallback_group_rate")),
                lane=_fmt_float(row.get("lane_erased_recovered_fallback_rate")),
                skeleton=_fmt_float(row.get("skeleton_recovered_fallback_rate")),
            )
        )
    lines.extend(["", "## Interpretation", "", str(report.get("recommendation"))])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_normalize_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _normalize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_normalize_json(item) for item in value]
    if isinstance(value, set):
        return sorted(_normalize_json(item) for item in value)
    if hasattr(value, "item"):
        try:
            return _normalize_json(value.item())
        except Exception:  # noqa: BLE001 - defensive conversion.
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _fmt_float(value: object) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.6f}"


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value


def _nonnegative_int(value: int, name: str) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative, got {value!r}")
    return value


def _git_stdout(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001 - git metadata is optional.
        return ""


def _command(args: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in ["python", "-m", "pulsefield_model.osu_core.fallback_forensic_audit", *args])


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit baseline motif fallback structural causes.")
    parser.add_argument("--chunk-cache-path", type=Path, default=DEFAULT_BEAT_CHUNK_CACHE_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--result-log-path", type=Path, default=DEFAULT_RESULT_LOG_PATH)
    parser.add_argument("--tables-csv-path", type=Path, default=DEFAULT_TABLES_CSV_PATH)
    parser.add_argument("--motif-vocab-size", type=int, default=DEFAULT_MOTIF_VOCAB_SIZE)
    parser.add_argument("--motif-min-n", type=int, default=DEFAULT_MOTIF_MIN_N)
    parser.add_argument("--motif-max-n", type=int, default=DEFAULT_MOTIF_MAX_N)
    parser.add_argument("--smoothing-alpha", type=float, default=DEFAULT_SMOOTHING_ALPHA)
    parser.add_argument("--rare-max-count", type=int, default=DEFAULT_RARE_MAX_COUNT)
    parser.add_argument("--top-motif-limit", type=int, default=20)
    parser.add_argument("--limit-chunks", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = audit_fallback_forensics(
        chunk_cache_path=args.chunk_cache_path,
        report_path=args.report_path,
        result_log_path=args.result_log_path,
        tables_csv_path=args.tables_csv_path,
        motif_vocab_size=args.motif_vocab_size,
        motif_min_n=args.motif_min_n,
        motif_max_n=args.motif_max_n,
        smoothing_alpha=args.smoothing_alpha,
        rare_max_count=args.rare_max_count,
        top_motif_limit=args.top_motif_limit,
        limit_chunks=args.limit_chunks,
        command=_command(sys.argv[1:]),
    )
    print(
        "fallback_forensic_audit "
        f"research_pass={report.get('pass_criteria', {}).get('research_pass')} "
        f"recommendation={report.get('recommendation')} "
        f"report={report.get('report_path')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
