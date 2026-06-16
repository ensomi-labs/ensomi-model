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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from pulsefield_model.osu_core.beat_chunk_pattern_audit import (
    DEFAULT_BEAT_CHUNK_CACHE_PATH,
    DEFAULT_BEAT_CHUNK_MAP_CACHE_PATH,
    _resolved_beatmap_path,
    _segment_ids,
)
from pulsefield_model.osu_core.beat_chunk_tokenizer_refinement_audit import (
    DEFAULT_MOTIF_MAX_N,
    DEFAULT_MOTIF_MIN_N,
    DEFAULT_RARE_MAX_COUNT,
    DEFAULT_REPORT_PATH as DEFAULT_REFINEMENT_REPORT_PATH,
    DEFAULT_SMOOTHING_ALPHA,
)
from pulsefield_model.osu_core.beat_representation import (
    DEFAULT_BEAT_REPRESENTATION_TIMING_CANONICALIZATION,
    DEFAULT_SNAP_DENOMINATOR,
    BeatEventKind,
    _KIND_ORDER,
    _RawBeatEvent,
    _beat_representation_timing_points,
    _snap_raw_event_to_beat_event,
)
from pulsefield_model.osu_core.beat_representation_audit import DEFAULT_DATASET_ROOT
from pulsefield_model.osu_core.hitobjects import ManiaHitObjectKind, parse_mania_hit_objects
from pulsefield_model.osu_core.timing import require_red_timing_points
from pulsefield_model.timing.canonicalization import require_timing_canonicalization


SCHEMA_VERSION: Final[int] = 1
DEFAULT_MOTIF_VOCAB_SIZE: Final[int] = 16_384
DEFAULT_REPORT_PATH: Final[Path] = Path(
    "artifacts/reports/audits/duration_ln_tokenization/duration_ln_tokenization_report.json",
)
DEFAULT_RESULT_LOG_PATH: Final[Path] = Path(
    "artifacts/reports/audits/duration_ln_tokenization/duration_ln_tokenization_result_log.md",
)
DEFAULT_COMPARISON_CSV_PATH: Final[Path] = Path(
    "artifacts/reports/audits/duration_ln_tokenization/duration_ln_variant_comparison.csv",
)
DEFAULT_RECONSTRUCTION_GUARD_PATH: Final[Path] = Path(
    "artifacts/reports/audits/duration_ln_tokenization/duration_ln_reconstruction_guard.json",
)
DEFAULT_D0_REPORT_PATH: Final[Path] = Path(
    "artifacts/reports/audits/duration_ln_tokenization/duration_ln_d0_diagnostic_report.json",
)
DEFAULT_BOOTSTRAP_SAMPLES: Final[int] = 200
CONTROL_TOKENS: Final[set[str]] = {"C0", "C1"}
PAIR_VARIANTS: Final[set[str]] = {"r3_short_pair", "r3_random_duration", "r3_exact_duration"}
VARIANT_NAMES: Final[tuple[str, ...]] = (
    "r0_delta",
    "r1_duration_start",
    "r3_short_pair",
    "r3_random_duration",
    "r3_exact_duration",
)


@dataclass(frozen=True)
class _MotifVocab:
    motif_to_id: dict[tuple[str, ...], str]
    motif_lengths: dict[str, int]


@dataclass(frozen=True)
class _PairInfo:
    start_key: tuple[int, int]
    end_key: tuple[int, int]
    duration_units: int
    duration_bucket: str


@dataclass(frozen=True)
class _TransformedToken:
    token: str
    source_groups: tuple[dict[str, Any], ...] = ()
    primary_offset: int | None = None
    is_side_info: bool = False


@dataclass
class _EncodedStats:
    token_counter: Counter[str] = field(default_factory=Counter)
    token_count: int = 0
    event_count: int = 0
    group_count: int = 0
    chunk_count: int = 0
    motif_token_count: int = 0
    motif_group_coverage_count: int = 0
    atom_token_count: int = 0
    side_info_token_count: int = 0
    code_bits: float = 0.0
    mapset_bits: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    mapset_events: dict[int, int] = field(default_factory=lambda: defaultdict(int))


@dataclass
class _LnBucketAgg:
    bucket: str
    split: str
    ln_count: int = 0
    event_count: int = 0
    group_refs: int = 0
    fallback_groups: int = 0
    code_bits: float = 0.0
    mapsets: set[int] = field(default_factory=set)

    def add(
        self,
        *,
        beatmap_set_id: int,
        group_refs: int,
        fallback_groups: int,
        code_bits: float,
    ) -> None:
        self.ln_count += 1
        self.event_count += 2
        self.group_refs += int(group_refs)
        self.fallback_groups += int(fallback_groups)
        self.code_bits += float(code_bits)
        self.mapsets.add(int(beatmap_set_id))


def audit_duration_ln_tokenization(
    *,
    chunk_cache_path: str | Path = DEFAULT_BEAT_CHUNK_CACHE_PATH,
    map_cache_path: str | Path = DEFAULT_BEAT_CHUNK_MAP_CACHE_PATH,
    refinement_report_path: str | Path = DEFAULT_REFINEMENT_REPORT_PATH,
    d0_report_path: str | Path = DEFAULT_D0_REPORT_PATH,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    report_path: str | Path | None = DEFAULT_REPORT_PATH,
    result_log_path: str | Path | None = DEFAULT_RESULT_LOG_PATH,
    comparison_csv_path: str | Path | None = DEFAULT_COMPARISON_CSV_PATH,
    reconstruction_guard_path: str | Path | None = DEFAULT_RECONSTRUCTION_GUARD_PATH,
    variant_names: Sequence[str] = VARIANT_NAMES,
    motif_vocab_size: int = DEFAULT_MOTIF_VOCAB_SIZE,
    motif_min_n: int = DEFAULT_MOTIF_MIN_N,
    motif_max_n: int = DEFAULT_MOTIF_MAX_N,
    smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
    rare_max_count: int = DEFAULT_RARE_MAX_COUNT,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    random_seed: int = 17,
    limit_chunks: int | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    """Run chart-only duration-aware LN tokenizer variants and controls."""

    started_at = time.perf_counter()
    chunk_cache_path = Path(chunk_cache_path)
    map_cache_path = Path(map_cache_path)
    refinement_report_path = Path(refinement_report_path)
    d0_report_path = Path(d0_report_path)
    dataset_root = Path(dataset_root)
    report_path = None if report_path is None else Path(report_path)
    result_log_path = None if result_log_path is None else Path(result_log_path)
    comparison_csv_path = None if comparison_csv_path is None else Path(comparison_csv_path)
    reconstruction_guard_path = None if reconstruction_guard_path is None else Path(reconstruction_guard_path)
    motif_vocab_size = _positive_int(motif_vocab_size, "motif_vocab_size")
    motif_min_n = _positive_int(motif_min_n, "motif_min_n")
    motif_max_n = _positive_int(motif_max_n, "motif_max_n")
    if motif_max_n < motif_min_n:
        raise ValueError("motif_max_n must be >= motif_min_n")
    if smoothing_alpha <= 0:
        raise ValueError(f"smoothing_alpha must be positive, got {smoothing_alpha!r}")
    bootstrap_samples = _nonnegative_int(bootstrap_samples, "bootstrap_samples")
    variant_names = _validate_variant_names(variant_names)

    chunk_df = _read_chunk_cache(chunk_cache_path, limit_chunks=limit_chunks)
    map_df = pd.read_parquet(map_cache_path)
    refinement_report = _read_json(refinement_report_path)
    d0_report = _read_json(d0_report_path) if d0_report_path.exists() else {}
    pair_lookup, start_tags, pair_metadata = _build_pair_metadata(
        map_df,
        dataset_root=dataset_root,
        source_ids=set(int(value) for value in chunk_df["source_row_index"].unique()),
    )
    train_distribution = _train_duration_distribution(pair_lookup, split_by_source=_split_by_source(chunk_df))
    random_bucket = _random_bucket_function(train_distribution, seed=random_seed)
    conflict_keys = _cross_split_artist_title_version_keys(chunk_df)
    split_event_counts = {str(split): int(frame["num_events"].sum()) for split, frame in chunk_df.groupby("split", dropna=False)}

    rows: list[dict[str, Any]] = []
    variant_reports: dict[str, dict[str, Any]] = {}
    reconstruction_rows: list[dict[str, Any]] = []
    for variant_name in variant_names:
        variant_report, variant_rows, reconstruction_row = _evaluate_duration_variant(
            chunk_df,
            variant_name=variant_name,
            motif_vocab_size=motif_vocab_size,
            motif_min_n=motif_min_n,
            motif_max_n=motif_max_n,
            smoothing_alpha=smoothing_alpha,
            rare_max_count=rare_max_count,
            pair_lookup=pair_lookup,
            start_tags=start_tags,
            random_bucket=random_bucket,
            conflict_keys=conflict_keys,
        )
        variant_reports[variant_name] = variant_report
        rows.extend(variant_rows)
        reconstruction_rows.append(reconstruction_row)

    selected_variant = _select_variant(variant_reports)
    selected = variant_reports[selected_variant]
    baseline = variant_reports.get("r0_delta", {})
    bootstrap = _bootstrap_mapset_delta(
        selected,
        baseline,
        samples=bootstrap_samples,
        seed=random_seed,
    )
    pass_criteria = _pass_criteria(
        selected_variant=selected_variant,
        selected=selected,
        baseline=baseline,
        reconstruction_rows=reconstruction_rows,
        refinement_report=refinement_report,
    )
    recommendation = _recommendation(pass_criteria, selected_variant=selected_variant)
    comparison_rows = pd.DataFrame(rows)
    if comparison_csv_path is not None:
        comparison_csv_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_rows.to_csv(comparison_csv_path, index=False)

    reconstruction_guard = {
        "schema_version": SCHEMA_VERSION,
        "pass": all(bool(row.get("pass")) for row in reconstruction_rows),
        "rows": reconstruction_rows,
    }
    if reconstruction_guard_path is not None:
        _write_json(reconstruction_guard_path, reconstruction_guard)

    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "duration-aware LN tokenization audit",
        "chunk_cache_path": chunk_cache_path.as_posix(),
        "map_cache_path": map_cache_path.as_posix(),
        "refinement_report_path": refinement_report_path.as_posix(),
        "d0_report_path": d0_report_path.as_posix(),
        "dataset_root": dataset_root.as_posix(),
        "report_path": None if report_path is None else report_path.as_posix(),
        "result_log_path": None if result_log_path is None else result_log_path.as_posix(),
        "comparison_csv_path": None if comparison_csv_path is None else comparison_csv_path.as_posix(),
        "reconstruction_guard_path": None if reconstruction_guard_path is None else reconstruction_guard_path.as_posix(),
        "limited": limit_chunks is not None,
        "limit_chunks": limit_chunks,
        "config": {
            "variant_names": list(variant_names),
            "motif_vocab_size": motif_vocab_size,
            "motif_min_n": motif_min_n,
            "motif_max_n": motif_max_n,
            "smoothing_alpha": smoothing_alpha,
            "rare_max_count": rare_max_count,
            "bootstrap_samples": bootstrap_samples,
            "random_seed": random_seed,
            "selection_policy": "choose non-R0 variant by lowest valid charged bits/event; report selected once on test",
            "duration_thresholds": {
                "micro_max_units": 3,
                "short_max_units": 12,
                "snap_denominator": DEFAULT_SNAP_DENOMINATOR,
            },
            "side_info_policy": "R3 bucket variants emit explicit PX duration residual tokens; exact-duration control encodes exact duration in the pair token.",
        },
        "split_event_counts": split_event_counts,
        "pair_metadata": pair_metadata,
        "d0_reference": _d0_reference(d0_report),
        "variants": variant_reports,
        "selected_variant": selected_variant,
        "bootstrap_mapset_delta": bootstrap,
        "reconstruction_guard": reconstruction_guard,
        "pass_criteria": pass_criteria,
        "recommendation": recommendation,
        "elapsed_s": time.perf_counter() - started_at,
        "code_commit": _git_stdout("rev-parse", "HEAD"),
        "code_dirty": bool(_git_stdout("status", "--porcelain")),
        "command": command,
    }
    report = _normalize_json(report)
    if report_path is not None:
        _write_json(report_path, report)
    if result_log_path is not None:
        _write_result_log(result_log_path, report)
    return report


def _evaluate_duration_variant(
    chunk_df: pd.DataFrame,
    *,
    variant_name: str,
    motif_vocab_size: int,
    motif_min_n: int,
    motif_max_n: int,
    smoothing_alpha: float,
    rare_max_count: int,
    pair_lookup: Mapping[int, Mapping[tuple[tuple[int, int], tuple[int, int]], _PairInfo]],
    start_tags: Mapping[int, Mapping[tuple[int, int], str]],
    random_bucket: Callable[[_PairInfo], str],
    conflict_keys: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    train_atoms = _iter_split_atom_sequences(
        chunk_df,
        variant_name=variant_name,
        split="train",
        pair_lookup=pair_lookup,
        start_tags=start_tags,
        random_bucket=random_bucket,
    )
    atom_counter: Counter[str] = Counter()
    candidates: Counter[tuple[str, ...]] = Counter()
    for tokens in train_atoms:
        atom_counter.update(tokens)
        max_n = min(motif_max_n, len(tokens))
        for ngram in range(motif_min_n, max_n + 1):
            for index in range(len(tokens) - ngram + 1):
                candidates[tuple(tokens[index : index + ngram])] += 1
    motif_candidates = _rank_motifs(candidates)
    vocab = _build_motif_vocab(motif_candidates[:motif_vocab_size])
    dictionary_bits = _dictionary_cost_bits_generic(
        vocab,
        atom_counter,
        smoothing_alpha=smoothing_alpha,
        motif_max_n=motif_max_n,
    )
    train_encoded = _encode_split_generic(
        chunk_df,
        variant_name=variant_name,
        split="train",
        vocab=vocab,
        atom_counter=atom_counter,
        smoothing_alpha=smoothing_alpha,
        pair_lookup=pair_lookup,
        start_tags=start_tags,
        random_bucket=random_bucket,
        dictionary_bits=dictionary_bits,
        rare_max_count=rare_max_count,
    )
    train_counter = train_encoded.token_counter
    model_universe = set(train_counter) | set(atom_counter) | set(vocab.motif_lengths) | CONTROL_TOKENS
    token_bits = _make_model_token_bits(
        train_counter,
        atom_counter,
        model_universe,
        smoothing_alpha=smoothing_alpha,
    )
    split_reports: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    bucket_tables: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "valid", "test"):
        encoded = train_encoded if split == "train" else _encode_split_generic(
            chunk_df,
            variant_name=variant_name,
            split=split,
            vocab=vocab,
            atom_counter=atom_counter,
            smoothing_alpha=smoothing_alpha,
            pair_lookup=pair_lookup,
            start_tags=start_tags,
            random_bucket=random_bucket,
            dictionary_bits=dictionary_bits,
            rare_max_count=rare_max_count,
            token_bits=token_bits,
        )
        if split == "train":
            _score_encoded_generic(encoded, token_bits=token_bits, dictionary_bits=dictionary_bits, rare_max_count=rare_max_count)
        split_reports[split] = _split_report(encoded, dictionary_bits=dictionary_bits)
        comparison_rows.append(_comparison_row(variant_name, split, split_reports[split]))
        bucket_rows = _ln_bucket_rows(
            encoded,
            split=split,
            dictionary_bits=dictionary_bits,
        )
        bucket_tables[split] = bucket_rows
        comparison_rows.extend(
            _comparison_row(variant_name, split, row, table="ln_bucket")
            for row in bucket_rows
        )

    same_song_filtered = _same_song_filtered_score(
        chunk_df,
        variant_name=variant_name,
        vocab=vocab,
        atom_counter=atom_counter,
        token_bits=token_bits,
        dictionary_bits=dictionary_bits,
        pair_lookup=pair_lookup,
        start_tags=start_tags,
        random_bucket=random_bucket,
        conflict_keys=conflict_keys,
        rare_max_count=rare_max_count,
    )
    reconstruction = _reconstruction_guard(
        chunk_df,
        variant_name=variant_name,
        vocab=vocab,
        pair_lookup=pair_lookup,
        start_tags=start_tags,
        random_bucket=random_bucket,
    )
    split_reports["same_song_filtered_test"] = same_song_filtered
    report = {
        "variant": variant_name,
        "motif_vocab_size": motif_vocab_size,
        "learned_motif_count": len(vocab.motif_to_id),
        "dictionary_cost_bits": dictionary_bits,
        "splits": split_reports,
        "ln_bucket_tables": bucket_tables,
        "reconstruction": reconstruction,
        "top_motifs": _top_motif_rows(motif_candidates, limit=20),
    }
    reconstruction_row = {
        "variant": variant_name,
        "checked_chunk_count": reconstruction["checked_chunk_count"],
        "mismatch_count": reconstruction["mismatch_count"],
        "pass": reconstruction["pass"],
        "examples": reconstruction["examples"],
    }
    return report, comparison_rows, reconstruction_row


def _encode_split_generic(
    chunk_df: pd.DataFrame,
    *,
    variant_name: str,
    split: str,
    vocab: _MotifVocab,
    atom_counter: Counter[str],
    smoothing_alpha: float,
    pair_lookup: Mapping[int, Mapping[tuple[tuple[int, int], tuple[int, int]], _PairInfo]],
    start_tags: Mapping[int, Mapping[tuple[int, int], str]],
    random_bucket: Callable[[_PairInfo], str],
    dictionary_bits: float,
    rare_max_count: int,
    token_bits: Callable[[str], float] | None = None,
) -> _EncodedStats:
    del atom_counter, smoothing_alpha, dictionary_bits, rare_max_count
    frame = chunk_df[chunk_df["split"].fillna("").astype(str) == split]
    trie = _build_motif_trie(vocab.motif_to_id)
    stats = _EncodedStats()
    bucket_aggs: dict[str, _LnBucketAgg] = {}
    for row in frame.itertuples(index=False):
        transformed = _transform_row(
            row,
            variant_name=variant_name,
            pair_lookup=pair_lookup,
            start_tags=start_tags,
            random_bucket=random_bucket,
        )
        source_keys_by_index = [_source_keys_for_token(token) for token in transformed]
        stats.chunk_count += 1
        stats.event_count += int(row.num_events)
        stats.group_count += int(row.num_groups)
        stats.mapset_events[int(row.beatmap_set_id)] += int(row.num_events)
        sequence = [f"C{int(row.bar_phase_half)}", *(token.token for token in transformed)]
        source_keys_by_index = [set(), *source_keys_by_index]
        encoded, covered_indexes = _encode_sequence(sequence, trie)
        for token in encoded:
            stats.token_counter[token] += 1
            stats.token_count += 1
            if token.startswith("M"):
                stats.motif_token_count += 1
            else:
                stats.atom_token_count += 1
                if token.startswith("PX:"):
                    stats.side_info_token_count += 1
        covered_group_count = 0
        for index in covered_indexes:
            covered_group_count += len(source_keys_by_index[index])
        stats.motif_group_coverage_count += covered_group_count
        if token_bits is not None:
            code_bits = sum(token_bits(token) for token in encoded)
            stats.code_bits += code_bits
            stats.mapset_bits[int(row.beatmap_set_id)] += code_bits
            _accumulate_ln_bucket_aggs(
                bucket_aggs,
                row=row,
                sequence=sequence,
                source_keys_by_index=source_keys_by_index,
                covered_indexes=covered_indexes,
                token_bits=token_bits,
                pair_lookup=pair_lookup,
            )
    stats.ln_bucket_aggs = bucket_aggs  # type: ignore[attr-defined]
    return stats


def _score_encoded_generic(
    stats: _EncodedStats,
    *,
    token_bits: Callable[[str], float],
    dictionary_bits: float,
    rare_max_count: int,
) -> None:
    del dictionary_bits, rare_max_count
    if stats.code_bits:
        return
    for token, count in stats.token_counter.items():
        stats.code_bits += float(count) * token_bits(token)


def _split_report(stats: _EncodedStats, *, dictionary_bits: float) -> dict[str, Any]:
    event_count = int(stats.event_count)
    group_count = int(stats.group_count)
    return {
        "event_count": event_count,
        "group_count": group_count,
        "chunk_count": int(stats.chunk_count),
        "encoded_token_count": int(stats.token_count),
        "motif_token_count": int(stats.motif_token_count),
        "atom_token_count": int(stats.atom_token_count),
        "side_info_token_count": int(stats.side_info_token_count),
        "motif_group_coverage_count": int(stats.motif_group_coverage_count),
        "motif_coverage_rate": float(stats.motif_group_coverage_count) / float(group_count) if group_count else 0.0,
        "fallback_group_rate": float(group_count - stats.motif_group_coverage_count) / float(group_count) if group_count else 0.0,
        "code_bits": float(stats.code_bits),
        "bits_per_event": float(stats.code_bits) / float(event_count) if event_count else 0.0,
        "dictionary_cost_bits": float(dictionary_bits),
        "dictionary_bits_per_event": float(dictionary_bits) / float(event_count) if event_count else 0.0,
        "bits_per_event_with_dictionary": float(stats.code_bits + dictionary_bits) / float(event_count) if event_count else 0.0,
        "tokens_per_event": float(stats.token_count) / float(event_count) if event_count else 0.0,
        "mapset_bits": dict(stats.mapset_bits),
        "mapset_events": dict(stats.mapset_events),
    }


def _iter_split_atom_sequences(
    chunk_df: pd.DataFrame,
    *,
    variant_name: str,
    split: str,
    pair_lookup: Mapping[int, Mapping[tuple[tuple[int, int], tuple[int, int]], _PairInfo]],
    start_tags: Mapping[int, Mapping[tuple[int, int], str]],
    random_bucket: Callable[[_PairInfo], str],
) -> Iterable[list[str]]:
    frame = chunk_df[chunk_df["split"].fillna("").astype(str) == split]
    for row in frame.itertuples(index=False):
        transformed = _transform_row(
            row,
            variant_name=variant_name,
            pair_lookup=pair_lookup,
            start_tags=start_tags,
            random_bucket=random_bucket,
        )
        yield [f"C{int(row.bar_phase_half)}", *(token.token for token in transformed)]


def _transform_row(
    row: Any,
    *,
    variant_name: str,
    pair_lookup: Mapping[int, Mapping[tuple[tuple[int, int], tuple[int, int]], _PairInfo]],
    start_tags: Mapping[int, Mapping[tuple[int, int], str]],
    random_bucket: Callable[[_PairInfo], str],
) -> list[_TransformedToken]:
    groups = json.loads(getattr(row, "groups_json"))
    source = int(row.source_row_index)
    segment_id = int(row.segment_id)
    start_units = int(row.start_beat_units)
    group_keys = [(segment_id, start_units + int(group["offset_units"])) for group in groups]
    key_to_index = {key: index for index, key in enumerate(group_keys)}
    start_to_pair: dict[int, tuple[int, _PairInfo]] = {}
    skipped: set[int] = set()
    if variant_name in PAIR_VARIANTS:
        pair_candidates = []
        for pair_info in pair_lookup.get(source, {}).values():
            start_index = key_to_index.get(pair_info.start_key)
            end_index = key_to_index.get(pair_info.end_key)
            if start_index is None or end_index is None or start_index == end_index:
                continue
            pair_candidates.append((start_index, end_index, pair_info.duration_units, pair_info))
        pair_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        used_any: set[int] = set()
        for start_index, end_index, _duration, pair_info in pair_candidates:
            if start_index in used_any or end_index in used_any:
                continue
            used_any.add(start_index)
            used_any.add(end_index)
            skipped.add(end_index)
            start_to_pair[start_index] = (end_index, pair_info)

    transformed: list[_TransformedToken] = []
    previous_offset = 0
    for index, group in enumerate(groups):
        if index in skipped:
            continue
        offset = int(group["offset_units"])
        delta = offset - previous_offset
        previous_offset = offset
        if index in start_to_pair:
            end_index, pair_info = start_to_pair[index]
            end_group = groups[end_index]
            transformed.extend(
                _pair_tokens(
                    delta=delta,
                    start_group=group,
                    end_group=end_group,
                    pair_info=pair_info,
                    variant_name=variant_name,
                    random_bucket=random_bucket,
                )
            )
            continue
        token = _delta_group_token(delta, group)
        if variant_name == "r1_duration_start":
            tag = start_tags.get(source, {}).get(group_keys[index])
            if tag:
                token = f"{token}:B{tag}"
        transformed.append(
            _TransformedToken(
                token=token,
                source_groups=(_group_with_offset(group, offset),),
                primary_offset=offset,
            ),
        )
    return transformed


def _pair_tokens(
    *,
    delta: int,
    start_group: Mapping[str, Any],
    end_group: Mapping[str, Any],
    pair_info: _PairInfo,
    variant_name: str,
    random_bucket: Callable[[_PairInfo], str],
) -> list[_TransformedToken]:
    start_payload = _group_payload(start_group)
    end_payload = _group_payload(end_group)
    if variant_name == "r3_exact_duration":
        token = f"PE:{delta}:{pair_info.duration_units}:{start_payload}:{end_payload}"
        return [
            _TransformedToken(
                token=token,
                source_groups=(
                    _group_with_offset(start_group, int(start_group["offset_units"])),
                    _group_with_offset(end_group, int(end_group["offset_units"])),
                ),
                primary_offset=int(start_group["offset_units"]),
            ),
        ]
    bucket = pair_info.duration_bucket if variant_name == "r3_short_pair" else random_bucket(pair_info)
    pair_token = f"P:{delta}:{bucket}:{start_payload}:{end_payload}"
    residual_token = f"PX:{pair_info.duration_units}"
    return [
        _TransformedToken(
            token=pair_token,
            source_groups=(
                _group_with_offset(start_group, int(start_group["offset_units"])),
                _group_with_offset(end_group, int(end_group["offset_units"])),
            ),
            primary_offset=int(start_group["offset_units"]),
        ),
        _TransformedToken(token=residual_token, is_side_info=True),
    ]


def _source_keys_for_token(token: _TransformedToken) -> set[tuple[int, int, int, int]]:
    keys = set()
    for group in token.source_groups:
        keys.add(
            (
                int(group["offset_units"]),
                int(group["tap_mask"]),
                int(group["ln_start_mask"]),
                int(group["ln_end_mask"]),
            ),
        )
    return keys


def _accumulate_ln_bucket_aggs(
    bucket_aggs: dict[str, _LnBucketAgg],
    *,
    row: Any,
    sequence: Sequence[str],
    source_keys_by_index: Sequence[set[tuple[int, int, int, int]]],
    covered_indexes: set[int],
    token_bits: Callable[[str], float],
    pair_lookup: Mapping[int, Mapping[tuple[tuple[int, int], tuple[int, int]], _PairInfo]],
) -> None:
    del pair_lookup
    groups = json.loads(getattr(row, "groups_json"))
    group_to_index: dict[tuple[int, int, int, int], int] = {}
    for index, keys in enumerate(source_keys_by_index):
        for key in keys:
            group_to_index[key] = index
    for group in groups:
        if not int(group["ln_start_mask"]) and not int(group["ln_end_mask"]):
            continue
        key = (
            int(group["offset_units"]),
            int(group["tap_mask"]),
            int(group["ln_start_mask"]),
            int(group["ln_end_mask"]),
        )
        index = group_to_index.get(key)
        if index is None:
            continue
        bucket = _ln_group_bucket(group)
        agg = bucket_aggs.setdefault(bucket, _LnBucketAgg(bucket=bucket, split=str(row.split)))
        fallback = index not in covered_indexes
        event_count = max(1, int(group.get("event_count", 1)))
        agg.add(
            beatmap_set_id=int(row.beatmap_set_id),
            group_refs=1,
            fallback_groups=int(fallback),
            code_bits=token_bits(sequence[index]) / float(event_count),
        )


def _ln_bucket_rows(stats: _EncodedStats, *, split: str, dictionary_bits: float) -> list[dict[str, Any]]:
    aggs: Mapping[str, _LnBucketAgg] = getattr(stats, "ln_bucket_aggs", {})
    rows = []
    dict_bits_per_event = float(dictionary_bits) / float(stats.event_count) if stats.event_count else 0.0
    for bucket, agg in sorted(aggs.items()):
        rows.append(
            {
                "bucket": bucket,
                "split": split,
                "ln_group_count": int(agg.group_refs),
                "ln_event_count": int(agg.event_count),
                "num_mapsets": len(agg.mapsets),
                "fallback_group_rate": float(agg.fallback_groups) / float(agg.group_refs) if agg.group_refs else 0.0,
                "bits_per_event": float(agg.code_bits) / float(agg.event_count) if agg.event_count else 0.0,
                "bits_per_event_with_dictionary": (
                    float(agg.code_bits) / float(agg.event_count) + dict_bits_per_event if agg.event_count else 0.0
                ),
            }
        )
    return rows


def _ln_group_bucket(group: Mapping[str, Any]) -> str:
    active = int(group["tap_mask"]) | int(group["ln_start_mask"]) | int(group["ln_end_mask"])
    mixed = active.bit_count() > 1
    if int(group["ln_start_mask"]) and int(group["ln_end_mask"]):
        kind = "ln_start_end"
    elif int(group["ln_start_mask"]):
        kind = "ln_start"
    else:
        kind = "ln_end"
    return f"{kind}|{'chord_ln_mixed' if mixed else 'ln_only'}"


def _encode_sequence(tokens: Sequence[str], trie: Mapping[str, Any]) -> tuple[list[str], set[int]]:
    encoded: list[str] = []
    covered_indexes: set[int] = set()
    index = 0
    while index < len(tokens):
        matched_token, matched_length = _match_motif(tokens, index, trie)
        if matched_token is not None:
            encoded.append(matched_token)
            covered_indexes.update(range(index, index + matched_length))
            index += matched_length
            continue
        encoded.append(tokens[index])
        index += 1
    return encoded, covered_indexes


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


def _build_motif_trie(motif_to_id: Mapping[tuple[str, ...], str]) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for motif, token_id in motif_to_id.items():
        node = root
        for token in motif:
            node = node.setdefault(token, {})
        node["__id__"] = token_id
        node["__length__"] = len(motif)
    return root


def _build_motif_vocab(candidates: Sequence[tuple[tuple[str, ...], int, float]]) -> _MotifVocab:
    motif_to_id = {motif: f"M{index}" for index, (motif, _count, _gain) in enumerate(candidates)}
    motif_lengths = {f"M{index}": len(motif) for index, (motif, _count, _gain) in enumerate(candidates)}
    return _MotifVocab(motif_to_id=motif_to_id, motif_lengths=motif_lengths)


def _rank_motifs(counter: Counter[tuple[str, ...]]) -> list[tuple[tuple[str, ...], int, float]]:
    candidates = []
    for motif, count in counter.items():
        if count < 2:
            continue
        gain = float(count) * float(len(motif) - 1)
        candidates.append((motif, int(count), gain))
    candidates.sort(key=lambda item: (-item[2], -item[1], -len(item[0]), item[0]))
    return candidates


def _dictionary_cost_bits_generic(
    vocab: _MotifVocab,
    atom_counter: Counter[str],
    *,
    smoothing_alpha: float,
    motif_max_n: int,
) -> float:
    if not vocab.motif_to_id:
        return 0.0
    token_bits = _make_atom_token_bits(atom_counter, smoothing_alpha=smoothing_alpha)
    length_bits = math.log2(max(2, int(motif_max_n) + 1))
    bits = 0.0
    for motif in vocab.motif_to_id:
        bits += length_bits
        bits += sum(token_bits(token) for token in motif)
    return bits


def _make_atom_token_bits(atom_counter: Counter[str], *, smoothing_alpha: float) -> Callable[[str], float]:
    total = sum(atom_counter.values())
    vocab_size = len(atom_counter) + 1
    denominator = float(total) + smoothing_alpha * float(vocab_size)
    escape_probability = smoothing_alpha / denominator if denominator > 0 else 1.0
    escape_bits = -math.log2(escape_probability) if escape_probability > 0 else 0.0

    def token_bits(token: str) -> float:
        count = atom_counter.get(token, 0)
        if count <= 0:
            return escape_bits + _payload_bits(token)
        probability = (float(count) + smoothing_alpha) / denominator if denominator > 0 else 1.0
        return -math.log2(probability) if probability > 0 else 0.0

    return token_bits


def _make_model_token_bits(
    train_counter: Counter[str],
    atom_counter: Counter[str],
    model_universe: set[str],
    *,
    smoothing_alpha: float,
) -> Callable[[str], float]:
    total = sum(train_counter.values())
    vocab_size = len(model_universe) + 1
    denominator = float(total) + smoothing_alpha * float(vocab_size)
    escape_probability = smoothing_alpha / denominator if denominator > 0 else 1.0
    escape_bits = -math.log2(escape_probability) if escape_probability > 0 else 0.0
    cache: dict[str, float] = {}

    def token_bits(token: str) -> float:
        cached = cache.get(token)
        if cached is not None:
            return cached
        count = train_counter.get(token, 0)
        if token not in model_universe and token not in atom_counter:
            value = escape_bits + _payload_bits(token)
        else:
            probability = (float(count) + smoothing_alpha) / denominator if denominator > 0 else 1.0
            value = -math.log2(probability) if probability > 0 else 0.0
        cache[token] = value
        return value

    return token_bits


def _payload_bits(token: str) -> float:
    return float(max(1, len(token.encode("utf-8"))) * 8)


def _build_pair_metadata(
    map_df: pd.DataFrame,
    *,
    dataset_root: Path,
    source_ids: set[int],
) -> tuple[
    dict[int, dict[tuple[tuple[int, int], tuple[int, int]], _PairInfo]],
    dict[int, dict[tuple[int, int], str]],
    dict[str, Any],
]:
    pair_lookup: dict[int, dict[tuple[tuple[int, int], tuple[int, int]], _PairInfo]] = defaultdict(dict)
    start_tags: dict[int, dict[tuple[int, int], str]] = defaultdict(dict)
    parse_errors = []
    hold_count = 0
    eligible_count = 0
    for row in map_df.itertuples(index=False):
        source = int(row.source_row_index)
        if source not in source_ids:
            continue
        if not bool(row.hard_gate_ok) or float(row.ln_ratio or 0.0) <= 0.0:
            continue
        pairs, error = _hold_pairs_for_map(row, dataset_root)
        if error is not None:
            parse_errors.append({"source_row_index": source, "beatmap_path": str(row.beatmap_path), "error": error})
            continue
        hold_count += len(pairs)
        for pair in pairs:
            duration_units = pair["duration_units"]
            if duration_units is None:
                continue
            tag = _duration_bucket(duration_units)
            start_tags[source][pair["start_key"]] = tag
            if tag not in {"micro", "short"}:
                continue
            info = _PairInfo(
                start_key=pair["start_key"],
                end_key=pair["end_key"],
                duration_units=int(duration_units),
                duration_bucket=tag,
            )
            pair_lookup[source][(info.start_key, info.end_key)] = info
            eligible_count += 1
    metadata = {
        "hold_count": hold_count,
        "eligible_short_pair_count": eligible_count,
        "source_count_with_pairs": len(pair_lookup),
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors[:20],
    }
    return dict(pair_lookup), dict(start_tags), metadata


def _hold_pairs_for_map(row: Any, dataset_root: Path) -> tuple[list[dict[str, Any]], str | None]:
    path = _resolved_beatmap_path(row, dataset_root)
    try:
        hitobjects = parse_mania_hit_objects(path, expected_key_count=4)
        timing = require_red_timing_points(path)
        canonicalization = require_timing_canonicalization(DEFAULT_BEAT_REPRESENTATION_TIMING_CANONICALIZATION)
        timing_points = _beat_representation_timing_points(timing, canonicalization=canonicalization)
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"
    raw_events: list[_RawBeatEvent] = []
    for index, hitobject in enumerate(hitobjects):
        if hitobject.kind == ManiaHitObjectKind.TAP:
            raw_events.append(_RawBeatEvent(kind=BeatEventKind.TAP, lane=hitobject.lane, time_ms=hitobject.start_time_ms, source_index=index))
        elif hitobject.kind == ManiaHitObjectKind.HOLD:
            raw_events.append(_RawBeatEvent(kind=BeatEventKind.HOLD_START, lane=hitobject.lane, time_ms=hitobject.start_time_ms, source_index=index))
            raw_events.append(_RawBeatEvent(kind=BeatEventKind.HOLD_END, lane=hitobject.lane, time_ms=hitobject.end_time_ms, source_index=index))
    raw_events.sort(key=lambda event: (event.time_ms, _KIND_ORDER[event.kind], event.lane, event.source_index))
    events = [
        _snap_raw_event_to_beat_event(
            event,
            timing_points,
            snap_denominator=DEFAULT_SNAP_DENOMINATOR,
            include_diagnostics=False,
            diagnostic_subdivisions=(),
        )
        for event in raw_events
    ]
    segment_ids = _segment_ids(events)
    by_source_kind: dict[tuple[int, BeatEventKind], tuple[Any, int]] = {}
    for raw_event, event, segment_id in zip(raw_events, events, segment_ids, strict=True):
        by_source_kind[(int(raw_event.source_index), raw_event.kind)] = (event, int(segment_id))
    pairs = []
    for index, hitobject in enumerate(hitobjects):
        if hitobject.kind != ManiaHitObjectKind.HOLD:
            continue
        start = by_source_kind.get((index, BeatEventKind.HOLD_START))
        end = by_source_kind.get((index, BeatEventKind.HOLD_END))
        if start is None or end is None:
            continue
        start_event, start_segment = start
        end_event, end_segment = end
        same_segment = int(start_segment) == int(end_segment)
        duration_units = int(end_event.beat_offset_numerator) - int(start_event.beat_offset_numerator) if same_segment else None
        pairs.append(
            {
                "duration_units": duration_units,
                "start_key": (int(start_segment), int(start_event.beat_offset_numerator)),
                "end_key": (int(end_segment), int(end_event.beat_offset_numerator)),
            }
        )
    return pairs, None


def _duration_bucket(duration_units: int) -> str:
    if duration_units <= 3:
        return "micro"
    if duration_units <= 12:
        return "short"
    if duration_units <= 48:
        return "medium"
    return "long"


def _train_duration_distribution(
    pair_metadata: Mapping[int, Mapping[tuple[tuple[int, int], tuple[int, int]], _PairInfo]],
    *,
    split_by_source: Mapping[int, str],
) -> tuple[str, ...]:
    buckets = []
    for source, pairs in pair_metadata.items():
        if split_by_source.get(int(source)) != "train":
            continue
        buckets.extend(pair.duration_bucket for pair in pairs.values())
    return tuple(buckets) or ("micro", "short")


def _random_bucket_function(train_distribution: Sequence[str], *, seed: int) -> Callable[[_PairInfo], str]:
    distribution = tuple(train_distribution)

    def random_bucket(pair_info: _PairInfo) -> str:
        key = f"{pair_info.start_key}:{pair_info.end_key}:{pair_info.duration_units}:{seed}".encode()
        digest = hashlib.sha256(key).digest()
        index = int.from_bytes(digest[:8], "big") % len(distribution)
        return distribution[index]

    return random_bucket


def _reconstruction_guard(
    chunk_df: pd.DataFrame,
    *,
    variant_name: str,
    vocab: _MotifVocab,
    pair_lookup: Mapping[int, Mapping[tuple[tuple[int, int], tuple[int, int]], _PairInfo]],
    start_tags: Mapping[int, Mapping[tuple[int, int], str]],
    random_bucket: Callable[[_PairInfo], str],
) -> dict[str, Any]:
    id_to_motif = {token_id: motif for motif, token_id in vocab.motif_to_id.items()}
    trie = _build_motif_trie(vocab.motif_to_id)
    checked = 0
    mismatches = 0
    examples = []
    for row in chunk_df.itertuples(index=False):
        transformed = _transform_row(
            row,
            variant_name=variant_name,
            pair_lookup=pair_lookup,
            start_tags=start_tags,
            random_bucket=random_bucket,
        )
        sequence = [token.token for token in transformed]
        encoded, _covered = _encode_sequence(sequence, trie)
        expanded = _expand_motifs(encoded, id_to_motif)
        reconstructed = _signature_from_transformed_tokens(expanded)
        expected = str(row.raw_signature or "")
        checked += 1
        if reconstructed == expected:
            continue
        mismatches += 1
        if len(examples) < 10:
            examples.append(
                {
                    "source_row_index": int(row.source_row_index),
                    "chunk_index": int(row.chunk_index),
                    "split": str(row.split),
                    "expected": expected,
                    "reconstructed": reconstructed,
                    "encoded_tokens": encoded[:20],
                }
            )
    return {"checked_chunk_count": checked, "mismatch_count": mismatches, "pass": mismatches == 0 and checked > 0, "examples": examples}


def _expand_motifs(tokens: Sequence[str], id_to_motif: Mapping[str, tuple[str, ...]]) -> list[str]:
    expanded = []
    for token in tokens:
        motif = id_to_motif.get(token)
        if motif is None:
            expanded.append(token)
        else:
            expanded.extend(motif)
    return expanded


def _signature_from_transformed_tokens(tokens: Sequence[str]) -> str:
    groups = []
    previous_offset = 0
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("PX:"):
            index += 1
            continue
        if token.startswith("D:"):
            group, previous_offset = _parse_delta_token(token, previous_offset)
            groups.append(group)
            index += 1
            continue
        if token.startswith("P:"):
            if index + 1 >= len(tokens) or not tokens[index + 1].startswith("PX:"):
                raise ValueError(f"paired token missing PX residual: {token!r}")
            parsed, previous_offset = _parse_pair_token(token, previous_offset, duration_token=tokens[index + 1])
            groups.extend(parsed)
            index += 2
            continue
        if token.startswith("PE:"):
            parsed, previous_offset = _parse_exact_pair_token(token, previous_offset)
            groups.extend(parsed)
            index += 1
            continue
        if token.startswith("C"):
            index += 1
            continue
        raise ValueError(f"unsupported transformed token: {token!r}")
    groups.sort(key=lambda group: int(group["offset_units"]))
    return ";".join(_atomic_token(group) for group in groups)


def _parse_delta_token(token: str, previous_offset: int) -> tuple[dict[str, Any], int]:
    parts = token.split(":")
    if len(parts) < 5:
        raise ValueError(f"bad delta token: {token!r}")
    delta = int(parts[1])
    offset = previous_offset + delta
    group = {
        "offset_units": offset,
        "tap_mask": int(parts[2]),
        "ln_start_mask": int(parts[3]),
        "ln_end_mask": int(parts[4]),
        "order_signature": ".",
    }
    for part in parts[5:]:
        if part.startswith("O"):
            group["order_signature"] = part[1:]
    return group, offset


def _parse_pair_token(token: str, previous_offset: int, *, duration_token: str) -> tuple[list[dict[str, Any]], int]:
    parts = token.split(":")
    if len(parts) != 11:
        raise ValueError(f"bad pair token: {token!r}")
    delta = int(parts[1])
    start_offset = previous_offset + delta
    duration = int(duration_token.split(":", 1)[1])
    start_group = _group_from_payload(start_offset, parts[3:7])
    end_group = _group_from_payload(start_offset + duration, parts[7:11])
    return [start_group, end_group], start_offset


def _parse_exact_pair_token(token: str, previous_offset: int) -> tuple[list[dict[str, Any]], int]:
    parts = token.split(":")
    if len(parts) != 11:
        raise ValueError(f"bad exact pair token: {token!r}")
    delta = int(parts[1])
    duration = int(parts[2])
    start_offset = previous_offset + delta
    start_group = _group_from_payload(start_offset, parts[3:7])
    end_group = _group_from_payload(start_offset + duration, parts[7:11])
    return [start_group, end_group], start_offset


def _group_from_payload(offset: int, payload: Sequence[str]) -> dict[str, Any]:
    return {
        "offset_units": offset,
        "tap_mask": int(payload[0]),
        "ln_start_mask": int(payload[1]),
        "ln_end_mask": int(payload[2]),
        "order_signature": "." if payload[3] == "_" else payload[3],
    }


def _group_payload(group: Mapping[str, Any]) -> str:
    order = str(group.get("order_signature", ".") or ".")
    return "{tap}:{start}:{end}:{order}".format(
        tap=int(group["tap_mask"]),
        start=int(group["ln_start_mask"]),
        end=int(group["ln_end_mask"]),
        order="_" if order == "." else order,
    )


def _group_with_offset(group: Mapping[str, Any], offset: int) -> dict[str, Any]:
    item = dict(group)
    item["offset_units"] = int(offset)
    return item


def _delta_group_token(delta: int, group: Mapping[str, Any]) -> str:
    token = "D:{delta}:{tap}:{start}:{end}".format(
        delta=int(delta),
        tap=int(group["tap_mask"]),
        start=int(group["ln_start_mask"]),
        end=int(group["ln_end_mask"]),
    )
    order = str(group.get("order_signature", ".") or ".")
    return token if order == "." else f"{token}:O{order}"


def _atomic_token(group: Mapping[str, Any]) -> str:
    token = "A:{offset}:{tap}:{start}:{end}".format(
        offset=int(group["offset_units"]),
        tap=int(group["tap_mask"]),
        start=int(group["ln_start_mask"]),
        end=int(group["ln_end_mask"]),
    )
    order = str(group.get("order_signature", ".") or ".")
    return token if order == "." else f"{token}:O{order}"


def _same_song_filtered_score(
    chunk_df: pd.DataFrame,
    *,
    variant_name: str,
    vocab: _MotifVocab,
    atom_counter: Counter[str],
    token_bits: Callable[[str], float],
    dictionary_bits: float,
    pair_lookup: Mapping[int, Mapping[tuple[tuple[int, int], tuple[int, int]], _PairInfo]],
    start_tags: Mapping[int, Mapping[tuple[int, int], str]],
    random_bucket: Callable[[_PairInfo], str],
    conflict_keys: set[str],
    rare_max_count: int,
) -> dict[str, Any]:
    del atom_counter
    if not conflict_keys:
        return {}
    frame = chunk_df.copy()
    frame["_artist_title_version_key"] = _key_series(frame, ["artist", "title", "version"])
    filtered = frame[~frame["_artist_title_version_key"].isin(conflict_keys)].drop(columns=["_artist_title_version_key"])
    encoded = _encode_split_generic(
        filtered,
        variant_name=variant_name,
        split="test",
        vocab=vocab,
        atom_counter=Counter(),
        smoothing_alpha=DEFAULT_SMOOTHING_ALPHA,
        pair_lookup=pair_lookup,
        start_tags=start_tags,
        random_bucket=random_bucket,
        dictionary_bits=dictionary_bits,
        rare_max_count=rare_max_count,
        token_bits=token_bits,
    )
    return _split_report(encoded, dictionary_bits=dictionary_bits)


def _bootstrap_mapset_delta(
    selected: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples <= 0:
        return {"samples": 0, "available": False}
    selected_test = selected.get("splits", {}).get("test", {})
    baseline_test = baseline.get("splits", {}).get("test", {})
    selected_bits = {int(k): float(v) for k, v in selected_test.get("mapset_bits", {}).items()}
    selected_events = {int(k): int(v) for k, v in selected_test.get("mapset_events", {}).items()}
    baseline_bits = {int(k): float(v) for k, v in baseline_test.get("mapset_bits", {}).items()}
    baseline_events = {int(k): int(v) for k, v in baseline_test.get("mapset_events", {}).items()}
    mapsets = sorted(set(selected_bits) & set(baseline_bits))
    if not mapsets:
        return {"samples": samples, "available": False, "reason": "no overlapping test mapsets"}
    selected_dict_bpe = float(selected_test.get("dictionary_bits_per_event", 0.0))
    baseline_dict_bpe = float(baseline_test.get("dictionary_bits_per_event", 0.0))
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        sample = [mapsets[rng.randrange(len(mapsets))] for _ in mapsets]
        selected_total_bits = sum(selected_bits[item] + selected_events[item] * selected_dict_bpe for item in sample)
        baseline_total_bits = sum(baseline_bits[item] + baseline_events[item] * baseline_dict_bpe for item in sample)
        events = sum(selected_events[item] for item in sample)
        if events:
            deltas.append((selected_total_bits - baseline_total_bits) / float(events))
    if not deltas:
        return {"samples": samples, "available": False, "reason": "empty bootstrap deltas"}
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "samples": samples,
        "available": True,
        "mapset_count": len(mapsets),
        "delta_bits_per_event_mean": float(values.mean()),
        "delta_bits_per_event_p025": float(np.quantile(values, 0.025)),
        "delta_bits_per_event_p975": float(np.quantile(values, 0.975)),
    }


def _select_variant(variant_reports: Mapping[str, Mapping[str, Any]]) -> str:
    candidates = [name for name in variant_reports if name != "r0_delta"]
    if not candidates:
        return "r0_delta"
    return min(
        candidates,
        key=lambda name: float(
            variant_reports[name].get("splits", {}).get("valid", {}).get("bits_per_event_with_dictionary", math.inf)
        ),
    )


def _pass_criteria(
    *,
    selected_variant: str,
    selected: Mapping[str, Any],
    baseline: Mapping[str, Any],
    reconstruction_rows: Sequence[Mapping[str, Any]],
    refinement_report: Mapping[str, Any],
) -> dict[str, Any]:
    selected_test = selected.get("splits", {}).get("test", {})
    selected_valid = selected.get("splits", {}).get("valid", {})
    baseline_test = baseline.get("splits", {}).get("test", {})
    reconstruction_pass = all(bool(row.get("pass")) for row in reconstruction_rows)
    selected_test_bits = float(selected_test.get("bits_per_event_with_dictionary", math.inf))
    baseline_test_bits = float(baseline_test.get("bits_per_event_with_dictionary", math.inf))
    global_nonregress = selected_test_bits <= baseline_test_bits + 0.01
    known_r0_bits = float(
        refinement_report.get("pass_criteria", {}).get("valid_selected_test_bits_per_event_with_dictionary", math.inf)
    )
    baseline_consistency = abs(baseline_test_bits - known_r0_bits) < 0.50 if math.isfinite(known_r0_bits) else True
    short_selected = _bucket_by_name(selected, "test", "ln_start|chord_ln_mixed")
    short_baseline = _bucket_by_name(baseline, "test", "ln_start|chord_ln_mixed")
    bucket_improves = False
    fallback_improves = False
    if short_selected and short_baseline:
        bucket_improves = float(short_selected["bits_per_event_with_dictionary"]) <= (
            float(short_baseline["bits_per_event_with_dictionary"]) - 0.05
        )
        fallback_improves = float(short_selected["fallback_group_rate"]) <= (
            float(short_baseline["fallback_group_rate"]) * 0.90
        )
    positive = bool(reconstruction_pass and global_nonregress and (bucket_improves or fallback_improves))
    return {
        "selected_variant": selected_variant,
        "reconstruction_pass": reconstruction_pass,
        "baseline_consistency_pass": baseline_consistency,
        "global_nonregress_pass": global_nonregress,
        "short_chord_ln_bucket_bits_improve": bucket_improves,
        "short_chord_ln_fallback_improve": fallback_improves,
        "valid_selected_bits_per_event_with_dictionary": selected_valid.get("bits_per_event_with_dictionary"),
        "test_selected_bits_per_event_with_dictionary": selected_test.get("bits_per_event_with_dictionary"),
        "test_baseline_bits_per_event_with_dictionary": baseline_test.get("bits_per_event_with_dictionary"),
        "research_pass": positive,
    }


def _bucket_by_name(report: Mapping[str, Any], split: str, bucket: str) -> Mapping[str, Any] | None:
    for row in report.get("ln_bucket_tables", {}).get(split, []):
        if row.get("bucket") == bucket:
            return row
    return None


def _recommendation(pass_criteria: Mapping[str, Any], *, selected_variant: str) -> str:
    if pass_criteria.get("research_pass"):
        return f"TEST_NEXT_ABLATION: {selected_variant} passed; compare against richer chord/LN factorization before mapper integration."
    return (
        f"MUTATE: {selected_variant} did not satisfy all LN tokenizer gates; "
        "prefer chord/LN factorization or cross-chunk occupancy before promoting duration-aware tokens."
    )


def _comparison_row(variant: str, split: str, values: Mapping[str, Any], *, table: str = "global") -> dict[str, Any]:
    row = {"variant": variant, "split": split, "table": table}
    row.update({key: value for key, value in values.items() if key not in {"mapset_bits", "mapset_events"}})
    return row


def _top_motif_rows(candidates: Sequence[tuple[tuple[str, ...], int, float]], *, limit: int) -> list[dict[str, Any]]:
    rows = []
    for index, (motif, count, gain) in enumerate(candidates[:limit]):
        rows.append({"rank": index + 1, "length": len(motif), "count": int(count), "gain": float(gain), "motif": " ".join(motif)})
    return rows


def _read_chunk_cache(path: Path, *, limit_chunks: int | None) -> pd.DataFrame:
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
        "chunk_index",
        "segment_id",
        "segment_chunk_index",
        "bar_phase_half",
        "start_beat_units",
        "length_units",
        "num_groups",
        "num_events",
        "raw_signature",
        "groups_json",
    ]
    frame = pd.read_parquet(path, columns=columns)
    if limit_chunks is not None:
        frame = frame.head(int(limit_chunks)).copy()
    return frame


def _split_by_source(chunk_df: pd.DataFrame) -> dict[int, str]:
    pairs = chunk_df[["source_row_index", "split"]].drop_duplicates("source_row_index")
    return {int(row.source_row_index): str(row.split) for row in pairs.itertuples(index=False)}


def _cross_split_artist_title_version_keys(chunk_df: pd.DataFrame) -> set[str]:
    maps = chunk_df[["source_row_index", "artist", "title", "version", "split"]].drop_duplicates("source_row_index")
    keyed = maps[["artist", "title", "version", "split"]].copy()
    keyed["key"] = _key_series(keyed, ["artist", "title", "version"])
    counts = keyed.drop_duplicates(["key", "split"]).groupby("key", dropna=False)["split"].nunique()
    return {str(key) for key, count in counts.items() if int(count) > 1}


def _key_series(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    result = pd.Series([""] * len(frame), index=frame.index)
    for column in columns:
        result = result + "|" + frame[column].map(_normalize_text)
    return result


def _normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).casefold().strip().split())


def _d0_reference(d0_report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "coverage": d0_report.get("coverage", {}),
        "duration_bucket_all": d0_report.get("key_tables", {}).get("duration_bucket_all", []),
        "interaction_all": d0_report.get("key_tables", {}).get("interaction_all", []),
        "cross_chunk_all": d0_report.get("key_tables", {}).get("cross_chunk_all", []),
    }


def _validate_variant_names(values: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(value) for value in values)
    unknown = sorted(set(names).difference(VARIANT_NAMES))
    if unknown:
        raise ValueError(f"unknown variant(s): {unknown}")
    if not names:
        raise ValueError("variant_names must not be empty")
    return names


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value


def _nonnegative_int(value: int, name: str) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(_normalize_json(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _write_result_log(path: Path, report: Mapping[str, Any]) -> None:
    selected = str(report.get("selected_variant"))
    selected_report = report.get("variants", {}).get(selected, {})
    baseline = report.get("variants", {}).get("r0_delta", {})
    pass_criteria = report.get("pass_criteria", {})
    lines = [
        "# Duration-Aware LN Tokenization Result Log",
        "",
        "## Summary",
        "",
        f"- Selected variant: `{selected}`",
        f"- Recommendation: {report.get('recommendation')}",
        f"- Research pass: {pass_criteria.get('research_pass')}",
        f"- Reconstruction pass: {pass_criteria.get('reconstruction_pass')}",
        f"- Baseline charged test bits/event: {_nested(baseline, 'splits', 'test', 'bits_per_event_with_dictionary')}",
        f"- Selected charged test bits/event: {_nested(selected_report, 'splits', 'test', 'bits_per_event_with_dictionary')}",
        f"- Selected valid charged bits/event: {_nested(selected_report, 'splits', 'valid', 'bits_per_event_with_dictionary')}",
        f"- Bootstrap delta mean: {_nested(report, 'bootstrap_mapset_delta', 'delta_bits_per_event_mean')}",
        f"- Bootstrap delta 95% interval: [{_nested(report, 'bootstrap_mapset_delta', 'delta_bits_per_event_p025')}, {_nested(report, 'bootstrap_mapset_delta', 'delta_bits_per_event_p975')}]",
        "",
        "## Variant Global Comparison",
        "",
        "| variant | split | charged bits/event | raw bits/event | fallback group rate | side info tokens |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for variant_name, variant in report.get("variants", {}).items():
        for split in ("valid", "test", "same_song_filtered_test"):
            row = variant.get("splits", {}).get(split)
            if not isinstance(row, Mapping) or not row:
                continue
            lines.append(
                "| {variant} | {split} | {charged} | {raw} | {fallback} | {side} |".format(
                    variant=variant_name,
                    split=split,
                    charged=_fmt_float(row.get("bits_per_event_with_dictionary")),
                    raw=_fmt_float(row.get("bits_per_event")),
                    fallback=_fmt_float(row.get("fallback_group_rate")),
                    side=row.get("side_info_token_count", 0),
                )
            )
    lines.extend(["", "## Gates", ""])
    for key, value in pass_criteria.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Interpretation", "", str(report.get("recommendation")), ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text("\n".join(lines), encoding="utf-8")
    tmp_path.replace(path)


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _fmt_float(value: object) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _normalize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_normalize_json(item) for item in value]
    if isinstance(value, set):
        return sorted(_normalize_json(item) for item in value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


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
    return " ".join(shlex.quote(part) for part in ["python", "-m", "pulsefield_model.osu_core.duration_ln_tokenization_audit", *args])


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run duration-aware LN tokenization audit variants.")
    parser.add_argument("--chunk-cache-path", type=Path, default=DEFAULT_BEAT_CHUNK_CACHE_PATH)
    parser.add_argument("--map-cache-path", type=Path, default=DEFAULT_BEAT_CHUNK_MAP_CACHE_PATH)
    parser.add_argument("--refinement-report-path", type=Path, default=DEFAULT_REFINEMENT_REPORT_PATH)
    parser.add_argument("--d0-report-path", type=Path, default=DEFAULT_D0_REPORT_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--result-log-path", type=Path, default=DEFAULT_RESULT_LOG_PATH)
    parser.add_argument("--comparison-csv-path", type=Path, default=DEFAULT_COMPARISON_CSV_PATH)
    parser.add_argument("--reconstruction-guard-path", type=Path, default=DEFAULT_RECONSTRUCTION_GUARD_PATH)
    parser.add_argument("--variant", dest="variant_names", action="append", choices=VARIANT_NAMES)
    parser.add_argument("--motif-vocab-size", type=int, default=DEFAULT_MOTIF_VOCAB_SIZE)
    parser.add_argument("--motif-min-n", type=int, default=DEFAULT_MOTIF_MIN_N)
    parser.add_argument("--motif-max-n", type=int, default=DEFAULT_MOTIF_MAX_N)
    parser.add_argument("--smoothing-alpha", type=float, default=DEFAULT_SMOOTHING_ALPHA)
    parser.add_argument("--rare-max-count", type=int, default=DEFAULT_RARE_MAX_COUNT)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument("--limit-chunks", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    raw_args = list(sys.argv[1:] if argv is None else argv)
    report = audit_duration_ln_tokenization(
        chunk_cache_path=args.chunk_cache_path,
        map_cache_path=args.map_cache_path,
        refinement_report_path=args.refinement_report_path,
        d0_report_path=args.d0_report_path,
        dataset_root=args.dataset_root,
        report_path=args.report_path,
        result_log_path=args.result_log_path,
        comparison_csv_path=args.comparison_csv_path,
        reconstruction_guard_path=args.reconstruction_guard_path,
        variant_names=tuple(args.variant_names) if args.variant_names else VARIANT_NAMES,
        motif_vocab_size=args.motif_vocab_size,
        motif_min_n=args.motif_min_n,
        motif_max_n=args.motif_max_n,
        smoothing_alpha=args.smoothing_alpha,
        rare_max_count=args.rare_max_count,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.random_seed,
        limit_chunks=args.limit_chunks,
        command=_command(raw_args),
    )
    print(
        "duration_ln_tokenization_audit "
        f"selected={report.get('selected_variant')} "
        f"research_pass={report.get('pass_criteria', {}).get('research_pass')} "
        f"report={report.get('report_path')}",
    )


if __name__ == "__main__":
    main()
