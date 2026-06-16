from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Sequence

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
    _make_model_token_bits,
    _match_motif,
    _rank_motifs,
    _score_encoded_generic,
    _split_report,
)
from pulsefield_model.osu_core.fallback_forensic_audit import (
    EXPECTED_R0_TEST_CHARGED_BITS_PER_EVENT,
    _baseline_tokens,
    _count_bucket,
    _fit_motif_model,
    _group_class,
    _iter_rows,
    _read_chunk_cache,
    _score_baseline_split,
    _skeleton_tokens,
)


SCHEMA_VERSION: Final[int] = 1
DEFAULT_REPORT_PATH: Final[Path] = Path(
    "artifacts/reports/audits/occupancy_factorized_tokenization/factorized_chord_ln_report.json",
)
DEFAULT_RESULT_LOG_PATH: Final[Path] = Path(
    "artifacts/reports/audits/occupancy_factorized_tokenization/factorized_chord_ln_result_log.md",
)
DEFAULT_COMPARISON_CSV_PATH: Final[Path] = Path(
    "artifacts/reports/audits/occupancy_factorized_tokenization/factorized_chord_ln_comparison.csv",
)


@dataclass(frozen=True)
class _FactorizedModel:
    vocab: _MotifVocab
    trie: Mapping[str, Any]
    skeleton_atom_counter: Counter[str]
    residual_atom_counter: Counter[str]
    train_counter: Counter[str]
    token_bits: Callable[[str], float]
    dictionary_bits: float
    learned_motif_count: int
    top_motifs: list[dict[str, Any]]


@dataclass
class _BucketAgg:
    split: str
    bucket: str
    group_count: int = 0
    event_count: int = 0
    fallback_group_count: int = 0
    code_bits: float = 0.0
    mapsets: set[int] = field(default_factory=set)

    def add(
        self,
        *,
        beatmap_set_id: int,
        event_count: int,
        fallback: bool,
        code_bits: float,
    ) -> None:
        self.group_count += 1
        self.event_count += int(event_count)
        self.fallback_group_count += int(fallback)
        self.code_bits += float(code_bits)
        self.mapsets.add(int(beatmap_set_id))


def audit_factorized_chord_ln_tokenizer(
    *,
    chunk_cache_path: str | Path = DEFAULT_BEAT_CHUNK_CACHE_PATH,
    report_path: str | Path | None = DEFAULT_REPORT_PATH,
    result_log_path: str | Path | None = DEFAULT_RESULT_LOG_PATH,
    comparison_csv_path: str | Path | None = DEFAULT_COMPARISON_CSV_PATH,
    motif_vocab_size: int = DEFAULT_MOTIF_VOCAB_SIZE,
    motif_min_n: int = DEFAULT_MOTIF_MIN_N,
    motif_max_n: int = DEFAULT_MOTIF_MAX_N,
    smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
    rare_max_count: int = DEFAULT_RARE_MAX_COUNT,
    top_motif_limit: int = 20,
    limit_chunks: int | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    """Run F1 skeleton-motif plus charged lane-residual tokenizer audit."""

    started_at = time.perf_counter()
    chunk_cache_path = Path(chunk_cache_path)
    report_path = None if report_path is None else Path(report_path)
    result_log_path = None if result_log_path is None else Path(result_log_path)
    comparison_csv_path = None if comparison_csv_path is None else Path(comparison_csv_path)
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
    factorized_model = _fit_factorized_model(
        chunk_df,
        motif_vocab_size=motif_vocab_size,
        motif_min_n=motif_min_n,
        motif_max_n=motif_max_n,
        smoothing_alpha=smoothing_alpha,
        top_motif_limit=top_motif_limit,
    )
    factorized_splits = {
        split: _score_factorized_split(
            chunk_df,
            split=split,
            model=factorized_model,
            rare_max_count=rare_max_count,
        )
        for split in ("train", "valid", "test")
    }
    rows = []
    for split in ("valid", "test"):
        rows.extend(
            _bucket_rows(
                _bucket_aggs(
                    chunk_df,
                    split=split,
                    variant="r0_delta",
                    token_bits=baseline_model.token_bits,
                    trie=baseline_model.trie,
                    dictionary_bits=baseline_model.dictionary_bits,
                ),
                variant="r0_delta",
                split=split,
                split_event_count=int(baseline_splits[split]["event_count"]),
                dictionary_bits=baseline_model.dictionary_bits,
            )
        )
        rows.extend(
            _bucket_rows(
                _bucket_aggs(
                    chunk_df,
                    split=split,
                    variant="f1_skeleton_lane_residual",
                    token_bits=factorized_model.token_bits,
                    trie=factorized_model.trie,
                    dictionary_bits=factorized_model.dictionary_bits,
                ),
                variant="f1_skeleton_lane_residual",
                split=split,
                split_event_count=int(factorized_splits[split]["event_count"]),
                dictionary_bits=factorized_model.dictionary_bits,
            )
        )
    reconstruction = _reconstruction_guard(chunk_df, factorized_model)
    pass_criteria = _pass_criteria(
        baseline_splits=baseline_splits,
        factorized_splits=factorized_splits,
        bucket_rows=rows,
        reconstruction=reconstruction,
    )
    recommendation = _recommendation(pass_criteria)
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "Chord/LN Factorized Tokenizer F1",
        "command": command,
        "code_commit": _git_stdout("rev-parse", "HEAD"),
        "code_dirty": bool(_git_stdout("status", "--short")),
        "chunk_cache_path": chunk_cache_path.as_posix(),
        "report_path": None if report_path is None else report_path.as_posix(),
        "result_log_path": None if result_log_path is None else result_log_path.as_posix(),
        "comparison_csv_path": None if comparison_csv_path is None else comparison_csv_path.as_posix(),
        "limited": limit_chunks is not None,
        "limit_chunks": limit_chunks,
        "elapsed_s": time.perf_counter() - started_at,
        "config": {
            "motif_vocab_size": motif_vocab_size,
            "motif_min_n": motif_min_n,
            "motif_max_n": motif_max_n,
            "smoothing_alpha": smoothing_alpha,
            "rare_max_count": rare_max_count,
            "residual_policy": "one charged residual token per source group: R:tap_mask:ln_start_mask:ln_end_mask[:Oorder]",
        },
        "dataset": _dataset_summary(chunk_df),
        "models": {
            "r0_delta": {
                "learned_motif_count": baseline_model.learned_motif_count,
                "dictionary_cost_bits": baseline_model.dictionary_bits,
            },
            "f1_skeleton_lane_residual": {
                "learned_motif_count": factorized_model.learned_motif_count,
                "dictionary_cost_bits": factorized_model.dictionary_bits,
                "skeleton_atom_vocab_size": len(factorized_model.skeleton_atom_counter),
                "residual_atom_vocab_size": len(factorized_model.residual_atom_counter),
                "top_motifs": factorized_model.top_motifs,
            },
        },
        "baseline_splits": baseline_splits,
        "factorized_splits": factorized_splits,
        "reconstruction": reconstruction,
        "bucket_tables": rows,
        "pass_criteria": pass_criteria,
        "recommendation": recommendation,
    }
    if report_path is not None:
        _write_json(report_path, report)
    if comparison_csv_path is not None:
        comparison_csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(comparison_csv_path, index=False)
    if result_log_path is not None:
        _write_result_log(result_log_path, report)
    return report


def _fit_factorized_model(
    chunk_df: pd.DataFrame,
    *,
    motif_vocab_size: int,
    motif_min_n: int,
    motif_max_n: int,
    smoothing_alpha: float,
    top_motif_limit: int,
) -> _FactorizedModel:
    skeleton_counter: Counter[str] = Counter()
    residual_counter: Counter[str] = Counter()
    candidate_counter: Counter[tuple[str, ...]] = Counter()
    for row in _iter_rows(chunk_df, split="train"):
        skeleton = _skeleton_tokens(row)
        residual = _residual_tokens(row)
        skeleton_counter.update(skeleton)
        residual_counter.update(residual)
        max_n = min(motif_max_n, len(skeleton))
        for ngram in range(motif_min_n, max_n + 1):
            for index in range(len(skeleton) - ngram + 1):
                candidate_counter[tuple(skeleton[index : index + ngram])] += 1
    candidates = _rank_motifs(candidate_counter)
    vocab = _build_motif_vocab(candidates[:motif_vocab_size])
    trie = _build_motif_trie(vocab.motif_to_id)
    train_counter: Counter[str] = Counter()
    for row in _iter_rows(chunk_df, split="train"):
        encoded_skeleton, _covered = _encode_skeleton_with_spans(_skeleton_tokens(row), trie)
        train_counter.update(token for token, _span in encoded_skeleton)
        train_counter.update(_residual_tokens(row))
    dictionary_bits = _dictionary_cost_bits_generic(
        vocab,
        skeleton_counter,
        smoothing_alpha=smoothing_alpha,
        motif_max_n=motif_max_n,
    )
    atom_counter = skeleton_counter + residual_counter
    model_universe = set(train_counter) | set(atom_counter) | set(vocab.motif_lengths) | CONTROL_TOKENS
    token_bits = _make_model_token_bits(
        train_counter,
        atom_counter,
        model_universe,
        smoothing_alpha=smoothing_alpha,
    )
    return _FactorizedModel(
        vocab=vocab,
        trie=trie,
        skeleton_atom_counter=skeleton_counter,
        residual_atom_counter=residual_counter,
        train_counter=train_counter,
        token_bits=token_bits,
        dictionary_bits=dictionary_bits,
        learned_motif_count=len(vocab.motif_to_id),
        top_motifs=_top_motifs(candidates[:top_motif_limit]),
    )


def _score_factorized_split(
    chunk_df: pd.DataFrame,
    *,
    split: str,
    model: _FactorizedModel,
    rare_max_count: int,
) -> dict[str, Any]:
    stats = _EncodedStats()
    residual_token_count = 0
    for row in _iter_rows(chunk_df, split=split):
        skeleton = _skeleton_tokens(row)
        residual = _residual_tokens(row)
        encoded_skeleton, covered_indexes = _encode_skeleton_with_spans(skeleton, model.trie)
        stats.chunk_count += 1
        stats.event_count += int(row.num_events)
        stats.group_count += int(row.num_groups)
        stats.mapset_events[int(row.beatmap_set_id)] += int(row.num_events)
        stats.motif_group_coverage_count += sum(1 for index in covered_indexes if index > 0)
        for token, _span in encoded_skeleton:
            stats.token_counter[token] += 1
            stats.token_count += 1
            if token.startswith("M"):
                stats.motif_token_count += 1
            else:
                stats.atom_token_count += 1
        for token in residual:
            stats.token_counter[token] += 1
            stats.token_count += 1
            stats.atom_token_count += 1
            stats.side_info_token_count += 1
            residual_token_count += 1
    _score_encoded_generic(
        stats,
        token_bits=model.token_bits,
        dictionary_bits=model.dictionary_bits,
        rare_max_count=rare_max_count,
    )
    report = _split_report(stats, dictionary_bits=model.dictionary_bits)
    report["residual_token_count"] = residual_token_count
    return report


def _bucket_aggs(
    chunk_df: pd.DataFrame,
    *,
    split: str,
    variant: str,
    token_bits: Callable[[str], float],
    trie: Mapping[str, Any],
    dictionary_bits: float,
) -> dict[str, _BucketAgg]:
    del dictionary_bits
    aggs: dict[str, _BucketAgg] = {}
    for row in _iter_rows(chunk_df, split=split):
        if variant == "r0_delta":
            group_bits, fallback_indexes = _baseline_group_bits(row, token_bits=token_bits, trie=trie)
        elif variant == "f1_skeleton_lane_residual":
            group_bits, fallback_indexes = _factorized_group_bits(row, token_bits=token_bits, trie=trie)
        else:
            raise ValueError(f"unknown variant: {variant}")
        groups = json.loads(str(row.groups_json or "[]"))
        for index, group in enumerate(groups):
            tap = int(group.get("tap_mask", 0))
            start = int(group.get("ln_start_mask", 0))
            end = int(group.get("ln_end_mask", 0))
            bucket = _group_class(tap, start, end)
            event_count = max(1, int(group.get("event_count", 1) or 1))
            agg = aggs.setdefault(bucket, _BucketAgg(split=split, bucket=bucket))
            agg.add(
                beatmap_set_id=int(row.beatmap_set_id),
                event_count=event_count,
                fallback=(index + 1) in fallback_indexes,
                code_bits=group_bits[index],
            )
    return aggs


def _bucket_rows(
    aggs: Mapping[str, _BucketAgg],
    *,
    variant: str,
    split: str,
    split_event_count: int,
    dictionary_bits: float,
) -> list[dict[str, Any]]:
    dict_bits_per_event = float(dictionary_bits) / float(split_event_count) if split_event_count else 0.0
    rows = []
    for bucket, agg in sorted(aggs.items()):
        charged = float(agg.code_bits) / float(agg.event_count) + dict_bits_per_event if agg.event_count else 0.0
        rows.append(
            {
                "variant": variant,
                "split": split,
                "bucket": bucket,
                "group_count": int(agg.group_count),
                "event_count": int(agg.event_count),
                "fallback_group_count": int(agg.fallback_group_count),
                "fallback_group_rate": float(agg.fallback_group_count) / float(agg.group_count) if agg.group_count else 0.0,
                "bits_per_event": float(agg.code_bits) / float(agg.event_count) if agg.event_count else 0.0,
                "bits_per_event_with_dictionary": charged,
                "mapset_count": len(agg.mapsets),
            }
        )
    return rows


def _baseline_group_bits(
    row: Any,
    *,
    token_bits: Callable[[str], float],
    trie: Mapping[str, Any],
) -> tuple[list[float], set[int]]:
    tokens = _baseline_tokens(row)
    groups = json.loads(str(row.groups_json or "[]"))
    bits = [0.0 for _group in groups]
    fallback_indexes: set[int] = set()
    encoded, covered = _encode_skeleton_with_spans(tokens, trie)
    for token, span in encoded:
        group_indexes = [index - 1 for index in span if index > 0]
        if not group_indexes:
            continue
        share = token_bits(token) / float(len(group_indexes))
        for group_index in group_indexes:
            bits[group_index] += share
    for index in range(1, len(tokens)):
        if index not in covered:
            fallback_indexes.add(index)
    return bits, fallback_indexes


def _factorized_group_bits(
    row: Any,
    *,
    token_bits: Callable[[str], float],
    trie: Mapping[str, Any],
) -> tuple[list[float], set[int]]:
    skeleton = _skeleton_tokens(row)
    residual = _residual_tokens(row)
    bits = [0.0 for _token in residual]
    fallback_indexes: set[int] = set()
    encoded, covered = _encode_skeleton_with_spans(skeleton, trie)
    for token, span in encoded:
        group_indexes = [index - 1 for index in span if index > 0]
        if not group_indexes:
            continue
        share = token_bits(token) / float(len(group_indexes))
        for group_index in group_indexes:
            bits[group_index] += share
    for index in range(1, len(skeleton)):
        if index not in covered:
            fallback_indexes.add(index)
    for index, token in enumerate(residual):
        bits[index] += token_bits(token)
    return bits, fallback_indexes


def _encode_skeleton_with_spans(
    tokens: Sequence[str],
    trie: Mapping[str, Any],
) -> tuple[list[tuple[str, tuple[int, ...]]], set[int]]:
    encoded: list[tuple[str, tuple[int, ...]]] = []
    covered_indexes: set[int] = set()
    index = 0
    while index < len(tokens):
        matched_token, matched_length = _match_motif(tokens, index, trie)
        if matched_token is not None:
            span = tuple(range(index, index + matched_length))
            encoded.append((matched_token, span))
            covered_indexes.update(span)
            index += matched_length
            continue
        encoded.append((tokens[index], (index,)))
        index += 1
    return encoded, covered_indexes


def _residual_tokens(row: Any) -> list[str]:
    groups = json.loads(str(row.groups_json or "[]"))
    tokens = []
    for group in groups:
        token = "R:{tap}:{start}:{end}".format(
            tap=int(group.get("tap_mask", 0)),
            start=int(group.get("ln_start_mask", 0)),
            end=int(group.get("ln_end_mask", 0)),
        )
        order_signature = str(group.get("order_signature", ".") or ".")
        tokens.append(token if order_signature == "." else f"{token}:O{order_signature}")
    return tokens


def _reconstruction_guard(chunk_df: pd.DataFrame, model: _FactorizedModel) -> dict[str, Any]:
    mismatch_count = 0
    checked_chunk_count = 0
    examples: list[dict[str, Any]] = []
    id_to_motif = {token_id: motif for motif, token_id in model.vocab.motif_to_id.items()}
    for row in _iter_rows(chunk_df):
        skeleton = _skeleton_tokens(row)
        residual = _residual_tokens(row)
        encoded, _covered = _encode_skeleton_with_spans(skeleton, model.trie)
        expanded = _expand_skeleton([token for token, _span in encoded], id_to_motif)
        reconstructed = _reconstruct_delta_tokens(expanded, residual)
        expected = _baseline_tokens(row)[1:]
        checked_chunk_count += 1
        if reconstructed == expected:
            continue
        mismatch_count += 1
        if len(examples) < 10:
            examples.append(
                {
                    "source_row_index": int(row.source_row_index),
                    "chunk_index": int(row.chunk_index),
                    "expected": expected[:20],
                    "reconstructed": reconstructed[:20],
                    "encoded_skeleton": [token for token, _span in encoded[:20]],
                    "residual": residual[:20],
                }
            )
    return {
        "checked_chunk_count": checked_chunk_count,
        "mismatch_count": mismatch_count,
        "pass": mismatch_count == 0 and checked_chunk_count > 0,
        "examples": examples,
    }


def _expand_skeleton(tokens: Sequence[str], id_to_motif: Mapping[str, tuple[str, ...]]) -> list[str]:
    expanded: list[str] = []
    for token in tokens:
        motif = id_to_motif.get(token)
        if motif is None:
            expanded.append(token)
        else:
            expanded.extend(motif)
    return expanded


def _reconstruct_delta_tokens(skeleton: Sequence[str], residual: Sequence[str]) -> list[str]:
    skeleton_groups = [token for token in skeleton if token.startswith("K:")]
    if len(skeleton_groups) != len(residual):
        return ["LENGTH_MISMATCH"]
    reconstructed = []
    for skeleton_token, residual_token in zip(skeleton_groups, residual, strict=True):
        delta = _skeleton_delta(skeleton_token)
        tap, start, end, order = _residual_payload(residual_token)
        suffix = "" if order == "." else f":O{order}"
        reconstructed.append(f"D:{delta}:{tap}:{start}:{end}{suffix}")
    return reconstructed


def _skeleton_delta(token: str) -> int:
    parts = token.split(":")
    if len(parts) < 2:
        return 0
    return int(parts[1])


def _residual_payload(token: str) -> tuple[int, int, int, str]:
    parts = token.split(":")
    if len(parts) < 4 or parts[0] != "R":
        return 0, 0, 0, "."
    order = "."
    if len(parts) > 4 and parts[4].startswith("O"):
        order = parts[4][1:]
    return int(parts[1]), int(parts[2]), int(parts[3]), order


def _pass_criteria(
    *,
    baseline_splits: Mapping[str, Mapping[str, Any]],
    factorized_splits: Mapping[str, Mapping[str, Any]],
    bucket_rows: Sequence[Mapping[str, Any]],
    reconstruction: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_test = baseline_splits.get("test", {})
    factorized_test = factorized_splits.get("test", {})
    baseline_bits = float(baseline_test.get("bits_per_event_with_dictionary", math.inf))
    factorized_bits = float(factorized_test.get("bits_per_event_with_dictionary", math.inf))
    baseline_consistency = abs(baseline_bits - EXPECTED_R0_TEST_CHARGED_BITS_PER_EVENT) < 0.05
    chord_delta = _bucket_delta(bucket_rows, split="test", bucket_names=("tap_ln_start", "tap_ln_end", "tap_ln_start_ln_end"))
    ln_delta = _bucket_delta(bucket_rows, split="test", bucket_names=("ln_start", "ln_end", "ln_start_ln_end"))
    global_nonregress = factorized_bits <= baseline_bits + 0.01
    chord_improve = chord_delta is not None and chord_delta <= -0.05
    return {
        "baseline_consistency_pass": baseline_consistency,
        "reconstruction_pass": bool(reconstruction.get("pass")),
        "global_nonregress_pass": global_nonregress,
        "chord_ln_bucket_improve_pass": chord_improve,
        "research_pass": bool(reconstruction.get("pass") and baseline_consistency and global_nonregress and chord_improve),
        "baseline_test_bits_per_event_with_dictionary": baseline_bits,
        "factorized_test_bits_per_event_with_dictionary": factorized_bits,
        "test_delta_bits_per_event": factorized_bits - baseline_bits,
        "test_chord_ln_bucket_delta_bits_per_event": chord_delta,
        "test_ln_bucket_delta_bits_per_event": ln_delta,
    }


def _bucket_delta(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    bucket_names: Sequence[str],
) -> float | None:
    by_variant = {"r0_delta": [0.0, 0], "f1_skeleton_lane_residual": [0.0, 0]}
    for row in rows:
        if row.get("split") != split or row.get("bucket") not in bucket_names:
            continue
        variant = str(row.get("variant"))
        if variant not in by_variant:
            continue
        events = int(row.get("event_count", 0) or 0)
        charged = float(row.get("bits_per_event_with_dictionary", 0.0) or 0.0)
        by_variant[variant][0] += charged * events
        by_variant[variant][1] += events
    if not by_variant["r0_delta"][1] or not by_variant["f1_skeleton_lane_residual"][1]:
        return None
    baseline = by_variant["r0_delta"][0] / by_variant["r0_delta"][1]
    factorized = by_variant["f1_skeleton_lane_residual"][0] / by_variant["f1_skeleton_lane_residual"][1]
    return factorized - baseline


def _recommendation(pass_criteria: Mapping[str, Any]) -> str:
    if pass_criteria.get("research_pass"):
        return "TEST_NEXT_COMBO: F1 passed; combine with occupancy only after O1 is measured."
    if pass_criteria.get("reconstruction_pass") and not pass_criteria.get("global_nonregress_pass"):
        return "KILL_OR_MUTATE_F1: charged residual cost or skeleton fragmentation regressed global bits/event."
    return "MUTATE_F1: factorized tokenizer did not satisfy reconstruction, global, and chord/LN gates."


def _dataset_summary(chunk_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "chunk_count": int(len(chunk_df)),
        "event_count": int(pd.to_numeric(chunk_df["num_events"], errors="coerce").fillna(0).sum()),
        "group_count": int(pd.to_numeric(chunk_df["num_groups"], errors="coerce").fillna(0).sum()),
        "map_count": int(chunk_df["source_row_index"].nunique()),
        "mapset_count": int(chunk_df["beatmap_set_id"].nunique()),
    }


def _top_motifs(candidates: Sequence[tuple[tuple[str, ...], int, float]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": index + 1,
            "length": len(motif),
            "support": int(count),
            "gain": float(gain),
            "motif": " ".join(motif),
        }
        for index, (motif, count, gain) in enumerate(candidates)
    ]


def _write_result_log(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pass_criteria = report.get("pass_criteria", {})
    lines = [
        "# Chord/LN Factorized Tokenizer Result Log",
        "",
        "## Summary",
        "",
        f"- Research pass: {pass_criteria.get('research_pass')}",
        f"- Recommendation: {report.get('recommendation')}",
        f"- Runtime seconds: {_fmt_float(report.get('elapsed_s'))}",
        f"- Reconstruction pass: {pass_criteria.get('reconstruction_pass')}",
        f"- Baseline test charged bits/event: {_fmt_float(pass_criteria.get('baseline_test_bits_per_event_with_dictionary'))}",
        f"- Factorized test charged bits/event: {_fmt_float(pass_criteria.get('factorized_test_bits_per_event_with_dictionary'))}",
        f"- Test delta bits/event: {_fmt_float(pass_criteria.get('test_delta_bits_per_event'))}",
        f"- Test chord/LN bucket delta bits/event: {_fmt_float(pass_criteria.get('test_chord_ln_bucket_delta_bits_per_event'))}",
        f"- Test LN bucket delta bits/event: {_fmt_float(pass_criteria.get('test_ln_bucket_delta_bits_per_event'))}",
        "",
        "## Interpretation",
        "",
        str(report.get("recommendation")),
    ]
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
    return " ".join(shlex.quote(part) for part in ["python", "-m", "pulsefield_model.osu_core.factorized_chord_ln_tokenizer_audit", *args])


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit skeleton motif plus charged lane residual tokenizer.")
    parser.add_argument("--chunk-cache-path", type=Path, default=DEFAULT_BEAT_CHUNK_CACHE_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--result-log-path", type=Path, default=DEFAULT_RESULT_LOG_PATH)
    parser.add_argument("--comparison-csv-path", type=Path, default=DEFAULT_COMPARISON_CSV_PATH)
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
    report = audit_factorized_chord_ln_tokenizer(
        chunk_cache_path=args.chunk_cache_path,
        report_path=args.report_path,
        result_log_path=args.result_log_path,
        comparison_csv_path=args.comparison_csv_path,
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
        "factorized_chord_ln_tokenizer_audit "
        f"research_pass={report.get('pass_criteria', {}).get('research_pass')} "
        f"recommendation={report.get('recommendation')} "
        f"report={report.get('report_path')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
