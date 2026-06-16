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
    _group_class,
    _iter_rows,
    _read_chunk_cache,
    _skeleton_tokens,
)


SCHEMA_VERSION: Final[int] = 1
DEFAULT_REPORT_PATH: Final[Path] = Path(
    "artifacts/reports/audits/occupancy_factorized_tokenization/selective_wildcard_report.json",
)
DEFAULT_RESULT_LOG_PATH: Final[Path] = Path(
    "artifacts/reports/audits/occupancy_factorized_tokenization/selective_wildcard_result_log.md",
)
DEFAULT_COMPARISON_CSV_PATH: Final[Path] = Path(
    "artifacts/reports/audits/occupancy_factorized_tokenization/selective_wildcard_comparison.csv",
)
DEFAULT_WILDCARD_VOCAB_SIZES: Final[tuple[int, ...]] = (1024, 4096, 8192)


@dataclass(frozen=True)
class _HybridModel:
    variant: str
    exact_vocab: _MotifVocab
    wildcard_vocab: _MotifVocab
    exact_trie: Mapping[str, Any]
    wildcard_trie: Mapping[str, Any]
    train_counter: Counter[str]
    exact_atom_counter: Counter[str]
    skeleton_atom_counter: Counter[str]
    residual_atom_counter: Counter[str]
    token_bits: Callable[[str], float]
    dictionary_bits: float
    exact_dictionary_bits: float
    wildcard_dictionary_bits: float
    wildcard_vocab_size: int
    top_wildcard_motifs: list[dict[str, Any]]


@dataclass
class _BucketAgg:
    split: str
    bucket: str
    group_count: int = 0
    event_count: int = 0
    fallback_group_count: int = 0
    code_bits: float = 0.0
    mapsets: set[int] = field(default_factory=set)

    def add(self, *, beatmap_set_id: int, event_count: int, fallback: bool, code_bits: float) -> None:
        self.group_count += 1
        self.event_count += int(event_count)
        self.fallback_group_count += int(fallback)
        self.code_bits += float(code_bits)
        self.mapsets.add(int(beatmap_set_id))


def audit_selective_wildcard_motifs(
    *,
    chunk_cache_path: str | Path = DEFAULT_BEAT_CHUNK_CACHE_PATH,
    report_path: str | Path | None = DEFAULT_REPORT_PATH,
    result_log_path: str | Path | None = DEFAULT_RESULT_LOG_PATH,
    comparison_csv_path: str | Path | None = DEFAULT_COMPARISON_CSV_PATH,
    exact_motif_vocab_size: int = DEFAULT_MOTIF_VOCAB_SIZE,
    wildcard_vocab_sizes: Sequence[int] = DEFAULT_WILDCARD_VOCAB_SIZES,
    motif_min_n: int = DEFAULT_MOTIF_MIN_N,
    motif_max_n: int = DEFAULT_MOTIF_MAX_N,
    smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
    rare_max_count: int = DEFAULT_RARE_MAX_COUNT,
    top_motif_limit: int = 20,
    limit_chunks: int | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    """Run P1 exact-first selective wildcard residual audit."""

    started_at = time.perf_counter()
    chunk_cache_path = Path(chunk_cache_path)
    report_path = None if report_path is None else Path(report_path)
    result_log_path = None if result_log_path is None else Path(result_log_path)
    comparison_csv_path = None if comparison_csv_path is None else Path(comparison_csv_path)
    exact_motif_vocab_size = _positive_int(exact_motif_vocab_size, "exact_motif_vocab_size")
    wildcard_vocab_sizes = tuple(sorted({_nonnegative_int(value, "wildcard_vocab_sizes") for value in wildcard_vocab_sizes}))
    if not wildcard_vocab_sizes:
        raise ValueError("wildcard_vocab_sizes must contain at least one value")
    motif_min_n = _positive_int(motif_min_n, "motif_min_n")
    motif_max_n = _positive_int(motif_max_n, "motif_max_n")
    if motif_max_n < motif_min_n:
        raise ValueError("motif_max_n must be >= motif_min_n")
    if smoothing_alpha <= 0:
        raise ValueError(f"smoothing_alpha must be positive, got {smoothing_alpha!r}")
    rare_max_count = _nonnegative_int(rare_max_count, "rare_max_count")

    chunk_df = _read_chunk_cache(chunk_cache_path, limit_chunks=limit_chunks)
    exact_candidates, exact_atom_counter = _learn_candidates(
        chunk_df,
        token_mapper=_baseline_tokens,
        motif_min_n=motif_min_n,
        motif_max_n=motif_max_n,
    )
    skeleton_candidates, skeleton_atom_counter = _learn_candidates(
        chunk_df,
        token_mapper=_skeleton_tokens,
        motif_min_n=motif_min_n,
        motif_max_n=motif_max_n,
    )
    exact_vocab = _build_prefixed_vocab(exact_candidates[:exact_motif_vocab_size], prefix="M")
    exact_dictionary_bits = _dictionary_cost_bits_generic(
        exact_vocab,
        exact_atom_counter,
        smoothing_alpha=smoothing_alpha,
        motif_max_n=motif_max_n,
    )
    models = [
        _fit_hybrid_model(
            chunk_df,
            exact_vocab=exact_vocab,
            exact_atom_counter=exact_atom_counter,
            skeleton_candidates=skeleton_candidates,
            skeleton_atom_counter=skeleton_atom_counter,
            wildcard_vocab_size=wildcard_size,
            exact_dictionary_bits=exact_dictionary_bits,
            smoothing_alpha=smoothing_alpha,
            motif_max_n=motif_max_n,
            top_motif_limit=top_motif_limit,
        )
        for wildcard_size in (0, *wildcard_vocab_sizes)
    ]

    reports: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for model in models:
        split_reports: dict[str, Any] = {}
        bucket_tables: dict[str, list[dict[str, Any]]] = {}
        for split in ("train", "valid", "test"):
            stats, bucket_aggs = _score_split(
                chunk_df,
                split=split,
                model=model,
                rare_max_count=rare_max_count,
            )
            split_reports[split] = _split_report(stats, dictionary_bits=model.dictionary_bits)
            split_reports[split]["wildcard_token_count"] = int(getattr(stats, "wildcard_token_count", 0))
            split_reports[split]["residual_token_count"] = int(getattr(stats, "residual_token_count", 0))
            rows.append(_comparison_row(model.variant, split, "global", "all", split_reports[split]))
            bucket_rows = _bucket_rows(
                bucket_aggs,
                variant=model.variant,
                split=split,
                split_event_count=int(split_reports[split]["event_count"]),
                dictionary_bits=model.dictionary_bits,
            )
            bucket_tables[split] = bucket_rows
            rows.extend(bucket_rows)
        reports[model.variant] = {
            "variant": model.variant,
            "wildcard_vocab_size": model.wildcard_vocab_size,
            "dictionary_cost_bits": model.dictionary_bits,
            "exact_dictionary_cost_bits": model.exact_dictionary_bits,
            "wildcard_dictionary_cost_bits": model.wildcard_dictionary_bits,
            "splits": split_reports,
            "bucket_tables": bucket_tables,
            "top_wildcard_motifs": model.top_wildcard_motifs,
        }

    selected_variant = _select_variant(reports)
    selected_model = next(model for model in models if model.variant == selected_variant)
    reconstruction = _reconstruction_guard(chunk_df, selected_model)
    pass_criteria = _pass_criteria(reports, selected_variant=selected_variant, reconstruction=reconstruction)
    recommendation = _recommendation(pass_criteria)
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "Selective Wildcard Motif Residual P1",
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
            "exact_motif_vocab_size": exact_motif_vocab_size,
            "wildcard_vocab_sizes": list(wildcard_vocab_sizes),
            "motif_min_n": motif_min_n,
            "motif_max_n": motif_max_n,
            "smoothing_alpha": smoothing_alpha,
            "rare_max_count": rare_max_count,
            "selection_policy": "choose lowest valid charged bits/event among selective wildcard variants; r0_delta is comparator only",
            "residual_policy": "wildcard groups emit charged R:tap:start:end[:Oorder] residual tokens",
        },
        "dataset": _dataset_summary(chunk_df),
        "external_baseline": {
            "variant": "r0_delta_chunk_local",
            "test_bits_per_event_with_dictionary": EXPECTED_R0_TEST_CHARGED_BITS_PER_EVENT,
        },
        "selected_variant": selected_variant,
        "variants": reports,
        "reconstruction": reconstruction,
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


def _learn_candidates(
    chunk_df: pd.DataFrame,
    *,
    token_mapper: Callable[[Any], list[str]],
    motif_min_n: int,
    motif_max_n: int,
) -> tuple[list[tuple[tuple[str, ...], int, float]], Counter[str]]:
    atom_counter: Counter[str] = Counter()
    candidate_counter: Counter[tuple[str, ...]] = Counter()
    for row in _iter_rows(chunk_df, split="train"):
        tokens = token_mapper(row)
        atom_counter.update(tokens)
        max_n = min(motif_max_n, len(tokens))
        for ngram in range(motif_min_n, max_n + 1):
            for index in range(len(tokens) - ngram + 1):
                candidate_counter[tuple(tokens[index : index + ngram])] += 1
    return _rank_motifs(candidate_counter), atom_counter


def _fit_hybrid_model(
    chunk_df: pd.DataFrame,
    *,
    exact_vocab: _MotifVocab,
    exact_atom_counter: Counter[str],
    skeleton_candidates: Sequence[tuple[tuple[str, ...], int, float]],
    skeleton_atom_counter: Counter[str],
    wildcard_vocab_size: int,
    exact_dictionary_bits: float,
    smoothing_alpha: float,
    motif_max_n: int,
    top_motif_limit: int,
) -> _HybridModel:
    variant = "r0_delta" if wildcard_vocab_size == 0 else f"p1_selective_wildcard_k{wildcard_vocab_size}"
    wildcard_vocab = _build_prefixed_vocab(skeleton_candidates[:wildcard_vocab_size], prefix="W")
    wildcard_dictionary_bits = _dictionary_cost_bits_generic(
        wildcard_vocab,
        skeleton_atom_counter,
        smoothing_alpha=smoothing_alpha,
        motif_max_n=motif_max_n,
    )
    exact_trie = _build_motif_trie(exact_vocab.motif_to_id)
    wildcard_trie = _build_motif_trie(wildcard_vocab.motif_to_id)
    residual_counter: Counter[str] = Counter()
    train_counter: Counter[str] = Counter()
    for row in _iter_rows(chunk_df, split="train"):
        encoded, _covered, _spans = _encode_row(
            row,
            exact_trie=exact_trie,
            wildcard_trie=wildcard_trie,
        )
        train_counter.update(encoded)
        residual_counter.update(token for token in encoded if token.startswith("R:"))
    atom_counter = exact_atom_counter + residual_counter
    model_universe = set(train_counter) | set(atom_counter) | set(exact_vocab.motif_lengths) | set(wildcard_vocab.motif_lengths) | CONTROL_TOKENS
    token_bits = _make_model_token_bits(
        train_counter,
        atom_counter,
        model_universe,
        smoothing_alpha=smoothing_alpha,
    )
    return _HybridModel(
        variant=variant,
        exact_vocab=exact_vocab,
        wildcard_vocab=wildcard_vocab,
        exact_trie=exact_trie,
        wildcard_trie=wildcard_trie,
        train_counter=train_counter,
        exact_atom_counter=exact_atom_counter,
        skeleton_atom_counter=skeleton_atom_counter,
        residual_atom_counter=residual_counter,
        token_bits=token_bits,
        dictionary_bits=exact_dictionary_bits + wildcard_dictionary_bits,
        exact_dictionary_bits=exact_dictionary_bits,
        wildcard_dictionary_bits=wildcard_dictionary_bits,
        wildcard_vocab_size=wildcard_vocab_size,
        top_wildcard_motifs=_top_motifs(skeleton_candidates[:top_motif_limit]) if wildcard_vocab_size else [],
    )


def _encode_row(
    row: Any,
    *,
    exact_trie: Mapping[str, Any],
    wildcard_trie: Mapping[str, Any],
) -> tuple[list[str], set[int], list[tuple[str, tuple[int, ...]]]]:
    exact_tokens = _baseline_tokens(row)
    skeleton_tokens = _skeleton_tokens(row)
    residual_tokens = _residual_tokens(row)
    encoded: list[str] = []
    covered: set[int] = set()
    spans: list[tuple[str, tuple[int, ...]]] = []
    index = 0
    while index < len(exact_tokens):
        exact_match, exact_length = _match_motif(exact_tokens, index, exact_trie)
        if exact_match is not None:
            span = tuple(range(index, index + exact_length))
            encoded.append(exact_match)
            covered.update(pos for pos in span if pos > 0)
            spans.append((exact_match, span))
            index += exact_length
            continue
        wildcard_match, wildcard_length = _match_motif(skeleton_tokens, index, wildcard_trie)
        wildcard_span = tuple(range(index, index + wildcard_length)) if wildcard_match is not None else ()
        wildcard_group_positions = [pos for pos in wildcard_span if pos > 0]
        if wildcard_match is not None and wildcard_group_positions:
            encoded.append(wildcard_match)
            spans.append((wildcard_match, wildcard_span))
            covered.update(wildcard_group_positions)
            for pos in wildcard_group_positions:
                residual = residual_tokens[pos - 1]
                encoded.append(residual)
                spans.append((residual, (pos,)))
            index += wildcard_length
            continue
        token = exact_tokens[index]
        encoded.append(token)
        spans.append((token, (index,)))
        index += 1
    return encoded, covered, spans


def _score_split(
    chunk_df: pd.DataFrame,
    *,
    split: str,
    model: _HybridModel,
    rare_max_count: int,
) -> tuple[_EncodedStats, dict[str, _BucketAgg]]:
    stats = _EncodedStats()
    bucket_aggs: dict[str, _BucketAgg] = {}
    for row in _iter_rows(chunk_df, split=split):
        encoded, covered, spans = _encode_row(row, exact_trie=model.exact_trie, wildcard_trie=model.wildcard_trie)
        group_bits = _group_bits(spans, token_bits=model.token_bits)
        groups = json.loads(str(row.groups_json or "[]"))
        stats.chunk_count += 1
        stats.event_count += int(row.num_events)
        stats.group_count += int(row.num_groups)
        stats.mapset_events[int(row.beatmap_set_id)] += int(row.num_events)
        stats.motif_group_coverage_count += len(covered)
        wildcard_token_count = 0
        residual_token_count = 0
        for token in encoded:
            stats.token_counter[token] += 1
            stats.token_count += 1
            if token.startswith("M") or token.startswith("W"):
                stats.motif_token_count += 1
                wildcard_token_count += int(token.startswith("W"))
            else:
                stats.atom_token_count += 1
                residual_token_count += int(token.startswith("R:"))
                stats.side_info_token_count += int(token.startswith("R:"))
        stats.wildcard_token_count = getattr(stats, "wildcard_token_count", 0) + wildcard_token_count  # type: ignore[attr-defined]
        stats.residual_token_count = getattr(stats, "residual_token_count", 0) + residual_token_count  # type: ignore[attr-defined]
        for pos, group in enumerate(groups, start=1):
            tap = int(group.get("tap_mask", 0))
            start = int(group.get("ln_start_mask", 0))
            end = int(group.get("ln_end_mask", 0))
            bucket = _group_class(tap, start, end)
            event_count = max(1, int(group.get("event_count", 1) or 1))
            agg = bucket_aggs.setdefault(bucket, _BucketAgg(split=split, bucket=bucket))
            agg.add(
                beatmap_set_id=int(row.beatmap_set_id),
                event_count=event_count,
                fallback=pos not in covered,
                code_bits=group_bits.get(pos, 0.0),
            )
    _score_encoded_generic(
        stats,
        token_bits=model.token_bits,
        dictionary_bits=model.dictionary_bits,
        rare_max_count=rare_max_count,
    )
    return stats, bucket_aggs


def _group_bits(
    spans: Sequence[tuple[str, tuple[int, ...]]],
    *,
    token_bits: Callable[[str], float],
) -> dict[int, float]:
    bits: dict[int, float] = {}
    for token, span in spans:
        group_positions = [pos for pos in span if pos > 0]
        if not group_positions:
            continue
        share = token_bits(token) / float(len(group_positions))
        for pos in group_positions:
            bits[pos] = bits.get(pos, 0.0) + share
    return bits


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
        bits = float(agg.code_bits) / float(agg.event_count) if agg.event_count else 0.0
        rows.append(
            {
                "variant": variant,
                "split": split,
                "table": "group_class",
                "bucket": bucket,
                "group_count": int(agg.group_count),
                "event_count": int(agg.event_count),
                "fallback_group_count": int(agg.fallback_group_count),
                "fallback_group_rate": float(agg.fallback_group_count) / float(agg.group_count) if agg.group_count else 0.0,
                "bits_per_event": bits,
                "bits_per_event_with_dictionary": bits + dict_bits_per_event,
                "mapset_count": len(agg.mapsets),
            }
        )
    return rows


def _reconstruction_guard(chunk_df: pd.DataFrame, model: _HybridModel) -> dict[str, Any]:
    exact_id_to_motif = {token_id: motif for motif, token_id in model.exact_vocab.motif_to_id.items()}
    wildcard_id_to_motif = {token_id: motif for motif, token_id in model.wildcard_vocab.motif_to_id.items()}
    checked_chunk_count = 0
    mismatch_count = 0
    examples: list[dict[str, Any]] = []
    for row in _iter_rows(chunk_df):
        encoded, _covered, _spans = _encode_row(row, exact_trie=model.exact_trie, wildcard_trie=model.wildcard_trie)
        reconstructed = _signature_from_hybrid_tokens(
            encoded,
            exact_id_to_motif=exact_id_to_motif,
            wildcard_id_to_motif=wildcard_id_to_motif,
        )
        expected = str(row.raw_signature or "")
        checked_chunk_count += 1
        if reconstructed == expected:
            continue
        mismatch_count += 1
        if len(examples) < 10:
            examples.append(
                {
                    "source_row_index": int(row.source_row_index),
                    "chunk_index": int(row.chunk_index),
                    "expected": expected,
                    "reconstructed": reconstructed,
                    "encoded": encoded[:30],
                }
            )
    return {
        "checked_chunk_count": checked_chunk_count,
        "mismatch_count": mismatch_count,
        "pass": mismatch_count == 0 and checked_chunk_count > 0,
        "examples": examples,
    }


def _signature_from_hybrid_tokens(
    encoded: Sequence[str],
    *,
    exact_id_to_motif: Mapping[str, tuple[str, ...]],
    wildcard_id_to_motif: Mapping[str, tuple[str, ...]],
) -> str:
    decoded: list[str] = []
    index = 0
    while index < len(encoded):
        token = encoded[index]
        exact = exact_id_to_motif.get(token)
        if exact is not None:
            decoded.extend(exact)
            index += 1
            continue
        wildcard = wildcard_id_to_motif.get(token)
        if wildcard is not None:
            index += 1
            for skeleton in wildcard:
                if skeleton.startswith("C"):
                    decoded.append(skeleton)
                    continue
                if index >= len(encoded):
                    decoded.append("D:0:0:0:0")
                    continue
                residual = encoded[index]
                index += 1
                decoded.append(_delta_from_skeleton_residual(skeleton, residual))
            continue
        decoded.append(token)
        index += 1
    return _signature_from_delta_tokens(decoded)


def _delta_from_skeleton_residual(skeleton: str, residual: str) -> str:
    delta = _skeleton_delta(skeleton)
    tap, start, end, order = _residual_payload(residual)
    suffix = "" if order == "." else f":O{order}"
    return f"D:{delta}:{tap}:{start}:{end}{suffix}"


def _signature_from_delta_tokens(tokens: Sequence[str]) -> str:
    offset = 0
    atoms = []
    for token in tokens:
        if token.startswith("C"):
            offset = 0
            continue
        if not token.startswith("D:"):
            continue
        parts = token.split(":")
        offset += int(parts[1])
        suffix = f":{parts[5]}" if len(parts) > 5 else ""
        atoms.append(f"A:{offset}:{parts[2]}:{parts[3]}:{parts[4]}{suffix}")
    return ";".join(atoms)


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


def _skeleton_delta(token: str) -> int:
    parts = token.split(":")
    return int(parts[1]) if len(parts) > 1 else 0


def _residual_payload(token: str) -> tuple[int, int, int, str]:
    parts = token.split(":")
    if len(parts) < 4 or parts[0] != "R":
        return 0, 0, 0, "."
    order = "."
    if len(parts) > 4 and parts[4].startswith("O"):
        order = parts[4][1:]
    return int(parts[1]), int(parts[2]), int(parts[3]), order


def _build_prefixed_vocab(
    candidates: Sequence[tuple[tuple[str, ...], int, float]],
    *,
    prefix: str,
) -> _MotifVocab:
    motif_to_id = {motif: f"{prefix}{index}" for index, (motif, _count, _gain) in enumerate(candidates)}
    motif_lengths = {f"{prefix}{index}": len(motif) for index, (motif, _count, _gain) in enumerate(candidates)}
    return _MotifVocab(motif_to_id=motif_to_id, motif_lengths=motif_lengths)


def _select_variant(reports: Mapping[str, Mapping[str, Any]]) -> str:
    candidates = [name for name in reports if name != "r0_delta"]
    if not candidates:
        return "r0_delta"
    return min(
        candidates,
        key=lambda name: float(reports[name].get("splits", {}).get("valid", {}).get("bits_per_event_with_dictionary", math.inf)),
    )


def _pass_criteria(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    selected_variant: str,
    reconstruction: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = reports.get("r0_delta", {})
    selected = reports.get(selected_variant, {})
    baseline_test = baseline.get("splits", {}).get("test", {}) if isinstance(baseline, Mapping) else {}
    selected_test = selected.get("splits", {}).get("test", {}) if isinstance(selected, Mapping) else {}
    baseline_bits = float(baseline_test.get("bits_per_event_with_dictionary", math.inf))
    selected_bits = float(selected_test.get("bits_per_event_with_dictionary", math.inf))
    chord_delta = _bucket_delta(reports, selected_variant=selected_variant, buckets=("tap_ln_start", "tap_ln_end", "tap_ln_start_ln_end"))
    ln_delta = _bucket_delta(reports, selected_variant=selected_variant, buckets=("ln_start", "ln_end", "ln_start_ln_end"))
    residual_count = int(selected_test.get("residual_token_count", 0) or 0)
    group_count = int(selected_test.get("group_count", 0) or 0)
    return {
        "selected_variant": selected_variant,
        "reconstruction_pass": bool(reconstruction.get("pass")),
        "baseline_consistency_pass": abs(baseline_bits - EXPECTED_R0_TEST_CHARGED_BITS_PER_EVENT) < 0.05,
        "global_nonregress_pass": selected_bits <= baseline_bits + 0.01,
        "chord_ln_bucket_improve_pass": chord_delta is not None and chord_delta <= -0.05,
        "ln_bucket_improve_pass": ln_delta is not None and ln_delta <= -0.05,
        "baseline_test_bits_per_event_with_dictionary": baseline_bits,
        "selected_test_bits_per_event_with_dictionary": selected_bits,
        "selected_delta_bits_per_event": selected_bits - baseline_bits,
        "test_chord_ln_bucket_delta_bits_per_event": chord_delta,
        "test_ln_bucket_delta_bits_per_event": ln_delta,
        "selected_test_residual_token_count": residual_count,
        "selected_test_residual_token_rate_per_group": float(residual_count) / float(group_count) if group_count else 0.0,
        "research_pass": bool(
            reconstruction.get("pass")
            and selected_bits <= baseline_bits + 0.01
            and ((chord_delta is not None and chord_delta <= -0.05) or (ln_delta is not None and ln_delta <= -0.05))
        ),
    }


def _bucket_delta(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    selected_variant: str,
    buckets: Sequence[str],
) -> float | None:
    totals = {"r0_delta": [0.0, 0.0], selected_variant: [0.0, 0.0]}
    for variant in totals:
        report = reports.get(variant, {})
        rows = report.get("bucket_tables", {}).get("test", []) if isinstance(report, Mapping) else []
        for row in rows:
            if row.get("bucket") not in buckets:
                continue
            events = float(row.get("event_count", 0) or 0)
            charged = float(row.get("bits_per_event_with_dictionary", 0.0) or 0.0)
            totals[variant][0] += charged * events
            totals[variant][1] += events
    if not totals["r0_delta"][1] or not totals[selected_variant][1]:
        return None
    return totals[selected_variant][0] / totals[selected_variant][1] - totals["r0_delta"][0] / totals["r0_delta"][1]


def _recommendation(pass_criteria: Mapping[str, Any]) -> str:
    if pass_criteria.get("research_pass"):
        return "TEST_NEXT_COMBO: selective wildcard residual passed; compare with O1 occupancy as a selective combined variant."
    if pass_criteria.get("reconstruction_pass") and not pass_criteria.get("global_nonregress_pass"):
        return "KILL_OR_MUTATE_P1: selective wildcard residual still regresses global charged bits/event."
    return "MUTATE_P1: selective wildcard residual did not satisfy reconstruction, global, and bucket gates."


def _comparison_row(variant: str, split: str, table: str, bucket: str, values: Mapping[str, Any]) -> dict[str, Any]:
    row = {"variant": variant, "split": split, "table": table, "bucket": bucket}
    row.update({key: value for key, value in values.items() if key not in {"mapset_bits", "mapset_events"}})
    return row


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


def _dataset_summary(chunk_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "chunk_count": int(len(chunk_df)),
        "event_count": int(pd.to_numeric(chunk_df["num_events"], errors="coerce").fillna(0).sum()),
        "group_count": int(pd.to_numeric(chunk_df["num_groups"], errors="coerce").fillna(0).sum()),
        "map_count": int(chunk_df["source_row_index"].nunique()),
        "mapset_count": int(chunk_df["beatmap_set_id"].nunique()),
    }


def _write_result_log(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pass_criteria = report.get("pass_criteria", {})
    lines = [
        "# Selective Wildcard Motif Result Log",
        "",
        "## Summary",
        "",
        f"- Research pass: {pass_criteria.get('research_pass')}",
        f"- Recommendation: {report.get('recommendation')}",
        f"- Runtime seconds: {_fmt_float(report.get('elapsed_s'))}",
        f"- Selected variant: {report.get('selected_variant')}",
        f"- Reconstruction pass: {pass_criteria.get('reconstruction_pass')}",
        f"- Baseline test charged bits/event: {_fmt_float(pass_criteria.get('baseline_test_bits_per_event_with_dictionary'))}",
        f"- Selected test charged bits/event: {_fmt_float(pass_criteria.get('selected_test_bits_per_event_with_dictionary'))}",
        f"- Selected delta bits/event: {_fmt_float(pass_criteria.get('selected_delta_bits_per_event'))}",
        f"- Chord/LN bucket delta: {_fmt_float(pass_criteria.get('test_chord_ln_bucket_delta_bits_per_event'))}",
        f"- LN bucket delta: {_fmt_float(pass_criteria.get('test_ln_bucket_delta_bits_per_event'))}",
        f"- Residual token rate/group: {_fmt_float(pass_criteria.get('selected_test_residual_token_rate_per_group'))}",
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


def _parse_vocab_sizes(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _git_stdout(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001 - git metadata is optional.
        return ""


def _command(args: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in ["python", "-m", "pulsefield_model.osu_core.selective_wildcard_motif_audit", *args])


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit exact-first selective wildcard motif residuals.")
    parser.add_argument("--chunk-cache-path", type=Path, default=DEFAULT_BEAT_CHUNK_CACHE_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--result-log-path", type=Path, default=DEFAULT_RESULT_LOG_PATH)
    parser.add_argument("--comparison-csv-path", type=Path, default=DEFAULT_COMPARISON_CSV_PATH)
    parser.add_argument("--exact-motif-vocab-size", type=int, default=DEFAULT_MOTIF_VOCAB_SIZE)
    parser.add_argument("--wildcard-vocab-sizes", type=_parse_vocab_sizes, default=DEFAULT_WILDCARD_VOCAB_SIZES)
    parser.add_argument("--motif-min-n", type=int, default=DEFAULT_MOTIF_MIN_N)
    parser.add_argument("--motif-max-n", type=int, default=DEFAULT_MOTIF_MAX_N)
    parser.add_argument("--smoothing-alpha", type=float, default=DEFAULT_SMOOTHING_ALPHA)
    parser.add_argument("--rare-max-count", type=int, default=DEFAULT_RARE_MAX_COUNT)
    parser.add_argument("--top-motif-limit", type=int, default=20)
    parser.add_argument("--limit-chunks", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = audit_selective_wildcard_motifs(
        chunk_cache_path=args.chunk_cache_path,
        report_path=args.report_path,
        result_log_path=args.result_log_path,
        comparison_csv_path=args.comparison_csv_path,
        exact_motif_vocab_size=args.exact_motif_vocab_size,
        wildcard_vocab_sizes=args.wildcard_vocab_sizes,
        motif_min_n=args.motif_min_n,
        motif_max_n=args.motif_max_n,
        smoothing_alpha=args.smoothing_alpha,
        rare_max_count=args.rare_max_count,
        top_motif_limit=args.top_motif_limit,
        limit_chunks=args.limit_chunks,
        command=_command(sys.argv[1:]),
    )
    print(
        "selective_wildcard_motif_audit "
        f"selected={report.get('selected_variant')} "
        f"research_pass={report.get('pass_criteria', {}).get('research_pass')} "
        f"recommendation={report.get('recommendation')} "
        f"report={report.get('report_path')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
