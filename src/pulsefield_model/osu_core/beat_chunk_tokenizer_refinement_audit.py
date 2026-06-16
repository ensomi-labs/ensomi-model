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
from typing import Any, Final, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from pulsefield_model.osu_core.beat_chunk_pattern_audit import (
    BASELINE_BEAT_INDEPENDENT_BITS_PER_EVENT,
    BASELINE_BEAT_JOINT_BITS_PER_EVENT,
    BASELINE_V2_BITS_PER_EVENT,
    DEFAULT_BEAT_CHUNK_CACHE_PATH,
    DEFAULT_BEAT_CHUNK_MAP_CACHE_PATH,
    DEFAULT_BEAT_CHUNK_REPORT_PATH,
    DEFAULT_MOTIF_MAX_N,
    DEFAULT_MOTIF_MIN_N,
    DEFAULT_MOTIF_VOCAB_SIZES,
    DEFAULT_SMOOTHING_ALPHA,
    _atomic_token,
    _is_missing_scalar,
    _json_loads,
    _normalize_json,
)
from pulsefield_model.osu_core.beat_structure_token_audit import DEFAULT_LE3_STRUCTURE_CACHE_PATH


SCHEMA_VERSION: Final[int] = 1
DEFAULT_REPORT_PATH: Final[Path] = Path(
    "artifacts/reports/audits/beat_chunk_tokenizer_refinement/"
    "beat_chunk_tokenizer_refinement_audit_le3.json",
)
DEFAULT_RESULT_LOG_PATH: Final[Path] = Path(
    "artifacts/reports/audits/beat_chunk_tokenizer_refinement/"
    "beat_chunk_tokenizer_refinement_result_log.md",
)
DEFAULT_CHUNK_UNITS: Final[int] = 96
DEFAULT_RARE_MAX_COUNT: Final[int] = 1
DEFAULT_TOP_MOTIF_LIMIT: Final[int] = 20
_CONTROL_TOKENS: Final[set[str]] = {"C0", "C1", "O0", "O1"}


@dataclass(frozen=True)
class _VariantConfig:
    name: str
    token_family: str
    offset_mode: str
    mirror_mode: str
    uses_orientation_token: bool
    uses_motif_vocab: bool
    lossless: bool = True


@dataclass(frozen=True)
class _MotifVocab:
    motif_to_id: dict[tuple[str, ...], str]
    motif_lengths: dict[str, int]


@dataclass
class _Encoded:
    token_counter: Counter[str]
    token_count: int = 0
    group_count: int = 0
    chunk_count: int = 0
    event_count: int = 0
    motif_token_count: int = 0
    motif_group_coverage_count: int = 0
    group_atom_token_count: int = 0
    whole_chunk_token_count: int = 0
    orientation_token_count: int = 0
    chunk_start_token_count: int = 0
    order_signature_atom_count: int = 0
    order_signature_chunk_count: int = 0


@dataclass(frozen=True)
class _TrainModel:
    train_counter: Counter[str]
    model_token_universe: set[str]
    train_group_counter: Counter[str]
    train_group_universe: set[str]
    group_field_counters: dict[str, Counter[str]]
    train_whole_universe: set[str]


def audit_beat_chunk_tokenizer_refinement(
    *,
    chunk_cache_path: str | Path = DEFAULT_BEAT_CHUNK_CACHE_PATH,
    map_cache_path: str | Path | None = DEFAULT_BEAT_CHUNK_MAP_CACHE_PATH,
    previous_report_path: str | Path | None = DEFAULT_BEAT_CHUNK_REPORT_PATH,
    beat_structure_cache_path: str | Path | None = DEFAULT_LE3_STRUCTURE_CACHE_PATH,
    report_path: str | Path | None = DEFAULT_REPORT_PATH,
    result_log_path: str | Path | None = DEFAULT_RESULT_LOG_PATH,
    motif_vocab_sizes: Sequence[int] = DEFAULT_MOTIF_VOCAB_SIZES,
    motif_min_n: int = DEFAULT_MOTIF_MIN_N,
    motif_max_n: int = DEFAULT_MOTIF_MAX_N,
    smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
    rare_max_count: int = DEFAULT_RARE_MAX_COUNT,
    top_motif_limit: int = DEFAULT_TOP_MOTIF_LIMIT,
    limit_chunks: int | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    """Run a cache-only tokenizer benchmark over hard-passed beat-chunk rows."""

    started_at = time.perf_counter()
    chunk_cache_path = Path(chunk_cache_path)
    map_cache_path = None if map_cache_path is None else Path(map_cache_path)
    previous_report_path = None if previous_report_path is None else Path(previous_report_path)
    beat_structure_cache_path = None if beat_structure_cache_path is None else Path(beat_structure_cache_path)
    report_path = None if report_path is None else Path(report_path)
    result_log_path = None if result_log_path is None else Path(result_log_path)
    motif_vocab_sizes = tuple(sorted({_positive_int(value, "motif_vocab_sizes") for value in motif_vocab_sizes}))
    motif_min_n = _positive_int(motif_min_n, "motif_min_n")
    motif_max_n = _positive_int(motif_max_n, "motif_max_n")
    if motif_max_n < motif_min_n:
        raise ValueError("motif_max_n must be >= motif_min_n")
    if smoothing_alpha <= 0:
        raise ValueError(f"smoothing_alpha must be positive, got {smoothing_alpha!r}")
    rare_max_count = _nonnegative_int(rare_max_count, "rare_max_count")
    top_motif_limit = _nonnegative_int(top_motif_limit, "top_motif_limit")
    if limit_chunks is not None and int(limit_chunks) < 0:
        raise ValueError(f"limit_chunks must be non-negative, got {limit_chunks!r}")

    chunk_df = _read_chunk_cache(chunk_cache_path, limit_chunks=limit_chunks)
    map_df = _read_map_cache(map_cache_path)
    previous_report = _read_json(previous_report_path)
    baselines = _baselines(previous_report)
    split_by_source = _split_by_source(chunk_df)
    v2_success_source_ids = _read_v2_success_source_ids(beat_structure_cache_path)
    split_matched_baselines = _split_matched_baselines(
        beat_structure_cache_path,
        split_by_source=split_by_source,
        smoothing_alpha=smoothing_alpha,
        rare_max_count=rare_max_count,
    )

    split_integrity = _split_integrity(chunk_df, map_df)
    guard = _guard_parity(previous_report, map_df)
    accounting = _accounting_summary(chunk_df, v2_success_source_ids)
    variants = _default_variants()
    variant_rows: list[dict[str, Any]] = []
    for variant in variants:
        variant_rows.extend(
            _evaluate_variant(
                chunk_df,
                variant=variant,
                motif_vocab_sizes=motif_vocab_sizes,
                motif_min_n=motif_min_n,
                motif_max_n=motif_max_n,
                smoothing_alpha=smoothing_alpha,
                rare_max_count=rare_max_count,
                top_motif_limit=top_motif_limit,
                v2_success_source_ids=v2_success_source_ids,
            )
        )

    best_valid = _best_variant(variant_rows, split="valid", require_motif=True)
    best_test = _best_variant(variant_rows, split="test", require_motif=True)
    reconstruction_checks = _reconstruction_checks(
        chunk_df,
        selected_variant=best_valid,
        motif_min_n=motif_min_n,
        motif_max_n=motif_max_n,
    )
    whole_failures = _whole_chunk_failure_summary(variant_rows)
    pass_criteria = _pass_criteria(
        split_integrity=split_integrity,
        guard=guard,
        selected_variant=best_valid,
        baselines=baselines,
        split_matched_baselines=split_matched_baselines,
        whole_failures=whole_failures,
        reconstruction_checks=reconstruction_checks,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "report_path": None if report_path is None else report_path.as_posix(),
        "result_log_path": None if result_log_path is None else result_log_path.as_posix(),
        "chunk_cache_path": chunk_cache_path.as_posix(),
        "chunk_cache_sha256": _sha256(chunk_cache_path),
        "map_cache_path": None if map_cache_path is None else map_cache_path.as_posix(),
        "map_cache_sha256": None if map_cache_path is None or not map_cache_path.exists() else _sha256(map_cache_path),
        "previous_report_path": None if previous_report_path is None else previous_report_path.as_posix(),
        "previous_report_sha256": (
            None if previous_report_path is None or not previous_report_path.exists() else _sha256(previous_report_path)
        ),
        "beat_structure_cache_path": None if beat_structure_cache_path is None else beat_structure_cache_path.as_posix(),
        "beat_structure_cache_sha256": (
            None if beat_structure_cache_path is None or not beat_structure_cache_path.exists() else _sha256(beat_structure_cache_path)
        ),
        "limited": limit_chunks is not None,
        "limit_chunks": limit_chunks,
        "config": {
            "motif_vocab_sizes": list(motif_vocab_sizes),
            "motif_min_n": motif_min_n,
            "motif_max_n": motif_max_n,
            "smoothing_alpha": smoothing_alpha,
            "rare_max_count": rare_max_count,
            "top_motif_limit": top_motif_limit,
            "scoring_policy": "train-estimated smoothed code length; chunk phase tokens included for all variants; mirror variants include orientation token for non-symmetric chunks",
            "unknown_group_policy": "unseen group atoms pay escape token plus structured field payload",
            "unknown_whole_chunk_policy": "unseen whole chunks pay escape token plus atomic group-sequence payload",
            "dictionary_cost_policy": "motif definitions are encoded once as train-smoothed group-token payloads plus a small length code and amortized per split",
        },
        "baselines": baselines,
        "split_matched_baselines": split_matched_baselines,
        "split_integrity": split_integrity,
        "guard_parity": guard,
        "accounting_summary": accounting,
        "reconstruction_checks": reconstruction_checks,
        "variant_results": variant_rows,
        "valid_selected_motif": best_valid,
        "best_test_motif_diagnostic": best_test,
        "whole_chunk_failure_summary": whole_failures,
        "pass_criteria": pass_criteria,
        "recommendation": _recommendation(pass_criteria),
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
        "bar_phase_half",
        "num_groups",
        "num_events",
        "raw_signature",
        "canonical_signature",
        "orientation",
        "is_mirror_symmetric",
        "groups_json",
        "canonical_groups_json",
    ]
    frame = pd.read_parquet(path, columns=columns)
    if limit_chunks is not None:
        frame = frame.head(int(limit_chunks)).copy()
    return frame


def _read_map_cache(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_v2_success_source_ids(path: Path | None) -> set[int] | None:
    if path is None or not path.exists():
        return None
    try:
        frame = pd.read_parquet(path, columns=["source_row_index", "v2_tokenization_ok"])
    except Exception:  # noqa: BLE001 - optional diagnostic input.
        return None
    ok = frame[frame["v2_tokenization_ok"].fillna(False).astype(bool)]
    return {int(value) for value in ok["source_row_index"].dropna().astype(int)}


def _split_by_source(chunk_df: pd.DataFrame) -> dict[int, str]:
    pairs = chunk_df[["source_row_index", "split"]].drop_duplicates("source_row_index")
    return {int(row.source_row_index): str(row.split) for row in pairs.itertuples(index=False)}


def _split_matched_baselines(
    path: Path | None,
    *,
    split_by_source: Mapping[int, str],
    smoothing_alpha: float,
    rare_max_count: int,
) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"available": False, "reason": "beat_structure_cache_missing"}
    columns = [
        "source_row_index",
        "event_count",
        "v2_tokenization_ok",
        "beat_joint_counter_json",
        "v2_token_counter_json",
    ]
    try:
        frame = pd.read_parquet(path, columns=columns)
    except Exception as exc:  # noqa: BLE001 - optional diagnostic input.
        return {"available": False, "reason": type(exc).__name__, "message": str(exc)}
    frame = frame[frame["source_row_index"].astype(int).isin(split_by_source)].copy()
    frame["split"] = frame["source_row_index"].astype(int).map(split_by_source)
    frame = frame[frame["split"].isin(["train", "valid", "test"])]
    v2_ok = frame["v2_tokenization_ok"].fillna(False).astype(bool)
    return {
        "available": True,
        "policy": "split-matched train-estimated smoothed token code length over per-map counters; v2 is success-only because four maps lack v2 tokens",
        "beat_joint": _score_counter_family(
            frame,
            counter_column="beat_joint_counter_json",
            smoothing_alpha=smoothing_alpha,
            rare_max_count=rare_max_count,
        ),
        "current_v2_success_only": _score_counter_family(
            frame[v2_ok],
            counter_column="v2_token_counter_json",
            smoothing_alpha=smoothing_alpha,
            rare_max_count=rare_max_count,
        ),
        "v2_tokenization_failure_count": int((~v2_ok).sum()),
        "v2_failed_source_row_indexes": [
            int(value) for value in frame.loc[~v2_ok, "source_row_index"].head(20).tolist()
        ],
    }


def _score_counter_family(
    frame: pd.DataFrame,
    *,
    counter_column: str,
    smoothing_alpha: float,
    rare_max_count: int,
) -> dict[str, Any]:
    if frame.empty:
        return {"available": False, "reason": "empty_frame"}
    train_counter = _aggregate_json_counter(frame[frame["split"] == "train"], counter_column)
    if not train_counter:
        return {"available": False, "reason": "empty_train_counter"}
    rows: dict[str, Any] = {"available": True}
    for split in ("train", "valid", "test"):
        split_frame = frame[frame["split"] == split]
        counter = _aggregate_json_counter(split_frame, counter_column)
        rows[split] = _score_counter_against_train(
            counter,
            train_counter=train_counter,
            event_count=int(pd.to_numeric(split_frame["event_count"], errors="coerce").fillna(0).sum()),
            smoothing_alpha=smoothing_alpha,
            rare_max_count=rare_max_count,
        )
        rows[split]["map_count"] = int(split_frame["source_row_index"].nunique())
    return rows


def _aggregate_json_counter(frame: pd.DataFrame, column: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for value in frame[column]:
        if _is_missing_scalar(value) or value == "":
            continue
        try:
            payload = json.loads(str(value))
        except json.JSONDecodeError:
            continue
        counter.update({str(key): int(count) for key, count in payload.items()})
    return counter


def _score_counter_against_train(
    counter: Counter[str],
    *,
    train_counter: Counter[str],
    event_count: int,
    smoothing_alpha: float,
    rare_max_count: int,
) -> dict[str, Any]:
    train_total = sum(train_counter.values())
    vocab_size = len(train_counter) + 1
    denominator = float(train_total) + smoothing_alpha * float(vocab_size)
    escape_probability = smoothing_alpha / denominator if denominator > 0 else 1.0
    escape_bits = -math.log2(escape_probability) if escape_probability > 0 else 0.0
    bits = 0.0
    unknown_count = 0
    rare_bits = 0.0
    for token, count in counter.items():
        train_count = train_counter.get(token, 0)
        if train_count <= 0:
            token_bits = escape_bits
            unknown_count += count
        else:
            probability = (float(train_count) + smoothing_alpha) / denominator if denominator > 0 else 1.0
            token_bits = -math.log2(probability)
        contribution = float(count) * token_bits
        bits += contribution
        if train_count <= rare_max_count:
            rare_bits += contribution
    token_count = sum(counter.values())
    return {
        "event_count": int(event_count),
        "token_count": int(token_count),
        "code_bits": bits,
        "bits_per_event": float(bits) / float(event_count) if event_count else 0.0,
        "entropy_bits_per_token_code": float(bits) / float(token_count) if token_count else 0.0,
        "unknown_token_count": int(unknown_count),
        "unknown_token_rate": float(unknown_count) / float(token_count) if token_count else 0.0,
        "rare_token_bits_fraction": float(rare_bits) / float(bits) if bits else 0.0,
        "smoothing_vocab_size": int(vocab_size),
    }


def _default_variants() -> tuple[_VariantConfig, ...]:
    return (
        _VariantConfig("atomic_abs_raw", "atomic", "absolute", "raw", False, False),
        _VariantConfig("atomic_abs_mirror", "atomic", "absolute", "mirror", True, False),
        _VariantConfig("whole_chunk_raw", "whole_chunk", "absolute", "raw", False, False),
        _VariantConfig("whole_chunk_mirror", "whole_chunk", "absolute", "mirror", True, False),
        _VariantConfig("motif_abs_raw", "motif", "absolute", "raw", False, True),
        _VariantConfig("motif_abs_mirror", "motif", "absolute", "mirror", True, True),
        _VariantConfig("motif_delta_raw", "motif", "delta", "raw", False, True),
        _VariantConfig("motif_delta_mirror", "motif", "delta", "mirror", True, True),
    )


def _evaluate_variant(
    chunk_df: pd.DataFrame,
    *,
    variant: _VariantConfig,
    motif_vocab_sizes: Sequence[int],
    motif_min_n: int,
    motif_max_n: int,
    smoothing_alpha: float,
    rare_max_count: int,
    top_motif_limit: int,
    v2_success_source_ids: set[int] | None,
) -> list[dict[str, Any]]:
    if variant.uses_motif_vocab:
        candidates = _learn_motif_candidates(chunk_df, variant=variant, motif_min_n=motif_min_n, motif_max_n=motif_max_n)
        sizes = tuple(motif_vocab_sizes)
    else:
        candidates = []
        sizes = (0,)

    rows: list[dict[str, Any]] = []
    for vocab_size in sizes:
        vocab = _build_motif_vocab(candidates[:vocab_size]) if variant.uses_motif_vocab else _MotifVocab({}, {})
        encoded_train = _encode_split(chunk_df, variant=variant, split="train", vocab=vocab)
        train_model = _train_model(chunk_df, variant=variant, encoded_train=encoded_train, vocab=vocab)
        dictionary_bits = (
            _dictionary_cost_bits(vocab, train_model, smoothing_alpha=smoothing_alpha, motif_max_n=motif_max_n)
            if variant.uses_motif_vocab
            else 0.0
        )
        split_metrics = {}
        for split in ("train", "valid", "test"):
            encoded = encoded_train if split == "train" else _encode_split(chunk_df, variant=variant, split=split, vocab=vocab)
            split_metrics[split] = _score_encoded(
                encoded,
                train_model=train_model,
                smoothing_alpha=smoothing_alpha,
                rare_max_count=rare_max_count,
                dictionary_bits=dictionary_bits,
            )
        if v2_success_source_ids is not None:
            encoded_success_test = _encode_split(
                chunk_df,
                variant=variant,
                split="test",
                vocab=vocab,
                source_ids=v2_success_source_ids,
            )
            split_metrics["test_v2_success_only"] = _score_encoded(
                encoded_success_test,
                train_model=train_model,
                smoothing_alpha=smoothing_alpha,
                rare_max_count=rare_max_count,
                dictionary_bits=dictionary_bits,
            )
        row = {
            "variant": variant.name,
            "token_family": variant.token_family,
            "offset_mode": variant.offset_mode,
            "mirror_mode": variant.mirror_mode,
            "uses_orientation_token": variant.uses_orientation_token,
            "uses_order_signature": True,
            "lossless_by_contract": variant.lossless,
            "motif_vocab_size": None if not variant.uses_motif_vocab else int(vocab_size),
            "learned_motif_count": len(vocab.motif_to_id),
            "dictionary_cost_bits": dictionary_bits,
            "train": split_metrics["train"],
            "valid": split_metrics["valid"],
            "test": split_metrics["test"],
            "test_v2_success_only": split_metrics.get("test_v2_success_only"),
            "generalization_gap_test_minus_train_bits_per_event": (
                split_metrics["test"]["bits_per_event"] - split_metrics["train"]["bits_per_event"]
                if split_metrics["test"]["event_count"] and split_metrics["train"]["event_count"]
                else None
            ),
            "top_motifs": _top_motifs(candidates[: min(int(vocab_size), top_motif_limit)]) if variant.uses_motif_vocab else [],
        }
        rows.append(row)
    return rows


def _learn_motif_candidates(
    chunk_df: pd.DataFrame,
    *,
    variant: _VariantConfig,
    motif_min_n: int,
    motif_max_n: int,
) -> list[tuple[tuple[str, ...], int, float]]:
    counter: Counter[tuple[str, ...]] = Counter()
    for row in _iter_rows(chunk_df, split="train"):
        tokens = _group_tokens_for_row(row, variant=variant)
        max_n = min(motif_max_n, len(tokens))
        for ngram in range(motif_min_n, max_n + 1):
            for index in range(len(tokens) - ngram + 1):
                counter[tuple(tokens[index : index + ngram])] += 1
    candidates: list[tuple[tuple[str, ...], int, float]] = []
    for motif, count in counter.items():
        if count < 2:
            continue
        gain = float(count) * float(len(motif) - 1)
        candidates.append((motif, int(count), gain))
    candidates.sort(key=lambda item: (-item[2], -item[1], -len(item[0]), item[0]))
    return candidates


def _build_motif_vocab(candidates: Sequence[tuple[tuple[str, ...], int, float]]) -> _MotifVocab:
    motif_to_id: dict[tuple[str, ...], str] = {}
    motif_lengths: dict[str, int] = {}
    for index, (motif, _count, _gain) in enumerate(candidates):
        token_id = f"M{index}"
        motif_to_id[motif] = token_id
        motif_lengths[token_id] = len(motif)
    return _MotifVocab(motif_to_id=motif_to_id, motif_lengths=motif_lengths)


def _encode_split(
    chunk_df: pd.DataFrame,
    *,
    variant: _VariantConfig,
    split: str,
    vocab: _MotifVocab,
    source_ids: set[int] | None = None,
) -> _Encoded:
    encoded = _Encoded(token_counter=Counter())
    motif_trie = _build_motif_trie(vocab.motif_to_id) if vocab.motif_to_id else {}
    for row in _iter_rows(chunk_df, split=split, source_ids=source_ids):
        group_tokens = _group_tokens_for_row(row, variant=variant)
        encoded.chunk_count += 1
        encoded.event_count += int(row.num_events)
        encoded.group_count += len(group_tokens)
        chunk_token = f"C{int(row.bar_phase_half)}"
        _add_token(encoded, chunk_token)
        encoded.chunk_start_token_count += 1
        if variant.uses_orientation_token and not bool(row.is_mirror_symmetric):
            orientation_token = f"O{int(row.orientation)}"
            _add_token(encoded, orientation_token)
            encoded.orientation_token_count += 1

        if variant.token_family == "whole_chunk":
            whole = _whole_chunk_token(row, variant=variant)
            _add_token(encoded, whole)
            encoded.whole_chunk_token_count += 1
            if any(_has_order_signature(token) for token in _signature_atoms(whole)):
                encoded.order_signature_chunk_count += 1
            continue

        index = 0
        chunk_has_order_signature = False
        while index < len(group_tokens):
            matched_token, matched_length = _match_motif(group_tokens, index, motif_trie)
            if matched_token is not None:
                _add_token(encoded, matched_token)
                encoded.motif_token_count += 1
                encoded.motif_group_coverage_count += matched_length
                index += matched_length
                continue
            token = group_tokens[index]
            _add_token(encoded, token)
            encoded.group_atom_token_count += 1
            if _has_order_signature(token):
                encoded.order_signature_atom_count += 1
                chunk_has_order_signature = True
            index += 1
        if chunk_has_order_signature:
            encoded.order_signature_chunk_count += 1
    return encoded


def _add_token(encoded: _Encoded, token: str) -> None:
    encoded.token_counter[token] += 1
    encoded.token_count += 1


def _train_model(
    chunk_df: pd.DataFrame,
    *,
    variant: _VariantConfig,
    encoded_train: _Encoded,
    vocab: _MotifVocab,
) -> _TrainModel:
    group_counter: Counter[str] = Counter()
    whole_universe: set[str] = set()
    for row in _iter_rows(chunk_df, split="train"):
        group_counter.update(_group_tokens_for_row(row, variant=variant))
        if variant.token_family == "whole_chunk":
            whole_universe.add(_whole_chunk_token(row, variant=variant))
    model_universe = set(encoded_train.token_counter)
    model_universe.update(_CONTROL_TOKENS)
    model_universe.update(group_counter)
    model_universe.update(vocab.motif_lengths)
    if variant.token_family == "whole_chunk":
        model_universe.update(whole_universe)
    return _TrainModel(
        train_counter=encoded_train.token_counter,
        model_token_universe=model_universe,
        train_group_counter=group_counter,
        train_group_universe=set(group_counter),
        group_field_counters=_group_field_counters(group_counter),
        train_whole_universe=whole_universe,
    )


def _dictionary_cost_bits(
    vocab: _MotifVocab,
    train_model: _TrainModel,
    *,
    smoothing_alpha: float,
    motif_max_n: int,
) -> float:
    if not vocab.motif_to_id:
        return 0.0
    group_total = sum(train_model.train_group_counter.values())
    group_vocab_size = len(train_model.train_group_universe) + 1
    denominator = float(group_total) + smoothing_alpha * float(group_vocab_size)
    escape_probability = smoothing_alpha / denominator if denominator > 0 else 1.0
    escape_bits = -math.log2(escape_probability) if escape_probability > 0 else 0.0
    length_bits = math.log2(max(2, int(motif_max_n) + 1))
    bits = 0.0
    for motif in vocab.motif_to_id:
        bits += length_bits
        for group_token in motif:
            if group_token not in train_model.train_group_universe:
                bits += escape_bits + _unknown_group_bits(
                    group_token,
                    train_model.group_field_counters,
                    smoothing_alpha,
                )
                continue
            probability = (float(train_model.train_group_counter[group_token]) + smoothing_alpha) / denominator
            bits += -math.log2(probability)
    return bits


def _score_encoded(
    encoded: _Encoded,
    *,
    train_model: _TrainModel,
    smoothing_alpha: float,
    rare_max_count: int,
    dictionary_bits: float = 0.0,
) -> dict[str, Any]:
    train_total = sum(train_model.train_counter.values())
    vocab_size = len(train_model.model_token_universe) + 1
    denominator = float(train_total) + smoothing_alpha * float(vocab_size)
    escape_probability = smoothing_alpha / denominator if denominator > 0 else 1.0
    escape_bits = -math.log2(escape_probability) if escape_probability > 0 else 0.0
    bits = 0.0
    known_token_count = 0
    unknown_group_atom_count = 0
    unknown_whole_chunk_count = 0
    rare_token_bits = 0.0
    long_tail_token_count = 0
    for token, count in encoded.token_counter.items():
        train_count = train_model.train_counter.get(token, 0)
        if _is_group_token(token) and token not in train_model.train_group_universe:
            token_bits = escape_bits + _unknown_group_bits(token, train_model.group_field_counters, smoothing_alpha)
            unknown_group_atom_count += count
        elif _is_whole_token(token) and token not in train_model.train_whole_universe:
            token_bits = escape_bits + _whole_payload_bits(token, train_model, smoothing_alpha)
            unknown_whole_chunk_count += count
        else:
            probability = (float(train_count) + smoothing_alpha) / denominator if denominator > 0 else 1.0
            token_bits = -math.log2(probability) if probability > 0 else 0.0
            known_token_count += count
        contribution = float(count) * token_bits
        bits += contribution
        if train_count <= rare_max_count:
            rare_token_bits += contribution
            long_tail_token_count += count
    event_count = encoded.event_count
    group_count = encoded.group_count
    chunk_count = encoded.chunk_count
    return {
        "encoded_token_count": int(encoded.token_count),
        "known_token_count": int(known_token_count),
        "unknown_token_count": int(unknown_group_atom_count + unknown_whole_chunk_count),
        "event_count": int(event_count),
        "group_count": int(group_count),
        "chunk_count": int(chunk_count),
        "code_bits": bits,
        "bits_per_event": float(bits) / float(event_count) if event_count else 0.0,
        "dictionary_cost_bits": float(dictionary_bits),
        "dictionary_bits_per_event": float(dictionary_bits) / float(event_count) if event_count else 0.0,
        "bits_per_event_with_dictionary": (
            float(bits + dictionary_bits) / float(event_count) if event_count else 0.0
        ),
        "bits_per_group": float(bits) / float(group_count) if group_count else 0.0,
        "bits_per_chunk": float(bits) / float(chunk_count) if chunk_count else 0.0,
        "tokens_per_event": float(encoded.token_count) / float(event_count) if event_count else 0.0,
        "tokens_per_group": float(encoded.token_count) / float(group_count) if group_count else 0.0,
        "tokens_per_chunk": float(encoded.token_count) / float(chunk_count) if chunk_count else 0.0,
        "motif_token_count": int(encoded.motif_token_count),
        "motif_group_coverage_count": int(encoded.motif_group_coverage_count),
        "motif_coverage_rate": float(encoded.motif_group_coverage_count) / float(group_count) if group_count else 0.0,
        "group_atom_token_count": int(encoded.group_atom_token_count),
        "fallback_group_rate": (
            float(group_count - encoded.motif_group_coverage_count) / float(group_count)
            if group_count and encoded.motif_token_count
            else (1.0 if group_count and encoded.group_atom_token_count else None)
        ),
        "whole_chunk_token_count": int(encoded.whole_chunk_token_count),
        "unknown_group_atom_count": int(unknown_group_atom_count),
        "unknown_group_atom_rate": (
            float(unknown_group_atom_count) / float(encoded.group_atom_token_count)
            if encoded.group_atom_token_count
            else 0.0
        ),
        "unknown_whole_chunk_count": int(unknown_whole_chunk_count),
        "unknown_whole_chunk_rate": (
            float(unknown_whole_chunk_count) / float(encoded.whole_chunk_token_count)
            if encoded.whole_chunk_token_count
            else 0.0
        ),
        "orientation_token_count": int(encoded.orientation_token_count),
        "orientation_token_rate_per_chunk": (
            float(encoded.orientation_token_count) / float(chunk_count) if chunk_count else 0.0
        ),
        "chunk_start_token_count": int(encoded.chunk_start_token_count),
        "order_signature_atom_count": int(encoded.order_signature_atom_count),
        "order_signature_chunk_count": int(encoded.order_signature_chunk_count),
        "order_signature_atom_rate": (
            float(encoded.order_signature_atom_count) / float(max(1, encoded.group_atom_token_count))
        ),
        "long_tail_token_count": int(long_tail_token_count),
        "rare_token_bits": rare_token_bits,
        "rare_token_bits_fraction": float(rare_token_bits) / float(bits) if bits else 0.0,
        "smoothing_vocab_size": int(vocab_size),
    }


def _iter_rows(
    chunk_df: pd.DataFrame,
    *,
    split: str | None = None,
    source_ids: set[int] | None = None,
) -> Iterable[Any]:
    frame = chunk_df
    if split is not None:
        frame = chunk_df[chunk_df["split"].fillna("").astype(str) == split]
    if source_ids is not None:
        frame = frame[frame["source_row_index"].astype(int).isin(source_ids)]
    yield from frame.itertuples(index=False)


def _group_tokens_for_row(row: Any, *, variant: _VariantConfig) -> list[str]:
    column = "canonical_groups_json" if variant.mirror_mode == "mirror" else "groups_json"
    groups = _json_loads(getattr(row, column))
    if variant.offset_mode == "delta":
        return _delta_tokens(groups)
    return [_atomic_token(group) for group in groups]


def _delta_tokens(groups: Sequence[Mapping[str, Any]]) -> list[str]:
    tokens: list[str] = []
    previous_offset = 0
    for group in groups:
        offset = int(group["offset_units"])
        delta = offset - previous_offset
        previous_offset = offset
        token = "D:{delta}:{tap}:{start}:{end}".format(
            delta=delta,
            tap=int(group["tap_mask"]),
            start=int(group["ln_start_mask"]),
            end=int(group["ln_end_mask"]),
        )
        order_signature = str(group.get("order_signature", ".") or ".")
        tokens.append(token if order_signature == "." else f"{token}:O{order_signature}")
    return tokens


def _whole_chunk_token(row: Any, *, variant: _VariantConfig) -> str:
    column = "canonical_signature" if variant.mirror_mode == "mirror" else "raw_signature"
    prefix = "WC" if variant.mirror_mode == "mirror" else "WR"
    return f"{prefix}:{getattr(row, column)}"


def _signature_atoms(whole_token: str) -> list[str]:
    if ":" not in whole_token:
        return []
    signature = whole_token.split(":", 1)[1]
    if not signature:
        return []
    return [item for item in signature.split(";") if item]


def _whole_payload_bits(token: str, train_model: _TrainModel, smoothing_alpha: float) -> float:
    atoms = _signature_atoms(token)
    if not atoms:
        return 1.0
    train_total = sum(train_model.train_group_counter.values())
    vocab_size = len(train_model.train_group_universe) + 1
    denominator = float(train_total) + smoothing_alpha * float(vocab_size)
    escape_probability = smoothing_alpha / denominator if denominator > 0 else 1.0
    escape_bits = -math.log2(escape_probability) if escape_probability > 0 else 0.0
    bits = math.log2(DEFAULT_CHUNK_UNITS + 1)
    for atom in atoms:
        if atom not in train_model.train_group_universe:
            bits += escape_bits + _unknown_group_bits(atom, train_model.group_field_counters, smoothing_alpha)
            continue
        probability = (float(train_model.train_group_counter.get(atom, 0)) + smoothing_alpha) / denominator
        bits += -math.log2(probability)
    return bits


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


def _reconstruction_checks(
    chunk_df: pd.DataFrame,
    *,
    selected_variant: Mapping[str, Any],
    motif_min_n: int,
    motif_max_n: int,
) -> dict[str, Any]:
    selected_vocab_size = int(selected_variant.get("motif_vocab_size") or 0) if isinstance(selected_variant, Mapping) else 0
    specs: list[tuple[str, int]] = []
    for name in ("motif_abs_raw", "motif_abs_mirror", "motif_delta_raw", "motif_delta_mirror"):
        if selected_vocab_size > 0:
            specs.append((name, selected_vocab_size))
    if isinstance(selected_variant, Mapping):
        selected_name = str(selected_variant.get("variant") or "")
        if selected_name and selected_vocab_size > 0 and (selected_name, selected_vocab_size) not in specs:
            specs.append((selected_name, selected_vocab_size))
    rows = []
    for name, vocab_size in specs:
        variant = _variant_by_name(name)
        if variant is None:
            continue
        candidates = _learn_motif_candidates(
            chunk_df,
            variant=variant,
            motif_min_n=motif_min_n,
            motif_max_n=motif_max_n,
        )
        vocab = _build_motif_vocab(candidates[:vocab_size])
        rows.append(_reconstruct_variant(chunk_df, variant=variant, vocab=vocab, motif_vocab_size=vocab_size))
    mismatch_count = sum(int(row["mismatch_count"]) for row in rows)
    checked_chunk_count = sum(int(row["checked_chunk_count"]) for row in rows)
    return {
        "checked_variant_count": len(rows),
        "checked_chunk_count": checked_chunk_count,
        "mismatch_count": mismatch_count,
        "pass": mismatch_count == 0 and checked_chunk_count > 0,
        "rows": rows,
    }


def _variant_by_name(name: str) -> _VariantConfig | None:
    for variant in _default_variants():
        if variant.name == name:
            return variant
    return None


def _reconstruct_variant(
    chunk_df: pd.DataFrame,
    *,
    variant: _VariantConfig,
    vocab: _MotifVocab,
    motif_vocab_size: int,
) -> dict[str, Any]:
    mismatch_count = 0
    checked_chunk_count = 0
    examples: list[dict[str, Any]] = []
    motif_trie = _build_motif_trie(vocab.motif_to_id)
    id_to_motif = {token_id: motif for motif, token_id in vocab.motif_to_id.items()}
    target_column = "canonical_signature" if variant.mirror_mode == "mirror" else "raw_signature"
    for row in _iter_rows(chunk_df):
        source_tokens = _group_tokens_for_row(row, variant=variant)
        encoded_tokens = _encode_group_sequence(source_tokens, motif_trie)
        decoded_tokens = _decode_group_sequence(encoded_tokens, id_to_motif)
        reconstructed = _signature_from_group_tokens(decoded_tokens, offset_mode=variant.offset_mode)
        expected = str(getattr(row, target_column) or "")
        checked_chunk_count += 1
        if reconstructed == expected:
            continue
        mismatch_count += 1
        if len(examples) < 10:
            examples.append(
                {
                    "source_row_index": int(getattr(row, "source_row_index")),
                    "chunk_index": int(getattr(row, "chunk_index")),
                    "split": str(getattr(row, "split")),
                    "expected": expected,
                    "reconstructed": reconstructed,
                    "encoded_tokens": encoded_tokens[:20],
                }
            )
    return {
        "variant": variant.name,
        "motif_vocab_size": int(motif_vocab_size),
        "checked_chunk_count": checked_chunk_count,
        "mismatch_count": mismatch_count,
        "pass": mismatch_count == 0 and checked_chunk_count > 0,
        "target_signature": target_column,
        "examples": examples,
    }


def _encode_group_sequence(tokens: Sequence[str], motif_trie: Mapping[str, Any]) -> list[str]:
    encoded: list[str] = []
    index = 0
    while index < len(tokens):
        matched_token, matched_length = _match_motif(tokens, index, motif_trie)
        if matched_token is not None:
            encoded.append(matched_token)
            index += matched_length
            continue
        encoded.append(tokens[index])
        index += 1
    return encoded


def _decode_group_sequence(tokens: Sequence[str], id_to_motif: Mapping[str, tuple[str, ...]]) -> list[str]:
    decoded: list[str] = []
    for token in tokens:
        motif = id_to_motif.get(token)
        if motif is None:
            decoded.append(token)
        else:
            decoded.extend(motif)
    return decoded


def _signature_from_group_tokens(tokens: Sequence[str], *, offset_mode: str) -> str:
    if offset_mode == "absolute":
        return ";".join(tokens)
    absolute_tokens: list[str] = []
    offset = 0
    for token in tokens:
        absolute, offset = _delta_token_to_absolute(token, previous_offset=offset)
        absolute_tokens.append(absolute)
    return ";".join(absolute_tokens)


def _delta_token_to_absolute(token: str, *, previous_offset: int) -> tuple[str, int]:
    parts = token.split(":")
    if len(parts) < 5 or parts[0] != "D":
        return token, previous_offset
    offset = previous_offset + int(parts[1])
    suffix = f":{parts[5]}" if len(parts) > 5 else ""
    return f"A:{offset}:{parts[2]}:{parts[3]}:{parts[4]}{suffix}", offset


def _group_field_counters(group_counter: Counter[str]) -> dict[str, Counter[str]]:
    counters = {
        "kind": Counter(),
        "offset": Counter(),
        "tap": Counter(),
        "start": Counter(),
        "end": Counter(),
        "order": Counter(),
    }
    for token, count in group_counter.items():
        parsed = _parse_group_token(token)
        for field, value in parsed.items():
            counters[field][str(value)] += count
    return counters


def _unknown_group_bits(token: str, field_counters: Mapping[str, Counter[str]], smoothing_alpha: float) -> float:
    parsed = _parse_group_token(token)
    cardinality = {
        "kind": 2,
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
        probability = (
            (float(counter.get(str(value), 0)) + smoothing_alpha) / denominator
            if denominator > 0
            else 1.0 / float(cardinality[field])
        )
        bits += -math.log2(probability)
    return bits


def _parse_group_token(token: str) -> dict[str, int]:
    parts = token.split(":")
    if len(parts) < 5 or parts[0] not in {"A", "D"}:
        return {"kind": 0, "offset": 0, "tap": 0, "start": 0, "end": 0, "order": 0}
    order = parts[5][1:] if len(parts) > 5 and parts[5].startswith("O") else "."
    return {
        "kind": 0 if parts[0] == "A" else 1,
        "offset": int(parts[1]),
        "tap": int(parts[2]),
        "start": int(parts[3]),
        "end": int(parts[4]),
        "order": _small_hash(order),
    }


def _is_group_token(token: str) -> bool:
    return token.startswith("A:") or token.startswith("D:")


def _is_whole_token(token: str) -> bool:
    return token.startswith("WR:") or token.startswith("WC:")


def _has_order_signature(token: str) -> bool:
    return ":O" in token


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


def _split_integrity(chunk_df: pd.DataFrame, map_df: pd.DataFrame) -> dict[str, Any]:
    map_rows = chunk_df[
        [
            "source_row_index",
            "beatmap_set_id",
            "beatmap_id",
            "beatmap_path",
            "artist",
            "title",
            "version",
            "split",
        ]
    ].drop_duplicates("source_row_index")
    mapset_split_counts = (
        chunk_df[["beatmap_set_id", "split"]]
        .drop_duplicates()
        .groupby("beatmap_set_id", dropna=False)["split"]
        .nunique()
    )
    map_split_counts = (
        chunk_df[["beatmap_id", "split"]]
        .drop_duplicates()
        .groupby("beatmap_id", dropna=False)["split"]
        .nunique()
    )
    split_summary = {}
    for split, frame in chunk_df.groupby("split", dropna=False):
        split_summary[str(split)] = {
            "chunk_count": int(len(frame)),
            "event_count": int(pd.to_numeric(frame["num_events"], errors="coerce").fillna(0).sum()),
            "group_count": int(pd.to_numeric(frame["num_groups"], errors="coerce").fillna(0).sum()),
            "map_count": int(frame["source_row_index"].nunique()),
            "mapset_count": int(frame["beatmap_set_id"].nunique()),
        }
    song_conflict_count = _key_split_conflict_count(map_rows, ["artist", "title"])
    song_version_conflict_count = _key_split_conflict_count(map_rows, ["artist", "title", "version"])
    null_mapset_count = int(map_rows["beatmap_set_id"].map(_is_missing_scalar).sum())
    duplicate_beatmap_id_count = _duplicate_nonnull_count(map_rows["beatmap_id"])
    duplicate_beatmap_path_count = _duplicate_nonnull_count(map_rows["beatmap_path"])
    return {
        "map_count": int(map_rows["source_row_index"].nunique()),
        "mapset_count": int(map_rows["beatmap_set_id"].nunique()),
        "null_mapset_id_count": null_mapset_count,
        "mapset_split_conflict_count": int((mapset_split_counts > 1).sum()),
        "map_split_conflict_count": int((map_split_counts > 1).sum()),
        "duplicate_beatmap_id_count": duplicate_beatmap_id_count,
        "duplicate_beatmap_path_count": duplicate_beatmap_path_count,
        "split_summary": split_summary,
        "map_cache_row_count": int(len(map_df)) if not map_df.empty else 0,
        "exact_artist_title_split_conflict_count": song_conflict_count,
        "exact_artist_title_version_split_conflict_count": song_version_conflict_count,
        "hard_pass": (
            null_mapset_count == 0
            and int((mapset_split_counts > 1).sum()) == 0
            and int((map_split_counts > 1).sum()) == 0
        ),
        "note": "Exact artist/title conflicts are diagnostic only; hard split unit is beatmap_set_id.",
    }


def _key_split_conflict_count(frame: pd.DataFrame, columns: Sequence[str]) -> int:
    if any(column not in frame.columns for column in columns):
        return 0
    keys = frame[[*columns, "split"]].drop_duplicates().copy()
    keys["key"] = ""
    for column in columns:
        keys["key"] = keys["key"] + "|" + keys[column].map(_normalize_text)
    counts = keys.groupby("key", dropna=False)["split"].nunique()
    return int((counts > 1).sum())


def _duplicate_nonnull_count(series: pd.Series) -> int:
    values = series[~series.map(_is_missing_scalar)]
    return int(values.duplicated(keep=False).sum())


def _guard_parity(previous_report: Mapping[str, Any], map_df: pd.DataFrame) -> dict[str, Any]:
    parity = previous_report.get("parity_summary", {}) if isinstance(previous_report, Mapping) else {}
    if isinstance(parity, Mapping) and parity:
        mismatch_keys = [
            "event_count_mismatch_count",
            "event_order_mismatch_count",
            "lane_mismatch_count",
            "action_mismatch_count",
            "chord_group_mismatch_count",
            "ln_pair_mismatch_count",
            "chunk_boundary_reconstruction_mismatch_count",
        ]
        return {
            "source": "previous_report",
            "hard_gate_pass": bool(parity.get("hard_gate_pass")),
            **{key: int(parity.get(key, 0) or 0) for key in mismatch_keys},
        }
    if map_df.empty:
        return {"source": "missing", "hard_gate_pass": False}
    mismatch_keys = [
        "event_count_mismatch_count",
        "event_order_mismatch_count",
        "lane_mismatch_count",
        "action_mismatch_count",
        "chord_group_mismatch_count",
        "ln_pair_mismatch_count",
        "chunk_boundary_reconstruction_mismatch_count",
    ]
    counts = {key: int(pd.to_numeric(map_df.get(key, 0), errors="coerce").fillna(0).sum()) for key in mismatch_keys}
    hard = bool(map_df["hard_gate_ok"].fillna(False).astype(bool).all()) and all(value == 0 for value in counts.values())
    return {"source": "map_cache", "hard_gate_pass": hard, **counts}


def _accounting_summary(chunk_df: pd.DataFrame, v2_success_source_ids: set[int] | None) -> dict[str, Any]:
    raw_order_chunks = chunk_df["raw_signature"].fillna("").astype(str).str.contains(":O", regex=False)
    canonical_order_chunks = chunk_df["canonical_signature"].fillna("").astype(str).str.contains(":O", regex=False)
    total_events = int(pd.to_numeric(chunk_df["num_events"], errors="coerce").fillna(0).sum())
    total_groups = int(pd.to_numeric(chunk_df["num_groups"], errors="coerce").fillna(0).sum())
    v2_summary: dict[str, Any] = {"available": v2_success_source_ids is not None}
    if v2_success_source_ids is not None:
        success = chunk_df[chunk_df["source_row_index"].astype(int).isin(v2_success_source_ids)]
        v2_summary.update(
            {
                "success_map_count": int(success["source_row_index"].nunique()),
                "failed_map_count": int(chunk_df["source_row_index"].nunique() - success["source_row_index"].nunique()),
                "success_event_count": int(pd.to_numeric(success["num_events"], errors="coerce").fillna(0).sum()),
                "success_event_fraction": (
                    float(pd.to_numeric(success["num_events"], errors="coerce").fillna(0).sum()) / float(total_events)
                    if total_events
                    else 0.0
                ),
            }
        )
    return {
        "chunk_count": int(len(chunk_df)),
        "event_count": total_events,
        "group_count": total_groups,
        "raw_unique_chunk_count": int(chunk_df["raw_signature"].nunique()),
        "canonical_unique_chunk_count": int(chunk_df["canonical_signature"].nunique()),
        "canonical_unique_over_raw_unique": (
            float(chunk_df["canonical_signature"].nunique()) / float(chunk_df["raw_signature"].nunique())
            if chunk_df["raw_signature"].nunique()
            else 0.0
        ),
        "raw_order_signature_chunk_count": int(raw_order_chunks.sum()),
        "raw_order_signature_chunk_rate": float(raw_order_chunks.mean()) if len(raw_order_chunks) else 0.0,
        "canonical_order_signature_chunk_count": int(canonical_order_chunks.sum()),
        "orientation_required_chunk_count": int((~chunk_df["is_mirror_symmetric"].fillna(False).astype(bool)).sum()),
        "orientation_required_chunk_rate": float((~chunk_df["is_mirror_symmetric"].fillna(False).astype(bool)).mean()) if len(chunk_df) else 0.0,
        "v2_success_subset": v2_summary,
    }


def _baselines(previous_report: Mapping[str, Any]) -> dict[str, float]:
    baselines = {
        "current_v2_bits_per_event": BASELINE_V2_BITS_PER_EVENT,
        "event_level_beat_joint_bits_per_event": BASELINE_BEAT_JOINT_BITS_PER_EVENT,
        "event_level_beat_independent_bits_per_event": BASELINE_BEAT_INDEPENDENT_BITS_PER_EVENT,
        "previous_best_motif_bits_per_event": 5.1241804761581236,
    }
    if not isinstance(previous_report, Mapping):
        return baselines
    previous_baselines = previous_report.get("baselines", {})
    if isinstance(previous_baselines, Mapping):
        for key in ("current_v2_bits_per_event", "event_level_beat_joint_bits_per_event", "event_level_beat_independent_bits_per_event"):
            if key in previous_baselines:
                baselines[key] = float(previous_baselines[key])
    comparison = previous_report.get("heldout_bits_event_comparison", {})
    if isinstance(comparison, Mapping):
        best_test = comparison.get("best_test", {})
        if isinstance(best_test, Mapping) and "bits_per_event" in best_test:
            baselines["previous_best_motif_bits_per_event"] = float(best_test["bits_per_event"])
    return baselines


def _best_variant(rows: Sequence[Mapping[str, Any]], *, split: str, require_motif: bool) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for row in rows:
        if require_motif and row.get("token_family") != "motif":
            continue
        metrics = row.get(split, {})
        if not isinstance(metrics, Mapping) or int(metrics.get("event_count", 0) or 0) <= 0:
            continue
        bits = float(metrics.get("bits_per_event", math.inf))
        if not best or bits < float(best.get(split, {}).get("bits_per_event", math.inf)):
            best = dict(row)
    return best


def _whole_chunk_failure_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary = {}
    for row in rows:
        if row.get("token_family") != "whole_chunk":
            continue
        test = row.get("test", {})
        train = row.get("train", {})
        if not isinstance(test, Mapping) or not isinstance(train, Mapping):
            continue
        summary[str(row.get("variant"))] = {
            "train_bits_per_event": train.get("bits_per_event"),
            "test_bits_per_event": test.get("bits_per_event"),
            "test_unknown_whole_chunk_rate": test.get("unknown_whole_chunk_rate"),
            "test_unknown_whole_chunk_count": test.get("unknown_whole_chunk_count"),
            "test_bits_per_chunk": test.get("bits_per_chunk"),
        }
    return summary


def _pass_criteria(
    *,
    split_integrity: Mapping[str, Any],
    guard: Mapping[str, Any],
    selected_variant: Mapping[str, Any],
    baselines: Mapping[str, float],
    split_matched_baselines: Mapping[str, Any],
    whole_failures: Mapping[str, Any],
    reconstruction_checks: Mapping[str, Any],
) -> dict[str, Any]:
    test = selected_variant.get("test", {}) if isinstance(selected_variant, Mapping) else {}
    common_test = selected_variant.get("test_v2_success_only", {}) if isinstance(selected_variant, Mapping) else {}
    test_bits = float(test.get("bits_per_event", math.inf)) if isinstance(test, Mapping) else math.inf
    test_bits_with_dictionary = (
        float(test.get("bits_per_event_with_dictionary", math.inf)) if isinstance(test, Mapping) else math.inf
    )
    common_test_bits = (
        float(common_test.get("bits_per_event", math.inf))
        if isinstance(common_test, Mapping) and common_test.get("event_count")
        else math.inf
    )
    beat_baseline = _baseline_test_bits(
        split_matched_baselines,
        family="beat_joint",
        fallback=float(baselines["event_level_beat_joint_bits_per_event"]),
    )
    v2_baseline = _baseline_test_bits(
        split_matched_baselines,
        family="current_v2_success_only",
        fallback=float(baselines["current_v2_bits_per_event"]),
    )
    whole_rates = [
        float(value.get("test_unknown_whole_chunk_rate", 0.0))
        for value in whole_failures.values()
        if isinstance(value, Mapping) and value.get("test_unknown_whole_chunk_rate") is not None
    ]
    whole_chunk_fail_pass = bool(whole_rates and min(whole_rates) >= 0.25)
    hard = bool(split_integrity.get("hard_pass")) and bool(guard.get("hard_gate_pass"))
    beats_event = test_bits < beat_baseline
    beats_v2 = test_bits < v2_baseline
    beats_v2_with_dictionary = test_bits_with_dictionary < v2_baseline
    beats_v2_common_subset = common_test_bits < v2_baseline if math.isfinite(common_test_bits) else None
    fallback_rate = float(test.get("fallback_group_rate", 1.0)) if isinstance(test, Mapping) else 1.0
    motif_coverage = float(test.get("motif_coverage_rate", 0.0)) if isinstance(test, Mapping) else 0.0
    unknown_group_rate = float(test.get("unknown_group_atom_rate", 1.0)) if isinstance(test, Mapping) else 1.0
    fallback_pass = fallback_rate <= 0.50
    coverage_pass = motif_coverage >= 0.25
    unknown_pass = unknown_group_rate <= 0.01
    reconstruction_pass = bool(reconstruction_checks.get("pass"))
    return {
        "hard_pass": hard,
        "split_integrity_pass": bool(split_integrity.get("hard_pass")),
        "guard_parity_pass": bool(guard.get("hard_gate_pass")),
        "selection_policy": "choose motif tokenizer by lowest valid bits/event; report that locked row on test",
        "valid_selected_variant": selected_variant.get("variant") if isinstance(selected_variant, Mapping) else None,
        "valid_selected_vocab_size": selected_variant.get("motif_vocab_size") if isinstance(selected_variant, Mapping) else None,
        "valid_selected_test_bits_per_event": None if not math.isfinite(test_bits) else test_bits,
        "valid_selected_test_bits_per_event_with_dictionary": (
            None if not math.isfinite(test_bits_with_dictionary) else test_bits_with_dictionary
        ),
        "valid_selected_test_v2_success_bits_per_event": (
            None if not math.isfinite(common_test_bits) else common_test_bits
        ),
        "split_matched_beat_joint_test_bits_per_event": beat_baseline,
        "split_matched_current_v2_success_test_bits_per_event": v2_baseline,
        "beats_event_level_beat_joint": bool(beats_event),
        "beats_current_v2": bool(beats_v2),
        "beats_current_v2_with_dictionary": bool(beats_v2_with_dictionary),
        "beats_current_v2_on_v2_success_subset": beats_v2_common_subset,
        "valid_selected_test_fallback_group_rate": fallback_rate,
        "fallback_rate_pass": fallback_pass,
        "valid_selected_test_motif_coverage_rate": motif_coverage,
        "motif_coverage_pass": coverage_pass,
        "valid_selected_test_unknown_group_atom_rate": unknown_group_rate,
        "unknown_group_atom_rate_pass": unknown_pass,
        "reconstruction_pass": reconstruction_pass,
        "reconstruction_mismatch_count": int(reconstruction_checks.get("mismatch_count", 0) or 0),
        "whole_chunk_failure_documented": whole_chunk_fail_pass,
        "research_pass": bool(
            hard
            and beats_event
            and beats_v2
            and beats_v2_with_dictionary
            and whole_chunk_fail_pass
            and fallback_pass
            and coverage_pass
            and unknown_pass
            and reconstruction_pass
        ),
        "mapper_integration_allowed": False,
    }


def _baseline_test_bits(
    split_matched_baselines: Mapping[str, Any],
    *,
    family: str,
    fallback: float,
) -> float:
    family_row = split_matched_baselines.get(family, {}) if isinstance(split_matched_baselines, Mapping) else {}
    test = family_row.get("test", {}) if isinstance(family_row, Mapping) else {}
    if isinstance(test, Mapping) and test.get("event_count") and test.get("bits_per_event") is not None:
        return float(test["bits_per_event"])
    return float(fallback)


def _recommendation(pass_criteria: Mapping[str, Any]) -> str:
    if not pass_criteria.get("hard_pass"):
        return "KILL_OR_FIX: split integrity or previous hard parity failed; tokenizer metrics are not trustworthy."
    if pass_criteria.get("research_pass"):
        return "TEST_NEXT_TOKENIZER_DIAGNOSTIC: benchmark passed; move to motif quality or chart-only embedding diagnostics, not mapper integration."
    if pass_criteria.get("beats_event_level_beat_joint") or pass_criteria.get("beats_current_v2"):
        return "MUTATE: compression signal remains but at least one sanity gate needs refinement before the next tokenizer card."
    return "KILL_OR_REDESIGN: heldout tokenizer compression no longer beats the event-level beat baseline."


def _write_result_log(path: Path, report: Mapping[str, Any]) -> None:
    best = report.get("valid_selected_motif", {})
    best_test = best.get("test", {}) if isinstance(best, Mapping) else {}
    best_common_test = best.get("test_v2_success_only", {}) if isinstance(best, Mapping) else {}
    split_baselines = report.get("split_matched_baselines", {})
    pass_criteria = report.get("pass_criteria", {})
    accounting = report.get("accounting_summary", {})
    split = report.get("split_integrity", {})
    reconstruction = report.get("reconstruction_checks", {})
    lines = [
        "# Result Log",
        "",
        "- Experiment: Beat-Chunk Motif Tokenizer Refinement Audit",
        "- Date: 2026-06-16",
        f"- Commit / dirty state: {report.get('code_commit')} / {report.get('code_dirty')}",
        f"- Runtime seconds: {report.get('elapsed_s')}",
        f"- Chunk cache: `{report.get('chunk_cache_path')}`",
        f"- Report: `{report.get('report_path') or DEFAULT_REPORT_PATH.as_posix()}`",
        "",
        "## Split And Guard",
        "",
        f"- Mapset split conflicts: {split.get('mapset_split_conflict_count') if isinstance(split, Mapping) else None}",
        f"- Map split conflicts: {split.get('map_split_conflict_count') if isinstance(split, Mapping) else None}",
        f"- Null mapset IDs: {split.get('null_mapset_id_count') if isinstance(split, Mapping) else None}",
        f"- Duplicate beatmap IDs: {split.get('duplicate_beatmap_id_count') if isinstance(split, Mapping) else None}",
        f"- Duplicate beatmap paths: {split.get('duplicate_beatmap_path_count') if isinstance(split, Mapping) else None}",
        f"- Previous hard parity pass: {report.get('guard_parity', {}).get('hard_gate_pass') if isinstance(report.get('guard_parity'), Mapping) else None}",
        f"- Exact artist/title split conflicts, diagnostic only: {split.get('exact_artist_title_split_conflict_count') if isinstance(split, Mapping) else None}",
        f"- Exact artist/title/version split conflicts, diagnostic only: {split.get('exact_artist_title_version_split_conflict_count') if isinstance(split, Mapping) else None}",
        "",
        "## Valid-Selected Motif Result",
        "",
        f"- Variant: {best.get('variant') if isinstance(best, Mapping) else None}",
        f"- Vocab size: {best.get('motif_vocab_size') if isinstance(best, Mapping) else None}",
        f"- Test bits/event: {best_test.get('bits_per_event') if isinstance(best_test, Mapping) else None}",
        f"- Test bits/event with amortized dictionary: {best_test.get('bits_per_event_with_dictionary') if isinstance(best_test, Mapping) else None}",
        f"- Dictionary bits/event on test: {best_test.get('dictionary_bits_per_event') if isinstance(best_test, Mapping) else None}",
        f"- Test bits/group: {best_test.get('bits_per_group') if isinstance(best_test, Mapping) else None}",
        f"- Test motif coverage: {best_test.get('motif_coverage_rate') if isinstance(best_test, Mapping) else None}",
        f"- Test fallback group rate: {best_test.get('fallback_group_rate') if isinstance(best_test, Mapping) else None}",
        f"- Test unknown group atom rate: {best_test.get('unknown_group_atom_rate') if isinstance(best_test, Mapping) else None}",
        f"- Test rare-token bits fraction: {best_test.get('rare_token_bits_fraction') if isinstance(best_test, Mapping) else None}",
        f"- Test v2-success-only bits/event: {best_common_test.get('bits_per_event') if isinstance(best_common_test, Mapping) else None}",
        "",
        "## Split-Matched Baselines",
        "",
        f"- Beat-joint test bits/event: {_nested(split_baselines, 'beat_joint', 'test', 'bits_per_event')}",
        f"- Current v2 success-only test bits/event: {_nested(split_baselines, 'current_v2_success_only', 'test', 'bits_per_event')}",
        f"- v2 tokenization failure count: {split_baselines.get('v2_tokenization_failure_count') if isinstance(split_baselines, Mapping) else None}",
        "",
        "## Reconstruction Guard",
        "",
        f"- Pass: {reconstruction.get('pass') if isinstance(reconstruction, Mapping) else None}",
        f"- Checked variants: {reconstruction.get('checked_variant_count') if isinstance(reconstruction, Mapping) else None}",
        f"- Checked chunk-variant pairs: {reconstruction.get('checked_chunk_count') if isinstance(reconstruction, Mapping) else None}",
        f"- Mismatches: {reconstruction.get('mismatch_count') if isinstance(reconstruction, Mapping) else None}",
        "",
        "## Variant Table",
        "",
        "| Variant | Vocab | Valid b/e | Test b/e | Test+dict b/e | Coverage | Fallback | Unknown group | Whole OOV |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *_variant_table_lines(report.get("variant_results", [])),
        "",
        "## Accounting",
        "",
        f"- Chunks: {accounting.get('chunk_count') if isinstance(accounting, Mapping) else None}",
        f"- Events: {accounting.get('event_count') if isinstance(accounting, Mapping) else None}",
        f"- Groups: {accounting.get('group_count') if isinstance(accounting, Mapping) else None}",
        f"- Raw unique chunks: {accounting.get('raw_unique_chunk_count') if isinstance(accounting, Mapping) else None}",
        f"- Canonical/raw unique ratio: {accounting.get('canonical_unique_over_raw_unique') if isinstance(accounting, Mapping) else None}",
        f"- Order-signature chunk count: {accounting.get('raw_order_signature_chunk_count') if isinstance(accounting, Mapping) else None}",
        f"- Orientation-required chunk rate: {accounting.get('orientation_required_chunk_rate') if isinstance(accounting, Mapping) else None}",
        "",
        "## Decision",
        "",
        f"- Research pass: {pass_criteria.get('research_pass') if isinstance(pass_criteria, Mapping) else None}",
        f"- Selection policy: {pass_criteria.get('selection_policy') if isinstance(pass_criteria, Mapping) else None}",
        f"- Fallback-rate pass: {pass_criteria.get('fallback_rate_pass') if isinstance(pass_criteria, Mapping) else None}",
        f"- Motif-coverage pass: {pass_criteria.get('motif_coverage_pass') if isinstance(pass_criteria, Mapping) else None}",
        f"- Unknown-group pass: {pass_criteria.get('unknown_group_atom_rate_pass') if isinstance(pass_criteria, Mapping) else None}",
        f"- Reconstruction pass: {pass_criteria.get('reconstruction_pass') if isinstance(pass_criteria, Mapping) else None}",
        f"- With-dictionary v2 pass: {pass_criteria.get('beats_current_v2_with_dictionary') if isinstance(pass_criteria, Mapping) else None}",
        f"- Recommendation: {report.get('recommendation')}",
        "- Mapper integration remains out of scope.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text("\n".join(lines), encoding="utf-8")
    tmp_path.replace(path)


def _variant_table_lines(rows: object) -> list[str]:
    if not isinstance(rows, list):
        return []
    lines: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        valid = row.get("valid", {})
        test = row.get("test", {})
        if not isinstance(valid, Mapping) or not isinstance(test, Mapping):
            continue
        lines.append(
            "| {variant} | {vocab} | {valid_bits} | {test_bits} | {test_dict_bits} | {coverage} | {fallback} | {unknown_group} | {whole_oov} |".format(
                variant=row.get("variant"),
                vocab="" if row.get("motif_vocab_size") is None else row.get("motif_vocab_size"),
                valid_bits=_fmt_float(valid.get("bits_per_event")),
                test_bits=_fmt_float(test.get("bits_per_event")),
                test_dict_bits=_fmt_float(test.get("bits_per_event_with_dictionary")),
                coverage=_fmt_float(test.get("motif_coverage_rate")),
                fallback=_fmt_float(test.get("fallback_group_rate")),
                unknown_group=_fmt_float(test.get("unknown_group_atom_rate")),
                whole_oov=_fmt_float(test.get("unknown_whole_chunk_rate")),
            )
        )
    return lines


def _fmt_float(value: object) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.4f}"


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def _small_hash(value: str) -> int:
    if value == ".":
        return 0
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def _normalize_text(value: object) -> str:
    if _is_missing_scalar(value):
        return ""
    return " ".join(str(value).casefold().split())


def _positive_int(value: object, field: str) -> int:
    integer = _nonnegative_int(value, field)
    if integer <= 0:
        raise ValueError(f"{field} must be positive, got {integer}")
    return integer


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field} must be an integer, got bool")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be an integer, got {type(value).__name__}") from exc
    if integer < 0:
        raise ValueError(f"{field} must be non-negative, got {integer}")
    return integer


def _git_stdout(*args: str) -> str | None:
    try:
        completed = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _parse_vocab_sizes(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _format_command(argv: Sequence[str] | None) -> str:
    args = list(sys.argv[1:] if argv is None else argv)
    return " ".join(shlex.quote(part) for part in ["python", "-m", "pulsefield_model.osu_core.beat_chunk_tokenizer_refinement_audit", *args])


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a cache-only beat-chunk tokenizer refinement benchmark.")
    parser.add_argument("--chunk-cache-path", type=Path, default=DEFAULT_BEAT_CHUNK_CACHE_PATH)
    parser.add_argument("--map-cache-path", type=Path, default=DEFAULT_BEAT_CHUNK_MAP_CACHE_PATH)
    parser.add_argument("--previous-report-path", type=Path, default=DEFAULT_BEAT_CHUNK_REPORT_PATH)
    parser.add_argument("--beat-structure-cache-path", type=Path, default=DEFAULT_LE3_STRUCTURE_CACHE_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--result-log-path", type=Path, default=DEFAULT_RESULT_LOG_PATH)
    parser.add_argument("--motif-vocab-sizes", type=_parse_vocab_sizes, default=DEFAULT_MOTIF_VOCAB_SIZES)
    parser.add_argument("--motif-min-n", type=int, default=DEFAULT_MOTIF_MIN_N)
    parser.add_argument("--motif-max-n", type=int, default=DEFAULT_MOTIF_MAX_N)
    parser.add_argument("--smoothing-alpha", type=float, default=DEFAULT_SMOOTHING_ALPHA)
    parser.add_argument("--rare-max-count", type=int, default=DEFAULT_RARE_MAX_COUNT)
    parser.add_argument("--top-motif-limit", type=int, default=DEFAULT_TOP_MOTIF_LIMIT)
    parser.add_argument("--limit-chunks", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    report = audit_beat_chunk_tokenizer_refinement(
        chunk_cache_path=args.chunk_cache_path,
        map_cache_path=args.map_cache_path,
        previous_report_path=args.previous_report_path,
        beat_structure_cache_path=args.beat_structure_cache_path,
        report_path=args.report_path,
        result_log_path=args.result_log_path,
        motif_vocab_sizes=args.motif_vocab_sizes,
        motif_min_n=args.motif_min_n,
        motif_max_n=args.motif_max_n,
        smoothing_alpha=args.smoothing_alpha,
        rare_max_count=args.rare_max_count,
        top_motif_limit=args.top_motif_limit,
        limit_chunks=args.limit_chunks,
        command=_format_command(argv),
    )
    print(json.dumps(_normalize_json(report), allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
