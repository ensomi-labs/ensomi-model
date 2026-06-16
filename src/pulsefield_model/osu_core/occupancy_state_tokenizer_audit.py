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
    _make_model_token_bits,
    _match_motif,
    _rank_motifs,
    _score_encoded_generic,
    _split_report,
)
from pulsefield_model.osu_core.fallback_forensic_audit import (
    EXPECTED_R0_TEST_CHARGED_BITS_PER_EVENT,
    _count_bucket,
    _group_class,
    _read_chunk_cache,
)


SCHEMA_VERSION: Final[int] = 1
DEFAULT_REPORT_PATH: Final[Path] = Path(
    "artifacts/reports/audits/occupancy_factorized_tokenization/occupancy_state_report.json",
)
DEFAULT_RESULT_LOG_PATH: Final[Path] = Path(
    "artifacts/reports/audits/occupancy_factorized_tokenization/occupancy_state_result_log.md",
)
DEFAULT_COMPARISON_CSV_PATH: Final[Path] = Path(
    "artifacts/reports/audits/occupancy_factorized_tokenization/occupancy_state_comparison.csv",
)


@dataclass(frozen=True)
class _StreamToken:
    token: str
    group: Mapping[str, Any] | None = None
    beatmap_set_id: int | None = None
    split: str | None = None
    chunk_index: int | None = None


@dataclass(frozen=True)
class _Stream:
    source_row_index: int
    segment_id: int
    split: str
    beatmap_set_id: int
    chunk_count: int
    event_count: int
    group_count: int
    raw_signatures: tuple[str, ...]
    tokens: tuple[_StreamToken, ...]
    invalid_transition_count: int
    segment_end_hold_count: int


@dataclass(frozen=True)
class _FittedStreamModel:
    variant: str
    vocab: _MotifVocab
    trie: Mapping[str, Any]
    atom_counter: Counter[str]
    train_counter: Counter[str]
    dictionary_bits: float
    token_bits: Callable[[str], float]
    learned_motif_count: int
    top_motifs: list[dict[str, Any]]


@dataclass
class _BucketAgg:
    split: str
    table: str
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


def audit_occupancy_state_tokenizer(
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
    """Run O1 chart-stream occupancy-state tokenizer audit."""

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
    streams_by_variant = {
        "chart_delta_raw": _build_streams(chunk_df, variant="chart_delta_raw"),
        "o1_occupancy_release_state": _build_streams(chunk_df, variant="o1_occupancy_release_state"),
    }
    reports: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []
    for variant, streams in streams_by_variant.items():
        model = _fit_stream_model(
            streams,
            variant=variant,
            motif_vocab_size=motif_vocab_size,
            motif_min_n=motif_min_n,
            motif_max_n=motif_max_n,
            smoothing_alpha=smoothing_alpha,
            top_motif_limit=top_motif_limit,
        )
        split_reports: dict[str, Any] = {}
        bucket_tables: dict[str, list[dict[str, Any]]] = {}
        for split in ("train", "valid", "test"):
            stats, bucket_aggs = _score_stream_split(
                streams,
                split=split,
                model=model,
                rare_max_count=rare_max_count,
            )
            split_reports[split] = _split_report(stats, dictionary_bits=model.dictionary_bits)
            rows.append(_comparison_row(variant, split, "global", split_reports[split]))
            bucket_rows = _bucket_rows(
                bucket_aggs,
                variant=variant,
                split=split,
                split_event_count=int(split_reports[split]["event_count"]),
                dictionary_bits=model.dictionary_bits,
            )
            bucket_tables[split] = bucket_rows
            rows.extend(bucket_rows)
        reconstruction = _reconstruction_guard(streams, model=model, variant=variant)
        reconstruction_rows.append({"variant": variant, **reconstruction})
        reports[variant] = {
            "variant": variant,
            "learned_motif_count": model.learned_motif_count,
            "dictionary_cost_bits": model.dictionary_bits,
            "atom_vocab_size": len(model.atom_counter),
            "train_encoded_vocab_size": len(model.train_counter),
            "splits": split_reports,
            "bucket_tables": bucket_tables,
            "state_diagnostics": _state_diagnostics(streams),
            "reconstruction": reconstruction,
            "top_motifs": model.top_motifs,
        }

    pass_criteria = _pass_criteria(reports)
    recommendation = _recommendation(pass_criteria)
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "Occupancy-State Tokenizer O1",
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
            "chart_stream_policy": "one C:<chunk_start_delta>:<bar_phase> token per chunk; group deltas are chart-ordered within each source/segment stream",
            "occupancy_policy": "release code is deterministic from active hold state only for release-all; invalid or partial releases encode exact masks",
        },
        "dataset": _dataset_summary(chunk_df),
        "external_baseline": {
            "variant": "r0_delta_chunk_local",
            "test_bits_per_event_with_dictionary": EXPECTED_R0_TEST_CHARGED_BITS_PER_EVENT,
            "source": "duration/fallback matched scorer baseline",
        },
        "variants": reports,
        "reconstruction_guard": {
            "pass": all(bool(row.get("pass")) for row in reconstruction_rows),
            "rows": reconstruction_rows,
        },
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


def _build_streams(chunk_df: pd.DataFrame, *, variant: str) -> list[_Stream]:
    streams: list[_Stream] = []
    ordered = chunk_df.sort_values(["source_row_index", "segment_id", "start_beat_units", "chunk_index"])
    for (source, segment), frame in ordered.groupby(["source_row_index", "segment_id"], sort=False):
        rows = list(frame.itertuples(index=False))
        if not rows:
            continue
        split = str(rows[0].split)
        beatmap_set_id = int(rows[0].beatmap_set_id)
        tokens, invalid_count, end_hold_count = _stream_tokens_for_rows(rows, variant=variant)
        streams.append(
            _Stream(
                source_row_index=int(source),
                segment_id=int(segment),
                split=split,
                beatmap_set_id=beatmap_set_id,
                chunk_count=len(rows),
                event_count=sum(int(row.num_events) for row in rows),
                group_count=sum(int(row.num_groups) for row in rows),
                raw_signatures=tuple(str(row.raw_signature or "") for row in rows),
                tokens=tuple(tokens),
                invalid_transition_count=invalid_count,
                segment_end_hold_count=end_hold_count,
            )
        )
    return streams


def _stream_tokens_for_rows(rows: Sequence[Any], *, variant: str) -> tuple[list[_StreamToken], int, int]:
    tokens: list[_StreamToken] = []
    previous_chunk_start: int | None = None
    previous_event_abs: int | None = None
    active_hold_mask = 0
    invalid_transition_count = 0
    for row in rows:
        chunk_start = int(row.start_beat_units)
        chunk_delta = chunk_start if previous_chunk_start is None else chunk_start - previous_chunk_start
        previous_chunk_start = chunk_start
        tokens.append(_StreamToken(token=f"C:{chunk_delta}:{int(row.bar_phase_half)}"))
        groups = json.loads(str(row.groups_json or "[]"))
        for group in groups:
            offset = int(group.get("offset_units", 0))
            event_abs = chunk_start + offset
            delta = event_abs - (chunk_start if previous_event_abs is None else previous_event_abs)
            previous_event_abs = event_abs
            tap = int(group.get("tap_mask", 0))
            start = int(group.get("ln_start_mask", 0))
            end = int(group.get("ln_end_mask", 0))
            invalid_transition_count += ((end & ~active_hold_mask) | (start & active_hold_mask)).bit_count()
            if variant == "chart_delta_raw":
                token = f"D:{delta}:{tap}:{start}:{end}"
            elif variant == "o1_occupancy_release_state":
                token = f"O:{delta}:{tap}:{start}:{_release_code(active_hold_mask, end)}"
            else:
                raise ValueError(f"unknown occupancy stream variant: {variant}")
            order_signature = str(group.get("order_signature", ".") or ".")
            if order_signature != ".":
                token = f"{token}:O{order_signature}"
            tokens.append(
                _StreamToken(
                    token=token,
                    group={
                        "offset_units": offset,
                        "event_abs_units": event_abs,
                        "tap_mask": tap,
                        "ln_start_mask": start,
                        "ln_end_mask": end,
                        "pre_hold_mask": active_hold_mask,
                        "order_signature": order_signature,
                        "event_count": max(1, int(group.get("event_count", 1) or 1)),
                    },
                    beatmap_set_id=int(row.beatmap_set_id),
                    split=str(row.split),
                    chunk_index=int(row.chunk_index),
                )
            )
            active_hold_mask = (active_hold_mask | start) & ~end
    return tokens, invalid_transition_count, active_hold_mask.bit_count()


def _release_code(active_hold_mask: int, end_mask: int) -> str:
    active_hold_mask = int(active_hold_mask)
    end_mask = int(end_mask)
    if end_mask == 0:
        return "E0"
    if end_mask & ~active_hold_mask:
        return f"EX{end_mask}"
    if end_mask == active_hold_mask:
        return "EA"
    return f"EM{end_mask}"


def _release_mask_from_code(active_hold_mask: int, code: str) -> int:
    if code == "E0":
        return 0
    if code == "EA":
        return int(active_hold_mask)
    if code.startswith("EM") or code.startswith("EX"):
        return int(code[2:])
    return 0


def _fit_stream_model(
    streams: Sequence[_Stream],
    *,
    variant: str,
    motif_vocab_size: int,
    motif_min_n: int,
    motif_max_n: int,
    smoothing_alpha: float,
    top_motif_limit: int,
) -> _FittedStreamModel:
    atom_counter: Counter[str] = Counter()
    candidate_counter: Counter[tuple[str, ...]] = Counter()
    for stream in streams:
        if stream.split != "train":
            continue
        token_strings = [token.token for token in stream.tokens]
        atom_counter.update(token_strings)
        max_n = min(motif_max_n, len(token_strings))
        for ngram in range(motif_min_n, max_n + 1):
            for index in range(len(token_strings) - ngram + 1):
                candidate_counter[tuple(token_strings[index : index + ngram])] += 1
    candidates = _rank_motifs(candidate_counter)
    vocab = _build_motif_vocab(candidates[:motif_vocab_size])
    trie = _build_motif_trie(vocab.motif_to_id)
    train_counter: Counter[str] = Counter()
    for stream in streams:
        if stream.split != "train":
            continue
        encoded, _covered = _encode_stream_with_spans(stream.tokens, trie)
        train_counter.update(token for token, _span in encoded)
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
    return _FittedStreamModel(
        variant=variant,
        vocab=vocab,
        trie=trie,
        atom_counter=atom_counter,
        train_counter=train_counter,
        dictionary_bits=dictionary_bits,
        token_bits=token_bits,
        learned_motif_count=len(vocab.motif_to_id),
        top_motifs=_top_motifs(candidates[:top_motif_limit]),
    )


def _score_stream_split(
    streams: Sequence[_Stream],
    *,
    split: str,
    model: _FittedStreamModel,
    rare_max_count: int,
) -> tuple[_EncodedStats, dict[tuple[str, str], _BucketAgg]]:
    stats = _EncodedStats()
    bucket_aggs: dict[tuple[str, str], _BucketAgg] = {}
    for stream in streams:
        if stream.split != split:
            continue
        encoded, covered = _encode_stream_with_spans(stream.tokens, model.trie)
        stats.chunk_count += stream.chunk_count
        stats.event_count += stream.event_count
        stats.group_count += stream.group_count
        stats.mapset_events[stream.beatmap_set_id] += stream.event_count
        for token, _span in encoded:
            stats.token_counter[token] += 1
            stats.token_count += 1
            if token.startswith("M"):
                stats.motif_token_count += 1
            else:
                stats.atom_token_count += 1
        for index in covered:
            if stream.tokens[index].group is not None:
                stats.motif_group_coverage_count += 1
        group_bits = _group_bits(stream.tokens, encoded=encoded, token_bits=model.token_bits)
        for index, token in enumerate(stream.tokens):
            if token.group is None:
                continue
            fallback = index not in covered
            event_count = int(token.group.get("event_count", 1) or 1)
            for table, bucket in _bucket_values(token.group):
                key = (table, bucket)
                agg = bucket_aggs.setdefault(key, _BucketAgg(split=split, table=table, bucket=bucket))
                agg.add(
                    beatmap_set_id=stream.beatmap_set_id,
                    event_count=event_count,
                    fallback=fallback,
                    code_bits=group_bits.get(index, 0.0),
                )
    _score_encoded_generic(
        stats,
        token_bits=model.token_bits,
        dictionary_bits=model.dictionary_bits,
        rare_max_count=rare_max_count,
    )
    return stats, bucket_aggs


def _encode_stream_with_spans(
    tokens: Sequence[_StreamToken],
    trie: Mapping[str, Any],
) -> tuple[list[tuple[str, tuple[int, ...]]], set[int]]:
    token_strings = [token.token for token in tokens]
    encoded: list[tuple[str, tuple[int, ...]]] = []
    covered: set[int] = set()
    index = 0
    while index < len(token_strings):
        matched_token, matched_length = _match_motif(token_strings, index, trie)
        if matched_token is not None:
            span = tuple(range(index, index + matched_length))
            encoded.append((matched_token, span))
            covered.update(span)
            index += matched_length
            continue
        encoded.append((token_strings[index], (index,)))
        index += 1
    return encoded, covered


def _group_bits(
    tokens: Sequence[_StreamToken],
    *,
    encoded: Sequence[tuple[str, tuple[int, ...]]],
    token_bits: Callable[[str], float],
) -> dict[int, float]:
    bits: dict[int, float] = {}
    for token, span in encoded:
        group_indexes = [index for index in span if tokens[index].group is not None]
        if not group_indexes:
            continue
        share = token_bits(token) / float(len(group_indexes))
        for index in group_indexes:
            bits[index] = bits.get(index, 0.0) + share
    return bits


def _bucket_values(group: Mapping[str, Any]) -> list[tuple[str, str]]:
    tap = int(group.get("tap_mask", 0))
    start = int(group.get("ln_start_mask", 0))
    end = int(group.get("ln_end_mask", 0))
    pre_hold = int(group.get("pre_hold_mask", 0))
    release_code = _release_code(pre_hold, end)
    return [
        ("overall", "all"),
        ("group_class", _group_class(tap, start, end)),
        ("active_hold_count", _count_bucket(pre_hold.bit_count())),
        ("release_code_kind", _release_code_kind(release_code)),
        ("group_class_x_active", f"{_group_class(tap, start, end)}|active_{_count_bucket(pre_hold.bit_count())}"),
    ]


def _release_code_kind(code: str) -> str:
    if code in {"E0", "EA"}:
        return code
    if code.startswith("EM"):
        return "EM_exact_partial"
    if code.startswith("EX"):
        return "EX_invalid_exact"
    return "unknown"


def _bucket_rows(
    aggs: Mapping[tuple[str, str], _BucketAgg],
    *,
    variant: str,
    split: str,
    split_event_count: int,
    dictionary_bits: float,
) -> list[dict[str, Any]]:
    dict_bits_per_event = float(dictionary_bits) / float(split_event_count) if split_event_count else 0.0
    rows = []
    for (table, bucket), agg in sorted(aggs.items()):
        bits = float(agg.code_bits) / float(agg.event_count) if agg.event_count else 0.0
        rows.append(
            {
                "variant": variant,
                "split": split,
                "table": table,
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


def _reconstruction_guard(
    streams: Sequence[_Stream],
    *,
    model: _FittedStreamModel,
    variant: str,
) -> dict[str, Any]:
    id_to_motif = {token_id: motif for motif, token_id in model.vocab.motif_to_id.items()}
    mismatch_count = 0
    checked_chunk_count = 0
    examples: list[dict[str, Any]] = []
    for stream in streams:
        encoded, _covered = _encode_stream_with_spans(stream.tokens, model.trie)
        expanded = _expand_motifs([token for token, _span in encoded], id_to_motif)
        reconstructed = _decode_stream_signatures(expanded, variant=variant)
        checked_chunk_count += len(stream.raw_signatures)
        if tuple(reconstructed) == stream.raw_signatures:
            continue
        for expected, got in zip(stream.raw_signatures, reconstructed, strict=False):
            if expected == got:
                continue
            mismatch_count += 1
            if len(examples) < 10:
                examples.append(
                    {
                        "source_row_index": stream.source_row_index,
                        "segment_id": stream.segment_id,
                        "expected": expected,
                        "reconstructed": got,
                        "encoded_prefix": [token for token, _span in encoded[:20]],
                    }
                )
        if len(reconstructed) != len(stream.raw_signatures):
            mismatch_count += abs(len(reconstructed) - len(stream.raw_signatures))
    return {
        "checked_chunk_count": checked_chunk_count,
        "mismatch_count": mismatch_count,
        "pass": mismatch_count == 0 and checked_chunk_count > 0,
        "examples": examples,
    }


def _expand_motifs(tokens: Sequence[str], id_to_motif: Mapping[str, tuple[str, ...]]) -> list[str]:
    expanded: list[str] = []
    for token in tokens:
        motif = id_to_motif.get(token)
        if motif is None:
            expanded.append(token)
        else:
            expanded.extend(motif)
    return expanded


def _decode_stream_signatures(tokens: Sequence[str], *, variant: str) -> list[str]:
    signatures: list[list[str]] = []
    current_groups: list[str] | None = None
    current_chunk_start = 0
    previous_chunk_start: int | None = None
    previous_event_abs: int | None = None
    active_hold_mask = 0
    for token in tokens:
        if token.startswith("C:"):
            if current_groups is not None:
                signatures.append(current_groups)
            parts = token.split(":")
            chunk_delta = int(parts[1]) if len(parts) > 1 else 0
            current_chunk_start = chunk_delta if previous_chunk_start is None else previous_chunk_start + chunk_delta
            previous_chunk_start = current_chunk_start
            current_groups = []
            continue
        if current_groups is None:
            current_groups = []
        if token.startswith("D:"):
            delta, tap, start, end, order = _parse_delta_token(token)
        elif token.startswith("O:"):
            delta, tap, start, end, order = _parse_occupancy_token(token, active_hold_mask=active_hold_mask)
        else:
            continue
        if previous_event_abs is None:
            event_abs = current_chunk_start + delta
        else:
            event_abs = previous_event_abs + delta
        previous_event_abs = event_abs
        offset = event_abs - current_chunk_start
        suffix = "" if order == "." else f":O{order}"
        current_groups.append(f"A:{offset}:{tap}:{start}:{end}{suffix}")
        active_hold_mask = (active_hold_mask | start) & ~end
    if current_groups is not None:
        signatures.append(current_groups)
    return [";".join(groups) for groups in signatures]


def _parse_delta_token(token: str) -> tuple[int, int, int, int, str]:
    parts = token.split(":")
    order = "."
    if len(parts) > 5 and parts[5].startswith("O"):
        order = parts[5][1:]
    return int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), order


def _parse_occupancy_token(token: str, *, active_hold_mask: int) -> tuple[int, int, int, int, str]:
    parts = token.split(":")
    order = "."
    if len(parts) > 5 and parts[5].startswith("O"):
        order = parts[5][1:]
    end = _release_mask_from_code(active_hold_mask, parts[4])
    return int(parts[1]), int(parts[2]), int(parts[3]), end, order


def _state_diagnostics(streams: Sequence[_Stream]) -> dict[str, Any]:
    by_split: dict[str, dict[str, int]] = {}
    for stream in streams:
        row = by_split.setdefault(stream.split, {"invalid_transition_count": 0, "segment_end_hold_count": 0, "stream_count": 0})
        row["invalid_transition_count"] += int(stream.invalid_transition_count)
        row["segment_end_hold_count"] += int(stream.segment_end_hold_count)
        row["stream_count"] += 1
    return {
        "by_split": by_split,
        "invalid_transition_count": sum(row["invalid_transition_count"] for row in by_split.values()),
        "segment_end_hold_count": sum(row["segment_end_hold_count"] for row in by_split.values()),
    }


def _pass_criteria(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    control = reports.get("chart_delta_raw", {})
    occupancy = reports.get("o1_occupancy_release_state", {})
    control_test = control.get("splits", {}).get("test", {}) if isinstance(control, Mapping) else {}
    occupancy_test = occupancy.get("splits", {}).get("test", {}) if isinstance(occupancy, Mapping) else {}
    control_bits = float(control_test.get("bits_per_event_with_dictionary", math.inf))
    occupancy_bits = float(occupancy_test.get("bits_per_event_with_dictionary", math.inf))
    external_bits = EXPECTED_R0_TEST_CHARGED_BITS_PER_EVENT
    reconstruction_pass = bool(control.get("reconstruction", {}).get("pass")) and bool(occupancy.get("reconstruction", {}).get("pass"))
    active_delta = _bucket_delta(
        reports,
        split="test",
        table="active_hold_count",
        buckets=("1", "2", "3plus"),
    )
    release_all_delta = _bucket_delta(
        reports,
        split="test",
        table="release_code_kind",
        buckets=("EA",),
    )
    return {
        "reconstruction_pass": reconstruction_pass,
        "external_global_nonregress_pass": occupancy_bits <= external_bits + 0.01,
        "chart_control_nonregress_pass": occupancy_bits <= control_bits + 0.01,
        "active_hold_bucket_improve_pass": active_delta is not None and active_delta <= -0.05,
        "release_all_bucket_improve_pass": release_all_delta is not None and release_all_delta <= -0.05,
        "external_r0_test_bits_per_event_with_dictionary": external_bits,
        "chart_delta_test_bits_per_event_with_dictionary": control_bits,
        "occupancy_test_bits_per_event_with_dictionary": occupancy_bits,
        "occupancy_vs_external_delta_bits_per_event": occupancy_bits - external_bits,
        "occupancy_vs_chart_delta_bits_per_event": occupancy_bits - control_bits,
        "active_hold_bucket_delta_bits_per_event": active_delta,
        "release_all_bucket_delta_bits_per_event": release_all_delta,
        "research_pass": bool(
            reconstruction_pass
            and occupancy_bits <= external_bits + 0.01
            and (active_delta is not None and active_delta <= -0.05)
        ),
    }


def _bucket_delta(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    split: str,
    table: str,
    buckets: Sequence[str],
) -> float | None:
    values: dict[str, list[float]] = {"chart_delta_raw": [0.0, 0.0], "o1_occupancy_release_state": [0.0, 0.0]}
    for variant in values:
        report = reports.get(variant, {})
        rows = report.get("bucket_tables", {}).get(split, []) if isinstance(report, Mapping) else []
        for row in rows:
            if row.get("table") != table or row.get("bucket") not in buckets:
                continue
            events = float(row.get("event_count", 0) or 0)
            charged = float(row.get("bits_per_event_with_dictionary", 0.0) or 0.0)
            values[variant][0] += charged * events
            values[variant][1] += events
    if not values["chart_delta_raw"][1] or not values["o1_occupancy_release_state"][1]:
        return None
    control = values["chart_delta_raw"][0] / values["chart_delta_raw"][1]
    occupancy = values["o1_occupancy_release_state"][0] / values["o1_occupancy_release_state"][1]
    return occupancy - control


def _recommendation(pass_criteria: Mapping[str, Any]) -> str:
    if pass_criteria.get("research_pass"):
        return "TEST_NEXT_COMBO: O1 passed; compare against selective wildcard residual before combining."
    if pass_criteria.get("chart_control_nonregress_pass") and not pass_criteria.get("external_global_nonregress_pass"):
        return "MUTATE_O1: occupancy release coding helps within chart-stream framing, but chart-stream cost regresses against r0_delta."
    if pass_criteria.get("active_hold_bucket_improve_pass") and not pass_criteria.get("external_global_nonregress_pass"):
        return "MUTATE_O1: active-hold buckets improve, but global cost fails; try selective active-state coding only on fallback spans."
    return "KILL_OR_DEPRIORITIZE_O1: occupancy-state release coding did not satisfy global or active-hold gates."


def _comparison_row(variant: str, split: str, table: str, values: Mapping[str, Any]) -> dict[str, Any]:
    row = {"variant": variant, "split": split, "table": table, "bucket": "all"}
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
        "# Occupancy-State Tokenizer Result Log",
        "",
        "## Summary",
        "",
        f"- Research pass: {pass_criteria.get('research_pass')}",
        f"- Recommendation: {report.get('recommendation')}",
        f"- Runtime seconds: {_fmt_float(report.get('elapsed_s'))}",
        f"- Reconstruction pass: {pass_criteria.get('reconstruction_pass')}",
        f"- External r0 test charged bits/event: {_fmt_float(pass_criteria.get('external_r0_test_bits_per_event_with_dictionary'))}",
        f"- Chart-delta test charged bits/event: {_fmt_float(pass_criteria.get('chart_delta_test_bits_per_event_with_dictionary'))}",
        f"- Occupancy test charged bits/event: {_fmt_float(pass_criteria.get('occupancy_test_bits_per_event_with_dictionary'))}",
        f"- Occupancy vs external delta: {_fmt_float(pass_criteria.get('occupancy_vs_external_delta_bits_per_event'))}",
        f"- Occupancy vs chart-delta delta: {_fmt_float(pass_criteria.get('occupancy_vs_chart_delta_bits_per_event'))}",
        f"- Active-hold bucket delta: {_fmt_float(pass_criteria.get('active_hold_bucket_delta_bits_per_event'))}",
        f"- Release-all bucket delta: {_fmt_float(pass_criteria.get('release_all_bucket_delta_bits_per_event'))}",
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
    return " ".join(shlex.quote(part) for part in ["python", "-m", "pulsefield_model.osu_core.occupancy_state_tokenizer_audit", *args])


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit chart-stream occupancy-state tokenizer.")
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
    report = audit_occupancy_state_tokenizer(
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
        "occupancy_state_tokenizer_audit "
        f"research_pass={report.get('pass_criteria', {}).get('research_pass')} "
        f"recommendation={report.get('recommendation')} "
        f"report={report.get('report_path')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
