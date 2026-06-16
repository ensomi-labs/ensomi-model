from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np
import pandas as pd

from pulsefield_model.osu_core.beat_chunk_pattern_audit import DEFAULT_BEAT_CHUNK_CACHE_PATH
from pulsefield_model.osu_core.beat_chunk_tokenizer_refinement_audit import (
    DEFAULT_REPORT_PATH as DEFAULT_REFINEMENT_REPORT_PATH,
    DEFAULT_RARE_MAX_COUNT,
    DEFAULT_SMOOTHING_ALPHA,
    _VariantConfig,
    _build_motif_trie,
    _build_motif_vocab,
    _encode_group_sequence,
    _evaluate_variant,
    _group_tokens_for_row,
    _learn_motif_candidates,
    _match_motif,
    _normalize_text,
    _read_chunk_cache,
    _read_v2_success_source_ids,
    _split_by_source,
    _split_matched_baselines,
)
from pulsefield_model.osu_core.beat_structure_token_audit import DEFAULT_LE3_STRUCTURE_CACHE_PATH


SCHEMA_VERSION: Final[int] = 1
DEFAULT_REPORT_PATH: Final[Path] = Path(
    "artifacts/reports/audits/beat_chunk_motif_quality/beat_chunk_motif_quality_audit_le3.json",
)
DEFAULT_RESULT_LOG_PATH: Final[Path] = Path(
    "artifacts/reports/audits/beat_chunk_motif_quality/beat_chunk_motif_quality_result_log.md",
)
DEFAULT_MOTIF_VOCAB_SIZE: Final[int] = 16_384
DEFAULT_MOTIF_MIN_N: Final[int] = 2
DEFAULT_MOTIF_MAX_N: Final[int] = 8
DEFAULT_TOP_MOTIF_LIMIT: Final[int] = 100


def audit_beat_chunk_motif_quality(
    *,
    chunk_cache_path: str | Path = DEFAULT_BEAT_CHUNK_CACHE_PATH,
    refinement_report_path: str | Path = DEFAULT_REFINEMENT_REPORT_PATH,
    beat_structure_cache_path: str | Path | None = DEFAULT_LE3_STRUCTURE_CACHE_PATH,
    report_path: str | Path | None = DEFAULT_REPORT_PATH,
    result_log_path: str | Path | None = DEFAULT_RESULT_LOG_PATH,
    motif_vocab_size: int = DEFAULT_MOTIF_VOCAB_SIZE,
    motif_min_n: int = DEFAULT_MOTIF_MIN_N,
    motif_max_n: int = DEFAULT_MOTIF_MAX_N,
    top_motif_limit: int = DEFAULT_TOP_MOTIF_LIMIT,
    smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
    rare_max_count: int = DEFAULT_RARE_MAX_COUNT,
    limit_chunks: int | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    """Audit selected beat-chunk motifs for support breadth and leakage sensitivity."""

    started_at = time.perf_counter()
    chunk_cache_path = Path(chunk_cache_path)
    refinement_report_path = Path(refinement_report_path)
    beat_structure_cache_path = None if beat_structure_cache_path is None else Path(beat_structure_cache_path)
    report_path = None if report_path is None else Path(report_path)
    result_log_path = None if result_log_path is None else Path(result_log_path)
    motif_vocab_size = _positive_int(motif_vocab_size, "motif_vocab_size")
    motif_min_n = _positive_int(motif_min_n, "motif_min_n")
    motif_max_n = _positive_int(motif_max_n, "motif_max_n")
    if motif_max_n < motif_min_n:
        raise ValueError("motif_max_n must be >= motif_min_n")
    top_motif_limit = _nonnegative_int(top_motif_limit, "top_motif_limit")
    if smoothing_alpha <= 0:
        raise ValueError(f"smoothing_alpha must be positive, got {smoothing_alpha!r}")

    chunk_df = _read_chunk_cache(chunk_cache_path, limit_chunks=limit_chunks)
    refinement_report = _read_json(refinement_report_path)
    guard = _guard_from_refinement(refinement_report)
    variant = _VariantConfig("motif_delta_raw", "motif", "delta", "raw", False, True)
    candidates = _learn_motif_candidates(
        chunk_df,
        variant=variant,
        motif_min_n=motif_min_n,
        motif_max_n=motif_max_n,
    )
    vocab = _build_motif_vocab(candidates[:motif_vocab_size])
    motif_support = _motif_support_audit(
        chunk_df,
        variant=variant,
        candidates=candidates,
        motif_vocab_size=motif_vocab_size,
        top_motif_limit=top_motif_limit,
    )
    conflicts = _cross_split_key_conflicts(chunk_df)
    filtered_df = _filter_conflict_keys(chunk_df, conflicts["artist_title_version_keys"])
    v2_success_source_ids = _read_v2_success_source_ids(beat_structure_cache_path)
    filtered_rows = _evaluate_variant(
        filtered_df,
        variant=variant,
        motif_vocab_sizes=(motif_vocab_size,),
        motif_min_n=motif_min_n,
        motif_max_n=motif_max_n,
        smoothing_alpha=smoothing_alpha,
        rare_max_count=rare_max_count,
        top_motif_limit=min(20, top_motif_limit),
        v2_success_source_ids=v2_success_source_ids,
    )
    filtered_baselines = _split_matched_baselines(
        beat_structure_cache_path,
        split_by_source=_split_by_source(filtered_df),
        smoothing_alpha=smoothing_alpha,
        rare_max_count=rare_max_count,
    )
    fallback = _fallback_concentration(chunk_df, variant=variant, vocab=vocab)
    filtered_row = filtered_rows[0] if filtered_rows else {}
    pass_criteria = _pass_criteria(
        guard=guard,
        motif_support=motif_support,
        filtered_row=filtered_row,
        filtered_baselines=filtered_baselines,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "chunk_cache_path": chunk_cache_path.as_posix(),
        "refinement_report_path": refinement_report_path.as_posix(),
        "beat_structure_cache_path": None if beat_structure_cache_path is None else beat_structure_cache_path.as_posix(),
        "report_path": None if report_path is None else report_path.as_posix(),
        "result_log_path": None if result_log_path is None else result_log_path.as_posix(),
        "limited": limit_chunks is not None,
        "limit_chunks": limit_chunks,
        "config": {
            "variant": variant.name,
            "motif_vocab_size": motif_vocab_size,
            "motif_min_n": motif_min_n,
            "motif_max_n": motif_max_n,
            "top_motif_limit": top_motif_limit,
            "smoothing_alpha": smoothing_alpha,
            "rare_max_count": rare_max_count,
        },
        "guard": guard,
        "cross_split_conflicts": conflicts,
        "filtered_compression": filtered_row,
        "filtered_split_matched_baselines": filtered_baselines,
        "motif_support": motif_support,
        "fallback_concentration": fallback,
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


def _motif_support_audit(
    chunk_df: pd.DataFrame,
    *,
    variant: _VariantConfig,
    candidates: Sequence[tuple[tuple[str, ...], int, float]],
    motif_vocab_size: int,
    top_motif_limit: int,
) -> dict[str, Any]:
    selected = list(candidates[:motif_vocab_size])
    top_count = min(top_motif_limit, len(selected))
    top_ids = {f"M{index}" for index in range(top_count)}
    motif_by_id = {f"M{index}": motif for index, (motif, _count, _gain) in enumerate(selected[:top_count])}
    train_support_by_id = {f"M{index}": int(count) for index, (_motif, count, _gain) in enumerate(selected[:top_count])}
    gain_by_id = {f"M{index}": float(gain) for index, (_motif, _count, gain) in enumerate(selected[:top_count])}
    vocab = _build_motif_vocab(selected)
    trie = _build_motif_trie(vocab.motif_to_id)
    stats: dict[str, dict[str, Any]] = {
        motif_id: {
            "occurrence_count": 0,
            "split_counts": Counter(),
            "mapsets": set(),
            "maps": set(),
            "density_bins": Counter(),
            "top_mapsets": Counter(),
            "difficulty_sum": 0.0,
            "density_sum": 0.0,
            "chord_sum": 0.0,
            "ln_sum": 0.0,
        }
        for motif_id in top_ids
    }
    for row in chunk_df.itertuples(index=False):
        tokens = _group_tokens_for_row(row, variant=variant)
        encoded = _encode_group_sequence(tokens, trie)
        for token in encoded:
            if token not in top_ids:
                continue
            item = stats[token]
            item["occurrence_count"] += 1
            item["split_counts"][str(row.split)] += 1
            item["mapsets"].add(int(row.beatmap_set_id))
            item["maps"].add(int(row.source_row_index))
            item["density_bins"][str(row.density_bin)] += 1
            item["top_mapsets"][int(row.beatmap_set_id)] += 1
            item["difficulty_sum"] += float(row.difficulty or 0.0)
            item["density_sum"] += float(row.density_events_per_second or 0.0)
            item["chord_sum"] += float(row.chord_ratio or 0.0)
            item["ln_sum"] += float(row.ln_ratio or 0.0)
    rows: list[dict[str, Any]] = []
    broad_count = 0
    single_mapset_dominated_count = 0
    for index, motif_id in enumerate(sorted(top_ids, key=lambda value: int(value[1:]))):
        motif = motif_by_id[motif_id]
        item = stats[motif_id]
        occurrence_count = int(item["occurrence_count"])
        top_mapset_count = item["top_mapsets"].most_common(1)[0][1] if item["top_mapsets"] else 0
        top_mapset_fraction = float(top_mapset_count) / float(occurrence_count) if occurrence_count else 0.0
        mapset_count = len(item["mapsets"])
        if mapset_count >= 25 and top_mapset_fraction <= 0.20:
            broad_count += 1
        if top_mapset_fraction >= 0.50:
            single_mapset_dominated_count += 1
        features = _motif_features(motif)
        rows.append(
            {
                "rank": index + 1,
                "motif_id": motif_id,
                "length_groups": len(motif),
                "train_candidate_support": train_support_by_id[motif_id],
                "compression_gain": gain_by_id[motif_id],
                "occurrence_count": occurrence_count,
                "mapset_count": mapset_count,
                "map_count": len(item["maps"]),
                "top_mapset_fraction": top_mapset_fraction,
                "split_counts": dict(item["split_counts"]),
                "top_density_bins": dict(item["density_bins"].most_common(5)),
                "mean_difficulty": _safe_div(item["difficulty_sum"], occurrence_count),
                "mean_density_events_per_second": _safe_div(item["density_sum"], occurrence_count),
                "mean_chord_ratio": _safe_div(item["chord_sum"], occurrence_count),
                "mean_ln_ratio": _safe_div(item["ln_sum"], occurrence_count),
                "tags": features["tags"],
                "grid": features["grid"],
                "motif": " ".join(motif),
            }
        )
    return {
        "top_motif_limit": top_count,
        "broad_top_motif_count": broad_count,
        "single_mapset_dominated_top_motif_count": single_mapset_dominated_count,
        "top_motifs": rows,
    }


def _cross_split_key_conflicts(chunk_df: pd.DataFrame) -> dict[str, Any]:
    maps = chunk_df[["source_row_index", "artist", "title", "version", "split"]].drop_duplicates("source_row_index")
    artist_title = _conflict_keys(maps, ["artist", "title"])
    artist_title_version = _conflict_keys(maps, ["artist", "title", "version"])
    return {
        "artist_title_conflict_count": len(artist_title),
        "artist_title_version_conflict_count": len(artist_title_version),
        "artist_title_keys": sorted(artist_title)[:100],
        "artist_title_version_keys": sorted(artist_title_version)[:100],
    }


def _filter_conflict_keys(chunk_df: pd.DataFrame, conflict_keys: Sequence[str]) -> pd.DataFrame:
    if not conflict_keys:
        return chunk_df.copy()
    frame = chunk_df.copy()
    frame["_artist_title_version_key"] = _key_series(frame, ["artist", "title", "version"])
    return frame[~frame["_artist_title_version_key"].isin(set(conflict_keys))].drop(columns=["_artist_title_version_key"])


def _conflict_keys(maps: pd.DataFrame, columns: Sequence[str]) -> set[str]:
    keyed = maps[[*columns, "split"]].copy()
    keyed["key"] = _key_series(keyed, columns)
    counts = keyed.drop_duplicates(["key", "split"]).groupby("key", dropna=False)["split"].nunique()
    return {str(key) for key, count in counts.items() if int(count) > 1}


def _key_series(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    result = pd.Series([""] * len(frame), index=frame.index)
    for column in columns:
        result = result + "|" + frame[column].map(_normalize_text)
    return result


def _fallback_concentration(
    chunk_df: pd.DataFrame,
    *,
    variant: _VariantConfig,
    vocab: Any,
) -> dict[str, Any]:
    trie = _build_motif_trie(vocab.motif_to_id)
    by_map: dict[int, dict[str, Any]] = defaultdict(lambda: {"groups": 0, "fallback_groups": 0, "events": 0, "chunks": 0})
    by_bucket: dict[str, dict[str, Any]] = defaultdict(lambda: {"groups": 0, "fallback_groups": 0, "events": 0, "chunks": 0})
    for row in chunk_df.itertuples(index=False):
        tokens = _group_tokens_for_row(row, variant=variant)
        covered = _motif_covered_group_count(tokens, trie)
        fallback_groups = len(tokens) - covered
        map_item = by_map[int(row.source_row_index)]
        bucket_item = by_bucket[str(row.density_bin)]
        for item in (map_item, bucket_item):
            item["groups"] += len(tokens)
            item["fallback_groups"] += fallback_groups
            item["events"] += int(row.num_events)
            item["chunks"] += 1
        map_item.update(
            {
                "source_row_index": int(row.source_row_index),
                "beatmap_set_id": int(row.beatmap_set_id),
                "beatmap_id": int(row.beatmap_id),
                "split": str(row.split),
                "title": str(row.title),
                "artist": str(row.artist),
                "version": str(row.version),
                "difficulty": float(row.difficulty or 0.0),
                "density_bin": str(row.density_bin),
            }
        )
    worst_maps = []
    for item in by_map.values():
        groups = int(item["groups"])
        item["fallback_group_rate"] = float(item["fallback_groups"]) / float(groups) if groups else 0.0
        worst_maps.append(dict(item))
    worst_maps.sort(key=lambda row: (row["fallback_group_rate"], row["groups"]), reverse=True)
    bucket_rows = []
    for bucket, item in by_bucket.items():
        groups = int(item["groups"])
        bucket_rows.append(
            {
                "bucket": bucket,
                "groups": groups,
                "events": int(item["events"]),
                "chunks": int(item["chunks"]),
                "fallback_groups": int(item["fallback_groups"]),
                "fallback_group_rate": float(item["fallback_groups"]) / float(groups) if groups else 0.0,
            }
        )
    bucket_rows.sort(key=lambda row: row["fallback_group_rate"], reverse=True)
    return {"worst_maps": worst_maps[:20], "bucket_summary": bucket_rows}


def _motif_covered_group_count(tokens: Sequence[str], trie: Mapping[str, Any]) -> int:
    covered = 0
    index = 0
    while index < len(tokens):
        matched_token, matched_length = _match_motif(tokens, index, trie)
        if matched_token is not None:
            covered += matched_length
            index += matched_length
        else:
            index += 1
    return covered


def _motif_features(motif: Sequence[str]) -> dict[str, Any]:
    lanes: list[int] = []
    chord_count = 0
    ln_count = 0
    grid: list[str] = []
    for token in motif:
        parts = token.split(":")
        if len(parts) < 5:
            continue
        delta = int(parts[1])
        tap = int(parts[2])
        start = int(parts[3])
        end = int(parts[4])
        active_mask = tap | start | end
        if int(active_mask).bit_count() >= 2:
            chord_count += 1
        if start or end:
            ln_count += 1
        tap_lanes = [lane for lane in range(4) if tap & (1 << lane)]
        if len(tap_lanes) == 1 and not start and not end:
            lanes.append(tap_lanes[0])
        grid.append(f"+{delta}:{_mask_grid(tap, start, end)}")
    tags = []
    if chord_count:
        tags.append("chord")
    if ln_count:
        tags.append("ln")
    if len(lanes) >= 2 and len(set(lanes)) == 1:
        tags.append("jack_like")
    if len(lanes) >= 4 and len(set(lanes)) == 2 and all(lanes[i] == lanes[i % 2] for i in range(len(lanes))):
        tags.append("trill_like")
    if len(lanes) >= 3 and all(abs(lanes[i + 1] - lanes[i]) == 1 for i in range(len(lanes) - 1)):
        tags.append("stair_like")
    if len(lanes) == len(motif) and not tags:
        tags.append("single_tap_motion")
    if not tags:
        tags.append("mixed")
    return {"tags": tags, "grid": " ".join(grid)}


def _mask_grid(tap: int, start: int, end: int) -> str:
    chars = []
    for lane in range(4):
        bit = 1 << lane
        if start & bit:
            chars.append("S")
        elif end & bit:
            chars.append("E")
        elif tap & bit:
            chars.append("T")
        else:
            chars.append(".")
    return "".join(chars)


def _guard_from_refinement(report: Mapping[str, Any]) -> dict[str, Any]:
    pass_criteria = report.get("pass_criteria", {}) if isinstance(report, Mapping) else {}
    return {
        "available": bool(pass_criteria),
        "research_pass": bool(pass_criteria.get("research_pass")),
        "reconstruction_pass": bool(pass_criteria.get("reconstruction_pass")),
        "reconstruction_mismatch_count": int(pass_criteria.get("reconstruction_mismatch_count", 0) or 0),
    }


def _pass_criteria(
    *,
    guard: Mapping[str, Any],
    motif_support: Mapping[str, Any],
    filtered_row: Mapping[str, Any],
    filtered_baselines: Mapping[str, Any],
) -> dict[str, Any]:
    test = filtered_row.get("test", {}) if isinstance(filtered_row, Mapping) else {}
    test_bits = float(test.get("bits_per_event", math.inf)) if isinstance(test, Mapping) else math.inf
    v2_bits = _nested(filtered_baselines, "current_v2_success_only", "test", "bits_per_event")
    if v2_bits is None:
        v2_bits = math.inf
    broad_count = int(motif_support.get("broad_top_motif_count", 0) or 0)
    top_limit = int(motif_support.get("top_motif_limit", 0) or 0)
    broad_rate = float(broad_count) / float(top_limit) if top_limit else 0.0
    support_pass = broad_rate >= 0.50
    compression_pass = test_bits < float(v2_bits)
    guard_pass = bool(guard.get("research_pass")) and bool(guard.get("reconstruction_pass"))
    return {
        "guard_pass": guard_pass,
        "filtered_test_bits_per_event": None if not math.isfinite(test_bits) else test_bits,
        "filtered_v2_test_bits_per_event": None if not math.isfinite(float(v2_bits)) else float(v2_bits),
        "filtered_compression_pass": compression_pass,
        "broad_top_motif_rate": broad_rate,
        "motif_support_pass": support_pass,
        "research_pass": bool(guard_pass and compression_pass and support_pass),
    }


def _recommendation(pass_criteria: Mapping[str, Any]) -> str:
    if not pass_criteria.get("guard_pass"):
        return "KILL_OR_FIX: tokenizer refinement guard did not pass; motif quality is not interpretable."
    if pass_criteria.get("research_pass"):
        return "TEST_CHART_ONLY_EMBEDDINGS: motif quality and leakage sensitivity pass; mapper integration remains out of scope."
    if pass_criteria.get("filtered_compression_pass"):
        return "MUTATE: compression survives filtering, but motif support breadth or quality needs refinement."
    return "KILL_OR_REDESIGN: filtered heldout compression does not beat v2."


def _write_result_log(path: Path, report: Mapping[str, Any]) -> None:
    pass_criteria = report.get("pass_criteria", {})
    filtered = report.get("filtered_compression", {})
    filtered_test = filtered.get("test", {}) if isinstance(filtered, Mapping) else {}
    conflicts = report.get("cross_split_conflicts", {})
    motif_support = report.get("motif_support", {})
    top_motifs = motif_support.get("top_motifs", []) if isinstance(motif_support, Mapping) else []
    lines = [
        "# Result Log",
        "",
        "- Experiment: Beat-Chunk Motif Quality And Leakage Sensitivity Audit",
        "- Date: 2026-06-16",
        f"- Runtime seconds: {report.get('elapsed_s')}",
        f"- Chunk cache: `{report.get('chunk_cache_path')}`",
        f"- Refinement report: `{report.get('refinement_report_path')}`",
        f"- Report: `{report.get('report_path') or DEFAULT_REPORT_PATH.as_posix()}`",
        "",
        "## Guard",
        "",
        f"- Refinement research pass: {_nested(report, 'guard', 'research_pass')}",
        f"- Refinement reconstruction pass: {_nested(report, 'guard', 'reconstruction_pass')}",
        "",
        "## Leakage Sensitivity",
        "",
        f"- Artist/title conflict keys: {conflicts.get('artist_title_conflict_count') if isinstance(conflicts, Mapping) else None}",
        f"- Artist/title/version conflict keys: {conflicts.get('artist_title_version_conflict_count') if isinstance(conflicts, Mapping) else None}",
        f"- Filtered test bits/event: {filtered_test.get('bits_per_event') if isinstance(filtered_test, Mapping) else None}",
        f"- Filtered test bits/event with dictionary: {filtered_test.get('bits_per_event_with_dictionary') if isinstance(filtered_test, Mapping) else None}",
        f"- Filtered v2 success-only test bits/event: {_nested(report, 'filtered_split_matched_baselines', 'current_v2_success_only', 'test', 'bits_per_event')}",
        "",
        "## Motif Support",
        "",
        f"- Top motif limit: {motif_support.get('top_motif_limit') if isinstance(motif_support, Mapping) else None}",
        f"- Broad top motif count: {motif_support.get('broad_top_motif_count') if isinstance(motif_support, Mapping) else None}",
        f"- Single-mapset dominated top motif count: {motif_support.get('single_mapset_dominated_top_motif_count') if isinstance(motif_support, Mapping) else None}",
        "",
        "| Rank | Support | Mapsets | Maps | Top mapset frac | Tags | Grid |",
        "| ---: | ---: | ---: | ---: | ---: | --- | --- |",
        *_top_motif_lines(top_motifs[:20]),
        "",
        "## Decision",
        "",
        f"- Research pass: {pass_criteria.get('research_pass') if isinstance(pass_criteria, Mapping) else None}",
        f"- Filtered compression pass: {pass_criteria.get('filtered_compression_pass') if isinstance(pass_criteria, Mapping) else None}",
        f"- Motif support pass: {pass_criteria.get('motif_support_pass') if isinstance(pass_criteria, Mapping) else None}",
        f"- Recommendation: {report.get('recommendation')}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text("\n".join(lines), encoding="utf-8")
    tmp_path.replace(path)


def _top_motif_lines(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = []
    for row in rows:
        lines.append(
            "| {rank} | {support} | {mapsets} | {maps} | {fraction:.3f} | {tags} | `{grid}` |".format(
                rank=row.get("rank"),
                support=row.get("occurrence_count"),
                mapsets=row.get("mapset_count"),
                maps=row.get("map_count"),
                fraction=float(row.get("top_mapset_fraction", 0.0) or 0.0),
                tags=",".join(row.get("tags", [])),
                grid=row.get("grid", ""),
            )
        )
    return lines


def _safe_div(numerator: float, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(_normalize_json(payload), allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _normalize_json(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_json(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_json(item) for item in value]
    if isinstance(value, set):
        return sorted(_normalize_json(item) for item in value)
    if isinstance(value, Counter):
        return {str(key): int(count) for key, count in value.items()}
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


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


def _format_command(argv: Sequence[str] | None) -> str:
    args = list(sys.argv[1:] if argv is None else argv)
    return " ".join(shlex.quote(part) for part in ["python", "-m", "pulsefield_model.osu_core.beat_chunk_motif_quality_audit", *args])


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit selected beat-chunk motif vocabulary quality.")
    parser.add_argument("--chunk-cache-path", type=Path, default=DEFAULT_BEAT_CHUNK_CACHE_PATH)
    parser.add_argument("--refinement-report-path", type=Path, default=DEFAULT_REFINEMENT_REPORT_PATH)
    parser.add_argument("--beat-structure-cache-path", type=Path, default=DEFAULT_LE3_STRUCTURE_CACHE_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--result-log-path", type=Path, default=DEFAULT_RESULT_LOG_PATH)
    parser.add_argument("--motif-vocab-size", type=int, default=DEFAULT_MOTIF_VOCAB_SIZE)
    parser.add_argument("--motif-min-n", type=int, default=DEFAULT_MOTIF_MIN_N)
    parser.add_argument("--motif-max-n", type=int, default=DEFAULT_MOTIF_MAX_N)
    parser.add_argument("--top-motif-limit", type=int, default=DEFAULT_TOP_MOTIF_LIMIT)
    parser.add_argument("--smoothing-alpha", type=float, default=DEFAULT_SMOOTHING_ALPHA)
    parser.add_argument("--rare-max-count", type=int, default=DEFAULT_RARE_MAX_COUNT)
    parser.add_argument("--limit-chunks", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    report = audit_beat_chunk_motif_quality(
        chunk_cache_path=args.chunk_cache_path,
        refinement_report_path=args.refinement_report_path,
        beat_structure_cache_path=args.beat_structure_cache_path,
        report_path=args.report_path,
        result_log_path=args.result_log_path,
        motif_vocab_size=args.motif_vocab_size,
        motif_min_n=args.motif_min_n,
        motif_max_n=args.motif_max_n,
        top_motif_limit=args.top_motif_limit,
        smoothing_alpha=args.smoothing_alpha,
        rare_max_count=args.rare_max_count,
        limit_chunks=args.limit_chunks,
        command=_format_command(argv),
    )
    print(json.dumps(_normalize_json(report), allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
