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
from typing import Any, Final, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds

from pulsefield_model.osu_core.beat_chunk_pattern_audit import DEFAULT_BEAT_CHUNK_CACHE_PATH
from pulsefield_model.osu_core.beat_chunk_motif_quality_audit import (
    DEFAULT_REPORT_PATH as DEFAULT_MOTIF_QUALITY_REPORT_PATH,
    _motif_features,
)
from pulsefield_model.osu_core.beat_chunk_tokenizer_refinement_audit import (
    DEFAULT_REPORT_PATH as DEFAULT_REFINEMENT_REPORT_PATH,
    _VariantConfig,
    _build_motif_trie,
    _build_motif_vocab,
    _encode_group_sequence,
    _group_tokens_for_row,
    _learn_motif_candidates,
    _read_chunk_cache,
)


SCHEMA_VERSION: Final[int] = 1
DEFAULT_REPORT_PATH: Final[Path] = Path(
    "artifacts/reports/audits/beat_chunk_motif_embeddings/beat_chunk_motif_embedding_audit_le3.json",
)
DEFAULT_RESULT_LOG_PATH: Final[Path] = Path(
    "artifacts/reports/audits/beat_chunk_motif_embeddings/beat_chunk_motif_embedding_result_log.md",
)
DEFAULT_MOTIF_VOCAB_SIZE: Final[int] = 16_384
DEFAULT_EMBEDDING_VOCAB_SIZE: Final[int] = 4_096
DEFAULT_MOTIF_MIN_N: Final[int] = 2
DEFAULT_MOTIF_MAX_N: Final[int] = 8
DEFAULT_CONTEXT_WINDOW: Final[int] = 8
DEFAULT_EMBEDDING_DIM: Final[int] = 32
DEFAULT_NEIGHBOR_K: Final[int] = 10
DEFAULT_EVAL_TOKEN_COUNT: Final[int] = 512
DEFAULT_RANDOM_SEED: Final[int] = 1729
DEFAULT_FIT_SPLIT: Final[str] = "train"
POWERED_TAG_MIN_MOTIFS: Final[int] = 30
POWERED_TAG_MIN_OCCURRENCES: Final[int] = 500


def audit_beat_chunk_motif_embeddings(
    *,
    chunk_cache_path: str | Path = DEFAULT_BEAT_CHUNK_CACHE_PATH,
    refinement_report_path: str | Path = DEFAULT_REFINEMENT_REPORT_PATH,
    motif_quality_report_path: str | Path = DEFAULT_MOTIF_QUALITY_REPORT_PATH,
    report_path: str | Path | None = DEFAULT_REPORT_PATH,
    result_log_path: str | Path | None = DEFAULT_RESULT_LOG_PATH,
    motif_vocab_size: int = DEFAULT_MOTIF_VOCAB_SIZE,
    embedding_vocab_size: int = DEFAULT_EMBEDDING_VOCAB_SIZE,
    motif_min_n: int = DEFAULT_MOTIF_MIN_N,
    motif_max_n: int = DEFAULT_MOTIF_MAX_N,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    neighbor_k: int = DEFAULT_NEIGHBOR_K,
    eval_token_count: int = DEFAULT_EVAL_TOKEN_COUNT,
    random_seed: int = DEFAULT_RANDOM_SEED,
    fit_split: str = DEFAULT_FIT_SPLIT,
    limit_chunks: int | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    """Build chart-only PPMI/SVD embeddings for selected beat-chunk motif tokens."""

    started_at = time.perf_counter()
    chunk_cache_path = Path(chunk_cache_path)
    refinement_report_path = Path(refinement_report_path)
    motif_quality_report_path = Path(motif_quality_report_path)
    report_path = None if report_path is None else Path(report_path)
    result_log_path = None if result_log_path is None else Path(result_log_path)
    motif_vocab_size = _positive_int(motif_vocab_size, "motif_vocab_size")
    embedding_vocab_size = _positive_int(embedding_vocab_size, "embedding_vocab_size")
    motif_min_n = _positive_int(motif_min_n, "motif_min_n")
    motif_max_n = _positive_int(motif_max_n, "motif_max_n")
    context_window = _positive_int(context_window, "context_window")
    embedding_dim = _positive_int(embedding_dim, "embedding_dim")
    neighbor_k = _positive_int(neighbor_k, "neighbor_k")
    eval_token_count = _positive_int(eval_token_count, "eval_token_count")
    if motif_max_n < motif_min_n:
        raise ValueError("motif_max_n must be >= motif_min_n")
    if fit_split != DEFAULT_FIT_SPLIT:
        raise ValueError("chart-only embedding diagnostics must fit PPMI/SVD on the train split only")

    chunk_df = _read_chunk_cache(chunk_cache_path, limit_chunks=limit_chunks)
    fit_df = chunk_df[chunk_df["split"].fillna("").astype(str) == fit_split].copy()
    refinement_report = _read_json(refinement_report_path)
    motif_quality_report = _read_json(motif_quality_report_path)
    guard = _guard(refinement_report, motif_quality_report)
    variant = _VariantConfig("motif_delta_raw", "motif", "delta", "raw", False, True)
    candidates = _learn_motif_candidates(
        chunk_df,
        variant=variant,
        motif_min_n=motif_min_n,
        motif_max_n=motif_max_n,
    )
    motif_vocab_size = min(motif_vocab_size, len(candidates))
    embedding_vocab_size = min(embedding_vocab_size, motif_vocab_size)
    vocab = _build_motif_vocab(candidates[:motif_vocab_size])
    embedded_ids = [f"M{index}" for index in range(embedding_vocab_size)]
    token_to_index = {token_id: index for index, token_id in enumerate(embedded_ids)}
    id_to_motif = {f"M{index}": motif for index, (motif, _count, _gain) in enumerate(candidates[:embedding_vocab_size])}
    metadata = _motif_metadata(fit_df, variant=variant, vocab=vocab, token_to_index=token_to_index, id_to_motif=id_to_motif)
    class_support = _class_support_preflight(metadata)
    cooc = _cooccurrence_matrix(
        fit_df,
        variant=variant,
        vocab=vocab,
        token_to_index=token_to_index,
        context_window=context_window,
    )
    ppmi = _ppmi_matrix(cooc)
    embedding, svd_stats = _svd_embedding(ppmi, embedding_dim=embedding_dim)
    matrix_health = _matrix_health(cooc, ppmi, embedding, metadata, svd_stats)
    diagnostics = _neighbor_diagnostics(
        embedding,
        metadata=metadata,
        neighbor_k=neighbor_k,
        eval_token_count=eval_token_count,
        random_seed=random_seed,
    )
    pass_criteria = _pass_criteria(
        guard=guard,
        cooc=cooc,
        ppmi=ppmi,
        svd_stats=svd_stats,
        matrix_health=matrix_health,
        diagnostics=diagnostics,
        class_support=class_support,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "chunk_cache_path": chunk_cache_path.as_posix(),
        "refinement_report_path": refinement_report_path.as_posix(),
        "motif_quality_report_path": motif_quality_report_path.as_posix(),
        "report_path": None if report_path is None else report_path.as_posix(),
        "result_log_path": None if result_log_path is None else result_log_path.as_posix(),
        "limited": limit_chunks is not None,
        "limit_chunks": limit_chunks,
        "config": {
            "variant": variant.name,
            "motif_vocab_size": motif_vocab_size,
            "embedding_vocab_size": embedding_vocab_size,
            "motif_min_n": motif_min_n,
            "motif_max_n": motif_max_n,
            "context_window": context_window,
            "embedding_dim": embedding_dim,
            "neighbor_k": neighbor_k,
            "eval_token_count": eval_token_count,
            "random_seed": random_seed,
            "fit_split": fit_split,
            "embedding_method": "motif co-occurrence PPMI plus sparse SVD",
        },
        "guard": guard,
        "class_support": class_support,
        "cooccurrence_summary": {
            "shape": list(cooc.shape),
            "nnz": int(cooc.nnz),
            "density": float(cooc.nnz) / float(cooc.shape[0] * cooc.shape[1]) if cooc.shape[0] and cooc.shape[1] else 0.0,
            "total_weight": float(cooc.sum()),
            "ppmi_nnz": int(ppmi.nnz),
        },
        "matrix_health": matrix_health,
        "svd": svd_stats,
        "neighbor_diagnostics": diagnostics,
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


def _motif_metadata(
    chunk_df: pd.DataFrame,
    *,
    variant: _VariantConfig,
    vocab: Any,
    token_to_index: Mapping[str, int],
    id_to_motif: Mapping[str, tuple[str, ...]],
) -> list[dict[str, Any]]:
    trie = _build_motif_trie(vocab.motif_to_id)
    metadata = [
        {
            "token_id": token_id,
            "rank": int(index) + 1,
            "motif": " ".join(id_to_motif[token_id]),
            "features": _embedding_labels(id_to_motif[token_id]),
            "quality_features": _motif_features(id_to_motif[token_id]),
            "occurrence_count": 0,
            "mapsets": set(),
            "maps": set(),
            "density_bins": Counter(),
            "top_mapsets": Counter(),
        }
        for token_id, index in sorted(token_to_index.items(), key=lambda item: item[1])
    ]
    for row in chunk_df.itertuples(index=False):
        encoded = _encoded_selected_tokens(row, variant=variant, trie=trie, token_to_index=token_to_index)
        for token_id in encoded:
            item = metadata[token_to_index[token_id]]
            item["occurrence_count"] += 1
            item["mapsets"].add(int(row.beatmap_set_id))
            item["maps"].add(int(row.source_row_index))
            item["density_bins"][str(row.density_bin)] += 1
            item["top_mapsets"][int(row.beatmap_set_id)] += 1
    for item in metadata:
        occurrence_count = int(item["occurrence_count"])
        top_mapset_count = item["top_mapsets"].most_common(1)[0][1] if item["top_mapsets"] else 0
        item["mapset_count"] = len(item["mapsets"])
        item["map_count"] = len(item["maps"])
        item["top_mapset"] = item["top_mapsets"].most_common(1)[0][0] if item["top_mapsets"] else None
        item["top_mapset_fraction"] = float(top_mapset_count) / float(occurrence_count) if occurrence_count else 0.0
        item["dominant_density_bin"] = item["density_bins"].most_common(1)[0][0] if item["density_bins"] else ""
        item["mapsets"] = None
        item["maps"] = None
        item["density_bins"] = dict(item["density_bins"].most_common(5))
        item["top_mapsets"] = dict(item["top_mapsets"].most_common(5))
    return metadata


def _cooccurrence_matrix(
    chunk_df: pd.DataFrame,
    *,
    variant: _VariantConfig,
    vocab: Any,
    token_to_index: Mapping[str, int],
    context_window: int,
) -> sparse.csr_matrix:
    pair_counter: Counter[tuple[int, int]] = Counter()
    trie = _build_motif_trie(vocab.motif_to_id)
    for _source_row_index, frame in chunk_df.groupby("source_row_index", sort=False):
        sequence: list[int] = []
        for row in frame.sort_values("chunk_index").itertuples(index=False):
            sequence.extend(token_to_index[token_id] for token_id in _encoded_selected_tokens(row, variant=variant, trie=trie, token_to_index=token_to_index))
        for index, center in enumerate(sequence):
            start = max(0, index - context_window)
            end = min(len(sequence), index + context_window + 1)
            for context_index in range(start, end):
                if context_index == index:
                    continue
                pair_counter[(center, sequence[context_index])] += 1
    if not pair_counter:
        return sparse.csr_matrix((len(token_to_index), len(token_to_index)), dtype=np.float64)
    rows = np.fromiter((item[0] for item in pair_counter), dtype=np.int32)
    cols = np.fromiter((item[1] for item in pair_counter), dtype=np.int32)
    data = np.fromiter(pair_counter.values(), dtype=np.float64)
    return sparse.coo_matrix((data, (rows, cols)), shape=(len(token_to_index), len(token_to_index))).tocsr()


def _class_support_preflight(metadata: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_tag: dict[str, dict[str, int]] = defaultdict(lambda: {"motif_count": 0, "occurrence_count": 0})
    for item in metadata:
        occurrence_count = int(item.get("occurrence_count", 0) or 0)
        tags = item.get("quality_features", {}).get("tags", [])
        for tag in tags:
            by_tag[str(tag)]["motif_count"] += 1
            by_tag[str(tag)]["occurrence_count"] += occurrence_count
    rows = []
    powered_count = 0
    for tag, counts in sorted(by_tag.items()):
        powered = (
            int(counts["motif_count"]) >= POWERED_TAG_MIN_MOTIFS
            and int(counts["occurrence_count"]) >= POWERED_TAG_MIN_OCCURRENCES
        )
        if powered:
            powered_count += 1
        rows.append({"tag": tag, **counts, "powered": powered})
    ln_row = next((row for row in rows if row["tag"] == "ln"), None)
    ln_powered = bool(ln_row and ln_row["powered"])
    return {
        "powered_tag_family_count": powered_count,
        "tag_rows": rows,
        "ln_powered": ln_powered,
        "ln_status": "POWERED" if ln_powered else "LN_UNDERPOWERED",
        "min_motifs": POWERED_TAG_MIN_MOTIFS,
        "min_occurrences": POWERED_TAG_MIN_OCCURRENCES,
    }


def _encoded_selected_tokens(row: Any, *, variant: _VariantConfig, trie: Mapping[str, Any], token_to_index: Mapping[str, int]) -> list[str]:
    tokens = _group_tokens_for_row(row, variant=variant)
    encoded = _encode_group_sequence(tokens, trie)
    return [token for token in encoded if token in token_to_index]


def _ppmi_matrix(cooc: sparse.csr_matrix) -> sparse.csr_matrix:
    if cooc.nnz == 0:
        return cooc.copy()
    coo = cooc.tocoo()
    total = float(coo.data.sum())
    row_sums = np.asarray(cooc.sum(axis=1)).ravel()
    col_sums = np.asarray(cooc.sum(axis=0)).ravel()
    denom = row_sums[coo.row] * col_sums[coo.col]
    valid = denom > 0
    values = np.zeros_like(coo.data, dtype=np.float64)
    values[valid] = np.log2((coo.data[valid] * total) / denom[valid])
    positive = values > 0
    return sparse.coo_matrix((values[positive], (coo.row[positive], coo.col[positive])), shape=cooc.shape).tocsr()


def _svd_embedding(ppmi: sparse.csr_matrix, *, embedding_dim: int) -> tuple[np.ndarray, dict[str, Any]]:
    if ppmi.nnz == 0 or min(ppmi.shape) <= 1:
        return np.zeros((ppmi.shape[0], 0), dtype=np.float32), {"available": False, "reason": "empty_ppmi"}
    k = min(embedding_dim, min(ppmi.shape) - 1)
    if k <= 0:
        return np.zeros((ppmi.shape[0], 0), dtype=np.float32), {"available": False, "reason": "dimension_too_small"}
    u, singular_values, _vt = svds(ppmi, k=k)
    order = np.argsort(singular_values)[::-1]
    singular_values = singular_values[order]
    u = u[:, order]
    embedding = u * np.sqrt(singular_values)
    finite = bool(np.isfinite(embedding).all())
    return embedding.astype(np.float32), {
        "available": finite,
        "embedding_dim": int(k),
        "singular_values": [float(value) for value in singular_values[:20]],
        "singular_value_sum": float(singular_values.sum()),
        "top_singular_value_share": float(singular_values[0] / singular_values.sum()) if singular_values.sum() else 0.0,
        "finite": finite,
    }


def _matrix_health(
    cooc: sparse.csr_matrix,
    ppmi: sparse.csr_matrix,
    embedding: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    svd_stats: Mapping[str, Any],
) -> dict[str, Any]:
    row_nnz = np.diff(ppmi.indptr) if ppmi.shape[0] else np.array([], dtype=np.int64)
    zero_rows = int((row_nnz == 0).sum()) if row_nnz.size else 0
    row_norms = np.linalg.norm(embedding, axis=1) if embedding.size else np.array([], dtype=np.float32)
    frequencies = np.array([float(item.get("occurrence_count", 0) or 0) for item in metadata], dtype=np.float64)
    if row_norms.size and frequencies.size and np.std(row_norms) > 0 and np.std(np.log1p(frequencies)) > 0:
        row_norm_frequency_correlation = float(np.corrcoef(row_norms, np.log1p(frequencies))[0, 1])
    else:
        row_norm_frequency_correlation = 0.0
    return {
        "nonzero_row_count": int((row_nnz > 0).sum()) if row_nnz.size else 0,
        "zero_row_count": zero_rows,
        "zero_row_rate": float(zero_rows) / float(ppmi.shape[0]) if ppmi.shape[0] else 1.0,
        "ppmi_positive_entry_rate": float(ppmi.nnz) / float(cooc.nnz) if cooc.nnz else 0.0,
        "top_singular_value_share": float(svd_stats.get("top_singular_value_share", 0.0) or 0.0),
        "row_norm": _distribution(row_norms),
        "row_norm_frequency_correlation": row_norm_frequency_correlation,
    }


def _neighbor_diagnostics(
    embedding: np.ndarray,
    *,
    metadata: Sequence[Mapping[str, Any]],
    neighbor_k: int,
    eval_token_count: int,
    random_seed: int,
) -> dict[str, Any]:
    if embedding.size == 0 or embedding.shape[0] == 0:
        return {"available": False, "reason": "empty_embedding"}
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    normalized = embedding / np.maximum(norms, 1e-12)
    sim = normalized @ normalized.T
    np.fill_diagonal(sim, -np.inf)
    eval_count = min(eval_token_count, embedding.shape[0])
    neighbor_k = min(neighbor_k, max(1, embedding.shape[0] - 1))
    rng = np.random.default_rng(random_seed)
    metrics = {
        "tag": [],
        "action": [],
        "rhythm": [],
        "motion": [],
        "density": [],
        "top_mapset": [],
        "cosine": [],
        "random_tag": [],
        "random_action": [],
        "random_rhythm": [],
        "random_motion": [],
        "random_density": [],
        "random_top_mapset": [],
        "density_matched_tag": [],
        "density_matched_action": [],
        "density_matched_rhythm": [],
        "density_matched_motion": [],
        "frequency_matched_tag": [],
        "frequency_matched_action": [],
        "frequency_matched_rhythm": [],
        "frequency_matched_motion": [],
    }
    density_index = _density_index(metadata)
    frequency_order = sorted(range(len(metadata)), key=lambda idx: float(metadata[idx].get("occurrence_count", 0) or 0))
    rank_by_index = {index: rank for rank, index in enumerate(frequency_order)}
    examples = []
    for index in range(eval_count):
        neighbors = np.argpartition(-sim[index], kth=neighbor_k - 1)[:neighbor_k]
        neighbors = neighbors[np.argsort(-sim[index, neighbors])]
        random_neighbors = _random_neighbors(rng, size=embedding.shape[0], exclude=index, k=neighbor_k)
        density_neighbors = _density_matched_neighbors(rng, metadata, density_index, exclude=index, k=neighbor_k)
        frequency_neighbors = _frequency_matched_neighbors(
            rng,
            frequency_order=frequency_order,
            rank_by_index=rank_by_index,
            exclude=index,
            k=neighbor_k,
        )
        item = metadata[index]
        metrics["tag"].append(_neighbor_tag_agreement(item, [metadata[int(value)] for value in neighbors]))
        metrics["action"].append(_label_agreement(item, [metadata[int(value)] for value in neighbors], "action_label"))
        metrics["rhythm"].append(_label_agreement(item, [metadata[int(value)] for value in neighbors], "rhythm_label"))
        metrics["motion"].append(_label_agreement(item, [metadata[int(value)] for value in neighbors], "motion_label"))
        metrics["density"].append(_density_agreement(item, [metadata[int(value)] for value in neighbors]))
        metrics["top_mapset"].append(_top_mapset_agreement(item, [metadata[int(value)] for value in neighbors]))
        metrics["cosine"].append(float(np.mean(sim[index, neighbors])))
        metrics["random_tag"].append(_neighbor_tag_agreement(item, [metadata[int(value)] for value in random_neighbors]))
        metrics["random_action"].append(_label_agreement(item, [metadata[int(value)] for value in random_neighbors], "action_label"))
        metrics["random_rhythm"].append(_label_agreement(item, [metadata[int(value)] for value in random_neighbors], "rhythm_label"))
        metrics["random_motion"].append(_label_agreement(item, [metadata[int(value)] for value in random_neighbors], "motion_label"))
        metrics["random_density"].append(_density_agreement(item, [metadata[int(value)] for value in random_neighbors]))
        metrics["random_top_mapset"].append(_top_mapset_agreement(item, [metadata[int(value)] for value in random_neighbors]))
        metrics["density_matched_tag"].append(_neighbor_tag_agreement(item, [metadata[int(value)] for value in density_neighbors]))
        metrics["density_matched_action"].append(_label_agreement(item, [metadata[int(value)] for value in density_neighbors], "action_label"))
        metrics["density_matched_rhythm"].append(_label_agreement(item, [metadata[int(value)] for value in density_neighbors], "rhythm_label"))
        metrics["density_matched_motion"].append(_label_agreement(item, [metadata[int(value)] for value in density_neighbors], "motion_label"))
        metrics["frequency_matched_tag"].append(_neighbor_tag_agreement(item, [metadata[int(value)] for value in frequency_neighbors]))
        metrics["frequency_matched_action"].append(_label_agreement(item, [metadata[int(value)] for value in frequency_neighbors], "action_label"))
        metrics["frequency_matched_rhythm"].append(_label_agreement(item, [metadata[int(value)] for value in frequency_neighbors], "rhythm_label"))
        metrics["frequency_matched_motion"].append(_label_agreement(item, [metadata[int(value)] for value in frequency_neighbors], "motion_label"))
        if index < 12 or index in {128, 256, 511}:
            examples.append(_neighbor_example(index, neighbors, sim, metadata))
    summary = {key: _mean(values) for key, values in metrics.items()}
    summary["tag_lift"] = summary["tag"] - summary["random_tag"]
    summary["action_lift"] = summary["action"] - summary["random_action"]
    summary["rhythm_lift"] = summary["rhythm"] - summary["random_rhythm"]
    summary["motion_lift"] = summary["motion"] - summary["random_motion"]
    summary["tag_lift_vs_density_matched"] = summary["tag"] - summary["density_matched_tag"]
    summary["action_lift_vs_density_matched"] = summary["action"] - summary["density_matched_action"]
    summary["rhythm_lift_vs_density_matched"] = summary["rhythm"] - summary["density_matched_rhythm"]
    summary["motion_lift_vs_density_matched"] = summary["motion"] - summary["density_matched_motion"]
    summary["tag_lift_vs_frequency_matched"] = summary["tag"] - summary["frequency_matched_tag"]
    summary["action_lift_vs_frequency_matched"] = summary["action"] - summary["frequency_matched_action"]
    summary["rhythm_lift_vs_frequency_matched"] = summary["rhythm"] - summary["frequency_matched_rhythm"]
    summary["motion_lift_vs_frequency_matched"] = summary["motion"] - summary["frequency_matched_motion"]
    summary["density_lift"] = summary["density"] - summary["random_density"]
    summary["top_mapset_lift"] = summary["top_mapset"] - summary["random_top_mapset"]
    summary["composite_structural_lift"] = float(
        np.mean([summary["tag_lift"], summary["action_lift"], summary["rhythm_lift"], summary["motion_lift"]])
    )
    summary["composite_lift_vs_density_matched"] = float(
        np.mean(
            [
                summary["tag_lift_vs_density_matched"],
                summary["action_lift_vs_density_matched"],
                summary["rhythm_lift_vs_density_matched"],
                summary["motion_lift_vs_density_matched"],
            ]
        )
    )
    summary["composite_lift_vs_frequency_matched"] = float(
        np.mean(
            [
                summary["tag_lift_vs_frequency_matched"],
                summary["action_lift_vs_frequency_matched"],
                summary["rhythm_lift_vs_frequency_matched"],
                summary["motion_lift_vs_frequency_matched"],
            ]
        )
    )
    summary["conservative_control_lift"] = min(
        summary["composite_structural_lift"],
        summary["composite_lift_vs_density_matched"],
        summary["composite_lift_vs_frequency_matched"],
    )
    return {"available": True, "eval_token_count": eval_count, "neighbor_k": neighbor_k, "summary": summary, "examples": examples}


def _neighbor_example(index: int, neighbors: np.ndarray, sim: np.ndarray, metadata: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    item = metadata[index]
    return {
        "token_id": item["token_id"],
        "rank": item["rank"],
        "motif": item["motif"],
        "tags": item["quality_features"]["tags"],
        "labels": item["features"],
        "grid": item["quality_features"]["grid"],
        "neighbors": [
            {
                "token_id": metadata[int(value)]["token_id"],
                "rank": metadata[int(value)]["rank"],
                "cosine": float(sim[index, int(value)]),
                "tags": metadata[int(value)]["quality_features"]["tags"],
                "labels": metadata[int(value)]["features"],
                "grid": metadata[int(value)]["quality_features"]["grid"],
            }
            for value in neighbors[:10]
        ],
    }


def _embedding_labels(motif: Sequence[str]) -> dict[str, Any]:
    deltas: list[str] = []
    lanes: list[int] = []
    chord = False
    ln = False
    for token in motif:
        parts = token.split(":")
        if len(parts) < 5:
            continue
        deltas.append(parts[1])
        tap = int(parts[2])
        start = int(parts[3])
        end = int(parts[4])
        active = tap | start | end
        chord = chord or active.bit_count() >= 2
        ln = ln or bool(start or end)
        tap_lanes = [lane for lane in range(4) if tap & (1 << lane)]
        if len(tap_lanes) == 1 and not start and not end:
            lanes.append(tap_lanes[0])
    if ln:
        action = "ln"
    elif chord:
        action = "chord"
    else:
        action = "tap"
    if len(lanes) >= 2:
        motion = ",".join(str(lanes[index + 1] - lanes[index]) for index in range(len(lanes) - 1))
    elif len(lanes) == 1:
        motion = "single"
    else:
        motion = "mixed"
    return {
        "action_label": action,
        "rhythm_label": ",".join(deltas),
        "motion_label": motion,
    }


def _neighbor_tag_agreement(center: Mapping[str, Any], neighbors: Sequence[Mapping[str, Any]]) -> float:
    center_tags = set(center["quality_features"]["tags"])
    if not neighbors:
        return 0.0
    return float(sum(bool(center_tags.intersection(neighbor["quality_features"]["tags"])) for neighbor in neighbors)) / float(len(neighbors))


def _label_agreement(center: Mapping[str, Any], neighbors: Sequence[Mapping[str, Any]], label: str) -> float:
    if not neighbors:
        return 0.0
    center_value = center["features"][label]
    return float(sum(neighbor["features"][label] == center_value for neighbor in neighbors)) / float(len(neighbors))


def _density_agreement(center: Mapping[str, Any], neighbors: Sequence[Mapping[str, Any]]) -> float:
    if not neighbors:
        return 0.0
    return float(sum(neighbor["dominant_density_bin"] == center["dominant_density_bin"] for neighbor in neighbors)) / float(len(neighbors))


def _top_mapset_agreement(center: Mapping[str, Any], neighbors: Sequence[Mapping[str, Any]]) -> float:
    if not neighbors or center.get("top_mapset") is None:
        return 0.0
    return float(sum(neighbor.get("top_mapset") == center.get("top_mapset") for neighbor in neighbors)) / float(len(neighbors))


def _random_neighbors(rng: np.random.Generator, *, size: int, exclude: int, k: int) -> np.ndarray:
    if size <= 1:
        return np.array([], dtype=np.int64)
    choices = np.arange(size)
    choices = choices[choices != exclude]
    replace = len(choices) < k
    return rng.choice(choices, size=k, replace=replace)


def _density_index(metadata: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    values: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(metadata):
        values[str(item.get("dominant_density_bin", ""))].append(index)
    return values


def _density_matched_neighbors(
    rng: np.random.Generator,
    metadata: Sequence[Mapping[str, Any]],
    density_index: Mapping[str, Sequence[int]],
    *,
    exclude: int,
    k: int,
) -> np.ndarray:
    density = str(metadata[exclude].get("dominant_density_bin", ""))
    choices = np.array([index for index in density_index.get(density, []) if index != exclude], dtype=np.int64)
    if choices.size == 0:
        return _random_neighbors(rng, size=len(metadata), exclude=exclude, k=k)
    return rng.choice(choices, size=k, replace=choices.size < k)


def _frequency_matched_neighbors(
    rng: np.random.Generator,
    *,
    frequency_order: Sequence[int],
    rank_by_index: Mapping[int, int],
    exclude: int,
    k: int,
) -> np.ndarray:
    if len(frequency_order) <= 1:
        return np.array([], dtype=np.int64)
    rank = int(rank_by_index[exclude])
    radius = max(25, k * 10)
    start = max(0, rank - radius)
    end = min(len(frequency_order), rank + radius + 1)
    choices = np.array([index for index in frequency_order[start:end] if index != exclude], dtype=np.int64)
    if choices.size < k:
        choices = np.array([index for index in frequency_order if index != exclude], dtype=np.int64)
    return rng.choice(choices, size=k, replace=choices.size < k)


def _guard(refinement_report: Mapping[str, Any], motif_quality_report: Mapping[str, Any]) -> dict[str, Any]:
    refinement = refinement_report.get("pass_criteria", {}) if isinstance(refinement_report, Mapping) else {}
    quality = motif_quality_report.get("pass_criteria", {}) if isinstance(motif_quality_report, Mapping) else {}
    return {
        "refinement_research_pass": bool(refinement.get("research_pass")),
        "refinement_reconstruction_pass": bool(refinement.get("reconstruction_pass")),
        "motif_quality_research_pass": bool(quality.get("research_pass")),
        "pass": bool(
            refinement.get("research_pass")
            and refinement.get("reconstruction_pass")
            and quality.get("research_pass")
        ),
    }


def _pass_criteria(
    *,
    guard: Mapping[str, Any],
    cooc: sparse.csr_matrix,
    ppmi: sparse.csr_matrix,
    svd_stats: Mapping[str, Any],
    matrix_health: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    class_support: Mapping[str, Any],
) -> dict[str, Any]:
    summary = diagnostics.get("summary", {}) if isinstance(diagnostics, Mapping) else {}
    composite_lift = float(summary.get("composite_structural_lift", 0.0) or 0.0)
    tag_lift = float(summary.get("tag_lift", 0.0) or 0.0)
    density_lift = float(summary.get("density_lift", 0.0) or 0.0)
    conservative_lift = float(summary.get("conservative_control_lift", 0.0) or 0.0)
    top_mapset = float(summary.get("top_mapset", 1.0) or 0.0)
    graph_pass = cooc.nnz > 0 and ppmi.nnz > 0 and bool(svd_stats.get("available"))
    powered_tag_family_count = int(class_support.get("powered_tag_family_count", 0) or 0)
    class_support_pass = powered_tag_family_count >= 2
    matrix_health_pass = (
        float(matrix_health.get("zero_row_rate", 1.0) or 1.0) <= 0.10
        and float(matrix_health.get("top_singular_value_share", 1.0) or 1.0) <= 0.60
        and abs(float(matrix_health.get("row_norm_frequency_correlation", 1.0) or 0.0)) <= 0.75
    )
    structural_pass = conservative_lift >= 0.05 and tag_lift > 0.02
    density_confound_pass = conservative_lift >= max(0.05, 0.25 * max(0.0, density_lift))
    mapset_pass = top_mapset <= 0.10
    return {
        "guard_pass": bool(guard.get("pass")),
        "graph_pass": graph_pass,
        "class_support_pass": class_support_pass,
        "powered_tag_family_count": powered_tag_family_count,
        "matrix_health_pass": matrix_health_pass,
        "structural_lift_pass": structural_pass,
        "density_confound_pass": density_confound_pass,
        "mapset_concentration_pass": mapset_pass,
        "composite_structural_lift": composite_lift,
        "conservative_control_lift": conservative_lift,
        "tag_lift": tag_lift,
        "density_lift": density_lift,
        "top_mapset_neighbor_agreement": top_mapset,
        "research_pass": bool(
            guard.get("pass")
            and graph_pass
            and class_support_pass
            and matrix_health_pass
            and structural_pass
            and density_confound_pass
            and mapset_pass
        ),
    }


def _recommendation(pass_criteria: Mapping[str, Any]) -> str:
    if not pass_criteria.get("guard_pass"):
        return "KILL_OR_FIX: upstream tokenizer/motif quality guards failed."
    if pass_criteria.get("research_pass"):
        return "TEST_RICHER_CHART_ONLY_EMBEDDINGS: chart-only embedding diagnostics passed; mapper remains a human-owner decision."
    if pass_criteria.get("graph_pass"):
        return "MUTATE: embedding graph built, but structure is weak or density/mapset confounded."
    return "KILL_OR_REDESIGN: chart-only motif co-occurrence embedding could not be built."


def _write_result_log(path: Path, report: Mapping[str, Any]) -> None:
    summary = _nested(report, "neighbor_diagnostics", "summary") or {}
    pass_criteria = report.get("pass_criteria", {})
    class_support = report.get("class_support", {})
    examples = _nested(report, "neighbor_diagnostics", "examples") or []
    lines = [
        "# Result Log",
        "",
        "- Experiment: Beat-Chunk Motif Chart-Only Embedding Diagnostics",
        "- Date: 2026-06-16",
        f"- Runtime seconds: {report.get('elapsed_s')}",
        f"- Chunk cache: `{report.get('chunk_cache_path')}`",
        f"- Report: `{report.get('report_path') or DEFAULT_REPORT_PATH.as_posix()}`",
        "",
        "## Guard",
        "",
        f"- Upstream guard pass: {_nested(report, 'guard', 'pass')}",
        f"- Refinement pass: {_nested(report, 'guard', 'refinement_research_pass')}",
        f"- Motif quality pass: {_nested(report, 'guard', 'motif_quality_research_pass')}",
        "",
        "## Embedding",
        "",
        f"- Fit split: {_nested(report, 'config', 'fit_split')}",
        f"- Co-occurrence nnz: {_nested(report, 'cooccurrence_summary', 'nnz')}",
        f"- PPMI nnz: {_nested(report, 'cooccurrence_summary', 'ppmi_nnz')}",
        f"- Embedding dim: {_nested(report, 'svd', 'embedding_dim')}",
        f"- SVD finite: {_nested(report, 'svd', 'finite')}",
        f"- Zero-row rate: {_nested(report, 'matrix_health', 'zero_row_rate')}",
        f"- Top singular value share: {_nested(report, 'matrix_health', 'top_singular_value_share')}",
        f"- Row-norm/frequency correlation: {_nested(report, 'matrix_health', 'row_norm_frequency_correlation')}",
        "",
        "## Class Support",
        "",
        f"- Powered tag families: {class_support.get('powered_tag_family_count') if isinstance(class_support, Mapping) else None}",
        f"- LN status: {class_support.get('ln_status') if isinstance(class_support, Mapping) else None}",
        *_class_support_lines(class_support.get("tag_rows", []) if isinstance(class_support, Mapping) else []),
        "",
        "## Neighbor Metrics",
        "",
        f"- Composite structural lift: {summary.get('composite_structural_lift') if isinstance(summary, Mapping) else None}",
        f"- Conservative control lift: {summary.get('conservative_control_lift') if isinstance(summary, Mapping) else None}",
        f"- Tag agreement / random: {summary.get('tag') if isinstance(summary, Mapping) else None} / {summary.get('random_tag') if isinstance(summary, Mapping) else None}",
        f"- Tag agreement / density-matched / frequency-matched: {summary.get('tag') if isinstance(summary, Mapping) else None} / {summary.get('density_matched_tag') if isinstance(summary, Mapping) else None} / {summary.get('frequency_matched_tag') if isinstance(summary, Mapping) else None}",
        f"- Action agreement / random: {summary.get('action') if isinstance(summary, Mapping) else None} / {summary.get('random_action') if isinstance(summary, Mapping) else None}",
        f"- Rhythm agreement / random: {summary.get('rhythm') if isinstance(summary, Mapping) else None} / {summary.get('random_rhythm') if isinstance(summary, Mapping) else None}",
        f"- Motion agreement / random: {summary.get('motion') if isinstance(summary, Mapping) else None} / {summary.get('random_motion') if isinstance(summary, Mapping) else None}",
        f"- Density agreement / random: {summary.get('density') if isinstance(summary, Mapping) else None} / {summary.get('random_density') if isinstance(summary, Mapping) else None}",
        f"- Top-mapset neighbor agreement: {summary.get('top_mapset') if isinstance(summary, Mapping) else None}",
        "",
        "## Examples",
        "",
        *_example_lines(examples[:8]),
        "",
        "## Decision",
        "",
        f"- Research pass: {pass_criteria.get('research_pass') if isinstance(pass_criteria, Mapping) else None}",
        f"- Class support pass: {pass_criteria.get('class_support_pass') if isinstance(pass_criteria, Mapping) else None}",
        f"- Matrix health pass: {pass_criteria.get('matrix_health_pass') if isinstance(pass_criteria, Mapping) else None}",
        f"- Structural lift pass: {pass_criteria.get('structural_lift_pass') if isinstance(pass_criteria, Mapping) else None}",
        f"- Density confound pass: {pass_criteria.get('density_confound_pass') if isinstance(pass_criteria, Mapping) else None}",
        f"- Mapset concentration pass: {pass_criteria.get('mapset_concentration_pass') if isinstance(pass_criteria, Mapping) else None}",
        f"- Recommendation: {report.get('recommendation')}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text("\n".join(lines), encoding="utf-8")
    tmp_path.replace(path)


def _class_support_lines(rows: object) -> list[str]:
    if not isinstance(rows, list):
        return []
    lines = ["", "| Tag | Motifs | Occurrences | Powered |", "| --- | ---: | ---: | --- |"]
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| {tag} | {motifs} | {occurrences} | {powered} |".format(
                tag=row.get("tag"),
                motifs=row.get("motif_count"),
                occurrences=row.get("occurrence_count"),
                powered=row.get("powered"),
            )
        )
    return lines


def _example_lines(examples: Sequence[Mapping[str, Any]]) -> list[str]:
    lines: list[str] = []
    for example in examples:
        lines.append(f"- `{example.get('token_id')}` rank {example.get('rank')}: `{example.get('grid')}`")
        neighbors = example.get("neighbors", [])
        if isinstance(neighbors, list):
            compact = [
                f"{item.get('token_id')}:{float(item.get('cosine', 0.0)):.3f}:`{item.get('grid')}`"
                for item in neighbors[:5]
                if isinstance(item, Mapping)
            ]
            lines.append(f"  nearest: {', '.join(compact)}")
    return lines


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _distribution(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"min": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(np.min(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


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
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field} must be an integer, got bool")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be an integer, got {type(value).__name__}") from exc
    if integer <= 0:
        raise ValueError(f"{field} must be positive, got {integer}")
    return integer


def _git_stdout(*args: str) -> str | None:
    try:
        completed = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _format_command(argv: Sequence[str] | None) -> str:
    args = list(sys.argv[1:] if argv is None else argv)
    return " ".join(shlex.quote(part) for part in ["python", "-m", "pulsefield_model.osu_core.beat_chunk_motif_embedding_audit", *args])


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit chart-only embeddings for beat-chunk motif tokens.")
    parser.add_argument("--chunk-cache-path", type=Path, default=DEFAULT_BEAT_CHUNK_CACHE_PATH)
    parser.add_argument("--refinement-report-path", type=Path, default=DEFAULT_REFINEMENT_REPORT_PATH)
    parser.add_argument("--motif-quality-report-path", type=Path, default=DEFAULT_MOTIF_QUALITY_REPORT_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--result-log-path", type=Path, default=DEFAULT_RESULT_LOG_PATH)
    parser.add_argument("--motif-vocab-size", type=int, default=DEFAULT_MOTIF_VOCAB_SIZE)
    parser.add_argument("--embedding-vocab-size", type=int, default=DEFAULT_EMBEDDING_VOCAB_SIZE)
    parser.add_argument("--motif-min-n", type=int, default=DEFAULT_MOTIF_MIN_N)
    parser.add_argument("--motif-max-n", type=int, default=DEFAULT_MOTIF_MAX_N)
    parser.add_argument("--context-window", type=int, default=DEFAULT_CONTEXT_WINDOW)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--neighbor-k", type=int, default=DEFAULT_NEIGHBOR_K)
    parser.add_argument("--eval-token-count", type=int, default=DEFAULT_EVAL_TOKEN_COUNT)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--limit-chunks", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    report = audit_beat_chunk_motif_embeddings(
        chunk_cache_path=args.chunk_cache_path,
        refinement_report_path=args.refinement_report_path,
        motif_quality_report_path=args.motif_quality_report_path,
        report_path=args.report_path,
        result_log_path=args.result_log_path,
        motif_vocab_size=args.motif_vocab_size,
        embedding_vocab_size=args.embedding_vocab_size,
        motif_min_n=args.motif_min_n,
        motif_max_n=args.motif_max_n,
        context_window=args.context_window,
        embedding_dim=args.embedding_dim,
        neighbor_k=args.neighbor_k,
        eval_token_count=args.eval_token_count,
        random_seed=args.random_seed,
        limit_chunks=args.limit_chunks,
        command=_format_command(argv),
    )
    print(json.dumps(_normalize_json(report), allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
