from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from pulsefield_model.data.mapper_sparse_windows_v2_1 import collate_mapper_v2_1_windows
from pulsefield_model.evals.c3_auxiliary_diagnostics import (
    DEFAULT_TOP_KS,
    build_datasets_from_config,
    build_unigram_top_ids,
    load_model_from_checkpoint,
    token_kind_lookup,
)
from pulsefield_model.training.common import select_torch_device
from pulsefield_model.training.mapper_common import _move_mapper_batch_tensors


SUMMARY_SCHEMA_VERSION = 1
DEFAULT_RANK_KS = (20, 50, 100, 200)


@dataclass(frozen=True)
class RawTokenParts:
    token: str
    payload: str
    family: str
    fields: tuple[str, ...]
    numeric_fields: tuple[str, ...]
    tail: str
    has_tail: bool


class _RawBucketAccumulator:
    def __init__(self) -> None:
        self.target_count = 0
        self.model_hits_at_20 = 0
        self.unigram_hits_at_20 = 0
        self.model_only_hits_at_20 = 0
        self.unigram_only_hits_at_20 = 0

    def update(self, *, model_hit: bool, unigram_hit: bool) -> None:
        self.target_count += 1
        if model_hit:
            self.model_hits_at_20 += 1
        if unigram_hit:
            self.unigram_hits_at_20 += 1
        if model_hit and not unigram_hit:
            self.model_only_hits_at_20 += 1
        if unigram_hit and not model_hit:
            self.unigram_only_hits_at_20 += 1

    def to_dict(self) -> dict[str, Any]:
        model_recall = _safe_divide(self.model_hits_at_20, self.target_count)
        unigram_recall = _safe_divide(self.unigram_hits_at_20, self.target_count)
        return {
            "target_count": self.target_count,
            "model_hits_at_20": self.model_hits_at_20,
            "unigram_hits_at_20": self.unigram_hits_at_20,
            "model_only_hits_at_20": self.model_only_hits_at_20,
            "unigram_only_hits_at_20": self.unigram_only_hits_at_20,
            "model_recall_at_20": model_recall,
            "unigram_recall_at_20": unigram_recall,
            "model_minus_unigram_recall_at_20": (
                None if model_recall is None or unigram_recall is None else model_recall - unigram_recall
            ),
        }


def parse_raw_token_text(token_text: str) -> RawTokenParts:
    if not str(token_text).startswith("RAW|"):
        raise ValueError(f"expected RAW token, got {token_text!r}")
    payload = str(token_text).split("|", 1)[1]
    parts = tuple(payload.split(":")) if payload else ("",)
    family = parts[0] if parts else ""
    fields = parts[1:]
    numeric_fields = tuple(field for field in fields if _is_int_string(field))
    tail_fields = tuple(field for field in fields if not _is_int_string(field))
    tail = ":".join(tail_fields)
    return RawTokenParts(
        token=str(token_text),
        payload=payload,
        family=family,
        fields=fields,
        numeric_fields=numeric_fields,
        tail=tail,
        has_tail=bool(tail_fields),
    )


def raw_bucket_keys(token_text: str, train_count: int) -> tuple[tuple[str, str], ...]:
    parts = parse_raw_token_text(token_text)
    field_1 = parts.numeric_fields[0] if len(parts.numeric_fields) >= 1 else "missing"
    field_2 = parts.numeric_fields[1] if len(parts.numeric_fields) >= 2 else "missing"
    field_3 = parts.numeric_fields[2] if len(parts.numeric_fields) >= 3 else "missing"
    field_4 = parts.numeric_fields[3] if len(parts.numeric_fields) >= 4 else "missing"
    return (
        ("family", parts.family or "missing"),
        ("numeric_field_count", str(len(parts.numeric_fields))),
        ("field_1", field_1),
        ("field_2", field_2),
        ("field_3", field_3),
        ("field_4", field_4),
        ("has_tail", "yes" if parts.has_tail else "no"),
        ("tail_prefix", _tail_prefix(parts.tail)),
        ("train_frequency", train_frequency_bucket(train_count)),
    )


def train_frequency_bucket(count: int) -> str:
    value = int(count)
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 4:
        return "2to4"
    if value <= 9:
        return "5to9"
    if value <= 24:
        return "10to24"
    if value <= 49:
        return "25to49"
    if value <= 99:
        return "50to99"
    if value <= 249:
        return "100to249"
    return "250plus"


def run_raw_structure_audit(
    *,
    config_path: Path,
    checkpoint_path: Path,
    batch_size: int,
    device_name: str,
    rank_ks: Sequence[int] = DEFAULT_RANK_KS,
    max_batches: int | None = None,
) -> dict[str, Any]:
    start = time.monotonic()
    rank_ks = tuple(sorted({int(k) for k in rank_ks}))
    if not rank_ks or rank_ks[0] <= 0:
        raise ValueError("rank_ks must contain positive integers")
    max_k = max(rank_ks)

    datasets = build_datasets_from_config(config_path)
    unigram_counts, unigram_top_ids = build_unigram_top_ids(
        datasets.train_dataset,
        datasets.token_lookup,
        limit=max_k,
    )
    kind_by_id = token_kind_lookup(datasets.token_by_id)
    raw_ids = tuple(sorted(token_id for token_id, kind in kind_by_id.items() if kind == "RAW"))
    raw_id_set = frozenset(raw_ids)
    unigram_raw_top_ids = tuple(
        token_id for token_id, _count in Counter({token_id: unigram_counts.get(token_id, 0) for token_id in raw_ids}).most_common(max_k)
    )

    device = select_torch_device(device_name)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device=device)
    loader = DataLoader(
        datasets.eval_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_mapper_v2_1_windows,
    )

    raw_id_tensor = torch.tensor([token_id - 1 for token_id in raw_ids], dtype=torch.long)
    totals = {
        "sample_count": 0,
        "available_sample_count": 0,
        "positive_sample_count": 0,
        "raw_positive_sample_count": 0,
        "raw_positive_label_count": 0,
    }
    model_hits_by_k = Counter({k: 0 for k in rank_ks})
    unigram_hits_by_k = Counter({k: 0 for k in rank_ks})
    model_raw_only_hits_by_k = Counter({k: 0 for k in rank_ks})
    unigram_raw_only_hits_by_k = Counter({k: 0 for k in rank_ks})
    full_rank_depth = Counter()
    raw_rank_depth = Counter()
    target_counts: Counter[int] = Counter()
    model_hit_counts: Counter[int] = Counter()
    unigram_hit_counts: Counter[int] = Counter()
    model_prediction_counts: Counter[int] = Counter()
    model_false_positive_counts: Counter[int] = Counter()
    bucket_accumulators: dict[tuple[str, str], _RawBucketAccumulator] = defaultdict(_RawBucketAccumulator)

    model.eval()
    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = _move_mapper_batch_tensors(raw_batch, device)
            output = model(batch)
            logits = output.c3_auxiliary_logits
            if not isinstance(logits, torch.Tensor):
                raise ValueError("checkpoint model did not produce c3_auxiliary_logits")
            logits_cpu = logits.detach().cpu()
            raw_logits_cpu = logits_cpu.index_select(dim=1, index=raw_id_tensor)
            full_top_ids = (torch.topk(logits_cpu, k=min(max_k, int(logits_cpu.shape[1])), dim=1).indices + 1).tolist()
            raw_top_positions = torch.topk(raw_logits_cpu, k=min(max_k, len(raw_ids)), dim=1).indices.tolist()
            raw_top_ids = [[raw_ids[position] for position in row] for row in raw_top_positions]
            token_rows, mask_rows, available_rows = _batch_token_rows(batch)
            totals["sample_count"] += int(logits_cpu.shape[0])
            for row_index in range(int(logits_cpu.shape[0])):
                if not available_rows[row_index]:
                    continue
                totals["available_sample_count"] += 1
                target_ids = frozenset(
                    token_id
                    for token_id, is_valid in zip(token_rows[row_index], mask_rows[row_index])
                    if is_valid and token_id > 0
                )
                if target_ids:
                    totals["positive_sample_count"] += 1
                raw_target_ids = frozenset(target_id for target_id in target_ids if target_id in raw_id_set)
                if not raw_target_ids:
                    continue
                totals["raw_positive_sample_count"] += 1
                totals["raw_positive_label_count"] += len(raw_target_ids)
                _update_raw_sample(
                    raw_target_ids=raw_target_ids,
                    full_model_top_ids=tuple(int(token_id) for token_id in full_top_ids[row_index]),
                    raw_model_top_ids=tuple(int(token_id) for token_id in raw_top_ids[row_index]),
                    unigram_top_ids=tuple(int(token_id) for token_id in unigram_top_ids),
                    unigram_raw_top_ids=tuple(int(token_id) for token_id in unigram_raw_top_ids),
                    rank_ks=rank_ks,
                    token_by_id=datasets.token_by_id,
                    unigram_counts=unigram_counts,
                    model_hits_by_k=model_hits_by_k,
                    unigram_hits_by_k=unigram_hits_by_k,
                    model_raw_only_hits_by_k=model_raw_only_hits_by_k,
                    unigram_raw_only_hits_by_k=unigram_raw_only_hits_by_k,
                    full_rank_depth=full_rank_depth,
                    raw_rank_depth=raw_rank_depth,
                    target_counts=target_counts,
                    model_hit_counts=model_hit_counts,
                    unigram_hit_counts=unigram_hit_counts,
                    model_prediction_counts=model_prediction_counts,
                    model_false_positive_counts=model_false_positive_counts,
                    bucket_accumulators=bucket_accumulators,
                )

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "C3 RAW-structure audit",
        "config_path": config_path.as_posix(),
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint": {
            "run_name": checkpoint.get("run_name"),
            "completed_steps": int(checkpoint.get("training_state", {}).get("step", 0)),
            "is_complete": bool(checkpoint.get("training_state", {}).get("is_complete", False)),
            "use_c3_auxiliary_target": bool(
                checkpoint.get("model_config", {}).get("use_c3_auxiliary_target", False)
            ),
            "use_c3_auxiliary_kind_heads": bool(
                checkpoint.get("model_config", {}).get("use_c3_auxiliary_kind_heads", False)
            ),
            "c3_auxiliary_vocab_size": int(checkpoint.get("model_config", {}).get("c3_auxiliary_vocab_size", 0)),
            "c3_auxiliary_kind_vocab_sizes": list(
                checkpoint.get("model_config", {}).get("c3_auxiliary_kind_vocab_sizes", []) or []
            ),
        },
        "dataset": {
            "train_window_count": len(datasets.train_dataset),
            "eval_window_count": len(datasets.eval_dataset),
            "batch_size": int(batch_size),
            "max_batches": max_batches,
        },
        "raw_vocab": {
            "raw_token_count": len(raw_ids),
            "raw_train_positive_count": int(sum(unigram_counts.get(token_id, 0) for token_id in raw_ids)),
            "top_train_raw_tokens": _top_token_rows(unigram_raw_top_ids[:20], datasets.token_by_id, unigram_counts),
        },
        "totals": totals,
        "raw_topk": _topk_summary(
            rank_ks=rank_ks,
            target_count=totals["raw_positive_label_count"],
            model_hits_by_k=model_hits_by_k,
            unigram_hits_by_k=unigram_hits_by_k,
            model_raw_only_hits_by_k=model_raw_only_hits_by_k,
            unigram_raw_only_hits_by_k=unigram_raw_only_hits_by_k,
        ),
        "rank_depth": {
            "full_vocab": _counter_to_rank_dict(full_rank_depth, totals["raw_positive_label_count"]),
            "raw_only": _counter_to_rank_dict(raw_rank_depth, totals["raw_positive_label_count"]),
        },
        "bucket_metrics": _bucket_metric_rows(bucket_accumulators),
        "top_raw_targets": _top_token_rows([token_id for token_id, _ in target_counts.most_common(20)], datasets.token_by_id, unigram_counts, target_counts=target_counts, model_hit_counts=model_hit_counts, unigram_hit_counts=unigram_hit_counts),
        "top_model_missed_raw_targets": _top_missed_rows(target_counts, model_hit_counts, datasets.token_by_id, unigram_counts),
        "top_unigram_only_raw_targets": _top_unigram_only_rows(target_counts, model_hit_counts, unigram_hit_counts, datasets.token_by_id, unigram_counts),
        "top_model_only_raw_hits": _top_model_only_rows(target_counts, model_hit_counts, unigram_hit_counts, datasets.token_by_id, unigram_counts),
        "top_model_raw_false_positives": _top_false_positive_rows(model_false_positive_counts, model_prediction_counts, datasets.token_by_id, unigram_counts),
        "decision": {},
        "elapsed_s": time.monotonic() - start,
    }
    summary["decision"] = _decision(summary)
    return _normalize_json(summary)


def _update_raw_sample(
    *,
    raw_target_ids: frozenset[int],
    full_model_top_ids: Sequence[int],
    raw_model_top_ids: Sequence[int],
    unigram_top_ids: Sequence[int],
    unigram_raw_top_ids: Sequence[int],
    rank_ks: Sequence[int],
    token_by_id: Mapping[int, str],
    unigram_counts: Mapping[int, int],
    model_hits_by_k: Counter[int],
    unigram_hits_by_k: Counter[int],
    model_raw_only_hits_by_k: Counter[int],
    unigram_raw_only_hits_by_k: Counter[int],
    full_rank_depth: Counter[str],
    raw_rank_depth: Counter[str],
    target_counts: Counter[int],
    model_hit_counts: Counter[int],
    unigram_hit_counts: Counter[int],
    model_prediction_counts: Counter[int],
    model_false_positive_counts: Counter[int],
    bucket_accumulators: dict[tuple[str, str], _RawBucketAccumulator],
) -> None:
    selected_k = 20
    model_top20 = frozenset(int(token_id) for token_id in full_model_top_ids[:selected_k])
    unigram_top20 = frozenset(int(token_id) for token_id in unigram_top_ids[:selected_k])
    for token_id in full_model_top_ids[:selected_k]:
        if str(token_by_id.get(int(token_id), "")).startswith("RAW|"):
            model_prediction_counts[int(token_id)] += 1
            if int(token_id) not in raw_target_ids:
                model_false_positive_counts[int(token_id)] += 1
    for k in rank_ks:
        model_hits_by_k[k] += len(raw_target_ids.intersection(full_model_top_ids[:k]))
        unigram_hits_by_k[k] += len(raw_target_ids.intersection(unigram_top_ids[:k]))
        model_raw_only_hits_by_k[k] += len(raw_target_ids.intersection(raw_model_top_ids[:k]))
        unigram_raw_only_hits_by_k[k] += len(raw_target_ids.intersection(unigram_raw_top_ids[:k]))
    full_rank_lookup = {int(token_id): rank + 1 for rank, token_id in enumerate(full_model_top_ids)}
    raw_rank_lookup = {int(token_id): rank + 1 for rank, token_id in enumerate(raw_model_top_ids)}
    for target_id in raw_target_ids:
        target_counts[target_id] += 1
        model_hit = target_id in model_top20
        unigram_hit = target_id in unigram_top20
        if model_hit:
            model_hit_counts[target_id] += 1
        if unigram_hit:
            unigram_hit_counts[target_id] += 1
        full_rank_depth[_rank_depth_bucket(full_rank_lookup.get(target_id))] += 1
        raw_rank_depth[_rank_depth_bucket(raw_rank_lookup.get(target_id))] += 1
        for bucket_type, bucket in raw_bucket_keys(token_by_id.get(target_id, ""), unigram_counts.get(target_id, 0)):
            bucket_accumulators[(bucket_type, bucket)].update(model_hit=model_hit, unigram_hit=unigram_hit)


def _batch_token_rows(batch: Mapping[str, Any]) -> tuple[list[list[int]], list[list[bool]], list[bool]]:
    tokens = batch["c3_side_stream_tokens"].detach().cpu()
    token_mask = batch.get("c3_side_stream_token_mask")
    if token_mask is None:
        effective_mask = tokens.ne(0)
    else:
        effective_mask = token_mask.detach().cpu().to(dtype=torch.bool) & tokens.ne(0)
    available = batch.get("c3_side_stream_available")
    if available is None:
        available_mask = torch.ones((int(tokens.shape[0]),), dtype=torch.bool)
    else:
        available_mask = available.detach().cpu().to(dtype=torch.bool).reshape(-1)
    return (
        [[int(value) for value in row] for row in tokens.tolist()],
        [[bool(value) for value in row] for row in effective_mask.tolist()],
        [bool(value) for value in available_mask.tolist()],
    )


def _topk_summary(
    *,
    rank_ks: Sequence[int],
    target_count: int,
    model_hits_by_k: Mapping[int, int],
    unigram_hits_by_k: Mapping[int, int],
    model_raw_only_hits_by_k: Mapping[int, int],
    unigram_raw_only_hits_by_k: Mapping[int, int],
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for k in rank_ks:
        model_hits = int(model_hits_by_k[k])
        unigram_hits = int(unigram_hits_by_k[k])
        model_raw_only_hits = int(model_raw_only_hits_by_k[k])
        unigram_raw_only_hits = int(unigram_raw_only_hits_by_k[k])
        model_recall = _safe_divide(model_hits, target_count)
        unigram_recall = _safe_divide(unigram_hits, target_count)
        model_raw_only_recall = _safe_divide(model_raw_only_hits, target_count)
        unigram_raw_only_recall = _safe_divide(unigram_raw_only_hits, target_count)
        rows[str(k)] = {
            "model_hits": model_hits,
            "unigram_hits": unigram_hits,
            "model_recall": model_recall,
            "unigram_recall": unigram_recall,
            "model_minus_unigram_recall": (
                None if model_recall is None or unigram_recall is None else model_recall - unigram_recall
            ),
            "model_raw_only_hits": model_raw_only_hits,
            "unigram_raw_only_hits": unigram_raw_only_hits,
            "model_raw_only_recall": model_raw_only_recall,
            "unigram_raw_only_recall": unigram_raw_only_recall,
            "model_raw_only_minus_unigram_raw_only_recall": (
                None
                if model_raw_only_recall is None or unigram_raw_only_recall is None
                else model_raw_only_recall - unigram_raw_only_recall
            ),
        }
    return rows


def _bucket_metric_rows(bucket_accumulators: Mapping[tuple[str, str], _RawBucketAccumulator]) -> list[dict[str, Any]]:
    rows = []
    for (bucket_type, bucket), accumulator in bucket_accumulators.items():
        row = {"bucket_type": bucket_type, "bucket": bucket}
        row.update(accumulator.to_dict())
        rows.append(row)
    rows.sort(key=lambda row: (str(row["bucket_type"]), -int(row["target_count"]), str(row["bucket"])))
    return rows


def _top_token_rows(
    token_ids: Sequence[int],
    token_by_id: Mapping[int, str],
    unigram_counts: Mapping[int, int],
    *,
    target_counts: Mapping[int, int] | None = None,
    model_hit_counts: Mapping[int, int] | None = None,
    unigram_hit_counts: Mapping[int, int] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for token_id in token_ids:
        token_text = token_by_id.get(int(token_id), "")
        row: dict[str, Any] = {
            "token_id": int(token_id),
            "token": token_text,
            "train_count": int(unigram_counts.get(int(token_id), 0)),
        }
        if token_text.startswith("RAW|"):
            parts = parse_raw_token_text(token_text)
            row.update(
                {
                    "family": parts.family,
                    "numeric_fields": list(parts.numeric_fields),
                    "has_tail": parts.has_tail,
                    "tail": parts.tail,
                }
            )
        if target_counts is not None:
            row["target_count"] = int(target_counts.get(int(token_id), 0))
        if model_hit_counts is not None:
            row["model_hits_at_20"] = int(model_hit_counts.get(int(token_id), 0))
        if unigram_hit_counts is not None:
            row["unigram_hits_at_20"] = int(unigram_hit_counts.get(int(token_id), 0))
        rows.append(row)
    return rows


def _top_missed_rows(
    target_counts: Mapping[int, int],
    model_hit_counts: Mapping[int, int],
    token_by_id: Mapping[int, str],
    unigram_counts: Mapping[int, int],
) -> list[dict[str, Any]]:
    missed = [
        token_id
        for token_id, target_count in sorted(target_counts.items(), key=lambda item: (-item[1], item[0]))
        if int(target_count) > int(model_hit_counts.get(token_id, 0))
    ]
    return _top_token_rows(missed[:20], token_by_id, unigram_counts, target_counts=target_counts, model_hit_counts=model_hit_counts)


def _top_unigram_only_rows(
    target_counts: Mapping[int, int],
    model_hit_counts: Mapping[int, int],
    unigram_hit_counts: Mapping[int, int],
    token_by_id: Mapping[int, str],
    unigram_counts: Mapping[int, int],
) -> list[dict[str, Any]]:
    token_ids = [
        token_id
        for token_id, _target_count in sorted(target_counts.items(), key=lambda item: (-item[1], item[0]))
        if int(unigram_hit_counts.get(token_id, 0)) > int(model_hit_counts.get(token_id, 0))
    ]
    return _top_token_rows(
        token_ids[:20],
        token_by_id,
        unigram_counts,
        target_counts=target_counts,
        model_hit_counts=model_hit_counts,
        unigram_hit_counts=unigram_hit_counts,
    )


def _top_model_only_rows(
    target_counts: Mapping[int, int],
    model_hit_counts: Mapping[int, int],
    unigram_hit_counts: Mapping[int, int],
    token_by_id: Mapping[int, str],
    unigram_counts: Mapping[int, int],
) -> list[dict[str, Any]]:
    token_ids = [
        token_id
        for token_id, _target_count in sorted(target_counts.items(), key=lambda item: (-item[1], item[0]))
        if int(model_hit_counts.get(token_id, 0)) > int(unigram_hit_counts.get(token_id, 0))
    ]
    return _top_token_rows(
        token_ids[:20],
        token_by_id,
        unigram_counts,
        target_counts=target_counts,
        model_hit_counts=model_hit_counts,
        unigram_hit_counts=unigram_hit_counts,
    )


def _top_false_positive_rows(
    false_positive_counts: Mapping[int, int],
    prediction_counts: Mapping[int, int],
    token_by_id: Mapping[int, str],
    unigram_counts: Mapping[int, int],
) -> list[dict[str, Any]]:
    rows = _top_token_rows(
        [token_id for token_id, _ in sorted(false_positive_counts.items(), key=lambda item: (-item[1], item[0]))[:20]],
        token_by_id,
        unigram_counts,
    )
    for row in rows:
        token_id = int(row["token_id"])
        row["model_predictions_at_20"] = int(prediction_counts.get(token_id, 0))
        row["model_false_positives_at_20"] = int(false_positive_counts.get(token_id, 0))
    return rows


def _counter_to_rank_dict(counter: Mapping[str, int], denominator: int) -> dict[str, Any]:
    order = ("1to20", "21to50", "51to100", "101to200", "over200")
    return {
        bucket: {
            "count": int(counter.get(bucket, 0)),
            "rate": _safe_divide(counter.get(bucket, 0), denominator),
        }
        for bucket in order
    }


def _rank_depth_bucket(rank: int | None) -> str:
    if rank is None:
        return "over200"
    if rank <= 20:
        return "1to20"
    if rank <= 50:
        return "21to50"
    if rank <= 100:
        return "51to100"
    if rank <= 200:
        return "101to200"
    return "over200"


def _decision(summary: Mapping[str, Any]) -> dict[str, Any]:
    raw20 = summary["raw_topk"]["20"]
    model_recall = raw20["model_recall"]
    unigram_recall = raw20["unigram_recall"]
    model_raw_only_recall = raw20["model_raw_only_recall"]
    unigram_raw_only_recall = raw20["unigram_raw_only_recall"]
    full_rank = summary["rank_depth"]["full_vocab"]
    rank_near_rate = sum(float(full_rank[bucket]["rate"] or 0.0) for bucket in ("21to50", "51to100", "101to200"))
    raw_only_beats_unigram = (
        model_raw_only_recall is not None
        and unigram_raw_only_recall is not None
        and float(model_raw_only_recall) > float(unigram_raw_only_recall)
    )
    if model_recall is not None and unigram_recall is not None and float(model_recall) > float(unigram_recall):
        route = "TEST_NEXT"
        next_step = "Retest reduced kind-head training with RAW-specific diagnostics online."
        interpretation = "RAW model recall now beats the unigram comparator, so the previous blocker is not present in this audit."
    elif raw_only_beats_unigram:
        route = "MUTATE"
        next_step = "Test RAW calibration or per-kind RAW ranking because RAW-only rank beats unigram but full-vocab top-20 does not."
        interpretation = (
            "The model ranks RAW targets well inside the RAW slice, but full-vocab competition hides them at K=20. "
            "This supports a RAW calibration/ranking mutation rather than an immediate grammar rewrite."
        )
    elif rank_near_rate >= 0.15:
        route = "MUTATE"
        next_step = "Test a smaller RAW split or calibration mutation, because many misses are rank-near by K=200."
        interpretation = (
            "RAW still trails unigram at K=20, but a meaningful share of missed labels appears between ranks 21 and 200. "
            "The next step should use the bucket tables to choose a grounded RAW split or calibration probe."
        )
    else:
        route = "MUTATE_TO_ORDERED_GRAMMAR"
        next_step = "Stop blind bag-head refinement and design an ordered C3 target-grammar card."
        interpretation = (
            "RAW misses are not rank-near under the current bag objective. This weakens the case for more bag-head tuning "
            "and points toward ordered C3 target representation."
        )
    return {
        "route": route,
        "model_raw_recall_at_20": model_recall,
        "unigram_raw_recall_at_20": unigram_recall,
        "model_minus_unigram_raw_recall_at_20": raw20["model_minus_unigram_recall"],
        "model_raw_only_recall_at_20": model_raw_only_recall,
        "unigram_raw_only_recall_at_20": unigram_raw_only_recall,
        "rank_near_rate_21_to_200": rank_near_rate,
        "raw_only_beats_unigram": raw_only_beats_unigram,
        "interpretation": interpretation,
        "next_step": next_step,
    }


def write_summary(summary: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown_report(summary: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw20 = summary["raw_topk"]["20"]
    raw50 = summary["raw_topk"].get("50", {})
    raw100 = summary["raw_topk"].get("100", {})
    decision = summary["decision"]
    bucket_rows = _selected_bucket_table_rows(summary["bucket_metrics"])
    key_findings = _key_finding_rows(summary)
    text = "\n".join(
        [
            "# C3 RAW-Structure Audit Result Report",
            "",
            "## Scope",
            "",
            "This P20 pass audits the P19 reduced kind-head checkpoint without retraining. It focuses only on RAW C3 labels because P19 tied unigram on REF, beat unigram on RES, and left RAW as the main blocker.",
            "",
            "## Result",
            "",
            f"Decision: {decision['route']}.",
            "",
            f"- eval windows: `{summary['dataset']['eval_window_count']}`",
            f"- RAW positive samples: `{summary['totals']['raw_positive_sample_count']}`",
            f"- RAW positive labels: `{summary['totals']['raw_positive_label_count']}`",
            f"- RAW vocab size: `{summary['raw_vocab']['raw_token_count']}`",
            f"- model RAW recall@20: `{_fmt_metric(raw20['model_recall'])}`",
            f"- unigram RAW recall@20: `{_fmt_metric(raw20['unigram_recall'])}`",
            f"- model minus unigram RAW recall@20: `{_fmt_metric(raw20['model_minus_unigram_recall'])}`",
            f"- model RAW-only recall@20: `{_fmt_metric(raw20['model_raw_only_recall'])}`",
            f"- unigram RAW-only recall@20: `{_fmt_metric(raw20['unigram_raw_only_recall'])}`",
            f"- model RAW recall@50: `{_fmt_metric(raw50.get('model_recall'))}`",
            f"- unigram RAW recall@50: `{_fmt_metric(raw50.get('unigram_recall'))}`",
            f"- model RAW recall@100: `{_fmt_metric(raw100.get('model_recall'))}`",
            f"- rank-near rate ranks 21-200: `{_fmt_metric(decision['rank_near_rate_21_to_200'])}`",
            f"- elapsed: `{float(summary['elapsed_s']):.2f}s`",
            "",
            "## Key Findings",
            "",
            *key_findings,
            "",
            "## Rank Depth",
            "",
            "| Bucket | Full-vocab count | Full-vocab rate | RAW-only count | RAW-only rate |",
            "| --- | ---: | ---: | ---: | ---: |",
            *_rank_rows(summary),
            "",
            "## Key Buckets",
            "",
            "| Bucket type | Bucket | Targets | Model hit@20 | Unigram hit@20 | Model recall | Unigram recall | Delta |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *bucket_rows,
            "",
            "## Top RAW Misses",
            "",
            "| Token | Target count | Train count | Model hits@20 | Fields | Tail |",
            "| --- | ---: | ---: | ---: | --- | --- |",
            *_token_rows(summary["top_model_missed_raw_targets"], include_unigram=False),
            "",
            "## Top Unigram-Only RAW Targets",
            "",
            "| Token | Target count | Train count | Model hits@20 | Unigram hits@20 | Fields | Tail |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
            *_token_rows(summary["top_unigram_only_raw_targets"], include_unigram=True),
            "",
            "## What Passed",
            "",
            "- The P19 checkpoint, config, reduced sidecar, and eval split are loadable.",
            "- The audit preserves the C3 legality constraint: labels are target-side only.",
            "- RAW diagnostics now expose frequency, parsed-field, tail/composite, and rank-depth structure.",
            "",
            "## What Surfaced",
            "",
            decision["interpretation"],
            "",
            "## Next Step",
            "",
            decision["next_step"],
            "",
        ]
    )
    output_path.write_text(text, encoding="utf-8")


def _key_finding_rows(summary: Mapping[str, Any]) -> list[str]:
    raw20 = summary["raw_topk"]["20"]
    raw50 = summary["raw_topk"].get("50", {})
    raw_only_gap = raw20.get("model_raw_only_minus_unigram_raw_only_recall")
    full_gap = raw20.get("model_minus_unigram_recall")
    has_tail_no = _find_bucket(summary["bucket_metrics"], "has_tail", "no")
    field_2_under = _bucket_value_list(summary["bucket_metrics"], bucket_type="field_2", sign="negative", limit=4)
    field_2_over = _bucket_value_list(summary["bucket_metrics"], bucket_type="field_2", sign="positive", limit=3)
    rows = [
        (
            f"- RAW-only top-20 gap is `{_fmt_metric(raw_only_gap)}`, versus full-vocab top-20 gap "
            f"`{_fmt_metric(full_gap)}`. This points to ranking/calibration pressure across concatenated kind heads, "
            "not only RAW separability."
        ),
        (
            f"- Full-vocab K=50 nearly closes the RAW gap: model `{_fmt_metric(raw50.get('model_recall'))}` "
            f"versus unigram `{_fmt_metric(raw50.get('unigram_recall'))}`."
        ),
        (
            f"- Rank-near targets are substantial: `{_fmt_metric(summary['decision']['rank_near_rate_21_to_200'])}` "
            "of RAW labels are between full-vocab ranks 21 and 200."
        ),
    ]
    if has_tail_no is not None:
        rows.append(
            f"- The reduced eval RAW targets are not tail/composite-heavy: `has_tail=no` covers "
            f"`{has_tail_no.get('target_count')}` / `{summary['totals']['raw_positive_label_count']}` RAW labels."
        )
    if field_2_under:
        rows.append(f"- Syntactic `field_2` underperforming buckets: {', '.join(field_2_under)}.")
    if field_2_over:
        rows.append(f"- Syntactic `field_2` overperforming buckets: {', '.join(field_2_over)}.")
    return rows


def _find_bucket(rows: Sequence[Mapping[str, Any]], bucket_type: str, bucket: str) -> Mapping[str, Any] | None:
    for row in rows:
        if row.get("bucket_type") == bucket_type and row.get("bucket") == bucket:
            return row
    return None


def _bucket_value_list(
    rows: Sequence[Mapping[str, Any]],
    *,
    bucket_type: str,
    sign: str,
    limit: int,
) -> list[str]:
    typed = [row for row in rows if row.get("bucket_type") == bucket_type and int(row.get("target_count", 0)) >= 20]
    if sign == "negative":
        typed.sort(key=lambda row: float(row.get("model_minus_unigram_recall_at_20") or 0.0))
        selected = [row for row in typed if float(row.get("model_minus_unigram_recall_at_20") or 0.0) < 0.0]
    elif sign == "positive":
        typed.sort(key=lambda row: float(row.get("model_minus_unigram_recall_at_20") or 0.0), reverse=True)
        selected = [row for row in typed if float(row.get("model_minus_unigram_recall_at_20") or 0.0) > 0.0]
    else:
        raise ValueError(f"unknown sign: {sign}")
    return [
        f"`{row['bucket']}` ({_fmt_metric(row.get('model_minus_unigram_recall_at_20'))})"
        for row in selected[: int(limit)]
    ]


def _rank_rows(summary: Mapping[str, Any]) -> list[str]:
    rows = []
    full = summary["rank_depth"]["full_vocab"]
    raw = summary["rank_depth"]["raw_only"]
    for bucket in ("1to20", "21to50", "51to100", "101to200", "over200"):
        rows.append(
            f"| {bucket} | {full[bucket]['count']} | {_fmt_metric(full[bucket]['rate'])} | "
            f"{raw[bucket]['count']} | {_fmt_metric(raw[bucket]['rate'])} |"
        )
    return rows


def _selected_bucket_table_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    selected_types = {"train_frequency", "field_1", "field_2", "has_tail", "tail_prefix"}
    filtered = [row for row in rows if row.get("bucket_type") in selected_types]
    filtered.sort(
        key=lambda row: (
            str(row.get("bucket_type")),
            float(row.get("model_minus_unigram_recall_at_20") or 0.0),
            -int(row.get("target_count", 0)),
        )
    )
    rendered = []
    for row in filtered[:30]:
        rendered.append(
            f"| {row['bucket_type']} | {row['bucket']} | {row['target_count']} | "
            f"{row['model_hits_at_20']} | {row['unigram_hits_at_20']} | "
            f"{_fmt_metric(row['model_recall_at_20'])} | {_fmt_metric(row['unigram_recall_at_20'])} | "
            f"{_fmt_metric(row['model_minus_unigram_recall_at_20'])} |"
        )
    return rendered


def _token_rows(rows: Sequence[Mapping[str, Any]], *, include_unigram: bool) -> list[str]:
    rendered = []
    for row in rows[:20]:
        fields = ",".join(str(field) for field in row.get("numeric_fields", []))
        tail = str(row.get("tail") or ".")
        if include_unigram:
            rendered.append(
                f"| `{row['token']}` | {row.get('target_count', 0)} | {row.get('train_count', 0)} | "
                f"{row.get('model_hits_at_20', 0)} | {row.get('unigram_hits_at_20', 0)} | `{fields}` | `{tail}` |"
            )
        else:
            rendered.append(
                f"| `{row['token']}` | {row.get('target_count', 0)} | {row.get('train_count', 0)} | "
                f"{row.get('model_hits_at_20', 0)} | `{fields}` | `{tail}` |"
            )
    return rendered


def _normalize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    if isinstance(value, Counter):
        return {str(key): _normalize_json(item) for key, item in value.items()}
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def _tail_prefix(tail: str) -> str:
    if not tail:
        return "none"
    if tail.startswith("O"):
        return "order"
    return tail[:1]


def _is_int_string(value: str) -> bool:
    if not value:
        return False
    return value.isdigit() or (value.startswith("-") and value[1:].isdigit())


def _safe_divide(numerator: float, denominator: float) -> float | None:
    if float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def _fmt_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and not math.isfinite(value):
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return str(value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit RAW structure in C3 auxiliary predictions.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--rank-k", type=int, nargs="+", default=list(DEFAULT_RANK_KS))
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_raw_structure_audit(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        device_name=args.device,
        rank_ks=args.rank_k,
        max_batches=args.max_batches,
    )
    write_summary(summary, args.summary_output)
    if args.report_output is not None:
        write_markdown_report(summary, args.report_output)
    print(
        "c3_raw_structure_audit "
        f"decision={summary['decision']['route']} "
        f"raw_recall_at_20={_fmt_metric(summary['raw_topk']['20']['model_recall'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
