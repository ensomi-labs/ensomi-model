from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from pulsefield_model.data.control_windows import DEFAULT_MAX_CACHED_MAPS
from pulsefield_model.data.mapper_sparse_windows_v2_1 import (
    DEFAULT_C3_SIDE_STREAM_MAX_TOKENS,
    MapperV21WindowDataset,
    collate_mapper_v2_1_windows,
    load_c3_side_stream_token_sidecar,
)
from pulsefield_model.models.control import ControlDemoGlobalEncoder, ControlDemoGlobalEncoderConfig
from pulsefield_model.models.mapper.v2_1 import MapperV21Config, MapperV21Model
from pulsefield_model.training.common import select_torch_device, split_train_eval_dataset
from pulsefield_model.training.mapper_common import _move_mapper_batch_tensors
from pulsefield_model.training.mapper_v2_1 import load_run_config


SUMMARY_SCHEMA_VERSION = 1
DEFAULT_TOP_KS = (1, 3, 5, 10, 20)


@dataclass(frozen=True)
class C3DiagnosticsDatasets:
    train_dataset: Dataset[Any]
    eval_dataset: Dataset[Any]
    token_lookup: Mapping[str, Mapping[int, Sequence[int]]]
    token_by_id: Mapping[int, str]


class C3TopKAccumulator:
    def __init__(self, *, top_ks: Sequence[int], kind_by_id: Mapping[int, str]) -> None:
        normalized_top_ks = tuple(sorted({int(k) for k in top_ks}))
        if not normalized_top_ks or normalized_top_ks[0] <= 0:
            raise ValueError("top_ks must contain positive integers")
        self.top_ks = normalized_top_ks
        self.kind_by_id = dict(kind_by_id)
        self.sample_count = 0
        self.available_sample_count = 0
        self.positive_sample_count = 0
        self.empty_available_sample_count = 0
        self.raw_token_count = 0
        self.positive_label_count = 0
        self.target_kind_counts: Counter[str] = Counter()
        self._systems = {
            "model": _TopKSystemAccumulator(self.top_ks, self.kind_by_id),
            "unigram": _TopKSystemAccumulator(self.top_ks, self.kind_by_id),
        }

    def update(
        self,
        *,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: torch.Tensor | None,
        available: torch.Tensor | None,
        unigram_top_ids: Sequence[int],
    ) -> None:
        if logits.ndim != 2:
            raise ValueError(f"logits must have shape [B,V], got {tuple(logits.shape)}")
        if tokens.ndim != 2 or int(tokens.shape[0]) != int(logits.shape[0]):
            raise ValueError(f"tokens must have shape [B,T], got {tuple(tokens.shape)}")
        if token_mask is None:
            effective_mask = tokens.ne(0)
        else:
            if token_mask.ndim != 2 or tuple(token_mask.shape) != tuple(tokens.shape):
                raise ValueError("token_mask must align with tokens")
            effective_mask = token_mask.to(dtype=torch.bool) & tokens.ne(0)
        if available is None:
            available_mask = torch.ones((int(logits.shape[0]),), dtype=torch.bool, device=tokens.device)
        else:
            available_mask = available.to(dtype=torch.bool).reshape(-1)
            if tuple(available_mask.shape) != (int(logits.shape[0]),):
                raise ValueError("available must align with batch size")

        batch_size = int(logits.shape[0])
        self.sample_count += batch_size
        max_k = min(max(self.top_ks), int(logits.shape[1]))
        model_top_ids = (torch.topk(logits.detach().cpu(), k=max_k, dim=1).indices + 1).tolist()
        token_rows = tokens.detach().cpu().tolist()
        mask_rows = effective_mask.detach().cpu().tolist()
        available_rows = available_mask.detach().cpu().tolist()
        unigram_top_ids = tuple(int(token_id) for token_id in unigram_top_ids[:max_k])

        for row_index in range(batch_size):
            if not bool(available_rows[row_index]):
                continue
            self.available_sample_count += 1
            valid_token_ids = [
                int(token_id)
                for token_id, is_valid in zip(token_rows[row_index], mask_rows[row_index])
                if bool(is_valid) and int(token_id) > 0
            ]
            self.raw_token_count += len(valid_token_ids)
            target_ids = frozenset(valid_token_ids)
            if not target_ids:
                self.empty_available_sample_count += 1
                continue
            self.positive_sample_count += 1
            self.positive_label_count += len(target_ids)
            for target_id in target_ids:
                self.target_kind_counts[self.kind_by_id.get(int(target_id), "UNKNOWN")] += 1
            self._systems["model"].update_sample(
                target_ids=target_ids,
                predicted_ids=tuple(int(token_id) for token_id in model_top_ids[row_index]),
            )
            self._systems["unigram"].update_sample(
                target_ids=target_ids,
                predicted_ids=unigram_top_ids,
            )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "sample_count": self.sample_count,
            "available_sample_count": self.available_sample_count,
            "positive_sample_count": self.positive_sample_count,
            "empty_available_sample_count": self.empty_available_sample_count,
            "raw_token_count": self.raw_token_count,
            "positive_label_count": self.positive_label_count,
            "target_kind_counts": dict(sorted(self.target_kind_counts.items())),
        }
        for name, accumulator in self._systems.items():
            payload[name] = accumulator.to_dict(
                positive_sample_count=self.positive_sample_count,
                positive_label_count=self.positive_label_count,
                target_kind_counts=self.target_kind_counts,
            )
        return payload


class _TopKSystemAccumulator:
    def __init__(self, top_ks: Sequence[int], kind_by_id: Mapping[int, str]) -> None:
        self.top_ks = tuple(top_ks)
        self.kind_by_id = dict(kind_by_id)
        self.hits_by_k = {k: 0 for k in self.top_ks}
        self.sample_hits_by_k = {k: 0 for k in self.top_ks}
        self.kind_hits_by_k = {k: Counter() for k in self.top_ks}
        self.predicted_kind_counts_by_k = {k: Counter() for k in self.top_ks}

    def update_sample(self, *, target_ids: frozenset[int], predicted_ids: Sequence[int]) -> None:
        for k in self.top_ks:
            top_ids = tuple(int(token_id) for token_id in predicted_ids[:k])
            for token_id in top_ids:
                self.predicted_kind_counts_by_k[k][self.kind_by_id.get(token_id, "UNKNOWN")] += 1
            hit_ids = target_ids.intersection(top_ids)
            self.hits_by_k[k] += len(hit_ids)
            if hit_ids:
                self.sample_hits_by_k[k] += 1
            for token_id in hit_ids:
                self.kind_hits_by_k[k][self.kind_by_id.get(token_id, "UNKNOWN")] += 1

    def to_dict(
        self,
        *,
        positive_sample_count: int,
        positive_label_count: int,
        target_kind_counts: Mapping[str, int],
    ) -> dict[str, Any]:
        topk: dict[str, Any] = {}
        for k in self.top_ks:
            hits = self.hits_by_k[k]
            prediction_count = int(k) * int(positive_sample_count)
            kind_hits = dict(sorted(self.kind_hits_by_k[k].items()))
            kind_recall = {
                kind: _safe_divide(kind_hits.get(kind, 0), target_count)
                for kind, target_count in sorted(target_kind_counts.items())
            }
            topk[str(k)] = {
                "hits": hits,
                "micro_recall": _safe_divide(hits, positive_label_count),
                "precision": _safe_divide(hits, prediction_count),
                "sample_hit_count": self.sample_hits_by_k[k],
                "sample_hit_rate": _safe_divide(self.sample_hits_by_k[k], positive_sample_count),
                "kind_hits": kind_hits,
                "kind_recall": kind_recall,
                "predicted_kind_counts": dict(sorted(self.predicted_kind_counts_by_k[k].items())),
            }
        return {"topk": topk}


def token_kind(token_text: str) -> str:
    if not token_text:
        return "UNKNOWN"
    return token_text.split("|", 1)[0] or "UNKNOWN"


def load_token_vocab(sidecar_path: Path) -> dict[int, str]:
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    raw_vocab = payload.get("token_vocab")
    if not isinstance(raw_vocab, Mapping):
        raise ValueError(f"C3 sidecar missing token_vocab: {sidecar_path}")
    token_by_id: dict[int, str] = {}
    for token_text, token_id in raw_vocab.items():
        normalized_id = int(token_id)
        if normalized_id <= 0:
            raise ValueError(f"C3 token ids must be positive, got {normalized_id}")
        token_by_id[normalized_id] = str(token_text)
    if not token_by_id:
        raise ValueError(f"C3 sidecar token_vocab is empty: {sidecar_path}")
    return token_by_id


def token_kind_lookup(token_by_id: Mapping[int, str]) -> dict[int, str]:
    return {int(token_id): token_kind(token_text) for token_id, token_text in token_by_id.items()}


def build_unigram_top_ids(
    train_dataset: Dataset[Any],
    token_lookup: Mapping[str, Mapping[int, Sequence[int]]],
    *,
    limit: int,
) -> tuple[dict[int, int], tuple[int, ...]]:
    if int(limit) <= 0:
        raise ValueError("limit must be positive")
    counts: Counter[int] = Counter()
    for beatmap_path, window_start_ms in iter_dataset_window_keys(train_dataset):
        token_ids = token_lookup.get(beatmap_path, {}).get(window_start_ms)
        if not token_ids:
            continue
        counts.update({int(token_id) for token_id in token_ids if int(token_id) > 0})
    top_ids = tuple(token_id for token_id, _ in counts.most_common(int(limit)))
    return dict(counts), top_ids


def iter_dataset_window_keys(dataset: Dataset[Any]):
    for index in range(len(dataset)):
        record = _control_record_for_dataset_index(dataset, index)
        yield record.beatmap_path.as_posix(), int(record.target_start_ms)


def build_datasets_from_config(config_path: Path) -> C3DiagnosticsDatasets:
    config = load_run_config(config_path)
    sidecar_raw = config.get("c3_side_stream_token_sidecar_path")
    if sidecar_raw is None:
        raise ValueError("run config must define c3_side_stream_token_sidecar_path")
    sidecar_path = Path(sidecar_raw)
    token_lookup = load_c3_side_stream_token_sidecar(sidecar_path)
    token_by_id = load_token_vocab(sidecar_path)
    dataset_kwargs: dict[str, Any] = {
        "dataset_root": Path(config.get("dataset_root", "dataset")),
        "include_full_song_context": bool(config.get("include_full_song_context", True)),
        "max_cached_maps": int(config.get("max_cached_maps") or DEFAULT_MAX_CACHED_MAPS),
        "progress": bool(config.get("dataset_progress", False)),
    }
    if config.get("index_path") is not None:
        dataset_kwargs["index_path"] = Path(config["index_path"])
    if config.get("control_v3_timeseries_path") is not None:
        dataset_kwargs["control_v3_timeseries_path"] = Path(config["control_v3_timeseries_path"])
    if config.get("mapper_record_cache_path") is not None:
        dataset_kwargs["mapper_record_cache_path"] = Path(config["mapper_record_cache_path"])
    if config.get("control_teacher_cache_dir") is not None:
        dataset_kwargs["control_teacher_cache_dir"] = Path(config["control_teacher_cache_dir"])
        dataset_kwargs["require_control_teacher_cache"] = bool(config.get("require_control_teacher_cache", False))
    dataset_kwargs.update(
        {
            "include_c3_side_stream_token_tensors": True,
            "c3_side_stream_token_ids_by_beatmap_path": token_lookup,
            "c3_side_stream_max_tokens": int(
                config.get("c3_side_stream_max_tokens", DEFAULT_C3_SIDE_STREAM_MAX_TOKENS)
            ),
        }
    )
    source_dataset = MapperV21WindowDataset(**dataset_kwargs)
    if config.get("eval_index_path") is not None:
        eval_kwargs = dict(dataset_kwargs)
        eval_kwargs["index_path"] = Path(config["eval_index_path"])
        eval_dataset: Dataset[Any] = MapperV21WindowDataset(**eval_kwargs)
        train_dataset: Dataset[Any] = source_dataset
    else:
        train_dataset, eval_dataset = split_train_eval_dataset(
            source_dataset,
            eval_fraction=float(config.get("eval_fraction", 0.1)),
            eval_size=config.get("eval_size"),
            seed=int(config.get("seed", 1337)),
        )
    if len(eval_dataset) == 0:
        eval_dataset = train_dataset
    return C3DiagnosticsDatasets(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        token_lookup=token_lookup,
        token_by_id=token_by_id,
    )


def run_diagnostics(
    *,
    config_path: Path,
    checkpoint_path: Path,
    batch_size: int,
    device_name: str,
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
    max_batches: int | None = None,
) -> dict[str, Any]:
    start = time.monotonic()
    datasets = build_datasets_from_config(config_path)
    max_k = max(int(k) for k in top_ks)
    unigram_counts, unigram_top_ids = build_unigram_top_ids(
        datasets.train_dataset,
        datasets.token_lookup,
        limit=max_k,
    )
    device = select_torch_device(device_name)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device=device)
    kind_by_id = token_kind_lookup(datasets.token_by_id)
    accumulator = C3TopKAccumulator(top_ks=top_ks, kind_by_id=kind_by_id)
    loader = DataLoader(
        datasets.eval_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_mapper_v2_1_windows,
    )
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
            accumulator.update(
                logits=logits,
                tokens=batch["c3_side_stream_tokens"],
                token_mask=batch.get("c3_side_stream_token_mask"),
                available=batch.get("c3_side_stream_available"),
                unigram_top_ids=unigram_top_ids,
            )

    metrics = accumulator.to_dict()
    comparison = _comparison_summary(metrics, max_k=max_k)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "config_path": config_path.as_posix(),
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint": {
            "run_name": checkpoint.get("run_name"),
            "completed_steps": int(checkpoint.get("training_state", {}).get("step", 0)),
            "is_complete": bool(checkpoint.get("training_state", {}).get("is_complete", False)),
            "use_c3_side_stream_conditioning": bool(
                checkpoint.get("model_config", {}).get("use_c3_side_stream_conditioning", False)
            ),
            "use_c3_auxiliary_target": bool(
                checkpoint.get("model_config", {}).get("use_c3_auxiliary_target", False)
            ),
            "c3_auxiliary_vocab_size": int(checkpoint.get("model_config", {}).get("c3_auxiliary_vocab_size", 0)),
        },
        "dataset": {
            "train_window_count": len(datasets.train_dataset),
            "eval_window_count": len(datasets.eval_dataset),
            "batch_size": int(batch_size),
            "max_batches": max_batches,
        },
        "unigram_baseline": {
            "counted_token_count": int(sum(unigram_counts.values())),
            "unique_token_count": int(len(unigram_counts)),
            "top_tokens": [
                {
                    "token_id": int(token_id),
                    "token": datasets.token_by_id.get(int(token_id), ""),
                    "kind": kind_by_id.get(int(token_id), "UNKNOWN"),
                    "count": int(unigram_counts.get(int(token_id), 0)),
                }
                for token_id in unigram_top_ids
            ],
        },
        "metrics": metrics,
        "comparison": comparison,
        "decision": _decision_from_comparison(comparison, metrics, max_k=max_k),
        "elapsed_s": time.monotonic() - start,
    }


def load_model_from_checkpoint(checkpoint_path: Path, *, device: torch.device) -> tuple[MapperV21Model, Mapping[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint must contain a mapping: {checkpoint_path}")
    model_config = MapperV21Config(**dict(checkpoint["model_config"]))
    control_config_payload = checkpoint.get("control_model_config")
    control_encoder = None
    if isinstance(control_config_payload, Mapping):
        control_encoder = ControlDemoGlobalEncoder(ControlDemoGlobalEncoderConfig(**dict(control_config_payload)))
    model = MapperV21Model(model_config, control_encoder=control_encoder)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, checkpoint


def write_summary(summary: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown_report(summary: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = summary["metrics"]
    comparison = summary["comparison"]
    max_k = int(comparison["selected_k"])
    topk_key = str(max_k)
    model_topk = metrics["model"]["topk"][topk_key]
    unigram_topk = metrics["unigram"]["topk"][topk_key]
    decision = summary["decision"]
    kind_rows = []
    for kind, target_count in sorted(metrics["target_kind_counts"].items()):
        model_recall = model_topk["kind_recall"].get(kind)
        unigram_recall = unigram_topk["kind_recall"].get(kind)
        kind_rows.append(
            f"| {kind} | {target_count} | {_fmt_metric(model_recall)} | {_fmt_metric(unigram_recall)} | "
            f"{model_topk['kind_hits'].get(kind, 0)} | {unigram_topk['kind_hits'].get(kind, 0)} |"
        )
    topk_rows = []
    for k in DEFAULT_TOP_KS:
        key = str(k)
        if key not in metrics["model"]["topk"]:
            continue
        model_row = metrics["model"]["topk"][key]
        unigram_row = metrics["unigram"]["topk"][key]
        topk_rows.append(
            f"| {k} | {_fmt_metric(model_row['micro_recall'])} | {_fmt_metric(unigram_row['micro_recall'])} | "
            f"{_fmt_metric(_relative_lift(model_row['micro_recall'], unigram_row['micro_recall']))} | "
            f"{_fmt_metric(model_row['sample_hit_rate'])} | {_fmt_metric(model_row['precision'])} |"
        )
    text = "\n".join(
        [
            "# C3 Auxiliary-Target Diagnostics Result Report",
            "",
            "## Scope",
            "",
            "This P14 pass scores the P13 enabled C3 auxiliary checkpoint without additional training. It rebuilds the same eval split, compares model top-K C3 bag-label recovery against a train-unigram baseline, and splits recovery by C3 token kind.",
            "",
            "## Result",
            "",
            f"Decision: {decision['route']}.",
            "",
            f"- completed: `{decision['completed']}`",
            f"- eval windows: `{summary['dataset']['eval_window_count']}`",
            f"- positive samples: `{metrics['positive_sample_count']}`",
            f"- positive labels: `{metrics['positive_label_count']}`",
            f"- model recall@{max_k}: `{_fmt_metric(model_topk['micro_recall'])}`",
            f"- unigram recall@{max_k}: `{_fmt_metric(unigram_topk['micro_recall'])}`",
            f"- relative lift@{max_k}: `{_fmt_metric(comparison['relative_lift_at_selected_k'])}`",
            f"- elapsed: `{float(summary['elapsed_s']):.2f}s`",
            "",
            "## Top-K Recovery",
            "",
            "| K | Model recall | Unigram recall | Relative lift | Model sample hit rate | Model precision |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
            *topk_rows,
            "",
            "## Kind Recovery At Selected K",
            "",
            "| Kind | Target labels | Model recall | Unigram recall | Model hits | Unigram hits |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *kind_rows,
            "",
            "## What Passed",
            "",
            "- The checkpoint and exact C3 sidecar were loadable under the P13 config.",
            "- The diagnostics preserve the legality constraint: C3 appears only as target-side labels.",
            "- The report now exposes whether the auxiliary head beats a common-label baseline and which token kinds are recovered.",
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


def _control_record_for_dataset_index(dataset: Dataset[Any], index: int) -> Any:
    if isinstance(dataset, Subset):
        return _control_record_for_dataset_index(dataset.dataset, int(dataset.indices[index]))
    records = getattr(dataset, "records", None)
    if records is None:
        raise ValueError("dataset does not expose records")
    mapper_record = records[int(index)]
    return getattr(mapper_record, "control_record", mapper_record)


def _safe_divide(numerator: float, denominator: float) -> float | None:
    if float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def _relative_lift(model_value: float | None, baseline_value: float | None) -> float | None:
    if model_value is None or baseline_value is None or baseline_value == 0.0:
        return None
    return (float(model_value) - float(baseline_value)) / float(baseline_value)


def _comparison_summary(metrics: Mapping[str, Any], *, max_k: int) -> dict[str, Any]:
    key = str(max_k)
    model_recall = metrics["model"]["topk"][key]["micro_recall"]
    unigram_recall = metrics["unigram"]["topk"][key]["micro_recall"]
    relative_lift = _relative_lift(model_recall, unigram_recall)
    non_raw_model_recall = {
        kind: value
        for kind, value in metrics["model"]["topk"][key]["kind_recall"].items()
        if kind != "RAW"
    }
    return {
        "selected_k": int(max_k),
        "model_recall_at_selected_k": model_recall,
        "unigram_recall_at_selected_k": unigram_recall,
        "absolute_lift_at_selected_k": (
            None if model_recall is None or unigram_recall is None else float(model_recall) - float(unigram_recall)
        ),
        "relative_lift_at_selected_k": relative_lift,
        "non_raw_model_recall_at_selected_k": non_raw_model_recall,
        "non_raw_recovered": any(value is not None and float(value) > 0.0 for value in non_raw_model_recall.values()),
    }


def _decision_from_comparison(
    comparison: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    max_k: int,
) -> dict[str, Any]:
    model_recall = comparison["model_recall_at_selected_k"]
    unigram_recall = comparison["unigram_recall_at_selected_k"]
    relative_lift = comparison["relative_lift_at_selected_k"]
    non_raw_recovered = bool(comparison["non_raw_recovered"])
    completed = model_recall is not None and unigram_recall is not None
    positive = completed and relative_lift is not None and float(relative_lift) >= 0.10 and non_raw_recovered
    beats_unigram = completed and float(model_recall) > float(unigram_recall)
    selected_topk_key = str(max_k)
    positive_label_count = int(metrics.get("positive_label_count", 0))
    model_hits = int(metrics["model"]["topk"][selected_topk_key]["hits"])
    unigram_hits = int(metrics["unigram"]["topk"][selected_topk_key]["hits"])
    if positive:
        route = "TEST_NEXT"
        interpretation = (
            f"The auxiliary head beats the train-unigram comparator at K={max_k} and recovers at least one non-RAW kind. "
            "This supports keeping the legal target-side auxiliary path for one more bounded experiment."
        )
        next_step = "Add online diagnostics or test a kind-balanced auxiliary target while keeping C3 target-side only."
    elif beats_unigram:
        route = "MUTATE"
        interpretation = (
            f"The auxiliary head beats unigram at K={max_k}, but it does not satisfy the full positive gate. "
            "The likely next move is a kind-balanced or decomposed label objective rather than longer identical training."
        )
        next_step = "Mutate toward kind-balanced C3 auxiliary supervision or separate RAW/REF/RES heads."
    else:
        route = "KILL"
        interpretation = (
            f"The auxiliary head does not beat the train-unigram comparator at K={max_k}: "
            f"model hits {model_hits}/{positive_label_count} positive labels while unigram hits "
            f"{unigram_hits}/{positive_label_count}. The P13 BCE loss decrease therefore did not translate into "
            "useful positive-label recovery; it was likely dominated by absent-label calibration. "
            "The current full-vocab bag auxiliary head should not be promoted without changing the objective."
        )
        next_step = "Kill or reformulate the full-vocab bag auxiliary target before spending more training runtime."
    return {
        "route": route,
        "completed": completed,
        "positive_gate_passed": positive,
        "beats_unigram": beats_unigram,
        "non_raw_recovered": non_raw_recovered,
        "interpretation": interpretation,
        "next_step": next_step,
    }


def _fmt_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and not math.isfinite(value):
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return str(value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score C3 auxiliary target top-K diagnostics.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--top-k", type=int, nargs="+", default=list(DEFAULT_TOP_KS))
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_diagnostics(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        device_name=args.device,
        top_ks=args.top_k,
        max_batches=args.max_batches,
    )
    write_summary(summary, args.summary_output)
    if args.report_output is not None:
        write_markdown_report(summary, args.report_output)
    print(
        "c3_auxiliary_diagnostics "
        f"decision={summary['decision']['route']} "
        f"eval_windows={summary['dataset']['eval_window_count']} "
        f"recall_at_{summary['comparison']['selected_k']}="
        f"{_fmt_metric(summary['comparison']['model_recall_at_selected_k'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
