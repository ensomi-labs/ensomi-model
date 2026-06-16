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
from torch.utils.data import DataLoader

from pulsefield_model.data.mapper_sparse_windows_v2_1 import collate_mapper_v2_1_windows
from pulsefield_model.evals.c3_auxiliary_diagnostics import (
    build_datasets_from_config,
    build_unigram_top_ids,
    load_model_from_checkpoint,
    token_kind_lookup,
)
from pulsefield_model.training.common import select_torch_device
from pulsefield_model.training.mapper_common import _move_mapper_batch_tensors


SUMMARY_SCHEMA_VERSION = 1
DEFAULT_TOP_K = 20


@dataclass(frozen=True)
class CalibrationTransform:
    name: str
    family: str
    kind_biases: Mapping[str, float]
    kind_scales: Mapping[str, float]


@dataclass(frozen=True)
class PredictionSample:
    sample_index: int
    logits: torch.Tensor
    target_ids: frozenset[int]


def apply_transform(
    logits: torch.Tensor,
    transform: CalibrationTransform,
    kind_index_by_name: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    transformed = logits.clone()
    for kind, indexes in kind_index_by_name.items():
        if indexes.numel() == 0:
            continue
        scale = float(transform.kind_scales.get(kind, 1.0))
        bias = float(transform.kind_biases.get(kind, 0.0))
        transformed.index_copy_(0, indexes, transformed.index_select(0, indexes) * scale + bias)
    return transformed


def score_samples(
    samples: Sequence[PredictionSample],
    *,
    transform: CalibrationTransform,
    kind_by_id: Mapping[int, str],
    kind_index_by_name: Mapping[str, torch.Tensor],
    unigram_top_ids: Sequence[int],
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    positive_sample_count = 0
    positive_label_count = 0
    hits = 0
    sample_hit_count = 0
    kind_target_counts: Counter[str] = Counter()
    kind_hit_counts: Counter[str] = Counter()
    predicted_kind_counts: Counter[str] = Counter()
    top_k = int(top_k)
    for sample in samples:
        if not sample.target_ids:
            continue
        positive_sample_count += 1
        positive_label_count += len(sample.target_ids)
        for target_id in sample.target_ids:
            kind_target_counts[kind_by_id.get(int(target_id), "UNKNOWN")] += 1
        transformed = apply_transform(sample.logits, transform, kind_index_by_name)
        top_ids = (torch.topk(transformed, k=min(top_k, int(transformed.numel()))).indices + 1).tolist()
        for predicted_id in top_ids:
            predicted_kind_counts[kind_by_id.get(int(predicted_id), "UNKNOWN")] += 1
        hit_ids = sample.target_ids.intersection(int(token_id) for token_id in top_ids)
        hits += len(hit_ids)
        if hit_ids:
            sample_hit_count += 1
        for hit_id in hit_ids:
            kind_hit_counts[kind_by_id.get(int(hit_id), "UNKNOWN")] += 1
    return _metrics_payload(
        positive_sample_count=positive_sample_count,
        positive_label_count=positive_label_count,
        hits=hits,
        sample_hit_count=sample_hit_count,
        kind_target_counts=kind_target_counts,
        kind_hit_counts=kind_hit_counts,
        predicted_kind_counts=predicted_kind_counts,
        top_k=top_k,
        extra={
            "transform": transform_to_dict(transform),
            "unigram_top_ids": [int(token_id) for token_id in unigram_top_ids[:top_k]],
        },
    )


def score_unigram_samples(
    samples: Sequence[PredictionSample],
    *,
    kind_by_id: Mapping[int, str],
    unigram_top_ids: Sequence[int],
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    positive_sample_count = 0
    positive_label_count = 0
    hits = 0
    sample_hit_count = 0
    kind_target_counts: Counter[str] = Counter()
    kind_hit_counts: Counter[str] = Counter()
    predicted_kind_counts: Counter[str] = Counter()
    predicted_ids = tuple(int(token_id) for token_id in unigram_top_ids[: int(top_k)])
    for sample in samples:
        if not sample.target_ids:
            continue
        positive_sample_count += 1
        positive_label_count += len(sample.target_ids)
        for target_id in sample.target_ids:
            kind_target_counts[kind_by_id.get(int(target_id), "UNKNOWN")] += 1
        for predicted_id in predicted_ids:
            predicted_kind_counts[kind_by_id.get(int(predicted_id), "UNKNOWN")] += 1
        hit_ids = sample.target_ids.intersection(predicted_ids)
        hits += len(hit_ids)
        if hit_ids:
            sample_hit_count += 1
        for hit_id in hit_ids:
            kind_hit_counts[kind_by_id.get(int(hit_id), "UNKNOWN")] += 1
    return _metrics_payload(
        positive_sample_count=positive_sample_count,
        positive_label_count=positive_label_count,
        hits=hits,
        sample_hit_count=sample_hit_count,
        kind_target_counts=kind_target_counts,
        kind_hit_counts=kind_hit_counts,
        predicted_kind_counts=predicted_kind_counts,
        top_k=int(top_k),
        extra={"unigram_top_ids": [int(token_id) for token_id in predicted_ids]},
    )


def candidate_transforms() -> tuple[CalibrationTransform, ...]:
    candidates: list[CalibrationTransform] = [
        CalibrationTransform(name="identity", family="identity", kind_biases={}, kind_scales={})
    ]
    for raw_bias in (-2.0, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0):
        candidates.append(
            CalibrationTransform(
                name=f"raw_bias_{raw_bias:+.1f}",
                family="raw_bias",
                kind_biases={"RAW": raw_bias},
                kind_scales={},
            )
        )
    bias_grid = (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
    res_bias_grid = (-1.0, -0.5, 0.0, 0.5, 1.0)
    for raw_bias in bias_grid:
        for res_bias in res_bias_grid:
            if raw_bias == 0.0 and res_bias == 0.0:
                continue
            candidates.append(
                CalibrationTransform(
                    name=f"kind_bias_raw{raw_bias:+.1f}_res{res_bias:+.1f}",
                    family="kind_bias",
                    kind_biases={"RAW": raw_bias, "REF": 0.0, "RES": res_bias},
                    kind_scales={},
                )
            )
    scale_grid = (0.75, 1.0, 1.25)
    raw_bias_grid = (0.0, 0.5, 1.0)
    small_res_bias_grid = (-0.5, 0.0, 0.5)
    for raw_scale in scale_grid:
        for ref_scale in scale_grid:
            for res_scale in scale_grid:
                for raw_bias in raw_bias_grid:
                    for res_bias in small_res_bias_grid:
                        if (
                            raw_scale == 1.0
                            and ref_scale == 1.0
                            and res_scale == 1.0
                            and raw_bias == 0.0
                            and res_bias == 0.0
                        ):
                            continue
                        candidates.append(
                            CalibrationTransform(
                                name=(
                                    f"scale_bias_rs{raw_scale:.2f}_fs{ref_scale:.2f}_"
                                    f"ss{res_scale:.2f}_rb{raw_bias:+.1f}_sb{res_bias:+.1f}"
                                ),
                                family="scale_bias",
                                kind_biases={"RAW": raw_bias, "REF": 0.0, "RES": res_bias},
                                kind_scales={"RAW": raw_scale, "REF": ref_scale, "RES": res_scale},
                            )
                        )
    deduped: dict[str, CalibrationTransform] = {}
    for candidate in candidates:
        deduped[candidate.name] = candidate
    return tuple(deduped.values())


def run_calibration_audit(
    *,
    config_path: Path,
    checkpoint_path: Path,
    batch_size: int,
    device_name: str,
    top_k: int = DEFAULT_TOP_K,
    max_batches: int | None = None,
) -> dict[str, Any]:
    started_at = time.monotonic()
    datasets = build_datasets_from_config(config_path)
    unigram_counts, unigram_top_ids = build_unigram_top_ids(
        datasets.train_dataset,
        datasets.token_lookup,
        limit=int(top_k),
    )
    kind_by_id = token_kind_lookup(datasets.token_by_id)
    kind_index_by_name = _kind_index_tensors(kind_by_id)
    device = select_torch_device(device_name)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device=device)
    samples = _collect_prediction_samples(
        model=model,
        eval_dataset=datasets.eval_dataset,
        batch_size=batch_size,
        device=device,
        max_batches=max_batches,
    )
    calibration_samples = tuple(sample for sample in samples if sample.sample_index % 2 == 0)
    heldout_samples = tuple(sample for sample in samples if sample.sample_index % 2 == 1)
    candidates = candidate_transforms()
    scored_candidates = []
    for candidate in candidates:
        calibration_metrics = score_samples(
            calibration_samples,
            transform=candidate,
            kind_by_id=kind_by_id,
            kind_index_by_name=kind_index_by_name,
            unigram_top_ids=unigram_top_ids,
            top_k=top_k,
        )
        scored_candidates.append(
            {
                "transform": candidate,
                "calibration": calibration_metrics,
            }
        )
    identity = candidates[0]
    identity_calibration = scored_candidates[0]["calibration"]
    unconstrained_row = max(
        scored_candidates,
        key=lambda row: (
            int(row["calibration"]["hits"]),
            int(row["calibration"]["kind_hits"].get("RAW", 0)),
            -_transform_complexity(row["transform"]),
            str(row["transform"].name),
        ),
    )
    guarded_candidates = [
        row
        for row in scored_candidates
        if _preservation_ratio(row["calibration"], identity_calibration) is None
        or float(_preservation_ratio(row["calibration"], identity_calibration) or 0.0) >= 0.90
    ]
    selected_row = max(
        guarded_candidates,
        key=lambda row: (
            int(row["calibration"]["hits"]),
            int(row["calibration"]["kind_hits"].get("RAW", 0)),
            -_transform_complexity(row["transform"]),
            str(row["transform"].name),
        ),
    )
    selected = selected_row["transform"]
    identity_heldout = score_samples(
        heldout_samples,
        transform=identity,
        kind_by_id=kind_by_id,
        kind_index_by_name=kind_index_by_name,
        unigram_top_ids=unigram_top_ids,
        top_k=top_k,
    )
    selected_heldout = score_samples(
        heldout_samples,
        transform=selected,
        kind_by_id=kind_by_id,
        kind_index_by_name=kind_index_by_name,
        unigram_top_ids=unigram_top_ids,
        top_k=top_k,
    )
    unconstrained_heldout = score_samples(
        heldout_samples,
        transform=unconstrained_row["transform"],
        kind_by_id=kind_by_id,
        kind_index_by_name=kind_index_by_name,
        unigram_top_ids=unigram_top_ids,
        top_k=top_k,
    )
    heldout_rows = [
        {
            "transform": row["transform"],
            "heldout": score_samples(
                heldout_samples,
                transform=row["transform"],
                kind_by_id=kind_by_id,
                kind_index_by_name=kind_index_by_name,
                unigram_top_ids=unigram_top_ids,
                top_k=top_k,
            ),
        }
        for row in scored_candidates
    ]
    oracle_heldout_row = max(
        heldout_rows,
        key=lambda row: (
            int(row["heldout"]["hits"]),
            int(row["heldout"]["kind_hits"].get("RAW", 0)),
            -_transform_complexity(row["transform"]),
            str(row["transform"].name),
        ),
    )
    unigram_calibration = score_unigram_samples(
        calibration_samples,
        kind_by_id=kind_by_id,
        unigram_top_ids=unigram_top_ids,
        top_k=top_k,
    )
    unigram_heldout = score_unigram_samples(
        heldout_samples,
        kind_by_id=kind_by_id,
        unigram_top_ids=unigram_top_ids,
        top_k=top_k,
    )
    top_calibration_rows = sorted(
        scored_candidates,
        key=lambda row: (
            -int(row["calibration"]["hits"]),
            -int(row["calibration"]["kind_hits"].get("RAW", 0)),
            _transform_complexity(row["transform"]),
            str(row["transform"].name),
        ),
    )[:20]
    top_heldout_rows = sorted(
        heldout_rows,
        key=lambda row: (
            -int(row["heldout"]["hits"]),
            -int(row["heldout"]["kind_hits"].get("RAW", 0)),
            _transform_complexity(row["transform"]),
            str(row["transform"].name),
        ),
    )[:20]
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "C3 kind-logit calibration audit",
        "config_path": config_path.as_posix(),
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint": {
            "run_name": checkpoint.get("run_name"),
            "completed_steps": int(checkpoint.get("training_state", {}).get("step", 0)),
            "is_complete": bool(checkpoint.get("training_state", {}).get("is_complete", False)),
            "use_c3_auxiliary_kind_heads": bool(
                checkpoint.get("model_config", {}).get("use_c3_auxiliary_kind_heads", False)
            ),
            "c3_auxiliary_kind_vocab_sizes": list(
                checkpoint.get("model_config", {}).get("c3_auxiliary_kind_vocab_sizes", []) or []
            ),
        },
        "dataset": {
            "train_window_count": len(datasets.train_dataset),
            "eval_window_count": len(datasets.eval_dataset),
            "collected_sample_count": len(samples),
            "calibration_sample_count": len(calibration_samples),
            "heldout_sample_count": len(heldout_samples),
            "batch_size": int(batch_size),
            "max_batches": max_batches,
            "split_policy": "alternating eval samples; even indexes calibrate, odd indexes held out",
        },
        "config": {
            "top_k": int(top_k),
            "candidate_count": len(candidates),
            "candidate_families": dict(sorted(Counter(candidate.family for candidate in candidates).items())),
        },
        "unigram_baseline": {
            "counted_token_count": int(sum(unigram_counts.values())),
            "unique_token_count": int(len(unigram_counts)),
            "heldout": unigram_heldout,
            "calibration": unigram_calibration,
            "top_tokens": [
                {
                    "token_id": int(token_id),
                    "token": datasets.token_by_id.get(int(token_id), ""),
                    "kind": kind_by_id.get(int(token_id), "UNKNOWN"),
                    "count": int(unigram_counts.get(int(token_id), 0)),
                }
                for token_id in unigram_top_ids[: int(top_k)]
            ],
        },
        "identity": {
            "transform": transform_to_dict(identity),
            "calibration": identity_calibration,
            "heldout": identity_heldout,
        },
        "selected": {
            "transform": transform_to_dict(selected),
            "calibration": selected_row["calibration"],
            "heldout": selected_heldout,
            "selection_policy": "best calibration hits with non-RAW preservation >= 0.90 versus identity",
        },
        "unconstrained_best_on_calibration": {
            "transform": transform_to_dict(unconstrained_row["transform"]),
            "calibration": unconstrained_row["calibration"],
            "heldout": unconstrained_heldout,
            "diagnostic_only": True,
        },
        "oracle_best_on_heldout": {
            "transform": transform_to_dict(oracle_heldout_row["transform"]),
            "heldout": oracle_heldout_row["heldout"],
            "diagnostic_only": True,
        },
        "top_calibration_candidates": [
            {
                "transform": transform_to_dict(row["transform"]),
                "calibration": _compact_metrics(row["calibration"]),
            }
            for row in top_calibration_rows
        ],
        "top_heldout_candidates": [
            {
                "transform": transform_to_dict(row["transform"]),
                "heldout": _compact_metrics(row["heldout"]),
            }
            for row in top_heldout_rows
        ],
        "elapsed_s": time.monotonic() - started_at,
    }
    summary["decision"] = _decision(summary)
    return _normalize_json(summary)


def transform_to_dict(transform: CalibrationTransform) -> dict[str, Any]:
    return {
        "name": transform.name,
        "family": transform.family,
        "kind_biases": {kind: float(value) for kind, value in sorted(transform.kind_biases.items())},
        "kind_scales": {kind: float(value) for kind, value in sorted(transform.kind_scales.items())},
    }


def write_summary(summary: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown_report(summary: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    decision = summary["decision"]
    identity = summary["identity"]["heldout"]
    selected = summary["selected"]["heldout"]
    unconstrained = summary["unconstrained_best_on_calibration"]["heldout"]
    unigram = summary["unigram_baseline"]["heldout"]
    oracle = summary["oracle_best_on_heldout"]["heldout"]
    text = "\n".join(
        [
            "# C3 Kind-Logit Calibration Result Report",
            "",
            "## Scope",
            "",
            "This P21 pass tests whether P20's RAW gap can be explained by post-hoc kind-logit calibration. It reuses the P19 checkpoint and reduced sidecar without retraining.",
            "",
            "## Result",
            "",
            f"Decision: {decision['route']}.",
            "",
            f"- candidate transforms: `{summary['config']['candidate_count']}`",
            f"- calibration samples: `{summary['dataset']['calibration_sample_count']}`",
            f"- held-out samples: `{summary['dataset']['heldout_sample_count']}`",
            f"- selected transform: `{summary['selected']['transform']['name']}`",
            f"- held-out identity hits@20: `{identity['hits']}` / `{identity['positive_label_count']}`",
            f"- held-out selected hits@20: `{selected['hits']}` / `{selected['positive_label_count']}`",
            f"- held-out unconstrained-calibration hits@20: `{unconstrained['hits']}` / `{unconstrained['positive_label_count']}`",
            f"- held-out unigram hits@20: `{unigram['hits']}` / `{unigram['positive_label_count']}`",
            f"- held-out oracle-calibration hits@20: `{oracle['hits']}` / `{oracle['positive_label_count']}`",
            f"- held-out identity RAW recall@20: `{_fmt_metric(identity['kind_recall'].get('RAW'))}`",
            f"- held-out selected RAW recall@20: `{_fmt_metric(selected['kind_recall'].get('RAW'))}`",
            f"- held-out unigram RAW recall@20: `{_fmt_metric(unigram['kind_recall'].get('RAW'))}`",
            f"- RAW gap closure: `{_fmt_metric(decision['raw_gap_closure'])}`",
            f"- non-RAW preservation ratio: `{_fmt_metric(decision['non_raw_preservation_ratio'])}`",
            f"- elapsed: `{float(summary['elapsed_s']):.2f}s`",
            "",
            "## Held-Out Comparison",
            "",
            "| System | Hits | Recall | RAW recall | REF recall | RES recall | Pred RAW | Pred REF | Pred RES |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            _comparison_row("identity", identity),
            _comparison_row("selected", selected),
            _comparison_row("unconstrained calibration", unconstrained),
            _comparison_row("unigram", unigram),
            _comparison_row("oracle heldout", oracle),
            "",
            "## Selected Transform",
            "",
            f"- family: `{summary['selected']['transform']['family']}`",
            f"- kind biases: `{summary['selected']['transform']['kind_biases']}`",
            f"- kind scales: `{summary['selected']['transform']['kind_scales']}`",
            f"- selection policy: `{summary['selected']['selection_policy']}`",
            "",
            "## Unconstrained Diagnostic Transform",
            "",
            f"- transform: `{summary['unconstrained_best_on_calibration']['transform']['name']}`",
            f"- family: `{summary['unconstrained_best_on_calibration']['transform']['family']}`",
            f"- kind biases: `{summary['unconstrained_best_on_calibration']['transform']['kind_biases']}`",
            f"- kind scales: `{summary['unconstrained_best_on_calibration']['transform']['kind_scales']}`",
            "",
            "## Top Calibration Candidates",
            "",
            "| Candidate | Family | Calibration hits | Calibration RAW recall |",
            "| --- | --- | ---: | ---: |",
            *_candidate_rows(summary["top_calibration_candidates"], partition="calibration"),
            "",
            "## Top Held-Out Candidates",
            "",
            "| Candidate | Family | Held-out hits | Held-out RAW recall |",
            "| --- | --- | ---: | ---: |",
            *_candidate_rows(summary["top_heldout_candidates"], partition="heldout"),
            "",
            "## What Passed",
            "",
            "- The P19 checkpoint, config, reduced sidecar, and eval split are loadable.",
            "- Calibration is post-hoc only; no target-derived C3 labels are used as model inputs.",
            "- Identity, selected calibration, held-out oracle, and reduced unigram are all reported separately.",
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


def _collect_prediction_samples(
    *,
    model: Any,
    eval_dataset: Any,
    batch_size: int,
    device: torch.device,
    max_batches: int | None,
) -> tuple[PredictionSample, ...]:
    loader = DataLoader(
        eval_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_mapper_v2_1_windows,
    )
    samples: list[PredictionSample] = []
    model.eval()
    sample_index = 0
    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = _move_mapper_batch_tensors(raw_batch, device)
            output = model(batch)
            logits = output.c3_auxiliary_logits
            if not isinstance(logits, torch.Tensor):
                raise ValueError("checkpoint model did not produce c3_auxiliary_logits")
            token_rows, mask_rows, available_rows = _batch_token_rows(batch)
            for row_index in range(int(logits.shape[0])):
                target_ids = frozenset(
                    token_id
                    for token_id, is_valid in zip(token_rows[row_index], mask_rows[row_index])
                    if available_rows[row_index] and is_valid and token_id > 0
                )
                samples.append(
                    PredictionSample(
                        sample_index=sample_index,
                        logits=logits[row_index].detach().cpu(),
                        target_ids=target_ids,
                    )
                )
                sample_index += 1
    return tuple(samples)


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


def _kind_index_tensors(kind_by_id: Mapping[int, str]) -> dict[str, torch.Tensor]:
    by_kind: dict[str, list[int]] = {"RAW": [], "REF": [], "RES": []}
    for token_id, kind in kind_by_id.items():
        if kind in by_kind:
            by_kind[kind].append(int(token_id) - 1)
    return {
        kind: torch.tensor(sorted(indexes), dtype=torch.long)
        for kind, indexes in by_kind.items()
    }


def _metrics_payload(
    *,
    positive_sample_count: int,
    positive_label_count: int,
    hits: int,
    sample_hit_count: int,
    kind_target_counts: Counter[str],
    kind_hit_counts: Counter[str],
    predicted_kind_counts: Counter[str],
    top_k: int,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    kind_recall = {
        kind: _safe_divide(kind_hit_counts.get(kind, 0), target_count)
        for kind, target_count in sorted(kind_target_counts.items())
    }
    payload = {
        "top_k": int(top_k),
        "positive_sample_count": int(positive_sample_count),
        "positive_label_count": int(positive_label_count),
        "hits": int(hits),
        "recall": _safe_divide(hits, positive_label_count),
        "precision": _safe_divide(hits, positive_sample_count * int(top_k)),
        "sample_hit_count": int(sample_hit_count),
        "sample_hit_rate": _safe_divide(sample_hit_count, positive_sample_count),
        "kind_target_counts": dict(sorted(kind_target_counts.items())),
        "kind_hits": dict(sorted(kind_hit_counts.items())),
        "kind_recall": kind_recall,
        "predicted_kind_counts": dict(sorted(predicted_kind_counts.items())),
        "non_raw_recall": _non_raw_recall(kind_target_counts, kind_hit_counts),
    }
    payload.update(dict(extra))
    return payload


def _non_raw_recall(kind_target_counts: Mapping[str, int], kind_hit_counts: Mapping[str, int]) -> float | None:
    denominator = sum(int(count) for kind, count in kind_target_counts.items() if kind != "RAW")
    numerator = sum(int(kind_hit_counts.get(kind, 0)) for kind in kind_target_counts if kind != "RAW")
    return _safe_divide(numerator, denominator)


def _decision(summary: Mapping[str, Any]) -> dict[str, Any]:
    identity = summary["identity"]["heldout"]
    selected = summary["selected"]["heldout"]
    unigram = summary["unigram_baseline"]["heldout"]
    oracle = summary["oracle_best_on_heldout"]["heldout"]
    unconstrained = summary["unconstrained_best_on_calibration"]["heldout"]
    hit_delta = int(selected["hits"]) - int(identity["hits"])
    unconstrained_hit_delta = int(unconstrained["hits"]) - int(identity["hits"])
    raw_identity = identity["kind_recall"].get("RAW")
    raw_selected = selected["kind_recall"].get("RAW")
    raw_unigram = unigram["kind_recall"].get("RAW")
    raw_gap_closure = _gap_closure(raw_identity, raw_selected, raw_unigram)
    non_raw_identity = identity.get("non_raw_recall")
    non_raw_selected = selected.get("non_raw_recall")
    non_raw_preservation = (
        None
        if non_raw_identity is None or non_raw_identity == 0.0 or non_raw_selected is None
        else float(non_raw_selected) / float(non_raw_identity)
    )
    positive = (
        (hit_delta >= 5 or (raw_gap_closure is not None and raw_gap_closure >= 0.25))
        and (non_raw_preservation is None or non_raw_preservation >= 0.90)
    )
    oracle_hit_delta = int(oracle["hits"]) - int(identity["hits"])
    oracle_non_raw_preservation = _preservation_ratio(oracle, identity)
    if positive:
        route = "TEST_NEXT"
        interpretation = (
            "A simple selected post-hoc calibration improves held-out recovery enough to justify a bounded train-time "
            "calibration/ranking experiment for C3 kind heads."
        )
        next_step = "Create a calibrated C3 kind-head training card."
    elif oracle_hit_delta >= 5 and oracle_non_raw_preservation is not None and oracle_non_raw_preservation >= 0.90:
        route = "MUTATE"
        interpretation = (
            "The selected calibration did not satisfy the held-out positive gate, but the held-out oracle sweep has "
            "some headroom. The current grid or split may be unstable; do not train this transform directly."
        )
        next_step = "Either repeat calibration on a larger slice or move to a RAW split if the oracle transform is extreme."
    elif unconstrained_hit_delta >= 5:
        route = "MUTATE_TO_RAW_SPLIT"
        interpretation = (
            "Unconstrained calibration can buy held-out hits only by overpromoting RAW and damaging non-RAW recovery. "
            "That is not a viable calibration path for the full C3 target; the next mutation should factor RAW rather "
            "than globally bias kind logits."
        )
        next_step = "Create a RAW split/factorization Experiment Card based on the P20 field buckets."
    else:
        route = "MUTATE_TO_RAW_SPLIT"
        interpretation = (
            "The calibration sweep does not show enough held-out headroom. This weakens the kind-logit calibration "
            "hypothesis and points back to P20's field-specific RAW split/factorization."
        )
        next_step = "Create a RAW split/factorization Experiment Card based on the P20 field buckets."
    return {
        "route": route,
        "heldout_hit_delta_vs_identity": hit_delta,
        "unconstrained_heldout_hit_delta_vs_identity": unconstrained_hit_delta,
        "oracle_heldout_hit_delta_vs_identity": oracle_hit_delta,
        "raw_gap_closure": raw_gap_closure,
        "non_raw_preservation_ratio": non_raw_preservation,
        "oracle_non_raw_preservation_ratio": oracle_non_raw_preservation,
        "positive_signal_observed": positive,
        "interpretation": interpretation,
        "next_step": next_step,
    }


def _gap_closure(identity: float | None, selected: float | None, comparator: float | None) -> float | None:
    if identity is None or selected is None or comparator is None:
        return None
    denominator = float(comparator) - float(identity)
    if denominator <= 0.0:
        return None
    return (float(selected) - float(identity)) / denominator


def _preservation_ratio(candidate: Mapping[str, Any], identity: Mapping[str, Any]) -> float | None:
    identity_non_raw = identity.get("non_raw_recall")
    candidate_non_raw = candidate.get("non_raw_recall")
    if identity_non_raw is None or candidate_non_raw is None or float(identity_non_raw) == 0.0:
        return None
    return float(candidate_non_raw) / float(identity_non_raw)


def _transform_complexity(transform: CalibrationTransform) -> float:
    bias_complexity = sum(abs(float(value)) for value in transform.kind_biases.values())
    scale_complexity = sum(abs(float(value) - 1.0) for value in transform.kind_scales.values())
    family_penalty = {"identity": 0.0, "raw_bias": 0.1, "kind_bias": 0.2, "scale_bias": 0.3}.get(
        transform.family,
        1.0,
    )
    return bias_complexity + scale_complexity + family_penalty


def _compact_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "hits": metrics.get("hits"),
        "recall": metrics.get("recall"),
        "kind_hits": metrics.get("kind_hits"),
        "kind_recall": metrics.get("kind_recall"),
        "predicted_kind_counts": metrics.get("predicted_kind_counts"),
        "non_raw_recall": metrics.get("non_raw_recall"),
    }


def _comparison_row(name: str, metrics: Mapping[str, Any]) -> str:
    predicted = metrics.get("predicted_kind_counts", {})
    kind_recall = metrics.get("kind_recall", {})
    return (
        f"| {name} | {metrics.get('hits')} | {_fmt_metric(metrics.get('recall'))} | "
        f"{_fmt_metric(kind_recall.get('RAW'))} | {_fmt_metric(kind_recall.get('REF'))} | "
        f"{_fmt_metric(kind_recall.get('RES'))} | {predicted.get('RAW', 0)} | "
        f"{predicted.get('REF', 0)} | {predicted.get('RES', 0)} |"
    )


def _candidate_rows(rows: Sequence[Mapping[str, Any]], *, partition: str) -> list[str]:
    rendered = []
    for row in rows[:10]:
        transform = row["transform"]
        metrics = row[partition]
        rendered.append(
            f"| `{transform['name']}` | {transform['family']} | {metrics['hits']} | "
            f"{_fmt_metric(metrics['kind_recall'].get('RAW'))} |"
        )
    return rendered


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


def _normalize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a post-hoc C3 kind-logit calibration audit.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_calibration_audit(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        device_name=args.device,
        top_k=args.top_k,
        max_batches=args.max_batches,
    )
    write_summary(summary, args.summary_output)
    if args.report_output is not None:
        write_markdown_report(summary, args.report_output)
    print(
        "c3_kind_logit_calibration "
        f"decision={summary['decision']['route']} "
        f"selected={summary['selected']['transform']['name']} "
        f"heldout_hits={summary['selected']['heldout']['hits']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
