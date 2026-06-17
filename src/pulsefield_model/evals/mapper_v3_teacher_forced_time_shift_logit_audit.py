from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from pulsefield_model.data.control_windows import DEFAULT_MAX_CACHED_MAPS
from pulsefield_model.data.mapper_sparse_windows_v3 import (
    MapperV3WindowDataset,
    collate_mapper_v3_windows,
)
from pulsefield_model.models.control import ControlDemoGlobalEncoder, ControlDemoGlobalEncoderConfig
from pulsefield_model.models.mapper.v3 import MapperV3Config, MapperV3Model, MapperV3Vocab
from pulsefield_model.training.common import split_train_eval_dataset
from pulsefield_model.training.mapper_common import _move_mapper_batch_tensors


SUMMARY_SCHEMA_VERSION = 1
DEFAULT_REPORT_ROOT = Path("artifacts/reports/audits/mapper_v2_1_grammar")
DEFAULT_SUMMARY_OUTPUT = DEFAULT_REPORT_ROOT / "target_grammar_v3_teacher_forced_time_shift_logit_audit_summary.json"
DEFAULT_REPORT_OUTPUT = DEFAULT_REPORT_ROOT / "target_grammar_v3_teacher_forced_time_shift_logit_audit_result_report.md"
EXPECTED_V3_CONTRACT = "v3_event_groups"
RIGID_PROXY_SHIFT_VALUES = (60, 100, 200)


@dataclass(frozen=True)
class LoadedV3Checkpoint:
    model: MapperV3Model
    checkpoint: Mapping[str, Any]


class TimeShiftLogitAccumulator:
    def __init__(self, *, vocab: MapperV3Vocab, top_ks: Sequence[int] = (1, 3, 5), max_examples: int = 16) -> None:
        self.vocab = vocab
        self.top_ks = tuple(sorted({int(k) for k in top_ks if int(k) > 0}))
        if not self.top_ks:
            raise ValueError("top_ks must contain at least one positive value")
        self.max_examples = int(max_examples)
        if self.max_examples < 0:
            raise ValueError("max_examples must be non-negative")
        self.time_shift_ids = tuple(int(token_id) for token_id in vocab.time_shift_token_ids)
        self.time_shift_values = tuple(int(value) for value in vocab.time_shift_values_ms)
        self.shift_index_by_token = {token_id: index for index, token_id in enumerate(self.time_shift_ids)}
        self.token_to_shift_index = torch.full((vocab.size,), -1, dtype=torch.long)
        for index, token_id in enumerate(self.time_shift_ids):
            self.token_to_shift_index[token_id] = index

        self.window_count = 0
        self.scored_token_count = 0
        self.time_shift_row_count = 0
        self.rank_sum = 0.0
        self.nll_sum = 0.0
        self.prob_sum = 0.0
        self.ranks: list[int] = []
        self.hits_by_k: Counter[int] = Counter()
        self.target_shift_counts: Counter[int] = Counter()
        self.argmax_shift_counts: Counter[int] = Counter()
        self.examples: list[dict[str, Any]] = []

    def update(
        self,
        *,
        logits_final: torch.Tensor,
        target_tokens: torch.Tensor,
        target_mask: torch.Tensor,
        metadata: Sequence[Mapping[str, Any]] = (),
        current_ms: torch.Tensor | None = None,
    ) -> None:
        if logits_final.ndim != 3:
            raise ValueError(f"logits_final must have shape [B,T,V], got {tuple(logits_final.shape)}")
        if target_tokens.ndim != 2 or target_mask.ndim != 2:
            raise ValueError("target_tokens and target_mask must have shape [B,T]")
        if tuple(logits_final.shape[:2]) != tuple(target_tokens.shape) or tuple(target_mask.shape) != tuple(target_tokens.shape):
            raise ValueError("logits, targets, and mask shapes must agree")
        if int(logits_final.shape[-1]) != int(self.vocab.size):
            raise ValueError(f"logits vocab size must be {self.vocab.size}, got {logits_final.shape[-1]}")

        device = logits_final.device
        token_to_shift_index = self.token_to_shift_index.to(device=device)
        target_shift_index = token_to_shift_index.index_select(0, target_tokens.reshape(-1)).reshape_as(target_tokens)
        scoring_mask = target_mask.to(device=device, dtype=torch.bool) & (target_shift_index >= 0)
        self.window_count += int(target_tokens.shape[0])
        self.scored_token_count += int(target_mask.to(dtype=torch.bool).sum().detach().cpu())
        row_count = int(scoring_mask.sum().detach().cpu())
        if row_count == 0:
            return

        ts_logits = logits_final.index_select(
            -1,
            torch.tensor(self.time_shift_ids, dtype=torch.long, device=device),
        )
        selected_logits = ts_logits[scoring_mask]
        selected_targets = target_shift_index[scoring_mask].to(dtype=torch.long)
        target_logits = selected_logits.gather(1, selected_targets.reshape(-1, 1)).reshape(-1)
        ranks = (selected_logits > target_logits.reshape(-1, 1)).sum(dim=1).to(dtype=torch.long) + 1
        log_probs = torch.log_softmax(selected_logits, dim=1)
        target_log_probs = log_probs.gather(1, selected_targets.reshape(-1, 1)).reshape(-1)
        target_probs = torch.exp(target_log_probs)
        argmax_indices = selected_logits.argmax(dim=1)

        self.time_shift_row_count += row_count
        self.rank_sum += float(ranks.to(dtype=torch.float32).sum().detach().cpu())
        self.nll_sum += float((-target_log_probs).sum().detach().cpu())
        self.prob_sum += float(target_probs.sum().detach().cpu())
        for rank in ranks.detach().cpu().tolist():
            rank_i = int(rank)
            self.ranks.append(rank_i)
            for k in self.top_ks:
                if rank_i <= k:
                    self.hits_by_k[k] += 1
        for index in selected_targets.detach().cpu().tolist():
            self.target_shift_counts[int(self.time_shift_values[int(index)])] += 1
        for index in argmax_indices.detach().cpu().tolist():
            self.argmax_shift_counts[int(self.time_shift_values[int(index)])] += 1

        if len(self.examples) < self.max_examples:
            self._add_examples(
                selected_logits=selected_logits.detach().cpu(),
                selected_targets=selected_targets.detach().cpu(),
                ranks=ranks.detach().cpu(),
                scoring_mask=scoring_mask.detach().cpu(),
                metadata=metadata,
                current_ms=None if current_ms is None else current_ms.detach().cpu(),
            )

    def to_dict(self) -> dict[str, Any]:
        target_distribution = _normalized_counter(self.target_shift_counts)
        argmax_distribution = _normalized_counter(self.argmax_shift_counts)
        top_shift = self.argmax_shift_counts.most_common(1)
        top_argmax_share = 0.0 if not top_shift or self.time_shift_row_count == 0 else top_shift[0][1] / self.time_shift_row_count
        return {
            "window_count": int(self.window_count),
            "scored_token_count": int(self.scored_token_count),
            "time_shift_row_count": int(self.time_shift_row_count),
            "time_shift_vocab_size": len(self.time_shift_ids),
            "time_shift_values_ms": list(self.time_shift_values),
            "time_shift_vocab_has_160": 160 in self.time_shift_values,
            "rank": _rank_summary(self.ranks),
            "recall_at_k": {
                str(k): _safe_ratio(self.hits_by_k[k], self.time_shift_row_count)
                for k in self.top_ks
            },
            "target_time_shift_nll": _safe_ratio(self.nll_sum, self.time_shift_row_count),
            "target_time_shift_prob_mean": _safe_ratio(self.prob_sum, self.time_shift_row_count),
            "target_shift_counts": dict(sorted(self.target_shift_counts.items())),
            "argmax_shift_counts": dict(sorted(self.argmax_shift_counts.items())),
            "target_shift_distribution": target_distribution,
            "argmax_shift_distribution": argmax_distribution,
            "argmax_top_shift_ms": None if not top_shift else int(top_shift[0][0]),
            "argmax_top_shift_share": top_argmax_share,
            "argmax_200ms_share": _counter_share(self.argmax_shift_counts, (200,), self.time_shift_row_count),
            "argmax_rigid_proxy_piece_share_60_100_200": _counter_share(
                self.argmax_shift_counts,
                RIGID_PROXY_SHIFT_VALUES,
                self.time_shift_row_count,
            ),
            "target_200ms_share": _counter_share(self.target_shift_counts, (200,), self.time_shift_row_count),
            "target_rigid_proxy_piece_share_60_100_200": _counter_share(
                self.target_shift_counts,
                RIGID_PROXY_SHIFT_VALUES,
                self.time_shift_row_count,
            ),
            "js_divergence_target_vs_argmax": jensen_shannon_divergence(target_distribution, argmax_distribution),
            "examples": list(self.examples),
        }

    def _add_examples(
        self,
        *,
        selected_logits: torch.Tensor,
        selected_targets: torch.Tensor,
        ranks: torch.Tensor,
        scoring_mask: torch.Tensor,
        metadata: Sequence[Mapping[str, Any]],
        current_ms: torch.Tensor | None,
    ) -> None:
        row_indices = torch.nonzero(scoring_mask, as_tuple=False)
        remaining = self.max_examples - len(self.examples)
        if remaining <= 0:
            return
        top_k = min(5, len(self.time_shift_values))
        worst_order = torch.argsort(ranks, descending=True).tolist()
        for selected_row in worst_order:
            if remaining <= 0:
                break
            batch_index, step_index = [int(value) for value in row_indices[int(selected_row)].tolist()]
            logits = selected_logits[int(selected_row)]
            target_index = int(selected_targets[int(selected_row)].item())
            top_indices = torch.topk(logits, k=top_k).indices.tolist()
            metadata_row = metadata[batch_index] if batch_index < len(metadata) and isinstance(metadata[batch_index], Mapping) else {}
            self.examples.append(
                {
                    "beatmap_path": metadata_row.get("beatmap_path"),
                    "window_start_ms": metadata_row.get("target_start_ms"),
                    "step_index": step_index,
                    "current_ms": None if current_ms is None else int(current_ms[batch_index, step_index].item()),
                    "target_shift_ms": int(self.time_shift_values[target_index]),
                    "target_rank": int(ranks[int(selected_row)].item()),
                    "top_shift_ms": [int(self.time_shift_values[int(index)]) for index in top_indices],
                }
            )
            remaining -= 1


def run_teacher_forced_time_shift_logit_audit(
    *,
    checkpoint_path: Path,
    training_report_path: Path,
    summary_output: Path = DEFAULT_SUMMARY_OUTPUT,
    report_output: Path | None = DEFAULT_REPORT_OUTPUT,
    device_name: str = "cpu",
    batch_size: int = 2,
    max_windows: int | None = None,
    disable_global_context_for_smoke: bool = False,
) -> dict[str, Any]:
    start = time.monotonic()
    training_report = _load_json_object(training_report_path)
    loaded = load_v3_checkpoint(
        checkpoint_path,
        device_name=device_name,
        disable_global_context_for_smoke=disable_global_context_for_smoke,
    )
    dataset = build_eval_dataset_from_report(
        training_report,
        max_windows=max_windows,
        include_full_song_context_override=False if disable_global_context_for_smoke else None,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_mapper_v3_windows,
    )
    device = torch.device(device_name)
    loaded.model.eval()
    accumulator = TimeShiftLogitAccumulator(vocab=loaded.model.vocab)
    with torch.inference_mode():
        for raw_batch in loader:
            batch = _move_mapper_batch_tensors(raw_batch, device)
            output = loaded.model(batch)
            accumulator.update(
                logits_final=output.logits_final,
                target_tokens=batch["target_fragment_tokens"],
                target_mask=batch["target_fragment_mask"],
                metadata=raw_batch.get("metadata", ()),
                current_ms=batch["target_fragment_states"]["current_ms"],
            )

    metrics = accumulator.to_dict()
    checks = {
        "checkpoint_exists": checkpoint_path.exists(),
        "training_report_exists": training_report_path.exists(),
        "checkpoint_model_v3": isinstance(loaded.model, MapperV3Model),
        "training_report_contract_v3": _training_report_contract(training_report) == EXPECTED_V3_CONTRACT,
        "dataset_non_empty": len(dataset) > 0,
        "time_shift_rows_positive": int(metrics["time_shift_row_count"]) > 0,
        "rank_metrics_finite": math.isfinite(float(metrics["target_time_shift_nll"]))
        and math.isfinite(float(metrics["target_time_shift_prob_mean"])),
        "faithful_global_context": not disable_global_context_for_smoke,
        "no_training_or_rollout": True,
    }
    decision = decision_from_metrics(metrics, checks=checks)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar v3 teacher-forced time-shift logit audit",
        "elapsed_s": time.monotonic() - start,
        "config": {
            "checkpoint_path": checkpoint_path.as_posix(),
            "training_report_path": training_report_path.as_posix(),
            "device": device_name,
            "batch_size": int(batch_size),
            "max_windows": max_windows,
            "disable_global_context_for_smoke": bool(disable_global_context_for_smoke),
        },
        "dataset": {
            "eval_window_count": len(dataset),
            "source_window_count": _source_window_count_from_report(training_report),
            "reported_eval_window_count": _reported_eval_window_count(training_report),
        },
        "checks": checks,
        "metrics": metrics,
        "decision": decision,
    }
    write_summary_json(summary, summary_output)
    if report_output is not None:
        write_report(summary, report_output)
    return summary


def load_v3_checkpoint(
    path: Path,
    *,
    device_name: str = "cpu",
    disable_global_context_for_smoke: bool = False,
) -> LoadedV3Checkpoint:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError(f"checkpoint could not be loaded safely with weights_only=True: {path}") from exc
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint must contain a mapping: {path}")
    model_config_raw = checkpoint.get("model_config")
    if not isinstance(model_config_raw, Mapping):
        raise ValueError("checkpoint missing model_config")
    control_config_raw = checkpoint.get("control_model_config")
    control_encoder = None
    if isinstance(control_config_raw, Mapping):
        control_encoder = ControlDemoGlobalEncoder(ControlDemoGlobalEncoderConfig(**dict(control_config_raw)))
    model = MapperV3Model(MapperV3Config(**dict(model_config_raw)), control_encoder=control_encoder)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint missing model_state_dict")
    model.load_state_dict(state)
    if disable_global_context_for_smoke:
        model.config = replace(model.config, use_global_context=False)
    model.to(torch.device(device_name))
    return LoadedV3Checkpoint(model=model, checkpoint=checkpoint)


def build_eval_dataset_from_report(
    report: Mapping[str, Any],
    *,
    max_windows: int | None = None,
    include_full_song_context_override: bool | None = None,
) -> Dataset[Any]:
    training_config = _mapping(report.get("training_config"))
    dataset_report = _mapping(training_config.get("dataset")) or _mapping(report.get("dataset"))
    mapper_record_cache_path = _required_path(dataset_report, "mapper_record_cache_path")
    record_cache_validity = _record_cache_validity_metadata(mapper_record_cache_path)
    dataset_root = (
        _optional_path(dataset_report.get("dataset_root"))
        or _optional_path(record_cache_validity.get("dataset_root"))
        or Path("dataset")
    )
    index_path = (
        _optional_path(dataset_report.get("index_path"))
        or _optional_path(dataset_report.get("source_index_path"))
        or _optional_path(record_cache_validity.get("source_index_path"))
    )
    control_teacher_cache_dir = _optional_path(dataset_report.get("control_teacher_cache_dir"))
    dataset_kwargs: dict[str, Any] = {}
    if index_path is not None:
        dataset_kwargs["index_path"] = index_path
    dataset = MapperV3WindowDataset(
        dataset_root=dataset_root,
        mapper_record_cache_path=mapper_record_cache_path,
        control_teacher_cache_dir=control_teacher_cache_dir,
        require_control_teacher_cache=bool(dataset_report.get("require_control_teacher_cache", False)),
        include_full_song_context=(
            bool(dataset_report.get("include_full_song_context", True))
            if include_full_song_context_override is None
            else bool(include_full_song_context_override)
        ),
        max_cached_maps=DEFAULT_MAX_CACHED_MAPS,
        progress=False,
        **dataset_kwargs,
    )
    _, eval_dataset = split_train_eval_dataset(
        dataset,
        eval_fraction=float(dataset_report.get("eval_fraction", 0.1)),
        eval_size=_optional_int(dataset_report.get("eval_size")),
        seed=int(training_config.get("seed", report.get("seed", 1337))),
    )
    if max_windows is not None:
        limit = max(0, min(int(max_windows), len(eval_dataset)))
        eval_dataset = Subset(eval_dataset, list(range(limit)))
    return eval_dataset


def decision_from_metrics(metrics: Mapping[str, Any], *, checks: Mapping[str, bool]) -> dict[str, Any]:
    failed_checks = [key for key, passed in checks.items() if not bool(passed)]
    if failed_checks:
        return {
            "route": "KILL_AUDIT_INPUTS",
            "reason": "failed checks: " + ", ".join(failed_checks),
            "interpretation": "The teacher-forced timing audit did not produce trustworthy input metrics.",
            "next_step": "Repair checkpoint/data loading or target-row masking before timing calibration work.",
        }
    recall = _mapping(metrics.get("recall_at_k"))
    recall_at_5 = float(recall.get("5", 0.0))
    top_shift_share = float(metrics.get("argmax_top_shift_share", 0.0))
    rigid_piece_share = float(metrics.get("argmax_rigid_proxy_piece_share_60_100_200", 0.0))
    median_rank = _optional_float(_mapping(metrics.get("rank")).get("median"))
    if recall_at_5 < 0.60 and (top_shift_share >= 0.40 or rigid_piece_share >= 0.60):
        return {
            "route": "MUTATE_TIMING_EMBEDDING_OR_LOSS",
            "reason": "teacher-forced time-shift recall is weak and predicted shift concentration is high",
            "interpretation": "The timing collapse is already visible under teacher forcing, so another rollout-only decode tweak is not the next best test.",
            "next_step": "Create a timing embedding/logit calibration or non-expected-shift timing objective card.",
        }
    if recall_at_5 >= 0.80 and (median_rank is not None and median_rank <= 3):
        return {
            "route": "TEST_DECODE_STATE_EXPOSURE_DIAGNOSTIC",
            "reason": "teacher-forced time-shift ranks are good enough that free-running state exposure is the likely bottleneck",
            "interpretation": "The model can rank target shifts under teacher forcing, but generated state still collapses.",
            "next_step": "Create a generated-state exposure diagnostic on high-leverage fixed-slice cases.",
        }
    return {
        "route": "MUTATE_TIMING_CALIBRATION_DIAGNOSTIC",
        "reason": "teacher-forced timing signal is mixed; neither clean logits nor obvious collapse is proven",
        "interpretation": "Use the rank/concentration buckets to design a narrower timing calibration mutation.",
        "next_step": "Inspect poor-rank examples and target/predicted shift distribution mismatch before training.",
    }


def write_summary_json(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    metrics = _mapping(summary.get("metrics"))
    rank = _mapping(metrics.get("rank"))
    recall = _mapping(metrics.get("recall_at_k"))
    decision = _mapping(summary.get("decision"))
    checks = _mapping(summary.get("checks"))
    lines = [
        "# Target Grammar v3 Teacher-Forced Time-Shift Logit Audit Result Report",
        "",
        "## Scope",
        "",
        "This artifact-only audit scores target time-shift rows under teacher forcing using the existing 500-step v3 checkpoint. It does not train, rerun rollout, change tokenizer behavior, or change mapper defaults.",
        "",
        "## Decision",
        "",
        f"Route: `{decision.get('route')}`.",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Next step: {decision.get('next_step')}",
        "",
        "## Checks",
        "",
        "| Check | Passed |",
        "| --- | ---: |",
    ]
    for key, passed in checks.items():
        lines.append(f"| `{key}` | `{bool(passed)}` |")
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            f"- eval windows: `{_mapping(summary.get('dataset')).get('eval_window_count')}`",
            f"- global context disabled for smoke: `{_mapping(summary.get('config')).get('disable_global_context_for_smoke')}`",
            f"- target time-shift rows: `{metrics.get('time_shift_row_count')}`",
            f"- recall@1 / @3 / @5: `{_fmt(recall.get('1'))}` / `{_fmt(recall.get('3'))}` / `{_fmt(recall.get('5'))}`",
            f"- median / p90 target rank: `{_fmt(rank.get('median'))}` / `{_fmt(rank.get('p90'))}`",
            f"- target time-shift NLL: `{_fmt(metrics.get('target_time_shift_nll'))}`",
            f"- mean target time-shift probability: `{_fmt(metrics.get('target_time_shift_prob_mean'))}`",
            f"- argmax top shift: `{metrics.get('argmax_top_shift_ms')}` ms at share `{_fmt(metrics.get('argmax_top_shift_share'))}`",
            f"- argmax 200ms share: `{_fmt(metrics.get('argmax_200ms_share'))}`",
            f"- argmax rigid-proxy piece share (`60/100/200`): `{_fmt(metrics.get('argmax_rigid_proxy_piece_share_60_100_200'))}`",
            f"- target rigid-proxy piece share (`60/100/200`): `{_fmt(metrics.get('target_rigid_proxy_piece_share_60_100_200'))}`",
            f"- time-shift vocab has 160ms token: `{metrics.get('time_shift_vocab_has_160')}`",
            f"- JS divergence target-vs-argmax: `{_fmt(metrics.get('js_divergence_target_vs_argmax'))}`",
            "",
            "## Top Target Shifts",
            "",
            "| Shift ms | Count |",
            "| ---: | ---: |",
        ]
    )
    for shift, count in _top_counter_rows(_mapping(metrics.get("target_shift_counts")), limit=12):
        lines.append(f"| `{shift}` | `{count}` |")
    lines.extend(["", "## Top Predicted Argmax Shifts", "", "| Shift ms | Count |", "| ---: | ---: |"])
    for shift, count in _top_counter_rows(_mapping(metrics.get("argmax_shift_counts")), limit=12):
        lines.append(f"| `{shift}` | `{count}` |")
    lines.extend(["", "## Poor-Rank Examples", "", "| Beatmap | Window | Step | Current ms | Target | Rank | Top shifts |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"])
    for example in metrics.get("examples", ()):
        row = _mapping(example)
        lines.append(
            "| {beatmap} | `{window}` | `{step}` | `{current}` | `{target}` | `{rank}` | `{top}` |".format(
                beatmap=str(row.get("beatmap_path") or "").replace("|", "\\|"),
                window=row.get("window_start_ms"),
                step=row.get("step_index"),
                current=row.get("current_ms"),
                target=row.get("target_shift_ms"),
                rank=row.get("target_rank"),
                top=", ".join(str(value) for value in row.get("top_shift_ms", ())),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(decision.get("interpretation")),
            "",
            "## What This Does Not Prove",
            "",
            "- It does not prove rollout quality improvement.",
            "- It does not prove v3 replacement readiness.",
            "- It does not evaluate the full 4k dataset.",
            "- It does not prove all timing objectives are bad or good.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def jensen_shannon_divergence(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    total_left = sum(max(float(left.get(key, 0.0)), 0.0) for key in keys)
    total_right = sum(max(float(right.get(key, 0.0)), 0.0) for key in keys)
    if total_left <= 0.0 or total_right <= 0.0:
        return 0.0
    divergence = 0.0
    for key in keys:
        p = max(float(left.get(key, 0.0)), 0.0) / total_left
        q = max(float(right.get(key, 0.0)), 0.0) / total_right
        m = 0.5 * (p + q)
        if p > 0.0:
            divergence += 0.5 * p * math.log2(p / m)
        if q > 0.0:
            divergence += 0.5 * q * math.log2(q / m)
    return divergence


def _rank_summary(ranks: Sequence[int]) -> dict[str, float | int | None]:
    if not ranks:
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    sorted_ranks = sorted(int(rank) for rank in ranks)
    return {
        "count": len(sorted_ranks),
        "mean": sum(sorted_ranks) / len(sorted_ranks),
        "median": _quantile(sorted_ranks, 0.5),
        "p90": _quantile(sorted_ranks, 0.9),
        "max": max(sorted_ranks),
    }


def _quantile(values: Sequence[int], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * float(q)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(values[lower])
    fraction = position - lower
    return float(values[lower]) * (1.0 - fraction) + float(values[upper]) * fraction


def _counter_share(counter: Counter[int], keys: Sequence[int], denominator: int) -> float:
    return _safe_ratio(sum(counter.get(int(key), 0) for key in keys), denominator)


def _normalized_counter(counter: Counter[int]) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {str(key): value / total for key, value in sorted(counter.items())}


def _top_counter_rows(counter: Mapping[str, Any], *, limit: int) -> list[tuple[str, int]]:
    rows = [(str(key), int(value)) for key, value in counter.items()]
    return sorted(rows, key=lambda item: (-item[1], item[0]))[:limit]


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _training_report_contract(report: Mapping[str, Any]) -> str | None:
    training_config = _mapping(report.get("training_config"))
    return _optional_str(training_config.get("mapper_token_contract"))


def _source_window_count_from_report(report: Mapping[str, Any]) -> int | None:
    dataset_report = _mapping(_mapping(report.get("training_config")).get("dataset")) or _mapping(report.get("dataset"))
    return _optional_int(dataset_report.get("source_window_count"))


def _reported_eval_window_count(report: Mapping[str, Any]) -> int | None:
    dataset_report = _mapping(_mapping(report.get("training_config")).get("dataset")) or _mapping(report.get("dataset"))
    return _optional_int(dataset_report.get("eval_window_count"))


def _required_path(mapping: Mapping[str, Any], key: str) -> Path:
    value = mapping.get(key)
    if value is None:
        raise ValueError(f"missing required path field: {key}")
    return Path(str(value))


def _record_cache_validity_metadata(cache_path: Path) -> Mapping[str, Any]:
    metadata_path = cache_path.with_suffix(".json")
    if not metadata_path.exists():
        return {}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(metadata, Mapping):
        return {}
    return _mapping(metadata.get("validity"))


def _optional_path(value: object) -> Path | None:
    if value is None:
        return None
    return Path(str(value))


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _safe_ratio(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if denominator <= 0.0:
        return 0.0
    return float(numerator) / denominator


def _fmt(value: object) -> str:
    numeric = _optional_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.6f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit teacher-forced v3 time-shift logits.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT.as_posix())
    parser.add_argument("--report-output", default=DEFAULT_REPORT_OUTPUT.as_posix())
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument(
        "--disable-global-context-for-smoke",
        action="store_true",
        help="Skip full-song context fields and disable global context in forward; this is loader/metric smoke only.",
    )
    args = parser.parse_args(argv)
    summary = run_teacher_forced_time_shift_logit_audit(
        checkpoint_path=Path(args.checkpoint),
        training_report_path=Path(args.training_report),
        summary_output=Path(args.summary_output),
        report_output=Path(args.report_output) if args.report_output else None,
        device_name=args.device,
        batch_size=int(args.batch_size),
        max_windows=args.max_windows,
        disable_global_context_for_smoke=bool(args.disable_global_context_for_smoke),
    )
    print(
        "mapper_v3_teacher_forced_time_shift_logit_audit_done "
        f"route={summary['decision']['route']} "
        f"rows={summary['metrics']['time_shift_row_count']} "
        f"recall5={summary['metrics']['recall_at_k'].get('5')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
