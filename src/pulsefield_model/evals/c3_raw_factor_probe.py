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
    build_datasets_from_config,
    iter_dataset_window_keys,
    load_model_from_checkpoint,
    token_kind_lookup,
)
from pulsefield_model.evals.c3_raw_structure_audit import parse_raw_token_text
from pulsefield_model.training.common import select_torch_device
from pulsefield_model.training.mapper_common import _move_mapper_batch_tensors


SUMMARY_SCHEMA_VERSION = 1
RAW_FACTORS = ("field_1", "field_2", "field_3", "field_4")
DEFAULT_TOP_KS = (1, 3, 5)


@dataclass(frozen=True)
class RawFactorVocab:
    token_fields: Mapping[int, Mapping[str, str]]
    value_token_indexes: Mapping[str, Mapping[str, tuple[int, ...]]]
    value_order: Mapping[str, tuple[str, ...]]


class _FieldAccumulator:
    def __init__(self, factors: Sequence[str], top_ks: Sequence[int]) -> None:
        self.factors = tuple(factors)
        self.top_ks = tuple(top_ks)
        self.target_counts: dict[str, int] = {factor: 0 for factor in self.factors}
        self.hit_counts: dict[str, dict[int, int]] = {
            factor: {int(k): 0 for k in self.top_ks} for factor in self.factors
        }
        self.sample_hit_counts: dict[str, dict[int, int]] = {
            factor: {int(k): 0 for k in self.top_ks} for factor in self.factors
        }
        self.positive_sample_counts: dict[str, int] = {factor: 0 for factor in self.factors}

    def update(self, *, target_values: Mapping[str, frozenset[str]], predicted_values: Mapping[str, Sequence[str]]) -> None:
        for factor in self.factors:
            targets = target_values.get(factor, frozenset())
            if not targets:
                continue
            self.positive_sample_counts[factor] += 1
            self.target_counts[factor] += len(targets)
            predicted = tuple(predicted_values.get(factor, ()))
            for k in self.top_ks:
                hit_count = len(targets.intersection(predicted[: int(k)]))
                self.hit_counts[factor][int(k)] += hit_count
                if hit_count:
                    self.sample_hit_counts[factor][int(k)] += 1

    def to_dict(self) -> dict[str, Any]:
        rows: dict[str, Any] = {}
        for factor in self.factors:
            rows[factor] = {
                "target_count": self.target_counts[factor],
                "positive_sample_count": self.positive_sample_counts[factor],
                "topk": {
                    str(k): {
                        "hits": self.hit_counts[factor][int(k)],
                        "recall": _safe_divide(self.hit_counts[factor][int(k)], self.target_counts[factor]),
                        "sample_hit_count": self.sample_hit_counts[factor][int(k)],
                        "sample_hit_rate": _safe_divide(
                            self.sample_hit_counts[factor][int(k)],
                            self.positive_sample_counts[factor],
                        ),
                    }
                    for k in self.top_ks
                },
            }
        return rows


def parse_raw_factor_fields(token_text: str) -> dict[str, str]:
    parsed = parse_raw_token_text(token_text)
    if len(parsed.numeric_fields) < 4:
        raise ValueError(f"RAW token must contain at least four numeric fields: {token_text!r}")
    return {factor: parsed.numeric_fields[index] for index, factor in enumerate(RAW_FACTORS)}


def build_raw_factor_vocab(token_by_id: Mapping[int, str]) -> RawFactorVocab:
    token_fields: dict[int, dict[str, str]] = {}
    value_token_ids: dict[str, dict[str, list[int]]] = {
        factor: defaultdict(list) for factor in RAW_FACTORS
    }
    for token_id, token_text in token_by_id.items():
        if not str(token_text).startswith("RAW|"):
            continue
        fields = parse_raw_factor_fields(str(token_text))
        token_fields[int(token_id)] = fields
        for factor, value in fields.items():
            value_token_ids[factor][value].append(int(token_id))
    value_token_indexes = {
        factor: {
            value: tuple(sorted(int(token_id) - 1 for token_id in token_ids))
            for value, token_ids in values.items()
        }
        for factor, values in value_token_ids.items()
    }
    value_order = {
        factor: tuple(sorted(values.keys(), key=_field_value_sort_key))
        for factor, values in value_token_ids.items()
    }
    return RawFactorVocab(
        token_fields=token_fields,
        value_token_indexes=value_token_indexes,
        value_order=value_order,
    )


def field_value_scores(
    logits: torch.Tensor,
    *,
    factor: str,
    vocab: RawFactorVocab,
    aggregation: str,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for value, indexes in vocab.value_token_indexes[factor].items():
        if not indexes:
            continue
        index_tensor = torch.tensor(indexes, dtype=torch.long)
        selected = logits.index_select(0, index_tensor)
        if aggregation == "max":
            score = float(torch.max(selected).item())
        elif aggregation == "logsumexp":
            score = float(torch.logsumexp(selected, dim=0).item())
        else:
            raise ValueError(f"unknown aggregation: {aggregation}")
        scores[value] = score
    return scores


def top_field_values(
    logits: torch.Tensor,
    *,
    vocab: RawFactorVocab,
    aggregation: str,
    limit: int,
) -> dict[str, tuple[str, ...]]:
    top_values: dict[str, tuple[str, ...]] = {}
    for factor in RAW_FACTORS:
        scores = field_value_scores(logits, factor=factor, vocab=vocab, aggregation=aggregation)
        top_values[factor] = tuple(
            value
            for value, _score in sorted(
                scores.items(),
                key=lambda item: (-item[1], _field_value_sort_key(item[0])),
            )[: int(limit)]
        )
    return top_values


def build_train_field_unigram_top_values(
    train_dataset: Any,
    token_lookup: Mapping[str, Mapping[int, Sequence[int]]],
    vocab: RawFactorVocab,
    *,
    limit: int,
) -> tuple[dict[str, Counter[str]], dict[str, tuple[str, ...]]]:
    counts: dict[str, Counter[str]] = {factor: Counter() for factor in RAW_FACTORS}
    for beatmap_path, window_start_ms in iter_dataset_window_keys(train_dataset):
        token_ids = token_lookup.get(beatmap_path, {}).get(window_start_ms)
        if not token_ids:
            continue
        raw_token_ids = {int(token_id) for token_id in token_ids if int(token_id) in vocab.token_fields}
        target_values = _target_factor_values(raw_token_ids, vocab=vocab)
        for factor, values in target_values.items():
            counts[factor].update(values)
    top_values = {
        factor: tuple(value for value, _count in counter.most_common(int(limit)))
        for factor, counter in counts.items()
    }
    return counts, top_values


def run_raw_factor_probe(
    *,
    config_path: Path,
    checkpoint_path: Path,
    batch_size: int,
    device_name: str,
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
    max_batches: int | None = None,
) -> dict[str, Any]:
    started_at = time.monotonic()
    top_ks = tuple(sorted({int(k) for k in top_ks}))
    if not top_ks or top_ks[0] <= 0:
        raise ValueError("top_ks must contain positive integers")
    max_k = max(top_ks)
    datasets = build_datasets_from_config(config_path)
    vocab = build_raw_factor_vocab(datasets.token_by_id)
    train_counts, train_top_values = build_train_field_unigram_top_values(
        datasets.train_dataset,
        datasets.token_lookup,
        vocab,
        limit=max_k,
    )
    kind_by_id = token_kind_lookup(datasets.token_by_id)
    device = select_torch_device(device_name)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device=device)
    loader = DataLoader(
        datasets.eval_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_mapper_v2_1_windows,
    )
    aggregators = {
        "max": _FieldAccumulator(RAW_FACTORS, top_ks),
        "logsumexp": _FieldAccumulator(RAW_FACTORS, top_ks),
        "unigram": _FieldAccumulator(RAW_FACTORS, top_ks),
    }
    joint = {
        "max": {str(k): {"hits": 0, "targets": 0} for k in top_ks},
        "logsumexp": {str(k): {"hits": 0, "targets": 0} for k in top_ks},
        "unigram": {str(k): {"hits": 0, "targets": 0} for k in top_ks},
    }
    exact_raw = {str(k): {"full_hits": 0, "raw_only_hits": 0, "targets": 0} for k in (20,)}
    totals = {
        "sample_count": 0,
        "available_sample_count": 0,
        "raw_positive_sample_count": 0,
        "raw_positive_label_count": 0,
    }
    top_misses: dict[str, Counter[str]] = {factor: Counter() for factor in RAW_FACTORS}
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
            token_rows, mask_rows, available_rows = _batch_token_rows(batch)
            totals["sample_count"] += int(logits_cpu.shape[0])
            for row_index in range(int(logits_cpu.shape[0])):
                if not available_rows[row_index]:
                    continue
                totals["available_sample_count"] += 1
                target_ids = frozenset(
                    token_id
                    for token_id, is_valid in zip(token_rows[row_index], mask_rows[row_index])
                    if is_valid and token_id in vocab.token_fields
                )
                if not target_ids:
                    continue
                totals["raw_positive_sample_count"] += 1
                totals["raw_positive_label_count"] += len(target_ids)
                target_values = _target_factor_values(target_ids, vocab=vocab)
                predicted_by_aggregation = {
                    aggregation: top_field_values(
                        logits_cpu[row_index],
                        vocab=vocab,
                        aggregation=aggregation,
                        limit=max_k,
                    )
                    for aggregation in ("max", "logsumexp")
                }
                predicted_by_aggregation["unigram"] = train_top_values
                for aggregation, predicted_values in predicted_by_aggregation.items():
                    aggregators[aggregation].update(target_values=target_values, predicted_values=predicted_values)
                    _update_joint_counts(
                        joint[aggregation],
                        target_ids=target_ids,
                        predicted_values=predicted_values,
                        vocab=vocab,
                        top_ks=top_ks,
                    )
                _update_exact_raw_counts(
                    exact_raw,
                    logits=logits_cpu[row_index],
                    target_ids=target_ids,
                    kind_by_id=kind_by_id,
                )
                _update_top_misses(
                    top_misses,
                    target_values=target_values,
                    predicted_values=predicted_by_aggregation["max"],
                    top_k=3,
                )

    field_metrics = {name: accumulator.to_dict() for name, accumulator in aggregators.items()}
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "C3 RAW factor-marginal probe",
        "config_path": config_path.as_posix(),
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint": {
            "run_name": checkpoint.get("run_name"),
            "completed_steps": int(checkpoint.get("training_state", {}).get("step", 0)),
            "is_complete": bool(checkpoint.get("training_state", {}).get("is_complete", False)),
            "use_c3_auxiliary_kind_heads": bool(
                checkpoint.get("model_config", {}).get("use_c3_auxiliary_kind_heads", False)
            ),
        },
        "dataset": {
            "train_window_count": len(datasets.train_dataset),
            "eval_window_count": len(datasets.eval_dataset),
            "batch_size": int(batch_size),
            "max_batches": max_batches,
        },
        "config": {
            "top_ks": [int(k) for k in top_ks],
            "factors": list(RAW_FACTORS),
        },
        "raw_factor_vocab": {
            factor: {
                "value_count": len(vocab.value_order[factor]),
                "values": list(vocab.value_order[factor]),
                "top_train_values": [
                    {"value": value, "count": int(train_counts[factor].get(value, 0))}
                    for value in train_top_values[factor]
                ],
            }
            for factor in RAW_FACTORS
        },
        "totals": totals,
        "field_metrics": field_metrics,
        "joint_factor_coverage": _joint_to_dict(joint),
        "exact_raw_token_recovery": _exact_raw_to_dict(exact_raw),
        "top_max_missed_values_at_3": {
            factor: [
                {"value": value, "count": int(count)}
                for value, count in counter.most_common(10)
            ]
            for factor, counter in top_misses.items()
        },
        "elapsed_s": time.monotonic() - started_at,
    }
    summary["decision"] = _decision(summary)
    return _normalize_json(summary)


def _target_factor_values(target_ids: set[int] | frozenset[int], *, vocab: RawFactorVocab) -> dict[str, frozenset[str]]:
    values: dict[str, set[str]] = {factor: set() for factor in RAW_FACTORS}
    for token_id in target_ids:
        fields = vocab.token_fields.get(int(token_id))
        if fields is None:
            continue
        for factor, value in fields.items():
            values[factor].add(value)
    return {factor: frozenset(factor_values) for factor, factor_values in values.items()}


def _update_joint_counts(
    joint_counts: dict[str, dict[str, int]],
    *,
    target_ids: frozenset[int],
    predicted_values: Mapping[str, Sequence[str]],
    vocab: RawFactorVocab,
    top_ks: Sequence[int],
) -> None:
    for k in top_ks:
        key = str(k)
        top_values = {
            factor: frozenset(predicted_values.get(factor, ())[: int(k)])
            for factor in RAW_FACTORS
        }
        for token_id in target_ids:
            fields = vocab.token_fields[int(token_id)]
            joint_counts[key]["targets"] += 1
            if all(fields[factor] in top_values[factor] for factor in RAW_FACTORS):
                joint_counts[key]["hits"] += 1


def _update_exact_raw_counts(
    exact_raw: dict[str, dict[str, int]],
    *,
    logits: torch.Tensor,
    target_ids: frozenset[int],
    kind_by_id: Mapping[int, str],
) -> None:
    top20 = tuple(int(token_id) for token_id in (torch.topk(logits, k=min(20, int(logits.numel()))).indices + 1).tolist())
    raw_indexes = [int(token_id) - 1 for token_id, kind in kind_by_id.items() if kind == "RAW"]
    raw_logits = logits.index_select(0, torch.tensor(sorted(raw_indexes), dtype=torch.long))
    raw_token_ids = [index + 1 for index in sorted(raw_indexes)]
    raw_top_positions = torch.topk(raw_logits, k=min(20, len(raw_token_ids))).indices.tolist()
    raw_top20 = tuple(raw_token_ids[position] for position in raw_top_positions)
    exact_raw["20"]["targets"] += len(target_ids)
    exact_raw["20"]["full_hits"] += len(target_ids.intersection(top20))
    exact_raw["20"]["raw_only_hits"] += len(target_ids.intersection(raw_top20))


def _update_top_misses(
    top_misses: Mapping[str, Counter[str]],
    *,
    target_values: Mapping[str, frozenset[str]],
    predicted_values: Mapping[str, Sequence[str]],
    top_k: int,
) -> None:
    for factor in RAW_FACTORS:
        predicted = frozenset(predicted_values.get(factor, ())[: int(top_k)])
        for value in target_values.get(factor, frozenset()) - predicted:
            top_misses[factor][value] += 1


def _joint_to_dict(joint: Mapping[str, Mapping[str, Mapping[str, int]]]) -> dict[str, Any]:
    return {
        aggregation: {
            k: {
                "hits": int(values["hits"]),
                "targets": int(values["targets"]),
                "coverage": _safe_divide(values["hits"], values["targets"]),
            }
            for k, values in rows.items()
        }
        for aggregation, rows in joint.items()
    }


def _exact_raw_to_dict(exact_raw: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    return {
        k: {
            "targets": int(values["targets"]),
            "full_hits": int(values["full_hits"]),
            "raw_only_hits": int(values["raw_only_hits"]),
            "full_recall": _safe_divide(values["full_hits"], values["targets"]),
            "raw_only_recall": _safe_divide(values["raw_only_hits"], values["targets"]),
        }
        for k, values in exact_raw.items()
    }


def _decision(summary: Mapping[str, Any]) -> dict[str, Any]:
    max_metrics = summary["field_metrics"]["max"]
    unigram_metrics = summary["field_metrics"]["unigram"]
    improved_fields = []
    weak_fields = []
    for factor in RAW_FACTORS:
        model_recall = max_metrics[factor]["topk"]["3"]["recall"]
        unigram_recall = unigram_metrics[factor]["topk"]["3"]["recall"]
        if model_recall is not None and unigram_recall is not None and float(model_recall) > float(unigram_recall):
            improved_fields.append(factor)
        else:
            weak_fields.append(factor)
    joint5 = summary["joint_factor_coverage"]["max"]["5"]["coverage"]
    exact_full = summary["exact_raw_token_recovery"]["20"]["full_recall"]
    exact_raw_only = summary["exact_raw_token_recovery"]["20"]["raw_only_recall"]
    joint_lift_vs_full = None if joint5 is None or exact_full is None else float(joint5) - float(exact_full)
    joint_lift_vs_raw_only = None if joint5 is None or exact_raw_only is None else float(joint5) - float(exact_raw_only)
    positive = len(improved_fields) >= 2 and joint_lift_vs_full is not None and joint_lift_vs_full >= 0.10
    if positive:
        route = "TEST_NEXT"
        interpretation = (
            "RAW factor marginals recover field structure substantially better than exact token top-20. "
            "This justifies a bounded RAW factor-head training card."
        )
        next_step = "Create a RAW factor-head smoke/training Experiment Card."
    elif joint_lift_vs_full is not None and joint_lift_vs_full >= 0.10:
        route = "MUTATE"
        interpretation = (
            "Joint factor coverage improves over exact token recovery, but too few fields beat unigram at K=3. "
            "Factorization has signal, but the field design or metric needs tightening before model changes."
        )
        next_step = "Refine RAW factor grouping or repeat with density/chord buckets before adding heads."
    else:
        route = "MUTATE_TO_ORDERED_GRAMMAR"
        interpretation = (
            "RAW factor marginals do not provide enough recovery over exact token recall. "
            "Simple RAW field heads are unlikely to solve the blocker alone."
        )
        next_step = "Create an ordered C3 target-grammar card or richer RAW context probe."
    return {
        "route": route,
        "improved_fields_at_3": improved_fields,
        "weak_fields_at_3": weak_fields,
        "joint_factor_coverage_at_5": joint5,
        "exact_full_raw_recall_at_20": exact_full,
        "exact_raw_only_recall_at_20": exact_raw_only,
        "joint_lift_vs_full_exact_at_5": joint_lift_vs_full,
        "joint_lift_vs_raw_only_exact_at_5": joint_lift_vs_raw_only,
        "positive_signal_observed": positive,
        "interpretation": interpretation,
        "next_step": next_step,
    }


def write_summary(summary: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown_report(summary: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    decision = summary["decision"]
    text = "\n".join(
        [
            "# C3 RAW Factor-Marginal Probe Result Report",
            "",
            "## Scope",
            "",
            "This P22 pass tests whether P19's RAW logits contain recoverable field-level structure even when exact RAW token top-20 recovery trails unigram.",
            "",
            "## Result",
            "",
            f"Decision: {decision['route']}.",
            "",
            f"- eval windows: `{summary['dataset']['eval_window_count']}`",
            f"- RAW positive samples: `{summary['totals']['raw_positive_sample_count']}`",
            f"- RAW positive labels: `{summary['totals']['raw_positive_label_count']}`",
            f"- exact full-vocab RAW recall@20: `{_fmt_metric(decision['exact_full_raw_recall_at_20'])}`",
            f"- exact RAW-only recall@20: `{_fmt_metric(decision['exact_raw_only_recall_at_20'])}`",
            f"- max-marginal joint factor coverage@5: `{_fmt_metric(decision['joint_factor_coverage_at_5'])}`",
            f"- joint lift vs full exact: `{_fmt_metric(decision['joint_lift_vs_full_exact_at_5'])}`",
            f"- improved fields@3: `{', '.join(decision['improved_fields_at_3']) or 'none'}`",
            f"- weak fields@3: `{', '.join(decision['weak_fields_at_3']) or 'none'}`",
            f"- elapsed: `{float(summary['elapsed_s']):.2f}s`",
            "",
            "## Field Recall At K=3",
            "",
            "| Field | Vocab | Max recall | Logsumexp recall | Unigram recall | Max hits | Unigram hits |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *_field_rows(summary, k="3"),
            "",
            "## Joint Factor Coverage",
            "",
            "| K | Max | Logsumexp | Unigram |",
            "| ---: | ---: | ---: | ---: |",
            *_joint_rows(summary),
            "",
            "## Top Max-Marginal Missed Values At K=3",
            "",
            "| Field | Missed values |",
            "| --- | --- |",
            *_miss_rows(summary),
            "",
            "## What Passed",
            "",
            "- The P19 checkpoint, config, reduced sidecar, and eval split are loadable.",
            "- RAW factor scoring is post-hoc; no C3 labels are used as model inputs.",
            "- Max, logsumexp, and train-unigram factor comparators are all reported.",
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


def _field_rows(summary: Mapping[str, Any], *, k: str) -> list[str]:
    rows = []
    for factor in RAW_FACTORS:
        max_row = summary["field_metrics"]["max"][factor]["topk"][k]
        logsumexp_row = summary["field_metrics"]["logsumexp"][factor]["topk"][k]
        unigram_row = summary["field_metrics"]["unigram"][factor]["topk"][k]
        rows.append(
            f"| {factor} | {summary['raw_factor_vocab'][factor]['value_count']} | "
            f"{_fmt_metric(max_row['recall'])} | {_fmt_metric(logsumexp_row['recall'])} | "
            f"{_fmt_metric(unigram_row['recall'])} | {max_row['hits']} | {unigram_row['hits']} |"
        )
    return rows


def _joint_rows(summary: Mapping[str, Any]) -> list[str]:
    rows = []
    for k in ("1", "3", "5"):
        rows.append(
            f"| {k} | {_fmt_metric(summary['joint_factor_coverage']['max'][k]['coverage'])} | "
            f"{_fmt_metric(summary['joint_factor_coverage']['logsumexp'][k]['coverage'])} | "
            f"{_fmt_metric(summary['joint_factor_coverage']['unigram'][k]['coverage'])} |"
        )
    return rows


def _miss_rows(summary: Mapping[str, Any]) -> list[str]:
    rows = []
    for factor in RAW_FACTORS:
        values = ", ".join(
            f"`{row['value']}` ({row['count']})" for row in summary["top_max_missed_values_at_3"][factor][:5]
        )
        rows.append(f"| {factor} | {values} |")
    return rows


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


def _field_value_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


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
    parser = argparse.ArgumentParser(description="Run a C3 RAW factor-marginal probe.")
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
    summary = run_raw_factor_probe(
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
        "c3_raw_factor_probe "
        f"decision={summary['decision']['route']} "
        f"joint_factor_coverage_at_5={_fmt_metric(summary['decision']['joint_factor_coverage_at_5'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
