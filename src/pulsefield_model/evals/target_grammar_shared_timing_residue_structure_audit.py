from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from pulsefield_model.evals.c3_auxiliary_diagnostics import build_datasets_from_config
from pulsefield_model.evals.target_grammar_v3_event_smoke import (
    _resolve_limit,
    _safe_ratio,
    _tokenized_window_for_dataset_index,
    parse_limit,
)
from pulsefield_model.models.mapper.v2_1.vocab import MapperV21Vocab
from pulsefield_model.models.mapper.v3.conversion import v2_1_tokens_to_v3_event_tokens
from pulsefield_model.models.mapper.v3.vocab import MapperV3Vocab


SUMMARY_SCHEMA_VERSION = 1
DEFAULT_CONFIG_PATH = Path("artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_enabled.yaml")
DEFAULT_SUMMARY_OUTPUT_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/"
    "target_grammar_shared_timing_residue_structure_audit_summary.json"
)
DEFAULT_REPORT_OUTPUT_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/"
    "target_grammar_shared_timing_residue_structure_audit_result_report.md"
)
DEFAULT_V3_MULTICASE_SUMMARY_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_multicase_timing_quality_summary.json"
)
DEFAULT_V3_DECODE_POLICY_SUMMARY_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_decode_policy_continuation_sweep_summary.json"
)
DEFAULT_V21_MATCHED_SUMMARY_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_matched_v21_timing_baseline_summary.json"
)
DEFAULT_GENERATED_STATE_SUMMARY_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/"
    "target_grammar_v3_generated_state_continuation_diagnostic_summary.json"
)
DEFAULT_V3_FULL_DATASET_SUMMARY_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_full_dataset_audit_summary.json"
)
DEFAULT_C3_TARGET_COMPLEXITY_SUMMARY_PATH = Path(
    "artifacts/reports/audits/context_adaptive_fallback_codec/c3_v3_target_complexity_comparison_summary.json"
)
DEFAULT_FACTORIZED_EVENT_SUMMARY_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_factorized_event_signature_proxy_summary.json"
)

EFFECTIVE_INTERVAL_VOCAB_THRESHOLD = 6.0
INTERVAL_160_200_SHARE_THRESHOLD = 0.60
RIGID_WINDOW_95_SHARE_THRESHOLD = 0.75
GENERATED_RIGID_RATIO_THRESHOLD = 0.95


@dataclass
class TimingResidueAccumulator:
    split_name: str
    source_vocab: MapperV21Vocab
    target_vocab: MapperV3Vocab
    window_count: int = 0
    event_window_count: int = 0
    interval_window_count: int = 0
    event_count: int = 0
    event_interval_count: int = 0
    v3_reconstruction_mismatches: int = 0
    time_shift_parity_mismatches: int = 0
    v2_1_time_shift_counts: Counter[int] = field(default_factory=Counter)
    v3_time_shift_counts: Counter[int] = field(default_factory=Counter)
    event_interval_counts: Counter[int] = field(default_factory=Counter)
    window_dominant_interval_ratios: list[float] = field(default_factory=list)
    mismatch_examples: list[dict[str, Any]] = field(default_factory=list)

    def update(self, *, token_ids: Sequence[int], dataset_index: int) -> None:
        normalized = tuple(int(token_id) for token_id in token_ids)
        conversion = v2_1_tokens_to_v3_event_tokens(
            normalized,
            source_vocab=self.source_vocab,
            target_vocab=self.target_vocab,
        )
        self.window_count += 1
        if conversion.reconstructed_v2_1_token_ids != normalized:
            self.v3_reconstruction_mismatches += 1
            if len(self.mismatch_examples) < 5:
                self.mismatch_examples.append(
                    {
                        "split": self.split_name,
                        "dataset_index": int(dataset_index),
                        "kind": "v3_reconstruction",
                        "baseline_prefix": [self.source_vocab.token_name(token_id) for token_id in normalized[:20]],
                        "reconstructed_prefix": [
                            self.source_vocab.token_name(token_id)
                            for token_id in conversion.reconstructed_v2_1_token_ids[:20]
                        ],
                    }
                )

        v2_shifts = time_shift_values_from_v2_1_tokens(normalized, vocab=self.source_vocab)
        v3_shifts = time_shift_values_from_v3_tokens(conversion.token_ids, vocab=self.target_vocab)
        self.v2_1_time_shift_counts.update(v2_shifts)
        self.v3_time_shift_counts.update(v3_shifts)
        if v2_shifts != v3_shifts:
            self.time_shift_parity_mismatches += 1
            if len(self.mismatch_examples) < 5:
                self.mismatch_examples.append(
                    {
                        "split": self.split_name,
                        "dataset_index": int(dataset_index),
                        "kind": "time_shift_parity",
                        "v2_1_time_shifts_prefix": list(v2_shifts[:20]),
                        "v3_time_shifts_prefix": list(v3_shifts[:20]),
                    }
                )

        event_times = event_times_from_v3_tokens(conversion.token_ids, vocab=self.target_vocab)
        self.event_count += len(event_times)
        if event_times:
            self.event_window_count += 1
        intervals = event_intervals_from_times(event_times)
        self.event_interval_counts.update(intervals)
        self.event_interval_count += len(intervals)
        if intervals:
            self.interval_window_count += 1
            dominant_count = max(Counter(intervals).values())
            self.window_dominant_interval_ratios.append(float(dominant_count) / float(len(intervals)))

    def to_dict(self) -> dict[str, Any]:
        shift_stats = distribution_stats(self.v2_1_time_shift_counts)
        v3_shift_stats = distribution_stats(self.v3_time_shift_counts)
        interval_stats = distribution_stats(self.event_interval_counts)
        window_stats = dominant_ratio_stats(self.window_dominant_interval_ratios)
        return {
            "split": self.split_name,
            "window_count": self.window_count,
            "event_window_count": self.event_window_count,
            "interval_window_count": self.interval_window_count,
            "event_count": self.event_count,
            "event_interval_count": self.event_interval_count,
            "v3_reconstruction_mismatches": self.v3_reconstruction_mismatches,
            "time_shift_parity_mismatches": self.time_shift_parity_mismatches,
            "time_shift": {
                **shift_stats,
                "v3_parity_stats": v3_shift_stats,
                "top_values": top_count_rows(self.v2_1_time_shift_counts, limit=20),
                "v3_top_values": top_count_rows(self.v3_time_shift_counts, limit=20),
                "share_200ms": share_for_values(self.v2_1_time_shift_counts, {200}),
                "share_le_100ms": share_for_predicate(self.v2_1_time_shift_counts, lambda value: value <= 100),
            },
            "event_intervals": {
                **interval_stats,
                "top_values": top_count_rows(self.event_interval_counts, limit=30),
                "share_160ms": share_for_values(self.event_interval_counts, {160}),
                "share_200ms": share_for_values(self.event_interval_counts, {200}),
                "share_160_200ms": share_for_values(self.event_interval_counts, {160, 200}),
                "share_le_200ms": share_for_predicate(self.event_interval_counts, lambda value: value <= 200),
                "dominant_window_ratio": window_stats,
            },
            "mismatch_examples": list(self.mismatch_examples),
        }


def time_shift_values_from_v2_1_tokens(token_ids: Sequence[int], *, vocab: MapperV21Vocab) -> tuple[int, ...]:
    return tuple(vocab.time_shift_value(int(token_id)) for token_id in token_ids if vocab.is_time_shift_token(int(token_id)))


def time_shift_values_from_v3_tokens(token_ids: Sequence[int], *, vocab: MapperV3Vocab) -> tuple[int, ...]:
    return tuple(vocab.time_shift_value(int(token_id)) for token_id in token_ids if vocab.is_time_shift_token(int(token_id)))


def event_times_from_v3_tokens(token_ids: Sequence[int], *, vocab: MapperV3Vocab) -> tuple[int, ...]:
    current_ms = 0
    event_times: list[int] = []
    for token_id in (int(token_id) for token_id in token_ids):
        if vocab.is_time_shift_token(token_id):
            current_ms += vocab.time_shift_value(token_id)
        elif vocab.is_event_token(token_id):
            event_times.append(current_ms)
    return tuple(event_times)


def event_intervals_from_times(event_times: Sequence[int]) -> tuple[int, ...]:
    intervals: list[int] = []
    previous: int | None = None
    for event_time in (int(value) for value in event_times):
        if previous is not None:
            delta = event_time - previous
            if delta > 0:
                intervals.append(delta)
        previous = event_time
    return tuple(intervals)


def distribution_stats(counts: Mapping[int, int]) -> dict[str, Any]:
    total = int(sum(int(value) for value in counts.values()))
    if total <= 0:
        return {
            "count": 0,
            "vocab_size": 0,
            "entropy_bits": None,
            "effective_vocab": None,
            "top1_share": None,
            "top3_share": None,
        }
    probabilities = [float(count) / float(total) for count in counts.values() if int(count) > 0]
    entropy = -sum(probability * math.log2(probability) for probability in probabilities)
    ordered = sorted((int(count) for count in counts.values()), reverse=True)
    return {
        "count": total,
        "vocab_size": len(counts),
        "entropy_bits": entropy,
        "effective_vocab": 2.0**entropy,
        "top1_share": float(ordered[0]) / float(total),
        "top3_share": float(sum(ordered[:3])) / float(total),
    }


def dominant_ratio_stats(ratios: Sequence[float]) -> dict[str, Any]:
    if not ratios:
        return {
            "window_count": 0,
            "mean": None,
            "median": None,
            "share_ge_0_80": None,
            "share_ge_0_90": None,
            "share_ge_0_95": None,
        }
    values = [float(value) for value in ratios]
    return {
        "window_count": len(values),
        "mean": sum(values) / float(len(values)),
        "median": statistics.median(values),
        "share_ge_0_80": _share(values, threshold=0.80),
        "share_ge_0_90": _share(values, threshold=0.90),
        "share_ge_0_95": _share(values, threshold=0.95),
    }


def share_for_values(counts: Mapping[int, int], values: set[int]) -> float | None:
    total = int(sum(int(value) for value in counts.values()))
    if total <= 0:
        return None
    selected = sum(int(count) for value, count in counts.items() if int(value) in values)
    return float(selected) / float(total)


def share_for_predicate(counts: Mapping[int, int], predicate: Any) -> float | None:
    total = int(sum(int(value) for value in counts.values()))
    if total <= 0:
        return None
    selected = sum(int(count) for value, count in counts.items() if bool(predicate(int(value))))
    return float(selected) / float(total)


def top_count_rows(counts: Mapping[int, int], *, limit: int) -> list[dict[str, Any]]:
    return [
        {"value_ms": int(value), "count": int(count)}
        for value, count in sorted(counts.items(), key=lambda item: (-int(item[1]), int(item[0])))[: int(limit)]
    ]


def run_shared_timing_residue_structure_audit(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    train_limit: int | None = 4096,
    eval_limit: int | None = None,
    v3_multicase_summary_path: Path = DEFAULT_V3_MULTICASE_SUMMARY_PATH,
    v3_decode_policy_summary_path: Path = DEFAULT_V3_DECODE_POLICY_SUMMARY_PATH,
    v21_matched_summary_path: Path = DEFAULT_V21_MATCHED_SUMMARY_PATH,
    generated_state_summary_path: Path = DEFAULT_GENERATED_STATE_SUMMARY_PATH,
    v3_full_dataset_summary_path: Path = DEFAULT_V3_FULL_DATASET_SUMMARY_PATH,
    c3_target_complexity_summary_path: Path = DEFAULT_C3_TARGET_COMPLEXITY_SUMMARY_PATH,
    factorized_event_summary_path: Path = DEFAULT_FACTORIZED_EVENT_SUMMARY_PATH,
    progress_interval: int = 0,
) -> dict[str, Any]:
    started_at = time.monotonic()
    source_vocab = MapperV21Vocab()
    target_vocab = MapperV3Vocab(source_vocab.time_shift_values_ms)
    datasets = build_datasets_from_config(config_path)
    train_window_count = _resolve_limit(train_limit, available_count=len(datasets.train_dataset), name="train_limit")
    eval_window_count = _resolve_limit(eval_limit, available_count=len(datasets.eval_dataset), name="eval_limit")

    train_audit = _audit_dataset_split(
        datasets.train_dataset,
        split_name="train",
        limit=train_window_count,
        source_vocab=source_vocab,
        target_vocab=target_vocab,
        progress_interval=progress_interval,
    )
    eval_audit = _audit_dataset_split(
        datasets.eval_dataset,
        split_name="eval",
        limit=eval_window_count,
        source_vocab=source_vocab,
        target_vocab=target_vocab,
        progress_interval=progress_interval,
    )
    total_audit = _merge_split_audits((train_audit, eval_audit), source_vocab=source_vocab, target_vocab=target_vocab)
    comparators = load_comparators(
        v3_multicase_summary_path=v3_multicase_summary_path,
        v3_decode_policy_summary_path=v3_decode_policy_summary_path,
        v21_matched_summary_path=v21_matched_summary_path,
        generated_state_summary_path=generated_state_summary_path,
        v3_full_dataset_summary_path=v3_full_dataset_summary_path,
        c3_target_complexity_summary_path=c3_target_complexity_summary_path,
        factorized_event_summary_path=factorized_event_summary_path,
    )
    comparison = comparison_from_audit(total_audit=total_audit, comparators=comparators)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar shared timing-residue structure audit",
        "config_path": config_path.as_posix(),
        "dataset": {
            "train_window_count": len(datasets.train_dataset),
            "train_scored_window_count": train_window_count,
            "eval_window_count": len(datasets.eval_dataset),
            "eval_scored_window_count": eval_window_count,
            "audit_window_count": total_audit.window_count,
            "full_dataset_audit": train_limit is None and eval_limit is None,
        },
        "vocab": {
            "v2_1_vocab_size": source_vocab.size,
            "v3_vocab_size": target_vocab.size,
            "shared_time_shift_values_ms": list(source_vocab.time_shift_values_ms),
        },
        "audit": {
            "train": train_audit.to_dict(),
            "eval": eval_audit.to_dict(),
            "total": total_audit.to_dict(),
        },
        "comparators": comparators,
        "comparison": comparison,
        "checks": checks_from_comparison(comparison),
        "decision": decision_from_comparison(comparison),
        "elapsed_s": time.monotonic() - started_at,
    }


def write_summary(summary: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown_report(summary: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    decision = summary["decision"]
    comparison = summary["comparison"]
    total = summary["audit"]["total"]
    interval = total["event_intervals"]
    shift = total["time_shift"]
    window = interval["dominant_window_ratio"]
    comparators = summary["comparators"]
    dataset = summary["dataset"]
    top_intervals = interval["top_values"][:14]
    top_shifts = shift["top_values"][:14]
    lines = [
        "# Target Grammar Shared Timing-Residue Structure Audit Result Report",
        "",
        "## Scope",
        "",
        "This artifact-only audit measures timing structure in current v2.1 teacher-forcing target streams and in the same streams after v3 event-token conversion. It compares that target timing structure with committed generated-rollout summaries that showed rigid-grid collapse.",
        "",
        "It does not change tokenizer semantics, model heads, loss, decode policy, training configs, or inference defaults.",
        "",
        "## Result",
        "",
        f"Decision: `{decision['route']}`.",
        "",
        f"- Reason: {decision['reason']}",
        f"- Train windows scored: `{dataset['train_scored_window_count']}` / `{dataset['train_window_count']}`",
        f"- Eval windows scored: `{dataset['eval_scored_window_count']}` / `{dataset['eval_window_count']}`",
        f"- v2.1/v3 time-shift parity mismatches: `{comparison['time_shift_parity_mismatches']}`",
        f"- v3 reconstruction mismatches: `{comparison['v3_reconstruction_mismatches']}`",
        f"- Target event-interval effective vocab: `{_fmt_metric(comparison['event_interval_effective_vocab'])}`",
        f"- Target interval `160/200ms` share: `{_fmt_pct(comparison['event_interval_160_200_share'])}`",
        f"- Target rigid-window share >=0.95: `{_fmt_pct(comparison['target_rigid_window_share_ge_0_95'])}`",
        f"- v3 generated mean dominant-spacing ratio: `{_fmt_metric(comparison['generated_v3_mean_dominant_spacing_ratio'])}`",
        f"- matched v2.1 generated mean dominant-spacing ratio: `{_fmt_metric(comparison['generated_v21_mean_dominant_spacing_ratio'])}`",
        f"- elapsed: `{float(summary['elapsed_s']):.2f}s`",
        "",
        "## Target Timing Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| time-shift entropy bits | `{_fmt_metric(shift['entropy_bits'])}` |",
        f"| time-shift effective vocab | `{_fmt_metric(shift['effective_vocab'])}` |",
        f"| time-shift top1 share | `{_fmt_pct(shift['top1_share'])}` |",
        f"| time-shift top3 share | `{_fmt_pct(shift['top3_share'])}` |",
        f"| event-interval entropy bits | `{_fmt_metric(interval['entropy_bits'])}` |",
        f"| event-interval effective vocab | `{_fmt_metric(interval['effective_vocab'])}` |",
        f"| event-interval top1 share | `{_fmt_pct(interval['top1_share'])}` |",
        f"| event-interval top3 share | `{_fmt_pct(interval['top3_share'])}` |",
        f"| event-interval 160ms share | `{_fmt_pct(interval['share_160ms'])}` |",
        f"| event-interval 200ms share | `{_fmt_pct(interval['share_200ms'])}` |",
        f"| event-interval <=200ms share | `{_fmt_pct(interval['share_le_200ms'])}` |",
        f"| mean window dominant interval ratio | `{_fmt_metric(window['mean'])}` |",
        f"| median window dominant interval ratio | `{_fmt_metric(window['median'])}` |",
        f"| rigid-window share >=0.80 | `{_fmt_pct(window['share_ge_0_80'])}` |",
        f"| rigid-window share >=0.90 | `{_fmt_pct(window['share_ge_0_90'])}` |",
        f"| rigid-window share >=0.95 | `{_fmt_pct(window['share_ge_0_95'])}` |",
        "",
        "## Top Event Intervals",
        "",
        "| Interval ms | Count |",
        "| ---: | ---: |",
        *[f"| `{row['value_ms']}` | {row['count']} |" for row in top_intervals],
        "",
        "## Top Time-Shift Tokens",
        "",
        "| Shift ms | Count |",
        "| ---: | ---: |",
        *[f"| `{row['value_ms']}` | {row['count']} |" for row in top_shifts],
        "",
        "## Comparator Snapshot",
        "",
        f"- v3 multicase route: `{comparators['v3_multicase']['decision_route']}`; generated systematic rigid grid: `{comparators['v3_multicase']['systematic_rigid_grid']}`.",
        f"- decode-policy sweep route: `{comparators['v3_decode_policy']['decision_route']}`.",
        f"- matched v2.1 route: `{comparators['matched_v2_1']['decision_route']}`; generated systematic rigid grid: `{comparators['matched_v2_1']['systematic_rigid_grid']}`.",
        f"- generated-state diagnostic route: `{comparators['generated_state']['decision_route']}`.",
        f"- v3 full-dataset route: `{comparators['v3_full_dataset']['decision_route']}`.",
        f"- C3 target-complexity route: `{comparators['c3_target_complexity']['decision_route']}`.",
        f"- factorized event route: `{comparators['factorized_event']['decision_route']}`.",
        "",
        "## What Passed",
        "",
        f"- v2.1/v3 time-shift parity: `{summary['checks']['time_shift_parity_exact']}`.",
        f"- v3 reconstruction exact on audited slice: `{summary['checks']['v3_reconstruction_exact']}`.",
        f"- Generated rigid-grid evidence available: `{summary['checks']['generated_rigid_evidence_available']}`.",
        f"- Target timing residue is rich enough for calibration follow-up: `{summary['checks']['target_timing_residue_rich']}`.",
        "",
        "## What Surfaced",
        "",
        decision["interpretation"],
        "",
        "This audit cannot approve v3 replacement. It only decides whether the next small experiment should target timing calibration/embedding or mutate the timing grammar itself.",
        "",
        "## Next Step",
        "",
        decision["next_step"],
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _audit_dataset_split(
    dataset: Any,
    *,
    split_name: str,
    limit: int,
    source_vocab: MapperV21Vocab,
    target_vocab: MapperV3Vocab,
    progress_interval: int,
) -> TimingResidueAccumulator:
    accumulator = TimingResidueAccumulator(split_name=split_name, source_vocab=source_vocab, target_vocab=target_vocab)
    started_at = time.monotonic()
    for index in range(int(limit)):
        tokenized = _tokenized_window_for_dataset_index(dataset, index)
        token_ids = tuple(int(token_id) for token_id in tokenized.target_fragment_ids)
        accumulator.update(token_ids=token_ids, dataset_index=index)
        if progress_interval > 0 and accumulator.window_count % int(progress_interval) == 0:
            elapsed_s = time.monotonic() - started_at
            print(
                "target_grammar_shared_timing_residue_audit_progress "
                f"split={split_name} windows={accumulator.window_count}/{limit} "
                f"parity_mismatches={accumulator.time_shift_parity_mismatches} "
                f"v3_mismatches={accumulator.v3_reconstruction_mismatches} elapsed_s={elapsed_s:.1f}",
                flush=True,
            )
    return accumulator


def _merge_split_audits(
    audits: Sequence[TimingResidueAccumulator],
    *,
    source_vocab: MapperV21Vocab,
    target_vocab: MapperV3Vocab,
) -> TimingResidueAccumulator:
    merged = TimingResidueAccumulator(split_name="total", source_vocab=source_vocab, target_vocab=target_vocab)
    for audit in audits:
        merged.window_count += audit.window_count
        merged.event_window_count += audit.event_window_count
        merged.interval_window_count += audit.interval_window_count
        merged.event_count += audit.event_count
        merged.event_interval_count += audit.event_interval_count
        merged.v3_reconstruction_mismatches += audit.v3_reconstruction_mismatches
        merged.time_shift_parity_mismatches += audit.time_shift_parity_mismatches
        merged.v2_1_time_shift_counts.update(audit.v2_1_time_shift_counts)
        merged.v3_time_shift_counts.update(audit.v3_time_shift_counts)
        merged.event_interval_counts.update(audit.event_interval_counts)
        merged.window_dominant_interval_ratios.extend(audit.window_dominant_interval_ratios)
        merged.mismatch_examples.extend(audit.mismatch_examples[: max(0, 5 - len(merged.mismatch_examples))])
    return merged


def load_comparators(
    *,
    v3_multicase_summary_path: Path,
    v3_decode_policy_summary_path: Path,
    v21_matched_summary_path: Path,
    generated_state_summary_path: Path,
    v3_full_dataset_summary_path: Path,
    c3_target_complexity_summary_path: Path,
    factorized_event_summary_path: Path,
) -> dict[str, Any]:
    v3_multicase = _load_optional_json(v3_multicase_summary_path)
    v3_decode_policy = _load_optional_json(v3_decode_policy_summary_path)
    matched_v21 = _load_optional_json(v21_matched_summary_path)
    generated_state = _load_optional_json(generated_state_summary_path)
    v3_full_dataset = _load_optional_json(v3_full_dataset_summary_path)
    c3_target_complexity = _load_optional_json(c3_target_complexity_summary_path)
    factorized_event = _load_optional_json(factorized_event_summary_path)
    return {
        "v3_multicase": _generated_summary_snapshot(v3_multicase, path=v3_multicase_summary_path),
        "v3_decode_policy": {
            "path": v3_decode_policy_summary_path.as_posix(),
            "available": v3_decode_policy is not None,
            "decision_route": _decision_route(v3_decode_policy),
        },
        "matched_v2_1": _generated_summary_snapshot(matched_v21, path=v21_matched_summary_path),
        "generated_state": {
            "path": generated_state_summary_path.as_posix(),
            "available": generated_state is not None,
            "decision_route": _decision_route(generated_state),
            "baseline_failure_counts": ((generated_state or {}).get("baseline_snapshot") or {}).get("failure_counts", {}),
        },
        "v3_full_dataset": {
            "path": v3_full_dataset_summary_path.as_posix(),
            "available": v3_full_dataset is not None,
            "decision_route": _decision_route(v3_full_dataset),
            "comparison": (v3_full_dataset or {}).get("comparison", {}),
        },
        "c3_target_complexity": {
            "path": c3_target_complexity_summary_path.as_posix(),
            "available": c3_target_complexity is not None,
            "decision_route": _decision_route(c3_target_complexity),
            "decision": (c3_target_complexity or {}).get("decision"),
        },
        "factorized_event": {
            "path": factorized_event_summary_path.as_posix(),
            "available": factorized_event is not None,
            "decision_route": _decision_route(factorized_event),
        },
    }


def comparison_from_audit(
    *,
    total_audit: TimingResidueAccumulator,
    comparators: Mapping[str, Any],
) -> dict[str, Any]:
    total = total_audit.to_dict()
    interval = total["event_intervals"]
    window = interval["dominant_window_ratio"]
    v3_generated_ratio = comparators["v3_multicase"].get("mean_dominant_spacing_ratio")
    v21_generated_ratio = comparators["matched_v2_1"].get("mean_dominant_spacing_ratio")
    generated_rigid = bool(
        comparators["v3_multicase"].get("systematic_rigid_grid")
        or (
            isinstance(v3_generated_ratio, (int, float))
            and float(v3_generated_ratio) >= GENERATED_RIGID_RATIO_THRESHOLD
        )
    ) and bool(
        comparators["matched_v2_1"].get("systematic_rigid_grid")
        or (
            isinstance(v21_generated_ratio, (int, float))
            and float(v21_generated_ratio) >= GENERATED_RIGID_RATIO_THRESHOLD
        )
    )
    missing = [
        key
        for key in (
            "v3_multicase",
            "v3_decode_policy",
            "matched_v2_1",
            "generated_state",
            "v3_full_dataset",
            "c3_target_complexity",
            "factorized_event",
        )
        if not bool(comparators[key].get("available"))
    ]
    return {
        "time_shift_parity_mismatches": total_audit.time_shift_parity_mismatches,
        "v3_reconstruction_mismatches": total_audit.v3_reconstruction_mismatches,
        "event_interval_effective_vocab": interval["effective_vocab"],
        "event_interval_entropy_bits": interval["entropy_bits"],
        "event_interval_top1_share": interval["top1_share"],
        "event_interval_top3_share": interval["top3_share"],
        "event_interval_160_200_share": interval["share_160_200ms"],
        "event_interval_le_200_share": interval["share_le_200ms"],
        "target_rigid_window_share_ge_0_95": window["share_ge_0_95"],
        "target_rigid_window_share_ge_0_90": window["share_ge_0_90"],
        "target_mean_window_dominant_interval_ratio": window["mean"],
        "generated_v3_mean_dominant_spacing_ratio": v3_generated_ratio,
        "generated_v21_mean_dominant_spacing_ratio": v21_generated_ratio,
        "generated_v3_systematic_rigid_grid": comparators["v3_multicase"].get("systematic_rigid_grid"),
        "generated_v21_systematic_rigid_grid": comparators["matched_v2_1"].get("systematic_rigid_grid"),
        "generated_rigid_evidence": generated_rigid,
        "missing_comparator_keys": missing,
    }


def checks_from_comparison(comparison: Mapping[str, Any]) -> dict[str, bool]:
    effective_vocab = float(comparison.get("event_interval_effective_vocab") or 0.0)
    interval_160_200 = float(comparison.get("event_interval_160_200_share") or 1.0)
    rigid_95 = float(comparison.get("target_rigid_window_share_ge_0_95") or 1.0)
    target_rich = (
        effective_vocab >= EFFECTIVE_INTERVAL_VOCAB_THRESHOLD
        and interval_160_200 <= INTERVAL_160_200_SHARE_THRESHOLD
        and rigid_95 <= RIGID_WINDOW_95_SHARE_THRESHOLD
    )
    return {
        "time_shift_parity_exact": int(comparison["time_shift_parity_mismatches"]) == 0,
        "v3_reconstruction_exact": int(comparison["v3_reconstruction_mismatches"]) == 0,
        "comparators_available": not comparison["missing_comparator_keys"],
        "generated_rigid_evidence_available": bool(comparison["generated_rigid_evidence"]),
        "target_timing_residue_rich": target_rich,
    }


def decision_from_comparison(comparison: Mapping[str, Any]) -> dict[str, Any]:
    checks = checks_from_comparison(comparison)
    failed_checks = [name for name, passed in checks.items() if not passed]
    kill_criteria: list[str] = []
    if not checks["time_shift_parity_exact"]:
        kill_criteria.append("time_shift_parity_mismatch")
    if not checks["v3_reconstruction_exact"]:
        kill_criteria.append("v3_reconstruction_mismatch")
    if not checks["comparators_available"]:
        kill_criteria.append("missing_generated_comparator")
    if kill_criteria:
        return {
            "route": "KILL_TIMING_RESIDUE_AUDIT_INPUTS",
            "reason": "timing-residue audit failed an input, parity, or reconstruction guard",
            "positive_gate_passed": False,
            "kill_criteria_triggered": True,
            "kill_criteria": kill_criteria,
            "failed_checks": failed_checks,
            "interpretation": (
                "The audit cannot route timing repair until target streams and comparator summaries are intact "
                "and v2.1/v3 timing parity is exact."
            ),
            "next_step": "Fix audit inputs or conversion parity before timing calibration work.",
        }
    if checks["target_timing_residue_rich"] and checks["generated_rigid_evidence_available"]:
        return {
            "route": "TEST_TIMING_LOSS_OR_EMBEDDING_CALIBRATION",
            "reason": "teacher-forcing targets preserve richer timing residue than the generated rigid-grid rollouts",
            "positive_gate_passed": True,
            "kill_criteria_triggered": False,
            "kill_criteria": [],
            "failed_checks": failed_checks,
            "interpretation": (
                "The target streams already contain timing diversity, and v3 preserves the same time-shift "
                "sequence as v2.1. The generated rigid grids therefore look more like calibration/exposure "
                "failure than an event-token representation failure."
            ),
            "next_step": "Create a bounded timing loss or timing-embedding calibration card with rigid-grid and second-window guards.",
        }
    if not checks["target_timing_residue_rich"]:
        return {
            "route": "MUTATE_TO_TIMING_GRAMMAR_REPAIR",
            "reason": "teacher-forcing target timing residue is too grid-dominated for a loss-only next step",
            "positive_gate_passed": False,
            "kill_criteria_triggered": True,
            "kill_criteria": ["target_timing_residue_not_rich"],
            "failed_checks": failed_checks,
            "interpretation": (
                "The target stream does not provide a strong enough contrast against generated rigid grids. "
                "A timing grammar or v2.1 grammar repair should be considered before another loss-only run."
            ),
            "next_step": "Mutate toward timing grammar repair or a v2.1 grammar repair card.",
        }
    return {
        "route": "TEST_TIMING_RESIDUE_BUCKET_DIAGNOSTIC",
        "reason": "target timing residue is rich, but generated rigid-grid comparator evidence is incomplete",
        "positive_gate_passed": False,
        "kill_criteria_triggered": False,
        "kill_criteria": [],
        "failed_checks": failed_checks,
        "interpretation": (
            "The target stream has timing diversity, but the generated comparator evidence is too weak for "
            "a calibration route. Split by difficulty/density before training."
        ),
        "next_step": "Run one timing-residue bucket diagnostic by density and difficulty.",
    }


def _generated_summary_snapshot(payload: Mapping[str, Any] | None, *, path: Path) -> dict[str, Any]:
    aggregate = (payload or {}).get("aggregate") or {}
    return {
        "path": path.as_posix(),
        "available": payload is not None,
        "decision_route": _decision_route(payload),
        "mean_dominant_spacing_ratio": _float(aggregate.get("mean_dominant_spacing_ratio")),
        "min_dominant_spacing_ratio": _float(aggregate.get("min_dominant_spacing_ratio")),
        "systematic_rigid_grid": bool(aggregate.get("systematic_rigid_grid", False)),
        "second_window_starved_case_count": _int(aggregate.get("second_window_starved_case_count")),
        "mean_f1_100ms": _float(aggregate.get("mean_f1_100ms")),
    }


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"summary must be a JSON object: {path}")
    return payload


def _decision_route(payload: Mapping[str, Any] | None) -> str | None:
    if payload is None:
        return None
    decision = payload.get("decision")
    if isinstance(decision, Mapping):
        route = decision.get("route")
        return None if route is None else str(route)
    if isinstance(decision, str):
        return decision
    return None


def _share(values: Sequence[float], *, threshold: float) -> float:
    return float(sum(1 for value in values if float(value) >= float(threshold))) / float(len(values))


def _float(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _fmt_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and not math.isfinite(value):
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return str(value)


def _fmt_pct(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and not math.isfinite(value):
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value) * 100.0:.2f}%"
    return str(value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit shared target timing residue for v2.1 and v3.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT_PATH)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT_PATH)
    parser.add_argument("--train-limit", type=parse_limit, default=4096)
    parser.add_argument("--eval-limit", type=parse_limit, default=None)
    parser.add_argument("--progress-interval", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_shared_timing_residue_structure_audit(
        config_path=args.config,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        progress_interval=args.progress_interval,
    )
    write_summary(summary, args.summary_output)
    if args.report_output is not None:
        write_markdown_report(summary, args.report_output)
    comparison = summary["comparison"]
    print(
        "target grammar shared timing-residue audit: "
        f"decision={summary['decision']['route']} "
        f"parity_mismatches={comparison['time_shift_parity_mismatches']} "
        f"v3_mismatches={comparison['v3_reconstruction_mismatches']} "
        f"interval_effective_vocab={_fmt_metric(comparison['event_interval_effective_vocab'])} "
        f"interval_160_200={_fmt_pct(comparison['event_interval_160_200_share'])}"
    )


if __name__ == "__main__":
    main()
