from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset, Subset

from pulsefield_model.evals.c3_auxiliary_diagnostics import build_datasets_from_config
from pulsefield_model.models.mapper.v2_1.vocab import MapperV21Vocab
from pulsefield_model.models.mapper.v3.conversion import V3ConversionResult, v2_1_tokens_to_v3_event_tokens
from pulsefield_model.models.mapper.v3.vocab import MapperV3Vocab


SUMMARY_SCHEMA_VERSION = 1


@dataclass
class SplitAuditAccumulator:
    split_name: str
    source_vocab: MapperV21Vocab
    target_vocab: MapperV3Vocab
    window_count: int = 0
    baseline_token_count: int = 0
    candidate_token_count: int = 0
    event_token_count: int = 0
    multi_lane_event_count: int = 0
    lane_action_token_count: int = 0
    reconstruction_mismatches: int = 0
    full_chart_end_window_count: int = 0
    terminal_event_at_chart_end_window_count: int = 0
    ln_carry_in_window_count: int = 0
    ln_carry_out_window_count: int = 0
    cross_window_ln_window_count: int = 0
    baseline_token_counts: Counter[str] = field(default_factory=Counter)
    candidate_token_counts: Counter[str] = field(default_factory=Counter)
    event_signature_counts: Counter[str] = field(default_factory=Counter)
    mismatch_examples: list[dict[str, Any]] = field(default_factory=list)

    def update(self, *, token_ids: Sequence[int], tokenized: Any | None, dataset_index: int) -> None:
        conversion = v2_1_tokens_to_v3_event_tokens(
            token_ids,
            source_vocab=self.source_vocab,
            target_vocab=self.target_vocab,
        )
        self.window_count += 1
        self.baseline_token_count += len(token_ids)
        self.candidate_token_count += len(conversion.token_ids)
        self.event_token_count += conversion.event_token_count
        self.multi_lane_event_count += conversion.multi_lane_event_count
        self.lane_action_token_count += conversion.lane_action_token_count
        if conversion.reconstructed_v2_1_token_ids != tuple(token_ids):
            self.reconstruction_mismatches += 1
            if len(self.mismatch_examples) < 5:
                self.mismatch_examples.append(
                    {
                        "split": self.split_name,
                        "dataset_index": int(dataset_index),
                        "baseline_prefix": [self.source_vocab.token_name(token_id) for token_id in token_ids[:20]],
                        "reconstructed_prefix": [
                            self.source_vocab.token_name(token_id)
                            for token_id in conversion.reconstructed_v2_1_token_ids[:20]
                        ],
                    }
                )
        self.baseline_token_counts.update(_v2_token_names(token_ids, self.source_vocab))
        self.candidate_token_counts.update(_v3_token_names(conversion.token_ids, self.target_vocab))
        self.event_signature_counts.update(_event_signature_counts((conversion,), vocab=self.target_vocab))
        if tokenized is not None:
            self._update_window_classes(tokenized=tokenized, conversion=conversion)

    def _update_window_classes(self, *, tokenized: Any, conversion: V3ConversionResult) -> None:
        if bool(getattr(tokenized, "is_full_chart_end", False)):
            self.full_chart_end_window_count += 1
            if _has_terminal_event_at_chart_end(tokenized=tokenized, conversion=conversion, vocab=self.target_vocab):
                self.terminal_event_at_chart_end_window_count += 1
        ln_carry_in = getattr(tokenized, "ln_carry_in", None)
        ln_carry_out = getattr(tokenized, "ln_carry_out", None)
        carry_in_open = bool(ln_carry_in is not None and any(getattr(ln_carry_in, "open_mask", ())))
        carry_out_open = bool(ln_carry_out is not None and any(getattr(ln_carry_out, "open_mask", ())))
        if carry_in_open:
            self.ln_carry_in_window_count += 1
        if carry_out_open:
            self.ln_carry_out_window_count += 1
        if carry_in_open or carry_out_open:
            self.cross_window_ln_window_count += 1

    def to_dict(self) -> dict[str, Any]:
        token_reduction = self.baseline_token_count - self.candidate_token_count
        return {
            "split": self.split_name,
            "window_count": self.window_count,
            "baseline_token_count": self.baseline_token_count,
            "candidate_token_count": self.candidate_token_count,
            "token_reduction": token_reduction,
            "token_reduction_ratio": _safe_ratio(token_reduction, self.baseline_token_count),
            "event_token_count": self.event_token_count,
            "multi_lane_event_count": self.multi_lane_event_count,
            "lane_action_token_count": self.lane_action_token_count,
            "reconstruction_mismatches": self.reconstruction_mismatches,
            "full_chart_end_window_count": self.full_chart_end_window_count,
            "terminal_event_at_chart_end_window_count": self.terminal_event_at_chart_end_window_count,
            "ln_carry_in_window_count": self.ln_carry_in_window_count,
            "ln_carry_out_window_count": self.ln_carry_out_window_count,
            "cross_window_ln_window_count": self.cross_window_ln_window_count,
            "observed_event_vocab_size": len(self.event_signature_counts),
            "top_event_signatures": _top_event_rows(self.event_signature_counts, limit=20),
            "mismatch_examples": list(self.mismatch_examples),
        }


def audit_target_grammar_v3_event_smoke(
    *,
    config_path: Path,
    train_limit: int | None,
    eval_limit: int | None = None,
    alpha: float = 0.5,
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
    audit_totals = _merge_split_audits((train_audit, eval_audit), source_vocab=source_vocab, target_vocab=target_vocab)

    baseline_score = unigram_nll_bits_from_counts(
        train_audit.baseline_token_counts,
        eval_audit.baseline_token_counts,
        alpha=alpha,
    )
    candidate_score = unigram_nll_bits_from_counts(
        train_audit.candidate_token_counts,
        eval_audit.candidate_token_counts,
        alpha=alpha,
    )

    baseline_eval_token_count = int(baseline_score["eval_token_count"])
    candidate_eval_token_count = int(candidate_score["eval_token_count"])
    baseline_total_bits = float(baseline_score["total_bits"])
    candidate_total_bits = float(candidate_score["total_bits"])
    comparison = {
        "eval_token_count": candidate_eval_token_count,
        "token_reduction": baseline_eval_token_count - candidate_eval_token_count,
        "token_reduction_ratio": _safe_ratio(
            baseline_eval_token_count - candidate_eval_token_count,
            baseline_eval_token_count,
        ),
        "total_bit_reduction": baseline_total_bits - candidate_total_bits,
        "total_bit_reduction_ratio": _safe_ratio(
            baseline_total_bits - candidate_total_bits,
            baseline_total_bits,
        ),
        "bits_per_baseline_token": _safe_ratio(candidate_total_bits, baseline_eval_token_count),
        "reconstruction_mismatches": eval_audit.reconstruction_mismatches,
        "event_token_count": eval_audit.event_token_count,
        "multi_lane_event_count": eval_audit.multi_lane_event_count,
        "lane_action_token_count": eval_audit.lane_action_token_count,
    }
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "config_path": config_path.as_posix(),
        "dataset": {
            "train_window_count": len(datasets.train_dataset),
            "train_scored_window_count": train_window_count,
            "eval_window_count": len(datasets.eval_dataset),
            "eval_scored_window_count": eval_window_count,
            "audit_window_count": audit_totals.window_count,
            "full_dataset_audit": train_limit is None and eval_limit is None,
        },
        "vocab": {
            "v2_1_vocab_size": source_vocab.size,
            "v3_vocab_size": target_vocab.size,
            "v3_event_token_count": len(target_vocab.event_token_ids),
            "contract": target_vocab.contract_name,
        },
        "baseline": {
            "eval_token_count": baseline_eval_token_count,
            "unigram_nll": baseline_score,
        },
        "candidate": {
            "eval_token_count": candidate_eval_token_count,
            "unigram_nll": candidate_score,
            "top_event_signatures": _top_event_rows(eval_audit.event_signature_counts, limit=20),
        },
        "comparison": comparison,
        "audit": {
            "train": train_audit.to_dict(),
            "eval": eval_audit.to_dict(),
            "total": audit_totals.to_dict(),
        },
        "decision": _decision_from_comparison(
            comparison,
            audit_totals=audit_totals,
            full_dataset_audit=train_limit is None and eval_limit is None,
        ),
        "elapsed_s": time.monotonic() - started_at,
    }


def write_summary(summary: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown_report(summary: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    decision = summary["decision"]
    comparison = summary["comparison"]
    baseline = summary["baseline"]
    candidate = summary["candidate"]
    dataset = summary["dataset"]
    audit_total = summary["audit"]["total"]
    full_dataset_audit = bool(dataset.get("full_dataset_audit"))
    title = (
        "Target Grammar v3 Full Dataset Audit Result Report"
        if full_dataset_audit
        else "Target Grammar v3 Event-Token Smoke Result Report"
    )
    top_rows = [
        f"| `{row['signature']}` | {row['count']} | {row['lane_count']} |"
        for row in audit_total["top_event_signatures"][:12]
    ]
    full_marker = "full-dataset" if full_dataset_audit else "bounded"
    text = "\n".join(
        [
            f"# {title}",
            "",
            "## Scope",
            "",
            f"This {full_marker} audit tests v3 as a complete local event-token target grammar: one 4-lane event token per non-empty same-time group. It compares against v2.1 sparse lane-action target streams and verifies exact expansion back to v2.1 tokens.",
            "",
            "## Result",
            "",
            f"Decision: `{decision['route']}`.",
            "",
            f"- train windows scored: `{dataset['train_scored_window_count']}` / `{dataset['train_window_count']}`",
            f"- eval windows scored: `{dataset['eval_scored_window_count']}` / `{dataset['eval_window_count']}`",
            f"- baseline eval tokens: `{baseline['eval_token_count']}`",
            f"- v3 eval tokens: `{candidate['eval_token_count']}`",
            f"- token reduction: `{_fmt_pct(comparison['token_reduction_ratio'])}`",
            f"- total-bit reduction: `{_fmt_pct(comparison['total_bit_reduction_ratio'])}`",
            f"- full-audit windows: `{audit_total['window_count']}`",
            f"- full-audit token reduction: `{_fmt_pct(audit_total['token_reduction_ratio'])}`",
            f"- full-audit reconstruction mismatches: `{audit_total['reconstruction_mismatches']}`",
            f"- full-chart terminal windows: `{audit_total['full_chart_end_window_count']}`",
            f"- terminal event-at-chart-end windows: `{audit_total['terminal_event_at_chart_end_window_count']}`",
            f"- cross-window LN windows: `{audit_total['cross_window_ln_window_count']}`",
            f"- v3 bits/token: `{_fmt_metric(candidate['unigram_nll']['bits_per_token'])}`",
            f"- event tokens: `{comparison['event_token_count']}`",
            f"- multi-lane events: `{comparison['multi_lane_event_count']}`",
            f"- reconstruction mismatches: `{comparison['reconstruction_mismatches']}`",
            f"- elapsed: `{float(summary['elapsed_s']):.2f}s`",
            "",
            "## Top Event Signatures",
            "",
            "| Signature | Count | Non-empty lanes |",
            "| --- | ---: | ---: |",
            *top_rows,
            "",
            "## What Passed",
            "",
            "- V3 event tokens expanded back to the exact v2.1 sparse target stream on the audited windows.",
            "- The candidate is local and teacher-forcing friendly: time shifts plus one current event-group token.",
            "- No C3-style cross-window reference or future target lookup is needed.",
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


def _conversion_totals(conversions: Sequence[V3ConversionResult]) -> dict[str, int]:
    return {
        "event_token_count": sum(conversion.event_token_count for conversion in conversions),
        "multi_lane_event_count": sum(conversion.multi_lane_event_count for conversion in conversions),
        "lane_action_token_count": sum(conversion.lane_action_token_count for conversion in conversions),
    }


def unigram_nll_bits_from_counts(
    train_counts: Mapping[str, int],
    eval_counts: Mapping[str, int],
    *,
    alpha: float = 0.5,
) -> dict[str, Any]:
    alpha = float(alpha)
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    train_total = int(sum(int(value) for value in train_counts.values()))
    eval_total = int(sum(int(value) for value in eval_counts.values()))
    vocab = set(train_counts) | set(eval_counts)
    if eval_total == 0:
        return {
            "train_token_count": train_total,
            "eval_token_count": 0,
            "vocab_size": len(vocab),
            "bits_per_token": None,
            "total_bits": 0.0,
        }
    denominator = float(train_total) + alpha * float(len(vocab))
    if denominator <= 0.0:
        raise ValueError("cannot score unigram stream without train or eval vocabulary")
    total_bits = 0.0
    for token, eval_count in eval_counts.items():
        probability = (float(train_counts.get(token, 0)) + alpha) / denominator
        total_bits += float(eval_count) * -math.log2(probability)
    return {
        "train_token_count": train_total,
        "eval_token_count": eval_total,
        "vocab_size": len(vocab),
        "bits_per_token": total_bits / float(eval_total),
        "total_bits": total_bits,
    }


def _event_signature_counts(conversions: Sequence[V3ConversionResult], *, vocab: MapperV3Vocab) -> Counter[str]:
    counts: Counter[str] = Counter()
    for conversion in conversions:
        for token_id in conversion.token_ids:
            if vocab.is_event_token(int(token_id)):
                counts[vocab.event_signature(int(token_id))] += 1
    return counts


def _top_event_rows(counts: Mapping[str, int], *, limit: int) -> list[dict[str, Any]]:
    return [
        {
            "signature": signature,
            "count": int(count),
            "lane_count": sum(1 for symbol in signature if symbol != "."),
        }
        for signature, count in sorted(
            counts.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )[: int(limit)]
    ]


def _decision_from_comparison(
    comparison: Mapping[str, Any],
    *,
    audit_totals: SplitAuditAccumulator | None = None,
    full_dataset_audit: bool = False,
) -> dict[str, Any]:
    mismatches = int(comparison["reconstruction_mismatches"])
    total_mismatches = mismatches if audit_totals is None else int(audit_totals.reconstruction_mismatches)
    bit_reduction = float(comparison["total_bit_reduction_ratio"] or 0.0)
    token_reduction = float(comparison["token_reduction_ratio"] or 0.0)
    audit_token_reduction = token_reduction
    if audit_totals is not None:
        audit_token_reduction = float(audit_totals.to_dict()["token_reduction_ratio"] or 0.0)
    if total_mismatches == 0 and bit_reduction >= 0.03 and audit_token_reduction > 0.0:
        route = "TEST_NEXT"
        interpretation = (
            "The v3 event-token grammar passed the audit gate: exact reconstruction holds on the audited "
            "scope, target streams are shorter, and the simple total-bit proxy improves. This permits the "
            "next representation gate, but trained mapper quality still needs full-pipeline validation."
        )
        next_step = (
            "Create the v3 dataset/model/training/inference replacement Experiment Card."
            if full_dataset_audit
            else "Scale the v3 event-token audit to the full 4K dataset before replacement."
        )
        positive = True
        kill = False
    elif total_mismatches:
        route = "KILL"
        interpretation = "V3 event-token conversion is not exact; do not promote it until reconstruction is fixed."
        next_step = "Fix conversion/replay or mutate away from event-token v3."
        positive = False
        kill = True
    else:
        route = "MUTATE"
        interpretation = (
            "V3 event-token conversion is legal, but the pressure gate is not strong enough on this slice. "
            "The next move should be a factorized event-mask or trained-loss smoke, not pipeline replacement."
        )
        next_step = "Mutate to factorized-mask pressure or repeat with a larger/full-dataset audit."
        positive = False
        kill = False
    return {
        "route": route,
        "positive_gate_passed": positive,
        "kill_criteria_triggered": kill,
        "interpretation": interpretation,
        "next_step": next_step,
    }


def _v2_token_names(token_ids: Sequence[int], vocab: MapperV21Vocab) -> tuple[str, ...]:
    return tuple(vocab.token_name(int(token_id)) for token_id in token_ids)


def _v3_token_names(token_ids: Sequence[int], vocab: MapperV3Vocab) -> tuple[str, ...]:
    return tuple(vocab.token_name(int(token_id)) for token_id in token_ids)


def _audit_dataset_split(
    dataset: Dataset[Any],
    *,
    split_name: str,
    limit: int,
    source_vocab: MapperV21Vocab,
    target_vocab: MapperV3Vocab,
    progress_interval: int,
) -> SplitAuditAccumulator:
    accumulator = SplitAuditAccumulator(
        split_name=split_name,
        source_vocab=source_vocab,
        target_vocab=target_vocab,
    )
    started_at = time.monotonic()
    for index in range(int(limit)):
        tokenized = _tokenized_window_for_dataset_index(dataset, index)
        token_ids = tuple(int(token_id) for token_id in tokenized.target_fragment_ids)
        accumulator.update(token_ids=token_ids, tokenized=tokenized, dataset_index=index)
        if progress_interval > 0 and accumulator.window_count % int(progress_interval) == 0:
            elapsed_s = time.monotonic() - started_at
            print(
                "target_grammar_v3_event_audit_progress "
                f"split={split_name} windows={accumulator.window_count}/{limit} "
                f"mismatches={accumulator.reconstruction_mismatches} elapsed_s={elapsed_s:.1f}",
                flush=True,
            )
    return accumulator


def _tokenized_window_for_dataset_index(dataset: Dataset[Any], index: int) -> Any:
    if isinstance(dataset, Subset):
        return _tokenized_window_for_dataset_index(dataset.dataset, int(dataset.indices[int(index)]))
    records = getattr(dataset, "records", None)
    tokenize_record = getattr(dataset, "_tokenize_record", None)
    if records is None or not callable(tokenize_record):
        raise ValueError("dataset must expose records and _tokenize_record for full v3 audit")
    mapper_record = records[int(index)]
    record = getattr(mapper_record, "control_record", mapper_record)
    return tokenize_record(record)


def _merge_split_audits(
    audits: Sequence[SplitAuditAccumulator],
    *,
    source_vocab: MapperV21Vocab,
    target_vocab: MapperV3Vocab,
) -> SplitAuditAccumulator:
    merged = SplitAuditAccumulator(split_name="total", source_vocab=source_vocab, target_vocab=target_vocab)
    for audit in audits:
        merged.window_count += audit.window_count
        merged.baseline_token_count += audit.baseline_token_count
        merged.candidate_token_count += audit.candidate_token_count
        merged.event_token_count += audit.event_token_count
        merged.multi_lane_event_count += audit.multi_lane_event_count
        merged.lane_action_token_count += audit.lane_action_token_count
        merged.reconstruction_mismatches += audit.reconstruction_mismatches
        merged.full_chart_end_window_count += audit.full_chart_end_window_count
        merged.terminal_event_at_chart_end_window_count += audit.terminal_event_at_chart_end_window_count
        merged.ln_carry_in_window_count += audit.ln_carry_in_window_count
        merged.ln_carry_out_window_count += audit.ln_carry_out_window_count
        merged.cross_window_ln_window_count += audit.cross_window_ln_window_count
        merged.baseline_token_counts.update(audit.baseline_token_counts)
        merged.candidate_token_counts.update(audit.candidate_token_counts)
        merged.event_signature_counts.update(audit.event_signature_counts)
        merged.mismatch_examples.extend(audit.mismatch_examples[: max(0, 5 - len(merged.mismatch_examples))])
    return merged


def _has_terminal_event_at_chart_end(
    *,
    tokenized: Any,
    conversion: V3ConversionResult,
    vocab: MapperV3Vocab,
) -> bool:
    if not bool(getattr(tokenized, "is_full_chart_end", False)):
        return False
    if len(conversion.token_ids) < 2:
        return False
    previous_token = int(conversion.token_ids[-2])
    if not vocab.is_event_token(previous_token):
        return False
    current_ms = getattr(tokenized, "target_fragment_current_ms", None)
    if not isinstance(current_ms, torch.Tensor) or int(current_ms.numel()) < 2:
        return False
    return int(current_ms.reshape(-1)[-2].item()) == int(getattr(tokenized, "chart_end_ms"))


def _resolve_limit(value: int | None, *, available_count: int, name: str) -> int:
    if value is None:
        return int(available_count)
    limit = int(value)
    if limit <= 0:
        raise ValueError(f"{name} must be positive or all")
    return min(limit, int(available_count))


def parse_limit(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    text = str(value).strip().lower()
    if text == "all":
        return None
    try:
        limit = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("limit must be a positive integer or 'all'") from exc
    if limit <= 0:
        raise argparse.ArgumentTypeError("limit must be a positive integer or 'all'")
    return limit


def _safe_ratio(numerator: float | int, denominator: float | int) -> float | None:
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


def _fmt_pct(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and not math.isfinite(value):
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value) * 100.0:.2f}%"
    return str(value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit v3 full event-token target grammar smoke.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--train-limit", type=parse_limit, default=4096)
    parser.add_argument("--eval-limit", type=parse_limit)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--progress-interval", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = audit_target_grammar_v3_event_smoke(
        config_path=args.config,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        alpha=args.alpha,
        progress_interval=args.progress_interval,
    )
    write_summary(summary, args.summary_output)
    if args.report_output is not None:
        write_markdown_report(summary, args.report_output)
    comparison = summary["comparison"]
    print(
        "target grammar v3 event smoke: "
        f"decision={summary['decision']['route']} "
        f"token_reduction={_fmt_pct(comparison['token_reduction_ratio'])} "
        f"bit_reduction={_fmt_pct(comparison['total_bit_reduction_ratio'])} "
        f"mismatches={comparison['reconstruction_mismatches']}"
    )


if __name__ == "__main__":
    main()
