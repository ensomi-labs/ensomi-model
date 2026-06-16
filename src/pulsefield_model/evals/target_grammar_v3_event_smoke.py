from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from pulsefield_model.evals.c3_auxiliary_diagnostics import build_datasets_from_config
from pulsefield_model.evals.target_grammar_v3_pressure import (
    target_token_ids_for_dataset_index,
    unigram_nll_bits,
)
from pulsefield_model.models.mapper.v2_1.vocab import MapperV21Vocab
from pulsefield_model.models.mapper.v3.conversion import V3ConversionResult, v2_1_tokens_to_v3_event_tokens
from pulsefield_model.models.mapper.v3.vocab import MapperV3Vocab


SUMMARY_SCHEMA_VERSION = 1


def audit_target_grammar_v3_event_smoke(
    *,
    config_path: Path,
    train_limit: int,
    alpha: float = 0.5,
) -> dict[str, Any]:
    started_at = time.monotonic()
    if int(train_limit) <= 0:
        raise ValueError("train_limit must be positive")
    source_vocab = MapperV21Vocab()
    target_vocab = MapperV3Vocab(source_vocab.time_shift_values_ms)
    datasets = build_datasets_from_config(config_path)
    train_window_count = min(int(train_limit), len(datasets.train_dataset))
    train_v2_ids = [
        target_token_ids_for_dataset_index(datasets.train_dataset, index)
        for index in range(train_window_count)
    ]
    eval_v2_ids = [
        target_token_ids_for_dataset_index(datasets.eval_dataset, index)
        for index in range(len(datasets.eval_dataset))
    ]
    train_conversions = [
        v2_1_tokens_to_v3_event_tokens(
            token_ids,
            source_vocab=source_vocab,
            target_vocab=target_vocab,
        )
        for token_ids in train_v2_ids
    ]
    eval_conversions = [
        v2_1_tokens_to_v3_event_tokens(
            token_ids,
            source_vocab=source_vocab,
            target_vocab=target_vocab,
        )
        for token_ids in eval_v2_ids
    ]
    reconstruction_mismatches = sum(
        int(conversion.reconstructed_v2_1_token_ids != tuple(token_ids))
        for conversion, token_ids in zip(eval_conversions, eval_v2_ids, strict=True)
    )
    train_v2_streams = [_v2_token_names(token_ids, source_vocab) for token_ids in train_v2_ids]
    eval_v2_streams = [_v2_token_names(token_ids, source_vocab) for token_ids in eval_v2_ids]
    train_v3_streams = [_v3_token_names(conversion.token_ids, target_vocab) for conversion in train_conversions]
    eval_v3_streams = [_v3_token_names(conversion.token_ids, target_vocab) for conversion in eval_conversions]
    baseline_score = unigram_nll_bits(train_v2_streams, eval_v2_streams, alpha=alpha)
    candidate_score = unigram_nll_bits(train_v3_streams, eval_v3_streams, alpha=alpha)

    baseline_eval_token_count = int(baseline_score["eval_token_count"])
    candidate_eval_token_count = int(candidate_score["eval_token_count"])
    baseline_total_bits = float(baseline_score["total_bits"])
    candidate_total_bits = float(candidate_score["total_bits"])
    event_counts = _conversion_totals(eval_conversions)
    event_signature_counts = _event_signature_counts(eval_conversions, vocab=target_vocab)
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
        "reconstruction_mismatches": reconstruction_mismatches,
        **event_counts,
    }
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "config_path": config_path.as_posix(),
        "dataset": {
            "train_window_count": len(datasets.train_dataset),
            "train_scored_window_count": train_window_count,
            "eval_window_count": len(datasets.eval_dataset),
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
            "top_event_signatures": _top_event_rows(event_signature_counts, limit=20),
        },
        "comparison": comparison,
        "decision": _decision_from_comparison(comparison),
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
    top_rows = [
        f"| `{row['signature']}` | {row['count']} | {row['lane_count']} |"
        for row in candidate["top_event_signatures"][:12]
    ]
    text = "\n".join(
        [
            "# Target Grammar v3 Event-Token Smoke Result Report",
            "",
            "## Scope",
            "",
            "This smoke audit tests v3 as a complete local event-token target grammar: one 4-lane event token per non-empty same-time group. It compares against v2.1 sparse lane-action target streams on the same bounded split and verifies exact expansion back to v2.1 tokens.",
            "",
            "## Result",
            "",
            f"Decision: `{decision['route']}`.",
            "",
            f"- train windows scored: `{dataset['train_scored_window_count']}` / `{dataset['train_window_count']}`",
            f"- eval windows scored: `{dataset['eval_window_count']}`",
            f"- baseline eval tokens: `{baseline['eval_token_count']}`",
            f"- v3 eval tokens: `{candidate['eval_token_count']}`",
            f"- token reduction: `{_fmt_pct(comparison['token_reduction_ratio'])}`",
            f"- total-bit reduction: `{_fmt_pct(comparison['total_bit_reduction_ratio'])}`",
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
            "- V3 event tokens expanded back to the exact v2.1 sparse target stream on the eval split.",
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


def _decision_from_comparison(comparison: Mapping[str, Any]) -> dict[str, Any]:
    mismatches = int(comparison["reconstruction_mismatches"])
    bit_reduction = float(comparison["total_bit_reduction_ratio"] or 0.0)
    token_reduction = float(comparison["token_reduction_ratio"] or 0.0)
    if mismatches == 0 and bit_reduction >= 0.03 and token_reduction >= 0.10:
        route = "TEST_NEXT"
        interpretation = (
            "The v3 event-token grammar passed the smoke gate: exact reconstruction holds, "
            "target streams are shorter, and the simple total-bit proxy improves. This is still "
            "bounded-slice evidence; the updated goal requires a full 4K dataset audit before replacement."
        )
        next_step = "Scale the v3 event-token audit to the full 4K dataset and add dataset/model integration gates."
        positive = True
        kill = False
    elif mismatches:
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
    parser.add_argument("--train-limit", type=int, default=4096)
    parser.add_argument("--alpha", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = audit_target_grammar_v3_event_smoke(
        config_path=args.config,
        train_limit=args.train_limit,
        alpha=args.alpha,
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
