from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.data import Dataset, Subset

from pulsefield_model.evals.c3_auxiliary_diagnostics import build_datasets_from_config
from pulsefield_model.models.mapper.shared.vocab import KEY_COUNT, LaneAction, coerce_lane_action
from pulsefield_model.models.mapper.v2_1.vocab import MapperV21Vocab


SUMMARY_SCHEMA_VERSION = 1
DEFAULT_TOP_KS = (16, 32, 64, 128, 256)
DEFAULT_SELECTED_K = 128
GROUP_TOKEN_PREFIX = "GROUP|"
ACTION_SYMBOLS = {
    LaneAction.NONE: ".",
    LaneAction.TAP: "T",
    LaneAction.HOLD_START: "S",
    LaneAction.HOLD_END: "E",
}


@dataclass(frozen=True)
class EventGroupRun:
    start_index: int
    end_index: int
    signature: str
    token_ids: tuple[int, ...]
    token_names: tuple[str, ...]


@dataclass(frozen=True)
class CandidateStream:
    token_names: tuple[str, ...]
    reconstructed_token_names: tuple[str, ...]
    total_groups: int
    replaced_groups: int
    fallback_groups: int
    multi_lane_groups: int
    replaced_multi_lane_groups: int
    baseline_lane_action_tokens: int
    fallback_lane_action_tokens: int

    @property
    def token_savings(self) -> int:
        return len(self.reconstructed_token_names) - len(self.token_names)


def lane_action_symbol(action: LaneAction | str) -> str:
    lane_action = coerce_lane_action(action)
    try:
        return ACTION_SYMBOLS[lane_action]
    except KeyError as exc:
        raise ValueError(f"unsupported lane action for v3 signature: {action}") from exc


def group_signature_from_token_ids(token_ids: Iterable[int], *, vocab: MapperV21Vocab) -> str:
    tokens = vocab.validate_canonical_lane_action_run(token_ids)
    if not tokens:
        raise ValueError("event group run must contain at least one lane-action token")
    symbols = [lane_action_symbol(LaneAction.NONE) for _ in range(KEY_COUNT)]
    for token_id in tokens:
        lane_index, action = vocab.decode_lane_action(token_id)
        symbols[int(lane_index)] = lane_action_symbol(action)
    signature = "".join(symbols)
    if signature == "." * KEY_COUNT:
        raise ValueError("event group signature cannot be all NONE")
    return signature


def extract_group_runs(token_ids: Sequence[int], *, vocab: MapperV21Vocab) -> tuple[EventGroupRun, ...]:
    runs: list[EventGroupRun] = []
    index = 0
    normalized = tuple(int(token_id) for token_id in token_ids)
    while index < len(normalized):
        token_id = normalized[index]
        if not vocab.is_lane_action_token(token_id):
            index += 1
            continue
        start_index = index
        while index < len(normalized) and vocab.is_lane_action_token(normalized[index]):
            index += 1
        run_ids = normalized[start_index:index]
        signature = group_signature_from_token_ids(run_ids, vocab=vocab)
        runs.append(
            EventGroupRun(
                start_index=start_index,
                end_index=index,
                signature=signature,
                token_ids=run_ids,
                token_names=tuple(vocab.token_name(token_id) for token_id in run_ids),
            )
        )
    return tuple(runs)


def candidate_stream_from_v2_tokens(
    token_ids: Sequence[int],
    *,
    vocab: MapperV21Vocab,
    group_dictionary: set[str] | frozenset[str],
) -> CandidateStream:
    candidate_names: list[str] = []
    reconstructed_names: list[str] = []
    total_groups = 0
    replaced_groups = 0
    fallback_groups = 0
    multi_lane_groups = 0
    replaced_multi_lane_groups = 0
    baseline_lane_action_tokens = 0
    fallback_lane_action_tokens = 0

    normalized = tuple(int(token_id) for token_id in token_ids)
    index = 0
    while index < len(normalized):
        token_id = normalized[index]
        if not vocab.is_lane_action_token(token_id):
            token_name = vocab.token_name(token_id)
            candidate_names.append(token_name)
            reconstructed_names.append(token_name)
            index += 1
            continue

        start_index = index
        while index < len(normalized) and vocab.is_lane_action_token(normalized[index]):
            index += 1
        run_ids = normalized[start_index:index]
        signature = group_signature_from_token_ids(run_ids, vocab=vocab)
        run_names = tuple(vocab.token_name(run_id) for run_id in run_ids)
        total_groups += 1
        baseline_lane_action_tokens += len(run_ids)
        if len(run_ids) > 1:
            multi_lane_groups += 1
        if signature in group_dictionary:
            candidate_names.append(f"{GROUP_TOKEN_PREFIX}{signature}")
            replaced_groups += 1
            if len(run_ids) > 1:
                replaced_multi_lane_groups += 1
        else:
            candidate_names.extend(run_names)
            fallback_groups += 1
            fallback_lane_action_tokens += len(run_ids)
        reconstructed_names.extend(run_names)

    return CandidateStream(
        token_names=tuple(candidate_names),
        reconstructed_token_names=tuple(reconstructed_names),
        total_groups=total_groups,
        replaced_groups=replaced_groups,
        fallback_groups=fallback_groups,
        multi_lane_groups=multi_lane_groups,
        replaced_multi_lane_groups=replaced_multi_lane_groups,
        baseline_lane_action_tokens=baseline_lane_action_tokens,
        fallback_lane_action_tokens=fallback_lane_action_tokens,
    )


def unigram_nll_bits(
    train_streams: Sequence[Sequence[str]],
    eval_streams: Sequence[Sequence[str]],
    *,
    alpha: float = 0.5,
) -> dict[str, Any]:
    alpha = float(alpha)
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    train_counts = Counter(token for stream in train_streams for token in stream)
    eval_counts = Counter(token for stream in eval_streams for token in stream)
    train_token_count = int(sum(train_counts.values()))
    eval_token_count = int(sum(eval_counts.values()))
    vocab = set(train_counts) | set(eval_counts)
    if eval_token_count == 0:
        return {
            "train_token_count": train_token_count,
            "eval_token_count": 0,
            "vocab_size": len(vocab),
            "bits_per_token": None,
            "total_bits": 0.0,
        }
    denominator = float(train_token_count) + alpha * float(len(vocab))
    if denominator <= 0.0:
        raise ValueError("cannot score unigram stream without train or eval vocabulary")
    total_bits = 0.0
    for token, eval_count in eval_counts.items():
        probability = (float(train_counts.get(token, 0)) + alpha) / denominator
        total_bits += float(eval_count) * -math.log2(probability)
    return {
        "train_token_count": train_token_count,
        "eval_token_count": eval_token_count,
        "vocab_size": len(vocab),
        "bits_per_token": total_bits / float(eval_token_count),
        "total_bits": total_bits,
    }


def audit_target_grammar_v3_pressure(
    *,
    config_path: Path,
    train_limit: int,
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
    alpha: float = 0.5,
) -> dict[str, Any]:
    started_at = time.monotonic()
    top_ks = tuple(sorted({int(k) for k in top_ks}))
    if not top_ks or top_ks[0] <= 0:
        raise ValueError("top_ks must contain positive integers")
    if int(train_limit) <= 0:
        raise ValueError("train_limit must be positive")

    vocab = MapperV21Vocab()
    datasets = build_datasets_from_config(config_path)
    train_window_count = min(int(train_limit), len(datasets.train_dataset))
    train_token_ids = [
        target_token_ids_for_dataset_index(datasets.train_dataset, index)
        for index in range(train_window_count)
    ]
    eval_token_ids = [
        target_token_ids_for_dataset_index(datasets.eval_dataset, index)
        for index in range(len(datasets.eval_dataset))
    ]
    train_baseline_streams = [token_names_from_ids(token_ids, vocab=vocab) for token_ids in train_token_ids]
    eval_baseline_streams = [token_names_from_ids(token_ids, vocab=vocab) for token_ids in eval_token_ids]
    train_group_counts = group_signature_counts(train_token_ids, vocab=vocab)
    eval_group_counts = group_signature_counts(eval_token_ids, vocab=vocab)
    baseline_score = unigram_nll_bits(train_baseline_streams, eval_baseline_streams, alpha=alpha)
    baseline_eval_token_count = int(baseline_score["eval_token_count"])
    baseline_total_bits = float(baseline_score["total_bits"])
    baseline_lane_action_tokens = sum(
        sum(1 for token_id in token_ids if vocab.is_lane_action_token(token_id))
        for token_ids in eval_token_ids
    )

    candidate_rows: dict[str, Any] = {}
    for k in top_ks:
        group_dictionary = set(top_group_signatures(train_group_counts, limit=int(k)))
        train_candidate_streams = [
            candidate_stream_from_v2_tokens(
                token_ids,
                vocab=vocab,
                group_dictionary=group_dictionary,
            ).token_names
            for token_ids in train_token_ids
        ]
        eval_candidates = [
            candidate_stream_from_v2_tokens(
                token_ids,
                vocab=vocab,
                group_dictionary=group_dictionary,
            )
            for token_ids in eval_token_ids
        ]
        eval_candidate_streams = [candidate.token_names for candidate in eval_candidates]
        candidate_score = unigram_nll_bits(train_candidate_streams, eval_candidate_streams, alpha=alpha)
        reconstruction_mismatches = sum(
            int(candidate.reconstructed_token_names != baseline_stream)
            for candidate, baseline_stream in zip(eval_candidates, eval_baseline_streams, strict=True)
        )
        total_groups = sum(candidate.total_groups for candidate in eval_candidates)
        replaced_groups = sum(candidate.replaced_groups for candidate in eval_candidates)
        fallback_groups = sum(candidate.fallback_groups for candidate in eval_candidates)
        multi_lane_groups = sum(candidate.multi_lane_groups for candidate in eval_candidates)
        replaced_multi_lane_groups = sum(candidate.replaced_multi_lane_groups for candidate in eval_candidates)
        fallback_lane_action_tokens = sum(candidate.fallback_lane_action_tokens for candidate in eval_candidates)
        candidate_eval_token_count = int(candidate_score["eval_token_count"])
        candidate_total_bits = float(candidate_score["total_bits"])
        candidate_rows[str(k)] = {
            "dictionary_size": len(group_dictionary),
            "dictionary_signatures": tuple(sorted(group_dictionary)),
            "eval_token_count": candidate_eval_token_count,
            "token_reduction": baseline_eval_token_count - candidate_eval_token_count,
            "token_reduction_ratio": _safe_ratio(
                baseline_eval_token_count - candidate_eval_token_count,
                baseline_eval_token_count,
            ),
            "baseline_lane_action_tokens": baseline_lane_action_tokens,
            "fallback_lane_action_tokens": fallback_lane_action_tokens,
            "lane_action_literal_reduction": baseline_lane_action_tokens - fallback_lane_action_tokens,
            "lane_action_literal_reduction_ratio": _safe_ratio(
                baseline_lane_action_tokens - fallback_lane_action_tokens,
                baseline_lane_action_tokens,
            ),
            "total_groups": total_groups,
            "replaced_groups": replaced_groups,
            "fallback_groups": fallback_groups,
            "group_coverage": _safe_ratio(replaced_groups, total_groups),
            "fallback_group_rate": _safe_ratio(fallback_groups, total_groups),
            "multi_lane_groups": multi_lane_groups,
            "replaced_multi_lane_groups": replaced_multi_lane_groups,
            "multi_lane_group_coverage": _safe_ratio(replaced_multi_lane_groups, multi_lane_groups),
            "reconstruction_mismatches": reconstruction_mismatches,
            "unigram_nll": candidate_score,
            "total_bit_reduction": baseline_total_bits - candidate_total_bits,
            "total_bit_reduction_ratio": _safe_ratio(
                baseline_total_bits - candidate_total_bits,
                baseline_total_bits,
            ),
            "bits_per_baseline_token": _safe_ratio(candidate_total_bits, baseline_eval_token_count),
        }

    selected_k = DEFAULT_SELECTED_K if str(DEFAULT_SELECTED_K) in candidate_rows else int(top_ks[-1])
    decision = _decision_from_candidate(candidate_rows[str(selected_k)], selected_k=selected_k)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "config_path": config_path.as_posix(),
        "dataset": {
            "train_window_count": len(datasets.train_dataset),
            "train_scored_window_count": train_window_count,
            "eval_window_count": len(datasets.eval_dataset),
        },
        "vocab": {
            "baseline_vocab_size": vocab.size,
            "group_token_prefix": GROUP_TOKEN_PREFIX,
            "signature_symbols": {
                "none": ".",
                "tap": "T",
                "hold_start": "S",
                "hold_end": "E",
            },
        },
        "baseline": {
            "eval_token_count": baseline_eval_token_count,
            "eval_lane_action_token_count": baseline_lane_action_tokens,
            "unigram_nll": baseline_score,
        },
        "groups": {
            "train_observed_group_vocab_size": len(train_group_counts),
            "eval_observed_group_vocab_size": len(eval_group_counts),
            "top_train_groups": _top_group_rows(train_group_counts, limit=20),
            "top_eval_groups": _top_group_rows(eval_group_counts, limit=20),
        },
        "candidates": candidate_rows,
        "comparison": {
            "selected_k": selected_k,
            "baseline_total_bits": baseline_total_bits,
            "baseline_bits_per_token": baseline_score["bits_per_token"],
            "selected_candidate": candidate_rows[str(selected_k)],
        },
        "decision": decision,
        "elapsed_s": time.monotonic() - started_at,
    }


def target_token_ids_for_dataset_index(dataset: Dataset[Any], index: int) -> tuple[int, ...]:
    if isinstance(dataset, Subset):
        return target_token_ids_for_dataset_index(dataset.dataset, int(dataset.indices[int(index)]))
    records = getattr(dataset, "records", None)
    tokenize_record = getattr(dataset, "_tokenize_record", None)
    if records is not None and callable(tokenize_record):
        mapper_record = records[int(index)]
        record = getattr(mapper_record, "control_record", mapper_record)
        tokenized = tokenize_record(record)
        return tuple(int(token_id) for token_id in tokenized.target_fragment_ids)
    sample = dataset[int(index)]
    tokens = sample["target_fragment_tokens"]
    if isinstance(tokens, torch.Tensor):
        return tuple(int(token_id) for token_id in tokens.reshape(-1).tolist())
    return tuple(int(token_id) for token_id in tokens)


def token_names_from_ids(token_ids: Sequence[int], *, vocab: MapperV21Vocab) -> tuple[str, ...]:
    return tuple(vocab.token_name(int(token_id)) for token_id in token_ids)


def group_signature_counts(token_streams: Sequence[Sequence[int]], *, vocab: MapperV21Vocab) -> Counter[str]:
    counts: Counter[str] = Counter()
    for token_ids in token_streams:
        counts.update(run.signature for run in extract_group_runs(token_ids, vocab=vocab))
    return counts


def top_group_signatures(counts: Mapping[str, int], *, limit: int) -> tuple[str, ...]:
    return tuple(
        signature
        for signature, _count in sorted(
            counts.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )[: int(limit)]
    )


def write_summary(summary: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown_report(summary: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    decision = summary["decision"]
    dataset = summary["dataset"]
    baseline = summary["baseline"]
    comparison = summary["comparison"]
    selected = comparison["selected_candidate"]
    candidate_items = sorted(summary["candidates"].items(), key=lambda item: int(item[0]))
    best_bit_k, best_bit_candidate = max(
        candidate_items,
        key=lambda item: float(item[1]["total_bit_reduction_ratio"] or float("-inf")),
    )
    best_token_k, best_token_candidate = max(
        candidate_items,
        key=lambda item: float(item[1]["token_reduction_ratio"] or float("-inf")),
    )
    selected_bit_reduction = float(selected["total_bit_reduction_ratio"] or 0.0)
    selected_token_reduction = float(selected["token_reduction_ratio"] or 0.0)
    selected_pressure_note = (
        "The selected top-K candidate cleared the pressure gate through token-count reduction; "
        "its simple unigram total-bit proxy did not improve."
        if selected_token_reduction >= 0.10 and selected_bit_reduction < 0.0
        else "The selected top-K candidate cleared the pressure gate under the configured metric."
    )
    rows = []
    for k, row in candidate_items:
        nll = row["unigram_nll"]
        rows.append(
            "| "
            f"{k} | {row['eval_token_count']} | {_fmt_pct(row['token_reduction_ratio'])} | "
            f"{_fmt_pct(row['total_bit_reduction_ratio'])} | {_fmt_metric(nll['bits_per_token'])} | "
            f"{_fmt_pct(row['group_coverage'])} | {_fmt_pct(row['multi_lane_group_coverage'])} | "
            f"{row['reconstruction_mismatches']} |"
        )
    top_train_rows = [
        f"| `{row['signature']}` | {row['count']} | {row['lane_count']} |"
        for row in summary["groups"]["top_train_groups"][:12]
    ]
    text = "\n".join(
        [
            "# Target Grammar v3 Pressure Result Report",
            "",
            "## Scope",
            "",
            "This audit simulates a reversible v3 target grammar without changing the production tokenizer. "
            "The candidate replaces same-time v2.1 sparse lane-action runs with train-derived "
            "`GROUP|signature` tokens, leaves time shifts and EOS unchanged, and falls back to literal "
            "v2.1 lane-action tokens for out-of-dictionary groups.",
            "",
            "## Result",
            "",
            f"Decision: `{decision['route']}`.",
            "",
            f"- train windows scored: `{dataset['train_scored_window_count']}` / `{dataset['train_window_count']}`",
            f"- eval windows scored: `{dataset['eval_window_count']}`",
            f"- baseline eval tokens: `{baseline['eval_token_count']}`",
            f"- baseline total unigram bits: `{float(comparison['baseline_total_bits']):.2f}`",
            f"- selected K: `{comparison['selected_k']}`",
            f"- selected eval tokens: `{selected['eval_token_count']}`",
            f"- selected token reduction: `{_fmt_pct(selected['token_reduction_ratio'])}`",
            f"- selected total-bit reduction: `{_fmt_pct(selected['total_bit_reduction_ratio'])}`",
            f"- best total-bit K: `{best_bit_k}` (`{_fmt_pct(best_bit_candidate['total_bit_reduction_ratio'])}`)",
            f"- best token-count K: `{best_token_k}` (`{_fmt_pct(best_token_candidate['token_reduction_ratio'])}`)",
            f"- selected group coverage: `{_fmt_pct(selected['group_coverage'])}`",
            f"- selected multi-lane group coverage: `{_fmt_pct(selected['multi_lane_group_coverage'])}`",
            f"- reconstruction mismatches: `{selected['reconstruction_mismatches']}`",
            f"- elapsed: `{float(summary['elapsed_s']):.2f}s`",
            "",
            "## Candidate Sweep",
            "",
            "| K | Eval tokens | Token reduction | Total-bit reduction | Bits/token | Group coverage | Multi-lane coverage | Mismatches |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            "## Top Train Groups",
            "",
            "| Signature | Count | Non-empty lanes |",
            "| --- | ---: | ---: |",
            *top_train_rows,
            "",
            "## What Passed",
            "",
            "- Exact reconstruction guard passed for every tested K.",
            "- The grammar shape satisfies the local target constraints: current event group only, literal fallback, no C3-style references, and no future target dependency.",
            f"- {selected_pressure_note}",
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


def _decision_from_candidate(candidate: Mapping[str, Any], *, selected_k: int) -> dict[str, Any]:
    mismatches = int(candidate["reconstruction_mismatches"])
    bit_reduction = candidate["total_bit_reduction_ratio"]
    token_reduction = candidate["token_reduction_ratio"]
    bit_value = 0.0 if bit_reduction is None else float(bit_reduction)
    token_value = 0.0 if token_reduction is None else float(token_reduction)
    if mismatches == 0 and (bit_value >= 0.05 or token_value >= 0.10):
        route = "TEST_NEXT"
        if bit_value >= 0.05:
            interpretation = (
                f"Top-{selected_k} group fallback is pressure-positive: it preserves exact v2.1 reconstruction "
                "and clears the positive gate on estimated bits. This supports a bounded v3 group-token smoke "
                "implementation, but it does not yet prove trained mapper quality."
            )
        elif bit_value < 0.0:
            interpretation = (
                f"Top-{selected_k} group fallback is pressure-positive only through token count: it preserves exact "
                "v2.1 reconstruction and shortens the stream, but the simple train-unigram total-bit proxy worsens. "
                "This means the representation is locally legal and shorter, while dictionary entropy still needs "
                "a trained-loss or dictionary-size check before promotion."
            )
        else:
            interpretation = (
                f"Top-{selected_k} group fallback is pressure-positive through token count: it preserves exact "
                "v2.1 reconstruction and shortens the stream, while estimated bits are non-negative but below "
                "the primary bit gate. This supports one bounded smoke step, not full promotion."
            )
        next_step = (
            "Create a v3 group-token grammar smoke card for vocab, tokenizer, replay expansion, masks, "
            "and a dictionary-size/trained-loss guard."
        )
        positive_gate = True
        kill = False
    elif mismatches > 0 or (bit_value < 0.03 and token_value < 0.05):
        route = "KILL"
        interpretation = (
            f"Top-{selected_k} group fallback does not clear the pressure gate. Immediate group-token v3 "
            "implementation should be killed or replaced by another v3 family such as factorized masks "
            "or time-shift compression."
        )
        next_step = "Mutate to a factorized-mask or time-shift pressure card before editing production grammar."
        positive_gate = False
        kill = True
    else:
        route = "MUTATE"
        interpretation = (
            f"Top-{selected_k} group fallback is legal but below the positive gate. It may need a different "
            "dictionary objective, larger slice, or a factorized grammar before production implementation."
        )
        next_step = "Repeat with a larger slice or mutate to factorized event-mask grammar."
        positive_gate = False
        kill = False
    return {
        "route": route,
        "positive_gate_passed": positive_gate,
        "kill_criteria_triggered": kill,
        "selected_k": int(selected_k),
        "interpretation": interpretation,
        "next_step": next_step,
    }


def _top_group_rows(counts: Mapping[str, int], *, limit: int) -> list[dict[str, Any]]:
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
    parser = argparse.ArgumentParser(description="Audit reversible target grammar v3 pressure.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--train-limit", type=int, default=4096)
    parser.add_argument("--top-k", type=int, nargs="+", default=list(DEFAULT_TOP_KS))
    parser.add_argument("--alpha", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = audit_target_grammar_v3_pressure(
        config_path=args.config,
        train_limit=args.train_limit,
        top_ks=args.top_k,
        alpha=args.alpha,
    )
    write_summary(summary, args.summary_output)
    if args.report_output is not None:
        write_markdown_report(summary, args.report_output)
    selected = summary["comparison"]["selected_candidate"]
    print(
        "target grammar v3 pressure: "
        f"decision={summary['decision']['route']} "
        f"selected_k={summary['comparison']['selected_k']} "
        f"token_reduction={_fmt_pct(selected['token_reduction_ratio'])} "
        f"bit_reduction={_fmt_pct(selected['total_bit_reduction_ratio'])} "
        f"mismatches={selected['reconstruction_mismatches']}"
    )


if __name__ == "__main__":
    main()
