from __future__ import annotations

import argparse
import json
import math
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
    unigram_nll_bits_from_counts,
)
from pulsefield_model.models.mapper.shared.vocab import KEY_COUNT
from pulsefield_model.models.mapper.v2_1.vocab import MapperV21Vocab
from pulsefield_model.models.mapper.v3.conversion import v2_1_tokens_to_v3_event_tokens
from pulsefield_model.models.mapper.v3.vocab import MapperV3Vocab, actions_from_event_signature


SUMMARY_SCHEMA_VERSION = 1
DEFAULT_CONFIG_PATH = Path("artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_enabled.yaml")
DEFAULT_FULL_DATASET_SUMMARY_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_full_dataset_audit_summary.json"
)
DEFAULT_CONDITIONED_SHORT_SUMMARY_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/"
    "target_grammar_v3_conditioned_event_distribution_short_rollout_summary.json"
)
DEFAULT_CE_WEIGHT_SUMMARY_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_token_ce_weight_training_gate_summary.json"
)
DEFAULT_SUMMARY_OUTPUT_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/"
    "target_grammar_v3_factorized_event_signature_proxy_summary.json"
)
DEFAULT_REPORT_OUTPUT_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/"
    "target_grammar_v3_factorized_event_signature_proxy_result_report.md"
)
EVENT_BITS_PER_EVENT_REDUCTION_THRESHOLD = 0.10
CURRENT_V3_TOTAL_BITS_MARGIN = 0.05


@dataclass
class SplitFactorizedProxyAccumulator:
    split_name: str
    source_vocab: MapperV21Vocab
    target_vocab: MapperV3Vocab
    window_count: int = 0
    baseline_token_count: int = 0
    current_v3_token_count: int = 0
    factorized_token_count: int = 0
    event_token_count: int = 0
    event_factor_token_count: int = 0
    multi_lane_event_count: int = 0
    lane_action_token_count: int = 0
    current_v3_reconstruction_mismatches: int = 0
    factor_reconstruction_mismatches: int = 0
    baseline_token_counts: Counter[str] = field(default_factory=Counter)
    current_v3_token_counts: Counter[str] = field(default_factory=Counter)
    factorized_token_counts: Counter[str] = field(default_factory=Counter)
    current_event_signature_counts: Counter[str] = field(default_factory=Counter)
    factor_event_token_counts: Counter[str] = field(default_factory=Counter)
    lane_count_factor_counts: Counter[str] = field(default_factory=Counter)
    lane_mask_factor_counts: Counter[str] = field(default_factory=Counter)
    action_factor_counts: Counter[str] = field(default_factory=Counter)
    mismatch_examples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def reconstruction_mismatches(self) -> int:
        return self.current_v3_reconstruction_mismatches + self.factor_reconstruction_mismatches

    def update(self, *, token_ids: Sequence[int], dataset_index: int) -> None:
        conversion = v2_1_tokens_to_v3_event_tokens(
            token_ids,
            source_vocab=self.source_vocab,
            target_vocab=self.target_vocab,
        )
        self.window_count += 1
        self.baseline_token_count += len(token_ids)
        self.current_v3_token_count += len(conversion.token_ids)
        self.event_token_count += conversion.event_token_count
        self.multi_lane_event_count += conversion.multi_lane_event_count
        self.lane_action_token_count += conversion.lane_action_token_count
        self.baseline_token_counts.update(self.source_vocab.token_name(int(token_id)) for token_id in token_ids)
        if conversion.reconstructed_v2_1_token_ids != tuple(int(token_id) for token_id in token_ids):
            self.current_v3_reconstruction_mismatches += 1
            if len(self.mismatch_examples) < 5:
                self.mismatch_examples.append(
                    {
                        "split": self.split_name,
                        "dataset_index": int(dataset_index),
                        "kind": "current_v3_reconstruction",
                        "baseline_prefix": [self.source_vocab.token_name(int(token_id)) for token_id in token_ids[:20]],
                        "reconstructed_prefix": [
                            self.source_vocab.token_name(int(token_id))
                            for token_id in conversion.reconstructed_v2_1_token_ids[:20]
                        ],
                    }
                )

        factorized_tokens: list[str] = []
        current_tokens: list[str] = []
        for token_id in conversion.token_ids:
            token_id = int(token_id)
            if self.target_vocab.is_event_token(token_id):
                signature = self.target_vocab.event_signature(token_id)
                current_tokens.append(f"EV_SIG_{signature}")
                self.current_event_signature_counts[signature] += 1
                factors = factorize_event_signature(signature)
                reconstructed = reconstruct_event_signature(factors)
                if reconstructed != signature:
                    self.factor_reconstruction_mismatches += 1
                    if len(self.mismatch_examples) < 5:
                        self.mismatch_examples.append(
                            {
                                "split": self.split_name,
                                "dataset_index": int(dataset_index),
                                "kind": "factor_reconstruction",
                                "signature": signature,
                                "factors": list(factors),
                                "reconstructed": reconstructed,
                            }
                        )
                factorized_tokens.extend(factors)
                self.factor_event_token_counts.update(factors)
                self.lane_count_factor_counts.update((factors[0],))
                self.lane_mask_factor_counts.update((factors[1],))
                self.action_factor_counts.update(factors[2:])
                self.event_factor_token_count += len(factors)
            else:
                token_name = self.target_vocab.token_name(token_id)
                current_tokens.append(token_name)
                factorized_tokens.append(token_name)
        self.current_v3_token_counts.update(current_tokens)
        self.factorized_token_counts.update(factorized_tokens)
        self.factorized_token_count += len(factorized_tokens)

    def to_dict(self) -> dict[str, Any]:
        event_factor_expansion_ratio = _safe_ratio(self.event_factor_token_count, self.event_token_count)
        total_token_expansion_ratio = _safe_ratio(self.factorized_token_count, self.current_v3_token_count)
        return {
            "split": self.split_name,
            "window_count": self.window_count,
            "baseline_token_count": self.baseline_token_count,
            "current_v3_token_count": self.current_v3_token_count,
            "factorized_token_count": self.factorized_token_count,
            "event_token_count": self.event_token_count,
            "event_factor_token_count": self.event_factor_token_count,
            "event_factor_expansion_ratio": event_factor_expansion_ratio,
            "total_token_expansion_ratio": total_token_expansion_ratio,
            "multi_lane_event_count": self.multi_lane_event_count,
            "lane_action_token_count": self.lane_action_token_count,
            "current_v3_reconstruction_mismatches": self.current_v3_reconstruction_mismatches,
            "factor_reconstruction_mismatches": self.factor_reconstruction_mismatches,
            "reconstruction_mismatches": self.reconstruction_mismatches,
            "observed_current_event_vocab_size": len(self.current_event_signature_counts),
            "observed_factor_event_vocab_size": len(self.factor_event_token_counts),
            "observed_lane_count_vocab_size": len(self.lane_count_factor_counts),
            "observed_lane_mask_vocab_size": len(self.lane_mask_factor_counts),
            "observed_action_vocab_size": len(self.action_factor_counts),
            "top_event_signatures": _top_event_rows(self.current_event_signature_counts, limit=20),
            "top_lane_count_factors": _top_count_rows(self.lane_count_factor_counts, limit=12),
            "top_lane_mask_factors": _top_count_rows(self.lane_mask_factor_counts, limit=20),
            "top_action_factors": _top_count_rows(self.action_factor_counts, limit=12),
            "mismatch_examples": list(self.mismatch_examples),
        }


def factorize_event_signature(signature: str) -> tuple[str, ...]:
    actions_from_event_signature(signature)
    mask = "".join("0" if symbol == "." else "1" for symbol in signature)
    active_actions = tuple(symbol for symbol in signature if symbol != ".")
    return (
        f"EV_COUNT_{len(active_actions)}",
        f"EV_MASK_{mask}",
        *(f"EV_ACTION_{symbol}" for symbol in active_actions),
    )


def reconstruct_event_signature(factors: Sequence[str]) -> str:
    if len(factors) < 3:
        raise ValueError(f"event factor sequence is too short: {factors!r}")
    count_token = str(factors[0])
    mask_token = str(factors[1])
    if not count_token.startswith("EV_COUNT_"):
        raise ValueError(f"missing EV_COUNT factor: {count_token!r}")
    if not mask_token.startswith("EV_MASK_"):
        raise ValueError(f"missing EV_MASK factor: {mask_token!r}")
    try:
        active_count = int(count_token.removeprefix("EV_COUNT_"))
    except ValueError as exc:
        raise ValueError(f"invalid EV_COUNT factor: {count_token!r}") from exc
    mask = mask_token.removeprefix("EV_MASK_")
    if len(mask) != KEY_COUNT or any(symbol not in {"0", "1"} for symbol in mask):
        raise ValueError(f"invalid EV_MASK factor: {mask_token!r}")
    action_symbols: list[str] = []
    for factor in factors[2:]:
        factor = str(factor)
        if not factor.startswith("EV_ACTION_"):
            raise ValueError(f"missing EV_ACTION factor: {factor!r}")
        symbol = factor.removeprefix("EV_ACTION_")
        if symbol not in {"T", "S", "E"}:
            raise ValueError(f"invalid EV_ACTION factor: {factor!r}")
        action_symbols.append(symbol)
    if active_count != mask.count("1") or active_count != len(action_symbols):
        raise ValueError(
            "event factor sequence has inconsistent active lane count: "
            f"count={active_count} mask={mask!r} actions={action_symbols!r}"
        )
    signature = ["."] * KEY_COUNT
    action_index = 0
    for lane_index, mask_symbol in enumerate(mask):
        if mask_symbol == "1":
            signature[lane_index] = action_symbols[action_index]
            action_index += 1
    reconstructed = "".join(signature)
    actions_from_event_signature(reconstructed)
    return reconstructed


def audit_mapper_v3_factorized_event_signature_proxy(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    full_dataset_summary_path: Path = DEFAULT_FULL_DATASET_SUMMARY_PATH,
    conditioned_short_summary_path: Path = DEFAULT_CONDITIONED_SHORT_SUMMARY_PATH,
    ce_weight_summary_path: Path = DEFAULT_CE_WEIGHT_SUMMARY_PATH,
    train_limit: int | None = 4096,
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
    total_audit = _merge_split_audits((train_audit, eval_audit), source_vocab=source_vocab, target_vocab=target_vocab)

    scores = _score_proxy(train_audit=train_audit, eval_audit=eval_audit, alpha=alpha)
    comparison = _comparison_from_scores(eval_audit=eval_audit, total_audit=total_audit, scores=scores)
    comparators = _load_comparators(
        full_dataset_summary_path=full_dataset_summary_path,
        conditioned_short_summary_path=conditioned_short_summary_path,
        ce_weight_summary_path=ce_weight_summary_path,
    )
    dataset = {
        "train_window_count": len(datasets.train_dataset),
        "train_scored_window_count": train_window_count,
        "eval_window_count": len(datasets.eval_dataset),
        "eval_scored_window_count": eval_window_count,
        "audit_window_count": total_audit.window_count,
        "full_dataset_proxy": train_limit is None and eval_limit is None,
        "bounded_runtime_fallback": not (train_limit is None and eval_limit is None),
        "bounded_runtime_reason": (
            "The existing full v3 audit artifact does not expose full event-signature counts; "
            "this run regenerates a bounded target stream to stay within the card runtime budget."
            if not (train_limit is None and eval_limit is None)
            else None
        ),
    }
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar v3 factorized event-signature proxy",
        "config_path": config_path.as_posix(),
        "dataset": dataset,
        "vocab": {
            "v2_1_vocab_size": source_vocab.size,
            "v3_vocab_size": target_vocab.size,
            "v3_event_token_count": len(target_vocab.event_token_ids),
            "factor_contract": "event_signature_to_lane_count_lane_mask_active_actions",
            "factor_tokens_are_runtime_vocab": False,
        },
        "baseline": {
            "name": "v2.1 sparse lane-action target stream",
            "unigram_nll": scores["baseline"],
        },
        "current_v3": {
            "name": "v3 single 4-lane event token",
            "unigram_nll": scores["current_v3"],
            "event_unigram_nll": scores["current_event"],
            "top_event_signatures": _top_event_rows(eval_audit.current_event_signature_counts, limit=20),
        },
        "factorized_proxy": {
            "name": "lane-count plus lane-mask plus active-lane action factors",
            "unigram_nll": scores["factorized"],
            "event_unigram_nll": scores["factor_event"],
            "top_lane_count_factors": _top_count_rows(eval_audit.lane_count_factor_counts, limit=12),
            "top_lane_mask_factors": _top_count_rows(eval_audit.lane_mask_factor_counts, limit=20),
            "top_action_factors": _top_count_rows(eval_audit.action_factor_counts, limit=12),
        },
        "comparison": comparison,
        "checks": _checks_from_comparison(comparison),
        "comparators": comparators,
        "audit": {
            "train": train_audit.to_dict(),
            "eval": eval_audit.to_dict(),
            "total": total_audit.to_dict(),
        },
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
    dataset = summary["dataset"]
    current_v3 = summary["current_v3"]
    factorized = summary["factorized_proxy"]
    baseline = summary["baseline"]
    checks = summary["checks"]
    comparators = summary["comparators"]
    factor_vocab = comparison["factor_vocab"]
    event_reduction_ratio = comparison["event_bits_per_original_event_reduction_ratio"]
    if isinstance(event_reduction_ratio, (int, float)) and float(event_reduction_ratio) < 0.0:
        tradeoff_note = (
            "In this run the factorized event stream is worse even after normalizing back to the original "
            "event count: factor tokens have a smaller vocabulary, but the count/mask/action sequence is long "
            "enough that event bits per original event increase."
        )
    else:
        tradeoff_note = (
            "The key issue is tradeoff shape: lane masks and action factors may concentrate event classes, "
            "but they also expand the target stream. A positive representation proxy would still not prove "
            "trained rollout calibration; it would only justify a bounded grammar implementation card."
        )
    lines = [
        "# Target Grammar v3 Factorized Event-Signature Proxy Result Report",
        "",
        "## Scope",
        "",
        "This artifact-only audit tests whether the current v3 4-lane event token has enough internal structure to justify a future factorized target grammar. It does not change the mapper vocab, tokenizer, model heads, training loop, inference, or replay.",
        "",
        "The proxy replaces each current v3 event signature with local factors: lane-count bucket, lane-mask bucket, and active-lane action symbols. Time-shift and special tokens stay unchanged. Each factor sequence is checked for exact reconstruction of the original v3 event signature.",
        "",
        "## Result",
        "",
        f"Decision: `{decision['route']}`.",
        "",
        f"- Reason: {decision['reason']}",
        f"- Train windows scored: `{dataset['train_scored_window_count']}` / `{dataset['train_window_count']}`",
        f"- Eval windows scored: `{dataset['eval_scored_window_count']}` / `{dataset['eval_window_count']}`",
        f"- Full-dataset proxy: `{dataset['full_dataset_proxy']}`",
        f"- Runtime fallback: `{dataset['bounded_runtime_fallback']}`",
        f"- Reconstruction mismatches: `{comparison['reconstruction_mismatches']}`",
        f"- v2.1 total bits: `{_fmt_metric(baseline['unigram_nll']['total_bits'])}`",
        f"- current v3 total bits: `{_fmt_metric(current_v3['unigram_nll']['total_bits'])}`",
        f"- factorized total bits: `{_fmt_metric(factorized['unigram_nll']['total_bits'])}`",
        f"- factorized vs v2.1 total-bit ratio: `{_fmt_metric(comparison['factorized_total_bits_vs_v2_1_ratio'])}`",
        f"- factorized vs current v3 total-bit ratio: `{_fmt_metric(comparison['factorized_total_bits_vs_current_v3_ratio'])}`",
        f"- event bits/original-event reduction: `{_fmt_pct(comparison['event_bits_per_original_event_reduction_ratio'])}`",
        f"- total token expansion ratio: `{_fmt_metric(comparison['token_expansion']['total_token_expansion_ratio'])}`",
        f"- event factor expansion ratio: `{_fmt_metric(comparison['token_expansion']['event_factor_expansion_ratio'])}`",
        f"- elapsed: `{float(summary['elapsed_s']):.2f}s`",
        "",
        "## Metric Table",
        "",
        "| Stream | Eval tokens | Vocab | Bits/token | Total bits |",
        "| --- | ---: | ---: | ---: | ---: |",
        _score_row("v2.1", baseline["unigram_nll"]),
        _score_row("current v3", current_v3["unigram_nll"]),
        _score_row("factorized proxy", factorized["unigram_nll"]),
        _score_row("current v3 events only", current_v3["event_unigram_nll"]),
        _score_row("factorized event factors only", factorized["event_unigram_nll"]),
        "",
        "## Factor Vocabulary",
        "",
        f"- lane-count factors: `{factor_vocab['lane_count_vocab_size']}`",
        f"- lane-mask factors: `{factor_vocab['lane_mask_vocab_size']}`",
        f"- action factors: `{factor_vocab['action_vocab_size']}`",
        f"- total event-factor vocab: `{factor_vocab['event_factor_vocab_size']}`",
        f"- full factorized stream vocab: `{factor_vocab['full_factorized_vocab_size']}`",
        "",
        "## Top Lane Masks",
        "",
        "| Mask factor | Count |",
        "| --- | ---: |",
        *[
            f"| `{row['token']}` | {row['count']} |"
            for row in factorized["top_lane_mask_factors"][:12]
        ],
        "",
        "## Top Event Signatures",
        "",
        "| Signature | Count | Non-empty lanes |",
        "| --- | ---: | ---: |",
        *[
            f"| `{row['signature']}` | {row['count']} | {row['lane_count']} |"
            for row in current_v3["top_event_signatures"][:12]
        ],
        "",
        "## What Passed",
        "",
        f"- Exact factor reconstruction: `{checks['exact_factor_reconstruction']}`.",
        f"- Exact current v3 reconstruction back to v2.1: `{checks['exact_current_v3_reconstruction']}`.",
        f"- Online-local factors only: `{checks['online_local_factors']}`.",
        f"- No cross-window side stream: `{checks['no_cross_window_side_stream']}`.",
        f"- Factorized total bits below v2.1: `{checks['factorized_below_v2_1']}`.",
        f"- Event bits/original-event reduction is meaningful: `{checks['event_bits_reduction_meaningful']}`.",
        "",
        "## What Surfaced",
        "",
        decision["interpretation"],
        "",
        tradeoff_note,
        "",
        "## Comparator Snapshot",
        "",
        f"- Full v3 audit route: `{comparators['full_dataset_audit']['decision_route']}`; full summary available: `{comparators['full_dataset_audit']['available']}`.",
        f"- Conditioned short rollout route: `{comparators['conditioned_short_rollout']['decision_route']}`; reason: `{comparators['conditioned_short_rollout']['reason']}`.",
        f"- CE-weight training gate route: `{comparators['ce_weight_training_gate']['decision_route']}`; reason: `{comparators['ce_weight_training_gate']['reason']}`.",
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
) -> SplitFactorizedProxyAccumulator:
    accumulator = SplitFactorizedProxyAccumulator(
        split_name=split_name,
        source_vocab=source_vocab,
        target_vocab=target_vocab,
    )
    started_at = time.monotonic()
    for index in range(int(limit)):
        tokenized = _tokenized_window_for_dataset_index(dataset, index)
        token_ids = tuple(int(token_id) for token_id in tokenized.target_fragment_ids)
        accumulator.update(token_ids=token_ids, dataset_index=index)
        if progress_interval > 0 and accumulator.window_count % int(progress_interval) == 0:
            elapsed_s = time.monotonic() - started_at
            print(
                "mapper_v3_factorized_event_signature_proxy_progress "
                f"split={split_name} windows={accumulator.window_count}/{limit} "
                f"mismatches={accumulator.reconstruction_mismatches} elapsed_s={elapsed_s:.1f}",
                flush=True,
            )
    return accumulator


def _merge_split_audits(
    audits: Sequence[SplitFactorizedProxyAccumulator],
    *,
    source_vocab: MapperV21Vocab,
    target_vocab: MapperV3Vocab,
) -> SplitFactorizedProxyAccumulator:
    merged = SplitFactorizedProxyAccumulator(split_name="total", source_vocab=source_vocab, target_vocab=target_vocab)
    for audit in audits:
        merged.window_count += audit.window_count
        merged.baseline_token_count += audit.baseline_token_count
        merged.current_v3_token_count += audit.current_v3_token_count
        merged.factorized_token_count += audit.factorized_token_count
        merged.event_token_count += audit.event_token_count
        merged.event_factor_token_count += audit.event_factor_token_count
        merged.multi_lane_event_count += audit.multi_lane_event_count
        merged.lane_action_token_count += audit.lane_action_token_count
        merged.current_v3_reconstruction_mismatches += audit.current_v3_reconstruction_mismatches
        merged.factor_reconstruction_mismatches += audit.factor_reconstruction_mismatches
        merged.baseline_token_counts.update(audit.baseline_token_counts)
        merged.current_v3_token_counts.update(audit.current_v3_token_counts)
        merged.factorized_token_counts.update(audit.factorized_token_counts)
        merged.current_event_signature_counts.update(audit.current_event_signature_counts)
        merged.factor_event_token_counts.update(audit.factor_event_token_counts)
        merged.lane_count_factor_counts.update(audit.lane_count_factor_counts)
        merged.lane_mask_factor_counts.update(audit.lane_mask_factor_counts)
        merged.action_factor_counts.update(audit.action_factor_counts)
        merged.mismatch_examples.extend(audit.mismatch_examples[: max(0, 5 - len(merged.mismatch_examples))])
    return merged


def _score_proxy(
    *,
    train_audit: SplitFactorizedProxyAccumulator,
    eval_audit: SplitFactorizedProxyAccumulator,
    alpha: float,
) -> dict[str, dict[str, Any]]:
    return {
        "baseline": unigram_nll_bits_from_counts(
            train_audit.baseline_token_counts,
            eval_audit.baseline_token_counts,
            alpha=alpha,
        ),
        "current_v3": unigram_nll_bits_from_counts(
            train_audit.current_v3_token_counts,
            eval_audit.current_v3_token_counts,
            alpha=alpha,
        ),
        "factorized": unigram_nll_bits_from_counts(
            train_audit.factorized_token_counts,
            eval_audit.factorized_token_counts,
            alpha=alpha,
        ),
        "current_event": unigram_nll_bits_from_counts(
            train_audit.current_event_signature_counts,
            eval_audit.current_event_signature_counts,
            alpha=alpha,
        ),
        "factor_event": unigram_nll_bits_from_counts(
            train_audit.factor_event_token_counts,
            eval_audit.factor_event_token_counts,
            alpha=alpha,
        ),
    }


def _comparison_from_scores(
    *,
    eval_audit: SplitFactorizedProxyAccumulator,
    total_audit: SplitFactorizedProxyAccumulator,
    scores: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    baseline_bits = float(scores["baseline"]["total_bits"])
    current_v3_bits = float(scores["current_v3"]["total_bits"])
    factorized_bits = float(scores["factorized"]["total_bits"])
    current_event_bits = float(scores["current_event"]["total_bits"])
    factor_event_bits = float(scores["factor_event"]["total_bits"])
    event_count = int(eval_audit.event_token_count)
    current_event_bits_per_original_event = _safe_ratio(current_event_bits, event_count)
    factor_event_bits_per_original_event = _safe_ratio(factor_event_bits, event_count)
    event_bits_reduction = None
    event_bits_reduction_ratio = None
    if current_event_bits_per_original_event is not None and factor_event_bits_per_original_event is not None:
        event_bits_reduction = current_event_bits_per_original_event - factor_event_bits_per_original_event
        event_bits_reduction_ratio = _safe_ratio(event_bits_reduction, current_event_bits_per_original_event)
    return {
        "reconstruction_mismatches": total_audit.reconstruction_mismatches,
        "current_v3_reconstruction_mismatches": total_audit.current_v3_reconstruction_mismatches,
        "factor_reconstruction_mismatches": total_audit.factor_reconstruction_mismatches,
        "v2_1_total_bits": baseline_bits,
        "current_v3_total_bits": current_v3_bits,
        "factorized_total_bits": factorized_bits,
        "current_v3_total_bit_reduction_vs_v2_1": baseline_bits - current_v3_bits,
        "current_v3_total_bit_reduction_ratio_vs_v2_1": _safe_ratio(baseline_bits - current_v3_bits, baseline_bits),
        "factorized_total_bit_reduction_vs_v2_1": baseline_bits - factorized_bits,
        "factorized_total_bit_reduction_ratio_vs_v2_1": _safe_ratio(baseline_bits - factorized_bits, baseline_bits),
        "factorized_total_bit_delta_vs_current_v3": factorized_bits - current_v3_bits,
        "factorized_total_bit_delta_ratio_vs_current_v3": _safe_ratio(factorized_bits - current_v3_bits, current_v3_bits),
        "factorized_total_bits_vs_v2_1_ratio": _safe_ratio(factorized_bits, baseline_bits),
        "factorized_total_bits_vs_current_v3_ratio": _safe_ratio(factorized_bits, current_v3_bits),
        "current_event_total_bits": current_event_bits,
        "factor_event_total_bits": factor_event_bits,
        "current_event_bits_per_original_event": current_event_bits_per_original_event,
        "factor_event_bits_per_original_event": factor_event_bits_per_original_event,
        "event_bits_per_original_event_reduction": event_bits_reduction,
        "event_bits_per_original_event_reduction_ratio": event_bits_reduction_ratio,
        "current_event_bits_per_event_token": scores["current_event"]["bits_per_token"],
        "factor_event_bits_per_factor_token": scores["factor_event"]["bits_per_token"],
        "token_expansion": {
            "current_v3_eval_token_count": int(scores["current_v3"]["eval_token_count"]),
            "factorized_eval_token_count": int(scores["factorized"]["eval_token_count"]),
            "current_v3_eval_event_token_count": event_count,
            "factorized_eval_event_factor_token_count": int(eval_audit.event_factor_token_count),
            "total_token_expansion_ratio": _safe_ratio(
                int(scores["factorized"]["eval_token_count"]),
                int(scores["current_v3"]["eval_token_count"]),
            ),
            "event_factor_expansion_ratio": _safe_ratio(eval_audit.event_factor_token_count, event_count),
        },
        "factor_vocab": {
            "lane_count_vocab_size": len(eval_audit.lane_count_factor_counts),
            "lane_mask_vocab_size": len(eval_audit.lane_mask_factor_counts),
            "action_vocab_size": len(eval_audit.action_factor_counts),
            "event_factor_vocab_size": len(eval_audit.factor_event_token_counts),
            "full_factorized_vocab_size": int(scores["factorized"]["vocab_size"]),
        },
        "event_counts": {
            "eval_event_token_count": event_count,
            "eval_multi_lane_event_count": eval_audit.multi_lane_event_count,
            "eval_lane_action_token_count": eval_audit.lane_action_token_count,
        },
    }


def _checks_from_comparison(comparison: Mapping[str, Any]) -> dict[str, bool]:
    factorized_below_v2_1 = float(comparison["factorized_total_bits"]) < float(comparison["v2_1_total_bits"])
    factorized_close_to_current_v3 = (
        float(comparison["factorized_total_bit_delta_ratio_vs_current_v3"] or 0.0)
        <= CURRENT_V3_TOTAL_BITS_MARGIN
    )
    event_reduction_ratio = float(comparison["event_bits_per_original_event_reduction_ratio"] or 0.0)
    return {
        "exact_current_v3_reconstruction": int(comparison["current_v3_reconstruction_mismatches"]) == 0,
        "exact_factor_reconstruction": int(comparison["factor_reconstruction_mismatches"]) == 0,
        "factorized_below_v2_1": factorized_below_v2_1,
        "factorized_close_to_current_v3": factorized_close_to_current_v3,
        "event_bits_reduction_meaningful": event_reduction_ratio >= EVENT_BITS_PER_EVENT_REDUCTION_THRESHOLD,
        "online_local_factors": True,
        "no_cross_window_side_stream": True,
        "no_future_target_dependency": True,
    }


def _decision_from_comparison(comparison: Mapping[str, Any]) -> dict[str, Any]:
    checks = _checks_from_comparison(comparison)
    failed_checks = [name for name, passed in checks.items() if not passed]
    kill_criteria: list[str] = []
    if int(comparison["reconstruction_mismatches"]) > 0:
        kill_criteria.append("reconstruction_mismatch")
    if not checks["factorized_below_v2_1"]:
        kill_criteria.append("factorized_total_bits_not_below_v2_1")
    v3_delta_ratio = float(comparison["factorized_total_bit_delta_ratio_vs_current_v3"] or 0.0)
    event_reduction_ratio = float(comparison["event_bits_per_original_event_reduction_ratio"] or 0.0)
    if (
        v3_delta_ratio > CURRENT_V3_TOTAL_BITS_MARGIN
        and event_reduction_ratio < EVENT_BITS_PER_EVENT_REDUCTION_THRESHOLD
    ):
        kill_criteria.append("factorized_bits_exceed_current_v3_margin_without_event_reduction")

    if "reconstruction_mismatch" in kill_criteria:
        route = "KILL_FACTORIZE_EVENT_SIGNATURE"
        reason = "factorized event-signature proxy tripped the hard reconstruction guard"
        interpretation = (
            "The factorized representation should not be implemented as a target grammar under this card. "
            "It failed exact reconstruction of the current v3 event signatures."
        )
        next_step = "Kill factorized event-signature repair and test another target-side or v2.1 grammar repair candidate."
        positive = False
    elif "factorized_total_bits_not_below_v2_1" in kill_criteria:
        route = "KILL_FACTORIZE_EVENT_SIGNATURE"
        reason = "factorized event-signature proxy failed the v2.1 total-bit guard"
        interpretation = (
            "Exact reconstruction passed, but the factorized stream gave back the current v3 representation "
            "advantage and scored worse than v2.1 under the unigram bit proxy. This is not a good target "
            "grammar replacement candidate from the current evidence."
        )
        next_step = "Kill factorized event-signature repair and test another target-side or v2.1 grammar repair candidate."
        positive = False
    elif checks["event_bits_reduction_meaningful"] and checks["factorized_close_to_current_v3"]:
        route = "TEST_FACTORIZE_V3_EVENT_SIGNATURE_GRAMMAR"
        reason = "event-factor proxy is reversible, below v2.1 total bits, and close to current v3 while reducing event bits"
        interpretation = (
            "The proxy gives enough representation-side evidence to justify a bounded v3.1 grammar card. "
            "This still does not prove trained rollout quality; it only says the factorization is not obviously "
            "worse by the local compression and reversibility gates."
        )
        next_step = "Create a v3.1 factorized-event grammar Experiment Card with a tiny training and rollout gate."
        positive = True
    elif checks["event_bits_reduction_meaningful"]:
        route = "MUTATE_TARGET_REPAIR"
        reason = "event factors reduce event bits, but the full-stream proxy is not close enough to current v3"
        interpretation = (
            "The decomposition is learning-relevant but not yet target-competitive as a direct replacement. "
            "The likely issue is target-length overhead: factorization concentrates event classes while adding "
            "extra supervised steps."
        )
        next_step = "Run one bucket-level diagnostic or mutate to a lighter mask/action target before grammar implementation."
        positive = False
    else:
        route = "KILL_FACTORIZE_EVENT_SIGNATURE"
        reason = "event-factor proxy did not produce a meaningful event-bit reduction"
        interpretation = (
            "The current single-token v3 event signature should remain the target-side comparator for now. "
            "The factorized form adds complexity without enough local evidence that it improves the event target."
        )
        next_step = "Move to another target repair or v2.1 grammar route instead of implementing event-signature factorization."
        positive = False
        if not kill_criteria:
            kill_criteria.append("event_bits_reduction_not_meaningful")

    return {
        "route": route,
        "reason": reason,
        "positive_gate_passed": positive,
        "kill_criteria_triggered": bool(kill_criteria),
        "kill_criteria": kill_criteria,
        "failed_checks": failed_checks,
        "interpretation": interpretation,
        "next_step": next_step,
    }


def _load_comparators(
    *,
    full_dataset_summary_path: Path,
    conditioned_short_summary_path: Path,
    ce_weight_summary_path: Path,
) -> dict[str, Any]:
    full_dataset = _load_optional_json(full_dataset_summary_path)
    conditioned_short = _load_optional_json(conditioned_short_summary_path)
    ce_weight = _load_optional_json(ce_weight_summary_path)
    return {
        "full_dataset_audit": {
            "path": full_dataset_summary_path.as_posix(),
            "available": full_dataset is not None,
            "decision_route": _decision_route(full_dataset),
            "comparison": (full_dataset or {}).get("comparison", {}),
            "dataset": (full_dataset or {}).get("dataset", {}),
        },
        "conditioned_short_rollout": {
            "path": conditioned_short_summary_path.as_posix(),
            "available": conditioned_short is not None,
            "decision_route": _decision_route(conditioned_short),
            "reason": ((conditioned_short or {}).get("decision") or {}).get("reason"),
        },
        "ce_weight_training_gate": {
            "path": ce_weight_summary_path.as_posix(),
            "available": ce_weight is not None,
            "decision_route": _decision_route(ce_weight),
            "reason": ((ce_weight or {}).get("decision") or {}).get("reason"),
        },
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


def _top_event_rows(counts: Mapping[str, int], *, limit: int) -> list[dict[str, Any]]:
    return [
        {
            "signature": signature,
            "count": int(count),
            "lane_count": sum(1 for symbol in signature if symbol != "."),
        }
        for signature, count in sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[: int(limit)]
    ]


def _top_count_rows(counts: Mapping[str, int], *, limit: int) -> list[dict[str, Any]]:
    return [
        {"token": token, "count": int(count)}
        for token, count in sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[: int(limit)]
    ]


def _score_row(name: str, score: Mapping[str, Any]) -> str:
    return "| {name} | {tokens} | {vocab} | {bits_per_token} | {total_bits} |".format(
        name=name,
        tokens=score["eval_token_count"],
        vocab=score["vocab_size"],
        bits_per_token=_fmt_metric(score["bits_per_token"]),
        total_bits=_fmt_metric(score["total_bits"]),
    )


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
    parser = argparse.ArgumentParser(description="Audit a factorized proxy for v3 event signatures.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--full-dataset-summary", type=Path, default=DEFAULT_FULL_DATASET_SUMMARY_PATH)
    parser.add_argument("--conditioned-short-summary", type=Path, default=DEFAULT_CONDITIONED_SHORT_SUMMARY_PATH)
    parser.add_argument("--ce-weight-summary", type=Path, default=DEFAULT_CE_WEIGHT_SUMMARY_PATH)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT_PATH)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT_PATH)
    parser.add_argument("--train-limit", type=parse_limit, default=4096)
    parser.add_argument("--eval-limit", type=parse_limit, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--progress-interval", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = audit_mapper_v3_factorized_event_signature_proxy(
        config_path=args.config,
        full_dataset_summary_path=args.full_dataset_summary,
        conditioned_short_summary_path=args.conditioned_short_summary,
        ce_weight_summary_path=args.ce_weight_summary,
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
        "mapper v3 factorized event-signature proxy: "
        f"decision={summary['decision']['route']} "
        f"factorized_vs_v2_1={_fmt_metric(comparison['factorized_total_bits_vs_v2_1_ratio'])} "
        f"factorized_vs_v3={_fmt_metric(comparison['factorized_total_bits_vs_current_v3_ratio'])} "
        f"event_reduction={_fmt_pct(comparison['event_bits_per_original_event_reduction_ratio'])} "
        f"mismatches={comparison['reconstruction_mismatches']}"
    )


if __name__ == "__main__":
    main()
