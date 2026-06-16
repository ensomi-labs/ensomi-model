from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from pulsefield_model.evals.mapper_v3_trained_runtime_rollout import run_trained_v3_runtime_rollout_smoke
from pulsefield_model.inference.mapper_v3_rollout import MapperV3GenerationStep
from pulsefield_model.models.mapper.shared.tokenizer import hitobjects_to_mapper_timepoints
from pulsefield_model.models.mapper.v3 import MapperV3Vocab
from pulsefield_model.osu_core.hitobjects import parse_mania_hit_objects


SUMMARY_SCHEMA_VERSION = 1
GATE_START_MS = 8000
OPPORTUNITY_MAX_RANK = 5
OPPORTUNITY_MIN_MARGIN = -2.0
MAX_FORCED_EVENTS_PER_WINDOW = 32
MIN_FORCED_EVENT_GAP_MS = 80
SECOND_SHARE_DELTA_SIGNAL = 0.15
MAX_EVENT_COUNT_RATIO = 1.25
MAX_F1_REGRESSION = 0.05
MAX_PASSLIKE_EVENT_RATIO_DELTA = 0.25
TIMING_F1_TOLERANCE_MS = 100


class SelectiveEventGate:
    def __init__(
        self,
        *,
        vocab: MapperV3Vocab,
        gate_start_ms: int = GATE_START_MS,
        max_event_rank: int = OPPORTUNITY_MAX_RANK,
        min_event_margin: float = OPPORTUNITY_MIN_MARGIN,
        max_forced_events_per_window: int = MAX_FORCED_EVENTS_PER_WINDOW,
        min_forced_event_gap_ms: int = MIN_FORCED_EVENT_GAP_MS,
        force_margin: float = 1e-3,
        max_examples: int = 24,
    ) -> None:
        self.vocab = vocab
        self.gate_start_ms = int(gate_start_ms)
        self.max_event_rank = int(max_event_rank)
        self.min_event_margin = float(min_event_margin)
        self.max_forced_events_per_window = int(max_forced_events_per_window)
        self.min_forced_event_gap_ms = int(min_forced_event_gap_ms)
        self.force_margin = float(force_margin)
        self.max_examples = int(max_examples)
        self.step_count = 0
        self.candidate_count = 0
        self.forced_count = 0
        self.forced_by_window: Counter[str] = Counter()
        self.skipped_counts: Counter[str] = Counter()
        self.last_forced_ms_by_window: dict[int, int] = {}
        self.forced_count_by_window_start: Counter[int] = Counter()
        self.examples: list[dict[str, Any]] = []

    def __call__(self, step: MapperV3GenerationStep, logits: torch.Tensor) -> torch.Tensor:
        self.step_count += 1
        flat_logits = torch.as_tensor(logits, dtype=torch.float32).reshape(-1)
        valid_mask = step.valid_token_mask.to(device=flat_logits.device, dtype=torch.bool).reshape(-1)
        if int(valid_mask.numel()) != int(flat_logits.numel()):
            raise ValueError(f"valid mask has {valid_mask.numel()} tokens but logits have {flat_logits.numel()}")
        masked = flat_logits.masked_fill(~valid_mask, -torch.inf)
        if not bool(torch.isfinite(masked).any().item()):
            self.skipped_counts["no_valid_token"] += 1
            return flat_logits

        argmax_id = int(torch.argmax(masked).item())
        current_ms = int(step.state.current_ms)
        window_label = "second" if current_ms >= self.gate_start_ms else "first"
        if current_ms < self.gate_start_ms:
            self.skipped_counts["before_gate_start"] += 1
            return flat_logits
        if current_ms >= int(step.chart_end_ms):
            self.skipped_counts["at_or_after_chart_end"] += 1
            return flat_logits
        if self.vocab.is_event_token(argmax_id):
            self.skipped_counts["argmax_event"] += 1
            return flat_logits

        event_ids = [
            int(token_id)
            for token_id in self.vocab.event_token_ids
            if 0 <= int(token_id) < int(valid_mask.numel()) and bool(valid_mask[int(token_id)].item())
        ]
        if not event_ids:
            self.skipped_counts["no_valid_event"] += 1
            return flat_logits
        event_id_tensor = torch.tensor(event_ids, dtype=torch.long, device=masked.device)
        event_logits = masked.index_select(0, event_id_tensor)
        best_event_index = int(torch.argmax(event_logits).item())
        best_event_id = event_ids[best_event_index]
        best_event_logit = float(event_logits[best_event_index].item())
        argmax_logit = float(masked[argmax_id].item())
        best_event_rank = int(torch.sum(masked > best_event_logit).item()) + 1
        best_event_margin = best_event_logit - argmax_logit
        if best_event_rank > self.max_event_rank:
            self.skipped_counts["rank_too_low"] += 1
            return flat_logits
        if best_event_margin < self.min_event_margin:
            self.skipped_counts["margin_too_low"] += 1
            return flat_logits
        self.candidate_count += 1
        window_start = int(step.write_start_ms)
        if self.forced_count_by_window_start[window_start] >= self.max_forced_events_per_window:
            self.skipped_counts["window_budget"] += 1
            return flat_logits
        last_forced_ms = self.last_forced_ms_by_window.get(window_start)
        if last_forced_ms is not None and current_ms - int(last_forced_ms) < self.min_forced_event_gap_ms:
            self.skipped_counts["forced_gap"] += 1
            return flat_logits

        adjusted = flat_logits.clone()
        adjusted[best_event_id] = max(float(adjusted[best_event_id].item()), argmax_logit + self.force_margin)
        self.forced_count += 1
        self.forced_by_window[window_label] += 1
        self.forced_count_by_window_start[window_start] += 1
        self.last_forced_ms_by_window[window_start] = current_ms
        if len(self.examples) < self.max_examples:
            self.examples.append(
                {
                    "step": int(step.token_index),
                    "current_ms": current_ms,
                    "window": window_label,
                    "argmax_token": self.vocab.token_name(argmax_id),
                    "best_event_token": self.vocab.token_name(best_event_id),
                    "best_event_rank": best_event_rank,
                    "best_event_margin_vs_argmax": best_event_margin,
                }
            )
        return adjusted

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_count": int(self.step_count),
            "candidate_count": int(self.candidate_count),
            "forced_count": int(self.forced_count),
            "forced_by_window": dict(sorted(self.forced_by_window.items())),
            "forced_count_by_window_start": {str(key): int(value) for key, value in sorted(self.forced_count_by_window_start.items())},
            "skipped_counts": dict(sorted(self.skipped_counts.items())),
            "examples": list(self.examples),
        }


def run_selective_event_gate_smoke(
    *,
    trace_summary_path: Path,
    baseline_summary_path: Path,
    summary_output_path: Path,
    report_output_path: Path,
    work_dir: Path,
    device: str = "auto",
    gate_start_ms: int = GATE_START_MS,
    max_event_rank: int = OPPORTUNITY_MAX_RANK,
    min_event_margin: float = OPPORTUNITY_MIN_MARGIN,
    max_forced_events_per_window: int = MAX_FORCED_EVENTS_PER_WINDOW,
    min_forced_event_gap_ms: int = MIN_FORCED_EVENT_GAP_MS,
) -> dict[str, Any]:
    start = time.monotonic()
    trace_summary = _load_json_object(trace_summary_path)
    baseline_summary = _load_json_object(baseline_summary_path)
    baseline_by_case_id = _baseline_by_case_id(baseline_summary)
    cases = [_mapping(result).get("case") for result in trace_summary.get("case_results", ())]
    cases = [dict(case) for case in cases if isinstance(case, Mapping)]
    if not cases:
        raise ValueError("trace summary does not contain selected cases")

    case_results: list[dict[str, Any]] = []
    for case in cases:
        case_results.append(
            run_gate_case(
                case=case,
                baseline_row=baseline_by_case_id[str(case["case_id"])],
                work_dir=work_dir,
                device=device,
                gate_start_ms=int(gate_start_ms),
                max_event_rank=int(max_event_rank),
                min_event_margin=float(min_event_margin),
                max_forced_events_per_window=int(max_forced_events_per_window),
                min_forced_event_gap_ms=int(min_forced_event_gap_ms),
            )
        )
    summary = summarize_smoke(
        trace_summary_path=trace_summary_path,
        baseline_summary_path=baseline_summary_path,
        work_dir=work_dir,
        elapsed_s=time.monotonic() - start,
        case_results=case_results,
        gate_config={
            "gate_start_ms": int(gate_start_ms),
            "max_event_rank": int(max_event_rank),
            "min_event_margin": float(min_event_margin),
            "max_forced_events_per_window": int(max_forced_events_per_window),
            "min_forced_event_gap_ms": int(min_forced_event_gap_ms),
        },
    )
    write_summary_json(summary, summary_output_path)
    write_report(summary, report_output_path)
    return summary


def run_gate_case(
    *,
    case: Mapping[str, Any],
    baseline_row: Mapping[str, Any],
    work_dir: Path,
    device: str,
    gate_start_ms: int,
    max_event_rank: int,
    min_event_margin: float,
    max_forced_events_per_window: int,
    min_forced_event_gap_ms: int,
) -> dict[str, Any]:
    manifest = _mapping(case.get("manifest"))
    baseline_summary_path = Path(str(manifest.get("summary_path")))
    baseline_case_summary = _load_json_object(baseline_summary_path)
    config = _mapping(baseline_case_summary.get("config"))
    case_id = str(case["case_id"])
    output_dir = work_dir / "rollouts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_summary = output_dir / f"{case_id}_selective_gate_summary.json"
    output_report = output_dir / f"{case_id}_selective_gate_report.md"
    vocab = MapperV3Vocab()
    gate = SelectiveEventGate(
        vocab=vocab,
        gate_start_ms=int(gate_start_ms),
        max_event_rank=int(max_event_rank),
        min_event_margin=float(min_event_margin),
        max_forced_events_per_window=int(max_forced_events_per_window),
        min_forced_event_gap_ms=int(min_forced_event_gap_ms),
    )
    candidate_summary = run_trained_v3_runtime_rollout_smoke(
        mapper_checkpoint_path=str(config["mapper_checkpoint_path"]),
        control_checkpoint_path=str(config["control_checkpoint_path"]),
        output_summary_path=output_summary,
        output_report_path=output_report,
        device_name=device,
        chart_end_ms=int(config.get("chart_end_ms", manifest.get("chart_end_ms", 16000))),
        max_tokens_per_window=int(config.get("max_tokens_per_window", 512)),
        audio_path=str(config.get("audio_path", manifest.get("audio_path"))),
        normalized_difficulty=float(config.get("normalized_difficulty", manifest.get("normalized_difficulty", 0.0))),
        include_control_attention_kv_cache=bool(config.get("include_control_attention_kv_cache", False)),
        temperature=float(config.get("temperature", 0.0)),
        top_p=config.get("top_p"),
        seed=int(config.get("seed", 1337)),
        real_audio=bool(config.get("real_audio", True)),
        audio_length_ms=_int(config.get("audio_length_ms")),
        beatthis_device=config.get("beatthis_device"),
        beatthis_float16=bool(config.get("beatthis_float16", False)),
        time_shift_length_penalty_alpha=float(config.get("time_shift_length_penalty_alpha", 0.0)),
        time_shift_delta_penalty_alpha=float(config.get("time_shift_delta_penalty_alpha", 0.0)),
        collect_logit_diagnostics=True,
        logit_top_k=5,
        logit_max_examples=64,
        logits_transform=gate,
        timepoint_preview_limit=4096,
    )
    chart_end_ms = int(candidate_summary["rollout"]["chart_end_ms"])
    generated_timepoints = _generated_timepoints(candidate_summary)
    reference_timepoints = _reference_timepoints(Path(str(manifest["beatmap_path"])), chart_end_ms=chart_end_ms)
    candidate_metrics = compute_time_metrics(
        generated_times=[int(timepoint["time_ms"]) for timepoint in generated_timepoints],
        reference_times=[int(timepoint.time_ms) for timepoint in reference_timepoints],
        chart_end_ms=chart_end_ms,
    )
    comparison = compare_to_baseline(candidate_metrics=candidate_metrics, baseline_row=baseline_row)
    role = str(case.get("role"))
    gate_stats = gate.to_dict()
    checks = _case_checks(
        role=role,
        candidate_summary=candidate_summary,
        candidate_metrics=candidate_metrics,
        comparison=comparison,
    )
    return {
        "case": dict(case),
        "candidate_summary_path": output_summary.as_posix(),
        "candidate_report_path": output_report.as_posix(),
        "gate_stats": gate_stats,
        "candidate": {
            "rollout": _compact_rollout(candidate_summary),
            "metrics": candidate_metrics,
        },
        "baseline": _baseline_projection(baseline_row),
        "comparison": comparison,
        "checks": checks,
    }


def compute_time_metrics(
    *,
    generated_times: Sequence[int],
    reference_times: Sequence[int],
    chart_end_ms: int,
) -> dict[str, Any]:
    generated = sorted(int(time_ms) for time_ms in generated_times if 0 <= int(time_ms) <= int(chart_end_ms))
    reference = sorted(int(time_ms) for time_ms in reference_times if 0 <= int(time_ms) <= int(chart_end_ms))
    second_window_count = sum(1 for time_ms in generated if time_ms >= GATE_START_MS)
    boundary_start = max(0, int(chart_end_ms) - 100)
    boundary_count = sum(1 for time_ms in generated if time_ms >= boundary_start)
    deltas = [b - a for a, b in zip(generated, generated[1:])]
    duplicate_or_nonincreasing = sum(1 for delta in deltas if delta <= 0)
    positive_deltas = [delta for delta in deltas if delta > 0]
    spacing_counts = Counter(str(delta) for delta in positive_deltas)
    dominant_spacing_ms = None
    dominant_spacing_ratio = 0.0
    if positive_deltas:
        dominant_spacing_ms, dominant_count = Counter(positive_deltas).most_common(1)[0]
        dominant_spacing_ratio = float(dominant_count) / float(len(positive_deltas))
    timing = timing_match(generated, reference, tolerance_ms=TIMING_F1_TOLERANCE_MS)
    return {
        "generated_event_count": len(generated),
        "reference_event_count": len(reference),
        "event_count_ratio": _safe_ratio(len(generated), len(reference)),
        "second_window_event_count": second_window_count,
        "second_window_event_share": _safe_ratio(second_window_count, len(generated)),
        "boundary_event_count": boundary_count,
        "boundary_event_ratio": _safe_ratio(boundary_count, len(generated)),
        "duplicate_or_nonincreasing_spacing_ratio": _safe_ratio(duplicate_or_nonincreasing, max(0, len(generated) - 1)),
        "dominant_spacing_ms": dominant_spacing_ms,
        "dominant_spacing_ratio": dominant_spacing_ratio,
        "spacing_counts": dict(sorted(spacing_counts.items(), key=lambda item: (-int(item[1]), int(item[0])))),
        "first_12_generated_times": generated[:12],
        "last_12_generated_times": generated[-12:],
        "first_12_reference_times": reference[:12],
        "last_12_reference_times": reference[-12:],
        "timing_match_100ms": timing,
    }


def timing_match(generated: Sequence[int], reference: Sequence[int], *, tolerance_ms: int) -> dict[str, Any]:
    generated_sorted = sorted(int(value) for value in generated)
    reference_sorted = sorted(int(value) for value in reference)
    gen_index = 0
    ref_index = 0
    matched = 0
    tolerance = int(tolerance_ms)
    while gen_index < len(generated_sorted) and ref_index < len(reference_sorted):
        gen_time = generated_sorted[gen_index]
        ref_time = reference_sorted[ref_index]
        delta = gen_time - ref_time
        if abs(delta) <= tolerance:
            matched += 1
            gen_index += 1
            ref_index += 1
        elif gen_time < ref_time - tolerance:
            gen_index += 1
        else:
            ref_index += 1
    precision = _safe_ratio(matched, len(generated_sorted))
    recall = _safe_ratio(matched, len(reference_sorted))
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "tolerance_ms": tolerance,
        "matched_generated": matched,
        "matched_reference": matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compare_to_baseline(*, candidate_metrics: Mapping[str, Any], baseline_row: Mapping[str, Any]) -> dict[str, Any]:
    baseline_timing = _mapping(baseline_row.get("timing_match_100ms"))
    candidate_timing = _mapping(candidate_metrics.get("timing_match_100ms"))
    return {
        "event_count_ratio_delta": _float(candidate_metrics.get("event_count_ratio")) - _float(baseline_row.get("event_count_ratio")),
        "second_window_event_share_delta": _float(candidate_metrics.get("second_window_event_share")) - _float(baseline_row.get("second_window_event_share")),
        "boundary_event_ratio_delta": _float(candidate_metrics.get("boundary_event_ratio")) - _float(baseline_row.get("boundary_event_ratio")),
        "duplicate_or_nonincreasing_spacing_ratio_delta": _float(candidate_metrics.get("duplicate_or_nonincreasing_spacing_ratio")) - _float(baseline_row.get("duplicate_or_nonincreasing_spacing_ratio")),
        "dominant_spacing_ratio_delta": _float(candidate_metrics.get("dominant_spacing_ratio")) - _float(baseline_row.get("dominant_spacing_ratio")),
        "f1_delta": _float(candidate_timing.get("f1")) - _float(baseline_timing.get("f1")),
    }


def summarize_smoke(
    *,
    trace_summary_path: Path,
    baseline_summary_path: Path,
    work_dir: Path,
    elapsed_s: float,
    case_results: Sequence[Mapping[str, Any]],
    gate_config: Mapping[str, Any],
) -> dict[str, Any]:
    low_bias = [result for result in case_results if _mapping(result.get("case")).get("role") == "low_bias_starved"]
    pass_like = [result for result in case_results if _mapping(result.get("case")).get("role") == "pass_like_control"]
    positive_low_bias = [result for result in low_bias if bool(_mapping(result.get("checks")).get("positive_low_bias_signal"))]
    illegal = [result for result in case_results if not bool(_mapping(result.get("checks")).get("legal"))]
    pass_like_regressed = [result for result in pass_like if bool(_mapping(result.get("checks")).get("pass_like_regressed"))]
    overproduced = [result for result in case_results if bool(_mapping(result.get("checks")).get("overproduced"))]
    decision = _decision(
        low_bias_count=len(low_bias),
        positive_low_bias_count=len(positive_low_bias),
        illegal_count=len(illegal),
        pass_like_regressed_count=len(pass_like_regressed),
        overproduced_count=len(overproduced),
    )
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar v3 selective event-gate smoke",
        "elapsed_s": float(elapsed_s),
        "inputs": {
            "trace_summary": trace_summary_path.as_posix(),
            "baseline_summary": baseline_summary_path.as_posix(),
        },
        "work_dir": work_dir.as_posix(),
        "gate_config": dict(gate_config),
        "thresholds": {
            "second_share_delta_signal": SECOND_SHARE_DELTA_SIGNAL,
            "max_event_count_ratio": MAX_EVENT_COUNT_RATIO,
            "max_f1_regression": MAX_F1_REGRESSION,
            "max_passlike_event_ratio_delta": MAX_PASSLIKE_EVENT_RATIO_DELTA,
        },
        "aggregate": {
            "case_count": len(case_results),
            "low_bias_case_count": len(low_bias),
            "positive_low_bias_count": len(positive_low_bias),
            "illegal_case_count": len(illegal),
            "pass_like_regressed_count": len(pass_like_regressed),
            "overproduced_count": len(overproduced),
        },
        "case_results": list(case_results),
        "decision": decision,
    }


def write_summary_json(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    decision = _mapping(summary.get("decision"))
    aggregate = _mapping(summary.get("aggregate"))
    lines = [
        "# Target Grammar v3 Selective Event-Gate Smoke Result Report",
        "",
        "## Scope",
        "",
        "This runtime-backed smoke tests an eval-only selective event gate on the five cases selected by the low-bias trace oracle. It does not change tokenizer, grammar, model weights, training, or runtime defaults.",
        "",
        "## Result",
        "",
        f"Decision: `{decision.get('route')}`.",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Recommended next step: {decision.get('next_step')}",
        f"- Positive low-bias cases: `{aggregate.get('positive_low_bias_count')}` / `{aggregate.get('low_bias_case_count')}`",
        f"- Illegal cases: `{aggregate.get('illegal_case_count')}`",
        f"- Overproduced cases: `{aggregate.get('overproduced_count')}`",
        f"- Pass-like regressed cases: `{aggregate.get('pass_like_regressed_count')}`",
        "",
        "## Case Table",
        "",
        "| Role | Case | legal | forced | 2nd share base->gate | event ratio base->gate | F1 base->gate | duplicate | boundary | positive |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in summary.get("case_results", ()):
        result_map = _mapping(result)
        case = _mapping(result_map.get("case"))
        baseline = _mapping(result_map.get("baseline"))
        candidate = _mapping(result_map.get("candidate"))
        metrics = _mapping(candidate.get("metrics"))
        checks = _mapping(result_map.get("checks"))
        gate_stats = _mapping(result_map.get("gate_stats"))
        lines.append(
            "| {role} | `{case_id}` | {legal} | {forced} | {base_second}->{cand_second} | {base_ratio}->{cand_ratio} | {base_f1}->{cand_f1} | {dup} | {boundary} | {positive} |".format(
                role=case.get("role"),
                case_id=case.get("case_id"),
                legal=checks.get("legal"),
                forced=gate_stats.get("forced_count"),
                base_second=_fmt(baseline.get("second_window_event_share")),
                cand_second=_fmt(metrics.get("second_window_event_share")),
                base_ratio=_fmt(baseline.get("event_count_ratio")),
                cand_ratio=_fmt(metrics.get("event_count_ratio")),
                base_f1=_fmt(_mapping(baseline.get("timing_match_100ms")).get("f1")),
                cand_f1=_fmt(_mapping(metrics.get("timing_match_100ms")).get("f1")),
                dup=_fmt(metrics.get("duplicate_or_nonincreasing_spacing_ratio")),
                boundary=_fmt(metrics.get("boundary_event_ratio")),
                positive=checks.get("positive_low_bias_signal"),
            )
        )
    lines.extend(
        [
            "",
            "## Gate Examples",
            "",
        ]
    )
    for result in summary.get("case_results", ()):
        result_map = _mapping(result)
        case = _mapping(result_map.get("case"))
        gate_stats = _mapping(result_map.get("gate_stats"))
        examples = gate_stats.get("examples")
        if not isinstance(examples, list) or not examples:
            continue
        lines.append(f"### `{case.get('case_id')}`")
        for row in examples[:8]:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                "- step={step}, ms={ms}, argmax={argmax}, forced={event}, rank={rank}, margin={margin}".format(
                    step=row.get("step"),
                    ms=row.get("current_ms"),
                    argmax=row.get("argmax_token"),
                    event=row.get("best_event_token"),
                    rank=_fmt(row.get("best_event_rank")),
                    margin=_fmt(row.get("best_event_margin_vs_argmax")),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(decision.get("interpretation")),
            "",
            "## Next Step",
            "",
            str(decision.get("next_step")),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _case_checks(
    *,
    role: str,
    candidate_summary: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    rollout = _mapping(candidate_summary.get("rollout"))
    legal = bool(rollout.get("completed")) and not bool(rollout.get("dead_end")) and not bool(rollout.get("max_tokens_exceeded"))
    event_count_ratio = _float(candidate_metrics.get("event_count_ratio"))
    f1_delta = _float(comparison.get("f1_delta"))
    second_delta = _float(comparison.get("second_window_event_share_delta"))
    overproduced = event_count_ratio > MAX_EVENT_COUNT_RATIO
    positive_low_bias = (
        role == "low_bias_starved"
        and legal
        and second_delta >= SECOND_SHARE_DELTA_SIGNAL
        and not overproduced
        and f1_delta >= -MAX_F1_REGRESSION
    )
    pass_like_regressed = (
        role == "pass_like_control"
        and (f1_delta < -MAX_F1_REGRESSION or _float(comparison.get("event_count_ratio_delta")) > MAX_PASSLIKE_EVENT_RATIO_DELTA)
    )
    return {
        "legal": legal,
        "overproduced": overproduced,
        "positive_low_bias_signal": positive_low_bias,
        "pass_like_regressed": pass_like_regressed,
    }


def _decision(
    *,
    low_bias_count: int,
    positive_low_bias_count: int,
    illegal_count: int,
    pass_like_regressed_count: int,
    overproduced_count: int,
) -> dict[str, Any]:
    if illegal_count:
        return {
            "route": "MUTATE_GATE_LEGALITY",
            "reason": f"{illegal_count} selected cases failed legality under the gate",
            "interpretation": "The selective gate is not stable enough for broader rollout.",
            "next_step": "Do not run full32; inspect the illegal cases or pivot to v2.1 grammar work.",
        }
    if pass_like_regressed_count:
        return {
            "route": "KILL_SELECTIVE_EVENT_GATE",
            "reason": f"{pass_like_regressed_count} pass-like controls regressed",
            "interpretation": "The selective gate affects non-target cases too much.",
            "next_step": "Pivot to v2.1 grammar improvement or training-side v3 instrumentation.",
        }
    if overproduced_count:
        return {
            "route": "MUTATE_GATE_BUDGET",
            "reason": f"{overproduced_count} cases exceeded the event-count ratio guard",
            "interpretation": "The gate can move continuation but needs stricter budget calibration before any wider audit.",
            "next_step": "Run a small threshold/cap sensitivity diagnostic on the same five cases.",
        }
    if positive_low_bias_count >= max(2, math.ceil(0.5 * low_bias_count)):
        return {
            "route": "TEST_FULL32_SELECTIVE_GATE",
            "reason": f"{positive_low_bias_count}/{low_bias_count} low-bias starved cases passed the positive signal",
            "interpretation": "The selective gate improved the targeted low-bias continuation cases without tripping smoke guards.",
            "next_step": "Create and run a full32 selective-gate audit before any replacement claim.",
        }
    if positive_low_bias_count > 0:
        return {
            "route": "TEST_GATE_SENSITIVITY",
            "reason": f"only {positive_low_bias_count}/{low_bias_count} low-bias cases passed",
            "interpretation": "The gate has partial signal but is not strong enough as configured.",
            "next_step": "Run one threshold/cap sensitivity diagnostic or pivot to v2.1 grammar improvement.",
        }
    return {
        "route": "MUTATE_TO_V2_1_GRAMMAR",
        "reason": "no low-bias starved cases passed the selective-gate signal",
        "interpretation": "The trace opportunities did not translate into safe rollout improvement.",
        "next_step": "Pivot to a v2.1 grammar improvement card or training-side v3 instrumentation.",
    }


def _generated_timepoints(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rollout = _mapping(summary.get("rollout"))
    timepoints = rollout.get("timepoints")
    if not isinstance(timepoints, list):
        raise ValueError("rollout summary does not contain timepoints")
    timepoint_count = _int(rollout.get("timepoint_count")) or 0
    if len(timepoints) < timepoint_count:
        raise ValueError(f"rollout timepoints preview is truncated: {len(timepoints)} < {timepoint_count}")
    return [dict(item) for item in timepoints if isinstance(item, Mapping)]


def _reference_timepoints(beatmap_path: Path, *, chart_end_ms: int) -> list[Any]:
    hitobjects = parse_mania_hit_objects(beatmap_path, expected_key_count=4)
    return [
        timepoint
        for timepoint in hitobjects_to_mapper_timepoints(hitobjects)
        if 0 <= int(timepoint.time_ms) <= int(chart_end_ms)
    ]


def _baseline_by_case_id(summary: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    runs = summary.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("baseline summary must contain runs")
    rows = {str(row.get("case_id")): row for row in runs if isinstance(row, Mapping) and row.get("case_id")}
    if not rows:
        raise ValueError("baseline summary has no case_id rows")
    return rows


def _baseline_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id": row.get("case_id"),
        "event_count_ratio": row.get("event_count_ratio"),
        "second_window_event_share": row.get("second_window_event_share"),
        "boundary_event_ratio": row.get("boundary_event_ratio"),
        "duplicate_or_nonincreasing_spacing_ratio": row.get("duplicate_or_nonincreasing_spacing_ratio"),
        "dominant_spacing_ratio": row.get("dominant_spacing_ratio"),
        "timing_match_100ms": row.get("timing_match_100ms"),
        "generated_event_count": row.get("generated_event_count"),
        "reference_event_count": row.get("reference_event_count"),
    }


def _compact_rollout(summary: Mapping[str, Any]) -> dict[str, Any]:
    rollout = _mapping(summary.get("rollout"))
    return {
        "window_count": rollout.get("window_count"),
        "token_count": rollout.get("token_count"),
        "timepoint_count": rollout.get("timepoint_count"),
        "completed": rollout.get("completed"),
        "dead_end": rollout.get("dead_end"),
        "max_tokens_exceeded": rollout.get("max_tokens_exceeded"),
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(number):
        return 0.0
    return number


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    denominator = float(denominator)
    if denominator == 0.0:
        return 0.0
    return float(numerator) / denominator


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    return f"{float(value):.6f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a selective event-gate smoke on v3 low-bias traced cases.")
    parser.add_argument("--trace-summary", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--work-dir", default="artifacts/tmp/mapper_v3_selective_event_gate_smoke")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--gate-start-ms", type=int, default=GATE_START_MS)
    parser.add_argument("--max-event-rank", type=int, default=OPPORTUNITY_MAX_RANK)
    parser.add_argument("--min-event-margin", type=float, default=OPPORTUNITY_MIN_MARGIN)
    parser.add_argument("--max-forced-events-per-window", type=int, default=MAX_FORCED_EVENTS_PER_WINDOW)
    parser.add_argument("--min-forced-event-gap-ms", type=int, default=MIN_FORCED_EVENT_GAP_MS)
    args = parser.parse_args(argv)
    summary = run_selective_event_gate_smoke(
        trace_summary_path=Path(args.trace_summary),
        baseline_summary_path=Path(args.baseline_summary),
        summary_output_path=Path(args.summary_output),
        report_output_path=Path(args.report_output),
        work_dir=Path(args.work_dir),
        device=str(args.device),
        gate_start_ms=int(args.gate_start_ms),
        max_event_rank=int(args.max_event_rank),
        min_event_margin=float(args.min_event_margin),
        max_forced_events_per_window=int(args.max_forced_events_per_window),
        min_forced_event_gap_ms=int(args.min_forced_event_gap_ms),
    )
    print(
        "mapper_v3_selective_event_gate_smoke_done "
        f"route={summary['decision']['route']} "
        f"positive={summary['aggregate']['positive_low_bias_count']}/{summary['aggregate']['low_bias_case_count']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
