from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from pulsefield_model.evals.mapper_v3_selective_event_gate_smoke import (
    _compact_rollout,
    _float,
    _fmt,
    _generated_timepoints,
    _load_json_object,
    _mapping,
    _reference_timepoints,
    compare_to_baseline,
    compute_time_metrics,
)
from pulsefield_model.evals.mapper_v3_trained_runtime_rollout import run_trained_v3_runtime_rollout_smoke
from pulsefield_model.inference.mapper_v3_rollout import MapperV3GenerationStep
from pulsefield_model.models.mapper.v3 import MapperV3Vocab


SUMMARY_SCHEMA_VERSION = 1
REPORT_DIR = Path("artifacts/reports/audits/mapper_v2_1_grammar")
DEFAULT_CASE_SUMMARY_PATH = REPORT_DIR / "target_grammar_v3_trace_conditioned_spacing_escape_summary.json"
DEFAULT_BASELINE_SUMMARY_PATH = REPORT_DIR / "target_grammar_v3_500step_fixed_slice_wide_audit_summary.json"
DEFAULT_SUMMARY_OUTPUT_PATH = REPORT_DIR / "target_grammar_v3_selective_completion_budget_signal_summary.json"
DEFAULT_REPORT_OUTPUT_PATH = REPORT_DIR / "target_grammar_v3_selective_completion_budget_signal_result_report.md"
DEFAULT_WORK_DIR = Path("artifacts/tmp/mapper_v3_selective_completion_budget_signal_smoke")

GATE_START_MS = 8000
OPPORTUNITY_MAX_RANK = 5
OPPORTUNITY_MIN_MARGIN = -2.0
MIN_FORCED_EVENT_GAP_MS = 80
MAX_FORCED_EVENTS_PER_WINDOW = 48
BUDGET_SLACK_EVENTS = 4
MIN_WINDOW_BUDGET = 0
MAX_WINDOW_BUDGET = 96
SECOND_SHARE_DELTA_SIGNAL = 0.15
MAX_EVENT_COUNT_RATIO = 1.25
MAX_F1_REGRESSION = 0.05
BOUNDARY_RATIO_TOLERANCE = 0.05


class ControlBudgetSelectiveEventGate:
    def __init__(
        self,
        *,
        vocab: MapperV3Vocab,
        gate_start_ms: int = GATE_START_MS,
        max_event_rank: int = OPPORTUNITY_MAX_RANK,
        min_event_margin: float = OPPORTUNITY_MIN_MARGIN,
        max_forced_events_per_window: int = MAX_FORCED_EVENTS_PER_WINDOW,
        min_forced_event_gap_ms: int = MIN_FORCED_EVENT_GAP_MS,
        budget_slack_events: int = BUDGET_SLACK_EVENTS,
        min_window_budget: int = MIN_WINDOW_BUDGET,
        max_window_budget: int = MAX_WINDOW_BUDGET,
        force_margin: float = 1e-3,
        max_examples: int = 24,
    ) -> None:
        self.vocab = vocab
        self.gate_start_ms = int(gate_start_ms)
        self.max_event_rank = int(max_event_rank)
        self.min_event_margin = float(min_event_margin)
        self.max_forced_events_per_window = int(max_forced_events_per_window)
        self.min_forced_event_gap_ms = int(min_forced_event_gap_ms)
        self.budget_slack_events = int(budget_slack_events)
        self.min_window_budget = int(min_window_budget)
        self.max_window_budget = int(max_window_budget)
        self.force_margin = float(force_margin)
        self.max_examples = int(max_examples)
        self.step_count = 0
        self.candidate_count = 0
        self.under_budget_candidate_count = 0
        self.forced_count = 0
        self.skipped_counts: Counter[str] = Counter()
        self.forced_count_by_window_start: Counter[int] = Counter()
        self.last_forced_ms_by_window: dict[int, int] = {}
        self.event_count_seen_by_window: dict[int, int] = {}
        self.density_stats_by_window: dict[int, dict[str, float]] = {}
        self.budget_by_window: dict[int, dict[str, Any]] = {}
        self.examples: list[dict[str, Any]] = []

    def observe_window_batch(self, write_start_ms: int, write_end_ms: int, batch: Mapping[str, Any]) -> None:
        density = batch.get("density_teacher_8s")
        if not isinstance(density, torch.Tensor):
            self.skipped_counts["density_missing"] += 1
            return
        density_tensor = density.detach().to(device="cpu", dtype=torch.float32)
        if density_tensor.ndim == 3:
            density_tensor = density_tensor[0]
        if density_tensor.ndim != 2 or int(density_tensor.shape[-1]) != 1:
            self.skipped_counts["density_bad_shape"] += 1
            return
        if not bool(torch.isfinite(density_tensor).all().item()):
            self.skipped_counts["density_nonfinite"] += 1
            return
        flat = density_tensor.reshape(-1)
        sigmoid_mass = float(torch.sigmoid(flat).sum().item())
        positive_mass = float(torch.clamp(flat, min=0.0).sum().item())
        self.density_stats_by_window[int(write_start_ms)] = {
            "write_start_ms": int(write_start_ms),
            "write_end_ms": int(write_end_ms),
            "frame_count": int(flat.numel()),
            "mean": float(flat.mean().item()) if int(flat.numel()) else 0.0,
            "min": float(flat.min().item()) if int(flat.numel()) else 0.0,
            "max": float(flat.max().item()) if int(flat.numel()) else 0.0,
            "sigmoid_mass": sigmoid_mass,
            "positive_mass": positive_mass,
        }

    def __call__(self, step: MapperV3GenerationStep, logits: torch.Tensor) -> torch.Tensor:
        self.step_count += 1
        window_start = int(step.write_start_ms)
        current_window_events = self._count_generated_events(step.generated_tokens)
        self.event_count_seen_by_window[window_start] = max(
            current_window_events,
            self.event_count_seen_by_window.get(window_start, 0),
        )

        flat_logits = torch.as_tensor(logits, dtype=torch.float32).reshape(-1)
        valid_mask = step.valid_token_mask.to(device=flat_logits.device, dtype=torch.bool).reshape(-1)
        if int(valid_mask.numel()) != int(flat_logits.numel()):
            raise ValueError(f"valid mask has {valid_mask.numel()} tokens but logits have {flat_logits.numel()}")
        masked = flat_logits.masked_fill(~valid_mask, -torch.inf)
        if not bool(torch.isfinite(masked).any().item()):
            self.skipped_counts["no_valid_token"] += 1
            return flat_logits

        current_ms = int(step.state.current_ms)
        if window_start < self.gate_start_ms or current_ms < self.gate_start_ms:
            self.skipped_counts["before_gate_start"] += 1
            return flat_logits
        if current_ms >= int(step.chart_end_ms):
            self.skipped_counts["at_or_after_chart_end"] += 1
            return flat_logits

        budget = self._budget_for_window(window_start)
        if budget is None:
            self.skipped_counts["no_online_budget"] += 1
            return flat_logits
        if current_window_events >= int(budget["event_budget"]):
            self.skipped_counts["budget_exhausted"] += 1
            return flat_logits

        argmax_id = int(torch.argmax(masked).item())
        if self.vocab.is_event_token(argmax_id):
            self.skipped_counts["argmax_event"] += 1
            return flat_logits

        best_event = self._best_valid_event(masked, valid_mask)
        if best_event is None:
            self.skipped_counts["no_valid_event"] += 1
            return flat_logits
        best_event_id, best_event_rank, best_event_margin, argmax_logit = best_event
        if best_event_rank > self.max_event_rank:
            self.skipped_counts["rank_too_low"] += 1
            return flat_logits
        if best_event_margin < self.min_event_margin:
            self.skipped_counts["margin_too_low"] += 1
            return flat_logits

        self.candidate_count += 1
        self.under_budget_candidate_count += 1
        if self.forced_count_by_window_start[window_start] >= self.max_forced_events_per_window:
            self.skipped_counts["forced_cap"] += 1
            return flat_logits
        last_forced_ms = self.last_forced_ms_by_window.get(window_start)
        if last_forced_ms is not None and current_ms - int(last_forced_ms) < self.min_forced_event_gap_ms:
            self.skipped_counts["forced_gap"] += 1
            return flat_logits

        adjusted = flat_logits.clone()
        adjusted[best_event_id] = max(float(adjusted[best_event_id].item()), argmax_logit + self.force_margin)
        self.forced_count += 1
        self.forced_count_by_window_start[window_start] += 1
        self.last_forced_ms_by_window[window_start] = current_ms
        if len(self.examples) < self.max_examples:
            self.examples.append(
                {
                    "step": int(step.token_index),
                    "current_ms": current_ms,
                    "write_start_ms": window_start,
                    "generated_events_in_window": current_window_events,
                    "event_budget": int(budget["event_budget"]),
                    "argmax_token": self.vocab.token_name(argmax_id),
                    "best_event_token": self.vocab.token_name(best_event_id),
                    "best_event_rank": int(best_event_rank),
                    "best_event_margin_vs_argmax": float(best_event_margin),
                }
            )
        return adjusted

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_count": int(self.step_count),
            "candidate_count": int(self.candidate_count),
            "under_budget_candidate_count": int(self.under_budget_candidate_count),
            "forced_count": int(self.forced_count),
            "forced_count_by_window_start": {
                str(key): int(value) for key, value in sorted(self.forced_count_by_window_start.items())
            },
            "event_count_seen_by_window": {
                str(key): int(value) for key, value in sorted(self.event_count_seen_by_window.items())
            },
            "density_stats_by_window": {
                str(key): dict(value) for key, value in sorted(self.density_stats_by_window.items())
            },
            "budget_by_window": {str(key): dict(value) for key, value in sorted(self.budget_by_window.items())},
            "skipped_counts": dict(sorted(self.skipped_counts.items())),
            "examples": list(self.examples),
        }

    def _count_generated_events(self, generated_tokens: Sequence[int]) -> int:
        return sum(1 for token_id in generated_tokens if self.vocab.is_event_token(int(token_id)))

    def _budget_for_window(self, window_start: int) -> dict[str, Any] | None:
        window_start = int(window_start)
        cached = self.budget_by_window.get(window_start)
        if cached is not None:
            return cached
        prev_start = window_start - self.gate_start_ms
        prev_events = self.event_count_seen_by_window.get(prev_start)
        prev_density = self.density_stats_by_window.get(prev_start)
        current_density = self.density_stats_by_window.get(window_start)
        if prev_events is None or prev_density is None or current_density is None:
            return None
        prev_mass = max(float(prev_density.get("sigmoid_mass", 0.0)), 1e-6)
        current_mass = max(float(current_density.get("sigmoid_mass", 0.0)), 0.0)
        density_ratio = current_mass / prev_mass
        raw_budget = float(prev_events) * density_ratio + float(self.budget_slack_events)
        event_budget = int(math.ceil(raw_budget))
        event_budget = max(self.min_window_budget, min(self.max_window_budget, event_budget))
        budget = {
            "window_start_ms": window_start,
            "previous_window_start_ms": prev_start,
            "previous_window_event_count": int(prev_events),
            "previous_density_sigmoid_mass": prev_mass,
            "current_density_sigmoid_mass": current_mass,
            "density_ratio": density_ratio,
            "budget_slack_events": int(self.budget_slack_events),
            "raw_event_budget": raw_budget,
            "event_budget": event_budget,
        }
        self.budget_by_window[window_start] = budget
        return budget

    def _best_valid_event(self, masked: torch.Tensor, valid_mask: torch.Tensor) -> tuple[int, int, float, float] | None:
        event_ids = [
            int(token_id)
            for token_id in self.vocab.event_token_ids
            if 0 <= int(token_id) < int(valid_mask.numel()) and bool(valid_mask[int(token_id)].item())
        ]
        if not event_ids:
            return None
        event_id_tensor = torch.tensor(event_ids, dtype=torch.long, device=masked.device)
        event_logits = masked.index_select(0, event_id_tensor)
        best_event_index = int(torch.argmax(event_logits).item())
        best_event_id = event_ids[best_event_index]
        best_event_logit = float(event_logits[best_event_index].item())
        argmax_id = int(torch.argmax(masked).item())
        argmax_logit = float(masked[argmax_id].item())
        best_event_rank = int(torch.sum(masked > best_event_logit).item()) + 1
        best_event_margin = best_event_logit - argmax_logit
        return best_event_id, best_event_rank, best_event_margin, argmax_logit


def run_selective_completion_budget_signal_smoke(
    *,
    case_summary_path: Path = DEFAULT_CASE_SUMMARY_PATH,
    baseline_summary_path: Path = DEFAULT_BASELINE_SUMMARY_PATH,
    summary_output_path: Path = DEFAULT_SUMMARY_OUTPUT_PATH,
    report_output_path: Path = DEFAULT_REPORT_OUTPUT_PATH,
    work_dir: Path = DEFAULT_WORK_DIR,
    device: str = "auto",
    gate_start_ms: int = GATE_START_MS,
    max_event_rank: int = OPPORTUNITY_MAX_RANK,
    min_event_margin: float = OPPORTUNITY_MIN_MARGIN,
    max_forced_events_per_window: int = MAX_FORCED_EVENTS_PER_WINDOW,
    min_forced_event_gap_ms: int = MIN_FORCED_EVENT_GAP_MS,
    budget_slack_events: int = BUDGET_SLACK_EVENTS,
) -> dict[str, Any]:
    start = time.monotonic()
    case_summary = _load_json_object(case_summary_path)
    baseline_summary = _load_json_object(baseline_summary_path)
    baseline_runs = _baseline_runs_by_case_id(baseline_summary)
    cases = _selected_cases(case_summary)
    case_results = [
        run_budget_case(
            case=case,
            baseline_run=baseline_runs[str(case["case_id"])],
            work_dir=work_dir,
            device=device,
            gate_start_ms=gate_start_ms,
            max_event_rank=max_event_rank,
            min_event_margin=min_event_margin,
            max_forced_events_per_window=max_forced_events_per_window,
            min_forced_event_gap_ms=min_forced_event_gap_ms,
            budget_slack_events=budget_slack_events,
        )
        for case in cases
    ]
    summary = summarize_smoke(
        case_summary_path=case_summary_path,
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
            "budget_slack_events": int(budget_slack_events),
            "budget_source": "previous_window_generated_events_scaled_by_online_density_teacher_8s_sigmoid_mass",
        },
    )
    write_summary_json(summary, summary_output_path)
    write_report(summary, report_output_path)
    return summary


def run_budget_case(
    *,
    case: Mapping[str, Any],
    baseline_run: Mapping[str, Any],
    work_dir: Path,
    device: str,
    gate_start_ms: int,
    max_event_rank: int,
    min_event_margin: float,
    max_forced_events_per_window: int,
    min_forced_event_gap_ms: int,
    budget_slack_events: int,
) -> dict[str, Any]:
    case_id = str(case["case_id"])
    baseline_case_summary = _load_json_object(Path(str(baseline_run["summary_path"])))
    config = _mapping(baseline_case_summary.get("config"))
    output_dir = work_dir / "rollouts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_summary = output_dir / f"{case_id}_budget_gate_summary.json"
    output_report = output_dir / f"{case_id}_budget_gate_report.md"
    vocab = MapperV3Vocab()
    gate = ControlBudgetSelectiveEventGate(
        vocab=vocab,
        gate_start_ms=int(gate_start_ms),
        max_event_rank=int(max_event_rank),
        min_event_margin=float(min_event_margin),
        max_forced_events_per_window=int(max_forced_events_per_window),
        min_forced_event_gap_ms=int(min_forced_event_gap_ms),
        budget_slack_events=int(budget_slack_events),
    )
    candidate_summary = run_trained_v3_runtime_rollout_smoke(
        mapper_checkpoint_path=str(config["mapper_checkpoint_path"]),
        control_checkpoint_path=str(config["control_checkpoint_path"]),
        output_summary_path=output_summary,
        output_report_path=output_report,
        device_name=device,
        chart_end_ms=int(config.get("chart_end_ms", baseline_run.get("chart_end_ms", 16000))),
        max_tokens_per_window=int(config.get("max_tokens_per_window", 512)),
        audio_path=str(config.get("audio_path", baseline_run.get("audio_path"))),
        normalized_difficulty=float(config.get("normalized_difficulty", baseline_run.get("normalized_difficulty", 0.0))),
        include_control_attention_kv_cache=bool(config.get("include_control_attention_kv_cache", False)),
        temperature=float(config.get("temperature", 0.0)),
        top_p=config.get("top_p"),
        seed=int(config.get("seed", 1337)),
        real_audio=bool(config.get("real_audio", True)),
        audio_length_ms=_int_or_none(config.get("audio_length_ms")),
        beatthis_device=config.get("beatthis_device"),
        beatthis_float16=bool(config.get("beatthis_float16", False)),
        time_shift_length_penalty_alpha=float(config.get("time_shift_length_penalty_alpha", 0.0)),
        time_shift_delta_penalty_alpha=float(config.get("time_shift_delta_penalty_alpha", 0.0)),
        collect_logit_diagnostics=True,
        logit_top_k=5,
        logit_max_examples=64,
        logits_transform=gate,
        window_batch_observer=gate.observe_window_batch,
        timepoint_preview_limit=4096,
    )
    chart_end_ms = int(candidate_summary["rollout"]["chart_end_ms"])
    generated_timepoints = _generated_timepoints(candidate_summary)
    reference_timepoints = _reference_timepoints(Path(str(case["beatmap_path"])), chart_end_ms=chart_end_ms)
    candidate_metrics = compute_time_metrics(
        generated_times=[int(timepoint["time_ms"]) for timepoint in generated_timepoints],
        reference_times=[int(timepoint.time_ms) for timepoint in reference_timepoints],
        chart_end_ms=chart_end_ms,
    )
    baseline_metrics = _mapping(case.get("baseline_metrics"))
    comparison = compare_to_baseline(candidate_metrics=candidate_metrics, baseline_row=baseline_metrics)
    checks = _case_checks(
        group=str(case.get("group")),
        candidate_summary=candidate_summary,
        candidate_metrics=candidate_metrics,
        comparison=comparison,
        gate_stats=gate.to_dict(),
    )
    return {
        "case_id": case_id,
        "group": str(case.get("group")),
        "audio_path": case.get("audio_path"),
        "beatmap_path": case.get("beatmap_path"),
        "difficulty": case.get("difficulty"),
        "chart_end_ms": chart_end_ms,
        "candidate_summary_path": output_summary.as_posix(),
        "candidate_report_path": output_report.as_posix(),
        "gate_stats": gate.to_dict(),
        "candidate": {
            "rollout": _compact_rollout(candidate_summary),
            "metrics": candidate_metrics,
        },
        "baseline": dict(baseline_metrics),
        "comparison": comparison,
        "checks": checks,
    }


def summarize_smoke(
    *,
    case_summary_path: Path,
    baseline_summary_path: Path,
    work_dir: Path,
    elapsed_s: float,
    case_results: Sequence[Mapping[str, Any]],
    gate_config: Mapping[str, Any],
) -> dict[str, Any]:
    primary = [row for row in case_results if str(row.get("group")) == "primary"]
    sentinel = [row for row in case_results if str(row.get("group")) == "sentinel"]
    aggregate = {
        "case_count": len(case_results),
        "primary_case_count": len(primary),
        "sentinel_case_count": len(sentinel),
        "primary_positive_count": _count_check(primary, "positive_primary_signal"),
        "primary_event_ratio_over_count": _count_check(primary, "event_ratio_over_cap"),
        "sentinel_event_ratio_over_count": _count_check(sentinel, "event_ratio_over_cap"),
        "illegal_case_count": _count_check(case_results, "illegal"),
        "duplicate_regression_count": _count_check(case_results, "duplicate_regressed"),
        "boundary_regression_count": _count_check(case_results, "boundary_regressed"),
        "budget_source_missing_count": _count_check(case_results, "budget_source_missing"),
        "primary_mean_f1_delta": _mean(_case_float(primary, ("comparison", "f1_delta"))),
        "sentinel_median_event_count_ratio": _median(_case_float(sentinel, ("candidate", "metrics", "event_count_ratio"))),
        "primary_mean_second_window_share_delta": _mean(
            _case_float(primary, ("comparison", "second_window_event_share_delta"))
        ),
        "total_forced_count": sum(_gate_forced_count(row) for row in case_results),
    }
    checks = _aggregate_checks(aggregate)
    decision = _decision(checks=checks, aggregate=aggregate)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar v3 selective completion-budget signal smoke",
        "elapsed_s": float(elapsed_s),
        "inputs": {
            "case_summary": case_summary_path.as_posix(),
            "baseline_summary": baseline_summary_path.as_posix(),
        },
        "work_dir": work_dir.as_posix(),
        "gate_config": dict(gate_config),
        "thresholds": {
            "second_share_delta_signal": SECOND_SHARE_DELTA_SIGNAL,
            "max_event_count_ratio": MAX_EVENT_COUNT_RATIO,
            "max_f1_regression": MAX_F1_REGRESSION,
            "boundary_ratio_tolerance": BOUNDARY_RATIO_TOLERANCE,
        },
        "aggregate": aggregate,
        "aggregate_checks": checks,
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
        "# Target Grammar v3 Selective Completion-Budget Signal Result Report",
        "",
        "## Scope",
        "",
        "This runtime-backed smoke tests a default-off v3 logits transform that promotes rank-near event tokens only when generated-prefix events are below an online control-density budget. It does not change tokenizer, grammar, replay, model weights, training, or runtime defaults.",
        "",
        "## Result",
        "",
        f"Decision: `{decision.get('route')}`.",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Recommended next step: {decision.get('next_step')}",
        f"- Primary positives: `{aggregate.get('primary_positive_count')}` / `{aggregate.get('primary_case_count')}`",
        f"- Primary event-ratio over-cap: `{aggregate.get('primary_event_ratio_over_count')}`",
        f"- Sentinel event-ratio over-cap: `{aggregate.get('sentinel_event_ratio_over_count')}`",
        f"- Sentinel median event ratio: `{_fmt(aggregate.get('sentinel_median_event_count_ratio'))}`",
        f"- Mean primary F1 delta: `{_fmt(aggregate.get('primary_mean_f1_delta'))}`",
        f"- Total forced events: `{aggregate.get('total_forced_count')}`",
        "",
        "## Case Table",
        "",
        "| Group | Case | legal | forced | budget | 2nd share base->candidate | event ratio base->candidate | F1 delta | positive | over cap |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for result in summary.get("case_results", ()):
        row = _mapping(result)
        baseline = _mapping(row.get("baseline"))
        candidate = _mapping(row.get("candidate"))
        metrics = _mapping(candidate.get("metrics"))
        comparison = _mapping(row.get("comparison"))
        checks = _mapping(row.get("checks"))
        gate_stats = _mapping(row.get("gate_stats"))
        budget = _second_window_budget(gate_stats)
        lines.append(
            "| {group} | `{case_id}` | {legal} | {forced} | {budget} | {base_second}->{cand_second} | {base_ratio}->{cand_ratio} | {f1_delta} | {positive} | {over_cap} |".format(
                group=row.get("group"),
                case_id=row.get("case_id"),
                legal=not bool(checks.get("illegal")),
                forced=gate_stats.get("forced_count"),
                budget="n/a" if budget is None else str(int(budget)),
                base_second=_fmt(baseline.get("second_window_event_share")),
                cand_second=_fmt(metrics.get("second_window_event_share")),
                base_ratio=_fmt(baseline.get("event_count_ratio")),
                cand_ratio=_fmt(metrics.get("event_count_ratio")),
                f1_delta=_fmt(comparison.get("f1_delta")),
                positive=checks.get("positive_primary_signal"),
                over_cap=checks.get("event_ratio_over_cap"),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(decision.get("interpretation")),
            "",
            "## Budget Trace Notes",
            "",
            "The transform uses only generated-prefix event counts and `density_teacher_8s` from the runtime control batch. Reference beatmaps are used only after generation for evaluation metrics.",
            "",
            "## Next Step",
            "",
            str(decision.get("next_step")),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _selected_cases(case_summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = case_summary.get("case_results")
    if not isinstance(rows, list) or not rows:
        raise ValueError("case summary must contain case_results")
    cases: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        case_id = row.get("case_id")
        group = row.get("group")
        baseline_metrics = row.get("baseline_metrics")
        if not case_id or group not in {"primary", "sentinel"} or not isinstance(baseline_metrics, Mapping):
            continue
        cases.append(
            {
                "case_id": str(case_id),
                "group": str(group),
                "audio_path": row.get("audio_path"),
                "beatmap_path": row.get("beatmap_path"),
                "difficulty": row.get("difficulty"),
                "chart_end_ms": row.get("chart_end_ms"),
                "baseline_metrics": dict(baseline_metrics),
            }
        )
    if len(cases) != 10:
        raise ValueError(f"expected 10 selected cases, found {len(cases)}")
    return cases


def _baseline_runs_by_case_id(summary: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    runs = summary.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("baseline summary must contain runs")
    rows = {str(row.get("case_id")): row for row in runs if isinstance(row, Mapping) and row.get("case_id")}
    if not rows:
        raise ValueError("baseline summary has no case_id rows")
    return rows


def _case_checks(
    *,
    group: str,
    candidate_summary: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    comparison: Mapping[str, Any],
    gate_stats: Mapping[str, Any],
) -> dict[str, Any]:
    rollout = _mapping(candidate_summary.get("rollout"))
    legal = bool(rollout.get("completed")) and not bool(rollout.get("dead_end")) and not bool(rollout.get("max_tokens_exceeded"))
    event_ratio = _float(candidate_metrics.get("event_count_ratio"))
    second_delta = _float(comparison.get("second_window_event_share_delta"))
    f1_delta = _float(comparison.get("f1_delta"))
    duplicate_delta = _float(comparison.get("duplicate_or_nonincreasing_spacing_ratio_delta"))
    boundary_delta = _float(comparison.get("boundary_event_ratio_delta"))
    budget_by_window = _mapping(gate_stats.get("budget_by_window"))
    reached_budget_window = int(rollout.get("window_count") or 0) >= 2
    budget_source_missing = legal and reached_budget_window and "8000" not in budget_by_window
    event_ratio_over_cap = event_ratio > MAX_EVENT_COUNT_RATIO
    duplicate_regressed = duplicate_delta > 0.0
    boundary_regressed = boundary_delta > BOUNDARY_RATIO_TOLERANCE
    positive_primary = (
        group == "primary"
        and legal
        and not event_ratio_over_cap
        and second_delta >= SECOND_SHARE_DELTA_SIGNAL
        and f1_delta >= -MAX_F1_REGRESSION
        and not duplicate_regressed
        and not boundary_regressed
        and not budget_source_missing
    )
    return {
        "illegal": not legal,
        "event_ratio_over_cap": event_ratio_over_cap,
        "duplicate_regressed": duplicate_regressed,
        "boundary_regressed": boundary_regressed,
        "budget_source_missing": budget_source_missing,
        "positive_primary_signal": positive_primary,
    }


def _aggregate_checks(aggregate: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "case_coverage": int(aggregate.get("primary_case_count") or 0) == 6
        and int(aggregate.get("sentinel_case_count") or 0) == 4,
        "budget_source_available": int(aggregate.get("budget_source_missing_count") or 0) == 0,
        "primary_positive_count": int(aggregate.get("primary_positive_count") or 0) >= 4,
        "primary_event_ratio_guard": int(aggregate.get("primary_event_ratio_over_count") or 0) == 0,
        "sentinel_event_ratio_guard": int(aggregate.get("sentinel_event_ratio_over_count") or 0) == 0,
        "sentinel_median_event_ratio_guard": (_float(aggregate.get("sentinel_median_event_count_ratio")) or 0.0)
        <= MAX_EVENT_COUNT_RATIO,
        "legality_guard": int(aggregate.get("illegal_case_count") or 0) == 0,
        "duplicate_guard": int(aggregate.get("duplicate_regression_count") or 0) == 0,
        "boundary_guard": int(aggregate.get("boundary_regression_count") or 0) == 0,
        "mean_primary_f1_guard": (_float(aggregate.get("primary_mean_f1_delta")) or 0.0) >= -MAX_F1_REGRESSION,
    }


def _decision(*, checks: Mapping[str, bool], aggregate: Mapping[str, Any]) -> dict[str, Any]:
    failed = [name for name, passed in checks.items() if not bool(passed)]
    if "budget_source_available" in failed:
        return {
            "route": "MUTATE_BUDGET_SOURCE",
            "reason": "the gate could not compute an online density budget for every selected case",
            "interpretation": "The selected mechanism is invalid until the runtime budget source is observable without target labels.",
            "next_step": "Audit the online control-density budget source before rerunning this gate.",
            "failed_checks": failed,
        }
    if not failed:
        return {
            "route": "TEST_NEXT_FULL32_BUDGET_SIGNAL",
            "reason": "all primary/sentinel guards passed and at least four primary cases improved continuation",
            "interpretation": "The budget-conditioned gate has a bounded positive signal on the adversarial 10-case slice.",
            "next_step": "Create a full32 budget-conditioned signal audit before any training or default change.",
            "failed_checks": [],
        }
    if "primary_event_ratio_guard" in failed or "sentinel_event_ratio_guard" in failed:
        reason = (
            f"event-ratio cap failed: primary_over={aggregate.get('primary_event_ratio_over_count')}, "
            f"sentinel_over={aggregate.get('sentinel_event_ratio_over_count')}"
        )
    elif "primary_positive_count" in failed:
        reason = f"only {aggregate.get('primary_positive_count')}/{aggregate.get('primary_case_count')} primary cases passed"
    else:
        reason = "one or more legality, duplicate, boundary, coverage, sentinel median, or F1 guards failed"
    return {
        "route": "KILL_SELECTIVE_COMPLETION_BUDGET_SIGNAL",
        "reason": reason,
        "interpretation": "The online budget-conditioned local decode repair is not strong enough as configured.",
        "next_step": "Pivot to v2.1 grammar improvement or a structural v3 grammar mutation card.",
        "failed_checks": failed,
    }


def _count_check(rows: Sequence[Mapping[str, Any]], check_name: str) -> int:
    return sum(1 for row in rows if bool(_mapping(row.get("checks")).get(check_name)))


def _case_float(rows: Sequence[Mapping[str, Any]], path: Sequence[str]) -> list[float]:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for key in path:
            value = _mapping(value).get(key)
        values.append(_float(value))
    return values


def _gate_forced_count(row: Mapping[str, Any]) -> int:
    return int(_mapping(row.get("gate_stats")).get("forced_count") or 0)


def _second_window_budget(gate_stats: Mapping[str, Any]) -> int | None:
    budget = _mapping(_mapping(gate_stats.get("budget_by_window")).get("8000"))
    if not budget:
        return None
    try:
        return int(budget.get("event_budget"))
    except (TypeError, ValueError):
        return None


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the v3 selective completion-budget signal smoke.")
    parser.add_argument("--case-summary", default=DEFAULT_CASE_SUMMARY_PATH)
    parser.add_argument("--baseline-summary", default=DEFAULT_BASELINE_SUMMARY_PATH)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT_PATH)
    parser.add_argument("--report-output", default=DEFAULT_REPORT_OUTPUT_PATH)
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--gate-start-ms", type=int, default=GATE_START_MS)
    parser.add_argument("--max-event-rank", type=int, default=OPPORTUNITY_MAX_RANK)
    parser.add_argument("--min-event-margin", type=float, default=OPPORTUNITY_MIN_MARGIN)
    parser.add_argument("--max-forced-events-per-window", type=int, default=MAX_FORCED_EVENTS_PER_WINDOW)
    parser.add_argument("--min-forced-event-gap-ms", type=int, default=MIN_FORCED_EVENT_GAP_MS)
    parser.add_argument("--budget-slack-events", type=int, default=BUDGET_SLACK_EVENTS)
    args = parser.parse_args(argv)
    summary = run_selective_completion_budget_signal_smoke(
        case_summary_path=Path(args.case_summary),
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
        budget_slack_events=int(args.budget_slack_events),
    )
    print(
        "mapper_v3_selective_completion_budget_signal_smoke_done "
        f"route={summary['decision']['route']} "
        f"primary_positive={summary['aggregate']['primary_positive_count']}/"
        f"{summary['aggregate']['primary_case_count']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
