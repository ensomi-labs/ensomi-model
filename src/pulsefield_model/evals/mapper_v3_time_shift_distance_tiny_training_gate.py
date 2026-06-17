from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence as RuntimeSequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from pulsefield_model.evals.mapper_v2_1_anti_rigid_spacing_guard import generated_metrics
from pulsefield_model.models.mapper.shared.tokenizer import hitobjects_to_mapper_timepoints
from pulsefield_model.osu_core.hitobjects import parse_mania_hit_objects


SUMMARY_SCHEMA_VERSION = 1
EXPECTED_V3_CONTRACT = "v3_event_groups"
DEFAULT_REPORT_ROOT = Path("artifacts/reports/audits/mapper_v2_1_grammar")
DEFAULT_SUMMARY_OUTPUT = DEFAULT_REPORT_ROOT / "target_grammar_v3_time_shift_distance_tiny_training_gate_summary.json"
DEFAULT_REPORT_OUTPUT = DEFAULT_REPORT_ROOT / "target_grammar_v3_time_shift_distance_tiny_training_gate_result_report.md"
RIGID_CASE_RATIO_THRESHOLD = 0.95


@dataclass(frozen=True)
class V3TrainingRun:
    label: str
    path: str
    run_name: str | None
    mapper_token_contract: str | None
    dataset_mapper_token_contract: str | None
    completed_steps: int
    is_complete: bool
    final_eval_loss_total: float
    final_eval_loss_token: float
    final_eval_time_shift_distance_loss: float
    final_eval_lambda_time_shift_distance: float
    final_eval_valid_tokens: float
    final_train_loss_total: float
    last_train_loss_total: float
    loss_config: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "path": self.path,
            "run_name": self.run_name,
            "mapper_token_contract": self.mapper_token_contract,
            "dataset_mapper_token_contract": self.dataset_mapper_token_contract,
            "completed_steps": self.completed_steps,
            "is_complete": self.is_complete,
            "final_eval_loss_total": self.final_eval_loss_total,
            "final_eval_loss_token": self.final_eval_loss_token,
            "final_eval_time_shift_distance_loss": self.final_eval_time_shift_distance_loss,
            "final_eval_lambda_time_shift_distance": self.final_eval_lambda_time_shift_distance,
            "final_eval_valid_tokens": self.final_eval_valid_tokens,
            "final_train_loss_total": self.final_train_loss_total,
            "last_train_loss_total": self.last_train_loss_total,
            "loss_config": dict(self.loss_config),
        }


def load_v3_training_run(path: str | Path, *, label: str) -> V3TrainingRun:
    report_path = Path(path)
    report = _read_json(report_path)
    training_config = _mapping(report.get("training_config"))
    dataset = _mapping(training_config.get("dataset")) or _mapping(report.get("dataset"))
    final_eval = _mapping(report.get("final_eval_metrics"))
    final_train = _mapping(report.get("final_train_metrics"))
    last_train = _mapping(report.get("last_train_metrics"))
    loss_config = _mapping(report.get("loss_config"))
    return V3TrainingRun(
        label=str(label),
        path=report_path.as_posix(),
        run_name=_optional_str(report.get("run_name")),
        mapper_token_contract=_optional_str(training_config.get("mapper_token_contract")),
        dataset_mapper_token_contract=_optional_str(dataset.get("mapper_token_contract")),
        completed_steps=_int_value(report.get("completed_steps"), name=f"{label}.completed_steps"),
        is_complete=bool(report.get("is_complete", False)),
        final_eval_loss_total=_metric(
            final_eval,
            "loss/total",
            label=f"{label}.final_eval_metrics",
            allow_non_finite=True,
        ),
        final_eval_loss_token=_metric(
            final_eval,
            "loss/token",
            label=f"{label}.final_eval_metrics",
            allow_non_finite=True,
        ),
        final_eval_time_shift_distance_loss=_metric(
            final_eval,
            "loss/time_shift_distance",
            label=f"{label}.final_eval_metrics",
            default=0.0,
            allow_non_finite=True,
        ),
        final_eval_lambda_time_shift_distance=_metric(
            final_eval,
            "phase/lambda_time_shift_distance",
            label=f"{label}.final_eval_metrics",
            default=_float(loss_config.get("lambda_time_shift_distance")) or 0.0,
        ),
        final_eval_valid_tokens=_metric(final_eval, "token/valid_count", label=f"{label}.final_eval_metrics"),
        final_train_loss_total=_metric(
            final_train,
            "loss/total",
            label=f"{label}.final_train_metrics",
            allow_non_finite=True,
        ),
        last_train_loss_total=_metric(
            last_train,
            "loss/total",
            label=f"{label}.last_train_metrics",
            allow_non_finite=True,
        ),
        loss_config=loss_config,
    )


def run_time_shift_distance_tiny_training_gate(
    *,
    baseline_report: str | Path,
    enabled_report: str | Path,
    summary_output: str | Path = DEFAULT_SUMMARY_OUTPUT,
    report_output: str | Path | None = DEFAULT_REPORT_OUTPUT,
    rollout_pairs_json: str | Path | None = None,
    min_completed_steps: int = 20,
    max_token_loss_regression_ratio: float = 0.10,
    max_total_loss_regression_ratio: float = 0.15,
    max_mean_rigid_delta: float = 0.02,
    max_new_starved_cases: int = 0,
) -> dict[str, Any]:
    baseline = load_v3_training_run(baseline_report, label="baseline")
    enabled = load_v3_training_run(enabled_report, label="enabled")
    rollout_results = (
        []
        if rollout_pairs_json is None
        else load_rollout_pair_results(Path(rollout_pairs_json))
    )
    summary = compare_time_shift_distance_gate(
        baseline=baseline,
        enabled=enabled,
        rollout_results=rollout_results,
        min_completed_steps=min_completed_steps,
        max_token_loss_regression_ratio=max_token_loss_regression_ratio,
        max_total_loss_regression_ratio=max_total_loss_regression_ratio,
        max_mean_rigid_delta=max_mean_rigid_delta,
        max_new_starved_cases=max_new_starved_cases,
        rollout_pairs_json=None if rollout_pairs_json is None else Path(rollout_pairs_json).as_posix(),
    )
    write_summary_json(summary, summary_output)
    if report_output is not None:
        write_report(summary, report_output)
    return summary


def compare_time_shift_distance_gate(
    *,
    baseline: V3TrainingRun,
    enabled: V3TrainingRun,
    rollout_results: Sequence[Mapping[str, Any]] = (),
    min_completed_steps: int = 20,
    max_token_loss_regression_ratio: float = 0.10,
    max_total_loss_regression_ratio: float = 0.15,
    max_mean_rigid_delta: float = 0.02,
    max_new_starved_cases: int = 0,
    rollout_pairs_json: str | None = None,
) -> dict[str, Any]:
    training_metrics = _training_metrics(baseline, enabled)
    training_checks = {
        "baseline_contract_v3": baseline.mapper_token_contract == EXPECTED_V3_CONTRACT,
        "baseline_dataset_contract_v3": baseline.dataset_mapper_token_contract == EXPECTED_V3_CONTRACT,
        "enabled_contract_v3": enabled.mapper_token_contract == EXPECTED_V3_CONTRACT,
        "enabled_dataset_contract_v3": enabled.dataset_mapper_token_contract == EXPECTED_V3_CONTRACT,
        "baseline_completed_steps": int(baseline.completed_steps) >= int(min_completed_steps),
        "enabled_completed_steps": int(enabled.completed_steps) >= int(min_completed_steps),
        "baseline_complete_flag": bool(baseline.is_complete),
        "enabled_complete_flag": bool(enabled.is_complete),
        "baseline_total_loss_finite": math.isfinite(float(baseline.final_eval_loss_total)),
        "enabled_total_loss_finite": math.isfinite(float(enabled.final_eval_loss_total)),
        "baseline_token_loss_finite": math.isfinite(float(baseline.final_eval_loss_token)),
        "enabled_token_loss_finite": math.isfinite(float(enabled.final_eval_loss_token)),
        "baseline_lambda_disabled": abs(float(baseline.final_eval_lambda_time_shift_distance)) <= 1e-12,
        "enabled_lambda_positive": float(enabled.final_eval_lambda_time_shift_distance) > 0.0,
        "enabled_time_shift_distance_loss_finite": math.isfinite(float(enabled.final_eval_time_shift_distance_loss)),
        "enabled_time_shift_distance_loss_positive": float(enabled.final_eval_time_shift_distance_loss) > 0.0,
        "token_loss_not_materially_worse": training_metrics["token_loss_regression_ratio"]
        <= float(max_token_loss_regression_ratio),
        "total_loss_not_materially_worse": training_metrics["total_loss_regression_ratio"]
        <= float(max_total_loss_regression_ratio),
        "valid_tokens_match": abs(baseline.final_eval_valid_tokens - enabled.final_eval_valid_tokens) <= 1e-6,
    }
    rollout_aggregate = aggregate_rollout_results(rollout_results)
    rollout_checks = None
    if rollout_results:
        rollout_checks = {
            "rollout_pair_count_positive": int(rollout_aggregate["pair_count"]) > 0,
            "baseline_all_legal": bool(rollout_aggregate["baseline"]["all_legal"]),
            "enabled_all_legal": bool(rollout_aggregate["enabled"]["all_legal"]),
            "no_new_starved_cases": int(rollout_aggregate["enabled_vs_baseline"]["new_starved_count"])
            <= int(max_new_starved_cases),
            "mean_rigid_not_worse": float(rollout_aggregate["enabled_vs_baseline"]["mean_dominant_spacing_ratio_delta"])
            <= float(max_mean_rigid_delta),
        }

    training_pass = all(bool(value) for value in training_checks.values())
    rollout_pass = rollout_checks is not None and all(bool(value) for value in rollout_checks.values())
    if not training_pass:
        route = "MUTATE_TIME_SHIFT_DISTANCE_OBJECTIVE"
        reason = "failed training checks: " + ", ".join(key for key, passed in training_checks.items() if not passed)
        next_step = "Do not run rollout; inspect scale/lambda or loss formulation."
    elif rollout_checks is None:
        route = "TEST_ROLLOUT_GATE"
        reason = "training checks passed; rollout-pair diagnostics still missing"
        next_step = "Run a 2-4 case rollout-pair diagnostic with full timepoint previews."
    elif rollout_pass:
        route = "TEST_NEXT_FULL32_TIME_SHIFT_DISTANCE"
        reason = "training and tiny rollout-pair gates passed"
        next_step = "Create a full32 500-step time-shift distance gate before changing defaults."
    else:
        route = "MUTATE_TIME_SHIFT_DISTANCE_OBJECTIVE"
        reason = "failed rollout checks: " + ", ".join(key for key, passed in rollout_checks.items() if not passed)
        next_step = "Mutate the timing objective or scale before a full32 run."

    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar v3 time-shift distance tiny training gate",
        "decision": {"route": route, "reason": reason},
        "config": {
            "min_completed_steps": int(min_completed_steps),
            "max_token_loss_regression_ratio": float(max_token_loss_regression_ratio),
            "max_total_loss_regression_ratio": float(max_total_loss_regression_ratio),
            "max_mean_rigid_delta": float(max_mean_rigid_delta),
            "max_new_starved_cases": int(max_new_starved_cases),
            "rollout_pairs_json": rollout_pairs_json,
        },
        "training_checks": training_checks,
        "training_metrics": training_metrics,
        "rollout_checks": rollout_checks,
        "rollout_aggregate": rollout_aggregate,
        "rollout_results": [dict(row) for row in rollout_results],
        "reports": {
            "baseline": baseline.to_dict(),
            "enabled": enabled.to_dict(),
        },
        "interpretation": _interpretation(route),
        "next_step": next_step,
    }


def load_rollout_pair_results(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    raw_pairs = payload.get("pairs", payload.get("rollout_pairs"))
    if not isinstance(raw_pairs, list):
        raise ValueError(f"rollout pair file must contain a pairs list: {path}")
    results: list[dict[str, Any]] = []
    for index, pair in enumerate(raw_pairs):
        if not isinstance(pair, Mapping):
            raise ValueError(f"rollout pair at index {index} must be an object")
        results.append(load_rollout_pair_result(pair))
    return results


def load_rollout_pair_result(pair: Mapping[str, Any]) -> dict[str, Any]:
    case_id = str(pair.get("case_id") or pair.get("label") or "case")
    baseline_summary_path = Path(str(pair["baseline_summary_path"]))
    enabled_summary_path = Path(str(pair["enabled_summary_path"]))
    beatmap_path = Path(str(pair["beatmap_path"]))
    baseline_summary = _read_json(baseline_summary_path)
    enabled_summary = _read_json(enabled_summary_path)
    chart_end_ms = int(pair.get("chart_end_ms") or _rollout_chart_end_ms(enabled_summary) or _rollout_chart_end_ms(baseline_summary))
    reference_times = _reference_times(beatmap_path, chart_end_ms=chart_end_ms)
    baseline_metrics = _metrics_from_rollout_summary(
        baseline_summary,
        reference_times=reference_times,
        chart_end_ms=chart_end_ms,
    )
    enabled_metrics = _metrics_from_rollout_summary(
        enabled_summary,
        reference_times=reference_times,
        chart_end_ms=chart_end_ms,
    )
    return {
        "case_id": case_id,
        "beatmap_path": beatmap_path.as_posix(),
        "chart_end_ms": chart_end_ms,
        "baseline_summary_path": baseline_summary_path.as_posix(),
        "enabled_summary_path": enabled_summary_path.as_posix(),
        "baseline_legal": _candidate_legal(baseline_summary),
        "enabled_legal": _candidate_legal(enabled_summary),
        "baseline_rollout": _rollout_status(baseline_summary),
        "enabled_rollout": _rollout_status(enabled_summary),
        "baseline_metrics": baseline_metrics,
        "enabled_metrics": enabled_metrics,
        "enabled_vs_baseline": compare_rollout_metrics(enabled_metrics, baseline_metrics),
    }


def compare_rollout_metrics(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    candidate_rigid = _float(candidate.get("dominant_spacing_ratio")) or 0.0
    baseline_rigid = _float(baseline.get("dominant_spacing_ratio")) or 0.0
    return {
        "dominant_spacing_ratio_delta": candidate_rigid - baseline_rigid,
        "f1_delta": _metric_f1(candidate) - _metric_f1(baseline),
        "event_count_ratio_delta": (_float(candidate.get("event_count_ratio")) or 0.0)
        - (_float(baseline.get("event_count_ratio")) or 0.0),
        "second_window_event_share_delta": (_float(candidate.get("second_window_event_share")) or 0.0)
        - (_float(baseline.get("second_window_event_share")) or 0.0),
        "candidate_starved": bool(candidate.get("starved")),
        "baseline_starved": bool(baseline.get("starved")),
        "new_starved": bool(candidate.get("starved")) and not bool(baseline.get("starved")),
    }


def aggregate_rollout_results(rollout_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    baseline = _aggregate_rollout_variant(rollout_results, "baseline")
    enabled = _aggregate_rollout_variant(rollout_results, "enabled")
    comparisons = [_mapping(row.get("enabled_vs_baseline")) for row in rollout_results]
    return {
        "pair_count": int(len(rollout_results)),
        "baseline": baseline,
        "enabled": enabled,
        "enabled_vs_baseline": {
            "new_starved_count": sum(1 for row in comparisons if bool(row.get("new_starved"))),
            "mean_dominant_spacing_ratio_delta": _mean(
                [_float(row.get("dominant_spacing_ratio_delta")) or 0.0 for row in comparisons]
            ),
            "mean_f1_delta": _mean([_float(row.get("f1_delta")) or 0.0 for row in comparisons]),
            "mean_event_count_ratio_delta": _mean(
                [_float(row.get("event_count_ratio_delta")) or 0.0 for row in comparisons]
            ),
            "mean_second_window_share_delta": _mean(
                [_float(row.get("second_window_event_share_delta")) or 0.0 for row in comparisons]
            ),
        },
    }


def write_summary_json(summary: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(summary: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_markdown(summary), encoding="utf-8")


def report_markdown(summary: Mapping[str, Any]) -> str:
    decision = _mapping(summary.get("decision"))
    training_metrics = _mapping(summary.get("training_metrics"))
    reports = _mapping(summary.get("reports"))
    baseline_report = _mapping(reports.get("baseline"))
    enabled_report = _mapping(reports.get("enabled"))
    rollout_aggregate = _mapping(summary.get("rollout_aggregate"))
    rollout_delta = _mapping(rollout_aggregate.get("enabled_vs_baseline"))
    lines = [
        "# Target Grammar v3 Time-Shift Distance Tiny Training Gate Result Report",
        "",
        "## Scope",
        "",
        "This gate compares matched v3 baseline and time-shift-distance-enabled training reports. Optional rollout pairs are used only when supplied. It does not change tokenizer behavior, decode defaults, or production mapper settings.",
        "",
        "## Decision",
        "",
        f"Route: `{decision.get('route')}`.",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Next step: {summary.get('next_step')}",
        "",
        "## Training Reports",
        "",
        f"- baseline: `{baseline_report.get('path')}`",
        f"- enabled: `{enabled_report.get('path')}`",
        f"- baseline steps: `{baseline_report.get('completed_steps')}`",
        f"- enabled steps: `{enabled_report.get('completed_steps')}`",
        f"- enabled eval `loss/time_shift_distance`: `{_fmt(training_metrics.get('enabled_time_shift_distance_loss'))}`",
        f"- token-loss delta: `{_fmt(training_metrics.get('token_loss_delta'))}`",
        f"- token-loss regression ratio: `{_fmt_ratio(training_metrics.get('token_loss_regression_ratio'))}`",
        f"- total-loss delta: `{_fmt(training_metrics.get('total_loss_delta'))}`",
        f"- total-loss regression ratio: `{_fmt_ratio(training_metrics.get('total_loss_regression_ratio'))}`",
        "",
        "## Training Checks",
        "",
        "| Check | Passed |",
        "| --- | ---: |",
    ]
    for key, value in _mapping(summary.get("training_checks")).items():
        lines.append(f"| `{key}` | `{bool(value)}` |")
    rollout_checks = summary.get("rollout_checks")
    lines.extend(["", "## Rollout Diagnostics", ""])
    if isinstance(rollout_checks, Mapping):
        lines.extend(
            [
                f"- rollout pairs: `{rollout_aggregate.get('pair_count')}`",
                f"- new starved cases: `{rollout_delta.get('new_starved_count')}`",
                f"- mean rigid delta: `{_fmt(rollout_delta.get('mean_dominant_spacing_ratio_delta'))}`",
                f"- mean F1 delta: `{_fmt(rollout_delta.get('mean_f1_delta'))}`",
                "",
                "| Check | Passed |",
                "| --- | ---: |",
            ]
        )
        for key, value in rollout_checks.items():
            lines.append(f"| `{key}` | `{bool(value)}` |")
    else:
        lines.append("No rollout-pair file was supplied. A training-pass result routes to `TEST_ROLLOUT_GATE`, not full32.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(summary.get("interpretation")),
            "",
            "## What This Does Not Prove",
            "",
            "- It does not prove v3 full32 replacement readiness.",
            "- It does not prove full-dataset convergence.",
            "- It does not prove generated quality unless rollout pairs are supplied and pass.",
            "",
        ]
    )
    return "\n".join(lines)


def _training_metrics(baseline: V3TrainingRun, enabled: V3TrainingRun) -> dict[str, float]:
    token_delta = enabled.final_eval_loss_token - baseline.final_eval_loss_token
    total_delta = enabled.final_eval_loss_total - baseline.final_eval_loss_total
    return {
        "baseline_total_loss": baseline.final_eval_loss_total,
        "enabled_total_loss": enabled.final_eval_loss_total,
        "total_loss_delta": total_delta,
        "total_loss_regression_ratio": _positive_regression_ratio(
            enabled.final_eval_loss_total,
            baseline.final_eval_loss_total,
        ),
        "baseline_token_loss": baseline.final_eval_loss_token,
        "enabled_token_loss": enabled.final_eval_loss_token,
        "token_loss_delta": token_delta,
        "token_loss_regression_ratio": _positive_regression_ratio(
            enabled.final_eval_loss_token,
            baseline.final_eval_loss_token,
        ),
        "enabled_time_shift_distance_loss": enabled.final_eval_time_shift_distance_loss,
        "baseline_valid_tokens": baseline.final_eval_valid_tokens,
        "enabled_valid_tokens": enabled.final_eval_valid_tokens,
    }


def _positive_regression_ratio(candidate: float, baseline: float) -> float:
    if not math.isfinite(float(candidate)) or not math.isfinite(float(baseline)):
        return float("inf")
    if float(baseline) == 0.0:
        return 0.0 if float(candidate) <= 0.0 else float("inf")
    return max(0.0, (float(candidate) - float(baseline)) / abs(float(baseline)))


def _aggregate_rollout_variant(rollout_results: Sequence[Mapping[str, Any]], variant: str) -> dict[str, Any]:
    metrics = [_mapping(row.get(f"{variant}_metrics")) for row in rollout_results]
    legal_count = sum(1 for row in rollout_results if bool(row.get(f"{variant}_legal")))
    dominant_ratios = [_float(row.get("dominant_spacing_ratio")) or 0.0 for row in metrics]
    return {
        "case_count": int(len(rollout_results)),
        "legal_count": int(legal_count),
        "all_legal": int(legal_count) == int(len(rollout_results)),
        "mean_f1_100ms": _mean([_metric_f1(row) for row in metrics]),
        "mean_dominant_spacing_ratio": _mean(dominant_ratios),
        "median_dominant_spacing_ratio": _median(dominant_ratios),
        "rigid_case_count": sum(1 for value in dominant_ratios if float(value) >= RIGID_CASE_RATIO_THRESHOLD),
        "starved_count": sum(1 for row in metrics if bool(row.get("starved"))),
        "mean_event_count_ratio": _mean([_float(row.get("event_count_ratio")) or 0.0 for row in metrics]),
        "mean_second_window_share": _mean([_float(row.get("second_window_event_share")) or 0.0 for row in metrics]),
    }


def _metrics_from_rollout_summary(
    summary: Mapping[str, Any],
    *,
    reference_times: Sequence[int],
    chart_end_ms: int,
) -> dict[str, Any]:
    generated_times = [int(timepoint["time_ms"]) for timepoint in _generated_timepoints(summary)]
    return generated_metrics(
        generated_times=generated_times,
        reference_times=reference_times,
        chart_end_ms=chart_end_ms,
    )


def _generated_timepoints(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rollout = _mapping(summary.get("rollout"))
    timepoints = rollout.get("timepoints")
    if not isinstance(timepoints, list):
        raise ValueError("rollout summary does not contain timepoints")
    timepoint_count = int(rollout.get("timepoint_count") or 0)
    if len(timepoints) < timepoint_count:
        raise ValueError(f"rollout timepoints preview is truncated: {len(timepoints)} < {timepoint_count}")
    return [dict(item) for item in timepoints if isinstance(item, Mapping)]


def _reference_times(beatmap_path: Path, *, chart_end_ms: int) -> list[int]:
    hitobjects = parse_mania_hit_objects(beatmap_path, expected_key_count=4)
    return [
        int(timepoint.time_ms)
        for timepoint in hitobjects_to_mapper_timepoints(hitobjects)
        if 0 <= int(timepoint.time_ms) <= int(chart_end_ms)
    ]


def _candidate_legal(summary: Mapping[str, Any]) -> bool:
    checks = _mapping(summary.get("checks"))
    rollout = _mapping(summary.get("rollout"))
    return bool(checks.get("rollout_not_dead_end") and checks.get("rollout_not_max_tokens_exceeded") and rollout.get("completed"))


def _rollout_status(summary: Mapping[str, Any]) -> dict[str, Any]:
    rollout = _mapping(summary.get("rollout"))
    windows = rollout.get("windows")
    terminal_ms = None
    if isinstance(windows, list) and windows:
        terminal_ms = _mapping(windows[-1]).get("terminal_ms")
    return {
        "completed": bool(rollout.get("completed")),
        "dead_end": bool(rollout.get("dead_end")),
        "max_tokens_exceeded": bool(rollout.get("max_tokens_exceeded")),
        "window_count": int(rollout.get("window_count") or 0),
        "token_count": int(rollout.get("token_count") or 0),
        "timepoint_count": int(rollout.get("timepoint_count") or 0),
        "terminal_ms": terminal_ms,
    }


def _rollout_chart_end_ms(summary: Mapping[str, Any]) -> int | None:
    rollout = _mapping(summary.get("rollout"))
    value = rollout.get("chart_end_ms") or _mapping(summary.get("config")).get("chart_end_ms")
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _metric_f1(metrics: Mapping[str, Any]) -> float:
    return _float(_mapping(metrics.get("timing_match_100ms")).get("f1")) or 0.0


def _interpretation(route: str) -> str:
    if route == "TEST_NEXT_FULL32_TIME_SHIFT_DISTANCE":
        return (
            "The auxiliary produced a finite training signal and did not regress the tiny rollout legality, "
            "rigid-spacing, or second-window guards. This supports a full32 500-step gate, not replacement."
        )
    if route == "TEST_ROLLOUT_GATE":
        return (
            "The matched training reports pass, but no rollout-pair diagnostics were supplied. The objective has "
            "earned a tiny rollout gate, not full32 escalation."
        )
    return (
        "The time-shift distance objective did not clear the current tiny gate. Do not scale it without changing "
        "the objective, lambda, scale, or training recipe."
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[Any]:
    return value if isinstance(value, RuntimeSequence) and not isinstance(value, (str, bytes, bytearray)) else ()


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _int_value(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got bool")
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _metric(
    metrics: Mapping[str, Any],
    key: str,
    *,
    label: str,
    default: float | None = None,
    allow_non_finite: bool = False,
) -> float:
    if key not in metrics:
        if default is not None:
            return float(default)
        raise ValueError(f"{label} missing required metric {key!r}")
    try:
        value = float(metrics.get(key))  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}.{key} must be numeric, got {metrics.get(key)!r}") from exc
    if not allow_non_finite and not math.isfinite(value):
        raise ValueError(f"{label}.{key} must be finite, got {metrics.get(key)!r}")
    return value


def _float(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: Sequence[float]) -> float:
    return float(math.fsum(values) / len(values)) if values else 0.0


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _fmt(value: object) -> str:
    numeric = _float(value)
    return "nan" if numeric is None else f"{numeric:.6f}"


def _fmt_ratio(value: object) -> str:
    numeric = _float(value)
    return "nan" if numeric is None else f"{100.0 * numeric:.2f}%"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare v3 time-shift distance tiny training gate reports.")
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--enabled-report", required=True)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT.as_posix())
    parser.add_argument("--report-output", default=DEFAULT_REPORT_OUTPUT.as_posix())
    parser.add_argument("--rollout-pairs-json")
    parser.add_argument("--min-completed-steps", type=int, default=20)
    parser.add_argument("--max-token-loss-regression-ratio", type=float, default=0.10)
    parser.add_argument("--max-total-loss-regression-ratio", type=float, default=0.15)
    parser.add_argument("--max-mean-rigid-delta", type=float, default=0.02)
    parser.add_argument("--max-new-starved-cases", type=int, default=0)
    args = parser.parse_args(argv)
    summary = run_time_shift_distance_tiny_training_gate(
        baseline_report=args.baseline_report,
        enabled_report=args.enabled_report,
        summary_output=args.summary_output,
        report_output=args.report_output,
        rollout_pairs_json=args.rollout_pairs_json,
        min_completed_steps=args.min_completed_steps,
        max_token_loss_regression_ratio=args.max_token_loss_regression_ratio,
        max_total_loss_regression_ratio=args.max_total_loss_regression_ratio,
        max_mean_rigid_delta=args.max_mean_rigid_delta,
        max_new_starved_cases=args.max_new_starved_cases,
    )
    print(
        "mapper_v3_time_shift_distance_tiny_training_gate_done "
        f"route={summary['decision']['route']} "
        f"token_loss_delta={summary['training_metrics']['token_loss_delta']:.6f} "
        f"summary={Path(args.summary_output).as_posix()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
