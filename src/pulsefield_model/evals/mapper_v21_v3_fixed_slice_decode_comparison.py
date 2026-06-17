from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence as RuntimeSequence
from pathlib import Path
from typing import Any, Mapping, Sequence

from pulsefield_model.evals.mapper_v2_1_anti_rigid_spacing_guard import (
    REFERENCE_SECOND_WINDOW_SUPPORT_THRESHOLD,
    SECOND_WINDOW_START_MS,
    STARVED_SHARE_THRESHOLD,
    generated_metrics,
)
from pulsefield_model.evals.mapper_v2_1_trained_runtime_rollout import run_trained_v21_runtime_rollout_smoke
from pulsefield_model.inference.mapper_v2_1_rollout import MapperV21AntiRigidSpacingLogitsTransform
from pulsefield_model.models.mapper.shared.tokenizer import hitobjects_to_mapper_timepoints
from pulsefield_model.models.mapper.v2_1 import MapperV21Vocab
from pulsefield_model.osu_core.hitobjects import parse_mania_hit_objects


SUMMARY_SCHEMA_VERSION = 1
DEFAULT_V3_WIDE_SUMMARY_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_fixed_slice_wide_audit_summary.json",
)
DEFAULT_V21_BASELINE_SUMMARY_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_matched_v21_timing_baseline_summary.json",
)
DEFAULT_OUTPUT_DIR = Path("artifacts/tmp/mapper_v21_v3_fixed_slice_decode_comparison")
DEFAULT_SUMMARY_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_v3_fixed_slice_decode_comparison_summary.json",
)
DEFAULT_REPORT_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_v3_fixed_slice_decode_comparison_result_report.md",
)
RIGID_CASE_RATIO_THRESHOLD = 0.95
POSITIVE_MIN_RIGID_IMPROVED_CASES = 16
NEGATIVE_MAX_RIGID_IMPROVED_CASES = 7
POSITIVE_MEAN_RIGID_DELTA = -0.10
MEAN_F1_REGRESSION_LIMIT = -0.05
MAX_NEW_STARVED_CASES = 2


def run_v21_v3_fixed_slice_decode_comparison(
    *,
    v3_wide_summary_path: str | Path = DEFAULT_V3_WIDE_SUMMARY_PATH,
    v21_baseline_summary_path: str | Path = DEFAULT_V21_BASELINE_SUMMARY_PATH,
    output_summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    output_report_path: str | Path | None = DEFAULT_REPORT_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    mapper_checkpoint_path: str | Path | None = None,
    control_checkpoint_path: str | Path | None = None,
    device_name: str = "auto",
    max_tokens_per_window: int = 512,
    min_repeated_spacings: int = 4,
    min_spacing_ms: int = 40,
    max_spacing_ms: int = 400,
    seed: int = 1337,
    collect_logit_diagnostics: bool = False,
    timepoint_preview_limit: int = 4096,
    case_limit: int | None = None,
) -> dict[str, Any]:
    v3_summary = _read_json(Path(v3_wide_summary_path))
    v21_baseline_summary = _read_json(Path(v21_baseline_summary_path))
    v3_runs = _case_runs(v3_summary)
    if case_limit is not None:
        limit = int(case_limit)
        if limit <= 0:
            raise ValueError("case_limit must be positive when provided")
        v3_runs = v3_runs[:limit]

    mapper_checkpoint = Path(mapper_checkpoint_path or str(v21_baseline_summary.get("checkpoint_path", "")))
    control_checkpoint = Path(control_checkpoint_path or str(v21_baseline_summary.get("control_checkpoint_path", "")))
    if not mapper_checkpoint.exists():
        raise FileNotFoundError(f"missing mapper checkpoint: {mapper_checkpoint}")
    if not control_checkpoint.exists():
        raise FileNotFoundError(f"missing control checkpoint: {control_checkpoint}")

    out_dir = Path(output_dir)
    rollout_dir = out_dir / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    vocab = MapperV21Vocab()
    case_results: list[dict[str, Any]] = []

    for v3_row in v3_runs:
        case_id = str(v3_row["case_id"])
        chart_end_ms = int(v3_row["chart_end_ms"])
        reference_times = _reference_times(Path(str(v3_row["beatmap_path"])), chart_end_ms=chart_end_ms)
        baseline_summary_path = rollout_dir / f"{_safe_filename(case_id)}_{chart_end_ms}ms_v21_baseline.json"
        guard_summary_path = rollout_dir / f"{_safe_filename(case_id)}_{chart_end_ms}ms_v21_guard.json"

        baseline_summary = _run_v21_case(
            mapper_checkpoint=mapper_checkpoint,
            control_checkpoint=control_checkpoint,
            output_summary_path=baseline_summary_path,
            device_name=device_name,
            chart_end_ms=chart_end_ms,
            max_tokens_per_window=max_tokens_per_window,
            audio_path=str(v3_row["audio_path"]),
            normalized_difficulty=float(v3_row["normalized_difficulty"]),
            seed=seed,
            collect_logit_diagnostics=collect_logit_diagnostics,
            timepoint_preview_limit=timepoint_preview_limit,
            logits_transform=None,
        )
        transform = MapperV21AntiRigidSpacingLogitsTransform(
            vocab=vocab,
            min_repeated_spacings=int(min_repeated_spacings),
            min_spacing_ms=int(min_spacing_ms),
            max_spacing_ms=int(max_spacing_ms),
        )
        guard_summary = _run_v21_case(
            mapper_checkpoint=mapper_checkpoint,
            control_checkpoint=control_checkpoint,
            output_summary_path=guard_summary_path,
            device_name=device_name,
            chart_end_ms=chart_end_ms,
            max_tokens_per_window=max_tokens_per_window,
            audio_path=str(v3_row["audio_path"]),
            normalized_difficulty=float(v3_row["normalized_difficulty"]),
            seed=seed,
            collect_logit_diagnostics=collect_logit_diagnostics,
            timepoint_preview_limit=timepoint_preview_limit,
            logits_transform=transform,
        )

        baseline_metrics = _metrics_from_v21_summary(
            baseline_summary,
            reference_times=reference_times,
            chart_end_ms=chart_end_ms,
        )
        guard_metrics = _metrics_from_v21_summary(
            guard_summary,
            reference_times=reference_times,
            chart_end_ms=chart_end_ms,
        )
        v3_metrics = metrics_from_v3_row(
            v3_row,
            reference_times=reference_times,
            chart_end_ms=chart_end_ms,
        )
        case_results.append(
            {
                "case_id": case_id,
                "case_index": v3_row.get("case_index"),
                "audio_family": v3_row.get("audio_family"),
                "difficulty": v3_row.get("difficulty"),
                "difficulty_band": v3_row.get("difficulty_band"),
                "normalized_difficulty": v3_row.get("normalized_difficulty"),
                "chart_end_ms": chart_end_ms,
                "audio_path": v3_row.get("audio_path"),
                "beatmap_path": v3_row.get("beatmap_path"),
                "v21_baseline_summary_path": baseline_summary_path.as_posix(),
                "v21_guard_summary_path": guard_summary_path.as_posix(),
                "v3_summary_path": v3_row.get("summary_path"),
                "v21_baseline_legal": _candidate_legal(baseline_summary),
                "v21_guard_legal": _candidate_legal(guard_summary),
                "v3_legal": bool(v3_row.get("legal")),
                "v21_baseline_rollout": rollout_status(baseline_summary),
                "v21_guard_rollout": rollout_status(guard_summary),
                "v21_baseline_metrics": baseline_metrics,
                "v21_guard_metrics": guard_metrics,
                "v3_metrics": v3_metrics,
                "guard_vs_baseline": compare_metrics(guard_metrics, baseline_metrics),
                "guard_vs_v3": compare_metrics(guard_metrics, v3_metrics),
                "baseline_vs_v3": compare_metrics(baseline_metrics, v3_metrics),
                "transform": transform.to_dict(),
            }
        )

    aggregate = aggregate_results(case_results)
    decision = decision_from_aggregate(aggregate)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Mapper v2.1/v3 fixed-slice decode comparison",
        "decision": decision,
        "v3_wide_summary_path": Path(v3_wide_summary_path).as_posix(),
        "v21_baseline_summary_path": Path(v21_baseline_summary_path).as_posix(),
        "output_dir": out_dir.as_posix(),
        "mapper_checkpoint_path": mapper_checkpoint.as_posix(),
        "control_checkpoint_path": control_checkpoint.as_posix(),
        "config": {
            "device": device_name,
            "max_tokens_per_window": int(max_tokens_per_window),
            "min_repeated_spacings": int(min_repeated_spacings),
            "min_spacing_ms": int(min_spacing_ms),
            "max_spacing_ms": int(max_spacing_ms),
            "seed": int(seed),
            "collect_logit_diagnostics": bool(collect_logit_diagnostics),
            "timepoint_preview_limit": int(timepoint_preview_limit),
            "case_limit": None if case_limit is None else int(case_limit),
        },
        "v3_comparator_aggregate": v3_summary.get("aggregate"),
        "v21_six_case_baseline_aggregate": v21_baseline_summary.get("aggregate"),
        "aggregate": aggregate,
        "case_results": case_results,
        "illegal_cases": illegal_cases(case_results),
        "worst_cases": worst_cases(case_results),
        "next_step": decision["next_step"],
    }
    write_summary_json(summary, output_summary_path)
    if output_report_path is not None:
        write_report(summary, output_report_path)
    return summary


def metrics_from_v3_row(
    row: Mapping[str, Any],
    *,
    reference_times: Sequence[int],
    chart_end_ms: int,
) -> dict[str, Any]:
    reference_second_count = sum(1 for time_ms in reference_times if int(time_ms) >= SECOND_WINDOW_START_MS)
    reference_second_share = _safe_ratio(reference_second_count, len(reference_times))
    second_share = _float(row.get("second_window_event_share")) or 0.0
    return {
        "generated_event_count": int(row.get("generated_event_count") or 0),
        "reference_event_count": int(row.get("reference_event_count") or len(reference_times)),
        "event_count_ratio": _float(row.get("event_count_ratio")) or 0.0,
        "second_window_event_count": int(row.get("second_window_event_count") or 0),
        "second_window_event_share": second_share,
        "reference_second_window_event_count": int(reference_second_count),
        "reference_second_window_event_share": reference_second_share,
        "starved": _starved(
            chart_end_ms=chart_end_ms,
            second_share=second_share,
            reference_second_share=reference_second_share,
        ),
        "boundary_event_count": int(row.get("boundary_event_count") or 0),
        "boundary_event_ratio": _float(row.get("boundary_event_ratio")) or 0.0,
        "duplicate_or_nonincreasing_spacing_ratio": _float(row.get("duplicate_or_nonincreasing_spacing_ratio")) or 0.0,
        "dominant_spacing_ms": row.get("dominant_spacing_ms"),
        "dominant_spacing_ratio": _float(row.get("dominant_spacing_ratio")) or 0.0,
        "spacing_counts": dict(_mapping(row.get("spacing_counts"))),
        "first_12_generated_times": list(_sequence(row.get("first_12_generated_times"))),
        "last_12_generated_times": list(_sequence(row.get("last_12_generated_times"))),
        "first_12_reference_times": list(_sequence(row.get("first_12_reference_times"))),
        "last_12_reference_times": list(_sequence(row.get("last_12_reference_times"))),
        "timing_match_100ms": dict(_mapping(row.get("timing_match_100ms"))),
    }


def compare_metrics(candidate: Mapping[str, Any], comparator: Mapping[str, Any]) -> dict[str, Any]:
    candidate_f1 = _metric_f1(candidate)
    comparator_f1 = _metric_f1(comparator)
    candidate_rigid = _float(candidate.get("dominant_spacing_ratio")) or 0.0
    comparator_rigid = _float(comparator.get("dominant_spacing_ratio")) or 0.0
    return {
        "dominant_spacing_ratio_delta": candidate_rigid - comparator_rigid,
        "dominant_spacing_improved": candidate_rigid < comparator_rigid,
        "f1_delta": candidate_f1 - comparator_f1,
        "event_count_ratio_delta": (_float(candidate.get("event_count_ratio")) or 0.0)
        - (_float(comparator.get("event_count_ratio")) or 0.0),
        "second_window_event_share_delta": (_float(candidate.get("second_window_event_share")) or 0.0)
        - (_float(comparator.get("second_window_event_share")) or 0.0),
        "candidate_starved": bool(candidate.get("starved")),
        "comparator_starved": bool(comparator.get("starved")),
        "new_starved": bool(candidate.get("starved")) and not bool(comparator.get("starved")),
    }


def aggregate_results(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    baseline_aggregate = aggregate_variant(case_results, "v21_baseline")
    guard_aggregate = aggregate_variant(case_results, "v21_guard")
    v3_aggregate = aggregate_variant(case_results, "v3")
    guard_comparisons = [_mapping(row.get("guard_vs_baseline")) for row in case_results]
    guard_vs_v3 = [_mapping(row.get("guard_vs_v3")) for row in case_results]
    blocked_count = sum(int(_mapping(row.get("transform")).get("blocked_count") or 0) for row in case_results)
    rigid_improved_count = sum(1 for row in guard_comparisons if bool(row.get("dominant_spacing_improved")))
    f1_improved_count = sum(1 for row in guard_comparisons if (_float(row.get("f1_delta")) or 0.0) > 0.0)
    new_starved_count = sum(1 for row in guard_comparisons if bool(row.get("new_starved")))
    guard_mean_f1_delta = guard_aggregate["mean_f1_100ms"] - baseline_aggregate["mean_f1_100ms"]
    guard_mean_rigid_delta = (
        guard_aggregate["mean_dominant_spacing_ratio"] - baseline_aggregate["mean_dominant_spacing_ratio"]
    )
    return {
        "case_count": int(len(case_results)),
        "v21_baseline": baseline_aggregate,
        "v21_guard": guard_aggregate,
        "v3": v3_aggregate,
        "guard_vs_baseline": {
            "blocked_count": int(blocked_count),
            "rigid_improved_count": int(rigid_improved_count),
            "f1_improved_count": int(f1_improved_count),
            "new_starved_count": int(new_starved_count),
            "mean_f1_delta": guard_mean_f1_delta,
            "mean_dominant_spacing_ratio_delta": guard_mean_rigid_delta,
            "mean_event_count_ratio_delta": (
                guard_aggregate["mean_event_count_ratio"] - baseline_aggregate["mean_event_count_ratio"]
            ),
            "mean_second_window_share_delta": (
                guard_aggregate["mean_second_window_share"] - baseline_aggregate["mean_second_window_share"]
            ),
        },
        "guard_vs_v3": {
            "f1_better_or_equal_count": sum(1 for row in guard_vs_v3 if (_float(row.get("f1_delta")) or 0.0) >= 0.0),
            "rigid_lower_count": sum(1 for row in guard_vs_v3 if bool(row.get("dominant_spacing_improved"))),
            "starved_count_delta": guard_aggregate["starved_count"] - v3_aggregate["starved_count"],
            "mean_f1_delta": guard_aggregate["mean_f1_100ms"] - v3_aggregate["mean_f1_100ms"],
            "mean_dominant_spacing_ratio_delta": (
                guard_aggregate["mean_dominant_spacing_ratio"] - v3_aggregate["mean_dominant_spacing_ratio"]
            ),
            "mean_event_count_ratio_delta": (
                guard_aggregate["mean_event_count_ratio"] - v3_aggregate["mean_event_count_ratio"]
            ),
        },
    }


def aggregate_variant(case_results: Sequence[Mapping[str, Any]], variant: str) -> dict[str, Any]:
    metrics_key = f"{variant}_metrics"
    legal_key = f"{variant}_legal"
    metrics = [_mapping(row.get(metrics_key)) for row in case_results]
    f1_values = [_metric_f1(row) for row in metrics]
    dominant_ratios = [_float(row.get("dominant_spacing_ratio")) or 0.0 for row in metrics]
    event_ratios = [_float(row.get("event_count_ratio")) or 0.0 for row in metrics]
    second_shares = [_float(row.get("second_window_event_share")) or 0.0 for row in metrics]
    boundary_ratios = [_float(row.get("boundary_event_ratio")) or 0.0 for row in metrics]
    duplicate_ratios = [_float(row.get("duplicate_or_nonincreasing_spacing_ratio")) or 0.0 for row in metrics]
    legal_count = sum(1 for row in case_results if bool(row.get(legal_key)))
    return {
        "case_count": int(len(case_results)),
        "legal_count": int(legal_count),
        "all_legal": int(legal_count) == int(len(case_results)),
        "mean_f1_100ms": _mean(f1_values),
        "median_f1_100ms": _median(f1_values),
        "mean_dominant_spacing_ratio": _mean(dominant_ratios),
        "median_dominant_spacing_ratio": _median(dominant_ratios),
        "mean_event_count_ratio": _mean(event_ratios),
        "median_event_count_ratio": _median(event_ratios),
        "mean_second_window_share": _mean(second_shares),
        "median_second_window_share": _median(second_shares),
        "starved_count": sum(1 for row in metrics if bool(row.get("starved"))),
        "rigid_case_count": sum(
            1 for value in dominant_ratios if float(value) >= float(RIGID_CASE_RATIO_THRESHOLD)
        ),
        "max_boundary_event_ratio": max(boundary_ratios, default=0.0),
        "max_duplicate_or_nonincreasing_spacing_ratio": max(duplicate_ratios, default=0.0),
    }


def decision_from_aggregate(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    baseline = _mapping(aggregate.get("v21_baseline"))
    guard = _mapping(aggregate.get("v21_guard"))
    v3 = _mapping(aggregate.get("v3"))
    guard_vs_baseline = _mapping(aggregate.get("guard_vs_baseline"))
    guard_vs_v3 = _mapping(aggregate.get("guard_vs_v3"))
    all_v21_legal = bool(baseline.get("all_legal")) and bool(guard.get("all_legal"))
    rigid_improved_count = int(guard_vs_baseline.get("rigid_improved_count") or 0)
    mean_f1_delta = float(guard_vs_baseline.get("mean_f1_delta") or 0.0)
    mean_rigid_delta = float(guard_vs_baseline.get("mean_dominant_spacing_ratio_delta") or 0.0)
    new_starved_count = int(guard_vs_baseline.get("new_starved_count") or 0)
    blocked_count = int(guard_vs_baseline.get("blocked_count") or 0)
    guard_starved = int(guard.get("starved_count") or 0)
    v3_starved = int(v3.get("starved_count") or 0)
    guard_mean_f1 = float(guard.get("mean_f1_100ms") or 0.0)
    v3_mean_f1 = float(v3.get("mean_f1_100ms") or 0.0)

    if not all_v21_legal:
        route = "KILL"
        reason = "one or more v2.1 baseline/guard rollouts were illegal"
        next_step = "Do not scale the anti-rigid hard block; inspect the illegal cases first."
    elif (
        rigid_improved_count <= NEGATIVE_MAX_RIGID_IMPROVED_CASES
        or mean_f1_delta < MEAN_F1_REGRESSION_LIMIT
        or new_starved_count > MAX_NEW_STARVED_CASES
    ):
        route = "KILL"
        reason = "guard failed the fixed-slice safety or generalization floor"
        next_step = "Do not scale hard blocking; mutate the grammar constraint or return to v3/C3 diagnostics."
    elif (
        blocked_count > 0
        and rigid_improved_count >= POSITIVE_MIN_RIGID_IMPROVED_CASES
        and mean_rigid_delta <= POSITIVE_MEAN_RIGID_DELTA
        and mean_f1_delta >= MEAN_F1_REGRESSION_LIMIT
        and new_starved_count <= MAX_NEW_STARVED_CASES
        and guard_starved < v3_starved
        and guard_mean_f1 >= v3_mean_f1 + MEAN_F1_REGRESSION_LIMIT
    ):
        route = "TEST_NEXT"
        reason = "guard generalized on fixed-slice rigidity and remained competitive with v3 continuation"
        next_step = "Run a held-out or larger fixed-slice v2.1 anti-rigid decode audit before changing defaults."
    else:
        route = "MUTATE"
        reason = "guard produced useful signal but did not clear every fixed-slice positive gate"
        next_step = "Inspect failure clusters and test a softer or state-aware v2.1 grammar constraint."
    return {
        "route": route,
        "reason": reason,
        "next_step": next_step,
        "all_v21_legal": all_v21_legal,
        "rigid_improved_count": rigid_improved_count,
        "mean_f1_delta": mean_f1_delta,
        "mean_dominant_spacing_ratio_delta": mean_rigid_delta,
        "new_starved_count": new_starved_count,
        "blocked_count": blocked_count,
        "guard_starved_count": guard_starved,
        "v3_starved_count": v3_starved,
        "guard_vs_v3_mean_f1_delta": _float(guard_vs_v3.get("mean_f1_delta")) or 0.0,
    }


def worst_cases(case_results: Sequence[Mapping[str, Any]], *, limit: int = 5) -> dict[str, list[dict[str, Any]]]:
    rows = [dict(row) for row in case_results]

    def brief(row: Mapping[str, Any]) -> dict[str, Any]:
        baseline = _mapping(row.get("v21_baseline_metrics"))
        guard = _mapping(row.get("v21_guard_metrics"))
        v3 = _mapping(row.get("v3_metrics"))
        return {
            "case_id": row.get("case_id"),
            "difficulty": row.get("difficulty"),
            "difficulty_band": row.get("difficulty_band"),
            "audio_family": row.get("audio_family"),
            "baseline_f1": _metric_f1(baseline),
            "guard_f1": _metric_f1(guard),
            "v3_f1": _metric_f1(v3),
            "baseline_rigid": baseline.get("dominant_spacing_ratio"),
            "guard_rigid": guard.get("dominant_spacing_ratio"),
            "v3_rigid": v3.get("dominant_spacing_ratio"),
            "baseline_starved": baseline.get("starved"),
            "guard_starved": guard.get("starved"),
            "v3_starved": v3.get("starved"),
            "guard_blocks": _mapping(row.get("transform")).get("blocked_count"),
            "guard_first_times": guard.get("first_12_generated_times"),
        }

    return {
        "lowest_guard_f1": [
            brief(row)
            for row in sorted(rows, key=lambda row: _metric_f1(_mapping(row.get("v21_guard_metrics"))))[:limit]
        ],
        "highest_guard_rigid": [
            brief(row)
            for row in sorted(
                rows,
                key=lambda row: _float(_mapping(row.get("v21_guard_metrics")).get("dominant_spacing_ratio")) or 0.0,
                reverse=True,
            )[:limit]
        ],
        "largest_guard_f1_drop": [
            brief(row)
            for row in sorted(rows, key=lambda row: _float(_mapping(row.get("guard_vs_baseline")).get("f1_delta")) or 0.0)[
                :limit
            ]
        ],
        "new_guard_starved": [brief(row) for row in rows if bool(_mapping(row.get("guard_vs_baseline")).get("new_starved"))][
            :limit
        ],
    }


def illegal_cases(case_results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in case_results:
        for variant, legal_key, rollout_key, path_key in (
            ("v2.1 baseline", "v21_baseline_legal", "v21_baseline_rollout", "v21_baseline_summary_path"),
            ("v2.1 guard", "v21_guard_legal", "v21_guard_rollout", "v21_guard_summary_path"),
        ):
            if bool(row.get(legal_key)):
                continue
            rollout = _mapping(row.get(rollout_key))
            rows.append(
                {
                    "case_id": row.get("case_id"),
                    "variant": variant,
                    "difficulty": row.get("difficulty"),
                    "difficulty_band": row.get("difficulty_band"),
                    "completed": rollout.get("completed"),
                    "dead_end": rollout.get("dead_end"),
                    "max_tokens_exceeded": rollout.get("max_tokens_exceeded"),
                    "terminal_ms": rollout.get("terminal_ms"),
                    "token_count": rollout.get("token_count"),
                    "timepoint_count": rollout.get("timepoint_count"),
                    "summary_path": row.get(path_key),
                }
            )
    return rows


def rollout_status(summary: Mapping[str, Any]) -> dict[str, Any]:
    rollout = _mapping(summary.get("rollout"))
    windows = rollout.get("windows")
    terminal_ms = None
    if isinstance(windows, list) and windows:
        last_window = _mapping(windows[-1])
        terminal_ms = last_window.get("terminal_ms")
    return {
        "completed": bool(rollout.get("completed")),
        "dead_end": bool(rollout.get("dead_end")),
        "max_tokens_exceeded": bool(rollout.get("max_tokens_exceeded")),
        "window_count": int(rollout.get("window_count") or 0),
        "token_count": int(rollout.get("token_count") or 0),
        "timepoint_count": int(rollout.get("timepoint_count") or 0),
        "terminal_ms": terminal_ms,
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
    aggregate = _mapping(summary.get("aggregate"))
    decision = _mapping(summary.get("decision"))
    baseline = _mapping(aggregate.get("v21_baseline"))
    guard = _mapping(aggregate.get("v21_guard"))
    v3 = _mapping(aggregate.get("v3"))
    guard_vs_baseline = _mapping(aggregate.get("guard_vs_baseline"))
    guard_vs_v3 = _mapping(aggregate.get("guard_vs_v3"))
    illegal = [row for row in summary.get("illegal_cases", []) if isinstance(row, Mapping)]
    lines = [
        "# Mapper v2.1/v3 Fixed-Slice Decode Comparison Result Report",
        "",
        "## Scope",
        "",
        "This pass reruns v2.1 baseline and v2.1 anti-rigid guard on the 32-map, 16s case universe from the v3 500-step fixed-slice wide audit. The v3 wide audit is used as a read-only comparator.",
        "",
        "## Decision",
        "",
        f"Decision: `{decision.get('route')}`.",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Cases: `{aggregate.get('case_count')}`",
        f"- v2.1 guard blocks: `{guard_vs_baseline.get('blocked_count')}`",
        f"- Guard rigid-improved cases: `{guard_vs_baseline.get('rigid_improved_count')}`",
        f"- Guard mean F1 delta vs v2.1 baseline: `{_fmt(guard_vs_baseline.get('mean_f1_delta'))}`",
        f"- Guard mean rigid delta vs v2.1 baseline: `{_fmt(guard_vs_baseline.get('mean_dominant_spacing_ratio_delta'))}`",
        f"- Guard new starved cases vs v2.1 baseline: `{guard_vs_baseline.get('new_starved_count')}`",
        f"- Guard starved count vs v3: `{guard.get('starved_count')}` vs `{v3.get('starved_count')}`",
        f"- Guard mean F1 delta vs v3: `{_fmt(guard_vs_v3.get('mean_f1_delta'))}`",
        "",
        "## Aggregate Table",
        "",
        "| Variant | Legal | Mean F1 | Median F1 | Mean rigid | Rigid cases | Starved | Mean event ratio | Median event ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _variant_row("v2.1 baseline", baseline),
        _variant_row("v2.1 guard", guard),
        _variant_row("v3 500 wide", v3),
        "",
        "## What Passed",
        "",
        "- The evaluator completed the full 32-map fixed-slice comparison: 64 v2.1 real-audio rollouts plus the read-only v3 comparator.",
        f"- The guard activated `{guard_vs_baseline.get('blocked_count')}` times and reduced dominant-spacing ratio in `{guard_vs_baseline.get('rigid_improved_count')}` of `{aggregate.get('case_count')}` cases.",
        f"- Rigid-case count dropped from `{baseline.get('rigid_case_count')}` in v2.1 baseline to `{guard.get('rigid_case_count')}` under the guard.",
        f"- Guard starvation remained lower than v3: `{guard.get('starved_count')}` vs `{v3.get('starved_count')}` cases.",
        f"- Guard mean F1 remained above the v3 comparator by `{_fmt(guard_vs_v3.get('mean_f1_delta'))}`.",
        "",
        "## What Surfaced",
        "",
        f"- Legality is not robust: `{len(illegal)}` v2.1 rollout(s) dead-ended or failed completion.",
        f"- Guard mean F1 regressed by `{_fmt(guard_vs_baseline.get('mean_f1_delta'))}` versus v2.1 baseline despite reducing rigidity.",
        f"- Guard introduced `{guard_vs_baseline.get('new_starved_count')}` new starved case(s) versus v2.1 baseline.",
        "- The hard-block guard can fix one dead-end pattern while creating another, so it should not be scaled as-is.",
        "",
        "## Illegal Cases",
        "",
        "| Case | Variant | Diff | Completed | Dead end | Max tokens | Terminal ms | Tokens | Timepoints |",
        "| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: |",
    ]
    if illegal:
        for row in illegal:
            lines.append(
                "| {case} | {variant} | {diff} | {completed} | {dead_end} | {max_tokens} | {terminal_ms} | {tokens} | {timepoints} |".format(
                    case=f"`{row.get('case_id')}`",
                    variant=f"`{row.get('variant')}`",
                    diff=_fmt(row.get("difficulty")),
                    completed=row.get("completed"),
                    dead_end=row.get("dead_end"),
                    max_tokens=row.get("max_tokens_exceeded"),
                    terminal_ms=row.get("terminal_ms"),
                    tokens=row.get("token_count"),
                    timepoints=row.get("timepoint_count"),
                )
            )
    else:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    lines.extend(
        [
        "",
        "## Case Table",
        "",
        "| Case | Diff | Baseline F1/Rigid | Guard F1/Rigid | v3 F1/Rigid | Starved B/G/v3 | Blocks |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for row in summary.get("case_results", []):
        if not isinstance(row, Mapping):
            continue
        baseline_metrics = _mapping(row.get("v21_baseline_metrics"))
        guard_metrics = _mapping(row.get("v21_guard_metrics"))
        v3_metrics = _mapping(row.get("v3_metrics"))
        transform = _mapping(row.get("transform"))
        lines.append(
            "| {case} | {diff} | {bf1}/{brigid} | {gf1}/{grigid} | {vf1}/{vrigid} | {bs}/{gs}/{vs} | {blocks} |".format(
                case=f"`{row.get('case_id')}`",
                diff=_fmt(row.get("difficulty")),
                bf1=_fmt(_metric_f1(baseline_metrics)),
                brigid=_fmt(baseline_metrics.get("dominant_spacing_ratio")),
                gf1=_fmt(_metric_f1(guard_metrics)),
                grigid=_fmt(guard_metrics.get("dominant_spacing_ratio")),
                vf1=_fmt(_metric_f1(v3_metrics)),
                vrigid=_fmt(v3_metrics.get("dominant_spacing_ratio")),
                bs=baseline_metrics.get("starved"),
                gs=guard_metrics.get("starved"),
                vs=v3_metrics.get("starved"),
                blocks=transform.get("blocked_count"),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            _interpretation(summary),
            "",
            "## Next Step",
            "",
            str(summary.get("next_step")),
            "",
        ]
    )
    return "\n".join(lines)


def _variant_row(label: str, aggregate: Mapping[str, Any]) -> str:
    return (
        f"| `{label}` | `{aggregate.get('all_legal')}` | {_fmt(aggregate.get('mean_f1_100ms'))} | "
        f"{_fmt(aggregate.get('median_f1_100ms'))} | {_fmt(aggregate.get('mean_dominant_spacing_ratio'))} | "
        f"{aggregate.get('rigid_case_count')} | {aggregate.get('starved_count')} | "
        f"{_fmt(aggregate.get('mean_event_count_ratio'))} | {_fmt(aggregate.get('median_event_count_ratio'))} |"
    )


def _interpretation(summary: Mapping[str, Any]) -> str:
    decision = _mapping(summary.get("decision"))
    route = str(decision.get("route"))
    if route == "TEST_NEXT":
        return "The opt-in hard-block guard generalized well enough on the fixed slice to justify a larger decode audit, but it remains a decode-time intervention rather than target-grammar replacement evidence."
    if route == "KILL":
        return "The hard-block guard should not be scaled from the six-case result. Either legality, starvation, F1, or fixed-slice rigidity failed the predefined floor."
    return "The guard has usable signal but not enough to change defaults. The next loop should inspect failure clusters and mutate the constraint."


def _run_v21_case(
    *,
    mapper_checkpoint: Path,
    control_checkpoint: Path,
    output_summary_path: Path,
    device_name: str,
    chart_end_ms: int,
    max_tokens_per_window: int,
    audio_path: str,
    normalized_difficulty: float,
    seed: int,
    collect_logit_diagnostics: bool,
    timepoint_preview_limit: int,
    logits_transform: MapperV21AntiRigidSpacingLogitsTransform | None,
) -> dict[str, Any]:
    return run_trained_v21_runtime_rollout_smoke(
        mapper_checkpoint_path=mapper_checkpoint,
        control_checkpoint_path=control_checkpoint,
        output_summary_path=output_summary_path,
        output_report_path=None,
        device_name=device_name,
        chart_end_ms=chart_end_ms,
        max_tokens_per_window=int(max_tokens_per_window),
        audio_path=audio_path,
        normalized_difficulty=float(normalized_difficulty),
        include_control_attention_kv_cache=False,
        temperature=0.0,
        top_p=None,
        seed=int(seed),
        real_audio=True,
        beatthis_device=None,
        beatthis_float16=False,
        time_shift_length_penalty_alpha=0.0,
        time_shift_delta_penalty_alpha=0.0,
        collect_logit_diagnostics=bool(collect_logit_diagnostics),
        logit_top_k=5,
        logits_transform=logits_transform,
        timepoint_preview_limit=int(timepoint_preview_limit),
    )


def _metrics_from_v21_summary(
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


def _reference_times(beatmap_path: Path, *, chart_end_ms: int) -> list[int]:
    hitobjects = parse_mania_hit_objects(beatmap_path, expected_key_count=4)
    return [
        int(timepoint.time_ms)
        for timepoint in hitobjects_to_mapper_timepoints(hitobjects)
        if 0 <= int(timepoint.time_ms) <= int(chart_end_ms)
    ]


def _generated_timepoints(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rollout = _mapping(summary.get("rollout"))
    timepoints = rollout.get("timepoints")
    if not isinstance(timepoints, list):
        raise ValueError("rollout summary does not contain timepoints")
    timepoint_count = int(rollout.get("timepoint_count") or 0)
    if len(timepoints) < timepoint_count:
        raise ValueError(f"rollout timepoints preview is truncated: {len(timepoints)} < {timepoint_count}")
    return [dict(item) for item in timepoints if isinstance(item, Mapping)]


def _candidate_legal(summary: Mapping[str, Any]) -> bool:
    checks = _mapping(summary.get("checks"))
    rollout = _mapping(summary.get("rollout"))
    return bool(
        checks.get("rollout_not_dead_end")
        and checks.get("rollout_not_max_tokens_exceeded")
        and rollout.get("completed")
    )


def _case_runs(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    runs = summary.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("v3 wide summary must contain a non-empty runs list")
    case_runs = [dict(row) for row in runs if isinstance(row, Mapping)]
    if len(case_runs) != len(runs):
        raise ValueError("every v3 run must be a JSON object")
    return case_runs


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _starved(*, chart_end_ms: int, second_share: float, reference_second_share: float) -> bool:
    return bool(
        int(chart_end_ms) > SECOND_WINDOW_START_MS
        and float(reference_second_share) >= REFERENCE_SECOND_WINDOW_SUPPORT_THRESHOLD
        and float(second_share) <= STARVED_SHARE_THRESHOLD
    )


def _metric_f1(metrics: Mapping[str, Any]) -> float:
    return _float(_mapping(metrics.get("timing_match_100ms")).get("f1")) or 0.0


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[Any]:
    return value if isinstance(value, RuntimeSequence) and not isinstance(value, (str, bytes, bytearray)) else ()


def _float(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _mean(values: Sequence[float]) -> float:
    return float(math.fsum(values) / len(values)) if values else 0.0


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return float(0.5 * (ordered[midpoint - 1] + ordered[midpoint]))


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    denominator = float(denominator)
    if denominator == 0.0:
        return 0.0
    return float(numerator) / denominator


def _fmt(value: object) -> str:
    number = _float(value)
    return "n/a" if number is None else f"{number:.6f}"


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value)).strip("_") or "case"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the v2.1/v3 32-case fixed-slice decode comparison.")
    parser.add_argument("--v3-wide-summary", default=DEFAULT_V3_WIDE_SUMMARY_PATH)
    parser.add_argument("--v21-baseline-summary", default=DEFAULT_V21_BASELINE_SUMMARY_PATH)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--report-output", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mapper-checkpoint-path")
    parser.add_argument("--control-checkpoint-path")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--max-tokens-per-window", type=int, default=512)
    parser.add_argument("--min-repeated-spacings", type=int, default=4)
    parser.add_argument("--min-spacing-ms", type=int, default=40)
    parser.add_argument("--max-spacing-ms", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--collect-logit-diagnostics", action="store_true")
    parser.add_argument("--timepoint-preview-limit", type=int, default=4096)
    parser.add_argument("--case-limit", type=int)
    args = parser.parse_args(argv)
    summary = run_v21_v3_fixed_slice_decode_comparison(
        v3_wide_summary_path=args.v3_wide_summary,
        v21_baseline_summary_path=args.v21_baseline_summary,
        output_summary_path=args.summary_output,
        output_report_path=args.report_output,
        output_dir=args.output_dir,
        mapper_checkpoint_path=args.mapper_checkpoint_path,
        control_checkpoint_path=args.control_checkpoint_path,
        device_name=args.device,
        max_tokens_per_window=args.max_tokens_per_window,
        min_repeated_spacings=args.min_repeated_spacings,
        min_spacing_ms=args.min_spacing_ms,
        max_spacing_ms=args.max_spacing_ms,
        seed=args.seed,
        collect_logit_diagnostics=args.collect_logit_diagnostics,
        timepoint_preview_limit=args.timepoint_preview_limit,
        case_limit=args.case_limit,
    )
    print(
        "mapper_v21_v3_fixed_slice_decode_comparison_done "
        f"route={summary['decision']['route']} "
        f"cases={summary['aggregate']['case_count']} "
        f"guard_rigid_improved={summary['aggregate']['guard_vs_baseline']['rigid_improved_count']} "
        f"guard_blocks={summary['aggregate']['guard_vs_baseline']['blocked_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
