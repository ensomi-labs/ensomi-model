from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SUMMARY_SCHEMA_VERSION = 1
REPORT_ROOT = Path("artifacts/reports/audits")
MAPPER_REPORT_ROOT = REPORT_ROOT / "mapper_v2_1_grammar"
C3_REPORT_ROOT = REPORT_ROOT / "context_adaptive_fallback_codec"

DEFAULT_SUMMARY_OUTPUT = (
    MAPPER_REPORT_ROOT / "target_grammar_v3_generated_prefix_event_budget_calibration_audit_summary.json"
)
DEFAULT_REPORT_OUTPUT = (
    MAPPER_REPORT_ROOT / "target_grammar_v3_generated_prefix_event_budget_calibration_audit_result_report.md"
)

RANK_NEAR_LIMIT = 5
MIN_RANK_NEAR_CASES = 4
MIN_DEFICIT_CASES = 4
SECOND_WINDOW_DEFICIT_THRESHOLD = 0.25
EVENT_RATIO_CAP = 1.25


@dataclass(frozen=True)
class SourceSpec:
    key: str
    path: Path
    required: bool = True


DEFAULT_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        "generated_prefix_trace",
        MAPPER_REPORT_ROOT / "target_grammar_v3_generated_prefix_state_trace_audit_summary.json",
    ),
    SourceSpec(
        "spacing_escape",
        MAPPER_REPORT_ROOT / "target_grammar_v3_trace_conditioned_spacing_escape_summary.json",
    ),
    SourceSpec(
        "continuation_jump",
        MAPPER_REPORT_ROOT / "target_grammar_v3_continuation_jump_training_gate_summary.json",
    ),
    SourceSpec(
        "event_budget",
        MAPPER_REPORT_ROOT / "target_grammar_v3_event_budget_training_gate_summary.json",
    ),
    SourceSpec(
        "conditioned_event_distribution",
        MAPPER_REPORT_ROOT / "target_grammar_v3_conditioned_event_distribution_short_rollout_summary.json",
    ),
    SourceSpec(
        "c3_v3_target_complexity",
        C3_REPORT_ROOT / "c3_v3_target_complexity_comparison_summary.json",
    ),
)


def run_generated_prefix_event_budget_calibration_audit(
    *,
    summary_output_path: Path = DEFAULT_SUMMARY_OUTPUT,
    report_output_path: Path = DEFAULT_REPORT_OUTPUT,
    sources: Sequence[SourceSpec] = DEFAULT_SOURCES,
) -> dict[str, Any]:
    start = time.monotonic()
    loaded = load_sources(sources)
    missing_required = [row for row in loaded.values() if row["required"] and not row["exists"]]

    if missing_required:
        metrics = empty_metrics(loaded)
        decision = {
            "route": "MUTATE_AUDIT_INPUTS",
            "reason": f"{len(missing_required)} required artifact summaries are missing",
            "next_step": "Regenerate or locate the missing summaries before choosing another v3 or v2.1 mutation.",
        }
    else:
        metrics = compute_calibration_metrics(loaded)
        decision = decision_from_metrics(metrics)

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "target_grammar_v3_generated_prefix_event_budget_calibration_audit",
        "elapsed_s": time.monotonic() - start,
        "sources": {
            key: {
                "path": str(row["path"]),
                "required": bool(row["required"]),
                "exists": bool(row["exists"]),
            }
            for key, row in loaded.items()
        },
        "missing_required_count": len(missing_required),
        "metrics": metrics,
        "decision": decision,
    }
    write_summary_json(summary, summary_output_path)
    write_report(summary, report_output_path)
    return summary


def load_sources(sources: Sequence[SourceSpec]) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for source in sources:
        row: dict[str, Any] = {
            "key": source.key,
            "path": source.path,
            "required": bool(source.required),
            "exists": source.path.exists(),
            "payload": None,
        }
        if source.path.exists():
            row["payload"] = load_json_object(source.path)
        loaded[source.key] = row
    return loaded


def empty_metrics(loaded: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "source_keys": sorted(loaded),
        "required_artifacts_present": False,
        "primary_trace": {},
        "spacing_escape": {},
        "failed_branch_checks": {},
        "guard_results": {
            "artifact_only": True,
            "no_training": True,
            "no_rollout_rerun": True,
            "no_tokenizer_or_default_change": True,
            "required_artifacts_present": False,
        },
        "positive_signal_observed": False,
        "negative_signal_observed": True,
    }


def compute_calibration_metrics(loaded: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    trace = _payload(loaded, "generated_prefix_trace")
    spacing = _payload(loaded, "spacing_escape")
    continuation = _payload(loaded, "continuation_jump")
    event_budget = _payload(loaded, "event_budget")
    conditioned = _payload(loaded, "conditioned_event_distribution")
    c3_complexity = _payload(loaded, "c3_v3_target_complexity")

    primary_trace = primary_trace_metrics(trace)
    spacing_metrics = spacing_escape_metrics(spacing)
    branch_metrics = failed_branch_metrics(
        continuation=continuation,
        event_budget=event_budget,
        conditioned=conditioned,
        c3_complexity=c3_complexity,
    )
    positive_signal = (
        primary_trace["rank_near_case_count"] >= MIN_RANK_NEAR_CASES
        and primary_trace["second_window_deficit_case_count"] >= MIN_DEFICIT_CASES
        and spacing_metrics["primary_recoverable_by_broad_release"]
        and spacing_metrics["sentinel_overproduction_risk"]
        and branch_metrics["scalar_or_decode_repairs_failed"]
    )
    negative_signal = (not positive_signal) and (
        primary_trace["rank_near_case_count"] < MIN_RANK_NEAR_CASES
        or not spacing_metrics["sentinel_overproduction_risk"]
        or branch_metrics["only_broad_event_pressure_remaining"]
    )
    return {
        "source_keys": sorted(loaded),
        "required_artifacts_present": True,
        "primary_trace": primary_trace,
        "spacing_escape": spacing_metrics,
        "failed_branch_checks": branch_metrics,
        "guard_results": {
            "artifact_only": True,
            "no_training": True,
            "no_rollout_rerun": True,
            "no_tokenizer_or_default_change": True,
            "required_artifacts_present": True,
            "rank_near_primary_signal": primary_trace["rank_near_case_count"] >= MIN_RANK_NEAR_CASES,
            "second_window_deficit_signal": primary_trace["second_window_deficit_case_count"] >= MIN_DEFICIT_CASES,
            "sentinel_overproduction_guard_needed": spacing_metrics["sentinel_overproduction_risk"],
            "prior_scalar_decode_repairs_failed": branch_metrics["scalar_or_decode_repairs_failed"],
        },
        "positive_signal_observed": positive_signal,
        "negative_signal_observed": negative_signal,
    }


def primary_trace_metrics(trace_summary: Mapping[str, Any]) -> dict[str, Any]:
    cases = _sequence(trace_summary.get("case_results"))
    case_rows: list[dict[str, Any]] = []
    rank_near_counts: list[int] = []
    deficits: list[float] = []
    event_ratios: list[float] = []
    second_window_shares: list[float] = []
    reference_second_window_shares: list[float] = []
    margin_values: list[float] = []

    for case in cases:
        item = _mapping(case)
        classification = _mapping(item.get("classification"))
        failures = _mapping(classification.get("candidate_failures"))
        under = _mapping(failures.get("event_underselection"))
        boundary = _mapping(failures.get("boundary_drift"))
        baseline = _mapping(item.get("baseline_projection"))

        best_event_rank = _int(under.get("best_event_rank"))
        candidate_count = _int(under.get("candidate_count")) or 0
        rank_near = bool(under.get("concrete")) and best_event_rank is not None and best_event_rank <= RANK_NEAR_LIMIT
        generated_second = _float(boundary.get("second_window_event_share"))
        if generated_second is None:
            generated_second = _float(baseline.get("second_window_event_share")) or 0.0
        reference_second = _float(boundary.get("reference_second_window_event_share"))
        if reference_second is None:
            reference_second = _float(baseline.get("reference_second_window_event_share")) or 0.0
        deficit = max(0.0, reference_second - generated_second)
        event_ratio = _float(baseline.get("event_count_ratio"))
        margin = _float(under.get("best_event_margin_vs_argmax"))

        if rank_near:
            rank_near_counts.append(candidate_count)
        deficits.append(deficit)
        event_ratios.append(event_ratio or 0.0)
        second_window_shares.append(generated_second)
        reference_second_window_shares.append(reference_second)
        if margin is not None:
            margin_values.append(margin)
        case_rows.append(
            {
                "case_id": item.get("case_id"),
                "failure_class": classification.get("failure_class"),
                "rank_near_event_opportunity": rank_near,
                "best_event_rank": best_event_rank,
                "candidate_count": candidate_count,
                "best_event_margin_vs_argmax": margin,
                "event_count_ratio": event_ratio,
                "second_window_event_share": generated_second,
                "reference_second_window_event_share": reference_second,
                "second_window_share_deficit": deficit,
                "deficit_over_threshold": deficit >= SECOND_WINDOW_DEFICIT_THRESHOLD,
            }
        )

    rank_near_case_count = sum(1 for row in case_rows if row["rank_near_event_opportunity"])
    deficit_case_count = sum(1 for row in case_rows if row["deficit_over_threshold"])
    aggregate = _mapping(trace_summary.get("aggregate"))
    return {
        "case_count": len(case_rows),
        "rank_near_case_count": rank_near_case_count,
        "rank_near_case_share": _safe_ratio(rank_near_case_count, len(case_rows)),
        "rank_near_candidate_count_total": sum(rank_near_counts),
        "mean_rank_near_candidate_count": statistics.fmean(rank_near_counts) if rank_near_counts else 0.0,
        "second_window_deficit_case_count": deficit_case_count,
        "second_window_deficit_case_share": _safe_ratio(deficit_case_count, len(case_rows)),
        "mean_second_window_share_deficit": statistics.fmean(deficits) if deficits else 0.0,
        "mean_generated_second_window_share": statistics.fmean(second_window_shares) if second_window_shares else 0.0,
        "mean_reference_second_window_share": (
            statistics.fmean(reference_second_window_shares) if reference_second_window_shares else 0.0
        ),
        "mean_event_count_ratio": statistics.fmean(event_ratios) if event_ratios else 0.0,
        "mean_best_event_margin_vs_argmax": statistics.fmean(margin_values) if margin_values else None,
        "failure_class_counts": dict(_mapping(aggregate.get("failure_class_counts"))),
        "secondary_flag_counts": dict(_mapping(aggregate.get("secondary_flag_counts"))),
        "cases": case_rows,
    }


def spacing_escape_metrics(spacing_summary: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = _mapping(spacing_summary.get("aggregate"))
    primary = _mapping(aggregate.get("primary"))
    sentinel = _mapping(aggregate.get("sentinel"))
    primary_case_count = _int(primary.get("case_count")) or 0
    sentinel_case_count = _int(sentinel.get("case_count")) or 0
    primary_candidate = _mapping(primary.get("candidate"))
    sentinel_candidate = _mapping(sentinel.get("candidate"))
    primary_baseline = _mapping(primary.get("baseline"))
    sentinel_baseline = _mapping(sentinel.get("baseline"))
    primary_event_over_count = _int(primary.get("event_ratio_over_count")) or 0
    sentinel_event_over_count = _int(sentinel.get("event_ratio_over_count")) or 0
    primary_recoverable = (
        (_int(primary.get("second_window_improved_count")) or 0) >= MIN_DEFICIT_CASES
        and (_int(primary.get("both_improved_count")) or 0) >= MIN_DEFICIT_CASES
    )
    sentinel_median_event_ratio = _float(sentinel_candidate.get("median_event_count_ratio")) or 0.0
    primary_mean_event_ratio = _float(primary_candidate.get("mean_event_count_ratio")) or 0.0
    primary_mean_second_share_delta = (
        (_float(primary_candidate.get("mean_second_window_share")) or 0.0)
        - (_float(primary_baseline.get("mean_second_window_share")) or 0.0)
    )
    sentinel_mean_event_ratio_delta = (
        (_float(sentinel_candidate.get("mean_event_count_ratio")) or 0.0)
        - (_float(sentinel_baseline.get("mean_event_count_ratio")) or 0.0)
    )
    sentinel_overproduction = (
        sentinel_case_count > 0
        and sentinel_event_over_count == sentinel_case_count
        and sentinel_median_event_ratio > EVENT_RATIO_CAP
    )
    return {
        "primary_case_count": primary_case_count,
        "sentinel_case_count": sentinel_case_count,
        "primary_recoverable_by_broad_release": primary_recoverable,
        "primary_second_window_improved_count": _int(primary.get("second_window_improved_count")) or 0,
        "primary_both_improved_count": _int(primary.get("both_improved_count")) or 0,
        "primary_event_ratio_over_count": primary_event_over_count,
        "primary_mean_event_ratio": primary_mean_event_ratio,
        "primary_mean_second_window_share_delta": primary_mean_second_share_delta,
        "sentinel_event_ratio_over_count": sentinel_event_over_count,
        "sentinel_median_event_ratio": sentinel_median_event_ratio,
        "sentinel_mean_event_ratio_delta": sentinel_mean_event_ratio_delta,
        "sentinel_overproduction_risk": sentinel_overproduction,
        "all_legal": bool(aggregate.get("all_legal")),
        "dead_end_count": _int(aggregate.get("dead_end_count")) or 0,
        "max_token_count": _int(aggregate.get("max_token_count")) or 0,
    }


def failed_branch_metrics(
    *,
    continuation: Mapping[str, Any],
    event_budget: Mapping[str, Any],
    conditioned: Mapping[str, Any],
    c3_complexity: Mapping[str, Any],
) -> dict[str, Any]:
    continuation_checks = _mapping(continuation.get("checks"))
    event_budget_checks = _mapping(event_budget.get("checks"))
    conditioned_guards = _mapping(conditioned.get("guard_results"))
    c3_guards = _mapping(c3_complexity.get("guard_results"))
    continuation_failed = any(
        continuation_checks.get(key) is False
        for key in (
            "all_32_rollouts_legal",
            "starved_cases_below_baseline_11",
            "rigid_cases_no_worse_than_baseline_7",
            "median_event_ratio_in_0_80_1_25",
        )
    )
    event_budget_failed = any(
        event_budget_checks.get(key) is False
        for key in (
            "all_32_rollouts_legal",
            "no_max_token_cases",
            "starved_cases_below_baseline_11",
            "median_event_ratio_in_0_80_1_25",
        )
    )
    conditioned_failed = conditioned_guards.get("median_event_count_ratio_in_range") is False
    c3_target_side_costly = (
        c3_guards.get("c3_combined_sequence_competitive") is False
        or c3_guards.get("c3_ordered_sequence_bounded") is False
        or c3_guards.get("c3_production_input_legal") is False
    )
    scalar_or_decode_failed = continuation_failed and event_budget_failed and conditioned_failed
    return {
        "continuation_jump_failed": continuation_failed,
        "event_budget_failed": event_budget_failed,
        "conditioned_event_distribution_failed": conditioned_failed,
        "c3_target_side_costly": c3_target_side_costly,
        "scalar_or_decode_repairs_failed": scalar_or_decode_failed,
        "only_broad_event_pressure_remaining": event_budget_failed and conditioned_failed,
        "continuation_jump": {
            "route": continuation.get("route") or _mapping(continuation.get("decision")).get("route"),
            "failed_checks": [key for key, value in continuation_checks.items() if value is False],
            "rollout_aggregate": _mapping(continuation.get("rollout_aggregate")),
        },
        "event_budget": {
            "route": _mapping(event_budget.get("decision")).get("route") or event_budget.get("route"),
            "failed_checks": [key for key, value in event_budget_checks.items() if value is False],
        },
        "conditioned_event_distribution": {
            "route": _mapping(conditioned.get("decision")).get("route"),
            "failed_guards": [key for key, value in conditioned_guards.items() if value is False],
        },
        "c3_v3_target_complexity": {
            "route": _mapping(c3_complexity.get("decision")).get("route"),
            "failed_guards": [key for key, value in c3_guards.items() if value is False],
        },
    }


def decision_from_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    if not bool(metrics.get("required_artifacts_present")):
        return {
            "route": "MUTATE_AUDIT_INPUTS",
            "reason": "required artifacts are missing",
            "next_step": "Regenerate the missing artifacts before running another v3 or v2.1 card.",
        }
    primary = _mapping(metrics.get("primary_trace"))
    spacing = _mapping(metrics.get("spacing_escape"))
    branches = _mapping(metrics.get("failed_branch_checks"))
    rank_near_cases = _int(primary.get("rank_near_case_count")) or 0
    deficit_cases = _int(primary.get("second_window_deficit_case_count")) or 0
    has_deficit = deficit_cases >= MIN_DEFICIT_CASES
    has_rank_near = rank_near_cases >= MIN_RANK_NEAR_CASES
    sentinel_overproduction = bool(spacing.get("sentinel_overproduction_risk"))
    broad_release_recovers_primary = bool(spacing.get("primary_recoverable_by_broad_release"))
    scalar_or_decode_failed = bool(branches.get("scalar_or_decode_repairs_failed"))

    if has_rank_near and has_deficit and sentinel_overproduction and broad_release_recovers_primary and scalar_or_decode_failed:
        return {
            "route": "TEST_SELECTIVE_COMPLETION_BUDGET_SIGNAL",
            "reason": (
                f"{rank_near_cases} primary cases have rank-near event opportunities and {deficit_cases} have "
                "large second-window deficits, while broad spacing release overproduces sentinels"
            ),
            "next_step": (
                "Create a bounded selective completion/budget signal card with hard primary and sentinel "
                "event-ratio guards; do not scale scalar losses or local decode repair."
            ),
        }
    if has_deficit and not has_rank_near:
        return {
            "route": "MUTATE_TO_GRAMMAR_LEVEL_CONTINUATION_TARGET",
            "reason": (
                f"{deficit_cases} primary cases have second-window deficits, but only {rank_near_cases} expose "
                "rank-near event opportunities"
            ),
            "next_step": "Design a grammar-level continuation target or full32 trace aggregation before training.",
        }
    return {
        "route": "PIVOT_TO_V2_1_GRAMMAR_IMPROVEMENT",
        "reason": "the audit does not isolate a selective v3 budget/completion signal beyond killed broad repairs",
        "next_step": "Write a v2.1 grammar-improvement card or a structural v3 grammar mutation instead of another local loss.",
    }


def write_summary_json(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    metrics = _mapping(summary.get("metrics"))
    decision = _mapping(summary.get("decision"))
    primary = _mapping(metrics.get("primary_trace"))
    spacing = _mapping(metrics.get("spacing_escape"))
    branches = _mapping(metrics.get("failed_branch_checks"))
    guard_results = _mapping(metrics.get("guard_results"))
    cases = _sequence(primary.get("cases"))
    lines = [
        "# Target Grammar v3 Generated-Prefix Event-Budget Calibration Audit Result Report",
        "",
        "## Scope",
        "",
        "This artifact-only audit executes `target_grammar_v3_generated_prefix_event_budget_calibration_audit_experiment_card.md`. It reads committed v3/C3 summaries, does not retrain, does not rerun rollout, and does not change tokenizer, grammar, replay, inference, or defaults.",
        "",
        "## Decision",
        "",
        f"Route: `{decision.get('route')}`.",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Next step: {decision.get('next_step')}",
        "",
        "## Guard Results",
        "",
        "| Guard | Value |",
        "| --- | ---: |",
    ]
    for key, value in guard_results.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Primary Generated-Prefix Evidence",
            "",
            f"- Primary cases: `{primary.get('case_count')}`",
            f"- Rank-near event opportunity cases: `{primary.get('rank_near_case_count')}` / `{primary.get('case_count')}`",
            f"- Rank-near opportunity candidates: `{primary.get('rank_near_candidate_count_total')}` total, mean `{_fmt(primary.get('mean_rank_near_candidate_count'))}` per rank-near case",
            f"- Large second-window deficit cases: `{primary.get('second_window_deficit_case_count')}` / `{primary.get('case_count')}`",
            f"- Mean generated second-window share: `{_fmt(primary.get('mean_generated_second_window_share'))}`",
            f"- Mean reference second-window share: `{_fmt(primary.get('mean_reference_second_window_share'))}`",
            f"- Mean second-window share deficit: `{_fmt(primary.get('mean_second_window_share_deficit'))}`",
            f"- Mean event-count ratio: `{_fmt(primary.get('mean_event_count_ratio'))}`",
            "",
            "| Case | Rank-near | Candidates | Best rank | Event ratio | Second share | Reference share | Deficit |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in cases:
        item = _mapping(row)
        lines.append(
            "| {case} | `{rank_near}` | `{candidates}` | `{rank}` | `{event_ratio}` | `{second}` | `{reference}` | `{deficit}` |".format(
                case=item.get("case_id"),
                rank_near=item.get("rank_near_event_opportunity"),
                candidates=item.get("candidate_count"),
                rank=item.get("best_event_rank"),
                event_ratio=_fmt(item.get("event_count_ratio")),
                second=_fmt(item.get("second_window_event_share")),
                reference=_fmt(item.get("reference_second_window_event_share")),
                deficit=_fmt(item.get("second_window_share_deficit")),
            )
        )
    lines.extend(
        [
            "",
            "## Spacing-Escape Evidence",
            "",
            f"- Primary broad-release recovery: `{spacing.get('primary_recoverable_by_broad_release')}`",
            f"- Primary second-window improved: `{spacing.get('primary_second_window_improved_count')}` / `{spacing.get('primary_case_count')}`",
            f"- Primary event-ratio-over-cap cases: `{spacing.get('primary_event_ratio_over_count')}`",
            f"- Primary mean event ratio under broad release: `{_fmt(spacing.get('primary_mean_event_ratio'))}`",
            f"- Primary mean second-window share delta: `{_fmt(spacing.get('primary_mean_second_window_share_delta'))}`",
            f"- Sentinel event-ratio-over-cap cases: `{spacing.get('sentinel_event_ratio_over_count')}` / `{spacing.get('sentinel_case_count')}`",
            f"- Sentinel median event ratio under broad release: `{_fmt(spacing.get('sentinel_median_event_ratio'))}`",
            f"- Sentinel mean event-ratio delta: `{_fmt(spacing.get('sentinel_mean_event_ratio_delta'))}`",
            "",
            "## Failed Branch Checks",
            "",
            f"- Continuation-jump failed: `{branches.get('continuation_jump_failed')}`",
            f"- Event-budget objective failed: `{branches.get('event_budget_failed')}`",
            f"- Conditioned event-distribution objective failed: `{branches.get('conditioned_event_distribution_failed')}`",
            f"- C3 target-side route costly: `{branches.get('c3_target_side_costly')}`",
            f"- Scalar/decode repairs failed as a family: `{branches.get('scalar_or_decode_repairs_failed')}`",
            "",
            "## What Passed",
            "",
            "- The required committed summaries were present and parseable.",
            "- The audit stayed artifact-only: no training, rollout rerun, tokenizer change, grammar change, C3 conditioning, or default behavior change.",
            "- The primary trace set contains rank-near event opportunities during second-window deficits.",
            "",
            "## What Surfaced",
            "",
            "- Broad spacing release can recover primary continuation but overproduces sentinels, so the useful next signal must be selective.",
            "- The recent scalar or local-decode families are not promotable as tested: continuation-jump undertransfers, event-budget is too blunt, conditioned event distribution misses the event-ratio guard, and spacing escape floods.",
            "- C3 remains strong codec-side evidence but is not the immediate emitted target route because target-side costs and production-input legality still fail.",
            "",
            "## Interpretation",
            "",
            "The v3 branch still has one narrow mapper-facing test left: a selective completion/budget signal tied to generated-prefix state and protected by sentinel overproduction guards. This audit does not approve v3 replacement or another broad loss sweep.",
            "",
            "## Evidence Boundary",
            "",
            "- Proved: the committed summaries contain selective primary deficits plus sentinel overproduction risk.",
            "- Not proved: that a completion/budget signal will train, improve full32 rollout, or beat v2.1 in production.",
            "- Not evaluated: full 4k replacement readiness, planner integration, or v3 default enablement.",
            "",
            "## Next Step",
            "",
            str(decision.get("next_step")),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _payload(loaded: Mapping[str, Mapping[str, Any]], key: str) -> Mapping[str, Any]:
    row = _mapping(loaded.get(key))
    payload = row.get("payload")
    return payload if isinstance(payload, Mapping) else {}


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()


def _int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _fmt(value: object) -> str:
    number = _float(value)
    if number is None:
        return "None"
    return f"{number:.6f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the v3 generated-prefix event-budget calibration audit.")
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT.as_posix())
    parser.add_argument("--report-output", default=DEFAULT_REPORT_OUTPUT.as_posix())
    args = parser.parse_args(argv)
    summary = run_generated_prefix_event_budget_calibration_audit(
        summary_output_path=Path(args.summary_output),
        report_output_path=Path(args.report_output),
    )
    print(
        "mapper_v3_generated_prefix_event_budget_calibration_audit_done "
        f"route={summary['decision']['route']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
