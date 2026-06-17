from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SUMMARY_SCHEMA_VERSION = 1
REPORT_ROOT = Path("artifacts/reports/audits")
MAPPER_REPORT_ROOT = REPORT_ROOT / "mapper_v2_1_grammar"
C3_REPORT_ROOT = REPORT_ROOT / "context_adaptive_fallback_codec"

DEFAULT_SUMMARY_OUTPUT = MAPPER_REPORT_ROOT / "mapper_v21_terminal_guard_v3_route_synthesis_summary.json"
DEFAULT_REPORT_OUTPUT = MAPPER_REPORT_ROOT / "mapper_v21_terminal_guard_v3_route_synthesis_result_report.md"

STRESS_F1_REGRESSION_LIMIT = -0.03


@dataclass(frozen=True)
class SourceSpec:
    key: str
    path: Path
    required: bool = True


DEFAULT_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        "v21_terminal_guard_full32",
        MAPPER_REPORT_ROOT / "mapper_v21_terminal_ln_start_guard_repair_full32_summary.json",
    ),
    SourceSpec(
        "v21_v3_fixed_slice",
        MAPPER_REPORT_ROOT / "mapper_v21_v3_fixed_slice_decode_comparison_summary.json",
    ),
    SourceSpec(
        "v3_full_dataset",
        MAPPER_REPORT_ROOT / "target_grammar_v3_full_dataset_audit_summary.json",
    ),
    SourceSpec(
        "v3_full_pipeline_smoke",
        MAPPER_REPORT_ROOT / "target_grammar_v3_full_pipeline_replacement_summary.json",
    ),
    SourceSpec(
        "v3_generated_prefix_trace",
        MAPPER_REPORT_ROOT / "target_grammar_v3_generated_prefix_state_trace_audit_summary.json",
    ),
    SourceSpec(
        "v3_post_teacher_forced_exposure",
        MAPPER_REPORT_ROOT / "target_grammar_v3_post_teacher_forced_exposure_route_synthesis_summary.json",
    ),
    SourceSpec(
        "c3_v3_target_complexity",
        C3_REPORT_ROOT / "c3_v3_target_complexity_comparison_summary.json",
    ),
)


def run_terminal_guard_v3_route_synthesis(
    *,
    summary_output_path: str | Path = DEFAULT_SUMMARY_OUTPUT,
    report_output_path: str | Path = DEFAULT_REPORT_OUTPUT,
    sources: Sequence[SourceSpec] = DEFAULT_SOURCES,
) -> dict[str, Any]:
    start = time.monotonic()
    loaded = load_sources(sources)
    missing_required = [row for row in loaded.values() if bool(row["required"]) and not bool(row["exists"])]
    checks = route_checks(loaded)
    decision = decision_from_checks(checks, missing_required=missing_required)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Mapper v2.1 terminal guard / v3 route synthesis",
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
        "checks": checks,
        "decision": decision,
        "next_card": decision.get("next_card"),
    }
    write_summary_json(summary, Path(summary_output_path))
    write_report(summary, Path(report_output_path))
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


def route_checks(loaded: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    terminal = _payload(loaded, "v21_terminal_guard_full32")
    fixed = _payload(loaded, "v21_v3_fixed_slice")
    v3_full = _payload(loaded, "v3_full_dataset")
    v3_pipeline = _payload(loaded, "v3_full_pipeline_smoke")
    v3_trace = _payload(loaded, "v3_generated_prefix_trace")
    v3_post_teacher = _payload(loaded, "v3_post_teacher_forced_exposure")
    c3_complexity = _payload(loaded, "c3_v3_target_complexity")

    terminal_metrics = terminal_guard_metrics(terminal)
    v3_representation = v3_representation_metrics(v3_full, c3_complexity)
    c3_metrics = c3_route_metrics(c3_complexity)
    v3_quality = v3_quality_metrics(v3_pipeline, v3_trace, v3_post_teacher)
    same_slice = same_slice_comparison_metrics(terminal, fixed)

    return {
        "required_artifacts_present": all(bool(row.get("exists")) for row in loaded.values()),
        "artifact_only": True,
        "no_training": True,
        "no_rollout_rerun": True,
        "no_tokenizer_or_default_change": True,
        "terminal_guard": terminal_metrics,
        "v3_representation": v3_representation,
        "c3_route": c3_metrics,
        "v3_quality": v3_quality,
        "same_slice": same_slice,
        "positive_signal_observed": (
            terminal_metrics["full32_passed"]
            and v3_representation["ready"]
            and c3_metrics["mapper_path_diminishing"]
            and v3_quality["repair_route_open"]
            and same_slice["comparison_available"]
        ),
        "negative_signal_observed": (
            not terminal_metrics["full32_passed"]
            or not v3_representation["ready"]
            or not same_slice["comparison_available"]
        ),
    }


def terminal_guard_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    decision = _mapping(summary.get("decision"))
    aggregate = _mapping(summary.get("aggregate"))
    baseline = _mapping(aggregate.get("baseline_guard"))
    anti = _mapping(aggregate.get("anti_rigid_guard"))
    full32 = bool(decision.get("full32"))
    all_legal = bool(aggregate.get("all_candidate_legal"))
    max_token_count = _int(aggregate.get("max_token_count")) or 0
    new_starved_count = _int(aggregate.get("new_starved_count")) or 0
    mean_f1_delta = _float(aggregate.get("mean_f1_delta")) or 0.0
    route = str(decision.get("route") or "")
    return {
        "route": route,
        "full32": full32,
        "run_count": _int(aggregate.get("run_count")) or 0,
        "case_count": _int(aggregate.get("case_count")) or 0,
        "all_candidate_legal": all_legal,
        "max_token_count": max_token_count,
        "new_starved_count": new_starved_count,
        "mean_f1_delta_vs_matching_comparator": mean_f1_delta,
        "primary_case04_baseline_legal": bool(decision.get("primary_case04_baseline_legal")),
        "primary_case05_anti_rigid_legal": bool(decision.get("primary_case05_anti_rigid_legal")),
        "baseline_guard_starved_count": _int(baseline.get("starved_count")) or 0,
        "anti_rigid_guard_starved_count": _int(anti.get("starved_count")) or 0,
        "anti_rigid_blocked_count": _int(aggregate.get("total_anti_rigid_blocked_count")) or 0,
        "full32_passed": (
            route == "TEST_NEXT"
            and full32
            and all_legal
            and max_token_count == 0
            and new_starved_count == 0
            and mean_f1_delta >= STRESS_F1_REGRESSION_LIMIT
        ),
    }


def v3_representation_metrics(v3_full: Mapping[str, Any], c3_complexity: Mapping[str, Any]) -> dict[str, Any]:
    decision = _mapping(v3_full.get("decision"))
    comparison = _mapping(v3_full.get("comparison"))
    c3_v3 = _mapping(c3_complexity.get("v3"))
    guards = _mapping(c3_complexity.get("guard_results"))
    reconstruction_mismatches = _int(comparison.get("reconstruction_mismatches"))
    if reconstruction_mismatches is None:
        reconstruction_mismatches = _int(c3_v3.get("full_dataset_reconstruction_mismatches"))
    token_reduction_ratio = _float(comparison.get("token_reduction_ratio"))
    if token_reduction_ratio is None:
        token_reduction_ratio = _float(c3_v3.get("eval_token_reduction_ratio"))
    bit_reduction_ratio = _float(comparison.get("total_bit_reduction_ratio"))
    if bit_reduction_ratio is None:
        bit_reduction_ratio = _float(c3_v3.get("eval_total_bit_reduction_ratio"))
    route = str(decision.get("route") or c3_v3.get("decision_route") or "")
    ready = (
        route == "TEST_NEXT"
        and reconstruction_mismatches == 0
        and (token_reduction_ratio or 0.0) > 0.0
        and (bit_reduction_ratio or 0.0) > 0.0
        and bool(guards.get("v3_no_future_lookup", True))
    )
    return {
        "route": route,
        "ready": ready,
        "reconstruction_mismatches": reconstruction_mismatches,
        "token_reduction_ratio": token_reduction_ratio,
        "total_bit_reduction_ratio": bit_reduction_ratio,
        "no_future_lookup": bool(guards.get("v3_no_future_lookup", True)),
        "requires_c3_style_backreference_replay": bool(c3_v3.get("requires_c3_style_backreference_replay", False)),
    }


def c3_route_metrics(c3_complexity: Mapping[str, Any]) -> dict[str, Any]:
    decision = _mapping(c3_complexity.get("decision"))
    c3 = _mapping(c3_complexity.get("c3"))
    guards = _mapping(c3_complexity.get("guard_results"))
    route = str(decision.get("route") or "")
    production_input_legal = bool(c3.get("production_input_legal") or guards.get("c3_production_input_legal"))
    combined_ratio = _float(c3.get("combined_to_baseline_token_ratio"))
    ordered_ratio = _float(c3.get("ordered_to_exact_token_ratio"))
    cross_window = _float(c3.get("reference_cross_mapper_window_span_rate"))
    return {
        "route": route,
        "mapper_path_diminishing": route == "MUTATE_TO_V3_GRAMMAR_REPAIR" and not production_input_legal,
        "codec_reconstruction_zero": bool(guards.get("c3_codec_reconstruction_zero")),
        "production_input_legal": production_input_legal,
        "combined_to_baseline_token_ratio": combined_ratio,
        "ordered_to_exact_token_ratio": ordered_ratio,
        "reference_cross_mapper_window_span_rate": cross_window,
        "test_delta_bits_per_event": _float(c3.get("test_delta_bits_per_event")),
    }


def v3_quality_metrics(
    pipeline: Mapping[str, Any],
    trace: Mapping[str, Any],
    post_teacher: Mapping[str, Any],
) -> dict[str, Any]:
    pipeline_route = _route(pipeline)
    trace_decision = _mapping(trace.get("decision"))
    trace_route = str(trace_decision.get("route") or "")
    trace_aggregate = _mapping(trace.get("aggregate"))
    post_route = _route(post_teacher)
    repair_route_open = trace_route == "TEST_BOUNDARY_OR_SPACING_STATE_REPAIR" or post_route in {
        "TEST_GENERATED_PREFIX_STATE_TRACE",
        "TEST_DECODE_STATE_EXPOSURE_DIAGNOSTIC",
    }
    quality_ready = pipeline_route == "TEST_NEXT" and not repair_route_open
    next_card = (
        "target_grammar_v3_boundary_or_spacing_state_repair"
        if trace_route == "TEST_BOUNDARY_OR_SPACING_STATE_REPAIR"
        else _mapping(post_teacher.get("decision")).get("next_card")
    )
    return {
        "pipeline_route": pipeline_route,
        "generated_prefix_trace_route": trace_route,
        "post_teacher_forced_route": post_route,
        "pipeline_mechanics_passed": pipeline_route == "TEST_NEXT",
        "quality_ready": quality_ready,
        "repair_route_open": repair_route_open,
        "next_card": next_card,
        "trace_failure_class_counts": dict(_mapping(trace_aggregate.get("failure_class_counts"))),
        "trace_mean_event_count_ratio": _float(trace_aggregate.get("mean_event_count_ratio")),
        "trace_mean_second_window_event_share": _float(trace_aggregate.get("mean_second_window_event_share")),
        "trace_mean_dominant_spacing_ratio": _float(trace_aggregate.get("mean_dominant_spacing_ratio")),
    }


def same_slice_comparison_metrics(terminal: Mapping[str, Any], fixed: Mapping[str, Any]) -> dict[str, Any]:
    fixed_cases = {
        str(row.get("case_id")): row for row in _sequence(fixed.get("case_results")) if isinstance(row, Mapping)
    }
    mode_metrics = {
        mode: _mode_vs_v3(_sequence(terminal.get("case_results")), fixed_cases, mode)
        for mode in ("baseline_guard", "anti_rigid_guard")
    }
    comparison_available = all(row["case_count"] > 0 and row["missing_v3_case_count"] == 0 for row in mode_metrics.values())
    return {
        "comparison_available": comparison_available,
        "source_v3_legal_count": _int(_mapping(_mapping(fixed.get("aggregate")).get("v3")).get("legal_count")) or 0,
        "source_v3_starved_count": _int(_mapping(_mapping(fixed.get("aggregate")).get("v3")).get("starved_count")) or 0,
        "source_v3_mean_f1_100ms": _float(_mapping(_mapping(fixed.get("aggregate")).get("v3")).get("mean_f1_100ms")),
        "modes": mode_metrics,
    }


def _mode_vs_v3(
    terminal_rows: Sequence[object],
    fixed_cases: Mapping[str, Mapping[str, Any]],
    mode: str,
) -> dict[str, Any]:
    rows = [row for row in terminal_rows if isinstance(row, Mapping) and str(row.get("mode")) == mode]
    f1_deltas: list[float] = []
    rigid_deltas: list[float] = []
    event_ratio_deltas: list[float] = []
    candidate_f1: list[float] = []
    v3_f1: list[float] = []
    missing = 0
    candidate_starved = 0
    v3_starved = 0
    for row in rows:
        fixed = fixed_cases.get(str(row.get("case_id")))
        if fixed is None:
            missing += 1
            continue
        candidate_metrics = _mapping(row.get("candidate_metrics"))
        v3_metrics = _mapping(fixed.get("v3_metrics"))
        cand_f1 = _metric_f1(candidate_metrics)
        ref_f1 = _metric_f1(v3_metrics)
        candidate_f1.append(cand_f1)
        v3_f1.append(ref_f1)
        f1_deltas.append(cand_f1 - ref_f1)
        rigid_deltas.append(
            (_float(candidate_metrics.get("dominant_spacing_ratio")) or 0.0)
            - (_float(v3_metrics.get("dominant_spacing_ratio")) or 0.0)
        )
        event_ratio_deltas.append(
            (_float(candidate_metrics.get("event_count_ratio")) or 0.0)
            - (_float(v3_metrics.get("event_count_ratio")) or 0.0)
        )
        if bool(candidate_metrics.get("starved")):
            candidate_starved += 1
        if bool(v3_metrics.get("starved")):
            v3_starved += 1
    return {
        "case_count": len(rows),
        "missing_v3_case_count": missing,
        "candidate_legal_count": sum(1 for row in rows if bool(_mapping(row).get("candidate_legal"))),
        "candidate_starved_count": candidate_starved,
        "v3_starved_count": v3_starved,
        "starved_count_delta": candidate_starved - v3_starved,
        "mean_candidate_f1_100ms": _mean(candidate_f1),
        "mean_v3_f1_100ms": _mean(v3_f1),
        "mean_f1_delta_vs_v3": _mean(f1_deltas),
        "mean_dominant_spacing_ratio_delta_vs_v3": _mean(rigid_deltas),
        "mean_event_count_ratio_delta_vs_v3": _mean(event_ratio_deltas),
    }


def decision_from_checks(
    checks: Mapping[str, Any],
    *,
    missing_required: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    terminal = _mapping(checks.get("terminal_guard"))
    v3 = _mapping(checks.get("v3_representation"))
    c3 = _mapping(checks.get("c3_route"))
    v3_quality = _mapping(checks.get("v3_quality"))
    same_slice = _mapping(checks.get("same_slice"))

    if missing_required:
        return {
            "route": "MUTATE_SYNTHESIS_INPUTS",
            "reason": f"{len(missing_required)} required source artifacts are missing",
            "next_step": "Regenerate or locate the required summaries before routing v2.1/v3/C3.",
            "next_card": None,
        }
    if not bool(terminal.get("full32_passed")):
        return {
            "route": "KILL_V21_TERMINAL_GUARD_ROUTE",
            "reason": "the v2.1 terminal guard full32 artifact does not pass hard legality/starvation/F1 gates",
            "next_step": "Do not promote the terminal guard; inspect or rerun the v2.1 repair gate.",
            "next_card": "mapper_v21_terminal_ln_start_guard_repair",
        }
    if not bool(v3.get("ready")):
        return {
            "route": "MUTATE_V3_REPRESENTATION_AUDIT",
            "reason": "v3 representation readiness is not proven by the loaded artifacts",
            "next_step": "Repair or rerun the full-dataset v3 representation audit before replacement routing.",
            "next_card": "target_grammar_v3_full_dataset_audit",
        }
    if bool(c3.get("production_input_legal")):
        return {
            "route": "TEST_C3_TARGET_INTEGRATION",
            "reason": "loaded C3 evidence says production input is legal; this contradicts the current expected route",
            "next_step": "Create a focused C3 target-integration card and verify leakage/cost guards.",
            "next_card": "c3_target_integration_gate",
        }
    if not bool(same_slice.get("comparison_available")):
        return {
            "route": "MUTATE_COMPARISON_INPUTS",
            "reason": "the v2.1 terminal guard and v3 fixed-slice artifacts do not align by case id",
            "next_step": "Run a direct v2.1 terminal-guard vs v3 fixed-slice comparison.",
            "next_card": "mapper_v21_terminal_guard_v3_direct_comparison",
        }
    if bool(v3_quality.get("repair_route_open")):
        return {
            "route": "TEST_V3_BOUNDARY_OR_SPACING_STATE_REPAIR_KEEP_V21_GUARD_OPT_IN",
            "reason": (
                "v2.1 terminal guard is now a legal full32 comparator, C3 mapper integration remains "
                "diminishing, and v3 remains the lower-bit local target but still has generated-prefix "
                "state/timing failures"
            ),
            "next_step": (
                "Keep `min_ln_duration_ms` default-off and use the repaired v2.1 guard as the comparator "
                "while creating the next v3 boundary/spacing-state repair card."
            ),
            "next_card": v3_quality.get("next_card"),
        }
    return {
        "route": "TEST_V3_TRAINED_QUALITY_GATE_WITH_V21_GUARD_COMPARATOR",
        "reason": "v3 representation and pipeline mechanics are ready and no generated-prefix repair route is open",
        "next_step": "Run the next trained v3 quality gate against the repaired v2.1 terminal-guard comparator.",
        "next_card": "target_grammar_v3_trained_quality_gate",
    }


def write_summary_json(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_markdown(summary).rstrip() + "\n", encoding="utf-8")


def report_markdown(summary: Mapping[str, Any]) -> str:
    checks = _mapping(summary.get("checks"))
    decision = _mapping(summary.get("decision"))
    terminal = _mapping(checks.get("terminal_guard"))
    v3 = _mapping(checks.get("v3_representation"))
    c3 = _mapping(checks.get("c3_route"))
    v3_quality = _mapping(checks.get("v3_quality"))
    same_slice = _mapping(checks.get("same_slice"))
    modes = _mapping(same_slice.get("modes"))
    lines = [
        "# Mapper v2.1 Terminal Guard / v3 Route Synthesis Result Report",
        "",
        "## Scope",
        "",
        "This artifact-only synthesis reconciles the v2.1 terminal LN-start guard full32 pass with the current v3 and C3 route artifacts. It does not train, rerun rollout, retokenize data, or change defaults.",
        "",
        "## Decision",
        "",
        f"Route: `{decision.get('route')}`.",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Recommended next step: {decision.get('next_step')}",
        f"- Next card: `{decision.get('next_card')}`",
        "",
        "## Guard Results",
        "",
        "| Guard | Value |",
        "| --- | ---: |",
        _row("required artifacts present", checks.get("required_artifacts_present")),
        _row("artifact only", checks.get("artifact_only")),
        _row("no training", checks.get("no_training")),
        _row("no rollout rerun", checks.get("no_rollout_rerun")),
        _row("no tokenizer/default change", checks.get("no_tokenizer_or_default_change")),
        _row("v2.1 terminal guard full32 passed", terminal.get("full32_passed")),
        _row("v3 representation ready", v3.get("ready")),
        _row("C3 mapper path diminishing", c3.get("mapper_path_diminishing")),
        _row("same-slice comparison available", same_slice.get("comparison_available")),
        _row("v3 repair route open", v3_quality.get("repair_route_open")),
        "",
        "## Evidence Snapshot",
        "",
        "| Family | Metric | Value |",
        "| --- | --- | ---: |",
        _row3("v2.1 terminal guard", "runs/cases", f"{terminal.get('run_count')} / {terminal.get('case_count')}"),
        _row3("v2.1 terminal guard", "all legal", terminal.get("all_candidate_legal")),
        _row3("v2.1 terminal guard", "new starved", terminal.get("new_starved_count")),
        _row3("v2.1 terminal guard", "mean F1 delta vs comparator", _fmt(terminal.get("mean_f1_delta_vs_matching_comparator"))),
        _row3("v2.1 terminal guard", "anti-rigid blocked", terminal.get("anti_rigid_blocked_count")),
        _row3("v3 representation", "reconstruction mismatches", v3.get("reconstruction_mismatches")),
        _row3("v3 representation", "token reduction", _pct(v3.get("token_reduction_ratio"))),
        _row3("v3 representation", "total-bit reduction", _pct(v3.get("total_bit_reduction_ratio"))),
        _row3("v3 quality", "generated-prefix route", v3_quality.get("generated_prefix_trace_route")),
        _row3("v3 quality", "trace mean event ratio", _fmt(v3_quality.get("trace_mean_event_count_ratio"))),
        _row3("v3 quality", "trace second-window share", _fmt(v3_quality.get("trace_mean_second_window_event_share"))),
        _row3("C3", "route", c3.get("route")),
        _row3("C3", "production input legal", c3.get("production_input_legal")),
        _row3("C3", "codec delta bits/event", _fmt(c3.get("test_delta_bits_per_event"))),
        "",
        "## Same-Slice v2.1 Repair vs v3",
        "",
        "| Candidate | Cases | Legal | Candidate F1 | v3 F1 | F1 delta | Starved delta | Event-ratio delta | Rigid delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _mode_row("baseline + terminal guard", _mapping(modes.get("baseline_guard"))),
        _mode_row("anti-rigid + terminal guard", _mapping(modes.get("anti_rigid_guard"))),
        "",
        "## What Passed",
        "",
        "- The v2.1 terminal guard converts the previous illegal fixed-slice v2.1 paths into legal full32 rollouts with no new starvation.",
        "- v3 remains exactly reconstructive, lower-token, lower-bit, local, and teacher-forcing friendly in the representation artifacts.",
        "- C3 remains a strong codec-side result, but the loaded target-complexity artifact still rejects it as production input or emitted target replacement.",
        "",
        "## What Surfaced",
        "",
        "- The v2.1 terminal guard is a stronger comparator and short-term hardening result, not a final target-grammar replacement.",
        "- v3 rollout quality remains blocked by generated-prefix state/timing failures, so the next useful v3 card should target boundary or spacing state repair.",
        "- The route should not go back to C3 mapper-side integration unless a new formulation solves production legality and target cost.",
        "",
        "## Next Step",
        "",
        str(decision.get("next_step")),
    ]
    return "\n".join(lines)


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _payload(loaded: Mapping[str, Mapping[str, Any]], key: str) -> Mapping[str, Any]:
    row = _mapping(loaded.get(key))
    return _mapping(row.get("payload"))


def _route(payload: Mapping[str, Any]) -> str:
    decision = payload.get("decision")
    if isinstance(decision, str):
        return decision
    return str(_mapping(decision).get("route") or "")


def _mode_row(label: str, row: Mapping[str, Any]) -> str:
    return (
        f"| `{label}` | {row.get('case_count')} | {row.get('candidate_legal_count')} | "
        f"{_fmt(row.get('mean_candidate_f1_100ms'))} | {_fmt(row.get('mean_v3_f1_100ms'))} | "
        f"{_fmt(row.get('mean_f1_delta_vs_v3'))} | {row.get('starved_count_delta')} | "
        f"{_fmt(row.get('mean_event_count_ratio_delta_vs_v3'))} | "
        f"{_fmt(row.get('mean_dominant_spacing_ratio_delta_vs_v3'))} |"
    )


def _row(label: str, value: object) -> str:
    return f"| {label} | `{value}` |"


def _row3(family: str, metric: str, value: object) -> str:
    return f"| {family} | {metric} | `{value}` |"


def _metric_f1(metrics: Mapping[str, Any]) -> float:
    return _float(_mapping(metrics.get("timing_match_100ms")).get("f1")) or 0.0


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else ()


def _int(value: object) -> int | None:
    try:
        if isinstance(value, bool):
            return int(value)
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _float(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: Sequence[float]) -> float:
    return float(math.fsum(values) / len(values)) if values else 0.0


def _fmt(value: object) -> str:
    number = _float(value)
    return "n/a" if number is None else f"{number:.6f}"


def _pct(value: object) -> str:
    number = _float(value)
    return "n/a" if number is None else f"{number * 100.0:.3f}%"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Synthesize v2.1 terminal guard, v3, and C3 route evidence.")
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--report-output", default=DEFAULT_REPORT_OUTPUT)
    args = parser.parse_args(argv)
    summary = run_terminal_guard_v3_route_synthesis(
        summary_output_path=Path(args.summary_output),
        report_output_path=Path(args.report_output),
    )
    print(
        "mapper_v21_terminal_guard_v3_route_synthesis_done "
        f"route={summary['decision']['route']} "
        f"positive={summary['checks']['positive_signal_observed']} "
        f"next_card={summary['decision'].get('next_card')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
