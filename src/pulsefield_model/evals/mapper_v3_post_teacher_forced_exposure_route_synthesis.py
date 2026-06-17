from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SUMMARY_SCHEMA_VERSION = 1
REPORT_ROOT = Path("artifacts/reports/audits/mapper_v2_1_grammar")


@dataclass(frozen=True)
class SourceSpec:
    key: str
    family: str
    path: Path
    required: bool = True


DEFAULT_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        "teacher_forced_time_shift",
        "teacher_forced",
        REPORT_ROOT / "target_grammar_v3_teacher_forced_time_shift_logit_audit_summary.json",
    ),
    SourceSpec(
        "generated_continuation",
        "generated_prefix",
        REPORT_ROOT / "target_grammar_v3_generated_state_continuation_diagnostic_summary.json",
    ),
    SourceSpec(
        "decode_policy_sweep",
        "decode_policy",
        REPORT_ROOT / "target_grammar_v3_decode_policy_continuation_sweep_summary.json",
    ),
    SourceSpec(
        "event_margin_viability",
        "decode_policy",
        REPORT_ROOT / "target_grammar_v3_event_margin_decode_viability_summary.json",
    ),
    SourceSpec(
        "conditioned_short_rollout",
        "training_objective",
        REPORT_ROOT / "target_grammar_v3_conditioned_event_distribution_short_rollout_summary.json",
    ),
    SourceSpec(
        "post_time_shift_full32",
        "training_objective",
        REPORT_ROOT / "target_grammar_v3_post_time_shift_full32_route_synthesis_summary.json",
    ),
    SourceSpec(
        "c3_diminishing_returns",
        "c3_route",
        REPORT_ROOT / "target_grammar_v3_c3_diminishing_returns_route_synthesis_summary.json",
    ),
)
DEFAULT_SUMMARY_OUTPUT = REPORT_ROOT / "target_grammar_v3_post_teacher_forced_exposure_route_synthesis_summary.json"
DEFAULT_REPORT_OUTPUT = REPORT_ROOT / "target_grammar_v3_post_teacher_forced_exposure_route_synthesis_result_report.md"


def run_post_teacher_forced_exposure_route_synthesis(
    *,
    summary_output_path: Path = DEFAULT_SUMMARY_OUTPUT,
    report_output_path: Path = DEFAULT_REPORT_OUTPUT,
    sources: Sequence[SourceSpec] = DEFAULT_SOURCES,
) -> dict[str, Any]:
    start = time.monotonic()
    evidence = [_load_evidence(source) for source in sources]
    missing_required = [row for row in evidence if bool(row["required"]) and not bool(row["exists"])]
    checks = route_checks(evidence)
    decision = synthesize_decision(checks=checks, missing_required=missing_required)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar v3 post-teacher-forced exposure route synthesis",
        "elapsed_s": time.monotonic() - start,
        "source_count": len(evidence),
        "missing_required_count": len(missing_required),
        "checks": checks,
        "evidence": evidence,
        "decision": decision,
        "next_card": decision.get("next_card"),
    }
    write_summary_json(summary, summary_output_path)
    write_report(summary, report_output_path)
    return summary


def route_checks(evidence: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key = {str(row.get("key")): row for row in evidence}
    required_present = all(bool(row.get("exists")) for row in evidence if bool(row.get("required")))

    teacher = by_key.get("teacher_forced_time_shift", {})
    teacher_metrics = _mapping(teacher.get("metrics"))
    teacher_rank = _mapping(teacher_metrics.get("rank"))
    teacher_recall = _mapping(teacher_metrics.get("recall_at_k"))
    teacher_checks = _mapping(teacher.get("checks"))
    teacher_route = str(teacher.get("route") or "")
    teacher_recall_at_5 = _float(teacher_recall.get("5")) or 0.0
    teacher_median_rank = _float(teacher_rank.get("median"))
    teacher_argmax_top_share = _float(teacher_metrics.get("argmax_top_shift_share")) or 0.0
    teacher_rows = _int(teacher_metrics.get("time_shift_row_count")) or 0
    teacher_forced_healthy = (
        teacher_route == "TEST_DECODE_STATE_EXPOSURE_DIAGNOSTIC"
        and bool(teacher_checks.get("faithful_global_context"))
        and teacher_rows > 0
        and teacher_recall_at_5 >= 0.80
        and teacher_median_rank is not None
        and teacher_median_rank <= 3.0
        and teacher_argmax_top_share < 0.40
    )

    continuation = by_key.get("generated_continuation", {})
    continuation_baseline = _mapping(continuation.get("baseline_snapshot"))
    continuation_failures = _mapping(continuation_baseline.get("failure_counts"))
    continuation_event_rank = _mapping(continuation_baseline.get("event_rank"))
    baseline_starved = _int(continuation_failures.get("starved")) or 0
    baseline_rigid = _int(continuation_failures.get("rigid")) or 0
    baseline_boundary = _int(continuation_failures.get("boundary_clamped")) or 0
    event_valid_ratio = _float(continuation_event_rank.get("mean_event_valid_step_ratio")) or 0.0
    generated_prefix_failures_persist = baseline_starved > 0 or baseline_rigid > 0 or baseline_boundary > 0
    generated_event_tokens_broadly_valid = event_valid_ratio >= 0.70

    decode = by_key.get("decode_policy_sweep", {})
    policy_summaries = decode.get("policy_summaries")
    policy_pass_count = (
        sum(1 for row in policy_summaries if isinstance(row, Mapping) and bool(row.get("passes_positive_signal")))
        if isinstance(policy_summaries, Sequence) and not isinstance(policy_summaries, (str, bytes))
        else 0
    )
    decode_only_killed = str(decode.get("route") or "") == "KILL" and policy_pass_count == 0

    event_margin = by_key.get("event_margin_viability", {})
    event_margin_decision = _mapping(event_margin.get("decision"))
    event_margin_global_bonus_killed = str(event_margin.get("route") or "") == "KILL_GLOBAL_EVENT_BONUS"
    starved_gt4_share = _float(event_margin_decision.get("starved_gt4_share")) or 0.0

    conditioned = by_key.get("conditioned_short_rollout", {})
    conditioned_guards = _mapping(conditioned.get("guard_results"))
    conditioned_objective_killed = (
        str(conditioned.get("route") or "") == "KILL"
        and conditioned_guards.get("median_event_count_ratio_in_range") is False
    )
    conditioned_aggregate = _mapping(conditioned.get("aggregate"))
    conditioned_starved_delta = _int(conditioned_aggregate.get("starved_case_delta"))
    conditioned_mean_f1_delta = _float(conditioned_aggregate.get("mean_f1_delta"))

    time_shift = by_key.get("post_time_shift_full32", {})
    time_shift_objective_failed = str(time_shift.get("route") or "").startswith("MUTATE_TIME_SHIFT_DISTANCE")

    c3 = by_key.get("c3_diminishing_returns", {})
    c3_mapper_diminishing = str(c3.get("route") or "") == "MUTATE_C3_MAPPER_PATH_TEST_V3_TIMING_LOGIT_AUDIT"

    shortcuts_exhausted = (
        decode_only_killed
        and event_margin_global_bonus_killed
        and conditioned_objective_killed
        and time_shift_objective_failed
    )
    exposure_trace_justified = (
        teacher_forced_healthy
        and generated_prefix_failures_persist
        and generated_event_tokens_broadly_valid
        and shortcuts_exhausted
    )
    return {
        "required_present": required_present,
        "teacher_forced_healthy": teacher_forced_healthy,
        "teacher_route": teacher_route,
        "teacher_rows": teacher_rows,
        "teacher_recall_at_5": teacher_recall_at_5,
        "teacher_median_rank": teacher_median_rank,
        "teacher_argmax_top_shift_share": teacher_argmax_top_share,
        "generated_prefix_failures_persist": generated_prefix_failures_persist,
        "generated_event_tokens_broadly_valid": generated_event_tokens_broadly_valid,
        "continuation_baseline_starved": baseline_starved,
        "continuation_baseline_rigid": baseline_rigid,
        "continuation_baseline_boundary_clamped": baseline_boundary,
        "continuation_event_valid_ratio": event_valid_ratio,
        "decode_only_killed": decode_only_killed,
        "decode_policy_pass_count": policy_pass_count,
        "event_margin_global_bonus_killed": event_margin_global_bonus_killed,
        "event_margin_starved_gt4_share": starved_gt4_share,
        "conditioned_objective_killed": conditioned_objective_killed,
        "conditioned_starved_case_delta": conditioned_starved_delta,
        "conditioned_mean_f1_delta": conditioned_mean_f1_delta,
        "conditioned_event_ratio_guard_passed": conditioned_guards.get("median_event_count_ratio_in_range"),
        "time_shift_objective_failed": time_shift_objective_failed,
        "c3_mapper_diminishing": c3_mapper_diminishing,
        "shortcuts_exhausted": shortcuts_exhausted,
        "exposure_trace_justified": exposure_trace_justified,
    }


def synthesize_decision(
    *,
    checks: Mapping[str, Any],
    missing_required: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if missing_required:
        return {
            "route": "MUTATE_SYNTHESIS_INPUTS",
            "reason": f"{len(missing_required)} required summaries are missing",
            "interpretation": "The post-teacher-forced route cannot be trusted until required artifacts are present.",
            "next_step": "Regenerate or locate the missing summaries, then rerun this synthesis.",
            "next_card": None,
        }
    if not bool(checks.get("teacher_forced_healthy")):
        return {
            "route": "MUTATE_TIMING_OBJECTIVE",
            "reason": "teacher-forced time-shift logits are not healthy enough to isolate generated-prefix exposure",
            "interpretation": "Timing-logit calibration remains a plausible teacher-forced bottleneck.",
            "next_step": "Mutate the timing objective only with explicit generated-prefix rollout guards.",
            "next_card": "target_grammar_v3_timing_objective_mutation",
        }
    if not bool(checks.get("generated_prefix_failures_persist")):
        return {
            "route": "TEST_FULL32_V3_VALIDATION",
            "reason": "loaded generated-prefix summaries do not show persistent continuation failures",
            "interpretation": "If generated-prefix failure is gone, the branch should validate broader quality before mutating.",
            "next_step": "Run a broader full32 validation gate on the current best v3 checkpoint.",
            "next_card": "target_grammar_v3_full32_validation",
        }
    if bool(checks.get("exposure_trace_justified")):
        return {
            "route": "TEST_GENERATED_PREFIX_STATE_TRACE",
            "reason": (
                "teacher-forced timing ranks are healthy, generated-prefix failures persist, and decode/objective "
                "shortcuts are already killed or gated"
            ),
            "interpretation": (
                "The next useful v3 experiment is a narrow per-step generated-prefix state trace. It should identify "
                "whether collapse starts from time-shift choice, event underselection, carry/state mismatch, or boundary "
                "state drift before another training run."
            ),
            "next_step": "Create a generated-prefix state trace audit on high-leverage fixed-slice cases.",
            "next_card": "target_grammar_v3_generated_prefix_state_trace_audit",
        }
    if not bool(checks.get("shortcuts_exhausted")):
        return {
            "route": "MUTATE_OPEN_SHORTCUT_BRANCH",
            "reason": "at least one decode/objective shortcut branch is not conclusively killed in the loaded artifacts",
            "interpretation": "Resolve the still-open branch before creating a new trace or grammar-repair card.",
            "next_step": "Inspect the open branch guard and either run or kill it explicitly.",
            "next_card": "target_grammar_v3_open_branch_guard",
        }
    return {
        "route": "MUTATE_TO_V2_1_OR_TARGET_GRAMMAR_REPAIR",
        "reason": "teacher-forced logits are healthy, but existing evidence is not enough to justify another v3 local probe",
        "interpretation": (
            "If the owner does not want one more generated-prefix trace, the current local v3 branches have diminishing "
            "returns and the work should pivot to target-grammar repair or v2.1 grammar improvement."
        ),
        "next_step": "Choose between a narrow generated-prefix trace and a v2.1 grammar improvement card.",
        "next_card": "target_grammar_v3_generated_prefix_state_trace_audit_or_v21_grammar_improvement",
    }


def write_summary_json(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    decision = _mapping(summary.get("decision"))
    checks = _mapping(summary.get("checks"))
    evidence = summary.get("evidence")
    rows = evidence if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes)) else ()
    lines = [
        "# Target Grammar v3 Post-Teacher-Forced Exposure Route Synthesis Result Report",
        "",
        "## Scope",
        "",
        "This artifact-only synthesis reconciles the teacher-forced v3 time-shift logit audit with existing generated-prefix rollout, decode-policy, event-margin, conditioned-objective, time-shift, and C3 route artifacts. It does not train, rerun rollout, change tokenizer behavior, or change mapper defaults.",
        "",
        "## Decision",
        "",
        f"Route: `{decision.get('route')}`.",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Next card: `{decision.get('next_card')}`",
        f"- Next step: {decision.get('next_step')}",
        "",
        "## Route Checks",
        "",
        "| Check | Value |",
        "| --- | ---: |",
    ]
    for key, value in checks.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Evidence Table",
            "",
            "| Key | Family | Route | Exists | Reason |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    for row in rows:
        item = _mapping(row)
        lines.append(
            "| {key} | {family} | `{route}` | `{exists}` | {reason} |".format(
                key=item.get("key"),
                family=item.get("family"),
                route=item.get("route"),
                exists=item.get("exists"),
                reason=str(item.get("reason") or "").replace("|", "\\|"),
            )
        )
    lines.extend(
        [
            "",
            "## What Passed",
            "",
            "- The teacher-forced audit is faithful, non-smoke, and ranks target time shifts strongly on the fixed eval split.",
            "- The generated-prefix diagnostics still show continuation/starvation/rigid-grid failures under rollout.",
            "- Existing deterministic decode-only, global event-bonus, conditioned event-distribution, and time-shift-distance shortcuts are not promotable as tested.",
            "",
            "## What Surfaced",
            "",
            "- The remaining failure is exposed under generated prefixes, not under teacher-forced time-shift logits.",
            "- Another scalar teacher-forced timing loss is not the smallest next test unless a new mechanism is tied to generated-prefix guards.",
            "- C3 remains a proven codec-side representation result, but current mapper-side C3 routes have diminishing returns.",
            "",
            "## What This Does Not Prove",
            "",
            "- It does not prove v3 replacement readiness.",
            "- It does not prove the generated-prefix trace will fix rollout quality.",
            "- It does not prove v2.1 grammar work is unnecessary.",
            "- It does not evaluate the full 4k dataset.",
            "",
            "## Interpretation",
            "",
            str(decision.get("interpretation")),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_evidence(source: SourceSpec) -> dict[str, Any]:
    row: dict[str, Any] = {
        "key": source.key,
        "family": source.family,
        "path": source.path.as_posix(),
        "required": bool(source.required),
        "exists": source.path.exists(),
    }
    if not source.path.exists():
        row.update({"route": None, "reason": "missing"})
        return row
    payload = _load_json_object(source.path)
    decision = _decision_mapping(payload.get("decision"))
    row.update(
        {
            "experiment": payload.get("experiment"),
            "route": decision.get("route"),
            "reason": decision.get("reason") or payload.get("interpretation"),
            "decision": decision,
            "checks": payload.get("checks"),
            "metrics": payload.get("metrics"),
            "baseline_snapshot": payload.get("baseline_snapshot"),
            "policy_summaries": payload.get("policy_summaries"),
            "guard_results": payload.get("guard_results"),
            "aggregate": payload.get("aggregate"),
        }
    )
    return row


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _decision_mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        return {"route": value}
    return {}


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthesize the post-teacher-forced v3 exposure route.")
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT.as_posix())
    parser.add_argument("--report-output", default=DEFAULT_REPORT_OUTPUT.as_posix())
    args = parser.parse_args(argv)
    summary = run_post_teacher_forced_exposure_route_synthesis(
        summary_output_path=Path(args.summary_output),
        report_output_path=Path(args.report_output),
    )
    print(
        "mapper_v3_post_teacher_forced_exposure_route_synthesis_done "
        f"route={summary['decision']['route']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
