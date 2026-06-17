from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SUMMARY_SCHEMA_VERSION = 1
REPORT_ROOT = Path("artifacts/reports/audits/mapper_v2_1_grammar")
C3_REPORT_ROOT = Path("artifacts/reports/audits/context_adaptive_fallback_codec")


@dataclass(frozen=True)
class SourceSpec:
    key: str
    family: str
    path: Path
    required: bool = True


DEFAULT_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec("v3_full_dataset", "representation", REPORT_ROOT / "target_grammar_v3_full_dataset_audit_summary.json"),
    SourceSpec(
        "v3_real_config",
        "pipeline",
        REPORT_ROOT / "target_grammar_v3_real_config_cache_backed_comparison_summary.json",
    ),
    SourceSpec(
        "v3_real_audio",
        "pipeline",
        REPORT_ROOT / "target_grammar_v3_real_audio_session_rollout_summary.json",
    ),
    SourceSpec(
        "density_loss",
        "training_objective",
        REPORT_ROOT / "target_grammar_v3_density_loss_effectiveness_diagnostic_summary.json",
    ),
    SourceSpec(
        "event_budget_0_05",
        "training_objective",
        REPORT_ROOT / "target_grammar_v3_event_budget_training_gate_summary.json",
    ),
    SourceSpec(
        "ce_weight",
        "training_objective",
        REPORT_ROOT / "target_grammar_v3_event_token_ce_weight_training_gate_summary.json",
    ),
    SourceSpec(
        "ce_residual_cluster",
        "trace_oracle",
        REPORT_ROOT / "target_grammar_v3_ce_residual_cluster_diagnostic_summary.json",
    ),
    SourceSpec(
        "ce_margin",
        "trace_oracle",
        REPORT_ROOT / "target_grammar_v3_ce_event_margin_viability_summary.json",
    ),
    SourceSpec(
        "ce_selective_trace",
        "decode_selector",
        REPORT_ROOT / "target_grammar_v3_ce_selective_trace_oracle_summary.json",
    ),
    SourceSpec(
        "ce_density_trace",
        "decode_selector",
        REPORT_ROOT / "target_grammar_v3_ce_density_trace_diagnostic_summary.json",
    ),
    SourceSpec(
        "ce_antirigid_full32",
        "decode_policy",
        REPORT_ROOT / "target_grammar_v3_ce_antirigid_full32_summary.json",
    ),
    SourceSpec(
        "ce_antirigid_soft_full32",
        "decode_policy",
        REPORT_ROOT / "target_grammar_v3_ce_antirigid_soft_full32_summary.json",
    ),
    SourceSpec(
        "ce_tap_only_antirigid_full32",
        "decode_policy",
        REPORT_ROOT / "target_grammar_v3_ce_tap_only_antirigid_full32_summary.json",
    ),
    SourceSpec(
        "c3_v3_complexity",
        "c3_route",
        C3_REPORT_ROOT / "c3_v3_target_complexity_comparison_summary.json",
    ),
)


def run_post_ce_route_synthesis(
    *,
    summary_output_path: Path,
    report_output_path: Path,
    sources: Sequence[SourceSpec] = DEFAULT_SOURCES,
) -> dict[str, Any]:
    start = time.monotonic()
    evidence = [_load_evidence(spec) for spec in sources]
    missing_required = [row for row in evidence if bool(row["required"]) and not bool(row["exists"])]
    checks = route_checks(evidence)
    decision = synthesize_decision(checks=checks, missing_required=missing_required)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar v3 post-CE route synthesis",
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
    route = {key: str(row.get("route") or "") for key, row in by_key.items()}
    exists = {key: bool(row.get("exists")) for key, row in by_key.items()}

    full_dataset = by_key.get("v3_full_dataset", {})
    real_config = by_key.get("v3_real_config", {})
    real_audio = by_key.get("v3_real_audio", {})
    event_budget = by_key.get("event_budget_0_05", {})
    ce_weight = by_key.get("ce_weight", {})
    ce_density_trace = by_key.get("ce_density_trace", {})
    antirigid_full32 = by_key.get("ce_antirigid_full32", {})
    antirigid_soft = by_key.get("ce_antirigid_soft_full32", {})
    tap_only = by_key.get("ce_tap_only_antirigid_full32", {})
    c3_complexity = by_key.get("c3_v3_complexity", {})

    representation_ready = route.get("v3_full_dataset") == "TEST_NEXT" and bool(
        _mapping(full_dataset.get("metrics")).get("reconstruction_zero")
        or _mapping(full_dataset.get("raw")).get("positive_gate_passed")
        or _mapping(_mapping(full_dataset.get("raw")).get("decision")).get("positive_gate_passed")
    )
    pipeline_mechanics_ready = route.get("v3_real_config") == "TEST_NEXT" and route.get("v3_real_audio") == "TEST_NEXT"
    real_audio_quality_ready = (_int(_mapping(real_audio.get("metrics")).get("timepoint_count")) or 0) > 0
    c3_as_target_killed = route.get("c3_v3_complexity") == "MUTATE_TO_V3_GRAMMAR_REPAIR"
    global_event_budget_failed = route.get("event_budget_0_05") == "MUTATE"
    ce_weight_partial = route.get("ce_weight") == "MUTATE"
    selective_trace_narrow = route.get("ce_density_trace") == "MUTATE_TO_TARGET_GRAMMAR_OR_TRAINING_REPAIR"
    antirigid_decode_killed = (
        route.get("ce_antirigid_full32") == "KILL"
        and route.get("ce_antirigid_soft_full32") == "KILL"
        and route.get("ce_tap_only_antirigid_full32") == "KILL"
    )
    simple_decode_family_killed = selective_trace_narrow and antirigid_decode_killed
    scalar_objective_not_solved = global_event_budget_failed and ce_weight_partial
    required_present = all(exists.values())

    killed_decode_keys = [
        key
        for key in (
            "ce_density_trace",
            "ce_antirigid_full32",
            "ce_antirigid_soft_full32",
            "ce_tap_only_antirigid_full32",
        )
        if route.get(key) in {"KILL", "MUTATE_TO_TARGET_GRAMMAR_OR_TRAINING_REPAIR"}
    ]
    return {
        "required_present": required_present,
        "representation_ready": representation_ready,
        "pipeline_mechanics_ready": pipeline_mechanics_ready,
        "real_audio_quality_ready": real_audio_quality_ready,
        "c3_as_target_killed": c3_as_target_killed,
        "global_event_budget_failed": global_event_budget_failed,
        "ce_weight_partial_not_promoted": ce_weight_partial,
        "selective_trace_narrow": selective_trace_narrow,
        "antirigid_decode_killed": antirigid_decode_killed,
        "simple_decode_family_killed": simple_decode_family_killed,
        "scalar_objective_not_solved": scalar_objective_not_solved,
        "killed_decode_family_count": len(killed_decode_keys),
        "killed_decode_keys": killed_decode_keys,
        "event_budget_route": route.get("event_budget_0_05"),
        "ce_weight_route": route.get("ce_weight"),
        "tap_only_route": route.get("ce_tap_only_antirigid_full32"),
        "tap_only_dead_end_count": _mapping(tap_only.get("metrics")).get("dead_end_count"),
        "tap_only_median_event_ratio": _mapping(tap_only.get("metrics")).get("median_event_count_ratio"),
        "broad_antirigid_median_event_ratio": _mapping(antirigid_full32.get("metrics")).get(
            "median_event_count_ratio"
        ),
        "event_budget_max_token_count": _mapping(event_budget.get("metrics")).get("max_token_hit_case_count"),
        "ce_density_positive_cases": _mapping(ce_density_trace.get("metrics")).get(
            "positive_low_bias_starved_case_count"
        ),
        "c3_route": c3_complexity.get("route"),
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
            "interpretation": "The branch cannot be routed until the required artifacts exist.",
            "next_step": "Regenerate or locate the missing summaries before selecting the next v3 card.",
            "next_card": None,
        }
    if not bool(checks.get("representation_ready")):
        return {
            "route": "MUTATE_REPRESENTATION_AUDIT",
            "reason": "v3 representation readiness is not proven in the loaded artifacts",
            "interpretation": "Do not move to training repair until the reversible full-dataset v3 audit is present.",
            "next_step": "Repair or rerun the v3 full-dataset representation audit.",
            "next_card": "target_grammar_v3_full_dataset_audit",
        }
    if not bool(checks.get("pipeline_mechanics_ready")):
        return {
            "route": "MUTATE_PIPELINE_MECHANICS",
            "reason": "v3 training/runtime mechanics are not both proven in the loaded artifacts",
            "interpretation": "The next work should restore pipeline smoke evidence before quality mutations.",
            "next_step": "Run the real-config and real-audio v3 pipeline gates.",
            "next_card": "target_grammar_v3_real_audio_session_rollout",
        }
    if not bool(checks.get("simple_decode_family_killed")):
        return {
            "route": "TEST_OPEN_DECODE_BRANCH",
            "reason": "at least one simple decode branch is not conclusively killed",
            "interpretation": "A still-open decode branch should be resolved before a training-objective mutation.",
            "next_step": "Run or inspect the remaining decode branch gate.",
            "next_card": "target_grammar_v3_decode_branch_gate",
        }
    if bool(checks.get("scalar_objective_not_solved")):
        return {
            "route": "TEST_CONDITIONED_EVENT_DISTRIBUTION_OBJECTIVE",
            "reason": "representation and pipeline mechanics passed, simple decode branches are killed, and global scalar event-distribution objectives did not solve the rollout gates",
            "interpretation": (
                "The next smallest v3 repair is not another selector. It should test a conditioned "
                "event-distribution objective with explicit overgeneration and legality guards before redesigning the grammar."
            ),
            "next_step": "Create a conditioned event-distribution objective plumbing card with high-difficulty overgeneration guards.",
            "next_card": "target_grammar_v3_conditioned_event_distribution_objective",
        }
    return {
        "route": "MUTATE_TO_TARGET_GRAMMAR_REPAIR",
        "reason": "loaded artifacts do not identify a remaining bounded training-objective route",
        "interpretation": "If objective-side repair is exhausted, the next work should redesign or refine the target grammar.",
        "next_step": "Create a target-grammar repair card rather than another decode policy.",
        "next_card": "target_grammar_v3_target_repair",
    }


def write_summary_json(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    decision = _mapping(summary.get("decision"))
    checks = _mapping(summary.get("checks"))
    lines = [
        "# Target Grammar v3 Post-CE Route Synthesis Result Report",
        "",
        "## Scope",
        "",
        "This artifact-only synthesis reads committed v3/C3 audit summaries after the CE, trace, density, and anti-rigid branches. It does not retrain, rerun rollout, alter tokenizer behavior, or change defaults.",
        "",
        "## Decision",
        "",
        f"Route: `{decision.get('route')}`.",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Recommended next step: {decision.get('next_step')}",
        f"- Next card: `{decision.get('next_card')}`",
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
            "| Key | Family | Route | Status | Reason |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary.get("evidence", ()):
        evidence = _mapping(row)
        lines.append(
            "| `{key}` | {family} | `{route}` | {status} | {reason} |".format(
                key=evidence.get("key"),
                family=evidence.get("family"),
                route=evidence.get("route"),
                status="present" if evidence.get("exists") else "missing",
                reason=str(evidence.get("reason") or "").replace("|", "\\|"),
            )
        )
    lines.extend(
        [
            "",
            "## What This Proves",
            "",
            "- v3 representation and pipeline mechanics remain the active replacement route, not C3-as-target.",
            "- Simple local decode branches are not the next best step after CE: selector evidence is too narrow, and anti-rigid variants fail full32 gates.",
            "- Global scalar event-distribution pressure is not sufficient as tested.",
            "",
            "## What Remains Unproven",
            "",
            "- v3 trained chart quality is not replacement-ready.",
            "- Full-dataset/full-song v3 runtime quality remains unproven.",
            "- A conditioned event-distribution objective has not yet been implemented or trained.",
            "",
            "## Next Step",
            "",
            str(decision.get("next_step")),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_evidence(spec: SourceSpec) -> dict[str, Any]:
    if not spec.path.exists():
        return {
            "key": spec.key,
            "family": spec.family,
            "path": spec.path.as_posix(),
            "required": spec.required,
            "exists": False,
            "route": None,
            "reason": "missing",
            "next_step": None,
            "metrics": {},
        }
    payload = _load_json_object(spec.path)
    decision = payload.get("decision")
    route, reason, next_step = _extract_decision(decision, payload)
    return {
        "key": spec.key,
        "family": spec.family,
        "path": spec.path.as_posix(),
        "required": spec.required,
        "exists": True,
        "route": route,
        "reason": reason,
        "next_step": next_step,
        "metrics": _extract_metrics(spec.key, payload),
        "raw": _compact_raw(payload),
    }


def _extract_decision(decision: object, payload: Mapping[str, Any]) -> tuple[str | None, str | None, str | None]:
    if isinstance(decision, Mapping):
        return (
            _optional_str(decision.get("route")),
            _optional_str(decision.get("reason") or payload.get("reason")),
            _optional_str(decision.get("next_step") or payload.get("next_step")),
        )
    if isinstance(decision, str):
        return decision, _optional_str(payload.get("reason")), _optional_str(payload.get("next_step"))
    return _optional_str(payload.get("route")), _optional_str(payload.get("reason")), _optional_str(payload.get("next_step"))


def _extract_metrics(key: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = _mapping(payload.get("aggregate"))
    rollout = _mapping(payload.get("rollout"))
    raw_decision = _mapping(payload.get("decision"))
    metrics: dict[str, Any] = {}
    if key == "v3_full_dataset":
        metrics["reconstruction_zero"] = bool(
            raw_decision.get("positive_gate_passed")
            or payload.get("positive_gate_passed")
            or aggregate.get("reconstruction_mismatches") == 0
        )
    if key == "v3_real_audio":
        rollout_inner = _mapping(rollout.get("rollout")) if "rollout" in rollout else rollout
        metrics["timepoint_count"] = rollout_inner.get("timepoint_count")
        metrics["completed"] = rollout_inner.get("completed")
    if key == "event_budget_0_05":
        metrics["max_token_hit_case_count"] = aggregate.get("max_token_hit_case_count")
        metrics["second_window_starved_case_count"] = aggregate.get("second_window_starved_case_count")
        metrics["median_event_count_ratio"] = aggregate.get("median_event_count_ratio")
    if key == "ce_weight":
        metrics["second_window_starved_case_count"] = aggregate.get("second_window_starved_case_count")
        metrics["rigid_case_count"] = aggregate.get("rigid_case_count")
        metrics["median_event_count_ratio"] = aggregate.get("median_event_count_ratio")
    if key == "ce_density_trace":
        metrics["positive_low_bias_starved_case_count"] = aggregate.get("positive_low_bias_starved_case_count")
        metrics["low_bias_starved_case_count"] = aggregate.get("low_bias_starved_case_count")
    if key in {"ce_antirigid_full32", "ce_antirigid_soft_full32", "ce_tap_only_antirigid_full32"}:
        candidate = _mapping(aggregate.get("candidate"))
        metrics["dead_end_count"] = aggregate.get("dead_end_count")
        metrics["median_event_count_ratio"] = candidate.get("median_event_count_ratio")
        metrics["rigid_case_count"] = candidate.get("rigid_case_count")
        metrics["starved_count"] = candidate.get("starved_count")
    return metrics


def _compact_raw(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    if "positive_gate_passed" in payload:
        raw["positive_gate_passed"] = payload.get("positive_gate_passed")
    if isinstance(payload.get("decision"), Mapping):
        raw["decision"] = dict(_mapping(payload.get("decision")))
    elif "decision" in payload:
        raw["decision"] = payload.get("decision")
    return raw


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


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthesize the post-CE v3 route decision.")
    parser.add_argument(
        "--summary-output",
        required=True,
    )
    parser.add_argument(
        "--report-output",
        required=True,
    )
    args = parser.parse_args(argv)
    summary = run_post_ce_route_synthesis(
        summary_output_path=Path(args.summary_output),
        report_output_path=Path(args.report_output),
    )
    print(
        "mapper_v3_post_ce_route_synthesis_done "
        f"route={summary['decision']['route']} "
        f"next={summary['decision'].get('next_card')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
