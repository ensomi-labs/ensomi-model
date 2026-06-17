from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence


SUMMARY_SCHEMA_VERSION = 1
SECOND_WINDOW_START_MS = 8000
OPPORTUNITY_MAX_RANK = 5
OPPORTUNITY_MIN_MARGIN = -2.0
DENSITY_LOOKBACK_MS = 1000
SPARSE_DENSITY_MAX_EVENTS = 1
POSITIVE_OPPORTUNITY_COUNT = 10
POSITIVE_STARVED_CASE_COUNT = 3


def run_ce_density_trace_diagnostic(
    *,
    trace_oracle_summary_path: Path,
    summary_output_path: Path,
    report_output_path: Path,
    density_lookback_ms: int = DENSITY_LOOKBACK_MS,
    sparse_density_max_events: int = SPARSE_DENSITY_MAX_EVENTS,
    positive_opportunity_count: int = POSITIVE_OPPORTUNITY_COUNT,
    positive_starved_case_count: int = POSITIVE_STARVED_CASE_COUNT,
) -> dict[str, Any]:
    start = time.monotonic()
    trace_oracle = _load_json_object(trace_oracle_summary_path)
    case_results = trace_oracle.get("case_results")
    if not isinstance(case_results, list) or not case_results:
        raise ValueError("trace oracle summary must contain non-empty case_results")

    cases = [
        analyze_density_case(
            trace_oracle_result=result,
            density_lookback_ms=int(density_lookback_ms),
            sparse_density_max_events=int(sparse_density_max_events),
            positive_opportunity_count=int(positive_opportunity_count),
        )
        for result in case_results
        if isinstance(result, Mapping)
    ]
    summary = summarize_density_diagnostic(
        cases=cases,
        trace_oracle_summary_path=trace_oracle_summary_path,
        elapsed_s=time.monotonic() - start,
        density_lookback_ms=int(density_lookback_ms),
        sparse_density_max_events=int(sparse_density_max_events),
        positive_opportunity_count=int(positive_opportunity_count),
        positive_starved_case_count=int(positive_starved_case_count),
    )
    write_summary_json(summary, summary_output_path)
    write_report(summary, report_output_path)
    return summary


def analyze_density_case(
    *,
    trace_oracle_result: Mapping[str, Any],
    density_lookback_ms: int = DENSITY_LOOKBACK_MS,
    sparse_density_max_events: int = SPARSE_DENSITY_MAX_EVENTS,
    positive_opportunity_count: int = POSITIVE_OPPORTUNITY_COUNT,
) -> dict[str, Any]:
    case = _mapping(trace_oracle_result.get("case"))
    trace_summary_path = Path(str(trace_oracle_result.get("trace_summary_path") or ""))
    if not trace_summary_path.exists():
        raise FileNotFoundError(f"missing trace summary: {trace_summary_path}")
    trace_summary = _load_json_object(trace_summary_path)
    diagnostics = _mapping(trace_summary.get("logit_diagnostics"))
    examples = diagnostics.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError(f"trace summary has no logit examples: {trace_summary_path}")
    rollout = _mapping(trace_summary.get("rollout"))
    timepoints = rollout.get("timepoints")
    if not isinstance(timepoints, list):
        raise ValueError(f"trace summary has no generated timepoints list: {trace_summary_path}")
    generated_times = sorted(
        int(timepoint["time_ms"])
        for timepoint in timepoints
        if isinstance(timepoint, Mapping) and _int(timepoint.get("time_ms")) is not None
    )
    second_window_generated_times = [value for value in generated_times if value >= SECOND_WINDOW_START_MS]

    second_steps = 0
    second_event_argmax = 0
    second_event_rank_near = 0
    second_event_margin_near = 0
    second_opportunities = 0
    sparse_second_steps = 0
    sparse_context_opportunities = 0
    dense_context_opportunities = 0
    opportunity_density_values: list[int] = []
    sparse_opportunity_examples: list[dict[str, Any]] = []
    longest_opportunity_run = 0
    longest_sparse_opportunity_run = 0
    current_opportunity_run = 0
    current_sparse_opportunity_run = 0

    for example in sorted((row for row in examples if isinstance(row, Mapping)), key=lambda row: _int(row.get("step")) or 0):
        current_ms = _int(example.get("current_ms"))
        if current_ms is None or current_ms < SECOND_WINDOW_START_MS:
            current_opportunity_run = 0
            current_sparse_opportunity_run = 0
            continue

        second_steps += 1
        argmax_kind = str(example.get("argmax_kind"))
        if argmax_kind == "event":
            second_event_argmax += 1
        best_event_rank = _float(example.get("best_event_rank"))
        best_event_margin = _float(example.get("best_event_margin_vs_argmax"))
        if best_event_rank is not None and best_event_rank <= OPPORTUNITY_MAX_RANK:
            second_event_rank_near += 1
        if best_event_margin is not None and best_event_margin >= OPPORTUNITY_MIN_MARGIN:
            second_event_margin_near += 1
        generated_prev_density = _count_prior_events(
            generated_times,
            current_ms=current_ms,
            lookback_ms=int(density_lookback_ms),
        )
        sparse_context = generated_prev_density <= int(sparse_density_max_events)
        if sparse_context:
            sparse_second_steps += 1
        opportunity = (
            argmax_kind != "event"
            and best_event_rank is not None
            and best_event_rank <= OPPORTUNITY_MAX_RANK
            and best_event_margin is not None
            and best_event_margin >= OPPORTUNITY_MIN_MARGIN
        )
        if opportunity:
            second_opportunities += 1
            opportunity_density_values.append(generated_prev_density)
            current_opportunity_run += 1
            longest_opportunity_run = max(longest_opportunity_run, current_opportunity_run)
            if sparse_context:
                sparse_context_opportunities += 1
                current_sparse_opportunity_run += 1
                longest_sparse_opportunity_run = max(
                    longest_sparse_opportunity_run,
                    current_sparse_opportunity_run,
                )
                if len(sparse_opportunity_examples) < 12:
                    sparse_opportunity_examples.append(
                        {
                            "step": _int(example.get("step")),
                            "current_ms": current_ms,
                            "argmax_token": example.get("argmax_token"),
                            "best_event_token": example.get("best_event_token"),
                            "best_event_rank": best_event_rank,
                            "best_event_margin_vs_argmax": best_event_margin,
                            "generated_prev_lookback_count": generated_prev_density,
                        }
                    )
            else:
                dense_context_opportunities += 1
                current_sparse_opportunity_run = 0
        else:
            current_opportunity_run = 0
            current_sparse_opportunity_run = 0

    step_count = _int(diagnostics.get("step_count")) or 0
    full_trace = len(examples) >= step_count if step_count else False
    legal = bool(rollout.get("completed")) and not bool(rollout.get("dead_end")) and not bool(
        rollout.get("max_tokens_exceeded")
    )
    positive = sparse_context_opportunities >= int(positive_opportunity_count)
    return {
        "case_id": str(case.get("case_id") or ""),
        "role": str(case.get("role") or ""),
        "required_event_bias": _float(case.get("required_event_bias")),
        "classes": list(case.get("classes") or ()),
        "trace_summary_path": trace_summary_path.as_posix(),
        "full_trace": full_trace,
        "legal": legal,
        "rollout_completed": bool(rollout.get("completed")),
        "dead_end": bool(rollout.get("dead_end")),
        "max_tokens_exceeded": bool(rollout.get("max_tokens_exceeded")),
        "step_count": step_count,
        "example_count": len(examples),
        "generated_timepoint_count": len(generated_times),
        "second_window_generated_timepoint_count": len(second_window_generated_times),
        "second_window_step_count": second_steps,
        "second_window_event_argmax_count": second_event_argmax,
        "second_window_event_argmax_share": _safe_ratio(second_event_argmax, second_steps),
        "second_window_event_rank_near_count": second_event_rank_near,
        "second_window_event_margin_near_count": second_event_margin_near,
        "second_window_opportunity_count": second_opportunities,
        "second_window_opportunity_share": _safe_ratio(second_opportunities, second_steps),
        "sparse_second_window_step_count": sparse_second_steps,
        "sparse_second_window_step_share": _safe_ratio(sparse_second_steps, second_steps),
        "sparse_context_opportunity_count": sparse_context_opportunities,
        "sparse_context_opportunity_share": _safe_ratio(sparse_context_opportunities, second_steps),
        "dense_context_opportunity_count": dense_context_opportunities,
        "median_opportunity_prev_density": _median_or_none(opportunity_density_values),
        "longest_opportunity_run": longest_opportunity_run,
        "longest_sparse_opportunity_run": longest_sparse_opportunity_run,
        "positive_density_signal": positive,
        "sparse_opportunity_examples": sparse_opportunity_examples,
    }


def summarize_density_diagnostic(
    *,
    cases: Sequence[Mapping[str, Any]],
    trace_oracle_summary_path: Path,
    elapsed_s: float,
    density_lookback_ms: int,
    sparse_density_max_events: int,
    positive_opportunity_count: int,
    positive_starved_case_count: int,
) -> dict[str, Any]:
    low_bias_starved = [case for case in cases if case.get("role") == "low_bias_starved"]
    controls = [case for case in cases if case.get("role") != "low_bias_starved"]
    positive_starved = [case for case in low_bias_starved if bool(case.get("positive_density_signal"))]
    incomplete = [case for case in cases if not bool(case.get("full_trace"))]
    illegal = [case for case in cases if not bool(case.get("legal"))]
    control_conflicts = [
        case for case in controls if (_int(case.get("sparse_context_opportunity_count")) or 0) >= positive_opportunity_count
    ]
    decision = _decision(
        low_bias_count=len(low_bias_starved),
        positive_low_bias_count=len(positive_starved),
        control_conflict_count=len(control_conflicts),
        incomplete_count=len(incomplete),
        illegal_count=len(illegal),
        positive_starved_case_count=positive_starved_case_count,
    )
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar v3 CE density-aware trace diagnostic",
        "elapsed_s": float(elapsed_s),
        "inputs": {
            "trace_oracle_summary": trace_oracle_summary_path.as_posix(),
        },
        "thresholds": {
            "second_window_start_ms": SECOND_WINDOW_START_MS,
            "opportunity_max_rank": OPPORTUNITY_MAX_RANK,
            "opportunity_min_margin": OPPORTUNITY_MIN_MARGIN,
            "density_lookback_ms": int(density_lookback_ms),
            "sparse_density_max_events": int(sparse_density_max_events),
            "positive_opportunity_count": int(positive_opportunity_count),
            "positive_starved_case_count": int(positive_starved_case_count),
        },
        "aggregate": {
            "case_count": len(cases),
            "low_bias_starved_case_count": len(low_bias_starved),
            "positive_low_bias_starved_case_count": len(positive_starved),
            "control_case_count": len(controls),
            "control_conflict_count": len(control_conflicts),
            "incomplete_trace_count": len(incomplete),
            "illegal_case_count": len(illegal),
        },
        "case_results": list(cases),
        "decision": decision,
    }


def write_summary_json(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    decision = _mapping(summary.get("decision"))
    aggregate = _mapping(summary.get("aggregate"))
    thresholds = _mapping(summary.get("thresholds"))
    lines = [
        "# Target Grammar v3 CE Density-Aware Trace Diagnostic Result Report",
        "",
        "## Scope",
        "",
        "This artifact-only diagnostic reuses the completed CE selective trace-oracle summaries. It does not retrain, change grammar, change tokenizer, or change decode behavior.",
        "",
        "## Result",
        "",
        f"Decision: `{decision.get('route')}`.",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Recommended next step: {decision.get('next_step')}",
        f"- Positive CE-starved cases: `{aggregate.get('positive_low_bias_starved_case_count')}` / `{aggregate.get('low_bias_starved_case_count')}`",
        f"- Control conflicts: `{aggregate.get('control_conflict_count')}`",
        f"- Incomplete traces: `{aggregate.get('incomplete_trace_count')}`",
        f"- Illegal traces: `{aggregate.get('illegal_case_count')}`",
        f"- Sparse density threshold: `<= {thresholds.get('sparse_density_max_events')}` generated events in previous `{thresholds.get('density_lookback_ms')}` ms",
        "",
        "## Case Table",
        "",
        "| Role | Case | sparse opp | total opp | sparse steps | second steps | event argmax share | second generated events | longest sparse run | positive |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for case in summary.get("case_results", ()):
        row = _mapping(case)
        lines.append(
            "| {role} | `{case_id}` | {sparse_opp} | {total_opp} | {sparse_steps} | {second_steps} | {event_share} | {second_events} | {run} | {positive} |".format(
                role=row.get("role"),
                case_id=row.get("case_id"),
                sparse_opp=row.get("sparse_context_opportunity_count"),
                total_opp=row.get("second_window_opportunity_count"),
                sparse_steps=row.get("sparse_second_window_step_count"),
                second_steps=row.get("second_window_step_count"),
                event_share=_fmt(row.get("second_window_event_argmax_share")),
                second_events=row.get("second_window_generated_timepoint_count"),
                run=row.get("longest_sparse_opportunity_run"),
                positive=row.get("positive_density_signal"),
            )
        )
    lines.extend(
        [
            "",
            "## Sparse Opportunity Examples",
            "",
        ]
    )
    for case in summary.get("case_results", ()):
        row = _mapping(case)
        examples = row.get("sparse_opportunity_examples")
        if not isinstance(examples, list) or not examples:
            continue
        lines.append(f"### `{row.get('case_id')}`")
        for example in examples[:6]:
            example_map = _mapping(example)
            lines.append(
                "- step={step}, ms={ms}, density={density}, argmax={argmax}, best_event={event}, rank={rank}, margin={margin}".format(
                    step=example_map.get("step"),
                    ms=example_map.get("current_ms"),
                    density=example_map.get("generated_prev_lookback_count"),
                    argmax=example_map.get("argmax_token"),
                    event=example_map.get("best_event_token"),
                    rank=_fmt(example_map.get("best_event_rank")),
                    margin=_fmt(example_map.get("best_event_margin_vs_argmax")),
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


def _decision(
    *,
    low_bias_count: int,
    positive_low_bias_count: int,
    control_conflict_count: int,
    incomplete_count: int,
    illegal_count: int,
    positive_starved_case_count: int,
) -> dict[str, Any]:
    if incomplete_count:
        return {
            "route": "MUTATE_TRACE_CAPTURE",
            "reason": f"{incomplete_count} traces are incomplete",
            "interpretation": "The density-aware diagnostic cannot judge selector viability without complete per-step traces.",
            "next_step": "Regenerate traces with enough logit examples before changing decode behavior.",
        }
    if illegal_count:
        return {
            "route": "MUTATE_RUNTIME_BASELINE",
            "reason": f"{illegal_count} traces are illegal under unchanged runtime",
            "interpretation": "The unchanged runtime baseline is not stable enough for density-aware selector analysis.",
            "next_step": "Repair or explain the baseline rerun before selector work.",
        }
    if positive_low_bias_count >= int(positive_starved_case_count) and control_conflict_count == 0:
        return {
            "route": "TEST_DENSITY_AWARE_SELECTOR_OR_OBJECTIVE",
            "reason": f"{positive_low_bias_count}/{low_bias_count} CE-starved cases have sparse-context opportunities and controls do not conflict",
            "interpretation": "The residual CE signal is broad and separable enough to justify one bounded density-aware mutation.",
            "next_step": "Create a bounded density-aware selector or event-objective Experiment Card with overproduction guards.",
        }
    if positive_low_bias_count > 0:
        return {
            "route": "MUTATE_TO_TARGET_GRAMMAR_OR_TRAINING_REPAIR",
            "reason": f"only {positive_low_bias_count}/{low_bias_count} CE-starved cases have sparse-context opportunity coverage",
            "interpretation": "Density context explains a narrow subset, but not enough of the remaining CE-starved failures to justify another simple selector.",
            "next_step": "Pivot to target-grammar/training repair, or add richer control/audio trace fields only if they define a concrete guard.",
        }
    return {
        "route": "KILL_DENSITY_SELECTOR_ROUTE",
        "reason": "no CE-starved cases have enough sparse-context second-window opportunities",
        "interpretation": "The density-aware selector route lacks local trace evidence.",
        "next_step": "Pivot to target-grammar or training-side repair.",
    }


def _count_prior_events(generated_times: Sequence[int], *, current_ms: int, lookback_ms: int) -> int:
    start_ms = int(current_ms) - int(lookback_ms)
    return sum(1 for value in generated_times if start_ms <= value < int(current_ms))


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
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _safe_ratio(numerator: int | float | None, denominator: int | float | None) -> float:
    if numerator is None or denominator in (None, 0):
        return 0.0
    return float(numerator) / float(denominator)


def _median_or_none(values: Sequence[int]) -> float | None:
    if not values:
        return None
    return float(median(values))


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.6f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the v3 CE density-aware trace diagnostic.")
    parser.add_argument("--trace-oracle-summary", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--density-lookback-ms", type=int, default=DENSITY_LOOKBACK_MS)
    parser.add_argument("--sparse-density-max-events", type=int, default=SPARSE_DENSITY_MAX_EVENTS)
    parser.add_argument("--positive-opportunity-count", type=int, default=POSITIVE_OPPORTUNITY_COUNT)
    parser.add_argument("--positive-starved-case-count", type=int, default=POSITIVE_STARVED_CASE_COUNT)
    args = parser.parse_args(argv)
    summary = run_ce_density_trace_diagnostic(
        trace_oracle_summary_path=Path(args.trace_oracle_summary),
        summary_output_path=Path(args.summary_output),
        report_output_path=Path(args.report_output),
        density_lookback_ms=int(args.density_lookback_ms),
        sparse_density_max_events=int(args.sparse_density_max_events),
        positive_opportunity_count=int(args.positive_opportunity_count),
        positive_starved_case_count=int(args.positive_starved_case_count),
    )
    print(
        "mapper_v3_ce_density_trace_diagnostic_done "
        f"route={summary['decision']['route']} "
        f"positive={summary['aggregate']['positive_low_bias_starved_case_count']}/"
        f"{summary['aggregate']['low_bias_starved_case_count']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
