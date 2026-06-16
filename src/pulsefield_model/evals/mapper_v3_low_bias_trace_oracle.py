from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from pulsefield_model.evals.mapper_v3_trained_runtime_rollout import run_trained_v3_runtime_rollout_smoke


SUMMARY_SCHEMA_VERSION = 1
LOW_BIAS_THRESHOLD = 2.0
HIGH_BIAS_THRESHOLD = 6.0
SECOND_WINDOW_START_MS = 8000
OPPORTUNITY_MAX_RANK = 5
OPPORTUNITY_MIN_MARGIN = -2.0
POSITIVE_OPPORTUNITY_COUNT = 10


def run_low_bias_trace_oracle(
    *,
    margin_summary_path: Path,
    manifest_path: Path,
    summary_output_path: Path,
    report_output_path: Path,
    work_dir: Path,
    device: str = "auto",
    low_bias_limit: int = 3,
    include_controls: bool = True,
    logit_top_k: int = 5,
    logit_max_examples: int = 2048,
    dry_run: bool = False,
) -> dict[str, Any]:
    start = time.monotonic()
    margin_summary = _load_json_object(margin_summary_path)
    manifest = _load_json_list(manifest_path)
    cases = select_trace_cases(
        margin_summary=margin_summary,
        manifest=manifest,
        low_bias_limit=int(low_bias_limit),
        include_controls=bool(include_controls),
    )
    case_results: list[dict[str, Any]] = []
    for case in cases:
        if bool(dry_run):
            case_results.append({"case": case, "trace": None, "analysis": {"dry_run": True}})
            continue
        case_results.append(
            run_trace_case(
                case=case,
                work_dir=work_dir,
                device=device,
                logit_top_k=int(logit_top_k),
                logit_max_examples=int(logit_max_examples),
            )
        )
    summary = summarize_trace_oracle(
        cases=cases,
        case_results=case_results,
        margin_summary_path=margin_summary_path,
        manifest_path=manifest_path,
        work_dir=work_dir,
        elapsed_s=time.monotonic() - start,
        dry_run=bool(dry_run),
    )
    write_summary_json(summary, summary_output_path)
    write_report(summary, report_output_path)
    return summary


def select_trace_cases(
    *,
    margin_summary: Mapping[str, Any],
    manifest: Sequence[Mapping[str, Any]],
    low_bias_limit: int = 3,
    include_controls: bool = True,
) -> list[dict[str, Any]]:
    baseline_name = str(margin_summary.get("baseline") or "")
    variants = margin_summary.get("variants")
    if not baseline_name or not isinstance(variants, Mapping) or baseline_name not in variants:
        raise ValueError("margin summary must contain a baseline variant")
    baseline = variants[baseline_name]
    case_rows = baseline.get("case_rows") if isinstance(baseline, Mapping) else None
    if not isinstance(case_rows, list) or not case_rows:
        raise ValueError("baseline variant must contain case_rows")
    manifest_by_case_id = {str(row.get("case_id")): row for row in manifest if row.get("case_id")}

    def with_manifest(row: Mapping[str, Any], role: str) -> dict[str, Any]:
        case_id = str(row.get("case_id") or "")
        manifest_row = manifest_by_case_id.get(case_id)
        if manifest_row is None:
            raise ValueError(f"case {case_id!r} is missing from manifest")
        return {
            "role": role,
            "case_id": case_id,
            "case_index": _int(row.get("case_index")),
            "required_event_bias": _float(row.get("required_event_bias")),
            "best_event_margin_median": _float(row.get("best_event_margin_median")),
            "best_event_rank_median": _float(row.get("best_event_rank_median")),
            "event_count_ratio": _float(row.get("event_count_ratio")),
            "second_window_event_share": _float(row.get("second_window_event_share")),
            "dominant_spacing_ratio": _float(row.get("dominant_spacing_ratio")),
            "classes": list(row.get("classes") or ()),
            "manifest": dict(manifest_row),
        }

    low_bias_starved = sorted(
        (
            row
            for row in case_rows
            if bool(row.get("starved")) and (_float(row.get("required_event_bias")) or math.inf) <= LOW_BIAS_THRESHOLD
        ),
        key=lambda row: (_float(row.get("required_event_bias")) or math.inf, str(row.get("case_id"))),
    )
    selected = [with_manifest(row, "low_bias_starved") for row in low_bias_starved[: int(low_bias_limit)]]
    if include_controls:
        high_bias_starved = sorted(
            (
                row
                for row in case_rows
                if bool(row.get("starved")) and (_float(row.get("required_event_bias")) or 0.0) >= HIGH_BIAS_THRESHOLD
            ),
            key=lambda row: (-(_float(row.get("required_event_bias")) or 0.0), str(row.get("case_id"))),
        )
        if high_bias_starved:
            selected.append(with_manifest(high_bias_starved[0], "high_bias_starved_control"))
        pass_like = sorted(
            (
                row
                for row in case_rows
                if bool(row.get("pass_like")) and (_float(row.get("second_window_event_share")) or 0.0) >= 0.40
            ),
            key=lambda row: (_float(row.get("required_event_bias")) or math.inf, str(row.get("case_id"))),
        )
        if pass_like:
            selected.append(with_manifest(pass_like[0], "pass_like_control"))
    return selected


def run_trace_case(
    *,
    case: Mapping[str, Any],
    work_dir: Path,
    device: str,
    logit_top_k: int,
    logit_max_examples: int,
) -> dict[str, Any]:
    manifest = _mapping(case.get("manifest"))
    summary_path = Path(str(manifest.get("summary_path")))
    previous_summary = _load_json_object(summary_path)
    config = _mapping(previous_summary.get("config"))
    case_id = str(case["case_id"])
    output_dir = work_dir / "rollouts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_summary = output_dir / f"{case_id}_trace_summary.json"
    output_report = output_dir / f"{case_id}_trace_report.md"
    trace_summary = run_trained_v3_runtime_rollout_smoke(
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
        logit_top_k=int(logit_top_k),
        logit_max_examples=int(logit_max_examples),
        timepoint_preview_limit=int(config.get("timepoint_preview_limit", 1024)),
    )
    analysis = analyze_trace_summary(trace_summary)
    return {
        "case": dict(case),
        "trace_summary_path": output_summary.as_posix(),
        "trace_report_path": output_report.as_posix(),
        "trace": _compact_trace_summary(trace_summary),
        "analysis": analysis,
    }


def analyze_trace_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = _mapping(summary.get("logit_diagnostics"))
    examples = diagnostics.get("examples")
    if not isinstance(examples, list):
        examples = []
    step_count = _int(diagnostics.get("step_count")) or 0
    full_trace = len(examples) >= step_count if step_count else False
    windows: dict[str, dict[str, Any]] = {
        "first": _new_window_counts(),
        "second": _new_window_counts(),
    }
    opportunities: list[dict[str, Any]] = []
    first_window_opportunities: list[dict[str, Any]] = []
    second_window_opportunities: list[dict[str, Any]] = []
    for example in examples:
        if not isinstance(example, Mapping):
            continue
        current_ms = _int(example.get("current_ms"))
        if current_ms is None:
            continue
        window = "second" if current_ms >= SECOND_WINDOW_START_MS else "first"
        counts = windows[window]
        counts["step_count"] += 1
        argmax_kind = str(example.get("argmax_kind"))
        if argmax_kind == "event":
            counts["event_argmax_count"] += 1
        best_event_rank = _float(example.get("best_event_rank"))
        best_event_margin = _float(example.get("best_event_margin_vs_argmax"))
        if best_event_rank is not None and best_event_rank <= OPPORTUNITY_MAX_RANK:
            counts["event_rank_near_count"] += 1
        if best_event_margin is not None and best_event_margin >= OPPORTUNITY_MIN_MARGIN:
            counts["event_margin_near_count"] += 1
        is_opportunity = (
            argmax_kind != "event"
            and best_event_rank is not None
            and best_event_rank <= OPPORTUNITY_MAX_RANK
            and best_event_margin is not None
            and best_event_margin >= OPPORTUNITY_MIN_MARGIN
        )
        if is_opportunity:
            row = {
                "step": _int(example.get("step")),
                "current_ms": current_ms,
                "argmax_token": example.get("argmax_token"),
                "best_event_token": example.get("best_event_token"),
                "best_event_rank": best_event_rank,
                "best_event_margin_vs_argmax": best_event_margin,
            }
            opportunities.append(row)
            if window == "second":
                second_window_opportunities.append(row)
            else:
                first_window_opportunities.append(row)
            counts["opportunity_count"] += 1
    for counts in windows.values():
        counts["opportunity_share"] = _safe_ratio(counts["opportunity_count"], counts["step_count"])
        counts["event_argmax_share"] = _safe_ratio(counts["event_argmax_count"], counts["step_count"])
    rollout = _mapping(summary.get("rollout"))
    return {
        "step_count": step_count,
        "example_count": len(examples),
        "full_trace": full_trace,
        "windows": windows,
        "second_window_opportunity_count": windows["second"]["opportunity_count"],
        "first_window_opportunity_count": windows["first"]["opportunity_count"],
        "positive_signal": windows["second"]["opportunity_count"] >= POSITIVE_OPPORTUNITY_COUNT,
        "rollout_completed": bool(rollout.get("completed")),
        "dead_end": bool(rollout.get("dead_end")),
        "max_tokens_exceeded": bool(rollout.get("max_tokens_exceeded")),
        "timepoint_count": _int(rollout.get("timepoint_count")),
        "opportunity_examples": opportunities[:24],
        "first_window_opportunity_examples": first_window_opportunities[:12],
        "second_window_opportunity_examples": second_window_opportunities[:12],
    }


def summarize_trace_oracle(
    *,
    cases: Sequence[Mapping[str, Any]],
    case_results: Sequence[Mapping[str, Any]],
    margin_summary_path: Path,
    manifest_path: Path,
    work_dir: Path,
    elapsed_s: float,
    dry_run: bool,
) -> dict[str, Any]:
    low_bias_results = [
        result
        for result in case_results
        if _mapping(result.get("case")).get("role") == "low_bias_starved"
    ]
    positive_low_bias = [
        result for result in low_bias_results if bool(_mapping(result.get("analysis")).get("positive_signal"))
    ]
    incomplete = [
        result
        for result in case_results
        if not bool(_mapping(result.get("analysis")).get("full_trace", bool(dry_run))) and not bool(dry_run)
    ]
    illegal = [
        result
        for result in case_results
        if bool(_mapping(result.get("analysis")).get("dead_end"))
        or bool(_mapping(result.get("analysis")).get("max_tokens_exceeded"))
    ]
    decision = _decision(
        low_bias_count=len(low_bias_results),
        positive_low_bias_count=len(positive_low_bias),
        incomplete_count=len(incomplete),
        illegal_count=len(illegal),
        dry_run=bool(dry_run),
    )
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar v3 low-bias trace oracle",
        "elapsed_s": float(elapsed_s),
        "inputs": {
            "margin_summary": margin_summary_path.as_posix(),
            "manifest": manifest_path.as_posix(),
        },
        "work_dir": work_dir.as_posix(),
        "dry_run": bool(dry_run),
        "selection": {
            "case_count": len(cases),
            "low_bias_case_count": len(low_bias_results),
            "case_ids": [str(case["case_id"]) for case in cases],
        },
        "thresholds": {
            "low_bias": LOW_BIAS_THRESHOLD,
            "high_bias": HIGH_BIAS_THRESHOLD,
            "second_window_start_ms": SECOND_WINDOW_START_MS,
            "opportunity_max_rank": OPPORTUNITY_MAX_RANK,
            "opportunity_min_margin": OPPORTUNITY_MIN_MARGIN,
            "positive_opportunity_count": POSITIVE_OPPORTUNITY_COUNT,
        },
        "case_results": list(case_results),
        "aggregate": {
            "positive_low_bias_count": len(positive_low_bias),
            "low_bias_case_count": len(low_bias_results),
            "incomplete_trace_count": len(incomplete),
            "illegal_case_count": len(illegal),
        },
        "decision": decision,
    }


def write_summary_json(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    decision = _mapping(summary.get("decision"))
    aggregate = _mapping(summary.get("aggregate"))
    lines = [
        "# Target Grammar v3 Low-Bias Trace Oracle Result Report",
        "",
        "## Scope",
        "",
        "This runtime-backed diagnostic reruns selected v3 500-step fixed-slice cases with expanded per-step logit examples. It does not change tokenizer, grammar, model weights, training, or runtime decode defaults.",
        "",
        "## Result",
        "",
        f"Decision: `{decision.get('route')}`.",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Recommended next step: {decision.get('next_step')}",
        f"- Positive low-bias cases: `{aggregate.get('positive_low_bias_count')}` / `{aggregate.get('low_bias_case_count')}`",
        f"- Incomplete traces: `{aggregate.get('incomplete_trace_count')}`",
        f"- Illegal cases: `{aggregate.get('illegal_case_count')}`",
        "",
        "## Case Table",
        "",
        "| Role | Case | full trace | second-window opportunities | first-window opportunities | second steps | timepoints | dead end | max token |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for result in summary.get("case_results", ()):
        result_map = _mapping(result)
        case = _mapping(result_map.get("case"))
        analysis = _mapping(result_map.get("analysis"))
        windows = _mapping(analysis.get("windows"))
        second = _mapping(windows.get("second"))
        lines.append(
            "| {role} | `{case_id}` | {full_trace} | {second_opp} | {first_opp} | {second_steps} | {timepoints} | {dead} | {max_token} |".format(
                role=case.get("role"),
                case_id=case.get("case_id"),
                full_trace=analysis.get("full_trace"),
                second_opp=analysis.get("second_window_opportunity_count"),
                first_opp=analysis.get("first_window_opportunity_count"),
                second_steps=second.get("step_count"),
                timepoints=analysis.get("timepoint_count"),
                dead=analysis.get("dead_end"),
                max_token=analysis.get("max_tokens_exceeded"),
            )
        )
    lines.extend(
        [
            "",
            "## Second-Window Opportunity Examples",
            "",
        ]
    )
    for result in summary.get("case_results", ()):
        result_map = _mapping(result)
        case = _mapping(result_map.get("case"))
        analysis = _mapping(result_map.get("analysis"))
        examples = analysis.get("second_window_opportunity_examples")
        if not isinstance(examples, list) or not examples:
            continue
        lines.append(f"### `{case.get('case_id')}`")
        for row in examples[:8]:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                "- step={step}, ms={ms}, argmax={argmax}, best_event={event}, rank={rank}, margin={margin}".format(
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


def _decision(
    *,
    low_bias_count: int,
    positive_low_bias_count: int,
    incomplete_count: int,
    illegal_count: int,
    dry_run: bool,
) -> dict[str, Any]:
    if dry_run:
        return {
            "route": "DRY_RUN",
            "reason": "case selection only",
            "interpretation": "No runtime trace was executed.",
            "next_step": "Run without --dry-run.",
        }
    if incomplete_count:
        return {
            "route": "MUTATE_TRACE_CAPTURE",
            "reason": f"{incomplete_count} traces did not cover all diagnostic steps",
            "interpretation": "The trace capture is insufficient to judge selective decode viability.",
            "next_step": "Increase logit_max_examples or reduce the selected case set.",
        }
    if illegal_count:
        return {
            "route": "MUTATE_RUNTIME_BASELINE",
            "reason": f"{illegal_count} selected cases failed baseline runtime legality",
            "interpretation": "The unchanged baseline rerun is not stable enough for selector analysis.",
            "next_step": "Repair or explain the baseline rerun before selector work.",
        }
    if low_bias_count and positive_low_bias_count >= max(2, math.ceil(0.5 * low_bias_count)):
        return {
            "route": "TEST_SELECTIVE_EVENT_GATE",
            "reason": f"{positive_low_bias_count}/{low_bias_count} low-bias starved cases have second-window opportunities",
            "interpretation": "The low-bias exception is real enough to justify one bounded selective event-gate smoke.",
            "next_step": "Create and run a selective event-gate smoke on these traced cases.",
        }
    if positive_low_bias_count > 0:
        return {
            "route": "TEST_DENSITY_AWARE_TRACE",
            "reason": f"only {positive_low_bias_count}/{low_bias_count} low-bias starved cases have enough second-window opportunities",
            "interpretation": "A simple selective gate is weakly supported at best. More context features are needed before changing decode behavior.",
            "next_step": "Either add density/control-aware trace fields or pivot to v2.1 grammar improvement.",
        }
    return {
        "route": "MUTATE_TO_V2_1_GRAMMAR",
        "reason": "low-bias cases do not expose enough second-window event opportunities",
        "interpretation": "The near-boundary event evidence does not occur where the starvation failure needs it. This kills the simple selective v3 decode route.",
        "next_step": "Pivot to a v2.1 grammar improvement card or broader training-side instrumentation.",
    }


def _new_window_counts() -> dict[str, Any]:
    return {
        "step_count": 0,
        "event_argmax_count": 0,
        "event_rank_near_count": 0,
        "event_margin_near_count": 0,
        "opportunity_count": 0,
        "opportunity_share": 0.0,
        "event_argmax_share": 0.0,
    }


def _compact_trace_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    rollout = _mapping(summary.get("rollout"))
    diagnostics = _mapping(summary.get("logit_diagnostics"))
    return {
        "decision": dict(_mapping(summary.get("decision"))),
        "rollout": {
            "window_count": rollout.get("window_count"),
            "token_count": rollout.get("token_count"),
            "timepoint_count": rollout.get("timepoint_count"),
            "completed": rollout.get("completed"),
            "dead_end": rollout.get("dead_end"),
            "max_tokens_exceeded": rollout.get("max_tokens_exceeded"),
        },
        "logit_diagnostics": {
            "step_count": diagnostics.get("step_count"),
            "event_valid_step_count": diagnostics.get("event_valid_step_count"),
            "event_top1_step_count": diagnostics.get("event_top1_step_count"),
            "event_topk_step_count": diagnostics.get("event_topk_step_count"),
            "example_count": len(diagnostics.get("examples") or ()),
            "best_event_rank": diagnostics.get("best_event_rank"),
            "best_event_margin_vs_argmax": diagnostics.get("best_event_margin_vs_argmax"),
        },
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON list: {path}")
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _int(value: Any) -> int | None:
    if value is None:
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


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    return f"{float(value):.6f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Trace low-bias v3 starved cases for selective decode opportunity.")
    parser.add_argument("--margin-summary", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--work-dir", default="artifacts/tmp/mapper_v3_low_bias_trace_oracle")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--low-bias-limit", type=int, default=3)
    parser.add_argument("--no-controls", action="store_true")
    parser.add_argument("--logit-top-k", type=int, default=5)
    parser.add_argument("--logit-max-examples", type=int, default=2048)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    summary = run_low_bias_trace_oracle(
        margin_summary_path=Path(args.margin_summary),
        manifest_path=Path(args.manifest),
        summary_output_path=Path(args.summary_output),
        report_output_path=Path(args.report_output),
        work_dir=Path(args.work_dir),
        device=str(args.device),
        low_bias_limit=int(args.low_bias_limit),
        include_controls=not bool(args.no_controls),
        logit_top_k=int(args.logit_top_k),
        logit_max_examples=int(args.logit_max_examples),
        dry_run=bool(args.dry_run),
    )
    print(
        "mapper_v3_low_bias_trace_oracle_done "
        f"route={summary['decision']['route']} "
        f"cases={summary['selection']['case_count']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
