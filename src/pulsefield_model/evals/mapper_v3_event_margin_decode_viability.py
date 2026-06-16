from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


SUMMARY_SCHEMA_VERSION = 1
STARVED_SHARE_THRESHOLD = 0.10
REFERENCE_SECOND_WINDOW_SUPPORT_THRESHOLD = 0.25
RIGID_DOMINANT_SPACING_RATIO_THRESHOLD = 0.90
UNDERGENERATION_RATIO_THRESHOLD = 0.70
OVERGENERATION_RATIO_THRESHOLD = 1.25
SMALL_EVENT_BIAS = 2.0
MODERATE_EVENT_BIAS = 4.0
LARGE_EVENT_BIAS = 6.0
EXTREME_EVENT_BIAS = 8.0


def parse_named_summary(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"summary must be formatted as name=path, got {value!r}")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"summary name cannot be empty: {value!r}")
    return name, Path(path)


def run_event_margin_decode_viability(
    *,
    summaries: Mapping[str, Path],
    baseline_name: str | None = None,
) -> dict[str, Any]:
    start = time.monotonic()
    if not summaries:
        raise ValueError("at least one summary is required")
    resolved_baseline = next(iter(summaries)) if baseline_name is None else str(baseline_name)
    if resolved_baseline not in summaries:
        raise ValueError(f"baseline {resolved_baseline!r} is not in summaries: {sorted(summaries)}")

    variants: dict[str, dict[str, Any]] = {}
    for name, path in summaries.items():
        loaded = _load_summary(path)
        runs = loaded.get("runs")
        if not isinstance(runs, list) or not runs:
            raise ValueError(f"summary {path} does not contain a non-empty runs list")
        classified = [_classify_run(run) for run in runs]
        variants[name] = _variant_report(name=name, path=path, summary=loaded, runs=classified)

    baseline = variants[resolved_baseline]
    cross_variant_evidence = _cross_variant_evidence(variants, baseline_name=resolved_baseline)
    decision = _decision(baseline, cross_variant_evidence=cross_variant_evidence)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar v3 event-margin decode viability",
        "elapsed_s": time.monotonic() - start,
        "inputs": {name: path.as_posix() for name, path in summaries.items()},
        "baseline": resolved_baseline,
        "thresholds": {
            "starved_second_window_share": STARVED_SHARE_THRESHOLD,
            "rigid_dominant_spacing_ratio": RIGID_DOMINANT_SPACING_RATIO_THRESHOLD,
            "small_event_bias": SMALL_EVENT_BIAS,
            "moderate_event_bias": MODERATE_EVENT_BIAS,
            "large_event_bias": LARGE_EVENT_BIAS,
            "extreme_event_bias": EXTREME_EVENT_BIAS,
        },
        "variants": variants,
        "cross_variant_evidence": cross_variant_evidence,
        "baseline_snapshot": {
            "name": resolved_baseline,
            "case_count": baseline["case_count"],
            "failure_counts": baseline["failure_counts"],
            "required_bias": baseline["required_bias"],
            "class_bias": baseline["class_bias"],
        },
        "decision": decision,
    }


def write_summary_json(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    variants = summary["variants"]
    baseline = summary["variants"][summary["baseline"]]
    decision = summary["decision"]
    lines = [
        "# Target Grammar v3 Event-Margin Decode Viability Result Report",
        "",
        "## Scope",
        "",
        "This artifact-only diagnostic checks whether existing v3 rollout logit margins support a simple global event-token decode bonus. It does not retrain, change grammar, change tokenizer, or change runtime defaults.",
        "",
        "## Result",
        "",
        f"Decision: `{decision['route']}`.",
        "",
        f"- Reason: {decision['reason']}",
        f"- Baseline: `{summary['baseline']}`",
        f"- Recommended next step: {decision['next_step']}",
        "",
        "## Variant Table",
        "",
        "| Variant | cases | starved | rigid | undergen | overgen | median required bias | starved >4 | starved >6 | pass-like <=4 | event top-k minus top-1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, report in variants.items():
        failure_counts = report["failure_counts"]
        required_bias = report["required_bias"]
        class_bias = report["class_bias"]
        event_rank = report["event_rank"]
        lines.append(
            "| {name} | {cases} | {starved} | {rigid} | {under} | {over} | {median_bias} | {starved_gt4} | {starved_gt6} | {pass_le4} | {gap} |".format(
                name=name,
                cases=report["case_count"],
                starved=failure_counts["starved"],
                rigid=failure_counts["rigid"],
                under=failure_counts["undergenerated"],
                over=failure_counts["overgenerated"],
                median_bias=_fmt(required_bias.get("median")),
                starved_gt4=class_bias["starved"]["gt_4"],
                starved_gt6=class_bias["starved"]["gt_6"],
                pass_le4=class_bias["pass_like"]["le_4"],
                gap=_fmt(event_rank.get("mean_event_topk_minus_top1_ratio")),
            )
        )
    lines.extend(
        [
            "",
            "## Cross-Variant Sanity Check",
            "",
        ]
    )
    cross_variant_evidence = summary.get("cross_variant_evidence") or {}
    low_margin_worse = cross_variant_evidence.get("lower_margin_worse_starvation") or []
    if low_margin_worse:
        for row in low_margin_worse:
            lines.append(
                "- `{name}` has lower median required bias ({bias}) than baseline but worse starvation ({starved} vs {baseline_starved}).".format(
                    name=row["name"],
                    bias=_fmt(row.get("median_required_bias")),
                    starved=row.get("starved"),
                    baseline_starved=row.get("baseline_starved"),
                )
            )
    else:
        lines.append("- No compared variant has both lower median required bias and worse starvation than baseline.")
    lines.extend(
        [
            "",
            "## Baseline Failure-Class Bias",
            "",
            "| Class | cases | median required bias | <=2 | <=4 | <=6 | >6 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for class_name in ("starved", "rigid", "undergenerated", "overgenerated", "pass_like", "failed"):
        row = baseline["class_bias"][class_name]
        lines.append(
            "| {name} | {count} | {median} | {le2} | {le4} | {le6} | {gt6} |".format(
                name=class_name,
                count=row["count"],
                median=_fmt(row.get("median")),
                le2=row["le_2"],
                le4=row["le_4"],
                le6=row["le_6"],
                gt6=row["gt_6"],
            )
        )
    lines.extend(
        [
            "",
            "## Highest Required-Bias Failed Cases",
            "",
        ]
    )
    for row in baseline["worst_failed_cases"][:12]:
        lines.append(
            "- `{case_id}`: required_bias={bias}, margin={margin}, rank={rank}, classes={classes}, event_ratio={event_ratio}, second_share={second_share}, rigid={rigid}".format(
                case_id=row["case_id"],
                bias=_fmt(row.get("required_event_bias")),
                margin=_fmt(row.get("best_event_margin_median")),
                rank=_fmt(row.get("best_event_rank_median")),
                classes=", ".join(row.get("classes", ())),
                event_ratio=_fmt(row.get("event_count_ratio")),
                second_share=_fmt(row.get("second_window_event_share")),
                rigid=_fmt(row.get("dominant_spacing_ratio")),
            )
        )
    lines.extend(
        [
            "",
            "## Low-Bias Starved Cases",
            "",
        ]
    )
    low_bias_starved = [
        row
        for row in baseline["case_rows"]
        if row["starved"] and row.get("required_event_bias") is not None and row["required_event_bias"] <= MODERATE_EVENT_BIAS
    ]
    for row in low_bias_starved[:12]:
        lines.append(
            "- `{case_id}`: required_bias={bias}, margin={margin}, rank={rank}, event_ratio={event_ratio}, second_share={second_share}".format(
                case_id=row["case_id"],
                bias=_fmt(row.get("required_event_bias")),
                margin=_fmt(row.get("best_event_margin_median")),
                rank=_fmt(row.get("best_event_rank_median")),
                event_ratio=_fmt(row.get("event_count_ratio")),
                second_share=_fmt(row.get("second_window_event_share")),
            )
        )
    if not low_bias_starved:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(decision["interpretation"]),
            "",
            "## Next Step",
            "",
            str(decision["next_step"]),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_summary(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid summary JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"summary must be a JSON object: {path}")
    return payload


def _classify_run(run: Mapping[str, Any]) -> dict[str, Any]:
    case_id = str(run.get("case_id") or run.get("beatmap_path") or "")
    if not case_id:
        raise ValueError(f"run row is missing case_id: {run}")
    margin = _float(run.get("best_event_margin_median"))
    required_bias = None if margin is None else max(0.0, -float(margin))
    second_share = _float(run.get("second_window_event_share"))
    reference_second_share = _float(run.get("reference_second_window_event_share"))
    reference_support = (
        True
        if reference_second_share is None
        else reference_second_share >= REFERENCE_SECOND_WINDOW_SUPPORT_THRESHOLD
    )
    starved = bool(reference_support and second_share is not None and second_share <= STARVED_SHARE_THRESHOLD)
    dominant_spacing_ratio = _float(run.get("dominant_spacing_ratio")) or 0.0
    event_ratio = _float(run.get("event_count_ratio"))
    dead_end = bool(run.get("dead_end"))
    max_token = bool(run.get("max_tokens_exceeded"))
    undergenerated = event_ratio is not None and event_ratio < UNDERGENERATION_RATIO_THRESHOLD
    overgenerated = event_ratio is not None and event_ratio > OVERGENERATION_RATIO_THRESHOLD
    rigid = dominant_spacing_ratio >= RIGID_DOMINANT_SPACING_RATIO_THRESHOLD
    failed = bool(starved or rigid or undergenerated or overgenerated or dead_end or max_token)
    pass_like = not failed
    logit_steps = _int(run.get("logit_step_count")) or _int(run.get("token_count")) or 0
    event_top1 = _int(run.get("event_top1_step_count")) or 0
    event_topk = _int(run.get("event_topk_step_count")) or 0
    event_valid = _int(run.get("event_valid_step_count")) or 0
    classes = []
    if starved:
        classes.append("starved")
    if rigid:
        classes.append("rigid")
    if undergenerated:
        classes.append("undergenerated")
    if overgenerated:
        classes.append("overgenerated")
    if dead_end:
        classes.append("dead_end")
    if max_token:
        classes.append("max_token")
    if pass_like:
        classes.append("pass_like")
    return {
        "case_id": case_id,
        "case_index": _int(run.get("case_index")),
        "best_event_margin_median": margin,
        "best_event_rank_median": _float(run.get("best_event_rank_median")),
        "required_event_bias": required_bias,
        "required_event_bias_bucket": _bias_bucket(required_bias),
        "starved": starved,
        "rigid": rigid,
        "undergenerated": undergenerated,
        "overgenerated": overgenerated,
        "dead_end": dead_end,
        "max_token": max_token,
        "failed": failed,
        "pass_like": pass_like,
        "classes": classes,
        "event_count_ratio": event_ratio,
        "second_window_event_share": second_share,
        "reference_second_window_event_share": reference_second_share,
        "dominant_spacing_ratio": dominant_spacing_ratio,
        "logit_step_count": logit_steps,
        "event_valid_step_count": event_valid,
        "event_top1_step_count": event_top1,
        "event_topk_step_count": event_topk,
        "event_valid_step_ratio": _safe_ratio(event_valid, logit_steps),
        "event_top1_step_ratio": _safe_ratio(event_top1, logit_steps),
        "event_topk_step_ratio": _safe_ratio(event_topk, logit_steps),
        "event_topk_minus_top1_ratio": _safe_ratio(event_topk - event_top1, logit_steps),
    }


def _variant_report(
    *,
    name: str,
    path: Path,
    summary: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    del name
    margin_missing = [run["case_id"] for run in runs if run.get("required_event_bias") is None]
    if len(margin_missing) == len(runs):
        raise ValueError(f"summary {path} has no best_event_margin_median values")
    counts = {
        "starved": sum(1 for run in runs if run["starved"]),
        "rigid": sum(1 for run in runs if run["rigid"]),
        "undergenerated": sum(1 for run in runs if run["undergenerated"]),
        "overgenerated": sum(1 for run in runs if run["overgenerated"]),
        "dead_end": sum(1 for run in runs if run["dead_end"]),
        "max_token": sum(1 for run in runs if run["max_token"]),
        "failed": sum(1 for run in runs if run["failed"]),
        "pass_like": sum(1 for run in runs if run["pass_like"]),
    }
    failed_rows = [run for run in runs if run["failed"] and run.get("required_event_bias") is not None]
    worst_failed = sorted(
        failed_rows,
        key=lambda run: (-float(run["required_event_bias"]), run["case_id"]),
    )
    return {
        "path": path.as_posix(),
        "experiment": summary.get("experiment"),
        "aggregate": dict(summary.get("aggregate") or {}),
        "case_count": len(runs),
        "margin_present_count": len(runs) - len(margin_missing),
        "margin_missing_case_ids": margin_missing,
        "failure_counts": counts,
        "required_bias": _bias_distribution(runs),
        "class_bias": {
            class_name: _bias_distribution([run for run in runs if run[class_name]])
            for class_name in ("starved", "rigid", "undergenerated", "overgenerated", "failed", "pass_like")
        },
        "event_rank": {
            "median_best_event_rank": _median(run.get("best_event_rank_median") for run in runs),
            "median_best_event_margin": _median(run.get("best_event_margin_median") for run in runs),
            "mean_event_valid_step_ratio": _mean(run.get("event_valid_step_ratio") for run in runs),
            "mean_event_top1_step_ratio": _mean(run.get("event_top1_step_ratio") for run in runs),
            "mean_event_topk_step_ratio": _mean(run.get("event_topk_step_ratio") for run in runs),
            "mean_event_topk_minus_top1_ratio": _mean(run.get("event_topk_minus_top1_ratio") for run in runs),
        },
        "worst_failed_cases": list(worst_failed[:16]),
        "case_rows": list(runs),
    }


def _bias_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [float(run["required_event_bias"]) for run in rows if run.get("required_event_bias") is not None]
    return {
        "count": len(values),
        "mean": _mean(values),
        "median": _median(values),
        "max": max(values) if values else None,
        "le_2": sum(1 for value in values if value <= SMALL_EVENT_BIAS),
        "le_4": sum(1 for value in values if value <= MODERATE_EVENT_BIAS),
        "le_6": sum(1 for value in values if value <= LARGE_EVENT_BIAS),
        "le_8": sum(1 for value in values if value <= EXTREME_EVENT_BIAS),
        "gt_2": sum(1 for value in values if value > SMALL_EVENT_BIAS),
        "gt_4": sum(1 for value in values if value > MODERATE_EVENT_BIAS),
        "gt_6": sum(1 for value in values if value > LARGE_EVENT_BIAS),
        "gt_8": sum(1 for value in values if value > EXTREME_EVENT_BIAS),
    }


def _cross_variant_evidence(
    variants: Mapping[str, Mapping[str, Any]],
    *,
    baseline_name: str,
) -> dict[str, Any]:
    baseline = variants[baseline_name]
    baseline_median = baseline["required_bias"].get("median")
    baseline_starved = int(baseline["failure_counts"]["starved"])
    lower_margin_worse_starvation: list[dict[str, Any]] = []
    if baseline_median is not None:
        for name, report in variants.items():
            if name == baseline_name:
                continue
            median_required_bias = report["required_bias"].get("median")
            starved = int(report["failure_counts"]["starved"])
            if median_required_bias is None:
                continue
            if float(median_required_bias) < float(baseline_median) and starved > baseline_starved:
                lower_margin_worse_starvation.append(
                    {
                        "name": name,
                        "median_required_bias": float(median_required_bias),
                        "starved": starved,
                        "baseline_median_required_bias": float(baseline_median),
                        "baseline_starved": baseline_starved,
                    }
                )
    return {
        "baseline": baseline_name,
        "lower_margin_worse_starvation": lower_margin_worse_starvation,
    }


def _decision(
    baseline: Mapping[str, Any],
    *,
    cross_variant_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    class_bias = baseline["class_bias"]
    starved = class_bias["starved"]
    rigid = class_bias["rigid"]
    failed = class_bias["failed"]
    pass_like = class_bias["pass_like"]
    starved_count = int(starved["count"])
    rigid_count = int(rigid["count"])
    failed_count = int(failed["count"])
    if failed_count == 0:
        return {
            "route": "KILL_DECODE_CALIBRATION_NOT_NEEDED",
            "reason": "baseline has no classified failed cases",
            "interpretation": "The current summary does not show a failure class that needs decode calibration.",
            "next_step": "Move to a broader replacement-readiness audit rather than a decode-bias experiment.",
        }
    starved_gt4_share = _safe_ratio(starved["gt_4"], starved_count) or 0.0
    starved_gt6_share = _safe_ratio(starved["gt_6"], starved_count) or 0.0
    rigid_gt4_share = _safe_ratio(rigid["gt_4"], rigid_count) or 0.0
    pass_like_le4 = int(pass_like["le_4"])
    low_bias_starved = int(starved["le_4"])
    if starved_count and starved_gt4_share >= 0.50:
        route = "KILL_GLOBAL_EVENT_BONUS"
        reason = (
            f"{starved['gt_4']}/{starved_count} starved cases need median event bias > {MODERATE_EVENT_BIAS:.1f}"
        )
        interpretation = (
            "Event tokens are often valid, but the median logit margin is too negative for a small global event bonus "
            "to be a safe next step. A global bonus large enough to move the hard starved cases would likely disturb "
            "already-passable or overgenerated cases."
        )
        if cross_variant_evidence.get("lower_margin_worse_starvation"):
            interpretation += (
                " Cross-variant evidence strengthens this: lower median event-margin pressure did not reliably reduce "
                "starvation in the compared v3 variants."
            )
        next_step = (
            "Do not run another global decode-bonus sweep. Use selective per-step trace/oracle instrumentation on the "
            "few low-bias starved cases, or pivot to v2.1 grammar improvement if the owner wants to stop v3 local probes."
        )
    elif starved_count and low_bias_starved >= max(2, math.ceil(0.50 * starved_count)) and pass_like_le4 == 0:
        route = "TEST_SIMPLE_EVENT_BONUS"
        reason = (
            f"{low_bias_starved}/{starved_count} starved cases need median event bias <= {MODERATE_EVENT_BIAS:.1f}"
        )
        interpretation = (
            "The margin evidence supports a small simple event-bonus smoke because most starved cases are close enough "
            "to the event decision boundary and pass-like cases are not obviously exposed."
        )
        next_step = "Run a bounded simple event-bonus rollout on high-leverage cases before full32."
    elif starved_count and low_bias_starved > 0:
        route = "TEST_SELECTIVE_TRACE_ORACLE"
        reason = (
            f"{low_bias_starved}/{starved_count} starved cases are low-bias, but the class is mixed"
        )
        interpretation = (
            "A global event bonus is not well supported, but some starved cases may be close enough for a selective "
            "state-aware selector. The current summaries are too coarse to define that selector safely."
        )
        next_step = "Add per-step trace/oracle instrumentation for the low-bias starved cases."
    elif rigid_count and rigid_gt4_share >= 0.50:
        route = "MUTATE_TO_V2_1_GRAMMAR"
        reason = (
            f"{rigid['gt_4']}/{rigid_count} rigid cases need median event bias > {MODERATE_EVENT_BIAS:.1f}"
        )
        interpretation = (
            "The remaining failures look too far from the event decision boundary for a simple decode calibration. "
            "This supports pausing v3 local decode probes and using the evidence to improve the v2.1 grammar path."
        )
        next_step = "Create a v2.1 grammar improvement card or a richer v3 trace card before any further training."
    else:
        route = "TEST_SELECTIVE_TRACE_ORACLE"
        reason = "event-margin evidence is mixed and not enough for a global bonus"
        interpretation = (
            "The current summaries do not support a simple global bonus, but they also do not fully rule out a selective "
            "per-step selector."
        )
        next_step = "Add trace-level evidence on the shared worst cases."
    return {
        "route": route,
        "reason": reason,
        "interpretation": interpretation,
        "next_step": next_step,
        "starved_gt4_share": starved_gt4_share,
        "starved_gt6_share": starved_gt6_share,
        "rigid_gt4_share": rigid_gt4_share,
    }


def _bias_bucket(value: float | None) -> str:
    if value is None:
        return "missing"
    if value <= SMALL_EVENT_BIAS:
        return "le_2"
    if value <= MODERATE_EVENT_BIAS:
        return "le_4"
    if value <= LARGE_EVENT_BIAS:
        return "le_6"
    if value <= EXTREME_EVENT_BIAS:
        return "le_8"
    return "gt_8"


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


def _safe_ratio(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _mean(values: Any) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return math.fsum(numeric) / len(numeric)


def _median(values: Any) -> float | None:
    numeric = sorted(float(value) for value in values if value is not None)
    if not numeric:
        return None
    midpoint = len(numeric) // 2
    if len(numeric) % 2:
        return numeric[midpoint]
    return 0.5 * (numeric[midpoint - 1] + numeric[midpoint])


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    return f"{float(value):.6f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit whether v3 event margins support simple decode calibration.")
    parser.add_argument("--summary", action="append", required=True, help="Named summary path as name=path")
    parser.add_argument("--baseline")
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--report-output", required=True)
    args = parser.parse_args(argv)

    summaries = dict(parse_named_summary(value) for value in args.summary)
    summary = run_event_margin_decode_viability(summaries=summaries, baseline_name=args.baseline)
    write_summary_json(summary, Path(args.summary_output))
    write_report(summary, Path(args.report_output))
    print(
        "mapper_v3_event_margin_decode_viability_done "
        f"route={summary['decision']['route']} "
        f"baseline={summary['baseline']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
