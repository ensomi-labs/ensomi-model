from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


SUMMARY_SCHEMA_VERSION = 1
STARVED_SHARE_THRESHOLD = 0.10
REFERENCE_SECOND_WINDOW_SUPPORT_THRESHOLD = 0.25
RIGID_DOMINANT_SPACING_RATIO_THRESHOLD = 0.90
UNDERGENERATION_RATIO_THRESHOLD = 0.70
OVERGENERATION_RATIO_THRESHOLD = 1.25
BOUNDARY_CLAMP_MIN_MS = 7860
BOUNDARY_CLAMP_MAX_MS = 8000
LATE_SPILL_MIN_MS = 15900


def parse_named_summary(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"summary must be formatted as name=path, got {value!r}")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"summary name cannot be empty: {value!r}")
    return name, Path(path)


def run_continuation_cluster_diagnostic(
    *,
    summaries: Mapping[str, Path],
    baseline_name: str,
) -> dict[str, Any]:
    start = time.monotonic()
    if baseline_name not in summaries:
        raise ValueError(f"baseline {baseline_name!r} is not in summaries: {sorted(summaries)}")

    variant_reports: dict[str, dict[str, Any]] = {}
    case_to_variants: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for name, path in summaries.items():
        loaded = _load_summary(path)
        runs = loaded.get("runs")
        if not isinstance(runs, list) or not runs:
            raise ValueError(f"summary {path} does not contain a non-empty runs list")
        classified = [_classify_run(run) for run in runs]
        variant_report = _variant_report(name=name, path=path, summary=loaded, runs=classified)
        variant_reports[name] = variant_report
        for row in classified:
            case_to_variants[row["case_id"]][name] = row

    matched_case_ids = sorted(case_id for case_id, by_variant in case_to_variants.items() if len(by_variant) == len(summaries))
    if not matched_case_ids:
        raise ValueError("no case ids are shared across all variants")
    baseline = variant_reports[baseline_name]
    variant_names = tuple(summaries)
    latest_name = variant_names[-1]
    overlap = _failure_overlap(
        case_to_variants,
        variant_names=variant_names,
        baseline_name=baseline_name,
        latest_name=latest_name,
        matched_case_ids=matched_case_ids,
    )
    decision = _decision(variant_reports, baseline_name=baseline_name, overlap=overlap)
    elapsed_s = time.monotonic() - start

    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar v3 generated-state continuation diagnostic",
        "elapsed_s": elapsed_s,
        "inputs": {name: path.as_posix() for name, path in summaries.items()},
        "baseline": baseline_name,
        "matched_case_count": len(matched_case_ids),
        "matched_case_ids": matched_case_ids,
        "variants": variant_reports,
        "overlap": overlap,
        "baseline_snapshot": {
            "name": baseline_name,
            "failure_counts": baseline["failure_counts"],
            "event_rank": baseline["event_rank"],
        },
        "decision": decision,
    }


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    variants = summary["variants"]
    decision = summary["decision"]
    lines = [
        "# Target Grammar v3 Generated-State Continuation Diagnostic Result Report",
        "",
        "## Scope",
        "",
        "This artifact-only diagnostic compares existing 32-case v3 rollout summaries. It does not retrain, change grammar, or change inference behavior. The goal is to choose the next bounded grammar/decode experiment after C3 mapper-side paths showed diminishing returns.",
        "",
        "## Result",
        "",
        f"Decision: `{decision['route']}`.",
        "",
        f"- Reason: {decision['reason']}",
        f"- Baseline: `{summary['baseline']}`",
        f"- Matched cases: `{summary['matched_case_count']}`",
        f"- Recommended next step: {decision['next_step']}",
        "",
        "## Failure-Class Table",
        "",
        "| Variant | legal | starved | rigid | boundary clamp | late spill | dead end | max token | undergen | overgen | median event ratio | mean F1 | event top-k minus top-1 | event valid ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, report in variants.items():
        aggregate = report["aggregate"]
        counts = report["failure_counts"]
        event_rank = report["event_rank"]
        lines.append(
            "| {name} | {legal} | {starved} | {rigid} | {boundary} | {late} | {dead} | {max_token} | {under} | {over} | {event_ratio} | {f1} | {gap} | {valid} |".format(
                name=name,
                legal=aggregate.get("all_legal"),
                starved=counts["starved"],
                rigid=counts["rigid"],
                boundary=counts["boundary_clamped"],
                late=counts["late_spill"],
                dead=counts["dead_end"],
                max_token=counts["max_token"],
                under=counts["undergenerated"],
                over=counts["overgenerated"],
                event_ratio=_fmt(aggregate.get("median_event_count_ratio")),
                f1=_fmt(aggregate.get("mean_f1_100ms")),
                gap=_fmt(event_rank.get("mean_event_topk_minus_top1_ratio")),
                valid=_fmt(event_rank.get("mean_event_valid_step_ratio")),
            )
        )
    lines.extend(
        [
            "",
            "## Shared Failure Overlap",
            "",
        ]
    )
    overlap = summary["overlap"]
    lines.extend(
        [
            f"- Cases starved in every variant: `{overlap['all_variant_starved_count']}`",
            f"- Cases boundary-clamped in every variant: `{overlap['all_variant_boundary_clamped_count']}`",
            f"- Cases rigid in every variant: `{overlap['all_variant_rigid_count']}`",
            f"- Cases starved in baseline and latest (`{overlap['latest']}`): `{overlap['baseline_and_latest_starved_count']}`",
            f"- Cases boundary-clamped in baseline and latest (`{overlap['latest']}`): `{overlap['baseline_and_latest_boundary_clamped_count']}`",
            "",
            "## Highest-Leverage Cases",
            "",
        ]
    )
    for row in overlap["top_overlap_cases"][:12]:
        lines.append(
            "- `{case_id}`: starved={starved}, boundary={boundary}, rigid={rigid}, variants={variants}".format(
                case_id=row["case_id"],
                starved=row["starved_variants"],
                boundary=row["boundary_clamped_variants"],
                rigid=row["rigid_variants"],
                variants=", ".join(row["present_variants"]),
            )
        )
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


def write_summary_json(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    generated_times = _generated_times(run)
    second_window_times = [time for time in generated_times if time >= 8000]
    pre_boundary_times = [time for time in generated_times if BOUNDARY_CLAMP_MIN_MS <= time <= BOUNDARY_CLAMP_MAX_MS]
    second_share = _float(run.get("second_window_event_share"))
    reference_second_share = _float(run.get("reference_second_window_event_share"))
    reference_support = (
        True
        if reference_second_share is None
        else reference_second_share >= REFERENCE_SECOND_WINDOW_SUPPORT_THRESHOLD
    )
    starved = bool(reference_support and second_share is not None and second_share <= STARVED_SHARE_THRESHOLD)
    late_spill = bool(starved and second_window_times and min(second_window_times) >= LATE_SPILL_MIN_MS)
    boundary_clamped = bool(starved and pre_boundary_times and (not second_window_times or late_spill))
    dominant_spacing_ratio = _float(run.get("dominant_spacing_ratio")) or 0.0
    event_ratio = _float(run.get("event_count_ratio"))
    logit_steps = _int(run.get("logit_step_count")) or _int(run.get("token_count")) or 0
    event_valid = _int(run.get("event_valid_step_count")) or 0
    event_top1 = _int(run.get("event_top1_step_count")) or 0
    event_topk = _int(run.get("event_topk_step_count")) or 0

    return {
        "case_id": case_id,
        "case_index": _int(run.get("case_index")),
        "legal": bool(run.get("legal", not bool(run.get("dead_end")) and not bool(run.get("max_tokens_exceeded")))),
        "completed": bool(run.get("completed")),
        "dead_end": bool(run.get("dead_end")),
        "max_token": bool(run.get("max_tokens_exceeded")),
        "starved": starved,
        "rigid": dominant_spacing_ratio >= RIGID_DOMINANT_SPACING_RATIO_THRESHOLD,
        "boundary_clamped": boundary_clamped,
        "late_spill": late_spill,
        "undergenerated": event_ratio is not None and event_ratio < UNDERGENERATION_RATIO_THRESHOLD,
        "overgenerated": event_ratio is not None and event_ratio > OVERGENERATION_RATIO_THRESHOLD,
        "generated_event_count": _int(run.get("generated_event_count")) or 0,
        "reference_event_count": _int(run.get("reference_event_count")) or 0,
        "event_count_ratio": event_ratio,
        "second_window_event_share": second_share,
        "reference_second_window_event_share": reference_second_share,
        "dominant_spacing_ms": _int(run.get("dominant_spacing_ms")),
        "dominant_spacing_ratio": dominant_spacing_ratio,
        "timing_match_100ms": _float(run.get("timing_match_100ms")),
        "generated_times": generated_times,
        "last_generated_times": [int(value) for value in run.get("last_12_generated_times", ())],
        "first_generated_times": [int(value) for value in run.get("first_12_generated_times", ())],
        "logit_step_count": logit_steps,
        "event_valid_step_count": event_valid,
        "event_top1_step_count": event_top1,
        "event_topk_step_count": event_topk,
        "event_valid_step_ratio": _safe_ratio(event_valid, logit_steps),
        "event_top1_step_ratio": _safe_ratio(event_top1, logit_steps),
        "event_topk_step_ratio": _safe_ratio(event_topk, logit_steps),
        "event_topk_minus_top1_ratio": _safe_ratio(event_topk - event_top1, logit_steps),
        "best_event_rank_median": _float(run.get("best_event_rank_median")),
        "best_event_margin_median": _float(run.get("best_event_margin_median")),
        "summary_path": run.get("summary_path"),
    }


def _generated_times(run: Mapping[str, Any]) -> list[int]:
    values: list[int] = []
    for key in ("first_12_generated_times", "last_12_generated_times"):
        raw = run.get(key)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            values.extend(int(value) for value in raw)
    return sorted(set(values))


def _variant_report(
    *,
    name: str,
    path: Path,
    summary: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    aggregate = dict(summary.get("aggregate") or {})
    counts = {
        "starved": sum(1 for run in runs if run["starved"]),
        "rigid": sum(1 for run in runs if run["rigid"]),
        "boundary_clamped": sum(1 for run in runs if run["boundary_clamped"]),
        "late_spill": sum(1 for run in runs if run["late_spill"]),
        "dead_end": sum(1 for run in runs if run["dead_end"]),
        "max_token": sum(1 for run in runs if run["max_token"]),
        "undergenerated": sum(1 for run in runs if run["undergenerated"]),
        "overgenerated": sum(1 for run in runs if run["overgenerated"]),
    }
    starved_runs = [run for run in runs if run["starved"]]
    boundary_among_starved = _safe_ratio(counts["boundary_clamped"], len(starved_runs))
    return {
        "path": path.as_posix(),
        "experiment": summary.get("experiment"),
        "aggregate": aggregate,
        "failure_counts": counts,
        "event_rank": {
            "mean_event_valid_step_ratio": _mean(run.get("event_valid_step_ratio") for run in runs),
            "mean_event_top1_step_ratio": _mean(run.get("event_top1_step_ratio") for run in runs),
            "mean_event_topk_step_ratio": _mean(run.get("event_topk_step_ratio") for run in runs),
            "mean_event_topk_minus_top1_ratio": _mean(run.get("event_topk_minus_top1_ratio") for run in runs),
            "median_best_event_rank": _median(run.get("best_event_rank_median") for run in runs),
            "median_best_event_margin": _median(run.get("best_event_margin_median") for run in runs),
        },
        "boundary_among_starved": boundary_among_starved,
        "worst_cases": _worst_cases(runs),
        "case_rows": list(runs),
    }


def _worst_cases(runs: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    def project(run: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "case_id": run["case_id"],
            "event_count_ratio": run.get("event_count_ratio"),
            "second_window_event_share": run.get("second_window_event_share"),
            "dominant_spacing_ratio": run.get("dominant_spacing_ratio"),
            "event_topk_minus_top1_ratio": run.get("event_topk_minus_top1_ratio"),
            "last_generated_times": run.get("last_generated_times"),
        }

    return {
        "starved": [project(run) for run in sorted(runs, key=lambda row: row.get("second_window_event_share") or 0.0)[:8]],
        "rigid": [project(run) for run in sorted(runs, key=lambda row: row.get("dominant_spacing_ratio") or 0.0, reverse=True)[:8]],
        "event_rank_gap": [
            project(run)
            for run in sorted(runs, key=lambda row: row.get("event_topk_minus_top1_ratio") or 0.0, reverse=True)[:8]
        ],
    }


def _failure_overlap(
    case_to_variants: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    variant_names: Sequence[str],
    baseline_name: str,
    latest_name: str,
    matched_case_ids: Sequence[str],
) -> dict[str, Any]:
    names = tuple(variant_names)
    if baseline_name not in names:
        raise ValueError(f"baseline {baseline_name!r} is not in variants: {names}")
    if latest_name not in names:
        raise ValueError(f"latest {latest_name!r} is not in variants: {names}")
    all_starved = 0
    all_boundary = 0
    all_rigid = 0
    baseline_latest_starved = 0
    baseline_latest_boundary = 0
    rows: list[dict[str, Any]] = []
    for case_id in matched_case_ids:
        by_variant = case_to_variants[case_id]
        starved_variants = [name for name in names if by_variant[name]["starved"]]
        boundary_variants = [name for name in names if by_variant[name]["boundary_clamped"]]
        rigid_variants = [name for name in names if by_variant[name]["rigid"]]
        if len(starved_variants) == len(names):
            all_starved += 1
        if len(boundary_variants) == len(names):
            all_boundary += 1
        if len(rigid_variants) == len(names):
            all_rigid += 1
        if by_variant[baseline_name]["starved"] and by_variant[latest_name]["starved"]:
            baseline_latest_starved += 1
        if by_variant[baseline_name]["boundary_clamped"] and by_variant[latest_name]["boundary_clamped"]:
            baseline_latest_boundary += 1
        rows.append(
            {
                "case_id": case_id,
                "present_variants": list(names),
                "starved_variants": starved_variants,
                "boundary_clamped_variants": boundary_variants,
                "rigid_variants": rigid_variants,
                "score": len(starved_variants) + len(boundary_variants) + len(rigid_variants),
            }
        )
    rows.sort(key=lambda row: (-int(row["score"]), row["case_id"]))
    return {
        "matched_case_count": len(matched_case_ids),
        "baseline": baseline_name,
        "latest": latest_name,
        "all_variant_starved_count": all_starved,
        "all_variant_boundary_clamped_count": all_boundary,
        "all_variant_rigid_count": all_rigid,
        "baseline_and_latest_starved_count": baseline_latest_starved,
        "baseline_and_latest_boundary_clamped_count": baseline_latest_boundary,
        "top_overlap_cases": rows[:24],
    }


def _decision(
    variant_reports: Mapping[str, Mapping[str, Any]],
    *,
    baseline_name: str,
    overlap: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = variant_reports[baseline_name]
    baseline_counts = baseline["failure_counts"]
    baseline_event_rank = baseline["event_rank"]
    latest_name = tuple(variant_reports)[-1]
    latest = variant_reports[latest_name]
    latest_counts = latest["failure_counts"]
    matched_case_count = int(overlap.get("matched_case_count") or 0)
    relative_boundary_threshold = max(1, math.ceil(float(matched_case_count) * 0.25))
    event_valid_high = (baseline_event_rank.get("mean_event_valid_step_ratio") or 0.0) >= 0.70
    topk_gap_high = (baseline_event_rank.get("mean_event_topk_minus_top1_ratio") or 0.0) >= 0.15
    boundary_persistent = (
        overlap.get("baseline_and_latest_boundary_clamped_count", 0) >= relative_boundary_threshold
        or latest_counts.get("boundary_clamped", 0) >= max(1, math.ceil(float(matched_case_count) * 0.25))
    )
    dead_end_dominant = latest_counts.get("dead_end", 0) >= max(6, latest_counts.get("starved", 0) // 2)
    if dead_end_dominant and not event_valid_high:
        route = "TEST_GRAMMAR_STATE_REPAIR"
        reason = "dead ends dominate and event-valid evidence is weak"
        interpretation = (
            "The current evidence points to grammar-state unavailability. The next card should instrument or repair the "
            "specific replay states that run out of valid continuation options."
        )
        next_step = "Create a minimal grammar-state repair or full state/logit instrumentation card."
    elif boundary_persistent and event_valid_high:
        route = "TEST_DECODE_TIMING_CALIBRATION"
        reason = "boundary/starvation persists while event tokens remain broadly valid"
        interpretation = (
            "The failure is mostly generated-state timing/ranking collapse rather than basic grammar legality. The next "
            "small test should change decode/target timing pressure on existing checkpoints, with high-difficulty and "
            "boundary-clamp guards."
        )
        next_step = "Create a bounded decode timing-calibration card before more training or grammar replacement."
    elif topk_gap_high:
        route = "TEST_EVENT_RANKING_CALIBRATION"
        reason = "event candidates are often rank-near but underselected"
        interpretation = (
            "The evidence favors event-ranking calibration. Test a decode-time or target-side rank objective before "
            "changing the v3 grammar."
        )
        next_step = "Create a bounded event-ranking calibration card on the fixed 32-case slice."
    elif baseline_counts.get("starved", 0) > 0 and latest_counts.get("starved", 0) >= baseline_counts.get("starved", 0):
        route = "MUTATE_TO_V2_1_GRAMMAR"
        reason = "recent v3 mutations do not reduce starvation versus baseline"
        interpretation = (
            "The available summaries do not reveal a small v3 local mutation with positive pressure. Pivot toward v2.1 "
            "grammar improvement unless richer instrumentation changes this conclusion."
        )
        next_step = "Create a v2.1 grammar improvement card rather than another scalar v3 loss."
    else:
        route = "TEST_TRACE_INSTRUMENTATION"
        reason = "current summaries are not decisive enough"
        interpretation = (
            "The summaries are too coarse for a safe grammar mutation. Add full per-step state/logit traces on the shared "
            "worst cases before changing grammar."
        )
        next_step = "Create an instrumentation card for full per-step generated-state traces."
    return {
        "route": route,
        "reason": reason,
        "interpretation": interpretation,
        "next_step": next_step,
    }


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
    parser = argparse.ArgumentParser(description="Classify mapper v3 generated-state continuation failures.")
    parser.add_argument("--summary", action="append", required=True, help="Named summary path as name=path")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--report-output", required=True)
    args = parser.parse_args(argv)

    summaries = dict(parse_named_summary(value) for value in args.summary)
    summary = run_continuation_cluster_diagnostic(summaries=summaries, baseline_name=str(args.baseline))
    write_summary_json(summary, Path(args.summary_output))
    write_report(summary, Path(args.report_output))
    print(
        "mapper_v3_continuation_cluster_diagnostic_done "
        f"route={summary['decision']['route']} "
        f"matched_cases={summary['matched_case_count']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
