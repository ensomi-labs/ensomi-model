from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from pulsefield_model.evals.mapper_v2_1_trained_runtime_rollout import run_trained_v21_runtime_rollout_smoke
from pulsefield_model.inference.mapper_v2_1_rollout import MapperV21AntiRigidSpacingLogitsTransform
from pulsefield_model.models.mapper.shared.tokenizer import hitobjects_to_mapper_timepoints
from pulsefield_model.models.mapper.v2_1 import MapperV21Vocab
from pulsefield_model.osu_core.hitobjects import parse_mania_hit_objects


SUMMARY_SCHEMA_VERSION = 1
DEFAULT_BASELINE_SUMMARY_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_matched_v21_timing_baseline_summary.json",
)
DEFAULT_OUTPUT_DIR = Path("artifacts/tmp/mapper_v21_anti_rigid_spacing_guard")
DEFAULT_SUMMARY_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_anti_rigid_spacing_guard_runtime_summary.json",
)
DEFAULT_REPORT_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_anti_rigid_spacing_guard_runtime_result_report.md",
)
TIMING_F1_TOLERANCE_MS = 100
SECOND_WINDOW_START_MS = 8_000
STARVED_SHARE_THRESHOLD = 0.10
REFERENCE_SECOND_WINDOW_SUPPORT_THRESHOLD = 0.25
MEAN_F1_REGRESSION_LIMIT = -0.05
RIGID_IMPROVEMENT_EPSILON = 1e-9


def run_v21_anti_rigid_spacing_gate(
    *,
    baseline_summary_path: str | Path = DEFAULT_BASELINE_SUMMARY_PATH,
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
    collect_logit_diagnostics: bool = True,
    timepoint_preview_limit: int = 4096,
) -> dict[str, Any]:
    baseline_summary = _read_json(Path(baseline_summary_path))
    baseline_runs_raw = baseline_summary.get("runs")
    if not isinstance(baseline_runs_raw, list) or not baseline_runs_raw:
        raise ValueError("baseline summary must contain a non-empty runs list")
    baseline_runs = [dict(row) for row in baseline_runs_raw if isinstance(row, Mapping)]
    if len(baseline_runs) != len(baseline_runs_raw):
        raise ValueError("every baseline run must be a JSON object")

    mapper_checkpoint = Path(mapper_checkpoint_path or str(baseline_summary.get("checkpoint_path", "")))
    control_checkpoint = Path(control_checkpoint_path or str(baseline_summary.get("control_checkpoint_path", "")))
    if not mapper_checkpoint.exists():
        raise FileNotFoundError(f"missing mapper checkpoint: {mapper_checkpoint}")
    if not control_checkpoint.exists():
        raise FileNotFoundError(f"missing control checkpoint: {control_checkpoint}")

    out_dir = Path(output_dir)
    rollout_dir = out_dir / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    vocab = MapperV21Vocab()
    case_results: list[dict[str, Any]] = []

    for baseline in baseline_runs:
        case_id = str(baseline.get("case_id", "case"))
        chart_end_ms = int(baseline["chart_end_ms"])
        output_case_summary = rollout_dir / f"{_safe_filename(case_id)}_{chart_end_ms}ms_summary.json"
        transform = MapperV21AntiRigidSpacingLogitsTransform(
            vocab=vocab,
            min_repeated_spacings=int(min_repeated_spacings),
            min_spacing_ms=int(min_spacing_ms),
            max_spacing_ms=int(max_spacing_ms),
        )
        candidate_summary = run_trained_v21_runtime_rollout_smoke(
            mapper_checkpoint_path=mapper_checkpoint,
            control_checkpoint_path=control_checkpoint,
            output_summary_path=output_case_summary,
            output_report_path=None,
            device_name=device_name,
            chart_end_ms=chart_end_ms,
            max_tokens_per_window=int(max_tokens_per_window),
            audio_path=str(baseline["audio_path"]),
            normalized_difficulty=float(baseline["normalized_difficulty"]),
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
            logits_transform=transform,
            timepoint_preview_limit=int(timepoint_preview_limit),
        )
        generated_timepoints = _generated_timepoints(candidate_summary)
        generated_times = [int(timepoint["time_ms"]) for timepoint in generated_timepoints]
        reference_times = _reference_times(Path(str(baseline["beatmap_path"])), chart_end_ms=chart_end_ms)
        metrics = generated_metrics(
            generated_times=generated_times,
            reference_times=reference_times,
            chart_end_ms=chart_end_ms,
        )
        comparison = compare_to_baseline(candidate_metrics=metrics, baseline_row=baseline)
        case_results.append(
            {
                "case_id": case_id,
                "label": baseline.get("label"),
                "chart_end_ms": chart_end_ms,
                "difficulty": baseline.get("difficulty"),
                "normalized_difficulty": baseline.get("normalized_difficulty"),
                "audio_path": baseline.get("audio_path"),
                "beatmap_path": baseline.get("beatmap_path"),
                "candidate_summary_path": output_case_summary.as_posix(),
                "legal": _candidate_legal(candidate_summary),
                "completed": bool(_mapping(candidate_summary.get("rollout")).get("completed")),
                "dead_end": bool(_mapping(candidate_summary.get("rollout")).get("dead_end")),
                "max_tokens_exceeded": bool(_mapping(candidate_summary.get("rollout")).get("max_tokens_exceeded")),
                "candidate_metrics": metrics,
                "baseline_metrics": _baseline_metrics(baseline, reference_times=reference_times),
                "comparison": comparison,
                "transform": transform.to_dict(),
            }
        )

    aggregate = aggregate_results(case_results)
    decision = decision_from_aggregate(aggregate)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Mapper v2.1 anti-rigid spacing guard runtime gate",
        "decision": decision,
        "baseline_summary_path": Path(baseline_summary_path).as_posix(),
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
        },
        "baseline_aggregate": baseline_summary.get("aggregate"),
        "aggregate": aggregate,
        "case_results": case_results,
        "next_step": decision["next_step"],
    }
    write_summary_json(summary, output_summary_path)
    if output_report_path is not None:
        write_report(summary, output_report_path)
    return summary


def generated_metrics(
    *,
    generated_times: Sequence[int],
    reference_times: Sequence[int],
    chart_end_ms: int,
) -> dict[str, Any]:
    generated = sorted(int(time_ms) for time_ms in generated_times if 0 <= int(time_ms) <= int(chart_end_ms))
    reference = sorted(int(time_ms) for time_ms in reference_times if 0 <= int(time_ms) <= int(chart_end_ms))
    second_window_count = sum(1 for time_ms in generated if time_ms >= SECOND_WINDOW_START_MS)
    reference_second_window_count = sum(1 for time_ms in reference if time_ms >= SECOND_WINDOW_START_MS)
    boundary_start = max(0, int(chart_end_ms) - 100)
    boundary_count = sum(1 for time_ms in generated if time_ms >= boundary_start)
    deltas = [right - left for left, right in zip(generated, generated[1:])]
    duplicate_or_nonincreasing = sum(1 for delta in deltas if delta <= 0)
    positive_deltas = [delta for delta in deltas if delta > 0]
    spacing_counts = Counter(str(delta) for delta in positive_deltas)
    dominant_spacing_ms = None
    dominant_spacing_ratio = 0.0
    if positive_deltas:
        dominant_spacing_ms, dominant_count = Counter(positive_deltas).most_common(1)[0]
        dominant_spacing_ratio = float(dominant_count) / float(len(positive_deltas))
    timing = timing_match(generated, reference, tolerance_ms=TIMING_F1_TOLERANCE_MS)
    reference_second_share = _safe_ratio(reference_second_window_count, len(reference))
    second_share = _safe_ratio(second_window_count, len(generated))
    starved = bool(
        int(chart_end_ms) > SECOND_WINDOW_START_MS
        and reference_second_share >= REFERENCE_SECOND_WINDOW_SUPPORT_THRESHOLD
        and second_share <= STARVED_SHARE_THRESHOLD
    )
    return {
        "generated_event_count": len(generated),
        "reference_event_count": len(reference),
        "event_count_ratio": _safe_ratio(len(generated), len(reference)),
        "second_window_event_count": second_window_count,
        "second_window_event_share": second_share,
        "reference_second_window_event_count": reference_second_window_count,
        "reference_second_window_event_share": reference_second_share,
        "starved": starved,
        "boundary_event_count": boundary_count,
        "boundary_event_ratio": _safe_ratio(boundary_count, len(generated)),
        "duplicate_or_nonincreasing_spacing_ratio": _safe_ratio(duplicate_or_nonincreasing, max(0, len(generated) - 1)),
        "dominant_spacing_ms": dominant_spacing_ms,
        "dominant_spacing_ratio": dominant_spacing_ratio,
        "spacing_counts": dict(sorted(spacing_counts.items(), key=lambda item: (-int(item[1]), int(item[0])))),
        "first_12_generated_times": generated[:12],
        "last_12_generated_times": generated[-12:],
        "first_12_reference_times": reference[:12],
        "last_12_reference_times": reference[-12:],
        "timing_match_100ms": timing,
    }


def timing_match(generated: Sequence[int], reference: Sequence[int], *, tolerance_ms: int) -> dict[str, Any]:
    generated_sorted = sorted(int(value) for value in generated)
    reference_sorted = sorted(int(value) for value in reference)
    gen_index = 0
    ref_index = 0
    matched = 0
    tolerance = int(tolerance_ms)
    while gen_index < len(generated_sorted) and ref_index < len(reference_sorted):
        gen_time = generated_sorted[gen_index]
        ref_time = reference_sorted[ref_index]
        delta = gen_time - ref_time
        if abs(delta) <= tolerance:
            matched += 1
            gen_index += 1
            ref_index += 1
        elif gen_time < ref_time - tolerance:
            gen_index += 1
        else:
            ref_index += 1
    precision = _safe_ratio(matched, len(generated_sorted))
    recall = _safe_ratio(matched, len(reference_sorted))
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "tolerance_ms": tolerance,
        "matched_generated": matched,
        "matched_reference": matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compare_to_baseline(*, candidate_metrics: Mapping[str, Any], baseline_row: Mapping[str, Any]) -> dict[str, Any]:
    baseline_timing = _mapping(baseline_row.get("timing_match_100ms"))
    candidate_timing = _mapping(candidate_metrics.get("timing_match_100ms"))
    baseline_rigid = _float(baseline_row.get("dominant_spacing_ratio")) or 0.0
    candidate_rigid = _float(candidate_metrics.get("dominant_spacing_ratio")) or 0.0
    baseline_second = _float(baseline_row.get("second_window_event_share")) or 0.0
    candidate_second = _float(candidate_metrics.get("second_window_event_share")) or 0.0
    baseline_f1 = _float(baseline_timing.get("f1")) or 0.0
    candidate_f1 = _float(candidate_timing.get("f1")) or 0.0
    return {
        "dominant_spacing_ratio_delta": candidate_rigid - baseline_rigid,
        "dominant_spacing_improved": candidate_rigid < baseline_rigid - RIGID_IMPROVEMENT_EPSILON,
        "second_window_event_share_delta": candidate_second - baseline_second,
        "event_count_ratio_delta": (_float(candidate_metrics.get("event_count_ratio")) or 0.0)
        - (_float(baseline_row.get("event_count_ratio")) or 0.0),
        "f1_delta": candidate_f1 - baseline_f1,
        "boundary_event_ratio_delta": (_float(candidate_metrics.get("boundary_event_ratio")) or 0.0)
        - (_float(baseline_row.get("boundary_event_ratio")) or 0.0),
    }


def aggregate_results(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    comparisons = [_mapping(row.get("comparison")) for row in case_results]
    candidate_metrics = [_mapping(row.get("candidate_metrics")) for row in case_results]
    baseline_metrics = [_mapping(row.get("baseline_metrics")) for row in case_results]
    legal_count = sum(1 for row in case_results if bool(row.get("legal")))
    rigid_improved = sum(1 for row in comparisons if bool(row.get("dominant_spacing_improved")))
    starved_count = sum(1 for row in candidate_metrics if bool(row.get("starved")))
    baseline_starved_count = sum(1 for row in baseline_metrics if bool(row.get("starved")))
    new_starved_count = sum(
        1
        for candidate, baseline in zip(candidate_metrics, baseline_metrics, strict=True)
        if bool(candidate.get("starved")) and not bool(baseline.get("starved"))
    )
    blocked_count = sum(int(_mapping(row.get("transform")).get("blocked_count") or 0) for row in case_results)
    candidate_f1 = [_float(_mapping(row.get("timing_match_100ms")).get("f1")) or 0.0 for row in candidate_metrics]
    baseline_f1 = [_float(_mapping(row.get("timing_match_100ms")).get("f1")) or 0.0 for row in baseline_metrics]
    candidate_rigid = [_float(row.get("dominant_spacing_ratio")) or 0.0 for row in candidate_metrics]
    baseline_rigid = [_float(row.get("dominant_spacing_ratio")) or 0.0 for row in baseline_metrics]
    mean_candidate_f1 = _mean(candidate_f1)
    mean_baseline_f1 = _mean(baseline_f1)
    return {
        "case_count": int(len(case_results)),
        "legal_count": int(legal_count),
        "all_legal": int(legal_count) == int(len(case_results)),
        "rigid_improved_count": int(rigid_improved),
        "mean_candidate_f1": mean_candidate_f1,
        "mean_baseline_f1": mean_baseline_f1,
        "mean_f1_delta": mean_candidate_f1 - mean_baseline_f1,
        "mean_candidate_dominant_spacing_ratio": _mean(candidate_rigid),
        "mean_baseline_dominant_spacing_ratio": _mean(baseline_rigid),
        "starved_count": int(starved_count),
        "baseline_starved_count": int(baseline_starved_count),
        "new_starved_count": int(new_starved_count),
        "blocked_count": int(blocked_count),
        "max_token_count": sum(1 for row in case_results if bool(row.get("max_tokens_exceeded"))),
    }


def decision_from_aggregate(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    all_legal = bool(aggregate.get("all_legal"))
    rigid_improved_count = int(aggregate.get("rigid_improved_count") or 0)
    mean_f1_delta = float(aggregate.get("mean_f1_delta") or 0.0)
    new_starved_count = int(aggregate.get("new_starved_count") or 0)
    blocked_count = int(aggregate.get("blocked_count") or 0)
    if not all_legal:
        route = "KILL_HARD_BLOCK"
        reason = "one or more anti-rigid rollouts were illegal"
        next_step = "Do not scale hard blocking; mutate to a finite penalty or return to v3 instrumentation."
    elif blocked_count == 0:
        route = "MUTATE"
        reason = "the guard never activated on the matched runtime cases"
        next_step = "Inspect why the generated-state detector missed runtime rigid grids before another rollout."
    elif rigid_improved_count >= 2 and mean_f1_delta >= MEAN_F1_REGRESSION_LIMIT and new_starved_count == 0:
        route = "TEST_NEXT"
        reason = "rigidity improved on at least two cases without legality, F1, or starvation guard failure"
        next_step = "Run a wider fixed-slice v2.1/v3 decode timing comparison before changing defaults."
    elif rigid_improved_count < 2:
        route = "MUTATE"
        reason = "dominant-spacing ratio did not improve on enough matched cases"
        next_step = "Mutate the guard to finite penalties or richer timing-state grammar before wider runtime."
    else:
        route = "KILL_HARD_BLOCK"
        reason = "rigidity moved but guard metrics regressed"
        next_step = "Do not scale hard blocking; test a softer penalty only if the failure examples justify it."
    return {
        "route": route,
        "reason": reason,
        "next_step": next_step,
        "all_legal": all_legal,
        "rigid_improved_count": rigid_improved_count,
        "mean_f1_delta": mean_f1_delta,
        "new_starved_count": new_starved_count,
        "blocked_count": blocked_count,
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
    aggregate = _mapping(summary.get("aggregate"))
    lines = [
        "# Mapper v2.1 Anti-Rigid Spacing Guard Runtime Result Report",
        "",
        "## Scope",
        "",
        "This gate reruns the six matched v2.1 timing-baseline cases with the opt-in hard-block anti-rigid spacing transform. It does not change mapper defaults, model weights, tokenization, or training.",
        "",
        "## Decision",
        "",
        f"Decision: `{decision.get('route')}`.",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Cases: `{aggregate.get('case_count')}`",
        f"- All legal: `{aggregate.get('all_legal')}`",
        f"- Transform blocked count: `{aggregate.get('blocked_count')}`",
        f"- Rigid-improved cases: `{aggregate.get('rigid_improved_count')}`",
        f"- Mean F1 delta: `{_fmt(aggregate.get('mean_f1_delta'))}`",
        f"- New starved cases: `{aggregate.get('new_starved_count')}`",
        f"- Mean dominant-spacing ratio: `{_fmt(aggregate.get('mean_baseline_dominant_spacing_ratio'))}` -> `{_fmt(aggregate.get('mean_candidate_dominant_spacing_ratio'))}`",
        "",
        "## Case Table",
        "",
        "| Case | Prefix | Legal | Blocks | Rigid base->guard | F1 base->guard | 2nd share base->guard | Events base->guard | Dominant spacing |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary.get("case_results", []):
        if not isinstance(row, Mapping):
            continue
        baseline = _mapping(row.get("baseline_metrics"))
        candidate = _mapping(row.get("candidate_metrics"))
        transform = _mapping(row.get("transform"))
        baseline_f1 = _float(_mapping(baseline.get("timing_match_100ms")).get("f1"))
        candidate_f1 = _float(_mapping(candidate.get("timing_match_100ms")).get("f1"))
        lines.append(
            "| {case} | {prefix} | {legal} | {blocks} | {base_rigid}->{cand_rigid} | {base_f1}->{cand_f1} | {base_second}->{cand_second} | {base_events}->{cand_events} | {dom} |".format(
                case=f"`{row.get('case_id')}`",
                prefix=row.get("chart_end_ms"),
                legal=row.get("legal"),
                blocks=transform.get("blocked_count"),
                base_rigid=_fmt(baseline.get("dominant_spacing_ratio")),
                cand_rigid=_fmt(candidate.get("dominant_spacing_ratio")),
                base_f1=_fmt(baseline_f1),
                cand_f1=_fmt(candidate_f1),
                base_second=_fmt(baseline.get("second_window_event_share")),
                cand_second=_fmt(candidate.get("second_window_event_share")),
                base_events=baseline.get("generated_event_count"),
                cand_events=candidate.get("generated_event_count"),
                dom=f"`{candidate.get('dominant_spacing_ms')}`",
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


def _interpretation(summary: Mapping[str, Any]) -> str:
    decision = _mapping(summary.get("decision"))
    route = str(decision.get("route"))
    if route == "TEST_NEXT":
        return "The hard-block guard produced enough runtime signal to justify a wider decode timing comparison, but it remains opt-in and is not a default grammar change."
    if route == "KILL_HARD_BLOCK":
        return "The hard-block variant should not be scaled. It either broke legality or traded rigidity for unacceptable quality/continuation regressions."
    return "The hard-block guard did not clear the matched-case signal gate. Treat the result as a mutation target, not a runtime improvement."


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


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
    return bool(
        checks.get("rollout_not_dead_end")
        and checks.get("rollout_not_max_tokens_exceeded")
        and _mapping(summary.get("rollout")).get("completed")
    )


def _baseline_metrics(row: Mapping[str, Any], *, reference_times: Sequence[int]) -> dict[str, Any]:
    reference_second_count = sum(1 for time_ms in reference_times if int(time_ms) >= SECOND_WINDOW_START_MS)
    reference_second_share = _safe_ratio(reference_second_count, len(reference_times))
    second_share = _float(row.get("second_window_event_share")) or 0.0
    return {
        "generated_event_count": int(row.get("generated_event_count") or 0),
        "reference_event_count": int(row.get("reference_event_count") or 0),
        "event_count_ratio": _float(row.get("event_count_ratio")) or 0.0,
        "second_window_event_share": second_share,
        "reference_second_window_event_count": int(reference_second_count),
        "reference_second_window_event_share": reference_second_share,
        "starved": bool(
            int(row.get("chart_end_ms") or 0) > SECOND_WINDOW_START_MS
            and reference_second_share >= REFERENCE_SECOND_WINDOW_SUPPORT_THRESHOLD
            and second_share <= STARVED_SHARE_THRESHOLD
        ),
        "boundary_event_ratio": _float(row.get("boundary_event_ratio")) or 0.0,
        "dominant_spacing_ms": row.get("dominant_spacing_ms"),
        "dominant_spacing_ratio": _float(row.get("dominant_spacing_ratio")) or 0.0,
        "timing_match_100ms": dict(_mapping(row.get("timing_match_100ms"))),
    }


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


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
    parser = argparse.ArgumentParser(description="Run the v2.1 anti-rigid spacing guard six-case runtime gate.")
    parser.add_argument("--baseline-summary", default=DEFAULT_BASELINE_SUMMARY_PATH)
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
    parser.add_argument("--no-logit-diagnostics", action="store_true")
    parser.add_argument("--timepoint-preview-limit", type=int, default=4096)
    args = parser.parse_args(argv)
    summary = run_v21_anti_rigid_spacing_gate(
        baseline_summary_path=args.baseline_summary,
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
        collect_logit_diagnostics=not bool(args.no_logit_diagnostics),
        timepoint_preview_limit=args.timepoint_preview_limit,
    )
    print(
        "mapper_v21_anti_rigid_spacing_gate_done "
        f"route={summary['decision']['route']} "
        f"cases={summary['aggregate']['case_count']} "
        f"rigid_improved={summary['aggregate']['rigid_improved_count']} "
        f"blocks={summary['aggregate']['blocked_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
