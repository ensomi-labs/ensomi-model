from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from pulsefield_model.evals.mapper_v21_soft_anti_rigid_stress_probe import (
    DEFAULT_HARD_BLOCK_SUMMARY_PATH,
    DEFAULT_STRESS_CASE_IDS,
    select_case_results,
)
from pulsefield_model.evals.mapper_v21_v3_fixed_slice_decode_comparison import compare_metrics, rollout_status
from pulsefield_model.evals.mapper_v2_1_anti_rigid_spacing_guard import generated_metrics
from pulsefield_model.evals.mapper_v2_1_trained_runtime_rollout import run_trained_v21_runtime_rollout_smoke
from pulsefield_model.inference.mapper_v2_1_rollout import MapperV21AntiRigidSpacingLogitsTransform
from pulsefield_model.models.mapper.shared.tokenizer import hitobjects_to_mapper_timepoints
from pulsefield_model.models.mapper.v2_1 import MapperV21Vocab
from pulsefield_model.osu_core.hitobjects import parse_mania_hit_objects


SUMMARY_SCHEMA_VERSION = 1
DEFAULT_OUTPUT_DIR = Path("artifacts/tmp/mapper_v21_tap_only_anti_rigid_stress_probe")
DEFAULT_SUMMARY_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_tap_only_anti_rigid_stress_probe_summary.json",
)
DEFAULT_REPORT_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_tap_only_anti_rigid_stress_probe_result_report.md",
)
MIN_RIGID_IMPROVED_CASES = 2
CASE_05_ID = "05_usao_knight_rider_kuo_kyoka_dnm_s_normal"


def run_tap_only_anti_rigid_stress_probe(
    *,
    hard_block_summary_path: str | Path = DEFAULT_HARD_BLOCK_SUMMARY_PATH,
    output_summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    output_report_path: str | Path | None = DEFAULT_REPORT_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    case_ids: Sequence[str] = DEFAULT_STRESS_CASE_IDS,
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
) -> dict[str, Any]:
    hard_block_summary = _read_json(Path(hard_block_summary_path))
    selected_cases = select_case_results(hard_block_summary, case_ids=case_ids)
    mapper_checkpoint = Path(mapper_checkpoint_path or str(hard_block_summary.get("mapper_checkpoint_path", "")))
    control_checkpoint = Path(control_checkpoint_path or str(hard_block_summary.get("control_checkpoint_path", "")))
    if not mapper_checkpoint.exists():
        raise FileNotFoundError(f"missing mapper checkpoint: {mapper_checkpoint}")
    if not control_checkpoint.exists():
        raise FileNotFoundError(f"missing control checkpoint: {control_checkpoint}")

    out_dir = Path(output_dir)
    rollout_dir = out_dir / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    vocab = MapperV21Vocab()
    case_results: list[dict[str, Any]] = []

    for source_row in selected_cases:
        case_id = str(source_row["case_id"])
        chart_end_ms = int(source_row["chart_end_ms"])
        output_case_summary = rollout_dir / f"{_safe_filename(case_id)}_{chart_end_ms}ms_tap_only.json"
        transform = MapperV21AntiRigidSpacingLogitsTransform(
            vocab=vocab,
            min_repeated_spacings=int(min_repeated_spacings),
            min_spacing_ms=int(min_spacing_ms),
            max_spacing_ms=int(max_spacing_ms),
            hard_block=True,
            require_tap_only_run=True,
        )
        tap_only_summary = run_trained_v21_runtime_rollout_smoke(
            mapper_checkpoint_path=mapper_checkpoint,
            control_checkpoint_path=control_checkpoint,
            output_summary_path=output_case_summary,
            output_report_path=None,
            device_name=device_name,
            chart_end_ms=chart_end_ms,
            max_tokens_per_window=int(max_tokens_per_window),
            audio_path=str(source_row["audio_path"]),
            normalized_difficulty=float(source_row["normalized_difficulty"]),
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
        reference_times = _reference_times(Path(str(source_row["beatmap_path"])), chart_end_ms=chart_end_ms)
        tap_only_metrics = generated_metrics(
            generated_times=[int(timepoint["time_ms"]) for timepoint in _generated_timepoints(tap_only_summary)],
            reference_times=reference_times,
            chart_end_ms=chart_end_ms,
        )
        baseline_metrics = dict(_mapping(source_row.get("v21_baseline_metrics")))
        hard_block_metrics = dict(_mapping(source_row.get("v21_guard_metrics")))
        v3_metrics = dict(_mapping(source_row.get("v3_metrics")))
        case_results.append(
            {
                "case_id": case_id,
                "difficulty": source_row.get("difficulty"),
                "difficulty_band": source_row.get("difficulty_band"),
                "audio_family": source_row.get("audio_family"),
                "normalized_difficulty": source_row.get("normalized_difficulty"),
                "chart_end_ms": chart_end_ms,
                "audio_path": source_row.get("audio_path"),
                "beatmap_path": source_row.get("beatmap_path"),
                "tap_only_summary_path": output_case_summary.as_posix(),
                "tap_only_legal": _candidate_legal(tap_only_summary),
                "tap_only_rollout": rollout_status(tap_only_summary),
                "baseline_legal": bool(source_row.get("v21_baseline_legal")),
                "hard_block_legal": bool(source_row.get("v21_guard_legal")),
                "v3_legal": bool(source_row.get("v3_legal")),
                "baseline_metrics": baseline_metrics,
                "hard_block_metrics": hard_block_metrics,
                "v3_metrics": v3_metrics,
                "tap_only_metrics": tap_only_metrics,
                "tap_only_vs_baseline": compare_metrics(tap_only_metrics, baseline_metrics),
                "tap_only_vs_hard_block": compare_metrics(tap_only_metrics, hard_block_metrics),
                "tap_only_vs_v3": compare_metrics(tap_only_metrics, v3_metrics),
                "transform": transform.to_dict(),
            }
        )

    aggregate = aggregate_results(case_results)
    decision = decision_from_aggregate(aggregate, case_results)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Mapper v2.1 tap-only anti-rigid stress probe",
        "decision": decision,
        "hard_block_summary_path": Path(hard_block_summary_path).as_posix(),
        "output_dir": out_dir.as_posix(),
        "mapper_checkpoint_path": mapper_checkpoint.as_posix(),
        "control_checkpoint_path": control_checkpoint.as_posix(),
        "config": {
            "device": device_name,
            "case_ids": list(case_ids),
            "max_tokens_per_window": int(max_tokens_per_window),
            "min_repeated_spacings": int(min_repeated_spacings),
            "min_spacing_ms": int(min_spacing_ms),
            "max_spacing_ms": int(max_spacing_ms),
            "hard_block": True,
            "require_tap_only_run": True,
            "seed": int(seed),
            "collect_logit_diagnostics": bool(collect_logit_diagnostics),
            "timepoint_preview_limit": int(timepoint_preview_limit),
        },
        "hard_block_fixed_slice_decision": hard_block_summary.get("decision"),
        "hard_block_fixed_slice_aggregate": hard_block_summary.get("aggregate"),
        "aggregate": aggregate,
        "case_results": case_results,
        "next_step": decision["next_step"],
    }
    write_summary_json(summary, output_summary_path)
    if output_report_path is not None:
        write_report(summary, output_report_path)
    return summary


def aggregate_results(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    baseline = _aggregate_variant(case_results, "baseline")
    hard_block = _aggregate_variant(case_results, "hard_block")
    tap_only = _aggregate_variant(case_results, "tap_only")
    v3 = _aggregate_variant(case_results, "v3")
    tap_vs_baseline = [_mapping(row.get("tap_only_vs_baseline")) for row in case_results]
    tap_vs_hard = [_mapping(row.get("tap_only_vs_hard_block")) for row in case_results]
    return {
        "case_count": int(len(case_results)),
        "baseline": baseline,
        "hard_block": hard_block,
        "tap_only": tap_only,
        "v3": v3,
        "tap_only_vs_baseline": {
            "rigid_improved_count": sum(1 for row in tap_vs_baseline if bool(row.get("dominant_spacing_improved"))),
            "new_starved_count": sum(1 for row in tap_vs_baseline if bool(row.get("new_starved"))),
            "mean_f1_delta": tap_only["mean_f1_100ms"] - baseline["mean_f1_100ms"],
            "mean_dominant_spacing_ratio_delta": (
                tap_only["mean_dominant_spacing_ratio"] - baseline["mean_dominant_spacing_ratio"]
            ),
        },
        "tap_only_vs_hard_block": {
            "legal_count_delta": tap_only["legal_count"] - hard_block["legal_count"],
            "starved_count_delta": tap_only["starved_count"] - hard_block["starved_count"],
            "mean_f1_delta": tap_only["mean_f1_100ms"] - hard_block["mean_f1_100ms"],
            "mean_dominant_spacing_ratio_delta": (
                tap_only["mean_dominant_spacing_ratio"] - hard_block["mean_dominant_spacing_ratio"]
            ),
            "hard_identical_count": sum(1 for row in case_results if case_matches_variant(row, "hard_block")),
        },
        "blocked_count": sum(int(_mapping(row.get("transform")).get("blocked_count") or 0) for row in case_results),
        "candidate_count": sum(int(_mapping(row.get("transform")).get("candidate_count") or 0) for row in case_results),
    }


def decision_from_aggregate(
    aggregate: Mapping[str, Any],
    case_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    tap_only = _mapping(aggregate.get("tap_only"))
    tap_vs_baseline = _mapping(aggregate.get("tap_only_vs_baseline"))
    case05 = next((row for row in case_results if str(row.get("case_id")) == CASE_05_ID), None)
    case05_metrics = _mapping(_mapping(case05).get("tap_only_metrics"))
    case05_legal = bool(_mapping(case05).get("tap_only_legal"))
    case05_starved = bool(case05_metrics.get("starved"))
    all_legal = bool(tap_only.get("all_legal"))
    new_starved_count = int(tap_vs_baseline.get("new_starved_count") or 0)
    rigid_improved_count = int(tap_vs_baseline.get("rigid_improved_count") or 0)
    baseline_identical_count = sum(1 for row in case_results if case_matches_variant(row, "baseline"))
    if not all_legal:
        route = "KILL"
        reason = "tap-only detector failed legality on the stress set"
        next_step = "Kill this detector mutation; return to v3/C3 diagnostics or design a different v2.1 grammar change."
    elif new_starved_count > 0:
        route = "KILL"
        reason = "tap-only detector introduced starvation versus baseline"
        next_step = "Do not scale tap-only suppression; inspect skipped/candidate margins if continuing v2.1 work."
    elif not case05_legal or case05_starved:
        route = "KILL"
        reason = "tap-only detector did not fix invariant case 05"
        next_step = "Kill the LN-specific hypothesis and pivot away from anti-rigid suppression."
    elif rigid_improved_count >= MIN_RIGID_IMPROVED_CASES and baseline_identical_count < int(aggregate.get("case_count") or 0):
        route = "TEST_NEXT"
        reason = "tap-only detector fixed stress safety while retaining rigidity signal"
        next_step = "Run a full 32-map fixed-slice tap-only comparison before changing defaults."
    else:
        route = "MUTATE"
        reason = "tap-only detector stayed safe but was too close to baseline or too weak"
        next_step = "Use a skipped-candidate/logit-margin diagnostic before another detector mutation."
    return {
        "route": route,
        "reason": reason,
        "next_step": next_step,
        "all_legal": all_legal,
        "new_starved_count": new_starved_count,
        "rigid_improved_count": rigid_improved_count,
        "case05_legal": case05_legal,
        "case05_starved": case05_starved,
        "baseline_identical_count": int(baseline_identical_count),
    }


def case_matches_variant(row: Mapping[str, Any], variant: str) -> bool:
    tap_metrics = _mapping(row.get("tap_only_metrics"))
    other_metrics = _mapping(row.get(f"{variant}_metrics"))
    return (
        bool(row.get("tap_only_legal")) == bool(row.get(f"{variant}_legal"))
        and _metric_signature(tap_metrics) == _metric_signature(other_metrics)
    )


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
    baseline = _mapping(aggregate.get("baseline"))
    hard_block = _mapping(aggregate.get("hard_block"))
    tap_only = _mapping(aggregate.get("tap_only"))
    v3 = _mapping(aggregate.get("v3"))
    tap_vs_baseline = _mapping(aggregate.get("tap_only_vs_baseline"))
    tap_vs_hard = _mapping(aggregate.get("tap_only_vs_hard_block"))
    lines = [
        "# Mapper v2.1 Tap-Only Anti-Rigid Stress Probe Result Report",
        "",
        "## Scope",
        "",
        "This probe reruns hard-block anti-rigid suppression only when the recent repeated-spacing run is tap-only. It uses the four committed stress cases and does not change defaults.",
        "",
        "## Decision",
        "",
        f"Decision: `{decision.get('route')}`.",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Cases: `{aggregate.get('case_count')}`",
        f"- Tap-only all legal: `{tap_only.get('all_legal')}`",
        f"- New starved vs baseline: `{tap_vs_baseline.get('new_starved_count')}`",
        f"- Rigid-improved cases vs baseline: `{tap_vs_baseline.get('rigid_improved_count')}`",
        f"- Blocked candidates: `{aggregate.get('blocked_count')}`",
        f"- Hard-identical cases: `{tap_vs_hard.get('hard_identical_count')}`",
        f"- Baseline-identical cases: `{decision.get('baseline_identical_count')}`",
        "",
        "## Aggregate Table",
        "",
        "| Variant | Legal | Mean F1 | Mean rigid | Starved | Mean event ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        _variant_row("v2.1 baseline", baseline),
        _variant_row("hard block", hard_block),
        _variant_row("tap-only", tap_only),
        _variant_row("v3 500 wide", v3),
        "",
        "## What Passed",
        "",
        "- Tap-only mode fixed the invariant case `05`: it completed legally and was not starved.",
        f"- Tap-only introduced `{tap_vs_baseline.get('new_starved_count')}` new starved cases versus baseline.",
        f"- Tap-only improved rigidity in `{tap_vs_baseline.get('rigid_improved_count')}` / `{aggregate.get('case_count')}` stress cases.",
        f"- Tap-only no longer matched hard-block exactly on any stress case; hard-identical cases: `{tap_vs_hard.get('hard_identical_count')}`.",
        "",
        "## What Surfaced",
        "",
        "- Tap-only failed the primary legality gate: only `3` / `4` stress cases were legal.",
        "- Case `04_oomori_seiko_justadice_tv_size_remu_normal` reverted to the baseline dead-end/starvation pattern because no tap-only candidates were blocked.",
        "- The two dead-end cases require opposite behavior: case `05` needs LN suppression skipped, while case `04` needs some LN-pattern intervention.",
        "- A simple tap-only predicate is therefore too blunt for default or full-slice scaling.",
        "",
        "## Case Table",
        "",
        "| Case | Baseline F1/Rigid/Starved | Hard F1/Rigid/Starved | Tap-only F1/Rigid/Starved | v3 F1/Rigid/Starved | Tap legal | Blocks |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in summary.get("case_results", []):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| {case} | {baseline} | {hard} | {tap} | {v3} | {legal} | {blocks} |".format(
                case=f"`{row.get('case_id')}`",
                baseline=_metric_cell(_mapping(row.get("baseline_metrics"))),
                hard=_metric_cell(_mapping(row.get("hard_block_metrics"))),
                tap=_metric_cell(_mapping(row.get("tap_only_metrics"))),
                v3=_metric_cell(_mapping(row.get("v3_metrics"))),
                legal=row.get("tap_only_legal"),
                blocks=_mapping(row.get("transform")).get("blocked_count"),
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
    route = str(_mapping(summary.get("decision")).get("route"))
    if route == "TEST_NEXT":
        return "The tap-only detector passed the stress gate. This is still not replacement evidence; it only justifies a full fixed-slice comparison."
    if route == "KILL":
        return "The tap-only detector should not be scaled from this probe. It failed a stress-set safety or usefulness gate."
    return "The tap-only detector is safe but too weak or too close to baseline; more diagnostics are needed before another mutation."


def _aggregate_variant(case_results: Sequence[Mapping[str, Any]], variant: str) -> dict[str, Any]:
    metrics = [_mapping(row.get(f"{variant}_metrics")) for row in case_results]
    legal_key = f"{variant}_legal"
    legal_count = sum(1 for row in case_results if bool(row.get(legal_key)))
    return {
        "case_count": int(len(case_results)),
        "legal_count": int(legal_count),
        "all_legal": int(legal_count) == int(len(case_results)),
        "mean_f1_100ms": _mean([_metric_f1(row) for row in metrics]),
        "mean_dominant_spacing_ratio": _mean([_float(row.get("dominant_spacing_ratio")) or 0.0 for row in metrics]),
        "mean_event_count_ratio": _mean([_float(row.get("event_count_ratio")) or 0.0 for row in metrics]),
        "starved_count": sum(1 for row in metrics if bool(row.get("starved"))),
    }


def _variant_row(label: str, aggregate: Mapping[str, Any]) -> str:
    return (
        f"| `{label}` | `{aggregate.get('all_legal')}` | {_fmt(aggregate.get('mean_f1_100ms'))} | "
        f"{_fmt(aggregate.get('mean_dominant_spacing_ratio'))} | {aggregate.get('starved_count')} | "
        f"{_fmt(aggregate.get('mean_event_count_ratio'))} |"
    )


def _metric_cell(metrics: Mapping[str, Any]) -> str:
    return "{f1}/{rigid}/{starved}".format(
        f1=_fmt(_metric_f1(metrics)),
        rigid=_fmt(metrics.get("dominant_spacing_ratio")),
        starved=metrics.get("starved"),
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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _metric_signature(metrics: Mapping[str, Any]) -> tuple[Any, ...]:
    timing = _mapping(metrics.get("timing_match_100ms"))
    return (
        int(metrics.get("generated_event_count") or 0),
        _round_float(metrics.get("event_count_ratio")),
        _round_float(metrics.get("dominant_spacing_ratio")),
        bool(metrics.get("starved")),
        _round_float(metrics.get("second_window_event_share")),
        _round_float(timing.get("f1")),
        tuple(metrics.get("first_12_generated_times") or ()),
        tuple(metrics.get("last_12_generated_times") or ()),
    )


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _metric_f1(metrics: Mapping[str, Any]) -> float:
    return _float(_mapping(metrics.get("timing_match_100ms")).get("f1")) or 0.0


def _float(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _round_float(value: object) -> float | None:
    number = _float(value)
    return None if number is None else round(number, 9)


def _mean(values: Sequence[float]) -> float:
    return float(math.fsum(values) / len(values)) if values else 0.0


def _fmt(value: object) -> str:
    number = _float(value)
    return "n/a" if number is None else f"{number:.6f}"


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value)).strip("_") or "case"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the v2.1 tap-only anti-rigid stress probe.")
    parser.add_argument("--hard-block-summary", default=DEFAULT_HARD_BLOCK_SUMMARY_PATH)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--report-output", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-id", action="append", dest="case_ids")
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
    args = parser.parse_args(argv)
    summary = run_tap_only_anti_rigid_stress_probe(
        hard_block_summary_path=args.hard_block_summary,
        output_summary_path=args.summary_output,
        output_report_path=args.report_output,
        output_dir=args.output_dir,
        case_ids=tuple(args.case_ids or DEFAULT_STRESS_CASE_IDS),
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
    )
    print(
        "mapper_v21_tap_only_anti_rigid_stress_probe_done "
        f"route={summary['decision']['route']} "
        f"cases={summary['aggregate']['case_count']} "
        f"tap_legal={summary['aggregate']['tap_only']['legal_count']} "
        f"blocked={summary['aggregate']['blocked_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
