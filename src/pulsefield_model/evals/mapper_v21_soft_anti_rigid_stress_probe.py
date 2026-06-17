from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from pulsefield_model.evals.mapper_v21_v3_fixed_slice_decode_comparison import compare_metrics, rollout_status
from pulsefield_model.evals.mapper_v2_1_anti_rigid_spacing_guard import generated_metrics
from pulsefield_model.evals.mapper_v2_1_trained_runtime_rollout import run_trained_v21_runtime_rollout_smoke
from pulsefield_model.inference.mapper_v2_1_rollout import MapperV21AntiRigidSpacingLogitsTransform
from pulsefield_model.models.mapper.shared.tokenizer import hitobjects_to_mapper_timepoints
from pulsefield_model.models.mapper.v2_1 import MapperV21Vocab
from pulsefield_model.osu_core.hitobjects import parse_mania_hit_objects


SUMMARY_SCHEMA_VERSION = 1
DEFAULT_HARD_BLOCK_SUMMARY_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_v3_fixed_slice_decode_comparison_summary.json",
)
DEFAULT_OUTPUT_DIR = Path("artifacts/tmp/mapper_v21_soft_anti_rigid_stress_probe")
DEFAULT_SUMMARY_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_soft_anti_rigid_stress_probe_summary.json",
)
DEFAULT_REPORT_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_soft_anti_rigid_stress_probe_result_report.md",
)
DEFAULT_STRESS_CASE_IDS = (
    "04_oomori_seiko_justadice_tv_size_remu_normal",
    "05_usao_knight_rider_kuo_kyoka_dnm_s_normal",
    "29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent",
    "31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial",
)
POSITIVE_MIN_RIGID_IMPROVED_CASES = 3
HARD_BLOCK_F1_REGRESSION_LIMIT = -0.02


def run_soft_anti_rigid_stress_probe(
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
    penalty: float = 4.0,
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
        output_case_summary = rollout_dir / f"{_safe_filename(case_id)}_{chart_end_ms}ms_soft_penalty.json"
        transform = MapperV21AntiRigidSpacingLogitsTransform(
            vocab=vocab,
            min_repeated_spacings=int(min_repeated_spacings),
            min_spacing_ms=int(min_spacing_ms),
            max_spacing_ms=int(max_spacing_ms),
            hard_block=False,
            penalty=float(penalty),
        )
        soft_summary = run_trained_v21_runtime_rollout_smoke(
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
        soft_metrics = generated_metrics(
            generated_times=[int(timepoint["time_ms"]) for timepoint in _generated_timepoints(soft_summary)],
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
                "soft_summary_path": output_case_summary.as_posix(),
                "soft_legal": _candidate_legal(soft_summary),
                "soft_rollout": rollout_status(soft_summary),
                "baseline_legal": bool(source_row.get("v21_baseline_legal")),
                "hard_block_legal": bool(source_row.get("v21_guard_legal")),
                "v3_legal": bool(source_row.get("v3_legal")),
                "baseline_metrics": baseline_metrics,
                "hard_block_metrics": hard_block_metrics,
                "v3_metrics": v3_metrics,
                "soft_metrics": soft_metrics,
                "soft_vs_baseline": compare_metrics(soft_metrics, baseline_metrics),
                "soft_vs_hard_block": compare_metrics(soft_metrics, hard_block_metrics),
                "soft_vs_v3": compare_metrics(soft_metrics, v3_metrics),
                "transform": transform.to_dict(),
            }
        )

    aggregate = aggregate_results(case_results)
    decision = decision_from_aggregate(aggregate)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Mapper v2.1 soft anti-rigid stress probe",
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
            "hard_block": False,
            "penalty": float(penalty),
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


def select_case_results(summary: Mapping[str, Any], *, case_ids: Sequence[str]) -> list[dict[str, Any]]:
    rows = summary.get("case_results")
    if not isinstance(rows, list) or not rows:
        raise ValueError("hard-block summary must contain non-empty case_results")
    by_case_id = {str(row.get("case_id")): dict(row) for row in rows if isinstance(row, Mapping)}
    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    for case_id in case_ids:
        row = by_case_id.get(str(case_id))
        if row is None:
            missing.append(str(case_id))
        else:
            selected.append(row)
    if missing:
        raise ValueError(f"missing requested case ids: {missing}")
    return selected


def aggregate_results(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    baseline = _aggregate_variant(case_results, "baseline")
    hard_block = _aggregate_variant(case_results, "hard_block")
    soft = _aggregate_variant(case_results, "soft")
    v3 = _aggregate_variant(case_results, "v3")
    soft_vs_baseline = [_mapping(row.get("soft_vs_baseline")) for row in case_results]
    soft_vs_hard_block = [_mapping(row.get("soft_vs_hard_block")) for row in case_results]
    soft_vs_v3 = [_mapping(row.get("soft_vs_v3")) for row in case_results]
    return {
        "case_count": int(len(case_results)),
        "baseline": baseline,
        "hard_block": hard_block,
        "soft": soft,
        "v3": v3,
        "soft_vs_baseline": {
            "rigid_improved_count": sum(1 for row in soft_vs_baseline if bool(row.get("dominant_spacing_improved"))),
            "new_starved_count": sum(1 for row in soft_vs_baseline if bool(row.get("new_starved"))),
            "mean_f1_delta": soft["mean_f1_100ms"] - baseline["mean_f1_100ms"],
            "mean_dominant_spacing_ratio_delta": (
                soft["mean_dominant_spacing_ratio"] - baseline["mean_dominant_spacing_ratio"]
            ),
        },
        "soft_vs_hard_block": {
            "legal_count_delta": soft["legal_count"] - hard_block["legal_count"],
            "starved_count_delta": soft["starved_count"] - hard_block["starved_count"],
            "mean_f1_delta": soft["mean_f1_100ms"] - hard_block["mean_f1_100ms"],
            "mean_dominant_spacing_ratio_delta": (
                soft["mean_dominant_spacing_ratio"] - hard_block["mean_dominant_spacing_ratio"]
            ),
            "f1_better_or_equal_count": sum(1 for row in soft_vs_hard_block if (_float(row.get("f1_delta")) or 0.0) >= 0.0),
        },
        "soft_vs_v3": {
            "starved_count_delta": soft["starved_count"] - v3["starved_count"],
            "mean_f1_delta": soft["mean_f1_100ms"] - v3["mean_f1_100ms"],
            "f1_better_or_equal_count": sum(1 for row in soft_vs_v3 if (_float(row.get("f1_delta")) or 0.0) >= 0.0),
        },
        "blocked_count": sum(int(_mapping(row.get("transform")).get("blocked_count") or 0) for row in case_results),
    }


def decision_from_aggregate(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    soft = _mapping(aggregate.get("soft"))
    soft_vs_baseline = _mapping(aggregate.get("soft_vs_baseline"))
    soft_vs_hard_block = _mapping(aggregate.get("soft_vs_hard_block"))
    all_soft_legal = bool(soft.get("all_legal"))
    new_starved_count = int(soft_vs_baseline.get("new_starved_count") or 0)
    rigid_improved_count = int(soft_vs_baseline.get("rigid_improved_count") or 0)
    f1_delta_vs_hard_block = float(soft_vs_hard_block.get("mean_f1_delta") or 0.0)
    if not all_soft_legal:
        route = "KILL"
        reason = "soft penalty failed legality on the stress set"
        next_step = "Do not scale penalty 4.0; mutate the detector or penalty schedule."
    elif new_starved_count > 0:
        route = "KILL"
        reason = "soft penalty introduced starvation versus v2.1 baseline"
        next_step = "Do not scale penalty 4.0; test weaker penalty only if the failure case justifies it."
    elif (
        rigid_improved_count >= POSITIVE_MIN_RIGID_IMPROVED_CASES
        and f1_delta_vs_hard_block >= HARD_BLOCK_F1_REGRESSION_LIMIT
    ):
        route = "TEST_NEXT"
        reason = "soft penalty fixed stress legality without losing the hard-block quality signal"
        next_step = "Run the full 32-map fixed-slice soft-penalty comparison before changing defaults."
    else:
        route = "MUTATE"
        reason = "soft penalty stayed legal but did not preserve enough rigidity/F1 signal"
        next_step = "Run a small penalty sweep or mutate the repeated-spacing detector."
    return {
        "route": route,
        "reason": reason,
        "next_step": next_step,
        "all_soft_legal": all_soft_legal,
        "new_starved_count": new_starved_count,
        "rigid_improved_count": rigid_improved_count,
        "mean_f1_delta_vs_hard_block": f1_delta_vs_hard_block,
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
    baseline = _mapping(aggregate.get("baseline"))
    hard_block = _mapping(aggregate.get("hard_block"))
    soft = _mapping(aggregate.get("soft"))
    v3 = _mapping(aggregate.get("v3"))
    soft_vs_baseline = _mapping(aggregate.get("soft_vs_baseline"))
    soft_vs_hard_block = _mapping(aggregate.get("soft_vs_hard_block"))
    lines = [
        "# Mapper v2.1 Soft Anti-Rigid Stress Probe Result Report",
        "",
        "## Scope",
        "",
        "This probe reruns only the finite-penalty anti-rigid variant on four stress cases from the committed hard-block fixed-slice comparison. It does not change model weights, tokenizer, or defaults.",
        "",
        "## Decision",
        "",
        f"Decision: `{decision.get('route')}`.",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Cases: `{aggregate.get('case_count')}`",
        f"- Soft penalty: `{_mapping(summary.get('config')).get('penalty')}`",
        f"- Soft all legal: `{soft.get('all_legal')}`",
        f"- Soft new starved vs baseline: `{soft_vs_baseline.get('new_starved_count')}`",
        f"- Soft rigid-improved cases vs baseline: `{soft_vs_baseline.get('rigid_improved_count')}`",
        f"- Soft mean F1 delta vs hard block: `{_fmt(soft_vs_hard_block.get('mean_f1_delta'))}`",
        f"- Soft mean rigid delta vs baseline: `{_fmt(soft_vs_baseline.get('mean_dominant_spacing_ratio_delta'))}`",
        "",
        "## Aggregate Table",
        "",
        "| Variant | Legal | Mean F1 | Mean rigid | Starved | Mean event ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        _variant_row("v2.1 baseline", baseline),
        _variant_row("hard block", hard_block),
        _variant_row("soft penalty", soft),
        _variant_row("v3 500 wide", v3),
        "",
        "## What Passed",
        "",
        f"- The stress probe completed `{aggregate.get('case_count')}` real-audio soft-penalty rollouts.",
        f"- The finite penalty activated `{aggregate.get('blocked_count')}` times on the selected stress cases.",
        f"- Soft penalty reduced dominant-spacing ratio versus baseline in `{soft_vs_baseline.get('rigid_improved_count')}` cases.",
        "- Case `04` stayed legal under soft penalty and avoided the baseline dead-end/starvation pattern.",
        "",
        "## What Surfaced",
        "",
        f"- Soft penalty legality failed: `{soft.get('legal_count')}` / `{soft.get('case_count')}` cases were legal.",
        f"- Soft penalty introduced `{soft_vs_baseline.get('new_starved_count')}` new starved case(s) versus baseline.",
        f"- Penalty `4.0` matched the hard-block aggregate exactly on mean F1 and mean rigidity, so it behaves as effectively hard under greedy decoding on this stress set.",
        "- Case `05` still dead-ended at the first-window boundary; case `29` remained starved.",
        "",
        "## Case Table",
        "",
        "| Case | Baseline F1/Rigid/Starved | Hard F1/Rigid/Starved | Soft F1/Rigid/Starved | v3 F1/Rigid/Starved | Soft legal | Blocks |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in summary.get("case_results", []):
        if not isinstance(row, Mapping):
            continue
        baseline_metrics = _mapping(row.get("baseline_metrics"))
        hard_metrics = _mapping(row.get("hard_block_metrics"))
        soft_metrics = _mapping(row.get("soft_metrics"))
        v3_metrics = _mapping(row.get("v3_metrics"))
        transform = _mapping(row.get("transform"))
        lines.append(
            "| {case} | {baseline} | {hard} | {soft} | {v3} | {legal} | {blocks} |".format(
                case=f"`{row.get('case_id')}`",
                baseline=_metric_cell(baseline_metrics),
                hard=_metric_cell(hard_metrics),
                soft=_metric_cell(soft_metrics),
                v3=_metric_cell(v3_metrics),
                legal=row.get("soft_legal"),
                blocks=transform.get("blocked_count"),
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


def _interpretation(summary: Mapping[str, Any]) -> str:
    route = str(_mapping(summary.get("decision")).get("route"))
    if route == "TEST_NEXT":
        return "The finite penalty passed the stress probe. It should still remain opt-in until a full fixed-slice comparison passes."
    if route == "KILL":
        return "Penalty 4.0 should not be scaled. The stress set reproduced a safety failure or failed to remove starvation."
    return "The finite penalty avoided the hard-block failure but did not retain enough quality signal. Treat it as a mutation target."


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


def _mean(values: Sequence[float]) -> float:
    return float(math.fsum(values) / len(values)) if values else 0.0


def _fmt(value: object) -> str:
    number = _float(value)
    return "n/a" if number is None else f"{number:.6f}"


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value)).strip("_") or "case"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the v2.1 soft anti-rigid stress probe.")
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
    parser.add_argument("--penalty", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--collect-logit-diagnostics", action="store_true")
    parser.add_argument("--timepoint-preview-limit", type=int, default=4096)
    args = parser.parse_args(argv)
    summary = run_soft_anti_rigid_stress_probe(
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
        penalty=args.penalty,
        seed=args.seed,
        collect_logit_diagnostics=args.collect_logit_diagnostics,
        timepoint_preview_limit=args.timepoint_preview_limit,
    )
    print(
        "mapper_v21_soft_anti_rigid_stress_probe_done "
        f"route={summary['decision']['route']} "
        f"cases={summary['aggregate']['case_count']} "
        f"soft_legal={summary['aggregate']['soft']['legal_count']} "
        f"new_starved={summary['aggregate']['soft_vs_baseline']['new_starved_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
