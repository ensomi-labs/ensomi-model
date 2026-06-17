from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from pulsefield_model.evals.mapper_v21_soft_anti_rigid_stress_probe import (
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
DEFAULT_SOURCE_SUMMARY_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_v3_fixed_slice_decode_comparison_summary.json",
)
DEFAULT_OUTPUT_DIR = Path("artifacts/tmp/mapper_v21_terminal_ln_start_guard_repair")
DEFAULT_SUMMARY_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_terminal_ln_start_guard_repair_summary.json",
)
DEFAULT_REPORT_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_terminal_ln_start_guard_repair_result_report.md",
)
CASE_04_ID = "04_oomori_seiko_justadice_tv_size_remu_normal"
CASE_05_ID = "05_usao_knight_rider_kuo_kyoka_dnm_s_normal"
STRESS_F1_REGRESSION_LIMIT = -0.03


def run_terminal_ln_start_guard_repair(
    *,
    source_summary_path: str | Path = DEFAULT_SOURCE_SUMMARY_PATH,
    output_summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    output_report_path: str | Path | None = DEFAULT_REPORT_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    case_ids: Sequence[str] = DEFAULT_STRESS_CASE_IDS,
    full32: bool = False,
    mapper_checkpoint_path: str | Path | None = None,
    control_checkpoint_path: str | Path | None = None,
    device_name: str = "auto",
    max_tokens_per_window: int = 512,
    min_ln_duration_ms: int = 20,
    min_repeated_spacings: int = 4,
    min_spacing_ms: int = 40,
    max_spacing_ms: int = 400,
    seed: int = 1337,
    collect_logit_diagnostics: bool = False,
    timepoint_preview_limit: int = 4096,
) -> dict[str, Any]:
    source_summary = _read_json(Path(source_summary_path))
    source_cases = (
        [dict(row) for row in source_summary.get("case_results", []) if isinstance(row, Mapping)]
        if bool(full32)
        else select_case_results(source_summary, case_ids=case_ids)
    )
    if not source_cases:
        raise ValueError("no source cases selected")

    mapper_checkpoint = Path(mapper_checkpoint_path or str(source_summary.get("mapper_checkpoint_path", "")))
    control_checkpoint = Path(control_checkpoint_path or str(source_summary.get("control_checkpoint_path", "")))
    if not mapper_checkpoint.exists():
        raise FileNotFoundError(f"missing mapper checkpoint: {mapper_checkpoint}")
    if not control_checkpoint.exists():
        raise FileNotFoundError(f"missing control checkpoint: {control_checkpoint}")

    out_dir = Path(output_dir)
    rollout_dir = out_dir / ("full32_rollouts" if bool(full32) else "stress_rollouts")
    rollout_dir.mkdir(parents=True, exist_ok=True)
    vocab = MapperV21Vocab()
    case_results: list[dict[str, Any]] = []

    for source_row in source_cases:
        case_id = str(source_row["case_id"])
        chart_end_ms = int(source_row["chart_end_ms"])
        reference_times = _reference_times(Path(str(source_row["beatmap_path"])), chart_end_ms=chart_end_ms)
        for mode in ("baseline_guard", "anti_rigid_guard"):
            transform = None
            if mode == "anti_rigid_guard":
                transform = MapperV21AntiRigidSpacingLogitsTransform(
                    vocab=vocab,
                    min_repeated_spacings=int(min_repeated_spacings),
                    min_spacing_ms=int(min_spacing_ms),
                    max_spacing_ms=int(max_spacing_ms),
                    hard_block=True,
                )
            output_case_summary = rollout_dir / (
                f"{_safe_filename(case_id)}_{chart_end_ms}ms_{mode}_minln{int(min_ln_duration_ms)}.json"
            )
            candidate_summary = run_trained_v21_runtime_rollout_smoke(
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
                min_ln_duration_ms=int(min_ln_duration_ms),
            )
            candidate_metrics = generated_metrics(
                generated_times=[int(timepoint["time_ms"]) for timepoint in _generated_timepoints(candidate_summary)],
                reference_times=reference_times,
                chart_end_ms=chart_end_ms,
            )
            comparator_prefix = "v21_baseline" if mode == "baseline_guard" else "v21_guard"
            comparator_metrics = dict(_mapping(source_row.get(f"{comparator_prefix}_metrics")))
            case_results.append(
                {
                    "case_id": case_id,
                    "case_index": source_row.get("case_index"),
                    "mode": mode,
                    "difficulty": source_row.get("difficulty"),
                    "difficulty_band": source_row.get("difficulty_band"),
                    "audio_family": source_row.get("audio_family"),
                    "normalized_difficulty": source_row.get("normalized_difficulty"),
                    "chart_end_ms": chart_end_ms,
                    "audio_path": source_row.get("audio_path"),
                    "beatmap_path": source_row.get("beatmap_path"),
                    "summary_path": output_case_summary.as_posix(),
                    "candidate_legal": _candidate_legal(candidate_summary),
                    "candidate_rollout": rollout_status(candidate_summary),
                    "previous_legal": bool(source_row.get(f"{comparator_prefix}_legal")),
                    "previous_rollout": dict(_mapping(source_row.get(f"{comparator_prefix}_rollout"))),
                    "candidate_metrics": candidate_metrics,
                    "previous_metrics": comparator_metrics,
                    "candidate_vs_previous": compare_metrics(candidate_metrics, comparator_metrics),
                    "transform": {"enabled": False} if transform is None else transform.to_dict(),
                }
            )

    aggregate = aggregate_results(case_results)
    decision = decision_from_aggregate(aggregate, case_results, full32=bool(full32))
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Mapper v2.1 terminal LN-start guard repair",
        "decision": decision,
        "source_summary_path": Path(source_summary_path).as_posix(),
        "output_dir": out_dir.as_posix(),
        "mapper_checkpoint_path": mapper_checkpoint.as_posix(),
        "control_checkpoint_path": control_checkpoint.as_posix(),
        "config": {
            "device": device_name,
            "case_ids": [str(row["case_id"]) for row in source_cases],
            "full32": bool(full32),
            "max_tokens_per_window": int(max_tokens_per_window),
            "min_ln_duration_ms": int(min_ln_duration_ms),
            "min_repeated_spacings": int(min_repeated_spacings),
            "min_spacing_ms": int(min_spacing_ms),
            "max_spacing_ms": int(max_spacing_ms),
            "seed": int(seed),
            "collect_logit_diagnostics": bool(collect_logit_diagnostics),
            "timepoint_preview_limit": int(timepoint_preview_limit),
        },
        "source_decision": source_summary.get("decision"),
        "source_aggregate": source_summary.get("aggregate"),
        "aggregate": aggregate,
        "case_results": case_results,
        "next_step": decision["next_step"],
    }
    write_summary_json(summary, output_summary_path)
    if output_report_path is not None:
        write_report(summary, output_report_path)
    return summary


def aggregate_results(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    baseline_guard = _aggregate_mode(case_results, "baseline_guard")
    anti_rigid_guard = _aggregate_mode(case_results, "anti_rigid_guard")
    comparisons = [_mapping(row.get("candidate_vs_previous")) for row in case_results]
    return {
        "run_count": int(len(case_results)),
        "case_count": len({str(row.get("case_id")) for row in case_results}),
        "baseline_guard": baseline_guard,
        "anti_rigid_guard": anti_rigid_guard,
        "all_candidate_legal": all(bool(row.get("candidate_legal")) for row in case_results),
        "max_token_count": sum(1 for row in case_results if bool(_mapping(row.get("candidate_rollout")).get("max_tokens_exceeded"))),
        "new_starved_count": sum(1 for row in comparisons if bool(row.get("new_starved"))),
        "mean_f1_delta": _mean([_float(row.get("f1_delta")) or 0.0 for row in comparisons]),
        "mean_dominant_spacing_ratio_delta": _mean(
            [_float(row.get("dominant_spacing_ratio_delta")) or 0.0 for row in comparisons]
        ),
        "mean_event_count_ratio_delta": _mean([_float(row.get("event_count_ratio_delta")) or 0.0 for row in comparisons]),
        "total_anti_rigid_blocked_count": sum(
            int(_mapping(row.get("transform")).get("blocked_count") or 0) for row in case_results
        ),
    }


def decision_from_aggregate(
    aggregate: Mapping[str, Any],
    case_results: Sequence[Mapping[str, Any]],
    *,
    full32: bool,
) -> dict[str, Any]:
    case04_baseline = _case_mode(case_results, CASE_04_ID, "baseline_guard")
    case05_anti = _case_mode(case_results, CASE_05_ID, "anti_rigid_guard")
    primary_case04_legal = bool(_mapping(case04_baseline).get("candidate_legal"))
    primary_case05_legal = bool(_mapping(case05_anti).get("candidate_legal"))
    all_legal = bool(aggregate.get("all_candidate_legal"))
    max_token_count = int(aggregate.get("max_token_count") or 0)
    new_starved_count = int(aggregate.get("new_starved_count") or 0)
    mean_f1_delta = float(aggregate.get("mean_f1_delta") or 0.0)
    if not primary_case04_legal or not primary_case05_legal:
        route = "KILL"
        reason = "terminal LN-start guard did not fix both primary illegal traces"
        next_step = "Kill this min-duration repair and return to v3 structural grammar or a deeper v2.1 carry-state mutation."
    elif max_token_count > 0:
        route = "KILL"
        reason = "terminal LN-start guard introduced a max-token rollout"
        next_step = "Do not widen this guard; inspect the max-token trace before another mutation."
    elif not all_legal:
        route = "KILL"
        reason = "terminal LN-start guard failed legality on the selected slice"
        next_step = "Do not scale this guard; inspect the remaining illegal case traces."
    elif new_starved_count > 0:
        route = "KILL"
        reason = "terminal LN-start guard introduced new starvation versus the matching comparator"
        next_step = "Do not scale this guard; inspect whether blocked late LN starts are suppressing required events."
    elif mean_f1_delta < STRESS_F1_REGRESSION_LIMIT:
        route = "KILL"
        reason = "terminal LN-start guard regressed mean F1 beyond the stress gate"
        next_step = "Do not scale this guard; try only a bounded 20/30/40ms stress sweep if the owner wants to continue."
    elif bool(full32):
        route = "TEST_NEXT"
        reason = "terminal LN-start guard passed the full32 legality and quality gate"
        next_step = "Compare this opt-in v2.1 repair against current v3 rollout failures before changing defaults."
    else:
        route = "TEST_NEXT"
        reason = "terminal LN-start guard passed the four-case stress gate"
        next_step = "Run the optional full32 widening gate before changing defaults."
    return {
        "route": route,
        "reason": reason,
        "next_step": next_step,
        "primary_case04_baseline_legal": primary_case04_legal,
        "primary_case05_anti_rigid_legal": primary_case05_legal,
        "all_candidate_legal": all_legal,
        "max_token_count": max_token_count,
        "new_starved_count": new_starved_count,
        "mean_f1_delta": mean_f1_delta,
        "full32": bool(full32),
    }


def write_summary_json(summary: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(summary: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_markdown(summary).rstrip() + "\n", encoding="utf-8")


def report_markdown(summary: Mapping[str, Any]) -> str:
    decision = _mapping(summary.get("decision"))
    aggregate = _mapping(summary.get("aggregate"))
    baseline = _mapping(aggregate.get("baseline_guard"))
    anti = _mapping(aggregate.get("anti_rigid_guard"))
    lines = [
        "# Mapper v2.1 Terminal LN-Start Guard Repair Result Report",
        "",
        "## Scope",
        "",
        "This eval reruns v2.1 fixed-slice real-audio cases with the existing grammar `min_ln_duration_ms` primitive exposed through runtime rollout. The guard is opt-in and mapper defaults remain unchanged.",
        "",
        "## Decision",
        "",
        f"Decision: `{decision.get('route')}`.",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Full32: `{decision.get('full32')}`",
        f"- Runs: `{aggregate.get('run_count')}`",
        f"- Cases: `{aggregate.get('case_count')}`",
        f"- Primary case 04 baseline legal: `{decision.get('primary_case04_baseline_legal')}`",
        f"- Primary case 05 anti-rigid legal: `{decision.get('primary_case05_anti_rigid_legal')}`",
        f"- All candidate legal: `{aggregate.get('all_candidate_legal')}`",
        f"- Max-token count: `{aggregate.get('max_token_count')}`",
        f"- New starved count: `{aggregate.get('new_starved_count')}`",
        f"- Mean F1 delta vs matching comparator: `{_fmt(aggregate.get('mean_f1_delta'))}`",
        f"- Anti-rigid blocked count: `{aggregate.get('total_anti_rigid_blocked_count')}`",
        "",
        "## Aggregate Table",
        "",
        "| Mode | Legal | Mean F1 | Starved | Mean rigid | Mean event ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        _mode_row("baseline + terminal guard", baseline),
        _mode_row("anti-rigid + terminal guard", anti),
        "",
        "## What Passed",
        "",
        f"- The eval completed `{aggregate.get('run_count')}` real-audio guarded rollouts.",
        f"- The terminal guard used `min_ln_duration_ms={_mapping(summary.get('config')).get('min_ln_duration_ms')}` and remained opt-in.",
        f"- Primary case `04` baseline legality: `{decision.get('primary_case04_baseline_legal')}`.",
        f"- Primary case `05` anti-rigid legality: `{decision.get('primary_case05_anti_rigid_legal')}`.",
        "",
        "## What Surfaced",
        "",
        _interpretation(summary),
        "",
        "## Case Table",
        "",
        "| Case | Mode | Previous legal | Candidate legal | Previous F1/Rigid/Starved | Candidate F1/Rigid/Starved | F1 delta | Terminal ms | Tokens | Blocks |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.get("case_results", []):
        if not isinstance(row, Mapping):
            continue
        previous_metrics = _mapping(row.get("previous_metrics"))
        candidate_metrics = _mapping(row.get("candidate_metrics"))
        comparison = _mapping(row.get("candidate_vs_previous"))
        rollout = _mapping(row.get("candidate_rollout"))
        transform = _mapping(row.get("transform"))
        lines.append(
            "| {case} | `{mode}` | {prev_legal} | {cand_legal} | {prev_metrics} | {cand_metrics} | {f1_delta} | {terminal} | {tokens} | {blocks} |".format(
                case=f"`{row.get('case_id')}`",
                mode=row.get("mode"),
                prev_legal=row.get("previous_legal"),
                cand_legal=row.get("candidate_legal"),
                prev_metrics=_metric_cell(previous_metrics),
                cand_metrics=_metric_cell(candidate_metrics),
                f1_delta=_fmt(comparison.get("f1_delta")),
                terminal=rollout.get("terminal_ms"),
                tokens=rollout.get("token_count"),
                blocks=transform.get("blocked_count", 0),
            )
        )
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            str(summary.get("next_step")),
            "",
        ]
    )
    return "\n".join(lines)


def _aggregate_mode(case_results: Sequence[Mapping[str, Any]], mode: str) -> dict[str, Any]:
    rows = [row for row in case_results if str(row.get("mode")) == mode]
    metrics = [_mapping(row.get("candidate_metrics")) for row in rows]
    legal_count = sum(1 for row in rows if bool(row.get("candidate_legal")))
    return {
        "run_count": int(len(rows)),
        "legal_count": int(legal_count),
        "all_legal": int(legal_count) == int(len(rows)),
        "mean_f1_100ms": _mean([_metric_f1(row) for row in metrics]),
        "mean_dominant_spacing_ratio": _mean([_float(row.get("dominant_spacing_ratio")) or 0.0 for row in metrics]),
        "mean_event_count_ratio": _mean([_float(row.get("event_count_ratio")) or 0.0 for row in metrics]),
        "starved_count": sum(1 for row in metrics if bool(row.get("starved"))),
    }


def _mode_row(label: str, aggregate: Mapping[str, Any]) -> str:
    return (
        f"| `{label}` | `{aggregate.get('legal_count')}` / `{aggregate.get('run_count')}` | "
        f"{_fmt(aggregate.get('mean_f1_100ms'))} | {aggregate.get('starved_count')} | "
        f"{_fmt(aggregate.get('mean_dominant_spacing_ratio'))} | {_fmt(aggregate.get('mean_event_count_ratio'))} |"
    )


def _interpretation(summary: Mapping[str, Any]) -> str:
    decision = _mapping(summary.get("decision"))
    route = str(decision.get("route"))
    if route == "TEST_NEXT" and bool(decision.get("full32")):
        return "The terminal guard passed the widened fixed-slice gate. It remains an opt-in grammar hardening result until compared against v3 and v2.1 default behavior in a replacement decision."
    if route == "TEST_NEXT":
        return "The terminal guard passed the stress gate. The next required check is the full32 widening run before changing defaults."
    return "The terminal guard failed a predefined legality, starvation, max-token, or F1 gate and should not be scaled as-is."


def _case_mode(case_results: Sequence[Mapping[str, Any]], case_id: str, mode: str) -> Mapping[str, Any]:
    return next(
        (row for row in case_results if str(row.get("case_id")) == str(case_id) and str(row.get("mode")) == str(mode)),
        {},
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


def _metric_cell(metrics: Mapping[str, Any]) -> str:
    return "{f1}/{rigid}/{starved}".format(
        f1=_fmt(_metric_f1(metrics)),
        rigid=_fmt(metrics.get("dominant_spacing_ratio")),
        starved=metrics.get("starved"),
    )


def _metric_f1(metrics: Mapping[str, Any]) -> float:
    return _float(_mapping(metrics.get("timing_match_100ms")).get("f1")) or 0.0


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


def _fmt(value: object) -> str:
    number = _float(value)
    return "n/a" if number is None else f"{number:.6f}"


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value)).strip("_") or "case"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the v2.1 terminal LN-start guard repair gate.")
    parser.add_argument("--source-summary", default=DEFAULT_SOURCE_SUMMARY_PATH)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--report-output", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--full32", action="store_true")
    parser.add_argument("--mapper-checkpoint-path")
    parser.add_argument("--control-checkpoint-path")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--max-tokens-per-window", type=int, default=512)
    parser.add_argument("--min-ln-duration-ms", type=int, default=20)
    parser.add_argument("--min-repeated-spacings", type=int, default=4)
    parser.add_argument("--min-spacing-ms", type=int, default=40)
    parser.add_argument("--max-spacing-ms", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--collect-logit-diagnostics", action="store_true")
    parser.add_argument("--timepoint-preview-limit", type=int, default=4096)
    args = parser.parse_args(argv)
    summary = run_terminal_ln_start_guard_repair(
        source_summary_path=args.source_summary,
        output_summary_path=args.summary_output,
        output_report_path=args.report_output,
        output_dir=args.output_dir,
        case_ids=tuple(args.case_ids or DEFAULT_STRESS_CASE_IDS),
        full32=args.full32,
        mapper_checkpoint_path=args.mapper_checkpoint_path,
        control_checkpoint_path=args.control_checkpoint_path,
        device_name=args.device,
        max_tokens_per_window=args.max_tokens_per_window,
        min_ln_duration_ms=args.min_ln_duration_ms,
        min_repeated_spacings=args.min_repeated_spacings,
        min_spacing_ms=args.min_spacing_ms,
        max_spacing_ms=args.max_spacing_ms,
        seed=args.seed,
        collect_logit_diagnostics=args.collect_logit_diagnostics,
        timepoint_preview_limit=args.timepoint_preview_limit,
    )
    print(
        "mapper_v21_terminal_ln_start_guard_repair_done "
        f"route={summary['decision']['route']} "
        f"runs={summary['aggregate']['run_count']} "
        f"all_legal={summary['aggregate']['all_candidate_legal']} "
        f"new_starved={summary['aggregate']['new_starved_count']} "
        f"mean_f1_delta={summary['aggregate']['mean_f1_delta']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
