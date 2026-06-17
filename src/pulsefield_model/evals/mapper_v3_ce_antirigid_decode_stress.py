from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from pulsefield_model.evals.mapper_v2_1_anti_rigid_spacing_guard import generated_metrics
from pulsefield_model.evals.mapper_v3_trained_runtime_rollout import run_trained_v3_runtime_rollout_smoke
from pulsefield_model.inference.mapper_v3_rollout import MapperV3AntiRigidSpacingLogitsTransform
from pulsefield_model.models.mapper.shared.tokenizer import hitobjects_to_mapper_timepoints
from pulsefield_model.models.mapper.v3 import MapperV3Vocab
from pulsefield_model.osu_core.hitobjects import parse_mania_hit_objects


SUMMARY_SCHEMA_VERSION = 1
RIGID_DOMINANT_SPACING_RATIO_THRESHOLD = 0.95
DEFAULT_BASELINE_SUMMARY_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/"
    "target_grammar_v3_event_token_ce_weight_training_gate_summary.json",
)
DEFAULT_OUTPUT_DIR = Path("artifacts/tmp/mapper_v3_ce_antirigid_decode_stress")
DEFAULT_SUMMARY_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_antirigid_decode_stress_summary.json",
)
DEFAULT_REPORT_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/"
    "target_grammar_v3_ce_antirigid_decode_stress_result_report.md",
)
DEFAULT_EXPERIMENT_CARD_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/"
    "target_grammar_v3_ce_antirigid_decode_stress_experiment_card.md",
)


def run_v3_ce_antirigid_decode_stress(
    *,
    baseline_summary_path: str | Path = DEFAULT_BASELINE_SUMMARY_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    output_summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    output_report_path: str | Path | None = DEFAULT_REPORT_PATH,
    device_name: str = "mps",
    min_repeated_spacings: int = 4,
    min_spacing_ms: int = 40,
    max_spacing_ms: int = 400,
    hard_block: bool = True,
    penalty: float = 8.0,
    require_time_shift_alternative: bool = True,
    require_tap_only_run: bool = False,
    max_examples: int = 12,
    max_tokens_per_window: int = 512,
    seed: int = 1337,
    case_limit: int | None = None,
) -> dict[str, Any]:
    baseline_summary_path = Path(baseline_summary_path)
    baseline_summary = _read_json(baseline_summary_path)
    stress_runs = select_rigid_stress_runs(baseline_summary)
    if case_limit is not None:
        limit = int(case_limit)
        if limit <= 0:
            raise ValueError("case_limit must be positive when provided")
        stress_runs = stress_runs[:limit]
    if not stress_runs:
        raise ValueError("baseline summary has no rigid stress cases")

    mapper_checkpoint = Path(str(baseline_summary.get("checkpoint_path", "")))
    control_checkpoint = Path(str(baseline_summary.get("control_checkpoint_path", "")))
    if not mapper_checkpoint.exists():
        raise FileNotFoundError(f"missing mapper checkpoint: {mapper_checkpoint}")
    if not control_checkpoint.exists():
        raise FileNotFoundError(f"missing control checkpoint: {control_checkpoint}")

    out_dir = Path(output_dir)
    rollout_dir = out_dir / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    vocab = MapperV3Vocab()
    case_results: list[dict[str, Any]] = []

    for index, baseline_row in enumerate(stress_runs, start=1):
        case_id = str(baseline_row["case_id"])
        chart_end_ms = int(baseline_row["chart_end_ms"])
        print(f"stress rollout {index:02d}/{len(stress_runs)} {case_id}", flush=True)
        transform = MapperV3AntiRigidSpacingLogitsTransform(
            vocab=vocab,
            min_repeated_spacings=int(min_repeated_spacings),
            min_spacing_ms=int(min_spacing_ms),
            max_spacing_ms=int(max_spacing_ms),
            hard_block=bool(hard_block),
            penalty=float(penalty),
            require_time_shift_alternative=bool(require_time_shift_alternative),
            require_tap_only_run=bool(require_tap_only_run),
            max_examples=int(max_examples),
        )
        summary_path = rollout_dir / f"{_safe_filename(case_id)}_{chart_end_ms}ms_antirigid_summary.json"
        report_path = rollout_dir / f"{_safe_filename(case_id)}_{chart_end_ms}ms_antirigid_report.md"
        rollout_summary = run_trained_v3_runtime_rollout_smoke(
            mapper_checkpoint_path=mapper_checkpoint,
            control_checkpoint_path=control_checkpoint,
            output_summary_path=summary_path,
            output_report_path=report_path,
            device_name=device_name,
            chart_end_ms=chart_end_ms,
            max_tokens_per_window=int(max_tokens_per_window),
            audio_path=str(baseline_row["audio_path"]),
            real_audio=True,
            normalized_difficulty=float(baseline_row["normalized_difficulty"]),
            temperature=0.0,
            top_p=None,
            seed=int(seed),
            time_shift_length_penalty_alpha=0.0,
            time_shift_delta_penalty_alpha=0.0,
            collect_logit_diagnostics=True,
            logit_top_k=5,
            logit_max_examples=int(max_examples),
            logits_transform=transform,
            timepoint_preview_limit=4096,
        )
        metrics = _metrics_from_rollout_summary(
            rollout_summary,
            beatmap_path=Path(str(baseline_row["beatmap_path"])),
            chart_end_ms=chart_end_ms,
        )
        case_result = {
            "case_id": case_id,
            "case_index": baseline_row.get("case_index"),
            "audio_family": baseline_row.get("audio_family"),
            "difficulty": baseline_row.get("difficulty"),
            "difficulty_band": baseline_row.get("difficulty_band"),
            "normalized_difficulty": baseline_row.get("normalized_difficulty"),
            "chart_end_ms": chart_end_ms,
            "audio_path": baseline_row.get("audio_path"),
            "beatmap_path": baseline_row.get("beatmap_path"),
            "summary_path": summary_path.as_posix(),
            "report_path": report_path.as_posix(),
            "baseline_metrics": _baseline_metrics(baseline_row),
            "candidate_metrics": metrics,
            "delta_vs_baseline": compare_metrics(metrics, _baseline_metrics(baseline_row)),
            "baseline_legal": bool(baseline_row.get("legal")),
            "candidate_legal": _candidate_legal(rollout_summary),
            "candidate_rollout": _rollout_status(rollout_summary),
            "transform": transform.to_dict(),
        }
        case_results.append(case_result)
        if index <= 3:
            rollout_status = case_result["candidate_rollout"]
            if bool(rollout_status["dead_end"]) or bool(rollout_status["max_tokens_exceeded"]):
                break

    aggregate = aggregate_results(case_results)
    decision = decision_from_aggregate(aggregate)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar v3 CE anti-rigid decode stress",
        "experiment_card": DEFAULT_EXPERIMENT_CARD_PATH.as_posix(),
        "baseline_summary_path": baseline_summary_path.as_posix(),
        "output_dir": out_dir.as_posix(),
        "mapper_checkpoint_path": mapper_checkpoint.as_posix(),
        "control_checkpoint_path": control_checkpoint.as_posix(),
        "code_commit": _git_stdout("rev-parse", "HEAD"),
        "code_dirty": bool(_git_stdout("status", "--short")),
        "config": {
            "device": device_name,
            "min_repeated_spacings": int(min_repeated_spacings),
            "min_spacing_ms": int(min_spacing_ms),
            "max_spacing_ms": int(max_spacing_ms),
            "hard_block": bool(hard_block),
            "penalty": float(penalty),
            "require_time_shift_alternative": bool(require_time_shift_alternative),
            "require_tap_only_run": bool(require_tap_only_run),
            "max_examples": int(max_examples),
            "max_tokens_per_window": int(max_tokens_per_window),
            "seed": int(seed),
            "case_limit": None if case_limit is None else int(case_limit),
        },
        "stress_case_count": len(stress_runs),
        "completed_case_count": len(case_results),
        "decision": decision,
        "aggregate": aggregate,
        "case_results": case_results,
        "worst_cases": worst_cases(case_results),
        "next_step": decision["next_step"],
    }
    write_summary_json(summary, output_summary_path)
    if output_report_path is not None:
        write_report(summary, output_report_path)
    return summary


def select_rigid_stress_runs(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in summary.get("runs", ()):
        if not isinstance(row, Mapping):
            continue
        if (_float(row.get("dominant_spacing_ratio")) or 0.0) >= RIGID_DOMINANT_SPACING_RATIO_THRESHOLD:
            rows.append(dict(row))
    return sorted(rows, key=lambda item: (int(item.get("case_index") or 0), str(item.get("case_id"))))


def compare_metrics(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dominant_spacing_ratio_delta": (_float(candidate.get("dominant_spacing_ratio")) or 0.0)
        - (_float(baseline.get("dominant_spacing_ratio")) or 0.0),
        "dominant_spacing_improved": (_float(candidate.get("dominant_spacing_ratio")) or 0.0)
        < (_float(baseline.get("dominant_spacing_ratio")) or 0.0),
        "f1_delta": _metric_f1(candidate) - _metric_f1(baseline),
        "event_count_ratio_delta": (_float(candidate.get("event_count_ratio")) or 0.0)
        - (_float(baseline.get("event_count_ratio")) or 0.0),
        "second_window_event_share_delta": (_float(candidate.get("second_window_event_share")) or 0.0)
        - (_float(baseline.get("second_window_event_share")) or 0.0),
        "candidate_starved": bool(candidate.get("starved")),
        "baseline_starved": bool(baseline.get("starved")),
        "new_starved": bool(candidate.get("starved")) and not bool(baseline.get("starved")),
    }


def aggregate_results(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    baseline_metrics = [_mapping(row.get("baseline_metrics")) for row in case_results]
    candidate_metrics = [_mapping(row.get("candidate_metrics")) for row in case_results]
    comparisons = [_mapping(row.get("delta_vs_baseline")) for row in case_results]
    transform_rows = [_mapping(row.get("transform")) for row in case_results]
    baseline = aggregate_metrics(baseline_metrics, legal=[bool(row.get("baseline_legal")) for row in case_results])
    candidate = aggregate_metrics(candidate_metrics, legal=[bool(row.get("candidate_legal")) for row in case_results])
    baseline_rigid = int(baseline.get("rigid_case_count") or 0)
    candidate_rigid = int(candidate.get("rigid_case_count") or 0)
    return {
        "case_count": len(case_results),
        "baseline": baseline,
        "candidate": candidate,
        "rigid_case_delta": candidate_rigid - baseline_rigid,
        "rigid_case_reduction": baseline_rigid - candidate_rigid,
        "starved_case_delta": int(candidate.get("starved_count") or 0) - int(baseline.get("starved_count") or 0),
        "mean_f1_delta": float(candidate.get("mean_f1_100ms") or 0.0) - float(baseline.get("mean_f1_100ms") or 0.0),
        "mean_dominant_spacing_ratio_delta": float(candidate.get("mean_dominant_spacing_ratio") or 0.0)
        - float(baseline.get("mean_dominant_spacing_ratio") or 0.0),
        "rigid_improved_count": sum(1 for row in comparisons if bool(row.get("dominant_spacing_improved"))),
        "new_starved_count": sum(1 for row in comparisons if bool(row.get("new_starved"))),
        "total_transform_blocks": sum(int(row.get("blocked_count") or 0) for row in transform_rows),
        "total_transform_candidates": sum(int(row.get("candidate_count") or 0) for row in transform_rows),
        "dead_end_count": sum(1 for row in case_results if bool(_mapping(row.get("candidate_rollout")).get("dead_end"))),
        "max_token_count": sum(
            1 for row in case_results if bool(_mapping(row.get("candidate_rollout")).get("max_tokens_exceeded"))
        ),
    }


def aggregate_metrics(rows: Sequence[Mapping[str, Any]], *, legal: Sequence[bool]) -> dict[str, Any]:
    dominant = [_float(row.get("dominant_spacing_ratio")) or 0.0 for row in rows]
    event_ratios = [_float(row.get("event_count_ratio")) or 0.0 for row in rows]
    second = [_float(row.get("second_window_event_share")) or 0.0 for row in rows]
    f1 = [_metric_f1(row) for row in rows]
    boundary = [_float(row.get("boundary_event_ratio")) or 0.0 for row in rows]
    return {
        "case_count": len(rows),
        "legal_count": sum(1 for value in legal if bool(value)),
        "all_legal": bool(rows) and all(bool(value) for value in legal),
        "rigid_case_count": sum(1 for value in dominant if value >= RIGID_DOMINANT_SPACING_RATIO_THRESHOLD),
        "starved_count": sum(1 for row in rows if bool(row.get("starved"))),
        "mean_dominant_spacing_ratio": _mean(dominant),
        "median_dominant_spacing_ratio": _median(dominant),
        "mean_event_count_ratio": _mean(event_ratios),
        "median_event_count_ratio": _median(event_ratios),
        "mean_second_window_share": _mean(second),
        "median_second_window_share": _median(second),
        "mean_f1_100ms": _mean(f1),
        "median_f1_100ms": _median(f1),
        "max_boundary_event_ratio": max(boundary, default=0.0),
    }


def decision_from_aggregate(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    dead_end_count = int(aggregate.get("dead_end_count") or 0)
    max_token_count = int(aggregate.get("max_token_count") or 0)
    rigid_reduction = int(aggregate.get("rigid_case_reduction") or 0)
    starved_delta = int(aggregate.get("starved_case_delta") or 0)
    mean_f1_delta = float(aggregate.get("mean_f1_delta") or 0.0)
    if dead_end_count > 0 or max_token_count > 0:
        return {
            "route": "KILL",
            "reason": "anti-rigid hard block created a dead-end or max-token stress failure",
            "next_step": "Do not scale this hard-block transform; inspect the failing stress rollout.",
        }
    if rigid_reduction >= 4 and starved_delta <= 0 and mean_f1_delta >= -0.03:
        return {
            "route": "TEST_NEXT",
            "reason": "anti-rigid hard block reduced stress rigidity without reviving starvation",
            "next_step": "Run the transform on the full 32-case CE-weight fixed slice before any default change.",
        }
    if rigid_reduction < 2 or starved_delta > 0 or mean_f1_delta < -0.03:
        return {
            "route": "KILL",
            "reason": "anti-rigid hard block failed a stress kill criterion",
            "next_step": "Stop hard-block decode suppression and mutate to finite penalty, tap-only, grammar-level timing diversity, or v2.1 grammar work.",
        }
    return {
        "route": "MUTATE",
        "reason": "anti-rigid hard block has partial stress signal but does not pass the positive gate",
        "next_step": "Mutate to finite penalty or tap-only stress probe before a full 32-case run.",
    }


def worst_cases(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def project(row: Mapping[str, Any]) -> dict[str, Any]:
        candidate = _mapping(row.get("candidate_metrics"))
        baseline = _mapping(row.get("baseline_metrics"))
        return {
            "case_id": row.get("case_id"),
            "baseline_rigid": baseline.get("dominant_spacing_ratio"),
            "candidate_rigid": candidate.get("dominant_spacing_ratio"),
            "baseline_f1": _metric_f1(baseline),
            "candidate_f1": _metric_f1(candidate),
            "candidate_second_window_share": candidate.get("second_window_event_share"),
            "candidate_event_count_ratio": candidate.get("event_count_ratio"),
            "transform_blocks": _mapping(row.get("transform")).get("blocked_count"),
        }

    return {
        "highest_candidate_rigid": [
            project(row)
            for row in sorted(
                case_results,
                key=lambda item: _float(_mapping(item.get("candidate_metrics")).get("dominant_spacing_ratio")) or 0.0,
                reverse=True,
            )[:8]
        ],
        "largest_f1_regression": [
            project(row)
            for row in sorted(
                case_results,
                key=lambda item: _float(_mapping(item.get("delta_vs_baseline")).get("f1_delta")) or 0.0,
            )[:8]
        ],
    }


def write_report(summary: Mapping[str, Any], path: str | Path) -> None:
    aggregate = _mapping(summary.get("aggregate"))
    baseline = _mapping(aggregate.get("baseline"))
    candidate = _mapping(aggregate.get("candidate"))
    decision = _mapping(summary.get("decision"))
    lines = [
        "# Target Grammar v3 CE Anti-Rigid Decode Stress Result Report",
        "",
        "## Scope",
        "",
        "This pass reruns only the rigid cases from the committed CE-weight v3 gate with an opt-in v3 anti-rigid logits transform. It does not change mapper defaults, model weights, tokenization, or training.",
        "",
        "## Decision",
        "",
        f"- route: `{decision.get('route')}`",
        f"- reason: {decision.get('reason')}",
        "",
        "## Aggregate",
        "",
        "| Metric | CE baseline stress | Anti-rigid candidate | Delta |",
        "| --- | ---: | ---: | ---: |",
        _metric_row("rigid cases", baseline.get("rigid_case_count"), candidate.get("rigid_case_count"), aggregate.get("rigid_case_delta")),
        _metric_row("starved cases", baseline.get("starved_count"), candidate.get("starved_count"), aggregate.get("starved_case_delta")),
        _metric_row("mean F1@100ms", baseline.get("mean_f1_100ms"), candidate.get("mean_f1_100ms"), aggregate.get("mean_f1_delta")),
        _metric_row("mean dominant spacing", baseline.get("mean_dominant_spacing_ratio"), candidate.get("mean_dominant_spacing_ratio"), aggregate.get("mean_dominant_spacing_ratio_delta")),
        _metric_row("mean event ratio", baseline.get("mean_event_count_ratio"), candidate.get("mean_event_count_ratio"), None),
        "",
        "## Transform",
        "",
        f"- total candidates: `{aggregate.get('total_transform_candidates')}`",
        f"- total blocks: `{aggregate.get('total_transform_blocks')}`",
        f"- dead-end cases: `{aggregate.get('dead_end_count')}`",
        f"- max-token cases: `{aggregate.get('max_token_count')}`",
        "",
        "## Worst Cases",
        "",
        "Highest candidate rigid ratios:",
    ]
    for row in _mapping(summary.get("worst_cases")).get("highest_candidate_rigid", [])[:5]:
        lines.append(
            "- `{case}`: rigid `{base}` -> `{cand}`, f1 `{bf1}` -> `{cf1}`, blocks `{blocks}`".format(
                case=row.get("case_id"),
                base=_fmt(row.get("baseline_rigid")),
                cand=_fmt(row.get("candidate_rigid")),
                bf1=_fmt(row.get("baseline_f1")),
                cf1=_fmt(row.get("candidate_f1")),
                blocks=row.get("transform_blocks"),
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
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _metric_row(label: str, baseline: Any, candidate: Any, delta: Any) -> str:
    return f"| {label} | `{_fmt(baseline)}` | `{_fmt(candidate)}` | `{_fmt(delta)}` |"


def _metrics_from_rollout_summary(
    summary: Mapping[str, Any],
    *,
    beatmap_path: Path,
    chart_end_ms: int,
) -> dict[str, Any]:
    rollout = _mapping(summary.get("rollout"))
    timepoints = rollout.get("timepoints")
    if not isinstance(timepoints, list):
        raise ValueError("rollout summary does not contain timepoints")
    timepoint_count = int(rollout.get("timepoint_count") or 0)
    if len(timepoints) < timepoint_count:
        raise ValueError(f"rollout timepoints preview is truncated: {len(timepoints)} < {timepoint_count}")
    generated_times = [int(timepoint["time_ms"]) for timepoint in timepoints]
    reference_times = _reference_times(beatmap_path, chart_end_ms=chart_end_ms)
    return generated_metrics(
        generated_times=generated_times,
        reference_times=reference_times,
        chart_end_ms=int(chart_end_ms),
    )


def _baseline_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "generated_event_count": int(row.get("generated_event_count") or 0),
        "reference_event_count": int(row.get("reference_event_count") or 0),
        "event_count_ratio": _float(row.get("event_count_ratio")) or 0.0,
        "second_window_event_count": int(row.get("second_window_event_count") or 0),
        "second_window_event_share": _float(row.get("second_window_event_share")) or 0.0,
        "reference_second_window_event_share": _float(row.get("reference_second_window_event_share")) or 0.0,
        "starved": bool(row.get("starved")),
        "boundary_event_ratio": _float(row.get("boundary_event_ratio")) or 0.0,
        "dominant_spacing_ms": row.get("dominant_spacing_ms"),
        "dominant_spacing_ratio": _float(row.get("dominant_spacing_ratio")) or 0.0,
        "timing_match_100ms": dict(_mapping(row.get("timing_match_100ms"))),
    }


def _reference_times(beatmap_path: Path, *, chart_end_ms: int) -> list[int]:
    hitobjects = parse_mania_hit_objects(beatmap_path, expected_key_count=4)
    return [
        int(timepoint.time_ms)
        for timepoint in hitobjects_to_mapper_timepoints(hitobjects)
        if 0 <= int(timepoint.time_ms) <= int(chart_end_ms)
    ]


def _candidate_legal(summary: Mapping[str, Any]) -> bool:
    checks = _mapping(summary.get("checks"))
    return bool(checks) and all(bool(value) for value in checks.values())


def _rollout_status(summary: Mapping[str, Any]) -> dict[str, Any]:
    rollout = _mapping(summary.get("rollout"))
    windows = rollout.get("windows")
    terminal_ms = None
    if isinstance(windows, list) and windows:
        terminal_ms = _mapping(windows[-1]).get("terminal_ms")
    return {
        "completed": bool(rollout.get("completed")),
        "dead_end": bool(rollout.get("dead_end")),
        "max_tokens_exceeded": bool(rollout.get("max_tokens_exceeded")),
        "window_count": int(rollout.get("window_count") or 0),
        "token_count": int(rollout.get("token_count") or 0),
        "timepoint_count": int(rollout.get("timepoint_count") or 0),
        "terminal_ms": terminal_ms,
    }


def write_summary_json(summary: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _metric_f1(row: Mapping[str, Any]) -> float:
    timing = _mapping(row.get("timing_match_100ms"))
    return _float(timing.get("f1")) or 0.0


def _mean(values: Sequence[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return float(math.fsum(clean) / len(clean)) if clean else 0.0


def _median(values: Sequence[float]) -> float:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    return float(statistics.median(clean)) if clean else 0.0


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value).lower()).strip("_")


def _git_stdout(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the v3 CE anti-rigid decode stress probe.")
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--device", default="mps", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--min-repeated-spacings", type=int, default=4)
    parser.add_argument("--min-spacing-ms", type=int, default=40)
    parser.add_argument("--max-spacing-ms", type=int, default=400)
    parser.add_argument("--soft-penalty", action="store_true")
    parser.add_argument("--penalty", type=float, default=8.0)
    parser.add_argument("--allow-no-time-shift-alternative", action="store_true")
    parser.add_argument("--require-tap-only-run", action="store_true")
    parser.add_argument("--max-examples", type=int, default=12)
    parser.add_argument("--max-tokens-per-window", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--case-limit", type=int)
    args = parser.parse_args(argv)
    summary = run_v3_ce_antirigid_decode_stress(
        baseline_summary_path=args.baseline_summary,
        output_dir=args.output_dir,
        output_summary_path=args.summary_output,
        output_report_path=args.report_output,
        device_name=args.device,
        min_repeated_spacings=args.min_repeated_spacings,
        min_spacing_ms=args.min_spacing_ms,
        max_spacing_ms=args.max_spacing_ms,
        hard_block=not bool(args.soft_penalty),
        penalty=args.penalty,
        require_time_shift_alternative=not bool(args.allow_no_time_shift_alternative),
        require_tap_only_run=args.require_tap_only_run,
        max_examples=args.max_examples,
        max_tokens_per_window=args.max_tokens_per_window,
        seed=args.seed,
        case_limit=args.case_limit,
    )
    print(
        "mapper_v3_ce_antirigid_decode_stress_done "
        f"route={summary['decision']['route']} "
        f"cases={summary['completed_case_count']} "
        f"rigid_delta={summary['aggregate']['rigid_case_delta']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
