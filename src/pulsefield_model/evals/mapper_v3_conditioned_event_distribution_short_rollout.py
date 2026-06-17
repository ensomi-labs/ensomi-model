from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from pulsefield_model.evals.mapper_v3_ce_antirigid_decode_stress import (
    _baseline_metrics,
    _candidate_legal,
    _git_stdout,
    _metrics_from_rollout_summary,
    _rollout_status,
    _safe_filename,
    aggregate_results,
    compare_metrics,
    write_summary_json,
)
from pulsefield_model.evals.mapper_v3_trained_runtime_rollout import run_trained_v3_runtime_rollout_smoke
from pulsefield_model.training.mapper_v3 import load_run_config, run_mapper_v3_phase_b_training


SUMMARY_SCHEMA_VERSION = 1
REPORT_ROOT = Path("artifacts/reports/audits/mapper_v2_1_grammar")
DEFAULT_STAGE1_SUMMARY_PATH = REPORT_ROOT / "target_grammar_v3_conditioned_event_distribution_objective_summary.json"
DEFAULT_CE_BASELINE_SUMMARY_PATH = REPORT_ROOT / "target_grammar_v3_event_token_ce_weight_training_gate_summary.json"
DEFAULT_PRE_CE_BASELINE_SUMMARY_PATH = REPORT_ROOT / "target_grammar_v3_500step_fixed_slice_wide_audit_summary.json"
DEFAULT_BASE_TRAINING_CONFIG_PATH = REPORT_ROOT / "target_grammar_v3_event_token_ce_weight_training_gate_enabled.yaml"
DEFAULT_OUTPUT_DIR = Path("artifacts/tmp/mapper_v3_conditioned_event_distribution_short_rollout")
DEFAULT_SUMMARY_PATH = REPORT_ROOT / "target_grammar_v3_conditioned_event_distribution_short_rollout_summary.json"
DEFAULT_REPORT_PATH = REPORT_ROOT / "target_grammar_v3_conditioned_event_distribution_short_rollout_result_report.md"
DEFAULT_EXPERIMENT_CARD_PATH = (
    REPORT_ROOT / "target_grammar_v3_conditioned_event_distribution_short_rollout_gate_experiment_card.md"
)
DEFAULT_LOSS_OVERRIDES: dict[str, float] = {
    "lambda_conditioned_event_distribution": 0.05,
    "conditioned_event_under_weight": 3.0,
    "conditioned_event_over_weight": 1.0,
    "conditioned_event_zero_target_over_weight": 2.0,
    "conditioned_event_high_difficulty_over_weight": 4.0,
    "conditioned_event_high_difficulty_min": 0.75,
    "event_token_loss_weight": 1.0,
}
CE_BASELINE_STARVED_CASES = 5
CE_BASELINE_RIGID_CASES = 11
MEDIAN_EVENT_RATIO_MIN = 0.80
MEDIAN_EVENT_RATIO_MAX = 1.25


def run_conditioned_event_distribution_short_rollout_gate(
    *,
    stage1_summary_path: str | Path = DEFAULT_STAGE1_SUMMARY_PATH,
    ce_baseline_summary_path: str | Path = DEFAULT_CE_BASELINE_SUMMARY_PATH,
    pre_ce_baseline_summary_path: str | Path = DEFAULT_PRE_CE_BASELINE_SUMMARY_PATH,
    base_training_config_path: str | Path = DEFAULT_BASE_TRAINING_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    output_summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    output_report_path: str | Path | None = DEFAULT_REPORT_PATH,
    experiment_card_path: str | Path = DEFAULT_EXPERIMENT_CARD_PATH,
    mapper_checkpoint_path: str | Path | None = None,
    control_checkpoint_path: str | Path | None = None,
    device_name: str = "mps",
    max_steps: int = 500,
    max_tokens_per_window: int = 512,
    seed: int = 1337,
    case_limit: int | None = None,
    reuse_existing_checkpoint: bool = False,
) -> dict[str, Any]:
    start = time.monotonic()
    stage1_summary_path = Path(stage1_summary_path)
    ce_baseline_summary_path = Path(ce_baseline_summary_path)
    pre_ce_baseline_summary_path = Path(pre_ce_baseline_summary_path)
    base_training_config_path = Path(base_training_config_path)
    out_dir = Path(output_dir)
    stage1_summary = _read_json(stage1_summary_path)
    ce_baseline_summary = _read_json(ce_baseline_summary_path)
    pre_ce_baseline_summary = _read_json(pre_ce_baseline_summary_path)
    _require_stage1_route(stage1_summary)
    selected_runs = select_short_rollout_cases(ce_baseline_summary)
    if case_limit is not None:
        limit = int(case_limit)
        if limit <= 0:
            raise ValueError("case_limit must be positive when provided")
        selected_runs = selected_runs[:limit]
    if not selected_runs:
        raise ValueError("CE baseline summary has no selected short-rollout cases")

    train_report_path: Path | None = None
    training_config_path: Path | None = None
    if mapper_checkpoint_path is None:
        train_result = train_conditioned_checkpoint(
            base_training_config_path=base_training_config_path,
            output_dir=out_dir / "train" / "run",
            device_name=device_name,
            max_steps=int(max_steps),
            seed=int(seed),
            reuse_existing_checkpoint=bool(reuse_existing_checkpoint),
        )
        mapper_checkpoint = train_result["checkpoint_path"]
        control_checkpoint = train_result["control_checkpoint_path"]
        train_report_path = train_result["report_path"]
        training_config_path = train_result["training_config_path"]
    else:
        mapper_checkpoint = Path(mapper_checkpoint_path)
        if control_checkpoint_path is None:
            control_checkpoint_path = ce_baseline_summary.get("control_checkpoint_path")
        if control_checkpoint_path is None:
            raise ValueError("control_checkpoint_path is required when mapper_checkpoint_path is supplied")
        control_checkpoint = Path(control_checkpoint_path)

    if not mapper_checkpoint.exists():
        raise FileNotFoundError(f"missing mapper checkpoint: {mapper_checkpoint}")
    if not control_checkpoint.exists():
        raise FileNotFoundError(f"missing control checkpoint: {control_checkpoint}")

    rollout_dir = out_dir / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    case_results: list[dict[str, Any]] = []
    for index, baseline_row in enumerate(selected_runs, start=1):
        case_id = str(baseline_row["case_id"])
        chart_end_ms = int(baseline_row["chart_end_ms"])
        print(f"conditioned short rollout {index:02d}/{len(selected_runs)} {case_id}", flush=True)
        summary_path = rollout_dir / f"{_safe_filename(case_id)}_{chart_end_ms}ms_conditioned_summary.json"
        report_path = rollout_dir / f"{_safe_filename(case_id)}_{chart_end_ms}ms_conditioned_report.md"
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
            logit_max_examples=12,
            timepoint_preview_limit=4096,
        )
        candidate_metrics = _metrics_from_rollout_summary(
            rollout_summary,
            beatmap_path=Path(str(baseline_row["beatmap_path"])),
            chart_end_ms=chart_end_ms,
        )
        baseline_metrics = _baseline_metrics(baseline_row)
        case_results.append(
            {
                "case_id": case_id,
                "case_index": baseline_row.get("case_index"),
                "selection_reasons": list(baseline_row.get("selection_reasons") or ()),
                "audio_family": baseline_row.get("audio_family"),
                "difficulty": baseline_row.get("difficulty"),
                "difficulty_band": baseline_row.get("difficulty_band"),
                "normalized_difficulty": baseline_row.get("normalized_difficulty"),
                "chart_end_ms": chart_end_ms,
                "audio_path": baseline_row.get("audio_path"),
                "beatmap_path": baseline_row.get("beatmap_path"),
                "summary_path": summary_path.as_posix(),
                "report_path": report_path.as_posix(),
                "baseline_metrics": baseline_metrics,
                "candidate_metrics": candidate_metrics,
                "delta_vs_baseline": compare_metrics(candidate_metrics, baseline_metrics),
                "baseline_legal": bool(baseline_row.get("legal")),
                "candidate_legal": _candidate_legal(rollout_summary),
                "candidate_rollout": _rollout_status(rollout_summary),
                "transform": {},
            }
        )
        if index <= 3:
            rollout_status = case_results[-1]["candidate_rollout"]
            if bool(rollout_status["dead_end"]) or bool(rollout_status["max_tokens_exceeded"]):
                break

    aggregate = aggregate_results(case_results)
    guard_results = gate_results_from_aggregate(aggregate)
    decision = decision_from_aggregate(aggregate, guard_results=guard_results)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar v3 conditioned event-distribution short rollout gate",
        "experiment_card": Path(experiment_card_path).as_posix(),
        "stage1_summary_path": stage1_summary_path.as_posix(),
        "ce_baseline_summary_path": ce_baseline_summary_path.as_posix(),
        "pre_ce_baseline_summary_path": pre_ce_baseline_summary_path.as_posix(),
        "base_training_config_path": base_training_config_path.as_posix(),
        "output_dir": out_dir.as_posix(),
        "mapper_checkpoint_path": mapper_checkpoint.as_posix(),
        "control_checkpoint_path": control_checkpoint.as_posix(),
        "train_report_path": None if train_report_path is None else train_report_path.as_posix(),
        "training_config_path": None if training_config_path is None else training_config_path.as_posix(),
        "code_commit": _git_stdout("rev-parse", "HEAD"),
        "code_dirty": bool(_git_stdout("status", "--short")),
        "elapsed_s": time.monotonic() - start,
        "config": {
            "device": device_name,
            "max_steps": int(max_steps),
            "max_tokens_per_window": int(max_tokens_per_window),
            "seed": int(seed),
            "case_limit": None if case_limit is None else int(case_limit),
            "reuse_existing_checkpoint": bool(reuse_existing_checkpoint),
            "loss_overrides": dict(DEFAULT_LOSS_OVERRIDES),
        },
        "baseline_targets": {
            "ce_starved_case_count": CE_BASELINE_STARVED_CASES,
            "ce_rigid_case_count": CE_BASELINE_RIGID_CASES,
            "median_event_ratio_min": MEDIAN_EVENT_RATIO_MIN,
            "median_event_ratio_max": MEDIAN_EVENT_RATIO_MAX,
            "ce_full32_aggregate": _baseline_aggregate(ce_baseline_summary),
            "pre_ce_full32_aggregate": _baseline_aggregate(pre_ce_baseline_summary),
        },
        "selected_case_count": len(selected_runs),
        "completed_case_count": len(case_results),
        "selection_mode": "ce_starved_plus_controls",
        "decision": decision,
        "guard_results": guard_results,
        "aggregate": aggregate,
        "case_results": case_results,
        "starvation_vs_flooding_table": starvation_vs_flooding_table(case_results),
        "next_step": decision["next_step"],
    }
    write_summary_json(summary, output_summary_path)
    if output_report_path is not None:
        write_report(summary, output_report_path)
    return summary


def train_conditioned_checkpoint(
    *,
    base_training_config_path: Path,
    output_dir: Path,
    device_name: str,
    max_steps: int,
    seed: int,
    reuse_existing_checkpoint: bool,
) -> dict[str, Path]:
    checkpoint_path = output_dir / "checkpoint.pt"
    report_path = output_dir / "report.json"
    training_config_path = output_dir.parent / "conditioned_event_distribution_enabled.yaml"
    if bool(reuse_existing_checkpoint) and checkpoint_path.exists() and report_path.exists():
        config = load_run_config(base_training_config_path)
        control_checkpoint = Path(str(config["init_from_control_checkpoint"]))
        return {
            "checkpoint_path": checkpoint_path,
            "report_path": report_path,
            "training_config_path": training_config_path,
            "control_checkpoint_path": control_checkpoint,
        }

    config = load_run_config(base_training_config_path)
    loss_config = dict(config["loss"])
    loss_config.update(DEFAULT_LOSS_OVERRIDES)
    config["loss"] = loss_config
    config["output_dir"] = output_dir.as_posix()
    config["run_name"] = "mapper_v3_conditioned_event_distribution_short_rollout"
    config["max_steps"] = int(max_steps)
    config["eval_every"] = min(int(config.get("eval_every") or 100), int(max_steps))
    config["save_every"] = min(int(config.get("save_every") or max_steps), int(max_steps))
    config["log_every"] = min(int(config.get("log_every") or 25), int(max_steps))
    config["seed"] = int(seed)
    config["device"] = str(device_name)
    training_config_path.parent.mkdir(parents=True, exist_ok=True)
    training_config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = run_mapper_v3_phase_b_training(
        dataset_root=Path(str(config.get("dataset_root", "dataset"))),
        index_path=_optional_path(config.get("index_path")),
        eval_index_path=_optional_path(config.get("eval_index_path")),
        control_v3_timeseries_path=_optional_path(config.get("control_v3_timeseries_path")),
        output_dir=output_dir,
        max_steps=int(config["max_steps"]),
        eval_every=int(config["eval_every"]),
        save_every=None if config.get("save_every") is None else int(config["save_every"]),
        log_every=None if config.get("log_every") is None else int(config["log_every"]),
        batch_size=int(config.get("batch_size", 2)),
        learning_rate=float(config.get("learning_rate", 2e-4)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        seed=int(config.get("seed", seed)),
        device_name=str(config.get("device", device_name)),
        run_name=str(config.get("run_name")),
        init_from_control_checkpoint=_optional_path(config.get("init_from_control_checkpoint")),
        resume_from=None,
        eval_fraction=float(config.get("eval_fraction", 0.1)),
        eval_size=None if config.get("eval_size") is None else int(config["eval_size"]),
        final_train_eval_size=(
            None if config.get("final_train_eval_size") is None else int(config["final_train_eval_size"])
        ),
        num_workers=int(config.get("num_workers", 0)),
        max_cached_maps=None if config.get("max_cached_maps") is None else int(config["max_cached_maps"]),
        dataset_progress=bool(config.get("dataset_progress", False)),
        mapper_record_cache_path=_optional_path(config.get("mapper_record_cache_path")),
        control_teacher_cache_dir=_optional_path(config.get("control_teacher_cache_dir")),
        require_control_teacher_cache=bool(config.get("require_control_teacher_cache", False)),
        precompute_control_teacher_cache=bool(config.get("precompute_control_teacher_cache", False)),
        control_teacher_precompute_batch_size=(
            None
            if config.get("control_teacher_precompute_batch_size") is None
            else int(config["control_teacher_precompute_batch_size"])
        ),
        control_teacher_cache_overwrite=bool(config.get("control_teacher_cache_overwrite", False)),
        include_full_song_context=bool(config.get("include_full_song_context", True)),
        skip_first_eval_pass=bool(config.get("skip_first_eval_pass", False)),
        mps_cleanup_every=None if config.get("mps_cleanup_every") is None else int(config["mps_cleanup_every"]),
        model_config_overrides=config["model"],
        control_model_config_overrides=config["control_model"],
        loss_config_overrides=config["loss"],
    )
    return {
        "checkpoint_path": result.checkpoint_path,
        "report_path": result.report_path,
        "training_config_path": training_config_path,
        "control_checkpoint_path": Path(str(config["init_from_control_checkpoint"])),
    }


def select_short_rollout_cases(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in summary.get("runs", ()) if isinstance(row, Mapping)]
    selected: dict[str, dict[str, Any]] = {}

    def add(row: Mapping[str, Any], reason: str) -> None:
        case_id = str(row.get("case_id"))
        if not case_id:
            return
        item = selected.setdefault(case_id, dict(row))
        reasons = list(item.get("selection_reasons") or ())
        if reason not in reasons:
            reasons.append(reason)
        item["selection_reasons"] = reasons

    for row in rows:
        if bool(row.get("starved")):
            add(row, "ce_starved")

    safe_controls = [
        row
        for row in rows
        if not bool(row.get("starved"))
        and (_float(row.get("dominant_spacing_ratio")) or 0.0) < 0.75
        and 0.8 <= (_float(row.get("event_count_ratio")) or 0.0) <= 1.25
    ]
    if safe_controls:
        add(sorted(safe_controls, key=lambda row: int(row.get("case_index") or 0))[0], "pass_like_control")

    overproduction_risks = [
        row for row in rows if not bool(row.get("starved")) and (_float(row.get("event_count_ratio")) or 0.0) > 1.25
    ]
    if overproduction_risks:
        add(
            max(overproduction_risks, key=lambda row: _float(row.get("event_count_ratio")) or 0.0),
            "overproduction_risk_control",
        )

    rigid_risks = [
        row
        for row in rows
        if not bool(row.get("starved")) and (_float(row.get("dominant_spacing_ratio")) or 0.0) >= 0.95
    ]
    if rigid_risks:
        add(
            max(rigid_risks, key=lambda row: _float(row.get("dominant_spacing_ratio")) or 0.0),
            "rigid_risk_control",
        )

    return sorted(selected.values(), key=lambda item: (int(item.get("case_index") or 0), str(item.get("case_id"))))


def gate_results_from_aggregate(aggregate: Mapping[str, Any]) -> dict[str, bool]:
    candidate = _mapping(aggregate.get("candidate"))
    median_event_ratio = float(candidate.get("median_event_count_ratio") or 0.0)
    return {
        "case_coverage": int(candidate.get("case_count") or 0) >= 6,
        "all_legal": bool(candidate.get("all_legal")),
        "no_dead_end_cases": int(aggregate.get("dead_end_count") or 0) == 0,
        "no_max_token_cases": int(aggregate.get("max_token_count") or 0) == 0,
        "starved_below_ce_baseline_5": int(candidate.get("starved_count") or 0) < CE_BASELINE_STARVED_CASES,
        "rigid_no_worse_than_ce_baseline_11": int(candidate.get("rigid_case_count") or 0)
        <= CE_BASELINE_RIGID_CASES,
        "median_event_count_ratio_in_range": MEDIAN_EVENT_RATIO_MIN <= median_event_ratio <= MEDIAN_EVENT_RATIO_MAX,
        "starved_improved_vs_selected_baseline": int(aggregate.get("starved_case_delta") or 0) < 0,
    }


def decision_from_aggregate(aggregate: Mapping[str, Any], *, guard_results: Mapping[str, bool]) -> dict[str, str]:
    failed = [key for key, passed in guard_results.items() if not bool(passed)]
    candidate = _mapping(aggregate.get("candidate"))
    if any(key in failed for key in ("all_legal", "no_dead_end_cases", "no_max_token_cases", "median_event_count_ratio_in_range")):
        return {
            "route": "KILL",
            "reason": "conditioned objective tripped a hard safety or event-count calibration guard: "
            + ", ".join(failed),
            "next_step": "Do not run full32; inspect event-count calibration failure or move to target-grammar repair.",
        }
    if failed:
        route = "MUTATE_TO_TARGET_GRAMMAR_REPAIR"
        if int(candidate.get("starved_count") or 0) < CE_BASELINE_STARVED_CASES:
            route = "TEST_STARVATION_VS_FLOODING_DIAGNOSTIC"
        return {
            "route": route,
            "reason": "failed checks: " + ", ".join(failed),
            "next_step": (
                "Run one starvation-vs-flooding diagnostic."
                if route == "TEST_STARVATION_VS_FLOODING_DIAGNOSTIC"
                else "Create a target-grammar repair card; the short conditioned objective did not beat CE starvation."
            ),
        }
    return {
        "route": "TEST_FULL32_CONDITIONED_EVENT_DISTRIBUTION",
        "reason": "short conditioned objective gate reduced starvation without overgeneration or legality regressions",
        "next_step": "Create and run a full32 conditioned-event-distribution gate before any replacement claim.",
    }


def starvation_vs_flooding_table(case_results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in case_results:
        baseline = _mapping(row.get("baseline_metrics"))
        candidate = _mapping(row.get("candidate_metrics"))
        delta = _mapping(row.get("delta_vs_baseline"))
        rows.append(
            {
                "case_id": row.get("case_id"),
                "selection_reasons": row.get("selection_reasons"),
                "baseline_starved": baseline.get("starved"),
                "candidate_starved": candidate.get("starved"),
                "baseline_event_count_ratio": baseline.get("event_count_ratio"),
                "candidate_event_count_ratio": candidate.get("event_count_ratio"),
                "event_count_ratio_delta": delta.get("event_count_ratio_delta"),
                "baseline_rigid": baseline.get("dominant_spacing_ratio"),
                "candidate_rigid": candidate.get("dominant_spacing_ratio"),
                "f1_delta": delta.get("f1_delta"),
                "legal": row.get("candidate_legal"),
                "dead_end": _mapping(row.get("candidate_rollout")).get("dead_end"),
                "max_tokens_exceeded": _mapping(row.get("candidate_rollout")).get("max_tokens_exceeded"),
            }
        )
    return rows


def write_report(summary: Mapping[str, Any], path: str | Path) -> None:
    aggregate = _mapping(summary.get("aggregate"))
    baseline = _mapping(aggregate.get("baseline"))
    candidate = _mapping(aggregate.get("candidate"))
    decision = _mapping(summary.get("decision"))
    config = _mapping(summary.get("config"))
    lines = [
        "# Target Grammar v3 Conditioned Event-Distribution Short Rollout Gate Result Report",
        "",
        "## Scope",
        "",
        "This pass trains or uses one conditioned-objective v3 checkpoint, then rolls only the CE-sensitive fixed-slice subset. It does not change tokenizer, grammar, decode defaults, or run full32.",
        "",
        "## Decision",
        "",
        f"- route: `{decision.get('route')}`",
        f"- reason: {decision.get('reason')}",
        f"- next step: {decision.get('next_step')}",
        "",
        "## Training",
        "",
        f"- mapper checkpoint: `{summary.get('mapper_checkpoint_path')}`",
        f"- control checkpoint: `{summary.get('control_checkpoint_path')}`",
        f"- training report: `{summary.get('train_report_path')}`",
        f"- training config: `{summary.get('training_config_path')}`",
        f"- max steps: `{config.get('max_steps')}`",
        f"- loss overrides: `{config.get('loss_overrides')}`",
        "",
        "## Aggregate",
        "",
        "| Metric | CE selected baseline | Conditioned candidate | Delta |",
        "| --- | ---: | ---: | ---: |",
        _metric_row("selected cases", baseline.get("case_count"), candidate.get("case_count"), None),
        _metric_row("starved cases", baseline.get("starved_count"), candidate.get("starved_count"), aggregate.get("starved_case_delta")),
        _metric_row("rigid cases", baseline.get("rigid_case_count"), candidate.get("rigid_case_count"), aggregate.get("rigid_case_delta")),
        _metric_row("mean F1@100ms", baseline.get("mean_f1_100ms"), candidate.get("mean_f1_100ms"), aggregate.get("mean_f1_delta")),
        _metric_row("median event ratio", baseline.get("median_event_count_ratio"), candidate.get("median_event_count_ratio"), None),
        _metric_row("max boundary ratio", baseline.get("max_boundary_event_ratio"), candidate.get("max_boundary_event_ratio"), None),
        "",
        "## Guard Results",
        "",
        *[f"- {key}: `{str(value).lower()}`" for key, value in _mapping(summary.get("guard_results")).items()],
        "",
        "## Starvation vs Flooding",
        "",
        "| Case | Reasons | Starved | Event Ratio | Rigid | F1 Delta | Legal |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.get("starvation_vs_flooding_table", ()):
        item = _mapping(row)
        lines.append(
            "| `{case}` | {reasons} | `{bs}` -> `{cs}` | `{ber}` -> `{cer}` | `{br}` -> `{cr}` | `{f1}` | `{legal}` |".format(
                case=item.get("case_id"),
                reasons=", ".join(str(reason) for reason in item.get("selection_reasons", ()) or ()),
                bs=item.get("baseline_starved"),
                cs=item.get("candidate_starved"),
                ber=_fmt(item.get("baseline_event_count_ratio")),
                cer=_fmt(item.get("candidate_event_count_ratio")),
                br=_fmt(item.get("baseline_rigid")),
                cr=_fmt(item.get("candidate_rigid")),
                f1=_fmt(item.get("f1_delta")),
                legal=item.get("legal"),
            )
        )
    lines.extend(
        [
            "",
            "## What This Proves",
            "",
            "- This proves only the short trained gate result for the selected CE-sensitive cases.",
            "- It can justify a full32 card only if all guards pass.",
            "",
            "## What Remains Unproven",
            "",
            "- Full32 conditioned-objective behavior.",
            "- Full-dataset/full-song v3 quality.",
            "- Replacement readiness for mapper or planner.",
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


def _require_stage1_route(summary: Mapping[str, Any]) -> None:
    decision = _mapping(summary.get("decision"))
    if decision.get("route") != "TEST_SHORT_ROLLOUT_GATE":
        raise ValueError("Stage 1 summary must route to TEST_SHORT_ROLLOUT_GATE")


def _baseline_aggregate(summary: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = _mapping(summary.get("aggregate"))
    keys = (
        "case_count",
        "all_legal",
        "dead_end_count",
        "max_token_count",
        "second_window_starved_case_count",
        "rigid_case_count",
        "median_event_count_ratio",
        "mean_f1_100ms",
        "max_boundary_event_ratio",
    )
    return {key: aggregate.get(key) for key in keys if key in aggregate}


def _metric_row(label: str, baseline: Any, candidate: Any, delta: Any) -> str:
    return f"| {label} | `{_fmt(baseline)}` | `{_fmt(candidate)}` | `{_fmt(delta)}` |"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return str(value)
    return f"{numeric:.6f}"


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _optional_path(value: object) -> Path | None:
    if value is None:
        return None
    return Path(str(value))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the v3 conditioned event-distribution short rollout gate.")
    parser.add_argument("--stage1-summary", type=Path, default=DEFAULT_STAGE1_SUMMARY_PATH)
    parser.add_argument("--ce-baseline-summary", type=Path, default=DEFAULT_CE_BASELINE_SUMMARY_PATH)
    parser.add_argument("--pre-ce-baseline-summary", type=Path, default=DEFAULT_PRE_CE_BASELINE_SUMMARY_PATH)
    parser.add_argument("--base-training-config", type=Path, default=DEFAULT_BASE_TRAINING_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--experiment-card", type=Path, default=DEFAULT_EXPERIMENT_CARD_PATH)
    parser.add_argument("--mapper-checkpoint-path", type=Path)
    parser.add_argument("--control-checkpoint-path", type=Path)
    parser.add_argument("--device", default="mps", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-tokens-per-window", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--reuse-existing-checkpoint", action="store_true")
    args = parser.parse_args(argv)
    summary = run_conditioned_event_distribution_short_rollout_gate(
        stage1_summary_path=args.stage1_summary,
        ce_baseline_summary_path=args.ce_baseline_summary,
        pre_ce_baseline_summary_path=args.pre_ce_baseline_summary,
        base_training_config_path=args.base_training_config,
        output_dir=args.output_dir,
        output_summary_path=args.summary_output,
        output_report_path=args.report_output,
        experiment_card_path=args.experiment_card,
        mapper_checkpoint_path=args.mapper_checkpoint_path,
        control_checkpoint_path=args.control_checkpoint_path,
        device_name=args.device,
        max_steps=args.max_steps,
        max_tokens_per_window=args.max_tokens_per_window,
        seed=args.seed,
        case_limit=args.case_limit,
        reuse_existing_checkpoint=args.reuse_existing_checkpoint,
    )
    aggregate = _mapping(summary["aggregate"])
    candidate = _mapping(aggregate.get("candidate"))
    print(
        "mapper_v3_conditioned_event_distribution_short_rollout_done "
        f"route={summary['decision']['route']} "
        f"cases={summary['completed_case_count']} "
        f"starved={candidate.get('starved_count')} "
        f"rigid={candidate.get('rigid_case_count')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
