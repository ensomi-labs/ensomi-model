from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from pulsefield_model.evals.mapper_v3_delta_event_factor_target_longer_training import _stability_metrics
from pulsefield_model.evals.mapper_v3_delta_event_factor_target_training_smoke import (
    _loss_total_check,
    _model_config_overrides,
)
from pulsefield_model.training.mapper_v3 import run_mapper_v3_phase_b_training


SUMMARY_SCHEMA_VERSION = 1
REPORT_ROOT = Path("artifacts/reports/audits/mapper_v2_1_grammar")
DEFAULT_INDEX_PATH = Path("artifacts/indexes/stage2_control_windows_4k_2to6_dense_local_bpm_norm_unique_le3.parquet")
DEFAULT_CONTROL_TEACHER_CACHE_DIR = Path("artifacts/cache/stage2_mapper_v2_1/control_teacher_d384_l3_stride16_step002000")
DEFAULT_MAPPER_RECORD_CACHE_PATH = Path(
    "artifacts/cache/stage2_mapper_v3/window_records/delta_event_factor_target_full4k_training.parquet"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/runs/stage2_mapper_v3/delta_event_factor_target_full4k_training")
DEFAULT_EXPERIMENT_CARD = REPORT_ROOT / "target_grammar_v3_delta_event_factor_target_full4k_training_experiment_card.md"
DEFAULT_SUMMARY_OUTPUT = REPORT_ROOT / "target_grammar_v3_delta_event_factor_target_full4k_training_summary.json"
DEFAULT_REPORT_OUTPUT = REPORT_ROOT / "target_grammar_v3_delta_event_factor_target_full4k_training_result_report.md"
DEFAULT_LONGER_TRAINING_SUMMARY = REPORT_ROOT / "target_grammar_v3_delta_event_factor_target_longer_training_summary.json"
DEFAULT_FULL4K_BIT_PROXY_SUMMARY = REPORT_ROOT / "target_grammar_v3_delta_event_full4k_bit_proxy_summary.json"
DEFAULT_FULL4K_LABEL_COVERAGE_SUMMARY = (
    REPORT_ROOT / "target_grammar_v3_delta_event_auxiliary_full4k_label_coverage_summary.json"
)


def run_delta_event_factor_target_full4k_training(
    *,
    dataset_root: str | Path = Path("dataset"),
    index_path: str | Path = DEFAULT_INDEX_PATH,
    control_teacher_cache_dir: str | Path = DEFAULT_CONTROL_TEACHER_CACHE_DIR,
    mapper_record_cache_path: str | Path = DEFAULT_MAPPER_RECORD_CACHE_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    summary_output_path: str | Path = DEFAULT_SUMMARY_OUTPUT,
    report_output_path: str | Path | None = DEFAULT_REPORT_OUTPUT,
    experiment_card_path: str | Path = DEFAULT_EXPERIMENT_CARD,
    longer_training_summary_path: str | Path = DEFAULT_LONGER_TRAINING_SUMMARY,
    full4k_bit_proxy_summary_path: str | Path = DEFAULT_FULL4K_BIT_PROXY_SUMMARY,
    full4k_label_coverage_summary_path: str | Path = DEFAULT_FULL4K_LABEL_COVERAGE_SUMMARY,
    max_steps: int = 8,
    eval_every: int = 4,
    batch_size: int = 4,
    eval_size: int = 16,
    final_train_eval_size: int = 16,
    learning_rate: float = 3e-4,
    lambda_delta_event_factor_target: float = 0.25,
    max_factor_loss_ratio: float = 1.20,
    seed: int = 20260703,
    device_name: str = "cpu",
    max_cached_maps: int = 8,
    dataset_progress: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    _validate_config(
        max_steps=max_steps,
        eval_every=eval_every,
        batch_size=batch_size,
        eval_size=eval_size,
        final_train_eval_size=final_train_eval_size,
        learning_rate=learning_rate,
        lambda_delta_event_factor_target=lambda_delta_event_factor_target,
        max_factor_loss_ratio=max_factor_loss_ratio,
        max_cached_maps=max_cached_maps,
    )
    record_cache_path = Path(mapper_record_cache_path)
    mapper_record_cache_existed_before_run = _mapper_record_cache_complete(record_cache_path)
    model_config_overrides = _model_config_overrides()
    loss_config_overrides = {
        "lambda_density": 0.0,
        "lambda_ln_close": 0.0,
        "lambda_adapter_reg": 0.0,
        "lambda_delta_event_factor_target": float(lambda_delta_event_factor_target),
    }
    result = run_mapper_v3_phase_b_training(
        dataset_root=Path(dataset_root),
        index_path=Path(index_path),
        output_dir=Path(output_dir),
        max_steps=int(max_steps),
        eval_every=int(eval_every),
        save_every=int(eval_every),
        log_every=int(eval_every),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        weight_decay=0.0,
        seed=int(seed),
        device_name=str(device_name),
        run_name="mapper_v3_delta_event_factor_target_full4k_training",
        eval_size=int(eval_size),
        final_train_eval_size=int(final_train_eval_size),
        num_workers=0,
        max_cached_maps=int(max_cached_maps),
        dataset_progress=bool(dataset_progress),
        mapper_record_cache_path=record_cache_path,
        control_teacher_cache_dir=Path(control_teacher_cache_dir),
        require_control_teacher_cache=True,
        include_full_song_context=False,
        skip_first_eval_pass=False,
        model_config_overrides=model_config_overrides,
        control_model_config_overrides={},
        loss_config_overrides=loss_config_overrides,
    )
    training_report = _read_json(result.report_path)
    context = _load_context(
        longer_training_summary_path=Path(longer_training_summary_path),
        full4k_bit_proxy_summary_path=Path(full4k_bit_proxy_summary_path),
        full4k_label_coverage_summary_path=Path(full4k_label_coverage_summary_path),
    )
    dataset = dataset_metrics(
        training_report,
        mapper_record_cache_path=record_cache_path,
        control_teacher_cache_dir=Path(control_teacher_cache_dir),
        context=context,
    )
    dataset["mapper_record_cache_existed_before_run"] = bool(mapper_record_cache_existed_before_run)
    stability = _stability_metrics(
        training_report,
        checkpoint_path=result.checkpoint_path,
        max_factor_loss_ratio=float(max_factor_loss_ratio),
    )
    loss_total_check = _loss_total_check(
        _mapping(training_report.get("final_eval_metrics")),
        _mapping(training_report.get("loss_config")),
    )
    checks = guard_checks(
        dataset=dataset,
        stability=stability,
        loss_total_check=loss_total_check,
        context=context,
    )
    decision = decision_from_checks(checks)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar v3 delta-event factor target full4k fixed-cache training",
        "experiment_card": Path(experiment_card_path).as_posix(),
        "elapsed_s": time.monotonic() - started,
        "index_path": Path(index_path).as_posix(),
        "control_teacher_cache_dir": Path(control_teacher_cache_dir).as_posix(),
        "mapper_record_cache_path": Path(mapper_record_cache_path).as_posix(),
        "training_output_dir": Path(output_dir).as_posix(),
        "config": {
            "max_steps": int(max_steps),
            "eval_every": int(eval_every),
            "batch_size": int(batch_size),
            "eval_size": int(eval_size),
            "final_train_eval_size": int(final_train_eval_size),
            "learning_rate": float(learning_rate),
            "lambda_delta_event_factor_target": float(lambda_delta_event_factor_target),
            "max_factor_loss_ratio": float(max_factor_loss_ratio),
            "seed": int(seed),
            "device": str(device_name),
            "max_cached_maps": int(max_cached_maps),
            "dataset_progress": bool(dataset_progress),
            "model": model_config_overrides,
            "loss": loss_config_overrides,
            "include_full_song_context": False,
            "no_inference_or_rollout_change": True,
            "no_mapper_default_change": True,
            "no_tokenizer_change": True,
            "uses_c3_backreference": False,
            "uses_future_lookup": False,
        },
        "context": context,
        "dataset": dataset,
        "stability": stability,
        "loss_total_check": loss_total_check,
        "checks": checks,
        "decision": decision,
        "next_step": decision["next_step"],
    }
    write_summary_json(summary, Path(summary_output_path))
    if report_output_path is not None:
        write_report(summary, Path(report_output_path))
    return summary


def dataset_metrics(
    report: Mapping[str, Any],
    *,
    mapper_record_cache_path: Path,
    control_teacher_cache_dir: Path,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    dataset = _mapping(report.get("dataset"))
    filter_report = _mapping(dataset.get("filter_report"))
    reported_record_cache = Path(str(dataset.get("mapper_record_cache_path") or mapper_record_cache_path))
    reported_teacher_cache = Path(str(dataset.get("control_teacher_cache_dir") or control_teacher_cache_dir))
    expected_audited_windows = _expected_metric(context, "full4k_bit_proxy", "audited_window_count")
    expected_candidate_windows = _expected_metric(context, "full4k_bit_proxy", "candidate_window_count")
    source_window_count = _int_or_none(dataset.get("source_window_count"))
    candidate_window_count = _int_or_none(filter_report.get("num_total_windows"))
    return {
        "source_window_count": source_window_count,
        "train_window_count": _int_or_none(dataset.get("train_window_count")),
        "eval_window_count": _int_or_none(dataset.get("eval_window_count")),
        "final_train_eval_window_count": _int_or_none(dataset.get("final_train_eval_window_count")),
        "filter_num_total_windows": candidate_window_count,
        "filter_num_mapper_eligible_windows": _int_or_none(filter_report.get("num_mapper_eligible_windows")),
        "filter_num_dropped_unsupported_action_windows": _int_or_none(
            filter_report.get("num_dropped_unsupported_action_windows")
        ),
        "filter_drop_rate": _float(filter_report.get("drop_rate")),
        "expected_full4k_audited_window_count": expected_audited_windows,
        "expected_full4k_candidate_window_count": expected_candidate_windows,
        "source_matches_full4k_bit_proxy": (
            expected_audited_windows is not None
            and source_window_count is not None
            and int(source_window_count) == int(expected_audited_windows)
        ),
        "candidate_windows_match_full4k_bit_proxy": (
            expected_candidate_windows is not None
            and candidate_window_count is not None
            and int(candidate_window_count) == int(expected_candidate_windows)
        ),
        "mapper_record_cache_path": reported_record_cache.as_posix(),
        "mapper_record_cache_exists": reported_record_cache.exists(),
        "mapper_record_cache_metadata_exists": reported_record_cache.with_suffix(".json").exists(),
        "control_teacher_cache_dir": reported_teacher_cache.as_posix(),
        "control_teacher_cache_dir_exists": reported_teacher_cache.exists(),
        "control_teacher_cache_file_count": _count_cache_files(reported_teacher_cache),
        "require_control_teacher_cache": bool(dataset.get("require_control_teacher_cache")),
        "include_delta_event_factor_target": bool(dataset.get("include_delta_event_factor_target")),
        "include_full_song_context": bool(dataset.get("include_full_song_context")),
        "mapper_token_contract": str(dataset.get("mapper_token_contract") or ""),
    }


def guard_checks(
    *,
    dataset: Mapping[str, Any],
    stability: Mapping[str, Any],
    loss_total_check: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, bool]:
    final_eval = _mapping(stability.get("final_eval_metrics"))
    final_train = _mapping(stability.get("final_train_metrics"))
    return {
        "longer_training_route_positive": _mapping(context.get("longer_training")).get("route")
        == "TEST_DELTA_EVENT_FACTOR_TARGET_FULL4K_TRAINING_CARD",
        "full4k_bit_proxy_route_positive": _mapping(context.get("full4k_bit_proxy")).get("route")
        == "TEST_DELTA_EVENT_FACTOR_TARGET_MODEL_CARD",
        "full4k_label_coverage_route_positive": _mapping(context.get("full4k_label_coverage")).get("route")
        == "TEST_DELTA_EVENT_FULL4K_BIT_PROXY_OR_FACTOR_TARGET_CARD",
        "source_windows_full4k_scale": int(dataset.get("source_window_count") or 0) >= 100_000,
        "source_matches_full4k_bit_proxy": bool(dataset.get("source_matches_full4k_bit_proxy")),
        "candidate_windows_match_full4k_bit_proxy": bool(dataset.get("candidate_windows_match_full4k_bit_proxy")),
        "train_windows_positive": int(dataset.get("train_window_count") or 0) > 0,
        "eval_windows_positive": int(dataset.get("eval_window_count") or 0) > 0,
        "mapper_record_cache_configured": bool(dataset.get("mapper_record_cache_path")),
        "mapper_record_cache_exists": bool(dataset.get("mapper_record_cache_exists")),
        "mapper_record_cache_metadata_exists": bool(dataset.get("mapper_record_cache_metadata_exists")),
        "control_teacher_cache_dir_exists": bool(dataset.get("control_teacher_cache_dir_exists")),
        "control_teacher_cache_file_count_matches_candidates": int(dataset.get("control_teacher_cache_file_count") or 0)
        == int(dataset.get("expected_full4k_candidate_window_count") or -1),
        "control_teacher_cache_required": bool(dataset.get("require_control_teacher_cache")),
        "dataset_factor_target_enabled": bool(dataset.get("include_delta_event_factor_target")),
        "completed_requested_steps": int(stability.get("completed_steps") or -1) == int(stability.get("max_steps") or -2),
        "training_marked_complete": bool(stability.get("is_complete")),
        "report_exists": bool(stability.get("report_exists")),
        "checkpoint_exists": bool(stability.get("checkpoint_exists")),
        "model_factor_target_enabled": bool(stability.get("model_use_delta_event_factor_target")),
        "loss_factor_lambda_positive": float(stability.get("loss_lambda_delta_event_factor_target") or 0.0) > 0.0,
        "eval_point_count_sufficient": int(stability.get("eval_point_count") or 0) >= 3,
        "all_eval_losses_finite": bool(stability.get("all_eval_losses_finite")),
        "factor_loss_ratio_within_limit": bool(stability.get("factor_loss_ratio_within_limit")),
        "final_eval_factor_loss_finite": _finite_positive(final_eval.get("loss/delta_event_factor_target")),
        "final_train_factor_loss_finite": _finite_positive(final_train.get("loss/delta_event_factor_target")),
        "final_eval_kind_labels_positive": float(final_eval.get("delta_event_factor/kind_label_count") or 0.0) > 0.0,
        "final_eval_delta_labels_positive": float(final_eval.get("delta_event_factor/delta_label_count") or 0.0) > 0.0,
        "final_eval_signature_labels_positive": float(final_eval.get("delta_event_factor/signature_label_count") or 0.0) > 0.0,
        "final_eval_end_gap_labels_positive": float(final_eval.get("delta_event_factor/end_gap_label_count") or 0.0) > 0.0,
        "loss_total_recomputed_matches": bool(loss_total_check.get("matches")),
        "no_inference_rollout_or_c3_change": True,
    }


def decision_from_checks(checks: Mapping[str, bool]) -> dict[str, str]:
    failed = [key for key, passed in checks.items() if not bool(passed)]
    if not failed:
        return {
            "route": "TEST_DELTA_EVENT_FACTOR_TARGET_FULL4K_LONGER_OR_PRODUCTION_CONFIG_CARD",
            "reason": "factor-target enabled v3 training stayed finite on the full4k fixed-cache path",
            "next_step": "Create a bounded longer-full4k or production-config factor-target card before rollout claims.",
        }
    context_failures = {
        "longer_training_route_positive",
        "full4k_bit_proxy_route_positive",
        "full4k_label_coverage_route_positive",
    }
    cache_or_dataset_failures = {
        "source_windows_full4k_scale",
        "source_matches_full4k_bit_proxy",
        "candidate_windows_match_full4k_bit_proxy",
        "train_windows_positive",
        "eval_windows_positive",
        "mapper_record_cache_configured",
        "mapper_record_cache_exists",
        "mapper_record_cache_metadata_exists",
        "control_teacher_cache_dir_exists",
        "control_teacher_cache_file_count_matches_candidates",
        "control_teacher_cache_required",
        "dataset_factor_target_enabled",
    }
    plumbing_failures = {
        "completed_requested_steps",
        "training_marked_complete",
        "report_exists",
        "checkpoint_exists",
        "model_factor_target_enabled",
        "loss_factor_lambda_positive",
        "eval_point_count_sufficient",
        "all_eval_losses_finite",
        "final_eval_factor_loss_finite",
        "final_train_factor_loss_finite",
        "final_eval_kind_labels_positive",
        "final_eval_delta_labels_positive",
        "final_eval_signature_labels_positive",
        "final_eval_end_gap_labels_positive",
        "loss_total_recomputed_matches",
    }
    if any(key in context_failures for key in failed):
        return {
            "route": "MUTATE_FACTOR_TARGET_FULL4K_CONTEXT_CHAIN",
            "reason": "required prior full4k/fixed-slice gates are missing or not positive: " + ", ".join(failed),
            "next_step": "Repair or rerun the prior gate chain before interpreting full4k factor-target training.",
        }
    if any(key in cache_or_dataset_failures for key in failed):
        return {
            "route": "MUTATE_FACTOR_TARGET_FULL4K_CACHE_OR_DATASET",
            "reason": "full4k factor-target training failed dataset/cache checks: " + ", ".join(failed),
            "next_step": "Repair mapper-record cache, control-teacher cache, or full4k dataset wiring before training.",
        }
    if "factor_loss_ratio_within_limit" in failed and "all_eval_losses_finite" not in failed:
        return {
            "route": "MUTATE_FACTOR_TARGET_FULL4K_LR_OR_LOSS_WEIGHT",
            "reason": "full4k factor-target losses stayed finite but exceeded the non-explosion ratio: "
            + ", ".join(failed),
            "next_step": "Tune learning rate, factor loss weight, or model scale before longer full4k training.",
        }
    if any(key in plumbing_failures for key in failed):
        return {
            "route": "MUTATE_FACTOR_TARGET_FULL4K_PLUMBING",
            "reason": "full4k factor-target training failed runner/metric checks: " + ", ".join(failed),
            "next_step": "Repair runner metrics, factor-target supervision, or loss accounting before scaling.",
        }
    return {
        "route": "MUTATE_FACTOR_TARGET_FULL4K_UNKNOWN",
        "reason": "full4k factor-target training failed checks: " + ", ".join(failed),
        "next_step": "Inspect failed checks and create a narrower diagnostic card.",
    }


def write_summary_json(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_markdown(summary).rstrip() + "\n", encoding="utf-8")


def report_markdown(summary: Mapping[str, Any]) -> str:
    decision = _mapping(summary.get("decision"))
    dataset = _mapping(summary.get("dataset"))
    stability = _mapping(summary.get("stability"))
    first_eval = _mapping(stability.get("first_eval_metrics"))
    final_eval = _mapping(stability.get("final_eval_metrics"))
    final_train = _mapping(stability.get("final_train_metrics"))
    loss_total = _mapping(summary.get("loss_total_check"))
    checks = _mapping(summary.get("checks"))
    lines = [
        "# Target Grammar v3 Delta-Event Factor Target Full4k Fixed-Cache Training Result Report",
        "",
        "## Scope",
        "",
        "This gate runs a bounded production v3 training loop over the full4k source index with cached control-teacher tensors and factorized delta-event target supervision enabled. It does not claim production convergence, rollout quality, online factor-row inference, or complete v3 replacement readiness.",
        "",
        "## Decision",
        "",
        f"Route: `{decision.get('route')}`.",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Recommended next step: {decision.get('next_step')}",
        "",
        "## Dataset And Cache",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        _row("source window count", dataset.get("source_window_count")),
        _row("train window count", dataset.get("train_window_count")),
        _row("eval window count", dataset.get("eval_window_count")),
        _row("candidate windows", dataset.get("filter_num_total_windows")),
        _row("eligible windows", dataset.get("filter_num_mapper_eligible_windows")),
        _row("expected audited windows", dataset.get("expected_full4k_audited_window_count")),
        _row("expected candidate windows", dataset.get("expected_full4k_candidate_window_count")),
        _row("mapper record cache", dataset.get("mapper_record_cache_path")),
        _row("mapper record cache existed before run", dataset.get("mapper_record_cache_existed_before_run")),
        _row("mapper record cache exists", dataset.get("mapper_record_cache_exists")),
        _row("control teacher cache files", dataset.get("control_teacher_cache_file_count")),
        _row("require control teacher cache", dataset.get("require_control_teacher_cache")),
        _row("dataset factor target", dataset.get("include_delta_event_factor_target")),
        "",
        "## Run",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        _row("completed steps", stability.get("completed_steps")),
        _row("max steps", stability.get("max_steps")),
        _row("eval steps", stability.get("eval_steps")),
        _row("report path", stability.get("report_path")),
        _row("checkpoint path", stability.get("checkpoint_path")),
        _row("model factor target", stability.get("model_use_delta_event_factor_target")),
        _row("loss factor lambda", stability.get("loss_lambda_delta_event_factor_target")),
        "",
        "## Stability Metrics",
        "",
        "| Metric | First Eval | Final Train | Final Eval |",
        "| --- | ---: | ---: | ---: |",
        _metric_row("loss/total", first_eval, final_train, final_eval),
        _metric_row("loss/token", first_eval, final_train, final_eval),
        _metric_row("loss/delta_event_factor_target", first_eval, final_train, final_eval),
        _metric_row("phase/lambda_delta_event_factor_target", first_eval, final_train, final_eval),
        _metric_row("delta_event_factor/kind_label_count", first_eval, final_train, final_eval),
        _metric_row("delta_event_factor/delta_label_count", first_eval, final_train, final_eval),
        _metric_row("delta_event_factor/signature_label_count", first_eval, final_train, final_eval),
        _metric_row("delta_event_factor/end_gap_label_count", first_eval, final_train, final_eval),
        "",
        "| Derived Metric | Value |",
        "| --- | ---: |",
        _row("first eval factor loss", _fmt(stability.get("first_eval_factor_loss"))),
        _row("final eval factor loss", _fmt(stability.get("final_eval_factor_loss"))),
        _row("factor loss ratio", _fmt(stability.get("factor_loss_ratio"))),
        _row("max factor loss ratio", _fmt(stability.get("max_factor_loss_ratio"))),
        _row("all eval losses finite", stability.get("all_eval_losses_finite")),
        "",
        "## Loss Total Accounting",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        _row("reported final eval total", _fmt(loss_total.get("reported"))),
        _row("recomputed final eval total", _fmt(loss_total.get("recomputed"))),
        _row("absolute delta", _fmt(loss_total.get("absolute_delta"))),
        _row("matches", loss_total.get("matches")),
        "",
        "## Checks",
        "",
        "| Check | Value |",
        "| --- | ---: |",
    ]
    lines.extend(_row(key, value) for key, value in checks.items())
    lines.extend(
        [
            "",
            "## What Passed",
            "",
            _what_passed(str(decision.get("route"))),
            "",
            "## What Surfaced",
            "",
            _what_surfaced(str(decision.get("route"))),
            "",
            "## What Is Not Proved",
            "",
            "- This does not prove full production training convergence.",
            "- This does not prove generated chart quality.",
            "- This does not implement online factor-row inference.",
            "- This does not make v3 the final mapper/planner replacement.",
            "",
            "## Verification",
            "",
            *_verification_lines(),
            "",
            "## Next Step",
            "",
            str(decision.get("next_step")),
        ]
    )
    return "\n".join(lines)


def _load_context(
    *,
    longer_training_summary_path: Path,
    full4k_bit_proxy_summary_path: Path,
    full4k_label_coverage_summary_path: Path,
) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if longer_training_summary_path.exists():
        data = _read_json(longer_training_summary_path)
        context["longer_training"] = _context_entry(longer_training_summary_path, data)
    if full4k_bit_proxy_summary_path.exists():
        data = _read_json(full4k_bit_proxy_summary_path)
        context["full4k_bit_proxy"] = _context_entry(full4k_bit_proxy_summary_path, data)
    if full4k_label_coverage_summary_path.exists():
        data = _read_json(full4k_label_coverage_summary_path)
        context["full4k_label_coverage"] = _context_entry(full4k_label_coverage_summary_path, data)
    return context


def _context_entry(path: Path, data: Mapping[str, Any]) -> dict[str, Any]:
    decision = _mapping(data.get("decision"))
    metrics = _mapping(data.get("metrics"))
    stability = _mapping(data.get("stability"))
    return {
        "path": path.as_posix(),
        "route": str(decision.get("route") or ""),
        "metrics": dict(metrics),
        "stability": dict(stability),
    }


def _expected_metric(context: Mapping[str, Any], section: str, key: str) -> int | None:
    value = _mapping(_mapping(context.get(section)).get("metrics")).get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validate_config(
    *,
    max_steps: int,
    eval_every: int,
    batch_size: int,
    eval_size: int,
    final_train_eval_size: int,
    learning_rate: float,
    lambda_delta_event_factor_target: float,
    max_factor_loss_ratio: float,
    max_cached_maps: int,
) -> None:
    if int(max_steps) <= 0:
        raise ValueError("max_steps must be positive")
    if int(eval_every) <= 0:
        raise ValueError("eval_every must be positive")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(eval_size) <= 0:
        raise ValueError("eval_size must be positive")
    if int(final_train_eval_size) <= 0:
        raise ValueError("final_train_eval_size must be positive")
    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if not math.isfinite(float(lambda_delta_event_factor_target)) or float(lambda_delta_event_factor_target) <= 0.0:
        raise ValueError("lambda_delta_event_factor_target must be positive")
    if not math.isfinite(float(max_factor_loss_ratio)) or float(max_factor_loss_ratio) <= 0.0:
        raise ValueError("max_factor_loss_ratio must be positive")
    if int(max_cached_maps) < 0:
        raise ValueError("max_cached_maps must be non-negative")


def _count_cache_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*.pt") if item.is_file())


def _mapper_record_cache_complete(path: Path) -> bool:
    return path.exists() and path.with_suffix(".json").exists()


def _verification_lines() -> list[str]:
    return [
        "```bash",
        "uv run python -m pulsefield_model.evals.mapper_v3_delta_event_factor_target_full4k_training",
        "uv run --group dev pytest tests/evals/test_mapper_v3_delta_event_factor_target_full4k_training.py tests/evals/test_mapper_v3_delta_event_factor_target_longer_training.py tests/evals/test_mapper_v3_delta_event_factor_target_training_smoke.py -q",
        "uv run python -m py_compile src/pulsefield_model/evals/mapper_v3_delta_event_factor_target_full4k_training.py",
        "python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_full4k_training_summary.json >/dev/null",
        "git diff --check",
        "```",
    ]


def _what_passed(route: str) -> str:
    if route == "TEST_DELTA_EVENT_FACTOR_TARGET_FULL4K_LONGER_OR_PRODUCTION_CONFIG_CARD":
        return (
            "- The real v3 runner completed a bounded full4k fixed-cache factor-target run.\n"
            "- The training source matched prior full4k proxy-audit eligible-window counts.\n"
            "- Multiple eval points had finite token and factor-target losses.\n"
            "- Final eval aggregate `loss/total` accounting remained exact."
        )
    return "- The gate produced an explicit route; inspect failed checks before larger full4k training."


def _what_surfaced(route: str) -> str:
    if route == "TEST_DELTA_EVENT_FACTOR_TARGET_FULL4K_LONGER_OR_PRODUCTION_CONFIG_CARD":
        return (
            "Factor-target supervision is mechanically stable on the full4k fixed-cache path, so the next "
            "question is longer full4k duration or production-config realism."
        )
    if route == "MUTATE_FACTOR_TARGET_FULL4K_CACHE_OR_DATASET":
        return "The limiting issue is full4k dataset/cache readiness, not factor-target model semantics."
    if route == "MUTATE_FACTOR_TARGET_FULL4K_LR_OR_LOSS_WEIGHT":
        return "Losses stayed finite but the factor target branch needs LR, loss-weight, or scale mutation."
    if route == "MUTATE_FACTOR_TARGET_FULL4K_CONTEXT_CHAIN":
        return "The prior evidence chain is incomplete or stale, so this full4k result should not be interpreted."
    return "The full4k factor-target training path needs runner, metric, or supervision repair before scaling."


def _finite_positive(value: object) -> bool:
    number = _float(value)
    return math.isfinite(number) and number > 0.0


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _float(value: object, *, default: float = math.nan) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float(default)


def _fmt(value: object) -> str:
    number = _float(value)
    return "n/a" if not math.isfinite(number) else f"{number:.6f}"


def _row(label: str, value: object) -> str:
    return f"| {label} | `{value}` |"


def _metric_row(
    label: str,
    first_eval: Mapping[str, Any],
    final_train: Mapping[str, Any],
    final_eval: Mapping[str, Any],
) -> str:
    return (
        f"| {label} | `{_fmt(first_eval.get(label))}` | "
        f"`{_fmt(final_train.get(label))}` | `{_fmt(final_eval.get(label))}` |"
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the v3 delta-event factor-target full4k fixed-cache gate.")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--index-path", default=DEFAULT_INDEX_PATH)
    parser.add_argument("--control-teacher-cache-dir", default=DEFAULT_CONTROL_TEACHER_CACHE_DIR)
    parser.add_argument("--mapper-record-cache-path", default=DEFAULT_MAPPER_RECORD_CACHE_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--report-output", default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-size", type=int, default=16)
    parser.add_argument("--final-train-eval-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lambda-delta-event-factor-target", type=float, default=0.25)
    parser.add_argument("--max-factor-loss-ratio", type=float, default=1.20)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-cached-maps", type=int, default=8)
    parser.add_argument("--dataset-progress", action="store_true", default=False)
    args = parser.parse_args(argv)
    summary = run_delta_event_factor_target_full4k_training(
        dataset_root=Path(args.dataset_root),
        index_path=Path(args.index_path),
        control_teacher_cache_dir=Path(args.control_teacher_cache_dir),
        mapper_record_cache_path=Path(args.mapper_record_cache_path),
        output_dir=Path(args.output_dir),
        summary_output_path=Path(args.summary_output),
        report_output_path=Path(args.report_output) if args.report_output else None,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        batch_size=args.batch_size,
        eval_size=args.eval_size,
        final_train_eval_size=args.final_train_eval_size,
        learning_rate=args.learning_rate,
        lambda_delta_event_factor_target=args.lambda_delta_event_factor_target,
        max_factor_loss_ratio=args.max_factor_loss_ratio,
        seed=args.seed,
        device_name=args.device,
        max_cached_maps=args.max_cached_maps,
        dataset_progress=args.dataset_progress,
    )
    print(
        "mapper_v3_delta_event_factor_target_full4k_training_done "
        f"route={summary['decision']['route']} "
        f"source_windows={summary['dataset']['source_window_count']} "
        f"factor_loss_ratio={summary['stability']['factor_loss_ratio']:.6f} "
        f"summary={Path(args.summary_output).as_posix()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
