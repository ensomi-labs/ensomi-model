from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from pulsefield_model.training.mapper_v3 import run_mapper_v3_phase_b_training


SUMMARY_SCHEMA_VERSION = 1
REPORT_ROOT = Path("artifacts/reports/audits/mapper_v2_1_grammar")
DEFAULT_INDEX_PATH = Path("artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet")
DEFAULT_CONTROL_TEACHER_CACHE_DIR = Path("artifacts/cache/stage2_mapper_v2_1/control_teacher_d384_l3_stride16_step002000")
DEFAULT_OUTPUT_DIR = Path("artifacts/runs/stage2_mapper_v3/delta_event_factor_target_training_smoke")
DEFAULT_EXPERIMENT_CARD = REPORT_ROOT / "target_grammar_v3_delta_event_factor_target_training_smoke_experiment_card.md"
DEFAULT_SUMMARY_OUTPUT = REPORT_ROOT / "target_grammar_v3_delta_event_factor_target_training_smoke_summary.json"
DEFAULT_REPORT_OUTPUT = REPORT_ROOT / "target_grammar_v3_delta_event_factor_target_training_smoke_result_report.md"
DEFAULT_MODEL_LOSS_SUMMARY = REPORT_ROOT / "target_grammar_v3_delta_event_factor_target_model_loss_plumbing_summary.json"


def run_delta_event_factor_target_training_smoke(
    *,
    dataset_root: str | Path = Path("dataset"),
    index_path: str | Path = DEFAULT_INDEX_PATH,
    control_teacher_cache_dir: str | Path = DEFAULT_CONTROL_TEACHER_CACHE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    summary_output_path: str | Path = DEFAULT_SUMMARY_OUTPUT,
    report_output_path: str | Path | None = DEFAULT_REPORT_OUTPUT,
    experiment_card_path: str | Path = DEFAULT_EXPERIMENT_CARD,
    model_loss_summary_path: str | Path = DEFAULT_MODEL_LOSS_SUMMARY,
    max_steps: int = 2,
    eval_every: int = 2,
    batch_size: int = 2,
    eval_size: int = 8,
    final_train_eval_size: int = 8,
    learning_rate: float = 2e-4,
    lambda_delta_event_factor_target: float = 0.25,
    seed: int = 20260701,
    device_name: str = "cpu",
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
    )
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
        log_every=1,
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        weight_decay=0.0,
        seed=int(seed),
        device_name=str(device_name),
        run_name="mapper_v3_delta_event_factor_target_training_smoke",
        eval_size=int(eval_size),
        final_train_eval_size=int(final_train_eval_size),
        num_workers=0,
        max_cached_maps=16,
        control_teacher_cache_dir=Path(control_teacher_cache_dir),
        require_control_teacher_cache=True,
        include_full_song_context=False,
        skip_first_eval_pass=True,
        model_config_overrides=model_config_overrides,
        control_model_config_overrides={},
        loss_config_overrides=loss_config_overrides,
    )
    training_report = _read_json(result.report_path)
    context = _load_context(Path(model_loss_summary_path))
    metrics = _training_metrics(training_report, result_checkpoint_path=result.checkpoint_path)
    loss_total_check = _loss_total_check(
        _mapping(training_report.get("final_eval_metrics")),
        _mapping(training_report.get("loss_config")),
    )
    checks = guard_checks(
        metrics=metrics,
        loss_total_check=loss_total_check,
        context=context,
    )
    decision = decision_from_checks(checks)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar v3 delta-event factor target training smoke",
        "experiment_card": Path(experiment_card_path).as_posix(),
        "elapsed_s": time.monotonic() - started,
        "index_path": Path(index_path).as_posix(),
        "control_teacher_cache_dir": Path(control_teacher_cache_dir).as_posix(),
        "training_output_dir": Path(output_dir).as_posix(),
        "config": {
            "max_steps": int(max_steps),
            "eval_every": int(eval_every),
            "batch_size": int(batch_size),
            "eval_size": int(eval_size),
            "final_train_eval_size": int(final_train_eval_size),
            "learning_rate": float(learning_rate),
            "lambda_delta_event_factor_target": float(lambda_delta_event_factor_target),
            "seed": int(seed),
            "device": str(device_name),
            "model": model_config_overrides,
            "loss": loss_config_overrides,
            "no_inference_or_rollout_change": True,
            "uses_c3_backreference": False,
            "uses_future_lookup": False,
        },
        "context": context,
        "training": metrics,
        "loss_total_check": loss_total_check,
        "checks": checks,
        "decision": decision,
        "next_step": decision["next_step"],
    }
    write_summary_json(summary, Path(summary_output_path))
    if report_output_path is not None:
        write_report(summary, Path(report_output_path))
    return summary


def guard_checks(
    *,
    metrics: Mapping[str, Any],
    loss_total_check: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, bool]:
    model_loss = _mapping(context.get("model_loss_plumbing"))
    final_eval = _mapping(metrics.get("final_eval_metrics"))
    final_train = _mapping(metrics.get("final_train_metrics"))
    last_train = _mapping(metrics.get("last_train_metrics"))
    return {
        "model_loss_route_positive": model_loss.get("route") == "TEST_DELTA_EVENT_FACTOR_TARGET_TRAINING_SMOKE_CARD",
        "completed_requested_steps": int(metrics.get("completed_steps") or -1) == int(metrics.get("max_steps") or -2),
        "training_marked_complete": bool(metrics.get("is_complete")),
        "report_exists": bool(metrics.get("report_exists")),
        "checkpoint_exists": bool(metrics.get("checkpoint_exists")),
        "dataset_factor_target_enabled": bool(metrics.get("dataset_include_delta_event_factor_target")),
        "model_factor_target_enabled": bool(metrics.get("model_use_delta_event_factor_target")),
        "loss_factor_lambda_positive": float(metrics.get("loss_lambda_delta_event_factor_target") or 0.0) > 0.0,
        "final_eval_loss_total_finite": _finite_positive(final_eval.get("loss/total"), allow_zero=False),
        "final_eval_factor_loss_finite": _finite_positive(final_eval.get("loss/delta_event_factor_target"), allow_zero=False),
        "final_train_factor_loss_finite": _finite_positive(final_train.get("loss/delta_event_factor_target"), allow_zero=False),
        "last_train_factor_loss_finite": _finite_positive(last_train.get("loss/delta_event_factor_target"), allow_zero=False),
        "final_eval_factor_lambda_positive": float(final_eval.get("phase/lambda_delta_event_factor_target") or 0.0) > 0.0,
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
            "route": "TEST_DELTA_EVENT_FACTOR_TARGET_LONGER_TRAINING_CARD",
            "reason": "factor-target enabled v3 training runner completed a bounded real-data smoke with valid metrics",
            "next_step": "Create a bounded longer-training card before rollout or replacement claims.",
        }
    if "loss_total_recomputed_matches" in failed:
        return {
            "route": "MUTATE_FACTOR_TARGET_TRAINING_METRIC_ACCOUNTING",
            "reason": "training smoke exposed incorrect aggregate loss accounting: " + ", ".join(failed),
            "next_step": "Repair mapper metric finalization before longer factor-target training.",
        }
    return {
        "route": "MUTATE_FACTOR_TARGET_TRAINING_SMOKE",
        "reason": "factor-target training smoke failed checks: " + ", ".join(failed),
        "next_step": "Repair the training runner, dataset enablement, or factor metrics before longer training.",
    }


def write_summary_json(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_markdown(summary).rstrip() + "\n", encoding="utf-8")


def report_markdown(summary: Mapping[str, Any]) -> str:
    decision = _mapping(summary.get("decision"))
    training = _mapping(summary.get("training"))
    final_eval = _mapping(training.get("final_eval_metrics"))
    final_train = _mapping(training.get("final_train_metrics"))
    last_train = _mapping(training.get("last_train_metrics"))
    loss_total = _mapping(summary.get("loss_total_check"))
    checks = _mapping(summary.get("checks"))
    lines = [
        "# Target Grammar v3 Delta-Event Factor Target Training Smoke Result Report",
        "",
        "## Scope",
        "",
        "This gate runs a short production v3 training smoke with factorized delta-event target supervision enabled. It does not claim stable training, rollout quality, online factor-row inference, or complete v3 replacement readiness.",
        "",
        "## Decision",
        "",
        f"Route: `{decision.get('route')}`.",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Recommended next step: {decision.get('next_step')}",
        "",
        "## Training Run",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        _row("completed steps", training.get("completed_steps")),
        _row("max steps", training.get("max_steps")),
        _row("is complete", training.get("is_complete")),
        _row("report path", training.get("report_path")),
        _row("checkpoint path", training.get("checkpoint_path")),
        _row("dataset factor target", training.get("dataset_include_delta_event_factor_target")),
        _row("model factor target", training.get("model_use_delta_event_factor_target")),
        _row("loss factor lambda", training.get("loss_lambda_delta_event_factor_target")),
        "",
        "## Metrics",
        "",
        "| Metric | Last Train | Final Train | Final Eval |",
        "| --- | ---: | ---: | ---: |",
        _metric_row("loss/total", last_train, final_train, final_eval),
        _metric_row("loss/token", last_train, final_train, final_eval),
        _metric_row("loss/delta_event_factor_target", last_train, final_train, final_eval),
        _metric_row("phase/lambda_delta_event_factor_target", last_train, final_train, final_eval),
        _metric_row("delta_event_factor/kind_label_count", last_train, final_train, final_eval),
        _metric_row("delta_event_factor/delta_label_count", last_train, final_train, final_eval),
        _metric_row("delta_event_factor/signature_label_count", last_train, final_train, final_eval),
        _metric_row("delta_event_factor/end_gap_label_count", last_train, final_train, final_eval),
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
            "- This does not prove long-run training stability.",
            "- This does not prove generated chart quality.",
            "- This does not implement factor-row inference or full replacement.",
            "",
            "## Verification",
            "",
            *_verification_lines(summary),
            "",
            "## Next Step",
            "",
            str(decision.get("next_step")),
        ]
    )
    return "\n".join(lines)


def _model_config_overrides() -> dict[str, Any]:
    return {
        "d_model": 32,
        "heads": 4,
        "layers": 1,
        "ffn_dim": 64,
        "dropout": 0.0,
        "max_seq_len": 1024,
        "state_prior_hidden_dim": 32,
        "ln_close_hidden_dim": 32,
        "lane_embedding_dim": 4,
        "age_embedding_dim": 4,
        "use_global_context": False,
        "global_conv_blocks": 0,
        "use_delta_event_factor_target": True,
    }


def _training_metrics(report: Mapping[str, Any], *, result_checkpoint_path: Path) -> dict[str, Any]:
    dataset = _mapping(report.get("dataset"))
    model_config = _mapping(report.get("model_config"))
    loss_config = _mapping(report.get("loss_config"))
    report_path = Path(str(report.get("_source_report_path") or ""))
    del report_path
    return {
        "completed_steps": report.get("completed_steps"),
        "max_steps": report.get("max_steps"),
        "is_complete": report.get("is_complete"),
        "report_path": str(_report_path_from_checkpoint(result_checkpoint_path)),
        "checkpoint_path": result_checkpoint_path.as_posix(),
        "report_exists": _report_path_from_checkpoint(result_checkpoint_path).exists(),
        "checkpoint_exists": result_checkpoint_path.exists(),
        "dataset_include_delta_event_factor_target": dataset.get("include_delta_event_factor_target"),
        "model_use_delta_event_factor_target": model_config.get("use_delta_event_factor_target"),
        "loss_lambda_delta_event_factor_target": loss_config.get("lambda_delta_event_factor_target"),
        "last_train_metrics": _mapping(report.get("last_train_metrics")),
        "final_train_metrics": _mapping(report.get("final_train_metrics")),
        "final_eval_metrics": _mapping(report.get("final_eval_metrics")),
    }


def _report_path_from_checkpoint(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name("report.json")


def _loss_total_check(metrics: Mapping[str, Any], loss_config: Mapping[str, Any]) -> dict[str, Any]:
    reported = _float(metrics.get("loss/total"))
    recomputed = (
        _float(metrics.get("loss/token"), default=0.0)
        + _float(loss_config.get("lambda_ln_close"), default=0.0) * _float(metrics.get("loss/ln_close"), default=0.0)
        + _float(loss_config.get("lambda_adapter_reg"), default=0.0) * _float(metrics.get("loss/adapter_reg"), default=0.0)
        + _float(loss_config.get("lambda_density"), default=0.0) * _float(metrics.get("loss/density"), default=0.0)
        + _float(loss_config.get("lambda_event_budget"), default=0.0) * _float(metrics.get("loss/event_budget"), default=0.0)
        + _float(loss_config.get("lambda_conditioned_event_distribution"), default=0.0)
        * _float(metrics.get("loss/conditioned_event_distribution"), default=0.0)
        + _float(loss_config.get("lambda_continuation_jump"), default=0.0)
        * _float(metrics.get("loss/continuation_jump"), default=0.0)
        + _float(loss_config.get("lambda_time_shift_distance"), default=0.0)
        * _float(metrics.get("loss/time_shift_distance"), default=0.0)
        + _float(loss_config.get("lambda_c3_auxiliary"), default=0.0) * _float(metrics.get("loss/c3_auxiliary"), default=0.0)
        + _float(loss_config.get("lambda_delta_event_auxiliary"), default=0.0)
        * _float(metrics.get("loss/delta_event_auxiliary"), default=0.0)
        + _float(loss_config.get("lambda_delta_event_factor_target"), default=0.0)
        * _float(metrics.get("loss/delta_event_factor_target"), default=0.0)
    )
    absolute_delta = abs(float(reported) - float(recomputed)) if math.isfinite(float(reported)) else math.inf
    return {
        "reported": float(reported),
        "recomputed": float(recomputed),
        "absolute_delta": float(absolute_delta),
        "matches": bool(absolute_delta <= 1e-5),
    }


def _load_context(path: Path) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if path.exists():
        data = _read_json(path)
        context["model_loss_plumbing"] = {
            "path": path.as_posix(),
            "route": str(_mapping(data.get("decision")).get("route") or ""),
        }
    return context


def _validate_config(
    *,
    max_steps: int,
    eval_every: int,
    batch_size: int,
    eval_size: int,
    final_train_eval_size: int,
    learning_rate: float,
    lambda_delta_event_factor_target: float,
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


def _verification_lines(summary: Mapping[str, Any]) -> list[str]:
    commands = summary.get("commands")
    if isinstance(commands, Sequence) and not isinstance(commands, (str, bytes, bytearray)):
        lines = ["```bash"]
        results: list[str] = []
        for item in commands:
            command = _mapping(item).get("command")
            result = _mapping(item).get("result")
            if command:
                lines.append(str(command))
            if result:
                results.append(str(result))
        lines.append("```")
        if results:
            lines.extend(["", "Results:", *[f"- {result}" for result in results]])
        return lines
    return [
        "```bash",
        "uv run python -m pulsefield_model.evals.mapper_v3_delta_event_factor_target_training_smoke",
        "uv run --group dev pytest tests/training/test_mapper_training_runner.py tests/training/test_mapper_v3.py tests/evals/test_mapper_v3_delta_event_factor_target_training_smoke.py -q",
        "uv run python -m py_compile src/pulsefield_model/training/mapper_runner.py src/pulsefield_model/training/mapper_common.py src/pulsefield_model/evals/mapper_v3_delta_event_factor_target_training_smoke.py",
        "python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_training_smoke_summary.json >/dev/null",
        "git diff --check",
        "```",
    ]


def _what_passed(route: str) -> str:
    if route == "TEST_DELTA_EVENT_FACTOR_TARGET_LONGER_TRAINING_CARD":
        return (
            "- The real v3 training runner completed the factor-target smoke.\n"
            "- The report and checkpoint were written.\n"
            "- Final train/eval metrics include finite factor-target losses and label counts.\n"
            "- Aggregated final eval `loss/total` includes the enabled factor-target term."
        )
    return "- The gate produced an explicit route; inspect failed checks before longer training."


def _what_surfaced(route: str) -> str:
    if route == "TEST_DELTA_EVENT_FACTOR_TARGET_LONGER_TRAINING_CARD":
        return "Factor-target supervision is runner-compatible for a short smoke; the next gate is longer training stability."
    if route == "MUTATE_FACTOR_TARGET_TRAINING_METRIC_ACCOUNTING":
        return "The training path runs far enough to expose metric-accounting drift; repair totals before trusting reports."
    return "The factor-target training smoke needs runner, dataset, or metric repair before scaling."


def _finite_positive(value: object, *, allow_zero: bool) -> bool:
    number = _float(value)
    if not math.isfinite(number):
        return False
    return number >= 0.0 if allow_zero else number > 0.0


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


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
    last_train: Mapping[str, Any],
    final_train: Mapping[str, Any],
    final_eval: Mapping[str, Any],
) -> str:
    return (
        f"| {label} | `{_fmt(last_train.get(label))}` | "
        f"`{_fmt(final_train.get(label))}` | `{_fmt(final_eval.get(label))}` |"
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the v3 delta-event factor-target training smoke.")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--index-path", default=DEFAULT_INDEX_PATH)
    parser.add_argument("--control-teacher-cache-dir", default=DEFAULT_CONTROL_TEACHER_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--report-output", default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-size", type=int, default=8)
    parser.add_argument("--final-train-eval-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lambda-delta-event-factor-target", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    summary = run_delta_event_factor_target_training_smoke(
        dataset_root=Path(args.dataset_root),
        index_path=Path(args.index_path),
        control_teacher_cache_dir=Path(args.control_teacher_cache_dir),
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
        seed=args.seed,
        device_name=args.device,
    )
    print(
        "mapper_v3_delta_event_factor_target_training_smoke_done "
        f"route={summary['decision']['route']} "
        f"final_eval_factor_loss={summary['training']['final_eval_metrics']['loss/delta_event_factor_target']:.6f} "
        f"summary={Path(args.summary_output).as_posix()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
