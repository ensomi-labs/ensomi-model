from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from pulsefield_model.evals.mapper_v3_delta_event_factor_target_training_smoke import (
    _loss_total_check,
    _model_config_overrides,
)
from pulsefield_model.training.mapper_v3 import run_mapper_v3_phase_b_training


SUMMARY_SCHEMA_VERSION = 1
REPORT_ROOT = Path("artifacts/reports/audits/mapper_v2_1_grammar")
DEFAULT_INDEX_PATH = Path("artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet")
DEFAULT_CONTROL_TEACHER_CACHE_DIR = Path("artifacts/cache/stage2_mapper_v2_1/control_teacher_d384_l3_stride16_step002000")
DEFAULT_OUTPUT_DIR = Path("artifacts/runs/stage2_mapper_v3/delta_event_factor_target_longer_training")
DEFAULT_EXPERIMENT_CARD = REPORT_ROOT / "target_grammar_v3_delta_event_factor_target_longer_training_experiment_card.md"
DEFAULT_SUMMARY_OUTPUT = REPORT_ROOT / "target_grammar_v3_delta_event_factor_target_longer_training_summary.json"
DEFAULT_REPORT_OUTPUT = REPORT_ROOT / "target_grammar_v3_delta_event_factor_target_longer_training_result_report.md"
DEFAULT_TRAINING_SMOKE_SUMMARY = REPORT_ROOT / "target_grammar_v3_delta_event_factor_target_training_smoke_summary.json"


def run_delta_event_factor_target_longer_training(
    *,
    dataset_root: str | Path = Path("dataset"),
    index_path: str | Path = DEFAULT_INDEX_PATH,
    control_teacher_cache_dir: str | Path = DEFAULT_CONTROL_TEACHER_CACHE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    summary_output_path: str | Path = DEFAULT_SUMMARY_OUTPUT,
    report_output_path: str | Path | None = DEFAULT_REPORT_OUTPUT,
    experiment_card_path: str | Path = DEFAULT_EXPERIMENT_CARD,
    training_smoke_summary_path: str | Path = DEFAULT_TRAINING_SMOKE_SUMMARY,
    max_steps: int = 16,
    eval_every: int = 8,
    batch_size: int = 4,
    eval_size: int = 16,
    final_train_eval_size: int = 16,
    learning_rate: float = 3e-4,
    lambda_delta_event_factor_target: float = 0.25,
    max_factor_loss_ratio: float = 1.15,
    seed: int = 20260702,
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
        max_factor_loss_ratio=max_factor_loss_ratio,
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
        log_every=int(eval_every),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        weight_decay=0.0,
        seed=int(seed),
        device_name=str(device_name),
        run_name="mapper_v3_delta_event_factor_target_longer_training",
        eval_size=int(eval_size),
        final_train_eval_size=int(final_train_eval_size),
        num_workers=0,
        max_cached_maps=16,
        control_teacher_cache_dir=Path(control_teacher_cache_dir),
        require_control_teacher_cache=True,
        include_full_song_context=False,
        skip_first_eval_pass=False,
        model_config_overrides=model_config_overrides,
        control_model_config_overrides={},
        loss_config_overrides=loss_config_overrides,
    )
    training_report = _read_json(result.report_path)
    context = _load_context(Path(training_smoke_summary_path))
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
        stability=stability,
        loss_total_check=loss_total_check,
        context=context,
    )
    decision = decision_from_checks(checks)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar v3 delta-event factor target longer training",
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
            "max_factor_loss_ratio": float(max_factor_loss_ratio),
            "seed": int(seed),
            "device": str(device_name),
            "model": model_config_overrides,
            "loss": loss_config_overrides,
            "no_inference_or_rollout_change": True,
            "uses_c3_backreference": False,
            "uses_future_lookup": False,
        },
        "context": context,
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


def guard_checks(
    *,
    stability: Mapping[str, Any],
    loss_total_check: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, bool]:
    final_eval = _mapping(stability.get("final_eval_metrics"))
    final_train = _mapping(stability.get("final_train_metrics"))
    return {
        "training_smoke_route_positive": _mapping(context.get("training_smoke")).get("route")
        == "TEST_DELTA_EVENT_FACTOR_TARGET_LONGER_TRAINING_CARD",
        "completed_requested_steps": int(stability.get("completed_steps") or -1) == int(stability.get("max_steps") or -2),
        "training_marked_complete": bool(stability.get("is_complete")),
        "report_exists": bool(stability.get("report_exists")),
        "checkpoint_exists": bool(stability.get("checkpoint_exists")),
        "dataset_factor_target_enabled": bool(stability.get("dataset_include_delta_event_factor_target")),
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
            "route": "TEST_DELTA_EVENT_FACTOR_TARGET_FULL4K_TRAINING_CARD",
            "reason": "factor-target enabled v3 training stayed finite and non-exploding across a longer fixed-slice run",
            "next_step": "Create a bounded full4k/fixed-cache factor-target training card before rollout or replacement claims.",
        }
    if "factor_loss_ratio_within_limit" in failed and "all_eval_losses_finite" not in failed:
        return {
            "route": "MUTATE_FACTOR_TARGET_LR_OR_LOSS_WEIGHT",
            "reason": "factor-target losses stayed finite but exceeded the non-explosion ratio: " + ", ".join(failed),
            "next_step": "Tune learning rate, factor loss weight, or model scale before full4k training.",
        }
    return {
        "route": "MUTATE_FACTOR_TARGET_LONGER_TRAINING_PLUMBING",
        "reason": "longer factor-target training failed checks: " + ", ".join(failed),
        "next_step": "Repair metrics, runner behavior, or factor-target supervision before scaling.",
    }


def write_summary_json(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_markdown(summary).rstrip() + "\n", encoding="utf-8")


def report_markdown(summary: Mapping[str, Any]) -> str:
    decision = _mapping(summary.get("decision"))
    stability = _mapping(summary.get("stability"))
    first_eval = _mapping(stability.get("first_eval_metrics"))
    final_eval = _mapping(stability.get("final_eval_metrics"))
    final_train = _mapping(stability.get("final_train_metrics"))
    loss_total = _mapping(summary.get("loss_total_check"))
    checks = _mapping(summary.get("checks"))
    lines = [
        "# Target Grammar v3 Delta-Event Factor Target Longer Training Result Report",
        "",
        "## Scope",
        "",
        "This gate runs a longer fixed-slice v3 training stability check with factorized delta-event target supervision enabled. It does not claim full4k stability, rollout quality, online factor-row inference, or complete v3 replacement readiness.",
        "",
        "## Decision",
        "",
        f"Route: `{decision.get('route')}`.",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Recommended next step: {decision.get('next_step')}",
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
        _row("dataset factor target", stability.get("dataset_include_delta_event_factor_target")),
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
            "- This does not prove full4k training stability.",
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


def _stability_metrics(
    report: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    max_factor_loss_ratio: float,
) -> dict[str, Any]:
    eval_points = _eval_points(report)
    first_eval = eval_points[0]["metrics"] if eval_points else {}
    final_eval = _mapping(report.get("final_eval_metrics"))
    final_train = _mapping(report.get("final_train_metrics"))
    first_factor = _float(_mapping(first_eval).get("loss/delta_event_factor_target"))
    final_factor = _float(final_eval.get("loss/delta_event_factor_target"))
    ratio = _safe_ratio(final_factor, first_factor)
    all_eval_losses_finite = bool(eval_points) and all(
        _finite_positive(point["metrics"].get("loss/total"))
        and _finite_positive(point["metrics"].get("loss/token"))
        and _finite_positive(point["metrics"].get("loss/delta_event_factor_target"))
        for point in eval_points
    )
    dataset = _mapping(report.get("dataset"))
    model_config = _mapping(report.get("model_config"))
    loss_config = _mapping(report.get("loss_config"))
    return {
        "completed_steps": report.get("completed_steps"),
        "max_steps": report.get("max_steps"),
        "is_complete": report.get("is_complete"),
        "report_path": _report_path_from_checkpoint(checkpoint_path).as_posix(),
        "checkpoint_path": checkpoint_path.as_posix(),
        "report_exists": _report_path_from_checkpoint(checkpoint_path).exists(),
        "checkpoint_exists": checkpoint_path.exists(),
        "dataset_include_delta_event_factor_target": dataset.get("include_delta_event_factor_target"),
        "model_use_delta_event_factor_target": model_config.get("use_delta_event_factor_target"),
        "loss_lambda_delta_event_factor_target": loss_config.get("lambda_delta_event_factor_target"),
        "eval_steps": [point["step"] for point in eval_points],
        "eval_point_count": len(eval_points),
        "first_eval_metrics": dict(first_eval),
        "final_eval_metrics": dict(final_eval),
        "final_train_metrics": dict(final_train),
        "first_eval_factor_loss": float(first_factor),
        "final_eval_factor_loss": float(final_factor),
        "factor_loss_ratio": float(ratio),
        "max_factor_loss_ratio": float(max_factor_loss_ratio),
        "factor_loss_ratio_within_limit": bool(ratio <= float(max_factor_loss_ratio)),
        "all_eval_losses_finite": bool(all_eval_losses_finite),
        "eval_factor_losses": [
            float(_float(point["metrics"].get("loss/delta_event_factor_target"))) for point in eval_points
        ],
        "eval_total_losses": [float(_float(point["metrics"].get("loss/total"))) for point in eval_points],
    }


def _eval_points(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    history = report.get("history")
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes, bytearray)):
        return points
    for entry in history:
        item = _mapping(entry)
        metrics = _mapping(item.get("eval"))
        if metrics:
            points.append({"step": int(item.get("step") or 0), "metrics": dict(metrics)})
    return points


def _report_path_from_checkpoint(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name("report.json")


def _load_context(path: Path) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if path.exists():
        data = _read_json(path)
        context["training_smoke"] = {
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
    max_factor_loss_ratio: float,
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
        "uv run python -m pulsefield_model.evals.mapper_v3_delta_event_factor_target_longer_training",
        "uv run --group dev pytest tests/evals/test_mapper_v3_delta_event_factor_target_longer_training.py tests/evals/test_mapper_v3_delta_event_factor_target_training_smoke.py -q",
        "uv run python -m py_compile src/pulsefield_model/evals/mapper_v3_delta_event_factor_target_longer_training.py",
        "python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_longer_training_summary.json >/dev/null",
        "git diff --check",
        "```",
    ]


def _what_passed(route: str) -> str:
    if route == "TEST_DELTA_EVENT_FACTOR_TARGET_FULL4K_TRAINING_CARD":
        return (
            "- The real v3 runner completed the longer fixed-slice factor-target run.\n"
            "- Multiple eval points had finite token and factor-target losses.\n"
            "- Final eval factor loss stayed within the non-explosion threshold.\n"
            "- Final eval aggregate `loss/total` accounting remained exact."
        )
    return "- The gate produced an explicit route; inspect failed checks before full4k training."


def _what_surfaced(route: str) -> str:
    if route == "TEST_DELTA_EVENT_FACTOR_TARGET_FULL4K_TRAINING_CARD":
        return "Factor-target supervision is stable enough on the fixed slice to justify a bounded full4k training card."
    if route == "MUTATE_FACTOR_TARGET_LR_OR_LOSS_WEIGHT":
        return "Losses stayed finite but the factor target branch needs LR, loss-weight, or scale mutation before full4k training."
    return "The longer fixed-slice run needs metric, runner, or supervision repair before scaling."


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


def _float(value: object, *, default: float = math.nan) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float(default)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(float(numerator)) or not math.isfinite(float(denominator)):
        return math.inf
    if float(denominator) <= 0.0:
        return math.inf
    return float(numerator) / float(denominator)


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
    parser = argparse.ArgumentParser(description="Run the v3 delta-event factor-target longer training gate.")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--index-path", default=DEFAULT_INDEX_PATH)
    parser.add_argument("--control-teacher-cache-dir", default=DEFAULT_CONTROL_TEACHER_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--report-output", default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-size", type=int, default=16)
    parser.add_argument("--final-train-eval-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lambda-delta-event-factor-target", type=float, default=0.25)
    parser.add_argument("--max-factor-loss-ratio", type=float, default=1.15)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    summary = run_delta_event_factor_target_longer_training(
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
        max_factor_loss_ratio=args.max_factor_loss_ratio,
        seed=args.seed,
        device_name=args.device,
    )
    print(
        "mapper_v3_delta_event_factor_target_longer_training_done "
        f"route={summary['decision']['route']} "
        f"factor_loss_ratio={summary['stability']['factor_loss_ratio']:.6f} "
        f"summary={Path(args.summary_output).as_posix()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
