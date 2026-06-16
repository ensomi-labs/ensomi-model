from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SUMMARY_SCHEMA_VERSION = 1
EXPECTED_V21_CONTRACT = "v2.1_sparse_lane_actions"
EXPECTED_V3_CONTRACT = "v3_event_groups"


@dataclass(frozen=True)
class MapperTrainingReportSummary:
    label: str
    path: str
    run_name: str | None
    mapper_token_contract: str | None
    dataset_mapper_token_contract: str | None
    completed_steps: int
    is_complete: bool
    final_eval_loss_total: float
    final_train_loss_total: float
    last_train_loss_total: float
    final_eval_valid_tokens: float
    final_train_valid_tokens: float
    train_window_count: int | None
    eval_window_count: int | None
    source_window_count: int | None
    parameter_count: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "path": self.path,
            "run_name": self.run_name,
            "mapper_token_contract": self.mapper_token_contract,
            "dataset_mapper_token_contract": self.dataset_mapper_token_contract,
            "completed_steps": self.completed_steps,
            "is_complete": self.is_complete,
            "final_eval_loss_total": self.final_eval_loss_total,
            "final_train_loss_total": self.final_train_loss_total,
            "last_train_loss_total": self.last_train_loss_total,
            "final_eval_valid_tokens": self.final_eval_valid_tokens,
            "final_train_valid_tokens": self.final_train_valid_tokens,
            "train_window_count": self.train_window_count,
            "eval_window_count": self.eval_window_count,
            "source_window_count": self.source_window_count,
            "parameter_count": self.parameter_count,
        }


def load_training_report_summary(
    path: str | Path,
    *,
    label: str,
) -> MapperTrainingReportSummary:
    report_path = Path(path)
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid mapper training report JSON: {report_path}") from exc
    if not isinstance(report, Mapping):
        raise ValueError(f"mapper training report must be a JSON object: {report_path}")

    training_config = _mapping(report.get("training_config"))
    dataset = _mapping(training_config.get("dataset")) or _mapping(report.get("dataset"))
    final_eval = _mapping(report.get("final_eval_metrics"))
    final_train = _mapping(report.get("final_train_metrics"))
    last_train = _mapping(report.get("last_train_metrics"))
    return MapperTrainingReportSummary(
        label=str(label),
        path=report_path.as_posix(),
        run_name=_optional_str(report.get("run_name")),
        mapper_token_contract=_optional_str(training_config.get("mapper_token_contract")),
        dataset_mapper_token_contract=_optional_str(dataset.get("mapper_token_contract")),
        completed_steps=_int_value(report.get("completed_steps"), name=f"{label}.completed_steps"),
        is_complete=bool(report.get("is_complete", False)),
        final_eval_loss_total=_float_metric(final_eval, "loss/total", label=f"{label}.final_eval_metrics"),
        final_train_loss_total=_float_metric(final_train, "loss/total", label=f"{label}.final_train_metrics"),
        last_train_loss_total=_float_metric(last_train, "loss/total", label=f"{label}.last_train_metrics"),
        final_eval_valid_tokens=_float_metric(final_eval, "token/valid_count", label=f"{label}.final_eval_metrics"),
        final_train_valid_tokens=_float_metric(final_train, "token/valid_count", label=f"{label}.final_train_metrics"),
        train_window_count=_optional_int(dataset.get("train_window_count")),
        eval_window_count=_optional_int(dataset.get("eval_window_count")),
        source_window_count=_optional_int(dataset.get("source_window_count")),
        parameter_count=_optional_int(report.get("parameter_count")),
    )


def compare_mapper_training_reports(
    *,
    v21: MapperTrainingReportSummary,
    v3: MapperTrainingReportSummary,
    min_completed_steps: int = 1,
) -> dict[str, Any]:
    checks = {
        "v21_contract": v21.mapper_token_contract == EXPECTED_V21_CONTRACT,
        "v21_dataset_contract": v21.dataset_mapper_token_contract == EXPECTED_V21_CONTRACT,
        "v3_contract": v3.mapper_token_contract == EXPECTED_V3_CONTRACT,
        "v3_dataset_contract": v3.dataset_mapper_token_contract == EXPECTED_V3_CONTRACT,
        "v21_completed_steps": int(v21.completed_steps) >= int(min_completed_steps),
        "v3_completed_steps": int(v3.completed_steps) >= int(min_completed_steps),
        "v21_complete_flag": bool(v21.is_complete),
        "v3_complete_flag": bool(v3.is_complete),
        "v21_eval_loss_finite": _is_finite(v21.final_eval_loss_total),
        "v3_eval_loss_finite": _is_finite(v3.final_eval_loss_total),
        "v21_eval_valid_tokens_positive": v21.final_eval_valid_tokens > 0.0,
        "v3_eval_valid_tokens_positive": v3.final_eval_valid_tokens > 0.0,
    }
    v3_token_ratio = _safe_divide(v3.final_eval_valid_tokens, v21.final_eval_valid_tokens)
    loss_delta = v3.final_eval_loss_total - v21.final_eval_loss_total
    token_reduction_ratio = 1.0 - v3_token_ratio if math.isfinite(v3_token_ratio) else float("nan")
    checks["v3_eval_valid_tokens_lower"] = math.isfinite(v3_token_ratio) and v3_token_ratio < 1.0
    route = "TEST_NEXT" if all(checks.values()) else "MUTATE"
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar v3 tiny trained comparison",
        "decision": {
            "route": route,
            "reason": _decision_reason(checks),
        },
        "checks": checks,
        "metrics": {
            "v3_eval_loss_delta": loss_delta,
            "v3_eval_valid_token_ratio": v3_token_ratio,
            "v3_eval_valid_token_reduction_ratio": token_reduction_ratio,
            "v2_1_eval_loss_total": v21.final_eval_loss_total,
            "v3_eval_loss_total": v3.final_eval_loss_total,
            "v2_1_eval_valid_tokens": v21.final_eval_valid_tokens,
            "v3_eval_valid_tokens": v3.final_eval_valid_tokens,
        },
        "reports": {
            "v2_1": v21.to_dict(),
            "v3": v3.to_dict(),
        },
        "interpretation": _interpretation(route),
        "next_step": (
            "Run a longer bounded v3-vs-v2.1 trained comparison with fixed split and comparable compute."
            if route == "TEST_NEXT"
            else "Repair report/training comparability before longer v3-vs-v2.1 runs."
        ),
    }


def write_comparison_report(summary: Mapping[str, Any], path: str | Path) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_comparison_report_markdown(summary), encoding="utf-8")


def write_summary_json(summary: Mapping[str, Any], path: str | Path) -> None:
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _comparison_report_markdown(summary: Mapping[str, Any]) -> str:
    decision = _mapping(summary.get("decision"))
    checks = _mapping(summary.get("checks"))
    metrics = _mapping(summary.get("metrics"))
    reports = _mapping(summary.get("reports"))
    v21 = _mapping(reports.get("v2_1"))
    v3 = _mapping(reports.get("v3"))
    lines = [
        "# Target Grammar v3 Tiny Trained Comparison Result Report",
        "",
        "## Scope",
        "",
        "This report compares matched tiny v2.1 and v3 mapper training reports produced by the shared mapper runner. It is a comparability and target-length gate, not a mapper-quality result.",
        "",
        "## Result",
        "",
        f"Decision: `{decision.get('route')}`.",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- v2.1 report: `{v21.get('path')}`",
        f"- v3 report: `{v3.get('path')}`",
        f"- v2.1 completed steps: `{v21.get('completed_steps')}`",
        f"- v3 completed steps: `{v3.get('completed_steps')}`",
        f"- v2.1 final eval loss: `{_fmt(metrics.get('v2_1_eval_loss_total'))}`",
        f"- v3 final eval loss: `{_fmt(metrics.get('v3_eval_loss_total'))}`",
        f"- v3 eval loss delta: `{_fmt(metrics.get('v3_eval_loss_delta'))}`",
        f"- v2.1 eval valid tokens: `{_fmt(metrics.get('v2_1_eval_valid_tokens'))}`",
        f"- v3 eval valid tokens: `{_fmt(metrics.get('v3_eval_valid_tokens'))}`",
        f"- v3 valid-token ratio: `{_fmt(metrics.get('v3_eval_valid_token_ratio'))}`",
        f"- v3 valid-token reduction: `{_fmt_ratio(metrics.get('v3_eval_valid_token_reduction_ratio'))}`",
        "",
        "## Checks",
        "",
        "| Check | Passed |",
        "| --- | ---: |",
    ]
    for key, passed in checks.items():
        lines.append(f"| `{key}` | `{bool(passed)}` |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(summary.get("interpretation")),
            "",
            "## What This Does Not Prove",
            "",
            "- It does not prove trained v3 mapper quality.",
            "- It does not prove convergence on the full 4K dataset.",
            "- It does not justify session-runtime/default replacement by itself.",
            "",
            "## Next Step",
            "",
            str(summary.get("next_step")),
            "",
        ]
    )
    return "\n".join(lines)


def _decision_reason(checks: Mapping[str, bool]) -> str:
    failed = [key for key, passed in checks.items() if not bool(passed)]
    if not failed:
        return "all tiny trained-comparison gates passed"
    return "failed checks: " + ", ".join(failed)


def _interpretation(route: str) -> str:
    if route == "TEST_NEXT":
        return (
            "The tiny paired comparison passed: both reports use the expected contracts, both completed, "
            "loss/token metrics are finite, and v3 preserves a lower valid-token count. This supports a "
            "longer bounded trained comparison, but not replacement yet."
        )
    return (
        "The tiny paired comparison did not pass all gates. Do not escalate to longer training until the "
        "failed report, contract, loss, or token-count checks are resolved."
    )


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _int_value(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got bool")
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _float_metric(metrics: Mapping[str, Any], key: str, *, label: str) -> float:
    try:
        value = float(metrics[key])
    except KeyError as exc:
        raise ValueError(f"{label} missing required metric {key!r}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}.{key} must be numeric, got {metrics.get(key)!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label}.{key} must be finite, got {value!r}")
    return value


def _is_finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _safe_divide(numerator: float, denominator: float) -> float:
    if float(denominator) == 0.0:
        return float("nan")
    return float(numerator) / float(denominator)


def _fmt(value: object) -> str:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(numeric):
        return "nan"
    return f"{numeric:.6f}"


def _fmt_ratio(value: object) -> str:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(numeric):
        return "nan"
    return f"{100.0 * numeric:.2f}%"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare tiny trained v2.1 and v3 mapper reports.")
    parser.add_argument("--v2-1-report", required=True)
    parser.add_argument("--v3-report", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--min-completed-steps", type=int, default=1)
    args = parser.parse_args(argv)

    v21 = load_training_report_summary(args.v2_1_report, label="v2.1")
    v3 = load_training_report_summary(args.v3_report, label="v3")
    summary = compare_mapper_training_reports(
        v21=v21,
        v3=v3,
        min_completed_steps=args.min_completed_steps,
    )
    write_summary_json(summary, args.summary_output)
    write_comparison_report(summary, args.report_output)
    print(
        "mapper_v3_training_comparison_done "
        f"route={summary['decision']['route']} "
        f"v3_token_ratio={summary['metrics']['v3_eval_valid_token_ratio']:.6f} "
        f"summary={Path(args.summary_output).as_posix()} "
        f"report={Path(args.report_output).as_posix()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
