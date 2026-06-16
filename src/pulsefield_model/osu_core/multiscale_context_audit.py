from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import pandas as pd


SCHEMA_VERSION: Final[int] = 1
DEFAULT_FORENSIC_REPORT_PATH: Final[Path] = Path(
    "artifacts/reports/audits/occupancy_factorized_tokenization/fallback_forensic_report.json",
)
DEFAULT_FORENSIC_TABLES_PATH: Final[Path] = Path(
    "artifacts/reports/audits/occupancy_factorized_tokenization/fallback_forensic_tables.csv",
)
DEFAULT_REPORT_PATH: Final[Path] = Path(
    "artifacts/reports/audits/occupancy_factorized_tokenization/multiscale_context_report.json",
)
DEFAULT_RESULT_LOG_PATH: Final[Path] = Path(
    "artifacts/reports/audits/occupancy_factorized_tokenization/multiscale_context_result_log.md",
)
DEFAULT_DECISION_TABLE_PATH: Final[Path] = Path(
    "artifacts/reports/audits/occupancy_factorized_tokenization/multiscale_context_decision.csv",
)
BOUNDARY_SHARE_THRESHOLD: Final[float] = 0.35
BOUNDARY_RATE_LIFT_THRESHOLD: Final[float] = 0.20


def audit_multiscale_context(
    *,
    forensic_report_path: str | Path = DEFAULT_FORENSIC_REPORT_PATH,
    forensic_tables_path: str | Path = DEFAULT_FORENSIC_TABLES_PATH,
    report_path: str | Path | None = DEFAULT_REPORT_PATH,
    result_log_path: str | Path | None = DEFAULT_RESULT_LOG_PATH,
    decision_table_path: str | Path | None = DEFAULT_DECISION_TABLE_PATH,
    command: str | None = None,
) -> dict[str, Any]:
    """Run E5 M0 boundary decision audit from full-cache forensic artifacts."""

    started_at = time.perf_counter()
    forensic_report_path = Path(forensic_report_path)
    forensic_tables_path = Path(forensic_tables_path)
    report_path = None if report_path is None else Path(report_path)
    result_log_path = None if result_log_path is None else Path(result_log_path)
    decision_table_path = None if decision_table_path is None else Path(decision_table_path)
    forensic_report = _read_json(forensic_report_path)
    tables = pd.read_csv(forensic_tables_path)
    decision_rows = _decision_rows(tables)
    summary = _summary(decision_rows, forensic_report)
    pass_criteria = _pass_criteria(summary)
    recommendation = _recommendation(pass_criteria)
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "Multi-Scale Context Boundary Diagnostic E5",
        "command": command,
        "code_commit": _git_stdout("rev-parse", "HEAD"),
        "code_dirty": bool(_git_stdout("status", "--short")),
        "forensic_report_path": forensic_report_path.as_posix(),
        "forensic_tables_path": forensic_tables_path.as_posix(),
        "report_path": None if report_path is None else report_path.as_posix(),
        "result_log_path": None if result_log_path is None else result_log_path.as_posix(),
        "decision_table_path": None if decision_table_path is None else decision_table_path.as_posix(),
        "elapsed_s": time.perf_counter() - started_at,
        "config": {
            "boundary_share_threshold": BOUNDARY_SHARE_THRESHOLD,
            "boundary_rate_lift_threshold": BOUNDARY_RATE_LIFT_THRESHOLD,
            "policy": "Run M1/M2 only if full-cache M0 boundary signal is strong; do not build shifted windows when boundary is secondary.",
        },
        "summary": summary,
        "pass_criteria": pass_criteria,
        "decision_rows": decision_rows,
        "recommendation": recommendation,
    }
    if report_path is not None:
        _write_json(report_path, report)
    if decision_table_path is not None:
        decision_table_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(decision_rows).to_csv(decision_table_path, index=False)
    if result_log_path is not None:
        _write_result_log(result_log_path, report)
    return report


def _decision_rows(tables: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for split in ("valid", "test"):
        split_rows = tables[(tables["split"] == split) & (tables["table"] == "boundary_bucket")].copy()
        overall = tables[(tables["split"] == split) & (tables["table"] == "overall") & (tables["bucket"] == "all")]
        total_fallback = int(overall["fallback_group_count"].iloc[0]) if len(overall) else int(split_rows["fallback_group_count"].sum())
        middle_rate = _bucket_rate(split_rows, "middle")
        for bucket in ("exact_start", "near_start_le12", "middle", "near_end_le12"):
            bucket_frame = split_rows[split_rows["bucket"] == bucket]
            if bucket_frame.empty:
                continue
            row = bucket_frame.iloc[0].to_dict()
            fallback = int(row.get("fallback_group_count", 0) or 0)
            rate = float(row.get("fallback_group_rate", 0.0) or 0.0)
            rows.append(
                {
                    "split": split,
                    "bucket": bucket,
                    "group_count": int(row.get("group_count", 0) or 0),
                    "fallback_group_count": fallback,
                    "fallback_group_rate": rate,
                    "fallback_share": float(fallback) / float(total_fallback) if total_fallback else 0.0,
                    "rate_lift_vs_middle": rate - middle_rate,
                    "lane_erased_recovered_fallback_rate": float(row.get("lane_erased_recovered_fallback_rate", 0.0) or 0.0),
                    "skeleton_recovered_fallback_rate": float(row.get("skeleton_recovered_fallback_rate", 0.0) or 0.0),
                    "mapset_count": int(row.get("mapset_count", 0) or 0),
                }
            )
    return rows


def _summary(rows: Sequence[Mapping[str, Any]], forensic_report: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    forensic_summary = forensic_report.get("forensic_summary", {}) if isinstance(forensic_report, Mapping) else {}
    for split in ("valid", "test"):
        split_rows = [row for row in rows if row.get("split") == split]
        boundary_rows = [row for row in split_rows if row.get("bucket") in {"exact_start", "near_start_le12", "near_end_le12"}]
        near_rows = [row for row in split_rows if row.get("bucket") in {"near_start_le12", "near_end_le12"}]
        max_lift = max((float(row.get("rate_lift_vs_middle", 0.0) or 0.0) for row in boundary_rows), default=0.0)
        boundary_share = sum(float(row.get("fallback_share", 0.0) or 0.0) for row in boundary_rows)
        near_share = sum(float(row.get("fallback_share", 0.0) or 0.0) for row in near_rows)
        source_split = forensic_summary.get(split, {}) if isinstance(forensic_summary, Mapping) else {}
        summary[split] = {
            "boundary_near_fallback_share_from_e1": source_split.get("boundary_near_fallback_share"),
            "computed_boundary_fallback_share": boundary_share,
            "computed_near_only_fallback_share": near_share,
            "max_boundary_rate_lift_vs_middle": max_lift,
            "middle_fallback_rate": next((row.get("fallback_group_rate") for row in split_rows if row.get("bucket") == "middle"), None),
            "rows": split_rows,
        }
    return summary


def _pass_criteria(summary: Mapping[str, Any]) -> dict[str, Any]:
    test = summary.get("test", {}) if isinstance(summary, Mapping) else {}
    boundary_share = float(test.get("computed_boundary_fallback_share", 0.0) or 0.0)
    rate_lift = float(test.get("max_boundary_rate_lift_vs_middle", 0.0) or 0.0)
    boundary_signal = boundary_share >= BOUNDARY_SHARE_THRESHOLD and rate_lift >= BOUNDARY_RATE_LIFT_THRESHOLD
    return {
        "m0_boundary_signal_pass": boundary_signal,
        "test_boundary_fallback_share": boundary_share,
        "test_max_boundary_rate_lift_vs_middle": rate_lift,
        "m1_shifted_window_recommended": boundary_signal,
        "m2_context_window_recommended": boundary_signal,
        "research_pass": boundary_signal,
    }


def _recommendation(pass_criteria: Mapping[str, Any]) -> str:
    if pass_criteria.get("research_pass"):
        return "TEST_M1_M2: boundary fallback is strong enough to justify shifted/context windows."
    return "KILL_OR_DEFER_E5: boundary effects are secondary; do not build shifted or multi-scale chunking yet."


def _bucket_rate(frame: pd.DataFrame, bucket: str) -> float:
    row = frame[frame["bucket"] == bucket]
    if row.empty:
        return 0.0
    return float(row["fallback_group_rate"].iloc[0])


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_result_log(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pass_criteria = report.get("pass_criteria", {})
    summary = report.get("summary", {})
    test = summary.get("test", {}) if isinstance(summary, Mapping) else {}
    lines = [
        "# Multi-Scale Context Boundary Diagnostic Result Log",
        "",
        "## Summary",
        "",
        f"- Research pass: {pass_criteria.get('research_pass')}",
        f"- Recommendation: {report.get('recommendation')}",
        f"- Runtime seconds: {_fmt_float(report.get('elapsed_s'))}",
        f"- Test boundary fallback share: {_fmt_float(pass_criteria.get('test_boundary_fallback_share'))}",
        f"- Test max boundary rate lift vs middle: {_fmt_float(pass_criteria.get('test_max_boundary_rate_lift_vs_middle'))}",
        f"- M1 shifted-window recommended: {pass_criteria.get('m1_shifted_window_recommended')}",
        f"- M2 context-window recommended: {pass_criteria.get('m2_context_window_recommended')}",
        f"- E1 boundary-near share: {_fmt_float(test.get('boundary_near_fallback_share_from_e1'))}",
        "",
        "## Interpretation",
        "",
        str(report.get("recommendation")),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_normalize_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _normalize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_normalize_json(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _normalize_json(value.item())
        except Exception:  # noqa: BLE001 - defensive conversion.
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _fmt_float(value: object) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _git_stdout(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001 - git metadata is optional.
        return ""


def _command(args: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in ["python", "-m", "pulsefield_model.osu_core.multiscale_context_audit", *args])


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E5 M0 boundary/multi-scale context decision audit.")
    parser.add_argument("--forensic-report-path", type=Path, default=DEFAULT_FORENSIC_REPORT_PATH)
    parser.add_argument("--forensic-tables-path", type=Path, default=DEFAULT_FORENSIC_TABLES_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--result-log-path", type=Path, default=DEFAULT_RESULT_LOG_PATH)
    parser.add_argument("--decision-table-path", type=Path, default=DEFAULT_DECISION_TABLE_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = audit_multiscale_context(
        forensic_report_path=args.forensic_report_path,
        forensic_tables_path=args.forensic_tables_path,
        report_path=args.report_path,
        result_log_path=args.result_log_path,
        decision_table_path=args.decision_table_path,
        command=_command(sys.argv[1:]),
    )
    print(
        "multiscale_context_audit "
        f"research_pass={report.get('pass_criteria', {}).get('research_pass')} "
        f"recommendation={report.get('recommendation')} "
        f"report={report.get('report_path')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
