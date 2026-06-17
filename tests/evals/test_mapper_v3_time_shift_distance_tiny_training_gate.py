from __future__ import annotations

import json
from pathlib import Path

from pulsefield_model.evals.mapper_v3_time_shift_distance_tiny_training_gate import (
    EXPECTED_V3_CONTRACT,
    V3TrainingRun,
    aggregate_rollout_results,
    compare_time_shift_distance_gate,
    load_v3_training_run,
    report_markdown,
    write_report,
    write_summary_json,
)


def test_training_only_pass_routes_to_rollout_gate(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    enabled_path = tmp_path / "enabled.json"
    _write_training_report(baseline_path, lambda_time_shift=0.0, time_shift_loss=0.0, token_loss=2.0)
    _write_training_report(enabled_path, lambda_time_shift=0.5, time_shift_loss=0.04, token_loss=2.05)

    summary = compare_time_shift_distance_gate(
        baseline=load_v3_training_run(baseline_path, label="baseline"),
        enabled=load_v3_training_run(enabled_path, label="enabled"),
        min_completed_steps=20,
    )

    assert summary["decision"]["route"] == "TEST_ROLLOUT_GATE"
    assert summary["training_checks"]["enabled_time_shift_distance_loss_positive"] is True
    assert summary["rollout_checks"] is None


def test_training_failure_mutates_when_enabled_loss_missing_signal(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    enabled_path = tmp_path / "enabled.json"
    _write_training_report(baseline_path, lambda_time_shift=0.0, time_shift_loss=0.0, token_loss=2.0)
    _write_training_report(enabled_path, lambda_time_shift=0.5, time_shift_loss=0.0, token_loss=2.0)

    summary = compare_time_shift_distance_gate(
        baseline=load_v3_training_run(baseline_path, label="baseline"),
        enabled=load_v3_training_run(enabled_path, label="enabled"),
        min_completed_steps=20,
    )

    assert summary["decision"]["route"] == "MUTATE_TIME_SHIFT_DISTANCE_OBJECTIVE"
    assert summary["training_checks"]["enabled_time_shift_distance_loss_positive"] is False
    assert "enabled_time_shift_distance_loss_positive" in summary["decision"]["reason"]


def test_training_failure_mutates_when_enabled_report_is_nan(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    enabled_path = tmp_path / "enabled.json"
    _write_training_report(baseline_path, lambda_time_shift=0.0, time_shift_loss=0.0, token_loss=2.0)
    _write_training_report(enabled_path, lambda_time_shift=0.5, time_shift_loss=float("nan"), token_loss=float("nan"))

    summary = compare_time_shift_distance_gate(
        baseline=load_v3_training_run(baseline_path, label="baseline"),
        enabled=load_v3_training_run(enabled_path, label="enabled"),
        min_completed_steps=20,
    )

    assert summary["decision"]["route"] == "MUTATE_TIME_SHIFT_DISTANCE_OBJECTIVE"
    assert summary["training_checks"]["enabled_total_loss_finite"] is False
    assert summary["training_checks"]["enabled_token_loss_finite"] is False
    assert summary["training_checks"]["enabled_time_shift_distance_loss_finite"] is False


def test_rollout_pairs_can_route_to_full32_gate() -> None:
    baseline = _run(lambda_time_shift=0.0, time_shift_loss=0.0, token_loss=2.0)
    enabled = _run(lambda_time_shift=0.5, time_shift_loss=0.02, token_loss=2.01)
    rollout_results = [
        _rollout_case(case_id="a", baseline_rigid=0.7, enabled_rigid=0.69, baseline_starved=False, enabled_starved=False),
        _rollout_case(case_id="b", baseline_rigid=0.9, enabled_rigid=0.91, baseline_starved=True, enabled_starved=True),
    ]

    summary = compare_time_shift_distance_gate(
        baseline=baseline,
        enabled=enabled,
        rollout_results=rollout_results,
        min_completed_steps=20,
    )

    assert summary["decision"]["route"] == "TEST_NEXT_FULL32_TIME_SHIFT_DISTANCE"
    assert summary["rollout_checks"]["enabled_all_legal"] is True
    assert summary["rollout_checks"]["no_new_starved_cases"] is True


def test_rollout_pairs_mutate_on_new_starvation() -> None:
    baseline = _run(lambda_time_shift=0.0, time_shift_loss=0.0, token_loss=2.0)
    enabled = _run(lambda_time_shift=0.5, time_shift_loss=0.02, token_loss=2.01)
    rollout_results = [
        _rollout_case(case_id="a", baseline_rigid=0.7, enabled_rigid=0.69, baseline_starved=False, enabled_starved=True),
    ]

    summary = compare_time_shift_distance_gate(
        baseline=baseline,
        enabled=enabled,
        rollout_results=rollout_results,
        min_completed_steps=20,
    )

    assert summary["decision"]["route"] == "MUTATE_TIME_SHIFT_DISTANCE_OBJECTIVE"
    assert summary["rollout_checks"]["no_new_starved_cases"] is False


def test_aggregate_rollout_results_tracks_rigid_and_starved_counts() -> None:
    aggregate = aggregate_rollout_results(
        [
            _rollout_case(case_id="a", baseline_rigid=1.0, enabled_rigid=0.8, baseline_starved=False, enabled_starved=False),
            _rollout_case(case_id="b", baseline_rigid=0.9, enabled_rigid=0.96, baseline_starved=False, enabled_starved=True),
        ]
    )

    assert aggregate["pair_count"] == 2
    assert aggregate["baseline"]["rigid_case_count"] == 1
    assert aggregate["enabled"]["rigid_case_count"] == 1
    assert aggregate["enabled_vs_baseline"]["new_starved_count"] == 1


def test_writers_emit_json_and_markdown(tmp_path: Path) -> None:
    summary = compare_time_shift_distance_gate(
        baseline=_run(lambda_time_shift=0.0, time_shift_loss=0.0, token_loss=2.0),
        enabled=_run(lambda_time_shift=0.5, time_shift_loss=0.02, token_loss=2.01),
        min_completed_steps=20,
    )
    summary_path = tmp_path / "summary.json"
    report_path = tmp_path / "report.md"

    write_summary_json(summary, summary_path)
    write_report(summary, report_path)

    assert json.loads(summary_path.read_text(encoding="utf-8"))["decision"]["route"] == "TEST_ROLLOUT_GATE"
    markdown = report_markdown(summary)
    assert "Route: `TEST_ROLLOUT_GATE`" in markdown
    assert "No rollout-pair file was supplied" in report_path.read_text(encoding="utf-8")


def _write_training_report(
    path: Path,
    *,
    lambda_time_shift: float,
    time_shift_loss: float,
    token_loss: float,
) -> None:
    payload = {
        "run_name": path.stem,
        "completed_steps": 20,
        "is_complete": True,
        "loss_config": {
            "lambda_time_shift_distance": float(lambda_time_shift),
            "time_shift_distance_scale_ms": 1000.0,
        },
        "training_config": {
            "mapper_token_contract": EXPECTED_V3_CONTRACT,
            "dataset": {
                "mapper_token_contract": EXPECTED_V3_CONTRACT,
            },
        },
        "final_eval_metrics": {
            "loss/total": float(token_loss) + float(lambda_time_shift) * float(time_shift_loss),
            "loss/token": float(token_loss),
            "loss/time_shift_distance": float(time_shift_loss),
            "phase/lambda_time_shift_distance": float(lambda_time_shift),
            "token/valid_count": 128.0,
        },
        "final_train_metrics": {
            "loss/total": float(token_loss) + 0.1,
        },
        "last_train_metrics": {
            "loss/total": float(token_loss) + 0.2,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run(*, lambda_time_shift: float, time_shift_loss: float, token_loss: float) -> V3TrainingRun:
    return V3TrainingRun(
        label="run",
        path="run.json",
        run_name="run",
        mapper_token_contract=EXPECTED_V3_CONTRACT,
        dataset_mapper_token_contract=EXPECTED_V3_CONTRACT,
        completed_steps=20,
        is_complete=True,
        final_eval_loss_total=float(token_loss) + float(lambda_time_shift) * float(time_shift_loss),
        final_eval_loss_token=float(token_loss),
        final_eval_time_shift_distance_loss=float(time_shift_loss),
        final_eval_lambda_time_shift_distance=float(lambda_time_shift),
        final_eval_valid_tokens=128.0,
        final_train_loss_total=float(token_loss),
        last_train_loss_total=float(token_loss),
        loss_config={"lambda_time_shift_distance": float(lambda_time_shift)},
    )


def _rollout_case(
    *,
    case_id: str,
    baseline_rigid: float,
    enabled_rigid: float,
    baseline_starved: bool,
    enabled_starved: bool,
) -> dict[str, object]:
    baseline_metrics = _metrics(rigid=baseline_rigid, starved=baseline_starved)
    enabled_metrics = _metrics(rigid=enabled_rigid, starved=enabled_starved)
    return {
        "case_id": case_id,
        "baseline_legal": True,
        "enabled_legal": True,
        "baseline_metrics": baseline_metrics,
        "enabled_metrics": enabled_metrics,
        "enabled_vs_baseline": {
            "dominant_spacing_ratio_delta": enabled_rigid - baseline_rigid,
            "f1_delta": 0.01,
            "event_count_ratio_delta": 0.0,
            "second_window_event_share_delta": 0.0,
            "candidate_starved": enabled_starved,
            "baseline_starved": baseline_starved,
            "new_starved": enabled_starved and not baseline_starved,
        },
    }


def _metrics(*, rigid: float, starved: bool) -> dict[str, object]:
    return {
        "dominant_spacing_ratio": float(rigid),
        "event_count_ratio": 1.0,
        "second_window_event_share": 0.0 if starved else 0.5,
        "starved": bool(starved),
        "timing_match_100ms": {"f1": 0.7},
    }
