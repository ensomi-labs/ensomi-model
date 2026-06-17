from __future__ import annotations

import pytest

from pulsefield_model.evals.mapper_v3_delta_event_factor_target_data_contract import (
    decision_from_checks,
    guard_checks,
    run_delta_event_factor_target_data_contract,
)


def test_guard_checks_route_to_model_loss_plumbing_when_contract_passes() -> None:
    checks = guard_checks(
        default_off_batch={},
        metrics=_passing_metrics(),
        context=_passing_context(),
    )
    decision = decision_from_checks(checks)

    assert all(checks.values())
    assert decision["route"] == "TEST_DELTA_EVENT_FACTOR_TARGET_MODEL_LOSS_PLUMBING_CARD"


def test_decision_kills_when_default_off_regresses() -> None:
    checks = {key: True for key in guard_checks(
        default_off_batch={},
        metrics=_passing_metrics(),
        context=_passing_context(),
    )}
    checks["default_off_factor_target_absent"] = False

    decision = decision_from_checks(checks)

    assert decision["route"] == "KILL_FACTOR_TARGET_DATA_CONTRACT_DEFAULT_REGRESSION"
    assert "default-off" in decision["reason"]


def test_decision_mutates_when_counts_do_not_match_tiny_model() -> None:
    checks = {key: True for key in guard_checks(
        default_off_batch={},
        metrics=_passing_metrics(),
        context=_passing_context(),
    )}
    checks["row_count_matches_tiny_model"] = False

    decision = decision_from_checks(checks)

    assert decision["route"] == "MUTATE_FACTOR_TARGET_DATA_CONTRACT"
    assert "row_count_matches_tiny_model" in decision["reason"]


def test_gate_rejects_nonpositive_window_limit(tmp_path) -> None:
    with pytest.raises(ValueError, match="window_limit must be positive"):
        run_delta_event_factor_target_data_contract(
            index_path=tmp_path / "missing.parquet",
            control_teacher_cache_dir=tmp_path / "cache",
            summary_output_path=tmp_path / "summary.json",
            report_output_path=tmp_path / "report.md",
            window_limit=0,
        )


def _passing_metrics() -> dict:
    return {
        "audited_window_count": 32,
        "row_count": 1479,
        "event_row_count": 1447,
        "end_row_count": 32,
        "max_row_len": 100,
        "row_mask_valid_count": 1479,
        "kind_label_count": 1479,
        "delta_label_count": 1447,
        "signature_label_count": 1447,
        "end_gap_label_count": 32,
        "reconstruction_mismatch_count": 0,
    }


def _passing_context() -> dict:
    return {
        "factor_target_tiny_model": {
            "route": "TEST_DELTA_EVENT_FACTOR_TARGET_PRODUCTION_PLUMBING_CARD",
            "row_count": 1479,
            "event_row_count": 1447,
            "end_row_count": 32,
        }
    }
