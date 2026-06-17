from __future__ import annotations

import pytest

from pulsefield_model.evals.mapper_v3_delta_event_factor_target_full4k_training import (
    dataset_metrics,
    decision_from_checks,
    guard_checks,
    run_delta_event_factor_target_full4k_training,
)
from pulsefield_model.evals.mapper_v3_delta_event_factor_target_longer_training import _stability_metrics


def test_guard_checks_route_to_longer_or_production_when_full4k_gate_passes(tmp_path) -> None:
    report = _report(tmp_path, final_factor=18.0)
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"fake")
    (tmp_path / "report.json").write_text("{}", encoding="utf-8")
    mapper_cache = tmp_path / "records.parquet"
    mapper_cache.write_bytes(b"fake")
    mapper_cache.with_suffix(".json").write_text("{}", encoding="utf-8")
    teacher_cache = tmp_path / "teacher"
    teacher_cache.mkdir()
    for index in range(4):
        (teacher_cache / f"{index}.pt").write_bytes(b"fake")

    dataset = dataset_metrics(
        report,
        mapper_record_cache_path=mapper_cache,
        control_teacher_cache_dir=teacher_cache,
        context=_passing_context(candidate_windows=4),
    )
    stability = _stability_metrics(report, checkpoint_path=checkpoint_path, max_factor_loss_ratio=1.20)
    checks = guard_checks(
        dataset=dataset,
        stability=stability,
        loss_total_check={"matches": True},
        context=_passing_context(candidate_windows=4),
    )
    decision = decision_from_checks(checks)

    assert all(checks.values())
    assert decision["route"] == "TEST_DELTA_EVENT_FACTOR_TARGET_FULL4K_LONGER_OR_PRODUCTION_CONFIG_CARD"


def test_decision_mutates_cache_or_dataset_when_counts_do_not_match(tmp_path) -> None:
    checks = _passing_checks(tmp_path)
    checks["source_matches_full4k_bit_proxy"] = False

    decision = decision_from_checks(checks)

    assert decision["route"] == "MUTATE_FACTOR_TARGET_FULL4K_CACHE_OR_DATASET"
    assert "source_matches_full4k_bit_proxy" in decision["reason"]


def test_decision_mutates_lr_when_factor_loss_ratio_exceeds_limit(tmp_path) -> None:
    checks = _passing_checks(tmp_path)
    checks["factor_loss_ratio_within_limit"] = False

    decision = decision_from_checks(checks)

    assert decision["route"] == "MUTATE_FACTOR_TARGET_FULL4K_LR_OR_LOSS_WEIGHT"
    assert "factor_loss_ratio_within_limit" in decision["reason"]


def test_decision_mutates_plumbing_when_loss_accounting_fails(tmp_path) -> None:
    checks = _passing_checks(tmp_path)
    checks["loss_total_recomputed_matches"] = False

    decision = decision_from_checks(checks)

    assert decision["route"] == "MUTATE_FACTOR_TARGET_FULL4K_PLUMBING"
    assert "loss_total_recomputed_matches" in decision["reason"]


def test_decision_mutates_context_chain_when_prior_route_is_missing(tmp_path) -> None:
    checks = _passing_checks(tmp_path)
    checks["full4k_bit_proxy_route_positive"] = False

    decision = decision_from_checks(checks)

    assert decision["route"] == "MUTATE_FACTOR_TARGET_FULL4K_CONTEXT_CHAIN"
    assert "full4k_bit_proxy_route_positive" in decision["reason"]


def test_gate_rejects_nonpositive_max_steps(tmp_path) -> None:
    with pytest.raises(ValueError, match="max_steps must be positive"):
        run_delta_event_factor_target_full4k_training(
            dataset_root=tmp_path / "dataset",
            index_path=tmp_path / "missing.parquet",
            control_teacher_cache_dir=tmp_path / "cache",
            mapper_record_cache_path=tmp_path / "records.parquet",
            output_dir=tmp_path / "run",
            summary_output_path=tmp_path / "summary.json",
            report_output_path=tmp_path / "report.md",
            max_steps=0,
        )


def _passing_checks(tmp_path) -> dict[str, bool]:
    report = _report(tmp_path, final_factor=18.0)
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"fake")
    (tmp_path / "report.json").write_text("{}", encoding="utf-8")
    mapper_cache = tmp_path / "records.parquet"
    mapper_cache.write_bytes(b"fake")
    mapper_cache.with_suffix(".json").write_text("{}", encoding="utf-8")
    teacher_cache = tmp_path / "teacher"
    teacher_cache.mkdir(exist_ok=True)
    for index in range(4):
        (teacher_cache / f"{index}.pt").write_bytes(b"fake")
    context = _passing_context(candidate_windows=4)
    dataset = dataset_metrics(
        report,
        mapper_record_cache_path=mapper_cache,
        control_teacher_cache_dir=teacher_cache,
        context=context,
    )
    stability = _stability_metrics(report, checkpoint_path=checkpoint_path, max_factor_loss_ratio=1.20)
    return {key: True for key in guard_checks(
        dataset=dataset,
        stability=stability,
        loss_total_check={"matches": True},
        context=context,
    )}


def _passing_context(*, candidate_windows: int) -> dict:
    return {
        "longer_training": {
            "route": "TEST_DELTA_EVENT_FACTOR_TARGET_FULL4K_TRAINING_CARD",
        },
        "full4k_bit_proxy": {
            "route": "TEST_DELTA_EVENT_FACTOR_TARGET_MODEL_CARD",
            "metrics": {
                "audited_window_count": 174515,
                "candidate_window_count": int(candidate_windows),
            },
        },
        "full4k_label_coverage": {
            "route": "TEST_DELTA_EVENT_FULL4K_BIT_PROXY_OR_FACTOR_TARGET_CARD",
        },
    }


def _report(tmp_path, *, final_factor: float) -> dict:
    return {
        "completed_steps": 8,
        "max_steps": 8,
        "is_complete": True,
        "dataset": {
            "source_window_count": 174515,
            "train_window_count": 174499,
            "eval_window_count": 16,
            "final_train_eval_window_count": 16,
            "filter_report": {
                "num_total_windows": 4,
                "num_mapper_eligible_windows": 174515,
                "num_dropped_unsupported_action_windows": 6181,
                "drop_rate": 0.0342,
            },
            "mapper_record_cache_path": (tmp_path / "records.parquet").as_posix(),
            "control_teacher_cache_dir": (tmp_path / "teacher").as_posix(),
            "require_control_teacher_cache": True,
            "include_delta_event_factor_target": True,
            "include_full_song_context": False,
            "mapper_token_contract": "v3_event_groups",
        },
        "model_config": {"use_delta_event_factor_target": True},
        "loss_config": {"lambda_delta_event_factor_target": 0.25},
        "history": [
            {"step": 1, "eval": _metrics(20.0)},
            {"step": 4, "eval": _metrics(19.0)},
            {"step": 8, "eval": _metrics(final_factor)},
        ],
        "final_train_metrics": _metrics(17.5),
        "final_eval_metrics": _metrics(final_factor),
    }


def _metrics(factor_loss: float) -> dict:
    return {
        "loss/total": 1.0 + 0.25 * float(factor_loss),
        "loss/token": 1.0,
        "loss/delta_event_factor_target": float(factor_loss),
        "phase/lambda_delta_event_factor_target": 0.25,
        "delta_event_factor/kind_label_count": 10.0,
        "delta_event_factor/delta_label_count": 9.0,
        "delta_event_factor/signature_label_count": 9.0,
        "delta_event_factor/end_gap_label_count": 1.0,
    }
