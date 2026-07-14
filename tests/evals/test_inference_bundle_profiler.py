from __future__ import annotations

import json
from pathlib import Path

import pytest

from pulsefield_model.evals.inference_bundle_profiler import (
    BundleBenchmarkConfig,
    StageRecorder,
    _diagnostic_summary,
    aggregate_session_rows,
    percentile_summary,
    write_json_summary,
)


def _window(*, start_ms: int, terminal_ms: int) -> dict[str, object]:
    return {
        "window_start_ms": start_ms,
        "window_end_ms": start_ms + 8_000,
        "wall_ms": 100.0 + start_ms / 1_000.0,
        "completed": True,
        "dead_end": False,
        "max_tokens_exceeded": False,
        "terminal_ms": terminal_ms,
    }


def _session(index: int, raw_hash: str = "raw", protocol_hash: str = "protocol") -> dict[str, object]:
    return {
        "session_id": f"s{index}",
        "prepare_audio_ms": 1_000.0 + index,
        "stream_ms": 500.0 + index,
        "request_compute_ms": 1_500.0 + 2 * index,
        "request_compute_rtf": (1_500.0 + 2 * index) / 16_000.0,
        "audio_seconds_per_compute_second": 16_000.0 / (1_500.0 + 2 * index),
        "first_protocol_token_ms": 120.0 + index,
        "raw_tokens_per_stream_second": 30.0,
        "protocol_tokens_per_stream_second": 10.0,
        "raw_token_sha256": raw_hash,
        "protocol_token_sha256": protocol_hash,
        "window_count": 2,
        "windows": [
            _window(start_ms=0, terminal_ms=8_000),
            _window(start_ms=8_000, terminal_ms=16_000),
        ],
    }


def test_percentile_summary_interpolates_tail_percentiles() -> None:
    summary = percentile_summary([1.0, 2.0, 3.0, 4.0])

    assert summary["count"] == 4
    assert summary["mean"] == pytest.approx(2.5)
    assert summary["p50"] == pytest.approx(2.5)
    assert summary["p95"] == pytest.approx(3.85)
    assert summary["p99"] == pytest.approx(3.97)


def test_aggregate_session_rows_requires_complete_stable_outputs() -> None:
    summary = aggregate_session_rows(
        [_session(0), _session(1)],
        audio_length_ms=16_000,
        expected_window_count=2,
    )

    assert summary["session_count"] == 2
    assert summary["window_latency_ms"]["count"] == 4
    assert summary["guards"]["outputs_stable"] is True
    assert summary["guards"]["all_passed"] is True

    changed = aggregate_session_rows(
        [_session(0), _session(1, raw_hash="changed")],
        audio_length_ms=16_000,
        expected_window_count=2,
    )
    assert changed["guards"]["outputs_stable"] is False
    assert changed["guards"]["all_passed"] is False

    wrong_baseline = aggregate_session_rows(
        [_session(0), _session(1)],
        audio_length_ms=16_000,
        expected_window_count=2,
        expected_raw_token_sha256="different",
        expected_protocol_token_sha256="protocol",
    )
    assert wrong_baseline["guards"]["outputs_stable"] is True
    assert wrong_baseline["guards"]["raw_matches_expected"] is False
    assert wrong_baseline["guards"]["all_passed"] is False


def test_unwarmed_aggregate_is_not_labeled_warm() -> None:
    summary = aggregate_session_rows(
        [_session(0)],
        audio_length_ms=16_000,
        expected_window_count=2,
        warmup_count=0,
    )

    assert summary["request_compute_rtf"]["count"] == 1
    assert summary["warm_metrics_eligible"] is False
    assert summary["warm_request_compute_rtf"] is None


def test_diagnostic_excludes_profiled_window_from_latency_breakdown() -> None:
    recorder = StageRecorder.__new__(StageRecorder)
    recorder.stage_samples = [
        {
            "phase": "diagnostic",
            "operation": "stream",
            "stage": "control.prepare_control_batch",
            "wall_ms": 100.0,
            "logical_batch_size": 1,
            "start_ms_values": [8_000],
        },
        {
            "phase": "diagnostic",
            "operation": "stream",
            "stage": "mapper.prepare_window",
            "wall_ms": 110.0,
            "window_start_ms": 8_000,
        },
    ]
    recorder.profiler_events = []
    summary = _diagnostic_summary(
        recorder,
        diagnostic_row={
            "windows": [
                {
                    "window_start_ms": 8_000,
                    "wall_ms": 9_999.0,
                    "profiler_active": True,
                },
            ],
        },
        trace_path=None,
    )

    row = summary["derived_window_breakdown"][0]
    assert row["latency_breakdown_eligible"] is False
    assert row["mapper_decode_ms"] is None
    assert summary["online_control_audio_seconds_per_compute_second"]["p50"] == pytest.approx(80.0)


def test_write_json_summary_rejects_jsonl_and_writes_json(tmp_path: Path) -> None:
    path = write_json_summary({"path": tmp_path, "rows": (1, 2)}, tmp_path / "summary.json")

    assert json.loads(path.read_text(encoding="utf-8"))["rows"] == [1, 2]
    with pytest.raises(ValueError, match=".json suffix"):
        write_json_summary({}, tmp_path / "summary.jsonl")


def test_bundle_benchmark_config_rejects_invalid_run_counts() -> None:
    with pytest.raises(ValueError, match="repeat"):
        BundleBenchmarkConfig(repeat=0)
    with pytest.raises(ValueError, match="warmup"):
        BundleBenchmarkConfig(warmup=-1)
