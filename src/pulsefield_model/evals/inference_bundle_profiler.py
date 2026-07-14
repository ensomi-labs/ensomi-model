from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import platform
import resource
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import torch
from torch.profiler import ProfilerActivity, record_function

from pulsefield_model.features.mel import stage2_log_mel_cache_path
from pulsefield_model.inference.audio_probe import audio_length_ms_from_file
from pulsefield_model.inference.defaults import (
    DEFAULT_CONTROL_CHECKPOINT_PATH,
    DEFAULT_MAPPER_CHECKPOINT_PATH,
    DEFAULT_MAPPER_MODEL_ID,
)
from pulsefield_model.inference.model_bundles.mapper_v2_1_sparse import MapperV21SparseStreamWithCache
from pulsefield_model.inference.routed_backend import RoutedInferenceBackend
from pulsefield_model.inference.session_runtime import SessionRuntime, SessionRuntimeConfig
from pulsefield_model.inference.stream_with_cache import DecoderWindow, StreamWithCacheConfig


SUMMARY_SCHEMA_VERSION = 1
DEFAULT_AUDIO_PATH = Path("dataset/0/1086533/audio.mp3")
DEFAULT_OUTPUT_DIR = Path("artifacts/evals/inference_bundle_mps_profile")
DEFAULT_TRACE_NAME = "bundle_diagnostic_cpu_trace.json"
DEFAULT_SUMMARY_NAME = "summary.json"

T = TypeVar("T")


@dataclass(frozen=True)
class BundleBenchmarkConfig:
    audio_path: Path = DEFAULT_AUDIO_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    device: str = "mps"
    repeat: int = 3
    warmup: int = 1
    difficulty: float = 4.0
    max_tokens: int = 512
    seed: int | None = 0
    mapper_checkpoint_path: Path = DEFAULT_MAPPER_CHECKPOINT_PATH
    control_checkpoint_path: Path = DEFAULT_CONTROL_CHECKPOINT_PATH
    beatthis_checkpoint: str = "final0"
    beatthis_device: str = "cpu"
    use_profiler: bool = True
    profile_window_index: int = 1
    expected_raw_token_sha256: str | None = None
    expected_protocol_token_sha256: str | None = None

    def __post_init__(self) -> None:
        if int(self.repeat) <= 0:
            raise ValueError("repeat must be positive")
        if int(self.warmup) < 0:
            raise ValueError("warmup must be non-negative")
        if int(self.max_tokens) <= 0:
            raise ValueError("max_tokens must be positive")
        if int(self.profile_window_index) < 0:
            raise ValueError("profile_window_index must be non-negative")
        _validate_optional_sha256(self.expected_raw_token_sha256, "expected_raw_token_sha256")
        _validate_optional_sha256(self.expected_protocol_token_sha256, "expected_protocol_token_sha256")


class StageRecorder:
    """Benchmark-only state shared by the async bundle and worker threads."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.diagnostic_enabled = False
        self.phase = "setup"
        self.run_index = -1
        self.session_id = ""
        self.operation = "idle"
        self.stage_samples: list[dict[str, Any]] = []
        self.window_samples: list[dict[str, Any]] = []
        self.profiler_events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def set_context(self, *, phase: str, run_index: int, session_id: str, diagnostic: bool) -> None:
        self.phase = str(phase)
        self.run_index = int(run_index)
        self.session_id = str(session_id)
        self.diagnostic_enabled = bool(diagnostic)
        self.operation = "idle"

    def set_operation(self, operation: str) -> None:
        self.operation = str(operation)

    def measure_stage(
        self,
        stage: str,
        fn: Callable[[], T],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> T:
        if not self.diagnostic_enabled:
            return fn()
        _synchronize_device(self.device)
        start = time.perf_counter()
        with record_function(f"bundle.{stage}"):
            output = fn()
        _synchronize_device(self.device)
        sample = {
            "phase": self.phase,
            "run_index": self.run_index,
            "session_id": self.session_id,
            "operation": self.operation,
            "stage": stage,
            "wall_ms": (time.perf_counter() - start) * 1_000.0,
            "synchronized": True,
            **dict(metadata or {}),
        }
        with self._lock:
            self.stage_samples.append(sample)
        return output

    def add_window(self, sample: Mapping[str, Any]) -> None:
        with self._lock:
            self.window_samples.append(dict(sample))

    def windows_for(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self.window_samples if row["session_id"] == session_id]

    def set_profiler_events(self, events: Sequence[Mapping[str, Any]]) -> None:
        with self._lock:
            self.profiler_events = [dict(row) for row in events]


class MeasuredSessionRuntime(SessionRuntime):
    def __init__(self, *, recorder: StageRecorder, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._recorder = recorder

    def prepare_control_batch(
        self,
        *,
        start_ms_values: Sequence[int],
        max_batch_size: int | None = None,
    ) -> Any:
        starts = tuple(int(value) for value in start_ms_values)
        return self._recorder.measure_stage(
            "control.prepare_control_batch",
            lambda: super(MeasuredSessionRuntime, self).prepare_control_batch(
                start_ms_values=starts,
                max_batch_size=max_batch_size,
            ),
            metadata={"start_ms_values": starts, "logical_batch_size": len(starts)},
        )

    def prepare_mapper_window(
        self,
        *,
        start_ms: int = 0,
        end_ms: int | None = None,
        include_control_attention_kv_cache: bool = False,
    ) -> Any:
        resolved_end_ms = int(start_ms) + 8_000 if end_ms is None else int(end_ms)
        return self._recorder.measure_stage(
            "mapper.prepare_window",
            lambda: super(MeasuredSessionRuntime, self).prepare_mapper_window(
                start_ms=int(start_ms),
                end_ms=resolved_end_ms,
                include_control_attention_kv_cache=bool(include_control_attention_kv_cache),
            ),
            metadata={"window_start_ms": int(start_ms), "window_end_ms": resolved_end_ms},
        )


class MeasuredMapperV21SparseStream(MapperV21SparseStreamWithCache):
    def __init__(
        self,
        config: StreamWithCacheConfig,
        *,
        recorder: StageRecorder,
        trace_path: Path,
        profile_window_index: int,
    ) -> None:
        self._recorder = recorder
        self._trace_path = trace_path
        self._profile_window_index = int(profile_window_index)
        self._window_index_by_session: dict[str, int] = {}
        self._profile_written = False

        def session_factory(
            session_id: str,
            model_runtime: Any,
            session_config: SessionRuntimeConfig,
        ) -> SessionRuntime:
            return MeasuredSessionRuntime(
                session_id=session_id,
                model_runtime=model_runtime,
                config=session_config,
                recorder=recorder,
            )

        super().__init__(config, session_runtime_factory=session_factory)

    def _generate_window(
        self,
        session_id: str,
        session_runtime: SessionRuntime,
        window: DecoderWindow,
        audio_length_ms: int,
    ) -> Any:
        window_index = self._window_index_by_session.get(session_id, 0)
        self._window_index_by_session[session_id] = window_index + 1
        should_profile = (
            self._recorder.diagnostic_enabled
            and not self._profile_written
            and window_index == self._profile_window_index
        )
        profiler_context = (
            torch.profiler.profile(
                activities=_profiler_activities(self._recorder.device),
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
                with_flops=False,
                acc_events=True,
            )
            if should_profile
            else nullcontext(None)
        )

        _synchronize_device(self._recorder.device)
        start = time.perf_counter()
        with profiler_context as active_profiler:
            with record_function("bundle.mapper.generate_window"):
                generated = super()._generate_window(
                    session_id,
                    session_runtime,
                    window,
                    audio_length_ms,
                )
        _synchronize_device(self._recorder.device)
        wall_ms = (time.perf_counter() - start) * 1_000.0

        tokens = tuple(int(token_id) for token_id in getattr(generated, "tokens", ()))
        terminal_state = getattr(generated, "terminal_state", None)
        self._recorder.add_window(
            {
                "phase": self._recorder.phase,
                "run_index": self._recorder.run_index,
                "session_id": session_id,
                "window_index": window_index,
                "window_start_ms": int(window.start_ms),
                "window_end_ms": int(window.end_ms),
                "wall_ms": wall_ms,
                "token_count": len(tokens),
                "token_ids": tokens,
                "token_sha256": _sha256_json(tokens),
                "completed": bool(getattr(generated, "completed", False)),
                "dead_end": bool(getattr(generated, "dead_end", False)),
                "max_tokens_exceeded": bool(getattr(generated, "max_tokens_exceeded", False)),
                "terminal_ms": int(getattr(terminal_state, "current_ms", -1)),
                "mps_memory": _device_memory_snapshot(self._recorder.device),
                "profiler_active": bool(should_profile),
            },
        )

        if active_profiler is not None:
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)
            active_profiler.export_chrome_trace(str(self._trace_path))
            self._recorder.set_profiler_events(_profiler_event_rows(active_profiler))
            self._profile_written = True
        return generated


async def run_bundle_benchmark(config: BundleBenchmarkConfig) -> dict[str, Any]:
    audio_path = config.audio_path.expanduser().resolve()
    output_dir = config.output_dir.expanduser().resolve()
    mapper_checkpoint = config.mapper_checkpoint_path.expanduser().resolve()
    control_checkpoint = config.control_checkpoint_path.expanduser().resolve()
    _require_file(audio_path, "audio")
    _require_file(mapper_checkpoint, "mapper checkpoint")
    _require_file(control_checkpoint, "control checkpoint")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.device)
    _validate_device(device)
    audio_length_ms = audio_length_ms_from_file(audio_path)
    if audio_length_ms is None:
        raise ValueError(f"could not determine audio duration: {audio_path}")
    mel_cache_path = stage2_log_mel_cache_path(audio_path)
    mel_cache_hit_before_run = mel_cache_path.exists()

    trace_path = output_dir / DEFAULT_TRACE_NAME
    stream_config = StreamWithCacheConfig(
        mapper_checkpoint_path=mapper_checkpoint,
        control_checkpoint_path=control_checkpoint,
        mapper_profile="v2_1_sparse",
        device=str(device),
        beatthis_checkpoint=config.beatthis_checkpoint,
        beatthis_device=config.beatthis_device,
        token_send_interval_s=0.0,
        max_tokens=int(config.max_tokens),
        temperature=0.0,
        top_p=None,
        use_incremental_mapper_decode=True,
        seed=config.seed,
    )
    recorder = StageRecorder(device)
    stream = MeasuredMapperV21SparseStream(
        stream_config,
        recorder=recorder,
        trace_path=trace_path,
        profile_window_index=int(config.profile_window_index),
    )
    backend = RoutedInferenceBackend(stream_config, mapper_backend=stream)

    startup_ms = await _timed_async(backend.startup, device=None)
    memory_before_mount = _device_memory_snapshot(device)
    cold_mount_ms = await _timed_async(
        lambda: backend.mount_model(DEFAULT_MAPPER_MODEL_ID),
        device=device,
    )
    memory_after_mount = _device_memory_snapshot(device)

    warmup_rows: list[dict[str, Any]] = []
    measured_rows: list[dict[str, Any]] = []
    diagnostic_row: dict[str, Any] | None = None
    try:
        for run_index in range(int(config.warmup)):
            warmup_rows.append(
                await _run_session(
                    backend=backend,
                    recorder=recorder,
                    audio_path=audio_path,
                    audio_length_ms=audio_length_ms,
                    difficulty=float(config.difficulty),
                    phase="warmup",
                    run_index=run_index,
                    diagnostic=False,
                ),
            )
        for run_index in range(int(config.repeat)):
            measured_rows.append(
                await _run_session(
                    backend=backend,
                    recorder=recorder,
                    audio_path=audio_path,
                    audio_length_ms=audio_length_ms,
                    difficulty=float(config.difficulty),
                    phase="measured",
                    run_index=run_index,
                    diagnostic=False,
                ),
            )
        if config.use_profiler:
            diagnostic_row = await _run_session(
                backend=backend,
                recorder=recorder,
                audio_path=audio_path,
                audio_length_ms=audio_length_ms,
                difficulty=float(config.difficulty),
                phase="diagnostic",
                run_index=0,
                diagnostic=True,
            )
    finally:
        shutdown_ms = await _timed_async(backend.shutdown, device=device)

    aggregate = aggregate_session_rows(
        measured_rows,
        audio_length_ms=audio_length_ms,
        expected_window_count=_expected_window_count(audio_length_ms, stream_config.decoder_window_ms),
        warmup_count=int(config.warmup),
        expected_raw_token_sha256=config.expected_raw_token_sha256,
        expected_protocol_token_sha256=config.expected_protocol_token_sha256,
    )
    diagnostic = _diagnostic_summary(
        recorder,
        diagnostic_row=diagnostic_row,
        trace_path=trace_path if trace_path.exists() else None,
    )
    status = "ok" if bool(aggregate["guards"]["all_passed"]) else "guard_failed"
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "inference_bundle_mps_profile",
        "status": status,
        "created_unix_s": time.time(),
        "machine": _machine_metadata(device),
        "config": {
            "audio_path": str(audio_path),
            "audio_length_ms": audio_length_ms,
            "difficulty": float(config.difficulty),
            "device": str(device),
            "repeat": int(config.repeat),
            "warmup": int(config.warmup),
            "mapper_profile": "v2_1_sparse",
            "mapper_checkpoint_path": str(mapper_checkpoint),
            "control_checkpoint_path": str(control_checkpoint),
            "beatthis_checkpoint": config.beatthis_checkpoint,
            "beatthis_device": config.beatthis_device,
            "max_tokens": int(config.max_tokens),
            "temperature": 0.0,
            "seed": config.seed,
            "expected_raw_token_sha256": config.expected_raw_token_sha256,
            "expected_protocol_token_sha256": config.expected_protocol_token_sha256,
            "token_send_interval_s": 0.0,
            "use_incremental_mapper_decode": True,
            "mel_cache_path": str(mel_cache_path),
            "mel_cache_hit_before_run": mel_cache_hit_before_run,
            "mel_cache_hit_after_run": mel_cache_path.exists(),
        },
        "lifecycle": {
            "registry_startup_ms": startup_ms,
            "cold_bundle_mount_ms": cold_mount_ms,
            "shutdown_ms": shutdown_ms,
            "memory_before_mount": memory_before_mount,
            "memory_after_mount": memory_after_mount,
        },
        "warmup_sessions": warmup_rows,
        "measured_sessions": measured_rows,
        "aggregate": aggregate,
        "diagnostic": diagnostic,
    }
    write_json_summary(summary, output_dir / DEFAULT_SUMMARY_NAME)
    return summary


async def _run_session(
    *,
    backend: RoutedInferenceBackend,
    recorder: StageRecorder,
    audio_path: Path,
    audio_length_ms: int,
    difficulty: float,
    phase: str,
    run_index: int,
    diagnostic: bool,
) -> dict[str, Any]:
    session_id = f"bundle-profile-{phase}-{run_index}"
    recorder.set_context(
        phase=phase,
        run_index=run_index,
        session_id=session_id,
        diagnostic=diagnostic,
    )
    recorder.set_operation("prepare_audio")
    prepare_audio_ms = await _timed_async(
        lambda: backend.prepare_audio(
            session_id=session_id,
            audio_path=audio_path,
            audio_length_ms=int(audio_length_ms),
            difficulty=float(difficulty),
            route="mapper",
        ),
        device=recorder.device,
    )
    memory_after_prepare = _device_memory_snapshot(recorder.device)

    protocol_tokens: list[tuple[int, int, tuple[str, ...]]] = []
    first_protocol_token_ms: float | None = None
    recorder.set_operation("stream")
    _synchronize_device(recorder.device)
    stream_start = time.perf_counter()
    async for token in backend.iter_hitobject_tokens(
        session_id=session_id,
        audio_path=audio_path,
        audio_length_ms=int(audio_length_ms),
        window=DecoderWindow(start_ms=0, end_ms=8_000),
    ):
        if first_protocol_token_ms is None:
            first_protocol_token_ms = (time.perf_counter() - stream_start) * 1_000.0
        protocol_tokens.append(
            (int(token.token_id), int(token.ms_in_ref_audio), tuple(str(action) for action in token.actions)),
        )
    _synchronize_device(recorder.device)
    stream_ms = (time.perf_counter() - stream_start) * 1_000.0
    memory_after_stream = _device_memory_snapshot(recorder.device)

    windows = recorder.windows_for(session_id)
    raw_token_payload = [
        (int(row["window_start_ms"]), tuple(int(token_id) for token_id in row["token_ids"])) for row in windows
    ]
    raw_token_count = sum(int(row["token_count"]) for row in windows)
    request_compute_ms = prepare_audio_ms + stream_ms
    recorder.set_operation("reset")
    reset_ms = await _timed_async(lambda: backend.reset_session(session_id), device=recorder.device)
    return {
        "phase": phase,
        "run_index": int(run_index),
        "session_id": session_id,
        "prepare_audio_ms": prepare_audio_ms,
        "stream_ms": stream_ms,
        "request_compute_ms": request_compute_ms,
        "request_compute_rtf": request_compute_ms / float(audio_length_ms),
        "audio_seconds_per_compute_second": float(audio_length_ms) / request_compute_ms,
        "first_protocol_token_ms": first_protocol_token_ms,
        "reset_ms": reset_ms,
        "window_count": len(windows),
        "windows": [_public_window_row(row) for row in windows],
        "raw_token_count": raw_token_count,
        "raw_tokens_per_stream_second": raw_token_count / (stream_ms / 1_000.0),
        "raw_token_sha256": _sha256_json(raw_token_payload),
        "protocol_token_count": len(protocol_tokens),
        "protocol_tokens_per_stream_second": len(protocol_tokens) / (stream_ms / 1_000.0),
        "protocol_token_sha256": _sha256_json(protocol_tokens),
        "memory_after_prepare": memory_after_prepare,
        "memory_after_stream": memory_after_stream,
    }


def aggregate_session_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    audio_length_ms: int,
    expected_window_count: int,
    warmup_count: int = 1,
    expected_raw_token_sha256: str | None = None,
    expected_protocol_token_sha256: str | None = None,
) -> dict[str, Any]:
    windows = [window for row in rows for window in row.get("windows", ())]
    raw_hashes = {str(row["raw_token_sha256"]) for row in rows}
    protocol_hashes = {str(row["protocol_token_sha256"]) for row in rows}
    row_guards = [
        {
            "session_id": str(row["session_id"]),
            "window_count_matches": int(row["window_count"]) == int(expected_window_count),
            "all_windows_completed": all(bool(window["completed"]) for window in row.get("windows", ())),
            "no_dead_end": not any(bool(window["dead_end"]) for window in row.get("windows", ())),
            "no_max_tokens_exceeded": not any(
                bool(window["max_tokens_exceeded"]) for window in row.get("windows", ())
            ),
            "terminal_reaches_audio_end": bool(row.get("windows"))
            and int(row["windows"][-1]["terminal_ms"]) >= int(audio_length_ms),
        }
        for row in rows
    ]
    outputs_stable = len(raw_hashes) == 1 and len(protocol_hashes) == 1
    raw_matches_expected = expected_raw_token_sha256 is None or raw_hashes == {expected_raw_token_sha256}
    protocol_matches_expected = (
        expected_protocol_token_sha256 is None or protocol_hashes == {expected_protocol_token_sha256}
    )
    all_rows_pass = bool(row_guards) and all(all(value for key, value in guard.items() if key != "session_id") for guard in row_guards)
    request_rtf = percentile_summary([float(row["request_compute_rtf"]) for row in rows])
    warm_metrics_eligible = int(warmup_count) > 0
    return {
        "session_count": len(rows),
        "warmup_count": int(warmup_count),
        "warm_metrics_eligible": warm_metrics_eligible,
        "expected_window_count_per_session": int(expected_window_count),
        "prepare_audio_ms": percentile_summary([float(row["prepare_audio_ms"]) for row in rows]),
        "stream_ms": percentile_summary([float(row["stream_ms"]) for row in rows]),
        "request_compute_ms": percentile_summary([float(row["request_compute_ms"]) for row in rows]),
        "request_compute_rtf": request_rtf,
        "warm_request_compute_rtf": request_rtf if warm_metrics_eligible else None,
        "audio_seconds_per_compute_second": percentile_summary(
            [float(row["audio_seconds_per_compute_second"]) for row in rows],
        ),
        "first_protocol_token_ms": percentile_summary(
            [float(row["first_protocol_token_ms"]) for row in rows if row["first_protocol_token_ms"] is not None],
        ),
        "window_latency_ms": percentile_summary([float(window["wall_ms"]) for window in windows]),
        "raw_tokens_per_stream_second": percentile_summary(
            [float(row["raw_tokens_per_stream_second"]) for row in rows],
        ),
        "protocol_tokens_per_stream_second": percentile_summary(
            [float(row["protocol_tokens_per_stream_second"]) for row in rows],
        ),
        "guards": {
            "rows": row_guards,
            "raw_output_hashes": sorted(raw_hashes),
            "protocol_output_hashes": sorted(protocol_hashes),
            "outputs_stable": outputs_stable,
            "expected_raw_token_sha256": expected_raw_token_sha256,
            "expected_protocol_token_sha256": expected_protocol_token_sha256,
            "raw_matches_expected": raw_matches_expected,
            "protocol_matches_expected": protocol_matches_expected,
            "all_passed": (
                all_rows_pass
                and outputs_stable
                and raw_matches_expected
                and protocol_matches_expected
            ),
        },
    }


def percentile_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "min": ordered[0],
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
    }


def write_json_summary(summary: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    if output_path.suffix != ".json":
        raise ValueError("bundle benchmark summary path must use a .json suffix")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def _diagnostic_summary(
    recorder: StageRecorder,
    *,
    diagnostic_row: Mapping[str, Any] | None,
    trace_path: Path | None,
) -> dict[str, Any]:
    rows = [dict(row) for row in recorder.stage_samples if row["phase"] == "diagnostic"]
    by_stage: dict[str, list[float]] = {}
    for row in rows:
        by_stage.setdefault(str(row["stage"]), []).append(float(row["wall_ms"]))

    control_rows = [row for row in rows if row["stage"] == "control.prepare_control_batch"]
    online_control_rows = [row for row in control_rows if row["operation"] == "stream"]

    def control_throughput(control_row: Mapping[str, Any], *, audio_seconds: bool) -> float:
        logical_batch_size = int(control_row["logical_batch_size"])
        numerator = logical_batch_size * (8.0 if audio_seconds else 1.0)
        return numerator / (float(control_row["wall_ms"]) / 1_000.0)

    stream_control_by_start = {
        int(row["start_ms_values"][0]): float(row["wall_ms"])
        for row in rows
        if row["operation"] == "stream"
        and row["stage"] == "control.prepare_control_batch"
        and len(row.get("start_ms_values", ())) == 1
    }
    setup_by_start = {
        int(row["window_start_ms"]): float(row["wall_ms"])
        for row in rows
        if row["operation"] == "stream" and row["stage"] == "mapper.prepare_window"
    }
    derived_windows = []
    for window in (diagnostic_row or {}).get("windows", ()):
        start_ms = int(window["window_start_ms"])
        setup_ms = setup_by_start.get(start_ms)
        control_ms = stream_control_by_start.get(start_ms, 0.0)
        total_ms = float(window["wall_ms"])
        profiler_active = bool(window.get("profiler_active", False))
        derived_windows.append(
            {
                "window_start_ms": start_ms,
                "window_total_ms": total_ms,
                "control_ms": control_ms,
                "profiler_active": profiler_active,
                "latency_breakdown_eligible": not profiler_active,
                "excluded_reason": "torch_profiler_overhead" if profiler_active else None,
                "mapper_context_setup_ms": (
                    None if profiler_active or setup_ms is None else max(0.0, setup_ms - control_ms)
                ),
                "mapper_decode_ms": (
                    None if profiler_active or setup_ms is None else max(0.0, total_ms - setup_ms)
                ),
            },
        )
    return {
        "enabled": diagnostic_row is not None,
        "headline_metrics_eligible": False,
        "caveat": "MPS exposes CPU-only torch.profiler activity; synchronized wall time is the device timing source.",
        "session": diagnostic_row,
        "synchronized_stage_samples": rows,
        "stage_summaries_ms": {stage: percentile_summary(values) for stage, values in sorted(by_stage.items())},
        "control_logical_windows_per_second": percentile_summary(
            [control_throughput(row, audio_seconds=False) for row in control_rows],
        ),
        "control_audio_seconds_per_compute_second": percentile_summary(
            [control_throughput(row, audio_seconds=True) for row in control_rows],
        ),
        "online_control_audio_seconds_per_compute_second": percentile_summary(
            [control_throughput(row, audio_seconds=True) for row in online_control_rows],
        ),
        "derived_window_breakdown": derived_windows,
        "trace_path": None if trace_path is None else str(trace_path),
        "profiler_events": recorder.profiler_events,
    }


def _profiler_event_rows(active_profiler: Any, *, limit: int = 40) -> list[dict[str, Any]]:
    events = list(active_profiler.key_averages())
    events.sort(key=lambda event: float(getattr(event, "self_cpu_time_total", 0.0)), reverse=True)
    top = events[: int(limit)]
    bundle_events = [event for event in events if str(getattr(event, "key", "")).startswith("bundle.")]
    selected = {str(getattr(event, "key", "")): event for event in (*bundle_events, *top)}
    return [
        {
            "key": key,
            "count": int(getattr(event, "count", 0)),
            "cpu_time_total_us": float(getattr(event, "cpu_time_total", 0.0)),
            "self_cpu_time_total_us": float(getattr(event, "self_cpu_time_total", 0.0)),
        }
        for key, event in selected.items()
    ]


def _profiler_activities(device: torch.device) -> list[ProfilerActivity]:
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda" and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    return activities


async def _timed_async(fn: Callable[[], Any], *, device: torch.device | None) -> float:
    if device is not None:
        _synchronize_device(device)
    start = time.perf_counter()
    await fn()
    if device is not None:
        _synchronize_device(device)
    return (time.perf_counter() - start) * 1_000.0


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _device_memory_snapshot(device: torch.device) -> dict[str, int] | None:
    if device.type == "mps" and hasattr(torch, "mps"):
        return {
            "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
            "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
            "recommended_max_bytes": int(torch.mps.recommended_max_memory()),
        }
    if device.type == "cuda" and torch.cuda.is_available():
        return {
            "current_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "driver_allocated_bytes": int(torch.cuda.memory_reserved(device)),
            "recommended_max_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        }
    return None


def _machine_metadata(device: torch.device) -> dict[str, Any]:
    max_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    max_rss_bytes = max_rss if sys.platform == "darwin" else max_rss * 1_024
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "profiler_supported_activities": sorted(str(value) for value in torch.profiler.supported_activities()),
        "process_max_rss_bytes": max_rss_bytes,
    }


def _public_window_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "token_ids"}


def _expected_window_count(audio_length_ms: int, decoder_window_ms: int) -> int:
    return max(1, int(math.ceil(int(audio_length_ms) / int(decoder_window_ms))))


def _percentile(ordered_values: Sequence[float], q: float) -> float:
    if not ordered_values:
        raise ValueError("ordered_values must not be empty")
    position = (len(ordered_values) - 1) * float(q)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered_values[lower])
    weight = position - lower
    return float(ordered_values[lower] * (1.0 - weight) + ordered_values[upper] * weight)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _validate_device(device: torch.device) -> None:
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is false")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")


def _validate_optional_sha256(value: str | None, name: str) -> None:
    if value is None:
        return
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256 digest")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile the production control + mapper inference bundle.")
    parser.add_argument("--audio-path", type=Path, default=DEFAULT_AUDIO_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--difficulty", type=float, default=4.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mapper-checkpoint", type=Path, default=DEFAULT_MAPPER_CHECKPOINT_PATH)
    parser.add_argument("--control-checkpoint", type=Path, default=DEFAULT_CONTROL_CHECKPOINT_PATH)
    parser.add_argument("--beatthis-checkpoint", default="final0")
    parser.add_argument("--beatthis-device", default="cpu")
    parser.add_argument("--profile-window-index", type=int, default=1)
    parser.add_argument("--expected-raw-token-sha256")
    parser.add_argument("--expected-protocol-token-sha256")
    parser.add_argument("--no-profiler", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = BundleBenchmarkConfig(
        audio_path=args.audio_path,
        output_dir=args.output_dir,
        device=args.device,
        repeat=args.repeat,
        warmup=args.warmup,
        difficulty=args.difficulty,
        max_tokens=args.max_tokens,
        seed=args.seed,
        mapper_checkpoint_path=args.mapper_checkpoint,
        control_checkpoint_path=args.control_checkpoint,
        beatthis_checkpoint=args.beatthis_checkpoint,
        beatthis_device=args.beatthis_device,
        use_profiler=not args.no_profiler,
        profile_window_index=args.profile_window_index,
        expected_raw_token_sha256=args.expected_raw_token_sha256,
        expected_protocol_token_sha256=args.expected_protocol_token_sha256,
    )
    summary = asyncio.run(run_bundle_benchmark(config))
    headline = {
        "status": summary["status"],
        "summary_path": str(config.output_dir.expanduser().resolve() / DEFAULT_SUMMARY_NAME),
        "cold_bundle_mount_ms": summary["lifecycle"]["cold_bundle_mount_ms"],
        "warm_request_compute_rtf": summary["aggregate"]["warm_request_compute_rtf"],
        "request_compute_rtf": summary["aggregate"]["request_compute_rtf"],
        "audio_seconds_per_compute_second": summary["aggregate"]["audio_seconds_per_compute_second"],
        "window_latency_ms": summary["aggregate"]["window_latency_ms"],
    }
    print(json.dumps(headline, indent=2, sort_keys=True))
    return 0 if summary["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
