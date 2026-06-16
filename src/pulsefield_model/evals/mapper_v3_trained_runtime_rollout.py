from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from pulsefield_model.inference.mapper_v3_rollout import (
    generate_full_song_rollout_v3,
    generated_v3_tokens_to_v2_1_tokens,
    rollout_to_timepoints_v3,
    session_window_batch_provider_v3,
)
from pulsefield_model.inference.model_runtime import ModelRuntimeConfig, load_model_runtime
from pulsefield_model.inference.session_runtime import SessionRuntime, SessionRuntimeConfig
from pulsefield_model.models.mapper.v3 import MapperV3Model, MapperV3Vocab
from pulsefield_model.timing.grid_fitting.types import TimingFitDiagnostics, TimingFitResult
from pulsefield_model.timing.schema import FittedTimingGrid, FrameTimingPrediction, TimingSegment


SUMMARY_SCHEMA_VERSION = 1


class SyntheticTimingProvider:
    def __init__(self, *, frame_count: int, source_path: str = "synthetic.wav") -> None:
        self.frame_count = int(frame_count)
        self.source_path = str(source_path)
        self.paths: list[str] = []

    def predict_file(self, audio_path: str | Path) -> FrameTimingPrediction:
        self.paths.append(Path(audio_path).as_posix())
        frame_indexes = np.arange(self.frame_count, dtype=np.float32)
        beat_prob = (np.sin(frame_indexes / 8.0) * 0.25 + 0.5).astype(np.float32)
        downbeat_prob = (np.cos(frame_indexes / 32.0) * 0.25 + 0.5).astype(np.float32)
        return FrameTimingPrediction(
            provider="synthetic",
            checkpoint_path="synthetic",
            source_path=self.source_path,
            beat_prob=beat_prob,
            downbeat_prob=downbeat_prob,
            frame_rate_hz=50.0,
        )


class FixedGridFitter:
    def __init__(self) -> None:
        self.prediction_count = 0
        self.result = TimingFitResult(
            grid=FittedTimingGrid((TimingSegment(offset_ms=0.0, beat_length_ms=500.0, meter=4),)),
            score=0.99,
            diagnostics=TimingFitDiagnostics(
                fit_score=0.99,
                selected_period_frames=25.0,
                selected_offset_frames=0.0,
                selected_bpm=120.0,
                candidate_count=1,
                half_tempo_score=0.0,
                double_tempo_score=0.0,
                raw_selected_bpm=120.0,
                raw_score=0.99,
                tempo_multiplier=1.0,
                segment_alias_switch_count=0,
                tempo_multiplier_distribution={"1.0": 1},
            ),
        )

    def fit(self, prediction: FrameTimingPrediction) -> TimingFitResult:
        del prediction
        self.prediction_count += 1
        return self.result


def run_trained_v3_runtime_rollout_smoke(
    *,
    mapper_checkpoint_path: str | Path,
    control_checkpoint_path: str | Path,
    output_summary_path: str | Path,
    output_report_path: str | Path | None = None,
    device_name: str = "auto",
    chart_end_ms: int = 1_000,
    max_tokens_per_window: int = 256,
    audio_path: str | Path = "synthetic.wav",
    normalized_difficulty: float = 0.5,
    include_control_attention_kv_cache: bool = False,
    temperature: float = 0.0,
    top_p: float | None = None,
    seed: int = 1337,
) -> dict[str, Any]:
    chart_end_ms = int(chart_end_ms)
    if chart_end_ms <= 0 or chart_end_ms % 10 != 0:
        raise ValueError("chart_end_ms must be positive and aligned to 10ms")
    frame_count = max(400, (chart_end_ms + 19) // 20)

    runtime = load_model_runtime(
        ModelRuntimeConfig(
            mapper_checkpoint_path=mapper_checkpoint_path,
            control_checkpoint_path=control_checkpoint_path,
            device=device_name,
            eager_load_beatthis=False,
        )
    )
    if not isinstance(runtime.mapper_model, MapperV3Model):
        raise TypeError(f"expected MapperV3Model runtime, got {type(runtime.mapper_model).__name__}")
    if not isinstance(runtime.vocab, MapperV3Vocab):
        raise TypeError(f"expected MapperV3Vocab runtime, got {type(runtime.vocab).__name__}")

    synthetic_provider = SyntheticTimingProvider(frame_count=frame_count, source_path=Path(audio_path).as_posix())
    runtime = replace(runtime, beatthis_provider=synthetic_provider)
    fitter = FixedGridFitter()
    mel = np.zeros((frame_count, 160), dtype=np.float32)

    def load_mel(path: str | Path) -> np.ndarray:
        del path
        return mel

    session_runtime = SessionRuntime(
        session_id="mapper-v3-trained-runtime-rollout-smoke",
        model_runtime=runtime,
        config=SessionRuntimeConfig(
            device=runtime.device,
            default_normalized_difficulty=float(normalized_difficulty),
            max_control_batch_size=4,
        ),
        mel_loader=load_mel,
        grid_fitter=fitter,
    )
    session_runtime.prepare_audio(audio_path, audio_length_ms=chart_end_ms, start_ms=0)
    session_runtime.prepare_full_control(max_batch_size=4)
    generator = torch.Generator(device=runtime.device)
    generator.manual_seed(int(seed))
    rollout = generate_full_song_rollout_v3(
        model=runtime.mapper_model,
        vocab=runtime.vocab,
        chart_end_ms=chart_end_ms,
        window_batch_provider=session_window_batch_provider_v3(
            session_runtime,
            include_control_attention_kv_cache=bool(include_control_attention_kv_cache),
        ),
        device=runtime.device,
        normalized_difficulty=float(normalized_difficulty),
        max_tokens_per_window=int(max_tokens_per_window),
        temperature=float(temperature),
        top_p=top_p,
        generator=generator,
    )
    timepoints = rollout_to_timepoints_v3(rollout, runtime.vocab)
    expanded_v2_1_tokens = generated_v3_tokens_to_v2_1_tokens(rollout.tokens, source_vocab=runtime.vocab)

    mapper_metadata = dict(runtime.checkpoint_metadata["mapper"])
    checks = {
        "mapper_version_v3": mapper_metadata.get("version") == "v3",
        "runtime_model_v3": isinstance(runtime.mapper_model, MapperV3Model),
        "runtime_vocab_v3": isinstance(runtime.vocab, MapperV3Vocab),
        "filtered_embedded_control_encoder": len(tuple(mapper_metadata.get("filtered_control_encoder_keys", ()))) > 0,
        "rollout_window_count_positive": len(rollout.windows) > 0,
        "rollout_token_count_positive": len(rollout.tokens) > 0,
        "rollout_not_dead_end": not bool(rollout.dead_end),
        "rollout_not_max_tokens_exceeded": not bool(rollout.max_tokens_exceeded),
        "v2_1_expansion_ok": len(expanded_v2_1_tokens) >= 0,
    }
    route = "TEST_NEXT" if all(checks.values()) else "MUTATE"
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar v3 trained runtime rollout",
        "decision": {
            "route": route,
            "reason": "all trained runtime rollout gates passed" if route == "TEST_NEXT" else "one or more trained runtime rollout gates failed",
        },
        "checks": checks,
        "config": {
            "mapper_checkpoint_path": Path(mapper_checkpoint_path).as_posix(),
            "control_checkpoint_path": Path(control_checkpoint_path).as_posix(),
            "device": str(runtime.device),
            "chart_end_ms": chart_end_ms,
            "max_tokens_per_window": int(max_tokens_per_window),
            "normalized_difficulty": float(normalized_difficulty),
            "include_control_attention_kv_cache": bool(include_control_attention_kv_cache),
            "temperature": float(temperature),
            "top_p": top_p,
            "seed": int(seed),
            "synthetic_frame_count": int(frame_count),
        },
        "runtime": {
            "mapper": mapper_metadata,
            "control": dict(runtime.checkpoint_metadata["control"]),
            "timing_provider_calls": list(synthetic_provider.paths),
            "grid_fit_count": int(fitter.prediction_count),
        },
        "rollout": {
            "chart_end_ms": int(rollout.chart_end_ms),
            "window_count": len(rollout.windows),
            "token_count": len(rollout.tokens),
            "timepoint_count": len(timepoints),
            "expanded_v2_1_token_count": len(expanded_v2_1_tokens),
            "completed": bool(rollout.completed),
            "dead_end": bool(rollout.dead_end),
            "max_tokens_exceeded": bool(rollout.max_tokens_exceeded),
            "windows": [
                {
                    "write_start_ms": int(window.write_start_ms),
                    "write_end_ms": int(window.write_end_ms),
                    "token_count": len(window.tokens),
                    "completed": bool(window.completed),
                    "dead_end": bool(window.dead_end),
                    "max_tokens_exceeded": bool(window.max_tokens_exceeded),
                    "terminal_ms": int(window.terminal_state.current_ms),
                }
                for window in rollout.windows
            ],
        },
        "interpretation": (
            "The trained v3 checkpoint loaded through ModelRuntime and executed the runtime-backed v3 online rollout path."
            if route == "TEST_NEXT"
            else "The v3 runtime boundary still needs repair before replacement work."
        ),
        "next_step": (
            "Run a longer v3 checkpoint and fuller session-runtime comparison."
            if route == "TEST_NEXT"
            else "Repair loader/provider/generation failure before longer v3 inference work."
        ),
    }
    write_summary_json(summary, output_summary_path)
    if output_report_path is not None:
        write_report(summary, output_report_path)
    return summary


def write_summary_json(summary: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(summary: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_report_markdown(summary), encoding="utf-8")


def _report_markdown(summary: Mapping[str, Any]) -> str:
    decision = _mapping(summary.get("decision"))
    config = _mapping(summary.get("config"))
    runtime = _mapping(summary.get("runtime"))
    mapper = _mapping(runtime.get("mapper"))
    rollout = _mapping(summary.get("rollout"))
    lines = [
        "# Target Grammar v3 Trained Runtime Rollout Result Report",
        "",
        "## Scope",
        "",
        "This smoke verifies that a trained v3 checkpoint can cross the inference-runtime boundary and execute the v3 online rollout path. It is not a generated chart-quality result.",
        "",
        "## Result",
        "",
        f"Decision: `{decision.get('route')}`.",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Mapper checkpoint: `{config.get('mapper_checkpoint_path')}`",
        f"- Control checkpoint: `{config.get('control_checkpoint_path')}`",
        f"- Device: `{config.get('device')}`",
        f"- Mapper runtime version: `{mapper.get('version')}`",
        f"- Filtered embedded control keys: `{len(tuple(mapper.get('filtered_control_encoder_keys', ())))}`",
        f"- Chart end: `{config.get('chart_end_ms')}` ms",
        f"- Windows: `{rollout.get('window_count')}`",
        f"- Generated v3 tokens: `{rollout.get('token_count')}`",
        f"- Generated event timepoints: `{rollout.get('timepoint_count')}`",
        f"- Expanded v2.1 tokens: `{rollout.get('expanded_v2_1_token_count')}`",
        f"- Completed: `{rollout.get('completed')}`",
        f"- Dead end: `{rollout.get('dead_end')}`",
        f"- Max tokens exceeded: `{rollout.get('max_tokens_exceeded')}`",
        "",
        "## Interpretation",
        "",
        str(summary.get("interpretation")),
        "",
        "## Next Step",
        "",
        str(summary.get("next_step")),
        "",
    ]
    return "\n".join(lines)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Smoke a trained v3 mapper checkpoint through runtime-backed rollout.")
    parser.add_argument("--mapper-checkpoint-path", required=True)
    parser.add_argument("--control-checkpoint-path", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--report-output")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--chart-end-ms", type=int, default=1_000)
    parser.add_argument("--max-tokens-per-window", type=int, default=256)
    parser.add_argument("--audio-path", default="synthetic.wav")
    parser.add_argument("--normalized-difficulty", type=float, default=0.5)
    parser.add_argument("--include-control-attention-kv-cache", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)
    summary = run_trained_v3_runtime_rollout_smoke(
        mapper_checkpoint_path=args.mapper_checkpoint_path,
        control_checkpoint_path=args.control_checkpoint_path,
        output_summary_path=args.summary_output,
        output_report_path=args.report_output,
        device_name=args.device,
        chart_end_ms=args.chart_end_ms,
        max_tokens_per_window=args.max_tokens_per_window,
        audio_path=args.audio_path,
        normalized_difficulty=args.normalized_difficulty,
        include_control_attention_kv_cache=args.include_control_attention_kv_cache,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    print(
        "mapper_v3_trained_runtime_rollout_done "
        f"route={summary['decision']['route']} "
        f"windows={summary['rollout']['window_count']} "
        f"tokens={summary['rollout']['token_count']} "
        f"timepoints={summary['rollout']['timepoint_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
