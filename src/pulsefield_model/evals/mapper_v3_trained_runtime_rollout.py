from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from pulsefield_model.inference.mapper_v3_rollout import (
    MapperV3GenerationStep,
    MapperV3LogitsTransform,
    generate_full_song_rollout_v3,
    generated_v3_tokens_to_v2_1_tokens,
    rollout_to_timepoints_v3,
    session_window_batch_provider_v3,
)
from pulsefield_model.inference.model_runtime import ModelRuntimeConfig, load_model_runtime
from pulsefield_model.inference.session_runtime import SessionRuntime, SessionRuntimeConfig
from pulsefield_model.inference.stream_with_cache import audio_length_ms_from_file
from pulsefield_model.models.mapper.v3 import MapperV3Model, MapperV3Vocab
from pulsefield_model.timing.grid_fitting.types import TimingFitDiagnostics, TimingFitResult
from pulsefield_model.timing.schema import FittedTimingGrid, FrameTimingPrediction, TimingSegment


SUMMARY_SCHEMA_VERSION = 1


class V3LogitDiagnosticsCollector:
    def __init__(self, *, vocab: MapperV3Vocab, top_k: int = 5, max_examples: int = 12) -> None:
        self.vocab = vocab
        self.top_k = int(top_k)
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        self.max_examples = int(max_examples)
        if self.max_examples < 0:
            raise ValueError("max_examples must be non-negative")
        self.step_count = 0
        self.event_valid_step_count = 0
        self.event_top1_step_count = 0
        self.event_topk_step_count = 0
        self.argmax_kind_counts: Counter[str] = Counter()
        self.topk_kind_counts: Counter[str] = Counter()
        self.valid_kind_counts: Counter[str] = Counter()
        self.best_event_ranks: list[int] = []
        self.best_event_margins: list[float] = []
        self.examples: list[dict[str, Any]] = []

    def observe(self, step: MapperV3GenerationStep, logits: torch.Tensor) -> None:
        flat_logits = torch.as_tensor(logits, dtype=torch.float32).reshape(-1)
        valid_mask = step.valid_token_mask.to(device=flat_logits.device, dtype=torch.bool).reshape(-1)
        if int(flat_logits.numel()) != int(valid_mask.numel()):
            raise ValueError(f"logits must contain {valid_mask.numel()} values, got {flat_logits.numel()}")
        if not bool(valid_mask.any().item()):
            return

        self.step_count += 1
        masked = flat_logits.masked_fill(~valid_mask, -torch.inf)
        valid_ids = torch.nonzero(valid_mask, as_tuple=False).reshape(-1).tolist()
        for token_id in valid_ids:
            self.valid_kind_counts[_token_kind(self.vocab, int(token_id))] += 1

        argmax_id = int(torch.argmax(masked).item())
        argmax_kind = _token_kind(self.vocab, argmax_id)
        self.argmax_kind_counts[argmax_kind] += 1

        top_k = min(self.top_k, len(valid_ids))
        top_ids = [int(token_id) for token_id in torch.topk(masked, k=top_k).indices.tolist()]
        top_kinds = [_token_kind(self.vocab, token_id) for token_id in top_ids]
        self.topk_kind_counts.update(top_kinds)
        if argmax_kind == "event":
            self.event_top1_step_count += 1
        if "event" in top_kinds:
            self.event_topk_step_count += 1

        event_ids = [
            int(token_id)
            for token_id in self.vocab.event_token_ids
            if 0 <= int(token_id) < int(valid_mask.numel()) and bool(valid_mask[int(token_id)].item())
        ]
        best_event_id: int | None = None
        best_event_rank: int | None = None
        best_event_margin: float | None = None
        if event_ids:
            self.event_valid_step_count += 1
            event_id_tensor = torch.tensor(event_ids, dtype=torch.long, device=masked.device)
            event_logits = masked.index_select(0, event_id_tensor)
            best_index = int(torch.argmax(event_logits).item())
            best_event_id = event_ids[best_index]
            best_event_logit = float(event_logits[best_index].item())
            argmax_logit = float(masked[argmax_id].item())
            best_event_rank = int(torch.sum(masked > best_event_logit).item()) + 1
            best_event_margin = best_event_logit - argmax_logit
            self.best_event_ranks.append(best_event_rank)
            self.best_event_margins.append(best_event_margin)

        if len(self.examples) < self.max_examples:
            self.examples.append(
                {
                    "step": int(step.token_index),
                    "current_ms": int(step.state.current_ms),
                    "argmax_token": self.vocab.token_name(argmax_id),
                    "argmax_kind": argmax_kind,
                    "top_tokens": [self.vocab.token_name(token_id) for token_id in top_ids],
                    "top_kinds": top_kinds,
                    "best_event_token": None if best_event_id is None else self.vocab.token_name(best_event_id),
                    "best_event_rank": best_event_rank,
                    "best_event_margin_vs_argmax": best_event_margin,
                }
            )

    def to_dict(self, emitted_tokens: Sequence[int]) -> dict[str, Any]:
        emitted_kind_counts = Counter(_token_kind(self.vocab, int(token_id)) for token_id in emitted_tokens)
        return {
            "enabled": True,
            "top_k": int(self.top_k),
            "step_count": int(self.step_count),
            "event_valid_step_count": int(self.event_valid_step_count),
            "event_valid_step_ratio": _safe_ratio(self.event_valid_step_count, self.step_count),
            "event_top1_step_count": int(self.event_top1_step_count),
            "event_top1_step_ratio": _safe_ratio(self.event_top1_step_count, self.step_count),
            "event_topk_step_count": int(self.event_topk_step_count),
            "event_topk_step_ratio": _safe_ratio(self.event_topk_step_count, self.step_count),
            "argmax_kind_counts": dict(sorted(self.argmax_kind_counts.items())),
            "topk_kind_counts": dict(sorted(self.topk_kind_counts.items())),
            "valid_kind_counts": dict(sorted(self.valid_kind_counts.items())),
            "emitted_kind_counts": dict(sorted(emitted_kind_counts.items())),
            "best_event_rank": _numeric_summary(self.best_event_ranks),
            "best_event_margin_vs_argmax": _numeric_summary(self.best_event_margins),
            "examples": list(self.examples),
        }


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
    real_audio: bool = False,
    audio_length_ms: int | None = None,
    beatthis_device: str | None = None,
    beatthis_float16: bool = False,
    time_shift_length_penalty_alpha: float = 0.0,
    time_shift_delta_penalty_alpha: float = 0.0,
    collect_logit_diagnostics: bool = False,
    logit_top_k: int = 5,
    logit_max_examples: int = 12,
    logits_transform: MapperV3LogitsTransform | None = None,
    timepoint_preview_limit: int = 32,
) -> dict[str, Any]:
    chart_end_ms = int(chart_end_ms)
    if chart_end_ms <= 0 or chart_end_ms % 10 != 0:
        raise ValueError("chart_end_ms must be positive and aligned to 10ms")
    timepoint_preview_limit = int(timepoint_preview_limit)
    if timepoint_preview_limit < 0:
        raise ValueError("timepoint_preview_limit must be non-negative")
    frame_count = max(400, (chart_end_ms + 19) // 20)

    runtime = load_model_runtime(
        ModelRuntimeConfig(
            mapper_checkpoint_path=mapper_checkpoint_path,
            control_checkpoint_path=control_checkpoint_path,
            device=device_name,
            beatthis_device=beatthis_device,
            beatthis_float16=bool(beatthis_float16),
            eager_load_beatthis=False,
        )
    )
    if not isinstance(runtime.mapper_model, MapperV3Model):
        raise TypeError(f"expected MapperV3Model runtime, got {type(runtime.mapper_model).__name__}")
    if not isinstance(runtime.vocab, MapperV3Vocab):
        raise TypeError(f"expected MapperV3Vocab runtime, got {type(runtime.vocab).__name__}")

    synthetic_provider = None
    fitter = None
    mel_loader = None
    resolved_audio_path = Path(audio_path)
    if bool(real_audio):
        resolved_audio_length_ms = _real_audio_length_ms(resolved_audio_path, audio_length_ms=audio_length_ms)
    else:
        synthetic_provider = SyntheticTimingProvider(frame_count=frame_count, source_path=resolved_audio_path.as_posix())
        runtime = replace(runtime, beatthis_provider=synthetic_provider)
        fitter = FixedGridFitter()
        mel = np.zeros((frame_count, 160), dtype=np.float32)

        def load_mel(path: str | Path) -> np.ndarray:
            del path
            return mel

        mel_loader = load_mel
        resolved_audio_length_ms = chart_end_ms

    session_runtime = SessionRuntime(
        session_id="mapper-v3-trained-runtime-rollout-smoke",
        model_runtime=runtime,
        config=SessionRuntimeConfig(
            device=runtime.device,
            default_normalized_difficulty=float(normalized_difficulty),
            max_control_batch_size=4,
        ),
        **({} if mel_loader is None else {"mel_loader": mel_loader}),
        **({} if fitter is None else {"grid_fitter": fitter}),
    )
    audio_cache = session_runtime.prepare_audio(resolved_audio_path, audio_length_ms=resolved_audio_length_ms, start_ms=0)
    full_control_cache = session_runtime.prepare_full_control(max_batch_size=4)
    generator = torch.Generator(device=runtime.device)
    generator.manual_seed(int(seed))
    logit_diagnostics = (
        V3LogitDiagnosticsCollector(vocab=runtime.vocab, top_k=int(logit_top_k), max_examples=int(logit_max_examples))
        if bool(collect_logit_diagnostics)
        else None
    )
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
        time_shift_length_penalty_alpha=float(time_shift_length_penalty_alpha),
        time_shift_delta_penalty_alpha=float(time_shift_delta_penalty_alpha),
        generator=generator,
        logits_transform=logits_transform,
        logits_observer=None if logit_diagnostics is None else logit_diagnostics.observe,
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
            "real_audio": bool(real_audio),
            "audio_path": resolved_audio_path.as_posix(),
            "audio_length_ms": int(resolved_audio_length_ms),
            "synthetic_frame_count": None if bool(real_audio) else int(frame_count),
            "beatthis_device": beatthis_device,
            "beatthis_float16": bool(beatthis_float16),
            "time_shift_length_penalty_alpha": float(time_shift_length_penalty_alpha),
            "time_shift_delta_penalty_alpha": float(time_shift_delta_penalty_alpha),
            "collect_logit_diagnostics": bool(collect_logit_diagnostics),
            "logit_top_k": int(logit_top_k),
            "logit_max_examples": int(logit_max_examples),
            "logits_transform_enabled": logits_transform is not None,
            "timepoint_preview_limit": int(timepoint_preview_limit),
        },
        "runtime": {
            "mapper": mapper_metadata,
            "control": dict(runtime.checkpoint_metadata["control"]),
            "timing_provider_calls": (
                list(synthetic_provider.paths)
                if synthetic_provider is not None
                else [resolved_audio_path.as_posix()]
            ),
            "grid_fit_count": None if fitter is None else int(fitter.prediction_count),
            "audio_cache": {
                "audio_path": audio_cache.audio_path.as_posix(),
                "audio_length_ms": int(audio_cache.audio_length_ms),
                "audio_length_source": audio_cache.audio_length_source,
                "source_frame_count": int(audio_cache.source_frame_count),
                "padded_frame_count": int(audio_cache.padded_frame_count),
                "timing_provider": str(audio_cache.beatthis_prediction.provider),
                "timing_checkpoint_path": str(audio_cache.beatthis_prediction.checkpoint_path),
                "timing_frame_rate_hz": float(audio_cache.beatthis_prediction.frame_rate_hz),
                "timing_fit_score": float(audio_cache.timing_fit_result.score),
            },
            "full_control_cache": {
                "window_count": len(full_control_cache.start_ms_values),
                "max_batch_size": int(full_control_cache.max_batch_size),
            },
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
            "timepoints": [_timepoint_to_dict(timepoint) for timepoint in timepoints[:timepoint_preview_limit]],
            "timepoint_preview_limit": int(timepoint_preview_limit),
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
        "logit_diagnostics": (
            {"enabled": False}
            if logit_diagnostics is None
            else logit_diagnostics.to_dict(rollout.tokens)
        ),
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


def _real_audio_length_ms(audio_path: Path, *, audio_length_ms: int | None) -> int:
    if audio_length_ms is not None:
        value = int(audio_length_ms)
        if value <= 0:
            raise ValueError("audio_length_ms must be positive")
        return value
    resolved = audio_length_ms_from_file(audio_path)
    if resolved is None:
        raise ValueError(f"could not infer audio length for real audio: {audio_path}")
    return int(resolved)


def _token_kind(vocab: MapperV3Vocab, token_id: int) -> str:
    token_id = int(token_id)
    if token_id == vocab.pad_id:
        return "pad"
    if token_id == vocab.bos_id:
        return "bos"
    if token_id == vocab.eos_id:
        return "eos"
    if vocab.is_time_shift_token(token_id):
        return "time_shift"
    if vocab.is_event_token(token_id):
        return "event"
    return "other"


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    denominator = float(denominator)
    if denominator == 0.0:
        return 0.0
    return float(numerator) / denominator


def _numeric_summary(values: Sequence[int | float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    midpoint = count // 2
    if count % 2:
        median = ordered[midpoint]
    else:
        median = 0.5 * (ordered[midpoint - 1] + ordered[midpoint])
    return {
        "count": int(count),
        "min": float(ordered[0]),
        "median": float(median),
        "mean": float(math.fsum(ordered) / count),
        "max": float(ordered[-1]),
    }


def _timepoint_to_dict(timepoint: Any) -> dict[str, Any]:
    return {
        "time_ms": int(timepoint.time_ms),
        "lane_actions": [str(getattr(action, "value", action)) for action in timepoint.lane_actions],
    }


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
    diagnostics = _mapping(summary.get("logit_diagnostics"))
    rank_summary = _mapping(diagnostics.get("best_event_rank"))
    margin_summary = _mapping(diagnostics.get("best_event_margin_vs_argmax"))
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
        f"- Real audio: `{config.get('real_audio')}`",
        f"- Audio path: `{config.get('audio_path')}`",
        f"- Audio length: `{config.get('audio_length_ms')}` ms",
        f"- Time-shift length penalty alpha: `{config.get('time_shift_length_penalty_alpha')}`",
        f"- Time-shift delta penalty alpha: `{config.get('time_shift_delta_penalty_alpha')}`",
        f"- Logit diagnostics: `{config.get('collect_logit_diagnostics')}`",
        f"- Mapper runtime version: `{mapper.get('version')}`",
        f"- Filtered embedded control keys: `{len(tuple(mapper.get('filtered_control_encoder_keys', ())))}`",
        f"- Chart end: `{config.get('chart_end_ms')}` ms",
        f"- Windows: `{rollout.get('window_count')}`",
        f"- Generated v3 tokens: `{rollout.get('token_count')}`",
        f"- Generated event timepoints: `{rollout.get('timepoint_count')}`",
        f"- Timepoint preview: `{rollout.get('timepoints')}`",
        f"- Expanded v2.1 tokens: `{rollout.get('expanded_v2_1_token_count')}`",
        f"- Completed: `{rollout.get('completed')}`",
        f"- Dead end: `{rollout.get('dead_end')}`",
        f"- Max tokens exceeded: `{rollout.get('max_tokens_exceeded')}`",
        "",
        "## Diagnostics",
        "",
        f"- Enabled: `{diagnostics.get('enabled')}`",
        f"- Event-valid steps: `{diagnostics.get('event_valid_step_count')}` / `{diagnostics.get('step_count')}`",
        f"- Event top-1 steps: `{diagnostics.get('event_top1_step_count')}`",
        f"- Event top-k steps: `{diagnostics.get('event_topk_step_count')}`",
        f"- Argmax kind counts: `{diagnostics.get('argmax_kind_counts')}`",
        f"- Emitted kind counts: `{diagnostics.get('emitted_kind_counts')}`",
        f"- Best-event rank median: `{rank_summary.get('median')}`",
        f"- Best-event margin median: `{margin_summary.get('median')}`",
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
    parser.add_argument("--real-audio", action="store_true")
    parser.add_argument("--audio-length-ms", type=int)
    parser.add_argument("--beatthis-device")
    parser.add_argument("--beatthis-float16", action="store_true")
    parser.add_argument("--normalized-difficulty", type=float, default=0.5)
    parser.add_argument("--include-control-attention-kv-cache", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--time-shift-length-penalty-alpha", type=float, default=0.0)
    parser.add_argument("--time-shift-delta-penalty-alpha", type=float, default=0.0)
    parser.add_argument("--collect-logit-diagnostics", action="store_true")
    parser.add_argument("--logit-top-k", type=int, default=5)
    parser.add_argument("--logit-max-examples", type=int, default=12)
    parser.add_argument("--timepoint-preview-limit", type=int, default=32)
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
        real_audio=args.real_audio,
        audio_length_ms=args.audio_length_ms,
        beatthis_device=args.beatthis_device,
        beatthis_float16=args.beatthis_float16,
        time_shift_length_penalty_alpha=args.time_shift_length_penalty_alpha,
        time_shift_delta_penalty_alpha=args.time_shift_delta_penalty_alpha,
        collect_logit_diagnostics=args.collect_logit_diagnostics,
        logit_top_k=args.logit_top_k,
        logit_max_examples=args.logit_max_examples,
        timepoint_preview_limit=args.timepoint_preview_limit,
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
