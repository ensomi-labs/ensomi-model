from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from pulsefield_model.evals.mapper_v3_trained_runtime_rollout import (
    FixedGridFitter,
    SyntheticTimingProvider,
    _numeric_summary,
    _real_audio_length_ms,
    _safe_ratio,
    _timepoint_to_dict,
)
from pulsefield_model.inference.mapper_v2_1_rollout import (
    MapperV21GenerationStep,
    generate_full_song_rollout_v2_1,
    rollout_to_timepoints_v2_1,
    session_window_batch_provider_v2_1,
)
from pulsefield_model.inference.model_runtime import ModelRuntimeConfig, load_model_runtime
from pulsefield_model.inference.session_runtime import SessionRuntime, SessionRuntimeConfig
from pulsefield_model.models.mapper.v2_1 import MapperV21Model, MapperV21Vocab


SUMMARY_SCHEMA_VERSION = 1


class V21LogitDiagnosticsCollector:
    def __init__(self, *, vocab: MapperV21Vocab, top_k: int = 5, max_examples: int = 12) -> None:
        self.vocab = vocab
        self.top_k = int(top_k)
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        self.max_examples = int(max_examples)
        if self.max_examples < 0:
            raise ValueError("max_examples must be non-negative")
        self.step_count = 0
        self.lane_action_valid_step_count = 0
        self.lane_action_top1_step_count = 0
        self.lane_action_topk_step_count = 0
        self.argmax_kind_counts: Counter[str] = Counter()
        self.topk_kind_counts: Counter[str] = Counter()
        self.valid_kind_counts: Counter[str] = Counter()
        self.best_lane_action_ranks: list[int] = []
        self.best_lane_action_margins: list[float] = []
        self.examples: list[dict[str, Any]] = []

    def observe(self, step: MapperV21GenerationStep, logits: torch.Tensor) -> None:
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
        if argmax_kind == "lane_action":
            self.lane_action_top1_step_count += 1
        if "lane_action" in top_kinds:
            self.lane_action_topk_step_count += 1

        lane_action_ids = [
            int(token_id)
            for token_id in self.vocab.lane_action_token_ids
            if 0 <= int(token_id) < int(valid_mask.numel()) and bool(valid_mask[int(token_id)].item())
        ]
        best_lane_action_id: int | None = None
        best_lane_action_rank: int | None = None
        best_lane_action_margin: float | None = None
        if lane_action_ids:
            self.lane_action_valid_step_count += 1
            lane_action_tensor = torch.tensor(lane_action_ids, dtype=torch.long, device=masked.device)
            lane_action_logits = masked.index_select(0, lane_action_tensor)
            best_index = int(torch.argmax(lane_action_logits).item())
            best_lane_action_id = lane_action_ids[best_index]
            best_lane_action_logit = float(lane_action_logits[best_index].item())
            argmax_logit = float(masked[argmax_id].item())
            best_lane_action_rank = int(torch.sum(masked > best_lane_action_logit).item()) + 1
            best_lane_action_margin = best_lane_action_logit - argmax_logit
            self.best_lane_action_ranks.append(best_lane_action_rank)
            self.best_lane_action_margins.append(best_lane_action_margin)

        if len(self.examples) < self.max_examples:
            self.examples.append(
                {
                    "step": int(step.token_index),
                    "current_ms": int(step.state.current_ms),
                    "argmax_token": self.vocab.token_name(argmax_id),
                    "argmax_kind": argmax_kind,
                    "top_tokens": [self.vocab.token_name(token_id) for token_id in top_ids],
                    "top_kinds": top_kinds,
                    "best_lane_action_token": (
                        None if best_lane_action_id is None else self.vocab.token_name(best_lane_action_id)
                    ),
                    "best_lane_action_rank": best_lane_action_rank,
                    "best_lane_action_margin_vs_argmax": best_lane_action_margin,
                }
            )

    def to_dict(self, emitted_tokens: Sequence[int]) -> dict[str, Any]:
        emitted_kind_counts = Counter(_token_kind(self.vocab, int(token_id)) for token_id in emitted_tokens)
        return {
            "enabled": True,
            "top_k": int(self.top_k),
            "step_count": int(self.step_count),
            "lane_action_valid_step_count": int(self.lane_action_valid_step_count),
            "lane_action_valid_step_ratio": _safe_ratio(self.lane_action_valid_step_count, self.step_count),
            "lane_action_top1_step_count": int(self.lane_action_top1_step_count),
            "lane_action_top1_step_ratio": _safe_ratio(self.lane_action_top1_step_count, self.step_count),
            "lane_action_topk_step_count": int(self.lane_action_topk_step_count),
            "lane_action_topk_step_ratio": _safe_ratio(self.lane_action_topk_step_count, self.step_count),
            "argmax_kind_counts": dict(sorted(self.argmax_kind_counts.items())),
            "topk_kind_counts": dict(sorted(self.topk_kind_counts.items())),
            "valid_kind_counts": dict(sorted(self.valid_kind_counts.items())),
            "emitted_kind_counts": dict(sorted(emitted_kind_counts.items())),
            "best_lane_action_rank": _numeric_summary(self.best_lane_action_ranks),
            "best_lane_action_margin_vs_argmax": _numeric_summary(self.best_lane_action_margins),
            "examples": list(self.examples),
        }


def run_trained_v21_runtime_rollout_smoke(
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
    if not isinstance(runtime.mapper_model, MapperV21Model):
        raise TypeError(f"expected MapperV21Model runtime, got {type(runtime.mapper_model).__name__}")
    if not isinstance(runtime.vocab, MapperV21Vocab):
        raise TypeError(f"expected MapperV21Vocab runtime, got {type(runtime.vocab).__name__}")

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
        session_id="mapper-v21-trained-runtime-rollout-smoke",
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
        V21LogitDiagnosticsCollector(vocab=runtime.vocab, top_k=int(logit_top_k))
        if bool(collect_logit_diagnostics)
        else None
    )
    rollout = generate_full_song_rollout_v2_1(
        model=runtime.mapper_model,
        vocab=runtime.vocab,
        chart_end_ms=chart_end_ms,
        window_batch_provider=session_window_batch_provider_v2_1(
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
        logits_observer=None if logit_diagnostics is None else logit_diagnostics.observe,
    )
    timepoints = rollout_to_timepoints_v2_1(rollout, runtime.vocab)

    mapper_metadata = dict(runtime.checkpoint_metadata["mapper"])
    checks = {
        "mapper_version_v2_1": mapper_metadata.get("version") == "v2_1",
        "runtime_model_v2_1": isinstance(runtime.mapper_model, MapperV21Model),
        "runtime_vocab_v2_1": isinstance(runtime.vocab, MapperV21Vocab),
        "filtered_embedded_control_encoder": len(tuple(mapper_metadata.get("filtered_control_encoder_keys", ()))) > 0,
        "rollout_window_count_positive": len(rollout.windows) > 0,
        "rollout_token_count_positive": len(rollout.tokens) > 0,
        "rollout_not_dead_end": not bool(rollout.dead_end),
        "rollout_not_max_tokens_exceeded": not bool(rollout.max_tokens_exceeded),
    }
    route = "TEST_NEXT" if all(checks.values()) else "MUTATE"
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Mapper v2.1 trained runtime rollout",
        "decision": {
            "route": route,
            "reason": (
                "all trained runtime rollout gates passed"
                if route == "TEST_NEXT"
                else "one or more trained runtime rollout gates failed"
            ),
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
            "The trained v2.1 checkpoint loaded through ModelRuntime and executed the runtime-backed online rollout path."
            if route == "TEST_NEXT"
            else "The v2.1 runtime boundary needs repair before baseline comparison."
        ),
        "next_step": (
            "Compare v2.1 rollout timing against the v3 multicase timing audit."
            if route == "TEST_NEXT"
            else "Repair loader/provider/generation failure before baseline comparison."
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


def _token_kind(vocab: MapperV21Vocab, token_id: int) -> str:
    token_id = int(token_id)
    if token_id == vocab.pad_id:
        return "pad"
    if token_id == vocab.bos_id:
        return "bos"
    if token_id == vocab.eos_id:
        return "eos"
    if vocab.is_time_shift_token(token_id):
        return "time_shift"
    if vocab.is_lane_action_token(token_id):
        return "lane_action"
    return "other"


def _report_markdown(summary: Mapping[str, Any]) -> str:
    decision = _mapping(summary.get("decision"))
    config = _mapping(summary.get("config"))
    runtime = _mapping(summary.get("runtime"))
    mapper = _mapping(runtime.get("mapper"))
    rollout = _mapping(summary.get("rollout"))
    diagnostics = _mapping(summary.get("logit_diagnostics"))
    rank_summary = _mapping(diagnostics.get("best_lane_action_rank"))
    margin_summary = _mapping(diagnostics.get("best_lane_action_margin_vs_argmax"))
    lines = [
        "# Mapper v2.1 Trained Runtime Rollout Result Report",
        "",
        "## Scope",
        "",
        "This smoke verifies that a trained v2.1 checkpoint can cross the inference-runtime boundary and execute the online rollout path. It is not a generated chart-quality result.",
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
        f"- Mapper runtime version: `{mapper.get('version')}`",
        f"- Chart end: `{config.get('chart_end_ms')}` ms",
        f"- Windows: `{rollout.get('window_count')}`",
        f"- Generated v2.1 tokens: `{rollout.get('token_count')}`",
        f"- Generated event timepoints: `{rollout.get('timepoint_count')}`",
        f"- Timepoint preview: `{rollout.get('timepoints')}`",
        f"- Completed: `{rollout.get('completed')}`",
        f"- Dead end: `{rollout.get('dead_end')}`",
        f"- Max tokens exceeded: `{rollout.get('max_tokens_exceeded')}`",
        "",
        "## Diagnostics",
        "",
        f"- Enabled: `{diagnostics.get('enabled')}`",
        f"- Lane-action-valid steps: `{diagnostics.get('lane_action_valid_step_count')}` / `{diagnostics.get('step_count')}`",
        f"- Lane-action top-1 steps: `{diagnostics.get('lane_action_top1_step_count')}`",
        f"- Lane-action top-k steps: `{diagnostics.get('lane_action_topk_step_count')}`",
        f"- Argmax kind counts: `{diagnostics.get('argmax_kind_counts')}`",
        f"- Emitted kind counts: `{diagnostics.get('emitted_kind_counts')}`",
        f"- Best lane-action rank median: `{rank_summary.get('median')}`",
        f"- Best lane-action margin median: `{margin_summary.get('median')}`",
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
    parser = argparse.ArgumentParser(description="Smoke a trained v2.1 mapper checkpoint through runtime-backed rollout.")
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
    parser.add_argument("--timepoint-preview-limit", type=int, default=32)
    args = parser.parse_args(argv)
    summary = run_trained_v21_runtime_rollout_smoke(
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
        timepoint_preview_limit=args.timepoint_preview_limit,
    )
    print(
        "mapper_v21_trained_runtime_rollout_done "
        f"route={summary['decision']['route']} "
        f"windows={summary['rollout']['window_count']} "
        f"tokens={summary['rollout']['token_count']} "
        f"timepoints={summary['rollout']['timepoint_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
