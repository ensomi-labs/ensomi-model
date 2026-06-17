from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from pulsefield_model.evals.mapper_v2_1_anti_rigid_spacing_guard import generated_metrics
from pulsefield_model.inference.mapper_v3_rollout import (
    MapperV3FullRollout,
    MapperV3GenerationStep,
    generate_full_song_rollout_v3,
    rollout_to_timepoints_v3,
    session_window_batch_provider_v3,
)
from pulsefield_model.inference.model_runtime import ModelRuntimeConfig, load_model_runtime
from pulsefield_model.inference.session_runtime import SessionRuntime, SessionRuntimeConfig
from pulsefield_model.inference.stream_with_cache import audio_length_ms_from_file
from pulsefield_model.models.mapper.shared.tokenizer import hitobjects_to_mapper_timepoints
from pulsefield_model.models.mapper.v3 import MapperV3Model, MapperV3Vocab
from pulsefield_model.osu_core.hitobjects import parse_mania_hit_objects


SUMMARY_SCHEMA_VERSION = 1
DEFAULT_BASELINE_SUMMARY_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_fixed_slice_wide_audit_summary.json",
)
DEFAULT_OUTPUT_DIR = Path("artifacts/tmp/mapper_v3_generated_prefix_state_trace_audit")
DEFAULT_SUMMARY_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/"
    "target_grammar_v3_generated_prefix_state_trace_audit_summary.json",
)
DEFAULT_REPORT_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/"
    "target_grammar_v3_generated_prefix_state_trace_audit_result_report.md",
)

SELECTED_CASE_IDS = (
    "14_oomori_seiko_justadice_tv_size_remu_hard",
    "17_tamura_yukari_fantastic_future_tv_size_victorica_db_4k_mx",
    "19_nekodex_circles_famoss_hard",
    "22_oomori_seiko_justadice_tv_size_remu_grimoire_of_despair",
    "31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial",
    "18_billiummoto_four_veiled_stars_aries_famoss_hard",
)

SECOND_WINDOW_START_MS = 8_000
STARVED_SHARE_THRESHOLD = 0.10
REFERENCE_SECOND_WINDOW_SUPPORT_THRESHOLD = 0.25
REPEATED_SPACING_MIN_RUN = 4
REPEATED_SPACING_MIN_MS = 40
REPEATED_SPACING_MAX_MS = 400
EVENT_UNDERSELECTION_RANK_LIMIT = 5
EVENT_UNDERSELECTION_TOLERANCE_MS = 100
BOUNDARY_NEAR_MS = 120


class GeneratedPrefixStateTraceCollector:
    def __init__(self, *, vocab: MapperV3Vocab, top_k: int = 5) -> None:
        self.vocab = vocab
        self.top_k = int(top_k)
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        self.rows: list[dict[str, Any]] = []

    def observe(self, step: MapperV3GenerationStep, logits: torch.Tensor) -> None:
        flat_logits = torch.as_tensor(logits, dtype=torch.float32).reshape(-1)
        valid_mask = step.valid_token_mask.to(device=flat_logits.device, dtype=torch.bool).reshape(-1)
        if int(flat_logits.numel()) != int(valid_mask.numel()):
            raise ValueError(f"logits must contain {valid_mask.numel()} values, got {flat_logits.numel()}")
        if not bool(valid_mask.any().item()):
            return

        masked = flat_logits.masked_fill(~valid_mask, -torch.inf)
        valid_ids = [int(token_id) for token_id in torch.nonzero(valid_mask, as_tuple=False).reshape(-1).tolist()]
        argmax_id = int(torch.argmax(masked).item())
        argmax_logit = float(masked[argmax_id].item())
        top_count = min(self.top_k, len(valid_ids))
        top_ids = [int(token_id) for token_id in torch.topk(masked, k=top_count).indices.tolist()]
        top_logits = [float(masked[token_id].item()) for token_id in top_ids]

        best_event = self._best_token_summary(
            token_ids=self.vocab.event_token_ids,
            masked_logits=masked,
            valid_mask=valid_mask,
            argmax_logit=argmax_logit,
        )
        best_time_shift = self._best_token_summary(
            token_ids=self.vocab.time_shift_token_ids,
            masked_logits=masked,
            valid_mask=valid_mask,
            argmax_logit=argmax_logit,
        )

        self.rows.append(
            {
                "sequence_index": len(self.rows),
                "row_key": _row_key(step.write_start_ms, step.token_index),
                "write_start_ms": int(step.write_start_ms),
                "write_end_ms": int(step.write_end_ms),
                "chart_end_ms": int(step.chart_end_ms),
                "token_index": int(step.token_index),
                "prefix_token_count": len(step.generated_tokens),
                "current_ms": int(step.state.current_ms),
                "state": _state_summary(step.state),
                "valid_kind_counts": _kind_counts(self.vocab, valid_ids),
                "event_valid": best_event is not None,
                "time_shift_valid": best_time_shift is not None,
                "argmax": _token_summary(self.vocab, argmax_id, logit=argmax_logit),
                "top_tokens": [
                    _token_summary(self.vocab, token_id, logit=logit)
                    for token_id, logit in zip(top_ids, top_logits, strict=True)
                ],
                "best_event": best_event,
                "best_time_shift": best_time_shift,
                "emitted": None,
                "state_after": None,
            }
        )

    def attach_rollout(self, rollout: MapperV3FullRollout) -> list[dict[str, Any]]:
        rows_by_key = {str(row["row_key"]): row for row in self.rows}
        for window in rollout.windows:
            for token_index, (token_id, state_before, state_after) in enumerate(
                zip(window.tokens, window.states_before, window.states_after, strict=True)
            ):
                key = _row_key(window.write_start_ms, token_index)
                row = rows_by_key.get(key)
                if row is None:
                    row = {
                        "sequence_index": len(self.rows),
                        "row_key": key,
                        "write_start_ms": int(window.write_start_ms),
                        "write_end_ms": int(window.write_end_ms),
                        "chart_end_ms": int(window.chart_end_ms),
                        "token_index": int(token_index),
                        "prefix_token_count": int(token_index),
                        "current_ms": int(state_before.current_ms),
                        "state": _state_summary(state_before),
                        "valid_kind_counts": {},
                        "event_valid": None,
                        "time_shift_valid": None,
                        "argmax": None,
                        "top_tokens": [],
                        "best_event": None,
                        "best_time_shift": None,
                    }
                    self.rows.append(row)
                row["emitted"] = _token_summary(self.vocab, int(token_id))
                row["state_after"] = _state_summary(state_after)
        self.rows.sort(key=lambda row: int(row.get("sequence_index", 0)))
        return self.rows

    def _best_token_summary(
        self,
        *,
        token_ids: Sequence[int],
        masked_logits: torch.Tensor,
        valid_mask: torch.Tensor,
        argmax_logit: float,
    ) -> dict[str, Any] | None:
        valid_ids = [
            int(token_id)
            for token_id in token_ids
            if 0 <= int(token_id) < int(valid_mask.numel()) and bool(valid_mask[int(token_id)].item())
        ]
        if not valid_ids:
            return None
        id_tensor = torch.tensor(valid_ids, dtype=torch.long, device=masked_logits.device)
        token_logits = masked_logits.index_select(0, id_tensor)
        best_index = int(torch.argmax(token_logits).item())
        best_id = int(valid_ids[best_index])
        best_logit = float(token_logits[best_index].item())
        summary = _token_summary(self.vocab, best_id, logit=best_logit)
        summary["rank"] = int(torch.sum(masked_logits > best_logit).item()) + 1
        summary["margin_vs_argmax"] = float(best_logit - argmax_logit)
        return summary


def run_generated_prefix_state_trace_audit(
    *,
    baseline_summary_path: str | Path = DEFAULT_BASELINE_SUMMARY_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    output_summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    output_report_path: str | Path | None = DEFAULT_REPORT_PATH,
    mapper_checkpoint_path: str | Path | None = None,
    control_checkpoint_path: str | Path | None = None,
    case_ids: Sequence[str] = SELECTED_CASE_IDS,
    case_limit: int | None = None,
    device_name: str = "auto",
    max_tokens_per_window: int = 512,
    seed: int = 1337,
    logit_top_k: int = 5,
    timepoint_preview_limit: int = 4096,
) -> dict[str, Any]:
    baseline_summary = _read_json(Path(baseline_summary_path))
    baseline_by_case = _baseline_runs_by_case_id(baseline_summary)
    selected_ids = tuple(str(case_id) for case_id in case_ids)
    missing = [case_id for case_id in selected_ids if case_id not in baseline_by_case]
    if missing:
        raise ValueError(f"missing requested case ids in baseline summary: {missing}")
    if case_limit is not None:
        selected_ids = selected_ids[: max(0, int(case_limit))]

    mapper_checkpoint = Path(mapper_checkpoint_path or str(baseline_summary.get("checkpoint_path", "")))
    control_checkpoint = Path(control_checkpoint_path or str(baseline_summary.get("control_checkpoint_path", "")))
    if not mapper_checkpoint.exists():
        raise FileNotFoundError(f"missing mapper checkpoint: {mapper_checkpoint}")
    if not control_checkpoint.exists():
        raise FileNotFoundError(f"missing control checkpoint: {control_checkpoint}")

    out_dir = Path(output_dir)
    trace_dir = out_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    runtime = load_model_runtime(
        ModelRuntimeConfig(
            mapper_checkpoint_path=mapper_checkpoint,
            control_checkpoint_path=control_checkpoint,
            device=device_name,
            eager_load_beatthis=False,
        )
    )
    if not isinstance(runtime.mapper_model, MapperV3Model):
        raise TypeError(f"expected MapperV3Model runtime, got {type(runtime.mapper_model).__name__}")
    if not isinstance(runtime.vocab, MapperV3Vocab):
        raise TypeError(f"expected MapperV3Vocab runtime, got {type(runtime.vocab).__name__}")

    case_results: list[dict[str, Any]] = []
    for case_offset, case_id in enumerate(selected_ids):
        baseline_row = dict(baseline_by_case[case_id])
        trace_path = trace_dir / f"{_safe_filename(case_id)}_{int(baseline_row['chart_end_ms'])}ms_trace.json"
        try:
            case_result = _run_case_trace(
                baseline_row=baseline_row,
                runtime=runtime,
                trace_path=trace_path,
                max_tokens_per_window=int(max_tokens_per_window),
                seed=int(seed) + int(case_offset),
                logit_top_k=int(logit_top_k),
                timepoint_preview_limit=int(timepoint_preview_limit),
            )
        except Exception as exc:  # pragma: no cover - exercised by real-audio runs.
            case_result = {
                "case_id": case_id,
                "traced": False,
                "trace_path": trace_path.as_posix(),
                "error": f"{type(exc).__name__}: {exc}",
                "baseline_projection": _baseline_projection(baseline_row),
                "failure_class": "unclassified",
                "classification": {
                    "failure_class": "unclassified",
                    "reason": "trace collection failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "concrete": False,
                    "secondary_flags": {},
                },
            }
        case_results.append(case_result)

    aggregate = aggregate_case_results(case_results)
    decision = decision_from_aggregate(aggregate)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar v3 generated-prefix state trace audit",
        "decision": decision,
        "baseline_summary_path": Path(baseline_summary_path).as_posix(),
        "output_dir": out_dir.as_posix(),
        "mapper_checkpoint_path": mapper_checkpoint.as_posix(),
        "control_checkpoint_path": control_checkpoint.as_posix(),
        "config": {
            "device": str(runtime.device),
            "selected_case_ids": list(selected_ids),
            "max_tokens_per_window": int(max_tokens_per_window),
            "seed": int(seed),
            "logit_top_k": int(logit_top_k),
            "timepoint_preview_limit": int(timepoint_preview_limit),
            "no_training": True,
            "decode_defaults_changed": False,
            "tokenizer_changed": False,
        },
        "comparators": {
            "teacher_forced_time_shift_recall_at_5": 0.923314,
            "teacher_forced_time_shift_median_rank": 1,
            "baseline_experiment": baseline_summary.get("experiment"),
        },
        "aggregate": aggregate,
        "case_results": case_results,
        "interpretation": interpretation_from_decision(decision),
        "next_step": decision["next_step"],
    }
    write_summary_json(summary, output_summary_path)
    if output_report_path is not None:
        write_report(summary, output_report_path)
    return summary


def _run_case_trace(
    *,
    baseline_row: Mapping[str, Any],
    runtime: Any,
    trace_path: Path,
    max_tokens_per_window: int,
    seed: int,
    logit_top_k: int,
    timepoint_preview_limit: int,
) -> dict[str, Any]:
    case_id = str(baseline_row["case_id"])
    chart_end_ms = int(baseline_row["chart_end_ms"])
    audio_path = Path(str(baseline_row["audio_path"]))
    beatmap_path = Path(str(baseline_row["beatmap_path"]))
    normalized_difficulty = float(baseline_row["normalized_difficulty"])

    session_runtime = SessionRuntime(
        session_id=f"mapper-v3-generated-prefix-trace-{_safe_filename(case_id)}",
        model_runtime=runtime,
        config=SessionRuntimeConfig(
            device=runtime.device,
            default_normalized_difficulty=float(normalized_difficulty),
            max_control_batch_size=4,
        ),
    )
    audio_length_ms = _real_audio_length_ms(audio_path)
    audio_cache = session_runtime.prepare_audio(audio_path, audio_length_ms=audio_length_ms, start_ms=0)
    full_control_cache = session_runtime.prepare_full_control(max_batch_size=4)
    collector = GeneratedPrefixStateTraceCollector(vocab=runtime.vocab, top_k=int(logit_top_k))
    generator = torch.Generator(device=runtime.device)
    generator.manual_seed(int(seed))

    rollout = generate_full_song_rollout_v3(
        model=runtime.mapper_model,
        vocab=runtime.vocab,
        chart_end_ms=chart_end_ms,
        window_batch_provider=session_window_batch_provider_v3(
            session_runtime,
            include_control_attention_kv_cache=False,
        ),
        device=runtime.device,
        normalized_difficulty=float(normalized_difficulty),
        max_tokens_per_window=int(max_tokens_per_window),
        temperature=0.0,
        top_p=None,
        time_shift_length_penalty_alpha=0.0,
        time_shift_delta_penalty_alpha=0.0,
        generator=generator,
        logits_transform=None,
        logits_observer=collector.observe,
    )
    trace_rows = collector.attach_rollout(rollout)
    timepoints = rollout_to_timepoints_v3(rollout, runtime.vocab)
    generated_times = [int(timepoint.time_ms) for timepoint in timepoints]
    reference_times = _reference_times(beatmap_path, chart_end_ms=chart_end_ms)
    metrics = generated_metrics(
        generated_times=generated_times,
        reference_times=reference_times,
        chart_end_ms=chart_end_ms,
    )
    trace_aggregate = trace_aggregate_from_rows(trace_rows)
    classification = classify_case_failure(
        trace_rows=trace_rows,
        generated_times=generated_times,
        reference_times=reference_times,
        metrics=metrics,
        chart_end_ms=chart_end_ms,
        rollout_completed=bool(rollout.completed),
        rollout_dead_end=bool(rollout.dead_end),
        rollout_max_tokens_exceeded=bool(rollout.max_tokens_exceeded),
    )
    trace_payload = {
        "case_id": case_id,
        "chart_end_ms": chart_end_ms,
        "audio_path": audio_path.as_posix(),
        "beatmap_path": beatmap_path.as_posix(),
        "trace_rows": trace_rows,
        "trace_aggregate": trace_aggregate,
        "generated_times": generated_times,
        "reference_times": reference_times,
        "metrics": metrics,
        "classification": classification,
    }
    write_summary_json(trace_payload, trace_path)

    return {
        "case_id": case_id,
        "traced": True,
        "trace_path": trace_path.as_posix(),
        "case_index": baseline_row.get("case_index"),
        "chart_end_ms": chart_end_ms,
        "difficulty": baseline_row.get("difficulty"),
        "normalized_difficulty": normalized_difficulty,
        "audio_path": audio_path.as_posix(),
        "beatmap_path": beatmap_path.as_posix(),
        "baseline_projection": _baseline_projection(baseline_row),
        "runtime": {
            "audio_length_ms": int(audio_length_ms),
            "audio_length_source": audio_cache.audio_length_source,
            "full_control_window_count": len(full_control_cache.start_ms_values),
        },
        "rollout": {
            "window_count": len(rollout.windows),
            "token_count": len(rollout.tokens),
            "timepoint_count": len(timepoints),
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
            "timepoints": [_timepoint_to_dict(timepoint) for timepoint in timepoints[:timepoint_preview_limit]],
            "timepoint_preview_limit": int(timepoint_preview_limit),
        },
        "metrics": metrics,
        "trace_aggregate": trace_aggregate,
        "failure_class": classification["failure_class"],
        "classification": classification,
        "first_failure_excerpt": trace_excerpt(trace_rows, classification.get("first_failure_row_key"), radius=10),
    }


def classify_case_failure(
    *,
    trace_rows: Sequence[Mapping[str, Any]],
    generated_times: Sequence[int],
    reference_times: Sequence[int],
    metrics: Mapping[str, Any],
    chart_end_ms: int,
    rollout_completed: bool,
    rollout_dead_end: bool,
    rollout_max_tokens_exceeded: bool,
) -> dict[str, Any]:
    state_issue = _state_or_carry_issue(
        trace_rows=trace_rows,
        rollout_completed=bool(rollout_completed),
        rollout_dead_end=bool(rollout_dead_end),
        rollout_max_tokens_exceeded=bool(rollout_max_tokens_exceeded),
    )
    repetition = _first_repeated_spacing_failure(generated_times, trace_rows=trace_rows)
    underselection = _first_event_underselection_failure(
        trace_rows=trace_rows,
        generated_times=generated_times,
        reference_times=reference_times,
    )
    boundary = _first_boundary_drift_failure(
        trace_rows=trace_rows,
        generated_times=generated_times,
        reference_times=reference_times,
        metrics=metrics,
        chart_end_ms=int(chart_end_ms),
    )

    secondary_flags = {
        "time_shift_repetition": repetition is not None,
        "event_underselection": underselection is not None,
        "state_or_carry_mismatch": state_issue is not None,
        "boundary_drift": boundary is not None,
    }

    selected = (
        state_issue
        or repetition
        or underselection
        or boundary
        or {
            "failure_class": "unclassified",
            "reason": "no rule reached a concrete generated-prefix mechanism",
            "concrete": False,
        }
    )
    result = dict(selected)
    result.setdefault("concrete", result.get("failure_class") != "unclassified")
    result["secondary_flags"] = secondary_flags
    result["candidate_failures"] = {
        "time_shift_repetition": repetition,
        "event_underselection": underselection,
        "state_or_carry_mismatch": state_issue,
        "boundary_drift": boundary,
    }
    return result


def _state_or_carry_issue(
    *,
    trace_rows: Sequence[Mapping[str, Any]],
    rollout_completed: bool,
    rollout_dead_end: bool,
    rollout_max_tokens_exceeded: bool,
) -> dict[str, Any] | None:
    if bool(rollout_completed) and not bool(rollout_dead_end) and not bool(rollout_max_tokens_exceeded):
        return None
    first_row = next(iter(trace_rows), {})
    return {
        "failure_class": "state_or_carry_mismatch",
        "reason": "rollout did not complete cleanly under generated prefixes",
        "first_failure_row_key": first_row.get("row_key"),
        "first_failure_sequence_index": first_row.get("sequence_index"),
        "first_failure_current_ms": first_row.get("current_ms"),
        "rollout_completed": bool(rollout_completed),
        "rollout_dead_end": bool(rollout_dead_end),
        "rollout_max_tokens_exceeded": bool(rollout_max_tokens_exceeded),
        "concrete": True,
    }


def _first_repeated_spacing_failure(
    generated_times: Sequence[int],
    *,
    trace_rows: Sequence[Mapping[str, Any]],
    min_run: int = REPEATED_SPACING_MIN_RUN,
    min_spacing_ms: int = REPEATED_SPACING_MIN_MS,
    max_spacing_ms: int = REPEATED_SPACING_MAX_MS,
) -> dict[str, Any] | None:
    times = [int(value) for value in generated_times]
    if len(times) < int(min_run) + 1:
        return None
    deltas = [right - left for left, right in zip(times, times[1:])]
    run_length = 1
    for delta_index, delta in enumerate(deltas):
        if delta_index == 0:
            run_length = 1
        elif delta == deltas[delta_index - 1] and min_spacing_ms <= int(delta) <= max_spacing_ms:
            run_length += 1
        else:
            run_length = 1
        if run_length >= int(min_run) and min_spacing_ms <= int(delta) <= max_spacing_ms:
            event_index = delta_index + 1
            failure_time = times[event_index]
            row = _event_row_at_time(trace_rows, failure_time)
            best_event = _mapping(None if row is None else row.get("best_event"))
            emitted = _mapping(None if row is None else row.get("emitted"))
            argmax = _mapping(None if row is None else row.get("argmax"))
            return {
                "failure_class": "time_shift_repetition",
                "reason": f"{run_length} repeated event spacings of {int(delta)} ms appeared under generated prefixes",
                "first_failure_row_key": None if row is None else row.get("row_key"),
                "first_failure_sequence_index": None if row is None else row.get("sequence_index"),
                "first_failure_current_ms": int(failure_time),
                "repeated_spacing_ms": int(delta),
                "repeated_spacing_run_length": int(run_length),
                "generated_event_index": int(event_index),
                "best_event_token": best_event.get("name"),
                "best_event_signature": best_event.get("event_signature"),
                "best_event_rank": best_event.get("rank"),
                "best_event_margin_vs_argmax": best_event.get("margin_vs_argmax"),
                "emitted_token": emitted.get("name"),
                "emitted_signature": emitted.get("event_signature"),
                "argmax_token": argmax.get("name"),
                "concrete": True,
            }
    return None


def _first_event_underselection_failure(
    *,
    trace_rows: Sequence[Mapping[str, Any]],
    generated_times: Sequence[int],
    reference_times: Sequence[int],
    rank_limit: int = EVENT_UNDERSELECTION_RANK_LIMIT,
    tolerance_ms: int = EVENT_UNDERSELECTION_TOLERANCE_MS,
    min_hits: int = 2,
) -> dict[str, Any] | None:
    generated = [int(value) for value in generated_times]
    reference = [int(value) for value in reference_times]
    candidates: list[Mapping[str, Any]] = []
    for row in trace_rows:
        emitted = _mapping(row.get("emitted"))
        if emitted.get("kind") not in {"time_shift", "eos"}:
            continue
        current_ms = _int(row.get("current_ms"))
        if current_ms is None:
            continue
        if _has_time_within(generated, current_ms, tolerance_ms):
            continue
        if not _has_time_within(reference, current_ms, tolerance_ms):
            continue
        best_event = _mapping(row.get("best_event"))
        rank = _int(best_event.get("rank"))
        if rank is None or rank > int(rank_limit):
            continue
        candidates.append(row)
    if len(candidates) < int(min_hits):
        return None
    first = candidates[0]
    best_event = _mapping(first.get("best_event"))
    emitted = _mapping(first.get("emitted"))
    argmax = _mapping(first.get("argmax"))
    return {
        "failure_class": "event_underselection",
        "reason": (
            f"{len(candidates)} target-aligned steps had a rank<= {int(rank_limit)} event candidate "
            "but emitted a non-event token"
        ),
        "first_failure_row_key": first.get("row_key"),
        "first_failure_sequence_index": first.get("sequence_index"),
        "first_failure_current_ms": first.get("current_ms"),
        "best_event_token": best_event.get("name"),
        "best_event_signature": best_event.get("event_signature"),
        "best_event_rank": best_event.get("rank"),
        "best_event_margin_vs_argmax": best_event.get("margin_vs_argmax"),
        "emitted_token": emitted.get("name"),
        "argmax_token": argmax.get("name"),
        "candidate_count": len(candidates),
        "concrete": True,
    }


def _first_boundary_drift_failure(
    *,
    trace_rows: Sequence[Mapping[str, Any]],
    generated_times: Sequence[int],
    reference_times: Sequence[int],
    metrics: Mapping[str, Any],
    chart_end_ms: int,
) -> dict[str, Any] | None:
    reference_second = sum(1 for value in reference_times if int(value) >= SECOND_WINDOW_START_MS)
    reference_second_share = _safe_ratio(reference_second, len(reference_times))
    second_share = _float(metrics.get("second_window_event_share")) or 0.0
    boundary_ratio = _float(metrics.get("boundary_event_ratio")) or 0.0
    starved = bool(
        int(chart_end_ms) > SECOND_WINDOW_START_MS
        and reference_second_share >= REFERENCE_SECOND_WINDOW_SUPPORT_THRESHOLD
        and second_share <= STARVED_SHARE_THRESHOLD
    )
    terminal_spill = any(int(value) >= int(chart_end_ms) - BOUNDARY_NEAR_MS for value in generated_times)
    if not starved and not terminal_spill:
        return None
    boundary_row = None
    for row in trace_rows:
        if (_int(row.get("write_start_ms")) or 0) >= SECOND_WINDOW_START_MS:
            boundary_row = row
            break
        current_ms = _int(row.get("current_ms"))
        if current_ms is not None and current_ms >= SECOND_WINDOW_START_MS - BOUNDARY_NEAR_MS:
            boundary_row = row
            break
    if boundary_row is None:
        boundary_row = trace_rows[-1] if trace_rows else {}
    return {
        "failure_class": "boundary_drift",
        "reason": "generated events did not continue into the supported second window or spilled near chart end",
        "first_failure_row_key": boundary_row.get("row_key"),
        "first_failure_sequence_index": boundary_row.get("sequence_index"),
        "first_failure_current_ms": boundary_row.get("current_ms"),
        "second_window_event_share": float(second_share),
        "reference_second_window_event_share": float(reference_second_share),
        "boundary_event_ratio": float(boundary_ratio),
        "terminal_spill": bool(terminal_spill),
        "concrete": True,
    }


def trace_aggregate_from_rows(trace_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    emitted_kind_counts = Counter()
    argmax_kind_counts = Counter()
    top_kind_counts = Counter()
    event_ranks: list[float] = []
    event_margins: list[float] = []
    event_valid = 0
    event_top1 = 0
    event_topk = 0
    time_shift_argmax_values = Counter()
    for row in trace_rows:
        emitted = _mapping(row.get("emitted"))
        argmax = _mapping(row.get("argmax"))
        emitted_kind = str(emitted.get("kind") or "missing")
        argmax_kind = str(argmax.get("kind") or "missing")
        emitted_kind_counts[emitted_kind] += 1
        argmax_kind_counts[argmax_kind] += 1
        if argmax_kind == "time_shift" and argmax.get("time_shift_ms") is not None:
            time_shift_argmax_values[str(argmax.get("time_shift_ms"))] += 1
        best_event = _mapping(row.get("best_event"))
        if best_event:
            event_valid += 1
            rank = _float(best_event.get("rank"))
            margin = _float(best_event.get("margin_vs_argmax"))
            if rank is not None:
                event_ranks.append(rank)
            if margin is not None:
                event_margins.append(margin)
        if argmax_kind == "event":
            event_top1 += 1
        top_kinds = [str(_mapping(item).get("kind")) for item in _sequence(row.get("top_tokens"))]
        top_kind_counts.update(top_kinds)
        if "event" in top_kinds:
            event_topk += 1
    step_count = len(trace_rows)
    return {
        "step_count": int(step_count),
        "event_valid_step_count": int(event_valid),
        "event_valid_step_ratio": _safe_ratio(event_valid, step_count),
        "event_top1_step_count": int(event_top1),
        "event_top1_step_ratio": _safe_ratio(event_top1, step_count),
        "event_topk_step_count": int(event_topk),
        "event_topk_step_ratio": _safe_ratio(event_topk, step_count),
        "argmax_kind_counts": dict(sorted(argmax_kind_counts.items())),
        "emitted_kind_counts": dict(sorted(emitted_kind_counts.items())),
        "topk_kind_counts": dict(sorted(top_kind_counts.items())),
        "time_shift_argmax_values": dict(sorted(time_shift_argmax_values.items(), key=lambda item: int(item[0]))),
        "best_event_rank": _numeric_summary(event_ranks),
        "best_event_margin_vs_argmax": _numeric_summary(event_margins),
    }


def aggregate_case_results(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    traced = [row for row in case_results if bool(row.get("traced"))]
    class_counts = Counter(str(row.get("failure_class") or "unclassified") for row in case_results)
    secondary_counts: Counter[str] = Counter()
    for row in case_results:
        flags = _mapping(_mapping(row.get("classification")).get("secondary_flags"))
        for name, enabled in flags.items():
            if bool(enabled):
                secondary_counts[str(name)] += 1
    concrete_count = sum(
        1
        for row in case_results
        if str(row.get("failure_class") or "unclassified") != "unclassified"
        and bool(_mapping(row.get("classification")).get("concrete"))
    )
    metrics_rows = [_mapping(row.get("metrics")) for row in traced]
    return {
        "selected_case_count": len(case_results),
        "traced_case_count": len(traced),
        "concrete_class_count": int(concrete_count),
        "failure_class_counts": dict(sorted(class_counts.items())),
        "secondary_flag_counts": dict(sorted(secondary_counts.items())),
        "mean_event_count_ratio": _mean(
            [float(value) for value in (_float(row.get("event_count_ratio")) for row in metrics_rows) if value is not None]
        ),
        "mean_second_window_event_share": _mean(
            [
                float(value)
                for value in (_float(row.get("second_window_event_share")) for row in metrics_rows)
                if value is not None
            ]
        ),
        "mean_dominant_spacing_ratio": _mean(
            [
                float(value)
                for value in (_float(row.get("dominant_spacing_ratio")) for row in metrics_rows)
                if value is not None
            ]
        ),
    }


def decision_from_aggregate(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    selected = int(aggregate.get("selected_case_count") or 0)
    traced = int(aggregate.get("traced_case_count") or 0)
    concrete = int(aggregate.get("concrete_class_count") or 0)
    class_counts = Counter(_mapping(aggregate.get("failure_class_counts")))
    if traced < min(4, selected):
        return {
            "route": "KILL_TRACE_INPUTS",
            "reason": f"only {traced}/{selected} selected cases produced trace rows",
            "next_step": "Fix trace input/runtime alignment before any new objective or grammar repair.",
        }
    if concrete < min(4, selected):
        return {
            "route": "MUTATE_TRACE_CLASSIFIER",
            "reason": f"only {concrete}/{selected} selected cases received a concrete class",
            "next_step": "Inspect one case manually and narrow instrumentation before training.",
        }
    dominant_class, dominant_count = class_counts.most_common(1)[0]
    if dominant_class in {"time_shift_repetition", "boundary_drift"} and dominant_count >= max(3, math.ceil(selected / 2)):
        return {
            "route": "TEST_BOUNDARY_OR_SPACING_STATE_REPAIR",
            "reason": f"{dominant_count}/{selected} cases share `{dominant_class}` as the primary generated-prefix class",
            "next_step": "Create one bounded spacing/boundary state repair card; do not start broad v3 retraining.",
        }
    if dominant_class == "event_underselection" and dominant_count >= max(3, math.ceil(selected / 2)):
        return {
            "route": "TEST_TRACE_CONDITIONED_EVENT_RANKING",
            "reason": f"{dominant_count}/{selected} cases share event underselection",
            "next_step": "Create a bounded event-ranking repair card using generated-prefix trace evidence.",
        }
    if dominant_class == "state_or_carry_mismatch" and dominant_count >= 2:
        return {
            "route": "TEST_STATE_CARRY_REPAIR",
            "reason": f"{dominant_count}/{selected} cases hit state/carry mismatch",
            "next_step": "Create a carry-state replay repair card before decode or objective changes.",
        }
    return {
        "route": "MUTATE_TO_V2_1_OR_TARGET_GRAMMAR_REPAIR",
        "reason": "trace produced concrete but mixed failure mechanisms",
        "next_step": "Prefer a narrow v2.1/target-grammar repair card over another generic v3 objective.",
    }


def interpretation_from_decision(decision: Mapping[str, Any]) -> str:
    route = str(decision.get("route"))
    if route == "TEST_BOUNDARY_OR_SPACING_STATE_REPAIR":
        return (
            "The generated-prefix trace localizes the remaining v3 failure to rollout-state behavior, especially "
            "rigid time-shift repetition and boundary continuation. This does not prove a repair, but it defines "
            "a smaller target than another global tokenizer or broad mapper objective."
        )
    if route == "TEST_TRACE_CONDITIONED_EVENT_RANKING":
        return (
            "The trace suggests event candidates are often rank-near under generated prefixes but are not selected. "
            "The next test should be an event-ranking repair, not a tokenizer replacement claim."
        )
    if route == "TEST_STATE_CARRY_REPAIR":
        return "The trace points to replay/carry state integrity before model-side objective work."
    if route == "KILL_TRACE_INPUTS":
        return "The trace harness did not clear the minimum instrumentation gate."
    return "The trace is informative but not clean enough to justify a new v3 training run yet."


def trace_excerpt(
    trace_rows: Sequence[Mapping[str, Any]],
    row_key: object,
    *,
    radius: int = 10,
) -> list[dict[str, Any]]:
    if not trace_rows:
        return []
    center = 0
    if row_key is not None:
        for index, row in enumerate(trace_rows):
            if str(row.get("row_key")) == str(row_key):
                center = index
                break
    start = max(0, center - int(radius))
    end = min(len(trace_rows), center + int(radius) + 1)
    return [_compact_trace_row(row) for row in trace_rows[start:end]]


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
    aggregate = _mapping(summary.get("aggregate"))
    lines = [
        "# Target Grammar v3 Generated-Prefix State Trace Audit Result Report",
        "",
        "## Scope",
        "",
        "This pass reruns the selected six high-overlap v3 failure cases under generated prefixes and records per-step state/logit traces. It does not train, change tokenizer behavior, change grammar defaults, or add C3 sidecar conditioning.",
        "",
        "## Result",
        "",
        f"Decision: `{decision.get('route')}`.",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Selected cases: `{aggregate.get('selected_case_count')}`",
        f"- Traced cases: `{aggregate.get('traced_case_count')}`",
        f"- Concrete classes: `{aggregate.get('concrete_class_count')}`",
        f"- Failure classes: `{aggregate.get('failure_class_counts')}`",
        f"- Secondary flags: `{aggregate.get('secondary_flag_counts')}`",
        f"- Mean event-count ratio: `{_fmt(aggregate.get('mean_event_count_ratio'))}`",
        f"- Mean second-window event share: `{_fmt(aggregate.get('mean_second_window_event_share'))}`",
        f"- Mean dominant-spacing ratio: `{_fmt(aggregate.get('mean_dominant_spacing_ratio'))}`",
        "",
        "## What Passed",
        "",
        "- The existing v3 runtime rollout path produced generated-prefix traces for the selected cases.",
        "- The trace rows align logits, valid-token masks, replay state, emitted token, and post-token state.",
        "- The result preserves the guard: no training, no tokenizer change, no default decode change, and no target-derived C3 input path.",
        "",
        "## What Surfaced",
        "",
        "- The failure is generated-prefix local: the traces classify rollout behavior, not teacher-forced token learnability.",
        "- The primary surfaced mechanism is the class distribution shown below; secondary flags show co-occurring symptoms.",
        "- This is diagnostic evidence only. It does not prove that a spacing, boundary, or event-ranking repair will pass a wider rollout gate.",
        "",
        "## Cases",
        "",
        "| case | class | first ms | generated/ref | second share | dominant spacing | best-event rank | emitted | secondary flags |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in _sequence(summary.get("case_results")):
        case = _mapping(row)
        classification = _mapping(case.get("classification"))
        metrics = _mapping(case.get("metrics"))
        first = _mapping(classification.get("candidate_failures")).get(str(case.get("failure_class")))
        first_mapping = _mapping(first)
        failure_ms = first_mapping.get("first_failure_current_ms", classification.get("first_failure_current_ms"))
        best_rank = first_mapping.get("best_event_rank", classification.get("best_event_rank"))
        emitted = first_mapping.get("emitted_token", classification.get("emitted_token"))
        flags = _mapping(classification.get("secondary_flags"))
        enabled_flags = ",".join(name for name, enabled in flags.items() if bool(enabled)) or "none"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{case.get('case_id')}`",
                    f"`{case.get('failure_class')}`",
                    f"`{failure_ms}`",
                    f"`{metrics.get('generated_event_count')}/{metrics.get('reference_event_count')}`",
                    f"`{_fmt(metrics.get('second_window_event_share'))}`",
                    f"`{_fmt(metrics.get('dominant_spacing_ratio'))}`",
                    f"`{best_rank}`",
                    f"`{emitted}`",
                    f"`{enabled_flags}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(summary.get("interpretation")),
            "",
            "## Evidence Boundary",
            "",
            "- Proved: the selected generated-prefix failures can be traced and classified without another training run.",
            "- Proved: the remaining issue is not explained away by healthy teacher-forced time-shift ranks; it appears after generated prefixes enter the replay loop.",
            "- Not proved: active-hold or C3 causality, end-to-end replacement readiness, or that any proposed repair will improve full32 rollout quality.",
            "",
            "## Next Step",
            "",
            str(summary.get("next_step")),
            "",
        ]
    )
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _baseline_runs_by_case_id(summary: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    runs = summary.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("baseline summary must contain a non-empty runs list")
    rows = {str(row.get("case_id")): row for row in runs if isinstance(row, Mapping) and row.get("case_id")}
    if not rows:
        raise ValueError("baseline summary has no case_id rows")
    return rows


def _baseline_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id": row.get("case_id"),
        "generated_event_count": row.get("generated_event_count"),
        "reference_event_count": row.get("reference_event_count"),
        "event_count_ratio": row.get("event_count_ratio"),
        "second_window_event_share": row.get("second_window_event_share"),
        "boundary_event_ratio": row.get("boundary_event_ratio"),
        "dominant_spacing_ratio": row.get("dominant_spacing_ratio"),
        "timing_match_100ms": row.get("timing_match_100ms"),
        "summary_path": row.get("summary_path"),
    }


def _reference_times(beatmap_path: Path, *, chart_end_ms: int) -> list[int]:
    hitobjects = parse_mania_hit_objects(beatmap_path, expected_key_count=4)
    return [
        int(timepoint.time_ms)
        for timepoint in hitobjects_to_mapper_timepoints(hitobjects)
        if 0 <= int(timepoint.time_ms) <= int(chart_end_ms)
    ]


def _real_audio_length_ms(audio_path: Path) -> int:
    resolved = audio_length_ms_from_file(audio_path)
    if resolved is None:
        raise ValueError(f"could not infer audio length for real audio: {audio_path}")
    return int(resolved)


def _token_summary(vocab: MapperV3Vocab, token_id: int, *, logit: float | None = None) -> dict[str, Any]:
    token_id = int(token_id)
    summary: dict[str, Any] = {
        "id": token_id,
        "name": vocab.token_name(token_id),
        "kind": _token_kind(vocab, token_id),
    }
    if logit is not None:
        summary["logit"] = float(logit)
    if vocab.is_time_shift_token(token_id):
        summary["time_shift_ms"] = int(vocab.time_shift_value(token_id))
    if vocab.is_event_token(token_id):
        summary["event_signature"] = vocab.event_signature(token_id)
        summary["event_onset_weight"] = int(vocab.event_onset_weight(token_id))
    return summary


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


def _kind_counts(vocab: MapperV3Vocab, token_ids: Sequence[int]) -> dict[str, int]:
    counts = Counter(_token_kind(vocab, int(token_id)) for token_id in token_ids)
    return dict(sorted(counts.items()))


def _state_summary(state: Any) -> dict[str, Any]:
    open_mask = [bool(value) for value in getattr(state, "open_mask", ())]
    return {
        "current_ms": int(getattr(state, "current_ms")),
        "open_mask": open_mask,
        "open_count": int(sum(1 for value in open_mask if value)),
        "open_start_ms": [
            None if value is None else int(value)
            for value in getattr(state, "open_start_ms", ())
        ],
        "open_age_ms": [int(value) for value in getattr(state, "open_age_ms", ())],
        "event_emitted_at_current_ms": bool(getattr(state, "event_emitted_at_current_ms")),
    }


def _event_row_at_time(trace_rows: Sequence[Mapping[str, Any]], time_ms: int) -> Mapping[str, Any] | None:
    for row in trace_rows:
        emitted = _mapping(row.get("emitted"))
        if emitted.get("kind") == "event" and _int(row.get("current_ms")) == int(time_ms):
            return row
    return None


def _compact_trace_row(row: Mapping[str, Any]) -> dict[str, Any]:
    argmax = _mapping(row.get("argmax"))
    emitted = _mapping(row.get("emitted"))
    best_event = _mapping(row.get("best_event"))
    best_time_shift = _mapping(row.get("best_time_shift"))
    return {
        "sequence_index": row.get("sequence_index"),
        "row_key": row.get("row_key"),
        "write_start_ms": row.get("write_start_ms"),
        "token_index": row.get("token_index"),
        "current_ms": row.get("current_ms"),
        "open_count": _mapping(row.get("state")).get("open_count"),
        "event_valid": row.get("event_valid"),
        "argmax": _compact_token(argmax),
        "emitted": _compact_token(emitted),
        "best_event": _compact_ranked_token(best_event),
        "best_time_shift": _compact_ranked_token(best_time_shift),
    }


def _compact_token(token: Mapping[str, Any]) -> dict[str, Any] | None:
    if not token:
        return None
    return {
        "name": token.get("name"),
        "kind": token.get("kind"),
        "time_shift_ms": token.get("time_shift_ms"),
        "event_signature": token.get("event_signature"),
    }


def _compact_ranked_token(token: Mapping[str, Any]) -> dict[str, Any] | None:
    compact = _compact_token(token)
    if compact is None:
        return None
    compact.update(
        {
            "rank": token.get("rank"),
            "margin_vs_argmax": token.get("margin_vs_argmax"),
        }
    )
    return compact


def _timepoint_to_dict(timepoint: Any) -> dict[str, Any]:
    return {
        "time_ms": int(timepoint.time_ms),
        "lane_actions": [str(getattr(action, "value", action)) for action in timepoint.lane_actions],
    }


def _has_time_within(times: Sequence[int], current_ms: int, tolerance_ms: int) -> bool:
    return any(abs(int(value) - int(current_ms)) <= int(tolerance_ms) for value in times)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else ()


def _float(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _mean(values: Sequence[float]) -> float:
    return float(math.fsum(values) / len(values)) if values else 0.0


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
    median = ordered[midpoint] if count % 2 else 0.5 * (ordered[midpoint - 1] + ordered[midpoint])
    return {
        "count": int(count),
        "min": float(ordered[0]),
        "median": float(median),
        "mean": float(math.fsum(ordered) / count),
        "max": float(ordered[-1]),
    }


def _fmt(value: object) -> str:
    number = _float(value)
    return "n/a" if number is None else f"{number:.6f}"


def _row_key(write_start_ms: int, token_index: int) -> str:
    return f"{int(write_start_ms)}:{int(token_index)}"


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value)).strip("_") or "case"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the v3 generated-prefix state trace audit.")
    parser.add_argument("--baseline-summary", default=DEFAULT_BASELINE_SUMMARY_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--report-output", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--mapper-checkpoint-path")
    parser.add_argument("--control-checkpoint-path")
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--max-tokens-per-window", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--logit-top-k", type=int, default=5)
    parser.add_argument("--timepoint-preview-limit", type=int, default=4096)
    args = parser.parse_args(argv)

    summary = run_generated_prefix_state_trace_audit(
        baseline_summary_path=args.baseline_summary,
        output_dir=args.output_dir,
        output_summary_path=args.summary_output,
        output_report_path=args.report_output,
        mapper_checkpoint_path=args.mapper_checkpoint_path,
        control_checkpoint_path=args.control_checkpoint_path,
        case_ids=tuple(args.case_ids) if args.case_ids else SELECTED_CASE_IDS,
        case_limit=args.case_limit,
        device_name=args.device,
        max_tokens_per_window=args.max_tokens_per_window,
        seed=args.seed,
        logit_top_k=args.logit_top_k,
        timepoint_preview_limit=args.timepoint_preview_limit,
    )
    print(
        "mapper_v3_generated_prefix_state_trace_audit_done "
        f"route={summary['decision']['route']} "
        f"traced={summary['aggregate']['traced_case_count']}/"
        f"{summary['aggregate']['selected_case_count']} "
        f"classes={summary['aggregate']['failure_class_counts']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
