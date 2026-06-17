from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from pulsefield_model.evals.mapper_v2_1_anti_rigid_spacing_guard import generated_metrics
from pulsefield_model.evals.mapper_v2_1_trained_runtime_rollout import run_trained_v21_runtime_rollout_smoke
from pulsefield_model.inference.mapper_v2_1_rollout import (
    MapperV21AntiRigidSpacingLogitsTransform,
    MapperV21GenerationStep,
)
from pulsefield_model.models.mapper.shared.generation_engine import apply_valid_mask
from pulsefield_model.models.mapper.shared.tokenizer import hitobjects_to_mapper_timepoints
from pulsefield_model.models.mapper.v2_1 import MapperV21Vocab
from pulsefield_model.models.mapper.v2_1.grammar import valid_token_mask
from pulsefield_model.models.mapper.v2_1.replay import (
    MapperReplayState,
    ReplayError,
    format_replay_state,
    replay_state_matches_carry,
    transition_replay_state,
)
from pulsefield_model.osu_core.hitobjects import parse_mania_hit_objects


SUMMARY_SCHEMA_VERSION = 1
DEFAULT_COMPARISON_SUMMARY_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_v3_fixed_slice_decode_comparison_summary.json",
)
DEFAULT_OUTPUT_DIR = Path("artifacts/tmp/mapper_v21_illegal_case_trace_audit")
DEFAULT_SUMMARY_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_illegal_case_trace_audit_summary.json",
)
DEFAULT_REPORT_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_illegal_case_trace_audit_result_report.md",
)
PRIMARY_CASE_IDS = (
    "04_oomori_seiko_justadice_tv_size_remu_normal",
    "05_usao_knight_rider_kuo_kyoka_dnm_s_normal",
)
STRESS_CASE_IDS = (
    "29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent",
    "31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial",
)
TRACE_MODES = ("baseline", "guard")
TRACE_TAIL_STEPS = 16
TRACE_TOP_K = 8
TIMING_F1_TOLERANCE_MS = 100


class MapperV21IllegalCaseTraceTransform:
    """Default-off logits transform that records trace data without changing semantics."""

    def __init__(
        self,
        *,
        vocab: MapperV21Vocab,
        mode: str,
        top_k: int = TRACE_TOP_K,
        tail_steps: int = TRACE_TAIL_STEPS,
        min_repeated_spacings: int = 4,
        min_spacing_ms: int = 40,
        max_spacing_ms: int = 400,
    ) -> None:
        if mode not in TRACE_MODES:
            raise ValueError(f"mode must be one of {TRACE_MODES}, got {mode!r}")
        self.vocab = vocab
        self.mode = str(mode)
        self.top_k = int(top_k)
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        self.tail_steps = int(tail_steps)
        if self.tail_steps <= 0:
            raise ValueError("tail_steps must be positive")
        self.inner: MapperV21AntiRigidSpacingLogitsTransform | None = None
        if mode == "guard":
            self.inner = MapperV21AntiRigidSpacingLogitsTransform(
                vocab=vocab,
                min_repeated_spacings=int(min_repeated_spacings),
                min_spacing_ms=int(min_spacing_ms),
                max_spacing_ms=int(max_spacing_ms),
            )
        self.step_count = 0
        self.changed_step_count = 0
        self.valid_kind_counts: Counter[str] = Counter()
        self.selected_kind_counts: Counter[str] = Counter()
        self.raw_invalid_top1_count = 0
        self.tail_records: list[dict[str, Any]] = []
        self.last_state_after: dict[str, Any] | None = None
        self.last_terminal_probe: dict[str, Any] | None = None

    @property
    def blocked_count(self) -> int:
        return 0 if self.inner is None else int(self.inner.blocked_count)

    def __call__(self, step: MapperV21GenerationStep, logits: torch.Tensor) -> torch.Tensor:
        raw_logits = torch.as_tensor(logits, dtype=torch.float32).reshape(-1)
        if self.inner is None:
            transformed = raw_logits.clone()
        else:
            transformed = self.inner(step, raw_logits)
            transformed = torch.as_tensor(transformed, dtype=torch.float32, device=raw_logits.device).reshape(-1)
        record = self._record_step(step, raw_logits=raw_logits, transformed_logits=transformed)
        self.step_count += 1
        for kind, count in record["valid_kind_counts"].items():
            self.valid_kind_counts[str(kind)] += int(count)
        selected = _mapping(record.get("selected_after_transform"))
        selected_kind = selected.get("kind")
        if isinstance(selected_kind, str):
            self.selected_kind_counts[selected_kind] += 1
        if record.get("raw_top1_invalid"):
            self.raw_invalid_top1_count += 1
        if record.get("changed_token_count", 0) > 0:
            self.changed_step_count += 1
        self.tail_records.append(record)
        if len(self.tail_records) > self.tail_steps:
            self.tail_records = self.tail_records[-self.tail_steps :]
        self.last_state_after = _mapping(record.get("state_after")) or None
        self.last_terminal_probe = self._terminal_probe(step, record)
        return transformed

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "mode": self.mode,
            "top_k": int(self.top_k),
            "tail_steps": int(self.tail_steps),
            "step_count": int(self.step_count),
            "changed_step_count": int(self.changed_step_count),
            "blocked_count": int(self.blocked_count),
            "raw_invalid_top1_count": int(self.raw_invalid_top1_count),
            "valid_kind_counts": dict(sorted(self.valid_kind_counts.items())),
            "selected_kind_counts": dict(sorted(self.selected_kind_counts.items())),
            "anti_rigid": {"enabled": self.inner is not None}
            if self.inner is None
            else self.inner.to_dict(),
            "tail_records": list(self.tail_records),
            "last_state_after": self.last_state_after,
            "last_terminal_probe": self.last_terminal_probe,
        }

    def _record_step(
        self,
        step: MapperV21GenerationStep,
        *,
        raw_logits: torch.Tensor,
        transformed_logits: torch.Tensor,
    ) -> dict[str, Any]:
        valid_mask = step.valid_token_mask.to(device=raw_logits.device, dtype=torch.bool).reshape(-1)
        if int(raw_logits.numel()) != int(valid_mask.numel()):
            raise ValueError(f"logits must contain {valid_mask.numel()} values, got {raw_logits.numel()}")
        if int(transformed_logits.numel()) != int(valid_mask.numel()):
            raise ValueError(
                f"transformed logits must contain {valid_mask.numel()} values, got {transformed_logits.numel()}",
            )

        changed_ids = _changed_token_ids(raw_logits, transformed_logits)
        selected_id = _selected_token_id(transformed_logits, valid_mask)
        state_after = None
        transition_error = None
        if selected_id is not None:
            try:
                state_after = transition_replay_state(
                    step.state,
                    int(selected_id),
                    position=int(step.token_index),
                    vocab=self.vocab,
                    write_start_ms=int(step.write_start_ms),
                    write_end_ms=int(step.write_end_ms),
                    chart_end_ms=int(step.chart_end_ms),
                    ln_carry_out=step.ln_carry_out,
                    is_full_chart_start=bool(step.is_full_chart_start),
                    is_full_chart_end=bool(step.is_full_chart_end),
                )
            except (ReplayError, ValueError) as exc:
                transition_error = str(exc)

        changed_tokens = [
            {
                "token_id": int(token_id),
                "token_name": self.vocab.token_name(int(token_id)),
                "kind": _token_kind(self.vocab, int(token_id)),
                "raw_logit": _float_or_none(raw_logits[int(token_id)]),
                "transformed_logit": _float_or_none(transformed_logits[int(token_id)]),
                "raw_valid_rank": _rank_among_valid(raw_logits, valid_mask, int(token_id)),
                "transformed_valid_rank": _rank_among_valid(transformed_logits, valid_mask, int(token_id)),
            }
            for token_id in changed_ids
        ]
        valid_time_shift_ids = [
            int(token_id)
            for token_id in self.vocab.time_shift_token_ids
            if 0 <= int(token_id) < int(valid_mask.numel()) and bool(valid_mask[int(token_id)].item())
        ]
        raw_top = _top_token_records(self.vocab, raw_logits, top_k=self.top_k)
        record = {
            "token_index": int(step.token_index),
            "generated_token_count": len(step.generated_tokens),
            "current_ms": int(step.state.current_ms),
            "write_start_ms": int(step.write_start_ms),
            "write_end_ms": int(step.write_end_ms),
            "chart_end_ms": int(step.chart_end_ms),
            "state": _state_to_dict(step.state),
            "ln_carry_in": step.ln_carry_in.to_dict(),
            "ln_carry_out": step.ln_carry_out.to_dict(),
            "valid_token_count": int(valid_mask.sum().item()),
            "valid_kind_counts": _valid_kind_counts(self.vocab, valid_mask),
            "valid_time_shift_count": len(valid_time_shift_ids),
            "valid_lane_action_count": sum(
                1
                for token_id in self.vocab.lane_action_token_ids
                if 0 <= int(token_id) < int(valid_mask.numel()) and bool(valid_mask[int(token_id)].item())
            ),
            "raw_top_tokens": raw_top,
            "raw_top1_invalid": bool(raw_top and not bool(valid_mask[int(raw_top[0]["token_id"])].item())),
            "top_valid_raw_tokens": _top_token_records(self.vocab, raw_logits, valid_mask=valid_mask, top_k=self.top_k),
            "top_valid_transformed_tokens": _top_token_records(
                self.vocab,
                transformed_logits,
                valid_mask=valid_mask,
                top_k=self.top_k,
            ),
            "selected_after_transform": None
            if selected_id is None
            else {
                "token_id": int(selected_id),
                "token_name": self.vocab.token_name(int(selected_id)),
                "kind": _token_kind(self.vocab, int(selected_id)),
            },
            "state_after": None if state_after is None else _state_to_dict(state_after),
            "transition_error": transition_error,
            "changed_token_count": len(changed_tokens),
            "changed_tokens": changed_tokens,
            "suppressed_only_valid_time_shift": bool(
                len(changed_ids) == 1 and int(changed_ids[0]) in valid_time_shift_ids and len(valid_time_shift_ids) == 1
            ),
        }
        return record

    def _terminal_probe(self, step: MapperV21GenerationStep, record: Mapping[str, Any]) -> dict[str, Any] | None:
        selected = _mapping(record.get("selected_after_transform"))
        state_after_raw = _mapping(record.get("state_after"))
        if not selected or not state_after_raw:
            return None
        state_after = _state_from_dict(state_after_raw)
        next_position = int(step.token_index) + 1
        next_mask = valid_token_mask(
            position=next_position,
            current_ms=int(state_after.current_ms),
            open_mask=state_after.open_mask,
            open_start_ms=state_after.open_start_ms,
            open_age_ms=state_after.open_age_ms,
            emitted_lane_mask=state_after.emitted_lane_mask,
            last_lane_index=int(state_after.last_lane_index),
            write_start_ms=int(step.write_start_ms),
            write_end_ms=int(step.write_end_ms),
            chart_end_ms=int(step.chart_end_ms),
            ln_carry_in=step.ln_carry_in,
            ln_carry_out=step.ln_carry_out,
            is_full_chart_start=bool(step.is_full_chart_start),
            is_full_chart_end=bool(step.is_full_chart_end),
            vocab=self.vocab,
        ).to(dtype=torch.bool)
        for token_id in (self.vocab.bos_id, self.vocab.eos_id):
            if 0 <= int(token_id) < int(next_mask.numel()):
                next_mask[int(token_id)] = False
        return {
            "next_position": int(next_position),
            "state": _state_to_dict(state_after),
            "matches_carry_out": replay_state_matches_carry(state_after, step.ln_carry_out),
            "valid_token_count": int(next_mask.sum().item()),
            "valid_kind_counts": _valid_kind_counts(self.vocab, next_mask),
            "invalid_reason_counts": _invalid_reason_counts(
                self.vocab,
                state_after,
                position=next_position,
                write_start_ms=int(step.write_start_ms),
                write_end_ms=int(step.write_end_ms),
                chart_end_ms=int(step.chart_end_ms),
                ln_carry_out=step.ln_carry_out,
                is_full_chart_start=bool(step.is_full_chart_start),
                is_full_chart_end=bool(step.is_full_chart_end),
            ),
        }


def run_v21_illegal_case_trace_audit(
    *,
    comparison_summary_path: str | Path = DEFAULT_COMPARISON_SUMMARY_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    output_summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    output_report_path: str | Path | None = DEFAULT_REPORT_PATH,
    case_ids: Sequence[str] = PRIMARY_CASE_IDS,
    include_stress: bool = False,
    device_name: str = "auto",
    max_tokens_per_window: int = 512,
    seed: int = 1337,
    timepoint_preview_limit: int = 4096,
) -> dict[str, Any]:
    comparison = _read_json(Path(comparison_summary_path))
    mapper_checkpoint = Path(str(comparison.get("mapper_checkpoint_path") or ""))
    control_checkpoint = Path(str(comparison.get("control_checkpoint_path") or ""))
    if not mapper_checkpoint.exists():
        raise FileNotFoundError(f"missing mapper checkpoint: {mapper_checkpoint}")
    if not control_checkpoint.exists():
        raise FileNotFoundError(f"missing control checkpoint: {control_checkpoint}")

    requested_case_ids = list(case_ids)
    if bool(include_stress):
        requested_case_ids.extend(case_id for case_id in STRESS_CASE_IDS if case_id not in requested_case_ids)
    case_rows = _case_rows_by_id(comparison)
    missing = [case_id for case_id in requested_case_ids if case_id not in case_rows]
    if missing:
        raise ValueError(f"case ids not found in comparison summary: {missing}")

    out_dir = Path(output_dir)
    rollout_dir = out_dir / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    vocab = MapperV21Vocab()
    run_results: list[dict[str, Any]] = []

    for case_id in requested_case_ids:
        case = case_rows[case_id]
        for mode in TRACE_MODES:
            trace = MapperV21IllegalCaseTraceTransform(vocab=vocab, mode=mode)
            output_case_summary = rollout_dir / f"{_safe_filename(case_id)}_16000ms_{mode}_trace.json"
            raw_summary = run_trained_v21_runtime_rollout_smoke(
                mapper_checkpoint_path=mapper_checkpoint,
                control_checkpoint_path=control_checkpoint,
                output_summary_path=output_case_summary,
                output_report_path=None,
                device_name=device_name,
                chart_end_ms=int(case["chart_end_ms"]),
                max_tokens_per_window=int(max_tokens_per_window),
                audio_path=str(case["audio_path"]),
                normalized_difficulty=float(case["normalized_difficulty"]),
                include_control_attention_kv_cache=False,
                temperature=0.0,
                top_p=None,
                seed=int(seed),
                real_audio=True,
                time_shift_length_penalty_alpha=0.0,
                time_shift_delta_penalty_alpha=0.0,
                collect_logit_diagnostics=False,
                logits_transform=trace,
                timepoint_preview_limit=int(timepoint_preview_limit),
            )
            trace_payload = trace.to_dict()
            reference_times = _reference_times(Path(str(case["beatmap_path"])), chart_end_ms=int(case["chart_end_ms"]))
            generated_times = _generated_times(raw_summary)
            metrics = generated_metrics(
                generated_times=generated_times,
                reference_times=reference_times,
                chart_end_ms=int(case["chart_end_ms"]),
            )
            expected = _expected_from_previous_summary(case, mode=mode)
            classification = classify_trace_result(summary=raw_summary, trace=trace_payload)
            case_summary = {
                **raw_summary,
                "experiment": "Mapper v2.1 illegal-case trace audit case",
                "case": {
                    "case_id": case_id,
                    "mode": mode,
                    "difficulty": case.get("difficulty"),
                    "normalized_difficulty": case.get("normalized_difficulty"),
                    "audio_path": case.get("audio_path"),
                    "beatmap_path": case.get("beatmap_path"),
                    "previous_summary_path": expected.get("summary_path"),
                },
                "trace": trace_payload,
                "metrics": metrics,
                "expected": expected,
                "trace_classification": classification,
            }
            legal = _case_legal(case_summary)
            write_summary_json(case_summary, output_case_summary)
            illegal_trace_tail = [] if legal else list(trace_payload.get("tail_records") or [])
            run_results.append(
                {
                    "case_id": case_id,
                    "mode": mode,
                    "summary_path": output_case_summary.as_posix(),
                    "legal": legal,
                    "completed": bool(_mapping(case_summary.get("rollout")).get("completed")),
                    "dead_end": bool(_mapping(case_summary.get("rollout")).get("dead_end")),
                    "max_tokens_exceeded": bool(_mapping(case_summary.get("rollout")).get("max_tokens_exceeded")),
                    "terminal_ms": _terminal_ms(case_summary),
                    "token_count": int(_mapping(case_summary.get("rollout")).get("token_count") or 0),
                    "timepoint_count": int(_mapping(case_summary.get("rollout")).get("timepoint_count") or 0),
                    "expected_legal": expected.get("legal"),
                    "reproduced_expected_legal": expected.get("legal") is None
                    or bool(expected.get("legal")) == legal,
                    "blocked_count": int(trace_payload.get("blocked_count") or 0),
                    "changed_step_count": int(trace_payload.get("changed_step_count") or 0),
                    "raw_invalid_top1_count": int(trace_payload.get("raw_invalid_top1_count") or 0),
                    "anti_rigid_examples": list(_mapping(trace_payload.get("anti_rigid")).get("examples") or []),
                    "terminal_probe": _terminal_probe_summary(trace_payload),
                    "illegal_trace_tail_records": illegal_trace_tail,
                    "classification": classification,
                    "metrics": {
                        "f1_100ms": _float_or_none(_mapping(metrics.get("timing_match_100ms")).get("f1")),
                        "dominant_spacing_ratio": _float_or_none(metrics.get("dominant_spacing_ratio")),
                        "second_window_event_share": _float_or_none(metrics.get("second_window_event_share")),
                        "event_count_ratio": _float_or_none(metrics.get("event_count_ratio")),
                        "starved": bool(metrics.get("starved")),
                    },
                }
            )

    aggregate = aggregate_trace_results(run_results)
    decision = decision_from_trace_aggregate(aggregate)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Mapper v2.1 illegal-case trace audit",
        "experiment_card": "mapper_v21_illegal_case_trace_audit_experiment_card.md",
        "comparison_summary_path": Path(comparison_summary_path).as_posix(),
        "output_dir": out_dir.as_posix(),
        "mapper_checkpoint_path": mapper_checkpoint.as_posix(),
        "control_checkpoint_path": control_checkpoint.as_posix(),
        "config": {
            "device": device_name,
            "max_tokens_per_window": int(max_tokens_per_window),
            "seed": int(seed),
            "timepoint_preview_limit": int(timepoint_preview_limit),
            "case_ids": list(requested_case_ids),
            "modes": list(TRACE_MODES),
            "include_stress": bool(include_stress),
        },
        "decision": decision,
        "aggregate": aggregate,
        "run_results": run_results,
        "next_step": decision["next_step"],
    }
    write_summary_json(summary, output_summary_path)
    if output_report_path is not None:
        write_report(summary, output_report_path)
    return summary


def classify_trace_result(*, summary: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, Any]:
    legal = _case_legal(summary)
    rollout = _mapping(summary.get("rollout"))
    terminal_probe = _mapping(trace.get("last_terminal_probe"))
    invalid_reasons = _mapping(terminal_probe.get("invalid_reason_counts"))
    reason_counts = {str(key): int(value) for key, value in invalid_reasons.items()}
    top_reason = max(reason_counts.items(), key=lambda item: item[1])[0] if reason_counts else None
    blocked_count = int(trace.get("blocked_count") or 0)
    changed_step_count = int(trace.get("changed_step_count") or 0)
    terminal_valid_count = int(terminal_probe.get("valid_token_count") or 0)
    tail_records = trace.get("tail_records") if isinstance(trace.get("tail_records"), list) else []
    recent_changed = any(int(_mapping(record).get("changed_token_count") or 0) > 0 for record in tail_records[-4:])

    if legal:
        family = "legal_contrast"
    elif bool(rollout.get("max_tokens_exceeded")):
        family = "max_token_limit"
    elif blocked_count > 0 and changed_step_count > 0 and recent_changed:
        family = "anti_rigid_removed_recent_transition"
    elif blocked_count > 0 and changed_step_count > 0:
        family = "anti_rigid_path_damage"
    elif terminal_valid_count == 0 and top_reason and "TIME_SHIFT to write_end_ms" in top_reason:
        family = "terminal_time_shift_carry_mismatch"
    elif terminal_valid_count == 0 and top_reason and "same-time lane-action" in top_reason:
        family = "same_time_lane_ordering_or_duplicate"
    elif terminal_valid_count == 0:
        family = "zero_valid_terminal_state"
    elif int(trace.get("raw_invalid_top1_count") or 0) > 0:
        family = "model_prefers_invalid_tokens"
    else:
        family = "unclassified_dead_end"

    return {
        "family": family,
        "legal": bool(legal),
        "completed": bool(rollout.get("completed")),
        "dead_end": bool(rollout.get("dead_end")),
        "max_tokens_exceeded": bool(rollout.get("max_tokens_exceeded")),
        "terminal_valid_count": terminal_valid_count,
        "top_invalid_reason": top_reason,
        "blocked_count": blocked_count,
        "changed_step_count": changed_step_count,
        "recent_changed": bool(recent_changed),
    }


def aggregate_trace_results(run_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    primary_illegal_targets = {
        ("04_oomori_seiko_justadice_tv_size_remu_normal", "baseline"),
        ("05_usao_knight_rider_kuo_kyoka_dnm_s_normal", "guard"),
    }
    reproduced = [
        row
        for row in run_results
        if (str(row.get("case_id")), str(row.get("mode"))) in primary_illegal_targets
        and bool(row.get("reproduced_expected_legal"))
        and not bool(row.get("legal"))
    ]
    classified_illegal = [
        row
        for row in run_results
        if not bool(row.get("legal")) and str(_mapping(row.get("classification")).get("family")) != "unclassified_dead_end"
    ]
    families = Counter(str(_mapping(row.get("classification")).get("family")) for row in run_results)
    return {
        "run_count": int(len(run_results)),
        "legal_count": sum(1 for row in run_results if bool(row.get("legal"))),
        "illegal_count": sum(1 for row in run_results if not bool(row.get("legal"))),
        "expected_reproduction_count": sum(1 for row in run_results if bool(row.get("reproduced_expected_legal"))),
        "primary_illegal_reproduced_count": int(len(reproduced)),
        "primary_illegal_target_count": len(primary_illegal_targets),
        "classified_illegal_count": int(len(classified_illegal)),
        "classification_families": dict(sorted(families.items())),
        "blocked_count": sum(int(row.get("blocked_count") or 0) for row in run_results),
        "changed_step_count": sum(int(row.get("changed_step_count") or 0) for row in run_results),
        "raw_invalid_top1_count": sum(int(row.get("raw_invalid_top1_count") or 0) for row in run_results),
    }


def decision_from_trace_aggregate(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    primary_reproduced = int(aggregate.get("primary_illegal_reproduced_count") or 0)
    primary_total = int(aggregate.get("primary_illegal_target_count") or 0)
    classified = int(aggregate.get("classified_illegal_count") or 0)
    illegal_count = int(aggregate.get("illegal_count") or 0)
    if primary_reproduced < primary_total:
        route = "MUTATE"
        reason = "one or more primary illegal cases did not reproduce"
        next_step = "Check checkpoint/config drift before designing a grammar repair."
    elif classified < illegal_count:
        route = "MUTATE"
        reason = "at least one illegal case remained unclassified"
        next_step = "Add the single missing trace dimension before changing grammar behavior."
    else:
        route = "TEST_NEXT"
        reason = "primary illegal cases reproduced and received concrete trace classifications"
        next_step = "Create a legality-first v2.1 grammar repair card targeted to the observed failure family."
    return {
        "route": route,
        "reason": reason,
        "next_step": next_step,
        "primary_illegal_reproduced_count": primary_reproduced,
        "primary_illegal_target_count": primary_total,
        "classified_illegal_count": classified,
        "illegal_count": illegal_count,
    }


def write_summary_json(summary: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(summary: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_markdown(summary), encoding="utf-8")


def report_markdown(summary: Mapping[str, Any]) -> str:
    decision = _mapping(summary.get("decision"))
    aggregate = _mapping(summary.get("aggregate"))
    lines = [
        "# Mapper v2.1 Illegal-Case Trace Audit Result Report",
        "",
        "## Scope",
        "",
        "This pass executes `mapper_v21_illegal_case_trace_audit_experiment_card.md`. "
        "It reruns selected v2.1 fixed-slice cases with a default-off logits trace transform. "
        "It does not change mapper defaults, tokenizer, grammar, replay, model weights, or training.",
        "",
        "## Decision",
        "",
        f"Decision: `{decision.get('route')}`.",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Runs: `{aggregate.get('run_count')}`",
        f"- Illegal runs: `{aggregate.get('illegal_count')}`",
        f"- Primary illegal reproduced: `{aggregate.get('primary_illegal_reproduced_count')}` / `{aggregate.get('primary_illegal_target_count')}`",
        f"- Classified illegal runs: `{aggregate.get('classified_illegal_count')}`",
        f"- Classification families: `{aggregate.get('classification_families')}`",
        f"- Anti-rigid blocked count: `{aggregate.get('blocked_count')}`",
        f"- Changed trace steps: `{aggregate.get('changed_step_count')}`",
        "",
        "## Case Table",
        "",
        "| Case | Mode | Legal | Expected legal | Terminal ms | Tokens | Timepoints | Blocks | Family | Top invalid reason |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in summary.get("run_results", []):
        if not isinstance(row, Mapping):
            continue
        classification = _mapping(row.get("classification"))
        lines.append(
            "| {case} | `{mode}` | `{legal}` | `{expected}` | `{terminal}` | `{tokens}` | `{timepoints}` | `{blocks}` | `{family}` | {reason} |".format(
                case=f"`{row.get('case_id')}`",
                mode=row.get("mode"),
                legal=row.get("legal"),
                expected=row.get("expected_legal"),
                terminal=row.get("terminal_ms"),
                tokens=row.get("token_count"),
                timepoints=row.get("timepoint_count"),
                blocks=row.get("blocked_count"),
                family=classification.get("family"),
                reason=f"`{classification.get('top_invalid_reason')}`",
            )
        )
    lines.extend(["", "## Illegal Trace Details", ""])
    for row in summary.get("run_results", []):
        if not isinstance(row, Mapping) or bool(row.get("legal")):
            continue
        classification = _mapping(row.get("classification"))
        terminal = _mapping(row.get("terminal_probe"))
        terminal_state = _mapping(terminal.get("state"))
        examples = row.get("anti_rigid_examples") if isinstance(row.get("anti_rigid_examples"), list) else []
        lines.extend(
            [
                f"### `{row.get('case_id')}` / `{row.get('mode')}`",
                "",
                f"- Family: `{classification.get('family')}`",
                f"- Terminal valid tokens: `{terminal.get('valid_token_count')}`",
                f"- Terminal matches carry-out: `{terminal.get('matches_carry_out')}`",
                f"- Terminal state current_ms: `{terminal_state.get('current_ms')}`",
                f"- Terminal open_mask: `{terminal_state.get('open_mask')}`",
                f"- Terminal open_start_ms: `{terminal_state.get('open_start_ms')}`",
                f"- Terminal emitted_lane_mask: `{terminal_state.get('emitted_lane_mask')}`",
                f"- Terminal invalid reasons: `{terminal.get('invalid_reason_counts')}`",
                f"- Anti-rigid examples: `{examples}`",
                "",
            ]
        )
    lines.extend(
        [
            "",
            "## What Passed",
            "",
            "- The audit used the existing runtime rollout path and recorded generated-prefix trace state.",
            "- Baseline mode uses an identity trace transform; guard mode delegates to the existing anti-rigid transform.",
            "- Per-case JSON artifacts include tail valid-mask/logit records and terminal invalid-reason counts.",
            "",
            "## Interpretation",
            "",
            _interpretation(summary),
            "",
            "## Next Step",
            "",
            str(summary.get("next_step")),
            "",
        ]
    )
    return "\n".join(lines)


def _interpretation(summary: Mapping[str, Any]) -> str:
    decision = _mapping(summary.get("decision"))
    if decision.get("route") == "TEST_NEXT":
        return (
            "The trace audit reproduced the target illegal cases and converted the aggregate legality failure "
            "into concrete failure families. The next loop can be a single legality-first grammar repair card."
        )
    return (
        "The trace audit did not yet provide enough stable evidence for a grammar repair. Do not mutate v2.1 "
        "grammar until the reproduction or missing trace field is fixed."
    )


def _case_rows_by_id(summary: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = summary.get("case_results")
    if not isinstance(rows, list) or not rows:
        raise ValueError("comparison summary must contain non-empty case_results")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        case_id = row.get("case_id")
        if isinstance(case_id, str):
            result[case_id] = row
    return result


def _expected_from_previous_summary(case: Mapping[str, Any], *, mode: str) -> dict[str, Any]:
    path_value = case.get("v21_guard_summary_path" if mode == "guard" else "v21_baseline_summary_path")
    if not isinstance(path_value, str):
        return {"summary_path": None, "legal": None}
    path = Path(path_value)
    if not path.exists():
        return {"summary_path": path.as_posix(), "legal": None}
    summary = _read_json(path)
    return {
        "summary_path": path.as_posix(),
        "legal": _case_legal(summary),
        "completed": bool(_mapping(summary.get("rollout")).get("completed")),
        "dead_end": bool(_mapping(summary.get("rollout")).get("dead_end")),
        "max_tokens_exceeded": bool(_mapping(summary.get("rollout")).get("max_tokens_exceeded")),
    }


def _case_legal(summary: Mapping[str, Any]) -> bool:
    rollout = _mapping(summary.get("rollout"))
    return bool(rollout.get("completed")) and not bool(rollout.get("dead_end")) and not bool(
        rollout.get("max_tokens_exceeded"),
    )


def _terminal_probe_summary(trace_payload: Mapping[str, Any]) -> dict[str, Any]:
    probe = _mapping(trace_payload.get("last_terminal_probe"))
    state = _mapping(probe.get("state"))
    return {
        "valid_token_count": int(probe.get("valid_token_count") or 0),
        "valid_kind_counts": dict(_mapping(probe.get("valid_kind_counts"))),
        "matches_carry_out": bool(probe.get("matches_carry_out")),
        "invalid_reason_counts": dict(_mapping(probe.get("invalid_reason_counts"))),
        "state": {
            "current_ms": state.get("current_ms"),
            "open_mask": state.get("open_mask"),
            "open_start_ms": state.get("open_start_ms"),
            "open_age_ms": state.get("open_age_ms"),
            "emitted_lane_mask": state.get("emitted_lane_mask"),
            "last_lane_index": state.get("last_lane_index"),
        },
    }


def _terminal_ms(summary: Mapping[str, Any]) -> int | None:
    windows = _mapping(summary.get("rollout")).get("windows")
    if not isinstance(windows, list) or not windows:
        return None
    last = windows[-1]
    return int(_mapping(last).get("terminal_ms") or 0)


def _generated_times(summary: Mapping[str, Any]) -> list[int]:
    rollout = _mapping(summary.get("rollout"))
    timepoints = rollout.get("timepoints")
    if not isinstance(timepoints, list):
        return []
    return [int(_mapping(timepoint).get("time_ms")) for timepoint in timepoints if "time_ms" in _mapping(timepoint)]


def _reference_times(beatmap_path: Path, *, chart_end_ms: int) -> list[int]:
    hitobjects = parse_mania_hit_objects(beatmap_path, expected_key_count=4)
    return [
        int(timepoint.time_ms)
        for timepoint in hitobjects_to_mapper_timepoints(hitobjects)
        if 0 <= int(timepoint.time_ms) <= int(chart_end_ms)
    ]


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


def _valid_kind_counts(vocab: MapperV21Vocab, valid_mask: torch.Tensor) -> dict[str, int]:
    mask = valid_mask.to(dtype=torch.bool).reshape(-1)
    counts: Counter[str] = Counter()
    for token_id in torch.nonzero(mask, as_tuple=False).reshape(-1).tolist():
        counts[_token_kind(vocab, int(token_id))] += 1
    return dict(sorted(counts.items()))


def _top_token_records(
    vocab: MapperV21Vocab,
    logits: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    top_k: int,
) -> list[dict[str, Any]]:
    values = torch.as_tensor(logits, dtype=torch.float32).reshape(-1)
    if valid_mask is not None:
        mask = valid_mask.to(device=values.device, dtype=torch.bool).reshape(-1)
        values = values.masked_fill(~mask, -torch.inf)
    finite_or_inf = values
    k = min(int(top_k), int(finite_or_inf.numel()))
    if k <= 0:
        return []
    top = torch.topk(finite_or_inf, k=k)
    records: list[dict[str, Any]] = []
    for rank, (token_id, value) in enumerate(zip(top.indices.tolist(), top.values.tolist(), strict=True), start=1):
        records.append(
            {
                "rank": int(rank),
                "token_id": int(token_id),
                "token_name": vocab.token_name(int(token_id)),
                "kind": _token_kind(vocab, int(token_id)),
                "logit": _float_or_none(value),
            }
        )
    return records


def _selected_token_id(logits: torch.Tensor, valid_mask: torch.Tensor) -> int | None:
    masked = apply_valid_mask(logits, valid_mask)
    if not bool(torch.isfinite(masked).any().item()):
        return None
    return int(torch.argmax(masked).item())


def _changed_token_ids(raw_logits: torch.Tensor, transformed_logits: torch.Tensor) -> list[int]:
    raw = torch.as_tensor(raw_logits, dtype=torch.float32).reshape(-1)
    transformed = torch.as_tensor(transformed_logits, dtype=torch.float32).reshape(-1)
    if int(raw.numel()) != int(transformed.numel()):
        raise ValueError("raw and transformed logits must have the same size")
    changed = raw != transformed
    return [int(token_id) for token_id in torch.nonzero(changed, as_tuple=False).reshape(-1).tolist()]


def _rank_among_valid(logits: torch.Tensor, valid_mask: torch.Tensor, token_id: int) -> int | None:
    token = int(token_id)
    mask = valid_mask.to(device=logits.device, dtype=torch.bool).reshape(-1)
    if not 0 <= token < int(mask.numel()) or not bool(mask[token].item()):
        return None
    values = torch.as_tensor(logits, dtype=torch.float32).reshape(-1)
    value = values[token]
    if not bool(torch.isfinite(value).item()):
        return None
    valid_values = values[mask]
    return int(torch.sum(valid_values > value).item()) + 1


def _invalid_reason_counts(
    vocab: MapperV21Vocab,
    state: MapperReplayState,
    *,
    position: int,
    write_start_ms: int,
    write_end_ms: int,
    chart_end_ms: int,
    ln_carry_out: Any,
    is_full_chart_start: bool,
    is_full_chart_end: bool,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for token_id in range(vocab.size):
        if token_id in {vocab.pad_id, vocab.bos_id}:
            continue
        try:
            transition_replay_state(
                state,
                token_id,
                position=int(position),
                vocab=vocab,
                write_start_ms=int(write_start_ms),
                write_end_ms=int(write_end_ms),
                chart_end_ms=int(chart_end_ms),
                ln_carry_out=ln_carry_out,
                is_full_chart_start=bool(is_full_chart_start),
                is_full_chart_end=bool(is_full_chart_end),
            )
        except (ReplayError, ValueError) as exc:
            counts[_normalize_reason(str(exc))] += 1
        else:
            counts["legal"] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _normalize_reason(reason: str) -> str:
    if reason.startswith("TIME_SHIFT to write_end_ms"):
        return "TIME_SHIFT to write_end_ms requires resulting state to equal ln_carry_out"
    if reason.startswith("TIME_SHIFT moves past target_end_ms"):
        return "TIME_SHIFT moves past target_end_ms"
    if reason.startswith("same-time lane-action"):
        return "same-time lane-action ordering/duplicate violation"
    if "is illegal on open lane" in reason:
        return "lane action illegal on open lane"
    if "HOLD_END is illegal on closed lane" in reason:
        return "HOLD_END illegal on closed lane"
    if reason.startswith("lane-action token is illegal after the target end"):
        return "lane-action token illegal after target end"
    if reason.startswith("EOS is legal only"):
        return "EOS not legal in this window"
    if reason.startswith("EOS requires"):
        return "EOS terminal-state requirement failed"
    return reason


def _state_to_dict(state: MapperReplayState) -> dict[str, Any]:
    return {
        "position": int(state.position),
        "current_ms": int(state.current_ms),
        "open_mask": [bool(value) for value in state.open_mask],
        "open_start_ms": [None if value is None else int(value) for value in state.open_start_ms],
        "open_age_ms": [int(value) for value in state.open_age_ms],
        "emitted_lane_mask": [bool(value) for value in state.emitted_lane_mask],
        "last_lane_index": int(state.last_lane_index),
        "repr": format_replay_state(state),
    }


def _state_from_dict(value: Mapping[str, Any]) -> MapperReplayState:
    starts = value.get("open_start_ms")
    if not isinstance(starts, list):
        starts = [None, None, None, None]
    return MapperReplayState(
        position=int(value.get("position") or 0),
        current_ms=int(value.get("current_ms") or 0),
        open_mask=tuple(bool(item) for item in value.get("open_mask", [False, False, False, False])),  # type: ignore[arg-type]
        open_start_ms=tuple(None if item is None else int(item) for item in starts),  # type: ignore[arg-type]
        open_age_ms=tuple(int(item) for item in value.get("open_age_ms", [0, 0, 0, 0])),  # type: ignore[arg-type]
        emitted_lane_mask=tuple(bool(item) for item in value.get("emitted_lane_mask", [False, False, False, False])),  # type: ignore[arg-type]
        last_lane_index=int(value.get("last_lane_index", -1)),
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _float_or_none(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value)).strip("_") or "case"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Trace v2.1 fixed-slice illegal rollout cases.")
    parser.add_argument("--comparison-summary", default=DEFAULT_COMPARISON_SUMMARY_PATH)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--report-output", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--include-stress", action="store_true")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--max-tokens-per-window", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--timepoint-preview-limit", type=int, default=4096)
    args = parser.parse_args(argv)
    summary = run_v21_illegal_case_trace_audit(
        comparison_summary_path=args.comparison_summary,
        output_dir=args.output_dir,
        output_summary_path=args.summary_output,
        output_report_path=args.report_output,
        case_ids=tuple(args.case_ids) if args.case_ids else PRIMARY_CASE_IDS,
        include_stress=bool(args.include_stress),
        device_name=args.device,
        max_tokens_per_window=int(args.max_tokens_per_window),
        seed=int(args.seed),
        timepoint_preview_limit=int(args.timepoint_preview_limit),
    )
    print(
        "mapper_v21_illegal_case_trace_audit_done "
        f"route={summary['decision']['route']} "
        f"runs={summary['aggregate']['run_count']} "
        f"illegal={summary['aggregate']['illegal_count']} "
        f"classified={summary['aggregate']['classified_illegal_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
