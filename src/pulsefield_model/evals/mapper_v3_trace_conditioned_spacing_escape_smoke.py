from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from pulsefield_model.evals.mapper_v2_1_anti_rigid_spacing_guard import generated_metrics
from pulsefield_model.evals.mapper_v3_trained_runtime_rollout import run_trained_v3_runtime_rollout_smoke
from pulsefield_model.inference.mapper_v3_rollout import MapperV3GenerationStep
from pulsefield_model.models.mapper.shared.tokenizer import hitobjects_to_mapper_timepoints
from pulsefield_model.models.mapper.v3 import MapperV3Vocab
from pulsefield_model.models.mapper.v3.replay import initial_replay_state, transition_replay_state
from pulsefield_model.osu_core.hitobjects import parse_mania_hit_objects


SUMMARY_SCHEMA_VERSION = 1
DEFAULT_BASELINE_SUMMARY_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_fixed_slice_wide_audit_summary.json",
)
DEFAULT_OUTPUT_DIR = Path("artifacts/tmp/mapper_v3_trace_conditioned_spacing_escape")
DEFAULT_SUMMARY_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/"
    "target_grammar_v3_trace_conditioned_spacing_escape_summary.json",
)
DEFAULT_REPORT_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/"
    "target_grammar_v3_trace_conditioned_spacing_escape_result_report.md",
)
DEFAULT_EXPERIMENT_CARD_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/"
    "target_grammar_v3_trace_conditioned_spacing_escape_experiment_card.md",
)

PRIMARY_CASE_IDS = (
    "14_oomori_seiko_justadice_tv_size_remu_hard",
    "17_tamura_yukari_fantastic_future_tv_size_victorica_db_4k_mx",
    "19_nekodex_circles_famoss_hard",
    "22_oomori_seiko_justadice_tv_size_remu_grimoire_of_despair",
    "31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial",
    "18_billiummoto_four_veiled_stars_aries_famoss_hard",
)
SENTINEL_CASE_IDS = (
    "23_usao_knight_rider_kuo_kyoka_expert",
    "29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent",
    "28_usao_knight_rider_kuo_kyoka_cs_lone_sonorous",
    "01_hatsuki_yura_guren_yasha_a_m_d_normal",
)

SECOND_WINDOW_START_MS = 8_000
RIGID_IMPROVEMENT_DELTA = -0.15
RIGID_TARGET_RATIO = 0.75
SECOND_WINDOW_IMPROVEMENT_DELTA = 0.10
SECOND_WINDOW_TARGET_SHARE = 0.15
EVENT_RATIO_MAX = 1.25
MEAN_F1_REGRESSION_LIMIT = -0.05
BOUNDARY_RATIO_TOLERANCE = 0.05


class TraceConditionedSpacingEscapeTransform:
    """Eval-only generated-prefix spacing and boundary escape transform."""

    def __init__(
        self,
        *,
        vocab: MapperV3Vocab,
        min_repeated_spacings: int = 4,
        min_spacing_ms: int = 40,
        max_spacing_ms: int = 400,
        spacing_penalty: float = 2.0,
        boundary_event_boost: float = 1.5,
        boundary_event_rank_limit: int = 5,
        boundary_event_margin_floor: float = -2.0,
        boundary_near_ms: int = 200,
        max_spacing_activations_per_window: int = 8,
        max_boundary_activations_per_window: int = 8,
        max_total_activations: int = 32,
        max_examples: int = 20,
    ) -> None:
        self.vocab = vocab
        self.min_repeated_spacings = int(min_repeated_spacings)
        if self.min_repeated_spacings <= 0:
            raise ValueError("min_repeated_spacings must be positive")
        self.min_spacing_ms = int(min_spacing_ms)
        self.max_spacing_ms = int(max_spacing_ms)
        if self.min_spacing_ms <= 0 or self.max_spacing_ms < self.min_spacing_ms:
            raise ValueError("spacing bounds must be positive and ordered")
        self.spacing_penalty = float(spacing_penalty)
        self.boundary_event_boost = float(boundary_event_boost)
        if self.spacing_penalty < 0.0 or self.boundary_event_boost < 0.0:
            raise ValueError("penalties and boosts must be non-negative")
        self.boundary_event_rank_limit = int(boundary_event_rank_limit)
        if self.boundary_event_rank_limit <= 0:
            raise ValueError("boundary_event_rank_limit must be positive")
        self.boundary_event_margin_floor = float(boundary_event_margin_floor)
        self.boundary_near_ms = int(boundary_near_ms)
        if self.boundary_near_ms < 0:
            raise ValueError("boundary_near_ms must be non-negative")
        self.max_spacing_activations_per_window = int(max_spacing_activations_per_window)
        self.max_boundary_activations_per_window = int(max_boundary_activations_per_window)
        self.max_total_activations = int(max_total_activations)
        self.max_examples = int(max_examples)
        if min(
            self.max_spacing_activations_per_window,
            self.max_boundary_activations_per_window,
            self.max_total_activations,
            self.max_examples,
        ) < 0:
            raise ValueError("activation caps and max_examples must be non-negative")
        self.spacing_activation_count = 0
        self.boundary_activation_count = 0
        self.candidate_count = 0
        self.window_spacing_counts: Counter[int] = Counter()
        self.window_boundary_counts: Counter[int] = Counter()
        self.examples: list[dict[str, Any]] = []

    @property
    def activation_count(self) -> int:
        return int(self.spacing_activation_count + self.boundary_activation_count)

    def __call__(self, step: MapperV3GenerationStep, logits: torch.Tensor) -> torch.Tensor:
        flat_logits = torch.as_tensor(logits, dtype=torch.float32).reshape(-1)
        transformed = flat_logits.clone()
        valid_mask = step.valid_token_mask.to(device=flat_logits.device, dtype=torch.bool).reshape(-1)
        if int(valid_mask.numel()) != int(flat_logits.numel()):
            raise ValueError(f"logits must contain {valid_mask.numel()} values, got {flat_logits.numel()}")
        if not bool(valid_mask.any().item()):
            return transformed

        if self.activation_count >= self.max_total_activations:
            return transformed

        spacing = self._spacing_candidate(step, valid_mask=valid_mask)
        boundary = self._boundary_candidate(step, flat_logits=flat_logits, valid_mask=valid_mask)
        if spacing is not None:
            self.candidate_count += 1
            write_start = int(step.write_start_ms)
            if (
                self.spacing_penalty > 0.0
                and self.window_spacing_counts[write_start] < self.max_spacing_activations_per_window
            ):
                token_id = int(spacing["token_id"])
                transformed[token_id] = transformed[token_id] - float(self.spacing_penalty)
                self.spacing_activation_count += 1
                self.window_spacing_counts[write_start] += 1
                self._record_example(
                    {
                        **spacing,
                        "mode": "spacing_penalty",
                        "penalty": float(self.spacing_penalty),
                    }
                )

        if self.activation_count >= self.max_total_activations:
            return transformed

        if boundary is not None:
            self.candidate_count += 1
            write_start = int(step.write_start_ms)
            if (
                self.boundary_event_boost > 0.0
                and self.window_boundary_counts[write_start] < self.max_boundary_activations_per_window
            ):
                token_id = int(boundary["token_id"])
                transformed[token_id] = transformed[token_id] + float(self.boundary_event_boost)
                self.boundary_activation_count += 1
                self.window_boundary_counts[write_start] += 1
                self._record_example(
                    {
                        **boundary,
                        "mode": "boundary_event_boost",
                        "boost": float(self.boundary_event_boost),
                    }
                )
        return transformed

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "min_repeated_spacings": int(self.min_repeated_spacings),
            "min_spacing_ms": int(self.min_spacing_ms),
            "max_spacing_ms": int(self.max_spacing_ms),
            "spacing_penalty": float(self.spacing_penalty),
            "boundary_event_boost": float(self.boundary_event_boost),
            "boundary_event_rank_limit": int(self.boundary_event_rank_limit),
            "boundary_event_margin_floor": float(self.boundary_event_margin_floor),
            "boundary_near_ms": int(self.boundary_near_ms),
            "max_spacing_activations_per_window": int(self.max_spacing_activations_per_window),
            "max_boundary_activations_per_window": int(self.max_boundary_activations_per_window),
            "max_total_activations": int(self.max_total_activations),
            "candidate_count": int(self.candidate_count),
            "activation_count": int(self.activation_count),
            "spacing_activation_count": int(self.spacing_activation_count),
            "boundary_activation_count": int(self.boundary_activation_count),
            "window_spacing_counts": {str(key): int(value) for key, value in sorted(self.window_spacing_counts.items())},
            "window_boundary_counts": {str(key): int(value) for key, value in sorted(self.window_boundary_counts.items())},
            "examples": list(self.examples),
        }

    def _spacing_candidate(
        self,
        step: MapperV3GenerationStep,
        *,
        valid_mask: torch.Tensor,
    ) -> dict[str, Any] | None:
        if not bool(step.state.event_emitted_at_current_ms):
            return None
        event_times = _generated_event_times(step, vocab=self.vocab)
        if not event_times or int(event_times[-1]) != int(step.state.current_ms):
            return None
        if len(event_times) <= int(self.min_repeated_spacings):
            return None
        spacings = [int(right) - int(left) for left, right in zip(event_times[:-1], event_times[1:], strict=True)]
        recent = spacings[-int(self.min_repeated_spacings) :]
        if len(set(recent)) != 1:
            return None
        spacing_ms = int(recent[0])
        if spacing_ms < int(self.min_spacing_ms) or spacing_ms > int(self.max_spacing_ms):
            return None
        try:
            first_piece_ms = int(self.vocab.decompose_time_shift_delta(spacing_ms)[0])
            token_id = int(self.vocab.time_shift_token_id(first_piece_ms))
        except (IndexError, ValueError):
            return None
        if token_id >= int(valid_mask.numel()) or not bool(valid_mask[token_id].item()):
            return None
        alternatives = [
            int(candidate_id)
            for candidate_id in self.vocab.time_shift_token_ids
            if int(candidate_id) != token_id
            and 0 <= int(candidate_id) < int(valid_mask.numel())
            and bool(valid_mask[int(candidate_id)].item())
        ]
        if not alternatives:
            return None
        return {
            "token_id": int(token_id),
            "token_name": self.vocab.token_name(token_id),
            "spacing_ms": int(spacing_ms),
            "first_piece_ms": int(first_piece_ms),
            "current_ms": int(step.state.current_ms),
            "write_start_ms": int(step.write_start_ms),
            "token_index": int(step.token_index),
            "recent_event_times": list(event_times[-(int(self.min_repeated_spacings) + 1) :]),
        }

    def _boundary_candidate(
        self,
        step: MapperV3GenerationStep,
        *,
        flat_logits: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, Any] | None:
        if bool(step.state.event_emitted_at_current_ms):
            return None
        if not self._boundary_context(step):
            return None
        masked = flat_logits.masked_fill(~valid_mask, -torch.inf)
        argmax_id = int(torch.argmax(masked).item())
        if self.vocab.is_event_token(argmax_id):
            return None
        valid_events = [
            int(token_id)
            for token_id in self.vocab.event_token_ids
            if 0 <= int(token_id) < int(valid_mask.numel()) and bool(valid_mask[int(token_id)].item())
        ]
        if not valid_events:
            return None
        event_tensor = torch.tensor(valid_events, dtype=torch.long, device=masked.device)
        event_logits = masked.index_select(0, event_tensor)
        best_index = int(torch.argmax(event_logits).item())
        best_event_id = int(valid_events[best_index])
        best_event_logit = float(event_logits[best_index].item())
        rank = int(torch.sum(masked > best_event_logit).item()) + 1
        margin = float(best_event_logit - float(masked[argmax_id].item()))
        if rank > int(self.boundary_event_rank_limit) or margin < float(self.boundary_event_margin_floor):
            return None
        return {
            "token_id": int(best_event_id),
            "token_name": self.vocab.token_name(best_event_id),
            "event_signature": self.vocab.event_signature(best_event_id),
            "event_rank": int(rank),
            "event_margin_vs_argmax": float(margin),
            "argmax_token": self.vocab.token_name(argmax_id),
            "current_ms": int(step.state.current_ms),
            "write_start_ms": int(step.write_start_ms),
            "token_index": int(step.token_index),
        }

    def _boundary_context(self, step: MapperV3GenerationStep) -> bool:
        if int(step.write_start_ms) >= SECOND_WINDOW_START_MS:
            return True
        return int(step.write_end_ms) - int(step.state.current_ms) <= int(self.boundary_near_ms)

    def _record_example(self, example: Mapping[str, Any]) -> None:
        if len(self.examples) < self.max_examples:
            self.examples.append(dict(example))


def run_v3_trace_conditioned_spacing_escape_smoke(
    *,
    baseline_summary_path: str | Path = DEFAULT_BASELINE_SUMMARY_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    output_summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    output_report_path: str | Path | None = DEFAULT_REPORT_PATH,
    primary_case_ids: Sequence[str] = PRIMARY_CASE_IDS,
    sentinel_case_ids: Sequence[str] = SENTINEL_CASE_IDS,
    mapper_checkpoint_path: str | Path | None = None,
    control_checkpoint_path: str | Path | None = None,
    device_name: str = "auto",
    case_limit: int | None = None,
    max_tokens_per_window: int = 512,
    seed: int = 1337,
    spacing_penalty: float = 2.0,
    boundary_event_boost: float = 1.5,
    boundary_event_rank_limit: int = 5,
    boundary_event_margin_floor: float = -2.0,
    max_spacing_activations_per_window: int = 8,
    max_boundary_activations_per_window: int = 8,
    max_total_activations: int = 32,
    max_examples: int = 20,
    experiment_card_path: str | Path = DEFAULT_EXPERIMENT_CARD_PATH,
) -> dict[str, Any]:
    baseline_summary = _read_json(Path(baseline_summary_path))
    baseline_by_case = _baseline_runs_by_case_id(baseline_summary)
    requested_rows: list[tuple[str, str]] = [
        *[(case_id, "primary") for case_id in primary_case_ids],
        *[(case_id, "sentinel") for case_id in sentinel_case_ids],
    ]
    if case_limit is not None:
        limit = int(case_limit)
        if limit <= 0:
            raise ValueError("case_limit must be positive when provided")
        requested_rows = requested_rows[:limit]
    missing = [case_id for case_id, _group in requested_rows if case_id not in baseline_by_case]
    if missing:
        raise ValueError(f"missing requested case ids in baseline summary: {missing}")

    mapper_checkpoint = Path(mapper_checkpoint_path or str(baseline_summary.get("checkpoint_path", "")))
    control_checkpoint = Path(control_checkpoint_path or str(baseline_summary.get("control_checkpoint_path", "")))
    if not mapper_checkpoint.exists():
        raise FileNotFoundError(f"missing mapper checkpoint: {mapper_checkpoint}")
    if not control_checkpoint.exists():
        raise FileNotFoundError(f"missing control checkpoint: {control_checkpoint}")

    out_dir = Path(output_dir)
    rollout_dir = out_dir / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    vocab = MapperV3Vocab()
    case_results: list[dict[str, Any]] = []
    for index, (case_id, group) in enumerate(requested_rows, start=1):
        baseline_row = dict(baseline_by_case[case_id])
        chart_end_ms = int(baseline_row["chart_end_ms"])
        print(f"trace-conditioned rollout {index:02d}/{len(requested_rows)} {group} {case_id}", flush=True)
        transform = TraceConditionedSpacingEscapeTransform(
            vocab=vocab,
            spacing_penalty=float(spacing_penalty),
            boundary_event_boost=float(boundary_event_boost),
            boundary_event_rank_limit=int(boundary_event_rank_limit),
            boundary_event_margin_floor=float(boundary_event_margin_floor),
            max_spacing_activations_per_window=int(max_spacing_activations_per_window),
            max_boundary_activations_per_window=int(max_boundary_activations_per_window),
            max_total_activations=int(max_total_activations),
            max_examples=int(max_examples),
        )
        summary_path = rollout_dir / f"{_safe_filename(case_id)}_{chart_end_ms}ms_spacing_escape_summary.json"
        candidate_summary = run_trained_v3_runtime_rollout_smoke(
            mapper_checkpoint_path=mapper_checkpoint,
            control_checkpoint_path=control_checkpoint,
            output_summary_path=summary_path,
            output_report_path=None,
            device_name=device_name,
            chart_end_ms=chart_end_ms,
            max_tokens_per_window=int(max_tokens_per_window),
            audio_path=str(baseline_row["audio_path"]),
            real_audio=True,
            normalized_difficulty=float(baseline_row["normalized_difficulty"]),
            temperature=0.0,
            top_p=None,
            seed=int(seed),
            time_shift_length_penalty_alpha=0.0,
            time_shift_delta_penalty_alpha=0.0,
            collect_logit_diagnostics=True,
            logit_top_k=5,
            logit_max_examples=int(max_examples),
            logits_transform=transform,
            timepoint_preview_limit=4096,
        )
        candidate_metrics = _metrics_from_rollout_summary(
            candidate_summary,
            beatmap_path=Path(str(baseline_row["beatmap_path"])),
            chart_end_ms=chart_end_ms,
        )
        baseline_metrics = _baseline_metrics(baseline_row)
        case_results.append(
            {
                "case_id": case_id,
                "group": group,
                "case_index": baseline_row.get("case_index"),
                "chart_end_ms": chart_end_ms,
                "difficulty": baseline_row.get("difficulty"),
                "normalized_difficulty": baseline_row.get("normalized_difficulty"),
                "audio_path": baseline_row.get("audio_path"),
                "beatmap_path": baseline_row.get("beatmap_path"),
                "summary_path": summary_path.as_posix(),
                "baseline_metrics": baseline_metrics,
                "candidate_metrics": candidate_metrics,
                "comparison": compare_metrics(candidate_metrics, baseline_metrics),
                "candidate_legal": _candidate_legal(candidate_summary),
                "candidate_rollout": _rollout_status(candidate_summary),
                "transform": transform.to_dict(),
            }
        )

    aggregate = aggregate_case_results(case_results)
    guard_results = guard_results_from_aggregate(aggregate)
    decision = decision_from_aggregate(aggregate, guard_results)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar v3 trace-conditioned spacing escape smoke",
        "experiment_card": Path(experiment_card_path).as_posix(),
        "baseline_summary_path": Path(baseline_summary_path).as_posix(),
        "output_dir": out_dir.as_posix(),
        "mapper_checkpoint_path": mapper_checkpoint.as_posix(),
        "control_checkpoint_path": control_checkpoint.as_posix(),
        "code_commit": _git_stdout("rev-parse", "HEAD"),
        "code_dirty": bool(_git_stdout("status", "--short")),
        "config": {
            "device": device_name,
            "case_limit": None if case_limit is None else int(case_limit),
            "max_tokens_per_window": int(max_tokens_per_window),
            "seed": int(seed),
            "spacing_penalty": float(spacing_penalty),
            "boundary_event_boost": float(boundary_event_boost),
            "boundary_event_rank_limit": int(boundary_event_rank_limit),
            "boundary_event_margin_floor": float(boundary_event_margin_floor),
            "max_spacing_activations_per_window": int(max_spacing_activations_per_window),
            "max_boundary_activations_per_window": int(max_boundary_activations_per_window),
            "max_total_activations": int(max_total_activations),
            "max_examples": int(max_examples),
            "no_training": True,
            "tokenizer_changed": False,
            "grammar_defaults_changed": False,
            "decode_defaults_changed": False,
            "target_leakage": False,
        },
        "decision": decision,
        "guard_results": guard_results,
        "aggregate": aggregate,
        "case_results": case_results,
        "worst_cases": worst_cases(case_results),
        "interpretation": interpretation_from_decision(decision),
        "next_step": decision["next_step"],
    }
    write_summary_json(summary, output_summary_path)
    if output_report_path is not None:
        write_report(summary, output_report_path)
    return summary


def compare_metrics(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    candidate_rigid = _float(candidate.get("dominant_spacing_ratio")) or 0.0
    baseline_rigid = _float(baseline.get("dominant_spacing_ratio")) or 0.0
    candidate_second = _float(candidate.get("second_window_event_share")) or 0.0
    baseline_second = _float(baseline.get("second_window_event_share")) or 0.0
    candidate_f1 = _metric_f1(candidate)
    baseline_f1 = _metric_f1(baseline)
    candidate_event_ratio = _float(candidate.get("event_count_ratio")) or 0.0
    return {
        "dominant_spacing_ratio_delta": candidate_rigid - baseline_rigid,
        "dominant_spacing_improved": (candidate_rigid - baseline_rigid) <= RIGID_IMPROVEMENT_DELTA
        or candidate_rigid <= RIGID_TARGET_RATIO,
        "second_window_event_share_delta": candidate_second - baseline_second,
        "second_window_improved": (candidate_second - baseline_second) >= SECOND_WINDOW_IMPROVEMENT_DELTA
        or candidate_second >= SECOND_WINDOW_TARGET_SHARE,
        "f1_delta": candidate_f1 - baseline_f1,
        "event_ratio_ok": candidate_event_ratio <= EVENT_RATIO_MAX,
        "event_count_ratio_delta": candidate_event_ratio - (_float(baseline.get("event_count_ratio")) or 0.0),
    }


def aggregate_case_results(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    primary = [row for row in case_results if row.get("group") == "primary"]
    sentinel = [row for row in case_results if row.get("group") == "sentinel"]
    candidate_metrics = [_mapping(row.get("candidate_metrics")) for row in case_results]
    baseline_metrics = [_mapping(row.get("baseline_metrics")) for row in case_results]
    return {
        "case_count": len(case_results),
        "primary_case_count": len(primary),
        "sentinel_case_count": len(sentinel),
        "all_legal": bool(case_results) and all(bool(row.get("candidate_legal")) for row in case_results),
        "dead_end_count": sum(1 for row in case_results if bool(_mapping(row.get("candidate_rollout")).get("dead_end"))),
        "max_token_count": sum(
            1 for row in case_results if bool(_mapping(row.get("candidate_rollout")).get("max_tokens_exceeded"))
        ),
        "primary": aggregate_group(primary),
        "sentinel": aggregate_group(sentinel),
        "overall": {
            "baseline": aggregate_metrics(baseline_metrics),
            "candidate": aggregate_metrics(candidate_metrics),
        },
        "total_candidate_count": sum(int(_mapping(row.get("transform")).get("candidate_count") or 0) for row in case_results),
        "total_activation_count": sum(int(_mapping(row.get("transform")).get("activation_count") or 0) for row in case_results),
        "total_spacing_activation_count": sum(
            int(_mapping(row.get("transform")).get("spacing_activation_count") or 0) for row in case_results
        ),
        "total_boundary_activation_count": sum(
            int(_mapping(row.get("transform")).get("boundary_activation_count") or 0) for row in case_results
        ),
    }


def aggregate_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    comparisons = [_mapping(row.get("comparison")) for row in rows]
    baseline_metrics = [_mapping(row.get("baseline_metrics")) for row in rows]
    candidate_metrics = [_mapping(row.get("candidate_metrics")) for row in rows]
    both_improved = [
        row
        for row in comparisons
        if bool(row.get("dominant_spacing_improved")) and bool(row.get("second_window_improved"))
    ]
    return {
        "case_count": len(rows),
        "baseline": aggregate_metrics(baseline_metrics),
        "candidate": aggregate_metrics(candidate_metrics),
        "rigidity_improved_count": sum(1 for row in comparisons if bool(row.get("dominant_spacing_improved"))),
        "second_window_improved_count": sum(1 for row in comparisons if bool(row.get("second_window_improved"))),
        "both_improved_count": len(both_improved),
        "event_ratio_over_count": sum(
            1 for row in candidate_metrics if (_float(row.get("event_count_ratio")) or 0.0) > EVENT_RATIO_MAX
        ),
        "mean_f1_delta": _mean([_float(row.get("f1_delta")) or 0.0 for row in comparisons]),
        "max_boundary_ratio_delta": (
            (_float(aggregate_metrics(candidate_metrics).get("max_boundary_event_ratio")) or 0.0)
            - (_float(aggregate_metrics(baseline_metrics).get("max_boundary_event_ratio")) or 0.0)
        ),
        "activation_count": sum(int(_mapping(row.get("transform")).get("activation_count") or 0) for row in rows),
        "spacing_activation_count": sum(
            int(_mapping(row.get("transform")).get("spacing_activation_count") or 0) for row in rows
        ),
        "boundary_activation_count": sum(
            int(_mapping(row.get("transform")).get("boundary_activation_count") or 0) for row in rows
        ),
    }


def aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    dominant = [_float(row.get("dominant_spacing_ratio")) or 0.0 for row in rows]
    event_ratios = [_float(row.get("event_count_ratio")) or 0.0 for row in rows]
    second = [_float(row.get("second_window_event_share")) or 0.0 for row in rows]
    f1 = [_metric_f1(row) for row in rows]
    boundary = [_float(row.get("boundary_event_ratio")) or 0.0 for row in rows]
    return {
        "case_count": len(rows),
        "mean_dominant_spacing_ratio": _mean(dominant),
        "median_dominant_spacing_ratio": _median(dominant),
        "mean_event_count_ratio": _mean(event_ratios),
        "median_event_count_ratio": _median(event_ratios),
        "mean_second_window_share": _mean(second),
        "median_second_window_share": _median(second),
        "mean_f1_100ms": _mean(f1),
        "median_f1_100ms": _median(f1),
        "max_boundary_event_ratio": max(boundary, default=0.0),
    }


def guard_results_from_aggregate(aggregate: Mapping[str, Any]) -> dict[str, bool]:
    primary = _mapping(aggregate.get("primary"))
    sentinel = _mapping(aggregate.get("sentinel"))
    sentinel_candidate = _mapping(sentinel.get("candidate"))
    primary_candidate = _mapping(primary.get("candidate"))
    primary_baseline = _mapping(primary.get("baseline"))
    return {
        "case_coverage": int(aggregate.get("primary_case_count") or 0) == len(PRIMARY_CASE_IDS)
        and int(aggregate.get("sentinel_case_count") or 0) == len(SENTINEL_CASE_IDS),
        "no_training": True,
        "no_tokenizer_or_default_decode_change": True,
        "no_target_leakage": True,
        "all_legal": bool(aggregate.get("all_legal")),
        "no_dead_end_cases": int(aggregate.get("dead_end_count") or 0) == 0,
        "no_max_token_cases": int(aggregate.get("max_token_count") or 0) == 0,
        "primary_event_ratio_guard": int(primary.get("event_ratio_over_count") or 0) == 0,
        "sentinel_median_event_ratio_guard": (_float(sentinel_candidate.get("median_event_count_ratio")) or 0.0)
        <= EVENT_RATIO_MAX,
        "max_boundary_ratio_guard": (
            (_float(primary_candidate.get("max_boundary_event_ratio")) or 0.0)
            <= (_float(primary_baseline.get("max_boundary_event_ratio")) or 0.0) + BOUNDARY_RATIO_TOLERANCE
        ),
        "mean_f1_guard": (_float(primary.get("mean_f1_delta")) or 0.0) >= MEAN_F1_REGRESSION_LIMIT,
    }


def decision_from_aggregate(aggregate: Mapping[str, Any], guard_results: Mapping[str, bool]) -> dict[str, Any]:
    primary = _mapping(aggregate.get("primary"))
    sentinel = _mapping(aggregate.get("sentinel"))
    primary_count = int(aggregate.get("primary_case_count") or 0)
    primary_threshold = min(4, primary_count)
    if not all(bool(value) for value in guard_results.values()):
        failed = [key for key, value in guard_results.items() if not bool(value)]
        if "sentinel_median_event_ratio_guard" in failed or "primary_event_ratio_guard" in failed:
            return {
                "route": "KILL_OVERPRODUCTION",
                "reason": f"trace-conditioned spacing escape failed overproduction guards: {failed}",
                "next_step": "Stop this local decode repair and pivot to training-side state/objective work or v2.1 grammar mutation.",
            }
        return {
            "route": "KILL_GUARD_FAILURE",
            "reason": f"trace-conditioned spacing escape failed guard checks: {failed}",
            "next_step": "Do not scale this policy; inspect the failed guard before any further decode repair.",
        }
    rigidity = int(primary.get("rigidity_improved_count") or 0)
    second = int(primary.get("second_window_improved_count") or 0)
    both = int(primary.get("both_improved_count") or 0)
    if both >= primary_threshold:
        return {
            "route": "TEST_FULL32_TRACE_CONDITIONED_SPACING_ESCAPE",
            "reason": f"{both}/{primary_count} primary cases improved both rigidity and second-window continuation",
            "next_step": "Create a full32 opt-in trace-conditioned spacing escape gate before any default change.",
        }
    if rigidity >= primary_threshold or second >= primary_threshold:
        return {
            "route": "MUTATE_ACTIVATION_POLICY",
            "reason": f"partial signal: rigidity {rigidity}/{primary_count}, second-window {second}/{primary_count}",
            "next_step": "Inspect activation traces and mutate thresholds before any full32 run.",
        }
    return {
        "route": "KILL_LOCAL_DECODE_REPAIR",
        "reason": f"insufficient primary signal: rigidity {rigidity}/{primary_count}, second-window {second}/{primary_count}",
        "next_step": "Stop local decode repair and route to training-side state/objective work or v2.1/target-grammar mutation.",
    }


def interpretation_from_decision(decision: Mapping[str, Any]) -> str:
    route = str(decision.get("route"))
    if route == "TEST_FULL32_TRACE_CONDITIONED_SPACING_ESCAPE":
        return "The bounded smoke found a trace-local repair surface, but this is not replacement evidence until a full32 opt-in gate passes."
    if route == "MUTATE_ACTIVATION_POLICY":
        return "The smoke produced partial signal only. Keep the evidence local and mutate thresholds before broader runs."
    if route == "KILL_OVERPRODUCTION":
        return "The smoke repeated the known anti-rigid overproduction failure mode. This argues against local decode repair."
    if route == "KILL_GUARD_FAILURE":
        return "The smoke failed a hard safety guard and should not be scaled."
    return "The smoke did not show enough local repair signal; route the next work away from local decode patches."


def worst_cases(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def project(row: Mapping[str, Any]) -> dict[str, Any]:
        baseline = _mapping(row.get("baseline_metrics"))
        candidate = _mapping(row.get("candidate_metrics"))
        comparison = _mapping(row.get("comparison"))
        transform = _mapping(row.get("transform"))
        return {
            "case_id": row.get("case_id"),
            "group": row.get("group"),
            "baseline_event_ratio": baseline.get("event_count_ratio"),
            "candidate_event_ratio": candidate.get("event_count_ratio"),
            "baseline_dominant_spacing": baseline.get("dominant_spacing_ratio"),
            "candidate_dominant_spacing": candidate.get("dominant_spacing_ratio"),
            "baseline_second_share": baseline.get("second_window_event_share"),
            "candidate_second_share": candidate.get("second_window_event_share"),
            "f1_delta": comparison.get("f1_delta"),
            "activations": transform.get("activation_count"),
            "spacing_activations": transform.get("spacing_activation_count"),
            "boundary_activations": transform.get("boundary_activation_count"),
        }

    return {
        "highest_event_ratio": [
            project(row)
            for row in sorted(
                case_results,
                key=lambda item: _float(_mapping(item.get("candidate_metrics")).get("event_count_ratio")) or 0.0,
                reverse=True,
            )[:8]
        ],
        "lowest_second_window_share": [
            project(row)
            for row in sorted(
                case_results,
                key=lambda item: _float(_mapping(item.get("candidate_metrics")).get("second_window_event_share")) or 0.0,
            )[:8]
        ],
        "largest_f1_regression": [
            project(row)
            for row in sorted(
                case_results,
                key=lambda item: _float(_mapping(item.get("comparison")).get("f1_delta")) or 0.0,
            )[:8]
        ],
    }


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
    primary = _mapping(aggregate.get("primary"))
    sentinel = _mapping(aggregate.get("sentinel"))
    primary_base = _mapping(primary.get("baseline"))
    primary_candidate = _mapping(primary.get("candidate"))
    sentinel_candidate = _mapping(sentinel.get("candidate"))
    lines = [
        "# Target Grammar v3 Trace-Conditioned Spacing Escape Smoke Result Report",
        "",
        "## Scope",
        "",
        "This pass runs an eval-only generated-prefix spacing/boundary escape policy on the six traced v3 failures plus four sentinel controls. It does not train, change tokenizer behavior, change grammar defaults, or use target labels during decode.",
        "",
        "## Decision",
        "",
        f"- route: `{decision.get('route')}`",
        f"- reason: {decision.get('reason')}",
        "",
        "## Aggregate",
        "",
        "| Metric | Baseline primary | Candidate primary |",
        "| --- | ---: | ---: |",
        _metric_row("mean dominant spacing", primary_base.get("mean_dominant_spacing_ratio"), primary_candidate.get("mean_dominant_spacing_ratio")),
        _metric_row("mean second-window share", primary_base.get("mean_second_window_share"), primary_candidate.get("mean_second_window_share")),
        _metric_row("mean event ratio", primary_base.get("mean_event_count_ratio"), primary_candidate.get("mean_event_count_ratio")),
        _metric_row("mean F1@100ms", primary_base.get("mean_f1_100ms"), primary_candidate.get("mean_f1_100ms")),
        "",
        f"- Primary rigidity-improved cases: `{primary.get('rigidity_improved_count')}` / `{primary.get('case_count')}`",
        f"- Primary second-window-improved cases: `{primary.get('second_window_improved_count')}` / `{primary.get('case_count')}`",
        f"- Primary both-improved cases: `{primary.get('both_improved_count')}` / `{primary.get('case_count')}`",
        f"- Sentinel median event ratio: `{_fmt(sentinel_candidate.get('median_event_count_ratio'))}`",
        f"- Total activations: `{aggregate.get('total_activation_count')}`",
        f"- Spacing activations: `{aggregate.get('total_spacing_activation_count')}`",
        f"- Boundary activations: `{aggregate.get('total_boundary_activation_count')}`",
        "",
        "## Guard Results",
        "",
        *[f"- {key}: `{str(value).lower()}`" for key, value in _mapping(summary.get("guard_results")).items()],
        "",
        "## Cases",
        "",
        "| group | case | event ratio | second share | dominant spacing | F1 delta | activations |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in _sequence(summary.get("case_results")):
        case = _mapping(row)
        candidate = _mapping(case.get("candidate_metrics"))
        comparison = _mapping(case.get("comparison"))
        transform = _mapping(case.get("transform"))
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{case.get('group')}`",
                    f"`{case.get('case_id')}`",
                    f"`{_fmt(candidate.get('event_count_ratio'))}`",
                    f"`{_fmt(candidate.get('second_window_event_share'))}`",
                    f"`{_fmt(candidate.get('dominant_spacing_ratio'))}`",
                    f"`{_fmt(comparison.get('f1_delta'))}`",
                    f"`{transform.get('activation_count')}`",
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
            "- Proved: whether this bounded eval-only local decode policy clears its six-case smoke and sentinel guards.",
            "- Not proved: full32 quality, default-decode readiness, training objective value, or target-grammar replacement readiness.",
            "",
            "## Next Step",
            "",
            str(summary.get("next_step")),
            "",
        ]
    )
    return "\n".join(lines)


def _metric_row(label: str, baseline: Any, candidate: Any) -> str:
    return f"| {label} | `{_fmt(baseline)}` | `{_fmt(candidate)}` |"


def _metrics_from_rollout_summary(
    summary: Mapping[str, Any],
    *,
    beatmap_path: Path,
    chart_end_ms: int,
) -> dict[str, Any]:
    rollout = _mapping(summary.get("rollout"))
    timepoints = rollout.get("timepoints")
    if not isinstance(timepoints, list):
        raise ValueError("rollout summary does not contain timepoints")
    timepoint_count = int(rollout.get("timepoint_count") or 0)
    if len(timepoints) < timepoint_count:
        raise ValueError(f"rollout timepoints preview is truncated: {len(timepoints)} < {timepoint_count}")
    generated_times = [int(timepoint["time_ms"]) for timepoint in timepoints if isinstance(timepoint, Mapping)]
    reference_times = _reference_times(beatmap_path, chart_end_ms=chart_end_ms)
    return generated_metrics(
        generated_times=generated_times,
        reference_times=reference_times,
        chart_end_ms=int(chart_end_ms),
    )


def _baseline_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "generated_event_count": int(row.get("generated_event_count") or 0),
        "reference_event_count": int(row.get("reference_event_count") or 0),
        "event_count_ratio": _float(row.get("event_count_ratio")) or 0.0,
        "second_window_event_share": _float(row.get("second_window_event_share")) or 0.0,
        "starved": bool(row.get("starved")),
        "boundary_event_ratio": _float(row.get("boundary_event_ratio")) or 0.0,
        "dominant_spacing_ms": row.get("dominant_spacing_ms"),
        "dominant_spacing_ratio": _float(row.get("dominant_spacing_ratio")) or 0.0,
        "timing_match_100ms": dict(_mapping(row.get("timing_match_100ms"))),
    }


def _reference_times(beatmap_path: Path, *, chart_end_ms: int) -> list[int]:
    hitobjects = parse_mania_hit_objects(beatmap_path, expected_key_count=4)
    return [
        int(timepoint.time_ms)
        for timepoint in hitobjects_to_mapper_timepoints(hitobjects)
        if 0 <= int(timepoint.time_ms) <= int(chart_end_ms)
    ]


def _candidate_legal(summary: Mapping[str, Any]) -> bool:
    checks = _mapping(summary.get("checks"))
    return bool(
        checks.get("rollout_not_dead_end")
        and checks.get("rollout_not_max_tokens_exceeded")
        and _mapping(summary.get("rollout")).get("completed")
    )


def _rollout_status(summary: Mapping[str, Any]) -> dict[str, Any]:
    rollout = _mapping(summary.get("rollout"))
    windows = rollout.get("windows")
    terminal_ms = None
    if isinstance(windows, list) and windows:
        terminal_ms = _mapping(windows[-1]).get("terminal_ms")
    return {
        "completed": bool(rollout.get("completed")),
        "dead_end": bool(rollout.get("dead_end")),
        "max_tokens_exceeded": bool(rollout.get("max_tokens_exceeded")),
        "window_count": int(rollout.get("window_count") or 0),
        "token_count": int(rollout.get("token_count") or 0),
        "timepoint_count": int(rollout.get("timepoint_count") or 0),
        "terminal_ms": terminal_ms,
    }


def _generated_event_times(step: MapperV3GenerationStep, *, vocab: MapperV3Vocab) -> tuple[int, ...]:
    state = initial_replay_state(step.ln_carry_in)
    event_times: list[int] = []
    for position, token_id in enumerate(step.generated_tokens):
        token = int(token_id)
        if vocab.is_event_token(token) and (not event_times or event_times[-1] != int(state.current_ms)):
            event_times.append(int(state.current_ms))
        state = transition_replay_state(
            state,
            token,
            position=position,
            vocab=vocab,
            write_start_ms=step.write_start_ms,
            write_end_ms=step.write_end_ms,
            chart_end_ms=step.chart_end_ms,
            ln_carry_out=step.ln_carry_out,
            is_full_chart_start=bool(step.is_full_chart_start),
            is_full_chart_end=bool(step.is_full_chart_end),
        )
    return tuple(event_times)


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


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else ()


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _metric_f1(row: Mapping[str, Any]) -> float:
    timing = _mapping(row.get("timing_match_100ms"))
    return _float(timing.get("f1")) or 0.0


def _mean(values: Sequence[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return float(math.fsum(clean) / len(clean)) if clean else 0.0


def _median(values: Sequence[float]) -> float:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    return float(statistics.median(clean)) if clean else 0.0


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value).lower()).strip("_") or "case"


def _git_stdout(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the v3 trace-conditioned spacing escape smoke.")
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--mapper-checkpoint-path")
    parser.add_argument("--control-checkpoint-path")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--max-tokens-per-window", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--spacing-penalty", type=float, default=2.0)
    parser.add_argument("--boundary-event-boost", type=float, default=1.5)
    parser.add_argument("--boundary-event-rank-limit", type=int, default=5)
    parser.add_argument("--boundary-event-margin-floor", type=float, default=-2.0)
    parser.add_argument("--max-spacing-activations-per-window", type=int, default=8)
    parser.add_argument("--max-boundary-activations-per-window", type=int, default=8)
    parser.add_argument("--max-total-activations", type=int, default=32)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--experiment-card", type=Path, default=DEFAULT_EXPERIMENT_CARD_PATH)
    args = parser.parse_args(argv)
    summary = run_v3_trace_conditioned_spacing_escape_smoke(
        baseline_summary_path=args.baseline_summary,
        output_dir=args.output_dir,
        output_summary_path=args.summary_output,
        output_report_path=args.report_output,
        mapper_checkpoint_path=args.mapper_checkpoint_path,
        control_checkpoint_path=args.control_checkpoint_path,
        device_name=args.device,
        case_limit=args.case_limit,
        max_tokens_per_window=args.max_tokens_per_window,
        seed=args.seed,
        spacing_penalty=args.spacing_penalty,
        boundary_event_boost=args.boundary_event_boost,
        boundary_event_rank_limit=args.boundary_event_rank_limit,
        boundary_event_margin_floor=args.boundary_event_margin_floor,
        max_spacing_activations_per_window=args.max_spacing_activations_per_window,
        max_boundary_activations_per_window=args.max_boundary_activations_per_window,
        max_total_activations=args.max_total_activations,
        max_examples=args.max_examples,
        experiment_card_path=args.experiment_card,
    )
    print(
        "mapper_v3_trace_conditioned_spacing_escape_done "
        f"route={summary['decision']['route']} "
        f"primary_both={summary['aggregate']['primary']['both_improved_count']}/"
        f"{summary['aggregate']['primary_case_count']} "
        f"activations={summary['aggregate']['total_activation_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
