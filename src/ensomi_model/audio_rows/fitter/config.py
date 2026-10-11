from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class GridFitterConfig:
    min_bpm: float = 20.0
    max_bpm: float = 1000.0
    bpm_step: float = 0.5
    offset_step_ms: float = 20.0
    offset_refine_step_ms: float = 1.0
    offset_refine_radius_ms: float = 10.0
    pulse_width_ms: float = 40.0
    double_tempo_score_ratio_threshold: float = 0.95
    max_segments: int = 16
    min_segment_duration_ms: float = 8000.0
    split_score_improvement_threshold: float = 0.02
    split_phase_change_threshold_ms: float = 10.0
    split_relative_interval_change_threshold: float = 0.025
    split_downbeat_signal_weight: float = 0.35
    autocorrelation_candidate_count: int = 16
    bpm_search_window_ratio: float = 0.08
    bpm_search_window_min_bpm: float = 2.0
    max_grid_candidates_per_segment: int = 1000
    max_split_candidates_per_segment: int = 4
    initial_batch_split_candidate_count: int = 16
    initial_batch_split_min_candidate_count: int = 8
    initial_batch_split_max_parent_score: float = 0.75
    long_prediction_duration_seconds: float = 600.0
    long_max_segments: int = 20
    long_max_grid_candidates_per_segment: int = 1000
    long_max_split_candidates_per_segment: int = 5
    long_downbeat_refine_candidate_count: int = 20
    downbeat_period_beats: int = 4
    downbeat_tie_score_margin: float = 0.05
    downbeat_split_score_bonus: float = 0.5
    downbeat_refine_candidate_count: int = 16
    merge_bpm_tolerance: float = 1.5
    merge_relative_bpm_tolerance: float = 0.005
    merge_phase_tolerance_ms: float = 35.0
    merge_many_similar_min_segments: int = 4
    merge_many_similar_bpm_tolerance: float = 2.0
    merge_alias_min_segments: int = 4
    merge_alias_bpm_tolerance: float = 2.0
    merge_alias_phase_tolerance_ms: float = 60.0
    merge_alias_max_fit_score: float = 0.92
    refine_offset_quantum_ms: float = 1.0
    refine_dominant_min_segments: int = 4
    refine_dominant_min_segment_count: int = 2
    refine_dominant_min_outlier_segments: int = 3
    refine_dominant_min_far_outlier_segments: int = 3
    refine_dominant_min_duration_ratio: float = 0.55
    refine_dominant_bpm_tolerance: float = 1.5
    refine_dominant_far_bpm_tolerance: float = 5.0
    refine_dominant_far_relative_bpm_tolerance: float = 0.05
    refine_dominant_max_first_anchor_ratio: float = 0.25
    refine_dominant_boundary_phase_tolerance_ms: float = 35.0
    refine_dominant_min_off_lattice_transitions: int = 2
    refine_dominant_bpm_snap_tolerance: float = 0.25
    refine_circular_phase_min_coherence: float = 0.25
    refine_structural_max_score_loss: float = 0.24
    alias_tempo_multipliers: tuple[float, ...] = (0.25, 1.0 / 3.0, 0.5, 1.0, 2.0, 3.0, 4.0)
    alias_score_tie_margin: float = 0.03
    alias_score_ratio_threshold: float = 0.97
    alias_preferred_min_bpm: float = 80.0
    alias_preferred_max_bpm: float = 240.0
    alias_preferred_band_bonus: float = 0.015
    alias_current_tempo_bonus: float = 0.04
    alias_downbeat_score_weight: float = 0.02
    alias_continuity_penalty: float = 0.02
    alias_semantic_promotion_in_band_min_bpm: float = 86.0
    alias_semantic_promotion_current_max_bpm: float = 100.0
    alias_semantic_promotion_score_ratio_threshold: float = 0.65
    alias_semantic_promotion_low_bpm_max_fit_score: float = 0.78
    alias_semantic_promotion_strong_score_ratio_threshold: float = 0.78
    alias_semantic_promotion_low_confidence_max_fit_score: float = 0.70
    alias_semantic_promotion_low_confidence_score_ratio_threshold: float = 0.60
    alias_semantic_promotion_low_confidence_max_candidate_bpm: float = 185.0
    alias_semantic_promotion_low_bpm_max_segments: int = 4
    alias_semantic_promotion_bonus: float = 0.35
    alias_collapse_score_ratio_threshold: float = 0.78
    alias_demotion_dropped_support_ratio_threshold: float = 0.35
    alias_promotion_inserted_support_ratio_threshold: float = 0.35
    alias_beat_match_tolerance_ms: float = 45.0


def _effective_config_for_prediction(
    frame_count: int,
    *,
    frame_rate_hz: float,
    config: GridFitterConfig,
) -> GridFitterConfig:
    duration_seconds = float(frame_count) / frame_rate_hz
    if duration_seconds < config.long_prediction_duration_seconds:
        return config
    return replace(
        config,
        max_segments=max(config.max_segments, config.long_max_segments),
        max_grid_candidates_per_segment=max(
            config.max_grid_candidates_per_segment,
            config.long_max_grid_candidates_per_segment,
        ),
        max_split_candidates_per_segment=max(
            config.max_split_candidates_per_segment,
            config.long_max_split_candidates_per_segment,
        ),
        downbeat_refine_candidate_count=max(
            config.downbeat_refine_candidate_count,
            config.long_downbeat_refine_candidate_count,
        ),
    )
