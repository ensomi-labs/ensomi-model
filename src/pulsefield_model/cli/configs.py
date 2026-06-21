from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


CONTROL_WINDOW_INDEX_PATH = (
    "artifacts/indexes/stage2_control_windows_4k_2to6_dense_local_bpm_norm_unique_le3.parquet"
)
CONTROL_V3_TIMESERIES_PATH = (
    "artifacts/features/control_v3_timeseries_4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3.parquet"
)
MAPPER_CHECKPOINT_PATH = (
    "artifacts/runs/stage2_mapper_v2/"
    "stage2_mapper_v2_phase_b_global_d768_l8_b1/checkpoint.pt"
)
CONTROL_CHECKPOINT_PATH = (
    "artifacts/runs/stage2_control_demo/"
    "stage2_control_demo_global_d384_l3_stride16_b6/checkpoints/checkpoint_step_002000.pt"
)
TIME_SHIFT_LENGTH_PENALTY = 5.2


def _control_feature_weights() -> dict[str, float]:
    return {
        "density_level": 1.00,
        "density_burst": 0.35,
        "hold_occupancy": 1.00,
        "ln_change_rate_gated": 0.60,
        "chord_ratio": 1.00,
        "jack_excess": 0.90,
        "jack_streak_exposure": 1.00,
        "hand_balance_signed": 0.60,
        "hand_imbalance_abs": 0.90,
        "repeat_exact": 1.00,
        "repeat_shift": 1.00,
        "repeat_motion": 1.00,
    }


def _control_smooth_l1_deltas() -> dict[str, float]:
    return {
        "density_level": 0.20,
        "density_burst": 0.10,
        "hold_occupancy": 0.10,
        "ln_change_rate_gated": 0.20,
        "chord_ratio": 0.10,
        "jack_excess": 0.10,
        "jack_streak_exposure": 0.10,
        "hand_balance_signed": 0.10,
        "hand_imbalance_abs": 0.10,
        "repeat_exact": 0.10,
        "repeat_shift": 0.10,
        "repeat_motion": 0.10,
    }


@dataclass
class PulsefieldCliConfig:
    training: Any = None
    inference: Any = None
    timing: Any = None
    data: Any = None
    osu_core: Any = None


@dataclass
class ControlEncoderModelConfig:
    mel_dim: int = 160
    timing_dim: int = 4
    context_frames: int = 600
    target_frames: int = 100
    target_offset: int = 250
    d_model: int = 256
    heads: int = 4
    layers: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.1
    conv_blocks: int = 2
    conv_kernel_size: int = 5


@dataclass
class ControlLossConfig:
    confidence_loss_weight: float = 0.25
    feature_weights: dict[str, float] = field(default_factory=_control_feature_weights)
    smooth_l1_deltas: dict[str, float] = field(default_factory=_control_smooth_l1_deltas)
    sparse_low: float = 0.10
    sparse_high: float = 0.50
    sparse_boost: float = 4.0
    hand_balance_imbalance_full_weight: float = 0.25
    ln_change_support_min: float = 2.0
    ln_change_support_full: float = 3.0


@dataclass
class ControlDemoGlobalModelConfig:
    mel_dim: int = 160
    timing_dim: int = 4
    context_frames: int = 600
    target_frames: int = 100
    target_offset: int = 250
    d_model: int = 384
    heads: int = 8
    layers: int = 3
    ffn_dim: int = 1536
    dropout: float = 0.1
    conv_blocks: int = 2
    conv_kernel_size: int = 5
    use_global_memory: bool = True
    global_stride: int = 16
    global_layers: int = 2
    global_ffn_dim: int = 1536
    global_conv_blocks: int = 1
    global_fusion_start_layer: int = 1
    global_gate_init: float = -2.94


@dataclass
class ControlDemoLossConfig:
    density_loss_weight: float = 1.0
    smooth_l1_delta: float = 0.20


@dataclass
class MapperTupleModelConfig:
    vocab_size: int | None = None
    mel_dim: int = 160
    timing_dim: int = 4
    difficulty_dim: int = 1
    control_dim: int = 384
    d_model: int = 384
    heads: int = 8
    layers: int = 4
    ffn_dim: int = 1536
    dropout: float = 0.1
    max_seq_len: int = 512
    density_frames: int = 4
    state_hidden_dim: int | None = None
    state_prior_hidden_dim: int = 64
    ln_close_hidden_dim: int = 128
    lane_embedding_dim: int = 8
    age_embedding_dim: int = 8
    num_age_buckets: int = 32
    age_cap_ms: int = 4000
    state_prior_adapter_scale: float | None = None
    state_prior_scale_init: float = 0.03
    state_prior_max_bias: float = 1.5
    close_scale: float = 0.05
    skip_scale: float = 0.0


@dataclass
class MapperV2ModelConfig(MapperTupleModelConfig):
    use_global_context: bool = True
    global_stride: int = 16
    global_layers: int = 1
    global_ffn_dim: int | None = None
    global_conv_blocks: int = 1
    global_conv_kernel_size: int = 5
    global_gate_init: float = -2.94


@dataclass
class MapperV21ModelConfig(MapperV2ModelConfig):
    max_seq_len: int = 1024


@dataclass
class MapperTuplePhaseBLossConfig:
    lambda_ln_close: float = 0.05
    lambda_adapter_reg: float = 1e-5
    lambda_density: float = 0.0
    lambda_density_teacher: float = 0.0
    close_pos_weight_max: float = 20.0
    density_calibration_scale: float = 1.0
    density_calibration_bias: float = 0.0


@dataclass
class MapperV21LossConfig:
    lambda_density: float = 0.0
    lambda_ln_close: float = 0.05
    lambda_adapter_reg: float = 1e-5
    ln_close_pos_weight: float = 1.0
    ln_close_focal_gamma: float = 1.5
    density_calibration_scale: float = 1.0
    density_calibration_bias: float = 0.0


@dataclass
class ControlTrainingConfig:
    dataset_root: str = "dataset"
    index_path: str | None = None
    eval_index_path: str | None = None
    control_v3_timeseries_path: str | None = None
    output_dir: str = "artifacts/runs/stage2_control/control_encoder"
    max_steps: int = 5000
    eval_every: int = 100
    save_every: int | None = None
    log_every: int | None = None
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    seed: int = 1337
    device: str = "auto"
    run_name: str = "control_encoder"
    resume_from: str | None = None
    eval_fraction: float = 0.1
    eval_size: int | None = None
    final_train_eval_size: int | None = 1024
    num_workers: int = 0
    max_cached_maps: int | None = None
    model: ControlEncoderModelConfig = field(default_factory=ControlEncoderModelConfig)
    loss: ControlLossConfig = field(default_factory=ControlLossConfig)


@dataclass
class ControlDemoGlobalTrainingConfig:
    dataset_root: str = "dataset"
    index_path: str | None = None
    eval_index_path: str | None = None
    control_v3_timeseries_path: str | None = None
    output_dir: str = "artifacts/runs/stage2_control_demo/control_demo_global_encoder"
    max_steps: int = 5000
    eval_every: int = 100
    save_every: int | None = None
    log_every: int | None = None
    batch_size: int = 8
    global_attention_budget: int | None = None
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    seed: int = 1337
    device: str = "auto"
    run_name: str = "control_demo_global_encoder"
    resume_from: str | None = None
    init_from_control_checkpoint: str | None = None
    eval_fraction: float = 0.1
    eval_size: int | None = None
    final_train_eval_size: int | None = 1024
    num_workers: int = 0
    max_cached_maps: int | None = None
    model: ControlDemoGlobalModelConfig = field(default_factory=ControlDemoGlobalModelConfig)
    loss: ControlDemoLossConfig = field(default_factory=ControlDemoLossConfig)


@dataclass
class MapperTupleTrainingConfig:
    dataset_root: str = "dataset"
    index_path: str | None = None
    eval_index_path: str | None = None
    control_v3_timeseries_path: str | None = None
    output_dir: str = "artifacts/runs/stage2_mapper_tuple/phase_b_teacher_forced"
    max_steps: int = 5000
    eval_every: int = 100
    save_every: int | None = None
    log_every: int | None = None
    mps_cleanup_every: int | None = None
    batch_size: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    seed: int = 1337
    device: str = "auto"
    run_name: str = "mapper_tuple_phase_b_teacher_forced"
    init_from_control_checkpoint: str | None = None
    init_from_mapper_checkpoint: str | None = None
    resume_from: str | None = None
    eval_fraction: float = 0.1
    eval_size: int | None = None
    final_train_eval_size: int | None = 1024
    num_workers: int = 0
    max_cached_maps: int | None = None
    dataset_progress: bool | None = None
    mapper_record_cache_path: str | None = None
    length_bucketed_batches: bool = False
    length_bucket_size_multiplier: int = 32
    control_teacher_cache_dir: str | None = None
    precompute_control_teacher_cache: bool = False
    precompute_control_teacher_cache_only: bool = False
    control_teacher_precompute_batch_size: int | None = None
    require_control_teacher_cache: bool = False
    control_teacher_cache_overwrite: bool = False
    model: MapperTupleModelConfig = field(default_factory=MapperTupleModelConfig)
    control_model: ControlDemoGlobalModelConfig = field(default_factory=ControlDemoGlobalModelConfig)
    loss: MapperTuplePhaseBLossConfig = field(default_factory=MapperTuplePhaseBLossConfig)


@dataclass
class MapperV2TrainingConfig(MapperTupleTrainingConfig):
    output_dir: str = "artifacts/runs/stage2_mapper_v2/phase_b_global_teacher_forced"
    batch_size: int = 2
    run_name: str = "mapper_v2_phase_b_global_teacher_forced"
    include_full_song_context: bool = True
    skip_first_eval_pass: bool = True
    model: MapperV2ModelConfig = field(default_factory=MapperV2ModelConfig)


@dataclass
class MapperV21TrainingConfig:
    dataset_root: str = "dataset"
    index_path: str | None = None
    eval_index_path: str | None = None
    control_v3_timeseries_path: str | None = None
    output_dir: str = "artifacts/runs/stage2_mapper_v2_1/phase_b_sparse_global"
    max_steps: int = 5000
    eval_every: int = 100
    save_every: int | None = None
    log_every: int | None = None
    batch_size: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    seed: int = 1337
    device: str = "auto"
    run_name: str = "mapper_v2_1_phase_b_sparse_global"
    init_from_control_checkpoint: str | None = None
    resume_from: str | None = None
    eval_fraction: float = 0.1
    eval_size: int | None = None
    final_train_eval_size: int | None = 1024
    num_workers: int = 0
    max_cached_maps: int | None = None
    dataset_progress: bool = False
    mapper_record_cache_path: str | None = None
    control_teacher_cache_dir: str | None = None
    require_control_teacher_cache: bool = False
    precompute_control_teacher_cache: bool = False
    precompute_control_teacher_cache_only: bool = False
    control_teacher_precompute_batch_size: int | None = None
    control_teacher_cache_overwrite: bool = False
    include_full_song_context: bool = True
    skip_first_eval_pass: bool = False
    mps_cleanup_every: int | None = None
    model: MapperV21ModelConfig = field(default_factory=MapperV21ModelConfig)
    control_model: ControlDemoGlobalModelConfig = field(default_factory=ControlDemoGlobalModelConfig)
    loss: MapperV21LossConfig = field(default_factory=MapperV21LossConfig)


@dataclass
class MapperV21OvernightConfig:
    training_config_name: str = "training/stage2_mapper_v2_1_phase_b_sparse_global_mps"
    training_config_dir: str | None = None
    training_config_overrides: list[str] = field(default_factory=list)
    trainer_overrides: list[str] = field(default_factory=list)
    output_dir: str | None = None
    max_steps: int | None = None
    save_every: int | None = None
    steps_per_process: int | None = None
    poll_seconds: float = 30.0
    post_save_grace_seconds: float = 3.0
    terminate_timeout_seconds: float = 60.0
    restart_delay_seconds: float = 20.0
    max_runs: int = 0
    max_consecutive_failures: int = 3
    log_dir: str | None = None
    dry_run: bool = False
    stop_when_complete: bool = True
    uv_command: str = "uv run --extra mps python -m pulsefield_model.training.mapper_v2_1"


@dataclass
class MapperV21ControlCacheOvernightConfig:
    training_config_name: str = "training/stage2_mapper_v2_1_phase_b_sparse_global_mps"
    training_config_dir: str | None = None
    training_config_overrides: list[str] = field(default_factory=list)
    trainer_overrides: list[str] = field(default_factory=list)
    output_dir: str | None = None
    poll_seconds: float = 30.0
    terminate_timeout_seconds: float = 60.0
    restart_delay_seconds: float = 20.0
    max_runs: int = 16000
    max_consecutive_failures: int = 3
    control_teacher_precompute_batch_size: int = 24
    log_dir: str | None = None
    dry_run: bool = False
    uv_command: str = "uv run --extra mps python -m pulsefield_model.training.mapper_v2_1"


@dataclass
class StreamWithCacheInferenceConfig:
    count: int = 5
    difficulty: float = 4.0
    seed: int | None = None
    index_path: str = CONTROL_WINDOW_INDEX_PATH
    dataset_root: str = "dataset"
    output_dir: str = "artifacts/inference/mapper_v2_cached_stream_random_diff4"
    mapper_checkpoint_path: str = MAPPER_CHECKPOINT_PATH
    control_checkpoint_path: str = CONTROL_CHECKPOINT_PATH
    device: str = "auto"
    beatthis_device: str | None = "cpu"
    beatthis_float16: bool = False
    eager_load_beatthis: bool = True
    canonicalization: str = "none"
    min_duration_s: float = 45.0
    max_duration_s: float = 120.0
    control_batch_size: int = 4
    precompute_full_control: bool = True
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float | None = None
    generation_seed: int | None = None
    use_incremental_mapper_decode: bool = True
    time_shift_length_penalty_alpha: float = TIME_SHIFT_LENGTH_PENALTY
    progress_interval_s: float = 15.0


@dataclass
class WsServerInferenceConfig:
    host: str = "localhost"
    port: int = 8765
    device: str = "auto"
    beatthis_device: str | None = "cpu"
    canonicalization: str = "none"
    difficulty: float = 4.0
    max_tokens: int = 512
    mapper_checkpoint_path: str = MAPPER_CHECKPOINT_PATH
    control_checkpoint_path: str = CONTROL_CHECKPOINT_PATH


@dataclass
class TimingFitterCliConfig:
    min_bpm: float = 20.0
    max_bpm: float = 1000.0
    max_segments: int = 16
    double_tempo_score_ratio_threshold: float | None = None
    canonicalization: str = "none"


@dataclass
class FitAudioTimingConfig(TimingFitterCliConfig):
    audio_path: str = MISSING
    checkpoint: str = "final0"
    device: str = "cpu"
    float16: bool = False
    emit_json: bool = False
    super_timing_shifts: bool = False
    super_timing_shift_ms: list[float] | None = None
    ramp_beat_grid: bool = False
    ramp_beat_grid_allow_no_hint: bool = False
    ramp_hint_start_ms: float | None = None
    ramp_hint_end_ms: float | None = None
    ramp_hint_start_bpm: float | None = None
    ramp_hint_end_bpm: float | None = None


@dataclass
class MockOsuExportTimingConfig(TimingFitterCliConfig):
    audio_path: str = MISSING
    output_dir: str = "artifacts/timing_mock_beatmaps"
    checkpoint: str = "final0"
    device: str = "cpu"
    float16: bool = False
    start_ms: int = 0
    end_ms: int | None = None
    max_beats: int | None = None
    title: str | None = None
    artist: str = "Unknown Artist"
    creator: str = "Pulsefield Timing Mock"
    version: str = "Timing grid mock [0011] [1100]"


@dataclass
class BeatmapIndexBuild4kConfig:
    dataset_root: str = "dataset"
    shard: str = "0"
    output_path: str = "artifacts/indexes/beatmap_index_4k.parquet"


@dataclass
class BeatmapIndexDropTimingAnomaliesConfig:
    source_index_path: str = "artifacts/indexes/beatmap_index_4k.parquet"
    dataset_root: str = "dataset"
    output_path: str = "artifacts/indexes/beatmap_index_4k_no_timing_anomalies.parquet"


@dataclass
class BeatmapIndexFilterDifficultyConfig:
    source_index_path: str = "artifacts/indexes/beatmap_index_4k_no_timing_anomalies.parquet"
    output_path: str = "artifacts/indexes/beatmap_index_4k_no_timing_anomalies_2to6.parquet"
    min_difficulty: float = 2.0
    max_difficulty: float = 6.0


@dataclass
class BeatmapIndexFilterLocalBpmUniqueConfig:
    source_index_path: str = "artifacts/indexes/beatmap_index_4k_no_timing_anomalies_2to6.parquet"
    dataset_root: str = "dataset"
    output_path: str = (
        "artifacts/indexes/"
        "beatmap_index_4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3.parquet"
    )
    report_path: str = (
        "artifacts/reports/indexes/"
        "beatmap_index_4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3.json"
    )
    max_local_bpm_norm_unique_per_beatmapset: int = 3
    bpm_round_decimals: int = 6
    dropped_example_limit: int = 20
    progress_every: int = 0


@dataclass
class BeatmapIndexDataConfig:
    command: str = MISSING
    build_4k: BeatmapIndexBuild4kConfig = field(default_factory=BeatmapIndexBuild4kConfig)
    drop_timing_anomalies: BeatmapIndexDropTimingAnomaliesConfig = field(
        default_factory=BeatmapIndexDropTimingAnomaliesConfig,
    )
    filter_difficulty: BeatmapIndexFilterDifficultyConfig = field(
        default_factory=BeatmapIndexFilterDifficultyConfig,
    )
    filter_local_bpm_unique: BeatmapIndexFilterLocalBpmUniqueConfig = field(
        default_factory=BeatmapIndexFilterLocalBpmUniqueConfig,
    )


@dataclass
class ControlWindowsDataConfig:
    source_index_path: str = (
        "artifacts/indexes/"
        "beatmap_index_4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3.parquet"
    )
    dataset_root: str = "dataset"
    control_v3_summary_path: str = (
        "artifacts/features/control_v3_map_summary_4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3.parquet"
    )
    output_path: str = CONTROL_WINDOW_INDEX_PATH
    report_path: str | None = (
        "artifacts/reports/indexes/stage2_control_windows_4k_2to6_dense_local_bpm_norm_unique_le3.json"
    )
    progress_every: int = 100


@dataclass
class ControlV3ArtifactDataConfig:
    index_path: str = "artifacts/indexes/beatmap_index_4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3.parquet"
    source_index_path: str = "artifacts/indexes/beatmap_index_4k_no_timing_anomalies.parquet"
    dataset_root: str = "dataset"
    timeseries_path: str = CONTROL_V3_TIMESERIES_PATH
    summary_path: str = (
        "artifacts/features/control_v3_map_summary_4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3.parquet"
    )
    metadata_path: str = (
        "artifacts/features/control_v3_artifact_metadata_4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3.json"
    )
    limit: int | None = None
    start: int = 0
    batch_maps: int = 32
    progress_every: int = 25


@dataclass
class BeatRepresentationOsuCoreConfig:
    beatmaps: list[str] = MISSING
    snap_denominator: int = 48
    timing_canonicalization: str = "bpm-80-160"
    diagnostics: bool = False
    diagnostic_subdivisions: list[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 6, 8, 12, 16, 24, 48],
    )
    expected_key_count: int = 4
    any_key_count: bool = False
    limit_events: int | None = None


@dataclass
class DifficultyOsuCoreConfig:
    osu: str = MISSING
    audio: str = MISSING
    speed: float = 1.0


_REGISTERED = False


def register_configs() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=PulsefieldCliConfig)
    cs.store(group="schema/training", name="control", node=ControlTrainingConfig)
    cs.store(group="schema/training", name="control_demo_global", node=ControlDemoGlobalTrainingConfig)
    cs.store(group="schema/training", name="mapper_tuple", node=MapperTupleTrainingConfig)
    cs.store(group="schema/training", name="mapper_v2", node=MapperV2TrainingConfig)
    cs.store(group="schema/training", name="mapper_v2_1", node=MapperV21TrainingConfig)
    cs.store(group="schema/training", name="mapper_v2_1_overnight", node=MapperV21OvernightConfig)
    cs.store(
        group="schema/training",
        name="mapper_v2_1_control_cache_overnight",
        node=MapperV21ControlCacheOvernightConfig,
    )
    cs.store(group="schema/inference", name="stream_with_cache", node=StreamWithCacheInferenceConfig)
    cs.store(group="schema/inference", name="ws_server", node=WsServerInferenceConfig)
    cs.store(group="schema/timing", name="fit_audio", node=FitAudioTimingConfig)
    cs.store(group="schema/timing", name="mock_osu_export", node=MockOsuExportTimingConfig)
    cs.store(group="schema/data", name="beatmap_index", node=BeatmapIndexDataConfig)
    cs.store(group="schema/data", name="control_windows", node=ControlWindowsDataConfig)
    cs.store(group="schema/data", name="control_v3_artifact", node=ControlV3ArtifactDataConfig)
    cs.store(group="schema/osu_core", name="beat_representation", node=BeatRepresentationOsuCoreConfig)
    cs.store(group="schema/osu_core", name="difficulty", node=DifficultyOsuCoreConfig)
    _REGISTERED = True


register_configs()


__all__ = [
    "BeatRepresentationOsuCoreConfig",
    "BeatmapIndexDataConfig",
    "ControlDemoGlobalTrainingConfig",
    "ControlTrainingConfig",
    "ControlV3ArtifactDataConfig",
    "ControlWindowsDataConfig",
    "DifficultyOsuCoreConfig",
    "FitAudioTimingConfig",
    "MapperTupleTrainingConfig",
    "MapperV21ControlCacheOvernightConfig",
    "MapperV21OvernightConfig",
    "MapperV21TrainingConfig",
    "MapperV2TrainingConfig",
    "MockOsuExportTimingConfig",
    "PulsefieldCliConfig",
    "StreamWithCacheInferenceConfig",
    "WsServerInferenceConfig",
    "register_configs",
]
