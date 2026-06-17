from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml
from torch.utils.data import DataLoader

from pulsefield_model.data.control_windows import DEFAULT_MAX_CACHED_MAPS
from pulsefield_model.data.mapper_sparse_windows_v3 import (
    MapperV3WindowDataset,
    collate_mapper_v3_windows,
)
from pulsefield_model.models.control import ControlDemoGlobalEncoder, ControlDemoGlobalEncoderConfig
from pulsefield_model.models.mapper.v3 import MapperV3Config, MapperV3Model, MapperV3ModelLoss
from pulsefield_model.models.mapper.v3.loss import MapperV3LossConfig
from pulsefield_model.training.common import (
    DEFAULT_FINAL_TRAIN_EVAL_SIZE,
    ControlTrainingResult,
    ResumableRandomBatchSampler,
    _json_safe,
    _metric_is_count,
    _set_deterministic_seed,
    limit_final_train_eval_dataset,
    split_train_eval_dataset,
)
from pulsefield_model.training.mapper_common import (
    _build_mapper_tuple_optimizer,
    _cleanup_mps_training_memory,
    _move_mapper_batch_tensors,
    precompute_mapper_tuple_phase_b_control_teacher_cache,
)
from pulsefield_model.training.mapper_runner import (
    MapperTrainingSpec,
    default_mapper_metric_finalizer,
    mapper_metrics_for_loader,
    resume_resumable_loader_cursor_or_advance,
    run_mapper_training,
)


DEFAULT_RUNS_ROOT = Path("artifacts/runs/stage2_mapper_v3")
DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_ROOT / "phase_b_event_global"
RUN_CONFIG_KEYS = {
    "dataset_root",
    "index_path",
    "eval_index_path",
    "control_v3_timeseries_path",
    "output_dir",
    "max_steps",
    "eval_every",
    "save_every",
    "log_every",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "seed",
    "device",
    "run_name",
    "init_from_control_checkpoint",
    "resume_from",
    "eval_fraction",
    "eval_size",
    "final_train_eval_size",
    "num_workers",
    "max_cached_maps",
    "dataset_progress",
    "mapper_record_cache_path",
    "control_teacher_cache_dir",
    "require_control_teacher_cache",
    "precompute_control_teacher_cache",
    "precompute_control_teacher_cache_only",
    "control_teacher_precompute_batch_size",
    "control_teacher_cache_overwrite",
    "include_full_song_context",
    "skip_first_eval_pass",
    "mps_cleanup_every",
    "model",
    "control_model",
    "loss",
}
MODEL_CONFIG_KEYS = {field.name for field in fields(MapperV3Config)}
CONTROL_MODEL_CONFIG_KEYS = {field.name for field in fields(ControlDemoGlobalEncoderConfig)}
LOSS_CONFIG_KEYS = {field.name for field in fields(MapperV3LossConfig)}


def load_run_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML run config: {path}") from exc
    if loaded is None:
        return {"model": {}, "control_model": {}, "loss": {}}
    if not isinstance(loaded, dict):
        raise ValueError(f"run config must be a mapping: {path}")
    config = dict(loaded)
    unknown = sorted(set(config) - RUN_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unknown run config keys: {unknown}")
    config["model"] = _normalized_section(config.get("model", {}), allowed=MODEL_CONFIG_KEYS, name="model config")
    config["control_model"] = _normalized_section(
        config.get("control_model", {}),
        allowed=CONTROL_MODEL_CONFIG_KEYS,
        name="control model config",
    )
    config["loss"] = _normalized_section(config.get("loss", {}), allowed=LOSS_CONFIG_KEYS, name="loss config")
    return config


def run_mapper_v3_phase_b_training(
    *,
    dataset_root: Path = Path("dataset"),
    index_path: Path | None = None,
    eval_index_path: Path | None = None,
    control_v3_timeseries_path: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_steps: int = 5000,
    eval_every: int = 100,
    save_every: int | None = None,
    log_every: int | None = None,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    seed: int = 1337,
    device_name: str = "auto",
    run_name: str = "mapper_v3_phase_b_event_global",
    init_from_control_checkpoint: Path | None = None,
    resume_from: Path | None = None,
    eval_fraction: float = 0.1,
    eval_size: int | None = None,
    final_train_eval_size: int | None = DEFAULT_FINAL_TRAIN_EVAL_SIZE,
    num_workers: int = 0,
    max_cached_maps: int | None = None,
    dataset_progress: bool = False,
    mapper_record_cache_path: Path | None = None,
    control_teacher_cache_dir: Path | None = None,
    require_control_teacher_cache: bool = False,
    precompute_control_teacher_cache: bool = False,
    control_teacher_precompute_batch_size: int | None = None,
    control_teacher_cache_overwrite: bool = False,
    include_full_song_context: bool = True,
    skip_first_eval_pass: bool = False,
    mps_cleanup_every: int | None = None,
    model_config_overrides: Mapping[str, Any] | None = None,
    control_model_config_overrides: Mapping[str, Any] | None = None,
    loss_config_overrides: Mapping[str, Any] | None = None,
) -> ControlTrainingResult:
    if resume_from is not None:
        raise ValueError("mapper v3 resume checkpoints are not implemented yet")
    _set_deterministic_seed(seed)
    control_model_config = ControlDemoGlobalEncoderConfig(**dict(control_model_config_overrides or {}))
    model_values = dict(model_config_overrides or {})
    model_values.setdefault("control_dim", control_model_config.d_model)
    model_config = MapperV3Config(**model_values)
    if model_config.control_dim != control_model_config.d_model:
        raise ValueError("mapper control_dim must match frozen control_model d_model")
    if model_config.use_global_context and not include_full_song_context:
        raise ValueError("Mapper V3 global context requires include_full_song_context=True")
    loss_config = MapperV3LossConfig(**dict(loss_config_overrides or {}))
    if float(loss_config.lambda_delta_event_factor_target) > 0.0 and not model_config.use_delta_event_factor_target:
        raise ValueError("lambda_delta_event_factor_target > 0 requires use_delta_event_factor_target=True")
    include_delta_event_factor_target = bool(
        model_config.use_delta_event_factor_target
        or float(loss_config.lambda_delta_event_factor_target) > 0.0
    )

    cache_precompute_reports: list[dict[str, Any]] = []
    source_control_dataset = None
    eval_control_dataset = None
    if precompute_control_teacher_cache:
        precompute_run = precompute_mapper_tuple_phase_b_control_teacher_cache(
            dataset_root=dataset_root,
            index_path=index_path,
            eval_index_path=eval_index_path,
            control_v3_timeseries_path=control_v3_timeseries_path,
            batch_size=batch_size,
            seed=seed,
            device_name=device_name,
            init_from_control_checkpoint=init_from_control_checkpoint,
            num_workers=num_workers,
            max_cached_maps=max_cached_maps,
            dataset_progress=dataset_progress,
            control_teacher_cache_dir=control_teacher_cache_dir,
            control_teacher_precompute_batch_size=control_teacher_precompute_batch_size,
            control_teacher_cache_overwrite=control_teacher_cache_overwrite,
            control_model_config=control_model_config,
        )
        cache_precompute_reports = precompute_run.reports
        source_control_dataset = precompute_run.source_control_dataset
        eval_control_dataset = precompute_run.eval_control_dataset

    if source_control_dataset is None:
        dataset_kwargs: dict[str, Any] = {"dataset_root": dataset_root}
        if index_path is not None:
            dataset_kwargs["index_path"] = index_path
        if control_v3_timeseries_path is not None:
            dataset_kwargs["control_v3_timeseries_path"] = control_v3_timeseries_path
    else:
        dataset_kwargs = {"control_dataset": source_control_dataset}
    dataset_kwargs.update(
        {
            "include_full_song_context": bool(include_full_song_context),
            "include_delta_event_factor_target": include_delta_event_factor_target,
            "max_cached_maps": DEFAULT_MAX_CACHED_MAPS if max_cached_maps is None else max_cached_maps,
            "progress": bool(dataset_progress),
        }
    )
    if mapper_record_cache_path is not None:
        dataset_kwargs["mapper_record_cache_path"] = mapper_record_cache_path
    if control_teacher_cache_dir is not None:
        dataset_kwargs["control_teacher_cache_dir"] = control_teacher_cache_dir
        dataset_kwargs["require_control_teacher_cache"] = bool(require_control_teacher_cache)
    train_source = MapperV3WindowDataset(**dataset_kwargs)
    if len(train_source) == 0:
        raise ValueError("MapperV3WindowDataset produced no training windows")

    if eval_index_path is not None:
        if eval_control_dataset is None:
            eval_kwargs = dict(dataset_kwargs)
            eval_kwargs["index_path"] = eval_index_path
        else:
            eval_kwargs = {
                "control_dataset": eval_control_dataset,
                "include_full_song_context": bool(include_full_song_context),
                "include_delta_event_factor_target": include_delta_event_factor_target,
                "max_cached_maps": DEFAULT_MAX_CACHED_MAPS if max_cached_maps is None else max_cached_maps,
                "progress": bool(dataset_progress),
            }
            if mapper_record_cache_path is not None:
                eval_kwargs["mapper_record_cache_path"] = mapper_record_cache_path
            if control_teacher_cache_dir is not None:
                eval_kwargs["control_teacher_cache_dir"] = control_teacher_cache_dir
                eval_kwargs["require_control_teacher_cache"] = bool(require_control_teacher_cache)
        eval_dataset = MapperV3WindowDataset(**eval_kwargs)
        train_dataset = train_source
    else:
        train_dataset, eval_dataset = split_train_eval_dataset(
            train_source,
            eval_fraction=eval_fraction,
            eval_size=eval_size,
            seed=seed,
        )
    if len(train_dataset) == 0:
        raise ValueError("training split is empty")
    if len(eval_dataset) == 0:
        eval_dataset = train_dataset

    loader = DataLoader(
        train_dataset,
        batch_sampler=ResumableRandomBatchSampler(
            train_dataset,
            batch_size=batch_size,
            seed=seed,
        ),
        num_workers=num_workers,
        collate_fn=collate_mapper_v3_windows,
    )
    train_eval_dataset = limit_final_train_eval_dataset(
        train_dataset,
        final_train_eval_size=final_train_eval_size,
        seed=seed,
    )
    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_mapper_v3_windows,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_mapper_v3_windows,
    )
    return _run_training(
        loader=loader,
        train_eval_loader=train_eval_loader,
        eval_loader=eval_loader,
        output_dir=output_dir,
        model_config=model_config,
        control_model_config=control_model_config,
        loss_config=loss_config,
        max_steps=max_steps,
        eval_every=eval_every,
        save_every=save_every,
        log_every=log_every,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        device_name=device_name,
        run_name=run_name,
        dataset_report={
            "train_window_count": len(train_dataset),
            "eval_window_count": len(eval_dataset),
            "source_window_count": len(train_source),
            "eval_index_path": eval_index_path.as_posix() if eval_index_path is not None else None,
            "eval_fraction": eval_fraction,
            "eval_size": eval_size,
            "final_train_eval_size": final_train_eval_size,
            "final_train_eval_window_count": len(train_eval_dataset),
            "filter_report": asdict(train_source.filter_report),
            "mapper_record_cache_path": (
                mapper_record_cache_path.as_posix() if mapper_record_cache_path is not None else None
            ),
            "num_workers": num_workers,
            "control_teacher_cache_dir": (
                control_teacher_cache_dir.as_posix() if control_teacher_cache_dir is not None else None
            ),
            "require_control_teacher_cache": bool(require_control_teacher_cache),
            "precompute_control_teacher_cache": bool(precompute_control_teacher_cache),
            "control_teacher_precompute_batch_size": control_teacher_precompute_batch_size,
            "control_teacher_cache_overwrite": bool(control_teacher_cache_overwrite),
            "control_teacher_cache_precompute": cache_precompute_reports,
            "include_full_song_context": bool(include_full_song_context),
            "include_delta_event_factor_target": include_delta_event_factor_target,
            "mapper_token_contract": "v3_event_groups",
        },
        init_from_control_checkpoint=init_from_control_checkpoint,
        progress_label="mapper_v3_phase_b",
        skip_first_eval_pass=skip_first_eval_pass,
        mps_cleanup_every=mps_cleanup_every,
    )


def _run_training(
    *,
    loader: DataLoader,
    train_eval_loader: DataLoader,
    eval_loader: DataLoader,
    output_dir: Path,
    model_config: MapperV3Config,
    control_model_config: ControlDemoGlobalEncoderConfig,
    loss_config: MapperV3LossConfig,
    max_steps: int,
    eval_every: int,
    save_every: int | None,
    log_every: int | None,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device_name: str,
    run_name: str,
    dataset_report: Mapping[str, Any],
    init_from_control_checkpoint: Path | None,
    progress_label: str,
    skip_first_eval_pass: bool = False,
    mps_cleanup_every: int | None = None,
) -> ControlTrainingResult:
    return run_mapper_training(
        loader=loader,
        train_eval_loader=train_eval_loader,
        eval_loader=eval_loader,
        output_dir=output_dir,
        spec=_mapper_v3_training_spec(
            model_config=model_config,
            control_model_config=control_model_config,
            loss_config=loss_config,
            progress_label=progress_label,
        ),
        max_steps=max_steps,
        eval_every=eval_every,
        save_every=save_every,
        log_every=log_every,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        device_name=device_name,
        run_name=run_name,
        dataset_report=dataset_report,
        init_from_control_checkpoint=init_from_control_checkpoint,
        resume_from=None,
        skip_first_eval_pass=skip_first_eval_pass,
        mps_cleanup_every=mps_cleanup_every,
    )


def _loss_for_raw_batch(
    model: MapperV3Model,
    loss_fn: MapperV3ModelLoss,
    raw_batch: Mapping[str, Any],
    *,
    device: torch.device,
):
    batch = _move_mapper_batch_tensors(raw_batch, device)
    output = model(batch)
    return loss_fn(output, batch)


def _mapper_v3_training_spec(
    *,
    model_config: MapperV3Config,
    control_model_config: ControlDemoGlobalEncoderConfig | None,
    loss_config: MapperV3LossConfig,
    progress_label: str = "mapper_v3_phase_b",
) -> MapperTrainingSpec:
    return MapperTrainingSpec(
        model_config=model_config,
        control_model_config=control_model_config,
        loss_config=loss_config,
        model_factory=_mapper_v3_model_factory,
        loss_factory=_mapper_v3_loss_factory,
        batch_loss_adapter=_mapper_v3_batch_loss_adapter,
        optimizer_factory=lambda model, learning_rate, weight_decay: _build_mapper_tuple_optimizer(
            model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        ),
        training_config_factory=lambda context: _mapper_v3_training_config(
            seed=context.seed,
            run_name=context.run_name,
            learning_rate=context.learning_rate,
            weight_decay=context.weight_decay,
            eval_every=context.eval_every,
            save_every=context.save_every,
            skip_first_eval_pass=context.skip_first_eval_pass,
            dataset_report=context.dataset_report,
            mps_cleanup_every=context.mps_cleanup_every,
        ),
        resume_checkpoint_loader=_load_mapper_v3_resume_checkpoint,
        metric_count_predicate=lambda key: _metric_is_count(key) or key.endswith("_count"),
        metric_fallback_weight_key="token/valid_count",
        metric_empty={"loss/total": float("nan"), "loss/token": float("nan"), "loss/density": 0.0},
        metric_finalizer=default_mapper_metric_finalizer,
        progress_label=progress_label,
        metric_divider=lambda numerator, denominator: numerator / max(denominator, 1e-12),
        resume_loader_cursor=resume_resumable_loader_cursor_or_advance,
        cleanup_device_memory=_cleanup_mps_training_memory,
    )


def _mapper_v3_model_factory(
    model_config: Any,
    control_encoder: ControlDemoGlobalEncoder | None,
) -> MapperV3Model:
    if not isinstance(model_config, MapperV3Config):
        raise TypeError("mapper v3 training requires MapperV3Config")
    return MapperV3Model(model_config, control_encoder=control_encoder)


def _mapper_v3_loss_factory(model: torch.nn.Module, loss_config: Any) -> MapperV3ModelLoss:
    if not isinstance(model, MapperV3Model):
        raise TypeError("mapper v3 training requires MapperV3Model")
    if not isinstance(loss_config, MapperV3LossConfig):
        raise TypeError("mapper v3 training requires MapperV3LossConfig")
    return MapperV3ModelLoss(loss_config, vocab=model.vocab)


def _load_mapper_v3_resume_checkpoint(path: object, context: object) -> Mapping[str, Any]:
    raise ValueError("mapper v3 resume checkpoints are not implemented yet")


def _mapper_v3_batch_loss_adapter(
    model: torch.nn.Module,
    loss_fn: Any,
    raw_batch: Mapping[str, Any],
    device: torch.device,
):
    if not isinstance(model, MapperV3Model):
        raise TypeError("mapper v3 training requires MapperV3Model")
    if not isinstance(loss_fn, MapperV3ModelLoss):
        raise TypeError("mapper v3 training requires MapperV3ModelLoss")
    return _loss_for_raw_batch(model, loss_fn, raw_batch, device=device)


@torch.inference_mode()
def metrics_for_loader(
    model: MapperV3Model,
    loss_fn: MapperV3ModelLoss,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float]:
    spec = _mapper_v3_training_spec(
        model_config=getattr(model, "config", MapperV3Config()),
        control_model_config=None,
        loss_config=loss_fn.config,
        progress_label="mapper_v3_phase_b",
    )
    return mapper_metrics_for_loader(model, loss_fn, loader, device=device, spec=spec)


def _mapper_v3_training_config(
    *,
    seed: int,
    run_name: str,
    learning_rate: float,
    weight_decay: float,
    eval_every: int,
    save_every: int,
    skip_first_eval_pass: bool,
    dataset_report: Mapping[str, Any],
    mps_cleanup_every: int | None,
) -> dict[str, Any]:
    return {
        "phase": "B",
        "seed": seed,
        "run_name": run_name,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "eval_every": eval_every,
        "skip_first_eval_pass": bool(skip_first_eval_pass),
        "save_every": save_every,
        "mps_cleanup_every": mps_cleanup_every,
        "mapper_token_contract": "v3_event_groups",
        "dataset": _json_safe(dataset_report),
    }


def _normalized_section(source: object, *, allowed: set[str], name: str) -> dict[str, Any]:
    if source is None:
        return {}
    if not isinstance(source, dict):
        raise ValueError(f"{name} must be a mapping")
    unknown = sorted(set(source) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} keys: {unknown}")
    return dict(source)


def main(argv: Sequence[str] | None = None) -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, _ = config_parser.parse_known_args(argv)
    config_defaults = load_run_config(config_args.config) if config_args.config is not None else {
        "model": {},
        "control_model": {},
        "loss": {},
    }
    model_defaults = config_defaults["model"]
    control_model_defaults = config_defaults["control_model"]
    loss_defaults = config_defaults["loss"]

    parser = argparse.ArgumentParser(description="Train the Stage 2 mapper v3 event-token Phase B model.")
    parser.add_argument("--config", default=config_args.config)
    parser.add_argument("--dataset-root", default=config_defaults.get("dataset_root", "dataset"))
    parser.add_argument("--index-path", default=config_defaults.get("index_path"))
    parser.add_argument("--eval-index-path", default=config_defaults.get("eval_index_path"))
    parser.add_argument("--control-v3-timeseries-path", default=config_defaults.get("control_v3_timeseries_path"))
    parser.add_argument("--output-dir", default=config_defaults.get("output_dir"))
    parser.add_argument("--max-steps", type=int, default=config_defaults.get("max_steps"))
    parser.add_argument("--eval-every", type=int, default=config_defaults.get("eval_every"))
    parser.add_argument("--save-every", type=int, default=config_defaults.get("save_every"))
    parser.add_argument("--log-every", type=int, default=config_defaults.get("log_every"))
    parser.add_argument("--batch-size", type=int, default=config_defaults.get("batch_size", 2))
    parser.add_argument("--learning-rate", type=float, default=config_defaults.get("learning_rate", 2e-4))
    parser.add_argument("--weight-decay", type=float, default=config_defaults.get("weight_decay", 0.01))
    parser.add_argument("--seed", type=int, default=config_defaults.get("seed", 1337))
    parser.add_argument("--device", default=config_defaults.get("device", "auto"), choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--run-name", default=config_defaults.get("run_name", "mapper_v3_phase_b_event_global"))
    parser.add_argument("--init-from-control-checkpoint", default=config_defaults.get("init_from_control_checkpoint"))
    parser.add_argument("--resume-from", default=config_defaults.get("resume_from"))
    parser.add_argument("--eval-fraction", type=float, default=config_defaults.get("eval_fraction", 0.1))
    parser.add_argument("--eval-size", type=int, default=config_defaults.get("eval_size"))
    parser.add_argument(
        "--final-train-eval-size",
        type=int,
        default=config_defaults.get("final_train_eval_size", DEFAULT_FINAL_TRAIN_EVAL_SIZE),
    )
    parser.add_argument("--num-workers", type=int, default=config_defaults.get("num_workers", 0))
    parser.add_argument("--max-cached-maps", type=int, default=config_defaults.get("max_cached_maps"))
    parser.add_argument(
        "--dataset-progress",
        action=argparse.BooleanOptionalAction,
        default=bool(config_defaults.get("dataset_progress", False)),
    )
    parser.add_argument("--mapper-record-cache-path", default=config_defaults.get("mapper_record_cache_path"))
    parser.add_argument("--control-teacher-cache-dir", default=config_defaults.get("control_teacher_cache_dir"))
    parser.add_argument(
        "--precompute-control-teacher-cache",
        action="store_true",
        default=bool(config_defaults.get("precompute_control_teacher_cache", False)),
    )
    parser.add_argument(
        "--precompute-control-teacher-cache-only",
        action="store_true",
        default=bool(config_defaults.get("precompute_control_teacher_cache_only", False)),
    )
    parser.add_argument(
        "--control-teacher-precompute-batch-size",
        type=int,
        default=config_defaults.get("control_teacher_precompute_batch_size"),
    )
    parser.add_argument(
        "--control-teacher-cache-overwrite",
        action="store_true",
        default=bool(config_defaults.get("control_teacher_cache_overwrite", False)),
    )
    parser.add_argument(
        "--require-control-teacher-cache",
        action="store_true",
        default=bool(config_defaults.get("require_control_teacher_cache", False)),
    )
    parser.add_argument(
        "--include-full-song-context",
        action=argparse.BooleanOptionalAction,
        default=bool(config_defaults.get("include_full_song_context", True)),
    )
    parser.add_argument(
        "--skip-first-eval-pass",
        action=argparse.BooleanOptionalAction,
        default=bool(config_defaults.get("skip_first_eval_pass", False)),
    )
    parser.add_argument("--mps-cleanup-every", type=int, default=config_defaults.get("mps_cleanup_every"))
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_DIR.as_posix()
    if args.max_steps is None:
        args.max_steps = 5000
    if args.eval_every is None:
        args.eval_every = 100

    init_from = Path(args.init_from_control_checkpoint) if args.init_from_control_checkpoint is not None else None
    resume_from = Path(args.resume_from) if args.resume_from is not None else None
    if args.precompute_control_teacher_cache_only:
        result = precompute_mapper_tuple_phase_b_control_teacher_cache(
            dataset_root=Path(args.dataset_root),
            index_path=Path(args.index_path) if args.index_path is not None else None,
            eval_index_path=Path(args.eval_index_path) if args.eval_index_path is not None else None,
            control_v3_timeseries_path=(
                Path(args.control_v3_timeseries_path) if args.control_v3_timeseries_path is not None else None
            ),
            batch_size=args.batch_size,
            seed=args.seed,
            device_name=args.device,
            init_from_control_checkpoint=init_from,
            num_workers=args.num_workers,
            max_cached_maps=args.max_cached_maps,
            dataset_progress=args.dataset_progress,
            control_teacher_cache_dir=(
                Path(args.control_teacher_cache_dir) if args.control_teacher_cache_dir is not None else None
            ),
            control_teacher_precompute_batch_size=args.control_teacher_precompute_batch_size,
            control_teacher_cache_overwrite=args.control_teacher_cache_overwrite,
            control_model_config_overrides=control_model_defaults,
        )
        for report in result.reports:
            print(
                "control_teacher_cache_report "
                f"split={report['split']} total={report['total_entries']} "
                f"computed={report['computed_entries']} skipped={report['skipped_entries']} "
                f"elapsed_s={float(report['elapsed_s']):.1f}",
            )
        return

    result = run_mapper_v3_phase_b_training(
        dataset_root=Path(args.dataset_root),
        index_path=Path(args.index_path) if args.index_path is not None else None,
        eval_index_path=Path(args.eval_index_path) if args.eval_index_path is not None else None,
        control_v3_timeseries_path=(
            Path(args.control_v3_timeseries_path) if args.control_v3_timeseries_path is not None else None
        ),
        output_dir=Path(args.output_dir),
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        save_every=args.save_every,
        log_every=args.log_every,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device_name=args.device,
        run_name=args.run_name,
        init_from_control_checkpoint=init_from,
        resume_from=resume_from,
        eval_fraction=args.eval_fraction,
        eval_size=args.eval_size,
        final_train_eval_size=args.final_train_eval_size,
        num_workers=args.num_workers,
        max_cached_maps=args.max_cached_maps,
        dataset_progress=args.dataset_progress,
        mapper_record_cache_path=Path(args.mapper_record_cache_path) if args.mapper_record_cache_path is not None else None,
        control_teacher_cache_dir=Path(args.control_teacher_cache_dir) if args.control_teacher_cache_dir is not None else None,
        require_control_teacher_cache=args.require_control_teacher_cache,
        precompute_control_teacher_cache=args.precompute_control_teacher_cache,
        control_teacher_precompute_batch_size=args.control_teacher_precompute_batch_size,
        control_teacher_cache_overwrite=args.control_teacher_cache_overwrite,
        include_full_song_context=args.include_full_song_context,
        skip_first_eval_pass=args.skip_first_eval_pass,
        mps_cleanup_every=args.mps_cleanup_every,
        model_config_overrides=model_defaults,
        control_model_config_overrides=control_model_defaults,
        loss_config_overrides=loss_defaults,
    )
    print(
        "mapper_v3_training_done "
        f"steps={result.completed_steps} final_loss={result.final_loss:.6f} "
        f"report={result.report_path.as_posix()} checkpoint={result.checkpoint_path.as_posix()}",
        flush=True,
    )


__all__ = [
    "load_run_config",
    "main",
    "metrics_for_loader",
    "run_mapper_v3_phase_b_training",
    "_mapper_v3_training_spec",
]


if __name__ == "__main__":
    main()
