from __future__ import annotations

from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from pulsefield_model.models.control import ControlDemoGlobalEncoder, ControlDemoGlobalEncoderConfig
from pulsefield_model.models.mapper.v3 import MapperV3Config, MapperV3Model, MapperV3ModelLoss
from pulsefield_model.models.mapper.v3.loss import MapperV3LossConfig
from pulsefield_model.training.common import _metric_is_count
from pulsefield_model.training.mapper_common import (
    _build_mapper_tuple_optimizer,
    _cleanup_mps_training_memory,
    _move_mapper_batch_tensors,
)
from pulsefield_model.training.mapper_runner import (
    MapperTrainingSpec,
    default_mapper_metric_finalizer,
    mapper_metrics_for_loader,
    resume_resumable_loader_cursor_or_advance,
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
        "dataset": dict(dataset_report),
    }


__all__ = [
    "metrics_for_loader",
    "_mapper_v3_training_spec",
]
