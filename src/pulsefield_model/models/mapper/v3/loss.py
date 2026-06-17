from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from pulsefield_model.models.mapper.shared.batch import MapperTokenContract
from pulsefield_model.models.mapper.shared.loss import (
    MapperTupleLossConfig,
    MapperTupleLossOutput,
    MapperTupleModelLoss,
)

from .vocab import MapperV3Vocab


@dataclass(frozen=True)
class MapperV3LossConfig(MapperTupleLossConfig):
    lambda_delta_event_auxiliary: float = 0.0
    delta_event_end_gap_loss_weight: float = 1.0


@dataclass(frozen=True)
class MapperV3LossOutput(MapperTupleLossOutput):
    delta_event_auxiliary_loss: torch.Tensor | None = None


class MapperV3ModelLoss(MapperTupleModelLoss):
    def __init__(
        self,
        config: MapperTupleLossConfig | MapperV3LossConfig | None = None,
        *,
        vocab: MapperV3Vocab | None = None,
    ) -> None:
        resolved_config = _coerce_v3_loss_config(config)
        _validate_v3_loss_config(resolved_config)
        resolved_vocab = MapperV3Vocab() if vocab is None else vocab
        super().__init__(
            resolved_config,
            vocab=resolved_vocab,
            token_contract=MapperTokenContract(
                name="v3",
                vocab=resolved_vocab,
                uses_chart_end_for_terminal_windows=True,
            ),
        )
        self.config = resolved_config
        self.vocab = resolved_vocab

    def forward(self, output: Any, batch: Mapping[str, torch.Tensor]) -> MapperV3LossOutput:
        loss = super().forward(output, batch)
        delta_event_auxiliary_loss_value = loss.total_loss.new_zeros(())
        total_loss = loss.total_loss
        metrics = dict(loss.metrics)
        metric_numerators = dict(loss.metric_numerators)
        metric_denominators = dict(loss.metric_denominators)
        if float(self.config.lambda_delta_event_auxiliary) > 0.0:
            delta_event_auxiliary_loss_value, aux_metrics, aux_weight = delta_event_auxiliary_loss(
                output,
                batch,
                vocab=self.vocab,
                end_gap_weight=float(self.config.delta_event_end_gap_loss_weight),
            )
            total_loss = total_loss + float(self.config.lambda_delta_event_auxiliary) * delta_event_auxiliary_loss_value
            metrics.update(aux_metrics)
            metrics["phase/lambda_delta_event_auxiliary"] = float(self.config.lambda_delta_event_auxiliary)
            metrics["phase/delta_event_end_gap_loss_weight"] = float(self.config.delta_event_end_gap_loss_weight)
            metrics["loss/total"] = float(total_loss.detach().cpu())
            metric_numerators["loss/delta_event_auxiliary"] = float(
                (delta_event_auxiliary_loss_value.detach() * aux_weight.clamp_min(1)).cpu()
            )
            metric_denominators["loss/delta_event_auxiliary"] = float(aux_weight.detach().cpu())
        else:
            metrics["phase/lambda_delta_event_auxiliary"] = float(self.config.lambda_delta_event_auxiliary)
            metrics["phase/delta_event_end_gap_loss_weight"] = float(self.config.delta_event_end_gap_loss_weight)
        return MapperV3LossOutput(
            total_loss=total_loss,
            token_loss=loss.token_loss,
            ln_close_loss=loss.ln_close_loss,
            density_loss=loss.density_loss,
            event_budget_loss=loss.event_budget_loss,
            conditioned_event_distribution_loss=loss.conditioned_event_distribution_loss,
            continuation_jump_loss=loss.continuation_jump_loss,
            time_shift_distance_loss=loss.time_shift_distance_loss,
            adapter_reg_loss=loss.adapter_reg_loss,
            metrics=metrics,
            metric_numerators=metric_numerators,
            metric_denominators=metric_denominators,
            delta_event_auxiliary_loss=delta_event_auxiliary_loss_value,
        )


def delta_event_auxiliary_loss(
    output: Any,
    batch: Mapping[str, torch.Tensor],
    *,
    vocab: MapperV3Vocab,
    end_gap_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    _require_positive_finite(end_gap_weight, "end_gap_weight")
    delta_logits = getattr(output, "delta_event_delta_logits", None)
    signature_logits = getattr(output, "delta_event_signature_logits", None)
    end_gap_logits = getattr(output, "delta_event_end_gap_logits", None)
    if not isinstance(delta_logits, torch.Tensor):
        raise ValueError("delta_event_delta_logits are required when lambda_delta_event_auxiliary > 0")
    if not isinstance(signature_logits, torch.Tensor):
        raise ValueError("delta_event_signature_logits are required when lambda_delta_event_auxiliary > 0")
    if not isinstance(end_gap_logits, torch.Tensor):
        raise ValueError("delta_event_end_gap_logits are required when lambda_delta_event_auxiliary > 0")
    if delta_logits.ndim != 3:
        raise ValueError(f"delta_event_delta_logits must have shape [B,T,D], got {tuple(delta_logits.shape)}")
    if tuple(signature_logits.shape[:2]) != tuple(delta_logits.shape[:2]) or signature_logits.ndim != 3:
        raise ValueError("delta_event_signature_logits must have shape [B,T,E] matching delta logits")
    if tuple(end_gap_logits.shape[:2]) != tuple(delta_logits.shape[:2]) or end_gap_logits.ndim != 3:
        raise ValueError("delta_event_end_gap_logits must have shape [B,T,G] matching delta logits")
    if int(signature_logits.shape[-1]) != len(vocab.event_token_ids):
        raise ValueError(
            "delta_event_signature_logits width must match v3 event-token count "
            f"({signature_logits.shape[-1]} != {len(vocab.event_token_ids)})"
        )

    target = _require_batch_tensor(batch, "target_fragment_tokens", ndim=2).to(
        device=delta_logits.device,
        dtype=torch.long,
    )
    target_mask = _require_batch_tensor(batch, "target_fragment_mask", ndim=2).to(
        device=delta_logits.device,
        dtype=torch.bool,
    )
    current_ms = _require_fragment_state_tensor(batch, "current_ms").to(device=delta_logits.device, dtype=torch.long)
    if tuple(target.shape) != tuple(delta_logits.shape[:2]):
        raise ValueError("target_fragment_tokens must align with delta_event logits")
    if tuple(target_mask.shape) != tuple(target.shape):
        raise ValueError("target_fragment_mask must align with target_fragment_tokens")
    if tuple(current_ms.shape) != tuple(target.shape):
        raise ValueError("target_fragment_states.current_ms must align with target_fragment_tokens")

    write_start_ms = _require_batch_vector(batch, "write_start_ms", batch_size=int(target.shape[0]), device=delta_logits.device)
    target_end_ms = _target_end_ms(batch, batch_size=int(target.shape[0]), device=delta_logits.device)
    labels = _delta_event_auxiliary_labels(
        target_tokens=target,
        target_mask=target_mask,
        current_ms=current_ms,
        write_start_ms=write_start_ms,
        target_end_ms=target_end_ms,
        vocab=vocab,
        delta_class_count=int(delta_logits.shape[-1]),
        end_gap_class_count=int(end_gap_logits.shape[-1]),
    )
    delta_loss, event_count = _masked_cross_entropy(delta_logits, labels["delta"])
    signature_loss, signature_count = _masked_cross_entropy(signature_logits, labels["signature"])
    end_loss, end_count = _masked_cross_entropy(end_gap_logits, labels["end_gap"])
    event_loss = (delta_loss + signature_loss) * 0.5 if event_count > 0 else delta_logits.reshape(-1)[:0].sum()
    total = event_loss + float(end_gap_weight) * end_loss
    aux_weight = delta_logits.new_tensor(float(event_count) + float(end_gap_weight) * float(end_count))
    metrics = {
        "loss/delta_event_auxiliary": float(total.detach().cpu()),
        "loss/delta_event_delta": float(delta_loss.detach().cpu()),
        "loss/delta_event_signature": float(signature_loss.detach().cpu()),
        "loss/delta_event_end_gap": float(end_loss.detach().cpu()),
        "delta_event_auxiliary/event_label_count": float(event_count),
        "delta_event_auxiliary/signature_label_count": float(signature_count),
        "delta_event_auxiliary/end_gap_label_count": float(end_count),
        "delta_event_auxiliary/delta_class_count": float(delta_logits.shape[-1]),
        "delta_event_auxiliary/signature_class_count": float(signature_logits.shape[-1]),
        "delta_event_auxiliary/end_gap_class_count": float(end_gap_logits.shape[-1]),
    }
    return total, metrics, aux_weight


def _delta_event_auxiliary_labels(
    *,
    target_tokens: torch.Tensor,
    target_mask: torch.Tensor,
    current_ms: torch.Tensor,
    write_start_ms: torch.Tensor,
    target_end_ms: torch.Tensor,
    vocab: MapperV3Vocab,
    delta_class_count: int,
    end_gap_class_count: int,
) -> dict[str, torch.Tensor]:
    ignore_index = -100
    delta_target = torch.full_like(target_tokens, ignore_index)
    signature_target = torch.full_like(target_tokens, ignore_index)
    end_gap_target = torch.full_like(target_tokens, ignore_index)
    event_index_by_token = {int(token_id): index for index, token_id in enumerate(vocab.event_token_ids)}
    batch_size, steps = target_tokens.shape
    for batch_index in range(batch_size):
        anchor_ms = int(write_start_ms[batch_index].item())
        last_valid_step = -1
        for step in range(steps):
            if not bool(target_mask[batch_index, step].item()):
                continue
            last_valid_step = step
            token_id = int(target_tokens[batch_index, step].item())
            if not vocab.is_event_token(token_id):
                continue
            event_ms = int(current_ms[batch_index, step].item())
            delta_ms = event_ms - anchor_ms
            delta_label = _time_grid_label(delta_ms, delta_class_count, name="delta-event delta")
            delta_target[batch_index, step] = int(delta_label)
            signature_target[batch_index, step] = int(event_index_by_token[token_id])
            anchor_ms = event_ms
        if last_valid_step >= 0:
            end_gap_ms = int(target_end_ms[batch_index].item()) - anchor_ms
            end_gap_target[batch_index, last_valid_step] = _time_grid_label(
                end_gap_ms,
                end_gap_class_count,
                name="delta-event end gap",
            )
    return {
        "delta": delta_target,
        "signature": signature_target,
        "end_gap": end_gap_target,
    }


def _masked_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, int]:
    valid = target.ne(-100)
    count = int(valid.sum().detach().cpu())
    if count <= 0:
        return logits.reshape(-1)[:0].sum(), 0
    return F.cross_entropy(logits[valid], target[valid], reduction="mean"), count


def _time_grid_label(delta_ms: int, class_count: int, *, name: str) -> int:
    value = int(delta_ms)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    if value % 10 != 0:
        raise ValueError(f"{name} must be on the 10ms grid, got {value}")
    label = value // 10
    if label >= int(class_count):
        raise ValueError(
            f"{name} {value}ms exceeds auxiliary head range {(int(class_count) - 1) * 10}ms"
        )
    return label


def _target_end_ms(batch: Mapping[str, torch.Tensor], *, batch_size: int, device: torch.device) -> torch.Tensor:
    write_end_ms = _require_batch_vector(batch, "write_end_ms", batch_size=batch_size, device=device)
    chart_end_ms = _require_batch_vector(batch, "chart_end_ms", batch_size=batch_size, device=device)
    is_full_chart_end = _require_batch_tensor(batch, "is_full_chart_end", ndim=1).to(device=device, dtype=torch.bool)
    if tuple(is_full_chart_end.shape) != (int(batch_size),):
        raise ValueError("is_full_chart_end must have shape [B]")
    return torch.where(is_full_chart_end, chart_end_ms, write_end_ms)


def _require_batch_tensor(batch: Mapping[str, torch.Tensor], key: str, *, ndim: int) -> torch.Tensor:
    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"batch[{key!r}] must be a torch.Tensor")
    if value.ndim != ndim:
        raise ValueError(f"batch[{key!r}] must be rank {ndim}, got shape {tuple(value.shape)}")
    return value


def _require_batch_vector(
    batch: Mapping[str, torch.Tensor],
    key: str,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    value = _require_batch_tensor(batch, key, ndim=1).to(device=device, dtype=torch.long)
    if tuple(value.shape) != (int(batch_size),):
        raise ValueError(f"{key} must have shape [B]")
    return value


def _require_fragment_state_tensor(batch: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    states = batch.get("target_fragment_states")
    if not isinstance(states, Mapping):
        raise ValueError("batch['target_fragment_states'] must be a mapping")
    value = states.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"target_fragment_states[{name!r}] must be a torch.Tensor")
    if value.ndim != 2:
        raise ValueError(f"target_fragment_states[{name!r}] must have shape [B,T]")
    return value


def _coerce_v3_loss_config(config: MapperTupleLossConfig | MapperV3LossConfig | None) -> MapperV3LossConfig:
    if config is None:
        return MapperV3LossConfig()
    if isinstance(config, MapperV3LossConfig):
        return config
    if isinstance(config, MapperTupleLossConfig):
        values = {field.name: getattr(config, field.name) for field in fields(MapperTupleLossConfig)}
        return MapperV3LossConfig(**values)
    raise TypeError(f"config must be a MapperV3LossConfig, got {type(config).__name__}")


def _validate_v3_loss_config(config: MapperV3LossConfig) -> None:
    _require_finite(config.lambda_delta_event_auxiliary, "lambda_delta_event_auxiliary")
    _require_positive_finite(config.delta_event_end_gap_loss_weight, "delta_event_end_gap_loss_weight")
    if float(config.lambda_delta_event_auxiliary) < 0.0:
        raise ValueError("lambda_delta_event_auxiliary must be non-negative")


def _require_finite(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite numeric")


def _require_positive_finite(value: float, name: str) -> None:
    _require_finite(value, name)
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive")


__all__ = [
    "MapperV3LossConfig",
    "MapperV3LossOutput",
    "MapperV3ModelLoss",
    "delta_event_auxiliary_loss",
]
