from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from pulsefield_model.models.mapper.shared.batch import MapperTokenContract as _MapperTokenContract
from pulsefield_model.models.mapper.shared.loss import (
    MapperTupleLossConfig as _MapperTupleLossConfig,
    MapperTupleLossOutput as _MapperTupleLossOutput,
    MapperTupleModelLoss as _MapperTupleModelLoss,
)

from .vocab import MapperV21Vocab as _MapperV21Vocab


@dataclass(frozen=True)
class MapperV21LossConfig(_MapperTupleLossConfig):
    lambda_c3_auxiliary: float = 0.0


@dataclass(frozen=True)
class MapperV21LossOutput(_MapperTupleLossOutput):
    pass


class MapperV21ModelLoss(_MapperTupleModelLoss):
    def __init__(self, config: MapperV21LossConfig | None = None, *, vocab: _MapperV21Vocab | None = None) -> None:
        resolved_config = MapperV21LossConfig() if config is None else config
        lambda_c3_auxiliary = resolved_config.lambda_c3_auxiliary
        if (
            not isinstance(lambda_c3_auxiliary, (int, float))
            or isinstance(lambda_c3_auxiliary, bool)
            or not math.isfinite(float(lambda_c3_auxiliary))
        ):
            raise ValueError("lambda_c3_auxiliary must be finite numeric")
        if float(lambda_c3_auxiliary) < 0.0:
            raise ValueError("lambda_c3_auxiliary must be non-negative")
        resolved_vocab = _MapperV21Vocab() if vocab is None else vocab
        super().__init__(
            resolved_config,
            vocab=resolved_vocab,
            token_contract=_MapperTokenContract(
                name="v2.1",
                vocab=resolved_vocab,
                requires_sparse_lane_state=True,
                uses_chart_end_for_terminal_windows=True,
            ),
        )
        self.config = resolved_config
        self.vocab = resolved_vocab

    def forward(self, output: Any, batch: Mapping[str, torch.Tensor]) -> MapperV21LossOutput:
        loss = super().forward(output, batch)
        c3_auxiliary_loss = loss.total_loss.new_zeros(())
        metrics = dict(loss.metrics)
        metric_numerators = dict(loss.metric_numerators)
        metric_denominators = dict(loss.metric_denominators)
        total_loss = loss.total_loss
        if float(self.config.lambda_c3_auxiliary) > 0.0:
            c3_auxiliary_loss, c3_metrics, c3_weight = _c3_auxiliary_bag_loss(output, batch)
            total_loss = total_loss + float(self.config.lambda_c3_auxiliary) * c3_auxiliary_loss
            metrics.update(c3_metrics)
            metrics["phase/lambda_c3_auxiliary"] = float(self.config.lambda_c3_auxiliary)
            metrics["loss/total"] = float(total_loss.detach().cpu())
            metric_numerators["loss/c3_auxiliary"] = float((c3_auxiliary_loss.detach() * c3_weight.clamp_min(1)).cpu())
            metric_denominators["loss/c3_auxiliary"] = float(c3_weight.detach().cpu())
        else:
            metrics["phase/lambda_c3_auxiliary"] = float(self.config.lambda_c3_auxiliary)
        return MapperV21LossOutput(
            total_loss=total_loss,
            token_loss=loss.token_loss,
            ln_close_loss=loss.ln_close_loss,
            density_loss=loss.density_loss,
            adapter_reg_loss=loss.adapter_reg_loss,
            metrics=metrics,
            metric_numerators=metric_numerators,
            metric_denominators=metric_denominators,
        )


def _c3_auxiliary_bag_loss(
    output: Any,
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    logits = getattr(output, "c3_auxiliary_logits", None)
    if not isinstance(logits, torch.Tensor):
        raise ValueError("c3_auxiliary_logits are required when lambda_c3_auxiliary > 0")
    if logits.ndim != 2:
        raise ValueError(f"c3_auxiliary_logits must have shape [B,V], got {tuple(logits.shape)}")
    token_source = batch.get("c3_side_stream_tokens")
    if not isinstance(token_source, torch.Tensor):
        raise ValueError("c3_side_stream_tokens are required when lambda_c3_auxiliary > 0")
    tokens = token_source.to(device=logits.device, dtype=torch.long)
    if tokens.ndim != 2 or int(tokens.shape[0]) != int(logits.shape[0]):
        raise ValueError(f"c3_side_stream_tokens must have shape [B,T], got {tuple(tokens.shape)}")
    mask_source = batch.get("c3_side_stream_token_mask")
    if mask_source is None:
        token_mask = tokens.ne(0)
    else:
        if not isinstance(mask_source, torch.Tensor):
            raise ValueError("c3_side_stream_token_mask must be a torch.Tensor")
        token_mask = mask_source.to(device=logits.device, dtype=torch.bool)
        if tuple(token_mask.shape) != tuple(tokens.shape):
            raise ValueError("c3_side_stream_token_mask must match c3_side_stream_tokens")
        token_mask = token_mask & tokens.ne(0)
    if tokens.numel():
        valid_tokens = tokens[token_mask]
        if valid_tokens.numel():
            min_token = int(valid_tokens.min().item())
            max_token = int(valid_tokens.max().item())
            if min_token <= 0 or max_token > int(logits.shape[1]):
                raise ValueError(
                    "c3_side_stream_tokens must be between 1 and "
                    f"c3_auxiliary_vocab_size={int(logits.shape[1])}, "
                    f"got min={min_token} max={max_token}"
                )
    available_source = batch.get("c3_side_stream_available")
    if available_source is None:
        sample_mask = torch.ones((int(logits.shape[0]),), device=logits.device, dtype=logits.dtype)
    else:
        if not isinstance(available_source, torch.Tensor):
            raise ValueError("c3_side_stream_available must be a torch.Tensor")
        available = available_source.to(device=logits.device, dtype=torch.bool).reshape(-1)
        if tuple(available.shape) != (int(logits.shape[0]),):
            raise ValueError(f"c3_side_stream_available must have shape [{int(logits.shape[0])}]")
        sample_mask = available.to(dtype=logits.dtype)

    target = torch.zeros_like(logits)
    valid_positions = token_mask.nonzero(as_tuple=False)
    if int(valid_positions.shape[0]):
        batch_index = valid_positions[:, 0]
        token_index = tokens[batch_index, valid_positions[:, 1]] - 1
        target[batch_index, token_index] = 1.0
    per_sample_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(dim=1)
    sample_weight = sample_mask.sum().clamp_min(1.0)
    loss = (per_sample_loss * sample_mask).sum() / sample_weight
    positive_count = target.sum()
    raw_token_count = token_mask.to(dtype=logits.dtype).sum()
    metrics = {
        "loss/c3_auxiliary": float(loss.detach().cpu()),
        "c3_auxiliary/sample_count": float(sample_mask.sum().detach().cpu()),
        "c3_auxiliary/token_count": float(raw_token_count.detach().cpu()),
        "c3_auxiliary/positive_label_count": float(positive_count.detach().cpu()),
    }
    return loss, metrics, sample_weight


__all__ = [
    "MapperV21LossConfig",
    "MapperV21LossOutput",
    "MapperV21ModelLoss",
]
