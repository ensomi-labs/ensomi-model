from __future__ import annotations

from pulsefield_model.models.mapper.shared.batch import MapperTokenContract
from pulsefield_model.models.mapper.shared.loss import (
    MapperTupleLossConfig,
    MapperTupleLossOutput,
    MapperTupleModelLoss,
)

from .vocab import MapperV3Vocab


class MapperV3ModelLoss(MapperTupleModelLoss):
    def __init__(self, config: MapperTupleLossConfig | None = None, *, vocab: MapperV3Vocab | None = None) -> None:
        resolved_vocab = MapperV3Vocab() if vocab is None else vocab
        super().__init__(
            config,
            vocab=resolved_vocab,
            token_contract=MapperTokenContract(
                name="v3",
                vocab=resolved_vocab,
                uses_chart_end_for_terminal_windows=True,
            ),
        )
        self.config = MapperTupleLossConfig() if config is None else config
        self.vocab = resolved_vocab


MapperV3LossConfig = MapperTupleLossConfig
MapperV3LossOutput = MapperTupleLossOutput


__all__ = [
    "MapperV3LossConfig",
    "MapperV3LossOutput",
    "MapperV3ModelLoss",
]
