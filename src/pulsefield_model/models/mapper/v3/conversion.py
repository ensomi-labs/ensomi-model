from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pulsefield_model.models.mapper.shared.vocab import KEY_COUNT, LaneAction
from pulsefield_model.models.mapper.v2_1.vocab import MapperV21Vocab

from .vocab import MapperV3Vocab


@dataclass(frozen=True)
class V3ConversionResult:
    token_ids: tuple[int, ...]
    reconstructed_v2_1_token_ids: tuple[int, ...]
    baseline_token_count: int
    event_token_count: int
    multi_lane_event_count: int
    lane_action_token_count: int

    @property
    def token_reduction(self) -> int:
        return self.baseline_token_count - len(self.token_ids)


def v2_1_tokens_to_v3_event_tokens(
    token_ids: Sequence[int],
    *,
    source_vocab: MapperV21Vocab | None = None,
    target_vocab: MapperV3Vocab | None = None,
) -> V3ConversionResult:
    source = MapperV21Vocab() if source_vocab is None else source_vocab
    target = MapperV3Vocab(source.time_shift_values_ms) if target_vocab is None else target_vocab
    converted: list[int] = []
    reconstructed: list[int] = []
    event_count = 0
    multi_lane_event_count = 0
    lane_action_token_count = 0

    normalized = tuple(int(token_id) for token_id in token_ids)
    index = 0
    while index < len(normalized):
        token_id = normalized[index]
        if token_id == source.pad_id:
            converted.append(target.pad_id)
            reconstructed.append(source.pad_id)
            index += 1
            continue
        if token_id == source.bos_id:
            converted.append(target.bos_id)
            reconstructed.append(source.bos_id)
            index += 1
            continue
        if token_id == source.eos_id:
            converted.append(target.eos_id)
            reconstructed.append(source.eos_id)
            index += 1
            continue
        if source.is_time_shift_token(token_id):
            converted.append(target.time_shift_token_id(source.time_shift_value(token_id)))
            reconstructed.append(token_id)
            index += 1
            continue
        if not source.is_lane_action_token(token_id):
            raise ValueError(f"unsupported v2.1 token id: {token_id}")

        start_index = index
        while index < len(normalized) and source.is_lane_action_token(normalized[index]):
            index += 1
        run_ids = source.validate_canonical_lane_action_run(normalized[start_index:index])
        actions = [LaneAction.NONE] * KEY_COUNT
        for run_id in run_ids:
            lane_index, action = source.decode_lane_action(run_id)
            actions[int(lane_index)] = action
        converted.append(target.encode_event(actions))
        reconstructed.extend(source.canonical_lane_action_token_ids(actions))
        event_count += 1
        lane_action_token_count += len(run_ids)
        if len(run_ids) > 1:
            multi_lane_event_count += 1

    return V3ConversionResult(
        token_ids=tuple(converted),
        reconstructed_v2_1_token_ids=tuple(reconstructed),
        baseline_token_count=len(normalized),
        event_token_count=event_count,
        multi_lane_event_count=multi_lane_event_count,
        lane_action_token_count=lane_action_token_count,
    )


def v3_event_tokens_to_v2_1_tokens(
    token_ids: Sequence[int],
    *,
    source_vocab: MapperV3Vocab | None = None,
    target_vocab: MapperV21Vocab | None = None,
) -> tuple[int, ...]:
    source = MapperV3Vocab() if source_vocab is None else source_vocab
    target = MapperV21Vocab(source.time_shift_values_ms) if target_vocab is None else target_vocab
    converted: list[int] = []
    for token_id in (int(value) for value in token_ids):
        if token_id == source.pad_id:
            converted.append(target.pad_id)
        elif token_id == source.bos_id:
            converted.append(target.bos_id)
        elif token_id == source.eos_id:
            converted.append(target.eos_id)
        elif source.is_time_shift_token(token_id):
            converted.append(target.time_shift_token_id(source.time_shift_value(token_id)))
        elif source.is_event_token(token_id):
            converted.extend(target.canonical_lane_action_token_ids(source.decode_event(token_id)))
        else:
            raise ValueError(f"unsupported v3 token id: {token_id}")
    return tuple(converted)


def token_names(token_ids: Sequence[int], *, vocab: MapperV3Vocab) -> tuple[str, ...]:
    return tuple(vocab.token_name(int(token_id)) for token_id in token_ids)
