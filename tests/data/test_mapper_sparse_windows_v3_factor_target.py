from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pulsefield_model.data.control_windows import ControlWindowRecord, normalize_difficulty
from pulsefield_model.data.mapper_sparse_windows_v3 import MapperV3WindowDataset, collate_mapper_v3_windows
from pulsefield_model.features.control_v3_targets import MODEL_FEATURE_NAMES
from pulsefield_model.models.mapper.v3 import MapperTimepoint, MapperV3Vocab
from pulsefield_model.models.mapper.v3.factor_target import END_KIND, EVENT_KIND
from pulsefield_model.models.mapper.v3.vocab import LaneAction


def test_mapper_v3_dataset_emits_factor_target_only_when_enabled() -> None:
    vocab = MapperV3Vocab()
    default_off = _MapperV3DatasetWithFullInputs(
        [_record("v3-factor.osu", difficulty=4.0)],
        timepoints_by_path={
            "v3-factor.osu": (
                MapperTimepoint(100, _actions(LaneAction.TAP)),
                MapperTimepoint(180, _actions(LaneAction.NONE, LaneAction.TAP)),
            ),
        },
        vocab=vocab,
    )
    default_on = _MapperV3DatasetWithFullInputs(
        [_record("v3-factor.osu", difficulty=4.0)],
        timepoints_by_path=default_off.timepoints_by_path,
        vocab=vocab,
        include_delta_event_factor_target=True,
    )

    off_sample = default_off[0]
    on_sample = default_on[0]
    off_batch = collate_mapper_v3_windows([off_sample], pad_id=vocab.pad_id)
    on_batch = collate_mapper_v3_windows([on_sample], pad_id=vocab.pad_id)

    assert "delta_event_factor_target" not in off_sample
    assert "delta_event_factor_target" not in off_batch
    assert on_sample["metadata"]["mapper_token_contract"] == "v3_event_groups"
    factor = on_batch["delta_event_factor_target"]
    assert factor["target_kind"].tolist() == [[EVENT_KIND, EVENT_KIND, END_KIND]]
    assert factor["row_count"].tolist() == [3]
    assert factor["event_row_count"].tolist() == [2]
    assert factor["end_row_count"].tolist() == [1]
    assert factor["row_mask"].tolist() == [[True, True, True]]
    assert factor["reconstruction_mismatch_count"].tolist() == [0]


def test_mapper_v3_collate_rejects_partial_factor_target_batch() -> None:
    vocab = MapperV3Vocab()
    dataset = _MapperV3DatasetWithFullInputs(
        [_record("v3-partial.osu", difficulty=4.0)],
        timepoints_by_path={
            "v3-partial.osu": (
                MapperTimepoint(100, _actions(LaneAction.TAP)),
            ),
        },
        vocab=vocab,
        include_delta_event_factor_target=True,
    )
    sample_with_target = dataset[0]
    sample_without_target = {key: value for key, value in sample_with_target.items() if key != "delta_event_factor_target"}

    with pytest.raises(ValueError, match="partial mapper v3 delta-event factor target batch"):
        collate_mapper_v3_windows([sample_with_target, sample_without_target], pad_id=vocab.pad_id)


def _actions(*actions: LaneAction) -> tuple[LaneAction, ...]:
    padded = list(actions)
    while len(padded) < 4:
        padded.append(LaneAction.NONE)
    return tuple(padded)


def _record(
    beatmap_path: str,
    *,
    difficulty: float,
    frame_count: int = 400,
    target_start_frame: int = 0,
) -> ControlWindowRecord:
    return ControlWindowRecord(
        beatmap_path=Path(beatmap_path),
        audio_path=Path(f"{beatmap_path}.mp3"),
        difficulty=difficulty,
        frame_count=frame_count,
        target_start_frame=target_start_frame,
    )


class _FullInputControlDataset:
    def __init__(self, records: list[ControlWindowRecord]) -> None:
        self.records = records

    def __getitem__(self, index: int):
        record = self.records[index]
        return {
            "full_mel": torch.ones(record.frame_count, 160, dtype=torch.float32),
            "full_dense_timing_v2": torch.ones(record.frame_count, 4, dtype=torch.float32),
            "frame_count": torch.tensor(record.frame_count, dtype=torch.long),
            "target_start_frame": torch.tensor(record.target_start_frame, dtype=torch.long),
            "difficulty": torch.tensor(record.difficulty, dtype=torch.float32),
            "normalized_difficulty": torch.tensor(normalize_difficulty(record.difficulty), dtype=torch.float32),
        }

    def target_loader(self, record: ControlWindowRecord) -> torch.Tensor:
        target = torch.zeros(100, len(MODEL_FEATURE_NAMES), dtype=torch.float32)
        target[:, MODEL_FEATURE_NAMES.index("density_level")] = 0.5
        target[:, MODEL_FEATURE_NAMES.index("density_confidence")] = 1.0
        return target


class _MapperV3DatasetWithFullInputs(MapperV3WindowDataset):
    def __init__(
        self,
        records: list[ControlWindowRecord],
        *,
        timepoints_by_path: dict[str, tuple[MapperTimepoint, ...]],
        **dataset_kwargs,
    ) -> None:
        self.timepoints_by_path = timepoints_by_path
        super().__init__(control_dataset=_FullInputControlDataset(records), **dataset_kwargs)

    def _load_timepoints(self, beatmap_path: Path) -> tuple:
        return self.timepoints_by_path.get(beatmap_path.as_posix(), ())
