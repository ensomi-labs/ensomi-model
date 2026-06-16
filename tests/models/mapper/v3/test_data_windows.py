import unittest
from pathlib import Path

import importlib.util

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("requires torch")

import torch

from pulsefield_model.data.control_windows import ControlWindowRecord, normalize_difficulty
from pulsefield_model.data.mapper_sparse_windows_v3 import MapperV3WindowDataset, collate_mapper_v3_windows
from pulsefield_model.features.control_v3_targets import MODEL_FEATURE_NAMES
from pulsefield_model.models.mapper.v3 import MapperTimepoint, MapperV3Vocab
from pulsefield_model.models.mapper.v3.vocab import LaneAction


def _actions(*actions: LaneAction) -> tuple[LaneAction, ...]:
    padded = list(actions)
    while len(padded) < 4:
        padded.append(LaneAction.NONE)
    return tuple(padded)


class MapperV3DataWindowTests(unittest.TestCase):
    def test_dataset_and_collate_emit_event_tokens_without_sparse_state(self) -> None:
        vocab = MapperV3Vocab()
        dataset = _MapperV3DatasetWithFullInputs(
            [_record("v3.osu", difficulty=4.0)],
            timepoints_by_path={
                "v3.osu": (
                    MapperTimepoint(1000, _actions(LaneAction.TAP, LaneAction.NONE, LaneAction.TAP)),
                ),
            },
        )

        sample = dataset[0]
        batch = collate_mapper_v3_windows([sample], pad_id=vocab.pad_id)

        token_ids = sample["target_fragment_tokens"].tolist()
        self.assertEqual(vocab.token_name(token_ids[0]), "TS_1000")
        self.assertEqual(vocab.event_signature(token_ids[1]), "T.T.")
        self.assertEqual(vocab.token_name(token_ids[2]), "EOS")
        self.assertEqual(int(sample["chart_end_ms"].item()), 1000)
        self.assertEqual(sample["metadata"]["mapper_token_contract"], "v3_event_groups")
        self.assertNotIn("emitted_lane_mask", sample["target_fragment_states"])
        self.assertNotIn("last_lane_index", sample["target_fragment_states"])

        states = batch["target_fragment_states"]
        self.assertEqual(tuple(states["open_mask"].shape), (1, sample["target_fragment_tokens"].shape[0], 4))
        self.assertNotIn("emitted_lane_mask", states)
        self.assertNotIn("last_lane_index", states)
        self.assertEqual(batch["chart_end_ms"].tolist(), [1000])
        self.assertEqual(tuple(batch["density_target_8s"].shape), (1, 400, 1))


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


if __name__ == "__main__":
    unittest.main()
