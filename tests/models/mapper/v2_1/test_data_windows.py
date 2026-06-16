import json
import tempfile
import unittest

import importlib.util

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("requires torch")
from pathlib import Path

import torch

from pulsefield_model.data.control_windows import ControlWindowRecord, normalize_difficulty
from pulsefield_model.data.mapper_sparse_windows_v2_1 import (
    C3_SIDE_STREAM_METADATA_CONTRACT,
    C3_SIDE_STREAM_TOKEN_PAD_ID,
    MapperV21WindowDataset,
    collate_mapper_v2_1_windows,
    load_c3_side_stream_token_sidecar,
)
from pulsefield_model.features.control_v3_targets import MODEL_FEATURE_NAMES
from pulsefield_model.models.mapper.v2_1.replay import NO_EMITTED_LANE_INDEX, ln_carry_state_tensors
from pulsefield_model.models.mapper.v2_1.tokenizer import MapperTimepoint, encode_mapper_window
from pulsefield_model.models.mapper.v2_1.vocab import LaneAction, MapperV21Vocab


def _actions(*actions: LaneAction) -> tuple[LaneAction, ...]:
    padded = list(actions)
    while len(padded) < 4:
        padded.append(LaneAction.NONE)
    return tuple(padded)


class MapperV21DataWindowTests(unittest.TestCase):
    def test_dataset_and_collate_preserve_sparse_lane_state(self) -> None:
        vocab = MapperV21Vocab()
        dataset = _MapperV21DatasetWithFullInputs(
            [_record("sparse.osu", difficulty=4.0)],
            timepoints_by_path={
                "sparse.osu": (
                    MapperTimepoint(1000, _actions(LaneAction.TAP, LaneAction.NONE, LaneAction.TAP)),
                ),
            },
        )
        sparse_sample = dataset[0]
        empty_sample = _sample(
            encode_mapper_window([], vocab=vocab, write_start_ms=8000, write_end_ms=16000),
        )

        batch = collate_mapper_v2_1_windows([sparse_sample, empty_sample], pad_id=vocab.pad_id)

        self.assertNotIn("c3_side_stream", sparse_sample["metadata"])
        self.assertNotIn("c3_side_stream_tokens", sparse_sample)
        self.assertNotIn("c3_side_stream_tokens", sparse_sample["metadata"])
        self.assertNotIn("c3_side_stream_tokens", batch)
        token_names = [vocab.token_name(token_id) for token_id in sparse_sample["target_fragment_tokens"][:3].tolist()]
        self.assertEqual(token_names, ["TS_1000", "LANE_1_TAP", "LANE_3_TAP"])
        self.assertEqual(int(sparse_sample["chart_end_ms"].item()), 1000)
        self.assertTrue(sparse_sample["is_full_chart_end"].item())
        self.assertEqual(vocab.token_name(int(sparse_sample["target_fragment_tokens"][-1].item())), "EOS")
        self.assertTrue(sparse_sample["target_fragment_states"]["emitted_lane_mask"][2, 0].item())
        self.assertEqual(int(sparse_sample["target_fragment_states"]["last_lane_index"][2].item()), 0)

        states = batch["target_fragment_states"]
        self.assertEqual(states["emitted_lane_mask"].shape, (2, sparse_sample["target_fragment_tokens"].shape[0], 4))
        self.assertTrue(states["emitted_lane_mask"][0, 2, 0].item())
        self.assertEqual(int(states["last_lane_index"][0, 2].item()), 0)
        self.assertFalse(states["emitted_lane_mask"][1, -1].any().item())
        self.assertEqual(int(states["last_lane_index"][1, -1].item()), NO_EMITTED_LANE_INDEX)
        self.assertFalse(batch["target_fragment_mask"][1, -1].item())
        self.assertEqual(tuple(batch["density_target_8s"].shape), (2, 400, 1))

    def test_c3_side_stream_shadow_metadata_is_opt_in(self) -> None:
        summary = {
            "selected_span_count": 7,
            "selected_fallback_literal_count": 23,
            "noncontiguous_main_stream_span_count": 4,
        }
        dataset = _MapperV21DatasetWithFullInputs(
            [_record("shadow.osu", difficulty=5.0, target_start_frame=400)],
            timepoints_by_path={
                "shadow.osu": (
                    MapperTimepoint(9000, _actions(LaneAction.TAP)),
                ),
            },
            include_c3_side_stream_metadata=True,
            c3_side_stream_metadata_by_beatmap_path={"shadow.osu": summary},
        )

        sample = dataset[0]
        batch = collate_mapper_v2_1_windows([sample], pad_id=MapperV21Vocab().pad_id)
        c3_metadata = sample["metadata"]["c3_side_stream"]

        self.assertEqual(c3_metadata["contract"], C3_SIDE_STREAM_METADATA_CONTRACT)
        self.assertEqual(c3_metadata["beatmap_path"], "shadow.osu")
        self.assertEqual(c3_metadata["window_start_ms"], 8000)
        self.assertEqual(c3_metadata["window_end_ms"], 16000)
        self.assertTrue(c3_metadata["summary_available"])
        self.assertEqual(c3_metadata["summary"], summary)
        self.assertEqual(batch["metadata"][0]["c3_side_stream"], c3_metadata)

    def test_c3_side_stream_token_sidecar_is_bounded_and_collated(self) -> None:
        dataset = _MapperV21DatasetWithFullInputs(
            [
                _record("side.osu", difficulty=4.0),
                _record("side.osu", difficulty=4.0, target_start_frame=400),
            ],
            timepoints_by_path={
                "side.osu": (
                    MapperTimepoint(1000, _actions(LaneAction.TAP)),
                    MapperTimepoint(9000, _actions(LaneAction.TAP)),
                ),
            },
            include_c3_side_stream_token_tensors=True,
            c3_side_stream_token_ids_by_beatmap_path={
                "side.osu": {
                    0: (11, 12, 13),
                    "8000": (21, 22, 23, 24, 25),
                },
            },
            c3_side_stream_max_tokens=4,
        )

        first = dataset[0]
        second = dataset[1]
        batch = collate_mapper_v2_1_windows([first, second], pad_id=MapperV21Vocab().pad_id)

        self.assertEqual(first["c3_side_stream_tokens"].tolist(), [11, 12, 13])
        self.assertEqual(first["c3_side_stream_token_mask"].tolist(), [True, True, True])
        self.assertTrue(first["c3_side_stream_available"].item())
        self.assertEqual(int(first["c3_side_stream_token_count"].item()), 3)
        self.assertFalse(first["c3_side_stream_truncated"].item())

        self.assertEqual(second["c3_side_stream_tokens"].tolist(), [21, 22, 23, 24])
        self.assertEqual(int(second["c3_side_stream_token_count"].item()), 5)
        self.assertTrue(second["c3_side_stream_truncated"].item())
        self.assertTrue(second["metadata"]["c3_side_stream_tokens"]["available"])
        self.assertEqual(second["metadata"]["c3_side_stream_tokens"]["token_count"], 5)
        self.assertEqual(second["metadata"]["c3_side_stream_tokens"]["clipped_token_count"], 4)
        self.assertEqual(second["metadata"]["c3_side_stream_tokens"]["pad_id"], C3_SIDE_STREAM_TOKEN_PAD_ID)

        self.assertEqual(batch["c3_side_stream_tokens"].tolist(), [[11, 12, 13, 0], [21, 22, 23, 24]])
        self.assertEqual(batch["c3_side_stream_token_mask"].tolist(), [[True, True, True, False], [True, True, True, True]])
        self.assertEqual(batch["c3_side_stream_available"].tolist(), [True, True])
        self.assertEqual(batch["c3_side_stream_token_count"].tolist(), [3, 5])
        self.assertEqual(batch["c3_side_stream_truncated"].tolist(), [False, True])

    def test_c3_side_stream_token_sidecar_missing_window_marks_unavailable(self) -> None:
        dataset = _MapperV21DatasetWithFullInputs(
            [_record("missing.osu", difficulty=4.0)],
            timepoints_by_path={
                "missing.osu": (
                    MapperTimepoint(1000, _actions(LaneAction.TAP)),
                ),
            },
            include_c3_side_stream_token_tensors=True,
            c3_side_stream_token_ids_by_beatmap_path={"missing.osu": {8000: (41, 42)}},
        )

        sample = dataset[0]
        batch = collate_mapper_v2_1_windows([sample], pad_id=MapperV21Vocab().pad_id)

        self.assertEqual(sample["c3_side_stream_tokens"].tolist(), [])
        self.assertFalse(sample["c3_side_stream_available"].item())
        self.assertFalse(sample["metadata"]["c3_side_stream_tokens"]["available"])
        self.assertEqual(sample["metadata"]["c3_side_stream_tokens"]["token_count"], 0)
        self.assertEqual(tuple(batch["c3_side_stream_tokens"].shape), (1, 0))
        self.assertEqual(batch["c3_side_stream_available"].tolist(), [False])

    def test_c3_side_stream_token_sidecar_json_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sidecar_path = Path(temp_dir) / "c3_sidecar.json"
            sidecar_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "contract": C3_SIDE_STREAM_METADATA_CONTRACT,
                        "windows": [
                            {
                                "beatmap_path": "json.osu",
                                "window_start_ms": 8000,
                                "token_ids": [31, 32, 33],
                            },
                        ],
                    },
                ),
                encoding="utf-8",
            )

            loaded = load_c3_side_stream_token_sidecar(sidecar_path)

        self.assertEqual(loaded, {"json.osu": {8000: (31, 32, 33)}})


def _sample(tokenized) -> dict:
    return {
        "mel_context": torch.zeros(400, 160, dtype=torch.float32),
        "timing_context": torch.zeros(400, 4, dtype=torch.float32),
        "context_padding_mask": torch.zeros(400, dtype=torch.bool),
        "difficulty": torch.zeros(1, dtype=torch.float32),
        "decoder_input_tokens": tokenized.decoder_input_tensor(),
        "target_fragment_tokens": tokenized.target_fragment_tensor(),
        "target_fragment_states": {
            "current_ms": tokenized.target_fragment_current_ms,
            "open_mask": tokenized.target_fragment_open_mask,
            "open_start_ms": tokenized.target_fragment_open_start_ms,
            "open_age_ms": tokenized.target_fragment_open_age_ms,
            "emitted_lane_mask": tokenized.target_fragment_emitted_lane_mask,
            "last_lane_index": tokenized.target_fragment_last_lane_index,
        },
        "ln_carry_in": ln_carry_state_tensors(tokenized.ln_carry_in),
        "ln_carry_out": ln_carry_state_tensors(tokenized.ln_carry_out),
        "close_labels": tokenized.close_labels,
        "close_label_mask": tokenized.close_label_mask,
        "density_target_8s": torch.zeros(400, 1, dtype=torch.float32),
        "density_confidence_8s": torch.ones(400, 1, dtype=torch.float32),
        "write_start_ms": torch.tensor(tokenized.write_start_ms, dtype=torch.long),
        "write_end_ms": torch.tensor(tokenized.write_end_ms, dtype=torch.long),
        "chart_end_ms": torch.tensor(tokenized.chart_end_ms, dtype=torch.long),
        "is_full_chart_start": torch.tensor(tokenized.is_full_chart_start, dtype=torch.bool),
        "is_full_chart_end": torch.tensor(tokenized.is_full_chart_end, dtype=torch.bool),
    }


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


class _MapperV21DatasetWithFullInputs(MapperV21WindowDataset):
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
