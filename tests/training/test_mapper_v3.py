import tempfile
import unittest

import importlib.util

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("requires torch")
import torch
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pulsefield_model.data.mapper_tuple_windows import MapperTupleWindowFilterReport
from pulsefield_model.models.control import ControlDemoGlobalEncoderConfig
from pulsefield_model.models.mapper.v3 import MapperV3Config, MapperV3LossConfig
from pulsefield_model.training import mapper_v3 as mapper_v3_training
from pulsefield_model.training.mapper_common import _move_mapper_batch_tensors
from pulsefield_model.training.mapper_v3 import load_run_config


_STALE_ROOT = "train" + "/"


class MapperV3PhaseBTrainingTests(unittest.TestCase):
    def test_phase_b_event_global_config_loads_v3_fields(self) -> None:
        config = load_run_config(
            "configs/training/stage2_mapper_v3_phase_b_event_global_mps.yaml",
        )

        self.assertTrue(config["include_full_song_context"])
        self.assertTrue(config["skip_first_eval_pass"])
        self.assertEqual(config["mps_cleanup_every"], 20)
        self.assertEqual(config["batch_size"], 2)
        self.assertEqual(config["model"]["max_seq_len"], 1024)
        self.assertTrue(config["model"]["use_global_context"])
        self.assertEqual(config["loss"]["lambda_density"], 0.05)
        self.assertFalse(config["precompute_control_teacher_cache"])
        self.assertFalse(config["precompute_control_teacher_cache_only"])
        self.assertEqual(config["control_teacher_precompute_batch_size"], 12)
        self.assertFalse(config["control_teacher_cache_overwrite"])
        self.assertIn("stage2_control_windows", config["index_path"])
        self.assertIn("stage2_mapper_v3/window_records", config["mapper_record_cache_path"])
        self.assertIn("stage2_mapper_v2_1/control_teacher", config["control_teacher_cache_dir"])
        _assert_artifacts_paths(
            self,
            config,
            (
                "index_path",
                "control_v3_timeseries_path",
                "output_dir",
                "init_from_control_checkpoint",
                "mapper_record_cache_path",
                "control_teacher_cache_dir",
            ),
        )

        MapperV3Config(**config["model"])
        ControlDemoGlobalEncoderConfig(**config["control_model"])
        MapperV3LossConfig(**config["loss"])

    def test_main_forwards_v3_training_options(self) -> None:
        train_result = SimpleNamespace(
            report_path=Path("report.json"),
            checkpoint_path=Path("checkpoint.pt"),
            final_loss=0.0,
            completed_steps=0,
        )
        with patch.object(
            mapper_v3_training,
            "run_mapper_v3_phase_b_training",
            return_value=train_result,
            autospec=True,
        ) as train:
            mapper_v3_training.main(
                [
                    "--config",
                    "configs/training/stage2_mapper_v3_phase_b_event_global_mps.yaml",
                    "--max-steps",
                    "1",
                    "--device",
                    "cpu",
                ],
            )

        train.assert_called_once()
        kwargs = train.call_args.kwargs
        self.assertTrue(kwargs["include_full_song_context"])
        self.assertTrue(kwargs["skip_first_eval_pass"])
        self.assertFalse(kwargs["precompute_control_teacher_cache"])
        self.assertEqual(kwargs["mps_cleanup_every"], 20)
        self.assertTrue(kwargs["require_control_teacher_cache"])
        self.assertEqual(kwargs["batch_size"], 2)
        self.assertIsNone(kwargs["resume_from"])
        self.assertEqual(kwargs["model_config_overrides"]["max_seq_len"], 1024)
        self.assertEqual(kwargs["loss_config_overrides"]["lambda_density"], 0.05)
        self.assertIn("stage2_mapper_v3/window_records", kwargs["mapper_record_cache_path"].as_posix())

    def test_cache_only_cli_runs_shared_control_teacher_precompute(self) -> None:
        precompute_result = SimpleNamespace(
            reports=[
                {
                    "split": "source",
                    "total_entries": 1,
                    "computed_entries": 1,
                    "skipped_entries": 0,
                    "elapsed_s": 0.0,
                }
            ],
        )
        with patch.object(
            mapper_v3_training,
            "precompute_mapper_tuple_phase_b_control_teacher_cache",
            return_value=precompute_result,
            autospec=True,
        ) as precompute:
            with patch.object(mapper_v3_training, "run_mapper_v3_phase_b_training", autospec=True) as train:
                mapper_v3_training.main(
                    [
                        "--config",
                        "configs/training/stage2_mapper_v3_phase_b_event_global_mps.yaml",
                        "--precompute-control-teacher-cache-only",
                        "--device",
                        "cpu",
                    ],
                )

        precompute.assert_called_once()
        train.assert_not_called()
        kwargs = precompute.call_args.kwargs
        self.assertEqual(kwargs["control_teacher_cache_dir"], Path("artifacts/cache/stage2_mapper_v2_1/control_teacher_d384_l3_stride16_step002000"))
        self.assertEqual(kwargs["control_teacher_precompute_batch_size"], 12)
        self.assertFalse(kwargs["control_teacher_cache_overwrite"])

    def test_resume_is_explicitly_unsupported_before_dataset_construction(self) -> None:
        with patch.object(mapper_v3_training, "MapperV3WindowDataset", autospec=True) as dataset:
            with self.assertRaisesRegex(ValueError, "resume checkpoints are not implemented"):
                mapper_v3_training.run_mapper_v3_phase_b_training(
                    resume_from=Path("artifacts/runs/stage2_mapper_v3/example/checkpoint.pt"),
                )

        dataset.assert_not_called()

    def test_training_config_records_v3_contract(self) -> None:
        config = mapper_v3_training._mapper_v3_training_config(
            seed=1337,
            run_name="mapper_v3_test",
            learning_rate=2e-4,
            weight_decay=0.01,
            eval_every=1,
            save_every=1,
            skip_first_eval_pass=True,
            dataset_report={
                "mapper_token_contract": "v3_event_groups",
                "mapper_record_cache_path": Path("artifacts/cache/stage2_mapper_v3/window_records/test.parquet"),
            },
            mps_cleanup_every=20,
        )

        self.assertEqual(config["mapper_token_contract"], "v3_event_groups")
        self.assertEqual(config["dataset"]["mapper_token_contract"], "v3_event_groups")
        self.assertEqual(
            config["dataset"]["mapper_record_cache_path"],
            "artifacts/cache/stage2_mapper_v3/window_records/test.parquet",
        )

    def test_run_config_accepts_v3_loss_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mapper_v3_event_budget.yaml"
            path.write_text(
                "\n".join(
                    [
                        "model:",
                        "  use_delta_event_auxiliary_target: true",
                        "  delta_event_delta_max_ms: 4000",
                        "  delta_event_end_gap_max_ms: 4000",
                        "  use_delta_event_factor_target: true",
                        "  delta_event_factor_delta_max_ms: 4000",
                        "  delta_event_factor_end_gap_max_ms: 4000",
                        "control_model: {}",
                        "loss:",
                        "  lambda_event_budget: 0.1",
                        "  lambda_conditioned_event_distribution: 0.2",
                        "  lambda_time_shift_distance: 0.3",
                        "  lambda_delta_event_auxiliary: 0.4",
                        "  lambda_delta_event_factor_target: 0.6",
                        "  event_token_loss_weight: 2.5",
                        "  conditioned_event_high_difficulty_over_weight: 3.0",
                        "  time_shift_distance_scale_ms: 500.0",
                        "  delta_event_end_gap_loss_weight: 0.5",
                        "  delta_event_factor_end_gap_loss_weight: 0.7",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_run_config(path)

        self.assertTrue(config["model"]["use_delta_event_auxiliary_target"])
        self.assertEqual(config["model"]["delta_event_delta_max_ms"], 4000)
        self.assertEqual(config["model"]["delta_event_end_gap_max_ms"], 4000)
        self.assertTrue(config["model"]["use_delta_event_factor_target"])
        self.assertEqual(config["model"]["delta_event_factor_delta_max_ms"], 4000)
        self.assertEqual(config["model"]["delta_event_factor_end_gap_max_ms"], 4000)
        self.assertEqual(config["loss"]["lambda_event_budget"], 0.1)
        self.assertEqual(config["loss"]["lambda_conditioned_event_distribution"], 0.2)
        self.assertEqual(config["loss"]["lambda_time_shift_distance"], 0.3)
        self.assertEqual(config["loss"]["lambda_delta_event_auxiliary"], 0.4)
        self.assertEqual(config["loss"]["lambda_delta_event_factor_target"], 0.6)
        self.assertEqual(config["loss"]["event_token_loss_weight"], 2.5)
        self.assertEqual(config["loss"]["conditioned_event_high_difficulty_over_weight"], 3.0)
        self.assertEqual(config["loss"]["time_shift_distance_scale_ms"], 500.0)
        self.assertEqual(config["loss"]["delta_event_end_gap_loss_weight"], 0.5)
        self.assertEqual(config["loss"]["delta_event_factor_end_gap_loss_weight"], 0.7)
        self.assertTrue(MapperV3Config(**config["model"]).use_delta_event_auxiliary_target)
        self.assertEqual(MapperV3Config(**config["model"]).delta_event_delta_max_ms, 4000)
        self.assertEqual(MapperV3Config(**config["model"]).delta_event_end_gap_max_ms, 4000)
        self.assertTrue(MapperV3Config(**config["model"]).use_delta_event_factor_target)
        self.assertEqual(MapperV3Config(**config["model"]).delta_event_factor_delta_max_ms, 4000)
        self.assertEqual(MapperV3Config(**config["model"]).delta_event_factor_end_gap_max_ms, 4000)
        self.assertEqual(MapperV3LossConfig(**config["loss"]).lambda_event_budget, 0.1)
        self.assertEqual(MapperV3LossConfig(**config["loss"]).lambda_conditioned_event_distribution, 0.2)
        self.assertEqual(MapperV3LossConfig(**config["loss"]).lambda_time_shift_distance, 0.3)
        self.assertEqual(MapperV3LossConfig(**config["loss"]).lambda_delta_event_auxiliary, 0.4)
        self.assertEqual(MapperV3LossConfig(**config["loss"]).lambda_delta_event_factor_target, 0.6)
        self.assertEqual(MapperV3LossConfig(**config["loss"]).event_token_loss_weight, 2.5)
        self.assertEqual(MapperV3LossConfig(**config["loss"]).conditioned_event_high_difficulty_over_weight, 3.0)
        self.assertEqual(MapperV3LossConfig(**config["loss"]).time_shift_distance_scale_ms, 500.0)
        self.assertEqual(MapperV3LossConfig(**config["loss"]).delta_event_end_gap_loss_weight, 0.5)
        self.assertEqual(MapperV3LossConfig(**config["loss"]).delta_event_factor_end_gap_loss_weight, 0.7)

    def test_factor_target_training_config_requests_dataset_fields_only_when_enabled(self) -> None:
        train_result = SimpleNamespace(
            report_path=Path("report.json"),
            checkpoint_path=Path("checkpoint.pt"),
            final_loss=0.0,
            completed_steps=0,
        )
        with patch.object(mapper_v3_training, "MapperV3WindowDataset", side_effect=_FakeMapperV3WindowDataset) as dataset:
            with patch.object(mapper_v3_training, "_run_training", return_value=train_result, autospec=True) as run:
                mapper_v3_training.run_mapper_v3_phase_b_training(
                    index_path=Path("train.parquet"),
                    eval_index_path=Path("eval.parquet"),
                    max_steps=1,
                    eval_every=1,
                    batch_size=1,
                    num_workers=0,
                    model_config_overrides={"use_delta_event_factor_target": True},
                    loss_config_overrides={"lambda_delta_event_factor_target": 1.0},
                )

        self.assertGreaterEqual(dataset.call_count, 2)
        self.assertTrue(all(call.kwargs["include_delta_event_factor_target"] for call in dataset.call_args_list))
        run.assert_called_once()
        self.assertTrue(run.call_args.kwargs["dataset_report"]["include_delta_event_factor_target"])

        with patch.object(mapper_v3_training, "MapperV3WindowDataset", side_effect=_FakeMapperV3WindowDataset) as dataset:
            with patch.object(mapper_v3_training, "_run_training", return_value=train_result, autospec=True) as run:
                mapper_v3_training.run_mapper_v3_phase_b_training(
                    index_path=Path("train.parquet"),
                    eval_index_path=Path("eval.parquet"),
                    max_steps=1,
                    eval_every=1,
                    batch_size=1,
                    num_workers=0,
                )

        self.assertGreaterEqual(dataset.call_count, 2)
        self.assertTrue(all(not call.kwargs["include_delta_event_factor_target"] for call in dataset.call_args_list))
        run.assert_called_once()
        self.assertFalse(run.call_args.kwargs["dataset_report"]["include_delta_event_factor_target"])

    def test_factor_target_loss_requires_matching_model_branch_before_dataset_construction(self) -> None:
        with patch.object(mapper_v3_training, "MapperV3WindowDataset", autospec=True) as dataset:
            with self.assertRaisesRegex(ValueError, "requires use_delta_event_factor_target=True"):
                mapper_v3_training.run_mapper_v3_phase_b_training(
                    model_config_overrides={},
                    loss_config_overrides={"lambda_delta_event_factor_target": 1.0},
                )

        dataset.assert_not_called()

    def test_factor_target_batch_survives_training_tensor_move(self) -> None:
        raw_batch = {
            "decoder_input_tokens": torch.tensor([[1]], dtype=torch.long),
            "target_fragment_tokens": torch.tensor([[2]], dtype=torch.long),
            "delta_event_factor_target": {
                "input_kind": torch.tensor([[0]], dtype=torch.long),
                "row_mask": torch.tensor([[True]], dtype=torch.bool),
            },
            "unrelated": torch.tensor([99], dtype=torch.long),
        }

        batch = _move_mapper_batch_tensors(raw_batch, torch.device("cpu"))

        self.assertIn("delta_event_factor_target", batch)
        self.assertIn("input_kind", batch["delta_event_factor_target"])
        self.assertEqual(batch["delta_event_factor_target"]["input_kind"].device.type, "cpu")
        self.assertNotIn("unrelated", batch)


def _assert_artifacts_paths(test: unittest.TestCase, config: dict[str, object], keys: tuple[str, ...]) -> None:
    for key in keys:
        value = str(config[key])
        test.assertTrue(value.startswith("artifacts/"), msg=f"{key}={value}")
        test.assertNotIn(_STALE_ROOT, value, msg=f"{key}={value}")


class _FakeMapperV3WindowDataset:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.records = [object(), object()]
        self.filter_report = MapperTupleWindowFilterReport(
            num_total_windows=2,
            num_mapper_eligible_windows=2,
            num_dropped_short_windows=0,
            num_dropped_cross_window_ln_windows=0,
            num_dropped_unsupported_action_windows=0,
            drop_rate=0.0,
            short_drop_rate=0.0,
            cross_window_ln_drop_rate=0.0,
            unsupported_action_drop_rate=0.0,
            drop_rate_by_difficulty={},
            drop_rate_by_song={},
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        raise AssertionError("fake mapper v3 dataset should not be iterated in this test")


if __name__ == "__main__":
    unittest.main()
