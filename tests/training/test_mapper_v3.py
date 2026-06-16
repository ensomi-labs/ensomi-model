import tempfile
import unittest

import importlib.util

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("requires torch")
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pulsefield_model.models.control import ControlDemoGlobalEncoderConfig
from pulsefield_model.models.mapper.v3 import MapperV3Config, MapperV3LossConfig
from pulsefield_model.training import mapper_v3 as mapper_v3_training
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

    def test_run_config_accepts_event_budget_loss_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mapper_v3_event_budget.yaml"
            path.write_text(
                "\n".join(
                    [
                        "model: {}",
                        "control_model: {}",
                        "loss:",
                        "  lambda_event_budget: 0.1",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_run_config(path)

        self.assertEqual(config["loss"]["lambda_event_budget"], 0.1)
        self.assertEqual(MapperV3LossConfig(**config["loss"]).lambda_event_budget, 0.1)


def _assert_artifacts_paths(test: unittest.TestCase, config: dict[str, object], keys: tuple[str, ...]) -> None:
    for key in keys:
        value = str(config[key])
        test.assertTrue(value.startswith("artifacts/"), msg=f"{key}={value}")
        test.assertNotIn(_STALE_ROOT, value, msg=f"{key}={value}")


if __name__ == "__main__":
    unittest.main()
