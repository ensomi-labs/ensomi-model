import pytest

from pulsefield_model.cli.hydra_utils import compose_cli_config


def test_hydra_training_config_composes_with_override() -> None:
    config = compose_cli_config(
        "training/stage2_mapper_v2_1_phase_b_sparse_global_mps",
        ["max_steps=1", "model.max_seq_len=1024"],
    )

    assert config.max_steps == 1
    assert config.model.max_seq_len == 1024
    assert config.control_teacher_precompute_batch_size == 12


def test_hydra_data_command_config_composes() -> None:
    config = compose_cli_config("data/beatmap_index", ["command=filter_difficulty"])

    assert config.command == "filter_difficulty"
    assert config.filter_difficulty.min_difficulty == 2.0


@pytest.mark.parametrize(
    ("config_name", "argv", "expected"),
    [
        ("training/mapper_v2_1", ["--max-steps", "1"], "max_steps=..."),
        ("timing/fit_audio", ["--json"], "emit_json=true"),
        ("training/mapper_v2_1", ["--config", "child.yaml"], "--config-name training/name"),
    ],
)
def test_legacy_argparse_flags_fail_with_hydra_guidance(
    config_name: str,
    argv: list[str],
    expected: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        compose_cli_config(config_name, argv)

    assert expected in capsys.readouterr().err

