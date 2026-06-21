from __future__ import annotations

import signal
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

from pulsefield_model.cli.configs import MapperV21ControlCacheOvernightConfig
from pulsefield_model.cli.hydra_utils import compose_cli_config, compose_config, to_config_object
from pulsefield_model.training.overnight import sleep_until_stop_or_timeout
from pulsefield_model.training.overnight import terminate_process_group


DEFAULT_CONFIG_PATH = Path("configs/training/stage2_mapper_v2_1_phase_b_sparse_global_mps.yaml")
DEFAULT_UV_COMMAND = "uv run --extra mps python -m pulsefield_model.training.mapper_v2_1"
DEFAULT_CONTROL_TEACHER_PRECOMPUTE_BATCH_SIZE = 24
DEFAULT_MAX_RUNS = 16000
_CONFIG_NAME = "training/mapper_v2_1_control_cache_overnight"


def write_cache_child_config(
    *,
    base_config: dict[str, object],
    output_dir: Path,
    config_path: Path,
    control_teacher_precompute_batch_size: int,
) -> None:
    child_config = dict(base_config)
    child_config["output_dir"] = output_dir.as_posix()
    child_config["precompute_control_teacher_cache_only"] = True
    child_config["control_teacher_precompute_batch_size"] = control_teacher_precompute_batch_size
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(child_config, sort_keys=False), encoding="utf-8")


def run_supervisor(args: MapperV21ControlCacheOvernightConfig, trainer_args: Sequence[str]) -> int:
    base_config = _load_training_config(args)
    output_dir = Path(args.output_dir or base_config.get("output_dir", "artifacts/runs/stage2_mapper_v2_1/overnight"))
    output_dir.mkdir(parents=True, exist_ok=True)
    supervisor_dir = output_dir / "overnight_control_cache_supervisor"
    log_dir = Path(args.log_dir) if args.log_dir is not None else supervisor_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    child_config_path = supervisor_dir / "mapper_v2_1_control_cache_child.yaml"
    write_cache_child_config(
        base_config=base_config,
        output_dir=output_dir,
        config_path=child_config_path,
        control_teacher_precompute_batch_size=int(args.control_teacher_precompute_batch_size),
    )

    stop_signal: int | None = None

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stop_signal
        if stop_signal is None:
            stop_signal = int(signum)
            print(f"overnight_cache_stop_requested signal={signum}", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def should_stop() -> bool:
        return stop_signal is not None

    run_count = 0
    consecutive_failures = 0
    while True:
        if should_stop():
            return 128 + int(stop_signal or signal.SIGTERM)
        if args.max_runs and run_count >= args.max_runs:
            print(f"overnight_cache_max_runs_reached runs={run_count}", flush=True)
            return 0

        run_count += 1
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = log_dir / f"attempt_{run_count:04d}_{stamp}.log"
        command = [
            *shlex.split(args.uv_command),
            *_child_config_args(child_config_path),
            "precompute_control_teacher_cache_only=true",
            *trainer_args,
        ]
        if args.dry_run:
            print("overnight_cache_dry_run " + " ".join(command), flush=True)
            return 0

        print(
            "overnight_cache_start "
            f"run={run_count} config={child_config_path} log={log_path}",
            flush=True,
        )
        with log_path.open("a", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=Path.cwd(),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            while process.poll() is None:
                if not sleep_until_stop_or_timeout(float(args.poll_seconds), should_stop):
                    terminate_process_group(process, timeout_s=float(args.terminate_timeout_seconds))
                    return 128 + int(stop_signal or signal.SIGTERM)
            return_code = int(process.returncode or 0)

        if return_code == 0:
            print(f"overnight_cache_complete run={run_count} log={log_path}", flush=True)
            return 0

        consecutive_failures += 1
        print(
            "overnight_cache_child_failed "
            f"run={run_count} returncode={return_code} failures={consecutive_failures} "
            f"log={log_path}",
            flush=True,
        )
        if consecutive_failures >= int(args.max_consecutive_failures):
            return return_code
        sleep_until_stop_or_timeout(float(args.restart_delay_seconds), should_stop)


def _load_training_config(config: MapperV21ControlCacheOvernightConfig) -> dict[str, Any]:
    composed = compose_config(
        config.training_config_name,
        overrides=config.training_config_overrides,
        config_dir=config.training_config_dir,
    )
    loaded = OmegaConf.to_container(composed, resolve=True)
    if not isinstance(loaded, Mapping):
        raise ValueError(f"training config must compose to a mapping: {config.training_config_name}")
    return dict(loaded)


def _child_config_args(config_path: Path) -> list[str]:
    return [
        "--config-dir",
        config_path.parent.resolve().as_posix(),
        "--config-name",
        config_path.stem,
    ]


def parse_args(argv: Sequence[str] | None = None) -> tuple[MapperV21ControlCacheOvernightConfig, list[str]]:
    config = to_config_object(compose_cli_config(_CONFIG_NAME, argv), MapperV21ControlCacheOvernightConfig)
    return config, list(config.trainer_overrides)


@hydra.main(version_base=None, config_path="../conf", config_name=_CONFIG_NAME)
def _hydra_main(config: DictConfig) -> None:
    typed_config = to_config_object(config, MapperV21ControlCacheOvernightConfig)
    raise SystemExit(run_supervisor(typed_config, typed_config.trainer_overrides))


def main(argv: Sequence[str] | None = None) -> None:
    if argv is None:
        _hydra_main()
        return
    args, trainer_args = parse_args(argv)
    raise SystemExit(run_supervisor(args, trainer_args))


if __name__ == "__main__":
    main(sys.argv[1:])
