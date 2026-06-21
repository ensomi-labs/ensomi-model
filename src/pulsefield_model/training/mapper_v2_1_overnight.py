from __future__ import annotations

import signal
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import hydra
from omegaconf import DictConfig, OmegaConf

from pulsefield_model.cli.configs import MapperV21OvernightConfig
from pulsefield_model.cli.hydra_utils import compose_cli_config, compose_config, to_config_object
from pulsefield_model.training.overnight import checkpoint_saved_after
from pulsefield_model.training.overnight import existing_resume_checkpoint
from pulsefield_model.training.overnight import next_saved_step_target
from pulsefield_model.training.overnight import read_progress
from pulsefield_model.training.overnight import sleep_until_stop_or_timeout
from pulsefield_model.training.overnight import terminate_process_group
from pulsefield_model.training.overnight import write_child_config


DEFAULT_CONFIG_PATH = Path("configs/training/stage2_mapper_v2_1_phase_b_sparse_global_mps.yaml")
DEFAULT_UV_COMMAND = "uv run --extra mps python -m pulsefield_model.training.mapper_v2_1"
_CONFIG_NAME = "training/mapper_v2_1_overnight"


def run_supervisor(args: MapperV21OvernightConfig, trainer_args: Sequence[str]) -> int:
    base_config = _load_training_config(args)
    output_dir = Path(args.output_dir or base_config.get("output_dir", "artifacts/runs/stage2_mapper_v2_1/overnight"))
    max_steps = int(args.max_steps or base_config.get("max_steps", 5000))
    save_every = int(args.save_every or base_config.get("save_every") or base_config.get("eval_every", 100))
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    if save_every <= 0:
        raise ValueError(f"save_every must be positive, got {save_every}")
    steps_per_process = int(args.steps_per_process or save_every)
    if steps_per_process <= 0:
        raise ValueError(f"steps_per_process must be positive, got {steps_per_process}")

    base_config["output_dir"] = output_dir.as_posix()
    base_config["max_steps"] = max_steps
    base_config["save_every"] = save_every

    output_dir.mkdir(parents=True, exist_ok=True)
    supervisor_dir = output_dir / "overnight_supervisor"
    log_dir = Path(args.log_dir) if args.log_dir is not None else supervisor_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    child_config_path = supervisor_dir / "mapper_v2_1_child.yaml"
    checkpoint_path = output_dir / "checkpoint.pt"
    report_path = output_dir / "report.json"
    stop_signal: int | None = None

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stop_signal
        if stop_signal is None:
            stop_signal = int(signum)
            print(f"overnight_stop_requested signal={signum}", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def should_stop() -> bool:
        return stop_signal is not None

    run_count = 0
    consecutive_failures = 0
    while True:
        if should_stop():
            return 128 + int(stop_signal or signal.SIGTERM)
        progress = read_progress(report_path, max_steps=max_steps)
        if progress.is_complete and args.stop_when_complete:
            print(f"overnight_complete step={progress.completed_steps}/{max_steps}", flush=True)
            return 0
        if args.max_runs and run_count >= args.max_runs:
            print(f"overnight_max_runs_reached runs={run_count} step={progress.completed_steps}/{max_steps}", flush=True)
            return 0

        resume_checkpoint = existing_resume_checkpoint(base_config, output_dir)
        write_child_config(
            base_config=base_config,
            output_dir=output_dir,
            config_path=child_config_path,
            resume_checkpoint=resume_checkpoint,
        )
        target_step = next_saved_step_target(
            completed_steps=progress.completed_steps,
            max_steps=max_steps,
            save_every=save_every,
            steps_per_process=steps_per_process,
        )
        previous_mtime_ns = checkpoint_path.stat().st_mtime_ns if checkpoint_path.is_file() else None
        run_count += 1
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = log_dir / f"attempt_{run_count:04d}_{stamp}.log"
        command = [
            *shlex.split(args.uv_command),
            *_child_config_args(child_config_path),
            *trainer_args,
        ]
        if args.dry_run:
            print("overnight_dry_run " + " ".join(command), flush=True)
            return 0

        print(
            "overnight_start "
            f"run={run_count} step={progress.completed_steps}/{max_steps} "
            f"target_step={target_step} resume_from={resume_checkpoint} log={log_path}",
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
            saved = False
            while process.poll() is None:
                if not sleep_until_stop_or_timeout(float(args.poll_seconds), should_stop):
                    terminate_process_group(process, timeout_s=float(args.terminate_timeout_seconds))
                    return 128 + int(stop_signal or signal.SIGTERM)
                current = read_progress(report_path, max_steps=max_steps)
                saved = checkpoint_saved_after(
                    output_dir=output_dir,
                    checkpoint_path=checkpoint_path,
                    completed_steps=current.completed_steps,
                    target_step=target_step,
                    previous_mtime_ns=previous_mtime_ns,
                )
                if saved:
                    print(
                        "overnight_saved "
                        f"run={run_count} step={current.completed_steps}/{max_steps} "
                        f"terminating_child_pid={process.pid}",
                        flush=True,
                    )
                    sleep_until_stop_or_timeout(float(args.post_save_grace_seconds), should_stop)
                    terminate_process_group(process, timeout_s=float(args.terminate_timeout_seconds))
                    break

            return_code = process.poll()

        current = read_progress(report_path, max_steps=max_steps)
        if not saved:
            saved = checkpoint_saved_after(
                output_dir=output_dir,
                checkpoint_path=checkpoint_path,
                completed_steps=current.completed_steps,
                target_step=target_step,
                previous_mtime_ns=previous_mtime_ns,
            )
        if saved:
            consecutive_failures = 0
            print(f"overnight_restart_ready run={run_count} step={current.completed_steps}/{max_steps}", flush=True)
            if current.is_complete and args.stop_when_complete:
                print(f"overnight_complete step={current.completed_steps}/{max_steps}", flush=True)
                return 0
            sleep_until_stop_or_timeout(float(args.restart_delay_seconds), should_stop)
            continue

        consecutive_failures += 1
        print(
            "overnight_child_failed "
            f"run={run_count} returncode={return_code} failures={consecutive_failures} "
            f"log={log_path}",
            flush=True,
        )
        if consecutive_failures >= int(args.max_consecutive_failures):
            return int(return_code or 1)
        sleep_until_stop_or_timeout(float(args.restart_delay_seconds), should_stop)


def _load_training_config(config: MapperV21OvernightConfig) -> dict[str, Any]:
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


def parse_args(argv: Sequence[str] | None = None) -> tuple[MapperV21OvernightConfig, list[str]]:
    config = to_config_object(compose_cli_config(_CONFIG_NAME, argv), MapperV21OvernightConfig)
    return config, list(config.trainer_overrides)


@hydra.main(version_base=None, config_path="../conf", config_name=_CONFIG_NAME)
def _hydra_main(config: DictConfig) -> None:
    typed_config = to_config_object(config, MapperV21OvernightConfig)
    raise SystemExit(run_supervisor(typed_config, typed_config.trainer_overrides))


def main(argv: Sequence[str] | None = None) -> None:
    if argv is None:
        _hydra_main()
        return
    args, trainer_args = parse_args(argv)
    raise SystemExit(run_supervisor(args, trainer_args))


if __name__ == "__main__":
    main(sys.argv[1:])
