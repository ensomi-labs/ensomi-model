from __future__ import annotations

import sys
from dataclasses import is_dataclass
from pathlib import Path
from typing import Sequence, TypeVar

from hydra import compose, initialize_config_dir, initialize_config_module
from omegaconf import DictConfig, OmegaConf

from pulsefield_model.cli.configs import register_configs


T = TypeVar("T")

LEGACY_FLAG_HINTS = {
    "--config": "Use `--config-name training/name` for packaged configs, or move the YAML under `src/pulsefield_model/conf`.",
    "--max-steps": "Use `max_steps=...`.",
    "--json": "Use `emit_json=true`.",
}


def reject_legacy_flags(argv: Sequence[str]) -> None:
    for token in argv:
        if token in LEGACY_FLAG_HINTS:
            print(
                f"{token} is an argparse-style option. {LEGACY_FLAG_HINTS[token]} "
                "Hydra overrides use `key=value` syntax.",
                file=sys.stderr,
            )
            raise SystemExit(2)


def compose_cli_config(default_config_name: str, argv: Sequence[str] | None) -> DictConfig:
    args = list(argv or ())
    reject_legacy_flags(args)
    config_name = _pop_config_name(args, default_config_name)
    config_dir = _pop_config_dir(args)
    return compose_config(config_name, overrides=args, config_dir=config_dir)


def compose_config(
    config_name: str,
    *,
    overrides: Sequence[str] | None = None,
    config_dir: str | Path | None = None,
) -> DictConfig:
    register_configs()
    resolved_overrides = list(overrides or ())
    if config_dir is not None:
        with initialize_config_dir(config_dir=Path(config_dir).expanduser().resolve().as_posix(), version_base=None):
            return compose(config_name=config_name, overrides=resolved_overrides)
    with initialize_config_module(config_module="pulsefield_model.conf", version_base=None):
        return compose(config_name=config_name, overrides=resolved_overrides)


def run_hydra_entrypoint(default_config_name: str, hydra_main: object, argv: Sequence[str] | None) -> object:
    if argv is None:
        reject_legacy_flags(sys.argv[1:])
        return hydra_main()
    return None


def to_config_object(config: DictConfig | T, config_type: type[T]) -> T:
    if isinstance(config, config_type):
        return config
    obj = OmegaConf.to_object(config)
    if isinstance(obj, config_type):
        return obj
    if isinstance(obj, dict):
        return config_type(**obj)
    if is_dataclass(obj):
        return config_type(**obj.__dict__)
    raise TypeError(f"expected {config_type.__name__}, got {type(obj).__name__}")


def _pop_config_name(args: list[str], default_config_name: str) -> str:
    for index, token in enumerate(tuple(args)):
        if token == "--config-name":
            try:
                value = args[index + 1]
            except IndexError as exc:
                raise SystemExit("--config-name requires a value") from exc
            del args[index : index + 2]
            return value
        if token.startswith("--config-name="):
            del args[index]
            return token.split("=", 1)[1]
    return default_config_name


def _pop_config_dir(args: list[str]) -> str | None:
    for index, token in enumerate(tuple(args)):
        if token == "--config-dir":
            try:
                value = args[index + 1]
            except IndexError as exc:
                raise SystemExit("--config-dir requires a value") from exc
            del args[index : index + 2]
            return value
        if token.startswith("--config-dir="):
            del args[index]
            return token.split("=", 1)[1]
    return None
