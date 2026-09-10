from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf


@dataclass
class Config:
    lens_root: str = "../beatmap-lens"
    output: str = ""
    device: str = "mps"
    seed: int = 73
    max_rows: int = 512
    holdout_fraction: float = 0.1
    width: int = 128
    heads: int = 4
    layers: int = 4
    dropout: float = 0.1
    condition_dropout: float = 0.15
    batch_size: int = 4
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    steps: int = 3000
    eval_every: int = 100
    log_every: int = 25
    patience: int = 8
    max_seconds: int = 7200
    sampling_steps: int = 16
    sample_scopes: int = 3
    sample_seeds: int = 2
    temperature: float = 1.0
    final_test: bool = True
    cpu_threads: int = 4


ConfigStore.instance().store(name="gold_diffusion_schema", node=Config)


def project(cfg):
    values = OmegaConf.to_container(cfg, resolve=True)
    unknown = set(values) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unsupported configuration fields: {sorted(unknown)}")
    config = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Config), cfg))
    for name in ("max_rows", "width", "heads", "layers", "batch_size", "steps", "eval_every", "log_every",
                 "patience", "max_seconds", "sampling_steps", "sample_scopes", "sample_seeds", "cpu_threads"):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive")
    if config.width % config.heads:
        raise ValueError("width must be divisible by heads")
    for name in ("dropout", "condition_dropout"):
        if not 0 <= getattr(config, name) < 1:
            raise ValueError(f"{name} must lie in [0,1)")
    if not 0 < config.holdout_fraction < 0.4:
        raise ValueError("holdout_fraction must lie in (0,0.4)")
    if config.device not in ("cpu", "mps", "cuda"):
        raise ValueError("device must be cpu, mps, or cuda")
    if (any(not math.isfinite(getattr(config, key)) for key in ("learning_rate", "temperature", "weight_decay"))
            or config.learning_rate <= 0 or config.temperature <= 0 or config.weight_decay < 0):
        raise ValueError("Invalid optimizer or sampling parameters")
    return config


@hydra.main(version_base=None, config_path="../../configs/hydra", config_name="gold_diffusion")
def main(cfg: DictConfig):
    config = project(cfg)
    out = Path(config.output or ("artifacts/gold_diffusion/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    (out / "runner.json").write_text(json.dumps(asdict(config), indent=2))
    from .run import train
    try:
        train(config, out)
    except Exception as error:
        (out / "failed.json").write_text(json.dumps({"error": repr(error)}))
        from .run import emit
        emit(out, "failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
