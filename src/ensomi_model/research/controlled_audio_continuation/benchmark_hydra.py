"""Torch-free process boundary for the local generation benchmark."""
from dataclasses import fields
import json
import time

from hydra import compose, initialize_config_module, main
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from .benchmark_config import BenchmarkConfig

ConfigStore.instance().store(name='controlled_audio_benchmark_schema', node=BenchmarkConfig)


def project_config(config):
    values = OmegaConf.to_container(config, resolve=True)
    if set(values) != {f.name for f in fields(BenchmarkConfig)}:
        raise ValueError('Unknown or missing controlled audio benchmark setting')
    result = BenchmarkConfig(**values)
    result.validate()
    return result


def compose_config(overrides=None):
    with initialize_config_module(version_base='1.3', config_module='ensomi_model.configs.hydra'):
        return project_config(compose(config_name='controlled_audio_benchmark', overrides=overrides or []))


@main(version_base='1.3', config_path='../../configs/hydra', config_name='controlled_audio_benchmark')
def cli(config: DictConfig):
    started = time.perf_counter()
    from .benchmark import run_benchmark
    imports = time.perf_counter() - started
    result = run_benchmark(project_config(config), resolved_yaml=OmegaConf.to_yaml(config, resolve=True),
                           runtime_import_seconds=imports)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    cli()
