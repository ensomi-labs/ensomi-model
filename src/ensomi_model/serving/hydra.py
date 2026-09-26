"""Torch-free launch boundary for the local demo service."""
from dataclasses import fields
from hydra import main
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from .config import DemoServiceConfig

ConfigStore.instance().store(name='controlled_demo_schema', node=DemoServiceConfig)

def project_config(config):
    values = OmegaConf.to_container(config, resolve=True)
    if set(values) != {f.name for f in fields(DemoServiceConfig)}:
        raise ValueError('Unknown or missing demo service setting')
    result = DemoServiceConfig(**values)
    result.validate()
    return result

@main(version_base='1.3', config_path='../configs/inference', config_name='controlled_demo')
def cli(config: DictConfig):
    settings = project_config(config)
    from .http import serve
    serve(settings)

if __name__ == '__main__':
    cli()
