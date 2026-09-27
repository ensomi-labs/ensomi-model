"""Native evaluation: exit 2 for failed gates, 3 while semantic review is pending."""
from dataclasses import fields
import json

from hydra import compose,initialize_config_module,main
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig,OmegaConf

from .qualification_config import QualificationConfig

ConfigStore.instance().store(name='native_gameplay_qualification_schema',node=QualificationConfig)


def project_config(config):
    values=OmegaConf.to_container(config,resolve=True)
    if set(values)!={f.name for f in fields(QualificationConfig)}:
        raise ValueError('Unknown or missing native qualification setting')
    result=QualificationConfig(**values);result.validate();return result


def compose_config(overrides=None):
    with initialize_config_module(version_base='1.3',config_module='ensomi_model.configs.hydra'):
        return project_config(compose(config_name='native_gameplay_qualification',overrides=overrides or []))


@main(version_base='1.3',config_path='../../configs/hydra',config_name='native_gameplay_qualification')
def cli(config:DictConfig):
    from .qualification import run_qualification
    result=run_qualification(project_config(config),resolved_yaml=OmegaConf.to_yaml(config,resolve=True),
        on_case=lambda record: print(json.dumps(dict(case=record['key'],completed=record['completed'],
            numeric_status=record.get('numeric_status'),reason=record.get('reason')),allow_nan=False),flush=True))
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)
    if result['candidate_status']=='failed':raise SystemExit(2)
    raise SystemExit(3)


if __name__=='__main__':cli()
