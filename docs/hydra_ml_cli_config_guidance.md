# Hydra CLI and Config Refactor Guidance

This repo already pins `hydra-core==1.3.3` and has root-level
`configs/training/*.yaml` files plus several `argparse` entry points. Use this
guide when migrating those ML training, data, inference, and timing CLIs to
Hydra without changing runtime behavior accidentally.

## Target Shape

- Put `@hydra.main(...)` only on real executable CLI edges, such as modules run
  with `python -m pulsefield_model.training.mapper_v2_1`. Keep library code,
  runners, data loaders, model factories, and tests free of Hydra global state.
- Move CLI body logic behind normal Python functions, for example
  `run_training(cfg: MapperV21TrainingConfig) -> TrainingResult`. The decorated
  function should adapt Hydra's config object to that function and do little
  else.
- Use structured dataclass configs registered with `ConfigStore` as schemas for
  YAML. Keep YAML as the experiment/user-facing layer, but validate it against
  typed dataclasses.
- Keep path behavior stable. Set `hydra.job.chdir: false` explicitly in primary
  configs even though Hydra 1.2+ defaults this way when `version_base` is modern.
  This repo already treats paths such as `dataset`, `artifacts/...`, and
  checkpoint paths as relative to the launch working directory.
- Use YAML config groups and `defaults` lists for swappable choices:
  dataset/index artifacts, device/runtime profile, mapper model size, control
  model size, loss weights, and experiment profile.
- Use the Compose API only where `@hydra.main` is not applicable: unit tests,
  notebooks, or internal composition code that has no command-line boundary.

## CLI Entry Points

Prefer one Hydra edge per executable command:

```python
@hydra.main(version_base=None, config_path="../conf", config_name="training/mapper_v2_1")
def main(cfg: MapperV21TrainingConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
```

Guidance:

- Use `version_base=None` for current Hydra 1.x defaults and to avoid implicit
  compatibility warnings.
- Keep decorators above or below local wrappers only if the wrapper uses
  `functools.wraps`; Hydra uses the wrapped function metadata to locate configs.
- Do not decorate reusable helpers such as `initialize_*`, model builders, or
  overnight supervisors. Those should receive config objects or plain values.

## Structured Configs

Create one typed root config per CLI family, then nested dataclasses for repeated
sections that already exist in YAML:

```python
from dataclasses import dataclass, field
from omegaconf import MISSING


@dataclass
class MapperModelConfig:
    d_model: int = 384
    heads: int = 8
    layers: int = 4
    dropout: float = 0.1


@dataclass
class MapperV21TrainingConfig:
    dataset_root: str = "dataset"
    index_path: str = MISSING
    output_dir: str = MISSING
    run_name: str = "mapper_v2_1"
    max_steps: int = 5000
    device: str = "auto"
    model: MapperModelConfig = field(default_factory=MapperModelConfig)
```

Register schemas at import time in a narrow config module:

```python
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(name="mapper_v2_1_schema", node=MapperV21TrainingConfig)
cs.store(group="model/mapper_v2_1", name="d384_l4_schema", node=MapperModelConfig)
```

Use `MISSING` for required dataset, index, checkpoint, and output paths. Keep
path fields as strings in config and convert to `Path` at the runtime boundary,
matching the current CLI behavior.

## YAML Groups And Defaults

Current whole-run YAML files can be split gradually. A practical target:

```text
conf/
  training/
    mapper_v2_1.yaml
  dataset/
    local_4k_2to6.yaml
  device/
    mps.yaml
    cuda.yaml
  model/
    mapper_v2_1/
      d384_l4.yaml
      d768_l8.yaml
  control_model/
    d384_l3_stride16.yaml
  loss/
    mapper_v2_1_default.yaml
```

Primary config example:

```yaml
defaults:
  - mapper_v2_1_schema
  - dataset: local_4k_2to6
  - device: mps
  - model/mapper_v2_1: d384_l4
  - control_model: d384_l3_stride16
  - loss: mapper_v2_1_default
  - _self_

hydra:
  job:
    chdir: false

run_name: stage2_mapper_v2_1_phase_b_sparse_global_d384_l4_b2
output_dir: artifacts/runs/stage2_mapper_v2_1/${run_name}
```

Rules for this repo:

- Put `_self_` last when the primary experiment file should override values
  brought in by groups.
- Use groups for meaningful variants, not for every scalar. For example,
  `model/mapper_v2_1=d768_l8` is useful; `learning_rate=0.0002` is usually just
  an override.
- Keep existing experiment names as config names where possible so old
  artifact/run naming remains traceable.

## Config Discovery And Packaging

For installed CLIs, prefer package-discoverable configs instead of relying on a
repo-root `configs/` directory. Put configs under something like
`src/pulsefield_model/conf/`, add `__init__.py` files where package discovery is
needed, and include YAML files as package data in `pyproject.toml`.

Options:

- `@hydra.main(config_path="../conf", config_name="training/mapper_v2_1")`:
  simple when the config directory is packaged next to the CLI module.
- `hydra.searchpath: [pkg://pulsefield_model.conf]`: useful when primary
  configs need to discover additional non-primary package configs.
- `--config-dir /abs/path/to/configs`: useful for local one-off configs outside
  the package; do not make it the only supported production path.

Use `python -m ... --info searchpath` and `--info defaults` to inspect what
Hydra actually discovered and composed.

## Compose API

Use Compose API in tests and non-CLI composition only:

```python
from hydra import compose, initialize_config_module


def test_mapper_config_composes() -> None:
    with initialize_config_module(
        version_base=None,
        config_module="pulsefield_model.conf",
        job_name="test_mapper_v2_1",
    ):
        cfg = compose(
            config_name="training/mapper_v2_1",
            overrides=["model/mapper_v2_1=d384_l4", "max_steps=20"],
        )
```

Avoid Compose API in normal executable commands. It bypasses several benefits of
`@hydra.main`, including Hydra help, tab completion, multirun, working directory
management, and logging management.

## Override Syntax For Users

Hydra application settings are not `argparse` flags. Arguments with `-` or `--`
control Hydra itself; app settings use override expressions:

```sh
uv run python -m pulsefield_model.training.mapper_v2_1 \
  device=mps \
  model/mapper_v2_1=d384_l4 \
  max_steps=120000 \
  batch_size=2 \
  include_full_song_context=false \
  model.dropout=0.1
```

Common translations:

- `--config configs/training/foo.yaml` becomes `--config-name training/foo` if
  `foo` is packaged, or `--config-dir /abs/configs --config-name training/foo`
  for an external directory.
- `--batch-size 2` becomes `batch_size=2`.
- `--learning-rate 0.0002` becomes `learning_rate=0.0002`.
- `--no-include-full-song-context` becomes `include_full_song_context=false`.
- `--d-model 768` becomes `model.d_model=768`, or better,
  `model/mapper_v2_1=d768_l8` when switching a known architecture profile.
- Add a new key with `+key=value`; append-or-overwrite with `++key=value`;
  remove with `~key`.

Quote the whole override in the shell when using spaces, lists, dictionaries,
interpolations, or characters the shell may consume:

```sh
'tags=[baseline,overnight]'
'output_dir=artifacts/runs/${run_name}'
'notes="phase b sparse global"'
```

## Migration Notes From Current `argparse`

1. Pick one CLI family first, preferably a training command that already reads a
   YAML config, such as mapper v2.1 or control demo training.
2. Introduce dataclasses that mirror the normalized current keys
   (`batch_size`, `learning_rate`, `control_model`, `loss`, etc.).
3. Keep the existing runtime function signatures stable at first. Convert the
   structured config to the same arguments currently passed from `argparse`.
4. Add the Hydra-decorated wrapper only at the module's CLI edge.
5. Convert existing YAML files into Hydra primary configs that include the schema
   and `hydra.job.chdir: false`.
6. Split repeated nested sections into groups only after the first end-to-end
   CLI path composes and tests pass.
7. For tests, call runtime functions directly. Add separate config composition
   tests with `initialize_config_module` or `initialize_config_dir`.
8. Update user-facing command examples from dashed flags to override syntax.

Avoid doing all CLIs in one pass. Data builders, timing tools, WebSocket server,
offline inference, and training commands have different stability and packaging
needs.

## Sources

- Hydra getting started and basic application example:
  https://hydra.cc/docs/intro/
- `@hydra.main` config file discovery:
  https://hydra.cc/docs/tutorials/basic/your_first_app/config_file/
- Defaults lists and config groups:
  https://hydra.cc/docs/tutorials/basic/your_first_app/defaults/
- Structured Config schema and ConfigStore:
  https://hydra.cc/docs/tutorials/structured_config/schema/
  https://hydra.cc/docs/tutorials/structured_config/config_store/
- `hydra.job.chdir` and output working directory behavior:
  https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/
- Compose API and unit test usage:
  https://hydra.cc/docs/advanced/compose_api/
  https://hydra.cc/docs/advanced/unit_testing/
- Override syntax and Hydra command-line flags:
  https://hydra.cc/docs/advanced/override_grammar/basic/
  https://hydra.cc/docs/advanced/hydra-command-line-flags/
- Config search path, package discovery, and packaging:
  https://hydra.cc/docs/advanced/search_path/
  https://hydra.cc/docs/advanced/app_packaging/
- `config_path`/`config_name` migration note:
  https://hydra.cc/docs/upgrades/0.11_to_1.0/config_path_changes/
- ML migration reference from fairseq:
  https://github.com/facebookresearch/fairseq/blob/main/docs/hydra_integration.md
