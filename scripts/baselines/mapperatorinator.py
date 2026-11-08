#!/usr/bin/env python3
"""Generate one V32 mania chart; see docs/guides/external_baselines.md for setup."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import tempfile
import time


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--star', type=float, required=True)
    parser.add_argument('--arm', choices=('natural', 'high-ln', 'inverse'), required=True)
    parser.add_argument('--output', type=Path, required=True, help='New .osu file; also writes .json')
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    args.audio, args.output = args.audio.resolve(), args.output.resolve()
    if not args.audio.is_file() or not math.isfinite(args.star) or args.star <= 0:
        parser.error('Audio must exist and star must be finite and positive')
    if args.seed < 0 or args.output.suffix != '.osu':
        parser.error('Seed must be nonnegative and output must end in .osu')
    if args.output.exists() or args.output.with_suffix('.json').exists():
        parser.error('Output or its receipt already exists')
    repo = Path(os.environ['MAPPERATORINATOR_REPO']).resolve()
    model = Path(os.environ['MAPPERATORINATOR_MODEL']).resolve()
    if not (model / 'model.safetensors').is_file():
        parser.error('MAPPERATORINATOR_MODEL must be the local gamemode=3 directory')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '2'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_HUB_CACHE'] = str(model.parent / 'hf-cache')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.setpriority(os.PRIO_PROCESS, 0, 10)
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import inference

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    arm = {
        'natural': {'hold_note_ratio': None, 'descriptors': None, 'cfg_scale': 1.0},
        'high-ln': {'hold_note_ratio': 0.8, 'descriptors': None, 'cfg_scale': 1.0},
        'inverse': {'hold_note_ratio': None, 'descriptors': ['gimmick/LN inverse'], 'cfg_scale': 3.0},
    }[args.arm]
    vocabulary = json.loads((model / 'tokenizer.json').read_text())['descriptor_idx']
    for descriptor in arm['descriptors'] or []:
        if descriptor not in vocabulary:
            raise ValueError(f'Descriptor absent from this checkpoint: {descriptor}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='mapper-', dir=args.output.parent) as work:
        with initialize_config_dir(config_dir=str(repo / 'configs/inference'), version_base='1.1'):
            cfg = compose(config_name='v32')
        settings = dict(audio_path=str(args.audio), output_path=work, model_path=str(model),
                        gamemode=3, keycount=4, difficulty=args.star, seed=args.seed,
                        device=os.environ.get('MAPPERATORINATOR_DEVICE', 'cpu'), precision='fp32',
                        auto_select_gamemode_model=False, generate_positions=False, export_osz=False,
                        use_server=False, max_batch_size=1, title=args.audio.stem, **arm)
        for key, value in settings.items():
            OmegaConf.update(cfg, key, value)
        resolved = OmegaConf.to_object(cfg)
        _, chart = inference.main.__wrapped__(resolved)
        shutil.copyfile(chart, args.output)
    receipt = dict(baseline='Mapperatorinator-v32-mania', audio=str(args.audio), star=args.star,
                   arm=args.arm, seed=args.seed, device=resolved.device, precision=resolved.precision,
                   controls=arm, model=str(model), repo=str(repo), threads=2,
                   supplied_timing=False, max_batch_size=1,
                   wall_seconds=time.monotonic() - started,
                   peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                   * (1 if sys.platform == 'darwin' else 1024))
    args.output.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
