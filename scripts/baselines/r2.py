#!/usr/bin/env python3
"""Run the released audio-to-chart System.make, optionally replacing its three kappa targets.

The source arm consumes an existing JSON array [held share, heads per row,
all-busy share] computed by stage 1b's source_levels. These are source-chart
statistics, not reference head rows or timing. The seed names the arrangement
seed; make's rows seed is seed - 954. Outputs omit copied audio and archives.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import tempfile
import time
from unittest.mock import patch


def main():
    started = time.monotonic()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audio', type=Path, required=True)
    p.add_argument('--star', type=float, required=True)
    p.add_argument('--arm', choices=['shipped', 'source'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--source-levels', type=Path)
    p.add_argument('--model-dir', type=Path, required=True)
    a = p.parse_args()
    if not a.audio.is_file() or not 2 <= a.star < 6 or a.seed < 954:
        p.error('Audio must exist, star must be in [2, 6), and arrangement seed must be at least 954')
    if a.output.suffix != '.osu' or any(a.output.with_suffix(suffix).exists() for suffix in ('.osu', '.json', '.npz')):
        p.error('Output must be a new .osu path with no existing JSON receipt or NPZ arrays')
    if (a.arm == 'source') != (a.source_levels is not None):
        p.error('Only the source arm requires --source-levels')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[name] = '1'
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.setpriority(os.PRIO_PROCESS, 0, 10)
    import numpy as np
    import torch
    from ensomi_model.system.make import System
    import ensomi_model.system.prior as prior

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    levels = np.asarray(json.loads(a.source_levels.read_text()), dtype=float) if a.source_levels else None
    if levels is not None and (levels.shape != (3,) or not np.isfinite(levels).all()
                               or not 0 <= levels[0] <= 1 or not 1 <= levels[1] <= 4
                               or not 0 <= levels[2] <= 1):
        p.error('Source levels must be finite [held share, heads per row, all-busy share]')
    model_dir = a.model_dir.resolve()
    system = System.load(json.loads((model_dir / 'system.json').read_text()), model_dir)
    original = prior.kappa_targets

    def targets(pred, ln_fraction=1.0, multiplier=1.0):
        return original(pred, ln_fraction, multiplier) if levels is None else levels.copy()

    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='r2-', dir=a.output.parent) as temp:
        archive = Path(temp) / 'chart.osz'
        with patch.object(prior, 'kappa_targets', targets):
            receipt = system.make(str(a.audio.resolve()), a.star, archive, seed=a.seed - 954,
                                  title=a.audio.stem, artist='Unknown', name=a.arm)
        shutil.copyfile(Path(temp) / 'chart/generated.osu', a.output)
        shutil.copyfile(Path(temp) / 'chart/arrays.npz', a.output.with_suffix('.npz'))
    receipt.update(baseline='R2-r2-1.0.1', arm=a.arm, requested_seed=a.seed, device='cpu', threads=1,
                   source_levels=None if levels is None else levels.tolist(),
                   supplied_timing=False, supplied_heads=False,
                   wall_seconds=time.monotonic() - started,
                   peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                   * (1 if sys.platform == 'darwin' else 1024))
    a.output.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
