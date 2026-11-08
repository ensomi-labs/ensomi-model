"""Short matched R2 runs using the phase-N trainer's optimizer and data draws.

Run from a frozen code snapshot via PYTHONPATH. --representation chooses R2 or
chronological release hazards; feature_blind hides numeric future inputs only.
The latter still uses a future-dependent cand-v1 scheduler. A rolling resume.pt
is overwritten and removed after final.pt; only final.pt is retained on success.
SIGTERM/SIGINT request a clean stop at the next optimizer boundary. All writes
require 100 GB free, and checkpoint writes reserve another 64 MB.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import psutil
import torch

from ensomi_model.r2 import train_ce as ce
from ensomi_model.r2.release_steps import ReleaseStepModel

MIN_FREE = 100_000_000_000


def disk_check(path, reserve=4096):
    free = shutil.disk_usage(path).free
    if free < MIN_FREE + reserve:
        raise RuntimeError(f'Disk guard: {free} free bytes, need {MIN_FREE + reserve}')


def atomic_json(path, value):
    disk_check(path.parent)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def construct(config, representation, *, write=False):
    original = ce.R2Model
    try:
        if representation != 'r2':
            ce.R2Model = lambda cfg, **kw: ReleaseStepModel(cfg, blind=representation == 'feature_blind', **kw)
        return ce.Trainer(config, write=write)
    finally:
        ce.R2Model = original


def save(trainer, path, representation):
    disk_check(path.parent, reserve=64_000_000)
    payload = trainer.payload()
    payload['representation'] = representation
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)


def train(args, config):
    run = Path(config.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    final, recovery = run / 'final.pt', run / 'resume.pt'
    if final.exists():
        raise ValueError('Final checkpoint already exists; use a fresh run directory')
    trainer = construct(config, args.representation)
    if args.resume and recovery.exists():
        payload = torch.load(recovery, weights_only=False, map_location='cpu')
        if payload['representation'] != args.representation:
            raise ValueError('Resume representation differs')
        if payload['config'] != asdict(config):
            raise ValueError('Resume configuration differs; retain the recorded recipe and budget')
        trainer.load(recovery)
    elif recovery.exists():
        raise ValueError('Run contains resume.pt; pass --resume')
    stop = []
    def request_stop(signum, frame):
        stop.append(signal.Signals(signum).name)
    old_handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGTERM, signal.SIGINT)}
    process = psutil.Process()
    rss_start = process.memory_info().rss
    started, initial = time.time(), trainer.state['exposures']
    previous_wall = trainer.state.get('wall_s', 0.)
    next_log = initial
    next_save = initial + args.recovery_every
    status, reason, safe_to_save = 'running', None, True
    recovery_exposures = initial if recovery.exists() else 0
    atomic_json(run / 'config.json', asdict(config))
    atomic_json(run / 'pid.json', dict(pid=os.getpid(), pgid=os.getpgrp(), started=started,
                                      representation=args.representation))
    with (run / 'train.jsonl').open('a') as log:
        try:
            while trainer.state['exposures'] < config.total_exposures:
                disk_check(run, 100_000_000)
                if stop or (run.parent / 'STOP').exists():
                    status, reason = 'stopped', ','.join(stop) or 'STOP file'
                    break
                rss = process.memory_info().rss
                if rss > config.rss_limit_gib * 1024 ** 3:
                    status, reason = 'stopped', 'RSS limit'
                    break
                if rss - rss_start > getattr(config, 'rss_growth_limit_gib', 6.) * 1024 ** 3:
                    status, reason = 'stopped', 'RSS growth limit since process start'
                    break
                safe_to_save = False
                batch = trainer.next_batch()
                info = trainer.step(batch)
                safe_to_save = True
                elapsed = time.time() - started
                trainer.state['wall_s'] = previous_wall + elapsed
                x = trainer.state['exposures']
                if x >= next_log:
                    disk_check(run)
                    row = dict(time=time.time(), exposures=x, steps=trainer.state['steps'],
                        head_rows_per_second=(x - initial) / max(elapsed, 1e-9),
                        rss_bytes=process.memory_info().rss, representation=args.representation, **info)
                    log.write(json.dumps(row, allow_nan=False) + '\n'); log.flush()
                    next_log = x + config.log_every
                if x >= next_save:
                    save(trainer, recovery, args.representation)
                    recovery_exposures = x
                    next_save = x + args.recovery_every
            else:
                status = 'complete'
        except (ce.NonFinite, RuntimeError) as exc:
            status, reason = 'stopped', f'{type(exc).__name__}: {exc}'
        if status == 'complete':
            save(trainer, final, args.representation)
            recovery.unlink(missing_ok=True)
        elif safe_to_save and shutil.disk_usage(run).free >= MIN_FREE + 64_000_000:
            save(trainer, recovery, args.representation)
            recovery_exposures = trainer.state['exposures']
        record = dict(status=status, reason=reason, representation=args.representation,
            exposures=trainer.state['exposures'], total_exposures=config.total_exposures,
            wall_s=trainer.state['wall_s'], time=time.time(), pid=os.getpid(),
            recovery_exposures=recovery_exposures)
        if shutil.disk_usage(run).free >= MIN_FREE + 4096:
            atomic_json(run / 'state.json', record)
        print(json.dumps(record), flush=True)
    for signum, handler in old_handlers.items():
        signal.signal(signum, handler)
    return 0 if status == 'complete' else 75


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--representation', choices=['r2', 'visible', 'feature_blind'], required=True)
    p.add_argument('--smoke-windows', type=int, default=0)
    p.add_argument('--smoke-out')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--recovery-every', type=int, default=200_000)
    args = p.parse_args()
    config = ce.TrainConfig.load(args.config)
    if args.smoke_windows:
        trainer = construct(config, args.representation)
        result = trainer.pilot(args.smoke_windows)
        result['representation'] = args.representation
        result['receipt'] = trainer.receipt('pilot')
        print(json.dumps(result, indent=2), flush=True)
        if args.smoke_out:
            atomic_json(Path(args.smoke_out), result)
        return 0
    return train(args, config)


if __name__ == '__main__':
    raise SystemExit(main())
