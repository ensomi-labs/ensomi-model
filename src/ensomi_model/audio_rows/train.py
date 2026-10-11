"""Train the head generator on windows of beats, teacher forced.

Each step draws ``batch`` charts and a window of ``window`` beats from each. A window's audio
alignment carries a lag and a tempo error on the chart grid (G1, ``grid='chart'``), none on the
corrected fitted grid (G2, ``grid='fitted'``); the window is split into sections of 8/16/32 beats
labelled with their own log(W_H / s). History dropout zeroes a window's previous-beat tokens and
label dropout its labels. Loss: count cross-entropy on every beat, lattice cross-entropy and slot
BCE on beats with a head, summed and divided by the step's beats.

Every ``eval_every`` steps the run writes ``model.pt`` and ``model-<step>.pt`` and logs the
per-factor NLL on fixed development windows with the label given, absent, and swapped with
another window's: an early sign of label use, not the measure (``evaluate.label_shuffle`` is).
Stop rules: a nonfinite loss; ``max_minutes``; and, from the ``kill_after``-th evaluation on, a
given-label development count NLL not below the entropy of the development count marginal::

    python -m ensomi_model.audio_rows.train --data <build> --grid chart --out <run> --seed 1
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .data import Dataset
from .lattice import MAX_COUNT
from .model import LABEL, HeadConfig, HeadModel, factor_losses


@dataclass
class TrainConfig:
    data: str = ''
    out: str = ''
    grid: str = 'chart'             # 'chart' (G1) or 'fitted' (G2)
    steps: int = 3000
    batch: int = 4
    window: int = 64
    lr: float = 2e-3
    lr_min: float = 2e-4
    seed: int = 1
    threads: int = 3
    eval_every: int = 300
    eval_windows: int = 160
    history_dropout: float = 0.25
    label_dropout: float = 0.3
    jitter_ms: float = 40.0         # chart grid only
    tempo_error: float = 0.001      # chart grid only
    section_beats: tuple = (8, 16, 32)
    kill_after: int = 3
    log_every: int = 50
    max_minutes: float = 40.0


def window_loss(model, w):
    f = factor_losses(model, **w)
    parts = dict(count=f['count'].sum(), lattice=f['lattice'][f['head']].sum(), slot=f['slot'][f['head']].sum())
    return sum(parts.values()), {k: v.item() for k, v in parts.items()} | dict(beats=len(f['count']),
                                                                             head_beats=int(f['head'].sum()))


def count_entropy(windows) -> float:
    p = np.bincount(torch.cat([w['count'] for w in windows]).numpy(), minlength=MAX_COUNT + 1).astype(float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum())


@torch.no_grad()
def dev_nll(model, windows) -> dict:
    """Mean NLL per beat of each factor with the label given, absent, and swapped with another window's."""
    model.eval()
    swap = np.random.default_rng(7).permutation(len(windows))
    out = {}
    for condition in ('given', 'absent', 'swapped'):
        sums = dict(count=0.0, lattice=0.0, slot=0.0, beats=0, head_beats=0)
        for i, w in enumerate(windows):
            extra = w['extra'].clone()
            if condition == 'absent':
                extra[:, LABEL:] = 0.0
            elif condition == 'swapped':
                other = windows[swap[i]]['extra']
                m = min(len(extra), len(other))
                extra[:m, LABEL:] = other[:m, LABEL:]
                extra[m:, LABEL:] = extra[:1, LABEL:]
            for k, v in window_loss(model, dict(w, extra=extra))[1].items():
                sums[k] += v
        out[condition] = dict(count=sums['count'] / sums['beats'], lattice=sums['lattice'] / max(sums['head_beats'], 1),
                              slot=sums['slot'] / max(sums['head_beats'], 1))
    model.train()
    return out


def train(cfg: TrainConfig):
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    chart_rng = np.random.default_rng(cfg.seed + 9001)     # chart draws, shared by runs with one seed
    align = cfg.grid == 'chart'
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Dataset(cfg.data, cfg.grid)
    model = HeadModel(HeadConfig())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.01)
    dev_rng = np.random.default_rng(1000)
    windows = []
    for _ in range(cfg.eval_windows):
        row = data.dev[int(dev_rng.integers(len(data.dev)))]
        start = int(dev_rng.integers(0, max(len(data.charts[row['sha']]['count']) - cfg.window, 0) + 1))
        windows.append(data.window(row, start, cfg.window, dev_rng))
    baseline = count_entropy(windows)
    record = dict(config=asdict(cfg), parameters=sum(p.numel() for p in model.parameters()),
                  train_charts=len(data.train), dev_charts=len(data.dev), count_entropy_dev=baseline,
                  label_mean=data.label_mean, label_std=data.label_std, bands=data.bands, grid=cfg.grid)
    (out / 'config.json').write_text(json.dumps(record, indent=1))
    log = open(out / 'train.jsonl', 'a')
    t0 = time.time()
    killed = None
    acc = dict(count=0.0, lattice=0.0, slot=0.0, beats=0, head_beats=0)
    n_eval = step = 0
    for step in range(1, cfg.steps + 1):
        lr = cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1 + math.cos(math.pi * step / cfg.steps))
        for g in opt.param_groups:
            g['lr'] = lr
        opt.zero_grad()
        batch = []
        for _ in range(cfg.batch):
            row = data.train[int(chart_rng.integers(len(data.train)))]
            start = int(rng.integers(0, max(len(data.charts[row['sha']]['count']) - cfg.window, 0) + 1))
            batch.append(data.window(row, start, cfg.window, rng, jitter_ms=cfg.jitter_ms * align,
                                     tempo_error=cfg.tempo_error * align, history_dropout=cfg.history_dropout,
                                     label_dropout=cfg.label_dropout, section_beats=cfg.section_beats))
        beats = sum(len(w['count']) for w in batch)
        for w in batch:
            total, parts = window_loss(model, w)
            (total / beats).backward()
            for k in acc:
                acc[k] += parts[k]
        if not all(math.isfinite(v) for v in acc.values()):
            killed = f'nonfinite loss at step {step}'
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % cfg.log_every == 0:
            log.write(json.dumps(dict(step=step, lr=lr, count=acc['count'] / acc['beats'],
                                      lattice=acc['lattice'] / max(acc['head_beats'], 1),
                                      slot=acc['slot'] / max(acc['head_beats'], 1), seconds=time.time() - t0)) + '\n')
            log.flush()
            acc = dict.fromkeys(acc, 0)
        if step % cfg.eval_every == 0 or step == cfg.steps:
            n_eval += 1
            ev = dict(step=step, seconds=time.time() - t0, count_entropy_dev=baseline, **dev_nll(model, windows))
            ev.update(delta_count_absent=ev['absent']['count'] - ev['given']['count'],
                      delta_count_swapped=ev['swapped']['count'] - ev['given']['count'])
            checkpoint = dict(model=model.state_dict(), config=asdict(model.cfg), record=record, step=step)
            torch.save(checkpoint, out / 'model.pt')
            torch.save(checkpoint, out / f'model-{step}.pt')
            print(json.dumps(ev), flush=True)
            if n_eval >= cfg.kill_after and ev['given']['count'] >= baseline:
                killed = (f'kill rule: dev count NLL {ev["given"]["count"]:.3f} '
                          f'not below the count entropy {baseline:.3f}')
                break
        if (time.time() - t0) / 60 > cfg.max_minutes:
            killed = f'time budget of {cfg.max_minutes} minutes at step {step}'
            break
    summary = dict(record, steps_done=step, killed=killed, wall_seconds=time.time() - t0)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(dict(steps_done=step, killed=killed, wall_seconds=summary['wall_seconds'])), flush=True)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    for f in fields(TrainConfig):
        if f.name != 'section_beats':
            p.add_argument('--' + f.name.replace('_', '-'), type=type(f.default), default=f.default,
                           required=f.name in ('data', 'out'))
    p.add_argument('--section-beats', type=int, nargs='+', default=TrainConfig.section_beats)
    a = vars(p.parse_args(argv))
    if a['grid'] not in ('chart', 'fitted'):
        p.error('--grid is chart or fitted')
    torch.set_num_interop_threads(1)
    train(TrainConfig(**dict(a, section_beats=tuple(a['section_beats']))))


if __name__ == '__main__':
    main()
