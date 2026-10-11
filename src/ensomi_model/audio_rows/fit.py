"""Train the figure generator on windows of beats, teacher forced.

Each step draws ``batch`` fit_train charts uniformly and a window of up to ``window`` beats from
each (start uniform). The audio is shifted by up to ±``shift_ms``; with probability
``section_rate`` a window carries true section requests on a share ``section_prob`` of its
sections (``beatdata.Beats.window``). The defaults are the trained recipe (pilot and full length):
section_rate .5, section_prob .3, copy heads trained on rests too; ``rowsgen.decode`` keeps them off
empty beats. A full-length run with section_rate .75, section_prob .5 and copy_rests off thinned
long songs far more at step 1,500 and was stopped. Loss: mean figure NLL over the step's beats
(``net.nll``), beats with a figure outside the vocabulary excluded.

The next batch is drawn in a background thread while the current one trains.

Every ``eval_every`` steps the run saves ``model.pt`` and ``model-<step>.pt`` and logs the NLL of
fixed development windows, given and with the copy path off: an early sign, not the ablation
(``ablate`` is). Kill rules: a nonfinite loss; ``max_minutes``; from the ``kill_after``-th
evaluation on, a development NLL not below the entropy of the development figure marginal; and,
with ``warn_step``, a development NLL above ``warn_nll`` at that step::

    python -m ensomi_model.audio_rows.fit --data <build> --out <run> --seed 1
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .beatdata import Beats
from .net import FigureNet, NetConfig, nll, to_device


@dataclass
class FitConfig:
    data: str = ''
    out: str = ''
    steps: int = 1500
    batch: int = 8
    window: int = 512
    lr: float = 2e-3
    lr_min: float = 2e-4
    warmup: int = 100
    seed: int = 1
    device: str = 'mps'
    threads: int = 4
    eval_every: int = 250
    eval_windows: int = 64
    shift_ms: float = 10.0
    section_rate: float = 0.5
    section_prob: float = 0.3
    copy_rests: bool = True
    drop: float = 0.1
    kill_after: int = 2
    warn_step: int = 0              # early warning: stop when the dev NLL at this step exceeds warn_nll
    warn_nll: float = 0.0
    log_every: int = 25
    max_minutes: float = 60.0


def collate(windows, device, beats=None, frame_multiple=1) -> dict:
    """Pad windows to one batch: frames with zeros, beats with ids -1 and valid False.

    ``beats`` and ``frame_multiple`` fix the padded shapes, so that the MPS allocator reuses its
    blocks instead of caching one per shape."""
    B = len(windows)
    L = beats or max(len(w['ids']) for w in windows)
    Fm = -(-max(len(w['mel']) for w in windows) // frame_multiple) * frame_multiple
    out = dict(mel=torch.zeros(B, Fm, windows[0]['mel'].shape[1]), pos_frames=torch.zeros(B, L, 24, dtype=torch.long),
               beat_frames=torch.zeros(B, L, 2, dtype=torch.long), extra=torch.zeros(B, L, windows[0]['extra'].shape[1]),
               ids=torch.full((B, L), -1, dtype=torch.long), valid=torch.zeros(B, L, dtype=torch.bool))
    for i, w in enumerate(windows):
        n, f = len(w['ids']), len(w['mel'])
        out['mel'][i, :f] = w['mel']
        out['pos_frames'][i, :n] = w['pos_frames']
        out['beat_frames'][i, :n] = w['beat_frames']
        out['extra'][i, :n] = w['extra']
        out['ids'][i, :n] = w['ids']
        out['valid'][i, :n] = True
    return {k: v.to(device) for k, v in out.items()}


def figure_entropy(ids, V) -> float:
    p = np.bincount(ids[ids >= 0], minlength=V).astype(float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum())


@torch.no_grad()
def dev_nll(model, batches) -> dict:
    model.eval()
    out = {}
    for name, off in (('given', ()), ('copy_off', ('copy',))):
        tot = n = 0.0
        for b in batches:
            loss, m = nll(model, b, drop=0.0, off=off)
            tot += float(loss.sum())
            n += float(m.sum())
        out[name] = tot / n
    model.train()
    return out


def fit(cfg: FitConfig):
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Beats(cfg.data)
    net_cfg = NetConfig(vocab=len(data.vocab), copy_rests=cfg.copy_rests)
    model = to_device(FigureNet(net_cfg, data.vocab.members), device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.01)
    dev_rng = np.random.default_rng(1000)
    dev_windows = []
    for _ in range(cfg.eval_windows):
        row = data.dev[int(dev_rng.integers(len(data.dev)))]
        start = int(dev_rng.integers(0, max(row['nb'] - cfg.window, 0) + 1))
        dev_windows.append(data.window(row, start, cfg.window, dev_rng))
    dev_batches = [collate(dev_windows[i:i + cfg.batch], device, cfg.window, 4096)
                   for i in range(0, len(dev_windows), cfg.batch)]
    baseline = figure_entropy(np.concatenate([w['ids'].numpy() for w in dev_windows]), len(data.vocab))
    record = dict(config=asdict(cfg), net=asdict(net_cfg), parameters=sum(p.numel() for p in model.parameters()),
                  train_charts=len(data.train), dev_charts=len(data.dev), figure_entropy_dev=baseline,
                  **{k: data.record[k] for k in ('vocab', 'label_mean', 'label_std', 'star_map', 'bands', 'coverage')})
    (out / 'config.json').write_text(json.dumps({k: v for k, v in record.items() if k != 'vocab'}, indent=1))
    log = open(out / 'train.jsonl', 'a')
    gen = torch.Generator(device=device).manual_seed(cfg.seed)

    def draw():
        windows = []
        for _ in range(cfg.batch):
            row = data.train[int(rng.integers(len(data.train)))]
            start = int(rng.integers(0, max(row['nb'] - cfg.window, 0) + 1))
            windows.append(data.window(row, start, cfg.window, rng, shift_ms=cfg.shift_ms,
                                       section_rate=cfg.section_rate, section_prob=cfg.section_prob))
        return collate(windows, 'cpu', cfg.window, 4096)

    prefetch = ThreadPoolExecutor(1)        # the next batch is read while this one trains
    pending = prefetch.submit(draw)
    t0 = time.time()
    killed = None
    acc = n_acc = data_s = 0.0
    n_eval = step = 0
    for step in range(1, cfg.steps + 1):
        lr = cfg.lr * min(1.0, step / cfg.warmup)
        if step > cfg.warmup:
            lr = cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1 + math.cos(math.pi * (step - cfg.warmup)
                                                                          / (cfg.steps - cfg.warmup)))
        for g in opt.param_groups:
            g['lr'] = lr
        td = time.time()
        batch = {k: v.to(device) for k, v in pending.result().items()}
        pending = prefetch.submit(draw)
        data_s += time.time() - td
        loss, m = nll(model, batch, drop=cfg.drop, rng=gen)
        total = loss.sum() / m.sum()
        opt.zero_grad()
        total.backward()
        if not math.isfinite(total.item()):
            killed = f'nonfinite loss at step {step}'
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        acc += float(loss.sum())
        n_acc += float(m.sum())
        if device.type == 'mps' and step % 25 == 0:
            torch.mps.empty_cache()
        if step % cfg.log_every == 0:
            log.write(json.dumps(dict(step=step, lr=lr, nll=acc / n_acc, seconds=time.time() - t0,
                                      data_seconds=data_s)) + '\n')
            log.flush()
            acc = n_acc = 0.0
        if step % cfg.eval_every == 0 or step == cfg.steps:
            n_eval += 1
            ev = dict(step=step, seconds=time.time() - t0, figure_entropy_dev=baseline, **dev_nll(model, dev_batches))
            state = {k: v.cpu() for k, v in model.state_dict().items()}
            checkpoint = dict(model=state, config=asdict(net_cfg), members=data.vocab.members, record=record, step=step)
            torch.save(checkpoint, out / 'model.pt')
            torch.save(checkpoint, out / f'model-{step}.pt')
            print(json.dumps(ev), flush=True)
            if cfg.warn_step and step == cfg.warn_step and ev['given'] > cfg.warn_nll:
                killed = f'early warning: dev NLL {ev["given"]:.3f} above {cfg.warn_nll:.3f} at step {step}'
                break
            if n_eval >= cfg.kill_after and ev['given'] >= baseline:
                killed = f'kill rule: dev NLL {ev["given"]:.3f} not below the figure entropy {baseline:.3f}'
                break
        if (time.time() - t0) / 60 > cfg.max_minutes:
            killed = f'time budget of {cfg.max_minutes} minutes at step {step}'
            break
    prefetch.shutdown(cancel_futures=True)
    summary = dict({k: v for k, v in record.items() if k != 'vocab'}, steps_done=step, killed=killed,
                   wall_seconds=time.time() - t0)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(dict(steps_done=step, killed=killed, wall_seconds=summary['wall_seconds'])), flush=True)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    for f in fields(FitConfig):
        kind = (lambda v: v.lower() in ('1', 'true', 'yes')) if isinstance(f.default, bool) else type(f.default)
        p.add_argument('--' + f.name.replace('_', '-'), type=kind, default=f.default, required=f.name in ('data', 'out'))
    torch.set_num_interop_threads(1)
    fit(FitConfig(**vars(p.parse_args(argv))))


if __name__ == '__main__':
    main()
