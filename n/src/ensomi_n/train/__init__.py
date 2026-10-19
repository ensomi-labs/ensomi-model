"""The rung-1 training loop: time-budgeted, checkpointed, resumable, with kill rules and a resource guard.

One call trains one arm (A1 or A2) on NE-S for one seed (design §5.1 R1):

- AdamW, peak lr 2e-3, 300 warm-up steps, then cosine decay over the *wall-clock* training budget
  (the schedule is a function of elapsed training time, so a resumed run follows the same curve);
- batch 32 of 20 s crops from `data.Sampler` (stretch and gain/tilt);
- a checkpoint every 2 minutes of training time (``ckpt-<seconds>.pt``; the last 5 are kept) and a line in
  ``progress.jsonl`` at least every 60 s;
- in-run probes on val-n1's S0 items at every checkpoint (excluded from the budget): M1 of the beat
  activation, with the kill rules of §5.1 (below 2 x chance at minute 4, or below half of BeatThis
  final0's M1 at minute 8);
- the resource guard of §4.3: MPS driver memory over 14 GB, a non-finite loss twice, or throughput under
  half of rung 0's measurement for 3 minutes.

The verdict of each arm (``finished``, ``killed:<rule>``, ``guard:<rule>``) is written to the arm's
``status.json``. ``resume=True`` continues from the newest checkpoint.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import queue
import threading
import time

import numpy as np

from .data import Sampler, song_input, to_torch

PEAK_LR = 2e-3
WARMUP = 300
BATCH = 32
CKPT_EVERY_S = 120.0
PROGRESS_EVERY_S = 60.0
KEEP = 5
MEM_LIMIT_GB = 14.0


def lr_at(step: int, elapsed: float, budget: float) -> float:
    if step < WARMUP:
        return PEAK_LR * (step + 1) / WARMUP
    frac = min(max(elapsed / budget, 0.0), 1.0)
    return PEAK_LR * 0.5 * (1.0 + math.cos(math.pi * frac))


def _mps_gb() -> float:
    import torch
    if torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / 2 ** 30
    return float('nan')


class Prefetch:
    def __init__(self, sampler: Sampler, batch: int, depth: int = 3):
        self.q: queue.Queue = queue.Queue(depth)
        self.stop = False
        self.sampler, self.batch = sampler, batch
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while not self.stop:
            self.q.put(self.sampler.batch(self.batch))

    def get(self):
        return self.q.get()

    def close(self):
        self.stop = True
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass


def infer_song(model, mel: np.ndarray, device: str) -> dict:
    """Head outputs and probe activations for one whole song (numpy, per frame)."""
    import torch
    with torch.inference_mode():
        x = torch.from_numpy(song_input(mel))[None].to(device)
        out = model(x)
        acts = model.heads.activations(out)
    return {k: v[0].float().cpu().numpy() for k, v in acts.items()}


def val_m1(model, val: list[dict], device: str) -> float:
    """Mean M1 of the beat activation over the given val items (each with ``mel``, ``windows``)."""
    from .. import probe
    model.eval()
    vals = []
    for it in val:
        acts = infer_song(model, it['mel'], device)
        vals.append(probe.m1(probe.Activation(acts['beat'], 0.0, 10.0), it['windows']))
    model.train()
    return float(np.nanmean(vals)) if vals else float('nan')


def train_arm(*, arm: str, seed: int, songs, val_s0: list[dict], refs: dict, out_dir, budget_s: float,
              device: str = 'mps', resume: bool = True, r0_crops_per_s: float | None = None, log=print) -> dict:
    import gc
    import torch
    from ..models import NES, Model
    # Release the previous arm's cached MPS blocks: the guard reads driver memory, which includes the
    # cache, and the first mac smoke stopped A2 at 18 GB after A1 had run in the same process.
    gc.collect()
    if device == 'mps':
        torch.mps.empty_cache()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ck_dir = Path(refs['bulk_run']) / f'{arm}-s{seed}'
    ck_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Model(NES(), arm).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=0.01)
    sampler = Sampler(songs, seed=1000 + seed)
    step, elapsed, history = 0, 0.0, []
    ckpts = sorted(ck_dir.glob('ckpt-*.pt'))
    if resume and ckpts:
        st = torch.load(ckpts[-1], map_location=device, weights_only=False)
        model.load_state_dict(st['model'])
        opt.load_state_dict(st['opt'])
        sampler.set_state(st['sampler'])
        step, elapsed, history = st['step'], st['elapsed'], st['history']
        log(f'resumed {arm} s{seed} at step {step}, {elapsed:.0f} s')
    pos_weight = refs['pos_weight']
    progress = out / 'progress.jsonl'
    status = dict(arm=arm, seed=seed, verdict='running')

    def write_progress(rec):
        with open(progress, 'a') as f:
            f.write(json.dumps(dict(rec, t=round(time.time(), 1))) + '\n')

    def save(tag_elapsed):
        path = ck_dir / f'ckpt-{int(round(tag_elapsed)):05d}.pt'
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sampler=sampler.state(), step=step,
                        elapsed=elapsed, history=history, arm=arm, seed=seed), path)
        for old in sorted(ck_dir.glob('ckpt-*.pt'))[:-KEEP]:
            old.unlink()
        return path

    pre = Prefetch(sampler, BATCH)
    nonfinite, slow_since = 0, None
    last_prog, last_ck = time.time(), elapsed
    window_t, window_n, losses = time.time(), 0, []
    model.train()
    verdict = 'finished'
    try:
        while elapsed < budget_s:
            t0 = time.time()
            x, tg = pre.get()
            x, tg = to_torch(x, tg, device)
            for g in opt.param_groups:
                g['lr'] = lr_at(step, elapsed, budget_s)
            o = model(x)
            loss, parts = model.heads.loss(o, tg, pos_weight)
            if not torch.isfinite(loss):
                nonfinite += 1
                opt.zero_grad(set_to_none=True)
                if nonfinite >= 2:
                    verdict = 'guard:nonfinite_loss'
                    break
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if device == 'mps':
                torch.mps.synchronize()
            step += 1
            dt = time.time() - t0
            elapsed += dt
            window_n += BATCH
            losses.append(float(loss.detach()))
            now = time.time()
            if now - last_prog >= PROGRESS_EVERY_S:
                rate = window_n / (now - window_t)
                mem = _mps_gb()
                rec = dict(arm=arm, seed=seed, step=step, elapsed_s=round(elapsed, 1), crops_per_s=round(rate, 2),
                           loss=round(float(np.mean(losses)), 5), parts={k: round(v, 5) for k, v in parts.items()},
                           lr=lr_at(step, elapsed, budget_s), mps_gb=round(mem, 2))
                write_progress(rec)
                log(json.dumps(rec))
                window_t, window_n, losses, last_prog = now, 0, [], now
                if mem > MEM_LIMIT_GB:
                    verdict = 'guard:mps_memory'
                    break
                if r0_crops_per_s and rate < 0.5 * r0_crops_per_s:
                    slow_since = slow_since or now
                    if now - slow_since >= 180:
                        verdict = 'guard:throughput'
                        break
                else:
                    slow_since = None
            if elapsed - last_ck >= CKPT_EVERY_S or elapsed >= budget_s:
                last_ck = elapsed
                te = time.time()
                m = val_m1(model, val_s0, device) if val_s0 else float('nan')
                history.append(dict(step=step, elapsed_s=round(elapsed, 1), val_s0_m1=round(m, 4),
                                    eval_s=round(time.time() - te, 1)))
                path = save(elapsed)
                write_progress(dict(arm=arm, seed=seed, event='checkpoint', path=str(path), **history[-1]))
                minute = elapsed / 60.0
                if 3.9 <= minute < 6.0 and m < 2.0 * refs['val_s0_chance'] and not any(
                        h.get('kill_checked_4') for h in history):
                    history[-1]['kill_checked_4'] = True
                    verdict = 'killed:below_2x_chance_at_4min'
                    break
                if 7.9 <= minute < 10.0 and m < 0.5 * refs['val_s0_final0'] and not any(
                        h.get('kill_checked_8') for h in history):
                    history[-1]['kill_checked_8'] = True
                    verdict = 'killed:below_half_final0_at_8min'
                    break
    finally:
        pre.close()
    if not (ck_dir / f'ckpt-{int(round(elapsed)):05d}.pt').exists():
        save(elapsed)
    status.update(verdict=verdict, step=step, elapsed_s=round(elapsed, 1), history=history,
                  checkpoints=[p.name for p in sorted(ck_dir.glob('ckpt-*.pt'))], ckpt_dir=str(ck_dir),
                  mps_gb_end=round(_mps_gb(), 2))
    (out / 'status.json').write_text(json.dumps(status, indent=1))
    del model, opt, pre
    gc.collect()
    if device == 'mps':
        torch.mps.empty_cache()
    return status
