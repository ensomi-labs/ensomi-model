"""N1 v0 training: resumable jobs under 55 minutes, chained, with kill rules and a resource guard.

One run trains for a total budget of training minutes (``total_min``), split into jobs of at most
``job_min`` wall minutes; each job resumes from the newest checkpoint, so a chain of jobs follows one
learning-rate curve (cosine over the total budget in training seconds, after a linear warm-up).

- Batch: 8 songs x 2 crops of 20 s (`data.Sampler`), AdamW, peak lr 1e-3, weight decay 0.01,
  gradient clip 1.0.
- Checkpoints every ``CKPT_EVERY_MIN`` training minutes in the bulk run directory (the last 8 kept,
  plus every checkpoint of the last 30 minutes of the budget for the checkpoint spread).
- Probes every ``PROBE_EVERY_MIN`` training minutes (outside the budget) on val-n1's random half,
  never on the panels: the loss on fixed crops (no augmentation), and M1 (window phase recovery at
  20 ms, `ensomi_n.probe`) of the beat activation on its S0 and hard items.
- Kill rules (guardrail 9; reference scale from `refs`): a non-finite loss twice; probe S0 M1 under
  twice chance from minute 12, or under half of BeatThis final0's M1 on the same items from minute 24;
  a stalled probe: the probe loss not under its best for 5 probes in a row while the learning rate is
  above 30 % of its peak, or more than 10 % above its best for 3 probes in a row; the resource guard
  (MPS driver memory over 15 GB, or throughput under half of this run's first-probe rate for 3 minutes).

Records (synced): ``<run>/progress.jsonl``, ``<run>/jobs.jsonl``, ``<run>/status.json`` and, when the
budget is done or a rule fires, ``<run>/verdict.json``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import queue
import threading
import time

import numpy as np

PEAK_LR = 1e-3
FINAL_LR_FRAC = 0.05
WARMUP = 500
SONGS_PER_BATCH = 8
CROPS_PER_SONG = 2
CKPT_EVERY_MIN = 10.0
PROBE_EVERY_MIN = 4.0
PROGRESS_EVERY_S = 60.0
KEEP = 8
MEM_LIMIT_GB = 15.0
SPREAD_MIN = 30.0


def lr_at(step: int, elapsed_s: float, total_s: float) -> float:
    if step < WARMUP:
        return PEAK_LR * (step + 1) / WARMUP
    frac = min(max(elapsed_s / total_s, 0.0), 1.0)
    return PEAK_LR * (FINAL_LR_FRAC + (1 - FINAL_LR_FRAC) * 0.5 * (1.0 + math.cos(math.pi * frac)))


def _mps_gb() -> float:
    import torch
    if torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / 2 ** 30
    return float('nan')


class Prefetch:
    def __init__(self, sampler, n_songs: int, depth: int = 4):
        self.q: queue.Queue = queue.Queue(depth)
        self.stop = False
        self.err = None
        self.sampler, self.n = sampler, n_songs
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        try:
            while not self.stop:
                self.q.put(self.sampler.batch(self.n))
        except Exception as exc:  # noqa: BLE001 - surfaced by get()
            self.err = exc
            self.q.put(None)

    def get(self):
        b = self.q.get()
        if b is None:
            raise RuntimeError(f'batch thread failed: {self.err!r}')
        return b

    def close(self):
        self.stop = True
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass


def probe_loss(model, crops, songs, pos_weight, device, *, memory: str = 'song', other=None, batch: int = 16):
    """Mean loss over fixed crops, and per-crop totals (memory: song, zero or other)."""
    import torch
    from .data import KEYS_B, KEYS_F, memo_input, pad_memos
    from .model import V0
    model.eval()
    totals, parts_all = [], {}
    with torch.inference_mode():
        for i in range(0, len(crops), batch):
            chunk = crops[i:i + batch]
            x = torch.from_numpy(np.stack([c[0] for c in chunk])).to(device)
            tg = {k: torch.from_numpy(np.stack([c[1][k] for c in chunk])).to(device) for k in KEYS_F + KEYS_B + ('level',)}
            for k in KEYS_F:
                tg[k] = tg[k].float()
            mem = None
            if memory != 'zero':
                idx = [c[2] for c in chunk]
                src = [songs[j] if memory == 'song' else songs[other[j]] for j in idx]
                memo, mask = pad_memos([memo_input(s) for s in src])
                mem = model.encoder.song_memory(torch.from_numpy(memo).to(device), torch.from_numpy(mask).to(device))
            out = model.heads(model.encoder(x, mem, use_memory=mem is not None))
            _, _, items, parts = V0.loss(out, tg, pos_weight, per_item=True)
            totals.append(items.cpu().numpy())
            for k, v in parts.items():
                parts_all.setdefault(k, []).append(v.cpu().numpy())
    model.train()
    tot = np.concatenate(totals)
    return float(tot.mean()), tot, {k: np.concatenate(v) for k, v in parts_all.items()}


def probe_m1(model, items, device) -> dict:
    """Mean M1 of the beat activation over probe items (each: song, windows, group)."""
    from .. import probe
    from .infer import song_outputs
    model.eval()
    vals: dict[str, list] = {}
    for it in items:
        acts = song_outputs(model, it['song'], device)
        vals.setdefault(it['group'], []).append(probe.m1(probe.Activation(acts['beat'], 0.0, 10.0), it['windows']))
    model.train()
    return {g: float(np.nanmean(v)) for g, v in vals.items()}


def train(*, songs, probe_crops, probe_songs, probe_items, refs: dict, run: Path, ck_dir: Path, seed: int,
          total_min: float, job_min: float, device: str = 'mps', log=print, model_kw: dict | None = None) -> dict:
    import torch
    from .data import Sampler, to_torch
    from .model import V0, n_params
    run.mkdir(parents=True, exist_ok=True)
    ck_dir.mkdir(parents=True, exist_ok=True)
    t_job = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = V0(**(model_kw or {})).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=0.01)
    sampler = Sampler(songs, seed=1000 + seed, crops_per_song=CROPS_PER_SONG)
    st = dict(step=0, elapsed=0.0, history=[], ref_rate=None, best_loss=None, since_best=0, above=0, kill_checked=[])
    ckpts = sorted(ck_dir.glob('ckpt-*.pt'))
    if ckpts:
        saved = torch.load(ckpts[-1], map_location=device, weights_only=False)
        model.load_state_dict(saved['model'])
        opt.load_state_dict(saved['opt'])
        sampler.set_state(saved['sampler'])
        st.update(saved['state'])
        log(f"resumed from {ckpts[-1].name}: step {st['step']}, {st['elapsed']:.0f} s")
    total_s = 60.0 * total_min
    pos_weight = refs['pos_weight']

    def write(name, rec):
        with open(run / name, 'a') as f:
            f.write(json.dumps(dict(rec, t=round(time.time(), 1))) + '\n')

    def save(tag=None):
        path = ck_dir / f"ckpt-{int(round(st['elapsed'])):06d}.pt"
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sampler=sampler.state(), state=st, seed=seed,
                        model_kw=model_kw or {}), path)
        olds = sorted(ck_dir.glob('ckpt-*.pt'))
        for old in olds[:-KEEP]:
            secs = int(old.stem.split('-')[1])
            if secs < total_s - 60.0 * SPREAD_MIN - 1:
                old.unlink()
        return path

    write('jobs.jsonl', dict(event='job_start', step=st['step'], elapsed_s=round(st['elapsed'], 1),
                             params=n_params(model), songs=len(songs)))
    pre = Prefetch(sampler, SONGS_PER_BATCH)
    nonfinite = 0
    slow_since = None
    last_prog = time.time()
    last_ck = st['elapsed'] // (60 * CKPT_EVERY_MIN)
    last_probe = st['elapsed'] // (60 * PROBE_EVERY_MIN)
    win_t, win_n, losses, parts = time.time(), 0, [], {}
    verdict = None
    model.train()
    try:
        while st['elapsed'] < total_s and time.time() - t_job < 60.0 * job_min:
            t0 = time.time()
            x, tg, memo, mask, owner = pre.get()
            x, tg, memo, mask = to_torch(x, tg, memo, mask, device)
            for g in opt.param_groups:
                g['lr'] = lr_at(st['step'], st['elapsed'], total_s)
            mem = model.encoder.song_memory(memo, mask)
            mem = mem[torch.from_numpy(owner).to(device)]
            out = model.heads(model.encoder(x, mem))
            loss, p = V0.loss(out, tg, pos_weight)
            if not torch.isfinite(loss):
                nonfinite += 1
                opt.zero_grad(set_to_none=True)
                write('progress.jsonl', dict(event='nonfinite_loss', step=st['step']))
                if nonfinite >= 2:
                    verdict = 'killed:nonfinite_loss'
                    break
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if device == 'mps':
                torch.mps.synchronize()
            st['step'] += 1
            st['elapsed'] += time.time() - t0
            win_n += x.shape[0]
            losses.append(float(loss.detach()))
            for k, v in p.items():
                parts.setdefault(k, []).append(v)
            now = time.time()
            if now - last_prog >= PROGRESS_EVERY_S:
                rate = win_n / (now - win_t)
                mem_gb = _mps_gb()
                rec = dict(step=st['step'], elapsed_s=round(st['elapsed'], 1), crops_per_s=round(rate, 2),
                           loss=round(float(np.mean(losses)), 5), parts={k: round(float(np.mean(v)), 4) for k, v in parts.items()},
                           lr=round(lr_at(st['step'], st['elapsed'], total_s), 7), mps_gb=round(mem_gb, 2))
                write('progress.jsonl', rec)
                win_t, win_n, losses, parts, last_prog = now, 0, [], {}, now
                if mem_gb > MEM_LIMIT_GB:
                    verdict = 'guard:mps_memory'
                    break
                if st['ref_rate'] is None and st['step'] > 200:
                    st['ref_rate'] = rate
                if st['ref_rate'] and rate < 0.5 * st['ref_rate']:
                    slow_since = slow_since or now
                    if now - slow_since >= 180:
                        verdict = 'guard:throughput'
                        break
                else:
                    slow_since = None
            # probes and kill rules
            k_probe = st['elapsed'] // (60 * PROBE_EVERY_MIN)
            if k_probe > last_probe or st['elapsed'] >= total_s:
                last_probe = k_probe
                te = time.time()
                lv, _, pv = probe_loss(model, probe_crops, probe_songs, pos_weight, device)
                m1 = probe_m1(model, probe_items, device)
                minute = st['elapsed'] / 60.0
                h = dict(step=st['step'], minute=round(minute, 2), probe_loss=round(lv, 5),
                         probe_parts={k: round(float(v.mean()), 4) for k, v in pv.items()},
                         m1={g: round(v, 4) for g, v in m1.items()}, probe_s=round(time.time() - te, 1))
                st['history'].append(h)
                write('progress.jsonl', dict(event='probe', **h))
                if st['best_loss'] is None or lv < st['best_loss']:
                    st['best_loss'], st['since_best'] = lv, 0
                else:
                    st['since_best'] += 1
                st['above'] = st['above'] + 1 if lv > 1.10 * st['best_loss'] else 0
                s0 = m1.get('S0', float('nan'))
                if minute >= 12 and 'chance' not in st['kill_checked']:
                    st['kill_checked'].append('chance')
                    if not s0 >= 2.0 * refs['probe_s0_chance']:
                        verdict = 'killed:s0_m1_below_2x_chance_at_12min'
                if minute >= 24 and 'final0' not in st['kill_checked'] and verdict is None:
                    st['kill_checked'].append('final0')
                    if not s0 >= 0.5 * refs['probe_s0_final0']:
                        verdict = 'killed:s0_m1_below_half_final0_at_24min'
                if verdict is None and st['since_best'] >= 5 and lr_at(st['step'], st['elapsed'], total_s) > 0.3 * PEAK_LR:
                    verdict = 'killed:probe_stalled'
                if verdict is None and st['above'] >= 3:
                    verdict = 'killed:probe_diverging'
                if verdict is not None:
                    break
            k_ck = st['elapsed'] // (60 * CKPT_EVERY_MIN)
            if k_ck > last_ck:
                last_ck = k_ck
                path = save()
                write('progress.jsonl', dict(event='checkpoint', path=path.name, step=st['step'],
                                             elapsed_s=round(st['elapsed'], 1)))
    finally:
        pre.close()
    path = save()
    done = st['elapsed'] >= total_s
    status = dict(step=st['step'], elapsed_s=round(st['elapsed'], 1), total_s=total_s, last_checkpoint=path.name,
                  verdict=verdict or ('finished' if done else 'job_done'), job_wall_s=round(time.time() - t_job, 1),
                  checkpoints=[p.name for p in sorted(ck_dir.glob('ckpt-*.pt'))], ck_dir=str(ck_dir))
    (run / 'status.json').write_text(json.dumps(status, indent=1))
    write('jobs.jsonl', dict(event='job_end', **{k: v for k, v in status.items() if k != 'checkpoints'}))
    if verdict is not None or done:
        (run / 'verdict.json').write_text(json.dumps(dict(status, history=st['history']), indent=1))
    return status
