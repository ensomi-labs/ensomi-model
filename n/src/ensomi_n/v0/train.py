"""N1 v0 training: resumable jobs under 55 minutes, chained, with kill rules and a resource guard.

One run trains for a total budget of training minutes (``total_min``), split into jobs of at most
``job_min`` wall minutes; each job resumes from the newest checkpoint, so a chain of jobs follows one
learning-rate curve (cosine over the total budget in training seconds, after a linear warm-up).

- Batch: 8 songs x 2 crops of 20 s (`data.Sampler`), AdamW, peak lr 1e-3, weight decay 0.01,
  gradient clip 1.0.
- Checkpoints every ``CKPT_EVERY_MIN`` training minutes in the bulk run directory (the last 24 kept,
  so a 150-minute run keeps all of them for checkpoints before the anneal's end, plus every
  checkpoint of the last 30 minutes of the budget for the checkpoint spread).
- Probes every ``PROBE_EVERY_MIN`` training minutes (outside the budget) on val-n1's random half,
  never on the panels: the loss on fixed crops (no augmentation), and M1 (window phase recovery at
  20 ms, `ensomi_n.probe`) of the beat activation on its S0 and hard items.
- Kill rules (guardrail 9; reference scale from `refs`), evaluated at every probe by `kill_verdict`
  from the probe history alone, so a resumed job reaches the same verdict:
  - a non-finite training loss twice, or a non-finite probe;
  - probe S0 M1 under twice chance at minute 12, or under half of BeatThis final0's M1 on the same
    items at minute 24 or, as a running mean of 3 probes, at any probe after it;
  - a stalled probe: the running mean of 3 probe losses not under its best for 6 probes in a row while
    the learning rate is above half its peak (`stalled`);
  - a diverging probe (`diverging`): see its docstring. In v0 this rule was "the probe loss more than
    10 % above its best for 3 probes in a row"; it stopped both v0 seeds (minutes 104 and 80 of 150)
    on swings of the level and phase terms while M1 held, before their anneal.
- Resource guard: allocated MPS tensors over 14 GB or driver memory over 17 GB, or throughput under
  half of this run's first rate for 3 minutes. Memory inputs are padded to 40 s buckets and the MPS
  cache is emptied every 100 steps: the first job's guard fired at minute 3 on driver memory (over
  15 GB, cache included) while per-batch memory shapes varied.

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
KEEP = 24
MEM_LIMIT_GB = 14.0           # tensors allocated (current_allocated_memory)
DRIVER_LIMIT_GB = 17.0        # driver memory, cache included (the MPS limit is 17.76 GB)
EMPTY_CACHE_EVERY = 100
SPREAD_MIN = 30.0


def lr_at(step: int, elapsed_s: float, total_s: float) -> float:
    if step < WARMUP:
        return PEAK_LR * (step + 1) / WARMUP
    frac = min(max(elapsed_s / total_s, 0.0), 1.0)
    return PEAK_LR * (FINAL_LR_FRAC + (1 - FINAL_LR_FRAC) * 0.5 * (1.0 + math.cos(math.pi * frac)))


def _mps_gb() -> tuple[float, float]:
    """(driver memory, allocated tensors) in GB on MPS; NaN elsewhere."""
    import torch
    if torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / 2 ** 30, torch.mps.current_allocated_memory() / 2 ** 30
    return float('nan'), float('nan')


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


def stalled(losses, window: int = 3, patience: int = 6) -> bool:
    """The mean of the last ``window`` probe losses has not gone below its best for ``patience`` probes.

    A single probe at high learning rate moves by about 0.1 on the fixed crops (seed 0, minutes 36-48:
    1.68, 1.65, 1.49, 1.63), so the rule reads a running mean, not single probes.
    """
    if len(losses) < window + patience:
        return False
    sm = np.convolve(np.asarray(losses, dtype=np.float64), np.ones(window) / window, mode='valid')
    best_before = sm[:-patience].min()
    return bool(sm[-patience:].min() >= best_before)


WINDOW = 3            # probes in a running mean
PATIENCE = 3          # consecutive probes a divergence condition must hold
SPREAD_PROBES = 9     # probes over which a probe's own spread is read
SPREAD_K = 3.0        # a rise or a drop must exceed this many spreads
RANGE_FRAC = 0.10     # ... and this share of the loss's learning range
M1_FLOOR = 0.02       # ... and, for M1, this much (chance 0.07, BeatThis final0 0.77-0.85 on the probe)
GROSS_RANGE_FRAC = 0.5
GROSS_M1_DROP = 0.10


def running_mean(x, window: int = WINDOW) -> np.ndarray:
    return np.convolve(np.asarray(x, dtype=np.float64), np.ones(window) / window, mode='valid')


def spread(x) -> float:
    """Robust SD of single probes about their centred 3-probe mean (NaN under 3 probes).

    A centred mean removes a linear trend, so a steady rise does not widen the spread it is read
    against; the residual of one probe about it has SD sqrt(2/3) of the probe's, hence the factor.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 3:
        return float('nan')
    r = x[1:-1] - (x[:-2] + x[1:-1] + x[2:]) / 3.0
    return float(1.4826 * np.median(np.abs(r - np.median(r))) / math.sqrt(2.0 / 3.0))


def m1_mean(h: dict) -> float:
    m = h.get('m1') or {}
    vals = [m[g] for g in ('S0', 'hard') if g in m]
    return float(np.mean(vals)) if vals else float('nan')


def divergence_state(history) -> dict | None:
    """The divergence statistics after the last probe of ``history`` (None before ``WINDOW`` probes).

    ``L`` is the running mean of probe losses, ``M`` that of probe M1 (S0 and hard averaged). The loss
    has risen when ``L`` is above its best by more than ``tau_l`` = max(``SPREAD_K`` x the probe
    loss's spread, ``RANGE_FRAC`` x the learning range, first probe minus best ``L``); M1 has fallen
    when ``M`` is under its best by more than ``tau_m`` = max(``SPREAD_K`` x M1's spread,
    ``M1_FLOOR``). Spreads are read over the last ``SPREAD_PROBES`` probes. Losses are compared by
    difference, not ratio, since the phase and period NLLs make the total free to cross zero.
    """
    if len(history) < WINDOW:
        return None
    loss = [h['probe_loss'] for h in history]
    m1 = [m1_mean(h) for h in history]
    lb, mb = running_mean(loss), running_mean(m1)
    best_l, best_m = float(lb.min()), float(np.nanmax(mb)) if np.isfinite(mb).any() else float('nan')
    rng = max(loss[0] - best_l, 0.0)
    recent = slice(max(0, len(history) - SPREAD_PROBES), len(history))
    sl, sm = spread(loss[recent]), spread(m1[recent])
    tau_l = max(SPREAD_K * (sl if np.isfinite(sl) else 0.0), RANGE_FRAC * rng)
    tau_m = max(SPREAD_K * (sm if np.isfinite(sm) else 0.0), M1_FLOOR)
    rise, drop = float(lb[-1] - best_l), float(best_m - mb[-1])
    return dict(L=float(lb[-1]), best_L=best_l, rise=rise, tau_l=tau_l, range=rng, M=float(mb[-1]), best_M=best_m,
                drop=drop, tau_m=tau_m, loss_up=rise > tau_l, m1_down=drop > tau_m,
                loss_gross=rise > GROSS_RANGE_FRAC * rng > 0.0, m1_gross=drop > GROSS_M1_DROP)


def diverging(history) -> str | None:
    """Real divergence, read on running means and against the probe's own spread.

    Fires when, for ``PATIENCE`` probes in a row, one of these holds (`divergence_state`):
    - the probe loss has risen and M1 has fallen (``loss_up_m1_down``);
    - the probe loss has given back more than half its learning range (``loss_gross``), whatever M1;
    - M1 has fallen by more than ``GROSS_M1_DROP`` (``m1_gross``), whatever the loss.

    A swing of single probes moves neither running mean by its threshold for three probes, and a rise
    of the level and phase terms with M1 holding is not divergence. On v0's two seeds, whose v0 rule
    fired at minutes 104 and 80, none of these fires; the loss condition alone holds at one probe
    (seed 0, minute 104) and M1's never (`n/tests/test_v0_killrule.py`).
    """
    if len(history) < WINDOW + PATIENCE - 1:
        return None
    states = [divergence_state(history[:len(history) - k]) for k in range(PATIENCE)]
    for name, keys in (('loss_up_m1_down', ('loss_up', 'm1_down')), ('loss_gross', ('loss_gross',)),
                       ('m1_gross', ('m1_gross',))):
        if all(s is not None and all(s[k] for k in keys) for s in states):
            return name
    return None


def kill_verdict(history, refs: dict, lr_frac: float) -> str | None:
    """The kill rules read at a probe, from the probe history (the last entry is the current probe).

    ``lr_frac`` is the current learning rate over its peak; ``refs`` carries the probe's chance and
    BeatThis final0 M1 (`ensomi_n.run.v0` refs).
    """
    h = history[-1]
    vals = [h['probe_loss']] + [v for v in (h.get('m1') or {}).values()]
    if not all(np.isfinite(v) for v in vals):
        return 'killed:probe_nonfinite'
    minute = h['minute']
    first = lambda m: next(i for i, x in enumerate(history) if x['minute'] >= m)
    s0 = (h.get('m1') or {}).get('S0', float('nan'))
    if minute >= 12 and first(12) == len(history) - 1 and not s0 >= 2.0 * refs['probe_s0_chance']:
        return 'killed:s0_m1_below_2x_chance_at_12min'
    if minute >= 24:
        half_f0 = 0.5 * refs['probe_s0_final0']
        if first(24) == len(history) - 1 and not s0 >= half_f0:
            return 'killed:s0_m1_below_half_final0_at_24min'
        s0s = [x.get('m1', {}).get('S0', float('nan')) for x in history[first(24):]]
        if len(s0s) >= WINDOW and not running_mean(s0s)[-1] >= half_f0:
            return 'killed:s0_m1_below_half_final0'
    if stalled([x['probe_loss'] for x in history]) and lr_frac > 0.5:
        return 'killed:probe_stalled'
    d = diverging(history)
    if d is not None:
        return f'killed:probe_diverging:{d}'
    return None


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
    st = dict(step=0, elapsed=0.0, history=[], ref_rate=None, best_loss=None, since_best=0)
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
            if device == 'mps' and st['step'] % EMPTY_CACHE_EVERY == 0:
                torch.mps.empty_cache()
            st['elapsed'] += time.time() - t0
            win_n += x.shape[0]
            losses.append(float(loss.detach()))
            for k, v in p.items():
                parts.setdefault(k, []).append(v)
            now = time.time()
            if now - last_prog >= PROGRESS_EVERY_S:
                rate = win_n / (now - win_t)
                mem_gb, alloc_gb = _mps_gb()
                rec = dict(step=st['step'], elapsed_s=round(st['elapsed'], 1), crops_per_s=round(rate, 2),
                           loss=round(float(np.mean(losses)), 5), parts={k: round(float(np.mean(v)), 4) for k, v in parts.items()},
                           lr=round(lr_at(st['step'], st['elapsed'], total_s), 7), mps_gb=round(mem_gb, 2),
                           mps_alloc_gb=round(alloc_gb, 2))
                write('progress.jsonl', rec)
                win_t, win_n, losses, parts, last_prog = now, 0, [], {}, now
                if alloc_gb > MEM_LIMIT_GB or mem_gb > DRIVER_LIMIT_GB:
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
                if device == 'mps':
                    torch.mps.empty_cache()     # probe buffers must not count against the memory guard
                minute = st['elapsed'] / 60.0
                h = dict(step=st['step'], minute=round(minute, 2), probe_loss=round(lv, 5),
                         probe_parts={k: round(float(v.mean()), 4) for k, v in pv.items()},
                         m1={g: round(v, 4) for g, v in m1.items()}, probe_s=round(time.time() - te, 1))
                st['history'].append(h)
                if st['best_loss'] is None or lv < st['best_loss']:
                    st['best_loss'], st['since_best'] = lv, 0
                else:
                    st['since_best'] += 1
                ds = divergence_state(st['history'])
                rule = None if ds is None else {k: (round(v, 5) if isinstance(v, float) else v) for k, v in ds.items()}
                verdict = kill_verdict(st['history'], refs, lr_at(st['step'], st['elapsed'], total_s) / PEAK_LR)
                write('progress.jsonl', dict(event='probe', **h, rule=rule, verdict=verdict))
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
