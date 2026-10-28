"""Paired chart-oracle and one-shot bar measurements on the two open panels."""
import argparse
import dataclasses
import importlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from base import *
from bars import bars_v
from ensomi_n.v0.evaluate import oracle_evidence, v0_evidence

dec = importlib.import_module('ensomi_n.v0.decode')
CFG = dec.DecodeConfig(reset_penalty=14.)
ARMS = ['baseline', 'level', 'down', 'boundary', 'beat_window', 'run_prior', 'run_lr']


def window_for(x, offset):
    ref = x['ref']
    m = shifted(saved('s0', x['item']), offset)
    r, p = notated(ref, m)
    for lo in np.arange(ref.lo_ms, ref.hi_ms, 16000.):
        hi = min(lo + 16000., ref.hi_ms)
        rr = r[(r >= lo) & (r < hi)]
        pp = p[(p >= lo) & (p < hi)]
        if len(rr) and geometry(rr, pp)['r5'] < .9:
            return dict(lo=float(lo), hi=float(hi), failing=True)
    lo = max(ref.lo_ms, (ref.lo_ms + ref.hi_ms - 16000.) / 2)
    return dict(lo=float(lo), hi=float(min(lo + 16000., ref.hi_ms)), failing=False)


def compare_map(a, b):
    same = len(a.beats) == len(b.beats)
    return dict(same_beat_count=same, same_downbeats=np.array_equal(a.downbeats, b.downbeats),
        same_segments=len(a.segments or ()) == len(b.segments or ()),
        max_beat_ms=float(np.max(np.abs(a.beats - b.beats))) if same and len(a.beats) else (0. if same else None))


def checked_baseline(ev, old):
    cache = {}
    orig = dec._track
    def capture(e, c):
        cache['track'] = orig(e, c)
        return tuple(a.copy() for a in cache['track'])
    dec._track = capture
    try:
        pm, diag = dec.decode(ev, CFG)
    finally:
        dec._track = orig
    rep = compare_map(pm, MusicMap.from_json(old))
    if not all(rep[k] for k in ['same_beat_count', 'same_downbeats', 'same_segments']) or rep['max_beat_ms'] > 1.:
        raise RuntimeError('Baseline reproduction guard: ' + str(rep))
    return pm, diag, cache['track'], rep


def decode_arm(ev, oracle, arm, track, window):
    e = dataclasses.replace(ev)
    if arm in ('level', 'down', 'boundary'):
        setattr(e, arm, getattr(oracle, arm))
    elif arm == 'beat_window':
        mask = (10 * np.arange(e.n_frames) >= window['lo']) & (10 * np.arange(e.n_frames) < window['hi'])
        e.beat = np.where(mask, oracle.beat, ev.beat)
        e.beat_off = np.where(mask, oracle.beat_off, ev.beat_off)
    orig_track, orig_bars = dec._track, dec._bars
    if arm in ('level', 'down', 'run_prior', 'run_lr'):
        dec._track = lambda _e, _c: tuple(a.copy() for a in track)
    occupancy = {}
    def bar_pass(e, beats, restart, cfg):
        d = dec._down_at(e, beats, cfg)
        if d is None or not len(beats):
            return orig_bars(e, beats, restart, cfg)
        mask, meters = bars_v(d, cfg, 'run', .2 if arm == 'run_lr' else None, restart)
        values, counts = np.unique(meters, return_counts=True)
        occupancy.update({str(k): int(v) for k, v in zip(values, counts)})
        return mask, meters
    if arm in ('run_prior', 'run_lr'):
        dec._bars = bar_pass
    try:
        pm, diag = dec.decode(e, CFG)
    finally:
        dec._track, dec._bars = orig_track, orig_bars
    return pm, dict(diag, meter_occupancy=occupancy)
