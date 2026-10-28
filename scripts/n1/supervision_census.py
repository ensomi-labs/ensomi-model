"""Latent-supervision census: what the charts on hand say beyond their red lines.

Labels only. The script parses `.osu` text, the 4K corpus table, std-taiko set metadata, the
dedup groups, the audio inventory (paths and sha256 recorded earlier) and file names in set
folders. It never opens an audio file.

Signals, per mode (4K mania, osu!standard, taiko):

a. difficulty ladders: per pair of difficulties on one audio file, the share of the easier one's
   onsets present in the harder one (within ``TOL`` ms); onset shares by metrical position per
   ladder rank and star band; per onset of the hardest difficulty, how many easier ones keep it,
   by metrical position and by accent (chord size, finish/clap, big note);
b. snap: onsets by divisor of the active red line (1, 2, 3, 4, 6, 8, 12, 16; else unsnapped, with
   5, 7, 24 and 32 noted), beat and bar classes, ternary runs;
c. hitsounds: whistle, finish and clap bits, sample sets and indices, by metrical position and
   tempo band; taiko don, kat and big notes; standard new combos;
d. keysounds: per-object sample files, storyboard samples, and sample files in set folders (names);
e. sections: kiai, breaks, bookmarks, volume changes;
f. repetition: per-bar onset patterns on the chart's grid, their recurrence, 4-bar blocks, and
   recurrence between kiai sections against non-kiai windows;
g. timing: red lines per chart, meters and meter changes, timing shared across difficulties, modes
   and sets of one audio file, and the agreement of independent timings of one file.

Sealed sets are excluded before any of their files is opened. A set folder is dropped when it is
linked, through corpus group ids, audio sha256, the dedup groups or the audio inventory, to any of:
timing holdout-1 or holdout-2 (audio sha256), a dedup member flagged holdout, a corpus chart with
``eval_split = heldout``, or the std-taiko hard holdout (its membership, checked against its
sha256). This is a superset of the census's sealed splits (``holdout1``, ``holdout2``,
``r2-heldout``, ``std-taiko-linked-sealed``, ``hard-holdout``). Outputs hold rows and examples
for unsealed sets only, and counts of dropped folders.

Run from the project root::

    python scripts/n1/supervision_census.py --out artifacts/n1-redesign-20261028/census --commit SHA
    python scripts/n1/supervision_census.py --selftest
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import unicodedata

import numpy as np

TOL = 2.0                                   # ms: snap, onset match and kiai tolerance
DIVS = (1, 2, 3, 4, 6, 8, 12, 16)           # coarsest first
DIAG_DIVS = (5, 7, 24, 32)                  # noted inside "unsnapped"
POS = ('db', 'b2', 'b3', 'b4', 'bx', 'h2', 'q4', 'e8', 's16', 't3', 't6', 't12', 'off')
CLS_POS = {2: 'h2', 4: 'q4', 8: 'e8', 16: 's16', 3: 't3', 6: 't6', 12: 't12'}
BANDS = ('lt100', '100-160', 'ge160')
WHISTLE, FINISH, CLAP = 2, 4, 8
Q = 48                                      # pattern quantum per beat (lcm of 16 and 12)
STATUSES_4K = ('ranked', 'loved')
AUDIO_EXT = ('.wav', '.ogg', '.mp3', '.flac')
HITSOUND_NAME = re.compile(r'^(normal|soft|drum)-(hit(normal|whistle|finish|clap)|slider(slide|whistle|tick))\d*\.(wav|ogg|mp3)$')
HEADER_SECTIONS = ('General', 'Metadata', 'Difficulty', 'Editor')
MODES = ('mania4k', 'std', 'taiko')


# ---------------------------------------------------------------- parsing

def _num(s, default=0.0):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def _int(s, default=0):
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return default


def parse_osu(data: bytes) -> dict:
    """The parts of a `.osu` file this census reads."""
    text = data.decode('utf-8-sig', errors='replace')
    sec = ''
    kv: dict[tuple[str, str], str] = {}
    tps, objs, breaks = [], [], []
    sb_samples = 0
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith('//'):
            continue
        if s[0] == '[' and s[-1] == ']':
            sec = s[1:-1].strip()
            continue
        if sec in HEADER_SECTIONS:
            k, sep, v = s.partition(':')
            if sep:
                kv[(sec, k.strip())] = v.strip()
        elif sec == 'TimingPoints':
            p = s.split(',')
            if len(p) < 2:
                continue
            t, L = _num(p[0], math.nan), _num(p[1], math.nan)
            if not (math.isfinite(t) and math.isfinite(L)):
                continue
            meter = _int(p[2], 4) if len(p) > 2 else 4
            red = (p[6].strip() == '1') if len(p) > 6 else L > 0
            tps.append(dict(t=t, L=L, meter=meter if meter > 0 else 4, sset=_int(p[3]) if len(p) > 3 else 0,
                            sidx=_int(p[4]) if len(p) > 4 else 0, vol=_int(p[5], 100) if len(p) > 5 else 100,
                            red=red, kiai=bool(_int(p[7]) & 1) if len(p) > 7 else False))
        elif sec == 'HitObjects':
            p = s.split(',')
            if len(p) < 5:
                continue
            t = _num(p[2], math.nan)
            if not math.isfinite(t):
                continue
            typ, hs = _int(p[3]), _int(p[4])
            o = dict(t=t, typ=typ, hs=hs, x=_num(p[0]), end=None, edges=None, slides=0, length=0.0)
            samp: list[str] = []
            if typ & 128:
                q = p[5].split(':') if len(p) > 5 else []
                o['end'] = _num(q[0], math.nan) if q else math.nan
                samp = q[1:]
            elif typ & 2:
                o['slides'] = max(1, _int(p[6], 1)) if len(p) > 6 else 1
                o['length'] = _num(p[7]) if len(p) > 7 else 0.0
                if len(p) > 8 and p[8]:
                    o['edges'] = [_int(e) for e in p[8].split('|')]
                samp = p[10].split(':') if len(p) > 10 else []
            elif typ & 8:
                o['end'] = _num(p[5], math.nan) if len(p) > 5 else math.nan
                samp = p[6].split(':') if len(p) > 6 else []
            else:
                samp = p[5].split(':') if len(p) > 5 else []
            o['nset'] = _int(samp[0]) if len(samp) > 0 else 0
            o['aset'] = _int(samp[1]) if len(samp) > 1 else 0
            o['idx'] = _int(samp[2]) if len(samp) > 2 else 0
            o['fname'] = samp[4].strip() if len(samp) > 4 else ''
            objs.append(o)
        elif sec == 'Events':
            if raw[:1] in (' ', '_'):
                continue
            p = s.split(',')
            if p[0].strip() in ('2', 'Break') and len(p) >= 3:
                a, b = _num(p[1], math.nan), _num(p[2], math.nan)
                if math.isfinite(a) and math.isfinite(b):
                    breaks.append((a, b))
            elif p[0].strip() in ('5', 'Sample'):
                sb_samples += 1
    g = lambda sec_, k, d='': kv.get((sec_, k), d)
    bookmarks = [_num(x, math.nan) for x in g('Editor', 'Bookmarks').split(',') if x.strip()]
    tps.sort(key=lambda d: (d['t'], 0 if d['red'] else 1))
    reds: list[tuple[float, float, int]] = []
    for d in tps:
        if d['red'] and d['L'] > 0:
            if reds and abs(reds[-1][0] - d['t']) < 0.5:
                reds[-1] = (d['t'], d['L'], d['meter'])           # same time: the later line applies
            else:
                reds.append((d['t'], d['L'], d['meter']))
    objs.sort(key=lambda o: o['t'])
    return dict(mode=_int(g('General', 'Mode', '0')), audio=g('General', 'AudioFilename'),
                keys=_num(g('Difficulty', 'CircleSize', '0')), sm=_num(g('Difficulty', 'SliderMultiplier', '1.4'), 1.4) or 1.4,
                version=g('Metadata', 'Version'), creator=g('Metadata', 'Creator'),
                tps=tps, reds=reds, objs=objs, breaks=breaks, sb_samples=sb_samples,
                bookmarks=[b for b in bookmarks if math.isfinite(b)])


# ---------------------------------------------------------------- the chart's grid

class Grid:
    """A chart's red lines as osu! applies them: bars restart at every red line."""

    def __init__(self, reds, end_ms: float):
        self.T = np.array([r[0] for r in reds], float)
        self.L = np.array([r[1] for r in reds], float)
        self.M = np.array([r[2] for r in reds], np.int64)
        nxt = np.append(self.T[1:], max(end_ms, self.T[-1]))
        span = (nxt - self.T) / self.L
        self.C = np.concatenate([[0.0], np.cumsum(span[:-1])])          # beat coordinate at each red line
        bars = np.maximum(np.ceil(span / self.M - 1e-6), 1).astype(np.int64)
        self.barofs = np.concatenate([[0], np.cumsum(bars[:-1])]).astype(np.int64)

    def locate(self, t) -> dict:
        t = np.asarray(t, float).reshape(-1)
        seg = np.clip(np.searchsorted(self.T, t + TOL, 'right') - 1, 0, len(self.T) - 1)
        L, m = self.L[seg], self.M[seg]
        x = (t - self.T[seg]) / L
        cls = np.zeros(len(t), np.int64)
        for d in DIVS:
            err = np.abs(x * d - np.round(x * d)) / d * L
            cls[(cls == 0) & (err <= TOL)] = d
        for d in DIAG_DIVS:
            err = np.abs(x * d - np.round(x * d)) / d * L
            cls[(cls == 0) & (err <= TOL)] = -d
        eps = TOL / L
        beat = np.floor(x + eps).astype(np.int64)
        bib = np.mod(beat, m)
        bar = np.floor((x + eps) / m).astype(np.int64)
        pos = np.full(len(t), POS.index('off'), np.int64)
        on = cls == 1
        pos[on & (bib == 0)] = POS.index('db')
        four = on & (bib > 0) & (m == 4)
        for k, name in ((1, 'b2'), (2, 'b3'), (3, 'b4')):
            pos[four & (bib == k)] = POS.index(name)
        pos[on & (bib > 0) & (m != 4)] = POS.index('bx')
        for c, name in CLS_POS.items():
            pos[cls == c] = POS.index(name)
        bpm = 60000.0 / L
        band = np.where(bpm < 100, 0, np.where(bpm < 160, 1, 2))
        return dict(seg=seg, x=x, cls=cls, beat=beat, bib=bib, bar=bar, gbar=self.barofs[seg] + bar,
                    pos48=np.round((x - bar * m) * Q).astype(np.int64), B=self.C[seg] + x, meter=m, L=L,
                    pos=pos, band=band)

    def beat_lines(self, lo: float, hi: float, cap: int = 20000) -> np.ndarray:
        out = []
        n = len(self.T)
        for i in range(n):
            a = self.T[i]
            b = self.T[i + 1] if i + 1 < n else hi + self.L[i]
            if b < lo or a > hi or not (10.0 <= 60000.0 / self.L[i] <= 1000.0):
                continue
            k0 = max(0, int(math.ceil((max(a, lo) - a) / self.L[i] - 1e-9)))
            k1 = int(math.floor((min(b, hi) - a) / self.L[i] - 1e-9))
            if k1 >= k0:
                out.append(a + self.L[i] * np.arange(k0, min(k1 + 1, k0 + cap)))
        return np.unique(np.concatenate(out)) if out else np.zeros(0)


def kiai_sections(tps, end_ms: float) -> list[tuple[float, float]]:
    out, on, start = [], False, 0.0
    for d in tps:
        if d['kiai'] and not on:
            on, start = True, d['t']
        elif not d['kiai'] and on:
            on = False
            out.append((start, d['t']))
    if on:
        out.append((start, max(end_ms, start)))
    return out


def near(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each of ``a`` to the nearest of sorted ``b``."""
    if len(b) == 0:
        return np.full(len(a), np.inf)
    i = np.clip(np.searchsorted(b, a), 1, len(b) - 1) if len(b) > 1 else np.zeros(len(a), np.int64)
    d = np.abs(a - b[i])
    if len(b) > 1:
        d = np.minimum(d, np.abs(a - b[i - 1]))
    return d


def same_timing(r1, r2) -> bool:
    if len(r1) != len(r2):
        return False
    return all(abs(a[0] - b[0]) <= 1.0 and abs(a[1] / b[1] - 1) <= 1e-6 and a[2] == b[2] for a, b in zip(r1, r2))


def grid_agreement(g1: Grid, r1: tuple, g2: Grid, r2: tuple) -> dict | None:
    """Median distance between two timings' beat lines over their common mapped range (either way, the smaller)."""
    lo, hi = max(r1[0], r2[0]), min(r1[1], r2[1])
    if hi - lo < 10000:
        return None
    b1, b2 = g1.beat_lines(lo, hi), g2.beat_lines(lo, hi)
    if len(b1) < 8 or len(b2) < 8:
        return None
    d12, d21 = near(b1, b2), near(b2, b1)
    m12, m21 = float(np.median(d12)), float(np.median(d21))
    coarse = d12 if m12 <= m21 else d21
    return dict(median=min(m12, m21), within5=float(np.mean(coarse <= 5.0)), within2=float(np.mean(coarse <= TOL)))


# ---------------------------------------------------------------- one chart

def analyse(p: dict, mode: str) -> dict | None:
    """Per-chart statistics and the arrays the set-level passes need, or None without timing or objects."""
    reds, objs = p['reds'], p['objs']
    if not reds or not objs:
        return None
    starts = np.array([o['t'] for o in objs])
    ends = [o['end'] for o in objs if o['end'] is not None and math.isfinite(o['end'])]
    end_ms = float(max([starts[-1]] + ends))
    g = Grid(reds, end_ms)
    on_t, inv = np.unique(starts, return_inverse=True)
    n_on = len(on_t)
    loc = g.locate(on_t)
    chord = np.bincount(inv, minlength=n_on)
    head_hs = np.array([(o['edges'][0] if o['edges'] else o['hs']) if o['typ'] & 2 else o['hs'] for o in objs], np.int64)
    circle = np.array([bool(o['typ'] & 1) for o in objs])
    on_bits = np.zeros(n_on, np.int64)
    np.bitwise_or.at(on_bits, inv, head_hs)
    nc = np.zeros(n_on, bool)
    np.logical_or.at(nc, inv, np.array([bool(o['typ'] & 4) for o in objs]))
    kat = np.zeros(n_on, bool)
    big = np.zeros(n_on, bool)
    don = np.zeros(n_on, bool)
    if mode == 'taiko':
        np.logical_or.at(kat, inv, circle & ((head_hs & (WHISTLE | CLAP)) != 0))
        np.logical_or.at(big, inv, circle & ((head_hs & FINISH) != 0))
        np.logical_or.at(don, inv, circle & ((head_hs & (WHISTLE | CLAP)) == 0))
    r = dict(mode=mode, n_obj=len(objs), n_on=n_on, dur_s=round((end_ms - starts[0]) / 1000.0, 2), n_red=len(reds),
             n_green=sum(1 for d in p['tps'] if not d['red']), version=p['version'])
    # b. snap and subdivision on unique onsets
    cls = loc['cls']
    for d in DIVS:
        r[f's{d}'] = int(np.sum(cls == d))
    r['s_off'] = int(np.sum(cls <= 0))
    for d in DIAG_DIVS:
        r[f'off{d}'] = int(np.sum(cls == -d))
    bpm = 60000.0 / loc['L']
    r['on_implausible_bpm'] = int(np.sum((bpm < 40) | (bpm > 400)))
    # a beat's divisor is the lcm of its snapped onsets' divisors: 3 divides it when a 1/3, 1/6 or 1/12
    # onset is present, 4 when a 1/4, 1/8, 1/12 or 1/16 one is (S1's classes)
    snapped = cls > 0
    bkey = loc['seg'][snapped] * 10_000_000 + loc['beat'][snapped] + 1_000_000
    ub, first, bi = np.unique(bkey, return_index=True, return_inverse=True)
    d3 = np.zeros(len(ub), bool)
    d4 = np.zeros(len(ub), bool)
    np.logical_or.at(d3, bi, np.isin(cls[snapped], (3, 6, 12)))
    np.logical_or.at(d4, bi, np.isin(cls[snapped], (4, 8, 12, 16)))
    bgbar = loc['gbar'][snapped][first]
    r.update(beats=len(ub), tern_beats=int(d3.sum()), bin_beats=int((d4 & ~d3).sum()), mixed_beats=int((d3 & d4).sum()))
    bars_all = np.unique(loc['gbar'])
    tern_bars = sorted(set(bgbar[d3].tolist()))
    bin_bars = set(bgbar[d4 & ~d3].tolist())
    runs = []
    for b in tern_bars:
        if runs and b - runs[-1][1] <= 2:
            runs[-1][1] = b
        else:
            runs.append([b, b])
    run_len = [b - a + 1 for a, b in runs]
    r.update(bars=len(bars_all), tern_bars=len(tern_bars), bin_bars=len(bin_bars),
             both_bars=len(set(tern_bars) & bin_bars), tern_runs=len(runs), tern_maxrun=max(run_len) if run_len else 0)
    # g. meters on segments that hold onsets
    segs_used = sorted(set(loc['seg'].tolist()))
    meters = [int(g.M[s]) for s in segs_used]
    r['meters'] = ';'.join(str(m) for m in sorted(set(meters)))
    r['meter_changes'] = sum(1 for a, b in zip(meters, meters[1:]) if a != b)
    r['non4_onsets'] = int(np.sum(loc['meter'] != 4))
    dom = np.bincount(loc['seg'], minlength=len(reds)).argmax()
    r['bpm_dom'] = round(60000.0 / float(g.L[dom]), 3)
    # e. sections
    kiai = kiai_sections(p['tps'], end_ms)
    kiai_info = []
    for a, b in kiai:
        la, lb = g.locate([a, b])['B']
        st = g.locate([a])
        kiai_info.append(dict(a=a, b=b, sec=(b - a) / 1000.0, beats=float(lb - la),
                              meter=int(st['meter'][0]), db=bool(st['pos'][0] == POS.index('db'))))
    real = [k for k in kiai_info if k['beats'] >= k['meter'] - 0.5]
    r['n_kiai'] = len(real)
    r['n_kiai_flash'] = len(kiai_info) - len(real)
    r['kiai_beats'] = ';'.join(str(round(k['beats'], 2)) for k in real)
    r['kiai_db'] = sum(k['db'] for k in real)
    lens = [round(k['beats']) for k in real]
    r['kiai_shared_len'] = sum(1 for i, x in enumerate(lens) if any(abs(x - y) <= 1 for j, y in enumerate(lens) if j != i))
    r['n_breaks'] = len(p['breaks'])
    r['break_s'] = ';'.join(str(round((b - a) / 1000.0, 1)) for a, b in p['breaks'])
    bm = np.array(sorted(p['bookmarks']))
    r['n_bookmarks'] = len(bm)
    if len(bm):
        bl = g.locate(bm)
        r['bookmarks_db'] = int(np.sum(bl['pos'] == POS.index('db')))
        r['bookmarks_on_beat'] = int(np.sum(bl['cls'] == 1))
        r['bookmark_gap_bars'] = round(float(np.median(np.diff(bl['B']) / bl['meter'][:-1])), 2) if len(bm) > 1 else ''
    else:
        r.update(bookmarks_db=0, bookmarks_on_beat=0, bookmark_gap_bars='')
    vols = [d['vol'] for d in p['tps']]
    r['vol_changes'] = sum(1 for a, b in zip(vols, vols[1:]) if a != b)
    # c/d. hitsounds, samples, keysounds
    sv_t = np.array([d['t'] for d in p['tps']])
    sv_red = np.array([d['red'] for d in p['tps']])
    sv_L = np.array([d['L'] for d in p['tps']])
    ev_t, ev_bits, ev_edge = list(on_t), list(on_bits), [0] * n_on
    if mode == 'std':
        for o, hh in zip(objs, head_hs):
            if not (o['typ'] & 2) or o['length'] <= 0:
                continue
            k = int(np.searchsorted(sv_t, o['t'] + 0.5, 'right')) - 1
            sv = 1.0
            if 0 <= k < len(sv_t) and not sv_red[k] and sv_L[k] < 0:
                sv = min(max(-100.0 / sv_L[k], 0.1), 10.0)
            Lr = float(g.L[np.clip(np.searchsorted(g.T, o['t'] + TOL, 'right') - 1, 0, len(g.T) - 1)])
            dur = o['length'] / (p['sm'] * 100.0 * sv) * Lr
            for i in range(1, o['slides'] + 1):
                ev_t.append(o['t'] + i * dur)
                ev_bits.append(o['edges'][i] if o['edges'] and len(o['edges']) > i else o['hs'])
                ev_edge.append(1)
    ev_t = np.array(ev_t, float)
    ev_bits = np.array(ev_bits, np.int64)
    ev_edge = np.array(ev_edge, np.int64)
    ev = g.locate(ev_t) if len(ev_t) > n_on else loc
    any_bits = (ev_bits & (WHISTLE | FINISH | CLAP)) != 0
    r.update(n_ev=len(ev_t), n_ev_edge=int(ev_edge.sum()), ev_hs=int(any_bits.sum()),
             n_whistle=int(np.sum((ev_bits & WHISTLE) != 0)), n_finish=int(np.sum((ev_bits & FINISH) != 0)),
             n_clap=int(np.sum((ev_bits & CLAP) != 0)))
    clap = (ev_bits & CLAP) != 0
    fin = (ev_bits & FINISH) != 0
    beat4 = (ev['cls'] == 1) & (ev['meter'] == 4)
    bb = beat4 & ((ev['bib'] == 1) | (ev['bib'] == 3))
    r.update(clap_beat4=int(np.sum(clap & beat4)), clap_bb=int(np.sum(clap & bb)), ev_bb=int(np.sum(bb)),
             ev_bb_clap=int(np.sum(bb & clap)), fin_db=int(np.sum(fin & (ev['pos'] == POS.index('db')))),
             ev_db=int(np.sum(ev['pos'] == POS.index('db'))))
    r['n_nc'] = int(nc.sum())
    r['nc_db'] = int(np.sum(nc & (loc['pos'] == POS.index('db'))))
    r['n_kat'], r['n_big'], r['n_don'] = int(kat.sum()), int(big.sum()), int(don.sum())
    r['chord2'] = int(np.sum(chord >= 2))
    fnames = [o['fname'] for o in objs if o['fname']]
    r['ks_obj'] = len(fnames)
    r['ks_files'] = len(set(f.casefold() for f in fnames))
    r['obj_idx'] = sum(1 for o in objs if o['idx'] > 0)
    r['tp_sidx_max'] = max((d['sidx'] for d in p['tps']), default=0)
    r['tp_sidx_custom'] = sum(1 for d in p['tps'] if d['sidx'] >= 1)
    r['sample_sets'] = ';'.join(str(x) for x in sorted({d['sset'] for d in p['tps']} |
                                                         {o['nset'] for o in objs if o['nset']} | {o['aset'] for o in objs if o['aset']}))
    r['sb_samples'] = p['sb_samples']
    holds = sum(1 for o in objs if o['typ'] & 128)
    r['holds'] = holds
    side = dict(grid=g, reds=reds, on_t=on_t, loc=loc, chord=chord, bits=on_bits, kat=kat, big=big, nc=nc,
                ev=ev, ev_bits=ev_bits, ev_edge=ev_edge, kiai=[(k['a'], k['b']) for k in real],
                kiai_all=[(k['a'], k['b']) for k in kiai_info], span=(float(starts[0]), end_ms), fnames=fnames)
    return dict(rec=r, side=side)


# ---------------------------------------------------------------- repetition

def _trivial(meter: int, pat: tuple) -> bool:
    if len(pat) <= 2:
        return True
    return any(pat == tuple(range(0, meter * Q, step)) for step in (Q, Q // 2, Q // 4))


def repetition(side: dict) -> dict:
    loc = side['loc']
    bars: dict[int, set] = defaultdict(set)
    meter_of: dict[int, int] = {}
    first_t: dict[int, float] = {}
    for gb, p48, m, t in zip(loc['gbar'].tolist(), loc['pos48'].tolist(), loc['meter'].tolist(), side['on_t'].tolist()):
        bars[gb].add(p48)
        meter_of[gb] = m
        first_t.setdefault(gb, t)
    if len(bars) < 8:
        return {}
    key = {b: (meter_of[b], tuple(sorted(v))) for b, v in bars.items()}
    triv = {b: _trivial(meter_of[b], key[b][1]) for b in bars}
    occ: dict[tuple, list] = defaultdict(list)
    for b in sorted(bars):
        occ[key[b]].append(b)
    order = sorted(bars)

    def rec(sel, far):
        n = hit = 0
        for b in sel:
            n += 1
            hit += any(o != b and abs(o - b) >= far for o in occ[key[b]])
        return n, hit

    out = {}
    nt = [b for b in order if not triv[b]]
    for name, sel in (('all', order), ('nontrivial', nt)):
        n, h1 = rec(sel, 1)
        _, h8 = rec(sel, 8)
        out[f'bars_{name}'] = n
        out[f'rec_{name}'] = h1
        out[f'rec8_{name}'] = h8
    out['distinct'] = len(occ)
    blocks: dict[tuple, list] = defaultdict(list)
    for b in order:
        if all((b + i) in bars for i in range(4)):
            blocks[tuple(key[b + i] for i in range(4))].append(b)
    nb = sum(len(v) for v in blocks.values())
    nbr = sum(sum(any(abs(o - b) >= 4 for o in v) for b in v) for v in blocks.values())
    nbn = [b for k, v in blocks.items() if not all(_trivial(m, p) for m, p in k) for b in v]
    nbnr = sum(sum(any(abs(o - b) >= 4 for o in blocks[k]) for b in v)
               for k, v in blocks.items() if not all(_trivial(m, p) for m, p in k))
    out.update(blocks=nb, blocks_rec=nbr, blocks_nt=len(nbn), blocks_nt_rec=nbnr)
    # kiai: which section each note-bearing bar starts in
    secs = side['kiai'][:8]
    sec_of = {}
    for b in order:
        t = first_t[b]
        for i, (a, e) in enumerate(secs):
            if a - TOL <= t < e - TOL:
                sec_of[b] = i
                break
    in_k = [b for b in order if b in sec_of]
    out_k = [b for b in order if b not in sec_of]
    for name, sel in (('kiai', in_k), ('nonkiai', out_k)):
        n, h = rec(sel, 8)
        out[f'n_{name}'] = n
        out[f'rec8_{name}'] = h
    groups = defaultdict(list)
    for b, i in sec_of.items():
        groups[i].append(b)
    groups = {i: sorted(v) for i, v in groups.items() if len(v) >= 2}
    if len(groups) >= 2:
        kiai_bars = set(sec_of)
        lo, hi = order[0], order[-1]
        cross, null = [], []
        for i, bi in groups.items():
            for j, bj in groups.items():
                if i == j:
                    continue
                kj = {key[b] for b in bj}
                cross.append(np.mean([key[b] in kj for b in bi]))
                w = bj[-1] - bj[0] + 1
                vals = []
                for s in range(lo, hi - w + 2):
                    win = range(s, s + w)
                    if any(x in kiai_bars for x in win):
                        continue
                    kw = {key[x] for x in win if x in bars}
                    if sum(1 for x in win if x in bars) < 2:
                        continue
                    vals.append(np.mean([key[b] in kw for b in bi]))
                null.append(float(np.mean(vals)) if vals else math.nan)
        ok = [(c, n) for c, n in zip(cross, null) if math.isfinite(n)]
        if ok:
            out['kiai_cross'] = float(np.mean([c for c, _ in ok]))
            out['kiai_null'] = float(np.mean([n for _, n in ok]))
            out['kiai_pairs'] = len(ok)
    return out


# ---------------------------------------------------------------- accumulators

class Acc:
    def __init__(self):
        self.c = defaultdict(Counter)        # name -> Counter(key -> count)
        self.lists = defaultdict(list)       # name -> values
        self.mom = defaultdict(lambda: [0, 0.0, 0.0])   # name|key -> n, sum, sum of squares
        self.examples = defaultdict(list)    # signal -> (score, id, note)

    def add(self, name, key, n=1):
        self.c[name][key] += n

    def val(self, name, v):
        self.lists[name].append(v)

    def moment(self, name, key, v):
        m = self.mom[(name, key)]
        m[0] += 1
        m[1] += v
        m[2] += v * v

    def example(self, signal, score, ident, note=''):
        self.examples[signal].append((score, ident, note))


def _cells(dims, mask):
    arrs = [np.asarray(a, np.int64).reshape(-1) for a, _ in dims]
    sizes = [len(lab) for _, lab in dims]
    if mask is not None:
        arrs = [a[mask] for a in arrs]
    if not len(arrs[0]):
        return None, sizes
    return np.ravel_multi_index(arrs, sizes), sizes


def _key(dims, sizes, f):
    idx = np.unravel_index(int(f), sizes)
    key = tuple(dims[d][1][idx[d]] for d in range(len(dims)))
    return key if len(key) > 1 else key[0]


def hist(A, name, dims, mask=None):
    """Count cells of integer arrays ``dims`` = [(array, labels), ...] into ``A.c[name]``."""
    flat, sizes = _cells(dims, mask)
    if flat is None:
        return
    cnt = np.bincount(flat, minlength=int(np.prod(sizes)))
    for f in np.flatnonzero(cnt):
        A.add(name, _key(dims, sizes, f), int(cnt[f]))


def moments(A, name, dims, values, mask=None):
    flat, sizes = _cells(dims, mask)
    if flat is None:
        return
    v = np.asarray(values, float).reshape(-1)
    v = v[mask] if mask is not None else v
    size = int(np.prod(sizes))
    n = np.bincount(flat, minlength=size)
    s1 = np.bincount(flat, weights=v, minlength=size)
    s2 = np.bincount(flat, weights=v * v, minlength=size)
    for f in np.flatnonzero(n):
        m = A.mom[(name, _key(dims, sizes, f))]
        m[0] += int(n[f])
        m[1] += float(s1[f])
        m[2] += float(s2[f])


def const(n, i):
    return np.full(n, i, np.int64)


def quant(v) -> dict:
    a = np.asarray([x for x in v if x is not None and math.isfinite(x)], float)
    if not len(a):
        return dict(n=0)
    q = np.quantile(a, [0.1, 0.25, 0.5, 0.75, 0.9])
    return dict(n=int(len(a)), mean=round(float(a.mean()), 4), p10=round(float(q[0]), 4), p25=round(float(q[1]), 4),
                p50=round(float(q[2]), 4), p75=round(float(q[3]), 4), p90=round(float(q[4]), 4))


def bins(v, edges) -> dict:
    a = np.asarray(v, float)
    out = {}
    lo = -math.inf
    for e in list(edges) + [math.inf]:
        out[f'<={e}' if math.isfinite(e) else f'>{edges[-1]}'] = int(np.sum((a > lo) & (a <= e)))
        lo = e
    return out


# ---------------------------------------------------------------- exclusion

class Union:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def build_exclusion(args, root: Path, corpus: list[dict], report: dict) -> set[str]:
    """Set folders (paths relative to the root) linked to any sealed set. No chart is opened here."""
    def shas(path):
        with open(root / path, newline='') as f:
            rows = list(csv.DictReader(f))
        out = {r['audio_sha256'].strip() for r in rows if (r.get('audio_sha256') or '').strip()}
        if not out:
            raise SystemExit(f'{path}: no audio_sha256 values')
        return out

    h1, h2 = shas(args.holdout1), shas(args.holdout2)
    mpath = root / args.hard_holdout / 'membership.json'
    data = mpath.read_bytes()
    if hashlib.sha256(data).hexdigest() != args.hard_holdout_sha256:
        raise SystemExit(f'{mpath}: sha256 does not match the seal; refusing to run')
    members = json.loads(data.decode('utf-8'))['members']
    u = Union()
    seeds: list[str] = []
    for r in corpus:
        g, d = 'g:' + str(r['group_id']), 'd:' + r['set_dir']
        u.union(g, d)
        if r['audio_sha256']:
            u.union(g, 'a:' + r['audio_sha256'])
            if r['audio_sha256'] in h1 or r['audio_sha256'] in h2:
                seeds.append(g)
        if r['eval_split'] == 'heldout':
            seeds.append(g)
    n_dedup_holdout = 0
    for grp in json.loads((root / args.groups).read_text()):
        ids = []
        for m in grp['members']:
            if m.get('sha256'):
                ids.append('a:' + m['sha256'])
            ids += ['g:' + x for x in m.get('corpus_group_ids', []) or []]
            if m.get('path'):
                ids.append('d:' + os.path.dirname(m['path']))
            if m.get('holdout'):
                n_dedup_holdout += 1
                seeds.append(ids[-1])
        for a, b in zip(ids, ids[1:]):
            u.union(a, b)
    n_inv = 0
    with open(root / args.inventory) as f:
        for line in f:
            if not line.strip():
                continue
            x = json.loads(line)
            if x.get('path') and x.get('sha256'):
                u.union('d:' + os.path.dirname(x['path']), 'a:' + x['sha256'])
                n_inv += 1
                if x.get('holdout'):
                    seeds.append('a:' + x['sha256'])
    hard_sets = 0
    for m in members:
        for s in m.get('sets', []):
            hard_sets += 1
            if s.get('set_id'):
                seeds.append('d:dataset/std-taiko/' + str(s['set_id']))
            if s.get('audio_sha256'):
                seeds.append('a:' + s['audio_sha256'])
            if s.get('key'):
                parts = str(s['key']).split('@')[0].split('/')
                if len(parts) >= 2:
                    seeds.append('d:dataset/' + '/'.join(parts[:2]))
            for v in s.values():
                for p in (v if isinstance(v, list) else [v]):
                    if isinstance(p, str) and p.startswith('dataset/'):
                        seeds.append('d:' + '/'.join(p.split('/')[:3]))
    sealed_roots = {u.find(x) for x in seeds}
    sealed_dirs = {k[2:] for k in list(u.p) if k.startswith('d:') and u.find(k) in sealed_roots}
    report['exclusion'] = dict(holdout1_shas=len(h1), holdout2_shas=len(h2), dedup_holdout_members=n_dedup_holdout,
                               inventory_rows=n_inv, hard_holdout_recordings=len(members), hard_holdout_sets=hard_sets,
                               corpus_heldout_rows=sum(r['eval_split'] == 'heldout' for r in corpus),
                               sealed_folders=len(sealed_dirs),
                               sha256=dict(membership=args.hard_holdout_sha256))
    if not (h1 and h2 and members and report['exclusion']['corpus_heldout_rows']):
        raise SystemExit('an exclusion list is empty; refusing to run')
    return sealed_dirs


# ---------------------------------------------------------------- one set folder

def list_files(d: Path) -> list[str]:
    out = []
    for base, _dirs, files in os.walk(d):
        rel = os.path.relpath(base, d)
        out += [f if rel == '.' else f'{rel}/{f}' for f in files]
    return out


def load_meta(d: Path) -> dict:
    mp = d / 'metadata.json'
    if not mp.is_file():
        return {}
    try:
        meta = json.loads(mp.read_text())
    except ValueError:
        return {}
    out = {}
    for b in (meta.get('beatmaps') or []) if isinstance(meta, dict) else []:
        if isinstance(b, dict) and b.get('checksum'):
            out[b['checksum']] = b
    return out


def process_set(source: str, set_dir: str, root: Path, corpus_rows: list[dict] | None, A: Acc, rows_out, state):
    d = root / set_dir
    files = list_files(d)
    osu = sorted(f for f in files if f.lower().endswith('.osu') and '/' not in f)
    meta = load_meta(d) if source == 'st' else {}
    nfc = lambda x: unicodedata.normalize('NFC', x)
    by_path = {nfc(Path(r['path']).name): r for r in corpus_rows or []}
    matched = 0
    charts = []
    other_modes = []
    for name in osu:
        raw = (d / name).read_bytes()
        try:
            p = parse_osu(raw)
        except Exception as e:                                    # a broken file is counted, not fatal
            A.add('errors', f'parse:{type(e).__name__}')
            continue
        star = math.nan
        if source == '4k':
            row = by_path.get(nfc(name))
            if row is None:
                label = f"x-mode{p['mode']}" + (f"-{int(p['keys'])}k" if p['mode'] == 3 else '')
                A.add('k4_folder_other_charts', label)
                if label != 'x-mode3-4k':                       # 4K charts outside the ranked/loved corpus
                    other_modes.append((p, label))
                continue
            matched += 1
            mode = 'mania4k'
            star = float(row['star']) if row.get('star') is not None else math.nan
        else:
            if p['mode'] not in (0, 1):
                A.add('st_other_modes', str(p['mode']))
                other_modes.append((p, f"x-mode{p['mode']}"))
                continue
            mode = 'std' if p['mode'] == 0 else 'taiko'
            b = meta.get(hashlib.md5(raw).hexdigest())
            if b is not None:
                star = float(b.get('difficulty_rating') or math.nan)
                if b.get('user_id') is not None:
                    A.add('guest_users', set_dir + '|' + str(b.get('user_id')))
        try:
            res = analyse(p, mode)
        except Exception as e:                                    # counted, and the folder goes on
            A.add('errors', f'analyse:{type(e).__name__}')
            state['failed'].append(f'{set_dir}/{name}: {type(e).__name__}: {e}'[:300])
            continue
        if res is None:
            A.add('skipped', mode)
            continue
        res['rec'].update(source=source, set_dir=set_dir, set_id=Path(set_dir).name, star=star,
                          audio=p['audio'], split=by_path[nfc(name)]['_split'] if source == '4k' else '')
        res['p'] = p
        charts.append(res)
    if source == '4k':
        A.add('k4_corpus_rows', 'listed', len(by_path))
        A.add('k4_corpus_rows', 'matched', matched)
    # d. folder samples (names only)
    audio_names = {c['p']['audio'].casefold() for c in charts} | {p['audio'].casefold() for p, _ in other_modes}
    samples = [f for f in files if f.lower().endswith(AUDIO_EXT) and f.casefold() not in audio_names]
    hs_named = [f for f in samples if HITSOUND_NAME.match(os.path.basename(f).lower())]
    folder = dict(n_samples=len(samples), n_hitsound_named=len(hs_named), n_other=len(samples) - len(hs_named),
                  osb=sum(1 for f in files if f.lower().endswith('.osb')))
    names_cf = {f.casefold() for f in files}
    modes_here = Counter(c['rec']['mode'] for c in charts)
    A.add('folders', source)
    A.val(f'folder_samples|{source}', folder['n_samples'])
    A.val(f'folder_other|{source}', folder['n_other'])
    if folder['n_other'] >= 20:
        A.example(f'folder_samples|{source}', folder['n_other'], set_dir, f"{folder['n_other']} non-hitsound samples")
    for c in charts:
        r = c['rec']
        r.update(folder_samples=folder['n_samples'], folder_other=folder['n_other'], folder_hs_named=folder['n_hitsound_named'])
        r['ks_exist'] = sum(1 for f in c['side']['fnames'] if f.casefold() in names_cf)
    # groups on one audio file and mode
    groups: dict[tuple, list] = defaultdict(list)
    for c in charts:
        groups[(c['p']['audio'].casefold(), c['rec']['mode'])].append(c)
    for (audio, mode), cs in groups.items():
        if all(math.isfinite(c['rec']['star']) for c in cs):
            cs.sort(key=lambda c: (c['rec']['star'], c['rec']['n_on']))
        else:
            A.add('ladder_order_by_onsets', mode)
            cs.sort(key=lambda c: c['rec']['n_on'])
        for i, c in enumerate(cs):
            c['rec']['ladder_n'] = len(cs)
            c['rec']['ladder_rank'] = i
            c['rec']['timing_as_top'] = int(same_timing(c['side']['reds'], cs[-1]['side']['reds']))
            if i < len(cs) - 1:
                A.add(f'timing_as_top|{mode}', bool(c['rec']['timing_as_top']))
        ladder(cs, mode, set_dir, A)
        top = cs[-1]
        rep = repetition(top['side'])
        top['rec'].update({f'rep_{k}': v for k, v in rep.items()})
        top['rec']['is_top'] = 1
        song(top, rep, mode, set_dir, A)
        kiai_sets = {tuple((round(a), round(b)) for a, b in c['side']['kiai_all']) for c in cs}
        A.add('kiai_same_across_diffs', (mode, len(cs) >= 2, len(kiai_sets) == 1))
        sha = state['sha_of'].get(f"{set_dir}/{audio}")
        state['reps'].append(dict(set_dir=set_dir, mode=mode, sha=sha, grid=top['side']['grid'], span=top['side']['span'],
                                  reds=top['side']['reds']))
    # g. several modes on one audio file within the folder
    by_audio = defaultdict(list)
    for c in charts:
        by_audio[c['p']['audio'].casefold()].append((c['rec']['mode'], c['side']['reds'], c['side']['grid'], c['side']['span']))
    for p, label in other_modes:
        if p['reds'] and p['objs']:
            end = max(o['t'] for o in p['objs'])
            by_audio[p['audio'].casefold()].append((label, p['reds'], Grid(p['reds'], end), (p['objs'][0]['t'], end)))
    for audio, items in by_audio.items():
        ms = sorted({m for m, *_ in items})
        if len(ms) < 2:
            continue
        A.add('multimode', (source, '+'.join(ms)))
        firsts = {}
        for m, reds, grid, span in items:
            firsts.setdefault(m, (reds, grid, span))
        keys = sorted(firsts)
        a = firsts[keys[0]]
        for k in keys[1:]:
            b = firsts[k]
            same = same_timing(a[0], b[0])
            A.add('multimode_same_timing', (source, same))
            if not same:
                ag = grid_agreement(a[1], a[2], b[1], b[2])
                if ag:
                    A.val(f'multimode_agree|{source}', ag['median'])
    for c in charts:
        rows_out.writerow({k: c['rec'].get(k, '') for k in rows_out.fieldnames})
        chart_acc(c, A)
    state['charts'] += len(charts)


def chart_acc(c, A: Acc):
    r, s = c['rec'], c['side']
    m = r['mode']
    A.add('charts', m)
    for d in DIVS:
        A.add(f'snap|{m}', f's{d}', r[f's{d}'])
    A.add(f'snap|{m}', 's_off', r['s_off'])
    for d in DIAG_DIVS:
        A.add(f'snap_diag|{m}', f'off{d}', r[f'off{d}'])
    A.add(f'onsets|{m}', 'n', r['n_on'])
    A.add(f'red|{m}', min(r['n_red'], 11))
    A.val(f'n_red|{m}', r['n_red'])
    for mt in r['meters'].split(';'):
        A.add(f'meter_charts|{m}', mt)
    A.add(f'meter_change|{m}', r['meter_changes'] > 0)
    if r['meter_changes'] > 0:
        A.example(f'meter_change|{m}', r['meter_changes'], r['set_dir'], f"{r['version']}: meters {r['meters']}")
    for red in s['reds']:
        A.add(f'meter_red|{m}', red[2])
    # hitsounds by position and tempo band
    ev, bits = s['ev'], s['ev_bits']
    dims = [(ev['pos'], POS), (ev['band'], BANDS)]
    hist(A, f'hs_all|{m}', dims)
    hist(A, f'hs_whistle|{m}', dims, (bits & WHISTLE) != 0)
    hist(A, f'hs_finish|{m}', dims, (bits & FINISH) != 0)
    hist(A, f'hs_clap|{m}', dims, (bits & CLAP) != 0)
    loc = s['loc']
    dims = [(loc['pos'], POS), (loc['band'], BANDS)]
    if m == 'taiko':
        hist(A, 'taiko_pos', dims)
        hist(A, 'taiko_kat', dims, s['kat'])
        hist(A, 'taiko_big', dims, s['big'])
    if m == 'std':
        hist(A, 'std_pos', dims[:1])
        hist(A, 'std_nc', dims[:1], s['nc'])
    if m == 'mania4k':
        hist(A, 'mania_chord', [(loc['pos'], POS), (np.minimum(s['chord'], 4), range(5))])
    share = r['ev_hs'] / r['n_ev'] if r['n_ev'] else 0.0
    A.val(f'hs_share|{m}', share)
    if r['n_clap'] >= 20 and r['clap_beat4'] >= 20:
        bbs = r['clap_bb'] / r['clap_beat4']
        A.val(f'clap_bb_share|{m}', bbs)
        if r['ev_bb']:
            A.val(f'bb_clap_cover|{m}', r['ev_bb_clap'] / r['ev_bb'])
    if r['n_finish'] >= 10:
        A.val(f'finish_db_share|{m}', r['fin_db'] / r['n_finish'])
    ks = r['ks_obj'] / r['n_obj'] if r['n_obj'] else 0.0
    A.val(f'ks_share|{m}', ks)
    if ks >= 0.5:
        A.example(f'keysound|{m}', ks, r['set_dir'], f"{r['version']}: {r['ks_obj']}/{r['n_obj']} objects, {r['ks_files']} files")
    A.val(f'sb_samples|{m}', r['sb_samples'])
    A.add(f'custom_idx|{m}', (r['tp_sidx_max'] >= 1, r['tp_sidx_max'] >= 2))
    A.add(f'sample_sets|{m}', len([x for x in r['sample_sets'].split(';') if x and x != '0']))
    # sections
    A.val(f'n_kiai|{m}', r['n_kiai'])
    A.val(f'n_kiai_flash|{m}', r['n_kiai_flash'])
    for k in (r['kiai_beats'].split(';') if r['kiai_beats'] else []):
        A.val(f'kiai_beats|{m}', float(k))
    A.add(f'kiai_start_db|{m}', 'db', r['kiai_db'])
    A.add(f'kiai_start_db|{m}', 'all', r['n_kiai'])
    if r['n_kiai'] >= 2:
        A.add(f'kiai_shared|{m}', r['kiai_shared_len'] >= 2)
        A.add(f'kiai_shared_sections|{m}', 'shared', r['kiai_shared_len'])
        A.add(f'kiai_shared_sections|{m}', 'all', r['n_kiai'])
    A.val(f'n_breaks|{m}', r['n_breaks'])
    for b in (r['break_s'].split(';') if r['break_s'] else []):
        A.val(f'break_s|{m}', float(b))
    A.val(f'n_bookmarks|{m}', r['n_bookmarks'])
    A.add(f'bookmarks|{m}', 'all', r['n_bookmarks'])
    A.add(f'bookmarks|{m}', 'db', r['bookmarks_db'])
    A.add(f'bookmarks|{m}', 'on_beat', r['bookmarks_on_beat'])
    if r['bookmark_gap_bars'] != '':
        A.val(f'bookmark_gap_bars|{m}', float(r['bookmark_gap_bars']))
    A.val(f'vol_changes|{m}', r['vol_changes'])


def ladder(cs: list, mode: str, set_dir: str, A: Acc):
    n = len(cs)
    A.add(f'ladder_size|{mode}', min(n, 9))
    if n < 2:
        return
    stars = [c['rec']['star'] for c in cs]
    for i in range(n):
        for j in range(i + 1, n):
            e, h = cs[i]['side']['on_t'], cs[j]['side']['on_t']
            cont = float(np.mean(near(e, h) <= TOL)) if len(e) else math.nan
            same = same_timing(cs[i]['side']['reds'], cs[j]['side']['reds'])
            adj = j == i + 1
            A.val(f'contain|{mode}|{"adj" if adj else "all"}|{"same" if same else "diff"}', cont)
            if adj:
                A.val(f'contain_adj|{mode}', cont)
                A.val(f'density_ratio_adj|{mode}', len(e) / max(len(h), 1))
    adjc = [float(np.mean(near(cs[i]['side']['on_t'], cs[i + 1]['side']['on_t']) <= TOL)) for i in range(n - 1)]
    if n >= 4:
        A.example(f'ladder_nested|{mode}', min(adjc), set_dir, f'{n} difficulties, min adjacent containment {min(adjc):.3f}')
    # metrical concentration by rank and star band
    for i, c in enumerate(cs):
        rank = 0 if i == 0 else 2 if i == n - 1 else 1
        st = c['rec']['star']
        band = int(min(max(st, 0), 8)) if math.isfinite(st) else 9
        pos = c['side']['loc']['pos']
        if n >= 3:
            hist(A, f'rank_pos|{mode}', [(const(len(pos), rank), ('easiest', 'middle', 'hardest')), (pos, POS)])
        hist(A, f'star_pos|{mode}', [(const(len(pos), band), [str(k) for k in range(9)] + ['na']), (pos, POS)])
    if n < 3:
        return
    # retention of the hardest chart's onsets by the easier ones
    top = cs[-1]['side']
    kept = np.zeros(len(top['on_t']))
    kept_easiest = near(top['on_t'], cs[0]['side']['on_t']) <= TOL
    for c in cs[:-1]:
        kept += near(top['on_t'], c['side']['on_t']) <= TOL
    frac = kept / (n - 1)
    pos = top['loc']['pos']
    if mode == 'mania4k':
        acc = np.minimum(top['chord'], 4)
        acc_lab = ('0', 'chord1', 'chord2', 'chord3', 'chord4')
    elif mode == 'std':
        acc = np.where((top['bits'] & CLAP) != 0, 2, np.where((top['bits'] & FINISH) != 0, 3, np.where((top['bits'] & WHISTLE) != 0, 1, 0)))
        acc_lab = ('none', 'whistle', 'clap', 'finish')       # clap wins over finish
    else:
        acc = np.where(top['big'], 2, np.where(top['kat'], 1, 0))
        acc_lab = ('don', 'kat', 'big')
    moments(A, f'retain|{mode}', [(pos, POS)], frac)
    moments(A, f'retain|{mode}', [(const(len(pos), 0), ('*',))], frac)
    moments(A, f'retain_easiest|{mode}', [(pos, POS)], kept_easiest.astype(float))
    moments(A, f'retain_acc|{mode}', [(pos, POS), (acc, acc_lab)], frac)
    A.add(f'retain_sets|{mode}', 'sets')


def song(top, rep: dict, mode: str, set_dir: str, A: Acc):
    r = top['rec']
    A.add(f'songs|{mode}', 'n')
    tb, nb = r['tern_beats'], r['beats']
    A.add(f'songs_tern|{mode}', ('tern>=1', tb >= 1))
    A.add(f'songs_tern|{mode}', ('tern>=4', tb >= 4))
    pure_tern = tb - r['mixed_beats']
    A.add(f'songs_tern|{mode}', ('song_mixed(pure_tern>=4,bin>=4)', pure_tern >= 4 and r['bin_beats'] >= 4))
    A.add(f'songs_tern|{mode}', ('beat_mixed>=4', r['mixed_beats'] >= 4))
    A.add(f'songs_tern|{mode}', ('tern_share_of_beats>=0.01', nb > 0 and tb / nb >= 0.01))
    if tb >= 4 and r['bars']:
        share = r['tern_bars'] / r['bars']
        A.val(f'tern_bar_share|{mode}', share)
        A.val(f'tern_maxrun|{mode}', r['tern_maxrun'])
        A.val(f'tern_runs|{mode}', r['tern_runs'])
        if share <= 0.1 and r['tern_runs'] >= 1:
            A.example(f'tern_local|{mode}', -share, set_dir, f"{r['version']}: {r['tern_bars']}/{r['bars']} bars, {r['tern_runs']} runs, max {r['tern_maxrun']}")
        if share >= 0.9:
            A.example(f'tern_whole|{mode}', share, set_dir, f"{r['version']}: {r['tern_bars']}/{r['bars']} bars")
    if rep:
        for k in ('all', 'nontrivial'):
            if rep[f'bars_{k}']:
                A.val(f'rep_rec|{mode}|{k}', rep[f'rec_{k}'] / rep[f'bars_{k}'])
                A.val(f'rep_rec8|{mode}|{k}', rep[f'rec8_{k}'] / rep[f'bars_{k}'])
        A.val(f'rep_distinct_share|{mode}', rep['distinct'] / max(rep['bars_all'], 1))
        if rep['blocks']:
            A.val(f'rep_block|{mode}', rep['blocks_rec'] / rep['blocks'])
        if rep['blocks_nt']:
            A.val(f'rep_block_nt|{mode}', rep['blocks_nt_rec'] / rep['blocks_nt'])
            if rep['blocks_nt'] >= 16:
                A.example(f'rep_block|{mode}', rep['blocks_nt_rec'] / rep['blocks_nt'], set_dir,
                          f"{r['version']}: {rep['blocks_nt_rec']}/{rep['blocks_nt']} non-trivial 4-bar blocks recur")
        if rep.get('n_kiai', 0) >= 4 and rep.get('n_nonkiai', 0) >= 4:
            A.val(f'rep_rec8_kiai|{mode}', rep['rec8_kiai'] / rep['n_kiai'])
            A.val(f'rep_rec8_nonkiai|{mode}', rep['rec8_nonkiai'] / rep['n_nonkiai'])
        if 'kiai_cross' in rep:
            A.val(f'kiai_cross|{mode}', rep['kiai_cross'])
            A.val(f'kiai_null|{mode}', rep['kiai_null'])
            A.add(f'kiai_cross_gt_null|{mode}', rep['kiai_cross'] > rep['kiai_null'])
    if r['n_kiai'] >= 3 and r['kiai_shared_len'] >= 3:
        A.example(f'kiai_recur|{mode}', r['kiai_shared_len'], set_dir, f"{r['version']}: kiai beats {r['kiai_beats']}")
    if mode in ('std', 'taiko') and r['n_clap'] >= 50 and r['clap_beat4']:
        A.example(f'clap_backbeat|{mode}', r['clap_bb'] / r['clap_beat4'], set_dir,
                  f"{r['version']}: {r['clap_bb']}/{r['clap_beat4']} on-beat claps on 2/4")


def cross_sets(state, A: Acc):
    by_sha = defaultdict(list)
    for x in state['reps']:
        if x['sha']:
            by_sha[x['sha']].append(x)
    for sha, xs in by_sha.items():
        dirs = sorted({x['set_dir'] for x in xs})
        if len(dirs) < 2:
            continue
        A.add('sha_multi_sets', 'recordings')
        A.add('sha_multi_sets', 'sets', len(dirs))
        firsts = {}
        for x in xs:
            firsts.setdefault(x['set_dir'], x)
        ds = dirs[:12]
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                a, b = firsts[ds[i]], firsts[ds[j]]
                kind = '+'.join(sorted({a['mode'], b['mode']}))
                same = same_timing(a['reds'], b['reds'])
                A.add('sha_pairs', (kind, same))
                ag = grid_agreement(a['grid'], a['span'], b['grid'], b['span'])
                if ag:
                    A.val(f'sha_agree|{kind}', ag['median'])
                    A.val(f'sha_within5|{kind}', ag['within5'])
                    if ag['median'] > 5:
                        A.example('sha_disagree', ag['median'], f"{ds[i]} vs {ds[j]}", f"{kind}, median {ag['median']:.1f} ms")


# ---------------------------------------------------------------- summary

def summarise(A: Acc) -> dict:
    out: dict = {}
    for name, c in sorted(A.c.items()):
        if name == 'guest_users':
            per_set = Counter(k.split('|')[0] for k in c)
            out['guest_users_per_set'] = dict(sorted(Counter(min(v, 5) for v in per_set.values()).items()))
            continue
        out[name] = {('|'.join(map(str, k)) if isinstance(k, tuple) else str(k)): v for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))}
    for name, v in sorted(A.lists.items()):
        out['q:' + name] = quant(v)
    for name in [k for k in A.lists if k.startswith(('n_kiai|', 'n_breaks|', 'n_bookmarks|', 'vol_changes|', 'sb_samples|', 'folder_'))]:
        out['bins:' + name] = bins(A.lists[name], [0, 1, 2, 3, 5, 10, 50])
    for name in [k for k in A.lists if k.startswith(('hs_share|', 'ks_share|', 'contain', 'tern_bar_share|', 'rep_', 'clap_bb', 'bb_clap', 'finish_db', 'sha_within5', 'kiai_cross', 'kiai_null'))]:
        out['bins:' + name] = bins(A.lists[name], [0, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    for name in [k for k in A.lists if k.startswith(('sha_agree|', 'multimode_agree|'))]:
        out['bins:' + name] = bins(A.lists[name], [1, 2, 5, 10, 20, 50])
    for name in [k for k in A.lists if k.startswith(('kiai_beats|',))]:
        out['top:' + name] = dict(Counter(round(x) for x in A.lists[name]).most_common(12))
    mom = defaultdict(dict)
    for (name, key), (n, s, s2) in A.mom.items():
        k = '|'.join(map(str, key)) if isinstance(key, tuple) else str(key)
        mom[name][k] = dict(n=n, mean=round(s / n, 4), sd=round(math.sqrt(max(s2 / n - (s / n) ** 2, 0.0)), 4))
    out['moments'] = {k: dict(sorted(v.items())) for k, v in sorted(mom.items())}
    # eta squared of retention on metrical position
    for name in [k for k in out['moments'] if k.startswith('retain|')]:
        cells = {k: A.mom[(name, k)] for k in POS if (name, k) in A.mom}
        n, s, s2 = A.mom[(name, '*')]
        mean = s / n
        sst = s2 - n * mean * mean
        ssb = sum(c[0] * (c[1] / c[0] - mean) ** 2 for c in cells.values())
        out['eta2:' + name] = round(ssb / sst, 4) if sst > 0 else None
    return out


def pick_examples(A: Acc, k: int = 5) -> dict:
    out = {}
    for sig, xs in sorted(A.examples.items()):
        xs = sorted(xs, key=lambda t: (-t[0], t[1]))
        out[sig] = [dict(id=i, score=round(float(s), 4), note=n) for s, i, n in xs[:k]]
    return out


# ---------------------------------------------------------------- main

FIELDS = ['source', 'split', 'set_id', 'set_dir', 'mode', 'version', 'star', 'audio', 'ladder_n', 'ladder_rank', 'is_top',
          'timing_as_top', 'n_obj', 'n_on', 'dur_s', 'n_red', 'n_green', 'bpm_dom', 'meters', 'meter_changes', 'non4_onsets',
          *[f's{d}' for d in DIVS], 's_off', *[f'off{d}' for d in DIAG_DIVS], 'on_implausible_bpm', 'beats', 'tern_beats',
          'bin_beats', 'mixed_beats', 'bars', 'tern_bars', 'bin_bars', 'both_bars', 'tern_runs', 'tern_maxrun', 'n_kiai',
          'n_kiai_flash', 'kiai_beats', 'kiai_db', 'kiai_shared_len', 'n_breaks', 'break_s', 'n_bookmarks', 'bookmarks_db',
          'bookmarks_on_beat', 'bookmark_gap_bars', 'vol_changes', 'n_ev', 'n_ev_edge', 'ev_hs', 'n_whistle', 'n_finish',
          'n_clap', 'clap_beat4', 'clap_bb', 'ev_bb', 'ev_bb_clap', 'fin_db', 'ev_db', 'n_nc', 'nc_db', 'n_kat', 'n_big',
          'n_don', 'chord2', 'holds', 'ks_obj', 'ks_files', 'ks_exist', 'obj_idx', 'tp_sidx_max', 'tp_sidx_custom',
          'sample_sets', 'sb_samples', 'folder_samples', 'folder_other', 'folder_hs_named',
          'rep_bars_all', 'rep_rec_all', 'rep_rec8_all', 'rep_bars_nontrivial', 'rep_rec_nontrivial', 'rep_rec8_nontrivial',
          'rep_distinct', 'rep_blocks', 'rep_blocks_rec', 'rep_blocks_nt', 'rep_blocks_nt_rec', 'rep_n_kiai', 'rep_rec8_kiai',
          'rep_n_nonkiai', 'rep_rec8_nonkiai', 'rep_kiai_cross', 'rep_kiai_null', 'rep_kiai_pairs']


def split_4k(r) -> str:
    if r['eval_split'] == 'calibration':
        return 'cal'
    dg = hashlib.sha256(('r2-fit-dev-v1:' + str(r['group_id'])).encode()).digest()
    return 'fit_dev' if int.from_bytes(dg[:8], 'big') % 10 == 0 else 'fit_train'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--root', default='.')
    ap.add_argument('--corpus', default='data/r2-corpus.parquet')
    ap.add_argument('--holdout1', default='artifacts/timing-20261009/opt/holdout/holdout.csv')
    ap.add_argument('--holdout2', default='artifacts/timing-20261009/opt/ft/pop/holdout2.csv')
    ap.add_argument('--groups', default='artifacts/bulk/music-dedup-20261015/groups.json')
    ap.add_argument('--inventory', default='artifacts/bulk/music-dedup-20261015/audio-inventory.jsonl')
    ap.add_argument('--hard-holdout', default='artifacts/n1-eval/hard-holdout-v1')
    ap.add_argument('--hard-holdout-sha256', default='00771c6491ced746bc1dd536c0cd3ef453a01798a658ebe3330ab89edec327bb')
    ap.add_argument('--out')
    ap.add_argument('--commit', default='')
    ap.add_argument('--limit-sets', type=int, default=0, help='trial: this many folders per source, by hashed order')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.out:
        ap.error('--out is required')
    t0 = time.time()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as pq
    cols = ['path', 'set_dir', 'beatmap_set_id', 'audio_filename', 'audio_sha256', 'api_status', 'group_id', 'eval_split', 'star']
    corpus = pq.read_table(str(root / args.corpus), columns=cols).to_pylist()
    report: dict = dict(commit=args.commit, args=vars(args), corpus_rows=len(corpus),
                        corpus_status=dict(Counter(str(r['api_status']) for r in corpus)))
    sealed = build_exclusion(args, root, corpus, report)
    print(f'[{time.time() - t0:.0f}s] exclusion: {len(sealed)} sealed folders', flush=True)
    by_dir: dict[str, list] = defaultdict(list)
    for r in corpus:
        if r['api_status'] in STATUSES_4K:
            r['_split'] = split_4k(r)
            by_dir[r['set_dir']].append(r)
    dirs_4k = sorted(d for d in by_dir if d not in sealed)
    st_base = root / 'dataset' / 'std-taiko'
    dirs_st = sorted(f'dataset/std-taiko/{p.name}' for p in st_base.iterdir() if p.is_dir())
    n_st_all = len(dirs_st)
    dirs_st = [d for d in dirs_st if d not in sealed]
    hard_ids = {str(s.get('set_id')) for m in json.loads((root / args.hard_holdout / 'membership.json').read_text())['members']
                for s in m.get('sets', [])}
    assert not any(Path(d).name in hard_ids for d in dirs_st), 'a hard-holdout set survived the exclusion'
    report['folders'] = dict(k4_ranked_loved=len(by_dir), k4_sealed=len(by_dir) - len(dirs_4k), k4_kept=len(dirs_4k),
                             st_all=n_st_all, st_sealed=n_st_all - len(dirs_st), st_kept=len(dirs_st),
                             k4_kept_by_split=dict(Counter(min(r['_split'] for r in by_dir[d]) for d in dirs_4k)))
    if args.limit_sets:
        hk = lambda d: hashlib.sha256(d.encode()).hexdigest()
        dirs_4k = sorted(dirs_4k, key=hk)[:args.limit_sets]
        dirs_st = sorted(dirs_st, key=hk)[:args.limit_sets]
    sha_of = {}
    for r in corpus:
        if r['audio_sha256'] and r['audio_filename']:
            sha_of[f"{r['set_dir']}/{r['audio_filename'].casefold()}"] = r['audio_sha256']
    with open(root / args.inventory) as f:
        for line in f:
            if '"std-taiko"' in line:
                x = json.loads(line)
                if x.get('sha256'):
                    dd, name = os.path.split(x['path'])
                    sha_of[f'{dd}/{name.casefold()}'] = x['sha256']
    A = Acc()
    state = dict(reps=[], sha_of=sha_of, charts=0, failed=[])
    with open(out / 'charts.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for source, dirs in (('4k', dirs_4k), ('st', dirs_st)):
            for i, d in enumerate(dirs):
                assert d not in sealed
                try:
                    process_set(source, d, root, by_dir[d] if source == '4k' else None, A, w, state)
                except Exception as e:                            # counted; the run goes on
                    A.add('errors', f'folder:{type(e).__name__}')
                    state['failed'].append(f'{d}: {type(e).__name__}: {e}'[:300])
                if i % 500 == 0:
                    print(f'[{time.time() - t0:.0f}s] {source} {i}/{len(dirs)} charts {state["charts"]}', flush=True)
    cross_sets(state, A)
    summary = summarise(A)
    report.update(elapsed_s=round(time.time() - t0, 1), charts=state['charts'], folders_done=dict(k4=len(dirs_4k), st=len(dirs_st)),
                  failed=state['failed'][:50], n_failed=len(state['failed']))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    (out / 'examples.json').write_text(json.dumps(pick_examples(A), indent=1, sort_keys=True))
    (out / 'run.json').write_text(json.dumps(report, indent=1, sort_keys=True, default=str))
    print(f'[{time.time() - t0:.0f}s] done: {state["charts"]} charts', flush=True)
    return 0


# ---------------------------------------------------------------- self-test (synthetic, absolute expectations)

def selftest() -> int:
    T0, L0 = 1000.5, 60000.0 / 123.45
    T1, L1 = T0 + 10.5 * L0, 60000.0 / 180.0                    # fractional boundary, meter 7
    notes = [(T0 + k * L0, 'db b2 b3 b4'.split()[k]) for k in range(4)]
    notes += [(T0 + 4 * L0 + L0 / 3, 't3'), (T0 + 5.5 * L0, 'h2'), (T0 + 6.25 * L0, 'q4'), (T0 + 6 * L0 + L0 / 5, 'off'),
              (T0 - L0, 'b4'), (T1, 'db'), (T1 + 3 * L1, 'bx'), (T1 + 7 * L1, 'db'), (T1 + 7 * L1 + L1 / 6, 't6'),
              (T1 + 8 * L1 + 5 * L1 / 12, 't12'), (T1 + 9 * L1 + 3 * L1 / 16, 's16')]
    hit = []
    for t, _ in notes:
        ms = math.floor(t) if t == T1 else int(round(t))            # the red line's note 0.77 ms before it
        hit.append(f'256,192,{ms},1,8,0:0:0:0:')
    t_sl = T0 + 8 * L0
    hit.append(f'100,100,{int(round(t_sl))},2,0,B|200:100,2,280,4|8|2,0:0|0:0|0:0,0:0:0:0:')
    text = '\n'.join([
        'osu file format v14', '[General]', 'AudioFilename: a.mp3', 'Mode: 0', '[Editor]',
        f'Bookmarks: {int(T0)},{int(T0 + 8 * L0)}', '[Difficulty]', 'SliderMultiplier:1.4',
        '[Events]', f'2,{int(T0 + 20000)},{int(T0 + 25000)}', '[TimingPoints]',
        f'{T0},{L0},4,1,0,100,1,0', f'{T0 + 2 * L0},-100,4,1,0,100,0,1', f'{t_sl - 1},-50,4,1,0,80,0,1',
        f'{T1},{L1},7,1,0,80,1,1', f'{T1 + 7 * L1},-100,7,1,0,80,0,0', '[HitObjects]', *hit, ''])
    p = parse_osu(text.encode())
    res = analyse(p, 'std')
    assert res is not None
    loc = res['side']['loc']
    got = [POS[i] for i in loc['pos'].tolist()]
    order = sorted(range(len(notes)), key=lambda i: notes[i][0])
    want = [notes[i][1] for i in order]
    on = res['side']['on_t'].tolist()
    i_sl = on.index(float(int(round(t_sl))))
    got_wo = [g_ for k, g_ in enumerate(got) if k != i_sl]
    assert got_wo == want, (got_wo, want)
    assert got[i_sl] == 'db', got[i_sl]                          # beat 8 of 4/4: a downbeat
    seg = loc['seg'].tolist()
    assert T1 - math.floor(T1) > 0.5
    assert seg[on.index(float(math.floor(T1)))] == 1, 'a note under 2 ms before a red line belongs to it'
    r = res['rec']
    assert r['n_kiai'] == 1 and abs(float(r['kiai_beats']) - 15.5) < 0.01, r['kiai_beats']
    assert r['n_bookmarks'] == 2 and r['bookmarks_db'] == 2, (r['n_bookmarks'], r['bookmarks_db'])
    assert r['n_breaks'] == 1 and r['meters'] == '4;7' and r['meter_changes'] == 1
    ev, bits, edge = res['side']['ev'], res['side']['ev_bits'], res['side']['ev_edge']
    et = res['side']['ev']['B'][edge == 1]
    # SV 2 at 1.4: 280 px per beat, so each slide lasts one beat: edges at beats 9 and 10
    assert np.allclose(et, [9.0, 10.0], atol=0.01), et
    assert bits[edge == 1].tolist() == [8, 2]
    assert POS[ev['pos'][edge == 1][1]] == 'b3'
    # 4/4 on-beat claps: beats 0-3, beat -1 and the edge at beat 9; on 2 and 4: beats 1, 3, -1 and 9
    assert r['clap_bb'] == 4 and r['clap_beat4'] == 6, (r['clap_bb'], r['clap_beat4'])
    assert r['tern_beats'] == 3 and r['mixed_beats'] == 1, (r['tern_beats'], r['mixed_beats'])
    # grid agreement: a copy shifted by 7 ms reads 7 ms; the same grid at double tempo reads 0
    g1 = Grid([(T0, L0, 4)], T0 + 60000)
    g2 = Grid([(T0 + 7, L0, 4)], T0 + 60000)
    g3 = Grid([(T0, L0 / 2, 4)], T0 + 60000)
    sp = (T0, T0 + 60000)
    assert abs(grid_agreement(g1, sp, g2, sp)['median'] - 7) < 1e-6
    assert grid_agreement(g1, sp, g3, sp)['median'] < 1e-6
    # repetition: 16 bars of one pattern with kiai on bars 4-7 and 12-15 recur fully
    side = dict(loc=None, on_t=None, kiai=[])
    ts = []
    for b in range(16):
        for q in (0, 1.5, 2, 3.5):
            ts.append(T0 + (4 * b + q) * L0)
    g = Grid([(T0, L0, 4)], ts[-1])
    side.update(loc=g.locate(ts), on_t=np.array(ts), kiai=[(T0 + 16 * L0, T0 + 32 * L0), (T0 + 48 * L0, T0 + 64 * L0)])
    rep = repetition(side)
    assert rep['rec_nontrivial'] == 16 and rep['rec8_all'] == 16 and rep['blocks'] == 13, rep
    assert rep['kiai_cross'] == 1.0 and rep['kiai_null'] == 1.0, rep
    print('selftest ok')
    return 0


if __name__ == '__main__':
    sys.exit(main())
