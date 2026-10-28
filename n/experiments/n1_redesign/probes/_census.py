"""Chart parsing helpers copied from supervision_census.py at b14d9f5.

Only parsing and chart-local label construction are included; no corpus discovery.
"""
from __future__ import annotations
import math
import os
import re
from collections import Counter
import numpy as np
TOL = 2.0
DIVS = (1, 2, 3, 4, 6, 8, 12, 16)
DIAG_DIVS = (5, 7, 24, 32)
POS = ("db", "b2", "b3", "b4", "bx", "h2", "q4", "e8", "s16", "t3", "t6", "t12", "off")
CLS_POS = {2: "h2", 4: "q4", 8: "e8", 16: "s16", 3: "t3", 6: "t6", 12: "t12"}
WHISTLE, FINISH, CLAP = 2, 4, 8
Q = 48
HEADER_SECTIONS = ("General", "Metadata", "Difficulty", "Editor")
HITSOUND_NAME = re.compile(r'^(normal|soft|drum)-(hit(normal|whistle|finish|clap)|slider(slide|whistle|tick))\d*\.(wav|ogg|mp3)$')


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
    sb_files: set[str] = set()
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
                if len(p) >= 4:
                    sb_files.add(p[3].strip().strip('"').replace('\\', '/').casefold())
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
                tps=tps, reds=reds, objs=objs, breaks=breaks, sb_samples=sb_samples, sb_files=sb_files,
                bookmarks=[b for b in bookmarks if math.isfinite(b)])


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
    other = [f for f in fnames if not HITSOUND_NAME.match(os.path.basename(f.replace('\\', '/')).casefold())]
    r['ks_obj'] = len(fnames)
    r['ks_files'] = len(set(f.casefold() for f in fnames))
    r['ks_obj_other'] = len(other)
    r['ks_files_other'] = len(set(f.casefold() for f in other))
    r['sb_files'] = len(p['sb_files'])
    r['sb_files_other'] = sum(1 for f in p['sb_files'] if not HITSOUND_NAME.match(os.path.basename(f)))
    r['obj_idx'] = sum(1 for o in objs if o['idx'] > 0)
    r['tp_sidx_max'] = max((d['sidx'] for d in p['tps']), default=0)
    r['tp_sidx_custom'] = sum(1 for d in p['tps'] if d['sidx'] >= 1)
    r['sample_sets'] = ';'.join(str(x) for x in sorted({d['sset'] for d in p['tps']} |
                                                         {o['nset'] for o in objs if o['nset']} | {o['aset'] for o in objs if o['aset']}))
    r['sb_samples'] = p['sb_samples']
    holds = sum(1 for o in objs if o['typ'] & 128)
    r['holds'] = holds
    side = dict(grid=g, reds=reds, on_t=on_t, ev_t=np.unique(ev_t), loc=loc, chord=chord, bits=on_bits, kat=kat, big=big, nc=nc,
                ev=ev, ev_bits=ev_bits, ev_edge=ev_edge, kiai=[(k['a'], k['b']) for k in real],
                kiai_all=[(k['a'], k['b']) for k in kiai_info], span=(float(starts[0]), end_ms), fnames=fnames)
    return dict(rec=r, side=side)
