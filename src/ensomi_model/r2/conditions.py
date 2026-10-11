"""Training condition draws (plan v4 section 8.1) and the effective-track invariant.

A track is a tuple of ``Interval(kind, a, b, value)``, [a, b) or [a, T] when b = T; kind 0
is LN share, kind 1 difficulty. Intervals of one kind never overlap. An empty track is
natural mode. Source values never conflict, so a drawn track is an effective track as if
every interval were a request added at 0^- (``request_set.from_track`` maps one to the other).

One draw (``draw_window``), all parameters in ``DrawConfig``:

1. Candidates. LN: a fresh random partition of the song into 8/16/32/64-beat pieces with at
   least one head; with ``p_whole_ln`` the single candidate is [0, T]; otherwise with
   ``p_long`` consecutive pieces are merged into runs of U{2..8} pieces. Difficulty: one cell
   length from ``star_lengths_s`` and one phase from ``star_phases_s`` whose cells partition
   the song, or with ``p_whole_star`` the whole song. Values are the source's own: LNShare_nu
   by prefix sums, the cached Difficulty_nu (or its residual against b(S)).
2. Dropout first, on the candidate lists: all (0.20), a kind (0.25), each candidate (0.20).
   Nothing later removes an interval.
3. Alignment (``selection='per_window'``). With ``p_align``, if a kind has a candidate with a
   decision in V: a kind uniformly, a lead u ~ U{0..32}, per candidate j = max(0, onset - u)
   and importance weight 3 for an LN candidate whose value differs from the share of the 64
   rows ending at j by z >= 2 binomial SE (n >= 20 heads), else 1; one candidate by weight.
   Otherwise the v1 start rule. Then per kind U{1..3} surviving candidates intersecting the
   window, the aligned one included, consecutive with p 0.5.
   ``selection='per_song'`` (v4's stage-1 arm A0, retired) keeps the v1 start rule and picks
   U{1..4} LN and U{1..3} difficulty candidates per song, consecutive with p 0.5.
4. Inverse-probability weight u = p_old(j) / p_new(j), p_new = (1 - p_align) p_old +
   p_align P_align(j), applied to every scored decision with V_k empty; 1 elsewhere.

``selection='natural'`` (phase N of plan v5) draws no candidates: the v1 start rule, an empty
track and weight 1. The lead u is capped at window - 1, so the aligned onset always lies in
the scored window (it matters only for windows shorter than ``lead_max``, as in DPO).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math

import numpy as np

from .common import ContractError
from .features import Chart, Interval
from .locality import RULE_L_VERSION, reads, scope_decisions
from .properties import NU_HASH, ln_share_rows

LN_BEATS = (8, 16, 32, 64)
WINDOW = 256
SELECTIONS = ('per_window', 'per_song', 'natural')


@dataclass(frozen=True)
class DrawConfig:
    ln_beats: tuple = LN_BEATS
    p_whole_ln: float = 0.10
    p_long: float = 0.25
    run_pieces: tuple = (2, 8)
    star_lengths_s: tuple = (30, 60, 120)
    star_phases_s: tuple = (0, 10, 20)
    p_whole_star: float = 0.10
    drop_all: float = 0.20
    drop_kind: float = 0.25
    drop_interval: float = 0.20
    selection: str = 'per_window'     # 'per_window' (aligned draw) | 'per_song' (v1 selection) | 'natural'
    p_align: float = 0.6
    lead_max: int = 32
    informative_rows: int = 64
    min_heads: int = 20
    z_min: float = 2.0
    informative_weight: float = 3.0
    per_window_max: int = 3
    per_song_ln_max: int = 4
    per_song_star_max: int = 3
    p_consecutive: float = 0.5
    ipw: bool = True
    star_value: str = 'absolute'      # 'absolute' tiled star | 'residual' against b(S)

    def __post_init__(self):
        if self.selection not in SELECTIONS:
            raise ContractError(f'selection must be one of {SELECTIONS}')
        if self.selection in ('per_song', 'natural') and (self.p_align or self.ipw):
            raise ContractError(f'{self.selection} selection uses the v1 start rule: p_align must be 0 and ipw off')
        if not 0.0 <= self.p_align <= 1.0:
            raise ContractError('p_align must lie in [0, 1]')

    @classmethod
    def from_dict(cls, data):
        data = dict(data or {})
        for key in ('ln_beats', 'run_pieces', 'star_lengths_s', 'star_phases_s'):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)

    def to_dict(self):
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

    def hash(self):
        """Version hash of the draw: its parameters, nu and rule L (the N-bar and manifest key)."""
        payload = dict(draw=self.to_dict(), nu=NU_HASH, rule_l=RULE_L_VERSION)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass
class Candidate:
    iv: Interval
    cls: str                 # 'piece-16', 'run', 'whole', 'cell-30', ...
    beats: float
    onset: int | None = None  # first decision that reads it (rule L, source occupancy)


@dataclass
class WindowDraw:
    start: int
    stop: int
    track: tuple
    weight: float            # u for decisions with V_k empty
    record: dict = field(default_factory=dict)
    candidates: list = field(default_factory=list)


def _choose(rng, items, count, p_consecutive):
    if rng.random() < p_consecutive:
        s = int(rng.integers(0, len(items) - count + 1))
        return items[s:s + count]
    idx = np.sort(rng.choice(len(items), size=count, replace=False))
    return [items[i] for i in idx]


def _row_counts(chart: Chart):
    key = 'row_counts'
    if key not in chart.cache:
        d = chart.derived()
        chart.cache[key] = (d.attack.sum(1)[:chart.K], d.ln_head.sum(1)[:chart.K])
    return chart.cache[key]


def ln_candidates(chart: Chart, rng, cfg: DrawConfig) -> list[Candidate]:
    g, T = chart.grid, chart.song_ms
    heads, lns = _row_counts(chart)
    if rng.random() < cfg.p_whole_ln:
        v = ln_share_rows(heads, lns, chart.head_ms, 0.0, T, T)
        return [] if v is None else [Candidate(Interval(0, 0.0, T, v), 'whole', float(g.beat(T) - g.beat(0.0)))]
    beat, end_beat = float(g.beat(0.0)), float(g.beat(T))
    pieces = []
    a = 0.0
    while beat < end_beat:
        step = cfg.ln_beats[int(rng.integers(0, len(cfg.ln_beats)))]
        beat += step
        b = min(float(g.time_of_beat(beat)), T)
        if b > a:
            pieces.append((a, b, step))
        a = b
    if rng.random() < cfg.p_long:
        runs, i = [], 0
        lo, hi = cfg.run_pieces
        while i < len(pieces):
            r = int(rng.integers(lo, hi + 1))
            part = pieces[i:i + r]
            runs.append((part[0][0], part[-1][1], sum(p[2] for p in part), 'run' if len(part) > 1 else f'piece-{part[0][2]}'))
            i += r
        spans = runs
    else:
        spans = [(a, b, s, f'piece-{s}') for a, b, s in pieces]
    out = []
    for a, b, beats, cls in spans:
        v = ln_share_rows(heads, lns, chart.head_ms, a, b, T)
        if v is not None:
            out.append(Candidate(Interval(0, a, b, v), cls, float(beats)))
    return out


def star_candidates(chart: Chart, cells, rng, cfg: DrawConfig) -> list[Candidate]:
    """``cells``: {(L_s, phase_s): [(a, b, value)]} plus key 'whole': [(0, T, value)] (values in frame units)."""
    if not cells:
        return []
    whole = cells.get('whole') or []
    if whole and rng.random() < cfg.p_whole_star:
        a, b, v = whole[0]
        return [Candidate(Interval(1, a, b, v), 'whole', float(chart.grid.beat(b) - chart.grid.beat(a)))]
    L = cfg.star_lengths_s[int(rng.integers(0, len(cfg.star_lengths_s)))]
    phase = cfg.star_phases_s[int(rng.integers(0, len(cfg.star_phases_s)))]
    return [Candidate(Interval(1, a, b, v), f'cell-{L}', float(chart.grid.beat(b) - chart.grid.beat(a)))
            for a, b, v in sorted(cells.get((L, phase), []))]


def onset_of(chart: Chart, iv: Interval):
    """The first decision (source occupancy) that reads ``iv`` under rule L, or None."""
    inside, _, _ = scope_decisions(chart, iv)
    held = chart.derived().held
    for k in inside:
        if reads(chart, iv, int(k), held[int(k)]):
            return int(k)
    return None


def p_old(K: int, window: int = WINDOW) -> np.ndarray:
    """The v1 start rule as a distribution over j in [0, K]."""
    p = np.full(K + 1, 0.75 / (K + 1))
    p[0] += 0.125
    p[max(0, K - (window - 1))] += 0.125
    return p


def old_start(rng, K, window=WINDOW):
    u = rng.random()
    if u < 0.125:
        return 0
    if u < 0.25:
        return max(0, K - (window - 1))
    return int(rng.integers(0, K + 1))


def _weights(chart: Chart, cands, js, cfg: DrawConfig):
    heads, lns = _row_counts(chart)
    ch = np.concatenate(([0], np.cumsum(heads)))
    cl = np.concatenate(([0], np.cumsum(lns)))
    w = np.ones(len(cands))
    z = np.full(len(cands), np.nan)
    for i, (c, j) in enumerate(zip(cands, js)):
        if c.iv.kind != 0 or c.cls == 'whole':
            continue
        lo = max(0, j - cfg.informative_rows)
        n = int(ch[j] - ch[lo])
        if n < cfg.min_heads:
            continue
        before = (cl[j] - cl[lo]) / n
        v = c.iv.value
        se = math.sqrt(v * (1.0 - v) / n)
        diff = abs(v - before)
        z[i] = (math.inf if diff > 0 else 0.0) if se == 0 else diff / se
        if z[i] >= cfg.z_min:
            w[i] = cfg.informative_weight
    return w, z


def lead_max(cfg: DrawConfig, window: int) -> int:
    """The largest lead: ``lead_max``, capped so the aligned onset lies in a window of ``window`` decisions."""
    return min(cfg.lead_max, window - 1)


def align_distribution(chart: Chart, survivors: dict, cfg: DrawConfig, window: int = WINDOW):
    """P_align(j) over j in [0, K] and the per-(kind, u) candidate weights, or None if nothing aligns."""
    K = chart.K
    kinds = [k for k, cs in survivors.items() if any(c.onset is not None for c in cs)]
    if not kinds:
        return None, kinds
    p = np.zeros(K + 1)
    leads = lead_max(cfg, window) + 1
    for kind in kinds:
        cs = [c for c in survivors[kind] if c.onset is not None]
        for u in range(leads):
            js = [max(0, c.onset - u) for c in cs]
            w, _ = _weights(chart, cs, js, cfg)
            for j, wi in zip(js, w / w.sum()):
                p[j] += wi / (len(kinds) * leads)
    return p, kinds


def _intersecting(cs, t_lo, t_hi, song_ms):
    """Candidates whose scope meets [t_lo, t_hi); a scope with b = T contains T (an EOS-only window)."""
    return [c for c in cs if c.iv.a < t_hi and (c.iv.b > t_lo or c.iv.b >= song_ms)]


def draw_window(chart: Chart, star_cells, rng, cfg: DrawConfig, *, star_conditions: bool, window: int = WINDOW):
    K = chart.K
    if cfg.selection == 'natural':
        start = old_start(rng, K, window)
        record = dict(natural=True, drop_all=False, aligned=False, weight=1.0, chosen=[])
        return WindowDraw(start, min(start + window, K + 1), (), 1.0, record, [])
    ln = ln_candidates(chart, rng, cfg)
    star = star_candidates(chart, star_cells, rng, cfg) if star_conditions else []
    record = dict(candidates_ln=len(ln), candidates_star=len(star), drop_all=False, aligned=False)
    survivors = {0: [], 1: []}
    if rng.random() < cfg.drop_all:
        record['drop_all'] = True
    else:
        for kind, cs in ((0, ln), (1, star)):
            if rng.random() < cfg.drop_kind:
                record[f'drop_kind_{kind}'] = True
                continue
            survivors[kind] = [c for c in cs if rng.random() >= cfg.drop_interval]
    survivors = {k: v for k, v in survivors.items() if v}
    for cs in survivors.values():
        for c in cs:
            c.onset = onset_of(chart, c.iv)
    weight = 1.0
    if cfg.selection == 'per_song':
        start = old_start(rng, K, window)
        stop = min(start + window, K + 1)
        chosen = []
        for kind, cs in sorted(survivors.items()):
            cap = cfg.per_song_ln_max if kind == 0 else cfg.per_song_star_max
            n = int(rng.integers(1, min(cap, len(cs)) + 1))
            chosen += _choose(rng, cs, n, cfg.p_consecutive)
    else:
        aligned = None
        pa, kinds = (align_distribution(chart, survivors, cfg, window) if (survivors and cfg.p_align > 0)
                     else (None, []))
        if pa is not None and rng.random() < cfg.p_align:
            kind = kinds[int(rng.integers(0, len(kinds)))]
            u = int(rng.integers(0, lead_max(cfg, window) + 1))
            cs = [c for c in survivors[kind] if c.onset is not None]
            js = [max(0, c.onset - u) for c in cs]
            w, z = _weights(chart, cs, js, cfg)
            i = int(rng.choice(len(cs), p=w / w.sum()))
            aligned, start = cs[i], js[i]
            record.update(aligned=True, aligned_kind=kind, lead=u, aligned_weight=float(w[i]),
                          aligned_z=None if np.isnan(z[i]) else float(z[i]), aligned_class=aligned.cls)
        else:
            start = old_start(rng, K, window)
        stop = min(start + window, K + 1)
        if pa is not None and cfg.ipw:
            po = p_old(K, window)
            weight = float(po[start] / ((1.0 - cfg.p_align) * po[start] + cfg.p_align * pa[start]))
        t_lo = chart.time(start)
        t_hi = chart.time(stop) if stop <= K else math.inf
        chosen = []
        for kind, cs in sorted(survivors.items()):
            avail = _intersecting(cs, t_lo, t_hi, chart.song_ms)
            if aligned is not None and aligned.iv.kind == kind and not any(c is aligned for c in avail):
                avail.append(aligned)      # its onset decision is scored; e.g. an EOS-only window and b < T
            avail = sorted(avail, key=lambda c: c.iv.a)
            if not avail:
                continue
            n = int(rng.integers(1, min(cfg.per_window_max, len(avail)) + 1))
            if aligned is not None and aligned.iv.kind == kind:
                at = next(i for i, c in enumerate(avail) if c is aligned)
                if rng.random() < cfg.p_consecutive:
                    lo = max(0, at - n + 1)
                    hi = min(at, len(avail) - n)
                    s = int(rng.integers(lo, hi + 1))
                    chosen += avail[s:s + n]
                else:
                    others = [c for c in avail if c is not aligned]
                    idx = rng.choice(len(others), size=n - 1, replace=False) if n > 1 else []
                    chosen += sorted([aligned] + [others[i] for i in idx], key=lambda c: c.iv.a)
            else:
                chosen += _choose(rng, avail, n, cfg.p_consecutive)
    track = tuple(c.iv for c in chosen)
    validate_track(track, song_ms=chart.song_ms, star_value=cfg.star_value)
    record.update(weight=weight, chosen=[dict(kind=c.iv.kind, cls=c.cls, beats=c.beats, onset=c.onset) for c in chosen])
    return WindowDraw(start, stop, track, weight, record, chosen)


def validate_track(track, *, song_ms: float | None = None, star_value: str = 'absolute'):
    """The effective-track invariant: finite positive-length scopes within [0, T], LN share in [0, 1],
    absolute star >= 0 (a residual may be negative), and no overlap within a kind."""
    for kind in (0, 1):
        ivs = sorted((iv for iv in track if iv.kind == kind), key=lambda iv: iv.a)
        for iv in ivs:
            if not (math.isfinite(iv.a) and math.isfinite(iv.b) and iv.b > iv.a):
                raise ContractError('Interval bounds must be finite with positive length')
            if song_ms is not None and not (0.0 <= iv.a and iv.b <= song_ms):
                raise ContractError('An interval lies within [0, T]')
            if not math.isfinite(iv.value):
                raise ContractError('Interval values are finite')
            if kind == 0 and not 0.0 <= iv.value <= 1.0:
                raise ContractError('LN share must be in [0, 1]')
            if kind == 1 and star_value == 'absolute' and iv.value < 0.0:
                raise ContractError('An absolute star value is nonnegative')
        if any(x.b > y.a for x, y in zip(ivs, ivs[1:])):
            raise ContractError('Intervals of one kind may not overlap')
