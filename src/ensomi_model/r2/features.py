"""Parameter-independent R2 inputs, computed in float64 on CPU from committed decisions.

``Chart`` holds the allowed inputs (head times H, grid, song length T) and the
decisions known so far. ``derive`` replays the codes into per-decision state
arrays; every array indexed by decision k describes the state *before* k, so a
query at k reads only decisions < k. History tokens [N,2,100], queries
[m,2,150], candidate features [C,34], placement relations [C,28] and condition
frames [*,16] follow spec sections 2 and 3. The left hand frame lists lanes
(0,1,2,3), the right hand frame (3,2,1,0).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .candidates import GapCandidates, build_candidates, phase
from .common import HAND_LANES, ContractError, GridArrays, psi_pair

HISTORY_DIM = 100
LANE_QUERY_DIM = 15
QUERY_DIM = 150
RELATION_DIM = 28
DENSITY_BEATS = (1, 2, 4, 8, 16, 32)
LOOKAHEAD = 16
FRAME_DIM = 16
TOKEN_DIM = 18
KINDS = ('ln', 'star')


@dataclass
class Chart:
    head_ms: np.ndarray
    song_ms: float
    grid: GridArrays
    actions: np.ndarray          # [n,4] codes of the decisions known so far, n <= K+1
    gap: np.ndarray              # [n,4] gap release times, NaN when absent
    cache: dict = field(default_factory=dict)

    @property
    def K(self):
        return len(self.head_ms)

    @property
    def n(self):
        return len(self.actions)

    def time(self, k):
        return float(self.head_ms[k]) if k < self.K else float(self.song_ms)

    def times(self, ks):
        ks = np.asarray(ks)
        return np.where(ks < self.K, self.head_ms[np.minimum(ks, self.K - 1)], self.song_ms)

    @property
    def head_beats(self):
        if 'hb' not in self.cache:
            self.cache['hb'] = self.grid.beat(self.head_ms)
            self.cache['song_beat'] = float(self.grid.beat(self.song_ms))
        return self.cache['hb']

    def beat_of(self, ks):
        hb = self.head_beats
        ks = np.asarray(ks)
        return np.where(ks < self.K, hb[np.minimum(ks, self.K - 1)], self.cache['song_beat'])

    def derived(self):
        if 'derived' not in self.cache:
            self.cache['derived'] = derive(self)
        return self.cache['derived']

    def candidates(self, k) -> GapCandidates:
        key = ('cand', k)
        if key not in self.cache:
            if k < 1 or k > self.K:
                raise ContractError('No gap before decision 0')
            self.cache[key] = build_candidates(self.head_ms[k - 1], self.time(k), self.grid)
        return self.cache[key]

    def with_decisions(self, actions, gap) -> 'Chart':
        c = Chart(self.head_ms, self.song_ms, self.grid, actions, gap)
        for key, v in self.cache.items():
            if key in ('hb', 'song_beat') or (isinstance(key, tuple) and key[0] == 'cand'):
                c.cache[key] = v
        return c


@dataclass
class Derived:
    held: np.ndarray        # [n+1,4] bool, before decision k
    start: np.ndarray       # [n+1,4] held LN start, NaN
    birth: np.ndarray       # [n+1,4] birth row, -1
    release: np.ndarray     # [n,4] release time committed by decision k, NaN
    row_release: np.ndarray  # [n,4] bool
    attack: np.ndarray      # [n,4] bool head placed in lane at decision k
    ln_head: np.ndarray     # [n,4] bool
    last_attack: np.ndarray  # [n+1,4] before decision k, NaN
    last_release: np.ndarray  # [n+1,4]
    cum_heads: np.ndarray   # [n+1] committed head objects before k
    cum_ln: np.ndarray      # [n+1]


def derive(chart: Chart) -> Derived:
    n, K = chart.n, chart.K
    acts = chart.actions.tolist()
    gap = chart.gap
    held = np.zeros((n + 1, 4), dtype=bool)
    start = np.full((n + 1, 4), np.nan)
    birth = np.full((n + 1, 4), -1, dtype=np.int64)
    release = np.full((n, 4), np.nan)
    row_rel = np.zeros((n, 4), dtype=bool)
    attack = np.zeros((n, 4), dtype=bool)
    ln_head = np.zeros((n, 4), dtype=bool)
    h = [False] * 4
    s = [np.nan] * 4
    b = [-1] * 4
    for k in range(n):
        held[k], start[k], birth[k] = h, s, b
        t = chart.time(k)
        for lane in range(4):
            c = acts[k][lane]
            if h[lane]:
                if c == 0:
                    if k == K:
                        raise ContractError('EOS keeps a hold')
                    continue
                if c == 1:
                    release[k, lane] = t
                    row_rel[k, lane] = True
                else:
                    release[k, lane] = gap[k, lane]
                h[lane], s[lane], b[lane] = False, np.nan, -1
                if c >= 3:
                    attack[k, lane] = True
                    if c == 4:
                        ln_head[k, lane] = True
                        h[lane], s[lane], b[lane] = True, t, k
            else:
                if c >= 3 or (k == K and c):
                    raise ContractError(f'Invalid code {c} on a free lane at decision {k}')
                if c:
                    attack[k, lane] = True
                    if c == 2:
                        ln_head[k, lane] = True
                        h[lane], s[lane], b[lane] = True, t, k
    held[n], start[n], birth[n] = h, s, b
    times = chart.times(np.arange(n))[:, None]
    att_t = np.where(attack, times, -np.inf)
    rel_t = np.where(np.isnan(release), -np.inf, release)
    last_attack = np.concatenate((np.full((1, 4), -np.inf), np.maximum.accumulate(att_t, 0)), 0)
    last_release = np.concatenate((np.full((1, 4), -np.inf), np.maximum.accumulate(rel_t, 0)), 0)
    last_attack[np.isinf(last_attack)] = np.nan
    last_release[np.isinf(last_release)] = np.nan
    cum_heads = np.concatenate(([0], np.cumsum(attack.sum(1))))
    cum_ln = np.concatenate(([0], np.cumsum(ln_head.sum(1))))
    return Derived(held, start, birth, release, row_rel, attack, ln_head, last_attack, last_release, cum_heads, cum_ln)


def _views(chart: Chart, u, k_rows, start):
    """Six views of times u relative to t_k, t_{k-1} and the held start: [...,12], zero where u is NaN."""
    valid = ~np.isnan(u) & ~np.isnan(start)
    uu = np.where(valid, u, 0.0)
    ss = np.where(valid, start, 0.0)
    tk = chart.times(k_rows)
    tp = chart.times(np.maximum(k_rows - 1, 0))
    bu, bs = chart.grid.beat(uu), chart.grid.beat(ss)
    bk, bp = chart.beat_of(k_rows), chart.beat_of(np.maximum(k_rows - 1, 0))
    tk = np.broadcast_to(tk, uu.shape)
    tp = np.broadcast_to(tp, uu.shape)
    bk = np.broadcast_to(bk, uu.shape)
    bp = np.broadcast_to(bp, uu.shape)
    out = np.concatenate((psi_pair(uu - tk, bu - bk), psi_pair(uu - tp, bu - bp), psi_pair(uu - ss, bu - bs)), -1)
    return out * valid[..., None]


def _onehot(values, size):
    out = np.zeros(values.shape + (size,))
    np.put_along_axis(out, np.clip(values, 0, size - 1)[..., None], 1.0, -1)
    return out


def history_tokens(chart: Chart, N: int) -> np.ndarray:
    """[N,2,100] tokens for head decisions 0..N-1 (EOS never becomes a token)."""
    if N > min(chart.n, chart.K):
        raise ContractError('History beyond the committed head decisions')
    d = chart.derived()
    ks = np.arange(N)
    acts = chart.actions[:N].astype(np.int64)
    rel = d.release[:N]
    lane = np.concatenate((_onehot(acts, 5), d.held[:N, :, None].astype(np.float64),
                           (~np.isnan(rel))[..., None].astype(np.float64),
                           _views(chart, rel, ks[:, None], d.start[:N])), -1)          # [N,4,19]
    t, b = chart.head_ms[:N], chart.head_beats[:N]
    tprev = np.concatenate(([t[0]], t[:-1])) if N else t
    bprev = np.concatenate(([b[0]], b[:-1])) if N else b
    shared = np.concatenate((psi_pair(t - tprev, b - bprev), psi_pair(t - chart.head_ms[0], b - chart.head_beats[0]),
                             phase(b), _onehot(d.attack[:N].sum(1), 5), _onehot(d.ln_head[:N].sum(1), 5)), -1)
    hands = [np.concatenate((lane[:, list(order)].reshape(N, 4 * 19), shared), -1) for order in HAND_LANES]
    return np.stack(hands, 1).astype(np.float32)


def lane_query(chart: Chart, ks) -> np.ndarray:
    """[m,4,15] per physical lane: held, LN age, last attack age, last release age, absence bits."""
    d = chart.derived()
    ks = np.asarray(ks)
    t = chart.times(ks)[:, None]
    bt = chart.beat_of(ks)[:, None]
    g = chart.grid

    def age(x):
        ok = ~np.isnan(x)
        xx = np.where(ok, x, 0.0)
        return psi_pair(np.where(ok, t - xx, 0.0), np.where(ok, bt - g.beat(xx), 0.0)) * ok[..., None], ok

    ln_age, _ = age(d.start[ks])
    att, att_ok = age(d.last_attack[ks])
    rel, rel_ok = age(d.last_release[ks])
    return np.concatenate((d.held[ks][..., None].astype(np.float64), ln_age, att, rel,
                           (~att_ok)[..., None].astype(np.float64), (~rel_ok)[..., None].astype(np.float64)), -1)


def query_features(chart: Chart, ks) -> np.ndarray:
    """[m,2,150] exact query features for decisions ks (EOS = K)."""
    ks = np.asarray(ks)
    K, T = chart.K, chart.song_ms
    g = chart.grid
    lanes = lane_query(chart, ks)
    t, bt = chart.times(ks), chart.beat_of(ks)
    kp = np.maximum(ks - 1, 0)
    tp, bp = chart.times(kp), chart.beat_of(kp)
    first = ks == 0
    gap = psi_pair(np.where(first, 0.0, t - tp), np.where(first, 0.0, bt - bp))
    remaining = psi_pair(T - t, chart.cache['song_beat'] - bt)
    shared = np.concatenate((gap, remaining, (t / T)[:, None], (ks / max(K, 1))[:, None],
                             first[:, None].astype(np.float64), (ks == K)[:, None].astype(np.float64),
                             phase(bt), (g.bpm(t) / 120.0)[:, None], (g.meter(t) / 4.0)[:, None]), -1)
    hb = chart.head_beats
    lo = np.searchsorted(hb, bt, side='right')
    dens = np.stack([np.log1p(np.searchsorted(hb, bt + h, side='right') - lo) for h in DENSITY_BEATS], -1)
    dens = np.where((ks == K)[:, None], 0.0, dens)
    ahead = []
    for i in range(1, LOOKAHEAD + 1):
        idx = ks + i
        ok = idx <= K - 1
        a, b = np.minimum(idx, K - 1), np.minimum(np.maximum(idx - 1, 0), K - 1)
        v = psi_pair(chart.head_ms[a] - chart.head_ms[b], hb[a] - hb[b])
        ahead.append(v * ok[:, None])
    shared = np.concatenate((shared, dens, np.concatenate(ahead, -1)), -1)
    hands = [np.concatenate((lanes[:, list(order)].reshape(len(ks), 4 * LANE_QUERY_DIM), shared), -1)
             for order in HAND_LANES]
    return np.stack(hands, 1).astype(np.float32)


# ---- conditions ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class Interval:
    kind: int        # 0 = ln_share, 1 = star
    a: float
    b: float
    value: float


def _interval_counts(chart: Chart, iv: Interval, k: int):
    d = chart.derived()
    ia = int(np.searchsorted(chart.head_ms, iv.a, side='left'))
    ib = int(np.searchsorted(chart.head_ms, iv.b, side='left'))
    hi = min(ib, k)
    heads = int(d.cum_heads[hi] - d.cum_heads[ia]) if hi > ia else 0
    lns = int(d.cum_ln[hi] - d.cum_ln[ia]) if hi > ia else 0
    remaining = max(0, ib - max(ia, k))
    return heads, lns, remaining


def _norm_value(iv: Interval):
    return 2.0 * iv.value - 1.0 if iv.kind == 0 else iv.value / 4.0


def frames(chart: Chart, track, times, k: int) -> np.ndarray:
    """[Q,2,16] FiLM frames (ln, star) at query times for decision k; absent kinds are zero."""
    times = np.asarray(times, dtype=np.float64)
    out = np.zeros(times.shape + (2, FRAME_DIM))
    if not track:
        return out
    g = chart.grid
    bv = g.beat(times)
    for kind in (0, 1):
        ivs = [iv for iv in track if iv.kind == kind]
        if not ivs:
            continue
        out[..., kind, 15] = 1.0  # presence
        for iv in ivs:
            active = (times >= iv.a) & (times < iv.b)
            if not active.any():
                continue
            heads, lns, remaining = _interval_counts(chart, iv, k)
            ba, bb = float(g.beat(iv.a)), float(g.beat(iv.b))
            f = np.concatenate((np.full(times.shape + (1,), _norm_value(iv)),
                                psi_pair(iv.a - times, ba - bv), psi_pair(iv.b - times, bb - bv),
                                ((times - iv.a) / (iv.b - iv.a))[..., None],
                                np.broadcast_to([np.log1p(heads), np.log1p(lns), lns / heads if heads else 0.0,
                                                 np.log1p(remaining), 1.0], times.shape + (5,))), -1)
            # channels: value 1, offsets 8, progress 1, counts 2, ratio 1, remaining 1, active 1 -> 15; presence 1
            out[..., kind, :15] = np.where(active[..., None], f, out[..., kind, :15])
    return out


def tokens(chart: Chart, track, times, k: int):
    """[Q,NI,18] whole-track interval tokens relative to each query time (token conditioner)."""
    times = np.asarray(times, dtype=np.float64)
    if not track:
        return np.zeros(times.shape + (0, TOKEN_DIM), dtype=np.float32)
    g = chart.grid
    bv = g.beat(times)
    out = []
    for iv in track:
        heads, lns, remaining = _interval_counts(chart, iv, k)
        ba, bb = float(g.beat(iv.a)), float(g.beat(iv.b))
        started = times >= iv.a
        active = started & (times < iv.b)
        progress = np.clip((times - iv.a) / (iv.b - iv.a), 0.0, 1.0)
        kind = np.zeros(times.shape + (2,))
        kind[..., iv.kind] = 1.0
        out.append(np.concatenate((kind, np.full(times.shape + (1,), _norm_value(iv)),
                                   psi_pair(iv.a - times, ba - bv), psi_pair(iv.b - times, bb - bv),
                                   active[..., None], progress[..., None],
                                   np.broadcast_to([np.log1p(heads), np.log1p(lns), lns / heads if heads else 0.0,
                                                    np.log1p(remaining)], times.shape + (4,)),
                                   started[..., None]), -1))
    return np.stack(out, -2).astype(np.float32)


# ---- release factors ----------------------------------------------------------------------------

@dataclass
class Factor:
    owner: int                 # decision row inside the batch
    orientation: int           # 0 forward, 1 mirrored
    codes: np.ndarray          # [4] oriented codes
    slot: int                  # releasing lane in the oriented frame
    lane: int                  # physical lane
    lane_feats: np.ndarray     # [15]
    cand_feats: np.ndarray     # [C,34]
    relations: np.ndarray      # [C,28]
    times: np.ndarray          # [C]
    target: int | None
    k: int


def make_factor(chart: Chart, k: int, owner: int, codes, lane: int, orientation: int, placed: dict,
                lane_feats, target_time=None) -> Factor:
    """One directed release factor: lane ``lane`` releasing in gap k, with ``placed`` visible releases."""
    d = chart.derived()
    births, starts = d.birth[k], d.start[k]
    cands = chart.candidates(k)
    target = None
    if target_time is not None:
        hit = np.flatnonzero(cands.times == target_time)
        if len(hit) != 1:
            raise ContractError(f'Release {target_time} at decision {k} is not a candidate')
        target = int(hit[0])
    oriented = np.asarray(codes)[::-1].copy() if orientation else np.asarray(codes).copy()
    return Factor(owner, orientation, oriented, lane if orientation == 0 else 3 - lane, lane, lane_feats[lane],
                  cands.lane_features(starts[lane], float(chart.grid.beat(starts[lane]))),
                  placement_relations(cands, chart.grid, orientation, placed, births, lane).astype(np.float32),
                  cands.times, target, k)


def gap_release_lanes(held, codes, orientation: int) -> list[int]:
    lanes = [l for l in range(4) if held[l] and codes[l] in (2, 3, 4)]
    return sorted(lanes, reverse=bool(orientation))


def row_placements(chart: Chart, k: int, held, codes) -> dict:
    t = chart.time(k)
    return {l: (t, True) for l in range(4) if held[l] and codes[l] == 1}


def placement_relations(cands: GapCandidates, grid: GridArrays, orientation: int, placed: dict, births, lane: int):
    """[C,28]: per oriented lane slot (u - r in ms/beats through psi, same birth, placed, on row)."""
    C = len(cands.times)
    out = np.zeros((C, 4, 7))
    for slot in range(4):
        p = slot if orientation == 0 else 3 - slot
        if p not in placed:
            continue
        r, on_row = placed[p]
        out[:, slot, :4] = psi_pair(cands.times - r, cands.beats - float(grid.beat(r)))
        out[:, slot, 4] = float(births[p] == births[lane] and births[p] >= 0)
        out[:, slot, 5] = 1.0
        out[:, slot, 6] = float(on_row)
    return out.reshape(C, 28)


def decision_factors(chart: Chart, k: int, owner: int, codes, releases, lane_feats, orientation: int):
    """Teacher-forced release factors of one decision in one orientation (placements in factor order)."""
    held = chart.derived().held[k]
    lanes = gap_release_lanes(held, codes, orientation)
    if not lanes:
        return []
    placed = row_placements(chart, k, held, codes)
    out = []
    for lane in lanes:
        out.append(make_factor(chart, k, owner, codes, lane, orientation, dict(placed), lane_feats,
                               float(releases[lane])))
        placed[lane] = (float(releases[lane]), False)
    return out
