"""Parameter-independent R2 inputs, computed in float64 on CPU from committed decisions.

``Chart`` holds the allowed inputs (head times H, grid, song length T) and the
decisions known so far. ``derive`` replays the codes into per-decision state
arrays; every array indexed by decision k describes the state *before* k, so a
query at k reads only decisions < k. History tokens [N,2,100], queries
[m,2,150], candidate features [C,34], placement relations [C,28] and condition
frames [*,17] follow spec sections 2 and 3 and plan v4 section 6.8. The left hand
frame lists lanes (0,1,2,3), the right hand frame (3,2,1,0).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .candidates import GapCandidates, build_candidates, phase
from .common import HAND_LANES, ContractError, GridArrays, psi_pair

HISTORY_DIM = 100
LANE_QUERY_DIM = 15
QUERY_DIM = 150
LN_LEVEL_DIM = 3
LN_LENGTH_DIM = 2
ANCHOR_DIM = 13
THETA_DIM = 11
LN_LEVEL_MODES = ('off', 'on')
LN_LEVEL_EPS = 1e-6
RELATION_DIM = 28
DENSITY_BEATS = (1, 2, 4, 8, 16, 32)
LOOKAHEAD = 16
FRAME_DIM = 17
TOKEN_DIM = 18
KINDS = ('ln_share', 'difficulty')   # frame kind index 0, 1; stage-4 style attributes are not framed
PRESENCE = ('none', 'anywhere')      # 'lead-in' is reserved and raises
STAR_VALUES = ('absolute', 'residual')
LEAD_IN_CHANNEL = FRAME_DIM - 1


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
    cum_repeat: np.ndarray  # [n+1] heads at rows j < k whose lane also has a head at row j - 1
    cum_held: np.ndarray    # [n+1] lanes held entering rows j < k
    cum_c3: np.ndarray
    cum_c4: np.ndarray
    cum_patterns: np.ndarray       # [n+1,16] head-mask counts
    cum_pattern_repeat: np.ndarray # [n+1,3] matching masks at lags 1, 2, 4
    cum_ln_log_length: np.ndarray  # log2 beat lengths of holds closed before k
    cum_closed_ln: np.ndarray
    cum_hand_heads: np.ndarray     # [n+1,2] physical left/right head counts


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
    repeat = np.zeros(n, dtype=np.int64)
    if n > 1:
        repeat[1:] = (attack[1:] & attack[:-1]).sum(1)
    cum_repeat = np.concatenate(([0], np.cumsum(repeat)))
    cum_held = np.concatenate(([0], np.cumsum(held[:n].sum(1))))
    def cumulative(values):
        return np.concatenate((np.zeros((1,) + values.shape[1:]), np.cumsum(values, axis=0)), axis=0)

    counts = attack.sum(1)
    patterns = (attack * (1 << np.arange(4))).sum(1)
    pattern_repeat = np.zeros((n, 3))
    for i, lag in enumerate((1, 2, 4)):
        pattern_repeat[lag:, i] = patterns[lag:] == patterns[:-lag]
    closed = ~np.isnan(release)
    log_length = np.zeros((n, 4))
    if chart.grid is not None:  # generation seed records build grid-free charts
        log_length[closed] = np.log2(np.maximum(chart.grid.beat(release[closed]) -
                                               chart.grid.beat(start[:n][closed]), 1e-4))
    return Derived(held, start, birth, release, row_rel, attack, ln_head, last_attack, last_release, cum_heads, cum_ln,
                   cum_repeat, cum_held, cumulative(counts >= 3), cumulative(counts == 4),
                   cumulative(np.eye(16)[patterns]), cumulative(pattern_repeat), cumulative(log_length.sum(1)),
                   cumulative(closed.sum(1)), cumulative(attack.reshape(n, 2, 2).sum(2)))


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


def ln_level_features(level: float | None) -> np.ndarray:
    """Three shared channels: known, level, logit(level); None makes all three zero.

    Known levels must be finite and in [0,1]. Only the logit's argument is clipped
    to [1e-6, 1-1e-6], preserving exact zero and one in the level channel.
    """
    if level is None:
        return np.zeros(LN_LEVEL_DIM, dtype=np.float32)
    if isinstance(level, (bool, np.bool_)) or not np.isscalar(level):
        raise ContractError('ln_level must be a number in [0,1] or None')
    try:
        value = float(level)
    except (TypeError, ValueError) as exc:
        raise ContractError('ln_level must be a number in [0,1] or None') from exc
    if isinstance(level, (str, bytes)) or not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ContractError('ln_level must be a finite number in [0,1] or None')
    clipped = np.clip(value, LN_LEVEL_EPS, 1.0 - LN_LEVEL_EPS)
    return np.array([1.0, value, np.log(clipped / (1.0 - clipped))], dtype=np.float32)


def ln_length_features(length: float | None) -> np.ndarray:
    """Known bit and median log2 hold length in beats; None gives two zeros."""
    from .ln_level import checked_length
    if length is None:
        return np.zeros(LN_LENGTH_DIM, dtype=np.float32)
    return np.array([1.0, checked_length(length)], dtype=np.float32)


def anchor_features(chart: Chart, ks) -> np.ndarray:
    """Thirteen committed-prefix channels, with own-hand share in each hand view."""
    ks = np.asarray(ks)
    d = chart.derived()
    rows, heads = np.maximum(ks, 1), np.maximum(d.cum_heads[ks], 1)
    nonempty = d.cum_patterns[ks, 1:]
    p = nonempty / np.maximum(nonempty.sum(1), 1)[:, None]
    entropy = -(p * np.log2(np.maximum(p, 1e-30))).sum(1)
    shared = np.column_stack((d.cum_heads[ks] / rows, d.cum_c3[ks] / rows, d.cum_c4[ks] / rows,
                              d.cum_repeat[ks] / heads, d.cum_held[ks] / (4 * rows),
                              d.cum_ln[ks] / heads,
                              d.cum_ln_log_length[ks] / np.maximum(d.cum_closed_ln[ks], 1), entropy,
                              d.cum_pattern_repeat[ks] / np.maximum(ks[:, None] - np.array([1, 2, 4]), 1),
                              np.log1p(ks)))
    own = d.cum_hand_heads[ks] / heads[:, None]
    return np.concatenate((np.broadcast_to(shared[:, None], (len(ks), 2, shared.shape[-1])),
                           own[..., None]), -1).astype(np.float32)


def theta_features(value) -> np.ndarray:
    """Known bit and ten standardised coordinates; own-hand share changes sign on mirroring."""
    out = np.zeros((2, THETA_DIM), dtype=np.float32)
    if value is not None:
        out[:, 0] = 1
        out[:, 1:] = value
        out[1, -1] *= -1
    return out


def query_features(chart: Chart, ks, *, ln_level: str = 'off', level: float | None = None,
                   ln_length: str = 'off', length: float | None = None, anchor: str = 'off',
                   theta: str = 'off', theta_value=None) -> np.ndarray:
    """[m,2,150] base queries at ks (EOS = K), followed by enabled reader channels.

    LN share adds three shared channels and length adds two. Anchor adds thirteen
    prefix channels; theta adds a known bit and ten standardised coordinates.
    Own-hand channels transform with the hand view. Disabled inputs add nothing.
    """
    if ln_level not in LN_LEVEL_MODES:
        raise ContractError(f'ln_level must be one of {LN_LEVEL_MODES}')
    if ln_length not in LN_LEVEL_MODES or (ln_length == 'on' and ln_level != 'on'):
        raise ContractError('ln_length must be off|on and requires ln_level on')
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
    out = np.stack(hands, 1).astype(np.float32)
    if ln_level == 'on':
        channels = np.broadcast_to(ln_level_features(level), out.shape[:-1] + (LN_LEVEL_DIM,))
        out = np.concatenate((out, channels), -1)
    if ln_length == 'on':
        channels = np.broadcast_to(ln_length_features(length), out.shape[:-1] + (LN_LENGTH_DIM,))
        out = np.concatenate((out, channels), -1)
    if anchor == 'on':
        out = np.concatenate((out, anchor_features(chart, ks)), -1)
    if theta == 'on':
        channels = np.broadcast_to(theta_features(theta_value), (len(ks), 2, THETA_DIM))
        out = np.concatenate((out, channels), -1)
    return out


# ---- conditions ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class Interval:
    """One effective-track interval: kind 0 LN share, kind 1 difficulty; [a, b), or [a, T] when b = T.

    ``value`` is what the frame encodes: the LN-share target, or for difficulty the absolute
    tiled star (``star_value='absolute'``) or the residual target - b(S) (``'residual'``).
    """
    kind: int
    a: float
    b: float
    value: float


def contains(iv: Interval, times, song_ms: float):
    """Activity test: a <= t < b, closed at T when b = T (a whole-song scope contains T)."""
    times = np.asarray(times, dtype=np.float64)
    return (times >= iv.a) & ((times < iv.b) | ((iv.b >= song_ms) & (times <= song_ms)))


def _row_bounds(chart: Chart, iv: Interval):
    ia = int(np.searchsorted(chart.head_ms, iv.a, side='left'))
    ib = chart.K if iv.b >= chart.song_ms else int(np.searchsorted(chart.head_ms, iv.b, side='left'))
    return ia, max(ia, ib)


def _interval_counts(chart: Chart, iv: Interval, k: int):
    """Committed head objects and LN heads of decisions < k with head time in the scope; remaining rows."""
    d = chart.derived()
    ia, ib = _row_bounds(chart, iv)
    hi = min(ib, k)
    heads = int(d.cum_heads[hi] - d.cum_heads[ia]) if hi > ia else 0
    lns = int(d.cum_ln[hi] - d.cum_ln[ia]) if hi > ia else 0
    remaining = max(0, ib - max(ia, k))
    return heads, lns, remaining


def star_proxies(chart: Chart, iv: Interval, k: int):
    """Committed difficulty statistics of the scope before decision k (mirror-invariant):
    mean chord size / 4, same-lane repeat rate, held-lane occupancy, LN share."""
    d = chart.derived()
    ia, ib = _row_bounds(chart, iv)
    hi = min(ib, k)
    rows = hi - ia
    if rows <= 0:
        return 0.0, 0.0, 0.0, 0.0
    heads = int(d.cum_heads[hi] - d.cum_heads[ia])
    lns = int(d.cum_ln[hi] - d.cum_ln[ia])
    held = int(d.cum_held[hi] - d.cum_held[ia])
    rep = int(d.cum_repeat[hi] - d.cum_repeat[ia + 1]) if rows > 1 else 0
    base = int(d.cum_heads[hi] - d.cum_heads[ia + 1]) if rows > 1 else 0
    return (heads / rows / 4.0, rep / base if base else 0.0, held / (4.0 * rows), lns / heads if heads else 0.0)


def encode_value(iv: Interval, star_value: str = 'absolute') -> float:
    if iv.kind == 0:
        return 2.0 * iv.value - 1.0
    if star_value == 'residual':
        return float(np.clip(iv.value / 0.5, -3.0, 3.0))
    return iv.value / 4.0


def frames(chart: Chart, track, times, k: int, *, presence: str = 'none', full_track=None,
           star_value: str = 'absolute') -> np.ndarray:
    """[Q,2,17] FiLM frames (LN share, difficulty) at query times for decision k.

    ``track`` holds the intervals decision k may read (rule L applied by the caller); a kind
    with no such interval active at a time is all zero there. Channels: value 1, offsets to
    the bounds 8, progress 1, committed statistics 4 (LN: log1p heads, log1p LNs, ratio, 0;
    difficulty: ``star_proxies``), log1p remaining head rows 1, active 1, and the reserved
    lead-in channel 1, which is zero under ``presence='none'``. ``presence='anywhere'``
    (v1 reproduction and power checks only) sets that channel for a kind on every query
    whenever ``full_track`` has an interval of the kind.
    """
    times = np.asarray(times, dtype=np.float64)
    out = np.zeros(times.shape + (2, FRAME_DIM))
    if presence not in PRESENCE:
        raise ContractError(f'presence must be one of {PRESENCE}; lead-in is reserved')
    if presence == 'anywhere':
        for kind in (0, 1):
            if any(iv.kind == kind for iv in (full_track if full_track is not None else track)):
                out[..., kind, LEAD_IN_CHANNEL] = 1.0
    if not track:
        return out
    g = chart.grid
    bv = g.beat(times)
    T = chart.song_ms
    for iv in track:
        active = contains(iv, times, T)
        if not active.any():
            continue
        if iv.kind == 0:
            heads, lns, remaining = _interval_counts(chart, iv, k)
            stats = [np.log1p(heads), np.log1p(lns), lns / heads if heads else 0.0, 0.0]
        else:
            _, _, remaining = _interval_counts(chart, iv, k)
            stats = list(star_proxies(chart, iv, k))
        ba, bb = float(g.beat(iv.a)), float(g.beat(iv.b))
        f = np.concatenate((np.full(times.shape + (1,), encode_value(iv, star_value)),
                            psi_pair(iv.a - times, ba - bv), psi_pair(iv.b - times, bb - bv),
                            ((times - iv.a) / (iv.b - iv.a))[..., None],
                            np.broadcast_to(stats + [np.log1p(remaining), 1.0], times.shape + (6,))), -1)
        out[..., iv.kind, :LEAD_IN_CHANNEL] = np.where(active[..., None], f, out[..., iv.kind, :LEAD_IN_CHANNEL])
    return out


def tokens(chart: Chart, track, times, k: int, *, star_value: str = 'absolute'):
    """[Q,NI,18] whole-track interval tokens relative to each query time (token conditioner).

    Each query sees every interval before it starts and after it ends, so this is a lead-in
    form: ``R2Config`` refuses the token conditioner unless ``token_lead_in`` is set. The value
    channel is ``encode_value``, as in the FiLM frames.
    """
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
        out.append(np.concatenate((kind, np.full(times.shape + (1,), encode_value(iv, star_value)),
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
