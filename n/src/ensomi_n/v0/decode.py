"""The v0 decoder: per-frame evidence to a music map (design §2.1 and §2.2, as built for N1 v0).

Input is an `Evidence` at 100 fps (frame ``i`` at ``10 i`` ms on osu!'s clock): a canonical beat
activation, and optionally sub-frame beat offsets, a notated downbeat activation, level-class
probabilities, a boundary activation and the F2 phase field (beat phase, concentration, log period).
The same decoder runs on N1's own heads, on BeatThis activations and on oracle impulses rendered from a
chart (`impulse_evidence`), so its ceiling and its share of any gap can be measured.

Steps:

1. **Canonical beat tracking.** A Viterbi pass over a frame-level HMM whose state is (period ``tau`` in
   whole frames, phase ``phi`` in ``[0, tau)``). The phase advances one frame per frame. At the end of
   a beat the period may change: a Gaussian cost in the log ratio of the periods (``sigma_cont`` per
   beat) for a continuing or drifting tempo, or a flat ``step_penalty`` for a tempo step to any period,
   whichever is cheaper. A *reset* ends a beat early, at a phase in ``[reset_min_phase * tau, tau -
   2]``, keeping the period, for ``reset_penalty``; it represents any same-tempo phase jump. With
   boundary evidence, the reset and step costs fall towards ``boundary_min_penalty`` as the boundary
   activation rises. Emissions: at ``phi = 0``, ``beat_weight`` times the log odds of the beat
   activation minus the log odds of ``beat_prior`` (a likelihood ratio: the activation is a posterior
   whose prior is the beat-frame rate); with a phase field, ``phase_weight * min(kappa, kappa_max) *
   cos(2 pi (phi / tau - phase))`` at every state and frame; with a period field, ``period_weight``
   times a Gaussian log density of ``log tau``, clipped at ``period_z_max`` standard deviations. The
   per-frame weights 0.02 count about one beat of frames as one observation, since a network's
   neighbouring frames are not independent. Only the beat states carry a value per frame: a non-beat
   state is the last beat plus the phase and period terms of the frames since, which are prefix sums
   (the cosine of a phase difference splits into products of cosines and sines). Resets are evaluated
   only at frames that could hold a beat (beat activation at least ``reset_min_beat``, a phase field
   within ``reset_phase_window`` of a beat, or a firing boundary). Back-pointers are stored for the beat
   states only; the transition type (continue, step or reset) follows from them.
2. **Beat times.** Each decoded beat frame moves to a neighbouring frame only when that frame's beat
   activation is strictly higher; its time is ``10 f + beat_off[f]`` where the activation there is at
   least ``offset_min_activation`` (else the frame centre), or a parabolic peak when no offsets are
   given. Beats are trimmed to one period before the first and after the last frame where the beat
   activation reaches ``trim_threshold``.
3. **Regimes.** Runs between resets and tempo steps are cut greedily into constant-tempo segments: a
   segment grows while a least-squares line ``t_i = a + i L`` keeps every residual within
   ``fit_tol_ms``; a segment needs ``min_constant_beats`` beats, and beats that start no such segment
   stay *free* (one red line per canonical beat, as mappers time performed music). Free beats lying on
   a neighbouring segment's grid are absorbed, the boundary between two segments moves by up to
   ``boundary_search`` beats to the split with the least squared residual, and neighbours with beat
   lengths within ``merge_ratio`` and phase continuous within ``merge_phase_ms`` are merged. The final
   least-squares fit gives each constant segment an exact beat length; nothing is rounded.
4. **Level.** Each constant segment, and each run of free beats, takes the level class with the highest
   mean log probability over its frames plus ``level_prior`` for the song's duration-weighted majority.
   Free runs choose among ratios 1, 2 and 4. Notated beats are the canonical beat divided by the ratio:
   ``r - 1`` points are inserted for ``r > 1``; for ``r = 0.5`` every other canonical beat is kept, the
   parity chosen by the downbeat activation (when the odd parity wins, the segment's first canonical
   beat becomes a one-beat segment at ratio 1, so no canonical line is lost).
5. **Bars and meter.** A Viterbi pass over the notated beats with states (meter ``m`` in ``meters``,
   position ``k``): the downbeat activation sampled at each beat (largest value within ``down_window``
   frames) is the emission at ``k = 0``, ``meter_logprior`` is paid per bar, a meter change costs
   ``meter_change_penalty``, and a bar may restart early for ``bar_restart_penalty`` at a regime start
   or ``bar_restart_anywhere`` elsewhere. Without downbeat evidence, bars are every 4 notated beats from
   each regime start.
6. **Output.** `MusicMap.from_beats` with the notated beats, the downbeat indices, a confidence per beat
   (the beat activation at a canonical beat's frame; inserted beats take the canonical beat before
   them) and one `Segment` per constant segment and per free canonical beat. A free beat's segment holds
   ``r`` notated beats, so the scorer's canonical anchoring (`ensomi_n.timing.Timing.anchors`) keeps
   exactly the canonical beats as lines.

Deviations from the parent's plan, and why:

- The HMM's period range is canonical, 77-167 BPM (36-78 frames), not 70-182 BPM. Beats outside
  [80, 160) BPM would leave the level ratio without a fixed meaning (a decoded 170 BPM grid is already
  "2 x canonical"); 77-167 BPM still holds both readings near the 80/160 fold (158 BPM tracks at 158 or
  79, 162 at 162 or 81).
- The tempo step is "any period for a flat cost", taken from the best beat end over all periods, so it
  costs one maximum per frame instead of a K x K matrix.
- Regimes are cut greedily, then boundaries are refined, instead of split recursively at the worst
  residual: a recursive split puts boundaries at the largest residual, which in a drift is mid-segment.
- The beat emission is a likelihood ratio (log odds minus the prior's), not ``log(beat)`` against
  ``log(1 - beat)``: with raw log odds a forward reset, which adds one beat, paid about 4.6 nats in a span
  where the beat head is quiet, and the fixture with a reset that only the phase field shows decoded a
  tempo drift instead. Leaving out the term common to all states at a frame changes no decision.
- The non-beat states are not stored (prefix sums instead), and resets are gated: together about 4x
  faster than the padded [K, P] array the plan described, which ran 2-5 s for 4 minutes.
- ``half`` is accepted and unused in v0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Callable, Sequence

import numpy as np

from ..mapformat import MusicMap
from ..timing import MERGE_MS, Segment

FRAME_MS = 10.0
LEVEL_CLASSES = (0.5, 1.0, 2.0, 4.0)
CONTINUE, STEP, RESET = 0, 1, 2

__all__ = ['Evidence', 'DecodeConfig', 'decode', 'impulse_evidence', 'LEVEL_CLASSES', 'FRAME_MS']


def _opt(x, n: int, name: str, width: int | None = None):
    if x is None:
        return None
    a = np.asarray(x, dtype=np.float64)
    shape = (n,) if width is None else (n, width)
    if a.shape != shape:
        raise ValueError(f'Evidence.{name} must have shape {shape}, got {a.shape}')
    return a


@dataclass
class Evidence:
    """Per-frame evidence at 100 fps; frame ``i`` sits at ``10 i`` ms on osu!'s clock.

    ``beat``: P(a canonical beat line at the frame). ``beat_off``: ms from the frame centre to the
    nearest beat (label minus frame), as the F1 offset head predicts. ``down``: P(a notated bar start).
    ``half``: P(a half-beat line), unused in v0. ``level``: ``[T, 4]`` probabilities over
    `LEVEL_CLASSES` (notated BPM over canonical BPM). ``boundary``: P(a tempo change or same-tempo reset).
    ``phase``, ``kappa``: the dominant von Mises component of the beat phase in [0, 1). ``logp``,
    ``logp_sigma``: log(canonical beat period in ms / 1000) and its standard deviation.
    """
    beat: np.ndarray
    beat_off: np.ndarray | None = None
    down: np.ndarray | None = None
    half: np.ndarray | None = None
    level: np.ndarray | None = None
    boundary: np.ndarray | None = None
    phase: np.ndarray | None = None
    kappa: np.ndarray | None = None
    logp: np.ndarray | None = None
    logp_sigma: np.ndarray | None = None

    def __post_init__(self):
        self.beat = np.asarray(self.beat, dtype=np.float64).reshape(-1)
        n = len(self.beat)
        for name in ('beat_off', 'down', 'half', 'boundary', 'phase', 'kappa', 'logp', 'logp_sigma'):
            setattr(self, name, _opt(getattr(self, name), n, name))
        self.level = _opt(self.level, n, 'level', len(LEVEL_CLASSES))
        if (self.phase is None) != (self.kappa is None):
            raise ValueError('phase and kappa come together')

    @property
    def n_frames(self) -> int:
        return len(self.beat)


@dataclass(frozen=True)
class DecodeConfig:
    """Penalties and weights of the decoder (log units unless stated). Defaults are hand-set for v0."""
    bpm_lo: float = 77.0                 # slowest canonical tempo of the HMM
    bpm_hi: float = 167.0                # fastest canonical tempo of the HMM
    sigma_cont: float = 0.025            # SD of the log period ratio per beat for a continuing tempo
    step_penalty: float = 9.0            # flat cost of a tempo step to any period
    reset_penalty: float = 10.0          # cost of a same-period phase reset
    boundary_min_penalty: float = 2.0    # step and reset cost where the boundary activation is 1
    boundary_floor: float = 0.01         # boundary activation at or below which the full costs apply
    reset_min_phase: float = 0.15        # earliest reset, as a share of the period
    reset_min_beat: float = 0.05         # resets only where the beat activation reaches this, or
    reset_phase_window: float = 0.06     # the phase field is this close to a beat, or a boundary fires
    beat_weight: float = 1.0
    beat_prior: float = 0.02             # share of frames that hold a beat, the activation's prior
    eps: float = 1e-3                    # activations are clipped to [eps, 1 - eps]
    phase_weight: float = 0.02           # per frame: a beat of frames counts about as one observation
    kappa_max: float = 8.0
    period_weight: float = 0.02          # per frame, as phase_weight
    period_sigma_min: float = 0.05
    period_z_max: float = 3.0
    offset_min_activation: float = 0.2   # beat activation from which the offset head is trusted
    trim_threshold: float = 0.3
    fit_tol_ms: float = 15.0             # largest residual of a constant segment
    min_constant_beats: int = 4
    boundary_search: int = 3             # beats a boundary between constant segments may move
    merge_ratio: float = 0.002
    merge_phase_ms: float = 10.0
    level_prior: float = 0.3             # bonus for the song-level majority class (mean log prob units)
    level_eps: float = 1e-4
    meters: tuple = (2, 3, 4, 5, 6, 7, 8)
    meter_logprior: tuple = (-2.0, -1.0, 0.0, -2.0, -1.5, -2.0, -2.5)   # per bar, aligned with meters
    meter_change_penalty: float = 4.0
    bar_restart_penalty: float = 0.5     # bar restart at a regime start
    bar_restart_anywhere: float = 12.0   # bar restart elsewhere
    down_weight: float = 1.0
    down_window: int = 2                 # frames either side of a beat for its downbeat activation


# ---------------------------------------------------------------- oracle evidence

def _render(times, n: int, floor: float):
    y = np.full(n, floor, dtype=np.float64)
    off = np.zeros(n, dtype=np.float64)
    if times is None or not len(times):
        return y, off
    pos = np.sort(np.asarray(times, dtype=np.float64)) / FRAME_MS
    pos = pos[(pos > -2.0) & (pos < n + 1.0)]
    if not len(pos):
        return y, off
    c = np.round(pos).astype(np.int64)
    for d, v in ((-1, 0.5), (1, 0.5), (0, 1.0)):
        j = c + d
        ok = (j >= 0) & (j < n)
        np.maximum.at(y, j[ok], v)
    j = np.arange(n, dtype=np.float64)
    i = np.clip(np.searchsorted(pos, j), 0, len(pos) - 1)
    il = np.clip(i - 1, 0, len(pos) - 1)
    left, right = pos[il] - j, pos[i] - j
    d = np.where(np.abs(left) <= np.abs(right), left, right)
    off = np.where(np.abs(d) <= 1.5, d * FRAME_MS, 0.0)
    return y, off


def _ratio_class(r) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    lr = np.log2(np.where(np.isfinite(r) & (r > 0), r, 1.0))
    return np.argmin(np.abs(lr[..., None] - np.log2(LEVEL_CLASSES)), axis=-1)


def impulse_evidence(n_frames: int, beats_ms, down_ms=None, level_ratio_at: Callable | np.ndarray | float | None = None,
                     boundaries_ms=None, floor: float = 0.01) -> Evidence:
    """Oracle evidence from label times: what a perfect head would output.

    Beats and downbeats are 1 at the nearest frame and 0.5 at the frames either side, ``floor``
    elsewhere, with exact sub-frame beat offsets for frames within 1.5 frames of a beat. The level is
    one-hot (0.97, others 0.01) on the class nearest ``level_ratio_at``: a callable of frame times in
    ms, an array of one ratio per frame, or one ratio (default 1). Boundaries are 1 within 3 frames.
    """
    n = int(n_frames)
    beat, off = _render(beats_ms, n, floor)
    down = _render(down_ms, n, floor)[0] if down_ms is not None else None
    t = FRAME_MS * np.arange(n)
    if level_ratio_at is None:
        ratio = np.ones(n)
    elif callable(level_ratio_at):
        ratio = np.asarray(level_ratio_at(t), dtype=np.float64) * np.ones(n)
    else:
        ratio = np.asarray(level_ratio_at, dtype=np.float64) * np.ones(n)
    cls = _ratio_class(ratio)
    level = np.full((n, len(LEVEL_CLASSES)), 0.01)
    level[np.arange(n), cls] = 0.97
    boundary = None
    if boundaries_ms is not None:
        boundary = np.full(n, floor)
        for b in np.asarray(boundaries_ms, dtype=np.float64):
            c = int(round(b / FRAME_MS))
            boundary[max(c - 3, 0):max(min(c + 4, n), 0)] = 1.0
    return Evidence(beat=beat, beat_off=off, down=down, level=level, boundary=boundary)


# ---------------------------------------------------------------- 1. beat tracking

def _penalties(ev: Evidence, cfg: DecodeConfig) -> tuple[np.ndarray, np.ndarray]:
    n = ev.n_frames
    if ev.boundary is None:
        return np.full(n, cfg.step_penalty), np.full(n, cfg.reset_penalty)
    b = np.clip(ev.boundary, cfg.boundary_floor, 1.0)
    frac = np.log(b) / math.log(cfg.boundary_floor)      # 1 at the floor, 0 at activation 1
    lo = cfg.boundary_min_penalty
    return lo + (cfg.step_penalty - lo) * frac, lo + (cfg.reset_penalty - lo) * frac


def _track(ev: Evidence, cfg: DecodeConfig):
    """Viterbi beat frames, the period at each and the transition type into each.

    The state (period ``k``, phase ``phi``) at frame ``t`` is the path whose last beat was at frame
    ``s = t - phi``, so only the beat states need a value per frame: ``V[k, s]``, the best score of a
    beat at ``s`` with period ``k``, emissions included. The per-frame phase and period terms of the
    frames after a beat are prefix sums (the cosine of a phase difference splits into products of
    cosines and sines of each phase), so a non-beat state's score is ``V[k, s] + Phi_k(s, t)`` in
    constant time. Paths may start anywhere: beats before frame 0 score 0.
    """
    n = ev.n_frames
    taus = np.arange(int(round(60000.0 / FRAME_MS / cfg.bpm_hi)), int(round(60000.0 / FRAME_MS / cfg.bpm_lo)) + 1)
    K, P = len(taus), int(taus.max())
    ar = np.arange(K)
    phi = np.arange(P)
    lo = np.ceil(cfg.reset_min_phase * taus).astype(np.int64)
    rok = (phi[None, :] >= lo[:, None]) & (phi[None, :] <= taus[:, None] - 2)     # [K, P] reset phases
    lr = np.log(taus[None, :] / taus[:, None])          # [from, to]
    G = -0.5 * (lr / cfg.sigma_cont) ** 2
    b = np.clip(ev.beat, cfg.eps, 1.0 - cfg.eps)
    # The activation is a posterior whose prior is the beat-frame rate; dividing by the prior odds makes
    # the emission a likelihood ratio, so an uninformative frame (activation at the prior) neither
    # favours nor penalises a beat, and a reset (one extra beat) is not charged for a quiet span.
    prior = math.log(cfg.beat_prior / (1.0 - cfg.beat_prior))
    bdiff = cfg.beat_weight * (np.log(b) - np.log1p(-b) - prior)
    pen_step, pen_reset = _penalties(ev, cfg)
    use_phase = ev.phase is not None and cfg.phase_weight > 0
    use_period = ev.logp is not None and cfg.period_weight > 0
    dense = use_phase or use_period
    gate = ev.beat >= cfg.reset_min_beat
    if ev.boundary is not None:
        gate |= ev.boundary > cfg.boundary_floor
    # Beat-state emission beyond bdiff, and the prefix sums of the non-beat emissions; index s + P.
    beat_em = np.zeros((n, K))
    if dense:
        CX = np.zeros((n + P, K))
        CY = np.zeros((n + P, K))
        CP = np.zeros((n + P, K))
        cosT = np.ones((n + P, K))
        sinT = np.zeros((n + P, K))
        tt = np.arange(-P, n)
        th = 2 * math.pi * np.mod(tt[:, None], taus[None, :]) / taus[None, :]
        cosT[:], sinT[:] = np.cos(th), np.sin(th)
        if use_phase:
            kap = cfg.phase_weight * np.clip(np.nan_to_num(ev.kappa), 0.0, cfg.kappa_max)
            ang = 2 * math.pi * np.nan_to_num(ev.phase)
            cp, sp = kap * np.cos(ang), kap * np.sin(ang)
            X = cp[:, None] * cosT[P:] + sp[:, None] * sinT[P:]
            Y = cp[:, None] * sinT[P:] - sp[:, None] * cosT[P:]
            CX[P:] = np.cumsum(X, axis=0)
            CY[P:] = np.cumsum(Y, axis=0)
            beat_em += cp[:, None]
            gate |= (np.cos(ang) >= math.cos(2 * math.pi * cfg.reset_phase_window)) & (kap > 0)
        if use_period:
            lt = np.log(taus * FRAME_MS / 1000.0)
            sig = np.maximum(np.nan_to_num(ev.logp_sigma, nan=1.0) if ev.logp_sigma is not None else np.ones(n),
                             cfg.period_sigma_min)
            z2 = np.minimum(((lt[None, :] - np.nan_to_num(ev.logp)[:, None]) / sig[:, None]) ** 2,
                            cfg.period_z_max ** 2)
            pterm = -0.5 * cfg.period_weight * z2          # [T, K]
            CP[P:] = np.cumsum(pterm, axis=0)
            beat_em += pterm

        def phi_sum(s_idx, t_idx, kk):
            """Phi_k(s, t) for buffer indices (s + P, t + P): emissions of frames s+1..t after a beat at s."""
            return (cosT[s_idx, kk] * (CX[t_idx, kk] - CX[s_idx, kk]) + sinT[s_idx, kk] * (CY[t_idx, kk] - CY[s_idx, kk])
                    + CP[t_idx, kk] - CP[s_idx, kk])
    # Continuing tempo: only period changes within `band` frames (beyond them the Gaussian cost
    # exceeds the flat step); the step branch covers any period.
    band = int(np.ceil(taus.max() * (math.exp(math.sqrt(2 * cfg.step_penalty) * cfg.sigma_cont) - 1.0))) + 1
    offs = np.arange(-band, band + 1)
    src_idx = ar[None, :] + offs[:, None]                     # [2B+1, K] source period index
    in_rng = (src_idx >= 0) & (src_idx < K)
    src_idx = np.clip(src_idx, 0, K - 1)
    Gb = np.where(in_rng, G[src_idx, ar[None, :]], -np.inf)   # [2B+1, K]
    V = np.zeros((n + P, K))                                  # beat-state scores at buffer index s + P
    Vf = V.reshape(-1)
    if dense:
        cosf, sinf, CXf, CYf, CPf = (x.reshape(-1) for x in (cosT, sinT, CX, CY, CP))
    ptr = np.zeros((n, K), dtype=np.int16)
    c_end = (P - taus) * K + ar                               # flat index of the beat ending at t-1, minus t*K
    rokT = rok.T                                              # [P, K]: row phi
    for t in range(n):
        tb = t + P
        ie = t * K + c_end
        end = Vf.take(ie)
        if dense:
            end = end + (cosf.take(ie) * (CX[tb - 1] - CXf.take(ie)) + sinf.take(ie) * (CY[tb - 1] - CYf.take(ie))
                         + CP[tb - 1] - CPf.take(ie))
        Mb = end[src_idx] + Gb
        jj = Mb.argmax(0)
        cont = Mb[jj, ar]
        kp = src_idx[jj, ar]
        jb = int(end.argmax())
        stepv = end[jb] - pen_step[t]
        use_step = stepv > cont
        v0 = np.where(use_step, stepv, cont)
        src = np.where(use_step, jb, kp)
        if gate[t]:
            # Row phi of these reversed blocks is the beat at s = t - 1 - phi.
            R = V[tb - P:tb][::-1]
            if dense:
                R = R + (cosT[tb - P:tb][::-1] * (CX[tb - 1] - CX[tb - P:tb][::-1])
                         + sinT[tb - P:tb][::-1] * (CY[tb - 1] - CY[tb - P:tb][::-1])
                         + CP[tb - 1] - CP[tb - P:tb][::-1])
            R = np.where(rokT, R, -np.inf)
            ri = R.argmax(0)
            rv = R[ri, ar] - pen_reset[t]
            use_reset = rv > v0
            v0 = np.where(use_reset, rv, v0)
            ptr[t] = np.where(use_reset, -(ri + 1), src)
        else:
            ptr[t] = src
        V[tb] = v0 + bdiff[t] + beat_em[t]
    # The best final state: a beat at s = n-1-phi with the period's remaining frames.
    last = n - 1 + P
    s_fin = last - np.where(phi[None, :] < taus[:, None], phi[None, :], 0)     # [K, P]
    F = V[s_fin, ar[:, None]]
    if dense:
        F = F + phi_sum(s_fin, last, ar[:, None])
    F = np.where(phi[None, :] < taus[:, None], F, -np.inf)
    k, ph = np.unravel_index(int(np.argmax(F)), F.shape)
    k, ph, t = int(k), int(ph), n - 1
    frames, kinds, periods = [], [], []
    while True:
        if ph > 0:
            if t - ph < 0:
                break
            t -= ph
            ph = 0
        frames.append(t)
        periods.append(int(taus[k]))
        if t == 0:
            kinds.append(CONTINUE)
            break
        p = int(ptr[t, k])
        if p < 0:
            kinds.append(RESET)
            ph = -p - 1
        else:
            # The flat step won exactly when the Gaussian cost from the chosen source exceeds it.
            kinds.append(STEP if G[p, k] < -pen_step[t] else CONTINUE)
            k = p
            ph = int(taus[p]) - 1
        t -= 1
    return np.array(frames[::-1], dtype=np.int64), np.array(kinds[::-1], dtype=np.int64), \
        np.array(periods[::-1], dtype=np.int64)


# ---------------------------------------------------------------- 2. beat times

def _beat_times(ev: Evidence, frames: np.ndarray, cfg: DecodeConfig) -> tuple[np.ndarray, np.ndarray]:
    n = ev.n_frames
    b = ev.beat
    f = frames.copy()
    for i, x in enumerate(frames):
        lo, hi = max(x - 1, 0), min(x + 1, n - 1)
        best = x
        for y in (lo, hi):
            if b[y] > b[best]:
                best = y
        f[i] = best
    t = f * FRAME_MS
    if ev.beat_off is not None:
        trust = b[f] >= cfg.offset_min_activation
        t = t + np.where(trust, ev.beat_off[f], 0.0)
    else:
        y0 = b[np.maximum(f - 1, 0)]
        y1 = b[f]
        y2 = b[np.minimum(f + 1, n - 1)]
        den = y0 - 2 * y1 + y2
        d = np.where(den < 0, 0.5 * (y0 - y2) / np.where(den < 0, den, -1.0), 0.0)
        t = t + FRAME_MS * np.clip(d, -0.5, 0.5)
    return t.astype(np.float64), f


def _trim(ev: Evidence, frames: np.ndarray, periods: np.ndarray, cfg: DecodeConfig) -> np.ndarray:
    strong = np.flatnonzero(ev.beat >= cfg.trim_threshold)
    if not len(strong) and ev.kappa is not None:
        strong = np.flatnonzero(ev.kappa >= 1.0)
    if not len(strong) or not len(frames):
        return np.zeros(len(frames), dtype=bool)
    f0, f1 = strong[0], strong[-1]
    return (frames >= f0 - periods) & (frames <= f1 + periods)


# ---------------------------------------------------------------- 3. regimes

def _fit(t: np.ndarray) -> tuple[float, float, float, float]:
    """(a, L, max |residual|, sum of squared residuals) of t_i = a + i L."""
    m = len(t)
    if m == 1:
        return float(t[0]), math.nan, 0.0, 0.0
    i = np.arange(m, dtype=np.float64)
    im, tm = i.mean(), t.mean()
    L = float(((i - im) * (t - tm)).sum() / ((i - im) ** 2).sum())
    a = float(tm - L * im)
    r = t - (a + L * i)
    return a, L, float(np.abs(r).max()), float((r * r).sum())


@dataclass
class _Regime:
    kind: str            # 'constant' or 'free'
    i0: int              # first canonical beat (index into the beat array)
    i1: int              # one past the last
    a: float = math.nan  # fitted time of beat i0 (constant)
    L: float = math.nan  # fitted canonical beat length (constant)
    start_kind: int = CONTINUE


def _greedy(t: np.ndarray, s: int, e: int, cfg: DecodeConfig) -> list[_Regime]:
    out: list[_Regime] = []
    i = s
    while i < e:
        best = None
        j = i + 2
        while j <= e:
            a, L, mx, _ = _fit(t[i:j])
            if mx <= cfg.fit_tol_ms and L > 0:
                best = (j, a, L)
                j += 1
            else:
                break
        if best is not None and best[0] - i >= cfg.min_constant_beats:
            out.append(_Regime('constant', i, best[0], best[1], best[2]))
            i = best[0]
        else:
            out.append(_Regime('free', i, i + 1))
            i += 1
    return out


def _refit(r: _Regime, t: np.ndarray) -> tuple[float, float, float, float]:
    return _fit(t[r.i0:r.i1])


def _absorb(regs: list[_Regime], t: np.ndarray, cfg: DecodeConfig) -> list[_Regime]:
    """Free beats on a neighbouring constant segment's grid join it, one beat at a time.

    Only the geometry decides: a free beat on the segment's grid within ``merge_phase_ms`` (and the
    refit still within ``fit_tol_ms``) joins, so a reset or step the tracker placed on a beat that lies
    on the grid is undone here.
    """
    changed = True
    while changed:
        changed = False
        for k, r in enumerate(regs):
            if r.kind != 'constant':
                continue
            if k > 0 and regs[k - 1].kind == 'free':
                i = r.i0 - 1
                if abs(t[i] - (r.a - r.L)) <= cfg.merge_phase_ms:
                    a, L, mx, _ = _fit(t[i:r.i1])
                    if mx <= cfg.fit_tol_ms:
                        regs[k - 1:k + 1] = [_Regime('constant', i, r.i1, a, L, regs[k - 1].start_kind)]
                        changed = True
                        break
            if k + 1 < len(regs) and regs[k + 1].kind == 'free':
                i = r.i1
                if abs(t[i] - (r.a + (r.i1 - r.i0) * r.L)) <= cfg.merge_phase_ms:
                    a, L, mx, _ = _fit(t[r.i0:i + 1])
                    if mx <= cfg.fit_tol_ms:
                        regs[k:k + 2] = [_Regime('constant', r.i0, i + 1, a, L, r.start_kind)]
                        changed = True
                        break
    return regs


def _refine_boundaries(regs: list[_Regime], t: np.ndarray, cfg: DecodeConfig) -> list[_Regime]:
    for k in range(len(regs) - 1):
        A, B = regs[k], regs[k + 1]
        if A.kind != 'constant' or B.kind != 'constant':
            continue
        best = None
        for s in range(max(A.i0 + cfg.min_constant_beats, A.i1 - cfg.boundary_search),
                       min(B.i1 - cfg.min_constant_beats, A.i1 + cfg.boundary_search) + 1):
            fa, fb = _fit(t[A.i0:s]), _fit(t[s:B.i1])
            if fa[2] <= cfg.fit_tol_ms and fb[2] <= cfg.fit_tol_ms:
                sse = fa[3] + fb[3]
                if best is None or sse < best[0] - 1e-9:
                    best = (sse, s, fa, fb)
        if best is not None and best[1] != A.i1:
            _, s, fa, fb = best
            regs[k] = _Regime('constant', A.i0, s, fa[0], fa[1], A.start_kind)
            regs[k + 1] = _Regime('constant', s, B.i1, fb[0], fb[1], B.start_kind)
    return regs


def _merge(regs: list[_Regime], t: np.ndarray, cfg: DecodeConfig) -> list[_Regime]:
    changed = True
    while changed:
        changed = False
        for k in range(len(regs) - 1):
            A, B = regs[k], regs[k + 1]
            if A.kind != 'constant' or B.kind != 'constant':
                continue
            if abs(A.L / B.L - 1.0) > cfg.merge_ratio:
                continue
            if abs(t[B.i0] - (A.a + (B.i0 - A.i0) * A.L)) > cfg.merge_phase_ms:
                continue
            a, L, mx, _ = _fit(t[A.i0:B.i1])
            if mx <= cfg.fit_tol_ms:
                regs[k:k + 2] = [_Regime('constant', A.i0, B.i1, a, L, A.start_kind)]
                changed = True
                break
    return regs


def _regimes(t: np.ndarray, kinds: np.ndarray, cfg: DecodeConfig) -> list[_Regime]:
    n = len(t)
    cuts = [0] + [i for i in range(1, n) if kinds[i] in (STEP, RESET)] + [n]
    regs: list[_Regime] = []
    for s, e in zip(cuts, cuts[1:]):
        part = _greedy(t, s, e, cfg)
        if part:
            part[0].start_kind = int(kinds[s]) if s > 0 else CONTINUE
        regs += part
    regs = _absorb(regs, t, cfg)
    regs = _refine_boundaries(regs, t, cfg)
    regs = _merge(regs, t, cfg)
    regs = _absorb(regs, t, cfg)
    for r in regs:
        if r.kind == 'constant':
            r.a, r.L = _refit(r, t)[:2]
    return regs


# ---------------------------------------------------------------- 4. level and notated beats

def _frame(ms) -> np.ndarray:
    return np.round(np.asarray(ms, dtype=np.float64) / FRAME_MS).astype(np.int64)


def _levels(ev: Evidence, groups: list[tuple[int, int, bool]], cfg: DecodeConfig) -> list[float]:
    """One ratio per group (frame range, whether it is a free run)."""
    if ev.level is None:
        return [1.0] * len(groups)
    p = ev.level / np.maximum(ev.level.sum(axis=1, keepdims=True), 1e-12)
    lp = np.log(p + cfg.level_eps)
    n = ev.n_frames
    means, durs = [], []
    for f0, f1, _ in groups:
        a, b = int(np.clip(f0, 0, n - 1)), int(np.clip(f1, 1, n))
        b = max(b, a + 1)
        means.append(lp[a:b].mean(axis=0))
        durs.append(b - a)
    means = np.array(means)
    song = int(np.argmax((means * np.array(durs)[:, None]).sum(axis=0)))
    out = []
    for m, (_, _, free) in zip(means, groups):
        s = m.copy()
        s[song] += cfg.level_prior
        if free:
            s[0] = -np.inf
        out.append(LEVEL_CLASSES[int(np.argmax(s))])
    return out


def _down_at(ev: Evidence, times: np.ndarray, cfg: DecodeConfig) -> np.ndarray | None:
    if ev.down is None:
        return None
    n = ev.n_frames
    f = np.clip(_frame(times), 0, n - 1)
    w = cfg.down_window
    out = np.empty(len(f))
    for i, x in enumerate(f):
        out[i] = ev.down[max(x - w, 0):min(x + w + 1, n)].max()
    return out


@dataclass
class _Seg:
    start: float
    step: float           # notated beat length
    beats: list           # notated beat times
    conf: list            # confidence per notated beat
    ratio: float
    free: bool
    regime_start: bool    # first segment of a regime (a constant segment or a free run)


def _notated(ev: Evidence, regs: list[_Regime], t: np.ndarray, conf: np.ndarray, cfg: DecodeConfig):
    canon = t.copy()
    for r in regs:
        if r.kind == 'constant':
            canon[r.i0:r.i1] = r.a + r.L * np.arange(r.i1 - r.i0)
    # Groups for the level decision: each constant segment, each run of free beats.
    groups, members = [], []
    k = 0
    while k < len(regs):
        r = regs[k]
        if r.kind == 'constant':
            j = k + 1
        else:
            j = k
            while j < len(regs) and regs[j].kind == 'free':
                j += 1
        i0, i1 = regs[k].i0, regs[j - 1].i1
        if i1 < len(canon):
            end = canon[i1]
        elif r.kind == 'constant':
            end = canon[i1 - 1] + r.L
        else:
            end = canon[i1 - 1] + (canon[i1 - 1] - canon[i1 - 2] if i1 - 2 >= 0 else 500.0)
        groups.append((int(_frame(canon[i0])), int(_frame(end)), r.kind == 'free'))
        members.append((k, j))
        k = j
    ratios = _levels(ev, groups, cfg)
    segs: list[_Seg] = []
    nb = len(canon)
    for (k, j), ratio in zip(members, ratios):
        r = regs[k]
        if r.kind == 'constant':
            i0, i1 = r.i0, r.i1
            nxt = canon[i1] if i1 < nb else None
            c = canon[i0:i1]
            cconf = conf[i0:i1]
            if ratio >= 1.0:
                q = int(round(ratio))
                step = r.L / q
                # Every canonical beat of the segment stays; the grid continues past the last one up to
                # half a notated beat before the next red line, as osu! applies a red line.
                m = (len(c) - 1) * q + 1
                if nxt is not None:
                    m = max(m, int(math.floor((nxt - 0.5 * step - r.a) / step + 1e-9)) + 1)
                jj = np.arange(m)
                bt = r.a + jj * step
                bc = cconf[np.minimum(jj // q, len(cconf) - 1)]
                segs.append(_Seg(float(r.a), float(step), list(bt), list(bc), ratio, False, True))
            else:
                parity = 0
                d = _down_at(ev, c, cfg)
                if d is not None and len(c) > 1:
                    parity = int(d[1::2].sum() > d[0::2].sum())
                step = 2 * r.L
                start = r.a + parity * r.L
                first = True
                if parity == 1:
                    segs.append(_Seg(float(r.a), float(r.L), [float(r.a)], [float(cconf[0])], 1.0, False, True))
                    first = False
                m = max((len(c) - 1 - parity) // 2 + 1, 1)
                if nxt is not None:
                    m = max(m, int(math.floor((nxt - 0.5 * step - start) / step + 1e-9)) + 1)
                bt = start + np.arange(m) * step
                idx = np.minimum(parity + 2 * np.arange(len(bt)), len(cconf) - 1)
                segs.append(_Seg(float(start), float(step), list(bt), list(cconf[idx]), ratio, False, first))
        else:
            q = int(round(ratio))
            for ii, rr in enumerate(regs[k:j]):
                i = rr.i0
                if i + 1 < nb:
                    gap = canon[i + 1] - canon[i]
                elif i > 0:
                    gap = canon[i] - canon[i - 1]
                else:
                    gap = 500.0
                step = gap / q
                bt = canon[i] + step * np.arange(q if i + 1 < nb else 1)
                segs.append(_Seg(float(canon[i]), float(step), list(bt), [float(conf[i])] * len(bt), ratio, True,
                                 ii == 0))
    return segs, canon, ratios


# ---------------------------------------------------------------- 5. bars

def _bars(ev: Evidence, beats: np.ndarray, restart: np.ndarray, cfg: DecodeConfig) -> tuple[np.ndarray, np.ndarray]:
    """(downbeat mask, meter per beat) over the notated beats."""
    n = len(beats)
    d = _down_at(ev, beats, cfg)
    if n == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int64)
    if d is None:
        down = np.zeros(n, dtype=bool)
        k = 0
        for i in range(n):
            if restart[i] or i == 0:
                k = 0
            down[i] = k % 4 == 0
            k += 1
        return down, np.full(n, 4, dtype=np.int64)
    meters = np.array(cfg.meters, dtype=np.int64)
    prior = np.array(cfg.meter_logprior, dtype=np.float64)
    M = len(meters)
    offs = np.concatenate([[0], np.cumsum(meters)[:-1]])
    S = int(meters.sum())
    m_of = np.repeat(np.arange(M), meters)
    k_of = np.concatenate([np.arange(m) for m in meters])
    first = offs
    last = offs + meters - 1
    notfirst = np.flatnonzero(k_of > 0)
    dc = np.clip(d, cfg.eps, 1.0 - cfg.eps)
    e_down = cfg.down_weight * np.log(dc)
    e_up = cfg.down_weight * np.log1p(-dc)
    C = np.where(np.eye(M, dtype=bool), 0.0, -cfg.meter_change_penalty)    # [from, to]
    v = np.where(k_of == 0, e_down[0], e_up[0]) + prior[m_of]
    back = np.zeros((n, M), dtype=np.int64)     # source state of each bar-start state
    for i in range(1, n):
        new = np.empty(S)
        new[notfirst] = v[notfirst - 1]
        endv = v[last]
        E = endv[:, None] + C
        src_e = E.argmax(0)
        best_e = E[src_e, np.arange(M)]
        src_e_state = last[src_e]
        pen = cfg.bar_restart_penalty if restart[i] else cfg.bar_restart_anywhere
        # best state of each meter, for a restart
        rmax = np.full(M, -np.inf)
        rarg = np.zeros(M, dtype=np.int64)
        for mi in range(M):
            seg = v[offs[mi]:offs[mi] + meters[mi]]
            a = int(seg.argmax())
            rmax[mi], rarg[mi] = seg[a], offs[mi] + a
        R = rmax[:, None] + C - pen
        src_r = R.argmax(0)
        best_r = R[src_r, np.arange(M)]
        use_r = best_r > best_e
        new[first] = np.where(use_r, best_r, best_e) + prior
        back[i] = np.where(use_r, rarg[src_r], src_e_state)
        new[first] += e_down[i]
        new[notfirst] += e_up[i]
        v = new
    s = int(np.argmax(v))
    states = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        states[i] = s
        if i == 0:
            break
        s = int(back[i, m_of[s]]) if k_of[s] == 0 else s - 1
    return k_of[states] == 0, meters[m_of[states]]


# ---------------------------------------------------------------- 6. the map

def decode(ev: Evidence, cfg: DecodeConfig = DecodeConfig(), *, model: str = 'n1-v0') -> tuple[MusicMap, dict]:
    """Decode evidence into a music map; returns the map and a diagnostics dict."""
    t0 = time.perf_counter()
    empty = MusicMap.from_beats(np.zeros(0), model=model)
    if ev.n_frames < 3:
        return empty, dict(n_beats=0, n_notated=0, n_segments=0, n_free=0, n_resets=0, n_steps=0, levels=[],
                           meters=[], seconds=time.perf_counter() - t0)
    frames, kinds, periods = _track(ev, cfg)
    keep = _trim(ev, frames, periods, cfg)
    frames, kinds, periods = frames[keep], kinds[keep], periods[keep]
    if len(frames) == 0:
        return empty, dict(n_beats=0, n_notated=0, n_segments=0, n_free=0, n_resets=0, n_steps=0, levels=[],
                           meters=[], seconds=time.perf_counter() - t0)
    if len(kinds):
        kinds[0] = CONTINUE
    t, f = _beat_times(ev, frames, cfg)
    order = np.argsort(t, kind='stable')
    t, f, kinds = t[order], f[order], kinds[order]
    ok = np.concatenate([[True], np.diff(t) > MERGE_MS])
    t, f, kinds = t[ok], f[ok], kinds[ok]
    conf = np.clip(ev.beat[f], 0.0, 1.0)
    regs = _regimes(t, kinds, cfg)
    segs, canon, ratios = _notated(ev, regs, t, conf, cfg)
    beats, cvals, restart, seg_of_beat = [], [], [], []
    starts = []
    for si, s in enumerate(segs):
        opened = False
        for b, c in zip(s.beats, s.conf):
            if beats and b <= beats[-1] + MERGE_MS:
                continue
            if not opened:
                starts.append(si)
            beats.append(float(b))
            cvals.append(float(c))
            restart.append(not opened and s.regime_start and si > 0)
            seg_of_beat.append(si)
            opened = True
    beats = np.array(beats)
    seg_of_beat = np.array(seg_of_beat, dtype=np.int64)
    down, meter = _bars(ev, beats, np.array(restart, dtype=bool), cfg)
    segments = []
    for si in starts:
        s = segs[si]
        idx = np.flatnonzero(seg_of_beat == si)
        vals, counts = np.unique(meter[idx], return_counts=True)
        m = int(vals[np.argmax(counts)]) if len(vals) else 4
        segments.append(Segment(float(beats[idx[0]]), float(s.step), m, bool(down[idx[0]])))
    pm = MusicMap.from_beats(beats, np.flatnonzero(down), np.array(cvals), segments=segments, model=model)
    diag = dict(n_beats=int(len(t)), n_notated=int(len(beats)),
                n_segments=int(sum(r.kind == 'constant' for r in regs)), n_free=int(sum(r.kind == 'free' for r in regs)),
                n_resets=int((kinds == RESET).sum()), n_steps=int((kinds == STEP).sum()),
                levels=[float(x) for x in ratios], meters=sorted({int(x) for x in meter}),
                segments_out=len(segments), seconds=round(time.perf_counter() - t0, 3))
    return pm, diag
