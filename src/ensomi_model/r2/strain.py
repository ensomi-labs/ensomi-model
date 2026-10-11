"""Frozen ras-v1 head workload and sampling-time head-mask tilt.

Four lane strains and one overall strain decay in seconds between head rows.
Each row pays its full interaction charge immediately; accumulated workload
never decays. Scopes only select charges, so neither actual nor reference
strain resets at a request boundary. The reference places one tap per row,
cycling lanes 0, 2, 1, 3 from the chart's first row. Changing this definition or
any constant below requires a new metric version.

``StrainTrace(head_ms, song_ms).extend(actions, gap)`` replays a complete chart
or an append-only prefix. Row-index workload queries are O(1); time scopes
use binary search. Building both traces takes O(K), including exact replay.
No model scores, star labels or style baselines enter this metric.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import numpy as np
import torch

from .common import ACTIONS, ContractError
from .state import R2State, advance, decision_at

RAS_VERSION: Final = 'ras-v1'
TAU_I: Final = -1.0 / math.log(0.125)
TAU_G: Final = -1.0 / math.log(0.30)
WEIGHT_I: Final = 1.0
WEIGHT_G: Final = 0.25
REFERENCE_LANES: Final = (0, 2, 1, 3)
# Index i means integer mask i + 1; bit l denotes lane l.
HEAD_MASKS = ((np.arange(1, 16)[:, None] >> np.arange(4)) & 1).astype(bool)
HEAD_MASKS.flags.writeable = False


def heads_from_codes(codes, held, *, eos=False):
    """Boolean [...,4] heads from held-before state; EOS contributes none."""
    codes = np.asarray(codes)
    if eos:
        return np.zeros_like(codes, dtype=bool)
    return np.where(held, (codes == 3) | (codes == 4), (codes == 1) | (codes == 2))


def action_head_masks(held, *, eos=False):
    """Integer masks [625] in ACTIONS order; zero denotes no heads."""
    return heads_from_codes(ACTIONS, held, eos=eos).astype(np.int64) @ (1 << np.arange(4))


def head_mask_log_probs(logp: torch.Tensor, held):
    """Log probability masses [15] in HEAD_MASKS order from masked [625] logp.

    Input must be the normalized action log probabilities for a head decision,
    with illegal actions at -inf. Output preserves its device and dtype. EOS
    has no nonempty mask and should not be passed here.
    """
    masks = torch.as_tensor(action_head_masks(held), device=logp.device)
    return torch.stack([torch.logsumexp(logp[masks == m], 0) for m in range(1, 16)])


@dataclass(frozen=True)
class StrainState:
    """Strains just after ``time_ms``; initial state is rest at chart time zero."""

    lane: tuple = (0.0, 0.0, 0.0, 0.0)
    overall: float = 0.0
    time_ms: float = 0.0

    def decayed(self, time_ms: float):
        dt = (float(time_ms) - self.time_ms) / 1000.0
        if not math.isfinite(dt) or dt < 0:
            raise ContractError('Strain times must be finite and nondecreasing')
        return StrainState(tuple(x * math.exp(-dt / TAU_I) for x in self.lane),
                           self.overall * math.exp(-dt / TAU_G), float(time_ms))

    def _cost(self, heads):
        q = np.asarray(heads, dtype=np.float64)
        m = q.sum(-1)
        return (WEIGHT_I * TAU_I * (q @ np.asarray(self.lane) + m / 2.0)
                + WEIGHT_G * TAU_G * (self.overall * m + m * m / 2.0))

    def mask_costs(self, time_ms: float):
        """Float64 [15] full charges at time_ms, without committing or rollout."""
        return self.decayed(time_ms)._cost(HEAD_MASKS)

    def append(self, time_ms: float, heads):
        """Return (new state, full row charge) for a boolean four-lane mask."""
        q = np.asarray(heads)
        if q.shape != (4,) or not np.isin(q, (0, 1)).all():
            raise ContractError('A strain head mask has four binary entries')
        before = self.decayed(time_ms)
        charge = float(before._cost(q))
        return StrainState(tuple(np.asarray(before.lane) + q), before.overall + int(q.sum()),
                           float(time_ms)), charge


@dataclass(frozen=True)
class ScopeWorkload:
    workload: float
    reference: float
    head_rows: int
    duration_sec: float

    @property
    def ratio(self):
        """sqrt(W / WH), or None when the scope contains no head rows."""
        return math.sqrt(self.workload / self.reference) if self.head_rows else None

    @property
    def intensity(self):
        """W per second, or None when the scope contains no head rows."""
        return self.workload / self.duration_sec if self.head_rows else None


class StrainTrace:
    """Append-only replay and workload prefixes over one fixed full head skeleton.

    ``extend`` accepts all committed decisions, including the source prefix;
    previously supplied decisions must not change. Its total replay work over
    successive extensions is O(K). ``state`` and ``replay`` describe only the
    committed prefix. Reference strain always starts at the chart's first row,
    even for queries over later scopes. EOS is replayed but never charged.
    """

    def __init__(self, head_ms, song_ms: float):
        self.head_ms = np.array(head_ms, dtype=np.float64, copy=True)
        self.song_ms = float(song_ms)
        if (self.head_ms.ndim != 1 or not np.isfinite(self.head_ms).all()
                or np.any(self.head_ms < 0) or np.any(np.diff(self.head_ms) <= 0)
                or not math.isfinite(self.song_ms) or self.song_ms <= 0
                or (len(self.head_ms) and self.head_ms[-1] >= self.song_ms)):
            raise ContractError('Strain trace needs increasing nonnegative heads before the song end')
        self.head_ms.flags.writeable = False
        self.state = StrainState()
        self.replay = R2State()
        self.charges = []
        self.workload_prefix = [0.0]
        self.reference_prefix = [0.0]
        reference = StrainState()
        for k, t in enumerate(self.head_ms):
            q = np.arange(4) == REFERENCE_LANES[k % 4]
            reference, charge = reference.append(t, q)
            self.reference_prefix.append(self.reference_prefix[-1] + charge)

    def extend(self, actions, gap):
        """Consume new rows from full committed arrays; return self.

        Source and generated decisions use the same exact replay. Invalid
        decisions raise ContractError before their strain charge is committed.
        Arrays must be append-only and use this trace's skeleton and song end.
        """
        if (np.shape(actions) != (len(actions), 4) or np.shape(gap) != np.shape(actions)
                or not self.replay.k <= len(actions) <= len(self.head_ms) + 1):
            raise ContractError('Strain trace requires an append-only [n,4] decision prefix')
        for k in range(self.replay.k, len(actions)):
            decision = decision_at(actions, gap, k)
            replay, _ = advance(self.replay, decision, self.head_ms, self.song_ms)
            if k < len(self.head_ms):
                q = heads_from_codes(decision.codes, self.replay.held)
                self.state, charge = self.state.append(self.head_ms[k], q)
                self.charges.append(charge)
                self.workload_prefix.append(self.workload_prefix[-1] + charge)
            self.replay = replay
        return self

    def row_bounds(self, a: float, b: float):
        """Row indices for [a,b), with finite 0 <= a <= b <= song_ms."""
        if not (math.isfinite(a) and math.isfinite(b) and 0 <= a <= b <= self.song_ms):
            raise ContractError('Strain scope must satisfy 0 <= a <= b <= song_ms')
        return tuple(int(i) for i in np.searchsorted(self.head_ms, (a, b), side='left'))

    def workload_rows(self, start: int, stop: int):
        """O(1) W for committed head rows [start,stop)."""
        if not 0 <= start <= stop < len(self.workload_prefix):
            raise ContractError('Workload query extends beyond committed head rows')
        return self.workload_prefix[stop] - self.workload_prefix[start]

    def reference_rows(self, start: int, stop: int):
        """O(1) WH for full-skeleton head rows [start,stop)."""
        if not 0 <= start <= stop < len(self.reference_prefix):
            raise ContractError('Reference query extends beyond the head skeleton')
        return self.reference_prefix[stop] - self.reference_prefix[start]

    def scope(self, a: float, b: float):
        """Workload, reference, ratio and intensity for a fully committed scope."""
        start, stop = self.row_bounds(a, b)
        return ScopeWorkload(self.workload_rows(start, stop), self.reference_rows(start, stop),
                             stop - start, (b - a) / 1000.0)

    def remaining_budget(self, r_target: float, a: float, b: float):
        """Exact r_target**2 * WH([a,b)) - W_committed([a,b)); may be negative."""
        if not math.isfinite(r_target) or r_target < 0:
            raise ContractError('Target strain ratio must be finite and nonnegative')
        start, stop = self.row_bounds(a, b)
        n = len(self.charges)
        committed = self.workload_rows(min(start, n), min(stop, n))
        return r_target ** 2 * self.reference_rows(start, stop) - committed


class FixedEtaBias:
    """Callable for ``continue_chart(..., head_mask_bias=tilt)``.

    Construct with the full ``(head_ms, song_ms, a, b, eta)``. Each call returns
    eta * dW(mask) / sS for the 15 masks, where sS = WH([a,b)) / K_scope.
    Outside [a,b), including EOS, it returns zeros. Empty scopes have no tilt.
    Positive eta rewards workload; negative eta penalizes it. All actions with
    the same head mask share a bias, preserving P(action | mask).

    ``trace`` consumes the source prefix on the first invocation and subsequent
    generated decisions on the next invocation. For early stopping, call
    ``tilt.trace.extend(actions, gap)`` on returned arrays to include the last
    sampled decision. Use a new instance for each generation trajectory.
    """

    def __init__(self, head_ms, song_ms: float, a: float, b: float, eta: float):
        self.trace = StrainTrace(head_ms, song_ms)
        start, stop = self.trace.row_bounds(a, b)
        if not math.isfinite(eta):
            raise ContractError('Strain tilt eta must be finite')
        self.a, self.b, self.eta = float(a), float(b), float(eta)
        self.scale = self.trace.reference_rows(start, stop) / (stop - start) if stop > start else None

    def __call__(self, k, chart, held, logp):
        self.trace.extend(chart.actions, chart.gap)
        if k != self.trace.replay.k or not np.array_equal(held, self.trace.replay.held):
            raise ContractError('Strain tilt and sampler replay states disagree')
        if k == chart.K or not self.a <= chart.time(k) < self.b or self.eta == 0.0:
            return np.zeros(15, dtype=np.float64)
        return self.eta * self.trace.state.mask_costs(chart.time(k)) / self.scale
