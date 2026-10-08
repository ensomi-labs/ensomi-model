"""Synthetic DPO labellers and pair building, test fixtures for ``train_dpo``.

Synthetic labellers exist to test the DPO mechanism with a known preferred direction; they are
not a preference source. ``LNShareLabeller``'s fixed target ignores the state's condition track
and would reward ignoring the condition. Nothing under ``src/`` imports this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from ensomi_model.r2.train_dpo import Pair, StartState, sample_branch, validate_pair


# ---- synthetic labellers ------------------------------------------------------------------------

def branch_events(state: StartState, branch, lo_ms=1.0, hi_ms=40.0) -> dict:
    """Counts over the branch's decisions: heads, LN heads, releases, gap releases and releases
    that come lo..hi ms before a head in another lane (heads after the branch are unknown)."""
    actions, gap = branch
    chart = state.chart(actions, gap)
    d = chart.derived()
    a, b = state.start, state.start + len(actions)
    rel = d.release[a:b]
    rows = chart.times(np.arange(b))
    attack = d.attack[:b]
    near = 0
    for i, lane in zip(*np.nonzero(~np.isnan(rel))):
        others = rows[np.delete(attack, lane, axis=1).any(1)]
        dt = others - rel[i, lane]
        near += int(((dt >= lo_ms) & (dt <= hi_ms)).any())
    return dict(heads=int(d.attack[a:b].sum()), ln_heads=int(d.ln_head[a:b].sum()),
                releases=int((~np.isnan(rel)).sum()), gap_releases=int((~np.isnan(rel) & ~d.row_release[a:b]).sum()),
                near_head_releases=near, decisions=b - a, eos=b == state.K + 1)


@dataclass(frozen=True)
class LNShareLabeller:
    """Utility -|LN share - target| over the branch's heads; statistic is the LN share."""
    target: float
    alpha: float = 20.0
    name: str = 'ln_share'

    def statistic(self, state, branch) -> float:
        e = branch_events(state, branch)
        return e['ln_heads'] / e['heads'] if e['heads'] else math.nan

    def utility(self, state, branch) -> float:
        return -abs(self.statistic(state, branch) - self.target)

    def label(self, u_a, u_b) -> float:
        return 0.5 if math.isnan(u_a) or math.isnan(u_b) else 1.0 / (1.0 + math.exp(-self.alpha * (u_a - u_b)))


@dataclass(frozen=True)
class NearHeadLabeller:
    """Utility -(number of releases 1..40 ms before another lane's head); statistic is that count
    per release."""
    lo_ms: float = 1.0
    hi_ms: float = 40.0
    alpha: float = 1.0
    name: str = 'near_head'

    def statistic(self, state, branch) -> float:
        e = branch_events(state, branch, self.lo_ms, self.hi_ms)
        return e['near_head_releases'] / e['releases'] if e['releases'] else math.nan

    def utility(self, state, branch) -> float:
        return -float(branch_events(state, branch, self.lo_ms, self.hi_ms)['near_head_releases'])

    label = LNShareLabeller.label


# ---- pair builder -------------------------------------------------------------------------------

def same_branch(a, b) -> bool:
    return np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1], equal_nan=True)


def build_pair(model, state: StartState, labeller, horizon=64, seeds=(0, 1), min_margin=0.05):
    """Two samples from ``state`` labelled by ``labeller``; None when identical or |q - 0.5| < min_margin."""
    a = sample_branch(model, state, horizon, seeds[0])
    b = sample_branch(model, state, horizon, seeds[1])
    if same_branch(a, b):
        return None
    u_a, u_b = labeller.utility(state, a), labeller.utility(state, b)
    q = labeller.label(u_a, u_b)
    if abs(q - 0.5) < min_margin:
        return None
    if q < 0.5:
        a, b, u_a, u_b, q, seeds = b, a, u_b, u_a, 1.0 - q, tuple(reversed(seeds))
    pair = Pair(state, a, b, q, dict(source='synthetic', labeller=labeller.name, labeller_config=asdict(labeller),
                                     utility_plus=u_a, utility_minus=u_b, seeds=[int(s) for s in seeds],
                                     horizon=int(horizon)))
    return validate_pair(pair)


def build_pairs(model, states, labeller, horizon=64, per_state=1, seed=0, min_margin=0.05, tries=4):
    """Up to ``per_state`` pairs per state, at most ``tries * per_state`` sibling draws each."""
    rng = np.random.default_rng(seed)
    out = []
    for state in states:
        made = 0
        for _ in range(tries * per_state):
            if made == per_state:
                break
            seeds = tuple(int(s) for s in rng.integers(0, 2 ** 31 - 1, size=2))
            pair = build_pair(model, state, labeller, horizon, seeds, min_margin)
            if pair is not None:
                out.append(pair)
                made += 1
    return out
