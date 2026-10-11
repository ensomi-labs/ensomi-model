"""Rule L: which decisions read which effective-track interval (plan v4 section 4.4).

Decision k < K produces the gap releases of the lanes it closes, strictly inside
(t_{k-1}, t_k), then the row at t_k; EOS (k = K) produces only the closes inside
(t_{K-1}, T). Decision k reads interval I, in every role and through every path, if and
only if every time it can produce lies in I:

    k < K:  t_k in I, and either no lane is held entering k or min C_k >= a;
    k = K:  a lane is held, min C_K >= a, and (b = T or max C_K < b).

Under rule L both locality conditions of the formulation hold exactly: no decision before
a scope's onset decision reads the request, and every decision that can produce a time at
or after b reads nothing of it, including the closes of holds headed inside the scope.
V_k depends on the skeleton, the scopes and the occupancy entering k only, so it is known
before the decision in teacher forcing and in sampling alike.
"""
from __future__ import annotations

import numpy as np

from .features import Chart, Interval, contains

RULE_L_VERSION = 'rule-L-v1'


def reads(chart: Chart, iv: Interval, k: int, held) -> bool:
    K, T = chart.K, chart.song_ms
    any_held = bool(np.asarray(held).any())
    if k < K:
        if not bool(contains(iv, chart.head_ms[k], T)):
            return False
        return not any_held or float(chart.candidates(k).times[0]) >= iv.a
    if not any_held:
        return False
    c = chart.candidates(K).times
    return float(c[0]) >= iv.a and (iv.b >= T or float(c[-1]) < iv.b)


def visible(chart: Chart, track, k: int, rule_l: bool = True) -> tuple:
    """V_k: the intervals of ``track`` decision k reads. ``rule_l=False`` (test power checks and
    v1 reproduction only) returns the whole track, leaving only the per-time activity test."""
    if not track:
        return ()
    if not rule_l:
        return tuple(track)
    held = chart.derived().held[k]
    return tuple(iv for iv in track if reads(chart, iv, k, held))


def visible_kinds(chart: Chart, track, ks, rule_l: bool = True) -> np.ndarray:
    """[m,2] bool: decision ks[i] reads some interval of kind 0 / 1."""
    out = np.zeros((len(ks), 2), dtype=bool)
    for i, k in enumerate(ks):
        for iv in visible(chart, track, int(k), rule_l):
            out[i, iv.kind] = True
    return out


def scope_decisions(chart: Chart, iv: Interval):
    """Decisions whose row time lies in the scope (EOS when b = T), the onset and exit decisions."""
    T, K = chart.song_ms, chart.K
    times = chart.times(np.arange(K + 1))
    inside = np.flatnonzero(contains(iv, times, T))
    onset = int(np.searchsorted(times, iv.a, side='left'))
    exit_ = None if iv.b >= T else int(np.searchsorted(times, iv.b, side='left'))
    return inside, onset, exit_


def masked(chart: Chart, iv: Interval):
    """What rule L leaves ungoverned in the scope of ``iv`` on a chart with known decisions.

    ``onset_masked``: the onset decision lies in the scope but reads nothing (a hold is open and
    a candidate precedes a). ``in_scope_outside_v``: head decisions with row time in the scope
    that do not read ``iv``. ``exit_closes``: gap releases inside [a, b) made by the exit
    decision. ``first_visible``: the first decision that reads ``iv`` (None if none does).
    """
    d = chart.derived()
    inside, onset, exit_ = scope_decisions(chart, iv)
    n = min(chart.n, chart.K + 1)
    outside, first = [], None
    for k in inside:
        k = int(k)
        if k >= n:
            break
        if reads(chart, iv, k, d.held[k]):
            first = k if first is None else first
        elif k < chart.K:
            outside.append(k)
    if first is None and chart.n > chart.K and reads(chart, iv, chart.K, d.held[chart.K]):
        first = chart.K
    exit_closes = []
    if exit_ is not None and exit_ < n:
        for lane in range(4):
            u = d.release[exit_, lane]
            if d.held[exit_, lane] and np.isfinite(u) and chart.actions[exit_, lane] in (2, 3, 4) and iv.a <= u < iv.b:
                exit_closes.append(dict(lane=lane, time=float(u)))
    onset_masked = bool(onset < n and onset < chart.K and len(inside) and int(inside[0]) == onset
                        and not reads(chart, iv, onset, d.held[onset]))
    return dict(onset=onset, exit=exit_, first_visible=first, onset_masked=onset_masked,
                onset_open_hold=bool(onset < n and d.held[min(onset, n - 1)].any()),
                in_scope_head_decisions=int(sum(1 for k in inside if k < chart.K)),
                in_scope_outside_v=outside, exit_closes=exit_closes)
