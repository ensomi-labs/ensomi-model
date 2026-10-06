"""Chart properties under one declared measurement semantics nu (``NU``).

Every R2 consumer of LN share or section difficulty reads this module: the training draw
values, the star relabel, the frame counters' test, request validation and every readout.
A property is computed from an object list and a scope in song milliseconds, never from a
run state or a request, so a readout depends only on chart content and nu.

Scope convention: ``scope_contains(a, b, T, t)`` is a <= t < b, and a <= t <= T when b = T
(a whole-song scope contains T). An object belongs to the scope of its head (start) time,
whatever its end, including a hold that is not yet closed (``end is None``).

Objects are ``PObject(start, end, lane, hold)``; ``as_pobjects`` converts the R2, osu_core
and difficulty object types.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .common import ContractError

__all__ = ['NU', 'NU_HASH', 'NU_ID', 'PROPERTIES_SOURCE_SHA256', 'PObject', 'as_pobjects', 'prefix_objects',
           'scope_contains', 'ln_share', 'ln_share_rows', 'row_range', 'difficulty', 'difficulty_trimmed',
           'tile', 'deviation', 'MIN_STAR_MS', 'HORIZON_MS', 'TILING_VERSION', 'CALCULATOR']

MIN_STAR_MS = 30_000.0
HORIZON_MS = 240_000.0
TILING_VERSION = 'tiled-star-v1 (240s horizon, 1ms seam, 30s min, head ownership, untrimmed tails)'
CALCULATOR = 'compute_mania_star_rating_20241007(objects, 4, clock_rate=1.0)'

NU = dict(
    version='r2-nu-v1',
    scope='[a, b) in song ms; [a, T] when b = T',
    ln_share=dict(
        id='ln_share/r2-nu-v1',
        heads='objects whose start time lies in the scope; one count per object',
        assignment='a hold belongs to the scope of its head, whatever its end, including a hold not yet closed',
        value='LN heads / heads',
        undefined='the scope owns no head',
        deviation='readout - target', resolution='1/n for n heads', mirror='invariant'),
    difficulty=dict(
        id='difficulty/r2-nu-v1',
        evaluator=CALCULATOR, tiling=TILING_VERSION, key_count=4, clock_rate=1.0,
        context="the scope's own objects only (heads in the scope, untrimmed tails); the surrounding chart "
                'is not scored',
        undefined='scope shorter than 30 s; scope owns no object; a seam cut leaves a nonpositive hold; '
                  'a hold headed in the scope is not closed',
        deviation='readout - target in star', resolution="the calculator's float output",
        mirror='measured by tests/r2/test_properties.py::test_t_m_mirror (see DEVIATIONS.md)'),
)
NU_HASH = hashlib.sha256(json.dumps(NU, sort_keys=True).encode()).hexdigest()
NU_ID = {'ln_share': NU['ln_share']['id'], 'difficulty': NU['difficulty']['id']}
PROPERTIES_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class PObject:
    start: float
    end: float | None      # None: a hold whose close is not materialised yet
    lane: int
    hold: bool


def as_pobjects(objects) -> list[PObject]:
    out = []
    for o in objects:
        if isinstance(o, PObject):
            out.append(o)
        elif hasattr(o, 'start_time_ms'):
            hold = getattr(o.kind, 'name', str(o.kind)).upper() == 'HOLD'
            out.append(PObject(float(o.start_time_ms), float(o.end_time_ms), int(o.lane), hold))
        else:  # osu_core.difficulty.RawHitObject
            out.append(PObject(float(o.start_time), float(o.end_time), int(o.column), o.end_time > o.start_time))
    return out


def prefix_objects(head_ms, actions, gap) -> list[PObject]:
    """Objects of a committed decision prefix; holds still open after the prefix have ``end=None``."""
    K = len(head_ms)
    held = [None] * 4
    out = []
    for k in range(len(actions)):
        t = float(head_ms[k]) if k < K else None
        for lane in range(4):
            c = int(actions[k][lane])
            if held[lane] is not None:
                if c == 0:
                    continue
                end = t if c == 1 else float(gap[k][lane])
                out.append(PObject(held[lane], end, lane, True))
                held[lane] = None
                if c == 3:
                    out.append(PObject(t, t, lane, False))
                elif c == 4:
                    held[lane] = t
            elif c == 1:
                out.append(PObject(t, t, lane, False))
            elif c == 2:
                held[lane] = t
    out += [PObject(s, None, lane, True) for lane, s in enumerate(held) if s is not None]
    return sorted(out, key=lambda o: (o.start, o.lane))


def scope_contains(a: float, b: float, T: float, t):
    t = np.asarray(t, dtype=np.float64)
    return (t >= a) & ((t < b) | ((b >= T) & (t <= T)))


def _owned(objects, a, b, T):
    return [o for o in as_pobjects(objects) if bool(scope_contains(a, b, T, o.start))]


def ln_share(objects, a: float, b: float, T: float) -> float | None:
    """LNShare_nu over [a, b) (or [a, T]); None when the scope owns no head."""
    owned = _owned(objects, a, b, T)
    if not owned:
        return None
    return sum(o.hold for o in owned) / len(owned)


def row_range(head_ms, a: float, b: float, T: float) -> tuple[int, int]:
    """Head-row indices [lo, hi) whose times lie in the scope."""
    lo = int(np.searchsorted(head_ms, a, side='left'))
    hi = len(head_ms) if b >= T else int(np.searchsorted(head_ms, b, side='left'))
    return lo, max(lo, hi)


def ln_share_rows(head_counts, ln_counts, head_ms, a: float, b: float, T: float) -> float | None:
    """The row-level form of LNShare_nu from per-head-row object counts (labels, draws, counters)."""
    lo, hi = row_range(head_ms, a, b, T)
    n = int(np.asarray(head_counts)[lo:hi].sum())
    return None if n == 0 else float(np.asarray(ln_counts)[lo:hi].sum()) / n


def deviation(readout, target):
    return None if readout is None else float(readout) - float(target)


# ---- difficulty ---------------------------------------------------------------------------------

def tile(owned: list[PObject], a: float, length: float):
    """Tiled object list of the scope (heads translated by -a, period ``length``, to 240 s) and seam stats.

    An LN of a non-final copy still held at the next copy's first head in its lane ends 1 ms before
    that head. Returns (None, info) when a cut leaves a nonpositive hold.
    """
    from ..osu_core.difficulty import RawHitObject
    n = math.ceil(HORIZON_MS / length)
    first = {}
    for o in owned:
        first[o.lane] = min(first.get(o.lane, math.inf), o.start - a)
    tiled, cuts = [], 0
    for r in range(n):
        for o in owned:
            h, e = o.start - a + r * length, o.end - a + r * length
            if o.end > o.start and r < n - 1:
                v = (r + 1) * length + first[o.lane]
                if e >= v:
                    e = v - 1.0
                    cuts += 1
                    if e <= h:
                        return None, dict(owned=len(owned), copies=n, cuts=cuts, invalid='nonpositive_cut')
            tiled.append(RawHitObject(start_time=h, end_time=e, column=o.lane))
    return tiled, dict(owned=len(owned), copies=n, cuts=cuts)


def _difficulty_of(owned, a, b):
    from ..osu_core.difficulty import compute_mania_star_rating_20241007
    if b - a < MIN_STAR_MS:
        return None, dict(invalid='scope_under_30s')
    if not owned:
        return None, dict(owned=0, invalid='empty_scope')
    if any(o.end is None for o in owned):
        return None, dict(owned=len(owned), invalid='open_hold')
    tiled, info = tile(owned, a, b - a)
    if tiled is None:
        return None, info
    return float(compute_mania_star_rating_20241007(tiled, 4, clock_rate=1.0)), info


def difficulty(objects, a: float, b: float, T: float):
    """Difficulty_nu over the scope: (value or None, info). The tiling period is b - a."""
    return _difficulty_of(_owned(objects, a, b, T), a, b)


def difficulty_trimmed(objects, a: float, b: float, T: float):
    """Diagnostic variant (never a target): every scope-headed hold closed at or after b is cut at
    max(head + 1 ms, b - 1 ms)."""
    owned = []
    for o in _owned(objects, a, b, T):
        if o.hold and o.end is not None and o.end >= b:
            o = PObject(o.start, max(o.start + 1.0, b - 1.0), o.lane, True)
        owned.append(o)
    return _difficulty_of(owned, a, b)


def require_kind(kind: str):
    if kind not in NU_ID:
        raise ContractError(f'Unknown property {kind!r}; R2 supports {sorted(NU_ID)}')
