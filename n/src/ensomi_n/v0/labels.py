"""N1 v0 labels: the rung-1 targets plus segment boundaries, from one reference set.

`V0Labels` extends `ensomi_n.targets.RecLabels` (canonical lines, their level, notated bars,
heads, the mapped range) with the starts of the reference's musical boundaries: red lines of
kind ``tempo`` or ``reset`` in `ensomi_n.eval.strata.classify_lines` (the census's kinds), and
the spans of detour runs. `dense_v0` adds to `ensomi_n.targets.dense`:

- ``bound``: 1 within one frame of a boundary, 0.5 out to three frames (design §2.1: focal BCE
  with ±3 frames' tolerance);
- detour runs weigh 0.5 in the beat-type losses (design §3.1: the grid in force continues
  inside them, unverified by the notes there).

Times are on the clock stretched by ``rate``, as in `targets.dense`.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..targets import FRAME_MS, RecLabels, dense

BOUND_KINDS = ('tempo', 'reset')
DETOUR_WEIGHT = 0.5


@dataclass
class V0Labels(RecLabels):
    bounds: np.ndarray = None          # boundary times, ms (tempo and reset lines inside the scored range)
    bound_kind: np.ndarray = None      # 0 tempo, 1 reset
    detours: np.ndarray = None         # [K, 2] detour-run spans, ms

    @classmethod
    def from_reference(cls, ref, key: str | None = None) -> 'V0Labels':
        from ..eval.strata import classify_lines
        base = RecLabels.from_reference(ref, key)
        lines = classify_lines(ref.segments or (), ref.heads, ref.lo_ms, ref.hi_ms)
        bounds, kinds, detours = [], [], []
        run = None
        for ln in lines:
            if ln.kind == 'detour':
                run = (ln.start_ms if run is None else run[0], ln.end_ms)
                continue
            if run is not None:
                detours.append((run[0], ln.start_ms))
                run = None
            if ln.kind in BOUND_KINDS and ref.lo_ms - 1.0 <= ln.start_ms <= ref.hi_ms + 1.0:
                bounds.append(ln.start_ms)
                kinds.append(BOUND_KINDS.index(ln.kind))
        if run is not None:
            detours.append((run[0], min(run[1], ref.hi_ms)))
        return cls(base.key, base.lines, base.line_beat, base.line_level, base.bars, base.heads, base.lo_ms,
                   base.hi_ms, np.asarray(bounds, dtype=np.float64), np.asarray(kinds, dtype=np.int64),
                   np.asarray(detours, dtype=np.float64).reshape(-1, 2))

    def to_json(self) -> dict:
        out = super().to_json()
        out['bounds'] = self.bounds.tolist()
        out['bound_kind'] = self.bound_kind.tolist()
        out['detours'] = self.detours.tolist()
        return out

    @classmethod
    def from_json(cls, d: dict) -> 'V0Labels':
        b = RecLabels.from_json(d)
        return cls(b.key, b.lines, b.line_beat, b.line_level, b.bars, b.heads, b.lo_ms, b.hi_ms,
                   np.asarray(d.get('bounds', []), dtype=np.float64),
                   np.asarray(d.get('bound_kind', []), dtype=np.int64),
                   np.asarray(d.get('detours', []), dtype=np.float64).reshape(-1, 2))


def _bound_target(times: np.ndarray, start_ms: float, n: int, frame_ms: float) -> np.ndarray:
    y = np.zeros(n, dtype=np.float32)
    if not len(times):
        return y
    pos = (np.asarray(times, dtype=np.float64) - start_ms) / frame_ms
    pos = pos[(pos > -4.0) & (pos < n + 3.0)]
    for c in np.round(pos).astype(np.int64):
        for d in range(-3, 4):
            j = c + d
            if 0 <= j < n:
                y[j] = max(y[j], 1.0 if abs(d) <= 1 else 0.5)
    return y


def dense_v0(lab: V0Labels, start_ms: float, n: int, rate: float = 1.0, frame_ms: float = FRAME_MS) -> dict:
    out = dense(lab, start_ms, n, rate=rate, frame_ms=frame_ms)
    out['bound'] = _bound_target(lab.bounds * rate, start_ms, n, frame_ms)
    if len(lab.detours):
        u = start_ms + frame_ms * np.arange(n)
        for a, b in lab.detours:
            out['weight'][(u >= a * rate) & (u < b * rate)] *= DETOUR_WEIGHT
    return out


def literal_flag(ref) -> bool:
    """The coordinator's interim rule: a plausible segment in range notated at 160 BPM or more, or a
    redundant same-tempo red line (a line on the previous plausible line's grid at its beat length)."""
    from ..timing import MERGE_MS
    segs = list(ref.segments or ())
    lo, hi = ref.lo_ms, ref.hi_ms
    prev = None
    for k, s in enumerate(segs):
        end = segs[k + 1].start_ms if k + 1 < len(segs) else math.inf
        if not s.plausible:
            continue
        in_range = end > lo and s.start_ms <= hi
        if in_range and s.bpm >= 160.0 - 1e-9:
            return True
        if in_range and prev is not None and abs(s.beat_ms / prev.beat_ms - 1.0) <= 1e-6:
            n_b = (s.start_ms - prev.start_ms) / prev.beat_ms
            if abs(n_b - round(n_b)) * prev.beat_ms <= MERGE_MS:
                return True
        prev = s
    return False
