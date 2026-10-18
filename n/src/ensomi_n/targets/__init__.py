"""Dense training targets at 100 fps from a reference read by `ensomi_n.labels` (design §3.1).

`RecLabels` is the compact per-recording form, built once from a `Reference`
(`from_reference`) and stored as JSON (`to_json`/`from_json`). `dense` turns
it into frame targets for one crop:

- beat (canonical lines), half-beat (midpoints of canonical lines one beat
  apart) and downbeat (notated bar starts): 1 at the nearest frame, 0.5 at
  the frames either side; with each, the signed offset from a frame to its
  label in ms for the frames within one frame of it;
- level: the notated level over the canonical one, class 0..3 for
  1/2, 1, 2 and 4 (notated BPM over canonical BPM);
- phase: canonical beat phase in [0, 1), ``((t - line_k) / beat_k) mod 1``
  from each line, so a reset is a jump; log period ``log(beat_k / 1000)``;
- bar phase: linear between notated bar starts.

Times in a crop are on the stretched clock: a stretch ``rate`` multiplies every
label time and length by ``rate`` (design §3.4). Losses are masked outside the
mapped range (first head minus one canonical beat to last head plus one), and
beat-type losses weigh ``BREAK_WEIGHT`` in head gaps of ``BREAK_MS`` or more.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math

import numpy as np

FRAME_MS = 10.0
BREAK_MS = 4000.0
BREAK_WEIGHT = 0.3
LEVEL_CLASSES = (0.5, 1.0, 2.0, 4.0)
OFFSET_RANGE_MS = 15.0       # offsets are trained within 1.5 frames of a label
_HALF_TOL = 0.25


@dataclass
class RecLabels:
    key: str
    lines: np.ndarray          # canonical line times, ms, all of the timing (with its margins)
    line_beat: np.ndarray      # canonical beat length at each line, ms
    line_level: np.ndarray     # level class at each line (index into LEVEL_CLASSES)
    bars: np.ndarray           # notated bar starts, ms
    heads: np.ndarray          # head times, ms (union over the set's difficulties)
    lo_ms: float               # first head
    hi_ms: float               # last head

    @classmethod
    def from_reference(cls, ref, key: str | None = None) -> 'RecLabels':
        timing = ref.timing
        lines = timing.canonical()
        folds = timing.folds()
        level = np.clip(1 - folds[lines.src_lo], 0, len(LEVEL_CLASSES) - 1) if len(lines) else np.zeros(0)
        heads = ref.heads if ref.heads is not None else np.zeros(0)
        heads = heads[ref.scorable(heads)] if len(heads) else heads
        return cls(key if key is not None else ref.key, np.asarray(lines.t, dtype=np.float64),
                   np.asarray(lines.beat_ms, dtype=np.float64), np.asarray(level, dtype=np.int64),
                   np.asarray(timing.bar_starts(), dtype=np.float64), np.asarray(heads, dtype=np.float64),
                   float(ref.lo_ms), float(ref.hi_ms))

    def to_json(self) -> dict:
        out = {}
        for k, v in asdict(self).items():
            out[k] = v.tolist() if isinstance(v, np.ndarray) else v
        return out

    @classmethod
    def from_json(cls, d: dict) -> 'RecLabels':
        return cls(d['key'], np.asarray(d['lines'], dtype=np.float64), np.asarray(d['line_beat'], dtype=np.float64),
                   np.asarray(d['line_level'], dtype=np.int64), np.asarray(d['bars'], dtype=np.float64),
                   np.asarray(d['heads'], dtype=np.float64), float(d['lo_ms']), float(d['hi_ms']))

    @property
    def mapped(self) -> tuple[float, float]:
        """(start, end) of the mapped range: one canonical beat beyond the first and last head."""
        if not len(self.lines):
            return self.lo_ms, self.hi_ms
        i = int(np.clip(np.searchsorted(self.lines, self.lo_ms), 0, len(self.lines) - 1))
        j = int(np.clip(np.searchsorted(self.lines, self.hi_ms), 0, len(self.lines) - 1))
        return self.lo_ms - float(self.line_beat[i]), self.hi_ms + float(self.line_beat[j])

    def half_lines(self) -> np.ndarray:
        """Midpoints of consecutive canonical lines that are one canonical beat apart (within 25 %)."""
        if len(self.lines) < 2:
            return np.zeros(0)
        gap = np.diff(self.lines)
        ok = np.abs(gap / self.line_beat[:-1] - 1.0) <= _HALF_TOL
        return (self.lines[:-1] + gap / 2.0)[ok]

    def breaks(self) -> np.ndarray:
        """[K, 2] head gaps of BREAK_MS or more."""
        h = self.heads
        if len(h) < 2:
            return np.zeros((0, 2))
        g = np.diff(h)
        k = np.flatnonzero(g >= BREAK_MS)
        return np.column_stack([h[k], h[k + 1]])


def _impulses(times: np.ndarray, start_ms: float, n: int, frame_ms: float):
    """Widened targets and offsets: (y [n], offset_ms [n], offset_mask [n])."""
    y = np.zeros(n, dtype=np.float32)
    if not len(times):
        return y, np.zeros(n, dtype=np.float32), np.zeros(n, dtype=bool)
    pos = np.sort((np.asarray(times, dtype=np.float64) - start_ms) / frame_ms)
    pos = pos[(pos > -2.0) & (pos < n + 1.0)]
    if not len(pos):
        return y, np.zeros(n, dtype=np.float32), np.zeros(n, dtype=bool)
    centre = np.round(pos).astype(np.int64)
    for d, val in ((-1, 0.5), (1, 0.5), (0, 1.0)):
        j = centre + d
        ok = (j >= 0) & (j < n)
        np.maximum.at(y, j[ok], val)
    # Each frame's nearest label: the offset is label minus frame centre, kept within 1.5 frames.
    j = np.arange(n, dtype=np.float64)
    i = np.clip(np.searchsorted(pos, j), 1, len(pos) - 1) if len(pos) > 1 else np.zeros(n, dtype=np.int64)
    if len(pos) > 1:
        left, right = pos[i - 1] - j, pos[i] - j
        d = np.where(np.abs(left) <= np.abs(right), left, right)
    else:
        d = pos[0] - j
    mask = np.abs(d) <= 1.5
    off = np.where(mask, d * frame_ms, 0.0).astype(np.float32)
    return y, off, mask


def dense(lab: RecLabels, start_ms: float, n: int, rate: float = 1.0, frame_ms: float = FRAME_MS) -> dict:
    """Frame targets for ``n`` frames starting at ``start_ms`` on the clock stretched by ``rate``.

    Frame ``j`` sits at ``start_ms + j * frame_ms``; a label at source time
    ``t`` sits at ``t * rate``.
    """
    u = start_ms + frame_ms * np.arange(n)
    lines = lab.lines * rate
    beat_len = lab.line_beat * rate
    a, b = lab.mapped
    mask = (u >= a * rate) & (u <= b * rate)
    weight = mask.astype(np.float32)
    for s, e in lab.breaks():
        weight[(u > s * rate) & (u < e * rate)] *= BREAK_WEIGHT
    out = dict(mask=mask, weight=weight)
    for name, times in (('beat', lines), ('half', lab.half_lines() * rate), ('down', lab.bars * rate)):
        y, off, om = _impulses(times, start_ms, n, frame_ms)
        out[name], out[name + '_off'], out[name + '_offmask'] = y, off, om & mask
    # Phase, period and level between consecutive lines.
    k = np.searchsorted(lines, u, side='right') - 1
    valid = (k >= 0) & (k < len(lines) - 1) & mask
    kk = np.clip(k, 0, max(len(lines) - 1, 0))
    if len(lines):
        L = beat_len[kk]
        gap_ok = np.zeros(n, dtype=bool)
        if len(lines) > 1:
            nxt = np.clip(kk + 1, 0, len(lines) - 1)
            gap_ok = (lines[nxt] - lines[kk]) <= 2.0 * L      # a gap of two beats or more (unmapped) is masked
        valid &= gap_ok
        phase = np.mod((u - lines[kk]) / L, 1.0)
        logp = np.log(L / 1000.0)
        level = np.where(valid, lab.line_level[kk], -1)
    else:
        phase = np.zeros(n)
        logp = np.zeros(n)
        level = np.full(n, -1)
    out['phase'] = phase.astype(np.float32)
    out['logp'] = logp.astype(np.float32)
    out['phase_mask'] = valid
    out['level'] = level.astype(np.int64)
    bars = lab.bars * rate
    m = np.searchsorted(bars, u, side='right') - 1
    bvalid = (m >= 0) & (m < len(bars) - 1) & mask
    if len(bars) > 1:
        mm = np.clip(m, 0, len(bars) - 2)
        span = bars[mm + 1] - bars[mm]
        bvalid &= span <= 16 * np.maximum(beat_len[kk] if len(lines) else span, 1.0)
        bar_phase = np.clip((u - bars[mm]) / span, 0.0, 1.0 - 1e-9)
    else:
        bar_phase = np.zeros(n)
        bvalid[:] = False
    out['bar_phase'] = bar_phase.astype(np.float32)
    out['bar_mask'] = bvalid
    return out


def label_rate(y: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(bool)
    return float(y[m].mean()) if m.any() else math.nan


def dump(labels: list[RecLabels], path) -> None:
    with open(path, 'w') as handle:
        json.dump([l.to_json() for l in labels], handle)
