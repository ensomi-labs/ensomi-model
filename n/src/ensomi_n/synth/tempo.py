"""Musical time to clock time: the tempo map, and edits of the rendered timeline.

A `TempoMap` maps a position on the beat axis (a float, in notated beats
from the clip's first bar) to milliseconds on the source timeline. It is
built from pieces, each continuing where the previous one ended:

- ``const``: a fixed beat length;
- ``ramp``: BPM moving linearly *in time* from ``bpm0`` to ``bpm1``, over a
  whole number of beats. The beat phase is the integral of the tempo,
  ``phi(x) = x (B + a x)`` with ``B = bpm0 / 60000`` beats per ms and
  ``a = (bpm1 - bpm0) / (2 T 60000)``; the time of a position is its root,
  ``x = 2 db / (B + sqrt(B^2 + 4 a db))``, and the ramp lasts
  ``T = 120000 n / (bpm0 + bpm1)`` ms for ``n`` beats;
- ``beats``: explicit beat lengths, one per beat, with time linear inside
  each beat (rubato);
- ``gap``: time passes and the beat axis does not (a span without pulse).
  A position at a gap's start maps to the time after it.

An `EditList` rewrites the rendered timeline: copies of source spans, a
slice repeated (stutter), a tape stop (the read rate falls linearly from 1
to 0) and silence. `EditList.map_onsets` gives where each source onset is
heard; `EditList.map_grid` gives where a source beat line lands, with the
grid holding through a repeat (a stutter replaces audio in place) and
dropped inside a tape stop or a removed span.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Piece:
    kind: str           # const, ramp, beats, gap
    b0: float
    b1: float
    t0: float
    t1: float
    beat_ms: float = math.nan            # const
    bpm0: float = math.nan               # ramp
    bpm1: float = math.nan
    times: tuple = ()                    # beats: times of b0, b0 + 1, ..., b1


class TempoMap:
    """Positions on the beat axis to source milliseconds (module docstring)."""

    def __init__(self, start_ms: float):
        self.pieces: list[Piece] = []
        self.start_ms = float(start_ms)

    @property
    def b_end(self) -> float:
        return self.pieces[-1].b1 if self.pieces else 0.0

    @property
    def t_end(self) -> float:
        return self.pieces[-1].t1 if self.pieces else self.start_ms

    def const(self, beats: float, beat_ms: float) -> 'TempoMap':
        b0, t0 = self.b_end, self.t_end
        self.pieces.append(Piece('const', b0, b0 + beats, t0, t0 + beats * beat_ms, beat_ms=float(beat_ms)))
        return self

    def ramp(self, beats: int, bpm0: float, bpm1: float) -> 'TempoMap':
        b0, t0 = self.b_end, self.t_end
        T = 120000.0 * beats / (bpm0 + bpm1)
        self.pieces.append(Piece('ramp', b0, b0 + beats, t0, t0 + T, bpm0=float(bpm0), bpm1=float(bpm1)))
        return self

    def per_beat(self, lengths_ms) -> 'TempoMap':
        b0, t0 = self.b_end, self.t_end
        times = t0 + np.concatenate([[0.0], np.cumsum(np.asarray(lengths_ms, dtype=np.float64))])
        self.pieces.append(Piece('beats', b0, b0 + len(lengths_ms), t0, float(times[-1]),
                                 times=tuple(float(x) for x in times)))
        return self

    def gap(self, ms: float) -> 'TempoMap':
        b0, t0 = self.b_end, self.t_end
        self.pieces.append(Piece('gap', b0, b0, t0, t0 + ms))
        return self

    def _piece(self, b: float, end: bool = False) -> Piece:
        if end:
            for p in self.pieces:
                if p.kind != 'gap' and p.b0 < b <= p.b1 + 1e-12:
                    return p
        for p in self.pieces:
            if p.kind != 'gap' and p.b0 <= b < p.b1:
                return p
        last = [p for p in self.pieces if p.kind != 'gap'][-1]
        if abs(b - last.b1) < 1e-9:
            return last
        raise ValueError(f'Beat position {b} lies outside the tempo map [0, {self.b_end}]')

    def time(self, b: float, end: bool = False) -> float:
        """Source ms of beat-axis position ``b``.

        At a gap's position the time after the gap is returned, as a bar
        starting there begins after it; ``end=True`` returns the time before
        the gap instead, where the bar ending there ends.
        """
        p = self._piece(float(b), end)
        db = float(b) - p.b0
        if p.kind == 'const':
            return p.t0 + db * p.beat_ms
        if p.kind == 'ramp':
            return p.t0 + ramp_time(db, p.bpm0, p.bpm1, p.t1 - p.t0)
        i = max(min(int(math.floor(db)), len(p.times) - 2), 0)
        return p.times[i] + (db - i) * (p.times[i + 1] - p.times[i])

    def step(self, b: float) -> float:
        """The notated beat length (ms) of the beat starting at ``b``: the piece's beat length, or for a
        ramp and for per-beat lengths the time to the next beat (the beat length a red line there needs)."""
        p = self._piece(float(b))
        if p.kind == 'const':
            return p.beat_ms
        if float(b) + 1.0 <= p.b1 + 1e-9:
            return self.time(float(b) + 1.0, end=True) - self.time(float(b))
        if p.kind == 'ramp':
            return 60000.0 / p.bpm1
        return p.times[-1] - p.times[-2]

    def times(self, bs) -> np.ndarray:
        return np.array([self.time(b) for b in bs], dtype=np.float64)

    def bpm_at(self, t_ms: float) -> float:
        """The notated tempo in force at a source time (NaN inside a gap)."""
        for p in self.pieces:
            if p.t0 <= t_ms < p.t1:
                if p.kind == 'const':
                    return 60000.0 / p.beat_ms
                if p.kind == 'ramp':
                    return p.bpm0 + (p.bpm1 - p.bpm0) * (t_ms - p.t0) / (p.t1 - p.t0)
                if p.kind == 'beats':
                    i = int(np.searchsorted(p.times, t_ms, side='right')) - 1
                    return 60000.0 / (p.times[i + 1] - p.times[i])
                return math.nan
        return math.nan


def ramp_time(db: float, bpm0: float, bpm1: float, T: float) -> float:
    """Milliseconds after a ramp's start at which ``db`` beats have passed (module docstring)."""
    B = bpm0 / 60000.0
    a = (bpm1 - bpm0) / (2.0 * T * 60000.0)
    if db <= 0:
        return 0.0
    return 2.0 * db / (B + math.sqrt(B * B + 4.0 * a * db))


# Edits of the rendered timeline.

@dataclass(frozen=True)
class Copy:
    a: float            # source ms
    b: float

    @property
    def out_ms(self) -> float:
        return self.b - self.a

    @property
    def src_span(self) -> tuple[float, float]:
        return self.a, self.b


@dataclass(frozen=True)
class Repeat:
    """``n`` copies of the source slice ``[a, a + slice_ms)``, each faded out over its last 3 ms to
    silence 1 ms before its end. The grid holds: the source span ``[a, a + n slice_ms)`` is replaced."""
    a: float
    slice_ms: float
    n: int

    @property
    def out_ms(self) -> float:
        return self.slice_ms * self.n

    @property
    def src_span(self) -> tuple[float, float]:
        return self.a, self.a + self.slice_ms * self.n


@dataclass(frozen=True)
class Tape:
    """A tape stop: the read rate falls linearly from 1 to 0 over ``out`` ms, reading ``out / 2`` ms of source."""
    a: float
    out: float

    @property
    def out_ms(self) -> float:
        return self.out

    @property
    def src_span(self) -> tuple[float, float]:
        return self.a, self.a + self.out / 2.0

    def tau(self, s: float) -> float:
        """Output ms after the piece's start at which source time ``s`` is read."""
        return self.out * (1.0 - math.sqrt(max(0.0, 1.0 - 2.0 * (s - self.a) / self.out)))

    def rate(self, s: float) -> float:
        return 1.0 - self.tau(s) / self.out


@dataclass(frozen=True)
class Silence:
    out: float

    @property
    def out_ms(self) -> float:
        return self.out

    @property
    def src_span(self):
        return None


#: Each repeat of a slice is silent over its last ``EDGE_MS`` and fades out over the ``FADE_MS`` before.
EDGE_MS = 1.0
FADE_MS = 3.0


class EditList:
    """Pieces laid end to end on the output timeline, from output time 0."""

    def __init__(self, pieces):
        self.pieces = list(pieces)
        self.starts = np.concatenate([[0.0], np.cumsum([p.out_ms for p in self.pieces])])[:-1]

    @property
    def out_ms(self) -> float:
        return float(sum(p.out_ms for p in self.pieces))

    def map_onsets(self, t_src: float) -> list[tuple[float, bool, float]]:
        """Where a source onset is heard: ``(output ms, effect, read rate)`` per hearing.

        ``effect`` marks onsets the edit made: repeats after the first, and onsets inside a tape stop.
        """
        out = []
        for start, p in zip(self.starts, self.pieces):
            if isinstance(p, Copy) and p.a <= t_src < p.b:
                out.append((start + t_src - p.a, False, 1.0))
            elif isinstance(p, Repeat) and p.a <= t_src < p.a + p.slice_ms - EDGE_MS:
                for i in range(p.n):
                    out.append((start + i * p.slice_ms + t_src - p.a, i > 0, 1.0))
            elif isinstance(p, Tape) and p.a <= t_src < p.src_span[1]:
                out.append((start + p.tau(t_src), True, p.rate(t_src)))
        return out

    def map_grid(self, t_src: float) -> float | None:
        """Where a source beat line lands; None where the edit removes the pulse."""
        for start, p in zip(self.starts, self.pieces):
            if isinstance(p, (Copy, Repeat)):
                a, b = p.src_span
                if a <= t_src < b:
                    return float(start + t_src - a)
        return None

    def map_grid_all(self, t_src: float) -> list[float]:
        """Every place a source beat line is heard: a span copied twice (an inserted repeat) gives the beat
        twice, once where the fragment starts and once where the music goes on; tape stops drop it."""
        out = []
        for start, p in zip(self.starts, self.pieces):
            if isinstance(p, (Copy, Repeat)):
                a, b = p.src_span
                if a <= t_src < b:
                    out.append(float(start + t_src - a))
        return out

    def map_span(self, a: float, b: float) -> list[tuple[float, float]]:
        """The output spans a source span ``[a, b)`` of pulse occupies (copies and repeats only)."""
        out = []
        for start, p in zip(self.starts, self.pieces):
            if isinstance(p, (Copy, Repeat)):
                pa, pb = p.src_span
                lo, hi = max(a, pa), min(b, pb)
                if hi > lo:
                    out.append((float(start + lo - pa), float(start + hi - pa)))
        merged: list[list[float]] = []
        for lo, hi in out:
            if merged and abs(lo - merged[-1][1]) < 1e-9:
                merged[-1][1] = hi
            else:
                merged.append([lo, hi])
        return [(lo, hi) for lo, hi in merged]

    def apply(self, x: np.ndarray, sample_rate: int) -> np.ndarray:
        """The output samples of one source layer."""
        sr = sample_rate
        n_out = int(round(self.out_ms * sr / 1000.0))
        y = np.zeros(n_out)
        src_t = np.arange(len(x)) / sr * 1000.0
        for start, p in zip(self.starts, self.pieces):
            m0 = int(math.ceil(start * sr / 1000.0 - 1e-7))
            m1 = min(int(math.ceil((start + p.out_ms) * sr / 1000.0 - 1e-7)), n_out)
            if m1 <= m0:
                continue
            tau = np.arange(m0, m1) / sr * 1000.0 - start          # output ms into the piece
            if isinstance(p, Silence):
                continue
            if isinstance(p, Copy):
                pos = p.a + tau
                env = np.ones(len(tau))
            elif isinstance(p, Repeat):
                within = np.mod(tau, p.slice_ms)
                pos = p.a + within
                env = np.clip((p.slice_ms - EDGE_MS - within) / FADE_MS, 0.0, 1.0)
            else:   # Tape
                pos = p.a + tau - tau * tau / (2.0 * p.out)
                env = np.ones(len(tau))
            y[m0:m1] = env * _read(x, pos * sr / 1000.0)
        return y


def _read(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """``x`` at fractional sample positions, by linear interpolation; positions within 1e-6 of a
    sample read that sample exactly, so plain copies stay bit-exact."""
    r = np.round(idx)
    exact = np.abs(idx - r) < 1e-6
    i0 = np.floor(idx).astype(np.int64)
    frac = idx - i0
    i0 = np.where(exact, r.astype(np.int64), i0)
    frac = np.where(exact, 0.0, frac)
    ok0 = (i0 >= 0) & (i0 < len(x))
    ok1 = (i0 + 1 >= 0) & (i0 + 1 < len(x))
    v0 = np.where(ok0, x[np.clip(i0, 0, len(x) - 1)], 0.0)
    v1 = np.where(ok1, x[np.clip(i0 + 1, 0, len(x) - 1)], 0.0)
    return v0 * (1.0 - frac) + v1 * frac
