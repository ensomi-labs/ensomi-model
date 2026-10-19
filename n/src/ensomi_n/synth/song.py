"""A clip as a symbolic program: layers, bars on a beat axis, notes, and what the ground truth needs.

A `Song` holds:

- ``tm``, the `TempoMap` from beat-axis positions to source milliseconds;
- ``bars``: each bar's start on the beat axis, its length in notated beats
  (a bar of 3.5 is 7/8 notated in quarters, cut short), its meter (the
  notated beats it shows) and its grouping;
- ``notes``: each on a layer, at a beat-axis position (an exact `Fraction`,
  so each beat's subdivision is known) or, in free time, at a time;
- ``hard_stops``: source times at which every monophonic note is cut
  (1 ms before), so an edit or a re-entry starts from silence;
- ``edits``: an optional `EditList` applied to the rendered layers;
- ``spans``: the stratum spans (source ms) the stratum's measures run on;
  none means the whole pulse;
- ``resets``: bars that start a new red line whatever the notation rule
  says (after a tape stop).

The grooves at the end place one bar of a style. Patterns are in beats of
the bar's notated unit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction as F
import math

import numpy as np

from .tempo import TempoMap
from .voices import draw_params

ROLES = ('beat', 'counter', 'texture', 'free')


@dataclass
class Note:
    layer: str
    t: float                    # source ms
    vel: float
    pitch: tuple = ()
    pos: F | None = None        # beat-axis position; None in free time
    dur: float | None = None    # explicit length in ms (pads)


@dataclass
class Bar:
    b0: F
    length: F
    meter: int
    grouping: tuple = ()
    beats: tuple = ()           # offsets of its notated beats

    @property
    def b1(self) -> F:
        return self.b0 + self.length


@dataclass
class Layer:
    name: str
    voice: str
    role: str
    gain: float
    params: dict
    period: str = ''            # the layer's own pulse, when it is not the beat (counter layers)


class Song:
    def __init__(self, setting: str, seed: int, rng: np.random.Generator, sample_rate: int,
                 target_s: float = 60.0):
        self.setting, self.seed, self.rng, self.sr = setting, seed, rng, sample_rate
        self.target_ms = 1000.0 * target_s
        start = float(rng.uniform(500.0, 1000.0))
        self.tm = TempoMap(start)
        self.bars: list[Bar] = []
        self.notes: list[Note] = []
        self.layers: dict[str, Layer] = {}
        self.hard_stops: list[float] = []
        self.edits = None
        self.spans: list[tuple[float, float]] | None = None
        self.resets: set[int] = set()
        self.params: dict = {}
        self.tail_ms = 1500.0
        self.room: dict | None = None
        self.gap_kinds: list[str] = []      # one per tempo-map gap, in order: 'beatless' or 'free'
        self.effects: list[tuple[float, float, str]] = []   # source spans of an edit's effect (stutter)

    # Building

    def layer(self, name: str, voice: str, role: str = 'beat', gain: float = 1.0, family: str = '',
              period: str = '', **overrides) -> Layer:
        if role not in ROLES:
            raise ValueError(f'Unknown role {role!r}')
        params = draw_params(voice, self.rng, self.sr, family)
        params.update(overrides)
        lay = Layer(name, voice, role, float(gain), params, period)
        self.layers[name] = lay
        return lay

    def bar(self, length, meter: int | None = None, grouping: tuple = (), beats: tuple | None = None) -> Bar:
        length = F(length)
        b0 = self.bars[-1].b1 if self.bars else F(0)
        if float(b0 + length) > self.tm.b_end + 1e-9:
            raise ValueError(f'Bar [{b0}, {b0 + length}) runs past the tempo map ({self.tm.b_end})')
        meter = int(math.ceil(length)) if meter is None else meter
        beats = tuple(F(j) for j in range(int(math.ceil(length)))) if beats is None else tuple(F(x) for x in beats)
        bar = Bar(b0, length, meter, tuple(grouping), beats)
        self.bars.append(bar)
        return bar

    def time(self, b, end: bool = False) -> float:
        return self.tm.time(float(b), end)

    def note(self, layer: str, b, vel: float, pitch=(), dur: float | None = None) -> None:
        if layer not in self.layers:
            raise KeyError(layer)
        b = F(b)
        self.notes.append(Note(layer, self.time(b), float(vel), tuple(pitch), b, dur))

    def note_ms(self, layer: str, t: float, vel: float, pitch=(), dur: float | None = None) -> None:
        self.notes.append(Note(layer, float(t), float(vel), tuple(pitch), None, dur))

    def vel(self, v: float, spread: float = 0.06) -> float:
        """A velocity with a little per-note variation (timing is never varied)."""
        return float(np.clip(v + self.rng.uniform(-spread, spread), 0.05, 1.0))

    def span(self, a_ms: float, b_ms: float) -> None:
        self.spans = (self.spans or []) + [(float(a_ms), float(b_ms))]

    def span_bars(self, i: int, j: int) -> None:
        """A stratum span over bars ``i`` to ``j - 1``."""
        j = min(j, len(self.bars))
        self.span(self.time(self.bars[i].b0), self.time(self.bars[j - 1].b1, end=True))

    def hard_stop(self, t_ms: float) -> None:
        """Cut every monophonic note ``EDGE_MS`` before ``t_ms``, so that audio after it starts from silence."""
        self.hard_stops.append(float(t_ms))

    # What the ground truth reads

    @property
    def end_ms(self) -> float:
        last = self.time(self.bars[-1].b1, end=True)
        free = max((n.t + (n.dur or 0.0) for n in self.notes), default=last)
        return max(last, free) + self.tail_ms

    def gap_before(self, i: int) -> bool:
        """Whether a gap (no pulse) of the tempo map sits right before bar ``i``."""
        if i == 0:
            return False
        b = float(self.bars[i].b0)
        return any(p.kind == 'gap' and abs(p.b0 - b) < 1e-9 for p in self.tm.pieces)

    def beat_positions(self) -> list[tuple[F, int, int]]:
        """Every notated beat: ``(position, bar index, beat index in bar)``."""
        return [(bar.b0 + off, i, j) for i, bar in enumerate(self.bars) for j, off in enumerate(bar.beats)]

    def pulse_spans(self) -> list[tuple[float, float]]:
        """Source spans with a pulse: runs of bars not broken by a gap."""
        spans: list[list[float]] = []
        for i, bar in enumerate(self.bars):
            a, b = self.time(bar.b0), self.time(bar.b1, end=True)
            if spans and not self.gap_before(i) and abs(spans[-1][1] - a) < 1e-6:
                spans[-1][1] = b
            else:
                spans.append([a, b])
        return [(a, b) for a, b in spans]


# Pitch material.

def midi(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


_MAJOR = {0: 'M', 2: 'm', 4: 'm', 5: 'M', 7: 'M', 9: 'm'}
_MINOR = {0: 'm', 3: 'M', 5: 'm', 7: 'm', 8: 'M', 10: 'M'}
_PROGRESSIONS = {False: ((0, 9, 5, 7), (0, 7, 9, 5), (0, 5, 7, 5), (9, 5, 0, 7)),
                 True: ((0, 8, 3, 10), (0, 5, 8, 7), (0, 10, 8, 10), (0, 3, 10, 8))}


class Harmony:
    """A key and a four-chord progression, one chord per bar."""

    def __init__(self, rng: np.random.Generator):
        self.root = int(rng.integers(36, 44))
        self.minor = bool(rng.integers(0, 2))
        options = _PROGRESSIONS[self.minor]
        self.prog = options[int(rng.integers(len(options)))]

    def _degree(self, i: int) -> int:
        return self.prog[i % len(self.prog)]

    def bass(self, i: int, fifth: bool = False) -> tuple:
        return (midi(self.root + self._degree(i) + (7 if fifth else 0)),)

    def chord_midi(self, i: int) -> tuple:
        d = self._degree(i)
        quality = (_MINOR if self.minor else _MAJOR)[d]
        r = self.root + 24 + d
        return (r, r + (3 if quality == 'm' else 4), r + 7)

    def chord(self, i: int) -> tuple:
        return tuple(midi(m) for m in self.chord_midi(i))

    def tone(self, i: int, k: int, octave: int = 1) -> tuple:
        """The k-th chord tone of bar ``i``'s chord, rising through octaves."""
        c = self.chord_midi(i)
        return (midi(c[k % 3] + 12 * (octave + (k // 3) % 2)),)


# Grooves: one bar of a style. Positions are Fractions of the bar's notated beat.

def h(x) -> F:
    return F(x)


def put(s: Song, bar: Bar, layer: str, pos, vel: float, pitch=()) -> None:
    """A note at ``pos`` beats into ``bar``; skipped when the bar (cut short) ends first."""
    pos = F(pos)
    if pos < bar.length:
        s.note(layer, bar.b0 + pos, vel, pitch)


def groove_44(s: Song, bar: Bar, i: int, style: str, harm: Harmony, crash_every: int = 8) -> None:
    """One 4/4 bar. ``edm``: four on the floor; ``band``: kick 1 and 3, backbeat; ``click``: a click
    track with a plucked arpeggio and bass."""
    b0 = bar.b0
    if style == 'edm':
        for q in range(4):
            put(s, bar, 'kick', q, s.vel(1.0 if q == 0 else 0.9))
        for q in (1, 3):
            put(s, bar, 'snare', q, s.vel(0.85))
        for q in range(4):
            put(s, bar, 'hat', q + h('1/2'), s.vel(0.7))
            if i % 2:
                put(s, bar, 'hat', q + h('1/4'), s.vel(0.3))
                put(s, bar, 'hat', q + h('3/4'), s.vel(0.3))
        for q in range(4):
            put(s, bar, 'bass', q + h('1/2'), s.vel(0.8), harm.bass(i))
        put(s, bar, 'chord', 0, s.vel(0.7), harm.chord(i))
        put(s, bar, 'chord', h('5/2'), s.vel(0.55), harm.chord(i))
    elif style == 'band':
        put(s, bar, 'kick', 0, s.vel(1.0))
        put(s, bar, 'kick', 2, s.vel(0.9))
        if i % 4 == 3:
            put(s, bar, 'kick', h('5/2'), s.vel(0.7))
        for q in (1, 3):
            put(s, bar, 'snare', q, s.vel(0.85))
        for e in range(8):
            put(s, bar, 'hat', h(e) / 2, s.vel(0.55 if e % 2 == 0 else 0.4))
        for pos, fifth in ((0, False), (h('3/2'), True), (2, False), (3, True)):
            put(s, bar, 'bass', pos, s.vel(0.8), harm.bass(i, fifth))
        put(s, bar, 'chord', 0, s.vel(0.6), harm.chord(i))
        put(s, bar, 'chord', 2, s.vel(0.5), harm.chord(i))
    elif style == 'click':
        for q in range(4):
            put(s, bar, 'click', q, s.vel(1.0 if q == 0 else 0.75), (2400.0 if q == 0 else 1600.0,))
        for e in range(8):
            put(s, bar, 'pluck', h(e) / 2, s.vel(0.6 if e % 2 == 0 else 0.45), harm.tone(i, e))
        put(s, bar, 'bass', 0, s.vel(0.8), harm.bass(i))
        put(s, bar, 'bass', 2, s.vel(0.7), harm.bass(i, True))
    else:
        raise ValueError(style)
    if crash_every and i % crash_every == 0 and 'crash' in s.layers:
        put(s, bar, 'crash', 0, s.vel(0.7))


def groove_grouped(s: Song, bar: Bar, i: int, harm: Harmony, unit: F = F(1), crash_every: int = 8) -> None:
    """One bar of an additive meter: ``bar.grouping`` in eighths (2+2+3 for 7/8). ``unit`` is one
    eighth in notated beats: 1 when the eighth is notated, 1/2 when the quarter is."""
    b0 = bar.b0
    starts = np.cumsum((0,) + bar.grouping[:-1])
    n = sum(bar.grouping)
    for g, st in enumerate(starts):
        put(s, bar, 'kick', unit * int(st), s.vel(1.0 if g == 0 else 0.8))
        put(s, bar, 'bass', unit * int(st), s.vel(0.85 if g == 0 else 0.7), harm.bass(i, fifth=g == len(starts) - 1))
    if len(starts) > 1:
        put(s, bar, 'snare', unit * int(starts[-1]), s.vel(0.85))
    for e in range(n):
        put(s, bar, 'hat', unit * e, s.vel(0.65 if e in starts else 0.45))
    put(s, bar, 'chord', 0, s.vel(0.65), harm.chord(i))
    if crash_every and i % crash_every == 0 and 'crash' in s.layers:
        put(s, bar, 'crash', 0, s.vel(0.7))


def drum_kit(s: Song, family: str, crash: bool = True, gains: dict | None = None) -> None:
    """The usual layers: kick, snare, hat, bass, chord, and a crash."""
    g = dict(kick=1.0, snare=0.7, hat=0.35, bass=0.55, chord=0.4, crash=0.25)
    g.update(gains or {})
    s.layer('kick', 'kick', gain=g['kick'], family=family)
    s.layer('snare', 'snare', gain=g['snare'], family=family)
    s.layer('hat', 'hat', gain=g['hat'], family=family)
    s.layer('bass', 'bass', gain=g['bass'], family=family)
    s.layer('chord', 'chord', gain=g['chord'], family=family)
    if crash:
        s.layer('crash', 'crash', role='texture', gain=g['crash'], family=family)


def bars_for(s: Song, bar_ms: float, minimum: int = 4) -> int:
    """How many bars of ``bar_ms`` fill the clip's target length."""
    return max(minimum, int(round(s.target_ms / bar_ms)))
