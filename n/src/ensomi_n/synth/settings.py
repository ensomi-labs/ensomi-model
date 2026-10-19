"""The settings of synthetic suite v0: one program per setting, filled from a seed.

Each setting builds a `Song`: its tempo map, bars, layers, notes and, where
the stratum needs them, edits, gaps and stratum spans. ``SETTINGS`` lists
them with their stratum and musical meaning. A setting draws everything it
varies (tempo, timbres, patterns, edit places) from the song's generator, so
(setting, seed) fixes the clip.

Notation (the ground truth's notated beat) is part of each setting: 4/4 is
notated in quarters at a tempo in [80, 160) BPM, additive meters in eighths
(meter 5 or 7 at twice the quarter tempo) except ``meter_7_8_reset``, which
is notated in quarters with a red line at every bar, as mappers often time
7/8. The canonical beat lines are the same under either notation; the level
and bar terms are not.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
import math
from typing import Callable

import numpy as np

from .render import CLIP_S
from .song import Harmony, Song, drum_kit, groove_44, groove_grouped, midi, put
from .tempo import Copy, EditList, Repeat, Silence, Tape

CLIP_MS = 1000.0 * CLIP_S
TAIL_MS = 1500.0


@dataclass(frozen=True)
class Setting:
    name: str
    stratum: str
    meaning: str
    build: Callable[[Song], None]


# Helpers.

def _end(s: Song) -> float:
    """Source time by which the pulse material should end."""
    return CLIP_MS - TAIL_MS


def _fs(s: Song, t_ms: float) -> float:
    """An edit point on the sample at or before ``t_ms``: a piece then never reads part of a note that
    starts exactly at its end, and the ground truth and the audio agree on which side a note falls."""
    return math.floor(t_ms * s.sr / 1000.0 + 1e-9) * 1000.0 / s.sr


def _room(s: Song, wet=(0.10, 0.20), rt=(0.35, 0.8)) -> None:
    r = s.rng
    s.room = dict(seed=int(r.integers(2 ** 31)), rt60_s=float(r.uniform(*rt)), predelay_ms=float(r.uniform(8, 15)),
                  lowpass=float(r.uniform(0.3, 0.6)), wet=float(r.uniform(*wet)))


def _int_bpm(s: Song, lo=96, hi=144) -> float:
    return float(s.rng.integers(lo, hi + 1))


def _fit(s: Song, bar_beats: float, beat_ms: float, until: float | None = None) -> int:
    until = _end(s) if until is None else until
    return max(0, int(math.floor((until - s.tm.t_end) / (float(bar_beats) * beat_ms) + 1e-9)))


def _const(s: Song, n: int, length, beat_ms: float, meter=None, grouping=(), beats=None):
    if n <= 0:
        return []
    s.tm.const(float(F(length)) * n, beat_ms)
    return [s.bar(length, meter, grouping, beats) for _ in range(n)]


def _kit(s: Song, family: str, gains=None, crash=True):
    drum_kit(s, family, crash=crash, gains=gains)


def _fill_44(s: Song, bars, start_index: int, style: str, harm: Harmony):
    for k, bar in enumerate(bars):
        groove_44(s, bar, start_index + k, style, harm)


# Grooves beyond song.py's.

def groove_lead(s: Song, bar, i: int, harm: Harmony, divs) -> None:
    """Kick, backbeat, quarter hats, and a plucked lead playing ``divs[q]`` notes in beat q (0 for none)."""
    put(s, bar, 'kick', 0, s.vel(1.0))
    put(s, bar, 'kick', 2, s.vel(0.9))
    put(s, bar, 'snare', 1, s.vel(0.8))
    put(s, bar, 'snare', 3, s.vel(0.8))
    put(s, bar, 'bass', 0, s.vel(0.8), harm.bass(i))
    put(s, bar, 'bass', 2, s.vel(0.7), harm.bass(i, True))
    for q in range(4):
        put(s, bar, 'hat', q, s.vel(0.5))
        d = divs[q]
        for k in range(d):
            put(s, bar, 'lead', q + F(k, d), s.vel(0.75 if k == 0 else 0.5), harm.tone(i, k, 2))
    put(s, bar, 'chord', 0, s.vel(0.5), harm.chord(i))


def groove_mixed_hats(s: Song, bar, i: int, harm: Harmony, hat_div: int, lead_div: int, beats=(0, 1, 2, 3)) -> None:
    """Both layers inside one beat: hats at 1/hat_div and the lead at 1/lead_div in the beats listed."""
    put(s, bar, 'kick', 0, s.vel(1.0))
    put(s, bar, 'kick', 2, s.vel(0.9))
    put(s, bar, 'snare', 1, s.vel(0.8))
    put(s, bar, 'snare', 3, s.vel(0.8))
    put(s, bar, 'bass', 0, s.vel(0.8), harm.bass(i))
    put(s, bar, 'chord', 0, s.vel(0.5), harm.chord(i))
    for q in range(4):
        if q in beats:
            for k in range(hat_div):
                put(s, bar, 'hat', q + F(k, hat_div), s.vel(0.55 if k == 0 else 0.35))
            for k in range(lead_div):
                put(s, bar, 'lead', q + F(k, lead_div), s.vel(0.7 if k == 0 else 0.5), harm.tone(i, k, 2))
        else:
            put(s, bar, 'hat', q, s.vel(0.5))


def groove_synco(s: Song, bar, i: int, harm: Harmony, style: str) -> None:
    """Syncopation with a weak beat. ``push``: kick on 1 (every other bar) and the and of 3, snare and bass on
    off-beats, hats off the beat. ``onedrop``: nothing on 1; kick and rim on 3, skank chords on every off-beat."""
    if style == 'push':
        if i % 2 == 0:
            put(s, bar, 'kick', 0, s.vel(0.9))
        put(s, bar, 'kick', F(5, 2), s.vel(0.85))
        put(s, bar, 'snare', F(3, 2), s.vel(0.8))
        put(s, bar, 'snare', F(7, 2), s.vel(0.75))
        for q in range(4):
            put(s, bar, 'hat', q + F(1, 2), s.vel(0.55))
        for pos in (F(3, 4), F(3, 2), F(9, 4), F(7, 2)):
            put(s, bar, 'bass', pos, s.vel(0.75), harm.bass(i))
        for pos in (F(1, 2), F(9, 4)):
            put(s, bar, 'chord', pos, s.vel(0.55), harm.chord(i))
    elif style == 'onedrop':
        put(s, bar, 'kick', 2, s.vel(0.9))
        put(s, bar, 'snare', 2, s.vel(0.6))
        for q in range(4):
            put(s, bar, 'chord', q + F(1, 2), s.vel(0.6), harm.chord(i))
            put(s, bar, 'hat', q, s.vel(0.2))
        for pos in (F(1, 2), F(5, 4), F(3, 1) + F(1, 2)):
            put(s, bar, 'bass', pos, s.vel(0.8), harm.bass(i, pos > 2))
    else:
        raise ValueError(style)


def groove_swing(s: Song, bar, i: int, harm: Harmony, ratio: F) -> None:
    """Ride on swung eighths (off-beat at ``ratio`` of the beat), backbeat, walking bass on the beats."""
    put(s, bar, 'kick', 0, s.vel(0.8))
    put(s, bar, 'snare', 1, s.vel(0.7))
    put(s, bar, 'snare', 3, s.vel(0.7))
    for q in range(4):
        put(s, bar, 'hat', q, s.vel(0.6))
        if q % 2 == 1:
            put(s, bar, 'hat', q + ratio, s.vel(0.45))
        put(s, bar, 'bass', q, s.vel(0.75), harm.tone(i, q, -2))
    put(s, bar, 'chord', 1 + ratio, s.vel(0.5), harm.chord(i))


def groove_halftime(s: Song, bar, i: int, harm: Harmony) -> None:
    """Half-time drums (kick on 1, snare on 3) under sixteenth hats."""
    put(s, bar, 'kick', 0, s.vel(1.0))
    put(s, bar, 'kick', F(5, 2), s.vel(0.7))
    put(s, bar, 'snare', 2, s.vel(0.9))
    for k in range(16):
        put(s, bar, 'hat', F(k, 4), s.vel(0.5 if k % 2 == 0 else 0.3))
    put(s, bar, 'bass', 0, s.vel(0.8), harm.bass(i))
    put(s, bar, 'chord', 0, s.vel(0.5), harm.chord(i))


def groove_piano(s: Song, bar, i: int, harm: Harmony) -> None:
    """A performed piano-like texture: a chord on 1, melody on the beats and some eighths, bass on 1 and 3."""
    put(s, bar, 'chord', 0, s.vel(0.7), harm.chord(i))
    put(s, bar, 'bass', 0, s.vel(0.8), harm.bass(i))
    put(s, bar, 'bass', 2, s.vel(0.6), harm.bass(i, True))
    for q in range(int(math.ceil(bar.length))):
        put(s, bar, 'lead', q, s.vel(0.7 if q == 0 else 0.55), harm.tone(i, q, 2))
        if s.rng.random() < 0.4:
            put(s, bar, 'lead', q + F(1, 2), s.vel(0.4), harm.tone(i, q + 1, 2))


def groove_poly(s: Song, bar, i: int, harm: Harmony, period: F, origin: F = F(0), riff=None) -> None:
    """Soft 4/4 drums under a loud second layer: blocks and bass every ``period`` beats from ``origin`` (the
    clip's first beat), or, with ``riff``, on the riff's positions within each ``period``-beat cycle."""
    put(s, bar, 'kick', 0, s.vel(0.6))
    for q in range(4):
        put(s, bar, 'hat', q, s.vel(0.35))
    if i % 2 == 1:
        put(s, bar, 'snare', 3, s.vel(0.4))
    b0, b1 = bar.b0, bar.b0 + bar.length
    k0 = math.ceil((b0 - origin) / period)
    k = k0
    while origin + k * period < b1:
        c0 = origin + k * period
        offsets = [F(0)] if riff is None else [F(x) for x in riff]
        for j, off in enumerate(offsets):
            g = c0 + off
            if b0 <= g < b1:
                s.note('block', g, s.vel(1.0 if j == 0 else 0.8))
                s.note('bass', g, s.vel(0.9 if j == 0 else 0.7), harm.tone(i, j, -2))
        k += 1


# The settings.

def _start(s: Song, family: str = 'edm', crash=True, gains=None):
    harm = Harmony(s.rng)
    _kit(s, family, gains=gains, crash=crash)
    _room(s)
    return harm


def control(style: str, family: str, chord_voice: str = 'chord'):
    def build(s: Song):
        harm = _start(s, family)
        if chord_voice != 'chord':
            s.layer('chord', chord_voice, gain=0.4, family=family)
        bpm = _int_bpm(s)
        q = 60000.0 / bpm
        bars = _const(s, _fit(s, 4, q), 4, q)
        _fill_44(s, bars, 0, style, harm)
        s.params.update(bpm=bpm, style=style, family=family)
    return build


def control_click(s: Song):
    harm = Harmony(s.rng)
    s.layer('click', 'click', gain=0.8)
    s.layer('pluck', 'pluck', gain=0.4)
    s.layer('bass', 'bass', gain=0.5)
    _room(s, wet=(0.05, 0.1))
    bpm = _int_bpm(s)
    q = 60000.0 / bpm
    bars = _const(s, _fit(s, 4, q), 4, q)
    _fill_44(s, bars, 0, 'click', harm)
    s.params.update(bpm=bpm)


def control_frac(s: Song):
    harm = _start(s, 'edm')
    k = _int_bpm(s, 100, 140)
    delta = float(s.rng.uniform(0.06, 0.24)) * (1 if s.rng.random() < 0.5 else -1)
    bpm = k + delta
    q = 60000.0 / bpm
    bars = _const(s, _fit(s, 4, q), 4, q)
    _fill_44(s, bars, 0, 'edm', harm)
    s.params.update(bpm=bpm, note='within 0.25 BPM of an integer, not on it: R2 rounds it (G2)')


def mixsub(mode: str):
    def build(s: Song):
        harm = _start(s, 'band', crash=False)
        s.layer('lead', 'pluck', gain=0.45, decay=float(s.rng.uniform(0.08, 0.15)))
        bpm = float(s.rng.uniform(120, 136))
        q = 60000.0 / bpm
        bars = _const(s, _fit(s, 4, q), 4, q)
        for i, bar in enumerate(bars):
            if mode == 'alt':          # 1/6 and 1/8 in alternate beats, the pattern rotating by bar
                divs = [6 if (b + i) % 2 == 0 else 8 for b in range(4)]
                groove_lead(s, bar, i, harm, divs)
            elif mode == 'sim':        # 1/8 hats and a 1/6 lead in the same beats; every 4th bar plain
                if i % 4 == 3:
                    groove_lead(s, bar, i, harm, [0, 0, 0, 0])
                else:
                    groove_mixed_hats(s, bar, i, harm, 8, 6)
            elif mode == 'triplet':    # 1/3 and 1/4 alternating, both together in odd bars' first two beats
                if i % 2 == 1:
                    groove_mixed_hats(s, bar, i, harm, 4, 3, beats=(0, 1))
                else:
                    groove_lead(s, bar, i, harm, [3, 4, 3, 4])
            else:
                raise ValueError(mode)
        s.params.update(bpm=bpm, mode=mode)
    return build


def meter(kind: str):
    def build(s: Song):
        harm = _start(s, 'band')
        s.layer('hat', 'hat', gain=0.35, family='edm')
        qbpm = float(s.rng.integers(100, 136))
        q = 60000.0 / qbpm
        e = q / 2.0
        i = 0
        if kind in ('5_8', '7_8'):
            n, grouping = (5, (3, 2)) if kind == '5_8' else (7, (2, 2, 3))
            bars = _const(s, _fit(s, n, e), n, e, n, grouping)
            for i, bar in enumerate(bars):
                groove_grouped(s, bar, i, harm, F(1))
        elif kind == '7_8_reset':
            bars = _const(s, _fit(s, F(7, 2), q), F(7, 2), q, 4, (2, 2, 2, 1), (0, 1, 2, 3))
            for i, bar in enumerate(bars):
                groove_grouped(s, bar, i, harm, F(1, 2))
        elif kind == '7_4':
            bars = _const(s, _fit(s, 7, q), 7, q, 7, (4, 4, 6))
            for i, bar in enumerate(bars):
                groove_grouped(s, bar, i, harm, F(1, 2))
        elif kind == 'alt_78_44':
            while _fit(s, 15, e) >= 1:
                for n, grouping in ((7, (2, 2, 3)), (8, (2, 2, 2, 2))):
                    bar = _const(s, 1, n, e, n, grouping)[0]
                    groove_grouped(s, bar, i, harm, F(1))
                    i += 1
        elif kind == 'alt_4_3':
            while _fit(s, 7, q) >= 1:
                bar = _const(s, 1, 4, q, 4)[0]
                groove_44(s, bar, i, 'band', harm)
                i += 1
                bar = _const(s, 1, 3, q, 3, (2, 2, 2))[0]
                groove_grouped(s, bar, i, harm, F(1, 2))
                i += 1
        else:
            raise ValueError(kind)
        s.params.update(quarter_bpm=qbpm, kind=kind)
    return build


def tempo_steps(s: Song):
    harm = _start(s, 'edm')
    bpm = _int_bpm(s, 95, 140)
    i = 0
    steps = []
    while True:
        q = 60000.0 / bpm
        n = int(s.rng.choice([8, 12]))
        n = min(n, _fit(s, 4, q))
        if n < 2:
            break
        if i:
            steps.append(s.tm.t_end)
        bars = _const(s, n, 4, q)
        _fill_44(s, bars, i, 'edm' if (i // 8) % 2 == 0 else 'band', harm)
        i += n
        factor = float(s.rng.uniform(1.08, 1.25))
        up = bpm * factor <= 152 and (bpm / factor < 90 or s.rng.random() < 0.5)
        bpm = float(round(bpm * factor if up else bpm / factor))
    for t in steps:
        s.span(t, t + 4 * 60000.0 / s.tm.bpm_at(t + 1.0))
    s.params.update(n_steps=len(steps))


def tempo_short(s: Song):
    harm = _start(s, 'band')
    bpm = _int_bpm(s, 100, 136)
    q = 60000.0 / bpm
    i = 0
    events = []
    while _fit(s, 4, q) >= 1:
        n = min(int(s.rng.integers(5, 8)), _fit(s, 4, q))
        bars = _const(s, n, 4, q)
        _fill_44(s, bars, i, 'band', harm)
        i += n
        if _fit(s, 12, q) < 1:
            continue
        t0 = s.tm.t_end
        if len(events) % 2 == 0:       # an inserted 2/4 bar
            bar = _const(s, 1, 2, q, 2)[0]
            put(s, bar, 'kick', 0, s.vel(1.0))
            put(s, bar, 'snare', 1, s.vel(0.85))
            for e in range(4):
                put(s, bar, 'hat', F(e, 2), s.vel(0.5))
            put(s, bar, 'bass', 0, s.vel(0.8), harm.bass(i))
            i += 1
            events.append(('bar_2_4', t0))
        else:                           # 1 to 4 bars at another tempo
            factor = float(s.rng.choice([0.8, 0.875, 1.125, 1.25]))
            qq = 60000.0 / round(bpm * factor)
            n2 = int(s.rng.integers(1, 5))
            bars = _const(s, n2, 4, qq)
            _fill_44(s, bars, i, 'edm', harm)
            i += n2
            events.append((f'{n2}_bars_x{factor}', t0))
        s.span(t0, s.tm.t_end + 2 * 4 * q)
    s.params.update(bpm=bpm, events=[e for e, _ in events])


def ramp(direction: str):
    def build(s: Song):
        harm = _start(s, 'band')
        if direction == 'accel':
            bpm0 = float(s.rng.uniform(88, 105))
        else:
            bpm0 = float(s.rng.uniform(125, 142))
        q0 = 60000.0 / bpm0
        i = 0
        bars = _const(s, 6, 4, q0)
        _fill_44(s, bars, i, 'band', harm)
        i += 6
        bpm = bpm0
        ramps = []
        while True:
            r = float(s.rng.uniform(0.15, 0.3))
            nb = int(s.rng.integers(8, 17))
            if len(ramps) % 2 == 0:
                target = bpm * (1 + r) if direction == 'accel' else bpm * (1 - r)
            else:
                target = bpm0
            beats = 4 * nb
            dur = 120000.0 * beats / (bpm + target)
            if s.tm.t_end + dur + 4 * 4 * 60000.0 / target > _end(s):
                break
            t0 = s.tm.t_end
            s.tm.ramp(beats, bpm, target)
            for _ in range(nb):
                bar = s.bar(4)
                groove_44(s, bar, i, 'band', harm)
                i += 1
            ramps.append((t0, s.tm.t_end, bpm, target))
            bpm = target
            n = min(4, _fit(s, 4, 60000.0 / bpm))
            bars = _const(s, n, 4, 60000.0 / bpm)
            _fill_44(s, bars, i, 'band', harm)
            i += n
        n = _fit(s, 4, 60000.0 / bpm)
        _fill_44(s, _const(s, n, 4, 60000.0 / bpm), i, 'band', harm)
        for t0, t1, a, b in ramps:
            s.span(t0, t1 + 2 * 4 * 60000.0 / b)
        s.params.update(ramps=[(round(a, 3), round(b, 3)) for _, _, a, b in ramps])
    return build


def rubato(sd_lo: float, sd_hi: float, fermata: bool):
    def build(s: Song):
        harm = Harmony(s.rng)
        s.layer('chord', 'chord', gain=0.45, family='soft')
        s.layer('lead', 'pluck', gain=0.5)
        s.layer('bass', 'bass', gain=0.5, family='band')
        _room(s, wet=(0.15, 0.25), rt=(0.6, 1.0))
        bpm0 = float(s.rng.uniform(85, 120))
        sd = float(s.rng.uniform(sd_lo, sd_hi))
        d0 = math.log(60000.0 / bpm0)
        lengths, logd = [], d0
        while s.tm.t_end + sum(lengths) < _end(s) - 4 * 60000.0 / 80.0:
            logd += s.rng.normal(0.0, sd) - 0.1 * (logd - d0)
            logd = min(max(logd, math.log(60000.0 / 150.0)), math.log(60000.0 / 85.0))
            lengths.append(math.exp(logd))
        n_bars = len(lengths) // 4
        lengths = lengths[:4 * n_bars]
        fermatas = []
        if fermata:
            for b in range(7, n_bars, 8):
                k = 4 * b + 3
                lengths[k] *= float(s.rng.uniform(1.8, 2.5))
                fermatas.append(k)
            while s.tm.t_end + sum(lengths) > _end(s):
                lengths = lengths[:-4]
                n_bars -= 1
        s.tm.per_beat(lengths)
        for i in range(n_bars):
            groove_piano(s, s.bar(4), i, harm)
        s.params.update(bpm0=bpm0, sd=sd, fermata_beats=fermatas)
    return build


def edit_stutter(s: Song):
    harm = _start(s, 'edm')
    bpm = _int_bpm(s, 110, 140)
    q = 60000.0 / bpm
    bars = _const(s, _fit(s, 4, q), 4, q)
    _fill_44(s, bars, 0, 'edm', harm)
    pieces, cursor = [], 0.0
    for b in range(5, len(bars) - 2, 6):
        beat = int(s.rng.integers(2, 4))
        a = _fs(s, s.time(bars[b].b0 + beat))
        slice_beats = F(1, int(s.rng.choice([4, 8, 16, 32])))
        span_beats = 4 - beat
        n = int(span_beats / slice_beats)
        pieces.append(Copy(cursor, a))
        pieces.append(Repeat(a, float(slice_beats) * q, n))
        cursor = _fs(s, a + n * float(slice_beats) * q)
        s.hard_stop(a)
        s.hard_stop(cursor)
        s.effects.append((a, cursor, f'stutter 1/{slice_beats.denominator}'))
        s.span(a, cursor + 4 * q)
    pieces.append(Copy(cursor, s.end_ms))
    s.edits = EditList(pieces)
    s.params.update(bpm=bpm, stutters=len(s.effects))


def edit_cut(s: Song):
    harm = _start(s, 'edm')
    bpm = _int_bpm(s, 100, 140)
    q = 60000.0 / bpm
    bars = _const(s, _fit(s, 4, q), 4, q)
    _fill_44(s, bars, 0, 'edm', harm)
    pieces, cursor, kinds = [], 0.0, []
    for b in range(4, len(bars) - 3, int(s.rng.integers(5, 8))):
        beat = int(s.rng.integers(1, 4))
        a = _fs(s, s.time(bars[b].b0 + beat))
        frac = F(int(s.rng.choice([1, 2, 3])), 4)
        if s.rng.random() < 0.5:     # cut: a fraction of a beat is removed after a beat
            cut_from = _fs(s, s.time(bars[b].b0 + beat + frac))
            cut_to = _fs(s, s.time(bars[b].b0 + beat + 1))
            s.hard_stop(cut_from)
            s.hard_stop(cut_to)
            pieces.append(Copy(cursor, cut_from))
            cursor = cut_to
            kinds.append(f'cut {frac} beat')
            s.span(a, a + 3 * 4 * q)
        else:                          # insert: the fraction of a beat after a beat is heard twice
            ins_to = _fs(s, s.time(bars[b].b0 + beat + frac))
            s.hard_stop(a)
            s.hard_stop(ins_to)
            pieces.append(Copy(cursor, ins_to))
            cursor = a
            kinds.append(f'insert {frac} beat')
            s.span(a, a + 3 * 4 * q)
    pieces.append(Copy(cursor, s.end_ms))
    s.edits = EditList(pieces)
    s.params.update(bpm=bpm, edits=kinds)


def edit_tapestop(s: Song):
    harm = _start(s, 'edm')
    bpm = _int_bpm(s, 100, 140)
    q = 60000.0 / bpm
    bars = _const(s, _fit(s, 4, q) + 8, 4, q)
    _fill_44(s, bars, 0, 'edm', harm)
    pieces, cursor, out, stops = [], 0.0, 0.0, []
    b = int(s.rng.integers(9, 13))
    while b + 6 < len(bars):
        a = _fs(s, s.time(bars[b].b0 + int(s.rng.integers(2, 4))))
        d = float(s.rng.uniform(500.0, 2000.0))
        g = float(s.rng.uniform(0.0, 800.0))
        nxt = next(k for k in range(b + 1, len(bars)) if s.time(bars[k].b0) >= a + d / 2.0 + 1.0)
        re = _fs(s, s.time(bars[nxt].b0))
        s.hard_stop(re)
        pieces += [Copy(cursor, a), Tape(a, d)] + ([Silence(g)] if g > 0 else [])
        out += (a - cursor) + d + g
        cursor = re
        stops.append((round(d), round(g)))
        s.span(re, re + 4 * 4 * q)
        b = nxt + int(s.rng.integers(8, 12))
        if out + 4 * 4 * q * 3 > _end(s):
            break
    pieces.append(Copy(cursor, s.end_ms))
    s.edits = EditList(pieces)
    s.params.update(bpm=bpm, tape_stops_ms=stops)


def poly(kind: str):
    def build(s: Song):
        harm = _start(s, 'band', crash=False, gains=dict(kick=0.7, snare=0.5, hat=0.3))
        s.layer('block', 'block', role='counter', gain=0.9)
        s.layers['bass'].role = 'counter'
        bpm = _int_bpm(s, 100, 136)
        q = 60000.0 / bpm
        bars = _const(s, _fit(s, 4, q), 4, q)
        for i, bar in enumerate(bars):
            if kind == '3_4':
                # Three evenly spaced notes in each bar of four beats.
                groove_poly(s, bar, i, harm, F(4, 3), bar.b0)
            elif kind == '3_2':
                groove_poly(s, bar, i, harm, F(3, 2))
            elif kind == '5_4':
                groove_poly(s, bar, i, harm, F(5), riff=(0, F(3, 2), 3, 4))
            else:
                raise ValueError(kind)
            if kind == '5_4':
                put(s, bar, 'snare', 1, s.vel(0.6))
                put(s, bar, 'snare', 3, s.vel(0.6))
        s.layers['block'].period = {'3_4': '4/3 beat', '3_2': '3/2 beat', '5_4': '5-beat riff'}[kind]
        s.params.update(bpm=bpm, kind=kind)
    return build


def _pads(s: Song, harm: Harmony, t0: float, t1: float, i: int, vel=0.6):
    """Pads over [t0, t1): overlapping held chords without attacks, and a drone under them."""
    t = t0
    k = 0
    while t < t1:
        dur = min(float(s.rng.uniform(3000, 5000)), t1 - t + 1500)
        s.note_ms('pad', t, vel, harm.chord(i + k), dur=dur)
        t += dur * 0.7
        k += 1
    s.note_ms('drone', t0, vel * 0.7, (midi(harm.root),), dur=t1 - t0 + 2000)


def beatless(kind: str):
    def build(s: Song):
        harm = _start(s, 'edm')
        s.layer('pad', 'pad', role='texture', gain=0.5)
        s.layer('drone', 'pad', role='texture', gain=0.35)
        if kind == 'free':
            s.layer('free', 'pluck', role='free', gain=0.5)
        bpm = _int_bpm(s, 100, 136)
        q = 60000.0 / bpm
        n1 = int(s.rng.integers(10, 15))
        bars = _const(s, n1, 4, q)
        _fill_44(s, bars, 0, 'edm', harm)
        gap = float(s.rng.uniform(12000, 20000)) if kind != 'free' else float(s.rng.uniform(10000, 16000))
        t0 = s.tm.t_end
        s.hard_stop(t0 + gap)
        s.tm.gap(gap)
        s.gap_kinds.append('beatless' if kind != 'free' else 'free')
        if kind == 'free':
            t = t0 + 300.0
            k = 0
            while t < t0 + gap - 500.0:
                s.note_ms('free', t, s.vel(0.6, 0.2), harm.tone(n1 + k // 6, k, 1 + (k % 2)))
                t += float(s.rng.lognormal(math.log(260.0), 0.6))
                k += 1
            _pads(s, harm, t0 - 1000.0, t0 + gap, n1, 0.35)
        else:
            _pads(s, harm, t0 - 1500.0, t0 + gap, n1)
        bpm2 = bpm if kind != 'tempo' else float(round(bpm * float(s.rng.choice([0.82, 0.88, 1.14, 1.22]))))
        q2 = 60000.0 / bpm2
        bars2 = _const(s, _fit(s, 4, q2), 4, q2)
        _fill_44(s, bars2, n1, 'edm', harm)
        re = s.time(bars2[0].b0)
        s.span(re, re + 4 * 4 * q2)
        s.params.update(bpm=bpm, bpm_after=bpm2, gap_ms=gap, kind=kind)
    return build


def sparse(kind: str):
    def build(s: Song):
        harm = _start(s, 'band')
        s.layer('pad', 'pad', role='texture', gain=0.4)
        s.layer('drone', 'pad', role='texture', gain=0.25)
        bpm = _int_bpm(s, 96, 136)
        q = 60000.0 / bpm
        i = 0
        while True:
            n = min(int(s.rng.integers(6, 9)), _fit(s, 4, q))
            if n < 1:
                break
            bars = _const(s, n, 4, q)
            _fill_44(s, bars, i, 'band', harm)
            i += n
            m = min(int(s.rng.integers(10, 15)), _fit(s, 4, q) - 6)
            if m < 4:
                continue
            t0 = s.tm.t_end
            bars = _const(s, m, 4, q)
            for k, bar in enumerate(bars):
                if kind == 'hits':      # one soft hit every other bar
                    if k % 2 == 0:
                        put(s, bar, 'snare', 0, s.vel(0.35))
                else:                   # a bass note on each downbeat and a soft hat on beat 3 of every other bar
                    put(s, bar, 'bass', 0, s.vel(0.6), harm.bass(i + k))
                    if k % 2 == 1:
                        put(s, bar, 'hat', 2, s.vel(0.3))
            _pads(s, harm, t0, s.tm.t_end, i, 0.45)
            i += m
            s.span(t0, s.tm.t_end + 4 * 4 * q)
        s.params.update(bpm=bpm, kind=kind)
    return build


def syncopation(style: str):
    def build(s: Song):
        harm = _start(s, 'band' if style == 'onedrop' else 'edm', crash=False)
        bpm = _int_bpm(s, 90, 130)
        q = 60000.0 / bpm
        bars = _const(s, _fit(s, 4, q), 4, q)
        for i, bar in enumerate(bars):
            groove_synco(s, bar, i, harm, style)
        s.params.update(bpm=bpm, style=style)
    return build


def swing(s: Song):
    harm = _start(s, 'band', crash=False)
    bpm = _int_bpm(s, 100, 140)
    ratio = F(round(float(s.rng.uniform(0.6, 0.72)) * 100), 100)
    q = 60000.0 / bpm
    bars = _const(s, _fit(s, 4, q), 4, q)
    for i, bar in enumerate(bars):
        groove_swing(s, bar, i, harm, ratio)
    s.params.update(bpm=bpm, ratio=float(ratio))


def level_80_160(s: Song):
    harm = _start(s, 'edm')
    bpm = 80.0 if s.seed % 2 == 0 else 160.0
    q = 60000.0 / bpm
    if bpm == 160.0:
        bars = _const(s, _fit(s, 4, q), 4, q)
        for i, bar in enumerate(bars):
            # Notated at 160 in 4/4: kick on 1, snare on 3, eighth hats; the same groove reads as 80 BPM.
            put(s, bar, 'kick', 0, s.vel(1.0))
            put(s, bar, 'snare', 2, s.vel(0.9))
            for k in range(8):
                put(s, bar, 'hat', F(k, 2), s.vel(0.5 if k % 2 == 0 else 0.3))
            put(s, bar, 'bass', 0, s.vel(0.8), harm.bass(i))
            if i % 2 == 0:
                put(s, bar, 'chord', 0, s.vel(0.5), harm.chord(i // 2))
    else:
        bars = _const(s, _fit(s, 4, q), 4, q)
        for i, bar in enumerate(bars):
            groove_halftime(s, bar, i, harm)
    s.params.update(bpm=bpm)


_LIST = [
    Setting('control_a', 'S0', 'Steady 4/4, integer tempo, electronic kit (four on the floor), bass, chord stabs',
            control('edm', 'edm')),
    Setting('control_b', 'S0', 'Steady 4/4, integer tempo, band kit (kick 1 and 3, backbeat, eighth hats), bass, chords',
            control('band', 'band')),
    Setting('control_c', 'S0', 'Steady 4/4, integer tempo, soft low kit, plucked chords',
            control('band', 'soft', 'pluck')),
    Setting('control_click', 'S0', 'Steady 4/4 click track with a plucked arpeggio and bass (an absolute check)',
            control_click),
    Setting('control_frac', 'S0f', 'As control_a at a tempo within 0.25 BPM of an integer, not on it', control_frac),
    Setting('mixsub_alt', 'S1', '1/6 and 1/8 lead notes in alternate beats around 128 BPM (#32\'s shape)',
            mixsub('alt')),
    Setting('mixsub_sim', 'S1', '1/8 hats and a 1/6 lead inside the same beats (mixed, divisor 24)', mixsub('sim')),
    Setting('mixsub_triplet', 'S1', '1/3 and 1/4 alternating, and together in some beats (divisor 12)',
            mixsub('triplet')),
    Setting('meter_5_8', 'S2', '5/8 (3+2), notated in eighths, meter 5', meter('5_8')),
    Setting('meter_7_8', 'S2', '7/8 (2+2+3), notated in eighths, meter 7', meter('7_8')),
    Setting('meter_7_8_reset', 'S2', '7/8 (2+2+2+1) notated in quarters with a red line at every bar',
            meter('7_8_reset')),
    Setting('meter_7_4', 'S2', '7/4 (2+2+3 quarters), meter 7', meter('7_4')),
    Setting('meter_alt_78_44', 'S2', 'Alternating 7/8 and 4/4 bars, notated in eighths', meter('alt_78_44')),
    Setting('meter_alt_4_3', 'S2', 'Alternating 4/4 and 3/4 bars, notated in quarters', meter('alt_4_3')),
    Setting('tempo_steps', 'S3s', 'Sections of 8 or 12 bars at integer tempi 8-25 % apart', tempo_steps),
    Setting('tempo_short', 'S3s', 'Inserted 2/4 bars, and 1-4 bars at another tempo', tempo_short),
    Setting('ramp_accel', 'S3r', 'Accelerando of 15-30 % over 8-16 bars (tempo linear in time), then back',
            ramp('accel')),
    Setting('ramp_rit', 'S3r', 'Ritardando of 15-30 % over 8-16 bars, then a tempo', ramp('rit')),
    Setting('rubato_low', 'S3p', 'Rubato: log beat length a mean-reverting random walk, per-beat SD 1-2 %',
            rubato(0.01, 0.02, False)),
    Setting('rubato_high', 'S3p', 'Rubato with per-beat SD 3-5 % and a fermata every 8 bars',
            rubato(0.03, 0.05, True)),
    Setting('edit_stutter', 'S9', 'Beat repeats of 1/4-1/32 beat slices filling the rest of a bar; the grid holds',
            edit_stutter),
    Setting('edit_cut', 'S9', 'Cuts and inserted repeats of 1/4-3/4 beat: phase jumps', edit_cut),
    Setting('edit_tapestop', 'S9', 'Tape stops of 0.5-2 s, optional silence, re-entry on a new bar', edit_tapestop),
    Setting('poly_3_4', 'S8', 'Three loud evenly spaced notes per 4/4 bar (3 against 4) over soft drums',
            poly('3_4')),
    Setting('poly_3_2', 'S8', 'A loud second pulse every 3/2 beats (dotted quarters) over soft drums', poly('3_2')),
    Setting('polymeter_5_4', 'S8', 'A loud five-beat riff over 4/4 drums (polymeter)', poly('5_4')),
    Setting('beatless', 'S6b', '10-14 bars of groove, 12-20 s of pads without attacks, re-entry at the same tempo',
            beatless('same')),
    Setting('beatless_tempo', 'S6b', 'As beatless, re-entering at another tempo', beatless('tempo')),
    Setting('free_time', 'S6b', 'Groove, then 10-16 s of free-time plucks without pulse, then re-entry',
            beatless('free')),
    Setting('sparse_hits', 'S6s', 'Spans of 10-14 bars with one soft hit every other bar over pads; the pulse goes on',
            sparse('hits')),
    Setting('sparse_bass', 'S6s', 'Spans with a bass note per bar and a soft hat every other bar over pads',
            sparse('bass')),
    Setting('synco_push', 'Sy', 'Syncopated groove: kick on 1 every other bar, off-beat snare, bass and hats',
            syncopation('push')),
    Setting('synco_onedrop', 'Sy', 'One drop: nothing on 1, kick and rim on 3, skank chords on the off-beats',
            syncopation('onedrop')),
    Setting('swing', 'S7', 'Swung eighths (ratio 0.60-0.72), backbeat, walking bass', swing),
    Setting('level_80_160', 'S5', 'Half-time groove at exactly 80 BPM (seed even) or 160 BPM (seed odd)',
            level_80_160),
]

SETTINGS: dict[str, Setting] = {x.name: x for x in _LIST}

STRATA = {
    'S0': 'control: steady 4/4, integer tempo',
    'S0f': 'control at a fractional tempo near an integer (R2 rounds it, G2)',
    'S1': 'mixed subdivision',
    'S2': 'odd and changing meter',
    'S3s': 'tempo steps and short sections',
    'S3r': 'ramps',
    'S3p': 'rubato',
    'S9': 'stutters and edits',
    'S8': 'polyrhythm and polymeter',
    'S6b': 'beatless spans and free time',
    'S6s': 'sparse spans',
    'Sy': 'syncopation with a weak beat',
    'S7': 'swing',
    'S5': 'metrical level at 80 and 160 BPM',
}
