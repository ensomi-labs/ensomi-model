"""The ground truth of a clip, read off its program, on the decoded-audio clock.

Every time here is milliseconds from the clip's first sample (sample n at
1000 n / 24000 ms): the decoded clock. Charts use osu!'s clock, about 20 ms
earlier; the evaluation converts with one constant, never per clip.

`ground_truth(song)` gives a dict with:

- ``beats``: the notated beats where they are heard, after edits, with their
  ``downbeat`` flags, ``bar`` and ``beat_in_bar`` indices, ``meter`` (the bar's
  notated beats), notated beat length ``step_ms`` and tempo ``bpm``;
- ``segments``: the red lines a careful mapper writes, derived from the
  heard beats: a new line wherever a beat leaves the current line's grid,
  its tempo or its bar count (resets, meter changes, tempo steps, edits).
  Only for clips whose tempo is piecewise constant; ramps and rubato have
  none (``segments_note`` says so), since one red line per beat is not a
  segment structure the level and boundary terms can read;
- ``subdivision_class`` per beat, from the exact positions of the beat and
  counter layers' notes inside it: plain, duple, triple, mixed, finer, or
  none (no note);
- ``layers``: per layer, its voice, role, period and heard onsets
  (``effect`` marks onsets an edit made: repeats and tape-stop onsets);
- ``no_pulse``: spans without a pulse (beatless, free time, tape stop,
  silence);
- ``pulse_spans`` and ``spans`` (the stratum spans the measures run on;
  the pulse spans when the setting names none);
- ``effects``: spans of an edit's effect where the grid holds (stutters).

`music_map`, `onset_file` and `osu_text` write it out.
"""
from __future__ import annotations

from fractions import Fraction as F
import math

import numpy as np

from ..mapformat import MusicMap
from ..timing import Segment

ON_GRID_MS = 0.5
DIVISORS = (1, 2, 3, 4, 6, 8, 12, 16)
CLASSES = ('none', 'plain', 'duple', 'triple', 'mixed', 'finer')


def _class(divisor: int) -> str:
    if divisor == 0:
        return 'finer'
    if divisor <= 2:
        return 'plain'
    if divisor & (divisor - 1) == 0:
        return 'duple'
    return 'mixed' if divisor % 4 == 0 else 'triple'


def subdivision_class(offsets) -> tuple[str, int]:
    """Class and required divisor of notes at exact offsets (fractions of a beat) inside one beat."""
    if not offsets:
        return 'none', 0
    lcm = 1
    for x in offsets:
        d = F(x).denominator
        if d not in DIVISORS:
            return 'finer', 0
        lcm = lcm * d // math.gcd(lcm, d)
    return _class(lcm), lcm


def derive_segments(t, step, down, meter) -> list[Segment]:
    """Red lines from heard beats (module docstring)."""
    segs: list[Segment] = []
    cur = None
    for k in range(len(t)):
        new = cur is None
        if not new:
            j = (t[k] - cur.start_ms) / cur.beat_ms
            jr = round(j)
            on = abs(j - jr) * cur.beat_ms <= ON_GRID_MS
            same = abs(step[k] / cur.beat_ms - 1.0) <= 1e-9
            # osu! counts bars from a red line, or from `meter` beats after it with omit-first-barline.
            on_bar = jr % cur.meter == 0 and (cur.bar_at_start or jr > 0)
            bar_ok = on_bar == bool(down[k])
            new = not (on and same and bar_ok and (not down[k] or meter[k] == cur.meter))
        if new:
            cur = Segment(float(t[k]), float(step[k]), int(meter[k]), bool(down[k]))
            segs.append(cur)
    return segs


def ground_truth(song, clip_ms: float) -> dict:
    tm = song.tm
    edits = song.edits
    rows = []
    positions = song.beat_positions()
    for idx, (pos, i, j) in enumerate(positions):
        t_src = song.time(pos)
        bar = song.bars[i]
        nxt = positions[idx + 1][0] if idx + 1 < len(positions) else bar.b1
        nxt = min(nxt, pos + 1, bar.b1)
        step = tm.step(float(pos))
        outs = [t_src] if edits is None else edits.map_grid_all(t_src)
        for t in outs:
            rows.append((t, pos, nxt, i, j, bar.meter, float(bar.length), step))
    rows.sort(key=lambda r: r[0])
    rows = [r for r in rows if r[0] <= clip_ms - 500.0]
    t = np.array([r[0] for r in rows])
    step = np.array([r[7] for r in rows])
    down = np.array([r[4] == 0 for r in rows])
    meter = np.array([r[5] for r in rows])

    # Subdivision class of each beat, from the notes of beat and counter layers on the beat axis.
    timed = [n for n in song.notes if n.pos is not None and song.layers[n.layer].role in ('beat', 'counter')]
    pos_sorted = sorted(timed, key=lambda n: n.pos)
    keys = [n.pos for n in pos_sorted]
    import bisect
    classes, divisors = [], []
    for r in rows:
        a, b = r[1], r[2]
        lo, hi = bisect.bisect_left(keys, a), bisect.bisect_left(keys, b)
        c, d = subdivision_class([keys[x] - a for x in range(lo, hi)])
        classes.append(c)
        divisors.append(d)

    piecewise = all(p.kind in ('const', 'gap') for p in tm.pieces)
    segments = derive_segments(t, step, down, meter) if piecewise else None

    layers = {}
    for name, lay in song.layers.items():
        on, eff = [], []
        for n in song.notes:
            if n.layer != name:
                continue
            heard = [(n.t, False, 1.0)] if edits is None else edits.map_onsets(n.t)
            for o, e, _ in heard:
                if o <= clip_ms:
                    on.append(o)
                    eff.append(e)
        order = np.argsort(on, kind='stable')
        layers[name] = dict(voice=lay.voice, role=lay.role, gain=lay.gain, period=lay.period,
                            onsets_ms=[float(on[k]) for k in order], effect=[bool(eff[k]) for k in order])

    no_pulse = []
    gaps = [p for p in tm.pieces if p.kind == 'gap']
    for p, kind in zip(gaps, song.gap_kinds or ['beatless'] * len(gaps)):
        no_pulse.append((p.t0, p.t1, kind))
    if edits is not None:
        from .tempo import Silence, Tape
        for start, p in zip(edits.starts, edits.pieces):
            if isinstance(p, Tape):
                no_pulse.append((float(start), float(start + p.out_ms), 'tapestop'))
            elif isinstance(p, Silence):
                no_pulse.append((float(start), float(start + p.out_ms), 'silence'))
    no_pulse = sorted((max(0.0, a), min(b, clip_ms), k) for a, b, k in no_pulse if a < clip_ms)
    # A tape stop and the silence after it are one span.
    merged = []
    for a, b, k in no_pulse:
        if merged and abs(a - merged[-1][1]) < 1e-6:
            merged[-1] = (merged[-1][0], b, merged[-1][2] + '+' + k)
        else:
            merged.append((a, b, k))

    def to_out(spans):
        if edits is None:
            out = list(spans)
        else:
            out = [x for a, b in spans for x in edits.map_span(a, b)]
        return [(float(a), float(min(b, clip_ms))) for a, b in out if a < clip_ms]

    pulse = to_out(song.pulse_spans())
    spans = to_out(song.spans) if song.spans else pulse
    effects = []
    for a, b, kind in song.effects:
        for x in to_out([(a, b)]):
            effects.append((x[0], x[1], kind))
    first, last = (float(t[0]), float(t[-1])) if len(t) else (math.nan, math.nan)
    return dict(
        clip_ms=float(clip_ms), first_beat_ms=first, last_beat_ms=last,
        beats=dict(t_ms=t.tolist(), downbeat=down.tolist(), bar=[int(r[3]) for r in rows],
                   beat_in_bar=[int(r[4]) for r in rows], meter=[int(m) for m in meter],
                   bar_beats=[r[6] for r in rows], step_ms=step.tolist(), bpm=(60000.0 / step).tolist()),
        segments=None if segments is None else [dict(start_ms=s.start_ms, beat_ms=s.beat_ms, meter=s.meter,
                                                     bar_at_start=s.bar_at_start) for s in segments],
        segments_note='' if piecewise else 'tempo ramps or rubato: no segment structure, beats are exact',
        subdivision_class=classes, subdivision_divisor=divisors,
        layers=layers, no_pulse=[list(x) for x in merged], pulse_spans=[list(x) for x in pulse],
        spans=[list(x) for x in spans], effects=[list(x) for x in effects],
    )


def music_map(gt: dict, audio: str, provenance: dict) -> MusicMap:
    """The ground truth as a music map on the decoded clock; the rest of the truth under ``extra['truth']``."""
    b = gt['beats']
    sub = []
    for c in gt['subdivision_class']:
        sub.append(None if c in ('none', 'finer') else [1.0 if c in ('triple', 'mixed') else 0.0,
                                                          1.0 if c in ('duple', 'mixed') else 0.0])
    segs = None
    if gt['segments']:
        segs = [Segment(s['start_ms'], s['beat_ms'], s['meter'], s['bar_at_start']) for s in gt['segments']]
    extra = {k: v for k, v in gt.items() if k not in ('beats',)}
    extra['beat_info'] = {k: v for k, v in b.items() if k not in ('t_ms', 'downbeat')}
    return MusicMap(beats=np.array(b['t_ms']), downbeats=np.flatnonzero(b['downbeat']), subdivision=sub,
                    segments=tuple(segs) if segs else None, model=dict(provenance), clock='decoded', audio=audio,
                    extra={'truth': extra})


def onset_file(gt: dict, provenance: dict) -> dict:
    return dict(format='ensomi-synth-onsets', version=1, clock='decoded', provenance=provenance, layers=gt['layers'],
                no_pulse=gt['no_pulse'], effects=gt['effects'])


def osu_text(gt: dict, audio: str, title: str) -> str:
    """The ground truth as an osu!mania file on the decoded clock: its red lines (one per beat where the clip
    has no segment structure) and one head per beat."""
    b = gt['beats']
    t, step, down, meter = b['t_ms'], b['step_ms'], b['downbeat'], b['meter']
    if gt['segments']:
        lines = [(s['start_ms'], s['beat_ms'], s['meter'], not s['bar_at_start']) for s in gt['segments']]
    else:
        lines = [(t[k], (t[k + 1] - t[k]) if k + 1 < len(t) else step[k], meter[k], not down[k])
                 for k in range(len(t))]
    out = ['osu file format v14', '', '[General]', f'AudioFilename: {audio}', 'Mode: 3', '', '[Metadata]',
           f'Title:{title}', 'Artist:ensomi-synth', 'Creator:ensomi-synth', 'Version:truth (decoded clock)',
           'BeatmapSetID:-1', '', '[Difficulty]', 'CircleSize:4', '', '[TimingPoints]']
    for off, beat, m, omit in lines:
        out.append(f'{off!r},{beat!r},{int(m)},1,0,100,1,{8 if omit else 0}')
    out += ['', '[HitObjects]']
    out += [f'64,192,{x!r},1,0,0:0:0:0:' for x in t]
    return '\n'.join(out) + '\n'
