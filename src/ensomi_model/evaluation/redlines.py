"""Red lines as written in a chart, and which of them state the music's beat.

osu!mania mappers write red lines for different purposes. Some state the
music's tempo, beat phase or bar structure. Some repeat the grid already in
force. Others serve only what the player sees: BPM values no music has (stops,
teleports, barline art), or a passage at another BPM that changes the scroll
speed while the notes stay on the music's grid. The beat grid used for
evaluation keeps the first kind and drops the last.

Each line gets a role:

- ``musical``: the first usable line (``first``); a change of tempo or offset
  the events need (``tempo``); or a bar line that moves the canonical beat
  phase (``phase``): at the notated BPM in force, on a half or quarter of its
  canonical beat, as a meter change at a notated BPM above 160 can be.
- ``redundant``: continues the grid in force on its canonical beats
  (``same_grid``). It may still start a bar.
- ``expressive``: dropped from the grid, with a reason:
  ``superseded`` (another red line at the same offset applies, as in osu!);
  ``implausible`` (BPM not finite or outside 20 to 1000);
  ``empty`` (no event in its segment, which is shorter than one canonical beat);
  ``short_bar`` (a bar line whose segment is shorter than one of its own bars
  in canonical beats: barline art or a bar the mapper did not complete);
  ``detour`` (a run of lines with another BPM or phase, after which the grid
  in force resumes, or the chart ends, while every event of the run is on a
  binary subdivision, 1/1 to 1/16, of that grid's canonical beat and is
  described by it at least as simply as by the run's own lines, counting
  ``log2`` of each event's denominator). Triplets do not count: a passage at
  3/2 or 2/3 of the tempo, a common real tempo change, lies on the triplets
  of the grid in force.

A line continues the grid in force when it has the same canonical BPM and
lies on a quarter of its canonical beat, both within ``SNAP_TOLERANCE_MS`` over
the line's whole segment, and every event its own grid places is also on the
grid in force. Such a line is redundant on a whole canonical beat; otherwise,
at the same notated BPM, it is a bar line, and bar lines are never detours: a
BPM change is what changes the scroll speed in osu!mania. Comparisons are in
canonical beats (see ``beats``), so renotating a whole chart at half or double
the BPM changes no role.

``musical_grid`` builds the evaluation grid: musical lines are its segments;
musical and redundant lines start bars, except those with osu!'s
omit-first-barline flag (the first line always starts one).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from ..osu_core.timing import MAX_VALID_RED_BEAT_LENGTH_MS, MIN_VALID_RED_BEAT_LENGTH_MS
from .beats import COMMON_DENOMINATORS, SNAP_TOLERANCE_MS, BarStart, BeatGrid, Segment, canonical_beat_length

OFF_GRID_BITS = 8.0
BINARY_DENOMINATORS = (1, 2, 4, 8, 16)

MUSICAL, REDUNDANT, EXPRESSIVE = 'musical', 'redundant', 'expressive'


@dataclass(frozen=True)
class RedLine:
    """A red (uninherited) timing point as written in the file."""
    offset_ms: float
    beat_length_ms: float
    meter: int = 4
    omit_first_barline: bool = False

    @property
    def plausible(self) -> bool:
        return (math.isfinite(self.offset_ms) and math.isfinite(self.beat_length_ms)
                and MIN_VALID_RED_BEAT_LENGTH_MS <= self.beat_length_ms <= MAX_VALID_RED_BEAT_LENGTH_MS)


@dataclass(frozen=True)
class LineRole:
    role: str
    reason: str


def read_red_lines(path: str | Path) -> list[RedLine]:
    """Every red line of an ``.osu`` file, ordered by offset, file order kept for ties.

    Nothing is rejected here: implausible values are kept for ``classify`` to
    label. A meter below 1 reads as 4, as osu! does.
    """
    lines, section = [], None
    with Path(path).open('r', encoding='utf-8-sig', errors='replace') as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith('//'):
                continue
            if line.startswith('[') and line.endswith(']'):
                section = line[1:-1]
                continue
            if section != 'TimingPoints':
                continue
            parts = [p.strip() for p in line.split('//', 1)[0].split(',')]
            if len(parts) < 2:
                raise ValueError(f'Malformed timing point: {line}')
            if len(parts) > 6 and parts[6] and int(float(parts[6])) == 0:
                continue
            meter = int(float(parts[2])) if len(parts) > 2 and parts[2] else 4
            effects = int(float(parts[7])) if len(parts) > 7 and parts[7] else 0
            lines.append(RedLine(float(parts[0]), float(parts[1]), meter if meter >= 1 else 4, bool(effects & 8)))
    return sorted(lines, key=lambda r: r.offset_ms)


def snap_denominators(times_ms, offset_ms: float, canonical_ms: float) -> np.ndarray:
    """Smallest canonical denominator in ``COMMON_DENOMINATORS`` within tolerance, else 0."""
    beat = (np.asarray(times_ms, dtype=np.float64) - offset_ms) / canonical_ms
    out = np.zeros(beat.shape, dtype=np.int16)
    for d in COMMON_DENOMINATORS:
        error = np.abs(beat - np.round(beat * d) / d) * canonical_ms
        out[(out == 0) & (error <= SNAP_TOLERANCE_MS)] = d
    return out


def _continues(grid_offset: float, grid_ms: float, offsets: np.ndarray, lengths: np.ndarray,
               ends: np.ndarray, per_beat: int = 1) -> np.ndarray:
    """Whether each line keeps the grid's beat length and lies on a ``1/per_beat`` of its beat.

    Both within tolerance over the line's segment, up to ``ends``.
    """
    step = grid_ms / per_beat
    position = (offsets - grid_offset) / step
    start_error = (position - np.round(position)) * step
    beats = np.maximum(ends - offsets, 0.0) / lengths
    end_error = start_error + beats * (lengths - grid_ms)
    return (np.abs(start_error) <= SNAP_TOLERANCE_MS) & (np.abs(end_error) <= SNAP_TOLERANCE_MS)


def _bits(denominators: np.ndarray) -> float:
    d = denominators.astype(np.float64)
    return float(np.where(d > 0, np.log2(np.maximum(d, 1.0)), OFF_GRID_BITS).sum())


def classify(lines: Sequence[RedLine], event_times_ms) -> list[LineRole]:
    """Role of every line in ``lines`` (offset order), given the chart's event times.

    Event times are heads and releases. Rules are applied in the order listed
    in the module docstring; the first surviving line is musical.
    """
    n = len(lines)
    roles: list[LineRole | None] = [None] * n
    events = np.sort(np.asarray(event_times_ms, dtype=np.float64))
    last_event = float(events[-1]) if len(events) else -math.inf
    for i in range(n):
        if i + 1 < n and lines[i + 1].offset_ms == lines[i].offset_ms:
            roles[i] = LineRole(EXPRESSIVE, 'superseded')
        elif not lines[i].plausible:
            roles[i] = LineRole(EXPRESSIVE, 'implausible')
    following = math.inf
    for i in reversed(range(n)):
        if roles[i] is not None:
            continue
        line = lines[i]
        start, end = line.offset_ms, following
        count = np.searchsorted(events, end, side='left') - np.searchsorted(events, start, side='left')
        if count == 0 and end - start < canonical_beat_length(line.beat_length_ms):
            roles[i] = LineRole(EXPRESSIVE, 'empty')
        else:
            following = start
    kept = [i for i in range(n) if roles[i] is None]
    if not kept:
        return roles  # type: ignore[return-value]
    offsets = np.array([lines[i].offset_ms for i in kept])
    notated = np.array([lines[i].beat_length_ms for i in kept])
    canonical = np.array([canonical_beat_length(b) for b in notated])
    bar_ms = canonical * np.array([lines[i].meter for i in kept])
    ends = np.append(offsets[1:], max(offsets[-1], last_event))
    first_event = np.append(np.searchsorted(events, offsets, side='left'), len(events))

    def keeps_events(m: int, g: int) -> bool:
        """No event of segment ``m`` is on line ``m``'s grid and off grid ``g``."""
        seg = events[first_event[m]:first_event[m + 1]]
        own = snap_denominators(seg, offsets[m], canonical[m]) > 0
        return bool(np.all(~own | (snap_denominators(seg, offsets[g], canonical[g]) > 0)))

    def resumes(g: int, lo: int, hi: int) -> np.ndarray:
        """Lines ``lo:hi`` at grid ``g``'s canonical BPM, on a quarter of its canonical beat."""
        part = slice(lo, hi)
        return _continues(offsets[g], canonical[g], offsets[part], canonical[part], ends[part], per_beat=4)

    def follow(m: int, g: int) -> LineRole | None:
        """Role of line ``m`` if it continues grid ``g``, else None."""
        if not resumes(g, m, m + 1)[0] or not keeps_events(m, g):
            return None
        one = slice(m, m + 1)
        same_bpm = abs(notated[m] - notated[g]) * (ends[m] - offsets[m]) / notated[m] <= SNAP_TOLERANCE_MS
        if _continues(offsets[g], canonical[g], offsets[one], canonical[one], ends[one])[0]:
            role = LineRole(REDUNDANT, 'same_grid')
        elif same_bpm:
            role = LineRole(MUSICAL, 'phase')
        else:
            return None
        if same_bpm and ends[m] - offsets[m] < bar_ms[m] - SNAP_TOLERANCE_MS:
            return LineRole(EXPRESSIVE, 'short_bar')
        return role

    grid = 0
    roles[kept[0]] = LineRole(MUSICAL, 'first')
    k = 1
    while k < len(kept):
        role = follow(k, grid)
        if role is None:
            on_grid = snap_denominators(events[first_event[k]:], offsets[grid], canonical[grid])
            off = np.flatnonzero(~np.isin(on_grid, BINARY_DENOMINATORS))
            stop = len(kept) if not len(off) else int(np.searchsorted(offsets, events[first_event[k] + off[0]],
                                                                      side='right'))
            back = k + 1 + np.flatnonzero(resumes(grid, k + 1, stop))
            j = next((int(m) for m in back if follow(int(m), grid) is not None), None)
            if j is None and not len(off):
                j = len(kept)
            if j is not None:
                stay = _bits(on_grid[:first_event[j] - first_event[k]])
                own = sum(_bits(snap_denominators(events[first_event[m]:first_event[m + 1]], offsets[m],
                                                  canonical[m])) for m in range(k, j))
                if stay <= own:
                    for m in range(k, j):
                        roles[kept[m]] = LineRole(EXPRESSIVE, 'detour')
                    k = j
                    continue
            role = LineRole(MUSICAL, 'tempo')
        roles[kept[k]] = role
        if role.role == MUSICAL:
            grid = k
        k += 1
    return roles  # type: ignore[return-value]


def musical_grid(lines: Sequence[RedLine], event_times_ms) -> tuple[BeatGrid, list[LineRole]]:
    """The evaluation grid of a chart and the role of each of its red lines."""
    roles = classify(lines, event_times_ms)
    segments, bars = [], []
    for line, role in zip(lines, roles):
        if role.role == MUSICAL:
            segments.append(Segment(line.offset_ms, line.beat_length_ms, line.meter))
        if role.role in (MUSICAL, REDUNDANT) and not (bars and line.omit_first_barline):
            bars.append(BarStart(line.offset_ms, line.meter))
    if not segments:
        raise ValueError('No red line with a plausible BPM')
    return BeatGrid(tuple(segments), tuple(bars)), roles
