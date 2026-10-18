"""The music map: the file an N model writes for one song, version 1.

A music map is JSON:

```json
{
  "format": "ensomi-music-map",
  "version": 1,
  "model": {"name": "n1-a", "version": "0.1.0"},
  "clock": "osu",
  "audio": "audio.mp3",
  "beats": [512.0, 1012.0, 1512.0],
  "downbeats": [0],
  "confidence": [0.97, 0.95, 0.4],
  "subdivision": [[0.1, 0.8], [0.0, 0.1], null],
  "segments": [{"start_ms": 512.0, "beat_ms": 500.0, "meter": 4}],
  "heads": []
}
```

- ``beats``: notated beat times in ms on ``clock``, strictly increasing, at
  the model's own tempo level. Required unless ``segments`` is given.
- ``downbeats``: indices into ``beats`` of bar starts. Absent, bars come from
  ``segments``; absent both, the map states no bars and every bar term
  scores 0.
- ``confidence``: per beat, the predicted probability that a mapper's beat
  line lies within 20 ms of it. ``null`` entries, or the field's absence,
  mean no confidence.
- ``subdivision``: per beat, ``[p_triple, p_duple]``: a triple subdivision,
  and a duple one finer than the half. Both at 0.5 or more mean mixed,
  neither plain. ``null`` means none.
- ``segments``: the model's red lines, ``start_ms``, ``beat_ms`` and
  ``meter`` (default 4), optionally ``bar_at_start`` (default true). With
  ``beats``, they give each beat its exact notated beat length and mark
  segment starts; without, they are the timing itself, applied over the
  scored range as osu! applies red lines.
- ``heads``: optional note heads the model placed; when ``subdivision`` is
  absent, each beat's class comes from them, as for a reference.
- ``clock``: ``osu`` for osu!'s timing convention, the charts' clock, or
  ``decoded`` for the decoded audio. A decoded map is scored only after one
  constant, measured once, converts it (``decoded_to_osu_ms``); the harness
  never fits it per song.
- ``model`` and ``audio`` are provenance. Unknown fields are kept in
  ``extra``.

`MusicMap.from_osu` reads any `.osu` as a map (its red lines and heads), so
R2's exported charts and second mappers' sets score the same way.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..timing import Segment, Timing

FORMAT = 'ensomi-music-map'
VERSION = 1
CLOCKS = ('osu', 'decoded')

__all__ = ['FORMAT', 'VERSION', 'MusicMap', 'load']


def _optional_per_beat(values, n: int, name: str, width: int | None = None) -> np.ndarray | None:
    if values is None:
        return None
    if len(values) != n:
        raise ValueError(f'{name} needs one entry per beat: {len(values)} for {n} beats')
    if width is None:
        arr = np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)
        bad = np.isfinite(arr) & ((arr < 0) | (arr > 1))
    else:
        arr = np.array([[np.nan] * width if v is None else [float(x) for x in v] for v in values],
                       dtype=np.float64).reshape(n, width)
        bad = np.isfinite(arr) & ((arr < 0) | (arr > 1))
    if np.any(bad):
        raise ValueError(f'{name} must be probabilities in [0, 1]')
    return arr


@dataclass
class MusicMap:
    """A music map in memory. See the module docstring for the fields.

    Arrays are numpy; ``confidence`` holds NaN where none is given, and
    ``subdivision`` is an ``(n, 2)`` array with NaN rows likewise.
    """
    beats: np.ndarray | None = None
    downbeats: np.ndarray | None = None
    confidence: np.ndarray | None = None
    subdivision: np.ndarray | None = None
    segments: tuple[Segment, ...] | None = None
    heads: np.ndarray | None = None
    model: dict = field(default_factory=dict)
    clock: str = 'osu'
    audio: str = ''
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.clock not in CLOCKS:
            raise ValueError(f'clock must be one of {CLOCKS}, got {self.clock!r}')
        if self.beats is None and not self.segments:
            raise ValueError('A music map needs beats or segments; an empty prediction has "beats": []')
        if self.beats is not None:
            self.beats = np.asarray(self.beats, dtype=np.float64).reshape(-1)
            n = len(self.beats)
            if not np.all(np.isfinite(self.beats)) or (n > 1 and not np.all(np.diff(self.beats) > 0)):
                raise ValueError('beats must be finite and strictly increasing')
            if self.downbeats is not None:
                d = np.asarray(self.downbeats)
                self.downbeats = np.flatnonzero(d) if d.dtype == bool else d.astype(np.int64).reshape(-1)
                if np.any((self.downbeats < 0) | (self.downbeats >= n)):
                    raise ValueError('downbeats must index beats')
            self.confidence = _optional_per_beat(self.confidence, n, 'confidence')
            self.subdivision = _optional_per_beat(self.subdivision, n, 'subdivision', 2)
        elif self.downbeats is not None or self.confidence is not None or self.subdivision is not None:
            raise ValueError('downbeats, confidence and subdivision index beats, so they need beats')
        if self.segments is not None:
            self.segments = tuple(sorted(self.segments, key=lambda s: s.start_ms))
            starts = [s.start_ms for s in self.segments]
            if any(b <= a for a, b in zip(starts, starts[1:])):
                raise ValueError('segment starts must increase strictly')
        if self.heads is not None:
            self.heads = np.unique(np.asarray(self.heads, dtype=np.float64))

    # Construction

    @classmethod
    def from_beats(cls, beats, downbeats=None, confidence=None, subdivision=None, *,
                   segments: Sequence[Segment] | None = None, model: str = '', version: str = '',
                   clock: str = 'osu', **extra) -> 'MusicMap':
        """A map from in-memory arrays, as a training loop holds them."""
        info = {k: v for k, v in (('name', model), ('version', version)) if v}
        return cls(np.asarray(beats, dtype=np.float64), downbeats, confidence, subdivision,
                   tuple(segments) if segments else None, None, info, clock, extra=extra)

    @classmethod
    def from_osu(cls, path: str | os.PathLike, model: str = '', version: str = '') -> 'MusicMap':
        """Any `.osu` read as a map: its red lines as segments, its heads as heads."""
        from ..labels import read_osu
        chart = read_osu(path)
        if not chart.red_lines:
            raise ValueError(f'{path}: no red line')
        info = {k: v for k, v in (('name', model), ('version', version)) if v}
        info.setdefault('source', str(path))
        return cls(segments=chart.segments(), heads=chart.heads, model=info, audio=chart.audio)

    @classmethod
    def from_json(cls, data: dict) -> 'MusicMap':
        if data.get('format') != FORMAT:
            raise ValueError(f'Not a music map: format is {data.get("format")!r}, expected {FORMAT!r}')
        if int(data.get('version', 0)) != VERSION:
            raise ValueError(f'Music map version {data.get("version")!r} is not supported (expected {VERSION})')
        known = {'format', 'version', 'model', 'clock', 'audio', 'beats', 'downbeats', 'confidence',
                 'subdivision', 'segments', 'heads'}
        segments = None
        if data.get('segments') is not None:
            segments = tuple(Segment(float(s['start_ms']), float(s['beat_ms']), int(s.get('meter', 4)),
                                     bool(s.get('bar_at_start', True))) for s in data['segments'])
        return cls(beats=data.get('beats'), downbeats=data.get('downbeats'), confidence=data.get('confidence'),
                   subdivision=data.get('subdivision'), segments=segments, heads=data.get('heads'),
                   model=dict(data.get('model') or {}), clock=data.get('clock', 'osu'),
                   audio=data.get('audio', ''), extra={k: v for k, v in data.items() if k not in known})

    def to_json(self) -> dict:
        out: dict[str, Any] = {'format': FORMAT, 'version': VERSION, 'model': self.model, 'clock': self.clock}
        if self.audio:
            out['audio'] = self.audio
        if self.beats is not None:
            out['beats'] = [float(x) for x in self.beats]
        if self.downbeats is not None:
            out['downbeats'] = [int(x) for x in self.downbeats]
        if self.confidence is not None:
            out['confidence'] = [None if not math.isfinite(x) else float(x) for x in self.confidence]
        if self.subdivision is not None:
            out['subdivision'] = [None if not np.all(np.isfinite(r)) else [float(r[0]), float(r[1])]
                                  for r in self.subdivision]
        if self.segments is not None:
            out['segments'] = [dict(start_ms=s.start_ms, beat_ms=s.beat_ms, meter=s.meter,
                                    **({} if s.bar_at_start else {'bar_at_start': False})) for s in self.segments]
        if self.heads is not None:
            out['heads'] = [float(x) for x in self.heads]
        out.update(self.extra)
        return out

    def dump(self, path: str | os.PathLike) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=1) + '\n', encoding='utf-8')

    # Evaluation views

    def on_osu_clock(self, decoded_to_osu_ms: float | None = None) -> 'MusicMap':
        """This map on osu!'s clock. A decoded map needs the one measured constant."""
        if self.clock == 'osu':
            return self
        if decoded_to_osu_ms is None:
            raise ValueError('This map is on the decoded clock; scoring it against charts needs the one '
                             'decoded-to-osu constant (decoded_to_osu_ms), measured once, never fitted per song')
        c = float(decoded_to_osu_ms)
        segs = None if self.segments is None else tuple(
            Segment(s.start_ms + c, s.beat_ms, s.meter, s.bar_at_start) for s in self.segments)
        return MusicMap(None if self.beats is None else self.beats + c, self.downbeats, self.confidence,
                        self.subdivision, segs, None if self.heads is None else self.heads + c,
                        dict(self.model), 'osu', self.audio, dict(self.extra))

    def timing(self, lo_ms: float, hi_ms: float) -> Timing:
        """The map's notated beats for scoring the range ``[lo_ms, hi_ms]``.

        A beat list is returned whole. Segments alone are applied over the
        range with margin, exactly as `Timing.from_segments` applies red lines.
        """
        if self.beats is not None:
            if not len(self.beats):
                return Timing.empty()
            return Timing.from_beats(self.beats, self.downbeats, self.segments)
        return Timing.from_segments(self.segments, lo_ms, hi_ms)

    def beat_confidence(self, timing: Timing) -> np.ndarray | None:
        """Confidence per notated beat of ``timing`` (from `timing`); None when the map has none."""
        if self.confidence is None or not np.any(np.isfinite(self.confidence)):
            return None
        return self.confidence

    def red_lines(self, timing: Timing) -> np.ndarray:
        """Times of the map's red lines: its segment starts, or for a bare beat list its tempo-span starts."""
        if self.segments:
            return np.array([s.start_ms for s in self.segments if s.plausible])
        from ..eval.measures import tempo_spans
        spans = tempo_spans(timing)
        return np.array([a for a, _, _ in spans])


def load(path: str | os.PathLike) -> MusicMap:
    """A prediction from a file: a music map (``.json``) or any ``.osu``."""
    path = Path(path)
    if path.suffix.lower() == '.osu':
        return MusicMap.from_osu(path)
    with path.open('r', encoding='utf-8') as handle:
        data = json.load(handle)
    return MusicMap.from_json(data)

