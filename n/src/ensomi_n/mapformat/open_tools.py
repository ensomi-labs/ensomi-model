"""Open-tool tempo and beat outputs read as music maps.

The open baselines (Ardour's bundled Vamp trackers and MiniBPM, openDAW's
TempoCNN) are run outside this package; their converters write one
normalized JSON per tool run and song, schema ``ensomi-open-baseline-v1``:

- ``arm``, ``tool``, ``song_id``: provenance; ``tool_version`` and
  ``provenance`` are kept as they are.
- ``status``: ``ok`` (the tool stated beats or a tempo), ``empty`` (the tool
  ran and stated nothing, such as MiniBPM's 0 for audio it cannot read; scored
  as an empty prediction, since that is what the user gets) or ``failed``
  (the run did not complete: a crash, a timeout, a refused input; never
  scored, and `read_open_map` raises `OpenRunFailed` with ``error``).
- ``kind``: ``beats`` or ``tempo``.
  - ``beats``: ``beats_s``, beat times in seconds from sample zero of the
    canonical decode, strictly increasing; ``downbeats``, indices into
    ``beats_s`` of bar starts, or null when the tool states no bars.
  - ``tempo``: ``bpm``, one tempo in quarter notes per minute, and ``grid``,
    how the host lays that tempo on the audio: ``anchor_s`` (the time of beat
    one) and ``meter`` (beats per bar from the anchor). A tool that states no
    phase has its grid from the host's convention, never from a chart.
- ``audio``: the decode the tool read (``pcm_sha256``, ``frames``,
  ``sample_rate_hz``, ``channels``, ``channel_used``).

**Clock.** As for Smart Tempo maps (`ensomi_n.mapformat.logic`), the times are
on the canonical decode; the map is returned on the ``decoded`` clock after
adding ``lead_ms``, the osu! clock rule's lead for that decode. Scoring needs
the one decoded-to-osu! constant; nothing here fits an offset.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from . import MusicMap
from ..timing import Segment

SCHEMA = 'ensomi-open-baseline-v1'
KINDS = ('beats', 'tempo')
STATUSES = ('ok', 'empty', 'failed')

__all__ = ['SCHEMA', 'KINDS', 'STATUSES', 'OpenRunFailed', 'load_open', 'read_open_map', 'stated_bpm']


class OpenRunFailed(ValueError):
    """The tool run did not complete, so there is no prediction to score."""


def load_open(path: str | os.PathLike) -> dict:
    """The normalized output's JSON, after checking its schema, status and kind."""
    with open(path, 'r', encoding='utf-8') as handle:
        data = json.load(handle)
    if data.get('schema') != SCHEMA:
        raise ValueError(f'{path}: schema {data.get("schema")!r}, expected {SCHEMA!r}')
    if data.get('status') not in STATUSES:
        raise ValueError(f'{path}: status {data.get("status")!r}, expected one of {STATUSES}')
    if data['status'] != 'failed' and data.get('kind') not in KINDS:
        raise ValueError(f'{path}: kind {data.get("kind")!r}, expected one of {KINDS}')
    return data


def stated_bpm(data: dict) -> float:
    """The one tempo a ``tempo`` output states; NaN for any other output."""
    if data.get('status') != 'ok' or data.get('kind') != 'tempo':
        return math.nan
    return float(data['bpm'])


def _beats_map(data: dict, lead_ms: float, model: dict, extra: dict) -> MusicMap:
    t = np.asarray([float(x) for x in data.get('beats_s') or []], dtype=np.float64)
    if not len(t):
        raise ValueError('an ok beats output needs beats_s; a run that found none has status "empty"')
    if not np.all(np.isfinite(t)) or (len(t) > 1 and not np.all(np.diff(t) > 0)):
        raise ValueError('beats_s must be finite and strictly increasing')
    down = data.get('downbeats')
    if down is not None:
        down = np.asarray(down, dtype=np.int64)
        if len(down) and (np.any(np.diff(down) <= 0) or down[0] < 0 or down[-1] >= len(t)):
            raise ValueError('downbeats must be increasing indices into beats_s')
    return MusicMap(beats=1000.0 * t + lead_ms, downbeats=down, model=model, clock='decoded', extra=extra)


def _tempo_map(data: dict, lead_ms: float, model: dict, extra: dict) -> MusicMap:
    bpm = float(data['bpm'])
    grid = data.get('grid') or {}
    if not (math.isfinite(bpm) and bpm > 0):
        raise ValueError(f'bpm {bpm!r}: an ok tempo output states a positive tempo; a 0 is status "empty"')
    if 'anchor_s' not in grid or 'meter' not in grid:
        raise ValueError('a tempo output needs grid.anchor_s and grid.meter: how the host lays the tempo on the audio')
    anchor, meter = float(grid['anchor_s']), int(grid['meter'])
    if not math.isfinite(anchor) or meter < 1:
        raise ValueError(f'grid {grid!r}: anchor_s must be finite and meter at least 1')
    seg = Segment(1000.0 * anchor + lead_ms, 60000.0 / bpm, meter, True)
    return MusicMap(segments=(seg,), model=model, clock='decoded', extra=extra)


def read_open_map(path: str | os.PathLike, *, lead_ms: float = 0.0) -> MusicMap:
    """A normalized open-tool output as a music map on the ``decoded`` clock (module docstring).

    ``lead_ms`` is the osu! clock rule's lead for the canonical decode the
    tool read. Raises `OpenRunFailed` for a ``failed`` run and ``ValueError``
    for an output that breaks the schema. An ``empty`` run is an empty
    prediction (``beats: []``), which scores 0 on every share.
    """
    if not math.isfinite(lead_ms) or lead_ms < 0:
        raise ValueError(f'lead_ms must be a finite lead of 0 or more, got {lead_ms!r}')
    data = load_open(path)
    if data['status'] == 'failed':
        raise OpenRunFailed(f'{path}: the {data.get("arm")} run failed: {data.get("error")}')
    model = dict(name=str(data.get('arm', '')), tool=str(data.get('tool', '')),
                 version=str(data.get('tool_version', '')))
    extra = {'open': dict(song_id=data.get('song_id'), arm=data.get('arm'), kind=data.get('kind'),
                          status=data['status'], lead_ms=float(lead_ms), source=str(path))}
    if data['status'] == 'empty':
        return MusicMap(beats=np.zeros(0), model=model, clock='decoded', extra=extra)
    if data['kind'] == 'beats':
        return _beats_map(data, float(lead_ms), model, extra)
    return _tempo_map(data, float(lead_ms), model, extra)
