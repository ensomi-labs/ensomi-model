"""Logic Pro Smart Tempo maps read as music maps.

The capture procedure (``docs/guides/logic_tempo_baseline.md``) saves each
Smart Tempo analysis as a normalized JSON map, schema
``logic-smart-tempo-native-v1``. The fields read here:

- ``beats``: one record per native beat marker, with ``ordinal`` (zero-based)
  and ``audio_file_seconds``, the native time in seconds from sample zero of
  the canonical full FFmpeg decode of the source. Records may also carry the
  editor's accessibility label (``raw_ax_label``: ``"Downbeat, Bar 3"`` or
  ``"Bar 3 Beat 9"``, where the beat number is the cumulative zero-based
  ordinal) and ``downbeat_from_ax``, filled only when the whole label list
  matched the native times.
- ``native_downbeat_time_seconds``: the native downbeat anchor (``db_time``).
- ``time_signature_events``: ``audio_file_seconds`` and ``signature`` (``n/d``).
- ``tempo_events``, ``mode`` and ``song_id`` are kept as provenance.

**Bars.** The map states one downbeat and the signatures. Its bars are read
as the editor numbers them: the anchor beat starts bar 1, beats before it
belong to bar 0 (a pickup), and from the anchor each bar holds the
numerator's count of beats, the signature in force at the bar's first beat
setting it. `ax_check` compares this reading with the editor's labels,
where a map carries them; that comparison is the reading's absolute check.
Only quarter-note signatures (``n/4``) are read: what a beat marker is under
another denominator is not established, so other denominators are refused.

**Clock.** The times are on the canonical decode, not on osu!'s clock. The
map is returned on the ``decoded`` clock after adding ``lead_ms``, the lead
of N's osu! clock rule for that decode (`ensomi_n.features.clock.file_lead`
with decoder ``'ffmpeg'``: 0 or 529 samples, read from the file's header), so
that it lies on the decoded clock osu! plays. Scoring against charts then
needs the one decoded-to-osu! constant (`MusicMap.on_osu_clock`); nothing
here fits an offset.
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import numpy as np

from . import MusicMap

SCHEMA = 'logic-smart-tempo-native-v1'
#: A downbeat anchor or a signature event this close to a beat (seconds) is on it.
ON_BEAT_S = 5e-4
_LABEL = re.compile(r'^\s*(?:(Downbeat),\s*)?Bar\s+(-?\d+)(?:\s+Beat\s+(\d+))?\s*$')

__all__ = ['SCHEMA', 'load_logic', 'bar_numbers', 'downbeats', 'ax_check', 'read_logic_map', 'summarize']


def load_logic(path: str | os.PathLike) -> dict:
    """The normalized map's JSON, after checking its schema and its beat list."""
    with open(path, 'r', encoding='utf-8') as handle:
        data = json.load(handle)
    if data.get('schema') != SCHEMA:
        raise ValueError(f'{path}: schema {data.get("schema")!r}, expected {SCHEMA!r}')
    beats = data.get('beats') or []
    if not beats:
        raise ValueError(f'{path}: no beats')
    ordinals = [b.get('ordinal') for b in beats]
    if ordinals != list(range(len(beats))):
        raise ValueError(f'{path}: beat ordinals are not 0..{len(beats) - 1} in order')
    return data


def _times(data: dict) -> np.ndarray:
    t = np.array([float(b['audio_file_seconds']) for b in data['beats']], dtype=np.float64)
    if not np.all(np.isfinite(t)) or (len(t) > 1 and not np.all(np.diff(t) > 0)):
        raise ValueError('Smart Tempo beat times must be finite and strictly increasing')
    return t


def _signatures(data: dict) -> list[tuple[float, int, int]]:
    out = []
    for ev in data.get('time_signature_events') or []:
        n, d = (int(x) for x in str(ev['signature']).split('/'))
        if n < 1:
            raise ValueError(f'signature {ev["signature"]!r}: no beats in a bar')
        if d != 4:
            raise ValueError(f'signature {ev["signature"]!r}: only n/4 signatures are read (what a beat marker is '
                             'under another denominator is not established)')
        out.append((float(ev['audio_file_seconds']), n, d))
    if not out:
        raise ValueError('a Smart Tempo map without a time signature states no bars')
    return sorted(out)


def _on_beat(t: np.ndarray, x: float) -> int:
    i = int(np.argmin(np.abs(t - x)))
    if abs(t[i] - x) > ON_BEAT_S:
        raise ValueError(f'{x!r} s is {abs(t[i] - x) * 1000:.3f} ms from the nearest beat')
    return i


def bar_numbers(data: dict) -> np.ndarray:
    """Bar number of every beat, as the editor numbers bars (module docstring): 0 before the anchor, 1 from it."""
    t = _times(data)
    anchor = _on_beat(t, float(data['native_downbeat_time_seconds']))
    sigs = _signatures(data)
    bars = np.zeros(len(t), dtype=np.int64)
    i, bar = anchor, 1
    while i < len(t):
        n = sigs[0][1]
        for when, num, _ in sigs:
            if when <= t[i] + ON_BEAT_S:
                n = num
        bars[i:i + n] = bar
        i, bar = i + n, bar + 1
    return bars


def downbeats(data: dict) -> np.ndarray:
    """Indices of the beats that start a bar (bar 1 on): the first beat of each bar number."""
    bars = bar_numbers(data)
    first = np.flatnonzero(np.diff(np.concatenate([[-1], bars])) != 0)
    return first[bars[first] >= 1]


def ax_check(data: dict) -> dict:
    """The bar reading against the editor's own labels, where the map carries them.

    Returns the number of labelled beats and of mismatches: downbeat flags,
    bar numbers and cumulative beat numbers. ``labels`` is 0 for a map whose
    accessibility list was not captured.
    """
    bars = bar_numbers(data)
    down = np.zeros(len(bars), dtype=bool)
    down[downbeats(data)] = True
    n = bad_down = bad_bar = bad_beat = unread = 0
    for i, b in enumerate(data['beats']):
        label = b.get('raw_ax_label')
        if not label:
            continue
        m = _LABEL.match(str(label))
        if not m:
            unread += 1
            continue
        n += 1
        is_down, bar, beat = m.group(1) is not None, int(m.group(2)), m.group(3)
        bad_down += is_down != bool(down[i])
        bad_bar += bar != int(bars[i])
        bad_beat += beat is not None and int(beat) != i
    return dict(labels=n, unreadable=unread, downbeat_mismatch=bad_down, bar_mismatch=bad_bar,
                beat_mismatch=bad_beat, ok=bool(n and not (bad_down or bad_bar or bad_beat or unread)))


def read_logic_map(path: str | os.PathLike, *, lead_ms: float = 0.0) -> MusicMap:
    """A normalized Smart Tempo map as a music map on the ``decoded`` clock (module docstring).

    ``lead_ms`` is the osu! clock rule's lead for the canonical FFmpeg decode
    of the source, the same for every map of that source.
    """
    if not math.isfinite(lead_ms) or lead_ms < 0:
        raise ValueError(f'lead_ms must be a finite lead of 0 or more, got {lead_ms!r}')
    data = load_logic(path)
    t = _times(data)
    d = downbeats(data)
    sigs = _signatures(data)
    tempos = [float(e['bpm']) for e in data.get('tempo_events') or []]
    info = dict(song_id=data.get('song_id'), mode=data.get('mode'), schema=SCHEMA, lead_ms=float(lead_ms),
                signatures=[f'{n}/{d_}' for _, n, d_ in sigs],
                downbeat_anchor_ms=1000.0 * float(data['native_downbeat_time_seconds']) + lead_ms,
                tempo_events=len(tempos), source=str(path))
    return MusicMap(beats=1000.0 * t + lead_ms, downbeats=d, model=dict(name='logic-smart-tempo',
                                                                        version=str(data.get('logic_version', '')),
                                                                        mode=str(data.get('mode', ''))),
                    clock='decoded', audio=str(data.get('source_filename', '')), extra={'logic': info})


def summarize(path: str | os.PathLike) -> dict:
    """What a map states, for a report: mode, beats, tempo range, signatures, bars and the label check."""
    data = load_logic(path)
    t = _times(data)
    tempos = np.array([float(e['bpm']) for e in data.get('tempo_events') or []], dtype=np.float64)
    iv = np.diff(t)
    return dict(song_id=data.get('song_id'), mode=data.get('mode'), beats=int(len(t)),
                first_s=float(t[0]), last_s=float(t[-1]),
                median_bpm=float(60.0 / np.median(iv)) if len(iv) else math.nan,
                tempo_min=float(tempos.min()) if len(tempos) else math.nan,
                tempo_max=float(tempos.max()) if len(tempos) else math.nan, tempo_events=int(len(tempos)),
                signatures=[f'{n}/{d}' for _, n, d in _signatures(data)],
                bars=int(len(downbeats(data))), ax=ax_check(data),
                validation=(data.get('validation') or {}).get('status'))
