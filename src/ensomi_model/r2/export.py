"""Decisions -> replay-checked rows -> hit objects -> ``.osu`` text, with legality required clean."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..evaluation.legality import violations
from ..osu_core.hitobjects import ManiaHitObjectKind
from .common import ContractError
from .state import replay_decisions, rows_to_objects


def decisions_to_objects(head_ms, song_ms, actions, gap):
    state, rows = replay_decisions(head_ms, song_ms, actions, gap)
    if not state.finalized:
        raise ContractError('Decisions do not reach EOS')
    return rows_to_objects(rows), rows


def _fmt(t: float) -> str:
    return str(int(t)) if float(t).is_integer() else repr(float(t))


def minimal_header(grid_segments, title='R2 generation') -> str:
    lines = ['osu file format v14', '[General]', 'Mode:3', '[Metadata]', f'Title:{title}', 'Artist:Ensomi',
             'Creator:Ensomi', 'Version:R2', '[Difficulty]', 'CircleSize:4', '[TimingPoints]']
    for offset, beat_length, meter in np.asarray(grid_segments):
        lines.append(f'{_fmt(offset)},{repr(float(beat_length))},{int(meter)},2,0,100,1,0')
    lines.append('[HitObjects]')
    return '\n'.join(lines) + '\n'


def osu_text(objects, header: str) -> str:
    body = []
    for o in sorted(objects, key=lambda o: (o.start_time_ms, o.lane)):
        x = 64 + 128 * o.lane
        if o.kind is ManiaHitObjectKind.HOLD:
            body.append(f'{x},192,{_fmt(o.start_time_ms)},128,0,{_fmt(o.end_time_ms)}:0:0:0:0:')
        else:
            body.append(f'{x},192,{_fmt(o.start_time_ms)},1,0,0:0:0:0:')
    return header + '\n'.join(body) + '\n'


def export_chart(head_ms, song_ms, actions, gap, destination: Path | None, header: str):
    objects, rows = decisions_to_objects(head_ms, song_ms, actions, gap)
    bad = violations(objects, song_span=(0.0, float(song_ms)))
    if bad:
        raise ContractError(f'Exported chart is illegal: {bad}')
    text = osu_text(objects, header)
    if destination is not None:
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_text(text)
    return objects, text
