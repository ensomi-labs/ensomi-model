"""Synthetic `.osu` writing for the fixtures."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pytest


def osu_text(red_lines: Sequence[tuple], heads: Iterable[float] = (), *, audio: str = 'audio.mp3',
             version: str = 'Hard', holds: Iterable[tuple[float, float]] = (),
             timing_lines: Iterable[str] = ()) -> str:
    """An osu!mania `.osu` file.

    ``red_lines`` holds ``(offset, beat_ms[, meter[, omit_first_barline]])``;
    ``timing_lines`` adds raw timing-point lines (inherited ones, old formats).
    """
    lines = ['osu file format v14', '', '[General]', f'AudioFilename: {audio}', 'Mode: 3', '',
             '[Metadata]', 'Title:fixture', f'Version:{version}', 'BeatmapSetID:-1', '',
             '[Difficulty]', 'CircleSize:4', '', '[TimingPoints]']
    for r in red_lines:
        offset, beat = r[0], r[1]
        meter = r[2] if len(r) > 2 else 4
        omit = r[3] if len(r) > 3 else False
        lines.append(f'{offset!r},{beat!r},{meter},1,0,100,1,{8 if omit else 0}')
    lines.extend(timing_lines)
    lines += ['', '[HitObjects]']
    for t in heads:
        lines.append(f'64,192,{t!r},1,0,0:0:0:0:')
    for start, end in holds:
        lines.append(f'192,192,{start!r},128,0,{end!r}:0:0:0:0:')
    return '\n'.join(lines) + '\n'


def write_osu(path: Path, red_lines, heads=(), **kwargs) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(osu_text(red_lines, heads, **kwargs), encoding='utf-8')
    return path


@pytest.fixture
def osu_writer(tmp_path):
    def make(name: str, red_lines, heads=(), **kwargs) -> Path:
        return write_osu(tmp_path / name, red_lines, heads, **kwargs)
    return make
