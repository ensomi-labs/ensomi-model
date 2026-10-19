"""The N line's `.osu` timing reader.

It reads the timing a mapper wrote, without judging it:

- every uninherited (red) timing point is a segment with its offset, beat
  length and meter. Per-bar metronome resets, odd and changing meters,
  fractional segment boundaries and non-integer tempos are all kept;
- inherited (green) points are ignored for timing;
- of several red lines at one offset the last applies, as in osu!;
- a red line with a tempo outside [20, 1000] BPM is kept as a segment that
  places no beats, and heads inside it weigh nothing.

R's reader (`ensomi_model.evaluation.redlines`) classifies lines and drops
resets, short bars and detours; this reader keeps them, and the two are
separate on purpose. It never imports the R line.

A beatmap set is the unit of reference: its difficulties share one timing, and
its heads are the union of theirs (`read_set`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..timing import Reference, Segment

__all__ = ['RedLine', 'Chart', 'read_osu', 'segments_from_red_lines', 'read_set', 'reference_from_charts',
           'find_sets']


@dataclass(frozen=True)
class RedLine:
    """An uninherited timing point as written: offset, beat length, meter and the omit-first-barline flag."""
    offset_ms: float
    beat_ms: float
    meter: int = 4
    omit_first_barline: bool = False


@dataclass(frozen=True)
class Chart:
    """The timing and heads of one `.osu` file.

    ``heads`` holds the distinct head times of every hit object, sorted.
    ``last_end_ms`` is the latest head or explicit end time (spinners and
    mania holds); slider ends, which need slider velocity, count as their head.
    """
    path: Path
    audio: str
    mode: int
    version: str
    set_id: int | None
    red_lines: tuple[RedLine, ...]
    heads: np.ndarray = field(repr=False)
    last_end_ms: float = math.nan

    @property
    def first_head_ms(self) -> float:
        return float(self.heads[0]) if len(self.heads) else math.nan

    @property
    def last_head_ms(self) -> float:
        return float(self.heads[-1]) if len(self.heads) else math.nan

    def segments(self) -> tuple[Segment, ...]:
        return segments_from_red_lines(self.red_lines)


def _int(value: str, default: int) -> int:
    try:
        return int(float(value))
    except ValueError:
        return default


def read_osu(path: str | os.PathLike) -> Chart:
    """Read the timing points, hit objects and audio name of an `.osu` file.

    A timing point is red when its ``uninherited`` field is 1, or, in files
    without that field, when its beat length is positive. A meter below 1
    reads as 4. Raises ``ValueError`` on a timing point or hit object with
    too few fields.
    """
    path = Path(path)
    section = None
    audio, mode, version, set_id = '', 0, '', None
    red: list[RedLine] = []
    heads: list[float] = []
    last_end = -math.inf
    with path.open('r', encoding='utf-8-sig', errors='replace') as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith('//'):
                continue
            if line.startswith('[') and line.endswith(']'):
                section = line[1:-1]
                continue
            if section in ('General', 'Metadata'):
                name, _, value = line.partition(':')
                name, value = name.strip(), value.strip()
                if name == 'AudioFilename':
                    audio = value
                elif name == 'Mode':
                    mode = _int(value, 0)
                elif name == 'Version':
                    version = value
                elif name == 'BeatmapSetID':
                    set_id = _int(value, -1)
                    set_id = set_id if set_id > 0 else None
            elif section == 'TimingPoints':
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 2:
                    raise ValueError(f'{path}: malformed timing point: {line}')
                offset, beat = float(parts[0]), float(parts[1])
                if len(parts) > 6 and parts[6]:
                    uninherited = _int(parts[6], 1) == 1
                else:
                    uninherited = beat > 0
                if not uninherited:
                    continue
                meter = _int(parts[2], 4) if len(parts) > 2 and parts[2] else 4
                effects = _int(parts[7], 0) if len(parts) > 7 and parts[7] else 0
                red.append(RedLine(offset, beat, meter if meter >= 1 else 4, bool(effects & 8)))
            elif section == 'HitObjects':
                parts = line.split(',')
                if len(parts) < 4:
                    raise ValueError(f'{path}: malformed hit object: {line}')
                time = float(parts[2])
                kind = _int(parts[3], 1)
                heads.append(time)
                end = time
                if kind & 8 and len(parts) > 5:
                    end = float(parts[5])
                elif kind & 128 and len(parts) > 5:
                    end = float(parts[5].split(':', 1)[0])
                last_end = max(last_end, end)
    red.sort(key=lambda r: r.offset_ms)
    head_array = np.unique(np.asarray(heads, dtype=np.float64))
    return Chart(path, audio, mode, version, set_id, tuple(red), head_array,
                 last_end if heads else math.nan)


def segments_from_red_lines(lines: Sequence[RedLine]) -> tuple[Segment, ...]:
    """One segment per red-line offset, the last line at an offset applying."""
    by_offset: dict[float, RedLine] = {}
    for line in sorted(lines, key=lambda r: r.offset_ms):
        by_offset[line.offset_ms] = line
    return tuple(Segment(r.offset_ms, r.beat_ms, r.meter, not r.omit_first_barline)
                 for r in by_offset.values())


def _same_timing(a: Chart, b: Chart) -> bool:
    sa, sb = a.segments(), b.segments()
    return len(sa) == len(sb) and all(
        abs(x.start_ms - y.start_ms) <= 1e-6 and abs(x.beat_ms - y.beat_ms) <= 1e-9 * abs(x.beat_ms)
        and x.meter == y.meter and x.bar_at_start == y.bar_at_start for x, y in zip(sa, sb))


def read_set(paths: str | os.PathLike | Iterable[str | os.PathLike], key: str = '') -> Reference:
    """The reference of one beatmap set: its timing and the union of its difficulties' heads.

    ``paths`` is the set's `.osu` files, or a directory whose `.osu` files
    (not recursive) are the set; ``key`` defaults to the directory's name.
    The timing comes from the difficulty with the most distinct heads (ties:
    the first path in sorted order). ``meta['timing_consistent']`` records
    whether every difficulty has the same red lines. The scored range is the
    union's first to last head. Raises ``ValueError`` when no file has a red
    line or a hit object.
    """
    if isinstance(paths, (str, os.PathLike)):
        folder = Path(paths)
        if not folder.is_dir():
            paths = [folder]
        else:
            paths = sorted(folder.glob('*.osu'))
            key = key or folder.name
    charts = [read_osu(p) for p in sorted(Path(p) for p in paths)]
    return reference_from_charts(charts, key)


def reference_from_charts(charts: Sequence[Chart], key: str = '') -> Reference:
    """`read_set` on charts already read, in the order given (`read_set` sorts them by path)."""
    charts = list(charts)
    if not charts:
        raise ValueError(f'No .osu file for set {key!r}')
    timed = [c for c in charts if c.red_lines]
    if not timed:
        raise ValueError(f'Set {key!r}: no red line in any difficulty')
    timing_chart = max(timed, key=lambda c: len(c.heads))
    heads = np.unique(np.concatenate([c.heads for c in charts]))
    if not len(heads):
        raise ValueError(f'Set {key!r}: no hit object in any difficulty')
    meta = dict(timing_chart=str(timing_chart.path), difficulties=[str(c.path) for c in charts],
                timing_consistent=all(_same_timing(timing_chart, c) for c in timed),
                audio=timing_chart.audio, mode=timing_chart.mode)
    return Reference.from_segments(timing_chart.segments(), heads, key=key, meta=meta)


def find_sets(roots: Iterable[str | os.PathLike]) -> dict[str, list[Path]]:
    """Group `.osu` files into sets, keyed for matching predictions.

    A set is the `.osu` files of one directory that name the same audio file,
    compared without case. Its key is the directory's path relative to the
    root it was found under (its name, for a file given directly), with
    ``/<audio stem>`` appended when the directory holds sets for several audio
    files. Raises ``ValueError`` when two sets get one key.
    """
    found: dict[tuple[Path, str], tuple[str, list[Path]]] = {}
    for root in roots:
        root = Path(root)
        files = sorted(root.rglob('*.osu')) if root.is_dir() else [root]
        for f in files:
            audio = read_osu_audio(f)
            if root.is_dir() and f.parent != root:
                rel = f.parent.relative_to(root).as_posix()
            else:
                rel = f.parent.resolve().name
            group = (f.parent.resolve(), audio.casefold())
            found.setdefault(group, (rel, []))[1].append(f)
    per_dir: dict[Path, int] = {}
    for d, _ in found:
        per_dir[d] = per_dir.get(d, 0) + 1
    sets: dict[str, list[Path]] = {}
    for (d, audio), (rel, files) in sorted(found.items(), key=lambda kv: kv[1][0]):
        key = rel if per_dir[d] == 1 else f'{rel}/{Path(audio).stem}'
        if key in sets:
            raise ValueError(f'Two reference sets share the key {key!r}')
        sets[key] = files
    return sets


def read_osu_audio(path: str | os.PathLike) -> str:
    """The ``AudioFilename`` of an `.osu` file, reading only its header."""
    with Path(path).open('r', encoding='utf-8-sig', errors='replace') as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith('AudioFilename'):
                return line.partition(':')[2].strip()
            if line in ('[TimingPoints]', '[HitObjects]'):
                break
    return ''
