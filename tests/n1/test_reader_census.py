"""E3's head rules (scripts/n1/reader_census.py) on charts that break the pair rule."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip('ensomi_n')
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'n1'))

import reader_census as rc  # noqa: E402
from ensomi_n.labels import read_osu, reference_from_charts  # noqa: E402

BEAT = 60000.0 / 130.0


def _write(path: Path, red, heads) -> Path:
    lines = ['osu file format v14', '', '[General]', 'AudioFilename: a.mp3', 'Mode: 3', '', '[Metadata]',
             'Title:t', 'Version:v', '', '[Difficulty]', 'CircleSize:4', '', '[TimingPoints]']
    lines += [f'{o!r},{b!r},4,1,0,100,1,0' for o, b in red]
    lines += ['', '[HitObjects]'] + [f'64,192,{t!r},1,0,0:0:0:0:' for t in heads]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def _rules(tmp_path, shift):
    # 130 BPM with a red line on every bar at an integer offset, as mappers write them, and heads at integer
    # offsets on the beats; ``shift`` moves the line of bar 10 later, holding its bar.
    red = [(float(round(k * 4 * BEAT)) + (shift if k == 10 else 0.0), BEAT) for k in range(30)]
    heads = np.round(np.arange(120) * BEAT)
    path = _write(tmp_path / 'c.osu', red, heads)
    ref = reference_from_charts([read_osu(path)], 'c')
    h = ref.heads[(ref.heads >= ref.lo_ms) & (ref.heads <= ref.hi_ms)]
    r_t = rc.r_lines(path, ref.lo_ms, ref.hi_ms)[0]
    n_t = ref.lines().t
    return h, rc.changed_mask(h, n_t, r_t), rc.changed_matched(h, n_t, r_t)


def test_lines_within_a_millisecond_change_no_head(tmp_path):
    h, pair, matched = _rules(tmp_path, 0.0)
    assert matched.sum() == 0
    assert pair.sum() == 30          # the pair rule flags the head on each bar line: a head between matching lines


def test_a_line_moved_25_ms_changes_the_heads_around_it(tmp_path):
    h, pair, matched = _rules(tmp_path, 25.0)
    # Bar 10's line moves 25 ms; R keeps the grid in force under it. N's previous segment still places its
    # beat at beat 40 (25 ms before the moved line), so the heads changed are exactly those of bar 10.
    np.testing.assert_array_equal(np.flatnonzero(matched), [40, 41, 42, 43])
