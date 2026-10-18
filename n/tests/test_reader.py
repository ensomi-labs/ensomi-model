"""The `.osu` reader keeps what the mapper wrote: resets, odd meters, fractional boundaries."""
from __future__ import annotations

import numpy as np
import pytest

from ensomi_n.labels import find_sets, read_osu, read_set
from ensomi_n.timing import Segment


def test_timing_points_and_hit_objects(osu_writer):
    path = osu_writer('set/a.osu', [
        (0, 500.0, 4), (4000, 400.0, 4), (4000, 500.0, 3), (5000, 500.0, 0), (5500.5, 461.538461538, 7, True),
        (9000, 0.001, 4), (9500, 500.0, 4)],
        heads=[100, 100, 600, 9200],
        holds=[(1100, 1600)],
        timing_lines=['1000,-50,4,1,0,100,0,0', '6000,375', '7000,-100'])
    chart = read_osu(path)
    assert chart.audio == 'audio.mp3' and chart.mode == 3 and chart.version == 'Hard' and chart.set_id is None
    assert chart.segments() == (
        Segment(0.0, 500.0, 4), Segment(4000.0, 500.0, 3), Segment(5000.0, 500.0, 4),
        Segment(5500.5, 461.538461538, 7, bar_at_start=False), Segment(6000.0, 375.0, 4),
        Segment(9000.0, 0.001, 4), Segment(9500.0, 500.0, 4))
    np.testing.assert_array_equal(chart.heads, [100.0, 600.0, 1100.0, 9200.0])
    assert chart.first_head_ms == 100.0 and chart.last_head_ms == 9200.0
    assert chart.last_end_ms == 9200.0
    assert not chart.segments()[5].plausible


def test_hold_end_counts_for_last_end(osu_writer):
    chart = read_osu(osu_writer('b.osu', [(0, 500.0)], heads=[0], holds=[(1000, 3000)]))
    assert chart.last_head_ms == 1000.0 and chart.last_end_ms == 3000.0


def test_heads_in_an_implausible_span_weigh_nothing(osu_writer):
    path = osu_writer('c.osu', [(0, 500.0), (2000, 0.001), (3000, 500.0)], heads=[0, 1000, 2500, 3000, 4000])
    ref = read_set([path])
    lines = ref.lines()
    w = ref.weights(lines)
    assert w.sum() == pytest.approx(4.0)
    assert not np.any((lines.t > 2000.0) & (lines.t < 3000.0))


def test_set_unions_heads_and_takes_timing_from_the_fullest_difficulty(osu_writer):
    a = osu_writer('s/easy.osu', [(0, 500.0)], heads=[0, 1000, 2000], version='Easy')
    b = osu_writer('s/hard.osu', [(0, 500.0)], heads=[0, 250, 500, 1000, 3000], version='Hard')
    ref = read_set([a, b], key='s')
    np.testing.assert_array_equal(ref.heads, [0.0, 250.0, 500.0, 1000.0, 2000.0, 3000.0])
    assert ref.lo_ms == 0.0 and ref.hi_ms == 3000.0
    assert ref.meta['timing_chart'].endswith('hard.osu') and ref.meta['timing_consistent'] is True
    c = osu_writer('s/insane.osu', [(10, 500.0)], heads=[0], version='Insane')
    assert read_set([a, b, c]).meta['timing_consistent'] is False


def test_seven_eight_with_resets_reads_every_downbeat(osu_writer):
    # The F2 fixture: 8 bars of 7/8 at quarter = 120, a reset on every downbeat 1750 ms apart.
    heads = [k * 1750 + j * 250 for k in range(8) for j in range(7)]
    ref = read_set([osu_writer('r.osu', [(k * 1750, 500.0, 4) for k in range(8)], heads=heads)])
    bars = ref.timing.bar_starts()
    np.testing.assert_array_equal(bars[(bars >= 0) & (bars < 14000)], np.arange(8) * 1750.0)
    assert len(ref.segments) == 8


def test_alternating_meters_read_every_line(osu_writer):
    lines, t = [], 0
    for k in range(8):
        meter = 4 if k % 2 == 0 else 3
        lines.append((t, 250.0, meter))
        t += meter * 250
    ref = read_set([osu_writer('alt.osu', lines, heads=[0, t - 250])])
    bars = ref.timing.bar_starts()
    np.testing.assert_array_equal(bars[(bars >= 0) & (bars < t)], [0, 1000, 1750, 2750, 3500, 4500, 5250, 6250])


def test_find_sets_keys(tmp_path, osu_writer):
    osu_writer('root/0/111/a.osu', [(0, 500.0)], heads=[0])
    osu_writer('root/0/111/b.osu', [(0, 500.0)], heads=[0])
    osu_writer('root/0/222/a.osu', [(0, 500.0)], heads=[0], audio='x.mp3')
    osu_writer('root/0/222/b.osu', [(0, 500.0)], heads=[0], audio='Y.ogg')
    sets = find_sets([tmp_path / 'root'])
    assert sorted(sets) == ['0/111', '0/222/x', '0/222/y']
    assert len(sets['0/111']) == 2
    direct = find_sets([tmp_path / 'root/0/111/a.osu'])
    assert list(direct) == ['111']


def test_set_without_red_lines_is_refused(osu_writer):
    with pytest.raises(ValueError, match='no red line'):
        read_set([osu_writer('n.osu', [], heads=[0])])


def test_read_set_takes_a_directory(osu_writer, tmp_path):
    osu_writer('dirset/a.osu', [(0, 500.0)], heads=[0, 500])
    osu_writer('dirset/b.osu', [(0, 500.0)], heads=[1000])
    ref = read_set(tmp_path / 'dirset')
    assert ref.key == 'dirset'
    np.testing.assert_array_equal(ref.heads, [0.0, 500.0, 1000.0])
    assert len(ref.meta['difficulties']) == 2
