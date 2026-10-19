"""Label-only strata on fixtures that can break them: fractional boundaries, 7/8 resets, ramps, the song's end."""
from __future__ import annotations

import numpy as np
import pytest

import json

from ensomi_n.eval.strata import classify_lines, flags, is_rate_edit, recording_strata, same_timing, set_strata
from ensomi_n.labels import read_osu, read_set
from ensomi_n.timing import Segment

B = 500.0  # 120 BPM


def strata(osu_writer, name, red, heads, **kw):
    path = osu_writer(name, red, heads=heads, version=kw.pop('version', 'Hard'))
    ref = read_set([path], key=name)
    return set_strata(ref, [read_osu(path)], **kw)


def test_control_is_only_s0(osu_writer):
    s = strata(osu_writer, 'c.osu', [(0, B, 4)], [k * B for k in range(64)])
    assert set(s.members) == {'S0'} and s.members['S0'] is None
    assert [line.kind for line in s.lines] == ['first']
    assert s.span_heads['S0'] == 64


def test_control_fails_near_80_and_on_rate_edits_and_tags(osu_writer):
    near = strata(osu_writer, 'n.osu', [(0, 60000 / 81.0, 4)], [k * 60000 / 81.0 for k in range(64)])
    assert 'S0' not in near.members and near.detail['S0']['fails'] == ['near_80_160']
    rate = strata(osu_writer, 'r.osu', [(0, B, 4)], [k * B for k in range(64)], rate_edit=True)
    assert rate.detail['S0']['fails'] == ['rate_edit']
    tagged = strata(osu_writer, 't.osu', [(0, B, 4)], [k * B for k in range(64)], tags=['tech/complex snap'])
    assert tagged.detail['S0']['fails'] == ['timing_tag']
    unknown = strata(osu_writer, 'u.osu', [(0, B, 4)], [k * B for k in range(64)], tags=['meta/swing'],
                     tags_known=False)
    assert 'S0' in unknown.members and unknown.detail['S0']['tags_checked'] is False


def test_seven_eight_in_two_notations_is_irregular_meter(osu_writer):
    heads = [k * 1750 + j * 250 for k in range(8) for j in range(7)]
    resets = strata(osu_writer, 'r78.osu', [(k * 1750, B, 4) for k in range(8)], heads)
    assert [line.kind for line in resets.lines] == ['first'] + ['reset'] * 7
    d = resets.detail['S2']
    # Bars 0-6 are counted; the last bar runs past the last head (13,750 ms).
    assert d['counted_bars'] == 7 and d['irregular_bars'] == 7 and d['bar_lengths'] == [3.5]
    assert set(resets.members) == {'S2', 'S9'} and resets.detail['S3']['musical_lines'] == 1
    assert set(recording_strata([resets])) == {'S2', 'S4', 'S9'}
    assert resets.detail['S9']['one_tempo_resets'] == 7
    meter7 = strata(osu_writer, 'm7.osu', [(0, 250.0, 7)], heads)
    assert meter7.detail['S2']['irregular_bars'] == 7 and meter7.detail['S2']['bar_lengths'] == [7.0]
    assert set(recording_strata([meter7])) == {'S2'}


def test_fractional_boundary_cuts_a_bar_and_opens_a_short_section(osu_writer):
    # 10.5 beats at 120 BPM, then 150 BPM: a 5,250 ms first section, a cut bar of 2.5 beats.
    heads = [k * B for k in range(11)] + [5250 + k * 400.0 for k in range(60)]
    s = strata(osu_writer, 'f.osu', [(0, B, 4), (5250, 400.0, 4)], heads)
    assert [line.kind for line in s.lines] == ['first', 'tempo']
    assert s.detail['S2']['cut_bars'] == 1 and s.detail['S2']['changing'] and not s.detail['S2']['irregular']
    assert s.detail['S4']['short_sections'] == 1 and recording_strata([s])['S4'] == {'f.osu': [(0.0, 5250.0)]}
    # 150/120 = 5/4, a metric modulation; the span is the new section up to the last head.
    assert s.detail['S5']['modulations'] == 1 and s.members['S5'] == [(5250.0, heads[-1])]
    assert 'S0' not in s.members and 'S3' not in s.members


def test_ramp_is_s3_and_keeps_s4_out(osu_writer):
    red, t, beat = [], 0.0, B
    for _ in range(6):
        red.append((t, beat, 4))
        t += 8 * beat
        beat /= 1.02
    beat *= 1.02  # the last section keeps the last ramp tempo: its line is redundant
    heads = [h for k, (a, b, _) in enumerate(red) for h in (a + np.arange(8) * b)] + [t + k * beat for k in range(40)]
    red.append((t, beat, 4))
    s = strata(osu_writer, 'ramp.osu', red, heads)
    assert [line.kind for line in s.lines] == ['first'] + ['tempo'] * 5 + ['same']
    assert s.detail['S3']['ramps'] == 1 and not s.detail['S3']['tempo_mapped']
    assert s.members['S3'] == [(0.0, red[5][0] + 8000.0)]
    rec = recording_strata([s])
    assert 'S3' in rec and 'S4' not in rec and s.s4_spans


def test_tempo_mapped_needs_eight_short_sections(osu_writer):
    bpms = [120, 124, 117, 126, 121, 128, 115, 123, 119, 125]
    red = [(k * 4000.0, 60000.0 / bpm, 4) for k, bpm in enumerate(bpms)]
    heads = [a + j * b / 2 for a, b, _ in red for j in range(int(4000 // (b / 2)))]
    s = strata(osu_writer, 'tm.osu', red, heads)
    assert [line.kind for line in s.lines] == ['first'] + ['tempo'] * 9
    assert s.detail['S3']['musical_lines'] == 10 and s.detail['S3']['tempo_mapped'] and s.members['S3'] is None
    few = strata(osu_writer, 'few.osu', red[:5], [h for h in heads if h < 20000])
    assert not few.detail['S3']['tempo_mapped'] and 'S3' not in few.members


def test_short_section_at_the_songs_end(osu_writer):
    heads = [k * B for k in range(120)] + [60000 + k * 400.0 for k in range(12)]
    s = strata(osu_writer, 'end.osu', [(0, B, 4), (60000, 400.0, 4)], heads)
    assert s.s4_spans == [(60000.0, heads[-1])]
    # Without the last head the section would be longer; the span ends at the last head, not past it.
    assert s.hi_ms == heads[-1]


def test_visual_detour_and_snap_detour(osu_writer):
    heads = [k * 250.0 for k in range(64)]
    # A run at 77 BPM from 8,000 to 9,000 ms; heads stay on the 1/2 grid of 120 BPM.
    s = strata(osu_writer, 'vd.osu', [(0, B, 4), (8000, 60000 / 77.0, 4), (9000, B, 4)], heads)
    assert [line.kind for line in s.lines] == ['first', 'detour', 'same']
    assert s.detail['S9']['detour_runs_with_heads'] == 1 and 'S0' not in s.members
    assert s.detail['S2']['changes'] == 0  # bars touching the detour are left out
    # A one-line run at 5/4 of the tempo holding a head off the 120 BPM binary grid, 400 ms long.
    snap = strata(osu_writer, 'sd.osu', [(0, B, 4), (8000, 400.0, 4), (8400, 600.0, 4), (9000, B, 4)],
                  [k * B for k in range(16)] + [8000, 8400, 8700] + [9000 + k * B for k in range(16)])
    assert [line.kind for line in snap.lines] == ['first', 'detour', 'detour', 'same']
    # A long run off the binary grid is a real tempo change, not a detour.
    real = strata(osu_writer, 'rt.osu', [(0, B, 4), (8000, B * 4 / 3, 4), (24000, B, 4)],
                  [k * B for k in range(16)] + [8000 + k * B * 2 / 3 for k in range(24)]
                  + [24000 + k * B for k in range(16)])
    assert [line.kind for line in real.lines] == ['first', 'tempo', 'tempo']


def test_half_time_section_is_a_level_change_not_a_detour(osu_writer):
    heads = [k * B for k in range(64)]
    s = strata(osu_writer, 'ht.osu', [(0, B, 4), (8000, 2 * B, 4), (16000, B, 4)], heads)
    assert [line.kind for line in s.lines] == ['first', 'tempo', 'tempo']
    assert s.detail['S5']['level_changes'] == 2


def test_mixed_subdivision_share(osu_writer):
    beat32 = [0, B / 6, B / 4, B / 3, B / 2]
    heads = [k * B for k in range(99)] + [99 * B + x for x in beat32]
    s = strata(osu_writer, 'mx.osu', [(0, B, 4)], heads)
    assert s.detail['S1']['mapped_beats'] == 100 and s.detail['S1']['mixed_beats'] == 1
    assert s.members['S1'] == [(96 * B, 100 * B)]
    assert 'S0' not in s.members and s.detail['S0']['fails'] == ['mixed_beat']
    less = strata(osu_writer, 'mx2.osu', [(0, B, 4)], [k * B for k in range(199)] + [199 * B + x for x in beat32])
    assert 'S1' not in less.members


def test_swing_and_triplets(osu_writer):
    swing = [k * B + x for k in range(32) for x in (0, 2 * B / 3)]
    s = strata(osu_writer, 'sw.osu', [(0, B, 4)], swing)
    assert s.detail['S7']['swing_bars'] == 7 and 'S7' in s.members  # the last bar runs past the last head
    trip = [k * B + x for k in range(32) for x in (0, B / 3, 2 * B / 3)]
    assert 'S7' not in strata(osu_writer, 'tr.osu', [(0, B, 4)], trip).members
    # Swing notated at 240 BPM: the swung eighth sits at 1/3 of the second notated beat.
    fast = strata(osu_writer, 'sw240.osu', [(0, B / 2, 8)], swing)
    assert fast.detail['S7']['swing_bars'] == 7


def test_polyrhythm_and_double_notation(osu_writer):
    base = [k * B for k in range(8)]
    dotted = [4000 + k * 0.75 * B for k in range(8)]
    tail = [dotted[-1] + 0.75 * B + k * B for k in range(8)]
    s = strata(osu_writer, 'pr.osu', [(0, B, 4)], base + dotted + tail)
    assert s.detail['S8']['runs'] == 1 and s.members['S8'] == [(4000.0, dotted[-1] + 0.75 * B)]
    # Triplet eighths notated at 240 BPM are 2/3 of a notated beat but 1/3 of a canonical one.
    trip = [k * B / 3 for k in range(48)]
    assert 'S8' not in strata(osu_writer, 'tp.osu', [(0, B / 2, 4)], trip).members


def test_sparse_gap_and_reentry(osu_writer):
    heads = [k * B for k in range(16)] + [13000 + k * B for k in range(32)]
    s = strata(osu_writer, 'gap.osu', [(0, B, 4)], heads)
    assert s.detail['S6']['gaps'] == 1
    (a, b), = s.members['S6']
    # [6500, 10500) is the first window of 8 beats with under 4 heads; the last, [10500, 14500), ends after
    # heads resume at 13 s; re-entry runs to the 4th downbeat after it.
    assert a == pytest.approx(6500.0) and b == pytest.approx(24000.0)


def test_off_lattice_heads_and_implausible_lines(osu_writer):
    heads = [k * B for k in range(32)] + [16 * B + B / 5]
    s = strata(osu_writer, 'ol.osu', [(0, B, 4)], heads)
    assert s.detail['S9']['off_lattice_heads'] == 1 and s.members['S9'] == [(16 * B, 17 * B)]
    stop = strata(osu_writer, 'st.osu', [(0, B, 4), (4000, 1e9, 4), (5000, B, 4)],
                  [k * B for k in range(32)])
    assert [line.kind for line in stop.lines] == ['first', 'implausible', 'same']
    assert stop.detail['S9']['implausible_with_heads'] == 1


def test_recording_rules(osu_writer):
    a = strata(osu_writer, 'a.osu', [(0, B, 4)], [k * B for k in range(64)])
    b = strata(osu_writer, 'b.osu', [(0, B / 2, 4)], [k * B for k in range(64)])
    rec = recording_strata([a, b])
    assert rec['S0'] == {'a.osu': None, 'b.osu': None}
    assert rec['S5'] == {'a.osu': None, 'b.osu': None}  # the two sets are notated an octave apart
    c = strata(osu_writer, 'c.osu', [(0, B, 4), (4000, B, 4)], [k * B for k in range(64)])
    assert 'S0' in recording_strata([a, c])  # a redundant line is no second musical line


def test_rate_edit_names_and_copies():
    assert is_rate_edit('Hard 1.1x') and is_rate_edit('x1.2 Insane') and is_rate_edit('0.9x')
    assert not is_rate_edit('Hard') and not is_rate_edit('1.0x') and not is_rate_edit('4K Lv.12')
    s = [Segment(0.0, 500.0, 4), Segment(1000.0, 400.0, 4)]
    assert same_timing(s, [Segment(0.4, 500.0, 4), Segment(1000.0, 400.0, 4)])
    assert not same_timing(s, [Segment(0.0, 500.0, 4)]) and not same_timing(s, [Segment(5.0, 500.0, 4), s[1]])
    assert [line.kind for line in classify_lines(s, [], 0.0, 2000.0)] == ['first', 'tempo']


def test_flags_interface_for_the_training_side(osu_writer):
    heads = [k * 1750 + j * 250 for k in range(8) for j in range(7)]
    ref = read_set([osu_writer('fl.osu', [(k * 1750, B, 4) for k in range(8)], heads=heads)])
    f = flags(ref)
    assert f['strata'] == ['S2', 'S4', 'S9'] and f['hard'] is True and f['flags']['S4.reset'] is True
    assert f['flags']['S0'] is False and json.loads(json.dumps(f)) == f
    control = flags(read_set([osu_writer('fc.osu', [(0, B, 4)], heads=[k * B for k in range(64)])]))
    assert control['strata'] == ['S0'] and control['hard'] is False
