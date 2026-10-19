"""The v0 decoder on fixtures built to break it (guardrail 10), scored by the evaluation harness.

Expected values are absolute: oracle evidence rendered from a chart must decode to BL-20 1.0 on that
chart, with the level, bars and resets the chart states. Fixtures cover several segments, a fractional
boundary, a same-tempo reset, notated levels 2 and 0.5, odd and alternating meters, rubato, noisy
evidence, a late start and an end mid-bar, the 80/160 fold, and a reset that only the phase field shows.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from ensomi_n.eval import score
from ensomi_n.timing import Reference, Segment, Timing
from ensomi_n.v0.decode import DecodeConfig, Evidence, decode, impulse_evidence


def _chart(segments, lo, hi):
    """A reference whose heads are the notated beats in [lo, hi]."""
    beats = Timing.from_segments(segments, lo, hi).beats
    heads = beats[(beats >= lo) & (beats <= hi)]
    return Reference.from_segments(segments, heads)


def _oracle(ref, n_frames=None, level_ratio=None, boundaries=None, floor=0.01):
    n = n_frames or int(ref.hi_ms / 10.0) + 300
    return impulse_evidence(n, ref.timing.canonical().t, ref.timing.bar_starts(), level_ratio, boundaries, floor)


def _phase_field(lines: np.ndarray, n: int, beat_ms: float) -> np.ndarray:
    """The F2 target of `ensomi_n.targets.dense`: ((t - line_k) / beat) mod 1 from the last line."""
    t = 10.0 * np.arange(n)
    k = np.clip(np.searchsorted(lines, t, side='right') - 1, 0, len(lines) - 1)
    return np.mod((t - lines[k]) / beat_ms, 1.0)


def test_one_tempo_oracle_is_exact():
    beat = 60000.0 / 128
    ref = _chart([Segment(1234.5, beat, 4)], 1234.5, 1234.5 + 380 * beat)
    pm, diag = decode(_oracle(ref))
    s = score(ref, pm)
    assert s.bl20 == pytest.approx(1.0, abs=1e-12)
    assert abs(s.offset_ms) < 1.0
    assert diag['n_segments'] == 1 and diag['n_free'] == 0
    assert len(pm.segments) == 1 and abs(pm.segments[0].beat_ms - beat) < 0.01
    assert s.bars_start == pytest.approx(1.0) and s.level == pytest.approx(1.0)


@pytest.mark.parametrize('boundaries', [False, True])
def test_tempo_change_at_a_fractional_boundary(boundaries):
    a = Segment(1000.0, 500.0, 4)
    b = Segment(1000.0 + 40 * 500.0 + 0.37 * 500.0, 400.0, 4)      # 0.37 beat after the old grid's last beat
    ref = _chart([a, b], 1000.0, b.start_ms + 150 * 400.0)
    pm, diag = decode(_oracle(ref, boundaries=[b.start_ms] if boundaries else None))
    s = score(ref, pm)
    assert s.bl20 == pytest.approx(1.0, abs=1e-12)
    assert s.boundary_recall == pytest.approx(1.0)
    assert 2 <= diag['n_segments'] + diag['n_free'] <= 3
    lengths = sorted(round(x.beat_ms, 2) for x in pm.segments)
    assert lengths[0] == pytest.approx(400.0, abs=0.01) and lengths[-1] == pytest.approx(500.0, abs=0.01)


def test_same_tempo_reset():
    a = Segment(1000.0, 500.0, 4)
    b = Segment(1000.0 + 40 * 500.0 + 0.37 * 500.0, 500.0, 4)
    ref = _chart([a, b], 1000.0, b.start_ms + 80 * 500.0)
    pm, diag = decode(_oracle(ref))
    s = score(ref, pm)
    assert s.bl20 == pytest.approx(1.0, abs=1e-12)
    assert s.reset_recall == pytest.approx(1.0)
    assert 2 <= diag['n_segments'] + diag['n_free'] <= 3
    assert s.bars_start == pytest.approx(1.0)


def test_notated_double_level_and_bars():
    seg = Segment(1000.0, 60000.0 / 180, 4)
    ref = _chart([seg], 1000.0, 1000.0 + 360 * seg.beat_ms)
    pm, diag = decode(_oracle(ref, level_ratio=2.0))
    s = score(ref, pm)
    assert diag['levels'] == [2.0]
    assert s.bl20 == pytest.approx(1.0, abs=1e-12)
    assert s.level == pytest.approx(1.0)
    assert s.bars_start == pytest.approx(1.0) and s.bars_same == pytest.approx(1.0)


def test_notated_half_level_keeps_every_line():
    seg = Segment(1250.0, 1000.0, 4)                  # 60 BPM notated, canonical 120
    ref = _chart([seg], 1250.0, 1250.0 + 150 * 1000.0)
    pm, diag = decode(_oracle(ref, level_ratio=0.5))
    s = score(ref, pm)
    assert s.bl20 == pytest.approx(1.0, abs=1e-12)
    assert s.level == pytest.approx(1.0)
    assert s.bars_start == pytest.approx(1.0)


def test_meter_seven():
    ref = _chart([Segment(1000.0, 600.0, 7)], 1000.0, 1000.0 + 7 * 30 * 600.0)
    pm, diag = decode(_oracle(ref))
    s = score(ref, pm)
    assert diag['meters'] == [7]
    assert s.bl20 == pytest.approx(1.0, abs=1e-12)
    assert s.bars_start == pytest.approx(1.0) and s.bars_same == pytest.approx(1.0)


def test_alternating_four_and_three():
    segs, t = [], 1000.0
    for i in range(40):
        m = 4 if i % 2 == 0 else 3
        segs.append(Segment(t, 500.0, m))
        t += m * 500.0
    ref = _chart(segs, 1000.0, t - 500.0)
    pm, diag = decode(_oracle(ref))
    s = score(ref, pm)
    assert s.bl20 == pytest.approx(1.0, abs=1e-12)
    assert s.bars_start == pytest.approx(1.0) and s.bars_same == pytest.approx(1.0)


def _rubato(seed=0, n=320, mean_ms=545.0):
    rng = np.random.default_rng(seed)
    x, out = 0.0, []
    for _ in range(n):
        x = 0.95 * x + rng.normal(0.0, 0.02)
        out.append(mean_ms * np.exp(x))
    return 1500.0 + np.concatenate([[0.0], np.cumsum(out)])


def test_rubato():
    beats = _rubato()
    ref = Reference.from_beats(beats, beats)
    ev = impulse_evidence(int(beats[-1] / 10) + 300, beats)
    pm, diag = decode(ev)
    s = score(ref, pm)
    assert s.bl20 >= 0.99


def test_free_beats_at_double_level_keep_only_canonical_lines():
    """A free beat's segment holds r notated beats; the scorer's anchoring keeps only the canonical ones."""
    beats = _rubato(seed=3)
    ref = Reference.from_beats(beats, beats)
    ev = impulse_evidence(int(beats[-1] / 10) + 300, beats, level_ratio_at=2.0)
    pm, diag = decode(ev, DecodeConfig(fit_tol_ms=3.0))          # tight fits: most beats stay free
    assert diag['n_free'] > 50
    s = score(ref, pm)
    assert s.bl20 >= 0.99
    lines = pm.timing(ref.lo_ms, ref.hi_ms).canonical().t
    inside = lines[(lines >= ref.lo_ms - 1.0) & (lines <= ref.hi_ms + 1.0)]
    assert abs(len(inside) - len(beats)) <= 1
    assert len(pm.beats) >= 2 * len(beats) - 4                  # the notated beats are at the double level


def test_noisy_evidence():
    rng = np.random.default_rng(7)
    seg = Segment(1000.0, 500.0, 4)
    ref = _chart([seg], 1000.0, 1000.0 + 300 * 500.0)
    lines = ref.timing.canonical().t
    shown = lines[rng.random(len(lines)) >= 0.10]
    shown = shown + rng.normal(0.0, 3.0, size=len(shown))
    n = int(ref.hi_ms / 10) + 300
    ev = impulse_evidence(n, shown, floor=0.05)
    spurious = rng.choice(n, size=n // 100, replace=False)
    beat = ev.beat.copy()
    beat[spurious] = np.maximum(beat[spurious], 0.5)
    pm, diag = decode(Evidence(beat, ev.beat_off))
    s = score(ref, pm)
    assert s.bl20 >= 0.97


def test_late_start_and_end_mid_bar():
    beat = 60000.0 / 140
    seg = Segment(800.0, beat, 4)
    lo, hi = 800.0 + 12 * beat, 800.0 + 221 * beat         # the last head is mid-bar (221 = 55 bars + 1)
    ref = _chart([seg], lo, hi)
    lines = ref.timing.canonical().t
    n = int(hi / 10) + 15                                   # the audio ends just after the last head
    ev = impulse_evidence(n, lines[lines >= 5000.0], ref.timing.bar_starts())
    pm, diag = decode(ev)
    s = score(ref, pm)
    assert s.bl20 == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize('bpm', [158.0, 80.0, 161.0])
def test_near_the_fold(bpm):
    beat = 60000.0 / bpm
    ref = _chart([Segment(700.0, beat, 4)], 700.0, 700.0 + 200 * beat)
    pm, diag = decode(_oracle(ref))
    s = score(ref, pm)
    assert s.bl20 == pytest.approx(1.0, abs=1e-12)


def test_phase_field_finds_a_reset_without_beat_evidence():
    a = Segment(1000.0, 500.0, 4)
    b = Segment(1000.0 + 56 * 500.0 + 0.37 * 500.0, 500.0, 4)    # 29185 ms, inside the gap
    ref = _chart([a, b], 1000.0, b.start_ms + 60 * 500.0)
    n = int(ref.hi_ms / 10) + 300
    lines = ref.timing.canonical().t
    ev = impulse_evidence(n, lines, ref.timing.bar_starts())
    gap = (27000.0, 33000.0)
    f0, f1 = int(gap[0] / 10), int(gap[1] / 10)
    beat, off = ev.beat.copy(), ev.beat_off.copy()
    beat[f0:f1] = 0.01
    off[f0:f1] = 0.0
    phase = _phase_field(lines, n, 500.0)
    # The F2 head gives a period with the phase; both are what it would output with kappa 4.
    with_phase = Evidence(beat, off, ev.down, phase=phase, kappa=np.full(n, 4.0), logp=np.full(n, np.log(0.5)),
                          logp_sigma=np.full(n, 0.05))
    without = Evidence(beat, off, ev.down)
    s_with = score(ref, decode(with_phase)[0], spans=[gap])
    s_without = score(ref, decode(without)[0], spans=[gap])
    assert s_with.bl20 == pytest.approx(1.0, abs=1e-12)
    assert s_without.bl20 < 0.9                                  # the fixture can break: phase is what finds it
    assert score(ref, decode(with_phase)[0]).reset_recall == pytest.approx(1.0)


def test_no_evidence_gives_an_empty_map():
    pm, diag = decode(Evidence(np.full(3000, 0.01)))
    assert len(pm.beats) == 0 and diag['n_beats'] == 0


def test_four_minutes_decode_quickly():
    beat = 60000.0 / 128
    ref = _chart([Segment(1000.0, beat, 4)], 1000.0, 1000.0 + 500 * beat)
    n = 24000
    ev = _oracle(ref, n_frames=n, level_ratio=1.0, boundaries=[])
    phase = _phase_field(ref.timing.canonical().t, n, beat)
    dense = Evidence(ev.beat, ev.beat_off, ev.down, level=ev.level, boundary=ev.boundary, phase=phase,
                     kappa=np.full(n, 3.0), logp=np.full(n, np.log(beat / 1000.0)), logp_sigma=np.full(n, 0.1))
    t0 = time.process_time()                # CPU time: other processes on the machine do not count
    pm, diag = decode(dense)
    assert time.process_time() - t0 < 3.0
    assert score(ref, pm).bl20 == pytest.approx(1.0, abs=1e-12)
