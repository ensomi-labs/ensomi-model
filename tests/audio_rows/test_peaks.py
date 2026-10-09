"""The peak refinement follows the sub-frame peak positions, not the 20 ms frame grid."""
import numpy as np

from ensomi_model.audio_rows.fitter import fit_segments
from ensomi_model.audio_rows.fitter.peaks import beat_peaks, refine_segments

FPS = 50.0


def _logits(beats, seconds=60.0, width_ms=12.0, peak=0.999, rng=None):
    """BeatThis-like logits: the beat probability is a Gaussian bump of height ``peak`` at each beat time (ms)
    on a floor of 0.002; returned as logits, with optional noise."""
    t = np.arange(int(seconds * FPS)) * 1000 / FPS
    bumps = np.exp(-0.5 * ((t[:, None] - beats[None]) / width_ms) ** 2).max(1)
    p = np.clip(0.002 + (peak - 0.002) * bumps, 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p))
    if rng is not None:
        x = x + rng.normal(0, 0.3, x.shape)
    return x.astype(np.float32)


def _lines(segments, lo, hi):
    out = []
    for k, s in enumerate(segments):
        end = segments[k + 1].offset_ms if k + 1 < len(segments) else hi
        j0 = int(np.ceil((max(lo, s.offset_ms if k else lo) - s.offset_ms) / s.beat_length_ms))
        j1 = int(np.floor((min(hi, end) - s.offset_ms) / s.beat_length_ms))
        out.append(s.offset_ms + np.arange(j0, j1 + 1) * s.beat_length_ms)
    return np.concatenate(out)


def test_peaks_sit_at_the_sub_frame_position():
    beats = np.arange(1007.0, 59_000.0, 352.4)                   # 170.26 BPM, off the frame grid
    times, probs = beat_peaks(_logits(beats), FPS)
    matched = times[(times > 900) & (times < 58_900)]
    assert len(matched) == len(beats)
    assert np.abs(matched - beats).max() < 1.0 and probs.min() > 0.5


def test_refined_grid_is_shift_equivariant_within_one_ms_and_tempo_exact():
    length = 320.0                                                 # 187.5 BPM: exactly 16 frames, the M4 case
    base = np.arange(1000.0, 59_000.0, length)
    reference = None
    for k in (0, 3, 5, 7, 10, 13, 15, 17, 19, 37):
        x = _logits(base + k)
        legacy = fit_segments(1 / (1 + np.exp(-x.astype(np.float64))), np.zeros_like(x), FPS)
        refined = refine_segments(legacy, x, FPS)
        lines = _lines(refined, 2_000, 58_000)
        if reference is None:
            reference = lines
        shifted = reference + k
        nearest = shifted[np.abs(shifted[:, None] - lines[None]).argmin(0)]
        assert np.abs(lines - nearest).max() <= 1.0, k
        assert abs(refined[-1].beat_length_ms - length) < 0.01 * length / 100, k   # tempo within 0.01 %


def test_refinement_keeps_the_downbeat_phase_and_segment_count():
    rng = np.random.default_rng(1)
    beats = np.r_[np.arange(250.0, 40_000.0, 500.0), np.arange(40_250.0, 79_750.0, 400.0)]
    x = _logits(beats + rng.normal(0, 4, beats.shape), seconds=80.0, rng=rng)
    down = _logits(beats[::4], seconds=80.0)
    legacy = fit_segments(1 / (1 + np.exp(-x.astype(np.float64))), 1 / (1 + np.exp(-down.astype(np.float64))), FPS)
    refined = refine_segments(legacy, x, FPS)
    assert len(refined) == len(legacy) == 2
    for a, b in zip(legacy, refined):
        assert abs(a.offset_ms - b.offset_ms) < 0.25 * a.beat_length_ms          # same beat, hence same downbeat phase
        assert abs(b.local_bpm - a.local_bpm) < 0.5
    assert np.allclose([s.local_bpm for s in refined], [120.0, 150.0], atol=0.05)


def test_one_tempo_in_several_legacy_segments_becomes_one_line():
    from ensomi_model.audio_rows.fitter.types import TimingSegment
    beats = np.arange(500.0, 119_500.0, 352.4)
    x = _logits(beats, seconds=120.0)
    legacy = (TimingSegment(500.0, 352.6), TimingSegment(60_408.0, 352.2))     # same tempo within 0.3 %, split in two
    refined = refine_segments(legacy, x, FPS)
    assert len(refined) == 1
    assert abs(refined[0].beat_length_ms - 352.4) < 0.005 and abs(refined[0].offset_ms - 500.0) <= 1.0


def test_segments_without_peaks_on_their_grid_keep_the_legacy_values():
    from ensomi_model.audio_rows.fitter.types import TimingSegment
    x = _logits(np.arange(500.0, 59_500.0, 400.0))
    legacy = (TimingSegment(700.0, 400.0),)                                     # half a beat off every peak
    assert refine_segments(legacy, x, FPS) == legacy


def test_bar_phase_follows_the_downbeats_and_keeps_the_line():
    from ensomi_model.audio_rows.fitter.peaks import downbeat_phase
    from ensomi_model.audio_rows.fitter.types import TimingSegment
    beats = np.arange(1000.0, 59_000.0, 500.0)
    x, down = _logits(beats), _logits(beats[2::4])                             # bars start at 2000 + 2000 k ms
    for committed, offset in (((TimingSegment(1500.0, 500.0),), 2000.0),        # first offset on the wrong beat
                              ((TimingSegment(1000.0, 500.0),), 2000.0),        # two beats off: the later bar start
                              ((TimingSegment(30_500.0, 500.0),), 30_000.0),    # mid-song (a merged refit)
                              ((TimingSegment(1250.0, 250.0),), 2000.0),        # raw line at twice the canonical tempo
                              ((TimingSegment(1500.0, 500.0), TimingSegment(40_000.0, 400.0)), 2000.0)):
        out = downbeat_phase(committed, x, down, FPS)
        assert out[0] == TimingSegment(offset, committed[0].beat_length_ms)    # the bar's beat nearest the old offset
        assert out[1:] == committed[1:]
    assert downbeat_phase((TimingSegment(1500.0, 500.0),), x, None, FPS) == (TimingSegment(1500.0, 500.0),)
