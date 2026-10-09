"""The Mel phase moves each segment's offset to the onsets of its beats and keeps every tempo."""
import numpy as np

from ensomi_model.audio_rows.fitter.mel_phase import MAPPER_OFFSET_MS, mel_phase, onset_envelope, onset_times
from ensomi_model.audio_rows.fitter.types import TimingSegment
from ensomi_model.audio_rows.grid import LAG_MS

FPS = 50.0
SECONDS = 60.0


def _logits(beats, width_ms=12.0):
    """BeatThis-like beat logits with a probability bump at each beat time (ms)."""
    t = np.arange(int(SECONDS * FPS)) * 1000 / FPS
    p = np.clip(0.002 + 0.997 * np.exp(-0.5 * ((t[:, None] - beats[None]) / width_ms) ** 2).max(1), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).astype(np.float32)


def _log_mel(onsets, bins=8, width_ms=8.0):
    """A [F, bins] log-Mel whose onset envelope is a Gaussian bump at each onset time (ms)."""
    k = np.arange(int(SECONDS * 100))
    env = np.exp(-0.5 * (((k * 10.0 + 15.0)[:, None] - np.asarray(onsets)[None]) / width_ms) ** 2).max(1)
    env[0] = 0.0
    return np.repeat(np.cumsum(env)[:, None] / bins, bins, axis=1)


def test_onsets_sit_at_the_envelope_peaks():
    onsets = 1000.0 + np.arange(100) * 487.3 + np.linspace(0.0, 10.0, 100)
    env = onset_envelope(_log_mel(onsets))
    found = onset_times(env, onsets + 20.0, 60.0)
    assert np.max(np.abs(found - onsets)) < 1.0
    assert np.all(np.isnan(onset_times(np.zeros(500), np.array([1000.0, 2000.0]))))


def test_grid_moves_to_the_onsets_and_keeps_its_tempo():
    peaks = np.arange(1000.0, 59_000.0, 500.0)              # BeatThis's clock; the chart's beats sit LAG_MS earlier
    for error in (-9.0, 0.0, 7.0):
        onsets = peaks - LAG_MS + error - MAPPER_OFFSET_MS   # the mappers' beat is error ms off the grid
        out = mel_phase((TimingSegment(1000.0, 500.0),), _log_mel(onsets), _logits(peaks), FPS, SECONDS * 1000)
        assert abs(out[0].offset_ms - (1000.0 + error)) < 1.0
        assert out[0].beat_length_ms == 500.0


def test_each_segment_takes_its_own_phase():
    a = np.arange(1000.0, 30_000.0, 500.0); b = np.arange(30_000.0, 59_800.0, 400.0)
    onsets = np.r_[a - LAG_MS + 5.0, b - LAG_MS - 6.0] - MAPPER_OFFSET_MS
    segments = (TimingSegment(1000.0, 500.0), TimingSegment(30_000.0, 400.0))
    out = mel_phase(segments, _log_mel(onsets), _logits(np.r_[a, b]), FPS, SECONDS * 1000)
    assert abs(out[0].offset_ms - 1005.0) < 1.0 and abs(out[1].offset_ms - 29_994.0) < 1.0
    assert [s.beat_length_ms for s in out] == [500.0, 400.0]


def test_one_tempo_in_several_segments_moves_as_one_line():
    peaks = np.arange(1000.0, 59_000.0, 500.0)
    segments = (TimingSegment(1000.0, 500.0), TimingSegment(20_000.0, 250.0))   # one tempo: corrected_grid keeps one line
    out = mel_phase(segments, _log_mel(peaks - LAG_MS + 4.0 - MAPPER_OFFSET_MS), _logits(peaks), FPS, SECONDS * 1000)
    shifts = [o.offset_ms - s.offset_ms for o, s in zip(out, segments)]
    assert abs(shifts[0] - 4.0) < 1.0 and abs(shifts[0] - shifts[1]) < 1e-6


def test_without_onsets_or_peaks_the_segments_stay():
    peaks = np.arange(1000.0, 59_000.0, 500.0)
    segments = (TimingSegment(1000.0, 500.0),)
    flat = np.zeros((int(SECONDS * 100), 8))
    assert mel_phase(segments, flat, _logits(peaks), FPS, SECONDS * 1000) == segments
    assert mel_phase(segments, _log_mel(peaks), np.full(int(SECONDS * FPS), -8.0, np.float32), FPS, SECONDS * 1000) == segments
