"""The Mel tempo refit follows onsets that drift against the grid and keeps grids that do not drift."""
import numpy as np

from ensomi_model.audio_rows.fitter.mel_phase import MAPPER_OFFSET_MS, mel_tempo
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


def _onsets(first_beat, length, end):
    """Onsets of the mappers' beats (chart clock) first_beat + n * length up to end."""
    return np.arange(first_beat, end, length) - MAPPER_OFFSET_MS


def test_a_drifting_tempo_follows_the_onsets():
    grid_length, true_length = 60000.0 / 120.4, 60000.0 / 120.45      # 0.25 BPM away from any integer
    peaks = np.arange(1000.0, 59_000.0, grid_length)
    onsets = _onsets(1000.0 - LAG_MS, true_length, 59_000.0)
    out = mel_tempo((TimingSegment(1000.0, grid_length),), _log_mel(onsets), _logits(peaks), FPS, SECONDS * 1000)
    assert len(out) == 1 and abs(out[0].beat_length_ms - true_length) < 0.02
    middle = 1000.0 + 58 * true_length            # a beat in the middle of the song lands on the onsets' line
    n = round((middle - out[0].offset_ms) / out[0].beat_length_ms)
    assert abs(out[0].offset_ms + n * out[0].beat_length_ms - middle) < 1.0


def test_one_line_near_an_integer_bpm_keeps_the_integer():
    true_length = 60000.0 / 119.9                                    # corrected_grid rounds the fitted 120 BPM
    peaks = np.arange(1000.0, 59_000.0, 500.0)
    onsets = _onsets(1000.0 - LAG_MS, true_length, 59_000.0)
    out = mel_tempo((TimingSegment(1000.0, 500.0),), _log_mel(onsets), _logits(peaks), FPS, SECONDS * 1000)
    assert len(out) == 1 and abs(out[0].beat_length_ms - 500.0) < 1e-9


def test_onsets_on_the_grid_leave_it_unchanged():
    peaks = np.arange(1000.0, 59_000.0, 500.0)
    segments = (TimingSegment(1000.0, 500.0),)
    out = mel_tempo(segments, _log_mel(_onsets(1000.0 - LAG_MS, 500.0, 59_000.0)), _logits(peaks), FPS, SECONDS * 1000)
    assert out == segments


def test_only_the_drifting_segment_changes():
    a = np.arange(1000.0, 30_000.0, 500.0); b = np.arange(30_000.0, 59_800.0, 400.0)
    true_b = 400.0 * (1 + 4e-4)
    onsets = np.r_[_onsets(1000.0 - LAG_MS, 500.0, 30_000.0 - LAG_MS), _onsets(30_000.0 - LAG_MS, true_b, 59_800.0 - LAG_MS)]
    segments = (TimingSegment(1000.0, 500.0), TimingSegment(30_000.0, 400.0))
    out = mel_tempo(segments, _log_mel(onsets), _logits(np.r_[a, b]), FPS, SECONDS * 1000)
    assert out[0] == segments[0]
    assert abs(out[1].beat_length_ms - true_b) < 0.02
