import numpy as np

from ensomi_model.audio_rows.fitter import fit_segments
from ensomi_model.audio_rows.grid import corrected_grid
from ensomi_model.r2.common import GridArrays, grid_from_arrays


def test_one_tempo_split_across_the_fold_becomes_one_native_canonical_segment():
    g, seg = corrected_grid([(30.0, 60000 / 159.9), (30030.0, 60000 / 160.1)], 60_000.0)
    assert len(seg) == 1
    assert np.all((g.bpm([0.0, 45_000.0]) >= 80) & (g.bpm([0.0, 45_000.0]) < 160))
    assert np.array_equal(g.lengths, GridArrays.from_grid(grid_from_arrays(seg, seg[:, [0, 2]])).lengths)


def test_real_tempo_changes_stay_and_every_offset_moves_30_ms_earlier():
    g, seg = corrected_grid([(30.0, 500.0), (30030.0, 600.0)], 60_000.0)
    assert seg[:, 0].tolist() == [0.0, 30_000.0]
    assert np.allclose(g.bpm([0.0, 40_000.0]), [120.0, 100.0])


def test_bpm_rounding_keeps_the_middle_beat_in_place():
    length = 60000 / 184.1
    _, seg = corrected_grid([(1_030.0, length)], 60_000.0)
    middle = 1_000.0 + round(59_000.0 / (2 * length)) * length       # lagged offset, beat nearest mid-segment
    notated = 60000 / 184
    assert np.isclose(seg[0, 1], 2 * notated)                          # 184 BPM folds to 92
    beats = (middle - seg[0, 0]) / notated
    assert abs(beats - round(beats)) < 1e-9


def test_fitter_recovers_the_tempo_segments_of_a_pulse_train():
    fps = 50.0
    t = np.arange(int(80 * fps)) * 1000 / fps
    beats = np.r_[np.arange(250.0, 40_000.0, 500.0), np.arange(40_250.0, 79_750.0, 400.0)]
    pulses = lambda times: np.exp(-0.5 * ((t[:, None] - times[None]) / 15.0) ** 2).max(1).astype(np.float32)
    segments = fit_segments(pulses(beats), pulses(beats[::4]), fps)
    assert len(segments) == 2
    assert np.allclose([s.local_bpm for s in segments], [120.0, 150.0], atol=0.5)
    assert segments[0].offset_ms == 250.0 and abs(segments[1].offset_ms - 40_250.0) < 400.0
