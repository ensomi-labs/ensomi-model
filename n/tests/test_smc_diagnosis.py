import numpy as np
import pytest

from ensomi_n.smc import minimal_peaks
from ensomi_n.smc_diagnosis import (
    bootstrap_ols, evidence_probes, level_remappings, ridge_readout, select_folds, timing_metrics,
)


def test_remappings_preserve_phase_fractional_times_and_end():
    p = np.array([4.83, 5.37, 5.94, 6.51, 7.08, 7.69, 8.30, 8.95])
    seq = level_remappings(p)
    assert len(seq) == 11
    np.testing.assert_array_equal(seq[2], [5.37, 6.51, 7.69, 8.95])
    np.testing.assert_array_equal(seq[5], [5.94, 7.69])
    np.testing.assert_array_equal(seq[9], [6.51, 8.95])
    assert len(seq[-1]) == 15
    assert seq[-1][0] == p[0] and seq[-1][-1] == p[-1]
    np.testing.assert_allclose(seq[-1][1::2], (p[:-1] + p[1:]) / 2)


def test_absolute_level_ceiling_and_signed_error():
    r = np.arange(5.25, 39, 1.)
    doubled = np.sort(np.r_[r, r + .5])
    raw = timing_metrics(r, doubled)
    assert raw['F70'] == pytest.approx(2 / 3)
    assert raw['CMLt'] == 0
    restored = timing_metrics(r, level_remappings(doubled)[1])
    assert restored['F70'] == restored['CMLt'] == restored['AMLt'] == 1
    early = timing_metrics(r, r - .025)
    assert early['F20'] == 0 and early['F35'] == 1
    assert early['signed_ms'] == pytest.approx(-25)


def test_heldout_labels_do_not_affect_own_readout():
    rng = np.random.default_rng(5)
    x, y = rng.normal(size=(24, 4)), rng.uniform(size=(24, 11))
    folds = np.repeat(np.arange(8), 3)
    a, fits_a = ridge_readout(x, y, folds)
    global_a = select_folds(y, folds)
    y[folds == 2] = rng.normal(size=(3, 11)) * 100
    b, fits_b = ridge_readout(x, y, folds)
    np.testing.assert_array_equal(a[folds == 2], b[folds == 2])
    np.testing.assert_array_equal(global_a[folds == 2], select_folds(y, folds)[folds == 2])
    assert fits_a[2] == fits_b[2]
    assert not set(fits_a[2]['train_indices']) & set(np.flatnonzero(folds == 2))


def test_threshold_is_strict_and_default_unchanged():
    x = np.zeros(200)
    x[[20, 50, 100]] = [.3, .5, .9]
    np.testing.assert_array_equal(minimal_peaks(x), [1.])
    np.testing.assert_array_equal(minimal_peaks(x, threshold=.3), [.5, 1.])
    with pytest.raises(ValueError):
        minimal_peaks(x, threshold=float('nan'))


def test_evidence_probes_reject_offbeat_pulse():
    r = np.arange(5.25, 39, .5)
    x = np.zeros(4000)
    x[np.round(r * 100).astype(int)] = 1
    aligned, wrong = evidence_probes(x, r), evidence_probes(x, r + .25)
    assert aligned['peak50'] == 1 and wrong['peak50'] == 0
    assert aligned['advantage'] > 0 and wrong['advantage'] < 0


def test_evidence_clock_and_normalization_match_after_resampling():
    t = np.arange(2001) / 50
    x = .5 + .4 * np.cos(2 * np.pi * t * 2)
    up = np.interp(np.arange(4000) / 100, t, x)
    ref = np.arange(5.23, 39, .5)
    assert evidence_probes(x, ref, fps=50) == pytest.approx(evidence_probes(up, ref))


def test_joint_model_distinguishes_a_confounder_from_a_tag():
    rng = np.random.default_rng(3)
    tempo = rng.normal(size=120)
    tag = (tempo + rng.normal(size=120) * .5 > 0).astype(float)
    x = np.column_stack([np.ones(120), tempo, tag])
    y = (2 + .4 * tempo)[:, None]
    assert y[tag == 1].mean() > y[tag == 0].mean()
    result = bootstrap_ols(x, y, samples=100)
    np.testing.assert_allclose(np.array(result['coef'])[:, 0], [2, .4, 0], atol=1e-12)
    assert result['singular'] == 0


@pytest.mark.parametrize('parity', [0, 1])
def test_free_half_level_fractional_boundary_and_last_beat(parity):
    from ensomi_n.v0.decode import DecodeConfig, Evidence, _Regime, _notated
    t = np.array([203.7, 732.1, 1288.4, 1815.8, 2403.2, 3029.6, 3642.1])
    level = np.tile([.97, .01, .01, .01], (450, 1))
    down = np.zeros(450)
    down[np.round(t[2 + parity::2] / 10).astype(int)] = 1
    ev = Evidence(np.ones(450), down=down, level=level)
    regs = [_Regime('constant', 0, 2, t[0], t[1] - t[0])]
    regs += [_Regime('free', i, i + 1) for i in range(2, len(t))]
    segs, _, ratios = _notated(ev, regs, t, np.ones(len(t)), DecodeConfig(allow_half_free=True))
    free = [s for s in segs if s.free]
    np.testing.assert_array_equal([b for s in free for b in s.beats], t[2 + parity::2])
    assert all(s.step > 0 and np.isfinite(s.step) for s in segs)
    assert ratios == [.5, .5]
    _, _, defaults = _notated(ev, regs, t, np.ones(len(t)), DecodeConfig())
    assert defaults[1] != .5


def test_widened_band_can_represent_a_slow_pulse():
    from ensomi_n.v0.decode import DecodeConfig, decode, impulse_evidence
    beats = np.arange(203.7, 13000, 1000.)
    ev = impulse_evidence(1400, beats, floor=.001)
    narrow, _ = decode(ev, DecodeConfig(reset_penalty=14.))
    wide, _ = decode(ev, DecodeConfig(bpm_lo=35., reset_penalty=14.))
    assert np.median(np.diff(wide.beats)) == pytest.approx(1000, abs=1)
    assert np.median(np.diff(narrow.beats)) == pytest.approx(500, abs=1)
    assert np.max(np.min(np.abs(wide.beats[:, None] - beats), axis=0)) < 1.5
