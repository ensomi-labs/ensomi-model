"""Absolute scorer fixtures, including fractional segment and clock boundaries."""
import numpy as np
import pytest

from ensomi_n.eval import score
from ensomi_n.timing import Reference, Segment
from .measures import (auc, boundary_auc, flux_peaks, interval, matching, onset_offset,
                       robust_sd, spacing_class, spectral_flux)


def test_robust_spread_and_constant_bootstrap():
    assert robust_sd([-2, -1, 0, 1, 2]) == pytest.approx(1.4826)
    assert interval([3., 3.]) == dict(value=3., lo=3., hi=3., n=2)


def test_spacing_threshold_and_duplicate_last_time():
    assert spacing_class([]) == 0
    assert spacing_class([13., 13.]) == 1
    assert spacing_class([13., 5013., 11013.]) == 2
    assert spacing_class([13., 5012.99, 11013.]) == 3


def test_injective_matching():
    assert matching([0., 100., 200.], [0., 100.], 70)['f'] == .8
    assert matching([0., 1.], [0.], 5)['recall'] == .5
    assert matching([0., 100.], [5., 105.], 5)['f'] == 1.
    assert matching([0., 100.], [5.01, 105.01], 5)['f'] == 0.


def test_auc_ties_and_equal_filtering():
    assert auc([1, 2], [0, 1]) == .875
    assert auc([0, 0], [0, 0]) == .5
    assert auc([1], [2, 3]) == 0.
    z = np.zeros(501)
    z[102] = 2.
    assert boundary_auc(z, [1000.], [1000.], 0., 5000.)['auc'] == 1.
    assert boundary_auc(z, [1000.], [1000.], 0., 5000., radius=0)['auc'] == .5


def test_harness_sign_on_fractional_reset_and_noninteger_tempo():
    ref = Reference.from_segments([Segment(100.25, 487.3), Segment(12077.17, 487.3),
                                   Segment(23333.33, 423.7)], np.arange(1000., 39000., 121.7))
    sc = score(ref, ref.timing.shifted(12.5))
    assert sc.offset_ms == pytest.approx(12.5)
    assert sc.bl20 == 1.
    assert sc.bl10 == 0.
    assert score(ref, ref.timing).bars_start == 1.


def test_flux_zero_and_known_pcm_translation():
    assert not spectral_flux(np.zeros(24000)).any()
    wave = np.zeros(24000)
    wave[12000] = 1.
    moved = np.zeros_like(wave)
    moved[12408] = 1.  # exactly 17 ms
    a = flux_peaks(spectral_flux(wave).mean(1))
    b = flux_peaks(spectral_flux(moved).mean(1))
    assert len(a) == len(b) == 1
    assert b[0] - a[0] == 17.


def test_onset_reference_direction_lag_and_no_reuse():
    lines = np.array([100., 600., 1100.])
    r = onset_offset(lines, np.ones(3), lines + 17., 20., lag=-3.)
    assert r['flux_offset_ms'] == 20.
    assert r['verdict'] == 'model'
    assert r['matched_peaks'] == 3
    assert onset_offset(lines, np.ones(3), lines - 3., 20., -3.)['verdict'] == 'chart'
    assert onset_offset([100., 105.], [1., 1.], [103.], 0., 0.)['matched_peaks'] == 1


def test_beat_matched_background_can_expose_generic_beat_response():
    z = np.zeros(501)
    z[::50] = 1.
    assert boundary_auc(z, [1000.], [1000.], 0., 5000., negative_times=np.arange(0., 5001., 500.))['auc'] == .5
    assert boundary_auc(z, [1000.], [1000.], 0., 5000.)['auc'] > .5
