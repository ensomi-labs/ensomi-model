import numpy as np
import pytest

from ensomi_n.v0.infer import windows
from .averaging import align_evidence, crop_starts, mean_evidence, shifted_wave


def test_pcm_translation_has_no_wrap():
    wave = np.arange(8)
    assert shifted_wave(wave, 3).tolist() == [0, 0, 0, 0, 1, 2, 3, 4]
    np.testing.assert_array_equal(shifted_wave(wave, 0), wave)
    with pytest.raises(ValueError):
        shifted_wave(wave, .5)


def test_crop_origin_and_end_coverage():
    for n in (192, 2000, 2004, 4532, 20000):
        assert crop_starts(n) == windows(n)
    assert crop_starts(4532, 125) == [0, 125, 1125, 2125, 2532]
    assert max(np.diff(crop_starts(4532, 125))) <= 1000


def test_inverse_alignment_of_absolute_event_positions():
    # A fractional event at original 43.7 ms, delayed by 8.75 ms.
    t = 10. * np.arange(10)
    acts = dict(beat=np.ones(10), beat_off=43.7 + 8.75 - t,
                level=np.tile([.1, .2, .3, .4], (10, 1)))
    aligned = align_evidence(acts, 8.75)
    np.testing.assert_allclose(t[:-1] + aligned['beat_off'][:-1], 43.7, atol=3e-6)
    np.testing.assert_allclose(aligned['level'].sum(-1), 1.)


def test_phase_wrap_and_duplicate_mean():
    a = dict(beat=np.ones(4), phase=np.array([.99, .01, .99, .01]))
    r = align_evidence(a, 5.)['phase'][0]
    assert min(abs(r), abs(1 - r)) < 1e-6
    b = dict(beat=np.ones(4), phase=1 - a['phase'])
    r = mean_evidence([a, b])['phase']
    assert np.max(np.minimum(abs(r), abs(1-r))) < 1e-6
    np.testing.assert_allclose(mean_evidence([a, a])['phase'], a['phase'], atol=1e-6)


def test_changed_map_and_topology_are_separate():
    from ensomi_n.mapformat import MusicMap
    from ensomi_n.timing import Segment
    from .shift_run import map_comparison, reproduction
    a = MusicMap.from_beats([100., 600., 1100., 1600.], downbeats=[0], segments=[Segment(100., 500.)])
    b = MusicMap.from_beats([103., 603., 1103., 1603.], downbeats=[0], segments=[Segment(103., 500.)])
    c = MusicMap.from_beats([106., 606., 1106., 1606.], downbeats=[0], segments=[Segment(106., 500.)])
    assert map_comparison(a, b, 0, 2000)['changed'] is False
    assert map_comparison(a, c, 0, 2000)['changed'] is True
    assert map_comparison(a, c, 0, 2000)['topology'] is False
    assert reproduction(a, a)['passed'] is True
    assert reproduction(a, b)['passed'] is False


def test_paired_reduction_and_recording_aggregation():
    from .shift_run import aggregate
    rows = []
    for seed in (0, 1):
        for rec in range(4):
            for arm in ('single', 'wave8', 'crop8'):
                rows.append(dict(rec=str(rec), seed=seed, arm=arm, strata=['S0'],
                                 changed=rec < (2 if arm == 'single' else 1), topology=False,
                                 r5=1., p5=1., bl20=.8, bars_start=.5, bars_same=.5, level=1.))
    result = aggregate(rows)
    assert result['s0/all/single']['changed']['value'] == .5
    assert result['s0/all/wave8']['relative_reduction']['value'] == .5
    assert result['s0/all/wave8']['relative_reduction']['denominator_changed'] == 2
    assert result['s0/all/wave8']['delta_bl20']['value'] == 0.
