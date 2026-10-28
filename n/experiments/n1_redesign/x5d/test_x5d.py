"""Absolute band boundaries, missing targets, soft chart labels and D2 adapter parity."""
from types import SimpleNamespace
from common import *
from bands import own_probabilities, ratios_from_probabilities, pooled_probabilities
from ensomi_n.timing import Reference, Segment
from decoder_measurements import CFG, dec
import level


def test_absolute_band_boundaries():
    bpm = np.array([20., 39.99, 40., 79.99, 80., 159.99, 160., 319.99, 320., 640.])
    np.testing.assert_array_equal(band_of(60000./bpm), [-2,-2,-1,-1,0,0,1,1,2,3])


def test_targets_use_notated_step_and_exclude_unmapped_frames():
    ref = Reference.from_segments([Segment(100., 500., 4), Segment(1455.5, 250., 4)],
                                 np.arange(100., 3001., 250.), lo_ms=100., hi_ms=3000.)
    times = np.array([0., 100., 1455., 1455.5, 3000., 3100.])
    y, valid = targets([ref], times)
    np.testing.assert_array_equal(valid, [False, True, True, True, True, False])
    np.testing.assert_array_equal(np.argmax(y[valid], axis=1)+BANDS[0], [0,0,1,1])
    assert not y[~valid].any()


def test_conflicting_charts_average_soft_labels():
    a = Reference.from_segments([Segment(0., 500., 4)], [0., 2000.], lo_ms=0., hi_ms=2000.)
    b = Reference.from_segments([Segment(0., 250., 4)], [0., 2000.], lo_ms=0., hi_ms=2000.)
    y, _ = targets([a,b], np.array([100., 1000.]))
    np.testing.assert_array_equal(y[:, 2:4], np.full((2,2), .5))


def test_own_band_adapter_matches_original_d2():
    rng = np.random.default_rng(8)
    ev = dec.Evidence(np.ones(100), level=rng.uniform(.01,1,(100,4)),
        logp=rng.uniform(np.log(.38),np.log(.77),100), logp_sigma=rng.uniform(.01,.1,100))
    groups = [(0,27,False),(27,70,False),(70,100,True)]
    periods = [60000./79,60000./163,500.]
    assert ratios_from_probabilities(own_probabilities(ev),groups,periods,CFG) == level.band_ratios(ev,groups,periods,CFG)


def test_absolute_band_does_not_halve_already_correct_pulse():
    p = np.full((100,7), .001); p[:,1] = .994
    assert ratios_from_probabilities(p,[(0,100,False)],[60000./79],CFG)==[1.]
    assert ratios_from_probabilities(p,[(0,100,False)],[60000./158],CFG)==[.5]


def test_pool_endpoints_preserve_argmax():
    a=np.array([[.01,.02,.9,.02,.02,.02,.01]])
    b=np.array([[.01,.9,.02,.02,.02,.02,.01]])
    assert np.argmax(pooled_probabilities(a,b,1))==2
    assert np.argmax(pooled_probabilities(a,b,0))==1
