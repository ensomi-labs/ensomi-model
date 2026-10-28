"""Readout invariants and matched recording statistics."""
import numpy as np
from n.experiments.n1_redesign.gpu.evaluate import auc,maxima
from n.experiments.n1_redesign.gpu.report import comparison


def test_auc_ties_and_missing_class():
    assert auc([0,0,1,1],[.1,.2,.3,.4])==1
    assert auc([0,1],[.5,.5])==.5
    assert auc([1,1],[.2,.4]) is None


def test_cell_interval_uses_frame_centres_and_excludes_next_beat():
    x=np.array([.1,.2,.3,.9,.4])
    np.testing.assert_allclose(maxima(x,[.5,25.5],[25.5,40.]),[.3,.9])


def test_comparison_pairs_only_matched_finite_recordings_and_uses_half_width():
    b0={'a':.4,'b':.4,'extra':99.};b1={'a':.45,'b':.45}
    f0={'a':.5,'b':.5};f1={'a':.6,'b':.6}
    r=comparison(b0,b1,f0,f1,0)
    assert r['n']==2
    np.testing.assert_allclose(r['margin'],.05)
    np.testing.assert_allclose(r['F1_minus_v01']['value'],.2)
    assert r['gain_prediction'] and not r['preserved'] and not r['harmed']
    assert comparison(b0,{},f0,f1,0) is None
