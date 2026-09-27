import numpy as np
import pytest
import torch

from ensomi_model.research.gameplay_evaluation.waiting import first_event_law,compare_waiting_laws
from ensomi_model.research.joint_audio_continuation.timing import sample_hazards


def test_censored_geometric_wait_is_not_an_expected_rollout_count():
    p=.001;n=2000
    law=first_event_law(np.full(n,np.log(p/(1-p))),np.ones(n,bool))
    report=law.report()
    assert report['event_probability']==pytest.approx(1-(1-p)**n)
    assert report['restricted_mean_wait_ms']==pytest.approx((1-(1-p)**n)/p)
    assert report['survival']['1000']==pytest.approx((1-p)**1000)
    assert report['median_wait_ms']==693
    assert law.event_mass.sum()+law.survival[-1]==pytest.approx(1)


def test_invalid_clocks_and_certain_event_preserve_elapsed_time():
    law=first_event_law([np.nan,0,np.nan,2],np.array([False,True,True,True]),
                        forced=np.array([True,False,True,False]))
    np.testing.assert_allclose(law.event_mass,[0,.5,.5,0])
    assert law.restricted_mean_ms==2.5
    assert law.survival[-1]==0
    no_event=first_event_law([np.nan]*4,np.zeros(4,bool))
    assert no_event.report()['median_wait_ms'] is None
    assert no_event.restricted_mean_ms==4


def test_equal_event_amount_retains_timing_and_censoring_distinctions():
    a=first_event_law([np.inf,-np.inf,-np.inf],np.ones(3,bool))
    b=first_event_law([-np.inf,-np.inf,np.inf],np.ones(3,bool))
    c=first_event_law([-np.inf]*3,np.ones(3,bool))
    assert a.report()['event_probability']==b.report()['event_probability']==1
    result=compare_waiting_laws(a,b)
    assert result['total_variation']==1
    assert result['restricted_wasserstein_ms']==result['restricted_mean_change_ms']==2
    assert compare_waiting_laws(b,c)['total_variation']==1
    assert compare_waiting_laws(b,c)['restricted_wasserstein_ms']==0


def test_waiting_cdf_matches_native_exponential_budget_sampler():
    logits=torch.tensor([-3.,1.,-2.,-.5,.2],dtype=torch.float64)
    valid=torch.tensor([True,False,True,True,True]);forced=torch.zeros(5,dtype=torch.bool)
    law=first_event_law(logits.numpy(),valid.numpy())
    generator=torch.Generator().manual_seed(274700)
    for q in (.01,.05,.2,.49,.9,.99):
        actual,_=sample_hazards(logits,valid,forced,generator,-np.log1p(-q))
        indices=np.flatnonzero(1-law.survival>=q)
        assert actual==(int(indices[0]) if len(indices) else None)
