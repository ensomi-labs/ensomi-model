import numpy as np
import pytest
from pulsefield_model.research.bounded_typed_continuation.calibration import COMPOSITIONS, GROUPS, fit
from pulsefield_model.research.scoped_style_modeling.dataset import ContractError


def test_fit_recovers_soft_finite_cost_and_rejects_invalid_preferences():
    rng=np.random.default_rng(2);logits=rng.normal(size=(8,256))
    lp=logits-np.log(np.exp(logits).sum(-1,keepdims=True))
    f=np.zeros((8,256,2));f[:,::2,0]=1.;f[:,1::3,1]=1.
    family=np.tile(GROUPS==COMPOSITIONS.index((1,0)),(8,1))
    f[:,np.flatnonzero(family[0])[0],:]=0.
    targets=np.array([np.flatnonzero(family[i]&(f[i].sum(-1)==0))[0] for i in range(8)])
    result=fit(lp,f,targets,8,lp,f,family,np.array(['a']*3+['b']*5))
    assert result['final']['source_nll']<result['initial']['source_nll']
    assert result['final']['native_nll']<.02 and all(0<w<20 for w in result['weights'])
    with pytest.raises(ContractError,match='zero and positive'):
        fit(lp,f,targets,8,lp,f,family&(f.sum(-1)==0),np.arange(8))
