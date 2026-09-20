from dataclasses import asdict, replace
import hashlib
import json

import numpy as np
import pytest
import torch

from pulsefield_model.research.bounded_typed_continuation.calibration import (
    COMPOSITIONS, FORMAT, GROUPS, calibrate, response_features,
)
from pulsefield_model.research.bounded_typed_continuation.contract import Arm, ROW_ACTIONS, Schedule, Timing
from pulsefield_model.research.bounded_typed_continuation.generation import Rollout, model_digest
from pulsefield_model.research.bounded_typed_continuation.model import BoundedModel, ModelConfig
from pulsefield_model.research.bounded_typed_continuation.response import response_costs
from pulsefield_model.research.bounded_typed_continuation.support import row_supports
from pulsefield_model.research.bounded_typed_continuation.train_hydra import compose_config
from pulsefield_model.research.bounded_typed_continuation.train_run import training_identity
from pulsefield_model.research.scoped_style_modeling.dataset import ContractError
from .test_generation import setup


def states():
    result=[]
    for mirrored in (False,True):
        timing=Timing((0.,50.,90.,117.,140.,200.),(True,True,False,True,True,False))
        state=Schedule(Arm.R1,timing)
        for a in [(1,0,0,0),(2,2,2,1)]:state,_=state.advance(a[::-1] if mirrored else a)
        result.append(state)
    return result + [s.advance((3,3,3,0) if i == 0 else (0,3,3,3))[0] for i,s in enumerate(result)]


@pytest.mark.parametrize('device',['cpu','mps'])
def test_calibration_preserves_composition_support_equal_cost_odds_and_finite_gradients(device):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    torch.manual_seed(81)
    queries=states();mask=torch.tensor(row_supports(queries),device=device)
    # Release-only states leave almost all composition groups empty.
    logits=torch.randn(len(queries),256,device=device,requires_grad=True)
    lp=logits.masked_fill(~mask,-torch.inf).log_softmax(-1)
    f=torch.tensor(response_features(queries),device=device)
    zero=torch.zeros(2,device=device)
    assert calibrate(lp,f,zero) is lp
    w=torch.tensor([2.,5.],device=device,requires_grad=True)
    out=calibrate(lp,f,w)
    assert torch.equal(torch.isfinite(out),mask)
    for group in range(len(COMPOSITIONS)):
        at=torch.tensor(GROUPS==group,device=device)
        torch.testing.assert_close(out[:,at].exp().sum(-1),lp[:,at].exp().sum(-1),atol=3e-7,rtol=2e-6)
    for i in range(len(queries)):
        for group in range(len(COMPOSITIONS)):
            ids=np.flatnonzero((GROUPS==group)&mask[i].cpu().numpy())
            for a,b in zip(ids[:-1],ids[1:]):
                if torch.equal(f[i,a],f[i,b]):
                    torch.testing.assert_close(out[i,a]-out[i,b],lp[i,a]-lp[i,b],atol=2e-6,rtol=0)
    loss=out.exp().mul(f.sum(-1)).sum();loss.backward()
    assert torch.isfinite(w.grad).all() and torch.isfinite(logits.grad).all() and w.grad[1]<0
    reverse=[ROW_ACTIONS.index(a[::-1]) for a in ROW_ACTIONS]
    torch.testing.assert_close(f[0],f[1,reverse],atol=0,rtol=0)
    reflected=calibrate(lp[0:1,reverse],f[1:2],w.detach())
    torch.testing.assert_close(reflected[0,reverse],out[0],atol=2e-6,rtol=0)
    for i,s in enumerate(queries):
        cost,legal=response_costs(s)
        np.testing.assert_array_equal(f[i].cpu().numpy().sum(-1)[legal],cost[legal])


@pytest.mark.parametrize('device',['cpu','mps'])
def test_native_calibrated_model_restore_and_configuration_identity(device):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    parent,timing,seed,_=setup(Arm.R1,device)
    model=BoundedModel(replace(parent.config,response_calibration=(2.,5.))).to(device).eval()
    model.load_state_dict(parent.state_dict())
    # Dense supplied timing makes the calibrated features active during rollout.
    timing=Timing(tuple(t*.2 for t in timing.times_ms),timing.onsets)
    seed=[replace(r,time_ms=r.time_ms*.2) for r in seed]
    run=Rollout.from_seed(model,timing,seed,{0:10,1:17});rng=torch.Generator().manual_seed(3)
    for _ in range(10):run.step(rng)
    copy,other=Rollout.restore(model,timing,run.snapshot(rng))
    while not run.state.finished:
        a,b=run.step(rng),copy.step(other)
        assert a==b
    assert torch.equal(rng.get_state(),other.get_state())
    assert model_digest(parent)!=model_digest(model)
    legacy=asdict(parent.config);legacy.pop('response_calibration')
    config=asdict(parent.config)
    assert training_identity({'model':legacy})==training_identity({'model':config})
    for field in ('seed_context',):
        if legacy[field]=='none':legacy.pop(field)
    for field,extra in [('long_memory',('memory_hidden','memory_stride')),('head_routing',('routing_hidden',)),('release_routing',('release_hidden',))]:
        if legacy[field]=='none':
            for key in (field,*extra):legacy.pop(key)
    digest=hashlib.sha256(json.dumps(legacy,sort_keys=True).encode())
    for name,tensor in parent.state_dict().items():
        digest.update(json.dumps((name,str(tensor.dtype),tuple(tensor.shape))).encode());digest.update(tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    assert model_digest(parent)==digest.hexdigest()


def test_config_and_bundle_require_explicit_fitted_provenance(tmp_path):
    from pulsefield_model.research.bounded_typed_continuation.generate_run import _model
    from pulsefield_model.research.bounded_typed_continuation.generate_config import GenerateConfig
    from pulsefield_model.research.oracle_time_continuation.storage import file_digest
    from pulsefield_model.research.bounded_typed_continuation.smoke_hydra import compose_config as smoke_config
    assert compose_config([]).model.response_calibration is None
    with pytest.raises(ContractError,match='fit response calibration separately'):
        compose_config(['model.arm=R1','model.response_calibration=[1,2]'])
    with pytest.raises(ContractError,match='fit response calibration separately'):
        smoke_config(['model.arm=R1','model.response_calibration=[1,2]'])
    for value in ((-1.,2.),(1.,float('nan')),(1.,),(True,2.)):
        with pytest.raises(ContractError):ModelConfig(Arm.R1,response_calibration=value)
    with pytest.raises(ContractError):ModelConfig(Arm.O1,response_calibration=(1.,2.))
    model=BoundedModel(ModelConfig(Arm.R1,hidden=8,levels=2,coupling_rank=2,response_calibration=(1.,2.)))
    value=dict(format=FORMAT,source_revision='a'*40,config={'model':json.loads(json.dumps(asdict(model.config)))},model=model.state_dict(),
               calibration=dict(parent_checkpoint_sha256='b'*64,data_sha256='c'*64,fit={'weights':[1.,2.]}))
    path=tmp_path/'model.pt'
    def load():
        torch.save(value,path)
        return _model(GenerateConfig(checkpoint_file=str(path),checkpoint_sha256=file_digest(path),device='cpu'))
    loaded,provenance=load();assert loaded.config==model.config and provenance['calibration']==value['calibration']
    value['format']='bounded-typed/corpus-training-v1'
    with pytest.raises(ContractError,match='model-only'):load()
    value['format']=FORMAT;value['calibration']['fit']['weights']=[2.,2.]
    with pytest.raises(ContractError,match='recorded fit'):load()
