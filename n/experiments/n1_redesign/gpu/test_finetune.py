"""Fine-tune wrappers preserve pulse outputs and the original random sampling stream."""
from pathlib import Path
import numpy as np
import torch
from ensomi_n.timing import Reference,Segment
from ensomi_n.v0.labels import V0Labels
from ensomi_n.v0.data import Song,Sampler
from ensomi_n.v0.model import Heads
from n.experiments.n1_redesign.gpu import finetune
from n.experiments.n1_redesign.gpu.targets import compact


def test_floor_changes_only_period_sigma():
    old=Heads(8);new=finetune.FloorHeads(8,.012)
    new.load_state_dict(old.state_dict())
    x=torch.randn(2,7,8)
    a,b=old(x),new(x)
    for k in a:
        if k=='logp_sigma':torch.testing.assert_close(b[k]-a[k],torch.full_like(a[k],.011))
        else:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)


def test_sampler_preserves_crops_weights_memory_and_rng(tmp_path,monkeypatch):
    mel=tmp_path/'mel.npy';memo=tmp_path/'memo.npy'
    rng=np.random.default_rng(3)
    np.save(mel,rng.normal(size=(6000,128)).astype(np.float16))
    np.save(memo,rng.normal(size=(1500,64)).astype(np.float16))
    ref=Reference.from_segments([Segment(.5,60000/123.45,4)],np.arange(1000.,55000.,250.))
    lab=V0Labels.from_reference(ref)
    song=Song('synthetic',str(mel),str(memo),0.,1.,6000,lab)
    monkeypatch.setattr(finetune,'load_target',lambda key:compact(ref))
    a=Sampler([song],seed=2001);b=finetune.TargetSampler([song],seed=2001)
    for _ in range(2):
        x,tg,m,mask,owner=a.batch(2);xx,tt,mm,mk,oo=b.batch(2)
        for v,w in ((x,xx),(m,mm),(mask,mk),(owner,oo)):np.testing.assert_array_equal(v,w)
        for k in tg:np.testing.assert_array_equal(tg[k],tt[k])
        assert a.state()==b.state()


def test_auxiliary_masks_cannot_train_unknown_frames():
    y=torch.randn(2,5,7,requires_grad=True)
    tg=dict(attack_y=torch.zeros(2,5),attack_w=torch.zeros(2,5),snap_y=torch.zeros(2,5,dtype=torch.long),
            snap_w=torch.zeros(2,5),sal_y=torch.zeros(2,5,3),sal_w=torch.zeros(2,5))
    loss,_=finetune.auxiliary_loss(y,tg,3.)
    loss.backward()
    assert loss.item()==0 and torch.count_nonzero(y.grad)==0
