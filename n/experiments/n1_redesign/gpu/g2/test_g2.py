"""Absolute band boundaries, feature normalisation and unchanged sampling."""
import numpy as np
import torch
from n.experiments.n1_redesign.gpu.g2.targets import expand
from n.experiments.n1_redesign.gpu.g2.x5d_targets import targets
from n.experiments.n1_redesign.gpu.g2.readout import fold
from n.experiments.n1_redesign.gpu.g2.finetune import auxiliary_loss,TargetSampler


def test_red_line_boundary_song_end_and_stretch_change_absolute_band():
    d=dict(lo_ms=123.25,hi_ms=1100.5,segments=[(123.25,750.),(888.75,375.)],unscorable=[(500.,600.)])
    times=np.array([123.24,123.25,500.,600.,888.74,888.75,1100.5,1100.51])
    y,v=targets([expand(d)],times)
    np.testing.assert_array_equal(v,[0,1,0,1,1,1,1,0])
    np.testing.assert_array_equal(y[v].argmax(1),[2,2,2,3,3])
    ys,vs=targets([expand(d,1.1)],times*1.1)
    np.testing.assert_array_equal(vs,v)
    np.testing.assert_array_equal(ys[v].argmax(1),[1,1,1,2,2])


def test_folded_standardisation_and_temperature():
    rng=np.random.default_rng(1)
    h=dict(mean=rng.normal(size=160).astype('f'),sd=np.full(160,2.,dtype='f'),
           w0=rng.normal(size=(7,160)).astype('f'),b0=np.arange(7,dtype='f'),temperature=np.array(2.))
    w,b=fold(h);x=np.tile(h['mean'],(2,1));x[1,0]+=2
    np.testing.assert_allclose(x@w.T+b,np.stack([h['b0']/2,(h['b0']+h['w0'][:,0])/2]),atol=2e-5)


def test_band_loss_does_not_penalise_unmapped_or_break_masked_frames():
    y=torch.zeros((1,3,7),requires_grad=True)
    t=dict(band_y=torch.nn.functional.one_hot(torch.tensor([[0,1,2]]),7).float(),band_w=torch.tensor([[1.,0.,1.]]))
    loss,_=auxiliary_loss(y,t,break_weight=torch.tensor([[1.,1.,0.]]));loss.backward()
    assert float(loss.detach())>0
    assert y.grad[0,0].abs().sum()>0
    assert y.grad[0,1:].abs().sum()==0


def test_out_of_support_augmentation_masks_only_affected_band_frames():
    from n.experiments.n1_redesign.gpu.g2.targets import mask_unsupported
    d=dict(lo_ms=0.,hi_ms=2000.,segments=[(0.,3000.),(1000.,500.)],unscorable=[])
    ref=expand(d,1.12);times=np.array([0.,1119.99,1120.,2240.])
    y,valid=targets(mask_unsupported([ref]),times)
    np.testing.assert_array_equal(valid,[False,False,True,True])
    np.testing.assert_array_equal(y[valid].argmax(1),[2,2])
    assert ref.scorable(times).all()
    assert len(ref.unscorable)==0


def test_absolute_band_projection_preserves_correct_pulse_and_fixes_octave():
    from n.experiments.n1_redesign.gpu.g2.bands import ratios_from_probabilities,CFG
    p=np.full((100,7),.001);p[:,1]=.994
    assert ratios_from_probabilities(p,[(0,100,False)],[60000./79],CFG)==[1.]
    assert ratios_from_probabilities(p,[(0,100,False)],[60000./158],CFG)==[.5]


def test_sampler_preserves_f0_crops_and_random_stream(tmp_path,monkeypatch):
    from n.experiments.n1_redesign.gpu.g2 import finetune as g2
    from n.experiments.n1_redesign.gpu.g2 import targets as target_module
    from ensomi_n.v0.data import Sampler,Song
    from ensomi_n.v0.labels import V0Labels
    from ensomi_n.timing import Reference,Segment
    rng=np.random.default_rng(3);mel=tmp_path/'mel.npy';memo=tmp_path/'memo.npy'
    np.save(mel,rng.normal(size=(6000,128)).astype(np.float16))
    np.save(memo,rng.normal(size=(1500,64)).astype(np.float16))
    ref=Reference.from_segments([Segment(.5,60000/123.45,4)],np.arange(1000.,55000.,250.))
    song=Song('synthetic',str(mel),str(memo),0.,1.,6000,V0Labels.from_reference(ref))
    monkeypatch.setattr(target_module,'database',lambda:{'synthetic':{'one':target_module.compact(ref)}})
    monkeypatch.setattr(target_module,'domain_decision',lambda:'strict')
    a=Sampler([song],seed=2001);b=g2.TargetSampler([song],seed=2001)
    for _ in range(2):
        x,t,m,mask,owner=a.batch(2);xx,tt,mm,mk,oo=b.batch(2)
        for v,w in ((x,xx),(m,mm),(mask,mk),(owner,oo)):np.testing.assert_array_equal(v,w)
        for k in t:np.testing.assert_array_equal(t[k],tt[k])
        assert a.state()==b.state()
