"""Absolute clock, masking and memory-free architecture checks for the event experiment."""
import numpy as np
import torch
from n.experiments.n1_redesign.gpu.g3.train import EventModel, learning_rate
from n.experiments.n1_redesign.gpu.g3.infer import song_logits
from n.experiments.n1_redesign.gpu.g3.mel import stack_frames
from n.experiments.n1_redesign.gpu.finetune import auxiliary_loss


def test_memory_free_model_has_only_event_outputs_and_encoder_receives_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(3000)
    model = EventModel().train()
    assert model.encoder.memory is None
    assert not any('cross' in name or 'memory' in name or 'heads.' in name for name,_ in model.named_parameters())
    y = model(torch.randn(2,128,128))
    assert y.shape == (2,128,7)
    y.square().mean().backward()
    assert model.encoder.stem.conv[0].weight.grad.abs().sum() > 0


def test_window_clock_including_short_and_fractional_final_window(tmp_path):
    from types import SimpleNamespace
    class Clock(torch.nn.Module):
        def forward(self,x):
            return x[...,:7]
    for n in (7,2001,4537):
        x = np.tile(np.arange(n,dtype=np.float32)[:,None],(1,128))
        p = tmp_path/f'{n}.npy'
        np.save(p,x)
        song = SimpleNamespace(mel=p,mean=1.,std=2.)
        y = song_logits(Clock(),song,'cpu',batch=3)
        np.testing.assert_array_equal(y,np.repeat(((np.arange(n)-1)/2)[:,None],7,axis=1))


def test_mel_lag_order_and_endpoint_clock():
    m = np.repeat(np.arange(5,dtype=np.float32)[:,None],128,axis=1)
    x = stack_frames(m,np.array([0,2,4]),0.,1.).reshape(3,5,128)
    np.testing.assert_array_equal(x[:,:,0],[[0,0,0,1,2],[0,1,2,3,4],[2,3,4,4,4]])


def test_event_loss_masks_missing_notes_and_breaks_without_pulse_loss():
    y = torch.zeros((1,3,7),requires_grad=True)
    tg = dict(attack_y=torch.tensor([[1.,0.,0.]]),attack_w=torch.ones(1,3),
              snap_y=torch.zeros((1,3),dtype=torch.long),snap_w=torch.zeros(1,3),
              sal_y=torch.zeros(1,3,3),sal_w=torch.zeros(1,3))
    tg['attack_w'][0,1] = 0
    loss,_ = auxiliary_loss(y,tg,1.,torch.tensor([[1.,1.,0.]]))
    loss.backward()
    assert np.isclose(float(loss.detach()),.5*np.log(2))
    assert y.grad[0,0,0] < 0
    assert y.grad[0,1:].abs().sum() == 0
    assert y.grad[...,1:].abs().sum() == 0


def test_lr_warmup_and_final_floor():
    assert learning_rate(0,0)==.001/500
    assert learning_rate(499,400)==.001
    assert np.isclose(learning_rate(2000,3000),.00005)


def test_loader_and_file_readouts_are_identical_without_feature_cache(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from n.experiments.n1_redesign.gpu import evaluate as e
    from n.experiments.n1_redesign.gpu.targets import compact
    from ensomi_n.timing import Reference,Segment
    ref = Reference.from_segments([Segment(3.25,60000/123.45,4)],np.arange(100.,12000.,100.))
    target = compact(ref)
    target['cell'][:] = -1
    item = dict(item='synthetic',key='one',panel='panel',rec='recording',ref=ref)
    (tmp_path/'scores').mkdir()
    (tmp_path/'scores/v01-s0-009001.json').write_text(json.dumps(dict(recordings=[dict(panel='panel',rec='recording',offset_ms=0.,cell='all')])))
    (tmp_path/'eval/test').mkdir(parents=True)
    y = np.random.default_rng(3).normal(size=(1300,7)).astype(np.float32)
    np.savez(tmp_path/'eval/test/synthetic.npz',new_logits=y)
    monkeypatch.setattr(e,'OUT',tmp_path)
    monkeypatch.setattr(e,'FEAT',tmp_path/'no_feature_cache')
    monkeypatch.setattr(e,'panel_items',lambda:[item])
    monkeypatch.setattr(e,'variant_key',lambda x:'variant')
    monkeypatch.setattr(e,'load_target',lambda *args:target)
    a = e.head_rows('test',0)
    b = e.head_rows('test',0,logits_loader=lambda sha:y,output_path=tmp_path/'custom.json')
    assert a == b
    assert a[0]['attack_auc'] is not None


def test_paired_bootstrap_excludes_missing_recordings_and_keeps_absolute_floor():
    from n.experiments.n1_redesign.gpu.g3.statistics import paired
    r = paired({'a':.5,'b':.6,'missing':1.},{'a':.6,'b':.7})
    assert r['n']==2 and r['control_unmatched']==1
    assert r['margin']==.01
    assert np.isclose(r['delta']['value'],.1)
