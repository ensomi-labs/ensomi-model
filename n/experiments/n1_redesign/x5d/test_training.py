"""Synthetic evidence that fitting, calibration and exported inference work."""
from common import *
from train import fit_head, logits_for, nll, temperature
from predict import head_logits
import torch
import pytest


@pytest.mark.parametrize('kind', ['linear', 'mlp'])
def test_readout_learns_feature_labels_and_exports_exactly(kind):
    rng = np.random.default_rng(41)
    x = rng.normal(size=(2048,160)).astype(np.float32)
    y = np.eye(7, dtype=np.float32)[np.argmax(x[:, :7], axis=1)]
    data = dict(x=x, y=y, weight=np.ones(len(x), np.float32))
    mean = np.zeros(160, np.float32); sd = np.ones(160, np.float32)
    model, info = fit_head(kind, data, data, mean, sd, 41, epochs=20)
    assert info['curve'][-1]['cal_nll'] < info['curve'][0]['cal_nll']-.2
    logits = logits_for(model, x, mean, sd)
    assert nll(logits, y).mean() < nll(logits[::-1], y).mean()
    head = dict(mean=mean, sd=sd)
    for i, layer in enumerate(m for m in model if isinstance(m, torch.nn.Linear)):
        head[f'w{i}'] = layer.weight.detach().numpy()
        head[f'b{i}'] = layer.bias.detach().numpy()
    np.testing.assert_allclose(head_logits(head, x), logits, atol=2e-6)


def test_temperature_improves_overconfident_predictions():
    rng = np.random.default_rng(19)
    z = rng.normal(size=(1000,7))
    p = np.exp(z-z.max(1,keepdims=True)); p /= p.sum(1,keepdims=True)
    target = np.eye(7)[[rng.choice(7, p=row) for row in p]]
    logits = 5*z
    t = temperature(logits, target, np.ones(len(z)))
    assert 3 < t < 7
    assert nll(logits, target, t).mean() < nll(logits, target).mean()
