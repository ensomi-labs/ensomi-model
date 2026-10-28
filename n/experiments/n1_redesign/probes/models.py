"""Small CPU probes with recording bootstrap weights and dev-only stopping."""
import copy
import time

import numpy as np
import torch
from torch import nn

from .common import OUT, dump, log, prereg_check

torch.set_num_threads(4)
torch.set_num_interop_threads(1)


def architecture(width, kind):
    if kind == 'logistic':
        return nn.Linear(width, 1)
    return nn.Sequential(nn.Linear(width, 256), nn.GELU(), nn.Linear(256, 1))


def predict(model, x, batch=8192):
    net = architecture(x.shape[1], model['kind'])
    net.load_state_dict(model['state'])
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            a = (np.asarray(x[i:i + batch], np.float32) - model['mean']) / model['sd']
            out.append(net(torch.from_numpy(a)).flatten().numpy())
    return np.concatenate(out) if out else np.empty(0)


class Predictor:
    def __init__(self, model):
        self.model = model
        self.net = architecture(len(model['mean']), model['kind'])
        self.net.load_state_dict(model['state'])
        self.net.eval()

    def __call__(self, x, batch=8192):
        out = []
        with torch.no_grad():
            for i in range(0, len(x), batch):
                a = (np.asarray(x[i:i + batch], np.float32) - self.model['mean']) / self.model['sd']
                out.append(self.net(torch.from_numpy(a)).flatten().numpy())
        return np.concatenate(out) if out else np.empty(0)


def fit(x, y, rec, xd, yd, *, kind, seed, bootstrap=None, epochs=24):
    prereg_check()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sx, ss = np.zeros(x.shape[1], float), np.zeros(x.shape[1], float)
    for start in range(0, len(x), 8192):
        chunk = np.asarray(x[start:start + 8192], np.float64)
        sx += chunk.sum(axis=0)
        ss += np.square(chunk).sum(axis=0)
    mean = (sx / len(x)).astype('float32')
    sd = np.maximum(np.sqrt(np.maximum(ss / len(x) - np.square(sx / len(x)), 0)), .01).astype('float32')
    net = architecture(x.shape[1], kind)
    opt = torch.optim.AdamW(net.parameters(), lr=.002 if kind == 'mlp' else .01, weight_decay=1e-4)
    weights = np.ones(len(y), np.float32)
    if bootstrap is not None:
        brng = np.random.default_rng(6100 + bootstrap)
        unique, inv = np.unique(rec, return_inverse=True)
        count = np.bincount(brng.integers(len(unique), size=len(unique)), minlength=len(unique))
        weights = count[inv].astype('float32')
    # Each recording receives equal total training weight, independent of
    # chart density or the number of reference sets sharing its audio.
    unique, inv, counts = np.unique(rec, return_inverse=True, return_counts=True)
    weights /= counts[inv]
    weights *= len(weights) / max(weights.sum(), 1)
    target = torch.as_tensor(y, dtype=torch.float32)
    best, stale, history = float('inf'), 0, []
    best_state = copy.deepcopy(net.state_dict())
    for epoch in range(epochs):
        net.train()
        order = rng.permutation(len(y))
        for start in range(0, len(y), 2048):
            idx = order[start:start + 2048]
            a = (np.asarray(x[idx], np.float32) - mean) / sd
            logits = net(torch.from_numpy(a)).flatten()
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target[idx], reduction='none')
            loss = (loss * torch.from_numpy(weights[idx])).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        net.eval()
        total = 0.
        with torch.no_grad():
            for start in range(0, len(yd), 8192):
                a = (np.asarray(xd[start:start + 8192], np.float32) - mean) / sd
                p = net(torch.from_numpy(a)).flatten()
                total += float(nn.functional.binary_cross_entropy_with_logits(
                    p, torch.as_tensor(yd[start:start + 8192], dtype=torch.float32), reduction='sum'))
        loss = total / max(len(yd), 1)
        history.append(loss)
        if loss < best - 1e-5:
            best, stale, best_state = loss, 0, copy.deepcopy(net.state_dict())
        else:
            stale += 1
        if stale >= 3:
            break
    return dict(kind=kind, mean=mean, sd=sd, state=best_state, dev_bce=best,
                epochs=len(history), history=history, seed=seed, bootstrap=bootstrap)


def save(path, model):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, path)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)
