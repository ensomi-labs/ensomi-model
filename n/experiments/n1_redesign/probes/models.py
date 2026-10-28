"""Small CPU probes with recording bootstrap weights and dev-only stopping."""
import copy
import mmap
import time

import numpy as np
import torch
from torch import nn

from .common import OUT, dump, log, prereg_check

torch.set_num_threads(4)
torch.set_num_interop_threads(1)


class DiskMatrix:
    """A matrix whose mappings exist only while a batch is copied or written."""
    def __init__(self, path, shape, dtype=np.float16):
        self.path, self.shape, self.dtype = path, shape, np.dtype(dtype)
        with open(path, 'wb') as stream:
            stream.truncate(int(np.prod(shape)) * self.dtype.itemsize)

    def __len__(self):
        return self.shape[0]

    def write(self, start, values):
        mapped = np.memmap(self.path, dtype=self.dtype, mode='r+',
            offset=start * self.shape[1] * self.dtype.itemsize,
            shape=(len(values), self.shape[1]))
        try:
            mapped[:] = values
            mapped.flush()
        finally:
            mapped._mmap.close()

    def read(self, index, dtype):
        mapped = np.memmap(self.path, dtype=self.dtype, mode='r', shape=self.shape)
        try:
            return np.array(mapped[index], dtype=dtype, copy=True)
        finally:
            mapped._mmap.close()


def batch_array(x, index, dtype):
    """Copy a batch, then release disk-backed source pages from resident memory."""
    if isinstance(x, DiskMatrix):
        return x.read(index, dtype)
    a = np.asarray(x[index], dtype=dtype)
    if isinstance(x, np.memmap):
        if np.shares_memory(a, x):
            a = a.copy()
        x._mmap.madvise(mmap.MADV_DONTNEED)
    return a


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
    unique, inv, counts = np.unique(rec, return_inverse=True, return_counts=True)
    copies = np.ones(len(unique))
    if bootstrap is not None:
        brng = np.random.default_rng(6100 + bootstrap)
        copies = np.bincount(brng.integers(len(unique), size=len(unique)), minlength=len(unique))
    resample_weight = copies[inv]
    sx, ss = np.zeros(x.shape[1], float), np.zeros(x.shape[1], float)
    for start in range(0, len(x), 8192):
        chunk = batch_array(x, slice(start, start + 8192), np.float64)
        w = resample_weight[start:start + len(chunk), None]
        sx += (chunk * w).sum(axis=0)
        ss += (np.square(chunk) * w).sum(axis=0)
    mass = resample_weight.sum()
    mean = (sx / mass).astype('float32')
    sd = np.maximum(np.sqrt(np.maximum(ss / mass - np.square(sx / mass), 0)), .01).astype('float32')
    net = architecture(x.shape[1], kind)
    opt = torch.optim.AdamW(net.parameters(), lr=.002 if kind == 'mlp' else .01, weight_decay=1e-4)
    weights = resample_weight.astype('float32')
    # Each recording receives equal total training weight, independent of
    # chart density or the number of reference sets sharing its audio.
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
            a = (batch_array(x, idx, np.float32) - mean) / sd
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
                a = (batch_array(xd, slice(start, start + 8192), np.float32) - mean) / sd
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
