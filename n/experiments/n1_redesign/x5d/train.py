"""Fit and calibrate the two preregistered CPU readouts on frozen features."""
import argparse
from datetime import datetime, timezone
import subprocess
import time

from common import *
from bands import pooled_probabilities
from measurements import interval
from scipy.special import log_softmax, softmax
from scipy.optimize import minimize_scalar
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def nll(logits, y, temperature=1.):
    return -(y * log_softmax(logits / temperature, axis=1)).sum(1)


def temperature(logits, y, weight):
    result = minimize_scalar(lambda logt: float(np.average(nll(logits, y, np.exp(logt)), weights=weight)),
        bounds=(np.log(.05), np.log(20.)), method='bounded')
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError('Temperature optimisation failed')
    return float(np.exp(result.x))


def logits_for(model, x, mean, sd):
    model.eval()
    out = []
    with torch.no_grad():
        for lo in range(0, len(x), 8192):
            batch = (np.asarray(x[lo:lo+8192]) - mean) / sd
            out.append(model(torch.from_numpy(batch)).numpy())
    return np.concatenate(out)


def fit_head(kind, fit, cal, mean, sd, rng_seed, epochs=60):
    torch.manual_seed(rng_seed)
    rng = np.random.default_rng(rng_seed)
    model = (torch.nn.Sequential(torch.nn.Linear(160, 7)) if kind == 'linear' else
             torch.nn.Sequential(torch.nn.Linear(160, 64), torch.nn.ReLU(), torch.nn.Linear(64, 7)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01 if kind == 'linear' else .001, weight_decay=1e-4)
    best, significant, stale = float('inf'), float('inf'), 0
    state, curve = None, []
    for epoch in range(1, epochs+1):
        model.train()
        order = rng.permutation(len(fit['x']))
        train_sum = train_weight = 0.
        for lo in range(0, len(order), 4096):
            ix = order[lo:lo+4096]
            x = torch.from_numpy((fit['x'][ix] - mean) / sd)
            y = torch.from_numpy(np.asarray(fit['y'][ix]))
            w = torch.from_numpy(np.asarray(fit['weight'][ix]))
            losses = -(y * torch.log_softmax(model(x), dim=1)).sum(1)
            loss = (w * losses).sum() / w.sum()
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite fit loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_sum += float((w * losses.detach()).sum())
            train_weight += float(w.sum())
        logits = logits_for(model, cal['x'], mean, sd)
        value = float(np.average(nll(logits, cal['y']), weights=cal['weight']))
        if not np.isfinite(value):
            raise RuntimeError('Non-finite calibration loss')
        curve.append(dict(epoch=epoch, fit_nll=train_sum/train_weight, cal_nll=value))
        # Save the earliest strict minimum; the patience threshold is separate.
        if value < best:
            best, best_epoch = value, epoch
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if value < significant - 1e-4:
            significant, stale = value, 0
        else:
            stale += 1
        print(kind, epoch, 'cal_nll', round(value, 6), 'patience', stale, flush=True)
        if stale >= 8:
            break
    model.load_state_dict(state)
    return model, dict(best_epoch=best_epoch, curve=curve)


def recording_values(losses, data):
    weight, owner = data['weight'], data['owner']
    return np.bincount(owner, weights=losses*weight) / np.bincount(owner, weights=weight)


def train(seed):
    check_prereg()
    folder = OUT / seed
    data_path = folder / 'data'
    metadata = json.loads((data_path / 'metadata.json').read_text())
    data = {split: {name: np.load(data_path / f'{split}-{name}.npy', mmap_mode='r')[:metadata['counts'][split]['samples']]
                   for name in ('x', 'y', 'own', 'weight', 'owner')} for split in ('fit', 'cal')}
    for split in data.values():
        if not all(np.isfinite(a).all() for a in split.values()):
            raise RuntimeError('Non-finite training data')
    with np.load(data_path / 'normalizer.npz') as z:
        mean, sd = z['mean'], z['sd']
    destination = folder / 'heads'
    destination.mkdir(exist_ok=False)
    summary = dict(seed=seed, started_utc=datetime.now(timezone.utc).isoformat(),
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        prereg=json.loads((OUT / 'prereg-identity.json').read_text()),
        cohort=metadata, device='cpu', threads=1, heads={})
    start = time.monotonic()
    for kind in ('linear', 'mlp'):
        model, info = fit_head(kind, data['fit'], data['cal'], mean, sd, 20261029+int(seed[1]))
        logits = logits_for(model, data['cal']['x'], mean, sd).astype(np.float64)
        cal = data['cal']
        temp = temperature(logits, cal['y'], cal['weight'])
        losses = nll(logits, cal['y'], temp)
        permuted = logits[np.random.default_rng(20261029).permutation(len(logits))]
        shuffled = nll(permuted, cal['y'], temp)
        original_rec, shuffled_rec = recording_values(losses, cal), recording_values(shuffled, cal)
        info.update(temperature=temp, cal_nll=interval(original_rec),
            uncalibrated_cal_nll=interval(recording_values(nll(logits, cal['y']), cal)),
            shuffled_nll=interval(shuffled_rec), shuffle_delta=interval(shuffled_rec-original_rec))
        arrays = dict(mean=mean, sd=sd, temperature=np.array(temp))
        for i, layer in enumerate(m for m in model if isinstance(m, torch.nn.Linear)):
            arrays[f'w{i}'] = layer.weight.detach().numpy()
            arrays[f'b{i}'] = layer.bias.detach().numpy()
        path = destination / f'{kind}.npz'
        np.savez(path, **arrays)
        np.save(destination / f'{kind}-cal-logits.npy', logits)
        info['sha256'] = sha(path)
        if kind == 'linear':
            p = softmax(logits/temp, axis=1)
            candidates = []
            for w in (0., .25, .5, .75, 1.):
                pp = pooled_probabilities(p, cal['own'], w)
                loss = -(cal['y'] * np.log(pp)).sum(1)
                candidates.append(dict(weight=w, nll=float(np.average(loss, weights=cal['weight']))))
            lowest = min(c['nll'] for c in candidates)
            weight = max(c['weight'] for c in candidates if c['nll'] <= lowest+1e-10)
            summary['pool'] = dict(weight=weight, candidates=candidates)
        summary['heads'][kind] = info
        dump(destination / 'selection.json', summary)
    summary['seconds'] = time.monotonic()-start
    summary['finished_utc'] = datetime.now(timezone.utc).isoformat()
    dump(destination / 'selection.json', summary)
    print(seed, 'fitted and calibrated', round(summary['seconds']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('seed', choices=SEEDS)
    train(parser.parse_args().seed)
