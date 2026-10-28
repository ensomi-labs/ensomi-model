"""X5d linear fitting and temperature calibration, copied from a00b9b6."""
import numpy as np
import torch
from scipy.special import log_softmax
from scipy.optimize import minimize_scalar

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
