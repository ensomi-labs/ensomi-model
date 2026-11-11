"""Inference-only ensembles with explicit waveform and frame coordinates."""
from __future__ import annotations

import numpy as np


def shifted_wave(wave, samples):
    """Delay without wrapping or changing length; samples is a nonnegative integer."""
    if int(samples) != samples or samples < 0:
        raise ValueError('delay must be a nonnegative integer sample count')
    y = np.zeros_like(wave)
    if samples == 0:
        y[:] = wave
    elif samples < len(y):
        y[samples:] = wave[:-samples]
    return y


def crop_starts(n, origin=0, crop=2000, hop=1000):
    """Origin-shifted crop schedule with unchanged first/last coverage."""
    if n <= crop:
        return [0]
    origin %= hop
    return sorted({0, n - crop} | set(range(origin, n - crop, hop)))


def interpolate(a, query):
    a = np.asarray(a)
    flat = a.reshape(len(a), -1)
    x = np.arange(len(a))
    return np.stack([np.interp(query, x, flat[:, k]) for k in range(flat.shape[1])], axis=-1).reshape(
        (len(query),) + a.shape[1:])


def align_evidence(acts, delay_ms):
    """Undo a known PCM delay on a 10-ms grid, interpolating event positions and circular phases.

    Output frame j is original time 10*j. Source frame i is original time
    10*i-delay_ms. Offset heads point to events in that original clock.
    Endpoint extrapolation is constant; evaluate away from padded edges.
    """
    n = len(acts['beat'])
    destination = 10. * np.arange(n)
    query = (destination + delay_ms) / 10.
    source = 10. * np.arange(n) - delay_ms
    result = {}
    for k, a in acts.items():
        if k in ('phase', 'bar_phase'):
            angle = 2 * np.pi * a
            c, s = interpolate(np.cos(angle), query), interpolate(np.sin(angle), query)
            value = np.remainder(np.arctan2(s, c) / (2 * np.pi), 1.)
        elif k.endswith('_off'):
            # Interpolate positions in the common clock, not the shifted frame labels.
            value = interpolate(source + a, query) - destination
        else:
            value = interpolate(a, query)
        result[k] = np.asarray(value, dtype=np.float32)
    return result


def mean_evidence(arms):
    if not arms:
        raise ValueError('an ensemble needs at least one member')
    result = {}
    for k in arms[0]:
        a = np.stack([r[k] for r in arms]).astype(np.float64)
        if k in ('phase', 'bar_phase'):
            c, s = np.cos(2 * np.pi * a).mean(0), np.sin(2 * np.pi * a).mean(0)
            value = np.remainder(np.arctan2(s, c) / (2 * np.pi), 1.)
        else:
            value = a.mean(0)
        if k == 'level':
            value /= value.sum(-1, keepdims=True)
        result[k] = value.astype(np.float32)
    return result


def song_evidence(model, mel, memo, mean, std, device='mps', origin=0, batch=8):
    """The v0 inference window rule on in-memory arrays; weights and source files stay unchanged."""
    import torch
    from ensomi_n.v0.data import CROP, MAX_MEMO_S
    from ensomi_n.v0.model import V0
    n = len(mel)
    total = n + (-n) % 4
    x = np.zeros((total, mel.shape[1]), np.float32)
    x[:n] = (np.asarray(mel, np.float32) - mean) / std
    x[n:] = x[:n].min() if n else 0.
    starts = crop_starts(total, origin)
    with torch.inference_mode():
        mem = None
        if model.encoder.memory is not None:
            m = (np.asarray(memo, np.float32) - mean) / std
            cap = int(MAX_MEMO_S * 1000 / 40)
            if len(m) > cap:
                center = len(m) // 2
                begin = int(np.clip(center - cap // 2, 0, len(m) - cap))
                m = m[begin:begin + cap]
            mt = torch.from_numpy(m[None]).to(device)
            mem = model.encoder.song_memory(mt, torch.ones(mt.shape[:2], dtype=torch.bool, device=device))
        output = None
        best = np.full(total, -1.)
        for i in range(0, len(starts), batch):
            chunk = starts[i:i + batch]
            xb = torch.from_numpy(np.stack([x[s:s + min(CROP, total)] for s in chunk])).to(device)
            mb = None if mem is None else mem.expand(len(chunk), -1, -1)
            heads = model.heads(model.encoder(xb, mb, use_memory=mem is not None))
            acts = {k:v.float().cpu().numpy() for k,v in V0.activations(heads).items()}
            if output is None:
                output = {k:np.zeros((total,) + v.shape[2:], np.float32) for k,v in acts.items()}
            length = xb.shape[1]
            centre = np.minimum(np.arange(length), length - 1 - np.arange(length)).astype(float)
            for j, s in enumerate(chunk):
                take = centre > best[s:s + length]
                idx = np.flatnonzero(take) + s
                for k,v in acts.items():
                    output[k][idx] = v[j][take]
                best[idx] = centre[take]
    return {k:v[:n] for k,v in output.items()}
