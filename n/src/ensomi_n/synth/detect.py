"""A simple onset detector for isolated layers, used by the tests.

The renderer leaves at least ``EDGE_MS`` of exact silence before every onset
of a monophonic layer (`render`), and every voice but the pad starts near
its full level. So on such a layer rendered alone, without the room, an
onset is the first sample above ``rel_threshold`` of the layer's peak after
``silence_ms`` below it, provided it or the next sample reaches ``onset_rel``
of the peak. That finds the first
sample at or after each onset: within one sample of the true time. The level
condition keeps a decaying noise tail that dips under the silence threshold
from counting.
"""
from __future__ import annotations

import numpy as np


def onsets(x: np.ndarray, sample_rate: int, silence_ms: float = 0.8, rel_threshold: float = 1e-4,
           onset_rel: float = 0.02) -> np.ndarray:
    """Onset times (ms) of an isolated layer."""
    a = np.abs(np.asarray(x, dtype=np.float64))
    if not len(a) or a.max() <= 0:
        return np.zeros(0)
    peak = a.max()
    quiet = a <= rel_threshold * peak
    m = max(1, int(round(silence_ms * sample_rate / 1000.0)))
    c = np.concatenate([[0], np.cumsum(~quiet)])
    idx = np.arange(len(a))
    loud_before = c[idx] - c[np.maximum(idx - m, 0)]
    # An onset read between samples (an edit reading at a fractional position) starts with one partial
    # sample, so the level is checked over the first two samples.
    level = np.maximum(a, np.concatenate([a[1:], [0.0]]))
    first = ~quiet & (level >= onset_rel * peak) & (loud_before == 0) & (idx >= m)
    return np.flatnonzero(first) * 1000.0 / sample_rate
