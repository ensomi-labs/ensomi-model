"""Target definitions copied from X5d a00b9b6; imports adapted only."""
import numpy as np
from .. import base
BANDS=np.arange(-2,5)

def band_of(beat_ms):
    return np.floor(np.log2(60000. / np.asarray(beat_ms) / 80.)).astype(int)

def targets(refs, times):
    """Mean one-hot chart labels; unmapped frames have zero weight."""
    target = np.zeros((len(times), len(BANDS)), np.float32)
    count = np.zeros(len(times), np.float32)
    for ref in refs:
        step = regime_steps(ref, times)
        valid = base.inside(times, ref) & np.isfinite(step) & (step > 0)
        ix = np.flatnonzero(valid)
        band = band_of(step[ix])
        if np.any((band < BANDS[0]) | (band > BANDS[-1])):
            raise ValueError('Target outside D2 band support')
        target[ix, band - BANDS[0]] += 1
        count[ix] += 1
    target /= np.maximum(count[:, None], 1)
    return target, count > 0

def regime_steps(ref, times):
    """Read half-open red-line intervals without the timing matcher's tolerance."""
    if not ref.segments:
        raise ValueError('Band targets require chart red lines')
    starts = np.array([s.start_ms for s in ref.segments])
    which = np.searchsorted(starts, times, side='right') - 1
    step = np.full(len(times), np.nan)
    valid = which >= 0
    step[valid] = np.array([s.beat_ms for s in ref.segments])[which[valid]]
    return step
