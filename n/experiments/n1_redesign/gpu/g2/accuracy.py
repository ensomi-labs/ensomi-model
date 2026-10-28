"""X5d frame/regime band confusion counts, copied from a00b9b6."""
import numpy as np
from .. import base
from .x5d_targets import targets,band_of,BANDS

def confusion(truth, pred):
    return np.bincount(7*np.asarray(truth, int)+np.asarray(pred, int), minlength=49).reshape(7,7)

def accuracy(ref, times, probabilities, strata):
    y, valid = targets([ref], times)
    truth, pred = y.argmax(1), probabilities.argmax(1)
    logp = np.log(probabilities+1e-4)
    result = {}
    for cell, spans in [('all', None)]+[(k,v) for k,v in strata.items() if k in ('S0','S5')]:
        mask = valid & base.inside(times, ref, spans)
        if not mask.any():
            continue
        frame = confusion(truth[mask], pred[mask])
        rt, rp, omitted = [], [], 0
        for i, seg in enumerate(ref.segments):
            end = ref.segments[i+1].start_ms if i+1 < len(ref.segments) else float('inf')
            in_regime = mask & (times >= seg.start_ms) & (times < end)
            if not in_regime.any():
                omitted += 1
                continue
            rt.append(int(band_of(seg.beat_ms))-BANDS[0])
            rp.append(int(logp[in_regime].mean(0).argmax()))
        result[cell] = dict(frame=frame, regime=confusion(rt,rp), omitted_regimes=omitted)
    return result
