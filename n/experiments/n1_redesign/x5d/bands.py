"""Direct absolute-band posteriors through the unchanged X5 D2 projection."""
from types import SimpleNamespace
from common import *
from scipy.special import ndtr, softmax
import level
from decoder_measurements import CFG, dec


def own_probabilities(ev):
    logbpm = np.log(60.) - ev.logp
    fold = np.floor((logbpm - np.log(80.)) / np.log(2.))
    canonical = logbpm - fold * np.log(2.)
    sigma = np.maximum(ev.logp_sigma if ev.logp_sigma is not None else .05, .001)
    probs = np.zeros((len(logbpm), len(BANDS)))
    lev = ev.level / np.maximum(ev.level.sum(1, keepdims=True), 1e-12)
    for k, mult in enumerate(dec.LEVEL_CLASSES):
        mu = canonical + np.log(mult)
        low = np.log(80.) + BANDS * np.log(2.)
        mass = ndtr((low[None, :] + np.log(2.) - mu[:, None]) / sigma[:, None])
        mass -= ndtr((low[None, :] - mu[:, None]) / sigma[:, None])
        probs += lev[:, k, None] * mass
    return probs


def own_from_logits(raw):
    raw = np.asarray(raw, np.float64)
    return own_probabilities(SimpleNamespace(level=softmax(raw[:, 7:11], axis=1),
        logp=raw[:, 18], logp_sigma=np.logaddexp(0., raw[:, 19]) + .001))


def pooled_probabilities(band, own, weight):
    return softmax(weight*np.log(band + 1e-4) + (1-weight)*np.log(own + 1e-4), axis=1)


def ratios_from_probabilities(probs, groups, periods, cfg):
    means, durs = [], []
    for f0, f1, _ in groups:
        a = int(np.clip(f0, 0, len(probs)-1))
        b = int(np.clip(f1, a+1, len(probs)))
        means.append(np.log(probs[a:b] + cfg.level_eps).mean(0))
        durs.append(b-a)
    means = np.asarray(means)
    song = int(np.argmax((means * np.array(durs)[:, None]).sum(0)))
    ratios = []
    for mean, period, (_, _, free) in zip(means, periods, groups):
        scores = mean.copy()
        scores[song] += cfg.level_prior
        choices = []
        for k in range(0 if free else -2, 4):
            band = int(band_of(period / 2.**k))
            where = np.flatnonzero(BANDS == band)
            if len(where):
                choices.append((float(scores[where[0]]), -abs(k), k))
        ratios.append(2.**max(choices)[2])
    return ratios


def decode_band(ev, probabilities, track, cfg=CFG):
    if probabilities.shape != (ev.n_frames, len(BANDS)):
        raise ValueError('Band posterior and evidence clocks differ')
    original = level.band_ratios
    level.band_ratios = lambda e, groups, periods, c: ratios_from_probabilities(probabilities, groups, periods, c)
    try:
        return level.decode_level(ev, None, 'own_d2', track, cfg)
    finally:
        level.band_ratios = original
