"""Level ceilings, held-out readouts and descriptive evidence probes for SMC.

Annotation-assisted remapping is a diagnostic ceiling. Its selected sequence
is shared across metrics; it is never presented as a deployable readout.
"""
from __future__ import annotations

import numpy as np

from .smc import events

LEVEL_ACTIONS = ('unchanged', 'half-0', 'half-1', 'third-0', 'third-1',
                 'third-2', 'quarter-0', 'quarter-1', 'quarter-2', 'quarter-3', 'double')


def level_remappings(prediction) -> list[np.ndarray]:
    """Eleven fixed sequences, decimated before trimming; no end extrapolation."""
    p = events(prediction)
    return [p] + [p[phase::q] for q in (2, 3, 4) for phase in range(q)] + [
        np.sort(np.concatenate([p, (p[:-1] + p[1:]) / 2]))]


def timing_metrics(reference, prediction) -> dict:
    """F at four tolerances and signed error of maximum-cardinality 70ms matches.

    Both sequences are trimmed below 5s. Error is prediction minus reference;
    None means no matched beats. Continuity uses mir_eval's default tolerances.
    """
    from mir_eval.beat import continuity, f_measure
    from mir_eval.util import match_events
    r, p = events(reference), events(prediction)
    r, p = r[r >= 5], p[p >= 5]
    out = {f'F{ms}': float(f_measure(r, p, f_measure_threshold=ms / 1000))
           for ms in (20, 35, 50, 70)}
    out.update(zip(('CMLc', 'CMLt', 'AMLc', 'AMLt'), map(float, continuity(r, p))))
    matches = match_events(r, p, window=.07)
    errors = [1000 * (p[j] - r[i]) for i, j in matches]
    out['signed_ms'] = float(np.median(errors)) if errors else None
    out['n_matches'] = len(matches)
    return out


def select_folds(utilities, folds) -> np.ndarray:
    """Choose one candidate per held-out fold by other folds' mean utility.

    Rows are excerpts, columns are candidates. Column order resolves ties.
    """
    y, folds = np.asarray(utilities, dtype=float), np.asarray(folds)
    if y.ndim != 2 or len(y) != len(folds) or not np.isfinite(y).all():
        raise ValueError('finite excerpt-by-candidate utilities required')
    if len(np.unique(folds)) < 2:
        raise ValueError('at least two folds required')
    chosen = np.zeros(len(y), dtype=int)
    for f in np.unique(folds):
        chosen[folds == f] = y[folds != f].mean(axis=0).argmax()
    return chosen


def ridge_readout(features, utilities, folds, *, penalty: float = 10.) -> tuple[np.ndarray, list]:
    """Fit candidate utilities on other folds; standardize using training rows only.

    The penalty is fixed externally, not selected using held-out labels. Returned
    diagnostics expose training keys through indices and each fold's coefficients.
    """
    x, y, folds = np.asarray(features, float), np.asarray(utilities, float), np.asarray(folds)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(y) != len(folds):
        raise ValueError('features, utilities and folds must align')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or penalty <= 0:
        raise ValueError('finite data and positive penalty required')
    if len(np.unique(folds)) < 2:
        raise ValueError('at least two folds required')
    chosen, fits = np.zeros(len(y), dtype=int), []
    for f in np.unique(folds):
        train, test = folds != f, folds == f
        mean, scale = x[train].mean(axis=0), x[train].std(axis=0)
        scale[scale < 1e-8] = 1
        z = np.column_stack([np.ones(len(x)), (x - mean) / scale])
        regularizer = np.eye(z.shape[1]) * penalty
        regularizer[0, 0] = 0
        coef = np.linalg.solve(z[train].T @ z[train] + regularizer, z[train].T @ y[train])
        chosen[test] = (z[test] @ coef).argmax(axis=1)
        fits.append(dict(fold=int(f), train_indices=np.flatnonzero(train).tolist(),
                         mean=mean.tolist(), scale=scale.tolist(), coefficients=coef.tolist()))
    return chosen, fits


def own_features(prediction, activation) -> np.ndarray:
    """Fixed readout features using only predicted beats and N1 output heads."""
    p = events(prediction)
    p = p[(p >= 5) & (p < 40)]
    ibi = np.diff(p)
    bpm = 60 / np.median(ibi) if len(ibi) else 120.
    cv = np.std(ibi) / np.mean(ibi) if len(ibi) else 0.
    z = activation
    features = [np.log(bpm), cv, *z['level'][500:4000].mean(axis=0),
                np.median(z['logp'][500:4000])]
    names = ('beat', 'down', 'half')
    features.extend(float(z[k][500:4000].mean()) for k in names)
    for candidate in level_remappings(prediction):
        times = candidate[(candidate >= 5) & (candidate < 40)]
        for k in names:
            features.append(float(np.interp(times * 100, np.arange(len(z[k])), z[k]).mean())
                            if len(times) else 0.)
    return np.asarray(features, float)


def evidence_probes(probability, reference, *, fps=100) -> dict:
    """Reference-centered evidence diagnostics; probabilities are not calibrated.

    Sharpness and beat-minus-midpoint advantage are scaled by the excerpt's
    p95-p5 range. These annotation-assisted probes are descriptive, not scores.
    """
    x, r = np.asarray(probability, float), events(reference)
    if fps != 100:
        # Normalize and probe both trackers on the same 100fps clock. BT
        # includes a frame at 40s, while N1 stops one frame before it.
        grid100 = np.arange(min(4000, int((len(x) - 1) * 100 / fps) + 1)) / 100
        x = np.interp(grid100, np.arange(len(x)) / fps, x)
        fps = 100
    r = r[(r >= 5) & (r < 40)]
    grid = np.arange(len(x)) / fps
    scale = np.quantile(x[int(5 * fps):int(40 * fps)], .95) - np.quantile(
        x[int(5 * fps):int(40 * fps)], .05) + 1e-6
    offsets = np.arange(-5, 6) / 100

    def peaks(times):
        samples = np.interp((times[:, None] + offsets).ravel(), grid, x).reshape(-1, 11)
        indices = samples.argmax(axis=1)
        return samples[np.arange(len(times)), indices], times + offsets[indices]

    if not len(r):
        raise ValueError('reference must contain beats in [5,40)')
    peak, when = peaks(r)
    flank = (np.interp(when - .1, grid, x) + np.interp(when + .1, grid, x)) / 2
    mid, _ = peaks((r[:-1] + r[1:]) / 2)
    return dict(at_beat=float(np.interp(r, grid, x).mean()), peak50=float(peak.mean()),
                sharpness=float(((peak - flank) / scale).mean()),
                advantage=float((peak[:-1] - mid).mean() / scale) if len(mid) else 0.)


def bootstrap_ols(x, y, *, samples=2000, seed=20261111) -> dict:
    """Joint exploratory associations with paired excerpt percentile intervals.

    Caller supplies the intercept. Singular resamples are counted and omitted;
    callers must not interpret unstable or confounded coefficients as effects.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('finite aligned design and outcome matrices required')
    rank = np.linalg.matrix_rank(x)
    if rank != x.shape[1]:
        raise ValueError('design is rank deficient')
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    rng, draws, singular = np.random.default_rng(seed), [], 0
    for _ in range(samples):
        ix = rng.integers(len(x), size=len(x))
        if np.linalg.matrix_rank(x[ix]) != rank:
            singular += 1
            continue
        draws.append(np.linalg.lstsq(x[ix], y[ix], rcond=None)[0])
    lo, hi = np.quantile(draws, [.025, .975], axis=0)
    vif = [1.]
    for j in range(1, x.shape[1]):
        residual = x[:, j] - np.delete(x, j, axis=1) @ np.linalg.lstsq(
            np.delete(x, j, axis=1), x[:, j], rcond=None)[0]
        vif.append(float(np.sum((x[:, j] - x[:, j].mean()) ** 2) / np.sum(residual ** 2)))
    return dict(coef=coef.tolist(), lo=lo.tolist(), hi=hi.tolist(), vif=vif,
                rank=int(rank), condition=float(np.linalg.cond(x)), singular=singular,
                samples=samples, n=len(x), r2=(1 - ((y - x @ coef) ** 2).sum(axis=0) /
                ((y - y.mean(axis=0)) ** 2).sum(axis=0)).tolist())
