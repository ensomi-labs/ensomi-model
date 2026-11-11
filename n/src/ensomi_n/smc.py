"""SMC benchmark utilities; times are seconds on the audio clock.

The scorer uses mir_eval defaults, including trimming beats before 5 s, and
averages excerpts equally. Optional mir_eval/scipy dependencies are imported
only where used. Annotation-derived diagnostics characterise errors; they do
not validate musical quality or select a per-excerpt correction.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

import numpy as np

SHIFTS_MS = np.arange(-80, 81, 5)
METRICS = ('F-measure', 'CMLc', 'CMLt', 'AMLc', 'AMLt', 'Cemgil', 'Goto', 'P-score', 'Information gain')


def extract_data(archive: Path, destination: Path, *, max_bytes: int = 1_000_000_000) -> dict:
    """Extract regular files into a new directory after validating the whole ZIP.

    Reject traversal, links, special files, duplicate paths and oversized output.
    No archive code is executed and no permissions from the archive are applied.
    The destination must not exist; a failure can leave partial extracted data.
    """
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        if len(members) > 20000 or sum(i.file_size for i in members) > max_bytes:
            raise ValueError('archive exceeds extraction budget')
        seen = set()
        for i in members:
            p = PurePosixPath(i.filename)
            mode = i.external_attr >> 16
            if (p.is_absolute() or '..' in p.parts or '\\' in i.filename
                    or not p.parts or ':' in p.parts[0]
                    or stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)
                    or p in seen):
                raise ValueError(f'unsafe archive member: {i.filename!r}')
            seen.add(p)
        destination.mkdir(parents=True, exist_ok=False)
        for i in members:
            target = destination.joinpath(*PurePosixPath(i.filename).parts)
            if i.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(i) as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
        return dict(members=len(members), bytes=sum(i.file_size for i in members))


def events(values) -> np.ndarray:
    """Validate finite, nonnegative, strictly increasing seconds; never repair input."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all() or (x < 0).any() or (np.diff(x) <= 0).any():
        raise ValueError('beats must be finite, nonnegative, strictly increasing seconds')
    return x


def score(reference, prediction, *, shift_ms: float = 0.0) -> dict:
    """mir_eval beat scores after adding shift_ms to predictions, then trimming at 5 s.

    Predictions are validated before shifting. The shift may move beats before
    zero; these are removed by the common 5 s trim, not clamped to zero.
    """
    import mir_eval.beat
    ref = events(reference)
    pred = events(prediction) + shift_ms / 1000.0
    result = mir_eval.beat.evaluate(ref[ref >= 5.0], pred[pred >= 5.0])
    names = dict(CMLc='Correct Metric Level Continuous', CMLt='Correct Metric Level Total',
                 AMLc='Any Metric Level Continuous', AMLt='Any Metric Level Total')
    return {k: float(result[names.get(k, k)]) for k in METRICS}


def shift_curve(reference, prediction, shifts_ms=SHIFTS_MS) -> np.ndarray:
    """F at each global candidate shift; descriptive, not a fitted headline score."""
    from mir_eval.beat import f_measure
    ref, pred = events(reference), events(prediction)
    ref = ref[ref >= 5.0]
    return np.array([f_measure(ref, (pred + s / 1000.0)[pred + s / 1000.0 >= 5.0])
                     for s in shifts_ms])


def heldout_shifts(curves, folds, shifts_ms=SHIFTS_MS) -> np.ndarray:
    """One shift per test fold, selected on all other folds' excerpt-mean F.

    Ties prefer the smallest absolute shift, then the negative shift. No held-out
    excerpt contributes to its own shift. At least two populated folds are required.
    """
    curves = np.asarray(curves, dtype=np.float64)
    folds = np.asarray(folds)
    shifts = np.asarray(shifts_ms)
    if curves.shape != (len(folds), len(shifts)) or not np.isfinite(curves).all():
        raise ValueError('invalid curves or fold alignment')
    if len(np.unique(folds)) < 2:
        raise ValueError('at least two populated folds are required')
    order = sorted(range(len(shifts)), key=lambda i: (abs(shifts[i]), shifts[i]))
    result = np.empty(len(folds), dtype=float)
    for fold in np.unique(folds):
        train = curves[folds != fold].mean(axis=0)
        best = max(order, key=lambda i: train[i])
        result[folds == fold] = shifts[best]
    return result


def paired_bootstrap(left, right, *, samples: int = 10000, seed: int = 20261111) -> dict:
    """Mean paired excerpt difference and percentile 95% interval (left minus right).

    Arrays must already refer to the same excerpts in the same order. This
    interval conditions on the given checkpoints; it is not seed uncertainty.
    """
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if samples < 1 or a.ndim != 1 or a.shape != b.shape or not len(a) or not np.isfinite(a - b).all():
        raise ValueError('paired finite nonempty vectors are required')
    delta = a - b
    rng = np.random.default_rng(seed)
    boot = np.concatenate([delta[rng.integers(len(a), size=(min(500, samples - i), len(a)))].mean(axis=1)
                           for i in range(0, samples, 500)])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return dict(n=len(a), difference=float(delta.mean()), lo=float(lo), hi=float(hi))


def minimal_peaks(probability, *, fps: int = 100) -> np.ndarray:
    """Strict >0.5 peaks in ±60 ms, with adjacent maxima averaged, in seconds.

    This reproduces Beat This 1.1.0's 7-frame max pool at 50 fps. At 100 fps
    the pool is 13 frames to preserve its actual temporal support. No learned
    sub-frame offsets or metrical-level predictions are used.
    """
    from scipy.ndimage import maximum_filter1d
    x = np.asarray(probability, dtype=float)
    if fps <= 0:
        raise ValueError('fps must be positive')
    if x.ndim != 1 or not np.isfinite(x).all() or ((x < 0) | (x > 1)).any():
        raise ValueError('expected a finite probability vector')
    if not len(x):
        return np.array([], dtype=float)
    width = int(round(0.06 * fps))
    peaks = np.flatnonzero((x > 0.5) & (x == maximum_filter1d(x, size=2 * width + 1, mode='constant', cval=-1)))
    # Beat This compares each next index to the running mean, not the last
    # index: a long plateau may produce several groups. Preserve that rule.
    means, count = [], 0
    for peak in peaks:
        if means and peak - means[-1] <= 1:
            count += 1
            means[-1] += (peak - means[-1]) / count
        else:
            means.append(float(peak))
            count = 1
    return np.asarray(means) / fps


def map_beats(music_map, duration: float) -> tuple[np.ndarray, int]:
    """Return nonnegative own-level beats in seconds and out-of-audio count.

    Do not reconstruct from its segments: a fractional boundary or level
    choice can make a reconstructed canonical grid a different prediction.
    Keep predictions at/beyond duration: the standard scorer only trims the
    beginning. Negative beats are excluded before mir_eval input validation.
    """
    if music_map.beats is None:
        raise ValueError('explicit map beats are required')
    times = np.asarray(music_map.beats, dtype=float) / 1000.0
    if times.ndim != 1 or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError('map beats must be finite and strictly increasing')
    outside = (times < 0) | (times >= duration)
    return events(times[times >= 0]), int(outside.sum())


def tempo_diagnostics(reference, prediction) -> dict:
    """Median-IBI tempo ratio and phase descriptors; not ground-truth failure labels."""
    ref, pred = events(reference), events(prediction)
    ref, pred = ref[ref >= 5], pred[pred >= 5]
    if len(ref) < 2:
        raise ValueError('reference needs two beats after trimming')
    ibi = np.diff(ref)
    tempo = float(60 / np.median(ibi))
    cv = float(ibi.std() / ibi.mean())
    result = dict(tempo_bpm=tempo, ibi_cv=cv,
                  tempo_band='below77' if tempo < 77 else 'above167' if tempo > 167 else 'in_band',
                  variability='steady' if cv <= .03 else 'high' if cv >= .10 else 'middle')
    if len(pred) < 2:
        return dict(result, tempo_ratio=None, ratio_class='insufficient', offbeat=False)
    ratio = float(np.median(ibi) / np.median(np.diff(pred)))
    ratios = {'quarter': .25, 'third': 1/3, 'half': .5, '2:3': 2/3,
              'same': 1., '3:2': 1.5, 'double': 2., 'triple': 3., 'quadruple': 4.}
    name = min(ratios, key=lambda k: abs(np.log(ratio / ratios[k])))
    if abs(np.log(ratio / ratios[name])) > np.log(1.1):
        name = 'other'
    # Interpolate the reference beat coordinate, excluding prediction tails.
    inside = pred[(pred >= ref[0]) & (pred <= ref[-1])]
    phase = np.interp(inside, ref, np.arange(len(ref))) % 1
    offbeat = bool(name == 'same' and len(phase) and np.mean(np.abs(phase - .5) <= .15) >= .5)
    return dict(result, tempo_ratio=ratio, ratio_class=name, offbeat=offbeat)


def failure_type(metrics: dict) -> str:
    """Fixed descriptive bins; thresholds are not validated human quality decisions."""
    if metrics['F-measure'] < .2:
        return 'total_failure'
    if metrics['AMLt'] >= .5 and metrics['CMLt'] < .5:
        return 'metrical_level_or_phase'
    if metrics['CMLt'] >= .8:
        return 'continuous'
    return 'continuity_or_partial'
