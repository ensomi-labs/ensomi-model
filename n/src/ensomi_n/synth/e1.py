"""E1: scoring the baselines on synthetic suite v0 against its ground truth.

Inputs are files: the ground truth written by ``python -m ensomi_n.synth
render`` (``<truth>/<id>/map.json`` and ``<truth>/manifest.jsonl``) and the
predictions the R-side script ``scripts/n1/r2_on_clips.py`` writes:

- ``<preds>/bt/<ckpt>/<id>.json``: raw BeatThis, ``beats_s`` and
  ``downbeats_s`` on the decoded clock (BeatThis reads the file as decoded);
- ``<preds>/r2/<ckpt>/<id>.json``: R2's red lines on osu!'s clock,
  ``export`` (as ``make`` writes them: whole-ms offsets, canonical beat
  length, meter 4) and ``fitted`` (integer-BPM rounding off, offsets unrounded).

The clock. The ground truth is on the decoded clock and charts, which R2 is
calibrated to, on osu!'s, about 20 ms earlier. One constant ``c`` (decoded to
osu!, ms) converts: the reference is the ground truth plus ``c``; R2's maps
are scored as they are; BeatThis's maps are decoded and get ``c`` through the
harness's own guard (``decoded_to_osu_ms``), so ``c`` cancels for them.
``measure_clock`` gives ``c`` from the control clips once; it is never fitted
per clip.

Arms:

- ``bt-<ckpt>``: BeatThis peaks as a beat list, its downbeats as bars;
- ``r2x-<ckpt>``: R2 as exported;
- ``r2f-<ckpt>``: R2 as fitted, each segment's notated level moved by a power
  of two to the ground truth's at its middle (the design's "level aligned to
  the reference"), meter 4 as R2 has no meter.

Measures per clip come from `ensomi_n.eval.score` on the clip's stratum spans
(the whole pulse for most settings), plus:

- ``level_fit``: the level term with the predicted beat length of a beat
  list read as the mean of its intervals within 25 % of their median, not
  the harness's median interval, which reads a 20 ms frame grid as a wrong
  tempo (the median of 24- and 25-frame intervals is 480 ms at 123 BPM);
- ``no_pulse_rate``: predicted beats per second inside spans without pulse.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..eval import score
from ..mapformat import MusicMap, load
from ..timing import Reference, Segment

MEASURES = ('bl20', 'bl10', 'bl5', 'offset_ms', 'level', 'level_fit', 'bars_start', 'bars_same', 'bars_precision',
            'reset_recall', 'boundary_recall', 'no_pulse_rate', 'chance20', 'n_scored')
LEVEL_TOL = 0.005


def load_truth(truth_dir) -> tuple[dict[str, MusicMap], list[dict]]:
    truth_dir = Path(truth_dir)
    rows = [json.loads(x) for x in (truth_dir / 'manifest.jsonl').read_text().splitlines() if x.strip()]
    maps = {r['id']: load(truth_dir / r['id'] / 'map.json') for r in rows}
    return maps, rows


def _intersect(a, b):
    out = []
    for x0, x1 in a:
        for y0, y1 in b:
            lo, hi = max(x0, y0), min(x1, y1)
            if hi > lo:
                out.append((lo, hi))
    return out


def reference(gt: MusicMap, c_ms: float, key: str = '') -> tuple[Reference, list]:
    """The ground truth on osu!'s clock (plus ``c_ms``), every beat weighing 1, and its stratum spans."""
    truth = gt.extra['truth']
    beats = gt.beats + c_ms
    segs = None
    if gt.segments:
        segs = [Segment(s.start_ms + c_ms, s.beat_ms, s.meter, s.bar_at_start) for s in gt.segments]
    ref = Reference.from_beats(beats, None, downbeats=gt.downbeats, segments=segs, lo_ms=float(beats[0]),
                               hi_ms=float(beats[-1]), key=key)
    pulse = [(a + c_ms, b + c_ms) for a, b in truth['pulse_spans']]
    spans = _intersect([(a + c_ms, b + c_ms) for a, b in truth['spans']], pulse)
    return ref, spans


def bt_map(pred: dict) -> MusicMap:
    beats = np.asarray(pred['beats_s'], dtype=np.float64) * 1000.0
    beats = np.unique(beats)
    downs = np.asarray(pred.get('downbeats_s', []), dtype=np.float64) * 1000.0
    idx = np.zeros(0, dtype=np.int64)
    if len(beats) and len(downs):
        # Each downbeat to its nearest beat (BeatThis's own postprocessing puts them on beats).
        j = np.searchsorted(beats, downs)
        lo = np.clip(j - 1, 0, len(beats) - 1)
        hi = np.clip(j, 0, len(beats) - 1)
        idx = np.unique(np.where(np.abs(beats[hi] - downs) < np.abs(beats[lo] - downs), hi, lo))
    return MusicMap.from_beats(beats, idx if len(beats) else None, clock='decoded', model='beatthis')


def r2_map(lines, gt: MusicMap | None = None, c_ms: float = 0.0, align: bool = False) -> MusicMap:
    """R2's red lines as a map on osu!'s clock; ``align`` moves each line's level to the ground truth's."""
    segs = []
    for off, length, meter in lines:
        length = float(length)
        if align and gt is not None:
            length = _aligned(length, float(off), lines, gt, c_ms)
        segs.append(Segment(float(off), length, int(meter)))
    uniq = {}
    for s in segs:
        uniq[s.start_ms] = s
    return MusicMap(segments=tuple(uniq.values()), model={'name': 'r2'})


def _aligned(length: float, off: float, lines, gt: MusicMap, c_ms: float) -> float:
    starts = [float(x[0]) for x in lines]
    k = starts.index(off)
    end = starts[k + 1] if k + 1 < len(starts) else float(gt.beats[-1]) + c_ms
    a, b = max(off, float(gt.beats[0]) + c_ms), min(end, float(gt.beats[-1]) + c_ms)
    mid = (a + b) / 2.0 if b > a else off
    t = gt.beats + c_ms
    i = int(np.clip(np.searchsorted(t, mid) - 1, 0, len(t) - 2))
    step = np.asarray(gt.extra['truth']['beat_info']['step_ms'])[i]
    return length * 2.0 ** round(math.log2(step / length))


def _beat_length_fit(beats: np.ndarray, a: float, b: float) -> float:
    inside = beats[(beats >= a - 2.0) & (beats < b - 2.0)]
    if len(inside) < 2:
        # A run of one beat reads the predicted interval holding its middle, as the harness does.
        k = int(np.searchsorted(beats, (a + b) / 2.0, side='right')) - 1
        return float(beats[k + 1] - beats[k]) if 0 <= k < len(beats) - 1 else math.nan
    d = np.diff(inside)
    m = np.median(d)
    keep = d[(d >= 0.75 * m) & (d <= 1.25 * m)]
    return float(keep.mean()) if len(keep) else math.nan


def level_fit(ref: Reference, pred_beats: np.ndarray, spans) -> float:
    """Duration-weighted share of reference tempo runs whose predicted beat length (mean of the intervals within
    25 % of their median) is within 0.5 % of the notated one."""
    from ..eval.measures import _ref_segments
    ok = tot = 0.0
    for a, b, step in _ref_segments(ref):
        a, b = max(a, ref.lo_ms), min(b, ref.hi_ms)
        if b <= a:
            continue
        dur = sum(max(0.0, min(b, y) - max(a, x)) for x, y in spans) if spans else b - a
        if dur <= 0:
            continue
        p = _beat_length_fit(pred_beats, a, b)
        tot += dur
        ok += dur * (math.isfinite(p) and abs(step / p - 1.0) <= LEVEL_TOL)
    return ok / tot if tot > 0 else math.nan


def no_pulse_rate(gt: MusicMap, pred: MusicMap, c_ms: float) -> float:
    spans = [(a + c_ms, b + c_ms) for a, b, _ in gt.extra['truth']['no_pulse']]
    total = sum(b - a for a, b in spans)
    if total <= 0:
        return math.nan
    if pred.clock == 'decoded':
        pred = pred.on_osu_clock(c_ms)
    beats = pred.beats if pred.beats is not None else pred.timing(0.0, gt.beats[-1] + c_ms + 5000.0).beats
    n = sum(int(((beats >= a) & (beats < b)).sum()) for a, b in spans)
    return n / (total / 1000.0)


def score_clip(gt: MusicMap, pred: MusicMap, c_ms: float, key: str = '') -> dict:
    ref, spans = reference(gt, c_ms, key)
    s = score(ref, pred, spans=spans, decoded_to_osu_ms=c_ms)
    out = dict(bl20=s.bl20, bl10=s.bl10, bl5=s.bl5, offset_ms=s.offset_ms, level=s.level, bars_start=s.bars_start,
               bars_same=s.bars_same, bars_precision=s.bars_precision, reset_recall=s.reset_recall,
               boundary_recall=s.boundary_recall, chance20=s.chance[20.0], n_scored=s.n_scored)
    if pred.beats is not None:
        p = pred.on_osu_clock(c_ms) if pred.clock == 'decoded' else pred
        out['level_fit'] = level_fit(ref, p.beats, spans)
    else:
        out['level_fit'] = s.level
    out['no_pulse_rate'] = no_pulse_rate(gt, pred, c_ms)
    return out


def load_preds(preds_dir, ids) -> dict[str, dict[str, MusicMap | list]]:
    """``{arm: {id: raw prediction}}`` for every arm found under ``preds_dir``."""
    preds_dir = Path(preds_dir)
    arms: dict[str, dict] = {}
    for kind in ('bt', 'r2'):
        for d in sorted((preds_dir / kind).glob('*')) if (preds_dir / kind).is_dir() else []:
            for cid in ids:
                f = d / f'{cid}.json'
                if not f.is_file():
                    continue
                p = json.loads(f.read_text())
                if kind == 'bt':
                    arms.setdefault(f'bt-{d.name}', {})[cid] = p
                else:
                    arms.setdefault(f'r2x-{d.name}', {})[cid] = p
                    arms.setdefault(f'r2f-{d.name}', {})[cid] = p
    return arms


def arm_map(arm: str, raw: dict, gt: MusicMap, c_ms: float) -> MusicMap:
    if arm.startswith('bt-'):
        return bt_map(raw)
    if arm.startswith('r2x-'):
        return r2_map(raw['export'])
    return r2_map(raw['fitted'], gt, c_ms, align=True)


def score_all(maps: dict[str, MusicMap], arms: dict, c_ms: float) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for arm, preds in arms.items():
        for cid, gt in maps.items():
            if cid not in preds:
                continue
            out.setdefault(arm, {})[cid] = score_clip(gt, arm_map(arm, preds[cid], gt, c_ms), c_ms, cid)
    return out


def measure_clock(maps: dict[str, MusicMap], arms: dict, control_ids, arm: str = 'r2f-final0') -> dict:
    """The decoded-to-osu constant: the median, over the control clips, of the arm's offset term (its
    head-weighted median signed error, prediction minus truth) with the truth on the decoded clock."""
    per = {}
    for cid in control_ids:
        if cid in arms.get(arm, {}):
            gt = maps[cid]
            s = score_clip(gt, arm_map(arm, arms[arm][cid], gt, 0.0), 0.0, cid)
            per[cid] = s['offset_ms']
    vals = np.array([v for v in per.values() if math.isfinite(v)])
    return dict(arm=arm, per_clip=per, c_ms=float(np.round(np.median(vals), 1)) if len(vals) else math.nan,
                spread_ms=float(vals.max() - vals.min()) if len(vals) else math.nan)


def _boot_mean(x: np.ndarray, rng, n: int = 2000) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if not len(x):
        return math.nan, math.nan
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def _boot_diff(x: np.ndarray, y: np.ndarray, rng, n: int = 2000) -> tuple[float, float]:
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if not len(x) or not len(y):
        return math.nan, math.nan
    mx = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    my = y[rng.integers(0, len(y), size=(n, len(y)))].mean(axis=1)
    d = mx - my
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def table(scores: dict, rows: list[dict], measure: str, seed: int = 0) -> dict:
    """Per arm and stratum: mean over clips, 95 % clip bootstrap, per-seed means and the seed spread."""
    rng = np.random.default_rng(seed)
    strata = sorted({r['stratum'] for r in rows})
    out = {}
    for arm, per in scores.items():
        for st in strata:
            clips = [r for r in rows if r['stratum'] == st and r['id'] in per]
            x = np.array([per[r['id']][measure] for r in clips], dtype=np.float64)
            fin = np.isfinite(x)
            by_seed = {}
            for sd in sorted({r['seed'] for r in clips}):
                v = np.array([per[r['id']][measure] for r in clips if r['seed'] == sd], dtype=np.float64)
                by_seed[sd] = float(np.nanmean(v)) if np.isfinite(v).any() else math.nan
            lo, hi = _boot_mean(x, rng)
            seeds = [v for v in by_seed.values() if math.isfinite(v)]
            out.setdefault(arm, {})[st] = dict(value=float(np.nanmean(x)) if fin.any() else math.nan, lo=lo, hi=hi,
                                               n=int(fin.sum()), by_seed=by_seed,
                                               seed_spread=float(max(seeds) - min(seeds)) if len(seeds) > 1
                                               else math.nan)
    return out


def rule(scores: dict, rows: list[dict], measure: str, family: str, ckpts=('final0', 'final1', 'final2'),
         main: str | None = None, control: str = 'S0', seed: int = 0) -> dict:
    """E1's "fails by construction" rule for one arm family (``bt``, ``r2x``, ``r2f``) and one measure.

    For each stratum s: Δ_s = mean(control) − mean(s) over clips. The family's Δ_s is that of ``main`` (one
    checkpoint) or, with ``main`` None, the mean over the checkpoints. Its interval resamples clips of both
    groups (two-sample bootstrap; for the checkpoint mean, clip means over checkpoints are resampled). The
    margin is the larger of the checkpoint spread (range of Δ_s over checkpoints) and the seed spread
    (|Δ_s on seed 0 clips − Δ_s on seed 1 clips|). Fails when Δ_s > margin and the interval's lower end > 0.
    """
    rng = np.random.default_rng(seed)
    arms = [f'{family}-{c}' for c in ckpts if f'{family}-{c}' in scores]
    if not arms:
        return {}
    ids = set.intersection(*(set(scores[a]) for a in arms))

    def clip_vals(st, sd=None, arm_list=None):
        al = arm_list or ([f'{family}-{main}'] if main else arms)
        cl = [r['id'] for r in rows if r['stratum'] == st and r['id'] in ids and (sd is None or r['seed'] == sd)]
        return np.array([np.nanmean([scores[a][c][measure] for a in al]) for c in cl], dtype=np.float64)

    out = {}
    for st in sorted({r['stratum'] for r in rows}):
        if st == control:
            continue
        x, y = clip_vals(control), clip_vals(st)
        if not np.isfinite(y).any() or not np.isfinite(x).any():
            continue
        delta = float(np.nanmean(x) - np.nanmean(y))
        lo, hi = _boot_diff(x, y, rng)
        per_ck = [float(np.nanmean(clip_vals(control, arm_list=[a])) - np.nanmean(clip_vals(st, arm_list=[a])))
                  for a in arms]
        ck_spread = max(per_ck) - min(per_ck) if len(per_ck) > 1 else math.nan
        per_seed = []
        for sd in (0, 1):
            xs, ys = clip_vals(control, sd), clip_vals(st, sd)
            if np.isfinite(xs).any() and np.isfinite(ys).any():
                per_seed.append(float(np.nanmean(xs) - np.nanmean(ys)))
        seed_spread = abs(per_seed[0] - per_seed[1]) if len(per_seed) == 2 else math.nan
        margin = float(np.nanmax([ck_spread, seed_spread])) if np.isfinite([ck_spread, seed_spread]).any() \
            else math.nan
        fails = bool(math.isfinite(margin) and delta > margin and lo > 0)
        out[st] = dict(delta=delta, lo=lo, hi=hi, checkpoint_deltas=dict(zip(arms, per_ck)),
                       checkpoint_spread=ck_spread, seed_deltas=per_seed, seed_spread=seed_spread, margin=margin,
                       fails=fails, n_control=int(np.isfinite(x).sum()), n_stratum=int(np.isfinite(y).sum()))
    return out
