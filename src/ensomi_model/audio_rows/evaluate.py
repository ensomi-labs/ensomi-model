"""Development measures of a trained head model. None is validated against the human's judgement;
they characterise, and the human's screen decides.

- ``head_f1``: greedy one-to-one matching of head times within 10, 20 and 40 ms of a reference.
- ``label_shuffle``: teacher-forced count NLL of complete fit_dev charts with their true 16-beat
  section labels, against two shuffles that keep the evaluated label marginal: sections permuted
  among equal-length sections of the chart, and whole label sequences deranged among charts of
  the same audio with identical beat edges. ΔNLL is nats per beat, with a 95 % interval from
  resampling song groups.
- ``dose_response``: at fixed audio, the realised log(W_H / s) per 16-beat section for every band
  request and seed.

Generation here uses the corrected fitted grid of each song from the build and the chart's T::

    python -m ensomi_model.audio_rows.evaluate --model <run>/model.pt --data <build> --out <dir>
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import Dataset, section_label, sections
from .generate import decode, realised, requests
from .grid import corrected_grid
from .model import LABEL, load_model

TOLERANCES_MS = (10, 20, 40)


def head_f1(gen, ref, tol_ms: float) -> dict:
    """Precision, recall and F1 of greedy one-to-one matching of sorted times within ``tol_ms``."""
    gen, ref = np.sort(np.asarray(gen, dtype=np.float64)), np.sort(np.asarray(ref, dtype=np.float64))
    i = j = tp = 0
    while i < len(gen) and j < len(ref):
        d = gen[i] - ref[j]
        if abs(d) <= tol_ms:
            tp += 1
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    p = tp / len(gen) if len(gen) else 0.0
    r = tp / len(ref) if len(ref) else 0.0
    return dict(precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0, n_gen=len(gen), n_ref=len(ref))


def bootstrap(records, field: str, seed: int = 711, n: int = 2000) -> dict:
    """Per-beat mean of ``field`` with a 95 % interval from resampling song groups."""
    groups = defaultdict(lambda: np.zeros(2))
    for r in records:
        groups[r['group']] += (r[field], r['beats'])
    vals = np.array(list(groups.values()))
    if not len(vals):
        return dict(mean=None, ci95=None, groups=0, charts=0, beats=0)
    sampled = vals[np.random.default_rng(seed).integers(len(vals), size=(n, len(vals)))].sum(1)
    return dict(mean=float(vals[:, 0].sum() / vals[:, 1].sum()),
                ci95=np.quantile(sampled[:, 0] / sampled[:, 1], [0.025, 0.975]).tolist(),
                groups=len(vals), charts=len(records), beats=int(vals[:, 1].sum()))


@torch.no_grad()
def label_shuffle(model, record, data: Dataset, seed: int = 973) -> dict:
    """Given-label count NLL, both shuffle ΔNLLs with intervals, the count marginal's entropy and
    per-chart records. Labels are standardised with the model's ``record``."""
    rng = np.random.default_rng(seed)

    def standardise(values, lengths):
        return np.repeat((values - record['label_mean']) / record['label_std'], lengths).astype(np.float32)

    def section_values(c):
        return np.array([section_label(c['wh_beat'], c['beat_ms'], a, b) for a, b in sections(len(c['count']))])

    by_edges = defaultdict(list)
    for r in data.dev:
        by_edges[(r['key'], data.charts[r['sha']]['beat_ms'].tobytes())].append(r)
    donor = {}
    for rs in by_edges.values():
        if len(rs) > 1:
            order, shift = rng.permutation(len(rs)), int(rng.integers(1, len(rs)))
            for i in range(len(rs)):
                donor[rs[order[i]]['sha']] = rs[order[(i + shift) % len(rs)]]
    records, counts = [], []
    for row in data.dev:
        c = data.charts[row['sha']]
        nb = len(c['count'])
        lengths = np.array([b - a for a, b in sections(nb)])
        true = section_values(c)
        w = data.window(row, 0, nb, rng, label=standardise(true, lengths))
        beat_vec, _ = model.beat_inputs(w['feats'], w['beat_frames'], w['slot_frames'])
        lat1 = F.one_hot(w['lattice'], 2).float()

        def nll(values):
            extra = w['extra'].clone()
            extra[:, LABEL] = torch.from_numpy(standardise(values, lengths))
            h, _ = model.run(beat_vec, w['prev_tokens'], extra)
            return float(F.cross_entropy(model.count_logits(h, lat1), w['count'], reduction='sum'))

        given = nll(true)
        shuffled = true.copy()
        for n in np.unique(lengths):
            idx = np.flatnonzero(lengths == n)
            shuffled[idx] = true[rng.permutation(idx)]
        r = dict(sha=row['sha'], group=row['group'], beats=nb, given=given, section_shuffle=nll(shuffled) - given)
        if row['sha'] in donor:
            r['chart_shuffle'] = nll(section_values(data.charts[donor[row['sha']]['sha']])) - given
        records.append(r)
        counts.append(w['count'].numpy())
    p = np.bincount(np.concatenate(counts)).astype(float)
    p = p[p > 0] / p.sum()
    return dict(given=bootstrap(records, 'given'), section_shuffle=bootstrap(records, 'section_shuffle'),
                same_audio_chart_shuffle=bootstrap([r for r in records if 'chart_shuffle' in r], 'chart_shuffle'),
                count_marginal_entropy=float(-(p * np.log(p)).sum()), records=records)


def generate_dev(model, record, data: Dataset, row, band: int, seed: int):
    """Heads for a development chart's audio on its corrected fitted grid, requesting ``band``."""
    grid, _ = corrected_grid(data.audio[row['key']]['segments'], row['song_ms'])
    wanted = requests(grid, row['song_ms'], band, record['bands'])
    label = (wanted - record['label_mean']) / record['label_std']
    return decode(model, data.features(row['key']), grid, row['song_ms'], label=label, seed=seed)


def dev_panel(data: Dataset, n: int, seed: int = 90210) -> list[dict]:
    """Up to ``n`` fit_dev charts, one per song group, in a seeded order."""
    seen, panel = set(), []
    for i in np.random.default_rng(seed).permutation(len(data.dev)):
        r = data.dev[i]
        if r['group'] not in seen and len(panel) < n:
            seen.add(r['group'])
            panel.append(r)
    return panel


def source_f1(model, record, data: Dataset, panel, seeds, cache) -> list[dict]:
    """F1 of generated heads against each panel chart, requesting the chart's own band."""
    from ..r2.cache import load_chart
    rows = []
    for row in panel:
        ref = load_chart(Path(cache) / 'charts' / row['file']).head_ms
        for seed in seeds:
            heads, _ = generate_dev(model, record, data, row, row['band'], seed)
            rows.append(dict(sha=row['sha'], group=row['group'], band=row['band'], seed=seed, K=len(heads),
                             K_source=len(ref), f1={t: head_f1(heads, ref, t)['f1'] for t in TOLERANCES_MS}))
    return rows


def dose_response(model, record, data: Dataset, panel, seeds, bands=(2, 3, 4, 5)) -> list[dict]:
    """Per song, band and seed: the requested and the realised log(W_H / s) per 16-beat section."""
    rows = []
    for row in panel:
        for band in bands:
            for seed in seeds:
                heads, beats = generate_dev(model, record, data, row, band, seed)
                scopes = sections(beats['nb'])
                rows.append(dict(sha=row['sha'], group=row['group'], band=band, seed=seed, K=len(heads),
                                 requested=record['bands'][band],
                                 realised=realised(heads, row['song_ms'], beats['beat_ms'], scopes),
                                 section_ms=[[float(beats['beat_ms'][a]), float(beats['beat_ms'][b])]
                                             for a, b in scopes]))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True, help='build directory from audio_rows.data')
    p.add_argument('--cache', default='artifacts/r2-cache/v1', help='R2 cache, for the source charts')
    p.add_argument('--out', required=True)
    p.add_argument('--grid', choices=('chart', 'fitted'), help='targets for the shuffle; default: the training grid')
    p.add_argument('--songs', type=int, default=128)
    p.add_argument('--dose-songs', type=int, default=12)
    p.add_argument('--seeds', type=int, nargs='+', default=(0, 1, 2))
    p.add_argument('--threads', type=int, default=3)
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    model, record = load_model(a.model)
    data = Dataset(a.data, a.grid or record['grid'])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shuffle = label_shuffle(model, record, data)
    (out / 'label_shuffle.json').write_text(json.dumps(shuffle, indent=1))
    panel = dev_panel(data, a.songs)
    f1 = source_f1(model, record, data, panel, a.seeds, a.cache)
    (out / 'source_f1.json').write_text(json.dumps(f1, indent=1))
    (out / 'dose.json').write_text(json.dumps(dose_response(model, record, data, panel[:a.dose_songs], a.seeds)))
    print(json.dumps(dict({k: v for k, v in shuffle.items() if k != 'records'},
                          source_f1={t: float(np.mean([r['f1'][t] for r in f1])) for t in TOLERANCES_MS},
                          generations=len(f1)), indent=1))


if __name__ == '__main__':
    main()
