"""B3 theta channel-use probe: realised shift of nh and held share per unit of requested theta shift.

On 6 panel charts per band, from the real first-third prefix and from B3's own 512-row BOS
generation, theta is estimated on the prefix rows; 256 rows are then generated with that theta
and with the nh or held coordinate shifted by +-1 within-band fit_train SD (seed 954 for all).
The slope is the pooled regression of realised on requested shift, in standardised units.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .bakeoff_panel import CACHE, PANEL, load_model
from .cache import load_chart
from .data import chart_from_cache
from .sampling import continue_chart
from .theta import COORDINATES, THETA_FILE, ThetaTable, decode, prefix_theta

ROWS, OWN, PER_BAND, SEED = 256, 512, 6, 954
PROBED = ('nh', 'held')


def realised(actions, start, stop):
    held, head, _ = decode(actions[:stop])
    return dict(nh=head[start:stop].sum(1).mean(), held=held[start:stop].mean())


def band_sd(table, cache_root):
    index = pd.read_parquet(Path(cache_root) / 'index.parquet')[['sha256', 'band']]
    train = table.table[table.table.role == 'fit_train'].drop(columns='band', errors='ignore').merge(index, on='sha256')
    return {int(b): {c: float(g[f'theta_{c}'].std()) for c in PROBED} for b, g in train.groupby('band')}


def probe(checkpoint, out_dir, *, panel_path=PANEL, cache_root=CACHE, theta_path=THETA_FILE):
    torch.set_num_threads(2)
    model = load_model(checkpoint)
    table = ThetaTable.load(theta_path)
    sds = band_sd(table, cache_root)
    charts = json.loads(Path(panel_path).read_text())['charts']
    charts = [c for b in (2, 3, 4, 5) for c in [c for c in charts if c['band'] == b][:PER_BAND]]
    rows = []
    for spec in charts:
        chart = chart_from_cache(load_chart(Path(cache_root) / 'charts' / spec['file']))
        skeleton = (chart.head_ms, chart.song_ms, chart.grid)
        s = chart.K // 3
        for start, (actions, gap) in (('real', (chart.actions[:s], chart.gap[:s])),
                                      ('own', continue_chart(model, *skeleton, seed=SEED, stop=OWN, theta='prior',
                                                             theta_table=table,
                                                             theta_source_sha256=spec['sha256']))):
            n = len(actions)
            stop = min(n + ROWS, chart.K)
            theta = prefix_theta(table, *skeleton, actions, gap)
            run = lambda vector: continue_chart(model, *skeleton, actions, gap, seed=SEED, stop=stop, theta=vector)[0]
            base = realised(run(theta), n, stop)
            for c in PROBED:
                for sign in (-1, 1):
                    shifted = theta.copy()
                    requested = sign * sds[spec['band']][c]
                    shifted[COORDINATES.index(c)] += requested
                    value = realised(run(shifted), n, stop)[c]
                    rows.append(dict(sha256=spec['sha256'], band=spec['band'], start=start, coordinate=c,
                                     requested=requested,
                                     realised=(value - base[c]) / table.sd[COORDINATES.index(c)]))
    frame = pd.DataFrame(rows)
    slopes = {f'{start}/{c}': float((g.requested * g.realised).sum() / (g.requested ** 2).sum())
              for (start, c), g in frame.groupby(['start', 'coordinate'])}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'probe.json').write_text(json.dumps(dict(checkpoint=str(checkpoint), slopes=slopes, band_sd=sds,
                                                    rows=rows), indent=2) + '\n')
    lines = ['| Prefix | Coordinate | Slope (realised / requested, standardised) |', '| --- | --- | ---: |']
    lines += [f"| {k.split('/')[0]} | {k.split('/')[1]} | {v:.3f} |" for k, v in slopes.items()]
    (out / 'probe.md').write_text('\n'.join(lines) + '\n')
    return slopes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', default='artifacts/r2-bakeoff-20261007/b3-probe')
    parser.add_argument('--panel', default=str(PANEL))
    args = parser.parse_args(argv)
    print(json.dumps(probe(args.checkpoint, args.out, panel_path=args.panel)))


if __name__ == '__main__':
    main()
