"""Supplement A3 with canonical-beat-matched background frames."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ensomi_n.v0.labels import V0Labels
from . import common as C
from . import measures as M


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    rows = []
    for x in C.items(a.root):
        if 'error' in x:
            continue
        lab = V0Labels.from_reference(x['ref'])
        for seed in (0, 1):
            path = a.root / C.CACHE / f'v01-s{seed}-009001/{x["item"][:2]}/{x["item"]}.npz'
            with np.load(path) as z:
                logits = z['logits'][:, 6].astype(float)
            for kind, code in [('tempo', 0), ('reset', 1)]:
                positive = lab.bounds[lab.bound_kind == code]
                for radius in (0, 3):
                    v = M.boundary_auc(logits, lab.bounds, positive, lab.lo_ms, lab.hi_ms,
                                       radius=radius, negative_times=lab.lines)
                    rows.append(dict(rec=x['rec'], key=x['key'], panel=x['panel'], seed=seed,
                                     kind=kind, radius=radius, **v))
    C.write_rows(a.out / 'a3-beat-control.jsonl', rows)
    C.write_json(a.out / 'a3-beat-control-summary.json', {
        f'{panel}/s{seed}/{kind}/radius{radius}':M.summarize(
            [r for r in rows if r['panel'] == panel and r['seed'] == seed and r['kind'] == kind and r['radius'] == radius], ['auc'])
        for panel in ('panel', 'val_hard') for seed in (0, 1) for kind in ('tempo', 'reset') for radius in (0, 3)})


if __name__ == '__main__':
    main()
