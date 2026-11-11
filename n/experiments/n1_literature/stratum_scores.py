"""Score saved averaging maps on the labelled spans, without further inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ensomi_n.eval import score
from ensomi_n.mapformat import MusicMap
from . import common as C
from . import measures as M

FIELDS = ('bl20', 'bars_start', 'bars_same', 'level')


def span_quality(ref, pred, spans):
    """Intersect labelled spans with the primary experiment's one-second inner range."""
    lo, hi = ref.lo_ms + 1000., ref.hi_ms - 1000.
    if spans is None:  # The stratum API uses None for the whole mapped range.
        spans = [(lo, hi)]
    clipped = [(max(a, lo), min(b, hi)) for a, b in spans if min(b, hi) > max(a, lo)]
    if not clipped:
        return None
    sc = score(ref, pred, spans=clipped)
    return {f: getattr(sc, f) for f in FIELDS}


def seed_null(rows):
    """Matched seed-1-minus-seed-0 contrasts for the original single-pass spans."""
    result = {}
    for st in sorted({r['stratum'] for r in rows}):
        seeds = [{r['rec']: r for r in rows if r['stratum'] == st and r['seed'] == s
                  and r['arm'] == 'single' and r['fresh_ms'] == 0} for s in (0, 1)]
        keys = sorted(seeds[0].keys() & seeds[1].keys())
        result[st] = {f: M.interval([seeds[1][k][f] - seeds[0][k][f] for k in keys
                                    if seeds[0][k][f] is not None and seeds[1][k][f] is not None])
                      for f in FIELDS}
    return result


def run(root, out):
    runtime = json.loads((out / 'b-runtime.json').read_text())
    cohort = json.loads((out / 'b-cohort.json').read_text())[:runtime['completed_recordings']]
    inputs = {x['key']: x for x in C.items(root, ('panel',)) if 'error' not in x}
    primary = {(r['key'], r['seed'], r['arm']): r for r in C.read_jsonl(out / 'b-recordings.jsonl')}
    rows = []
    for i, chosen in enumerate(cohort):
        x = inputs[chosen['key']]
        regions = dict(all=[(x['ref'].lo_ms, x['ref'].hi_ms)], **x['strata'])
        for seed in (0, 1):
            maps = json.loads((out / f'b-maps-{i:03d}-s{seed}.json').read_text())
            for name, raw in maps.items():
                arm, fresh = name.split('/')
                pred = MusicMap.from_json(raw)
                for stratum, spans in regions.items():
                    values = span_quality(x['ref'], pred, spans)
                    if values is None:
                        continue
                    if stratum == 'all':
                        old = primary[x['key'], seed, arm]
                        old = old['fresh'] if fresh == '500' else old
                        for f in FIELDS:
                            expected = old[f] if old[f] is not None else np.nan
                            if not np.isclose(values[f], expected, atol=1e-10, rtol=0, equal_nan=True):
                                raise RuntimeError(f'primary score mismatch: {x["key"]}/{seed}/{name}/{f}')
                    rows.append(dict(key=x['key'], rec=x['rec'], seed=seed, arm=arm,
                                     fresh_ms=int(fresh), stratum=stratum, **values))
    base = {(r['rec'], r['seed'], r['fresh_ms'], r['stratum']): r for r in rows if r['arm'] == 'single'}
    groups = {}
    for r in rows:
        b = base[r['rec'], r['seed'], r['fresh_ms'], r['stratum']]
        for f in FIELDS:
            r['delta_' + f] = r[f] - b[f]
        groups.setdefault(f"s{r['seed']}/{r['stratum']}/{r['arm']}/{r['fresh_ms']}", []).append(r)
    C.write_rows(out / 'b-span-recordings.jsonl', rows)
    summary = {k: M.summarize(rs, list(FIELDS) + ['delta_' + f for f in FIELDS]) for k, rs in groups.items()}
    C.write_json(out / 'b-span-summary.json', summary)
    C.write_json(out / 'b-span-seed-null.json', seed_null(rows))
    print(f'Scored {len(cohort)} recordings on labelled spans; all primary scores reproduced.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    C.storage_guard(a.out)
    run(a.root, a.out)
    C.storage_guard(a.out)


if __name__ == '__main__':
    main()
