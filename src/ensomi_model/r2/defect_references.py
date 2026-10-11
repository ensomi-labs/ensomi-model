"""Closure-owned free-run defects and fit-train references on cached, snapped releases.

The reference denominator is holds, pooled within ``common.band_of(cache star)``.
Generated holds belong to the model when their closing decision is at or after the
first generated decision, including seed-open holds closed by the model. Source
holds already closed by the prefix do not enter either numerator or denominator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .common import band_of
from .properties import prefix_objects

VERSION = 'defect-references-v1'
FILENAME = VERSION + '.json'
METRICS = ('holds_lt60', 'release_1_40')
DEFINITION = dict(hold_owner='closing decision >= first generated decision',
                  short='end - start < 60 ms',
                  near_head='another-lane head 1 through 40 ms after the release, inclusive',
                  denominator='closed holds', source='fit_train cached snapped releases',
                  band='common.band_of(cache star)')


def closed_holds(head_ms, actions, gap, first_generated=0):
    """Return (start, end, lane, closing decision) for model-owned closed holds."""
    held = [None] * 4
    out = []
    for k, row in enumerate(actions):
        t = float(head_ms[k]) if k < len(head_ms) else None
        for lane, code in enumerate(row):
            code = int(code)
            if held[lane] is not None:
                if code == 0:
                    continue
                end = t if code == 1 else float(gap[k, lane])
                if k >= first_generated:
                    out.append((held[lane], end, lane, k))
                held[lane] = t if code == 4 else None
            elif code == 2:
                held[lane] = t
    return out


def model_defects(head_ms, actions, gap, first_generated=0):
    holds = closed_holds(head_ms, actions, gap, first_generated)
    objects = prefix_objects(head_ms, actions, gap)
    heads = [np.sort([o.start for o in objects if o.lane == lane]) for lane in range(4)]
    near = 0
    for _, end, lane, _ in holds:
        near += any(np.searchsorted(heads[l], end + 40, side='right') >
                    np.searchsorted(heads[l], end + 1, side='left') for l in range(4) if l != lane)
    return dict(holds=len(holds), holds_lt60=sum(end - start < 60 for start, end, _, _ in holds),
                release_1_40=int(near))


def dense_births(objects, head_ms):
    """LN heads among heads whose next distinct head row occurs within 60 ms."""
    dense = set(np.asarray(head_ms[:-1])[np.diff(head_ms) <= 60].tolist())
    rows = [o for o in objects if o.start in dense]
    count = sum(o.hold for o in rows)
    return dict(heads=len(rows), ln_heads=count, rate=count / len(rows) if rows else None)


def _rates(counts):
    return {key + '_rate': counts[key] / counts['holds'] if counts['holds'] else None for key in METRICS}


def build_reference(cache):
    """Build from every fit_train chart; never read fit_dev releases for reference rates."""
    from .data import Corpus
    corpus = Corpus(cache, 'fit_train', star_conditions=False, capacity=1)
    bands = {str(b): dict(charts=0, holds=0, holds_lt60=0, release_1_40=0) for b in (2, 3, 4, 5)}
    for row in corpus.table.sort_values('sha256').itertuples():
        chart = corpus.chart(row.sha256)
        counts = model_defects(chart.head_ms, chart.actions, chart.gap)
        acc = bands[str(band_of(float(row.star)))]
        acc['charts'] += 1
        for key, value in counts.items():
            acc[key] += value
    for counts in bands.values():
        counts.update(_rates(counts))
    overall = {key: sum(v[key] for v in bands.values()) for key in ('charts', 'holds', *METRICS)}
    overall.update(_rates(overall))
    return dict(version=VERSION, definition=DEFINITION, bands=bands, overall=overall,
                index_sha256=hashlib.sha256((Path(cache) / 'index.parquet').read_bytes()).hexdigest(),
                fit_train_sha256=hashlib.sha256('\n'.join(sorted(corpus.files)).encode()).hexdigest())


def load_reference(cache):
    """Read the versioned reference, rejecting changed definitions or a different cache index."""
    path = Path(cache) / FILENAME
    reference = json.loads(path.read_text())
    if reference.get('version') != VERSION or reference.get('definition') != DEFINITION:
        raise ValueError(f'Incompatible defect reference: {path}')
    if reference.get('index_sha256') != hashlib.sha256((Path(cache) / 'index.parquet').read_bytes()).hexdigest():
        raise ValueError(f'Defect reference cache index changed: {path}')
    return reference


def defect_summary(runs, reference):
    """Count ratios overall and by band; only the overall count comparisons bind guard iv."""
    def summary(rows):
        holds = sum(r['holds'] for r in rows)
        out = dict(runs=len(rows), holds=holds)
        for metric in METRICS:
            observed = sum(r[metric] for r in rows)
            rates = [reference['bands'][str(r['band'])][metric + '_rate'] for r in rows]
            expected = sum(r['holds'] * rate for r, rate in zip(rows, rates)) if all(
                rate is not None for rate in rates) else None
            out[metric] = dict(observed=observed, expected=expected,
                               ratio=observed / expected if expected else (0.0 if observed == 0 and expected == 0 else None),
                               pass_guard=bool(expected is not None and observed <= 1.25 * expected))
        out['guard_iv'] = bool(rows and all(out[m]['pass_guard'] for m in METRICS))
        return out
    overall = summary(runs)
    return dict(version='r2-defects-v2', reference_version=reference['version'], definition=DEFINITION,
                reference_sha256=hashlib.sha256(json.dumps(reference, sort_keys=True,
                                                           separators=(',', ':')).encode()).hexdigest(),
                overall=overall, by_band={str(b): summary([r for r in runs if int(r['band']) == b])
                                         for b in (2, 3, 4, 5)}, guard_iv=overall['guard_iv'], runs=runs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--cache', default='artifacts/r2-cache/v1')
    ap.add_argument('--output', default=None)
    args = ap.parse_args(argv)
    path = Path(args.output) if args.output else Path(args.cache) / FILENAME
    result = build_reference(args.cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(path=str(path), **result['overall'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
