"""Mapper-to-mapper timing spread on same-audio pairs, on the mac (research notes ``mapper-spread-20261021``).

Labels only: `.osu` files, the corpus table, the holdout lists, the dedup
groups and set metadata; std-taiko audio files are opened only to hash their
bytes, as the census does, and never decoded. CPU only. Run from the project
root with ``n/src`` of one code commit on ``PYTHONPATH``::

    nice -n 19 .venv/bin/python scripts/n1/mapper_spread.py --commit SHA \\
        --out artifacts/bulk/mapper-spread-20261021/run \\
        --holdout artifacts/n1-eval/hard-holdout-v1 --holdout-sha256 00771c64...

Units and splits come from the census driver (`census_run.build_units`, the
same inputs and defaults). Sealing:

- std-taiko is left out unless ``--with-std-taiko``: its directories are passed
  over by name before any file in them is opened;
- the hard holdout's membership is checked against ``--holdout-sha256`` first
  (`census.load_exclusion`), and its std-taiko set directories are passed over
  before any of their files is opened;
- std-taiko audio is hashed only for units outside the sealed splits;
- every unit of a sealed split (holdout1, holdout2, r2-heldout,
  std-taiko-linked-sealed) or listed in the hard holdout is dropped before any
  `.osu` is read for pairing.

Outputs in ``--out``: ``run.json`` (inputs, counts, timings), ``summary.json``
and ``summary.md`` (the tables), ``pairs.jsonl`` (one row per analysed pair, all
from unsealed splits) and ``lines.npz`` (each pair's per-line errors and head
weights, both directions). ``--preflight-only`` stops after the units.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

import census_run  # noqa: E402

from ensomi_n.eval import census, spread  # noqa: E402

HARD_HOLDOUT_SHA256 = '00771c6491ced746bc1dd536c0cd3ef453a01798a658ebe3330ab89edec327bb'


def std_dirs(exclusion: dict) -> set[str]:
    """std-taiko set directory names of the hard holdout: from its census keys, and its set ids."""
    out = set(exclusion['set_ids'])
    for key in exclusion['keys']:
        parts = key.split('@', 1)[0].split('/')
        if len(parts) >= 2 and parts[0] == 'std-taiko':
            out.add(parts[1])
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--root', default='.')
    p.add_argument('--corpus', default='data/r2-corpus.parquet')
    p.add_argument('--holdout1', default='artifacts/timing-20261009/opt/holdout/holdout.csv')
    p.add_argument('--holdout2', default='artifacts/timing-20261009/opt/ft/pop/holdout2.csv')
    p.add_argument('--groups', default='artifacts/bulk/music-dedup-20261015/groups.json')
    p.add_argument('--tags', default='artifacts/n1-inventory-20261014/tags.json')
    p.add_argument('--holdout', default='artifacts/n1-eval/hard-holdout-v1')
    p.add_argument('--holdout-sha256', default=HARD_HOLDOUT_SHA256)
    p.add_argument('--out', required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=20261021)
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--with-std-taiko', action='store_true',
                   help='also read std-taiko sets; off by default, since splitting off std-taiko-linked-sealed reads '
                        'their headers (the census found 1 differing std-taiko pair)')
    args = p.parse_args(argv)
    root = Path(args.root)
    out = root / args.out
    t0 = time.time()
    report: dict = dict(commit=args.commit, started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                        sealed=list(census_run.SEALED), holdout=args.holdout, holdout_sha256=args.holdout_sha256)
    exclusion = census.load_exclusion(root / args.holdout, args.holdout_sha256)
    skip = std_dirs(exclusion)
    if not args.with_std_taiko:
        base = root / 'dataset' / 'std-taiko'
        skip |= {d.name for d in base.iterdir() if d.is_dir()} if base.is_dir() else set()
    report['hard_holdout'] = dict(sets=len(exclusion['keys']), std_dirs_skipped=len(skip))
    units = census_run.build_units(args, report, skip_std=skip,
                                   hash_if=lambda u: u.split not in census_run.SEALED)
    kept, drops = spread.unsealed(units, exclusion, census_run.SEALED)
    report['units'] = dict(built=len(units), kept=len(kept), dropped=drops,
                           kept_by_split=census._count(u.split for u in kept))
    report['t_units_s'] = round(time.time() - t0, 1)
    print(json.dumps(report, indent=1, default=str), flush=True)
    if args.preflight_only:
        return 0
    out.mkdir(parents=True, exist_ok=True)
    t1 = time.time()
    result = spread.run(kept, root, progress=lambda msg: print(msg, flush=True))
    report['t_pairs_s'] = round(time.time() - t1, 1)
    print(json.dumps(result['funnel'], indent=1), flush=True)
    t2 = time.time()
    summary = spread.summarize(result['pairs'], draws=args.bootstrap, seed=args.seed)
    report['t_summary_s'] = round(time.time() - t2, 1)
    report['t_total_s'] = round(time.time() - t0, 1)
    meta = dict(run=report, bootstrap=args.bootstrap, seed=args.seed,
                rules=dict(copy_ms=spread.COPY_MS, copy_share=spread.COPY_SHARE, min_shared_ms=spread.MIN_SHARED_MS,
                           min_lines=spread.MIN_LINES, fast_bpm=spread.FAST_BPM, bar_tol_ms=spread.BAR_TOL_MS,
                           half_beat=spread.HALF_BEAT, near_beats=spread.NEAR_BEATS))
    spread.write(out, result, summary, meta)
    (out / 'run.json').write_text(json.dumps(spread._jsonable(dict(report, funnel=result['funnel'])), indent=1) + '\n')
    print(f"done: {result['funnel']['pairs']} pairs in {result['funnel']['recordings']} recordings; "
          f"{result['funnel']['errors']} errors; {out}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
