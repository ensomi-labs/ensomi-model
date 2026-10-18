"""Command line of the N line's timing evaluation.

``python -m ensomi_n.eval score --refs REF... --preds DIR [--strata TABLE.csv] --out REPORT.json``
scores every reference set under ``--refs`` against its prediction in
``--preds``. A set keyed K (see `ensomi_n.labels.find_sets`) takes the first
of ``DIR/K.json``, ``DIR/K.osu`` or the directory ``DIR/K/``, which must hold
one music map (``.json``) or `.osu` files (the first in sorted order is
used). A set without a prediction is scored as an empty prediction and listed
under ``inputs.missing_preds``, so that every arm is scored on the same
recordings; ``--skip-missing`` leaves such sets out instead.

``python -m ensomi_n.eval nulls [--out NULLS.json]`` prints the exact BL-τ
chance per canonical tempo and checks the scorer against every null.

Exit status: 0 on success; 1 when ``nulls`` finds a null the scorer misses;
2 on bad arguments or unreadable input.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

from .. import __version__
from ..labels import find_sets, read_set
from ..mapformat import load
from . import nulls
from .measures import TAUS, score
from .report import ROLES, CHARACTERISES, Entry, build_strata, read_strata


def _find_pred(preds: Path, key: str) -> Path | None:
    for candidate in (preds / f'{key}.json', preds / f'{key}.osu'):
        if candidate.is_file():
            return candidate
    folder = preds / key
    if folder.is_dir():
        maps = sorted(folder.glob('*.json'))
        if len(maps) == 1:
            return maps[0]
        if len(maps) > 1:
            raise ValueError(f'{folder}: several music maps; keep one per set')
        charts = sorted(folder.glob('*.osu'))
        if charts:
            return charts[0]
    return None


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def run_score(args) -> dict:
    sets = find_sets(args.refs)
    if not sets:
        raise ValueError(f'No .osu file under {args.refs}')
    table = read_strata(args.strata) if args.strata else {}
    unknown = sorted(set(table) - set(sets))
    preds = Path(args.preds)
    entries, by_span, per_set, missing, errors = [], {}, [], [], []
    used_preds = set()
    for key, files in sets.items():
        try:
            ref = read_set(files, key=key)
        except ValueError as exc:
            errors.append(dict(key=key, error=str(exc)))
            continue
        pred_path = _find_pred(preds, key)
        if pred_path is None:
            missing.append(key)
            if args.skip_missing:
                continue
            pred = None
        else:
            used_preds.add(pred_path.resolve())
            pred = load(pred_path)
        kw = dict(flag_below=args.flag_below, confident_at=args.confident_at,
                  decoded_to_osu_ms=args.decoded_to_osu_ms)
        whole = score(ref, pred, **kw)
        info = table.get(key, {})
        strata = dict(info.get('strata', {}))
        for name, spans in strata.items():
            if spans is not None:
                by_span[(key, name)] = score(ref, pred, spans=spans, **kw)
        recording = info.get('recording') or key
        entries.append(Entry(whole, recording, info.get('source', ''), info.get('reference'), strata))
        per_set.append(dict(whole.summary(), recording=recording, source=info.get('source', ''),
                            strata=sorted(strata), pred=str(pred_path) if pred_path else None,
                            timing_consistent=ref.meta.get('timing_consistent'),
                            timing_chart=ref.meta.get('timing_chart')))
    all_preds = {p.resolve() for p in preds.rglob('*') if p.suffix.lower() in ('.json', '.osu')} \
        if preds.is_dir() else set()
    report = dict(
        harness=dict(package='ensomi-n', version=__version__, deciding='bl20', taus_ms=list(TAUS),
                     bootstrap=args.bootstrap, seed=args.seed, flag_below=args.flag_below,
                     confident_at=args.confident_at, decoded_to_osu_ms=args.decoded_to_osu_ms),
        roles=dict(ROLES, other=CHARACTERISES),
        nulls=nulls.table(),
        inputs=dict(refs=[str(r) for r in args.refs], preds=str(preds), strata=args.strata,
                    sets=len(sets), scored=len(per_set), missing_preds=missing,
                    unused_preds=sorted(str(p) for p in all_preds - used_preds), errors=errors,
                    strata_keys_without_refs=unknown,
                    timing_inconsistent=sorted(r['key'] for r in per_set if r['timing_consistent'] is False)),
        strata=build_strata(entries, by_span, bootstrap=args.bootstrap, seed=args.seed),
        sets=per_set,
    )
    return _jsonable(report)


def run_nulls(args) -> dict:
    checks = nulls.check_all(phases=args.phases)
    return _jsonable(dict(harness=dict(package='ensomi-n', version=__version__), table=nulls.table(),
                          checks=checks, **{'pass': all(c['pass'] for c in checks)}))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m ensomi_n.eval', description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('score', help='score predictions against reference charts')
    p.add_argument('--refs', nargs='+', required=True, help='.osu files or directories of reference sets')
    p.add_argument('--preds', required=True, help='directory of music maps (.json) or .osu predictions')
    p.add_argument('--strata', help='strata table (CSV): key, recording, stratum, source, start_ms, end_ms, '
                                    'reference')
    p.add_argument('--out', required=True, help='report path (JSON)')
    p.add_argument('--bootstrap', type=int, default=2000, help='recording-bootstrap resamples (0: none)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--flag-below', type=float, default=0.5,
                   help='confidence under which a window is flagged "listen here"')
    p.add_argument('--confident-at', type=float, default=0.9, help='confidence from which a beat is confident')
    p.add_argument('--decoded-to-osu-ms', type=float, default=None,
                   help='the one measured constant that moves decoded-clock maps onto osu! time')
    p.add_argument('--skip-missing', action='store_true', help='leave out sets without a prediction')
    q = sub.add_parser('nulls', help='exact chance values and a check of the scorer against them')
    q.add_argument('--out', help='write the check as JSON')
    q.add_argument('--phases', type=int, default=1000)
    args = parser.parse_args(argv)
    try:
        if args.command == 'score':
            report = run_score(args)
            Path(args.out).write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
            top = report['strata'].get('all', {})
            bl = (top.get('bl20') or {})
            print(f"{report['inputs']['scored']} sets, {top.get('n_recordings', 0)} recordings; "
                  f"BL-20 {bl.get('value')} [{bl.get('lo')}, {bl.get('hi')}]; "
                  f"missing predictions {len(report['inputs']['missing_preds'])}; report {args.out}")
            return 0
        result = run_nulls(args)
        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
        print('canonical BPM  beat ms  ' + '  '.join(f'BL-{t:g} chance' for t in TAUS))
        for row in result['table']:
            print(f"{row['bpm']:13.1f}  {row['beat_ms']:7.2f}  "
                  + '  '.join(f"{row[f'bl{t:g}']:13.4f}" for t in TAUS))
        for c in result['checks']:
            print(f"{c['bpm']:.0f} BPM: whole shift {c['whole_shift']}, per-beat phase {c['beat_phase']}, "
                  f"pass {c['pass']}")
        return 0 if result['pass'] else 1
    except (ValueError, OSError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
