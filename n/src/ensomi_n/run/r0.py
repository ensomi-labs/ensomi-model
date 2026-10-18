"""Rung 0 steps (design §5.1 R0): labels and manifest, Mel cache, clock fixtures, onset check.

Run from the ensomi-model project root with ``PYTHONPATH=<tarball>/n/src``::

    python -m ensomi_n.run.r0 labels  --run artifacts/n1-night-20261018/runs/<id> --bulk artifacts/bulk/n1-cache
    python -m ensomi_n.run.r0 cache   --run ... --bulk ...
    python -m ensomi_n.run.r0 clock   --run ...
    python -m ensomi_n.run.r0 onsets  --run ... --bulk ...

Every step appends to ``<run>/progress.jsonl`` and writes its result JSON into
``<run>``; bulk outputs (labels, Mel arrays) go under ``--bulk``. Each step is
resumable: rerunning skips finished work.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

TIMING = 'artifacts/timing-20261009'
DEFAULTS = dict(
    corpus='data/r2-corpus.parquet',
    groups='artifacts/n1-night-20261018/inputs/groups.json',
    holdout1=f'{TIMING}/opt/holdout/holdout.csv',
    holdout2=f'{TIMING}/opt/ft/pop/holdout2.csv',
    panel=f'{TIMING}/opt/ft/acc/pops/fit_dev.json',
    reset=f'{TIMING}/repr/reset/population.csv',
)


def _progress(run: Path):
    def write(rec: dict):
        rec = dict(rec, t=round(time.time(), 1))
        with open(run / 'progress.jsonl', 'a') as handle:
            handle.write(json.dumps(rec) + '\n')
    return write


def _dump(obj, path: Path) -> None:
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))
    os.replace(tmp, path)


def cmd_labels(a) -> None:
    from ..data import items as I, manifest as M
    run, bulk = Path(a.run), Path(a.bulk)
    run.mkdir(parents=True, exist_ok=True)
    progress = _progress(run)
    rows = M.load_corpus(a.corpus)
    groups = json.load(open(a.groups))
    holdout = M.read_csv_shas(a.holdout1) | M.read_csv_shas(a.holdout2)
    panel_audio = {p['audio_rel'] for p in json.load(open(a.panel))}
    import csv
    reset_audio = {r['audio_rel'] for r in csv.DictReader(open(a.reset))}
    recs = M.build_recordings(rows, groups=groups, holdout_shas=holdout, panel_audio=panel_audio,
                              reset_audio=reset_audio)
    sha = M.write_json(recs, run / 'recordings.json')
    progress(dict(step='recordings', n=len(recs), sha256=sha))
    items = I.recording_items(recs, roles=('n_train',)) + I.panel_items(recs, rows, a.panel, a.reset)
    labels_dir = bulk / I.READER_VERSION
    summary = I.read_all(items, a.root, labels_dir, processes=a.processes, progress=progress)
    by_item = {s['item'] + ':' + s['panel']: s for s in summary}
    ok = [s for s in summary if s.get('ok')]
    bad_grid = sorted(s['item'] for s in ok if not (s['on_grid'] >= 0.99))
    train_rows = [s for s in ok if s['panel'] == '']
    hard = {s['rec']: bool(s['hard']) for s in train_rows}
    excluded = set(s['rec'] for s in train_rows if not (s['on_grid'] >= 0.99))
    sel = M.select(recs, hard, excluded=excluded, n_val_hard=a.val_hard, n_val_random=a.val_random,
                   n_train=a.train, n_train_hard=a.train_hard)
    item_of = {it['rec']: it for it in items if it['panel'] == ''}
    chosen = dict(train=[item_of[r] for r in sel['train']], val_hard=[item_of[r] for r in sel['val_hard']],
                  val_random=[item_of[r] for r in sel['val_random']],
                  panel=[it for it in items if it['panel'] == 'fit_dev'],
                  reset=[it for it in items if it['panel'] == 'reset'])
    for name, lst in chosen.items():
        for it in lst:
            s = by_item.get(it['item'] + ':' + it['panel'], {})
            it['strata'] = s.get('strata', [])
            it['hard'] = bool(s.get('hard', False))
            it['label_ok'] = bool(s.get('ok', False))
    split_sha = M.write_json(chosen, run / 'split.json')
    _dump(summary, run / 'labels-summary.json')
    n_ok = len(ok)
    from collections import Counter
    strata_counts = {name: dict(Counter(x for it in lst for x in it['strata'])) for name, lst in chosen.items()}
    result = dict(
        reader=I.READER_VERSION, labels_dir=str(labels_dir), recordings_sha256=sha, split_sha256=split_sha,
        roles=dict(Counter(r['role'] for r in recs)), items=len(items), read_ok=n_ok,
        read_errors=[dict(item=s['item'], panel=s['panel'], error=s['error']) for s in summary if not s.get('ok')][:50],
        n_read_errors=len(summary) - n_ok,
        on_grid_share_ge_099=round(1 - len(bad_grid) / max(n_ok, 1), 5), on_grid_below=bad_grid,
        gate_labels=bool(n_ok and (1 - len(bad_grid) / n_ok) >= 0.99),
        resets=int(sum(s['n_resets'] for s in ok)), sets_with_resets=int(sum(s['n_resets'] > 0 for s in ok)),
        meters=dict(Counter(m for s in ok for m in s['meters'])),
        expressive=dict(implausible_lines=int(sum(s['expressive']['implausible'] for s in ok)),
                        sets_with_implausible=int(sum(s['expressive']['implausible'] > 0 for s in ok)),
                        tiny_segments=int(sum(s['expressive']['tiny'] for s in ok))),
        timing_inconsistent_sets=int(sum(not s['timing_consistent'] for s in ok)),
        flags_pool={k: int(sum(s[k] for s in train_rows if k in s)) for k in
                    ('multi_tempo', 'reset', 'short', 'meter', 'tempo_mapped', 'mixed', 'level', 'glitch', 'hard')},
        selection=dict(pool=sel['pool'], pool_hard=sel['pool_hard'], train=len(sel['train']),
                       train_hard=len(sel['train_hard']), val_hard=len(sel['val_hard']),
                       val_random=len(sel['val_random']), salt=sel['salt']),
        strata_interim=strata_counts, interim_strata=True)
    _dump(result, run / 'labels-result.json')
    progress(dict(step='labels_done', ok=n_ok, gate_labels=result['gate_labels']))


def _audio_jobs(split: dict, root: str) -> list[tuple[str, str]]:
    seen = {}
    for lst in split.values():
        for it in lst:
            seen.setdefault(it['audio_sha256'], os.path.join(root, it['audio_rel']))
    return [(p, s) for s, p in sorted(seen.items())]


def cmd_cache(a) -> None:
    from ..features import cache
    run, bulk = Path(a.run), Path(a.bulk)
    progress = _progress(run)
    split = json.load(open(run / 'split.json'))
    jobs = _audio_jobs(split, a.root)
    if a.limit:
        jobs = jobs[:a.limit]
    progress(dict(step='cache_start', n=len(jobs)))
    res = cache.build(jobs, bulk / 'mel', run / 'cache-index.jsonl', processes=a.processes,
                      progress=lambda r: progress(dict(step='cache', **r)))
    from ..features.mel import DEFAULT
    res.update(mel=DEFAULT.key(), root=str(bulk / 'mel'))
    _dump(res, run / 'cache-result.json')
    progress(dict(step='cache_done', **res['counts']))


def cmd_clock(a) -> None:
    from ..features import fixtures
    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    rep = fixtures.run(Path(a.bulk) / 'clock-fixtures')
    _dump(rep, run / 'clock-result.json')
    _progress(run)(dict(step='clock_done', passed=rep['passed']))


def cmd_onsets(a) -> None:
    """§3.2 set-offset check: per item, median (Mel onset - head-holding line); flags beyond 3 x 7.41 ms."""
    from ..data import checks, items as I
    from ..features import cache
    run, bulk = Path(a.run), Path(a.bulk)
    split = json.load(open(run / 'split.json'))
    labels_dir = bulk / I.READER_VERSION
    rows = []
    for name, lst in split.items():
        for it in lst:
            try:
                mel = cache.load_mel(bulk / 'mel', it['audio_sha256'], mmap=False).astype(np.float32)
                lab = I.load_labels(labels_dir, it['item'])
                r = checks.onset_residual(lab, checks.onset_times(mel))
            except Exception as exc:
                r = float('nan')
            rows.append(dict(item=it['item'], split=name, residual_ms=None if np.isnan(r) else round(r, 2)))
    vals = np.array([r['residual_ms'] for r in rows if r['residual_ms'] is not None])
    med = float(np.median(vals)) if len(vals) else float('nan')
    for r in rows:
        r['flag'] = r['residual_ms'] is not None and abs(r['residual_ms'] - med) > checks.OFFSET_FLAG_MS
    res = dict(population_median_ms=round(med, 2), n=len(rows), n_valid=int(len(vals)),
               n_flagged=int(sum(r['flag'] for r in rows)),
               flagged_by_split={k: int(sum(r['flag'] for r in rows if r['split'] == k)) for k in split},
               robust_sd_ms=round(float(1.4826 * np.median(np.abs(vals - med))), 2) if len(vals) else None,
               rows=rows)
    _dump(res, run / 'onsets-result.json')
    _progress(run)(dict(step='onsets_done', n_flagged=res['n_flagged']))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog='python -m ensomi_n.run.r0')
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('labels', 'cache', 'clock', 'onsets'):
        s = sub.add_parser(name)
        s.add_argument('--run', required=True)
        s.add_argument('--bulk', default='artifacts/bulk/n1-cache')
        s.add_argument('--root', default='.', help='project root holding dataset/')
        s.add_argument('--processes', type=int, default=4)
        if name == 'labels':
            for k, v in DEFAULTS.items():
                s.add_argument('--' + k, default=v)
            s.add_argument('--train', type=int, default=1600)
            s.add_argument('--train-hard', type=int, default=800)
            s.add_argument('--val-hard', type=int, default=150)
            s.add_argument('--val-random', type=int, default=150)
        if name == 'cache':
            s.add_argument('--limit', type=int, default=0)
    a = p.parse_args(argv)
    dict(labels=cmd_labels, cache=cmd_cache, clock=cmd_clock, onsets=cmd_onsets)[a.cmd](a)


if __name__ == '__main__':
    main()
