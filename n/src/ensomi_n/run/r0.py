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
        flags_pool=dict({k: int(sum(bool(s['flags'].get(k)) for s in train_rows)) for k in
                         ('multi_tempo', 'reset', 'short', 'meter', 'tempo_mapped', 'mixed', 'level', 'glitch')},
                        hard=int(sum(bool(s['hard']) for s in train_rows))),
        selection=dict(pool=sel['pool'], pool_hard=sel['pool_hard'], train=len(sel['train']),
                       train_hard=len(sel['train_hard']), val_hard=len(sel['val_hard']),
                       val_random=len(sel['val_random']), salt=sel['salt'], version=sel['version'], val_random_hard=sel['val_random_hard'],
                       train_rest_hard=sel['train_rest_hard']),
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


EXTRA_SUBDIVISIONS = dict(fine=(32, 48), tuplet=(5, 7, 9, 10, 14, 18))


def _grid_residual(ref, heads: np.ndarray, subdivisions) -> tuple[np.ndarray, np.ndarray]:
    """(residual ms to the nearest of ``subdivisions`` of the notated beat, before-first-beat mask) per head."""
    b, step = ref.timing.beats, ref.timing.step_ms
    if not len(b) or not len(heads):
        return np.full(len(heads), np.inf), np.ones(len(heads), dtype=bool)
    from ..timing import MERGE_MS
    i = np.searchsorted(b, heads + MERGE_MS, side='right') - 1
    before = i < 0
    i = np.maximum(i, 0)
    s = np.where(np.isfinite(step[i]) & (step[i] > 0), step[i], np.nan)
    x = (heads - b[i]) / s
    res = np.full(len(heads), np.inf)
    for d in subdivisions:
        res = np.minimum(res, np.abs(x - np.round(x * d) / d) * s)
    res[before | ~np.isfinite(res)] = np.inf
    return res, before


def cmd_gridcheck(a) -> None:
    """Why sets fall below 0.99 heads on grid: finer or tuplet subdivisions, a 3 ms tolerance, or a
    difficulty with its own timing. Reads the sets below 0.99 and a random sample above (read only)."""
    from ..data import checks
    from ..labels import read_osu, read_set
    from ..timing import MERGE_MS, Reference
    run = Path(a.run)
    summary = json.load(open(run / 'labels-summary.json'))
    split = json.load(open(run / 'split.json'))
    charts_of = {}
    for lst in split.values():
        for it in lst:
            charts_of[(it['item'], it['panel'])] = it['charts']
    recs = {r['rec']: r for r in json.load(open(run / 'recordings.json'))} if (run / 'recordings.json').exists() else {}
    for r in recs.values():
        if r.get('charts'):
            charts_of.setdefault((r['audio_sha256'], ''), r['charts'])
    bad = [s for s in summary if s.get('ok') and not (s['on_grid'] >= checks.ON_GRID_MIN)]
    good = [s for s in summary if s.get('ok') and s['on_grid'] >= checks.ON_GRID_MIN]
    rng = np.random.default_rng(0)
    sample = [good[i] for i in rng.choice(len(good), size=min(a.sample, len(good)), replace=False)]
    std = checks.SUBDIVISIONS
    ext = std + EXTRA_SUBDIVISIONS['fine'] + EXTRA_SUBDIVISIONS['tuplet']
    rows = []
    for group, lst in (('below', bad), ('sample_above', sample)):
        for s in lst:
            key = (s['item'], 'fit_dev' if s['panel'] == 'fit_dev' else ('reset' if s['panel'] == 'reset' else ''))
            charts = charts_of.get(key)
            if not charts:
                rows.append(dict(item=s['item'], group=group, error='charts not found'))
                continue
            try:
                paths = [os.path.join(a.root, c) for c in charts]
                ref = read_set(paths, key=s['item'])
                h = ref.heads[ref.scorable(ref.heads)]
                r_std, before = _grid_residual(ref, h, std)
                r_ext, _ = _grid_residual(ref, h, ext)
                r_fine, _ = _grid_residual(ref, h, std + EXTRA_SUBDIVISIONS['fine'])
                off = r_std > MERGE_MS
                row = dict(item=s['item'], group=group, panel=s['panel'], strata=s.get('strata'), n_heads=int(len(h)),
                           on_grid=float(np.mean(~off)) if len(h) else None,
                           with_fine=float(np.mean(r_fine <= MERGE_MS)) if len(h) else None,
                           with_fine_tuplet=float(np.mean(r_ext <= MERGE_MS)) if len(h) else None,
                           tol3=float(np.mean(r_std <= 3.0)) if len(h) else None,
                           tol5=float(np.mean(r_std <= 5.0)) if len(h) else None,
                           before_first_beat=int(before.sum()),
                           off_residual_ms_q=[round(float(x), 2) for x in np.percentile(r_std[off & np.isfinite(r_std)],
                                                                                        [10, 50, 90])]
                           if np.any(off & np.isfinite(r_std)) else None,
                           n_charts=len(paths), timing_consistent=bool(ref.meta.get('timing_consistent')))
                per = []
                for pth in paths:
                    c = read_osu(pth)
                    hc = c.heads[ref.scorable(c.heads)] if len(c.heads) else c.heads
                    if not len(hc):
                        continue
                    rc, _ = _grid_residual(ref, hc, std)
                    own = None
                    if c.red_lines:
                        try:
                            oref = Reference.from_segments(c.segments(), c.heads)
                            ho = oref.heads[oref.scorable(oref.heads)]
                            ro, _ = _grid_residual(oref, ho, std)
                            own = float(np.mean(ro <= MERGE_MS)) if len(ho) else None
                        except Exception:
                            own = None
                    per.append(dict(chart=os.path.basename(pth), n=int(len(hc)), on_set_timing=float(np.mean(rc <= MERGE_MS)),
                                    on_own_timing=own, timing_chart=str(pth) == ref.meta.get('timing_chart')))
                row['charts'] = per
                worst = min(per, key=lambda x: x['on_set_timing']) if per else None
                row['worst_chart_on_set'] = worst['on_set_timing'] if worst else None
                row['worst_chart_on_own'] = worst['on_own_timing'] if worst else None
                rows.append(row)
            except Exception as exc:
                rows.append(dict(item=s['item'], group=group, error=f'{type(exc).__name__}: {exc}'))
    below = [r for r in rows if r['group'] == 'below' and 'error' not in r and r['n_heads']]

    def share(key):
        return round(float(np.mean([r[key] >= checks.ON_GRID_MIN for r in below])), 4) if below else None
    own_fix = [r for r in below if r['worst_chart_on_own'] is not None and r['worst_chart_on_set'] < checks.ON_GRID_MIN
               and r['worst_chart_on_own'] >= checks.ON_GRID_MIN]
    res = dict(n_below=len(bad), n_read=len(below), n_errors=sum('error' in r for r in rows),
               pass_with_fine=share('with_fine'), pass_with_fine_tuplet=share('with_fine_tuplet'),
               pass_tol3=share('tol3'), pass_tol5=share('tol5'),
               below_with_a_chart_on_own_timing_only=len(own_fix),
               below_timing_inconsistent=sum(not r['timing_consistent'] for r in below),
               on_grid_q_below=[round(float(x), 4) for x in np.percentile([r['on_grid'] for r in below], [10, 25, 50, 75, 90])]
               if below else None,
               subdivisions=dict(standard=std, extra=EXTRA_SUBDIVISIONS), rows=rows)
    _dump(res, run / 'gridcheck-result.json')
    print(json.dumps({k: v for k, v in res.items() if k != 'rows'}, indent=1))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog='python -m ensomi_n.run.r0')
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('labels', 'cache', 'clock', 'onsets', 'gridcheck'):
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
        if name == 'gridcheck':
            s.add_argument('--sample', type=int, default=150)
    a = p.parse_args(argv)
    dict(labels=cmd_labels, cache=cmd_cache, clock=cmd_clock, onsets=cmd_onsets, gridcheck=cmd_gridcheck)[a.cmd](a)


if __name__ == '__main__':
    main()
