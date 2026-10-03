"""Calibration harness v0: the trivial event-field evaluator (head rate), end to end on the R2 corpus.

Run on the machine that holds ``dataset/``, from the repository root::

    python -m ensomi_model.evaluation.harness.run --out artifacts/eval-harness-v0-20261003/run-2 \\
        --provenance ../.sync/cp/jobs/<job id>/provenance.json

Order: fit pass (values, descriptions) on the fit split; the model fitted, its
bandwidth chosen by cross-fitted log-likelihood and written to ``model.json``
before any calibration chart is read; second fit pass (cross-fitted song
statistics and continuation windows: the song nulls); calibration pass (floor,
transforms, injections); song p-values; report. The held-out split is never
requested.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any

import numpy as np

from .access import CorpusAccess
from .cases import DEFAULT_FAMILY, INJECTION_STARS, calibration_chart, fit_null, fit_values, init_worker, load_family
from .chain import STATS, SongNull
from .trivial import bpm_band, fold_of, star_band

DEFAULT_SEED = 20261003
NULL_WHOLE, NULL_CONTINUATION = 'whole', 'continuation'
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def log(message: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def _subsample(rows: list[dict], limit: int | None, seed: int) -> list[dict]:
    """A seeded subsample of whole song groups, at least ``limit`` groups (all rows of each kept group)."""
    if not limit:
        return rows
    groups = sorted({r['group_id'] for r in rows})
    if limit >= len(groups):
        return rows
    keep = {groups[i] for i in np.random.default_rng(seed).choice(len(groups), size=limit, replace=False)}
    return [r for r in rows if r['group_id'] in keep]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--corpus', default='data/r2-corpus.parquet')
    parser.add_argument('--out', required=True)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--family', default=DEFAULT_FAMILY,
                        help='the event family, module:attribute (a field.EventFamily or a factory of one)')
    parser.add_argument('--provenance', help="control plane's provenance.json for this job (git identity)")
    parser.add_argument('--fit-limit', type=int, help='seeded subsample of fit song groups (smoke runs only)')
    parser.add_argument('--calibration-limit', type=int, help='seeded subsample of calibration song groups')
    parser.add_argument('--report-only', action='store_true', help='re-run analysis and report from state.pkl')
    parser.add_argument('--previous', help="an earlier run's report.json, summarised beside this run")
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(argv)
    for name in THREAD_VARIABLES:  # one BLAS thread per worker process: no oversubscription
        os.environ.setdefault(name, '1')
    started = time.time()
    phases: dict[str, Any] = {}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        from .report import analyse, write_outputs
        with (out / 'state.pkl').open('rb') as handle:
            fit, cal, model_info, phases = pickle.load(handle)
        phases = dict(phases, report_only_rerun=True)
        write_outputs(out, fit, cal, analyse(fit, cal, model_info, args), model_info, args, phases, argv)
        log('report rewritten from state.pkl')
        return 0
    access = CorpusAccess(args.corpus, out / 'access_log.jsonl')
    fit_rows = _subsample(access.rows('fit', caller='harness.run.main: fit pass'), args.fit_limit, args.seed)
    folds = [fold_of(r['group_id']) for r in fit_rows]
    log(f'fit charts: {len(fit_rows)}')

    family = load_family(args.family)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(args.family,)) as pool:
        fit_out = list(pool.map(fit_values, zip(fit_rows, folds), chunksize=16))
    phases['fit_values_s'] = time.time() - started
    log(f'fit pass 1 done in {phases["fit_values_s"]:.0f}s; errors: {sum("error" in r for r in fit_out)}')
    model = family.fit([r['suff'] for r in fit_out if 'suff' in r])
    model_info = write_model(out, model, family)
    log(f'bandwidth={model.bandwidth} loglik={model.loglik}')

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(args.family, model)) as pool:
        null_out = list(pool.map(fit_null, zip(fit_rows, folds), chunksize=16))
        phases['fit_null_s'] = time.time() - t0
        log(f'fit pass 2 done in {phases["fit_null_s"]:.0f}s; errors: {sum("error" in r for r in null_out)}')
        fit = build_fit(fit_rows, folds, fit_out, null_out, model)

        cal_rows = _subsample(access.rows('calibration', caller='harness.run.main: calibration pass'),
                              args.calibration_limit, args.seed + 1)
        tasks = calibration_tasks(cal_rows, model, args.seed)
        log(f'calibration charts: {len(cal_rows)}; injected: {sum(t[1]["inject"] for t in tasks)}')
        t1 = time.time()
        cal_out = []
        for i, result in enumerate(pool.map(calibration_chart, tasks, chunksize=1)):
            cal_out.append(result)
            if (i + 1) % 100 == 0:
                log(f'calibration {i + 1}/{len(tasks)} ({time.time() - t1:.0f}s)')
        phases['calibration_pass_s'] = time.time() - t1

    from .report import analyse, write_outputs
    t2 = time.time()
    cal = build_calibration(cal_rows, cal_out, fit)
    phases['scoring_s'] = time.time() - t2
    with (out / 'state.pkl').open('wb') as handle:
        pickle.dump((fit, cal, model_info, phases), handle, protocol=pickle.HIGHEST_PROTOCOL)
    analysis = analyse(fit, cal, model_info, args)
    phases['analysis_s'] = time.time() - t2
    phases['total_s'] = time.time() - started
    write_outputs(out, fit, cal, analysis, model_info, args, phases, argv)
    log(f'done in {phases["total_s"]:.0f}s; outputs in {out}')
    return 0


def write_model(out: Path, model, family) -> dict[str, Any]:
    """``model.json``: the fitted densities as bin counts, the bandwidth and the selection table."""
    from .chain import (GIVEN_FRACTION, H_FIELD_S, NULL_H_LOG_SECONDS, NULL_H_STAR, NULL_LOG_SECONDS_ROUND, SCAN_S,
                        STEP_MS, WINDOWS)
    info = dict(family=family.name, judges=sorted(family.judges),
                estimator_settings=dict(rate_kernel_sigma_s=getattr(family, 'sigma_s', None), field_kernel_h_s=H_FIELD_S,
                                        field_step_ms=STEP_MS, scan_s=SCAN_S, null_h_star=NULL_H_STAR,
                                        null_h_log_seconds=NULL_H_LOG_SECONDS,
                                        null_log_seconds_round=NULL_LOG_SECONDS_ROUND,
                                        continuation_given_fraction=GIVEN_FRACTION,
                                        continuation_windows=[list(w) for w in WINDOWS],
                                        scoring='every song scored with the model and reference without its fold'),
                selected_before_calibration=True, **model.describe())
    (out / 'model.json').write_text(json.dumps(info, indent=1) + '\n')
    return {k: v for k, v in info.items() if k != 'keys'} | dict(
        keys={k: {kk: vv for kk, vv in v.items() if kk != 'counts'} for k, v in info['keys'].items()})


def build_fit(rows: list[dict], folds: list[int], values: list[dict], nulls: list[dict], model) -> dict[str, Any]:
    star = np.array([r['star'] for r in rows], dtype=np.float64)
    fit: dict[str, Any] = dict(rows=rows, results=values, null_results=nulls, fold=np.array(folds), star=star)
    ok = np.array(['error' not in v and 'error' not in n for v, n in zip(values, nulls)])
    whole = {name: np.array([n['whole'][name] if o else np.nan for n, o in zip(nulls, ok)]) for name in STATS}
    seconds = np.array([n['whole']['seconds'] if o else np.nan for n, o in zip(nulls, ok)])
    good = ok & (seconds > 0)
    fit['null'] = {NULL_WHOLE: SongNull(star[good], np.log(seconds[good]), np.flatnonzero(good),
                                        {k: v[good] for k, v in whole.items()})}
    fit['whole_stats'] = dict(whole, seconds=seconds)
    w_star, w_log, w_song, w_stats, w_meta = [], [], [], {name: [] for name in STATS}, []
    for i, (n, o) in enumerate(zip(nulls, ok)):
        if not o:
            continue
        for w in n['windows']:
            if not w['seconds'] > 0:
                continue
            w_star.append(star[i])
            w_log.append(math.log(w['seconds']))
            w_song.append(i)
            w_meta.append((w['fraction'], w['position']))
            for name in STATS:
                w_stats[name].append(w[name])
    fit['null'][NULL_CONTINUATION] = SongNull(np.array(w_star), np.array(w_log), np.array(w_song, dtype=np.int64),
                                              {k: np.array(v) for k, v in w_stats.items()})
    fit['window_meta'] = np.array(w_meta)
    fit['window_song'] = np.array(w_song, dtype=np.int64)
    fit['window_stats'] = {k: np.array(v) for k, v in w_stats.items()}
    fit['window_seconds'] = np.exp(np.array(w_log))
    return fit


def calibration_tasks(rows: list[dict], model, seed: int) -> list[tuple[dict, dict]]:
    modes = {k: model.mode_rate(k) for k in range(len(model.counts)) if model.mode_rate(k) is not None}
    rng = np.random.default_rng(seed + 7)
    by_key: dict[tuple[int, int], list[int]] = {}
    inject = [INJECTION_STARS[0] <= r['star'] < INJECTION_STARS[1] for r in rows]
    keys = [(int(star_band(r['star'])), int(bpm_band(r['canonical_bpm']))) for r in rows]
    for i, k in enumerate(keys):
        if inject[i]:
            by_key.setdefault(k, []).append(i)
    tasks = []
    for i, row in enumerate(rows):
        donor = None
        if inject[i]:
            cands = [j for j in by_key[keys[i]] if rows[j]['group_id'] != row['group_id']]
            donor = rows[int(rng.choice(cands))] if cands else None
        tasks.append((row, dict(seed=seed, chart_seed=int(row['sha256'][:8], 16), inject=inject[i], modes=modes,
                                donor=donor)))
    return tasks


def build_calibration(rows: list[dict], results: list[dict], fit: dict) -> dict[str, Any]:
    """Records of every case as columns, with song p-values from the nulls."""
    records, chart_of = [], []
    for i, r in enumerate(results):
        for rec in r.get('records', []):
            records.append(rec)
            chart_of.append(i)
    chart_of = np.array(chart_of, dtype=np.int64)
    star = np.array([rows[int(c)]['star'] for c in chart_of], dtype=np.float64)
    scope = np.array([rec['scope'] for rec in records])
    seconds = np.array([rec.get('seconds', np.nan) for rec in records], dtype=np.float64)
    stats = {name: np.array([rec.get(name, np.nan) for rec in records], dtype=np.float64) for name in STATS}
    p = {name: np.full(len(records), np.nan) for name in STATS}
    n_eff = np.full(len(records), np.nan)
    for sc in (NULL_WHOLE, NULL_CONTINUATION):
        sel = np.flatnonzero(scope == sc)
        if not len(sel):
            continue
        pv, ne = fit['null'][sc].p_values(star[sel], seconds[sel], {k: v[sel] for k, v in stats.items()})
        for name in STATS:
            p[name][sel] = pv[name]
        n_eff[sel] = ne
    return dict(rows=rows, results=results, records=records, chart=chart_of, star=star, scope=scope,
                seconds=seconds, stats=stats, p=p, n_eff=n_eff)


if __name__ == '__main__':
    raise SystemExit(main())
