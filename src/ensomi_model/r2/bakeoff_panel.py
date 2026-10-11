"""Fixed long-chart selection and process-parallel generation for the R2 bakeoff."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .cache import load_chart
from .data import chart_from_cache
from .model import R2Config, R2Model
from .sampling import continue_chart

ROOT = Path('artifacts/r2-bakeoff-20261007')
PANEL = ROOT / 'panel.json'
REFERENCE = ROOT / 'reference.json'
CHECKPOINT = Path('artifacts/r2-runs/r2-phaseN-20261006/checkpoints/ckpt-0056000574.pt')
X0 = Path('artifacts/r2-collapse-20261007/phase0-astra/x0-sealed/selection.json')
CACHE = Path('artifacts/r2-cache/v1')
SYSTEMS = ('b0', 'b1', 'b2', 'b3', 'b3-unknown', 'b3-oracle', 'd0-phi', 'd0-env')


def select_panel(table, excluded, *, seed=954, groups_per_band=25, minimum_rows=None):
    """Seeded group ordering with matching for distinct groups across all bands."""
    rng = np.random.default_rng(seed)
    minimum_rows = minimum_rows or {2: 1000, 3: 1500, 4: 1500, 5: 1500}
    excluded_sha = {row['sha256'] for row in excluded}
    excluded_groups = {row['group_id'] for row in excluded}
    excluded_groups.update(table.loc[table.sha256.isin(excluded_sha), 'group_id'])
    selected, counts, tables, ordered = [], {}, {}, {}
    quotas = {b: groups_per_band for b in (2, 3, 4, 5)} if isinstance(groups_per_band, int) else groups_per_band
    for band in (2, 3, 4, 5):
        threshold = minimum_rows[band]
        eligible = table[(table.role == 'fit_dev') & (table.band == band) & (table.K >= threshold)
                         & ~table.group_id.isin(excluded_groups) & ~table.sha256.isin(excluded_sha)]
        groups = sorted(eligible.group_id.unique())
        rng.shuffle(groups)
        counts[str(band)] = len(groups)
        if len(groups) < quotas[band]:
            raise ValueError(f'Band {band} has {len(groups)} eligible groups; need {quotas[band]}')
        tables[band], ordered[band] = eligible, groups
    owners, assigned = {}, {}

    def assign(slot, seen):
        for group in ordered[slot[0]]:
            if group in seen:
                continue
            seen.add(group)
            if group not in owners or assign(owners[group], seen):
                owners[group], assigned[slot] = slot, group
                return True
        return False

    for band in sorted(quotas, key=lambda b: (counts[str(b)], b)):
        for i in range(quotas[band]):
            if not assign((band, i), set()):
                raise ValueError('Eligible song groups cannot satisfy all band quotas without reuse')
    for band in (2, 3, 4, 5):
        for i in range(quotas[band]):
            group = assigned[(band, i)]
            rows = tables[band][tables[band].group_id == group].sort_values('sha256')
            row = rows.iloc[int(rng.integers(len(rows)))].to_dict()
            selected.append({k: row[k] for k in ('sha256', 'group_id', 'band', 'K', 'file')})
    return dict(seed=seed, role='fit_dev', groups_per_band=groups_per_band,
                selection='seeded eligible-group shuffle; distinct-group matching; uniform chart within group',
                minimum_rows={str(b): minimum_rows[b] for b in (2, 3, 4, 5)},
                excluded_sha256=sorted(excluded_sha), excluded_group_id=sorted(excluded_groups),
                eligible_groups=counts, bos_seeds=[954, 955, 956], prefix_seed=954,
                max_rows=2560, charts=selected)


def write_panel(cache_root=CACHE, excluded_path=X0, output_path=PANEL, *, band3_min_rows=1500, band3_groups=25):
    table = pd.read_parquet(Path(cache_root) / 'index.parquet')
    excluded = json.loads(Path(excluded_path).read_text())['selected']
    panel = select_panel(table, excluded, groups_per_band={2: 25, 3: band3_groups, 4: 25, 5: 25},
                         minimum_rows={2: 1000, 3: band3_min_rows, 4: 1500, 5: 1500})
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(panel, indent=2) + '\n')
    return panel


def run_specs(panel, system, *, max_charts=None, max_rows=None, modes=None, seeds=None):
    """Prefix runs start at floor(K/3) and reach EOS; max_rows bounds smoke suffixes."""
    out = []
    cap = panel['max_rows'] if max_rows is None else max_rows
    for row in panel['charts'][:max_charts]:
        selected_seeds = panel['bos_seeds'] if system in ('b0', 'b1', 'b2', 'b3') else [954]
        for seed in selected_seeds:
            if seeds is None or seed in seeds:
                out.append(dict(row, mode='bos', start=0, stop=min(row['K'] + 1, cap), seed=seed))
        if not system.startswith('d0') and (seeds is None or panel['prefix_seed'] in seeds):
            start = row['K'] // 3
            stop = row['K'] + 1 if max_rows is None else min(row['K'] + 1, start + max_rows)
            out.append(dict(row, mode='prefix', start=start, stop=stop,
                            seed=panel['prefix_seed']))
    return [r for r in out if modes is None or r['mode'] in modes]


def checkpoint_path(path):
    path = Path(path)
    if path.is_dir():
        directory = path / 'checkpoints'
        return directory / json.loads((directory / 'latest.json').read_text())['path']
    return path


def load_model(path):
    data = torch.load(checkpoint_path(path), map_location='cpu', weights_only=False)
    model = R2Model(R2Config(**data['model_config'])).eval()
    model.load_state_dict(data['model'])
    return model


_WORKER = None


def _init_worker(checkpoint, cache_root, runs_dir, system, theta_table, score, score_path, statistics):
    global _WORKER
    torch.set_num_threads(2)
    scorer = None
    if system.startswith('d0'):
        from .block_select import load_scorer
        scorer = load_scorer(score, score_path, statistics)
    if system in ('b3', 'b3-oracle'):
        from .theta import THETA_FILE, ThetaTable
        theta_table = ThetaTable.load(theta_table or THETA_FILE)
    _WORKER = (load_model(checkpoint), Path(cache_root), Path(runs_dir), system, theta_table, scorer)


def _generate(spec):
    model, cache_root, runs_dir, system, theta_table, scorer = _WORKER
    chart = chart_from_cache(load_chart(cache_root / 'charts' / spec['file']))
    start, stop = spec['start'], spec['stop']
    record = {}
    kwargs = dict(prefix_actions=chart.actions[:start], prefix_gap=chart.gap[:start],
                  seed=spec['seed'], stop=stop)
    if system.startswith('d0'):
        from .block_select import select_blocks
        actions, gap, record = select_blocks(model, chart, scorer, band=spec['band'], **kwargs)
    else:
        mode = {'b3': 'prior', 'b3-unknown': 'unknown', 'b3-oracle': 'oracle'}.get(system, 'unknown')
        if system == 'b3' and spec['mode'] == 'prefix':  # the donor prior could contradict the real prefix
            from .theta import prefix_theta
            mode = prefix_theta(theta_table, chart.head_ms, chart.song_ms, chart.grid,
                                chart.actions[:start], chart.gap[:start])
        actions, gap = continue_chart(model, chart.head_ms, chart.song_ms, chart.grid, **kwargs,
                                      theta=mode, theta_table=theta_table,
                                      theta_source_sha256=spec['sha256'], theta_record=record)
        if not isinstance(mode, str):
            record['mode'] = 'prefix'
    stem = f"{spec['mode']}-{spec['sha256'][:16]}-{spec['seed']}"
    path = runs_dir / f'{stem}.npz'
    np.savez_compressed(path, actions=actions, gap_release_ms=gap, head_ms=chart.head_ms,
                        song_ms=chart.song_ms, start=start, stop=len(actions), seed=spec['seed'],
                        sha256=spec['sha256'], band=spec['band'], mode=spec['mode'],
                        cap_export_closure='midpoint_to_next_head' if len(actions) <= chart.K else 'none',
                        metadata=json.dumps(record))
    print(f'{system} {stem}: {len(actions) - start} decisions', flush=True)
    return dict(spec, path=str(path), metadata=record)


def run_panel(panel_path, checkpoint, system, *, cache_root=CACHE, output_dir=None, workers=2,
              max_charts=None, max_rows=None, modes=None, seeds=None, theta_table=None,
              score='envelope', score_path=None, statistics=None, reference_path=REFERENCE,
              measures=('m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7'), nll_windows=64):
    """Write a fresh runs directory, then score after generation workers have exited."""
    torch.set_num_threads(2)
    panel = json.loads(Path(panel_path).read_text())
    output_dir = Path(output_dir or ROOT / system)
    runs_dir = output_dir / 'runs'
    runs_dir.mkdir(parents=True, exist_ok=False)
    specs = run_specs(panel, system, max_charts=max_charts, max_rows=max_rows, modes=modes, seeds=seeds)
    initargs = (str(checkpoint_path(checkpoint)), str(cache_root), str(runs_dir), system, theta_table,
                score, score_path or str(reference_path), statistics)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=_init_worker, initargs=initargs) as pool:
        runs = list(pool.map(_generate, specs))
    (output_dir / 'runs.json').write_text(json.dumps(dict(system=system, checkpoint=str(checkpoint_path(checkpoint)),
                                                        panel=str(panel_path), score=score if system.startswith('d0') else None,
                                                        score_path=str(score_path) if score_path else None,
                                                        statistics=statistics, runs=runs), indent=2) + '\n')
    from .collapse_eval import evaluate_system
    return evaluate_system(panel_path, runs_dir, system, cache_root, output_dir / 'evaluation.json',
                           reference_path=reference_path, checkpoint=checkpoint_path(checkpoint),
                           measures=measures, nll_windows=nll_windows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    select = sub.add_parser('select')
    select.add_argument('--cache', default=str(CACHE))
    select.add_argument('--exclude', default=str(X0))
    select.add_argument('--out', default=str(PANEL))
    select.add_argument('--band3-min-rows', type=int, default=1500)
    select.add_argument('--band3-groups', type=int, default=25)
    run = sub.add_parser('run')
    run.add_argument('--system', choices=SYSTEMS, required=True)
    run.add_argument('--checkpoint', default=str(CHECKPOINT))
    run.add_argument('--panel', default=str(PANEL))
    run.add_argument('--cache', default=str(CACHE))
    run.add_argument('--out')
    run.add_argument('--workers', type=int, default=2)
    run.add_argument('--max-charts', type=int)
    run.add_argument('--max-rows', type=int)
    run.add_argument('--modes', help='Comma-separated bos,prefix')
    run.add_argument('--seeds', help='Comma-separated integer seeds')
    run.add_argument('--theta-table')
    run.add_argument('--score', choices=('phi', 'envelope'), default='envelope')
    run.add_argument('--score-path')
    run.add_argument('--statistics', default='pent,ng4,rep1,loop,nh,c3,c4,jack,fjack,bus4,held,lmax,hmax,lock,hlock')
    run.add_argument('--reference', default=str(REFERENCE))
    run.add_argument('--measures', default='m1,m2,m3,m4,m5,m6,m7')
    run.add_argument('--nll-windows', type=int, default=64)
    compare = sub.add_parser('compare')
    compare.add_argument('paths', nargs='+')
    compare.add_argument('--out', default=str(ROOT / 'comparison.md'))
    compare.add_argument('--measures', required=True)
    compare.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args(argv)
    if args.command == 'select':
        if args.band3_min_rows < 1 or args.band3_groups < 1:
            parser.error('Band 3 row minimum and group count must be positive')
        write_panel(args.cache, args.exclude, args.out,
                    band3_min_rows=args.band3_min_rows, band3_groups=args.band3_groups)
    elif args.command == 'run':
        if args.workers < 1 or (args.max_rows is not None and args.max_rows < 1):
            parser.error('workers and max-rows must be positive')
        run_panel(args.panel, args.checkpoint, args.system, cache_root=args.cache, output_dir=args.out,
                  workers=args.workers, max_charts=args.max_charts, max_rows=args.max_rows,
                  modes=args.modes.split(',') if args.modes else None,
                  seeds=[int(s) for s in args.seeds.split(',')] if args.seeds else None,
                  theta_table=args.theta_table, score=args.score, score_path=args.score_path,
                  statistics=args.statistics.split(','), reference_path=args.reference,
                  measures=args.measures.split(','), nll_windows=args.nll_windows)
    else:
        from .collapse_eval import compare_systems
        compare_systems(args.paths, args.out, measures=args.measures.split(','), bootstrap=args.bootstrap)


if __name__ == '__main__':
    main()
