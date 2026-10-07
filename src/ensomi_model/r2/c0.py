"""C0 source distributions, frozen-model controls, and ras-v1 proxy overhead.

``natural`` draws four scopes per chart and length, without replacing rejected
draws. Source and illustrative trajectories carry strain from chart start.
The compressed records retain exact scope endpoints and workloads for later
controller scans; ``natural`` never loads model weights. ``scan`` and
``controller`` retain reproducible scope continuations and full likelihoods;
``overhead`` measures byte-identical zero-bias sampling against plain sampling.
``screen`` builds blind comparison assets without collecting judgments.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import sys
import time

LENGTHS_SEC = (2, 4, 8, 16, 32, 60)
DRAWS_PER_CHART = 4
MIN_HEAD_ROWS = 8
PINNED_CHECKPOINT = 'artifacts/r2-runs/r2-phaseN-20261006/checkpoints/ckpt-0052000205.pt'
THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS')
METRICS = ('r', 'workload_per_sec', 'reference_per_sec', 'head_rows',
           'greedy_single_r', 'four_lane_chord_r', 'source_position_r',
           'source_position_workload', 'source_minus_greedy_r')


def configure_threads():
    """Set the CLI's numerical libraries to one CPU thread before measurement."""
    for name in THREAD_ENV:
        os.environ[name] = '1'
    import numpy as np
    import pyarrow as pa
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    return dict(python=platform.python_version(), numpy=np.__version__,
                pyarrow=pa.__version__, torch=torch.__version__, device='cpu',
                environment={name: os.environ[name] for name in THREAD_ENV},
                torch_threads=torch.get_num_threads(),
                torch_interop_threads=torch.get_num_interop_threads(),
                pyarrow_cpu_threads=pa.cpu_count(), pyarrow_io_threads=pa.io_thread_count())


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_fit_dev(cache):
    """Return SHA-sorted source records after checking stored split assignments.

    Only index metadata is read here. A disagreement with splits.json or the
    split hash rule raises before any source chart is opened. Cached stars
    enter only the existing common.band_of grouping.
    """
    import pyarrow.parquet as pq
    from .common import ContractError, band_of
    from .splits import verify_assignment

    cache = Path(cache)
    verify_assignment(cache / 'splits.json')
    groups = json.loads((cache / 'splits.json').read_text())['groups']
    table = pq.read_table(cache / 'index.parquet', use_threads=False)
    records = []
    for row in table.to_pylist():
        if groups.get(row['group_id']) != row['role']:
            raise ContractError(f"Cache split mismatch for {row['sha256']}")
        if row['role'] != 'fit_dev':
            continue
        star = row.get('star')
        row['cache_star_band'] = (str(band_of(float(star)))
                                  if star is not None and math.isfinite(float(star)) else 'missing')
        records.append(row)
    records.sort(key=lambda row: row['sha256'])
    if not records or len({row['sha256'] for row in records}) != len(records):
        raise ContractError('Expected nonempty, unique fit_dev chart records')
    return records


def approximate_prefixes(head_ms):
    """Return greedy-single and four-lane-chord workload prefixes, ignoring LNs.

    Both trajectories begin at rest at chart time zero and visit every head
    row. Greedy chooses the least decayed lane strain, ties by lane index.
    These illustrative bounds are not exact feasible LN-constrained bounds;
    greedy need not be a lower bound for every scope with source carry-in.
    """
    import numpy as np
    from .strain import StrainState

    greedy, ceiling = StrainState(), StrainState()
    greedy_prefix, ceiling_prefix = [0.0], [0.0]
    singles = np.eye(4, dtype=bool)
    chord = np.ones(4, dtype=bool)
    for t in head_ms:
        lane = int(np.argmin(greedy.decayed(t).lane))
        greedy, charge = greedy.append(t, singles[lane])
        greedy_prefix.append(greedy_prefix[-1] + charge)
        ceiling, charge = ceiling.append(t, chord)
        ceiling_prefix.append(ceiling_prefix[-1] + charge)
    return greedy_prefix, ceiling_prefix


def measure_source_chart(record, dec, rng):
    """Return 24 scope records, including exclusions, on a complete source chart.

    Starts are independent uniform draws on [0, song_ms - length_ms); an
    equal-length song has start zero. Short songs consume no RNG draws. No
    insufficient-head draw is replaced. All admitted metrics use [a_ms,b_ms).
    """
    from .strain import StrainTrace

    trace = StrainTrace(dec.head_ms, dec.song_ms).extend(dec.actions, dec.gap_release_ms)
    greedy, ceiling = approximate_prefixes(dec.head_ms)
    rows = []
    for length in LENGTHS_SEC:
        duration_ms = 1000.0 * length
        for draw in range(DRAWS_PER_CHART):
            row = dict(chart_sha256=record['sha256'], group_id=record['group_id'],
                       chart_file=record['file'], chart_file_sha256=record['file_sha256'],
                       cache_star_band=record['cache_star_band'], song_ms=dec.song_ms,
                       length_sec=length, draw=draw)
            if dec.song_ms < duration_ms:
                row['status'] = 'short_chart'
                rows.append(row)
                continue
            a = float(rng.uniform(0.0, dec.song_ms - duration_ms))
            b = min(a + duration_ms, dec.song_ms)
            i, j = trace.row_bounds(a, b)
            row.update(a_ms=a, b_ms=b, start_row=i, stop_row=j, head_rows=j - i)
            if j - i < MIN_HEAD_ROWS:
                row['status'] = 'insufficient_heads'
                rows.append(row)
                continue
            measured = trace.scope(a, b)
            low_w, high_w = greedy[j] - greedy[i], ceiling[j] - ceiling[i]
            low_r = math.sqrt(low_w / measured.reference)
            high_r = math.sqrt(high_w / measured.reference)
            r = measured.ratio
            row.update(status='admitted', workload=measured.workload,
                       reference=measured.reference, r=r,
                       workload_per_sec=measured.intensity,
                       reference_per_sec=measured.reference / measured.duration_sec,
                       greedy_single_workload=low_w, four_lane_chord_workload=high_w,
                       greedy_single_r=low_r, four_lane_chord_r=high_r,
                       source_position_r=(r - low_r) / (high_r - low_r),
                       source_position_workload=(measured.workload - low_w) / (high_w - low_w),
                       source_minus_greedy_r=r - low_r)
            rows.append(row)
    return rows


def describe(values):
    """Exact equal-scope quantiles using NumPy's linear interpolation method."""
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return dict(N=0)
    if not np.isfinite(values).all():
        raise ValueError('Summary input contains a nonfinite metric')
    quantiles = (5, 10, 25, 50, 75, 90, 95, 99)
    result = dict(N=len(values), min=float(values.min()), max=float(values.max()),
                  mean=float(values.mean()), std=float(values.std()))
    result.update({f'p{q}': float(v) for q, v in zip(
        quantiles, np.quantile(values, [q / 100 for q in quantiles], method='linear'))})
    return result


def summarize_scopes(rows):
    """Summarize one length/group, with all attempted and excluded draws counted."""
    admitted = [row for row in rows if row['status'] == 'admitted']
    counts = Counter(row['status'] for row in rows)
    metrics = {name: describe([row[name] for row in admitted]) for name in METRICS}
    result = dict(N=len(admitted), charts=len({row['chart_sha256'] for row in rows}),
                  charts_admitted=len({row['chart_sha256'] for row in admitted}),
                  charts_short=len({row['chart_sha256'] for row in rows
                                    if row['status'] == 'short_chart'}),
                  requested_scopes=len(rows), sampled_scopes=len(rows) - counts['short_chart'],
                  excluded_short_chart=counts['short_chart'],
                  excluded_insufficient_heads=counts['insufficient_heads'], metrics=metrics)
    for key, predicate in (
            ('r_below_one', lambda row: row['r'] < 1),
            ('source_below_greedy', lambda row: row['source_minus_greedy_r'] < 0),
            ('source_below_greedy_tolerance_1e_10', lambda row: row['source_minus_greedy_r'] < -1e-10),
            ('source_above_chord_tolerance_1e_10', lambda row: row['r'] > row['four_lane_chord_r'] + 1e-10)):
        count = sum(predicate(row) for row in admitted)
        result[key] = dict(count=count, fraction=count / len(admitted) if admitted else None)
    if admitted:
        r = metrics['r']
        relative_span = (r['p90'] - r['p10']) / r['p50']
        result['spread'] = dict(p90_over_p10=r['p90'] / r['p10'],
                               p90_minus_p10_over_p50=relative_span,
                               coefficient_of_variation=r['std'] / r['mean'],
                               nearly_constant=relative_span < 0.05)
        result['controller_targets'] = dict(low=r['p10'], middle=r['p50'], high=r['p90'],
                                            outlier=max(r['p99'], 1.5 * r['p50']))
        keys = ('chart_sha256', 'length_sec', 'draw', 'a_ms', 'b_ms', 'head_rows', 'r',
                'greedy_single_r', 'four_lane_chord_r', 'source_minus_greedy_r')
        result['largest_below_greedy_excursion'] = {
            key: min(admitted, key=lambda row: row['source_minus_greedy_r'])[key] for key in keys}
    return result


def write_report(path, summary):
    """Write the natural distribution tables and reproducible measurement contract."""
    lines = ['# R2 C0: natural ras-v1 distribution', '',
             f"Source fit_dev charts: {summary['charts_processed']}; "
             f"source head rows: {summary['head_rows_processed']:,}.", '',
             'Each chart contributes four independent uniform scopes per length over '
             '`[0, song_ms - L]`. Scopes with fewer than eight head rows are excluded '
             'without replacement. Strain and the cyclic reference carry in from chart start.', '',
             'Quantiles weight each admitted scope equally and use linear interpolation. '
             'Cached star bands are grouping metadata only.', '',
             '| L (s) | N | Short charts | <8 heads | r min | r p10 | r p50 | r p90 | r p99 | r max | r <1 |',
             '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for length, group in summary['by_length'].items():
        r = group['metrics']['r']
        if not group['N']:
            lines.append(f"| {length} | 0 | {group['charts_short']} | "
                         f"{group['excluded_insufficient_heads']} | - | - | - | - | - | - | - |")
            continue
        lines.append(f"| {length} | {group['N']} | {group['charts_short']} | "
                     f"{group['excluded_insufficient_heads']} | " +
                     ' | '.join(f"{r[name]:.5f}" for name in ('min', 'p10', 'p50', 'p90', 'p99', 'max')) +
                     f" | {group['r_below_one']['count']} |")
    spreads = [group['spread']['p90_minus_p10_over_p50']
               for group in summary['by_length'].values() if group['N']]
    if spreads:
        lines += ['', f'The normalized p10-to-p90 span ranges from {min(spreads):.3f} '
                  f'to {max(spreads):.3f} of p50 across lengths. '
                  'The complete distribution and controller targets are tabulated below.']
    lines += ['', '## Illustrative bounds', '',
              'Greedy singles choose the least current lane strain at every head row; '
              'the ceiling places a four-lane chord at every head row. Both ignore LN '
              'constraints and start at chart time zero. These are approximate bounds, '
              'not exact feasible constrained bounds. Source carry-in can place individual '
              'scopes below the greedy trajectory.', '',
              '`source_position_r = (r_source - r_greedy) / (r_chord - r_greedy)`. '
              'Positions are not clipped. Workload positions and all requested quantiles '
              'are retained in `part2.json`.', '',
              '| L (s) | Greedy r p50 | Chord r p50 | Position p10 | Position p50 | Position p90 | Below greedy |',
              '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for length, group in summary['by_length'].items():
        if not group['N']:
            continue
        metrics = group['metrics']
        lines.append(f"| {length} | {metrics['greedy_single_r']['p50']:.5f} | "
                     f"{metrics['four_lane_chord_r']['p50']:.5f} | " +
                     ' | '.join(f"{metrics['source_position_r'][q]:.5f}" for q in ('p10', 'p50', 'p90')) +
              f" | {group['source_below_greedy_tolerance_1e_10']['count']} |")
    lines += ['', '| L (s) | Illustrative metric | p5 | p10 | p25 | p50 | p75 | p90 | p95 |',
              '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for length, group in summary['by_length'].items():
        if not group['N']:
            continue
        for metric, label in (('greedy_single_r', 'Greedy r'),
                              ('four_lane_chord_r', 'Chord r'),
                              ('source_position_r', 'Source position in r')):
            stats = group['metrics'][metric]
            values = ' | '.join(f"{stats[f'p{q}']:.5f}" for q in (5, 10, 25, 50, 75, 90, 95))
            lines.append(f'| {length} | {label} | {values} |')
    lines += ['', '## Controller targets and spread', '',
              'Targets are natural p10, p50, p90, and `max(p99, 1.5 * p50)`. '
              'The descriptive near-constant screen is `(p90 - p10) / p50 < 0.05`; '
              'it is not a significance test.', '',
              '| L (s) | Low | Middle | High | Outlier | (p90 - p10) / p50 |',
              '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for length, group in summary['by_length'].items():
        if not group['N']:
            continue
        targets = group['controller_targets']
        lines.append(f"| {length} | " + ' | '.join(f'{v:.7f}' for v in targets.values()) +
                     f" | {group['spread']['p90_minus_p10_over_p50']:.5f} |")
    lines += ['', f"Near-constant stop triggered: `{summary['stop_nearly_constant']}`.", '',
              '## Natural quantiles by length and cache band', '',
              'Band `all` pools every admitted scope at that length. Numbered bands '
              'use the existing cache star through `common.band_of` solely for grouping. '
              '`WH/s` is the cyclic-reference workload per second.', '']
    for metric, title in (('r', 'r'), ('workload_per_sec', 'W/s'),
                          ('reference_per_sec', 'WH/s'), ('head_rows', 'Head rows')):
        lines += [f'### {title}', '',
                  '| L (s) | Cache band | N | p5 | p10 | p25 | p50 | p75 | p90 | p95 |',
                  '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for length, pooled in summary['by_length'].items():
            groups = [('all', pooled), *summary['by_length_and_cache_star_band'][length].items()]
            for band, group in groups:
                stats = group['metrics'][metric]
                values = ' | '.join(f"{stats[f'p{q}']:.5f}" if group['N'] else '-'
                                    for q in (5, 10, 25, 50, 75, 90, 95))
                lines.append(f"| {length} | {band} | {group['N']} | {values} |")
        lines.append('')
    lines += ['## Provenance', '', f"- Seed: `{summary['seed']}`; RNG: NumPy PCG64.",
              f"- Measured wall time: {summary['wall_seconds']:.3f} seconds.",
              f"- Pinned checkpoint (hash only): `{summary['provenance']['checkpoint']['path']}`.",
              f"- Checkpoint SHA-256: `{summary['provenance']['checkpoint']['sha256']}`.",
              '- Model sampling and training: none.',
              '- Records: `part2-scopes.jsonl.gz`; summary and band tables: `part2.json`.']
    for name, sha in summary['provenance']['changed_files_sha256'].items():
        lines.append(f'- Source SHA-256 `{name}`: `{sha}`.')
    lines += ['',
              '```sh', summary['command'], '```', '']
    Path(path).write_text('\n'.join(lines))


def run_natural(cache, output, seed, checkpoint, *, command=None):
    """Scan every fit_dev source chart and write fresh part2 artifacts.

    Existing part2 outputs are refused. Cache file hashes and exact replay are
    checked; any failure aborts, retaining a failure receipt and partial gzip
    records. A completed summary is written only after every chart succeeds.
    """
    started = time.perf_counter()
    environment = configure_threads()
    import numpy as np
    from .cache import load_chart
    from .common import ContractError
    from .strain import RAS_VERSION

    cache, output, checkpoint = Path(cache), Path(output), Path(checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    outputs = ('part2.json', 'part2-scopes.jsonl.gz', 'part2-provenance.json',
               'part2-failure.json', 'report.md')
    if any((output / name).exists() for name in outputs):
        raise FileExistsError('Use an output directory without existing part2 artifacts')
    source = Path(__file__).parent
    changed_files = [source / 'c0.py', source / 'README.md']
    dependencies = [source / name for name in ('strain.py', 'cache.py', 'common.py', 'splits.py', 'state.py')]
    file_key = lambda path: str(path.relative_to(Path.cwd())) if path.is_absolute() else str(path)
    provenance = dict(started_utc=datetime.now(timezone.utc).isoformat(), environment=environment,
                      checkpoint=dict(path=str(checkpoint), sha256=file_sha256(checkpoint), use='hash only'),
                      cache={name: file_sha256(cache / name) for name in ('index.parquet', 'splits.json')},
                      changed_files_sha256={file_key(path): file_sha256(path) for path in changed_files},
                      dependency_files_sha256={file_key(path): file_sha256(path) for path in dependencies})
    (output / 'part2-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    rows, processed, head_rows = [], 0, 0
    try:
        records = load_fit_dev(cache)
        rng = np.random.Generator(np.random.PCG64(seed))
        chart_digest = hashlib.sha256()
        with gzip.open(output / 'part2-scopes.jsonl.gz', 'wt', encoding='utf-8') as stream:
            for record in records:
                chart_path = cache / 'charts' / record['file']
                actual_sha = file_sha256(chart_path)
                if actual_sha != record['file_sha256']:
                    raise ContractError(f"Cached chart hash mismatch for {record['sha256']}")
                chart_digest.update(f"{record['sha256']} {actual_sha}\n".encode())
                dec = load_chart(chart_path)
                if dec.K != record['K'] or dec.song_ms != record['song_ms']:
                    raise ContractError(f"Cached chart metadata mismatch for {record['sha256']}")
                chart_rows = measure_source_chart(record, dec, rng)
                for row in chart_rows:
                    stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
                rows.extend(chart_rows)
                processed += 1
                head_rows += dec.K
                if processed % 100 == 0 or processed == len(records):
                    print(json.dumps(dict(charts=processed, total=len(records), head_rows=head_rows,
                                          wall_seconds=time.perf_counter() - started)), flush=True)
        by_length, by_band = {}, {}
        bands = sorted({record['cache_star_band'] for record in records})
        for length in LENGTHS_SEC:
            selected = [row for row in rows if row['length_sec'] == length]
            by_length[str(length)] = summarize_scopes(selected)
            by_band[str(length)] = {band: summarize_scopes(
                [row for row in selected if row['cache_star_band'] == band]) for band in bands}
        stop = any(group.get('spread', {}).get('nearly_constant', True) for group in by_length.values())
        provenance['source_charts_sha256'] = chart_digest.hexdigest()
        summary = dict(part=2, status='complete', metric=RAS_VERSION, seed=seed,
                       command=command, cache=str(cache), provenance=provenance,
                       charts_processed=processed, head_rows_processed=head_rows,
                       charts_by_cache_star_band=dict(Counter(r['cache_star_band'] for r in records)),
                       scope_records=len(rows), by_length=by_length, by_length_and_cache_star_band=by_band,
                       methods=dict(lengths_sec=LENGTHS_SEC, draws_per_chart_length=DRAWS_PER_CHART,
                                    minimum_head_rows=MIN_HEAD_ROWS, rng='numpy.random.PCG64',
                                    chart_order='sha256 ascending', scope_sampling='uniform song-time start',
                                    replace_exclusions=False, scope_end='half-open',
                                    reference='full-head-skeleton cycle 0,2,1,3 from chart start',
                                    strain_carry_in='entire chart prefix for all trajectories',
                                    quantiles='equal admitted scopes; numpy linear',
                                    approximate_bounds='LN constraints ignored; greedy is not a scope-wise floor',
                                    source_position='(source - greedy) / (chord - greedy), unclipped',
                                    near_constant_screen='any L: (p90-p10)/p50 < 0.05 or no admitted scopes',
                                    cache_star_use='common.band_of grouping only'),
                       stop_nearly_constant=stop, failures=[], skipped=[],
                       wall_seconds=time.perf_counter() - started)
        summary['provenance']['scope_records_sha256'] = file_sha256(output / 'part2-scopes.jsonl.gz')
        write_report(output / 'report.md', summary)
        (output / 'part2.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        return summary
    except Exception as exc:
        failure = dict(status='failed', error_type=type(exc).__name__, error=str(exc),
                       charts_processed=processed, head_rows_processed=head_rows,
                       wall_seconds=time.perf_counter() - started, provenance=provenance)
        (output / 'part2-failure.json').write_text(json.dumps(failure, indent=2) + '\n')
        raise


def tilted_moments(log_mass, costs, scale, eta):
    """Return (expected charge, KL(P_eta || P_0), probabilities) on finite support.

    Zero base masses remain zero. The computation is float64 and normalized in
    log space; costs are the exact 15 charges and scale is WH_scope/K_scope.
    """
    import numpy as np

    lm, costs = np.asarray(log_mass, dtype=np.float64), np.asarray(costs, dtype=np.float64)
    if lm.shape != (15,) or costs.shape != (15,) or not np.isfinite(costs).all() or scale <= 0:
        raise ValueError('Expected 15 mask masses/costs and a positive workload scale')
    supported = np.isfinite(lm)
    if not supported.any() or np.isnan(lm).any() or np.isposinf(lm).any():
        raise ValueError('Mask masses need nonempty finite support, with zeros represented by -inf')
    base = lm[supported]
    base = base - np.logaddexp.reduce(base)
    tilt = float(eta) * costs[supported] / scale
    tilted = base + tilt
    tilted -= np.logaddexp.reduce(tilted)
    p = np.exp(tilted)
    probabilities = np.zeros(15)
    probabilities[supported] = p
    return float(p @ costs[supported]), max(0.0, float(p @ (tilted - base))), probabilities


def solve_eta(log_mass, costs, scale, desired, kappa, eta_limit=8.0):
    """Match expected charge within independently computed +/- KL-feasible bounds.

    Returns eta, expectation, KL, endpoint limits, and signed expected-minus-
    desired mismatch. Unreachable requests use the closest feasible endpoint.
    Both KL bounds include eta=0, and all bisections retain the feasible side.
    """
    if not all(math.isfinite(x) for x in (desired, kappa, eta_limit)) or kappa < 0 or eta_limit <= 0:
        raise ValueError('Controller desired charge and bounds must be finite, with nonnegative KL')

    def moments(eta):
        return tilted_moments(log_mass, costs, scale, eta)

    bounds, reasons = [], []
    for sign in (-1, 1):
        endpoint = sign * eta_limit
        if moments(endpoint)[1] <= kappa:
            bounds.append(endpoint)
            reasons.append('eta')
            continue
        feasible, infeasible = 0.0, eta_limit
        for _ in range(48):
            middle = (feasible + infeasible) / 2
            if moments(sign * middle)[1] <= kappa:
                feasible = middle
            else:
                infeasible = middle
        bounds.append(sign * feasible)
        reasons.append('kl')
    low, high = bounds
    e_low, e_high = moments(low)[0], moments(high)[0]
    limit, side = None, None
    if desired < e_low:
        eta, limit, side = low, reasons[0], 'lower'
    elif desired > e_high:
        eta, limit, side = high, reasons[1], 'upper'
    elif desired == moments(0.0)[0]:
        eta = 0.0
    else:
        for _ in range(48):
            middle = (low + high) / 2
            if moments(middle)[0] < desired:
                low = middle
            else:
                high = middle
        eta = (low + high) / 2
    expected, kl, _ = moments(eta)
    return dict(eta=eta, expected_charge=expected, kl=kl,
                mismatch=expected - desired, limit=limit, limit_side=side,
                eta_lower=bounds[0], eta_upper=bounds[1],
                lower_limit=reasons[0], upper_limit=reasons[1],
                expected_lower=e_low, expected_upper=e_high)


def verify_controller_solver():
    """Independent finite-support, two-point analytic, and monotonicity checks."""
    import numpy as np

    costs = np.arange(1.0, 16.0)
    lm = np.full(15, -np.inf)
    lm[[0, 14]] = np.log([0.7, 0.3])
    # For two costs 1 and 15, an interior target fixes p=(target-1)/14.
    desired = 9.4
    p = (desired - 1) / 14
    exact_eta = (math.log(p / (1 - p)) - math.log(0.3 / 0.7)) / 14
    exact_kl = p * math.log(p / 0.3) + (1 - p) * math.log((1 - p) / 0.7)
    solved = solve_eta(lm, costs, 1, desired, 1.0)
    assert abs(solved['eta'] - exact_eta) < 1e-11
    assert abs(solved['kl'] - exact_kl) < 1e-11
    assert abs(solved['expected_charge'] - desired) < 1e-10
    expectations = [tilted_moments(lm, costs, 1, eta)[0] for eta in np.linspace(-8, 8, 321)]
    assert np.all(np.diff(expectations) >= -1e-12)
    rows = []
    for kappa in (0.0, 0.01, 0.25, 1.0):
        for target in (-100.0, 1.0, 5.2, 9.4, 15.0, 100.0):
            row = solve_eta(lm, costs, 1, target, kappa)
            assert abs(row['eta']) <= 8 and row['kl'] <= kappa + 2e-12
            if row['limit_side'] == 'lower':
                assert row['expected_charge'] >= target - 1e-10
            if row['limit_side'] == 'upper':
                assert row['expected_charge'] <= target + 1e-10
            rows.append(dict(kappa=kappa, target=target, **row))
    singleton = np.full(15, -np.inf)
    singleton[4] = 0.0
    for target in (-1, 5, 100):
        row = solve_eta(singleton, costs, 1, target, 0.25)
        assert row['expected_charge'] == 5 and row['kl'] == 0
    return dict(status='passed', analytic_eta=exact_eta, analytic_kl=exact_kl,
                monotone_grid_points=len(expectations), constraint_cases=rows,
                singleton_cases=3, zero_probability_support=True)


class MeasurementBias:
    """Fixed/controller bias with a replay-backed trace and per-decision audit.

    Call ``finish`` with returned sampling arrays to charge the final row.
    Audit probabilities are computed from original phase-N mask masses, before
    changing action logits. Each instance owns one generation trajectory.
    """

    def __init__(self, head_ms, song_ms, a, b, *, eta=0.0, target=None, kappa=None):
        from .strain import StrainTrace

        self.trace = StrainTrace(head_ms, song_ms)
        self.start, self.stop = self.trace.row_bounds(a, b)
        self.reference = self.trace.reference_rows(self.start, self.stop)
        self.scale = self.reference / (self.stop - self.start)
        self.eta, self.target, self.kappa = eta, target, kappa
        self.rows = []

    def __call__(self, k, chart, held, logp):
        import numpy as np
        from .strain import head_mask_log_probs

        self.trace.extend(chart.actions, chart.gap)
        if k != self.trace.replay.k or not np.array_equal(held, self.trace.replay.held):
            raise ValueError('C0 replay and sampler held states disagree')
        if not self.start <= k < self.stop:
            return np.zeros(15)
        costs = self.trace.state.mask_costs(chart.time(k))
        lm = head_mask_log_probs(logp.double(), held).detach().cpu().numpy()
        row = dict(k=k, time_ms=chart.time(k), supported_masks=int(np.isfinite(lm).sum()))
        if self.target is None:
            eta = self.eta
            expected, kl, _ = tilted_moments(lm, costs, self.scale, eta)
            row.update(eta=eta, expected_charge=expected, kl=kl, limit=None, limit_side=None)
        else:
            committed = self.trace.workload_rows(self.start, k)
            budget = self.target ** 2 * self.reference - committed
            remaining_reference = self.trace.reference_rows(k, self.stop)
            row_reference = self.trace.reference_rows(k, k + 1)
            desired = budget * row_reference / remaining_reference
            solution = solve_eta(lm, costs, self.scale, desired, self.kappa)
            eta = solution['eta']
            row.update(committed_workload=committed, remaining_budget=budget,
                       remaining_reference=remaining_reference, row_reference=row_reference,
                       desired_charge=desired, **solution)
        row.update(mask_costs=costs.tolist(),
                   natural_mask_log_probs=[float(x) if np.isfinite(x) else None for x in lm])
        self.rows.append(row)
        return eta * costs / self.scale if eta else np.zeros(15)

    def finish(self, actions, gap):
        self.trace.extend(actions, gap)
        for row in self.rows:
            row['realized_charge'] = self.trace.charges[row['k']]


def load_measurement_model(checkpoint):
    """Load the complete trained phase-N state by the trainer's reference loader."""
    import torch
    from .train_ce import reference_model

    data = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if data['config']['phase'] != 'natural':
        raise ValueError('C0 requires a phase-N checkpoint')
    dtype_name = data['config'].get('dtype', 'float32')
    model = reference_model(data, 'cpu', getattr(torch, dtype_name))
    return model, dict(path=str(checkpoint), sha256=file_sha256(checkpoint),
                       exposures=int(data['state']['exposures']), model_config=data['model_config'],
                       training_config=data['config'], loading_device='cpu', loading_dtype=dtype_name,
                       loader='train_ce.reference_model: strict complete state_dict; eval; frozen parameters',
                       conditioning_track=[])


def generation_anchors(cache, output, seed):
    """Persist 24 distinct SHA-shuffled fit_dev charts and common start times.

    Each start is uniform on the feasible subset of [20%,70%] song time for
    a 16-second scope and at least eight heads in the first two seconds. Since
    lengths share their start, this guarantees eight heads for every length.
    Chart rejection and all RNG draws are recorded. Cached star metadata is
    neither read nor used in selection.
    """
    import numpy as np
    import pyarrow.parquet as pq
    from .cache import load_chart
    from .splits import verify_assignment

    path = Path(output) / 'part3-anchors.json'
    if path.exists():
        result = json.loads(path.read_text())
        if result['seed'] != seed or result['cache'] != str(cache):
            raise ValueError('Existing C0 anchors have a different seed/cache')
        return result
    verify_assignment(Path(cache) / 'splits.json')
    groups = json.loads((Path(cache) / 'splits.json').read_text())['groups']
    records = pq.read_table(Path(cache) / 'index.parquet', columns=[
        'sha256', 'group_id', 'role', 'file', 'file_sha256'], use_threads=False).to_pylist()
    if any(groups.get(r['group_id']) != r['role'] for r in records):
        raise ValueError('C0 cache split mismatch')
    records = sorted((r for r in records if r['role'] == 'fit_dev'), key=lambda r: r['sha256'])
    rng = np.random.Generator(np.random.PCG64(seed))
    anchors, rejected = [], []
    for ix in rng.permutation(len(records)):
        r = records[int(ix)]
        dec = load_chart(Path(cache) / 'charts' / r['file'])
        lo, hi = 0.2 * dec.song_ms, min(0.7 * dec.song_ms, dec.song_ms - 16000)
        # Exact intervals where a half-open two-second window contains >=8 heads.
        intervals = []
        for j in range(len(dec.head_ms) - 7):
            left, right = max(lo, float(dec.head_ms[j + 7]) - 2000), min(hi, float(dec.head_ms[j]))
            if right > left:
                if intervals and left <= intervals[-1][1]:
                    intervals[-1][1] = max(intervals[-1][1], right)
                else:
                    intervals.append([left, right])
        if not intervals:
            rejected.append(dict(chart_sha256=r['sha256'], reason='no_feasible_common_start'))
            continue
        lengths = np.array([v - u for u, v in intervals])
        draw = float(rng.uniform(0, lengths.sum()))
        j = min(int(np.searchsorted(np.cumsum(lengths), draw, side='right')), len(intervals) - 1)
        a = intervals[j][0] + draw - float(lengths[:j].sum())
        actual = file_sha256(Path(cache) / 'charts' / r['file'])
        if actual != r['file_sha256']:
            raise ValueError(f"C0 cached chart hash mismatch: {r['sha256']}")
        anchors.append(dict(anchor=len(anchors), chart_sha256=r['sha256'], group_id=r['group_id'],
                            chart_file=r['file'], chart_file_sha256=actual, a_ms=a,
                            song_ms=dec.song_ms, feasible_start_ms=float(lengths.sum()),
                            uniform_feasible_draw_ms=draw, role='scan' if len(anchors) < 16 else 'controller'))
        if len(anchors) == 24:
            break
    if len(anchors) != 24:
        raise ValueError('Insufficient eligible distinct fit_dev charts for C0')
    result = dict(seed=seed, cache=str(cache), anchors=anchors, rejected=rejected,
                  rule='uniform feasible start in 20%-70% song time; L16 fits; >=8 heads in L2',
                  rng='numpy PCG64: SHA-sorted chart permutation then one uniform draw per admitted chart')
    path.write_text(json.dumps(result, indent=2) + '\n')
    return result


def load_trajectory(path):
    """Return (Chart, metadata, original prefix actions, original prefix gaps).

    NPZ contains the full source head skeleton and beat grid, generated actions
    through stop_row-1, and the exact source prefix. It deliberately ends before
    scope exit; callers exporting a complete .osu must supply a legal tail/EOS.
    """
    import numpy as np
    from .common import GridArrays, grid_from_arrays
    from .features import Chart

    with np.load(path, allow_pickle=False) as z:
        chart = Chart(z['head_ms'], float(z['song_ms']),
                      GridArrays.from_grid(grid_from_arrays(z['grid_segments'], z['grid_bars'])),
                      z['actions'], z['gap'])
        return chart, json.loads(str(z['metadata'])), z['prefix_actions'], z['prefix_gap']


def measure_generated_scope(model, dec, spec, destination):
    """Generate one scope, score complete decisions, and retain replayable arrays."""
    import numpy as np
    import torch
    from .cache import chart_grid
    from .common import GridArrays
    from .features import Chart
    from .sampling import continue_chart
    from .strain import heads_from_codes

    started = time.perf_counter()
    a, b = spec['a_ms'], spec['a_ms'] + 1000 * spec['length_sec']
    start, stop = (int(k) for k in np.searchsorted(dec.head_ms, (a, b), side='left'))
    if stop - start < MIN_HEAD_ROWS:
        raise ValueError('Frozen C0 anchor has fewer than eight scope heads')
    tilt = MeasurementBias(dec.head_ms, dec.song_ms, a, b, eta=spec.get('eta', 0.0),
                           target=spec.get('target'), kappa=spec.get('kappa'))
    grid = GridArrays.from_grid(chart_grid(dec))
    with torch.inference_mode():
        actions, gap = continue_chart(model, dec.head_ms, dec.song_ms, grid,
                                      dec.actions[:start], dec.gap_release_ms[:start], track=(),
                                      seed=spec['sampling_seed'], stop=stop, head_mask_bias=tilt)
        assert np.array_equal(actions[:start], dec.actions[:start])
        assert np.array_equal(gap[:start], dec.gap_release_ms[:start], equal_nan=True)
        tilt.finish(actions, gap)
        chart = Chart(dec.head_ms, dec.song_ms, grid, actions, gap)
        action_lp, release_lp = [], []
        for j in range(start, stop, 128):
            scored = model.window(chart, j, min(stop, j + 128), track=())
            action_lp.extend(scored.action.cpu().double().tolist())
            release_lp.extend(scored.release.cpu().double().tolist())
    if not (np.isfinite(action_lp).all() and np.isfinite(release_lp).all()):
        raise ValueError('C0 generated likelihood is nonfinite')
    held = chart.derived().held[:stop]
    heads = heads_from_codes(actions, held)
    q = heads[start:stop]
    sizes = q.sum(axis=1)
    ln = np.where(held[start:stop], actions[start:stop] == 4, actions[start:stop] == 2) & q
    pairs = q[1:] & q[:-1]
    measured = tilt.trace.scope(a, b)
    halfway = int(np.searchsorted(dec.head_ms, (a + b) / 2, side='left'))
    first = tilt.trace.workload_rows(start, halfway)
    second = tilt.trace.workload_rows(halfway, stop)
    for i, row in enumerate(tilt.rows):
        row.update(action_log_likelihood=action_lp[i], release_log_likelihood=release_lp[i],
                   head_mask=int(q[i].astype(int) @ (1 << np.arange(4))))
    result = dict(**spec, b_ms=b, start_row=start, stop_row=stop, head_rows=stop - start,
                  r=measured.ratio, workload=measured.workload, reference=measured.reference,
                  workload_per_sec=measured.intensity, chord_size_counts=np.bincount(sizes, minlength=5)[1:].tolist(),
                  heads=int(sizes.sum()), ln_heads=int(ln.sum()), ln_share=float(ln.sum() / sizes.sum()),
                  same_lane_repeat_count=int(pairs.sum()), repeat_denominator=int(q[:-1].sum()),
                  same_lane_repeat_rate=float(pairs.sum() / q[:-1].sum()),
                  kl_mean=float(np.mean([r['kl'] for r in tilt.rows])),
                  kl_max=max(r['kl'] for r in tilt.rows),
                  singleton_supported_mask_rate=float(np.mean([r['supported_masks'] == 1 for r in tilt.rows])),
                  action_log_likelihood_per_decision=float(np.mean(action_lp)),
                  release_log_likelihood_per_decision=float(np.mean(release_lp)),
                  full_log_likelihood_per_decision=float(np.mean(np.array(action_lp) + release_lp)),
                  first_half_workload=first, second_half_workload=second,
                  first_half_workload_share=first / measured.workload,
                  second_half_workload_share=second / measured.workload,
                  saturation_counts=dict(Counter(r['limit'] or 'none' for r in tilt.rows)),
                  saturation_side_counts=dict(Counter(r['limit_side'] or 'none' for r in tilt.rows)),
                  eta_min=min(r['eta'] for r in tilt.rows), eta_max=max(r['eta'] for r in tilt.rows))
    if spec.get('target') is not None:
        result['log_ratio_error'] = math.log(result['r'] / spec['target'])
        result['terminal_budget'] = spec['target'] ** 2 * result['reference'] - result['workload']
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    trajectory = destination.with_suffix('.npz')
    np.savez_compressed(trajectory, head_ms=dec.head_ms, song_ms=dec.song_ms,
                        grid_segments=dec.grid_segments, grid_bars=dec.grid_bars,
                        actions=actions, gap=gap, prefix_actions=dec.actions[:start],
                        prefix_gap=dec.gap_release_ms[:start], metadata=json.dumps(result))
    with gzip.open(destination.with_suffix('.decisions.jsonl.gz'), 'wt', encoding='utf-8') as stream:
        for row in tilt.rows:
            stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
    result['trajectory'] = str(trajectory)
    result['decisions'] = str(destination.with_suffix('.decisions.jsonl.gz'))
    result['wall_seconds'] = time.perf_counter() - started
    destination.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def measurement_specs(part, anchors, targets, seed, seeds):
    """Deterministically shuffle the full two-seed design, then filter seed count."""
    import numpy as np

    rows = []
    selected = anchors[:16] if part == 'part3a' else anchors[16:]
    for anchor in selected:
        for length in ((4, 16) if part == 'part3a' else (2, 4, 8, 16)):
            for seed_index in range(2):
                shared = dict(anchor=anchor['anchor'], chart_sha256=anchor['chart_sha256'],
                              chart_file=anchor['chart_file'], a_ms=anchor['a_ms'],
                              length_sec=length, seed_index=seed_index,
                              sampling_seed=seed + 10000 + 100 * anchor['anchor'] + seed_index)
                treatments = ([dict(eta=eta, treatment='fixed') for eta in (-4, -2, -1, 0, 1, 2, 4)]
                              if part == 'part3a' else
                              [dict(target=value, target_name=name, kappa=kappa, treatment='controller')
                               for name, value in targets[str(length)].items() for kappa in (0.25, 1.0)]
                              + [dict(eta=0.0, treatment='reference')])
                for treatment in treatments:
                    spec = dict(**shared, **treatment)
                    spec['id'] = f'{part}-{len(rows):04d}'
                    rows.append(spec)
    order = np.random.Generator(np.random.PCG64(seed + (31 if part == 'part3a' else 32))).permutation(len(rows))
    return [rows[int(i)] for i in order if rows[int(i)]['seed_index'] < seeds]


def summarize_generated(rows, seed):
    """Equal-scope summaries and chart-cluster bootstrap intervals of median r."""
    import numpy as np

    if not rows:
        return dict(N=0)
    metrics = ('r', 'workload_per_sec', 'same_lane_repeat_rate', 'ln_share', 'kl_mean',
               'singleton_supported_mask_rate', 'action_log_likelihood_per_decision',
               'release_log_likelihood_per_decision', 'full_log_likelihood_per_decision',
               'first_half_workload_share', 'second_half_workload_share')
    result = dict(N=len(rows), charts=len({r['anchor'] for r in rows}),
                  metrics={k: describe([r[k] for r in rows]) for k in metrics})
    chords = np.sum([r['chord_size_counts'] for r in rows], axis=0)
    result['chord_size_distribution'] = (chords / chords.sum()).tolist()
    result['pooled_repeat_rate'] = sum(r['same_lane_repeat_count'] for r in rows) / sum(r['repeat_denominator'] for r in rows)
    result['pooled_ln_share'] = sum(r['ln_heads'] for r in rows) / sum(r['heads'] for r in rows)
    result['pooled_per_decision_kl'] = sum(r['kl_mean'] * r['head_rows'] for r in rows) / sum(r['head_rows'] for r in rows)
    result['saturation_counts'] = dict(sum((Counter(r['saturation_counts']) for r in rows), Counter()))
    result['saturation_fraction'] = 1 - result['saturation_counts'].get('none', 0) / sum(r['head_rows'] for r in rows)
    result['saturation_side_counts'] = dict(sum((Counter(r['saturation_side_counts']) for r in rows), Counter()))
    clusters = sorted({r['anchor'] for r in rows})
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = []
    for _ in range(1000):
        sample = rng.choice(clusters, len(clusters), replace=True)
        draws.append(float(np.median([r['r'] for c in sample for r in rows if r['anchor'] == c])))
    result['median_r_cluster_bootstrap_95pct'] = np.quantile(draws, [0.025, 0.975]).tolist()
    if 'log_ratio_error' in rows[0]:
        errors = np.array([r['log_ratio_error'] for r in rows])
        requested = {r['target'] for r in rows}
        result.update(requested_r=rows[0]['target'] if len(requested) == 1 else None,
                      log_ratio_error=describe(errors),
                      within_log_005=float(np.mean(np.abs(errors) <= 0.05)),
                      within_log_010=float(np.mean(np.abs(errors) <= 0.10)))
        refs = [r['reference_second_half_share_delta'] for r in rows]
        result['second_half_share_minus_reference'] = describe(refs)
        for metric in ('full_log_likelihood_per_decision', 'ln_share', 'same_lane_repeat_rate'):
            result[metric + '_minus_reference'] = describe([r[metric + '_minus_reference'] for r in rows])
    return result


def model_provenance(checkpoint, environment, output, part):
    """Snapshot exact measurement dependencies before a model-backed command."""
    import shutil

    source = Path(__file__).parent
    paths = [source / name for name in ('c0.py', 'strain.py', 'sampling.py', 'model.py', 'train_ce.py',
                                       'features.py', 'state.py', 'cache.py', 'common.py', 'candidates.py')]
    hashes = {str(p.relative_to(Path.cwd())): file_sha256(p) for p in paths}
    snapshot = Path(output) / f'{part}-source'
    snapshot.mkdir(exist_ok=True)
    for p in paths:
        copied = snapshot / p.name
        if copied.exists() and file_sha256(copied) != hashes[str(p.relative_to(Path.cwd()))]:
            raise ValueError(f'{part} source changed after measurement started: {p.name}')
        if not copied.exists():
            shutil.copyfile(p, copied)
    return dict(started_utc=datetime.now(timezone.utc).isoformat(), environment=environment,
                code_sha256=hashes, source_snapshot=str(snapshot),
                checkpoint_sha256=file_sha256(checkpoint))


def run_model_part(part, cache, output, seed, checkpoint, *, command, pilot_only=False, seeds=2):
    """Run/resume fixed scopes; first 20 full measurements always form a timing pilot.

    Existing scope receipts are reused only with identical design, code snapshot,
    and checkpoint identity. Full runs require both pilots and freeze one versus
    two seeds from their combined three-hour projection before further sampling.
    """
    started = time.perf_counter()
    environment = configure_threads()
    from .cache import load_chart

    cache, output, checkpoint = Path(cache), Path(output), Path(checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    anchors = generation_anchors(cache, output, seed)['anchors']
    natural = json.loads((output / 'part2.json').read_text())
    targets = {length: group['controller_targets'] for length, group in natural['by_length'].items()}
    specs = measurement_specs(part, anchors, targets, seed, seeds)
    provenance = model_provenance(checkpoint, environment, output, part)
    model, loading = load_measurement_model(checkpoint)
    provenance['checkpoint'] = loading
    manifest_path = output / f'{part}-manifest.json'
    manifest = dict(seed=seed, checkpoint=loading, targets=targets, specs=measurement_specs(part, anchors, targets, seed, 2),
                    code_sha256=provenance['code_sha256'], command=command,
                    part2_sha256=file_sha256(output / 'part2.json'))
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        if any(prior[k] != manifest[k] for k in ('seed', 'checkpoint', 'targets', 'specs', 'code_sha256', 'part2_sha256')):
            raise ValueError('Existing C0 manifest differs from the requested experiment')
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    if part == 'part3b':
        (output / 'controller-solver-checks.json').write_text(json.dumps(verify_controller_solver(), indent=2) + '\n')
    if not pilot_only:
        pilots = [json.loads((output / f'{p}-pilot.json').read_text()) for p in ('part3a', 'part3b')]
        projection = sum(p['projected_full_design_seconds'] for p in pilots)
        expected_seeds = 1 if projection > 3 * 3600 else 2
        if seeds != expected_seeds:
            raise ValueError(f'Combined pilot projection requires {expected_seeds} seeds, got {seeds}')
        plan = dict(projected_two_seed_seconds=projection, seeds=seeds,
                    reason='combined projection >3 hours' if seeds == 1 else 'combined projection <=3 hours',
                    scan_scopes=224 * seeds, controller_scopes=256 * seeds, reference_scopes=32 * seeds)
        (output / 'part3-timing-plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    rows, charts = [], {}
    work = specs[:20] if pilot_only else specs
    for index, spec in enumerate(work):
        receipt = output / 'trajectories' / (spec['id'] + '.json')
        if receipt.exists():
            row = json.loads(receipt.read_text())
            if any(row[k] != value for k, value in spec.items()):
                raise ValueError(f"Existing scope spec mismatch: {spec['id']}")
        else:
            sha = spec['chart_sha256']
            if sha not in charts:
                path = cache / 'charts' / spec['chart_file']
                anchor = next(a for a in anchors if a['chart_sha256'] == sha)
                if file_sha256(path) != anchor['chart_file_sha256']:
                    raise ValueError('Anchor chart content changed')
                charts[sha] = load_chart(path)
            row = measure_generated_scope(model, charts[sha], spec, receipt)
        rows.append(row)
        if (index + 1) % 10 == 0 or index + 1 == len(work):
            print(json.dumps(dict(part=part, complete=index + 1, total=len(work),
                                  wall_seconds=time.perf_counter() - started)), flush=True)
    if pilot_only:
        pilot = dict(part=part, status='pilot_complete', seed=seed, scope_ids=[r['id'] for r in rows],
                     N=len(rows), scope_seconds=sum(r['wall_seconds'] for r in rows),
                     mean_scope_seconds=sum(r['wall_seconds'] for r in rows) / len(rows),
                     full_design_scopes=len(measurement_specs(part, anchors, targets, seed, 2)),
                     projected_full_design_seconds=sum(r['wall_seconds'] for r in rows) / len(rows) * len(specs),
                     timing_boundary='trace/reference prep, generation, full model likelihood, metrics, NPZ and audit writes',
                     provenance=provenance, command=command, wall_seconds=time.perf_counter() - started)
        (output / f'{part}-pilot.json').write_text(json.dumps(pilot, indent=2) + '\n')
        return pilot
    groups, wrong_direction = {}, []
    if part == 'part3a':
        for length in (4, 16):
            groups[str(length)] = {str(eta): summarize_generated(
                [r for r in rows if r['length_sec'] == length and r['eta'] == eta], seed + length)
                for eta in (-4, -2, -1, 0, 1, 2, 4)}
    else:
        refs = {(r['anchor'], r['length_sec'], r['sampling_seed']): r
                for r in rows if r['treatment'] == 'reference'}
        for row in rows:
            if row['treatment'] == 'controller':
                ref = refs[(row['anchor'], row['length_sec'], row['sampling_seed'])]
                row['reference_second_half_share_delta'] = row['second_half_workload_share'] - ref['second_half_workload_share']
                for metric in ('full_log_likelihood_per_decision', 'ln_share', 'same_lane_repeat_rate'):
                    row[metric + '_minus_reference'] = row[metric] - ref[metric]
        for length in (2, 4, 8, 16):
            groups[str(length)] = {}
            for kappa in (0.25, 1.0):
                groups[str(length)][str(kappa)] = {name: summarize_generated(
                    [r for r in rows if r['length_sec'] == length and r.get('kappa') == kappa and r['target_name'] == name], seed + length)
                    for name in targets[str(length)]}
                medians = [groups[str(length)][str(kappa)][name]['metrics']['r']['p50']
                           for name in ('low', 'middle', 'high', 'outlier')]
                if any(y < x for x, y in zip(medians, medians[1:])):
                    wrong_direction.append(dict(length_sec=length, kappa=kappa, ordered_medians=medians))
            groups[str(length)]['reference'] = summarize_generated(
                [r for r in refs.values() if r['length_sec'] == length], seed + length)
    summary = dict(part=part, status='complete', N=len(rows), seeds=seeds, seed=seed,
                   controller_scopes=sum(r['treatment'] == 'controller' for r in rows),
                   reference_scopes=sum(r['treatment'] == 'reference' for r in rows),
                   by_length=groups, wrong_direction=wrong_direction, stop_wrong_direction=bool(wrong_direction),
                   scope_wall_seconds=sum(r['wall_seconds'] for r in rows),
                   invocation_wall_seconds=time.perf_counter() - started,
                   wall_seconds=sum(r['wall_seconds'] for r in rows),
                   provenance=provenance, command=command, skipped=[], failures=[],
                   uncertainty='1000 chart-cluster bootstrap resamples; 95% percentile CI for median r; scope quantile spread',
                   repeat_rate='overlapping heads on adjacent in-scope rows / heads on all in-scope rows except the last',
                   singleton_supported_mask_rate='fraction of decisions with exactly one finite-probability head mask',
                   likelihood='empty-track R2Model.window: action + logmeanexp of forward/reverse release-pointer likelihoods',
                   trajectory_api='load_trajectory(path) -> Chart, metadata, source_prefix_actions, source_prefix_gap; tail/EOS unfinished',
                   records=f'{part}-scopes.jsonl.gz')
    summary['pooled_by_length'] = {
        str(length): summarize_generated([r for r in rows if r['length_sec'] == length
                                         and r['treatment'] != 'reference'], seed + length)
        for length in ((4, 16) if part == 'part3a' else (2, 4, 8, 16))}
    summary['aggregate_weighting'] = (
        'metric quantiles/means weight scopes equally; pooled KL weights head decisions; '
        'chord distribution weights head rows; pooled LN share weights individual heads; '
        'pooled repeat rate weights predecessor heads; per-length controller attainment pools '
        'all four target names and both kappas equally by scope')
    with gzip.open(output / summary['records'], 'wt', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
    (output / f'{part}.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    plot_model_part(output, summary)
    update_model_report(output)
    return summary


def plot_model_part(output, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    part = summary['part']
    lengths = list(summary['by_length'])
    fig, axes = plt.subplots(1, len(lengths), figsize=(4 * len(lengths), 3.4), squeeze=False)
    for length, ax in zip(lengths, axes[0]):
        group = summary['by_length'][length]
        if part == 'part3a':
            xs = [float(x) for x in group]
            ys = [g['metrics']['r']['p50'] for g in group.values()]
            ax.plot(xs, ys, marker='o')
            ax.fill_between(xs, [g['metrics']['r']['p10'] for g in group.values()],
                            [g['metrics']['r']['p90'] for g in group.values()], alpha=0.2)
            ax.set_xlabel('eta')
        else:
            for kappa in ('0.25', '1.0'):
                data = [group[kappa][name] for name in ('low', 'middle', 'high', 'outlier')]
                xs = [g['requested_r'] for g in data]
                ax.plot(xs, [g['metrics']['r']['p50'] for g in data], marker='o', label=f'KL <= {kappa}')
            ax.plot([min(xs), max(xs)], [min(xs), max(xs)], ':', color='gray')
            ax.set_xlabel('Requested r')
            ax.legend(fontsize=8)
        ax.set_ylabel('Achieved r (median)')
        ax.set_title(f'L = {length} s')
    fig.tight_layout()
    fig.savefig(Path(output) / f'{part}-response.png', dpi=130)
    plt.close(fig)


class ZeroCostBias:
    """Overhead-only trace plus all 15 costs; no probabilities or telemetry."""

    def __init__(self, head_ms, song_ms):
        from .strain import StrainTrace
        self.trace = StrainTrace(head_ms, song_ms)

    def __call__(self, k, chart, held, logp):
        import numpy as np
        self.trace.extend(chart.actions, chart.gap)
        if k < chart.K:
            self.trace.state.mask_costs(chart.time(k))
        return np.zeros(15)


def run_overhead(cache, output, seed, checkpoint, *, command):
    """Five alternating plain/zero-cost pairs on four common fit_dev anchors.

    Timers include source-prefix model preparation in both modes and all trace/
    reference preparation in the callback mode. Model load, chart I/O, audit,
    and likelihood scoring are outside timers. Actions and NaN-bearing gaps
    must match byte for byte for every pair. No telemetry runs in the callback.
    """
    import numpy as np
    environment = configure_threads()
    from .cache import chart_grid, load_chart
    from .common import GridArrays
    from .sampling import continue_chart

    started = time.perf_counter()
    output = Path(output)
    provenance = model_provenance(checkpoint, environment, output, 'overhead')
    model, loading = load_measurement_model(checkpoint)
    provenance['checkpoint'] = loading
    anchors = generation_anchors(cache, output, seed)['anchors'][:4]
    rows = []
    for anchor in anchors:
        dec = load_chart(Path(cache) / 'charts' / anchor['chart_file'])
        grid = GridArrays.from_grid(chart_grid(dec))
        a, b = anchor['a_ms'], anchor['a_ms'] + 4000
        start, stop = (int(k) for k in np.searchsorted(dec.head_ms, (a, b), side='left'))
        kwargs = dict(prefix_actions=dec.actions[:start], prefix_gap=dec.gap_release_ms[:start],
                      track=(), seed=seed + 10000 + 100 * anchor['anchor'], stop=stop)
        continue_chart(model, dec.head_ms, dec.song_ms, grid, **kwargs)
        for repeat in range(5):
            values, outputs = {}, {}
            for mode in (('plain', 'proxy') if repeat % 2 == 0 else ('proxy', 'plain')):
                t0 = time.perf_counter()
                hook = ZeroCostBias(dec.head_ms, dec.song_ms) if mode == 'proxy' else None
                actions, gap = continue_chart(model, dec.head_ms, dec.song_ms, grid, head_mask_bias=hook, **kwargs)
                if hook is not None:
                    hook.trace.extend(actions, gap)
                values[mode] = time.perf_counter() - t0
                outputs[mode] = (actions, gap)
            assert all(x.dtype == y.dtype and x.tobytes() == y.tobytes()
                       for x, y in zip(outputs['plain'], outputs['proxy']))
            rows.append(dict(anchor=anchor['anchor'], chart_sha256=anchor['chart_sha256'],
                             length_sec=4, repeat=repeat, first='plain' if repeat % 2 == 0 else 'proxy',
                             start_row=start, stop_row=stop, a_ms=a, b_ms=b, sampling_seed=kwargs['seed'],
                             plain_seconds=values['plain'], proxy_seconds=values['proxy'],
                             overhead_fraction=values['proxy'] / values['plain'] - 1, byte_identical=True))
        print(json.dumps(dict(part='overhead', charts=len(rows) // 5, wall_seconds=time.perf_counter() - started)), flush=True)
    summary = dict(part='overhead', status='complete', N=len(rows), rows=rows, provenance=provenance,
                   command=command, wall_seconds=time.perf_counter() - started,
                   by_chart={a['chart_sha256']: describe([r['overhead_fraction'] for r in rows if r['anchor'] == a['anchor']]) for a in anchors},
                   pooled=describe([r['overhead_fraction'] for r in rows]), all_byte_identical=True,
                   target_fraction=0.05, meets_target=float(np.median([r['overhead_fraction'] for r in rows])) < 0.05,
                   timing_boundary='model prefix prep and sampling in both; trace/reference creation, prefix replay, 15 costs and final-row charge only in proxy',
                   excluded_from_timers='model load, cache I/O, telemetry, metrics, likelihood scoring',
                   noise='concurrent phase-N training untouched; alternating paired order; five pairs/chart; no overhead tuning')
    (output / 'overhead.json').write_text(json.dumps(summary, indent=2) + '\n')
    update_model_report(output)
    return summary


def update_model_report(output):
    """Preserve the natural report verbatim and replace only generated model tables."""
    output = Path(output)
    path = output / 'report.md'
    marker = '\n<!-- C0 MODEL MEASUREMENTS -->\n'
    original = path.read_text().split(marker)[0]
    lines = [marker, '## Model sampling methods', '',
             'The checkpoint is pinned by file and SHA-256 in each part summary. No training is performed. '
             'Sixteen scan charts and eight disjoint controller charts share their own start across lengths. '
             'Starts are uniform on the feasible subset of 20–70% song time where 16 seconds fit and '
             'the first two seconds contain at least eight heads. All source decisions strictly before '
             'the first in-scope head are retained. Sampling stops before the first head at or after '
             'scope exit. Strain and reference both carry in from chart start.', '',
             'The full two-seed design is shuffled deterministically. Each part times its first 20 '
             'scopes including generation, complete likelihood, metrics and output, before the full run. '
             'A combined projection above three hours selects one common seed; otherwise two are used. '
             'Anchor, sampling, ordering and bootstrap seeds are recorded. One CPU compute thread is used.', '',
             'KL is exact over the 15 head-mask masses of each original model distribution. '
             'Full phase-N log-likelihood includes action and the equal forward/reverse release-pointer '
             'mixture, teacher forced on generated history under an empty track. Same-lane repeat rate '
             'divides repeated heads on adjacent in-scope rows by all heads on the preceding rows '
             '(last row excluded). LN share divides LN starts by all heads. Singleton support means '
             'exactly one finite-mass mask, not a one-note chord. Chord distributions and all scope '
             'quantiles are retained in JSON summaries.', '',
             'For the analytic controller, remaining budget is r*² WH_scope minus committed scope W. '
             'The desired next charge apportions it by the current reference charge divided by '
             'remaining reference workload. Independent positive/negative KL bisections restrict '
             'eta to both |eta| ≤ 8 and KL ≤ kappa, then an expectation bisection finds the charge. '
             'An unreachable request uses the closest feasible endpoint. The audit records budgets, '
             'all costs and base masses, expected/realized charges, eta, KL, mismatch, and active limit.', '',
             'Each trajectory NPZ preserves source prefix arrays, full head skeleton and beat grid, '
             'and generated decisions. `c0.load_trajectory` reconstructs a Chart for downstream use; '
             'scope continuations still need a legal tail and EOS for complete-chart export. '
             'Per-decision records use gzip JSONL. Bootstrap intervals resample charts (1,000 draws); '
             'only 16 or 8 charts are represented, so the intervals and scope spreads remain finite-sample evidence.']
    for part in ('part3a', 'part3b'):
        summary_path = output / f'{part}.json'
        if not summary_path.exists():
            continue
        s = json.loads(summary_path.read_text())
        pilot = json.loads((output / f'{part}-pilot.json').read_text())
        lines += ['', f'## {part}: {s["N"]} scopes', '',
                  f'Pilot: 20 scopes in {pilot["scope_seconds"]:.2f} s; full two-seed projection '
                  f'{pilot["projected_full_design_seconds"]:.2f} s. Completed scope wall time '
                  f'{s["scope_wall_seconds"]:.2f} s; {s["seeds"]} sampling seed(s).', '']
        if part == 'part3a':
            lines += ['| L (s) | eta | N | r p10 | r p50 | r p90 | W/s mean | KL/decision | full LL/decision | LN share | repeat rate |',
                      '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
            for length, groups in s['by_length'].items():
                for eta, g in groups.items():
                    m = g['metrics']
                    lines.append(f'| {length} | {eta} | {g["N"]} | {m["r"]["p10"]:.3f} | {m["r"]["p50"]:.3f} | {m["r"]["p90"]:.3f} | '
                                 f'{m["workload_per_sec"]["mean"]:.3f} | {g["pooled_per_decision_kl"]:.3f} | {m["full_log_likelihood_per_decision"]["mean"]:.3f} | '
                                 f'{g["pooled_ln_share"]:.3f} | {g["pooled_repeat_rate"]:.3f} |')
        else:
            lines += ['| L (s) | target | kappa | requested r | achieved p50 | log error p10/p50/p90 | within .05 | within .10 | saturated | second-half W share delta |',
                      '| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |']
            for length, kappas in s['by_length'].items():
                for kappa in ('0.25', '1.0'):
                    for name, g in kappas[kappa].items():
                        e = g['log_ratio_error']
                        lines.append(f'| {length} | {name} | {kappa} | {g["requested_r"]:.3f} | {g["metrics"]["r"]["p50"]:.3f} | '
                                     f'{e["p10"]:.3f}/{e["p50"]:.3f}/{e["p90"]:.3f} | {g["within_log_005"]:.3f} | {g["within_log_010"]:.3f} | '
                                     f'{g["saturation_fraction"]:.3f} | {g["second_half_share_minus_reference"]["p50"]:+.3f} |')
            lines += ['', f'Wrong-direction stop: `{s["stop_wrong_direction"]}`. Evidence: `{json.dumps(s["wrong_direction"])}`.', '',
                      'Saturation counts distinguish KL-limited and eta-limited decisions and their lower/upper '
                      'endpoints in `part3b.json`; per-decision audit files retain signed mismatch. '
                      'First/second-half workload shares and paired phase-N deltas expose concentration near scope exit.']
            lines += ['', 'Pooled per-length attainment and departure (all targets and both KL caps):', '',
                      '| L (s) | N | within .05 | within .10 | log error p50 | KL/decision | full LL/decision | LL delta vs phase N | LN share | repeat rate |',
                      '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
            for length, g in s['pooled_by_length'].items():
                lines.append(f'| {length} | {g["N"]} | {g["within_log_005"]:.3f} | {g["within_log_010"]:.3f} | '
                             f'{g["log_ratio_error"]["p50"]:.3f} | {g["pooled_per_decision_kl"]:.3f} | '
                             f'{g["metrics"]["full_log_likelihood_per_decision"]["mean"]:.3f} | '
                             f'{g["full_log_likelihood_per_decision_minus_reference"]["mean"]:+.3f} | '
                             f'{g["pooled_ln_share"]:.3f} | {g["pooled_repeat_rate"]:.3f} |')
        lines += ['', s['aggregate_weighting'] + '.']
        lines += ['', f'![{part} response]({part}-response.png)', '',
                  f'Checkpoint SHA-256: `{s["provenance"]["checkpoint"]["sha256"]}`.', '',
                  '```sh', s['command'], '```', '', 'Measurement source SHA-256:', '']
        lines += [f'- `{p}`: `{h}`' for p, h in s['provenance']['code_sha256'].items()]
    overhead_path = output / 'overhead.json'
    if overhead_path.exists():
        s = json.loads(overhead_path.read_text())
        lines += ['', '## Proxy-only overhead', '',
                  f'Pooled median paired overhead: {100 * s["pooled"]["p50"]:.2f}%; '
                  f'p10–p90 {100 * s["pooled"]["p10"]:.2f}%–{100 * s["pooled"]["p90"]:.2f}%. '
                  f'The <5% target is {"met" if s["meets_target"] else "not met"}. '
                  'All paired actions and gaps are byte-identical.', '',
                  s['timing_boundary'] + '. ' + s['excluded_from_timers'] + '.', '',
                  'Five alternating paired measurements per chart include trace/reference and source-prefix '
                  'preparation in the callback path. Concurrent phase-N training may perturb timings; '
                  'the proxy is not tuned against these measurements.', '',
                  '| Chart SHA | median overhead | p10 | p90 |', '| --- | ---: | ---: | ---: |']
        for chart, g in s['by_chart'].items():
            lines.append(f'| {chart} | {100*g["p50"]:.2f}% | {100*g["p10"]:.2f}% | {100*g["p90"]:.2f}% |')
    path.write_text(original + '\n'.join(lines) + '\n')


def close_screen_scope(chart, a_ms, b_ms):
    """Keep every prefix/scope decision and close remaining holds at scope end.

    The head skeleton is truncated before b; the original song duration remains
    the EOS upper bound. Require b < song_ms, so every closure is strictly inside
    the final gap. No new heads or in-scope actions are inserted. Return a complete
    Chart and assert that head-only ras-v1 scope workload is unchanged.
    """
    import numpy as np
    from .features import Chart
    from .state import replay_decisions
    from .strain import StrainTrace

    stop = int(np.searchsorted(chart.head_ms, b_ms, side='left'))
    if not 0 <= a_ms < b_ms < chart.song_ms or chart.n != stop or not stop:
        raise ValueError('Screen closure needs decisions through the scope and b < song_ms')
    heads = chart.head_ms[:stop].copy()
    state, _ = replay_decisions(heads, chart.song_ms, chart.actions, chart.gap)
    eos = np.where(state.held, 2, 0).astype(chart.actions.dtype)[None]
    releases = np.where(state.held, b_ms, np.nan)[None]
    complete = Chart(heads, chart.song_ms, chart.grid,
                     np.concatenate((chart.actions, eos)), np.concatenate((chart.gap, releases)))
    before = StrainTrace(chart.head_ms, chart.song_ms).extend(chart.actions, chart.gap).scope(a_ms, b_ms)
    trace = StrainTrace(heads, chart.song_ms).extend(complete.actions, complete.gap)
    after = trace.scope(a_ms, b_ms)
    if not trace.replay.finalized or before != after:
        raise ValueError('Screen EOS changed the measured scope or did not finalize')
    return complete


def screen_pair_candidates(rows, category, iqr):
    """Enumerate unique same-scope pairs; keep all failed acceptance counts."""
    from itertools import combinations

    groups, counts, pairs = {}, Counter(), []
    for row in rows:
        if row['length_sec'] not in (8, 16):
            continue
        if category == 'fixed' and (row.get('treatment') != 'fixed' or not row.get('eta')):
            continue
        if category != 'fixed' and row.get('eta') != 0:
            continue
        groups.setdefault((row['anchor'], row['length_sec']), []).append(row)
    for group in groups.values():
        for x, y in combinations(sorted(group, key=lambda r: r['id']), 2):
            if category == 'fixed' and not (x['eta'] * y['eta'] < 0 and x['sampling_seed'] == y['sampling_seed']):
                continue
            if category != 'fixed' and x['sampling_seed'] == y['sampling_seed']:
                continue
            counts['considered'] += 1
            if category == 'natural_gap' and abs(x['r'] - y['r']) < iqr[str(x['length_sec'])]:
                counts['below_iqr'] += 1
                continue
            if category == 'near_equal' and abs(math.log(x['r'] / y['r'])) > .03:
                counts['above_log_gap'] += 1
                continue
            delta = abs(x['ln_share'] - y['ln_share'])
            counts['ln_matched' if delta <= .10 else 'ln_exception_candidate'] += 1
            pairs.append(dict(category=category, sides=[x, y], ln_share_difference=delta))
    return pairs, dict(counts)


def select_screen_pairs(candidates, count, rng):
    """Prefer LN matches, then chart diversity and unused samples; randomize ties."""
    candidates = [candidates[int(i)] for i in rng.permutation(len(candidates))]
    selected, used_charts, used_samples = [], Counter(), Counter()
    while candidates and len(selected) < count:
        def rank(pair):
            x, y = pair['sides']
            return (pair['ln_share_difference'] > .10, used_charts[x['anchor']],
                    used_samples[x['id']] + used_samples[y['id']])
        best = min(range(len(candidates)), key=lambda i: rank(candidates[i]))
        pair = candidates.pop(best)
        selected.append(pair)
        used_charts[pair['sides'][0]['anchor']] += 1
        for side in pair['sides']:
            used_samples[side['id']] += 1
    return selected


def run_screen(cache, output, seed, checkpoint, *, measurements, corpus_root, max_seeds=16, command=None):
    """Build exactly 48 blind pairs, or retain an explicit failed-build receipt.

    Frozen part2 IQRs and part3 anchors come from ``measurements``. Existing model
    samples are reused only when their checkpoint hash matches. New checkpoint
    runs use the same source anchors and generate fresh samples. Candidates with
    a held lane at entry are excluded to retain the physical source prefix.
    Selection is mechanical; no perceived-difficulty judgment is produced.
    """
    import csv
    import shutil

    started = time.perf_counter()
    environment = configure_threads()
    import numpy as np
    import pyarrow.parquet as pq
    from .cache import load_chart
    from .export import export_chart, minimal_header
    from .state import replay_decisions
    from .strain import StrainTrace
    from .splits import verify_assignment

    cache, output, measurements, corpus_root = map(Path, (cache, output, measurements, corpus_root))
    if max_seeds < 0:
        raise ValueError('max_seeds must be nonnegative')
    screen = output / 'screen'
    screen.mkdir(parents=True, exist_ok=False)
    private = screen / 'private'
    private.mkdir()
    rng = np.random.Generator(np.random.PCG64(seed))
    ab_rng = np.random.Generator(np.random.PCG64(seed + 1))
    provenance = model_provenance(checkpoint, environment, output, 'part4')
    provenance['command'] = command
    natural = json.loads((measurements / 'part2.json').read_text())
    anchors = json.loads((measurements / 'part3-anchors.json').read_text())['anchors']
    iqr = {str(length): natural['by_length'][str(length)]['metrics']['r']['p75'] -
                       natural['by_length'][str(length)]['metrics']['r']['p25'] for length in (8, 16)}
    verify_assignment(cache / 'splits.json')
    groups = json.loads((cache / 'splits.json').read_text())['groups']
    records = pq.read_table(cache / 'index.parquet', columns=[
        'sha256', 'path', 'role', 'group_id', 'file', 'file_sha256', 'audio_sha256'],
        use_threads=False).to_pylist()
    records = {row['sha256']: row for row in records}
    eligible, excluded, charts, audio = [], [], {}, {}
    for anchor in anchors:
        ix = anchor['anchor']
        record = records[anchor['chart_sha256']]
        if record['role'] != 'fit_dev' or groups[record['group_id']] != 'fit_dev':
            raise ValueError('Screen source is outside fit_dev')
        source = corpus_root / record['path']
        if not source.is_file():
            excluded.append(dict(anchor=ix, reason='missing_source'))
            continue
        filename = next((line.split(':', 1)[1].strip() for line in source.read_text(
            encoding='utf-8-sig', errors='replace').splitlines() if line.startswith('AudioFilename:')), None)
        audio_path = source.parent / filename if filename else None
        if audio_path is None or not audio_path.is_file():
            excluded.append(dict(anchor=ix, reason='missing_audio'))
            continue
        actual_audio = file_sha256(audio_path)
        if record['audio_sha256'] and actual_audio != record['audio_sha256']:
            raise ValueError('Screen audio hash differs from cached corpus')
        chart_path = cache / 'charts' / record['file']
        if file_sha256(chart_path) != record['file_sha256']:
            raise ValueError('Screen source cache hash mismatch')
        dec = load_chart(chart_path)
        start = int(np.searchsorted(dec.head_ms, anchor['a_ms'], side='left'))
        state, _ = replay_decisions(dec.head_ms, dec.song_ms, dec.actions, dec.gap_release_ms, upto=start)
        if any(state.held):
            excluded.append(dict(anchor=ix, reason='held_lane_at_scope_entry'))
            continue
        if anchor['a_ms'] + 16000 >= dec.song_ms:
            excluded.append(dict(anchor=ix, reason='no_strict_eos_upper_bound'))
            continue
        eligible.append(anchor)
        charts[ix], audio[ix] = dec, (audio_path, actual_audio)
    rows, reused = [], Counter()
    eligible_ids = {a['anchor'] for a in eligible}
    for part in ('part3a', 'part3b'):
        summary = json.loads((measurements / f'{part}.json').read_text())
        if summary['provenance']['checkpoint']['sha256'] != provenance['checkpoint_sha256']:
            continue
        with gzip.open(measurements / f'{part}-scopes.jsonl.gz', 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                if row['anchor'] in eligible_ids and row['length_sec'] in (8, 16) and 'eta' in row:
                    rows.append(row)
                    reused[part] += 1
    model, generated, search_rounds = None, [], []

    def generate(anchor, length, eta, index):
        nonlocal model
        sampling_seed = seed + 10000 + 100 * anchor['anchor'] + index
        identity = f'sample-{len(generated):04d}'
        spec = dict(anchor=anchor['anchor'], chart_sha256=anchor['chart_sha256'],
                    chart_file=anchor['chart_file'], a_ms=anchor['a_ms'], length_sec=length,
                    sampling_seed=sampling_seed, seed_index=index, eta=eta,
                    treatment='fixed' if eta else 'reference', id=identity)
        destination = private / 'candidates' / identity
        if model is None:
            model, checkpoint_info = load_measurement_model(checkpoint)
            provenance['checkpoint'] = checkpoint_info
        result = measure_generated_scope(model, charts[anchor['anchor']], spec, destination)
        rows.append(result)
        generated.append(result)

    try:
        fixed, fixed_counts = screen_pair_candidates(rows, 'fixed', iqr)
        if len(fixed) < 16:
            for anchor in eligible:
                for eta in (-1.0, 1.0):
                    generate(anchor, 16, eta, 0)
            fixed, fixed_counts = screen_pair_candidates(rows, 'fixed', iqr)
        for index in range(max_seeds + 1):
            gaps, gap_counts = screen_pair_candidates(rows, 'natural_gap', iqr)
            near, near_counts = screen_pair_candidates(rows, 'near_equal', iqr)
            search_rounds.append(dict(round=index, generated=len(generated),
                                      natural_gap=gap_counts, near_equal=near_counts))
            if (sum(p['ln_share_difference'] <= .10 for p in gaps) >= 16 and
                    sum(p['ln_share_difference'] <= .10 for p in near) >= 8):
                break
            if index == max_seeds:
                break
            for anchor in eligible:
                for length in (8, 16):
                    generate(anchor, length, 0.0, index + 1)
            print(json.dumps(dict(stage='screen_candidate_search', round=index + 1,
                                  generated=len(generated))), flush=True)
        selected = (select_screen_pairs(fixed, 16, rng) +
                    select_screen_pairs(gaps, 16, rng) + select_screen_pairs(near, 8, rng))
        if Counter(p['category'] for p in selected) != Counter(fixed=16, natural_gap=16, near_equal=8):
            raise ValueError('Bounded screen search could not meet category counts; IQR unchanged')
        # Constructed comparisons use source prefixes with no open holds.
        for number in range(8):
            anchor = eligible[number % len(eligible)]
            dec = charts[anchor['anchor']]
            a, b = anchor['a_ms'], anchor['a_ms'] + 8000
            start, stop = np.searchsorted(dec.head_ms, (a, b), side='left')
            sides = []
            for side in range(2):
                actions, gap = dec.actions[:stop].copy(), dec.gap_release_ms[:stop].copy()
                actions[start:], gap[start:] = 0, np.nan
                for k in range(start, stop):
                    if side == 0:
                        actions[k, (0, 2, 1, 3)[(k - start) % 4]] = 1
                    elif number < 4:
                        actions[k, 0] = 1
                    else:
                        actions[k, :] = 1
                trace = StrainTrace(dec.head_ms, dec.song_ms).extend(actions, gap)
                value = trace.scope(a, b)
                identity = f'constructed-{number:02d}-{side}'
                meta = dict(anchor=anchor['anchor'], chart_sha256=anchor['chart_sha256'],
                            chart_file=anchor['chart_file'], a_ms=a, b_ms=b, length_sec=8,
                            start_row=int(start), stop_row=int(stop), eta=None, sampling_seed=None,
                            id=identity, ln_share=0.0, r=value.ratio, workload=value.workload,
                            reference=value.reference, construction=('cyclic_singles' if side == 0 else
                            'repeated_lane_singles' if number < 4 else 'four_lane_chords'))
                trajectory = private / f'{identity}.npz'
                np.savez_compressed(trajectory, head_ms=dec.head_ms, song_ms=dec.song_ms,
                                    grid_segments=dec.grid_segments, grid_bars=dec.grid_bars,
                                    actions=actions, gap=gap, prefix_actions=dec.actions[:start],
                                    prefix_gap=dec.gap_release_ms[:start], metadata=json.dumps(meta))
                sides.append(dict(**meta, trajectory=str(trajectory)))
            selected.append(dict(category='mechanical', sides=sides, ln_share_difference=0.0))
        selected = [selected[int(i)] for i in rng.permutation(48)]
        manifest, key, copied_audio = [], [], {}
        verified = Counter()
        chart_names = {ix: f'chart-{i + 1:03d}' for i, ix in enumerate(sorted({
            p['sides'][0]['anchor'] for p in selected}))}
        (screen / 'charts').mkdir()
        (screen / 'audio').mkdir()
        for number, pair in enumerate(selected, 1):
            pair_id = f'pair-{number:03d}'
            sides = pair['sides'][::(-1 if ab_rng.integers(2) else 1)]
            ix = sides[0]['anchor']
            dec = charts[ix]
            chart_name = chart_names[ix]
            if ix not in copied_audio:
                original, digest = audio[ix]
                target = screen / 'audio' / (chart_name + original.suffix.lower())
                shutil.copyfile(original, target)
                if file_sha256(target) != digest:
                    raise ValueError('Copied screen audio changed bytes')
                copied_audio[ix] = dict(path=str(target.relative_to(screen)), sha256=digest)
            a, b = sides[0]['a_ms'], sides[0]['b_ms']
            paths, objects_by_side, head_rows, sealed_sides = {}, [], [], {}
            for label, side in zip(('A', 'B'), sides):
                chart, meta, prefix_actions, prefix_gap = load_trajectory(side['trajectory'])
                start, stop = side['start_row'], side['stop_row']
                if not (np.array_equal(chart.head_ms, dec.head_ms) and
                        np.array_equal(chart.actions[:start], dec.actions[:start]) and
                        np.array_equal(chart.gap[:start], dec.gap_release_ms[:start], equal_nan=True) and
                        np.array_equal(prefix_actions, dec.actions[:start]) and
                        np.array_equal(prefix_gap, dec.gap_release_ms[:start], equal_nan=True)):
                    raise ValueError('Screen source prefix/skeleton changed')
                if (side['a_ms'], side['b_ms']) != (a, b):
                    raise ValueError('Screen pair scope mismatch')
                complete = close_screen_scope(chart, a, b)
                path = screen / 'charts' / f'{pair_id}-{label}.osu'
                header = minimal_header(dec.grid_segments, title=chart_name)
                header = header.replace('Mode:3\n', f'Mode:3\nAudioFilename:../{copied_audio[ix]["path"]}\n')
                header = header.replace('Version:R2', f'Version:{label}')
                objects, _ = export_chart(complete.head_ms, complete.song_ms, complete.actions,
                                          complete.gap, path, header)
                measured = StrainTrace(complete.head_ms, complete.song_ms).extend(
                    complete.actions, complete.gap).scope(a, b)
                if not math.isclose(measured.ratio, side['r'], rel_tol=1e-12):
                    raise ValueError('Export ratio differs from selected candidate')
                objects_by_side.append([(o.start_time_ms, o.end_time_ms, o.lane, o.kind.value) for o in objects])
                head_rows.append(sorted({o.start_time_ms for o in objects}))
                paths[label] = str(path.relative_to(screen))
                sealed_sides[label] = dict(**side, osu_sha256=file_sha256(path),
                                          closure_ms=b, trajectory_sha256=file_sha256(side['trajectory']))
                verified['legal_exports'] += 1
                verified['unchanged_scope_ras'] += 1
            if objects_by_side[0] == objects_by_side[1]:
                raise ValueError('Screen sides have identical arrangements')
            if head_rows[0] != head_rows[1] or head_rows[0] != dec.head_ms[:sides[0]['stop_row']].tolist():
                raise ValueError('Export changed source head times')
            prefixes = [[o for o in objects if o[0] < a] for objects in objects_by_side]
            if prefixes[0] != prefixes[1]:
                raise ValueError('Physical prefix differs across screen sides')
            verified['same_prefix_and_head_times'] += 1
            verified['different_arrangements'] += 1
            manifest.append(dict(pair_id=pair_id, chart=chart_name,
                                 scope_window=dict(start_song_ms=a, end_song_ms=b), files=paths))
            higher = 'same' if sides[0]['r'] == sides[1]['r'] else ('A' if sides[0]['r'] > sides[1]['r'] else 'B')
            key.append(dict(pair_id=pair_id, category=pair['category'], which_higher_r=higher,
                            ln_share_difference=pair['ln_share_difference'], sides=sealed_sides))
        key_document = dict(seed=seed, ab_seed=seed + 1, pairs=key, provenance=provenance,
                            source_exclusions=excluded, reused_candidates=dict(reused),
                            search_rounds=search_rounds, fixed_candidates=fixed_counts,
                            iqr=iqr, generated_candidates=generated,
                            source_audio={str(ix): dict(original=str(audio[ix][0]), **item)
                                          for ix, item in copied_audio.items()})
        (screen / 'KEY-sealed.json').write_text(json.dumps(key_document, indent=2) + '\n')
        (screen / 'pairs.json').write_text(json.dumps(manifest, indent=2) + '\n')
        with (screen / 'judge-sheet.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(('pair_id', 'harder', 'confidence', 'note'))
            writer.writerows((row['pair_id'], '', '', '') for row in manifest)
        (screen / 'instructions.md').write_text(
            '# Blind comparison screen\n\nOpen the two `.osu` files listed in `pairs.json` with their '
            'audio assets. Each file includes the complete source prefix and the marked scope. '
            'Compare only the marked scope at 1x playback.\n\n'
            'Fill `judge-sheet.csv`: harder = A, B, or same; confidence = high or low; '
            'note is optional. Leave undecided responses blank.\n\n'
            'Do not open `KEY-sealed.json`, `private/`, or measurement artifacts before judging. '
            'No judgments have been prefilled.\n')
        with (screen / 'judge-sheet.csv').open(newline='') as stream:
            judge = list(csv.DictReader(stream))
        assert len(judge) == 48 and all(not r[k] for r in judge for k in ('harder', 'confidence', 'note'))
        assert all(set(r) == {'pair_id', 'chart', 'scope_window', 'files'} for r in manifest)
        counts = Counter(p['category'] for p in selected)
        summary = dict(part='4', status='complete', pairs=48, category_counts=dict(counts),
                       distinct_charts_by_category={category: len({p['sides'][0]['anchor']
                           for p in selected if p['category'] == category}) for category in counts},
                       judgments_collected=0, seed=seed, ab_seed=seed + 1,
                       sampling_seed_formula='seed + 10000 + 100*anchor + new_seed_index (1-based)',
                       generated_candidates=len(generated), reused_candidates=dict(reused),
                       search_rounds=len(search_rounds), max_seed_rounds=max_seeds,
                       eligible_source_charts=len(eligible), selected_source_charts=len(copied_audio),
                       source_exclusion_counts=dict(Counter(x['reason'] for x in excluded)),
                       ln_matched_pairs=sum(p['ln_share_difference'] <= .10 for p in selected),
                       ln_exception_pairs=sum(p['ln_share_difference'] > .10 for p in selected),
                       natural_gap_iqr=iqr, near_equal_max_abs_log_ratio=.03,
                       verification=dict(verified, blank_judge_rows=48, blind_manifest_allowlist=True,
                                         copied_audio_hashes_verified=len(copied_audio)),
                       methods=dict(selection='Prefer LN difference <=.10; chart diversity; unused samples; PCG64 ties',
                                    entry='No held lane at first scope head; complete exact source prefix',
                                    natural_gap='Two eta=0 phase-N samples; abs(rA-rB) >= frozen part2 length IQR',
                                    mechanical='Four cyclic vs repeated-lane singles; four cyclic singles vs four-lane chords',
                                    closure='Truncate heads before b; close all remaining holds at b; EOS upper bound is original song_ms; b < song_ms',
                                    blinding='PCG64 shuffled pairs; independent seed+1 side swaps; only sealed key maps categories and values'),
                       assets=dict(osu=96, audio=len(copied_audio), manifest='screen/pairs.json',
                                   judge_sheet='screen/judge-sheet.csv', key='screen/KEY-sealed.json'),
                       provenance=provenance, wall_seconds=time.perf_counter() - started, command=command)
        (output / 'part4.json').write_text(json.dumps(summary, indent=2) + '\n')
        return summary
    except Exception as exc:
        failure = dict(part='4', status='failed', error=f'{type(exc).__name__}: {exc}',
                       wall_seconds=time.perf_counter() - started, provenance=provenance,
                       eligible=len(eligible), generated=len(generated), search_rounds=search_rounds)
        (output / 'part4-failure.json').write_text(json.dumps(failure, indent=2) + '\n')
        raise


class AllocationBias(MeasurementBias):
    """Allocate a realized remaining budget using masked natural expectations.

    N's running numerator includes the current history's expectation and all
    earlier in-scope expectations. NB updates that estimate even inside its
    exact-zero band. Reference charges and the eight-row prior use C0 units.
    """

    def __init__(self, head_ms, song_ms, a, b, *, rule, target, kappa, rho0, stats):
        super().__init__(head_ms, song_ms, a, b, target=target, kappa=kappa)
        if rule not in ('R', 'N', 'NB', 'reference'):
            raise ValueError('Unknown allocation rule')
        self.rule, self.rho0, self.stats = rule, rho0, stats
        self.natural_sum = 0.0
        self.prior_fallbacks = 0

    def __call__(self, k, chart, held, logp):
        import numpy as np
        from .strain import head_mask_log_probs

        self.trace.extend(chart.actions, chart.gap)
        if k != self.trace.replay.k or not np.array_equal(held, self.trace.replay.held):
            raise ValueError('Allocation replay and sampler held states disagree')
        if not self.start <= k < self.stop:
            return np.zeros(15)
        costs = self.trace.state.mask_costs(chart.time(k))
        lm = head_mask_log_probs(logp.double(), held).detach().cpu().numpy()
        expected0, _, _ = tilted_moments(lm, costs, self.scale, 0.0)
        self.natural_sum += expected0
        committed = self.trace.workload_rows(self.start, k)
        remaining = self.trace.reference_rows(k, self.stop)
        row_reference = self.trace.reference_rows(k, k + 1)
        elapsed_reference = self.trace.reference_rows(self.start, k + 1)
        rho2 = (self.natural_sum + 8 * self.scale * self.rho0 ** 2) / (elapsed_reference + 8 * self.scale)
        forecast = expected0 + rho2 * self.trace.reference_rows(k + 1, self.stop)
        projected = math.sqrt((committed + forecast) / self.reference)
        budget = None if self.target is None else self.target ** 2 * self.reference - committed
        desired = None
        in_band = self.rule == 'NB' and abs(math.log(projected / self.target)) <= 0.02
        if self.rule == 'reference' or in_band:
            solution = dict(eta=0.0, expected_charge=expected0, kl=0.0, limit=None, limit_side=None)
        else:
            desired = (budget * row_reference / remaining if self.rule == 'R' or forecast <= 0
                       else expected0 * budget / forecast)
            solution = solve_eta(lm, costs, self.scale, desired, self.kappa)
        fallbacks = self.stats.get('fallbacks', 0)
        row = dict(k=k, time_ms=chart.time(k), supported_masks=int(np.isfinite(lm).sum()),
                   committed_workload=committed, remaining_budget=budget,
                   remaining_reference=remaining, row_reference=row_reference,
                   desired_charge=desired, natural_expected_charge=expected0,
                   natural_expected_sum=self.natural_sum, elapsed_reference=elapsed_reference,
                   rho2=rho2, forecast_remaining_workload=forecast, projected_r=projected,
                   neutral_band=in_band, forecast_fallback=forecast <= 0,
                   min_hold_fallback=bool(fallbacks - self.prior_fallbacks),
                   mask_costs=costs.tolist(),
                   natural_mask_log_probs=[float(x) if np.isfinite(x) else None for x in lm], **solution)
        self.prior_fallbacks = fallbacks
        self.rows.append(row)
        eta = solution['eta']
        return eta * costs / self.scale if eta else np.zeros(15)


def allocation_hold_audit(chart, start, rows):
    """Count closed holds by release decision; unfinished holds are excluded."""
    import numpy as np

    derived = chart.derived()
    counts = dict(prefix_holds=0, prefix_short_holds=0, generated_holds=0,
                  generated_short_holds=0, generated_prefix_start_holds=0,
                  generated_prefix_start_short_holds=0)
    shorts = []
    by_k = {row['k']: row for row in rows}
    for k, codes in enumerate(chart.actions):
        durations = []
        for lane in np.flatnonzero(derived.held[k] & (codes > 0)):
            end = chart.time(k) if codes[lane] == 1 else float(chart.gap[k, lane])
            begin = float(derived.start[k, lane])
            duration = end - begin
            kind = 'prefix' if k < start else 'generated'
            counts[kind + '_holds'] += 1
            counts[kind + '_short_holds'] += int(duration <= 60)
            if k >= start and begin < chart.time(start):
                counts['generated_prefix_start_holds'] += 1
                counts['generated_prefix_start_short_holds'] += int(duration <= 60)
            if k >= start:
                if codes[lane] >= 2 and not np.any(chart.candidates(k).times == end):
                    raise ValueError('Allocation sampled a release outside its candidates')
                if duration <= 60 and not by_k[k]['min_hold_fallback']:
                    raise ValueError('Allocation sampled a short hold without fallback')
                durations.append(dict(lane=int(lane), start_ms=begin, end_ms=end, duration_ms=duration))
            if duration <= 60:
                shorts.append(dict(k=k, lane=int(lane), start_ms=begin, end_ms=end, inherited=k < start))
        if k in by_k:
            by_k[k]['completed_holds'] = durations
    return dict(**counts, short_holds=shorts)


ALLOCATION_CHECKPOINT_SHA256 = '09271d55d177e0c0ab3a8f3b1b293f66fc14feb338a7b657fdc4fe26d44c56b1'
ALLOCATION_SOURCE_FILES = ('src/ensomi_model/r2/sampling.py', 'src/ensomi_model/r2/evaluate.py',
                           'src/ensomi_model/r2/c0.py', 'src/ensomi_model/r2/README.md',
                           'tests/r2/test_min_hold.py')
_ALLOCATION_WORKER = None


def allocation_worker_init(cache, output, checkpoint):
    global _ALLOCATION_WORKER
    environment = configure_threads()
    model, loading = load_measurement_model(checkpoint)
    _ALLOCATION_WORKER = dict(cache=Path(cache), output=Path(output), model=model,
                              charts={}, checkpoint=loading, environment=environment)


def allocation_worker(spec):
    from .cache import load_chart

    state = _ALLOCATION_WORKER
    if spec['chart_sha256'] not in state['charts']:
        path = state['cache'] / 'charts' / spec['chart_file']
        if file_sha256(path) != spec['chart_file_sha256']:
            raise ValueError('Frozen allocation anchor chart changed')
        state['charts'][spec['chart_sha256']] = load_chart(path)
    row = measure_allocation_scope(state['model'], state['charts'][spec['chart_sha256']], spec,
                                   state['output'] / 'trajectories' / (spec['id'] + '.json'))
    row['worker_pid'] = os.getpid()
    return row


def allocation_specs(anchors, targets, seed, references=None):
    """Freeze the original C0 sampling seeds and shuffle controls independently."""
    import numpy as np

    specs = []
    for anchor in anchors:
        for length in (2, 4, 8, 16):
            for seed_index in range(2):
                ref_id = f"ref-a{anchor['anchor']}-l{length}-s{seed_index}"
                shared = {key: anchor[key] for key in ('anchor', 'chart_sha256', 'chart_file',
                                                       'chart_file_sha256', 'a_ms')}
                shared.update(length_sec=length, seed_index=seed_index,
                              sampling_seed=seed + 10000 + 100 * anchor['anchor'] + seed_index,
                              rho0=targets[str(length)]['middle'], min_hold_ms=60, reference_id=ref_id)
                if references is None:
                    specs.append(dict(**shared, id=ref_id, rule='reference', treatment='reference',
                                      eta=0.0, target=None, kappa=None))
                    continue
                ref = references[ref_id]
                for name, target in dict(targets[str(length)], own=ref['r']).items():
                    for rule in ('R', 'N', 'NB'):
                        for kappa in (0.25, 1.0):
                            specs.append(dict(**shared, id=f"{rule}-a{anchor['anchor']}-l{length}-{name}-k{kappa}-s{seed_index}",
                                              rule=rule, treatment='controller', target=target, target_name=name,
                                              kappa=kappa, reference_trajectory=ref['trajectory']))
    if references is not None:
        order = np.random.Generator(np.random.PCG64(seed + 63)).permutation(len(specs))
        specs = [specs[int(i)] for i in order]
    return specs


def allocation_summary(rows):
    """C0 aggregate weights, with exact-zero and short-hold counters."""
    import numpy as np

    if not rows:
        return dict(N=0)
    result = summarize_generated(rows, 2026100703 + rows[0]['length_sec'])
    decisions = sum(row['head_rows'] for row in rows)
    result.update(decisions=decisions, eta_zero_fraction=sum(row['eta_zero_count'] for row in rows) / decisions,
                  full_sample_reproduction_fraction=float(np.mean([row['full_sample_reproduction'] for row in rows])),
                  fallback_count=sum(row['min_hold']['fallbacks'] for row in rows),
                  generated_holds=sum(row['hold_audit']['generated_holds'] for row in rows),
                  generated_short_holds=sum(row['hold_audit']['generated_short_holds'] for row in rows),
                  inherited_prefix_holds=sum(row['hold_audit']['prefix_holds'] for row in rows),
                  inherited_prefix_short_holds=sum(row['hold_audit']['prefix_short_holds'] for row in rows))
    return result


def allocation_paired_comparisons(rows):
    """Pair on anchor, seed, length, target, and cap before comparing departure."""
    import numpy as np

    def key(row):
        return row['anchor'], row['sampling_seed'], row['length_sec'], row['target_name'], row['kappa']

    baseline = {key(row): row for row in rows if row['rule'] == 'R'}
    output = []
    for rule in ('N', 'NB'):
        for length in (2, 4, 8, 16):
            for target in ('low', 'middle', 'high', 'outlier', 'own'):
                for kappa in (0.25, 1.0):
                    pairs = [(baseline[key(row)], row) for row in rows if row['rule'] == rule
                             and row['length_sec'] == length and row['target_name'] == target
                             and row['kappa'] == kappa and key(row) in baseline]
                    if not pairs:
                        continue
                    subsets = dict(all=pairs,
                                   both_within_005=[p for p in pairs if max(abs(p[0]['log_ratio_error']), abs(p[1]['log_ratio_error'])) <= .05],
                                   matched_abs_error_001=[p for p in pairs if abs(abs(p[0]['log_ratio_error']) - abs(p[1]['log_ratio_error'])) <= .01])
                    row = dict(rule=rule, length_sec=length, target_name=target, kappa=kappa, subsets={})
                    for name, selected in subsets.items():
                        out = dict(N=len(selected))
                        if selected:
                            out.update(absolute_log_error_delta=float(np.mean([abs(b['log_ratio_error']) - abs(a['log_ratio_error']) for a, b in selected])),
                                       within_005_delta=float(np.mean([int(abs(b['log_ratio_error']) <= .05) - int(abs(a['log_ratio_error']) <= .05) for a, b in selected])),
                                       kl_delta=float(np.mean([b['kl_mean'] - a['kl_mean'] for a, b in selected])),
                                       full_ll_delta=float(np.mean([b['full_log_likelihood_per_decision'] - a['full_log_likelihood_per_decision'] for a, b in selected])),
                                       eta_zero_delta=float(np.mean([b['eta_zero_count'] / b['head_rows'] - a['eta_zero_count'] / a['head_rows'] for a, b in selected])))
                        row['subsets'][name] = out
                    output.append(row)
    return output


def run_allocation(cache, output, measurements, checkpoint, *, stage, workers, command):
    """Run masked paired references, time 20 controls, then resume the frozen grid."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    import numpy as np

    started = time.perf_counter()
    environment = configure_threads()
    cache, output, measurements, checkpoint = map(Path, (cache, output, measurements, checkpoint))
    if output.resolve() == measurements.resolve() or measurements.resolve() in output.resolve().parents:
        raise ValueError('Allocation outputs must be outside the read-only C0 input directory')
    if file_sha256(checkpoint) != ALLOCATION_CHECKPOINT_SHA256:
        raise ValueError('Allocation requires the frozen 48M checkpoint')
    output.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((measurements / 'part3-anchors.json').read_text())
    if frozen['cache'] != str(cache):
        raise ValueError('Allocation cache differs from frozen C0 anchors')
    anchors = [a for a in frozen['anchors'] if 16 <= a['anchor'] <= 23]
    if [a['anchor'] for a in anchors] != list(range(16, 24)):
        raise ValueError('Allocation requires exactly frozen anchors 16 through 23')
    natural = json.loads((measurements / 'part2.json').read_text())
    targets = {str(length): natural['by_length'][str(length)]['controller_targets'] for length in (2, 4, 8, 16)}
    provenance = model_provenance(checkpoint, environment, output, 'allocation')
    provenance['whole_task_source_sha256'] = {p: file_sha256(p) for p in ALLOCATION_SOURCE_FILES}
    provenance['inputs'] = {str(measurements / name): file_sha256(measurements / name)
                            for name in ('part3-anchors.json', 'part2.json')}
    manifest = dict(seed=frozen['seed'], anchors=anchors, targets=targets, min_hold_ms=60,
                    checkpoint=dict(path=str(checkpoint), sha256=ALLOCATION_CHECKPOINT_SHA256),
                    whole_task_source_sha256=provenance['whole_task_source_sha256'],
                    input_sha256=provenance['inputs'], code_sha256=provenance['code_sha256'])
    path = output / 'allocation-manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Allocation manifest differs from the frozen experiment')
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    refs_specs = allocation_specs(anchors, targets, frozen['seed'])
    references = {}
    if stage != 'references':
        for spec in refs_specs:
            row = json.loads((output / 'trajectories' / (spec['id'] + '.json')).read_text())
            if not row['zero_tilt_identity']:
                raise ValueError('Masked zero-tilt reference identity failed')
            references[spec['id']] = row
    specs = refs_specs if stage == 'references' else allocation_specs(anchors, targets, frozen['seed'], references)
    full_count = len(specs)
    if stage == 'pilot':
        specs = specs[:20]
    elif stage in ('full', 'summarize'):
        pilot = json.loads((output / 'allocation-pilot.json').read_text())
        if pilot['workers'] != workers:
            raise ValueError('Allocation pilot and full run need identical worker counts')
        if pilot['projected_full_design_seconds'] > 90 * 60:
            specs = [s for s in specs if not (s['rule'] == 'R' and s['kappa'] == .25)]
    rows, pending = [], []
    for spec in specs:
        receipt = output / 'trajectories' / (spec['id'] + '.json')
        if receipt.exists():
            row = json.loads(receipt.read_text())
            if any(row.get(k) != v for k, v in spec.items()):
                raise ValueError(f"Allocation receipt spec mismatch: {spec['id']}")
            rows.append(row)
        else:
            pending.append(spec)
    if stage == 'summarize' and pending:
        raise ValueError('Allocation summaries require the complete selected grid')
    execution_started = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=allocation_worker_init,
                                 initargs=(str(cache), str(output), str(checkpoint))) as pool:
            for row in pool.map(allocation_worker, pending, chunksize=1):
                rows.append(row)
                if len(rows) % 20 == 0 or len(rows) == len(specs):
                    print(json.dumps(dict(stage=stage, complete=len(rows), total=len(specs),
                                          wall_seconds=time.perf_counter() - started)), flush=True)
    execution_seconds = time.perf_counter() - execution_started
    result = dict(stage=stage, status='complete', N=len(rows), workers=workers,
                  newly_measured=len(pending), execution_seconds=execution_seconds,
                  wall_seconds=time.perf_counter() - started, command=command, provenance=provenance,
                  scope_wall_seconds=sum(r['wall_seconds'] for r in rows))
    if stage == 'pilot':
        if len(pending) != 20:
            raise ValueError('Timing pilot requires exactly 20 newly measured controls')
        result.update(scope_ids=[s['id'] for s in specs], full_design_scopes=full_count,
                      projected_full_design_seconds=execution_seconds / 20 * full_count,
                      timing_boundary='four spawned workers including model load, sampling, full LL, identity for own, and file writes',
                      schedule=[{k: s[k] for k in ('rule', 'anchor', 'length_sec', 'target_name', 'kappa', 'seed_index')} for s in specs])
    if stage == 'references':
        if len(rows) != 64 or not all(r['zero_tilt_identity'] for r in rows):
            raise ValueError('All 64 masked references must reproduce the no-bias arrays')
        result['zero_tilt_identity_count'] = len(rows)
        result['generated_short_holds'] = sum(r['hold_audit']['generated_short_holds'] for r in rows)
        result['fallback_count'] = sum(r['min_hold']['fallbacks'] for r in rows)
    if stage in ('full', 'summarize'):
        for row in rows:
            ref = references[row['reference_id']]
            row['reference_second_half_share_delta'] = row['second_half_workload_share'] - ref['second_half_workload_share']
            for metric in ('full_log_likelihood_per_decision', 'ln_share', 'same_lane_repeat_rate'):
                row[metric + '_minus_reference'] = row[metric] - ref[metric]
        result['deviations'] = [] if len(rows) == 1920 else ['R kappa .25 omitted because the 20-scope projection exceeded 90 minutes']
        result['groups'] = {rule: {str(length): {str(kappa): {name: allocation_summary([r for r in rows
                            if r['rule'] == rule and r['length_sec'] == length and r['kappa'] == kappa and r['target_name'] == name])
                            for name in ('low', 'middle', 'high', 'outlier', 'own')}
                            for kappa in (.25, 1.)} for length in (2, 4, 8, 16)} for rule in ('R', 'N', 'NB')}
        result['pooled'] = {rule: {str(length): allocation_summary([r for r in rows if r['rule'] == rule and r['length_sec'] == length])
                                  for length in (2, 4, 8, 16)} for rule in ('R', 'N', 'NB')}
        result['paired_comparisons'] = allocation_paired_comparisons(rows)
        with gzip.open(output / 'allocation-scopes.jsonl.gz', 'wt', encoding='utf-8') as stream:
            for row in sorted(rows, key=lambda r: r['id']):
                stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
        write_allocation_report(output, result)
    (output / f'allocation-{stage}.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def write_allocation_report(output, summary):
    """Write attained workload and original phase-N likelihood beside masked KL."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def triplet(d):
        return '/'.join(f"{d[k]:.3f}" for k in ('p10', 'p50', 'p90'))

    def cells(g):
        ll = g['metrics']['full_log_likelihood_per_decision']['mean']
        dll = g['full_log_likelihood_per_decision_minus_reference']['mean']
        return (f"{g['N']} | {triplet(g['metrics']['r'])} | {triplet(g['log_ratio_error'])} | "
                f"{g['within_log_005']:.3f} | {g['within_log_010']:.3f} | {g['pooled_per_decision_kl']:.4f} | "
                f"{g['eta_zero_fraction']:.3f} | {ll:.4f} | {dll:+.4f} | {g['saturation_fraction']:.3f} | "
                f"{g['second_half_share_minus_reference']['p50']:+.4f} | {g['pooled_repeat_rate']:.4f} | "
                f"{g['pooled_ln_share']:.4f} | {g['generated_short_holds']}/{g['generated_holds']} | "
                f"{g['inherited_prefix_short_holds']}/{g['inherited_prefix_holds']} | {g['fallback_count']} | "
                f"{g['full_sample_reproduction_fraction']:.3f}")

    columns = ('N | r p10/p50/p90 | log error p10/p50/p90 | within .05 | within .10 | KL/decision | '
               'exact eta=0 | full LL/decision | LL delta vs reference | saturated | second-half share delta p50 | '
               'repeat rate | LN share | generated ≤60 ms/holds | inherited ≤60 ms/holds | fallbacks | exact arrays')
    lines = ['# R2 allocation under minimum-hold decoding', '',
             f"Completed {summary['N']} controlled scopes and 64 paired masked phase-N references. "
             'Every generation uses the 48M checkpoint and `min_hold_ms=60`.', '',
             '## Measurement definitions', '',
             'The metric, carried source prefix, continuous cyclic reference, half-open scope, 15 head-mask costs, '
             'and scale `sS=WH/K` are the frozen C0 definitions. Anchors 16–23, lengths 2/4/8/16 seconds, '
             'and both original C0 sampling seeds are reused. Targets are frozen source p10, p50, p90, '
             '`max(p99,1.5*p50)`, and the achieved ratio of the paired zero-tilt masked reference (`own`).', '',
             'R requests `B*dWHk/WHrem`. N requests `ek*B/Fk`, where `ek` is the masked phase-N '
             'expectation at the current controller history, `rho2=(sum(ej)+8*sS*rho0²)/(sum(dWHj)+8*sS)`, '
             'and `Fk=ek+rho2*WH(k+1:end)`. Both sums include the current in-scope row, and `rho0` is the '
             'frozen natural median for the length. N falls back to R only when `Fk<=0`. NB updates the '
             'same estimate on every row, but uses exactly eta zero if '
             '`abs(log(sqrt((Wcommitted+Fk)/WH)/target))<=.02`. Else it uses N. '
             'All nonzero decisions use the unchanged C0 solver, `abs(eta)<=8` and head-mask KL cap .25 or 1.', '',
             'The duration mask is applied and renormalized before the callback computes `ek` or head-mask KL. '
             'The reported full LL uses the original, unmasked phase-N density on each sampled history: '
             '`R2Model.window` joint-action LL plus the log-mean-exp of both directed release-pointer orientations. '
             'It includes all pointer choices and orientation marginalization, and is not merely head-mask log mass. '
             'It is deliberately not renormalized over minimum-hold support, so it has the same scoring definition '
             'as C0. Both controlled samples and paired references are scored this way.', '',
             'Sampling reuses the unchanged Gumbel/orientation/pointer draw policy. All 64 reference arrays were '
             'checked byte-for-byte against same-seed masked sampling without a callback. The own-target array '
             'rate compares complete prefix-plus-scope action and gap arrays with that reference. '
             'Scopes stop before the exit row, as in C0; open holds are not artificially closed. Hold rates count '
             'completed holds by release decision: inherited means release before generation begins; generated '
             'includes closures of holds opened in the supplied prefix. Every generated gap release is checked '
             'against its candidate set and every nonfallback generated hold is strictly longer than 60 ms.', '',
             'Quantiles, attainment, LL means and paired LL deltas weight scopes equally. KL, saturation and exact-zero '
             'shares weight head decisions; LN share weights individual heads; repeat rate weights predecessor heads. '
             'Second-half changes compare each run with its same-anchor/length/seed reference. Pooled rows mix all '
             'five target strata and caps; the paired tables retain target and cap to avoid that confounding.', '',
             '## Per-rule, length, target and cap', '',
             '| Rule | L | Target | kappa | ' + columns + ' |',
             '| ' + ' | '.join(['---'] * 21) + ' |']
    for rule, lengths in summary['groups'].items():
        for length, caps in lengths.items():
            for cap, targets in caps.items():
                for name, group in targets.items():
                    if group['N']:
                        lines.append(f'| {rule} | {length} | {name} | {cap} | {cells(group)} |')
    lines += ['', '## Pooled by rule and length', '', '| Rule | L | ' + columns + ' |',
              '| ' + ' | '.join(['---'] * 19) + ' |']
    for rule, lengths in summary['pooled'].items():
        for length, group in lengths.items():
            lines.append(f'| {rule} | {length} | {cells(group)} |')
    lines += ['', '## Paired departure at comparable attainment', '',
              'Each N/NB run is paired with R on anchor, sampling seed, length, target and cap. '
              'The matched-error subset requires the absolute log errors to differ by at most .01; '
              'the second subset requires both errors to be at most .05. Delta is N/NB minus R. '
              'Negative KL delta indicates less head-mask tilt; positive LL delta indicates greater '
              'original phase-N likelihood. These are descriptive selected subsets, not randomized '
              'comparisons at an exactly fixed achieved ratio.', '',
              '| Rule | L | Target | kappa | all N | attainment .05 delta | abs-error delta | matched N | matched KL delta | matched LL delta | both ≤.05 N | both KL delta | both LL delta |',
              '| ' + ' | '.join(['---'] * 13) + ' |']
    for row in summary['paired_comparisons']:
        a, m, b = (row['subsets'][k] for k in ('all', 'matched_abs_error_001', 'both_within_005'))
        def val(d, k):
            return f'{d[k]:+.4f}' if k in d else '-'
        lines.append(f"| {row['rule']} | {row['length_sec']} | {row['target_name']} | {row['kappa']} | {a['N']} | "
                     f"{val(a, 'within_005_delta')} | {val(a, 'absolute_log_error_delta')} | {m['N']} | "
                     f"{val(m, 'kl_delta')} | {val(m, 'full_ll_delta')} | {b['N']} | {val(b, 'kl_delta')} | {val(b, 'full_ll_delta')} |")
    lines += ['', '## Provenance and execution', '',
              f"Checkpoint: `artifacts/r2-runs/r2-phaseN-20261006/checkpoints/ckpt-0048000198.pt`; SHA-256 `{ALLOCATION_CHECKPOINT_SHA256}`.", '',
              f"The command used {summary['workers']} compute processes, one numerical-library thread per process. "
              'No training was performed. `allocation-manifest.json` freezes anchors, targets, source hashes and input hashes. '
              '`allocation-source/` preserves runtime sources; `baseline-source/c0.py` preserves the pre-extension command implementation. '
              'The compressed scope and per-decision files retain exact measurements; each NPZ retains replayable arrays. '
              'No sealed screen key or private screen data is an input.', '',
              f"Full invocation wall time: {summary['wall_seconds']:.3f} seconds; summed measured scope wall times: {summary['scope_wall_seconds']:.3f} seconds.", '',
              'Deviations: ' + ('; '.join(summary['deviations']) or 'none') + '.', '',
              '```sh', summary['command'], '```', '', '| Whole-task source/test | SHA-256 |', '| --- | --- |']
    for path, sha in summary['provenance']['whole_task_source_sha256'].items():
        lines.append(f'| `{path}` | `{sha}` |')
    lines += ['', 'Exact commands, stage wall times, the first-20 schedule and worker count are in '
              '`allocation-references.json`, `allocation-pilot.json`, `allocation-full.json`, and `commands.json`. '
              'Final tests and source identities are recorded separately in `final-verification.json`.', '',
              '![Attainment and departure](allocation-response.png)', '']
    (Path(output) / 'report.md').write_text('\n'.join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for rule, lengths in summary['pooled'].items():
        xs = [int(x) for x in lengths]
        axes[0].plot(xs, [g['within_log_005'] for g in lengths.values()], marker='o', label=rule)
        axes[1].plot(xs, [g['pooled_per_decision_kl'] for g in lengths.values()], marker='o', label=rule)
    axes[0].set(ylabel='Fraction within |log error| ≤ .05', xlabel='Scope length (s)', ylim=(0, 1.05))
    axes[1].set(ylabel='Masked head KL / decision', xlabel='Scope length (s)')
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(Path(output) / 'allocation-response.png', dpi=150)
    plt.close(fig)


def measure_allocation_scope(model, dec, spec, destination):
    """Generate one scope, score complete decisions, and retain replayable arrays."""
    import numpy as np
    import torch
    from .cache import chart_grid
    from .common import GridArrays
    from .features import Chart
    from .sampling import continue_chart
    from .strain import heads_from_codes

    started = time.perf_counter()
    a, b = spec['a_ms'], spec['a_ms'] + 1000 * spec['length_sec']
    start, stop = (int(k) for k in np.searchsorted(dec.head_ms, (a, b), side='left'))
    if stop - start < MIN_HEAD_ROWS:
        raise ValueError('Frozen C0 anchor has fewer than eight scope heads')
    stats = {}
    tilt = AllocationBias(dec.head_ms, dec.song_ms, a, b, rule=spec['rule'],
                          target=spec.get('target'), kappa=spec.get('kappa'), rho0=spec['rho0'], stats=stats)
    grid = GridArrays.from_grid(chart_grid(dec))
    with torch.inference_mode():
        actions, gap = continue_chart(model, dec.head_ms, dec.song_ms, grid,
                                      dec.actions[:start], dec.gap_release_ms[:start], track=(),
                                      seed=spec['sampling_seed'], stop=stop, head_mask_bias=tilt,
                                      min_hold_ms=60, min_hold_stats=stats)
        assert np.array_equal(actions[:start], dec.actions[:start])
        assert np.array_equal(gap[:start], dec.gap_release_ms[:start], equal_nan=True)
        zero_identity = None
        if spec['rule'] == 'reference':
            plain = continue_chart(model, dec.head_ms, dec.song_ms, grid,
                                   dec.actions[:start], dec.gap_release_ms[:start], track=(),
                                   seed=spec['sampling_seed'], stop=stop, min_hold_ms=60)
            zero_identity = all(x.dtype == y.dtype and x.tobytes() == y.tobytes()
                                for x, y in zip((actions, gap), plain))
            if not zero_identity:
                raise ValueError('Masked phase-N zero-tilt arrays differ from no-bias arrays')
        tilt.finish(actions, gap)
        chart = Chart(dec.head_ms, dec.song_ms, grid, actions, gap)
        action_lp, release_lp = [], []
        for j in range(start, stop, 128):
            scored = model.window(chart, j, min(stop, j + 128), track=())
            action_lp.extend(scored.action.cpu().double().tolist())
            release_lp.extend(scored.release.cpu().double().tolist())
    if not (np.isfinite(action_lp).all() and np.isfinite(release_lp).all()):
        raise ValueError('C0 generated likelihood is nonfinite')
    held = chart.derived().held[:stop]
    heads = heads_from_codes(actions, held)
    q = heads[start:stop]
    sizes = q.sum(axis=1)
    ln = np.where(held[start:stop], actions[start:stop] == 4, actions[start:stop] == 2) & q
    pairs = q[1:] & q[:-1]
    measured = tilt.trace.scope(a, b)
    halfway = int(np.searchsorted(dec.head_ms, (a + b) / 2, side='left'))
    first = tilt.trace.workload_rows(start, halfway)
    second = tilt.trace.workload_rows(halfway, stop)
    for i, row in enumerate(tilt.rows):
        row.update(action_log_likelihood=action_lp[i], release_log_likelihood=release_lp[i],
                   head_mask=int(q[i].astype(int) @ (1 << np.arange(4))))
    result = dict(**spec, b_ms=b, start_row=start, stop_row=stop, head_rows=stop - start,
                  r=measured.ratio, workload=measured.workload, reference=measured.reference,
                  workload_per_sec=measured.intensity, chord_size_counts=np.bincount(sizes, minlength=5)[1:].tolist(),
                  heads=int(sizes.sum()), ln_heads=int(ln.sum()), ln_share=float(ln.sum() / sizes.sum()),
                  same_lane_repeat_count=int(pairs.sum()), repeat_denominator=int(q[:-1].sum()),
                  same_lane_repeat_rate=float(pairs.sum() / q[:-1].sum()),
                  kl_mean=float(np.mean([r['kl'] for r in tilt.rows])),
                  kl_max=max(r['kl'] for r in tilt.rows),
                  singleton_supported_mask_rate=float(np.mean([r['supported_masks'] == 1 for r in tilt.rows])),
                  action_log_likelihood_per_decision=float(np.mean(action_lp)),
                  release_log_likelihood_per_decision=float(np.mean(release_lp)),
                  full_log_likelihood_per_decision=float(np.mean(np.array(action_lp) + release_lp)),
                  first_half_workload=first, second_half_workload=second,
                  first_half_workload_share=first / measured.workload,
                  second_half_workload_share=second / measured.workload,
                  saturation_counts=dict(Counter(r['limit'] or 'none' for r in tilt.rows)),
                  saturation_side_counts=dict(Counter(r['limit_side'] or 'none' for r in tilt.rows)),
                  eta_min=min(r['eta'] for r in tilt.rows), eta_max=max(r['eta'] for r in tilt.rows))
    if spec.get('target') is not None:
        result['log_ratio_error'] = math.log(result['r'] / spec['target'])
        result['terminal_budget'] = spec['target'] ** 2 * result['reference'] - result['workload']
    result.update(min_hold=dict(ms=60, **stats), zero_tilt_identity=zero_identity,
                  eta_zero_count=sum(row['eta'] == 0 for row in tilt.rows),
                  hold_audit=allocation_hold_audit(chart, start, tilt.rows),
                  action_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
                  gap_sha256=hashlib.sha256(gap.tobytes()).hexdigest(),
                  full_sample_reproduction=spec['rule'] == 'reference')
    if spec['rule'] != 'reference':
        with np.load(spec['reference_trajectory'], allow_pickle=False) as ref:
            result['full_sample_reproduction'] = all(x.dtype == y.dtype and x.tobytes() == y.tobytes()
                for x, y in ((actions, ref['actions']), (gap, ref['gap'])))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    trajectory = destination.with_suffix('.npz')
    np.savez_compressed(trajectory, head_ms=dec.head_ms, song_ms=dec.song_ms,
                        grid_segments=dec.grid_segments, grid_bars=dec.grid_bars,
                        actions=actions, gap=gap, prefix_actions=dec.actions[:start],
                        prefix_gap=dec.gap_release_ms[:start], metadata=json.dumps(result))
    with gzip.open(destination.with_suffix('.decisions.jsonl.gz'), 'wt', encoding='utf-8') as stream:
        for row in tilt.rows:
            stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
    result['trajectory'] = str(trajectory)
    result['decisions'] = str(destination.with_suffix('.decisions.jsonl.gz'))
    result['wall_seconds'] = time.perf_counter() - started
    destination.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    natural = commands.add_parser('natural', help='Measure all fit_dev source charts')
    natural.add_argument('--cache', type=Path, required=True)
    natural.add_argument('--output', type=Path, required=True)
    natural.add_argument('--seed', type=int, required=True)
    natural.add_argument('--checkpoint', type=Path, default=Path(PINNED_CHECKPOINT),
                         help='Checkpoint recorded by filename and hash only')
    screen = commands.add_parser('screen', help='Build 48 blind pairs without judging')
    for name in ('cache', 'output', 'checkpoint', 'measurements'):
        screen.add_argument('--' + name, type=Path, required=True)
    screen.add_argument('--seed', type=int, required=True)
    screen.add_argument('--corpus-root', type=Path, default=Path('.'),
                        help='Root against which cached source paths resolve')
    screen.add_argument('--max-seeds', type=int, default=16,
                        help='Maximum additional eta=0 seed rounds; never relax the IQR threshold')
    for name in ('scan', 'controller', 'overhead'):
        child = commands.add_parser(name)
        child.add_argument('--cache', type=Path, required=True)
        child.add_argument('--output', type=Path, required=True)
        child.add_argument('--seed', type=int, required=True,
                           help='Common anchor/design seed, shared across all three commands')
        child.add_argument('--checkpoint', type=Path, required=True)
        if name != 'overhead':
            child.add_argument('--pilot-only', action='store_true')
            child.add_argument('--seeds', type=int, choices=(1, 2), default=2)
    allocation = commands.add_parser('allocation', help='Compare masked R/N/NB workload allocation')
    for name in ('cache', 'output', 'checkpoint', 'measurements'):
        allocation.add_argument('--' + name, type=Path, required=True)
    allocation.add_argument('--stage', choices=('references', 'pilot', 'full', 'summarize'), required=True)
    allocation.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    args = parser.parse_args(argv)
    command = ' '.join(f'{name}=1' for name in THREAD_ENV) + ' PYTHONPATH=src .venv/bin/python -m ensomi_model.r2.c0 '
    command += shlex.join(sys.argv[1:] if argv is None else argv)
    if args.command == 'allocation':
        summary = run_allocation(args.cache, args.output, args.measurements, args.checkpoint,
                                 stage=args.stage, workers=args.workers, command=command)
    elif args.command == 'natural':
        summary = run_natural(args.cache, args.output, args.seed, args.checkpoint, command=command)
    elif args.command == 'screen':
        summary = run_screen(args.cache, args.output, args.seed, args.checkpoint,
                             measurements=args.measurements, corpus_root=args.corpus_root,
                             max_seeds=args.max_seeds, command=command)
    elif args.command == 'overhead':
        summary = run_overhead(args.cache, args.output, args.seed, args.checkpoint, command=command)
    else:
        summary = run_model_part('part3a' if args.command == 'scan' else 'part3b',
                                 args.cache, args.output, args.seed, args.checkpoint, command=command,
                                 pilot_only=args.pilot_only, seeds=args.seeds)
    print(json.dumps(dict(status=summary['status'], output=str(args.output),
                          stop_nearly_constant=summary.get('stop_nearly_constant'),
                          stop_wrong_direction=summary.get('stop_wrong_direction'),
                          wall_seconds=summary.get('wall_seconds'))), flush=True)


if __name__ == '__main__':
    main()
