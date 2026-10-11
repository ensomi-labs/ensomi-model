"""Whole-song LN levels and a versioned empirical prior from fit_train sources.

A level is LN heads / all heads in [0, T], under ``properties.ln_share``.
The prior preserves one observation per cached training chart, including repeated
values. It conditions on ``band_of(star)`` and head-row density ``1000 * K / T``.
Density terciles are fitted separately within each band using NumPy's linear
quantiles at 1/3 and 2/3; a density on a threshold belongs to the upper cell.
Empty cells are errors at sampling time, with no cross-cell fallback.
Each song's draw uses its exact float64 head times and duration together with
the run seed, independently of action sampling and the supplied chart prefix.

Build from the successful cache index (whose n_ln and n_objects counts survive
head-row merging and release snapping unchanged)::

    .venv/bin/python -m ensomi_model.r2.ln_level --cache artifacts/r2-cache/v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
from numbers import Real
from pathlib import Path

import numpy as np

from .common import ContractError, band_of
from .properties import NU_HASH, ln_share_rows

PRIOR_VERSION = 'r2-ln-level-prior-v1'
PRIOR_FILE = 'ln-level-prior-v1.json'
PRIOR_SEED_DOMAIN = 'r2-ln-level-prior-v1/sample'
JOINT_PRIOR_VERSION = 'r2-ln-level-prior-v2'
JOINT_PRIOR_FILE = 'ln-level-prior-v2.json'
JOINT_PRIOR_SEED_DOMAIN = 'r2-ln-level-prior-v2/sample'
MIN_LENGTH_HOLDS = 10
LENGTH_BEAT_FLOOR = 1e-4
LENGTH_STATISTIC = 'median(log2(max(grid.beat(release) - grid.beat(head), 1e-4))); all holds including EOS'


def checked_length(value) -> float:
    """Return a finite log2 length; reject booleans and nonnumeric values."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ContractError('LN length must be a finite number in log2 beats')
    value = float(value)
    if not np.isfinite(value) or abs(value) > np.finfo(np.float32).max:
        raise ContractError('LN length must be a finite float32 number in log2 beats')
    return value


def checked_level(value) -> float:
    """Return a finite level in [0, 1]; reject booleans and nonnumeric values."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ContractError('LN level must be a finite number in [0, 1]')
    level = float(value)
    if not np.isfinite(level) or not 0 <= level <= 1:
        raise ContractError('LN level must be a finite number in [0, 1]')
    return level


def whole_ln_level(chart) -> float:
    """Source LN share over [0, T]; require every head row and at least one head.

    EOS need not be present because it owns no heads. This function reads source
    decisions explicitly; no level is attached to a chart or inherited by a prefix.
    """
    if chart.n < chart.K:
        raise ContractError('Whole-song LN level requires every source head row')
    d = chart.derived()
    value = ln_share_rows(d.attack.sum(1), d.ln_head.sum(1), chart.head_ms,
                         0.0, chart.song_ms, chart.song_ms)
    if value is None:
        raise ContractError('Whole-song LN level is undefined without source heads')
    return checked_level(value)


def whole_ln_length(chart) -> float | None:
    """Median log2 length of every source hold in grid beats, or None below ten holds.

    Requires the complete source through EOS, including snapped gap releases.
    Invalid or unclosed holds and nonpositive/nonfinite beat lengths raise. The
    1e-4 beat floor matches the corpus length statistic; it never repairs invalid
    timing. Length is measured across grid segments, not using the head's BPM.
    """
    if chart.n != chart.K + 1:
        raise ContractError('Whole-song LN length requires every source decision through EOS')
    d = chart.derived()
    if not np.array_equal(chart.actions[-1], np.where(d.held[-2], 2, 0)) or d.held[-1].any():
        raise ContractError('Whole-song LN length requires valid EOS closure')
    closed = d.held[:-1] & (chart.actions != 0)
    starts, ends = d.start[:-1][closed], d.release[closed]
    if (len(starts) != int(d.ln_head.sum()) or not np.isfinite(starts).all()
            or not np.isfinite(ends).all() or np.any(ends <= starts)):
        raise ContractError('Whole-song LN length requires finite positive, closed holds')
    rows, _ = np.nonzero(closed)
    gap_release = chart.actions[closed] != 1
    previous = chart.head_ms[np.maximum(rows - 1, 0)]
    current = chart.times(rows)
    if np.any(gap_release & ((rows == 0) | (ends <= previous) | (ends >= current))):
        raise ContractError('Whole-song LN length has a release outside its decision gap')
    beats = chart.grid.beat(ends) - chart.grid.beat(starts)
    if not np.isfinite(beats).all() or np.any(beats <= 0):
        raise ContractError('Whole-song LN length requires finite positive grid-beat lengths')
    if len(beats) < MIN_LENGTH_HOLDS:
        return None
    return checked_length(float(np.median(np.log2(np.maximum(beats, LENGTH_BEAT_FLOOR)))))


def skeleton_density(head_ms, song_ms: float) -> float:
    """Whole-song head rows per second, including silence in the song duration."""
    if not np.isfinite(song_ms) or song_ms <= 0:
        raise ContractError('LN prior density requires a finite positive song duration')
    return 1000.0 * len(head_ms) / float(song_ms)


def skeleton_sha256(head_ms, song_ms: float) -> str:
    """Stable song identity: uint64 row count, then little-endian float64 H and T."""
    head = np.asarray(head_ms, dtype='<f8')
    if head.ndim != 1 or not np.isfinite(head).all():
        raise ContractError('LN prior requires finite one-dimensional head times')
    digest = hashlib.sha256(len(head).to_bytes(8, 'little'))
    digest.update(head.tobytes())
    digest.update(np.asarray([song_ms], dtype='<f8').tobytes())
    return digest.hexdigest()


class EmpiricalLNPrior:
    """Validated prior JSON and one independent, seed-reproducible empirical draw."""

    def __init__(self, data: dict, *, sha256=None):
        if data.get('version') not in (PRIOR_VERSION, JOINT_PRIOR_VERSION) or data.get('nu_hash') != NU_HASH:
            raise ContractError('LN prior version or measurement semantics do not match')
        self.with_length = data['version'] == JOINT_PRIOR_VERSION
        self.version = data['version']
        self.seed_domain = JOINT_PRIOR_SEED_DOMAIN if self.with_length else PRIOR_SEED_DOMAIN
        if self.with_length and data.get('length') != dict(statistic=LENGTH_STATISTIC, min_holds=MIN_LENGTH_HOLDS):
            raise ContractError('LN prior length measurement semantics do not match')
        if data.get('role') != 'fit_train':
            raise ContractError('LN prior must be fitted on fit_train only')
        for band in range(2, 6):
            part = data.get('bands', {}).get(str(band))
            if part is None:
                raise ContractError(f'LN prior is missing band {band}')
            thresholds = np.asarray(part.get('density_thresholds', []), dtype=float)
            if thresholds.shape != (2,) or not np.isfinite(thresholds).all() or np.any(np.diff(thresholds) < 0):
                raise ContractError(f'LN prior band {band} needs two ordered density thresholds')
            cells = part.get('cells', [])
            if len(cells) != 3:
                raise ContractError(f'LN prior band {band} needs three density cells')
            for i, cell in enumerate(cells):
                shares, shas, lengths = self._support(cell)
                if cell.get('tercile') != i or cell.get('count') != len(shares) or len(shas) != len(shares):
                    raise ContractError(f'LN prior band {band} cell {i} count does not match its sources')
                for value in shares:
                    checked_level(value)
                for value in lengths:
                    if value is not None:
                        checked_length(value)
            if part.get('count') != sum(c['count'] for c in cells):
                raise ContractError(f'LN prior band {band} count does not match its cells')
        if data.get('count') != sum(p['count'] for p in data['bands'].values()):
            raise ContractError('LN prior count does not match its bands')
        sources = [sha for p in data['bands'].values() for c in p['cells'] for sha in self._support(c)[1]]
        if len(set(sources)) != len(sources):
            raise ContractError('LN prior contains repeated source chart identities')
        self.data, self.sha256 = data, sha256

    def _support(self, cell):
        if self.with_length:
            observations = cell.get('observations', [])
            if any(set(o) != {'share', 'length', 'source_sha256'} for o in observations):
                raise ContractError('Joint LN prior observations need share, length and source_sha256')
            return ([o['share'] for o in observations], [o['source_sha256'] for o in observations],
                    [o['length'] for o in observations])
        return cell.get('shares', []), cell.get('source_sha256', []), []

    @classmethod
    def load(cls, path):
        raw = Path(path).read_bytes()
        return cls(json.loads(raw), sha256=hashlib.sha256(raw).hexdigest())

    @classmethod
    def fit(cls, records, *, cache_index_sha256=None, with_length=False):
        """Fit from index records; ignore every role other than fit_train.

        Each record supplies sha256, role, star, K, song_ms, n_objects and n_ln.
        Joint v2 records also supply whole-source ln_length (None below ten holds).
        Sorting by source SHA fixes empirical support order independently of
        index row order. Invalid training counts or an absent band raise.
        """
        records = sorted((r for r in records if r['role'] == 'fit_train'), key=lambda r: r['sha256'])
        observations = []
        for r in records:
            heads, holds, rows = int(r['n_objects']), int(r['n_ln']), int(r['K'])
            if heads <= 0 or not 0 <= holds <= heads or rows <= 0 or rows > heads:
                raise ContractError(f'Invalid cached head counts for LN prior source {r["sha256"]}')
            if not np.isfinite(r['star']) or not np.isfinite(r['song_ms']) or r['song_ms'] <= 0:
                raise ContractError(f'Invalid star or song duration for LN prior source {r["sha256"]}')
            observations.append(dict(sha256=r['sha256'], band=band_of(float(r['star'])),
                                     density=1000.0 * rows / float(r['song_ms']),
                                     heads=heads, ln_heads=holds, share=holds / heads))
            if with_length:
                if 'ln_length' not in r or (r['ln_length'] is None) != (holds < MIN_LENGTH_HOLDS):
                    raise ContractError(f'LN prior source {r["sha256"]} length does not match its hold count')
                observations[-1]['length'] = None if r['ln_length'] is None else checked_length(r['ln_length'])
        bands = {}
        for band in range(2, 6):
            obs = [o for o in observations if o['band'] == band]
            if not obs:
                raise ContractError(f'LN prior has no fit_train sources in band {band}')
            thresholds = np.quantile([o['density'] for o in obs], [1 / 3, 2 / 3], method='linear')
            cells = [dict(tercile=i, count=0, shares=[], source_sha256=[]) for i in range(3)]
            for o in obs:
                cell = cells[int(np.searchsorted(thresholds, o['density'], side='right'))]
                cell['shares'].append(o['share'])
                cell['source_sha256'].append(o['sha256'])
                cell['count'] += 1
            if with_length:
                lengths = {o['sha256']: o['length'] for o in obs}
                for cell in cells:
                    cell['observations'] = [dict(share=share, length=lengths[sha], source_sha256=sha)
                                            for share, sha in zip(cell.pop('shares'), cell.pop('source_sha256'))]
            bands[str(band)] = dict(count=len(obs), density_thresholds=thresholds.tolist(), cells=cells)
        source_hash = hashlib.sha256(json.dumps(observations, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        return cls(dict(version=JOINT_PRIOR_VERSION if with_length else PRIOR_VERSION,
                        nu_hash=NU_HASH, role='fit_train', count=len(observations),
                        cache_index_sha256=cache_index_sha256, fit_sources_sha256=source_hash,
                        level='n_ln / n_objects: LN heads / all heads over [0, T]',
                        band='common.band_of(star)', density='1000 * K / song_ms: head rows per second',
                        terciles=dict(scope='within each band', quantiles=[1 / 3, 2 / 3],
                                      method='linear', threshold_ties='upper cell'),
                        sampling=dict(distribution='uniform over source observations, including repeated shares',
                                      seed_domain=JOINT_PRIOR_SEED_DOMAIN if with_length else PRIOR_SEED_DOMAIN,
                                      seed_inputs='run seed and skeleton SHA-256',
                                      skeleton_identity='uint64 row count, little-endian float64 head times and T',
                                      empty_cell='error'), bands=bands,
                        **(dict(length=dict(statistic=LENGTH_STATISTIC, min_holds=MIN_LENGTH_HOLDS))
                           if with_length else {})))

    def sample(self, *, star: float, head_ms, song_ms: float, seed: int):
        """Return (level, metadata), with an RNG independent of action sampling.

        The run seed and exact skeleton identify the draw, so different songs
        in the same cell can draw different observations. Call once per song
        and retain the result for every generated decision; BOS and prefix
        continuations with the same seed share the draw.
        """
        if not isinstance(star, Real) or not np.isfinite(star):
            raise ContractError('LN prior sampling requires a finite source star rating')
        band, density = band_of(float(star)), skeleton_density(head_ms, song_ms)
        part = self.data['bands'][str(band)]
        tercile = int(np.searchsorted(part['density_thresholds'], density, side='right'))
        cell = part['cells'][tercile]
        if not cell['count']:
            raise ContractError(f'LN prior band {band} density cell {tercile} has no source observations')
        skeleton_hash = skeleton_sha256(head_ms, song_ms)
        digest = hashlib.sha256(f'{self.seed_domain}\0{int(seed)}\0{skeleton_hash}'.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
        index = int(rng.integers(cell['count']))
        shares, shas, _ = self._support(cell)
        return float(shares[index]), dict(
            prior_version=self.version, prior_sha256=self.sha256, band=band, density=density,
            density_tercile=tercile, density_thresholds=part['density_thresholds'],
            support_count=cell['count'], support_index=index, sampled_source_sha256=shas[index],
            prior_seed=int(seed), seed_domain=self.seed_domain, skeleton_sha256=skeleton_hash)

    def sample_joint(self, **kwargs):
        """Return ((share, length or None), metadata) from one v2 source observation."""
        if not self.with_length:
            raise ContractError('LN length requires the joint v2 prior')
        share, record = self.sample(**kwargs)
        cell = self.data['bands'][str(record['band'])]['cells'][record['density_tercile']]
        length = cell['observations'][record['support_index']]['length']
        return (share, length), record


def resolve_ln_level(mode, *, source_ln_level=None, ln_prior=None, star=None,
                     head_ms=(), song_ms=None, seed=954):
    """Return (value or None, record) for unknown, oracle, prior or a numeric level.

    Oracle requires an explicit source value. Prior requires a fitted prior and
    source star; neither mode can recover a source value from committed history.
    """
    if mode is None or (isinstance(mode, str) and mode == 'unknown'):
        return None, dict(mode='unknown', known=False, value=None)
    if isinstance(mode, Real):
        value = checked_level(mode)
        return value, dict(mode='fixed', known=True, value=value)
    if mode == 'oracle':
        value = checked_level(source_ln_level)
        return value, dict(mode='oracle', known=True, value=value)
    if mode == 'prior':
        if not isinstance(ln_prior, EmpiricalLNPrior):
            raise ContractError('LN prior sampling requires an EmpiricalLNPrior')
        value, record = ln_prior.sample(star=star, head_ms=head_ms, song_ms=song_ms, seed=seed)
        return value, dict(record, mode='prior', known=True, value=value)
    raise ContractError('LN level mode must be unknown, oracle, prior or a finite number in [0, 1]')


def resolve_ln_controls(mode, *, ln_length=None, source_ln_level=None, source_ln_length=None,
                        ln_prior=None, star=None, head_ms=(), song_ms=None, seed=954):
    """Return (share, length, record) with one joint prior draw or explicit source values.

    Unknown clears both inputs. Oracle uses only the two source arguments. A
    numeric share accepts an optional log2 length; None leaves length unknown.
    A supplied fixed length with a nonnumeric mode raises instead of being ignored.
    """
    if ln_length is not None and not isinstance(mode, Real):
        raise ContractError('A fixed LN length requires a fixed numeric LN share')
    if isinstance(mode, str) and mode == 'prior':
        if not isinstance(ln_prior, EmpiricalLNPrior):
            raise ContractError('LN prior sampling requires an EmpiricalLNPrior')
        (share, length), record = ln_prior.sample_joint(star=star, head_ms=head_ms, song_ms=song_ms, seed=seed)
        record = dict(record, mode='prior', known=True, value=share)
    else:
        share, record = resolve_ln_level(mode, source_ln_level=source_ln_level)
        length = source_ln_length if mode == 'oracle' else ln_length if share is not None else None
    length = None if length is None else checked_length(length)
    record['length'] = dict(known=length is not None, value=length, units='log2 beats')
    return share, length, record


def build_prior(cache_root, *, out=None, with_length=False):
    """Write the fitted JSON and return the validated prior, including its file hash."""
    from .cache import CacheIndex, load_chart
    from .data import chart_from_cache
    if with_length and out is None:
        raise ContractError('Joint LN prior fitting requires an explicit output path')
    root = Path(cache_root)
    index = CacheIndex.open(root)
    records = index.table.to_dict('records')
    if with_length:
        for row in records:
            if row['role'] == 'fit_train':
                chart = chart_from_cache(load_chart(root / 'charts' / row['file']))
                row['ln_length'] = whole_ln_length(chart)
                d = chart.derived()
                if int(d.ln_head.sum()) != int(row['n_ln']) or int(d.attack.sum()) != int(row['n_objects']):
                    raise ContractError(f'LN prior source {row["sha256"]} cached counts differ from decisions')
    prior = EmpiricalLNPrior.fit(records, cache_index_sha256=index.summary.get('index_sha256'),
                               with_length=with_length)
    path = Path(out) if out is not None else root / PRIOR_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prior.data, sort_keys=True, indent=1) + '\n')
    return EmpiricalLNPrior.load(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default='artifacts/r2-cache/v1')
    parser.add_argument('--out')
    parser.add_argument('--with-length', action='store_true', help='Fit the joint v2 prior; requires --out')
    args = parser.parse_args(argv)
    prior = build_prior(args.cache, out=args.out, with_length=args.with_length)
    print(json.dumps(dict(version=prior.version, count=prior.data['count'], sha256=prior.sha256,
                          bands={b: dict(count=p['count'], thresholds=p['density_thresholds'],
                                         cells=[c['count'] for c in p['cells']])
                                 for b, p in prior.data['bands'].items()}), indent=2))


if __name__ == '__main__':
    main()
