"""Per-chart work of the harness, run in worker processes.

Fit split, first pass (``fit_values``): the family's values under the whole
scope, as sufficient statistics for its model; targeted descriptions; legality
as read. Fit split, second pass (``fit_null``, once the model is fitted): every
fit song scored cross-fitted (model and reference without its fold) under the
whole scope and in the same-shape continuation windows, for the song nulls.
Calibration (``calibration_chart``): the chart as read, the identity ``.osu``
round trip, every must-not-flag transform, and, for charts at 2 to 7 stars,
every injection at every dose, under the timing and skeleton conditions and
both scopes. Every transformed or injected chart is written to ``.osu``, parsed
back and checked for legality before it is judged. Song p-values are computed
later, in the parent, from the nulls.

Every song is scored as a fit song of its fold is: a calibration or injected
chart gets the fold of its song group (``trivial.fold_of``), and the model and
the reference without that fold. The song null (cross-fitted fit songs) and the
songs it judges are then scored by one procedure.

The family is loaded from ``--family`` (``module:attribute``, a family or a
factory of one); any ``field.EventFamily`` runs here unchanged.

A record's ``family_hash`` covers what the family computes from what it reads
(the scored heads' times, values, surprisals and ranks, the field at those
heads, the scope statistics and the scored time); ``full_hash`` adds the field
read at every scored release. Times enter both shifted back by the transform's
``time_offset``.
"""
from __future__ import annotations

import hashlib
from importlib import import_module
import json
import math
import traceback
from typing import Any

import numpy as np

from ..case import Chart, Condition, Context, EvalCase, Scope, Skeleton, Spans, Timing, evaluate
from ..beats import event_times
from ..legality import violations
from ..osu_text import round_trip
from .access import load_chart
from .arrays import Objects
from .chain import GIVEN_FRACTION, HEAD, RELEASE, FieldOperator, continuation_cut, window_chart, windows
from .descriptions import density_log_ratio, describe
from .injections import CONTINUATION, FAMILIES, WHOLE, Source, dose_label
from .transforms import TRANSFORMS
from .trivial import fold_of

SCOPES = (WHOLE, CONTINUATION)
CONDITIONS = ('timing', 'skeleton')
INJECTION_STARS = (2.0, 7.0)
DEFAULT_FAMILY = 'ensomi_model.evaluation.harness.trivial:HeadRate'
COV_EDGES = np.linspace(-0.2, 0.2, 801)
_WORKER: dict[str, Any] = {}


def load_family(spec: str = DEFAULT_FAMILY):
    """The family named by ``module:attribute``: an ``EventFamily``, or a callable returning one."""
    module, _, attribute = spec.partition(':')
    obj = getattr(import_module(module), attribute)
    if isinstance(obj, type) or (callable(obj) and not hasattr(obj, 'values')):
        obj = obj()
    return obj


def init_worker(spec: str = DEFAULT_FAMILY, model=None) -> None:
    """Worker initializer: the family and, once fitted, its model, shared by every case of the process."""
    _WORKER['family'] = load_family(spec)
    _WORKER['model'] = model
    _WORKER['operators'] = {}


def family():
    if 'family' not in _WORKER:
        init_worker()
    return _WORKER['family']


def operator_for(fold: int) -> FieldOperator:
    """The family with the model and reference without ``fold``."""
    ops = _WORKER['operators']
    if fold not in ops:
        ops[fold] = FieldOperator(family(), _WORKER['model'], fold)
    return ops[fold]


def song_span(chart: Chart, grid) -> tuple[float, float]:
    """The song's time as the harness knows it: from 0 (or the first head, if earlier) to one canonical
    bar past the last event. Audio length is not read."""
    times = event_times(chart.objects)
    last = grid.segments[-1]
    return min(0.0, float(times.min())), float(times.max()) + last.meter * last.canonical_beat_length_ms


def make_case(chart: Chart, grid, scope: str, star: float, *, t: float, source: Chart | None = None,
              skeleton: bool = False) -> EvalCase:
    timing = [Skeleton.from_chart(source or chart, grid) if skeleton else Timing(grid)]
    if scope == WHOLE:
        return EvalCase(chart, Condition.of(*timing, star=star), Scope.whole())
    given = Context.from_chart(source or chart, Spans.before(t))
    return EvalCase(chart, Condition.of(*timing, given, star=star), Scope.continuation(t))


def _r(x, decimals: int = 9) -> np.ndarray:
    return np.round(np.asarray(x, dtype=np.float64), decimals) + 0.0


def summarize(out: dict, offset: float = 0.0) -> dict[str, Any]:
    """Scalars and hashes of one operator output (see the module docstring)."""
    st = out['stats']
    sc = out['scored']
    kind, readout = out['event_kind'], out['readout']
    finite = np.isfinite(readout).all(axis=1) if len(readout) else np.zeros(0, dtype=bool)
    family = hashlib.sha256()
    for a in (out['time'][sc] - offset, _r(out['value'][sc]), _r(out['s'][sc]), _r(out['u_low'][sc], 12),
              _r(out['u_high'][sc], 12), _r(readout[kind == HEAD])):
        family.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    scalars = [None if not np.isfinite(v) else round(float(v), 9) for v in list(st['mean']) + list(st['worst'])]
    family.update(json.dumps([scalars, round(st['seconds'], 6),
                              [[a - offset, b - offset] for a, b in out['domain']]]).encode())
    full = hashlib.sha256(family.digest())
    full.update(np.ascontiguousarray(out['event_time'] - offset, dtype=np.float64).tobytes())
    full.update(np.ascontiguousarray(kind, dtype=np.int8).tobytes())
    full.update(np.ascontiguousarray(_r(readout), dtype=np.float64).tobytes())
    return dict(status='ok', mean_low=float(st['mean'][0]), mean_high=float(st['mean'][1]),
                worst_low=float(st['worst'][0]), worst_high=float(st['worst'][1]), seconds=float(st['seconds']),
                heads=int(len(out['time'])), scored_heads=int(sc.sum()), ranked_heads=int(out['ranked'].sum()),
                scored_events=int(len(kind)), scored_events_with_score=int(finite.sum()),
                scored_releases=int((kind == RELEASE).sum()),
                releases_with_score=int((finite & (kind == RELEASE)).sum()),
                family_hash=family.hexdigest()[:16], full_hash=full.hexdigest()[:16])


def run_case(case: EvalCase, operator, offset: float = 0.0, keep: bool = False) -> tuple[dict, dict | None]:
    """``evaluate`` with one operator; the record, and the raw output when ``keep``."""
    result = evaluate(case, [operator])['results'][operator.name]
    if 'skipped' in result:
        return dict(status='skipped', reason=result['skipped']), None
    return summarize(result, offset), (result if keep else None)


def covariance(source: dict, stretched: dict, factor: float) -> dict[str, Any]:
    """Per-head rate of the stretched chart over the source's, times the stretch factor (1 when the rate
    scales by exactly 1/factor); heads matched in time order."""
    v0, v1 = source['value'], stretched['value']
    if len(v0) != len(v1) or not len(v0):
        return dict(cov_n=0)
    log_ratio = v1 - v0 + math.log(factor)
    hist = np.histogram(np.clip(log_ratio, COV_EDGES[0], COV_EDGES[-1]), bins=COV_EDGES)[0].astype(np.int32)
    return dict(cov_n=int(len(v0)), cov_hist=hist, cov_median=float(np.exp(np.median(log_ratio))),
                cov_within_1pct=float(np.mean(np.abs(np.exp(log_ratio) - 1) <= 0.01)))


def _error(exc: Exception) -> str:
    return f'{type(exc).__name__}: {exc} @ {traceback.extract_tb(exc.__traceback__)[-1].lineno}'


def fit_values(task: tuple[dict, int]) -> dict[str, Any]:
    """First fit pass: sufficient statistics of the family's values, descriptions, legality as read."""
    row, fold = task
    out: dict[str, Any] = dict(path=row['path'])
    try:
        chart = load_chart(row)
        fam = family()
        values = fam.values(EvalCase(chart, Condition.of(star=row['star']), Scope.whole()))
        out['suff'] = fam.sufficient(values, fold)
        out['heads'] = int(len(values))
        grid = chart.musical_grid()[0]
        desc = describe(chart, grid)
        desc['density_log_ratio'] = density_log_ratio(chart, grid, continuation_cut(chart))
        out['desc'] = desc
        out['legal'] = violations(chart.objects, song_span=song_span(chart, grid))
    except Exception as exc:  # noqa: BLE001 - recorded per chart and reported
        out['error'] = _error(exc)
    return out


def _stats(st: dict) -> dict[str, float]:
    return dict(mean_low=float(st['mean'][0]), mean_high=float(st['mean'][1]), worst_low=float(st['worst'][0]),
                worst_high=float(st['worst'][1]), seconds=float(st['seconds']))


def fit_null(task: tuple[dict, int]) -> dict[str, Any]:
    """Second fit pass: the song scored cross-fitted, whole and in every continuation window."""
    row, fold = task
    out: dict[str, Any] = dict(path=row['path'])
    try:
        op = operator_for(fold)
        chart = load_chart(row)
        condition = Condition.of(star=row['star'])
        out['whole'] = _stats(op(EvalCase(chart, condition, Scope.whole()))['stats'])
        out['windows'] = []
        for lam, rho, a, b in windows(chart):
            part = chart if lam == 1.0 else window_chart(chart, a, b)
            cut = a + GIVEN_FRACTION * (b - a)
            st = op(EvalCase(part, condition, Scope.continuation(cut)))['stats']
            out['windows'].append(dict(fraction=lam, position=rho, **_stats(st)))
    except Exception as exc:  # noqa: BLE001
        out['error'] = _error(exc)
    return out


def _same(a: Chart, b: Chart) -> bool:
    key = lambda o: (o.start_time_ms, o.lane, o.end_time_ms, o.kind.value)  # noqa: E731
    return sorted(map(key, a.objects)) == sorted(map(key, b.objects))


def calibration_chart(task: tuple[dict, dict]) -> dict[str, Any]:
    row, cfg = task
    op = operator_for(fold_of(row['group_id']))
    records: list[dict] = []
    out: dict[str, Any] = dict(path=row['path'], records=records)
    identity: dict[tuple[str, str], dict] = {}

    def add(kind, name, dose, scope, condition, result, **extra):
        rec = dict(kind=kind, name=name, dose=dose, scope=scope, condition=condition, **result, **extra)
        ref = identity.get((scope, condition))
        if ref is not None and rec.get('status') == 'ok' and ref.get('status') == 'ok' and kind != 'identity':
            rec['same_family'] = rec['family_hash'] == ref['family_hash']
            rec['same_full'] = rec['full_hash'] == ref['full_hash']
        records.append(rec)
        return rec
    try:
        star = row['star']
        chart = load_chart(row)
        grid0 = chart.musical_grid()[0]
        t0 = continuation_cut(chart)
        for scope in SCOPES:
            add('original', 'original', '', scope, 'timing', run_case(make_case(chart, grid0, scope, star, t=t0), op)[0])
        desc = describe(chart, grid0)
        desc['density_log_ratio'] = density_log_ratio(chart, grid0, t0)
        out['desc'] = desc
        src = round_trip(chart)
        grid = src.musical_grid()[0]
        span = song_span(src, grid)
        t = continuation_cut(src)
        src_violations = violations(src.objects, song_span=span)
        out['legal'] = src_violations
        raw_identity = None
        for scope in SCOPES:
            rec, raw = run_case(make_case(src, grid, scope, star, t=t), op, keep=scope == WHOLE)
            identity[(scope, 'timing')] = add('identity', 'identity', '', scope, 'timing', rec)
            raw_identity = raw if scope == WHOLE else raw_identity
            rec, _ = run_case(make_case(src, grid, scope, star, t=t, skeleton=True), op)
            identity[(scope, 'skeleton')] = add('identity', 'identity', '', scope, 'skeleton', rec)
        for i, transform in enumerate(TRANSFORMS):
            _transform(transform, src, star, cfg['seed'] + 1000 * i + cfg['chart_seed'] % 1000, add, op, raw_identity)
        if cfg['inject']:
            if src_violations:
                out['inject_skipped'] = 'source chart illegal after round trip'
            else:
                donor = None
                if cfg.get('donor') is not None:
                    donor_chart = round_trip(load_chart(cfg['donor']))
                    donor = (donor_chart, donor_chart.musical_grid()[0])
                modes = cfg['modes']
                source = Source(src, grid, star, mode_rate=lambda k: modes.get(int(k)), donor=donor, span=span)
                _inject(source, span, t, cfg, add, op)
    except Exception as exc:  # noqa: BLE001
        out['error'] = _error(exc)
    return out


def _transform(transform, src: Chart, star: float, seed: int, add, op, raw_identity) -> None:
    def every(status: dict, **extra):
        for scope in SCOPES:
            for condition in CONDITIONS:
                add('transform', transform.name, '', scope, condition, status, **extra)
    try:
        made = transform.apply(src, seed)
        if isinstance(made, str):
            every(dict(status='n/a', reason=made))
            return
        rt = round_trip(made.chart, **made.extras)
        g = made.grid if made.grid is not None else rt.musical_grid()[0]
        bad = violations(rt.objects, song_span=song_span(rt, g))
        if bad:
            every(dict(status='discarded', reason=_codes(bad)))
            return
        tt = continuation_cut(rt)
        for scope in SCOPES:
            for condition in CONDITIONS:
                keep = transform.stretch is not None and scope == WHOLE and condition == 'timing'
                rec, raw = run_case(make_case(rt, g, scope, star, t=tt, skeleton=condition == 'skeleton'), op,
                                    offset=made.time_offset, keep=keep)
                if raw is not None and raw_identity is not None:
                    rec.update(covariance(raw_identity, raw, transform.stretch))
                add('transform', transform.name, '', scope, condition, rec)
    except Exception as exc:  # noqa: BLE001
        every(dict(status='error', reason=_error(exc)))


def _codes(v: dict[str, int]) -> str:
    return ','.join(f'{k}:{n}' for k, n in sorted(v.items()))


def splice(source: Objects, injected: Objects, t: float, in_place: bool) -> tuple[Objects, int]:
    """The source's given part (heads before ``t``) and the injected chart's scored part (heads at or after).

    A scored object that a given hold still covers is a seam conflict: for a
    family that edits objects in place it goes back to its source state when
    the source had its head in the scored part, otherwise it is removed.
    Returns the chart and the number of conflicts repaired.
    """
    given = source.take(source.start < t)
    picked = np.flatnonzero(injected.start >= t)
    scored = injected.take(picked)
    reach = np.full(8, -np.inf)
    occupied = np.where(given.hold, given.end, given.start)
    for lane in np.unique(given.lane):
        reach[lane] = occupied[given.lane == lane].max()
    bad = scored.start <= reach[np.clip(scored.lane, 0, 7)]
    if not bad.any():
        return Objects.concat([given, scored]), 0
    keep = ~bad
    parts = [given, scored.take(keep)]
    if in_place and len(injected) == len(source):
        back = picked[bad]
        back = back[source.start[back] >= t]
        parts.append(source.take(back))
    return Objects.concat(parts), int(bad.sum())


def _inject(source: Source, span, t: float, cfg: dict, add, op) -> None:
    src = source.chart
    src_objects = source.objects
    for fi, family in enumerate(FAMILIES):
        conditions = CONDITIONS if family.skeleton == 'applicable' else ('timing',)
        for di, dose in enumerate(family.doses):
            label = dose_label(dose)
            rng = np.random.default_rng([cfg['seed'], fi, di, cfg['chart_seed']])
            try:
                injected = family.apply(source, dose, rng)
            except Exception as exc:  # noqa: BLE001
                for scope in family.scopes:
                    add('injection', family.name, label, scope, 'timing', dict(status='error', reason=_error(exc)))
                continue
            if injected is None:
                for scope in family.scopes:
                    add('injection', family.name, label, scope, 'timing',
                        dict(status='n/a', reason='no donor or no scored part'))
                continue
            note = injected.note or {}
            extra = dict(requested=note.get('requested'), done=note.get('done'))
            for scope in family.scopes:
                try:
                    if scope == WHOLE:
                        objs, repaired = injected, 0
                    else:
                        objs, repaired = splice(src_objects, injected, t, family.in_place)
                    rt = round_trip(Chart(objs.to_objects(), src.red_lines, src.keys, src.source))
                    bad = violations(rt.objects, song_span=span)
                    if bad:
                        add('injection', family.name, label, scope, 'timing',
                            dict(status='discarded', reason=_codes(bad)), seam_repaired=repaired, **extra)
                        continue
                    noop = _same(rt, src)
                    desc = math.nan
                    if family.name == 'F4':
                        desc = density_log_ratio(rt, None, t)
                    elif scope == WHOLE:
                        desc = describe(rt, source.grid)[family.description_of(dose)]
                    for condition in conditions:
                        rec, _ = run_case(make_case(rt, source.grid, scope, source.star, t=t, source=src,
                                                    skeleton=condition == 'skeleton'), op)
                        add('injection', family.name, label, scope, condition, rec, noop=noop, desc=desc,
                            seam_repaired=repaired, **extra)
                except Exception as exc:  # noqa: BLE001
                    add('injection', family.name, label, scope, 'timing', dict(status='error', reason=_error(exc)))
