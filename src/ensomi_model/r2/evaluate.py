"""In-run evaluation of plan v4 section 9: teacher-forced manifests, counterfactual response,
representation probe, free-run panels, own-history calibration and the G3 diagnostics.

``Evaluator.teacher_forced`` (every checkpoint) reads the natural manifest (the selection
primary) and the condition manifests (per kind, per factor, per stratum, null contrast,
masked in-scope decisions, train-versus-dev onset NLL, the counterfactual response and the
representation probe). ``Evaluator.panels`` (full evaluations) runs the free-run panels of
section 9.4, the calibration of 9.5 and G3 (a1), (a2), (b), (f) of 9.7, plus (c) when asked.
With ``conditions=False`` (phase N of plan v5, a model that reads no condition) only the
natural parts run: the natural manifest, natural from BOS, the prefix panel (guard (i)), the
calibration (guard (iii)), legality, defects and G3 (c).
``min_hold_ms`` optionally changes only natural BOS and prefix-natural sampling;
their records include the threshold and decision/fallback counts. Teacher-forced,
calibration, conditioned and G3 sampling retain their existing decoding.
Every generation writes its record (``generate.generation_record``) to ``records.jsonl``.
All readouts go through ``properties``; random seeds are 954-956 unless stated.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from .common import ACTIONS
from .data import Corpus, track_from_json
from .features import Chart, Interval
from .generate import chart_seed_record, defects, generate, generation_record
from .locality import reads, scope_decisions
from .properties import ln_share, prefix_objects, scope_contains
from .request_set import Request, RequestSet, Target, effective_track
from .sampling import continue_chart

SEEDS = (954, 955, 956)
A_PANEL = (  # recheck panel, artifacts/r2-recheck-20261005/control/manifest.json "panel"
    '0f6ab03bf173e0708fda5173892e5197725cb36e0e5589cd8800227b17b4978d',
    '4cd643f8a13ec4198834458e81204318e5d76956533b67bff7aba4d0d0b84f82',
    '999daedcbcac1562cefb73fee689c52cecd5c72fb17d4cfe301444a2bb25f4eb',
    'd87bef87c8fd029ed1f494344bed6385626004dbf8fc8194153b7008e8faf77a',
    '005b65e1e6ce378baeae0469e979a9ffd0235d3364cc63dbaf381ea83e7d14ca',
    '01d33d1c69b85afbb2c499ed370d3cd9f5b15210c575cdea18545d7831087668',
    '02f5e44e904efeacfaa55c377afe0a1d89a07b5459ce4bcb00cdd873231f6999',
    '031c25e648a1ee9f65f71037d825637b55843fce0abf39342daf7def75750614')
ONSET_VALUES = (0.05, 0.3, 0.6, 0.9)
WHOLE_VALUES = (0.0, 0.1, 0.3, 0.6, 0.9)
RESIDUALS = (-0.5, -0.25, 0.0, 0.25, 0.5)
BOOT = 2000


# ---- statistics ---------------------------------------------------------------------------------

def chart_bootstrap(groups, fn, rng, n=BOOT):
    """SE of ``fn(selected rows)`` under resampling of charts (``groups``: {chart: [row indices]})."""
    keys = list(groups)
    vals = []
    for _ in range(n):
        pick = rng.choice(len(keys), len(keys), replace=True)
        rows = [i for j in pick for i in groups[keys[j]]]
        v = fn(rows)
        if v is not None and np.isfinite(v):
            vals.append(v)
    return float(np.std(vals, ddof=1)) if len(vals) > 2 else None


def slope_of(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.ptp(x) == 0:
        return None
    return float(np.polyfit(x, y, 1)[0])


def deviation_summary(devs, undefined=0):
    d = np.asarray([v for v in devs if v is not None], float)
    if not len(d):
        return dict(n=0, undefined=undefined)
    return dict(n=len(d), undefined=undefined, mean=float(d.mean()), sd=float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                p10=float(np.quantile(d, .1)), p50=float(np.median(d)), p90=float(np.quantile(d, .9)),
                mad=float(np.abs(d).mean()))


def following(rows, rng, key='target'):
    """Slope and MAE of realised against requested over defined rows, with chart-bootstrap SEs."""
    ok = [i for i, r in enumerate(rows) if r['readout'] is not None]
    groups = {}
    for i in ok:
        groups.setdefault(rows[i]['sha'], []).append(i)

    def slope(sel):
        return slope_of([rows[i][key] for i in sel], [rows[i]['readout'] for i in sel])

    def mae(sel):
        return float(np.mean([abs(rows[i]['readout'] - rows[i][key]) for i in sel])) if sel else None
    return dict(slope=slope(ok), slope_se=chart_bootstrap(groups, slope, rng), mae=mae(ok),
                mae_se=chart_bootstrap(groups, mae, rng), runs=len(rows),
                deviation=deviation_summary([r['readout'] - r[key] for r in rows if r['readout'] is not None],
                                            sum(r['readout'] is None for r in rows)))


def paired(diffs_by_chart, rng):
    keys = [k for k, v in diffs_by_chart.items() if v]
    if not keys:
        return dict(mean=None, se=None, charts=0)
    means = np.array([np.mean(diffs_by_chart[k]) for k in keys])
    boot = [means[rng.choice(len(keys), len(keys), replace=True)].mean() for _ in range(BOOT)]
    return dict(mean=float(means.mean()), se=float(np.std(boot, ddof=1)), charts=len(keys))


# ---- organisation statistics (G3) -----------------------------------------------------------------

def organisation(objects, a, b, T):
    """Chord-size histogram, same-lane repeat, directional four-note runs, fixed-pair alternation,
    outer-lane and left-hand head shares over the heads of the scope."""
    heads = sorted([o for o in objects if bool(scope_contains(a, b, T, o.start))], key=lambda o: (o.start, o.lane))
    if not heads:
        return None
    rows = {}
    for o in heads:
        rows.setdefault(o.start, set()).add(o.lane)
    times = sorted(rows)
    sizes = np.array([len(rows[t]) for t in times])
    chord = np.bincount(np.minimum(sizes, 4), minlength=5)[1:] / len(sizes)
    rep = sum(len(rows[t1] & rows[t0]) for t0, t1 in zip(times, times[1:]))
    base = sum(len(rows[t]) for t in times[1:])
    singles = [next(iter(rows[t])) if len(rows[t]) == 1 else None for t in times]
    stream = trill = windows = 0
    for i in range(len(singles) - 3):
        w = singles[i:i + 4]
        if any(x is None for x in w):
            continue
        windows += 1
        d = np.diff(w)
        stream += bool((d > 0).all() or (d < 0).all())
        trill += bool(w[0] == w[2] and w[1] == w[3] and w[0] != w[1])
    lanes = np.array([o.lane for o in heads])
    return dict(chord=chord.tolist(), repeat=rep / base if base else None,
                stream=stream / windows if windows else None, trill=trill / windows if windows else None,
                outer=float(np.isin(lanes, (0, 3)).mean()), left=float(np.isin(lanes, (0, 1)).mean()),
                ln_share=float(np.mean([o.hold for o in heads])))


def js(p, q):
    p, q = np.asarray(p, float) + 1e-12, np.asarray(q, float) + 1e-12
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    return float(0.5 * (p * np.log(p / m)).sum() + 0.5 * (q * np.log(q / m)).sum())


SCALARS = ('repeat', 'stream', 'trill', 'outer', 'left')


# ---- panels ---------------------------------------------------------------------------------------

def prefix_panel(dev: Corpus, seed: int = 954):
    """A's 8 recheck charts plus two fit_dev charts per star band 2-5 (one at or below the band's median
    length, one above), lengths 50-1,500 rows."""
    have = [s for s in A_PANEL if s in dev.files]
    rng = np.random.default_rng(seed)
    t = dev.table[(dev.table.K >= 50) & (dev.table.K <= 1500) & ~dev.table.sha256.isin(have)].sort_values('sha256')
    extra = []
    for band in (2, 3, 4, 5):
        rows = t[t.band == band]
        if not len(rows):
            continue
        med = rows.K.median()
        for part in (rows[rows.K <= med], rows[rows.K > med]):
            if len(part):
                extra.append(part.sha256.iloc[int(rng.integers(0, len(part)))])
    return have + extra[:16 - len(have)], have


def third_row(chart: Chart, frac: float):
    """Index of the head row nearest frac of the song, and its time."""
    k = int(np.argmin(np.abs(chart.head_ms - frac * chart.song_ms)))
    return k, float(chart.head_ms[k])


def onset_scope(chart: Chart, frac: float, beats: int):
    g = chart.grid
    b0 = float(g.beat(0.0))
    target = float(g.beat(frac * chart.song_ms))
    start_beat = b0 + round((target - b0) / beats) * beats
    a = float(g.time_of_beat(start_beat))
    b = min(float(g.time_of_beat(start_beat + beats)), chart.song_ms)
    return a, b


class Evaluator:
    def __init__(self, model, cfg, manifests, *, star, baseline=None, write_dir=None, g3c=False, conditions=True,
                 min_hold_ms=None):
        self.model, self.cfg, self.m, self.star, self.baseline = model, cfg, manifests, star, baseline
        self.conditions = conditions
        self.min_hold_ms = min_hold_ms
        self.dev = Corpus(cfg.cache, 'fit_dev', star_conditions=False)
        self.write_dir = Path(write_dir) if write_dir else None
        self.g3c = g3c
        self.rng = np.random.default_rng(cfg.seed_validation)
        self._cpu = None
        self.code = None

    @property
    def cpu(self):
        if self._cpu is None:
            self._cpu = copy.deepcopy(self.model).to('cpu', torch.float32).eval()
        return self._cpu

    # ---- teacher forced ----------------------------------------------------------------------------

    def _nll(self, chart, start, stop, track):
        out = self.model.window(chart, start, stop, track)
        return out, (-(out.action + out.release)).double().cpu().numpy()

    def natural(self):
        per = []
        for w in self.m['natural']['windows']:
            chart = self.dev.chart(w['sha256'])
            _, ell = self._nll(chart, w['start'], w['stop'], ())
            per.append(dict(sha256=w['sha256'], group_id=w['group_id'], nll=float(ell.sum()), decisions=len(ell)))
        total = sum(p['nll'] for p in per) / sum(p['decisions'] for p in per)
        return dict(nll_per_decision=total, windows=per)

    def _v_lists(self, chart, track):
        """{interval index: sorted decisions (whole chart) that read it}."""
        d = chart.derived()
        out = {}
        for j, iv in enumerate(track):
            inside, _, _ = scope_decisions(chart, iv)
            out[j] = [int(k) for k in inside if reads(chart, iv, int(k), d.held[int(k)])]
        return out

    def condition(self, windows, corpus, onset_only=False):
        acc = {}

        def add(key, value):
            s = acc.setdefault(key, [0.0, 0])
            s[0] += value
            s[1] += 1
        outside = {}
        for w in windows:
            chart = corpus.chart(w['sha256'])
            track = track_from_json(w['track'])
            if not track:
                continue
            out, ell = self._nll(chart, w['start'], w['stop'], track)
            gov = (-out.governed_ln).double().cpu().numpy()
            base = None if onset_only else self._nll(chart, w['start'], w['stop'], ())
            vl = self._v_lists(chart, track)
            for i, k in enumerate(out.ks):
                for j, iv in enumerate(track):
                    kind = 'ln' if iv.kind == 0 else 'star'
                    lst = vl[j]
                    if k in lst:
                        pos = lst.index(int(k))
                        stratum = 'onset' if pos < 16 else 'end' if pos >= len(lst) - 16 else 'middle'
                        if onset_only and stratum != 'onset':
                            continue
                        add(f'{kind}/{stratum}/whole', float(ell[i]))
                        if kind == 'ln' and not out.eos[i]:
                            add(f'{kind}/{stratum}/governed', float(gov[i]))
                        if base is not None:
                            add(f'{kind}/{stratum}/null_contrast', float(base[1][i] - ell[i]))
                        if kind == 'ln' and stratum == 'onset' and not onset_only:
                            z = self._z(chart, iv, w['start'])
                            if z is not None and z >= 2:
                                add('ln/informative_onset/whole', float(ell[i]))
                                add('ln/informative_onset/governed', float(gov[i]))
                    elif k < chart.K and bool(scope_contains(iv.a, iv.b, chart.song_ms, chart.time(int(k)))):
                        _, onset, _ = scope_decisions(chart, iv)
                        key = f'{kind}/' + ('masked_onset' if int(k) == onset else 'other')
                        outside[key] = outside.get(key, 0) + 1
        return dict(nll={k: dict(mean=v[0] / v[1], n=v[1]) for k, v in sorted(acc.items())},
                    in_scope_outside_v=outside)

    def _z(self, chart, iv, j):
        from .conditions import Candidate, DrawConfig, _weights
        _, z = _weights(chart, [Candidate(iv, 'piece', 0.0)], [j], DrawConfig())
        return None if np.isnan(z[0]) else float(z[0])

    def counterfactual(self, windows, corpus):
        """Same state, value swapped (LN 0 / 0.9; residual -0.5 / +0.5): expected-statistic change and action KL
        on onset and late decisions in V."""
        acc = {}
        for w in windows:
            chart = corpus.chart(w['sha256'])
            track = track_from_json(w['track'])
            vl = self._v_lists(chart, track)
            for j, iv in enumerate(track):
                if iv.kind == 1 and self.cfg.star_value != 'residual':
                    continue
                lst = [k for k in vl[j] if w['start'] <= k < w['stop'] and k < chart.K]
                if not lst:
                    continue
                pos = {k: vl[j].index(k) for k in lst}
                lo, hi = (0.0, 0.9) if iv.kind == 0 else (-0.5, 0.5)
                tr = [list(track), list(track)]
                tr[0][j] = Interval(iv.kind, iv.a, iv.b, lo)
                tr[1][j] = Interval(iv.kind, iv.a, iv.b, hi)
                ks = np.array(lst)
                outs = [self.model.window(chart, int(ks.min()), int(ks.max()) + 1, tuple(t)) for t in tr]
                sel = ks - ks.min()
                p = [o.logp[sel].exp().double().cpu().numpy() for o in outs]
                held = chart.derived().held[ks]
                ln_frac = _ln_fraction(held)
                stat = [(pi * ln_frac).sum(1) if iv.kind == 0 else (pi * _heads(held)).sum(1) for pi in p]
                kl = (p[1] * (np.log(np.maximum(p[1], 1e-300)) - np.log(np.maximum(p[0], 1e-300)))).sum(1)
                for i, k in enumerate(ks):
                    stratum = 'onset' if pos[int(k)] < 16 else 'late'
                    key = f"{'ln' if iv.kind == 0 else 'star'}/{stratum}"
                    s = acc.setdefault(key, dict(delta=[], kl=[]))
                    s['delta'].append(float(stat[1][i] - stat[0][i]))
                    s['kl'].append(float(kl[i]))
        return {k: dict(n=len(v['delta']), delta_mean=float(np.mean(v['delta'])), kl_mean=float(np.mean(v['kl'])))
                for k, v in acc.items()}

    def probe(self, windows, corpus):
        """Ridge (lambda 1) from the hand vectors at onset decisions in V to the frame value; 5-fold CV R^2."""
        xs, ys = {0: [], 1: []}, {0: [], 1: []}
        for w in windows:
            chart = corpus.chart(w['sha256'])
            track = track_from_json(w['track'])
            vl = self._v_lists(chart, track)
            for j, iv in enumerate(track):
                ks = [k for k in vl[j][:16] if w['start'] <= k < w['stop']]
                if not ks:
                    continue
                z = self.model.window_hands(chart, np.array(ks), track).reshape(len(ks), -1).double().cpu().numpy()
                xs[iv.kind] += list(z)
                ys[iv.kind] += [iv.value] * len(ks)
        out = {}
        for kind in (0, 1):
            X, y = np.array(xs[kind]), np.array(ys[kind])
            if len(y) < 10 or np.ptp(y) == 0:
                out['ln' if kind == 0 else 'star'] = dict(n=len(y), r2=None)
                continue
            folds = np.arange(len(y)) % 5
            pred = np.zeros_like(y)
            for f in range(5):
                tr, te = folds != f, folds == f
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
                A = (X[tr] - mu) / sd
                beta = np.linalg.solve(A.T @ A + np.eye(A.shape[1]), A.T @ (y[tr] - y[tr].mean()))
                pred[te] = (X[te] - mu) / sd @ beta + y[tr].mean()
            out['ln' if kind == 0 else 'star'] = dict(n=len(y), r2=float(1 - ((pred - y) ** 2).sum() /
                                                                       ((y - y.mean()) ** 2).sum()))
        return out

    @torch.no_grad()
    def teacher_forced(self):
        self.model.eval()
        t0 = time.time()
        if not self.conditions:
            return dict(natural_manifest=self.natural(), teacher_forced_s=time.time() - t0)
        train = Corpus(self.cfg.cache, 'fit_train', star_conditions=False)
        dev_windows = self.m['condition']['windows']
        cond = self.condition(dev_windows, self.dev)
        onset_train = self.condition(self.m['condition_train']['windows'], train, onset_only=True)
        onset_dev = self.condition(dev_windows, self.dev, onset_only=True)
        tvd = {k: dict(train=onset_train['nll'][k]['mean'], dev=onset_dev['nll'][k]['mean'],
                       dev_minus_train=onset_dev['nll'][k]['mean'] - onset_train['nll'][k]['mean'])
               for k in onset_dev['nll'] if k in onset_train['nll']}
        return dict(natural_manifest=self.natural(), condition_manifest=cond, train_vs_dev_onset=tvd,
                    counterfactual=self.counterfactual(dev_windows, self.dev), probe=self.probe(dev_windows, self.dev),
                    teacher_forced_s=time.time() - t0)

    # ---- free runs ---------------------------------------------------------------------------------

    def _record(self, record):
        if self.write_dir is None:
            return
        self.write_dir.mkdir(parents=True, exist_ok=True)
        with (self.write_dir / 'records.jsonl').open('a') as f:
            f.write(json.dumps(record, default=str) + '\n')

    def run(self, sha, requests: RequestSet, s: int, seed: int, *, stop=None, close_scope=None, panel=''):
        chart = self.dev.chart(sha)
        if self.min_hold_ms is not None and panel in ('natural_bos', 'prefix_natural'):
            stats = {}
            track = effective_track(requests, chart.head_ms, chart.grid, star_value=self.cpu.config.star_value,
                                    baseline=self.baseline)
            acts, gap = continue_chart(self.cpu, chart.head_ms, chart.song_ms, chart.grid,
                                       chart.actions[:s] if s else None, chart.gap[:s] if s else None,
                                       track=track.track, seed=seed, stop=stop, close_scope=close_scope,
                                       min_hold_ms=self.min_hold_ms, min_hold_stats=stats)
            generated = chart.with_decisions(acts, gap)
            record = generation_record(chart=generated, requests=requests, effective=track,
                                       seed_rec=chart_seed_record(chart.head_ms, chart.actions[:s], chart.gap[:s]),
                                       random_seed=seed, model=self.cpu, baseline=self.baseline, code=self.code,
                                       objects=prefix_objects(chart.head_ms, acts, gap),
                                       complete=len(acts) == chart.K + 1)
            record['min_hold'] = dict(ms=self.min_hold_ms, **stats)
        else:
            acts, gap, record = generate(self.cpu, chart.head_ms, chart.song_ms, chart.grid, requests,
                                         seed_actions=chart.actions[:s] if s else None,
                                         seed_gap=chart.gap[:s] if s else None, random_seed=seed, stop=stop,
                                         close_scope=close_scope, baseline_model=self.baseline, code=self.code,
                                         train_config=None)
        record.update(panel=panel, sha256=sha)
        self._record(record)
        return chart, acts, gap, record

    def _request_set(self, chart, specs, frontier):
        rs = RequestSet(chart.song_ms)
        for i, (kind, a, b, v) in enumerate(specs):
            rs.add(Request(f'{kind}-{i}', (a, b), {kind: Target(float(v))}), frontier)
        return rs

    def panels(self):
        from .receipts import code_identity
        self.code = code_identity()
        t0 = time.time()
        rng = np.random.default_rng(self.cfg.seed_validation)
        prefix, a8 = prefix_panel(self.dev)
        out = dict(panel_charts=dict(prefix=prefix, onset=list(a8)))
        legal = dict(runs=0, illegal=0, heads_missing=0)
        defect = dict(holds=0, holds_le60=0, release_1_40=0)

        def account(chart, acts, gap, record):
            legal['runs'] += 1
            if len(acts) == chart.K + 1:
                from .export import decisions_to_objects
                from ..evaluation.legality import violations
                try:
                    objects, _ = decisions_to_objects(chart.head_ms, chart.song_ms, acts, gap)
                    legal['illegal'] += bool(violations(objects, song_span=(0.0, float(chart.song_ms))))
                    starts = np.unique([o.start_time_ms for o in objects])
                    legal['heads_missing'] += not np.array_equal(starts, chart.head_ms)
                except Exception:  # noqa: BLE001
                    legal['illegal'] += 1
            d = record['defects'] or {}
            defect['holds'] += d.get('holds', 0)
            defect['holds_le60'] += d.get('holds_le60', 0)
            defect['release_1_40'] += d.get('release_1_40_before_other_head', 0)

        # natural from BOS
        nat = []
        src_def = dict(holds=0, holds_le60=0, release_1_40=0)
        for sha in prefix:
            src = self.dev.chart(sha)
            src_obj = prefix_objects(src.head_ms, src.actions, src.gap)
            d = defects(src_obj, src.head_ms)
            src_def['holds'] += d['holds']
            src_def['holds_le60'] += d['holds_le60']
            src_def['release_1_40'] += d['release_1_40_before_other_head']
            src_org = organisation(src_obj, 0.0, src.song_ms, src.song_ms)
            for seed in SEEDS:
                chart, acts, gap, rec = self.run(sha, RequestSet(src.song_ms), 0, seed, panel='natural_bos')
                account(chart, acts, gap, rec)
                obj = prefix_objects(chart.head_ms, acts, gap)
                org = organisation(obj, 0.0, chart.song_ms, chart.song_ms)
                third = chart.song_ms / 3
                nat.append(dict(sha=sha, seed=seed, ln_share=ln_share(obj, 0.0, chart.song_ms, chart.song_ms),
                                source=ln_share(src_obj, 0.0, src.song_ms, src.song_ms),
                                drift=_sub(ln_share(obj, 2 * third, chart.song_ms, chart.song_ms),
                                           ln_share(obj, 0.0, third, chart.song_ms)),
                                chord_js=js(org['chord'], src_org['chord']) if org and src_org else None))
        out['natural_bos'] = dict(ln_share_minus_source=paired(_by(nat, lambda r: r['ln_share'] - r['source']), rng),
                                  drift=paired(_by(nat, lambda r: r['drift']), rng),
                                  chord_js_mean=float(np.mean([r['chord_js'] for r in nat if r['chord_js'] is not None])))
        # natural continuation on the prefix panel: guard (i)
        cont = []
        cont_runs = {}
        for sha in prefix:
            src = self.dev.chart(sha)
            k3, t3 = third_row(src, 1 / 3)
            real = ln_share(prefix_objects(src.head_ms, src.actions, src.gap), t3, src.song_ms, src.song_ms)
            for seed in SEEDS:
                chart, acts, gap, rec = self.run(sha, RequestSet(src.song_ms), k3, seed, panel='prefix_natural')
                account(chart, acts, gap, rec)
                obj = prefix_objects(chart.head_ms, acts, gap)
                cont_runs[(sha, seed)] = obj
                cont.append(dict(sha=sha, seed=seed, gen=ln_share(obj, t3, chart.song_ms, chart.song_ms), real=real))
        diff = paired(_by(cont, lambda r: _sub(r['gen'], r['real'])), rng)
        gen_means = [np.mean([r['gen'] for r in cont if r['sha'] == s and r['gen'] is not None]) for s in prefix]
        reals = [next(r['real'] for r in cont if r['sha'] == s) for s in prefix]
        sd_ratio = float(np.std(gen_means, ddof=1) / np.std(reals, ddof=1)) if len(prefix) > 2 else None
        out['prefix_natural'] = dict(mean_difference=diff, sd_ratio=sd_ratio,
                                     guard_i=bool(diff['mean'] is not None and abs(diff['mean']) <= 0.05),
                                     sd_ratio_flag=bool(sd_ratio is not None and not 0.5 <= sd_ratio <= 2.0))
        if self.conditions:
            self.condition_panels(out, a8, rng, account)
        out['calibration'] = self.calibration(prefix)
        out['calibration']['guard_iii'] = bool(out['calibration']['gap'] is not None
                                               and abs(out['calibration']['gap']) <= 0.05)
        if self.conditions:
            out['g3'] = self.g3(prefix, cont_runs, rng, account)
        elif self.g3c:
            out['g3'] = dict(c=self.g3_c(prefix, rng))
        out['legality'] = legal
        out['source_defects'] = dict(src_def, release_1_40_rate=src_def['release_1_40'] / src_def['holds']
                                     if src_def['holds'] else None)
        out['defects'] = dict(defect, holds_le60_rate=defect['holds_le60'] / defect['holds'] if defect['holds'] else None,
                              release_1_40_rate=defect['release_1_40'] / defect['holds'] if defect['holds'] else None)
        out['panels_s'] = time.time() - t0
        return dict(panels=out)

    def condition_panels(self, out, a8, rng, account):
        """Span following at onsets (guard (ii)), whole-song LN, the half-song switch, residual star."""
        # span following at onsets: guard (ii)
        rows = []
        for sha in a8:
            src = self.dev.chart(sha)
            for beats in (16, 32):
                for frac in (1 / 3, 2 / 3):
                    a, b = onset_scope(src, frac, beats)
                    s = int(np.searchsorted(src.head_ms, a, side='left'))
                    if s == 0 or b <= a:
                        continue
                    for v in ONSET_VALUES:
                        for seed in SEEDS:
                            rs = self._request_set(src, [('ln_share', a, b, v)], float(src.head_ms[s - 1]))
                            chart, acts, gap, rec = self.run(sha, rs, s, seed, close_scope=(a, b), panel='onset')
                            account(chart, acts, gap, rec)
                            t = rec['targets'][0]
                            rows.append(dict(sha=sha, seed=seed, beats=beats, target=v, readout=t['readout'],
                                             masked_onset=t['rule_l']['onset_masked']))
        out['onset'] = dict(all=following(rows, rng),
                            by_length={L: following([r for r in rows if r['beats'] == L], rng) for L in (16, 32)},
                            masked_onset_share=float(np.mean([r['masked_onset'] for r in rows])) if rows else None)
        o = out['onset']['all']
        out['onset']['guard_ii'] = bool(o['slope'] is not None and o['slope'] >= 0.7 and o['mae'] is not None
                                        and o['mae'] <= 0.15)
        # whole-song LN
        rows = []
        for sha in a8:
            src = self.dev.chart(sha)
            for v in WHOLE_VALUES:
                for seed in SEEDS:
                    rs = self._request_set(src, [('ln_share', 0.0, src.song_ms, v)], None)
                    chart, acts, gap, rec = self.run(sha, rs, 0, seed, panel='whole')
                    account(chart, acts, gap, rec)
                    rows.append(dict(sha=sha, seed=seed, target=v, readout=rec['targets'][0]['readout']))
        out['whole'] = following(rows, rng)
        # switch at half song, both requests added at 0^-
        sw = {}
        for sha in a8:
            src = self.dev.chart(sha)
            half = src.song_ms / 2
            for order in ((0.1, 0.6), (0.6, 0.1)):
                for seed in SEEDS:
                    rs = self._request_set(src, [('ln_share', 0.0, half, order[0]),
                                                 ('ln_share', half, src.song_ms, order[1])], None)
                    chart, acts, gap, rec = self.run(sha, rs, 0, seed, panel='switch')
                    account(chart, acts, gap, rec)
                    t = rec['targets']
                    sw.setdefault((sha, seed), {})[order] = (t[0]['readout'], t[1]['readout'],
                                                             t[1]['rule_l']['onset_masked'])
        did, err = {}, []
        for (sha, seed), r in sw.items():
            x, y = r[(0.1, 0.6)], r[(0.6, 0.1)]
            if None in x[:2] or None in y[:2]:
                continue
            did.setdefault(sha, []).append(((x[1] - x[0]) - (y[1] - y[0])) / 2)
            err += [abs(x[0] - 0.1), abs(x[1] - 0.6), abs(y[0] - 0.6), abs(y[1] - 0.1)]
        out['switch'] = dict(did_half=paired(did, rng), per_half_mae=float(np.mean(err)) if err else None,
                             boundary_masked_share=float(np.mean([v[o][2] for v in sw.values() for o in v])) if sw else None)
        # residual star (stage 2)
        if self.cfg.star_value == 'residual' and self.baseline is not None:
            rows = []
            for sha in a8:
                src = self.dev.chart(sha)
                for L in (30_000.0, 60_000.0):
                    a = round(src.song_ms / 3 / 10_000.0) * 10_000.0
                    b = a + L
                    s = int(np.searchsorted(src.head_ms, a, side='left'))
                    if b > src.song_ms or s == 0:
                        continue
                    b_s = float(self.baseline.predict(src.head_ms, a, b))
                    for r in RESIDUALS:
                        for seed in SEEDS:
                            rs = self._request_set(src, [('difficulty', a, b, b_s + r)], float(src.head_ms[s - 1]))
                            chart, acts, gap, rec = self.run(sha, rs, s, seed, close_scope=(a, b), panel='star')
                            account(chart, acts, gap, rec)
                            t = rec['targets'][0]
                            from .properties import difficulty_trimmed
                            obj = prefix_objects(chart.head_ms, acts, gap)
                            trim, _ = difficulty_trimmed(obj, a, b, chart.song_ms)
                            rows.append(dict(sha=sha, seed=seed, length=L, target=r,
                                             readout=None if t['readout'] is None else t['readout'] - b_s,
                                             tail=None if t['readout'] is None or trim is None else
                                             abs(t['readout'] - trim)))
            out['residual_star'] = dict(all=following(rows, rng),
                                        by_length={str(int(L / 1000)): following([r for r in rows if r['length'] == L],
                                                                                 rng) for L in (30_000.0, 60_000.0)},
                                        abs_delta_tail=deviation_summary([r['tail'] for r in rows]))

    @torch.no_grad()
    def calibration(self, charts, states=96, horizon=64):
        """Expected LN forecast on real against own-sampled histories from the same start states."""
        diffs = {}
        for sha in charts:
            chart = self.dev.chart(sha)
            if chart.K <= horizon + 1:
                continue
            for s in np.unique(np.linspace(1, chart.K - horizon, states).astype(int)):
                real = _forecast(self.cpu, chart, s, s + horizon)
                acts, gap = continue_chart(self.cpu, chart.head_ms, chart.song_ms, chart.grid, chart.actions[:s],
                                           chart.gap[:s], seed=int(s) + 954, stop=s + horizon)
                own = _forecast(self.cpu, chart.with_decisions(acts, gap), s, s + horizon)
                diffs.setdefault(sha, []).append(own - real)
        res = paired(diffs, np.random.default_rng(955))
        return dict(gap=res['mean'], se=res['se'], charts=res['charts'], states_per_chart=states, horizon=horizon)

    def g3(self, prefix, cont_runs, rng, account):
        out = {}
        # (a1) identity after the exit decision
        ok, checked = True, 0
        for sha in prefix[:4]:
            src = self.dev.chart(sha)
            k3, t3 = third_row(src, 1 / 3)
            _, t2 = third_row(src, 1 / 2)
            pre = ln_share(prefix_objects(src.head_ms, src.actions[:k3], src.gap[:k3]), 0.0, t3, src.song_ms)
            v = 0.9 if (pre or 0.0) < 0.5 else 0.05
            exit_k = int(np.searchsorted(src.head_ms, t2, side='left'))
            for seed in SEEDS:
                rs = self._request_set(src, [('ln_share', t3, t2, v)], float(src.head_ms[k3 - 1]))
                chart, acts, gap, _ = self.run(sha, rs, k3, seed, stop=exit_k, panel='g3_a1_override')
                stop = min(exit_k + 64, chart.K + 1)
                with_ = generate(self.cpu, chart.head_ms, chart.song_ms, chart.grid, rs.copy(), seed_actions=acts,
                                 seed_gap=gap, random_seed=seed + 1, stop=stop)
                without = generate(self.cpu, chart.head_ms, chart.song_ms, chart.grid, RequestSet(chart.song_ms),
                                   seed_actions=acts, seed_gap=gap, random_seed=seed + 1, stop=stop)
                same_rows = np.array_equal(with_[0], without[0]) and np.array_equal(with_[1], without[1],
                                                                                    equal_nan=True)
                c = chart.with_decisions(with_[0], with_[1])
                from .request_set import effective_track
                eff = effective_track(rs, chart.head_ms, chart.grid, star_value=self.cpu.config.star_value,
                                      baseline=self.baseline).track
                lp_w = self.cpu.window(c, exit_k, stop, eff).total
                lp_o = self.cpu.window(c, exit_k, stop, ()).total
                ok &= bool(same_rows and torch.equal(lp_w, lp_o))
                checked += 1
        out['a1'] = dict(identical=ok, runs=checked)
        # (a2) persistence through history, (f) unspecified against explicit zero, (b) identity under a change
        a2, scope_share, f_diff, b_change, boundary = {}, [], {}, {}, {}
        for sha in prefix:
            src = self.dev.chart(sha)
            k3, t3 = third_row(src, 1 / 3)
            _, t2 = third_row(src, 1 / 2)
            _, t23 = third_row(src, 2 / 3)
            pre = ln_share(prefix_objects(src.head_ms, src.actions[:k3], src.gap[:k3]), 0.0, t3, src.song_ms)
            v = 0.9 if (pre or 0.0) < 0.5 else 0.05
            g0 = float(src.head_ms[k3 - 1])
            for seed in SEEDS:
                nat = cont_runs[(sha, seed)]
                chart, acts, gap, rec = self.run(sha, self._request_set(src, [('ln_share', t3, t2, v)], g0), k3, seed,
                                                 panel='g3_a2')
                account(chart, acts, gap, rec)
                ov = prefix_objects(chart.head_ms, acts, gap)
                a2.setdefault(sha, []).append(_sub(ln_share(ov, t2, chart.song_ms, chart.song_ms),
                                                   ln_share(nat, t2, chart.song_ms, chart.song_ms)))
                scope_share.append(rec['targets'][0]['readout'])
                e = int(np.searchsorted(chart.head_ms, t2, side='left'))
                t_end = chart.time(min(e + 32, chart.K))
                for name, objs in (('override', ov), ('natural', nat)):
                    s_ = boundary.setdefault(name, dict(holds=0, holds_le60=0, release_1_40=0))
                    d = defects([o for o in objs if t2 <= o.start < t_end], chart.head_ms)
                    s_['holds'] += d['holds']
                    s_['holds_le60'] += d['holds_le60']
                    s_['release_1_40'] += d['release_1_40_before_other_head']
                chart, acts, gap, rec = self.run(sha, self._request_set(src, [('ln_share', t3, t23, 0.0)], g0), k3,
                                                 seed, panel='g3_f')
                account(chart, acts, gap, rec)
                zero = prefix_objects(chart.head_ms, acts, gap)
                f_diff.setdefault(sha, []).append(_sub(ln_share(zero, t3, t23, chart.song_ms),
                                                       ln_share(nat, t3, t23, chart.song_ms)))
                chart, acts, gap, rec = self.run(sha, self._request_set(src, [('ln_share', t3, t23, 0.3)], g0), k3,
                                                 seed, panel='g3_b')
                account(chart, acts, gap, rec)
                req = organisation(prefix_objects(chart.head_ms, acts, gap), t3, t23, chart.song_ms)
                base = organisation(nat, t3, t23, chart.song_ms)
                if req and base:
                    for key in SCALARS:
                        if req[key] is not None and base[key] is not None:
                            b_change.setdefault(key, {}).setdefault(sha, []).append(req[key] - base[key])
                    b_change.setdefault('chord_js', {}).setdefault(sha, []).append(js(req['chord'], base['chord']))
        out['a2'] = dict(delta_post=paired(a2, rng), override_scope_share=deviation_summary(scope_share),
                         defects_32_after_boundary=boundary)
        f = paired(f_diff, rng)
        out['f'] = dict(f, distinct=bool(f['mean'] is not None and f['se'] and abs(f['mean']) > 2 * f['se']))
        ref = {}
        for sha in prefix:
            src = self.dev.chart(sha)
            k3, t3 = third_row(src, 1 / 3)
            _, t23 = third_row(src, 2 / 3)
            for s0, s1 in ((954, 955), (955, 956)):
                x = organisation(cont_runs[(sha, s0)], t3, t23, src.song_ms)
                y = organisation(cont_runs[(sha, s1)], t3, t23, src.song_ms)
                if x and y:
                    for key in SCALARS:
                        if x[key] is not None and y[key] is not None:
                            ref.setdefault(key, {}).setdefault(sha, []).append(abs(x[key] - y[key]))
                    ref.setdefault('chord_js', {}).setdefault(sha, []).append(js(x['chord'], y['chord']))
        out['b'] = {}
        for key in list(SCALARS) + ['chord_js']:
            ch = paired(b_change.get(key, {}), rng)
            rf = paired(ref.get(key, {}), rng)
            kept = None
            if ch['mean'] is not None and rf['mean'] is not None and rf['se'] is not None:
                kept = bool(abs(ch['mean']) < rf['mean'] + 2 * rf['se'])
            out['b'][key] = dict(change=ch, reference_abs_difference=rf, kept=kept)
        if self.g3c:
            out['c'] = self.g3_c(prefix, rng)
        return out

    def g3_c(self, prefix, rng, seeds=tuple(range(954, 962))):
        stats = {}
        for sha in prefix:
            src = self.dev.chart(sha)
            k3, t3 = third_row(src, 1 / 3)
            for seed in seeds:
                chart, acts, gap, _ = self.run(sha, RequestSet(src.song_ms), k3, seed, panel='g3_c')
                org = organisation(prefix_objects(chart.head_ms, acts, gap), t3, chart.song_ms, chart.song_ms)
                if org:
                    for key in list(SCALARS) + ['ln_share']:
                        if org[key] is not None:
                            stats.setdefault(key, {}).setdefault(sha, []).append(org[key])
        out = {}
        for key, by in stats.items():
            keys = [k for k, v in by.items() if len(v) > 1]
            if len(keys) < 3:
                continue

            def ratio(sel):
                within = np.mean([np.var(by[keys[i]], ddof=1) for i in sel])
                between = np.var([np.mean(by[keys[i]]) for i in sel], ddof=1)
                return within / between if between > 0 else None
            idx = list(range(len(keys)))
            r = ratio(idx)
            boot = [ratio(list(rng.choice(len(keys), len(keys), replace=True))) for _ in range(BOOT)]
            boot = [b for b in boot if b is not None and np.isfinite(b)]
            out[key] = dict(within_over_between=r, se=float(np.std(boot, ddof=1)) if len(boot) > 2 else None)
        return out


def _sub(x, y):
    return None if x is None or y is None else x - y


def _by(rows, fn):
    out = {}
    for r in rows:
        v = fn(r)
        if v is not None:
            out.setdefault(r['sha'], []).append(v)
    return out


def _ln_fraction(held):
    h = np.where(held[:, None, :], ACTIONS[None] >= 3, (ACTIONS[None] == 1) | (ACTIONS[None] == 2)).sum(-1)
    l = np.where(held[:, None, :], ACTIONS[None] == 4, ACTIONS[None] == 2).sum(-1)
    return np.where(h > 0, l / np.maximum(h, 1), 0.0)


def _heads(held):
    return np.where(held[:, None, :], ACTIONS[None] >= 3, (ACTIONS[None] == 1) | (ACTIONS[None] == 2)).sum(-1)


@torch.no_grad()
def _forecast(model, chart, lo, hi):
    """Expected LN heads / expected heads over decisions [lo, hi) given the chart's own history."""
    out = model.window(chart, lo, hi, ())
    p = out.logp.exp().double().numpy()
    held = chart.derived().held[out.ks]
    heads = (p * _heads(held)).sum(1)
    lns = (p * np.where(held[:, None, :], ACTIONS[None] == 4, ACTIONS[None] == 2).sum(-1)).sum(1)
    keep = ~out.eos
    return float(lns[keep].sum() / heads[keep].sum())
