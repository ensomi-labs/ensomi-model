"""P5a repetition retrieval, P5b recurrence and P5c downbeat evidence."""
from collections import defaultdict
import json

import numpy as np

from .common import (ROOT, OUT, auc, by_tempo, dump, features, interp, interval, labels, log, manifests,
                     nearest, norm, prereg_check, stable_seed, units)
from .tokens import bar_tokens, context, token_load
from .cells import decoded


def runs(mask):
    q = np.flatnonzero(mask)
    if not len(q):
        return []
    return np.split(q, np.flatnonzero(np.diff(q) > 1) + 1)


def regions(z, p=None):
    tb = np.unique(z['bars'][z['y'] == 1])
    groups = np.split(tb, np.flatnonzero(np.diff(tb) > 2) + 1) if len(tb) else []
    result = []
    for group in groups:
        mask = (z['bars'] >= group[0]) & (z['bars'] <= group[-1])
        if p is None or np.any(mask & (z['y'] == 1) & (p < .5)):
            result.append(mask)
    return result


def pattern_matrix(z):
    p = z['patterns'].astype('float32')
    inter = p @ p.T
    union = p.sum(axis=1)[:, None] + p.sum(axis=1)[None, :] - inter
    jac = inter / np.maximum(union, 1)
    exact = (jac == 1) & (z['bar_meter'][:, None] == z['bar_meter'][None, :])
    return exact, jac


def aggregate(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for key, value in row.items():
            if key not in ('rec', 'unit') and value is not None and np.isfinite(value):
                grouped[key][row['rec']].append(value)
    return {k: dict(**interval([np.mean(v) for v in g.values()]),
                    recordings={r: float(np.mean(v)) for r, v in g.items()}) for k, g in grouped.items()}


def retrieval(seed):
    prereg_check()
    rows = []
    pairs_by_recording = defaultdict(lambda: defaultdict(list))
    queries_by_recording = defaultdict(lambda: defaultdict(list))
    for i, u in enumerate(units()):
        if u['cohort'] not in ('panel', 'val_hard'):
            continue
        z, t = labels(u), token_load(seed, u)
        ix = np.flatnonzero(z['nontrivial'])
        if len(ix) < 2:
            continue
        exact, jac = pattern_matrix(z)
        exact, jac = exact[np.ix_(ix, ix)], jac[np.ix_(ix, ix)]
        lag = abs(z['bar_ids'][ix, None] - z['bar_ids'][None, ix])
        rng = np.random.default_rng(stable_seed('P5a:' + u['rec']))
        score = {k: norm(bar_tokens(t[k + 'z'], z)[ix]) @ norm(bar_tokens(t[k + 'z'], z)[ix]).T
                 for k in ('F', 'M')}
        score['periodic'] = (lag % 8 == 0).astype(float)
        score['random'] = rng.random(lag.shape)
        bar_time = np.array([z['beats'][np.flatnonzero(z['bars'] == b)[0]] for b in z['bar_ids'][ix]])
        ki = np.zeros(len(ix), bool)
        for a, b in z['kiai']:
            ki |= (bar_time >= a) & (bar_time < b)
        strata = {'all': lag >= 8, 'lag8-15': (lag >= 8) & (lag < 16),
                  'lag16-31': (lag >= 16) & (lag < 32), 'lag32+': lag >= 32,
                  'kiai': (lag >= 8) & ki[:, None] & ki[None, :],
                  'nonkiai': (lag >= 8) & ~ki[:, None] & ~ki[None, :]}
        band = 'ge160' if np.median(60000 / z['length']) >= 160 else 'lt160'
        strata[band] = lag >= 8
        row = dict(rec=u['rec'], unit=u['id'])
        for st, eligible in strata.items():
            for positive, truth in [('exact', exact), ('jaccard80', jac >= .8)]:
                a, b = np.where(np.triu(truth & eligible, 1))
                na, nb = np.where(np.triu((jac <= .3) & eligible, 1))
                order = np.argsort(lag[na, nb], kind='stable')
                na, nb = na[order], nb[order]
                neg_lag = lag[na, nb]
                pos, neg = [], []
                for aa, bb in zip(a, b):
                    ll = lag[aa, bb]
                    low, high = np.searchsorted(neg_lag, [ll * .9, ll * 1.1], side='left')
                    high = np.searchsorted(neg_lag, ll * 1.1, side='right')
                    if high <= low:
                        continue
                    pick = rng.integers(low, high)
                    pos.append((aa, bb)); neg.append((na[pick], nb[pick]))
                if pos:
                    pa, pb = np.array(pos).T
                    na, nb = np.array(neg).T
                    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
                    for name, s in score.items():
                        key = f'{st}:{positive}:auc:{name}'
                        values = np.r_[s[pa, pb], s[na, nb]]
                        pairs_by_recording[key][u['rec']].append((y, values))
                        if len(pos) >= 5:
                            row[key] = auc(y, values)
                    row[f'{st}:{positive}:pairs'] = len(pos)
            for name, s in score.items():
                precision, chance = [], []
                for q in range(len(ix)):
                    targets = np.flatnonzero(eligible[q])
                    if len(targets) < 5:
                        continue
                    # A deterministic random tie order keeps the periodic prior
                    # from using the target's temporal order as a second score.
                    targets = rng.permutation(targets)
                    best = targets[np.argsort(-s[q, targets], kind='stable')[:5]]
                    precision.append(exact[q, best].mean())
                    chance.append(exact[q, targets].mean())
                if precision:
                    row[f'{st}:p5:{name}'] = float(np.mean(precision))
                    row[f'{st}:chance'] = float(np.mean(chance))
                    queries_by_recording[f'{st}:p5:{name}'][u['rec']].extend(precision)
                    if name == 'F':
                        queries_by_recording[f'{st}:chance'][u['rec']].extend(chance)
        rows.append(row)
        if i % 50 == 0:
            log('P5a_progress', seed=seed, units=i)
    summary = aggregate(rows)
    for key, groups in pairs_by_recording.items():
        values = {}
        for rec, chunks in groups.items():
            y = np.concatenate([c[0] for c in chunks])
            if sum(y == 1) >= 5:
                values[rec] = auc(y, np.concatenate([c[1] for c in chunks]))
        summary[key] = dict(**interval(list(values.values())), recordings=values)
    for key, groups in queries_by_recording.items():
        values = {rec: float(np.mean(v)) for rec, v in groups.items()}
        summary[key] = dict(**interval(list(values.values())), recordings=values)
    dump(OUT / f'P5a-s{seed}.json', dict(summary=summary, rows=rows))
    log('P5a_done', seed=seed, units=len(rows))


def recurrence(seed):
    prereg_check()
    rows, b_spans = [], []
    for u in units():
        if u['cohort'] not in ('panel', 'val_hard'):
            continue
        z = labels(u)
        p = np.load(OUT / 'm0' / f's{seed}' / (u['id'] + '.npy'))
        exact, _ = pattern_matrix(z)
        lag = abs(z['bar_ids'][:, None] - z['bar_ids'][None, :])
        row = dict(rec=u['rec'], unit=u['id'])
        for tag, regs in [('failing', regions(z, p)), ('allT', regions(z))]:
            if not regs:
                continue
            row[tag + ':tier_i'] = float(len(regs) >= 2)
            repeats = []
            for r in regs:
                source = np.isin(z['bar_ids'], z['bars'][r])
                repeats.append(bool(np.any(exact & (lag >= 8) & source[:, None] & ~source[None, :])))
            row[tag + ':tier_ii'] = float(any(repeats))
            row[tag + ':regions'] = len(regs)
        try:
            d = decoded(seed, u, z)
        except FileNotFoundError:
            rows.append(row)
            continue
        for span in runs(~d['matched']):
            if len(span) < 2:
                continue
            lo = d['beats'][span[0]] - d['offset']
            hi = d['beats'][span[-1]] - d['offset'] + d['length'][span[-1]]
            inside = (z['beats'] >= lo) & (z['beats'] < hi)
            source = np.isin(z['bar_ids'], z['bars'][inside])
            target = np.any(exact & source[:, None] & ~source[None, :], axis=0)
            tracked = []
            for b in z['bar_ids'][target]:
                chart = z['beats'][z['bars'] == b]
                tracked.append(bool(np.all(abs(nearest(chart, d['beats'] - d['offset'])[0]) <= 15)))
            b_spans.append(dict(rec=u['rec'], unit=u['id'], lo_ms=lo, hi_ms=hi,
                                repeat=bool(target.any()), tracked_repeat=bool(any(tracked))))
        rows.append(row)
    br = []
    for rec in sorted({r['rec'] for r in b_spans}):
        spans = [r for r in b_spans if r['rec'] == rec]
        repeated = [r for r in spans if r['repeat']]
        br.append(dict(rec=rec, B_repeat=np.mean([r['repeat'] for r in spans]),
                       B_tracked_any=np.mean([r['tracked_repeat'] for r in spans]),
                       B_tracked_repeat=np.mean([r['tracked_repeat'] for r in repeated]) if repeated else None))
    summary, bsummary = aggregate(rows), aggregate(br)
    dump(OUT / f'P5b-s{seed}.json', dict(summary=summary, B_summary=bsummary,
                                        by_tempo=by_tempo(summary), B_by_tempo=by_tempo(bsummary),
                                        rows=rows, B_spans=b_spans))
    log('P5b_done', seed=seed, spans=len(b_spans))


def downbeat(seed):
    prereg_check()
    rows, cache = [], manifests(seed)
    groups = defaultdict(lambda: defaultdict(list))
    cohort = json.loads((ROOT / 'artifacts/n1-stage0-20261028/events-cohort.json').read_text())
    extra = {r['rec'] for r in cohort['selected']}
    for u in units():
        if u['cohort'] not in ('panel', 'val_hard'):
            continue
        z = labels(u)
        d = decoded(seed, u, z)
        pred_down = d['beats'][d['down']] - d['offset']
        same = np.mean(abs(nearest(z['beats'][z['down']], pred_down)[0]) <= 20)
        if same >= .9 and u['rec'] not in extra:
            continue
        _, logits, _ = features(cache[u['sha']])
        j = np.round(z['beats'] / 10).astype(int)
        down_index = cache[u['sha']].get('head_layout', ['beat', 'half', 'down']).index('down')
        raw = np.max(logits[np.clip(j[:, None] + np.arange(-1, 2), 0, len(logits) - 1), down_index], axis=1)
        t = norm(context(token_load(seed, u)['Fz'], before=0, after=3))
        s = t @ t.T
        ix = np.arange(len(raw))
        lag = abs(ix[:, None] - ix[None, :])
        s[lag < 16] = -np.inf
        rng = np.random.default_rng(stable_seed('P5c:' + u['id']))
        avg, rand, periodic = [], [], []
        for i in ix:
            sel = np.argsort(-s[i])[:5]
            sel = sel[np.isfinite(s[i, sel])]
            avg.append(np.mean(raw[np.r_[i, sel]]))
            matched = []
            for v in sel:
                at = np.flatnonzero(lag[i] == lag[i, v])
                matched.append(rng.choice(at))
            rand.append(np.mean(raw[np.r_[i, matched].astype(int)]))
            per = i + np.r_[-4 * np.arange(4, 9), 4 * np.arange(4, 9)]
            per = per[(per >= 0) & (per < len(raw))]
            periodic.append(np.mean(raw[np.r_[i, per]]))
        rows.append(dict(rec=u['rec'], unit=u['id'], same_bars=same, stage0_cohort=int(u['rec'] in extra),
                         raw=auc(z['down'], raw), feature=auc(z['down'], avg),
                         random=auc(z['down'], rand), periodic=auc(z['down'], periodic)))
        for name, values in dict(y=z['down'], raw=raw, feature=avg, random=rand, periodic=periodic).items():
            groups[u['rec']][name].append(values)
    summary = aggregate(rows)
    for name in ('raw', 'feature', 'random', 'periodic'):
        values = {rec: auc(np.concatenate(g['y']), np.concatenate(g[name])) for rec, g in groups.items()}
        values = {rec: v for rec, v in values.items() if v is not None}
        summary[name] = dict(**interval(list(values.values())), recordings=values)
    dump(OUT / f'P5c-s{seed}.json', dict(summary=summary, by_tempo=by_tempo(summary), rows=rows))


def span_evidence(seed):
    """P5d: feature-selected span alignment; never modifies a decoded line."""
    prereg_check()
    cache = manifests(seed)
    by_id = {u['id']: u for u in units()}
    spans = json.loads((OUT / f'P5b-s{seed}.json').read_text())['B_spans']
    grouped = defaultdict(list)
    for span in spans:
        grouped[span['unit']].append(span)
    rows = []
    for uid, members in grouped.items():
        u = by_id[uid]; z = labels(u)
        d = decoded(seed, u, z)
        t = token_load(seed, u)['Fz']
        _, logits, _ = features(cache[u['sha']])
        raw = logits[:, 0:1]
        for span in members:
            q = np.flatnonzero((z['beats'] >= span['lo_ms']) & (z['beats'] < span['hi_ms']))
            pred = d['beats'][(d['beats'] - d['offset'] >= span['lo_ms']) &
                              (d['beats'] - d['offset'] < span['hi_ms'])]
            if not len(q) or not len(pred):
                continue
            width, start = len(q), int(q[0])
            query = norm(t[q].ravel()[None, :])[0]
            candidates = []
            for anchor in np.flatnonzero(z['down']):
                if abs(anchor - start) < 16:
                    continue
                scores = []
                for shift in range(-2, 3):
                    at = int(anchor + shift)
                    if at < 0 or at + width > len(t) or abs(at - start) < 16:
                        continue
                    token = norm(t[at:at + width].ravel()[None, :])[0]
                    scores.append((float(query @ token), at, shift))
                if scores:
                    candidates.append(max(scores))
            candidates.sort(reverse=True)
            chosen = candidates[:3]
            own_chart = float(interp(raw, z['beats'][q]).mean())
            own_pred = float(interp(raw, pred).mean())
            chart_scores, pred_scores = [own_chart], [own_pred]
            relative_chart = np.arange(width, dtype=float)
            relative_pred = np.interp(pred, z['beats'], np.arange(len(t))) - start
            for _, at, shift in chosen:
                chart_times = np.interp(at + relative_chart, np.arange(len(t)), z['beats'])
                pred_times = np.interp(at + relative_pred, np.arange(len(t)), z['beats'])
                chart_scores.append(float(interp(raw, chart_times).mean()))
                pred_scores.append(float(interp(raw, pred_times).mean()))
            rows.append(dict(rec=u['rec'], unit=uid, lo_ms=span['lo_ms'], hi_ms=span['hi_ms'],
                own_prefers_chart=float(own_chart > own_pred),
                averaged_prefers_chart=float(np.mean(chart_scores) > np.mean(pred_scores)),
                own_difference=own_chart - own_pred,
                averaged_difference=float(np.mean(chart_scores) - np.mean(pred_scores)),
                similar_spans=len(chosen)))
    dump(OUT / f'P5d-s{seed}.json', dict(summary=aggregate(rows), rows=rows, characterizes_only=True,
        alignment='Candidate spans are anchored at chart bar starts; choose offset -2..2 by feature cosine and sample by piecewise-linear notated-beat coordinate.'))
