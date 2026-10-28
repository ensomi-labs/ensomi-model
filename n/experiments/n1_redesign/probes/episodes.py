"""Cell-only hint updates, simulator, learned reach and absolute invariants."""
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from functools import lru_cache

import numpy as np
from scipy.special import expit, logit
from scipy.stats import t as student
import torch
from torch import nn

from .common import OUT, dump, labels, log, norm, prereg_check, stable_seed, units
from .tokens import bar_tokens, context, token_load
from .repeats import aggregate, pattern_matrix, regions, runs

torch.set_num_threads(4)

GAINS = (0., .125, .25, .5, 1., 2.)
THRESHOLDS = (.2, .4, .6, .8)
ARMS = ('mark', 'local', 'graph', 'learned0', 'learned1', 'bias', 'random',
        'mel', 'periodic', 'repeat', 'everywhere')


def pulse_hash(z):
    h = hashlib.sha256()
    for k in ('beats', 'red', 'bars'):
        a = np.ascontiguousarray(z[k])
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


@lru_cache(maxsize=1)
def calibration():
    return json.loads((OUT / 'runtime.json').read_text())['simulator']


def noise_scale():
    return calibration()['residual_robust_sd_ms'] / (1.4826 * student.ppf(.75, 3))


def read_mark(time_ms, z, offset=None):
    if offset is None:
        offset = calibration()['offset_mean_ms']
    t = time_ms - offset
    i = int(np.clip(np.searchsorted(z['beats'], t, 'right') - 1, 0, len(z['beats']) - 1))
    phase = (t - z['beats'][i]) / z['length'][i]
    tern = np.array([1, 2, 4, 5, 7, 8, 10, 11]) / 12
    duple = np.arange(17) / 16
    dt = np.min(abs(phase - tern)) * z['length'][i]
    dd = np.min(abs(phase - duple)) * z['length'][i]
    scale = noise_scale()
    llr = float(student.logpdf(dt / scale, 3) - student.logpdf(dd / scale, 3))
    return dict(beat=i, L=llr, kind=0, time_ms=float(time_ms))


def eligible(z, region, p=None, failing=False):
    q = region[z['head_beat']] & (z['divisor'] == 3)
    if failing:
        q &= p[z['head_beat']] < .5
    return np.flatnonzero(q)


def simulate(z, p, region, c, *, simulator, position, run_length, confirming,
             identity, null=None, training=False):
    rng = np.random.default_rng(stable_seed((identity, simulator, position, run_length, confirming, null)))
    candidates = eligible(z, region, p, failing=training)
    surprise = []
    if null != 'confirming' and len(candidates):
        if null == 'self':
            candidates = candidates[p[z['head_beat'][candidates]] >= .5]
            if not len(candidates):
                return None
        if training and null is None:
            pick = int(rng.integers(len(candidates)))
        elif position == 3:
            mean = norm(c[region].mean(axis=0, keepdims=True))[0]
            pick = int(np.argmin(c[z['head_beat'][candidates]] @ mean))
        else:
            third = np.array_split(np.arange(len(candidates)), 3)[position]
            if not len(third):
                return None
            pick = int(rng.choice(third))
        ix = candidates[pick:min(pick + run_length, len(candidates))]
        if len(ix) < run_length:
            ix = candidates[max(0, pick - run_length + 1):pick + 1]
        if len(ix) < run_length:
            return None
        surprise = z['heads'][ix].tolist()
        if null == 'uninformative':
            at = np.flatnonzero(region)
            surprise = rng.uniform(z['beats'][at[0]], z['beats'][at[-1]] + z['length'][at[-1]], run_length).tolist()
    confirm = np.flatnonzero(np.isin(z['divisor'], (1, 2, 4, 8, 16)))
    ct = z['heads'][rng.choice(confirm, confirming, replace=len(confirm) < confirming)].tolist() if confirming and len(confirm) else []
    times = surprise + ct
    cfg = calibration()
    offset = rng.normal(cfg['offset_mean_ms'], cfg['offset_sd_ms'])
    noise = rng.standard_t(3, len(times)) * noise_scale()
    marks = [read_mark(t + offset + e, z) for t, e in zip(times, noise)]
    return marks


def contiguous(s, i, theta):
    q = s >= theta
    q[i] = True
    lo = hi = i
    while lo > 0 and q[lo - 1]:
        lo -= 1
    while hi + 1 < len(s) and q[hi + 1]:
        hi += 1
    mask = np.zeros(len(s), bool); mask[lo:hi + 1] = True
    return mask


def kernel(s, theta):
    return np.clip((s - theta) / max(1 - theta, 1e-6), 0, 1)


def sparsemax(x):
    ordered, _ = torch.sort(x, dim=-1, descending=True)
    ranks = torch.arange(1, x.shape[-1] + 1, device=x.device, dtype=x.dtype)
    total = ordered.cumsum(-1)
    support = 1 + ranks * ordered > total
    size = support.sum(-1, keepdim=True).clamp_min(1)
    threshold = (total.gather(-1, size - 1) - 1) / size
    return (x - threshold).clamp_min(0)


class Reach(nn.Module):
    def __init__(self, width=192):
        super().__init__()
        self.q = nn.Sequential(nn.Linear(width + 1, 12), nn.GELU(), nn.Linear(12, 32))
        self.k = nn.Sequential(nn.Linear(width + 2, 12), nn.GELU(), nn.Linear(12, 32))
        self.lag = nn.Embedding(16, 1)
        self.weight = nn.Parameter(torch.ones(2))
        self.null = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.lag.weight)

    def forward(self, c, m0, marks):
        if not marks:
            return torch.zeros_like(m0)
        ix = torch.tensor([m['beat'] for m in marks], dtype=torch.long)
        kind = torch.tensor([m['kind'] for m in marks], dtype=torch.long)
        value = torch.tensor([m['L'] for m in marks], dtype=torch.float32)
        q = self.q(torch.cat([c, m0[:, None]], dim=1))
        k = self.k(torch.cat([c[ix], nn.functional.one_hot(kind, 2).float()], dim=1))
        distance = abs(torch.arange(len(c))[:, None] - ix[None, :])
        bucket = torch.floor(torch.log1p(distance.float()) / math.log(2) * 1.5).long().clamp(max=15)
        score = q @ k.T / math.sqrt(32) + self.lag(bucket).squeeze(-1)
        alpha = sparsemax(torch.cat([score, self.null.expand(len(c), 1)], dim=1))[:, :-1]
        return (alpha * (self.weight[kind] * value)[None, :]).sum(dim=1)


@dataclass
class Song:
    unit: dict
    z: dict
    p: np.ndarray
    c: np.ndarray
    cm: np.ndarray
    similarity: np.ndarray
    mel_similarity: np.ndarray
    bar_similarity: np.ndarray
    exact: np.ndarray
    regions: list
    hash: str


def song(seed, u):
    z = labels(u)
    p = np.load(OUT / 'm0' / f's{seed}' / (u['id'] + '.npy'))
    t = token_load(seed, u)
    c, cm = norm(context(t['Fz'])), norm(context(t['Mz']))
    b = norm(bar_tokens(t['Fz'], z))
    exact, _ = pattern_matrix(z)
    return Song(u, z, p, c, cm, c @ c.T, cm @ cm.T, b @ b.T, exact, regions(z, p), pulse_hash(z))


def update(s, marks, arm, config, learned=None, rng_seed=0):
    before = pulse_hash(s.z)
    assert before == s.hash, 'Pulse was mutated before propagation'
    n = len(s.p)
    delta, local = np.zeros(n, np.float32), np.zeros(n, bool)
    g = config.get('gain', 1.)
    if marks and g:
        if arm.startswith('learned'):
            assert learned is not None
            learned.eval()
            with torch.no_grad():
                delta = learned(torch.from_numpy(s.c), torch.from_numpy(logit(np.clip(s.p, 1e-6, 1 - 1e-6))), marks).numpy() * g
            for m in marks:
                local |= contiguous(s.similarity[m['beat']], m['beat'], config['local'])
        else:
            rng = np.random.default_rng(rng_seed)
            for mark in marks:
                i, llr = mark['beat'], mark['L']
                sim = s.mel_similarity[i] if arm == 'mel' else s.similarity[i]
                loc = contiguous(sim, i, config.get('local', .4))
                local |= loc
                w = kernel(sim, config.get('local', .4)) * loc
                if arm == 'mark':
                    w[:] = 0; w[i] = 1
                    local[:] = False; local[[m['beat'] for m in marks]] = True
                elif arm in ('graph', 'random', 'mel'):
                    nonlocal_w = kernel(sim, config.get('theta', .6)) * ~loc
                    rr = runs(nonlocal_w > 0)
                    rr.sort(key=lambda q: -float(nonlocal_w[q].sum()))
                    kept = np.zeros(n, np.float32)
                    for q in rr[:config.get('K', 3)]:
                        kept[q] = nonlocal_w[q]
                    if arm == 'random':
                        vals = kept[kept > 0]
                        target = rng.choice(np.flatnonzero(~loc), len(vals), replace=False)
                        kept[:] = 0; kept[target] = rng.permutation(vals)
                    w = np.maximum(w, kept)
                elif arm == 'repeat':
                    bar = int(np.searchsorted(s.z['bar_ids'], s.z['bars'][i]))
                    targets = s.z['bar_ids'][s.bar_similarity[bar] >= config.get('repeat', 1.01)]
                    w[np.isin(s.z['bars'], targets) & (s.z['bib'] == s.z['bib'][i])] = 1
                elif arm == 'periodic':
                    q = (abs(s.z['bars'] - s.z['bars'][i]) % 8 == 0) & (s.z['bib'] == s.z['bib'][i])
                    w[q] = 1
                elif arm in ('bias', 'everywhere'):
                    w[:] = 1
                delta += w.astype('float32') * llr * (1 if arm == 'everywhere' else g)
    new = s.p.copy()
    changed_value = delta != 0
    new[changed_value] = expit(logit(np.clip(s.p[changed_value], 1e-6, 1 - 1e-6)) + delta[changed_value])
    declared = np.flatnonzero((new != s.p) | ((new > .5) != (s.p > .5)))
    assert before == pulse_hash(s.z), 'Pulse changed under cell conditioning'
    assert np.array_equal(declared, np.flatnonzero(new != s.p)), 'Declared changes disagree with posterior'
    if not marks:
        assert new.tobytes() == s.p.tobytes(), 'No-mark update changed M0'
    return new, local, declared


def measures(s, region, marks, p, local):
    z, base = s.z, s.p > .5
    pred = p > .5
    changed = pred != base
    at = np.array([m['beat'] for m in marks], int)
    excluded = np.zeros(len(p), bool)
    for i in at:
        excluded[max(0, i - 1):i + 2] = True
    failing = (z['y'] == 1) & ~base & ~excluded
    numerator = lambda q: float(pred[q].mean()) if q.any() else None
    other = np.zeros(len(p), bool)
    for r in s.regions:
        if not np.any(r & region):
            other |= r
    source = np.isin(z['bar_ids'], z['bars'][region])
    lag = abs(z['bar_ids'][:, None] - z['bar_ids'][None, :])
    target = np.any(s.exact & source[:, None] & ~source[None, :] & (lag >= 8), axis=0)
    tier2 = np.isin(z['bars'], z['bar_ids'][target])
    outside = ~region
    ch = changed & outside
    labelled = z['y'] >= 0
    toward = ch & labelled & (pred == z['y'])
    away = ch & labelled & (pred != z['y'])
    duration = np.minimum(z['length'], np.r_[np.diff(z['beats']), z['length'][-1]])
    result = dict(M1e=numerator(failing & region), M1e_n=int(sum(failing & region)),
                  R_binary_flipped=float(np.mean(pred[region & (z['y'] == 0) & ~base])) if np.any(region & (z['y'] == 0) & ~base) else None,
                  M2=float(sum(local & region) / max(sum(local | region), 1)),
                  M3i=numerator(failing & other), M3i_n=int(sum(failing & other)),
                  M3ii=numerator(failing & tier2), M3ii_n=int(sum(failing & tier2)),
                  M4_toward=float(sum(toward) / sum(ch)) if ch.any() else 0.,
                  M4_away=float(sum(away) / sum(ch)) if ch.any() else 0.,
                  M4_unlabelled=float(sum(ch & ~labelled) / sum(ch)) if ch.any() else 0.,
                  M4_n=int(sum(ch)),
                  M7=float(duration[outside & (changed | (abs(p - s.p) > .1))].sum() / 1000),
                  M8=float(changed.mean()), changed=int(changed.sum()),
                  tier2_precision=float(np.mean(pred[tier2 & changed & labelled] == z['y'][tier2 & changed & labelled])) if np.any(tier2 & changed & labelled) else None)
    if region.any():
        bars = z['bars'][region]
        dist = np.min(abs(z['bars'][:, None] - bars[None, :]), axis=1)
        for name, q in [('0-7', dist < 8), ('8-15', (dist >= 8) & (dist < 16)),
                        ('16-31', (dist >= 16) & (dist < 32)), ('32+', dist >= 32)]:
            result['M3i_lag' + name] = numerator(failing & other & q)
            result['M3ii_lag' + name] = numerator(failing & tier2 & q)
    return result


def learned_path(seed, train_seed):
    return OUT / 'models' / f'P7-s{seed}-learned{train_seed}.pt'


def load_learned(seed, train_seed):
    m = Reach()
    m.load_state_dict(torch.load(learned_path(seed, train_seed), map_location='cpu', weights_only=False)['state'])
    return m


def train_learned(seed, train_seed):
    prereg_check()
    path = learned_path(seed, train_seed)
    if path.exists():
        return
    torch.manual_seed(7100 + train_seed)
    model = Reach()
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.001)
    fit_units = [u for u in units() if u['cohort'] == 'probe-fit']
    dev_units = [u for u in units() if u['cohort'] == 'probe-dev']
    rng = np.random.default_rng(7100 + train_seed)
    best, stale, history = float('inf'), 0, []
    for epoch in range(20):
        model.train()
        used = 0
        for ix in rng.permutation(len(fit_units)):
            s = song(seed, fit_units[ix])
            valid = [r for r in s.regions if len(eligible(s.z, r, s.p, True))]
            if not valid:
                continue
            region = valid[rng.integers(len(valid))]
            confirming = rng.random() < .2
            marks = simulate(s.z, s.p, region, s.c, simulator=epoch, position=int(rng.integers(3)),
                             run_length=1, confirming=8 if confirming else 0,
                             identity=s.unit['id'], null='confirming' if confirming else None, training=True)
            if marks is None:
                continue
            c = torch.from_numpy(s.c)
            base = torch.from_numpy(logit(np.clip(s.p, 1e-6, 1 - 1e-6)))
            delta = model(c, base, marks)
            if not confirming and rng.random() < .5:
                after = expit((base + delta).detach().numpy())
                remaining = [r for r in regions(s.z, after) if not np.any(r & region)
                             and len(eligible(s.z, r, after, True))]
                if remaining:
                    second = remaining[rng.integers(len(remaining))]
                    more = simulate(s.z, after, second, s.c, simulator=epoch + 100,
                        position=int(rng.integers(3)), run_length=1, confirming=0,
                        identity=s.unit['id'], training=True)
                    if more:
                        marks += more
                        delta = model(c, base, marks)
            q = s.z['y'] >= 0
            loss = nn.functional.binary_cross_entropy_with_logits((base + delta)[q],
                        torch.as_tensor(s.z['y'][q], dtype=torch.float32))
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.)
            opt.step(); used += 1
            assert pulse_hash(s.z) == s.hash, 'Training episode changed the pulse'
        model.eval()
        losses, no_marks, shuffled = [], [], []
        with torch.no_grad():
            for u in dev_units:
                s = song(seed, u)
                valid = [r for r in s.regions if len(eligible(s.z, r))]
                if not valid:
                    continue
                r = valid[stable_seed(u['id']) % len(valid)]
                marks = simulate(s.z, s.p, r, s.c, simulator=0, position=1,
                                 run_length=1, confirming=0, identity=u['id'])
                if not marks:
                    continue
                c, base = torch.from_numpy(s.c), torch.from_numpy(logit(np.clip(s.p, 1e-6, 1 - 1e-6)))
                q = s.z['y'] >= 0
                target = torch.as_tensor(s.z['y'][q], dtype=torch.float32)
                losses.append(float(nn.functional.binary_cross_entropy_with_logits((base + model(c, base, marks))[q], target)))
                no_marks.append(float(nn.functional.binary_cross_entropy_with_logits(base[q], target)))
                shuffled_marks = [dict(m, beat=(m['beat'] + len(base) // 2) % len(base)) for m in marks]
                shuffled.append(float(nn.functional.binary_cross_entropy_with_logits((base + model(c, base, shuffled_marks))[q], target)))
                assert pulse_hash(s.z) == s.hash, 'Development episode changed the pulse'
        val = float(np.mean(losses)) if losses else float('inf')
        history.append(dict(epoch=epoch, dev_bce=val, no_mark_bce=float(np.mean(no_marks)),
                            shuffled_bce=float(np.mean(shuffled)), train_episodes=used,
                            dev_recordings=len(losses),
                            mark_delta_nll=float(np.mean(no_marks)) - val,
                            shuffled_delta_nll=float(np.mean(shuffled)) - val))
        log('P7_learned_epoch', seed=seed, train_seed=train_seed, **history[-1])
        if val < best - 1e-5:
            best, stale = val, 0
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(state=model.state_dict(), parameters=sum(p.numel() for p in model.parameters()),
                            history=history, train_seed=train_seed), path)
        else:
            stale += 1
        if stale >= 3:
            break
    dump(OUT / f'P7-learned-s{seed}-{train_seed}-history.json', history)


def tuning_units(seed):
    fail, nofail = [], []
    for u in units():
        if u['cohort'] != 'probe-dev':
            continue
        z = labels(u)
        p = np.load(OUT / 'm0' / f's{seed}' / (u['id'] + '.npy'))
        regs = regions(z, p)
        if any(len(eligible(z, r)) for r in regs):
            fail.append(u)
        elif not regs:
            nofail.append(u)
    nofail.sort(key=lambda u: stable_seed('nofail:' + u['id']))
    return fail + nofail[:len(fail)]


def repeat_threshold(seed, dev):
    values = defaultdict(lambda: defaultdict(list))
    thresholds = (.5, .6, .7, .8, .9, .95, .99, 1.01)
    for u in dev:
        s = song(seed, u)
        z = s.z
        lag = abs(z['bar_ids'][:, None] - z['bar_ids'][None, :])
        eligible_pair = np.triu((lag >= 8) & z['nontrivial'][:, None] & z['nontrivial'][None, :], 1)
        for th in thresholds:
            q = eligible_pair & (s.bar_similarity >= th)
            if q.any():
                values[th][u['rec']].append(float(s.exact[q].mean()))
    stats = {str(th): dict(n=len(g), precision=float(np.mean([np.mean(v) for v in g.values()])))
             for th, g in values.items() if g}
    eligible_th = [float(th) for th, v in stats.items() if v['precision'] >= .8 and v['n'] >= 40]
    return min(eligible_th) if eligible_th else 1.01, stats


def tune(seed, models):
    path = OUT / f'P7-config-s{seed}.json'
    if path.exists():
        return json.loads(path.read_text())
    dev = tuning_units(seed)
    chosen, diagnostics = {}, {}
    rep_th, rep_stats = repeat_threshold(seed, dev)
    for arm in ARMS:
        base = dict(local=chosen.get('local', {}).get('local', .4), theta=.6, K=3, repeat=rep_th)
        if arm == 'local':
            configs = [dict(base, local=th, gain=g) for th in THRESHOLDS for g in GAINS]
        elif arm in ('graph', 'mel'):
            configs = [dict(base, theta=th, K=k, gain=g) for th in (.4, .6, .8) for k in (1, 3, 8) for g in GAINS]
        elif arm == 'random':
            configs = [dict(chosen['graph'], gain=g) for g in GAINS]
        elif arm == 'everywhere':
            configs = [dict(base, gain=1.)]
        else:
            configs = [dict(base, gain=g) for g in GAINS]
        observations = [defaultdict(lambda: defaultdict(list)) for _ in configs]
        for u in dev:
            s = song(seed, u)
            valid = [r for r in s.regions if len(eligible(s.z, r))]
            r = valid[stable_seed(u['id']) % len(valid)] if valid else np.zeros(len(s.p), bool)
            for kind, count in [('real', 0), ('real', 8), ('confirming', 8)]:
                if kind == 'real' and not valid:
                    continue
                marks = simulate(s.z, s.p, r, s.c, simulator=0, position=1,
                                 run_length=1, confirming=count, identity=u['id'],
                                 null='confirming' if kind == 'confirming' else None)
                if marks is None:
                    continue
                for i, cfg in enumerate(configs):
                    p, loc, _ = update(s, marks, arm, cfg, models.get(arm), stable_seed(u['id']))
                    m = measures(s, r, marks, p, loc)
                    for key in (['M8'] if kind == 'confirming' else ['M1e', 'M3i', 'M4_away']):
                        if m[key] is not None:
                            observations[i][key][u['rec']].append(m[key])
        stats = []
        for cfg, ob in zip(configs, observations):
            means = {k: float(np.mean([np.mean(v) for v in g.values()])) for k, g in ob.items() if g}
            feasible = means.get('M8', 1) <= .01 and means.get('M4_away', 1) <= .20
            stats.append(dict(config=cfg, means=means, n={k: len(g) for k, g in ob.items()},
                              feasible=feasible, objective=means.get('M1e', 0) + means.get('M3i', 0)))
        allowed = [s for s in stats if s['feasible']]
        if allowed:
            selected = sorted(allowed, key=lambda a: (-a['objective'], a['config']['gain'],
                              -a['config']['theta'], a['config']['K'], -a['config']['local']))[0]
        else:
            selected = dict(config=dict(base, gain=0.), means={}, n={}, feasible=False, objective=0.)
            if arm == 'everywhere':
                selected = stats[0]
        chosen[arm] = selected['config']
        diagnostics[arm] = dict(selected=selected, grid=stats)
        log('P7_tuned', seed=seed, arm=arm, selected=selected)
    result = dict(configs=chosen, diagnostics=diagnostics, repeat_precision=rep_stats)
    dump(path, result)
    return result


def fixtures():
    z = dict(beats=np.array([0., 500., 1000., 1125.5, 1525.5, 1925.5, 2325.5]),
             length=np.array([500., 500., 500., 400., 400., 400., 400.]),
             red=np.array([[0., 500., 4], [1125.5, 400., 3], [1925.5, 400., 4]]),
             bars=np.array([0, 0, 0, 1, 1, 2, 2]), bib=np.array([0, 1, 2, 0, 1, 0, 1]),
             bar_ids=np.arange(3), y=np.array([0, 1, -1, 1, 0, 1, 0]))
    rng = np.random.default_rng(19)
    c = norm(rng.normal(size=(7, 192)).astype('float32'))
    p = np.array([.1, .2, .3, .4, .5, .6, .7], np.float32)
    s = Song({}, z, p, c, c.copy(), c @ c.T, c @ c.T, np.eye(3), np.eye(3, dtype=bool), [], pulse_hash(z))
    checked = 0
    for arm in ARMS:
        m = Reach() if arm.startswith('learned') else None
        cfg = dict(gain=.5, local=.4, theta=.6, K=3, repeat=.8)
        for marks in ([], [dict(beat=6, L=2., kind=0)], [dict(beat=3, L=-1., kind=0)]):
            after, loc, declared = update(s, marks, arm, cfg, m)
            assert pulse_hash(z) == s.hash
            assert np.array_equal(declared, np.flatnonzero(after != p))
            if not marks:
                assert after.tobytes() == p.tobytes()
            checked += 1
    return dict(passed=True, updates=checked, pulse_sha256=s.hash,
                cases=['multiple red segments', 'fractional boundary', 'meter reset', 'last beat', 'no marks'])


def evaluate(seed, models, configs):
    import gzip
    fixture_result = fixtures()
    dump(OUT / f'P7-invariants-s{seed}.json', dict(fixtures=fixture_result, complete=False))
    panel = [u for u in units() if u['cohort'] in ('panel', 'val_hard')]
    summary = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0., 0])))
    checked, skipped, region_count, nofail_count, fail_count = 0, 0, 0, 0, 0
    curves = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0., 0])))
    failing_ids, nofail_ids = [], []
    for u in panel:
        z = labels(u)
        p0 = np.load(OUT / 'm0' / f's{seed}' / (u['id'] + '.npy'))
        regs = regions(z, p0)
        if any(len(eligible(z, r)) for r in regs):
            failing_ids.append(u)
        elif not regs:
            nofail_ids.append(u)
    nofail_ids.sort(key=lambda u: stable_seed('panel-nofail:' + u['id']))
    chosen_nofail = {u['id'] for u in nofail_ids[:len(failing_ids)]}

    def record(s, arm, kind, position, sim, run_length, k, values):
        band = 'ge160' if np.median(60000 / s.z['length']) >= 160 else 'lt160'
        groups = [f'{arm}:{kind}:all', f'{arm}:{kind}:run{run_length}',
                  f'{arm}:{kind}:k{k}', f'{arm}:{kind}:pos{position}',
                  f'{arm}:{kind}:sim{sim}', f'{arm}:{kind}:{band}']
        for group in groups:
            for key, v in values.items():
                if v is not None:
                    acc = summary[group][key][s.unit['rec']]
                    acc[0] += v; acc[1] += 1

    with gzip.open(OUT / f'P7-episodes-s{seed}.jsonl.gz', 'wt') as raw:
        for num, u in enumerate(panel):
            s = song(seed, u)
            for arm in ARMS:
                update(s, [], arm, configs[arm], models.get(arm)); checked += 1
            regs = [r for r in s.regions if len(eligible(s.z, r))]
            if not regs:
                if u['id'] not in chosen_nofail:
                    continue
                nofail_count += 1
                r = np.zeros(len(s.p), bool)
                for sim in (0, 1):
                    marks = simulate(s.z, s.p, r, s.c, simulator=sim, position=1,
                                     run_length=1, confirming=8, identity=u['id'], null='confirming')
                    for arm in ARMS:
                        p, loc, _ = update(s, marks, arm, configs[arm], models.get(arm), num)
                        checked += 1
                        values = measures(s, r, marks, p, loc)
                        record(s, arm, 'confirming-nofail', 1, sim, 1, 8, values)
                continue
            fail_count += 1
            donors = [v for v in failing_ids if v['rec'] != u['rec']]
            donor = song(seed, donors[stable_seed(u['id']) % len(donors)]) if donors else s
            donor_regs = [r for r in donor.regions if len(eligible(donor.z, r))]
            for ri, r in enumerate(regs):
                region_count += 1
                for sim in (0, 1):
                    for position in range(4):
                        for run_length in (1, 3):
                            for k in (0, 8):
                                for kind in ('real', 'self', 'uninformative'):
                                    marks = simulate(s.z, s.p, r, s.c, simulator=sim, position=position,
                                            run_length=run_length, confirming=k, identity=(u['id'], ri),
                                            null=None if kind == 'real' else kind)
                                    if marks is None:
                                        skipped += 1; continue
                                    arms = list(ARMS)
                                    for arm in arms:
                                        p, loc, declared = update(s, marks, arm, configs[arm], models.get(arm),
                                                                 stable_seed((u['id'], ri, sim, position, k)))
                                        checked += 1
                                        values = measures(s, r, marks, p, loc)
                                        record(s, arm, kind, position, sim, run_length, k, values)
                                        raw.write(json.dumps(dict(unit=u['id'], rec=u['rec'], region=ri,
                                            arm=arm, null=kind, position=position, simulator=sim,
                                            run=run_length, confirming=k, declared=len(declared), **values)) + '\n')
                                    if kind == 'real' and run_length == 1 and k == 0 and position < 3:
                                        for arm in ('graph', 'learned0', 'learned1'):
                                            for gain in (0., .0625, .125, .25, .5, 1., 2., 4.):
                                                cfg = dict(configs[arm], gain=gain)
                                                p, loc, _ = update(s, marks, arm, cfg, models.get(arm))
                                                checked += 1
                                                values = measures(s, r, marks, p, loc)
                                                for key in ('M3i', 'M4_away'):
                                                    if values[key] is not None:
                                                        acc = curves[f'{arm}:{gain}'][key][u['rec']]
                                                        acc[0] += values[key]; acc[1] += 1
                                    if kind == 'real':
                                        # Transfer a donor mark's whole-song relative time.
                                        dr = donor_regs[ri % len(donor_regs)] if donor_regs else np.ones(len(donor.p), bool)
                                        dm = simulate(donor.z, donor.p, dr, donor.c, simulator=sim,
                                            position=position, run_length=run_length, confirming=k,
                                            identity=(donor.unit['id'], ri))
                                        if dm:
                                            ds = donor.z['beats'][0]
                                            dd = donor.z['beats'][-1] + donor.z['length'][-1] - ds
                                            ts = s.z['beats'][0]
                                            td = s.z['beats'][-1] + s.z['length'][-1] - ts
                                            shuffled = [read_mark(ts + (m['time_ms'] - ds) / dd * td, s.z) for m in dm]
                                            for arm in ('learned0', 'learned1'):
                                                p, loc, _ = update(s, shuffled, arm, configs[arm], models[arm])
                                                checked += 1
                                                values = measures(s, r, shuffled, p, loc)
                                                record(s, arm, 'shuffle', position, sim, run_length, k, values)
                for sim in (0, 1):
                    marks = simulate(s.z, s.p, r, s.c, simulator=sim, position=1,
                                     run_length=1, confirming=8, identity=u['id'], null='confirming')
                    for arm in ARMS:
                        p, loc, _ = update(s, marks, arm, configs[arm], models.get(arm), num)
                        checked += 1
                        record(s, arm, 'confirming-fail', 1, sim, 1, 8, measures(s, r, marks, p, loc))
            if num % 20 == 0:
                log('P7_read', seed=seed, units=num, regions=region_count, checked=checked)
    def finish(groups):
        from .common import interval
        return {group: {metric: dict(**interval([v[0] / v[1] for v in recs.values()]),
                       recordings={r: float(v[0] / v[1]) for r, v in recs.items()})
                       for metric, recs in metrics.items()} for group, metrics in groups.items()}
    result = dict(summary=finish(summary), curves=finish(curves),
                  regions=region_count, failing_units=fail_count, nonfailing_units=nofail_count,
                  skipped_mark_draws=skipped, invariant_checks=checked)
    dump(OUT / f'P7-s{seed}.json', result)
    dump(OUT / f'P7-invariants-s{seed}.json', dict(fixtures=fixture_result, complete=True,
                                                episode_updates_checked=checked, breaches=0))
    return result


def run(seed):
    prereg_check()
    dump(OUT / f'P7-fixtures-s{seed}.json', fixtures())
    for train_seed in (0, 1):
        train_learned(seed, train_seed)
    models = {f'learned{k}': load_learned(seed, k) for k in (0, 1)}
    configs = tune(seed, models)['configs']
    evaluate(seed, models, configs)
