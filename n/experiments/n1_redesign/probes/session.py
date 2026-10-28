"""Private-session replay and simulator characterization; all outputs are artifacts."""
from collections import Counter, defaultdict
import json

import numpy as np
from scipy.special import expit

from .common import ROOT, OUT, digest, dump, features, interp, labels, log, manifests, nearest, norm, prereg_check, units
from .models import Predictor, load
from .attacks import model_path as attack_path
from .cells import inputs, model_path as cell_path
from .episodes import (Song, calibration, load_learned, noise_scale, pulse_hash,
                       read_mark, update)
from .tokens import context, standardize_song
from ._reader import read_marks, pulse_labels


def load_session():
    cfg = json.loads((OUT / 'runtime.json').read_text())['session']
    p = ROOT / cfg['marks']
    assert digest(p) == cfg['marks_sha256']
    latest = {}
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if r['type'] == 'save':
            pid = r['passage_id']
            if pid not in latest or r['time'] >= latest[pid]['time']:
                latest[pid] = r
    passages = json.loads((ROOT / cfg['session']).read_text())['passages']
    identities = json.loads((ROOT / cfg['identities']).read_text())['passages']
    counts = Counter(m['kind'] for r in latest.values() for m in r['marks'])
    counts['passages'] = len(passages)
    counts['recordings'] = len({p['audio']['sha256'] for p in passages})
    assert dict(counts) == cfg['expected_counts']
    ip = OUT / 'inputs.json'
    provenance = json.loads(ip.read_text())
    provenance['session'] = {key: dict(path=str(ROOT / cfg[key]), sha256=digest(ROOT / cfg[key]))
                             for key in ('marks', 'session', 'identities')}
    dump(ip, provenance)
    return cfg, passages, latest, identities


def decoded_song(seed, u, cache):
    d, _, g = inputs(seed, u, cache, 'decoded', full_song=True)
    p = expit(Predictor(load(cell_path(seed, 'mlp', 'T24')))(d['T24'])).astype('float32')
    bars = np.cumsum(g['down']).astype(int)
    ub = np.unique(bars)
    bib = np.zeros(len(p), int)
    for b in ub:
        q = np.flatnonzero(bars == b); bib[q] = np.arange(len(q))
    z = dict(g, bars=bars, bar_ids=ub, bib=bib, y=np.full(len(p), -1))
    c = norm(context(standardize_song(d['T24'])))
    cm = norm(context(standardize_song(d['Mel24'])))
    return Song(u, z, p, c, cm, c @ c.T, cm @ cm.T, np.eye(len(ub)),
                np.eye(len(ub), dtype=bool), [], pulse_hash(z))


def replay(seed):
    prereg_check()
    cfg, passages, latest, identities = load_session()
    cache = manifests(seed)
    by_sha = {u['sha']: u for u in units()}
    configs = json.loads((OUT / f'P7-config-s{seed}.json').read_text())['configs']
    models = {f'learned{k}': load_learned(seed, k) for k in (0, 1)}
    results, missed, attack_rows = {}, [], []
    for passage in passages:
        pid, sha = passage['id'], passage['audio']['sha256']
        if sha not in by_sha or sha not in cache:
            missed.append(dict(passage=pid, reason='No allowed feature/reference membership'))
            continue
        u = by_sha[sha]
        try:
            s = decoded_song(seed, u, cache)
        except FileNotFoundError as exc:
            missed.append(dict(passage=pid, reason=str(exc))); continue
        marks = latest[pid]['marks']
        reading, pairs, runs = read_marks(s.z['beats'], marks, offset=cfg['reader_offset_ms'])
        # The stage-0 reader routes accents to a separate layer and between-cell
        # runs to a span operation. Neither can become a ternary cell update.
        accepted = [r for r in reading if r['class'] in ('line', 'cell')]
        applied = [read_mark(r['time_ms'], s.z, offset=cfg['reader_offset_ms']) for r in accepted]
        entry = dict(counts=dict(Counter(r['class'] for r in reading)), pairs=pairs, runs=runs,
                     pulse_before=s.hash, arms={})
        for arm in ('local', 'graph', 'learned0', 'learned1'):
            p, loc, declared = update(s, applied, arm, configs[arm], models.get(arm))
            outside = declared[(s.z['beats'][declared] < passage['start_ms']) |
                               (s.z['beats'][declared] >= passage['end_ms'])]
            source = [read_mark(r['time_ms'], s.z, offset=cfg['reader_offset_ms'])['beat']
                      for r in reading if r['class'] == 'cell' and r['subdivision'] == 3]
            regions_outside = []
            from .repeats import runs as index_runs
            mask = np.zeros(len(p), bool); mask[outside] = True
            for run in index_runs(mask):
                regions_outside.append([float(s.z['beats'][run[0]]),
                    float(s.z['beats'][run[-1]] + s.z['length'][run[-1]])])
            between = [r for r in reading if r['class'] == 'between']
            # Test the between-only intervention separately from other marks in
            # the same passage; otherwise their legitimate updates confound it.
            bp, _, bd = update(s, [], arm, configs[arm], models.get(arm))
            source = sorted(set(source))
            before_ternary = s.p[source] > .5
            after_ternary = p[source] > .5
            entry['arms'][arm] = dict(pulse_unchanged=s.hash == pulse_hash(s.z),
                ternary_source_beats=len(source),
                ternary_before=float(before_ternary.mean()) if source else None,
                ternary_after=float(after_ternary.mean()) if source else None,
                newly_ternary_source_beats=int(np.sum(after_ternary & ~before_ternary)),
                triplets_pass=bool(source and np.any(~before_ternary) and np.all(after_ternary)),
                between_marks=len(between), between_only_unchanged=not len(bd),
                U_expectation='unassessed: U probe not fitted', outside_regions=regions_outside,
                declared_beats=len(declared), map_changes=int(np.sum((p > .5) != (s.p > .5))))
        results[pid] = entry
        times = np.array(sorted(m['time_ms'] for m in marks if m['kind'] == 'beat'))
        off = ~pulse_labels(passage['proposals'], times)
        f, _, _ = features(cache[sha])
        attack = Predictor(load(attack_path(seed, 'mlp', 'F')))
        corrected = times[off] - calibration()['offset_mean_ms']
        ix = np.clip(np.searchsorted(s.z['beats'], corrected, 'right') - 1, 0, len(s.p) - 1)
        shifted = corrected + s.z['length'][ix] / 32
        a, b = expit(attack(interp(f, corrected))), expit(attack(interp(f, shifted)))
        attack_rows.append(dict(passage=pid, rec=u['rec'], marks=len(a),
                                mean_score=float(a.mean()) if len(a) else None,
                                mean_shifted=float(b.mean()) if len(b) else None,
                                mean_difference=float((a - b).mean()) if len(a) else None))
    expectations = {}
    for arm in ('local', 'graph', 'learned0', 'learned1'):
        def result(pid, test):
            if pid not in results:
                return 'unassessed: missing input'
            return 'pass' if test(results[pid]['arms'][arm]) else 'fail'
        def triplets_result(pid):
            if pid in results and results[pid]['arms'][arm]['ternary_before'] == 1.:
                return 'unassessed: source cells were already ternary'
            return result(pid, lambda r: r['triplets_pass'])
        expectations[arm] = dict(
            triplets={pid: triplets_result(pid) for pid in cfg['expectations']['triplets']},
            layers={pid: result(pid, lambda r: r['pulse_unchanged']) for pid in cfg['expectations']['layers']},
            between={pid: result(pid, lambda r: r['between_marks'] > 0 and r['between_only_unchanged'])
                     for pid in cfg['expectations']['between']},
            unbeat_pair_U='unassessed: separate U probe was not fitted',
            pulse='pass' if len(results) == len(passages) and all(
                r['arms'][arm]['pulse_unchanged'] for r in results.values()) else 'unassessed: missing passages')
    dump(OUT / f'P8-s{seed}.json', dict(passages=results, missing=missed, expectations=cfg['expectations'],
                                    expectation_results=expectations, attack_marks=attack_rows, design_only=True))
    log('P8_done', seed=seed, passages=len(results), missing=len(missed))


def robust_sd(x):
    return float(1.4826 * np.median(abs(x - np.median(x)))) if len(x) else None


def phases(times, z):
    if not len(times):
        return None
    i = np.clip(np.searchsorted(z['beats'], times, 'right') - 1, 0, len(z['beats']) - 1)
    phase = (times - z['beats'][i]) / z['length'][i]
    cls = np.full(len(times), 3)
    for k, cells in reversed(list(enumerate(([.5], [1 / 3, 2 / 3], [.25, .75])))):
        distance = np.min(abs(phase[:, None] - cells), axis=1) * z['length'][i]
        cls[distance <= 20] = k
    return np.bincount(cls, minlength=4) / max(len(times), 1)


def simulator_bands():
    prereg_check()
    cfg, passages, latest, _ = load_session()
    bounds = json.loads((OUT / 'runtime.json').read_text())['simulator_bands']
    by_sha = {u['sha']: u for u in units()}
    rng = np.random.default_rng(20261028)
    records, owner_res, sim_res, owner_phase, sim_phase, cover = [], [], [], [], [], []
    for p in passages:
        if p['audio']['sha256'] not in by_sha:
            continue
        z = labels(by_sha[p['audio']['sha256']])
        times = np.array(sorted(m['time_ms'] for m in latest[p['id']]['marks'] if m['kind'] == 'beat'))
        lines = z['beats'][(z['beats'] >= p['start_ms']) & (z['beats'] < p['end_ms'])]
        heads = z['heads'][(z['heads'] >= p['start_ms']) & (z['heads'] < p['end_ms'])]
        offheads = heads[abs(nearest(heads, z['beats'])[0]) > 30]
        targets = []
        for pool, count in [(lines, len(times) // 2), (offheads, len(times) - len(times) // 2)]:
            chosen = set()
            if len(pool):
                for _ in range(max(100, count * 10)):
                    start = rng.integers(len(pool)); length = rng.integers(3, 9)
                    chosen.update(range(start, min(start + length, len(pool))))
                    if len(chosen) >= min(count, len(pool)):
                        break
                selected = np.array(sorted(chosen))[:count]
                targets.extend(pool[selected])
        targets = np.array(sorted(targets))
        errors = rng.standard_t(3, len(targets)) * noise_scale()
        simulated = targets + rng.normal(calibration()['offset_mean_ms'], calibration()['offset_sd_ms']) + errors
        corrected = times - calibration()['offset_mean_ms']
        res = nearest(corrected, np.unique(np.r_[lines, heads]))[0]
        res -= np.median(res)
        errors -= np.median(errors)
        owner_res.extend(res); sim_res.extend(errors)
        op, sp = phases(corrected, z), phases(simulated - calibration()['offset_mean_ms'], z)
        if op is not None and sp is not None:
            owner_phase.append(op); sim_phase.append(sp)
        owner_off = ~pulse_labels(p['proposals'], times)
        simulated_off = ~pulse_labels(p['proposals'], simulated)
        coverage = abs(nearest(corrected[owner_off], z['heads'])[0]) <= bounds['chart_coverage_ms']
        cover.extend(coverage)
        def run_lengths(t):
            if not len(t): return []
            t = np.sort(t)
            ix = np.clip(np.searchsorted(z['beats'], t, 'right') - 1, 0, len(z['beats']) - 1)
            cuts = np.flatnonzero(np.diff(t) > 2 * z['length'][ix[:-1]]) + 1
            return [len(v) for v in np.split(t, cuts)]
        records.append(dict(passage=p['id'], owner_n=len(times), simulator_n=len(simulated),
            owner_offpulse=float(owner_off.mean()) if len(owner_off) else None,
            simulator_offpulse=float(simulated_off.mean()) if len(simulated) else None,
            owner_runs=run_lengths(times), simulator_runs=run_lengths(simulated),
            owner_residual_robust_sd_ms=robust_sd(res), simulator_residual_robust_sd_ms=robust_sd(errors)))
    a, b = np.array(owner_res), np.array(sim_res)
    sd = robust_sd(b)
    tail = float(np.mean(abs(b) > bounds['tail_threshold_ms']))
    comparable = [r for r in records if r['simulator_offpulse'] is not None and r['owner_offpulse'] is not None]
    offdelta = float(np.mean([r['simulator_offpulse'] - r['owner_offpulse'] for r in comparable]))
    tv = float(np.sum(abs(np.mean(owner_phase, axis=0) - np.mean(sim_phase, axis=0))) / 2)
    bands = dict(robust_sd=bounds['robust_sd_ms'][0] <= sd <= bounds['robust_sd_ms'][1],
                 beyond15=bounds['tail_share'][0] <= tail <= bounds['tail_share'][1],
                 offpulse=abs(offdelta) <= bounds['offpulse_tolerance'], phase_tv=tv <= bounds['phase_tv_max'])
    dump(OUT / 'P6.json', dict(records=records, bands=bands, fixed_bands=bounds, simulator_optimistic=not all(bands.values()),
        owner_residual_robust_sd_ms=robust_sd(a), simulator_residual_robust_sd_ms=sd,
        simulator_beyond15_share=tail, offpulse_difference=offdelta, phase_tv=tv,
        comparison_passages=dict(offpulse=len(comparable), phase=len(owner_phase)),
        owner_chart_coverage=dict(n=len(cover), share=float(np.mean(cover)) if cover else None),
        deviation='Residuals are centered per passage; simulated residual uses intended source event, owner residual uses nearest chart line/head. Run lengths use uniform 3..8 target candidates.',
        design_only=True))
