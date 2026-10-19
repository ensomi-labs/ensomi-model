"""Rung 1's reading under the pre-set rule (design §5.1 R1; pre-registration in ``r1-prereg.md``).

Per panel and interim stratum: each arm's M1 (S2: M1-bar) as the mean over items of the two seeds'
final checkpoints, with a recording-cluster bootstrap; the controls (BeatThis final0/1/2 shifted by
their fixed constants, with +-5 ms; R2 as exported; chance). The decision reads the fit_dev panel, the
only one with all three BeatThis checkpoints:

- m_s = max(seed spread, BeatThis final0/1/2 spread, last-three-checkpoint spread), spreads as ranges;
- Delta_s = arm M1 - best BeatThis checkpoint's M1 (best chosen per stratum by its mean), with a paired
  recording-cluster bootstrap interval;
- O1: Delta_s > m_s with the interval excluding 0 on >= 2 of S2 (M1-bar), S3, S4, S5, S9 for either arm;
  O2: O1 and S0's Delta < -m_S0; O3: no rule stratum with Delta_s > m_s and val-n1 S0 M1 gained < 0.01
  over the last third of training; O4: as O3 but still rising by >= 0.01; otherwise ``unmatched``.
- A2 vs A1: A2 - A1 > m_s on >= 2 rule strata keeps F2; A1 >= A2 on all retires F2 as a family;
  otherwise both go to R2.

Strata with fewer than 20 recordings on the panel are reported and not counted.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import probe

RULE_STRATA = ('S2', 'S3', 'S4', 'S5', 'S9')
ALL_STRATA = ('S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S9')
MIN_RECORDINGS = 20
ARMS = ('A1', 'A2')
BT = ('bt-final0', 'bt-final1', 'bt-final2')


def _measure(stratum: str) -> str:
    return 'm1_bar' if stratum == 'S2' else 'm1'


def _rows_to_table(rows) -> dict:
    return {(r['panel'], r['item']): r for r in rows if not r.get('missing')}


def _items(table: dict, panel: str, stratum: str | None) -> list:
    return sorted(k for k, r in table.items() if k[0] == panel and (stratum is None or stratum in r['strata']))


def _vals(table, keys, measure) -> np.ndarray:
    return np.array([table[k][measure] if k in table else np.nan for k in keys], dtype=np.float64)


def _mean(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float('nan')


def _load(run: Path):
    arms = {}
    for arm in ARMS:
        for seed in (0, 1):
            files = sorted(run.glob(f'scores-{arm}-s{seed}-ckpt-*.json'))
            if files:
                arms[(arm, seed)] = [_rows_to_table(json.load(open(f))['rows']) for f in files]
    ctrl = json.load(open(run / 'scores-controls.json'))['rows'] if (run / 'scores-controls.json').exists() else {}
    return arms, {k: _rows_to_table(v) for k, v in ctrl.items()}


def _history_gain(run: Path, arm: str) -> float | None:
    gains = []
    for seed in (0, 1):
        p = run / f'{arm}-s{seed}' / 'status.json'
        if not p.exists():
            continue
        h = [x for x in json.load(open(p))['history'] if np.isfinite(x.get('val_s0_m1', np.nan))]
        if len(h) < 2:
            continue
        end = h[-1]
        target = 2.0 / 3.0 * end['elapsed_s']
        ref = min(h, key=lambda x: abs(x['elapsed_s'] - target))
        gains.append(end['val_s0_m1'] - ref['val_s0_m1'])
    return float(np.mean(gains)) if gains else None


def verdict(run: Path, r0: Path, panel: str = 'panel') -> dict:
    arms, ctrl = _load(run)
    tables, decision = {}, {}
    rec_of = {}
    for t in list(ctrl.values()) + [x for v in arms.values() for x in v]:
        for k, r in t.items():
            rec_of[k] = r['rec']
    for pnl in ('panel', 'val_hard', 'val_random', 'reset'):
        tab = {}
        if arms:
            base = {}
            for v in arms.values():
                base.update(v[-1])
        else:
            base = ctrl.get('bt-final0', {})
        for s in (None,) + ALL_STRATA:
            name = s or 'all'
            keys = _items(base, pnl, s) if base else []
            if not keys:
                continue
            recs = [rec_of[k] for k in keys]
            n_rec = len(set(recs))
            row = dict(n_items=len(keys), n_recordings=n_rec, counted=n_rec >= MIN_RECORDINGS)
            for meas in ('m1', 'm1_bar', 'm3'):
                cells = {}
                for arm in ARMS:
                    finals = [arms[(arm, sd)][-1] for sd in (0, 1) if (arm, sd) in arms]
                    if not finals:
                        continue
                    per_seed = [_vals(f, keys, meas) for f in finals]
                    avg = np.nanmean(np.vstack(per_seed), axis=0)
                    m, lo, hi = probe.bootstrap_mean(avg, recs)
                    seeds = [_mean(v) for v in per_seed]
                    ck_spread = []
                    for sd in (0, 1):
                        if (arm, sd) in arms:
                            ck_spread.append(float(np.ptp([_mean(_vals(c, keys, meas)) for c in arms[(arm, sd)][-3:]])))
                    cells[arm] = dict(mean=m, lo=lo, hi=hi, seeds=seeds,
                                      seed_spread=float(np.ptp(seeds)) if len(seeds) > 1 else None,
                                      ckpt_spread=max(ck_spread) if ck_spread else None)
                for c in list(BT) + [b + d for b in BT for d in ('-5ms', '+5ms')] + ['r2']:
                    if c in ctrl:
                        v = _vals(ctrl[c], keys, meas)
                        if np.isfinite(v).any():
                            m, lo, hi = probe.bootstrap_mean(v, recs)
                            cells[c] = dict(mean=m, lo=lo, hi=hi, n=int(np.isfinite(v).sum()))
                if meas in ('m1', 'm1_bar'):
                    ch = _vals(base, keys, 'chance' if meas == 'm1' else 'chance_bar')
                    cells['chance'] = dict(mean=_mean(ch))
                row[meas] = cells
            m2 = {}
            for arm in ARMS:
                finals = [arms[(arm, sd)][-1] for sd in (0, 1) if (arm, sd) in arms]
                if finals:
                    m2[arm] = {c: _mean([f[k]['m2'][c] for f in finals for k in keys if k in f])
                               for c in ('tempo', 'octave', '3:2', 'other')}
            for c in list(BT) + ['r2']:
                if c in ctrl:
                    m2[c] = {cl: _mean([ctrl[c][k]['m2'][cl] for k in keys if k in ctrl[c]])
                             for cl in ('tempo', 'octave', '3:2', 'other')}
            row['m2'] = m2
            tab[name] = row
            if pnl == panel and s is not None and arms and all(b in ctrl for b in BT):
                meas = _measure(s)
                bt_means = {b: _mean(_vals(ctrl[b], keys, meas)) for b in BT}
                best = max(bt_means, key=lambda b: (bt_means[b] if np.isfinite(bt_means[b]) else -1))
                bt_spread = float(np.ptp([bt_means[b] for b in BT]))
                for arm in ARMS:
                    finals = [arms[(arm, sd)][-1] for sd in (0, 1) if (arm, sd) in arms]
                    if not finals:
                        continue
                    avg = np.nanmean(np.vstack([_vals(f, keys, meas) for f in finals]), axis=0)
                    d, lo, hi = probe.bootstrap_diff(avg, _vals(ctrl[best], keys, meas), recs)
                    cell = row[meas][arm]
                    noise = [x for x in (cell['seed_spread'], bt_spread, cell['ckpt_spread']) if x is not None]
                    m_s = max(noise) if noise else float('nan')
                    decision.setdefault(arm, {})[s] = dict(measure=meas, delta=d, lo=lo, hi=hi, m_s=m_s,
                                                           seed_spread=cell['seed_spread'], bt_spread=bt_spread,
                                                           ckpt_spread=cell['ckpt_spread'], best_bt=best,
                                                           counted=row['counted'], n_recordings=n_rec,
                                                           beyond=bool(d > m_s and lo > 0),
                                                           above_margin=bool(d > m_s))
        tables[pnl] = tab
    # The reading.
    readings = {}
    for arm, dec in decision.items():
        rule = [s for s in RULE_STRATA if s in dec and dec[s]['counted']]
        beyond = [s for s in rule if dec[s]['beyond']]
        above = [s for s in rule if dec[s]['above_margin']]
        s0 = dec.get('S0')
        gain = _history_gain(run, arm)
        if len(beyond) >= 2:
            r = 'O2' if s0 and s0['delta'] < -s0['m_s'] else 'O1'
        elif not above:
            r = 'O4' if gain is not None and gain >= 0.01 else 'O3'
        else:
            r = 'unmatched'
        readings[arm] = dict(reading=r, beyond=beyond, above_margin=above, counted_rule_strata=rule,
                             val_s0_gain_last_third=gain)
    means = {arm: float(np.mean([dec[s]['delta'] for s in RULE_STRATA if s in dec and dec[s]['counted']] or [np.nan]))
             for arm, dec in decision.items()}
    better = max(means, key=lambda k: means[k] if np.isfinite(means[k]) else -9) if means else None
    order = dict(O2=0, O1=1, unmatched=2, O4=3, O3=4)
    overall = min((v['reading'] for v in readings.values()), key=lambda x: order[x]) if readings else 'incomplete'
    a2a1 = 'incomplete'
    if 'A1' in decision and 'A2' in decision:
        diffs = {}
        for s in RULE_STRATA:
            if s in decision['A1'] and decision['A1'][s]['counted']:
                d = decision['A2'][s]['delta'] - decision['A1'][s]['delta']
                m = max(decision['A1'][s]['m_s'], decision['A2'][s]['m_s'])
                diffs[s] = dict(a2_minus_a1=d, m=m)
        if sum(v['a2_minus_a1'] > v['m'] for v in diffs.values()) >= 2:
            a2a1 = 'F2 stays a family'
        elif diffs and all(v['a2_minus_a1'] <= 0 for v in diffs.values()):
            a2a1 = 'F2 retired as a family; phase becomes an auxiliary loss'
        else:
            a2a1 = 'both go to R2'
        a2a1 = dict(result=a2a1, per_stratum=diffs)
    nxt = dict(O1='Own-encoder families go ahead; R2 trains NE-M',
               O2='Own-encoder families go ahead; R2 trains NE-M, plus a BeatThis-logit input channel as an R2 arm',
               O3='Own encoders deferred; R2 runs F1 decoder on BeatThis evidence (shape A v0), R3 runs F5',
               O4='R1b: the better arm, seed 0, 40 minutes, the same rule',
               unmatched='No row of the pre-set table matches; the build-order consequence is for the coordinator',
               incomplete='Incomplete').get(overall)
    res = dict(reading=overall, per_arm=readings, a2_vs_a1=a2a1, next=nxt, decision_panel=panel,
               better_arm=better, mean_rule_delta=means,
               decision=decision, tables=tables, interim_strata=True,
               note='Strata are interim recording-level flags (ensomi_n.data.strata); decision panel is fit_dev 322')
    (run / 'verdict.json').write_text(json.dumps(res, indent=1, default=float))
    return res
