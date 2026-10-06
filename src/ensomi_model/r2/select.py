"""The frozen checkpoint selection rule and the default adherence report (plan v4 section 9.6).

Candidates are regular checkpoints after warm-up (safe checkpoints excluded) with a full
evaluation whose free runs are all legal with every head present. The primary is the
natural-manifest per-decision NLL, read as the mean over the checkpoint and its two
predecessors, with a paired song-group bootstrap SE (2,000 resamples). Guards: (i) binding,
the prefix-panel chart-paired continuation LN-share difference within +-0.05; (ii) onset
span following slope >= 0.7 and MAE <= 0.15; (iii) own-history gap <= 0.05; (iv) holds
<= 60 ms at most 0.5 % and releases 1-40 ms before another head at most the source panel's
rate; (a1) the released-property identity check. The selected checkpoint is the earliest
candidate passing every guard whose primary is within 2 SE of the minimum among passing
candidates. The adherence report lists, per property and scope length, the deviation
distribution of the selected checkpoint's panels, undefined readouts apart. A phase-N run
(config ``phase: natural``, plan v5) has no conditioned panels: guards (ii) and (a1) do not
apply and no adherence report is written.

    .venv/bin/python -m ensomi_model.r2.select --run artifacts/r2-runs/<run-id>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from .operating_point import SELECTION_RULE, operating_point

BOOT = 2000


def load_evals(run: Path):
    rows = [json.loads(l) for l in (run / 'evals.jsonl').read_text().splitlines() if l.strip()]
    return [r for r in rows if 'error' not in r]


def window_table(record):
    w = record['natural_manifest']['windows']
    return np.array([x['nll'] for x in w]), np.array([x['decisions'] for x in w]), [x['group_id'] for x in w]


def primary_samples(chain, rng, groups):
    """Per bootstrap resample of song groups, the 3-checkpoint mean per-decision NLL of ``chain``."""
    tabs = [window_table(r) for r in chain]
    uniq = sorted(set(groups))
    gi = {g: np.flatnonzero(np.array(groups) == g) for g in uniq}
    point = float(np.mean([t[0].sum() / t[1].sum() for t in tabs]))
    out = []
    for _ in range(BOOT):
        ix = np.concatenate([gi[uniq[j]] for j in rng.choice(len(uniq), len(uniq), replace=True)])
        out.append(np.mean([t[0][ix].sum() / t[1][ix].sum() for t in tabs]))
    return point, np.array(out)


def guards(record, phase='conditions'):
    p = record.get('panels', {})
    g = {}
    g['legal'] = bool(p.get('legality', {}).get('illegal', 1) == 0 and p.get('legality', {}).get('heads_missing', 1) == 0)
    g['i'] = bool(p.get('prefix_natural', {}).get('guard_i'))
    g['ii'] = bool(p.get('onset', {}).get('guard_ii'))
    g['iii'] = bool(p.get('calibration', {}).get('guard_iii'))
    d = p.get('defects', {})
    src = p.get('source_defects', {})
    g['iv'] = bool(d.get('holds_le60_rate') is not None and d['holds_le60_rate'] <= 0.005 and
                   (d.get('release_1_40_rate') or 0.0) <= (src.get('release_1_40_rate') or 0.0))
    g['a1'] = bool(p.get('g3', {}).get('a1', {}).get('identical'))
    if phase == 'natural':
        del g['ii'], g['a1']
    return g


def select(run: Path, warmup: int, phase='conditions'):
    evals = load_evals(run)
    regular = [r for r in evals if not r.get('safe') and 'natural_manifest' in r]
    regular.sort(key=lambda r: r['exposures'])
    rng = np.random.default_rng(954)
    cands = []
    for i, r in enumerate(regular):
        if r['exposures'] <= warmup or i < 2 or 'panels' not in r:
            continue
        chain = regular[i - 2:i + 1]
        groups = window_table(r)[2]
        point, samples = primary_samples(chain, np.random.default_rng(954), groups)
        cands.append(dict(checkpoint=r['checkpoint'], exposures=r['exposures'], primary=point, samples=samples,
                          guards=guards(r, phase), record=r))
    passing = [c for c in cands if all(c['guards'].values())]
    result = dict(rule=SELECTION_RULE, candidates=[dict(checkpoint=c['checkpoint'], exposures=c['exposures'],
                                                        primary=c['primary'], guards=c['guards']) for c in cands])
    if not passing:
        result.update(selected=None, reason='no selection: no candidate passes every guard',
                      failing={c['checkpoint']: [k for k, v in c['guards'].items() if not v] for c in cands})
        return result, None
    best = min(passing, key=lambda c: c['primary'])
    for c in sorted(passing, key=lambda c: c['exposures']):
        se = float(np.std(c['samples'] - best['samples'], ddof=1))
        c['se_to_min'] = se
        if c['primary'] - best['primary'] <= 2 * se:
            result.update(selected=c['checkpoint'], exposures=c['exposures'], primary=c['primary'],
                          minimum=dict(checkpoint=best['checkpoint'], primary=best['primary']), se_to_min=se)
            return result, c['record']
    raise AssertionError


def adherence(record):
    p = record['panels']
    out = dict(onset={L: v['deviation'] for L, v in p['onset']['by_length'].items()},
               onset_masked_share=p['onset'].get('masked_onset_share'),
               whole_song=p['whole']['deviation'], switch=dict(did_half=p['switch']['did_half'],
                                                             per_half_mae=p['switch']['per_half_mae']))
    if 'residual_star' in p:
        out['residual_star'] = {L: v['deviation'] for L, v in p['residual_star']['by_length'].items()}
        out['abs_delta_tail'] = p['residual_star']['abs_delta_tail']
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--warmup-exposures', type=int, default=None)
    a = ap.parse_args(argv)
    run = Path(a.run)
    config = json.loads((run / 'config.json').read_text())
    warmup = a.warmup_exposures if a.warmup_exposures is not None else int(config.get('warmup_exposures') or 0)
    phase = config.get('phase') or 'conditions'
    result, record = select(run, warmup, phase)
    result['phase'] = phase
    if record is not None and phase != 'natural':    # a phase-N model has no adherence to report
        import torch
        ckpt = torch.load(run / 'checkpoints' / record['checkpoint'], map_location='cpu', weights_only=False)
        op = operating_point(config, ckpt['model_config'])
        report = dict(checkpoint=record['checkpoint'], operating_point=op['hash'],
                      operating_point_definition=op['definition'], adherence=adherence(record))
        text = json.dumps(report, indent=1, default=str)
        path = run / f'adherence-{Path(record["checkpoint"]).stem}.json'
        path.write_text(text)
        result['adherence_report'] = dict(path=path.name, sha256=hashlib.sha256(text.encode()).hexdigest())
        result['operating_point'] = op['hash']
    (run / 'selection.json').write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps({k: v for k, v in result.items() if k != 'candidates'}, indent=1, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
