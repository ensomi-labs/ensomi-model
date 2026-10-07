"""Conditioned generation with a request set, and the generation record (plan v4 sections 5.7, 11.5).

``generate`` takes a request set (each request with its addition boundary g_u), an optional
chart seed (the first s committed decisions; g0 = t_{s-1}), a random seed and the reserved
``baseline`` slot (only ``None``). It validates the set against the frontier, builds the
effective track, generates with rule L applied per decision, and returns the decisions with
a record of intended against realised: every request as given (withdrawn ones included), the
effective track with provenance and flags, what rule L left ungoverned in each scope, each
target's readout under nu and its deviation (undefined kept undefined), co-active targets
with their deviations, holds headed in a scope and closed at or after its end, and the free-run
defect counts. Readouts are computed from the generated objects, never copied from a request.

Request CLI (mac, repository root)::

    .venv/bin/python -m ensomi_model.r2.generate --checkpoint <ckpt.pt> --sha <fit_dev chart sha256> \
        --requests requests.json [--seed-decisions N | --continue <previous record.json>] \
        [--random-seed 954] [--stop N] --out artifacts/r2-generate/<id>

``requests.json`` is a list of ``Request.to_json`` objects; they are added at the chart seed's
frontier. ``--continue`` resumes from a previous record's decisions with its request set kept.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from .features import Chart
from .locality import RULE_L_VERSION, masked
from .operating_point import DECODING, operating_point
from .properties import (NU_HASH, NU_ID, PROPERTIES_SOURCE_SHA256, deviation, difficulty, ln_share, prefix_objects,
                         scope_contains)
from .request_set import KIND_NAME, RequestSet, effective_track
from .sampling import BASELINE_NONE, continue_chart, require_no_baseline


def chart_seed_record(head_ms, actions, gap):
    s = 0 if actions is None else len(actions)
    if not s:
        return dict(decisions=0, g0='0^-', sha256=None, open_holds=[])
    h = hashlib.sha256(np.ascontiguousarray(actions, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(np.nan_to_num(gap, nan=-1.0), dtype=np.float64).tobytes())
    chart = Chart(np.asarray(head_ms), 0.0, None, np.asarray(actions), np.asarray(gap))
    starts = chart.derived().start[s] if s <= len(head_ms) else np.full(4, np.nan)
    g0 = float(head_ms[s - 1]) if s <= len(head_ms) else 'T'
    return dict(decisions=s, g0=g0, sha256=h.hexdigest(),
                open_holds=[dict(lane=l, start=float(x)) for l, x in enumerate(starts) if np.isfinite(x)])


def frontier_of(head_ms, s: int):
    return None if s == 0 else float(head_ms[s - 1])


def defects(objects, head_ms) -> dict:
    holds = [o for o in objects if o.hold and o.end is not None]
    starts = np.array([o.start for o in objects])
    lanes = np.array([o.lane for o in objects])
    near = sum(bool(((starts >= o.end + 1) & (starts <= o.end + 40) & (lanes != o.lane)).any()) for o in holds)
    return dict(holds=len(holds), holds_le60=sum((o.end - o.start) <= 60 for o in holds),
                release_1_40_before_other_head=near)


def readouts(chart: Chart, objects, effective, baseline=None, complete=True):
    """Per target: readout under nu over S, deviation, crossing holds and rule-L masking."""
    T = chart.song_ms
    out = []
    for iv, prov in zip(effective.track, effective.provenance):
        kind = KIND_NAME[iv.kind]
        if kind == 'ln_share':
            value, info = ln_share(objects, iv.a, iv.b, T), {}
        else:
            value, info = difficulty(objects, iv.a, iv.b, T)
        m = masked(chart, iv)
        crossing = [dict(lane=o.lane, start=o.start, end=o.end) for o in objects
                    if o.hold and bool(scope_contains(iv.a, iv.b, T, o.start)) and (o.end is None or o.end >= iv.b)
                    and iv.b < T]
        entry = dict(prov, readout=value, deviation=deviation(value, prov['target']), undefined=value is None,
                     undefined_reason=info.get('invalid') if value is None else None, crossing=crossing,
                     rule_l=dict(first_visible=m['first_visible'], onset=m['onset'], exit=m['exit'],
                                 onset_masked=m['onset_masked'], in_scope_outside_v=m['in_scope_outside_v'],
                                 exit_closes=m['exit_closes']),
                     ungovernable=m['first_visible'] is None, complete=complete)
        if kind == 'difficulty' and prov.get('b_s') is not None and value is not None:
            entry['realised_residual'] = value - prov['b_s']
        out.append(entry)
    return out


def generation_record(*, chart: Chart, requests: RequestSet, effective, seed_rec, random_seed, model,
                      checkpoint=None, train_config=None, baseline=None, objects=None, complete=True, code=None):
    targets = readouts(chart, objects, effective, baseline, complete)
    by = {(t['request'], t['kind']): t for t in targets}
    co = [dict(pair, deviation_a=by[(pair['a']['request'], pair['a']['kind'])]['deviation'],
               deviation_b=by[(pair['b']['request'], pair['b']['kind'])]['deviation']) for pair in effective.co_active]
    op = operating_point(train_config, asdict(model.config))
    return dict(code=code, checkpoint=checkpoint, nu=dict(hash=NU_HASH, ids=NU_ID),
                evaluators=dict(properties_sha256=PROPERTIES_SOURCE_SHA256),
                baseline_model_sha256=getattr(baseline, 'sha256', None),
                operating_point=op['hash'], operating_point_definition=op['definition'], decoding=DECODING,
                rule_l=RULE_L_VERSION if model.config.rule_l else 'off', presence=model.config.presence,
                random_seed=int(random_seed), chart_seed=seed_rec, baseline=BASELINE_NONE,
                requests=requests.to_json(),
                effective_track=[dict(p, frame_value=iv.value)   # provenance already names kind, a and b
                                 for iv, p in zip(effective.track, effective.provenance)],
                flags=effective.flags, targets=targets, co_active=co,
                defects=defects(objects, chart.head_ms) if objects is not None else None, complete=complete)


def generate(model, head_ms, song_ms, grid, requests: RequestSet, *, seed_actions=None, seed_gap=None,
             random_seed: int = 954, stop=None, close_scope=None, baseline=None, baseline_model=None,
             checkpoint=None, train_config=None, code=None, ln_level='unknown', ln_length=None,
             source_ln_level=None, source_ln_length=None, ln_prior=None, star=None,
             theta='unknown', theta_table=None, theta_source_sha256=None):
    """Return (actions, gap, record); whole-song LN inputs follow ``continue_chart`` modes."""
    require_no_baseline(baseline)
    s = 0 if seed_actions is None else len(seed_actions)
    requests.check_frontier(frontier_of(head_ms, s))
    effective = effective_track(requests, head_ms, grid, star_value=model.config.star_value, baseline=baseline_model)
    level_stats = {}
    theta_record = {}
    acts, gap = continue_chart(model, head_ms, song_ms, grid, seed_actions, seed_gap, track=effective.track,
                               seed=random_seed, stop=stop, close_scope=close_scope, ln_level=ln_level,
                               ln_length=ln_length, source_ln_level=source_ln_level, source_ln_length=source_ln_length,
                               ln_prior=ln_prior, star=star, ln_level_stats=level_stats,
                               theta=theta, theta_table=theta_table,
                               theta_source_sha256=theta_source_sha256, theta_record=theta_record)
    chart = Chart(np.asarray(head_ms), float(song_ms), grid, acts, gap)
    objects = prefix_objects(head_ms, acts, gap)
    record = generation_record(chart=chart, requests=requests, effective=effective,
                               seed_rec=chart_seed_record(head_ms, seed_actions, seed_gap), random_seed=random_seed,
                               model=model, checkpoint=checkpoint, train_config=train_config,
                               baseline=baseline_model, objects=objects, complete=len(acts) == len(head_ms) + 1,
                               code=code)
    if level_stats:
        record['ln_level'] = level_stats
    if theta_record:
        record['theta'] = theta_record
    return acts, gap, record


RECORD_FIELDS = ('code', 'checkpoint', 'nu', 'evaluators', 'operating_point', 'decoding', 'rule_l', 'presence',
                 'random_seed', 'chart_seed', 'baseline', 'requests', 'effective_track', 'flags', 'targets',
                 'co_active', 'defects')


def main(argv=None):
    import torch
    from .data import Corpus
    from .export import export_chart, minimal_header
    from .model import R2Config, R2Model
    from .receipts import code_identity
    from .request_set import Request
    p = argparse.ArgumentParser(description='R2 generation with a request set')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--cache', default='artifacts/r2-cache/v1')
    p.add_argument('--sha', required=True)
    p.add_argument('--requests', default=None)
    p.add_argument('--seed-decisions', type=int, default=0)
    p.add_argument('--continue', dest='cont', default=None)
    p.add_argument('--random-seed', type=int, default=954)
    p.add_argument('--stop', type=int, default=None)
    p.add_argument('--ln-level', default='unknown', help='unknown, oracle, prior, or a numeric share in [0, 1]')
    p.add_argument('--ln-length', type=float, help='Optional median log2 length in beats with a fixed share')
    p.add_argument('--ln-prior', help='Empirical prior JSON; joint v2 when the length input is enabled')
    p.add_argument('--theta', default='unknown', help='prior, unknown, oracle, or a JSON standardised 10-vector')
    p.add_argument('--theta-table', default=None)
    p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    data = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    model = R2Model(R2Config(**data['model_config'])).eval()
    model.load_state_dict(data['model'])
    corpus = Corpus(a.cache, 'fit_dev', star_conditions=False)
    chart = corpus.chart(a.sha)
    from .ln_level import EmpiricalLNPrior, JOINT_PRIOR_FILE, PRIOR_FILE, whole_ln_length, whole_ln_level
    mode = a.ln_level if a.ln_level in ('unknown', 'oracle', 'prior') else float(a.ln_level)
    level_kwargs = dict(ln_level=mode, ln_length=a.ln_length)
    level_kwargs.update(theta=a.theta if a.theta in ('unknown', 'oracle', 'prior') else json.loads(a.theta),
                        theta_table=a.theta_table, theta_source_sha256=a.sha)
    if model.config.ln_level == 'on':
        if mode == 'oracle':
            level_kwargs['source_ln_level'] = whole_ln_level(chart)
            if model.config.ln_length == 'on':
                level_kwargs['source_ln_length'] = whole_ln_length(chart)
        elif mode == 'prior':
            filename = JOINT_PRIOR_FILE if model.config.ln_length == 'on' else PRIOR_FILE
            level_kwargs['ln_prior'] = EmpiricalLNPrior.load(a.ln_prior or Path(a.cache) / filename)
            level_kwargs['star'] = float(corpus.table.set_index('sha256').loc[a.sha, 'star'])
    baseline_model = None
    if model.config.star_value == 'residual':
        from .baseline import load_baseline
        baseline_model = load_baseline(a.cache)
    if a.cont:
        prev = json.loads(Path(a.cont).read_text())
        rs = RequestSet.from_json(prev['requests'])
        dec = np.load(Path(a.cont).with_suffix('.npz'))
        seed_actions, seed_gap = dec['actions'], dec['gap']
    else:
        rs = RequestSet(chart.song_ms)
        seed_actions = chart.actions[:a.seed_decisions] if a.seed_decisions else None
        seed_gap = chart.gap[:a.seed_decisions] if a.seed_decisions else None
    s = 0 if seed_actions is None else len(seed_actions)
    for r in json.loads(Path(a.requests).read_text()) if a.requests else []:
        rs.add(Request.from_json(r), frontier_of(chart.head_ms, s))
    acts, gap, record = generate(model, chart.head_ms, chart.song_ms, chart.grid, rs, seed_actions=seed_actions,
                                 seed_gap=seed_gap, random_seed=a.random_seed, stop=a.stop,
                                 baseline_model=baseline_model, train_config=data.get('config'),
                                 checkpoint=dict(path=a.checkpoint, sha256=hashlib.sha256(
                                     Path(a.checkpoint).read_bytes()).hexdigest()), code=code_identity(), **level_kwargs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f'{a.sha[:16]}-{a.random_seed}'
    np.savez(out / f'{stem}.npz', actions=acts, gap=gap)
    if record['complete']:
        dec = corpus.chart(a.sha)
        export_chart(dec.head_ms, dec.song_ms, acts, gap, out / f'{stem}.osu',
                     minimal_header(_segments(corpus, a.sha), f'R2 {a.sha[:12]} seed {a.random_seed}'))
    (out / f'{stem}.json').write_text(json.dumps(record, indent=1, default=str))
    print(json.dumps(dict(record=str(out / f'{stem}.json'), targets=record['targets']), indent=1, default=str))
    return 0


def _segments(corpus, sha):
    from .cache import load_chart
    return load_chart(corpus.root / 'charts' / corpus.files[sha]).grid_segments


if __name__ == '__main__':
    sys.exit(main())
