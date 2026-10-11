"""Requests, the validator and generation under rule L (plan v4 sections 4.6, 5.2, 5.7, amended for q10).

T-R validator, T-A validity, T-C withdrawal before the start and no cancellation after it, T-L1 and
T-L2 (the two locality conditions in free run), record completeness, readouts never echo, the
baseline slot. Engineering tests, not evidence about learning.
"""
import json

import numpy as np
import pytest
import torch

from ensomi_model.r2.common import ContractError
from ensomi_model.r2.features import Chart
from ensomi_model.r2.generate import RECORD_FIELDS, generate, readouts
from ensomi_model.r2.properties import ln_share, prefix_objects
from ensomi_model.r2.request_set import Eta, Request, RequestSet, Target, effective_track
from ensomi_model.r2.sampling import continue_chart

from .helpers import tiny_model
from .locality_fixture import T, fixture_chart

LN, STAR = 'ln_share', 'difficulty'


@pytest.fixture(scope='module')
def chart():
    return fixture_chart()


@pytest.fixture(scope='module')
def model():
    return tiny_model(torch.float64)


def req(i, a, b, **targets):
    return Request(str(i), (a, b), {k: Target(v) for k, v in targets.items()})


# ---- T-R ------------------------------------------------------------------------------------------

def test_t_r_overlap_reserved_fields_and_flags(chart):
    rs = RequestSet(T)
    rs.add(req('a', 2000.0, 3000.0, ln_share=0.5))
    with pytest.raises(ContractError, match='same-quantity overlap'):
        rs.add(req('b', 2500.0, 3500.0, ln_share=0.2))
    with pytest.raises(ContractError, match='Unsupported nu'):
        rs.add(Request('c', (2500.0, 3500.0), {LN: Target(0.2, nu='ln_share/other')}))
    rs.add(req('adj', 3000.0, 4000.0, ln_share=0.5))                      # adjacent: accepted, kept separate
    long = req('star', 0.0, T, difficulty=3.0)
    with pytest.raises(ContractError, match='30 s'):
        rs.add(long)                                                        # the fixture song is under 30 s
    rs2 = RequestSet(40_000.0)
    rs2.add(req('ln', 1000.0, 2000.0, ln_share=0.3))
    rs2.add(req('st', 0.0, 40_000.0, difficulty=3.0))                       # different quantity: both apply
    head = np.array([1000.0 + 100 * k for k in range(80)])
    from .helpers import two_segment_grid
    eff = effective_track(rs2, head, two_segment_grid())
    assert len(eff.track) == 2 and eff.co_active == [dict(a=dict(request='ln', kind=LN), b=dict(request='st', kind=STAR))]
    eff1 = effective_track(rs, chart.head_ms, chart.grid)
    assert [(iv.a, iv.b) for iv in eff1.track] == [(2000.0, 3000.0), (3000.0, 4000.0)]   # no merge
    for bad in (Request('x', (5000.0, 6000.0), {LN: Target(0.2, strength='high')}),
                Request('x', (5000.0, 6000.0), {LN: Target(0.2)}, eta=Eta(priority=1)),
                Request('x', (5000.0, 6000.0), {LN: Target(0.2)}, eta=Eta(transitions=(('lead-in', 500.0),))),
                Request('x', (5000.0, 6000.0), {LN: Target(0.2)}, demand={'x': 1}),
                Request('x', (5000.0, 6000.0), {LN: Target(0.2)}, style={'named': {'jack': 'prominent'}}),
                Request('x', (5000.0, 6000.0), {}),
                Request('x', (5000.0, 6000.0), {LN: Target((0.2, 0.4))}),
                Request('x', (5000.0, 6000.0), {LN: Target(1.2)}),
                Request('x', (5000.0, 10_000.0), {LN: Target(0.2)})):
        with pytest.raises(ContractError):
            RequestSet(T).add(bad)
    with pytest.raises(ContractError, match='at or after its start'):
        RequestSet(T).add(req('late', 5000.0, 6000.0, ln_share=0.2), frontier=5000.0)
    headless = RequestSet(T).add(req('h', 100.0, 900.0, ln_share=0.0))
    flags = [f['flag'] for f in effective_track(headless, chart.head_ms, chart.grid).flags]
    assert any(f.startswith('unattainable') for f in flags) and any(f.startswith('ungovernable') for f in flags)


# ---- T-A ------------------------------------------------------------------------------------------

def test_t_a_validity(chart, model):
    RequestSet(T).add(req('zero', 0.0, 1500.0, ln_share=0.4))               # a = 0 at the start of generation
    s = 20                                                                    # chart seed of 20 decisions
    g0 = float(chart.head_ms[s - 1])
    with pytest.raises(ContractError):
        RequestSet(T).add(req('x', g0, 4000.0, ln_share=0.4), frontier=g0)
    rs = RequestSet(T).add(req('ok', g0 + 1.0, 4000.0, ln_share=0.4), frontier=g0)
    # a continuation call whose frontier has passed a keeps the request
    acts, gap, rec = generate(model, chart.head_ms, chart.song_ms, chart.grid, rs, seed_actions=chart.actions[:s],
                              seed_gap=chart.gap[:s], random_seed=3, stop=30)
    acts2, gap2, rec2 = generate(model, chart.head_ms, chart.song_ms, chart.grid, rs, seed_actions=acts,
                                 seed_gap=gap, random_seed=3, stop=35)
    assert [e['request']['id'] for e in rec2['requests']['entries']] == ['ok']
    assert len(rec2['effective_track']) == 1
    with pytest.raises(ContractError):
        rs.check_frontier(float(chart.head_ms[s - 3]))                        # cannot resume before its addition


# ---- T-C (amended: q10) ---------------------------------------------------------------------------

def test_t_c_withdrawal_and_no_cancellation_after_the_start(chart, model):
    s = 20
    g0 = float(chart.head_ms[s - 1])
    keep = req('keep', 3050.0, 3950.0, ln_share=0.6)
    gone = req('gone', 5000.0, 6000.0, ln_share=0.9)
    with_w = RequestSet(T).add(keep, g0).add(gone, g0)
    with_w.withdraw('gone', g0)
    never = RequestSet(T).add(keep, g0)
    a1, g1, r1 = generate(model, chart.head_ms, chart.song_ms, chart.grid, with_w, seed_actions=chart.actions[:s],
                          seed_gap=chart.gap[:s], random_seed=7)
    a2, g2, r2 = generate(model, chart.head_ms, chart.song_ms, chart.grid, never, seed_actions=chart.actions[:s],
                          seed_gap=chart.gap[:s], random_seed=7)
    assert np.array_equal(a1, a2) and np.array_equal(g1, g2, equal_nan=True)
    assert [e['withdrawn'] for e in r1['requests']['entries']] == [False, True]      # the record lists it
    rs = RequestSet(T).add(req('run', 3050.0, 3950.0, ln_share=0.6), g0)
    for frontier in (3050.0, 3100.0, 3949.0, 5000.0):
        with pytest.raises(ContractError, match='neither cancelled nor changed'):
            rs.copy().withdraw('run', frontier)
        with pytest.raises(ContractError, match='neither cancelled nor changed'):
            rs.copy().replace('run', req('new', 3500.0, 3950.0, ln_share=0.1), frontier)
    rep = rs.copy().replace('run', req('new', 3100.0, 3950.0, ln_share=0.1), 3000.0)  # before the start: allowed
    assert [e.request.id for e in rep.active()] == ['new']


# ---- T-L1, T-L2 -----------------------------------------------------------------------------------

def tf_lps(model, acts, gap, chart, track, lo, hi):
    c = Chart(chart.head_ms, chart.song_ms, chart.grid, acts, gap)
    with torch.no_grad():
        return model.window(c, lo, hi, track).total.numpy()


@pytest.mark.parametrize('s', [5, 20])
@pytest.mark.parametrize('seed', [954, 955, 956])
def test_t_l1_law_before_a(chart, model, s, seed):
    g0 = float(chart.head_ms[s - 1])
    other = req('other', 5000.0, 6000.0, ln_share=0.3)
    for u in (req('u1', 3050.0, 3950.0, ln_share=0.9), req('u2', 2000.0, 3000.0, ln_share=0.1)):
        if not g0 < u.scope[0]:
            continue
        full = RequestSet(T).add(u, g0).add(other, g0)
        less = RequestSet(T).add(other, g0)
        a1, g1, r1 = generate(model, chart.head_ms, T, chart.grid, full, seed_actions=chart.actions[:s],
                              seed_gap=chart.gap[:s], random_seed=seed)
        a2, g2, _ = generate(model, chart.head_ms, T, chart.grid, less, seed_actions=chart.actions[:s],
                             seed_gap=chart.gap[:s], random_seed=seed)
        first = next(t['rule_l']['first_visible'] for t in r1['targets'] if t['request'] == u.id)
        a = u.scope[0]
        rows1 = [(o.start, o.end, o.lane) for o in prefix_objects(chart.head_ms, a1, g1) if o.start < a]
        rows2 = [(o.start, o.end, o.lane) for o in prefix_objects(chart.head_ms, a2, g2) if o.start < a]
        assert [(s_, e if e is None or e < a else None, l) for s_, e, l in rows1] == \
               [(s_, e if e is None or e < a else None, l) for s_, e, l in rows2]
        assert np.array_equal(a1[:first], a2[:first]) and np.array_equal(g1[:first], g2[:first], equal_nan=True)
        e1 = effective_track(full, chart.head_ms, chart.grid).track
        e2 = effective_track(less, chart.head_ms, chart.grid).track
        assert np.array_equal(tf_lps(model, a1, g1, chart, e1, s, first), tf_lps(model, a1, g1, chart, e2, s, first))


@pytest.mark.parametrize('seed', [954, 955, 956])
def test_t_l2_law_after_b(chart, model, seed):
    s = 15
    g0 = float(chart.head_ms[s - 1])
    u = req('u', 3050.0, 3950.0, ln_share=0.9)
    full = RequestSet(T).add(u, g0)
    exit_k = 30
    pre_a, pre_g, _ = generate(model, chart.head_ms, T, chart.grid, full, seed_actions=chart.actions[:s],
                               seed_gap=chart.gap[:s], random_seed=seed, stop=exit_k)
    a1, g1, _ = generate(model, chart.head_ms, T, chart.grid, full, seed_actions=pre_a, seed_gap=pre_g,
                         random_seed=seed + 10)
    a2, g2, _ = generate(model, chart.head_ms, T, chart.grid, RequestSet(T), seed_actions=pre_a, seed_gap=pre_g,
                         random_seed=seed + 10)
    assert np.array_equal(a1, a2) and np.array_equal(g1, g2, equal_nan=True)   # closes of S-born holds included
    e1 = effective_track(full, chart.head_ms, chart.grid).track
    assert np.array_equal(tf_lps(model, a1, g1, chart, e1, exit_k, chart.K + 1),
                          tf_lps(model, a1, g1, chart, (), exit_k, chart.K + 1))


# ---- the generation record ------------------------------------------------------------------------

def test_record_completeness_and_no_echo(chart, model):
    s = 15
    g0 = float(chart.head_ms[s - 1])
    rs = RequestSet(T).add(req('u', 3050.0, 3950.0, ln_share=0.9), g0)
    acts, gap, rec = generate(model, chart.head_ms, T, chart.grid, rs, seed_actions=chart.actions[:s],
                              seed_gap=chart.gap[:s], random_seed=954, code=dict(package_sha256='x'))
    for f in RECORD_FIELDS:
        assert f in rec, f
    for f in ('operating_point', 'nu', 'decoding', 'random_seed', 'chart_seed', 'baseline', 'requests',
              'effective_track', 'targets', 'defects'):
        assert rec[f] is not None, f
    assert rec['chart_seed']['decisions'] == s and rec['chart_seed']['g0'] == g0
    assert rec['baseline'].startswith('none')
    t = rec['targets'][0]
    for f in ('readout', 'deviation', 'undefined', 'crossing', 'rule_l', 'target', 'nu', 'strength'):
        assert f in t, f
    objects = prefix_objects(chart.head_ms, acts, gap)
    assert t['readout'] == ln_share(objects, 3050.0, 3950.0, T)
    # overwrite the request record: the readout of the same chart does not change
    forged = RequestSet(T).add(req('u', 3050.0, 3950.0, ln_share=0.1), g0)
    eff = effective_track(forged, chart.head_ms, chart.grid)
    again = readouts(Chart(chart.head_ms, T, chart.grid, acts, gap), objects, eff)
    assert again[0]['readout'] == t['readout'] and again[0]['deviation'] != t['deviation']
    json.dumps(rec, default=str)


def test_baseline_slot_admits_only_none(chart, model):
    with pytest.raises(ContractError):
        continue_chart(model, chart.head_ms, T, chart.grid, baseline={'identity': 1})
    with pytest.raises(ContractError):
        generate(model, chart.head_ms, T, chart.grid, RequestSet(T), baseline='ref')
