"""Plan v4 sections 4.6 and 6.4: rule L, ownership equal to visibility and the condition-scoped loss.

T-V (visibility, input reach, loss ownership, power checks), T-P1, T-P2, T-P3a, T-P3b, T-E, T-G,
T-Z and the input lesions, on the fixture of ``locality_fixture.py`` with tiny models whose
zero-initialised layers carry small random weights. Engineering tests, not evidence about learning.
"""
import numpy as np
import pytest
import torch

from ensomi_model.r2.common import ACTIONS
from ensomi_model.r2.features import FRAME_DIM, Interval, contains, frames
from ensomi_model.r2.locality import masked, visible
from ensomi_model.r2.loss import LossConfig, masks, window_loss, window_terms
from ensomi_model.r2.model import LN_GROUPS, held_pattern

from .helpers import tiny_model
from .locality_fixture import (ADJ_A, ADJ_B, EXPECTED_V, I1, I2, STAR, T, WHOLE, ZERO, fixture_chart, with_value)

CFG = LossConfig(n_bar=200.0, n_bar_ln=60.0, n_bar_star=60.0, lambda_ln=1.0, lambda_star=1.0)


@pytest.fixture(scope='module')
def chart():
    return fixture_chart()


@pytest.fixture(scope='module')
def model():
    return tiny_model(torch.float64)


def lps(model, chart, track, start=0, stop=None):
    """Per-decision (action, release) log-probabilities as numpy arrays."""
    stop = chart.K + 1 if stop is None else stop
    with torch.no_grad():
        out = model.window(chart, start, stop, track)
    return out.action.numpy().copy(), out.release.numpy().copy(), out


def v_of(chart, iv, rule_l=True):
    return [k for k in range(chart.K + 1) if iv in visible(chart, (iv,), k, rule_l)]


# ---- T-V ------------------------------------------------------------------------------------------

def test_t_v1_visibility_matches_the_hand_list(chart):
    for name, iv in (('I1', I1), ('I2', I2), ('WHOLE', WHOLE), ('ADJ_A', ADJ_A), ('ADJ_B', ADJ_B)):
        assert v_of(chart, iv) == EXPECTED_V[name], name
    m1, m2 = masked(chart, I1), masked(chart, I2)
    assert m1['onset'] == 10 and m1['onset_masked'] and m1['first_visible'] == 11
    assert m2['onset'] == 21 and not m2['onset_masked'] and m2['exit'] == 30
    assert m2['exit_closes'] == [] and I2.b <= chart.gap[30, 1] < 4000.0       # closed at or after b
    assert chart.candidates(30).times.min() < I2.b < chart.candidates(30).times.max()   # case (iii)
    assert np.isfinite(chart.gap[33, 2])                                                # case (iv)


EDITS = {
    'I1': [with_value(I1, 0.1), Interval(0, 2000.0, 2950.0, 0.7), None],
    'I2': [with_value(I2, 0.1), Interval(0, 3050.0, 3940.0, 0.6), None],
    'WHOLE': [with_value(WHOLE, 0.95), None],
}


@pytest.mark.parametrize('name', ['I1', 'I2', 'WHOLE'])
def test_t_v2_input_reach(model, chart, name):
    iv = dict(I1=I1, I2=I2, WHOLE=WHOLE)[name]
    a0, r0, _ = lps(model, chart, (iv,))
    vk = set(EXPECTED_V[name])
    for edit in EDITS[name]:
        track = () if edit is None else (edit,)
        a1, r1, _ = lps(model, chart, track)
        for k in range(chart.K + 1):
            if k not in vk:
                assert a0[k] == a1[k] and r0[k] == r1[k], (name, edit, k)
        if edit is not None and edit.value != iv.value:
            for k in vk:
                assert a0[k] != a1[k] or r0[k] != r1[k], (name, k)


def test_t_v3_loss_ownership_equals_visibility(model, chart):
    track = (I1, I2, STAR)
    _, _, out = lps(model, chart, track)
    natural, ln, star = masks(out)
    want_ln = set(EXPECTED_V['I1']) | set(EXPECTED_V['I2'])
    assert set(np.flatnonzero(ln)) == want_ln
    assert set(np.flatnonzero(star)) == set(v_of(chart, STAR))
    assert set(np.flatnonzero(natural)) == set(range(chart.K + 1)) - want_ln - set(v_of(chart, STAR))


def test_t_v4_power_rule_l_off_and_birth_role(chart):
    """Without rule L the masked onset (i) and the exit decision (iii) read the scope; with the v1 birth
    role restored too, the close three decisions after b (iv) reads it."""
    off = tiny_model(torch.float64, rule_l=False)
    a0, r0, _ = lps(off, chart, (I1,))
    a1, r1, _ = lps(off, chart, (with_value(I1, 0.1),))
    assert a0[10] != a1[10]                                     # (i): onset decision 10 reads I1
    a0, r0, _ = lps(off, chart, (I2,))
    a1, r1, _ = lps(off, chart, (with_value(I2, 0.1),))
    assert r0[30] != r1[30]                                     # (iii): exit decision's candidates in [a, b)
    assert a0[33] == a1[33] and r0[33] == r1[33]                # (iv) needs the birth role
    birth = tiny_model(torch.float64, rule_l=False, birth_role=True)
    a0, r0, _ = lps(birth, chart, (I2,))
    a1, r1, _ = lps(birth, chart, (with_value(I2, 0.1),))
    assert r0[33] != r1[33]                                     # (iv): the birth-role frame reads S


def test_t_e_whole_song_scope_contains_t(model, chart):
    assert bool(contains(WHOLE, T, T)) and not bool(contains(I1, I1.b, T))
    row = frames(chart, (WHOLE,), [T], chart.K)
    assert row[0, 0, FRAME_DIM - 2] == 1.0                      # active at the EOS row time
    a0, r0, _ = lps(model, chart, (WHOLE,))
    a1, r1, _ = lps(model, chart, (with_value(WHOLE, 0.95),))
    assert r0[chart.K] != r1[chart.K] and a0[chart.K] == a1[chart.K] == 0.0
    # the EOS reads the scope through the row role as well: lesion the row role at EOS only
    film = model.film

    def lesioned(z, cond):
        cond = cond.clone()
        cond[..., :2 * FRAME_DIM] = 0
        return type(film).forward(film, z, cond)
    with torch.no_grad():
        base = model.window(chart, chart.K, chart.K + 1, (WHOLE,)).release[0]
        model.film.forward = lesioned
        try:
            cut = model.window(chart, chart.K, chart.K + 1, (WHOLE,)).release[0]
        finally:
            del model.film.forward
    assert base != cut


# ---- T-G, T-Z -------------------------------------------------------------------------------------

def test_t_g_governed_split(model, chart):
    _, _, out = lps(model, chart, (I1,))
    d = chart.derived()
    for i in (5, 12, 14, 29):
        logp = out.logp[i].numpy()
        g = LN_GROUPS[held_pattern(d.held[i])[0]]
        legal = np.isfinite(logp)
        p = np.exp(logp)
        for a in np.flatnonzero(legal):
            same = legal & (g == g[a])
            lse = np.log(p[same].sum())
            assert abs(np.exp((logp[a] - lse) + lse) - p[a]) < 1e-12
        held = d.held[i]
        heads = np.where(held[None], ACTIONS >= 3, (ACTIONS == 1) | (ACTIONS == 2)).sum(1)
        lns = np.where(held[None], ACTIONS == 4, ACTIONS == 2).sum(1)
        frac = np.where(heads > 0, lns / np.maximum(heads, 1), 0.0)
        full = (p * frac)[legal].sum()
        groups = {}
        for a in np.flatnonzero(legal):
            groups.setdefault(g[a], []).append(a)
        via = sum(p[m].sum() * sum(p[a] / p[m].sum() * frac[a] for a in m) for m in map(np.array, groups.values()))
        assert abs(full - via) < 1e-12
    # the target decision's governed log-probability is what the window reports
    tgt = [int(((c[0] * 5 + c[1]) * 5 + c[2]) * 5 + c[3]) for c in chart.actions[:chart.K + 1]]
    i = 14
    logp = out.logp[i].numpy()
    g = LN_GROUPS[held_pattern(d.held[i])[0]]
    same = np.isfinite(logp) & (g == g[tgt[i]])
    assert abs(float(out.governed_ln[i]) - (logp[tgt[i]] - np.log(np.exp(logp[same]).sum()))) < 1e-12


def test_t_z_explicit_zero_differs_from_no_request(chart):
    m = tiny_model(torch.float64)
    a0, r0, _ = lps(m, chart, ())
    a1, r1, _ = lps(m, chart, (ZERO,))
    assert a0[42] != a1[42]
    res = tiny_model(torch.float64, star_value='residual')
    s = with_value(STAR, 0.0)                                      # difficulty equal to b(S): residual 0
    a0, _, _ = lps(res, chart, ())
    a1, _, _ = lps(res, chart, (s,))
    k = v_of(chart, s)[3]
    assert a0[k] != a1[k]


# ---- T-P1, T-P2, T-P3 -------------------------------------------------------------------------------

def kind_losses(model, chart, track, start, stop, weight=1.0, grad=False):
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        out = model.window(chart, start, stop, track)
        terms = window_terms(out, weight)
        total, parts = window_loss(terms, CFG)
    return out, terms, parts


def test_t_p1_targets_outside_omega_do_not_move_the_terms(model, chart):
    track = (I1, STAR)
    start, stop = 5, 40
    _, _, parts = kind_losses(model, chart, track, start, stop)
    last_ln = max(EXPECTED_V['I1'])
    # change the decision right after the last Omega_LN decision: L_LN unchanged
    acts, gap = chart.actions.copy(), chart.gap.copy()
    k = last_ln + 1
    acts[k] = (1, 1, 0, 0) if tuple(acts[k]) != (1, 1, 0, 0) else (1, 0, 1, 0)
    gap[k] = np.nan
    from ensomi_model.r2.features import Chart
    other = Chart(chart.head_ms, chart.song_ms, chart.grid, acts[:k + 1], gap[:k + 1])
    _, _, p2 = kind_losses(model, other, track, start, k + 1)
    _, _, p1 = kind_losses(model, chart, track, start, k + 1)
    assert float(p1['ln']) == float(p2['ln'])
    # change the release targets at the last Omega_LN decision that releases: L_LN unchanged, L_star moves
    rel = [k for k in EXPECTED_V['I1'] if np.isfinite(chart.gap[k]).any()][-1]
    lane = int(np.flatnonzero(np.isfinite(chart.gap[rel]))[0])
    others = [u for u in chart.candidates(rel).times if u != chart.gap[rel, lane]]
    gap2 = chart.gap.copy()
    gap2[rel, lane] = others[0]
    moved = Chart(chart.head_ms, chart.song_ms, chart.grid, chart.actions, gap2)
    _, _, q1 = kind_losses(model, chart, track, start, rel + 1)
    _, _, q2 = kind_losses(model, moved, track, start, rel + 1)
    assert float(q1['ln']) == float(q2['ln'])
    assert rel in v_of(chart, STAR) and float(q1['star']) != float(q2['star'])
    assert np.isfinite(float(parts['ln']))


@pytest.mark.parametrize('kind', ['ln', 'star'])
def test_t_p2_gradients_are_zero_outside_omega(model, chart, kind):
    track = (I1, I2, STAR)
    out, terms, parts = kind_losses(model, chart, track, 0, 50, grad=True)
    natural, ln, star = masks(out)
    own = ln if kind == 'ln' else star
    g_logp, g_rel = torch.autograd.grad(parts[kind], [out.logp, out.release], allow_unused=True)
    rows = g_logp.abs().sum(1).numpy()
    assert (rows[~own] == 0).all() and (rows[own & ~out.eos] > 0).all()
    if kind == 'ln':
        assert g_rel is None or float(g_rel.abs().sum()) == 0.0
    else:
        has = out.gap_lns > 0
        r = g_rel.abs().numpy()
        assert (r[~own] == 0).all() and (r[own & has] > 0).all()


def test_t_p3a_fixed_divisors(model, chart):
    _, _, p1 = kind_losses(model, chart, (I1,), 5, 40)
    _, _, p0 = kind_losses(model, chart, (), 40, 80)        # a natural-only window joins the batch
    for kind in ('ln', 'star'):
        assert float(p0[kind]) == 0.0
    total_one = float(p1['ln'])
    total_two = float(p1['ln']) + float(p0['ln'])
    assert total_one == total_two


def test_t_p3b_input_locality(chart):
    """(i) An interval in no V_k of a scored decision leaves L_kappa unchanged under presence 'none', rule L
    and FiLM, and moves it under presence 'anywhere' or rule L off. (ii) Likewise for an interval of the
    other kind."""
    masked_only = Interval(0, 4000.0, 4350.0, 0.5)      # its only decision in [25, 31) is the masked onset 30
    star_out = Interval(1, 8000.0, 8600.0, 2.0)         # far from the window
    start, stop = 25, 31
    m = tiny_model(torch.float64)
    base = kind_losses(m, chart, (I2,), start, stop)[2]
    for track in ((I2, masked_only), (I2, with_value(masked_only, 0.9)), (I2, star_out)):
        p = kind_losses(m, chart, track, start, stop)[2]
        assert float(p['ln']) == float(base['ln']), track
    anywhere = tiny_model(torch.float64, presence='anywhere')
    b = kind_losses(anywhere, chart, (I2,), start, stop)[2]
    assert float(kind_losses(anywhere, chart, (I2, star_out), start, stop)[2]['ln']) != float(b['ln'])
    off = tiny_model(torch.float64, rule_l=False)
    b = kind_losses(off, chart, (I2, masked_only), start, stop)[2]
    assert float(kind_losses(off, chart, (I2, with_value(masked_only, 0.9)), start, stop)[2]['ln']) != float(b['ln'])


def test_t_p3b_token_conditioner_breaks_locality(chart):
    tok = tiny_model(torch.float64, 'tokens')
    future = Interval(0, 8000.0, 8600.0, 0.1)
    a0, _, _ = lps(tok, chart, (future,), 0, 10)
    a1, _, _ = lps(tok, chart, (with_value(future, 0.9),), 0, 10)
    assert (a0 != a1).any()                                      # expected: tokens announce a future scope


# ---- input lesions per named input ----------------------------------------------------------------

CHANNELS = dict(value=0, offset_a=1, offset_b=5, progress=9, stat0=10, stat1=11, stat2=12, stat3=13,
                remaining=14, active=15)


def lesion_lp(model, chart, track, k, idx):
    film = model.film

    def lesioned(z, cond):
        cond = cond.clone()
        cond[..., idx] = 0
        return type(film).forward(film, z, cond)
    with torch.no_grad():
        model.film.forward = lesioned
        try:
            out = model.window(chart, k, k + 1, track)
        finally:
            del model.film.forward
    return float(out.action[0]), float(out.release[0])


@pytest.mark.parametrize('kind,channel', [(0, 'value'), (0, 'stat0'), (0, 'stat1'), (0, 'stat2'), (0, 'remaining'),
                                          (0, 'active'), (0, 'progress'), (0, 'offset_a'), (0, 'offset_b'),
                                          (1, 'value'), (1, 'stat0'), (1, 'stat1'), (1, 'stat2'), (1, 'stat3')])
def test_lesion_each_frame_channel(chart, kind, channel):
    m = tiny_model(torch.float64, star_value='residual')
    star = with_value(STAR, 0.3)
    k = 18 if kind == 0 else 29                     # late in I1 (counters non-zero) / late in the star scope
    track = (I1, star) if kind == 0 else (star, I2)
    with torch.no_grad():
        base = m.window(chart, k, k + 1, track)
    idx = [CHANNELS[channel] + kind * FRAME_DIM]     # row role
    assert lesion_lp(m, chart, track, k, idx)[0] != float(base.action[0]), (kind, channel)


def test_lesion_roles_and_presence(chart):
    m = tiny_model(torch.float64)
    k = 15                                           # reads I1 and gap-releases lane 3 with candidates in I1
    assert np.isfinite(chart.gap[k, 3]) and k in EXPECTED_V['I1']
    with torch.no_grad():
        base = m.window(chart, k, k + 1, (I1,))
    row = list(range(0, 2 * FRAME_DIM))
    cand = list(range(2 * FRAME_DIM, 4 * FRAME_DIM))
    assert lesion_lp(m, chart, (I1,), k, row)[0] != float(base.action[0])
    assert lesion_lp(m, chart, (I1,), k, cand)[1] != float(base.release[0])
    p = tiny_model(torch.float64, presence='anywhere')
    with torch.no_grad():
        b = p.window(chart, 5, 6, (I2,))
    assert lesion_lp(p, chart, (I2,), 5, [FRAME_DIM - 1])[0] != float(b.action[0])


def test_unused_inputs_are_not_accepted():
    from ensomi_model.r2.common import ContractError
    from ensomi_model.r2.model import R2Config
    for bad in (dict(presence='lead-in'), dict(conditioner='tokens'), dict(star_value='relative')):
        with pytest.raises(ContractError):
            R2Config(**bad)
