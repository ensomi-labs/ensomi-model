"""N1 v0.1 kill rules on v0's recorded probe traces and on synthetic divergence (numpy only).

The traces are the probe histories of `v0-s0-20261019` and `v0-s1-20261019` (``verdict.json``,
what the rule saw): minute, step, probe loss, M1 on S0 and on hard. v0's rule (the probe loss more than
10 % above its best for 3 probes) stopped them at minutes 104 and 80 of 150 while M1 held.
"""
from __future__ import annotations

import math

import numpy as np

from ensomi_n.v0 import train as T

SEED0 = [
    (4.01, 217, 3.19603, 0.6066, 0.6365), (8.01, 422, 2.50961, 0.7723, 0.7325), (12.01, 636, 2.33878, 0.8745, 0.8296),
    (16.01, 836, 2.05817, 0.8900, 0.8567), (20.02, 1043, 1.93508, 0.8914, 0.8895), (24.01, 1248, 2.19627, 0.9084, 0.8887),
    (28.01, 1454, 1.73974, 0.9293, 0.9194), (32.01, 1660, 1.84753, 0.9256, 0.8967), (36.00, 1864, 1.68299, 0.9229, 0.9178),
    (40.00, 2063, 1.64704, 0.9286, 0.9292), (44.01, 2261, 1.48914, 0.9579, 0.9202), (48.01, 2456, 1.63162, 0.9415, 0.9111),
    (52.01, 2656, 1.48620, 0.9400, 0.9351), (56.02, 2849, 1.29805, 0.9380, 0.9374), (60.00, 3042, 1.34684, 0.9428, 0.9210),
    (64.01, 3232, 1.40843, 0.9386, 0.9379), (68.01, 3417, 1.34997, 0.9594, 0.9346), (72.01, 3605, 1.27974, 0.9627, 0.9443),
    (76.00, 3793, 1.17265, 0.9534, 0.9338), (80.01, 3983, 1.25645, 0.9444, 0.9387), (84.00, 4171, 0.96171, 0.9586, 0.9431),
    (88.01, 4360, 0.94510, 0.9610, 0.9463), (92.01, 4547, 0.96393, 0.9602, 0.9434), (96.01, 4734, 1.05410, 0.9591, 0.9416),
    (100.01, 4922, 1.26358, 0.9659, 0.9433), (104.02, 5111, 1.26726, 0.9656, 0.9368),
]
SEED1 = [
    (4.01, 220, 3.23723, 0.6378, 0.6698), (8.01, 430, 2.56587, 0.6856, 0.6993), (12.02, 636, 2.37101, 0.8509, 0.8323),
    (16.01, 834, 2.20146, 0.8584, 0.8497), (20.01, 1017, 2.08880, 0.8962, 0.8718), (24.01, 1194, 1.89681, 0.9168, 0.8927),
    (28.01, 1372, 2.47098, 0.8705, 0.8945), (32.00, 1546, 2.11647, 0.9127, 0.8875), (36.01, 1723, 1.66794, 0.9070, 0.9054),
    (40.01, 1899, 1.60279, 0.9378, 0.9284), (44.00, 2071, 1.58794, 0.9321, 0.9202), (48.01, 2238, 1.65238, 0.9368, 0.9335),
    (52.01, 2403, 1.59507, 0.9422, 0.9357), (56.01, 2567, 1.63184, 0.9348, 0.9289), (60.01, 2731, 1.47033, 0.9341, 0.9408),
    (64.00, 2896, 1.42994, 0.9446, 0.9362), (68.02, 3063, 1.15835, 0.9576, 0.9428), (72.00, 3225, 1.34908, 0.9460, 0.9445),
    (76.00, 3390, 1.28293, 0.9560, 0.9401), (80.00, 3555, 1.27578, 0.9547, 0.9444),
]
REFS = dict(probe_s0_chance=0.07139, probe_hard_chance=0.07130, probe_s0_final0=0.84985, probe_hard_final0=0.77402)
TOTAL_S = 9000.0


def _hist(rows):
    return [dict(minute=m, step=s, probe_loss=l, m1=dict(S0=a, hard=b)) for m, s, l, a, b in rows]


def _verdicts(hist):
    """The verdict at every probe, as the training loop would read it."""
    out = []
    for i in range(1, len(hist) + 1):
        h = hist[i - 1]
        lr = T.lr_at(h['step'], 60.0 * h['minute'], TOTAL_S) / T.PEAK_LR
        out.append(T.kill_verdict(hist[:i], REFS, lr))
    return out


def _extend(rows, n, loss, s0, hard, dt=4.0):
    """``rows`` followed by ``n`` synthetic probes; loss, s0 and hard are functions of k = 1..n."""
    m, s = rows[-1][0], rows[-1][1]
    return list(rows) + [(m + dt * k, s + 190 * k, loss(k), s0(k), hard(k)) for k in range(1, n + 1)]


def _first_fire(hist, start):
    v = _verdicts(hist)
    hits = [(i - start + 1, x) for i, x in enumerate(v) if x is not None]
    return hits[0] if hits else (None, None)


def _v0_rule(losses):
    """v0's diverging rule: the probe loss above 1.10 x its best for 3 probes in a row (first index)."""
    best, above = None, 0
    for i, lv in enumerate(losses):
        best = lv if best is None or lv < best else best
        above = above + 1 if lv > 1.10 * best else 0
        if above >= 3:
            return i
    return None


def test_fixture_reproduces_v0_kill_points():
    # absolute check of the fixture: v0's rule fires on the last probe of each trace, as recorded
    assert _v0_rule([r[2] for r in SEED0]) == len(SEED0) - 1
    assert _v0_rule([r[2] for r in SEED1]) == len(SEED1) - 1


def test_no_rule_fires_on_v0_traces():
    for rows in (SEED0, SEED1):
        assert _verdicts(_hist(rows)) == [None] * len(rows)


def test_m1_held_on_v0_traces():
    # M1's running mean never fell by more than its threshold, and the loss rose past its own only at
    # seed 0's last probe (a single probe, short of the patience)
    for rows in (SEED0, SEED1):
        h = _hist(rows)
        states = [T.divergence_state(h[:i]) for i in range(T.WINDOW, len(h) + 1)]
        assert not any(s['m1_down'] or s['m1_gross'] or s['loss_gross'] for s in states)
        assert max(s['drop'] for s in states) < 0.005
    up = [s['loss_up'] for s in (T.divergence_state(_hist(SEED0)[:i]) for i in range(T.WINDOW, len(SEED0) + 1))]
    assert up[-1] and not any(up[:-1])
    assert not any(T.divergence_state(_hist(SEED1)[:i])['loss_up'] for i in range(T.WINDOW, len(SEED1) + 1))


def test_fires_on_rising_loss_with_falling_m1():
    base = SEED0[:22]                                   # up to minute 88, the best probe loss
    rows = _extend(base, 10, lambda k: 0.945 + 0.15 * k, lambda k: 0.961 - 0.02 * k, lambda k: 0.946 - 0.02 * k)
    k, verdict = _first_fire(_hist(rows), len(base))
    assert verdict == 'killed:probe_diverging:loss_up_m1_down'
    assert 3 <= k <= 6, k                                # within 24 training minutes of the onset


def test_fires_on_slow_divergence_with_falling_m1():
    base = SEED1[:17]                                   # up to minute 68
    rows = _extend(base, 15, lambda k: 1.30 + 0.06 * k, lambda k: 0.955 - 0.008 * k, lambda k: 0.943 - 0.008 * k)
    k, verdict = _first_fire(_hist(rows), len(base))
    assert verdict is not None and verdict.startswith('killed:probe_diverging')
    assert k <= 10, k


def test_fires_on_gross_loss_rise_with_m1_holding():
    base = SEED0[:22]
    rows = _extend(base, 12, lambda k: 0.945 + 0.3 * k, lambda k: 0.96, lambda k: 0.945)
    k, verdict = _first_fire(_hist(rows), len(base))
    assert verdict == 'killed:probe_diverging:loss_gross'
    assert k <= 8, k


def test_fires_on_m1_collapse_with_flat_loss():
    base = SEED0[:22]
    rows = _extend(base, 10, lambda k: 0.95 + 0.05 * (-1) ** k, lambda k: 0.96 - 0.05 * k, lambda k: 0.946 - 0.05 * k)
    k, verdict = _first_fire(_hist(rows), len(base))
    assert verdict == 'killed:probe_diverging:m1_gross'
    assert k <= 6, k


def test_m1_under_half_final0_fires_at_any_probe_after_24():
    base = SEED0[:22]
    rows = _extend(base, 4, lambda k: 0.95, lambda k: 0.30, lambda k: 0.30)
    v = _verdicts(_hist(rows))
    assert v[:22] == [None] * 22
    assert 'killed:s0_m1_below_half_final0' in v or any(x and x.startswith('killed:probe_diverging') for x in v)
    # the reference-scale rule alone (no divergence state: a flat M1 at 0.30 from minute 24 on)
    flat = [r if r[0] < 24 else (r[0], r[1], r[2], 0.30, 0.30) for r in SEED0[:10]]
    vf = _verdicts(_hist(flat))
    assert vf[5] == 'killed:s0_m1_below_half_final0_at_24min'


def test_one_time_checks_read_the_first_probe_past_their_minute():
    rows = [(r[0], r[1], r[2], 0.10 if r[0] >= 12 else r[3], r[4]) for r in SEED0[:4]]
    assert _verdicts(_hist(rows))[2] == 'killed:s0_m1_below_2x_chance_at_12min'
    assert _verdicts(_hist(SEED0[:4])) == [None] * 4


def test_single_spike_does_not_fire():
    base = SEED0[:22]
    spikes = {2: (2.6, 0.80), 6: (2.2, 0.85)}           # two isolated bad probes (loss, M1)
    rows = _extend(base, 10, lambda k: spikes.get(k, (0.95, 0.96))[0], lambda k: spikes.get(k, (0.95, 0.96))[1],
                   lambda k: spikes.get(k, (0.95, 0.945))[1])
    assert _verdicts(_hist(rows)) == [None] * len(rows)


def test_nonfinite_probe_fires_at_once():
    rows = _extend(SEED0[:22], 1, lambda k: math.nan, lambda k: 0.96, lambda k: 0.94)
    assert _verdicts(_hist(rows))[-1] == 'killed:probe_nonfinite'


def test_spread_ignores_a_linear_trend():
    x = 2.0 - 0.05 * np.arange(9)
    assert T.spread(x) < 1e-12
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 0.1, 4000)
    assert abs(T.spread(noise) - 0.1) < 0.01
    assert abs(T.spread(noise + 0.03 * np.arange(4000)) - 0.1) < 0.01
