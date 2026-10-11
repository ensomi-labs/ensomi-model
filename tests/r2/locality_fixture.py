"""The plan v4 section 4.6 locality fixture: one chart with every boundary case of rule L.

Head rows every 100 ms from 1,000 ms (K = 80, decision k at 1000 + 100 k ms), lane 0 taps on
every row, T = 9,150 ms. Holds:

- lane 1, 1900-2050: open entering decision 10 (2000) -> (i) the onset of LN_I1 = [2000, 3000)
  is masked; closed inside I1 by decision 11 -> (v) a hold born before a and closed inside S.
- lane 3, 2200-2450: born and closed inside I1.
- lane 1, 3700-3960 and lane 2, 3800-4250: born inside LN_I2 = [3050, 3950), which starts inside
  gap (3000, 3100) with no hold open -> (ii). The exit decision 30 (4000) closes lane 1 at 3960
  with candidates on both sides of b = 3950 -> (iii); lane 2 is closed by decision 33, three
  decisions after the exit -> (iv).
- lane 3, 8900-9100: open at EOS -> (vi) with the whole-song scope [0, T].
- (vii) adjacent LN scopes [5000, 6000) and [6000, 7000); (viii) LN 0 on [5000, 6000);
  (ix) a difficulty interval [2300, 4500) overlapping I1 and I2 (its onset, decision 13, is masked).
"""
from __future__ import annotations

import numpy as np

from ensomi_model.r2.features import Interval

from .helpers import chart_from_objects, hold, tap

T = 9150.0
HEADS = [1000.0 + 100.0 * k for k in range(80)]
HOLDS = [(1900.0, 2050.0, 1), (2200.0, 2450.0, 3), (3700.0, 3960.0, 1), (3800.0, 4250.0, 2),
         (5100.0, 5350.0, 1), (6300.0, 6550.0, 2), (8900.0, 9100.0, 3)]

I1 = Interval(0, 2000.0, 3000.0, 0.7)
I2 = Interval(0, 3050.0, 3950.0, 0.6)
WHOLE = Interval(0, 0.0, T, 0.4)
ADJ_A = Interval(0, 5000.0, 6000.0, 0.2)
ADJ_B = Interval(0, 6000.0, 7000.0, 0.8)
ZERO = Interval(0, 5000.0, 6000.0, 0.0)
STAR = Interval(1, 2300.0, 4500.0, 3.5)

# V(I): the decisions that read I (rule L), listed by hand from the docstring.
EXPECTED_V = {
    'I1': list(range(11, 20)),           # decision 10 (2000) is the masked onset
    'I2': list(range(21, 30)),           # decision 30 (4000) is the exit decision
    'WHOLE': list(range(0, 81)),         # EOS (80) reads it: lane 3 is held and every candidate lies in [0, T]
    'ADJ_A': list(range(40, 50)), 'ADJ_B': list(range(50, 60)),
}


def fixture_chart():
    objects = [hold(a, b, lane) for a, b, lane in HOLDS]
    for t in HEADS:
        objects.append(tap(t, 0))
    chart, dec = chart_from_objects(objects, T)
    assert np.array_equal(chart.head_ms, np.array(HEADS)) and chart.K == 80
    return chart


def with_value(iv: Interval, value: float) -> Interval:
    return Interval(iv.kind, iv.a, iv.b, value)
