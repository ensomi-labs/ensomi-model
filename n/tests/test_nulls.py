"""BL-τ chance with no fitted offset: exact values, and the scorer checked against them (guardrail 2)."""
from __future__ import annotations

import pytest

from ensomi_n.eval import nulls


def test_exact_chance_values():
    rows = {r['bpm']: r for r in nulls.table()}
    assert rows[80.0]['bl20'] == pytest.approx(40 / 750, abs=1e-15)
    assert rows[120.0]['bl20'] == pytest.approx(0.08, abs=1e-15)
    assert rows[160.0]['bl20'] == pytest.approx(40 / 375, abs=1e-15)
    assert rows[120.0]['bl10'] == pytest.approx(0.04, abs=1e-15)
    assert rows[120.0]['bl5'] == pytest.approx(0.02, abs=1e-15)


def test_deciding_statistic_stays_under_the_design_bound():
    # E0 rule: a statistic may decide only if its random-phase chance is 0.15 or less on [80, 160).
    assert max(r['bl20'] for r in nulls.table()) <= 40 / 375 < 0.15


@pytest.mark.parametrize('bpm', [80.0, 120.0, 159.0])
def test_scorer_meets_every_null(bpm):
    result = nulls.check(bpm, phases=200, n_beats=32)
    exp = result['expected']
    assert result['fixed']['self'] == {'bl20': 1.0, 'bl10': 1.0, 'bl5': 1.0}
    assert result['fixed']['half_beat'] == {'bl20': 0.0, 'bl10': 0.0, 'bl5': 0.0}
    assert result['fixed']['third_beat'] == {'bl20': 0.0, 'bl10': 0.0, 'bl5': 0.0}
    for k in exp:
        assert abs(result['whole_shift'][k] - exp[k]) <= 1 / 200 + 1e-12
        assert abs(result['beat_phase'][k] - exp[k]) <= 0.01
    assert result['pass']
