import json

import numpy as np
import pytest

from ensomi_model.research.gameplay_evaluation.rhythm_lattice import head_lattice, lattice_coverage
from ensomi_model.research.gameplay_evaluation.temporal import ChartTrace, Scope
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow


def test_main_layer_survives_sparse_half_unit_ornaments():
    main = np.arange(32)*150
    ornaments = main[::4]+75
    observed = lattice_coverage(np.r_[main, ornaments])
    assert observed['heads'] == 40
    assert observed['coarsest_at_coverage']['0.8']['unit_ms'] == pytest.approx(150, abs=.3)
    assert observed['coarsest_at_coverage']['0.95']['unit_ms'] == pytest.approx(75, abs=.2)
    assert observed['maximum_coverage'] == 1


def test_equal_density_does_not_imply_equal_shared_phase():
    regular = 100+np.arange(40)*150
    displaced = regular+np.random.default_rng(901).integers(-30, 31, len(regular))
    a, b = map(lattice_coverage, (regular, displaced))
    assert a['heads'] == b['heads']
    assert a['maximum_coverage'] == 1
    assert b['maximum_coverage'] < .8 and b['coarsest_at_coverage']['0.8'] is None


def test_native_rounding_and_nonbinary_units_need_no_bpm():
    observed = lattice_coverage(np.round(np.arange(40)*1000/3))
    assert observed['coarsest_at_coverage']['0.95']['unit_ms'] == pytest.approx(1000/3, abs=.3)
    assert observed['maximum_coverage'] == 1


def test_translation_and_chord_multiplicity_preserve_unit_coverage():
    times = np.arange(24)*150
    a, b = map(lattice_coverage, (times, np.repeat(times+120000, 4)))
    assert a['heads'] == b['heads'] == 24
    for level, record in a['coarsest_at_coverage'].items():
        other = b['coarsest_at_coverage'][level]
        assert record['unit_ms'] == other['unit_ms']
        assert record['covered_fraction'] == other['covered_fraction']
        assert record['anchor_ms']+120000 == other['anchor_ms']


def test_sparse_and_unsupported_scales_stay_unobserved():
    for times, status in (([], 'insufficient_events'), ([0, 500], 'insufficient_events'),
                          (np.arange(12), 'no_candidates')):
        result = lattice_coverage(times)
        assert result['status'] == status and result['maximum_coverage'] is None
        assert all(v is None for v in result['coarsest_at_coverage'].values())
        json.dumps(result, allow_nan=False)


def test_scope_uses_only_H_inside_its_observed_bounds():
    rows = [CompleteRow(0, (2, 0, 0, 0)), CompleteRow(70, (3, 0, 0, 0))]
    rows += [CompleteRow(100+i*150, (1, 0, 0, 0)) for i in range(20)]
    rows += [CompleteRow(3011, (0, 1, 0, 0))]
    result = head_lattice(ChartTrace(rows, 3100), Scope('local', 100, 3000))
    plain = lattice_coverage([100+i*150 for i in range(20)])
    assert result == dict(start_ms=100, end_ms=3000, **plain)
