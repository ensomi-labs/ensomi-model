import pytest

from ensomi_model.osu_core.difficulty import RawHitObject
from ensomi_model.research.gameplay_evaluation.ln_fragmentation import (
    short_ln_exposure, fit_fragmentation_reference, fragmentation_check,
)


def record(group, index, heads, holds, short):
    return dict(group_id=group, source_sha256=f'{group}-{index}', stars=4.,
                heads=heads, LN_heads=holds, short_LNs=short, threshold_ms=80.)


def test_true_long_hold_across_scope_edge_is_not_a_short_crop():
    objects = [RawHitObject(0, 2000, 0), RawHitObject(100, 100, 1),
               RawHitObject(120, 170, 2), RawHitObject(500, 520, 3)]
    value = short_ln_exposure(objects, 0, 200)
    assert value['heads'] == 3 and value['LN_heads'] == 2 and value['short_LNs'] == 1
    assert value['burden'] == pytest.approx(1/3)
    assert value['fraction_of_LNs'] == .5
    assert value['resolved_through_ms'] == 2001


def test_many_difficulties_of_one_song_do_not_dominate_reference():
    sources = [record('one_song', i, 100, 100, 100) for i in range(100)]
    sources += [record(f'song{i}', 0, 100, 10, 0) for i in range(20)]
    reference = fit_fragmentation_reference(sources, quantile=.95)
    assert reference['maximum_burden'] == 0.
    assert reference['reference_exceedance_weight'] == pytest.approx(1/21)


def test_requested_amount_selects_reference_not_realized_generated_amount():
    sources = [record('tap', 0, 100, 10, 1), record('ln', 0, 100, 90, 30)]
    low = fit_fragmentation_reference(sources, requested_LN_fraction=.1)
    high = fit_fragmentation_reference(sources, requested_LN_fraction=.8)
    value = short_ln_exposure([RawHitObject(i*200, i*200+40, i%4) for i in range(10)], 0, 2000)
    assert low['maximum_burden'] == .01 and high['maximum_burden'] == .3
    assert fragmentation_check(value, low)['status'] == 'failed'
    assert fragmentation_check(value, high)['status'] == 'failed'


def test_empty_scope_and_tap_scope_have_different_denominators():
    empty = short_ln_exposure([], 0, 1000)
    taps = short_ln_exposure([RawHitObject(0, 0, 0)], 0, 1000)
    assert empty['burden'] is None and taps['burden'] == 0
    assert empty['fraction_of_LNs'] is None and taps['fraction_of_LNs'] is None


def test_shortness_is_a_prevalence_coordinate_not_an_individual_ban():
    reference = fit_fragmentation_reference([record('a', 0, 100, 50, 10)])
    objects = [RawHitObject(i*200, i*200+(40 if i==0 else 0), i%4) for i in range(20)]
    observed = short_ln_exposure(objects, 0, 4000)
    assert observed['short_LNs'] == 1
    assert fragmentation_check(observed, reference)['status'] == 'passed'
