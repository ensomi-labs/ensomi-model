from itertools import product
import random

import pytest

from ensomi_model.osu_core.difficulty import RawHitObject, compute_mania_star_rating_20241007
from ensomi_model.research.gameplay_evaluation.head_difficulty import head_timing_star_lower_bound_20241007


def stars(objects, rate=1.):
    return compute_mania_star_rating_20241007(objects, 4, rate)


def test_first_object_and_first_chord_have_different_information_in_H():
    floor = head_timing_star_lower_bound_20241007
    assert floor(()) == floor((0,)) == 0.
    chord = [RawHitObject(0., 0., k) for k in range(4)]
    assert stars(chord) > floor((0,))
    # With only two single TAPs the selected strain has no earlier processed
    # object in its lane. The floor is exact, including the initial overall 1.
    pair = [RawHitObject(0., 0., 0), RawHitObject(400., 400., 1)]
    assert floor((0, 400)) == pytest.approx(.018 * (3 + .3**.4))
    assert floor((0, 400)) == pytest.approx(stars(pair))


@pytest.mark.parametrize('rate', [1., 1.5])
def test_bound_survives_all_layouts_and_hold_choices_at_fixed_H(rate):
    times = (0, 1, 400, 801)
    floor = head_timing_star_lower_bound_20241007(times, clock_rate=rate)
    for columns in product(range(4), repeat=len(times)):
        for long_flags in ((False,)*4, (True,)*4, (True, False, True, False)):
            objects = []
            for i, (t, k, is_long) in enumerate(zip(times, columns, long_flags)):
                following = next((u for u, c in zip(times[i+1:], columns[i+1:]) if c == k), t+221)
                end = min(t+220, following-1) if is_long else t
                objects.append(RawHitObject(t, max(t, end), k))
            assert floor <= stars(objects, rate)+1e-10
            # Adding other columns to the first row changes the first
            # difficulty-object boundary, not the mandatory H sequence.
            first = [RawHitObject(0., 0., k) for k in range(4) if k != columns[0]]
            assert floor <= stars(first+objects, rate)+1e-10


def test_random_legal_charts_include_chords_releases_and_long_empty_advances():
    rng = random.Random(280034)
    for _ in range(128):
        objects = []
        starts = [None]*4
        t = rng.choice((0, 1, 399, 400, 401, 3000))
        for _ in range(rng.randint(1, 160)):
            for k in range(4):
                if starts[k] is not None:
                    if rng.random() < .35:
                        objects.append(RawHitObject(starts[k], t, k))
                        starts[k] = None
                elif rng.random() < .45:
                    if rng.random() < .4:
                        starts[k] = t
                    else:
                        objects.append(RawHitObject(t, t, k))
            t += rng.choice((1, 2, 20, 40, 99, 100, 399, 400, 401, 1200))
        objects.extend(RawHitObject(start, t, k) for k, start in enumerate(starts) if start is not None)
        times = sorted({o.start_time for o in objects})
        floor = head_timing_star_lower_bound_20241007(times)
        assert floor <= stars(objects)+1e-10


def test_prefix_floor_does_not_need_future_heads_or_open_LN_tails():
    head_times = (0, 90, 180, 400, 490, 1200, 2401)
    values = [head_timing_star_lower_bound_20241007(head_times[:i]) for i in range(len(head_times)+1)]
    assert all(b >= a for a, b in zip(values, values[1:]))
    # Same observed H, different legal future tails and appended heads.
    for tail in (500., 3000.):
        objects = [RawHitObject(0., tail, 0),
                   *[RawHitObject(t, t, 1+i%3) for i, t in enumerate(head_times[1:])],
                   RawHitObject(4000., 4000., 0)]
        assert values[-1] <= stars(objects)+1e-10


def test_dense_mandatory_timing_can_exceed_a_request_before_layout_is_chosen():
    times = tuple(range(0, 4000, 31))
    assert head_timing_star_lower_bound_20241007(times) > 2.
    cyclic = [RawHitObject(t, t, i%4) for i, t in enumerate(times)]
    assert head_timing_star_lower_bound_20241007(times) <= stars(cyclic)


@pytest.mark.parametrize('times', [(2, 1), (1, 1), (-1,), (.5,), (float('nan'),)])
def test_rejects_times_outside_the_declared_native_H_measure(times):
    with pytest.raises(ValueError):
        head_timing_star_lower_bound_20241007(times)
