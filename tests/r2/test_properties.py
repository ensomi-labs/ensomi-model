"""The ``properties`` module and nu (plan v4 section 5.1): T-U, T-M, the row/object LN-share agreement,
committed counters equal LNShare_nu, open holds, scopes closed at T, and labels equal readouts on the cache."""
from pathlib import Path

import numpy as np
import pytest

from ensomi_model.r2.features import Interval, _interval_counts
from ensomi_model.r2.properties import (NU, PObject, as_pobjects, difficulty, ln_share, ln_share_rows, prefix_objects,
                                        scope_contains)

from .helpers import random_decisions, random_skeleton, two_segment_grid

CACHE = Path('artifacts/r2-cache/v1')


def charts(n, seed=0, K=(40, 200)):
    rng = np.random.default_rng(seed)
    grid = two_segment_grid()
    for _ in range(n):
        head, song = random_skeleton(rng, int(rng.integers(*K)), lo=40, hi=300)
        yield random_decisions(rng, (head, song, grid)), rng


def row_counts(chart):
    d = chart.derived()
    return d.attack.sum(1)[:chart.K], d.ln_head.sum(1)[:chart.K]


def test_t_u_undefined_readouts_are_not_zero():
    objs = [PObject(1000.0, 1000.0, 0, False), PObject(1200.0, 1500.0, 1, True)]
    assert ln_share(objs, 2000.0, 3000.0, 10_000.0) is None
    assert ln_share(objs, 0.0, 3000.0, 10_000.0) == 0.5
    v, info = difficulty(objs, 40_000.0, 80_000.0, 100_000.0)
    assert v is None and info['invalid'] == 'empty_scope'
    v, info = difficulty(objs, 0.0, 20_000.0, 100_000.0)
    assert v is None and info['invalid'] == 'scope_under_30s'
    v, _ = difficulty(objs, 0.0, 30_000.0, 100_000.0)
    assert v is not None and v > 0


def test_open_holds_count_for_ln_share_and_make_difficulty_undefined():
    objs = [PObject(1000.0, 1000.0, 0, False), PObject(1200.0, None, 1, True)]
    assert ln_share(objs, 0.0, 5000.0, 100_000.0) == 0.5
    assert difficulty(objs, 0.0, 30_000.0, 100_000.0)[1]['invalid'] == 'open_hold'


def test_scope_with_b_equal_t_contains_t():
    assert bool(scope_contains(0.0, 100.0, 100.0, 100.0)) and not bool(scope_contains(0.0, 100.0, 200.0, 100.0))


def test_row_and_object_forms_agree_and_counters_equal_nu():
    n_scopes = 0
    for chart, rng in charts(200):
        T = chart.song_ms
        objects = prefix_objects(chart.head_ms, chart.actions, chart.gap)
        heads, lns = row_counts(chart)
        for _ in range(10):
            a = float(rng.uniform(0, T * 0.9))
            b = T if rng.random() < 0.2 else float(rng.uniform(a + 1, T))
            obj = ln_share(objects, a, b, T)
            row = ln_share_rows(heads, lns, chart.head_ms, a, b, T)
            assert (obj is None and row is None) or abs(obj - row) < 1e-12
            k = int(rng.integers(0, chart.K + 1))
            h, l, _ = _interval_counts(chart, Interval(0, a, b, 0.5), k)
            committed = prefix_objects(chart.head_ms, chart.actions[:k], chart.gap[:k])
            want = ln_share(committed, a, b, T)
            assert (want is None and h == 0) or (h and abs(l / h - want) < 1e-12)
            n_scopes += 1
    assert n_scopes == 2000


def test_t_m_mirror():
    """LN share is mirror-invariant; the tiled star is measured: equality within 1e-9 on 200 charts."""
    worst = 0.0
    for chart, rng in charts(200, seed=4, K=(150, 400)):
        T = chart.song_ms
        objects = prefix_objects(chart.head_ms, chart.actions, chart.gap)
        mirrored = [PObject(o.start, o.end, 3 - o.lane, o.hold) for o in objects]
        a, b = 0.0, max(30_000.0, T)
        assert ln_share(objects, a, T, T) == ln_share(mirrored, a, T, T)
        x, _ = difficulty(objects, a, b, max(b, T))
        y, _ = difficulty(mirrored, a, b, max(b, T))
        if x is not None:
            worst = max(worst, abs(x - y))
    print('T-M tiled-star mirror difference, worst', worst)
    assert worst <= 1e-9, f'tiled star is not mirror-invariant: worst |difference| {worst}; declare it in NU'


def test_nu_declares_every_field():
    for kind in ('ln_share', 'difficulty'):
        for f in ('undefined', 'deviation', 'resolution', 'mirror'):
            assert NU[kind][f]


def test_as_pobjects_accepts_every_object_type():
    from ensomi_model.osu_core.difficulty import RawHitObject
    from .helpers import hold, tap
    a = as_pobjects([tap(10, 0), hold(20, 50, 1)])
    b = as_pobjects([RawHitObject(10.0, 10.0, 0), RawHitObject(20.0, 50.0, 1)])
    assert a == b == [PObject(10.0, 10.0, 0, False), PObject(20.0, 50.0, 1, True)]


@pytest.mark.skipif(not (CACHE / 'labels' / 'star-v2.json.gz').exists(), reason='v2 labels not built here')
def test_labels_equal_readouts_on_the_cache():
    """For 200 fit_train charts the stored v2 label equals Difficulty_nu of the cache representation, and
    the v1 label (from the .osu) agrees within 0.001 star except where the representations differ."""
    import gzip
    import json
    from ensomi_model.r2.cache import CacheIndex
    from ensomi_model.r2.labels import chart_objects, load_star_labels_v1
    data = json.loads(gzip.decompress((CACHE / 'labels' / 'star-v2.json.gz').read_bytes()))
    index = CacheIndex.open(CACHE).table
    train = index[index.role == 'fit_train'].sort_values('sha256').head(200)
    v1, _ = load_star_labels_v1(CACHE)
    differ = compared = 0
    for r in train.itertuples():
        with np.load(CACHE / 'charts' / r.file) as z:
            objects = chart_objects(z)
            T = float(z['song_ms'])
        for l in data['charts'][r.sha256]:
            value, _ = difficulty(objects, l['a'], l['b'], T)
            assert value == l['value'], (r.sha256, l)
        old = {(a, b): v for a, b, v in v1.get(r.sha256, [])}
        new = {(l['a'], l['b']): l['value'] for l in data['charts'][r.sha256]}
        for key, v in old.items():
            if key in new and new[key] is not None:
                compared += 1
                differ += abs(new[key] - v) > 0.001
    print('labels v1 vs v2 compared', compared, 'differ by more than 0.001 star', differ)
    assert compared > 0


@pytest.mark.skipif(not (CACHE / 'labels' / 'star-v2.json.gz').exists(), reason='v2 labels not built here')
def test_draw_ln_values_equal_nu_on_the_cache():
    from ensomi_model.r2.conditions import DrawConfig, ln_candidates
    from ensomi_model.r2.data import Corpus
    corpus = Corpus(CACHE, 'fit_train', star_conditions=False)
    rng = np.random.default_rng(1)
    for sha in sorted(corpus.files)[:50]:
        chart = corpus.chart(sha)
        objects = prefix_objects(chart.head_ms, chart.actions, chart.gap)
        for c in ln_candidates(chart, rng, DrawConfig()):
            assert abs(c.iv.value - ln_share(objects, c.iv.a, c.iv.b, chart.song_ms)) < 1e-9
