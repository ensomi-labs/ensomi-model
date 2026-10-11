import numpy as np

from ensomi_model.r2.strain import StrainTrace
from ensomi_model.r2.theta import decode
from ensomi_model.system.arrange import arrange
from ensomi_model.system.prior import (METHODS, TARGETS, Prior, arrangement_targets, choose, fit_band, kappa_targets,
                                       rows_features)

from .helpers import grid, skeleton, tiny_r2


def test_rows_features_of_an_even_skeleton():
    head = 1000.0 + 250.0 * np.arange(200)
    song = float(head[-1] + 1000)
    x = rows_features(head, song, grid())
    assert x.shape == (4,)
    assert np.isclose(x[0], np.log(200 / (song / 1000)))
    assert np.isclose(x[1], np.log(250.0))
    assert np.isclose(x[2], np.log(StrainTrace(head, song).reference_prefix[-1] / (song / 1000)))
    assert np.isclose(x[3], np.log(120.0))


def test_arrangement_targets_read_the_arrangement():
    rng = np.random.default_rng(2)
    head, song = skeleton(rng, 80)
    actions, gap, _ = arrange(tiny_r2(), head, song, grid(), seed=1, target=[0, 1, 0], kappa=0)
    y = arrangement_targets(head, song, actions, gap)
    held, heads, _ = decode(actions[:len(head)])
    assert y.shape == (len(TARGETS),)
    assert np.isclose(y[0], held.mean()) and np.isclose(y[1], heads.sum(1).mean())
    assert np.isclose(y[2] + y[3], (held | heads).all(1).mean())
    assert np.isclose(np.exp(y[4]), StrainTrace(head, song).extend(actions, gap).scope(0, song).ratio)


def test_kappa_targets_follow_the_demo_source_targets():
    pred = dict(held=0.2, heads=1.5, busy_free=0.04, quad=0.01, log_r=0.3)
    assert np.allclose(kappa_targets(pred), [0.2, 1.5, 0.05])
    assert np.allclose(kappa_targets(pred, 0.5, 1.2), [0.1, 1.8, 0.04 * 0.5 + 0.01])
    assert np.allclose(kappa_targets(pred, 0.0, 0.9), [0.0, 1.35, 0.009])


def synthetic(n, rng):
    X = rng.normal(size=(n, 4))
    Y = np.column_stack((0.1 + 0.05 * X[:, 0], 1.4 + 0.2 * X[:, 0], 0.02 + 0 * X[:, 0], 0.01 + 0 * X[:, 0],
                         0.2 + 0.1 * X[:, 1])) + rng.normal(0, 0.01, size=(n, 5))
    return X, Y


def test_predictors_and_their_clipping():
    rng = np.random.default_rng(0)
    X, Y = synthetic(400, rng)
    rec = dict(bands={str(b): fit_band(X, Y) for b in range(2, 6)})
    x = np.array([1.0, 0.0, 0.0, 0.0])
    for m in METHODS:
        p = Prior(dict(rec, method=m)).predict(x, 3)
        assert set(p) == set(TARGETS)
        assert 1 <= p['heads'] <= 4 and 0 <= p['held'] <= 1
    lin = Prior(dict(rec, method='linear')).predict(x, 3)
    assert abs(lin['heads'] - 1.6) < 0.01 and abs(lin['held'] - 0.15) < 0.01
    far = Prior(dict(rec, method='linear')).predict(np.array([-100.0, 0, 0, 0]), 3)
    assert far['held'] == 0.0 and far['heads'] == 1.0


def test_choose_keeps_the_plainest_that_holds_up():
    rng = np.random.default_rng(1)
    n = 300
    groups = np.arange(n) // 3
    same = np.abs(rng.normal(size=(n, 5)))
    errors = {'median': same + 0.5, 'linear': same, 'knn': same + 1e-4 * rng.normal(size=(n, 5))}
    method, table = choose(errors, groups)
    assert method == 'linear' and not table['median']['holds_up'] and table['knn']['holds_up']
    errors['median'] = same.copy()
    assert choose(errors, groups)[0] == 'median'
    worse = np.zeros_like(same)
    worse[:, 0] = 1.0                          # each method loses clearly on one target: none holds up
    errors = {'median': same + 3 * worse, 'linear': same + np.roll(worse, 1, 1), 'knn': same + 0.2 * np.roll(worse, 2, 1)}
    method, table = choose(errors, groups)
    assert method == 'knn' and not any(v['holds_up'] for v in table.values())


def test_fit_on_a_small_cache(tmp_path):
    import json
    import pandas as pd
    rng = np.random.default_rng(3)
    (tmp_path / 'cache' / 'charts').mkdir(parents=True)
    model = tiny_r2()
    rows = []
    for i in range(24):
        head, song = skeleton(rng, 40 + 5 * i)
        actions, gap, _ = arrange(model, head, song, grid(), seed=i, target=[0, 1, 0], kappa=0)
        name = f'c{i:02d}.npz'
        np.savez(tmp_path / 'cache' / 'charts' / name, head_ms=head, actions=actions.astype(np.int8),
                 gap_release_ms=gap, release_orig_ms=gap, grid_segments=np.array([[0.0, 500.0, 4]]),
                 grid_bars=np.array([[0.0, 4]]), song_ms=song)
        rows.append(dict(sha256=f'{i:064x}', file=name, role='fit_dev' if i % 3 == 0 else 'fit_train',
                         band=2 + (i // 2) % 4, group_id=f'g{i // 2}'))
    pd.DataFrame(rows).to_parquet(tmp_path / 'cache' / 'index.parquet')
    from ensomi_model.system.prior import fit
    record = fit(tmp_path / 'cache', tmp_path / 'out', workers=1)
    saved = json.loads((tmp_path / 'out' / 'prior.json').read_text())
    assert saved['method'] in METHODS and set(saved['bands']) == {'2', '3', '4', '5'}
    assert record['charts']['fit_train'] + record['charts']['fit_dev'] == 24 and not record['failed']
    p = Prior.load(tmp_path / 'out' / 'prior.json').predict(rows_features(head, song, grid()), 4)
    assert set(p) == set(TARGETS)
