import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

scripts = Path(__file__).resolve().parents[2] / 'scripts/r2'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('release_seed_compare', scripts / 'release_seed_compare.py')
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def fixture():
    nll = {'A0': [10., 20.], 'A1': [12., 24.], 'A2': [20., 30.],
           'A0-s2': [13., 16.], 'A1-s2': [15., 21.]}
    style = {'A0': [.4, .6], 'A1': [.45, .65], 'A2': [.5, .7],
             'A0-s2': [.7, .2], 'A1-s2': [.75, .35]}
    arms = {a: [] for a in compare.ARMS}
    original, old = [], []
    for i in range(2):
        meta = dict(sha=str(i), source_class='normal', band=2, source_head_rows=100)
        for cp in (48, 56, 64):
            original.append(dict(meta, checkpoint_m=cp, whole_chart_nll=10. + i * 10 + (cp - 56) / 8))
            for mode in ('bos', 'prefix'):
                for kappa in (0, 3):
                    for seed in (954, 955, 956):
                        old.append(dict(meta, mode=mode, kappa=kappa, seed=seed, arm='baseline', checkpoint=cp,
                                        **{m: .3 + .001 * (cp - 56) + .01 * (seed - 955) for m in compare.METRICS}))
        for arm in arms:
            runs = [dict(mode=mode, kappa=kappa, seed=seed, legal=True,
                         **{m: style[arm][i] + .01 * (seed - 955) for m in compare.METRICS})
                    for mode in ('bos', 'prefix') for kappa in (0, 3) for seed in (954, 955, 956)]
            arms[arm].append(dict(meta, whole_chart_nll=nll[arm][i], runs=runs))
    return arms, pd.DataFrame(original), pd.DataFrame(old)


def test_training_floor_is_rms_song_difference_and_mean_keeps_full_floor():
    data = compare.analyze_rows(*fixture())
    floors = data['training_seed_floors']
    nll_floor = floors[(floors.band == 'all') & (floors.source_class == 'all') &
                       (floors.metric == 'whole_chart_nll')].set_index('arm')
    assert nll_floor.loc['A0', 'training_seed_rms'] == pytest.approx(np.sqrt(12.5))
    assert nll_floor.loc['A0', 'effect'] == -.5
    assert nll_floor.loc['A1', 'training_seed_rms'] == 3.
    nll = data['nll']; rows = nll[(nll.band == 'all') & (nll.source_class == 'all')]
    np.testing.assert_allclose(rows.effect, [3., 3.5, 3.25, -7.])
    np.testing.assert_allclose(rows.effect_over_floor, np.array([3., 3.5, 3.25, 7.]) / np.sqrt(12.5))
    assert rows.songs.tolist() == [2, 2, 2, 2]
    fr_floor = floors[(floors.band == 'all') & (floors.source_class == 'all') &
                     (floors.metric == 'ln_share') & (floors['mode'] == 'prefix') & (floors.kappa == 0)]
    assert fr_floor.set_index('arm').loc['A0', 'training_seed_rms'] == pytest.approx(np.sqrt(.125))
    fr = data['fr']
    primary = fr[(fr.band == 'all') & (fr.source_class == 'all') & (fr.metric == 'ln_share') &
                 (fr['mode'] == 'prefix') & (fr.kappa == 0)]
    np.testing.assert_allclose(primary.effect, [.05, .1, .075, -.05])
    np.testing.assert_allclose(primary.effect_over_floor, np.array([.05, .1, .075, .05]) / np.sqrt(.125))
    assert 'A2 has no second training seed' in compare.report(data, 'test', 'out')


def test_missing_sampling_seed_or_changed_panel_stops_comparison():
    arms, original, old = fixture()
    arms['A1-s2'][0]['runs'].pop()
    with pytest.raises(ValueError, match='sampling protocol differs'):
        compare.analyze_rows(arms, original, old)
    arms, original, old = fixture()
    arms['A0-s2'][0]['source_head_rows'] += 1
    with pytest.raises(ValueError, match='panel or metadata differs'):
        compare.analyze_rows(arms, original, old)
