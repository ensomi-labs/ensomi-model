import pandas as pd

from ensomi_model.r2.bakeoff_panel import run_specs, select_panel


def table():
    rows = []
    for band in (2, 3, 4, 5):
        for group in range(6):
            for chart in range(2):
                key = f'{band}-{group}-{chart}'
                rows.append(dict(sha256=key, group_id=f'{band}-{group}', band=band,
                                 K=1100 if band == 2 else 1600, file=key + '.npz', role='fit_dev'))
    rows.append(dict(sha256='train', group_id='train', band=2, K=9000, file='train.npz', role='fit_train'))
    return pd.DataFrame(rows)


def test_group_selection_is_deterministic_and_excludes_entire_group():
    source = table()
    excluded = [dict(sha256='2-0-0', group_id='2-0')]
    panel = select_panel(source, excluded, groups_per_band=4)
    assert panel == select_panel(source.sample(frac=1, random_state=73), excluded, groups_per_band=4)
    assert len(panel['charts']) == 16
    assert len({r['group_id'] for r in panel['charts']}) == 16
    assert not any(r['group_id'] == '2-0' for r in panel['charts'])
    assert all(r['K'] >= (1000 if r['band'] == 2 else 1500) for r in panel['charts'])


def test_panel_modes_caps_and_diagnostics():
    panel = select_panel(table(), [], groups_per_band=1)
    runs = run_specs(panel, 'b0', max_charts=2, max_rows=200)
    assert len(runs) == 8
    assert {r['seed'] for r in runs if r['mode'] == 'bos'} == {954, 955, 956}
    assert all(r['stop'] - r['start'] == 200 for r in runs)
    assert all(r['start'] == r['K'] // 3 for r in runs if r['mode'] == 'prefix')
    for system in ('b3-unknown', 'b3-oracle'):
        assert len(run_specs(panel, system)) == 8
        assert {r['seed'] for r in run_specs(panel, system)} == {954}
    d0 = run_specs(panel, 'd0-phi')
    assert len(d0) == 4
    assert {(r['mode'], r['seed']) for r in d0} == {('bos', 954)}
    assert all(r['stop'] == r['K'] + 1 for r in run_specs(panel, 'b0') if r['mode'] == 'prefix')


def test_group_matching_does_not_reuse_cross_band_groups():
    rows = []
    for band, groups in {2: ('a', 'b'), 3: ('a', 'b'), 4: ('a', 'c'), 5: ('c', 'd')}.items():
        for group in groups:
            rows.append(dict(sha256=f'{band}-{group}', group_id=group, band=band, K=1600,
                             file=f'{band}-{group}.npz', role='fit_dev'))
    panel = select_panel(pd.DataFrame(rows), [], groups_per_band=1)
    assert {r['group_id'] for r in panel['charts']} == {'a', 'b', 'c', 'd'}


def test_explicit_band3_threshold_and_quota_are_recorded():
    source = table()
    source.loc[source.band == 3, 'K'] = 1498
    quotas = {2: 2, 3: 1, 4: 2, 5: 2}
    panel = select_panel(source, [], groups_per_band=quotas,
                         minimum_rows={2: 1000, 3: 1498, 4: 1500, 5: 1500})
    assert panel['minimum_rows']['3'] == 1498
    assert panel['groups_per_band'] == quotas
    assert len([r for r in panel['charts'] if r['band'] == 3]) == 1
