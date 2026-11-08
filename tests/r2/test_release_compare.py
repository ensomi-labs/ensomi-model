import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

scripts = Path(__file__).resolve().parents[2] / 'scripts/r2'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('release_compare', scripts / 'release_compare.py')
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def test_paired_effect_excludes_unpaired_nan_and_uses_larger_floor():
    result = compare.paired_effect(pd.Series([3., np.nan, 7.]), pd.Series([1., 8., 2.]),
        dict(sampling_rms=.5, checkpoint_rms=1., current_sampling_rms=2.))
    assert result['songs'] == 2 and result['effect'] == 3.5
    assert result['effect_over_floor'] == 1.75


def test_comparison_order_whole_chart_floor_and_a2_qualification(tmp_path):
    results = tmp_path / 'results'; results.mkdir()
    stage1 = tmp_path / 'stage1'; (stage1 / 'fr').mkdir(parents=True)
    source = dict(sha='chart', source_class='normal', band=2)
    pd.DataFrame([dict(source, checkpoint_m=cp, whole_chart_nll=nll)
                  for cp, nll in [(48, 10.), (56, 11.), (64, 12.)]]).to_csv(results / 'released-nll.csv', index=False)
    old = [dict(source, arm='baseline', checkpoint=cp, seed=seed, mode='prefix', kappa=0,
                **{m: .2 + .01 * (cp - 56) + .02 * (seed - 954) for m in compare.METRICS})
           for cp in [48, 56, 64] for seed in [954, 955, 956]]
    (stage1 / 'fr/chart.jsonl').write_text('\n'.join(json.dumps(r) for r in old))
    counts = dict(head_gaps=3, changed_prefix_gaps=2, held_gaps=2, changed_prefix_held_gaps=1,
                  full_occupancy_gaps=1, completion_pressure_gaps=1, forced_events_two_orders=1)
    for arm, nll in [('A0', 15.), ('A1', 13.), ('A2', 14.)]:
        folder = results / arm; folder.mkdir(); (folder / 'COMPLETE').touch()
        runs = [dict(mode='prefix', kappa=0, seed=seed, **{m: .3 for m in compare.METRICS},
                     leaks=counts, forced_full_occupancy=0 if arm == 'A0' else 1,
                     forced_eos=0, duration_hazards=[], eos_releases=0, legal=True, fallbacks=0,
                     eos_release_offset_from_last_head_ms=[]) for seed in [954, 955, 956]]
        (folder / 'chart.json').write_text(json.dumps(dict(source, whole_chart_nll=nll, runs=runs)))
    data = compare.analyze(results, stage1)
    nll = data['nll']; all_rows = nll[(nll.band == 'all') & (nll.source_class == 'all')]
    assert all_rows.comparison.tolist() == ['A0 - released56', 'A1 - A0', 'A1 - A2']
    np.testing.assert_allclose(all_rows.effect, [4., -2., -1.])
    np.testing.assert_allclose(all_rows.checkpoint_rms, np.sqrt(2.))
    a2 = data['leak_summary']; a2 = a2[a2.arm == 'A2']
    assert a2.interpretation.str.contains('tie is inconclusive').all()
    assert (a2.actual_full_force_per_gap == 1 / 3).all()
    assert 'tie is inconclusive' in compare.report(data, 'test')
