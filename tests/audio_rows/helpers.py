"""A three-chart build in the layout ``audio_rows.data.build`` writes, without audio or BeatThis.

One audio file; one fit_train chart and two fit_dev charts with different head densities."""
import json

import numpy as np

from ensomi_model.audio_rows.audio import FEATURE_DIM
from ensomi_model.audio_rows.data import GRIDS, chart_targets
from ensomi_model.audio_rows.grid import corrected_grid
from ensomi_model.r2.common import GridArrays, grid_from_arrays

BANDS = {2: 1.8, 3: 2.3, 4: 2.7, 5: 3.1}


def tiny_build(root, song_ms=40_000.0):
    rng = np.random.default_rng(0)
    for d in ('audio', *GRIDS):
        (root / d).mkdir(parents=True)
    np.save(root / 'audio' / 'a.npy', rng.normal(size=(int(song_ms / 10), FEATURE_DIM)).astype(np.float16))
    raw = [[100.0, 500.0]]
    grids = dict(chart=GridArrays.from_grid(grid_from_arrays(np.array([[70.0, 500.0, 4]]), np.array([[70.0, 4]]))),
                 fitted=corrected_grid(raw, song_ms)[0])
    rows = []
    for i, role in enumerate(('fit_train', 'fit_dev', 'fit_dev')):
        quarters = grids['chart'].time_of_beat(np.arange(4, 4 * 75) / 4)
        heads = np.sort(rng.choice(quarters, size=150 + 50 * i, replace=False))
        for name, grid in grids.items():
            np.savez(root / name / f'c{i}.npz', **chart_targets(heads, song_ms, grid))
        rows.append(dict(sha=f'c{i}', file=f'c{i}.npz', group=f'g{i}', role=role, band=3, song_ms=song_ms,
                         audio='', key='a'))
    index = dict(rows=rows, audio={'a': dict(segments=raw, song_ms=song_ms)}, failed={}, label_mean=2.0,
                 label_std=1.0, bands=BANDS)
    (root / 'index.json').write_text(json.dumps(index))
    return root
