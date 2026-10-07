import numpy as np
import pandas as pd

from ensomi_model.r2 import block_select
from ensomi_model.r2.block_select import candidate_seeds, fit_phi, load_scorer, phi, select_blocks
from ensomi_model.r2.features import Chart
from ensomi_model.r2.sampling import continue_chart

from .helpers import random_decisions, random_skeleton, tiny_model, two_segment_grid


def source_model():
    rng = np.random.default_rng(281)
    head, song = random_skeleton(rng, 129)
    return (random_decisions(rng, (head, song, two_segment_grid())),
            tiny_model(hidden=16, levels=2, expansion=1, rank=4).eval())


def test_reused_history_cache_matches_prefix_replay():
    source, model = source_model()
    cache = {}
    a, g = continue_chart(model, source.head_ms, source.song_ms, source.grid, seed=19, stop=64,
                          _history_out=cache)
    kwargs = dict(prefix_actions=a, prefix_gap=g, seed=20, stop=128)
    expected = continue_chart(model, source.head_ms, source.song_ms, source.grid, **kwargs)
    actual = continue_chart(model, source.head_ms, source.song_ms, source.grid, **kwargs,
                             _history_cache=(cache['cache'], cache['marks']))
    for left, right in zip(actual, expected):
        assert left.tobytes() == right.tobytes()


def test_block_selection_commits_highest_score_and_is_deterministic():
    source, model = source_model()
    score = lambda chart, start, stop, band: float(phi(chart, start, stop)[0])
    a, g, record = select_blocks(model, source, score, band=3, stop=128)
    again = select_blocks(model, source, score, band=3, stop=128)
    assert a.tobytes() == again[0].tobytes()
    assert g.tobytes() == again[1].tobytes()
    assert record == again[2]
    assert len(record['blocks']) == 2
    for block in record['blocks']:
        assert block['selected'] == int(np.argmax(block['scores']))
        chart = Chart(source.head_ms, source.song_ms, source.grid, a, g)
        assert score(chart, block['start'], block['stop'], 3) == block['scores'][block['selected']]
    assert candidate_seeds(954, 0) != candidate_seeds(954, 1)
    assert len(set(candidate_seeds(954, 0))) == 4


def test_phi_uses_held_state_from_before_the_block():
    source, _ = source_model()
    d = source.derived()
    actual = phi(source, 64, 128)
    assert actual[0] == d.attack[64:128].sum() / 64
    assert actual[1] == d.held[64:128].mean()
    assert actual[2] == (d.attack[64:128].sum(1) >= 3).mean()


def test_phi_fit_uses_only_fit_train_and_reloads(tmp_path, monkeypatch):
    source, _ = source_model()
    table = pd.DataFrame([dict(sha256='a', role='fit_train', file='train.npz'),
                          dict(sha256='b', role='fit_dev', file='dev.npz')])
    loaded = []
    monkeypatch.setattr(block_select.pd, 'read_parquet', lambda path: table)
    def load(path):
        loaded.append(path.name)
        return source
    monkeypatch.setattr(block_select, 'load_chart', load)
    monkeypatch.setattr(block_select, 'chart_from_cache', lambda chart: chart)
    path = tmp_path / 'phi.json'
    metadata = fit_phi(tmp_path, path)
    assert loaded == ['train.npz']
    assert metadata['charts'] == 1 and metadata['blocks'] == 2
    assert metadata['ridge'] == 1.0
    scorer = load_scorer('phi', path)
    assert np.isfinite(scorer(source, 64, 128, 3))
