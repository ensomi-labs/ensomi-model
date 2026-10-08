from dataclasses import fields
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from ensomi_model.r2.features import Chart, history_tokens
from ensomi_model.r2.sampling import IncrementalState, _history_token, continue_chart

from .helpers import random_decisions, random_skeleton, tiny_model, two_segment_grid
from .sampling_reference import continue_chart as reference_continue_chart

CHECKPOINT = Path('artifacts/r2-runs/r2-phaseN-20261006/checkpoints/ckpt-0056000574.pt')


@pytest.mark.parametrize('prefix', [0, 17])
def test_incremental_arrays_and_tokens_match_replay(prefix):
    rng = np.random.default_rng(871)
    head, song = random_skeleton(rng, 91)
    source = random_decisions(rng, (head, song, two_segment_grid()))
    actions = np.zeros_like(source.actions)
    gap = np.full_like(source.gap, np.nan)
    actions[:prefix], gap[:prefix] = source.actions[:prefix], source.gap[:prefix]
    state = IncrementalState(head, song, source.grid, actions, gap, prefix)
    for k in range(prefix, source.n):
        state.append(source.actions[k], source.gap[k])
        expected = Chart(head, song, source.grid, actions[:k + 1], gap[:k + 1]).derived()
        actual = state.chart.derived()
        for field in fields(expected):
            np.testing.assert_array_equal(getattr(actual, field.name), getattr(expected, field.name))
        if k < source.K:
            np.testing.assert_array_equal(_history_token(state.chart, k), history_tokens(source, k + 1)[k])


@pytest.mark.parametrize('prefix', [0, 17])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_sampling_matches_reference(prefix, dtype):
    rng = np.random.default_rng(341)
    head, song = random_skeleton(rng, 40)
    source = random_decisions(rng, (head, song, two_segment_grid()))
    model = tiny_model(dtype, hidden=16, levels=2, expansion=1, rank=4).eval()
    kwargs = dict(prefix_actions=source.actions[:prefix], prefix_gap=source.gap[:prefix], seed=954)
    expected = reference_continue_chart(model, head, song, source.grid, **kwargs)
    actual = continue_chart(model, head, song, source.grid, **kwargs)
    for a, b in zip(actual, expected):
        assert a.tobytes() == b.tobytes()


@pytest.mark.skipif(os.environ.get('R2_TEST_REAL_SAMPLER') != '1', reason='opt-in 56M cache test')
@pytest.mark.parametrize('chart_index', [0, 1, 2])
def test_real_56m_600_rows_from_bos_and_prefix(chart_index):
    from ensomi_model.r2.data import Corpus
    from ensomi_model.r2.model import R2Config, R2Model
    torch.set_num_threads(2)
    corpus = Corpus('artifacts/r2-cache/v1', 'fit_dev', star_conditions=False)
    table = corpus.table[corpus.table.K >= 2100].copy()
    table['ln_share'] = table.n_ln / table.n_objects
    choices = [table.sort_values('ln_share', ascending=False).iloc[0]]
    choices.extend(table.sort_values('sha256').iloc[:2].to_dict('records'))
    row = choices[chart_index]
    source = corpus.chart(row['sha256'])
    data = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    model = R2Model(R2Config(**data['model_config'])).eval()
    model.load_state_dict(data['model'])
    for prefix in (0, source.K // 3):
        kwargs = dict(prefix_actions=source.actions[:prefix], prefix_gap=source.gap[:prefix],
                      seed=954, stop=prefix + 600)
        expected = reference_continue_chart(model, source.head_ms, source.song_ms, source.grid, **kwargs)
        actual = continue_chart(model, source.head_ms, source.song_ms, source.grid, **kwargs)
        for a, b in zip(actual, expected):
            assert a.tobytes() == b.tobytes()
