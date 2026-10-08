import numpy as np
import torch

from ensomi_model.audio_rows.audio import FEATURE_DIM
from ensomi_model.audio_rows.data import Dataset
from ensomi_model.audio_rows.generate import decode
from ensomi_model.audio_rows.model import HeadModel, factor_losses
from ensomi_model.r2.common import GridArrays, grid_from_arrays

from .helpers import tiny_build


def test_triple_beats_give_no_probability_to_counts_above_twelve():
    model = HeadModel()
    logits = model.count_logits(torch.zeros(1, model.cfg.hidden), torch.tensor([[0.0, 1.0]]))
    assert torch.isneginf(logits[0, 13:]).all() and torch.isfinite(logits[0, :13]).all()


def test_partial_first_and_last_beats_emit_exactly_their_sampled_counts():
    torch.manual_seed(0)
    grid = GridArrays.from_grid(grid_from_arrays(np.array([[100.0, 500.0, 4]]), np.array([[100.0, 4]])))
    song_ms = 8_105.0
    feats = np.random.default_rng(0).normal(size=(811, FEATURE_DIM)).astype(np.float16)
    heads, rec = decode(HeadModel().eval(), feats, grid, song_ms, seed=4)
    assert len(heads) == rec['count'].sum() == rec['mask'].sum()
    assert heads.min() >= 0.0 and heads.max() <= song_ms - 2.0


def test_training_steps_on_a_data_window_lower_its_loss(tmp_path):
    data = Dataset(tiny_build(tmp_path), 'fitted')
    w = data.window(data.train[0], 3, 64, np.random.default_rng(0), jitter_ms=40.0, tempo_error=0.001,
                    section_beats=(8, 16, 32))
    torch.manual_seed(0)
    model = HeadModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(5):
        f = factor_losses(model, **w)
        loss = f['count'].sum() + f['lattice'][f['head']].sum() + f['slot'][f['head']].sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
