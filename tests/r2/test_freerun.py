"""Pre-run test 8: a briefly trained model free-runs a whole short chart legally; hand-swap identity."""
import numpy as np
import pytest
import torch

from ensomi_model.r2.export import export_chart, minimal_header
from ensomi_model.r2.features import Interval
from ensomi_model.r2.model import R2Config, R2Model
from ensomi_model.r2.report import chart_summary
from ensomi_model.r2.sampling import continue_chart

from .helpers import chart_from_objects
from .test_leakage_causality import source_objects


@pytest.fixture(scope='module')
def trained():
    rng = np.random.default_rng(1)
    objects, song = source_objects(rng, K=160)
    chart, dec = chart_from_objects(objects, song)
    torch.manual_seed(171)
    model = R2Model(R2Config())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(200):
        start = int(rng.integers(0, chart.K + 1))
        out = model.window(chart, start, min(start + 32, chart.K + 1))
        loss = -out.total.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)
    return model, chart, dec


@pytest.mark.parametrize('seed_rows', [0, 100])
def test_freerun_whole_chart(trained, seed_rows):
    model, chart, dec = trained
    track = (Interval(0, 0.0, dec.song_ms, 0.3),) if seed_rows else ()
    acts, gap = continue_chart(model, dec.head_ms, dec.song_ms, chart.grid,
                               dec.actions[:seed_rows] if seed_rows else None,
                               dec.gap_release_ms[:seed_rows] if seed_rows else None, track=track, seed=954)
    assert np.array_equal(acts[:seed_rows], dec.actions[:seed_rows])
    objects, text = export_chart(dec.head_ms, dec.song_ms, acts, gap, None, minimal_header(dec.grid_segments))
    summary = chart_summary(objects, dec.head_ms, dec.song_ms, chart.grid)
    assert summary['violations'] == {}
    assert summary['heads_present']
    assert len(acts) == chart.K + 1  # EOS decided; export would raise on an open hold


@pytest.mark.parametrize('mode', ['film', 'tokens'])
def test_conditioner_hand_swap_identity(mode):
    torch.manual_seed(0)
    model = R2Model(R2Config(conditioner=mode, token_lead_in=mode == 'tokens')).double()
    with torch.no_grad():
        for p in list(model.film.parameters()) + list(model.tokens.parameters()):
            p.normal_(0, 0.1)
    z = torch.randn(7, 2, 128, dtype=torch.float64)
    if mode == 'film':
        cond = torch.randn(7, model.film.inputs, dtype=torch.float64)   # 2 roles x 2 kinds x 17 channels
    else:
        cond = torch.randn(7, 3, 18, dtype=torch.float64)
    out = model.condition(z, cond)
    swapped = model.condition(z.flip(1), cond)
    assert torch.allclose(out.flip(1), swapped, atol=1e-12, rtol=0)
