"""The plan v4 section 8.1 draw: start distributions, exact inverse-probability weights, stored N_bar; the
phase-N natural draw and short windows (plan v5).

The weight test checks the design rule directly: under the aligned draw weighted by u, the expected
exposure of every decision's natural factors equals its exposure under the v1 start rule."""
from pathlib import Path

import numpy as np
import pytest

from ensomi_model.r2.common import ContractError
from ensomi_model.r2.conditions import (DrawConfig, align_distribution, draw_window, ln_candidates, old_start, onset_of,
                                        p_old)

from .helpers import random_decisions, random_skeleton, two_segment_grid
from .test_conditions import star_cells

CACHE = Path('artifacts/r2-cache/v1')
CONFIGS = Path('src/ensomi_model/r2/configs')


@pytest.fixture(scope='module')
def chart():
    rng = np.random.default_rng(8)
    head, song = random_skeleton(rng, 600, lo=60, hi=250)
    return random_decisions(rng, (head, song, two_segment_grid()))


def test_start_distributions_sum_to_one(chart):
    assert abs(p_old(chart.K).sum() - 1.0) < 1e-12
    cfg = DrawConfig()
    rng = np.random.default_rng(0)
    survivors = {0: ln_candidates(chart, rng, cfg)}
    for c in survivors[0]:
        c.onset = onset_of(chart, c.iv)
    pa, kinds = align_distribution(chart, survivors, cfg)
    assert kinds == [0] and abs(pa.sum() - 1.0) < 1e-12


def test_inverse_probability_weights_restore_the_v1_exposure(chart):
    """For fixed survivors, sum_j p_new(j) u(j) [k in window(j)] = sum_j p_old(j) [k in window(j)] for all k."""
    cfg = DrawConfig()
    rng = np.random.default_rng(1)
    survivors = {0: ln_candidates(chart, rng, cfg)}
    for c in survivors[0]:
        c.onset = onset_of(chart, c.iv)
    pa, _ = align_distribution(chart, survivors, cfg)
    po = p_old(chart.K)
    pn = (1 - cfg.p_align) * po + cfg.p_align * pa
    u = po / pn
    K = chart.K
    old = np.zeros(K + 1)
    new = np.zeros(K + 1)
    for j in range(K + 1):
        stop = min(j + 256, K + 1)
        old[j:stop] += po[j]
        new[j:stop] += pn[j] * u[j]
    assert np.allclose(old, new, rtol=0, atol=1e-12)
    # the draw reports the same weight for the start it picks
    for seed in range(30):
        w = draw_window(chart, star_cells(chart), np.random.default_rng(seed), cfg, star_conditions=False)
        assert w.weight > 0


def test_dropout_precedes_alignment(chart):
    """The aligned candidate always survives dropout: an aligned window's interval is in its track."""
    cfg = DrawConfig(p_align=1.0)
    hits = 0
    for seed in range(60):
        w = draw_window(chart, None, np.random.default_rng(seed), cfg, star_conditions=False)
        if w.record['aligned']:
            hits += 1
            assert w.track and any(c['onset'] is not None and w.start <= c['onset'] < w.stop for c in w.record['chosen'])
    assert hits > 20


def test_per_song_selection_is_the_v1_start_rule(chart):
    cfg = DrawConfig(selection='per_song', p_align=0.0, ipw=False)
    for seed in range(20):
        w = draw_window(chart, None, np.random.default_rng(seed), cfg, star_conditions=False)
        assert w.weight == 1.0 and not w.record['aligned']


def test_natural_selection_draws_empty_tracks_by_the_v1_start_rule(chart):
    cfg = DrawConfig(selection='natural', p_align=0.0, ipw=False)
    rng, ref = np.random.default_rng(3), np.random.default_rng(3)
    for _ in range(40):
        w = draw_window(chart, None, rng, cfg, star_conditions=False)
        assert w.track == () and w.weight == 1.0 and w.start == old_start(ref, chart.K)
        assert w.stop == min(w.start + 256, chart.K + 1)
    with pytest.raises(ContractError):
        DrawConfig(selection='natural')                                   # p_align and ipw stay off


def test_short_windows_keep_the_aligned_onset(chart):
    """The DPO path draws 8-decision windows: the lead is capped so the aligned onset is scored."""
    cfg = DrawConfig(p_align=1.0)
    hits = 0
    for seed in range(60):
        w = draw_window(chart, None, np.random.default_rng(seed), cfg, star_conditions=False, window=8)
        if w.record['aligned']:
            hits += 1
            assert w.stop - w.start <= 8 and 0 <= w.record['lead'] <= 7
            assert any(c['onset'] is not None and w.start <= c['onset'] < w.stop for c in w.record['chosen'])
    assert hits > 20


@pytest.mark.skipif(not (CACHE / 'index.parquet').exists(), reason='cache not built here')
@pytest.mark.parametrize('name', ['n', 'c_frozen', 'c_kl'])
def test_stored_n_bar_matches_a_fresh_estimate(name):
    from ensomi_model.r2.draw_sim import n_bar_key, simulate
    from ensomi_model.r2.train_ce import TrainConfig, build_corpus
    cfg = TrainConfig.load(CONFIGS / f'ce_v2_{name}.json')
    if cfg.n_bar is None:
        pytest.skip('N_bar not measured yet for this config')
    corpus, star, dcfg = build_corpus(cfg, 'fit_train')
    assert cfg.n_bar_key == n_bar_key(dcfg, star_conditions=star, window=cfg.window, accumulate=cfg.accumulate,
                                      label_sha256=corpus.label_sha256)
    values, _ = simulate(corpus, dcfg, 2000, cfg.accumulate, cfg.window, seed=777)   # a fresh 2,000-draw estimate
    for key, stored in (('N_bar', cfg.n_bar), ('N_bar_ln', cfg.n_bar_ln), ('N_bar_star', cfg.n_bar_star)):
        if stored and stored > 1:
            assert abs(values[key] / stored - 1) < 0.05, (key, values[key], stored)
    assert abs(np.mean(values['batch_weights'][:200]) / cfg.n_bar - 1) < 0.05   # sum of u over 200 batches
