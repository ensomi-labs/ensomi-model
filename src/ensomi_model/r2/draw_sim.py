"""Stage-0 draw simulation (plan v4 section 8.3): the fixed loss divisors and draw properties D1-D8.

Draws windows from fit_train with a training config's draw, evaluates rule L on the source
occupancy, and reports

- N_bar: mean over batches of ``accumulate`` windows of sum_j w_j (w_j = u on decisions with
  V_j empty, 1 otherwise); N_bar_ln, N_bar_star: mean count of Omega_kappa factors per batch
  (an action factor per head decision plus one per gap release, on decisions reading the kind);
- D1 onset coverage, D2 song-length independence, D3 informativeness, D4 active share,
  D6 natural effective sample size, D7 long-span coverage, D8 rule-L cost, each against the
  threshold the plan states, and the weight distribution.

The result's ``n_bar_key`` must equal the trainer's (``n_bar_key``) for the stored N_bar to be
used. Each phase has its own draw (plan v5): phase N's natural draw has N_bar_ln = N_bar_star = 0,
and the two phase-C configs share one draw. Run on the mac::

    .venv/bin/python -m ensomi_model.r2.draw_sim --config src/ensomi_model/r2/configs/ce_v2_n.json \
        --draws 2000 --out artifacts/r2-stage0/draw-sim/n.json
    .venv/bin/python -m ensomi_model.r2.draw_sim --config src/ensomi_model/r2/configs/ce_v2_c_frozen.json \
        --draws 2000 --out artifacts/r2-stage0/draw-sim/c.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from .conditions import DrawConfig
from .locality import masked, reads

THRESHOLDS = dict(D1=0.70, D3=0.02, D4_ln=0.25, D4_star=0.25, D6=0.50, D7=0.20, D8_piece16=0.05)


def n_bar_key(draw: DrawConfig, *, star_conditions: bool, window: int, accumulate: int, label_sha256) -> str:
    payload = dict(draw=draw.hash(), star_conditions=bool(star_conditions), window=int(window),
                   accumulate=int(accumulate), labels=label_sha256 if star_conditions else None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def gap_release_counts(chart, ks):
    d = chart.derived()
    return np.array([int(sum(d.held[k][l] and chart.actions[k][l] in (2, 3, 4) for l in range(4))) for k in ks])


def window_measure(chart, draw, cfg: DrawConfig):
    """Per-window quantities on the source occupancy."""
    ks = np.arange(draw.start, draw.stop)
    d = chart.derived()
    vis = np.zeros((len(ks), 2), dtype=bool)
    reads_iv = np.zeros((len(ks), len(draw.track)), dtype=bool)
    for i, k in enumerate(ks):
        for j, iv in enumerate(draw.track):
            if reads(chart, iv, int(k), d.held[int(k)]):
                reads_iv[i, j] = True
                vis[i, iv.kind] = True
    natural = ~vis.any(1)
    w = np.where(natural, draw.weight, 1.0)
    factors = (ks < chart.K).astype(np.int64) + gap_release_counts(chart, ks)
    head = ks < chart.K
    return ks, vis, reads_iv, natural, w, factors, head


def _z(chart, iv, j, cfg: DrawConfig):
    from .conditions import _weights, Candidate
    _, z = _weights(chart, [Candidate(iv, 'piece', 0.0)], [j], cfg)
    return z[0]


def simulate(corpus, draw_cfg: DrawConfig, draws: int, accumulate: int, window: int, seed: int):
    rng = np.random.default_rng(seed)
    batch_w, batch_ln, batch_star = [], [], []
    acc = [0.0, 0, 0]
    d1_active = d1_scored = 0
    per_window_active, per_window_K = [], []
    d3_hits = scored_heads = 0
    d4 = np.zeros(2)
    natural_w = []
    d7_long = d7_ln = 0
    d8 = defaultdict(lambda: dict(in_scope=0, outside=0, onsets=0, onset_open_hold=0, onset_masked=0))
    weights_aligned, weights_other = [], []
    t0 = time.time()
    for n in range(draws):
        dr = corpus.draw(rng, window)
        chart = corpus.chart(dr.sha)
        ks, vis, reads_iv, natural, w, factors, head = window_measure(chart, dr, draw_cfg)
        acc[0] += float(w.sum())
        acc[1] += int(factors[vis[:, 0]].sum())
        acc[2] += int(factors[vis[:, 1]].sum())
        if (n + 1) % accumulate == 0:
            batch_w.append(acc[0])
            batch_ln.append(acc[1])
            batch_star.append(acc[2])
            acc = [0.0, 0, 0]
        (weights_aligned if dr.record.get('aligned') else weights_other).append(dr.weight)
        natural_w += w[natural].tolist()
        scored_heads += int(head.sum())
        d4 += (vis & head[:, None]).sum(0)
        active = 0
        for j, iv in enumerate(dr.track):
            m = masked(chart, iv)
            meta = dr.record['chosen'][j]
            key = f"{'ln' if iv.kind == 0 else 'star'}:{meta['cls']}"
            s = d8[key]
            s['in_scope'] += m['in_scope_head_decisions']
            s['outside'] += len(m['in_scope_outside_v'])
            s['onsets'] += 1
            s['onset_open_hold'] += int(m['onset_open_hold'])
            s['onset_masked'] += int(m['onset_masked'])
            if not reads_iv[:, j].any():
                continue
            active += 1
            d1_active += 1
            f = m['first_visible']
            if f is not None and dr.start <= f < dr.stop:
                d1_scored += 1
                if iv.kind == 0:
                    z = _z(chart, iv, dr.start, draw_cfg)
                    if np.isfinite(z) and z >= draw_cfg.z_min:
                        near = (ks >= f) & (ks < f + 16) & reads_iv[:, j] & head
                        d3_hits += int(near.sum())
            if iv.kind == 0:
                beats = float(chart.grid.beat(iv.b) - chart.grid.beat(iv.a))
                lv = reads_iv[:, j] & head
                d7_ln += int(lv.sum())
                if beats > 64:
                    d7_long += int(lv.sum())
        per_window_active.append(active)
        per_window_K.append(chart.K)
        if (n + 1) % 200 == 0:
            print(f'[{time.time() - t0:7.1f}s] {n + 1}/{draws}', flush=True)
    natural_w = np.array(natural_w)
    ess = float(natural_w.sum() ** 2 / (natural_w ** 2).sum()) if len(natural_w) else None
    X = np.array(per_window_K, dtype=float)
    Y = np.array(per_window_active, dtype=float)
    A = np.column_stack([np.ones_like(X), X])
    beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
    resid = Y - A @ beta
    cov = resid @ resid / max(1, len(Y) - 2) * np.linalg.inv(A.T @ A)
    slope, slope_se = float(beta[1]), float(np.sqrt(cov[1, 1]))
    d8_out = {k: dict(v, outside_share=v['outside'] / v['in_scope'] if v['in_scope'] else None,
                      open_hold_onset_share=v['onset_open_hold'] / v['onsets'] if v['onsets'] else None)
              for k, v in sorted(d8.items())}
    values = dict(
        N_bar=float(np.mean(batch_w)), N_bar_ln=float(np.mean(batch_ln)), N_bar_star=float(np.mean(batch_star)),
        N_bar_se=float(np.std(batch_w, ddof=1) / np.sqrt(len(batch_w))), batches=len(batch_w),
        N_bar_ln_se=float(np.std(batch_ln, ddof=1) / np.sqrt(len(batch_ln))),
        N_bar_star_se=float(np.std(batch_star, ddof=1) / np.sqrt(len(batch_star))),
        batch_weights=[float(x) for x in batch_w],
        D1=d1_scored / d1_active if d1_active else None,
        D2=dict(slope_per_row=slope, slope_se=slope_se, contains_zero=bool(abs(slope) <= 2 * slope_se)),
        D3=d3_hits / scored_heads if scored_heads else None,
        D4_ln=float(d4[0] / scored_heads), D4_star=float(d4[1] / scored_heads),
        D6=ess / len(natural_w) if ess else None,
        D7=d7_long / d7_ln if d7_ln else None,
        D8=d8_out,
        weights=dict(aligned=_q(weights_aligned), other=_q(weights_other),
                     aligned_share=len(weights_aligned) / draws))
    p16 = d8_out.get('ln:piece-16', {}).get('outside_share')
    checks = dict(D1=_ge(values['D1'], THRESHOLDS['D1']), D2=values['D2']['contains_zero'],
                  D3=_ge(values['D3'], THRESHOLDS['D3']), D4_ln=_ge(values['D4_ln'], THRESHOLDS['D4_ln']),
                  D4_star=_ge(values['D4_star'], THRESHOLDS['D4_star']), D6=_ge(values['D6'], THRESHOLDS['D6']),
                  D7=_ge(values['D7'], THRESHOLDS['D7']),
                  D8_piece16=None if p16 is None else bool(p16 <= THRESHOLDS['D8_piece16']))
    return values, checks


def _ge(x, t):
    return None if x is None else bool(x >= t)


def _q(xs):
    if not xs:
        return None
    xs = np.asarray(xs)
    return dict(n=len(xs), mean=float(xs.mean()), q05=float(np.quantile(xs, .05)), q50=float(np.median(xs)),
                q95=float(np.quantile(xs, .95)), min=float(xs.min()), max=float(xs.max()))


def run(config_path, draws, out, seed=20261006):
    from .train_ce import TrainConfig, build_corpus
    cfg = TrainConfig.load(config_path)
    corpus, star, draw_cfg = build_corpus(cfg, 'fit_train')
    t0 = time.time()
    values, checks = simulate(corpus, draw_cfg, draws, cfg.accumulate, cfg.window, seed)
    key = n_bar_key(draw_cfg, star_conditions=star, window=cfg.window, accumulate=cfg.accumulate,
                    label_sha256=corpus.label_sha256)
    result = dict(config=str(config_path), draws=draws, seed=seed, accumulate=cfg.accumulate, window=cfg.window,
                  star_conditions=star, draw=draw_cfg.to_dict(), draw_hash=draw_cfg.hash(), n_bar_key=key,
                  labels_sha256=corpus.label_sha256, thresholds=THRESHOLDS, values=values, checks=checks,
                  wall_s=time.time() - t0,
                  config_values=dict(n_bar=values['N_bar'], n_bar_ln=values['N_bar_ln'],
                                     n_bar_star=values['N_bar_star'], n_bar_key=key))
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=1))
        from .receipts import write_receipt
        write_receipt(Path(out).with_suffix('.receipt.json'), config=dict(config=str(config_path), draws=draws),
                      outputs=dict(n_bar_key=key), seed=seed, device='cpu')
    print(json.dumps(result, indent=1), flush=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--draws', type=int, default=2000)
    p.add_argument('--seed', type=int, default=20261006)
    p.add_argument('--out', default=None)
    a = p.parse_args(argv)
    run(a.config, a.draws, a.out, a.seed)


if __name__ == '__main__':
    sys.exit(main())
