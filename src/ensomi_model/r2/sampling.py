"""Free-running generation: sample a whole continuation decision by decision.

Each decision: mask the support, sample the joint action (CPU float64 Gumbel
maximum), draw a fair orientation, then the gap releases one lane at a time in
that orientation's order from the directed pointer, and commit. The EOS
decision closes every hold inside (t_last, T). One ``torch.Generator`` per
continuation, seeded from an integer that is logged by the caller. The learned
history is extended with the TCN's online cache (fixed weights only).
"""
from __future__ import annotations

import numpy as np
import torch

from .common import ACTIONS, ContractError, GridArrays
from .features import (Chart, gap_release_lanes, history_tokens, lane_query, make_factor, query_features,
                       row_placements)
from .state import action_support_mask, replay_decisions


def _gumbel_argmax(logits: torch.Tensor, generator) -> int:
    logits = logits.detach().cpu().to(torch.float64)
    u = torch.rand(len(logits), generator=generator, dtype=torch.float64).clamp_min(torch.finfo(torch.float64).tiny)
    return int((logits - (-u.log()).log()).argmax())


@torch.no_grad()
def continue_chart(model, head_ms, song_ms: float, grid: GridArrays, prefix_actions=None, prefix_gap=None,
                   track=(), seed: int = 954, stop: int | None = None):
    """Return (actions [n,4], gap [n,4]) for decisions 0..stop-1 (default through EOS)."""
    was_training = model.training
    model.eval()
    K = len(head_ms)
    stop = K + 1 if stop is None else stop
    s = 0 if prefix_actions is None else len(prefix_actions)
    if s > K:
        raise ContractError('Prefix longer than the head skeleton')
    actions = np.zeros((K + 1, 4), dtype=np.int64)
    gap = np.full((K + 1, 4), np.nan)
    if s:
        actions[:s], gap[:s] = prefix_actions, prefix_gap
        replay_decisions(head_ms, song_ms, actions[:s], gap[:s])  # raises if the prefix is not legal
    generator = torch.Generator().manual_seed(int(seed))
    temporal = model.temporal
    cache = temporal.empty_cache()
    marks = []
    stride = model.config.stride
    if s:
        chart = Chart(head_ms, song_ms, grid, actions[:s], gap[:s])
        toks = model._t(history_tokens(chart, min(s, K)))
        for i in range(len(toks)):
            cache = temporal.append(cache, toks[i])
            if i % stride == 0:
                marks.append(temporal.read(cache))
    base = Chart(head_ms, song_ms, grid, actions[:0], gap[:0])
    for k in range(s, stop):
        chart = base.with_decisions(actions[:k], gap[:k])
        held = chart.derived().held[k]
        before = temporal.read(cache).to(model.dtype)[None]
        mk = torch.stack(marks) if marks else None
        vis = torch.ones(1, len(marks), dtype=torch.bool, device=model.device) if marks else None
        qf = model._t(query_features(chart, [k]))
        z = model.hands(before, qf, mk, vis, model.row_condition(chart, track, [k]))
        mask = action_support_mask(held, k == K)
        logp = model.action_log_probs(z, mask[None])[0]
        codes = ACTIONS[_gumbel_argmax(logp, generator)]
        orientation = int(torch.randint(2, (), generator=generator))
        lanes = gap_release_lanes(held, codes, orientation)
        rel = np.full(4, np.nan)
        if lanes:
            placed = row_placements(chart, k, held, codes)
            lq = lane_query(chart, [k]).astype(np.float32)[0]
            for lane in lanes:
                f = make_factor(chart, k, 0, codes, lane, orientation, dict(placed), lq)
                ctx = model.factor_contexts(z, [f])
                scores, _, _ = model.all_pair_scores(ctx, [f], chart, track)
                u = float(f.times[_gumbel_argmax(scores, generator)])
                placed[lane] = (u, False)
                rel[lane] = u
        actions[k], gap[k] = codes, rel
        base = chart
        if k < K:
            nxt = base.with_decisions(actions[:k + 1], gap[:k + 1])
            tok = model._t(history_tokens(nxt, k + 1)[k])
            cache = temporal.append(cache, tok)
            if k % stride == 0:
                marks.append(temporal.read(cache))
    if was_training:
        model.train()
    return actions[:stop], gap[:stop]
