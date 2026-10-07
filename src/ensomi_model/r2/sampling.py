"""Free-running generation: sample a whole continuation decision by decision.

Each decision: mask the support, sample the joint action (CPU float64 Gumbel
maximum), draw a fair orientation, then the gap releases one lane at a time in
that orientation's order from the directed pointer, and commit. The EOS
decision closes every hold inside (t_last, T). One ``torch.Generator`` per
continuation, seeded from an integer that is logged by the caller. The learned
history is extended with the TCN's online cache (fixed weights only).

``track`` is the effective track (``request_set.effective_track``); the model applies
rule L per decision. Requests, their validation and the generation record live in
``generate.py``. The ``baseline`` slot (the formulation's baseline style rho) admits
only ``None``: R2 carries identity through committed history alone.
"""
from __future__ import annotations

import numpy as np
import torch

from .common import ACTIONS, ContractError, GridArrays
from .features import (Chart, gap_release_lanes, history_tokens, lane_query, make_factor, query_features,
                       row_placements)
from .state import action_support_mask, replay_decisions
from .strain import action_head_masks
from .ln_level import resolve_ln_controls, resolve_ln_level


def _gumbel_argmax(logits: torch.Tensor, generator) -> int:
    logits = logits.detach().cpu().to(torch.float64)
    u = torch.rand(len(logits), generator=generator, dtype=torch.float64).clamp_min(torch.finfo(torch.float64).tiny)
    return int((logits - (-u.log()).log()).argmax())


BASELINE_NONE = 'none (not implemented; identity carried by committed history only)'


def require_no_baseline(baseline):
    if baseline is not None:
        raise ContractError('The baseline style rho is not implemented in R2; the baseline slot admits only None')


def _scope_closed(chart: Chart, k: int, a: float, b: float) -> bool:
    """After decision k: no lane still holds an LN headed in [a, b)."""
    starts = chart.derived().start[k + 1]
    return not any(np.isfinite(x) and a <= x < b for x in starts)


@torch.no_grad()
def continue_chart(model, head_ms, song_ms: float, grid: GridArrays, prefix_actions=None, prefix_gap=None,
                   track=(), seed: int = 954, stop: int | None = None, *, baseline=None, close_scope=None,
                   head_mask_bias=None, min_hold_ms=None, min_hold_stats=None, ln_level='unknown',
                   source_ln_level=None, ln_prior=None, star=None, ln_level_stats=None,
                   ln_length=None, source_ln_length=None):
    """Return (actions [n,4], gap [n,4]) for decisions 0..n-1, n = stop (default through EOS).

    ``close_scope=(a, b)`` ends generation early, at the first decision at or after the exit
    decision of [a, b) after which no LN headed in [a, b) is still held (realised-response runs).

    ``head_mask_bias(k, chart, held, logp)`` optionally returns 15 finite additive
    biases, ordered by masks 1..15 (bit l denotes lane l). Inputs are read-only:
    chart contains every committed decision < k, including the source prefix;
    held is its replayed held-before state; logp is the normalized [625] action
    log probability tensor after support masking. The next call exposes the last sampled
    action and gaps; a call also occurs at EOS. Biases apply only to the action
    Gumbel maximum, with zero for the empty mask/EOS. The release pointer is
    unchanged. None and all-zero biases preserve the original sampling bits.
    A caller collecting a trace must consume returned arrays after early stop.

    ``min_hold_ms`` optionally requires generated releases to be strictly more
    than this many milliseconds after their held start. The action and release
    pointer supports are masked before head biasing. If no legal action survives
    (including forced EOS closure), both supports revert for that whole decision.
    Existing prefix releases are unchanged. None preserves the original sampling
    bits. The threshold must be finite and nonnegative.
    ``min_hold_stats``, when supplied, accumulates integer ``decisions`` and
    ``fallbacks`` counts; decisions include generated head rows and EOS, exclude
    the supplied prefix, and count each row once regardless of release pointers.

    For an LN-level-enabled model, ``ln_level`` is unknown, oracle, prior, or a
    fixed numeric share in [0, 1]. Oracle requires ``source_ln_level``; prior
    requires ``ln_prior`` and ``star``. Resolve once per whole song with an RNG
    independent of action sampling and hold the value fixed through EOS.
    ``ln_level_stats`` receives the resolved mode, value and prior provenance.
    With the length input enabled, oracle also uses ``source_ln_length``; prior
    draws share and length jointly from v2. A fixed share optionally accepts
    ``ln_length`` in log2 beats. Unknown clears both inputs; absent length stays
    unknown. Metadata includes the length's known bit, value and units.
    A disabled model ignores these inputs and keeps the original sampling path.
    """
    require_no_baseline(baseline)
    if min_hold_ms is not None and (not np.isfinite(min_hold_ms) or min_hold_ms < 0):
        raise ContractError('Minimum hold duration must be finite and nonnegative')
    level_enabled = getattr(model.config, 'ln_level', 'off') == 'on'
    length_enabled = getattr(model.config, 'ln_length', 'off') == 'on'
    level = length = None
    if level_enabled:
        if length_enabled:
            level, length, level_record = resolve_ln_controls(
                ln_level, ln_length=ln_length, source_ln_level=source_ln_level, source_ln_length=source_ln_length,
                ln_prior=ln_prior, star=star, head_ms=head_ms, song_ms=song_ms, seed=seed)
        else:
            level, level_record = resolve_ln_level(ln_level, source_ln_level=source_ln_level, ln_prior=ln_prior,
                                                   star=star, head_ms=head_ms, song_ms=song_ms, seed=seed)
        if ln_level_stats is not None:
            ln_level_stats.update(level_record)
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
    exit_k = None if close_scope is None else int(np.searchsorted(head_ms, close_scope[1], side='left'))
    n = stop
    for k in range(s, stop):
        chart = base.with_decisions(actions[:k], gap[:k])
        held = chart.derived().held[k]
        before = temporal.read(cache).to(model.dtype)[None]
        mk = torch.stack(marks) if marks else None
        vis = torch.ones(1, len(marks), dtype=torch.bool, device=model.device) if marks else None
        if length_enabled:
            qf = model.query_features(chart, [k], ln_level=level, ln_length=length)
        else:
            qf = model.query_features(chart, [k], ln_level=level) if level_enabled else model._t(query_features(chart, [k]))
        z = model.hands(before, qf, mk, vis, model.row_condition(chart, track, [k]))
        mask = action_support_mask(held, k == K)
        hold_masked, fallback = False, False
        if min_hold_ms is not None:
            restricted = mask.copy()
            starts = chart.derived().start[k]
            for lane in np.flatnonzero(held):
                if chart.time(k) - starts[lane] <= min_hold_ms:
                    restricted &= ACTIONS[:, lane] != 1
                if not np.any(chart.candidates(k).times - starts[lane] > min_hold_ms):
                    restricted &= ACTIONS[:, lane] < 2
            if restricted.any():
                mask, hold_masked = restricted, True
            else:
                # Restoring only the row support could choose an empty pointer support.
                fallback = True
        if min_hold_stats is not None:
            min_hold_stats['decisions'] = min_hold_stats.get('decisions', 0) + 1
            min_hold_stats['fallbacks'] = min_hold_stats.get('fallbacks', 0) + int(fallback)
        logp = model.action_log_probs(z, mask[None])[0]
        if head_mask_bias is not None:
            bias = torch.as_tensor(head_mask_bias(k, chart, held, logp), dtype=torch.float64, device='cpu')
            if bias.shape != (15,) or not torch.isfinite(bias).all():
                raise ContractError('Head-mask bias must contain 15 finite values')
            if torch.any(bias != 0):
                ids = torch.as_tensor(action_head_masks(held, eos=k == K))
                action_bias = torch.cat((bias.new_zeros(1), bias))[ids]
                logp = logp.detach().cpu().to(torch.float64) + action_bias
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
                if hold_masked:
                    keep = torch.as_tensor(f.times - starts[lane] > min_hold_ms, device=scores.device)
                    scores = scores.masked_fill(~keep, -torch.inf).log_softmax(0)
                u = float(f.times[_gumbel_argmax(scores, generator)])
                placed[lane] = (u, False)
                rel[lane] = u
        actions[k], gap[k] = codes, rel
        base = chart
        if exit_k is not None and k >= exit_k:
            done = base.with_decisions(actions[:k + 1], gap[:k + 1])
            if _scope_closed(done, k, *close_scope):
                n = k + 1
                break
        if k < K:
            nxt = base.with_decisions(actions[:k + 1], gap[:k + 1])
            tok = model._t(history_tokens(nxt, k + 1)[k])
            cache = temporal.append(cache, tok)
            if k % stride == 0:
                marks.append(temporal.read(cache))
    if was_training:
        model.train()
    return actions[:n], gap[:n]
