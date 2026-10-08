"""Free-running generation: sample a whole continuation decision by decision.

Each decision: mask the support, sample the joint action (CPU float64 Gumbel
maximum), draw a fair orientation, then the gap releases one lane at a time in
that orientation's order from the directed pointer, and commit. The EOS
decision closes every hold inside (t_last, T). One ``torch.Generator`` per
continuation, seeded from an integer that is logged by the caller. The learned
history is extended with the TCN's online cache (fixed weights only).

``track`` is the effective track (``request_set.effective_track``); the model applies
rule L per decision. Requests, their validation and the generation record live in
``generate.py``. The separate ``baseline`` slot admits only ``None``; optional
chart identity enters through the configured self-anchor and theta readers.
"""
from __future__ import annotations

from dataclasses import fields

import numpy as np
import torch

from .common import ACTIONS, HAND_LANES, ContractError, GridArrays, psi_pair
from .candidates import phase
from .features import (Chart, Derived, _onehot, _views, gap_release_lanes, history_tokens, lane_query,
                       make_factor, row_placements)
from .state import action_support_mask, replay_decisions
from .strain import action_head_masks
from .ln_level import resolve_ln_controls, resolve_ln_level


def _gumbel_argmax(logits: torch.Tensor, generator) -> int:
    logits = logits.detach().cpu().to(torch.float64)
    u = torch.rand(len(logits), generator=generator, dtype=torch.float64).clamp_min(torch.finfo(torch.float64).tiny)
    return int((logits - (-u.log()).log()).argmax())


BASELINE_NONE = 'none (not implemented)'


def require_no_baseline(baseline):
    if baseline is not None:
        raise ContractError('The baseline style rho is not implemented in R2; the baseline slot admits only None')


def _scope_closed(chart: Chart, k: int, a: float, b: float) -> bool:
    """After decision k: no lane still holds an LN headed in [a, b)."""
    starts = chart.derived().start[k + 1]
    return not any(np.isfinite(x) and a <= x < b for x in starts)


class IncrementalState:
    """Committed chart with preallocated replay arrays; append changes its live views."""

    def __init__(self, head_ms, song_ms, grid, actions, gap, n=0):
        self.actions, self.gap = actions, gap
        self.chart = Chart(head_ms, song_ms, grid, actions[:n], gap[:n])
        initial = self.chart.derived()
        self.arrays = {}
        self.extra = {}
        for f in fields(initial):
            value = getattr(initial, f.name)
            extra = len(value) - n
            arr = np.zeros((len(actions) + extra,) + value.shape[1:], dtype=value.dtype)
            arr[:len(value)] = value
            self.arrays[f.name], self.extra[f.name] = arr, extra
        self._view(n)

    def _view(self, n):
        self.chart.actions, self.chart.gap = self.actions[:n], self.gap[:n]
        self.chart.cache['derived'] = Derived(**{name: arr[:n + self.extra[name]]
                                                for name, arr in self.arrays.items()})

    def append(self, codes, gap):
        k, d = self.chart.n, self.arrays
        self.actions[k], self.gap[k] = codes, gap
        t = self.chart.time(k)
        held = d['held'][k]
        closed = held & (codes != 0)
        attack = np.where(held, codes >= 3, (codes == 1) | (codes == 2))
        ln_head = np.where(held, codes == 4, codes == 2)
        release = np.where(closed, np.where(codes == 1, t, gap), np.nan)
        d['held'][k + 1] = (held & ~closed) | ln_head
        d['start'][k + 1] = np.where(ln_head, t, np.where(closed, np.nan, d['start'][k]))
        d['birth'][k + 1] = np.where(ln_head, k, np.where(closed, -1, d['birth'][k]))
        d['release'][k], d['row_release'][k] = release, closed & (codes == 1)
        d['attack'][k], d['ln_head'][k] = attack, ln_head
        d['last_attack'][k + 1] = np.where(attack, t, d['last_attack'][k])
        d['last_release'][k + 1] = np.fmax(release, d['last_release'][k])
        d['cum_heads'][k + 1] = d['cum_heads'][k] + attack.sum()
        d['cum_ln'][k + 1] = d['cum_ln'][k] + ln_head.sum()
        repeat = (attack & d['attack'][k - 1]).sum() if k else 0
        d['cum_repeat'][k + 1] = d['cum_repeat'][k] + repeat
        d['cum_held'][k + 1] = d['cum_held'][k] + held.sum()
        heads = attack.sum()
        d['cum_c3'][k + 1] = d['cum_c3'][k] + (heads >= 3)
        d['cum_c4'][k + 1] = d['cum_c4'][k] + (heads == 4)
        pattern = int((attack * (1 << np.arange(4))).sum())
        d['cum_patterns'][k + 1] = d['cum_patterns'][k]
        d['cum_patterns'][k + 1, pattern] += 1
        d['cum_pattern_repeat'][k + 1] = d['cum_pattern_repeat'][k]
        for i, lag in enumerate((1, 2, 4)):
            if k >= lag:
                d['cum_pattern_repeat'][k + 1, i] += np.array_equal(attack, d['attack'][k - lag])
        lengths = self.chart.grid.beat(release[closed]) - self.chart.grid.beat(d['start'][k, closed])
        d['cum_ln_log_length'][k + 1] = d['cum_ln_log_length'][k] + np.log2(np.maximum(lengths, 1e-4)).sum()
        d['cum_closed_ln'][k + 1] = d['cum_closed_ln'][k] + closed.sum()
        d['cum_hand_heads'][k + 1] = d['cum_hand_heads'][k] + attack.reshape(2, 2).sum(1)
        self._view(k + 1)


def _history_token(chart, k):
    """The k-th history token, with the same float operations as history_tokens."""
    d, ks = chart.derived(), np.array([k])
    acts, rel = chart.actions[ks].astype(np.int64), d.release[ks]
    lane = np.concatenate((_onehot(acts, 5), d.held[ks, :, None].astype(np.float64),
                           (~np.isnan(rel))[..., None].astype(np.float64),
                           _views(chart, rel, ks[:, None], d.start[ks])), -1)
    t, b = chart.head_ms[ks], chart.head_beats[ks]
    prev = max(k - 1, 0)
    shared = np.concatenate((psi_pair(t - chart.head_ms[prev], b - chart.head_beats[prev]),
                             psi_pair(t - chart.head_ms[0], b - chart.head_beats[0]), phase(b),
                             _onehot(d.attack[ks].sum(1), 5), _onehot(d.ln_head[ks].sum(1), 5)), -1)
    hands = [np.concatenate((lane[:, list(order)].reshape(1, 76), shared), -1) for order in HAND_LANES]
    return np.stack(hands, 1).astype(np.float32)[0]


@torch.no_grad()
def continue_chart(model, head_ms, song_ms: float, grid: GridArrays, prefix_actions=None, prefix_gap=None,
                   track=(), seed: int = 954, stop: int | None = None, *, baseline=None, close_scope=None,
                   head_mask_bias=None, min_hold_ms=None, min_hold_stats=None, ln_level='unknown',
                   source_ln_level=None, ln_prior=None, star=None, ln_level_stats=None,
                   ln_length=None, source_ln_length=None, theta='unknown', theta_table=None,
                   theta_source_sha256=None, theta_record=None,
                   _history_cache=None, _history_out=None):
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

    ``theta`` accepts unknown, oracle, prior, or a standardised ten-coordinate
    vector. It is resolved once and held fixed; ``theta_record`` receives its
    value and donor provenance. Disabled theta readers ignore it.
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
    theta_value = None
    theta_enabled = getattr(model.config, 'theta', 'off') == 'on'
    if theta_enabled:
        from .theta import resolve_theta
        theta_value, record = resolve_theta(theta, head_ms=head_ms, song_ms=song_ms,
                                            grid=grid, seed=seed, table=theta_table,
                                            source_sha256=theta_source_sha256)
        if theta_record is not None:
            theta_record.update(record)
    query_kwargs = {}
    if level_enabled:
        query_kwargs['ln_level'] = level
    if length_enabled:
        query_kwargs['ln_length'] = length
    if theta_enabled:
        query_kwargs['theta'] = theta_value
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
    cache, marks = (temporal.empty_cache(), []) if _history_cache is None else _history_cache
    marks = list(marks)
    stride = model.config.stride
    if s and _history_cache is None:
        chart = Chart(head_ms, song_ms, grid, actions[:s], gap[:s])
        toks = model._t(history_tokens(chart, min(s, K)))
        for i in range(len(toks)):
            cache = temporal.append(cache, toks[i])
            if i % stride == 0:
                marks.append(temporal.read(cache))
    state = IncrementalState(head_ms, song_ms, grid, actions, gap, s)
    exit_k = None if close_scope is None else int(np.searchsorted(head_ms, close_scope[1], side='left'))
    n = stop
    for k in range(s, stop):
        chart = state.chart
        held = chart.derived().held[k]
        before = temporal.read(cache).to(model.dtype)[None]
        mk = torch.stack(marks) if marks else None
        vis = torch.ones(1, len(marks), dtype=torch.bool, device=model.device) if marks else None
        qf = model.query_features(chart, [k], **query_kwargs)
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
        state.append(codes, rel)
        if exit_k is not None and k >= exit_k:
            if _scope_closed(chart, k, *close_scope):
                n = k + 1
                break
        if k < K:
            tok = model._t(_history_token(chart, k))
            cache = temporal.append(cache, tok)
            if k % stride == 0:
                marks.append(temporal.read(cache))
    if was_training:
        model.train()
    if _history_out is not None:
        _history_out.update(cache=cache, marks=marks)
    return actions[:n], gap[:n]
