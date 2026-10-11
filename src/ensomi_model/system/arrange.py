"""Frozen R2 arranges a whole head skeleton under the demo's controls, with an optional copy bias underneath.

``arrange`` is the demo's ``continue_nudged`` with its 'new' concept loop (the loop the demo ships), ported
unchanged in behaviour, plus one step: when ``copy`` is given, its 15 head-mask biases are added to R2's masked
action log-probabilities right after R2's step, before any control acts (the ``head_mask_bias`` contract of
``r2.sampling.continue_chart``). Every control then acts on R2 + copy, and the closed loops re-target on that base.

Per decision, in order: R2's action distribution; the copy bias; the tap-only mask and the band's playability
masks (fallback to a legal single head on the oldest lane when nothing survives); the kappa level force on
(held share, heads per row, all-busy share) over the last 64 rows against ``target``; inside a stream or jack
scope, the chord tilt and the concept tilt, both class-preserving; then the Gumbel maximum, the orientation and
the gap releases, as R2's own sampler draws them.
"""
from __future__ import annotations

import numpy as np
import torch

from ..r2.common import ACTIONS
from ..r2.features import Interval, gap_release_lanes, lane_query, make_factor, row_placements
from ..r2.locality import reads
from ..r2.properties import row_range
from ..r2.sampling import IncrementalState, _gumbel_argmax, _history_token
from ..r2.state import action_support_mask
from ..r2.strain import action_head_masks
from .controls import (CHANCE, CLASSES, FEATURES, SUBTYPES, Playability, chord_feature, fallback_action, log_normalize,
                       logit, scope_statistics, tilt, transition_features)

BIT = 1 << np.arange(4)
CANDIDATES = []
for _pattern in range(16):
    _held = np.array([(_pattern >> lane) & 1 for lane in range(4)], bool)
    _heads = np.where(_held, ACTIONS >= 3, (ACTIONS == 1) | (ACTIONS == 2))
    _ln = np.where(_held, ACTIONS == 4, ACTIONS == 2)
    CANDIDATES.append(np.column_stack((_ln.sum(1), _heads.sum(1), (_held | _heads).all(1))).astype(float))


def candidate_bias(held, running, target, kappa):
    pattern = int(np.asarray(held, dtype=int) @ np.array([1, 2, 4, 8]))
    return -(CANDIDATES[pattern] @ (np.asarray(kappa) * (np.asarray(running) - np.asarray(target))))


def mask_log_probs(logp, held):
    """[15] log masses of head masks 1..15 from [625] action log-probabilities (numpy, -inf allowed)."""
    ids = action_head_masks(held)
    out = np.full(15, -np.inf)
    for m in range(1, 16):
        v = logp[ids == m]
        v = v[np.isfinite(v)]
        if len(v):
            out[m - 1] = np.logaddexp.reduce(v)
    return out


@torch.no_grad()
def arrange(model, head_ms, song_ms, grid, *, seed, target, kappa, stop=None, ln_fraction=1., requests=(),
            masks=None, copy=None):
    """Full BOS sampling; target order held, heads per row, all-busy share. Masks are one band's config.

    Resolved requests contain concept, a_ms, b_ms, target, q, band; optional w selects open loop. Callers
    validate external arguments. ``copy(k, hist, lp15)`` returns 15 head-mask biases from the committed head
    masks ``hist[:k]`` and R2's log mask marginal ``lp15`` at row k (``copy.CopyBias``).
    Returns (actions, gap, trace)."""
    target = np.asarray(target, dtype=float)
    kappa = np.broadcast_to(np.asarray(kappa, dtype=float), (3,))
    K = len(head_ms)
    stop = K + 1 if stop is None else stop
    model.eval()
    actions = np.zeros((K + 1, 4), np.int64)
    gap = np.full((K + 1, 4), np.nan)
    state = IncrementalState(head_ms, song_ms, grid, actions, gap)
    generator = torch.Generator().manual_seed(int(seed))
    temporal = model.temporal
    cache = temporal.empty_cache()
    marks = []
    busy = np.zeros(K + 2)
    hist = np.zeros(K + 1, np.int64)
    action_nll = np.zeros(stop)
    bias_abs = np.zeros(stop)
    copy_abs = np.zeros(stop)
    stream_rows = np.zeros(K, bool)
    scopes = []
    for r in requests:
        lo, hi = row_range(head_ms, r['a_ms'], r['b_ms'], song_ms)
        if r['concept'] == 'stream' or 'chord_target' in r:
            stream_rows[lo:hi] = True
        scopes.append(dict(request=r, lo=lo, hi=hi, N=max(0, hi - max(1, lo)), n=0, success=0, chance=0.,
                           probability=0., visible=0, readout=0., expected=0., expected_chance=0.,
                           chord_N=hi - lo, chord_n=0, chord_success=0, chord_probability=0.))
    guard = Playability(masks) if masks is not None else None
    kl = np.zeros(stop)
    gain = np.full(stop, np.nan)
    request_id = np.full(stop, -1, int)
    marginal_error = np.zeros(stop)
    attainable = np.full(stop, np.nan)
    chord_gain = np.full(stop, np.nan)
    chord_marginal_error = np.zeros(stop)
    mask_firings = np.zeros((stop, 3), int)
    mask_mass = np.zeros((stop, 3))
    heads_force_active = np.zeros(stop, bool)
    heads_window_rows = np.zeros(stop, int)
    heads_force_running = np.full(stop, np.nan)
    base_expect = np.full(stop, np.nan)
    fallback_lane = np.full(stop, -1, int)
    fallback_elapsed_ms = np.full(stop, np.nan)
    fallback_floor_ms = np.full(stop, np.nan)
    fallbacks = []
    for k in range(stop):
        chart = state.chart
        d = chart.derived()
        held = d.held[k]
        before = temporal.read(cache).to(model.dtype)[None]
        mk = torch.stack(marks) if marks else None
        vis = torch.ones(1, len(marks), dtype=torch.bool, device=model.device) if marks else None
        z = model.hands(before, model.query_features(chart, [k]), mk, vis, model.row_condition(chart, (), [k]))
        plain = model.action_log_probs(z, action_support_mask(held, k == K)[None])[0]
        if copy is not None and k < K:
            p64 = plain.detach().cpu().to(torch.float64).numpy()
            b15 = np.asarray(copy(k, hist, mask_log_probs(p64, held)), dtype=np.float64)
            copy_abs[k] = float(np.abs(b15).max())
            plain = torch.from_numpy(log_normalize(p64 + np.r_[0., b15][action_head_masks(held)]))
        logp = plain
        fallback = None
        pattern = int(np.asarray(held, dtype=int) @ np.array([1, 2, 4, 8]))
        heads, ln, features = FEATURES[pattern]
        if k < K and (ln_fraction == 0 or guard is not None):
            logp = plain.detach().cpu().to(torch.float64).numpy().copy()
            if ln_fraction == 0:
                logp[ln.any(1)] = -np.inf
            legal_logp = logp.copy()
            if guard is not None:
                rejected = guard.rejections(chart, k, held)
                for j, bad in enumerate(rejected.values()):
                    mask_firings[k, j] = bool(np.any(bad & np.isfinite(logp)))
                    mask_mass[k, j] = float(np.exp(log_normalize(logp)[bad]).sum())
                logp[np.logical_or.reduce(list(rejected.values()))] = -np.inf
            if guard is not None and not np.isfinite(logp).any():
                elapsed = head_ms[k] - d.last_attack[k]
                fallback = fallback_action(legal_logp, heads, elapsed)
                lane = int(heads[fallback].argmax())
                fallback_lane[k] = lane
                fallback_elapsed_ms[k] = elapsed[lane]
                fallback_floor_ms[k] = masks['reattack_floor_ms']
                fallbacks.append(dict(row=k, time_ms=float(head_ms[k]), lane=lane, elapsed_ms=float(elapsed[lane]),
                                      floor_ms=float(masks['reattack_floor_ms']), action_index=fallback,
                                      action=ACTIONS[fallback].tolist(), held_on_entry=held.tolist(),
                                      selected_lane_held=bool(held[lane]),
                                      violated_masks=[name for name, bad in rejected.items() if bad[fallback]]))
                logp[fallback] = 0.
            logp = torch.from_numpy(log_normalize(logp))
        if np.any(kappa) and k < K and k:
            start = max(0, k - 64)
            n = k - start
            running = np.array([(d.cum_held[k] - d.cum_held[start]) / (4 * n),
                                (d.cum_heads[k] - d.cum_heads[start]) / n, (busy[k] - busy[start]) / n])
            gains = kappa.copy()
            include = ~stream_rows[start:k]
            heads_window_rows[k] = int(include.sum())
            if stream_rows[k] or not include.any():
                gains[1] = 0
            elif stream_rows[start:k].any():
                running[1] = d.attack[start:k][include].sum(1).mean()
            heads_force_running[k] = running[1]
            heads_force_active[k] = gains[1] != 0
            bias = candidate_bias(held, running, target, gains)
            logp = logp.detach().cpu().to(torch.float64) + torch.from_numpy(bias)
            bias_abs[k] = np.max(np.abs(bias[np.isfinite(plain.cpu().numpy())]))
        active = None
        feature = None
        allowed = None
        if k < K:
            for i, s in enumerate(scopes):
                if not s['lo'] <= k < s['hi']:
                    continue
                r = s['request']
                request_id[k] = i
                visible = reads(chart, Interval(0, r['a_ms'], r['b_ms'], 0.), k, held)
                base = log_normalize(logp.detach().cpu().to(torch.float64).numpy())
                if 'chord_target' in r:
                    cf = chord_feature(heads.sum(1), r['subtype'])
                    s['chord_probability'] += float(np.exp(base)[cf].sum())
                    if visible:
                        needed = (s['chord_N'] * r['chord_target'] - s['chord_success']) / (s['chord_N'] - s['chord_n'])
                        wc = float(np.clip(logit(needed) - logit(s['chord_probability'] / (s['chord_n'] + 1)), -8, 8))
                        chord_gain[k] = wc
                        changed, chord_marginal_error[k] = tilt(base, cf, CLASSES[pattern, 'chord'], wc)
                        if wc:
                            base = changed
                            logp = torch.from_numpy(base)
                if k:
                    eligible = heads & ~held
                    jf, sf = transition_features(eligible, d.attack[k - 1], d.attack[k - 2] if k >= 2 else None)
                    feature = (jf if r['concept'] == 'jack' else sf) & eligible.any(1)
                    allowed = np.isfinite(base)
                    jr, sr = transition_features(heads, d.attack[k - 1], d.attack[k - 2] if k >= 2 else None)
                    chance = CHANCE[d.attack[k - 1].sum(), heads.sum(1)]
                    readout = jr - chance if r['concept'] == 'jack' else sr.astype(float)
                    mass = np.exp(base)
                    expected = float(mass @ readout)
                    expected_chance = float(mass @ chance)
                    base_expect[k] = expected
                    if visible and not r.get('chord_only', False):
                        n = s['n']
                        if 'w' in r:
                            w = r['w']
                        elif n == 0:
                            w = 0.
                        else:
                            offset = s['expected_chance'] / n if r['concept'] == 'jack' else 0.
                            needed = (s['N'] * r['target'] - s['readout']) / (s['N'] - n)
                            w = float(np.clip(logit(needed + offset) - logit(s['expected'] / n + offset), -8, 8))
                        gain[k] = w
                        group = 'stream_subtype' if r['concept'] == 'stream' and r.get('subtype') else r['concept']
                        changed, marginal_error[k] = tilt(base, feature, CLASSES[pattern, group], w)
                        if w:
                            logp = torch.from_numpy(changed)
                        s['visible'] += 1
                    s['expected'] += expected
                    s['expected_chance'] += expected_chance
                active = s
                break
        normalized = log_normalize(logp.detach().cpu().to(torch.float64).numpy())
        finite = np.isfinite(normalized)
        base_plain = plain.detach().cpu().to(torch.float64).numpy()
        kl[k] = float(np.exp(normalized[finite]) @ (normalized[finite] - base_plain[finite]))
        chosen = _gumbel_argmax(logp, generator) if fallback is None else fallback
        codes = ACTIONS[chosen]
        action_nll[k] = -float(plain[chosen])
        orientation = int(torch.randint(2, (), generator=generator))
        lanes = gap_release_lanes(held, codes, orientation)
        rel = np.full(4, np.nan)
        if lanes:
            placed = row_placements(chart, k, held, codes)
            lq = lane_query(chart, [k]).astype(np.float32)[0]
            for lane in lanes:
                factor = make_factor(chart, k, 0, codes, lane, orientation, dict(placed), lq)
                ctx = model.factor_contexts(z, [factor])
                scores, _, _ = model.all_pair_scores(ctx, [factor], chart, ())
                u = float(factor.times[_gumbel_argmax(scores, generator)])
                placed[lane] = (u, False)
                rel[lane] = u
        state.append(codes, rel)
        busy[k + 1] = busy[k] + bool((held | chart.derived().attack[k]).all())
        if k < K:
            actual = chart.derived().attack[k]
            hist[k] = int(actual.astype(np.int64) @ BIT)
            if guard is not None:
                guard.observe(actual)
            if active is not None:
                if k:
                    j, s = transition_features(actual, d.attack[k - 1], d.attack[k - 2] if k >= 2 else None)
                    active['success'] += int(j if active['request']['concept'] == 'jack' else s)
                    chance_k = CHANCE[d.attack[k - 1].sum(), actual.sum()]
                    active['chance'] += chance_k
                    active['readout'] += (float(j) - chance_k) if active['request']['concept'] == 'jack' else float(s)
                    active['n'] += 1
                    possible = allowed & feature
                    if active['request']['concept'] == 'stream':
                        possible &= heads.sum(1) == actual.sum()
                    attainable[k] = bool(possible.any())
                if 'chord_target' in active['request']:
                    active['chord_success'] += int(chord_feature(actual.sum(), active['request']['subtype']))
                    active['chord_n'] += 1
        if k < K:
            cache = temporal.append(cache, model._t(_history_token(chart, k)))
            if k % model.config.stride == 0:
                marks.append(temporal.read(cache))
    receipts = []
    for i, s in enumerate(scopes):
        r = s['request']
        stats = scope_statistics(head_ms, state.chart.derived().attack[:min(stop, K)], r['a_ms'], r['b_ms'], song_ms)
        selected = request_id == i
        visible = selected & np.isfinite(gain)
        achieved = stats['S' if r['concept'] == 'stream' else 'J_ex']
        receipts.append(dict(r, **stats, achieved=achieved, loop='new',
                             in_band=(r['band'][0] <= r['target'] <= r['band'][1] and r.get('preset_in_band', True))
                             if 'band' in r else None,
                             target_error=achieved - r['target'] if achieved is not None else None,
                             attainable_maximum=float(np.nanmean(attainable[selected])) if selected.any() else None,
                             attainable_statistic='raw_J' if r['concept'] == 'jack' else 'raw_S',
                             bound_share=float((abs(gain[visible]) >= 8).mean()) if visible.any() else None,
                             visible_decisions=int(visible.sum()), mask_firings=mask_firings[selected].sum(0).tolist(),
                             fallback_count=sum(s['lo'] <= f['row'] < s['hi'] for f in fallbacks),
                             fallbacks=[f for f in fallbacks if s['lo'] <= f['row'] < s['hi']],
                             kl_per_decision=float(kl[selected].mean()) if selected.any() else None,
                             max_class_marginal_error=float(marginal_error[selected].max()) if selected.any() else None))
        if 'chord_target' in r:
            cv = selected & np.isfinite(chord_gain)
            chord_achieved = stats[SUBTYPES[r['subtype']]['share']]
            receipts[-1].update(
                chord_achieved=chord_achieved,
                chord_error=chord_achieved - r['chord_target'] if chord_achieved is not None else None,
                chord_in_band=r['chord_band'][0] <= r['chord_target'] <= r['chord_band'][1] and r.get('preset_in_band', True),
                chord_bound_share=float((abs(chord_gain[cv]) >= 8).mean()) if cv.any() else None,
                both_bound_share=float(((abs(chord_gain[selected]) >= 8) & (abs(gain[selected]) >= 8)).mean())
                if selected.any() else None,
                chord_visible_decisions=int(cv.sum()), in_scope_decisions=int(selected.sum()),
                max_chord_class_marginal_error=float(chord_marginal_error[selected].max()) if selected.any() else None)
        receipts[-1]['stream_statistic'] = 'S_prime' if r['concept'] == 'stream' else None
        receipts[-1]['band_status'] = (('in band' if receipts[-1]['in_band'] else 'out of band, playability not promised')
                                       if 'band' in r else 'not resolved')
    return actions[:stop], gap[:stop], dict(
        action_nll=action_nll, bias_abs=bias_abs, copy_abs=copy_abs, kl=kl, gain=gain, chord_gain=chord_gain,
        request_id=request_id, marginal_error=marginal_error, chord_marginal_error=chord_marginal_error,
        attainable=attainable, mask_firings=mask_firings, mask_mass=mask_mass, heads_force_active=heads_force_active,
        heads_window_rows=heads_window_rows, heads_force_running=heads_force_running, base_expect=base_expect,
        fallback_lane=fallback_lane, fallback_elapsed_ms=fallback_elapsed_ms, fallback_floor_ms=fallback_floor_ms,
        fallback_count=len(fallbacks), fallbacks=fallbacks, requests=receipts)
