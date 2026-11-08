"""Chronological release hazards on R2's snapped chart space.

Each candidate time offers keep/release on active lanes before the next head
allocation. A gap marginalizes two fixed lane orders. The temporal network
still commits one completed head row, with the same 511-row context as R2;
within-gap state is exact. Pointer scores become Bernoulli logits, rather than
a categorical distribution over future times. Head likelihood is conditional
on the releases already chosen.

The scheduler uses cand-v1 and grammar completion masks. Hiding future numeric
features does not make that scheduler causal: candidate times and the last
legal opportunity depend on the next head. ``blind`` is only feature blindness.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F

from .candidates import GapCandidates, SNAP_ONE_HOT, phase
from .common import ACTIONS, HAND_LANES, action_index, psi_pair
from .features import (DENSITY_BEATS, FRAME_DIM, LOOKAHEAD, Factor,
                       placement_relations, query_features)
from .model import R2Model, WindowOut
from .state import action_support_mask


def point_candidates(a, b, times, grid, *, blind=False):
    """Candidate features at given clock times, including the row endpoint.

    Blind features cannot use b: future-relative views, gap fraction and the
    three next-head-dependent anchor flags are zero. The caller owns support.
    """
    times = np.asarray(times, dtype=float)
    beats = grid.beat(times)
    local, seg = grid.local_beat(times)
    lengths = grid.lengths[seg]
    snap = np.zeros((len(times), len(SNAP_ONE_HOT)))
    found = np.zeros(len(times), bool)
    for i, den in enumerate(SNAP_ONE_HOT):
        hit = ~found & (abs(local * den - np.round(local * den)) * lengths / den <= 1.)
        snap[hit, i] = 1
        found |= hit
    rounded = lambda x: np.floor(x + .5)
    grid_flag = np.zeros(len(times), bool)
    for s, (offset, length) in enumerate(zip(grid.offsets, grid.lengths)):
        lo = -np.inf if s == 0 else offset
        hi = np.inf if s + 1 == len(grid.offsets) else grid.offsets[s + 1]
        for den in (32, 24):
            position = offset + np.round((times - offset) * den / length) * length / den
            grid_flag |= (rounded(position) == times) & (position >= lo) & (position < hi)
    flags = np.zeros((len(times), 4))
    flags[:, 3] = grid_flag
    future = np.zeros((len(times), 4))
    fraction = np.zeros((len(times), 1))
    if not blind:
        bb = float(grid.beat(b))
        future = psi_pair(times - b, beats - bb)
        fraction[:, 0] = (times - a) / (b - a)
        for j, v in enumerate((rounded((a + b) / 2), rounded(grid.time_of_beat(bb - .125)),
                                rounded(grid.time_of_beat(bb - .25)))):
            flags[:, j] = (times == v) & (a < v < b)
    static = np.concatenate((future, psi_pair(times - a, beats - grid.beat(a)), fraction,
                             flags, snap, (grid.bpm(times) / 120.)[:, None], phase(beats)), -1)
    return GapCandidates(a, b if not blind else math.nan, times, flags, static, beats)


def lane_state_features(times, grid, held, start, last_attack, last_release):
    times = np.asarray(times)[:, None]
    beats = grid.beat(times)
    def age(values):
        ok = np.isfinite(values)
        safe = np.where(ok, values, 0.)
        views = psi_pair(times - safe, beats - grid.beat(safe)) * ok[..., None]
        return views, ok
    ln, _ = age(start)
    att, aok = age(last_attack)
    rel, rok = age(last_release)
    return np.concatenate((held[..., None], ln, att, rel, (~aok)[..., None], (~rok)[..., None]), -1)


def release_queries(times, grid, lane_features, previous_head, song_ms, past_heads,
                    future_heads=None):
    """[N,2,150] release queries; no future argument is needed for blind views.

    Total row count, next-row index fraction, future density and lookahead are
    absent when future_heads is None, as are song-end distance and progress.
    """
    times = np.asarray(times)
    beats = grid.beat(times)
    gap = psi_pair(times - previous_head, beats - grid.beat(previous_head))
    remaining = (np.zeros((len(times), 4)) if future_heads is None else
                 psi_pair(song_ms - times, grid.beat(song_ms) - beats))
    progress = np.zeros(len(times)) if future_heads is None else times / song_ms
    # Row-kind channel: -1 denotes release, while natural queries use 0/1 EOS.
    shared = np.column_stack((gap, remaining, progress, np.zeros(len(times)),
                              np.zeros(len(times)), -np.ones(len(times)), phase(beats),
                              grid.bpm(times) / 120., grid.meter(times) / 4.))
    dens = np.zeros((len(times), len(DENSITY_BEATS)))
    ahead = np.zeros((len(times), LOOKAHEAD * 4))
    if future_heads is not None:
        future_heads = np.asarray(future_heads)
        fb = grid.beat(future_heads)
        next_time = future_heads[0] if len(future_heads) else song_ms
        next_beat = float(grid.beat(next_time))
        # Keep R2's complete known-row gap and sixteen subsequent row gaps.
        # Current candidate age/clock still enters the lane and pointer views.
        shared[:, :4] = psi_pair(next_time - previous_head, next_beat - grid.beat(previous_head))
        for j, width in enumerate(DENSITY_BEATS):
            dens[:, j] = np.log1p(max(0, np.searchsorted(fb, next_beat + width, side='right') - 1))
        for j in range(min(LOOKAHEAD, len(future_heads) - 1)):
            ahead[:, 4 * j:4 * (j + 1)] = psi_pair(future_heads[j + 1] - future_heads[j], fb[j + 1] - fb[j])
        total = len(past_heads) + len(future_heads)
        shared[:, 9] = len(past_heads) / max(total, 1)
    shared = np.concatenate((shared, dens, ahead), -1)
    return np.stack([np.concatenate((lane_features[:, order].reshape(len(times), 60), shared), -1)
                     for order in HAND_LANES], 1).astype(np.float32)


def head_support(held, released, on_row, eos=False):
    """Exactly the R2 actions compatible with already committed releases."""
    support = action_support_mask(held, eos).copy()
    for lane in range(4):
        if held[lane]:
            if on_row[lane]:
                support &= ACTIONS[:, lane] == 1
            elif released[lane]:
                support &= ACTIONS[:, lane] >= 2
            else:
                support &= ACTIONS[:, lane] == 0
    if not support.any():
        raise ValueError('Release decisions leave no legal head allocation')
    return support


def release_forced(active, last_gap_candidate, eos, remaining_lanes):
    """Completion rules expose scheduler information even with blind features."""
    return bool(last_gap_candidate and (eos or (np.all(active) and remaining_lanes == 1)))


def after_releases(chart, k, released_times):
    d = chart.derived()
    held, starts = d.held[k].copy(), d.start[k].copy()
    last = d.last_release[k].copy()
    released = np.isfinite(released_times)
    held[released], starts[released], last[released] = False, np.nan, released_times[released]
    lanes = lane_state_features([chart.time(k)], chart.grid, held[None], starts[None],
                                d.last_attack[k:k + 1], last[None])
    q = query_features(chart, [k])
    for hand, order in enumerate(HAND_LANES):
        q[:, hand, :60] = lanes[:, order].reshape(1, 60)
    return q


def step_factors(chart, k, u, active, placed, lane_features, point, owner, orientation,
                 *, last_gap_candidate=False, eos=False, target_release=None):
    """One time slice, with gold earlier lane outcomes when target_release is set."""
    active = active.copy()
    placed = dict(placed)
    factors, forced, targets = [], [], []
    order = [l for l in (range(3, -1, -1) if orientation else range(4)) if active[l]]
    for i, lane in enumerate(order):
        force = release_forced(active, last_gap_candidate, eos, len(order) - i)
        factor = make_step_factor(chart, k, lane, point, lane_features, placed, owner, orientation)
        hit = target_release is not None and target_release[lane] == u
        if force and target_release is not None and not hit:
            raise ValueError('Source chart violates release completion support')
        factors.append(factor); forced.append(force); targets.append(hit)
        if hit:
            placed[lane] = (u, u == chart.time(k))
            active[lane] = False
    return factors, forced, targets


def make_step_factor(chart, k, lane, point, lane_features, placed, owner, orientation):
    d = chart.derived()
    codes = np.array([2 if l in placed else 0 for l in range(4)])
    if orientation:
        codes = codes[::-1].copy()
    return Factor(owner, orientation, codes, 3 - lane if orientation else lane, lane,
        lane_features[lane], point.lane_features(d.start[k, lane], chart.grid.beat(d.start[k, lane])),
        placement_relations(point, chart.grid, orientation, placed, d.birth[k], lane).astype(np.float32),
        point.times, 0, k)


class ReleaseStepModel(R2Model):
    """Same parameters as R2; the likelihood order and normalization differ."""
    def __init__(self, config, *, blind=False, verbose=False):
        super().__init__(config, verbose=verbose)
        if config.memory != 'none':
            raise ValueError('Release-step comparison uses the phase-N memory=none recipe')
        self.blind = blind

    def before_rows(self, chart, ks):
        enc = self.encode_history(chart, min(int(max(ks)), chart.K))
        bos = self.temporal.boundary[0].to(self.dtype).expand(2, self.config.hidden)
        if enc is None:
            return bos.expand(len(ks), 2, self.config.hidden)
        idx = torch.as_tensor(np.maximum(np.asarray(ks) - 1, 0), device=self.device)
        first = torch.as_tensor(np.asarray(ks) == 0, device=self.device)[:, None, None]
        return torch.where(first, bos, enc.index_select(0, idx))

    def step_hands(self, before, query):
        cond = before.new_zeros(len(before), self.config.roles * 2 * FRAME_DIM)
        return self.hands(before, self._t(query), None, None, cond)

    def window(self, chart, start, stop, track=(), **kwargs):
        if track or any(kwargs.get(k) is not None for k in ('ln_level', 'ln_length', 'theta')):
            raise ValueError('Release-step comparison fits only the natural recipe')
        if kwargs.get('pairs') or kwargs.get('history_start', 0):
            raise ValueError('Release-step likelihood requires full history and does not expose R2 pointer pairs')
        if not 0 <= start < stop <= min(chart.n, chart.K + 1):
            raise ValueError('Window outside known chart')
        ks = np.arange(start, stop)
        before = self.before_rows(chart, ks)
        d = chart.derived()
        queries, query_rows, packed = [], [], []
        head_queries, head_masks, head_targets = [], [], []
        query_offset = 0
        for row, k in enumerate(ks):
            rel = d.release[k]
            if k and d.held[k].any():
                cand = chart.candidates(k)
                times = cand.times if k == chart.K else np.r_[cand.times, chart.time(k)]
                relevant = np.where(np.isfinite(rel), rel, chart.time(k))
                times = times[times <= np.max(relevant[d.held[k]])]
                earlier = np.isfinite(rel)[None] & (rel[None] < times[:, None])
                active = d.held[k][None] & ~earlier
                starts = np.broadcast_to(d.start[k], active.shape).copy()
                last = np.broadcast_to(d.last_release[k], active.shape).copy()
                starts[earlier] = np.nan
                last = np.where(earlier, rel[None], last)
                lanes = lane_state_features(times, chart.grid, active, starts,
                    np.broadcast_to(d.last_attack[k], active.shape), last)
                queries.append(release_queries(times, chart.grid, lanes, chart.head_ms[k - 1], chart.song_ms,
                    chart.head_ms[:k], None if self.blind else chart.head_ms[k:]))
                query_rows.extend([row] * len(times))
                point = point_candidates(chart.head_ms[k - 1], chart.time(k), times, chart.grid, blind=self.blind)
                for orientation in (0, 1):
                    order = np.arange(3, -1, -1) if orientation else np.arange(4)
                    for at, lane in enumerate(order):
                        idx = np.flatnonzero(active[:, lane])
                        if not len(idx):
                            continue
                        u = times[idx]
                        simultaneous = (rel[None] == u[:, None]) & np.isin(np.arange(4), order[:at])[None]
                        visible = earlier[idx] | simultaneous
                        codes = np.where(visible[:, order], 2, 0)
                        relations = np.zeros((len(idx), 4, 7))
                        for slot, other in enumerate(order):
                            if not np.isfinite(rel[other]):
                                continue
                            ok = visible[:, other]
                            relations[:, slot, :4] = psi_pair(u - rel[other], chart.grid.beat(u) -
                                chart.grid.beat(rel[other])) * ok[:, None]
                            relations[:, slot, 4] = ok & (d.birth[k, other] == d.birth[k, lane])
                            relations[:, slot, 5] = ok
                            relations[:, slot, 6] = ok & (rel[other] == chart.time(k))
                        force = (u == cand.times[-1]) & (k == chart.K or
                            (at == 3 and bool(d.held[k].all())))
                        if k != chart.K:
                            force &= ~visible.any(1)
                        target = rel[lane] == u
                        if np.any(force & ~target):
                            raise ValueError('Source chart violates release completion support')
                        packed.append(dict(owner=idx + query_offset, orientation=np.full(len(idx), orientation),
                            codes=codes, slot=np.full(len(idx), at), lanes=lanes[idx, lane],
                            candidate=point.lane_features(d.start[k, lane], chart.grid.beat(d.start[k, lane]))[idx],
                            relations=relations.reshape(len(idx), 28), force=force, target=target,
                            row=np.full(len(idx), row * 2 + orientation)))
                query_offset += len(times)
            released = np.isfinite(rel)
            head_queries.append(after_releases(chart, k, rel)[0])
            head_masks.append(head_support(d.held[k], released, d.row_release[k], k == chart.K))
            head_targets.append(action_index(chart.actions[k]))
        z_head = self.step_hands(before, np.stack(head_queries))
        logp = self.action_log_probs(z_head, np.stack(head_masks))
        action = logp[torch.arange(len(ks), device=self.device), torch.as_tensor(head_targets, device=self.device)]
        release = action.new_zeros(len(ks))
        pair_count = 0
        if packed:
            pack = {key: np.concatenate([p[key] for p in packed]) for key in packed[0]}
            q_before = before[torch.as_tensor(query_rows, device=self.device)]
            z = self.step_hands(q_before, np.concatenate(queries))
            zz = z[torch.as_tensor(pack['owner'], device=self.device)]
            flip = torch.as_tensor(pack['orientation'] == 1, device=self.device)[:, None]
            left, right = torch.where(flip, zz[:, 1], zz[:, 0]), torch.where(flip, zz[:, 0], zz[:, 1])
            codes = torch.as_tensor(pack['codes'], device=self.device)
            slots = self._t(np.eye(4)[pack['slot']])
            base = self.pointer_in(torch.cat((left, right, self.code_embedding(codes).flatten(1),
                                               slots, self._t(pack['lanes'])), -1))
            pair_count = len(base)
            relation, candidate = self._t(pack['relations']), self._t(pack['candidate'])
            scores = []
            for lo in range(0, pair_count, self.config.candidate_budget):
                hi = min(pair_count, lo + self.config.candidate_budget)
                idx = torch.arange(lo, hi, device=self.device)
                if torch.is_grad_enabled() and pair_count > self.config.candidate_budget:
                    from torch.utils.checkpoint import checkpoint
                    value = checkpoint(self.pair_scores, base, idx, relation[lo:hi], candidate[lo:hi],
                                       None, use_reentrant=False)
                else:
                    value = self.pair_scores(base, idx, relation[lo:hi], candidate[lo:hi], None)
                scores.append(value)
            signs = self._t(np.where(pack['target'], 1., -1.))
            terms = F.logsigmoid(torch.cat(scores) * signs)
            terms = torch.where(torch.as_tensor(pack['force'], device=self.device), terms.new_zeros(()), terms)
            sums = release.new_zeros(2 * len(ks)).index_add(
                0, torch.as_tensor(pack['row'], device=self.device), terms).reshape(len(ks), 2)
            release = torch.logsumexp(sums, -1) - math.log(2)
        gap_lns = np.sum(np.isfinite(chart.gap[ks]), 1)
        return WindowOut(action, release, ks, ks == chart.K, gap_lns,
                         action.new_zeros(len(ks)), np.zeros((len(ks), 2), bool), logp, pair_count)



@torch.no_grad()
def continue_release_steps(model, head_ms, song_ms, grid, *, seed, target=(0., 1., 0.), kappa=0.,
                           masks=None, prefix_actions=None, prefix_gap=None):
    """Sample the same release-step factors used by window(), closing every hold.

    Controls act on head allocation after the release steps. When playability
    rules eliminate every compatible head action, fall back within that exact
    grammar support. The trace separates forced release outcomes from sampled
    hazards, and records ages/probabilities for own-history calibration.
    """
    from .sampling import IncrementalState, _gumbel_argmax, _history_token
    from ..system.arrange import candidate_bias
    from ..system.controls import Playability, fallback_action, log_normalize
    from .strain import heads_from_codes
    model.eval()
    K = len(head_ms)
    actions = np.zeros((K + 1, 4), int)
    gap = np.full((K + 1, 4), np.nan)
    state = IncrementalState(head_ms, song_ms, grid, actions, gap)
    cache = model.temporal.empty_cache()
    rng = torch.Generator().manual_seed(int(seed))
    guard = Playability(masks) if masks is not None else None
    busy = np.zeros(K + 2)
    gains = np.broadcast_to(kappa, (3,))
    if (prefix_actions is None) != (prefix_gap is None):
        raise ValueError('Prefix actions and gaps must be supplied together')
    prefix = 0 if prefix_actions is None else len(prefix_actions)
    if prefix > K or (prefix and (np.shape(prefix_actions) != (prefix, 4) or np.shape(prefix_gap) != (prefix, 4))):
        raise ValueError('Prefix must have matching [P,4] arrays without EOS')
    age_edges = np.array([60., 125., 250., 500., 1000., 2000.])
    # opportunities, releases, expected releases, forced opportunities
    hazards = np.zeros((len(age_edges) + 1, 4))
    fallback_count, forced_count = 0, 0
    for k in range(K + 1):
        chart = state.chart
        d = chart.derived()
        held = d.held[k].copy()
        if k < prefix:
            state.append(prefix_actions[k], prefix_gap[k])
        else:
            before = model.temporal.read(cache).to(model.dtype)[None]
            rel = np.full(4, np.nan)
            if held.any():
                cand = chart.candidates(k)
                times = cand.times if k == K else np.r_[cand.times, chart.time(k)]
                orientation = int(torch.randint(2, (), generator=rng))
                active = held.copy()
                placed = {}
                for u in times:
                    if not active.any():
                        break
                    starts, last = d.start[k].copy(), d.last_release[k].copy()
                    done = np.isfinite(rel)
                    starts[done], last[done] = np.nan, rel[done]
                    lanes = lane_state_features([u], grid, active[None], starts[None],
                                                d.last_attack[k:k + 1], last[None])[0]
                    q = release_queries([u], grid, lanes[None], head_ms[k - 1], song_ms,
                                         head_ms[:k], None if model.blind else head_ms[k:])
                    z = model.step_hands(before, q)
                    point = point_candidates(head_ms[k - 1], chart.time(k), [u], grid, blind=model.blind)
                    order = [l for l in (range(3, -1, -1) if orientation else range(4)) if active[l]]
                    for i, lane in enumerate(order):
                        force = release_forced(active, u == cand.times[-1], k == K, len(order) - i)
                        factor = make_step_factor(chart, k, lane, point, lanes, placed, 0, orientation)
                        score, _, _ = model.all_pair_scores(model.factor_contexts(z, [factor]), [factor], chart, ())
                        probability = 1. if force else float(score.sigmoid()[0])
                        release = bool(force or torch.rand((), generator=rng) < probability)
                        age = u - d.start[k, lane]
                        age_bin = np.searchsorted(age_edges, age, side='right')
                        hazards[age_bin] += (1, release, probability, force)
                        forced_count += force
                        if release:
                            rel[lane] = u
                            active[lane] = False
                            placed[lane] = (u, u == chart.time(k))
            row_release = np.isfinite(rel) & (rel == chart.time(k))
            support = head_support(held, np.isfinite(rel), row_release, k == K)
            z = model.step_hands(before, after_releases(chart, k, rel))
            lp = model.action_log_probs(z, support[None])[0].cpu().double().numpy()
            fallback = None
            if k < K and guard is not None:
                old = lp.copy()
                rejected = guard.rejections(chart, k, held)
                lp[np.logical_or.reduce(list(rejected.values()))] = -np.inf
                if not np.isfinite(lp).any():
                    heads = heads_from_codes(ACTIONS, held)
                    elapsed = head_ms[k] - d.last_attack[k]
                    fallback = fallback_action(old, heads, elapsed)
                    lp[fallback] = 0.
                    fallback_count += 1
                lp = log_normalize(lp)
            if k and k < K and np.any(gains):
                lo = max(0, k - 64)
                n = k - lo
                running = np.array([(d.cum_held[k] - d.cum_held[lo]) / (4 * n),
                                     (d.cum_heads[k] - d.cum_heads[lo]) / n,
                                     (busy[k] - busy[lo]) / n])
                lp += candidate_bias(held, running, target, gains)
            chosen = fallback if fallback is not None else _gumbel_argmax(torch.from_numpy(lp), rng)
            releases = np.where(row_release, np.nan, rel)
            state.append(ACTIONS[chosen], releases)
        actual = chart.derived().attack[k]
        busy[k + 1] = busy[k] + bool((held | actual).all())
        if k < K:
            if guard is not None:
                guard.observe(actual)
            cache = model.temporal.append(cache, model._t(_history_token(chart, k)))
    return actions, gap, dict(fallback_count=fallback_count, forced_releases=forced_count,
        age_edges_ms=age_edges.tolist(), hazards=hazards.tolist())
