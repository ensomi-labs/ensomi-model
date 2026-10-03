"""R2 model: TCN history with landmark reads, exact query MLP, conditioner, 625-way joint head
and a flat release pointer marginalised over two mirror orientations.

log p(D_k | s_k) = log P(A_k | s_k) + logaddexp(log Q_fwd, log Q_mirrored) - log 2, where
Q_fwd scores the gap releases autoregressively in lane order 0..3 and
Q_mirrored is the same directed pointer evaluated on the fully mirrored
state/action/placements (hands swapped, lane slots reversed). The action head
is the R1 ``JointHead`` with five codes per lane, left = 5 a0 + a1 and
right = 5 a3 + a2.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..research.bounded_typed_continuation.model import JointHead
from ..research.bounded_typed_continuation.temporal import FiniteTemporal, TemporalConfig, pointwise
from .candidates import CANDIDATE_DIM
from .common import ACTIONS, ContractError, action_index
from .features import (FRAME_DIM, HISTORY_DIM, LANE_QUERY_DIM, QUERY_DIM, RELATION_DIM, TOKEN_DIM, Chart,
                       decision_factors, frames, history_tokens, lane_query, query_features, tokens)
from .state import action_support_mask

MAX_PARAMETERS = 4_500_000


@dataclass(frozen=True)
class R2Config:
    hidden: int = 128
    levels: int = 8
    expansion: int = 4
    rank: int = 16
    memory: str = 'landmarks'      # 'landmarks' | 'none'
    stride: int = 64
    conditioner: str = 'film'      # 'film' | 'tokens'
    code_dim: int = 8
    candidate_budget: int = 8192

    def __post_init__(self):
        if self.memory not in ('landmarks', 'none') or self.conditioner not in ('film', 'tokens'):
            raise ContractError('memory must be landmarks|none and conditioner film|tokens')


class JointHead5(JointHead):
    """R1 joint head over all 625 five-code lane combinations."""

    def __init__(self, hidden, rank):
        super().__init__(hidden, 5, rank)
        self.register_buffer('left', torch.tensor([int(a[0] * 5 + a[1]) for a in ACTIONS]), persistent=False)
        self.register_buffer('right', torch.tensor([int(a[3] * 5 + a[2]) for a in ACTIONS]), persistent=False)


class FiLM(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(6 * FRAME_DIM, 128), nn.GELU(), nn.Linear(128, 2 * hidden))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, z, cond):
        gamma, delta = self.mlp(cond).chunk(2, -1)
        return z + gamma * self.norm(z) + delta


class TokenConditioner(nn.Module):
    """Cross-attention from each hand query over the whole announced track plus a null token."""

    def __init__(self, hidden, heads=4):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(TOKEN_DIM, 128), nn.GELU(), nn.Linear(128, hidden))
        self.kind = nn.Embedding(2, hidden)
        self.null = nn.Parameter(torch.randn(hidden) * 0.02)
        self.norm = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.output = nn.Linear(hidden, hidden)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, z, toks):
        """z [Q,H], toks [Q,NI,18] -> [Q,H]."""
        Q, H = z.shape
        keys = self.null.to(z.dtype).expand(Q, 1, H)
        if toks.shape[1]:
            kinds = toks[..., :2].argmax(-1)
            keys = torch.cat((keys, self.encode(toks) + self.kind(kinds)), 1)
        attended, _ = self.attention(self.norm(z)[:, None], keys, keys, need_weights=False)
        return z + self.output(attended[:, 0])


@dataclass
class WindowOut:
    action: torch.Tensor     # [m] log P(A_k | s_k)
    release: torch.Tensor    # [m] log Q(U_k | s_k, A_k)
    ks: np.ndarray
    eos: np.ndarray          # [m] bool
    gap_lns: np.ndarray      # [m] number of gap releases

    @property
    def total(self):
        return self.action + self.release


class R2Model(nn.Module):
    def __init__(self, config: R2Config = R2Config(), *, verbose=False):
        super().__init__()
        self.config = c = config
        H = c.hidden
        self.temporal = FiniteTemporal(TemporalConfig(HISTORY_DIM, H, c.levels, c.expansion))
        self.exact = nn.Sequential(nn.Linear(QUERY_DIM, H), nn.GELU(), nn.Linear(H, H))
        self.fuse = nn.Sequential(nn.LayerNorm(2 * H), nn.Linear(2 * H, H), nn.GELU(), nn.Linear(H, H))
        self.lm_query, self.lm_key, self.lm_value = nn.Linear(H, H), nn.Linear(H, H), nn.Linear(H, H)
        self.lm_out = nn.Linear(H, H, bias=False)
        nn.init.zeros_(self.lm_out.weight)
        self.film = FiLM(H)
        self.tokens = TokenConditioner(H)
        self.joint = JointHead5(H, c.rank)
        self.code_embedding = nn.Embedding(5, c.code_dim)
        self.pointer_in = nn.Linear(2 * H + 4 * c.code_dim + 4 + LANE_QUERY_DIM, H)
        self.pointer_relation = nn.Linear(RELATION_DIM, H, bias=False)
        self.pointer_out = nn.Linear(H, H)
        self.candidate = nn.Sequential(nn.Linear(CANDIDATE_DIM, H), nn.GELU(), nn.Linear(H, H))
        self.candidate_bias = nn.Linear(CANDIDATE_DIM, 1)
        counts = self.parameter_counts()
        if verbose:
            print({'parameters': counts}, flush=True)
        if counts['total'] > MAX_PARAMETERS:
            raise ContractError(f'Model has {counts["total"]} parameters, above {MAX_PARAMETERS}')

    def parameter_counts(self):
        out = {name: sum(p.numel() for p in m.parameters()) for name, m in self.named_children()}
        out['total'] = sum(p.numel() for p in self.parameters())
        return out

    @property
    def device(self):
        return self.exact[0].weight.device

    @property
    def dtype(self):
        return self.exact[0].weight.dtype

    def _t(self, array):
        return torch.as_tensor(np.ascontiguousarray(array), dtype=self.dtype).to(self.device)

    # ---- hand vectors ----------------------------------------------------------------------------

    def read_landmarks(self, h, marks, visible):
        """h [m,2,H], marks [L,2,H] TCN outputs, visible [m,L] bool -> [m,2,H]."""
        if marks is None or marks.shape[0] == 0:
            return torch.zeros_like(h)
        q = self.lm_query(h)
        k, v = self.lm_key(marks), self.lm_value(marks)
        scores = torch.einsum('mhd,lhd->mhl', q, k) / math.sqrt(h.shape[-1])
        vis = visible[:, None, :]
        weights = scores.masked_fill(~vis, -1e9).softmax(-1) * vis.any(-1, keepdim=True)
        return self.lm_out(torch.einsum('mhl,lhd->mhd', weights, v))

    def condition(self, h, cond):
        """Apply the run's conditioner with the same weights to both hands."""
        if self.config.conditioner == 'film':
            return self.film(h, cond[:, None, :])
        return torch.stack([self.tokens(h[:, i], cond) for i in range(2)], 1)

    def hands(self, before, qf, marks, visible, cond):
        h = pointwise(self.fuse, torch.cat((before, pointwise(self.exact, qf)), -1))
        if self.config.memory == 'landmarks':
            h = h + self.read_landmarks(h, marks, visible)
        return self.condition(h, cond)

    def row_condition(self, chart: Chart, track, ks):
        ks = np.asarray(ks)
        if self.config.conditioner == 'film':
            out = np.zeros((len(ks), 3, 2, FRAME_DIM))
            if track:
                for i, k in enumerate(ks):
                    out[i, 0] = frames(chart, track, [chart.time(k)], int(k))[0]
            return self._t(out.reshape(len(ks), -1))
        return self._t(np.concatenate([tokens(chart, track, [chart.time(k)], int(k)) for k in ks], 0)
                       if track else np.zeros((len(ks), 0, TOKEN_DIM)))

    def encode_history(self, chart: Chart, N: int):
        if N <= 0:
            return None
        raw = self._t(history_tokens(chart, N))
        return self.temporal(raw[None])[0]

    def window_hands(self, chart: Chart, ks, track=()):
        ks = np.asarray(ks)
        stop = int(ks.max()) + 1
        N = min(stop - 1, chart.K)
        enc = self.encode_history(chart, N)
        H = self.config.hidden
        bos = self.temporal.boundary[0].to(self.dtype).expand(2, H)
        if enc is None:
            before = bos.expand(len(ks), 2, H)
            marks, visible = None, None
        else:
            idx = torch.as_tensor(np.maximum(ks - 1, 0), device=self.device)
            first = torch.as_tensor(ks == 0, device=self.device)[:, None, None]
            before = torch.where(first, bos.expand(len(ks), 2, H), enc.index_select(0, idx))
            pos = np.arange(0, N, self.config.stride)
            marks = enc[torch.as_tensor(pos, device=self.device)]
            visible = torch.as_tensor(pos[None, :] < ks[:, None], device=self.device)
        qf = self._t(query_features(chart, ks))
        return self.hands(before, qf, marks, visible, self.row_condition(chart, track, ks))

    # ---- likelihood ------------------------------------------------------------------------------

    def action_log_probs(self, z, masks):
        logits = self.joint(z)
        mask = torch.as_tensor(np.asarray(masks), device=z.device)
        return logits.masked_fill(~mask, -torch.inf).log_softmax(-1)

    def window(self, chart: Chart, start: int, stop: int, track=()) -> WindowOut:
        """Teacher-forced log-probabilities of decisions [start, stop) of ``chart``."""
        if not 0 <= start < stop <= min(chart.n, chart.K + 1):
            raise ContractError('Window outside the known decisions')
        ks = np.arange(start, stop)
        d = chart.derived()
        z = self.window_hands(chart, ks, track)
        masks = np.stack([action_support_mask(d.held[k], k == chart.K) for k in ks])
        targets = np.array([action_index(chart.actions[k]) for k in ks])
        if not masks[np.arange(len(ks)), targets].all():
            raise ContractError('A teacher-forced action is outside its support')
        logp = self.action_log_probs(z, masks)
        action = logp.gather(1, torch.as_tensor(targets, device=z.device)[:, None])[:, 0]
        lanes = lane_query(chart, ks).astype(np.float32)
        factors = []
        for i, k in enumerate(ks):
            for o in (0, 1):
                factors += decision_factors(chart, int(k), i, chart.actions[k], chart.gap[k], lanes[i], o)
        release = self.release_from_factors(z, factors, chart, track, len(ks))
        gap_lns = np.array([sum(1 for f in factors if f.owner == i and f.orientation == 0) for i in range(len(ks))])
        return WindowOut(action, release, ks, ks == chart.K, gap_lns)

    def release_from_factors(self, z, factors, chart, track, m):
        if not factors:
            return z.new_zeros(m)
        lp = self.factor_log_probs(z, factors, chart, track)
        slot = torch.as_tensor([2 * f.owner + f.orientation for f in factors], device=z.device)
        sums = z.new_zeros(2 * m).index_add(0, slot, lp).reshape(m, 2)
        has = torch.zeros(m, dtype=torch.bool, device=z.device)
        has[torch.as_tensor([f.owner for f in factors], device=z.device)] = True
        return torch.where(has, torch.logsumexp(sums, -1) - math.log(2.0), torch.zeros_like(sums[:, 0]))

    def factor_contexts(self, z, factors):
        owners = torch.as_tensor([f.owner for f in factors], device=z.device)
        flip = torch.as_tensor([f.orientation == 1 for f in factors], device=z.device)[:, None]
        zz = z.index_select(0, owners)
        left = torch.where(flip, zz[:, 1], zz[:, 0])
        right = torch.where(flip, zz[:, 0], zz[:, 1])
        codes = torch.as_tensor(np.stack([f.codes for f in factors]), device=z.device, dtype=torch.long)
        slot = torch.zeros(len(factors), 4, dtype=z.dtype, device=z.device)
        slot[torch.arange(len(factors)), torch.as_tensor([f.slot for f in factors], device=z.device)] = 1.0
        lane = self._t(np.stack([f.lane_feats for f in factors]))
        return self.pointer_in(torch.cat((left, right, self.code_embedding(codes).flatten(1), slot, lane), -1))

    def pair_condition(self, factors, chart, track):
        if self.config.conditioner == 'film':
            if not track:
                return None
            d = chart.derived()
            out = []
            for f in factors:
                C = len(f.times)
                fr = np.zeros((C, 3, 2, FRAME_DIM))
                fr[:, 0] = frames(chart, track, [chart.time(f.k)], f.k)[0]
                fr[:, 1] = frames(chart, track, f.times, f.k)
                fr[:, 2] = frames(chart, track, [d.start[f.k][f.lane]], f.k)[0]
                out.append(fr.reshape(C, -1))
            return self._t(np.concatenate(out))
        if not track:
            return self._t(np.zeros((sum(len(f.times) for f in factors), 0, TOKEN_DIM)))
        return self._t(np.concatenate([tokens(chart, track, f.times, f.k) for f in factors]))

    def pair_scores(self, base, owner, rel, cand, cond):
        q = self.pointer_out(nn.functional.gelu(base.index_select(0, owner) + self.pointer_relation(rel)))
        if self.config.conditioner == 'film':
            if cond is None:
                cond = q.new_zeros(1, 6 * FRAME_DIM).expand(len(q), -1)
            q = self.film(q, cond)
        else:
            q = self.tokens(q, cond)
        return (q * self.candidate(cand)).sum(-1) / math.sqrt(q.shape[-1]) + self.candidate_bias(cand)[:, 0]

    def all_pair_scores(self, base, factors, chart, track):
        sizes = [len(f.times) for f in factors]
        owner = torch.as_tensor(np.repeat(np.arange(len(factors)), sizes), device=base.device)
        rel = self._t(np.concatenate([f.relations for f in factors]))
        cand = self._t(np.concatenate([f.cand_feats for f in factors]))
        cond = self.pair_condition(factors, chart, track)
        P, budget = len(owner), self.config.candidate_budget
        if P <= budget or not torch.is_grad_enabled():
            return self.pair_scores(base, owner, rel, cand, cond), owner, sizes
        parts = []
        for s in range(0, P, budget):
            e = min(P, s + budget)
            parts.append(checkpoint(self.pair_scores, base, owner[s:e], rel[s:e], cand[s:e],
                                    None if cond is None else cond[s:e], use_reentrant=False))
        return torch.cat(parts), owner, sizes

    def factor_log_probs(self, z, factors, chart, track):
        base = self.factor_contexts(z, factors)
        scores, owner, sizes = self.all_pair_scores(base, factors, chart, track)
        F = len(factors)
        peak = torch.full((F,), -torch.inf, dtype=scores.dtype, device=scores.device)
        peak = peak.scatter_reduce(0, owner, scores.detach(), reduce='amax')
        total = scores.new_zeros(F).index_add(0, owner, (scores - peak.index_select(0, owner)).exp())
        lse = total.log() + peak
        offsets = np.concatenate(([0], np.cumsum(sizes)[:-1]))
        targets = torch.as_tensor(offsets + np.array([f.target for f in factors]), device=scores.device)
        return scores.index_select(0, targets) - lse

    def sequence_log_prob(self, chart: Chart, start: int, stop: int, track=()):
        """Sum of complete decision log-probabilities over [start, stop) (DPO uses this)."""
        return self.window(chart, start, stop, track).total.sum()

    def decision_log_prob(self, chart: Chart, k: int, track=()):
        return self.window(chart, k, k + 1, track).total[0]


def config_dict(config: R2Config):
    return asdict(config)
