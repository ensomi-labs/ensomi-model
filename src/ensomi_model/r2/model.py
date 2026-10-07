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
from .features import (FRAME_DIM, HISTORY_DIM, LANE_QUERY_DIM, LN_LENGTH_DIM, LN_LEVEL_DIM, LN_LEVEL_MODES,
                       PRESENCE, QUERY_DIM, RELATION_DIM, STAR_VALUES,
                       TOKEN_DIM, Chart, decision_factors, frames, history_tokens, lane_query, query_features, tokens)
from .locality import visible
from .state import action_support_mask

MAX_PARAMETERS = 4_500_000
CONDITIONERS = ('film', 'tokens')   # conditioner modules; every other parameter is the natural model


@dataclass(frozen=True)
class R2Config:
    hidden: int = 128
    levels: int = 8
    expansion: int = 4
    rank: int = 16
    memory: str = 'landmarks'      # 'landmarks' | 'none'
    stride: int = 64
    conditioner: str = 'film'      # 'film' | 'tokens' (a lead-in form; needs token_lead_in)
    code_dim: int = 8
    candidate_budget: int = 8192
    checkpoint_temporal: bool = True  # recompute TCN block activations in backward: memory only, same values
    max_parameters: int = MAX_PARAMETERS
    rule_l: bool = True            # False: test power checks and v1 reproduction only
    birth_role: bool = False       # True: v1's birth-role frame; test power checks and v1 reproduction only
    presence: str = 'none'         # 'anywhere': v1's presence bit; test power checks and v1 reproduction only
    star_value: str = 'absolute'   # 'absolute' tiled star or 'residual' target - b(S) in the difficulty frame
    token_lead_in: bool = False    # the token conditioner shows intervals before and after them
    film_width: int = 128          # FiLM MLP hidden width (conditioning capacity)
    film_layers: int = 1           # FiLM MLP hidden layers
    identity_gate: bool = True     # False: FiLM(z, 0) acts on natural decisions; test power checks only
    ln_level: str = 'off'          # whole-song LN share: three extra shared query channels when on
    ln_length: str = 'off'         # whole-song median log2 hold length in beats: known bit and value

    def __post_init__(self):
        if self.memory not in ('landmarks', 'none') or self.conditioner not in ('film', 'tokens'):
            raise ContractError('memory must be landmarks|none and conditioner film|tokens')
        if self.film_width < 1 or self.film_layers < 1:
            raise ContractError('film_width and film_layers are positive')
        if self.conditioner == 'tokens' and not self.token_lead_in:
            raise ContractError('The token conditioner shows every interval before its start and after its end; '
                                'it is a lead-in form and raises under the default eta')
        if self.presence not in PRESENCE:
            raise ContractError(f'presence must be one of {PRESENCE}; lead-in presence is reserved')
        if self.star_value not in STAR_VALUES:
            raise ContractError(f'star_value must be one of {STAR_VALUES}')
        if self.ln_level not in LN_LEVEL_MODES:
            raise ContractError(f'ln_level must be one of {LN_LEVEL_MODES}')
        if self.ln_length not in LN_LEVEL_MODES or (self.ln_length == 'on' and self.ln_level != 'on'):
            raise ContractError('ln_length must be off|on and requires ln_level on')

    @property
    def roles(self):
        return 3 if self.birth_role else 2


class JointHead5(JointHead):
    """R1 joint head over all 625 five-code lane combinations."""

    def __init__(self, hidden, rank):
        super().__init__(hidden, 5, rank)
        self.register_buffer('left', torch.tensor([int(a[0] * 5 + a[1]) for a in ACTIONS]), persistent=False)
        self.register_buffer('right', torch.tensor([int(a[3] * 5 + a[2]) for a in ACTIONS]), persistent=False)


class FiLM(nn.Module):
    """Frames of (row, candidate[, birth]) roles x 2 kinds -> (gamma, delta) on a hidden vector.

    The MLP has ``layers`` hidden layers of ``width`` (default one of 128, 42,112 parameters with
    the norm); its output layer is zero-initialised. With ``gate`` (the default), a row whose frame
    is all zero, i.e. a decision or candidate pair that reads no interval (rule L), passes through
    unchanged: FiLM(z, 0) = z exactly, so training this module never moves a natural decision.
    """

    def __init__(self, hidden, roles=2, width=128, layers=1, gate=True):
        super().__init__()
        self.inputs = roles * 2 * FRAME_DIM
        self.gate = gate
        dims = [self.inputs] + [width] * layers
        mods = []
        for a, b in zip(dims, dims[1:]):
            mods += [nn.Linear(a, b), nn.GELU()]
        self.mlp = nn.Sequential(*mods, nn.Linear(dims[-1], 2 * hidden))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, z, cond):
        if self.gate:
            reads = (cond != 0).any(-1, keepdim=True)
            if not bool(reads.any()):
                return z
        gamma, delta = self.mlp(cond).chunk(2, -1)
        out = z + gamma * self.norm(z) + delta
        return torch.where(reads, out, z) if self.gate else out


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
    governed_ln: torch.Tensor = None   # [m] log P(tap-vs-LN split | head mask, release types); 0 at EOS
    visible: np.ndarray = None         # [m,2] bool: decision reads an LN / difficulty interval (V_k)
    logp: torch.Tensor = None          # [m,625] masked action log-softmax
    candidate_pairs: int = 0           # release-candidate scores computed (both orientations)
    pair_logp: torch.Tensor = None     # [P] log-probability of every candidate within its release factor
    pair_rows: np.ndarray = None       # [P] decision row of each candidate pair

    @property
    def total(self):
        return self.action + self.release

    @property
    def factors(self):
        """Scored factors per decision: the action factor (head rows) plus one per gap release."""
        return (~self.eos).astype(np.int64) + self.gap_lns


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
        self.film = FiLM(H, c.roles, c.film_width, c.film_layers, c.identity_gate)
        self.tokens = TokenConditioner(H)
        self.joint = JointHead5(H, c.rank)
        self.code_embedding = nn.Embedding(5, c.code_dim)
        self.pointer_in = nn.Linear(2 * H + 4 * c.code_dim + 4 + LANE_QUERY_DIM, H)
        self.pointer_relation = nn.Linear(RELATION_DIM, H, bias=False)
        self.pointer_out = nn.Linear(H, H)
        self.candidate = nn.Sequential(nn.Linear(CANDIDATE_DIM, H), nn.GELU(), nn.Linear(H, H))
        self.candidate_bias = nn.Linear(CANDIDATE_DIM, 1)
        if c.ln_level == 'on':
            # Preserve every legacy parameter's initialization and the caller's RNG stream.
            with torch.random.fork_rng(devices=[]):
                self.ln_level_reader = nn.Linear(LN_LEVEL_DIM, H, bias=False)
                nn.init.zeros_(self.ln_level_reader.weight)
        if c.ln_length == 'on':
            with torch.random.fork_rng(devices=[]):
                self.ln_length_reader = nn.Linear(LN_LENGTH_DIM, H, bias=False)
                nn.init.zeros_(self.ln_length_reader.weight)
        counts = self.parameter_counts()
        if verbose:
            print({'parameters': counts}, flush=True)
        if counts['total'] > c.max_parameters:
            raise ContractError(f'Model has {counts["total"]} parameters, above {c.max_parameters}')

    def parameter_counts(self):
        out = {name: sum(p.numel() for p in m.parameters()) for name, m in self.named_children()}
        out['total'] = sum(p.numel() for p in self.parameters())
        return out

    def parameter_split(self):
        """(conditioning, natural): lists of (name, parameter). Conditioning is the configured
        conditioner module, the path a frozen-base phase C trains; natural is every parameter outside
        both conditioner modules, the model phase N trains. The unused conditioner is in neither."""
        conditioning, natural = [], []
        for name, p in self.named_parameters():
            module = name.split('.', 1)[0]
            if module == self.config.conditioner:
                conditioning.append((name, p))
            elif module not in CONDITIONERS:
                natural.append((name, p))
        return conditioning, natural

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
        if self.config.ln_level == 'off':
            exact = pointwise(self.exact, qf)
        else:
            base = pointwise(self.exact[0], qf[..., :QUERY_DIM].contiguous())
            added = pointwise(self.ln_level_reader, qf[..., QUERY_DIM:QUERY_DIM + LN_LEVEL_DIM])
            base = base + added
            if self.config.ln_length == 'on':
                base = base + pointwise(self.ln_length_reader, qf[..., QUERY_DIM + LN_LEVEL_DIM:])
            exact = pointwise(self.exact[2], self.exact[1](base))
        h = pointwise(self.fuse, torch.cat((before, exact), -1))
        if self.config.memory == 'landmarks':
            h = h + self.read_landmarks(h, marks, visible)
        return self.condition(h, cond)

    def visible_sets(self, chart: Chart, track, ks):
        """{k: V_k} under the model's rule-L switch."""
        return {int(k): visible(chart, track, int(k), self.config.rule_l) for k in np.asarray(ks)}

    def _frames(self, chart, vis, track, times, k):
        c = self.config
        return frames(chart, vis, times, k, presence=c.presence, full_track=track, star_value=c.star_value)

    def row_condition(self, chart: Chart, track, ks, vis=None):
        ks = np.asarray(ks)
        if self.config.conditioner == 'film':
            out = np.zeros((len(ks), self.config.roles, 2, FRAME_DIM))
            if track:
                vis = vis if vis is not None else self.visible_sets(chart, track, ks)
                for i, k in enumerate(ks):
                    out[i, 0] = self._frames(chart, vis[int(k)], track, [chart.time(k)], int(k))[0]
            return self._t(out.reshape(len(ks), -1))
        return self._t(np.concatenate([tokens(chart, track, [chart.time(k)], int(k), star_value=self.config.star_value)
                                       for k in ks], 0)
                       if track else np.zeros((len(ks), 0, TOKEN_DIM)))

    def encode_history(self, chart: Chart, N: int):
        if N <= 0:
            return None
        raw = self._t(history_tokens(chart, N))
        return self.temporal(raw[None], checkpoint=self.config.checkpoint_temporal)[0]

    def query_features(self, chart: Chart, ks, ln_level: float | None = None, ln_length: float | None = None):
        """Model-dtype query tensor with the configured whole-song LN-level channels."""
        return self._t(query_features(chart, ks, ln_level=self.config.ln_level, level=ln_level,
                                      ln_length=self.config.ln_length, length=ln_length))

    def window_hands(self, chart: Chart, ks, track=(), vis=None, *, ln_level: float | None = None,
                     ln_length: float | None = None):
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
        qf = self.query_features(chart, ks, ln_level, ln_length)
        return self.hands(before, qf, marks, visible, self.row_condition(chart, track, ks, vis))

    # ---- likelihood ------------------------------------------------------------------------------

    def action_log_probs(self, z, masks):
        logits = self.joint(z)
        mask = torch.as_tensor(np.asarray(masks), device=z.device)
        return logits.masked_fill(~mask, -torch.inf).log_softmax(-1)

    def window(self, chart: Chart, start: int, stop: int, track=(), *, pairs=False,
               ln_level: float | None = None, ln_length: float | None = None) -> WindowOut:
        """Teacher-forced log-probabilities of decisions [start, stop) of ``chart``; ``pairs`` also keeps
        every candidate's log-probability within its release factor (the phase-C KL term needs them).
        ``ln_level`` is a whole-song level in [0,1], or None for unknown, used only with the input on.
        ``ln_length`` is the whole-song median log2 hold length in beats, or None for unknown.
        """
        if not 0 <= start < stop <= min(chart.n, chart.K + 1):
            raise ContractError('Window outside the known decisions')
        ks = np.arange(start, stop)
        d = chart.derived()
        vis = self.visible_sets(chart, track, ks) if track else {int(k): () for k in ks}
        z = self.window_hands(chart, ks, track, vis, ln_level=ln_level, ln_length=ln_length)
        held = d.held[ks]
        masks = np.stack([action_support_mask(d.held[k], k == chart.K) for k in ks])
        targets = np.array([action_index(chart.actions[k]) for k in ks])
        if not masks[np.arange(len(ks)), targets].all():
            raise ContractError('A teacher-forced action is outside its support')
        logp = self.action_log_probs(z, masks)
        target_t = torch.as_tensor(targets, device=z.device)
        action = logp.gather(1, target_t[:, None])[:, 0]
        lanes = lane_query(chart, ks).astype(np.float32)
        factors = []
        for i, k in enumerate(ks):
            for o in (0, 1):
                factors += decision_factors(chart, int(k), i, chart.actions[k], chart.gap[k], lanes[i], o)
        pair_logp = pair_rows = None
        if pairs:
            release, pair_logp = self.release_and_pairs(z, factors, chart, track, len(ks))
            pair_rows = np.repeat(np.array([f.owner for f in factors], dtype=np.int64),
                                  np.array([len(f.times) for f in factors], dtype=np.int64))
        else:
            release = self.release_from_factors(z, factors, chart, track, len(ks))
        gap_lns = np.array([sum(1 for f in factors if f.owner == i and f.orientation == 0) for i in range(len(ks))])
        kinds = np.zeros((len(ks), 2), dtype=bool)
        for i, k in enumerate(ks):
            for iv in vis[int(k)]:
                kinds[i, iv.kind] = True
        return WindowOut(action, release, ks, ks == chart.K, gap_lns, governed_split(logp, target_t, held, masks),
                         kinds, logp, int(sum(len(f.times) for f in factors)), pair_logp, pair_rows)

    def release_from_factors(self, z, factors, chart, track, m):
        """[m] release log-likelihood of each decision."""
        return self.release_and_pairs(z, factors, chart, track, m)[0]

    def release_and_pairs(self, z, factors, chart, track, m):
        """([m] release log-likelihood of each decision, [P] candidate log-probabilities within their factors)."""
        if not factors:
            return z.new_zeros(m), z.new_zeros(0)
        lp, pair_logp = self.factor_pair_log_probs(z, factors, chart, track)
        slot = torch.as_tensor([2 * f.owner + f.orientation for f in factors], device=z.device)
        sums = z.new_zeros(2 * m).index_add(0, slot, lp).reshape(m, 2)
        has = torch.zeros(m, dtype=torch.bool, device=z.device)
        has[torch.as_tensor([f.owner for f in factors], device=z.device)] = True
        return torch.where(has, torch.logsumexp(sums, -1) - math.log(2.0), torch.zeros_like(sums[:, 0])), pair_logp

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
            vis = {}
            out = []
            for f in factors:
                if f.k not in vis:
                    vis[f.k] = visible(chart, track, f.k, self.config.rule_l)
                v = vis[f.k]
                C = len(f.times)
                fr = np.zeros((C, self.config.roles, 2, FRAME_DIM))
                fr[:, 0] = self._frames(chart, v, track, [chart.time(f.k)], f.k)[0]
                fr[:, 1] = self._frames(chart, v, track, f.times, f.k)
                if self.config.birth_role:
                    fr[:, 2] = self._frames(chart, v, track, [d.start[f.k][f.lane]], f.k)[0]
                out.append(fr.reshape(C, -1))
            return self._t(np.concatenate(out))
        if not track:
            return self._t(np.zeros((sum(len(f.times) for f in factors), 0, TOKEN_DIM)))
        return self._t(np.concatenate([tokens(chart, track, f.times, f.k, star_value=self.config.star_value)
                                       for f in factors]))

    def pair_scores(self, base, owner, rel, cand, cond):
        q = self.pointer_out(nn.functional.gelu(base.index_select(0, owner) + self.pointer_relation(rel)))
        if self.config.conditioner == 'film':
            if cond is None:
                cond = q.new_zeros(1, self.film.inputs).expand(len(q), -1)
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
        """[F] target log-probability of each factor."""
        return self.factor_pair_log_probs(z, factors, chart, track)[0]

    def factor_pair_log_probs(self, z, factors, chart, track):
        """([F] target log-probability of each factor, [P] log-probability of every candidate in its factor)."""
        base = self.factor_contexts(z, factors)
        scores, owner, sizes = self.all_pair_scores(base, factors, chart, track)
        F = len(factors)
        peak = torch.full((F,), -torch.inf, dtype=scores.dtype, device=scores.device)
        peak = peak.scatter_reduce(0, owner, scores.detach(), reduce='amax')
        total = scores.new_zeros(F).index_add(0, owner, (scores - peak.index_select(0, owner)).exp())
        lse = total.log() + peak
        pair_logp = scores - lse.index_select(0, owner)
        offsets = np.concatenate(([0], np.cumsum(sizes)[:-1]))
        targets = torch.as_tensor(offsets + np.array([f.target for f in factors]), device=scores.device)
        return pair_logp.index_select(0, targets), pair_logp

    def sequence_log_prob(self, chart: Chart, start: int, stop: int, track=(), *, ln_level: float | None = None,
                          ln_length: float | None = None):
        """Sum of complete decision log-probabilities over [start, stop) (DPO uses this)."""
        return self.window(chart, start, stop, track, ln_level=ln_level, ln_length=ln_length).total.sum()

    def decision_log_prob(self, chart: Chart, k: int, track=(), *, ln_level: float | None = None,
                          ln_length: float | None = None):
        return self.window(chart, k, k + 1, track, ln_level=ln_level, ln_length=ln_length).total[0]


def config_dict(config: R2Config):
    return asdict(config)


def _lane_groups():
    """[16 held patterns, 625] group index of the tap-versus-LN split: per lane, free codes 1 and 2
    (tap, LN head) merge and held codes 3 and 4 (gap release + tap / + LN head) merge."""
    out = np.zeros((16, len(ACTIONS)), dtype=np.int64)
    free, held = np.array([0, 1, 1, 0, 0]), np.array([0, 1, 2, 3, 3])
    for p in range(16):
        h = [(p >> lane) & 1 for lane in range(4)]
        g = np.zeros(len(ACTIONS), dtype=np.int64)
        for lane in range(4):
            g = g * 4 + np.where(h[lane], held[ACTIONS[:, lane]], free[ACTIONS[:, lane]])
        out[p] = g
    return out


LN_GROUPS = _lane_groups()


def held_pattern(held) -> np.ndarray:
    held = np.asarray(held, dtype=np.int64).reshape(-1, 4)
    return held[:, 0] + 2 * held[:, 1] + 4 * held[:, 2] + 8 * held[:, 3]


def governed_split(logp, targets, held, masks):
    """log P(LN-ness | head mask, release types) = log P(a) - log sum over its group of P(a') [m]."""
    groups = torch.as_tensor(LN_GROUPS[held_pattern(held)], device=logp.device)       # [m,625]
    same = (groups == groups.gather(1, targets[:, None])) & torch.as_tensor(np.asarray(masks), device=logp.device)
    group_lp = logp.masked_fill(~same, -torch.inf).logsumexp(-1)
    return logp.gather(1, targets[:, None])[:, 0] - group_lp
