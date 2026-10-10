"""The figure generator: per canonical beat, a distribution over the figure vocabulary.

    P(f | beat b) = g_0 softmax(U(h_b) . E[f] + c_f + sum_{p in f} a_b(p)) + sum_k g_k copy_k(f)

- Frame encoder: 128 standardised log-Mel bins at 10 ms, dilated 1-D convolutions (about ±310 ms).
- Position vectors: the encoder at each of the beat's 24 position times (``figures.POSITIONS``).
- Audio context A_b: the beat's position vectors and frame mean, then a transformer over beats,
  non-causal, local window ±``window`` beats with a learned bias per lag.
- Decoder: a GRU over beats reading A_b, the previous beat's figure, and ``beatdata``'s extra
  inputs (bar phase, progress, BPM / 120, the song request, the section request and its flag).
- a_b(p): an MLP of the decoder state, the position vector and the position, added to every figure
  that occupies p.
- Copy: ``heads`` attention heads from beat b to beats b-1 .. b-``lags`` (query and key from the
  audio context, a learned bias per lag, and whether the figures 1 and 2 beats before b and before
  the attended beat are equal), each with a null slot that falls back to the generator. Each head
  points at the attended beat's figure; the gate g from the decoder state mixes the generator and
  the heads.

Teacher forcing scores only the target figure (``nll``); ``Decoder`` runs it one beat at a time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .audio import MEL_BINS
from .beatdata import EXTRA_DIM
from .figures import NPOS


@dataclass
class NetConfig:
    vocab: int = 0
    enc: int = 64
    dilations: tuple = (1, 2, 4, 8, 16)
    ctx: int = 128
    ctx_layers: int = 2
    ctx_heads: int = 4
    window: int = 64
    hidden: int = 256
    fig: int = 64
    heads: int = 4
    lags: int = 64
    key: int = 16

    @classmethod
    def from_dict(cls, d):
        return cls(**dict(d, dilations=tuple(d['dilations'])))


class FrameEncoder(nn.Module):
    def __init__(self, cfg: NetConfig):
        super().__init__()
        self.inp = nn.Linear(MEL_BINS, cfg.enc)
        self.convs = nn.ModuleList([nn.Conv1d(cfg.enc, cfg.enc, 3, dilation=d, padding=d) for d in cfg.dilations])

    def forward(self, x):
        """[B, F, 128] -> [B, F, enc]."""
        h = F.gelu(self.inp(x)).transpose(1, 2)
        for c in self.convs:
            h = h + F.gelu(c(h))
        return h.transpose(1, 2)


class LocalAttention(nn.Module):
    """Pre-norm self-attention over beats with a learned bias per lag in [-window, window]."""

    def __init__(self, d, heads, window):
        super().__init__()
        self.heads, self.window = heads, window
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.bias = nn.Parameter(torch.zeros(heads, 2 * window + 1))

    def forward(self, x, valid):
        """x [B, L, d]; valid [B, L] bool."""
        B, L, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, L, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        lag = torch.arange(L, device=x.device)[None] - torch.arange(L, device=x.device)[:, None]
        near = lag.abs() <= self.window
        s = q @ k.transpose(-1, -2) / (d // self.heads) ** 0.5
        s = s + self.bias[:, lag.clamp(-self.window, self.window) + self.window]
        self_ = torch.eye(L, dtype=torch.bool, device=x.device)      # keeps padded queries finite
        s = s.masked_fill(~((near[None, None] & valid[:, None, None, :]) | self_), -torch.inf)
        a = torch.softmax(s, -1)
        x = x + self.out((a @ v).transpose(1, 2).reshape(B, L, d))
        return x + self.ff(self.norm2(x))


def _back(x, lags: int, fill):
    """[B, L, ...] -> [B, L, lags, ...]: entry [:, l, k-1] is x[:, l - k], ``fill`` before the start."""
    pad = torch.full((x.shape[0], lags) + x.shape[2:], fill, dtype=x.dtype, device=x.device)
    xp = torch.cat((pad, x), 1)
    idx = torch.arange(x.shape[1], device=x.device)[:, None] + lags - torch.arange(1, lags + 1, device=x.device)[None]
    return xp[:, idx]


class FigureNet(nn.Module):
    def __init__(self, cfg: NetConfig, members):
        super().__init__()
        self.cfg = cfg
        self.register_buffer('members', torch.as_tensor(np.asarray(members), dtype=torch.float32))   # [V, 24]
        V = cfg.vocab
        self.encoder = FrameEncoder(cfg)
        self.pos_in = nn.Linear(NPOS * cfg.enc, cfg.ctx)
        self.mean_in = nn.Linear(cfg.enc, cfg.ctx)
        self.extra_in = nn.Linear(EXTRA_DIM, cfg.ctx)
        self.context = nn.ModuleList([LocalAttention(cfg.ctx, cfg.ctx_heads, cfg.window) for _ in range(cfg.ctx_layers)])
        self.ctx_norm = nn.LayerNorm(cfg.ctx)
        self.fig_in = nn.Embedding(V + 1, cfg.fig)          # index V: start or unknown
        self.fig_in_pos = nn.Parameter(torch.randn(NPOS, cfg.fig) * 0.02)
        self.fig_in_count = nn.Embedding(NPOS + 1, cfg.fig)
        self.gru = nn.GRU(cfg.ctx + cfg.fig + EXTRA_DIM, cfg.hidden, batch_first=True)
        self.prior = nn.Linear(cfg.hidden, cfg.fig)
        self.fig_out = nn.Embedding(V, cfg.fig)
        self.fig_out_pos = nn.Parameter(torch.randn(NPOS, cfg.fig) * 0.02)
        self.fig_out_count = nn.Embedding(NPOS + 1, cfg.fig)
        self.register_buffer('fig_counts', self.members.sum(1).long())
        for e in (self.fig_in, self.fig_in_count, self.fig_out, self.fig_out_count):
            nn.init.normal_(e.weight, std=0.02)
        self.fig_bias = nn.Parameter(torch.zeros(V))
        self.pos_emb = nn.Parameter(torch.zeros(NPOS, 16))
        self.unary_h = nn.Linear(cfg.hidden, 64)
        self.unary = nn.Sequential(nn.Linear(64 + cfg.enc + 16, 64), nn.GELU(), nn.Linear(64, 1))
        self.q = nn.Linear(cfg.ctx, cfg.heads * cfg.key)
        self.k = nn.Linear(cfg.ctx, cfg.heads * cfg.key)
        self.lag_bias = nn.Parameter(torch.zeros(cfg.heads, cfg.lags))
        self.eq_w = nn.Parameter(torch.zeros(cfg.heads, 2))
        self.null = nn.Linear(cfg.hidden, cfg.heads)
        self.gate = nn.Linear(cfg.hidden, 1 + cfg.heads)

    # audio side, whole window at once
    def audio(self, mel, pos_frames, beat_frames, extra, valid):
        """mel [B, F, 128]; pos_frames [B, L, 24]; beat_frames [B, L, 2]; extra [B, L, E]; valid [B, L].

        Returns position vectors [B, L, 24, enc], audio context [B, L, ctx], copy queries and keys
        [B, L, heads, key]."""
        enc = self.encoder(mel)
        B, L = pos_frames.shape[:2]
        bidx = torch.arange(B, device=mel.device)[:, None]
        pos = enc[bidx, pos_frames.reshape(B, -1)].view(B, L, NPOS, -1)
        cs = torch.cat((enc.new_zeros(B, 1, enc.shape[2]), enc.cumsum(1)), 1)
        n = (beat_frames[..., 1] - beat_frames[..., 0]).clamp_min(1).to(enc.dtype)
        mean = (cs[bidx, beat_frames[..., 1]] - cs[bidx, beat_frames[..., 0]]) / n[..., None]
        x = self.pos_in(pos.flatten(2)) + self.mean_in(mean) + self.extra_in(extra)
        for layer in self.context:
            x = layer(x, valid)
        A = self.ctx_norm(x)
        H, K = self.cfg.heads, self.cfg.key
        return pos, A, self.q(A).view(B, L, H, K), self.k(A).view(B, L, H, K)

    def input_table(self):
        """[V + 1, fig] previous-figure embeddings: free, plus positions and count (the unknown token free only)."""
        comp = self.members @ self.fig_in_pos + self.fig_in_count(self.fig_counts)
        return self.fig_in.weight + torch.cat((comp, comp.new_zeros(1, comp.shape[1])))

    def output_table(self):
        """[V, fig]: free, plus positions and count, so rare figures share their parts' statistics."""
        return self.fig_out.weight + self.members @ self.fig_out_pos + self.fig_out_count(self.fig_counts)

    def generator_logits(self, h, pos, pos_on=1.0, table=None):
        """h [N, hidden], pos [N, 24, enc] -> [N, V] generator logits."""
        u = self.unary_h(h)[:, None].expand(-1, NPOS, -1)
        a = self.unary(torch.cat((u, pos, self.pos_emb.expand(len(h), -1, -1)), -1))[..., 0] * pos_on
        table = self.output_table() if table is None else table
        return self.prior(h) @ table.T + self.fig_bias + a @ self.members.T

    def copy_scores(self, q, kb, eq):
        """q [N, heads, key]; kb [N, lags, heads, key]; eq [N, lags, 2] -> [N, heads, lags] (no mask)."""
        s = torch.einsum('nhd,nkhd->nhk', q, kb) / self.cfg.key ** 0.5
        return s + self.lag_bias + torch.einsum('nkc,hc->nhk', eq, self.eq_w)

    def mixture(self, h, logits, scores, back_ok):
        """Gate, generator log-probabilities and copy attention (null last) for N beats."""
        s = scores.masked_fill(~back_ok[:, None, :], -torch.inf)
        att = torch.softmax(torch.cat((s, self.null(h)[..., None]), -1), -1)       # [N, heads, lags + 1]
        return torch.softmax(self.gate(h), -1), torch.log_softmax(logits, -1), att


def eq_features(ids, lags):
    """[B, L, lags, 2]: figure before b equals figure before b - k, for 1 and 2 beats before; ids -1 unknown."""
    out = []
    for d in (1, 2):
        before = torch.cat((torch.full_like(ids[:, :d], -1), ids[:, :-d]), 1)
        back = _back(before, lags, -1)
        out.append(((before[:, :, None] == back) & (before[:, :, None] >= 0)).float())
    return torch.stack(out, -1)


def nll(model: FigureNet, batch, *, drop=0.1, rng=None, off=()):
    """Per-beat negative log-likelihood [B, L] of the target figures (0 where invalid) and the valid mask.

    ``batch``: mel, pos_frames, beat_frames, extra, ids [B, L] (-1 unknown or padding), valid [B, L].
    ``drop``: share of previous-figure inputs replaced by the unknown token (training only).
    ``off``: ablations, any of 'copy' (gate forced to the generator), 'unary' (position scores
    zero), 'copy_audio' (copy query and key from a shuffled beat order)."""
    cfg = model.cfg
    ids, valid = batch['ids'], batch['valid']
    pos, A, q, k = model.audio(batch['mel'], batch['pos_frames'], batch['beat_frames'], batch['extra'], valid)
    V = cfg.vocab
    prev = torch.cat((torch.full_like(ids[:, :1], V), ids[:, :-1]), 1)
    prev = torch.where(prev < 0, V, prev)
    if drop:
        prev = torch.where(torch.rand(prev.shape, generator=rng, device=prev.device) < drop, V, prev)
    h, _ = model.gru(torch.cat((A, model.input_table()[prev], batch['extra']), -1))
    B, L = ids.shape
    if 'copy_audio' in off:
        g = torch.Generator().manual_seed(5)
        idx = torch.arange(L).repeat(B, 1)
        for i, n in enumerate(valid.sum(1).tolist()):
            idx[i, :n] = torch.randperm(n, generator=g)
        k = k[torch.arange(B)[:, None], idx.to(k.device)]
    m = valid & (ids >= 0)
    hm, posm = h[m], pos[m]
    logits = model.generator_logits(hm, posm, 0.0 if 'unary' in off else 1.0)
    kb = _back(k, cfg.lags, 0.0)[m]
    back_ids = _back(ids, cfg.lags, -1)[m]                                   # [N, lags]
    back_ok = _back(valid, cfg.lags, False)[m] & (back_ids >= 0)
    scores = model.copy_scores(q[m], kb, eq_features(ids, cfg.lags)[m])
    g, logp, att = model.mixture(hm, logits, scores, back_ok)
    target = ids[m]
    lp_gen = logp.gather(1, target[:, None])[:, 0]
    hit = (back_ids == target[:, None]).float()                              # [N, lags]
    p_copy = (att[..., :-1] * hit[:, None]).sum(-1) + att[..., -1] * lp_gen.exp()[:, None]   # [N, heads]
    if 'copy' in off:
        p = lp_gen.exp()
    else:
        p = g[:, 0] * lp_gen.exp() + (g[:, 1:] * p_copy).sum(-1)
    out = torch.zeros(B, L, device=ids.device)
    out[m] = -torch.log(p.clamp_min(1e-30))
    return out, m


def load_net(path, map_location='cpu'):
    """(FigureNet in eval mode, training record) from a ``fit`` checkpoint."""
    data = torch.load(path, map_location=map_location, weights_only=False)
    model = FigureNet(NetConfig.from_dict(data['config']), data['members'])
    model.load_state_dict(data['model'])
    return model.eval(), data['record']


def config_dict(cfg: NetConfig) -> dict:
    return asdict(cfg)
