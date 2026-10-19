"""N1 v0 network: a local encoder with whole-song context, and F1 and F2 heads trained together.

Sized from rung 0's measurements on the mac (NE-S: 30 crops/s at batch 8, 24 at batch 32; NE-M ran
out of MPS memory at batch 8 with 60 s crops), so that about three hours give five to six passes
over the curated audio:

- **Stem and local path**: NE-S's (`ensomi_n.models.encoder`): Conv2d 1→24→48→64 with frequency
  pooling to 4 bins, D 160, ten dilated residual blocks (receptive field ±10 s) at 100 fps.
- **Mid path** at 25 fps: a strided convolution to D 192, three transformer blocks over the crop
  (self-attention with rotary positions), each cross-attending to the song memory.
- **Song memory**, the whole-song context path (design principle 3, so that it can be ablated):
  the whole song's ``memo-v1`` input (25 fps, 64 bands, standardised with the song's statistics) →
  a strided convolution to 12.5 fps at D 128 → four dilated residual blocks with channel layer norm
  (receptive field about ±5 s) → 32 learned latents that read it (two cross-attention blocks, key
  padding masked) and attend to each other (two blocks). It never sees the crop's position.
- **Up path**: transposed convolution back to 100 fps, a skip from the local path, one residual block.
- **Heads** (one linear map): F1 (design §2.1) beat, half-beat and downbeat logits with sub-frame
  offsets, a boundary logit (tempo change or reset); F2 (§2.2) beat phase as a two-component von
  Mises mixture, log period with its sigma, bar phase with one concentration; one level head
  (notated over canonical BPM: 1/2, 1, 2, 4) shared by both.

``memory_mode`` selects the input-use check (guardrail 3): ``'song'`` (normal), ``'zero'`` (the
cross-attention to the memory is skipped), or a memory from another song passed in by the caller.
"""
from __future__ import annotations

import math

from .._torch import F, nn, torch
from ..models.encoder import LocalPath, NE_S_DILATIONS, ResBlock, Stem, _rope
from ..models.heads import _log_i0, _vm_logpdf

N_LEVEL = 4
OFFSET_MS = 15.0


class MHA(nn.Module):
    def __init__(self, d: int, heads: int, d_kv: int | None = None, rope: bool = False):
        super().__init__()
        self.h, self.rope = heads, rope
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d_kv or d, 2 * d)
        self.o = nn.Linear(d, d)

    def forward(self, x, ctx=None, key_mask=None):
        ctx = x if ctx is None else ctx
        b, t, d = x.shape
        q = self.q(x).view(b, t, self.h, -1).transpose(1, 2)
        k, v = self.kv(ctx).view(b, ctx.shape[1], 2, self.h, -1).permute(2, 0, 3, 1, 4)
        if self.rope:
            q, k = _rope(q), _rope(k)
        mask = None if key_mask is None else key_mask[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(y.transpose(1, 2).reshape(b, t, d))


class Block(nn.Module):
    """Pre-norm block: optional self-attention, optional cross-attention, FFN."""

    def __init__(self, d: int, heads: int, *, self_att: bool = True, d_ctx: int | None = None, rope: bool = False,
                 ffn: int = 4):
        super().__init__()
        self.sa = MHA(d, heads, rope=rope) if self_att else None
        self.n1 = nn.LayerNorm(d) if self_att else None
        self.ca = MHA(d, heads, d_kv=d_ctx) if d_ctx else None
        self.nc = nn.LayerNorm(d) if d_ctx else None
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ffn * d), nn.GELU(), nn.Linear(ffn * d, d))

    def forward(self, x, ctx=None, key_mask=None, use_ctx: bool = True):
        if self.sa is not None:
            x = x + self.sa(self.n1(x))
        if self.ca is not None and ctx is not None and use_ctx:
            x = x + self.ca(self.nc(x), ctx, key_mask)
        return x + self.ff(self.n2(x))


class ConvLN(nn.Module):
    """Dilated residual Conv1d block with channel layer norm (no batch statistics over padding)."""

    def __init__(self, d: int, dilation: int, kernel: int = 5):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.n1 = nn.LayerNorm(d)
        self.c1 = nn.Conv1d(d, d, kernel, padding=pad, dilation=dilation)
        self.n2 = nn.LayerNorm(d)
        self.c2 = nn.Conv1d(d, d, kernel, padding=pad, dilation=dilation)

    def forward(self, x):                        # [B, S, D]
        h = self.c1(F.gelu(self.n1(x)).transpose(1, 2)).transpose(1, 2)
        h = self.c2(F.gelu(self.n2(h)).transpose(1, 2)).transpose(1, 2)
        return x + h


class SongMemory(nn.Module):
    def __init__(self, n_in: int = 64, d: int = 128, n_latents: int = 32, heads: int = 4,
                 dilations=(1, 2, 4, 8)):
        super().__init__()
        self.inp = nn.Linear(n_in, d)
        self.down = nn.Conv1d(d, d, 4, stride=2, padding=1)
        self.convs = nn.ModuleList(ConvLN(d, k) for k in dilations)
        self.norm = nn.LayerNorm(d)
        self.latents = nn.Parameter(torch.randn(n_latents, d) * 0.02)
        self.read = nn.ModuleList(Block(d, heads, self_att=False, d_ctx=d) for _ in range(2))
        self.think = nn.ModuleList(Block(d, heads) for _ in range(2))

    def forward(self, memo, mask=None):          # memo [B, S, 64] at 25 fps, mask [B, S] True = valid
        h = self.inp(memo)
        h = self.down(h.transpose(1, 2)).transpose(1, 2)                  # [B, S/2, d]
        m = None
        if mask is not None:
            m = mask[:, ::2][:, :h.shape[1]]
            if m.shape[1] < h.shape[1]:
                m = F.pad(m, (0, h.shape[1] - m.shape[1]), value=False)
            m = m.clone()
            m[:, 0] = True
            h = h * m[..., None]
        for blk in self.convs:
            h = blk(h)
            if m is not None:
                h = h * m[..., None]
        h = self.norm(h)
        z = self.latents.unsqueeze(0).expand(memo.shape[0], -1, -1)
        for blk in self.read:
            z = blk(z, h, m)
        for blk in self.think:
            z = blk(z)
        return z


class NEv0(nn.Module):
    def __init__(self, d: int = 160, d_mid: int = 192, mid_layers: int = 3, heads: int = 4, d_mem: int = 128,
                 n_latents: int = 32, n_mels: int = 128, dilations=NE_S_DILATIONS, memory: bool = True):
        super().__init__()
        self.d = d
        self.stem = Stem(d, n_mels)
        self.local = LocalPath(d, dilations)
        self.down = nn.Sequential(nn.Conv1d(d, d_mid, 4, stride=4), nn.GELU())
        self.memory = SongMemory(64, d_mem, n_latents, heads) if memory else None
        self.mid = nn.ModuleList(Block(d_mid, heads, d_ctx=d_mem if memory else None, rope=True)
                                 for _ in range(mid_layers))
        self.mid_norm = nn.LayerNorm(d_mid)
        self.up = nn.ConvTranspose1d(d_mid, d, 4, stride=4)
        self.skip = nn.Linear(d, d)
        self.out_block = ResBlock(d, 1, 5)
        self.norm = nn.BatchNorm1d(d)

    def song_memory(self, memo, mask=None):
        return None if self.memory is None else self.memory(memo, mask)

    def forward(self, mel, mem=None, use_memory: bool = True):
        """``mel`` [B, T, 128] (T a multiple of 4); ``mem`` the latents from `song_memory`, [B, L, d_mem]."""
        loc = self.local(self.stem(mel))                       # [B, T, d]
        h = self.down(loc.transpose(1, 2)).transpose(1, 2)     # [B, T/4, d_mid]
        for blk in self.mid:
            h = blk(h, mem, None, use_ctx=use_memory and mem is not None)
        h = self.mid_norm(h)
        u = self.up(h.transpose(1, 2))[:, :, :loc.shape[1]]
        u = u + self.skip(loc).transpose(1, 2)
        u = self.out_block(u)
        return F.gelu(self.norm(u)).transpose(1, 2)


# Head layout: F1 logits beat, half, down (0:3), offsets (3:6), boundary (6); level (7:11);
# F2 w, c1, s1, c2, s2, k1, k2 (11:18), logp, logp_sigma (18:20), bc, bs, bk (20:23).
N_OUT = 23


class Heads(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, N_OUT)

    def forward(self, h):
        y = self.lin(h)
        k = F.softplus(y[..., 16:18]) + 1e-3
        return dict(logits=y[..., 0:3], offset=OFFSET_MS * torch.tanh(y[..., 3:6]), bound=y[..., 6],
                    level=y[..., 7:11], w=y[..., 11], c1=y[..., 12], s1=y[..., 13], c2=y[..., 14], s2=y[..., 15],
                    k1=k[..., 0], k2=k[..., 1], logp=y[..., 18], logp_sigma=F.softplus(y[..., 19]) + 1e-3,
                    bc=y[..., 20], bs=y[..., 21], bk=F.softplus(y[..., 22]) + 1e-3)


def _focal_bce(logit, y, gamma: float = 2.0, pos_weight: float = 1.0):
    p = torch.sigmoid(logit)
    bce = F.binary_cross_entropy_with_logits(logit, y, reduction='none')
    pt = torch.where(y > 0.5, p, 1 - p)
    w = torch.where(y > 0.5, torch.full_like(y, pos_weight), torch.ones_like(y))
    return w * (1 - pt) ** gamma * bce


class V0(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.encoder = NEv0(**kw)
        self.heads = Heads(self.encoder.d)

    def forward(self, mel, memo=None, memo_mask=None, *, mem=None, use_memory: bool = True):
        if mem is None and memo is not None and self.encoder.memory is not None:
            mem = self.encoder.song_memory(memo, memo_mask)
        return self.heads(self.encoder(mel, mem, use_memory))

    @staticmethod
    def loss(out, tg, pos_weight: dict, f2_weight: float = 0.5, per_item: bool = False):
        """Total loss and its parts. ``per_item`` also returns each crop's total (for the ablation intervals)."""
        w = tg['weight']
        wsum = w.sum(dim=1).clamp_min(1.0)
        parts, items = {}, 0.0
        for i, name in enumerate(('beat', 'half', 'down')):
            pw = torch.as_tensor(pos_weight[name], device=w.device)
            bce = F.binary_cross_entropy_with_logits(out['logits'][..., i], tg[name], reduction='none', pos_weight=pw)
            l = (bce * w).sum(1) / wsum
            om = tg[name + '_offmask'].float() * (w > 0)
            hub = F.huber_loss(out['offset'][..., i] / 5.0, tg[name + '_off'] / 5.0, reduction='none', delta=1.0)
            lo = (hub * om).sum(1) / om.sum(1).clamp_min(1.0)
            parts[name], parts[name + '_off'] = l, lo
            items = items + l + 0.5 * lo
        mask = tg['mask'].float()
        fb = _focal_bce(out['bound'], tg['bound'], pos_weight=float(pos_weight.get('bound', 1.0)))
        parts['bound'] = (fb * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        items = items + 0.5 * parts['bound']
        lv = tg['level']
        ce = F.cross_entropy(out['level'].transpose(1, 2), lv.clamp_min(0), reduction='none')
        ml = (lv >= 0).float()
        parts['level'] = (ce * ml).sum(1) / ml.sum(1).clamp_min(1.0)
        items = items + 0.5 * parts['level']
        # F2
        x = 2 * math.pi * tg['phase']
        lp = torch.logaddexp(F.logsigmoid(out['w']) + _vm_logpdf(x, out['c1'], out['s1'], out['k1']),
                             F.logsigmoid(-out['w']) + _vm_logpdf(x, out['c2'], out['s2'], out['k2']))
        m = tg['phase_mask'].float() * w
        msum = m.sum(1).clamp_min(1.0)
        parts['phase'] = -(lp * m).sum(1) / msum
        sig = out['logp_sigma']
        gnll = 0.5 * ((out['logp'] - tg['logp']) / sig) ** 2 + torch.log(sig)
        parts['period'] = (gnll * m).sum(1) / msum
        xb = 2 * math.pi * tg['bar_phase']
        mb = tg['bar_mask'].float() * w
        parts['bar'] = -(_vm_logpdf(xb, out['bc'], out['bs'], out['bk']) * mb).sum(1) / mb.sum(1).clamp_min(1.0)
        items = items + f2_weight * (parts['phase'] + 0.5 * parts['period'] + 0.5 * parts['bar'])
        total = items.mean()
        scal = {k: float(v.detach().mean()) for k, v in parts.items()}
        if per_item:
            return total, scal, items.detach(), {k: v.detach() for k, v in parts.items()}
        return total, scal

    @staticmethod
    def activations(out) -> dict:
        """Per-frame evidence for the decoder and the probes (torch tensors, [B, T])."""
        p = torch.sigmoid(out['logits'])
        first = out['w'] >= 0
        c = torch.where(first, out['c1'], out['c2'])
        s = torch.where(first, out['s1'], out['s2'])
        k = torch.where(first, out['k1'], out['k2'])
        return dict(beat=p[..., 0], half=p[..., 1], down=p[..., 2], beat_off=out['offset'][..., 0],
                    down_off=out['offset'][..., 2], bound=torch.sigmoid(out['bound']),
                    level=torch.softmax(out['level'], dim=-1), phase=torch.remainder(torch.atan2(s, c) / (2 * math.pi), 1.0),
                    kappa=k, logp=out['logp'], logp_sigma=out['logp_sigma'],
                    bar_phase=torch.remainder(torch.atan2(out['bs'], out['bc']) / (2 * math.pi), 1.0), bar_kappa=out['bk'])


def n_params(m) -> int:
    return sum(p.numel() for p in m.parameters())
