"""The shared encoder NE (design §2.0): NE-S for rung 1, NE-M from rung 2.

Input: per-song standardised log-Mel [B, T, 128] at 100 fps. Output: frame
features [B, T, D] at 100 fps.

- Stem: Conv2d 1→24→48→64 (3×3) with frequency pooling 2·4·4 (128 → 4 bins),
  then a linear map from 64 × 4 to D.
- Local path: residual blocks of two dilated Conv1d (kernel 5) each.
- NE-M adds a mid path at 25 fps (strided convolution, transformer layers with
  rotary positions, full attention over the crop), a song memory (the whole
  song at 6.25 fps read by learned latents) that every mid layer cross-attends
  to, and an up path back to 100 fps with a skip from the local path.
"""
from __future__ import annotations

import math

from .._torch import F, nn, torch


class Stem(nn.Module):
    def __init__(self, d: int, n_mels: int = 128):
        super().__init__()
        chans = (1, 24, 48, 64)
        pools = (2, 4, 4)
        layers = []
        for cin, cout, pool in zip(chans[:-1], chans[1:], pools):
            layers += [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.GELU(),
                       nn.MaxPool2d((1, pool))]
        self.conv = nn.Sequential(*layers)
        bins = n_mels // math.prod(pools)
        self.proj = nn.Linear(chans[-1] * bins, d)

    def forward(self, mel):                      # [B, T, M]
        x = self.conv(mel.unsqueeze(1))          # [B, C, T, M']
        b, c, t, m = x.shape
        return self.proj(x.permute(0, 2, 1, 3).reshape(b, t, c * m))   # [B, T, D]


class ResBlock(nn.Module):
    def __init__(self, d: int, dilation: int, kernel: int = 5, dropout: float = 0.0):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.n1 = nn.BatchNorm1d(d)
        self.c1 = nn.Conv1d(d, d, kernel, padding=pad, dilation=dilation)
        self.n2 = nn.BatchNorm1d(d)
        self.c2 = nn.Conv1d(d, d, kernel, padding=pad, dilation=dilation)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                        # [B, D, T]
        h = self.c1(F.gelu(self.n1(x)))
        h = self.c2(self.drop(F.gelu(self.n2(h))))
        return x + h


class LocalPath(nn.Module):
    def __init__(self, d: int, dilations, kernel: int = 5):
        super().__init__()
        self.blocks = nn.ModuleList(ResBlock(d, k, kernel) for k in dilations)
        self.norm = nn.BatchNorm1d(d)

    def forward(self, x):                        # [B, T, D]
        h = x.transpose(1, 2)
        for blk in self.blocks:
            h = blk(h)
        return F.gelu(self.norm(h)).transpose(1, 2)


def receptive_field_frames(dilations, kernel: int = 5) -> int:
    """One-sided receptive field of a local path, in frames."""
    return sum(2 * (kernel - 1) // 2 * d for d in dilations)


NE_S_DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128, 1, 4)
NE_M_DILATIONS = (1, 2, 4, 8, 16, 32, 1, 2)


class NES(nn.Module):
    """NE-S: stem to D 160, ten residual blocks (receptive field about ±10 s)."""

    def __init__(self, d: int = 160, n_mels: int = 128, dilations=NE_S_DILATIONS):
        super().__init__()
        self.d = d
        self.stem = Stem(d, n_mels)
        self.local = LocalPath(d, dilations)

    def forward(self, mel, song=None):
        return self.local(self.stem(mel))


def _rope(x, base: float = 10000.0):
    """Rotary position embedding over the last dimension of [B, H, T, Dh]."""
    t, dh = x.shape[-2], x.shape[-1]
    half = dh // 2
    freqs = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = torch.arange(t, device=x.device, dtype=torch.float32)[:, None] * freqs[None, :]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:2 * half]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos, x[..., 2 * half:]], dim=-1)


class Attention(nn.Module):
    def __init__(self, d: int, heads: int, d_kv: int | None = None, rope: bool = False):
        super().__init__()
        self.h, self.rope = heads, rope
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d_kv or d, 2 * d)
        self.o = nn.Linear(d, d)

    def forward(self, x, ctx=None):
        ctx = x if ctx is None else ctx
        b, t, d = x.shape
        q = self.q(x).view(b, t, self.h, -1).transpose(1, 2)
        k, v = self.kv(ctx).view(b, ctx.shape[1], 2, self.h, -1).permute(2, 0, 3, 1, 4)
        if self.rope:
            q, k = _rope(q), _rope(k)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.o(y.transpose(1, 2).reshape(b, t, d))


class TBlock(nn.Module):
    def __init__(self, d: int, heads: int, d_mem: int | None = None, ffn: int = 4, rope: bool = True):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.att = Attention(d, heads, rope=rope)
        self.cross = None
        if d_mem is not None:
            self.nc = nn.LayerNorm(d)
            self.cross = Attention(d, heads, d_kv=d_mem)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ffn * d), nn.GELU(), nn.Linear(ffn * d, d))

    def forward(self, x, mem=None):
        x = x + self.att(self.n1(x))
        if self.cross is not None and mem is not None:
            x = x + self.cross(self.nc(x), mem)
        return x + self.ff(self.n2(x))


class CrossBlock(nn.Module):
    def __init__(self, d: int, heads: int, d_kv: int, ffn: int = 4):
        super().__init__()
        self.nc = nn.LayerNorm(d)
        self.cross = Attention(d, heads, d_kv=d_kv)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ffn * d), nn.GELU(), nn.Linear(ffn * d, d))

    def forward(self, x, ctx):
        x = x + self.cross(self.nc(x), ctx)
        return x + self.ff(self.n2(x))


class SongMemory(nn.Module):
    """The whole song at 6.25 fps, read by ``n_latents`` learned latents."""

    def __init__(self, d_in: int, d: int = 256, n_latents: int = 48, heads: int = 8):
        super().__init__()
        self.proj = nn.Linear(d_in, d)
        self.latents = nn.Parameter(torch.randn(n_latents, d) * 0.02)
        self.read = nn.ModuleList(CrossBlock(d, heads, d) for _ in range(2))
        self.self_ = nn.ModuleList(TBlock(d, heads, rope=False) for _ in range(2))

    def forward(self, song):                     # [B, S, d_in] at 6.25 fps
        ctx = self.proj(song)
        z = self.latents.unsqueeze(0).expand(song.shape[0], -1, -1)
        for blk in self.read:
            z = blk(z, ctx)
        for blk in self.self_:
            z = blk(z)
        return z


class NEM(nn.Module):
    """NE-M: local path (D 256, ±2.6 s), mid path at 25 fps with song memory, up path to 100 fps."""

    def __init__(self, d: int = 256, d_mid: int = 320, mid_layers: int = 6, heads: int = 8, d_mem: int = 256,
                 n_latents: int = 48, d_up: int = 128, n_mels: int = 128, dilations=NE_M_DILATIONS):
        super().__init__()
        self.d = d_up
        self.stem = Stem(d, n_mels)
        self.local = LocalPath(d, dilations)
        self.down = nn.Sequential(nn.Conv1d(d, d_mid, 4, stride=4), nn.GELU())
        self.memory = SongMemory(n_mels, d_mem, n_latents, heads)
        self.mid = nn.ModuleList(TBlock(d_mid, heads, d_mem=d_mem) for _ in range(mid_layers))
        self.up = nn.ConvTranspose1d(d_mid, d_up, 4, stride=4)
        self.skip = nn.Linear(d, d_up)
        self.up_blocks = nn.ModuleList(ResBlock(d_up, 1, 5) for _ in range(2))
        self.norm = nn.BatchNorm1d(d_up)

    def forward(self, mel, song=None):
        """``song``: the whole song's standardised log-Mel pooled to 6.25 fps, [B, S, 128]."""
        loc = self.local(self.stem(mel))                     # [B, T, d]
        h = self.down(loc.transpose(1, 2)).transpose(1, 2)   # [B, T/4, d_mid]
        mem = self.memory(song) if song is not None else None
        for blk in self.mid:
            h = blk(h, mem)
        u = self.up(h.transpose(1, 2))[:, :, :loc.shape[1]]  # [B, d_up, T]
        u = u + self.skip(loc).transpose(1, 2)
        for blk in self.up_blocks:
            u = blk(u)
        return F.gelu(self.norm(u)).transpose(1, 2)


def n_params(module) -> int:
    return sum(p.numel() for p in module.parameters())
