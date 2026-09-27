"""Nonlinear fine-feature summaries at several scales with full-audio attention."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class AudioPyramid(nn.Module):
    def __init__(self, fine_width, output_width, width=256, layers=3, heads=4):
        super().__init__()
        self.project = nn.Linear(6*fine_width, width)
        self.layers = nn.ModuleList(nn.TransformerEncoderLayer(width, heads, 4*width,
            dropout=0., activation='gelu', batch_first=True, norm_first=True) for _ in range(layers))
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, output_width, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, fine, valid):
        batch, frames, channels = fine.shape
        padding = (-frames) % 50
        mask = F.pad(valid, (0, padding)).reshape(batch, -1, 50)
        counts = mask.sum(-1)
        values = F.pad(fine*valid[..., None], (0, 0, 0, padding)).reshape(batch, -1, 50, channels)
        mean = values.sum(-2)/counts.clamp_min(1)[..., None]
        maximum = values.masked_fill(~mask[..., None], -torch.inf).amax(-2)
        base = torch.cat((mean, torch.where(counts[..., None]>0, maximum, 0.)), -1)
        pieces = [base]
        for cells in (4, 16):
            pad = (cells//2, cells-1-cells//2)
            weighted = F.avg_pool1d(F.pad((base*counts[..., None]).transpose(1, 2), pad), cells, stride=1)
            denominator = F.avg_pool1d(F.pad(counts[:, None].to(base), pad), cells, stride=1)
            pieces.append((weighted/denominator.clamp_min(1e-9)).transpose(1, 2))
        tokens = self.project(torch.cat(pieces, -1))
        time = .25+.5*torch.arange(tokens.shape[1], device=tokens.device, dtype=tokens.dtype)
        frequency = torch.exp(-math.log(10000)*torch.arange(0,tokens.shape[-1],2,
            device=tokens.device,dtype=tokens.dtype)/tokens.shape[-1])
        position = torch.stack(((time[:, None]*frequency).sin(),(time[:, None]*frequency).cos()),-1).flatten(-2)
        real = counts > 0
        tokens = (tokens+position)*real[..., None]
        for layer in self.layers:
            tokens = layer(tokens,src_key_padding_mask=~real)*real[..., None]
        return self.output(self.norm(tokens))*real[..., None]
