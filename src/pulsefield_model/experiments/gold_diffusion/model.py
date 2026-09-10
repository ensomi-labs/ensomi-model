from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import MASK, ROWS, replay


class Block(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(width, width)
        self.history_kv = nn.Linear(width, width * 2)
        self.self_qkv = nn.Linear(width, width * 3)
        self.out = nn.Linear(width, width)
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(3)])
        self.ff = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(width * 4, width), nn.Dropout(dropout))

    def heads_view(self, x):
        return x.reshape(*x.shape[:-1], self.heads, -1).transpose(1, 2)

    def forward(self, x, history, history_bias, target_bias, has_history):
        q = self.heads_view(self.query(self.norms[0](x)))
        key, value = self.history_kv(history).chunk(2, -1)
        read = F.scaled_dot_product_attention(q, self.heads_view(key), self.heads_view(value), attn_mask=history_bias)
        x = x + self.out(read.transpose(1, 2).flatten(2)) * has_history[:, None, None]
        q, k, v = [self.heads_view(t) for t in self.self_qkv(self.norms[1](x)).chunk(3, -1)]
        mixed = F.scaled_dot_product_attention(q, k, v, attn_mask=target_bias)
        x = x + self.out(mixed.transpose(1, 2).flatten(2))
        return x + self.ff(self.norms[2](x))


class Denoiser(nn.Module):
    def __init__(self, width=128, heads=4, layers=4, dropout=0.1):
        super().__init__()
        self.row = nn.Embedding(257, width, padding_idx=0)
        self.style = nn.Embedding(20, width)
        self.audio = nn.Sequential(nn.Conv1d(128, 64, 5, padding=2), nn.GELU(),
                                   nn.Conv1d(64, 64, 5, padding=2), nn.GELU(),
                                   nn.AvgPool1d(15, stride=13), nn.Flatten(), nn.Linear(192, width))
        self.noise = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))
        self.facts = nn.Linear(4 + heads + 1, width)
        self.blocks = nn.ModuleList([Block(width, heads, dropout) for _ in range(layers)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 255))
        self.register_buffer("scales", torch.logspace(-1, 1, heads))
        self.direction = nn.Parameter(torch.zeros(heads))

    def context(self, batch):
        b, k = batch["target"].shape
        audio = self.audio(batch["mel"].reshape(b * k, 41, 128).transpose(1, 2)).reshape(b, k, -1)
        history = self.row(batch["history"])
        delta = batch["times"][:, :, None] - batch["history_times"][:, None, :]
        hb = -torch.log1p(delta.clamp_min(0)[:, None] / self.scales[None, :, None, None])
        mass = torch.log1p((hb.exp() * batch["history_valid"][:, None, None]).sum(-1)).transpose(1, 2)
        hb = hb.masked_fill(~batch["history_valid"][:, None, None], -1e4)
        delta = batch["times"][:, :, None] - batch["times"][:, None, :]
        tb = (-torch.log1p(delta.abs()[:, None] / self.scales[None, :, None, None])
              + self.direction[None, :, None, None] * delta.sign()[:, None])
        tb = tb.masked_fill(~batch["valid"][:, None, None], -1e4)
        facts = torch.cat((batch["entry"][:, None].expand(-1, k, -1), mass,
                           torch.log1p(batch["times"].clamp_min(0))[:, :, None]), -1)
        return audio + self.facts(facts), history, hb, tb, batch["history_valid"].any(-1)

    def forward(self, noisy, s, batch, context=None):
        audio_facts, history, hb, tb, has_history = self.context(batch) if context is None else context
        style = self.style(batch["style"] + torch.arange(5, device=noisy.device) * 4).sum(1)
        noise = self.noise(torch.stack((s, torch.sin(s * torch.pi), torch.cos(s * torch.pi)), -1))
        x = self.row(noisy) + audio_facts + (style + noise)[:, None]
        for block in self.blocks:
            x = block(x, history, hb, tb, has_history)
        return self.output(x)


def masked_loss(model, batch, generator=None):
    s = torch.rand(len(batch["target"]), generator=generator).to(batch["target"].device).clamp_min(1e-5)
    mask = (torch.rand(batch["target"].shape, generator=generator).to(s.device) < s[:, None]) & batch["valid"]
    noisy = batch["target"].masked_fill(mask, MASK)
    logits = model(noisy, s, batch)
    ce = F.cross_entropy(logits.transpose(1, 2), (batch["target"] - 1).clamp_min(0), reduction="none")
    loss = ((ce * mask).sum(-1) / (s * batch["valid"].sum(-1))).mean()
    return loss


def transitions():
    result = np.full((16, 255), -1, dtype=np.int64)
    for state in range(16):
        for code in range(1, 256):
            try:
                result[state, code - 1] = replay([code], state)
            except ValueError:
                pass
    return result


TRANSITIONS = transitions()


def legal_path(logits, fixed, entry, exit_state, rng):
    """Sample the unary-logit product conditioned on LN boundaries and fixed rows.

    This is constrained decoding, not an exact unconstrained MDLM reverse kernel.
    """
    score = np.asarray(logits, dtype=np.float64).copy()
    for i, code in enumerate(fixed):
        if code != MASK:
            score[i, np.arange(255) != code - 1] = -np.inf
    backward = np.full((len(score) + 1, 16), -np.inf)
    backward[-1, exit_state] = 0
    safe = TRANSITIONS.clip(0)
    legal = TRANSITIONS >= 0
    for i in range(len(score) - 1, -1, -1):
        edges = score[i][None] + backward[i + 1, safe]
        edges[~legal] = -np.inf
        backward[i] = np.logaddexp.reduce(edges, axis=1)
        offset = np.max(backward[i])
        if np.isfinite(offset):
            backward[i] -= offset
    if not np.isfinite(backward[0, entry]):
        raise ValueError("No legal completion for the declared skeleton and LN boundaries")
    state, result = entry, []
    for i in range(len(score)):
        weights = score[i] + backward[i + 1, safe[state]]
        weights[~legal[state]] = -np.inf
        p = np.exp(weights - np.max(weights))
        action = int(rng.choice(255, p=p / p.sum()))
        result.append(action + 1)
        state = TRANSITIONS[state, action]
    return np.array(result, dtype=np.int64)


@torch.no_grad()
def sample(model, batch, entry, exit_state, steps, seed, temperature=1.0):
    if len(batch["target"]) != 1:
        raise ValueError("Sampling requires one unpadded scope")
    model.eval()
    rng = np.random.default_rng(seed)
    k = int(batch["valid"].sum())
    fixed = np.full(k, MASK, np.int64)
    order = rng.permutation(k)
    context = model.context(batch)
    for step in range(steps):
        s = torch.tensor([(fixed == MASK).mean()], dtype=torch.float32, device=batch["target"].device)
        noisy = torch.as_tensor(fixed[None], device=s.device)
        logits = model(noisy, s, batch, context)[0].cpu().numpy() / temperature
        proposal = legal_path(logits, fixed, entry, exit_state, rng)
        chosen = order[:int(np.ceil(k * (step + 1) / steps))]
        fixed[chosen] = proposal[chosen]
    if replay(fixed, entry) != exit_state:
        raise AssertionError("Sampler violated exit occupancy")
    return fixed
