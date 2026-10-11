"""Expected difficulty proxies of a scope under the model's per-row action distributions (stage 2).

``expected_proxies`` returns the four ``features.star_proxies`` statistics of the whole scope
(mean chord size / 4, same-lane repeat rate, held-lane occupancy, LN share) on an own-sampled
history, with every decision that reads the scope (rule L) entering through its expected head,
LN-head and same-lane-repeat counts under exp(log P(a)), and every other decision of the scope
through its sampled action. Held-lane occupancy is history and carries no gradient.
"""
from __future__ import annotations

import numpy as np
import torch

from .common import ACTIONS
from .model import held_pattern


def _tables():
    heads = np.zeros((16, len(ACTIONS)))
    lns = np.zeros((16, len(ACTIONS)))
    mask = np.zeros((16, len(ACTIONS), 4))
    for p in range(16):
        held = np.array([(p >> lane) & 1 for lane in range(4)], dtype=bool)
        h = np.where(held[None], ACTIONS >= 3, (ACTIONS == 1) | (ACTIONS == 2))
        l = np.where(held[None], ACTIONS == 4, ACTIONS == 2)
        heads[p], lns[p], mask[p] = h.sum(1), l.sum(1), h
    return heads, lns, mask


HEADS, LNS, HEAD_MASK = _tables()


def expected_proxies(chart, out, iv):
    """[4] tensor over the scope's head rows ``out.ks`` (decisions k0..k1-1 of ``chart``)."""
    ks = out.ks
    d = chart.derived()
    pat = held_pattern(d.held[ks])
    probs = out.logp.exp()
    dt = probs.dtype
    reads = torch.as_tensor(out.visible[:, iv.kind] & ~out.eos, device=probs.device)
    t = lambda x: torch.as_tensor(x, dtype=dt, device=probs.device)  # noqa: E731
    e_heads = (probs * t(HEADS[pat])).sum(1)
    e_lns = (probs * t(LNS[pat])).sum(1)
    idx = np.array([((a[0] * 5 + a[1]) * 5 + a[2]) * 5 + a[3] for a in chart.actions[ks].astype(np.int64)])
    r_heads = t(HEADS[pat, idx])
    r_lns = t(LNS[pat, idx])
    heads = torch.where(reads, e_heads, r_heads)
    lns = torch.where(reads, e_lns, r_lns)
    prev_mask = np.zeros((len(ks), 4))
    if len(ks) > 1:
        pp = held_pattern(d.held[ks[:-1]])
        prev_mask[1:] = HEAD_MASK[pp, idx[:-1]]
    overlap = np.einsum('mal,ml->ma', HEAD_MASK[pat], prev_mask)
    e_rep = (probs * t(overlap)).sum(1)
    r_rep = t(overlap[np.arange(len(ks)), idx])
    rep = torch.where(reads, e_rep, r_rep)
    rows = len(ks)
    held = float(d.held[ks].sum())
    total = heads.sum()
    later = heads[1:].sum() if rows > 1 else heads.new_zeros(())
    zero = heads.new_zeros(())
    return torch.stack((total / rows / 4.0, rep[1:].sum() / later if rows > 1 else zero,
                        heads.new_tensor(held / (4.0 * rows)), lns.sum() / total))
