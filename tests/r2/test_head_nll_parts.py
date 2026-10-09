import numpy as np
import torch

from ensomi_model.r2.common import ACTIONS, action_index
from ensomi_model.r2.model import head_nll_parts

from .helpers import random_decisions, random_skeleton, tiny_model, two_segment_grid
from .test_ln_level_model import SMALL


def test_head_nll_parts_sum_and_held_aware_categories():
    rng = np.random.default_rng(171)
    head, song = random_skeleton(rng, 30)
    chart = random_decisions(rng, (head, song, two_segment_grid()))
    net = tiny_model(torch.float64, **SMALL).eval()
    out = net.window(chart, 0, chart.n)
    held = chart.derived().held[:chart.n]
    target = torch.tensor([action_index(a) for a in chart.actions])
    count, lane, rest = head_nll_parts(out.logp, target, held, out.release)
    torch.testing.assert_close(count + lane + rest, -out.total)
    assert (count >= -1e-12).all() and (lane >= -1e-12).all() and (rest >= -1e-12).all()
    for k in range(chart.n):
        heads = np.where(held[k], ACTIONS >= 3, (ACTIONS == 1) | (ACTIONS == 2))
        mask = heads[target[k]]
        nc = heads.sum(1) == mask.sum()
        nm = (heads == mask).all(1)
        pcount = out.logp[k, nc].exp().sum()
        pmask = out.logp[k, nm].exp().sum()
        torch.testing.assert_close(count[k], -pcount.log())
        torch.testing.assert_close(lane[k], -(pmask / pcount).log())
