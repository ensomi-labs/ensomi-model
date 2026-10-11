"""The CE loss of plan v4 section 6, with the two training phases of plan v5: uniform
per-decision CE with fixed divisors, inverse-probability weights on natural decisions,
condition-scoped terms owned by visibility, and the KL term of a KL-held phase C.

Per scored decision j, l_j = -(log P(A_j) + log Q(U_j)). For a window drawn with weight u:

    L_base = (1 / N_bar)      sum over j in B of  w_j l_j,      w_j = u if V_j is empty, else 1
    L_ln   = (1 / N_bar_ln)   sum over decisions reading an LN interval of
                              -log P(tap-vs-LN split | head mask, release types)
    L_star = (1 / N_bar_star) sum over decisions reading a difficulty interval of l_j
    L_kl   = (1 / N_bar)      sum over j in D of  w_j KL_j
    L      = L_base + lambda_ln L_ln + lambda_star L_star + kl_weight L_kl

B is every scored decision (phase N; phase C with ``natural_ce``) or only the decisions with
V_j non-empty (phase C otherwise: natural decisions are held by the frozen base or by the KL
term, not fitted again). D is the natural decisions (``kl_decisions='natural'``) or every
decision (``'all'``). KL_j compares the model with the frozen phase-N reference on decision j:
the action factor's KL over the legal actions, plus each release factor's KL over its
candidates on the teacher-forced context, averaged over the two orientations. ``forward`` is
KL(reference || model), ``reverse`` KL(model || reference). The reference reads no condition.

Omega_kappa (the factors a kind's term scores) is exactly the factors of decisions with an
interval of that kind in V_k (rule L), so a term never scores a factor that cannot read its
condition. The divisors are fixed expected counts per batch (``draw_sim``), so a window's
terms do not depend on the other windows of its batch, and backpropagating window by window
gives the batch gradient exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .common import ContractError

KL_DIRECTIONS = ('forward', 'reverse')
KL_DECISIONS = ('natural', 'all')


@dataclass(frozen=True)
class LossConfig:
    n_bar: float
    n_bar_ln: float
    n_bar_star: float
    lambda_ln: float = 0.0
    lambda_star: float = 0.0
    natural_ce: bool = True       # L_base scores natural decisions too (False: decisions in V only)
    kl_weight: float = 0.0

    def __post_init__(self):
        for name in ('n_bar', 'n_bar_ln', 'n_bar_star'):
            if not getattr(self, name) or getattr(self, name) <= 0:
                raise ContractError(f'{name} must be a positive expected count from draw_sim')
        if self.kl_weight < 0:
            raise ContractError('kl_weight is nonnegative')


def masks(out):
    """(natural, omega_ln, omega_star) decision masks of a WindowOut."""
    vis = out.visible
    return ~vis.any(1), vis[:, 0], vis[:, 1]


def window_terms(out, weight: float = 1.0) -> dict:
    """Unnormalised sums of one window: base (weighted; all decisions and decisions in V only),
    LN governed split, difficulty whole decision."""
    natural, ln, star = masks(out)
    ell = -(out.action + out.release)
    nat_t = torch.as_tensor(natural, device=ell.device)
    w = torch.where(nat_t, ell.new_tensor(float(weight)), ell.new_tensor(1.0))
    ln_t = torch.as_tensor(ln, device=ell.device)
    star_t = torch.as_tensor(star, device=ell.device)
    zero = ell.new_zeros(())
    return dict(base=(w * ell).sum(), base_conditioned=ell[~nat_t].sum() if (~natural).any() else zero,
                ln=(-out.governed_ln * ln_t).sum() if ln.any() else zero,
                star=(ell * star_t).sum() if star.any() else zero, ln_whole=(ell * ln_t).sum() if ln.any() else zero,
                weights=float(w.detach().sum()), decisions=len(ell), natural=int(natural.sum()),
                factors_ln=int(out.factors[ln].sum()), factors_star=int(out.factors[star].sum()),
                ln_decisions=int(ln.sum()), star_decisions=int(star.sum()))


def _kl_terms(ref, mod, direction):
    """Elementwise KL summands from finite log-probabilities of the reference and the model."""
    if direction == 'forward':
        return ref.exp() * (ref - mod)
    return mod.exp() * (mod - ref)


def decision_kl(out, ref, direction: str) -> torch.Tensor:
    """[m] KL_j between the model's window ``out`` and the reference's ``ref`` on the same decisions
    (both from ``window(..., pairs=True)``)."""
    if direction not in KL_DIRECTIONS:
        raise ContractError(f'kl_direction must be one of {KL_DIRECTIONS}')
    if out.pair_logp is None or ref.pair_logp is None:
        raise ContractError('decision_kl needs windows scored with pairs=True')
    if not (np.array_equal(out.ks, ref.ks) and np.array_equal(out.pair_rows, ref.pair_rows)):
        raise ContractError('The model and its reference scored different decisions or factors')
    legal = torch.isfinite(ref.logp)
    zero = torch.zeros_like(out.logp)
    mod = torch.where(legal, out.logp, zero)                     # masked actions enter as 0, never as -inf
    r = torch.where(legal, ref.logp.to(out.logp.dtype), zero)
    action = torch.where(legal, _kl_terms(r, mod, direction), zero).sum(-1)
    if not len(out.pair_rows):
        return action
    rows = torch.as_tensor(out.pair_rows, device=out.logp.device)
    pairs = _kl_terms(ref.pair_logp.to(out.pair_logp.dtype), out.pair_logp, direction)
    return action + 0.5 * action.new_zeros(len(action)).index_add(0, rows, pairs)   # mean over the orientations


def window_kl(out, ref, weight: float, decisions: str, direction: str) -> torch.Tensor:
    """Unnormalised sum over D of w_j KL_j for one window."""
    if decisions not in KL_DECISIONS:
        raise ContractError(f'kl_decisions must be one of {KL_DECISIONS}')
    kl = decision_kl(out, ref, direction)
    natural, _, _ = masks(out)
    nat_t = torch.as_tensor(natural, device=kl.device)
    w = torch.where(nat_t, kl.new_tensor(float(weight)), kl.new_tensor(1.0))
    if decisions == 'natural':
        w = torch.where(nat_t, w, kl.new_tensor(0.0))
    return (w * kl).sum()


def window_loss(terms: dict, cfg: LossConfig, kl=None):
    """This window's share of the batch loss and its named parts (all on the fixed divisors)."""
    base = (terms['base'] if cfg.natural_ce else terms['base_conditioned']) / cfg.n_bar
    ln = terms['ln'] / cfg.n_bar_ln
    star = terms['star'] / cfg.n_bar_star
    total = base + cfg.lambda_ln * ln + cfg.lambda_star * star
    parts = dict(base=base, ln=ln, star=star)
    if kl is not None:
        parts['kl'] = kl / cfg.n_bar
        total = total + cfg.kl_weight * parts['kl']
    return total, parts
