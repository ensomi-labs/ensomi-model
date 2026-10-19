"""Rung-1 heads on the shared encoder, their losses and their probe activations.

- `FrameHeads` (family F1, arm A1): beat, half-beat and downbeat logits, each
  with a sub-frame offset (``OFFSET_MS * tanh``), and the level class.
- `PhaseHeads` (family F2, arm A2): beat phase as a two-component von Mises
  mixture, log beat period with its sigma, bar phase with one concentration,
  and the level class.

Losses follow design §2.1 and §2.2 (weighted BCE with the widened targets,
Huber on offsets within a frame and a half of a label, cross-entropy on level;
mixture von Mises NLL, Gaussian NLL, von Mises NLL). ``activations`` gives the
decoder-free activations the probe reads (design §5.1 M1): A1's beat and
downbeat probabilities; A2's ``kappa (1 + cos mu) / 2`` of the dominant
component, its bar analogue, and its predicted period.
"""
from __future__ import annotations

import math

from .._torch import F, nn, torch

OFFSET_MS = 15.0
N_LEVEL = 4


class FrameHeads(nn.Module):
    kind = 'A1'

    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, 3 + 3 + N_LEVEL)

    def forward(self, h):
        y = self.lin(h)
        return dict(logits=y[..., 0:3], offset=OFFSET_MS * torch.tanh(y[..., 3:6]), level=y[..., 6:6 + N_LEVEL])

    @staticmethod
    def loss(out, tg, pos_weight):
        w = tg['weight']
        total, parts = 0.0, {}
        for i, name in enumerate(('beat', 'half', 'down')):
            pw = torch.as_tensor(pos_weight[name], device=w.device)
            bce = F.binary_cross_entropy_with_logits(out['logits'][..., i], tg[name], reduction='none', pos_weight=pw)
            l = (bce * w).sum() / w.sum().clamp_min(1.0)
            om = tg[name + '_offmask'].float() * (w > 0)
            hub = F.huber_loss(out['offset'][..., i] / 5.0, tg[name + '_off'] / 5.0, reduction='none', delta=1.0)
            lo = (hub * om).sum() / om.sum().clamp_min(1.0)
            parts[name] = float(l.detach())
            parts[name + '_off'] = float(lo.detach())
            total = total + l + 0.5 * lo
        lv = tg['level']
        ce = F.cross_entropy(out['level'].reshape(-1, N_LEVEL), lv.reshape(-1).clamp_min(0), reduction='none')
        m = (lv.reshape(-1) >= 0).float()
        lce = (ce * m).sum() / m.sum().clamp_min(1.0)
        parts['level'] = float(lce.detach())
        total = total + 0.5 * lce
        return total, parts

    @staticmethod
    def activations(out) -> dict:
        p = torch.sigmoid(out['logits'])
        return dict(beat=p[..., 0], half=p[..., 1], down=p[..., 2], beat_off=out['offset'][..., 0])


_I0_SMALL = (1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.0360768, 0.0045813)
_I0_LARGE = (0.39894228, 0.01328592, 0.00225319, -0.00157565, 0.00916281, -0.02057706, 0.02635537,
             -0.01647633, 0.00392377)


def _poly(x, coeffs):
    out = coeffs[-1] + 0.0 * x
    for c in reversed(coeffs[:-1]):
        out = out * x + c
    return out


def _log_i0(kappa, xp=None):
    """log I0(kappa) for kappa >= 0 from Abramowitz and Stegun 9.8.1/9.8.2 (relative error < 2e-7).

    Elementwise arithmetic only, so it runs on any torch device (``torch.special.i0e`` may lack an MPS
    kernel); ``xp`` = numpy evaluates the same formula for the tests.
    """
    xp = torch if xp is None else xp
    small = kappa <= 3.75
    k_lo = xp.where(small, kappa, 3.75)
    k_hi = xp.where(small, 3.75, kappa)
    lo = xp.log(_poly((k_lo / 3.75) ** 2, _I0_SMALL))
    hi = k_hi - 0.5 * xp.log(k_hi) + xp.log(_poly(3.75 / k_hi, _I0_LARGE))
    return xp.where(small, lo, hi)


def _vm_logpdf(x, c, s, kappa):
    """log von Mises density at angle ``x`` with mean direction (c, s) (unnormalised vector)."""
    norm = torch.sqrt(c * c + s * s).clamp_min(1e-6)
    cos_d = (c * torch.cos(x) + s * torch.sin(x)) / norm
    return kappa * cos_d - _log_i0(kappa) - math.log(2 * math.pi)


class PhaseHeads(nn.Module):
    kind = 'A2'

    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, 16)

    def forward(self, h):
        y = self.lin(h)
        kappa = F.softplus(y[..., 5:7]) + 1e-3
        return dict(w=y[..., 0], c1=y[..., 1], s1=y[..., 2], c2=y[..., 3], s2=y[..., 4], k1=kappa[..., 0],
                    k2=kappa[..., 1], logp=y[..., 7], logp_sigma=F.softplus(y[..., 8]) + 1e-3,
                    bc=y[..., 9], bs=y[..., 10], bk=F.softplus(y[..., 11]) + 1e-3, level=y[..., 12:16])

    @staticmethod
    def loss(out, tg, pos_weight=None):
        w = tg['weight']
        x = 2 * math.pi * tg['phase']
        lw = F.logsigmoid(out['w'])
        lw2 = F.logsigmoid(-out['w'])
        lp = torch.logaddexp(lw + _vm_logpdf(x, out['c1'], out['s1'], out['k1']),
                             lw2 + _vm_logpdf(x, out['c2'], out['s2'], out['k2']))
        m = tg['phase_mask'].float() * w
        l_phase = -(lp * m).sum() / m.sum().clamp_min(1.0)
        sig = out['logp_sigma']
        gnll = 0.5 * ((out['logp'] - tg['logp']) / sig) ** 2 + torch.log(sig)
        l_per = (gnll * m).sum() / m.sum().clamp_min(1.0)
        xb = 2 * math.pi * tg['bar_phase']
        mb = tg['bar_mask'].float() * w
        l_bar = -(_vm_logpdf(xb, out['bc'], out['bs'], out['bk']) * mb).sum() / mb.sum().clamp_min(1.0)
        lv = tg['level']
        ce = F.cross_entropy(out['level'].reshape(-1, N_LEVEL), lv.reshape(-1).clamp_min(0), reduction='none')
        ml = (lv.reshape(-1) >= 0).float()
        l_lev = (ce * ml).sum() / ml.sum().clamp_min(1.0)
        total = l_phase + 0.5 * l_per + 0.5 * l_bar + 0.5 * l_lev
        return total, dict(phase=float(l_phase.detach()), period=float(l_per.detach()), bar=float(l_bar.detach()),
                           level=float(l_lev.detach()))

    @staticmethod
    def activations(out) -> dict:
        first = out['w'] >= 0
        c = torch.where(first, out['c1'], out['c2'])
        s = torch.where(first, out['s1'], out['s2'])
        k = torch.where(first, out['k1'], out['k2'])
        cos_mu = c / torch.sqrt(c * c + s * s).clamp_min(1e-6)
        bcos = out['bc'] / torch.sqrt(out['bc'] ** 2 + out['bs'] ** 2).clamp_min(1e-6)
        phase = torch.remainder(torch.atan2(s, c) / (2 * math.pi), 1.0)
        return dict(beat=k * (1 + cos_mu) / 2, down=out['bk'] * (1 + bcos) / 2, period_ms=1000.0 * torch.exp(out['logp']),
                    phase=phase, kappa=k)


HEADS = dict(A1=FrameHeads, A2=PhaseHeads)


class Model(nn.Module):
    def __init__(self, encoder, arm: str):
        super().__init__()
        self.encoder = encoder
        self.heads = HEADS[arm](encoder.d)
        self.arm = arm

    def forward(self, mel, song=None):
        return self.heads(self.encoder(mel, song))
