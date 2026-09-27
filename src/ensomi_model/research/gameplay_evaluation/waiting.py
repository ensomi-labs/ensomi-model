"""Next-event diagnostics on a fixed prefix's no-event branch.

These laws measure timing sensitivity, not expected event counts after renewed
history, musical quality or playability. Each array position is one native ms.
"""
from dataclasses import dataclass

import numpy as np

from ..scoped_style_modeling.dataset import ContractError


@dataclass(frozen=True)
class WaitingLaw:
    event_mass: np.ndarray
    survival: np.ndarray

    @property
    def restricted_mean_ms(self):
        """E[min(next-event wait, horizon)], including right-censored mass."""
        return float(1+self.survival[:-1].sum())

    def report(self, *, horizons_ms=(20,50,100,250,500,1000,2000)):
        reached=np.flatnonzero(self.survival<=.5)
        return dict(horizon_ms=len(self.survival),
            event_probability=float(self.event_mass.sum()),
            censored_probability=float(self.survival[-1]),
            restricted_mean_wait_ms=self.restricted_mean_ms,
            median_wait_ms=int(reached[0])+1 if len(reached) else None,
            survival={str(t):float(self.survival[t-1]) for t in horizons_ms if 0<t<=len(self.survival)})


def first_event_law(logits, valid, *, forced=None):
    """Resolve native Bernoulli hazards, retaining a distinct no-event atom.

    Inputs cover every ms after the observation clock through the censoring
    horizon. Invalid clocks have zero hazard; valid forced clocks have hazard
    one. Ignored nonfinite logits do not contaminate survival. The caller must
    supply the actual conditional hazards: a separately conditioned release
    wait cannot be reconstructed from its unconditioned logits alone.
    """
    logits=np.asarray(logits,dtype=np.float64)
    valid=np.asarray(valid)
    forced=np.zeros_like(valid) if forced is None else np.asarray(forced)
    if (logits.ndim!=1 or not len(logits) or valid.shape!=logits.shape or
            forced.shape!=logits.shape or valid.dtype!=bool or forced.dtype!=bool):
        raise ContractError('Waiting laws require a nonempty native-ms vector and aligned boolean masks')
    learned=valid&~forced
    if np.isnan(logits[learned]).any():
        raise ContractError('Valid learned hazards cannot be NaN')
    mass=np.where(learned,np.logaddexp(0,np.where(learned,logits,0)),0.)
    mass=np.where(valid&forced,np.inf,mass)
    survival=np.exp(-np.cumsum(mass))
    event=np.r_[1.,survival[:-1]]-survival
    return WaitingLaw(event,survival)


def compare_waiting_laws(reference,candidate):
    """Compare equal-horizon, equally clocked first-event distributions.

    Total variation distinguishes a final-ms event from right censoring.
    Wasserstein distance concerns min(wait,horizon), merging those two outcomes
    at the horizon; it has ms units. Positive mean change means a longer wait.
    """
    if reference.survival.shape!=candidate.survival.shape:
        raise ContractError('Paired waiting laws require the same native-ms horizon')
    a,b=reference,candidate
    return dict(total_variation=float((np.abs(a.event_mass-b.event_mass).sum()+
                                      abs(a.survival[-1]-b.survival[-1]))/2),
        restricted_wasserstein_ms=float(np.abs(a.survival[:-1]-b.survival[:-1]).sum()),
        restricted_mean_change_ms=b.restricted_mean_ms-a.restricted_mean_ms,
        restricted_mean_relative_change=b.restricted_mean_ms/a.restricted_mean_ms-1)
