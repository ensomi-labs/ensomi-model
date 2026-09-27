"""Factual scoped LN accounting and an optional learned R1 allocation law.

Declared-interval counts include overrides; effective-owner counts do not.
Neither is a prefix quota. Request identities survive interruption/resumption,
and only the current request's facts enter the allocation readout.
"""
from bisect import bisect_right
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import nn

from ..planned_audio_continuation.counts import ROW_COUNTS
from ..scoped_style_modeling.dataset import ContractError


FEATURES = (
    'declared_heads_asinh32', 'declared_LNs_asinh32',
    'declared_fraction', 'declared_nonempty',
    'owned_heads_asinh32', 'owned_LNs_asinh32',
    'owned_fraction', 'owned_nonempty',
    'owned_elapsed_fraction', 'owned_remaining_seconds_asinh',
)


@dataclass(frozen=True)
class LnScopeProgram:
    requests: tuple
    intervals: tuple
    starts: tuple

    @classmethod
    def build(cls, controls):
        requests = tuple((s.start_ms, s.end_ms, s.ln_fraction)
                         for s in controls.spans if s.ln_fraction is not None)
        edges = sorted({v for a, b, _ in requests for v in (a, b)})
        intervals = []
        for a, b in zip(edges, edges[1:]):
            owner = next((i for i in range(len(requests)-1, -1, -1)
                          if requests[i][0] <= a < requests[i][1]), None)
            if owner is not None:
                intervals.append((a, b, owner))
        return cls(requests, tuple(intervals), tuple(v[0] for v in intervals))

    def owner(self, now):
        i = bisect_right(self.starts, now)-1
        return (self.intervals[i][2] if i >= 0 and now < self.intervals[i][1]
                else None)


@dataclass(frozen=True)
class LnScopeState:
    program: LnScopeProgram
    declared: tuple
    owned: tuple

    @classmethod
    def empty(cls, controls):
        program = LnScopeProgram.build(controls)
        return cls(program, ((0, 0),)*len(program.requests), ((0, 0),)*len(program.requests))

    def update_controls(self, controls):
        """Extend announced requests without reassigning already observed rows."""
        program = LnScopeProgram.build(controls)
        n = len(self.program.requests)
        if program.requests[:n] != self.program.requests:
            raise ContractError('LN allocation updates must retain earlier requests')
        extra = ((0, 0),)*(len(program.requests)-n)
        return LnScopeState(program, self.declared+extra, self.owned+extra)

    def observe(self, row):
        heads = row.actions.count(1)+row.actions.count(2)
        longs = row.actions.count(2)
        if heads == 0:
            return self
        owner = self.program.owner(row.time_ms)
        declared = tuple((h+heads, l+longs) if a <= row.time_ms < b else (h, l)
                         for (h, l), (a, b, _) in zip(self.declared, self.program.requests))
        owned = tuple((h+heads, l+longs) if i == owner else (h, l)
                      for i, (h, l) in enumerate(self.owned))
        return replace(self, declared=declared, owned=owned)

    def features(self, now):
        owner = self.program.owner(now)
        values = np.zeros(len(FEATURES), np.float32)
        if owner is None:
            return values
        for offset, counts in ((0, self.declared[owner]), (4, self.owned[owner])):
            h, l = counts
            values[offset:offset+4] = (
                np.arcsinh(h/32), np.arcsinh(l/32), l/h if h else 0., float(h > 0))
        elapsed = remaining = 0
        for a, b, i in self.program.intervals:
            if i == owner:
                elapsed += max(0, min(now, b)-a)
                remaining += max(0, b-max(now, a))
        values[8:] = (elapsed/(elapsed+remaining), np.arcsinh(remaining/1000))
        return values


def features_before_rows(rows, controls, times):
    """Query factual counts strictly before each timestamp, in caller order.

    Rows must be the actual ordered source/generated trace. A current target
    row and future endpoints never contribute to its own observations.
    """
    times = np.asarray(times)
    result = np.zeros((len(times), len(FEATURES)), np.float32)
    state = LnScopeState.empty(controls)
    iterator = iter(rows)
    row = next(iterator, None)
    for index in np.argsort(times, kind='stable'):
        now = times[index]
        while row is not None and row.time_ms < now:
            state = state.observe(row)
            row = next(iterator, None)
        result[index] = state.features(now)
    return result


class ScopedLnAllocation(nn.Module):
    """Learn conditional LN-count odds; preserve neural head/release mass.

    Recovery preference applied later may change deployed family mass.
    Unknown LN requests are bitwise identity. The context arm has the same
    parameters and inputs except that accounting observations are zeroed.
    """
    def __init__(self, audio_width, context_width, preview_width, control_width,
                 known_index, mode, hidden=64):
        super().__init__()
        if mode not in ('context', 'progress'):
            raise ValueError('Scoped LN allocation requires context or progress')
        self.mode, self.known_index = mode, known_index
        width = audio_width+context_width+preview_width+control_width+len(FEATURES)
        self.readout = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        counts = torch.tensor(ROW_COUNTS)
        groups = 5*counts[:, 0]+counts[:, 2]
        self.register_buffer('groups', groups, persistent=False)
        self.register_buffer('members', torch.arange(25)[:, None] == groups, persistent=False)
        self.register_buffer('longs', counts[:, 1], persistent=False)

    def forward(self, log_probs, audio, context, preview, control, allocation_features):
        if allocation_features is None or allocation_features.shape != (len(log_probs), len(FEATURES)):
            raise ContractError('Scoped LN allocation requires factual pre-row observations')
        observed = (allocation_features if self.mode == 'progress'
                    else torch.zeros_like(allocation_features))
        joined = torch.cat((audio, context, preview, control, observed), -1)
        bias = self.readout(joined).squeeze(-1)
        known = control[:, self.known_index] > 0
        bias = torch.where(known, bias, 0.)
        active = (self.members[None] & torch.isfinite(log_probs[:, None])).any(-1)

        def normalizers(values):
            scores = values[:, None].masked_fill(~self.members[None], -torch.inf)
            return torch.where(active[..., None], scores, 0.).logsumexp(-1)

        tilted = log_probs+bias[:, None]*self.longs
        delta = normalizers(tilted)-normalizers(log_probs)
        result = tilted-delta[:, self.groups]
        return torch.where(known[:, None], result, log_probs)
