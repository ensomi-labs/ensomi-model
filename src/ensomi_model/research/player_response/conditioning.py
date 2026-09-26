"""Causal time-window observations supplied directly to the complete-row policy."""
import numpy as np
import torch
from torch import nn

from ..bounded_typed_continuation.features import RELATIVE_LANES
from .state import CommittedPlayState, DEFAULT_WINDOWS_MS

LANE_WIDTH = 3*len(DEFAULT_WINDOWS_MS)+6


def features_at(state, time_ms):
    """Return [column,feature] observations before a prospective event.

    Rates use asinh(Hz/4), held fractions remain in [0,1]. The final coordinates
    retain the previous two complete attack masks, asinh ages in seconds, and
    availability bits. No request or future endpoint enters these coordinates.
    Advancing the query does not mutate the committed state.
    """
    current = state.advance(time_ms)
    facts = current.window_facts()
    history = np.zeros((4, 6))
    for i, (time, mask) in enumerate(current.last_head_groups, 2-len(current.last_head_groups)):
        history[:, i] = [(mask >> column) & 1 for column in range(4)]
        history[:, 2+i] = np.arcsinh((time_ms-time)/1000)
        history[:, 4+i] = 1
    return np.concatenate((np.arcsinh(facts['attack_hz'].T/4),
        np.arcsinh(facts['release_hz'].T/4), facts['held_fraction'].T, history), -1).astype(np.float32)


def features_before_rows(rows, indices):
    """Replay complete rows once and gather possibly repeated pre-row queries."""
    indices = tuple(int(i) for i in indices)
    if not indices:
        return np.empty((0, 4, LANE_WIDTH), np.float32)
    wanted, values = set(indices), {}
    state = CommittedPlayState()
    for i, row in enumerate(rows):
        if i in wanted:
            values[i] = features_at(state, row.time_ms)
        if len(values) == len(wanted):
            break
        state = state.observe(row)
    return np.stack([values[i] for i in indices])


class PlayerCondition(nn.Module):
    """Zero-initialized, mirror-equivariant addition to R1's two hand contexts."""
    def __init__(self, hidden):
        super().__init__()
        self.projection = nn.Linear(4*LANE_WIDTH, hidden, bias=False)
        nn.init.zeros_(self.projection.weight)
        self.register_buffer('relative', torch.tensor(RELATIVE_LANES), persistent=False)

    def forward(self, features):
        return self.projection(features[:, self.relative].flatten(-2))
