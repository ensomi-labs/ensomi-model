"""Head-mask reweighting that preserves conditional action choices.

Every action with the same set of note-head lanes receives the same score shift.
A frozen base therefore retains P(action | head mask, state), including head
kinds and releases, while the new module can change routing and chord size.
"""
import torch
from torch import nn

from .contract import ROW_ACTIONS


class HeadRouting(nn.Module):
    def __init__(self, hand_width, hidden):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(2 * hand_width, hidden), nn.GELU(),
                                   nn.Linear(hidden, 16, bias=False))
        nn.init.zeros_(self.score[-1].weight)
        mirror = [sum(((mask >> c) & 1) << (3 - c) for c in range(4)) for mask in range(16)]
        masks = [sum(1 << c for c, action in enumerate(row) if action in (1, 2)) for row in ROW_ACTIONS]
        self.register_buffer('mirror', torch.tensor(mirror), persistent=False)
        self.register_buffer('masks', torch.tensor(masks), persistent=False)

    def forward(self, hands, onsets, candidate_indices=None):
        direct = self.score(torch.cat((hands[:, 0], hands[:, 1]), -1))
        reflected = self.score(torch.cat((hands[:, 1], hands[:, 0]), -1))[:, self.mirror]
        masks = self.masks if candidate_indices is None else self.masks[candidate_indices]
        scores = ((direct + reflected) / 2)[:, masks]
        # Exact zero at R, avoiding even a common floating-point logit shift.
        return torch.where(onsets[:, None], scores, torch.zeros_like(scores))


class ReleaseRouting(nn.Module):
    """Reweight release sets while preserving conditional head and kind choices."""
    def __init__(self, hand_width, hidden):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(2 * hand_width, hidden), nn.GELU(),
                                   nn.Linear(hidden, 16, bias=False))
        nn.init.zeros_(self.score[-1].weight)
        mirror = [sum(((mask >> c) & 1) << (3 - c) for c in range(4)) for mask in range(16)]
        masks = [sum(1 << c for c, action in enumerate(row) if action == 3) for row in ROW_ACTIONS]
        self.register_buffer('mirror', torch.tensor(mirror), persistent=False)
        self.register_buffer('masks', torch.tensor(masks), persistent=False)

    def forward(self, hands, occupied, candidate_indices=None):
        direct = self.score(torch.cat((hands[:, 0], hands[:, 1]), -1))
        reflected = self.score(torch.cat((hands[:, 1], hands[:, 0]), -1))[:, self.mirror]
        masks = self.masks if candidate_indices is None else self.masks[candidate_indices]
        scores = ((direct + reflected) / 2)[:, masks]
        # With no held lane, all legal actions have the same empty release set.
        # Omit even that common shift to retain exact no-hold probabilities.
        return torch.where(occupied[:, None], scores, torch.zeros_like(scores))
