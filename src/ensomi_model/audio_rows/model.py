"""The per-beat head generator and its input contract.

Inputs, per song: ``features`` [F, FEATURE_DIM] at 10 ms frames (``audio.features``: 128
standardised log-Mel bins, then the BeatThis beat and downbeat probabilities). Per canonical beat:
the frame span [start, end) of the beat, one frame per lattice slot ([nb, 2, 16], from the slot
times on the grid), the previous beat's decision token, and ``extra_inputs``: bar phase one-hot
(beat index mod 4), progress t/T at the beat start, local canonical BPM / 120 (as R2 reads it),
the standardised section label and its presence bit.

An encoder over frames (non-causal dilated convolutions, ±30 frames of context) gives a mean
vector over each beat's frames and one vector per lattice slot. A GRU over beats reads the beat
vector, the previous token and the extra inputs. Three factors per beat:

    lattice (duple or triple) | h;  count 0..16 | h, lattice;  slot mask | h, lattice, count, slot vectors.

Triple beats have 12 slots, so their counts above 12 get no probability. Generation
(``generate.decode``) and teacher forcing (``factor_losses``) use only the methods below; a
replacement model keeps these signatures.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .audio import FEATURE_DIM
from .lattice import MAX_COUNT, MAX_SLOTS, SLOTS

TOKEN_DIM = 2 + (MAX_COUNT + 1) + MAX_SLOTS     # previous beat: lattice, count, mask
EXTRA_DIM = 4 + 1 + 1 + 2                       # bar phase, progress, BPM / 120, label value, label presence
LABEL = 6                                       # ``extra_inputs`` label value; LABEL + 1, the last, its presence


@dataclass
class HeadConfig:
    enc: int = 64
    hidden: int = 128
    dilations: tuple = (1, 2, 4, 8)
    kernel: int = 5
    token: int = 32

    @classmethod
    def from_dict(cls, d):
        return cls(**dict(d, dilations=tuple(d['dilations'])))


def extra_inputs(beat_index, beat_ms, bpm, song_ms: float, label=None) -> torch.Tensor:
    """[nb, EXTRA_DIM] from global beat indices, beat start times, canonical BPM per beat and the
    standardised label per beat (None: absent)."""
    nb = len(beat_index)
    value, presence = (np.zeros(nb), np.zeros(nb)) if label is None else (np.asarray(label), np.ones(nb))
    bar = np.eye(4)[np.asarray(beat_index) % 4]
    return torch.from_numpy(np.column_stack((bar, np.asarray(beat_ms) / song_ms, np.asarray(bpm) / 120.0,
                                             value, presence)).astype(np.float32))


def token(lattice, count, mask) -> torch.Tensor:
    """[n, TOKEN_DIM] decision tokens from int lattice [n], int count [n], bool mask [n, 16]."""
    return torch.cat((F.one_hot(lattice.long(), 2).float(), F.one_hot(count.long(), MAX_COUNT + 1).float(),
                      mask.float()), -1)


class Encoder(nn.Module):
    def __init__(self, cfg: HeadConfig):
        super().__init__()
        self.inp = nn.Linear(FEATURE_DIM, cfg.enc)
        self.convs = nn.ModuleList([nn.Conv1d(cfg.enc, cfg.enc, cfg.kernel, dilation=d, padding=d * (cfg.kernel // 2))
                                    for d in cfg.dilations])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[F, FEATURE_DIM] -> [F, enc]."""
        h = F.gelu(self.inp(x)).T[None]
        for c in self.convs:
            h = h + F.gelu(c(h))
        return h[0].T


class HeadModel(nn.Module):
    def __init__(self, cfg: HeadConfig = HeadConfig()):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.token = nn.Linear(TOKEN_DIM, cfg.token)
        self.gru = nn.GRU(cfg.enc + cfg.token + EXTRA_DIM, cfg.hidden, batch_first=True)
        self.lattice_head = nn.Linear(cfg.hidden, 2)
        self.count_head = nn.Linear(cfg.hidden + 2, MAX_COUNT + 1)
        self.slot_head = nn.Sequential(nn.Linear(cfg.hidden + cfg.enc + 2 + MAX_COUNT + 1 + MAX_SLOTS, 128), nn.GELU(),
                                       nn.Linear(128, 1))

    def beat_inputs(self, feats: torch.Tensor, beat_frames: torch.Tensor, slot_frames: torch.Tensor):
        """feats [F, FEATURE_DIM]; beat_frames [nb, 2] frame span; slot_frames [nb, 2, 16] frame per slot.

        Returns the mean encoder vector per beat [nb, enc] and the slot vectors [nb, 2, 16, enc]."""
        enc = self.encoder(feats)
        cs = torch.cat((enc.new_zeros(1, enc.shape[1]), enc.cumsum(0)), 0)
        n = (beat_frames[:, 1] - beat_frames[:, 0]).clamp_min(1).to(enc.dtype)
        beat_vec = (cs[beat_frames[:, 1]] - cs[beat_frames[:, 0]]) / n[:, None]
        return beat_vec, enc[slot_frames]

    def run(self, beat_vec: torch.Tensor, prev_tokens: torch.Tensor, extra: torch.Tensor, h0=None):
        """GRU over beats: [nb, enc], [nb, TOKEN_DIM], [nb, EXTRA_DIM] -> [nb, hidden], final state."""
        out, h = self.gru(torch.cat((beat_vec, F.gelu(self.token(prev_tokens)), extra), -1)[None], h0)
        return out[0], h

    def lattice_logits(self, h):
        return self.lattice_head(h)

    def count_logits(self, h, lattice_onehot):
        logits = self.count_head(torch.cat((h, lattice_onehot), -1))
        limit = torch.where(lattice_onehot[..., 1] > 0.5, SLOTS[1], SLOTS[0])
        return logits.masked_fill(torch.arange(MAX_COUNT + 1, device=h.device) > limit[..., None], -torch.inf)

    def slot_logits(self, h, slot_vec, lattice_onehot, count_onehot):
        """h [nb, hidden]; slot_vec [nb, 16, enc] of the chosen lattice -> [nb, 16] logits."""
        nb = h.shape[0]
        pos = torch.eye(MAX_SLOTS, device=h.device, dtype=h.dtype).expand(nb, -1, -1)
        ctx = torch.cat((h, lattice_onehot, count_onehot), -1)[:, None].expand(-1, MAX_SLOTS, -1)
        return self.slot_head(torch.cat((ctx, slot_vec, pos), -1))[..., 0]


def factor_losses(model: HeadModel, feats, beat_frames, slot_frames, prev_tokens, extra, lattice, count, mask):
    """Per-beat negative log-likelihoods of the three factors, teacher forced.

    Returns dict(lattice [nb], count [nb], slot [nb], head [nb] bool). Lattice and slot terms count
    only on beats with a head (``head``). The slot term is the per-slot binary cross-entropy summed
    over the true lattice's slots, not a normalised likelihood of the top-n subset."""
    beat_vec, sv = model.beat_inputs(feats, beat_frames, slot_frames)
    h, _ = model.run(beat_vec, prev_tokens, extra)
    lat1 = F.one_hot(lattice, 2).float()
    nll_lat = F.cross_entropy(model.lattice_logits(h), lattice, reduction='none')
    nll_cnt = F.cross_entropy(model.count_logits(h, lat1), count, reduction='none')
    cnt1 = F.one_hot(count, MAX_COUNT + 1).float()
    logits = model.slot_logits(h, sv[torch.arange(len(lattice)), lattice], lat1, cnt1)
    valid = torch.arange(MAX_SLOTS) < torch.where(lattice == 0, SLOTS[0], SLOTS[1])[:, None]
    bce = F.binary_cross_entropy_with_logits(logits, mask.float(), reduction='none')
    return dict(lattice=nll_lat, count=nll_cnt, slot=(bce * valid).sum(1), head=count > 0)


def load_model(path):
    """(HeadModel in eval mode, training record) from a ``train`` checkpoint."""
    data = torch.load(path, map_location='cpu', weights_only=False)
    model = HeadModel(HeadConfig.from_dict(data['config']))
    model.load_state_dict(data['model'])
    return model.eval(), data['record']
