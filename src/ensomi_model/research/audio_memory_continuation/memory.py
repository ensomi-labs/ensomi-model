"""Elapsed-time sampling and context-dependent reads of committed event memory."""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..scoped_style_modeling.dataset import ContractError


def memory_indices(event_times, known_indices, query_times, *, span_ms=64000, cell_ms=500):
    """Select the latest known event per absolute clock cell, or -1.

    known_indices names the last observed event for each query. It is essential:
    a hazard-bin anchor can follow the target event within that same bin. Times
    alone cannot determine the causal prefix. Empty cells are not fake events.
    """
    times = np.asarray(event_times, dtype=np.int64)
    known = np.asarray(known_indices, dtype=np.int64)
    queries = np.asarray(query_times, dtype=np.int64)
    if known.shape != queries.shape or known.ndim != 1 or np.any(known >= len(times)) or np.any(known < -1):
        raise ContractError('Memory queries require one valid known-prefix index per query')
    if span_ms <= 0 or cell_ms <= 0 or span_ms % cell_ms:
        raise ContractError('Memory span must be a positive multiple of its clock cell')
    cells = queries[:, None]//cell_ms-np.arange(span_ms//cell_ms, -1, -1)
    result = np.minimum(np.searchsorted(times, (cells+1)*cell_ms, side='left')-1, known[:, None])
    if not len(times):
        return np.full_like(result, -1)
    selected = times[result.clip(0)]
    valid = ((result >= 0) & (selected//cell_ms == cells) &
             (selected >= queries[:, None]-span_ms) & (selected < queries[:, None]))
    return np.where(valid, result, -1)


@dataclass(frozen=True)
class MemoryView:
    history: torch.Tensor  # events, hands, history width
    audio: torch.Tensor  # events, audio width
    indices: torch.Tensor  # queries, cells; -1 is unobserved
    age_seconds: torch.Tensor

    @property
    def valid(self):
        return self.indices >= 0


def gather_memory(history, audio, event_times, known_indices, query_times, *, span_ms=64000, cell_ms=500):
    """Gather differentiable values; selection uses exact CPU integer clocks."""
    times = np.asarray(event_times, dtype=np.int64)
    query = np.asarray(query_times, dtype=np.int64)
    indices = memory_indices(times, known_indices, query, span_ms=span_ms, cell_ms=cell_ms)
    if len(times):
        ages = np.where(indices >= 0, (query[:, None]-times[indices.clip(0)])/1000, 0.)
    else:
        ages = np.zeros(indices.shape)
    return MemoryView(history,audio,torch.as_tensor(indices,device=history.device),history.new_tensor(ages))


class HistoryAttention(nn.Module):
    """Shared hand-coordinate attention with a zero-initialized history residual.

    Musical queries can select different memories for the two hands. A fixed
    zero-valued null key makes BOS/all-empty reads finite and exactly zero.
    The caller owns the causal memory selection and its raw-history provenance.
    """
    def __init__(self, query_width, history_width, audio_width, width=128, heads=4):
        super().__init__()
        if width % heads:
            raise ContractError('Attention width must be divisible by its head count')
        self.width, self.heads = width, heads
        self.query_norm = nn.LayerNorm(query_width)
        self.memory_norm = nn.LayerNorm(history_width+audio_width)
        self.query = nn.Linear(query_width, width, bias=False)
        self.key = nn.Linear(history_width+audio_width, width, bias=False)
        self.value = nn.Linear(history_width+audio_width, width, bias=False)
        self.time_bias = nn.Sequential(nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, heads, bias=False))
        self.output = nn.Linear(width, history_width, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, query, memory):
        if memory is None:
            raise ContractError('Audio-memory model requires a causal committed-history view')
        n, hands = query.shape[:2]
        if hands != 2 or memory.indices.shape[:1] != (n,):
            raise ContractError('History attention requires aligned two-hand queries and memory')
        dimension = self.width//self.heads
        paired = memory.audio.unsqueeze(-2).expand(-1, 2, -1)
        raw = self.memory_norm(torch.cat((memory.history, paired), -1))
        q = self.query(self.query_norm(query)).reshape(n, 2, self.heads, dimension)
        selected_indices = torch.cat((memory.indices.new_full((n,1),-1),memory.indices),-1)+1
        def table(projection):
            values = projection(raw).reshape(len(raw),2,self.heads,dimension)
            return torch.cat((values.new_zeros((1,2,self.heads,dimension)),values),0)
        keys,values = table(self.key),table(self.value)
        ages = memory.age_seconds
        clocks = torch.stack((torch.log1p(ages), torch.exp(-ages/2), torch.exp(-ages/16)), -1)
        bias = self.time_bias(clocks).permute(0, 2, 1)
        bias = bias.masked_fill(~memory.valid[:, None], -torch.inf)
        # A null memory is always available, including at genuine BOS.
        bias = torch.cat((torch.zeros_like(bias[..., :1]), bias), -1)
        def attend(query,keys,values,indices,bias):
            k,v = (x[indices].permute(0,2,3,1,4) for x in (keys,values))
            return F.scaled_dot_product_attention(query.unsqueeze(-2),k,v,
                attn_mask=bias[:,None,:,None],dropout_p=0.).reshape(len(query),2,self.width)
        if torch.is_grad_enabled() and n>128:
            # Recompute query-specific gathers, retaining shared tables and every
            # gradient rather than all expanded query/cell/channel activations.
            value = torch.cat([checkpoint(attend,q[i:i+128],keys,values,
                selected_indices[i:i+128],bias[i:i+128],use_reentrant=False)
                for i in range(0,n,128)])
        else:
            value = attend(q,keys,values,selected_indices,bias)
        return self.output(value)
