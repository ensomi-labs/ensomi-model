"""Teacher scoring with current-weight history/audio pairs and explicit prefix limits."""
from dataclasses import dataclass

import numpy as np
import torch

from ..bounded_typed_continuation.contract import Arm
from ..bounded_typed_continuation.features import CONTENT_DIM
from ..joint_audio_continuation.batching import interpolate_audio
from ..joint_audio_continuation.intervals import _pad_first
from ..planned_audio_continuation.features import skeleton_tokens
from ..planned_audio_continuation.intervals import (
    collate_interval as collate_planned, score_interval as score_planned,
)
from .memory import gather_memory


@dataclass(frozen=True)
class MemoryBatch:
    planned: object
    row_times: np.ndarray
    head_times: np.ndarray
    row_raw: torch.Tensor
    row_valid: torch.Tensor
    release_raw: torch.Tensor
    head_raw: torch.Tensor
    head_valid: torch.Tensor
    row_offset: int
    head_offset: int


def collate_interval(example,model,device='cpu'):
    """Keep source memory through the interval, with per-query causal selection.

    Full prefix re-encoding is deliberate: sampled old states must have the same
    current-weight receptive context as native append operations. The shared
    local scorer may recompute a suffix, but no stale learned cache is retained.
    """
    planned = collate_planned(example,model.config,device,recovery=model.recovery,
                              player_state=model.player_condition is not None)
    source = example.chart.source
    all_times = source.rows['time']
    first,stop = np.searchsorted(all_times,[example.start_ms,example.end_ms])
    times = all_times[:stop].astype(np.int64)
    roles = np.isin(source.rows['actions'][:stop],(1,2)).any(-1)
    heads = times[roles]
    previous = [None,*times[:-1]] if len(times) else []
    previous_heads = [None,*heads[:-1]] if len(heads) else []
    raw = source.content(Arm.R0,0,int(stop)) if stop else np.empty((0,2,CONTENT_DIM),np.float32)
    tensor = lambda value: torch.as_tensor(value,device=device)
    row_start = max(0,int(first)-model.temporal.config.receptive_tokens)
    head_start = max(0,int(np.sum(roles[:first]))-model.head_temporal.config.receptive_tokens)
    return MemoryBatch(planned,times,heads,tensor(_pad_first(raw)[None]),
        tensor(_pad_first(np.ones(len(times),bool))[None]),
        tensor(_pad_first(skeleton_tokens(times,previous,roles))[None]),
        tensor(_pad_first(skeleton_tokens(heads,previous_heads))[None]),
        tensor(_pad_first(np.ones(len(heads),bool))[None]),row_start,head_start)


def score_interval(model,batch,controls,encoded_full):
    """Score the same H/R/row law as native memory queries, including long waits.

    encoded_full must retain its current audio graph when jointly training.
    Memory chart selection uses explicit last-known indices, never a future
    row merely because its time precedes a hazard bin's final-millisecond anchor.
    """
    x = batch.planned.inputs.base
    encoded = model.condition_audio(encoded_full,None)
    downstream = model.condition_audio(encoded_full,None,downstream=True)
    def values(module,raw,valid,count):
        # Keep the existing 128-event bucket through key/value projections.
        # Selection still uses real event times/indices and can never read padding.
        return module(raw,valid)[0] if count else encoded.new_empty((0,2,module.config.hidden))
    def music(times,audio):
        padded=_pad_first(times,edge=True)
        return interpolate_audio(audio,torch.as_tensor(padded[None],device=audio.device),
            frame_counts=x.frame_count)[0]
    contexts = {
        'head': (values(model.head_temporal,batch.head_raw,batch.head_valid,len(batch.head_times)),
                 music(batch.head_times,encoded),batch.head_times,batch.head_offset),
        'release': (values(model.skeleton_temporal,batch.release_raw,batch.row_valid,len(batch.row_times)),
                    music(batch.row_times,downstream),batch.row_times,batch.row_offset),
        'row': (values(model.temporal,batch.row_raw,batch.row_valid,len(batch.row_times)),
                music(batch.row_times,downstream),batch.row_times,batch.row_offset),
    }
    def options(kind,times,indices):
        history,audio,event_times,offset = contexts[kind]
        return dict(memory=gather_memory(history,audio,event_times,indices.detach().cpu().numpy()+offset,
            times.detach().cpu().numpy(),span_ms=model.memory_config.span_ms,cell_ms=model.memory_config.cell_ms))
    return score_planned(model,batch.planned.inputs,None,controls=controls,encoded_full=encoded_full,
                         history_options=options)
