"""Native memory writes belong to chosen H or committed rows, never proposals."""
from dataclasses import dataclass

import numpy as np
import torch

from ..controlled_audio_continuation.generation import ControlledSession
from ..joint_audio_continuation.batching import interpolate_audio
from ..planned_audio_continuation.generation import HeadPlanner
from .memory import MemoryView, memory_indices


@dataclass(frozen=True)
class EventMemory:
    times: tuple = ()
    history: tuple = ()
    audio: tuple = ()

    def append(self,time_ms,history,audio):
        return EventMemory((*self.times,time_ms),(*self.history,history),(*self.audio,audio))

    def truncate(self,count):
        return EventMemory(self.times[:count],self.history[:count],self.audio[:count])

    def query(self,times,config,like,history_width):
        query = times.detach().cpu().numpy()
        indices = memory_indices(self.times,np.full(len(query),len(self.times)-1),query,
            span_ms=config.span_ms,cell_ms=config.cell_ms)
        chosen = np.unique(indices[indices>=0])
        if len(chosen):
            history = torch.stack([self.history[i] for i in chosen])
            audio = torch.stack([self.audio[i] for i in chosen])
            selected_times = np.asarray(self.times)[indices.clip(0)]
            ages = np.where(indices>=0,(query[:,None]-selected_times)/1000,0.)
            compact = np.where(indices>=0,np.searchsorted(chosen,indices),-1)
        else:
            history = like.new_empty((0,2,history_width))
            audio = like.new_empty((0,like.shape[-1]))
            ages,compact = np.zeros(indices.shape),indices
        return MemoryView(history,audio,torch.as_tensor(compact,device=like.device),like.new_tensor(ages))


class MemoryHeadPlanner(HeadPlanner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.event_memory = EventMemory()

    def query_options(self,times):
        return dict(memory=self.event_memory.query(times,self.model.memory_config,self.encoded,
                                                     self.model.config.head_hidden))

    def record_head(self,time_ms):
        audio = interpolate_audio(self.encoded,torch.tensor([[time_ms]],device=self.device))[0,0]
        self.event_memory = self.event_memory.append(time_ms,self.model.head_temporal.read(self.cache),audio)

    def update_controls(self,*args,**kwargs):
        super().update_controls(*args,**kwargs)
        self.event_memory = self.event_memory.truncate(len(self.generated))


class MemorySession(ControlledSession):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,planner_type=MemoryHeadPlanner,**kwargs)
        self.row_memory,self.release_memory = EventMemory(),EventMemory()

    def row_options(self,now):
        options = super().row_options(now)
        times = torch.tensor([now],device=self.device)
        options['memory'] = self.row_memory.query(times,self.model.memory_config,self.downstream_encoded,
                                                  self.model.config.hidden)
        return options

    def release_options(self,times):
        return dict(memory=self.release_memory.query(times,self.model.memory_config,self.downstream_encoded,
                                                        self.model.config.skeleton_hidden))

    def record_row(self,row):
        super().record_row(row)
        audio = interpolate_audio(self.downstream_encoded,torch.tensor([[row.time_ms]],device=self.device))[0,0]
        self.row_memory = self.row_memory.append(row.time_ms,self.model.temporal.read(self.row_cache),audio)
        self.release_memory = self.release_memory.append(row.time_ms,self.model.skeleton_temporal.read(self.skeleton_cache),audio)
