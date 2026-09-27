"""Full-audio pyramid and separate context-queried H/R/R1 memories."""
from dataclasses import asdict, dataclass

import torch

from ..controlled_audio_continuation.model import ControlledAudioModel
from ..joint_audio_continuation.context_model import coarse_at_frames
from ..joint_audio_continuation.model import JointAudioModel
from ..oracle_time_continuation.features import TIME_DIM
from ..planned_audio_continuation.features import RELEASE_CLOCK_DIM
from ..planned_audio_continuation.model import PlannedModelConfig
from ..scoped_style_modeling.dataset import ContractError
from ..typed_audio_continuation.program import Recovery
from .audio import AudioPyramid
from .memory import HistoryAttention


@dataclass(frozen=True)
class MemoryConfig:
    span_ms: int = 64000
    cell_ms: int = 500
    width: int = 128
    heads: int = 4
    audio_width: int = 256
    audio_layers: int = 3

    def __post_init__(self):
        if (any(type(v) is not int or v<=0 for v in vars(self).values()) or
                self.span_ms % self.cell_ms or self.width % self.heads or
                self.audio_width % self.heads or self.audio_width % 2):
            raise ContractError('Audio memory dimensions and elapsed-time cells must be positive and aligned')


class AudioMemoryModel(ControlledAudioModel):
    def __init__(self, config, *, memory_config=MemoryConfig(), **options):
        super().__init__(config, **options)
        self.memory_config = memory_config
        a, c = config.conditioned_audio_width, self.head_control.in_features
        w, h = memory_config.width, memory_config.heads
        self.audio_pyramid = AudioPyramid(config.audio_width, config.context_width,
            memory_config.audio_width, memory_config.audio_layers, h)
        self.head_memory = HistoryAttention(config.head_hidden+a+2*TIME_DIM+c, config.head_hidden,a,w,h)
        self.release_memory = HistoryAttention(config.skeleton_hidden+a+RELEASE_CLOCK_DIM+c,
            config.skeleton_hidden,a,w,h)
        self.row_memory = HistoryAttention(config.hidden,config.hidden,a,w,h)

    @property
    def requires_full_audio_queries(self):
        return True

    def encode_crop(self, *args, **kwargs):
        raise ContractError('Audio-memory training requires a differentiable complete-song encoding')

    def encode_audio(self, mel, valid=None):
        if valid is None:
            valid = torch.ones(mel.shape[:2],dtype=torch.bool,device=mel.device)
        fine = JointAudioModel.encode_audio(self,mel,valid)
        coarse, counts = self.encode_coarse(mel,valid)
        coarse = coarse+self.audio_pyramid(fine,valid)
        frames = torch.arange(mel.shape[1],device=mel.device)[None].expand(len(mel),-1)
        global_values = coarse_at_frames(coarse,counts,frames,valid.sum(-1))
        return torch.cat((fine,global_values),-1)

    def head_logits(self,audio,history,clocks,*,control,memory=None):
        shared = torch.cat((audio,clocks,control),-1)[:,None].expand(-1,2,-1)
        context = history+self.head_memory(torch.cat((history,shared),-1),memory)
        return super().head_logits(audio,context,clocks,control=control)

    def release_logits(self,audio,history,clocks,*,control,hold_audio=None,memory=None):
        shared = torch.cat((audio,control),-1)[:,None].expand(-1,2,-1)
        query = torch.cat((history,shared,clocks),-1)
        context = history+self.release_memory(query,memory)
        return super().release_logits(audio,context,clocks,control=control,hold_audio=hold_audio)

    def planned_row_log_probs(self,audio,history,exact,legal,occupancy,preview,local,timing,
                              *,control,memory=None,**options):
        query = (self.condition(history,exact,audio)+self.preview_condition(preview).unsqueeze(-2)
                 +self.row_control(control).unsqueeze(-2))
        context = history+self.row_memory(query,memory)
        return super().planned_row_log_probs(audio,context,exact,legal,occupancy,preview,local,timing,
                                              control=control,**options)

    def checkpoint(self):
        return dict(format='controlled-audio-memory/v1',model_config=asdict(self.config),
            memory_config=asdict(self.memory_config),probability_options=self.probability_options(),
            model=self.state_dict())


def initialize(base, *, memory_config=MemoryConfig()):
    options = base.probability_options()
    options['recovery'] = base.recovery
    model = AudioMemoryModel(base.config,memory_config=memory_config,**options)
    receipt = model.load_state_dict(base.state_dict(),strict=False)
    prefixes = ('audio_pyramid.','head_memory.','release_memory.','row_memory.')
    if receipt.unexpected_keys or any(not key.startswith(prefixes) for key in receipt.missing_keys):
        raise ContractError('Audio-memory initialization did not consume the complete baseline')
    return model


def load_model(path,*,device='cpu'):
    saved = torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('format')!='controlled-audio-memory/v1':
        raise ContractError('Expected a controlled-audio-memory/v1 checkpoint')
    options = dict(saved['probability_options'])
    options['recovery'] = Recovery(**options['recovery'])
    model = AudioMemoryModel(PlannedModelConfig(**saved['model_config']),
        memory_config=MemoryConfig(**saved['memory_config']),**options)
    model.load_state_dict(saved['model'],strict=True)
    return model.to(device).eval()
