"""A persistent segment choice with a nonlinear complete-row R1 decoder."""
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch
from torch import nn

from ..bounded_typed_continuation.contract import ROW_ACTIONS
from ..bounded_typed_continuation.features import RELATIVE_LANES, TIME_DIM
from ..bounded_typed_continuation.temporal import FiniteTemporal
from ..controlled_audio_continuation.model import ControlledAudioModel
from ..planned_audio_continuation.model import PlannedModelConfig
from ..scoped_style_modeling.dataset import ContractError
from ..typed_audio_continuation.program import Recovery


@dataclass(frozen=True)
class SegmentConfig:
    states: int = 4
    span_ms: int = 4000
    local_levels: int = 3
    hidden: int = 256
    code_width: int = 64
    audio_queries: int = 8

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in vars(self).values()):
            raise ValueError('Segment dimensions and elapsed-time extent must be positive integers')


class SegmentPrior(nn.Module):
    def __init__(self, config, segment, control_width):
        super().__init__()
        self.heads = nn.Sequential(nn.Linear(3*TIME_DIM, 64), nn.GELU(), nn.Linear(64, 64))
        self.controls = nn.Sequential(nn.Linear(control_width, 32), nn.GELU())
        width = config.hidden+segment.audio_queries*(config.conditioned_audio_width+32)+64+2
        self.body = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, segment.hidden), nn.GELU(),
                                  nn.Linear(segment.hidden, segment.hidden), nn.GELU())
        self.output = nn.Linear(segment.hidden, segment.states)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self, context, audio, head_features, control, extent):
        head = (self.heads(head_features).mean(0) if len(head_features) else context.new_zeros(64))
        value = torch.cat((context.mean(0), audio.flatten(), self.controls(control).flatten(), head, extent))
        return self.output(self.body(value)).log_softmax(-1)


class SegmentDecoder(nn.Module):
    """Shared hand views produce joint rows; one code modulates two hidden layers."""
    def __init__(self, width, hidden, code_width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.first = nn.Linear(width, hidden)
        self.second = nn.Linear(hidden, hidden)
        self.film1 = nn.Linear(code_width, 2*hidden)
        self.film2 = nn.Linear(code_width, 2*hidden)
        self.output = nn.Linear(hidden, 256)
        for film in (self.film1,self.film2):
            nn.init.normal_(film.weight,std=.02);nn.init.zeros_(film.bias)
        nn.init.normal_(self.output.weight,std=.02);nn.init.zeros_(self.output.bias)
        actions = np.asarray(ROW_ACTIONS)
        indices = np.stack([actions[:,lanes]@np.array([64,16,4,1]) for lanes in RELATIVE_LANES])
        self.register_buffer('indices',torch.from_numpy(indices),persistent=False)

    def forward(self, hands, code, candidate_indices=None):
        def modulate(x, film):
            scale,bias = film(code).chunk(2,-1)
            return nn.functional.gelu(x*(1+scale.tanh())+bias)
        value = modulate(self.first(self.norm(hands)),self.film1)
        value = modulate(self.second(value),self.film2)
        indices = self.indices if candidate_indices is None else self.indices[:,candidate_indices]
        if candidate_indices is not None:
            return torch.stack([nn.functional.linear(value[:,h],self.output.weight[indices[h]],
                self.output.bias[indices[h]]) for h in range(2)],1).mean(1)
        relative = self.output(value)
        return relative.gather(-1,indices[None].expand(len(hands),-1,-1)).mean(1)


class BirthOrigins:
    """Immutable factual birth rows, indexed only by already active LN starts."""
    def __init__(self, values=()):
        data = dict(values)
        self.times = np.array(sorted(data),dtype=np.int64)
        self.actions = np.array([data[t] for t in self.times],dtype=np.int8).reshape(-1,4)
        self.times.setflags(write=False);self.actions.setflags(write=False)

    @classmethod
    def from_rows(cls,rows):
        return cls((int(r.time_ms),r.actions) for r in rows if 2 in r.actions)

    def at(self,starts):
        starts = np.asarray(starts,dtype=np.int64)
        active = starts >= 0
        result = np.zeros((*starts.shape,16),dtype=np.float32)
        if not active.any():return result
        indices = np.searchsorted(self.times,starts[active])
        if (np.any(indices>=len(self.times)) or not np.array_equal(self.times[indices],starts[active])):
            raise ContractError('Active LN has no observed birth-row context')
        result[active] = np.eye(4,dtype=np.float32)[self.actions[indices]].reshape(-1,16)
        return result


class BoundSegment:
    """A nonmutating conditional view; forks can share frozen neural weights."""
    def __init__(self,model,code,births):
        if type(code) is not int or not 0<=code<model.segment_config.states:
            raise ValueError('Segment code is outside its categorical support')
        self.model,self.code,self.births = model,code,births

    def __getattr__(self,name):return getattr(self.model,name)

    def hold_audio_options(self,encoded,starts,times,**kwargs):
        values = self.model.hold_audio_options(encoded,starts,times,**kwargs)
        values['birth_actions'] = encoded.new_tensor(self.births.at(starts.detach().cpu().numpy()))
        return values

    def planned_row_log_probs(self,*args,**kwargs):
        return self.model.planned_row_log_probs(*args,plan_code=self.code,**kwargs)


class SegmentAudioModel(ControlledAudioModel):
    def __init__(self,config,*,segment_config=SegmentConfig(),**options):
        super().__init__(config,**options)
        if (self.release_policy!='r1_joint' or self.hold_cues is None or self.player_condition is not None
                or self.scope_allocation is not None or self.layout_modulation is not None
                or self.ln_conditioning!='direct'):
            raise ContractError('Segment R1 requires joint releases, hold audio and its own unmodified row decoder')
        self.segment_config = segment_config
        self.prior_temporal = self.temporal
        self.temporal = FiniteTemporal(replace(self.temporal.config,levels=segment_config.local_levels))
        for name in ('joint','route_residual','release_residual','composition','row_consequence'):
            delattr(self,name)
        self.decoder = SegmentDecoder(config.hidden,segment_config.hidden,segment_config.code_width)
        self.birth_encoder = nn.Sequential(nn.Linear(16,32),nn.GELU())
        self.birth_row = nn.Linear(4*32,config.hidden,bias=False)
        self.register_buffer('birth_relative',torch.tensor(RELATIVE_LANES),persistent=False)
        self.plan_prior = SegmentPrior(config,segment_config,self.row_control.in_features)
        self.plan_codes = nn.Embedding(segment_config.states,segment_config.code_width)
        nn.init.normal_(self.plan_codes.weight,std=.2)

    def bind(self,code,births):return BoundSegment(self,code,births)

    def planned_row_log_probs(self,audio,history,exact,legal,occupancy,preview,local,timing,
                              *,control,plan_code,birth_actions,hold_audio,response_allowed=None,
                              candidate_indices=None):
        allowed = legal if response_allowed is None else legal & response_allowed
        hands = (self.condition(history,exact,audio)+self.preview_condition(preview)[:,None]
                 +self.row_control(control)[:,None]+self.hold_cues.row_values(audio,hold_audio))
        birth_rows=birth_actions.reshape(len(audio),4,4,4)
        views=torch.stack([birth_rows[:,lanes][:,:,lanes].flatten(-2) for lanes in RELATIVE_LANES],1)
        born=self.birth_encoder(views)*hold_audio[:,self.birth_relative,-1:]
        hands=hands+self.birth_row(born.flatten(-2))
        scores = self.decoder(hands,self.plan_codes.weight[plan_code],candidate_indices)
        return scores.masked_fill(~allowed,-torch.inf).log_softmax(-1)

    def checkpoint(self):
        return dict(format='segment-audio/v1',model_config=asdict(self.config),
            segment_config=asdict(self.segment_config),probability_options=self.probability_options(),
            model=self.state_dict())


def initialize(parent,*,segment_config=SegmentConfig(),seed=280930):
    options = parent.probability_options()
    options.update(recovery=parent.recovery,layout_modulation=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = SegmentAudioModel(parent.config,segment_config=segment_config,**options)
    targets = model.state_dict();copied={};omitted=[]
    for name,value in parent.state_dict().items():
        if name.startswith('temporal.'):
            copied['prior_'+name]=value
        if name in targets and targets[name].shape==value.shape:
            copied[name]=value
        elif not name.startswith('temporal.'):
            omitted.append(name)
    receipt=model.load_state_dict(copied,strict=False)
    if receipt.unexpected_keys:raise ContractError('Unconsumed parent tensors in segment initialization')
    return model,dict(copied=sorted(copied),omitted=omitted,initialized=receipt.missing_keys)


def configure_materializer(model):
    trainable={'temporal','exact','fuse','audio_residual','context_condition','preview_condition',
               'row_control','hold_cues','release_log_scale','decoder','birth_encoder','birth_row',
               'plan_prior','plan_codes'}
    for name,p in model.named_parameters():p.requires_grad_(name.split('.')[0] in trainable)
    return dict(trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                frozen=sum(p.numel() for p in model.parameters() if not p.requires_grad))


def load_model(path,*,device='cpu'):
    value=torch.load(path,map_location='cpu',weights_only=True)
    if value.get('format')!='segment-audio/v1':raise ContractError('Expected segment-audio/v1')
    options=dict(value['probability_options']);options['recovery']=Recovery(**options['recovery'])
    model=SegmentAudioModel(PlannedModelConfig(**value['model_config']),
        segment_config=SegmentConfig(**value['segment_config']),**options)
    model.load_state_dict(value['model'],strict=True)
    return model.to(device).eval()
