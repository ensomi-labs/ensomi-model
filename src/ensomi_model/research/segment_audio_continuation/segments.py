"""Elapsed-time plan boundaries and exact segment-level marginal likelihood."""
from dataclasses import replace

import numpy as np
import torch

from ..bounded_typed_continuation.features import CONTENT_DIM,TIME_DIM,content_features,time_features
from ..controlled_audio_continuation.joint_trace import collate_joint_trace,score_joint_trace
from ..joint_audio_continuation.batching import interpolate_audio
from ..joint_audio_continuation.intervals import _pad_first
from ..joint_audio_continuation.state import exact_features
from ..oracle_time_continuation.replay import ExactReplayState,commit
from ..planned_audio_continuation.features import preview_after,preview_features
from .model import BirthOrigins


def next_boundary(start_ms,duration_ms,controls,span_ms):
    end=min(duration_ms+1,(start_ms//span_ms+1)*span_ms)
    source=getattr(controls,'source',controls)
    edges=sorted({t for s in source.spans for t in (s.start_ms,s.end_ms) if start_ms<t<end})
    n=2+len(source.style_names)
    for t in edges:
        known=controls.at([t-1,t])[:,n:2*n].any(0)
        owners=[]
        for time in (t-1,t):
            fields=[None]*n
            for s in source.spans:
                if s.start_ms<=time<s.end_ms:
                    values=(s.stars,s.ln_fraction,*(s.style.get(k) for k in source.style_names))
                    for i,v in enumerate(values):
                        if v is not None:fields[i]=(s.start_ms,s.end_ms,v)
            owners.append(fields)
        # A hidden field's source extent cannot become an implicit plan clock.
        if any(k and a!=b for k,a,b in zip(known,*owners)):return t
    return end


def collate_segment(rows,heads,start_ms,end_ms,duration_ms,model,*,device='cpu'):
    """Reset only the local learned observation; retain the actual full prefix."""
    trace=collate_joint_trace(rows,heads,start_ms,end_ms,duration_ms,model,device=device)
    times=np.array([r.time_ms for r in trace.row.rows])
    first=int(np.searchsorted(times,start_ms))
    history_start=max(0,first-model.temporal.config.receptive_tokens)
    local=trace.row.rows[first:]
    raw=content_features(local,[trace.row.rows[i-1].time_ms if i else None
        for i in range(first,len(trace.row.rows))],[[None]*4 for _ in local])
    # Padding gives the dense encoder an addressable tensor even when only
    # release survival is observed. No padding token is valid or committed.
    padded=_pad_first(raw) if len(raw) else np.zeros((1,2,CONTENT_DIM),np.float32)
    valid=_pad_first(np.ones(len(raw),bool)) if len(raw) else np.zeros(1,bool)
    tensor=lambda x:torch.as_tensor(x,device=device)
    row=replace(trace.row,raw=tensor(padded[None]),history_valid=tensor(valid[None]),
        history_indices=trace.row.history_indices-(first-history_start),history_boundary=int(first>0))
    release=(None if trace.release is None else replace(trace.release,
        history_indices=trace.release.history_indices-(first-history_start)))
    return replace(trace,row=row,release=release)


def prefix_observation(model,rows,start_ms):
    before=tuple(r for r in rows if r.time_ms<start_ms)
    replay=ExactReplayState()
    for row in before:replay=commit(replay,row)
    module=model.prior_temporal
    if not before:return replay,module.boundary[0].expand(2,-1)
    first=max(0,len(before)-module.config.receptive_tokens)
    raw=content_features(before[first:],[before[i-1].time_ms if i else None
        for i in range(first,len(before))],[[None]*4 for _ in before[first:]])
    value=module(module.input.weight.new_tensor(raw[None]))[0,-1]
    return replay,value


def plan_prior(model,encoded,replay,history,heads,controls,start_ms,end_ms,duration_ms):
    """Read future audio/H and incoming facts, never future row materialization."""
    cfg=model.segment_config;device=encoded.device
    heads=np.asarray(heads,dtype=np.int64)
    times=(start_ms+(np.arange(cfg.audio_queries)+.5)*(end_ms-start_ms)/cfg.audio_queries).astype(np.int64)
    audio=interpolate_audio(encoded,torch.as_tensor(times[None],device=device))[0]
    at_start=interpolate_audio(encoded,torch.tensor([[start_ms]],device=device))[0]
    preview=preview_after(heads,start_ms-1,model.config.lookahead)
    view=preview_features([preview],[start_ms-1],[False],duration_ms,model.config.lookahead)
    exact=encoded.new_tensor(exact_features([replay],[start_ms]))
    control=encoded.new_tensor(controls.at(times,encoding=model.control_encoding))
    incoming=(model.condition(history[None],exact,at_start)
        +model.row_control(encoded.new_tensor(controls.at([start_ms],encoding=model.control_encoding)))[:,None]
        +model.preview_condition(encoded.new_tensor(view))[:,None])[0]
    first,stop=np.searchsorted(heads,(start_ms,end_ms))
    selected=heads[first:stop]
    previous=np.r_[heads[first-1] if first else np.nan,selected[:-1]] if len(selected) else np.empty(0)
    hf=time_features(np.stack((selected-start_ms,selected-previous,end_ms-selected),-1))
    hf=hf.reshape(len(selected),-1) if len(selected) else np.empty((0,3*TIME_DIM),np.float32)
    extent=encoded.new_tensor([np.log1p(len(selected)),(end_ms-start_ms)/1000])
    return model.plan_prior(incoming,audio,encoded.new_tensor(hf),control,extent)


def marginal_log_probability(prior,conditionals):
    """One mixture after whole-segment products, with an exact posterior."""
    prior=prior.cpu().double().log_softmax(-1)
    conditionals=conditionals.cpu().double()
    joint=prior+conditionals
    return joint.logsumexp(-1),joint.log_softmax(-1)


def score_segment(model,trace,heads,controls,encoded,*,recovery_preference=None):
    """Enumerate all private codes while sharing factual physical observations.

    Raw actor likelihood is the default. A declared recovery preference must
    match the proposal being scored; response-guided selection is not included.
    Audio/H must stay fixed for an outcome gradient that uses only this factor.
    """
    r=trace.row
    replay,history=prefix_observation(model,r.rows,r.start_ms)
    prior=plan_prior(model,encoded,replay,history,heads,controls,r.start_ms,r.end_ms,r.duration_ms)
    births=BirthOrigins.from_rows(r.rows)
    conditional=torch.stack([score_joint_trace(model.bind(k,births),trace,controls,encoded,
        ln_feedback=None,recovery_preference=recovery_preference).log_probability
        for k in range(model.segment_config.states)])
    probability,posterior=marginal_log_probability(prior,conditional)
    return dict(log_probability=probability,prior=prior,posterior=posterior,conditional=conditional)
