"""Fresh audio/H/R1 construction and a complete source likelihood.

No actor checkpoint is loaded. Corpus selection and runtime response acceptance
remain separate from this bootstrap objective.
"""
import torch
from torch.nn import functional as F

from ..planned_audio_continuation.model import PlannedModelConfig
from ..planned_audio_continuation.intervals import collate_interval,score_heads
from ..typed_audio_continuation.program import Recovery
from .model import SegmentAudioModel,SegmentConfig
from .segments import collate_segment,score_segment


STYLE_NAMES=('jack-organization','stream-organization','trill-organization','tech','ln-coordination')


def fresh_model(mean,std,*,seed,config=None,segment_config=None):
    """Initialize every learned module; copy only explicit input statistics.

    Hidden weights use the constructors' random initialization. Zero residual
    projections and initial event-rate biases retain their documented roles;
    this is not an all-zero neural network. No optimizer state is inherited.
    """
    config=config or PlannedModelConfig(hidden=128,audio_width=64,audio_levels=4,
        history_levels=7,context_width=96,context_layers=2,context_heads=4,
        head_hidden=64,head_levels=5,skeleton_hidden=32,skeleton_levels=3,
        lookahead=16,bounded_head=False,condition_full_holds=True,minimum_action_gap_ms=20)
    segment_config=segment_config or SegmentConfig(states=1,local_levels=5,
        hidden=192,code_width=32,continuous_context=True,reset_local_history=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model=SegmentAudioModel(config,segment_config=segment_config,
            style_names=STYLE_NAMES,recovery=Recovery(config.minimum_action_gap_ms,1,1),
            hold_audio_width=24,ln_conditioning='direct',head_audio_modulation=config.bounded_head,
            release_policy='r1_joint')
    model.set_audio_normalization(mean,std)
    return model


def source_log_probability(model,example,controls,encoded_full):
    """Return H and conditional R/R1 scores on one declared plan interval.

    The interval is a plan boundary, not a crop that implicitly closes holds.
    encoded_full covers the entire song and remains differentiable during joint
    audio learning; callers must not reuse it after an optimizer update.
    No response energy or quantity feedback changes the imitation distribution.
    """
    device=encoded_full.device
    batch=collate_interval(example,model.config,device,recovery=model.recovery,release_policy='r1_joint')
    head_audio=model.condition_audio(encoded_full)
    logits=score_heads(model,batch.inputs,head_audio,
        audio_starts=torch.zeros_like(batch.inputs.base.mel_start),controls=controls).cpu().double()
    event=batch.head_event.cpu();valid=batch.inputs.head_valid.cpu()
    head=torch.where(event,-F.softplus(-logits),-F.softplus(logits)).masked_fill(~valid,0.).sum()
    rows=tuple(example.chart.source.row(i) for i in range(len(example.chart.source.rows)))
    heads=tuple(int(r.time_ms) for r in rows if any(a in (1,2) for a in r.actions))
    trace=collate_segment(rows,heads,example.start_ms,example.end_ms,example.chart.duration_ms,model,device=device)
    row=score_segment(model,trace,heads,controls,encoded_full,recovery_preference=None)['log_probability']
    return dict(head=head,materialization=row,joint=head+row,
        head_events=int(event.sum()),row_events=len(trace.row.targets),release_risk_clocks=len(trace.event))
