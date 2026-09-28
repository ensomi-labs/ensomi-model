"""Private plans persist across publications; exact holds survive plan changes."""
import torch

from ..bounded_typed_continuation.features import content_features
from ..controlled_audio_continuation.generation import ControlledSession
from ..controlled_audio_continuation.response_guidance import ResponseGuidedSession
from ..planned_audio_continuation.session import _fork_rng
from .model import BirthOrigins
from .segments import next_boundary,plan_condition


class SegmentSession(ControlledSession):
    def __init__(self,*args,**kwargs):
        kwargs.setdefault('ln_feedback',None)
        kwargs.setdefault('recovery_preference',None)
        super().__init__(*args,**kwargs)
        self.prior_cache=self.model.prior_temporal.empty_cache()
        self.plan_rng=torch.Generator(device='cpu').manual_seed(kwargs.get('seed',260926)^0x29C1)
        self.plan_code=None;self.plan_context=None;self.plan_end=0;self.plan_events=[]
        self.births=BirthOrigins()

    @property
    def row_model(self):
        return self.model.bind(self.plan_code,self.births,self.plan_context)

    @torch.inference_mode()
    def activate_plan(self):
        start=max(0,self.cursor+1)
        end=next_boundary(start,self.duration_ms,self.controls,self.model.segment_config.span_ms)
        while (not self.planner.finished and (not self.planner.queue or self.planner.queue[-1]<end)):
            self.planner.fill(len(self.planner.queue)+self.model.config.lookahead+1)
        prior,self.plan_context=plan_condition(self.model,self.downstream_encoded,self.replay,
            self.model.prior_temporal.read(self.prior_cache),self.planner.generated,self.controls,
            start,end,self.duration_ms)
        prior=prior.cpu().double().log_softmax(-1)
        self.plan_code=int(torch.multinomial(prior.exp(),1,generator=self.plan_rng))
        self.plan_end=end
        if self.model.segment_config.reset_local_history:
            self.row_cache=self.model.temporal.empty_cache(truncated_start=bool(self.rows))
        self.plan_events.append(dict(kind='selected',start_ms=start,end_ms=end,code=self.plan_code,prior=prior.exp().tolist()))

    def step(self,*,stop_at=None):
        if self.plan_code is None or self.cursor>=self.plan_end-1:self.activate_plan()
        limit=self.duration_ms if stop_at is None else min(stop_at,self.duration_ms)
        return super().step(stop_at=min(limit,self.plan_end-1))

    def record_row(self,row):
        super().record_row(row)
        previous=self.rows[-2].time_ms if len(self.rows)>1 else None
        self.prior_cache=self.model.prior_temporal.append(self.prior_cache,
            self.tensor(content_features([row],[previous],[[None]*4])[0]))
        active={int(t) for t in self.replay.open_ln_start_ms if t is not None}
        old=dict(zip(self.births.times,self.births.actions))
        if 2 in row.actions:old[int(row.time_ms)]=row.actions
        self.births=BirthOrigins((t,old[t]) for t in active)

    def fork(self,*,retry_seed=None):
        result=super().fork(retry_seed=retry_seed)
        result.plan_rng=_fork_rng(self.plan_rng)
        if retry_seed is not None:result.plan_rng.manual_seed(retry_seed^0x29C1)
        result.plan_events=list(self.plan_events)
        return result

    def update_controls(self,span):
        super().update_controls(span)
        if span.start_ms<self.plan_end:
            self.plan_end=span.start_ms
            self.plan_events.append(dict(kind='control_cut',at_ms=span.start_ms))


class GuidedSegmentSession(SegmentSession,ResponseGuidedSession):
    """The same segment proposal plus independently calibrated response guidance."""
