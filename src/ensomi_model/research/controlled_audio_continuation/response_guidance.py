"""Independent committed-response guidance for R1 proposals before publication.

This sampling policy is distinct from the actor likelihood. It guides waiting
versus releasing before R is chosen and complete rows at H/R afterward.
Continuation acceptance still evaluates the actually proposed future.
"""
import numpy as np

from ..bounded_typed_continuation.contract import ROW_ACTIONS
from ..player_response.action_response import ActionResponseState,candidate_work
from .generation import ControlledSession


class ResponseGuidedSession(ControlledSession):
    def __init__(self,*args,response_reference,response_strength=4.,**kwargs):
        super().__init__(*args,**kwargs)
        if self.model.release_policy != 'r1_joint':
            raise ValueError('Response guidance needs waiting/release coupling before the R clock commits')
        if not response_reference.work_windows_ms or not np.isfinite(response_strength) or response_strength < 0:
            raise ValueError('Response guidance requires a calibrated work reference and finite nonnegative strength')
        self.response_reference=response_reference
        self.response_strength=response_strength
        self.response_state=ActionResponseState()

    def response_cost(self,times,actions):
        times=np.asarray(times)
        result=np.zeros((len(times),len(actions)))
        if self.response_strength == 0 or not len(times):
            return result
        control=self.controls.at(times)
        known=control[:,2+len(self.controls.style_names)] > 0
        stars=2*control[:,0]+4
        for level in np.unique(stars[known]):
            selected=known&(stars==level)
            work=candidate_work(self.response_state,times[selected],actions,
                                self.response_reference.limits(float(level)))
            budget,_=self.response_reference.work_limit(float(level),4000)
            result[selected]=self.response_strength*work/max(budget,1e-12)
        return result

    def release_candidate_cost(self,times,actions):
        return self.response_cost(times,actions)

    def prefer_rows(self,log_probs,legal,now):
        scores=super().prefer_rows(log_probs,legal,now)
        if self.response_strength == 0:
            return scores
        cost=self.response_cost([now],ROW_ACTIONS)[0]
        return (scores-scores.new_tensor(cost)).log_softmax(-1)

    def record_row(self,row):
        super().record_row(row)
        self.response_state=self.response_state.observe(row,is_terminal=self.replay.is_complete)

    def step(self,**options):
        update=super().step(**options)
        self.response_state=self.response_state.advance(self.cursor)
        return update
