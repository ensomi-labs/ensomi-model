"""Bounded planning over private real-time continuations of the row policy.

The selected trajectory has a different law from native row sampling. Native
row log probabilities remain proposal probabilities, not selected-policy scores.
This first planner uses sustained attacks only; it does not certify total demand.
"""
from dataclasses import dataclass
import time

import numpy as np
import torch

from ..player_response.envelope import sustained_response
from ..player_response.state import CommittedPlayState
from ..scoped_style_modeling.dataset import ContractError


@dataclass(frozen=True)
class FrontierPlanning:
    horizon_ms: int = 4000
    publication_ms: int = 2000
    maximum_candidates: int = 4

    def __post_init__(self):
        if (any(type(x) is not int or x <= 0 for x in
                (self.horizon_ms, self.publication_ms, self.maximum_candidates)) or
                self.publication_ms > self.horizon_ms):
            raise ContractError('Frontier planning requires positive time/candidate bounds and a covered publication')


class ResponsePlanner:
    """Own one publication boundary and choose among bounded private futures.

    Every candidate uses the same full audio, H law and controls. Retry RNGs
    change R/R1 only. An open LN at the publication boundary remains open;
    hypothetical later releases are not published early or stored as true facts.
    Response state persists across controls, including committed no-row time.
    """
    def __init__(self, session, envelope, *, seed, config=FrontierPlanning()):
        self.session, self.envelope, self.config = session, envelope, config
        self.rng = np.random.default_rng(seed ^ 0x6F17)
        self.state = (CommittedPlayState.from_rows(session.rows, session.coverage)
                      if session.coverage >= 0 else CommittedPlayState())
        self.decisions = []

    def update_controls(self, span):
        self.session.update_controls(span)

    @torch.inference_mode()
    def publish_to(self, end_ms):
        """Publish in bounded increments, retaining only selected prefix states.

        The session's original wall-clock/row limits cover all private proposals.
        Exceptions leave the last published session and demand state unchanged.
        An exhausted candidate budget selects the least excess found; it does
        not guarantee a zero-cost or playable continuation.
        """
        while self.session.coverage < min(end_ms, self.session.duration_ms):
            current = self.session
            start = current.coverage
            publish_end = min(end_ms, current.duration_ms, max(0, start)+self.config.publication_ms)
            forecast_end = min(current.duration_ms, max(0, start)+self.config.horizon_ms)
            ranges = tuple((s.start_ms, s.end_ms, s.stars)
                           for s in current.controls.resolved_ranges(start, forecast_end))
            tick = time.perf_counter()
            best, best_cost, proposals = None, float('inf'), []
            for i in range(self.config.maximum_candidates):
                retry_seed = None if i == 0 else int(self.rng.integers(0, 2**31))
                proposal = current.fork(retry_seed=retry_seed)
                proposal.publish_to(publish_end)
                prefix = proposal.fork()
                proposal.publish_to(forecast_end)
                future = proposal.rows[len(current.rows):]
                _, report = sustained_response(self.state, future, forecast_end, self.envelope, ranges)
                cost = report['excess_seconds']
                if not np.isfinite(cost):
                    raise ContractError('A frontier proposal has nonfinite response cost')
                proposals.append(dict(index=i, retry_seed=retry_seed, **report))
                if cost < best_cost:
                    best, best_cost, selected = prefix, cost, i
                if cost == 0.:
                    break
            state = self.state
            for row in best.rows[len(current.rows):]:
                state = state.observe(row)
            self.state, self.session = state.advance(publish_end), best
            self.decisions.append(dict(start_ms=start, publication_end_ms=publish_end,
                forecast_end_ms=forecast_end, selected=selected, proposals=proposals,
                service_seconds=time.perf_counter()-tick))
        return self.session.coverage
