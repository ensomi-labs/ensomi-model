"""Actual R/R1 continuation likelihood on a fixed timing-only H proposal.

Release survival depends on R1 parameters in joint-release mode. Row-only
scores therefore omit policy-gradient credit even when H and audio are frozen.
A local endpoint censors the physical continuation; it never closes an LN.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from ..oracle_time_continuation.replay import ExactReplayState, commit
from ..planned_audio_continuation.features import LNProjection, preview_after, release_masks
from ..planned_audio_continuation.spacing import row_release_window
from ..scoped_style_modeling.dataset import ContractError
from ..typed_audio_continuation.response_preference import RecoveryPreference
from .joint_release import ReleaseQueries, release_queries, release_logits
from .trace import RowTrace, collate_row_trace, score_row_trace


@dataclass(frozen=True)
class JointTrace:
    row: RowTrace
    release: ReleaseQueries | None
    event: torch.Tensor
    forced: torch.Tensor


@dataclass(frozen=True)
class JointTraceScores:
    row: torch.Tensor
    release: torch.Tensor

    @property
    def log_probability(self):
        return self.row.sum()+self.release.sum()


def collate_joint_trace(rows, head_times, start_ms, end_ms, duration_ms, model, *, device='cpu'):
    """Read genuine rows from BOS and score the half-open observed continuation.

    The H plan must include its complete actual future through audio EOF. It
    may be a frozen model proposal or source-H diagnostic. No source actions
    beyond end_ms are read. History, occupancy, event targets and all observed
    no-event milliseconds belong to the supplied trajectory. This does not
    score the H factor, response-selected policy, or an unobserved future.
    """
    if model.release_policy != 'r1_joint':
        raise ContractError('Joint traces require the R1-owned release policy')
    row = collate_row_trace(rows, head_times, start_ms, end_ms, duration_ms, model, device=device)
    times = np.asarray([r.time_ms for r in row.rows], dtype=np.int64)
    first = int(np.searchsorted(times, start_ms))
    history_start = max(0, first-model.temporal.config.receptive_tokens)
    heads = np.asarray(head_times, dtype=np.int64)
    replay = ExactReplayState()
    for r in row.rows[:first]:
        replay = commit(replay, r)
    cursor = start_ms-1
    states, clocks, previews, history, events, forced = [], [], [], [], [], []
    for index in range(first, len(row.rows)+1):
        target = row.rows[index] if index < len(row.rows) else None
        stop = int(target.time_ms) if target is not None else end_ms-1
        if any(replay.occupancy) and stop > cursor:
            preview = preview_after(heads, cursor, model.config.lookahead)
            window = row_release_window(replay, cursor, preview, duration_ms, model.recovery)
            native = np.arange(cursor+1, stop+1, dtype=np.int64)[None]
            valid, deadline = release_masks([LNProjection(replay.open_ln_start_ms, cursor)],
                native, [preview], duration_ms, minimum_action_gap_ms=model.recovery, windows=[window])
            selected = native[valid]
            event = ((selected == stop) & (target is not None) &
                     _release_only(target))
            if np.any(deadline[valid] & ~event):
                raise ContractError('Trace survives a forced R deadline without its required event')
            states.extend([replay]*len(selected));clocks.extend(selected)
            previews.extend([preview]*len(selected))
            history.extend([index-history_start-1]*len(selected))
            events.extend(event);forced.extend(deadline[valid])
        if target is not None:
            if _release_only(target) and (not clocks or clocks[-1] != stop or not events[-1]):
                raise ContractError('Trace release has no eligible native R query')
            replay = commit(replay, target, is_terminal=target.time_ms == duration_ms)
        cursor = stop
    queries = (release_queries(states, clocks, previews, duration_ms, model.recovery, history,
        lookahead=model.config.lookahead, device=device) if clocks else None)
    tensor = lambda x: torch.as_tensor(x, dtype=torch.bool, device=device)
    return JointTrace(row, queries, tensor(events), tensor(forced))


def _release_only(row):
    return row is not None and not any(a in (1, 2) for a in row.actions)


def score_joint_trace(model, trace, controls, encoded_full, *, frame_count=None,
                      ln_feedback=None, recovery_preference=RecoveryPreference(head_pressure=4.)):
    """Return chosen-row and per-native-clock log factors with current weights.

    This reconstructs the ordinary ControlledSession proposal with the declared
    LN/recovery preferences. Independent response guidance, a private plan, or
    any other sampler intervention requires its own consistent factor scoring.
    Forced release times have unit event mass but retain their mark likelihood.
    Empty observed time while held contributes survival even without target rows.
    Updating parameters that also change H or a learned plan prior requires
    those trajectory factors too; this conditional score does not include them.
    """
    q = score_row_trace(model, trace.row, controls, encoded_full, frame_count=frame_count,
        ln_feedback=ln_feedback, recovery_preference=recovery_preference)
    selected = q[torch.arange(len(trace.row.targets)), trace.row.targets.cpu()]
    if trace.release is None:
        return JointTraceScores(selected, selected.new_empty((0,)))
    encoded = model.condition_audio(encoded_full, None,
        downstream=not model.config.profile_head_rate_downstream)
    past = model.temporal(trace.row.raw, trace.row.history_valid)[0]
    indices = trace.release.history_indices
    contexts = torch.where((indices >= 0)[:, None, None], past[indices.clamp_min(0)],
                           model.temporal.boundary[0].expand(len(indices), 2, -1))
    frames = torch.tensor([encoded.shape[1] if frame_count is None else frame_count], device=encoded.device)
    logits = release_logits(model, trace.release, contexts, encoded, controls,
        audio_starts=torch.zeros_like(frames), frame_counts=frames, preference=recovery_preference)
    # sample_hazards converts network logits before computing its survival mass.
    logits = logits.cpu().double()
    active = ~trace.forced.cpu()
    factors = torch.where(trace.event.cpu(), -F.softplus(-logits), -F.softplus(logits))
    factors = torch.where(active, factors, 0.)
    return JointTraceScores(selected, factors)
