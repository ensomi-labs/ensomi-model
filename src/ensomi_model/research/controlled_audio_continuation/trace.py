"""Differentiable R1 scoring on actual prefixes and possibly open private futures.

Only the supplied timing-only H plan can provide future chart information.
The true audio duration remains distinct from a local response horizon.
"""
from dataclasses import dataclass

import numpy as np
import torch

from ..bounded_typed_continuation.contract import ROW_ACTIONS
from ..bounded_typed_continuation.features import content_features
from ..joint_audio_continuation.batching import interpolate_audio
from ..joint_audio_continuation.intervals import _pad_first
from ..joint_audio_continuation.state import exact_features
from ..oracle_time_continuation.replay import ExactReplayState, commit
from ..planned_audio_continuation.features import (
    consequences, ln_start_times, preview_after, preview_features, row_support,
)
from ..planned_audio_continuation.spacing import allowed_rows
from ..player_response.conditioning import features_before_rows
from ..scoped_style_modeling.dataset import ContractError
from ..typed_audio_continuation.response_preference import RecoveryPreference
from .allocation import LnAmountFeedback
from .sampling import replay_trace_scores


@dataclass(frozen=True)
class RowTrace:
    rows: tuple
    times: np.ndarray
    targets: torch.Tensor
    raw: torch.Tensor
    history_valid: torch.Tensor
    history_indices: torch.Tensor
    exact: torch.Tensor
    legal: torch.Tensor
    occupancy: torch.Tensor
    preview: torch.Tensor
    local: torch.Tensor
    timing: torch.Tensor
    response_allowed: torch.Tensor
    hold_starts: torch.Tensor
    player_features: torch.Tensor
    start_ms: int
    end_ms: int
    duration_ms: int
    history_boundary: int = 0


def collate_row_trace(rows, head_times, start_ms, end_ms, duration_ms, model, *, device='cpu'):
    """Collate [start,end) rows using their own complete physical history.

    Supply the complete H timing plan through real audio termination, and rows
    from BOS through the requested exclusive endpoint. An open LN at a local
    endpoint is valid. Targets and historical content use actual materialized
    actions; every query excludes its own target action. Padding is unobserved.
    """
    if not 0 <= start_ms < end_ms <= duration_ms+1 or model.config.row_factorization != 'flat':
        raise ContractError('Row traces require a native interval and complete-row R1')
    rows = tuple(r for r in rows if r.time_ms < end_ms)
    times = np.array([r.time_ms for r in rows], dtype=np.int64)
    first = int(np.searchsorted(times, start_ms))
    indices = np.arange(first, len(rows))
    query_times = times[indices]
    heads = np.asarray(head_times, dtype=np.int64)
    roles = np.isin(times, heads)
    if tuple(times[roles]) != tuple(heads[heads < end_ms]):
        raise ContractError('A trace must materialize every past and proposed H')
    replay, replays = ExactReplayState(), []
    for i, row in enumerate(rows):
        if roles[i] != any(a in (1, 2) for a in row.actions):
            raise ContractError('A trace row disagrees with its timing-only head role')
        if i >= first:
            replays.append(replay)
        replay = commit(replay, row, is_terminal=row.time_ms == duration_ms)
    previews = [preview_after(heads, t, model.config.lookahead) for t in query_times]
    legal = row_support(replays, query_times, roles[indices], previews, duration_ms)
    response = (np.stack([allowed_rows(s, int(t), p, duration_ms, model.recovery)
                         for s, t, p in zip(replays, query_times, previews)])
                if len(indices) else np.empty((0, 256), bool))
    targets = np.array([ROW_ACTIONS.index(r.actions) for r in rows[first:]], dtype=np.int64)
    if any(not (legal[i, k] and response[i, k]) for i, k in enumerate(targets)):
        raise ContractError('An actual trace row is outside native R1 support')
    history_start = max(0, first-model.temporal.config.receptive_tokens)
    raw = content_features(rows[history_start:],
        [rows[i-1].time_ms if i else None for i in range(history_start, len(rows))],
        [[None]*4 for _ in range(history_start, len(rows))])
    local, timing = consequences(replays, query_times, previews, duration_ms)
    preview = preview_features(previews, query_times, roles[indices], duration_ms, model.config.lookahead)
    tensor = lambda a: torch.as_tensor(a, device=device)
    pad = lambda a: tensor(_pad_first(a, 64, edge=True))
    return RowTrace(rows, query_times, tensor(targets), tensor(_pad_first(raw)[None]),
        tensor(_pad_first(np.ones(len(raw), bool))[None]), pad(indices-history_start-1),
        pad(exact_features(replays, query_times.tolist())), pad(legal),
        pad(np.asarray([s.occupancy for s in replays], bool).reshape(-1, 4)), pad(preview),
        pad(local), pad(timing), pad(response),
        pad(ln_start_times([s.open_ln_start_ms for s in replays])),
        pad(features_before_rows(rows, indices)), start_ms, end_ms, duration_ms)


def score_row_trace(model, trace, controls, encoded_full, *, frame_count=None,
                    ln_feedback=LnAmountFeedback(),
                    recovery_preference=RecoveryPreference(head_pressure=4.)):
    """Return native row log probabilities using a complete-song audio encoding.

    Frozen audio encodings may be reused across R1 updates. The trace contains
    deterministic facts/raw rows, never cached neural history from old weights.
    This scores native proposals, not the response planner's selected policy.
    frame_count declares the real frame count when the complete encoding is
    padded. Optional preferences use native defaults; None disables a preference.
    """
    n = len(trace.times)
    if not n:
        return encoded_full.new_empty((0, 256)).cpu().double()
    encoded = model.condition_audio(encoded_full, None,
        downstream=not model.config.profile_head_rate_downstream)
    frames = encoded.shape[1] if frame_count is None else frame_count
    frame_counts = torch.tensor([frames], device=encoded.device)
    audio_starts = torch.zeros_like(frame_counts)
    query_times = torch.as_tensor(_pad_first(trace.times, 64, edge=True), device=encoded.device)
    audio = interpolate_audio(encoded, query_times[None], audio_starts, frame_counts)[0]
    history = model.temporal(trace.raw, trace.history_valid)[0]
    indices = trace.history_indices
    context = torch.where((indices >= 0)[:, None, None], history[indices.clamp_min(0)],
                          model.temporal.boundary[trace.history_boundary].expand(len(indices), 2, -1))
    options = {} if model.player_condition is None else dict(player_features=trace.player_features)
    options.update(model.hold_audio_options(encoded, trace.hold_starts, query_times,
        audio_starts=audio_starts, frame_counts=frame_counts))
    q = model.planned_row_log_probs(audio, context, trace.exact, trace.legal, trace.occupancy,
        trace.preview, trace.local, trace.timing, control=audio.new_tensor(
            controls.at(query_times.detach().cpu().numpy(), encoding=model.control_encoding)),
        response_allowed=trace.response_allowed, **options)[:n]
    return replay_trace_scores(q, trace.rows, controls, trace.start_ms, trace.end_ms,
        trace.duration_ms, ln_feedback=ln_feedback, recovery_preference=recovery_preference)
