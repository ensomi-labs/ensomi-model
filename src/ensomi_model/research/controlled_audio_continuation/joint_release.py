"""R1 owns waiting and release identities before a release clock is selected.

Off-H queries extend the complete-row law with a virtual all-empty action.
It means survival at that clock and is never appended to physical history.
The same nonempty energies select the release mark after the event occurs.
"""
from dataclasses import dataclass, fields

import numpy as np
import torch

from ..bounded_typed_continuation.contract import ROW_ACTIONS
from ..joint_audio_continuation.batching import interpolate_audio
from ..joint_audio_continuation.state import exact_features
from ..planned_audio_continuation.features import consequences, ln_start_times, preview_features
from ..planned_audio_continuation.spacing import allowed_rows
from ..typed_audio_continuation.response_preference import RecoveryPreference


_ALL = np.asarray(ROW_ACTIONS)
RELEASE_INDICES = np.flatnonzero(~np.isin(_ALL, (1, 2)).any(-1))
RELEASE_ACTIONS = _ALL[RELEASE_INDICES]
_RELEASES = RELEASE_ACTIONS == 3
assert tuple(RELEASE_ACTIONS[0]) == (0, 0, 0, 0)


@dataclass(frozen=True)
class ReleaseQueries:
    times: torch.Tensor
    history_indices: torch.Tensor
    exact: torch.Tensor
    legal: torch.Tensor
    occupancy: torch.Tensor
    preview: torch.Tensor
    local: torch.Tensor
    timing: torch.Tensor
    hold_starts: torch.Tensor
    duration_ms: int

    def to(self, device):
        return ReleaseQueries(**{f.name: (getattr(self, f.name).to(device)
            if isinstance(getattr(self, f.name), torch.Tensor) else getattr(self, f.name))
            for f in fields(self)})


def release_queries(replays, times, previews, duration_ms, recovery, history_indices,
                    *, lookahead=16, device='cpu'):
    """Build hypothetical release/wait queries from the unchanged actual prefix.

    The supplied history indices identify each query's last observed row, not
    the last source row before its candidate timestamp. This distinction is
    essential for finite waits extending beyond an actually observed release.
    Wait stays in the raw law even at a deadline; the scheduler conditions the
    forces that final clock separately without renormalizing earlier hazards.
    Nonempty marks retain
    physical occupancy, recovery and future-H feasibility support.
    """
    times = np.asarray(times, dtype=np.int64)
    if not (len(times) == len(replays) == len(previews) == len(history_indices)):
        raise ValueError('Joint release queries need aligned factual prefixes and clocks')
    occupied = np.asarray([s.occupancy for s in replays], dtype=bool).reshape(-1, 4)
    legal = ~(_RELEASES[None] & ~occupied[:, None]).any(-1)
    for i, (state, time, preview) in enumerate(zip(replays, times, previews)):
        legal[i] &= allowed_rows(state, int(time), preview, duration_ms, recovery, actions=RELEASE_ACTIONS)
        if time == duration_ms:
            legal[i] &= (_RELEASES == occupied[i]).all(-1)
        if not legal[i, 1:].any():
            raise ValueError('A valid release clock has no feasible nonempty R1 mark')
    legal[:, 0] = True
    local, timing = consequences(replays, times, previews, duration_ms)
    as_tensor = lambda value: torch.as_tensor(value, device=device)
    return ReleaseQueries(as_tensor(times), as_tensor(np.asarray(history_indices, np.int64)),
        as_tensor(exact_features(replays, times.tolist())), as_tensor(legal), as_tensor(occupied),
        as_tensor(preview_features(previews, times, [False]*len(times), duration_ms, lookahead)),
        as_tensor(local), as_tensor(timing), as_tensor(ln_start_times([s.open_ln_start_ms for s in replays])),
        duration_ms)


def release_logits(model, queries, contexts, encoded, controls, *, audio_starts=None,
                   frame_counts=None, preference=RecoveryPreference(head_pressure=4.), chunk_size=256,
                   candidate_cost=None):
    """Return R's native-ms hazard from the same R1 energies as its mark law.

    contexts contains current-weight row-history observations, aligned to the
    hypothetical query prefixes. Full-audio inputs match the row path. The flow
    log scale changes event odds only and cancels from conditional release marks.
    Clock likelihood plus the conditional nonempty row likelihood is therefore
    the joint wait/mark likelihood, with explicit forced deadline atoms.
    Optional candidate_cost receives native clocks and the sixteen complete
    wait/release actions. A deployment response policy must apply the same
    energy to actual row materialization; this is not the actor's raw law.
    """
    if model.release_policy != 'r1_joint' or len(contexts) != len(queries.times):
        raise ValueError('Joint release requires its model mode and aligned row contexts')
    if not len(queries.times):
        return encoded.new_empty((0,))
    indices = torch.as_tensor(RELEASE_INDICES, device=encoded.device)
    times_np = queries.times.detach().cpu().numpy()
    condition_np = controls.at(times_np, encoding=model.control_encoding)
    costs = None
    if preference is not None:
        known = condition_np[:, 2+len(controls.style_names)] > 0
        stars = np.where(known, 2*condition_np[:, 0]+4, np.nan)
        starts = queries.hold_starts.detach().cpu().numpy()
        age = np.where(starts >= 0, times_np[:, None]-starts, np.nan)
        costs = preference.cost(age, stars[:, None], 'hr') @ _RELEASES.T
        costs[times_np == queries.duration_ms] = 0
    output = []
    for first in range(0, len(queries.times), chunk_size):
        s = slice(first, first+chunk_size)
        audio = interpolate_audio(encoded, queries.times[s][None], audio_starts, frame_counts)[0]
        options = model.hold_audio_options(encoded, queries.hold_starts[s], queries.times[s],
            audio_starts=audio_starts, frame_counts=frame_counts)
        q = model.planned_row_log_probs(audio, contexts[s], queries.exact[s], queries.legal[s],
            queries.occupancy[s], queries.preview[s], queries.local[s], queries.timing[s],
            control=audio.new_tensor(condition_np[s]), candidate_indices=indices, **options)
        if costs is not None:
            q = q-q.new_tensor(costs[s])
        if candidate_cost is not None:
            extra = candidate_cost(times_np[s], RELEASE_ACTIONS)
            if extra is not None:
                q = q-q.new_tensor(extra)
        output.append(q[:, 1:].logsumexp(-1)-q[:, 0]+model.release_log_scale)
    return torch.cat(output)
