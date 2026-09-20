"""Soft response calibration that preserves each head/LN-count marginal.

The base network is frozen. Two globally fitted costs change allocation only
within a complete-row composition. This does not preserve future trajectory
statistics, impose a gameplay constraint, or certify a difficulty level.
"""
import numpy as np

from ..scoped_style_modeling.dataset import ContractError
from .response import ACTIONS, HEADS, HEAD_COUNTS, LN_COUNTS, response_costs

FORMAT = 'bounded-typed/response-calibration-v1'
COMPOSITIONS = tuple((h, ln) for h in range(5) for ln in range(h + 1))
GROUPS = np.asarray([COMPOSITIONS.index((h, ln)) for h, ln in zip(HEAD_COUNTS, LN_COUNTS)])
GROUP_MASK = np.arange(len(COMPOSITIONS))[:, None] == GROUPS[None]


def response_features(states):
    """Return current and additional optimistic under30ms counts, N x256 x2.

    Unsupported actions have zero features and remain excluded by the policy.
    Only exact native clocks and externally supplied R/H candidates are read.
    """
    result = []
    for state in states:
        total, legal = response_costs(state)
        short = [any(t is not None and state.time_ms - t < 30. for t in clocks)
                 for clocks in zip(state.replay.last_lane_attack_ms, state.replay.last_lane_release_ms)]
        current = (HEADS & np.asarray(short)[None]).sum(-1)
        result.append(np.where(legal[:, None], np.stack((current, total - current), -1), 0))
    return np.asarray(result, dtype=np.float32).reshape(-1, len(ACTIONS), 2)


def calibrate(log_probs, features, weights):
    """Tilt supported log probabilities within composition, retaining its mass.

    Inputs are N x256 log probabilities, N x256 x2 costs and two coefficients.
    Empty composition groups never enter a logsumexp of all negative infinities,
    which keeps gradients finite. Constant zero coefficients without gradients
    return the base tensor exactly, including its unsupported entries.
    """
    import torch

    if (log_probs.ndim != 2 or log_probs.shape[1] != 256 or
            features.shape != (*log_probs.shape, 2) or weights.shape != (2,)):
        raise ContractError('Response calibration needs aligned 256-action probabilities and two costs')
    if not weights.requires_grad and bool((weights == 0).all()):
        return log_probs
    mask = torch.as_tensor(GROUP_MASK, device=log_probs.device)
    groups = torch.as_tensor(GROUPS, device=log_probs.device)
    valid = (torch.isfinite(log_probs)[:, None] & mask[None]).any(-1)
    before = log_probs[:, None].masked_fill(~mask[None], -torch.inf)
    after = (log_probs - features @ weights)[:, None].masked_fill(~mask[None], -torch.inf)
    # Every unused group gets the same finite dummy normalizer on both sides.
    before = before.masked_fill(~valid[..., None], 0.)
    after = after.masked_fill(~valid[..., None], 0.)
    correction = before.logsumexp(-1) - after.logsumexp(-1)
    return log_probs - features @ weights + correction[:, groups]


def fit(source_log_probs, source_features, source_targets, source_onsets,
        native_log_probs, native_features, native_families, native_groups, *,
        native_weight=.25, regularization=.0001, check=lambda: None):
    """Fit two nonnegative costs by source CE and group-mean native preference.

    Each native family must contain both zero- and positive-cost legal actions
    from exactly one composition. Zero-cost alternatives have both features zero,
    so the objective is convex. No model weights, corpus files or held-out data
    are read here. The caller owns TRAIN provenance and resource limits; check is
    called on every optimizer objective evaluation. Failure returns no bundle.
    """
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    def arrays(lp, feat):
        lp, feat = np.asarray(lp, np.float64), np.asarray(feat, np.float64)
        if (lp.ndim != 2 or lp.shape[1] != 256 or feat.shape != (*lp.shape, 2) or
                not len(lp) or np.isnan(lp).any() or np.isposinf(lp).any() or
                not np.isfinite(feat).all() or (feat < 0).any() or
                not np.isfinite(lp).any(-1).all() or not np.allclose(logsumexp(lp, -1), 0, atol=2e-6)):
            raise ContractError('Calibration cache requires normalized supported probabilities and finite nonnegative costs')
        return lp, feat
    source_lp, source_feat = arrays(source_log_probs, source_features)
    native_lp, native_feat = arrays(native_log_probs, native_features)
    targets = np.asarray(source_targets)
    family = np.asarray(native_families)
    groups = np.asarray(native_groups)
    if (targets.shape != (len(source_lp),) or targets.dtype.kind not in 'iu' or
            ((targets < 0) | (targets >= 256)).any() or type(source_onsets) is not int or
            source_onsets <= 0 or family.shape != native_lp.shape or family.dtype != np.bool_ or
            groups.shape != (len(native_lp),) or not family.any(-1).all() or
            (family & ~np.isfinite(native_lp)).any()):
        raise ContractError('Calibration labels, families and group ownership must align with supported actions')
    source_selected = source_lp[np.arange(len(source_lp)), targets]
    if not np.isfinite(source_selected).all():
        raise ContractError('Calibration source targets must be supported')
    if any(len(set(GROUPS[f])) != 1 for f in family):
        raise ContractError('A native calibration family must have one head/LN composition')
    good = family & (native_feat.sum(-1) == 0)
    if not good.any(-1).all() or not (family & ~good).any(-1).all():
        raise ContractError('Native calibration families need zero and positive response costs')
    if (not np.isfinite(native_weight) or not 0 < native_weight <= 1 or
            not np.isfinite(regularization) or regularization <= 0):
        raise ContractError('Calibration needs positive finite preference and regularization weights')
    source_family = GROUPS[None] == GROUPS[targets, None]
    source_base = np.where(source_family, source_lp, -np.inf)
    native_base = np.where(family, native_lp, -np.inf)
    source_z = logsumexp(source_base, -1)
    native_good = logsumexp(np.where(good, native_lp, -np.inf), -1)
    selected_features = source_feat[np.arange(len(source_lp)), targets]
    unique, counts = np.unique(groups, return_counts=True)
    group_counts = dict(zip(unique.tolist(), counts.tolist()))
    native_scale = np.asarray([1. / (len(unique) * group_counts[g]) for g in groups.tolist()])

    def objective(weights, details=False):
        check()
        source_logits = source_base - source_feat @ weights
        native_logits = native_base - native_feat @ weights
        sz, nz = logsumexp(source_logits, -1), logsumexp(native_logits, -1)
        source_loss = (-source_selected + selected_features @ weights + sz - source_z).sum() / source_onsets
        native_loss = ((nz - native_good) * native_scale).sum()
        source_expected = (np.exp(source_logits - sz[:, None])[..., None] * source_feat).sum(1)
        native_expected = (np.exp(native_logits - nz[:, None])[..., None] * native_feat).sum(1)
        grad = ((selected_features - source_expected).sum(0) / source_onsets -
                native_weight * (native_expected * native_scale[:, None]).sum(0) + regularization * weights)
        loss = source_loss + native_weight * native_loss + .5 * regularization * (weights @ weights)
        if not np.isfinite(loss) or not np.isfinite(grad).all():
            raise ContractError('Calibration objective or gradient became nonfinite')
        return dict(source_nll=float(source_loss), native_nll=float(native_loss), objective=float(loss)) if details else (loss, grad)

    initial = objective(np.zeros(2), details=True)
    result = minimize(objective, np.zeros(2), method='L-BFGS-B', jac=True, bounds=[(0., None)] * 2,
                      options=dict(maxiter=100, ftol=1e-12, gtol=1e-8))
    if not result.success or not np.isfinite(result.x).all():
        raise ContractError(f'Response calibration did not converge: {result.message}')
    return dict(weights=result.x.tolist(), initial=initial, final=objective(result.x, details=True),
                iterations=int(result.nit), evaluations=int(result.nfev), message=str(result.message),
                native_groups=len(unique), native_queries=len(native_lp), source_onsets=source_onsets,
                source_rows=len(source_lp), native_weight=native_weight, regularization=regularization)
