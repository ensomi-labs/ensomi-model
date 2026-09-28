"""Exact diagnostic factors of a deployed complete-row probability law."""
import numpy as np

from ..bounded_typed_continuation.contract import ROW_ACTIONS


_ACTIONS = np.asarray(ROW_ACTIONS)
_HEAD_SIGNATURE = np.where(_ACTIONS == 3, 0, _ACTIONS)
_RELEASE_COUNT = (_ACTIONS == 3).sum(-1)


def _logsumexp(values):
    maximum = values.max(-1)
    return maximum+np.log(np.exp(values-maximum[:, None]).sum(-1))


def row_likelihood_parts(log_probabilities, observed_actions):
    """Split factual row NLL into head, release-count and release-identity terms.

    Input is [query,256] normalized log probability in ROW_ACTIONS order and
    aligned [query,4] observed actions. The caller owns query-clock alignment,
    physical support and the inclusion of deployed sampling preferences. Every
    observed row must have finite probability; unsupported targets raise rather
    than turning an undefined conditional into a zero loss. Masked alternatives
    may be -infinity. Other nonfinite values and normalization errors above
    2e-5 raise ValueError; smaller floating-point errors are renormalized.

    U fixes the exact current TAP/LN heads and columns, K the number released,
    and V the released subset. Returned per-query nats satisfy
    row_nll = head_nll + release_count_nll + release_identity_nll. The last two
    condition on U and (U,K), respectively. These target conditions diagnose
    a joint law; they are not extra inputs to generation or gameplay scores.
    identity_choices counts finite alternatives at (U,K), including extremely
    small probabilities. A singleton has zero identity loss, not demonstrated
    coordination skill. Empty aligned queries return empty arrays.
    """
    lp = np.asarray(log_probabilities, dtype=np.float64)
    observed = np.asarray(observed_actions)
    if (lp.ndim != 2 or lp.shape[1] != len(_ACTIONS) or
            observed.shape != (len(lp), 4) or
            not np.isin(observed, np.arange(4)).all()):
        raise ValueError('Row factorization requires aligned probabilities and four action codes')
    if np.isnan(lp).any() or np.isposinf(lp).any() or not np.isfinite(lp).any(-1).all():
        raise ValueError('Row probabilities require finite support and no NaN or positive infinity')
    normalizer = _logsumexp(lp)
    errors = np.abs(np.expm1(normalizer))
    if np.any(errors > 2e-5):
        raise ValueError('Row probabilities must be normalized')
    lp = lp-normalizer[:, None]
    targets = (observed[:, None] == _ACTIONS[None]).all(-1).argmax(-1)
    joint = lp[np.arange(len(lp)), targets]
    if not np.isfinite(joint).all():
        raise ValueError('Observed rows must have finite probability for conditional factorization')
    same_head = (_HEAD_SIGNATURE[None] == _HEAD_SIGNATURE[targets, None]).all(-1)
    same_count = _RELEASE_COUNT[None] == _RELEASE_COUNT[targets, None]
    head = _logsumexp(np.where(same_head, lp, -np.inf))
    head_count = _logsumexp(np.where(same_head & same_count, lp, -np.inf))
    return dict(row_nll=-joint, head_nll=-head,
                release_count_nll=head-head_count,
                release_identity_nll=head_count-joint,
                identity_choices=(same_head & same_count & np.isfinite(lp)).sum(-1),
                normalization_error=errors)
