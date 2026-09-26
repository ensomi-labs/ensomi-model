"""Score-function credit for conditional response and trajectory preservation."""
import numpy as np


def paired_gap_coefficients(outcomes, target_gap):
    """Return coefficients for squared error of the expected high-minus-low gap.

    outcomes has shape (2, draws), ordered low/high, with at least two draws.
    Draw indices must be independent; low/high may share randomness within an
    index. Multiply each coefficient by that trajectory's chosen log-probability
    sum and add. The returned coefficients already include the draw average.

    Excluding the entire current pair keeps both the gap estimate and the
    centering baseline independent of its sampled action. This estimates a
    squared mean response, without adding a penalty on outcome variance. Inputs
    are observed outcomes, not differentiable predictions.
    """
    values = np.asarray(outcomes, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 2 or values.shape[1] < 2:
        raise ValueError('Paired response requires low/high by two or more draws')
    count = values.shape[1]
    gaps = values[1]-values[0]-target_gap
    other_gap = (gaps.sum()-gaps)/(count-1)
    other_value = (values.sum(-1, keepdims=True)-values)/(count-1)
    return 2/count*np.array([[-1.], [1.]])*other_gap*(values-other_value)


def trajectory_kl_coefficients(chosen, reference, baseline=0.):
    """Return detached causal credit for KL(current trajectory || reference).

    Inputs are aligned vectors of actual chosen row log probabilities on the
    same generated history and finite support. Fixed environment factors must
    cancel between policies. The baseline must be independent of this trajectory.

    Multiply by current chosen row log probabilities and sum. Cost-to-go includes
    effects on later visited states. The explicit derivative of a sampled log-q
    term has zero expectation, so the coefficient is detached. Callers own scope,
    duration and draw normalization; future decisions outside that scope receive
    no credit from this term.
    """
    ratio = (chosen-reference).detach()
    return (ratio.flip(0).cumsum(0).flip(0)-baseline).detach()
