import numpy as np
import pytest

from ensomi_model.research.bounded_typed_continuation.contract import ROW_ACTIONS
from ensomi_model.research.gameplay_evaluation.row_likelihood import row_likelihood_parts


def law(probabilities):
    result = np.full((1, len(ROW_ACTIONS)), -np.inf)
    for row, p in probabilities.items():
        result[0, ROW_ACTIONS.index(row)] = np.log(p)
    return result


def example():
    return law({(0, 0, 1, 0): .1, (3, 0, 1, 0): .2,
                (0, 3, 1, 0): .4, (3, 3, 1, 0): .1,
                (0, 0, 0, 1): .2})


def test_exact_head_count_and_identity_chain():
    parts = row_likelihood_parts(example(), [[3, 0, 1, 0]])
    assert parts['head_nll'] == pytest.approx([-np.log(.8)])
    assert parts['release_count_nll'] == pytest.approx([-np.log(.6/.8)])
    assert parts['release_identity_nll'] == pytest.approx([-np.log(.2/.6)])
    assert parts['identity_choices'].tolist() == [2]
    assert parts['row_nll'] == pytest.approx(
        parts['head_nll']+parts['release_count_nll']+parts['release_identity_nll'])


def test_release_count_energy_changes_count_but_not_identity_odds():
    q = example()
    counts = np.array([row.count(3) for row in ROW_ACTIONS])
    altered = q+2*counts
    altered -= np.log(np.exp(altered).sum())
    before, after = [row_likelihood_parts(p, [[3, 0, 1, 0]]) for p in (q, altered)]
    assert not np.allclose(before['release_count_nll'], after['release_count_nll'])
    assert before['release_identity_nll'] == pytest.approx(after['release_identity_nll'])


def test_mirror_preserves_diagnostic_losses_and_support_sizes():
    q = example()
    mirrored = np.empty_like(q)
    for index, row in enumerate(ROW_ACTIONS):
        mirrored[:, ROW_ACTIONS.index(row[::-1])] = q[:, index]
    left = row_likelihood_parts(q, [[3, 0, 1, 0]])
    right = row_likelihood_parts(mirrored, [[0, 1, 0, 3]])
    for field in left:
        assert left[field] == pytest.approx(right[field])


def test_extremely_small_probabilities_keep_finite_losses_and_support():
    q = np.full((1, 256), -np.inf)
    q[0, ROW_ACTIONS.index((3, 0, 1, 0))] = 0.
    q[0, ROW_ACTIONS.index((0, 3, 1, 0))] = -1000.
    result = row_likelihood_parts(q, [[0, 3, 1, 0]])
    assert result['row_nll'].tolist() == [1000.]
    assert result['release_identity_nll'].tolist() == [1000.]
    assert result['identity_choices'].tolist() == [2]


def test_forced_identity_and_empty_queries_do_not_invent_evidence():
    result = row_likelihood_parts(law({(3, 0, 0, 0): 1.}), [[3, 0, 0, 0]])
    assert result['release_identity_nll'].tolist() == [0.]
    assert result['identity_choices'].tolist() == [1]
    empty = row_likelihood_parts(np.empty((0, 256)), np.empty((0, 4), dtype=int))
    assert all(value.shape == (0,) for value in empty.values())


def test_normalization_roundoff_and_unsupported_target_have_distinct_meanings():
    exact = row_likelihood_parts(example(), [[3, 0, 1, 0]])
    rounded = row_likelihood_parts(example()+1e-7, [[3, 0, 1, 0]])
    assert rounded['row_nll'] == pytest.approx(exact['row_nll'])
    assert rounded['normalization_error'][0] > 0
    with pytest.raises(ValueError, match='normalized'):
        row_likelihood_parts(example()+.01, [[3, 0, 1, 0]])
    with pytest.raises(ValueError, match='finite probability'):
        row_likelihood_parts(example(), [[2, 0, 0, 0]])
    with pytest.raises(ValueError, match='finite support'):
        row_likelihood_parts(np.full((1, 256), -np.inf), [[3, 0, 1, 0]])
