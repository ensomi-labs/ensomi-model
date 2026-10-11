import numpy as np
import pytest
import torch

from ensomi_model.export.r2_graph import R2Stepper, export_graphs
from ensomi_model.evaluation.beats import BarStart, BeatGrid, Segment
from ensomi_model.r2.common import GridArrays
from ensomi_model.r2.features import Chart, Interval, decision_factors, history_tokens, lane_query
from ensomi_model.r2.model import R2Config, R2Model
from ensomi_model.r2.sampling import continue_chart


def fixture(memory, rows=23):
    rng = np.random.default_rng(8102)
    head = 500. + np.cumsum(rng.integers(25, 400, rows))
    song = head[-1] + 500.
    grid = GridArrays.from_grid(BeatGrid((Segment(0., 500., 4),), (BarStart(0., 4),)))
    torch.manual_seed(0)
    model = R2Model(R2Config(hidden=16, levels=3, expansion=1, rank=4,
                             memory=memory, stride=3, ln_level='on', ln_length='on',
                             anchor='on', theta='on')).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.ndim >= 1 and float(parameter.abs().sum()) == 0 and 'bias' not in name:
                parameter.normal_(0, .05)
    actions, gap = continue_chart(model, head, song, grid, seed=8102)
    chart = Chart(head, song, grid, actions, gap)
    track = (Interval(0, 0., song, .6), Interval(1, 0., song, 4.))
    query_kwargs = dict(ln_level=.2, ln_length=.75, theta=np.linspace(-1, 1, 10))
    return model, chart, track, query_kwargs


@pytest.mark.parametrize('memory', ['none', 'landmarks'])
@torch.no_grad()
def test_stepper_matches_dense_history_hands_and_release_scores(memory):
    torch.set_num_threads(1)
    model, chart, track, query_kwargs = fixture(memory)
    stepper = R2Stepper(model, None, chart.K)
    dense_hands = model.window_hands(chart, np.arange(chart.K + 1), track, **query_kwargs)
    dense_history = model.encode_history(chart, chart.K)
    history = model._t(history_tokens(chart, chart.K))
    cache = model.temporal.empty_cache()
    release_count = 0
    for k in range(chart.K + 1):
        hands, logits = stepper.predict(chart, k, track, **query_kwargs)
        torch.testing.assert_close(hands, dense_hands[k:k + 1], atol=3e-6, rtol=1e-5)
        torch.testing.assert_close(logits, model.joint(dense_hands[k:k + 1]), atol=3e-6, rtol=1e-5)
        for orientation in (0, 1):
            factors = decision_factors(chart, k, 0, chart.actions[k], chart.gap[k],
                                       lane_query(chart, [k])[0], orientation)
            for factor in factors:
                expected = model.all_pair_scores(model.factor_contexts(hands, [factor]), [factor], chart, track)[0]
                actual = stepper.release(hands, factor, chart, track)
                torch.testing.assert_close(actual, expected, atol=3e-6, rtol=1e-5)
                release_count += 1
        if k < chart.K:
            stepper.commit(chart, k)
            cache = model.temporal.append(cache, history[k])
            torch.testing.assert_close(stepper.state['last'], dense_history[k], atol=3e-6, rtol=1e-5)
            for i, buffer in enumerate(cache.buffers):
                torch.testing.assert_close(stepper.state[f'buffer_{i}'], buffer, atol=0, rtol=0)
            assert int(stepper.state['rows']) == k + 1
            if memory == 'landmarks':
                count = k // model.config.stride + 1
                assert int(stepper.state['landmark_valid'].sum()) == count
                torch.testing.assert_close(stepper.state['landmarks'][:count],
                                           dense_history[:k + 1:model.config.stride], atol=3e-6, rtol=1e-5)
    assert release_count > 0


@pytest.mark.parametrize('memory', ['none', 'landmarks'])
@torch.no_grad()
def test_onnx_stepper_matches_torch_with_dynamic_candidate_and_landmark_counts(tmp_path, memory):
    ort = pytest.importorskip('onnxruntime')
    pytest.importorskip('onnx')
    torch.set_num_threads(1)
    model, _, _, _ = fixture(memory)
    schemas = export_graphs(model, tmp_path, max_head_rows=33)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    sessions = {name: ort.InferenceSession(schema['path'], sess_options=options,
                                          providers=['CPUExecutionProvider'])
                for name, schema in schemas.items()}
    for name, session in sessions.items():
        assert [value.name for value in session.get_inputs()] == [value['name'] for value in schemas[name]['inputs']]
        assert [value.name for value in session.get_outputs()] == [value['name'] for value in schemas[name]['outputs']]
    calls = dict.fromkeys(sessions, 0)
    candidate_counts = set()

    def graph_call(name, feeds, fn):
        calls[name] += 1
        expected = fn()
        arrays = {key: value.detach().cpu().numpy() for key, value in feeds.items()}
        actual = sessions[name].run(None, arrays)
        result = {key: torch.from_numpy(value) for key, value in zip(expected, actual)}
        for key in expected:
            torch.testing.assert_close(result[key], expected[key], atol=1e-5, rtol=2e-5)
        return result

    # Both capacities differ from the export example; neither needs a new graph.
    for rows in (7, 23):
        _, chart, track, query_kwargs = fixture(memory, rows)
        stepper = R2Stepper(model, graph_call, chart.K)
        dense = model.window_hands(chart, np.arange(chart.K + 1), track, **query_kwargs)
        for k in range(chart.K + 1):
            hands, logits = stepper.predict(chart, k, track, **query_kwargs)
            torch.testing.assert_close(hands, dense[k:k + 1], atol=1e-5, rtol=2e-5)
            torch.testing.assert_close(logits, model.joint(dense[k:k + 1]), atol=1e-5, rtol=2e-5)
            for orientation in (0, 1):
                factors = decision_factors(chart, k, 0, chart.actions[k], chart.gap[k],
                                           lane_query(chart, [k])[0], orientation)
                for factor in factors:
                    candidate_counts.add(len(factor.times))
                    stepper.release(hands, factor, chart, track)
            if k < chart.K:
                stepper.commit(chart, k)
    assert all(count > 0 for count in calls.values())
    assert len(candidate_counts) > 1
