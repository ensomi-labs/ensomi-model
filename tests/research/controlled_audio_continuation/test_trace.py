import numpy as np
import pytest
import torch

from ensomi_model.research.controlled_audio_continuation.generation import ControlledSession
from ensomi_model.research.controlled_audio_continuation.sampling import replay_row_scores
from ensomi_model.research.controlled_audio_continuation.trace import collate_row_trace, score_row_trace
from ensomi_model.research.oracle_time_continuation.replay import ExactReplayState, commit
from ensomi_model.research.planned_audio_continuation.intervals import collate_interval, score_interval
from ensomi_model.research.typed_audio_continuation.controls import ControlSchedule, ControlSpan
from controlled_audio_continuation.test_ownership import model
from controlled_audio_continuation.test_player_condition import conditioned
from controlled_audio_continuation.test_sampling import source


class Scope:
    def __init__(self, chart, a, b):
        self.chart, self.start_ms, self.end_ms = chart, a, b
        self.weight_per_second = 1000/(b-a)


def setup(device):
    torch.set_num_threads(1)
    torch.manual_seed(275)
    net = conditioned(model()).to(device).eval()
    torch.nn.init.normal_(net.player_condition.projection.weight, std=.1)
    c = source()
    rows = tuple(c.source.row(i) for i in range(len(c.source.rows)))
    heads = tuple(r.time_ms for r in rows if any(a in (1, 2) for a in r.actions))
    controls = ControlSchedule((ControlSpan(0, 1001, stars=4, ln_fraction=.7),
                                ControlSpan(400, 650, ln_fraction=.3)), net.style_names)
    encoded = net.encode_audio(torch.from_numpy(c.mel)[None].to(device)).detach()
    return net, c, rows, heads, controls, encoded


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='MPS unavailable'))])
def test_open_private_trace_matches_full_source_row_likelihood_and_gradients(device):
    net, c, rows, heads, controls, encoded = setup(device)
    scope = Scope(c, 100, 601)
    batch = collate_interval(scope, net.config, device, recovery=net.recovery)
    full = score_interval(net, batch.inputs, None, controls=controls, encoded_full=encoded)
    expected = replay_row_scores(full.row[:len(batch.row_index)], scope, controls)
    private = tuple(r for r in rows if r.time_ms < scope.end_ms)
    replay = ExactReplayState()
    for row in private:
        replay = commit(replay, row)
    assert any(replay.occupancy)
    trace = collate_row_trace(private, heads, scope.start_ms, scope.end_ms, c.duration_ms, net, device=device)
    actual = score_row_trace(net, trace, controls, encoded)
    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-5)
    losses = [-q[torch.arange(len(trace.targets)), trace.targets.cpu()].sum() for q in (expected, actual)]
    gradients = [torch.autograd.grad(loss, net.player_condition.projection.weight)[0] for loss in losses]
    torch.testing.assert_close(*gradients, atol=3e-4, rtol=3e-5)
    assert gradients[0].abs().sum() > 0


def test_private_trace_scoring_is_partition_independent_including_empty_intervals():
    net, c, rows, heads, controls, encoded = setup('cpu')
    full = collate_row_trace(rows, heads, 0, 1001, c.duration_ms, net)
    expected = score_row_trace(net, full, controls, encoded)
    pieces = []
    for a in range(0, 1001, 137):
        trace = collate_row_trace(rows, heads, a, min(a+137, 1001), c.duration_ms, net)
        pieces.append(score_row_trace(net, trace, controls, encoded))
    torch.testing.assert_close(torch.cat(pieces), expected, atol=3e-5, rtol=3e-6)


def test_native_probabilities_replay_before_unresolved_hold_endpoints():
    torch.manual_seed(276)
    net = conditioned(model()).eval()
    torch.nn.init.normal_(net.player_condition.projection.weight, std=.1)
    mel = np.random.default_rng(32).normal(size=(120, 128)).astype(np.float32)
    controls = ControlSchedule((ControlSpan(0, 1201, stars=4, ln_fraction=.8),), net.style_names)
    recorded = []
    class Recorder(ControlledSession):
        def prefer_rows(self, *args):
            value = super().prefer_rows(*args)
            recorded.append(value.clone())
            return value
    heads = tuple(range(0, 1200, 130))
    with torch.inference_mode():
        session = Recorder(net, mel, 1200, controls, seed=22, head_times=heads)
        session.publish_to(1200)
    replay, cuts = ExactReplayState(), []
    for i, row in enumerate(session.rows):
        replay = commit(replay, row)
        if any(replay.occupancy) and 400 < row.time_ms < 1000:
            cuts.append((i+1, row.time_ms+1))
    assert cuts
    count, end = cuts[-1]
    trace = collate_row_trace(session.rows[:count], heads, 0, end, 1200, net)
    encoded = net.encode_audio(torch.from_numpy(mel)[None])
    q = score_row_trace(net, trace, controls, encoded)
    torch.testing.assert_close(q, torch.stack(recorded[:count]), atol=3e-5, rtol=3e-6)
