from dataclasses import asdict, replace

import numpy as np
import torch

from ensomi_model.research.controlled_audio_continuation.generation import ControlledSession
from ensomi_model.research.controlled_audio_continuation.model import ControlledAudioModel, load_model
from ensomi_model.research.controlled_audio_continuation.sampling import replay_row_scores
from ensomi_model.research.joint_audio_continuation.intervals import IntervalExample
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
from ensomi_model.research.planned_audio_continuation.intervals import collate_interval, score_interval
from ensomi_model.research.player_response.conditioning import features_at, features_before_rows, PlayerCondition
from ensomi_model.research.player_response.state import CommittedPlayState
from ensomi_model.research.typed_audio_continuation.controls import ControlSchedule, ControlSpan
from controlled_audio_continuation.test_ownership import model
from controlled_audio_continuation.test_sampling import source
from planned_audio_continuation.test_distribution import chart


def conditioned(original):
    options = dict(original.probability_options(), recovery=original.recovery, player_state=True)
    net = ControlledAudioModel(original.config, **options)
    receipt = net.load_state_dict(original.state_dict(), strict=False)
    assert receipt.missing_keys == ['player_condition.projection.weight'] and not receipt.unexpected_keys
    return net


def test_observation_features_are_causal_and_preserve_held_time_through_silence():
    prefix = [CompleteRow(0, (2, 0, 0, 0)), CompleteRow(200, (0, 1, 0, 0))]
    a = prefix+[CompleteRow(41000, (3, 0, 1, 0))]
    b = prefix+[CompleteRow(41000, (0, 0, 0, 1))]
    expected = features_at(CommittedPlayState.from_rows(prefix, 10000), 41000)
    np.testing.assert_array_equal(features_before_rows(a, [2])[0], expected)
    np.testing.assert_array_equal(features_before_rows(b, [2])[0], expected)
    assert expected[0, 12:18].tolist() == [1.]*6
    assert not expected[:, :12].any()


def test_player_projection_mirrors_with_the_two_hand_contexts():
    layer = PlayerCondition(32)
    torch.nn.init.normal_(layer.projection.weight)
    rows = [CompleteRow(0, (1, 2, 0, 0)), CompleteRow(120, (0, 0, 0, 1))]
    mirrored = [CompleteRow(r.time_ms, r.actions[::-1]) for r in rows]
    a, b = [torch.from_numpy(features_at(CommittedPlayState.from_rows(r, 300), 500))[None]
            for r in (rows, mirrored)]
    torch.testing.assert_close(layer(a), layer(b).flip(1), atol=1e-6, rtol=1e-6)


def test_checkpoint_loader_restores_the_declared_player_condition(tmp_path):
    net = conditioned(model())
    torch.nn.init.normal_(net.player_condition.projection.weight, std=.1)
    path = tmp_path/'player.pt'
    torch.save(dict(format='controlled-audio/v1', model_config=asdict(net.config),
        probability_options=net.probability_options(), model=net.state_dict()), path)
    restored = load_model(path)
    assert restored.probability_options() == net.probability_options()
    torch.testing.assert_close(restored.player_condition.projection.weight,
                               net.player_condition.projection.weight, atol=0, rtol=0)


def test_zero_projection_preserves_old_law_and_nonzero_projection_only_conditions_r1():
    torch.set_num_threads(1)
    torch.manual_seed(273)
    old = model().eval()
    net = conditioned(old).eval()
    c = source()
    controls = ControlSchedule((ControlSpan(0, 1001, stars=4, ln_fraction=.4),), old.style_names)
    batch = collate_interval(IntervalExample(c, 0, 1001), net.config, recovery=net.recovery, player_state=True)
    encoded = old.encode_audio(torch.from_numpy(c.mel)[None]).detach()
    scores = [score_interval(m, batch.inputs, None, controls=controls, encoded_full=encoded) for m in (old, net)]
    for field in ('head', 'release', 'row'):
        torch.testing.assert_close(getattr(scores[0], field), getattr(scores[1], field), atol=0, rtol=0)
    torch.nn.init.normal_(net.player_condition.projection.weight, std=.1)
    changed = score_interval(net, batch.inputs, None, controls=controls, encoded_full=encoded)
    torch.testing.assert_close(changed.head, scores[0].head, atol=0, rtol=0)
    torch.testing.assert_close(changed.release, scores[0].release, atol=0, rtol=0)
    assert not torch.equal(changed.row, scores[0].row)
    (-changed.row[torch.arange(len(batch.row_index)), batch.row_index].sum()).backward()
    assert net.player_condition.projection.weight.grad.abs().sum() > 0


def test_live_player_condition_matches_source_replay_across_no_row_boundaries():
    torch.manual_seed(274)
    net = conditioned(model()).eval()
    torch.nn.init.normal_(net.player_condition.projection.weight, std=.1)
    mel = np.random.default_rng(31).normal(size=(100, 128)).astype(np.float32)
    controls = ControlSchedule((ControlSpan(0, 1001, stars=4, ln_fraction=.7),), net.style_names)
    recorded = []
    class Recorder(ControlledSession):
        def prefer_rows(self, *args):
            q = super().prefer_rows(*args)
            recorded.append(q.clone())
            return q
    with torch.inference_mode():
        session = Recorder(net, mel, 1000, controls, seed=19, head_times=(0, 100, 220, 340, 460, 610, 850))
        for end in (377, 799, 1000):
            session.publish_to(end)
            assert session.play_state.time_ms == end
    c = replace(chart([r.time_ms for r in session.rows], [r.actions for r in session.rows], 1000), mel=mel)
    encoded = net.encode_audio(torch.from_numpy(mel)[None])
    pieces = []
    for i in range(IntervalExample(c, 0, 137).count):
        example = IntervalExample(c, i, 137)
        batch = collate_interval(example, net.config, recovery=net.recovery, player_state=True)
        scores = score_interval(net, batch.inputs, None, controls=controls, encoded_full=encoded)
        pieces.append(replay_row_scores(scores.row[:len(batch.row_index)], example, controls))
    torch.testing.assert_close(torch.cat(pieces), torch.stack(recorded), atol=3e-5, rtol=3e-6)
