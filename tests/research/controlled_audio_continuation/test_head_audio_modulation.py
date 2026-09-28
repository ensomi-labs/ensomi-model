"""Audio/control interaction in H, with unchanged row ownership and clock law."""
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from ensomi_model.research.controlled_audio_continuation.generation import ControlledSession
from ensomi_model.research.controlled_audio_continuation.model import ControlledAudioModel, load_model
from ensomi_model.research.joint_audio_continuation.intervals import IntervalExample
from ensomi_model.research.planned_audio_continuation.features import head_clocks
from ensomi_model.research.planned_audio_continuation.intervals import collate_interval, score_interval
from ensomi_model.research.typed_audio_continuation.controls import ControlSchedule, ControlSpan
from planned_audio_continuation.test_distribution import chart, config
from controlled_audio_continuation.test_sampling import source


DEVICES = ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='MPS unavailable'))]


def model(enabled):
    torch.set_num_threads(1)
    return ControlledAudioModel(replace(config(), lookahead=16, bounded_head=True,
        condition_full_holds=True, minimum_action_gap_ms=60), style_names=('tech',),
        head_audio_modulation=enabled).eval()


def controls(net):
    return ControlSchedule((ControlSpan(0, 1001, stars=3, ln_fraction=.4),
                            ControlSpan(200, 800, stars=5, style={'tech': 1})), net.style_names)


@pytest.mark.parametrize('device', DEVICES)
def test_zero_initialization_preserves_all_factors_and_strict_checkpoint_loading(tmp_path, device):
    torch.manual_seed(281090)
    base, added = model(False), model(True)
    receipt = added.load_state_dict(base.state_dict(), strict=False)
    assert receipt.missing_keys == ['head_audio_modulation.weight'] and not receipt.unexpected_keys
    base.to(device); added.to(device)
    c = source()
    encoded = base.encode_audio(torch.from_numpy(c.mel)[None].to(device))
    batch = collate_interval(IntervalExample(c, 0, 1001), base.config, device, recovery=base.recovery)
    values = [score_interval(n, batch.inputs, None, controls=controls(n), encoded_full=encoded)
              for n in (base, added)]
    for a, b in zip(vars(values[0]).values(), vars(values[1]).values()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for net in (base, added):
        path = tmp_path/('modulated.pt' if net is added else 'legacy.pt')
        torch.save(dict(format='controlled-audio/v1', model_config=asdict(net.config),
            probability_options=net.probability_options(), model=net.state_dict()), path)
        loaded = load_model(path, device=device)
        assert (loaded.head_audio_modulation is not None) == (net is added)
        for name, value in net.state_dict().items():
            torch.testing.assert_close(value, loaded.state_dict()[name], atol=0, rtol=0)


@pytest.mark.parametrize('device', DEVICES)
def test_base_can_learn_audio_control_interaction_even_at_BOS_and_preserves_residual(device):
    torch.manual_seed(281091)
    net = model(True).to(device)
    width = net.head_control.in_features
    audio = torch.randn(2, net.config.conditioned_audio_width, device=device)
    c0, c1 = torch.zeros(2, width, device=device), torch.ones(2, width, device=device)*.2
    history = torch.randn(2, 2, net.config.head_hidden, device=device)
    clocks = torch.from_numpy(head_clocks([None, 0], [100, 10000])).to(device)
    with torch.no_grad():
        net.head_base.weight.normal_(std=.2)
    def effect(control):
        return net.controlled_head_parts(audio, history, clocks, control=control)
    b0, r0, g0 = effect(c0)
    b1, r1, g1 = effect(c1)
    before = (b1-b0)[1]-(b1-b0)[0]
    torch.testing.assert_close(before, torch.zeros_like(before), atol=2e-6, rtol=0)
    gradient = torch.autograd.grad(before.square().sum()+before.sum(), net.head_audio_modulation.weight)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > .01
    with torch.no_grad():
        net.head_audio_modulation.weight.normal_(std=.03)
    a0, s0, t0 = effect(c0)
    a1, s1, t1 = effect(c1)
    assert ((a1-a0)[1]-(a1-a0)[0]).abs().max() > .01
    assert not torch.equal(a1[0], b1[0])
    for old, new in ((r0, s0), (r1, s1), (g0, t0), (g1, t1)):
        torch.testing.assert_close(old, new, atol=0, rtol=0)
    assert torch.equal(s1[0], torch.zeros_like(s1[0]))
    assert bool((s1.abs() <= net.config.head_bound*t1[:, None]).all())


@pytest.mark.parametrize('device', DEVICES)
def test_nonzero_modulation_changes_H_but_not_R_or_R1_on_identical_factual_state(device):
    torch.manual_seed(281092)
    net = model(True).to(device)
    c = source()
    encoded = net.encode_audio(torch.from_numpy(c.mel)[None].to(device))
    batch = collate_interval(IntervalExample(c, 0, 1001), net.config, device, recovery=net.recovery)
    def run():
        return score_interval(net, batch.inputs, None, controls=controls(net), encoded_full=encoded)
    before = run()
    with torch.no_grad():
        net.head_audio_modulation.weight.normal_(std=.1)
    after = run()
    assert not torch.equal(before.head, after.head)
    torch.testing.assert_close(before.release, after.release, atol=0, rtol=0)
    torch.testing.assert_close(before.row, after.row, atol=0, rtol=0)


def test_nonzero_native_H_scores_match_dense_scoring_and_publication_partitions():
    torch.manual_seed(281093)
    net = model(True)
    with torch.no_grad():
        net.head_audio_modulation.weight.normal_(std=.03)
        net.head_base.bias.fill_(-3.)
    mel = np.random.default_rng(281094).normal(size=(100, 128)).astype(np.float32)
    cc = controls(net)
    native = {}
    original = net.head_logits
    def tracked(audio, history, clocks, *, control):
        result = original(audio, history, clocks, control=control)
        for clock, c, value in zip(clocks.numpy(), control.numpy(), result.detach()):
            native[(clock.tobytes(), c.tobytes())] = value.clone()
        return result
    net.head_logits = tracked
    session = ControlledSession(net, mel, 1000, cc, seed=281095, ln_feedback=None)
    for end in (333, 701, 1000):
        session.publish_to(end)
    net.head_logits = original
    other = ControlledSession(net, mel, 1000, cc, seed=281095, ln_feedback=None)
    other.publish_to(1000)
    assert session.rows == other.rows and session.rows
    assert session.coverage == 1000 and not any(session.replay.occupancy)
    generated = replace(chart([r.time_ms for r in session.rows],
        [r.actions for r in session.rows], 1000), mel=mel)
    batch = collate_interval(IntervalExample(generated, 0, 1001), net.config, recovery=net.recovery)
    with torch.no_grad():
        encoded = net.encode_audio(torch.from_numpy(mel)[None])
        scored = score_interval(net, batch.inputs, None, controls=cc, encoded_full=encoded)
    vectors = cc.at(batch.inputs.base.timing_times.numpy(), encoding=net.control_encoding)
    for i in torch.where(batch.inputs.base.timing_valid.any(-1))[0]:
        key = (batch.inputs.head_clock[i].numpy().tobytes(), vectors[i].tobytes())
        torch.testing.assert_close(scored.head[i], native[key], atol=2e-5, rtol=2e-6)


def test_rejects_modulation_without_its_bounded_base():
    with pytest.raises(ValueError, match='bounded audio base'):
        ControlledAudioModel(config(), head_audio_modulation=True)
