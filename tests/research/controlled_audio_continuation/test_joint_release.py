from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from ensomi_model.research.bounded_typed_continuation.contract import ROW_ACTIONS
from ensomi_model.research.controlled_audio_continuation.model import ControlledAudioModel, load_model
from ensomi_model.research.controlled_audio_continuation.generation import ControlledSession
from ensomi_model.research.controlled_audio_continuation.joint_release import (
    RELEASE_INDICES, release_queries, release_logits,
)
from ensomi_model.research.joint_audio_continuation.batching import interpolate_audio
from ensomi_model.research.joint_audio_continuation.intervals import IntervalExample
from ensomi_model.research.oracle_time_continuation.replay import ExactReplayState, commit
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
from ensomi_model.research.planned_audio_continuation.counts import COUNT_MARKS
from ensomi_model.research.planned_audio_continuation.features import HeadPreview
from ensomi_model.research.planned_audio_continuation.intervals import collate_interval, score_interval, interval_losses
from ensomi_model.research.typed_audio_continuation.controls import ControlSchedule, ControlSpan
from ensomi_model.research.typed_audio_continuation.program import Recovery
from ensomi_model.research.typed_audio_continuation.response_preference import RecoveryPreference
from planned_audio_continuation.test_distribution import chart, config


def model(**options):
    torch.set_num_threads(1)
    return ControlledAudioModel(replace(config(), lookahead=16, bounded_head=True,
        condition_full_holds=True, minimum_action_gap_ms=60), style_names=('tech',),
        recovery=Recovery(60, 25, 21), ln_conditioning='direct', release_policy='r1_joint', **options)


def fixture():
    net = model()
    state = commit(ExactReplayState(), CompleteRow(0, (2, 2, 0, 0)))
    queries = release_queries([state], [100], [HeadPreview((300, 600), True)], 1000,
                              net.recovery, [-1])
    encoded = torch.randn(1, 101, net.config.conditioned_audio_width)
    controls = ControlSchedule((ControlSpan(0, 1001, stars=4, ln_fraction=.4),), net.style_names)
    context = net.temporal.boundary[0].expand(1, 2, -1)
    return net, queries, context, encoded, controls


def scores(net, query, context, encoded, controls, *, dense=False, include_wait=True):
    audio = interpolate_audio(encoded, query.times[None])[0]
    allowed = query.legal.clone()
    if not include_wait:
        allowed[:, 0] = False
    selection = {}
    if dense:
        full = torch.zeros(len(allowed), 256, dtype=torch.bool)
        full[:, RELEASE_INDICES] = allowed
        allowed = full
    else:
        selection['candidate_indices'] = torch.tensor(RELEASE_INDICES)
    return net.planned_row_log_probs(audio, context, query.exact, allowed, query.occupancy,
        query.preview, query.local, query.timing,
        control=torch.tensor(controls.at(query.times.numpy(), encoding=net.control_encoding)), **selection)


def test_release_only_candidates_match_dense_row_law_and_gradients():
    torch.manual_seed(731)
    net, query, context, encoded, controls = fixture()
    dense = scores(net, query, context, encoded, controls, dense=True)
    fast = scores(net, query, context, encoded, controls)
    torch.testing.assert_close(fast, dense[:, RELEASE_INDICES], atol=2e-6, rtol=2e-6)
    target = int(torch.nonzero(query.legal[0, 1:])[0, 0])+1
    params = tuple(net.parameters())
    left = torch.autograd.grad(-dense[0, RELEASE_INDICES[target]], params, allow_unused=True, retain_graph=True)
    right = torch.autograd.grad(-fast[0, target], params, allow_unused=True)
    for a, b, p in zip(left, right, params):
        torch.testing.assert_close(torch.zeros_like(p) if a is None else a,
                                   torch.zeros_like(p) if b is None else b, atol=3e-6, rtol=2e-5)


def test_event_hazard_and_actual_row_mark_are_one_joint_law():
    torch.manual_seed(94)
    net, query, context, encoded, controls = fixture()
    q = scores(net, query, context, encoded, controls)
    scaled = q.clone()
    scaled[:, 1:] += net.release_log_scale
    joint = scaled.log_softmax(-1)
    logits = release_logits(net, query, context, encoded, controls, preference=None)
    torch.testing.assert_close(-torch.nn.functional.softplus(logits), joint[:, 0], atol=2e-6, rtol=2e-6)
    conditional = joint[:, 1:]-joint[:, 1:].logsumexp(-1, keepdim=True)
    actual = scores(net, query, context, encoded, controls, dense=True, include_wait=False)
    torch.testing.assert_close(conditional, actual[:, RELEASE_INDICES[1:]], atol=2e-6, rtol=2e-6)


def test_single_held_finger_still_has_a_learned_wait_decision():
    net = model()
    for p in net.parameters():
        torch.nn.init.zeros_(p)
    state = commit(ExactReplayState(), CompleteRow(0, (2, 0, 0, 0)))
    query = release_queries([state], [100], [HeadPreview((300,), True)], 1000, net.recovery, [-1])
    context = net.temporal.boundary[0].expand(1, 2, -1)
    encoded = torch.zeros(1, 101, net.config.conditioned_audio_width)
    controls = ControlSchedule((ControlSpan(0, 1001, stars=4),), net.style_names)
    before = release_logits(net, query, context, encoded, controls, preference=None)
    with torch.no_grad():
        net.composition.readout[-1].bias[COUNT_MARKS.index((0, 0, 0))] += 3
    after = release_logits(net, query, context, encoded, controls, preference=None)
    torch.testing.assert_close(after-before, torch.tensor([-3.]), atol=1e-6, rtol=1e-6)
    mark = scores(net, query, context, encoded, controls, include_wait=False).exp()
    assert mark[0].max() == 1


def test_short_hold_preference_changes_event_odds_even_for_one_release_mark():
    net = model()
    state = commit(ExactReplayState(), CompleteRow(0, (2, 0, 0, 0)))
    query = release_queries([state], [30], [HeadPreview((300,), True)], 1000, net.recovery, [-1])
    context = net.temporal.boundary[0].expand(1, 2, -1)
    encoded = torch.zeros(1, 101, net.config.conditioned_audio_width)
    controls = ControlSchedule((ControlSpan(0, 1001, stars=4),), net.style_names)
    preference = RecoveryPreference(head_pressure=4.)
    raw = release_logits(net, query, context, encoded, controls, preference=None)
    preferred = release_logits(net, query, context, encoded, controls, preference=preference)
    expected = float(preference.cost(30, 4, 'hr'))
    torch.testing.assert_close(preferred-raw, torch.tensor([-expected]), atol=2e-6, rtol=2e-6)
    assert expected > 0


def test_joint_clock_reads_the_real_row_history_while_H_stays_timing_only():
    torch.manual_seed(928)
    net = model()
    torch.nn.init.normal_(net.composition.readout[-1].weight, std=.05)
    controls = ControlSchedule((ControlSpan(0, 501, stars=4),), net.style_names)
    a = chart([0, 100, 400], [(2, 0, 0, 0), (0, 1, 0, 0), (3, 0, 0, 0)], 500)
    b = chart([0, 100, 400], [(2, 0, 0, 0), (0, 0, 1, 1), (3, 0, 0, 0)], 500)
    batches = [collate_interval(IntervalExample(c, 0, 501), net.config, recovery=net.recovery,
                               release_policy='r1_joint') for c in (a, b)]
    coarse = net.encode_coarse(torch.from_numpy(a.mel)[None])
    outputs = [score_interval(net, x.inputs, coarse, controls=controls) for x in batches]
    torch.testing.assert_close(outputs[0].head, outputs[1].head, rtol=0, atol=0)
    assert not torch.allclose(outputs[0].release, outputs[1].release)
    interval_losses(outputs[0], batches[0])[-1].backward()
    assert net.release_log_scale.grad is not None and torch.isfinite(net.release_log_scale.grad)
    assert net.temporal.boundary.grad is not None
    assert all(p.grad is None for p in net.release_clock.parameters())


def test_raw_survival_keeps_deadline_atoms_and_never_commits_wait_rows():
    net = model()
    with torch.no_grad():
        net.composition.readout[-1].weight.zero_()
        net.composition.readout[-1].bias.fill_(-20)
        net.composition.readout[-1].bias[COUNT_MARKS.index((4, 4, 0))] = 20
        net.release_log_scale.fill_(-100)
    controls = ControlSchedule((ControlSpan(0, 301, stars=4),), net.style_names)
    session = ControlledSession(net, np.zeros((31, 128), np.float32), 300, controls,
        seed=73, head_times=[0, 100], ln_feedback=None, max_seconds=30)
    session.publish_to(300)
    assert session.rows[0].actions == (2, 2, 2, 2)
    assert any(r.time_ms == 75 and 3 in r.actions for r in session.rows)
    assert all(any(r.actions) for r in session.rows)
    assert session.conditioned_waits == 0 and session.deadline_events >= 1
    assert not any(session.replay.occupancy)


def test_mode_and_flow_scale_survive_strict_checkpoint_reload(tmp_path):
    net = model()
    path = tmp_path/'joint.pt'
    torch.save(dict(format='controlled-audio/v1', model_config=asdict(net.config),
                    probability_options=net.probability_options(), model=net.state_dict()), path)
    loaded = load_model(path)
    assert loaded.release_policy == 'r1_joint'
    torch.testing.assert_close(loaded.release_log_scale, net.release_log_scale)


def test_native_joint_hazards_replay_with_identical_prefixes_and_publication_cuts():
    torch.manual_seed(939)
    net = model(hold_audio_width=4)
    torch.nn.init.normal_(net.composition.readout[-1].weight, std=.025)
    mel = np.random.default_rng(91).normal(size=(100, 128)).astype(np.float32)
    controls = ControlSchedule((ControlSpan(0, 1001, stars=4, ln_fraction=.6),
                                ControlSpan(303, 507, ln_fraction=.2)), net.style_names)
    heads = (0, 100, 200, 300, 500, 750, 900)
    records = {}
    class Recorder(ControlledSession):
        def release_clock_logits(self, anchors, native, projection, preview, previous, valid):
            values = super().release_clock_logits(anchors, native, projection, preview, previous, valid)
            for t, q in zip(native.cpu().numpy().reshape(-1)[valid.reshape(-1)],
                            values.detach().cpu().numpy()[valid.reshape(-1)]):
                records[(self.replay.row_count, int(t))] = float(q)
            return values
    a = Recorder(net, mel, 1000, controls, head_times=heads, seed=78, ln_feedback=None)
    a.publish_to(1000)
    b = ControlledSession(net, mel, 1000, controls, head_times=heads, seed=78, ln_feedback=None)
    for end in (*range(137, 1000, 137), 1000):
        b.publish_to(end)
    assert a.rows == b.rows and records
    generated = chart([r.time_ms for r in a.rows], [r.actions for r in a.rows], 1000)
    batch = collate_interval(IntervalExample(generated, 0, 1001), net.config,
                            recovery=net.recovery, release_policy='r1_joint')
    assert not batch.inputs.release_waits
    with torch.no_grad():
        encoded = net.encode_audio(torch.from_numpy(mel)[None])
        scored = score_interval(net, batch.inputs, None, controls=controls, encoded_full=encoded)
    queries = batch.inputs.joint_release_queries
    expected = torch.tensor([records[(int(i)+1, int(t))]
                             for i, t in zip(queries.history_indices, queries.times)])
    observed = scored.release.flatten()[batch.inputs.joint_release_destinations]
    torch.testing.assert_close(observed, expected, atol=3e-5, rtol=3e-6)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS unavailable')
def test_joint_release_full_audio_loss_and_gradient_match_cpu_on_mps():
    torch.manual_seed(948)
    cpu = model(hold_audio_width=4)
    torch.nn.init.normal_(cpu.composition.readout[-1].weight, std=.025)
    gpu = model(hold_audio_width=4).to('mps')
    gpu.load_state_dict(cpu.state_dict())
    source = chart([0, 100, 250, 400],
                   [(2, 2, 0, 0), (0, 0, 1, 0), (3, 0, 0, 1), (0, 3, 0, 0)], 500)
    controls = ControlSchedule((ControlSpan(0, 501, stars=4, ln_fraction=.4),), cpu.style_names)
    results = []
    for device, net in (('cpu', cpu), ('mps', gpu)):
        mel = torch.from_numpy(source.mel)[None].to(device)
        encoded = net.encode_audio(mel)
        batch = collate_interval(IntervalExample(source, 0, 501), net.config, device,
                                recovery=net.recovery, release_policy='r1_joint')
        scores = score_interval(net, batch.inputs, None, controls=controls, encoded_full=encoded)
        loss = interval_losses(scores, batch)[-1]
        gradient = torch.autograd.grad(loss, (net.release_log_scale,
            net.composition.readout[-1].weight, net.audio_input.weight))
        assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in gradient)
        assert gradient[-1].abs().sum() > 0
        results.append((loss.detach().cpu(), *(g.cpu() for g in gradient)))
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, atol=5e-4, rtol=5e-4)
