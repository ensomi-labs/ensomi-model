from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from ensomi_model.research.controlled_audio_continuation.generation import ControlledSession
from ensomi_model.research.controlled_audio_continuation.model import ControlledAudioModel, load_model
from ensomi_model.research.controlled_audio_continuation.scope_allocation import (
    FEATURES, LnScopeState, ScopedLnAllocation, features_before_rows,
)
from ensomi_model.research.controlled_audio_continuation.sampling import replay_row_scores
from ensomi_model.research.joint_audio_continuation.intervals import IntervalExample
from ensomi_model.research.oracle_time_continuation.replay import ExactReplayState, commit
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
from ensomi_model.research.planned_audio_continuation.intervals import collate_interval, score_interval
from ensomi_model.research.typed_audio_continuation.controls import ControlSchedule, ControlSpan
from planned_audio_continuation.test_distribution import chart, config


def setup(mode):
    return ControlledAudioModel(replace(config(), lookahead=16, bounded_head=True,
        condition_full_holds=True, minimum_action_gap_ms=60), style_names=('tech',),
        scope_allocation=mode).eval()


def test_request_identity_keeps_declared_and_effective_counts_across_overrides():
    schedule = ControlSchedule((ControlSpan(0, 1000, ln_fraction=.5),))
    state = LnScopeState.empty(schedule).observe(CompleteRow(100, (2, 1, 0, 0)))
    newer = ControlSchedule((*schedule.spans, ControlSpan(200, 400, ln_fraction=.5),
                             ControlSpan(150, 800, style={'tech': 1})), ('tech',))
    state = state.update_controls(newer)
    assert state.program.owner(300) == 1 and state.program.owner(400) == 0
    child = state.observe(CompleteRow(300, (3, 1, 1, 0)))
    assert state.declared == ((2, 1), (0, 0))
    assert child.declared == ((4, 1), (2, 0))
    assert child.owned == ((2, 1), (2, 0))
    assert child.features(400)[2] == .25 and child.features(400)[6] == .5
    assert child.features(400)[8] == pytest.approx(.25)
    assert child.features(500)[8] > child.features(400)[8]
    assert state.observe(CompleteRow(150, (3, 0, 0, 0))) is state
    assert not child.features(1000).any()


def test_source_features_exclude_current_target_and_future_and_keep_query_order():
    controls = ControlSchedule((ControlSpan(0, 1000, ln_fraction=.5),))
    rows = [CompleteRow(100, (2, 0, 0, 0)), CompleteRow(300, (0, 1, 0, 0)),
            CompleteRow(800, (3, 0, 0, 0))]
    queries = [301, 100, 300, 300, 800]
    features = features_before_rows(rows, controls, queries)
    np.testing.assert_array_equal(features, features_before_rows(rows[:2], controls, queries))
    assert features[0, 2] == .5 and features[1, 3] == 0
    assert features[2, 2] == features[3, 2] == 1


def test_old_replay_and_finite_history_can_agree_while_scope_allocation_differs():
    # Both traces are legal and have identical total head/row counts. Common
    # release/attack clocks overwrite old clocks; 512 common rows exceed R1's
    # 511-token receptive field. The earlier LN-start totals remain different.
    a, b = [], []
    for i in range(64):
        a.extend((CompleteRow(400*i, (2, 0, 0, 0)), CompleteRow(400*i+100, (3, 1, 0, 0))))
        b.extend((CompleteRow(400*i, (1, 0, 0, 0)), CompleteRow(400*i+100, (0, 1, 0, 0))))
    common = [CompleteRow(25600, (2, 0, 0, 0)), CompleteRow(25700, (3, 0, 1, 0))]
    common += [CompleteRow(26000+200*i, tuple(int(k == i % 4) for k in range(4)))
               for i in range(512)]
    controls = ControlSchedule((ControlSpan(0, 200000, ln_fraction=.3),))
    states, accounting = [], []
    for prefix in (a, b):
        exact, scope = ExactReplayState(), LnScopeState.empty(controls)
        for row in (*prefix, *common):
            exact, scope = commit(exact, row), scope.observe(row)
        states.append(exact); accounting.append(scope)
    assert states[0] == states[1]
    assert accounting[0].declared[0][0] == accounting[1].declared[0][0]
    assert accounting[0].declared[0][1]-accounting[1].declared[0][1] == 64
    assert not np.array_equal(accounting[0].features(130000), accounting[1].features(130000))


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps',
    marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS unavailable'))])
def test_zero_init_unknown_identity_and_neural_family_invariance_with_gradients(device):
    torch.manual_seed(781)
    layer = ScopedLnAllocation(3, 4, 2, 6, 4, 'progress').to(device)
    raw = torch.randn(3, 256, device=device)
    raw[:, :10] = -torch.inf
    raw = raw.log_softmax(-1)
    inputs = [torch.randn(3, n, device=device) for n in (3, 4, 2, 6, len(FEATURES))]
    inputs[3][:, 4] = torch.tensor([1., 0., 1.], device=device)
    assert torch.equal(layer(raw, *inputs), raw)
    with torch.no_grad():
        layer.readout[-1].weight.normal_(std=.2)
    changed = layer(raw, *inputs)
    assert torch.equal(changed[1], raw[1])
    torch.testing.assert_close(changed.logsumexp(-1), torch.zeros(3, device=device), atol=2e-6, rtol=0)
    for group in torch.unique(layer.groups):
        mask = layer.groups == group
        torch.testing.assert_close(changed[:, mask].logsumexp(-1), raw[:, mask].logsumexp(-1),
                                   atol=2e-6, rtol=2e-6)
        for longs in range(5):
            selected = mask & (layer.longs == longs) & torch.isfinite(raw[0])
            if selected.any():
                shift = changed[0, selected]-raw[0, selected]
                torch.testing.assert_close(shift, shift[0].expand_as(shift), atol=2e-6, rtol=2e-6)
    (-changed[:, 40].sum()).backward()
    assert torch.isfinite(layer.readout[0].weight.grad).all()
    assert layer.readout[0].weight.grad[:, -len(FEATURES):].abs().sum() > 0
    layer.mode = 'context'
    other = [*inputs[:-1], inputs[-1]+20]
    assert torch.equal(layer(raw, *inputs), layer(raw, *other))


def test_checkpoint_modes_and_unknown_ln_native_identity(tmp_path):
    torch.set_num_threads(1); torch.manual_seed(795)
    base = setup('none')
    assert 'scope_allocation' not in base.probability_options()
    mel = np.random.default_rng(81).normal(size=(100, 128)).astype(np.float32)
    controls = ControlSchedule((ControlSpan(0, 1001, stars=4, style={'tech': 1}),), base.style_names)
    with torch.inference_mode():
        direct = ControlledSession(base, mel, 1000, controls, seed=57, ln_feedback=None)
        direct.publish_to(1000)
    for mode in ('context', 'progress'):
        model = setup(mode); missing = model.load_state_dict(base.state_dict(), strict=False)
        assert all(k.startswith('scope_allocation.') for k in missing.missing_keys)
        with torch.no_grad():
            model.scope_allocation.readout[-1].weight.normal_(std=.5)
        path = tmp_path/(mode+'.pt')
        torch.save(dict(format='controlled-audio/v1', model_config=asdict(model.config),
            probability_options=model.probability_options(), model=model.state_dict()), path)
        model = load_model(path)
        assert model.scope_allocation_mode == mode
        with torch.inference_mode():
            generated = ControlledSession(model, mel, 1000, controls, seed=57, ln_feedback=None)
            generated.publish_to(1000)
        assert generated.rows == direct.rows


def test_native_and_factual_probability_reconstruction_and_private_state():
    torch.set_num_threads(1); torch.manual_seed(799)
    model = setup('progress')
    with torch.no_grad():
        model.scope_allocation.readout[-1].weight.normal_(std=.2)
    controls = ControlSchedule((ControlSpan(0, 1001, stars=3, ln_fraction=.8),
        ControlSpan(350, 650, ln_fraction=.2), ControlSpan(500, 800, style={'tech': 1})), model.style_names)
    mel = np.random.default_rng(81).normal(size=(100, 128)).astype(np.float32)
    recorded = []
    class Recorder(ControlledSession):
        def prefer_rows(self, *args):
            q = super().prefer_rows(*args); recorded.append(q.clone()); return q
    with torch.inference_mode():
        session = Recorder(model, mel, 1000, controls, seed=81, ln_feedback=None,
                           head_times=(0, 100, 200, 300, 450, 550, 650, 750, 900))
        session.publish_to(333)
        original = session.scope_state
        private = session.fork()
        private.publish_to(600)
        assert session.scope_state is original and private.scope_state is not original
        # The recorder belongs to this test, not the branch-isolated session.
        recorded[:] = recorded[:len(session.rows)]
        session.publish_to(1000)
    c = replace(chart([r.time_ms for r in session.rows], [r.actions for r in session.rows], 1000), mel=mel)
    encoded = model.encode_audio(torch.from_numpy(mel)[None])
    scores = []
    for i in range(IntervalExample(c, 0, 137).count):
        example = IntervalExample(c, i, 137)
        batch = collate_interval(example, model.config, recovery=model.recovery)
        def observations(kind, times, indices):
            if kind != 'row':
                return {}
            return dict(allocation_features=torch.tensor(features_before_rows(
                session.rows, controls, times.numpy())))
        raw = score_interval(model, batch.inputs, None, controls=controls,
                             encoded_full=encoded, history_options=observations)
        scores.append(replay_row_scores(raw.row[:len(batch.row_index)], example, controls, ln_feedback=None))
    replayed = torch.cat(scores)
    torch.testing.assert_close(replayed, torch.stack(recorded), atol=3e-5, rtol=3e-6)


def test_live_announcement_extends_accounting_without_rewriting_committed_prefix():
    torch.set_num_threads(1); torch.manual_seed(807)
    model = setup('progress')
    controls = ControlSchedule((ControlSpan(0, 1501, stars=4, ln_fraction=.4),), model.style_names)
    with torch.inference_mode():
        session = ControlledSession(model, np.zeros((150, 128), np.float32), 1500,
                                    controls, seed=123, ln_feedback=None)
        session.publish_to(400)
        prefix, counts, exact = tuple(session.rows), session.scope_state.declared, session.replay
        assert counts[0][0] > 0
        session.update_controls(ControlSpan(500, 900, ln_fraction=.7))
        assert session.scope_state.declared == (*counts, (0, 0))
        assert tuple(session.rows) == prefix and session.replay == exact
        session.publish_to(1000)
        assert tuple(session.rows[:len(prefix)]) == prefix
        assert session.scope_state.program.owner(1000) == 0
        assert session.scope_state.declared[0][0] > counts[0][0]
        assert session.scope_state.declared[1][0] > 0
