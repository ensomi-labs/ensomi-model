import numpy as np
import pytest
import torch

from ensomi_model.audio_rows.net import FigureNet, NetConfig, _back, eq_features
from ensomi_model.export.rows_graph import (
    ENCODER_INPUTS, STEP_INPUTS, RowsSongEncoder, RowsStep, export_rows_graphs, rows_example_inputs,
)


def _model():
    torch.manual_seed(31)
    cfg = NetConfig(vocab=6, enc=8, dilations=(1, 2), ctx=16, ctx_layers=2,
                    ctx_heads=2, window=2, hidden=12, fig=8, heads=2, lags=3, key=4)
    members = np.zeros((cfg.vocab, 24), np.float32)
    for i in range(1, cfg.vocab):
        members[i, :i] = 1
    model = FigureNet(cfg, members).eval()
    with torch.no_grad():
        model.eq_w.normal_(0, .3)
        model.lag_bias.normal_(0, .3)
    return model


def _sequence(model, frames=59, beats=8):
    inputs = list(rows_example_inputs(model, frames, beats)['rows_encoder'])
    inputs[0] = torch.randn_like(inputs[0])
    inputs[3] = torch.randn_like(inputs[3])
    inputs[4][0, -1] = False
    return tuple(inputs)


def _step_inputs(model, encoder_inputs, ids):
    pos, context, query, key = model.audio(*encoder_inputs)
    extra = encoder_inputs[3]
    prev = torch.cat((torch.full_like(ids[:, :1], model.cfg.vocab), ids[:, :-1]), 1)
    h, _ = model.gru(torch.cat((context, model.input_table()[prev], extra), -1))
    back_keys = _back(key, model.cfg.lags, 0.0)
    eq = eq_features(ids, model.cfg.lags)
    back_ids = _back(ids, model.cfg.lags, -1)
    back_ok = back_ids > 0
    return pos, context, query, extra, prev, h, back_keys, eq, back_ok


@torch.no_grad()
def test_rows_step_matches_full_recurrent_forward():
    model = _model()
    encoder_inputs = _sequence(model)
    ids = torch.tensor([[2, 0, 2, 2, 4, 0, 4, 1]])
    pos, context, query, extra, prev, full_h, back_keys, eq, back_ok = _step_inputs(model, encoder_inputs, ids)
    step = RowsStep(model)
    hidden = torch.zeros(1, 1, model.cfg.hidden)
    temperature = torch.tensor(.8)
    for b in range(ids.shape[1]):
        outputs = step(context[:, b], pos[:, b], query[:, b], extra[:, b], prev[:, b], hidden,
                       back_keys[:, b], eq[:, b], back_ok[:, b], temperature)
        hidden, hb, logits, scores, gate, logp, attention = outputs
        expected_h = full_h[:, b]
        expected_logits = model.generator_logits(expected_h, pos[:, b])
        expected_scores = model.copy_scores(query[:, b], back_keys[:, b], eq[:, b])
        expected_mixture = model.mixture(expected_h, expected_logits / temperature, expected_scores, back_ok[:, b])
        for actual, expected in zip((hb, logits, scores, gate, logp, attention),
                                    (expected_h, expected_logits, expected_scores, *expected_mixture)):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(hidden[0], expected_h, atol=1e-6, rtol=1e-5)
        assert torch.isfinite(attention).all()
        assert torch.count_nonzero(attention[..., :-1].masked_select(~back_ok[:, b, None])) == 0


@torch.no_grad()
def test_rows_onnx_dynamic_song_and_carried_state(tmp_path):
    pytest.importorskip('onnx')
    ort = pytest.importorskip('onnxruntime')
    model = _model()
    paths = export_rows_graphs(model, tmp_path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    encoder = ort.InferenceSession(str(paths['rows_encoder']), options, providers=['CPUExecutionProvider'])
    step = ort.InferenceSession(str(paths['rows_step']), options, providers=['CPUExecutionProvider'])
    for frames, beats in ((59, 8), (91, 5)):
        inputs = _sequence(model, frames, beats)
        expected = model.audio(*inputs)
        for wrapped, source in zip(RowsSongEncoder(model)(*inputs), expected):
            torch.testing.assert_close(wrapped, source)
        actual = encoder.run(None, {n: t.numpy() for n, t in zip(ENCODER_INPUTS, inputs)})
        for a, e in zip(actual, expected):
            np.testing.assert_allclose(a, e.numpy(), atol=2e-6, rtol=2e-5)

    inputs = _sequence(model)
    ids = torch.tensor([[2, 0, 2, 2, 4, 0, 4, 1]])
    pos, context, query, extra, prev, full_h, back_keys, eq, back_ok = _step_inputs(model, inputs, ids)
    hidden = np.zeros((1, 1, model.cfg.hidden), np.float32)
    for b in range(ids.shape[1]):
        args = (context[:, b], pos[:, b], query[:, b], extra[:, b], prev[:, b], torch.from_numpy(hidden),
                back_keys[:, b], eq[:, b], back_ok[:, b], torch.tensor(.8))
        expected = RowsStep(model)(*args)
        actual = step.run(None, {n: t.numpy() for n, t in zip(STEP_INPUTS, args)})
        for a, e in zip(actual, expected):
            np.testing.assert_allclose(a, e.numpy(), atol=2e-6, rtol=2e-5)
        hidden = actual[0]
        np.testing.assert_allclose(actual[1], full_h[:, b].numpy(), atol=2e-6, rtol=2e-5)
