"""Float32 rows graphs; the caller owns figure history, masks and sampling."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from ..audio_rows.beatdata import EXTRA_DIM
from ..audio_rows.figures import NPOS
from ..audio_rows.net import FigureNet


ENCODER_INPUTS = ('mel', 'pos_frames', 'beat_frames', 'extra', 'valid')
ENCODER_OUTPUTS = ('pos', 'context', 'query', 'key')
STEP_INPUTS = ('context', 'pos', 'query', 'extra', 'prev', 'hidden', 'back_keys', 'eq', 'back_ok', 'temperature')
STEP_OUTPUTS = ('hidden_out', 'hb', 'logits', 'scores', 'gate', 'logp', 'attention')


class _LocalAttention(nn.Module):
    """The rows attention with an equality mask instead of ONNX EyeLike(bool)."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x, valid):
        layer = self.layer
        B, L, d = x.shape
        qkv = layer.qkv(layer.norm1(x)).view(B, L, 3, layer.heads, d // layer.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        index = torch.arange(L, device=x.device)
        lag = index[None] - index[:, None]
        scores = q @ k.transpose(-1, -2) / (d // layer.heads) ** .5
        scores = scores + layer.bias[:, lag.clamp(-layer.window, layer.window) + layer.window]
        allowed = ((lag.abs() <= layer.window)[None, None] & valid[:, None, None, :]) | (lag == 0)
        attention = torch.softmax(scores.masked_fill(~allowed, -torch.inf), -1)
        x = x + layer.out((attention @ v).transpose(1, 2).reshape(B, L, d))
        return x + layer.ff(layer.norm2(x))


class RowsSongEncoder(nn.Module):
    """Encode one song: frame and beat counts are dynamic; batch size is one."""

    def __init__(self, model: FigureNet):
        super().__init__()
        self.model = deepcopy(model)
        self.model.context = nn.ModuleList(_LocalAttention(layer) for layer in self.model.context)

    def forward(self, mel, pos_frames, beat_frames, extra, valid):
        return self.model.audio(mel, pos_frames, beat_frames, extra, valid)


class RowsStep(nn.Module):
    """One beat with explicit GRU state and copy history, all newest lag first.

    ``hidden`` is [1, 1, hidden], initially zero; ``prev`` is [1] int64,
    initially the vocabulary size. ``back_keys`` is [1, lags, heads, key],
    ``eq`` is [1, lags, 2] float32 and ``back_ok`` is [1, lags] bool.
    The caller applies the quiet-rest policy in ``back_ok`` and supplies
    ``temperature`` as a scalar float32. Returned logits precede temperature;
    ``logp`` includes it. The final attention slot is the generator fallback.
    """

    def __init__(self, model: FigureNet):
        super().__init__()
        self.model = model

    def forward(self, context, pos, query, extra, prev, hidden, back_keys, eq, back_ok, temperature):
        x = torch.cat((context, self.model.input_table()[prev], extra), -1)
        out, hidden_out = self.model.gru(x[:, None], hidden)
        hb = out[:, 0]
        logits = self.model.generator_logits(hb, pos)
        scores = self.model.copy_scores(query, back_keys, eq)
        gate, logp, attention = self.model.mixture(hb, logits / temperature, scores, back_ok)
        return hidden_out, hb, logits, scores, gate, logp, attention


def rows_example_inputs(model: FigureNet, frames: int = 128, beats: int = 4) -> dict[str, tuple]:
    """CPU inputs for both graphs, including the first beat's empty history."""
    cfg = model.cfg
    edges = torch.linspace(0, frames, beats + 1).long()
    pos_frames = torch.linspace(0, frames - 1, beats * NPOS).long().reshape(1, beats, NPOS)
    encoder = (
        torch.zeros(1, frames, 128), pos_frames,
        torch.stack((edges[:-1], edges[1:]), -1)[None],
        torch.zeros(1, beats, EXTRA_DIM), torch.ones(1, beats, dtype=torch.bool),
    )
    step = (
        torch.zeros(1, cfg.ctx), torch.zeros(1, NPOS, cfg.enc), torch.zeros(1, cfg.heads, cfg.key),
        torch.zeros(1, EXTRA_DIM), torch.full((1,), cfg.vocab, dtype=torch.int64),
        torch.zeros(1, 1, cfg.hidden), torch.zeros(1, cfg.lags, cfg.heads, cfg.key),
        torch.zeros(1, cfg.lags, 2), torch.zeros(1, cfg.lags, dtype=torch.bool), torch.tensor(1.0),
    )
    return {'rows_encoder': encoder, 'rows_step': step}


def export_rows_graphs(model: FigureNet, output_dir, opset: int = 17) -> dict[str, Path]:
    """Write and check rows_encoder.onnx and rows_step.onnx from a CPU float32 model."""
    import onnx

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = rows_example_inputs(model)
    graphs = {}
    for name, wrapper, inputs, outputs, dynamic in (
        ('rows_encoder', RowsSongEncoder(model), ENCODER_INPUTS, ENCODER_OUTPUTS,
         {'mel': {1: 'frames'}, **{n: {1: 'beats'} for n in ENCODER_INPUTS[1:] + ENCODER_OUTPUTS}}),
        ('rows_step', RowsStep(model), STEP_INPUTS, STEP_OUTPUTS, None),
    ):
        path = output_dir / f'{name}.onnx'
        with torch.no_grad():
            torch.onnx.export(wrapper.eval(), examples[name], str(path), input_names=list(inputs),
                              output_names=list(outputs), dynamic_axes=dynamic, opset_version=opset,
                              dynamo=False)
        onnx.checker.check_model(str(path))
        graphs[name] = path
    return graphs
