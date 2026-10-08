"""Wiring from a training checkpoint to an exported chart, with BeatThis replaced by a fixed song."""
import json
from types import SimpleNamespace

import numpy as np
import torch

from ensomi_model.audio_rows.audio import FEATURE_DIM, Song
from ensomi_model.audio_rows.generate import Generator
from ensomi_model.audio_rows.model import load_model
from ensomi_model.audio_rows.train import TrainConfig, train
from ensomi_model.r2.model import R2Config, R2Model

from .helpers import BANDS, tiny_build


def test_a_trained_checkpoint_generates_an_osu_chart_beside_its_audio(tmp_path):
    torch.manual_seed(0)
    summary = train(TrainConfig(data=str(tiny_build(tmp_path / 'build')), out=str(tmp_path / 'run'), grid='fitted',
                                steps=2, eval_every=2, eval_windows=2, window=16, threads=1))
    assert summary['steps_done'] == 2 and summary['killed'] is None
    head, record = load_model(tmp_path / 'run' / 'model.pt')
    song = Song(np.random.default_rng(1).normal(size=(1_000, FEATURE_DIM)).astype(np.float16),
                np.array([[130.0, 500.0]]), 10_000.0)
    generator = Generator(head, record, R2Model(R2Config()).eval(), SimpleNamespace(listen=lambda path: song))
    audio = tmp_path / 'song.OGG'
    audio.write_bytes(b'not decoded here')
    result = generator.generate(generator.prepare(audio), tmp_path / 'out', band=4, seed=1)

    text = (tmp_path / 'out' / 'band-4-seed-1.osu').read_text()
    assert 'AudioFilename:audio.ogg' in text and (tmp_path / 'out' / 'audio.ogg').exists()
    starts = {float(line.split(',')[2]) for line in text.split('[HitObjects]\n')[1].splitlines() if line}
    assert len(starts) == result['K'] > 0
    assert np.allclose(result['requested_log_wh_per_s'], BANDS[4])
    assert json.loads((tmp_path / 'out' / 'band-4-seed-1.json').read_text())['segments'][0][0] == 100.0
