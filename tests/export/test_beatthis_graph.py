import numpy as np
import pytest
import torch

from ensomi_model.export.beatthis_graph import (
    BeatThisGraph, beatthis_frontend_constants, export_beatthis_graph, write_frontend_arrays,
)


@torch.no_grad()
def test_beatthis_dynamic_time_matches_network(tmp_path):
    pytest.importorskip('onnx')
    ort = pytest.importorskip('onnxruntime')
    beat_this = pytest.importorskip('beat_this.model.beat_tracker')
    torch.manual_seed(17)
    model = beat_this.BeatThis(transformer_dim=32, n_layers=2, head_dim=8, stem_dim=8).eval()
    path = export_beatthis_graph(model, tmp_path / 'beatthis.onnx')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    wrapper = BeatThisGraph(model)
    for frames in (17, 41, 81):
        spect = torch.randn(1, frames, 128)
        expected = model(spect)
        actual = session.run(None, {'spect': spect.numpy()})
        for name, traced, exported in zip(('beat', 'downbeat'), wrapper(spect), actual):
            torch.testing.assert_close(traced, expected[name])
            np.testing.assert_allclose(exported, expected[name].numpy(), atol=1e-5, rtol=2e-4)
    rotary = model.transformer_blocks.layers[0][0].rotary_embed
    assert rotary.cache_if_possible


def test_frontend_constants_match_installed_frontend():
    from types import SimpleNamespace

    preprocessing = pytest.importorskip('beat_this.preprocessing')
    frontend = preprocessing.LogMelSpect()
    spec = beatthis_frontend_constants(SimpleNamespace(spect=frontend))
    assert spec['sample_rate'] == 22050
    assert spec['n_fft'] == spec['win_length'] == 1024
    assert spec['hop_length'] == 441
    assert spec['fps'] == 50
    assert (spec['f_min'], spec['f_max'], spec['n_mels']) == (30, 11000, 128)
    assert spec['power'] == 1
    assert spec['normalized'] == 'frame_length'
    assert spec['stft_scale'] == 1 / 32
    assert spec['mel_scale'] == 'slaney' and spec['mel_norm'] is None
    assert spec['center'] and spec['pad_mode'] == 'reflect' and spec['onesided']
    torch.testing.assert_close(frontend.spect_class.spectrogram.window, torch.hann_window(spec['win_length']))


@pytest.mark.parametrize('frames,starts', [(100, [-6]), (1488, [-6]), (1489, [-6, -5]),
                                         (1500, [-6, 6]), (3000, [-6, 1482, 1506])])
def test_chunk_edges_and_overlap_match_installed_beatthis(frames, starts):
    inference = pytest.importorskip('beat_this.inference')
    spect = torch.arange(frames, dtype=torch.float32)[:, None].repeat(1, 128)
    chunks, actual_starts = inference.split_piece(spect, 1500, border_size=6, avoid_short_end=True)
    assert list(actual_starts) == starts
    expected = torch.full((frames,), -1000.0)
    predictions = []
    for i, (start, chunk) in enumerate(zip(starts, chunks)):
        left = max(0, -start)
        right = max(0, min(6, start + 1500 - frames))
        expected_chunk = torch.nn.functional.pad(spect[max(start, 0):min(start + 1500, frames)], (0, 0, left, right))
        torch.testing.assert_close(chunk, expected_chunk)
        values = torch.full((len(chunk),), float(i))
        predictions.append({'beat': values, 'downbeat': -values})
    for start, prediction in reversed(list(zip(starts, predictions))):
        expected[start + 6:min(frames, start + 1494)] = prediction['beat'][6:-6]
    beat, downbeat = inference.aggregate_prediction(predictions, starts, frames, 1500, 6, 'keep_first', 'cpu')
    torch.testing.assert_close(beat, expected)
    torch.testing.assert_close(downbeat, -expected)


def test_frontend_arrays_preserve_installed_coefficients_and_notice(tmp_path):
    from importlib.metadata import distribution
    from types import SimpleNamespace

    preprocessing = pytest.importorskip('beat_this.preprocessing')
    from ensomi_model.features.mel_base import MUSIC_MEL_CACHE_CONFIG, _mel_layer

    beatthis = preprocessing.LogMelSpect()
    rows = _mel_layer(MUSIC_MEL_CACHE_CONFIG)
    manifest = write_frontend_arrays(SimpleNamespace(spect=beatthis), tmp_path)
    for name, key, tensor in (
        ('beatthis', 'mel_filter', beatthis.spect_class.mel_scale.fb),
        ('beatthis', 'hann_window', beatthis.spect_class.spectrogram.window),
        ('rows', 'mel_filter', rows.mel_basis),
        ('rows', 'hann_window', rows.stft.window_mask.reshape(-1)),
    ):
        entry = manifest[name][key]
        path = tmp_path / entry['path']
        assert entry['byte_order'] == 'little' and entry['dtype'] == 'float32'
        assert path.stat().st_size == entry['byte_length']
        stored = np.fromfile(path, dtype='<f4').reshape(entry['shape'])
        np.testing.assert_array_equal(stored, tensor.numpy())
    assert manifest['beatthis']['mel_filter']['axes'] == ['fft_bin', 'mel_bin']
    assert manifest['rows']['mel_filter']['axes'] == ['mel_bin', 'fft_bin']
    cfg = manifest['rows']['frontend']
    assert (cfg['sample_rate'], cfg['n_fft'], cfg['hop_length']) == (24000, 960, 240)
    assert cfg['stft_scale'] == 1 and cfg['power'] == 2 and cfg['mel_norm'] == 'slaney_area'
    assert cfg['log_floor'] == 1e-5 and cfg['standardization']['std_ddof'] == 0
    assert cfg['feature_storage_dtype'] == 'float16'
    package = distribution('beat-this')
    original = package.locate_file(next(p for p in package.files if p.name == 'LICENSE'))
    assert (tmp_path / manifest['beatthis_license']).read_bytes() == original.read_bytes()
