"""BeatThis network graph on log-Mel chunks; waveform processing stays outside ONNX."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn


INPUTS = ('spect',)
OUTPUTS = ('beat_logits', 'downbeat_logits')


class BeatThisGraph(nn.Module):
    """One [1, time, 128] float32 chunk to two [1, time] raw logit arrays.

    A private model copy disables the rotary cache so the exported positions
    follow the runtime chunk length. The source network and its cache are unchanged.
    """

    def __init__(self, model):
        super().__init__()
        from rotary_embedding_torch import RotaryEmbedding

        self.model = deepcopy(model).cpu().float().eval()
        for module in self.model.modules():
            if isinstance(module, RotaryEmbedding):
                module.cache_if_possible = False

    def forward(self, spect):
        outputs = self.model(spect)
        return outputs['beat'], outputs['downbeat']


def beatthis_example_inputs(frames: int = 64) -> tuple:
    return (torch.zeros(1, frames, 128),)


def beatthis_frontend_constants(frames) -> dict:
    """JSON values for the installed Audio2Frames frontend and chunk aggregation."""
    frontend = frames.spect
    mel = frontend.spect_class
    spect = mel.spectrogram
    scale = mel.mel_scale
    return {
        'sample_rate': mel.sample_rate,
        'mono': 'arithmetic_mean_of_channels',
        'resampling': {'library': 'soxr', 'quality': 'HQ'},
        'n_fft': mel.n_fft,
        'win_length': mel.win_length,
        'hop_length': mel.hop_length,
        'window': 'hann_periodic',
        'pad': spect.pad,
        'center': spect.center,
        'pad_mode': spect.pad_mode,
        'onesided': spect.onesided,
        'normalized': spect.normalized,
        'stft_scale': mel.n_fft ** -.5,
        'power': spect.power,
        'n_mels': mel.n_mels,
        'f_min': mel.f_min,
        'f_max': mel.f_max,
        'mel_scale': scale.mel_scale,
        'mel_norm': scale.norm,
        'log': 'log1p',
        'log_multiplier': frontend.log_multiplier,
        'fps': mel.sample_rate / mel.hop_length,
        'frame_count': '1 + floor(sample_count / hop_length)',
        'frame_time_seconds': 'frame_index * hop_length / sample_rate',
        'chunk_size': 1500,
        'border_size': 6,
        'avoid_short_end': True,
        'overlap_mode': 'keep_first',
        'split': {
            'first_start': -6,
            'start_step': 1488,
            'start_stop_exclusive': 'frame_count - border_size',
            'shift_last_when': 'frame_count > chunk_size - 2 * border_size',
            'shifted_last_start': 'frame_count - (chunk_size - border_size)',
            'source_start': 'max(start, 0)',
            'source_stop_exclusive': 'min(start + chunk_size, frame_count)',
            'left_zero_frames': 'max(0, -start)',
            'right_zero_frames': 'max(0, min(border_size, start + chunk_size - frame_count))',
        },
        'aggregate': {
            'initial_logit': -1000.0,
            'prediction_crop_left': 6,
            'prediction_crop_right': 6,
            'write_order': 'reverse_chunk_order',
            'destination_start': 'start + border_size',
            'destination_stop_exclusive': 'min(frame_count, start + chunk_size - border_size)',
            'overlap_operation': 'overwrite',
        },
    }


def rows_frontend_constants() -> dict:
    """The rows waveform-to-model-input scale, including float16 feature storage."""
    from ..features.mel_base import LOG_MEL_FLOOR, MUSIC_MEL_CACHE_CONFIG

    cfg = MUSIC_MEL_CACHE_CONFIG
    return {
        'sample_rate': cfg.sample_rate,
        'n_fft': cfg.n_fft,
        'win_length': cfg.win_length,
        'hop_length': cfg.hop_length,
        'window': 'hann_periodic',
        'center': False,
        'onesided': True,
        'stft_scale': 1.0,
        'stft_magnitude': 'sqrt(real ** 2 + imag ** 2)',
        'power': 2.0,
        'n_mels': cfg.mel_bins,
        'f_min': cfg.fmin,
        'f_max': cfg.fmax,
        'mel_scale': 'slaney',
        'mel_norm': 'slaney_area',
        'log': 'natural_log',
        'log_floor': LOG_MEL_FLOOR,
        'frame_count': 'ceil(sample_count / hop_length)',
        'right_zero_samples': 'max(0, (frame_count - 1) * hop_length + n_fft - sample_count)',
        'empty_waveform_frames': 0,
        'frame_read_index': 'floor(time_ms / 10)',
        'frame_center_ms': cfg.n_fft / cfg.sample_rate * 500.0,
        'standardization': {
            'axes': ['frame', 'mel_bin'],
            'std_ddof': 0,
            'epsilon': 1e-6,
            'formula': '(log_mel - mean) / (std + epsilon)',
        },
        'feature_storage_dtype': 'float16',
        'model_input_dtype': 'float32',
    }


def write_frontend_arrays(frames, out) -> dict:
    """Write exact frontend buffers as row-major little-endian float32 and the MIT notice.

    Returned paths are relative to ``out``. Mel filter axes retain each
    library's order: BeatThis [fft_bin, mel_bin], rows [mel_bin, fft_bin].
    """
    from importlib.metadata import distribution
    import shutil

    from ..features.mel_base import MUSIC_MEL_CACHE_CONFIG, _mel_layer

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    beatthis = frames.spect.spect_class
    rows = _mel_layer(MUSIC_MEL_CACHE_CONFIG)

    def write(name, tensor, axes):
        array = tensor.detach().cpu().numpy().astype('<f4', copy=False)
        filename = f'{name}.bin'
        array.tofile(out / filename)
        return dict(path=filename, dtype='float32', byte_order='little', shape=list(array.shape),
                    axes=axes, byte_length=array.nbytes)

    arrays = {
        'beatthis': {
            'mel_filter': write('beatthis-mel-filter', beatthis.mel_scale.fb, ['fft_bin', 'mel_bin']),
            'hann_window': write('beatthis-hann-window', beatthis.spectrogram.window, ['sample']),
        },
        'rows': {
            'mel_filter': write('rows-mel-filter', rows.mel_basis, ['mel_bin', 'fft_bin']),
            'hann_window': write('rows-hann-window', rows.stft.window_mask.reshape(-1), ['sample']),
            'frontend': rows_frontend_constants(),
        },
    }
    package = distribution('beat-this')
    license_path = next(p for p in package.files if p.name == 'LICENSE')
    shutil.copyfile(package.locate_file(license_path), out / 'BeatThis-LICENSE.txt')
    arrays['beatthis_license'] = 'BeatThis-LICENSE.txt'
    return arrays


def export_beatthis_graph(model, path, opset: int = 17) -> Path:
    """Write a float32 network with dynamic time and check the serialized ONNX."""
    import onnx

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = BeatThisGraph(model)
    with torch.no_grad():
        torch.onnx.export(wrapper, beatthis_example_inputs(), str(path), input_names=list(INPUTS),
                          output_names=list(OUTPUTS), dynamic_axes={n: {1: 'time'} for n in INPUTS + OUTPUTS},
                          opset_version=opset, dynamo=False)
    onnx.checker.check_model(str(path))
    return path
