import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from n1_redesign.probes.packet import render


def test_native_rate_two_plays_and_click_rms():
    sr = 44100
    source = (.1 * np.sin(2 * np.pi * 220 * np.arange(sr * 3) / sr)).astype('float32')
    output, meta = render(source, sr, 500, 2500, [1000])
    size, gap = sr * 2, round(.5 * sr)
    np.testing.assert_array_equal(output[:size], source[round(.5 * sr):round(2.5 * sr)])
    np.testing.assert_array_equal(output[size:size + gap], np.zeros(gap))
    diff = output[size + gap:] - output[:size]
    start, end = round(.5 * sr), round(.5 * sr) + meta['click_samples']
    assert np.count_nonzero(diff[:start]) == 0 and np.count_nonzero(diff[end:]) == 0
    click_rms = np.sqrt(np.mean(diff[start:end].astype(float) ** 2))
    assert abs(20 * np.log10(click_rms / meta['source_rms']) + 12) < .001


def test_packet_validation_rejects_changed_source_audio(tmp_path, monkeypatch):
    import soundfile as sf
    from n1_redesign.probes import packet
    monkeypatch.setattr(packet, 'OUT', tmp_path)
    directory = tmp_path / 'packet'; directory.mkdir()
    (directory / 'packet.md').write_text('Instructions')
    (directory / 'answers.csv').write_text('id,answer,note\n01,,\n')
    sr = 8000
    source = (.1 * np.sin(np.arange(sr * 2) * .1)).astype('float32')
    wave, meta = render(source, sr, 0, 1000, [500])
    key = dict(id='01', native_rate=sr, lo_ms=0, click_times_ms=[500], **meta)
    path = directory / '01.wav'
    sf.write(path, wave, sr, subtype='PCM_16')
    packet.validate_packet(directory, [key])
    raw, sr = sf.read(path, dtype='int16')
    raw[-10] += 1
    sf.write(path, raw, sr, subtype='PCM_16')
    with pytest.raises(AssertionError):
        packet.validate_packet(directory, [key])
