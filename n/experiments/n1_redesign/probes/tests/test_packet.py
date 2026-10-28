import sys
from pathlib import Path

import numpy as np

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
