"""osu!-clock lead rule and log-Mel frame convention (numpy parts; the encoded click tracks run on the mac)."""
from __future__ import annotations

import numpy as np
import pytest

from ensomi_n.features import clock
from ensomi_n.features.fixtures import click_track, locate
from ensomi_n.features.mel import DEFAULT, filterbank, frame_count, log_mel


def _frame_header(mono: bool = False) -> bytes:
    # MPEG-1 Layer III, 128 kbps, 44.1 kHz, no padding.
    return bytes([0xFF, 0xFB, 0x90, 0xC0 if mono else 0x00])


def _mp3(info: bytes | None, ext: bytes | None = None, id3: bool = False) -> bytes:
    size = 144 * 128000 // 44100
    head = _frame_header()
    first = bytearray(size)
    first[:4] = head
    if info is not None:
        x = 4 + 32
        first[x:x + 4] = info
        first[x + 4:x + 8] = (1).to_bytes(4, 'big')        # frame count only
        if ext is not None:
            first[x + 12:x + 12 + len(ext)] = ext
    audio = bytearray(size)
    audio[:4] = head
    body = bytes(first) + bytes(audio) * 3
    if id3:
        tag = b'ID3\x03\x00\x00' + bytes([0, 0, 0, 20]) + bytes(20)
        body = tag + body
    return body


def _lame_ext(encoder: bytes, delay: int, padding: int = 1000) -> bytes:
    ext = bytearray(36)
    ext[:9] = encoder.ljust(9, b'\x00')
    ext[21] = delay >> 4
    ext[22] = ((delay & 0xF) << 4) | (padding >> 8)
    ext[23] = padding & 0xFF
    return bytes(ext)


@pytest.mark.parametrize('kind,payload,lead_sf,lead_ff', [
    ('lame', _mp3(b'Info', _lame_ext(b'LAME3.100', 576)), 0, 0),
    ('lame', _mp3(b'Info', _lame_ext(b'LAME3.100', 0)), 529, 529),
    ('lame', _mp3(b'Xing', _lame_ext(b'GOGO', 0)), 529, 0),
    ('xing', _mp3(b'Xing', bytes(36)), 529, 0),
    ('plain', _mp3(None), 0, 0),
])
def test_lead_rule_per_header_class(tmp_path, kind, payload, lead_sf, lead_ff):
    path = tmp_path / 'a.mp3'
    path.write_bytes(payload)
    start = clock.read_mp3_start(path)
    assert start.kind == kind
    assert start.sample_rate == 44100
    assert clock.lead_samples(start, 'libsndfile') == lead_sf
    assert clock.lead_samples(start, 'ffmpeg') == lead_ff


def test_id3_tag_is_skipped_and_delay_read(tmp_path):
    path = tmp_path / 'a.mp3'
    path.write_bytes(_mp3(b'Info', _lame_ext(b'Lavc61.3', 576), id3=True))
    start = clock.read_mp3_start(path)
    assert (start.kind, start.encoder, start.delay) == ('lame', 'Lavc61.3', 576)


def test_non_mp3_leads_nothing(tmp_path):
    path = tmp_path / 'a.ogg'
    path.write_bytes(b'OggS' + bytes(100))
    assert clock.file_lead(path, 'libsndfile')[0] == 0


def test_filterbank_shape_and_area_normalisation():
    fb = filterbank()
    assert fb.shape == (128, 481)
    assert np.all(fb >= 0) and np.all(fb.sum(axis=1) > 0)


def test_frame_count_and_centring_on_clicks():
    sr = DEFAULT.sample_rate
    times = (1000.0, 2003.0, 3507.5, 5011.0)
    wave = click_track(sr, 6.0, times)
    mel = log_mel(wave)
    assert mel.shape == (frame_count(len(wave)), 128) == (600, 128)
    found = locate(mel, times)
    assert np.max(np.abs(found - np.array(times))) <= 1.0     # frame i sits at 10 i ms


def test_lead_prepended_moves_onsets_by_its_length():
    sr = DEFAULT.sample_rate
    times = (1000.0, 2500.0)
    wave = click_track(sr, 4.0, times)
    lead = int(round(0.012 * sr))
    shifted = np.concatenate([np.zeros(lead, dtype=np.float32), wave])
    found = locate(log_mel(shifted), np.array(times) + 12.0)
    assert np.max(np.abs(found - (np.array(times) + 12.0))) <= 1.0
