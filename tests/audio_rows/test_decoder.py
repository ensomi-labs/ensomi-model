"""MP3 header classes and the lead each decoder needs to sit on osu!'s clock, on synthetic frames."""
import numpy as np

from ensomi_model.audio_rows.decoder import LEAD_SAMPLES, lead_samples, lead_seconds, mp3_header

FRAME = 417                              # MPEG-1 layer III, 128 kbps, 44.1 kHz, stereo, no padding


def _frames(xing: bytes = b'', count: int = 3) -> bytes:
    body = bytearray(bytes([0xFF, 0xFB, 0x90, 0x00]) + bytes(32) + xing)
    body += bytes(FRAME - len(body))
    return bytes(body) + bytes([0xFF, 0xFB, 0x90, 0x00]) + bytes(FRAME - 4) * (count - 1)


def _info(encoder: bytes = b'LAME3.100', delay: int = 576) -> bytes:
    ext = bytearray(36)
    ext[:9] = encoder.ljust(9, b'\x00')
    ext[21] = delay >> 4
    ext[22] = (delay & 0xF) << 4
    return b'Info' + (15).to_bytes(4, 'big') + (1000).to_bytes(4, 'big') + (417000).to_bytes(4, 'big') + bytes(100) + (0).to_bytes(4, 'big') + bytes(ext)


def _write(tmp_path, name, data, id3=False):
    p = tmp_path / name
    if id3:
        data = b'ID3\x04\x00\x00' + bytes([0, 0, 0, 20]) + bytes(20) + data
    p.write_bytes(data)
    return p


def test_lame_tag_with_a_delay_needs_no_lead_on_either_decoder(tmp_path):
    p = _write(tmp_path, 'a.mp3', _frames(_info()), id3=True)
    h = mp3_header(p)
    assert (h.tag, h.encoder, h.delay, h.sample_rate) == ('lame', 'LAME3.100', 576, 44100)
    assert lead_samples(h, 'mpg123') == 0 and lead_samples(h, 'ffmpeg') == 0


def test_xing_without_a_lame_extension_is_early_on_mpg123_only(tmp_path):
    p = _write(tmp_path, 'x.mp3', _frames(_info(encoder=b'\x00' * 9)))
    h = mp3_header(p)
    assert h.tag == 'xing'
    assert lead_samples(h, 'mpg123') == LEAD_SAMPLES and lead_samples(h, 'ffmpeg') == 0
    assert np.isclose(lead_seconds(p, 'mpg123'), 529 / 44100) and lead_seconds(p, 'ffmpeg') == 0.0


def test_lame_tag_with_a_zero_delay_is_early_on_both_decoders(tmp_path):
    p = _write(tmp_path, 'z.mp3', _frames(_info(encoder=b'Lavf58.29', delay=0)))
    h = mp3_header(p)
    assert h.tag == 'lame' and h.delay == 0
    assert lead_samples(h, 'mpg123') == LEAD_SAMPLES and lead_samples(h, 'ffmpeg') == LEAD_SAMPLES


def test_no_header_and_non_mp3_files_need_no_lead(tmp_path):
    p = _write(tmp_path, 'n.mp3', _frames())
    h = mp3_header(p)
    assert h.tag == 'none' and lead_samples(h, 'mpg123') == 0 and lead_samples(h, 'ffmpeg') == 0
    o = tmp_path / 's.ogg'
    o.write_bytes(b'OggS')
    assert lead_seconds(o, 'mpg123') == 0.0
