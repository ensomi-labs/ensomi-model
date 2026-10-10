"""MP3 start trimming: the lead osu!'s decoder keeps and ours removes, per header class.

osu! decodes with BASS under a flag (config 68) that trims the encoder delay plus 529 samples
only when a LAME tag carries a non-zero delay field, and trims nothing otherwise. Our two
decoders differ from it on two header classes (measured 2026-10-09 against the WAV source and on
corpus files, `~/ensomi/.sync/cp/scratch/timing/step3/report-step3.md`, E0):

- libsndfile/mpg123 (BeatThis's loader) trims 529 samples whenever a Xing/Info header carries a
  frame count, so it is 529 samples early on Xing-only files and on LAME tags with a zero delay;
- ffmpeg (pydub, the Mel loader) trims the delay plus 529 only when the encoder string is LAME,
  Lavf or Lavc, so it is 529 samples early on LAME tags with a zero delay only.

Both agree with BASS on LAME tags with a delay, on files without a header and on OGG.
``lead_seconds`` gives the time to prepend to a decoder's output so that its clock is osu!'s,
the clock the mappers' notes are on. About 2 % of corpus songs are affected.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

LEAD_SAMPLES = 529
_FFMPEG_ENCODERS = ('LAME', 'Lavf', 'Lavc')
_BITRATES_V1 = (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320)
_BITRATES_V2 = (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160)
_SAMPLE_RATES = {3: (44100, 48000, 32000), 2: (22050, 24000, 16000), 0: (11025, 12000, 8000)}


@dataclass(frozen=True)
class Mp3Header:
    tag: str                # 'lame' (Xing/Info with a LAME extension), 'xing' (without), 'vbri', 'none', 'nosync'
    encoder: str            # the extension's encoder string, '' when absent
    delay: int              # the extension's encoder delay field, -1 when absent
    sample_rate: int        # from the first frame header, -1 when none was found


def _frame(b: bytes, j: int):
    if j + 4 > len(b) or b[j] != 0xFF or (b[j + 1] & 0xE0) != 0xE0:
        return None
    version, layer = (b[j + 1] >> 3) & 3, (b[j + 1] >> 1) & 3
    bitrate_index, rate_index, padding, mode = b[j + 2] >> 4, (b[j + 2] >> 2) & 3, (b[j + 2] >> 1) & 1, b[j + 3] >> 6
    if version == 1 or layer != 1 or bitrate_index in (0, 15) or rate_index == 3:
        return None
    sample_rate = _SAMPLE_RATES[version][rate_index]
    bitrate = (_BITRATES_V1 if version == 3 else _BITRATES_V2)[bitrate_index] * 1000
    length = (144 if version == 3 else 72) * bitrate // sample_rate + padding
    side_info = (17 if mode == 3 else 32) if version == 3 else (9 if mode == 3 else 17)
    return sample_rate, length, side_info


def mp3_header(path) -> Mp3Header:
    """Parse the first MPEG frame (after any ID3v2 tag) for a Xing/Info header and its LAME extension."""
    with open(path, 'rb') as f:
        head = f.read(10)
        start = 0
        if head[:3] == b'ID3':
            size = ((head[6] & 0x7f) << 21) | ((head[7] & 0x7f) << 14) | ((head[8] & 0x7f) << 7) | (head[9] & 0x7f)
            start = 10 + size + (10 if head[5] & 0x10 else 0)
        f.seek(start)
        b = f.read(65536)
    for j in range(len(b) - 4):
        frame = _frame(b, j)
        if frame is None:
            continue
        sample_rate, length, side_info = frame
        if _frame(b, j + length) is None:            # the next frame must sync too
            continue
        if b[j + 36:j + 40] == b'VBRI':
            return Mp3Header('vbri', '', -1, sample_rate)
        x = j + 4 + side_info
        if b[x:x + 4] not in (b'Xing', b'Info'):
            return Mp3Header('none', '', -1, sample_rate)
        flags = int.from_bytes(b[x + 4:x + 8], 'big')
        p = x + 8 + 4 * bool(flags & 1) + 4 * bool(flags & 2) + 100 * bool(flags & 4) + 4 * bool(flags & 8)
        encoder = b[p:p + 9]
        if len(encoder) < 9 or not all(32 <= c < 127 for c in encoder[:4]):
            return Mp3Header('xing', '', -1, sample_rate)
        delay = (b[p + 21] << 4) | (b[p + 22] >> 4)
        return Mp3Header('lame', encoder.decode('latin-1').strip('\x00 '), delay, sample_rate)
    return Mp3Header('nosync', '', -1, -1)


def lead_samples(header: Mp3Header, decoder: str) -> int:
    """Samples at the file start that osu!'s decoder keeps and ``decoder`` ('mpg123' or 'ffmpeg') removes."""
    if decoder == 'mpg123':
        early = header.tag == 'xing' or (header.tag == 'lame' and header.delay == 0)
    elif decoder == 'ffmpeg':
        early = header.tag == 'lame' and header.delay == 0 and header.encoder.startswith(_FFMPEG_ENCODERS)
    else:
        raise ValueError(f'unknown decoder {decoder!r}')
    return LEAD_SAMPLES if early else 0


def lead_seconds(path, decoder: str) -> float:
    """Seconds to prepend to ``decoder``'s output of ``path`` to put it on osu!'s clock; 0 for non-MP3 files."""
    if Path(path).suffix.lower() != '.mp3':
        return 0.0
    header = mp3_header(path)
    lead = lead_samples(header, decoder)
    return lead / header.sample_rate if lead and header.sample_rate > 0 else 0.0
