"""osu!'s clock: the samples at an MP3's start that osu! keeps and our decoder drops.

osu! plays audio through BASS, which removes an MP3's encoder delay plus the
decoder's 529-sample delay only when the first frame carries a LAME extension
with a non-zero delay field, and removes nothing otherwise. Mappers place notes
on what BASS plays, so a decoder that removes more puts every onset early.

- libsndfile (through libmpg123, the loader N uses) removes the delay plus 529
  samples whenever a Xing/Info frame is present. It is therefore 529 samples
  early on a Xing/Info frame without a LAME extension, and on a LAME
  extension whose delay field is 0.
- ffmpeg removes the delay plus 529 only when the extension's encoder string
  starts with LAME, Lavf or Lavc. It is 529 samples early on such an extension
  with a zero delay field.
- All agree on files without a Xing/Info frame and on Ogg Vorbis.

`lead_samples` gives the silence to prepend to a decoder's output so that its
clock is osu!'s. This module is the N line's own implementation of the rule
(version ``LEAD_RULE``); it is checked against encoded click tracks
(``tests/test_clock.py`` here, and the mac fixture run), not against another
implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

LEAD_RULE = 'osu-lead-v1'
DECODER_DELAY = 529
FFMPEG_ENCODERS = ('LAME', 'Lavf', 'Lavc')

_KBPS = {
    'v1': (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320),
    'v2': (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
}
_RATES = {3: (44100, 48000, 32000), 2: (22050, 24000, 16000), 0: (11025, 12000, 8000)}


@dataclass(frozen=True)
class Mp3Start:
    """What the first audio frame of an MP3 says about gapless playback.

    ``kind`` is ``lame`` (a Xing/Info frame with a LAME-style extension),
    ``xing`` (a Xing/Info frame without one), ``vbri``, ``plain`` (the first
    frame is audio) or ``unsynced`` (no two consecutive frames found).
    ``delay`` is the extension's encoder-delay field, -1 without one.
    """
    kind: str
    encoder: str = ''
    delay: int = -1
    sample_rate: int = -1


def _layer3_frame(buf: bytes, i: int):
    """(sample_rate, frame_bytes, side_info_bytes) of an MPEG-1/2/2.5 Layer III header at ``i``, else None."""
    if i + 4 > len(buf):
        return None
    b0, b1, b2, b3 = buf[i], buf[i + 1], buf[i + 2], buf[i + 3]
    if b0 != 0xFF or (b1 & 0xE0) != 0xE0:
        return None
    version = (b1 >> 3) & 0x3          # 0: MPEG-2.5, 2: MPEG-2, 3: MPEG-1
    layer = (b1 >> 1) & 0x3            # 1: Layer III
    rate_bits = b2 >> 4
    sr_bits = (b2 >> 2) & 0x3
    if version == 1 or layer != 1 or rate_bits in (0, 15) or sr_bits == 3:
        return None
    sample_rate = _RATES[version][sr_bits]
    kbps = _KBPS['v1' if version == 3 else 'v2'][rate_bits]
    padding = (b2 >> 1) & 0x1
    mono = (b3 >> 6) == 3
    if version == 3:
        size = 144 * kbps * 1000 // sample_rate + padding
        side = 17 if mono else 32
    else:
        size = 72 * kbps * 1000 // sample_rate + padding
        side = 9 if mono else 17
    return sample_rate, size, side


def _skip_id3(head: bytes) -> int:
    if len(head) >= 10 and head[:3] == b'ID3':
        size = 0
        for byte in head[6:10]:
            size = (size << 7) | (byte & 0x7F)
        return 10 + size + (10 if head[5] & 0x10 else 0)
    return 0


def read_mp3_start(path: str | os.PathLike, scan_bytes: int = 1 << 16) -> Mp3Start:
    """Classify the first MPEG frame of ``path`` (after an ID3v2 tag)."""
    with open(path, 'rb') as handle:
        start = _skip_id3(handle.read(10))
        handle.seek(start)
        buf = handle.read(scan_bytes)
    for i in range(max(len(buf) - 4, 0)):
        frame = _layer3_frame(buf, i)
        if frame is None:
            continue
        sample_rate, size, side = frame
        if size <= 4 or _layer3_frame(buf, i + size) is None:
            continue                        # a false sync: the next frame must follow
        if buf[i + 36:i + 40] == b'VBRI':
            return Mp3Start('vbri', sample_rate=sample_rate)
        x = i + 4 + side
        if buf[x:x + 4] not in (b'Xing', b'Info'):
            return Mp3Start('plain', sample_rate=sample_rate)
        flags = int.from_bytes(buf[x + 4:x + 8], 'big')
        p = x + 8
        p += 4 if flags & 0x1 else 0        # frame count
        p += 4 if flags & 0x2 else 0        # byte count
        p += 100 if flags & 0x4 else 0      # seek table
        p += 4 if flags & 0x8 else 0        # quality
        ext = buf[p:p + 24]
        if len(ext) < 24 or not all(32 <= c < 127 for c in ext[:4]):
            return Mp3Start('xing', sample_rate=sample_rate)
        encoder = ext[:9].decode('latin-1').rstrip('\x00 ')
        delay = (ext[21] << 4) | (ext[22] >> 4)
        return Mp3Start('lame', encoder, delay, sample_rate)
    return Mp3Start('unsynced')


def lead_samples(start: Mp3Start, decoder: str) -> int:
    """Samples, at the file's rate, to prepend to ``decoder``'s output: 0 or 529."""
    if decoder == 'libsndfile':
        early = start.kind == 'xing' or (start.kind == 'lame' and start.delay == 0)
    elif decoder == 'ffmpeg':
        early = start.kind == 'lame' and start.delay == 0 and start.encoder.startswith(FFMPEG_ENCODERS)
    else:
        raise ValueError(f'unknown decoder {decoder!r}')
    return DECODER_DELAY if early else 0


def file_lead(path: str | os.PathLike, decoder: str) -> tuple[int, int, Mp3Start | None]:
    """(lead samples, file sample rate or -1, header) for ``path``; non-MP3 files lead 0."""
    if Path(path).suffix.lower() != '.mp3':
        return 0, -1, None
    start = read_mp3_start(path)
    return lead_samples(start, decoder), start.sample_rate, start
