"""Click-track fixtures for osu!'s clock (absolute check, design §4.1 and guardrail 10).

A WAV holds short tone bursts at known times. It is encoded as:

- ``lame``: MP3 with a Xing/Info frame and a LAME extension carrying the
  encoder delay (ffmpeg's default);
- ``lame-delay0``: the same file with the extension's delay field set to 0;
- ``xing``: the same file with the extension blanked (a Xing/Info frame only);
- ``plain``: MP3 without a Xing/Info frame;
- ``ogg``: Ogg Vorbis; and the WAV itself.

Where each burst must sit on osu!'s clock follows from what BASS removes: the
encoder delay ``D`` plus 529 samples for ``lame``, nothing for the other MP3
classes (so their bursts sit ``(D + 529) / sr`` late), nothing for Ogg and
WAV. Each file is decoded by `decode.decode` (libsndfile, and ffmpeg as a
second decoder), turned into the cached log-Mel, and each burst is located as
the parabolic peak of the band-summed Mel power near it. A case passes when
every burst is within ``TOLERANCE_MS`` of where it must be. The expected
values are fixed by the rule above, not by another implementation.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from . import clock
from .decode import decode, _ffmpeg, resample
from .mel import DEFAULT, log_mel

SOURCE_RATE = 44100
TOLERANCE_MS = 2.0
CLICK_MS = (1000.0, 2003.0, 3507.5, 5011.0, 7250.25, 9001.0)


def click_track(rate: int = SOURCE_RATE, seconds: float = 10.0, times_ms=CLICK_MS) -> np.ndarray:
    """Hann-windowed 2 kHz bursts of 4 ms centred at ``times_ms``, over a quiet noise floor."""
    n = int(rate * seconds)
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1e-4, n).astype(np.float64)
    half = int(0.002 * rate)
    k = np.arange(-half, half + 1)
    for t in times_ms:
        c = t / 1000.0 * rate
        i0 = int(np.floor(c))
        idx = i0 + k
        dt = (idx - c) / rate
        burst = 0.8 * np.sin(2 * np.pi * 2000.0 * dt) * 0.5 * (1 + np.cos(np.pi * dt / 0.002))
        x[idx] += burst
    return x.astype(np.float32)


def locate(mel: np.ndarray, near_ms, frame_ms: float = 10.0, search_ms: float = 60.0) -> np.ndarray:
    """Sub-frame time of the band-summed power peak nearest each of ``near_ms``."""
    energy = np.exp(np.asarray(mel, dtype=np.float64)).sum(axis=1)
    out = []
    for t in near_ms:
        a = max(int((t - search_ms) / frame_ms), 1)
        b = min(int((t + search_ms) / frame_ms) + 1, len(energy) - 1)
        i = a + int(np.argmax(energy[a:b]))
        y0, y1, y2 = np.log(energy[i - 1:i + 2])
        denom = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / denom if denom < 0 else 0.0
        out.append((i + frac) * frame_ms)
    return np.array(out)


def _write_wav(path: Path, x: np.ndarray, rate: int) -> None:
    import soundfile
    soundfile.write(str(path), x, rate, subtype='PCM_16')


def _lame_extension_offset(data: bytes) -> tuple[int, int]:
    """(offset of the LAME extension's encoder string, sample rate) in an MP3's bytes."""
    start = clock._skip_id3(data[:10])
    buf = data[start:start + (1 << 16)]
    for i in range(len(buf) - 4):
        frame = clock._layer3_frame(buf, i)
        if frame is None or clock._layer3_frame(buf, i + frame[1]) is None:
            continue
        x = i + 4 + frame[2]
        if buf[x:x + 4] not in (b'Xing', b'Info'):
            raise ValueError('no Xing/Info frame')
        flags = int.from_bytes(buf[x + 4:x + 8], 'big')
        p = x + 8 + 4 * bool(flags & 1) + 4 * bool(flags & 2) + 100 * bool(flags & 4) + 4 * bool(flags & 8)
        return start + p, frame[0]
    raise ValueError('no MPEG frame')


def make_files(out: Path) -> dict:
    """Write the fixture files; returns {case: (path, expected late samples at SOURCE_RATE or None)}."""
    out.mkdir(parents=True, exist_ok=True)
    wav = out / 'clicks.wav'
    _write_wav(wav, click_track(), SOURCE_RATE)
    files = {'wav': (wav, 0)}
    if shutil.which('ffmpeg') is None:
        return files
    run = lambda *a: subprocess.run(['ffmpeg', '-v', 'error', '-y', *a], check=True, capture_output=True)
    lame = out / 'lame.mp3'
    run('-i', str(wav), '-c:a', 'libmp3lame', '-b:a', '192k', str(lame))
    data = bytearray(lame.read_bytes())
    p, _ = _lame_extension_offset(bytes(data))
    delay = (data[p + 21] << 4) | (data[p + 22] >> 4)
    files['lame'] = (lame, 0)
    d0 = bytearray(data)
    d0[p + 21] = 0
    d0[p + 22] &= 0x0F
    (out / 'lame-delay0.mp3').write_bytes(bytes(d0))
    files['lame-delay0'] = (out / 'lame-delay0.mp3', delay + clock.DECODER_DELAY)
    xg = bytearray(data)
    xg[p:p + 36] = bytes(36)
    (out / 'xing.mp3').write_bytes(bytes(xg))
    files['xing'] = (out / 'xing.mp3', delay + clock.DECODER_DELAY)
    plain = out / 'plain.mp3'
    run('-i', str(wav), '-c:a', 'libmp3lame', '-b:a', '192k', '-write_xing', '0', str(plain))
    files['plain'] = (plain, delay + clock.DECODER_DELAY)
    ogg = out / 'clicks.ogg'
    try:
        run('-i', str(wav), '-c:a', 'libvorbis', '-q:a', '6', str(ogg))
    except subprocess.CalledProcessError:
        import soundfile
        soundfile.write(str(ogg), click_track(), SOURCE_RATE, format='OGG', subtype='VORBIS')
    files['ogg'] = (ogg, 0)
    files['_encoder_delay'] = delay
    return files


def run(out_dir: str | Path) -> dict:
    """Build, decode and check every case with both decoders; returns the report."""
    out = Path(out_dir)
    files = make_files(out)
    delay = files.pop('_encoder_delay', None)
    cases = []
    for name, (path, late) in files.items():
        start = clock.read_mp3_start(path) if path.suffix == '.mp3' else None
        for decoder in ('libsndfile', 'ffmpeg'):
            if decoder == 'ffmpeg' and shutil.which('ffmpeg') is None:
                continue
            if decoder == 'libsndfile':
                dec = decode(path)
                wave, lead = dec.wave, dec.lead_samples
                used = dec.decoder
            else:
                raw, sr = _ffmpeg(path)
                lead = clock.file_lead(path, 'ffmpeg')[0]
                raw = np.concatenate([np.zeros(lead, dtype=np.float32), raw]) if lead else raw
                wave, used = resample(raw, sr, DEFAULT.sample_rate), 'ffmpeg'
            mel = log_mel(wave)
            expected = np.array(CLICK_MS) + 1000.0 * late / SOURCE_RATE
            found = locate(mel, expected)
            err = found - expected
            cases.append(dict(case=name, decoder=decoder, used=used, header=start.kind if start else '',
                              header_delay=start.delay if start else None, lead_samples=int(lead),
                              expected_late_ms=round(1000.0 * late / SOURCE_RATE, 3),
                              max_abs_err_ms=round(float(np.max(np.abs(err))), 3),
                              errors_ms=[round(float(e), 3) for e in err],
                              passed=bool(np.max(np.abs(err)) <= TOLERANCE_MS)))
    report = dict(rule=clock.LEAD_RULE, mel=DEFAULT.key(), tolerance_ms=TOLERANCE_MS, encoder_delay=delay,
                  cases=cases, passed=bool(cases) and all(c['passed'] for c in cases),
                  classes_covered=sorted({c['case'] for c in cases}))
    (out / 'clock-report.json').write_text(json.dumps(report, indent=1))
    return report
