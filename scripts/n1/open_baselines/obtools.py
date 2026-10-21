"""The open baselines' file formats: canonical WAVs, Ardour's tempo rounding, raw-to-normalized converters, probes.

Standard library and numpy only, so the Mac runs it with the project's environment. The normalized output is
schema ``ensomi-open-baseline-v1``, read by `ensomi_n.mapformat.open_tools`; each converter here writes one
normalized output per arm from one tool's raw output, and never reads a chart.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

SCHEMA = 'ensomi-open-baseline-v1'

#: WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT, WAVE_FORMAT_EXTENSIBLE (the subformat GUID's first two bytes give the tag).
PCM, FLOAT, EXTENSIBLE = 1, 3, 0xFFFE


# ---------------------------------------------------------------- WAV

def wav_info(path, *, digest: bool = True) -> dict:
    """The fmt and data chunks of a RIFF/WAVE file, and the sha256 of the data payload.

    The capture records hash the full interleaved sample payload without headers or metadata
    (``pcm_sha256``); ``digest`` computes the same over the data chunk. ``tag`` is the effective format tag:
    the subformat's for WAVE_FORMAT_EXTENSIBLE. Raises ValueError for a file that is not RIFF/WAVE or lacks a
    fmt or data chunk.
    """
    p = Path(path)
    with open(p, 'rb') as f:
        head = f.read(12)
        if len(head) < 12 or head[:4] != b'RIFF' or head[8:12] != b'WAVE':
            raise ValueError(f'{p}: not a RIFF/WAVE file')
        fmt = None
        data = None
        pos = 12
        size_total = p.stat().st_size
        while pos + 8 <= size_total:
            f.seek(pos)
            cid, size = struct.unpack('<4sI', f.read(8))
            if cid == b'fmt ':
                body = f.read(size)
                tag, channels, rate, _, align, bits = struct.unpack('<HHIIHH', body[:16])
                fmt = dict(format_tag=tag, channels=channels, sample_rate_hz=rate, block_align=align,
                           bits_per_sample=bits, tag=tag)
                if tag == EXTENSIBLE and len(body) >= 26:
                    fmt['tag'] = struct.unpack('<H', body[24:26])[0]
            elif cid == b'data':
                data = dict(data_offset=pos + 8, data_bytes=min(size, size_total - pos - 8))
            pos += 8 + size + (size & 1)
        if fmt is None or data is None:
            raise ValueError(f'{p}: missing fmt or data chunk')
        out = dict(fmt, **data)
        out['frames'] = out['data_bytes'] // out['block_align'] if out['block_align'] else 0
        if digest:
            h = hashlib.sha256()
            f.seek(out['data_offset'])
            left = out['data_bytes']
            while left > 0:
                block = f.read(min(1 << 20, left))
                if not block:
                    break
                h.update(block)
                left -= len(block)
            out['pcm_sha256'] = h.hexdigest()
    return out


def read_wav(path) -> tuple[np.ndarray, int]:
    """All channels as float32 ``(frames, channels)`` and the sample rate; 32-bit float or 16-bit integer only."""
    info = wav_info(path, digest=False)
    raw = Path(path).read_bytes()[info['data_offset']:info['data_offset'] + info['frames'] * info['block_align']]
    if info['tag'] == FLOAT and info['bits_per_sample'] == 32:
        x = np.frombuffer(raw, dtype='<f4').astype(np.float32)
    elif info['tag'] == PCM and info['bits_per_sample'] == 16:
        x = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0
    else:
        raise ValueError(f"{path}: format {info['tag']} at {info['bits_per_sample']} bits is not read here")
    return x.reshape(-1, info['channels']), int(info['sample_rate_hz'])


def write_wav_float(path, x: np.ndarray, rate: int, *, extensible: bool = False) -> None:
    """32-bit float WAV, as ``ffmpeg -c:a pcm_f32le`` writes one; ``extensible`` writes WAVE_FORMAT_EXTENSIBLE."""
    x = np.asarray(x, dtype='<f4')
    if x.ndim == 1:
        x = x[:, None]
    ch = x.shape[1]
    payload = x.tobytes()
    align = 4 * ch
    if extensible:
        guid = struct.pack('<H', FLOAT) + b'\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71'
        mask = {1: 0x4, 2: 0x3}.get(ch, 0)
        fmt = struct.pack('<HHIIHHHHI', EXTENSIBLE, ch, rate, rate * align, align, 32, 22, 32, mask) + guid
    else:
        fmt = struct.pack('<HHIIHHH', FLOAT, ch, rate, rate * align, align, 32, 0)
    fact = struct.pack('<I', x.shape[0])
    body = b'WAVE' + b'fmt ' + struct.pack('<I', len(fmt)) + fmt + b'fact' + struct.pack('<I', 4) + fact \
        + b'data' + struct.pack('<I', len(payload)) + payload
    Path(path).write_bytes(b'RIFF' + struct.pack('<I', len(body)) + body)


# ---------------------------------------------------------------- Ardour 9.8's tempo rounding

def c_round(x: float) -> float:
    """C's ``round``: halves away from zero (Python's `round` takes halves to even)."""
    return math.copysign(math.floor(abs(x) + 0.5), x)


def ardour_rounded_tempo(qpm: float, seconds: float) -> tuple[float, float]:
    """``ARDOUR::estimate_audio_tempo_source``'s rounding (``libs/ardour/utils.cc`` lines 873-903 at ``d66bd6c``).

    The estimate is forced to a whole number of 4/4 bars over the analysed audio: the beat count is rounded,
    then the bar count; a bar count of 18 or fewer goes up to the next of 1, 2, 4, 8 or 16 bars (17 and 18
    stay), and the tempo becomes the resulting beat count over the duration. Returns ``(qpm, beatcount)``; a 0
    estimate (MiniBPM's "too short") is returned unchanged with beat count 0, as Ardour leaves it.
    """
    if qpm == 0.0:
        return 0.0, 0.0
    beatcount = c_round((seconds / 60.0) * qpm)
    barcount = c_round(beatcount / 4.0)
    if barcount <= 18:
        for pwr in range(5):
            bc = float(2 ** pwr)
            if barcount <= bc:
                barcount = bc
                break
    beatcount = c_round(barcount * 4.0)
    return beatcount / (seconds / 60.0), beatcount


def ardour_source_window(frames: int, rate: int) -> tuple[int, int] | None:
    """The samples ``AudioSource::estimate_tempo`` hands MiniBPM (``audiosource.cc`` lines 136-175): ten seconds
    from the middle, ``(start, count)``; None below ten seconds, where Ardour estimates nothing."""
    ten = int(rate) * 10
    if frames < ten:
        return None
    return max(0, (frames - ten) // 2), ten


# ---------------------------------------------------------------- normalized outputs

def _base(arm: str, tool: str, song_id: str, audio: dict, tool_version: str = '') -> dict:
    return dict(schema=SCHEMA, arm=arm, tool=tool, tool_version=tool_version, song_id=song_id, clock='decoded',
                audio=dict(audio))


def failed_output(arm: str, tool: str, song_id: str, audio: dict, error: str, tool_version: str = '') -> dict:
    d = _base(arm, tool, song_id, audio, tool_version)
    d.update(status='failed', kind=None, error=str(error))
    return d


def _vamp_time(item) -> float:
    sec, nsec = int(item[0]), int(item[1])
    return sec + nsec * 1e-9


def vamp_outputs(raw: dict, song_id: str, audio: dict, tool_version: str = '') -> dict:
    """Normalized outputs ``{arm: output}`` from the raw JSON of one ``ardour_vamp.lua`` run.

    The arm comes from the run's tag. ``qm-barbeattracker`` runs (``qm-bbt-bpb<k>``) give ``qm-bbt`` (beats per
    bar 4, the plugin's default) or ``qm-bbt-m<k>`` (beats per bar k, an oracle's input when k comes from the
    chart): beats from output 0, bar starts at the beats labelled 1, checked against output 1's bar times.
    ``qm-tempotracker`` gives ``qm-tempo``: beats from output 0 and no bars. A run with an error, an
    unanalysed file, a session at another rate than the file, or a read length other than the canonical frame
    count is ``failed``; a completed run without beats is ``empty``.
    """
    run = str(raw.get('run', ''))
    if run.startswith('qm-bbt-bpb'):
        bpb = int(run[len('qm-bbt-bpb'):])
        arm = 'qm-bbt' if bpb == 4 else f'qm-bbt-m{bpb}'
    elif run == 'qm-tempo':
        bpb, arm = None, 'qm-tempo'
    else:
        raise ValueError(f'unknown ardour_vamp.lua run {run!r}')
    tool = 'ardour-vamp'
    prov = {k: raw.get(k) for k in ('run', 'plugin', 'plugin_name', 'plugin_version', 'parameters', 'step',
                                    'block', 'session_rate', 'readable_length', 'channels', 'channel',
                                    'analyze_rv', 'runner')}
    problems = []
    if raw.get('error'):
        problems.append(f"runner error: {raw['error']}")
    got = (raw.get('parameters') or {}).get('bpb')
    if bpb is not None and not raw.get('error') and (got is None or int(round(float(got))) != bpb):
        problems.append(f'the plugin reports beats per bar {got}, the run asked for {bpb}')
    if raw.get('analyze_rv') not in (0, None) and not raw.get('error'):
        problems.append(f"analyze returned {raw.get('analyze_rv')}")
    if raw.get('session_rate') is not None and int(raw['session_rate']) != int(audio['sample_rate_hz']):
        problems.append(f"session at {raw['session_rate']} Hz, file at {audio['sample_rate_hz']} Hz (Ardour would resample)")
    if raw.get('readable_length') is not None and int(raw['readable_length']) != int(audio['frames']):
        problems.append(f"Ardour read {raw['readable_length']} frames, the canonical decode has {audio['frames']}")
    if problems:
        out = failed_output(arm, tool, song_id, audio, '; '.join(problems), tool_version)
        out['provenance'] = prov
        return {arm: out}
    outs = raw.get('outputs') or {}
    beats = outs.get('0') or []
    t = np.array([_vamp_time(b) for b in beats], dtype=np.float64)
    d = _base(arm, tool, song_id, audio, tool_version)
    d['provenance'] = prov
    if not len(t):
        d.update(status='empty', kind='beats', beats_s=[], downbeats=None)
        return {arm: d}
    if np.any(np.diff(t) <= 0):
        out = failed_output(arm, tool, song_id, audio, 'beat times are not strictly increasing', tool_version)
        out['provenance'] = prov
        return {arm: out}
    d.update(status='ok', kind='beats', beats_s=[float(x) for x in t])
    if arm == 'qm-tempo':
        d['downbeats'] = None
        tempo = [float(x[3]) for x in outs.get('2') or [] if x[3] is not None]
        prov['tempo_track'] = dict(changes=len(tempo), min=min(tempo) if tempo else None,
                                   max=max(tempo) if tempo else None)
        return {arm: d}
    down = [i for i, b in enumerate(beats) if str(b[2]).strip() == '1']
    bars = np.array([_vamp_time(b) for b in outs.get('1') or []], dtype=np.float64)
    prov['bar_check'] = dict(bars_output=int(len(bars)), labelled_one=len(down),
                             equal=bool(len(bars) == len(down) and np.array_equal(bars, t[down])))
    d['downbeats'] = down
    return {arm: d}


def minibpm_outputs(raw: dict, song_id: str, audio: dict, tool_version: str = '') -> dict:
    """Normalized outputs of one ``minibpm_cli`` run: three arms, each one tempo laid from the file start in 4/4.

    - ``mbpm-raw``: MiniBPM's own estimate over the whole file's left channel, before Ardour rounds it;
    - ``mbpm-clip``: that estimate after ``ardour_rounded_tempo`` over the file's duration, the clip trigger's
      tempo for a file dropped into a slot (``AudioTrigger::estimate_tempo``); the trigger plays the clip from
      its first sample, so the grid starts there;
    - ``mbpm-source``: MiniBPM on the ten seconds from the middle, rounded over those ten seconds, the tempo
      ``AudioSource::estimate_tempo`` stores for a recording or an embedded file.
    A MiniBPM result of 0 is ``empty``.
    """
    tool = 'ardour-minibpm'
    rate, frames = int(audio['sample_rate_hz']), int(audio['frames'])
    out = {}
    problems = []
    if raw.get('error'):
        problems.append(str(raw['error']))
    if raw.get('frames') is not None and int(raw['frames']) != frames:
        problems.append(f"read {raw['frames']} frames, the canonical decode has {frames}")
    if raw.get('sample_rate') is not None and int(raw['sample_rate']) != rate:
        problems.append(f"read {raw['sample_rate']} Hz, the canonical decode is {rate} Hz")
    if problems:
        return {arm: failed_output(arm, tool, song_id, audio, '; '.join(problems), tool_version)
                for arm in ('mbpm-raw', 'mbpm-clip', 'mbpm-source')}
    whole = raw.get('whole') or {}
    middle = raw.get('middle10')
    seconds = frames / rate
    est = float(whole.get('bpm', 0.0))
    cands = whole.get('candidates')
    rounded, beats = ardour_rounded_tempo(est, seconds)
    rows = [('mbpm-raw', est, dict(minibpm=est, candidates=cands, seconds=seconds)),
            ('mbpm-clip', rounded, dict(minibpm=est, rounded_beatcount=beats, seconds=seconds,
                                        rule='utils.cc estimate_audio_tempo_source rounding over the whole file'))]
    if middle is None:
        rows.append(('mbpm-source', None, dict(reason='under ten seconds: Ardour estimates nothing')))
    else:
        m_est = float(middle.get('bpm', 0.0))
        win = ardour_source_window(frames, rate)
        m_round, m_beats = ardour_rounded_tempo(m_est, 10.0)
        rows.append(('mbpm-source', m_round, dict(minibpm=m_est, rounded_beatcount=m_beats, window=win,
                                                  candidates=middle.get('candidates'),
                                                  rule='audiosource.cc estimate_tempo: ten seconds from the middle')))
    for arm, bpm, prov in rows:
        d = _base(arm, tool, song_id, audio, tool_version)
        d['provenance'] = prov
        if bpm is None or bpm == 0.0:
            d.update(status='empty', kind='tempo', bpm=0.0, grid=dict(anchor_s=0.0, meter=4))
        else:
            d.update(status='ok', kind='tempo', bpm=float(bpm), grid=dict(anchor_s=0.0, meter=4))
        out[arm] = d
    return out


def tempocnn_outputs(raw: dict, song_id: str, audio: dict, tool_version: str = '') -> dict:
    """Normalized output of one ``tempocnn_one.mjs`` run, arm ``tcnn``.

    openDAW only shows the BPM in a dialog and writes nothing (``RegionContextMenu.ts`` lines 194-206), so the
    grid is a user's step, not the host's: the BPM as the project tempo with the region at the timeline's
    start, in openDAW's default 4/4.
    """
    arm, tool = 'tcnn', 'opendaw-tempocnn'
    prov = {k: raw.get(k) for k in ('confidence', 'topCandidates', 'opendaw_wav_decoder', 'opendaw_left_equal',
                                    'model_sha256', 'onnxruntime', 'node', 'ms', 'channel_used')}
    problems = []
    if raw.get('error'):
        problems.append(str(raw['error']))
    if raw.get('frames') is not None and int(raw['frames']) != int(audio['frames']):
        problems.append(f"read {raw['frames']} frames, the canonical decode has {audio['frames']}")
    if raw.get('sampleRate') is not None and int(raw['sampleRate']) != int(audio['sample_rate_hz']):
        problems.append(f"read {raw['sampleRate']} Hz, the canonical decode is {audio['sample_rate_hz']} Hz")
    if problems:
        out = failed_output(arm, tool, song_id, audio, '; '.join(problems), tool_version)
        out['provenance'] = prov
        return {arm: out}
    d = _base(arm, tool, song_id, audio, tool_version)
    d['provenance'] = prov
    bpm = raw.get('bpm')
    if bpm is None or not float(bpm) > 0:
        d.update(status='empty', kind='tempo', bpm=0.0, grid=dict(anchor_s=0.0, meter=4))
    else:
        d.update(status='ok', kind='tempo', bpm=float(bpm), grid=dict(anchor_s=0.0, meter=4))
    return {arm: d}


def dump(path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1) + '\n')
    tmp.replace(p)


# ---------------------------------------------------------------- probes

PROBE_BPM = 120.0
PROBE_FIRST_S = 1.0
PROBE_SECONDS = 40.0


def click_probe(rate: int, *, bpm: float = PROBE_BPM, first_s: float = PROBE_FIRST_S,
                seconds: float = PROBE_SECONDS) -> tuple[np.ndarray, np.ndarray]:
    """A stereo click track, identical channels: a decaying sine burst on every beat (1760 Hz on the downbeat of
    each 4/4 bar, 1320 Hz otherwise), each starting exactly on its beat's sample. Returns ``(audio, beats_s)``.

    It measures what a tracker reports for attacks at known times, a fact about the tool's clock; it is never
    applied to the songs.
    """
    n = int(round(seconds * rate))
    x = np.zeros(n, dtype=np.float64)
    beats = []
    k = 0
    while True:
        t = first_s + k * 60.0 / bpm
        start = int(round(t * rate))
        if start >= n - int(0.05 * rate):
            break
        beats.append(start / rate)
        f = 1760.0 if k % 4 == 0 else 1320.0
        m = np.arange(int(0.03 * rate))
        burst = 0.8 * np.sin(2 * np.pi * f * m / rate) * np.exp(-m / (0.006 * rate))
        x[start:start + len(m)] += burst[:max(0, n - start)]
        k += 1
    return np.stack([x, x], axis=1).astype(np.float32), np.array(beats)


def probe_offsets(beats_s, truth_s, lo_s: float = 5.0, hi_s: float = 35.0) -> dict:
    """Signed distance (ms) from each true beat in ``[lo_s, hi_s]`` to the nearest stated beat: median, spread,
    and the share within 20 ms."""
    b = np.asarray(beats_s, dtype=np.float64)
    tr = np.asarray(truth_s, dtype=np.float64)
    tr = tr[(tr >= lo_s) & (tr <= hi_s)]
    if not len(b) or not len(tr):
        return dict(n=int(len(tr)), median_ms=None, mad_ms=None, within20=None)
    j = np.clip(np.searchsorted(b, tr), 1, len(b) - 1) if len(b) > 1 else np.zeros(len(tr), dtype=int)
    cand = np.stack([b[np.maximum(j - 1, 0)], b[j]], axis=1)
    e = cand - tr[:, None]
    e = e[np.arange(len(tr)), np.argmin(np.abs(e), axis=1)] * 1000.0
    med = float(np.median(e))
    return dict(n=int(len(tr)), median_ms=med, mad_ms=float(np.median(np.abs(e - med))),
                within20=float(np.mean(np.abs(e) <= 20.0)))
