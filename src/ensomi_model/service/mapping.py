"""Chart-generation messages <-> ``System.make``: request to arguments, the exported ``.osu`` to a ``Chart``, and what
the loaded model says it can do."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

import numpy as np
from ensomi.protocol.v1 import chart_pb2

from ..osu_core.hitobjects import ManiaHitObjectKind, parse_mania_hit_objects

MODEL_NAME = 'Ensomi R2'
MODEL_VERSION = 'r2-1.0'
MIN_STAR, MAX_STAR_EXCLUSIVE = 2.0, 6.0

LONG_NOTES = {chart_pb2.LONG_NOTE_AMOUNT_UNSPECIFIED: 'natural', chart_pb2.LONG_NOTE_AMOUNT_TAP_ONLY: 'tap',
              chart_pb2.LONG_NOTE_AMOUNT_HALF: 'half', chart_pb2.LONG_NOTE_AMOUNT_NATURAL: 'natural'}
PATTERNS = {chart_pb2.PATTERN_KIND_STREAM: ('stream', None), chart_pb2.PATTERN_KIND_JUMPSTREAM: ('stream', 'jumpstream'),
            chart_pb2.PATTERN_KIND_CHORDSTREAM: ('stream', 'chordstream'), chart_pb2.PATTERN_KIND_JACK: ('jack', None),
            chart_pb2.PATTERN_KIND_CHORDJACK: ('jack', 'chordjack')}

W, E, N = (chart_pb2.CONTROL_STATUS_WORKS, chart_pb2.CONTROL_STATUS_EXPERIMENTAL, chart_pb2.CONTROL_STATUS_NOT_WORKING)
CONTROLS = [
    ('star', W, 'Whole-song star rating; both blind pairs called the right way.'),
    ('jumpstream', W, 'A span of 8 to 32 seconds in that style; works in blind pairs.'),
    ('chordstream', W, 'A span of 8 to 32 seconds in that style; works in blind pairs.'),
    ('stream', E, 'Reaches its target band in measurement (48 of 48 runs); not screened on its own.'),
    ('jack', E, 'Reaches its target band in measurement (35 of 48 runs); not screened on its own.'),
    ('section_star', E, 'An absolute star target for one span; not screened, and easier spans land short.'),
    ('chordjack', N, 'Out of band on generated rows (18 of 28 runs).'),
    ('long_notes', N, 'Tap only, half or natural; half removes about a third and a blind pair could not tell.'),
]


class RequestError(Exception):
    """A request the service refuses; ``code`` is the ``ErrorEvent.code``."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def check_star(star, what='star'):
    if not MIN_STAR <= star < MAX_STAR_EXCLUSIVE:
        raise RequestError('star_out_of_range', f'{what} {star} is outside [{MIN_STAR:g}, {MAX_STAR_EXCLUSIVE:g})')


def make_arguments(request, cache_dir):
    """``GenerateChartRequest`` -> ``System.make`` keyword arguments (the output ``.osz`` goes in ``cache_dir``)."""
    if request.audio.WhichOneof('ref') != 'local_path':
        raise RequestError('audio_unreadable', 'only audio.local_path is supported')
    audio = Path(request.audio.local_path)
    try:
        audio.open('rb').close()
    except OSError as error:
        raise RequestError('audio_unreadable', f'cannot read {audio}: {error.strerror}') from error
    check_star(request.star)
    sections, stream, jack = [], [], []
    for s in request.controls.sections:
        if not s.start_ms < s.end_ms:
            raise RequestError('unsupported_control', f'section {s.start_ms}-{s.end_ms} ms is empty')
        if s.WhichOneof('target') == 'star':
            check_star(s.star, 'section star')
            sections.append(dict(start_ms=s.start_ms, end_ms=s.end_ms, star=s.star))
        elif s.WhichOneof('target') == 'pattern':
            if s.pattern.kind not in PATTERNS or not 0 <= s.pattern.strength <= 1:
                raise RequestError('unsupported_control', 'a pattern needs a kind and a strength in [0, 1]')
            concept, subtype = PATTERNS[s.pattern.kind]
            text = f'{s.start_ms},{s.end_ms},{s.pattern.strength!r}' + (f',{subtype}' if subtype else '')
            (stream if concept == 'stream' else jack).append(text)
        else:
            raise RequestError('unsupported_control', 'a section needs a star or a pattern')
    if request.controls.long_notes not in LONG_NOTES:
        raise RequestError('unsupported_control', f'unknown long_notes value {request.controls.long_notes}')
    return dict(audio=str(audio), star=request.star, out=str(Path(cache_dir) / f'{uuid.uuid4().hex}.osz'),
                sections=sections, stream=stream, jack=jack, ln=LONG_NOTES[request.controls.long_notes],
                seed=request.seed if request.HasField('seed') else 0, name='r2',
                title=request.title if request.HasField('title') else None,
                artist=request.artist if request.HasField('artist') else None)


def chart_from_osu(osu, duration_ms):
    """The exported ``.osu`` as a ``Chart``: whole-ms times, sorted by start then lane."""
    objects = sorted(parse_mania_hit_objects(osu), key=lambda o: (o.start_time_ms, o.lane))
    chart = chart_pb2.Chart(duration_ms=duration_ms)
    for o in objects:
        hold = o.kind == ManiaHitObjectKind.HOLD
        chart.objects.add(lane=o.lane, start_ms=round(o.start_time_ms), end_ms=round(o.end_time_ms) if hold else None)
    return chart


def generate(system, arguments, progress):
    """Run ``System.make`` and return the ``ChartResult``. The work directory ``make`` leaves is removed; the ``.osz``
    stays."""
    receipt = system.make(progress=progress, **arguments)
    work = Path(arguments['out']).with_suffix('')
    try:
        duration_ms = round(float(np.load(work / 'arrays.npz')['song_ms']))
        chart = chart_from_osu(work / 'generated.osu', duration_ms)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return chart_pb2.ChartResult(chart=chart, receipt=chart_pb2.GenerationReceipt(
        model_name=MODEL_NAME, model_version=MODEL_VERSION, seed=arguments['seed'], star=arguments['star'],
        receipt_json=json.dumps(receipt), osz_path=str(Path(arguments['out']).resolve())))


def capabilities(model_dir):
    """``ModelCapabilities`` with the checksums from ``<model_dir>/SHA256SUMS``."""
    message = chart_pb2.ModelCapabilities(model_name=MODEL_NAME, model_version=MODEL_VERSION, min_star=MIN_STAR,
                                          max_star_exclusive=MAX_STAR_EXCLUSIVE)
    for line in (Path(model_dir) / 'SHA256SUMS').read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        message.files.add(path=path, sha256_hex=digest)
    for name, status, note in CONTROLS:
        message.controls.add(name=name, status=status, note=note)
    return message
