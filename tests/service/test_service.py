import asyncio
from pathlib import Path
import threading

import numpy as np
import pytest
from ensomi.protocol.v1 import chart_pb2, envelope_pb2, inference_pb2

from ensomi_model.service import mapping
from ensomi_model.service.mapping import RequestError, capabilities, chart_from_osu, make_arguments
from ensomi_model.service.server import Service

HEADER = 'osu file format v14\n[General]\nMode:3\n[Difficulty]\nCircleSize:4\n[HitObjects]\n'
OSU = HEADER + ('448,192,3000,128,0,3600:0:0:0:0:\n64,192,1000,1,0,0:0:0:0:\n320,192,1000,1,0,0:0:0:0:\n'
                '192,192,2000,128,0,2500:0:0:0:0:\n64,192,2600,1,0,0:0:0:0:\n')


def request(audio, star=3.5, **kwargs):
    message = chart_pb2.GenerateChartRequest(audio=inference_pb2.AudioAssetRef(local_path=str(audio)), star=star)
    for key, value in kwargs.items():
        if key == 'controls':
            message.controls.CopyFrom(value)
        else:
            setattr(message, key, value)
    return message


def section(a, b, **target):
    s = chart_pb2.SectionRequest(start_ms=a, end_ms=b)
    if 'star' in target:
        s.star = target['star']
    else:
        s.pattern.CopyFrom(chart_pb2.PatternRequest(kind=target['kind'], strength=target['strength']))
    return s


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / 'song.mp3'
    path.write_bytes(b'audio')
    return path


def test_request_maps_to_make_arguments(audio, tmp_path):
    controls = chart_pb2.ChartControls(long_notes=chart_pb2.LONG_NOTE_AMOUNT_HALF, sections=[
        section(1000, 20000, star=4.5),
        section(30000, 50000, kind=chart_pb2.PATTERN_KIND_JUMPSTREAM, strength=0.7),
        section(60000, 70000, kind=chart_pb2.PATTERN_KIND_STREAM, strength=0.25),
        section(80000, 90000, kind=chart_pb2.PATTERN_KIND_JACK, strength=1.0),
        section(100000, 110000, kind=chart_pb2.PATTERN_KIND_CHORDJACK, strength=0.5),
        section(120000, 130000, kind=chart_pb2.PATTERN_KIND_CHORDSTREAM, strength=0.0)])
    a = make_arguments(request(audio, 3.5, seed=7, title='T', artist='A', controls=controls), tmp_path)
    assert a['audio'] == str(audio) and a['star'] == 3.5 and a['ln'] == 'half' and a['seed'] == 7
    assert (a['title'], a['artist']) == ('T', 'A')
    assert a['sections'] == [dict(start_ms=1000, end_ms=20000, star=4.5)]
    assert a['stream'] == ['30000,50000,0.7,jumpstream', '60000,70000,0.25', '120000,130000,0.0,chordstream']
    assert a['jack'] == ['80000,90000,1.0', '100000,110000,0.5,chordjack']
    assert a['out'].startswith(str(tmp_path)) and a['out'].endswith('.osz')


def test_defaults_and_tap_only(audio, tmp_path):
    a = make_arguments(request(audio), tmp_path)
    assert (a['ln'], a['seed'], a['title'], a['artist'], a['sections'], a['stream'], a['jack']) == (
        'natural', 0, None, None, [], [], [])
    tap = chart_pb2.ChartControls(long_notes=chart_pb2.LONG_NOTE_AMOUNT_TAP_ONLY)
    assert make_arguments(request(audio, controls=tap), tmp_path)['ln'] == 'tap'


@pytest.mark.parametrize('star', [7.0, 6.0, 1.99, float('nan')])
def test_star_outside_range_is_refused(audio, tmp_path, star):
    with pytest.raises(RequestError) as error:
        make_arguments(request(audio, star), tmp_path)
    assert error.value.code == 'star_out_of_range'


def test_bad_requests_are_refused(audio, tmp_path):
    with pytest.raises(RequestError) as error:
        make_arguments(request(tmp_path / 'missing.mp3'), tmp_path)
    assert error.value.code == 'audio_unreadable'
    uri = chart_pb2.GenerateChartRequest(audio=inference_pb2.AudioAssetRef(uri='https://x/y.mp3'), star=3)
    with pytest.raises(RequestError):
        make_arguments(uri, tmp_path)
    for bad in (section(5000, 5000, star=3.0), section(0, 9000, star=9.0),
                section(0, 9000, kind=chart_pb2.PATTERN_KIND_UNSPECIFIED, strength=0.5),
                section(0, 9000, kind=chart_pb2.PATTERN_KIND_JACK, strength=1.5),
                chart_pb2.SectionRequest(start_ms=0, end_ms=9000)):
        with pytest.raises(RequestError):
            make_arguments(request(audio, controls=chart_pb2.ChartControls(sections=[bad])), tmp_path)


def test_chart_conversion_orders_objects_and_marks_holds(tmp_path):
    osu = tmp_path / 'generated.osu'
    osu.write_text(OSU)
    chart = chart_from_osu(osu, 4000)
    assert chart.duration_ms == 4000
    assert [(o.lane, o.start_ms, o.end_ms if o.HasField('end_ms') else None) for o in chart.objects] == [
        (0, 1000, None), (2, 1000, None), (1, 2000, 2500), (0, 2600, None), (3, 3000, 3600)]


def test_capabilities_list_the_files_and_every_control(tmp_path):
    (tmp_path / 'SHA256SUMS').write_text(f'{"a" * 64}  system.json\n{"b" * 64}  rows/model.pt\n')
    c = capabilities(tmp_path)
    assert (c.model_name, c.model_version, c.min_star, c.max_star_exclusive) == ('Ensomi R2', 'r2-1.0', 2.0, 6.0)
    assert [(f.path, f.sha256_hex) for f in c.files] == [('system.json', 'a' * 64), ('rows/model.pt', 'b' * 64)]
    status = {k.name: k.status for k in c.controls}
    W, E, N = chart_pb2.CONTROL_STATUS_WORKS, chart_pb2.CONTROL_STATUS_EXPERIMENTAL, chart_pb2.CONTROL_STATUS_NOT_WORKING
    assert status == dict(star=W, jumpstream=W, chordstream=W, stream=E, jack=E, section_star=E, chordjack=N,
                          long_notes=N)


class StubSystem:
    """Writes what ``System.make`` leaves: the .osz, and a work directory with generated.osu and arrays.npz."""

    def __init__(self, gate=None):
        self.gate, self.calls = gate, []

    def make(self, audio, star, out, *, progress=None, **kwargs):
        self.calls.append(dict(audio=audio, star=star, out=out, **kwargs))
        for phase in ('timing', 'rows', 'arrangement', 'export'):
            progress(phase)
        if self.gate:
            self.gate.wait(10)
        work = Path(out).with_suffix('')
        work.mkdir()
        (work / 'generated.osu').write_text(OSU)
        np.savez(work / 'arrays.npz', song_ms=np.float64(4000.0))
        Path(out).write_bytes(b'osz')
        return dict(star=star)


class FakeSocket:
    def __init__(self):
        self.incoming, self.sent = asyncio.Queue(), []

    async def send(self, data):
        envelope = envelope_pb2.Envelope()
        envelope.ParseFromString(data)
        self.sent.append(envelope)

    def __aiter__(self):
        return self

    async def __anext__(self):
        frame = await self.incoming.get()
        if frame is None:
            raise StopAsyncIteration
        return frame

    def put(self, **payload):
        self.incoming.put_nowait(envelope_pb2.Envelope(**payload).SerializeToString())

    async def until(self, predicate):
        for _ in range(500):
            if any(predicate(e) for e in self.sent):
                return
            await asyncio.sleep(0.01)
        raise AssertionError([e.WhichOneof('payload') for e in self.sent])


@pytest.fixture
def model_dir(tmp_path):
    (tmp_path / 'model' / 'x').mkdir(parents=True)
    (tmp_path / 'model' / 'SHA256SUMS').write_text(f'{"c" * 64}  system.json\n')
    return tmp_path / 'model'


def run(model_dir, tmp_path, script, system=None):
    async def main():
        service = Service(system or StubSystem(), model_dir, tmp_path / 'cache')
        ws = FakeSocket()
        connection = asyncio.create_task(service.handle_connection(ws))
        await script(ws)
        ws.incoming.put_nowait(None)
        await connection
        await asyncio.gather(*service.tasks)
        return ws.sent
    return asyncio.run(main())


def test_round_trip_capabilities_then_a_chart(model_dir, tmp_path, audio):
    async def script(ws):
        ws.put(message_id='m1', capabilities_request=chart_pb2.CapabilitiesRequest())
        ws.put(session_id='s1', message_id='m2', generate_chart=request(audio, 3.5, seed=3))
        await ws.until(lambda e: e.WhichOneof('payload') == 'chart_result')
    system = StubSystem()
    sent = run(model_dir, tmp_path, script, system)
    assert sent[0].WhichOneof('payload') == 'model_capabilities' and sent[0].correlation_id == 'm1'
    assert sent[0].model_capabilities.files[0].sha256_hex == 'c' * 64
    chain = [(e.WhichOneof('payload'), e.generation_progress.phase) for e in sent[1:]]
    assert chain == [('generation_progress', p) for p in (
        chart_pb2.GENERATION_PHASE_LOADING, chart_pb2.GENERATION_PHASE_TIMING, chart_pb2.GENERATION_PHASE_ROWS,
        chart_pb2.GENERATION_PHASE_ARRANGEMENT, chart_pb2.GENERATION_PHASE_EXPORT)] + [('chart_result', 0)]
    assert all(e.session_id == 's1' and e.correlation_id == 'm2' for e in sent[1:])
    assert [e.sequence for e in sent] == list(range(1, len(sent) + 1))
    result = sent[-1].chart_result
    assert result.chart.duration_ms == 4000 and len(result.chart.objects) == 5
    assert (result.receipt.model_version, result.receipt.seed, result.receipt.star) == ('r2-1.0', 3, 3.5)
    assert Path(result.receipt.osz_path).read_bytes() == b'osz'
    assert not Path(system.calls[0]['out']).with_suffix('').exists()      # work directory removed
    assert system.calls[0]['seed'] == 3 and system.calls[0]['name'] == 'r2'


def test_errors_are_events(model_dir, tmp_path, audio):
    async def script(ws):
        ws.put(session_id='s1', message_id='m1', generate_chart=request(audio, 7.0))
        ws.put(session_id='s2', message_id='m2', generate_chart=request(tmp_path / 'nope.mp3'))
        ws.put(message_id='m3', generate_chart=request(audio))
        ws.put(session_id='s4', message_id='m4', stop_session=inference_pb2.StopSessionRequest(reason='x'))
        ws.incoming.put_nowait(b'\xff\xff garbage')
        await ws.until(lambda e: e.WhichOneof('payload') == 'error' and e.message_id and e.correlation_id == '')
    sent = run(model_dir, tmp_path, script)
    errors = {e.correlation_id: e.error for e in sent}
    assert errors['m1'].code == 'star_out_of_range' and '7' in errors['m1'].message
    assert errors['m2'].code == 'audio_unreadable'
    assert errors['m2'].error_code == inference_pb2.INFERENCE_ERROR_CODE_AUDIO_NOT_FOUND
    assert errors['m3'].code == 'protocol_error' and errors['m4'].code == 'session_not_found'
    assert errors[''].code == 'protocol_error'


def test_make_failures_are_events(model_dir, tmp_path, audio):
    class Failing:
        def make(self, *a, **k):
            raise {'value': ValueError('span too short'), 'other': RuntimeError('boom')}[k['title']]

    async def script(ws):
        ws.put(session_id='s1', message_id='m1', generate_chart=request(audio, title='value'))
        ws.put(session_id='s2', message_id='m2', generate_chart=request(audio, title='other'))
        await ws.until(lambda e: e.correlation_id == 'm2' and e.WhichOneof('payload') == 'error')
    errors = {e.correlation_id: e.error for e in run(model_dir, tmp_path, script, Failing()) if e.WhichOneof('payload') == 'error'}
    assert (errors['m1'].code, errors['m1'].message) == ('unsupported_control', 'span too short')
    assert errors['m2'].code == 'generation_failed' and 'boom' in errors['m2'].message


def test_stop_discards_the_result(model_dir, tmp_path, audio):
    gate = threading.Event()

    async def script(ws):
        ws.put(session_id='s1', message_id='m1', generate_chart=request(audio))
        await ws.until(lambda e: e.generation_progress.phase == chart_pb2.GENERATION_PHASE_EXPORT)
        ws.put(session_id='s1', message_id='m2', stop_session=inference_pb2.StopSessionRequest(reason='client_stop'))
        await ws.until(lambda e: e.WhichOneof('payload') == 'status')
        gate.set()
    sent = run(model_dir, tmp_path, script, StubSystem(gate))
    kinds = [e.WhichOneof('payload') for e in sent]
    assert 'chart_result' not in kinds and 'error' not in kinds
    assert sent[-1].status.status == inference_pb2.ENDPOINT_STATUS_STOPPED and sent[-1].correlation_id == 'm2'
    assert list((tmp_path / 'cache').iterdir()) == []                            # the discarded .osz is removed


def test_generations_run_one_at_a_time(model_dir, tmp_path, audio):
    gate = threading.Event()
    system = StubSystem(gate)

    async def script(ws):
        ws.put(session_id='a', message_id='ma', generate_chart=request(audio))
        ws.put(session_id='b', message_id='mb', generate_chart=request(audio))
        await ws.until(lambda e: e.correlation_id == 'ma' and e.generation_progress.phase == 5)
        await asyncio.sleep(0.1)
        assert len(system.calls) == 1 and not any(e.correlation_id == 'mb' for e in ws.sent)
        gate.set()
        await ws.until(lambda e: e.correlation_id == 'mb' and e.WhichOneof('payload') == 'chart_result')
    run(model_dir, tmp_path, script, system)
    assert len(system.calls) == 2
