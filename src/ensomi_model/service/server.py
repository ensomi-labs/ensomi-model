"""The websocket: binary protobuf ``Envelope`` frames, one generation at a time, ``System.make`` off the event loop.

    python -m ensomi_model.service --model-dir DIR [--host 127.0.0.1] [--port 8765] [--cache-dir DIR]

Flow, as in ``chart.proto``: ``CapabilitiesRequest`` is answered with ``ModelCapabilities``. ``GenerateChartRequest``
(non-empty ``session_id``) is answered with ``GenerationProgress`` per phase and then ``ChartResult``, or with an
``ErrorEvent``. Generations queue; a second request waits for the first. ``StopSessionRequest`` discards the session's
result (a running ``make`` cannot be interrupted, so the next generation starts after it ends) and is answered with a
``StatusEvent`` stopped; the session id is free again at once. Every reply carries the request's ``message_id`` as
``correlation_id``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import time
import traceback
import uuid

from ensomi.protocol.v1 import chart_pb2, envelope_pb2, inference_pb2
from google.protobuf.message import DecodeError
from websockets.exceptions import ConnectionClosed

from . import mapping

NODE_ID = 'ensomi-model.chart'
PHASES = dict(timing=chart_pb2.GENERATION_PHASE_TIMING, rows=chart_pb2.GENERATION_PHASE_ROWS,
              arrangement=chart_pb2.GENERATION_PHASE_ARRANGEMENT, export=chart_pb2.GENERATION_PHASE_EXPORT)
PROTOCOL = inference_pb2.INFERENCE_ERROR_CODE_PROTOCOL_ERROR
ERROR_CODES = dict(audio_unreadable=inference_pb2.INFERENCE_ERROR_CODE_AUDIO_NOT_FOUND,
                   generation_failed=inference_pb2.INFERENCE_ERROR_CODE_INFERENCE_FAILED)


class Service:
    def __init__(self, system, model_dir, cache_dir):
        self.system, self.cache_dir = system, Path(cache_dir)
        self.capabilities = mapping.capabilities(model_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.queue = asyncio.Lock()
        self.tasks = set()

    async def handle_connection(self, ws):
        """``ws``: async iterable of received frames with an async ``send(bytes)``."""
        connection = Connection(ws)
        try:
            async for frame in ws:
                await self.handle_frame(connection, frame)
        finally:
            for generation in connection.sessions.values():
                generation.stopped = True

    async def handle_frame(self, connection, frame):
        envelope = envelope_pb2.Envelope()
        try:
            envelope.ParseFromString(frame)
        except DecodeError:
            return await connection.error(envelope, 'protocol_error', 'invalid protobuf envelope')
        payload = envelope.WhichOneof('payload')
        if payload == 'capabilities_request':
            await connection.send(envelope, model_capabilities=self.capabilities)
        elif payload in ('generate_chart', 'stop_session') and not envelope.session_id:
            await connection.error(envelope, 'protocol_error', f'{payload} needs a session_id')
        elif payload == 'generate_chart':
            if envelope.session_id in connection.sessions:
                return await connection.error(envelope, 'protocol_error', 'session already generating')
            generation = connection.sessions[envelope.session_id] = Generation()
            task = asyncio.get_running_loop().create_task(self.generate(connection, envelope, generation))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
        elif payload == 'stop_session':
            generation = connection.sessions.pop(envelope.session_id, None)
            if generation is None:
                return await connection.error(envelope, 'session_not_found', 'no generation in this session')
            generation.stopped = True
            await connection.send(envelope, status=inference_pb2.StatusEvent(
                status=inference_pb2.ENDPOINT_STATUS_STOPPED, reason=envelope.stop_session.reason or 'client_stop'))
        else:
            await connection.error(envelope, 'protocol_error', f'unsupported payload: {payload}')

    async def generate(self, connection, request, generation):
        loop = asyncio.get_running_loop()
        try:
            arguments = mapping.make_arguments(request.generate_chart, self.cache_dir)
            async with self.queue:
                if generation.stopped:
                    return
                await connection.send(request, generation_progress=chart_pb2.GenerationProgress(
                    phase=chart_pb2.GENERATION_PHASE_LOADING))
                phases = asyncio.Queue()

                def work():
                    try:
                        return mapping.generate(self.system, arguments,
                                                lambda p: loop.call_soon_threadsafe(phases.put_nowait, p))
                    finally:
                        loop.call_soon_threadsafe(phases.put_nowait, None)

                running = loop.run_in_executor(None, work)
                while (phase := await phases.get()) is not None:
                    if not generation.stopped:
                        await connection.send(request, generation_progress=chart_pb2.GenerationProgress(
                            phase=PHASES[phase]))
                result = await running
            if generation.stopped:
                Path(result.receipt.osz_path).unlink(missing_ok=True)
            else:
                await connection.send(request, chart_result=result)
        except mapping.RequestError as error:
            await connection.error(request, error.code, str(error))
        except ValueError as error:                     # System.make refuses a request, such as a span past the song
            await connection.error(request, 'unsupported_control', str(error))
        except Exception as error:
            traceback.print_exc()
            await connection.error(request, 'generation_failed', f'{type(error).__name__}: {error}')
        finally:
            if connection.sessions.get(request.session_id) is generation:
                del connection.sessions[request.session_id]


class Generation:
    stopped = False


class Connection:
    def __init__(self, ws):
        self.ws, self.sequence = ws, 0
        self.sessions = {}                              # session_id -> the Generation not yet finished or stopped

    async def send(self, request, **payload):
        """One reply to ``request``: ``payload`` is the single oneof field, for example ``chart_result=...``."""
        self.sequence += 1
        envelope = envelope_pb2.Envelope(session_id=request.session_id, sequence=self.sequence,
                                         sent_at_unix_ms=int(time.time() * 1000), source_node_id=NODE_ID,
                                         message_id=str(uuid.uuid4()), correlation_id=request.message_id, **payload)
        try:
            await self.ws.send(envelope.SerializeToString())
        except ConnectionClosed:
            pass

    async def error(self, request, code, message):
        await self.send(request, error=inference_pb2.ErrorEvent(
            code=code, message=message, error_code=ERROR_CODES.get(code, PROTOCOL), phase='generate_chart',
            route=inference_pb2.INFERENCE_ROUTE_MAPPER, error_kind='chart'))


async def serve(service, host, port):
    from websockets.asyncio.server import serve as websocket_serve

    async def handler(ws):
        try:
            await service.handle_connection(ws)
        except ConnectionClosed:
            pass

    async with websocket_serve(handler, host, port, max_size=1 << 24):
        print(f'ensomi-model service on ws://{host}:{port}', flush=True)
        await asyncio.Future()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--model-dir', required=True, help='a model directory: system.json, SHA256SUMS and the weights')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--cache-dir', default='ensomi-service-cache', help='where the .osz files are written')
    p.add_argument('--threads', type=int, default=1)
    a = p.parse_args(argv)
    import torch
    from ..system.make import System
    torch.set_num_threads(a.threads)
    system = System.load(json.loads((Path(a.model_dir) / 'system.json').read_text()), a.model_dir)
    asyncio.run(serve(Service(system, a.model_dir, a.cache_dir), a.host, a.port))


if __name__ == '__main__':
    main()
