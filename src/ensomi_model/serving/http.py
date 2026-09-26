"""Loopback JSON transport for confirmed H/R/R1 publication windows."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import shutil
from .runtime import DemoRuntime, ServiceError


def handler_for(runtime):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def send_json(self, code, value):
            data = json.dumps(value, allow_nan=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def dispatch(self):
            path = self.path.strip('/').split('/')
            if self.command == 'GET' and path == ['health']:
                return runtime.health()
            if self.command == 'DELETE' and len(path) == 2 and path[0] == 'sessions':
                return runtime.cancel(path[1])
            if self.command != 'POST':
                raise ServiceError('Unknown endpoint', 404)
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                raise ServiceError('Expected application/json', 415)
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 32768:
                raise ServiceError('Request body size is invalid', 413)
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ServiceError('Expected a JSON object')
            if path == ['sessions']:
                return runtime.create(body)
            if len(path) == 3 and path[0] == 'sessions' and path[2] == 'window':
                return runtime.window(path[1], body)
            raise ServiceError('Unknown endpoint', 404)

        def handle_request(self):
            try:
                self.send_json(200, self.dispatch())
            except ServiceError as error:
                self.send_json(error.status, dict(error=str(error)))
            except (ValueError, KeyError, TypeError) as error:
                self.send_json(400, dict(error=str(error)))
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as error:
                self.send_json(500, dict(error=f'Generation failed: {error}'))

        do_GET = do_POST = do_DELETE = handle_request
    return Handler


def serve(config):
    if not shutil.which('ffmpeg'):
        raise RuntimeError('FFmpeg is required to decode reference audio')
    runtime = DemoRuntime(config)
    server = ThreadingHTTPServer((config.host, config.port), handler_for(runtime))
    print(json.dumps(dict(listening=f'http://{config.host}:{config.port}', **runtime.health())), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for identity in list(runtime.sessions):
            runtime.cancel(identity)
        server.server_close()
