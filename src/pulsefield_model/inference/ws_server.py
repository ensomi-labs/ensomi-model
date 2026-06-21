from __future__ import annotations

import asyncio
import json
import traceback
from collections.abc import Mapping, Sequence
from typing import Any

import hydra
from omegaconf import DictConfig

from pulsefield_model.cli.configs import WsServerInferenceConfig
from pulsefield_model.cli.hydra_utils import compose_cli_config, to_config_object
from pulsefield_model.data.control_windows import normalize_difficulty
from pulsefield_model.inference.errors import (
    PeerDisconnected,
    ProtocolError,
    is_expected_socket_disconnect,
)
from pulsefield_model.inference.protocol_adapter import PulsefieldProtocolAdapter
from pulsefield_model.inference.service_models import (
    AudioCommand,
    ErrorEvent,
    ServiceCommand,
    ServiceEvent,
    StopCommand,
)
from pulsefield_model.inference.ws_framing import (
    accept_websocket_handshake,
    close_writer,
    drain_writer,
    encode_server_binary_frame,
    read_client_binary_frame,
    send_http_error,
)
from pulsefield_model.inference.ws_endpoint import (
    PULSEFIELD_WS_URL,
    InferenceEndpoint,
    InferenceError,
    WsEndpointConfig,
)
from pulsefield_model.timing.canonicalization import TIMING_CANONICALIZATION_CHOICES


_CONFIG_NAME = "inference/ws_server"


async def serve_forever(endpoint: InferenceEndpoint | None = None) -> None:
    config = WsEndpointConfig()
    endpoint = InferenceEndpoint(config=config) if endpoint is None else endpoint
    server = await asyncio.start_server(
        lambda reader, writer: _handle_websocket_client(endpoint, reader, writer),
        host=endpoint.config.host,
        port=endpoint.config.port,
    )
    sockets = ", ".join(str(sock.getsockname()) for sock in server.sockets or ())
    print(f"ws_server listening on {PULSEFIELD_WS_URL} ({sockets})", flush=True)
    async with server:
        await server.serve_forever()


async def _handle_websocket_client(
    endpoint: InferenceEndpoint,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    peer = _WebSocketPeer(writer)
    owner = object()
    owned_session_ids: set[str] = set()
    try:
        try:
            await accept_websocket_handshake(reader, writer)
        except ProtocolError as exc:
            await send_http_error(writer, status=400, reason="Bad Request", body=str(exc))
            return
        while True:
            try:
                payload = await read_client_binary_frame(reader, writer)
            except ProtocolError as exc:
                await peer.send_event(ErrorEvent(code="protocol_error", message=str(exc)))
                break
            if payload is None:
                break
            try:
                command = peer.decode_inbound_frame(payload)
                await endpoint.handle_command(command, peer, owner=owner)
                _reset_peer_stream_if_needed(peer, command)
                _track_owned_session(
                    owned_session_ids,
                    command=command,
                )
            except ProtocolError as exc:
                await peer.send_event(ErrorEvent(code="protocol_error", message=str(exc)))
            except InferenceError as exc:
                _log_inference_error(exc)
                try:
                    await peer.send_event(exc.to_service_event())
                except PeerDisconnected:
                    return
            except PeerDisconnected:
                return
    except Exception as exc:
        if not is_expected_socket_disconnect(exc):
            raise
    finally:
        try:
            await _stop_owned_sessions(endpoint, owned_session_ids, owner=owner)
        finally:
            await close_writer(writer)


class _WebSocketPeer:
    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self._writer = writer
        self._send_lock = asyncio.Lock()
        self._protocol_adapter = PulsefieldProtocolAdapter()

    def decode_inbound_frame(self, payload: bytes) -> ServiceCommand:
        return self._protocol_adapter.decode_inbound_frame(payload)

    def reset_session_stream(self, session_id: str) -> None:
        self._protocol_adapter.reset_session_stream(session_id)

    async def send_event(self, payload: ServiceEvent | Mapping[str, Any]) -> None:
        async with self._send_lock:
            try:
                for frame_payload in self._protocol_adapter.serialize_outbound_event(payload):
                    self._writer.write(encode_server_binary_frame(frame_payload))
                await drain_writer(self._writer)
            except Exception as exc:
                if is_expected_socket_disconnect(exc):
                    raise PeerDisconnected("websocket peer disconnected") from exc
                raise


def _log_inference_error(exc: InferenceError) -> None:
    print(
        "ws_inference_error "
        + json.dumps(
            {
                "session_id": exc.session_id,
                "phase": exc.phase,
                "route": exc.route,
                "code": exc.code,
                "message": str(exc),
            },
            separators=(",", ":"),
            ensure_ascii=False,
        ),
        flush=True,
    )
    traceback.print_exception(type(exc), exc, exc.__traceback__)


def _track_owned_session(
    owned_session_ids: set[str],
    *,
    command: ServiceCommand,
) -> None:
    if isinstance(command, AudioCommand):
        owned_session_ids.add(command.session_id)
        return
    if isinstance(command, StopCommand):
        owned_session_ids.discard(command.session_id)


def _reset_peer_stream_if_needed(peer: _WebSocketPeer, command: ServiceCommand) -> None:
    if isinstance(command, (AudioCommand, StopCommand)):
        peer.reset_session_stream(command.session_id)


async def _stop_owned_sessions(
    endpoint: InferenceEndpoint,
    owned_session_ids: set[str],
    *,
    owner: object,
) -> None:
    for session_id in tuple(owned_session_ids):
        await endpoint.stop_session(
            session_id,
            reason="peer_disconnect",
            owner=owner,
            transition="peer_disconnect",
        )
    owned_session_ids.clear()


@hydra.main(version_base=None, config_path="../conf", config_name=_CONFIG_NAME)
def _hydra_main(config: DictConfig) -> int:
    return run_ws_server_from_config(config)


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        return _hydra_main()
    return run_ws_server_from_config(compose_cli_config(_CONFIG_NAME, argv))


def run_ws_server_from_config(config: WsServerInferenceConfig | DictConfig) -> int:
    cfg = _normalize_ws_server_inference_config(config)
    endpoint_config = ws_endpoint_config_from_inference_config(cfg)
    asyncio.run(serve_forever(InferenceEndpoint(config=endpoint_config)))
    return 0


def ws_endpoint_config_from_inference_config(config: WsServerInferenceConfig) -> WsEndpointConfig:
    return WsEndpointConfig(
        host=config.host,
        port=int(config.port),
        mapper_checkpoint_path=config.mapper_checkpoint_path,
        control_checkpoint_path=config.control_checkpoint_path,
        device=config.device,
        beatthis_device=config.beatthis_device,
        canonicalization=config.canonicalization,
        default_difficulty=float(config.difficulty),
        max_tokens=int(config.max_tokens),
    )


def _normalize_ws_server_inference_config(config: WsServerInferenceConfig | DictConfig) -> WsServerInferenceConfig:
    cfg = to_config_object(config, WsServerInferenceConfig)
    _validate_ws_server_inference_config(cfg)
    return cfg


def _validate_ws_server_inference_config(config: WsServerInferenceConfig) -> None:
    if int(config.port) <= 0:
        raise ValueError("port must be positive")
    if int(config.max_tokens) <= 0:
        raise ValueError("max_tokens must be positive")
    if config.canonicalization not in TIMING_CANONICALIZATION_CHOICES:
        choices = ", ".join(TIMING_CANONICALIZATION_CHOICES)
        raise ValueError(f"canonicalization must be one of: {choices}")
    normalize_difficulty(float(config.difficulty))


if __name__ == "__main__":
    raise SystemExit(main())
