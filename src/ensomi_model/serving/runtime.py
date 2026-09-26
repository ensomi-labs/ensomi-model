"""Serialized native generation, immutable audio caching and owned session state."""
from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
import threading
import time

import numpy as np
import torch

from ..features.audio import load_audio_file
from ..features.mel_base import MUSIC_MEL_CACHE_CONFIG, compute_log_mel_10ms
from ..research.joint_audio_continuation.data import digest, frontend_identity
from ..research.controlled_audio_continuation.model import load_model
from ..research.controlled_audio_continuation.generation import ControlledSession
from ..research.typed_audio_continuation.controls import ControlSchedule, ControlSpan

PROTOCOL = 'ensomi-demo/v1'
WINDOW_MS = 1000

class ServiceError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status

@dataclass
class AudioFeatures:
    mel: np.ndarray
    encoded: torch.Tensor
    duration_ms: int

    @property
    def size(self):
        return self.mel.nbytes + self.encoded.numel() * self.encoded.element_size()

@dataclass
class Session:
    cancelled: threading.Event = field(default_factory=threading.Event)
    used: float = field(default_factory=time.monotonic)
    generator: object = None
    sequence: int = 0
    last_window: dict | None = None


def number(value, name, low, high, *, optional=False):
    if optional and value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ServiceError(f'{name} must be a finite number in [{low}, {high}]')
    return value


def request_controls(value, style_names, duration_ms):
    if not isinstance(value, dict) or set(value) - {'difficulty', 'ln_fraction', 'style'}:
        raise ServiceError('Unknown control fields')
    difficulty = number(value.get('difficulty'), 'difficulty', 1., 8., optional=True)
    fraction = number(value.get('ln_fraction'), 'ln_fraction', 0., 1., optional=True)
    style = value.get('style', {})
    if not isinstance(style, dict) or set(style) - set(style_names):
        raise ServiceError('Style is absent from the loaded model vocabulary')
    style = {k: number(v, k, -1., 1.) for k, v in style.items()}
    return ControlSchedule((ControlSpan(0, duration_ms+1, stars=difficulty,
        ln_fraction=fraction, style=style),), style_names)


class DemoRuntime:
    def __init__(self, config):
        config.validate()
        if digest(config.checkpoint_file) != config.checkpoint_sha256:
            raise ServiceError('Checkpoint SHA-256 mismatch')
        torch.set_num_threads(1)
        self.config = config
        self.model = load_model(config.checkpoint_file, device='cpu')
        if self.model.config.profile_count:
            raise ServiceError('The demo requires an unprofiled controlled model')
        self.work_lock, self.registry_lock = threading.Lock(), threading.Lock()
        self.sessions, self.audio = {}, OrderedDict()
        frontend = str(frontend_identity()).encode()
        self.cache_root = Path(config.cache_dir) / hashlib.sha256(frontend).hexdigest()[:16]
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def health(self):
        with self.registry_lock:
            self._prune()
            active = len(self.sessions)
        return dict(protocol=PROTOCOL, ready=True, style_names=list(self.model.style_names),
                    checkpoint_sha256=self.config.checkpoint_sha256, window_ms=WINDOW_MS, active_sessions=active)

    def _prune(self):
        now = time.monotonic()
        for identity, session in list(self.sessions.items()):
            if now-session.used > self.config.session_idle_seconds:
                session.cancelled.set()
                del self.sessions[identity]

    def cancel(self, identity):
        with self.registry_lock:
            session = self.sessions.pop(identity, None)
            if session is not None:
                session.cancelled.set()
        return dict(cancelled=True)

    def _session(self, identity):
        with self.registry_lock:
            self._prune()
            session = self.sessions.get(identity)
            if session is None or session.cancelled.is_set():
                raise ServiceError('Session expired or cancelled; start again', 404)
            session.used = time.monotonic()
            return session

    @torch.inference_mode()
    def _features(self, path):
        identity = digest(path)
        if identity in self.audio:
            self.audio.move_to_end(identity)
            return self.audio[identity], True
        cached = self.cache_root / (identity+'.npz')
        mel = None
        if cached.exists():
            try:
                with np.load(cached, allow_pickle=False) as data:
                    mel, samples = data['mel'], int(data['samples'])
                if (samples <= 0 or mel.shape != ((samples+239)//240, 128) or
                        mel.dtype != np.float32 or not np.isfinite(mel).all()):
                    mel = None
            except (ValueError, OSError, KeyError):
                mel = None
        if mel is None:
            wave = load_audio_file(path, MUSIC_MEL_CACHE_CONFIG.sample_rate)
            if not len(wave) or not np.isfinite(wave).all():
                raise ServiceError('Audio must decode to a nonempty finite waveform')
            samples = len(wave)
            mel = compute_log_mel_10ms(wave, sample_rate=MUSIC_MEL_CACHE_CONFIG.sample_rate,
                                      config=MUSIC_MEL_CACHE_CONFIG)
            if digest(path) != identity:
                raise ServiceError('Audio changed during decoding; retry')
            temporary = cached.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez(stream, mel=mel, samples=samples)
            temporary.replace(cached)
        encoded = self.model.encode_audio(torch.from_numpy(mel)[None])
        result = AudioFeatures(mel, encoded, samples*1000//24000)
        self.audio[identity] = result
        while self.audio and sum(v.size for v in self.audio.values()) > self.config.cache_mib*1024**2:
            self.audio.popitem(last=False)
        return result, False

    def create(self, body):
        if set(body) - {'session_id', 'audio_path', 'controls', 'seed'}:
            raise ServiceError('Unknown session fields')
        identity = body.get('session_id')
        if not isinstance(identity, str) or not 1 <= len(identity) <= 80 or not all(c.isalnum() or c == '-' for c in identity):
            raise ServiceError('session_id must be a client-generated identifier')
        path = body.get('audio_path')
        if not isinstance(path, str) or not Path(path).is_absolute() or not Path(path).is_file():
            raise ServiceError('audio_path must be an accessible absolute local file')
        seed = body.get('seed', 17)
        if type(seed) is not int or not 0 <= seed < 2**62:
            raise ServiceError('Invalid generation seed')
        request_controls(body.get('controls', {}), self.model.style_names, 1)
        with self.registry_lock:
            self._prune()
            if identity in self.sessions:
                raise ServiceError('Session already exists', 409)
            if len(self.sessions) >= self.config.max_sessions:
                raise ServiceError('Generation service is busy; stop another session first', 503)
            session = self.sessions[identity] = Session()
        try:
            with self.work_lock, torch.inference_mode():
                if session.cancelled.is_set():
                    raise ServiceError('Session cancelled', 410)
                features, hit = self._features(Path(path))
                if session.cancelled.is_set():
                    raise ServiceError('Session cancelled', 410)
                controls = request_controls(body.get('controls', {}), self.model.style_names, features.duration_ms)
                session.generator = ControlledSession(self.model, features.mel, features.duration_ms,
                    controls, seed=seed, encoded_audio=features.encoded,
                    max_seconds=self.config.request_budget_seconds)
                session.generator.stop_callback = lambda: 'cancelled' if session.cancelled.is_set() else None
            return dict(protocol=PROTOCOL, session_id=identity, duration_ms=features.duration_ms, cache_hit=hit)
        except Exception:
            self.cancel(identity)
            raise

    def window(self, identity, body):
        if set(body) != {'after_sequence', 'through_ms'}:
            raise ServiceError('Window requests require after_sequence and through_ms')
        sequence = body['after_sequence']
        if type(sequence) is not int or sequence < 0:
            raise ServiceError('after_sequence must be a nonnegative integer')
        target = number(body['through_ms'], 'through_ms', 0, 86400000)
        session = self._session(identity)
        with self.work_lock, torch.inference_mode():
            if session.cancelled.is_set() or session.generator is None:
                raise ServiceError('Session is not available', 409)
            if sequence == session.sequence-1 and session.last_window is not None:
                return session.last_window
            if sequence != session.sequence:
                raise ServiceError('Window sequence does not match the session', 409)
            generator = session.generator
            end = min(math.ceil(target), generator.duration_ms)
            if sequence:
                end = min(end, generator.coverage+WINDOW_MS)
            before = len(generator.rows)
            # The research guard is per compute request; playback idle time is not compute time.
            generator.started = time.perf_counter()
            try:
                while generator.coverage < end:
                    generator.publish_to(min(end, max(0, generator.coverage)+WINDOW_MS))
            except Exception:
                self.cancel(identity)
                raise
            completed = generator.coverage == generator.duration_ms and not any(generator.replay.occupancy)
            session.sequence += 1
            session.used = time.monotonic()
            session.last_window = dict(protocol=PROTOCOL, session_id=identity, sequence=session.sequence,
                coverage_ms=max(0, generator.coverage), end_of_stream=completed,
                rows=[dict(time_ms=r.time_ms, actions=list(r.actions)) for r in generator.rows[before:]])
            return session.last_window
