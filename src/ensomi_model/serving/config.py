"""Process settings for the local playable demo."""
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DemoServiceConfig:
    checkpoint_file: str = ''
    checkpoint_sha256: str = ''
    host: str = '127.0.0.1'
    port: int = 8766
    cache_dir: str = 'artifacts/cache/controlled-demo'
    cache_mib: int = 256
    max_sessions: int = 2
    session_idle_seconds: int = 120
    request_budget_seconds: int = 60

    def validate(self):
        if not self.checkpoint_file or not Path(self.checkpoint_file).is_file():
            raise ValueError('checkpoint_file must name a controlled-audio checkpoint')
        if len(self.checkpoint_sha256) != 64 or any(c not in '0123456789abcdef' for c in self.checkpoint_sha256):
            raise ValueError('checkpoint_sha256 must be a lowercase SHA-256')
        if self.host not in ('127.0.0.1', 'localhost'):
            raise ValueError('The local demo service binds only to loopback')
        for name in ('port', 'cache_mib', 'max_sessions', 'session_idle_seconds', 'request_budget_seconds'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.port > 65535 or not self.cache_dir:
            raise ValueError('Invalid port or cache directory')
