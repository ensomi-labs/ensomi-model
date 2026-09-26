"""Settings for a bounded, local controlled-generation performance panel."""
from dataclasses import dataclass, field
import math


CONDITIONS = ('unspecified', 'base', 'low', 'high', 'ln_high', 'jack', 'stream',
              'trill', 'tech', 'switch')


@dataclass
class BenchmarkConfig:
    checkpoint_file: str = ''
    checkpoint_sha256: str = ''
    panel_file: str = ''
    panel_sha256: str = ''
    output_dir: str = ''
    case_names: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=lambda: ['base'])
    device: str = 'cpu'
    cpu_threads: int = 1
    repeats: int = 3
    seed: int = 261900
    window_ms: int = 8000
    limit_ms: int = 0
    max_seconds: float = 180.
    preprocess: bool = False
    profile_stages: bool = False
    cache_encoded: bool = False

    def validate(self):
        for name in ('checkpoint_file', 'panel_file', 'output_dir'):
            if not getattr(self, name).strip():
                raise ValueError(f'{name} is required')
        for name in ('checkpoint_sha256', 'panel_sha256'):
            value = getattr(self, name)
            if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError(f'{name} requires a lowercase SHA-256')
        if self.device not in ('cpu', 'mps'):
            raise ValueError('device must be cpu or mps')
        for name in ('cpu_threads', 'repeats', 'window_ms'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.limit_ms) is not int or self.limit_ms < 0:
            raise ValueError('limit_ms must be a nonnegative integer; zero means complete audio')
        if type(self.seed) is not int or not 0 <= self.seed < 2**62:
            raise ValueError('seed must be an integer in [0, 2**62)')
        if not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError('max_seconds must be finite and positive')
        if not self.conditions or len(set(self.conditions)) != len(self.conditions):
            raise ValueError('conditions must be nonempty and unique')
        if set(self.conditions) - set(CONDITIONS):
            raise ValueError(f'conditions must be drawn from {CONDITIONS}')
        if len(set(self.case_names)) != len(self.case_names):
            raise ValueError('case_names must be unique')
        for name in ('preprocess', 'profile_stages', 'cache_encoded'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be boolean')
