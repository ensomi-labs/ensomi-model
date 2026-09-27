"""Typed, Torch-free startup settings for native gameplay qualification."""
from dataclasses import dataclass
import math


@dataclass
class QualificationConfig:
    checkpoint_file: str = ''
    checkpoint_sha256: str = ''
    plan_file: str = ''
    plan_sha256: str = ''
    output_dir: str = ''
    envelope_file: str | None = None
    envelope_sha256: str = ''
    ln_feedback: bool = True
    cpu_threads: int = 1
    max_seconds: float = 1800.
    max_case_seconds: float = 300.
    footprint_limit_bytes: int = 8 * 1024**3
    lead_ms: int = 2000
    service_ms: int = 2000
    startup_seconds_limit: float = 2.
    service_seconds_limit: float = 2.
    difficulty_error_limit: float = 1.
    ln_fraction_error_limit: float = .1

    def validate(self):
        for name in ('checkpoint_file','plan_file','output_dir'):
            if not isinstance(getattr(self,name),str) or not getattr(self,name).strip():
                raise ValueError(f'Qualification requires {name}')
        for name in ('checkpoint_sha256','plan_sha256','envelope_sha256'):
            value=getattr(self,name)
            if name=='envelope_sha256' and self.envelope_file is None and value=='':continue
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError(f'{name} must be a lowercase SHA-256')
        if bool(self.envelope_file)!=bool(self.envelope_sha256):
            raise ValueError('Envelope file and identity must be supplied together')
        if type(self.ln_feedback) is not bool:raise ValueError('ln_feedback must be boolean')
        for name in ('cpu_threads','footprint_limit_bytes','lead_ms','service_ms'):
            if type(getattr(self,name)) is not int or getattr(self,name)<=0:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('max_seconds','max_case_seconds','startup_seconds_limit','service_seconds_limit',
                     'difficulty_error_limit','ln_fraction_error_limit'):
            value=getattr(self,name)
            if isinstance(value,bool) or not math.isfinite(value) or value<=0:
                raise ValueError(f'{name} must be finite and positive')
