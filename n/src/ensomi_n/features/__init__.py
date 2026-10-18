"""Audio features for N models: decoding on osu!'s clock, the log-Mel and its cache.

- `clock`: the MP3 start classes and the lead rule (``osu-lead-v1``);
- `decode`: libsndfile (ffmpeg as fallback) to mono 24 kHz on osu!'s clock;
- `mel`: the 128-band, 10 ms log-Mel; frame ``i`` sits at ``10 i`` ms;
- `cache`: float16 arrays keyed by audio sha256 and the configuration hash;
- `fixtures`: encoded click tracks that check the clock absolutely.

Only `clock` and `mel` are numpy-only; decoding needs soundfile and torchaudio.
"""
from .clock import LEAD_RULE, Mp3Start, file_lead, lead_samples, read_mp3_start
from .mel import DEFAULT, MelConfig, log_mel

__all__ = ['LEAD_RULE', 'Mp3Start', 'file_lead', 'lead_samples', 'read_mp3_start', 'DEFAULT', 'MelConfig', 'log_mel']
