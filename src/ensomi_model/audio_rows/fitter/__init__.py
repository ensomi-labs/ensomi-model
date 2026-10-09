"""The legacy BeatThis grid fitter, cut to the call the audio-rows path makes.

Source: ``src/ensomi_model/timing/grid_fitting`` at ``5c56e28`` (branch ``legacy/v2``), run with its
default ``GridFitterConfig``: tempo-alias canonicalisation on, no 80/160 BPM folding. Kept modules
and private function names match the source so each file diffs against it. Removed: config
validation, the fit diagnostics and ramp detection (they never changed the segments), branches
the default config never takes, and the ``FrameTimingPrediction`` wrapper. On the same beat and
downbeat probabilities the segments equal the source's.

``peaks.refine_segments`` is new (2026-10-09): it refits each segment's offset and beat length to
BeatThis's sub-frame peaks after this fit, which the audio-rows path applies in ``audio.Listener``.
"""
from __future__ import annotations

import numpy as np

from .alias import _canonicalize_tempo_aliases
from .config import GridFitterConfig, _effective_config_for_prediction
from .refinement import _refine_timing_segments
from .scoring import _candidate_period_frame_bounds
from .segment_fit import _fit_segment_range
from .segments import _timing_segments_from_fits
from .splitting import _split_segment_range
from .types import TimingSegment

__all__ = ['GridFitterConfig', 'TimingSegment', 'fit_segments']


def fit_segments(beat_prob, downbeat_prob, frame_rate_hz: float,
                 config: GridFitterConfig = GridFitterConfig()) -> tuple[TimingSegment, ...]:
    """Tempo segments, in increasing offset, from frame beat and downbeat probabilities.

    Raises ValueError when the input is shorter than one period at ``config.min_bpm`` (3 s at the
    defaults) or the beat probability is constant.
    """
    signal = np.asarray(beat_prob, dtype=np.float64)
    downbeat_signal = np.asarray(downbeat_prob, dtype=np.float64)
    frame_count = signal.shape[0]
    config = _effective_config_for_prediction(frame_count, frame_rate_hz=frame_rate_hz, config=config)
    if frame_count < _candidate_period_frame_bounds(frame_rate_hz, config=config)[1]:
        raise ValueError(f'{frame_count} frames are too short to fit a beat grid')
    if float(np.linalg.norm(signal - float(np.mean(signal)))) == 0.0:
        raise ValueError('beat_prob contains no beat signal')

    frame_times_ms = np.arange(frame_count, dtype=np.float64) / frame_rate_hz * 1000.0
    initial_fit = _fit_segment_range(signal, frame_times_ms=frame_times_ms, downbeat_signal=downbeat_signal,
                                     start_frame=0, end_frame=frame_count, config=config)
    segment_fits = _split_segment_range(signal, frame_times_ms=frame_times_ms, downbeat_signal=downbeat_signal,
                                        fit=initial_fit, config=config, remaining_splits=config.max_segments - 1)
    segment_fits = _canonicalize_tempo_aliases(segment_fits, signal, frame_times_ms=frame_times_ms,
                                               downbeat_signal=downbeat_signal, config=config)
    return _refine_timing_segments(_timing_segments_from_fits(segment_fits, frame_times_ms, config=config),
                                   frame_times_ms, beat_signal=signal, config=config)
