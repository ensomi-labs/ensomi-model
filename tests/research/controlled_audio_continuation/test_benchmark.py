"""Measurement boundaries, configuration projection and trajectory preservation."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from ensomi_model.research.controlled_audio_continuation.benchmark import (
    measure_generation, playback_slack,
)
from ensomi_model.research.controlled_audio_continuation.benchmark_config import BenchmarkConfig
from ensomi_model.research.controlled_audio_continuation.benchmark_hydra import compose_config
from controlled_audio_continuation.test_layout_modulation import model


def test_slack_uses_previous_published_watermark_and_keeps_startup_fixed():
    windows = [dict(rows=30, coverage_ms=8000, elapsed_seconds=1., service_seconds=1.),
               dict(rows=60, coverage_ms=16000, elapsed_seconds=1.2, service_seconds=.2),
               dict(rows=90, coverage_ms=24000, elapsed_seconds=1.5, service_seconds=.3)]
    assert playback_slack(windows)['minimum_slack_seconds'] == pytest.approx(5.8)
    stalled = playback_slack(windows, stall_seconds=20.)
    assert stalled['missed'] == 1
    assert stalled['minimum_slack_seconds'] == pytest.approx(-6.5)
    windows[0]['rows'] = 29
    assert playback_slack(windows)['intervals'] == 1
    assert playback_slack(windows)['minimum_slack_seconds'] == pytest.approx(13.7)


def test_hydra_projects_overrides_and_rejects_unused_keys():
    overrides = ['checkpoint_file=x', 'checkpoint_sha256='+'a'*64,
                 'panel_file=y', 'panel_sha256='+'b'*64, 'output_dir=z',
                 'conditions=[low,ln_high,switch]', 'case_names=[one,two]',
                 'window_ms=2000', 'limit_ms=32000', 'repeats=2', 'cpu_threads=2',
                 'preprocess=true', 'profile_stages=true', 'cache_encoded=true', 'seed=51']
    config = compose_config(overrides)
    assert config == BenchmarkConfig(checkpoint_file='x', checkpoint_sha256='a'*64,
        panel_file='y', panel_sha256='b'*64, output_dir='z', conditions=['low','ln_high','switch'],
        case_names=['one','two'], window_ms=2000, limit_ms=32000, repeats=2, cpu_threads=2,
        preprocess=True, profile_stages=True, cache_encoded=True, seed=51)
    with pytest.raises(ValueError, match='Unknown or missing'):
        compose_config(overrides+['+unused=true'])
    with pytest.raises(ValueError, match='conditions'):
        replace(config, conditions=['unknown']).validate()


def test_diagnostic_and_encoded_reuse_preserve_rows_and_settled_coverage():
    torch.manual_seed(341)
    net = model(True)
    mel = np.random.default_rng(16).normal(size=(100, 128)).astype(np.float32)
    case = dict(name='synthetic', duration_ms=1000)
    config = BenchmarkConfig(window_ms=333, max_seconds=30.)
    plain, native = measure_generation(net, mel, case, 'base', config, 918)
    with torch.inference_mode():
        encoded = net.encode_audio(torch.from_numpy(mel)[None])
    cached, reused = measure_generation(net, mel, case, 'base', config, 918, encoded=encoded)
    detailed, profiled = measure_generation(net, mel, case, 'base', config, 918, profile=True)
    assert native.completed and reused.completed and profiled.completed
    assert native.rows == reused.rows == profiled.rows
    assert plain['row_sequence_sha256'] == cached['row_sequence_sha256'] == detailed['row_sequence_sha256']
    assert [w['coverage_ms'] for w in plain['windows']] == [333, 666, 999, 1000]
    assert detailed['stages']['row_neural']['n'] == len(native.rows)
    assert sum(s['seconds'] for s in detailed['stages'].values()) < detailed['generation_seconds']
    assert 'encode_audio' not in net.__dict__
