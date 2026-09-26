"""Full-audio latency and settled-window traces for the H/R/R1 research runtime.

Primary runs synchronize only at publication boundaries. Separate diagnostic
runs synchronize each timed stage and must reproduce the primary row sequence.
The encoded-audio reuse probe patches only a private, serial benchmark model;
it is not a concurrent service cache. No generated state is shared across runs.
"""
from collections import defaultdict
from contextlib import ExitStack, nullcontext
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import time
from unittest.mock import patch

import numpy as np
import psutil
import torch

from ...features.audio import load_audio_file
from ...features.mel_base import MUSIC_MEL_CACHE_CONFIG, compute_log_mel_10ms
from ..joint_audio_continuation.data import digest, frontend_identity
from ..joint_audio_continuation.generation import NativeGeneration, _synchronize, save_rollout
from ..planned_audio_continuation import session as session_module
from ..planned_audio_continuation.generation import HeadPlanner
from ..typed_audio_continuation.controls import ControlSchedule, ControlSpan
from .generation import ControlledSession
from .model import load_model


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return dict(n=0, median=None, p95=None, maximum=None)
    return dict(n=len(values), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), maximum=float(values.max()))


def controls_for(model, duration_ms, condition):
    if condition == 'unspecified':
        return ControlSchedule((), model.style_names)
    stars = {'low': 1.5, 'high': 6.}.get(condition, 3.)
    styles = {'jack': 'jack-organization', 'stream': 'stream-organization',
              'trill': 'trill-organization', 'tech': 'tech'}
    style = {styles[condition]: 1.} if condition in styles else {}
    return ControlSchedule((ControlSpan(0, duration_ms + 1, stars=stars,
        ln_fraction=.7 if condition == 'ln_high' else .2, style=style),), model.style_names)


def playback_slack(windows, *, lead_seconds=2., slowdown=1., stall_seconds=0.):
    """Replay publication times; readiness requires 30 rows and 8 s of coverage.

    A single optional stall precedes the slowest post-start window. Slowdown
    scales generation and startup together. These traces omit transport/rendering.
    """
    ready = next((i for i, w in enumerate(windows)
                  if w['rows'] >= 30 and w['coverage_ms'] >= 8000), None)
    if ready is None or ready == len(windows) - 1:
        return dict(intervals=0, minimum_slack_seconds=None, missed=0)
    worst = max(range(ready + 1, len(windows)), key=lambda i: windows[i]['service_seconds'])
    start = windows[ready]['elapsed_seconds'] * slowdown
    slack = [windows[i-1]['coverage_ms'] / 1000 -
             (windows[i]['elapsed_seconds'] * slowdown + (stall_seconds if i >= worst else 0) - start) -
             lead_seconds for i in range(ready+1, len(windows))]
    return dict(intervals=len(slack), minimum_slack_seconds=min(slack),
                missed=sum(s < 0 for s in slack))


class StageTimers(ExitStack):
    """Nonoverlapping named calls; unassigned work remains a measured residual."""
    def __init__(self, model):
        super().__init__()
        self.device = next(model.parameters()).device
        self.samples = defaultdict(list)
        targets = [
            (HeadPlanner, 'fill', 'head_plan'),
            (model, 'release_logits', 'release_neural'),
            (ControlledSession, 'release_window', 'release_support'),
            (model, 'planned_row_log_probs', 'row_neural'),
            (ControlledSession, 'prefer_rows', 'row_policy'),
            (model.temporal, 'append', 'row_history_append'),
            (model.skeleton_temporal, 'append', 'skeleton_history_append'),
            (session_module, 'commit', 'commit'),
        ]
        targets += [(session_module, name, 'release_support') for name in
                    ('release_clocks', 'release_masks', 'conditioned_release_logits')]
        targets += [(session_module, name, 'row_support') for name in
                    ('row_support', 'preview_features', 'consequences', 'spaced_rows')]
        for owner, name, stage in targets:
            self.enter_context(patch.object(owner, name, self.wrap(getattr(owner, name), stage)))

    def wrap(self, function, stage):
        def timed(*args, **kwargs):
            _synchronize(self.device)
            started = time.perf_counter()
            value = function(*args, **kwargs)
            _synchronize(self.device)
            self.samples[stage].append(time.perf_counter() - started)
            return value
        return timed

    def report(self):
        return {name: dict(seconds=sum(values), **distribution(values))
                for name, values in self.samples.items()}


def memory_snapshot(device):
    result = dict(process_rss_bytes=psutil.Process().memory_info().rss,
                  system_available_bytes=psutil.virtual_memory().available,
                  system_swap_used_bytes=psutil.swap_memory().used)
    if device == 'mps':
        result.update(mps_active_bytes=torch.mps.current_allocated_memory(),
                      mps_driver_bytes=torch.mps.driver_allocated_memory())
    return result


@torch.inference_mode()
def measure_generation(model, mel, case, condition, config, seed, *, encoded=None, profile=False):
    duration = case['duration_ms']
    stop = min(duration, config.limit_ms or duration)
    controls = controls_for(model, duration, condition)
    if encoded is not None and model.config.profile_count:
        raise ValueError('The encoded reuse probe requires an unprofiled controlled model')
    reuse = patch.object(model, 'encode_audio', lambda value: encoded) if encoded is not None else nullcontext()
    with reuse, StageTimers(model) if profile else nullcontext() as timer:
        _synchronize(next(model.parameters()).device)
        started = time.perf_counter()
        session = ControlledSession(model, mel, duration, controls, seed=seed, max_seconds=config.max_seconds)
        windows, first30, ready, update_seconds = [], None, None, None
        switched = False
        while session.coverage < stop:
            end = min(stop, max(0, session.coverage) + config.window_ms)
            if condition == 'switch' and not switched:
                end = min(end, 63999)
            tick = time.perf_counter()
            while session.coverage < end:
                session.step(stop_at=end)
                if first30 is None and len(session.rows) >= 30:
                    _synchronize(session.device)
                    first30 = time.perf_counter() - started
            _synchronize(session.device)
            service = time.perf_counter() - tick
            elapsed = time.perf_counter() - started
            windows.append(dict(coverage_ms=session.coverage, rows=len(session.rows),
                                service_seconds=service, elapsed_seconds=elapsed))
            if ready is None and session.coverage >= 8000 and len(session.rows) >= 30:
                ready = elapsed
            if condition == 'switch' and not switched and session.coverage == 63999:
                prefix = tuple(session.rows)
                tick = time.perf_counter()
                session.update_controls(ControlSpan(64000, 96000, stars=4.5, ln_fraction=.6,
                    style={'trill-organization': 1.}))
                _synchronize(session.device)
                update_seconds = time.perf_counter() - tick
                if tuple(session.rows) != prefix:
                    raise AssertionError('Control update changed the published prefix')
                switched = True
        _synchronize(session.device)
        seconds = time.perf_counter() - started
    rows = tuple(session.rows)
    complete = session.coverage == duration and not any(session.replay.occupancy)
    heads = sum(any(a in (1, 2) for a in row.actions) for row in rows)
    head_objects = sum(a in (1, 2) for row in rows for a in row.actions)
    ln_objects = sum(a == 2 for row in rows for a in row.actions)
    metrics = dict(case=case['name'], condition=condition, seed=seed, duration_ms=duration,
        measured_coverage_ms=session.coverage, completed=complete, rows=len(rows), head_rows=heads,
        release_rows=len(rows)-heads, head_objects=head_objects, ln_objects=ln_objects,
        generation_seconds=seconds, audio_encode_seconds=session.audio_seconds,
        ready_seconds=ready, first30_seconds=first30, control_update_seconds=update_seconds,
        rows_per_second=len(rows)/seconds, rtf=seconds/(stop/1000),
        windows=windows, window_service_seconds=distribution([w['service_seconds'] for w in windows]),
        stages=timer.report() if timer is not None else None,
        memory_end=memory_snapshot(config.device),
        row_sequence_sha256=hashlib.sha256(json.dumps([asdict(row) for row in rows]).encode()).hexdigest(),
        head_sequence_sha256=hashlib.sha256(json.dumps(session.planner.generated).encode()).hexdigest(),
        controls=[asdict(span) for span in session.controls.spans],
        virtual_playback={str(k): playback_slack(windows, slowdown=k) for k in (1., 4., 10.)},
        virtual_one_second_stall=playback_slack(windows, stall_seconds=1.),
        forced_deadline_releases=session.deadline_events)
    return metrics, NativeGeneration(rows, complete, 'complete' if complete else 'benchmark_prefix',
                                      session.coverage, {})


def preprocess_case(case, repeats):
    records = []
    for repeat in range(repeats):
        tick = time.perf_counter()
        wave = load_audio_file(case['audio_file'], MUSIC_MEL_CACHE_CONFIG.sample_rate)
        decoded = time.perf_counter() - tick
        if len(wave)*1000//MUSIC_MEL_CACHE_CONFIG.sample_rate != case['duration_ms']:
            raise ValueError('Decoded audio duration differs from the pinned panel')
        tick = time.perf_counter()
        mel = compute_log_mel_10ms(wave, sample_rate=MUSIC_MEL_CACHE_CONFIG.sample_rate,
                                  config=MUSIC_MEL_CACHE_CONFIG)
        mel_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        cached = np.load(case['mel_file'], allow_pickle=False)
        cache_seconds = time.perf_counter() - tick
        if not np.array_equal(mel, cached):
            raise ValueError('Fresh canonical Mel differs from the pinned cached Mel')
        records.append(dict(repeat=repeat, decode_seconds=decoded, mel_seconds=mel_seconds,
            mel_cache_read_seconds=cache_seconds, mel_bytes=mel.nbytes, waveform_bytes=wave.nbytes,
            decoded_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),
            mel_array_sha256=hashlib.sha256(mel.tobytes()).hexdigest()))
    return records


def run_benchmark(config, *, resolved_yaml='', runtime_import_seconds=0.):
    config.validate()
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output/'config.yaml').write_text(resolved_yaml)
    for name in ('checkpoint', 'panel'):
        if digest(getattr(config, name+'_file')) != getattr(config, name+'_sha256'):
            raise ValueError(f'{name} bytes differ from their pinned SHA-256')
    panel = json.loads(Path(config.panel_file).read_text())
    cases = [c for c in panel['cases'] if not config.case_names or c['name'] in config.case_names]
    if not cases or set(config.case_names) - {c['name'] for c in cases}:
        raise ValueError('Requested case is absent from the panel')
    if len({c['name'] for c in cases}) != len(cases):
        raise ValueError('Panel case names must be unique')
    for case in cases:
        for kind in ('audio', 'mel'):
            if digest(case[kind+'_file']) != case[kind+'_sha256']:
                raise ValueError(f'{case["name"]}: {kind} bytes differ from the panel')
    torch.set_num_threads(config.cpu_threads)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    model = load_model(config.checkpoint_file, device=config.device)
    _synchronize(next(model.parameters()).device)
    load_seconds = time.perf_counter() - started
    environment = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
        platform=platform.platform(), cpu_threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
        git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        git_diff_sha256=hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'])).hexdigest(),
        benchmark_sha256=digest(__file__), runtime_import_seconds=runtime_import_seconds,
        model_load_seconds=load_seconds, parameters=sum(p.numel() for p in model.parameters()),
        model_config=asdict(model.config), probability_options=model.probability_options(),
        frontend=frontend_identity(), memory_initial=memory_snapshot(config.device))
    write_json(output/'environment.json', environment)
    write_json(output/'inputs.json', cases)
    preprocessing = {}
    if config.preprocess:
        for case in cases:
            preprocessing[case['name']] = preprocess_case(case, config.repeats)
            write_json(output/'preprocessing.json', preprocessing)
    mels = {case['name']: np.load(case['mel_file'], allow_pickle=False) for case in cases}
    encoded, cache_fill = {}, {}
    if config.cache_encoded:
        for name, mel in mels.items():
            tick = time.perf_counter()
            with torch.inference_mode():
                encoded[name] = model.encode_audio(torch.as_tensor(mel, device=config.device)[None])
            _synchronize(next(model.parameters()).device)
            cache_fill[name] = dict(seconds=time.perf_counter()-tick,
                bytes=encoded[name].numel()*encoded[name].element_size())
        write_json(output/'encoded_cache_fill.json', cache_fill)
    # Each process records one first-use pass separately, then shuffles warm jobs.
    warmup, _ = measure_generation(model, mels[cases[0]['name']], cases[0], config.conditions[0],
        config, config.seed, encoded=encoded.get(cases[0]['name']))
    write_json(output/'first_use.json', warmup)
    jobs = [(i, condition, repeat) for repeat in range(config.repeats)
            for i in range(len(cases)) for condition in config.conditions]
    random.Random(config.seed).shuffle(jobs)
    write_json(output/'jobs.json', jobs)
    summaries = []
    for index, condition, repeat in jobs:
        case = cases[index]
        seed = config.seed + 1000*index + repeat
        metrics, native = measure_generation(model, mels[case['name']], case, condition,
            config, seed, encoded=encoded.get(case['name']))
        name = f'{case["name"]}-{condition}-{repeat}'
        saved = save_rollout(output/name, native)
        metrics.update(repeat=repeat, reparse_pass=saved.get('reparse_pass'),
                       mechanics=saved['mechanics'])
        if config.profile_stages and repeat == 0:
            diagnostic, _ = measure_generation(model, mels[case['name']], case, condition,
                config, seed, encoded=encoded.get(case['name']), profile=True)
            if diagnostic['row_sequence_sha256'] != metrics['row_sequence_sha256']:
                raise AssertionError('Stage instrumentation changed the trajectory')
            write_json(output/name/'stages.json', diagnostic)
        write_json(output/name/'measurement.json', metrics)
        summary = {key: value for key, value in metrics.items()
                   if key not in ('windows', 'controls', 'mechanics')}
        summaries.append(summary)
        with (output/'measurements.jsonl').open('a') as stream:
            stream.write(json.dumps(summary, allow_nan=False)+'\n')
        print(json.dumps({k: summary[k] for k in ('case', 'condition', 'repeat',
            'generation_seconds', 'ready_seconds', 'completed')}), flush=True)
    result = dict(runs=len(summaries), complete_runs=sum(r['completed'] for r in summaries),
        output=str(output.resolve()), model_load_seconds=load_seconds,
        generation_seconds=distribution([r['generation_seconds'] for r in summaries]),
        ready_seconds=distribution([r['ready_seconds'] for r in summaries if r['ready_seconds'] is not None]))
    write_json(output/'summary.json', result)
    return result
