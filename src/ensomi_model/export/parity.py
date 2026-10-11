"""Replay every recorded graph input, then replay Python decode loops with ONNX and recorded draws.

    python -m ensomi_model.export.parity --model-dir DIR --reference artifacts/runtime-ref

Reports numerical differences on matched graph inputs separately from decode
identity with carried ORT state. This is a runtime equivalence check, not a
chart-quality evaluation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from .recording import OrtRuntime, RandomSource, Recording
from .reference import SONGS, compare_objects, make_kwargs


def network_replay(recording, runtime, tolerance=1e-4):
    report = {}
    for call in recording.index['graphs']:
        name = call['name']
        feeds = {k: torch.from_numpy(recording.get(v).copy()) for k, v in call['inputs'].items()}
        output = runtime.graph(name, feeds, None)
        entry = report.setdefault(name, dict(calls=0, outputs={}))
        entry['calls'] += 1
        for key, actual in output.items():
            expected = recording.get(call['outputs'][key])
            actual = actual.numpy()
            finite = np.isfinite(expected) & np.isfinite(actual)
            nonfinite_equal = np.array_equal(np.isnan(expected), np.isnan(actual)) and np.array_equal(
                np.isposinf(expected), np.isposinf(actual)) and np.array_equal(np.isneginf(expected), np.isneginf(actual))
            delta = np.abs(actual[finite].astype(np.float64) - expected[finite].astype(np.float64))
            maximum = float(delta.max(initial=0.))
            out = entry['outputs'].setdefault(key, dict(max_abs=0., within_tolerance=True, nonfinite_equal=True))
            if maximum > out['max_abs']:
                index = tuple(int(x) for x in np.unravel_index(
                    np.where(finite, np.abs(actual.astype(np.float64) - expected.astype(np.float64)), -1).argmax(),
                    expected.shape))
                out.update(max_abs=maximum, scope=call['scope'], step=call['step'], index=index,
                           reference=float(expected[index]), onnx=float(actual[index]))
            out['nonfinite_equal'] &= nonfinite_equal
            out['within_tolerance'] &= maximum <= tolerance and nonfinite_equal
    for name, entry in report.items():
        entry['max_abs'] = max(o['max_abs'] for o in entry['outputs'].values())
        entry['within_tolerance'] = all(o['within_tolerance'] for o in entry['outputs'].values())
        entry['cpu_seconds'] = runtime.timings[name]['seconds']
    return report


def decode_replay(system, model_dir, reference, song, recording, runtime):
    from ..system.make import _jsonable
    example = Path(model_dir) / 'examples' / song
    receipt = json.loads((example / 'receipt.json').read_text())
    random = RandomSource(replay=recording)
    out = Path(reference) / 'bulk' / song / 'ort.osz'
    t = time.perf_counter()
    result = {}
    try:
        made = system.make(SONGS[song], out=out, recorder=runtime, random_source=random, **make_kwargs(receipt))
        result.update(published=compare_objects(out.with_suffix('') / 'generated.osu', example / 'generated.osu'),
                      reference=compare_objects(out.with_suffix('') / 'generated.osu',
                                                Path(reference) / 'bulk' / song / 'reference' / 'generated.osu'),
                      head_rows=made['rows']['K'], difficulty_passes=made['controls']['difficulty']['passes'])
    except (ValueError, RuntimeError) as e:
        result['error'] = f'{type(e).__name__}: {e}'
    result.update(seconds=time.perf_counter() - t, graph_cpu=dict(runtime.timings),
                  draws_consumed=random.position, draws_recorded=len(recording.index['draws']),
                  draw_differences=random.differences)
    result['chart_identical'] = bool(result.get('reference', {}).get('identical'))
    result['draws_identical'] = random.position == len(recording.index['draws']) and not random.differences
    result['identical'] = result['chart_identical'] and result['draws_identical']
    return _jsonable(result)


def main(argv=None):
    import onnxruntime
    from ..system.make import System
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--model-dir', type=Path, default=Path('artifacts/release-r2/bulk/hf/ensomi-r2'))
    p.add_argument('--reference', '--out', dest='reference', type=Path, default=Path('artifacts/runtime-ref'))
    p.add_argument('--onnx-dir', type=Path)
    p.add_argument('--songs', choices=tuple(SONGS), nargs='+', default=list(SONGS))
    p.add_argument('--tolerance', type=float, default=1e-4)
    a = p.parse_args(argv)
    graph_dir = a.onnx_dir or a.model_dir / 'onnx'
    if not graph_dir.is_dir():
        p.error('Export the ONNX graphs first')
    for song in a.songs:
        if not (a.reference / 'bulk' / song / 'index.json').is_file():
            p.error(f'Missing recording for {song}; run ensomi_model.export.reference first')
    torch.set_num_threads(1)
    system = System.load(json.loads((a.model_dir / 'system.json').read_text()), a.model_dir)
    report = dict(tolerance=a.tolerance, torch=torch.__version__, onnxruntime=onnxruntime.__version__, cpu_threads=1,
                  provider='CPUExecutionProvider',
                  scope='matched graph inputs and whole-song Python decode with ORT carried state', songs={})
    for song in a.songs:
        recording = Recording(a.reference / 'bulk' / song)
        runtime = OrtRuntime(graph_dir)
        numerical = network_replay(recording, runtime, a.tolerance)
        runtime.timings = defaultdict(lambda: dict(calls=0, seconds=0.0))
        decode = decode_replay(system, a.model_dir, a.reference, song, recording, runtime)
        report['songs'][song] = dict(networks=numerical, decode=decode)
        (a.reference / 'parity.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps(dict(song=song, networks={k: v['max_abs'] for k, v in numerical.items()}, decode=decode)), flush=True)
    report['passed'] = all(all(v['within_tolerance'] for v in s['networks'].values()) and s['decode']['identical']
                           for s in report['songs'].values())
    (a.reference / 'parity.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
