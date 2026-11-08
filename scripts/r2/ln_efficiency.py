"""Small CPU latency and parameter census of a released system directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import tempfile
import time

import numpy as np
import torch

from separation import disk_guard, write_json
from ensomi_model.system import make


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--panel', required=True)
    p.add_argument('--published-audio', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    root, out = Path(args.model_dir), Path(args.out)
    disk_guard(out)
    torch.set_num_threads(1)
    system = make.System.load(json.loads((root / 'system.json').read_text()), model_dir=root)
    counts = system.r2.parameter_counts()
    assert system.r2.config.memory == 'none' and system.r2.config.identity_gate
    dormant = ('film', 'tokens', 'lm_query', 'lm_key', 'lm_value', 'lm_out')
    parameters = dict(r2=counts, r2_natural_inference=counts['total'] - sum(counts[k] for k in dormant),
                      dormant_modules=dormant, rows=sum(p.numel() for p in system.rows.model.parameters()),
                      timing=sum(p.numel() for p in system.rows.listener.frames.model.parameters()))
    panel = json.loads(Path(args.panel).read_text())['charts']
    examples = [(args.published_audio, 2.5, 'published')]
    examples += [(next(r for r in panel if r['ln_class'] == cls)['audio'], 3.5, cls)
                 for cls in ('normal', 'ln_heavy', 'inverse')]
    records = []
    original_predict, original_export = system.prior.predict, make.write_export
    durations = {}

    def prior(*a, **kw):
        t = time.monotonic()
        result = original_predict(*a, **kw)
        durations['prior'] = time.monotonic() - t
        return result

    def export(*a, **kw):
        t = time.monotonic()
        result = original_export(*a, **kw)
        durations['export'] = time.monotonic() - t
        return result

    system.prior.predict, make.write_export = prior, export
    try:
        with tempfile.TemporaryDirectory(prefix='latency-', dir=out) as temporary:
            for i, (audio, star, label) in enumerate(examples):
                durations.clear()
                t = time.monotonic()
                receipt = system.make(audio, star, Path(temporary) / f'{i}.osz', seed=0, name=label,
                                      title='Latency probe', artist='Unknown')
                row = dict(label=label, audio=audio, star=star, seed=0, threads=1,
                           total_with_export=time.monotonic() - t,
                           seconds=dict(receipt['seconds'], **durations),
                           song_ms=float(np.load(Path(temporary) / str(i) / 'arrays.npz')['song_ms']),
                           rows=receipt['rows']['K'],
                           arrangement_passes=len(receipt['controls']['difficulty']['passes']),
                           peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                records.append(row)
                print(json.dumps(row), flush=True)
    finally:
        system.prior.predict, make.write_export = original_predict, original_export
    write_json(out / 'efficiency.json', dict(parameters=parameters, songs=records))


if __name__ == '__main__':
    main()
