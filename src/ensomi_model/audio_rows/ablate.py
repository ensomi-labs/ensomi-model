"""Teacher-forced input ablations of a trained figure generator on fit_dev charts.

Each chart is scored from its first beat, up to ``max_beats`` beats, with the true history. The
conditions change one input:

- ``request_same_audio``: the song request of another chart of the same audio with a different
  request (charts without one are skipped);
- ``request_shuffled``: the song request of another development chart;
- ``audio_roll``: the song's Mel rotated by half its length;
- ``copy_off``: the gate forced to the generator;
- ``copy_audio``: the copy heads' audio query and key from shuffled beats;
- ``unary_off``: the position audio scores set to zero;
- ``sections_given``: the true log(W_H / s) of every 16-beat section given as a section request.

ΔNLL is nats per beat against the given condition, with a 95 % interval from resampling song
groups (``evaluate.bootstrap``)::

    python -m ensomi_model.audio_rows.ablate --model <run>/model.pt --data <build> --out <file.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .beatdata import Beats
from .data import section_label
from .evaluate import bootstrap
from .fit import collate
from .net import load_net, nll, to_device

CONDITIONS = ('given', 'request_same_audio', 'request_shuffled', 'audio_roll', 'copy_off', 'copy_audio', 'unary_off',
              'sections_given')


@torch.no_grad()
def ablate(model, data: Beats, max_beats=1024, batch=8, device='cpu', seed=973, limit=None):
    rng = np.random.default_rng(seed)
    rows = sorted(data.dev, key=lambda r: r['sha'])
    if limit:
        rows = [rows[i] for i in np.sort(rng.permutation(len(rows))[:limit])]
    by_key = {}
    for r in rows:
        by_key.setdefault(r['key'], []).append(r)
    same = {}
    for r in rows:
        others = [o for o in by_key[r['key']] if abs(o['label'] - r['label']) > 0.05]
        if others:
            same[r['sha']] = others[int(rng.integers(len(others)))]['label']
    perm = rng.permutation(len(rows))
    shuffled = {r['sha']: rows[perm[(i + 1) % len(rows)]]['label'] for i, r in enumerate([rows[j] for j in perm])}

    def windows(cond, chunk):
        out = []
        for r in chunk:
            kw = {}
            if cond == 'request_same_audio':
                kw['request'] = same[r['sha']]
            elif cond == 'request_shuffled':
                kw['request'] = shuffled[r['sha']]
            elif cond == 'audio_roll':
                kw['roll'] = 0.5
            elif cond == 'sections_given':
                c = data.chart(r)
                n = min(max_beats, r['nb'])
                beat_ms = np.clip(c['edges'], 0.0, r['song_ms'])
                kw['sections'] = [(a, min(a + 16, n), section_label(c['wh'], beat_ms, a, min(a + 16, n)))
                                  for a in range(0, n, 16)]
            out.append(data.window(r, 0, max_beats, np.random.default_rng(0), **kw))
        return out

    off = dict(copy_off=('copy',), copy_audio=('copy_audio',), unary_off=('unary',))
    records = {r['sha']: dict(sha=r['sha'], group=r['group'], band=r['band']) for r in rows}
    for cond in CONDITIONS:
        todo = [r for r in rows if cond != 'request_same_audio' or r['sha'] in same]
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            b = collate(windows(cond, chunk), device, max_beats, 4096)
            loss, m = nll(model, b, drop=0.0, off=off.get(cond, ()))
            for r, l, mm in zip(chunk, loss.sum(1).tolist(), m.sum(1).tolist()):
                records[r['sha']][cond] = l
                records[r['sha']]['beats'] = mm
            if torch.device(device).type == 'mps' and i % (25 * batch) == 0:
                torch.mps.empty_cache()
    recs = list(records.values())
    for r in recs:
        for cond in CONDITIONS[1:]:
            if cond in r:
                r['d_' + cond] = r[cond] - r['given']
    summary = dict(given=bootstrap(recs, 'given'))
    for cond in CONDITIONS[1:]:
        summary['d_' + cond] = bootstrap([r for r in recs if cond in r], 'd_' + cond)
    return summary, recs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='mps')
    p.add_argument('--max-beats', type=int, default=1024)
    p.add_argument('--limit', type=int)
    a = p.parse_args(argv)
    model, _ = load_net(a.model)
    to_device(model, a.device)
    summary, recs = ablate(model, Beats(a.data), a.max_beats, device=a.device, limit=a.limit)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(model=a.model, summary=summary), indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
