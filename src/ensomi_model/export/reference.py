"""Record the four released example songs through System.make, including every difficulty pass.

    python -m ensomi_model.export.reference --model-dir DIR --out artifacts/runtime-ref

Raw arrays and generated charts live in out/bulk/SONG. Each out/SONG/summary.json
reports counts, sizes and exact hitobject identity against the published example.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from .recording import Recorder


SONGS = {
    'hesitation-snow': 'dataset/0/2486779/audio.ogg',
    'saigetsu': 'dataset/0/1403084/audio.mp3',
    'catch-the-moment': 'dataset/0/576112/audio.mp3',
    'adventure': 'dataset/0/2202551/audio.ogg',
}


def object_keys(path):
    from ..osu_core.hitobjects import parse_mania_hit_objects
    return sorted((o.start_time_ms, o.lane, o.end_time_ms) for o in parse_mania_hit_objects(path))


def compare_objects(actual, expected):
    a, b = object_keys(actual), object_keys(expected)
    differences = [dict(index=i, actual=x, expected=y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
    return dict(identical=a == b, actual_objects=len(a), expected_objects=len(b),
                first_differences=differences[:10])


def make_kwargs(receipt):
    return dict(star=receipt['star'], seed=receipt['seed'], title=receipt['metadata']['title'],
                artist=receipt['metadata']['artist'], copy='off')


def record_song(system, model_dir, out, song):
    t = time.perf_counter()
    example = Path(model_dir) / 'examples' / song
    receipt = json.loads((example / 'receipt.json').read_text())
    bulk = Path(out) / 'bulk' / song
    rec = Recorder(bulk)
    try:
        result = system.make(SONGS[song], out=bulk / 'reference.osz', recorder=rec, **make_kwargs(receipt))
    finally:
        recorded = rec.close()
    comparison = compare_objects(bulk / 'reference' / 'generated.osu', example / 'generated.osu')
    summary = dict(song=song, audio=SONGS[song], settings=make_kwargs(receipt), recording=recorded,
                   published=comparison, head_rows=result['rows']['K'],
                   difficulty_passes=result['controls']['difficulty']['passes'], seconds=time.perf_counter() - t,
                   artifacts=dict(index=str(bulk / 'index.json'), chart=str(bulk / 'reference' / 'generated.osu')))
    small = Path(out) / song
    small.mkdir(parents=True, exist_ok=True)
    (small / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return summary


def main(argv=None):
    from ..system.make import System
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--songs', choices=tuple(SONGS), nargs='+', default=list(SONGS))
    a = p.parse_args(argv)
    for song in a.songs:
        if not Path(SONGS[song]).is_file():
            p.error(f'Missing audio for {song}: {SONGS[song]}')
        if not (a.model_dir / 'examples' / song / 'receipt.json').is_file():
            p.error(f'Missing example receipt for {song}')
    torch.set_num_threads(1)
    system = System.load(json.loads((a.model_dir / 'system.json').read_text()), a.model_dir)
    failed = False
    for song in a.songs:
        summary = record_song(system, a.model_dir, a.out, song)
        failed |= not summary['published']['identical']
        print(json.dumps(summary), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
