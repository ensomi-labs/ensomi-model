"""``python -m ensomi_n.synth``: list the settings, or render clips with their ground truth.

    python -m ensomi_n.synth list
    python -m ensomi_n.synth render --audio DIR --truth DIR [--settings a,b] [--seeds 0,1] [--commit SHA]

``render`` writes each clip's WAV to ``--audio``, its ground truth to
``--truth/<id>/``, and ``--truth/manifest.jsonl`` (one row per clip, with the
WAV's sha256) plus ``--truth/settings.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from . import SETTINGS, STRATA, VERSION, make, write_clip


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog='python -m ensomi_n.synth')
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('list')
    r = sub.add_parser('render')
    r.add_argument('--audio', required=True)
    r.add_argument('--truth', required=True)
    r.add_argument('--settings', default='')
    r.add_argument('--seeds', default='0,1')
    r.add_argument('--commit', default='')
    a = p.parse_args(argv)
    if a.cmd == 'list':
        for s in SETTINGS.values():
            print(f'{s.name:18s} {s.stratum:4s} {s.meaning}')
        return 0
    names = [x for x in a.settings.split(',') if x] or list(SETTINGS)
    seeds = [int(x) for x in a.seeds.split(',') if x]
    truth = Path(a.truth)
    truth.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for name in names:
        for seed in seeds:
            t1 = time.time()
            clip = make(name, seed)
            row = write_clip(clip, a.audio, truth, a.commit)
            row['render_s'] = round(time.time() - t1, 2)
            rows.append(row)
            print(json.dumps({k: row[k] for k in ('id', 'n_beats', 'audio_sha256', 'render_s')}), flush=True)
    with (truth / 'manifest.jsonl').open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    (truth / 'settings.json').write_text(json.dumps(dict(
        version=VERSION, commit=a.commit, strata=STRATA,
        settings={s.name: dict(stratum=s.stratum, meaning=s.meaning) for s in SETTINGS.values()}), indent=1) + '\n')
    print(f'{len(rows)} clips in {time.time() - t0:.1f} s', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
