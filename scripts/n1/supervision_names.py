"""Companion to `supervision_census.py`: do keysound file names name parts, and local ternary examples.

Label-only. It lists file names in unsealed 4K set folders and reads per-object sample names and
storyboard sample names from their `.osu` files; no audio file is opened. Sealed folders are
dropped with the census's own exclusion before anything is listed. With ``--charts`` it also picks
local-ternary examples from the census's per-chart table.

    python scripts/n1/supervision_names.py --out artifacts/n1-redesign-20261028/census \\
        --charts artifacts/n1-redesign-20261028/census/charts.csv
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import supervision_census as sc  # noqa: E402

PARTS = {                                   # unambiguous tokens only: no two-letter channel-like ones (ch, oh, se, ...)
    'kick': 'kick', 'bd': 'kick', 'bassdrum': 'kick', 'snare': 'snare', 'sd': 'snare',
    'clap': 'clap', 'hat': 'hat', 'hh': 'hat', 'hihat': 'hat', 'cymbal': 'cymbal',
    'crash': 'cymbal', 'cym': 'cymbal', 'ride': 'cymbal', 'splash': 'cymbal', 'china': 'cymbal', 'tom': 'tom',
    'perc': 'perc', 'shaker': 'perc', 'tamb': 'perc', 'drum': 'drum', 'drums': 'drum',
    'bass': 'bass', 'sub': 'bass', 'piano': 'keys', 'pf': 'keys', 'pno': 'keys',
    'keys': 'keys', 'organ': 'keys', 'synth': 'synth', 'syn': 'synth', 'lead': 'synth',
    'pad': 'synth', 'arp': 'synth', 'pluck': 'synth', 'saw': 'synth', 'guitar': 'guitar', 'gt': 'guitar',
    'gtr': 'guitar', 'string': 'strings', 'strings': 'strings', 'str': 'strings', 'violin': 'strings',
    'vocal': 'vocal', 'vox': 'vocal', 'voice': 'vocal', 'chorus': 'vocal', 'vocals': 'vocal',
    'bell': 'bell', 'bells': 'bell', 'brass': 'brass', 'trumpet': 'brass', 'sax': 'brass', 'flute': 'wind',
    'fx': 'fx', 'sfx': 'fx', 'noise': 'fx', 'riser': 'fx', 'chord': 'chord', 'chords': 'chord',
}
TOKEN = re.compile(r'[a-z]+|\d+')


def tokens(name: str) -> list[str]:
    stem = os.path.splitext(os.path.basename(name.replace('\\', '/')))[0].casefold()
    return TOKEN.findall(stem)


def part_of(name: str) -> str | None:
    for t in tokens(name):
        if t in PARTS:
            return PARTS[t]
    return None


def local_ternary(path: Path, k: int = 5) -> dict:
    """Top charts whose ternary beats sit in a few short runs: 8+ ternary beats, at most 15 % of bars, runs of 2-8 bars."""
    rows = defaultdict(list)
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            if r['is_top'] != '1' or not r['bars']:
                continue
            tb, bars, run = int(r['tern_beats']), int(r['bars']), int(r['tern_maxrun'])
            if tb >= 8 and bars and int(r['tern_bars']) / bars <= 0.15 and 2 <= run <= 8:
                key = hashlib.sha256((r['set_dir'] + r['version']).encode()).hexdigest()
                rows[r['mode']].append((key, dict(id=r['set_dir'], version=r['version'], tern_bars=int(r['tern_bars']),
                                                  bars=bars, runs=int(r['tern_runs']), maxrun=run)))
    return {m: dict(n=len(v), examples=[x for _, x in sorted(v)[:k]]) for m, v in rows.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    for name, default in (('--root', '.'), ('--corpus', 'data/r2-corpus.parquet'),
                          ('--holdout1', 'artifacts/timing-20261009/opt/holdout/holdout.csv'),
                          ('--holdout2', 'artifacts/timing-20261009/opt/ft/pop/holdout2.csv'),
                          ('--groups', 'artifacts/bulk/music-dedup-20261015/groups.json'),
                          ('--inventory', 'artifacts/bulk/music-dedup-20261015/audio-inventory.jsonl'),
                          ('--hard-holdout', 'artifacts/n1-eval/hard-holdout-v1'),
                          ('--hard-holdout-sha256', '00771c6491ced746bc1dd536c0cd3ef453a01798a658ebe3330ab89edec327bb')):
        ap.add_argument(name, default=default)
    ap.add_argument('--out', required=True)
    ap.add_argument('--charts')
    args = ap.parse_args(argv)
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as pq
    cols = ['path', 'set_dir', 'beatmap_set_id', 'audio_filename', 'audio_sha256', 'api_status', 'group_id', 'eval_split', 'star']
    corpus = pq.read_table(str(root / args.corpus), columns=cols).to_pylist()
    report: dict = {}
    sealed = sc.build_exclusion(args, root, corpus, report)
    by_dir = defaultdict(list)
    for r in corpus:
        if r['api_status'] in sc.STATUSES_4K:
            by_dir[r['set_dir']].append(r)
    dirs = sorted(d for d in by_dir if d not in sealed)
    folder_tok, folder_part = Counter(), Counter()
    folders_with = Counter()
    n_files = 0
    obj_part, obj_tok, sb_part = Counter(), Counter(), Counter()
    charts_part = Counter()
    examples = defaultdict(list)
    for d in dirs:
        assert d not in sealed
        files = sc.list_files(root / d)
        audio = {(r['audio_filename'] or '').casefold() for r in by_dir[d]}
        samples = [f for f in files if f.lower().endswith(sc.AUDIO_EXT) and f.casefold() not in audio
                   and not sc.HITSOUND_NAME.match(os.path.basename(f).lower())]
        n_files += len(samples)
        parts = Counter(part_of(f) for f in samples)
        for f in samples:
            folder_tok.update(set(t for t in tokens(f) if not t.isdigit()))
        folder_part.update(parts)
        if len(samples) >= 11:
            folders_with['>=11 samples'] += 1
            named = sum(v for p, v in parts.items() if p)
            if named / len(samples) >= 0.5:
                folders_with['>=11 samples, half part-named'] += 1
                examples['part_named_folder'].append((named, d, dict(Counter({p: v for p, v in parts.items() if p}).most_common(6))))
        for r in by_dir[d]:
            path = root / r['path']
            if not path.is_file():
                continue
            p = sc.parse_osu(path.read_bytes())
            names = [o['fname'] for o in p['objs'] if o['fname'] and not sc.HITSOUND_NAME.match(os.path.basename(o['fname'].replace('\\', '/')).casefold())]
            if len(names) >= max(10, 0.5 * len(p['objs'])):
                cp = Counter(part_of(f) for f in names)
                obj_part.update(cp)
                for f in names:
                    obj_tok.update(set(t for t in tokens(f) if not t.isdigit()))
                share = sum(v for k, v in cp.items() if k) / len(names)
                charts_part['keysounded'] += 1
                charts_part['keysounded, half part-named'] += share >= 0.5
            sbn = [f for f in p['sb_files'] if not sc.HITSOUND_NAME.match(os.path.basename(f))]
            if len(sbn) >= 10:
                sb_part.update(part_of(f) for f in sbn)
                charts_part['sb-keysounded'] += 1
    res = dict(folders=len(dirs), sample_files=n_files, folders_with=dict(folders_with),
               folder_parts={str(k): v for k, v in folder_part.most_common()},
               folder_top_tokens=folder_tok.most_common(60),
               object_keysound_parts={str(k): v for k, v in obj_part.most_common()},
               object_top_tokens=obj_tok.most_common(40),
               storyboard_keysound_parts={str(k): v for k, v in sb_part.most_common()},
               charts=dict(charts_part), exclusion=report['exclusion'],
               examples={k: [dict(id=d, named=n, parts=p) for n, d, p in sorted(v, key=lambda x: (-x[0], x[1]))[:6]]
                         for k, v in examples.items()})
    if args.charts:
        res['local_ternary'] = local_ternary(Path(args.charts))
    (out / 'names.json').write_text(json.dumps(res, indent=1, sort_keys=True))
    print(json.dumps(dict(folders=len(dirs), sample_files=n_files, charts=dict(charts_part))))
    return 0


if __name__ == '__main__':
    sys.exit(main())
