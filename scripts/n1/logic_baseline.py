"""Logic Pro Smart Tempo as a commercial baseline: scored against the charts beside R2 and N1 v0.

The question (research notes, ``n1-eval-20261018`` ``d-smart-tempo-baseline`` and ``o-smart-tempo-captured``):
how do Logic Pro 12.4's untouched Smart Tempo maps, in Constant and Variable mode, compare with the charts'
timing on the 48 songs of tiers A-C of the Logic listening list, beside R2 as exported and v0 (both seeds), with
N's measures? Every number characterises: no measure here has been shown to agree with the human's ear.

Clock rule (the rule raw BeatThis follows: an unfitted tracker on a decoded clock, one constant, never fitted
per song). A Smart Tempo time is on the canonical FFmpeg decode of the source (time zero at its first sample).
`ensomi_n.mapformat.logic.read_logic_map` adds the ffmpeg lead of N's osu! clock rule (`osu-lead-v1`,
``features.clock.file_lead(audio, 'ffmpeg')``: 0 or 529 samples, read from the file's header), which puts the
map on the decoded clock osu! plays. The map then goes through the harness guard with one decoded-to-osu!
constant ``c``; the sources give two and no third, so both are scored and neither is chosen here:

- ``c = -20.0`` ms, the a-priori osu! convention (research notes ``timing-phase#o-osu-offsets``), the second
  constant of E1;
- ``c = -28.7`` ms, the constant E1 measured once on synthetic attacks (``n1-night-20261018/e1-report`` §5).

v0's and R2's maps are on osu!'s clock already and are scored as they are.

Subcommands (run from the project root with ``PYTHONPATH=<src>/src:<src>/n/src``):

- ``prep``: the 48 songs (audio, reference charts, hard span, v0 exposure, clock lead) into ``songs.json`` and a
  split-shaped ``logic-split.json`` for `bt_logits.py` and `r2_export.py`;
- ``cache``: the log-Mel, song statistics and memory input of every song, into a new bulk directory;
- ``v0maps``: v0 decoded on the CPU for every song, at the last three checkpoints of both seeds, with seed 0's
  cal-sweep decoder setting, and the reproduction check against the maps the v0 analysis kept;
- ``score``: every arm on every song, on the whole song, the hard span and the stratum's spans; tier A
  agreement and tier B pulse checks; per-song records;
- ``report``: tables per tier, per stratum for tier C and per song, with song-cluster bootstraps.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
TAU = 20.0
CLOCK_CONSTANTS = {'c20': -20.0, 'c287': -28.7}
CLOCK_SOURCES = {
    'c20': 'a-priori osu! convention (timing-phase#o-osu-offsets; E1 reports it beside its measured constant)',
    'c287': 'E1 measured once: median offset of R2 as fitted against synthetic S0 ground truth (e1-report §5)',
}
MODES = ('constant', 'variable')
SEEDS = {'s0': 'v0-s0-20261019', 's1': 'v0-s1-20261019'}
BOOT = 2000
AGREE = 0.9          # tier A: two arms agree on a span when each has this share of its lines matched within 20 ms
PULSE_HIGH, PULSE_LOW = 0.8, 0.2   # tier B classification thresholds
FINE_LEVEL = 1.25    # tier B: an arm whose median beat is at most this many notated beats can hit both readings
ATTR_PASS = 0.9      # attribution: BL-20 at or above this is a pass
ATTR_FOLD = 0.3      # attribution: union hits above BL-20 by this much is a fold-phase loss at an octave level
ATTR_LEVEL = 0.5     # attribution: duration share at family/other level from which the pulse itself differs
#: attribution: a whole-grid shift up to this size is clock- or convention-sized (BeatThis sits 30 ms after the
#: charts, timing-phase#o-measure-returned; the two clock constants differ by 8.7 ms); larger is a phase shift.
ATTR_CLOCK_MS = 35.0

#: Per-stratum margins m = max(checkpoint, seed spread) of v0 - R2, BL-20 non-free, N reference, full spans
#: (research notes `n1-night-20261018/v0-analysis-tables.md`, "N reference, full spans; all"). A scale for
#: differences on these songs; they were measured for v0 against R2 on other songs, not for Smart Tempo.
MARGINS = {
    'panel': dict(all=0.018, S0=0.011, S1=0.024, S2=0.036, S3=0.030, S4=0.046, S5=0.026, S6=0.038, S7=0.073,
                  S8=0.110, S9=0.025),
    'val_hard': dict(all=0.011, S0=0.275, S1=0.010, S2=0.027, S3=0.018, S4=0.023, S5=0.047, S6=0.086, S7=0.088,
                     S8=0.063, S9=0.059),
}
MARGIN_N_BELOW_10 = {'panel': ('S7', 'S8'), 'val_hard': ('S0', 'S7', 'S8')}

# Tier A: the v0 packet's passages (listen-v0/work/picks.json); tier C: the 27-passage packet's clip windows
# (listen-gentle/*/*.json, window_s), from 4 s into the clip (the hard part) to its end; where the clip starts at
# 0 the hard part's start comes from the list's index. Tier B: the whole chart.
SONGS = [
    ('001', 'A01', 'A', 'S8', 'dataset/0/2146184/audio.mp3', (101964.5, 113964.5)),
    ('002', 'A02', 'A', 'S8', 'dataset/0/1433856/audio.mp3', (147130.0, 159130.0)),
    ('003', 'A03', 'A', 'S8', 'dataset/0/1247923/audio.ogg', (602.0, 12602.0)),
    ('004', 'A04', 'A', 'S8', 'dataset/0/591171/DJ HERO - Noisia  Groundhog (Beat Juggle).mp3', (33035.0, 45035.0)),
    ('005', 'A05', 'A', 'S6', 'dataset/0/2304311/audio.mp3', (149618.9, 169618.9)),
    ('006', 'A06', 'A', 'S6', 'dataset/0/1392416/audio.mp3', (30994.3, 50994.3)),
    ('007', 'A07', 'A', 'S6', 'dataset/0/2409371/audio.mp3', (209352.8, 229352.8)),
    ('008', 'A08', 'A', 'S6', 'dataset/0/1885239/audio.mp3', (191439.4, 211439.4)),
    ('009', 'A09', 'A', 'S1', 'dataset/0/2014470/audio.mp3', (10231.3, 30231.3)),
    ('010', 'A10', 'A', 'S1', 'dataset/0/1871690/audio.mp3', (62282.4, 82282.4)),
    ('011', 'A11', 'A', 'S1', 'dataset/0/911470/Terminal 11 - Illegal Nervous Habits - 05 Adcar.mp3',
     (148501.1, 168501.1)),
    ('012', 'A12', 'A', 'S1', 'dataset/0/1673284/audio.mp3', (58737.0, 78737.0)),
    ('013', 'B01', 'B', 'S2', 'dataset/0/1701849/audio.mp3', None),
    ('014', 'B02', 'B', 'S2', 'dataset/0/2042835/audio.mp3', None),
    ('015', 'B03', 'B', 'S2', 'dataset/0/2514569/audio.mp3', None),
    ('016', 'B04', 'B', 'S2', 'dataset/0/988031/audio.mp3', None),
    ('017', 'B05', 'B', 'S2', 'std-taiko:664841', None),
    ('018', 'B06', 'B', 'S2', 'dataset/0/2158889/audio.mp3', None),
    ('019', 'B07', 'B', 'S2', 'dataset/0/969976/audio.mp3', None),
    ('020', 'B08', 'B', 'S2', 'dataset/0/853406/Contain.mp3', None),
    ('021', 'B09', 'B', 'S2', 'std-taiko:414667', None),
    ('022', 'C01', 'C', 'S1', 'dataset/0/1854144/audio.mp3', (16750.0, 42750.0)),
    ('023', 'C02', 'C', 'S1', 'dataset/0/2254787/audio.mp3', (151750.0, 177750.0)),
    ('024', 'C03', 'C', 'S1', 'dataset/0/2046962/audio.ogg', (157250.0, 183250.0)),
    ('025', 'C04', 'C', 'S2', 'dataset/0/1969233/audio.mp3', (36000.0, 62000.0)),
    ('026', 'C05', 'C', 'S2', 'dataset/0/1785169/audio.mp3', (91750.0, 117750.0)),
    ('027', 'C06', 'C', 'S2', 'dataset/0/1936860/audio.mp3', (34269.0, 56869.0)),
    ('028', 'C07', 'C', 'S3', 'dataset/0/2577371/audio.ogg', (63808.0, 89552.0)),
    ('029', 'C08', 'C', 'S3', 'dataset/0/1787322/audio.mp3', (27359.0, 52878.0)),
    ('030', 'C09', 'C', 'S3', 'dataset/0/2021976/audio.mp3', (51100.0, 76106.0)),
    ('031', 'C10', 'C', 'S4', 'dataset/0/2054068/audio.mp3', (83545.4, 105755.3)),
    ('032', 'C11', 'C', 'S4', 'dataset/0/2495364/audio.ogg', (85000.0, 101978.0)),
    ('033', 'C12', 'C', 'S4', 'dataset/0/2210015/audio.mp3', (500.0, 25830.0)),
    ('034', 'C13', 'C', 'S5', 'dataset/0/1280470/audio.mp3', (144750.0, 170750.0)),
    ('035', 'C14', 'C', 'S5', 'dataset/0/1899965/audio.mp3', (31292.0, 50522.0)),
    ('036', 'C15', 'C', 'S5', 'dataset/0/1470543/audio.mp3', (53802.0, 63802.0)),
    ('037', 'C16', 'C', 'S6', 'dataset/0/2011396/audio.mp3', (90500.0, 111597.5)),
    ('038', 'C17', 'C', 'S6', 'dataset/0/2491179/audio.mp3', (135785.0, 155096.3)),
    ('039', 'C18', 'C', 'S6', 'dataset/0/1412093/audio.mp3', (179500.0, 200745.0)),
    ('040', 'C19', 'C', 'S7', 'dataset/0/1039475/audio.mp3', (800.0, 24734.0)),
    ('041', 'C20', 'C', 'S7', 'dataset/0/2089358/audio.ogg', (206500.0, 232500.0)),
    ('042', 'C21', 'C', 'S7', 'dataset/0/1042161/audio.mp3', (200.0, 26250.0)),
    ('043', 'C22', 'C', 'S8', 'dataset/0/1972938/audio.ogg', (143033.0, 149033.0)),
    ('044', 'C23', 'C', 'S8', 'dataset/0/1080799/audio.mp3', (27250.0, 51387.0)),
    ('045', 'C24', 'C', 'S8', "dataset/0/178426/XeoN - Thor ~ Deepin' Absonant Mix ~.mp3", (111000.0, 132446.0)),
    ('046', 'C25', 'C', 'S9', 'dataset/0/2168878/audio.mp3', (226370.0, 251201.0)),
    ('047', 'C26', 'C', 'S9', 'dataset/0/1726083/audio.mp3', (50000.0, 62942.0)),
    ('048', 'C27', 'C', 'S9', 'dataset/0/1164662/audio.mp3', (62694.0, 87374.0)),
]


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


#: The text log goes to the bulk directory; the results directory holds only Markdown and JSON, and its
#: ``progress.json`` keeps the messages logged so far.
LOG_DIR: Path | None = None


def _log(out: Path, msg: str) -> None:
    line = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} {msg}'
    out.mkdir(parents=True, exist_ok=True)
    if LOG_DIR is not None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / 'log.txt', 'a') as f:
            f.write(line + '\n')
    p = out / 'progress.json'
    try:
        msgs = json.loads(p.read_text()) if p.exists() else []
    except ValueError:
        msgs = []
    msgs.append(line)
    _dump(p, msgs)
    print(line, flush=True)


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, default=_jsonable) + '\n')
    os.replace(tmp, path)


def _jsonable(x):
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (set, tuple)):
        return list(x)
    raise TypeError(f'not JSON: {type(x).__name__}')


def _clean(v):
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return round(v, 6) if math.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


# ---------------------------------------------------------------- prep

def _std_taiko_audio(root: Path, set_id: str) -> tuple[str, list[str]]:
    """The audio of a std-taiko set's osu!standard charts (AudioFilename, matched without regard to case), and those charts."""
    from ensomi_n.eval.census import read_header
    d = root / 'dataset' / 'std-taiko' / set_id
    charts, names = [], set()
    for p in sorted(d.glob('*.osu')):
        h = read_header(p)
        if str(h.get('Mode', '0')).strip() != '0':
            continue
        charts.append(p)
        names.add(str(h.get('AudioFilename', '')).strip().casefold())
    if len(names) != 1:
        raise ValueError(f'std-taiko {set_id}: mode-0 charts name {sorted(names)} audio files, expected one')
    want = names.pop()
    files = [p for p in d.iterdir() if p.is_file() and p.name.casefold() == want]
    if len(files) != 1:
        raise ValueError(f'std-taiko {set_id}: {len(files)} files match AudioFilename {want!r}')
    return str(files[0].relative_to(root)), [str(p.relative_to(root)) for p in charts]


def v0_exposure(sha: str, recs: set[str], curated: list[dict]) -> tuple[str, str]:
    """('trained' | 'held-out', why) for one song from v0's curated data (``v0data2-20261019/curated.json``).

    A song is trained-on when any record of its recording (its audio sha256, or a recording key of it, which
    also joins other audio copies of the same recording) has role ``train`` and ``keep`` true.
    """
    mine = [r for r in curated if r.get('audio_sha256') == sha or r.get('item') == sha or r.get('rec') in recs]
    kept = [r for r in mine if r.get('role') == 'train' and r.get('keep') is True]
    if kept:
        r = kept[0]
        via = '' if r.get('audio_sha256') == sha else f" through another audio copy ({r.get('audio_rel')})"
        return 'trained', f"kept for training{via}, source {r.get('source')}, weight {r.get('weight')}"
    if not mine:
        return 'held-out', 'no curated record'
    roles = sorted({str(r.get('role')) for r in mine})
    why = sorted({', '.join(r.get('reasons') or []) for r in mine if r.get('role') == 'train'} - {''})
    return 'held-out', f"curated roles {roles}, not kept for training" + (f" (dropped: {'; '.join(why)})" if why else '')


def _same_audio(root: Path, chart: str, audio_rel: str) -> bool:
    """The chart lies beside the audio file and names it (``AudioFilename``, without regard to case)."""
    from ensomi_n.labels import read_osu_audio
    c, au = Path(chart), Path(audio_rel)
    return c.parent == au.parent and read_osu_audio(root / c).casefold() == au.name.casefold()


def _directory_charts(root: Path, audio_rel: str) -> list[str]:
    """4K mania charts (Mode 3, CircleSize 4) beside the audio file that name it."""
    from ensomi_n.eval.census import read_header
    au = Path(audio_rel)
    out = []
    for p in sorted((root / au.parent).glob('*.osu')):
        h = read_header(p)
        if h.get('Mode', '').strip() == '3' and h.get('CircleSize', '').strip() in ('4', '4.0') \
                and h.get('AudioFilename', '').strip().casefold() == au.name.casefold():
            out.append(str(p.relative_to(root)))
    return out


def cmd_prep(a) -> None:
    from ensomi_n.features.clock import file_lead, read_mp3_start
    from ensomi_n.mapformat.logic import summarize
    root, out, bulk = Path(a.root), Path(a.out), Path(a.bulk)
    integrity = json.load(open(Path(a.logic) / 'source-integrity.json'))
    expected = {s['id']: s['expected_sha256'] for s in integrity['sources']}
    names = {s['id']: s['source_filename'] for s in integrity['sources']}
    if sorted(expected) != [s[0] for s in SONGS]:
        raise SystemExit('source-integrity.json ids differ from the song table')
    split = json.load(open(Path(a.r0) / 'split.json'))
    in_split = {}
    for lst, items in split.items():
        if isinstance(items, list):
            for it in items:
                in_split.setdefault(it['audio_sha256'], []).append((lst, it))
    recordings = json.load(open(Path(a.r0) / 'recordings.json'))
    by_audio: dict[str, list] = {}
    for r in recordings:
        for rel in set(r.get('audio_rels') or []) | ({r['audio_rel']} if r.get('audio_rel') else set()):
            by_audio.setdefault(rel, []).append(r)
    curated = json.load(open(a.curated))
    songs, problems = [], []
    for sid, lid, tier, group, audio, span in SONGS:
        rec = dict(id=sid, list_id=lid, tier=tier, group=group, source_filename=names[sid],
                   hard_span_ms=list(span) if span else None)
        charts, how = None, ''
        if audio.startswith('std-taiko:'):
            audio, charts = _std_taiko_audio(root, audio.split(':', 1)[1])
            how = 'std-taiko: the mode-0 charts of the set'
            parts = Path(audio).parts
            rec['rec'] = f'rec:s:std-taiko/{parts[-2]}/{parts[-1].casefold()}'
        rec['audio_rel'] = audio
        sha = sha256_file(root / audio)
        rec['audio_sha256'] = sha
        if sha != expected[sid]:
            problems.append(f'{sid}: sha256 {sha} != expected {expected[sid]}')
        listed = in_split.get(sha, [])
        rec['r0_lists'] = sorted({lst for lst, _ in listed})
        rows = by_audio.get(audio, [])
        rec['r0_role'] = rows[0].get('role') if rows else ('std-taiko' if 'std-taiko' in audio else None)
        if charts is None and listed:
            pick = next((it for lst, it in listed if lst in ('panel', 'val_hard')), listed[0][1])
            charts, how = list(pick['charts']), f"r0 split.json ({', '.join(rec['r0_lists'])})"
            rec['rec'] = pick.get('rec', sha)
        if charts is None and rows:
            mine = [c for c in rows[0]['charts'] if _same_audio(root, c, audio)]
            if mine:
                charts, how = mine, 'r0 recordings.json, the charts beside this audio file'
            rec['rec'] = rows[0].get('rec', sha)
        if charts is None:
            charts, how = _directory_charts(root, audio), '4K charts beside this audio file that name it'
        rec.setdefault('rec', sha)
        rec['charts'], rec['charts_from'] = charts, how
        if not charts:
            problems.append(f'{sid}: no reference charts')
        bad = [c for c in charts if not (root / c).is_file()]
        if bad:
            problems.append(f'{sid}: missing chart files {bad}')
        recs = {rec['rec']} | {r.get('rec') for r in rows if r.get('rec')}
        rec['v0_exposure'], rec['v0_exposure_why'] = v0_exposure(sha, recs, curated)
        rec['logic'] = {}
        for mode in MODES:
            try:
                summ = summarize(Path(a.logic) / 'bulk' / 'maps' / f'{sid}__{mode}.json')
            except Exception as exc:  # noqa: BLE001
                problems.append(f'{sid} {mode}: Smart Tempo map unreadable: {type(exc).__name__}: {exc}')
                continue
            rec['logic'][mode] = summ
            if summ['ax']['labels'] and not summ['ax']['ok']:
                problems.append(f"{sid} {mode}: the bar reading disagrees with the editor's labels {summ['ax']}")
        lead, rate, start = file_lead(root / audio, 'ffmpeg')
        lead_sf = file_lead(root / audio, 'libsndfile')[0]
        if Path(audio).suffix.lower() == '.mp3':
            st = read_mp3_start(root / audio)
            rec['mp3_start'] = dict(kind=st.kind, encoder=st.encoder, delay=st.delay, sample_rate=st.sample_rate)
        rec['lead_samples_ffmpeg'] = lead
        rec['lead_samples_libsndfile'] = lead_sf
        rec['lead_rate'] = rate
        rec['lead_ms'] = 1000.0 * lead / rate if lead else 0.0
        songs.append(rec)
    labelled = sum(1 for x in songs for m in x['logic'].values() if m['ax']['labels'])
    _dump(out / 'songs.json', dict(songs=songs, problems=problems, maps_with_labels=labelled,
                                   clock_constants=CLOCK_CONSTANTS, clock_sources=CLOCK_SOURCES))
    _dump(bulk / 'logic-split.json', {'logic': [dict(item=s['audio_sha256'], audio_sha256=s['audio_sha256'],
                                                     audio_rel=s['audio_rel'], rec=s['rec'], charts=s['charts'])
                                                for s in songs]})
    _log(out, f'prep: {len(songs)} songs, {len(problems)} problems; exposure '
              f'{sum(s["v0_exposure"] == "trained" for s in songs)} trained, '
              f'{sum(s["v0_exposure"] == "held-out" for s in songs)} held out')
    for p in problems:
        _log(out, 'PROBLEM ' + p)
    if problems:
        raise SystemExit(1)


# ---------------------------------------------------------------- cache and v0 maps

def cmd_cache(a) -> None:
    from ensomi_n.features import cache
    from ensomi_n.v0.songs import stats_one
    out, bulk, root = Path(a.out), Path(a.bulk), Path(a.root)
    songs = json.load(open(out / 'songs.json'))['songs']
    jobs = [(str(root / s['audio_rel']), s['audio_sha256']) for s in songs]
    res = cache.build(jobs, bulk / 'mel', bulk / 'cache-index.jsonl', processes=1)
    index = {}
    for s in songs:
        r = stats_one((s['audio_sha256'], str(bulk)))
        index[s['audio_sha256']] = r
    _dump(bulk / 'song-index.json', index)
    errors = {k: v['error'] for k, v in index.items() if 'error' in v}
    _log(out, f'cache: {res["counts"]} in {res["wall_s"]} s; stats errors {len(errors)}')
    if errors or res['counts'].get('error') or res['counts'].get('sha_mismatch'):
        raise SystemExit(f'cache failed: {errors or res["counts"]}')


def _stored_maps(stored: Path, run: str) -> dict:
    """{checkpoint tag: {audio sha256: map json}} from the v0 analysis's kept maps."""
    out: dict[str, dict] = {}
    for p in sorted((stored / run).glob('maps-*.jsonl.gz')):
        tag = p.name.split('-')[1]
        with gzip.open(p, 'rt') as f:
            for line in f:
                d = json.loads(line)
                out.setdefault(tag, {})[d['item']] = d['map']
    return out


def map_difference(a: dict, b: dict) -> dict:
    """Difference of two music-map JSONs: beat and segment counts, and the largest difference of a beat time, a
    segment start or a segment's beat length (ms) where the counts agree (None where they do not)."""
    def arr(m, key, field=None):
        v = m.get(key) or []
        return np.asarray([x[field] for x in v] if field else v, dtype=np.float64)
    d = dict(n_a=len(a.get('beats') or []), n_b=len(b.get('beats') or []),
             segments_a=len(a.get('segments') or []), segments_b=len(b.get('segments') or []),
             downbeats_equal=list(a.get('downbeats') or []) == list(b.get('downbeats') or []))
    worst = 0.0
    for key, field in (('beats', None), ('segments', 'start_ms'), ('segments', 'beat_ms')):
        x, y = arr(a, key, field), arr(b, key, field)
        if len(x) != len(y):
            d['max_abs_ms'] = None
            return d
        if len(x):
            worst = max(worst, float(np.max(np.abs(x - y))))
    d['max_abs_ms'] = worst
    return d


def cmd_v0maps(a) -> None:
    import torch
    torch.set_num_threads(a.threads)
    from ensomi_n.v0 import evaluate as ev
    from ensomi_n.v0.data import Song
    from ensomi_n.v0.decode import DecodeConfig, decode
    from ensomi_n.v0.infer import song_outputs
    from ensomi_n.v0.songs import memo_path
    from ensomi_n.features.cache import mel_path
    out, bulk = Path(a.out), Path(a.bulk)
    songs = json.load(open(out / 'songs.json'))['songs']
    index = json.load(open(bulk / 'song-index.json'))
    cfg_kw = ev.pick_settings(Path(a.tune_dir))
    if not cfg_kw:
        raise SystemExit(f'no tune.json under {a.tune_dir}: the decoder setting is not known')
    _log(out, f'v0maps: decode setting {cfg_kw} (seed 0 cal sweep)')
    repro = {}
    for seed, run in SEEDS.items():
        cks = ev._checkpoints(Path(a.bulk_runs) / run, 'last3')
        cks = cks[-1:] + cks[:-1]           # the last checkpoint (the primary arm) first
        stored = _stored_maps(Path(a.stored_maps), run)
        for ck in cks:
            tag = ck.stem.split('-')[1]
            path = bulk / 'v0-maps' / run / f'maps-{tag}.jsonl.gz'
            if not path.exists():
                model = ev._load_model(ck, 'cpu')
                t0 = time.time()
                tmp = path.with_name(path.name + '.tmp')
                tmp.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(tmp, 'wt') as f:
                    for s in songs:
                        sha = s['audio_sha256']
                        st = index[sha]
                        song = Song(sha, str(mel_path(bulk / 'mel', sha)), str(memo_path(bulk, sha)), st['mean'],
                                    st['std'], st['frames'], None)
                        acts = song_outputs(model, song, 'cpu')
                        mmap, diag = decode(ev.v0_evidence(acts), DecodeConfig(**cfg_kw), model=f'v0@{tag}')
                        f.write(json.dumps(dict(id=s['id'], item=sha, map=mmap.to_json(), diag=diag), default=float) + '\n')
                os.replace(tmp, path)
                del model
                _log(out, f'v0maps {seed} {tag}: {len(songs)} songs in {time.time() - t0:.0f} s')
            fresh = {}
            with gzip.open(path, 'rt') as f:
                for line in f:
                    d = json.loads(line)
                    fresh[d['item']] = d['map']
            for sha, m in stored.get(tag, {}).items():
                if sha in fresh:
                    repro.setdefault(f'{seed}@{tag}', {})[sha] = map_difference(m, fresh[sha])
    _dump(out / 'v0-reproduction.json', repro)
    worst = [(k, sha, d) for k, v in repro.items() for sha, d in v.items()
             if d['max_abs_ms'] is None or d['max_abs_ms'] > 0.01 or not d['downbeats_equal']]
    _log(out, f'v0maps: reproduction on {sum(len(v) for v in repro.values())} (checkpoint, song) pairs, '
              f'{len(worst)} differ by more than 0.01 ms or in downbeats')


# ---------------------------------------------------------------- arms

def _gz_maps(path: Path) -> dict:
    from ensomi_n.mapformat import MusicMap
    out = {}
    with gzip.open(path, 'rt') as f:
        for line in f:
            d = json.loads(line)
            out[d['item']] = MusicMap.from_json(d['map'])
    return out


def load_arms(a, songs: list[dict]) -> tuple[dict, dict]:
    """{arm: {audio sha256: MusicMap on osu!'s clock}} and the provenance of each arm."""
    from ensomi_n.mapformat.logic import read_logic_map
    from ensomi_n.v0.evaluate import r2_map
    bulk, logic = Path(a.bulk), Path(a.logic)
    arms: dict[str, dict] = {}
    prov: dict[str, dict] = {}
    for mode in MODES:
        for cname, c in CLOCK_CONSTANTS.items():
            arm = f'st-{mode}@{cname}'
            arms[arm] = {}
            prov[arm] = dict(kind='smart-tempo', mode=mode, clock_constant_ms=c, clock_source=CLOCK_SOURCES[cname])
        for s in songs:
            m = read_logic_map(logic / 'bulk' / 'maps' / f"{s['id']}__{mode}.json", lead_ms=s['lead_ms'])
            for cname, c in CLOCK_CONSTANTS.items():
                arms[f'st-{mode}@{cname}'][s['audio_sha256']] = m.on_osu_clock(c)
    for seed, run in SEEDS.items():
        for p in sorted((bulk / 'v0-maps' / run).glob('maps-*.jsonl.gz')):
            tag = p.name[len('maps-'):-len('.jsonl.gz')]
            arms[f'v0{seed}@{tag}'] = _gz_maps(p)
            prov[f'v0{seed}@{tag}'] = dict(kind='v0', seed=seed, run=run, checkpoint=tag, source=str(p))
    # R2 as exported: the export the v0 analysis used where it exists, else this run's (BeatThis final0 on the CPU).
    # This run's export of every song is kept as `r2-fresh`; on the songs with both it is a recompute null.
    arms['r2'], arms['r2-fresh'], used = {}, {}, {}
    for s in songs:
        sha = s['audio_sha256']
        fresh = bulk / 'r2-export' / f'{sha}.json'
        if fresh.exists():
            arms['r2-fresh'][sha] = r2_map(fresh)
        stored = Path(a.r2_stored) / f'{sha}.json'
        p = stored if stored.exists() else fresh
        if p.exists():
            arms['r2'][sha] = r2_map(p)
            used[sha] = str(p)
    prov['r2'] = dict(kind='r2-as-exported', files=used)
    prov['r2-fresh'] = dict(kind='r2-as-exported, recomputed here', dir=str(bulk / 'r2-export'))
    return arms, prov


def last_checkpoints(arms: dict) -> dict:
    """{'s0': arm name of seed 0's last checkpoint, 's1': ...}."""
    out = {}
    for seed in SEEDS:
        tags = sorted(k for k in arms if k.startswith(f'v0{seed}@'))
        if tags:
            out[seed] = tags[-1]
    return out


# ---------------------------------------------------------------- measures beyond SetScore

def _bl_o(sc) -> float:
    """BL-20 with the set offset removed (characterises only, as the harness's other _o measures)."""
    from ensomi_n.eval.measures import share
    b = sc.beats
    e, w = b['e'], b['w']
    o = sc.offset_ms if math.isfinite(sc.offset_ms) else 0.0
    ok = np.isfinite(e) & (np.abs(e - o) <= TAU + 1e-9)
    return share(ok, w) if w.sum() > 0 else math.nan


def _union_hits(sc, mmap, lo: float, hi: float) -> float:
    """Head-weighted share of the scored reference lines with any of the arm's notated beats or canonical lines within
    20 ms: what BL-20 would be if the arm's canonical fold kept every beat it states."""
    from ensomi_n.eval.measures import PRED_MARGIN_MS, nearest, share
    tm = mmap.timing(lo - PRED_MARGIN_MS, hi + PRED_MARGIN_MS)
    if not len(tm):
        return math.nan
    pts = np.union1d(tm.beats, tm.canonical().t)
    e, _ = nearest(sc.beats['t'], pts)
    w = sc.beats['w']
    return share(np.isfinite(e) & (np.abs(e) <= TAU + 1e-9), w) if w.sum() > 0 else math.nan


def attribution(m: dict) -> str:
    """Why an arm's BL-20 is lost on a span (rules fixed before any number, module constants)."""
    bl, union, lev = m.get('bl20'), m.get('union20'), m.get('level_classes') or {}
    if bl is None or not math.isfinite(bl):
        return 'unscored'
    if bl >= ATTR_PASS:
        return 'pass'
    if union is not None and math.isfinite(union) and union - bl >= ATTR_FOLD:
        return 'level: octave with the fold on the other phase'
    if (lev.get('family') or 0.0) >= ATTR_LEVEL:
        return 'level: another pulse (a 3:2, 4:3 or 5:4 family ratio)'
    blo, o = m.get('bl20_o'), m.get('offset_ms')
    if blo is not None and math.isfinite(blo) and blo >= ATTR_PASS and o is not None and math.isfinite(o):
        if 10.0 < abs(o) <= ATTR_CLOCK_MS:
            return 'whole-grid offset of clock size'
        if abs(o) > ATTR_CLOCK_MS:
            return 'whole-grid phase shift'
    if (lev.get('other') or 0.0) >= ATTR_LEVEL:
        return 'tempo off by more than 0.5 % (no level ratio)'
    return 'phase error'


def score_span(ref, mmap, spans) -> dict:
    from ensomi_n.eval import score
    from ensomi_n.v0.evaluate import free_split
    sc = score(ref, mmap, spans=spans)
    s = sc.summary()
    keep = ('bl20', 'bl10', 'bl5', 'bl20_beat', 'chance20', 'bl10_o', 'bl5_o', 'offset_ms', 'level', 'bars_start',
            'bars_same', 'bars_precision', 'bars_start_o', 'bars_same_o', 'pred_has_bars', 'free_share',
            'free_share_pred', 'boundary_recall', 'boundary_precision', 'reset_recall', 'heads', 'n_lines',
            'n_scored', 'minutes', 'edits_per_min')
    m = {k: s.get(k) for k in keep}
    m['level_classes'] = dict(sc.level_classes)
    m['bl20_o'] = _bl_o(sc)
    m['union20'] = _union_hits(sc, mmap, ref.lo_ms, ref.hi_ms)
    m.update(free_split(ref, mmap, spans))
    m['attribution'] = attribution(m)
    return m


def canonical_in(mmap, lo: float, hi: float) -> np.ndarray:
    from ensomi_n.eval.measures import PRED_MARGIN_MS
    tm = mmap.timing(lo - PRED_MARGIN_MS, hi + PRED_MARGIN_MS)
    t = tm.canonical().t if len(tm) else np.zeros(0)
    return t[(t >= lo) & (t <= hi)]


def matched_share(x: np.ndarray, y: np.ndarray, tau: float = TAU) -> float:
    """Share of the lines ``x`` with a line of ``y`` within ``tau`` ms."""
    from ensomi_n.eval.measures import nearest
    if not len(x):
        return math.nan
    e, _ = nearest(x, np.asarray(y, dtype=np.float64))
    return float(np.mean(np.isfinite(e) & (np.abs(e) <= tau + 1e-9)))


def agreement(lines: dict) -> dict:
    """Tier A: directed matched shares between every pair of arms' canonical lines on a span, agreement, groups."""
    names = list(lines)
    share = {x: {y: matched_share(lines[x], lines[y]) for y in names if y != x} for x in names}
    agree = {}
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            a, b = share[x][y], share[y][x]
            agree[f'{x}|{y}'] = bool(math.isfinite(a) and math.isfinite(b) and min(a, b) >= AGREE)
    groups, seen = [], set()
    for x in names:
        if x in seen:
            continue
        group, todo = {x}, [x]
        while todo:
            u = todo.pop()
            for v in names:
                if v not in group and agree.get(f'{u}|{v}', agree.get(f'{v}|{u}', False)):
                    group.add(v)
                    todo.append(v)
        seen |= group
        groups.append(sorted(group))
    return dict(n_lines={k: int(len(v)) for k, v in lines.items()}, matched=share, agree=agree, groups=groups)


def pulse_verdict(med: float, hn: float, hr: float, hs: float) -> str:
    """Tier B's rule (module constants), from an arm's median line spacing in notated beats and its hit shares on
    N's own lines (``hn``), R's own lines (``hr``) and the lines both readings share (``hs``).

    A continuous pulse a notated beat off R's hits N's own lines in the bars where the readings differ and misses
    the shared lines in the others, so the shared lines separate it from a restart.
    """
    if not math.isfinite(med):
        return 'no lines'
    if med <= FINE_LEVEL:
        return 'too fine to tell (lines at most 1.25 notated beats apart)'
    if not (math.isfinite(hn) and math.isfinite(hr) and math.isfinite(hs)):
        return 'not scored'
    high, low = PULSE_HIGH, PULSE_LOW
    if hs >= high and hr >= high and hn <= low:
        return 'continuous pulse (R reading)'
    if hs >= high and hn >= high and hr <= low:
        return 'restarts each bar (N reading)'
    if hs <= low and hn >= high and hr <= low:
        return 'continuous pulse a notated beat off R'
    if hs >= high and hn >= high and hr >= high:
        return 'hits both'
    return 'neither or mixed'


def pulse(ref, r_lines: np.ndarray, maps: dict) -> dict:
    """Tier B: in the scored range, the lines where N's reading (canonical lines restart every bar) and R's (a
    continuous pulse) differ, and how often each arm hits either within 20 ms.

    Two views per arm: ``native``, the beats the arm states at its own level, and ``canonical``, its canonical
    lines (its beats folded into 80-160 BPM, the phase restarting where its own bars allow, as the measures fold
    it). An arm whose lines are at most 1.25 notated beats apart hits both readings and cannot tell them apart.
    """
    from ensomi_n.eval.measures import PRED_MARGIN_MS, nearest
    lo, hi = ref.lo_ms, ref.hi_ms
    n_t = ref.lines().t
    n_t = n_t[(n_t >= lo) & (n_t <= hi)]
    r_t = np.asarray(r_lines, dtype=np.float64)
    r_t = r_t[(r_t >= lo) & (r_t <= hi)]

    def only(x, y):
        if not len(x) or not len(y):
            return x
        e, _ = nearest(x, y)
        return x[~(np.isfinite(e) & (np.abs(e) <= TAU + 1e-9))]
    n_only, r_only = only(n_t, r_t), only(r_t, n_t)
    shared = np.setdiff1d(n_t, n_only)
    notated = float(np.median(ref.timing.step_ms[(ref.timing.beats >= lo) & (ref.timing.beats <= hi)]))
    out = dict(n_lines=int(len(n_t)), r_lines=int(len(r_t)), n_only=int(len(n_only)), r_only=int(len(r_only)),
               shared=int(len(shared)), notated_beat_ms=notated, arms={})
    for arm, m in maps.items():
        tm = m.timing(lo - PRED_MARGIN_MS, hi + PRED_MARGIN_MS)
        views = {'native': tm.beats if len(tm) else np.zeros(0),
                 'canonical': tm.canonical().t if len(tm) else np.zeros(0)}
        res = {}
        for view, t in views.items():
            b = t[(t >= lo) & (t <= hi)]
            med = float(np.median(np.diff(b))) / notated if len(b) > 2 else math.nan
            hn, hr, hs = matched_share(n_only, b), matched_share(r_only, b), matched_share(shared, b)
            res[view] = dict(median_line_notated=med, hit_n_only=hn, hit_r_only=hr, hit_shared=hs,
                             verdict=pulse_verdict(med, hn, hr, hs))
        out['arms'][arm] = res
    return out


def logic_meta(path: Path) -> dict:
    from ensomi_n.mapformat.logic import summarize
    return summarize(path)


def cmd_score(a) -> None:
    from ensomi_n.v0.evaluate import eval_items
    out = Path(a.out)
    meta = json.load(open(out / 'songs.json'))
    songs = meta['songs']
    by_sha = {s['audio_sha256']: s for s in songs}
    split = {'logic': [dict(audio_sha256=s['audio_sha256'], charts=s['charts'], rec=s['rec']) for s in songs]}
    items = eval_items(split, ['logic'], Path(a.root), corpus=a.corpus, tags=a.tags)
    arms, prov = load_arms(a, songs)
    last = last_checkpoints(arms)
    _dump(out / 'arms.json', dict(arms={k: dict(prov.get(k, {}), n=len(v)) for k, v in arms.items()}, last=last))
    sys.path.insert(0, str(HERE))
    import reader_census as rc
    rows = []
    for x in items:
        s = by_sha[x['item']]
        row = dict(id=s['id'], list_id=s['list_id'], tier=s['tier'], group=s['group'], item=x['item'],
                   v0_exposure=s['v0_exposure'], key=x['key'])
        if 'error' in x:
            row['error'] = x['error']
            rows.append(row)
            _log(out, f"score {s['id']}: reference error {x['error']}")
            continue
        ref = x['ref']
        lo, hi = ref.lo_ms, ref.hi_ms
        hard = s['hard_span_ms']
        spans = {'whole': None, 'hard': [tuple(hard)] if hard else None}
        if s['group'] in x['strata']:
            spans['stratum'] = x['strata'][s['group']]
        inside = (ref.timing.beats >= lo) & (ref.timing.beats <= hi)
        steps = ref.timing.step_ms[inside]
        row.update(lo_ms=lo, hi_ms=hi, strata=sorted(x['strata']), fast57=x['fast57'], literal=x['literal'],
                   stratum_in_census=s['group'] in x['strata'], timing_chart=ref.meta.get('timing_chart'),
                   ref_notated_bpm_median=float(60000.0 / np.median(steps)) if len(steps) else None,
                   ref_meters=sorted({int(seg.meter) for seg in (ref.segments or ()) if seg.plausible}),
                   ref_red_lines=len(ref.segments or ()))
        row['logic'] = {mode: logic_meta(Path(a.logic) / 'bulk' / 'maps' / f"{s['id']}__{mode}.json") for mode in MODES}
        res = {}
        for arm, maps in arms.items():
            m = maps.get(x['item'])
            if m is None:
                continue
            res[arm] = {}
            for name, sp in spans.items():
                try:
                    res[arm][name] = score_span(ref, m, sp)
                except Exception as exc:  # noqa: BLE001
                    res[arm][name] = dict(error=f'{type(exc).__name__}: {exc}', trace=traceback.format_exc()[-300:])
        row['scores'] = res
        # Who agrees with whom on the hard span (the whole scored range for tier B), canonical lines at 20 ms.
        a0, b0 = hard if hard else (lo, hi)
        t_ref = ref.lines().t
        for cname in CLOCK_CONSTANTS:
            lines = {f'st-{mode}': canonical_in(arms[f'st-{mode}@{cname}'][x['item']], a0, b0) for mode in MODES}
            for seed, arm in last.items():
                lines[f'v0{seed}'] = canonical_in(arms[arm][x['item']], a0, b0)
            if x['item'] in arms['r2']:
                lines['r2'] = canonical_in(arms['r2'][x['item']], a0, b0)
            lines['mapper'] = t_ref[(t_ref >= a0) & (t_ref <= b0)]
            row.setdefault('agreement', {})[cname] = agreement(lines)
        if s['tier'] == 'B':
            try:
                r_t = rc.r_lines(ref.meta['timing_chart'], lo, hi)[0]
                for cname in CLOCK_CONSTANTS:
                    maps = {f'st-{mode}': arms[f'st-{mode}@{cname}'][x['item']] for mode in MODES}
                    for seed, arm in last.items():
                        maps[f'v0{seed}'] = arms[arm][x['item']]
                    if x['item'] in arms['r2']:
                        maps['r2'] = arms['r2'][x['item']]
                    row.setdefault('pulse', {})[cname] = pulse(ref, r_t, maps)
            except Exception as exc:  # noqa: BLE001
                row['pulse_error'] = f'{type(exc).__name__}: {exc}'
        rows.append(_clean(row))
        _log(out, f"score {s['id']} {s['list_id']}: {len(res)} arms")
    _dump(out / 'items.json', rows)
    _log(out, f'score: {len(rows)} songs, {sum("error" in r for r in rows)} reference errors')


# ---------------------------------------------------------------- report

def _get(row, arm, span, measure):
    try:
        v = row['scores'][arm][span][measure]
    except (KeyError, TypeError):
        return math.nan
    return math.nan if v is None else float(v)


def oracle_value(row, span, measure, cname) -> float:
    """Best of the two Smart Tempo modes on this song: an oracle (it reads the reference), never the baseline."""
    vals = [_get(row, f'st-{mode}@{cname}', span, measure) for mode in MODES]
    vals = [v for v in vals if math.isfinite(v)]
    return max(vals) if vals else math.nan


def boot_mean(v: np.ndarray, n: int = BOOT, seed: int = 0) -> dict:
    v = np.asarray([x for x in v if math.isfinite(x)], dtype=np.float64)
    if not len(v):
        return dict(value=None, lo=None, hi=None, n=0)
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return dict(value=float(v.mean()), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)), n=int(len(v)))


def arm_values(rows, arm, span, measure, cname) -> np.ndarray:
    if arm == 'st-oracle':
        return np.array([oracle_value(r, span, measure, cname) for r in rows])
    if arm.startswith('st-'):
        arm = f'{arm}@{cname}'
    return np.array([_get(r, arm, span, measure) for r in rows])


def _nanmean(x, axis=None):
    """``np.nanmean`` without the warning on an all-NaN or empty slice (it returns NaN there)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(np.asarray(x, dtype=np.float64), axis=axis)


def v0_mean_values(rows, last: dict, span, measure) -> np.ndarray:
    vals = np.array([[_get(r, last[s], span, measure) for s in sorted(last)] for r in rows])
    return _nanmean(vals, axis=1) if vals.size else np.zeros(0)


def table_block(rows, last, span, measure, cname) -> dict:
    """Per arm: mean with a song bootstrap; paired differences against R2 and v0 (held-out songs only for v0)."""
    out = {'n_songs': len(rows), 'arms': {}, 'paired': {}}
    names = ['st-constant', 'st-variable', 'st-oracle'] + [f'v0{s}' for s in sorted(last)] + ['v0mean', 'r2']
    vals = {}
    for arm in names:
        if arm.startswith('v0') and arm != 'v0mean':
            vals[arm] = np.array([_get(r, last[arm[2:]], span, measure) for r in rows])
        elif arm == 'v0mean':
            vals[arm] = v0_mean_values(rows, last, span, measure)
        else:
            vals[arm] = arm_values(rows, arm, span, measure, cname)
        out['arms'][arm] = boot_mean(vals[arm])
    held = np.array([r['v0_exposure'] == 'held-out' for r in rows], dtype=bool)
    for st in ('st-constant', 'st-variable', 'st-oracle'):
        d = vals[st] - vals['r2']
        out['paired'][f'{st} - r2'] = boot_mean(d)
        for v0 in ('v0mean',) + tuple(f'v0{s}' for s in sorted(last)):
            out['paired'][f'{st} - {v0} (held-out)'] = boot_mean(np.where(held, vals[st] - vals[v0], np.nan))
            out['paired'][f'{st} - {v0} (trained-on, apart)'] = boot_mean(np.where(~held, vals[st] - vals[v0], np.nan))
    if len(last) == 2:
        s0, s1 = (vals[f'v0{s}'] for s in sorted(last))
        out['v0_seed_spread'] = abs(float(_nanmean(s0)) - float(_nanmean(s1))) if np.isfinite(s0).any() else None
    return out


def ckpt_spread(rows, arms_all: list[str], span, measure) -> dict:
    out = {}
    for seed in SEEDS:
        names = [a for a in arms_all if a.startswith(f'v0{seed}@')]
        means = [_nanmean([_get(r, a, span, measure) for r in rows]) for a in names]
        means = [m for m in means if math.isfinite(m)]
        out[seed] = float(max(means) - min(means)) if len(means) > 1 else None
    return out


MEASURES = ('bl20', 'bl20_nonfree', 'bl10', 'offset_ms', 'level', 'bars_start', 'bars_same', 'bl20_o', 'union20')


SYSTEMS = {'mapper': ('mapper',), 'v0': ('v0s0', 'v0s1'), 'r2': ('r2',), 'st': ('st-constant', 'st-variable')}


def system_agreement(ag: dict) -> dict:
    """Which of the four readings (the chart's mapper, v0 either seed, R2, Smart Tempo either mode) agree on a span.

    Two readings agree when any of their arms agree (`agreement`, `AGREE`). ``all_disagree`` is true when no two of
    the four do.
    """
    agree = ag.get('agree') or {}

    def pair(x, y):
        return agree.get(f'{x}|{y}', agree.get(f'{y}|{x}', False))
    names = list(SYSTEMS)
    out = {}
    for i, u in enumerate(names):
        for v in names[i + 1:]:
            out[f'{u}|{v}'] = any(pair(x, y) for x in SYSTEMS[u] for y in SYSTEMS[v])
    return dict(pairs=out, all_disagree=not any(out.values()),
                chart_alone=not any(out[f'mapper|{v}'] for v in names[1:]))


def r2_null(rows, files: dict) -> dict:
    """R2 recomputed here (BeatThis final0 on the CPU) against the export the v0 analysis used, on songs with both."""
    out = {}
    for span in ('whole', 'hard'):
        d = []
        for r in rows:
            src = files.get(r['item'], '')
            if 'n1-cache/r2-export' not in src:
                continue
            a, b = _get(r, 'r2', span, 'bl20'), _get(r, 'r2-fresh', span, 'bl20')
            if math.isfinite(a) and math.isfinite(b):
                d.append(b - a)
        d = np.array(d)
        out[span] = dict(n=int(len(d)), mean=float(d.mean()) if len(d) else None,
                         max_abs=float(np.abs(d).max()) if len(d) else None,
                         nonzero=int(np.sum(np.abs(d) > 1e-9)))
    return out


def cmd_report(a) -> None:
    out = Path(a.out)
    rows = json.load(open(out / 'items.json'))
    errors = [dict(id=r['id'], error=r['error']) for r in rows if 'error' in r]
    rows = [r for r in rows if 'scores' in r]
    arms_meta = json.load(open(out / 'arms.json'))
    last = arms_meta['last']
    all_arms = sorted(arms_meta['arms'])
    songs_meta = {s['id']: s for s in json.load(open(out / 'songs.json'))['songs']}
    repro_path = out / 'v0-reproduction.json'
    repro = json.load(open(repro_path)) if repro_path.exists() else {}
    tiers = {t: [r for r in rows if r['tier'] == t] for t in 'ABC'}
    rep = dict(clock_rule=dict(constants=CLOCK_CONSTANTS, sources=CLOCK_SOURCES,
                               lead_ms={s['id']: s['lead_ms'] for s in songs_meta.values()},
                               lead_rule='osu-lead-v1 for the ffmpeg decoder (ensomi_n.features.clock.file_lead)'),
               last_checkpoints=last, margins=MARGINS, margins_n_below_10=MARGIN_N_BELOW_10,
               rules=dict(agree=AGREE, pulse_high=PULSE_HIGH, pulse_low=PULSE_LOW, fine_level=FINE_LEVEL,
                          attr_pass=ATTR_PASS, attr_fold=ATTR_FOLD, attr_level=ATTR_LEVEL, attr_clock_ms=ATTR_CLOCK_MS,
                          boot=BOOT),
               counts=dict(songs=len(rows) + len(errors), scored=len(rows), reference_errors=errors,
                           per_tier={t: len(v) for t, v in tiers.items()},
                           held_out={t: sum(r['v0_exposure'] == 'held-out' for r in v) for t, v in tiers.items()},
                           arms={k: v.get('n') for k, v in arms_meta['arms'].items()}),
               v0_reproduction=dict(pairs=sum(len(v) for v in repro.values()),
                                    worst_ms=max((d['max_abs_ms'] for v in repro.values() for d in v.values()
                                                  if d.get('max_abs_ms') is not None), default=None),
                                    count_or_downbeat_differences=sum(1 for v in repro.values() for d in v.values()
                                                                      if d.get('max_abs_ms') is None
                                                                      or not d.get('downbeats_equal'))),
               r2_recompute=r2_null(rows, (arms_meta['arms'].get('r2') or {}).get('files', {})),
               tiers={}, strata_c={}, songs=[])
    for tier, rs in tiers.items():
        rep['tiers'][tier] = {}
        for span in ('whole', 'hard'):
            rep['tiers'][tier][span] = {cname: {m: table_block(rs, last, span, m, cname) for m in MEASURES}
                                        for cname in CLOCK_CONSTANTS}
            rep['tiers'][tier][span]['v0_checkpoint_spread'] = {m: ckpt_spread(rs, all_arms, span, m)
                                                                for m in ('bl20', 'bl20_nonfree')}
    for st in ('S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9'):
        rs = [r for r in tiers['C'] if r['group'] == st]
        rep['strata_c'][st] = {span: {cname: {m: table_block(rs, last, span, m, cname)
                                              for m in ('bl20', 'bl20_nonfree', 'offset_ms', 'level')}
                                      for cname in CLOCK_CONSTANTS}
                               for span in ('hard', 'stratum', 'whole')}
    keep = MEASURES + ('attribution', 'chance20', 'level_classes', 'free_w', 'bl5', 'bars_precision')
    for r in rows:
        meta = songs_meta.get(r['id'], {})
        song = dict(id=r['id'], list_id=r['list_id'], tier=r['tier'], group=r['group'], v0_exposure=r['v0_exposure'],
                    v0_exposure_why=meta.get('v0_exposure_why'), strata=r.get('strata'),
                    stratum_in_census=r.get('stratum_in_census'), hard_span_ms=meta.get('hard_span_ms'),
                    ref_notated_bpm_median=r.get('ref_notated_bpm_median'), ref_meters=r.get('ref_meters'),
                    logic={m: {k: (r.get('logic') or {}).get(m, {}).get(k) for k in
                               ('median_bpm', 'tempo_min', 'tempo_max', 'signatures', 'beats', 'bars', 'ax')}
                           for m in MODES},
                    values={})
        for span in ('whole', 'hard', 'stratum'):
            for arm in sorted(r['scores']):
                if arm.startswith('v0') and arm not in last.values():
                    continue
                m = r['scores'][arm].get(span)
                if not m or 'error' in m:
                    if m and 'error' in m:
                        song.setdefault('score_errors', {})[f'{arm}/{span}'] = m['error']
                    continue
                song['values'].setdefault(span, {})[arm] = {k: m.get(k) for k in keep}
        if 'agreement' in r:
            song['agreement'] = r['agreement']
            song['systems'] = {c: system_agreement(ag) for c, ag in r['agreement'].items()}
        if 'pulse' in r:
            song['pulse'] = r['pulse']
        if 'pulse_error' in r:
            song['pulse_error'] = r['pulse_error']
        rep['songs'].append(song)
    rep['all_disagree'] = [dict(list_id=s['list_id'], group=s['group'], clock=c, pairs=v['pairs'])
                           for s in rep['songs'] for c, v in (s.get('systems') or {}).items() if v['all_disagree']]
    rep['low_bl20'] = low_bl20(rows, last)
    _dump(out / 'results.json', _clean(rep))
    (out / 'RESULTS.md').write_text(markdown(rep))
    _log(out, 'report: results.json and RESULTS.md written')


def low_bl20(rows, last) -> list[dict]:
    """Songs where no arm reaches BL-20 0.5 against the chart on the hard span (the whole song for tier B)."""
    out = []
    for r in rows:
        span = 'hard' if r['tier'] != 'B' else 'whole'
        for cname in CLOCK_CONSTANTS:
            vals = {'st-constant': _get(r, f'st-constant@{cname}', span, 'bl20'),
                    'st-variable': _get(r, f'st-variable@{cname}', span, 'bl20'),
                    'r2': _get(r, 'r2', span, 'bl20')}
            for seed, arm in last.items():
                vals[f'v0{seed}'] = _get(r, arm, span, 'bl20')
            finite = [v for v in vals.values() if math.isfinite(v)]
            if finite and max(finite) < 0.5:
                out.append(dict(id=r['id'], list_id=r['list_id'], tier=r['tier'], group=r['group'], clock=cname,
                                span=span, bl20=vals))
    return out


def _f(v, nd=3) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return '–'
    if isinstance(v, (int, float)):
        return f'{v:.{nd}f}'
    return str(v)


def _ci(d) -> str:
    if not d or d.get('value') is None:
        return '–'
    return f"{d['value']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}] (n {d['n']})"


def _dci(d) -> str:
    if not d or d.get('value') is None:
        return '–'
    return f"{d['value']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] (n {d['n']})"


def markdown(rep) -> str:
    C = rep['clock_rule']['constants']
    last = rep['last_checkpoints']
    L = ['# Logic Pro Smart Tempo against the charts, beside R2 and N1 v0', '',
         'Generated by `scripts/n1/logic_baseline.py report`. Every number characterises: no measure here has been '
         "shown to agree with the human's ear, and the human's listening decides. `ST oracle` is the better of the two "
         'Smart Tempo modes on each song, chosen by reading the chart: an oracle, never the baseline. Comparisons with '
         'v0 use only songs v0 never trained on; trained-on songs are reported apart.', '',
         '## Clock rule', '',
         'Smart Tempo times are on the canonical FFmpeg decode (time zero at its first sample). The reader adds the '
         "ffmpeg lead of N's osu! clock rule (`osu-lead-v1`; read from each file's header: "
         f"{sorted(set(rep['clock_rule']['lead_ms'].values()))} ms over these songs). One decoded-to-osu! constant c "
         'then goes through the harness guard, as for raw BeatThis, an unfitted tracker on a decoded clock; no '
         'per-song offset is fitted. The sources give two constants and no way to choose; both are scored:', '']
    for k, v in C.items():
        L.append(f'- `{k}`: c = {v} ms: {rep["clock_rule"]["sources"][k]}.')
    L += ['', 'v0 and R2 are on osu!\'s clock and scored as they are: v0 at the last checkpoint of each seed '
          f'({", ".join(f"{k} `{v}`" for k, v in last.items())}), R2 as exported.', '',
          '## Counts and checks', '',
          f"- Songs scored: {rep['counts']['scored']} of {rep['counts']['songs']} (per tier {rep['counts']['per_tier']}); "
          f"held out from v0's training per tier: {rep['counts']['held_out']}; reference errors: "
          f"{rep['counts']['reference_errors'] or 'none'}.",
          f"- Maps per arm: {rep['counts']['arms']}.",
          f"- v0 maps decoded here against those the v0 analysis kept: {rep['v0_reproduction']['pairs']} (checkpoint, "
          f"song) pairs, largest beat difference {_f(rep['v0_reproduction']['worst_ms'], 4)} ms, "
          f"{rep['v0_reproduction']['count_or_downbeat_differences']} with another beat count or other downbeats.",
          f"- R2 recomputed here against the export the v0 analysis used (BL-20 difference): {rep['r2_recompute']}.",
          '']
    for tier, spans in rep['tiers'].items():
        L += [f'## Tier {tier}', '']
        for span in ('whole', 'hard'):
            if tier == 'B' and span == 'hard':
                continue
            for cname in C:
                blocks = spans[span][cname]
                L += [f'### Tier {tier}, {"whole song" if span == "whole" else "hard span"}, clock `{cname}`', '',
                      'Song means with 95 % song-bootstrap intervals (2,000 resamples).', '',
                      '| Measure | ST constant | ST variable | ST oracle | v0 s0 | v0 s1 | R2 |',
                      '| --- | --- | --- | --- | --- | --- | --- |']
                for m, blk in blocks.items():
                    A = blk['arms']
                    L.append(f"| {m} | {_ci(A.get('st-constant'))} | {_ci(A.get('st-variable'))} | "
                             f"{_ci(A.get('st-oracle'))} | {_ci(A.get('v0s0'))} | {_ci(A.get('v0s1'))} | {_ci(A.get('r2'))} |")
                L += ['', '| Paired difference, BL-20 | mean [95 %] (n songs) |', '| --- | --- |']
                for k, d in blocks['bl20']['paired'].items():
                    if ' - v0s' not in k:            # per-seed differences are in results.json
                        L.append(f'| {k} | {_dci(d)} |')
                L += ['', f"v0 seed spread of the mean BL-20 here: {_f(blocks['bl20'].get('v0_seed_spread'))}; "
                      f"checkpoint spread per seed: {spans[span]['v0_checkpoint_spread']['bl20']}.", '']
    L += ['## Tier C per stratum, BL-20', '',
          'Three songs per stratum. `hard` is the listening clip from its hard part to its end; `stratum` is the '
          "census's spans of that stratum on the chart. m is the v0 − R2 margin of the v0 analysis (panel / val-n1 "
          'hard, N reference, full spans), a scale only: it was measured for v0 against R2 on other songs.', '']
    for span in ('hard', 'stratum'):
        for cname in C:
            L += [f'### {span}, clock `{cname}`', '',
                  '| Stratum | n | ST constant | ST variable | v0 mean | R2 | ST const − R2 | ST var − R2 | m |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
            for st, d in rep['strata_c'].items():
                blk = d[span][cname]['bl20']
                A, P = blk['arms'], blk['paired']
                L.append(f"| {st} | {blk['n_songs']} | {_ci(A.get('st-constant'))} | {_ci(A.get('st-variable'))} | "
                         f"{_ci(A.get('v0mean'))} | {_ci(A.get('r2'))} | {_dci(P.get('st-constant - r2'))} | "
                         f"{_dci(P.get('st-variable - r2'))} | {rep['margins']['panel'].get(st)} / "
                         f"{rep['margins']['val_hard'].get(st)} |")
            L.append('')
    L += ['## Per song', '',
          'BL-20 on the hard span (tier B: the whole song). Chart: the notated BPM median and meters. Smart Tempo: '
          'median BPM and signature, constant / variable. Attribution (clock `c20`, constant; variable) follows the '
          'rules in `results.json` `rules`.', '',
          '| Song | Stratum | v0 | Chart | ST stated | ST const c20 / c287 | ST var c20 / c287 | v0 s0 / s1 | R2 | Attribution |',
          '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for s in rep['songs']:
        span = 'whole' if s['tier'] == 'B' else 'hard'
        V = s['values'].get(span, {})

        def g(arm, k='bl20'):
            return (V.get(arm) or {}).get(k)
        lg = s['logic']
        stated = ' / '.join(f"{_f(lg[m].get('median_bpm'), 1)} {','.join(lg[m].get('signatures') or [])}" for m in MODES)
        L.append(f"| {s['list_id']} | {s['group']} | {s['v0_exposure']} | {_f(s.get('ref_notated_bpm_median'), 1)} "
                 f"{s.get('ref_meters')} | {stated} | {_f(g('st-constant@c20'))} / {_f(g('st-constant@c287'))} | "
                 f"{_f(g('st-variable@c20'))} / {_f(g('st-variable@c287'))} | {_f(g(last.get('s0', '')))} / "
                 f"{_f(g(last.get('s1', '')))} | {_f(g('r2'))} | {g('st-constant@c20', 'attribution') or '–'}; "
                 f"{g('st-variable@c20', 'attribution') or '–'} |")
    L += ['', '## Who agrees on each hard span (canonical lines, 20 ms)', '',
          f"Two arms agree when each has at least {rep['rules']['agree']} of its canonical lines in the span matched "
          'by the other within 20 ms. Groups join arms that agree, directly or through another arm.', '']
    for s in rep['songs']:
        for cname, ag in (s.get('agreement') or {}).items():
            sysv = s['systems'][cname]
            L.append(f"- {s['list_id']} `{cname}`: groups {ag['groups']}; lines {ag['n_lines']}"
                     + ('; **no two readings agree**' if sysv['all_disagree'] else '')
                     + ('; the chart agrees with none' if sysv['chart_alone'] and not sysv['all_disagree'] else ''))
    L += ['', '## Tier B: continuous pulse or restart each bar', '',
          "Lines where N's reading (canonical lines restart every bar) and R's (a continuous pulse) differ, and the share "
          "of each hit within 20 ms by an arm's stated beats (`native`) and by its canonical lines (`canonical`); the "
          'median line spacing in notated beats.', '']
    for s in rep['songs']:
        for cname, p in (s.get('pulse') or {}).items():
            L.append(f"- {s['list_id']} `{cname}` (N-only {p['n_only']}, R-only {p['r_only']} lines):")
            for k, views in p['arms'].items():
                L.append(f'  - {k}: ' + '; '.join(
                    f"{v} {d['verdict']} (N-only hit {_f(d['hit_n_only'], 2)}, R-only hit {_f(d['hit_r_only'], 2)}, "
                    f"shared hit {_f(d['hit_shared'], 2)}, spacing {_f(d['median_line_notated'], 2)})"
                    for v, d in views.items()))
        if s.get('pulse_error'):
            L.append(f"- {s['list_id']}: pulse check failed: {s['pulse_error']}")
    L += ['', '## Songs where no two readings agree, and songs where no arm reaches BL-20 0.5', '']
    for d in rep['all_disagree']:
        L.append(f"- no two agree: {d['list_id']} ({d['group']}, `{d['clock']}`)")
    for d in rep['low_bl20']:
        L.append(f"- BL-20 < 0.5 for every arm: {d['list_id']} ({d['group']}, `{d['clock']}`, {d['span']}): "
                 + ', '.join(f'{k} {_f(v)}' for k, v in d['bl20'].items()))
    return '\n'.join(L) + '\n'


# ---------------------------------------------------------------- main

def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=('prep', 'cache', 'v0maps', 'score', 'report'))
    p.add_argument('--root', default='.')
    p.add_argument('--logic', default='artifacts/reports/logic-tempo-baseline-20261020')
    p.add_argument('--out', default='artifacts/reports/logic-tempo-baseline-20261020/analysis')
    p.add_argument('--bulk', default='artifacts/bulk/logic-tempo-baseline-20261020')
    p.add_argument('--r0', default='artifacts/n1-night-20261018/runs/r0-20261019a')
    p.add_argument('--curated', default='artifacts/n1-night-20261018/runs/v0data2-20261019/curated.json')
    p.add_argument('--tune-dir', default='artifacts/n1-night-20261018/runs/v0-s0-20261019/eval')
    p.add_argument('--bulk-runs', default='artifacts/bulk/n1-runs')
    p.add_argument('--stored-maps', default='artifacts/bulk/n1-v0-analysis')
    p.add_argument('--r2-stored', default='artifacts/bulk/n1-cache/r2-export')
    p.add_argument('--corpus', default='data/r2-corpus.parquet')
    p.add_argument('--tags', default='artifacts/n1-inventory-20261014/tags.json')
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args(argv)
    global LOG_DIR
    LOG_DIR = Path(a.bulk)
    dict(prep=cmd_prep, cache=cmd_cache, v0maps=cmd_v0maps, score=cmd_score, report=cmd_report)[a.cmd](a)


if __name__ == '__main__':
    main()
