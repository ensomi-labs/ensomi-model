"""Ardour's detectors and openDAW's TempoCNN as reference baselines, beside Smart Tempo, v0 and R2.

The question (research notes, ``n1-eval-20261018`` ``d-reference-baselines``): the two open tools run on the 48
songs of the Smart Tempo baseline (``smart-tempo-baseline-20261020`` ``o-st-scored``) and are scored as Smart
Tempo was, so all of them sit in one table with N1 v0 and R2. Every number characterises: no measure here has
been shown to agree with the human's ear.

Arms (`ARMS`; every one reads the canonical decode the Smart Tempo capture read, its left channel, at its own
sample rate, and none reads a chart unless its name says oracle):

- ``qm-bbt``: Ardour 9.8's bundled ``qm-barbeattracker`` with its defaults, 4 beats per bar: beats and bars.
- ``qm-bbt-oracle``: the same tracker run at each beats-per-bar the chart states, the best run per song and
  measure. It reads the chart twice; it is an oracle and never a baseline.
- ``qm-tempo``: Ardour 9.8's bundled ``qm-tempotracker`` with its defaults: beats, no bars.
- ``mbpm-clip``: MiniBPM over the whole file, rounded by Ardour to whole 4/4 bars over the file, the tempo of a
  clip trigger holding the dropped file; its grid starts at the clip's first sample, where the trigger plays it.
  The power-of-two bar count applies only to 18 bars or fewer (``obtools.ardour_rounded_tempo``); every song
  here is longer, so it moves the beat count over the song by at most 2.5 beats.
- ``mbpm-raw``: MiniBPM's estimate before Ardour rounds it; ``mbpm-source``: MiniBPM on ten seconds from the
  middle, rounded over those ten seconds, the tempo Ardour stores on a recording or an embedded file. Their
  grids also start at the file's first sample; no host lays them that way, so their beat-line terms show only
  what one tempo without a phase gives.
- ``tcnn``: openDAW's TempoCNN task. openDAW only shows the BPM; its grid is a user's step (the BPM as the project
  tempo, the region at the timeline's start, 4/4), not the host's.

The BPM-only arms state no phase. Their tempo is scored by the level term (0.5 %) and by `tempo_terms` (the
tempo-estimation literature's accuracies at 4 %), both blind to the phase; their beat-line terms score the grid
named above.

Clock rule: as for Smart Tempo (``logic_baseline.py``), the times are on the canonical decode; the reader adds the
osu! clock rule's lead (0 ms on all 48 files) and one decoded-to-osu! constant goes through the harness guard,
c20 = -20.0 ms and c287 = -28.7 ms, both scored, one per run, never per song.

Subcommands, in order (run from the project root with ``PYTHONPATH=n/src``): ``prep``, ``preflight``, the tool
runs ``ardour``, ``minibpm`` and ``tempocnn``, ``status``, ``score``, ``report``. ``CODEX_BRIEF.md`` beside this
file is the procedure on the Mac.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import logic_baseline as lb  # noqa: E402
import obtools as tools  # noqa: E402

CLOCK_CONSTANTS = lb.CLOCK_CONSTANTS
ARMS = {
    'qm-bbt': dict(tool='ardour-vamp', kind='beats', grid='stated by the tool', oracle=False,
                   what="Ardour 9.8's qm-barbeattracker, defaults (4 beats per bar), left channel"),
    'qm-tempo': dict(tool='ardour-vamp', kind='beats', grid='stated by the tool; no bars', oracle=False,
                     what="Ardour 9.8's qm-tempotracker, defaults, left channel"),
    'mbpm-clip': dict(tool='ardour-minibpm', kind='tempo', grid="the host's: 4/4 from the clip's first sample",
                      oracle=False, what="MiniBPM on the whole file, rounded by Ardour to whole 4/4 bars (clip trigger)"),
    'mbpm-raw': dict(tool='ardour-minibpm', kind='tempo', grid='4/4 from the file start; no host lays it so',
                     oracle=False, what="MiniBPM's own estimate on the whole file, before Ardour's rounding"),
    'mbpm-source': dict(tool='ardour-minibpm', kind='tempo', grid='4/4 from the file start',
                        oracle=False, what='MiniBPM on 10 s from the middle, rounded over 10 s (recorded or embedded source)'),
    'tcnn': dict(tool='opendaw-tempocnn', kind='tempo', grid="a user's step: 4/4 from the timeline start",
                 oracle=False, what="openDAW's TempoCNN task on the whole file, left channel; one integer BPM"),
}
ORACLE = 'qm-bbt-oracle'
ORACLE_WHAT = 'best qm-barbeattracker run per song and measure over the beats-per-bar values the chart states'
BEAT_ARMS = ('qm-bbt', 'qm-tempo')
TEMPO_ARMS = ('mbpm-clip', 'mbpm-raw', 'mbpm-source', 'tcnn')
OPEN_ARMS = ('qm-bbt', ORACLE, 'qm-tempo', 'mbpm-clip', 'mbpm-raw', 'mbpm-source', 'tcnn')
OLD_ARMS = ('st-constant', 'st-variable', 'v0s0', 'v0s1', 'v0mean', 'r2')
TOOL_RUNS = {'ardour': ('qm-bbt', 'qm-tempo'), 'minibpm': ('mbpm-clip', 'mbpm-raw', 'mbpm-source'),
             'tempocnn': ('tcnn',)}
ACC_TOL = 0.04
ACC2_FACTORS = (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0)
BPB_RANGE = (2, 16)          # qm-barbeattracker's beats-per-bar range (BarBeatTrack.cpp)
#: Stopping rule: a tool whose runs fail on more than this many of the 48 songs is not scored; the report says so.
MAX_FAILED = 4
PROBES = {'P44': 44100, 'P48': 48000}
SYSTEMS_OPEN = dict(lb.SYSTEMS, **{'qm-bbt': ('qm-bbt',), 'qm-tempo': ('qm-tempo',)})


def _dump(path: Path, obj) -> None:
    lb._dump(Path(path), obj)


def _load(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _log(a, msg: str) -> None:
    lb.LOG_DIR = Path(a.bulk)
    lb._log(Path(a.out), msg)


# ---------------------------------------------------------------- prep

def oracle_bpbs(meters) -> list[int]:
    """The beats-per-bar values the meter oracle runs: the chart's meters, clamped to the tracker's range."""
    lo, hi = BPB_RANGE
    return sorted({min(hi, max(lo, int(m))) for m in meters or []})


def canonical_record(sid: str, manifest_in: dict, captures: dict) -> dict:
    """What the Smart Tempo capture recorded about song ``sid``'s canonical decode: the expected PCM hash, frames,
    rate and channels, from the manifest and from every capture record, which must agree."""
    conv = manifest_in.get('conversion') or {}
    want = dict(pcm_sha256=str(conv.get('decoded_pcm_sha256', '')).replace('SHA256=', '').lower(),
                frames=int(conv.get('sample_count', manifest_in.get('analysis_input_sample_count', -1))),
                sample_rate_hz=int((conv.get('audio') or {}).get('stream', {}).get('sample_rate', -1)),
                channels=int((conv.get('audio') or {}).get('stream', {}).get('channels', -1)))
    problems = []
    for mode, cap in captures.items():
        au = cap.get('audio') or {}
        got = dict(pcm_sha256=str(au.get('pcm_sha256', '')).lower(), frames=int(au.get('sample_count', -1)),
                   sample_rate_hz=int(au.get('sample_rate_hz', -1)), channels=int(au.get('channels', -1)))
        if got != want:
            problems.append(f'{sid}: the {mode} capture record {got} differs from the manifest {want}')
        if cap.get('canonical_pcm_matches') is not True:
            problems.append(f'{sid}: the {mode} capture record does not say canonical_pcm_matches')
    return dict(expected=want, problems=problems)


def check_wav(path: Path, expected: dict) -> tuple[dict, list[str]]:
    """The canonical WAV against the capture's record: 32-bit float, frames, rate, channels and PCM sha256."""
    info = tools.wav_info(path)
    problems = []
    if not (info['tag'] == tools.FLOAT and info['bits_per_sample'] == 32):
        problems.append(f"{path}: format {info['tag']} at {info['bits_per_sample']} bits, expected 32-bit float")
    for k in ('frames', 'sample_rate_hz', 'channels', 'pcm_sha256'):
        if info[k] != expected[k]:
            problems.append(f'{path}: {k} {info[k]} != recorded {expected[k]}')
    return info, problems


def cmd_prep(a) -> None:
    out, bulk, logic, logic_out = Path(a.out), Path(a.bulk), Path(a.logic), Path(a.logic_out)
    songs = _load(logic_out / 'songs.json')['songs']
    items = {r['id']: r for r in _load(logic_out / 'items.json')}
    manifest = {r['id']: r for r in _load(logic / 'bulk' / 'manifest.json')['inputs']}
    jobs, records, problems = [], [], []
    for s in songs:
        sid = s['id']
        caps = {}
        for mode in lb.MODES:
            p = logic / 'bulk' / 'raw-resu' / f'{sid}__{mode}.capture.json'
            if p.is_file():
                caps[mode] = _load(p)
            else:
                problems.append(f'{sid}: capture record {p} missing')
        rec = canonical_record(sid, manifest[sid], caps)
        problems += rec['problems']
        wav = Path(a.wav_dir) / f'{sid}.wav' if a.wav_dir else Path(manifest[sid]['prepared_input_path'])
        info, bad = ({}, [f'{sid}: canonical WAV {wav} not found']) if not wav.is_file() else check_wav(wav, rec['expected'])
        problems += bad
        meters = (items.get(sid) or {}).get('ref_meters') or []
        bpbs = oracle_bpbs(meters)
        jobs.append(dict(id=sid, list_id=s['list_id'], kind='song', wav=str(wav), rate=rec['expected']['sample_rate_hz'],
                         channels=rec['expected']['channels'], frames=rec['expected']['frames'],
                         pcm_sha256=rec['expected']['pcm_sha256'], lead_ms=s['lead_ms'], oracle_bpbs=bpbs,
                         bpbs=sorted(set([4] + bpbs))))
        records.append(dict(id=sid, list_id=s['list_id'], wav=str(wav), expected=rec['expected'],
                            found={k: info.get(k) for k in ('tag', 'format_tag', 'bits_per_sample', 'frames',
                                                            'sample_rate_hz', 'channels', 'pcm_sha256')},
                            ok=not bad and not rec['problems'], ref_meters=meters, oracle_bpbs=bpbs))
    for pid, rate in PROBES.items():
        x, truth = tools.click_probe(rate)
        p = bulk / 'probes' / f'{pid}.wav'
        p.parent.mkdir(parents=True, exist_ok=True)
        tools.write_wav_float(p, x, rate, extensible=True)
        info = tools.wav_info(p)
        _dump(bulk / 'probes' / f'{pid}.truth.json', dict(id=pid, sample_rate=rate, bpm=tools.PROBE_BPM,
                                                          beats_s=truth.tolist()))
        jobs.append(dict(id=pid, list_id=pid, kind='probe', wav=str(p), rate=rate, channels=2, frames=info['frames'],
                         pcm_sha256=info['pcm_sha256'], lead_ms=0.0, oracle_bpbs=[4], bpbs=[4]))
    _dump(bulk / 'jobs.json', jobs)
    _dump(out / 'open-songs.json', dict(songs=records, problems=problems,
                                        wav_source=a.wav_dir or 'manifest prepared_input_path'))
    _log(a, f'prep: {len(records)} songs, {sum(r["ok"] for r in records)} canonical WAVs verified, '
            f'{len(problems)} problems; {len(PROBES)} probes written')
    for p in problems:
        _log(a, 'PROBLEM ' + p)
    if problems:
        raise SystemExit(1)


# ---------------------------------------------------------------- preflight

def lb_namespace(a, out: Path) -> argparse.Namespace:
    """The arguments ``logic_baseline.py score`` and ``report`` take, pointing at the Smart Tempo run's inputs."""
    return argparse.Namespace(root=a.root, logic=a.logic, out=str(out), bulk=a.logic_bulk, r0=a.r0,
                              curated=a.curated, tune_dir=a.tune_dir, bulk_runs=a.bulk_runs,
                              stored_maps=a.stored_maps, r2_stored=a.r2_stored, corpus=a.corpus, tags=a.tags,
                              threads=1)


def json_diff(x, y, path: str = '', out=None, tol: float = 1e-9) -> list:
    """Differences between two JSON values as ``(kind, path, x, y)``, numbers equal within ``tol``.

    ``kind`` is ``string`` when both sides are strings (a path written another way, say) and ``value`` for every
    other difference: a number, a boolean, a null, a missing key or a list of another length.
    """
    out = [] if out is None else out
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            if k not in x or k not in y:
                out.append(('value', f'{path}/{k}', x.get(k, '<absent>'), y.get(k, '<absent>')))
            else:
                json_diff(x[k], y[k], f'{path}/{k}', out, tol)
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            out.append(('value', f'{path}/len', len(x), len(y)))
        for i, (u, v) in enumerate(zip(x, y)):
            json_diff(u, v, f'{path}/{i}', out, tol)
    elif isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool) \
            and not isinstance(y, bool):
        if not (abs(float(x) - float(y)) <= tol or (math.isnan(float(x)) and math.isnan(float(y)))):
            out.append(('value', path, x, y))
    elif isinstance(x, str) and isinstance(y, str):
        if x != y:
            out.append(('string', path, x, y))
    elif x != y:
        out.append(('value', path, x, y))
    return out


def cmd_preflight(a) -> None:
    """Re-score the Smart Tempo run's arms from its inputs with ``logic_baseline.py`` and compare with its synced
    ``results.json`` and ``items.json``. Any difference stops the run: new arms are added only to a reproduced
    baseline."""
    out, logic_out = Path(a.out), Path(a.logic_out)
    pre = out / 'preflight'
    pre.mkdir(parents=True, exist_ok=True)
    for name in ('songs.json', 'v0-reproduction.json'):
        if (logic_out / name).is_file():
            shutil.copyfile(logic_out / name, pre / name)
    ns = lb_namespace(a, pre)
    lb.LOG_DIR = Path(a.bulk) / 'preflight-log'
    lb.cmd_score(ns)
    lb.cmd_report(ns)
    res = {}
    for name in ('results.json', 'items.json'):
        d = json_diff(_load(logic_out / name), _load(pre / name))
        values = [x for x in d if x[0] == 'value']
        strings = [x for x in d if x[0] == 'string']
        # Equal means every number, flag and structure is; differing strings (paths) are listed, not decisive.
        res[name] = dict(equal=not values, n_differences=len(values), n_string_differences=len(strings),
                         first=[list(map(str, x)) for x in values[:100]],
                         strings=[list(map(str, x)) for x in strings[:100]])
    _dump(pre / 'compare.json', res)
    _log(a, f"preflight: results.json {'equal' if res['results.json']['equal'] else 'DIFFERS'} "
            f"({res['results.json']['n_differences']}), items.json "
            f"{'equal' if res['items.json']['equal'] else 'DIFFERS'} ({res['items.json']['n_differences']})")
    if not all(r['equal'] for r in res.values()):
        raise SystemExit('preflight: the Smart Tempo run does not reproduce; see preflight/compare.json')


# ---------------------------------------------------------------- tool runs

def _jobs(a) -> list[dict]:
    jobs = _load(Path(a.bulk) / 'jobs.json')
    if a.ids:
        want = set(a.ids.split(','))
        jobs = [j for j in jobs if j['id'] in want]
    return jobs


def _audio(job: dict) -> dict:
    return dict(pcm_sha256=job['pcm_sha256'], frames=job['frames'], sample_rate_hz=job['rate'],
                channels=job['channels'], channel_used=0, wav=job['wav'])


def _write_outputs(a, job: dict, outs: dict) -> None:
    for arm, d in outs.items():
        _dump(Path(a.bulk) / 'maps' / a.tag / arm / f"{job['id']}.json", d)


def _run(cmd: list, timeout: float, log: Path) -> dict:
    """Run one tool process with a timeout; its output goes to ``log`` (bulk), its outcome is returned."""
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        rc, so, se, timed_out = p.returncode, p.stdout, p.stderr, False
    except subprocess.TimeoutExpired as exc:
        rc, timed_out = None, True
        so = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or '')
        se = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or '')
    wall = time.time() - t0
    log.write_text(f'$ {" ".join(map(str, cmd))}\nrc {rc} timed_out {timed_out} wall {wall:.1f} s\n'
                   f'--- stdout\n{so}\n--- stderr\n{se}\n')
    return dict(rc=rc, timed_out=timed_out, wall_s=wall, stdout=so, stderr_tail=se[-800:])


def _why(res: dict) -> str:
    if res['timed_out']:
        return f"timed out after {res['wall_s']:.0f} s"
    return f"no output (exit {res['rc']}): {res['stderr_tail'][-300:].strip()}"


def ardour_version(ardour_lua: str) -> str:
    plist = Path(ardour_lua).resolve().parent.parent / 'Info.plist'
    try:
        with open(plist, 'rb') as f:
            d = plistlib.load(f)
        return f"Ardour {d.get('CFBundleShortVersionString', '?')} ({d.get('CFBundleVersion', '?')})"
    except (OSError, plistlib.InvalidFileException):
        return 'Ardour (Info.plist unreadable)'


def cmd_ardour(a) -> None:
    bulk = Path(a.bulk)
    version = ardour_version(a.ardour_lua)
    for job in _jobs(a):
        sess = bulk / 'sessions' / a.tag / job['id']
        if sess.exists():
            shutil.rmtree(sess)
        sess.parent.mkdir(parents=True, exist_ok=True)
        prefix = bulk / 'raw' / 'ardour' / a.tag / job['id']
        prefix.parent.mkdir(parents=True, exist_ok=True)
        for old in prefix.parent.glob(f"{job['id']}__*.raw.json"):
            old.unlink()
        bpbs = ','.join(str(k) for k in job['bpbs'])
        cmd = [a.ardour_lua, str(HERE / 'ardour_vamp.lua'), job['wav'], str(job['rate']), str(sess), str(prefix), bpbs]
        res = _run(cmd, a.timeout, prefix.with_suffix('.log'))
        outs = {}
        for tag in [f'qm-bbt-bpb{k}' for k in job['bpbs']] + ['qm-tempo']:
            raw_path = Path(f'{prefix}__{tag}.raw.json')
            arm = 'qm-tempo' if tag == 'qm-tempo' else ('qm-bbt' if tag == 'qm-bbt-bpb4' else f"qm-bbt-m{tag[10:]}")
            if raw_path.is_file():
                try:
                    outs.update(tools.vamp_outputs(_load(raw_path), job['id'], _audio(job), version))
                except (ValueError, KeyError) as exc:
                    outs[arm] = tools.failed_output(arm, 'ardour-vamp', job['id'], _audio(job),
                                                    f'unreadable raw output: {exc}', version)
            else:
                outs[arm] = tools.failed_output(arm, 'ardour-vamp', job['id'], _audio(job), _why(res), version)
        for d in outs.values():
            d.setdefault('provenance', {})['wall_s'] = res['wall_s']
        _write_outputs(a, job, outs)
        if all(d['status'] != 'failed' for d in outs.values()) and sess.exists():
            shutil.rmtree(sess)
        _log(a, f"ardour {a.tag} {job['id']}: " + ', '.join(f"{k} {v['status']}" for k, v in sorted(outs.items()))
             + f" in {res['wall_s']:.0f} s")


def cmd_minibpm(a) -> None:
    bulk = Path(a.bulk)
    version = f'Ardour 9.8 minibpm.cc sha256 {a.minibpm_src_sha256}' if a.minibpm_src_sha256 else 'Ardour 9.8 minibpm.cc'
    for job in _jobs(a):
        log = bulk / 'raw' / 'minibpm' / a.tag / f"{job['id']}.log"
        res = _run([a.minibpm_bin, job['wav']], a.timeout, log)
        try:
            raw = json.loads(res['stdout']) if res['stdout'].strip() else None
        except ValueError:
            raw = None
        if raw is None:
            outs = {arm: tools.failed_output(arm, 'ardour-minibpm', job['id'], _audio(job), _why(res), version)
                    for arm in TOOL_RUNS['minibpm']}
        else:
            _dump(log.with_suffix('.raw.json'), raw)
            outs = tools.minibpm_outputs(raw, job['id'], _audio(job), version)
        for d in outs.values():
            d.setdefault('provenance', {})['wall_s'] = res['wall_s']
        _write_outputs(a, job, outs)
        _log(a, f"minibpm {a.tag} {job['id']}: " + ', '.join(
            f"{k} {v['status']} {v.get('bpm', '')}" for k, v in sorted(outs.items())))


def opendaw_commit(checkout: str) -> str:
    head = Path(checkout) / '.git' / 'HEAD'
    try:
        ref = head.read_text().strip()
        if ref.startswith('ref: '):
            p = Path(checkout) / '.git' / ref[5:]
            if p.is_file():
                return p.read_text().strip()
            packed = (Path(checkout) / '.git' / 'packed-refs').read_text().splitlines()
            return next((ln.split()[0] for ln in packed if ln.endswith(ref[5:])), ref)
        return ref
    except OSError:
        return 'unknown'


def cmd_tempocnn(a) -> None:
    bulk = Path(a.bulk)
    h = Path(a.harness)
    version = f'openDAW {opendaw_commit(a.opendaw)[:12]} TempoDetectionTask, model {tools_sha(a.model)[:12]}'
    for job in _jobs(a):
        raw_path = bulk / 'raw' / 'tempocnn' / a.tag / f"{job['id']}.raw.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if raw_path.exists():
            raw_path.unlink()
        cmd = [a.node, '--import', str(h / 'node_resolve.mjs'), str(h / 'tempocnn_one.mjs'), a.opendaw, a.model,
               job['wav'], str(raw_path)]
        res = _run(cmd, a.timeout, raw_path.with_suffix('.log'))
        if raw_path.is_file():
            outs = tools.tempocnn_outputs(_load(raw_path), job['id'], _audio(job), version)
        else:
            outs = {'tcnn': tools.failed_output('tcnn', 'opendaw-tempocnn', job['id'], _audio(job), _why(res), version)}
        for d in outs.values():
            d.setdefault('provenance', {})['wall_s'] = res['wall_s']
        _write_outputs(a, job, outs)
        _log(a, f"tempocnn {a.tag} {job['id']}: {outs['tcnn']['status']} {outs['tcnn'].get('bpm', '')}")


def tools_sha(path: str) -> str:
    return lb.sha256_file(path) if path and Path(path).is_file() else 'unknown'


# ---------------------------------------------------------------- status

def arm_files(bulk: Path, tag: str) -> dict:
    """{arm: {id: path}} of the normalized outputs written under ``maps/<tag>``."""
    base = bulk / 'maps' / tag
    out = {}
    if base.is_dir():
        for d in sorted(p for p in base.iterdir() if p.is_dir()):
            out[d.name] = {p.stem: p for p in sorted(d.glob('*.json'))}
    return out


def same_output(x: dict, y: dict) -> bool:
    keys = ('status', 'kind', 'beats_s', 'downbeats', 'bpm', 'grid')
    return all(x.get(k) == y.get(k) for k in keys)


def cmd_status(a) -> None:
    bulk, out = Path(a.bulk), Path(a.out)
    jobs = _load(bulk / 'jobs.json')
    songs = [j['id'] for j in jobs if j['kind'] == 'song']
    main = arm_files(bulk, 'main')
    rows, failures = {}, []
    for arm, files in main.items():
        rows[arm] = {}
        for sid in songs + list(PROBES):
            p = files.get(sid)
            if p is None:
                rows[arm][sid] = 'not run'
                continue
            d = _load(p)
            rows[arm][sid] = d['status']
            if d['status'] == 'failed':
                failures.append(dict(arm=arm, id=sid, error=d.get('error')))
    nulls = {}
    for arm, files in arm_files(bulk, 'null').items():
        for sid, p in files.items():
            m = main.get(arm, {}).get(sid)
            nulls.setdefault(arm, {})[sid] = None if m is None else same_output(_load(p), _load(m))
    probes = {}
    for pid in PROBES:
        if not (bulk / 'probes' / f'{pid}.truth.json').is_file():
            continue
        truth = _load(bulk / 'probes' / f'{pid}.truth.json')
        for arm, files in main.items():
            if pid not in files:
                continue
            d = _load(files[pid])
            if d['status'] != 'ok':
                probes.setdefault(pid, {})[arm] = dict(status=d['status'])
            elif d['kind'] == 'beats':
                probes.setdefault(pid, {})[arm] = tools.probe_offsets(d['beats_s'], truth['beats_s'])
            else:
                probes.setdefault(pid, {})[arm] = dict(bpm=d['bpm'], true_bpm=truth['bpm'])
    counts = {arm: {k: sum(1 for sid in songs if v.get(sid) == k) for k in ('ok', 'empty', 'failed', 'not run')}
              for arm, v in rows.items()}
    _dump(out / 'run-status.json', dict(songs=len(songs), counts=counts, per_song=rows, failures=failures,
                                        nulls=nulls, probes=probes, max_failed=MAX_FAILED))
    _log(a, 'status: ' + '; '.join(f'{arm} {c}' for arm, c in counts.items()))


# ---------------------------------------------------------------- scoring

def song_spans(s: dict, x: dict) -> dict:
    """The spans ``logic_baseline.py score`` scores (whole song, hard span, the stratum's census spans)."""
    hard = s['hard_span_ms']
    spans = {'whole': None, 'hard': [tuple(hard)] if hard else None}
    if s['group'] in x['strata']:
        spans['stratum'] = x['strata'][s['group']]
    return spans


def tempo_terms(ref, bpm: float, spans) -> dict:
    """One stated tempo against the chart's notated tempo on the scored region, duration-weighted over the chart's
    segments: Acc1, the share within 4 %; Acc2, within 4 % of 1, 2, 1/2, 3 or 1/3 times it (the tempo-estimation
    literature's accuracies); and the weighted median ratio. A tool that stated no tempo (``bpm`` 0) scores 0."""
    from ensomi_n.eval.measures import _Region, _ref_segments, weighted_median
    lo, hi = ref.lo_ms, ref.hi_ms
    region = _Region(lo, hi, spans)
    dur, ratios = [], []
    for a, b, step in _ref_segments(ref):
        a_c, b_c = max(a, lo), min(b, hi)
        if b_c <= a_c:
            continue
        d = float(region.overlap(a_c, b_c))
        if d > 0:
            dur.append(d)
            ratios.append(bpm * step / 60000.0)
    if not dur:
        return dict(acc1=math.nan, acc2=math.nan, ratio=math.nan)
    w, r = np.asarray(dur), np.asarray(ratios)
    if not bpm > 0:
        return dict(acc1=0.0, acc2=0.0, ratio=math.nan)
    acc1 = np.abs(r - 1.0) <= ACC_TOL
    acc2 = np.any([np.abs(r / f - 1.0) <= ACC_TOL for f in ACC2_FACTORS], axis=0)
    return dict(acc1=float((w * acc1).sum() / w.sum()), acc2=float((w * acc2).sum() / w.sum()),
                ratio=weighted_median(r, w))


def load_open_arms(a, songs: list[dict]) -> tuple[dict, dict, dict]:
    """``({arm@c: {sha: MusicMap on osu!'s clock}}, {arm: {id: status or error}}, {arm: {id: stated bpm}})``."""
    from ensomi_n.mapformat.open_tools import load_open, read_open_map, stated_bpm
    files = arm_files(Path(a.bulk), 'main')
    maps, status, bpm = {}, {}, {}
    for arm in sorted(files):
        status[arm], bpm[arm] = {}, {}
        for cname in CLOCK_CONSTANTS:
            maps[f'{arm}@{cname}'] = {}
        for s in songs:
            p = files[arm].get(s['id'])
            if p is None:
                status[arm][s['id']] = 'not run'
                continue
            d = load_open(p)
            status[arm][s['id']] = d['status'] if d['status'] != 'failed' else f"failed: {d.get('error')}"
            if d['status'] == 'failed':
                continue
            bpm[arm][s['id']] = 0.0 if d['status'] == 'empty' else stated_bpm(d)
            m = read_open_map(p, lead_ms=s['lead_ms'])
            for cname, c in CLOCK_CONSTANTS.items():
                maps[f'{arm}@{cname}'][s['audio_sha256']] = m.on_osu_clock(c)
    return maps, status, bpm


def failed_tools(status: dict) -> dict:
    """{arm: failed count} for arms past the stopping rule (`MAX_FAILED`). A song an `ARMS` arm has no output for
    counts as failed; the meter oracle's runs (``qm-bbt-m<k>``) exist only for songs whose chart states k."""
    out = {}
    for arm, v in status.items():
        n = sum(1 for x in v.values() if x.startswith('failed') or (x == 'not run' and arm in ARMS))
        if n > MAX_FAILED:
            out[arm] = n
    return out


def systems(ag: dict, groups: dict) -> dict:
    """`logic_baseline.system_agreement` over any named groups of arms."""
    agree = ag.get('agree') or {}

    def pair(x, y):
        return agree.get(f'{x}|{y}', agree.get(f'{y}|{x}', False))
    names = [n for n in groups if any(m in ag.get('n_lines', {}) for m in groups[n])]
    pairs = {}
    for i, u in enumerate(names):
        for v in names[i + 1:]:
            pairs[f'{u}|{v}'] = any(pair(x, y) for x in groups[u] for y in groups[v])
    return dict(pairs=pairs)


def cmd_score(a) -> None:
    from ensomi_n.v0.evaluate import eval_items
    out = Path(a.out)
    pre = out / 'preflight'
    compare = _load(pre / 'compare.json')
    if not all(r['equal'] for r in compare.values()):
        raise SystemExit('score: the preflight did not reproduce the Smart Tempo run')
    songs = _load(pre / 'songs.json')['songs']
    by_sha = {s['audio_sha256']: s for s in songs}
    old_rows = {r['id']: r for r in _load(pre / 'items.json')}
    ns = lb_namespace(a, pre)
    arms_old, _ = lb.load_arms(ns, songs)
    last = lb.last_checkpoints(arms_old)
    maps, status, bpm = load_open_arms(a, songs)
    stopped = failed_tools(status)
    for arm in stopped:
        for cname in CLOCK_CONSTANTS:
            maps.pop(f'{arm}@{cname}', None)
    jobs = {j['id']: j for j in _load(Path(a.bulk) / 'jobs.json')}
    split = {'logic': [dict(audio_sha256=s['audio_sha256'], charts=s['charts'], rec=s['rec']) for s in songs]}
    items = eval_items(split, ['logic'], Path(a.root), corpus=a.corpus, tags=a.tags)
    import reader_census as rc
    rows, checks = [], []
    for x in items:
        s = by_sha[x['item']]
        row = copy.deepcopy(old_rows[s['id']])
        if 'error' in x:
            rows.append(row)
            continue
        ref = x['ref']
        lo, hi = ref.lo_ms, ref.hi_ms
        spans = song_spans(s, x)
        # The new arms use these spans; an old arm rescored on them must give the preflight's value.
        for arm in ('r2', 'st-constant@c20'):
            if x['item'] in arms_old.get(arm, {}):
                for name, sp in spans.items():
                    v = lb._clean(lb.score_span(ref, arms_old[arm][x['item']], sp)['bl20'])
                    w = ((row['scores'].get(arm) or {}).get(name) or {}).get('bl20')
                    if not ((v is None and w is None) or (v is not None and w is not None and abs(v - w) <= 1e-9)):
                        checks.append(dict(id=s['id'], arm=arm, span=name, rescored=v, preflight=w))
        for arm, mm in maps.items():
            m = mm.get(x['item'])
            if m is None:
                continue
            row['scores'][arm] = {}
            for name, sp in spans.items():
                try:
                    row['scores'][arm][name] = lb.score_span(ref, m, sp)
                except Exception as exc:  # noqa: BLE001
                    row['scores'][arm][name] = dict(error=f'{type(exc).__name__}: {exc}')
        opened = dict(status={arm: v.get(s['id'], 'not run') for arm, v in status.items()},
                      oracle_bpbs=jobs.get(s['id'], {}).get('oracle_bpbs'), stated_bpm={}, tempo={})
        st_tempo = row.get('logic', {}).get('constant', {})
        st_bpm = st_tempo.get('tempo_min') if st_tempo.get('tempo_min') == st_tempo.get('tempo_max') else None
        for arm in TEMPO_ARMS + ('st-constant',):
            b = st_bpm if arm == 'st-constant' else bpm.get(arm, {}).get(s['id'])
            if b is None or arm in stopped:
                continue
            opened['stated_bpm'][arm] = b
            opened['tempo'][arm] = {name: tempo_terms(ref, float(b), sp) for name, sp in spans.items()}
        row['open'] = opened
        hard = s['hard_span_ms']
        a0, b0 = hard if hard else (lo, hi)
        t_ref = ref.lines().t
        for cname in CLOCK_CONSTANTS:
            lines = {f'st-{mode}': lb.canonical_in(arms_old[f'st-{mode}@{cname}'][x['item']], a0, b0, lo, hi)
                     for mode in lb.MODES}
            for seed, arm in last.items():
                lines[f'v0{seed}'] = lb.canonical_in(arms_old[arm][x['item']], a0, b0, lo, hi)
            if x['item'] in arms_old['r2']:
                lines['r2'] = lb.canonical_in(arms_old['r2'][x['item']], a0, b0, lo, hi)
            for arm in BEAT_ARMS:
                m = maps.get(f'{arm}@{cname}', {}).get(x['item'])
                if m is not None:
                    lines[arm] = lb.canonical_in(m, a0, b0, lo, hi)
            lines['mapper'] = t_ref[(t_ref >= a0) & (t_ref <= b0)]
            ag = lb.agreement(lines)
            row.setdefault('agreement_open', {})[cname] = dict(ag, systems=systems(ag, SYSTEMS_OPEN))
        if s['tier'] == 'B':
            try:
                r_t = rc.r_lines(ref.meta['timing_chart'], lo, hi)[0]
                for cname in CLOCK_CONSTANTS:
                    pm = {arm.split('@')[0]: mm[x['item']] for arm, mm in maps.items()
                          if arm.endswith(f'@{cname}') and arm.split('@')[0].startswith('qm-') and x['item'] in mm}
                    if pm:
                        row.setdefault('pulse_open', {})[cname] = lb.pulse(ref, r_t, pm)
            except Exception as exc:  # noqa: BLE001
                row['pulse_open_error'] = f'{type(exc).__name__}: {exc}'
        rows.append(lb._clean(row))
        _log(a, f"score {s['id']} {s['list_id']}: {sum(1 for k in row['scores'] if k.split('@')[0] in ARMS or k.startswith('qm-bbt-m'))} open arm-clocks")
    _dump(out / 'items.json', rows)
    _dump(out / 'arms.json', dict(arms=ARMS, oracle={ORACLE: ORACLE_WHAT}, last=last, status=status,
                                  stopped=stopped, clock_constants=CLOCK_CONSTANTS))
    _dump(out / 'score-checks.json', dict(rescored_old_arms_differ=checks))
    _log(a, f'score: {len(rows)} songs; {len(checks)} old-arm rescoring differences; stopped arms {stopped or "none"}')
    if checks:
        raise SystemExit('score: an old arm rescored on the new spans differs from the preflight; see score-checks.json')


# ---------------------------------------------------------------- report

def oracle_arms(row: dict) -> list[str]:
    """The qm-barbeattracker runs the meter oracle chooses among: one per beats-per-bar value the chart states."""
    return ['qm-bbt' if k == 4 else f'qm-bbt-m{k}' for k in (row.get('open') or {}).get('oracle_bpbs') or []]


def open_value(row: dict, arm: str, span: str, measure: str, cname: str) -> float:
    if arm == ORACLE:
        vals = [lb._get(row, f'{x}@{cname}', span, measure) for x in oracle_arms(row)]
        vals = [v for v in vals if math.isfinite(v)]
        return max(vals) if vals else math.nan
    return lb._get(row, f'{arm}@{cname}', span, measure)


def old_values(rows: list[dict], last: dict, arm: str, span: str, measure: str, cname: str) -> np.ndarray:
    """An old arm's per-song values, as ``logic_baseline.table_block`` reads them."""
    if arm.startswith('v0') and arm != 'v0mean':
        return np.array([lb._get(r, last[arm[2:]], span, measure) for r in rows])
    if arm == 'v0mean':
        return lb.v0_mean_values(rows, last, span, measure)
    return lb.arm_values(rows, arm, span, measure, cname)


def open_block(rows: list[dict], last: dict, span: str, measure: str, cname: str, arms=OPEN_ARMS) -> dict:
    """Per arm: song mean with a song bootstrap; per open arm, paired differences against Smart Tempo's two modes,
    R2 and v0 (held-out songs only; trained-on songs apart). Old arms' values are read as Smart Tempo's report
    reads them."""
    out = {'n_songs': len(rows), 'arms': {}, 'paired': {}}
    vals = {}
    for arm in OLD_ARMS:
        vals[arm] = old_values(rows, last, arm, span, measure, cname)
        out['arms'][arm] = lb.boot_mean(vals[arm])
    held = np.array([r['v0_exposure'] == 'held-out' for r in rows], dtype=bool)
    for arm in arms:
        v = np.array([open_value(r, arm, span, measure, cname) for r in rows])
        out['arms'][arm] = lb.boot_mean(v)
        for other in ('st-constant', 'st-variable', 'r2'):
            out['paired'][f'{arm} - {other}'] = lb.boot_mean(v - vals[other])
        out['paired'][f'{arm} - v0mean (held-out)'] = lb.boot_mean(np.where(held, v - vals['v0mean'], np.nan))
        out['paired'][f'{arm} - v0mean (trained-on, apart)'] = lb.boot_mean(
            np.where(~held, v - vals['v0mean'], np.nan))
    return out


def _tempo_value(row: dict, arm: str, span: str, k: str) -> float:
    v = ((((row.get('open') or {}).get('tempo') or {}).get(arm) or {}).get(span) or {}).get(k)
    return math.nan if v is None else float(v)


def _level_class(row: dict, key: str, span: str, cls: str) -> float:
    lc = (((row.get('scores') or {}).get(key) or {}).get(span) or {}).get('level_classes') or {}
    v = lc.get(cls)
    return math.nan if v is None else float(v)


def tempo_block(rows: list[dict], span: str) -> dict:
    """Per one-tempo arm (and Smart Tempo's Constant mode): song means of Acc1, Acc2 and the level term, and of
    the level-class shares (octave, family ratio, another tempo, none stated). The level term does not depend on
    the clock constant, so it is read at ``c20``."""
    out = {}
    for arm in TEMPO_ARMS + ('st-constant',):
        key = f'{arm}@c20'
        res = {k: lb.boot_mean(np.array([_tempo_value(r, arm, span, k) for r in rows])) for k in ('acc1', 'acc2')}
        res['level'] = lb.boot_mean(np.array([lb._get(r, key, span, 'level') for r in rows]))
        for cls in ('octave', 'family', 'other', 'missing'):
            res[f'level_{cls}'] = lb.boot_mean(np.array([_level_class(r, key, span, cls) for r in rows]))
        out[arm] = res
    return out


def pulse_summary(rows: list[dict]) -> dict:
    """Tier A: on how many hard spans each open beat tracker agrees with the mapper, v0, R2 and Smart Tempo."""
    out = {}
    for cname in CLOCK_CONSTANTS:
        res = {}
        for arm in BEAT_ARMS:
            counts = {other: 0 for other in ('mapper', 'v0', 'r2', 'st')}
            n = 0
            for r in rows:
                if r['tier'] != 'A':
                    continue
                pairs = ((r.get('agreement_open') or {}).get(cname) or {}).get('systems', {}).get('pairs', {})
                if not any(arm in k for k in pairs):
                    continue
                n += 1
                for other in counts:
                    counts[other] += bool(pairs.get(f'{other}|{arm}', pairs.get(f'{arm}|{other}', False)))
            res[arm] = dict(songs=n, agrees_with=counts)
        out[cname] = res
    return out


def cmd_report(a) -> None:
    out = Path(a.out)
    rows = _load(out / 'items.json')
    errors = [dict(id=r['id'], error=r['error']) for r in rows if 'error' in r]
    rows = [r for r in rows if 'scores' in r]
    meta = _load(out / 'arms.json')
    last = meta['last']
    status = _load(out / 'run-status.json') if (out / 'run-status.json').is_file() else {}
    pre = _load(out / 'preflight' / 'compare.json')
    tiers = {t: [r for r in rows if r['tier'] == t] for t in 'ABC'}
    rep = dict(question='d-reference-baselines: the open tools on the Smart Tempo baseline 48 songs, scored as '
                        'Smart Tempo was', arms=ARMS, oracle={ORACLE: ORACLE_WHAT}, stopped=meta['stopped'],
               clock_constants=CLOCK_CONSTANTS, preflight=pre, last_checkpoints=last,
               rules=dict(acc_tol=ACC_TOL, acc2_factors=ACC2_FACTORS, max_failed=MAX_FAILED, boot=lb.BOOT,
                          agree=lb.AGREE),
               counts=dict(songs=len(rows) + len(errors), scored=len(rows), reference_errors=errors,
                           per_tier={t: len(v) for t, v in tiers.items()},
                           held_out={t: sum(r['v0_exposure'] == 'held-out' for r in v) for t, v in tiers.items()}),
               run_status=status, tiers={}, strata_c={}, tempo={}, pulse_tier_a=pulse_summary(rows), songs=[])
    for tier, rs in tiers.items():
        rep['tiers'][tier] = {span: {cname: {m: open_block(rs, last, span, m, cname) for m in lb.MEASURES}
                                     for cname in CLOCK_CONSTANTS} for span in ('whole', 'hard')}
        rep['tempo'][tier] = {span: tempo_block(rs, span) for span in ('whole', 'hard')}
    for st in ('S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9'):
        rs = [r for r in tiers['C'] if r['group'] == st]
        rep['strata_c'][st] = {span: {cname: {m: open_block(rs, last, span, m, cname)
                                              for m in ('bl20', 'bl20_nonfree', 'offset_ms', 'level')}
                                      for cname in CLOCK_CONSTANTS} for span in ('hard', 'stratum', 'whole')}
    keep = lb.MEASURES + ('attribution', 'chance20', 'level_classes', 'bl5', 'bars_precision')
    for r in rows:
        song = dict(id=r['id'], list_id=r['list_id'], tier=r['tier'], group=r['group'], v0_exposure=r['v0_exposure'],
                    ref_notated_bpm_median=r.get('ref_notated_bpm_median'), ref_meters=r.get('ref_meters'),
                    open=r.get('open'), values={})
        for span in ('whole', 'hard', 'stratum'):
            for arm in sorted(r['scores']):
                if arm.split('@')[0] not in ARMS and not arm.startswith('qm-bbt-m'):
                    continue
                m = r['scores'][arm].get(span)
                if m and 'error' not in m:
                    song['values'].setdefault(span, {})[arm] = {k: m.get(k) for k in keep}
                elif m:
                    song.setdefault('score_errors', {})[f'{arm}/{span}'] = m['error']
        for k in ('agreement_open', 'pulse_open', 'pulse_open_error'):
            if k in r:
                song[k] = r[k]
        rep['songs'].append(song)
    _dump(out / 'results.json', lb._clean(rep))
    (out / 'RESULTS.md').write_text(markdown(rep))
    _log(a, 'report: results.json and RESULTS.md written')


def _f(v, nd=3) -> str:
    return lb._f(v, nd)


def markdown(rep: dict) -> str:
    C = rep['clock_constants']
    L = ['# Ardour and openDAW detectors beside Smart Tempo, v0 and R2', '',
         'Generated by `scripts/n1/open_baselines/open_baselines.py report`. Every number characterises: no measure '
         "here has been shown to agree with the human's ear. `qm-bbt-oracle` reads the chart (its beats-per-bar "
         'values and the choice among them) and is never a baseline. Comparisons with v0 use only songs v0 never '
         'trained on. Smart Tempo, v0 and R2 values are the Smart Tempo run\'s, reproduced by the preflight.', '',
         '## Arms', '', '| Arm | What | Grid |', '| --- | --- | --- |']
    for k, v in rep['arms'].items():
        L.append(f"| `{k}` | {v['what']} | {v['grid']} |")
    L.append(f"| `{ORACLE}` | {ORACLE_WHAT} (oracle) | stated by the tool |")
    pf = rep['preflight']
    L += ['', '## Checks', '',
          f"- Preflight: Smart Tempo `results.json` {'reproduced' if pf['results.json']['equal'] else 'NOT reproduced'}"
          f" ({pf['results.json']['n_differences']} differences); `items.json` "
          f"{'reproduced' if pf['items.json']['equal'] else 'NOT reproduced'} ({pf['items.json']['n_differences']}).",
          f"- Songs scored: {rep['counts']['scored']} of {rep['counts']['songs']}; per tier {rep['counts']['per_tier']}; "
          f"held out from v0 {rep['counts']['held_out']}.",
          f"- Arms stopped by the failure rule (more than {rep['rules']['max_failed']} failed songs): "
          f"{rep['stopped'] or 'none'}."]
    rs = rep.get('run_status') or {}
    never = [arm for arm in rep['arms'] if arm not in rs.get('counts', {})]
    L.append(f"- Arms with no output at all (tool not run): {', '.join(f'`{x}`' for x in never) or 'none'}.")
    if rs:
        L += ['', f"### Runs per arm ({rs.get('songs')} songs)", '', '| Arm | ok | empty | failed | not run |',
              '| --- | --- | --- | --- | --- |']
        for arm, c in sorted(rs.get('counts', {}).items()):
            L.append(f"| `{arm}` | {c['ok']} | {c['empty']} | {c['failed']} | {c['not run']} |")
        if rs.get('failures'):
            L += ['', 'Failed runs:', '']
            L += [f"- `{f['arm']}` {f['id']}: {f['error']}" for f in rs['failures']]
        if rs.get('nulls'):
            L += ['', 'Determinism null (a second run on the same file): ' + '; '.join(
                f"`{arm}` {sum(1 for v in d.values() if v)} of {len(d)} identical" for arm, d in rs['nulls'].items())]
        if rs.get('probes'):
            L += ['', '### Synthetic click probes (120 BPM; a fact about each tool, never applied)', '',
                  '| Probe | Arm | median offset ms | MAD ms | within 20 ms | BPM |', '| --- | --- | --- | --- | --- | --- |']
            for pid, arms in rs['probes'].items():
                for arm, d in sorted(arms.items()):
                    L.append(f"| {pid} | `{arm}` | {_f(d.get('median_ms'), 1)} | {_f(d.get('mad_ms'), 1)} | "
                             f"{_f(d.get('within20'), 2)} | {_f(d.get('bpm'), 2)} |")
    names = ('st-constant', 'st-variable', 'v0s0', 'v0s1', 'r2') + OPEN_ARMS
    for tier, spans in rep['tiers'].items():
        L += ['', f'## Tier {tier}', '']
        for span in ('whole', 'hard'):
            if tier == 'B' and span == 'hard':
                continue
            for cname in C:
                blocks = spans[span][cname]
                L += [f'### Tier {tier}, {"whole song" if span == "whole" else "hard span"}, clock `{cname}`', '',
                      'Song means with 95 % song-bootstrap intervals. BPM-only arms (`mbpm-*`, `tcnn`) state no '
                      'phase: their bl* terms score the grid named under Arms.', '',
                      '| Measure | ' + ' | '.join(names) + ' |', '| --- |' + ' --- |' * len(names)]
                for m, blk in blocks.items():
                    A = blk['arms']
                    L.append(f'| {m} | ' + ' | '.join(lb._ci(A.get(n)) for n in names) + ' |')
                L += ['', '| Paired difference, BL-20 | mean [95 %] (n songs) |', '| --- | --- |']
                for k, d in blocks['bl20']['paired'].items():
                    if 'trained-on' not in k:
                        L.append(f'| {k} | {lb._dci(d)} |')
                L.append('')
    L += ['## One-tempo arms: tempo terms', '',
          "Acc1: share of the chart's duration whose notated tempo is within 4 % of the stated tempo; Acc2: also "
          'allowing 2, 1/2, 3 and 1/3 times it. `level` is the harness term (0.5 %); the level-class shares say '
          'where the rest went. Smart Tempo Constant states one tempo too and is listed for comparison.', '']
    for tier, spans in rep['tempo'].items():
        for span, blk in spans.items():
            if tier == 'B' and span == 'hard':
                continue
            L += [f'### Tier {tier}, {"whole song" if span == "whole" else "hard span"}', '',
                  '| Arm | Acc1 | Acc2 | level | octave | family | other | missing |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- |']
            for arm, d in blk.items():
                L.append(f"| `{arm}` | {lb._ci(d['acc1'])} | {lb._ci(d['acc2'])} | {lb._ci(d['level'])} | "
                         f"{_f(d['level_octave'].get('value'))} | {_f(d['level_family'].get('value'))} | "
                         f"{_f(d['level_other'].get('value'))} | {_f(d['level_missing'].get('value'))} |")
            L.append('')
    L += ['## Tier C per stratum, BL-20, hard span', '']
    for cname in C:
        L += [f'### clock `{cname}`', '', '| Stratum | n | ST const | ST var | v0 mean | R2 | qm-bbt | qm-tempo | '
              'mbpm-clip | tcnn |', '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for st, d in rep['strata_c'].items():
            A = d['hard'][cname]['bl20']['arms']
            L.append(f"| {st} | {d['hard'][cname]['bl20']['n_songs']} | " + ' | '.join(
                lb._ci(A.get(n)) for n in ('st-constant', 'st-variable', 'v0mean', 'r2', 'qm-bbt', 'qm-tempo',
                                           'mbpm-clip', 'tcnn')) + ' |')
        L.append('')
    L += ['## Tier A: whose pulse the open trackers pick', '',
          f"Two readings agree on a hard span when each has at least {rep['rules']['agree']} of its canonical lines "
          'matched by the other within 20 ms (as for Smart Tempo, `o-st-pulse`).', '']
    for cname, d in rep['pulse_tier_a'].items():
        for arm, v in d.items():
            L.append(f"- `{cname}` `{arm}`: of {v['songs']} spans, agrees with the mapper on "
                     f"{v['agrees_with']['mapper']}, v0 {v['agrees_with']['v0']}, R2 {v['agrees_with']['r2']}, "
                     f"Smart Tempo {v['agrees_with']['st']}.")
    L += ['', '## Per song', '',
          'BL-20 on the hard span (tier B: the whole song) at `c20`, and the stated tempo of the one-tempo arms.', '',
          '| Song | Stratum | Chart BPM | qm-bbt | oracle | qm-tempo | mbpm-clip | tcnn | MiniBPM raw / clip / source | '
          'TempoCNN | ST const |', '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for s in rep['songs']:
        span = 'whole' if s['tier'] == 'B' else 'hard'
        V = s['values'].get(span, {})
        sb = (s.get('open') or {}).get('stated_bpm', {})

        def g(arm):
            return _f((V.get(f'{arm}@c20') or {}).get('bl20'))
        ora = [(V.get(f'{x}@c20') or {}).get('bl20') for x in
               ['qm-bbt' if k == 4 else f'qm-bbt-m{k}' for k in (s.get('open') or {}).get('oracle_bpbs') or []]]
        ora = [v for v in ora if v is not None]
        L.append(f"| {s['list_id']} | {s['group']} | {_f(s.get('ref_notated_bpm_median'), 1)} {s.get('ref_meters')} | "
                 f"{g('qm-bbt')} | {_f(max(ora) if ora else None)} | {g('qm-tempo')} | {g('mbpm-clip')} | {g('tcnn')} | "
                 f"{_f(sb.get('mbpm-raw'), 2)} / {_f(sb.get('mbpm-clip'), 2)} / {_f(sb.get('mbpm-source'), 2)} | "
                 f"{_f(sb.get('tcnn'), 0)} | {_f(sb.get('st-constant'), 2)} |")
    return '\n'.join(L) + '\n'


# ---------------------------------------------------------------- main

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('cmd', choices=('prep', 'preflight', 'ardour', 'minibpm', 'tempocnn', 'status', 'score', 'report'))
    p.add_argument('--root', default='.')
    p.add_argument('--logic', default='artifacts/reports/logic-tempo-baseline-20261020')
    p.add_argument('--logic-out', default='artifacts/reports/logic-tempo-baseline-20261020/analysis')
    p.add_argument('--logic-bulk', default='artifacts/bulk/logic-tempo-baseline-20261020')
    p.add_argument('--out', default='artifacts/reports/open-baselines-20261021')
    p.add_argument('--bulk', default='artifacts/bulk/open-baselines-20261021')
    p.add_argument('--wav-dir', default='', help="directory of the canonical NNN.wav files; default: the Smart "
                                                 "Tempo manifest's prepared_input_path")
    p.add_argument('--r0', default='artifacts/n1-night-20261018/runs/r0-20261019a')
    p.add_argument('--curated', default='artifacts/n1-night-20261018/runs/v0data2-20261019/curated.json')
    p.add_argument('--tune-dir', default='artifacts/n1-night-20261018/runs/v0-s0-20261019/eval')
    p.add_argument('--bulk-runs', default='artifacts/bulk/n1-runs')
    p.add_argument('--stored-maps', default='artifacts/bulk/n1-v0-analysis')
    p.add_argument('--r2-stored', default='artifacts/bulk/n1-cache/r2-export')
    p.add_argument('--corpus', default='data/r2-corpus.parquet')
    p.add_argument('--tags', default='artifacts/n1-inventory-20261014/tags.json')
    p.add_argument('--ids', default='', help='comma-separated job ids (songs NNN, probes P44 and P48); default all')
    p.add_argument('--tag', default='main', help='output set: main, or null for the determinism re-run')
    p.add_argument('--timeout', type=float, default=900.0, help='seconds per song and tool run')
    p.add_argument('--ardour-lua', default='ref-proj/ardour-demo/Ardour9.app/Contents/MacOS/ardour9-lua')
    p.add_argument('--minibpm-bin', default='')
    p.add_argument('--minibpm-src-sha256', default='')
    p.add_argument('--node', default='node')
    p.add_argument('--opendaw', default='ref-proj/opendaw')
    p.add_argument('--model', default='artifacts/bulk/open-daws-20261021/opendaw/tempo-cnn-v0.onnx')
    p.add_argument('--harness', default='artifacts/bulk/open-baselines-20261021/tempocnn-harness')
    a = p.parse_args(argv)
    lb.LOG_DIR = Path(a.bulk)
    dict(prep=cmd_prep, preflight=cmd_preflight, ardour=cmd_ardour, minibpm=cmd_minibpm, tempocnn=cmd_tempocnn,
         status=cmd_status, score=cmd_score, report=cmd_report)[a.cmd](a)


if __name__ == '__main__':
    main()
