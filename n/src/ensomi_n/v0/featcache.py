"""Cache the input and raw output of v0.1's linear heads on the evaluation clock.

Run from the repository with ``python -m ensomi_n.v0.featcache --root DATA_ROOT
--out OUTPUT``. Existing Mel/memory arrays and saved curation/split records are
required; this command never decodes audio or rebuilds labels. Only kept training
inputs, the fit_dev panel and val-n1 hard are allowed. Exclusion metadata is read
before any song features or chart headers; sealed audio and labels are not read.

Each SHA-keyed NPZ stores float16 ``feat[T,160]`` (after encoder norm/GELU),
``logits[T,23]`` (before head nonlinearities), and scalar ``t0_ms`` and ``fps``.
The manifest retains all chart/recording memberships when inputs share audio.
Five panel inputs must reproduce E2 in float32 before the full cache proceeds;
the report separately measures the error from storing logits as float16.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from ..features.cache import sha256_file
from ..features.mel import DEFAULT

RUNS = Path('artifacts/n1-night-20261018/runs')
HEAD_LAYOUT = ['beat', 'half', 'down', 'beat_offset_raw', 'half_offset_raw', 'down_offset_raw',
               'boundary', 'level_half', 'level_one', 'level_two', 'level_four', 'phase_weight',
               'phase_c1', 'phase_s1', 'phase_c2', 'phase_s2', 'kappa1_raw', 'kappa2_raw',
               'log_period', 'log_period_sigma_raw', 'bar_cos', 'bar_sin', 'bar_kappa_raw']


def read_json(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def check_membership(rows, exclusion, excluded_recordings):
    """Reject an entire input list on a sealed identity or a foreign source path."""
    blocked_recs = {r['rec'] for r in excluded_recordings}
    blocked_audio = {r.get('audio_sha256') for r in excluded_recordings} | exclusion['audio_sha256']
    blocked_paths = {p for r in excluded_recordings for p in r.get('audio_rels', [])}
    blocked_sets = {str(s) for s in exclusion['set_ids']}
    for r in rows:
        p = Path(r['audio_rel'])
        if (p.is_absolute() or '..' in p.parts or len(p.parts) < 4
                or p.parts[:2] not in (('dataset', '0'), ('dataset', 'std-taiko'))
                or r.get('source') not in ('4k', 'std', 'taiko')):
            raise ValueError(f"Input outside the curated osu! sources: {r['audio_rel']}")
        if (r['audio_sha256'] in blocked_audio or r['rec'] in blocked_recs
                or r['audio_rel'] in blocked_paths or str(r.get('set_id')) in blocked_sets
                or p.parent.name in blocked_sets
                or r['rec'] in exclusion['recordings'] or r.get('census_key') in exclusion['keys']):
            raise ValueError(f"Sealed input in requested list: {r['audio_sha256']}")


def prepare(root):
    from ..data.manifest import read_csv_shas
    from ..eval.census import load_exclusion, read_header
    from .curate import SEAL
    from .data import load_songs
    from .evaluate import inference_song

    data = root / RUNS / 'v0data2-20261019'
    r0 = root / RUNS / 'r0-20261019a'
    bulk = root / 'artifacts/bulk/n1-cache'
    paths = [data / 'curated.json', r0 / 'split.json', r0 / 'recordings.json', bulk / 'song-index-v0.json']
    curated, split, recs, index = map(read_json, paths)
    if (len(split['panel']) != 322 or len({r['rec'] for r in split['panel']}) != 262
            or len(split['val_hard']) != 150):
        raise ValueError('Saved panels differ from the v0.1 cohort')
    training = [dict(r, split='fit_train') for r in curated if r['role'] == 'train' and r.get('keep')]
    rows = list(training)
    for panel, name in (('panel', 'fit_dev'), ('val_hard', 'val-n1')):
        rows.extend(dict(r, split=name, source='4k') for r in split[panel])
    exclusion = load_exclusion(root / SEAL[0], SEAL[1])  # identifiers only, the curation seal
    for relative in ('artifacts/timing-20261009/opt/holdout/holdout.csv',
                     'artifacts/timing-20261009/opt/ft/pop/holdout2.csv'):
        exclusion['audio_sha256'] |= read_csv_shas(root / relative)
    check_membership(rows, exclusion, [r for r in recs if r['role'].startswith('excluded:')])

    # Header reads are confined to selected sets after the identity check above.
    refs = {r['rec']: r for r in recs}
    for r in rows:
        metadata = (root / r['audio_rel']).parent / 'metadata.json'
        beatmaps = read_json(metadata).get('beatmaps', []) if metadata.exists() else []
        by_md5 = {b['checksum']: b for b in beatmaps if b.get('checksum')}
        if r['source'] == '4k':
            charts = r.get('charts', refs[r['rec']].get('charts', []))
        else:
            charts = [str(p.relative_to(root)) for p in sorted((root / r['audio_rel']).parent.glob('*.osu'))]
        ids, sets = set(), set()
        kept_charts = []
        for chart in charts:
            p = Path(chart)
            if p.is_absolute() or '..' in p.parts or p.parent != Path(r['audio_rel']).parent:
                raise ValueError(f'Chart outside selected audio directory: {chart}')
            h = read_header(root / chart)
            if h.get('AudioFilename', '').casefold() != Path(r['audio_rel']).name.casefold():
                continue
            if r['source'] != '4k' and h.get('Mode', '0') != ('0' if r['source'] == 'std' else '1'):
                continue
            kept_charts.append(chart)
            beatmap_id = int(h.get('BeatmapID', '-1'))
            if beatmap_id <= 0 and by_md5:
                entry = by_md5.get(hashlib.md5((root / chart).read_bytes()).hexdigest(), {})
                beatmap_id = int(entry.get('id', -1))
            if beatmap_id <= 0:
                # Older charts can omit IDs and predate the API's checksum; require an unambiguous difficulty.
                matches = [b for b in beatmaps if b['version'] == h.get('Version')
                           and b['mode_int'] == int(h.get('Mode', '0'))]
                if len(matches) == 1:
                    beatmap_id = int(matches[0]['id'])
            if beatmap_id > 0:
                ids.add(beatmap_id)
            else:
                raise ValueError(f'Missing beatmap identifier: {chart}')
            if int(h.get('BeatmapSetID', '-1')) > 0:
                sets.add(int(h['BeatmapSetID']))
        if r.get('set_id'):
            sets.add(int(r['set_id']))
        if {str(s) for s in sets} & {str(s) for s in exclusion['set_ids']}:
            raise ValueError(f"Sealed set in requested list: {r['audio_sha256']}")
        if not kept_charts or not ids or not sets:
            raise ValueError(f"Missing chart identifiers: {r['audio_sha256']}")
        r.update(beatmap_ids=sorted(ids), set_ids=sorted(sets), charts=kept_charts)

    loaded = load_songs(training, index, bulk, bulk / 'labels-v0b')
    if len(loaded) != len(training):
        raise ValueError(f'Training loader accepted {len(loaded)}/{len(training)} curated inputs')
    expected = read_json(root / RUNS / 'v01-s0-20261020/config.json')
    if expected['seed'] != 0 or len(training) != expected['songs']:
        raise ValueError('Curation differs from the seed-0 training run')
    songs, members = {}, {}
    for r in rows:
        sha = r['audio_sha256']
        members.setdefault(sha, []).append(r)
        if sha not in songs:
            songs[sha] = inference_song(sha, index, bulk)
            s = songs[sha]
            if (not Path(s.mel).is_file() or not Path(s.memo).is_file() or s.frames <= 0
                    or not np.isfinite([s.mean, s.std]).all() or s.std <= 0):
                raise ValueError(f'Missing or invalid cached song input: {sha}')
    for sha, rs in members.items():
        if len({r['split'] for r in rs}) != 1:
            raise ValueError(f'Audio shared across requested splits: {sha}')
    return songs, members, {str(p.relative_to(root)): sha256_file(p) for p in paths}


def capture_song(model, song, device):
    """Observe the evaluator's linear call, stitching its tensors with the same edge rule.

    Returns float32 features, raw logits and the evaluator's original activations.
    The temporary hook is removed even when inference raises. No model state changes.
    """
    from .infer import song_outputs, windows
    n = song.frames
    total = n + (-n) % 4
    starts = windows(total)
    feat = np.empty((total, model.heads.lin.in_features), dtype=np.float32)
    logits = np.empty((total, model.heads.lin.out_features), dtype=np.float32)
    best = np.full(total, -1.0)
    cursor = 0

    def observe(module, inputs, output):
        nonlocal cursor
        h, y = inputs[0].float().cpu().numpy(), output.float().cpu().numpy()
        length = y.shape[1]
        centre = np.minimum(np.arange(length), length - 1 - np.arange(length))
        for j in range(len(y)):
            start = starts[cursor]
            take = centre > best[start:start + length]
            idx = np.flatnonzero(take) + start
            feat[idx], logits[idx], best[idx] = h[j, take], y[j, take], centre[take]
            cursor += 1

    hook = model.heads.lin.register_forward_hook(observe)
    try:
        acts = song_outputs(model, song, device, batch=8)
    finally:
        hook.remove()
    if cursor != len(starts) or (best < 0).any() or len(acts['beat']) != n:
        raise ValueError(f'Feature clock/window mismatch: {song.item}')
    return feat[:n], logits[:n], acts


def evidence_from_logits(y):
    """Apply the existing head nonlinearities to stored raw logits on the CPU."""
    import torch
    from .model import Heads, V0
    heads = Heads(y.shape[-1])
    heads.lin = torch.nn.Identity()
    with torch.inference_mode():
        return {k: v.numpy() for k, v in V0.activations(heads(torch.from_numpy(y.astype(np.float32)))).items()}


def differences(actual, expected):
    if set(actual) != set(expected):
        raise ValueError('E2 evidence fields differ')
    result = {}
    for key, value in actual.items():
        if value.shape != expected[key].shape or not np.isfinite(value).all():
            raise ValueError(f'E2 evidence shape or finite-value mismatch: {key}')
        delta = np.abs(value.astype(np.float64) - expected[key])
        result[key] = float(delta.max(initial=0))
        if key in ('phase', 'bar_phase'):
            result[key + '_circular'] = float(np.minimum(delta, 1 - delta).max(initial=0))
    return result


def reproduce(sha, logits, acts, saved, e2):
    with np.load(e2 / f'{sha}.npz', allow_pickle=False) as z:
        old = {k: z[k] for k in z.files}
    full = differences(acts, old)
    if not all(np.allclose(acts[k], old[k], atol=1e-6, rtol=1e-5) for k in old):
        raise ValueError(f'E2 float32 mismatch for {sha}: {full}')
    reconstructed = evidence_from_logits(logits)
    if not all(np.allclose(reconstructed[k], acts[k], atol=2e-6, rtol=1e-5) for k in acts):
        raise ValueError(f'Raw logit reconstruction mismatch for {sha}')
    with np.load(saved, allow_pickle=False) as z:
        stored = z['logits']
    if not np.array_equal(stored, logits.astype(np.float16)):
        raise ValueError(f'Stored logits differ from float16 conversion: {sha}')
    return dict(sha256=sha, full_precision_max_abs=full,
                cached_logits_evidence_max_abs=differences(evidence_from_logits(stored), old),
                raw_logit_float16_max_abs=float(np.abs(stored.astype(np.float32) - logits).max()))


def write_npz(path, feat, logits):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    arrays = dict(feat=feat.astype(np.float16), logits=logits.astype(np.float16),
                  t0_ms=np.float64(0), fps=np.float64(DEFAULT.fps))
    if not np.isfinite(arrays['feat']).all() or not np.isfinite(arrays['logits']).all():
        raise ValueError('Non-finite feature/logit after float16 conversion')
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def summary(out, records, planned, checks, elapsed, status, error):
    splits = {}
    for split in ('fit_train', 'fit_dev', 'val-n1'):
        rs = [r for r in records if r['split'] == split]
        splits[split] = dict(inputs=len(rs), planned=planned.get(split, 0),
                             recordings=len({m['rec'] for r in rs for m in r['memberships']}),
                             hours=sum(r['duration_s'] for r in rs) / 3600)
    size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    result = dict(status=status, error=error, splits=splits, bytes=size, wall_s=elapsed,
                  reproduction=checks, free_bytes=shutil.disk_usage(out).free)
    dump(out.parent / 'summary.json', result)
    text = [f'Status: {status}', '', '| Split | Inputs / planned | Recordings | Hours |',
            '| --- | ---: | ---: | ---: |']
    for name, s in splits.items():
        text.append(f"| {name} | {s['inputs']} / {s['planned']} | {s['recordings']} | {s['hours']:.4f} |")
    text += ['', f'Disk: {size / 1e9:.3f} GB ({size / 2**30:.3f} GiB). Wall: {elapsed / 60:.2f} minutes.',
             'Hours use the full cached audio frame coverage (T / 100), not the mapped chart span.',
             'fit_dev has 322 chart-set entries, 262 recording groups and 316 unique audio inputs.',
             'Clock: t0_ms = 0; frame i is at 10i ms on osu!\'s clock (osu-lead-v1).', '',
             f'E2 reproduction: {len(checks)} inputs; per-field maxima in summary.json.']
    if checks:
        fp = max(v for c in checks for v in c['full_precision_max_abs'].values())
        half = max(v for c in checks for v in c['cached_logits_evidence_max_abs'].values())
        text += [f'Max absolute evidence difference, float32: {fp:.9g}; from stored float16 logits: {half:.9g}.',
                 'The float16 difference includes logit rounding and the head nonlinearities; phase is in cycles,',
                 'offsets in milliseconds, and probabilities/concentrations retain their own units.']
    if error:
        text += ['', f'Stopped: {error}']
    text += ['', 'Only identifier metadata was used for exclusions; no sealed audio or labels were read.',
             'No training or probe experiment was run. Provenance and head layout are in every manifest row.']
    (out.parent / 'summary.md').write_text('\n'.join(text) + '\n')
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--check-only', action='store_true', help='Build and verify five panel inputs, then stop.')
    p.add_argument('--minutes', type=float, default=90, help='Wall limit for this invocation.')
    a = p.parse_args(argv)
    started = time.monotonic()
    root, out = a.root.resolve(), a.out.resolve()
    if not out.is_relative_to(root / 'artifacts'):
        p.error('--out must be inside DATA_ROOT/artifacts')
    if shutil.disk_usage(root).free < 140e9:
        p.error('Less than 80 GB would remain after reserving the 60 GB cache budget')
    out.mkdir(parents=True, exist_ok=True)
    records, checks, planned = [], [], {}
    status, error = 'stopped', None
    try:
        os.nice(10)
        import torch
        from .model import V0
        torch.set_num_threads(1)
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS is required')
        songs, members, hashes = prepare(root)
        planned = Counter(rs[0]['split'] for rs in members.values())
        raw_bytes = sum(s.frames * 366 for s in songs.values())
        if raw_bytes * 1.01 > 60e9:
            raise RuntimeError(f'Uncompressed size exceeds the 60 GB budget: {raw_bytes}')
        ckpt = root / 'artifacts/bulk/n1-runs/v01-s0-20261020/ckpt-009001.pt'
        e2 = root / 'artifacts/astra-consult-20261021/astra/exp/evidence/s0/009001'
        src = Path(__file__).resolve().parents[4]
        commit = subprocess.check_output(['git', '-C', str(src), 'rev-parse', 'HEAD'], text=True).strip()
        provenance = dict(checkpoint_id='v01-s0-20261020/ckpt-009001', checkpoint_sha256=sha256_file(ckpt),
                          code_commit=commit, script_sha256=sha256_file(__file__), input_sha256=hashes,
                          head_layout=HEAD_LAYOUT, t0_ms=0.0, fps=DEFAULT.fps, lead_rule=DEFAULT.lead_rule,
                          mel_config=DEFAULT.key())
        selected = sorted(sha for sha, rs in members.items() if rs[0]['split'] == 'fit_dev')
        selected = [selected[i] for i in np.linspace(0, len(selected) - 1, 5, dtype=int)]
        for sha in selected:
            if not (e2 / f'{sha}.npz').is_file():
                raise FileNotFoundError(f'Missing E2 evidence: {sha}')
        order = selected + ([] if a.check_only else sorted(set(songs) - set(selected)))
        old = {}
        manifest = out / 'manifest.jsonl'
        if manifest.exists():
            old = {r['sha256']: r for r in map(json.loads, manifest.read_text().splitlines())}
            if set(old) - set(songs):
                raise ValueError('Existing manifest contains inputs outside the allowed cohort')
            for r in old.values():
                if any(r.get(k) != v for k, v in provenance.items()):
                    raise ValueError('Existing cache provenance differs; use a fresh output directory')
        state = torch.load(ckpt, map_location='cpu', weights_only=True)
        model = V0(**(state.get('model_kw') or {}))
        model.load_state_dict(state['model'])
        model.eval().to('mps')
        size = sum((out / r['npz']).stat().st_size for r in old.values())
        print(json.dumps(dict(event='prepared', planned=planned, raw_gb=raw_bytes / 1e9)), flush=True)
        # Rewrite rows in inference order. Existing arrays are reused only with identical provenance.
        with manifest.open('w') as handle:
            for number, sha in enumerate(order, 1):
                if time.monotonic() - started >= a.minutes * 60:
                    raise TimeoutError('Inference wall budget reached')
                s = songs[sha]
                reserve = s.frames * 366 * 1.01 + 10000
                if shutil.disk_usage(out).free - reserve < 80e9 or size + reserve > 60e9:
                    raise RuntimeError('Disk budget reached (80 GB reserve / 60 GB cache)')
                path = out / sha[:2] / f'{sha}.npz'
                if sha not in old or sha in selected:
                    feat, logits, acts = capture_song(model, s, 'mps')
                    if feat.shape != (s.frames, 160) or logits.shape != (s.frames, 23):
                        raise ValueError(f'Unexpected model feature/head width: {sha}')
                    previous = path.stat().st_size if path.exists() else 0
                    write_npz(path, feat, logits)
                    size += path.stat().st_size - previous
                    if sha in selected:
                        check = reproduce(sha, logits, acts, path, e2)
                        checks.append(check)
                        dump(out.parent / 'reproduction.json', checks)
                        print(json.dumps(dict(event='reproduction', **check)), flush=True)
                    del feat, logits, acts
                rs = members[sha]
                row = dict(sha256=sha, split=rs[0]['split'], duration_s=s.frames / DEFAULT.fps,
                           frames=s.frames, npz=str(path.relative_to(out)),
                           set_ids=sorted({i for r in rs for i in r['set_ids']}),
                           beatmap_ids=sorted({i for r in rs for i in r['beatmap_ids']}),
                           memberships=[{k: r[k] for k in ('rec', 'audio_rel', 'set_ids', 'beatmap_ids', 'charts')}
                                        for r in rs], **provenance)
                handle.write(json.dumps(row) + '\n')
                handle.flush()
                records.append(row)
                if number % 16 == 0:
                    torch.mps.empty_cache()
                if number % 25 == 0 or number == len(order):
                    print(json.dumps(dict(done=number, total=len(order), gb=size / 1e9,
                                          wall_s=round(time.monotonic() - started, 1))), flush=True)
        status = 'checked' if a.check_only else 'complete'
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        print(error, flush=True)
    result = summary(out, records, planned, checks, time.monotonic() - started, status, error)
    print(json.dumps({k: v for k, v in result.items() if k != 'reproduction'}), flush=True)
    if error:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
