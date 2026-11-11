"""Run SMC arms with explicit input manifests and isolated output directories.

Run from the checkout, never from a downloaded dataset directory, with
PYTHONPATH=n/src. Beat This and N1 use their existing trained frontends.
The `gate` command must succeed before other arms can be inferred or scored.
Outputs are local research artifacts, not repository sources.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ensomi_n.smc import (METRICS, SHIFTS_MS, events, failure_type, heldout_shifts,
                          map_beats, minimal_peaks, score, shift_curve, tempo_diagnostics)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def save_beats(out, arm, key, times, duration):
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError('predicted beats must be finite and strictly increasing')
    outside = (times < 0) | (times >= duration)
    times = events(times[times >= 0])
    p = out / 'beats' / arm / f'{key}.beats'
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(p, times, fmt='%.10f')
    return int(outside.sum())


def require_gate(out, manifest=None):
    p = out / 'gate.json'
    record = json.loads(p.read_text()) if p.exists() else {}
    if not record.get('passed'):
        raise RuntimeError('BT-cv gate must pass before running or reading other arms')
    if sha256(manifest or out / 'manifest.json') != record.get('manifest_sha256'):
        raise RuntimeError('manifest changed since the BT-cv gate; repeat verification and gate')


def prepare(a):
    """Verify audio/annotations and build a manifest from already extracted data."""
    import soundfile as sf
    canonical, bt = a.canonical.resolve(), a.annotations.resolve()
    pairs = [line.split() for line in (bt / '8-folds.split').read_text().splitlines()]
    folds = dict(pairs)
    if len(folds) != len(pairs):
        raise ValueError('duplicate fold assignments')
    rows, differences = [], []
    for audio in sorted((canonical / 'SMC_MIREX_Audio').glob('*.wav')):
        key = audio.stem.lower()
        candidates = list((canonical / 'SMC_MIREX_Annotations_05_08_2014').glob(audio.stem + '_*.txt'))
        tags = list((canonical / 'SMC_MIREX_Tags').glob(audio.stem + '*'))
        if len(candidates) != 1 or len(tags) != 1:
            raise ValueError(f'{key}: expected one corrected annotation and tag file')
        ann = candidates[0]
        bt_ann = bt / 'annotations' / 'beats' / f'{key}.beats'
        ref, other = events(np.loadtxt(ann, ndmin=1)), events(np.loadtxt(bt_ann, ndmin=1))
        info = sf.info(audio)
        if info.duration != 40 or info.channels != 1 or info.samplerate != 44100 or ref[-1] > info.duration:
            raise ValueError(f'{key}: unexpected audio format, duration or annotation range')
        if not np.array_equal(ref, other):
            differences.append(dict(key=key, canonical=ref.tolist(), beat_this=other.tolist()))
        rows.append(dict(key=key, audio=str(audio), annotation=str(ann), bt_annotation=str(bt_ann),
                         fold=int(folds[key]), duration=info.duration, rate=info.samplerate,
                         channels=info.channels, frames=info.frames, tag=tags[0].read_text(),
                         easy=int(key[-3:]) >= 271, audio_sha256=sha256(audio)))
    if len(rows) != 217 or set(folds) != {r['key'] for r in rows} or set(folds.values()) != set(map(str, range(8))):
        raise ValueError('expected exactly 217 excerpts assigned once across eight folds')
    write_json(a.out / 'annotation-differences.json', differences)
    if differences:
        raise ValueError('annotation differences need review before inference')
    write_json(a.out / 'manifest.json', rows)
    print(f'verified {len(rows)} excerpts, {sum(r["duration"] for r in rows)} s', flush=True)


def bt(a, rows):
    import torch
    from beat_this.inference import Audio2Frames
    from beat_this.model.postprocessor import Postprocessor
    from beat_this.preprocessing import load_audio
    torch.set_num_threads(a.threads)
    arm = a.arm
    if arm != 'BT-cv':
        require_gate(a.out, a.manifest)
    models = range(8) if arm == 'BT-cv' else [None]
    records = []
    t0 = time.time()
    for fold in models:
        path = a.models / f'fold{fold}.ckpt' if fold is not None else a.checkpoint
        ck = torch.load(path, map_location='cpu', weights_only=True)
        data_kw = ck.get('datamodule_hyper_parameters', {})
        if fold is not None and data_kw.get('fold') != fold:
            raise ValueError(f'{path}: checkpoint fold {data_kw.get("fold")} != {fold}')
        records.append(dict(path=str(path.resolve()), sha256=sha256(path), datamodule=data_kw,
                            hyper_parameters=ck.get('hyper_parameters', {})))
        del ck
        frames = Audio2Frames(str(path), device=a.device, float16=False)
        post = Postprocessor()
        subset = [r for r in rows if fold is None or r['fold'] == fold]
        for i, row in enumerate(subset):
            signal, sr = load_audio(row['audio'])
            b, d = frames(signal, sr)
            b, d = b.cpu(), d.cpu()
            times, downbeats = post(b, d)
            outside = save_beats(a.out, arm, row['key'], times, row['duration'])
            cache = a.out / 'activations' / arm
            cache.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache / f'{row["key"]}.npz', beat=b.numpy(), down=d.numpy())
            if i % 10 == 0:
                print(arm, fold, i, row['key'], 'seconds', round(time.time() - t0, 1), flush=True)
            if outside:
                print('out-of-audio beats', row['key'], outside, flush=True)
        del frames
        if a.device == 'mps':
            torch.mps.empty_cache()
    write_json(a.out / f'{arm}-provenance.json', dict(models=records, seconds=time.time() - t0,
                                                    device=a.device, float16=False))


def n1(a, rows):
    require_gate(a.out, a.manifest)
    import torch
    from ensomi_n.features.decode import decode as decode_audio
    from ensomi_n.features.mel import log_mel
    from ensomi_n.v0.data import Song
    from ensomi_n.v0.infer import song_outputs
    from ensomi_n.v0.model import V0
    from ensomi_n.v0.songs import memo_from_mel
    torch.set_num_threads(a.threads)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    model = V0(**ck['model_kw'])
    model.load_state_dict(ck['model'])
    model.to(a.device).eval()
    metadata = dict(checkpoint=str(a.checkpoint.resolve()), sha256=sha256(a.checkpoint), seed=ck['seed'],
                    step=ck['state']['step'], elapsed=ck['state']['elapsed'], model_kw=ck['model_kw'],
                    parameters=sum(p.numel() for p in model.parameters()), device=a.device)
    del ck
    cache = a.out / 'activations' / a.arm
    cache.mkdir(parents=True, exist_ok=True)
    inputs = a.out / 'n1-input'
    inputs.mkdir(exist_ok=True)
    t0 = time.time()
    for i, row in enumerate(rows):
        mp, memo = inputs / f'{row["key"]}.mel.npy', inputs / f'{row["key"]}.memo.npy'
        if not mp.exists():
            decoded = decode_audio(row['audio'])
            if decoded.lead_samples != 0:
                raise ValueError('SMC WAV should not acquire an MP3 lead')
            mel = log_mel(decoded.wave).astype(np.float16)
            np.save(mp, mel)
            np.save(memo, memo_from_mel(mel))
        mel = np.load(mp).astype(np.float32)
        song = Song(row['key'], str(mp), str(memo), float(mel.mean()), float(mel.std() + 1e-6), len(mel), None)
        acts = song_outputs(model, song, a.device)
        if len(acts['beat']) != 4000 or any(not np.isfinite(x).all() for x in acts.values()):
            raise ValueError(f'{row["key"]}: invalid 40 s inference output')
        np.savez_compressed(cache / f'{row["key"]}.npz', **acts)
        times = minimal_peaks(acts['beat'], fps=100)
        save_beats(a.out, a.arm + '-peaks', row['key'], times, row['duration'])
        if i % 10 == 0:
            print(a.arm, i, row['key'], 'seconds', round(time.time() - t0, 1), flush=True)
    metadata['seconds'] = time.time() - t0
    write_json(a.out / f'{a.arm}-provenance.json', metadata)


def decode_one(job):
    out, arm, variant, row = job
    from ensomi_n.v0.decode import DecodeConfig, Evidence, decode
    from ensomi_n.v0.evaluate import bt_evidence, v0_evidence
    cache = out / 'activations' / arm / f'{row["key"]}.npz'
    if arm == 'BT-cv':
        evidence = bt_evidence(cache, 0., int(round(row['duration'] * 100)))
        name = 'BT-cv-n1decode'
    else:
        with np.load(cache) as z:
            evidence = (Evidence(beat=z['beat'], down=z['down']) if variant == 'beatonly'
                        else v0_evidence(dict(z), f1_only=variant == 'f1decode'))
        if variant == 'canonical':
            evidence = replace(evidence, level=None)
        name = arm + '-' + variant
    music_map, diagnostic = decode(evidence, DecodeConfig(reset_penalty=14.))
    times, outside = map_beats(music_map, row['duration'])
    save_beats(out, name, row['key'], times, row['duration'])
    write_json(out / 'maps' / name / f'{row["key"]}.json', music_map.to_json())
    return dict(key=row['key'], arm=name, outside_audio=outside, **diagnostic)


def decode_arm(a, rows):
    require_gate(a.out, a.manifest)
    import multiprocessing as mp
    variants = a.variants.split(',') if a.arm != 'BT-cv' else ['full']
    if set(variants) - {'full', 'canonical', 'f1decode', 'beatonly'}:
        raise ValueError(f'unknown decoder variants: {variants}')
    jobs = [(a.out, a.arm, v, row) for v in variants for row in rows]
    result = []
    with mp.get_context('spawn').Pool(a.processes) as pool:
        for i, r in enumerate(pool.imap_unordered(decode_one, jobs)):
            result.append(r)
            if i % 25 == 0:
                print('decode', a.arm, i, '/', len(jobs), flush=True)
    write_json(a.out / f'{a.arm}-decode-{"-".join(variants)}.json', result)


def score_arm(a, rows):
    if a.arm != 'BT-cv':
        require_gate(a.out, a.manifest)
    result, curves = [], []
    for row in rows:
        ref = events(np.loadtxt(row['annotation'], ndmin=1))
        pred = events(np.loadtxt(a.out / 'beats' / a.arm / f'{row["key"]}.beats', ndmin=1))
        metrics = score(ref, pred)
        result.append(dict(key=row['key'], fold=row['fold'], easy=row['easy'], tag=row['tag'],
                           metrics=metrics, failure=failure_type(metrics), n_prediction=len(pred),
                           **tempo_diagnostics(ref, pred)))
        curves.append(shift_curve(ref, pred))
    curves = np.array(curves)
    output = dict(arm=a.arm, n=len(rows), mean={k: float(np.mean([r['metrics'][k] for r in result])) for k in METRICS},
                  rows=result, shifts_ms=SHIFTS_MS.tolist(), shift_curves=curves.tolist())
    if a.arm.startswith('N1'):
        shifts = heldout_shifts(curves, [r['fold'] for r in rows])
        shifted = []
        for row, s in zip(rows, shifts):
            ref = np.loadtxt(row['annotation'], ndmin=1)
            pred = np.loadtxt(a.out / 'beats' / a.arm / f'{row["key"]}.beats', ndmin=1)
            shifted.append(dict(key=row['key'], shift_ms=float(s), metrics=score(ref, pred, shift_ms=s)))
        output['heldout_shift'] = dict(rows=shifted, mean={k: float(np.mean([r['metrics'][k] for r in shifted])) for k in METRICS})
    write_json(a.out / 'scores' / f'{a.arm}.json', output)
    print(a.arm, output['mean'], flush=True)
    return output


def gate(a, rows):
    a.arm = 'BT-cv'
    result = score_arm(a, rows)
    reproduced = result['mean']['F-measure']
    passed = abs(reproduced - .627) <= .01
    record = dict(passed=passed, published=.627, reproduced=reproduced, tolerance=.01,
                  table='Foscarin et al., ISMIR 2024, Table 1', n=len(rows),
                  manifest_sha256=sha256(a.manifest), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    write_json(a.out / 'gate.json', record)
    if not passed:
        raise RuntimeError(f'BT-cv reproduction gate failed: {reproduced}')
    print('gate passed', record, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'bt', 'n1', 'decode', 'score', 'gate'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--canonical', type=Path)
    p.add_argument('--annotations', type=Path)
    p.add_argument('--arm')
    p.add_argument('--models', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--device', default='mps')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--processes', type=int, default=4)
    p.add_argument('--variants', default='full,canonical,f1decode')
    a = p.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.command == 'prepare':
        return prepare(a)
    a.manifest = a.manifest or a.out / 'manifest.json'
    rows = json.loads(a.manifest.read_text())
    if len(rows) != 217 or len({r['key'] for r in rows}) != 217:
        raise ValueError('expected all 217 unique SMC excerpts')
    commands = dict(bt=bt, n1=n1, decode=decode_arm, score=score_arm, gate=gate)
    commands[a.command](a, rows)


if __name__ == '__main__':
    main()
