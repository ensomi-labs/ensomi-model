#!/usr/bin/env python3
# Generation flow adapted from Keytoyze/Mug-Diffusion webui.py.
# Upstream notice for that code:
# MIT License
#
# Copyright (c) 2023 MuG Diffusion developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Generate one CPU Mug chart. Weights and outputs are non-commercial; keep private."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import resource
import sys
import time


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--star', type=float, required=True)
    parser.add_argument('--arm', choices=('natural', 'high-ln', 'source'), required=True)
    parser.add_argument('--ln-ratio', type=float, help='Source LN-head share, required for the source arm')
    parser.add_argument('--ln-tag', action='store_true', help='Add the LN tag to the source arm')
    parser.add_argument('--output', type=Path, required=True, help='New .osu file; also writes .json')
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    args.audio, args.output = args.audio.resolve(), args.output.resolve()
    if not args.audio.is_file() or not math.isfinite(args.star) or not 1 <= args.star <= 8:
        parser.error('Audio must exist and star must be in [1, 8]')
    if args.seed < 0 or args.output.suffix != '.osu':
        parser.error('Seed must be nonnegative and output must end in .osu')
    if args.arm == 'source':
        if args.ln_ratio is None or not math.isfinite(args.ln_ratio) or not 0 <= args.ln_ratio <= 1:
            parser.error('The source arm requires --ln-ratio in [0, 1]')
    elif args.ln_ratio is not None or args.ln_tag:
        parser.error('--ln-ratio and --ln-tag apply only to the source arm')
    if any(path.exists() for path in (args.output, args.output.with_suffix('.json'),
                                      args.output.with_suffix('.raw-hitobjects.txt'))):
        parser.error('Output, its receipt or raw-object sidecar already exists')
    repo = Path(os.environ['MUG_DIFFUSION_REPO']).resolve()
    weights = Path(os.environ['MUG_DIFFUSION_WEIGHTS']).resolve()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[name] = '2'
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.setpriority(os.PRIO_PROCESS, 0, 10)
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    import numpy as np
    import torch
    import yaml
    from omegaconf import OmegaConf
    from scripts.mapping import load_model_from_config
    from mug.data.convertor import parse_osu_file, save_osu_file
    from mug.data.utils import gridify, remove_intractable_mania_mini_jacks
    from mug.diffusion.ddim import DDIMSampler
    from mug.util import feature_dict_to_embedding_ids, load_audio_without_cache

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = OmegaConf.load(weights / 'model.yaml')
    model = load_model_from_config(cfg, str(weights / 'model.ckpt'), verbose=True).cpu()
    features = {'sr': args.star}
    if args.arm == 'high-ln':
        features.update(ln_ratio=0.8, ln=1)
    elif args.arm == 'source':
        features['ln_ratio'] = args.ln_ratio
        if args.ln_tag:
            features['ln'] = 1
    feature_yaml = yaml.safe_load((repo / 'configs/mug/mania_beatmap_features.yaml').read_text())

    def condition(values):
        ids = feature_dict_to_embedding_ids(values, feature_yaml)
        return model.model.cond_stage_model(torch.tensor([ids], dtype=torch.float32))

    with torch.no_grad():
        dataset = cfg.data.params.common_params
        frame_s = dataset.n_fft / 4 / dataset.sr
        audio = load_audio_without_cache(str(args.audio), dataset.n_mels, dataset.n_fft // 4,
                                         dataset.n_fft, dataset.sr, None)
        # Match webui.py: expand the latent sequence instead of truncating at the training window.
        ratio = dataset.max_audio_frame // model.z_length
        frames = audio.shape[1]
        model.z_length = (int(frames / ratio / 32) + 1) * 32
        padded_frames = model.z_length * ratio
        audio = np.pad(audio, ((0, 0), (0, padded_frames - frames)))
        wave = model.model.wave_model(torch.from_numpy(audio[None]).float())
        samples, _ = DDIMSampler(model).sample(
            S=100, c=condition(features), w=wave, batch_size=1,
            unconditional_guidance_scale=5.0, unconditional_conditioning=condition({}),
            eta=0.0, verbose=False)
        notes = model.model.decode(samples).cpu().numpy()[0]
    _, meta = parse_osu_file(str(repo / 'asset/template.osu'), {
        'frame_ms': frame_s * dataset.audio_note_window_ratio * 1000,
        'max_frame': padded_frames // dataset.audio_note_window_ratio, 'from_logits': True})
    raw = meta.convertor.array_to_objects(notes, meta)
    postprocess = {}

    def refine(objects):
        snapped, bpm, offset = gridify(objects, verbose=False)
        final = remove_intractable_mania_mini_jacks(snapped, verbose=True, jack_interval=90)
        removed = list((Counter(snapped) - Counter(final)).elements())
        postprocess.update(raw_objects=len(objects), snapped_objects=len(snapped), final_objects=len(final),
                           removed_or_moved_objects=len(removed),
                           removed_or_moved_holds=sum(int(s.split(',')[3]) & 128 != 0 for s in removed))
        return bpm, offset, final

    # Record the unrefined objects for private diagnosis without changing the model's postprocessing.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.raw-hitobjects.txt').write_text('\n'.join(raw) + '\n')
    save_osu_file(meta, notes, path=str(args.output), gridify=refine, override={
        'Title': args.audio.stem, 'TitleUnicode': args.audio.stem, 'Artist': 'Unknown',
        'ArtistUnicode': 'Unknown', 'AudioFilename': args.audio.name,
        'Creator': f'MuG Diffusion v{cfg.version}', 'Version': args.arm,
        'AIMode': f'MuG Diffusion v{cfg.version}', 'AIPrompt': json.dumps(features, sort_keys=True)})
    # The frame converter silently ignores orphan held frames and terminates holds at the array end.
    starts, held = notes[:4] > 0, notes[8:12] > 0
    orphan = np.zeros_like(held)
    for lane in range(4):
        active = False
        for i in range(notes.shape[1]):
            orphan[lane, i] = bool(held[lane, i] and not starts[lane, i] and not active)
            active = bool(starts[lane, i] or (active and held[lane, i]))
    boundary_lanes = np.flatnonzero(held[:, -1] & ~starts[:, -1] & ~orphan[:, -1]).tolist()
    boundary_heads = []
    for lane in boundary_lanes:
        index = int(np.flatnonzero(starts[lane])[-1])
        start_ms = int(round((index + np.clip(notes[4 + lane, index], 0, 1)) * meta.convertor.frame_ms))
        boundary_heads.append(dict(lane=lane, start_ms=start_ms))
    receipt = dict(baseline=f'Mug-Diffusion-{cfg.version}', audio=str(args.audio), star=args.star,
                   arm=args.arm, seed=args.seed, device='cpu', threads=2, features=features,
                   feature_embedding_ids=feature_dict_to_embedding_ids(features, feature_yaml),
                   steps=100, cfg_scale=5.0, eta=0.0, snap=True, minijack_ms=90,
                   weights=str(weights), repo=str(repo), supplied_timing=False,
                   audio_frames=frames, padded_audio_frames=padded_frames, latent_length=model.z_length,
                   frame_ms=meta.convertor.frame_ms, postprocess=postprocess,
                   orphan_held_frames=int(orphan.sum()),
                   held_lanes_at_array_end=len(boundary_lanes), array_terminated_heads=boundary_heads,
                   wall_seconds=time.monotonic() - started,
                   peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                   * (1 if sys.platform == 'darwin' else 1024))
    args.output.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
