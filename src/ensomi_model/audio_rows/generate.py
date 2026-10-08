"""Audio to ``.osu``: corrected grid, the whole head skeleton beat by beat, then natural R2.

``decode`` encodes the song once, then takes one GRU step per canonical beat: the lattice and the
count are sampled from their categoricals, the slots by Gumbel-top-n over the lattice's slot
logits, so the sampled count holds; offsets are zero. Slots outside [0, T - 2 ms] are unavailable,
since R2 needs T at least 2 ms after the last head. A partial first or last beat's count is
limited to its available slots, so the recorded count equals the emitted heads.

Requests are log(W_H / s) per section (``data`` defines the label). ``--band`` asks every beat for
the fit_train median of that band; ``--sections`` overrides intervals with a JSON list such as
``[{"start_ms": 30000, "end_ms": 60000, "band": 5}, {"start_ms": 90000, "end_ms": 120000,
"log_wh_per_s": 2.27}]``. A request applies to the canonical beats starting in its half-open
interval; a later entry wins where entries overlap. Requests are workload targets, not star
ratings; the model attains them only statistically.

R2 receives the complete head list, K, the audio duration T and the corrected grid before row 0,
with no prefix, no request track and ``ln_level='unknown'``::

    python -m ensomi_model.audio_rows.generate --audio <file> --model <run>/model.pt \\
        --r2 <phase-N ckpt> --beatthis <beat_this-final0.ckpt> --band 3 --seeds 0 1 2 --out <dir>

``<dir>`` receives ``band-<b>-seed-<s>.osu`` per seed, the audio beside them (the ``.osu`` names
it) and ``band-<b>-seed-<s>.json`` with stage seconds and requested and realised workload per
16-beat section.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..r2.common import MIN_GAP_MS
from ..r2.export import export_chart, minimal_header
from ..r2.model import R2Config, R2Model
from ..r2.sampling import continue_chart
from .data import beat_workload, frame_indices, section_label, sections
from .grid import corrected_grid
from .lattice import MAX_COUNT, MAX_SLOTS, SLOTS, beat_range, heads_from_targets, slot_times
from .model import TOKEN_DIM, extra_inputs, load_model, token

@torch.no_grad()
def decode(model, feats: np.ndarray, grid, song_ms: float, label=None, seed: int = 0, temperature: float = 1.0):
    """(head times [K] ms, per-beat record) for a whole song.

    ``label``: standardised log(W_H / s) per beat of ``beat_range(grid, song_ms)``, or None for
    the unconditioned mode."""
    b0, nb = beat_range(grid, song_ms)
    beat_ms = np.clip(grid.time_of_beat(b0 + np.arange(nb + 1)), 0.0, song_ms)
    st = slot_times(grid, b0, nb)
    available = (st >= 0) & (st <= song_ms - MIN_GAP_MS)       # NaN slots compare False
    beat_frames, slot_frames = frame_indices(beat_ms, st, 0, len(feats))
    beat_vec, sv = model.beat_inputs(torch.from_numpy(feats.astype(np.float32)), beat_frames, slot_frames)
    extra = extra_inputs(b0 + np.arange(nb), beat_ms[:-1], grid.bpm(beat_ms[:-1]), song_ms, label)
    gen = torch.Generator().manual_seed(int(seed))
    h = None
    prev = torch.zeros(1, TOKEN_DIM)
    lattice = np.zeros(nb, dtype=np.int64)
    count = np.zeros(nb, dtype=np.int64)
    mask = np.zeros((nb, MAX_SLOTS), dtype=bool)
    for b in range(nb):
        out, h = model.run(beat_vec[b:b + 1], prev, extra[b:b + 1], h)
        l = int(torch.multinomial(F.softmax(model.lattice_logits(out) / temperature, -1)[0], 1, generator=gen))
        lat1 = F.one_hot(torch.tensor([l]), 2).float()
        ok = torch.from_numpy(available[b, l, :SLOTS[l]])
        count_logits = model.count_logits(out, lat1) / temperature
        count_logits[:, int(ok.sum()) + 1:] = -torch.inf
        n = int(torch.multinomial(F.softmax(count_logits, -1)[0], 1, generator=gen))
        if n:
            cnt1 = F.one_hot(torch.tensor([n]), MAX_COUNT + 1).float()
            logits = model.slot_logits(out, sv[b:b + 1, l], lat1, cnt1)[0, :SLOTS[l]] / temperature
            u = torch.rand(SLOTS[l], generator=gen).clamp_min(1e-12)
            top = torch.topk(logits.masked_fill(~ok, -torch.inf) - (-u.log()).log(), n).indices.numpy()
            mask[b, top] = True
        lattice[b], count[b] = l, n
        prev = token(torch.tensor([l]), torch.tensor([n]), torch.from_numpy(mask[b:b + 1]))
    heads = heads_from_targets(grid, b0, lattice, mask)
    return heads, dict(b0=b0, nb=nb, lattice=lattice, count=count, mask=mask, beat_ms=beat_ms)


def requests(grid, song_ms: float, band: int, bands: dict, overrides=()) -> np.ndarray:
    """log(W_H / s) requested per beat of ``beat_range(grid, song_ms)``."""
    b0, nb = beat_range(grid, song_ms)
    start = grid.time_of_beat(b0 + np.arange(nb))
    values = np.full(nb, bands[band])
    for s in overrides:
        if not 0 <= s['start_ms'] < s['end_ms'] <= song_ms:
            raise ValueError(f'Section {s} must satisfy 0 <= start_ms < end_ms <= {song_ms:.0f} ms')
        values[(start >= s['start_ms']) & (start < s['end_ms'])] = s.get('log_wh_per_s', bands[s.get('band', band)])
    return values


def realised(head_ms, song_ms: float, beat_ms, scopes) -> list[float]:
    """log(W_H / s) of generated heads per [a, b) beat scope, labelled as the training data is."""
    wh = beat_workload(head_ms, song_ms, beat_ms)
    return [section_label(wh, beat_ms, a, b) for a, b in scopes]


def load_r2(path) -> R2Model:
    data = torch.load(path, map_location='cpu', weights_only=False)
    model = R2Model(R2Config(**data['model_config'])).eval()
    model.load_state_dict(data['model'])
    return model


class Generator:
    """The models held once; ``prepare`` per song, ``generate`` per request and seed.

    ``record`` is the head model's training record (label normalisation and band requests);
    ``listener`` is an ``audio.Listener``."""

    def __init__(self, head, record, r2, listener):
        self.head, self.record, self.r2, self.listener = head, record, r2, listener

    @classmethod
    def load(cls, head_checkpoint, r2_checkpoint, beatthis_checkpoint):
        from .audio import Listener
        return cls(*load_model(head_checkpoint), load_r2(r2_checkpoint), Listener(beatthis_checkpoint))

    def prepare(self, audio) -> dict:
        song = self.listener.listen(audio)
        grid, seg = corrected_grid(song.segments, song.song_ms)
        return dict(audio=Path(audio), song=song, grid=grid, seg=seg)

    def generate(self, prepared, out, band: int, seed: int = 0, overrides=(), title='Audio rows') -> dict:
        song, grid = prepared['song'], prepared['grid']
        rec = self.record
        t0 = time.perf_counter()
        wanted = requests(grid, song.song_ms, band, rec['bands'], overrides)
        heads, beats = decode(self.head, song.features, grid, song.song_ms,
                              label=(wanted - rec['label_mean']) / rec['label_std'], seed=seed)
        t1 = time.perf_counter()
        actions, gap = continue_chart(self.r2, heads, song.song_ms, grid, seed=954 + seed, track=(), ln_level='unknown')
        t2 = time.perf_counter()
        out = Path(out)
        name = f'band-{band}-seed-{seed}'
        audio_name = 'audio' + prepared['audio'].suffix.lower()
        header = minimal_header(prepared['seg'], title).replace('[General]\n',
                                                               f'[General]\nAudioFilename:{audio_name}\n')
        export_chart(heads, song.song_ms, actions, gap, out / f'{name}.osu', header)
        if not (out / audio_name).exists():
            shutil.copy2(prepared['audio'], out / audio_name)
        t3 = time.perf_counter()
        scopes = sections(beats['nb'])
        dt = np.diff(beats['beat_ms'])
        result = dict(band=band, seed=seed, K=len(heads), song_ms=song.song_ms, raw_segments=song.segments.tolist(),
                      segments=prepared['seg'].tolist(),
                      seconds=dict(song.seconds, heads=t1 - t0, r2=t2 - t1, export=t3 - t2),
                      section_ms=[[float(beats['beat_ms'][a]), float(beats['beat_ms'][b])] for a, b in scopes],
                      requested_log_wh_per_s=[float(np.log(np.average(np.exp(wanted[a:b]), weights=dt[a:b])))
                                              for a, b in scopes],
                      realised_log_wh_per_s=realised(heads, song.song_ms, beats['beat_ms'], scopes))
        (out / f'{name}.json').write_text(json.dumps(result, indent=1))
        return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--audio', required=True)
    p.add_argument('--model', required=True, help='head model checkpoint from audio_rows.train')
    p.add_argument('--r2', required=True, help='R2 checkpoint')
    p.add_argument('--beatthis', required=True, help='BeatThis final0 checkpoint')
    p.add_argument('--out', required=True)
    p.add_argument('--band', type=int, choices=(2, 3, 4, 5), default=3)
    p.add_argument('--seeds', type=int, nargs='+', default=(0,))
    p.add_argument('--sections', help='JSON list of {start_ms, end_ms, band or log_wh_per_s}')
    p.add_argument('--threads', type=int, default=3)
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    t = time.perf_counter()
    generator = Generator.load(a.model, a.r2, a.beatthis)
    print(json.dumps(dict(setup_seconds=time.perf_counter() - t)), flush=True)
    prepared = generator.prepare(a.audio)
    overrides = json.loads(Path(a.sections).read_text()) if a.sections else ()
    for seed in a.seeds:
        result = generator.generate(prepared, a.out, a.band, seed, overrides)
        print(json.dumps(dict(seed=seed, K=result['K'], seconds=result['seconds'])), flush=True)


if __name__ == '__main__':
    main()
