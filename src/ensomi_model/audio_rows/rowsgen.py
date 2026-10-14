"""Whole-song head rows from the figure generator, one canonical beat at a time.

``decode`` encodes the song's audio once, then per beat runs one GRU step, mixes the generator
and the copy heads (``net``), removes figures with a position outside [0, T - 2 ms] (R2 needs T at
least 2 ms after the last head) or less than a sixteenth of a beat before the next beat
(``available_positions``), and samples a figure. The generator's logits are divided by
``temperature``; the copy heads are not. Heads sit exactly on the figure positions.

Beats are those of the grid's whole-beat frame (``GridArrays.whole_beats``), so every position lies
on its own segment's beat lines; on the grid's continuous beats, positions after a boundary at a
fractional beat would sit that fraction of a beat off them. A leftover under half a beat before a
boundary gets no positions; a longer one is a beat that the boundary cuts short, of which only the
positions at least a sixteenth of a beat before the boundary are available, the space a whole beat
leaves after its last position. R2 reads the continuous beats, the coordinate it was trained on.

The copy heads never attend to empty beats while decoding, whatever the training recipe
(``copy_rests`` turns that off for diagnostics). With copied rests, emptiness reinforced itself in
free-running long songs: on the pilot weights the last fifth of long songs came out .89 below the
request, and +.01 with the heads kept off empty beats (human charts -.27).

``quiet_rests`` relaxes the rule where the music is quiet; the default is .40 (``QUIET_RESTS``, adopted
2026-10-11; 0 restores the strict rule). On the 96-song panel (full2-s1 5000, first chart per song), .40 raised
the rest share of quiet interior sections from .053 to .076 (human .085); the last fifth of the 16 long songs
was at -.10 (rule on: -.00, human -.15).

Section requests also steer at decode time: the learned section input moved a section by only
16-21 % of its request in the pilot. Inside a section, every figure's log-probability gains
``tilt`` x its head count, with ``tilt`` = ``section_gain`` x (request - realised), clipped to
±``TILT_MAX``. Realised is log W_H/s over the section's beats so far, extended back to the last 4
beats while the section is shorter; W_H uses the reference strain of the rows decoded so far
(``r2.strain``). A proportional rule on the running mean: integrating the error overshot.

Requests are log(W_H / s) (``beatdata``): ``request`` for the song, ``sections`` a list of
(start_ms, end_ms, log W_H/s) applied to the beats starting in [start_ms, end_ms).
``star_request`` maps a star rating to a song request with the training record's map.

``Rows`` is the interface the running system uses: audio file, request and seed to the Listener
grid, the head times on its positions and K, in the form R2's sampler takes
(``r2.sampling.continue_chart(r2, head_ms, song_ms, grid)``). The CLI writes one JSON per seed and,
with ``--r2``, an ``.osu`` arranged by natural R2 for viewing::

    python -m ensomi_model.audio_rows.rowsgen --audio <file> --model <run>/model.pt \
        --beatthis <beat_this-final0.ckpt> --star 3.5 --seeds 0 1 --out <dir> [--r2 <ckpt>] \
        [--sections '[{"start_ms": 30000, "end_ms": 60000, "star": 4.5}]']
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from ..r2.common import MIN_GAP_MS
from ..r2.strain import REFERENCE_LANES, StrainState
from .audio import HOP_MS, MEL_BINS
from .data import LABEL_FLOOR
from .figures import POSITIONS, head_times
from .lattice import beat_range
from .net import FigureNet, load_net


SECTION_GAIN = 3.0      # tilt per unit of log W_H/s that the section so far lies below its request
TILT_MAX = 2.0
QUIET_RESTS = 0.40      # default ``quiet_rests``: copied rests allowed on the quietest 40 % of beats
REST_WINDOW = 4         # beats on each side of a beat for its local intensity (``quiet_rests``)


def star_request(record, star: float) -> float:
    m = record['star_map']
    return float(np.interp(star, m['star'], m['request']))


def song_inputs(grid, song_ms: float, record, request: float, sections=()):
    """Per-beat inputs of a whole song on ``grid``: b0, edges [nb+1], position times [nb, 24], extra [nb, E]."""
    b0, nb = beat_range(grid, song_ms)
    edges = grid.time_of_beat(b0 + np.arange(nb + 1))
    pos_ms = grid.time_of_beat(b0 + np.arange(nb)[:, None] + POSITIONS[None] / 48.0)
    beat_ms = np.clip(edges[:-1], 0.0, song_ms)
    std = lambda v: (v - record['label_mean']) / record['label_std']
    sec = np.zeros((nb, 2))
    for a, b, value in sections:
        sec[(beat_ms >= a) & (beat_ms < b)] = (std(value), 1.0)
    extra = np.column_stack((np.eye(4)[(b0 + np.arange(nb)) % 4], beat_ms / song_ms, grid.bpm(beat_ms) / 120.0,
                             np.full(nb, std(request)), sec)).astype(np.float32)
    return b0, edges, pos_ms, extra


def available_positions(grid, edges, pos_ms, song_ms: float) -> np.ndarray:
    """[nb, 24] positions a figure may use: inside [0, T - 2 ms], and at least a sixteenth of its segment's beat
    before the next beat starts. A whole beat's last position (15/16) is exactly that far from the next beat, so
    only a beat cut short by the next segment loses positions, and none lands nearer that segment's first one."""
    end = edges[1:] - grid.lengths[grid.segment(edges[:-1])] / 16 + 1e-6       # 1e-6 ms: float error at 15/16
    return (pos_ms >= 0) & (pos_ms <= song_ms - MIN_GAP_MS) & (pos_ms <= end[:, None])


def quiet_rank(mel, beat_frames, window: int = REST_WINDOW) -> np.ndarray:
    """[nb] rank among the song's beats (0 quietest, 1 loudest) of each beat's local intensity: the mean
    standardised log-Mel over the 128 bins (``measures.audio_curves``) from beat b - ``window`` to b + ``window``."""
    nb = len(beat_frames)
    cs = np.concatenate(([0.0], np.cumsum(np.asarray(mel[:, :MEL_BINS], dtype=np.float32).mean(1), dtype=np.float64)))
    lo = beat_frames[np.maximum(np.arange(nb) - window, 0), 0]
    hi = beat_frames[np.minimum(np.arange(nb) + window, nb - 1), 1]
    return np.argsort(np.argsort((cs[hi] - cs[lo]) / np.maximum(hi - lo, 1), kind='stable')) / max(nb - 1, 1)


@torch.no_grad()
def decode(model: FigureNet, record, mel, grid, song_ms: float, request: float, sections=(), seed: int = 0,
           temperature: float = 1.0, off=(), copy_rests: bool = False, section_gain: float = SECTION_GAIN,
           quiet_rests: float = QUIET_RESTS, *, recorder=None, random_source=None):
    """(head times [K] ms, per-beat record) for a whole song; ``mel`` [F, >=128] standardised log-Mel.

    ``off``: 'copy' forces the gate to the generator (diagnostics). ``copy_rests`` lets the copy
    heads attend to empty beats, ``section_gain`` 0 turns the section tilt off (diagnostics; see the
    module docstring). ``quiet_rests`` u lets the copy heads attend to empty beats on beats whose
    ``quiet_rank`` is below u, so rests copy where the music is quiet (default ``QUIET_RESTS``; 0: never)."""
    cfg = model.cfg
    grid = grid.whole_beats()
    b0, edges, pos_ms, extra = song_inputs(grid, song_ms, record, request, sections)
    nb = len(extra)
    beat_ms = np.clip(edges, 0.0, song_ms)
    target = np.full(nb, np.nan)
    for a, b, value in sections:
        target[(beat_ms[:-1] >= a) & (beat_ms[:-1] < b)] = value
    steer = bool(section_gain) and not np.isnan(target).all()
    counts = model.fig_counts.float()
    strain, n_heads, sec_b0 = StrainState(), 0, 0
    wh = np.zeros(nb)
    tilts = np.zeros(nb, dtype=np.float32)
    nf = len(mel)
    pos_frames = np.clip((pos_ms / HOP_MS).astype(np.int64), 0, nf - 1)
    bf = np.clip((edges / HOP_MS).astype(np.int64), 0, nf)
    beat_frames = np.stack((bf[:-1], np.maximum(bf[1:], bf[:-1] + 1)), 1).clip(0, nf)
    available = available_positions(grid, edges, pos_ms, song_ms)
    members = model.members.bool().numpy()
    allowed = torch.from_numpy(~(members[None] & ~available[:, None]).any(-1))          # [nb, V]
    t = lambda x: torch.from_numpy(np.asarray(x))[None]
    ex = t(extra)
    audio_inputs = dict(mel=t(np.asarray(mel[:, :MEL_BINS], dtype=np.float32)), pos_frames=t(pos_frames),
                        beat_frames=t(beat_frames), extra=ex, valid=torch.ones(1, nb, dtype=torch.bool))
    if recorder is None:
        pos, A, q, k = model.audio(*audio_inputs.values())
    else:
        from ..export.rows_graph import RowsSongEncoder, RowsStep
        encoder, step = RowsSongEncoder(model), RowsStep(model)
        def encode():
            return dict(zip(('pos', 'context', 'query', 'key'), encoder(*audio_inputs.values())))
        encoded = recorder.graph('rows_encoder', audio_inputs, encode)
        pos, A, q, k = (encoded[key] for key in ('pos', 'context', 'query', 'key'))
        recorder.event('rows/grid', edges=edges, position_ms=pos_ms, available=available, allowed=allowed,
                       beat_frames=beat_frames, target=target)
    pos, A, q, k, ex = pos[0], A[0], q[0], k[0], ex[0]
    V, lags = cfg.vocab, cfg.lags
    quiet = quiet_rank(mel, beat_frames) < quiet_rests
    gen = torch.Generator().manual_seed(int(seed))
    ids = np.full(nb, -1, dtype=np.int64)
    gates = np.zeros((nb, 1 + cfg.heads), dtype=np.float32)
    h = None
    prev = V
    table_in, table_out = model.input_table(), model.output_table()
    for b in range(nb):
        j = np.arange(b - 1, b - lags - 1, -1)
        ok = j >= 0
        jj = np.where(ok, j, 0)
        back_ids = torch.from_numpy(np.where(ok, ids[jj], -1))
        eq = []
        for d in (1, 2):
            before = ids[b - d] if b - d >= 0 else -1
            bj = np.where(jj - d >= 0, ids[np.maximum(jj - d, 0)], -1)
            eq.append((ok & (bj == before) & (before >= 0)).astype(np.float32))
        att_ok = ok if copy_rests or quiet[b] else ok & (np.where(ok, ids[jj], -1) != 0)
        if recorder is None:
            x = torch.cat((A[b], table_in[prev], ex[b]))[None, None]
            out, h = model.gru(x, h)
            hb = out[0]
            logits = model.generator_logits(hb, pos[b:b + 1], table=table_out) / temperature
            scores = model.copy_scores(q[b:b + 1], k[torch.from_numpy(jj)][None],
                                       torch.from_numpy(np.stack(eq, -1))[None])
            g, logp, att = model.mixture(hb, logits, scores, torch.from_numpy(att_ok)[None])
        else:
            inputs = dict(context=A[b:b + 1], pos=pos[b:b + 1], query=q[b:b + 1], extra=ex[b:b + 1],
                          prev=torch.tensor([prev]), hidden=torch.zeros(1, 1, cfg.hidden) if h is None else h,
                          back_keys=k[torch.from_numpy(jj)][None], eq=torch.from_numpy(np.stack(eq, -1))[None],
                          back_ok=torch.from_numpy(att_ok)[None], temperature=torch.tensor(temperature, dtype=torch.float32))
            def run_step():
                return dict(zip(('hidden_out', 'hb', 'logits', 'scores', 'gate', 'logp', 'attention'),
                                step(*inputs.values())))
            output = recorder.graph('rows_step', inputs, run_step)
            h, g, logp, att = (output[key] for key in ('hidden_out', 'gate', 'logp', 'attention'))
        p_gen = logp[0].exp()
        if 'copy' in off:
            p = p_gen
        else:
            p = g[0, 0] * p_gen
            okt = torch.from_numpy(att_ok)
            for hd in range(cfg.heads):
                p_h = (att[0, hd, -1] * p_gen).index_add(0, back_ids[okt], att[0, hd, :-1][okt])
                p = p + g[0, 1 + hd] * p_h
        if recorder is not None:
            recorder.event(f'rows/{b}/mixture', generator=p_gen, mixed=p, back_ids=back_ids,
                           quiet=quiet[b], copy_allowed=att_ok)
        p = p * allowed[b]
        if recorder is not None:
            recorder.event(f'rows/{b}/allowed', probability=p)
        if steer and not np.isnan(target[b]):
            if b == 0 or target[b - 1] != target[b]:         # a section starts (NaN never equals)
                sec_b0 = b
            a = max(min(sec_b0, b - 4), 0)
            if b > a and beat_ms[b] > beat_ms[a]:
                realised = np.log(max(wh[a:b].sum() / ((beat_ms[b] - beat_ms[a]) / 1000.0), LABEL_FLOOR))
                tilts[b] = np.clip(section_gain * (target[b] - realised), -TILT_MAX, TILT_MAX)
                p = p * torch.exp(float(tilts[b]) * counts)
        probability = p / p.sum()
        f = (int(torch.multinomial(probability, 1, generator=gen)) if random_source is None else
             random_source.multinomial(probability, gen, f'rows/{b}/figure'))
        if recorder is not None:
            recorder.event(f'rows/{b}/sample', probability=probability, tilt=tilts[b], chosen=f)
        ids[b] = f
        gates[b] = g[0].numpy()
        prev = f
        if steer:
            for t_ms in np.round(pos_ms[b][members[f]], 3):
                strain, charge = strain.append(t_ms, np.arange(4) == REFERENCE_LANES[n_heads % 4])
                n_heads += 1
                wh[b] += charge
    figures = np.asarray(record['vocab'], dtype=np.int64)[ids]
    heads = np.unique(head_times(grid, b0, figures))
    return heads, dict(b0=b0, nb=nb, ids=ids, figures=figures, edges=edges, gates=gates, tilts=tilts)


def section_requests(record, sections) -> list:
    """(start_ms, end_ms, log W_H/s) from dicts with start_ms, end_ms and star or request."""
    return [(s['start_ms'], s['end_ms'], s['request'] if 'request' in s else star_request(record, s['star']))
            for s in sections]


class Rows:
    """The rows model and the Listener held once; ``prepare`` per audio file, ``heads`` per request and seed."""

    def __init__(self, model, record, listener):
        self.model, self.record, self.listener = model, record, listener

    @classmethod
    def load(cls, checkpoint, beatthis_checkpoint):
        from .audio import Listener
        return cls(*load_net(checkpoint), Listener(beatthis_checkpoint))

    def prepare(self, audio, *, recorder=None) -> dict:
        from .grid import corrected_grid
        song = self.listener.listen(audio) if recorder is None else self.listener.listen(audio, recorder=recorder)
        grid, seg = corrected_grid(song.segments, song.song_ms)
        return dict(audio=Path(audio), song=song, grid=grid, seg=seg, song_ms=song.song_ms)

    def heads(self, prepared, *, star=None, request=None, sections=(), seed=0, temperature=1.0,
              recorder=None, random_source=None) -> dict:
        """Head times on the Listener grid. ``star`` or ``request`` (log W_H/s) sets the song request;
        ``sections`` are dicts with start_ms, end_ms and star or request."""
        if request is None:
            request = star_request(self.record, star)
        t = time.perf_counter()
        heads, beats = decode(self.model, self.record, prepared['song'].features, prepared['grid'], prepared['song_ms'],
                              request, section_requests(self.record, sections), seed, temperature,
                              recorder=recorder, random_source=random_source)
        return dict(head_ms=heads, K=len(heads), grid=prepared['grid'], segments=prepared['seg'],
                    song_ms=prepared['song_ms'], request=request, beats=beats, seconds=time.perf_counter() - t)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--audio', required=True)
    p.add_argument('--model', required=True, help='figure generator checkpoint from audio_rows.fit')
    p.add_argument('--beatthis', required=True, help='BeatThis final0 checkpoint')
    p.add_argument('--out', required=True)
    q = p.add_mutually_exclusive_group(required=True)
    q.add_argument('--star', type=float)
    q.add_argument('--request', type=float, help='song log(W_H / s)')
    p.add_argument('--sections', default='[]', help='JSON list of {start_ms, end_ms, star or request}')
    p.add_argument('--seeds', type=int, nargs='+', default=(0,))
    p.add_argument('--r2', help='R2 checkpoint: also write an .osu arranged by natural R2')
    p.add_argument('--threads', type=int, default=3)
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    rows = Rows.load(a.model, a.beatthis)
    prepared = rows.prepare(a.audio)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r2 = None
    if a.r2:
        from .generate import load_r2
        r2 = load_r2(a.r2)
    for seed in a.seeds:
        r = rows.heads(prepared, star=a.star, request=a.request, sections=json.loads(a.sections), seed=seed)
        name = f'rows-seed-{seed}'
        rec = dict(audio=str(a.audio), star=a.star, request=r['request'], sections=json.loads(a.sections), seed=seed,
                   K=r['K'], song_ms=r['song_ms'], segments=r['segments'].tolist(), head_ms=r['head_ms'].tolist(),
                   seconds=dict(prepared['song'].seconds, heads=r['seconds']))
        if r2 is not None:
            from ..r2.export import export_chart, minimal_header
            from ..r2.sampling import continue_chart
            t = time.perf_counter()
            actions, gap = continue_chart(r2, r['head_ms'], r['song_ms'], r['grid'], seed=954 + seed, track=(),
                                          ln_level='unknown')
            rec['seconds']['r2'] = time.perf_counter() - t
            audio_name = 'audio' + prepared['audio'].suffix.lower()
            header = minimal_header(r['segments'], f'Rows seed {seed}').replace(
                '[General]\n', f'[General]\nAudioFilename:{audio_name}\n')
            export_chart(r['head_ms'], r['song_ms'], actions, gap, out / f'{name}.osu', header)
            if not (out / audio_name).exists():
                shutil.copy2(prepared['audio'], out / audio_name)
        (out / f'{name}.json').write_text(json.dumps(rec))
        print(json.dumps(dict(seed=seed, K=r['K'], seconds=rec['seconds'])), flush=True)


if __name__ == '__main__':
    main()
