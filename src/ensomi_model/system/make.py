"""One command: audio, a star request and the controls in; a playable osu!mania 4K ``.osz`` and a receipt out.

    python -m ensomi_model.system.make --config <system.json> --audio <file> --star 3.5 --out <x.osz> \
        [--sections '[{"start_ms": 60000, "end_ms": 90000, "star": 4.5}]'] [--ln tap|half|natural] \
        [--difficulty-q Q | --difficulty off] [--stream A_MS,B_MS,Q[,jumpstream|chordstream]] \
        [--jack A_MS,B_MS,Q[,chordjack]] [--copy on|off] [--copy-jacks S] [--seed N] [--name LABEL] \
        [--title TEXT] [--artist TEXT]

Stages:
1. The Listener's beat grid (``audio_rows.audio``, ``audio_rows.grid``).
2. Head rows from the rows model (``audio_rows.rowsgen.Rows``): the star maps to the song's log(W_H / s) request;
   ``--sections`` are the rows model's own section requests (a star or a request per span).
3. The arrangement's band, kappa 3 targets, natural LN level and base r from the corpus prior given the rows
   (``system.prior``): these replace what the R2 demo read from a source chart.
4. Frozen R2 56M arranges the rows (``system.arrange``) with the demo's four controls and, when on, the copy path:
   - LN amount: tap (no long notes), half (half the natural level), natural (the prior's held share; the demo
     called this "mapped", the source chart's own level);
   - difficulty: whole-song r at quantile q of human charts in the band, mapped to a heads-per-row multiplier m from
     the prior's base r for these rows; one correction pass when the first misses by more than the band's seed SD;
     default q per band from the config;
   - stream and jack sections (8-32 s) with the demo's presets, targets looked up on these rows' gaps.
5. Export with the replay and legality checks, the audio, and ``receipt.json`` in one ``.osz``. Every time in the
   ``.osu`` is a whole millisecond: heads are rounded before R2 arranges them, gap releases after (kept strictly
   between their neighbouring rows), timing-point offsets at export. Not every osu! client is known to parse
   fractional times. Title and artist default to a cached human chart of the same audio (``cache_index``, matched
   by the audio file's sha256), else the audio file's stem and "Unknown".

The config (``system.json``) names the checkpoints and calibrations (paths relative to the job's cwd):
``rows_model``, ``beatthis``, ``r2``, ``calibration`` (the demo's), ``prior``, ``copy`` (weights, ``default``
on/off, ``strength``, ``reason``), ``default_q`` per band and, optionally, ``cache_index`` (the R2 cache's
``index.parquet``, for song metadata). No statistic here is validated against the human.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import zipfile

import numpy as np

from ..r2.common import MIN_GAP_MS, band_of
from .controls import DIFFICULTY_M_EDGE, difficulty_multiplier, lookup_band, lookup_subtype_band, scope_gap, validate_requests

LN = {'tap': 0.0, 'half': 0.5, 'natural': 1.0}
NAMES = {'stream': ('jumpstream', 'chordstream'), 'jack': ('chordjack',)}
KAPPA = (3.0, 3.0, 3.0)
R2_SEED = 954                     # R2's seed is R2_SEED + --seed; the rows use --seed


def parse_scope(text, concept):
    """a_ms,b_ms,q[,subtype] -> request dict."""
    parts = text.split(',')
    if len(parts) not in (3, 4):
        raise ValueError(f'--{concept} wants a_ms,b_ms,q[,subtype]; got {text!r}')
    request = dict(concept=concept, a_ms=float(parts[0]), b_ms=float(parts[1]), q=float(parts[2]))
    if len(parts) == 4:
        if parts[3] not in NAMES[concept]:
            raise ValueError(f'--{concept} subtype must be one of {NAMES[concept]}')
        request['subtype'] = parts[3]
    return request


def resolve_requests(head_ms, song_ms, requests, band, calibration, scopes=None):
    """External q requests -> targets on these rows (the demo's ``resolve_requests``)."""
    validate_requests(requests, song_ms)
    if requests and scopes is None:
        import pandas as pd
        scopes = pd.read_parquet(calibration['scope_table'])
    resolved = []
    for r in requests:
        if 'q' not in r:
            raise ValueError('Each request must give q')
        gap = scope_gap(head_ms, r['a_ms'], r['b_ms'], song_ms)
        if gap is None:
            raise ValueError('Request scope must contain at least two head rows')
        if r.get('subtype'):
            values = lookup_subtype_band(scopes, r['subtype'], band, r['b_ms'] - r['a_ms'], gap, r['q'], calibration)
        else:
            values = lookup_band(scopes, r['concept'], band, r['b_ms'] - r['a_ms'], gap, r['q'])
        row = dict(r, **values, median_gap_ms=gap)
        if r['concept'] == 'stream':
            row['rows_carry_stream'] = gap <= calibration['stream_eligibility_gap_ms']
        else:
            floor = calibration['masks'][str(band)]['reattack_floor_ms']
            row['reattack_floor_ms'] = floor
            row['rows_allow_jack'] = bool(gap >= floor)
        resolved.append(row)
    return resolved


def clean_heads(head_ms, song_ms):
    """Heads rounded to whole milliseconds, sorted, at least MIN_GAP_MS apart (later ones dropped) and at least
    MIN_GAP_MS before the song end."""
    out = []
    for t in np.sort(np.round(np.asarray(head_ms, dtype=np.float64))):
        if t >= 0 and t <= song_ms - MIN_GAP_MS and (not out or t - out[-1] >= MIN_GAP_MS):
            out.append(t)
    return np.array(out), len(head_ms) - len(out)


def whole_ms_releases(head_ms, song_ms, gap):
    """Gap releases rounded to whole milliseconds and kept strictly between the decision's neighbouring rows:
    decision k releases in (head k-1, head k), the last decision before the song end. Heads are whole ms and at
    least MIN_GAP_MS apart, so the interval holds a whole ms."""
    gap = np.asarray(gap, dtype=np.float64)
    lo = np.concatenate(([-np.inf], head_ms))[:len(gap)] + 1
    hi = np.concatenate((head_ms, [np.ceil(song_ms)]))[:len(gap)] - 1
    return np.clip(np.round(gap), lo[:, None], hi[:, None])


def song_metadata(audio, index):
    """(title, artist) from a cached human chart of the same audio file (sha256), else the file's stem and
    "Unknown"."""
    if index and Path(index).exists():
        import pandas as pd
        from ..osu_core.metadata import parse_osu_metadata
        rows = pd.read_parquet(index, columns=['path', 'audio_sha256'])
        for path in rows.path[rows.audio_sha256 == file_sha(audio)]:
            if Path(path).exists():
                meta = parse_osu_metadata(path)
                if meta.title:
                    return meta.title, meta.artist or 'Unknown'
    return Path(audio).stem, 'Unknown'


def song_r(head_ms, song_ms, actions, gap):
    from ..r2.strain import StrainTrace
    return StrainTrace(head_ms, song_ms).extend(actions, gap).scope(0, song_ms).ratio


def difficulty_passes(run, head_ms, song_ms, cal, band, q, base_r):
    """Whole-song difficulty: m from the band map given the base r, clamped to [lo, DIFFICULTY_M_EDGE], at most
    one correction pass. ``run(m)`` arranges at multiplier m. Returns (actions, gap, trace, info)."""
    values = cal['difficulty'][str(band)]['values']
    target_r = float(np.quantile(values, q))
    fit = cal['difficulty_fit']['by_band'][str(band)]
    lo, hi = cal['difficulty_fit']['domain']
    hi = min(hi, DIFFICULTY_M_EDGE)
    m_raw = difficulty_multiplier(target_r, base_r, fit)
    m = float(np.clip(m_raw, lo, hi))
    passes = []
    for attempt in range(2):
        t0 = time.monotonic()
        a, g, tr = run(m)
        achieved = song_r(head_ms, song_ms, a, g)
        passes.append(dict(multiplier=m, achieved=achieved, fallback_count=tr['fallback_count'],
                           wall_seconds=time.monotonic() - t0))
        if attempt or abs(achieved - target_r) <= fit['seed_sd']:
            break
        corrected = float(np.clip(m * np.exp((np.log(target_r) - np.log(achieved)) / fit['beta']), lo, hi))
        if corrected == m:
            break
        m = corrected
    info = dict(q=q, target_r=target_r, band=np.quantile(values, [.05, .95]).tolist(), map_band=band, base_r=base_r,
                base_r_source='prior', fit=fit, m_requested=m_raw, domain=[lo, hi], clamped=bool(m_raw < lo or m_raw > hi),
                m_final=m, achieved_r=achieved, error=achieved - target_r,
                within_seed_sd=bool(abs(achieved - target_r) <= fit['seed_sd']), passes=passes,
                tested_band_q=[0.1, 0.75], above_tested_band=bool(q > 0.75))
    return a, g, tr, info


def file_sha(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            sha.update(block)
    return sha.hexdigest()


def write_export(head_ms, song_ms, seg, actions, gap, audio, workdir, osz, receipt, bookmarks=(), title='Ensomi',
                 artist='Unknown'):
    """Replay/legality-checked .osu and the audio in one .osz, plus the receipt. Timing-point offsets are written as
    whole milliseconds; heads and releases must already be whole."""
    from ..evaluation.redlines import read_red_lines
    from ..osu_core.hitobjects import parse_mania_hit_objects
    from ..r2.cache import decisions_from_objects
    from ..r2.export import export_chart, minimal_header
    from ..r2.state import replay_decisions
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    name = 'audio' + Path(audio).suffix.lower()
    shutil.copyfile(audio, workdir / name)
    seg = np.array(seg, dtype=np.float64)
    seg[:, 0] = np.round(seg[:, 0])
    header = minimal_header(seg, re.sub(r'[\r\n]', ' ', title))
    header = header.replace('Artist:Ensomi\n', 'Artist:' + re.sub(r'[\r\n]', ' ', artist) + '\n')
    header = header.replace('[General]\n', f'[General]\nAudioFilename:{name}\n')
    header = header.replace('Version:R2', f"Version:Ensomi {receipt['setting']} s{receipt['seed']}\n"
                            'BeatmapID:0\nBeatmapSetID:-1')
    if bookmarks:
        header = header.replace('[Difficulty]', '[Editor]\nBookmarks:' + ','.join(str(int(t)) for t in bookmarks)
                                + '\n[Difficulty]')
    header = header.replace('[HitObjects]\n', f"// rows={(receipt['rows']['model_sha256'] or '-')[:12]} "
                            f"r2={(receipt['r2']['sha256'] or '-')[:12]} seed={receipt['seed']}\n[HitObjects]\n")
    path = workdir / 'generated.osu'
    objects, _ = export_chart(head_ms, song_ms, actions, gap, path, header)
    parsed = parse_mania_hit_objects(path)
    key = lambda o: (o.start_time_ms, o.lane, o.end_time_ms)
    if sorted(parsed, key=key) != sorted(objects, key=key):
        raise RuntimeError('Export round trip changed objects')
    if not np.array_equal(np.unique([o.start_time_ms for o in parsed]), head_ms):
        raise RuntimeError('Export changed head times')
    dec = decisions_from_objects(parsed, read_red_lines(path), song_ms)
    state, _ = replay_decisions(dec.head_ms, dec.song_ms, dec.actions, dec.gap_release_ms)
    if not state.finalized or np.any(state.held):
        raise RuntimeError('Export did not close every hold')
    receipt['export'] = dict(legal=True, head_rows_equal=True, replay_finalized=True, objects=len(parsed),
                             bookmarks_ms=[int(t) for t in bookmarks])
    (workdir / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    osz = Path(osz)
    osz.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(osz, 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in (path, workdir / name, workdir / 'receipt.json'):
            archive.write(p, p.name)
    return osz


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    return x


class System:
    """The models held once (rows, Listener, R2, copy path) and the calibrations; ``make`` per request."""

    def __init__(self, config: dict, rows, r2, cal, prior, copy=None, hashes=None, scopes=None):
        self.config, self.rows, self.r2, self.cal, self.prior, self.copy = config, rows, r2, cal, prior, copy
        self.hashes = hashes or {}
        self._scopes = scopes

    @classmethod
    def load(cls, config: dict):
        from ..audio_rows.rowsgen import Rows
        from ..r2.bakeoff_panel import load_model
        from .prior import Prior
        copy = None
        hashes = {k: file_sha(config[k]) for k in ('rows_model', 'r2', 'prior')}
        if config.get('copy', {}).get('weights'):
            from .copytrain import load_model as load_copy
            copy = load_copy(config['copy']['weights']).to('cpu')
            hashes['copy'] = file_sha(config['copy']['weights'])
        return cls(config, Rows.load(config['rows_model'], config['beatthis']), load_model(config['r2']),
                   json.loads(Path(config['calibration']).read_text()), Prior.load(config['prior']), copy, hashes)

    @property
    def scopes(self):
        if self._scopes is None:
            import pandas as pd
            self._scopes = pd.read_parquet(self.cal['scope_table'])
        return self._scopes

    def copy_setting(self, copy=None, strength=None):
        """(on, strength, reason) from the request and the config's default."""
        c = self.config.get('copy', {})
        on = c.get('default') == 'on' if copy in (None, 'default') else copy == 'on'
        if on and self.copy is None:
            raise ValueError('Copy requested but the config names no copy weights')
        s = float(c.get('strength', 1.0) if strength is None else strength)
        if not 0 <= s <= 1:
            raise ValueError('--copy-jacks must be in [0, 1]')
        return on, s, c.get('reason')

    def make(self, audio, star, out, *, sections=(), ln='natural', difficulty_q='default', stream=(), jack=(),
             copy=None, copy_jacks=None, seed=0, name=None, title=None, artist=None, prepared=None):
        from .arrange import arrange
        from .copy import CopyBias
        from .prior import kappa_targets, rows_features
        if not 2 <= star < 6:
            raise ValueError('The star request must be in [2, 6)')
        if ln not in LN:
            raise ValueError(f'--ln must be one of {sorted(LN)}')
        t_all = time.monotonic()
        seconds = {}
        if prepared is None:
            t = time.monotonic()
            prepared = self.rows.prepare(audio)
            seconds['listener'] = time.monotonic() - t
        rows = self.rows.heads(prepared, star=star, sections=list(sections), seed=seed)
        seconds['rows'] = rows['seconds']
        song_ms, grid = rows['song_ms'], rows['grid']
        head_ms, dropped = clean_heads(rows['head_ms'], song_ms)
        if len(head_ms) < 2:
            raise ValueError('The rows model placed fewer than two heads')
        band = band_of(star)
        masks = self.cal['masks'][str(band)]
        features = rows_features(head_ms, song_ms, grid)
        pred = self.prior.predict(features, band)
        external = [parse_scope(t, 'stream') for t in stream] + [parse_scope(t, 'jack') for t in jack]
        requests = resolve_requests(head_ms, song_ms, external, band, self.cal, self.scopes if external else None)
        ln_fraction = LN[ln]
        on, strength, reason = self.copy_setting(copy, copy_jacks)
        mel = prepared['song'].features
        r2_seed = R2_SEED + seed
        bias_record = {}

        def run(m):
            cb = CopyBias(self.copy, head_ms, grid, mel, strength) if on else None
            a, g, tr = arrange(self.r2, head_ms, song_ms, grid, seed=r2_seed,
                               target=kappa_targets(pred, ln_fraction, m), kappa=KAPPA, ln_fraction=ln_fraction,
                               requests=requests, masks=masks, copy=cb)
            bias_record['copy_max_abs'] = float(tr['copy_abs'].max()) if on else None
            return a, g, tr

        q = self.config['default_q'][str(band)] if difficulty_q == 'default' else difficulty_q
        t = time.monotonic()
        if q is None or q == 'off':
            multiplier, difficulty = 1.0, None
            actions, gap, trace = run(1.0)
        else:
            q = float(q)
            if not 0 <= q <= 1:
                raise ValueError('Difficulty q must be in [0, 1]')
            if not self.cal['difficulty_fit']['accepted']:
                raise ValueError('Difficulty map failed its monotonicity gate')
            actions, gap, trace, difficulty = difficulty_passes(run, head_ms, song_ms, self.cal, band, q,
                                                                float(np.exp(pred['log_r'])))
            difficulty['default'] = difficulty_q == 'default'
            multiplier = difficulty['m_final']
        seconds['r2'] = time.monotonic() - t
        gap = whole_ms_releases(head_ms, song_ms, gap)
        from ..r2.theta import decode
        K = len(head_ms)
        held, heads, ln_heads = decode(actions[:K])
        targets = kappa_targets(pred, ln_fraction, multiplier)
        quad_source = float(pred['quad'] * K)
        quad = dict(generated=int(heads.all(1).sum()), source_expected=quad_source, source='prior quad share x rows',
                    head_rows=K, band_p95_share=self.cal['quad_p95'][str(band)])
        quad['violates'] = bool(quad['generated'] > 2 * quad_source and quad['generated'] / K > quad['band_p95_share'])
        bookmarks = [t for r in requests for t in (r['a_ms'], r['b_ms'])]
        setting = re.sub(r'[^A-Za-z0-9_-]', '-', name or Path(out).stem)
        sections_out = [dict(concept=r['concept'], subtype=r.get('subtype'), q=r['q'], a_ms=r['a_ms'], b_ms=r['b_ms'],
                             target=r['target'], band=r['band'], achieved=r['achieved'], in_band=r['in_band'],
                             band_status=r['band_status'], chord_target=r.get('chord_target'),
                             chord_achieved=r.get('chord_achieved'), chord_in_band=r.get('chord_in_band'),
                             bound_share=r['bound_share'], temporal_eligible=r.get('temporal_eligible'),
                             eligibility_reasons=r.get('eligibility_reasons'),
                             rows_carry_stream=r.get('rows_carry_stream'), rows_allow_jack=r.get('rows_allow_jack'),
                             target_error=r['target_error'], fallback_count=r['fallback_count'],
                             median_gap_ms=r['median_gap_ms']) for r in trace['requests']]
        c = self.config
        receipt = dict(
            audio=str(audio), star=star, band=band, seed=seed, r2_seed=r2_seed, setting=setting,
            config={k: c[k] for k in ('rows_model', 'beatthis', 'r2', 'calibration', 'prior') if k in c},
            rows=dict(model=c.get('rows_model'), model_sha256=self.hashes.get('rows_model'), request=rows['request'],
                      sections=list(sections), K=K, dropped_close_heads=dropped,
                      segments=np.asarray(rows['segments']).tolist()),
            r2=dict(checkpoint=c.get('r2'), sha256=self.hashes.get('r2'), kappa=list(KAPPA), concept_loop='new'),
            prior=dict(path=c.get('prior'), sha256=self.hashes.get('prior'), method=self.prior.method,
                       features=dict(zip(('log_rows_s', 'log_gap_ms', 'log_wh_s', 'log_bpm'), features.tolist())),
                       predicted=pred),
            copy=dict(on=on, strength=strength if on else None, weights=c.get('copy', {}).get('weights'),
                      sha256=self.hashes.get('copy'), default=c.get('copy', {}).get('default'),
                      reason=reason, max_abs_bias=bias_record.get('copy_max_abs')),
            controls=dict(
                ln=dict(mode=ln, fraction=ln_fraction, natural_held_share=pred['held'], target_held_share=float(targets[0]),
                        achieved_held_share=float(held.mean()), ln_heads=int(ln_heads.sum()),
                        target_human_quantile=float(np.mean(np.asarray(self.cal['ln_source_levels'][str(band)])
                                                            <= targets[0])),
                        renamed='the demo\'s "mapped" (the source chart\'s level) is "natural" (the prior\'s level)'),
                difficulty=difficulty, sections=sections_out),
            targets=targets.tolist(), achieved=dict(held_share=float(held.mean()), heads_per_row=float(heads.sum(1).mean()),
                                                     busy_share=float((held | heads).all(1).mean()),
                                                     r=song_r(head_ms, song_ms, actions, gap)),
            requests=trace['requests'],
            bookmarks=[dict(concept=r['concept'], subtype=r.get('subtype'), a_ms=r['a_ms'], b_ms=r['b_ms'])
                       for r in requests],
            fallback_count=trace['fallback_count'], fallbacks=trace['fallbacks'],
            mask_firings_total=dict(zip(('reattack', 'quad', 'jack'), trace['mask_firings'][:K].sum(0).tolist())),
            quad_guard=quad, seconds=dict(seconds, total=time.monotonic() - t_all),
            claims='Characterisation only; no statistic here is validated against the human.')
        if title is None or artist is None:
            meta = song_metadata(prepared['audio'], c.get('cache_index'))
            title, artist = title or meta[0], artist or meta[1]
        receipt['metadata'] = dict(title=title, artist=artist)
        receipt = _jsonable(receipt)
        work = Path(out).with_suffix('')
        write_export(head_ms, song_ms, rows['segments'], actions, gap, prepared['audio'], work, out, receipt,
                     bookmarks, title=title, artist=artist)
        np.savez_compressed(work / 'arrays.npz', actions=actions, gap_release_ms=gap, head_ms=head_ms,
                            song_ms=song_ms)
        return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True, help='system.json')
    p.add_argument('--audio', required=True)
    p.add_argument('--star', type=float, required=True, help='star request, 2 <= star < 6')
    p.add_argument('--sections', default='[]', help='rows sections: JSON list of {start_ms, end_ms, star or request}')
    p.add_argument('--ln', choices=sorted(LN), default='natural')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--difficulty-q', type=float, help='quantile of whole-song r among human charts of the band')
    g.add_argument('--difficulty', choices=('default', 'off'), default='default')
    p.add_argument('--stream', action='append', default=[], metavar='A_MS,B_MS,Q[,SUBTYPE]')
    p.add_argument('--jack', action='append', default=[], metavar='A_MS,B_MS,Q[,chordjack]')
    p.add_argument('--copy', choices=('on', 'off', 'default'), default='default')
    p.add_argument('--copy-jacks', type=float, help='copy jack dial in [0, 1]; default from the config')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', required=True, help='output .osz')
    p.add_argument('--name', help='short label for the difficulty name and work directory')
    p.add_argument('--title', help='song title in the .osu; default from a cached chart of the audio, else its stem')
    p.add_argument('--artist', help='song artist in the .osu; default from a cached chart of the audio, else "Unknown"')
    p.add_argument('--threads', type=int, default=1)
    a = p.parse_args(argv)
    import torch
    torch.set_num_threads(a.threads)
    system = System.load(json.loads(Path(a.config).read_text()))
    q = a.difficulty_q if a.difficulty_q is not None else a.difficulty
    receipt = system.make(a.audio, a.star, a.out, sections=json.loads(a.sections), ln=a.ln, difficulty_q=q,
                          stream=a.stream, jack=a.jack, copy=a.copy, copy_jacks=a.copy_jacks, seed=a.seed, name=a.name,
                          title=a.title, artist=a.artist)
    d = receipt['controls']['difficulty']
    print(json.dumps(dict(osz=a.out, K=receipt['rows']['K'], band=receipt['band'], copy=receipt['copy']['on'],
                          difficulty=None if d is None else {k: d[k] for k in ('q', 'target_r', 'achieved_r', 'm_final',
                                                                                'clamped')},
                          ln=receipt['controls']['ln']['achieved_held_share'], fallbacks=receipt['fallback_count'],
                          quad_violates=receipt['quad_guard']['violates'], seconds=receipt['seconds'])))


if __name__ == '__main__':
    main()
