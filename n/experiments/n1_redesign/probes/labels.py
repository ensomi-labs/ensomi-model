"""Chart-derived labels on N's literal notated clock."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import pickle

import numpy as np

from . import _census as sc
from ._names import part_of
from .common import (ROOT, OUT, PREV, CACHE, assert_open, digest, dump, exclusion,
                     labels, log, manifests, nearest, split, stable_seed, unit_path)


def beat_labels(beats, lengths, heads, divisors):
    # A chart-rounded head up to 2 ms before a line belongs to that line.
    bi = np.clip(np.searchsorted(beats, heads + 2, side='right') - 1, 0, len(beats) - 1)
    count = np.bincount(bi, minlength=len(beats))
    tern = np.zeros(len(beats), bool)
    off = np.zeros(len(beats), bool)
    bad = np.zeros(len(beats), bool)
    np.logical_or.at(tern, bi, np.isin(divisors, (3, 6, 12)))
    np.logical_or.at(off, bi, np.isin(divisors, (2, 4, 8, 16)))
    np.logical_or.at(bad, bi, divisors <= 0)
    y = np.full(len(beats), -1, np.int8)
    y[(count >= 2) & ~bad & off & ~tern] = 0
    y[(count >= 2) & ~bad & tern] = 1
    return y, count, bi


def make_labels(paths, ranks):
    from ensomi_n.labels import read_set
    ref = read_set(paths)
    charts = [sc.parse_osu(Path(p).read_bytes()) for p in paths]
    grid_chart = charts[paths.index(ref.meta['timing_chart'])]
    g = sc.Grid(grid_chart['reds'], ref.hi_ms)
    t = ref.timing
    # Keep the beat that contains the first head and the song's final beat.
    start = max(0, np.searchsorted(t.beats, ref.lo_ms + 2, 'right') - 1)
    keep = (np.arange(len(t.beats)) >= start) & (t.beats <= ref.hi_ms + 2) & (t.beats >= 0)
    keep &= ref.scorable(t.beats)
    beats, length = t.beats[keep], t.step_ms[keep]
    if not len(beats):
        raise ValueError('No scorable notated beats')
    heads = ref.heads[(ref.heads >= beats[0] - 2) & ref.scorable(ref.heads)]
    loc = g.locate(heads)
    y, counts, bi = beat_labels(beats, length, heads, loc['cls'])
    bl = g.locate(beats)
    bars = bl['gbar']
    ub = np.unique(bars)
    # Difficulty ranks come from the census; density is a fallback only when
    # the census lacks this allowed reference chart.
    rank = [ranks.get((str(Path(p).parent.relative_to(ROOT)), c['version']),
                      (float('-inf'), len(c['objs']))) for p, c in zip(paths, charts)]
    order = sorted(range(len(charts)), key=lambda i: rank[i])
    hard = charts[order[-1]]
    ht, chord = np.unique([o['t'] for o in hard['objs']], return_counts=True)
    hloc = g.locate(ht)
    patterns = np.zeros((len(ub), 16 * 48), bool)
    nontrivial = np.zeros(len(ub), bool)
    meters = np.zeros(len(ub), int)
    for j, b in enumerate(ub):
        meters[j] = bl['meter'][np.flatnonzero(bars == b)[0]]
        cells = np.unique(hloc['pos48'][hloc['gbar'] == b])
        cells = cells[(cells >= 0) & (cells < patterns.shape[1])]
        patterns[j, cells] = True
        nontrivial[j] = len(cells) >= 3 and not any(
            np.array_equal(cells, np.arange(0, meters[j] * 48, d)) for d in (48, 24, 12))
    # Ladders must be in one folder on one audio, and have at least 3 charts.
    hard_folder = Path(paths[order[-1]]).parent
    easier = [i for i in order[:-1] if Path(paths[i]).parent == hard_folder
              and charts[i]['audio'].casefold() == hard['audio'].casefold()]
    retention = np.full(len(ht), np.nan, np.float32)
    if len(easier) >= 2:
        kept = []
        for i in easier:
            c = charts[i]
            side = sc.analyse(c, 'std' if c['mode'] == 0 else 'mania4k')['side']
            kept.append(sc.near(ht, side['ev_t']) <= 2)
        retention = np.mean(kept, axis=0).astype('float32')
    ev_t, ev_hs, ev_part, ev_chart, ev_eligible, ev_stratum = [], [], [], [], [], []
    for j, c in enumerate(charts):
        if not c['reds'] or not c['objs']:
            continue
        a = sc.analyse(c, {0: 'std', 1: 'taiko', 3: 'mania4k'}.get(c['mode'], 'mania4k'))
        r, side = a['rec'], a['side']
        addition = r['ev_hs'] / max(r['n_ev'], 1)
        backbeat = r['clap_bb'] / max(r['clap_beat4'], 1)
        hs_ok = addition >= .25 and .30 <= backbeat <= .95
        # Census ev_t is unique while ev_bits preserves individual edge events;
        # reconstruct their times from the event locator, which stays aligned.
        ev = side['ev']
        times = side['grid'].T[ev['seg']] + ev['x'] * ev['L']
        ev_t.extend(times)
        ev_hs.extend(side['ev_bits'])
        ev_part.extend([-1] * len(times))
        ev_chart.extend([j] * len(times))
        ev_eligible.extend([hs_ok] * len(times))
        ev_stratum.extend([-1] * len(times))
        named = [o for o in c['objs'] if o['fname']]
        if c['mode'] == 3 and len(named) >= max(10, len(c['objs']) * .5):
            nf = len({o['fname'].casefold() for o in named})
            st = 0 if nf <= 16 else 1 if nf >= 48 else 2
            for o in c['objs']:
                ev_t.append(o['t']); ev_hs.append(-1)
                part = part_of(o['fname'])
                ev_part.append({'kick': 0, 'snare': 1, 'hat': 2}.get(part, 3))
                ev_chart.append(j); ev_eligible.append(False); ev_stratum.append(st)
    kiai = np.array(sc.kiai_sections(grid_chart['tps'], ref.hi_ms), float).reshape(-1, 2)
    return dict(beats=beats, length=length, down=t.downbeat[keep], bars=bars,
                bib=bl['bib'], meter=bl['meter'], y=y, count=counts, heads=heads,
                divisor=loc['cls'], head_beat=bi, bar_ids=ub, patterns=patterns,
                nontrivial=nontrivial, bar_meter=meters, kiai=kiai,
                hard_heads=ht, retention=retention, chord=chord,
                hard_pos=hloc['pos'], hard_divisor=hloc['cls'],
                event_t=np.array(ev_t), event_bits=np.array(ev_hs),
                event_part=np.array(ev_part), event_chart=np.array(ev_chart),
                event_eligible=np.array(ev_eligible), event_stratum=np.array(ev_stratum),
                red=np.array(grid_chart['reds']),
                timing_consistent=np.array(ref.meta['timing_consistent']),
                mode=np.array(grid_chart['mode']))


def prepare(seed=0):
    rows = manifests(seed)
    e = exclusion()
    refs = pickle.loads((PREV / 'references.pkl').read_bytes())
    assert len(refs) == 472
    by_sha = defaultdict(list)
    for x in refs:
        sha = x['input']['audio_sha256']
        assert_open(dict(sha256=sha, rec=x['rec']), e)
        by_sha[sha].append(x)
    ranks = {}
    cp = ROOT / 'artifacts/n1-redesign-20261028/census/charts.csv'
    with cp.open() as f:
        for r in csv.DictReader(f):
            ranks[r['set_dir'], r['version']] = (float(r['star'] or '-inf'), int(r['n_on']))
    us, failed = [], []
    for sha, r in sorted(rows.items()):
        is_train = r['split'] in ('fit_train', 'train', 'probe-fit', 'probe-dev')
        cohort = split(sha) if is_train else r['split']
        if cohort is None:
            continue
        if sha in by_sha:
            specs = [dict(key=x['key'], rec=x['rec'], charts=x['charts'],
                          audio=x['input']['audio_rel'], cohort=x['panel']) for x in by_sha[sha]]
        else:
            specs = [dict(key=sha + ':' + str(i), rec=m.get('rec', sha), charts=m['charts'],
                          audio=m['audio_rel'], cohort=(m['ext'] + '_extension') if m.get('ext') else cohort)
                     for i, m in enumerate(r['memberships'])]
        for s in specs:
            u = dict(s, sha=sha, id=f'{stable_seed(s["key"]):08x}', cache_split=r['split'])
            try:
                path = unit_path(u)
                if not path.exists():
                    paths = [str(ROOT / p) for p in s['charts']]
                    for p in paths:
                        if Path(p).parent != (ROOT / s['audio']).parent:
                            raise ValueError('Chart outside declared audio folder')
                    z = make_labels(paths, ranks)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, **z)
                us.append(u)
            except Exception as exc:
                failed.append(dict(key=s['key'], error=repr(exc)))
        if len(us) % 100 == 0:
            log('labels_progress', seed=seed, units=len(us))
    counts = defaultdict(Counter)
    for u in us:
        z = labels(u)
        c = counts[u['cohort']]
        c.update(units=1, beats=len(z['beats']), heads=len(z['heads']),
                 ternary=int(sum(z['y'] == 1)), binary_fine=int(sum(z['y'] == 0)),
                 laddered_heads=int(np.isfinite(z['retention']).sum()),
                 nontrivial_bars=int(z['nontrivial'].sum()),
                 timing_inconsistent=int(not z['timing_consistent']))
    for k, c in counts.items():
        c['recordings'] = len({u['rec'] for u in us if u['cohort'] == k})
        c['audio_inputs'] = len({u['sha'] for u in us if u['cohort'] == k})
    dump(OUT / f'counts-s{seed}.json', dict(counts))
    dump(OUT / f'labels-failures-s{seed}.json', failed)
    assert len({u['id'] for u in us}) == len(us), 'Reference identifier collision'
    dump(OUT / 'units.json', us)
    ip = OUT / 'inputs.json'
    inputs = json.loads(ip.read_text()) if ip.exists() else {}
    inputs[f'seed{seed}'] = dict(manifests={str(Path(d) / 'manifest.jsonl'):
        digest(Path(d) / 'manifest.jsonl') for d in sorted({r['cache_dir'] for r in rows.values()})},
        checkpoint_sha256=sorted({r['checkpoint_sha256'] for r in rows.values()}),
        cache_rows=len(rows), labels=len(us), failed=len(failed), clock='osu-lead-v1; t0=0; fps=100')
    for p in (PREV / 'references.pkl', cp, cp.with_name('names.json')):
        inputs[str(p)] = digest(p)
    from ensomi_n.features.mel import DEFAULT
    inputs['bulk_mel'] = dict(path=str(ROOT / 'artifacts/bulk/n1-cache/mel'), config_key=DEFAULT.key(),
        dtype='float16', bands=128, fps=100, t0_ms=0,
        alignment_check='Every consumed feature/logit/Mel frame count must match exactly')
    inputs['charts'] = {p: digest(ROOT / p) for p in sorted({p for u in us for p in u['charts']})}
    inputs[f'decoded_maps_seed{seed}'] = {u['sha']: digest(p) for u in us
        if u['cohort'] in ('panel', 'val_hard')
        for p in [PREV / 'e2-maps' / f's{seed}' / '009001' / (u['sha'] + '.json')] if p.exists()}
    cohort_path = ROOT / 'artifacts/n1-stage0-20261028/events-cohort.json'
    inputs[str(cohort_path)] = digest(cohort_path)
    dump(ip, inputs)
    log('labels_done', seed=seed, counts=dict(counts), failures=len(failed))
    return us
