"""Object parser copied from supervision-census b14d9f5; slider events use its timing rule."""
import math
import numpy as np
HEADER_SECTIONS = ('General', 'Metadata', 'Difficulty', 'Editor')

def _num(s, default=0.0):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default

def _int(s, default=0):
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return default

def parse_osu(data: bytes) -> dict:
    """The parts of a `.osu` file this census reads."""
    text = data.decode('utf-8-sig', errors='replace')
    sec = ''
    kv: dict[tuple[str, str], str] = {}
    tps, objs, breaks = [], [], []
    sb_samples = 0
    sb_files: set[str] = set()
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith('//'):
            continue
        if s[0] == '[' and s[-1] == ']':
            sec = s[1:-1].strip()
            continue
        if sec in HEADER_SECTIONS:
            k, sep, v = s.partition(':')
            if sep:
                kv[(sec, k.strip())] = v.strip()
        elif sec == 'TimingPoints':
            p = s.split(',')
            if len(p) < 2:
                continue
            t, L = _num(p[0], math.nan), _num(p[1], math.nan)
            if not (math.isfinite(t) and math.isfinite(L)):
                continue
            meter = _int(p[2], 4) if len(p) > 2 else 4
            red = (p[6].strip() == '1') if len(p) > 6 else L > 0
            tps.append(dict(t=t, L=L, meter=meter if meter > 0 else 4, sset=_int(p[3]) if len(p) > 3 else 0,
                            sidx=_int(p[4]) if len(p) > 4 else 0, vol=_int(p[5], 100) if len(p) > 5 else 100,
                            red=red, kiai=bool(_int(p[7]) & 1) if len(p) > 7 else False))
        elif sec == 'HitObjects':
            p = s.split(',')
            if len(p) < 5:
                continue
            t = _num(p[2], math.nan)
            if not math.isfinite(t):
                continue
            typ, hs = _int(p[3]), _int(p[4])
            o = dict(t=t, typ=typ, hs=hs, x=_num(p[0]), end=None, edges=None, slides=0, length=0.0)
            samp: list[str] = []
            if typ & 128:
                q = p[5].split(':') if len(p) > 5 else []
                o['end'] = _num(q[0], math.nan) if q else math.nan
                samp = q[1:]
            elif typ & 2:
                o['slides'] = max(1, _int(p[6], 1)) if len(p) > 6 else 1
                o['length'] = _num(p[7]) if len(p) > 7 else 0.0
                if len(p) > 8 and p[8]:
                    o['edges'] = [_int(e) for e in p[8].split('|')]
                samp = p[10].split(':') if len(p) > 10 else []
            elif typ & 8:
                o['end'] = _num(p[5], math.nan) if len(p) > 5 else math.nan
                samp = p[6].split(':') if len(p) > 6 else []
            else:
                samp = p[5].split(':') if len(p) > 5 else []
            o['nset'] = _int(samp[0]) if len(samp) > 0 else 0
            o['aset'] = _int(samp[1]) if len(samp) > 1 else 0
            o['idx'] = _int(samp[2]) if len(samp) > 2 else 0
            o['fname'] = samp[4].strip() if len(samp) > 4 else ''
            objs.append(o)
        elif sec == 'Events':
            if raw[:1] in (' ', '_'):
                continue
            p = s.split(',')
            if p[0].strip() in ('2', 'Break') and len(p) >= 3:
                a, b = _num(p[1], math.nan), _num(p[2], math.nan)
                if math.isfinite(a) and math.isfinite(b):
                    breaks.append((a, b))
            elif p[0].strip() in ('5', 'Sample'):
                sb_samples += 1
                if len(p) >= 4:
                    sb_files.add(p[3].strip().strip('"').replace('\\', '/').casefold())
    g = lambda sec_, k, d='': kv.get((sec_, k), d)
    bookmarks = [_num(x, math.nan) for x in g('Editor', 'Bookmarks').split(',') if x.strip()]
    tps.sort(key=lambda d: (d['t'], 0 if d['red'] else 1))
    reds: list[tuple[float, float, int]] = []
    for d in tps:
        if d['red'] and d['L'] > 0:
            if reds and abs(reds[-1][0] - d['t']) < 0.5:
                reds[-1] = (d['t'], d['L'], d['meter'])           # same time: the later line applies
            else:
                reds.append((d['t'], d['L'], d['meter']))
    objs.sort(key=lambda o: o['t'])
    return dict(mode=_int(g('General', 'Mode', '0')), audio=g('General', 'AudioFilename'),
                keys=_num(g('Difficulty', 'CircleSize', '0')), sm=_num(g('Difficulty', 'SliderMultiplier', '1.4'), 1.4) or 1.4,
                version=g('Metadata', 'Version'), creator=g('Metadata', 'Creator'),
                tps=tps, reds=reds, objs=objs, breaks=breaks, sb_samples=sb_samples, sb_files=sb_files,
                bookmarks=[b for b in bookmarks if math.isfinite(b)])


def onsets_and_events(p):
    """Distinct heads and, for standard, slider repeats/tails on the census clock."""
    heads = np.unique([o['t'] for o in p['objs']])
    events = list(heads)
    if p['mode'] == 0 and p['reds']:
        times = np.array([d['t'] for d in p['tps']])
        reds = np.array([r[0] for r in p['reds']])
        for o in p['objs']:
            if not (o['typ'] & 2) or o['length'] <= 0:
                continue
            k = int(np.searchsorted(times, o['t'] + .5, 'right')) - 1
            sv = 1.
            if 0 <= k < len(times) and not p['tps'][k]['red'] and p['tps'][k]['L'] < 0:
                sv = min(max(-100. / p['tps'][k]['L'], .1), 10.)
            ri = int(np.clip(np.searchsorted(reds, o['t'] + 2., 'right') - 1, 0, len(reds)-1))
            dur = o['length'] / (p['sm'] * 100. * sv) * p['reds'][ri][1]
            events.extend(o['t'] + i * dur for i in range(1, o['slides'] + 1))
    return heads, np.unique(events)
