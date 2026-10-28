"""X5d recording-level structural aggregation, copied from a00b9b6."""
from collections import defaultdict,Counter
from .structure import CLASSES

def structural_rows(rows,cell='all'):
    groups=defaultdict(list)
    for r in rows:
        if cell=='all':spans=r['spans']
        elif cell in r.get('strata',{}):
            masks=r['strata'][cell]
            spans=r['spans'] if masks is None else [s for s in r['spans'] if any(s['lo']<b and s['hi']>a for a,b in masks)]
        else:continue
        if spans:
            groups[(r['panel'],r['rec'],r['seed']+':'+r['arm'],r['aligned'])].extend(spans)
    return [dict(panel=p,rec=rec,arm=arm,aligned=aligned,cell=cell,
        ready=int(all(s['cls'] in CLASSES[:3] for s in ss)),counts=dict(Counter(s['cls'] for s in ss)),spans=len(ss))
        for (p,rec,arm,aligned),ss in sorted(groups.items())]
