"""Matched recording comparisons under the fixed G1 bootstrap margin rule."""
from collections import defaultdict
from functools import lru_cache
import json
import numpy as np
from .common import OUT
from ..scoring import interval

CELLS=(('panel','all'),('val_hard','all'),('pooled','S0'),('pooled','S5'))
METRICS=('n_level','n_bars_same','guarded_ready','raw_guarded_ready','n_bl20',
         'structural_ready','raw_structural_ready','raw_n_level','raw_n_bars_same','raw_n_bl20')


@lru_cache(maxsize=32)
def read(name,seed):
    folder='baseline' if name.startswith('v01-') or name=='F0-base' else 'scores'
    return json.loads((OUT/folder/f'{name}-s{seed}.json').read_text())['recordings']


def select(rows,panel,cell,metric):
    return {(r['panel'],r['rec']):r[metric] for r in rows if r['cell']==cell and (panel=='pooled' or r['panel']==panel)
            and np.isfinite(r.get(metric,np.nan))}


def paired(b0,b1,control,treatment):
    keys=sorted(set(b0)&set(b1)&set(control)&set(treatment))
    if not keys:return None
    a,b,c,d=[np.array([x[k] for k in keys]) for x in (b0,b1,control,treatment)]
    diff=interval(d-c);half=(diff['ci95'][1]-diff['ci95'][0])/2
    spread=abs(float(a.mean()-b.mean()))
    return dict(n=len(keys),control=interval(c),treatment=interval(d),delta=diff,
                margin=max(spread,half),seed_spread=spread,paired_half_width=half)


def noise():
    out={}
    for p,c in CELLS:
        row={}
        for m in METRICS:
            values={}
            for s,cks in [(0,('009001','008137','008400')),(1,('009001','007973','008401'))]:
                for ck in cks:values[f's{s}-{ck}']=interval(list(select(read(f'v01-{ck}',s),p,c,m).values()))
                values[f'F0-s{s}']=interval(list(select(read('F0-base',s),p,c,m).values()))
            a=select(read('v01-009001',0),p,c,m);b=select(read('v01-009001',1),p,c,m)
            keys=sorted(set(a)&set(b))
            row[m]=dict(values=values,paired_seed_null=interval([b[k]-a[k] for k in keys]),
                        checkpoint_range={str(s):max(values[f's{s}-{ck}']['value'] for ck in cks)-min(values[f's{s}-{ck}']['value'] for ck in cks)
                            for s,cks in [(0,('009001','008137','008400')),(1,('009001','007973','008401'))]})
        out[f'{p}/{c}']=row
    return out
