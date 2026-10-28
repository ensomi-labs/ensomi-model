"""Check copied D2 maps against X5d on deterministic open-panel inputs."""
import gzip
import json
import numpy as np
from scipy.special import softmax
from ensomi_n.v0.evaluate import v0_evidence
from .common import OUT,X5D,FEAT,dump
from .bands import decode_band,CFG
from .level import decode_level
from .. import base
from ..scoring import panel_items


def reproduce():
    rows=[]
    selected=sorted(panel_items(),key=lambda x:x['key'])[:5]
    for seed in (0,1):
        cache=FEAT/f'v01-s{seed}-009001'
        manifest={r['sha256']:r for r in map(json.loads,(cache/'manifest.jsonl').read_text().splitlines())}
        with np.load(X5D/f's{seed}/heads/linear.npz') as z:head={k:z[k] for k in z.files}
        for x in selected:
            ev=v0_evidence(base.evidence(f's{seed}',x['item']))
            with np.load(X5D.parent/'tracks'/f's{seed}'/f"{x['item']}.npz") as z:
                track=tuple(z[k] for k in ('frames','kinds','periods'))
            with gzip.open(X5D/'controls'/f's{seed}'/f"{x['rec']}.json.gz",'rt') as f:controls=json.load(f)
            with gzip.open(X5D/f's{seed}/maps'/f"{x['rec']}.json.gz",'rt') as f:old=json.load(f)['maps']
            with np.load(cache/manifest[x['item']]['npz']) as z:feat=z['feat'].astype(np.float32)
            logits=((feat-head['mean'])/head['sd'])@head['w0'].T+head['b0']
            probs=softmax(logits.astype(np.float64)/float(head['temperature']),axis=1)
            own,_=decode_level(ev,None,'own_d2',track,CFG)
            band,_=decode_band(ev,probs,track,CFG)
            comparisons=0
            for arm,m,expected in [('own_d2',own,controls),('band_d2',band,old)]:
                actual=m.to_json()
                for field in ('beats','segments','downbeats'):
                    if actual.get(field)!=expected[arm][x['key']].get(field):
                        raise RuntimeError(f'Copied D2 reproduction failed: seed {seed}, {arm}, {field}')
                    comparisons+=1
            rows.append(dict(seed=seed,audio_sha256=x['item'],key=x['key'],comparisons=comparisons))
    dump(OUT/'decoder-reproduction.json',dict(passed=True,rows=rows,comparisons=sum(r['comparisons'] for r in rows)))
    print(json.dumps(dict(event='decoder_reproduction',passed=True,comparisons=sum(r['comparisons'] for r in rows))))


if __name__=='__main__':reproduce()
