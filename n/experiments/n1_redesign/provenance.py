"""Hash only the explicit open inputs consumed by the decoder measurements."""
from datetime import datetime,timezone
import platform
import subprocess
from base import *


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(1024*1024):digest.update(block)
    return digest.hexdigest()


def main():
    stage=ROOT/'artifacts/n1-stage0-20261028'
    old=json.loads((stage/'inputs.json').read_text())
    matches={p:dict(expected=d,actual=sha(p)) for p,d in old['files'].items()}
    for p,r in matches.items():
        if r['expected']!=r['actual']:raise RuntimeError('Stage 0 input identity changed: '+p)
    xs=items()
    paths=[PREV/'references.pkl',PREV/'e1-summary.json',stage/'events-cohort.json',stage/'inputs.json',
           stage/'baseline-null.json',stage/'decoder-recordings.csv',
           ROOT/'artifacts/bulk/mapper-spread-20261021/run/pairs.jsonl']
    for seed in ('s0','s1'):
        for item in sorted({x['item'] for x in xs}):
            paths += [PREV/'evidence'/seed/'009001'/(item+'.npz'),PREV/'e2-maps'/seed/'009001'/(item+'.json')]
    value=dict(stage0_identity_checks=matches,files={str(p):sha(p) for p in paths},
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        code_hashes={p.name:sha(p) for p in HERE.glob('*.py')},python=sys.version,
        numpy=np.__version__,platform=platform.platform(),utc=datetime.now(timezone.utc).isoformat())
    dump(OUT/'input-hashes.json',value)
    print('Hashed',len(paths),'explicit input files; stage 0 identities agree.',flush=True)


if __name__=='__main__':main()
