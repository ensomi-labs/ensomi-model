"""Remove only the regenerable artifacts authorised for G2, recording their sizes."""
import json
import re
import shutil
from datetime import datetime,timezone
from .common import OUT,G1,dump


def remove(paths):
    log=OUT/'deletions.json'
    records=json.loads(log.read_text()) if log.exists() else []
    for path in paths:
        if not path.exists():continue
        if path.is_symlink():raise RuntimeError(f'Refusing to remove a symlink: {path}')
        files=[path] if path.is_file() else [p for p in path.rglob('*') if p.is_file()]
        if any(p.is_symlink() for p in files):raise RuntimeError('Unexpected symlink in regenerable cache')
        record=dict(path=str(path),files=len(files),bytes=sum(p.stat().st_size for p in files),
                    utc=datetime.now(timezone.utc).isoformat(),status='pending')
        records.append(record);dump(log,records)
        if path.is_file():path.unlink()
        else:shutil.rmtree(path)
        record['status']='deleted';dump(log,records)


def control(seed):
    if not (OUT/f'evaluation-s{seed}-DONE.json').exists():raise RuntimeError('Control not yet scored')
    folder=OUT/'control'/f's{seed}'
    data=folder/'data'
    if data.exists():
        for name in ('metadata.json','fit-inventory.json','cal-inventory.json'):
            shutil.copyfile(data/name,folder/name)
        remove([data])


def finish():
    if not (OUT/'report.md').exists():raise RuntimeError('Write the G2 report before inference cleanup')
    for seed in (0,1):
        if not (OUT/f'evaluation-s{seed}-DONE.json').exists():raise RuntimeError('Evaluation incomplete')
        control(seed)
    paths=[G1/'input-cache']
    for root,names in ((G1,('F0-s0','F1-s0','F0-s1','F1-s1')),(OUT,('F0-s0','F2-s0','F0-s1','F2-s1'))):
        for name in names:
            paths.extend(p for p in (root/'eval'/name).glob('*.npz') if re.fullmatch(r'[0-9a-f]{64}\.npz',p.name))
    remove(paths)
    dump(OUT/'cleanup-DONE.json',dict(status='complete'))


if __name__=='__main__':finish()
