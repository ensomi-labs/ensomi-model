"""Serial MPS jobs with a single CPU-only Mel fitting worker."""
import datetime
import json
import os
import subprocess
import sys
import threading
from .common import OUT, DEADLINE, check_prereg, disk_guard, dump, log


def command(module, args, label):
    dest = OUT/'logs'
    dest.mkdir(exist_ok=True)
    log('start',label=label)
    with (dest/f'{label}.log').open('w') as f:
        result = subprocess.run([sys.executable,'-m',f'n.experiments.n1_redesign.gpu.g3.{module}',*args],stdout=f,stderr=subprocess.STDOUT)
    log('exit',label=label,returncode=result.returncode)
    if result.returncode:
        raise RuntimeError(f'{label} failed; see {dest / (label + ".log")}')


def main():
    check_prereg()
    failures = []
    skipped = []
    def mel_worker():
        for seed in (0,1):
            try:
                command('mel',['--seed',str(seed)],f'mel-s{seed}')
            except Exception as exc:
                failures.append(dict(task=f'mel-s{seed}',error=str(exc)))
    worker = threading.Thread(target=mel_worker)
    worker.start()
    try:
        for seed in (0,1):
            if datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(minutes=60)>DEADLINE:
                skipped.append(dict(seed=seed,reason='deadline'))
                log('skipped',seed=seed,reason='deadline')
                continue
            disk_guard(250_000_000)
            command('train',['--seed',str(seed)],f'train-s{seed}')
            command('infer',['infer','--seed',str(seed)],f'infer-s{seed}')
            command('infer',['score','--seed',str(seed)],f'score-s{seed}')
    except Exception as exc:
        failures.append(dict(task='MPS_queue',error=str(exc)))
        log('queue_failed',error=str(exc))
    finally:
        worker.join()
        dump(OUT/'queue-DONE.json',dict(status='complete' if not failures and not skipped else 'incomplete',failures=failures,skipped=skipped))
        log('queue_finished',failures=failures,skipped=skipped)


if __name__=='__main__':
    main()
