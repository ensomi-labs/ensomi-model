"""Wait for the producer's DONE marker, then run the fixed seed-1 pipeline."""
from datetime import datetime, timezone, timedelta
import subprocess
import time

from common import *


def main():
    check_prereg()
    start = json.loads((OUT / 'start.json').read_text())
    deadline = datetime.fromisoformat(start['seed1_wait_deadline'])
    polls = [json.loads(line) for line in (OUT / 'job.log').read_text().splitlines()
             if json.loads(line).get('event')=='seed1_poll']
    next_poll = datetime.fromisoformat(polls[-1]['utc'])+timedelta(minutes=5) if polls else datetime.now(timezone.utc)
    state_path = OUT / 'seed1-pipeline.json'
    if state_path.exists():
        raise RuntimeError('Fresh seed-1 pipeline state required')
    state = dict(status='waiting', deadline_utc=deadline.isoformat())
    dump(state_path,state)
    while True:
        now = datetime.now(timezone.utc)
        delay = (min(next_poll,deadline)-now).total_seconds()
        if delay>0:
            time.sleep(delay)
        now = datetime.now(timezone.utc)
        marker = cache_root('s1') / 'DONE'
        ready = marker.exists()
        append(OUT / 'job.log',dict(event='seed1_poll',utc=now.isoformat(),done=ready))
        print(now.isoformat(),'seed1 DONE',ready,flush=True)
        if ready:
            state.update(status='ready', observed_utc=now.isoformat(), done_sha256=sha(marker),
                producer_done_utc=datetime.fromtimestamp(marker.stat().st_mtime,timezone.utc).isoformat())
            dump(state_path,state)
            break
        if now>=deadline:
            state.update(status='unavailable',finished_utc=now.isoformat())
            dump(state_path,state)
            deviations=json.loads((OUT/'deviations.json').read_text())
            deviations.append('Seed 1 DONE was absent at the three-hour deadline. Only seed 0 was evaluated; a both-seed conclusion is unavailable.')
            dump(OUT/'deviations.json',deviations)
            subprocess.run([sys.executable,str(HERE/'summarize.py'),'--seeds','s0'],check=True)
            return
        next_poll = now+timedelta(minutes=5)
    for phase in ('prepare','train','evaluate','summarize'):
        # Every phase uses the same frozen decisions; panels never select a model.
        command = [sys.executable,str(HERE/f'{phase}.py')]
        command += ['s1'] if phase in ('prepare','train','evaluate') else ['--seeds','s0','s1']
        if phase=='evaluate':
            command += ['--workers','2']
        state.update(status=phase,phase_started_utc=datetime.now(timezone.utc).isoformat())
        dump(state_path,state)
        with (OUT/f's1-{phase}.log').open('w') as log:
            result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            state.update(status='failed',failed_phase=phase,returncode=result.returncode)
            dump(state_path,state)
            raise RuntimeError(f'Seed 1 {phase} failed; inspect the preserved log')
    state.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat())
    dump(state_path,state)
    print('Both-seed pipeline complete',flush=True)


if __name__=='__main__':
    main()
