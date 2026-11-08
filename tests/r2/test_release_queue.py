import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location('release_queue',
    Path(__file__).resolve().parents[2] / 'scripts/r2/release_queue.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def test_queue_runs_in_order_and_skips_completed_jobs(tmp_path, monkeypatch):
    a, b = tmp_path / 'a', tmp_path / 'b'
    manifest = dict(out=str(tmp_path), owned_roots=[str(tmp_path)], jobs=[
        dict(name='a', argv=[sys.executable, '-c', 'from pathlib import Path; Path("a").write_text("first")'],
             cwd=str(tmp_path), complete_file=str(a)),
        dict(name='b', argv=[sys.executable, '-c',
             'from pathlib import Path; assert Path("a").read_text()=="first"; Path("b").write_text("second")'],
             cwd=str(tmp_path), complete_file=str(b))])
    path = tmp_path / 'jobs.json'
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(queue, 'guard', lambda *args: None)
    monkeypatch.setattr(sys, 'argv', ['queue', '--manifest', str(path)])
    assert queue.main() == 0
    assert b.read_text() == 'second'
    assert not (tmp_path / 'queue.lock').exists()
    a.write_text('leave completed output alone')
    assert queue.main() == 0
    assert a.read_text() == 'leave completed output alone'
    (tmp_path / 'STOP').touch()
    with pytest.raises(SystemExit, match='STOP'):
        queue.main()


def test_disk_guard_leaves_room_for_atomic_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(queue.shutil, 'disk_usage', lambda _: SimpleNamespace(free=100_050_000_000))
    assert 'free disk' in queue.guard(tmp_path, [tmp_path])
    monkeypatch.setattr(queue.shutil, 'disk_usage', lambda _: SimpleNamespace(free=110_000_000_000))
    monkeypatch.setattr(queue, 'owned_bytes', lambda _: 5_950_000_000)
    assert 'owned outputs' in queue.guard(tmp_path, [tmp_path])


def test_watchdog_requests_clean_child_stop_and_does_not_start_next_job(tmp_path, monkeypatch):
    child = '''from pathlib import Path
import signal
import time
def stop(signum, frame):
    Path("recovered").write_text("clean stop")
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
Path("ready").touch()
while True:
    time.sleep(.1)
'''
    manifest = dict(out=str(tmp_path), owned_roots=[str(tmp_path)], jobs=[
        dict(name='first', argv=[sys.executable, '-c', child], cwd=str(tmp_path),
             complete_file=str(tmp_path / 'final')),
        dict(name='second', argv=[sys.executable, '-c', 'raise RuntimeError("must not start")'],
             cwd=str(tmp_path), complete_file=str(tmp_path / 'other'))])
    path = tmp_path / 'jobs.json'
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(queue, 'guard', lambda *args:
                        'synthetic low disk' if (tmp_path / 'ready').exists() else None)
    monkeypatch.setattr(sys, 'argv', ['queue', '--manifest', str(path)])
    assert queue.main() == 75
    assert (tmp_path / 'recovered').read_text() == 'clean stop'
    assert not (tmp_path / 'second.log').exists()
    assert not (tmp_path / 'queue.lock').exists()


def test_detach_uses_nohup_nice_caffeinate_and_new_session(tmp_path, monkeypatch):
    spec = dict(out=str(tmp_path), owned_roots=[str(tmp_path)])
    (tmp_path / 'STOP').touch()
    monkeypatch.setattr(queue, 'guard', lambda *args: None)
    monkeypatch.setattr(queue.shutil, 'which', lambda name: '/usr/bin/' + name)
    calls = []
    def spawn(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(pid=987654)
    monkeypatch.setattr(queue.subprocess, 'Popen', spawn)
    with pytest.raises(SystemExit, match='STOP'):
        queue.detach(tmp_path / 'jobs.json', spec)
    assert queue.detach(tmp_path / 'jobs.json', spec, resume=True) == 0
    argv, kwargs = calls[0]
    assert argv[:6] == ['/usr/bin/nohup', '/usr/bin/nice', '-n', '10', '/usr/bin/caffeinate', '-i']
    assert kwargs['start_new_session'] and kwargs['stdin'] == queue.subprocess.DEVNULL
    assert not (tmp_path / 'STOP').exists()
    (tmp_path / 'queue.lock').write_text(str(__import__('os').getpid()))
    with pytest.raises(SystemExit, match='still alive'):
        queue.detach(tmp_path / 'jobs.json', spec, resume=True)


def test_queue_resumes_only_progressing_rss_guard_stops(tmp_path, monkeypatch):
    child = '''from pathlib import Path
import json
state=Path("state.json")
if not state.exists():
    state.write_text(json.dumps(dict(reason="RSS growth limit since process start", recovery_exposures=20)))
    raise SystemExit(75)
Path("final").touch()
'''
    manifest = dict(out=str(tmp_path), owned_roots=[str(tmp_path)], jobs=[
        dict(name='train', argv=[sys.executable, '-c', child], cwd=str(tmp_path),
             complete_file=str(tmp_path / 'final'), state_file=str(tmp_path / 'state.json'), rss_restarts=2)])
    path = tmp_path / 'jobs.json'; path.write_text(json.dumps(manifest))
    monkeypatch.setattr(queue, 'guard', lambda *args: None)
    monkeypatch.setattr(sys, 'argv', ['queue', '--manifest', str(path)])
    assert queue.main() == 0
    assert (tmp_path / 'final').exists()
