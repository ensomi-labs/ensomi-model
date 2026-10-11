"""The R2 launcher's restart budget: progress is a new regular checkpoint, whatever latest.json points at.
Engineering tests with a fake trainer; no cache or model needed."""
import json

from ensomi_model.r2 import launch


def write_checkpoint(run, name):
    """As the trainer does: the file, then latest.json pointing at it (a safe one included)."""
    (run / 'checkpoints' / name).touch()
    (run / 'checkpoints' / 'latest.json').write_text(json.dumps(dict(path=name)))


def fake_trainer(monkeypatch, trainer):
    monkeypatch.setattr(launch.subprocess, 'call', trainer)
    monkeypatch.setattr(launch.time, 'sleep', lambda seconds: None)


def test_latest_checkpoint_ignores_safe_and_tagged_checkpoints(tmp_path):
    (tmp_path / 'checkpoints').mkdir()
    assert launch.latest_checkpoint(tmp_path) is None
    for name in ('ckpt-0000000000.pt', 'ckpt-0004000995.pt', 'ckpt-0004000995-timing.pt',
                 'ckpt-0009046664-safe.pt'):
        write_checkpoint(tmp_path, name)
    assert launch.latest_checkpoint(tmp_path) == 'ckpt-0004000995.pt'


def test_resource_stops_after_new_checkpoints_do_not_exhaust_the_restart_budget(tmp_path, monkeypatch):
    """Every trainer process writes a regular checkpoint, then stops at a resource limit with a safe one;
    each resume made progress, so the budget clears and the run reaches the end."""
    (tmp_path / 'checkpoints').mkdir()
    stops = launch.MAX_RESTARTS + 2
    calls = []

    def trainer(cmd, **kw):
        calls.append(cmd)
        i = len(calls)
        write_checkpoint(tmp_path, f'ckpt-{10 * i:010d}.pt')
        if i > stops:
            return 0
        write_checkpoint(tmp_path, f'ckpt-{10 * i + 5:010d}-safe.pt')
        return 4

    fake_trainer(monkeypatch, trainer)
    assert launch.supervise(tmp_path, tmp_path / 'code', resume=False) == 0
    assert len(calls) == stops + 1


def test_a_crash_loop_without_new_checkpoints_still_stops(tmp_path, monkeypatch):
    (tmp_path / 'checkpoints').mkdir()
    write_checkpoint(tmp_path, 'ckpt-0000000000.pt')
    calls = []

    def trainer(cmd, **kw):
        calls.append(cmd)
        write_checkpoint(tmp_path, f'ckpt-{len(calls):010d}-safe.pt')
        return 4

    fake_trainer(monkeypatch, trainer)
    assert launch.supervise(tmp_path, tmp_path / 'code', resume=False) == 4
    assert len(calls) == launch.MAX_RESTARTS + 1
