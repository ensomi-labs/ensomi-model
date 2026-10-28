import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from n1_redesign.probes import labels, run


def test_dispatch_records_source_without_reading_inputs(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, 'argv', ['probes', 'labels', '--seed', '1'])
    monkeypatch.setattr(labels, 'prepare', lambda seed: calls.append(seed))
    events = []
    monkeypatch.setattr(run, 'log', lambda event, **kw: events.append((event, kw)))
    run.main()
    assert calls == [1]
    assert events[0][0] == 'stage_start'
    assert 'run.py' in events[0][1]['source_sha256']
    assert events[-1][0] == 'stage_done'
