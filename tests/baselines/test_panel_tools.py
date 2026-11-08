"""Absolute replay fixtures and a real child-process deadline for the panel tools."""
import datetime
import gzip
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ensomi_model.research.oracle_time_continuation.replay import ExactReplayState, commit
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
from ensomi_model.research.chart.actions import EMPTY, TAP, LN_START, LN_CLOSE


def module(name):
    path = Path(__file__).resolve().parents[2] / 'scripts/baselines' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def code():
    return SimpleNamespace(ExactReplayState=ExactReplayState, commit=commit, CompleteRow=CompleteRow,
                           EMPTY=EMPTY, TAP=TAP, LN_START=LN_START, LN_CLOSE=LN_CLOSE)


def test_replay_preserves_cross_lane_holds_and_rejects_same_lane_reattack():
    analyze = module('panel_analyze')
    events = [(100, 0, LN_START), (200, 1, LN_START), (300, 0, LN_CLOSE), (400, 1, LN_CLOSE)]
    assert analyze.exact_events(events, code())['complete']
    assert not analyze.exact_events(events + [(250, 0, TAP)], code())['complete']
    assert not analyze.exact_events(events + [(300, 0, TAP)], code())['complete']


def test_raw_events_expose_dropped_untimed_and_unmatched_events(tmp_path):
    analyze = module('panel_analyze')
    groups = [dict(type='hold_note', time=100, x=64, timestamp_present=True),
              dict(type='hold_note_end', time=200, x=64, timestamp_present=True),
              dict(type='hold_note_end', time=300, x=192, timestamp_present=True),
              dict(type='hold_note', time=0, x=320, timestamp_present=False)]
    path = tmp_path / 'events.json.gz'
    with gzip.open(path, 'wt') as f:
        json.dump(dict(groups=groups), f)
    result = analyze.raw_mapper(path, code())
    assert not result['grammar']
    assert result['errors'] == dict(unmatched_end=1, missing_timestamp=1)


def test_deadline_stops_only_owned_child(tmp_path, monkeypatch):
    queue = module('panel_run')
    monkeypatch.setattr(queue.shutil, 'disk_usage', lambda _: SimpleNamespace(free=101_000_000_000))
    (tmp_path / 'results').mkdir()
    marker = tmp_path / 'child.pid'
    sentinel = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    try:
        job = dict(id='fixture', system='r2', song='fixture', arm='shipped', seed=954, threads=1,
                   source_class='normal', audio_seconds=1, receipt=str(tmp_path/'receipt.json'),
                   log=str(tmp_path/'child.log'), argv=[sys.executable, '-c',
                   'import os,pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)',
                   str(marker)])
        plan = dict(jobs=[job], cwd=str(tmp_path), retained_root=str(tmp_path),
                    deadline_utc=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=2)).isoformat())
        queue.run(plan, tmp_path)
        assert sentinel.poll() is None
        assert marker.exists()
        with pytest.raises(ProcessLookupError):
            os.kill(int(marker.read_text()), 0)
        records = [json.loads(s) for s in (tmp_path/'results/jobs.jsonl').read_text().splitlines()]
        assert records[-1]['status'] == 'interrupted'
        assert any(r.get('reason') == 'deadline; reserve report time' for r in records)
    finally:
        sentinel.terminate()
        sentinel.wait()
