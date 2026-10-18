"""The command line end to end, on synthetic sets."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from ensomi_n.eval.__main__ import main
from ensomi_n.mapformat import MusicMap

N_ROOT = Path(__file__).resolve().parents[1]
B = 500.0
HEADS = [k * B for k in range(32)]


def make_inputs(tmp_path, osu_writer):
    osu_writer('refs/0/111/easy.osu', [(0, B, 4)], heads=HEADS[::2], version='Easy')
    osu_writer('refs/0/111/hard.osu', [(0, B, 4)], heads=HEADS, version='Hard')
    osu_writer('refs/0/222/hard.osu', [(0, B, 4)], heads=HEADS)
    osu_writer('refs/0/333/hard.osu', [(0, B, 4)], heads=HEADS)
    preds = tmp_path / 'preds'
    (preds / '0').mkdir(parents=True)
    MusicMap.from_beats(np.arange(32) * B, downbeats=np.arange(0, 32, 4), model='fixture').dump(preds / '0/111.json')
    osu_writer('preds/0/222/r2.osu', [(B / 2, B, 4)], heads=[0])
    osu_writer('preds/0/999.osu', [(0, B, 4)])
    (tmp_path / 'strata.csv').write_text('key,recording,stratum,source,start_ms,end_ms\n'
                                         '0/111,songA,S2,ranked,,\n'
                                         '0/222,songA,S2,ranked,,\n'
                                         '0/333,songB,S4,loved,0,4000\n', encoding='utf-8')
    return preds


def test_score_command(tmp_path, osu_writer, capsys):
    preds = make_inputs(tmp_path, osu_writer)
    out = tmp_path / 'report.json'
    code = main(['score', '--refs', str(tmp_path / 'refs'), '--preds', str(preds),
                 '--strata', str(tmp_path / 'strata.csv'), '--out', str(out), '--bootstrap', '100'])
    assert code == 0
    report = json.loads(out.read_text())
    inputs = report['inputs']
    assert inputs['sets'] == 3 and inputs['scored'] == 3
    assert inputs['missing_preds'] == ['0/333']
    assert [Path(p).name for p in inputs['unused_preds']] == ['999.osu']
    sets = {s['key']: s for s in report['sets']}
    assert sets['0/111']['bl20'] == 1.0 and sets['0/222']['bl20'] == 0.0 and sets['0/333']['bl20'] == 0.0
    assert sets['0/111']['heads'] == 32.0
    strata = report['strata']
    # songA's two sets average to 0.5; songB scores 0 as an empty prediction.
    assert strata['all']['n_recordings'] == 2 and strata['all']['bl20']['value'] == 0.25
    assert strata['S2']['bl20']['value'] == 0.5 and strata['S2|ranked']['n_recordings'] == 1
    assert strata['S4']['bl20']['value'] == 0.0
    assert report['nulls'][4]['bpm'] == 120.0 and report['nulls'][4]['bl20'] == 0.08
    assert report['roles']['bl20'].startswith('decides')
    assert 'BL-20 0.25' in capsys.readouterr().out


def test_skip_missing_and_bad_input(tmp_path, osu_writer):
    preds = make_inputs(tmp_path, osu_writer)
    out = tmp_path / 'r.json'
    assert main(['score', '--refs', str(tmp_path / 'refs'), '--preds', str(preds), '--out', str(out),
                 '--bootstrap', '0', '--skip-missing']) == 0
    assert json.loads(out.read_text())['inputs']['scored'] == 2
    assert main(['score', '--refs', str(tmp_path / 'nothing'), '--preds', str(preds), '--out', str(out)]) == 2


def test_module_entry_point_runs(tmp_path, osu_writer):
    preds = make_inputs(tmp_path, osu_writer)
    out = tmp_path / 'm.json'
    result = subprocess.run([sys.executable, '-m', 'ensomi_n.eval', 'score', '--refs', str(tmp_path / 'refs'),
                             '--preds', str(preds), '--out', str(out), '--bootstrap', '0'],
                            cwd=tmp_path, capture_output=True, text=True,
                            env={'PYTHONPATH': str(N_ROOT / 'src'), 'PATH': ''})
    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text())['strata']['all']['n_recordings'] == 3


def test_nulls_command(tmp_path):
    out = tmp_path / 'nulls.json'
    assert main(['nulls', '--phases', '50', '--out', str(out)]) == 0
    data = json.loads(out.read_text())
    assert data['pass'] is True and len(data['checks']) == 5
    # Seeded and exact: any change to how the scorer computes BL shows here.
    checks = {c['bpm']: c for c in data['checks']}
    assert checks[80.0]['whole_shift'] == {'bl20': 0.04, 'bl10': 0.04, 'bl5': 0.0}
    assert checks[159.0]['whole_shift'] == {'bl20': 0.12, 'bl10': 0.04, 'bl5': 0.04}
    assert checks[80.0]['beat_phase'] == {'bl20': 0.05245, 'bl10': 0.02665, 'bl5': 0.0122}
    assert checks[120.0]['beat_phase'] == {'bl20': 0.07945, 'bl10': 0.0398, 'bl5': 0.01845}
    assert checks[159.0]['beat_phase'] == {'bl20': 0.1081, 'bl10': 0.0522, 'bl5': 0.02625}
