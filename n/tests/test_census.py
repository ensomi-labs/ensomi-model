"""The census, E2's rule, and the hard holdout's draw, seal and exclusion, on synthetic sets."""
from __future__ import annotations

import csv
import hashlib
import json

import numpy as np
import pytest

from ensomi_n.eval import census
from ensomi_n.eval.__main__ import main

B = 500.0
CONTROL = ([(0, B, 4)], [k * B for k in range(64)])
RESETS = ([(k * 1750, B, 4) for k in range(8)], [k * 1750 + j * 250 for k in range(8) for j in range(7)])


def make(tmp_path, osu_writer, spec):
    """spec: (key, split, recording, kind, creator, mode) -> a units table; audio bytes beside each chart."""
    rows = []
    for key, split, rec, kind, creator, mode in spec:
        red, heads = kind
        path = osu_writer(f'data/{key}/a.osu', red, heads=heads)
        text = path.read_text().replace('[Metadata]', f'[Metadata]\nCreator:{creator}')
        path.write_text(text)
        (path.parent / 'audio.mp3').write_bytes(key.encode())
        rows.append(dict(file=f'data/{key}/a.osu', key=key, source=split.split('-')[0], recording=rec, split=split,
                         status='ranked', mode=mode, set_id=key.split('/')[-1], tags='', tags_known='0'))
    units = tmp_path / 'units.csv'
    with units.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return units


SPEC = (
    [(f'd/{i}', 'fit_dev', f'r{i}', CONTROL, f'm{i}', 'mania4k') for i in range(3)]
    + [('d/9', 'fit_dev', 'r0', RESETS, 'other', 'mania4k')]  # a second mapper for r0, timing differs
    + [('d/8', 'fit_dev', 'r1', CONTROL, 'copier', 'mania4k')]  # a copy for r1
    + [(f'h/{i}', 'holdout2', f'h{i}', RESETS, f'm{i}', 'mania4k') for i in range(2)]
    + [(f's/{i}', 'std-taiko', f's{i}', RESETS if i % 2 else CONTROL, f'x{i}', 'taiko') for i in range(30)]
    + [('s/linked', 'std-taiko', 'r2', RESETS, 'x', 'std')]  # shares a recording with fit_dev: never drawn
)


def test_tabulate_counts_strata_pairs_and_criteria(tmp_path, osu_writer):
    units = census.read_units(make(tmp_path, osu_writer, SPEC))
    results = census.compute(units, tmp_path)
    assert all(r.strata is not None for r in results)
    t = census.tabulate(results)['splits']
    fd = t['fit_dev']
    assert fd['all']['recordings'] == 3 and fd['all']['sets'] == 5
    # r0 holds a control set and a reset set: S0 needs every set, so r0 leaves S0; r1 and r2 stay.
    assert fd['S0']['recordings'] == 2 and fd['S4']['recordings'] == 1 and fd['S2']['recordings'] == 1
    assert fd['all']['pairs_differing'] == 1 and fd['all']['pairs_copies'] == 1
    assert fd['S4']['pairs_differing'] == 1 and fd['S4']['criteria']['S4.reset'] == 1
    assert t['holdout2']['S4']['recordings'] == 2 and t['holdout2']['S4']['span_heads'] > 0
    assert t['std-taiko|taiko']['all']['recordings'] == 30
    status = census.e2_status(census.tabulate(results), floors=('fit_dev',))
    assert status['S4']['status'].startswith('characterises') and status['S4']['holdout_recordings'] == 2
    assert status['S10']['status'].startswith('not computed')


def test_draw_seal_and_exclude(tmp_path, osu_writer):
    units = census.read_units(make(tmp_path, osu_writer, SPEC))
    results = census.compute(units, tmp_path)
    a = census.draw(results, pool='std-taiko', per_stratum=4, keep_design=2, seed=7)
    b = census.draw(results, pool='std-taiko', per_stratum=4, keep_design=2, seed=7)
    assert a == b
    rule = a['rule']
    assert rule['pool_recordings']['S0'] == 15 and rule['pool_recordings']['S4'] == 15
    assert rule['held_recordings']['S0'] >= 4 and rule['held_recordings']['S4'] >= 4
    drawn = {m['recording'] for m in a['members']}
    assert 's2' not in drawn and drawn <= {f's{i}' for i in range(30)}
    out = tmp_path / 'hold' / 'hard-holdout-v1'
    sha = census.seal(a, out, tmp_path, commit='abc', name='hard-holdout-v1', date='2026-10-19')
    text = (out / census.MEMBERSHIP).read_bytes()
    assert hashlib.sha256(text).hexdigest() == sha and sha in (out / 'membership.sha256').read_text()
    member = json.loads(text)['members'][0]['sets'][0]
    assert member['audio_sha256'] == hashlib.sha256(member['key'].encode()).hexdigest()
    assert 'E7' in (out / 'README.md').read_text() and json.loads((out / 'rule.json').read_text())['commit'] == 'abc'
    with pytest.raises(ValueError, match='never redrawn'):
        census.seal(a, out, tmp_path)
    ex = census.load_exclusion(out, sha)
    assert ex['recordings'] == drawn and len(ex['set_ids']) == len(drawn)
    with pytest.raises(ValueError, match='does not match'):
        census.load_exclusion(out, '0' * 64)


def test_cli_census_keeps_sealed_splits_to_counts(tmp_path, osu_writer, capsys):
    units = make(tmp_path, osu_writer, SPEC)
    hold = tmp_path / 'hh'
    assert main(['draw', '--units', str(units), '--root', str(tmp_path), '--out', str(hold),
                 '--per-stratum', '4', '--keep-design', '2', '--seed', '3']) == 0
    sha = capsys.readouterr().out.split('membership sha256 ')[1].split()[0]
    out = tmp_path / 'census'
    assert main(['census', '--units', str(units), '--root', str(tmp_path), '--out', str(out),
                 '--sealed', 'holdout2', '--holdout', str(hold), '--holdout-sha256', sha]) == 0
    doc = json.loads((out / 'census.json').read_text())
    splits = doc['table']['splits']
    held = census.load_exclusion(hold, sha)
    assert splits['hard-holdout']['all']['sets'] == len(held['keys'])
    assert splits['std-taiko-design']['all']['sets'] == 31 - len(held['keys'])
    files = sorted(p.name for p in out.glob('strata-*.csv'))
    assert files == ['strata-fit_dev.csv', 'strata-std-taiko-design.csv']
    with (out / 'strata-std-taiko-design.csv').open() as f:
        design = {row['key'] for row in csv.DictReader(f)}
    assert design and not design & held['keys']
    assert '| S4 short sections and resets |' in (out / 'census.md').read_text()
    assert main(['census', '--units', str(units), '--out', str(out), '--holdout', str(hold)]) == 2
