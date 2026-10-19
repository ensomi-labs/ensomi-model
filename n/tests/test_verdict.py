"""The rung-1 rule on synthetic score files: O1 when an arm clears BeatThis on two hard strata, O3 when not."""
from __future__ import annotations

import json

import numpy as np

from ensomi_n.run.verdict import verdict

STRATA = ['S0', 'S2', 'S3', 'S4', 'S5', 'S9']


def _rows(values: dict, rng, n=30):
    rows = []
    for s, v in values.items():
        for i in range(n):
            x = float(np.clip(v + rng.normal(0, 0.01), 0, 1))
            rows.append(dict(item=f'{s}-{i}', panel='panel', rec=f'r-{s}-{i}', strata=[s], m1=x, m1_bar=x, m3=x,
                             chance=0.08, chance_bar=0.02, weight=1.0,
                             m2=dict(tempo=x, octave=0.0, **{'3:2': 0.0}, other=1 - x)))
    return rows


def _write(tmp_path, arm_values: dict, bt_values: dict, gain=0.0):
    rng = np.random.default_rng(0)
    for arm, vals in arm_values.items():
        for seed in (0, 1):
            for k, ck in enumerate(('ckpt-00420', 'ckpt-00540', 'ckpt-00660')):
                (tmp_path / f'scores-{arm}-s{seed}-{ck}.json').write_text(json.dumps(dict(rows=_rows(vals, rng))))
            d = tmp_path / f'{arm}-s{seed}'
            d.mkdir(exist_ok=True)
            hist = [dict(elapsed_s=t, val_s0_m1=0.5 + gain * t / 660) for t in (120, 240, 360, 480, 600, 660)]
            (d / 'status.json').write_text(json.dumps(dict(history=hist)))
    ctrl = {f'bt-final{k}': _rows(bt_values, rng) for k in range(3)}
    (tmp_path / 'scores-controls.json').write_text(json.dumps(dict(rows=ctrl)))


def test_o1_when_two_hard_strata_clear(tmp_path):
    bt = dict(S0=0.9, S2=0.5, S3=0.4, S4=0.5, S5=0.6, S9=0.5)
    arm = dict(bt, S3=0.7, S4=0.8)
    _write(tmp_path, dict(A1=arm, A2={k: v - 0.05 for k, v in bt.items()}), bt)
    res = verdict(tmp_path, tmp_path)
    assert res['reading'] == 'O1'
    assert set(res['per_arm']['A1']['beyond']) == {'S3', 'S4'}
    assert res['better_arm'] == 'A1'
    assert res['a2_vs_a1']['result'] == 'F2 retired as a family; phase becomes an auxiliary loss'


def test_o3_and_o4_when_nothing_clears(tmp_path):
    bt = dict(S0=0.9, S2=0.5, S3=0.4, S4=0.5, S5=0.6, S9=0.5)
    arm = {k: v - 0.1 for k, v in bt.items()}
    _write(tmp_path, dict(A1=arm, A2=arm), bt, gain=0.0)
    assert verdict(tmp_path, tmp_path)['reading'] == 'O3'
    _write(tmp_path, dict(A1=arm, A2=arm), bt, gain=0.06)
    assert verdict(tmp_path, tmp_path)['reading'] == 'O4'
