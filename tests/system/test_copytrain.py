import hashlib
import json

import numpy as np
import pandas as pd
import torch

from ensomi_model.system import copytrain
from ensomi_model.system.arrange import arrange
from ensomi_model.system.copy import CopyConfig, CopyPath
from ensomi_model.r2.theta import decode

from .helpers import grid, skeleton, tiny_r2


def fake_inputs(tmp_path, n_charts=14):
    rng = np.random.default_rng(0)
    model = tiny_r2()
    (tmp_path / 'cache' / 'charts').mkdir(parents=True)
    (tmp_path / 'tf').mkdir()
    (tmp_path / 'rows' / 'features').mkdir(parents=True)
    index, rows, shas = [], [], []
    for i in range(n_charts):
        head, song = skeleton(rng, 60 + 7 * i)
        actions, gap, _ = arrange(model, head, song, grid(), seed=i, target=[0, 1, 0], kappa=0)
        sha = hashlib.sha256(str(i).encode()).hexdigest()
        shas.append(sha)
        np.savez(tmp_path / 'cache' / 'charts' / f'{i}.npz', head_ms=head, actions=actions.astype(np.int8),
                 gap_release_ms=gap, release_orig_ms=gap, grid_segments=np.array([[0.0, 500.0, 4]]),
                 grid_bars=np.array([[0.0, 4]]), song_ms=song)
        index.append(dict(sha256=sha, file=f'{i}.npz'))
        _, heads, _ = decode(actions[:len(head)])
        np.savez(tmp_path / 'tf' / f'{sha[:16]}.npz', lp=np.log(rng.dirichlet(np.ones(15), len(head))).astype(np.float32),
                 h=(heads.astype(int) @ (1 << np.arange(4))).astype(np.uint8), b=grid().beat(head), K=len(head))
        key = f'audio{i % 5}'
        np.save(tmp_path / 'rows' / 'features' / f'{key}.npy',
                rng.normal(size=(int(song / 10) + 50, 130)).astype(np.float16))
        rows.append(dict(sha=sha, key=key))
    pd.DataFrame(index).to_parquet(tmp_path / 'cache' / 'index.parquet')
    (tmp_path / 'rows' / 'index.json').write_text(json.dumps(dict(features=str(tmp_path / 'rows' / 'features'),
                                                                  rows=rows[:-1])))      # the last chart has no audio
    (tmp_path / 'panels.json').write_text(json.dumps(dict(train=shas[:8], V=shas[8:10], P=shas[10:12],
                                                          TR=shas[12:])))
    return tmp_path


def test_prepare_train_and_eval_run(tmp_path):
    root = fake_inputs(tmp_path)
    common = ['--tf', str(root / 'tf'), '--panels', str(root / 'panels.json'), '--cache', str(root / 'cache'),
              '--out', str(root / 'out'), '--report', str(root / 'report')]
    copytrain.main(['prepare', '--rows-data', str(root / 'rows')] + common)
    summary = json.loads((root / 'report' / 'audio-summary.json').read_text())
    assert summary['written'] == 13 and summary['missing'] == {'TR': 1}
    torch.manual_seed(0)
    copytrain.main(['train', 'arm', '--steps', '20', '--every', '10', '--pilot', '0', '--charts', '4',
                    '--queries', '16'] + common)
    log = [json.loads(x) for x in (root / 'report' / 'train-arm.jsonl').read_text().splitlines()]
    assert log[0]['train_charts'] == 8 and log[-1]['event'] == 'end' and log[-1]['status'] == 'finished'
    assert (root / 'out' / 'models' / 'arm-best.pt').exists() and (root / 'out' / 'models' / 'arm-10.pt').exists()
    old = CopyPath(CopyConfig.old())
    torch.save(dict(state=old.state_dict(), args=dict(arch='copy', variant='full', heads=8, hidden=32), step=1),
               root / 'old.pt')
    copytrain.main(['eval', f"arm={root / 'out' / 'models' / 'arm-best.pt'},old={root / 'old.pt'}", '--tag', 't']
                   + common)
    table = pd.read_csv(root / 'report' / 'copyeval-t.csv')
    assert {'path-R2', 'nocopy-R2', 'audio_shuffled-path', 'path-path[arm]'} <= set(table.comparison)
    per = pd.read_csv(root / 'report' / 'copyeval-per-chart-t.csv')
    nocopy = per[(per.variant == 'nocopy') & (per.model == 'old')].set_index('sha256').total
    r2 = per[(per.variant == 'R2') & (per.model == 'old')].set_index('sha256').total
    assert np.allclose(nocopy, r2)        # untrained betas are 1: the path without copy terms is R2
    att = pd.read_csv(root / 'report' / 'copy-attention-P-t.csv')
    assert len(att) == 4 and att.null.between(0, 1).all()
