"""G1's frozen linear fit on five stacked, song-normalized log-Mel frames."""
import argparse
import gc
import json
import os
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ensomi_n.v0.labels import V0Labels, dense_v0
from ..targets import dense_new, load_target, NEW_KEYS
from ..evaluate import head_rows
from ensomi_n.features.cache import mel_path
from .common import OUT, G1, BULK, dump, probe_bucket, check_prereg, disk_guard


def stack_frames(mel, ix, mean, std):
    """640 columns ordered by lag -2..2, with endpoint replication at song edges."""
    indices = np.clip(np.asarray(ix)[:,None] + np.arange(-2,3), 0, len(mel)-1)
    return ((np.asarray(mel[indices], np.float32)-mean)/std).reshape(len(ix), -1)


def fit_and_score(seed):
    torch.set_num_threads(1)
    os.nice(10)
    check_prereg()
    disk_guard()
    started = time.monotonic()
    rows = [r for r in json.loads((G1/'cohorts.json').read_text())['training'] if probe_bucket(r['audio_sha256']) in (0,1)]
    index = json.loads((BULK/'song-index-v0.json').read_text())
    rng = np.random.default_rng(3000+seed)
    x = np.empty((len(rows)*512, 640), np.float32)
    ys = {k:[] for k in NEW_KEYS}
    offset = 0
    selection = []
    for i, row in enumerate(rows,1):
        sha = row['audio_sha256']
        mel = np.load(mel_path(BULK/'mel', sha), mmap_mode='r')
        st = index[sha]
        tg = dense_new(load_target(sha), 0, len(mel))
        lab = V0Labels.from_json(json.loads((BULK/'labels-v0b'/sha[:2]/f'{sha}.json').read_text()))
        tg['attack_w'] *= dense_v0(lab,0,len(mel))['weight']
        eligible = np.flatnonzero((tg['attack_w']>0)|(tg['snap_w']>0)|(tg['sal_w']>0))
        ix = rng.choice(eligible, min(len(eligible),512), replace=False)
        x[offset:offset+len(ix)] = stack_frames(mel,ix,st['mean'],st['std'])
        offset += len(ix)
        for k in ys:
            ys[k].append(tg[k][ix])
        selection.append(dict(audio_sha256=sha, frames=len(ix)))
        if i%200==0:
            print(json.dumps(dict(event='mel_fit_data',seed=seed,done=i,total=len(rows))),flush=True)
    x = torch.from_numpy(x[:offset])
    tg = {k:torch.from_numpy(np.concatenate(v)) for k,v in ys.items()}
    del ys, mel
    pw = json.loads((G1/'target-rates.json').read_text())['attack_pos_weight']
    head = nn.Linear(640,7)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    opt = torch.optim.LBFGS(head.parameters(), lr=1., max_iter=120, history_size=20,
                           line_search_fn='strong_wolfe', tolerance_grad=1e-6)
    calls = 0
    first = last = None
    fit_start = time.monotonic()
    def weighted(v,w):
        return (v*w).sum()/w.sum().clamp_min(1.)
    def closure():
        nonlocal calls, first, last
        if time.monotonic()-fit_start > 600:
            raise TimeoutError('Linear fit exceeded ten optimisation minutes')
        opt.zero_grad()
        y = head(x)
        attack = weighted(F.binary_cross_entropy_with_logits(y[:,0],tg['attack_y'],reduction='none',pos_weight=torch.tensor(pw)),tg['attack_w'])
        snap = weighted(F.cross_entropy(y[:,1:4],tg['snap_y'],reduction='none'),tg['snap_w'])
        sal = weighted(F.binary_cross_entropy_with_logits(y[:,4:7],tg['sal_y'],reduction='none').mean(-1),tg['sal_w'])
        loss = .5*attack+.5*snap+.25*sal
        if not torch.isfinite(loss):
            raise ValueError('Non-finite linear fit loss')
        loss.backward()
        calls += 1
        last = float(loss.detach())
        if first is None:
            first = last
        return loss
    failure = None
    try:
        opt.step(closure)
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
    dest = OUT/'fits'
    dest.mkdir(exist_ok=True)
    torch.save(dict(head=head.state_dict(),seed=seed), dest/f'mel-s{seed}.pt')
    info = dict(seed=seed,recordings=len(rows),frames=len(x),features=640,closure_calls=calls,
                initial_loss=first,final_loss=last,failed=failure,zero_initialised=bool(failure),
                wall_s=time.monotonic()-started,optimisation_s=time.monotonic()-fit_start,
                sample_cap_per_recording=512,selection=selection)
    dump(dest/f'mel-s{seed}.json',info)
    del opt, x, tg
    gc.collect()
    w, b = head.weight.detach().numpy(), head.bias.detach().numpy()
    songs = json.loads((G1/'inference-songs.json').read_text())
    def load(sha):
        song = songs[sha]
        m = np.load(song['mel'],mmap_mode='r')
        y = np.empty((len(m),7),np.float32)
        for a in range(0,len(m),4096):
            ix = np.arange(a,min(a+4096,len(m)))
            y[ix] = stack_frames(m,ix,song['mean'],song['std'])@w.T+b
        return y
    head_rows(f'Mel-s{seed}',seed,logits_loader=load,output_path=OUT/'heads'/f'Mel-s{seed}.json')
    print(json.dumps({k:v for k,v in info.items() if k!='selection'}),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed',type=int,required=True)
    fit_and_score(p.parse_args().seed)
