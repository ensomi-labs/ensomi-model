"""Train and score the re-keyed copy path on frozen R2's cached mask marginals (human charts, R2's targets).

Inputs, from the 2026-10-09 copy study and the rows build of 2026-10-10:
- ``--tf``: per chart ``<sha[:16]>.npz`` with lp [n, 15] (frozen 56M log mask marginal, teacher forced on the
  human arrangement), h [n] (head masks), b [n + 8] (head beats), n = min(K, 2,048);
- ``--panels``: ``panels.json`` (train 1,600 fit_train charts; V, P, TR, Q held-out panels);
- ``--rows-data``: the rows build (``audio_rows.beatdata``) for each chart's audio key and standardised log-Mel.

Subcommands::

    prepare --tf T --panels P --rows-data D --out O       per-row audio -> O/audio/<sha16>.npy (float16)
    train NAME --tf T --panels P --out O [--key-tol 1 --audio 1 --exclude-rows 2 --exclude-beats 2] [--steps 1500]
    eval NAME=CKPT[,NAME=CKPT...] --tf T --panels P --out O   teacher-forced NLL with ablations

``--out`` holds the large files (audio, checkpoints); logs, tables and summaries go to ``--report`` (default
``--out``).

Training (unchanged from the study): 8 charts x 64 query rows per step against every earlier row of the chart,
AdamW lr 3e-3, 200 warm-up steps, cosine to 10 %, gradient clip 1. Validation on V every 500 steps. Kill rules
(guardrail 9): at step 500 the V gain over R2 must reach half the log-linear probe's V gain (``--ref``, .067);
later, stop when V NLL exceeds its best by .005 twice in a row. Charts whose audio is missing from the rows
build are left out of every arm, so all arms see the same charts.

``eval`` scores P, V and TR: the path, its copy terms zeroed (R2 with the count/lane betas), R2 alone, and for
audio arms the audio rows shuffled within the chart (the input-use ablation of guardrail 3). It also writes the
head-averaged attention summary per row (``copy.attention_stats``) for P.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

from .copy import CopyConfig, CopyPath, attention_stats, config_dict, parts_nll, row_audio, timing_keys

DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
CAP = 2048


def _sha16(sha):
    return sha[:16]


def load(sha, tf_dir, audio_dir):
    with np.load(Path(tf_dir) / f'{_sha16(sha)}.npz') as f:
        d = dict(lp=f['lp'], h=f['h'].astype(np.int64), b=f['b'])
    d['keys'] = timing_keys(d['b'], len(d['h']))
    d['aud'] = np.load(Path(audio_dir) / f'{_sha16(sha)}.npy').astype(np.float32)
    return d


def prepare(tf_dir: Path, panels: Path, rows_data: Path, cache: Path, out: Path, report: Path):
    """Per-row audio of every panel chart whose audio is in the rows build."""
    import pyarrow.parquet as pq
    from ..r2.cache import load_chart
    from ..r2.common import GridArrays, grid_from_arrays
    t0 = time.time()
    index = json.loads((rows_data / 'index.json').read_text())
    features = Path(index['features'])
    key_of = {r['sha']: r['key'] for r in index['rows']}
    files = pq.read_table(cache / 'index.parquet', columns=['sha256', 'file']).to_pandas().set_index('sha256').file
    pn = json.loads(panels.read_text())
    dest = out / 'audio'
    dest.mkdir(parents=True, exist_ok=True)
    missing = {}
    done = 0
    for name, shas in pn.items():
        for sha in shas:
            target = dest / f'{_sha16(sha)}.npy'
            if target.exists():
                continue
            if sha not in key_of:
                missing.setdefault(name, []).append(sha)
                continue
            dec = load_chart(cache / 'charts' / files.loc[sha])
            with np.load(tf_dir / f'{_sha16(sha)}.npz') as f:
                n = len(f['h'])
            led = rows_data / 'mel' / f'{key_of[sha]}.npy'
            mel = np.load(led if led.exists() else features / f'{key_of[sha]}.npy', mmap_mode='r')
            grid = GridArrays.from_grid(grid_from_arrays(dec.grid_segments, dec.grid_bars))
            np.save(target, row_audio(mel, dec.head_ms[:n], grid).astype(np.float16))
            done += 1
    summary = dict(written=done, missing={k: len(v) for k, v in missing.items()}, missing_shas=missing,
                   charts={k: len(v) for k, v in pn.items()}, seconds=round(time.time() - t0, 1))
    (report / 'audio-summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'missing_shas'}), flush=True)


def available(shas, audio_dir):
    return [s for s in shas if (Path(audio_dir) / f'{_sha16(s)}.npy').exists()]


class Packed:
    """All charts of a set in flat device tensors (rows capped at 2,048 per chart)."""

    def __init__(self, data):
        self.n = np.array([len(d['h']) for d in data])
        self.off = np.concatenate([[0], np.cumsum(self.n)[:-1]])
        cat = lambda key, dtype: torch.as_tensor(np.concatenate([d[key][:len(d['h'])] for d in data]), dtype=dtype,
                                                 device=DEV)
        self.h, self.lp = cat('h', torch.long), cat('lp', torch.float32)
        self.b, self.keys, self.aud = cat('b', torch.float32), cat('keys', torch.long), cat('aud', torch.float32)

    def batch(self, charts, rows, pad=512):
        """charts [B] indices, rows: list of sorted local row arrays (equal length q) -> model inputs, targets.
        Keys are padded to a multiple of pad so that MPS sees few distinct shapes."""
        J = max(int(r.max()) for r in rows)
        J = max(pad, -(-J // pad) * pad)
        off = torch.as_tensor(self.off[charts], device=DEV)[:, None]
        n = torch.as_tensor(self.n[charts], device=DEV)[:, None]
        j = torch.arange(J, device=DEV)[None]
        g = off + torch.minimum(j, n - 1)
        rq = torch.as_tensor(np.stack(rows), device=DEV)
        gq = off + rq
        return (rq, self.b[gq], self.keys[gq], self.aud[gq], self.lp[gq], self.h[g], self.b[g], self.keys[g],
                self.aud[g], j < n), self.h[gq]


@torch.no_grad()
def score_chart(model, d, block=256, shuffle_audio=False, stats=False, **kw):
    """Per-row (count, lanes, total) NLL [n, 3] of R2 + path and of R2 alone; with ``stats`` also the
    per-row attention summary (``attention_stats``)."""
    n = len(d['h'])
    t = torch.as_tensor(d['h'], device=DEV)
    lp = torch.as_tensor(d['lp'], device=DEV)
    b = torch.as_tensor(d['b'], dtype=torch.float32, device=DEV)
    keys = torch.as_tensor(d['keys'], device=DEV)
    aud = torch.as_tensor(d['aud'], device=DEV)
    if shuffle_audio:           # audio rows of the same chart in a fixed random order; everything else unchanged
        aud = aud[torch.as_tensor(np.random.default_rng(0).permutation(n), device=DEV)]
    out, summary = [], []
    for s in range(0, n, block):
        rows = torch.arange(s, min(n, s + block), device=DEV)
        trace = {} if stats else None
        out.append(model.chart_bias(rows, t, b, keys, aud, lp[rows], pad=256, trace=trace, **kw))
        if stats and trace:
            summary.append(attention_stats(trace))
    bias = torch.cat(out)
    res = (torch.stack(parts_nll(lp, bias, t), 1).cpu().numpy(), torch.stack(parts_nll(lp, 0 * bias, t), 1).cpu().numpy())
    if stats:
        return res + ({k: np.concatenate([x[k] for x in summary]) for k in summary[0]} if summary else {},)
    return res


def panel_gain(model, data):
    model.eval()
    v = [score_chart(model, d) for d in data]
    model.train()
    return float(np.mean([x[0][:, 2].mean() for x in v])), float(np.mean([x[1][:, 2].mean() for x in v]))


def train(a):
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    pn = json.loads(Path(a.panels).read_text())
    audio_dir = Path(a.out) / 'audio'
    start = time.monotonic()
    shas = available(pn['train'][:a.limit] if a.limit else pn['train'], audio_dir)
    train_data = [load(s, a.tf, audio_dir) for s in shas]
    val = [load(s, a.tf, audio_dir) for s in available(pn['V'], audio_dir)]
    cfg = CopyConfig(heads=a.heads, hidden=a.hidden, key_tol=a.key_tol, audio=bool(a.audio),
                     exclude_rows=a.exclude_rows, exclude_beats=a.exclude_beats)
    model = CopyPath(cfg).to(DEV)
    models = Path(a.out) / 'models'
    models.mkdir(parents=True, exist_ok=True)
    log = open(Path(a.report) / f'train-{a.name}.jsonl', 'w')

    def emit(**rec):
        rec.update(name=a.name, seconds=round(time.monotonic() - start, 1))
        line = json.dumps(rec, default=str)
        print(line, flush=True)
        log.write(line + '\n')
        log.flush()

    def save(path, step):
        args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
        torch.save(dict(state=model.state_dict(), config=config_dict(cfg), args=args, step=step), path)

    emit(event='start', params=sum(p.numel() for p in model.parameters()), train_charts=len(train_data),
         val_charts=len(val), rows=int(sum(len(d['h']) for d in train_data)), device=DEV, config=config_dict(cfg),
         **vars(a))
    pack = Packed(train_data)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))))
    best, bad, status = float('inf'), 0, 'finished'
    run_loss = run_base = 0.0
    for step in range(1, a.steps + 1):
        charts = rng.choice(len(pack.n), a.charts, replace=False)
        rows = [np.sort(rng.choice(pack.n[c], a.queries, replace=pack.n[c] < a.queries)) for c in charts]
        inputs, t = pack.batch(charts, rows)
        lp = inputs[4]
        bias = model.bias(*inputs)
        loss = parts_nll(lp, bias, t)[2].mean()
        with torch.no_grad():
            base = parts_nll(lp, 0 * bias, t)[2].mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run_loss += float(loss.detach())
        run_base += float(base)
        if step % 100 == 0:
            emit(event='train', step=step, nll=run_loss / 100, r2=run_base / 100, gain=(run_base - run_loss) / 100,
                 lr=sched.get_last_lr()[0])
            run_loss = run_base = 0.0
            if DEV == 'mps' and step % 500 == 0:
                torch.mps.empty_cache()
        if step % a.every == 0 or step == a.steps:
            v, r2 = panel_gain(model, val)
            gain = r2 - v
            emit(event='val', step=step, nll=v, r2=r2, gain=gain, ref=a.ref)
            save(models / f'{a.name}.pt', step)
            save(models / f'{a.name}-{step}.pt', step)
            if a.ref is not None and step == a.pilot and gain < 0.5 * a.ref:
                status = f'killed at step {step}: V gain {gain:.4f} < half the probe reference {a.ref:.4f}'
                break
            if v < best:
                best, bad = v, 0
                save(models / f'{a.name}-best.pt', step)
            elif v > best + 0.005:
                bad += 1
                if bad >= 2:
                    status = f'killed: V NLL {v:.4f} above best {best:.4f} + .005 twice'
                    break
            else:
                bad = 0
    emit(event='end', status=status, best_val=best)
    log.close()
    if status.startswith('killed'):
        sys.exit(3)


def load_model(path):
    """(CopyPath on DEV, name) from a checkpoint of this module or of the 2026-10-09 study ('full' variant)."""
    data = torch.load(path, map_location='cpu', weights_only=False)
    if 'config' in data:
        cfg = CopyConfig(**data['config'])
    else:
        args = data['args']
        if args.get('arch', 'copy') != 'copy' or args.get('variant', 'full') != 'full':
            raise ValueError(f'{path}: only the study\'s full copy path maps onto CopyPath')
        cfg = CopyConfig(heads=args['heads'], hidden=args['hidden'], key_tol=0, audio=False, exclude_rows=0,
                         exclude_beats=0.0)
    model = CopyPath(cfg)
    model.load_state_dict(data['state'])
    return model.to(DEV).eval()


def boot(x, reps=2000, seed=712):
    x = np.asarray(x, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(reps, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def evaluate(names, paths, a):
    """Teacher-forced NLL per chart and variant; paired chart bootstraps against R2 and against the path."""
    import pandas as pd
    pn = json.loads(Path(a.panels).read_text())
    audio_dir = Path(a.out) / 'audio'
    rows, att = [], []
    for name, path in zip(names, paths):
        model = load_model(path)
        variants = {'path': {}, 'nocopy': dict(use_copy=False)}
        if model.cfg.audio:
            variants['audio_shuffled'] = dict(shuffle_audio=True)
        for panel in ('P', 'V', 'TR'):
            for sha in available(pn[panel], audio_dir):
                d = load(sha, a.tf, audio_dir)
                for label, kw in variants.items():
                    res = score_chart(model, d, stats=label == 'path' and panel == 'P', **kw)
                    out, base = res[0], res[1]
                    if len(res) == 3 and res[2]:
                        att.append(dict(model=name, sha256=sha, rows=len(d['h']),
                                        **{k: float(np.mean(v)) for k, v in res[2].items()}))
                    rows.append(dict(model=name, panel=panel, sha256=sha, variant=label, count=out[:, 0].mean(),
                                     lanes=out[:, 1].mean(), total=out[:, 2].mean()))
                rows.append(dict(model=name, panel=panel, sha256=sha, variant='R2', count=base[:, 0].mean(),
                                 lanes=base[:, 1].mean(), total=base[:, 2].mean()))
        print(name, 'scored', flush=True)
    df = pd.DataFrame(rows)
    out = Path(a.report)
    df.to_csv(out / f'copyeval-per-chart-{a.tag}.csv', index=False)
    pd.DataFrame(att).to_csv(out / f'copy-attention-P-{a.tag}.csv', index=False)
    summary = []
    for (name, panel), part in df.groupby(['model', 'panel']):
        t = part.pivot(index='sha256', columns='variant', values=['count', 'lanes', 'total'])
        for label in [v for v in ('path', 'nocopy', 'audio_shuffled') if ('total', v) in t.columns]:
            for ref in ['R2'] + (['path'] if label != 'path' else []):
                row = dict(model=name, panel=panel, comparison=f'{label}-{ref}', charts=len(t))
                for k in ('count', 'lanes', 'total'):
                    row[k], row[k + '_lo'], row[k + '_hi'] = boot(t[k][label] - t[k][ref])
                summary.append(row)
    # paired differences between models on the same charts (path NLL), against the first model
    tot = df[df.variant == 'path'].pivot_table(index=['panel', 'sha256'], columns='model', values='total')
    for name in names[1:]:
        for panel in ('P', 'V', 'TR'):
            x = tot.loc[panel]
            row = dict(model=name, panel=panel, comparison=f'path-path[{names[0]}]', charts=len(x))
            row['total'], row['total_lo'], row['total_hi'] = boot(x[name] - x[names[0]])
            summary.append(row)
    pd.DataFrame(summary).to_csv(out / f'copyeval-{a.tag}.csv', index=False)
    print(pd.DataFrame(summary).to_string(), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('cmd', choices=('prepare', 'train', 'eval'))
    p.add_argument('name', nargs='?', help='train: run name; eval: NAME=checkpoint[,NAME=checkpoint...]')
    p.add_argument('--tf', type=Path, default=Path('artifacts/r2-copy-20261009/bulk/tf/56M'))
    p.add_argument('--panels', type=Path, default=Path('artifacts/r2-copy-20261009/panels.json'))
    p.add_argument('--rows-data', type=Path)
    p.add_argument('--cache', type=Path, default=Path('artifacts/r2-cache/v1'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--report', type=Path, help='logs and tables (default: --out)')
    p.add_argument('--tag', default='p2')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--pilot', type=int, default=500)
    p.add_argument('--every', type=int, default=500)
    p.add_argument('--ref', type=float, default=0.067, help='log-linear probe V gain (nats/row) for the kill rule')
    p.add_argument('--charts', type=int, default=8)
    p.add_argument('--queries', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--key-tol', type=int, default=1)
    p.add_argument('--audio', type=int, default=1)
    p.add_argument('--exclude-rows', type=int, default=2)
    p.add_argument('--exclude-beats', type=float, default=2.0)
    p.add_argument('--limit', type=int, default=0, help='use only this many training charts (smoke tests)')
    a = p.parse_args(argv)
    a.report = a.report or a.out
    a.report.mkdir(parents=True, exist_ok=True)
    if a.cmd == 'prepare':
        prepare(a.tf, a.panels, a.rows_data, a.cache, a.out, a.report)
    elif a.cmd == 'train':
        train(a)
    else:
        pairs = [x.split('=', 1) for x in a.name.split(',')]
        evaluate([n for n, _ in pairs], [q for _, q in pairs], a)


if __name__ == '__main__':
    main()
