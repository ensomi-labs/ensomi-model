"""R2 cross-entropy trainer: checkpoints, exact resume, NaN recovery, logs, pilot and free-run modes.

Modes (run on the mac from the repository root, or from a frozen code copy):

    train     python -m ensomi_model.r2.train_ce --config <run>/config.json [--resume]
    pilot     python -m ensomi_model.r2.train_ce --config <cfg> --pilot 200 --device mps
    freerun   python -m ensomi_model.r2.train_ce --config <run>/config.json --freerun-only <checkpoint>

Exit codes: 0 finished (exposure budget or clock), 3 stopped after the NaN-event
limit, 4 stopped by the resource guard, anything else is a crash (the
supervisor resumes from the latest checkpoint).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, fields
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import psutil
import torch

from ..research.oracle_time_continuation.runtime import ResourceConfig, ResourceGuard, ResourceLimit, atomic_checkpoint
from .common import ContractError
from .data import Corpus, chart_from_cache, load_or_build_manifest, track_from_json
from .cache import load_chart
from .export import export_chart, minimal_header
from .labels import load_star_labels
from .model import R2Config, R2Model
from .receipts import code_identity
from .report import chart_summary
from .sampling import continue_chart

GiB = 1024 ** 3
EXIT_NAN, EXIT_RESOURCE = 3, 4


@dataclass
class TrainConfig:
    run_dir: str = 'artifacts/r2-runs/dev'
    cache: str = 'artifacts/r2-cache/v1'
    device: str = 'cpu'
    threads: int = 4
    dtype: str = 'float32'
    conditioner: str = 'film'
    memory: str = 'landmarks'
    levels: int = 8
    star_conditions: str = 'auto'        # auto: on iff the star label file is complete
    total_exposures: int = 8_000_000     # head decisions; set from the pilot's throughput
    warmup_exposures: int = 50_000
    lr: float = 3e-4
    lr_min: float = 3e-5
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.01
    clip: float = 1.0
    accumulate: int = 4
    window: int = 256
    checkpoint_every: int = 250_000
    log_every: int = 2_000
    seed_weights: int = 171
    seed_draws: int = 471
    seed_validation: int = 954
    freerun: bool = True
    freerun_seeds: list = field(default_factory=lambda: [954, 955, 956])
    max_nan_events: int = 3
    stop_at_unix: float = 0.0
    rss_limit_gib: float = 12.0
    mps_limit_gib: float = 8.0

    @classmethod
    def load(cls, path, **overrides):
        data = json.loads(Path(path).read_text()) if path else {}
        data.update({k: v for k, v in overrides.items() if v is not None})
        names = {f.name for f in fields(cls)}
        unknown = set(data) - names
        if unknown:
            raise ContractError(f'Unknown config keys: {sorted(unknown)}')
        return cls(**data)


class NonFinite(Exception):
    def __init__(self, indices, what):
        super().__init__(f'non-finite {what} in windows {indices}')
        self.indices, self.what = list(indices), what


def lr_at(cfg: TrainConfig, exposures: int) -> float:
    if exposures < cfg.warmup_exposures:
        return cfg.lr * max(exposures, 1) / cfg.warmup_exposures
    span = max(1, cfg.total_exposures - cfg.warmup_exposures)
    progress = min(1.0, (exposures - cfg.warmup_exposures) / span)
    return cfg.lr_min + (cfg.lr - cfg.lr_min) * 0.5 * (1 + math.cos(math.pi * progress))


def cache_hashes(cache_root: Path):
    summary = json.loads((Path(cache_root) / 'summary.json').read_text())
    out = dict(index_sha256=summary['index_sha256'], splits_sha256=summary['splits_sha256'],
               version=summary['version'], candidates=summary['candidates'])
    star = Path(cache_root) / 'labels' / 'star_summary.json'
    if star.exists():
        out['star_labels'] = json.loads(star.read_text())
    return out


def mps_bytes(device):
    if torch.device(device).type != 'mps':
        return 0, 0
    return int(torch.mps.driver_allocated_memory()), int(torch.mps.current_allocated_memory())


class Stats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.d = dict(action=0.0, release=0.0, heads=0, gap_lns=0, eos=0.0, eos_n=0, natural=0.0, natural_n=0,
                      conditioned=0.0, conditioned_n=0, windows=0, loss=0.0, release_on_gap_heads=0.0)

    def add(self, out, track):
        d = self.d
        act = out.action.detach().cpu().double().numpy()
        rel = out.release.detach().cpu().double().numpy()
        head = ~out.eos
        d['action'] -= float(act[head].sum())
        d['release'] -= float(rel[head].sum())
        d['heads'] += int(head.sum())
        d['gap_lns'] += int(out.gap_lns[head].sum())
        d['eos'] -= float((act + rel)[out.eos].sum())
        d['eos_n'] += int(out.eos.sum())
        key = 'conditioned' if track else 'natural'
        d[key] -= float((act + rel).sum())
        d[key + '_n'] += len(act)
        d['windows'] += 1
        d['loss'] -= float((act + rel).mean())

    def summary(self):
        d = self.d

        def div(a, b):
            return a / b if b else None

        return dict(action_nll_per_decision=div(d['action'], d['heads']),
                    release_nll_per_decision=div(d['release'], d['heads']),
                    release_nll_per_gap_ln=div(d['release'], d['gap_lns']),
                    eos_nll=div(d['eos'], d['eos_n']), eos_count=d['eos_n'],
                    natural_nll_per_decision=div(d['natural'], d['natural_n']), natural_decisions=d['natural_n'],
                    conditioned_nll_per_decision=div(d['conditioned'], d['conditioned_n']),
                    conditioned_decisions=d['conditioned_n'], head_decisions=d['heads'], gap_lns=d['gap_lns'],
                    windows=d['windows'], window_mean_loss=div(d['loss'], d['windows']))


class Trainer:
    def __init__(self, cfg: TrainConfig, *, write=True):
        self.cfg = cfg
        self.write = write
        self.run = Path(cfg.run_dir)
        torch.set_num_threads(cfg.threads)
        self.device = torch.device(cfg.device)
        self.dtype = getattr(torch, cfg.dtype)
        labels, complete = load_star_labels(Path(cfg.cache))
        self.star = {'auto': complete, 'on': True, 'off': False}[cfg.star_conditions]
        self.star_complete = complete
        self.corpus = Corpus(cfg.cache, 'fit_train', star_conditions=self.star)
        torch.manual_seed(cfg.seed_weights)
        self.model = R2Model(R2Config(memory=cfg.memory, levels=cfg.levels, conditioner=cfg.conditioner),
                             verbose=write).to(self.device, self.dtype)
        decay = [p for p in self.model.parameters() if p.ndim >= 2]
        other = [p for p in self.model.parameters() if p.ndim < 2]
        self.opt = torch.optim.AdamW([dict(params=decay, weight_decay=cfg.weight_decay),
                                      dict(params=other, weight_decay=0.0)],
                                     lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)
        self.rng = np.random.default_rng(cfg.seed_draws)
        self.state = dict(exposures=0, windows=0, steps=0, lr_mult=1.0, nan_events=0, skip=[],
                          wall_s=0.0, restarts=0)
        self.stats = Stats()
        self.guard = None

    # ---- persistence -----------------------------------------------------------------------------

    @property
    def ckpt_dir(self):
        return self.run / 'checkpoints'

    def append(self, name, record):
        if not self.write:
            return
        with (self.run / name).open('a') as f:
            f.write(json.dumps(record, allow_nan=True) + '\n')

    def payload(self):
        return dict(model=self.model.state_dict(), optimizer=self.opt.state_dict(),
                    draw_rng=self.rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                    state=copy.deepcopy(self.state), config=asdict(self.cfg),
                    model_config=asdict(self.model.config), cache=cache_hashes(Path(self.cfg.cache)),
                    star_conditions=self.star)

    def save(self, tag=None):
        path = self.ckpt_dir / self.ckpt_name(tag)
        cfg = ResourceConfig(checkpoint_max_bytes=512 * 1024 ** 2)
        info = atomic_checkpoint(path, self.payload(), cfg)
        (self.ckpt_dir / 'latest.json').write_text(json.dumps(dict(path=path.name, **self.state_brief())))
        return path, info

    def state_brief(self):
        return {k: self.state[k] for k in ('exposures', 'windows', 'steps', 'lr_mult', 'nan_events')}

    def latest(self):
        marker = self.ckpt_dir / 'latest.json'
        if marker.exists():
            return self.ckpt_dir / json.loads(marker.read_text())['path']
        found = sorted(self.ckpt_dir.glob('ckpt-*.pt'))
        return found[-1] if found else None

    def load(self, path):
        data = torch.load(path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(data['model'])
        self.opt.load_state_dict(data['optimizer'])
        self.rng.bit_generator.state = data['draw_rng']
        torch.set_rng_state(data['torch_rng'])
        self.state = data['state']
        if data.get('star_conditions', self.star) != self.star:
            raise ContractError('Checkpoint was trained with a different star_conditions setting')
        return data

    # ---- one optimizer step ----------------------------------------------------------------------

    def next_batch(self):
        batch = []
        while len(batch) < self.cfg.accumulate:
            index = self.state['windows']
            draw = self.corpus.draw(self.rng, self.cfg.window)
            self.state['windows'] += 1
            if index in self.state['skip']:
                continue
            batch.append((index, draw))
        return batch

    def step(self, batch, stats=None):
        cfg = self.cfg
        lr = lr_at(cfg, self.state['exposures']) * self.state['lr_mult']
        for group in self.opt.param_groups:
            group['lr'] = lr
        self.opt.zero_grad(set_to_none=True)
        heads, losses = 0, []
        self.model.train()
        for index, draw in batch:
            chart = self.corpus.chart(draw.sha)
            out = self.model.window(chart, draw.start, draw.stop, draw.track)
            loss = -out.total.mean()
            if not torch.isfinite(loss):
                raise NonFinite([index], 'loss')
            (loss / len(batch)).backward()
            losses.append(loss.item())
            heads += int((~out.eos).sum())
            if stats is not None:
                stats.add(out, draw.track)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.clip)
        if not torch.isfinite(norm):
            raise NonFinite([i for i, _ in batch], 'gradient')
        self.opt.step()
        self.state['steps'] += 1
        self.state['exposures'] += heads
        return dict(lr=lr, heads=heads, loss=float(np.mean(losses)), grad_norm=float(norm))

    # ---- evaluation ------------------------------------------------------------------------------

    def manifest(self):
        return load_or_build_manifest(self.cfg.cache, self.run / 'fit_dev_manifest.json', star_conditions=self.star)

    @torch.no_grad()
    def evaluate(self):
        manifest = self.manifest()
        dev = Corpus(self.cfg.cache, 'fit_dev', star_conditions=False)
        stats = Stats()
        self.model.eval()
        t0 = time.time()
        for w in manifest['windows']:
            chart = dev.chart(w['sha256'])
            out = self.model.window(chart, w['start'], w['stop'], track_from_json(w['track']))
            stats.add(out, w['track'])
        record = dict(exposures=self.state['exposures'], fit_dev=stats.summary(), eval_s=time.time() - t0)
        if self.cfg.freerun:
            record['freerun'] = self.freerun(manifest, dev)
        self.model.train()
        return record

    def freerun(self, manifest, dev, seeds=None):
        cpu_model = copy.deepcopy(self.model).to('cpu', torch.float32).eval()
        out = []
        folder = self.run / 'freerun' / f'{self.state["exposures"]:010d}'
        for sha in manifest['freerun']:
            dec = load_chart(Path(self.cfg.cache) / 'charts' / dev.files[sha])
            chart = chart_from_cache(dec)
            source_objects, _ = export_chart(dec.head_ms, dec.song_ms, dec.actions, dec.gap_release_ms, None, '')
            entry = dict(sha256=sha, K=chart.K, source=chart_summary(source_objects, dec.head_ms, dec.song_ms,
                                                                       chart.grid), seeds={})
            for seed in (seeds or self.cfg.freerun_seeds):
                t0 = time.time()
                try:
                    acts, gap = continue_chart(cpu_model, dec.head_ms, dec.song_ms, chart.grid, seed=seed)
                    dest = folder / f'{sha[:16]}-{seed}.osu' if self.write else None
                    objects, _ = export_chart(dec.head_ms, dec.song_ms, acts, gap, dest,
                                              minimal_header(dec.grid_segments, f'R2 {sha[:12]} seed {seed}'))
                    summary = chart_summary(objects, dec.head_ms, dec.song_ms, chart.grid)
                    summary.update(eos_closed=True, seconds=time.time() - t0)
                except ContractError as exc:
                    summary = dict(error=str(exc), seconds=time.time() - t0)
                entry['seeds'][str(seed)] = summary
            out.append(entry)
        return out

    # ---- run loop --------------------------------------------------------------------------------

    def receipt(self, mode):
        return dict(entry_point=' '.join(sys.argv), mode=mode, config=asdict(self.cfg), code=code_identity(),
                    cache=cache_hashes(Path(self.cfg.cache)), star_conditions=self.star,
                    star_labels_complete=self.star_complete, seeds=dict(weights=self.cfg.seed_weights,
                                                                        draws=self.cfg.seed_draws,
                                                                        validation=self.cfg.seed_validation),
                    device=self.cfg.device, torch=torch.__version__, host=platform.node(), pid=os.getpid(),
                    ens_job=os.environ.get('ENS_JOB_ID'), started_unix=time.time(),
                    parameters=self.model.parameter_counts())

    def handle_nonfinite(self, exc: NonFinite):
        events = self.state['nan_events'] + 1
        mult = self.state['lr_mult'] * 0.5
        skip = sorted(set(self.state['skip']) | set(exc.indices))
        record = dict(event='nonfinite', what=exc.what, windows=exc.indices, exposures=self.state['exposures'],
                      nan_events=events, lr_mult=mult, time=time.time())
        if events >= self.cfg.max_nan_events:
            record['action'] = 'stop'
            self.append('events.jsonl', record)
            return False
        latest = self.latest()
        self.load(latest)
        self.state.update(nan_events=events, lr_mult=mult, skip=skip)
        record.update(action='reloaded', checkpoint=latest.name)
        self.append('events.jsonl', record)
        return True

    def train(self, resume: bool):
        cfg = self.cfg
        self.run.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(exist_ok=True)
        if resume and self.latest() is not None:
            latest = self.latest()
            self.load(latest)
            self.state['restarts'] = self.state.get('restarts', 0) + 1
            self.append('events.jsonl', dict(event='resume', checkpoint=latest.name, time=time.time(),
                                             **self.state_brief()))
            if self.state['exposures'] > 0 and latest.name not in self.evaluated():
                self.eval_and_log(latest)  # the previous process stopped before this checkpoint's evaluation
        else:
            if self.latest() is not None:
                raise ContractError('Run directory already has checkpoints; pass --resume')
            (self.run / 'config.json').write_text(json.dumps(asdict(cfg), indent=1))
            self.save()
            self.append('events.jsonl', dict(event='start', time=time.time(), star_conditions=self.star))
        (self.run / f'receipt-{int(time.time())}.json').write_text(json.dumps(self.receipt('train'), indent=1))
        resources = (self.run / 'resources.jsonl').open('a')
        self.guard = ResourceGuard(cfg.device, ResourceConfig(
            driver_limit_bytes=int(min(cfg.mps_limit_gib, 8) * GiB), rss_limit_bytes=int(cfg.rss_limit_gib * GiB),
            allocator_ceiling_bytes=8 * GiB, output_max_bytes=2 * GiB), log=resources)
        manifest = self.manifest()
        next_log = (self.state['exposures'] // cfg.log_every + 1) * cfg.log_every
        next_ckpt = (self.state['exposures'] // cfg.checkpoint_every + 1) * cfg.checkpoint_every
        t_start, t_last, heads_last = time.time(), time.time(), self.state['exposures']
        wall0 = self.state['wall_s']
        status = 'finished'
        self.stats.reset()
        while self.state['exposures'] < cfg.total_exposures:
            if cfg.stop_at_unix and time.time() >= cfg.stop_at_unix:
                status = 'clock'
                break
            batch = self.next_batch()
            try:
                info = self.step(batch, self.stats)
            except NonFinite as exc:
                if not self.handle_nonfinite(exc):
                    self.write_run_json('nan_limit', t_start, wall0)
                    return EXIT_NAN
                self.stats.reset()
                next_log = (self.state['exposures'] // cfg.log_every + 1) * cfg.log_every
                next_ckpt = (self.state['exposures'] // cfg.checkpoint_every + 1) * cfg.checkpoint_every
                continue
            self.state['wall_s'] = wall0 + time.time() - t_start
            if self.state['exposures'] >= next_log:
                now = time.time()
                try:
                    res = self.guard.check('log', exposures=self.state['exposures'])
                except ResourceLimit as exc:
                    self.append('events.jsonl', dict(event='resource_limit', error=str(exc), time=time.time(),
                                                     **self.state_brief()))
                    self.save('safe')
                    self.write_run_json('resource_limit', t_start, wall0)
                    return EXIT_RESOURCE
                drv, cur = mps_bytes(cfg.device)
                record = dict(exposures=self.state['exposures'], steps=self.state['steps'],
                              windows_drawn=self.state['windows'], lr=info['lr'], grad_norm=info['grad_norm'],
                              decisions_per_s=(self.state['exposures'] - heads_last) / max(1e-9, now - t_last),
                              rss_gib=res['rss_bytes'] / GiB, mps_driver_gib=drv / GiB, mps_current_gib=cur / GiB,
                              wall_s=self.state['wall_s'], time=now, **self.stats.summary())
                self.append('train.jsonl', record)
                self.stats.reset()
                t_last, heads_last = now, self.state['exposures']
                next_log = (self.state['exposures'] // cfg.log_every + 1) * cfg.log_every
            if self.state['exposures'] >= next_ckpt:
                self.checkpoint_and_eval()
                next_ckpt = (self.state['exposures'] // cfg.checkpoint_every + 1) * cfg.checkpoint_every
                self.write_run_json('running', t_start, wall0)
        if self.latest() is None or self.latest().name != self.ckpt_name():
            self.checkpoint_and_eval()
        self.write_run_json(status, t_start, wall0)
        return 0

    def ckpt_name(self, tag=None):
        return f'ckpt-{self.state["exposures"]:010d}{"-" + tag if tag else ""}.pt'

    def checkpoint_and_eval(self):
        path, info = self.save()
        self.eval_and_log(path, info['checkpoint_bytes'])

    def eval_and_log(self, path, size=None):
        """Evaluation failures are logged, never fatal: an unattended run keeps training."""
        try:
            record = self.evaluate()
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.model.train()
            record = dict(exposures=self.state['exposures'], error=f'{type(exc).__name__}: {exc}')
            self.append('events.jsonl', dict(event='eval_error', checkpoint=path.name, time=time.time(),
                                             traceback=traceback.format_exc()[-4000:]))
        record.update(checkpoint=path.name, checkpoint_bytes=size, time=time.time())
        self.append('evals.jsonl', record)

    def evaluated(self):
        log = self.run / 'evals.jsonl'
        if not log.exists():
            return set()
        return {json.loads(line).get('checkpoint') for line in log.read_text().splitlines() if line.strip()}

    def write_run_json(self, status, t_start, wall0):
        if not self.write:
            return
        wall = self.state['wall_s']
        data = dict(status=status, device=self.cfg.device, exposures=self.state['exposures'],
                    total_exposures=self.cfg.total_exposures, steps=self.state['steps'],
                    windows=self.state['windows'], wall_s=wall,
                    decisions_per_s_wall=self.state['exposures'] / max(wall, 1e-9),
                    nan_events=self.state['nan_events'], lr_mult=self.state['lr_mult'],
                    restarts=self.state.get('restarts', 0), updated=time.time())
        (self.run / 'run.json').write_text(json.dumps(data, indent=1))

    # ---- pilot -----------------------------------------------------------------------------------

    def pilot(self, windows: int, save_to=None):
        """Train ``windows`` windows from initialization; report throughput and memory as JSON."""
        proc = psutil.Process()
        per, rss, drv = [], [], []
        losses = []
        heads_total, t_total = 0, 0.0
        steps = max(1, windows // self.cfg.accumulate)
        drv0 = mps_bytes(self.cfg.device)[0]
        for s in range(steps):
            batch = self.next_batch()
            t0 = time.time()
            info = self.step(batch)
            if self.device.type == 'mps':
                torch.mps.synchronize()
            dt = time.time() - t0
            per.append(dt)
            losses.append(info['loss'])
            if s >= 2:  # exclude warm-up steps from throughput
                heads_total += info['heads']
                t_total += dt
            rss.append(proc.memory_info().rss)
            drv.append(mps_bytes(self.cfg.device)[0])
        k = max(1, len(losses) // 5)
        result = dict(device=self.cfg.device, threads=self.cfg.threads, conditioner=self.cfg.conditioner,
                      memory=self.cfg.memory, levels=self.cfg.levels, windows=steps * self.cfg.accumulate,
                      steps=steps, head_decisions=self.state['exposures'],
                      decisions_per_s=heads_total / max(t_total, 1e-9),
                      seconds_per_step=dict(median=float(np.median(per)), p90=float(np.percentile(per, 90)),
                                            max=float(np.max(per))),
                      peak_rss_gib=max(rss) / GiB, final_rss_gib=rss[-1] / GiB,
                      mps_driver_gib_start=drv0 / GiB, mps_driver_gib_peak=max(drv) / GiB,
                      mps_driver_gib_series=[d / GiB for d in drv[::max(1, len(drv) // 20)]],
                      mps_monotone_growth=bool(len(drv) > 2 and all(b >= a for a, b in zip(drv, drv[1:]))
                                               and drv[-1] > drv[0]),
                      loss_first=float(np.mean(losses[:k])), loss_last=float(np.mean(losses[-k:])),
                      all_finite=bool(np.all(np.isfinite(losses))),
                      parameters=self.model.parameter_counts()['total'], star_conditions=self.star)
        if save_to:
            Path(save_to).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.payload(), save_to)
            result['checkpoint'] = str(save_to)
        return result


def main(argv=None):
    p = argparse.ArgumentParser(description='R2 CE trainer')
    p.add_argument('--config', default=None, help='JSON TrainConfig; defaults when omitted')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--pilot', type=int, default=0, help='train N windows from init and print JSON')
    p.add_argument('--pilot-out', default=None)
    p.add_argument('--pilot-checkpoint', default=None)
    p.add_argument('--freerun-only', default=None, help='checkpoint to free-run on the manifest charts')
    p.add_argument('--device', default=None)
    p.add_argument('--threads', type=int, default=None)
    p.add_argument('--memory', default=None)
    p.add_argument('--levels', type=int, default=None)
    p.add_argument('--conditioner', default=None)
    p.add_argument('--run-dir', default=None)
    a = p.parse_args(argv)
    frozen = os.environ.get('R2_FROZEN_CODE')
    if frozen and not Path(__file__).resolve().is_relative_to(Path(frozen).resolve()):
        raise SystemExit(f'trainer imported from {__file__}, not from the frozen copy {frozen}')
    cfg = TrainConfig.load(a.config, device=a.device, threads=a.threads, memory=a.memory, levels=a.levels,
                           conditioner=a.conditioner, run_dir=a.run_dir)
    if a.pilot:
        trainer = Trainer(cfg, write=False)
        result = trainer.pilot(a.pilot, a.pilot_checkpoint)
        result['receipt'] = trainer.receipt('pilot')
        text = json.dumps(result, indent=1)
        print(text, flush=True)
        if a.pilot_out:
            Path(a.pilot_out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.pilot_out).write_text(text)
        return 0
    if a.freerun_only:
        trainer = Trainer(cfg, write=True)
        trainer.run.mkdir(parents=True, exist_ok=True)
        data = torch.load(a.freerun_only, map_location='cpu', weights_only=False)
        trainer.model.load_state_dict(data['model'])
        trainer.state = data['state']
        manifest = trainer.manifest()
        dev = Corpus(cfg.cache, 'fit_dev', star_conditions=False)
        report = dict(checkpoint=str(a.freerun_only), exposures=trainer.state['exposures'],
                      freerun=trainer.freerun(manifest, dev))
        print(json.dumps(report, indent=1), flush=True)
        trainer.append('freerun_only.jsonl', report)
        return 0
    trainer = Trainer(cfg)
    return trainer.train(a.resume)


if __name__ == '__main__':
    sys.exit(main())
