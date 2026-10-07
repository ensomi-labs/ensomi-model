"""R2 cross-entropy trainer: checkpoints, exact resume, NaN recovery, logs, pilot and free-run modes.

Modes (run on the mac from the repository root, or from a frozen code copy):

    train     python -m ensomi_model.r2.train_ce --config <run>/config.json [--resume]
    pilot     python -m ensomi_model.r2.train_ce --config <cfg> --pilot 200 --device cpu
    freerun   python -m ensomi_model.r2.train_ce --config <run>/config.json --freerun-only <checkpoint>

Two phases (plan v5), chosen by ``phase``:

- ``natural`` (phase N): natural chart structure by teacher-forced CE with no conditions
  (``draw.selection='natural'``: the v1 start rule, empty tracks). The optimiser holds the
  natural parameters only; the conditioning path keeps its initialisation.
- ``conditions`` (phase C): ``init_from`` names a phase-N checkpoint whose natural parameters
  initialise the model. The conditioning path starts with a zero output layer and passes every
  decision that reads no interval through unchanged (``R2Model.film``), so phase C starts as the
  phase-N model exactly. ``base_mode='frozen'``: the optimiser holds the conditioning path only
  (``R2Model.parameter_split``); natural decisions cannot move. ``base_mode='kl'``: every
  parameter trains and a frozen copy of the phase-N model is the reference of the KL term
  (``kl_weight``, ``kl_direction``, ``kl_decisions``); ``natural_ce`` says whether L_base still
  scores natural decisions.

The loss is ``loss.py`` (uniform per-decision CE on fixed divisors, inverse-probability
weights on natural decisions, condition terms owned by rule-L visibility, weights
``lambda_ln`` and ``lambda_star``). The divisors ``n_bar*`` come from ``draw_sim`` and are
refused unless ``n_bar_key`` matches the configured draw. ``mu_star > 0`` adds the
relaxed-proxy term on one window in ``proxy_every`` and the true-F3 calibration measurement
on one update in ``f3_every`` (residual difficulty only). Undecided values stay null in the
configs and ``check_config`` refuses to start until they are set.

The learning rate is the warm-up plus cosine of v1 unless ``lr_schedule`` lists phases
``{"until": exposures, "shape": "constant"|"linear"|"cosine", "lr_start": x, "lr_end": y}``;
each phase runs from the previous ``until`` (0 for the first) to its own, and exposures past
the last phase keep its ``lr_end``.

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
import sys
import time

import numpy as np
import psutil
import torch

from ..research.oracle_time_continuation.runtime import (ResourceConfig, ResourceGuard, ResourceLimit,
                                                        atomic_checkpoint, snapshot)
from .common import ContractError
from .conditions import DrawConfig
from .data import Corpus, chart_from_cache, load_or_build_manifests, manifest_version
from .cache import load_chart
from .draw_sim import n_bar_key
from .export import export_chart, minimal_header
from .features import LN_LEVEL_MODES
from .locality import RULE_L_VERSION
from .loss import KL_DECISIONS, KL_DIRECTIONS, LossConfig, window_kl, window_loss, window_terms
from .model import CONDITIONERS, MAX_PARAMETERS, R2Config, R2Model
from .properties import NU_HASH, PROPERTIES_SOURCE_SHA256
from .receipts import code_identity
from .report import chart_summary
from .sampling import continue_chart

GiB = 1024 ** 3
EXIT_NAN, EXIT_RESOURCE = 3, 4
NO_SWAP_TRIP = 1 << 60   # the system-wide swap-growth trip measures other processes; disabled (plan Q-D)
LR_SHAPES = ('constant', 'linear', 'cosine')
PHASES = ('natural', 'conditions')
BASE_MODES = ('frozen', 'kl')
PHASE_C_KEYS = ('init_from', 'base_mode', 'lambda_ln', 'lambda_star', 'mu_star')
KL_KEYS = ('kl_weight', 'kl_direction', 'kl_decisions', 'natural_ce')
BUDGET_KEYS = ('total_exposures', 'checkpoint_every', 'g3c_exposures')
COSINE_KEYS = ('lr', 'lr_min', 'warmup_exposures')    # the v1 warm-up plus cosine, used without lr_schedule
NATURAL_ARCH = ('hidden', 'levels', 'expansion', 'rank', 'memory', 'stride', 'code_dim')


@dataclass
class TrainConfig:
    run_dir: str = 'artifacts/r2-runs/dev'
    cache: str = 'artifacts/r2-cache/v1'
    phase: str | None = None             # 'natural' (phase N) | 'conditions' (phase C)
    init_from: str | None = None         # phase C: the phase-N checkpoint (its natural parameters)
    warm_start: str | None = None        # phase N: complete weights; fresh optimizer and exposure counter
    ln_level: str = 'off'
    ln_length: str = 'off'
    ln_prior: str | None = None          # explicit evaluation prior path; never fitted by the trainer
    ln_level_dropout: float = 0.3        # independent whole-window dropout to the unknown input
    seed_ln_level: int = 1471
    base_mode: str | None = None         # phase C: 'frozen' | 'kl'
    natural_ce: bool | None = None       # phase C, kl: L_base also scores natural decisions
    kl_weight: float | None = None       # phase C, kl: weight of L_kl
    kl_direction: str | None = None      # phase C, kl: 'forward' KL(reference || model) | 'reverse'
    kl_decisions: str | None = None      # phase C, kl: 'natural' (V_k empty) | 'all'
    device: str = 'cpu'
    threads: int = 4
    dtype: str = 'float32'
    conditioner: str = 'film'
    memory: str = 'landmarks'
    levels: int = 8
    hidden: int = 128
    expansion: int = 4
    rank: int = 16
    max_parameters: int = MAX_PARAMETERS
    film_width: int = 128                # conditioning path capacity (FiLM MLP width and hidden layers)
    film_layers: int = 1
    presence: str = 'none'
    rule_l: bool = True                  # False only for pilot overhead measurement and power checks
    star_value: str = 'absolute'         # 'residual': difficulty frame value is target - b(S)
    star_conditions: str = 'auto'        # auto: on iff the v2 label file is complete
    draw: dict = field(default_factory=dict)   # DrawConfig fields; empty means the DrawConfig defaults
    lambda_ln: float | None = None       # phase C
    lambda_star: float | None = None     # phase C
    mu_star: float | None = None         # phase C: relaxed-proxy term (residual difficulty only); 0 is off
    proxy_every: int = 4
    f3_every: int = 16
    f3_samples: int = 4
    n_bar: float | None = None           # from draw_sim, with the matching n_bar_key
    n_bar_ln: float | None = None
    n_bar_star: float | None = None
    n_bar_key: str | None = None
    total_exposures: int | None = None   # head decisions of drawn windows, per phase
    warmup_exposures: int | None = None
    lr: float | None = None
    lr_min: float | None = None
    lr_schedule: list | None = None
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.01
    clip: float = 1.0
    accumulate: int = 4
    window: int = 256
    checkpoint_every: int | None = None
    full_eval_every: int = 2             # full evaluation at every n-th checkpoint, teacher-forced at the others
    g3c_exposures: list | None = None    # G3 (c) at the first full evaluation past each; [] for none
    log_every: int = 2_000
    seed_weights: int = 171
    seed_draws: int = 471
    seed_validation: int = 954
    freerun: bool = True
    freerun_seeds: list = field(default_factory=lambda: [954, 955, 956])
    max_nan_events: int = 3
    stop_at_unix: float = 0.0
    rss_limit_gib: float = 12.0
    rss_growth_limit_gib: float = 2.0    # trainer RSS growth between checkpoints
    mps_limit_gib: float = 8.0

    @classmethod
    def load(cls, path, **overrides):
        data = json.loads(Path(path).read_text()) if path else {}
        data.update({k: v for k, v in overrides.items() if v is not None})
        names = {f.name for f in fields(cls)}
        unknown = set(data) - names
        if unknown:
            raise ContractError(f'Unknown config keys: {sorted(unknown)}')
        cfg = cls(**data)
        validate_schedule(cfg.lr_schedule)
        return cfg


def check_config(cfg: TrainConfig):
    """Refuse a configuration whose undecided values are still null (plan v5).

    Every run needs ``phase``, the budget (``total_exposures``, ``checkpoint_every``,
    ``g3c_exposures``) and a learning rate: ``lr_schedule``, or ``lr``, ``lr_min`` and
    ``warmup_exposures`` for the v1 warm-up plus cosine. Phase N draws no conditions and leaves
    every phase-C key null. Phase C needs ``init_from``, ``base_mode``, ``lambda_ln``,
    ``lambda_star`` and ``mu_star``; with ``base_mode='kl'`` also ``kl_weight``, ``kl_direction``,
    ``kl_decisions`` and ``natural_ce``, which stay null under ``'frozen'``.
    """
    if cfg.phase not in PHASES:
        raise ContractError(f'phase must be one of {PHASES} (got {cfg.phase!r})')
    if cfg.ln_level not in LN_LEVEL_MODES:
        raise ContractError(f'ln_level must be one of {LN_LEVEL_MODES}')
    if cfg.ln_length not in LN_LEVEL_MODES or (cfg.ln_length == 'on' and cfg.ln_level != 'on'):
        raise ContractError('ln_length must be off|on and requires ln_level on')
    if (isinstance(cfg.ln_level_dropout, bool) or not isinstance(cfg.ln_level_dropout, (int, float))
            or not math.isfinite(cfg.ln_level_dropout) or not 0.0 <= cfg.ln_level_dropout <= 1.0):
        raise ContractError('ln_level_dropout must be a finite number in [0,1]')
    if isinstance(cfg.seed_ln_level, bool) or not isinstance(cfg.seed_ln_level, int) or cfg.seed_ln_level < 0:
        raise ContractError('seed_ln_level must be a nonnegative integer')
    if cfg.warm_start is not None and (not isinstance(cfg.warm_start, str) or not cfg.warm_start):
        raise ContractError('warm_start must be a checkpoint path or null')
    if cfg.phase != 'natural' and (cfg.warm_start is not None or cfg.ln_level != 'off'):
        raise ContractError('warm_start and ln_level on apply only to phase natural')
    missing, inapplicable = [], []

    def need(*keys):
        missing.extend(k for k in keys if getattr(cfg, k) is None)

    def unset(*keys):
        inapplicable.extend(k for k in keys if getattr(cfg, k) is not None)
    need(*BUDGET_KEYS)
    if cfg.lr_schedule is None:
        need(*COSINE_KEYS)
    selection = draw_config(cfg).selection
    if cfg.phase == 'natural':
        unset(*PHASE_C_KEYS, *KL_KEYS)
        if selection != 'natural' or cfg.star_conditions != 'off':
            raise ContractError("phase 'natural' draws no conditions: draw.selection 'natural', star_conditions 'off'")
    else:
        need(*PHASE_C_KEYS)
        if selection == 'natural':
            raise ContractError("phase 'conditions' needs a condition draw, not draw.selection 'natural'")
        if cfg.base_mode is not None and cfg.base_mode not in BASE_MODES:
            raise ContractError(f'base_mode must be one of {BASE_MODES}')
        if cfg.base_mode == 'kl':
            need(*KL_KEYS)
            if cfg.kl_direction is not None and cfg.kl_direction not in KL_DIRECTIONS:
                raise ContractError(f'kl_direction must be one of {KL_DIRECTIONS}')
            if cfg.kl_decisions is not None and cfg.kl_decisions not in KL_DECISIONS:
                raise ContractError(f'kl_decisions must be one of {KL_DECISIONS}')
        else:
            unset(*KL_KEYS)
    if missing or inapplicable:
        raise ContractError(f'phase {cfg.phase!r}: undecided keys still null {missing}; keys that do not apply to '
                            f'this phase and must stay null {inapplicable}')


def trainable_parameters(model: R2Model, cfg: TrainConfig):
    """[(name, parameter)] the phase trains, from ``R2Model.parameter_split``: phase N the natural
    model; phase C with a frozen base the conditioning path only; phase C with a KL-held base both.
    Every other parameter gets ``requires_grad=False``; the optimiser holds exactly the returned list."""
    conditioning, natural = model.parameter_split()
    if cfg.phase == 'natural':
        chosen = natural
    elif cfg.base_mode == 'frozen':
        chosen = conditioning
    else:
        chosen = natural + conditioning
    ids = {id(p) for _, p in chosen}
    for p in model.parameters():
        p.requires_grad_(id(p) in ids)
    return chosen


def make_optimizer(named, cfg: TrainConfig):
    """AdamW over exactly ``named``; weight decay on matrices only."""
    decay = [p for _, p in named if p.ndim >= 2]
    other = [p for _, p in named if p.ndim < 2]
    return torch.optim.AdamW([dict(params=decay, weight_decay=cfg.weight_decay), dict(params=other, weight_decay=0.0)],
                             lr=cfg.lr if cfg.lr is not None else 0.0, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_phase_n(path, model: R2Model):
    """Load a phase-N checkpoint's natural parameters into ``model`` and return (init record, checkpoint).

    The conditioning path is not loaded: it keeps its own initialisation (zero output layer), so it
    may be sized independently of phase N and the model starts as the phase-N model exactly.
    """
    from .operating_point import recipe_hash
    data = torch.load(path, map_location='cpu', weights_only=False)
    if (data.get('config') or {}).get('phase') != 'natural':
        raise ContractError(f'init_from must be a phase-N checkpoint (config phase "natural"): {path}')
    theirs = {k: data['model_config'].get(k) for k in NATURAL_ARCH}
    mine = {k: getattr(model.config, k) for k in NATURAL_ARCH}
    if theirs != mine:
        raise ContractError(f'init_from has another natural architecture: {theirs} against {mine}')
    natural = {k: v for k, v in data['model'].items() if k.split('.', 1)[0] not in CONDITIONERS}
    result = model.load_state_dict(natural, strict=False)
    wrong = [k for k in result.missing_keys if k.split('.', 1)[0] not in CONDITIONERS] + list(result.unexpected_keys)
    if wrong:
        raise ContractError(f'init_from does not match the natural parameters: {wrong}')
    record = dict(path=str(path), sha256=file_sha256(path), exposures=int(data['state']['exposures']),
                  recipe_hash=recipe_hash(data['config']), model_config=data['model_config'])
    return record, data


def reference_model(data, device, dtype) -> R2Model:
    """The frozen phase-N model of the KL term, exactly as it was trained (it never reads a condition)."""
    ref = R2Model(R2Config(**data['model_config']))
    ref.load_state_dict(data['model'])
    ref = ref.to(device, dtype).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def load_warm_start(path, model: R2Model):
    """Load every source weight; only explicitly enabled new zero LN readers may be absent.

    Both runs must be phase N. Model configuration must match, except that an off
    source may initialize an on target. Optimizer, counters and RNGs are not loaded.
    """
    from .operating_point import recipe_hash
    data = torch.load(path, map_location='cpu', weights_only=False)
    if (data.get('config') or {}).get('phase') != 'natural':
        raise ContractError(f'warm_start must be a phase-N checkpoint: {path}')
    try:
        source = R2Config(**data['model_config'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError('warm_start has an invalid model_config') from exc
    theirs, mine = asdict(source), asdict(model.config)
    allowed = set()
    for name in ('ln_level', 'ln_length'):
        if theirs[name] == 'off' and mine[name] == 'on':
            theirs[name] = 'on'
            allowed.add(f'{name}_reader.weight')
    if theirs != mine:
        mismatch = {k: (theirs[k], mine[k]) for k in mine if theirs[k] != mine[k]}
        raise ContractError(f'warm_start model configuration mismatch: {mismatch}')
    source_state = data['model']
    target_state = model.state_dict()
    missing = set(target_state) - set(source_state)
    unexpected = set(source_state) - set(target_state)
    if missing != allowed or unexpected:
        raise ContractError(f'warm_start parameter keys mismatch: missing {sorted(missing)}, '
                            f'allowed missing {sorted(allowed)}, unexpected {sorted(unexpected)}')
    shapes = {k: (tuple(v.shape), tuple(target_state[k].shape)) for k, v in source_state.items()
              if v.shape != target_state[k].shape}
    if shapes:
        raise ContractError(f'warm_start parameter shape mismatch: {shapes}')
    if any(torch.count_nonzero(target_state[name]).item() for name in allowed):
        raise ContractError('warm_start requires every new LN reader to be exactly zero')
    model.load_state_dict({**target_state, **source_state}, strict=True)
    record = dict(path=str(path), sha256=file_sha256(path), exposures=int(data['state']['exposures']),
                  recipe_hash=recipe_hash(data['config']), model_config=data['model_config'],
                  missing_keys=sorted(allowed), loaded_parameters=len(source_state))
    return record


class NonFinite(Exception):
    def __init__(self, indices, what):
        super().__init__(f'non-finite {what} in windows {indices}')
        self.indices, self.what = list(indices), what


def validate_schedule(phases):
    if phases is None:
        return
    if not isinstance(phases, list) or not phases:
        raise ContractError('lr_schedule is a non-empty list of phases')
    last = 0
    for ph in phases:
        unknown = set(ph) - {'until', 'shape', 'lr_start', 'lr_end'}
        if unknown:
            raise ContractError(f'Unknown lr_schedule keys {sorted(unknown)}')
        if not int(ph['until']) > last:
            raise ContractError('lr_schedule phases end at strictly increasing exposures')
        last = int(ph['until'])
        if ph.get('shape', 'constant') not in LR_SHAPES:
            raise ContractError(f'lr_schedule shape must be one of {LR_SHAPES}')
        for key in ('lr_start', 'lr_end'):
            if key in ph and not float(ph[key]) > 0:
                raise ContractError('lr_schedule rates are positive')
        if 'lr_start' not in ph:
            raise ContractError('every lr_schedule phase names lr_start')


def schedule_lr(phases, exposures: int) -> float:
    lo = 0
    for i, ph in enumerate(phases):
        hi = int(ph['until'])
        a, b = float(ph['lr_start']), float(ph.get('lr_end', ph['lr_start']))
        if exposures < hi or i == len(phases) - 1:
            if exposures >= hi:
                return b
            x = (exposures - lo) / (hi - lo)
            shape = ph.get('shape', 'constant')
            if shape == 'constant':
                return a
            if shape == 'linear':
                return a + (b - a) * x
            return b + (a - b) * 0.5 * (1 + math.cos(math.pi * x))
        lo = hi
    raise AssertionError


def lr_at(cfg: TrainConfig, exposures: int) -> float:
    if cfg.lr_schedule:
        return schedule_lr(cfg.lr_schedule, exposures)
    if exposures < cfg.warmup_exposures:
        return cfg.lr * max(exposures, 1) / cfg.warmup_exposures
    span = max(1, cfg.total_exposures - cfg.warmup_exposures)
    progress = min(1.0, (exposures - cfg.warmup_exposures) / span)
    return cfg.lr_min + (cfg.lr - cfg.lr_min) * 0.5 * (1 + math.cos(math.pi * progress))


def cache_hashes(cache_root: Path):
    summary = json.loads((Path(cache_root) / 'summary.json').read_text())
    out = dict(index_sha256=summary['index_sha256'], splits_sha256=summary['splits_sha256'],
               version=summary['version'], candidates=summary['candidates'])
    star = Path(cache_root) / 'labels' / 'star-v2_summary.json'
    if star.exists():
        out['star_labels'] = json.loads(star.read_text())
    return out


def mps_bytes(device):
    if torch.device(device).type != 'mps':
        return 0, 0
    return int(torch.mps.driver_allocated_memory()), int(torch.mps.current_allocated_memory())


def draw_config(cfg: TrainConfig) -> DrawConfig:
    data = dict(cfg.draw)
    data.setdefault('star_value', cfg.star_value)
    if data['star_value'] != cfg.star_value:
        raise ContractError('draw.star_value and star_value differ')
    return DrawConfig.from_dict(data)


def star_setting(cfg: TrainConfig):
    summary = Path(cfg.cache) / 'labels' / 'star-v2_summary.json'
    complete = bool(summary.exists() and json.loads(summary.read_text()).get('complete'))
    if cfg.star_conditions == 'on' and not complete:
        raise ContractError("star_conditions 'on' needs the complete v2 label file (labels/star-v2.json.gz)")
    return {'auto': complete, 'on': True, 'off': False}[cfg.star_conditions], complete


def build_corpus(cfg: TrainConfig, role: str):
    from .baseline import load_baseline
    star, _ = star_setting(cfg)
    dcfg = draw_config(cfg)
    baseline = load_baseline(cfg.cache) if cfg.star_value == 'residual' else None
    return Corpus(cfg.cache, role, star_conditions=star, draw=dcfg, baseline=baseline), star, dcfg


def model_config(cfg: TrainConfig) -> R2Config:
    return R2Config(memory=cfg.memory, levels=cfg.levels, conditioner=cfg.conditioner, hidden=cfg.hidden,
                    expansion=cfg.expansion, rank=cfg.rank, max_parameters=cfg.max_parameters,
                    presence=cfg.presence, rule_l=cfg.rule_l, star_value=cfg.star_value,
                    film_width=cfg.film_width, film_layers=cfg.film_layers, ln_level=cfg.ln_level,
                    ln_length=cfg.ln_length)


class Stats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.d = dict(action=0.0, release=0.0, heads=0, gap_lns=0, eos=0.0, eos_n=0, natural=0.0, natural_n=0,
                      conditioned=0.0, conditioned_n=0, windows=0, loss_base=0.0, loss_ln=0.0, loss_star=0.0,
                      loss_kl=0.0,
                      ln_governed=0.0, ln_whole=0.0, ln_n=0, star_whole=0.0, star_n=0, weights=0.0,
                      factors_ln=0, factors_star=0, proxy=0.0, proxy_n=0)

    def add(self, out, terms=None, parts=None):
        d = self.d
        act = out.action.detach().cpu().double().numpy()
        rel = out.release.detach().cpu().double().numpy()
        gov = out.governed_ln.detach().cpu().double().numpy()
        head = ~out.eos
        natural = ~out.visible.any(1)
        d['action'] -= float(act[head].sum())
        d['release'] -= float(rel[head].sum())
        d['heads'] += int(head.sum())
        d['gap_lns'] += int(out.gap_lns[head].sum())
        d['eos'] -= float((act + rel)[out.eos].sum())
        d['eos_n'] += int(out.eos.sum())
        d['natural'] -= float((act + rel)[natural].sum())
        d['natural_n'] += int(natural.sum())
        d['conditioned'] -= float((act + rel)[~natural].sum())
        d['conditioned_n'] += int((~natural).sum())
        ln, star = out.visible[:, 0], out.visible[:, 1]
        d['ln_governed'] -= float(gov[ln].sum())
        d['ln_whole'] -= float((act + rel)[ln].sum())
        d['ln_n'] += int(ln.sum())
        d['star_whole'] -= float((act + rel)[star].sum())
        d['star_n'] += int(star.sum())
        d['windows'] += 1
        if terms is not None:
            d['weights'] += terms['weights']
            d['factors_ln'] += terms['factors_ln']
            d['factors_star'] += terms['factors_star']
        if parts is not None:
            d['loss_base'] += float(parts['base'].detach())
            d['loss_ln'] += float(parts['ln'].detach())
            d['loss_star'] += float(parts['star'].detach())
            if 'kl' in parts:
                d['loss_kl'] += float(parts['kl'].detach())

    def summary(self, accumulate=1):
        d = self.d

        def div(a, b):
            return a / b if b else None

        batches = d['windows'] / max(1, accumulate)
        return dict(action_nll_per_decision=div(d['action'], d['heads']),
                    release_nll_per_decision=div(d['release'], d['heads']),
                    release_nll_per_gap_ln=div(d['release'], d['gap_lns']),
                    eos_nll=div(d['eos'], d['eos_n']), eos_count=d['eos_n'],
                    natural_nll_per_decision=div(d['natural'], d['natural_n']), natural_decisions=d['natural_n'],
                    conditioned_nll_per_decision=div(d['conditioned'], d['conditioned_n']),
                    conditioned_decisions=d['conditioned_n'], head_decisions=d['heads'], gap_lns=d['gap_lns'],
                    ln_governed_nll_per_decision=div(d['ln_governed'], d['ln_n']),
                    ln_whole_nll_per_decision=div(d['ln_whole'], d['ln_n']), ln_decisions=d['ln_n'],
                    star_whole_nll_per_decision=div(d['star_whole'], d['star_n']), star_decisions=d['star_n'],
                    loss_base=div(d['loss_base'], batches), loss_ln=div(d['loss_ln'], batches),
                    loss_star=div(d['loss_star'], batches), loss_kl=div(d['loss_kl'], batches),
                    weights_per_batch=div(d['weights'], batches),
                    factors_ln_per_batch=div(d['factors_ln'], batches),
                    factors_star_per_batch=div(d['factors_star'], batches),
                    proxy_loss=div(d['proxy'], d['proxy_n']), proxy_terms=d['proxy_n'], windows=d['windows'])


class Trainer:
    def __init__(self, cfg: TrainConfig, *, write=True):
        check_config(cfg)
        self.cfg = cfg
        self.write = write
        self.run = Path(cfg.run_dir)
        torch.set_num_threads(cfg.threads)
        self.device = torch.device(cfg.device)
        self.dtype = getattr(torch, cfg.dtype)
        self.corpus, self.star, self.draw_cfg = build_corpus(cfg, 'fit_train')
        _, self.star_complete = star_setting(cfg)
        self.baseline = self.corpus.baseline
        self.key = n_bar_key(self.draw_cfg, star_conditions=self.star, window=cfg.window, accumulate=cfg.accumulate,
                             label_sha256=self.corpus.label_sha256)
        if cfg.n_bar_key != self.key or cfg.n_bar is None:
            raise ContractError(f'n_bar values are missing or were measured for another draw (config key '
                                f'{cfg.n_bar_key}, this draw {self.key}); run draw_sim on this config')
        phase_c = cfg.phase == 'conditions'
        if phase_c and (not cfg.n_bar_ln or (self.star and not cfg.n_bar_star)):
            raise ContractError('Phase C needs n_bar_ln, and n_bar_star with star conditions on, from draw_sim')
        self.loss_cfg = LossConfig(cfg.n_bar, cfg.n_bar_ln or math.inf, cfg.n_bar_star or math.inf,
                                   cfg.lambda_ln or 0.0, cfg.lambda_star or 0.0,
                                   natural_ce=not phase_c or bool(cfg.natural_ce), kl_weight=cfg.kl_weight or 0.0)
        if cfg.mu_star and (cfg.star_value != 'residual' or not self.star):
            raise ContractError('The relaxed-proxy term needs residual difficulty conditioning with star labels')
        torch.manual_seed(cfg.seed_weights)
        self.model = R2Model(model_config(cfg), verbose=write)
        self.init, self.reference = None, None
        self.warm_start = load_warm_start(cfg.warm_start, self.model) if cfg.warm_start is not None else None
        if phase_c:
            self.init, data = load_phase_n(cfg.init_from, self.model)
            if cfg.base_mode == 'kl':
                self.reference = reference_model(data, self.device, self.dtype)
        self.model = self.model.to(self.device, self.dtype)
        self.trainable = trainable_parameters(self.model, cfg)
        self.opt = make_optimizer(self.trainable, cfg)
        self.rng = np.random.default_rng(cfg.seed_draws)
        self.ln_level_rng = np.random.default_rng(cfg.seed_ln_level) if cfg.ln_level == 'on' else None
        self.state = dict(exposures=0, windows=0, steps=0, lr_mult=1.0, nan_events=0, skip=[],
                          wall_s=0.0, restarts=0, checkpoints=0)
        self.stats = Stats()
        self.guard = None
        self.rss_mark = None
        self.segment = None

    # ---- persistence -----------------------------------------------------------------------------

    @property
    def ckpt_dir(self):
        return self.run / 'checkpoints'

    def append(self, name, record):
        if not self.write:
            return
        if name in ('train.jsonl', 'resources.jsonl'):
            (self.run / 'logs').mkdir(parents=True, exist_ok=True)
            seg = self.segment if self.segment is not None else self.state['exposures']
            path = self.run / 'logs' / f'{name[:-6]}-{seg:010d}.jsonl'
        else:
            path = self.run / name
        with path.open('a') as f:
            f.write(json.dumps(record, allow_nan=True) + '\n')

    def payload(self):
        return dict(model=self.model.state_dict(), optimizer=self.opt.state_dict(),
                    draw_rng=self.rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                    ln_level_rng=self.ln_level_rng.bit_generator.state if self.ln_level_rng is not None else None,
                    state=copy.deepcopy(self.state), config=asdict(self.cfg),
                    model_config=asdict(self.model.config), cache=cache_hashes(Path(self.cfg.cache)),
                    star_conditions=self.star, draw_hash=self.draw_cfg.hash(), n_bar_key=self.key, nu_hash=NU_HASH,
                    phase=self.cfg.phase, base_mode=self.cfg.base_mode, init=self.init, warm_start=self.warm_start)

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
        if (data.get('phase'), data.get('base_mode')) != (self.cfg.phase, self.cfg.base_mode):
            raise ContractError('Checkpoint belongs to another phase or base mode')
        if (data.get('init') or {}).get('sha256') != (self.init or {}).get('sha256'):
            raise ContractError('Checkpoint was initialised from another phase-N checkpoint')
        if (data.get('warm_start') or {}).get('sha256') != (self.warm_start or {}).get('sha256'):
            raise ContractError('Checkpoint belongs to another warm_start')
        saved_cfg = data.get('config') or {}
        if saved_cfg.get('ln_level', 'off') != self.cfg.ln_level:
            raise ContractError('Checkpoint has another ln_level setting')
        if saved_cfg.get('ln_length', 'off') != self.cfg.ln_length:
            raise ContractError('Checkpoint has another ln_length setting')
        if self.ln_level_rng is not None:
            if any(saved_cfg.get(k) != getattr(self.cfg, k) for k in ('ln_level_dropout', 'seed_ln_level')):
                raise ContractError('Checkpoint has another LN-level dropout configuration')
            if data.get('ln_level_rng') is None:
                raise ContractError('Checkpoint is missing the LN-level dropout RNG state')
        if data.get('star_conditions', self.star) != self.star:
            raise ContractError('Checkpoint was trained with a different star_conditions setting')
        if data.get('n_bar_key', self.key) != self.key:
            raise ContractError('Checkpoint was trained with a different draw')
        self.model.load_state_dict(data['model'])
        self.opt.load_state_dict(data['optimizer'])
        self.rng.bit_generator.state = data['draw_rng']
        if self.ln_level_rng is not None:
            self.ln_level_rng.bit_generator.state = data['ln_level_rng']
        torch.set_rng_state(data['torch_rng'])
        self.state = data['state']
        self.state.setdefault('checkpoints', 0)
        return data

    # ---- one optimizer step ----------------------------------------------------------------------

    def next_batch(self):
        batch = []
        while len(batch) < self.cfg.accumulate:
            index = self.state['windows']
            draw = self.corpus.draw(self.rng, self.cfg.window)
            if self.ln_level_rng is not None:
                dropped = self.ln_level_rng.random() < self.cfg.ln_level_dropout
                draw.ln_level = None if dropped else self.corpus.source_ln_level(draw.sha)
                if self.cfg.ln_length == 'on':
                    draw.ln_length = None if dropped else self.corpus.source_ln_length(draw.sha)
            self.state['windows'] += 1
            if index in self.state['skip']:
                continue
            batch.append((index, draw))
        return batch

    def step(self, batch, stats=None, probe=None):
        cfg = self.cfg
        lr = lr_at(cfg, self.state['exposures']) * self.state['lr_mult']
        for group in self.opt.param_groups:
            group['lr'] = lr
        self.opt.zero_grad(set_to_none=True)
        heads, losses = 0, []
        self.model.train()
        proc = psutil.Process() if probe is not None else None
        for index, draw in batch:
            chart = self.corpus.chart(draw.sha)
            level = dict(ln_level=draw.ln_level) if cfg.ln_level == 'on' else {}
            if cfg.ln_length == 'on':
                level['ln_length'] = draw.ln_length
            out = self.model.window(chart, draw.start, draw.stop, draw.track,
                                    pairs=self.reference is not None, **level)
            terms = window_terms(out, draw.weight)
            kl = None
            if self.reference is not None:
                with torch.no_grad():
                    ref = self.reference.window(chart, draw.start, draw.stop, (), pairs=True)
                kl = window_kl(out, ref, draw.weight, cfg.kl_decisions, cfg.kl_direction)
            loss, parts = window_loss(terms, self.loss_cfg, kl)
            if cfg.mu_star and index % cfg.proxy_every == 0:
                own = self.relaxed_proxy(chart, draw)
                if own is not None:
                    loss = loss + cfg.lambda_star * cfg.mu_star * own
                    if stats is not None:
                        stats.d['proxy'] += float(own.detach())
                        stats.d['proxy_n'] += 1
            if not torch.isfinite(loss):
                raise NonFinite([index], 'loss')
            if loss.requires_grad:      # frozen base: a window with no decision in V reaches no trainable parameter
                loss.backward()
            losses.append(loss.item())
            heads += int((~out.eos).sum())
            if stats is not None:
                stats.add(out, terms, parts)
            if probe is not None:
                probe.append(dict(window=index, decisions=len(out.ks), factors=int(out.factors.sum()),
                                  candidate_pairs=int(out.candidate_pairs), rss_bytes=proc.memory_info().rss))
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in self.trainable], cfg.clip)
        if not torch.isfinite(norm):
            raise NonFinite([i for i, _ in batch], 'gradient')
        self.opt.step()
        self.state['steps'] += 1
        self.state['exposures'] += heads
        info = dict(lr=lr, heads=heads, loss=float(np.sum(losses)), grad_norm=float(norm))
        if cfg.mu_star and cfg.f3_every and self.state['steps'] % cfg.f3_every == 0:
            info['f3'] = self.f3_calibration(batch)
        return info

    # ---- stage-2 relaxed-proxy term ----------------------------------------------------------------

    def proxy_target(self, chart, draw):
        """The first difficulty interval of the window that some scored decision reads, or None."""
        from .locality import visible
        for k in range(draw.start, draw.stop):
            for iv in visible(chart, draw.track, k, self.cfg.rule_l):
                if iv.kind == 1:
                    return iv
        return None

    def relaxed_proxy(self, chart, draw):
        """(g(E_theta[proxies over S | own history]) - v_res)^2 with S sampled once from the real prefix.

        Gradient flows through the action probabilities of decisions that read S (rule L);
        decisions of S that do not read it enter with their sampled values and no gradient.
        """
        from .proxy import expected_proxies
        iv = self.proxy_target(chart, draw)
        if iv is None:
            return None
        k0 = int(np.searchsorted(chart.head_ms, iv.a, side='left'))
        k1 = chart.K if iv.b >= chart.song_ms else int(np.searchsorted(chart.head_ms, iv.b, side='left'))
        if k1 <= k0:
            return None
        seed = int(self.rng.integers(0, 2 ** 31 - 1))
        acts, gap = continue_chart(self.model, chart.head_ms, chart.song_ms, chart.grid, chart.actions[:k0],
                                   chart.gap[:k0], track=draw.track, seed=seed, stop=k1)
        own = chart.with_decisions(acts, gap)
        out = self.model.window(own, k0, k1, draw.track)
        proxies = expected_proxies(own, out, iv)
        return (self.baseline.proxy_residual(proxies) - iv.value) ** 2

    @torch.no_grad()
    def f3_calibration(self, batch):
        """True F3 as a measurement: k samples of a window's difficulty scope, run until its last
        scope-headed LN is released; realised residual against g of the realised proxies."""
        from .features import Interval, star_proxies
        from .properties import difficulty, prefix_objects
        for index, draw in batch:
            chart = self.corpus.chart(draw.sha)
            iv = self.proxy_target(chart, draw)
            if iv is None:
                continue
            k0 = int(np.searchsorted(chart.head_ms, iv.a, side='left'))
            b_s = float(self.baseline.predict(chart.head_ms, iv.a, iv.b))
            rows = []
            for _ in range(self.cfg.f3_samples):
                seed = int(self.rng.integers(0, 2 ** 31 - 1))
                acts, gap = continue_chart(self.model, chart.head_ms, chart.song_ms, chart.grid, chart.actions[:k0],
                                           chart.gap[:k0], track=draw.track, seed=seed, close_scope=(iv.a, iv.b))
                objects = prefix_objects(chart.head_ms, acts, gap)
                value, info = difficulty(objects, iv.a, iv.b, chart.song_ms)
                own = chart.with_decisions(acts, gap)
                proxies = np.array(star_proxies(own, Interval(1, iv.a, iv.b, iv.value), len(acts)))
                rows.append(dict(seed=seed, realised=value, realised_residual=None if value is None else value - b_s,
                                 g_realised=float(self.baseline.proxy_residual(proxies)), invalid=info.get('invalid')))
            ok = [r for r in rows if r['realised_residual'] is not None]
            record = dict(event='f3', exposures=self.state['exposures'], window=index, a=iv.a, b=iv.b,
                          v_res=iv.value, b_s=b_s, samples=rows,
                          surrogate_gap=float(np.mean([r['g_realised'] - r['realised_residual'] for r in ok]))
                          if ok else None, time=time.time())
            self.append('f3.jsonl', record)
            return record
        return None

    # ---- evaluation ------------------------------------------------------------------------------

    def manifests(self):
        return load_or_build_manifests(self.cfg.cache, self.run / 'manifests.json', star_conditions=self.star,
                                       draw=self.draw_cfg, baseline=self.baseline)

    @torch.no_grad()
    def evaluate(self, full=True):
        from .evaluate import Evaluator
        done = self.state.setdefault('g3c_done', [])
        due = [x for x in self.cfg.g3c_exposures if x not in done and self.state['exposures'] >= x]
        g3c = bool(full and self.cfg.freerun and due)
        ev = Evaluator(self.model, self.cfg, self.manifests(), star=self.star, baseline=self.baseline,
                       write_dir=(self.run / 'freerun' / f'{self.state["exposures"]:010d}') if self.write else None,
                       g3c=g3c, conditions=self.cfg.phase == 'conditions')
        record = dict(exposures=self.state['exposures'], **ev.teacher_forced())
        if full and self.cfg.freerun:
            record.update(ev.panels())
            if g3c:
                done += due
        self.model.train()
        return record

    def freerun(self, manifest, dev, seeds=None):
        """Natural free runs from BOS on the manifest's short fit_dev charts (smoke and freerun-only mode)."""
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
        from .operating_point import recipe_hash
        conditioning, _ = self.model.parameter_split()
        return dict(entry_point=' '.join(sys.argv), mode=mode, config=asdict(self.cfg), code=code_identity(),
                    phase=self.cfg.phase, base_mode=self.cfg.base_mode, init=self.init, warm_start=self.warm_start,
                    trainable=dict(parameters=sum(p.numel() for _, p in self.trainable),
                                   modules=sorted({n.split('.', 1)[0] for n, _ in self.trainable}),
                                   conditioning_parameters=sum(p.numel() for _, p in conditioning)),
                    cache=cache_hashes(Path(self.cfg.cache)), star_conditions=self.star,
                    star_labels_complete=self.star_complete, seeds=dict(weights=self.cfg.seed_weights,
                                                                        draws=self.cfg.seed_draws,
                                                                        validation=self.cfg.seed_validation),
                    ln_level=dict(mode=self.cfg.ln_level, dropout=self.cfg.ln_level_dropout,
                                  seed=self.cfg.seed_ln_level, length=self.cfg.ln_length),
                    conditioning=dict(conditioner=self.cfg.conditioner, presence=self.cfg.presence,
                                      rule_l=RULE_L_VERSION if self.cfg.rule_l else 'off', eta='default',
                                      star_value=self.cfg.star_value, birth_role=False),
                    loss=dict(lambda_ln=self.cfg.lambda_ln, lambda_star=self.cfg.lambda_star, mu_ln=0.0,
                              mu_star=self.cfg.mu_star, ipw=self.draw_cfg.ipw, n_bar=self.cfg.n_bar,
                              n_bar_ln=self.cfg.n_bar_ln, n_bar_star=self.cfg.n_bar_star, n_bar_key=self.key,
                              natural_ce=self.loss_cfg.natural_ce, kl_weight=self.cfg.kl_weight,
                              kl_direction=self.cfg.kl_direction, kl_decisions=self.cfg.kl_decisions),
                    draw=self.draw_cfg.to_dict(), draw_hash=self.draw_cfg.hash(), nu_hash=NU_HASH,
                    properties_sha256=PROPERTIES_SOURCE_SHA256, labels_sha256=self.corpus.label_sha256,
                    baseline_sha256=self.baseline.sha256 if self.baseline is not None else None,
                    recipe_hash=recipe_hash(asdict(self.cfg)),
                    manifests={k: manifest_version(k, self.draw_cfg, self.corpus.label_sha256, self.star)
                               for k in ('natural', 'condition')},
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

    def check_resources(self):
        """The guard's RSS, driver and available-memory limits plus the trainer's own RSS growth since
        the last checkpoint; writes one compact resources record per check."""
        res = None
        try:
            res = self.guard.check('log', exposures=self.state['exposures'])
            grown = res['rss_bytes'] - (self.rss_mark if self.rss_mark is not None else res['rss_bytes'])
            if grown > self.cfg.rss_growth_limit_gib * GiB:
                raise ResourceLimit(f'trainer RSS grew {grown} bytes since the last checkpoint, above '
                                    f'{self.cfg.rss_growth_limit_gib} GiB')
        finally:
            value = res if res is not None else snapshot(self.cfg.device)
            self.append('resources.jsonl', dict(t=round(value['time'], 1), x=self.state['exposures'],
                                                rss=value['rss_bytes'], avail=value['available_bytes'],
                                                swap=value['swap_bytes'], p=value['pressure_level']))
        return res

    def train(self, resume: bool):
        cfg = self.cfg
        self.run.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(exist_ok=True)
        if resume and self.latest() is not None:
            latest = self.latest()
            self.load(latest)
            self.segment = self.state['exposures']
            self.state['restarts'] = self.state.get('restarts', 0) + 1
            self.append('events.jsonl', dict(event='resume', checkpoint=latest.name, time=time.time(),
                                             **self.state_brief()))
            if self.state['exposures'] > 0 and latest.name not in self.evaluated():
                count = self.state['checkpoints']
                full = count > 0 and count % max(1, cfg.full_eval_every) == 0
                self.eval_and_log(latest, full=full)
        else:
            if self.latest() is not None:
                raise ContractError('Run directory already has checkpoints; pass --resume')
            (self.run / 'config.json').write_text(json.dumps(asdict(cfg), indent=1))
            self.segment = 0
            self.save()
            self.append('events.jsonl', dict(event='start', time=time.time(), star_conditions=self.star))
        (self.run / f'receipt-{int(time.time())}.json').write_text(json.dumps(self.receipt('train'), indent=1))
        self.guard = ResourceGuard(cfg.device, ResourceConfig(
            driver_limit_bytes=int(min(cfg.mps_limit_gib, 8) * GiB), rss_limit_bytes=int(cfg.rss_limit_gib * GiB),
            allocator_ceiling_bytes=8 * GiB, output_max_bytes=2 * GiB, max_swap_growth_bytes=NO_SWAP_TRIP), log=None)
        self.rss_mark = psutil.Process().memory_info().rss
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
                    res = self.check_resources()
                except ResourceLimit as exc:
                    snap = snapshot(cfg.device)
                    self.append('events.jsonl', dict(event='resource_limit', error=str(exc), time=time.time(),
                                                     rss_bytes=snap['rss_bytes'], available_bytes=snap['available_bytes'],
                                                     pressure_level=snap['pressure_level'], rss_mark=self.rss_mark,
                                                     **self.state_brief()))
                    self.save('safe')
                    self.write_run_json('resource_limit', t_start, wall0)
                    return EXIT_RESOURCE
                drv, cur = mps_bytes(cfg.device)
                record = dict(exposures=self.state['exposures'], steps=self.state['steps'],
                              windows_drawn=self.state['windows'], lr=info['lr'], grad_norm=info['grad_norm'],
                              decisions_per_s=(self.state['exposures'] - heads_last) / max(1e-9, now - t_last),
                              rss_gib=res['rss_bytes'] / GiB, mps_driver_gib=drv / GiB, mps_current_gib=cur / GiB,
                              wall_s=self.state['wall_s'], time=now, **self.stats.summary(cfg.accumulate))
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
        self.state['checkpoints'] = self.state.get('checkpoints', 0) + 1
        path, info = self.save()
        self.segment = self.state['exposures']
        self.rss_mark = psutil.Process().memory_info().rss
        full = self.state['checkpoints'] % max(1, self.cfg.full_eval_every) == 0
        self.eval_and_log(path, info['checkpoint_bytes'], full)

    def eval_and_log(self, path, size=None, full=True):
        """Evaluation failures are logged, never fatal: an unattended run keeps training."""
        started = time.perf_counter()
        try:
            record = self.evaluate(full)
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.model.train()
            record = dict(exposures=self.state['exposures'], error=f'{type(exc).__name__}: {exc}')
            self.append('events.jsonl', dict(event='eval_error', checkpoint=path.name, time=time.time(),
                                             traceback=traceback.format_exc()[-4000:]))
        record.update(checkpoint=path.name, checkpoint_bytes=size, safe=path.stem.endswith('-safe'), full=full,
                      time=time.time(), eval_s=time.perf_counter() - started)
        self.append('evals.jsonl', record)
        if self.guard is not None:
            self.rss_mark = psutil.Process().memory_info().rss

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
        """Train ``windows`` windows from initialization; report throughput and per-window memory as JSON."""
        proc = psutil.Process()
        per, rss, drv = [], [], []
        losses, probe = [], []
        heads_total, t_total = 0, 0.0
        steps = max(1, windows // self.cfg.accumulate)
        drv0 = mps_bytes(self.cfg.device)[0]
        for s in range(steps):
            batch = self.next_batch()
            t0 = time.time()
            info = self.step(batch, probe=probe)
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
                      memory=self.cfg.memory, levels=self.cfg.levels, rule_l=self.cfg.rule_l,
                      mu_star=self.cfg.mu_star, windows=steps * self.cfg.accumulate,
                      steps=steps, head_decisions=self.state['exposures'],
                      decisions_per_s=heads_total / max(t_total, 1e-9),
                      seconds_per_step=dict(median=float(np.median(per)), p90=float(np.percentile(per, 90)),
                                            max=float(np.max(per))),
                      peak_rss_gib=max(rss) / GiB, final_rss_gib=rss[-1] / GiB,
                      per_window=probe,
                      mps_driver_gib_start=drv0 / GiB, mps_driver_gib_peak=max(drv) / GiB,
                      mps_driver_gib_series=[d / GiB for d in drv[::max(1, len(drv) // 20)]],
                      mps_monotone_growth=bool(len(drv) > 2 and all(b >= a for a, b in zip(drv, drv[1:]))
                                               and drv[-1] > drv[0]),
                      loss_first=float(np.mean(losses[:k])), loss_last=float(np.mean(losses[-k:])),
                      all_finite=bool(np.all(np.isfinite(losses))),
                      parameters=self.model.parameter_counts()['total'], star_conditions=self.star)
        if len(probe) > 2:
            x = np.array([[p['factors'], p['candidate_pairs']] for p in probe], dtype=float)
            y = np.array([p['rss_bytes'] for p in probe], dtype=float) / GiB
            result['rss_vs_load'] = dict(corr_factors=float(np.corrcoef(x[:, 0], y)[0, 1]),
                                         corr_candidate_pairs=float(np.corrcoef(x[:, 1], y)[0, 1]))
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
    p.add_argument('--hidden', type=int, default=None)
    p.add_argument('--expansion', type=int, default=None)
    p.add_argument('--rank', type=int, default=None)
    p.add_argument('--max-parameters', type=int, default=None)
    p.add_argument('--conditioner', default=None)
    p.add_argument('--run-dir', default=None)
    p.add_argument('--set', action='append', default=[], help='config override key=value (JSON values)')
    a = p.parse_args(argv)
    frozen = os.environ.get('R2_FROZEN_CODE')
    if frozen and not Path(__file__).resolve().is_relative_to(Path(frozen).resolve()):
        raise SystemExit(f'trainer imported from {__file__}, not from the frozen copy {frozen}')
    extra = {}
    for item in a.set:
        key, value = item.split('=', 1)
        try:
            extra[key] = json.loads(value)
        except json.JSONDecodeError:
            extra[key] = value
    cfg = TrainConfig.load(a.config, device=a.device, threads=a.threads, memory=a.memory, levels=a.levels,
                           hidden=a.hidden, expansion=a.expansion, rank=a.rank, max_parameters=a.max_parameters,
                           conditioner=a.conditioner, run_dir=a.run_dir, **extra)
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
        manifest = trainer.manifests()
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
