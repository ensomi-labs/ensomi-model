"""R2 sequence DPO on synthetic pairs: loss, pair builder, synthetic labellers, trainer and CLI.

A pair is two continuations y+ and y- of the same committed start state x (prefix decisions,
head skeleton, grid, song length, condition track) over the same scored head horizon, with a
soft label q = P(y+ preferred) in [0, 1]. Each branch is replayed on its own states and both
models score every decision of it with ``sequence_log_prob`` (the reference without gradient,
in eval mode):

    R(y) = log pi_theta(y | x) - log pi_0(y | x),        Delta = R(y+) - R(y-)
    loss = mean over pairs of [-q log sigma(beta Delta) - (1 - q) log sigma(-beta Delta)]
           + lambda_CE * CE(anchor windows)

CE(anchor) is the CE trainer's loss, the window mean of decision NLL averaged over windows.
R sums over decisions; no per-decision or per-release normalisation (design section 5). Pairs
here are synthetic: two samples from one start state drawn with the R2 sampler, labelled by a
known utility through a fixed logistic, q = sigma(alpha (u+ - u-)), oriented so q >= 0.5.

CLI (mac, repository root; a CE checkpoint is the initial policy and the frozen reference):

    python -m ensomi_model.r2.train_dpo --checkpoint <ce.pt> --run-dir artifacts/r2-dpo/<id> \
        [--set key=value ...] [--resume]
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from .common import ContractError, GridArrays
from .features import Chart
from .sampling import continue_chart
from .state import replay_decisions


# ---- start states and pairs ---------------------------------------------------------------------

@dataclass(eq=False)
class StartState:
    """Committed start state: inputs, decisions [0, start) and the condition track."""
    head_ms: np.ndarray
    song_ms: float
    grid: GridArrays
    prefix_actions: np.ndarray
    prefix_gap: np.ndarray
    track: tuple = ()
    group: str = ''      # cluster for standard errors (song group)
    sha: str = ''        # cache chart identity, when the state comes from the cache
    _base: Chart | None = field(default=None, repr=False)
    _warm: int = field(default=0, repr=False)

    def __post_init__(self):
        self.prefix_actions = np.asarray(self.prefix_actions, dtype=np.int64).reshape(-1, 4)
        self.prefix_gap = np.asarray(self.prefix_gap, dtype=np.float64).reshape(-1, 4)
        if len(self.prefix_gap) != len(self.prefix_actions) or self.start > self.K:
            raise ContractError('A start state has at most K prefix decisions with matching gap rows')

    @property
    def K(self):
        return len(self.head_ms)

    @property
    def start(self):
        return len(self.prefix_actions)

    def stop(self, horizon: int) -> int:
        return min(self.start + int(horizon), self.K + 1)

    def chart(self, actions, gap) -> Chart:
        """Chart of the prefix followed by ``actions``/``gap``; gap candidates are shared across branches."""
        if self._base is None:
            self._base = Chart(self.head_ms, self.song_ms, self.grid, self.prefix_actions[:0], self.prefix_gap[:0])
            self._base.head_beats  # noqa: B018  fills the beat caches that with_decisions copies
        stop = self.start + len(actions)
        for k in range(max(1, self.start, self._warm), stop):
            self._base.candidates(k)
        self._warm = max(self._warm, stop)
        return self._base.with_decisions(np.concatenate((self.prefix_actions, np.asarray(actions, np.int64))),
                                         np.concatenate((self.prefix_gap, np.asarray(gap, np.float64))))

    def mirrored(self) -> 'StartState':
        return StartState(self.head_ms, self.song_ms, self.grid, self.prefix_actions[:, ::-1].copy(),
                          self.prefix_gap[:, ::-1].copy(), self.track, self.group, self.sha)


def mirror_branch(branch):
    actions, gap = branch
    return np.asarray(actions)[:, ::-1].copy(), np.asarray(gap)[:, ::-1].copy()


@dataclass(eq=False)
class Pair:
    state: StartState
    plus: tuple          # (actions [h,4], gap [h,4]) of the preferred continuation
    minus: tuple
    q: float             # P(plus preferred)
    meta: dict = field(default_factory=dict)

    @property
    def stop(self):
        return self.state.start + len(self.plus[0])

    def mirrored(self) -> 'Pair':
        return Pair(self.state.mirrored(), mirror_branch(self.plus), mirror_branch(self.minus), self.q,
                    dict(self.meta, mirrored=not self.meta.get('mirrored', False)))


def validate_pair(pair: Pair):
    """Equal horizons, q in [0, 1], and both branches replay legally from the same prefix."""
    s = pair.state
    (pa, pg), (ma, mg) = pair.plus, pair.minus
    if not (np.shape(pa) == np.shape(ma) == np.shape(pg) == np.shape(mg)) or len(pa) == 0:
        raise ContractError('Both branches need the same nonzero number of decisions')
    if pair.stop > s.K + 1:
        raise ContractError('Branch runs past EOS')
    if not 0.0 <= pair.q <= 1.0:
        raise ContractError('Soft label outside [0, 1]')
    for a, g in (pair.plus, pair.minus):
        replay_decisions(s.head_ms, s.song_ms, np.concatenate((s.prefix_actions, a)),
                         np.concatenate((s.prefix_gap, g)))
    return pair


def branch_log_prob(model, state: StartState, branch) -> torch.Tensor:
    """Sum of complete decision log-probabilities of a branch, replayed on its own states."""
    actions, gap = branch
    return model.sequence_log_prob(state.chart(actions, gap), state.start, state.start + len(actions), state.track)


def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()


# ---- loss ---------------------------------------------------------------------------------------

@dataclass
class AnchorWindow:
    chart: Chart
    start: int
    stop: int
    track: tuple = ()


@dataclass
class DPOResult:
    loss: torch.Tensor           # differentiable, or detached when dpo_loss ran with backward=True
    preference: float
    ce: float | None
    delta: np.ndarray            # [P] Delta per pair
    ratio_plus: np.ndarray       # [P] R(y+)
    ratio_minus: np.ndarray      # [P] R(y-)
    q: np.ndarray
    decisions: np.ndarray        # [P] decisions per branch
    beta: float

    def stats(self) -> dict:
        bd = self.beta * self.delta
        side = np.sign(self.q - 0.5)
        informative = side != 0
        hit = np.where(bd * side > 0, 1.0, np.where(bd == 0, 0.5, 0.0))
        return dict(pairs=len(bd), loss=self.loss.item(), preference_loss=self.preference, anchor_ce=self.ce,
                    beta_delta_mean=float(bd.mean()), beta_delta_std=float(bd.std()),
                    beta_delta_min=float(bd.min()), beta_delta_max=float(bd.max()),
                    beta_ratio_plus_mean=float(self.beta * self.ratio_plus.mean()),
                    beta_ratio_minus_mean=float(self.beta * self.ratio_minus.mean()),
                    preference_accuracy=float(hit[informative].mean()) if informative.any() else None,
                    q_mean=float(self.q.mean()), decisions_per_branch=float(self.decisions.mean()))


def preference_loss(beta_delta: torch.Tensor, q: float) -> torch.Tensor:
    """-q log sigma(beta Delta) - (1 - q) log sigma(-beta Delta)."""
    return -(q * F.logsigmoid(beta_delta) + (1.0 - q) * F.logsigmoid(-beta_delta))


def window_ce(model, window: AnchorWindow) -> torch.Tensor:
    return -model.window(window.chart, window.start, window.stop, window.track).total.mean()


def dpo_loss(policy, reference, pairs, anchor_windows=(), beta=0.1, lambda_ce=0.2, *, backward=False) -> DPOResult:
    """Soft-label sequence DPO over ``pairs`` plus ``lambda_ce`` times the anchor CE.

    With ``backward=True`` every pair term and anchor window is backpropagated as soon as it is
    computed (one branch graph alive at a time); the accumulated gradient is the gradient of
    the same total loss, and the returned loss is detached.
    """
    pairs = list(pairs)
    if not pairs:
        raise ContractError('dpo_loss needs at least one pair')
    reference.train(False)
    P = len(pairs)
    total = 0.0
    pref = 0.0
    deltas, plus, minus, qs, lengths = [], [], [], [], []
    for pair in pairs:
        with torch.no_grad():
            ref_plus = branch_log_prob(reference, pair.state, pair.plus)
            ref_minus = branch_log_prob(reference, pair.state, pair.minus)
        r_plus = branch_log_prob(policy, pair.state, pair.plus) - ref_plus
        r_minus = branch_log_prob(policy, pair.state, pair.minus) - ref_minus
        delta = r_plus - r_minus
        term = preference_loss(beta * delta, float(pair.q)) / P
        if not torch.isfinite(term):
            raise ContractError(f'Non-finite preference term (Delta {float(delta)})')
        if backward:
            term.backward()
            term = term.detach()
        total = total + term
        pref += term.item()
        deltas.append(delta.item())
        plus.append(r_plus.item())
        minus.append(r_minus.item())
        qs.append(float(pair.q))
        lengths.append(len(pair.plus[0]))
    ce = None
    windows = list(anchor_windows)
    if windows and lambda_ce:
        ce = 0.0
        for w in windows:
            c = window_ce(policy, w) / len(windows)
            if not torch.isfinite(c):
                raise ContractError('Non-finite anchor CE')
            term = lambda_ce * c
            if backward:
                term.backward()
                term, c = term.detach(), c.detach()
            total = total + term
            ce += c.item()
    loss = total if torch.is_tensor(total) else torch.tensor(total)
    return DPOResult(loss, pref, ce, np.array(deltas), np.array(plus), np.array(minus), np.array(qs),
                     np.array(lengths), float(beta))


# ---- synthetic labellers ------------------------------------------------------------------------

def branch_events(state: StartState, branch, lo_ms=1.0, hi_ms=40.0) -> dict:
    """Counts over the branch's decisions: heads, LN heads, releases, gap releases and releases
    that come lo..hi ms before a head in another lane (heads after the branch are unknown)."""
    actions, gap = branch
    chart = state.chart(actions, gap)
    d = chart.derived()
    a, b = state.start, state.start + len(actions)
    rel = d.release[a:b]
    rows = chart.times(np.arange(b))
    attack = d.attack[:b]
    near = 0
    for i, lane in zip(*np.nonzero(~np.isnan(rel))):
        others = rows[np.delete(attack, lane, axis=1).any(1)]
        dt = others - rel[i, lane]
        near += int(((dt >= lo_ms) & (dt <= hi_ms)).any())
    return dict(heads=int(d.attack[a:b].sum()), ln_heads=int(d.ln_head[a:b].sum()),
                releases=int((~np.isnan(rel)).sum()), gap_releases=int((~np.isnan(rel) & ~d.row_release[a:b]).sum()),
                near_head_releases=near, decisions=b - a, eos=b == state.K + 1)


@dataclass(frozen=True)
class LNShareLabeller:
    """Utility -|LN share - target| over the branch's heads; statistic is the LN share."""
    target: float = 0.5
    alpha: float = 20.0
    name: str = 'ln_share'

    def statistic(self, state, branch) -> float:
        e = branch_events(state, branch)
        return e['ln_heads'] / e['heads'] if e['heads'] else math.nan

    def utility(self, state, branch) -> float:
        return -abs(self.statistic(state, branch) - self.target)

    def label(self, u_a, u_b) -> float:
        return 0.5 if math.isnan(u_a) or math.isnan(u_b) else 1.0 / (1.0 + math.exp(-self.alpha * (u_a - u_b)))


@dataclass(frozen=True)
class NearHeadLabeller:
    """Utility -(number of releases 1..40 ms before another lane's head); statistic is that count
    per release."""
    lo_ms: float = 1.0
    hi_ms: float = 40.0
    alpha: float = 1.0
    name: str = 'near_head'

    def statistic(self, state, branch) -> float:
        e = branch_events(state, branch, self.lo_ms, self.hi_ms)
        return e['near_head_releases'] / e['releases'] if e['releases'] else math.nan

    def utility(self, state, branch) -> float:
        return -float(branch_events(state, branch, self.lo_ms, self.hi_ms)['near_head_releases'])

    label = LNShareLabeller.label


def make_labeller(name: str, **kw):
    cls = {'ln_share': LNShareLabeller, 'near_head': NearHeadLabeller}[name]
    names = {f.name for f in fields(cls)} - {'name'}
    return cls(**{k: v for k, v in kw.items() if k in names and v is not None})


# ---- pair builder -------------------------------------------------------------------------------

def sample_branch(model, state: StartState, horizon: int, seed: int):
    """One continuation of ``horizon`` head decisions (fewer at the chart end, EOS included)."""
    stop = state.stop(horizon)
    actions, gap = continue_chart(model, state.head_ms, state.song_ms, state.grid, state.prefix_actions,
                                  state.prefix_gap, track=state.track, seed=int(seed), stop=stop)
    return actions[state.start:stop].copy(), gap[state.start:stop].copy()


def same_branch(a, b) -> bool:
    return np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1], equal_nan=True)


def build_pair(model, state: StartState, labeller, horizon=64, seeds=(0, 1), min_margin=0.05):
    """Two samples from ``state`` labelled by ``labeller``; None when identical or |q - 0.5| < min_margin."""
    a = sample_branch(model, state, horizon, seeds[0])
    b = sample_branch(model, state, horizon, seeds[1])
    if same_branch(a, b):
        return None
    u_a, u_b = labeller.utility(state, a), labeller.utility(state, b)
    q = labeller.label(u_a, u_b)
    if abs(q - 0.5) < min_margin:
        return None
    if q < 0.5:
        a, b, u_a, u_b, q, seeds = b, a, u_b, u_a, 1.0 - q, tuple(reversed(seeds))
    pair = Pair(state, a, b, q, dict(source='synthetic', labeller=labeller.name, labeller_config=asdict(labeller),
                                     utility_plus=u_a, utility_minus=u_b, seeds=[int(s) for s in seeds],
                                     horizon=int(horizon)))
    return validate_pair(pair)


def build_pairs(model, states, labeller, horizon=64, per_state=1, seed=0, min_margin=0.05, tries=4):
    """Up to ``per_state`` pairs per state, at most ``tries * per_state`` sibling draws each."""
    rng = np.random.default_rng(seed)
    out = []
    for state in states:
        made = 0
        for _ in range(tries * per_state):
            if made == per_state:
                break
            seeds = tuple(int(s) for s in rng.integers(0, 2 ** 31 - 1, size=2))
            pair = build_pair(model, state, labeller, horizon, seeds, min_margin)
            if pair is not None:
                out.append(pair)
                made += 1
    return out


# ---- monitors -----------------------------------------------------------------------------------

def cluster_mean_se(values, groups):
    """Mean over clusters of the cluster means, and its standard error (NaN with one cluster)."""
    by = {}
    for v, g in zip(values, groups):
        if not math.isnan(v):
            by.setdefault(g, []).append(v)
    means = np.array([np.mean(v) for v in by.values()])
    if not len(means):
        return math.nan, math.nan, 0
    se = float(means.std(ddof=1) / math.sqrt(len(means))) if len(means) > 1 else math.nan
    return float(means.mean()), se, len(means)


@torch.no_grad()
def evaluate_samples(policy, reference, states, labeller, horizon, seeds) -> dict:
    """Fresh samples from the current policy at each state: Monte Carlo KL to the reference per
    decision, (log pi_theta - log pi_0) / decisions averaged, and the labeller statistic."""
    reference.train(False)
    kl, stat, groups, per_state = [], [], [], []
    for i, state in enumerate(states):
        values = []
        for seed in seeds:
            branch = sample_branch(policy, state, horizon, seed)
            lp = float(branch_log_prob(policy, state, branch))
            l0 = float(branch_log_prob(reference, state, branch))
            kl.append((lp - l0) / len(branch[0]))
            groups.append(state.group or str(i))
            values.append(labeller.statistic(state, branch) if labeller is not None else math.nan)
        stat += values
        per_state.append(float(np.nanmean(values)) if not np.all(np.isnan(values)) else math.nan)
    kl_mean, kl_se, clusters = cluster_mean_se(kl, groups)
    stat_mean, stat_se, _ = cluster_mean_se(stat, groups)
    return dict(kl_per_decision=kl_mean, kl_per_decision_se=kl_se, statistic=stat_mean, statistic_se=stat_se,
                statistic_per_state=per_state, samples=len(kl), clusters=clusters,
                labeller=None if labeller is None else labeller.name)


# ---- trainer ------------------------------------------------------------------------------------

@dataclass
class DPOConfig:
    checkpoint: str = ''                 # CE checkpoint: initial policy and frozen reference
    run_dir: str = 'artifacts/r2-dpo/dev'
    cache: str = 'artifacts/r2-cache/v1'
    device: str = 'cpu'
    threads: int = 2
    dtype: str = 'float32'
    beta: float = 0.1
    lambda_ce: float = 0.2
    lr: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.01
    clip: float = 1.0
    warmup_steps: int = 50
    steps: int = 256
    pairs_per_step: int = 8
    anchor_windows: int = 8
    anchor_window: int = 256
    horizon: int = 64
    train_states: int = 128
    pairs_per_state: int = 1
    held_states: int = 32
    eval_windows: int = 16
    labeller: str = 'ln_share'
    target: float = 0.5
    alpha: float | None = None           # labeller default when None
    min_margin: float = 0.05
    eval_every: int = 128
    eval_seeds: list = field(default_factory=lambda: [954, 955])
    seed_states: int = 2468
    seed_pairs: int = 1357
    seed_anchor: int = 4321

    @classmethod
    def load(cls, path=None, **overrides):
        data = json.loads(Path(path).read_text()) if path else {}
        data.update({k: v for k, v in overrides.items() if v is not None})
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ContractError(f'Unknown config keys: {sorted(unknown)}')
        return cls(**data)


class DPOTrainer:
    """AdamW on the policy; 50-step linear warmup then constant LR; one shuffled pass over the
    pair pool after another; fresh anchor windows per update; checkpoint and evaluate every
    ``eval_every`` updates and at the end."""

    def __init__(self, policy, reference, cfg: DPOConfig, pairs, anchor_draw=None, held_states=(), labeller=None,
                 eval_windows=(), run_dir=None, reference_sha256=None):
        self.cfg = cfg
        self.policy = policy
        self.reference = freeze(reference)
        self.pairs = list(pairs)
        if not self.pairs:
            raise ContractError('No pairs to train on')
        self.anchor_draw = anchor_draw
        self.held = list(held_states)
        self.labeller = labeller
        self.eval_windows = list(eval_windows)
        self.run = Path(run_dir) if run_dir else None
        self.reference_sha256 = reference_sha256
        params = [p for p in policy.parameters() if p.requires_grad]
        self.params = params
        self.opt = torch.optim.AdamW([dict(params=[p for p in params if p.ndim >= 2], weight_decay=cfg.weight_decay),
                                      dict(params=[p for p in params if p.ndim < 2], weight_decay=0.0)],
                                     lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)
        self.pair_rng = np.random.default_rng(cfg.seed_pairs)
        self.anchor_rng = np.random.default_rng(cfg.seed_anchor)
        self.state = dict(step=0, cursor=0, passes=0, order=self.pair_rng.permutation(len(self.pairs)).tolist())
        self.last_eval = None

    # ---- one update ------------------------------------------------------------------------------

    def lr_now(self):
        return self.cfg.lr * min(1.0, (self.state['step'] + 1) / max(1, self.cfg.warmup_steps))

    def next_pairs(self):
        out = []
        while len(out) < self.cfg.pairs_per_step:
            if self.state['cursor'] >= len(self.state['order']):
                self.state.update(cursor=0, passes=self.state['passes'] + 1,
                                  order=self.pair_rng.permutation(len(self.pairs)).tolist())
            out.append(self.pairs[self.state['order'][self.state['cursor']]])
            self.state['cursor'] += 1
        return out

    def step(self) -> dict:
        cfg = self.cfg
        lr = self.lr_now()
        for group in self.opt.param_groups:
            group['lr'] = lr
        batch = self.next_pairs()
        anchors = ([self.anchor_draw(self.anchor_rng) for _ in range(cfg.anchor_windows)]
                   if cfg.lambda_ce and self.anchor_draw is not None else [])
        self.policy.train()
        self.opt.zero_grad(set_to_none=True)
        result = dpo_loss(self.policy, self.reference, batch, anchors, cfg.beta, cfg.lambda_ce, backward=True)
        norm = torch.nn.utils.clip_grad_norm_(self.params, cfg.clip)
        if not torch.isfinite(norm):
            raise ContractError('Non-finite gradient norm')
        self.opt.step()
        self.state['step'] += 1
        return dict(step=self.state['step'], lr=lr, grad_norm=float(norm), passes=self.state['passes'],
                    **result.stats())

    # ---- evaluation ------------------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict:
        cfg = self.cfg
        record = dict(step=self.state['step'])
        pool = dpo_loss(self.policy, self.reference, self.pairs, (), cfg.beta, 0.0).stats()
        record['pool'] = {k: pool[k] for k in ('pairs', 'preference_loss', 'preference_accuracy', 'beta_delta_mean',
                                               'beta_delta_std', 'beta_delta_min', 'beta_delta_max')}
        if self.eval_windows:
            self.policy.eval()
            record['eval_ce'] = float(np.mean([float(window_ce(self.policy, w)) for w in self.eval_windows]))
        if self.held:
            record['held'] = evaluate_samples(self.policy, self.reference, self.held, self.labeller, cfg.horizon,
                                              cfg.eval_seeds)
        self.policy.train()
        return record

    # ---- persistence -----------------------------------------------------------------------------

    def payload(self):
        config = getattr(self.policy, 'config', None)
        return dict(policy=self.policy.state_dict(), optimizer=self.opt.state_dict(), state=copy.deepcopy(self.state),
                    pair_rng=self.pair_rng.bit_generator.state, anchor_rng=self.anchor_rng.bit_generator.state,
                    config=asdict(self.cfg), model_config=asdict(config) if config is not None else None,
                    reference_sha256=self.reference_sha256, pairs=len(self.pairs))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + '.tmp')
        torch.save(self.payload(), tmp)
        os.replace(tmp, path)
        return path

    def load(self, path):
        data = torch.load(path, map_location='cpu', weights_only=False)
        if data.get('reference_sha256') != self.reference_sha256:
            raise ContractError('Checkpoint was trained against a different reference')
        if data.get('pairs') != len(self.pairs):
            raise ContractError('Checkpoint was trained on a different pair pool')
        self.policy.load_state_dict(data['policy'])
        self.opt.load_state_dict(data['optimizer'])
        self.state = data['state']
        self.pair_rng.bit_generator.state = data['pair_rng']
        self.anchor_rng.bit_generator.state = data['anchor_rng']
        return data

    @property
    def ckpt_dir(self):
        return None if self.run is None else self.run / 'checkpoints'

    def latest(self):
        if self.run is None or not (self.ckpt_dir / 'latest.json').exists():
            return None
        return self.ckpt_dir / json.loads((self.ckpt_dir / 'latest.json').read_text())['path']

    def append(self, name, record):
        if self.run is not None:
            self.run.mkdir(parents=True, exist_ok=True)
            with (self.run / name).open('a') as f:
                f.write(json.dumps(record, allow_nan=True) + '\n')

    def checkpoint_and_eval(self):
        record = self.evaluate()
        if self.run is not None:
            path = self.save(self.ckpt_dir / f'dpo-{self.state["step"]:06d}.pt')
            (self.ckpt_dir / 'latest.json').write_text(json.dumps(dict(path=path.name, step=self.state['step'])))
            record['checkpoint'] = path.name
        record['time'] = time.time()
        self.append('evals.jsonl', record)
        self.last_eval = record
        return record

    def train(self, steps=None):
        target = self.cfg.steps if steps is None else int(steps)
        record = None
        while self.state['step'] < target:
            t0 = time.time()
            record = self.step()
            record.update(seconds=time.time() - t0, time=time.time())
            self.append('train.jsonl', record)
            if self.state['step'] % self.cfg.eval_every == 0 or self.state['step'] == target:
                self.checkpoint_and_eval()
        return record

    # ---- from a CE checkpoint and the cache --------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, cfg: DPOConfig, resume=False):
        from .data import Corpus
        from .model import R2Config, R2Model
        from .receipts import write_receipt

        torch.set_num_threads(cfg.threads)
        run = Path(cfg.run_dir)
        data = torch.load(cfg.checkpoint, map_location='cpu', weights_only=False)
        device, dtype = torch.device(cfg.device), getattr(torch, cfg.dtype)
        reference = R2Model(R2Config(**data['model_config']))
        reference.load_state_dict(data['model'])
        reference = reference.to(device, dtype)
        policy = copy.deepcopy(reference)
        sha = file_sha256(cfg.checkpoint)
        star = bool(data.get('star_conditions', False))
        fit = Corpus(cfg.cache, 'fit_train', star_conditions=star)
        dev = Corpus(cfg.cache, 'fit_dev', star_conditions=False)
        labeller = make_labeller(cfg.labeller, target=cfg.target, alpha=cfg.alpha)
        held_draws = draws(dev, cfg.held_states, cfg.horizon, cfg.seed_states + 1)
        held = [corpus_state(dev, d) for d in held_draws]
        eval_windows = [AnchorWindow(dev.chart(d.sha), d.start, d.stop, d.track)
                        for d in draws(dev, cfg.eval_windows, cfg.anchor_window, cfg.seed_states + 2)]
        pairs_file = run / 'pairs.pt'
        if resume and pairs_file.exists():
            pairs = load_pairs(pairs_file, fit)
        else:
            if (run / 'checkpoints' / 'latest.json').exists():
                raise ContractError('Run directory already has checkpoints; pass --resume')
            states = [corpus_state(fit, d) for d in draws(fit, cfg.train_states, cfg.horizon, cfg.seed_states)]
            pairs = build_pairs(reference, states, labeller, cfg.horizon, cfg.pairs_per_state, cfg.seed_pairs,
                                cfg.min_margin)
            run.mkdir(parents=True, exist_ok=True)
            save_pairs(pairs_file, pairs, reference_sha256=sha)
            (run / 'config.json').write_text(json.dumps(asdict(cfg), indent=1))
            (run / 'pairs_summary.json').write_text(json.dumps(pairs_summary(pairs, len(states)), indent=1))

        def anchor_draw(rng):
            d = fit.draw(rng, cfg.anchor_window)
            return AnchorWindow(fit.chart(d.sha), d.start, d.stop, d.track)

        trainer = cls(policy, reference, cfg, pairs, anchor_draw, held, labeller, eval_windows, run, sha)
        latest = trainer.latest() if resume else None
        if latest is not None:
            trainer.load(latest)
            trainer.append('events.jsonl', dict(event='resume', checkpoint=latest.name, step=trainer.state['step'],
                                                time=time.time()))
        else:
            trainer.append('events.jsonl', dict(event='start', time=time.time(), pairs=len(pairs)))
        write_receipt(run / f'receipt-{int(time.time())}.json', mode='dpo', config=asdict(cfg),
                      reference_checkpoint=cfg.checkpoint, reference_sha256=sha, star_conditions=star,
                      model_config=data['model_config'], pairs=len(pairs), resumed_from=latest and latest.name,
                      torch=torch.__version__, ens_job=os.environ.get('ENS_JOB_ID'))
        return trainer


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def draws(corpus, n, window, seed):
    rng = np.random.default_rng(seed)
    return [corpus.draw(rng, window) for _ in range(n)]


def corpus_state(corpus, draw) -> StartState:
    chart = corpus.chart(draw.sha)
    group = str(corpus.table.loc[corpus.table.sha256 == draw.sha, 'group_id'].iloc[0])
    s = draw.start
    return StartState(chart.head_ms, chart.song_ms, chart.grid, chart.actions[:s], chart.gap[:s], draw.track,
                      group, draw.sha)


def save_pairs(path, pairs, **extra):
    from .data import track_to_json
    records = [dict(sha=p.state.sha, start=p.state.start, track=track_to_json(p.state.track), group=p.state.group,
                    plus=p.plus, minus=p.minus, q=p.q, meta=p.meta) for p in pairs]
    torch.save(dict(records=records, **extra), path)


def load_pairs(path, corpus):
    from .data import track_from_json
    data = torch.load(path, map_location='cpu', weights_only=False)
    out = []
    for r in data['records']:
        chart = corpus.chart(r['sha'])
        s = r['start']
        state = StartState(chart.head_ms, chart.song_ms, chart.grid, chart.actions[:s], chart.gap[:s],
                           track_from_json(r['track']), r['group'], r['sha'])
        out.append(validate_pair(Pair(state, tuple(r['plus']), tuple(r['minus']), r['q'], r['meta'])))
    return out


def pairs_summary(pairs, states):
    q = np.array([p.q for p in pairs]) if pairs else np.zeros(0)
    return dict(pairs=len(pairs), states=states, q_mean=float(q.mean()) if len(q) else None,
                q_min=float(q.min()) if len(q) else None, eos_pairs=sum(p.stop == p.state.K + 1 for p in pairs),
                decisions_per_branch=float(np.mean([len(p.plus[0]) for p in pairs])) if pairs else None)


def _value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def main(argv=None):
    p = argparse.ArgumentParser(description='R2 sequence DPO on synthetic pairs')
    p.add_argument('--checkpoint', default=None, help='CE checkpoint: initial policy and frozen reference')
    p.add_argument('--config', default=None, help='JSON DPOConfig (the run directory config on --resume)')
    p.add_argument('--run-dir', default=None)
    p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', help='config override, JSON value')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args(argv)
    overrides = {k: _value(v) for k, v in (s.split('=', 1) for s in a.set)}
    config = a.config
    if a.resume and config is None and a.run_dir and (Path(a.run_dir) / 'config.json').exists():
        config = Path(a.run_dir) / 'config.json'
    cfg = DPOConfig.load(config, checkpoint=a.checkpoint, run_dir=a.run_dir, **overrides)
    if not cfg.checkpoint:
        raise SystemExit('--checkpoint is required')
    trainer = DPOTrainer.from_checkpoint(cfg, resume=a.resume)
    last = trainer.train()
    print(json.dumps(dict(step=trainer.state['step'], last=last), allow_nan=True), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
