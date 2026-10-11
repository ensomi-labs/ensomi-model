"""CE window draws over the R2 cache and the two fixed fit_dev manifests.

Draw: group uniform, chart uniform in the group, then ``conditions.draw_window`` (candidates,
dropout first, alignment or the v1 start rule, per-window or per-song interval selection, the
inverse-probability weight). The v1 start rule: BOS with p 0.125, max(0, K - 255) with
p 0.125, else uniform in [0, K]. A window scores up to 256 decisions from its start j; EOS
(decision K) is included exactly when the window reaches it.

Manifests (plan v4 section 8.3), both versioned by a hash of the draw parameters, the label
file, nu and rule L, and rebuilt when that hash changes:

- natural: 64 fit_dev windows by the v1 start rule with empty tracks (random seed 954);
  draw-independent and shared by every arm; it carries the selection primary.
- condition: 64 fit_dev windows by the configured draw, stratified so each kind has at least
  24 windows with a scored onset, at least 8 with an LN span longer than 64 beats, 8 with a
  value switch at a boundary inside the window and 8 with a masked onset; diagnostics only.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .cache import CacheIndex, load_chart
from .common import ContractError, GridArrays, grid_from_arrays
from .conditions import WINDOW, DrawConfig, draw_window, old_start
from .features import Chart, Interval
from .labels import load_cells
from .locality import RULE_L_VERSION, masked
from .ln_level import whole_ln_length, whole_ln_level
from .properties import NU_HASH

NATURAL_SEED = 954


@dataclass
class Draw:
    sha: str
    start: int
    stop: int
    track: tuple
    record: dict
    weight: float = 1.0
    ln_level: float | None = None
    ln_length: float | None = None
    theta: np.ndarray | None = None
    history_start: int = 0


def chart_from_cache(dec) -> Chart:
    grid = GridArrays.from_grid(grid_from_arrays(dec.grid_segments, dec.grid_bars))
    return Chart(dec.head_ms, dec.song_ms, grid, dec.actions.astype(np.int64), dec.gap_release_ms)


class Corpus:
    def __init__(self, cache_root, role: str, *, star_conditions: bool, draw: DrawConfig = DrawConfig(),
                 baseline=None, capacity: int = 128):
        self.root = Path(cache_root)
        self.index = CacheIndex.open(self.root)
        table = self.index.table
        self.table = table[table.role == role].reset_index(drop=True)
        if not len(self.table):
            raise ContractError(f'No {role} charts in the cache')
        self.role = role
        self.groups = sorted(self.table.group_id.unique())
        self.members = {g: sorted(s) for g, s in self.table.groupby('group_id').sha256}
        self.files = dict(zip(self.table.sha256, self.table.file))
        self.K = dict(zip(self.table.sha256, self.table.K))
        self.band = dict(zip(self.table.sha256, self.table.band))
        self.draw_cfg = draw
        self.star_conditions = star_conditions
        self.baseline = baseline
        if draw.star_value == 'residual' and star_conditions and baseline is None:
            raise ContractError('Residual difficulty draws need the frozen baseline b(S)')
        self.cells, self.star_complete, self.label_sha256 = (load_cells(self.root) if star_conditions
                                                              else ({}, False, None))
        self.capacity = capacity
        self._charts: OrderedDict = OrderedDict()
        self._frame_cells: dict = {}
        self._source_ln_levels: dict[str, float] = {}
        self._source_ln_lengths: dict[str, float | None] = {}

    def chart(self, sha: str) -> Chart:
        if sha in self._charts:
            self._charts.move_to_end(sha)
            return self._charts[sha]
        chart = chart_from_cache(load_chart(self.root / 'charts' / self.files[sha]))
        self._charts[sha] = chart
        if len(self._charts) > self.capacity:
            self._charts.popitem(last=False)
        return chart

    def star_cells(self, sha: str, chart: Chart):
        """{(L_s, phase_s) | 'whole': [(a, b, frame value)]}: absolute star, or star - b(S) when residual."""
        cells = self.cells.get(sha)
        if not cells or self.draw_cfg.star_value == 'absolute':
            return cells
        if sha not in self._frame_cells:
            self._frame_cells[sha] = {key: [(a, b, v - float(self.baseline.predict(chart.head_ms, a, b)))
                                            for a, b, v in cs] for key, cs in cells.items()}
            if len(self._frame_cells) > 4 * self.capacity:
                self._frame_cells.pop(next(iter(self._frame_cells)))
        return self._frame_cells[sha]

    def source_ln_level(self, sha: str) -> float:
        """Whole-song source LN share, independent of the draw and its scored window."""
        if sha not in self._source_ln_levels:
            self._source_ln_levels[sha] = whole_ln_level(self.chart(sha))
        return self._source_ln_levels[sha]

    def source_ln_length(self, sha: str) -> float | None:
        """Whole-source median log2 length in beats, cached independently of window draws."""
        if sha not in self._source_ln_lengths:
            self._source_ln_lengths[sha] = whole_ln_length(self.chart(sha))
        return self._source_ln_lengths[sha]

    def pick(self, rng):
        group = self.groups[int(rng.integers(0, len(self.groups)))]
        members = self.members[group]
        return members[int(rng.integers(0, len(members)))]

    def draw(self, rng, window: int = WINDOW) -> Draw:
        sha = self.pick(rng)
        chart = self.chart(sha)
        w = draw_window(chart, self.star_cells(sha, chart), rng, self.draw_cfg,
                        star_conditions=self.star_conditions, window=window)
        return Draw(sha, w.start, w.stop, w.track, w.record, w.weight)


def track_to_json(track):
    return [[iv.kind, iv.a, iv.b, iv.value] for iv in track]


def track_from_json(data):
    return tuple(Interval(int(k), float(a), float(b), float(v)) for k, a, b, v in data)


def manifest_version(kind: str, draw: DrawConfig, label_sha256, star_conditions: bool) -> str:
    payload = dict(kind=kind, draw=draw.hash() if kind == 'condition' else None, labels=label_sha256,
                   nu=NU_HASH, rule_l=RULE_L_VERSION, star_conditions=star_conditions, window=WINDOW)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_natural_manifest(corpus: Corpus, *, seed: int = NATURAL_SEED, windows: int = 64):
    rng = np.random.default_rng(seed)
    groups = list(corpus.groups)
    chosen = [groups[i] for i in rng.choice(len(groups), size=min(windows, len(groups)), replace=False)]
    out = []
    for g in chosen:
        members = corpus.members[g]
        sha = members[int(rng.integers(0, len(members)))]
        K = corpus.K[sha]
        start = old_start(rng, K)
        out.append(dict(sha256=sha, group_id=g, start=start, stop=min(start + WINDOW, K + 1), track=[], weight=1.0))
    return out


def window_strata(chart: Chart, start: int, stop: int, track) -> dict:
    """Which condition-manifest strata a window belongs to."""
    out = dict(onset_ln=False, onset_star=False, long_ln=False, switch=False, masked_onset=False)
    for iv in track:
        m = masked(chart, iv)
        f = m['first_visible']
        if f is not None and start <= f < stop:
            out['onset_ln' if iv.kind == 0 else 'onset_star'] = True
        if m['onset_masked'] and start <= m['onset'] < stop:
            out['masked_onset'] = True
        if iv.kind == 0:
            beats = float(chart.grid.beat(iv.b) - chart.grid.beat(iv.a))
            if beats > 64:
                out['long_ln'] = True
    t_lo, t_hi = chart.time(start), chart.time(min(stop, chart.K))
    for kind in (0, 1):
        ivs = sorted((iv for iv in track if iv.kind == kind), key=lambda iv: iv.a)
        for x, y in zip(ivs, ivs[1:]):
            if x.b == y.a and x.value != y.value and t_lo < y.a <= t_hi:
                out['switch'] = True
    return out


QUOTAS = dict(onset_ln=24, onset_star=24, long_ln=8, switch=8, masked_onset=8)


def build_condition_manifest(corpus: Corpus, *, seed: int = NATURAL_SEED, windows: int = 64, pool: int = 4000):
    """Greedy quota filling over a pool of draws, then random fill; one window per draw."""
    rng = np.random.default_rng(seed + 1)
    drawn = []
    for _ in range(pool):
        d = corpus.draw(rng)
        chart = corpus.chart(d.sha)
        drawn.append((d, window_strata(chart, d.start, d.stop, d.track)))
    chosen, counts = [], {k: 0 for k in QUOTAS}
    for key, quota in QUOTAS.items():
        for i, (d, s) in enumerate(drawn):
            if counts[key] >= quota or len(chosen) >= windows:
                break
            if s[key] and i not in chosen:
                chosen.append(i)
                for k2 in QUOTAS:
                    counts[k2] += int(s[k2])
    rest = [i for i in range(len(drawn)) if i not in chosen and drawn[i][0].track]
    for i in rng.permutation(len(rest))[:max(0, windows - len(chosen))]:
        chosen.append(rest[i])
    out = []
    for i in chosen[:windows]:
        d, s = drawn[i]
        out.append(dict(sha256=d.sha, start=d.start, stop=d.stop, track=track_to_json(d.track), weight=d.weight,
                        strata=s, record={k: v for k, v in d.record.items() if k != 'chosen'}))
    return out, counts


def freerun_charts(corpus: Corpus, rng, charts: int = 4, max_rows: int = 600):
    short = corpus.table[(corpus.table.K <= max_rows) & (corpus.table.K >= 50)]
    short_groups = sorted(short.group_id.unique())
    picks = []
    for i in rng.permutation(len(short_groups))[:charts]:
        rows = short[short.group_id == short_groups[i]].sort_values('sha256')
        picks.append(rows.sha256.iloc[int(rng.integers(0, len(rows)))])
    return sorted(picks)


def build_manifests(cache_root, *, star_conditions: bool, draw: DrawConfig, baseline=None):
    dev = Corpus(cache_root, 'fit_dev', star_conditions=star_conditions, draw=draw, baseline=baseline)
    natural = build_natural_manifest(dev)
    condition, counts = build_condition_manifest(dev)
    train = Corpus(cache_root, 'fit_train', star_conditions=star_conditions, draw=draw, baseline=baseline)
    condition_train, train_counts = build_condition_manifest(train, seed=NATURAL_SEED + 2)
    return dict(
        natural=dict(version=manifest_version('natural', draw, dev.label_sha256, star_conditions), seed=NATURAL_SEED,
                     windows=natural),
        condition=dict(version=manifest_version('condition', draw, dev.label_sha256, star_conditions),
                       seed=NATURAL_SEED + 1, windows=condition, strata=counts, quotas=QUOTAS),
        condition_train=dict(version=manifest_version('condition', draw, dev.label_sha256, star_conditions),
                             seed=NATURAL_SEED + 3, role='fit_train', windows=condition_train, strata=train_counts,
                             purpose='train-versus-dev NLL on onset decisions (memorisation of up-weighted onsets)'),
        freerun=freerun_charts(dev, np.random.default_rng(NATURAL_SEED)),
        role='fit_dev', star_conditions=star_conditions, draw=draw.to_dict(),
        cache_index_sha256=dev.index.summary.get('index_sha256'))


def load_or_build_manifests(cache_root, path: Path, *, star_conditions: bool, draw: DrawConfig, baseline=None):
    path = Path(path)
    labels = Corpus(cache_root, 'fit_dev', star_conditions=star_conditions, draw=draw,
                    baseline=baseline).label_sha256 if star_conditions else None
    want = (manifest_version('natural', draw, labels, star_conditions),
            manifest_version('condition', draw, labels, star_conditions))
    if path.exists():
        data = json.loads(path.read_text())
        if (data.get('natural', {}).get('version'), data.get('condition', {}).get('version')) == want:
            return data
    data = build_manifests(cache_root, star_conditions=star_conditions, draw=draw, baseline=baseline)
    path.write_text(json.dumps(data, indent=1))
    return data
