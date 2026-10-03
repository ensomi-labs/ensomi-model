"""CE window draws over the R2 cache and the fixed fit_dev manifest.

Draw: group uniform, chart uniform in the group. Window start j: BOS with
p 0.125, max(0, K - 255) with p 0.125, else uniform in [0, K]. Score up to 256
decisions from j; EOS (decision K) is included exactly when the window reaches it.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .cache import CacheIndex, load_chart
from .common import ContractError, GridArrays, grid_from_arrays
from .conditions import draw_track
from .features import Chart, Interval
from .labels import load_star_labels

WINDOW = 256


@dataclass
class Draw:
    sha: str
    start: int
    stop: int
    track: tuple
    record: dict


def chart_from_cache(dec) -> Chart:
    grid = GridArrays.from_grid(grid_from_arrays(dec.grid_segments, dec.grid_bars))
    return Chart(dec.head_ms, dec.song_ms, grid, dec.actions.astype(np.int64), dec.gap_release_ms)


class Corpus:
    def __init__(self, cache_root, role: str, *, star_conditions: bool, capacity: int = 128):
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
        self.star_conditions = star_conditions
        self.star_labels, self.star_complete = load_star_labels(self.root) if star_conditions else ({}, False)
        self.capacity = capacity
        self._charts: OrderedDict = OrderedDict()

    def chart(self, sha: str) -> Chart:
        if sha in self._charts:
            self._charts.move_to_end(sha)
            return self._charts[sha]
        chart = chart_from_cache(load_chart(self.root / 'charts' / self.files[sha]))
        self._charts[sha] = chart
        if len(self._charts) > self.capacity:
            self._charts.popitem(last=False)
        return chart

    def draw(self, rng, window: int = WINDOW) -> Draw:
        group = self.groups[int(rng.integers(0, len(self.groups)))]
        members = self.members[group]
        sha = members[int(rng.integers(0, len(members)))]
        chart = self.chart(sha)
        K = chart.K
        u = rng.random()
        if u < 0.125:
            start = 0
        elif u < 0.25:
            start = max(0, K - (window - 1))
        else:
            start = int(rng.integers(0, K + 1))
        stop = min(start + window, K + 1)
        track, record = draw_track(chart, self.star_labels.get(sha), rng, star_conditions=self.star_conditions)
        return Draw(sha, start, stop, track, record)


def track_to_json(track):
    return [[iv.kind, iv.a, iv.b, iv.value] for iv in track]


def track_from_json(data):
    return tuple(Interval(int(k), float(a), float(b), float(v)) for k, a, b, v in data)


def build_manifest(cache_root, *, star_conditions: bool, seed: int = 954, windows: int = 64, charts: int = 4,
                   max_rows: int = 600):
    corpus = Corpus(cache_root, 'fit_dev', star_conditions=star_conditions)
    rng = np.random.default_rng(seed)
    groups = list(corpus.groups)
    chosen = [groups[i] for i in rng.choice(len(groups), size=min(windows, len(groups)), replace=False)]
    out = []
    for g in chosen:
        members = corpus.members[g]
        sha = members[int(rng.integers(0, len(members)))]
        chart = corpus.chart(sha)
        K = chart.K
        u = rng.random()
        start = 0 if u < 0.125 else max(0, K - (WINDOW - 1)) if u < 0.25 else int(rng.integers(0, K + 1))
        track, _ = draw_track(chart, corpus.star_labels.get(sha), rng, star_conditions=star_conditions)
        out.append(dict(sha256=sha, group_id=g, start=start, stop=min(start + WINDOW, K + 1),
                        track=track_to_json(track)))
    short = corpus.table[(corpus.table.K <= max_rows) & (corpus.table.K >= 50)]
    short_groups = sorted(short.group_id.unique())
    picks = []
    for i in rng.permutation(len(short_groups))[:charts]:
        rows = short[short.group_id == short_groups[i]].sort_values('sha256')
        picks.append(rows.sha256.iloc[int(rng.integers(0, len(rows)))])
    return dict(seed=seed, role='fit_dev', star_conditions=star_conditions, windows=out, freerun=sorted(picks),
                cache_index_sha256=corpus.index.summary.get('index_sha256'))


def load_or_build_manifest(cache_root, path: Path, *, star_conditions: bool):
    path = Path(path)
    if path.exists():
        data = json.loads(path.read_text())
        if data.get('star_conditions') == star_conditions:
            return data
    data = build_manifest(cache_root, star_conditions=star_conditions)
    path.write_text(json.dumps(data, indent=1))
    return data
