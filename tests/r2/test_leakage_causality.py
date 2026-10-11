"""Pre-run tests 3 and 4: release leakage and causality."""
import ast
from pathlib import Path

import numpy as np
import torch

import ensomi_model.r2 as r2
from ensomi_model.r2.common import ACTIONS
from ensomi_model.r2.features import Chart, frames, history_tokens, query_features
from ensomi_model.r2.state import action_support_mask

from .helpers import chart_from_objects, hold, random_decisions, random_skeleton, sample_track, tap, tiny_model, two_segment_grid


def source_objects(rng, K=120):
    head, song = random_skeleton(rng, K, lo=60, hi=400)
    objects, free_at = [], [0.0] * 4
    for t in head:
        lanes = [l for l in range(4) if free_at[l] < t]
        if not lanes:
            continue
        for lane in rng.choice(lanes, size=min(len(lanes), int(rng.integers(1, 3))), replace=False):
            if rng.random() < 0.4:
                later = head[head > t]
                if len(later) >= 2:
                    e = float(rng.uniform(t + 5, later[min(len(later) - 1, int(rng.integers(1, 5)))]))
                    e = float(np.floor(e))
                    if e > t:
                        objects.append(hold(t, e, int(lane)))
                        free_at[lane] = e
                        continue
            objects.append(tap(t, int(lane)))
            free_at[lane] = t
    return objects, max(song, max(o.end_time_ms for o in objects) + 50.0)


def shifted(objects, rng, song):
    by_lane = {l: sorted([o for o in objects if o.lane == l], key=lambda o: o.start_time_ms) for l in range(4)}
    out = []
    for lane, objs in by_lane.items():
        for i, o in enumerate(objs):
            if o.end_time_ms > o.start_time_ms:
                nxt = objs[i + 1].start_time_ms if i + 1 < len(objs) else song
                lo, hi = o.start_time_ms + 1, nxt - 1
                e = float(np.floor(rng.uniform(lo, hi))) if hi > lo else o.end_time_ms
                out.append(hold(o.start_time_ms, max(e, o.start_time_ms + 1), lane))
            else:
                out.append(o)
    return out


def test_release_shift_leaves_inputs_unchanged():
    rng = np.random.default_rng(7)
    for trial in range(6):
        objects, song = source_objects(rng)
        a, da = chart_from_objects(objects, song)
        b, db = chart_from_objects(shifted(objects, rng, song), song)
        assert np.array_equal(da.head_ms, db.head_ms) and np.array_equal(da.grid_segments, db.grid_segments)
        assert da.song_ms == db.song_ms
        differs = np.flatnonzero((a.actions != b.actions).any(1) | ~np.isclose(a.gap, b.gap, equal_nan=True).all(1))
        assert len(differs), 'shift changed no label'
        first = int(differs[0])
        K = a.K
        qa, qb = query_features(a, np.arange(K + 1)), query_features(b, np.arange(K + 1))
        # look-ahead (density and next gaps) are timing-only: identical for every decision
        assert np.array_equal(qa[..., 80:], qb[..., 80:])
        for k in range(1, K + 1):
            ca, cb = a.candidates(k), b.candidates(k)
            assert np.array_equal(ca.times, cb.times) and np.array_equal(ca.static, cb.static)
        for k in range(0, first + 1):
            assert np.array_equal(history_tokens(a, min(k, K)), history_tokens(b, min(k, K)))
            assert np.array_equal(qa[k], qb[k])
            assert np.array_equal(frames(a, (), [a.time(k)], k), frames(b, (), [b.time(k)], k))
        # the labels' own history tokens differ once the changed decision is committed
        N = min(K, first + 1)
        if first < K:
            assert not np.array_equal(history_tokens(a, N), history_tokens(b, N))


def test_r2_never_uses_release_aware_grid():
    root = Path(r2.__file__).parent
    for path in root.glob('*.py'):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or '').endswith('evaluation.case'), path
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'musical_grid':
                assert node.args, f'{path}: zero-argument .musical_grid() is the release-aware chart grid'


def test_future_label_does_not_change_earlier_log_probs():
    model = tiny_model(torch.float64)
    rng = np.random.default_rng(9)
    grid = two_segment_grid()
    checked = 0
    for trial in range(12):
        K = int(rng.integers(20, 140))
        head, song = random_skeleton(rng, K, lo=20, hi=300)
        chart = random_decisions(rng, (head, song, grid))
        k = int(rng.integers(1, K - 1))
        j = int(rng.integers(0, k + 1))
        acts, gap = chart.actions.copy(), chart.gap.copy()
        prefix = chart.with_decisions(acts[:k + 1], gap[:k + 1])
        held = prefix.derived().held[k + 1]
        choices = [a for a in np.flatnonzero(action_support_mask(held, k + 1 == K))
                   if not np.array_equal(ACTIONS[a], acts[k + 1])]
        new = ACTIONS[rng.choice(choices)]
        acts[k + 1] = new
        gap[k + 1] = np.nan
        cands = prefix.candidates(k + 1).times
        for lane in range(4):
            if held[lane] and new[lane] in (2, 3, 4):
                gap[k + 1, lane] = cands[rng.integers(0, len(cands))]
        track = sample_track(rng, chart) if trial % 2 else ()
        with torch.no_grad():
            before = model.window(chart.with_decisions(chart.actions[:k + 2], chart.gap[:k + 2]), j, k + 2, track)
            after = model.window(Chart(head, song, grid, acts[:k + 2], gap[:k + 2]), j, k + 2, track)
            full = model.window(chart, j, k + 1, track)
        n = k + 1 - j
        assert torch.allclose(before.total[:n], after.total[:n], rtol=0, atol=1e-12)
        assert torch.allclose(before.total[:n], full.total[:n], rtol=0, atol=1e-12)
        checked += n
    assert checked > 0
