"""N1 v0 evaluation pieces on synthetic references: oracle evidence through the decoder, BeatThis-style
evidence with its lag, R2's exported red lines, per-stratum scoring and the paired bootstrap."""
from __future__ import annotations

import json

import numpy as np

from ensomi_n.eval.strata import recording_strata, set_strata
from ensomi_n.timing import Reference, Segment
from ensomi_n.v0 import evaluate as E


def _ref(segs, hi=90000.0, step=125.0, key='k'):
    return Reference.from_segments(segs, np.arange(segs[0].start_ms, hi, step), key=key)


def _item(ref, rec='r'):
    st = set_strata(ref, (), key=ref.key)
    rs = recording_strata([st])
    return dict(key=ref.key, rec=rec, ref=ref, source='ranked', literal=False,
                strata={n: s[ref.key] for n, s in rs.items()})


def test_oracle_through_decoder_scores_one():
    from ensomi_n.v0.decode import decode
    ref = _ref([Segment(600.0, 468.75), Segment(30600.0 + 0.37 * 468.75, 468.75), Segment(60000.0, 375.0)])
    n = int(95000 / 10)
    mmap, diag = decode(E.oracle_evidence(ref, n))
    sc = E.score_item(ref, _item(ref)['strata'], mmap)
    assert sc['all'].bl20 > 0.999
    assert sc['all'].reset_recall == 1.0


def test_bt_evidence_undoes_the_lag(tmp_path):
    beats = np.arange(1000.0, 60000.0, 500.0)
    frames = np.zeros(3100)
    late = (beats + 30.0) / 20.0              # BeatThis peaks 30 ms late at 50 fps
    for p in late:
        i = int(np.floor(p))
        frames[i] += 1 - (p - i)
        frames[i + 1] += p - i
    logit = np.log(np.clip(frames, 1e-4, 1 - 1e-4) / (1 - np.clip(frames, 1e-4, 1 - 1e-4)))
    np.savez(tmp_path / 'x.npz', beat=logit, down=logit)
    ev = E.bt_evidence(tmp_path / 'x.npz', 30.0, 6100)
    peaks = np.flatnonzero((ev.beat[1:-1] > ev.beat[:-2]) & (ev.beat[1:-1] >= ev.beat[2:]) & (ev.beat[1:-1] > 0.4)) + 1
    assert np.allclose(peaks[:10] * 10.0, beats[:10], atol=10.0)


def test_r2_map_and_paired(tmp_path):
    p = tmp_path / 'r2.json'
    p.write_text(json.dumps(dict(red_lines=[[600.0, 500.0, 4], [600.0, 500.0, 4], [30600.0, 400.0, 4]])))
    m = E.r2_map(p)
    assert len(m.segments) == 2 and m.segments[1].beat_ms == 400.0
    refs = [_ref([Segment(600.0, 500.0)], key=f'k{i}') for i in range(4)]
    items = [_item(r, rec=f'r{i // 2}') for i, r in enumerate(refs)]
    good = {x['key']: E.score_item(x['ref'], x['strata'], x['ref'].timing) for x in items}
    shifted = {x['key']: E.score_item(x['ref'], x['strata'], x['ref'].timing.shifted(250.0)) for x in items}
    d = E.paired(items, good, shifted)
    assert abs(d['all']['delta'] - 1.0) < 1e-9 and d['all']['n_recordings'] == 2
    t = E.aggregate_arm(items, good, bootstrap=50)
    assert abs(t['all']['bl20']['value'] - 1.0) < 1e-9
    assert 'S0' in t
