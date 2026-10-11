"""Pre-run tests 5 (split firewall) and 6 (cache -> decisions -> advance -> .osu -> parse round trip)."""
import json
from pathlib import Path

import numpy as np
import pytest

from ensomi_model.evaluation.legality import violations
from ensomi_model.osu_core.hitobjects import parse_mania_hit_objects
from ensomi_model.r2.cache import load_chart, process_row
from ensomi_model.r2.export import export_chart, minimal_header
from ensomi_model.r2.splits import SplitError, role_of, verify_assignment

from .helpers import chart_from_objects
from .test_leakage_causality import source_objects

CACHE = Path('artifacts/r2-cache/v1')


def test_builder_refuses_non_fit_rows_before_opening_files(tmp_path):
    for split in ('calibration', 'heldout'):
        row = dict(eval_split=split, path='does/not/exist.osu', set_dir='does/not', audio_filename='x.mp3',
                   audio_present=True, sha256='0' * 64, group_id='g', star=3.0)
        with pytest.raises(SplitError):
            process_row(row, str(tmp_path), str(tmp_path))


def test_fit_dev_rule_is_the_salted_hash():
    import hashlib
    for g in ('a', 'song:123', '757712db069dfab3'):
        digest = hashlib.sha256(('r2-fit-dev-v1:' + g).encode()).digest()
        assert role_of(g) == ('fit_dev' if int.from_bytes(digest[:8], 'big') % 10 == 0 else 'fit_train')


@pytest.mark.skipif(not (CACHE / 'splits.json').exists(), reason='cache not built here')
def test_stored_split_manifest_reproduces():
    assert verify_assignment(CACHE / 'splits.json') > 4000
    import pyarrow.parquet as pq
    index = pq.read_table(CACHE / 'index.parquet').to_pandas()
    assert all(role_of(g) == r for g, r in zip(index.group_id, index.role))


def roundtrip(head_ms, song_ms, actions, gap, grid_segments, tmp_path, name):
    objects, text = export_chart(head_ms, song_ms, actions, gap, tmp_path / name, minimal_header(grid_segments))
    parsed = parse_mania_hit_objects(tmp_path / name)
    assert not violations(parsed, song_span=(0.0, float(song_ms)))
    return parsed


def test_synthetic_roundtrip(tmp_path):
    rng = np.random.default_rng(4)
    for i in range(8):
        objects, song = source_objects(rng)
        chart, dec = chart_from_objects(objects, song)
        parsed = roundtrip(dec.head_ms, dec.song_ms, dec.actions, dec.gap_release_ms, dec.grid_segments, tmp_path,
                           f'c{i}.osu')
        assert sorted((o.start_time_ms, o.lane) for o in parsed) == sorted((o.start_time_ms, o.lane) for o in objects)
        src = sorted((o.start_time_ms, o.lane, o.end_time_ms) for o in objects)
        out = sorted((o.start_time_ms, o.lane, o.end_time_ms) for o in parsed)
        snap = np.abs(dec.gap_release_ms - dec.release_orig_ms)
        bound = float(np.nanmax(snap)) if np.isfinite(snap).any() else 0.0
        assert max(abs(a[2] - b[2]) for a, b in zip(src, out)) <= bound + 1e-9


@pytest.mark.skipif(not (CACHE / 'index.parquet').exists(), reason='cache not built here')
def test_real_cache_roundtrip(tmp_path):
    import pyarrow.parquet as pq
    from ensomi_model.r2.cache import Excluded  # noqa: F401
    index = pq.read_table(CACHE / 'index.parquet').to_pandas().sample(40, random_state=1)
    for _, row in index.iterrows():
        assert row.role in ('fit_train', 'fit_dev')  # the cache admits fit rows only
        dec = load_chart(CACHE / 'charts' / row.file)
        parsed = roundtrip(dec.head_ms, dec.song_ms, dec.actions, dec.gap_release_ms, dec.grid_segments, tmp_path,
                           row.file + '.osu')
        source = parse_mania_hit_objects(Path(row.path))
        merged = lambda t: dec.head_ms[int(np.searchsorted(dec.head_ms, t, side='right')) - 1]  # noqa: E731
        assert sorted((o.start_time_ms, o.lane) for o in parsed) == sorted((merged(o.start_time_ms), o.lane)
                                                                         for o in source)
        snap = np.abs(dec.gap_release_ms - dec.release_orig_ms)
        bound = float(np.nanmax(snap)) if np.isfinite(snap).any() else 0.0
        src = {(merged(o.start_time_ms), o.lane): o.end_time_ms for o in source}
        err = max(abs(src[(o.start_time_ms, o.lane)] - o.end_time_ms) for o in parsed)
        assert err <= bound + 1e-9, (row.path, err, bound)
        assert row.snap_max == pytest.approx(bound)
