import stat
import zipfile

import numpy as np
import pytest

from ensomi_n.mapformat import MusicMap
from ensomi_n.smc import (events, extract_data, heldout_shifts, map_beats,
                          minimal_peaks, paired_bootstrap, score, shift_curve,
                          tempo_diagnostics)
from ensomi_n.timing import Segment


@pytest.fixture
def reference():
    pytest.importorskip('mir_eval')
    return np.arange(0, 40, .5)


def test_absolute_scoring_fixtures(reference):
    perfect = score(reference, reference)
    for metric in ('F-measure', 'CMLc', 'CMLt', 'AMLc', 'AMLt'):
        assert perfect[metric] == 1
    assert score(reference, reference + .25)['F-measure'] == 0
    double = score(reference, np.arange(0, 40, .25))
    assert double['F-measure'] == pytest.approx(2/3)
    assert double['CMLt'] == 0
    # The doubled sequence has one final half-beat beyond the reference end.
    assert double['AMLt'] == pytest.approx(139/140)
    assert score(reference, np.arange(0, 39.5001, .25))['AMLt'] == 1


def test_trim_and_shift_sign(reference):
    # The only error is before 5 s; beats exactly at 5 s are retained.
    pred = reference.copy()
    pred[1] += .2
    assert score(reference, pred)['F-measure'] == 1
    late = reference + .08
    assert score(reference, late, shift_ms=-80)['F-measure'] == 1
    assert shift_curve(reference, late)[0] == 1
    assert score(reference, late, shift_ms=80)['F-measure'] == 0


def test_map_clock_preserves_fractional_boundary_level_and_end():
    # A second segment starts off the first segment's grid and at a new level.
    beats = [-100, 100.125, 568.875, 1037.625, 1311.2, 1561.2, 1811.2, 2000]
    m = MusicMap.from_beats(beats, segments=[Segment(100.125, 468.75, 4), Segment(1311.2, 250, 3)])
    times, dropped = map_beats(m, 2)
    assert times.tolist() == [b / 1000 for b in beats[1:]]
    assert times[3] == pytest.approx(1.3112, abs=1e-12)
    assert dropped == 2


def test_heldout_shift_never_uses_scored_fold():
    curves = np.array([[1, 0, 0], [0, 0, 1], [0, 0, 1]], dtype=float)
    folds = [0, 1, 1]
    assert heldout_shifts(curves, folds, [-5, 0, 5]).tolist() == [5, -5, -5]
    curves[0] = [0, 0, 100]
    assert heldout_shifts(curves, folds, [-5, 0, 5])[0] == 5
    assert np.all(heldout_shifts(np.ones((3, 3)), folds, [-5, 0, 5]) == 0)
    with pytest.raises(ValueError):
        heldout_shifts(curves, [0, 0, 0], [-5, 0, 5])


def test_peak_support_threshold_and_plateau():
    pytest.importorskip('scipy')
    p = np.zeros(60)
    p[0] = .7
    p[10:13] = .9
    p[30] = .5
    p[40], p[46] = .8, .9
    assert minimal_peaks(p).tolist() == [0, .105, .12, .46]


def test_peak_picker_matches_beat_this():
    pytest.importorskip('torch')
    pytest.importorskip('beat_this')
    import torch
    from beat_this.model.postprocessor import Postprocessor
    rng = np.random.default_rng(7)
    logits = rng.normal(size=1000).astype(np.float32)
    logits[20:27] = 10
    expected, _ = Postprocessor()(torch.tensor(logits), torch.tensor(logits))
    actual = minimal_peaks(1 / (1 + np.exp(-logits)), fps=50)
    np.testing.assert_array_equal(actual, expected)


def test_paired_null_and_tempo_classes(reference):
    assert paired_bootstrap([1, 2, 3], [1, 2, 3]) == dict(n=3, difference=0, lo=0, hi=0)
    assert tempo_diagnostics(reference, reference + .25)['offbeat']
    assert tempo_diagnostics(reference, np.arange(0, 40, .25))['ratio_class'] == 'double'


@pytest.mark.parametrize('beats', [[1, 1], [2, 1], [np.nan], [-.1], [[1, 2]]])
def test_invalid_beats_are_not_silently_repaired(beats):
    with pytest.raises(ValueError):
        events(beats)


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'a/../../escape', 'a\\escape', 'C:/escape'])
def test_archive_traversal_rejected_before_writing(tmp_path, name):
    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('ok.txt', 'safe')
        z.writestr(name, 'bad')
    with pytest.raises(ValueError):
        extract_data(archive, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_archive_symlink_budget_and_existing_destination(tmp_path):
    archive = tmp_path / 'bad.zip'
    info = zipfile.ZipInfo('link')
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr(info, '/outside')
    with pytest.raises(ValueError):
        extract_data(archive, tmp_path / 'out')
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('a/data.txt', '1234')
    with pytest.raises(ValueError):
        extract_data(archive, tmp_path / 'out', max_bytes=3)
    extract_data(archive, tmp_path / 'out')
    assert (tmp_path / 'out/a/data.txt').read_text() == '1234'
    with pytest.raises(FileExistsError):
        extract_data(archive, tmp_path / 'out')
