import numpy as np
import torch

from ensomi_model.audio_rows.figures import POSITIONS
from ensomi_model.audio_rows.net import FigureNet, NetConfig
from ensomi_model.audio_rows.rowsgen import decode
from ensomi_model.export.recording import RandomSource, Recorder, Recording
from ensomi_model.r2.common import GridArrays, grid_from_arrays
from ensomi_model.r2.model import R2Config, R2Model
from ensomi_model.system.arrange import arrange


def test_recording_snapshots_mutated_tensor_and_reuses_unchanged_bytes(tmp_path):
    rec = Recorder(tmp_path)
    value = torch.tensor([1., 2.])
    first = rec.put('first', value)
    repeated = rec.put('repeated', value)
    value.add_(3.)
    changed = rec.put('changed', value)
    rec.close()
    stored = Recording(tmp_path)
    np.testing.assert_array_equal(stored.get(first), [1., 2.])
    np.testing.assert_array_equal(stored.get(changed), [4., 5.])
    assert stored.index['arrays'][first]['byte_offset'] == stored.index['arrays'][repeated]['byte_offset']
    assert stored.index['arrays'][first]['byte_offset'] != stored.index['arrays'][changed]['byte_offset']


def test_recorded_multinomial_preserves_choice_and_generator(tmp_path):
    rec = Recorder(tmp_path)
    random = RandomSource(recorder=rec)
    a, b = torch.Generator().manual_seed(19), torch.Generator().manual_seed(19)
    for i, n in enumerate((3, 625, 8853)):
        p = torch.arange(1, n + 1).float().pow(-.5)
        p /= p.sum()
        assert random.multinomial(p, a, str(i)) == int(torch.multinomial(p, 1, generator=b))
        assert torch.equal(a.get_state(), b.get_state())
    rec.close()
    replay = RandomSource(replay=Recording(tmp_path))
    for i, n in enumerate((3, 625, 8853)):
        p = torch.arange(1, n + 1).float().pow(-.5)
        p /= p.sum()
        replay.multinomial(p, a, str(i))
    assert replay.differences == []


def test_recorder_preserves_rows_and_arrangement(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(12)
    members = np.zeros((3, len(POSITIONS)), np.float32)
    members[1, 0] = 1
    members[2, :2] = 1
    cfg = NetConfig(vocab=3, enc=4, dilations=(1,), ctx=8, ctx_layers=1, ctx_heads=2,
                    window=4, hidden=12, fig=4, heads=2, lags=4, key=2)
    model = FigureNet(cfg, members).eval()
    record = dict(vocab=[0, 1, 3], label_mean=2., label_std=1.)
    mel = np.random.default_rng(3).normal(size=(1200, 130)).astype(np.float16)
    g = GridArrays.from_grid(grid_from_arrays(np.array([[0., 500., 4.]]), np.array([[0., 4.]])))
    song = 12000.
    heads, beats = decode(model, record, mel, g, song, 2., seed=3)
    rec = Recorder(tmp_path)
    random = RandomSource(recorder=rec)
    recorded_heads, recorded_beats = decode(model, record, mel, g, song, 2., seed=3,
                                            recorder=rec, random_source=random)
    np.testing.assert_array_equal(recorded_heads, heads)
    np.testing.assert_array_equal(recorded_beats['ids'], beats['ids'])
    head = np.arange(500., 11000., 250.)
    r2 = R2Model(R2Config(hidden=16, levels=2, expansion=1, rank=4, memory='none')).eval()
    kwargs = dict(seed=954, target=(.1, 1.4, .1), kappa=(3., 3., 3.))
    a, gap, _ = arrange(r2, head, song, g, **kwargs)
    ar, gr, _ = arrange(r2, head, song, g, recorder=rec, random_source=random, **kwargs)
    np.testing.assert_array_equal(ar, a)
    np.testing.assert_array_equal(gr, gap)
    summary = rec.close()
    assert summary['graph_calls']['rows_step'] == len(beats['ids'])
    assert summary['graph_calls']['r2_commit'] == len(head)
    assert summary['graph_calls']['r2_release'] > 0
    stored = Recording(tmp_path)
    assert stored.get(stored.index['arrays'].index(next(x for x in stored.index['arrays']
                                                        if x['name'].endswith('/history_token')))).dtype == np.float32
