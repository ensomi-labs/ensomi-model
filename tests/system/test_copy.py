import numpy as np
import torch

from ensomi_model.r2.common import GridArrays, grid_from_arrays
from ensomi_model.system.copy import (CopyBias, CopyConfig, CopyPath, jack_dial, key_code, match_rates, row_audio,
                                      timing_keys)
from ensomi_model.system.copytrain import load_model


def grid(beat_ms=500.0):
    return GridArrays.from_grid(grid_from_arrays(np.array([[0.0, beat_ms, 4]]), np.array([[0.0, 4]])))


def random_path(cfg, seed=0):
    torch.manual_seed(seed)
    model = CopyPath(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.3)
    return model.eval()


def test_key_code_tolerance():
    q = torch.tensor([[12, 12, 12, 12, 12, 12, 12, 12, 0]])
    k = torch.tensor([[12, 13, 12, 12, -1, 12, 12, 11, 47]])
    exact = int(key_code(q, k, 0)[0])
    tol = int(key_code(q, k, 1)[0])
    bits = lambda c: [(c >> i) & 1 for i in range(9)]
    assert bits(exact) == [1, 0, 1, 1, 0, 1, 1, 0, 0]
    assert bits(tol) == [1, 1, 1, 1, 0, 1, 1, 1, 1]       # undefined entries never match; phase is circular


def test_jack_dial_is_monotone_with_fixed_endpoints():
    rng = np.random.default_rng(3)
    lp = np.log(rng.dirichlet(np.ones(15)))
    b = rng.normal(0, 2, 15)
    prev = 0b0101
    jack = (np.arange(1, 16) & prev) != 0

    def mass(bias):
        p = np.exp(lp + bias)
        return p[jack].sum() / p.sum()

    masses = [mass(jack_dial(lp, b, prev, s)) for s in np.linspace(0, 1, 6)]
    assert abs(masses[0] - mass(np.zeros(15))) < 1e-9
    assert abs(masses[-1] - mass(b)) < 1e-9
    assert np.all(np.diff(masses) * np.sign(masses[-1] - masses[0]) >= -1e-12)
    within = lambda bias: np.exp(lp + bias)[jack] / np.exp(lp + bias)[jack].sum()
    assert np.allclose(within(jack_dial(lp, b, prev, 0.0)), within(b))   # inside the jack class the copy still acts
    assert np.array_equal(jack_dial(lp, b, 0, 0.0), b)                    # no previous row: nothing to keep


def test_exclusion_never_attends_to_the_last_rows():
    rng = np.random.default_rng(0)
    K = 120
    b = np.cumsum(rng.choice([0.25, 0.5, 1.0], K + 8))
    keys = torch.as_tensor(timing_keys(b, K))
    bt = torch.as_tensor(b, dtype=torch.float32)
    h = torch.as_tensor(rng.integers(1, 16, K))
    aud = torch.as_tensor(rng.normal(size=(K + 8, 2, 32)).astype(np.float32))
    lp = torch.log_softmax(torch.randn(K, 15), -1)
    model = random_path(CopyConfig(exclude_rows=2, exclude_beats=2.0))
    rows = torch.arange(10, K)
    trace = {}
    model.chart_bias(rows, h, bt, keys, aud, lp[rows], trace=trace)
    a = trace['attention'][0, :, :-1].sum(-1)            # [q, J]
    d, r = trace['d'][0], trace['r'][0]
    assert float(a[(r <= 2) | (d <= 2.008)].sum()) == 0.0
    assert float(a[(r > 2) & (d > 2.01) & (r > 0)].sum()) > 0


def test_runtime_bias_equals_teacher_forced_bias():
    rng = np.random.default_rng(1)
    K = 60
    head_ms = 1000 + np.cumsum(rng.choice([125.0, 250.0, 500.0], K))
    g = grid()
    mel = rng.normal(size=(int(head_ms[-1] / 10) + 200, 130)).astype(np.float16)
    model = random_path(CopyConfig())
    cb = CopyBias(model, head_ms, g, mel, strength=1.0, record=True)
    hist = rng.integers(1, 16, K)
    lp = np.log(rng.dirichlet(np.ones(15), size=K))
    online = np.stack([cb(k, hist, lp[k]) for k in range(K)])
    b = g.beat(head_ms)
    batch = model.chart_bias(torch.arange(K), torch.as_tensor(hist), torch.as_tensor(b, dtype=torch.float32),
                             torch.as_tensor(timing_keys(b, K)), torch.as_tensor(row_audio(mel, head_ms, g)),
                             torch.as_tensor(lp, dtype=torch.float32)).detach().numpy()
    assert np.allclose(online[1:], batch[1:], atol=1e-5)
    assert len(cb.rows) == K - 1 and 0 <= cb.rows[-1]['null'] <= 1


def test_row_audio_is_unit_and_repeats_with_the_sound():
    g = grid(500.0)
    F = 3000
    mel = np.zeros((F, 128), np.float32)
    pattern = np.random.default_rng(2).normal(size=(50, 128))
    for start in range(0, F - 50, 400):          # the same 0.5 s sound every 4 s (8 beats)
        mel[start:start + 50] = pattern
    head_ms = np.array([100.0, 2100.0, 4100.0, 6100.0])
    a = row_audio(mel, head_ms, g)
    assert a.shape == (4, 2, 32)
    assert np.allclose(np.linalg.norm(a, axis=2), 1, atol=1e-3)
    assert float(a[0, 0] @ a[2, 0]) > 0.99          # 4 s apart: same sound
    assert float(a[0, 0] @ a[1, 0]) < 0.9           # 2 s apart: silence against sound


def test_match_rates_on_repeating_rows():
    b = np.concatenate([np.array([0, 0.5, 0.75, 1, 2, 2.5, 3, 3.25]) + 4 * i for i in range(12)])
    m = match_rates(b, tol=1)
    assert m['exact'] > 0.8 and m['tolerant'] >= m['exact'] and m['partial7'] >= m['tolerant']


def test_study_checkpoint_loads_as_the_old_config(tmp_path):
    old = random_path(CopyConfig.old())
    path = tmp_path / 'full.pt'
    torch.save(dict(state=old.state_dict(), args=dict(arch='copy', variant='full', heads=8, hidden=32), step=3), path)
    model = load_model(path).cpu()
    assert model.cfg == CopyConfig.old()
    for (n1, p1), (n2, p2) in zip(old.state_dict().items(), model.state_dict().items()):
        assert n1 == n2 and torch.equal(p1, p2.cpu())


def test_gradients_reach_audio_and_key_embeddings():
    rng = np.random.default_rng(4)
    K = 40
    b = np.cumsum(rng.choice([0.5, 1.0], K + 8))
    model = CopyPath(CopyConfig(exclude_rows=0, exclude_beats=0))
    with torch.no_grad():
        model.gate.weight.normal_(0, 0.1)
    rows = torch.arange(5, K)
    lp = torch.log_softmax(torch.randn(len(rows), 15), -1)
    bias = model.chart_bias(rows, torch.as_tensor(rng.integers(1, 16, K)), torch.as_tensor(b, dtype=torch.float32),
                            torch.as_tensor(timing_keys(b, K)),
                            torch.as_tensor(rng.normal(size=(K + 8, 2, 32)).astype(np.float32)), lp)
    bias.pow(2).sum().backward()
    assert model.w_audio.weight.grad.abs().sum() > 0 and model.e_key.weight.grad.abs().sum() > 0
