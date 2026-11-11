import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from ensomi_n.mapformat import MusicMap


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[2] / 'scripts/n1/smc_benchmark.py'
    spec = importlib.util.spec_from_file_location('smc_benchmark', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gate_rejects_missing_failed_and_stale_results(runner, tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('[]')
    with pytest.raises(RuntimeError, match='gate must pass'):
        runner.require_gate(tmp_path)
    gate = tmp_path / 'gate.json'
    gate.write_text(json.dumps(dict(passed=False)))
    with pytest.raises(RuntimeError, match='gate must pass'):
        runner.require_gate(tmp_path)
    gate.write_text(json.dumps(dict(passed=True, manifest_sha256=runner.sha256(manifest))))
    runner.require_gate(tmp_path)
    manifest.write_text('[{}]')
    with pytest.raises(RuntimeError, match='manifest changed'):
        runner.require_gate(tmp_path)


def test_beat_only_ablation_has_matching_decoder_inputs(runner, tmp_path, monkeypatch):
    import ensomi_n.v0.decode as decoder
    path = tmp_path / 'activations/N1-s0'
    path.mkdir(parents=True)
    beat, down = np.full(4000, .9), np.full(4000, .1)
    np.savez(path / 'smc_001.npz', beat=beat, down=down)
    seen = []

    def decode(evidence, config):
        seen.append(evidence)
        assert config.reset_penalty == 14
        return MusicMap.from_beats([5123.45, 40000]), dict(n_beats=2)

    monkeypatch.setattr(decoder, 'decode', decode)
    row = dict(key='smc_001', duration=40)
    result = runner.decode_one((tmp_path, 'N1-s0', 'beatonly', row))
    assert result['arm'] == 'N1-s0-beatonly'
    assert result['outside_audio'] == 1
    assert seen[0].phase is None and seen[0].logp is None
    assert seen[0].level is None and seen[0].boundary is None and seen[0].beat_off is None
    np.testing.assert_array_equal(seen[0].beat, beat)
    np.testing.assert_array_equal(seen[0].down, down)
    np.testing.assert_allclose(np.loadtxt(tmp_path / 'beats/N1-s0-beatonly/smc_001.beats'), [5.12345, 40])
