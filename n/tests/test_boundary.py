"""The N line never imports the R line, and the root distribution leaves it out."""
from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import pytest

N_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = 'ensomi_model'


def _modules() -> list[Path]:
    return sorted((N_ROOT / 'src').rglob('*.py')) + sorted((N_ROOT / 'tests').rglob('*.py'))


def _forbidden(name: str | None) -> bool:
    return bool(name) and (name == FORBIDDEN or name.startswith(FORBIDDEN + '.'))


def test_no_module_imports_the_r_line():
    offenders = []
    for path in _modules():
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                offenders += [(path, a.name) for a in node.names if _forbidden(a.name)]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and _forbidden(node.module):
                offenders.append((path, node.module))
            elif isinstance(node, ast.Call):
                name = getattr(node.func, 'id', None) or getattr(node.func, 'attr', None)
                if name in ('__import__', 'import_module') and node.args:
                    arg = node.args[0]
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and _forbidden(arg.value):
                        offenders.append((path, arg.value))
    assert not offenders, offenders
    assert len(_modules()) > 8


def test_importing_every_module_loads_no_r_module(tmp_path):
    modules = ['ensomi_n'] + sorted(
        '.'.join(p.relative_to(N_ROOT / 'src').with_suffix('').parts).removesuffix('.__init__')
        for p in (N_ROOT / 'src').rglob('*.py'))
    code = ('import importlib, sys\n'
            f'for m in {modules!r}:\n'
            '    importlib.import_module(m)\n'
            f'bad = sorted(m for m in sys.modules if m == {FORBIDDEN!r} or m.startswith({FORBIDDEN!r} + "."))\n'
            'assert not bad, bad\n'
            'print(len(sys.modules))\n')
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path, capture_output=True, text=True,
                            env={'PYTHONPATH': str(N_ROOT / 'src'), 'PATH': ''})
    assert result.returncode == 0, result.stderr


def test_root_distribution_leaves_n_out():
    root = N_ROOT.parent / 'pyproject.toml'
    if not root.is_file():
        pytest.skip('not inside the ensomi-model checkout')
    tomllib = pytest.importorskip('tomllib')
    config = tomllib.loads(root.read_text(encoding='utf-8'))
    assert config['tool']['setuptools']['packages']['find']['where'] == ['src']
    assert config['tool']['pytest']['ini_options']['testpaths'] == ['tests']
    assert config['project']['name'] == 'ensomi-model'


def test_n_distribution_is_mit():
    tomllib = pytest.importorskip('tomllib')
    config = tomllib.loads((N_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    assert config['project']['name'] == 'ensomi-n'
    assert config['project']['license'] == 'MIT'
    assert config['project']['license-files'] == ['LICENSE']
    text = (N_ROOT / 'LICENSE').read_text(encoding='utf-8')
    assert text.startswith('MIT License') and 'Copyright (c) 2026 Ensomi Labs' in text
    assert all(not d.startswith('ensomi-model') for d in config['project']['dependencies'])
