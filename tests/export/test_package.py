import hashlib
import json
import subprocess
import sys

import pytest

from ensomi_model.export import package
from ensomi_model.export.package import write_manifest


def make_package(root):
    """A small exported package: two graphs, the four front-end arrays, two state arrays, two scope columns
    (one upper-case, which sorts first), the notice, and two files outside the package."""
    config = dict(
        format='ensomi-onnx-runtime-v1',
        graphs=dict(rows_step=dict(file='rows_step.onnx'), beatthis=dict(file='beatthis.onnx')),
        frontend_arrays=dict(
            beatthis=dict(mel_filter=dict(path='beatthis-mel-filter.bin'), hann_window=dict(path='beatthis-hann-window.bin')),
            rows=dict(mel_filter=dict(path='rows-mel-filter.bin'), hann_window=dict(path='rows-hann-window.bin')),
            beatthis_license='BeatThis-LICENSE.txt'),
        state=dict(r2_initial=dict(rows=dict(file='r2_initial_rows.bin'), last=dict(file='r2_initial_last.bin'))),
        calibration=dict(scope_table=dict(columns=[dict(name='band', file='scopes-band.bin'),
                                                   dict(name='S', file='scopes-S.bin')])))
    (root / 'runtime.json').write_text(json.dumps(config))
    for name in ('rows_step.onnx', 'beatthis.onnx', 'beatthis-mel-filter.bin', 'beatthis-hann-window.bin',
                 'rows-mel-filter.bin', 'rows-hann-window.bin', 'BeatThis-LICENSE.txt', 'r2_initial_rows.bin',
                 'r2_initial_last.bin', 'scopes-band.bin', 'scopes-S.bin', 'rows-smoke.json', 'LICENSE-beat_this'):
        (root / name).write_bytes(name.encode() * len(name))
    return root


def test_fields_in_format_order(tmp_path):
    root = make_package(tmp_path)
    write_manifest(root, ['beatthis.onnx'], ['BeatThis-LICENSE.txt'])
    manifest = json.loads((root / package.MANIFEST).read_text())
    assert list(manifest) == ['format', 'model', 'license', 'notices', 'pipeline', 'min_runtime', 'config', 'files']
    assert manifest['format'] == package.FORMAT
    assert list(manifest['model'].items()) == [('id', package.MODEL_ID), ('name', package.MODEL_NAME),
                                               ('version', package.MODEL_VERSION)]
    assert manifest['license'] == package.LICENSE
    assert list(manifest['pipeline'].items()) == list(package.PIPELINE.items())
    assert (manifest['min_runtime'], manifest['config']) == (package.MIN_RUNTIME, package.CONFIG)
    assert all(list(entry) == ['path', 'size', 'sha256'] for entry in manifest['files'])


def test_sizes_hashes_sorted_paths_and_notices(tmp_path):
    root = make_package(tmp_path)
    manifest = write_manifest(root, ['scopes-band.bin', 'beatthis.onnx', 'scopes-S.bin'], ['BeatThis-LICENSE.txt'])
    paths = [entry['path'] for entry in manifest['files']]
    assert paths == ['BeatThis-LICENSE.txt', 'beatthis.onnx', 'runtime.json', 'scopes-S.bin', 'scopes-band.bin']
    assert set(manifest['notices']) <= set(paths)
    for entry in manifest['files']:
        data = (root / entry['path']).read_bytes()
        assert entry['size'] == len(data)
        assert entry['sha256'] == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize('path', ['/beatthis.onnx', '../beatthis.onnx', 'a/../beatthis.onnx', 'a//beatthis.onnx',
                                  './beatthis.onnx', 'a\\beatthis.onnx', ''])
def test_paths_are_relative_and_slash_separated(tmp_path, path):
    with pytest.raises(ValueError):
        write_manifest(make_package(tmp_path), [path])


def test_nested_path(tmp_path):
    root = make_package(tmp_path)
    (root / 'graphs').mkdir()
    (root / 'graphs' / 'r2.onnx').write_bytes(b'r2')
    assert 'graphs/r2.onnx' in [entry['path'] for entry in write_manifest(root, ['graphs/r2.onnx'])['files']]


def test_second_run_gives_the_same_bytes(tmp_path):
    root = make_package(tmp_path)
    write_manifest(root, ['beatthis.onnx', 'scopes-S.bin'], ['BeatThis-LICENSE.txt'])
    first = (root / package.MANIFEST).read_bytes()
    write_manifest(root, ['scopes-S.bin', 'beatthis.onnx'], ['BeatThis-LICENSE.txt'])
    assert (root / package.MANIFEST).read_bytes() == first
    text = first.decode()
    assert text == json.dumps(json.loads(text), indent=2) + '\n'


def test_directory_command_lists_the_runtime_files(tmp_path):
    root = make_package(tmp_path)
    # A fresh interpreter, so the import check sees only what the command loads.
    check = ('import sys; from ensomi_model.export.package import main; main([sys.argv[1]]); '
             'assert not {"torch", "onnx"} & set(sys.modules), sys.modules.keys() & {"torch", "onnx"}')
    subprocess.run([sys.executable, '-c', check, str(root)], check=True, capture_output=True)
    manifest = json.loads((root / package.MANIFEST).read_text())
    assert [entry['path'] for entry in manifest['files']] == [
        'BeatThis-LICENSE.txt', 'beatthis-hann-window.bin', 'beatthis-mel-filter.bin', 'beatthis.onnx',
        'r2_initial_last.bin', 'r2_initial_rows.bin', 'rows-hann-window.bin', 'rows-mel-filter.bin',
        'rows_step.onnx', 'runtime.json', 'scopes-S.bin', 'scopes-band.bin']
    assert manifest['notices'] == ['BeatThis-LICENSE.txt']
