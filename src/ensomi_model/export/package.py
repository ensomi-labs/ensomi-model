"""The package manifest ``ensomi-package.json``: the released model's identity, licence and files.

The Ensomi runtime owns the format, ``ensomi-package/1``, and reads it from runtime 0.3.0. The manifest sits
at the package root beside ``runtime.json`` and lists every file the package needs except itself, with byte
size and SHA-256. The ONNX exporter writes it for the files it wrote. For a package exported earlier, this
command derives the files from ``runtime.json`` by the runtime's rule and writes the manifest beside it:

    python -m ensomi_model.export.package DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FORMAT = 'ensomi-package/1'
MANIFEST = 'ensomi-package.json'
CONFIG = 'runtime.json'
# The released model's identity; the model service reports the same name and version.
MODEL_ID = 'ensomi-r2'
MODEL_NAME = 'Ensomi R2'
MODEL_VERSION = 'r2-1.0'
LICENSE = 'CC-BY-NC-SA-4.0 AND MIT'
PIPELINE = dict(id='r2', version=1)
MIN_RUNTIME = '0.3.0'


def check_path(path):
    if '\\' in path or any(part in ('', '.', '..') for part in path.split('/')):
        raise ValueError(f'{path!r}: package paths are relative and /-separated, without empty, . or .. parts')
    return path


def file_entry(root, path):
    data = (root / path).read_bytes()
    return dict(path=path, size=len(data), sha256=hashlib.sha256(data).hexdigest())


def write_manifest(root, files, notices=()):
    """Write ``root/ensomi-package.json`` and return it as a dict.

    ``files`` and ``notices`` are paths relative to ``root``; ``runtime.json`` and every notice are listed
    in ``files`` whether or not ``files`` names them. Each listed file is read for its size and SHA-256.
    The same files give the same bytes.
    """
    root = Path(root)
    paths = sorted({check_path(p) for p in (CONFIG, *files, *notices)})
    manifest = dict(format=FORMAT, model=dict(id=MODEL_ID, name=MODEL_NAME, version=MODEL_VERSION),
                    license=LICENSE, notices=sorted(set(notices)), pipeline=PIPELINE, min_runtime=MIN_RUNTIME,
                    config=CONFIG, files=[file_entry(root, p) for p in paths])
    (root / MANIFEST).write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def config_files(config):
    """The files ``runtime.json`` names, other than notices, as ensomi-runtime 0.2 reads them.

    The graph files are what ``model_files`` (``crates/ffi/src/generator.rs``) adds to
    ``Models::required_files`` (``crates/generate/src/models.rs``): the Mel filters and Hann windows of
    both front ends, the initial R2 state and the scope-table columns.
    """
    files = [graph['file'] for graph in config['graphs'].values()]
    for side in ('beatthis', 'rows'):
        for array in ('hann_window', 'mel_filter'):
            files.append(config['frontend_arrays'][side][array]['path'])
    files += [spec['file'] for spec in config['state']['r2_initial'].values()]
    files += [column['file'] for column in config['calibration']['scope_table']['columns']]
    return files


def notice_files(config):
    """The licence notices ``runtime.json`` names: BeatThis's MIT licence at ``frontend_arrays.beatthis_license``."""
    notice = config['frontend_arrays'].get('beatthis_license')
    return [notice] if notice else []


def write_package_manifest(root):
    """Write the manifest of an exported package directory from the files its ``runtime.json`` names."""
    config = json.loads((Path(root) / CONFIG).read_text())
    return write_manifest(root, config_files(config), notice_files(config))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('package', type=Path, help='package directory holding runtime.json')
    a = p.parse_args(argv)
    if not (a.package / CONFIG).is_file():
        p.error(f'{a.package} has no {CONFIG}')
    manifest = write_package_manifest(a.package)
    print(json.dumps(dict(files=len(manifest['files']), bytes=sum(f['size'] for f in manifest['files']))))


if __name__ == '__main__':
    main()
