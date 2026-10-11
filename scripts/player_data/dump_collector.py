#!/usr/bin/env python
"""Collect osu!mania player data from the monthly data.ppy.sh performance dumps.

Runs on bings-mac from the repository root (paths below are relative to it). Idempotent: each
run fetches the data.ppy.sh listing once, finds mania `random_10000` dumps that are not yet
collected, and for each one, oldest first:

1. HEAD for size, Last-Modified and ETag; checks free disk (stops below --min-free-gb).
2. Gets the tarball: reuses a complete copy in the dump directory, else clones a local copy
   from a seed directory (for dumps fetched before this collector existed), else downloads it
   with resume. Verifies the size against HEAD and records the SHA-256.
3. Checks the licence: data.ppy.sh/LICENCE.txt once per run, and any licence file inside the
   tarball. A text that differs from the reference (the text recorded with the first dump)
   stops the run before any parsing.
4. Streams the tarball and writes the tables below to Parquet under
   artifacts/player-data-dumps/<dump-date>/, next to the tarball, then a manifest.json.
5. Rebuilds artifacts/player-data-dumps/collection.json and, when a dump was added, the panel
   report (dump_panel.py).

Tables kept: sample_users, osu_user_stats_mania, scores (mania rows, plus `in_corpus`),
osu_scores_mania_high, osu_user_beatmap_playcount (all rows), osu_beatmaps (mania rows) and
osu_beatmap_failtimes (mania beatmaps). The difficulty and attribute tables are not extracted;
they stay in the kept tarball.

Only data.ppy.sh is contacted. Per-player data stays in Parquet on the mac; JSON, Markdown and
log outputs (which mirror to the control plane) carry aggregates only.

Usage: .venv/bin/python scripts/player_data/dump_collector.py [--dry-run] [--include-top]
       [--only YYYY-MM-DD ...] [--no-panel] [--retry-blocked] [--job-id ID]
Exit codes: 0 done or nothing new; 1 a dump failed (retried next run); 2 blocked (format
change, disk, hash mismatch); 3 licence text changed (nothing parsed).
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import errno
import fcntl
import hashlib
import http.client
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import mysqldump_parse as mp  # noqa: E402

OUT_REL = os.path.join('artifacts', 'player-data-dumps')
CORPUS_REL = os.path.join('data', 'r2-corpus.parquet')
DEFAULT_SEED_DIRS = [os.path.join('artifacts', 'player-data-20261003', 'dump')]
BASE_URL = 'https://data.ppy.sh/'
LICENCE_URL = BASE_URL + 'LICENCE.txt'
UA = 'ensomi-player-data/2 (statistical analysis; monthly dump collector)'
NAME_RE = re.compile(r'^(\d{4})_(\d{2})_(\d{2})_performance_mania_(random_10000|top_10000)\.tar\.bz2$')
LICENCE_MEMBER_RE = re.compile(r'(?i)^(licen[cs]e|copying|terms)([._-].*)?$')
DOWNLOAD_ATTEMPTS = 5
EXTRACT_SIZE_FACTOR = 1.5      # Parquet output plus staging, as a multiple of the tarball size

# Hashes recorded for tarballs fetched before this collector existed. A seeded copy must match.
KNOWN_SHA256 = {
    # artifacts/player-data-20261003/failtimes/manifest.json, job 20261003-player-data-failtimes
    '2026_09_01_performance_mania_random_10000.tar.bz2':
        '2fa58ed17417deb9d395a36bdff97fa53d19681675622a32c7bd9f3c03cad81e',
}

# data.ppy.sh/LICENCE.txt as fetched with the first dump (job 20261003-player-data-failtimes).
REFERENCE_LICENCE_SHA256 = 'b5c69b6c9797ccfee25294e0829b74852a255be0b11ea5201ae506c9f64e6daa'
REFERENCE_LICENCE_TEXT = (
    'All data provided here is done so with the intention of it being used for statistical analysis\n'
    'and testing osu! subsystems.\n\n'
    'Permission is NOT implicitly granted to deploy this in production use of any kind.\n'
    'Should you wish to publicly use/expose the data provided here, please contact me first at contact@ppy.sh.\n\n'
    'Please see https://github.com/ppy/osu-performance for more information.\n\n'
    'Thanks,\nppy\n')

TABLES = ('sample_users', 'osu_user_stats_mania', 'scores', 'osu_scores_mania_high',
          'osu_user_beatmap_playcount', 'osu_beatmaps', 'osu_beatmap_failtimes')
ROWS_KEPT = {
    'sample_users': 'all',
    'osu_user_stats_mania': 'all',
    'scores': 'ruleset_id = 3 (mania); in_corpus added',
    'osu_scores_mania_high': 'all',
    'osu_user_beatmap_playcount': 'all',
    'osu_beatmaps': 'playmode = 3 (mania)',
    'osu_beatmap_failtimes': 'beatmap_id in this dump\'s mania osu_beatmaps',
}
SKIPPED_TABLES_NOTE = (
    'Not extracted: osu_beatmap_difficulty, osu_beatmap_difficulty_attribs and osu_difficulty_attribs '
    '(difficulty and attribute tables, skipped by design); osu_beatmapsets, osu_counts and '
    'osu_beatmap_performance_blacklist (not requested). All remain in the kept tarball.')
# Columns the extraction or the panel report depends on. A dump without them is blocked.
REQUIRED_COLUMNS = {
    'sample_users': ['user_id'],
    'osu_user_stats_mania': ['user_id', 'rank_score', 'playcount'],
    'scores': ['id', 'user_id', 'ruleset_id', 'beatmap_id', 'ended_at'],
    'osu_scores_mania_high': ['score_id', 'user_id', 'beatmap_id', 'date'],
    'osu_user_beatmap_playcount': ['user_id', 'beatmap_id', 'playcount'],
    'osu_beatmaps': ['beatmap_id', 'playmode'],
    'osu_beatmap_failtimes': ['beatmap_id', 'type'],
}
# Column lists of the 2026-09-01 random_10000 dump. Differences are recorded, not fatal.
REFERENCE_COLUMNS = {
    'sample_users': ['user_id', 'username', 'user_warnings', 'user_type'],
    'osu_user_stats_mania': [
        'user_id', 'count300', 'count100', 'count50', 'countMiss', 'accuracy_total', 'accuracy_count', 'accuracy',
        'playcount', 'ranked_score', 'total_score', 'x_rank_count', 'xh_rank_count', 's_rank_count',
        'sh_rank_count', 'a_rank_count', 'rank', 'level', 'replay_popularity', 'fail_count', 'exit_count',
        'max_combo', 'country_acronym', 'rank_score', 'rank_score_index', 'accuracy_new', 'last_update',
        'last_played', 'total_seconds_played'],
    'scores': [
        'id', 'user_id', 'ruleset_id', 'beatmap_id', 'has_replay', 'preserve', 'ranked', 'rank', 'passed',
        'accuracy', 'max_combo', 'total_score', 'data', 'pp', 'legacy_score_id', 'legacy_total_score',
        'started_at', 'ended_at', 'unix_updated_at', 'build_id'],
    'osu_scores_mania_high': [
        'score_id', 'beatmap_id', 'user_id', 'score', 'maxcombo', 'rank', 'count50', 'count100', 'count300',
        'countmiss', 'countgeki', 'countkatu', 'perfect', 'enabled_mods', 'date', 'pp', 'replay', 'hidden',
        'country_acronym'],
    'osu_user_beatmap_playcount': ['user_id', 'beatmap_id', 'playcount'],
    'osu_beatmaps': [
        'beatmap_id', 'beatmapset_id', 'user_id', 'filename', 'checksum', 'version', 'total_length',
        'hit_length', 'countTotal', 'countNormal', 'countSlider', 'countSpinner', 'diff_drain', 'diff_size',
        'diff_overall', 'diff_approach', 'playmode', 'approved', 'last_update', 'difficultyrating', 'max_combo',
        'playcount', 'passcount', 'youtube_preview', 'score_version', 'osu_file_version', 'deleted_at', 'bpm',
        'lazer_only'],
    'osu_beatmap_failtimes': ['beatmap_id', 'type'] + [f'p{i}' for i in range(1, 101)],
}


class Blocked(RuntimeError):
    """A condition that needs a human: format change, disk, hash mismatch."""


class LicenceChanged(RuntimeError):
    """The licence text differs from the reference. Nothing is parsed."""


# ---- small helpers ----------------------------------------------------------------------------
def utcnow():
    return dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def log(msg):
    print(f'[{utcnow()}] {msg}', flush=True)


def rel(path):
    return os.path.relpath(os.path.abspath(path), REPO)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, sort_keys=False, default=str)
        f.write('\n')
    os.replace(tmp, path)


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def sh(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=60, cwd=REPO).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f'error: {e!r}'


def norm_text(s):
    return ' '.join(s.split())


def detect_job_id(arg):
    if arg:
        return arg
    if os.environ.get('TMUX'):
        name = sh(['tmux', 'display-message', '-p', '#S'])
        if name and not name.startswith('error'):
            return name
    return None


def http(url, method='GET', headers=None, timeout=60):
    head = {'User-Agent': UA}
    head.update(headers or {})
    req = urllib.request.Request(url, headers=head, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers


# ---- listing ------------------------------------------------------------------------------------
class DumpEntry:
    def __init__(self, name):
        m = NAME_RE.match(name)
        if not m:
            raise ValueError(name)
        y, mo, d, kind = m.groups()
        self.name = name
        self.kind = kind
        self.dump_date = f'{y}-{mo}-{d}'
        self.url = BASE_URL + name
        self.dir_name = self.dump_date if kind == 'random_10000' else f'{self.dump_date}_{kind}'

    def __repr__(self):
        return self.name


def parse_listing(html):
    names = set()
    for href in re.findall(r'''href\s*=\s*["']([^"']+)["']''', html):
        base = href.rsplit('/', 1)[-1]
        if base.endswith('.tar.bz2'):
            names.add(base)
    return sorted(names)


def fetch_listing():
    status, body, _ = http(BASE_URL)
    if status != 200:
        raise RuntimeError(f'listing returned HTTP {status}')
    names = parse_listing(body.decode('utf-8', 'replace'))
    return {'url': BASE_URL, 'status': status, 'fetched_at': utcnow(),
            'sha256': hashlib.sha256(body).hexdigest(),
            'mania_files': [n for n in names if '_mania_' in n]}


# ---- licence ------------------------------------------------------------------------------------
def licence_record(data, source):
    text = data.decode('utf-8', 'replace')
    return {'source': source, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
            'same_bytes_as_reference': hashlib.sha256(data).hexdigest() == REFERENCE_LICENCE_SHA256,
            'same_text_as_reference': norm_text(text) == norm_text(REFERENCE_LICENCE_TEXT),
            'text': text if norm_text(text) != norm_text(REFERENCE_LICENCE_TEXT) else None}


def fetch_site_licence():
    status, body, _ = http(LICENCE_URL)
    if status != 200:
        raise RuntimeError(f'{LICENCE_URL} returned HTTP {status}')
    rec = licence_record(body, LICENCE_URL)
    rec['fetched_at'] = utcnow()
    return rec


# ---- tarball acquisition ------------------------------------------------------------------------
def head(entry):
    status, _, hd = http(entry.url, 'HEAD')
    if status != 200:
        raise RuntimeError(f'HEAD {entry.url} returned HTTP {status}')
    return {'status': status, 'content_length': int(hd.get('Content-Length') or 0),
            'last_modified': hd.get('Last-Modified'), 'etag': hd.get('ETag'), 'checked_at': utcnow()}


def check_disk(need_bytes, min_free_bytes, what):
    free = shutil.disk_usage(REPO).free
    if free - need_bytes < min_free_bytes:
        raise Blocked(f'{what}: free disk {free / 1e9:.1f} GB minus {need_bytes / 1e9:.1f} GB needed would fall '
                      f'below the {min_free_bytes / 1e9:.0f} GB floor')
    return free


def clone_or_copy(src, dest):
    tmp = dest + '.copy-tmp'
    if os.path.exists(tmp):
        os.remove(tmp)
    if platform.system() == 'Darwin':
        r = subprocess.run(['cp', '-c', src, tmp], capture_output=True, text=True)
        if r.returncode == 0:
            os.replace(tmp, dest)
            return 'apfs_clone'
    shutil.copyfile(src, tmp)
    os.replace(tmp, dest)
    return 'copy'


def download(url, dest, size, etag):
    """Resumable download into dest + '.part'; returns a record of the attempts."""
    part = dest + '.part'
    attempts = []
    t0 = time.time()
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have > size:
            os.remove(part)
            have = 0
        if have == size:
            break
        headers = {}
        if have:
            headers['Range'] = f'bytes={have}-'
            if etag:
                headers['If-Range'] = etag
        rec = {'attempt': attempt, 'started_at': utcnow(), 'bytes_at_start': have}
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA, **headers})
            with urllib.request.urlopen(req, timeout=120) as r:
                rec['status'] = r.status
                if have and r.status == 206:
                    cr = r.headers.get('Content-Range', '')
                    if not cr.startswith(f'bytes {have}-'):
                        raise RuntimeError(f'unexpected Content-Range {cr!r}')
                    mode = 'ab'
                elif r.status == 200:
                    mode, have = 'wb', 0
                else:
                    raise RuntimeError(f'HTTP {r.status}')
                last = time.time()
                with open(part, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        if time.time() - last > 30:
                            log(f'  download {os.path.getsize(part) / 1e6:.0f} of {size / 1e6:.0f} MB')
                            last = time.time()
        except (urllib.error.URLError, OSError, http.client.HTTPException, RuntimeError) as e:
            rec['error'] = repr(e)[:200]
            log(f'  download attempt {attempt} failed: {rec["error"]}')
        rec['bytes_at_end'] = os.path.getsize(part) if os.path.exists(part) else 0
        rec['finished_at'] = utcnow()
        attempts.append(rec)
        if rec['bytes_at_end'] == size:
            break
        if attempt < DOWNLOAD_ATTEMPTS:
            time.sleep(min(300, 30 * attempt))
    got = os.path.getsize(part) if os.path.exists(part) else 0
    if got != size:
        raise RuntimeError(f'download incomplete after {len(attempts)} attempts: {got} of {size} B '
                           f'(partial file kept for resume)')
    os.replace(part, dest)
    secs = round(time.time() - t0, 1)
    return {'attempts': attempts, 'seconds': secs, 'mb_per_s': round(size / 1e6 / secs, 2) if secs else None}


def acquire_tarball(entry, dump_dir, hd, seed_dirs):
    dest = os.path.join(dump_dir, entry.name)
    size = hd['content_length']
    rec = {'file': entry.name, 'path': rel(dest)}
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        rec['acquired_by'] = 'already_present'
    else:
        for seed in seed_dirs:
            src = os.path.join(REPO, seed, entry.name)
            if os.path.exists(src) and os.path.getsize(src) == size:
                log(f'  local copy found at {rel(src)}; copying (no download)')
                rec['acquired_by'] = 'local_' + clone_or_copy(src, dest)
                rec['source'] = rel(src)
                break
        else:
            log(f'  downloading {entry.url} ({size / 1e6:.1f} MB)')
            rec['acquired_by'] = 'download'
            rec['download'] = download(entry.url, dest, size, hd.get('etag'))
    rec['bytes'] = os.path.getsize(dest)
    if rec['bytes'] != size:
        raise Blocked(f'{entry.name}: local size {rec["bytes"]} != HEAD Content-Length {size}')
    rec['size_matches_head'] = True
    t0 = time.time()
    rec['sha256'] = sha256_file(dest)
    rec['hash_seconds'] = round(time.time() - t0, 1)
    known = KNOWN_SHA256.get(entry.name)
    if known:
        rec['earlier_recorded_sha256'] = known
        rec['matches_earlier_record'] = rec['sha256'] == known
        if not rec['matches_earlier_record']:
            raise Blocked(f'{entry.name}: SHA-256 {rec["sha256"]} differs from the earlier record {known}')
    return rec


# ---- tar streaming ------------------------------------------------------------------------------
class TarStream:
    """Sequential tar reader over `bzip2 -dc` (decompression in its own process) or Python bz2."""

    def __init__(self, path):
        self.path = path
        self.proc = None
        if shutil.which('bzip2'):
            self.proc = subprocess.Popen(['bzip2', '-dc', path], stdout=subprocess.PIPE, bufsize=1 << 20)
            self.tf = tarfile.open(fileobj=self.proc.stdout, mode='r|')
        else:
            self.tf = tarfile.open(path, mode='r|bz2')

    def __enter__(self):
        return self.tf

    def __exit__(self, exc_type, exc, tb):
        self.tf.close()
        if self.proc is not None:
            if exc_type is not None:
                self.proc.kill()
            self.proc.stdout.close()
            rc = self.proc.wait()
            if exc_type is None and rc != 0:
                raise RuntimeError(f'bzip2 -dc exited with {rc}')
        return False


def table_of(member_name):
    base = os.path.basename(member_name)
    return base[:-4] if base.endswith('.sql') else None


def scan_members(tar_path):
    """Pass 1: member list and any licence file, without parsing tables."""
    members, licences = [], []
    with TarStream(tar_path) as tf:
        for m in tf:
            rec = {'name': m.name, 'type': 'file' if m.isfile() else ('dir' if m.isdir() else 'other'),
                   'bytes': m.size, 'mtime_utc': dt.datetime.fromtimestamp(m.mtime, dt.timezone.utc)
                   .strftime('%Y-%m-%dT%H:%M:%SZ') if m.mtime else None}
            members.append(rec)
            if m.isfile() and LICENCE_MEMBER_RE.match(os.path.basename(m.name)):
                data = tf.extractfile(m).read(1 << 20)
                licences.append(licence_record(data, f'tarball member {m.name}'))
    return members, licences


class ParquetSink:
    """Buffers parsed rows, types them per column, filters, and appends to one Parquet file."""

    def __init__(self, path, columns, sql_types, keep=None, derive=(), batch_rows=200_000):
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.pa = pa
        self.path = path
        self.columns = list(columns)
        self.types = [mp.arrow_type(t) for t in sql_types]
        self.keep = keep
        self.derive = list(derive)
        fields = [pa.field(c, t) for c, t in zip(self.columns, self.types)]
        fields += [pa.field(name, typ) for name, typ, _ in self.derive]
        self.schema = pa.schema(fields)
        self.writer = pq.ParquetWriter(path, self.schema, compression='zstd')
        self.batch_rows = batch_rows
        self.buf = []
        self.rows_seen = 0
        self.rows_kept = 0
        self.unparsed = collections.Counter()

    def add(self, rows):
        self.buf.extend(rows)
        if len(self.buf) >= self.batch_rows:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        cols = list(zip(*self.buf))
        self.buf = []
        arrays = []
        for name, typ, col in zip(self.columns, self.types, cols):
            arr, bad = mp.to_arrow_column(col, typ)
            if bad:
                self.unparsed[name] += bad
            arrays.append(arr)
        t = self.pa.Table.from_arrays(arrays, names=self.columns)
        self.rows_seen += t.num_rows
        if self.keep is not None:
            t = t.filter(self.keep(t))
        for name, _, fn in self.derive:
            t = t.append_column(name, fn(t))
        self.rows_kept += t.num_rows
        if t.num_rows:
            self.writer.write_table(t.cast(self.schema))

    def close(self):
        self.flush()
        self.writer.close()
        return {'rows_seen': self.rows_seen, 'rows_kept': self.rows_kept,
                'timestamps_unparsed_to_null': dict(self.unparsed)}


def schema_check(table, columns):
    missing_required = [c for c in REQUIRED_COLUMNS[table] if c not in columns]
    ref = REFERENCE_COLUMNS[table]
    diff = {'same_as_reference': columns == ref,
            'missing_vs_reference': [c for c in ref if c not in columns],
            'extra_vs_reference': [c for c in columns if c not in ref]}
    if not diff['same_as_reference'] and not diff['missing_vs_reference'] and not diff['extra_vs_reference']:
        diff['order_differs'] = True
    return missing_required, diff


def extract_tables(tar_path, stage_dir, corpus_ids, batch_rows=200_000):
    """Pass 2: stream the tarball once and write the kept tables into stage_dir.

    Returns {table: stats}. Raises Blocked when a table is missing or lacks a required column.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    corpus_arr = pa.array(sorted(corpus_ids), type=pa.int64())
    stats = {}
    mania_ids = None
    failtimes_all = os.path.join(stage_dir, '_osu_beatmap_failtimes.all_modes.parquet')

    def keep_for(table):
        if table == 'scores':
            return lambda t: pc.fill_null(pc.equal(t['ruleset_id'], 3), False)
        if table == 'osu_beatmaps':
            return lambda t: pc.fill_null(pc.equal(t['playmode'], 3), False)
        return None

    def derive_for(table):
        if table == 'scores':
            return [('in_corpus', pa.bool_(),
                     lambda t: pc.fill_null(pc.is_in(t['beatmap_id'], value_set=corpus_arr), False))]
        return []

    t_start = time.time()
    with TarStream(tar_path) as tf:
        for m in tf:
            table = table_of(m.name) if m.isfile() else None
            if table not in TABLES:
                continue
            if table in stats:
                raise Blocked(f'table {table} appears twice in the tarball')
            t0 = time.time()
            out = failtimes_all if table == 'osu_beatmap_failtimes' else os.path.join(stage_dir, f'{table}.parquet')
            holder = {}
            reader = mp.MemberReader(m.name)

            def on_schema(columns, sql_types, table=table, out=out, holder=holder):
                missing, diff = schema_check(table, columns)
                holder['schema'] = diff
                if missing:
                    raise Blocked(f'{table}: required columns missing {missing}; columns are {columns}')
                holder['sink'] = ParquetSink(out, columns, sql_types, keep_for(table), derive_for(table),
                                             batch_rows=batch_rows)

            def on_rows(rows, holder=holder):
                holder['sink'].add(rows)

            log(f'  parsing {table} ({m.size / 1e6:.1f} MB)')
            reader.read(tf.extractfile(m), on_schema, on_rows)
            if 'sink' not in holder:
                raise Blocked(f'{table}: no CREATE TABLE found in {m.name}')
            st = reader.summary()
            st.update(holder['sink'].close())
            st['schema_vs_reference'] = holder['schema']
            st['rows_kept_rule'] = ROWS_KEPT[table]
            st['parse_seconds'] = round(time.time() - t0, 1)
            if st['rows_seen'] != st['rows_in_member']:
                raise RuntimeError(f'{table}: typed rows {st["rows_seen"]} != parsed rows {st["rows_in_member"]}')
            if table == 'osu_beatmaps':
                mania_ids = pq.read_table(out, columns=['beatmap_id']).column('beatmap_id').combine_chunks()
                st['mania_beatmaps'] = len(mania_ids)
            if table == 'scores':
                flag = pq.read_table(out, columns=['in_corpus']).column('in_corpus')
                st['rows_in_corpus'] = int(pc.sum(flag).as_py() or 0)
            stats[table] = st
            if table == 'osu_beatmap_failtimes':
                log(f'  {table}: {st["rows_seen"]:,} rows read in {st["parse_seconds"]} s '
                    '(filtered to mania beatmaps after osu_beatmaps)')
            else:
                log(f'  {table}: {st["rows_kept"]:,} of {st["rows_seen"]:,} rows kept in {st["parse_seconds"]} s')
    missing = [t for t in TABLES if t not in stats]
    if missing:
        raise Blocked(f'tables missing from the tarball: {missing}')

    # failtimes come before osu_beatmaps in the tarball; filter to mania beatmaps afterwards.
    ft = pq.read_table(failtimes_all)
    kept = ft.filter(pc.is_in(ft['beatmap_id'], value_set=mania_ids))
    pq.write_table(kept, os.path.join(stage_dir, 'osu_beatmap_failtimes.parquet'), compression='zstd')
    stats['osu_beatmap_failtimes']['rows_kept'] = kept.num_rows
    stats['osu_beatmap_failtimes']['mania_beatmaps_with_failtimes'] = len(pc.unique(kept['beatmap_id']))
    os.remove(failtimes_all)
    log(f'  osu_beatmap_failtimes: {kept.num_rows:,} of {ft.num_rows:,} rows kept (mania)')

    # cross-table checks (counts only)
    sample = pq.read_table(os.path.join(stage_dir, 'sample_users.parquet'), columns=['user_id'])['user_id']
    for table in ('osu_user_stats_mania', 'scores', 'osu_scores_mania_high', 'osu_user_beatmap_playcount'):
        users = pq.read_table(os.path.join(stage_dir, f'{table}.parquet'), columns=['user_id'])['user_id']
        outside = pc.sum(pc.invert(pc.is_in(users, value_set=sample))).as_py() or 0
        stats[table]['rows_with_user_not_in_sample_users'] = int(outside)
        stats[table]['distinct_users'] = len(pc.unique(users))
    for table in TABLES:
        p = os.path.join(stage_dir, f'{table}.parquet')
        stats[table]['parquet_file'] = f'{table}.parquet'
        stats[table]['parquet_bytes'] = os.path.getsize(p)
    stats['_seconds'] = round(time.time() - t_start, 1)
    return stats


# ---- corpus, provenance -------------------------------------------------------------------------
def load_corpus_ids():
    import pyarrow.parquet as pq
    path = os.path.join(REPO, CORPUS_REL)
    t = pq.read_table(path, columns=['api_beatmap_id', 'beatmap_id']).to_pydict()
    ids = set()
    for api_id, file_id in zip(t['api_beatmap_id'], t['beatmap_id']):
        if api_id:
            ids.add(int(api_id))
        elif file_id and file_id > 0:
            ids.add(int(file_id))
    return ids, {'path': CORPUS_REL, 'sha256': sha256_file(path), 'rows': len(t['beatmap_id']),
                 'distinct_beatmap_ids': len(ids),
                 'id_rule': 'api_beatmap_id when set, else the .osu file BeatmapID when > 0'}


def provenance(job_id):
    import pyarrow
    rec = {
        'job_id': job_id,
        'scripts': {rel(p): sha256_file(p) for p in (os.path.abspath(__file__), os.path.join(HERE, 'mysqldump_parse.py'))},
        'host': {'system': platform.system(), 'release': platform.release(), 'machine': platform.machine(),
                 'python': sys.version.split()[0], 'pyarrow': pyarrow.__version__},
        'mac_git': {'head': sh(['git', 'rev-parse', 'HEAD']), 'branch': sh(['git', 'rev-parse', '--abbrev-ref', 'HEAD']),
                    'note': "The mac's .git drifts behind the control plane (workspace rule 1); "
                            'control_plane_provenance is the identity recorded by ens run.'},
    }
    if job_id:
        p = os.path.join(REPO, '..', '.sync', 'cp', 'jobs', job_id, 'provenance.json')
        rec['control_plane_provenance'] = read_json(p)
    return rec


# ---- collection state ---------------------------------------------------------------------------
def dump_dirs(out_dir):
    if not os.path.isdir(out_dir):
        return []
    return sorted(d for d in os.listdir(out_dir)
                  if re.match(r'^\d{4}-\d{2}-\d{2}', d) and os.path.isdir(os.path.join(out_dir, d)))


def manifest_of(out_dir, dir_name):
    return read_json(os.path.join(out_dir, dir_name, 'manifest.json'))


def build_collection(out_dir, listing, job_id):
    import pyarrow  # noqa: F401  (environment check only)
    listed = set(listing['mania_files']) if listing else None
    dumps = []
    for d in dump_dirs(out_dir):
        man = manifest_of(out_dir, d)
        if not man:
            continue
        tb = man.get('tarball') or {}
        tables = man.get('tables') or {}
        parquet_bytes = sum(v.get('parquet_bytes', 0) for k, v in tables.items() if isinstance(v, dict))
        dumps.append({
            'dir': d, 'name': man['dump']['name'], 'kind': man['dump']['kind'], 'dump_date': man['dump']['dump_date'],
            'status': man.get('status'), 'error': man.get('error'), 'url': man['dump']['url'],
            'head_last_modified': (man.get('head') or {}).get('last_modified'),
            'tarball_bytes': tb.get('bytes'), 'tarball_sha256': tb.get('sha256'), 'acquired_by': tb.get('acquired_by'),
            'licence_site_same_text': ((man.get('licence') or {}).get('site') or {}).get('same_text_as_reference'),
            'licence_in_tarball': [x.get('same_text_as_reference') for x in (man.get('licence') or {}).get('in_tarball', [])],
            'rows': {k: v.get('rows_kept') for k, v in tables.items() if isinstance(v, dict)},
            'table_snapshot_utc': {k: v.get('dump_completed_utc') for k, v in tables.items() if isinstance(v, dict)},
            'parquet_bytes': parquet_bytes,
            'still_listed': (man['dump']['name'] in listed) if listed is not None else None,
            'collected_at': man.get('finished_at'), 'job_id': (man.get('provenance') or {}).get('job_id'),
        })
    complete = [x for x in dumps if x['status'] == 'complete']
    complete_names = {x['name'] for x in complete}

    def listed_not_collected(kind):
        if not listing:
            return None
        return sorted(n for n in listing['mania_files']
                      if NAME_RE.match(n) and DumpEntry(n).kind == kind and n not in complete_names)

    out = {
        'schema': 'ensomi.player-dump-collection/1',
        'updated_at': utcnow(), 'updated_by_job': job_id,
        'listing': listing,
        'dumps': dumps,
        'complete_random_10000': [x['dump_date'] for x in complete if x['kind'] == 'random_10000'],
        'listed_not_collected': {'random_10000': listed_not_collected('random_10000'),
                                 'top_10000 (off unless --include-top)': listed_not_collected('top_10000')},
        'collected_not_listed_any_more': sorted(x['name'] for x in complete if x['still_listed'] is False),
        'disk': {'collection_bytes': sum((x['tarball_bytes'] or 0) + x['parquet_bytes'] for x in dumps),
                 'free_bytes': shutil.disk_usage(REPO).free},
        'note': 'Aggregates only. Per-player tables are Parquet files in each dump directory on the mac.',
    }
    write_json(os.path.join(out_dir, 'collection.json'), out)
    return out


# ---- one dump -----------------------------------------------------------------------------------
def collect_one(entry, out_dir, args, site_licence, corpus_ids, corpus_info, job_id):
    dump_dir = os.path.join(out_dir, entry.dir_name)
    os.makedirs(dump_dir, exist_ok=True)
    man_path = os.path.join(dump_dir, 'manifest.json')
    man = {'schema': 'ensomi.player-dump-manifest/1', 'status': 'running', 'started_at': utcnow(),
           'dump': {'name': entry.name, 'kind': entry.kind, 'dump_date': entry.dump_date, 'url': entry.url},
           'provenance': provenance(job_id), 'corpus': corpus_info, 'rows_kept': ROWS_KEPT,
           'skipped_tables': SKIPPED_TABLES_NOTE}
    t0 = time.time()
    try:
        man['head'] = hd = head(entry)
        size = hd['content_length']
        if not size:
            raise Blocked(f'HEAD gave no Content-Length for {entry.url}')
        dest = os.path.join(dump_dir, entry.name)
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        part = dest + '.part'
        have = max(have, os.path.getsize(part) if os.path.exists(part) else 0)
        need = max(0, size - have) + int(EXTRACT_SIZE_FACTOR * size)
        man['free_disk_bytes_before'] = check_disk(need, args.min_free_gb * 1e9, entry.name)
        man['tarball'] = acquire_tarball(entry, dump_dir, hd, args.seed_dir)
        log(f'  tarball sha256 {man["tarball"]["sha256"]} ({man["tarball"]["acquired_by"]})')

        t1 = time.time()
        members, licences = scan_members(dest)
        man['members'] = members
        man['scan_seconds'] = round(time.time() - t1, 1)
        man['licence'] = {'reference_sha256': REFERENCE_LICENCE_SHA256,
                          'reference': 'data.ppy.sh/LICENCE.txt as fetched with the first dump '
                                       '(job 20261003-player-data-failtimes, 2026-10-03)',
                          'site': site_licence, 'in_tarball': licences,
                          'note': None if licences else 'The tarball contains no licence file; the site '
                                                        'LICENCE.txt (fetched this run) is the licence of record.'}
        changed = [x for x in licences if not x['same_text_as_reference']]
        if changed:
            raise LicenceChanged(f'{entry.name}: licence file inside the tarball differs from the reference: '
                                 f'{[x["source"] for x in changed]}')
        tables_present = {table_of(x['name']) for x in members if x['type'] == 'file'}
        missing = [t for t in TABLES if t not in tables_present]
        if missing:
            raise Blocked(f'{entry.name}: tables missing from the tarball: {missing}')

        stage = os.path.join(dump_dir, '.staging')
        shutil.rmtree(stage, ignore_errors=True)
        os.makedirs(stage)
        check_disk(int(EXTRACT_SIZE_FACTOR * size), args.min_free_gb * 1e9, entry.name)
        stats = extract_tables(dest, stage, corpus_ids)
        man['extract_seconds'] = stats.pop('_seconds')
        for table in TABLES:
            os.replace(os.path.join(stage, f'{table}.parquet'), os.path.join(dump_dir, f'{table}.parquet'))
        shutil.rmtree(stage, ignore_errors=True)
        man['tables'] = stats
        man['status'] = 'complete'
        man['error'] = None
        return man, 0
    except LicenceChanged as e:
        man['status'], man['error'] = 'blocked_licence', str(e)
        return man, 3
    except Blocked as e:
        man['status'], man['error'] = 'blocked', str(e)
        return man, 2
    except mp.DumpFormatError as e:
        man['status'], man['error'] = 'blocked', f'format: {e}'
        return man, 2
    except Exception as e:  # noqa: BLE001
        man['status'], man['error'] = 'failed', repr(e)[:500]
        man['traceback'] = traceback.format_exc()[-3000:]
        return man, 1
    finally:
        man['finished_at'] = utcnow()
        man['seconds'] = round(time.time() - t0, 1)
        shutil.rmtree(os.path.join(dump_dir, '.staging'), ignore_errors=True)
        write_json(man_path, man)


# ---- main ---------------------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--dry-run', action='store_true', help='listing and plan only; no HEAD, download or parse')
    ap.add_argument('--include-top', action='store_true',
                    help='also collect mania top_10000 dumps (about 2 GB each); off by default')
    ap.add_argument('--only', nargs='*', default=None, metavar='YYYY-MM-DD', help='restrict to these dump dates')
    ap.add_argument('--no-panel', action='store_true', help='do not rebuild the panel report after adding a dump')
    ap.add_argument('--retry-blocked', action='store_true', help='retry dumps whose manifest says blocked')
    ap.add_argument('--seed-dir', action='append', default=None,
                    help='directory (relative to the repo) holding earlier downloaded tarballs; '
                         f'default {DEFAULT_SEED_DIRS}')
    ap.add_argument('--min-free-gb', type=float, default=50.0)
    ap.add_argument('--job-id', default=None, help='recorded in manifests (detected from tmux when omitted)')
    args = ap.parse_args(argv)
    if args.seed_dir is None:
        args.seed_dir = list(DEFAULT_SEED_DIRS)
    return args


def main(argv=None):
    args = parse_args(argv)
    os.chdir(REPO)
    out_dir = os.path.join(REPO, OUT_REL)
    os.makedirs(os.path.join(out_dir, 'log'), exist_ok=True)
    lock_f = open(os.path.join(out_dir, '.collector.lock'), 'w')
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        if e.errno in (errno.EAGAIN, errno.EACCES):
            log('another collector run holds the lock; exiting')
            return 0
        raise
    job_id = detect_job_id(args.job_id)
    run = {'started_at': utcnow(), 'job_id': job_id, 'dry_run': args.dry_run, 'include_top': args.include_top,
           'new_found': [], 'collected': [], 'blocked': [], 'failed': [], 'skipped_blocked': []}
    rc = 0
    listing = None
    try:
        log(f'collector start (job {job_id}); output {OUT_REL}')
        listing = fetch_listing()
        run['listing_sha256'] = listing['sha256']
        kinds = {'random_10000'} | ({'top_10000'} if args.include_top else set())
        entries = []
        for n in listing['mania_files']:
            if NAME_RE.match(n):
                e = DumpEntry(n)
                if e.kind in kinds and (not args.only or e.dump_date in args.only):
                    entries.append(e)
        entries.sort(key=lambda e: (e.dump_date, e.kind))
        log(f'listing: {len(listing["mania_files"])} mania files; candidates {[e.name for e in entries]}')
        todo = []
        for e in entries:
            man = manifest_of(out_dir, e.dir_name)
            status = (man or {}).get('status')
            if status == 'complete':
                continue
            if status in ('blocked', 'blocked_licence') and not args.retry_blocked:
                run['skipped_blocked'].append(e.name)
                log(f'WARNING {e.name} is {status} ({(man or {}).get("error")}); pass --retry-blocked after a fix')
                continue
            todo.append(e)
        run['new_found'] = [e.name for e in todo]
        if not todo:
            log('nothing new to collect')
        elif args.dry_run:
            log(f'dry run: would collect {run["new_found"]}')
        else:
            site_licence = fetch_site_licence()
            run['licence_site_sha256'] = site_licence['sha256']
            if not site_licence['same_text_as_reference']:
                raise LicenceChanged(f'{LICENCE_URL} text differs from the reference (sha256 {site_licence["sha256"]})')
            corpus_ids, corpus_info = load_corpus_ids()
            for e in todo:
                log(f'collecting {e.name}')
                man, code = collect_one(e, out_dir, args, site_licence, corpus_ids, corpus_info, job_id)
                log(f'{e.name}: {man["status"]} in {man["seconds"]} s' + (f' ({man["error"]})' if man.get('error') else ''))
                if code == 0:
                    run['collected'].append(e.name)
                elif code == 1:
                    run['failed'].append(e.name)
                    rc = max(rc, 1)
                else:
                    run['blocked'].append(e.name)
                    rc = max(rc, code)
                if code == 3 or (code == 1 and 'download incomplete' in (man.get('error') or '')):
                    log('stopping the run: ' + ('licence changed' if code == 3 else 'download keeps failing'))
                    break
    except LicenceChanged as e:
        log(f'STOP: {e}')
        run['error'] = str(e)
        rc = 3
    except Exception as e:  # noqa: BLE001
        log(f'ERROR: {e!r}')
        run['error'] = repr(e)[:500]
        rc = max(rc, 1)
    try:
        coll = build_collection(out_dir, listing, job_id)
        run['complete_random_10000'] = coll['complete_random_10000']
        if run['collected'] and not args.no_panel and len(coll['complete_random_10000']) >= 2:
            import dump_panel
            log('rebuilding the panel report')
            dump_panel.main(['--job-id', job_id or ''])
            run['panel_rebuilt'] = True
    except Exception as e:  # noqa: BLE001
        log(f'ERROR after collection: {e!r}')
        run['post_error'] = repr(e)[:500]
        rc = max(rc, 1)
    run['finished_at'] = utcnow()
    run['exit_code'] = rc
    with open(os.path.join(out_dir, 'log', 'runs.jsonl'), 'a') as f:
        f.write(json.dumps(run) + '\n')
    log(f'collector done: collected {run["collected"]}, blocked {run["blocked"]}, failed {run["failed"]}; exit {rc}')
    return rc


if __name__ == '__main__':
    sys.exit(main())
