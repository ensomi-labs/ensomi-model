"""Streaming reader for the mysqldump members of the data.ppy.sh performance dumps.

Each `.sql` member holds one table: header comments, a `CREATE TABLE` block, extended
`INSERT INTO ... VALUES (...),(...);` lines (one statement per line), and a trailer line
`-- Dump completed on YYYY-MM-DD HH:MM:SS`. The dumps set `TIME_ZONE='+00:00'`, so timestamps
are UTC.

Values come back as `str` or `None` (SQL NULL). Typing happens per column in Arrow
(`arrow_type`, `to_arrow_column`), so the per-value work in Python stays small. The core parser
needs only the standard library; the Arrow helpers import pyarrow when called.
"""
from __future__ import annotations

import hashlib
import re

CREATE_COL_RE = re.compile(r'^\s*`([^`]+)`\s+(.*?),?\s*$')
INSERT_RE = re.compile(r'^INSERT INTO `([^`]+)`(?: \(([^)]*)\))? VALUES ')
TRAILER_RE = re.compile(r'^-- Dump completed on (\d{4}-\d{2}-\d{2})\s+(\d{1,2}):(\d{2}):(\d{2})')
SERVER_RE = re.compile(r'^-- Server version\s+(.+)$')

# One value: a quoted string (unrolled loop, so a malformed line cannot backtrack
# catastrophically), NULL, or a bare literal such as a number.
_TOKEN = re.compile(r"(?:_binary\s*)?'([^'\\]*(?:\\.[^'\\]*)*)'|(NULL)(?=[,)])|([^,()']+)", re.S)
_QUOTED = re.compile(r"'[^']*'")
_ESC_RE = re.compile(r'\\(.)', re.S)
_ESC = {'0': '\0', 'b': '\b', 'n': '\n', 'r': '\r', 't': '\t', 'Z': '\x1a'}


class DumpFormatError(ValueError):
    """The member does not look like the mysqldump output this reader expects."""


def unescape(s: str) -> str:
    """Undo mysqldump string escaping (\\0 \\b \\n \\r \\t \\Z, and \\x for any other x)."""
    if '\\' not in s:
        return s
    return _ESC_RE.sub(lambda m: _ESC.get(m.group(1), m.group(1)), s)


def _fast_rows(body: str):
    """Rows of a VALUES body by plain splitting, or None when that would be unsafe.

    Safe when the body has no backslash, no quoted 'NULL', no `_binary` introducer, and no
    quoted token that contains a comma or parenthesis: then quotes can be dropped and the text
    split on '),(' and ','.
    """
    if '\\' in body or "'NULL'" in body or '_binary' in body:
        return None
    if "'" in body:
        if body.count("'") % 2:
            return None
        quoted = ''.join(_QUOTED.findall(body))
        if ',' in quoted or '(' in quoted or ')' in quoted:
            return None
        body = body.replace("'", '')
    inner = body.rstrip()
    if inner.endswith(';'):
        inner = inner[:-1]
    if not (inner.startswith('(') and inner.endswith(')')):
        return None
    parts = inner[1:-1].split('),(')
    if 'NULL' in inner:
        return [[None if v == 'NULL' else v for v in p.split(',')] for p in parts]
    return [p.split(',') for p in parts]


def _slow_rows(body: str):
    """Rows of a VALUES body by a tokenizer that honours quoting and escapes."""
    rows = []
    pos, n = 0, len(body)
    match = _TOKEN.match
    while True:
        if pos >= n or body[pos] != '(':
            raise DumpFormatError(f'expected "(" at offset {pos}')
        pos += 1
        row = []
        while True:
            m = match(body, pos)
            if m is None:
                raise DumpFormatError(f'cannot read a value at offset {pos}')
            s, null, bare = m.group(1), m.group(2), m.group(3)
            if s is not None:
                row.append(unescape(s))
            elif null is not None:
                row.append(None)
            else:
                row.append(bare.strip())
            pos = m.end()
            c = body[pos] if pos < n else ''
            if c == ',':
                pos += 1
                continue
            if c == ')':
                pos += 1
                break
            raise DumpFormatError(f'expected "," or ")" at offset {pos}')
        rows.append(row)
        c = body[pos] if pos < n else ''
        if c == ',':
            pos += 1
            continue
        if c in (';', '', '\n', '\r'):
            return rows
        raise DumpFormatError(f'unexpected text after a row at offset {pos}')


def parse_values(body: str):
    """All rows of one extended INSERT body, '(...),(...);', as lists of str or None.

    Returns (rows, used_fast_path).
    """
    rows = _fast_rows(body)
    if rows is not None:
        return rows, True
    return _slow_rows(body), False


def parse_insert(line: str):
    """(table, explicit column list or None, rows, used_fast_path) for one INSERT line."""
    m = INSERT_RE.match(line)
    if not m:
        raise DumpFormatError('not an INSERT line')
    cols = [c.strip(' `') for c in m.group(2).split(',')] if m.group(2) else None
    rows, fast = parse_values(line[m.end():])
    return m.group(1), cols, rows, fast


def parse_create(create_sql: str):
    """[(column, sql_type)] from a CREATE TABLE block, in table order."""
    cols = []
    for line in create_sql.splitlines()[1:]:
        if line.lstrip().startswith(')'):
            break
        m = CREATE_COL_RE.match(line)
        if m:
            cols.append((m.group(1), m.group(2)))
    return cols


def trailer_utc(line: str):
    """'-- Dump completed on 2026-09-01  5:48:21' -> '2026-09-01T05:48:21Z' (dumps run in UTC)."""
    m = TRAILER_RE.match(line)
    if not m:
        return None
    d, hh, mm, ss = m.groups()
    return f'{d}T{int(hh):02d}:{mm}:{ss}Z'


class MemberReader:
    """Reads one mysqldump member line by line.

    `on_schema(columns, sql_types)` is called once, after the CREATE TABLE block and before any
    rows. `on_rows(rows)` is called for every INSERT line with the parsed rows. Both may raise to
    stop reading.
    """

    def __init__(self, name: str):
        self.name = name
        self.table = None
        self.header: list[str] = []
        self.create_sql = None
        self.columns: list[str] | None = None
        self.sql_types: list[str] | None = None
        self.server_version = None
        self.trailer = None
        self.dump_completed_utc = None
        self.bytes = 0
        self.insert_lines = 0
        self.fast_lines = 0
        self.rows = 0

    def read(self, lines, on_schema=None, on_rows=None):
        in_create, buf = False, []
        for raw in lines:
            self.bytes += len(raw)
            line = raw.decode('utf-8', 'replace') if isinstance(raw, (bytes, bytearray)) else raw
            if in_create:
                buf.append(line.rstrip('\n'))
                if line.startswith(')'):
                    in_create = False
                    self._set_schema('\n'.join(buf), on_schema)
                continue
            if line.startswith('INSERT INTO'):
                if self.columns is None:
                    raise DumpFormatError(f'{self.name}: INSERT before CREATE TABLE')
                try:
                    table, cols, rows, fast = parse_insert(line)
                except DumpFormatError as e:
                    raise DumpFormatError(f'{self.name}: INSERT line {self.insert_lines + 1}: {e}') from None
                if table != self.table:
                    raise DumpFormatError(f'{self.name}: INSERT into {table}, expected {self.table}')
                if cols is not None and cols != self.columns:
                    raise DumpFormatError(f'{self.name}: INSERT column list differs from CREATE TABLE')
                ncol = len(self.columns)
                widths = set(map(len, rows))
                if widths != {ncol}:
                    raise DumpFormatError(f'{self.name}: row widths {sorted(widths)} != {ncol} columns '
                                          f'(INSERT line {self.insert_lines + 1})')
                self.insert_lines += 1
                self.fast_lines += fast
                self.rows += len(rows)
                if on_rows is not None:
                    on_rows(rows)
                continue
            if line.startswith('CREATE TABLE'):
                m = re.match(r'CREATE TABLE `([^`]+)`', line)
                self.table = m.group(1) if m else None
                in_create, buf = True, [line.rstrip('\n')]
                continue
            if line.startswith('-- Dump completed'):
                self.trailer = line.strip()
                self.dump_completed_utc = trailer_utc(self.trailer)
            elif self.create_sql is None and len(self.header) < 40 and line.startswith(('--', '/*')):
                self.header.append(line.strip()[:300])
                m = SERVER_RE.match(line.strip())
                if m:
                    self.server_version = m.group(1).strip()
        if in_create:
            raise DumpFormatError(f'{self.name}: CREATE TABLE block not closed')
        return self

    def _set_schema(self, create_sql, on_schema):
        self.create_sql = create_sql
        pairs = parse_create(create_sql)
        if not pairs:
            raise DumpFormatError(f'{self.name}: no columns in CREATE TABLE')
        self.columns = [c for c, _ in pairs]
        self.sql_types = [t for _, t in pairs]
        if on_schema is not None:
            on_schema(self.columns, self.sql_types)

    def summary(self):
        return {
            'member': self.name, 'table': self.table, 'member_bytes': self.bytes,
            'insert_lines': self.insert_lines, 'fast_path_lines': self.fast_lines, 'rows_in_member': self.rows,
            'columns': self.columns, 'sql_types': self.sql_types,
            'create_table_sha256': hashlib.sha256(self.create_sql.encode()).hexdigest() if self.create_sql else None,
            'server_version': self.server_version, 'trailer': self.trailer,
            'dump_completed_utc': self.dump_completed_utc,
        }


# ---- Arrow typing ---------------------------------------------------------------------------
_INT = {'tinyint', 'smallint', 'mediumint', 'int', 'integer', 'bigint', 'year'}
_FLOAT = {'float', 'double', 'decimal', 'real', 'numeric'}
_TIME = {'timestamp', 'datetime'}


def arrow_type(sql_type: str):
    """Arrow type for a MySQL column type. Integers -> int64, reals -> float64,
    timestamp/datetime -> timestamp[s, UTC], everything else (char, enum, json, ...) -> string."""
    import pyarrow as pa
    m = re.match(r'\s*([a-zA-Z]+)', sql_type)
    base = m.group(1).lower() if m else ''
    if base in _INT:
        return pa.int64()
    if base in _FLOAT:
        return pa.float64()
    if base in _TIME:
        return pa.timestamp('s', tz='UTC')
    return pa.string()


def to_arrow_column(values, typ):
    """(array, n_unparsed): str/None values cast to `typ`. For timestamps, values that do not
    parse (e.g. MySQL zero dates) become null and are counted in n_unparsed."""
    import pyarrow as pa
    import pyarrow.compute as pc
    arr = pa.array(values, type=pa.string())
    if typ == pa.string():
        return arr, 0
    if pa.types.is_timestamp(typ):
        ts = pc.strptime(arr, format='%Y-%m-%d %H:%M:%S', unit='s', error_is_null=True)
        return ts.cast(typ), ts.null_count - arr.null_count
    return arr.cast(typ), 0
