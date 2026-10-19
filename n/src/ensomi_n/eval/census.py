"""The strata census (E2) and the sealed hard-strata holdout, from labels only.

Input is a units table (CSV, one row per `.osu` file) naming each chart's set
and what the census needs to know about it:

``file`` (path under ``--root``), ``key`` (the set), ``source`` (for example
``4k`` or ``std-taiko``), ``recording`` (the decision unit: every set of one
recording, dedup groups merged, shares it), ``split`` (``fit_dev``,
``holdout2``, ``std-taiko``, ...), ``status`` (``ranked``, ``loved``),
``mode`` (``mania4k``, ``std``, ``taiko``), ``set_id``, ``tags`` (voted tags,
``;``-separated) and ``tags_known`` (1 when the voted tags were looked up).

`compute` reads every set's charts once and computes its strata
(`ensomi_n.eval.strata`). `tabulate` counts, per split and stratum, the
recordings, sets and heads, the criteria behind each stratum, and the
differing second-mapper pairs; a recording's strata within a split come from
its sets in that split. `e2_status` applies E2's rule. `draw` and `seal` make
the hard-strata holdout; `load_exclusion` is how a training or design
manifest leaves it out.

Sealed splits (the holdouts) appear only as counts: no per-set or
per-recording row of a sealed split is written anywhere.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from ..labels import read_osu, reference_from_charts
from .strata import HARD, NAMES, NEEDS_LOGITS, STRATA, SetStrata, is_rate_edit, recording_strata, same_timing, \
    set_strata

E2_MIN_RECORDINGS = 40
E2_MIN_PAIRS = 10
DRAW_PER_STRATUM = 40
DRAW_KEEP_DESIGN = 10
DRAW_SEED = 20261018
MEMBERSHIP = 'membership.json'


@dataclass
class Unit:
    """One set as the census sees it (module docstring)."""
    key: str
    files: list[str]
    source: str = ''
    recording: str = ''
    split: str = ''
    status: str = ''
    mode: str = ''
    set_id: str = ''
    tags: list[str] = field(default_factory=list)
    tags_known: bool = False


@dataclass
class SetResult:
    unit: Unit
    strata: SetStrata | None
    error: str = ''
    creators: frozenset = frozenset()
    segments: tuple = ()
    audio: str = ''
    extra: dict = field(default_factory=dict)


def read_units(path: str | os.PathLike) -> list[Unit]:
    """The units table, one `Unit` per set key, files in table order. Raises ``ValueError`` on a key
    whose rows disagree on recording, split or source."""
    units: dict[str, Unit] = {}
    with open(path, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        need = {'file', 'key', 'recording', 'split'}
        if reader.fieldnames is None or not need <= set(reader.fieldnames):
            raise ValueError(f'{path}: a units table needs the columns {sorted(need)}')
        for row in reader:
            key = row['key'].strip()
            u = units.get(key)
            tags = [t.strip() for t in (row.get('tags') or '').split(';') if t.strip()]
            if u is None:
                units[key] = Unit(key, [row['file']], (row.get('source') or '').strip(), row['recording'].strip(),
                                  row['split'].strip(), (row.get('status') or '').strip(),
                                  (row.get('mode') or '').strip(), (row.get('set_id') or '').strip(), tags,
                                  (row.get('tags_known') or '0').strip() in ('1', 'true', 'True'))
                continue
            for name in ('recording', 'split', 'source'):
                if (row.get(name) or '').strip() != getattr(u, name):
                    raise ValueError(f'{path}: set {key!r} names two values of {name}')
            u.files.append(row['file'])
            u.tags = sorted(set(u.tags) | set(tags))
    return list(units.values())


def read_header(path: str | os.PathLike) -> dict:
    """``[General]``, ``[Metadata]`` and ``[Difficulty]`` fields of an `.osu` file, read up to its timing points."""
    out, section = {}, None
    with Path(path).open('r', encoding='utf-8-sig', errors='replace') as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith('[') and line.endswith(']'):
                section = line[1:-1]
                if section in ('TimingPoints', 'HitObjects', 'Events'):
                    break
                continue
            if section in ('General', 'Metadata', 'Difficulty') and ':' in line:
                name, _, value = line.partition(':')
                out[name.strip()] = value.strip()
    return out


def compute(units: Sequence[Unit], root: str | os.PathLike = '.', *,
            progress: Callable[[int, int], None] | None = None,
            on_set: Callable | None = None) -> list[SetResult]:
    """Read each unit's charts once and compute its strata. A set that cannot be read is kept with its error.

    ``on_set(unit, ref, charts, paths)``, when given, runs on every readable set
    in the same pass; its dict is kept as ``SetResult.extra`` (an exception in
    it is recorded there as ``{'error': ...}``).
    """
    root = Path(root)
    out = []
    for i, u in enumerate(units):
        try:
            paths = sorted(root / f for f in u.files)
            gone = [p for p in paths if not p.is_file()]
            paths = [p for p in paths if p.is_file()]
            charts = [read_osu(p) for p in paths]
            headers = [read_header(p) for p in paths]
            ref = reference_from_charts(charts, u.key)
            rate = is_rate_edit(*[h.get('Version', '') for h in headers], ref.meta.get('audio', ''))
            st = set_strata(ref, charts, tags=u.tags, tags_known=u.tags_known, rate_edit=rate, key=u.key)
            creators = frozenset(h.get('Creator', '').casefold() for h in headers if h.get('Creator'))
            extra = {}
            if on_set is not None:
                try:
                    extra = on_set(u, ref, charts, paths) or {}
                except Exception as exc:  # noqa: BLE001 - recorded, never fatal for the census
                    extra = dict(error=f'{type(exc).__name__}: {exc}')
            if gone:
                extra['missing_files'] = len(gone)
            out.append(SetResult(u, st, '', creators, tuple(ref.segments or ()), ref.meta.get('audio', ''), extra))
        except Exception as exc:  # noqa: BLE001 - one unreadable set is counted, never fatal for the census
            out.append(SetResult(u, None, f'{type(exc).__name__}: {exc}'))
        if progress is not None:
            progress(i + 1, len(units))
    return out


def criteria(s: SetStrata) -> dict[str, bool]:
    """The criteria behind each stratum, for the census's characterising counts."""
    d = s.detail
    return {
        'S1.canonical_mixed>=1%': d['S1']['canonical_mixed_share'] >= 0.01,
        'S2.irregular': d['S2']['irregular'], 'S2.changing': d['S2']['changing'],
        'S2.irregular>=4bars': d['S2']['irregular_bars'] >= 4,
        'S3.tempo_mapped': d['S3']['tempo_mapped'], 'S3.ramps': d['S3']['ramps'] > 0,
        'S3.tag_variable_timing': d['S3']['tag_variable_timing'], 'S3.tag_accelerating': d['S3']['tag_accelerating'],
        'S4.short_section': d['S4']['short_sections'] > 0, 'S4.reset': d['S4']['resets'] > 0,
        'S5.near_80_160': 'S5' in s.members and d['S5']['near_80_160_ms'] >= 8000.0,
        'S5.level_change': d['S5']['level_changes'] > 0, 'S5.modulation': d['S5']['modulations'] > 0,
        'S6.gap': d['S6']['gaps'] > 0, 'S6.sparse_window': d['S6']['sparse_windows'] > 0,
        'S7.tag_swing': d['S7']['tag_swing'], 'S7.swing_bars': d['S7']['swing_bars'] >= 4,
        'S9.detour_with_heads': d['S9']['detour_runs_with_heads'] > 0,
        'S9.implausible_with_heads': d['S9']['implausible_with_heads'] > 0,
        'S9.one_tempo_reset': d['S9']['one_tempo_resets'] > 0,
        'S9.off_lattice_head': d['S9']['off_lattice_heads'] > 0,
        'S9.off_lattice>=4heads': d['S9']['off_lattice_heads'] >= 4,
        'S9.sub_beat_segment': d['S9']['sub_beat_segments'] > 0,
        'S0.tags_unchecked': not d['S0']['tags_checked'],
    }


def _group(results: Iterable[SetResult], split_of: Callable[[SetResult], str]):
    groups: dict[tuple[str, str], list[SetResult]] = {}
    for r in results:
        if r.strata is None:
            continue
        groups.setdefault((split_of(r), r.unit.recording), []).append(r)
    return groups


def _pairs(sets: Sequence[SetResult]) -> tuple[int, int]:
    """Differing and copied second-mapper pairs among one recording's sets: different set ids, no shared creator."""
    diff = copies = 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            if a.unit.set_id and a.unit.set_id == b.unit.set_id:
                continue
            if not a.creators or not b.creators or a.creators & b.creators:
                continue
            if same_timing(a.segments, b.segments):
                copies += 1
            else:
                diff += 1
    return diff, copies


def tabulate(results: Sequence[SetResult], *, split_of: Callable[[SetResult], str] | None = None,
             set_flags: dict[str, dict[str, bool]] | None = None) -> dict:
    """Counts per split (and ``split|status``, ``split|mode``) and stratum (module docstring).

    ``set_flags`` adds, per stratum, the number of recordings with a set
    carrying each flag (E3's reader differences, for one).
    """
    split_of = split_of or (lambda r: r.unit.split)
    set_flags = set_flags or {}
    groups = _group(results, split_of)
    table: dict[str, dict] = {}

    def bucket(name: str) -> dict:
        return table.setdefault(name, {'all': _empty(), **{s: _empty() for s in STRATA}})

    for (split, rec), sets in sorted(groups.items()):
        member = recording_strata([r.strata for r in sets])
        diff, copies = _pairs(sets)
        names = [split]
        statuses = {r.unit.status for r in sets if r.unit.status}
        modes = {r.unit.mode for r in sets if r.unit.mode}
        names += [f'{split}|{v}' for v in sorted(statuses)] + [f'{split}|{v}' for v in sorted(modes)]
        for name in names:
            b = bucket(name)
            sub = [r for r in sets if (name == split or name.split('|', 1)[1] in (r.unit.status, r.unit.mode))]
            sub_keys = {r.unit.key for r in sub}
            _add(b['all'], 'all', sub, None, diff, copies, [criteria(r.strata) for r in sub], set_flags)
            for stratum, spans_by_set in member.items():
                keys = set(spans_by_set) & sub_keys
                if not keys:
                    continue
                rows = [r for r in sub if r.unit.key in keys]
                _add(b[stratum], stratum, rows, spans_by_set, diff, copies,
                     [criteria(r.strata) for r in rows], set_flags)
    for b in table.values():
        for v in b.values():
            v['criteria'] = dict(sorted(v['criteria'].items()))
            v['flags'] = dict(sorted(v['flags'].items()))
    errors = [dict(key=r.unit.key, split=split_of(r), error=r.error) for r in results if r.strata is None]
    return dict(splits=dict(sorted(table.items())), errors=len(errors),
                errors_by_split=_count(e['split'] for e in errors), error_examples=[e['error'] for e in errors[:5]])


def _empty() -> dict:
    return dict(recordings=0, sets=0, heads=0, span_heads=0, pairs_differing=0, pairs_copies=0, criteria={},
                flags={})


def _add(v: dict, stratum: str, rows: Sequence[SetResult], spans_by_set, diff: int, copies: int,
         crit: Sequence[dict], set_flags: dict) -> None:
    v['recordings'] += 1
    v['sets'] += len(rows)
    v['heads'] += sum(r.strata.heads for r in rows)
    for r in rows:
        spans = None if spans_by_set is None else spans_by_set.get(r.unit.key)
        if spans is None:
            v['span_heads'] += r.strata.heads
        elif stratum == 'S4':
            v['span_heads'] += r.strata.s4_heads
        else:
            v['span_heads'] += r.strata.span_heads.get(stratum, 0)
    v['pairs_differing'] += diff
    v['pairs_copies'] += copies
    for name in {k for c in crit for k, ok in c.items() if ok}:
        v['criteria'][name] = v['criteria'].get(name, 0) + 1
    flags = {k for r in rows for k, ok in set_flags.get(r.unit.key, {}).items() if ok}
    for name in flags:
        v['flags'][name] = v['flags'].get(name, 0) + 1


def _count(items: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items()))


def e2_status(table: dict, *, holdout: str = 'holdout2', hard: str = 'hard-holdout',
              floors: Sequence[str] = ('fit_dev',), min_recordings: int = E2_MIN_RECORDINGS,
              min_pairs: int = E2_MIN_PAIRS) -> dict[str, dict]:
    """E2's rule per stratum: decided on ``holdout`` with 40 or more recordings there, else on the
    ``hard`` holdout with 40 or more, else characterises; a floor counts with 10 or more differing
    pairs (reported for ``holdout``, ``hard`` and each of ``floors``)."""
    splits = table['splits']
    out = {}
    for s in STRATA:
        n_h = splits.get(holdout, {}).get(s, {}).get('recordings', 0)
        n_x = splits.get(hard, {}).get(s, {}).get('recordings', 0)
        if n_h >= min_recordings:
            decided = f'decided on {holdout}'
        elif n_x >= min_recordings:
            decided = f'decided on {hard}'
        else:
            decided = 'characterises only (the human\'s listening judges)'
        pairs = {name: splits.get(name, {}).get(s, {}).get('pairs_differing', 0) for name in (holdout, hard, *floors)}
        out[s] = dict(name=NAMES[s], holdout_recordings=n_h, hard_recordings=n_x, status=decided,
                      differing_pairs=pairs, floor={k: v >= min_pairs for k, v in pairs.items()})
    out['S10'] = dict(name=NAMES['S10'], status='not computed: needs BeatThis logits; characterises only by design')
    return out


# ---------------------------------------------------------------- the hard-strata holdout

def draw(results: Sequence[SetResult], *, pool: str = 'std-taiko', per_stratum: int = DRAW_PER_STRATUM,
         keep_design: int = DRAW_KEEP_DESIGN, seed: int = DRAW_SEED, strata: Sequence[str] = STRATA) -> dict:
    """Draw the hard-strata holdout from labels alone.

    The pool is every recording whose sets all lie in split ``pool``. A
    stratum with ``N`` pool recordings gets ``per_stratum`` when ``N`` is at
    least ``per_stratum + keep_design``, else ``N // 2``, so design keeps some
    of every stratum. Strata are filled rarest first: recordings already drawn
    count towards every stratum they are in, and the rest are drawn uniformly
    without replacement (numpy ``default_rng(seed)``) from the stratum's pool
    recordings in sorted order. Returns the drawn recordings with their sets,
    and counts only.
    """
    by_rec: dict[str, list[SetResult]] = {}
    other: set[str] = set()
    for r in results:
        if r.strata is None:
            continue
        if r.unit.split == pool:
            by_rec.setdefault(r.unit.recording, []).append(r)
        else:
            other.add(r.unit.recording)
    recs = sorted(k for k in by_rec if k not in other)
    member = {k: set(recording_strata([r.strata for r in by_rec[k]])) for k in recs}
    pool_n = {s: sum(s in member[k] for k in recs) for s in strata}
    target = {s: (per_stratum if pool_n[s] >= per_stratum + keep_design else pool_n[s] // 2) for s in strata}
    rng = np.random.default_rng(seed)
    drawn: set[str] = set()
    order = sorted(strata, key=lambda s: (pool_n[s], s))
    for s in order:
        have = sum(s in member[k] for k in drawn)
        cand = [k for k in recs if s in member[k] and k not in drawn]
        need = max(target[s] - have, 0)
        if need and cand:
            pick = rng.choice(len(cand), size=min(need, len(cand)), replace=False)
            drawn |= {cand[i] for i in sorted(pick.tolist())}
    held = {s: sum(s in member[k] for k in drawn) for s in strata}
    rows = []
    for k in sorted(drawn):
        rows.append(dict(recording=k, sets=[dict(key=r.unit.key, set_id=r.unit.set_id, mode=r.unit.mode,
                                                 audio=r.audio, files=list(r.unit.files)) for r in by_rec[k]]))
    rule = dict(pool=f'recordings whose sets all lie in split {pool!r} ({len(recs)} recordings; '
                     f'{len(other & set(by_rec))} left out for sets in other splits)',
                per_stratum=per_stratum, keep_design=keep_design, seed=seed, order=order,
                target=target, pool_recordings=pool_n, held_recordings=held, drawn_recordings=len(drawn),
                method='rarest stratum first; numpy default_rng(seed).choice without replacement over the '
                       "stratum's undrawn pool recordings in sorted key order, up to the target less those "
                       'already drawn')
    return dict(rule=rule, members=rows)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _audio_path(root: Path, files: Sequence[str], audio: str) -> Path | None:
    """The set's audio file, matched without regard to case."""
    folder = (root / files[0]).parent
    exact = folder / audio
    if exact.is_file():
        return exact
    for p in folder.iterdir():
        if p.name.casefold() == audio.casefold() and p.is_file():
            return p
    return None


README = """# Hard-strata holdout {name} (sealed)

Drawn {date} from `dataset/std-taiko/` labels alone, before any N1 design looked at std-taiko,
by the E2 census of the N1 evaluation design (research notes `n1-eval-20261018` §5.2 and §7;
decision `d-n1-eval-answers`: a sealed hard-strata holdout is drawn now; holdout-2 stays for N1's
one decision look).

- `{membership}`: the drawn recordings, each with its sets (set key, beatmap-set id, mode, `.osu`
  files, audio path and audio sha256). Its sha256 is in `membership.sha256` and must match
  `{sha}`.
- `rule.json`: the drawing rule, seed, code commit, pool, targets and counts per stratum. Counts
  only.

**Who may open it, and when.** Only E7, N1's decision look, opens these recordings' labels and
audio, once, to score them. Until then:

- The membership is used only to exclude. Every training, design and analysis manifest that reads
  `dataset/std-taiko/` (or any copy of its audio) drops every set id, every audio sha256 and every
  recording listed here, and checks this file's sha256 first (`ensomi_n.eval.census.load_exclusion`).
- Nobody looks at the per-recording labels, charts, audio or results of these sets, including the
  census author; reports give counts per stratum and split only.
- Opening it early, or changing it, makes a new holdout: draw `v2` from recordings nobody has looked
  at, and record why.
"""


def seal(drawn: dict, out_dir: str | os.PathLike, root: str | os.PathLike = '.', *, commit: str = '',
         name: str = 'hard-holdout-v1', date: str = '') -> str:
    """Write the sealed holdout (membership with audio sha256, its sha256, the rule, a README).

    Refuses to overwrite an existing membership. Returns the membership's sha256.
    """
    out = Path(out_dir)
    root = Path(root)
    if (out / MEMBERSHIP).exists():
        raise ValueError(f'{out / MEMBERSHIP} exists; a sealed holdout is never redrawn in place')
    out.mkdir(parents=True, exist_ok=True)
    members = []
    missing = 0
    for rec in drawn['members']:
        sets = []
        for s in rec['sets']:
            path = _audio_path(root, s['files'], s['audio']) if s['audio'] else None
            missing += path is None
            sets.append(dict(s, audio_sha256=_sha256_file(path) if path else None))
        members.append(dict(recording=rec['recording'], sets=sets))
    text = json.dumps(dict(format='ensomi-n-hard-holdout', version=1, name=name, members=members),
                      sort_keys=True, indent=1) + '\n'
    (out / MEMBERSHIP).write_text(text, encoding='utf-8')
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    (out / 'membership.sha256').write_text(f'{sha}  {MEMBERSHIP}\n', encoding='utf-8')
    rule = dict(drawn['rule'], name=name, commit=commit, date=date, membership_sha256=sha,
                audio_missing=missing, sets=sum(len(m['sets']) for m in members))
    (out / 'rule.json').write_text(json.dumps(rule, sort_keys=True, indent=1) + '\n', encoding='utf-8')
    (out / 'README.md').write_text(README.format(name=name, date=date or 'on the date in rule.json',
                                                 membership=MEMBERSHIP, sha=sha), encoding='utf-8')
    return sha


def load_exclusion(path: str | os.PathLike, sha256: str) -> dict[str, set[str]]:
    """What a manifest must leave out: ``set_ids``, ``audio_sha256``, ``recordings`` and ``keys``.

    ``path`` is the holdout directory or its membership file; ``sha256`` the
    expected digest of the membership file. Raises ``ValueError`` when it does
    not match, so a manifest never runs against a changed or missing seal.
    """
    p = Path(path)
    p = p / MEMBERSHIP if p.is_dir() else p
    data = p.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != sha256.strip().lower():
        raise ValueError(f'{p}: sha256 {got} does not match the expected {sha256}')
    members = json.loads(data.decode('utf-8'))['members']
    sets = [s for m in members for s in m['sets']]
    return dict(set_ids={s['set_id'] for s in sets if s['set_id']},
                audio_sha256={s['audio_sha256'] for s in sets if s.get('audio_sha256')},
                recordings={m['recording'] for m in members}, keys={s['key'] for s in sets})


# ---------------------------------------------------------------- outputs

def strata_rows(results: Sequence[SetResult], *, sealed: Callable[[SetResult], bool],
                split_of: Callable[[SetResult], str] | None = None) -> list[dict]:
    """Rows of the eval strata table (`read_strata` format) for every set not in a sealed split."""
    groups = _group([r for r in results if not sealed(r)], split_of or (lambda r: r.unit.split))
    rows = []
    for (split, rec), sets in sorted(groups.items()):
        status = {r.unit.key: r.unit.status for r in sets}
        for stratum, spans_by_set in sorted(recording_strata([r.strata for r in sets]).items()):
            for key, spans in sorted(spans_by_set.items()):
                base = dict(key=key, recording=rec, stratum=stratum, source=status.get(key, ''), split=split)
                if spans is None:
                    rows.append(dict(base, start_ms='', end_ms=''))
                else:
                    rows += [dict(base, start_ms=f'{a:.3f}', end_ms=f'{b:.3f}') for a, b in spans]
    return rows


def markdown(table: dict, status: dict | None = None, *, splits: Sequence[str] | None = None) -> str:
    """The census as Markdown tables: recordings / sets / span heads / differing pairs per stratum and split."""
    names = list(splits) if splits else [n for n in table['splits'] if '|' not in n]
    lines = ['| Stratum | ' + ' | '.join(names) + ' |', '| --- |' + ' ---: |' * len(names)]
    for s in ('all',) + STRATA:
        cells = []
        for n in names:
            v = table['splits'].get(n, {}).get(s)
            cells.append(f"{v['recordings']} / {v['sets']} / {v['span_heads']} / {v['pairs_differing']}" if v else '0')
        label = 'all' if s == 'all' else f'{s} {NAMES[s]}'
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    lines.append('')
    lines.append('Cells: recordings / sets / heads in the stratum\'s spans / differing second-mapper pairs.')
    if status:
        lines += ['', '| Stratum | Holdout recordings | Hard-holdout recordings | E2 status | Differing pairs |',
                  '| --- | ---: | ---: | --- | --- |']
        for s, v in status.items():
            if s == 'S10':
                lines.append(f"| S10 {v['name']} | | | {v['status']} | |")
                continue
            pairs = ', '.join(f'{k} {n}' + (' (floor)' if v['floor'][k] else '') for k, n in v['differing_pairs'].items())
            lines.append(f"| {s} {v['name']} | {v['holdout_recordings']} | {v['hard_recordings']} | {v['status']} | "
                         f'{pairs} |')
    return '\n'.join(lines) + '\n'


def write_census(out_dir: str | os.PathLike, table: dict, status: dict, rows: list[dict], *,
                 meta: dict | None = None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc = dict(meta or {}, strata=dict(NAMES), hard=list(HARD), needs_logits=list(NEEDS_LOGITS),
               e2=status, table=table)
    (out / 'census.json').write_text(json.dumps(_jsonable(doc), indent=1, sort_keys=True) + '\n', encoding='utf-8')
    (out / 'census.md').write_text(markdown(table, status), encoding='utf-8')
    by_source: dict[str, list[dict]] = {}
    for r in rows:
        by_source.setdefault(r.get('split', ''), []).append(r)
    for split, part in by_source.items():
        with (out / f'strata-{split}.csv').open('w', newline='', encoding='utf-8') as handle:
            w = csv.DictWriter(handle, fieldnames=['key', 'recording', 'stratum', 'source', 'start_ms', 'end_ms'],
                               extrasaction='ignore')
            w.writeheader()
            w.writerows(part)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in (sorted(x) if isinstance(x, (set, frozenset)) else x)]
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x
