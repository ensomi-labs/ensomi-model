#!/usr/bin/env python3
"""Export the public R2 data-provenance tables from the count and label tables of the extraction.

The provenance extraction of 2026-10-17 wrote four tables from private per-chart records: a count
table for the 26 core stages, one for the 12 supplemental roles, and two combined label tables that
list every exact ``Creator`` or ``Artist`` label in each of the 38 stages. This script checks each
input against its recorded SHA-256 and the invariants below, then writes into ``--out``:

- ``stage_counts.csv``: one row per stage, built from the two count tables;
- ``creator_labels_by_stage.csv`` and ``artist_labels_by_stage.csv``: the label tables' rows,
  values and order unchanged, when both label inputs are given;
- ``export_record.json``: the sources, the transformations and each output's rows and SHA-256.
  Without the label inputs it marks the two label tables pending.

Every check runs and every output is built and parsed back before the first file is written; a
failed check writes nothing. Standard library only::

    python scripts/provenance/r2_tables.py \\
        --core-counts <dir>/stage_counts.csv \\
        --supplemental-counts <dir>/calibration_lineage_counts.csv \\
        [--creator-labels <dir>/mapper_all_stages.csv --artist-labels <dir>/artist_all_stages.csv] \\
        --out docs/research/r2_data_provenance
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from pathlib import Path

EVIDENCE_DATE = '2026-10-17'
SCRIPT = 'scripts/provenance/r2_tables.py'

CORE_STAGES = (
    'frozen_corpus_inventory', 'r2_cache_fit_train', 'r2_cache_fit_dev', 'rows_fit_train', 'rows_fit_dev',
    'prior_fit_train', 'prior_fit_dev', 'calibration_song_population', 'calibration_scope_population',
    'lens_D_measured', 'lens_V2_measured', 'calibration_lens_stream_grades', 'calibration_lens_jack_grades',
    'calibration_lens_jumpstream', 'calibration_lens_chordstream', 'calibration_lens_chordjack',
    'calibration_difficulty_pilot', 'timing_tuning_observed', 'timing_dev_observed', 'timing_holdout_successful',
    'r2_selected56M_guard_evaluation', 'rows_evaluation_panel96', 'system_evaluation_panel',
    'early_listening_selection', 'final_listening_selection', 'release_preview_source_selection',
)
SUPPLEMENTAL_STAGES = (
    'kappa_comparison_panel', 'kappa_original_mean_donors', 'kappa_original_human_null', 'kappa_original_pair_draws',
    'kappa_followup_x0_sources', 'kappa_followup_lookup_candidates', 'kappa_followup_drawn_donors',
    'kappa_followup_human_drift_null', 'rows_quiet_rests_calibration', 'system_default_q_calibration',
    'natural_nll_validation', 'run_freerun_manifest',
)

# The SHA-256 of each input as the extraction recorded it.
EXPECTED_SHA256 = {
    'core_counts': '82c888ece56985d5ad3bd71d3974b77721b9f53852fd48a627f8d302f279c969',
    'supplemental_counts': '883a19c09daa46ba3921d22dc936a8ba62e0fb0ce57fea70a3083447c53470d1',
    'creator_labels': 'de3d82b01e7e605826bfa81e7a48747a7414e58b313599a41de6a48111cc99f6',
    'artist_labels': '8045d7d123566c817b80542c46cd442dff11f78cdba0ceb8dcd4860977676470',
}
SOURCES = {
    'core_counts': ('stage_counts.csv', 'counts for each of the 26 core stages'),
    'supplemental_counts': ('calibration_lineage_counts.csv', 'counts for each of the 12 supplemental roles'),
    'creator_labels': ('mapper_all_stages.csv', 'every exact Creator label in each of the 38 stages'),
    'artist_labels': ('artist_all_stages.csv', 'every exact Artist label in each of the 38 stages'),
}

CORE_HEADER = (
    'stage', 'source_records', 'distinct_charts', 'unmatched_charts', 'distinct_groups', 'unmatched_source_rows',
    'distinct_audio_hashes', 'distinct_positive_beatmap_set_ids', 'missing_or_nonpositive_beatmap_id',
    'missing_or_nonpositive_beatmap_set_id', 'distinct_creator_labels', 'distinct_artist_labels', 'missing_creator',
    'missing_artist', 'missing_audio_hash', 'cache_roles', 'corpus_eval_splits', 'source', 'rule',
)
SUPPLEMENTAL_HEADER = (
    'stage', 'charts', 'groups', 'audio_hashes', 'raw_creator_labels', 'raw_artist_labels', 'record_occurrences',
    'unmatched', 'outside_existing_measured_eligible_union', 'cache_fit_train', 'cache_fit_dev',
    'calibration_song_population', 'difficulty_pilot', 'rows_panel96', 'evidence_path',
)
LABEL_HEADER = (
    'stage', 'raw_label', 'charts', 'stage_chart_denominator', 'distinct_audio_hashes', 'distinct_groups',
    'evidence_group',
)
LABEL_RENAMES = {'distinct_audio_hashes': 'audio_hashes', 'distinct_groups': 'groups', 'evidence_group': 'population'}
LABEL_OUT_HEADER = tuple(LABEL_RENAMES.get(c, c) for c in LABEL_HEADER)
LABEL_OUTPUTS = {'creator_labels': 'creator_labels_by_stage.csv', 'artist_labels': 'artist_labels_by_stage.csv'}

STAGE_OUT_HEADER = (
    'stage', 'population', 'description', 'charts', 'groups', 'audio_hashes', 'creator_labels', 'artist_labels',
    'source_records', 'source_record_unit', 'positive_beatmap_set_ids', 'missing_or_nonpositive_beatmap_id',
    'missing_or_nonpositive_beatmap_set_id', 'missing_creator', 'missing_artist', 'missing_audio_hash',
    'charts_fit_train', 'charts_fit_dev', 'charts_outside_fit',
)
CORE_ONLY = (
    'positive_beatmap_set_ids', 'missing_or_nonpositive_beatmap_id', 'missing_or_nonpositive_beatmap_set_id',
    'missing_creator', 'missing_artist', 'missing_audio_hash',
)
CORE_MAP = {
    'charts': 'distinct_charts', 'groups': 'distinct_groups', 'audio_hashes': 'distinct_audio_hashes',
    'creator_labels': 'distinct_creator_labels', 'artist_labels': 'distinct_artist_labels',
    'source_records': 'source_records', 'positive_beatmap_set_ids': 'distinct_positive_beatmap_set_ids',
    'missing_or_nonpositive_beatmap_id': 'missing_or_nonpositive_beatmap_id',
    'missing_or_nonpositive_beatmap_set_id': 'missing_or_nonpositive_beatmap_set_id',
    'missing_creator': 'missing_creator', 'missing_artist': 'missing_artist',
    'missing_audio_hash': 'missing_audio_hash',
}
SUPPLEMENTAL_MAP = {
    'charts': 'charts', 'groups': 'groups', 'audio_hashes': 'audio_hashes', 'creator_labels': 'raw_creator_labels',
    'artist_labels': 'raw_artist_labels', 'source_records': 'record_occurrences',
    'charts_fit_train': 'cache_fit_train', 'charts_fit_dev': 'cache_fit_dev',
}
CACHE_ROLE_COLUMNS = {'fit_train': 'charts_fit_train', 'fit_dev': 'charts_fit_dev', 'not_in_cache': 'charts_outside_fit'}

# Public description and source-record unit of each stage, written for this export.
STAGES = {
    'frozen_corpus_inventory': ('Frozen corpus: every chart in the inventory; not a training population',
                                'corpus row'),
    'r2_cache_fit_train': ('R2: eligible weight-training charts', 'cache index row'),
    'r2_cache_fit_dev': ('R2: development pool', 'cache index row'),
    'rows_fit_train': ('Rows model: eligible weight-training charts', 'rows index row'),
    'rows_fit_dev': ('Rows model: development pool', 'rows index row'),
    'prior_fit_train': ('Corpus prior: charts with a numerical row in the published prior', 'cache index row'),
    'prior_fit_dev': ('Corpus prior: charts used to select the prediction method', 'cache index row'),
    'calibration_song_population': ('Calibration: training charts with at least 256 head rows',
                                    'calibration chart-table row'),
    'calibration_scope_population': ('Calibration: charts of the 132,053 admitted scopes', 'scope window'),
    'lens_D_measured': ('Lens D: charts of measured sections with machine labels', 'Lens section'),
    'lens_V2_measured': ('Lens V2: charts of measured sections with human validation labels', 'Lens section'),
    'calibration_lens_stream_grades': ('Calibration: Lens D sections with a positive stream grade',
                                       'Lens section'),
    'calibration_lens_jack_grades': ('Calibration: Lens D sections with a positive jack grade', 'Lens section'),
    'calibration_lens_jumpstream': ('Calibration: Lens D jumpstream references', 'Lens section'),
    'calibration_lens_chordstream': ('Calibration: Lens D chordstream references', 'Lens section'),
    'calibration_lens_chordjack': ('Calibration: Lens D chordjack references', 'Lens section'),
    'calibration_difficulty_pilot': ('Calibration: difficulty response pilot on supplied source head rows',
                                     'pilot selection entry'),
    'timing_tuning_observed': ('Timing system: observed tuning charts', 'timing record, per chart and arm'),
    'timing_dev_observed': ('Timing system: observed development charts', 'timing record, per chart and arm'),
    'timing_holdout_successful': ('Timing system: evaluated charts of the final holdout, all inside R2 fit_train',
                                  'timing record, per chart and arm'),
    'r2_selected56M_guard_evaluation': ('R2: corrected checkpoint guard panel of the selected checkpoint',
                                        'guard evaluation record, per chart, seed and start'),
    'rows_evaluation_panel96': ('Rows model: final evaluation panel over 96 audio hashes', 'panel chart entry'),
    'system_evaluation_panel': ('End-to-end system: evaluation panel', 'panel entry'),
    'early_listening_selection': ('Early listening: selected source charts; selection is not proof of playback',
                                  'listening slot'),
    'final_listening_selection': ('Final listening: selected source charts; selection is not proof of playback',
                                  'listening slot'),
    'release_preview_source_selection': ('Release previews: source chart selections', 'play slot'),
    'kappa_comparison_panel': ('Restoring force: gain comparison panel', 'panel entry'),
    'kappa_original_mean_donors': ('Restoring force: original nearest-32 mean-target donors',
                                   'donor-list entry, 32 per panel chart'),
    'kappa_original_human_null': ('Restoring force: original human-pair null references',
                                  'pair member, two per pair'),
    'kappa_original_pair_draws': ('Restoring force: drawn pair-difference target sources',
                                  'pair member in a target draw, two per draw'),
    'kappa_followup_x0_sources': ('Restoring force follow-up: diagnostic sources outside the comparison panel',
                                  'draw record'),
    'kappa_followup_lookup_candidates': ('Restoring force follow-up: nearest-32 candidate donors',
                                         'candidate-list entry, 32 per lookup'),
    'kappa_followup_drawn_donors': ('Restoring force follow-up: drawn joint-target donors', 'draw record'),
    'kappa_followup_human_drift_null': ('Restoring force follow-up: admitted human-drift null references',
                                        'pair member, two per admitted pair'),
    'rows_quiet_rests_calibration': ('Rows decoding: quiet-rest default selection, first chart of each panel '
                                     'audio item', 'panel audio item'),
    'system_default_q_calibration': ('End-to-end system: default difficulty-quantile selection, reconstructed '
                                     'from the controls subset', 'panel entry with controls'),
    'natural_nll_validation': ('R2 run: natural-NLL validation manifest', 'manifest window'),
    'run_freerun_manifest': ('R2 run: free-generation manifest', 'manifest entry'),
}

_COUNT = re.compile(r'(0|[1-9][0-9]*)\Z')
_PATHLIKE = re.compile(r'[/\\]|\.(csv|jsonl?|parquet|py|md|osu|osz|pt)\b|artifacts|dataset', re.IGNORECASE)
FORMULA_START = ('=', '+', '-', '@')


class ExportError(ValueError):
    """An input or output failed a check; nothing was written."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def line_endings(text: str) -> str:
    if '\r' not in text:
        return 'LF'
    return 'CRLF' if text.count('\r\n') == text.count('\n') == text.count('\r') else 'mixed'


def read_table(key: str, data: bytes, expected_sha256: str, header: tuple) -> tuple[list[dict], dict]:
    """Rows of one input CSV as dicts of strings, after its hash, encoding and header checks."""
    name = SOURCES[key][0]
    digest = sha256(data)
    if digest != expected_sha256:
        raise ExportError(f'{name}: SHA-256 {digest} differs from the expected {expected_sha256}')
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise ExportError(f'{name}: not valid UTF-8 ({exc})') from None
    if text.startswith('\ufeff'):
        raise ExportError(f'{name}: starts with a byte-order mark')
    records = list(csv.reader(io.StringIO(text, newline='')))
    if not records or tuple(records[0]) != header:
        raise ExportError(f'{name}: header {records[0] if records else []} differs from {list(header)}')
    rows = []
    for number, record in enumerate(records[1:], 2):
        if len(record) != len(header):
            raise ExportError(f'{name}: record {number} has {len(record)} fields, expected {len(header)}')
        rows.append(dict(zip(header, record)))
    info = dict(name=name, contents=SOURCES[key][1], sha256=digest, rows=len(rows), line_endings=line_endings(text))
    return rows, info


def count(name: str, row: dict, field: str) -> int:
    value = row[field]
    if not _COUNT.match(value):
        raise ExportError(f'{name}: {row["stage"]}: {field} {value!r} is not a non-negative decimal integer')
    return int(value)


def check_stage_ids(name: str, rows: list[dict], expected: tuple) -> None:
    got = tuple(r['stage'] for r in rows)
    if got != expected:
        raise ExportError(f'{name}: stages {list(got)} differ from the {len(expected)} expected, in order')


def at_most(name: str, stage: str, values: dict, limit_field: str) -> None:
    for field, value in values.items():
        if value > values[limit_field]:
            raise ExportError(f'{name}: {stage}: {field} {value} exceeds {limit_field} {values[limit_field]}')


def core_row(row: dict) -> dict:
    name, stage = SOURCES['core_counts'][0], row['stage']
    for field in ('unmatched_charts', 'unmatched_source_rows'):
        if count(name, row, field):
            raise ExportError(f'{name}: {stage}: {field} is {row[field]}, expected 0')
    out = {column: count(name, row, field) for column, field in CORE_MAP.items()}
    at_most(name, stage, {k: v for k, v in out.items() if k != 'source_records'}, 'charts')
    try:
        roles = json.loads(row['cache_roles'])
    except json.JSONDecodeError as exc:
        raise ExportError(f'{name}: {stage}: cache_roles is not JSON ({exc})') from None
    if not isinstance(roles, dict) or not set(roles) <= set(CACHE_ROLE_COLUMNS):
        raise ExportError(f'{name}: {stage}: cache_roles {roles!r} has keys outside {sorted(CACHE_ROLE_COLUMNS)}')
    for role, column in CACHE_ROLE_COLUMNS.items():
        value = roles.get(role, 0)
        if type(value) is not int or value < 0:
            raise ExportError(f'{name}: {stage}: cache_roles[{role!r}] is {value!r}')
        out[column] = value
    if sum(out[c] for c in CACHE_ROLE_COLUMNS.values()) != out['charts']:
        raise ExportError(f'{name}: {stage}: cache roles sum to {sum(roles.values())}, not {out["charts"]} charts')
    return out


def supplemental_row(row: dict) -> dict:
    name, stage = SOURCES['supplemental_counts'][0], row['stage']
    if count(name, row, 'unmatched'):
        raise ExportError(f'{name}: {stage}: unmatched is {row["unmatched"]}, expected 0')
    out = {column: count(name, row, field) for column, field in SUPPLEMENTAL_MAP.items()}
    at_most(name, stage, {k: v for k, v in out.items() if k != 'source_records'}, 'charts')
    out['charts_outside_fit'] = out['charts'] - out['charts_fit_train'] - out['charts_fit_dev']
    if out['charts_outside_fit'] < 0:
        raise ExportError(f'{name}: {stage}: cache_fit_train + cache_fit_dev exceeds charts')
    return out


def stage_rows(core: list[dict], supplemental: list[dict]) -> list[dict]:
    """The rows of stage_counts.csv: core stages, then supplemental roles, in source order."""
    check_stage_ids(SOURCES['core_counts'][0], core, CORE_STAGES)
    check_stage_ids(SOURCES['supplemental_counts'][0], supplemental, SUPPLEMENTAL_STAGES)
    rows = []
    for population, source, build in (('core', core, core_row), ('supplemental', supplemental, supplemental_row)):
        for row in source:
            values = build(row)
            description, unit = STAGES[row['stage']]
            rows.append({'stage': row['stage'], 'population': population, 'description': description,
                         'source_record_unit': unit, **values})
    by_stage = {r['stage']: r for r in rows}
    # Supplemental label counts count distinct values, an empty one included; core counts count nonempty
    # labels. They agree only while every supplemental chart lies in fit_train or fit_dev and neither has
    # an empty label, which the export record states.
    fit = [by_stage['r2_cache_fit_train'], by_stage['r2_cache_fit_dev']]
    if any(r[c] for r in fit for c in ('missing_creator', 'missing_artist')):
        raise ExportError('fit_train or fit_dev has an empty Creator or Artist; label-count definitions differ')
    for stage in SUPPLEMENTAL_STAGES:
        if by_stage[stage]['charts_outside_fit']:
            raise ExportError(f'{stage}: {by_stage[stage]["charts_outside_fit"]} charts outside fit_train and fit_dev')
    return rows


def label_rows(key: str, rows: list[dict], stages: list[dict]) -> None:
    """Check one combined label table against the stage counts; raises ExportError on the first failure."""
    name = SOURCES[key][0]
    field = key                     # the stage rows' creator_labels or artist_labels
    expected = {s['stage']: s for s in stages}
    seen, labels, charts = set(), {}, {}
    for number, row in enumerate(rows, 2):
        stage = row['stage']
        if stage not in expected:
            raise ExportError(f'{name}: record {number}: unknown stage {stage!r}')
        if row['evidence_group'] != expected[stage]['population']:
            raise ExportError(f'{name}: record {number}: evidence_group {row["evidence_group"]!r} for {stage}')
        label = row['raw_label']
        if label == '':
            raise ExportError(f'{name}: record {number}: empty raw_label in {stage}')
        if (stage, label) in seen:
            raise ExportError(f'{name}: record {number}: repeated label in {stage}')
        seen.add((stage, label))
        values = {f: count(name, row, f) for f in ('charts', 'stage_chart_denominator', 'distinct_audio_hashes',
                                                   'distinct_groups')}
        if values['charts'] < 1:
            raise ExportError(f'{name}: record {number}: charts is 0 in {stage}')
        if values['stage_chart_denominator'] != expected[stage]['charts']:
            raise ExportError(f'{name}: record {number}: stage_chart_denominator {values["stage_chart_denominator"]} '
                              f'differs from the {expected[stage]["charts"]} charts of {stage}')
        at_most(name, stage, {k: values[k] for k in ('charts', 'distinct_audio_hashes', 'distinct_groups')},
                'charts')
        labels[stage] = labels.get(stage, 0) + 1
        charts[stage] = charts.get(stage, 0) + values['charts']
    for stage, s in expected.items():
        if labels.get(stage, 0) != s[field]:
            raise ExportError(f'{name}: {stage} has {labels.get(stage, 0)} label rows, expected {s[field]}')
        if charts.get(stage, 0) != s['charts']:
            raise ExportError(f'{name}: {stage}: label charts sum to {charts.get(stage, 0)}, expected {s["charts"]}')


def label_report(rows: list[dict]) -> dict:
    """Counts of labels a spreadsheet or a reader may mishandle, over rows and over distinct labels."""
    def counts(labels):
        return {'leading_formula_character': sum(l.startswith(FORMULA_START) for l in labels),
                'leading_or_trailing_whitespace': sum(l != l.strip() for l in labels),
                'non_ascii': sum(not l.isascii() for l in labels)}
    labels = [r['raw_label'] for r in rows]
    return {'rows': counts(labels), 'distinct_labels': counts(sorted(set(labels)))}


def csv_field(value: str) -> str:
    if any(c in value for c in ',"\r\n'):
        return '"' + value.replace('"', '""') + '"'
    return value


def encode_csv(header: tuple, rows: list[list[str]]) -> bytes:
    """RFC 4180 fields, LF line endings, UTF-8 without a byte-order mark."""
    return ''.join(','.join(csv_field(v) for v in record) + '\n' for record in [list(header), *rows]).encode('utf-8')


def verify_output(name: str, data: bytes, header: tuple, rows: list[list[str]], label_column: int | None) -> None:
    """Parse an output back and require exactly the intended values; no path-like text outside labels."""
    text = data.decode('utf-8')
    if text.startswith('\ufeff'):
        raise ExportError(f'{name}: byte-order mark in output')
    parsed = list(csv.reader(io.StringIO(text, newline='')))
    if parsed != [list(header), *rows]:
        raise ExportError(f'{name}: output does not parse back to the intended values')
    for record in parsed:
        for column, value in enumerate(record):
            if column != label_column and _PATHLIKE.search(value):
                raise ExportError(f'{name}: path-like text {value!r} outside the label column')


def stage_output(rows: list[dict]) -> list[list[str]]:
    out = []
    for row in rows:
        out.append(['' if row['population'] == 'supplemental' and c in CORE_ONLY else str(row[c])
                    for c in STAGE_OUT_HEADER])
    return out


def transformations(with_labels: bool) -> list[dict]:
    steps = [{
        'output': 'stage_counts.csv',
        'from': ['stage_counts.csv', 'calibration_lineage_counts.csv'],
        'rows': 'one per stage: the 26 core stages, then the 12 supplemental roles, each in source order; '
                'stage ids unchanged',
        'added': {'population': "'core' for rows of the core table, 'supplemental' for the other",
                  'description': 'written for this export', 'source_record_unit': 'written for this export'},
        'core_columns': {**{out: src for out, src in CORE_MAP.items()},
                         'charts_fit_train': 'cache_roles["fit_train"], 0 when absent',
                         'charts_fit_dev': 'cache_roles["fit_dev"], 0 when absent',
                         'charts_outside_fit': 'cache_roles["not_in_cache"], 0 when absent'},
        'supplemental_columns': {**SUPPLEMENTAL_MAP,
                                 'charts_outside_fit': 'charts - cache_fit_train - cache_fit_dev; the two cache '
                                                       'roles are disjoint by chart',
                                 **{c: 'empty: not recorded for supplemental roles' for c in CORE_ONLY}},
        'dropped': {'core': ['unmatched_charts (checked to be 0)', 'unmatched_source_rows (checked to be 0)',
                             'cache_roles (split into the three charts_ columns)', 'corpus_eval_splits',
                             'source (local paths)', 'rule (internal extraction rule)'],
                    'supplemental': ['unmatched (checked to be 0)', 'outside_existing_measured_eligible_union',
                                     'calibration_song_population', 'difficulty_pilot', 'rows_panel96',
                                     'evidence_path (local paths)']},
        'notes': ['Supplemental label counts count distinct values including an empty one; core label counts '
                  'count nonempty labels. Every supplemental chart lies in fit_train or fit_dev, and neither has '
                  'an empty Creator or Artist, so the two definitions give the same counts here.'],
    }]
    for key, output in LABEL_OUTPUTS.items():
        steps.append({
            'output': output,
            'from': [SOURCES[key][0]],
            'rows': 'every source row, in source order; no row dropped, merged or added',
            'values': 'unchanged, raw_label byte for byte',
            'header_renamed': LABEL_RENAMES,
            'line_endings': 'source line endings replaced by LF',
            'status': 'written' if with_labels else 'pending: source table not available to this export',
        })
    return steps


def export(inputs: dict[str, bytes], expected: dict[str, str] = EXPECTED_SHA256) -> dict[str, bytes]:
    """Every output file name -> bytes, after every check. ``inputs`` holds the two count tables and either
    both label tables or neither."""
    labels_given = [k for k in LABEL_OUTPUTS if k in inputs]
    if labels_given and len(labels_given) != len(LABEL_OUTPUTS):
        raise ExportError('give both label tables or neither')
    core, core_info = read_table('core_counts', inputs['core_counts'], expected['core_counts'], CORE_HEADER)
    supp, supp_info = read_table('supplemental_counts', inputs['supplemental_counts'],
                                 expected['supplemental_counts'], SUPPLEMENTAL_HEADER)
    stages = stage_rows(core, supp)
    for s in stages:
        for field in ('description', 'source_record_unit'):
            if _PATHLIKE.search(s[field]):
                raise ExportError(f'{s["stage"]}: path-like text in {field}')
    outputs = {}
    table = stage_output(stages)
    outputs['stage_counts.csv'] = encode_csv(STAGE_OUT_HEADER, table)
    verify_output('stage_counts.csv', outputs['stage_counts.csv'], STAGE_OUT_HEADER, table, None)

    sources, reports = [core_info, supp_info], {}
    for key, output in LABEL_OUTPUTS.items():
        expected_rows = sum(s[key] for s in stages)     # a stage row's creator_labels or artist_labels
        if not labels_given:
            sources.append(dict(name=SOURCES[key][0], contents=SOURCES[key][1], sha256=expected[key],
                                rows=expected_rows, status='not available to this export; sha256 is the recorded '
                                'hash the export requires, and rows the sum of the stages\' label counts'))
            continue
        rows, info = read_table(key, inputs[key], expected[key], LABEL_HEADER)
        label_rows(key, rows, stages)
        sources.append(info)
        values = [[r[c] for c in LABEL_HEADER] for r in rows]
        outputs[output] = encode_csv(LABEL_OUT_HEADER, values)
        verify_output(output, outputs[output], LABEL_OUT_HEADER, values, LABEL_OUT_HEADER.index('raw_label'))
        reports[output] = dict(rows=len(values), labels=label_report(rows))

    files = [{'name': 'stage_counts.csv', 'rows': len(table), 'sha256': sha256(outputs['stage_counts.csv'])}]
    for key, output in LABEL_OUTPUTS.items():
        if output in outputs:
            files.append({'name': output, 'rows': reports[output]['rows'], 'sha256': sha256(outputs[output]),
                          'labels_a_reader_may_mishandle': reports[output]['labels']})
        else:
            files.append({'name': output, 'status': 'pending', 'expected_rows': sum(s[key] for s in stages)})
    record = {
        'package': 'Ensomi R2 data provenance tables',
        'evidence_date': EVIDENCE_DATE,
        'export_script': SCRIPT,
        'sources': sources,
        'transformations': transformations(bool(labels_given)),
        'format': 'UTF-8 without byte-order mark; comma-separated; RFC 4180 quoting (a field is quoted when it '
                  'holds a comma, double quote, CR or LF, and inner quotes are doubled); LF line endings',
        'outputs': files,
    }
    outputs['export_record.json'] = (json.dumps(record, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    return outputs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--core-counts', type=Path, required=True)
    parser.add_argument('--supplemental-counts', type=Path, required=True)
    parser.add_argument('--creator-labels', type=Path)
    parser.add_argument('--artist-labels', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    paths = {'core_counts': args.core_counts, 'supplemental_counts': args.supplemental_counts,
             'creator_labels': args.creator_labels, 'artist_labels': args.artist_labels}
    inputs = {k: p.read_bytes() for k, p in paths.items() if p is not None}
    try:
        outputs = export(inputs)
        if not any(k in inputs for k in LABEL_OUTPUTS):
            present = [n for n in LABEL_OUTPUTS.values() if (args.out / n).exists()]
            if present:
                raise ExportError(f'{present} exist in {args.out}; give the label inputs so the record matches')
    except ExportError as exc:
        print(f'r2_tables: {exc}; nothing written', file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in outputs.items():
        (args.out / name).write_bytes(data)
    print(json.dumps({name: {'bytes': len(data), 'sha256': sha256(data)} for name, data in outputs.items()},
                     indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
