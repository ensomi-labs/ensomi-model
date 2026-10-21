# R2 data provenance tables

Count and label tables for [R2 data provenance](../r2_data_provenance.md),
which describes each stage and the limits of the evidence. They cover
**Ensomi R2 1.0 and 1.0.1** at the evidence snapshot of **2026-10-17** and
record attribution, not rights clearance or permission.

| File | Contents | Status |
| --- | --- | --- |
| [`stage_counts.csv`](stage_counts.csv) | one row for each of the 38 stages | published |
| [`creator_labels_by_stage.csv`](creator_labels_by_stage.csv) | every exact `Creator` label in each stage, 13,568 rows | published |
| [`artist_labels_by_stage.csv`](artist_labels_by_stage.csv) | every exact `Artist` label in each stage, 23,949 rows | published |
| [`export_record.json`](export_record.json) | sources and their SHA-256, transformations, each output's rows and SHA-256 | published |

## Method

Every stage record joins the frozen corpus by chart SHA-256 (timing records
through its unique path column), and every record joined. A *chart* is one
distinct source `.osu` SHA-256, a single difficulty. A *label* is an exact
nonempty frozen `Creator` or `Artist` string, unnormalised, with no aliases
merged or collaborations split: a metadata claim, not a verified person or
rights holder. An *audio hash* is a recorded SHA-256 of exact audio-file bytes.
A *group* keeps related sources within the same data partition, linking charts
that share a set folder or ID, a beatmap ID, a normalized artist and title,
identical audio bytes or an older R1 group. Neither groups nor audio hashes
count songs. Every count is within its stage. A training stage lists eligible
charts, not whether or how often each was sampled; `frozen_corpus_inventory`,
all 22,066 frozen charts, is not the training population.

Stages overlap: never add their counts. `fit_train` and `fit_dev` are disjoint
by chart, group and audio hash. `r2_cache_fit_train`, `rows_fit_train` and
`prior_fit_train` are the same 11,368 charts, as are the three `fit_dev`
stages. `timing_holdout_successful` lies inside `fit_train`, held out from the
timing experiment, not from R2's training; `lens_V2_measured` has 130
`fit_train` and 10 `fit_dev` charts; every supplemental role lies inside
`fit_train` ∪ `fit_dev`.

## `stage_counts.csv`

Rows: the 26 core stages, then the 12 supplemental roles.

| Column | Unit | Meaning |
| --- | --- | --- |
| `stage` | id | stable stage id, the same in the label tables |
| `population` | `core` or `supplemental` | `core`: the stages of the main inventory; `supplemental`: the restoring-force, decoder-default and run-manifest roles |
| `description` | text | what the stage is |
| `charts` | charts | distinct source charts in the stage |
| `groups` | groups | distinct groups of those charts |
| `audio_hashes` | audio hashes | distinct recorded audio hashes of those charts |
| `creator_labels` | labels | distinct exact `Creator` strings |
| `artist_labels` | labels | distinct exact `Artist` strings |
| `source_records` | source records | the stage's records before deduplication to charts |
| `source_record_unit` | text | what one source record is, such as a scope window or a Lens section |
| `positive_beatmap_set_ids` | IDs | distinct positive osu! beatmap set IDs |
| `missing_or_nonpositive_beatmap_id` | charts | charts without a positive beatmap ID |
| `missing_or_nonpositive_beatmap_set_id` | charts | charts without a positive set ID |
| `missing_creator`, `missing_artist` | charts | charts with an empty `Creator` or `Artist` |
| `missing_audio_hash` | charts | charts without a recorded audio hash |
| `charts_fit_train` | charts | the stage's charts in R2's training role, `fit_train` |
| `charts_fit_dev` | charts | the stage's charts in the development role, `fit_dev` |
| `charts_outside_fit` | charts | the stage's charts in neither |

Supplemental rows leave `positive_beatmap_set_ids` and the five `missing_`
columns empty: not recorded, not zero.

## Label tables

A row of [`creator_labels_by_stage.csv`](creator_labels_by_stage.csv) or
[`artist_labels_by_stage.csv`](artist_labels_by_stage.csv) is a label in a
stage, not a person, with the raw label byte for byte; no row is dropped,
merged or normalised. The export renames three columns and converts CRLF line
endings to LF, as [`export_record.json`](export_record.json) lists.

| Column | Unit | Meaning |
| --- | --- | --- |
| `stage` | id | the stage, as in `stage_counts.csv` |
| `raw_label` | text | the exact frozen label |
| `charts` | charts | the stage's charts that carry this label |
| `stage_chart_denominator` | charts | the stage's `charts` |
| `audio_hashes` | audio hashes | distinct recorded audio hashes of those charts |
| `groups` | groups | distinct groups of those charts |
| `population` | `core` or `supplemental` | as in `stage_counts.csv` |

Rows follow the stage order of `stage_counts.csv`, then descending `charts` and
label within a core stage, label within a supplemental role (Unicode code
point order). To find a name, filter `stage` and search `raw_label`, also for
part of the name.

## File format

UTF-8 without a byte-order mark, comma-separated, a header row, LF line
endings and RFC 4180 quoting. Labels are verbatim: 49 `Creator` and three
`Artist` labels begin with `-`, which a spreadsheet may read as a formula, so
import every column as text; none begins or ends with whitespace. The export
record counts such labels.

## Regenerating the tables

[`scripts/provenance/r2_tables.py`](../../../scripts/provenance/r2_tables.py)
writes these files from the extraction's
[private](../r2_data_provenance.md#what-remains-private) tables, refusing any
input whose SHA-256 differs from the recorded value:

```sh
python scripts/provenance/r2_tables.py \
  --core-counts <dir>/stage_counts.csv \
  --supplemental-counts <dir>/calibration_lineage_counts.csv \
  --creator-labels <dir>/mapper_all_stages.csv \
  --artist-labels <dir>/artist_all_stages.csv \
  --out docs/research/r2_data_provenance
```
