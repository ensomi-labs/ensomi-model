# R2 data provenance tables

These tables accompany [R2 data provenance](../r2_data_provenance.md), which
describes each stage and the limits of the evidence. They cover **Ensomi R2
1.0 and 1.0.1** only, from the evidence snapshot of **2026-10-17**. They count
source charts and the metadata labels those charts carry in each stage of
R2's training, calibration, selection and evaluation. They are attribution
records: a label is a claim in a chart's metadata, and appearing in a table is
not a rights clearance or a finding of permission.

| File | Contents | Status |
| --- | --- | --- |
| [`stage_counts.csv`](stage_counts.csv) | one row for each of the 38 stages | published |
| `creator_labels_by_stage.csv` | every exact `Creator` label in each stage, 13,568 rows | pending |
| `artist_labels_by_stage.csv` | every exact `Artist` label in each stage, 23,949 rows | pending |
| [`export_record.json`](export_record.json) | sources and their SHA-256, transformations, each output's rows and SHA-256 | published |

## Method

- **Snapshot.** The counts describe the frozen corpus metadata and the stage
  records as they stood on 2026-10-17.
- **Joins.** Every stage record is joined to the frozen corpus by chart
  SHA-256; timing records, which store chart paths, join through the corpus's
  unique path column. Every record joined.
- **Labels.** A label is an exact nonempty frozen `Creator` or `Artist` string.
  No aliases were merged, no collaboration strings were split, and nothing was
  normalised: case, spacing, punctuation and Unicode are as recorded. A label
  is not a verified person or rights holder. `Creator` may omit a guest
  difficulty's author; `Artist` does not list composers, performers, remixers,
  sampled musicians or rights holders.
- **Units.** A *chart* is one distinct source `.osu` SHA-256, a single
  difficulty. An *audio hash* is a recorded SHA-256 of exact audio-file bytes;
  it is not a song, a composition or necessarily a distinct recording, and the
  audit did not rehash audio. A *group* is a leakage-control group, one of the
  groups used to keep related sources within the same data partition: it links
  charts that share a set folder or ID, a beatmap ID, a normalized artist and
  title, identical audio bytes or an older R1 group. A group is not a song
  either. A *source record* is one appearance of a chart in a stage's own
  record; what it is varies by stage and is named in `source_record_unit`.
  Neither groups nor audio hashes count unique songs.
- **Denominators.** Every count is within its stage. A label table's
  `stage_chart_denominator` is the stage's `charts` in `stage_counts.csv`.
- **Eligibility, not exposure.** A training stage lists the charts eligible
  for training. It does not show whether or how often each chart was sampled,
  or what any chart contributed to the weights or to a generated chart.
- **The inventory is not the training population.** `frozen_corpus_inventory`
  holds all 22,066 frozen charts; 11,368 of them are the weight-training
  population.

### Overlaps

Stages overlap; never add their counts to obtain a number of charts, mappers
or artists.

- `fit_train` and `fit_dev` are disjoint by chart, by group and by audio hash.
- R2, the rows model and the corpus prior use the identical 11,368 charts:
  `r2_cache_fit_train`, `rows_fit_train` and `prior_fit_train` are the same
  set, as are the three `fit_dev` stages.
- The timing system's final holdout, `timing_holdout_successful`, lies inside
  `fit_train`: it was held out from the timing experiment, not from R2's
  training charts.
- Lens V2, `lens_V2_measured`, has 130 `fit_train` and 10 `fit_dev` charts.
- Every supplemental role lies inside `fit_train` ∪ `fit_dev`.

## `stage_counts.csv`

Rows: the 26 core stages, then the 12 supplemental roles.

| Column | Unit | Meaning |
| --- | --- | --- |
| `stage` | id | stable stage id, the same in the label tables |
| `population` | `core` or `supplemental` | `core`: the stages of the main inventory; `supplemental`: the restoring-force, decoder-default and run-manifest roles |
| `description` | text | what the stage is |
| `charts` | charts | distinct source charts in the stage |
| `groups` | groups | distinct leakage-control groups of those charts |
| `audio_hashes` | audio hashes | distinct recorded audio hashes of those charts |
| `creator_labels` | labels | distinct exact `Creator` strings |
| `artist_labels` | labels | distinct exact `Artist` strings |
| `source_records` | source records | records of the stage before deduplication to charts |
| `source_record_unit` | text | what one source record is, such as a scope window or a Lens section |
| `positive_beatmap_set_ids` | IDs | distinct positive osu! beatmap set IDs |
| `missing_or_nonpositive_beatmap_id` | charts | charts without a positive beatmap ID |
| `missing_or_nonpositive_beatmap_set_id` | charts | charts without a positive set ID |
| `missing_creator`, `missing_artist` | charts | charts with an empty `Creator` or `Artist` |
| `missing_audio_hash` | charts | charts without a recorded audio hash |
| `charts_fit_train` | charts | the stage's charts in R2's training role, `fit_train` |
| `charts_fit_dev` | charts | the stage's charts in the development role, `fit_dev` |
| `charts_outside_fit` | charts | the stage's charts in neither |

The supplemental extraction did not record `positive_beatmap_set_ids` or the
five `missing_` columns, so they are empty for supplemental rows. An empty
cell there means not recorded, not zero.

## Label tables

`creator_labels_by_stage.csv` and `artist_labels_by_stage.csv` are pending:
the combined label tables they are exported from were not available to this
export. [`export_record.json`](export_record.json) records the SHA-256 each
source must have and the rows the stage counts imply. The rows are
label-in-stage pairs, not mappers or musicians.

Each table will hold every source row with its raw label byte for byte and its
stage's chart denominator; no row is dropped, merged or normalised. The export
renames three columns and replaces the source's CRLF line endings with LF, and
`export_record.json` lists both changes.

| Column | Unit | Meaning |
| --- | --- | --- |
| `stage` | id | the stage, as in `stage_counts.csv` |
| `raw_label` | text | the exact frozen label |
| `charts` | charts | the stage's charts that carry this label |
| `stage_chart_denominator` | charts | the stage's charts |
| `audio_hashes` | audio hashes | distinct recorded audio hashes of those charts |
| `groups` | groups | distinct leakage-control groups of those charts |
| `population` | `core` or `supplemental` | as in `stage_counts.csv` |

Rows run stage by stage in the order of `stage_counts.csv`. Within a core
stage, labels are ordered by descending `charts`, then by label; within a
supplemental role, by label. Labels sort by Unicode code point.

To look up a name, filter `stage`, for example to `r2_cache_fit_train` for
the 11,368 eligible training charts, then search `raw_label`. Labels are not
merged, so also search for part of a name: a name can appear in several
spellings or inside a collaboration string.

## File format

UTF-8 without a byte-order mark, comma-separated, a header row, LF line
endings and RFC 4180 quoting: a field that holds a comma, a double quote, CR or
LF is enclosed in double quotes, and a double quote inside it is doubled.

Labels are verbatim. Some begin with `=`, `+`, `-` or `@`, which a spreadsheet
may read as a formula, and some begin or end with spaces. Import the label
tables with every column as text and without trimming. The export record
counts such labels.

## What remains private

The per-chart records that these tables aggregate are derived from the private
dataset and are not published: the chart inventory with each chart's stage
memberships, the per-stage lists of chart hashes, the training cache and its
indices, the timing, guard and Lens records, the panel and listening
selections, and the supplemental roles' per-chart source rows. Repeating the
extraction needs those records.

## Regenerating the tables

[`scripts/provenance/r2_tables.py`](../../../scripts/provenance/r2_tables.py)
writes these files from the extraction's tables and checks each input's
SHA-256 against the recorded value; it refuses any other input. Its inputs are
private, so only their holders can run it:

```sh
python scripts/provenance/r2_tables.py \
  --core-counts <dir>/stage_counts.csv \
  --supplemental-counts <dir>/calibration_lineage_counts.csv \
  --creator-labels <dir>/mapper_all_stages.csv \
  --artist-labels <dir>/artist_all_stages.csv \
  --out docs/research/r2_data_provenance
```

## Corrections and requests

To correct an attribution, to ask how a chart or song was used, or to say that
you are interested in the enrollment and opt-out process being explored, open a
[data provenance issue](https://github.com/ensomi-labs/ensomi-model/issues/new?template=data-provenance.yml).
Issues are public: include no personal data and no audio. Opening an issue
does not by itself remove anything from a released model.
