# R2 data provenance: charts, mappers and music

Evidence snapshot: **2026-10-17**. This report traces source attribution and
data-use roles for **Ensomi R2 1.0 and 1.0.1**: timing, rows, arrangement,
corpus prior, calibration, selection, evaluation and previews. It does not
audit every historical Ensomi model or establish permission to use the source
charts and music. The [R2 system report](r2_system.md#code-and-training-lineage)
identifies the releases and their training lineage; its
[file inventory](r2_system.md#files-and-licences) lists the published files and
licence statements.

The released rows and arrangement models were trained from scratch, with the
same **11,368 eligible training charts**, carrying **1,129 distinct `Creator`
labels, 2,445 distinct `Artist` labels and 5,026 recorded exact audio-byte
hashes**. These charts belong to **4,167 leakage-control groups**, which are
not a count of songs. The rows model learns from music features and
mapper-supplied timing and head positions. R2 learns arrangements from chart
rows and actions, without audio features. The same corpus also supplies
numerical priors, decoder calibration, timing references and evaluation
material. The published prior retains per-chart numerical feature and target
rows. BeatThis contributes separate externally pretrained weights.

The complete frozen inventory contains 22,066 charts, 1,398 `Creator` labels
and 3,384 `Artist` labels. That inventory is not the weight-training corpus.
Counts below identify eligible or observed source membership; they do not
measure each source's sampled training exposure or contribution to an output.
Metadata labels identify neither verified people nor verified rights holders.

## Permissions and release policy — 2026-10-21

Ensomi will retain its existing released models with disclosure of their data
provenance. Future released models will use data covered by permissive
licences or explicit permission suitable for the intended training and
release. This policy is separate from the 2026-10-17 evidence snapshot; the
audit does not establish that the existing corpus meets that requirement.

Ensomi is exploring contributor enrollment and opt-out requests, including a
convenient way to verify that decisions come from the relevant chart or music
rights holders. The process and its effect on existing models have not been
established. No retroactive removal of a source's learned influence is claimed.

## Attribution labels

These are the ten most frequent exact metadata labels in the shared eligible
training population. Every chart count has the same denominator: **11,368
charts**. Ordering by chart count gives extra weight to music with several
mapped difficulties; the audio-hash column shows the corresponding number of
distinct recorded byte hashes for each artist label.

| `Creator` label | Charts / 11,368 |
| --- | ---: |
| Murumoo | 214 |
| keksikosu | 184 |
| FAMoss | 171 |
| ML-ysg | 122 |
| Drum-Hitnormal | 119 |
| Savageeeee | 117 |
| Carpihat | 111 |
| Kibitz | 104 |
| Syadow- | 99 |
| Ainer | 94 |

| `Artist` label | Charts / 11,368 | Audio hashes |
| --- | ---: | ---: |
| Camellia | 240 | 134 |
| Various Artists | 176 | 168 |
| xi | 93 | 31 |
| t+pazolite | 78 | 28 |
| Frums | 65 | 35 |
| Chroma | 62 | 23 |
| LeaF | 58 | 17 |
| lapix | 53 | 26 |
| Kobaryo | 51 | 14 |
| USAO | 51 | 18 |

The local searchable lists are [mapper labels by stage][mappers] and
[artist labels by stage][artists]. These CSV appendices include every
label across all 38 core and supplemental roles, with chart denominators,
audio counts and group counts. The
[chart inventory][charts] retains titles, difficulties, IDs, hashes and core
stage memberships. Supplemental calibration and selection roles have their
own linked tables below. These files are generated local evidence under
`artifacts/`; they are not shipped with the model or publicly packaged with
this report and may be absent in a fresh clone. Their local
[README][appendices] describes reproduction, which requires the named source
datasets and run artifacts.

`Creator` and `Artist` are source-header claims, not verified identities.
No aliases were merged or collaboration strings split. `Creator` may omit a
guest difficulty's author; `Artist` does not enumerate composers, performers,
remixers, sampled musicians or rights holders. “Various Artists” remains one
literal label. The columns therefore count labels, not people, and imply no
ownership or permission finding.

## Source collection, admission and identity

The audit starts from a local collection of osu!mania charts and records of
their associated audio. The [corpus builder][corpus-source] reads charts from
`dataset/0` and `dataset/supplemental`, associates available acquisition records
under `dataset/metadata/acquisitions/`, and freezes the inventory in
`data/r2-corpus.parquet`. This establishes the local inputs and their recorded
origins; it does not reconstruct a complete acquisition history or verify
permission from chart and music rights holders.

The [R2 admission rule][split-source] requires `eval_split == 'fit'`, ranked or
loved API status, and computed star rating **2 <= star <= 6**. Ranked/loved
status is a corpus-selection criterion, not evidence of training permission.
This training boundary differs from the released request interval documented
in the [system report](r2_system.md#what-you-can-ask-for).

Of 12,551 candidates, 12,531 were retained: 15 lacked audio, four failed
chart-legality checks (`illegal_source`), and one had a collision when nearby
head rows were merged. These are technical exclusions, not permission
findings. The retained cache has 11,368
`fit_train` charts and 1,163 `fit_dev` charts. Their group assignments are
disjoint. The corpus's separate `calibration` and `heldout` splits are outside
this cache; their names should not be confused with later calibration and
holdout roles inside the fit population.

Three identity units remain separate:

- A **chart** is one distinct source `.osu` SHA-256, including its difficulty.
- An **audio hash** is a recorded SHA-256 of exact file bytes, not a distinct
  composition or necessarily a distinct recording. This audit did not rehash
  audio.
- A **group** is a leakage-control cluster. The builder connects set
  directories and IDs, beatmap IDs, normalized artist/title, matching audio
  bytes and older R1 group IDs. The 4,167 training groups and 424 development
  groups therefore must not be read as counts of unique songs.

Every attribution join uses chart SHA-256; timing records that store paths
join those paths to the frozen chart table. At the evidence date, all 22,066
local `.osu` files matched their frozen hashes and the metadata fields checked
by the extraction. There were no unmatched stage records or cache-to-corpus
identity mismatches. The inventory retains frozen labels separately from the
headers read during the audit and their Unicode alternatives.

Metadata ambiguity remains even with exact joins: 15 of the training set's
5,026 audio hashes have multiple `Artist` labels, while 287 of its 4,287 raw
artist/title pairs with an audio hash span multiple hashes. Those observations
cannot distinguish spelling differences, collaborations, edits or incorrect
metadata. The full inventory has 35 missing audio hashes; none occur in the
25 cache-derived stage populations below. Nineteen training charts lack a
positive beatmap ID and six lack a positive set ID, limiting lookup by osu! ID
without removing their hash identity. [Integrity details][integrity] preserve
the conflicting strings and all missing-value counts.

## Stage-by-stage source populations

The numeric columns count distinct source chart hashes, exact nonempty
metadata labels or recorded audio hashes **within the named stage**.
Repeated experiment variants, seeds and sections are deduplicated to charts.
Stages overlap heavily: their totals cannot be added to obtain a count of
unique charts or contributors. An associated audio hash identifies a chart's
audio record; it does not mean that a chart-only stage consumed audio. The
[machine-readable table][stage-counts] additionally records source-record
counts, groups, cache roles and exact extraction rules.

| Stage and data role | Charts | `Creator` labels | `Artist` labels | Audio hashes | Membership evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| Frozen corpus: inventory universe | 22,066 | 1,398 | 3,384 | 8,049 | [Corpus][corpus] |
| R2: eligible weight training | 11,368 | 1,129 | 2,445 | 5,026 | [Cache, `fit_train`][cache] |
| R2: development pool | 1,163 | 318 | 363 | 497 | [Cache, `fit_dev`][cache] |
| Rows: eligible weight training | 11,368 | 1,129 | 2,445 | 5,026 | [Rows index, `fit_train`][rows-index] |
| Rows: development pool | 1,163 | 318 | 363 | 497 | [Rows index, `fit_dev`][rows-index] |
| Prior: fit numerical source rows | 11,368 | 1,129 | 2,445 | 5,026 | [Prior and cache][prior] |
| Prior: method selection | 1,163 | 318 | 363 | 497 | [Prior and cache][prior] |
| Calibration: chart-level population | 11,118 | 1,111 | 2,388 | 4,942 | [Song table][songs] |
| Calibration: 132,053 admitted scopes | 11,117 | 1,111 | 2,388 | 4,941 | [Scope table][scopes] |
| Lens D: measured machine-label references | 817 | 552 | 654 | 811 | [Lens, 2,086 sections][lens] |
| Lens V2: measured human-label validation | 140 | 113 | 97 | 135 | [Lens, 568 sections][lens] |
| Calibration: positive D stream grades | 625 | 462 | 516 | 622 | [Lens, 793 sections][lens] |
| Calibration: positive D jack grades | 510 | 381 | 430 | 506 | [Lens, 618 sections][lens] |
| Calibration: D jumpstream references | 313 | 246 | 260 | 310 | [Lens, 366 sections][lens] |
| Calibration: D chordstream references | 177 | 162 | 160 | 177 | [Lens, 202 sections][lens] |
| Calibration: D chordjack references | 208 | 184 | 192 | 207 | [Lens, 240 sections][lens] |
| Calibration: difficulty response pilot | 16 | 16 | 15 | 16 | [Pilot selection][difficulty] |
| Timing: observed tuning | 2,254 | 458 | 666 | 962 | [Tuning records][timing-train] |
| Timing: observed development | 806 | 215 | 233 | 316 | [Development records][timing-dev] |
| Timing: successful final experiment holdout | 519 | 208 | 237 | 289 | [Holdout records][timing-test] |
| R2: corrected checkpoint guard panel | 16 | 15 | 16 | 16 | [56M records][guards] |
| Rows: final 96-audio evaluation panel | 231 | 83 | 93 | 96 | [Panel][rows-panel] |
| End-to-end system: evaluation panel | 48 | 46 | 47 | 48 | [Panel][system-panel] |
| Early listening: selected sources | 6 | 6 | 6 | 6 | [Selection][early] |
| Final listening: selected sources | 29 | 27 | 29 | 29 | [Selection][final] |
| Release previews: source selections | 4 | 4 | 4 | 4 | [Four play slots][final] |

### What enters the learned models

**Rows (FigureNet).** The released run is `full2-s1/model-5000.pt`, trained
for 5,000 steps. [FigureNet training][rows-fit] constructs a fresh model and
optimizer. The `pilot`, `full` and `full2` runs use the same source population
but are separate initializations, not successive fine-tunes. The earlier rows
network is a development comparison, not an ancestor of the released weights.

[Data preparation][rows-data] supplies standardized 128-bin log-Mel music
features, the mapper's own beat grid, and per-beat figures derived from human
head positions (the times at which taps or long notes begin). The
8,853-figure vocabulary, workload normalization and
star-to-workload mapping are fitted on `fit_train`. Out-of-vocabulary target
beats are omitted from loss. All indexed charts pass the loader's minimum
beat-length filter. Development targets do not train the weights.

The shared listener computes both Mel features and BeatThis probabilities,
but the [released rows decoder][rows-decode] reads only the first 128 Mel
channels. BeatThis probabilities influence the fitted inference grid, not
FigureNet's direct audio inputs. The fixed human training grid means this
stage learns figures on supplied timing; its training loss cannot measure
errors caused by fitting a grid from audio at inference.

**R2 arrangement.** The selected natural-phase run has `init_from=null`;
its [frozen trainer][r2-trainer] constructs a new `R2Model`. It uses supplied
head rows and grids, joint lane actions, taps, long-note heads and releases.
Audio features never enter this model, although audio availability and duration
affect cache eligibility and song-end timing. Supplied heads constrain R2 to
arrangement; arrangement-only evaluation cannot test whether a model found
musical head times from sound.

The released checkpoint is at 56,000,574 decisions from the 64M-decision
run. The [corrected guard selection][guard-selection] selected it; the
original run's selection artifact recorded no selection. The corrected
evaluation covers seven checkpoints from 16M to 64M decisions, all using the
same 16-chart guard set wholly in `fit_dev`. Source defect references come
from `fit_train`. Neither released Ensomi model inherits R1 weights. R1
contributes architecture and chart-replay design, plus corpus grouping and
partition ancestry; these are separate from parameter inheritance. Condition
fine-tunes, preference arms and copy-model training do not belong to this
released weight path.

The indices establish the eligible population, not every optimizer draw.
Rows training samples chart windows with replacement, and R2 uses its corpus
sampler. Exact per-chart draw histories are unavailable in the inspected
evidence, so these counts do not show whether every eligible chart was sampled
or how often it contributed to training.

### Numerical data, calibration and evaluation

The [corpus prior][prior-source] measures four head-row features and five
arrangement targets from each training chart. It selects a prediction method
on `fit_dev`; the released method uses the median of 64 nearby training rows
within a star band. Its membership is reconstructed from cache roles and
the prior's recorded failure list, which is empty. The published numerical
prior retains chart-derived feature/target rows. Removing names does not
remove that dependence on source charts.

Chart-level masks, long-note references, difficulty and quad references use
11,118 training charts with at least 256 head rows, excluding 250 shorter
training charts. Scope calibration admits 132,053 local windows from 11,117
charts: four sampled scopes at each of 8, 16 and 32 seconds, subject to
length and head-count checks. The scope and whole-chart denominators differ
by one chart and must remain separate.

The Lens chart-pattern annotations have separate D and V2 partitions.
Lens D provides recognized **machine labels** and chart measurements for
stream/jack grade and subtype calibration. Jumpstream uses positive stream
labels in headmass tercile 1; chordstream and chordjack use the relevant
positive label in tercile 2. Lens V2 provides recognized **human labels** for
validation and does not fit those thresholds. Its 140 source charts include
130 from R2 `fit_train` and ten from `fit_dev`: it is held apart as a label
validation partition, not an unseen-chart test of R2. Unrecognized cells are
not treated as observed negatives. The difficulty pilot also uses supplied
human head rows, so its response measurements exclude audio-to-head errors.

Timing fitting uses mapper tempo/grid and head placements as references for
audio-derived beat and onset signals. It fits algorithm choices, not an
additional Ensomi neural model. Tuning records contain 975 song indices but
962 audio hashes; development records contain 322 indices but 316 hashes.
These indices are experiment identifiers, not unique-song counts. The final
test selected 521 charts over 290 audio items; one audio loader failure left
519 evaluated charts and 289 hashes. **All 519 belong to R2 `fit_train`.**
They were held out from that timing experiment, not from the global R2
training corpus.

End-to-end panels use music to drive timing and rows, with source charts as
references. The system panel contains 48 source charts; its control checks
use a 24-song subset at two seeds. Listening selections mix generated pairs,
human-reference pairs and full-song play slots. Membership proves selection
or measurement as recorded, not that every selected item was listened to.
Quality conclusions remain in the
[system report](r2_system.md#what-has-been-checked).

### Additional calibration and selection populations

Twelve additional roles cover the restoring-force experiment and its
immediate follow-up, later decoder defaults, and run manifests. The
restoring force steers generated arrangement statistics toward target values;
its gain controls the strength of that correction. These roles cover 4,532
distinct charts, all within the eligible training/development union. They add
data-use roles, not new corpus charts. The
[supplementary count table][supplement-counts] records exact evidence paths
and overlaps; [source rows][supplement-sources],
[mapper labels][supplement-mappers] and [artist labels][supplement-artists]
make every attribution searchable.

| Role | Charts | `Creator` labels | `Artist` labels | Audio hashes |
| --- | ---: | ---: | ---: | ---: |
| Restoring-force gain comparison panel | 48 | 44 | 46 | 48 |
| Original nearest-32 mean-target donors | 1,341 | 476 | 690 | 1,021 |
| Original human-pair null references | 3,157 | 530 | 947 | 1,344 |
| Actually drawn pair-difference target sources | 186 | 79 | 90 | 93 |
| Follow-up X0 diagnostic sources | 8 | 8 | 8 | 8 |
| Follow-up nearest-32 candidate donors | 1,505 | 510 | 744 | 1,127 |
| Follow-up actually drawn joint-target donors | 102 | 89 | 91 | 101 |
| Follow-up admitted human-drift null references | 3,010 | 522 | 911 | 1,281 |
| Quiet-rest default selection | 96 | 83 | 93 | 96 |
| Default difficulty quantile selection | 24 | 24 | 23 | 24 |
| Natural-NLL validation manifest | 64 | 60 | 63 | 64 |
| Run free-generation manifest | 4 | 4 | 4 | 4 |

The gain comparison uses 48 `fit_dev` charts with source heads, grid and song
end fixed; oracle variants also receive the source's whole-song arrangement
levels. Donor charts supply target statistics, while human-pair null
references measure variation between source charts. No audio features enter
these comparisons. All sixteen difficulty-pilot charts are in this panel.
Original mean donors and follow-up candidate donors are `fit_train`;
human-null references mix training and development charts.
One chart in each candidate-donor set falls outside the 11,118-chart
chart-level calibration subset. Actual target donors, candidate pools and
null references have distinct roles and cannot be counted interchangeably.

This experiment informs the decode-time restoring-force gain described by the
[system report](r2_system.md#r2). Its donor-target recipes do not define the
released prior, and its source-grid results do not evaluate audio timing.
The [calibration lineage appendix][supplement] records the gain's released
source location and its immediate experimental lineage.

Quiet-rest selection uses one source chart for each of the 96 audio items
in the broader 231-chart rows panel. It compares source rests and rhythm with
audio-driven rows at two seeds. The default difficulty-quantile population is
reconstructed from the 24 control-enabled entries in the 48-chart system
panel and the controls table; no separate per-quantile source roster was
recovered. Those quantile measurements used the earlier quiet-rest setting,
while final end-to-end runs used the adopted setting; the two calibration
contexts are distinct. These choices change decoding, not the neural weights.

The natural-phase negative-log-likelihood (NLL) validation manifest contains
64 windows from 64 `fit_dev` charts; the separate run free-generation manifest
contains four `fit_dev` charts. These manifest populations differ from the
corrected 16-chart guard panel. As elsewhere, a manifest records membership,
not proof of every downstream use.

## Public preview source identities

The four released previews use the audio associated with these source chart
selections, all in `fit_dev`. A named source mapper is not thereby the author
of the generated lane arrangement. Hashes and IDs remain searchable in the
[chart inventory][charts].

| Preview | Source `Artist` | Source `Title` | Source `Creator` | Source difficulty | Beatmap ID |
| --- | --- | --- | --- | --- | ---: |
| SaiGetsu | Otokaze | SaiGetsu (Cut Ver.) | Irone OSU | 4K Vibrant Seasons | 2894207 |
| Catch the Moment | LiSA | Catch the Moment | Maxus | Live the Moment | 1219748 |
| Hesitation Snow | fripSide | Hesitation Snow -version 2024- (Cut Ver.) | FerdiXIA- | Hard | 5483425 |
| Adventure | YOASOBI | Adventure | AelSan | Memorable Memories | 4661716 |

## External BeatThis training

Ensomi uses pretrained BeatThis `final0` for beat/downbeat inference and
does not fine-tune it. The
[upstream model description](https://github.com/CPJKU/beat_this/tree/v1.1.0#available-models)
identifies `final0` as the seed-0 main model trained with the usual training
and validation partitions, excluding GTZAN. Processing a user's song at
inference does not establish that the song trained BeatThis.

The [paper, section 4.1](https://arxiv.org/html/2407.21658v1#S4.SS1) reports
4,556 selected tracks. The pinned
[annotation release v1.0](https://github.com/CPJKU/beat_this_annotations/tree/c3c47fd37d3074d9f8119f18bbf460f909609f22)
instead contains 4,557 nominal candidate IDs across 15 split files: ASAP,
Ballroom, Beatles, Candombe, Filosax, Groove MIDI, GuitarSet, Hainsworth,
Harmonix, HJDB, JAAH, RWC's four subsets combined, Simac, SMC and TapCorrect.
The one-entry difference remains unresolved. Loader exclusions are a
possible explanation, not an established cause. Neither count is an
independently verified ledger of examples consumed by the original run.

The inputs are audio-derived spectrograms with beat/downbeat annotations.
No osu! mapping corpus is identified in the published source list.
Harmonix has rhythm-game authoring origins, but that does not establish use
of released osu! or Rock Band gameplay charts. Metadata matches identify
Ciara's “1, 2 Step,” Lil Wayne's “6 Foot 7 Foot” and The Jackson 5's “ABC”
as nominal Harmonix candidates; they do not verify consumption by `final0`.
The [upstream provenance appendix][beatthis] records sources and collection
counts, and the [candidate TSV][beatthis-candidates] preserves every split ID.

The external musician roster remains incomplete. Completing it requires
dataset-specific metadata joins, reconciliation of loader exclusions and
separate attribution for recording artists, performers, composers and
annotators. An upstream track ID or filename cannot establish all those roles.

## Evidence scope and remaining gaps

The inventory resolves all chart identities in the 26 core and twelve
supplemental populations against frozen metadata. It checks chart bytes and
headers as they stood on 2026-10-17, using recorded source and configuration
evidence without loading model checkpoints, running inference or opening
audio. Source-file hashes in the local [extraction summary][summary] identify
the inputs to that snapshot. Audio hashes are recorded corpus values, not
independent audio-byte verification by this audit.

This covers the released weight inputs and the enumerated prior, calibration,
timing and evaluation populations. It is not a complete ledger of every
historical experiment or design influence. Earlier timing exploration and
every historical listening screen are not all reduced to one source manifest
here. Their influence does not establish additional inherited weights. The
exact optimizer draw history, resolved mapper and musician identities,
complete BeatThis musician roster, and source permissions also remain outside
the evidence.

Supplied timing and head rows limit what the models can learn and what their
evaluations can measure, as described for each stage. Stage membership
establishes eligibility or observed use as specified; it does not establish
chart quality, permission, or an individual source's causal contribution to
model parameters. Independent per-source inspection and reproduction require
the local appendices and source files, which are not publicly packaged here.

[appendices]: ../../artifacts/reports/r2-provenance-20261017/README.md
[mappers]: ../../artifacts/reports/r2-provenance-20261017/bulk/mapper_all_stages.csv
[artists]: ../../artifacts/reports/r2-provenance-20261017/bulk/artist_all_stages.csv
[charts]: ../../artifacts/reports/r2-provenance-20261017/bulk/charts.csv
[integrity]: ../../artifacts/reports/r2-provenance-20261017/metadata_integrity.md
[stage-counts]: ../../artifacts/reports/r2-provenance-20261017/stage_counts.csv
[summary]: ../../artifacts/reports/r2-provenance-20261017/attribution_summary.json
[beatthis]: ../../artifacts/reports/r2-provenance-20261017/beatthis-provenance.md
[beatthis-candidates]: ../../artifacts/reports/r2-provenance-20261017/bulk/beatthis-candidate-tracks.tsv
[supplement]: ../../artifacts/reports/r2-provenance-20261017/calibration-lineage.md
[supplement-counts]: ../../artifacts/reports/r2-provenance-20261017/calibration_lineage_counts.csv
[supplement-sources]: ../../artifacts/reports/r2-provenance-20261017/bulk/calibration_lineage_sources.csv
[supplement-mappers]: ../../artifacts/reports/r2-provenance-20261017/bulk/calibration_lineage_mappers.csv
[supplement-artists]: ../../artifacts/reports/r2-provenance-20261017/bulk/calibration_lineage_artists.csv
[corpus-source]: ../../src/ensomi_model/evaluation/corpus.py
[split-source]: ../../src/ensomi_model/r2/splits.py
[rows-fit]: ../../src/ensomi_model/audio_rows/fit.py
[rows-data]: ../../src/ensomi_model/audio_rows/beatdata.py
[rows-decode]: ../../src/ensomi_model/audio_rows/rowsgen.py
[prior-source]: ../../src/ensomi_model/system/prior.py
[r2-trainer]: ../../artifacts/r2-runs/r2-phaseN-20261006/code/ensomi_model/r2/train_ce.py
[corpus]: ../../data/r2-corpus.parquet
[cache]: ../../artifacts/r2-cache/v1/index.parquet
[rows-index]: ../../artifacts/rows-e2e-20261010/p1/bulk/data/index.json
[prior]: ../../artifacts/rows-e2e-20261010/p2/prior/prior.json
[songs]: ../../artifacts/r2-demo2-20261009/bulk/songs.parquet
[scopes]: ../../artifacts/r2-demo2-20261009/bulk/step2a/scopes.parquet
[lens]: ../../artifacts/r2-demo2-20261009/bulk/step2a/lens_sections.parquet
[difficulty]: ../../artifacts/r2-demo2-20261009/pilot_selection.json
[timing-train]: ../../artifacts/timing-20261009/opt/x8/hyb_fit_train.jsonl
[timing-dev]: ../../artifacts/timing-20261009/opt/x10/fit_dev_charts.jsonl
[timing-test]: ../../artifacts/timing-20261009/opt/x14/holdout_charts.jsonl
[guards]: ../../artifacts/r2-guard-v2-20261007/56M/records.jsonl
[guard-selection]: ../../artifacts/r2-guard-v2-20261007/selection.json
[rows-panel]: ../../artifacts/rows-e2e-20261010/p1/panel96/panel.json
[system-panel]: ../../artifacts/rows-e2e-20261010/p2/panel/panel.json
[early]: ../../artifacts/rows-e2e-20261010/early/selection.json
[final]: ../../artifacts/rows-e2e-20261010/p3/selection.json
