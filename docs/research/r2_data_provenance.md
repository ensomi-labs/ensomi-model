# R2 data provenance: charts, mappers and music

Evidence snapshot: **2026-10-17**. This report traces source attribution and
data-use roles for **Ensomi R2 1.0 and 1.0.1**: timing, rows, arrangement,
corpus prior, calibration, selection, evaluation and previews. It does not
audit R1 or any other historical Ensomi model. It establishes attribution and
stage membership. It is not a rights clearance or a complete review of how the
source charts and music were acquired, and it does not establish permission to
use them. The
[R2 system report](r2_system.md#code-and-training-lineage) identifies the
releases and their training lineage; its
[file inventory](r2_system.md#files-and-licences) lists the published files and
licence statements. The [published tables](r2_data_provenance/README.md) give
the counts behind this report for every stage.

The released rows and arrangement models were trained from scratch, with the
same **11,368 eligible training charts**, carrying **1,129 distinct `Creator`
labels, 2,445 distinct `Artist` labels and 5,026 recorded exact audio-byte
hashes**. These charts fall into **4,167 leakage-control groups**, groups used
to keep related sources within the same data partition. Neither the groups nor
the audio hashes are a count of songs. The rows model learns from music
features and mapper-supplied timing and head positions. R2 learns arrangements
from chart rows and actions, without audio features. The same corpus also
supplies numerical priors, decoder calibration, timing references and
evaluation material. The published prior retains one row of numerical features
and targets for each training chart, and the published calibration and scope
table are also derived from the charts. BeatThis `final0` contributes separate
externally pretrained weights, with their own upstream provenance and an
incomplete record of the musicians behind them.

The complete frozen inventory contains 22,066 charts, 1,398 `Creator` labels
and 3,384 `Artist` labels. That inventory is not the weight-training corpus.
Counts below identify eligible or observed source membership; they do not
measure each source's sampled training exposure or contribution to an output.
Metadata labels identify neither verified people nor verified rights holders.

## Permissions and release policy — 2026-10-21

The existing released models, `r2-1.0` and `r2-1.0.1`, remain available with
this disclosure of their data provenance. Future released models will use data
covered by permissive licences or by explicit permission suitable for the
intended training and release uses. The data review for such a release
considers, as applicable, inherited weights, upstream components such as
pretrained beat trackers, distillation from earlier models, and derived data
such as priors and calibration tables, not only the training charts. This
policy is separate from the 2026-10-17 evidence snapshot; the audit does not
establish that the existing corpus meets that requirement.

Ensomi is exploring contributor enrollment (opt-in) and opt-out requests,
including a convenient way to verify that a decision comes from the relevant
chart or music rights holder. The process and its effect on existing models
have not been established. No retroactive removal of a source's learned
influence is claimed.

## Corrections and requests

To correct an attribution, to ask how a chart or song was used, or to say that
you are interested in the enrollment and opt-out process being explored, open a
[data provenance issue](https://github.com/ensomi-labs/ensomi-model/issues/new?template=data-provenance.yml).
Issues are public: include no personal data and no audio. Opening an issue
does not by itself remove anything from a released model.

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

[`stage_counts.csv`](r2_data_provenance/stage_counts.csv) gives the chart,
group, audio-hash and label counts of all 38 core and supplemental roles. Two
label tables, one for `Creator` and one for `Artist`, list every exact label in
each role with that role's chart denominator and the label's chart, audio-hash
and group counts; the
[tables' README](r2_data_provenance/README.md#label-tables) records them as
pending. The per-chart records behind these counts, which place each chart in
its stages, remain private ([What remains private](#what-remains-private)).

`Creator` and `Artist` are source-header claims, not verified identities.
No aliases were merged or collaboration strings split. `Creator` may omit a
guest difficulty's author; `Artist` does not enumerate composers, performers,
remixers, sampled musicians or rights holders. “Various Artists” remains one
literal label. The columns therefore count labels, not people, and imply no
ownership or permission finding.

## Source collection, admission and identity

The audit starts from a local collection of osu!mania charts and records of
their associated audio. The [corpus builder][corpus-source] reads charts from
two roots of that collection, associates the acquisition records available for
them, and freezes the inventory in [`data/r2-corpus.parquet`][corpus]. This
establishes the local inputs and their recorded origins; it does not
reconstruct a complete acquisition history or verify permission from chart and
music rights holders.

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
- A **group** is a leakage-control cluster, which keeps related sources within
  the same data partition. The builder connects set directories and IDs,
  beatmap IDs, normalized artist/title, matching audio bytes and older R1 group
  IDs. The 4,167 training groups and 424 development groups therefore must not
  be read as counts of unique songs.

Every attribution join uses chart SHA-256; timing records that store paths
join those paths to the frozen chart table. At the evidence date, all 22,066
local `.osu` files matched their frozen hashes and the metadata fields checked
by the extraction (`Creator`, `Artist`, `Title`, `Version`, `BeatmapID` and
`BeatmapSetID`), with no UTF-8 decoding replacements. There were no unmatched
stage records or cache-to-corpus identity mismatches. The private inventory
retains frozen labels separately from the headers read during the audit and
their Unicode alternatives.

Metadata ambiguity remains even with exact joins:

| Population | Audio hashes with more than one `Artist` label | Raw artist/title pairs with an audio hash that span more than one hash |
| --- | ---: | ---: |
| Frozen corpus | 37 / 8,049 | 483 / 6,409 |
| `fit_train` | 15 / 5,026 | 287 / 4,287 |
| `fit_dev` | 1 / 497 | 29 / 433 |

Those observations cannot distinguish spelling differences, aliases,
collaborations, edits or incorrect metadata. The full inventory has 35 missing
audio hashes; none occur in the 25 cache-derived stage populations below. Every
frozen chart has a nonempty `Creator`, `Artist` and `Title`; one frozen
`Version` is empty. Nineteen training charts lack a positive beatmap ID and six
lack a positive set ID, limiting lookup by osu! ID without removing their hash
identity. [`stage_counts.csv`](r2_data_provenance/stage_counts.csv) gives the
missing-value counts of every core stage; the conflicting strings themselves
are per-chart records and remain private.

## Stage-by-stage source populations

The numeric columns count distinct source chart hashes, exact nonempty
metadata labels or recorded audio hashes **within the named stage**.
Repeated experiment variants, seeds and sections are deduplicated to charts.
Stages overlap heavily: their totals cannot be added to obtain a count of
unique charts or contributors. An associated audio hash identifies a chart's
audio record; it does not mean that a chart-only stage consumed audio.
[`stage_counts.csv`](r2_data_provenance/stage_counts.csv) adds, under each
stage id, its groups, its source-record count and unit, beatmap set IDs,
missing values, and its charts in `fit_train` and `fit_dev`.

| Stage and data role | Charts | `Creator` labels | `Artist` labels | Audio hashes | Stage id |
| --- | ---: | ---: | ---: | ---: | --- |
| Frozen corpus: inventory universe | 22,066 | 1,398 | 3,384 | 8,049 | `frozen_corpus_inventory` |
| R2: eligible weight training | 11,368 | 1,129 | 2,445 | 5,026 | `r2_cache_fit_train` |
| R2: development pool | 1,163 | 318 | 363 | 497 | `r2_cache_fit_dev` |
| Rows: eligible weight training | 11,368 | 1,129 | 2,445 | 5,026 | `rows_fit_train` |
| Rows: development pool | 1,163 | 318 | 363 | 497 | `rows_fit_dev` |
| Prior: fit numerical source rows | 11,368 | 1,129 | 2,445 | 5,026 | `prior_fit_train` |
| Prior: method selection | 1,163 | 318 | 363 | 497 | `prior_fit_dev` |
| Calibration: chart-level population | 11,118 | 1,111 | 2,388 | 4,942 | `calibration_song_population` |
| Calibration: 132,053 admitted scopes | 11,117 | 1,111 | 2,388 | 4,941 | `calibration_scope_population` |
| Lens D: measured machine-label references, 2,086 sections | 817 | 552 | 654 | 811 | `lens_D_measured` |
| Lens V2: measured human-label validation, 568 sections | 140 | 113 | 97 | 135 | `lens_V2_measured` |
| Calibration: positive D stream grades, 793 sections | 625 | 462 | 516 | 622 | `calibration_lens_stream_grades` |
| Calibration: positive D jack grades, 618 sections | 510 | 381 | 430 | 506 | `calibration_lens_jack_grades` |
| Calibration: D jumpstream references, 366 sections | 313 | 246 | 260 | 310 | `calibration_lens_jumpstream` |
| Calibration: D chordstream references, 202 sections | 177 | 162 | 160 | 177 | `calibration_lens_chordstream` |
| Calibration: D chordjack references, 240 sections | 208 | 184 | 192 | 207 | `calibration_lens_chordjack` |
| Calibration: difficulty response pilot | 16 | 16 | 15 | 16 | `calibration_difficulty_pilot` |
| Timing: observed tuning | 2,254 | 458 | 666 | 962 | `timing_tuning_observed` |
| Timing: observed development | 806 | 215 | 233 | 316 | `timing_dev_observed` |
| Timing: successful final experiment holdout | 519 | 208 | 237 | 289 | `timing_holdout_successful` |
| R2: corrected checkpoint guard panel, records of the 56M checkpoint | 16 | 15 | 16 | 16 | `r2_selected56M_guard_evaluation` |
| Rows: final 96-audio evaluation panel | 231 | 83 | 93 | 96 | `rows_evaluation_panel96` |
| End-to-end system: evaluation panel | 48 | 46 | 47 | 48 | `system_evaluation_panel` |
| Early listening: selected sources | 6 | 6 | 6 | 6 | `early_listening_selection` |
| Final listening: selected sources | 29 | 27 | 29 | 29 | `final_listening_selection` |
| Release previews: source selections, four play slots | 4 | 4 | 4 | 4 | `release_preview_source_selection` |

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

**R2 arrangement.** The selected natural-phase run has `init_from=null`; its
trainer constructs a new `R2Model`. The run's frozen copy of that trainer is
byte-identical to
[`src/ensomi_model/r2/train_ce.py` at commit `3954031`][r2-trainer], the commit
R2 was trained from. It uses supplied head rows and grids, joint lane actions,
taps, long-note heads and releases. Audio features never enter this model,
although audio availability and duration affect cache eligibility and song-end
timing. Supplied heads constrain R2 to arrangement; arrangement-only evaluation
cannot test whether a model found musical head times from sound.

The released checkpoint is at 56,000,574 decisions from the 64M-decision
run. The corrected guard selection selected it; the original run's selection
record recorded no selection. The corrected evaluation covers seven checkpoints
from 16M to 64M decisions, all using the same 16-chart guard set wholly in
`fit_dev`. Source defect references come from `fit_train`. Neither released
Ensomi model inherits R1 weights. R1 contributes architecture and chart-replay
design, plus corpus grouping and partition ancestry; these are separate from
parameter inheritance. Condition fine-tunes, preference arms and copy-model
training do not belong to this released weight path.

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
the prior's recorded failure list, which is empty. The
[published prior][prior] keeps one row of the four standardized features and
five targets for each of the 11,368 training charts, grouped by star band.
Removing names does not remove that dependence on source charts.

Chart-level masks, long-note references, difficulty and quad references use
11,118 training charts with at least 256 head rows, excluding 250 shorter
training charts. Scope calibration admits 132,053 local windows from 11,117
charts: four sampled scopes at each of 8, 16 and 32 seconds, subject to
length and head-count checks. The scope and whole-chart denominators differ
by one chart and must remain separate. The published calibration and scope
table hold numbers derived from these charts; the scope table keeps one row for
each of the 132,053 scopes, without chart hashes or paths.

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
use a 24-chart subset at two seeds. Listening selections mix generated pairs,
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
data-use roles, not new corpus charts.
[`stage_counts.csv`](r2_data_provenance/stage_counts.csv) gives their counts,
their source-record units and their charts in `fit_train` and `fit_dev`; the
pending label tables cover their labels too. Their per-chart source rows
remain private.

| Role | Charts | `Creator` labels | `Artist` labels | Audio hashes | Stage id |
| --- | ---: | ---: | ---: | ---: | --- |
| Restoring-force gain comparison panel | 48 | 44 | 46 | 48 | `kappa_comparison_panel` |
| Original nearest-32 mean-target donors | 1,341 | 476 | 690 | 1,021 | `kappa_original_mean_donors` |
| Original human-pair null references | 3,157 | 530 | 947 | 1,344 | `kappa_original_human_null` |
| Actually drawn pair-difference target sources | 186 | 79 | 90 | 93 | `kappa_original_pair_draws` |
| Follow-up diagnostic sources at human-marked windows | 8 | 8 | 8 | 8 | `kappa_followup_x0_sources` |
| Follow-up nearest-32 candidate donors | 1,505 | 510 | 744 | 1,127 | `kappa_followup_lookup_candidates` |
| Follow-up actually drawn joint-target donors | 102 | 89 | 91 | 101 | `kappa_followup_drawn_donors` |
| Follow-up admitted human-drift null references | 3,010 | 522 | 911 | 1,281 | `kappa_followup_human_drift_null` |
| Quiet-rest default selection | 96 | 83 | 93 | 96 | `rows_quiet_rests_calibration` |
| Default difficulty quantile selection | 24 | 24 | 23 | 24 | `system_default_q_calibration` |
| Natural-NLL validation manifest | 64 | 60 | 63 | 64 | `natural_nll_validation` |
| Run free-generation manifest | 4 | 4 | 4 | 4 | `run_freerun_manifest` |

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
released prior, and its source-grid results do not evaluate audio timing. The
released Python system fixes the gains for held-lane share, heads per row and
all-busy-row share to `(3, 3, 3)`
([`system/make.py` at `r2-1.0.1`](https://github.com/ensomi-labs/ensomi-model/blob/r2-1.0.1/src/ensomi_model/system/make.py#L54);
the same tuple at `r2-1.0`), and the arranger disables the heads-per-row gain
inside stream and subtype scopes. The experiment and its follow-up both applied
the raw coefficient three, without scaling by a standard deviation. The
coefficient was not estimated as a continuous optimum.

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
of the generated lane arrangement. The beatmap IDs identify the source
beatmaps on osu!; their chart and audio hashes are kept in the private
inventory.

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
instead contains 4,557 nominal candidate IDs across 15 `single.split` files,
one per collection:

| Collection | Nominal IDs |
| --- | ---: |
| ASAP | 473 |
| Ballroom | 685 |
| Beatles | 180 |
| Candombe | 35 |
| Filosax | 48 |
| Groove MIDI | 336 |
| GuitarSet | 180 |
| Hainsworth | 222 |
| Harmonix | 911 |
| HJDB | 235 |
| JAAH | 113 |
| RWC, four subsets combined | 226 |
| Simac | 595 |
| SMC | 217 |
| TapCorrect | 101 |
| **Total** | **4,557** |

These count upstream split entries, not musicians, unique compositions,
training excerpts or examples verified as consumed by `final0`; for `final0`,
entries of both the train and the validation split are training candidates.
The one-entry difference remains unresolved. The
[upstream loader](https://github.com/CPJKU/beat_this/blob/v1.1.0/beat_this/dataset/dataset.py)
can omit candidates with missing augmented spectrograms or unsuitable
annotation shape; this is a possible explanation, not an established cause.
Neither count is an independently verified ledger of examples consumed by the
original run.

The inputs are audio-derived spectrograms with beat/downbeat annotations.
No osu! mapping corpus is identified in the published source list.
Harmonix has rhythm-game authoring origins, but that does not establish use
of released osu! or Rock Band gameplay charts. Joining the Harmonix
[metadata](https://github.com/urinieto/harmonixset/blob/64abeb509429e73d74559fb98e621dac866efea1/dataset/metadata.csv)
to its
[BeatThis split](https://github.com/CPJKU/beat_this_annotations/blob/c3c47fd37d3074d9f8119f18bbf460f909609f22/harmonix/single.split)
identifies Ciara's “1, 2 Step,” Lil Wayne's “6 Foot 7 Foot” and The Jackson 5's
“ABC” as nominal Harmonix candidates; these matches do not verify consumption
by `final0`.

The external musician roster remains incomplete. Completing it requires
dataset-specific metadata joins, reconciliation of loader exclusions and
separate attribution for recording artists, performers, composers and
annotators. An upstream track ID or filename cannot establish all those roles.

## Evidence scope and remaining gaps

The inventory resolves all chart identities in the 26 core and twelve
supplemental populations against frozen metadata, the tracked
[`data/r2-corpus.parquet`][corpus] (SHA-256
`49ab9e1bbbdcf7a08bf86ee02181515dfa04bdc9efa3be25fe6bd4b0e6d9ee4b`, the hash the
R2 cache records). It checks chart bytes and headers as they stood on
2026-10-17, using recorded source and configuration evidence without loading
model checkpoints, running inference or opening audio. The extraction recorded
the SHA-256 of each input file;
[`export_record.json`](r2_data_provenance/export_record.json) gives those of
the count tables the published files were built from. Audio hashes are
recorded corpus values, not independent audio-byte verification by this audit.

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
the private records below.

### What remains private

Per-item records derived from the private dataset are not published: the chart
inventory that joins each chart's identity to its stage memberships and to the
headers read during the audit, the per-stage lists of chart hashes, the R2
training cache and the rows and prior indices, the timing, guard and Lens
records, the panel and listening selections, the supplemental roles' per-chart
source rows, and the conflicting strings behind the integrity counts. The
published tables are aggregates of these records, and repeating the extraction
needs them and the local chart collection. The frozen corpus table that the
inventory joins against is tracked in this repository. The BeatThis candidate
table is not republished: it concatenates the 15 public split files at the
pinned annotation revision, keeping each entry's dataset, track ID and
original split.

[corpus-source]: ../../src/ensomi_model/evaluation/corpus.py
[split-source]: ../../src/ensomi_model/r2/splits.py
[rows-fit]: ../../src/ensomi_model/audio_rows/fit.py
[rows-data]: ../../src/ensomi_model/audio_rows/beatdata.py
[rows-decode]: ../../src/ensomi_model/audio_rows/rowsgen.py
[prior-source]: ../../src/ensomi_model/system/prior.py
[r2-trainer]: https://github.com/ensomi-labs/ensomi-model/blob/39540317be42aa53104939c759e03610d64e6a3e/src/ensomi_model/r2/train_ce.py
[corpus]: ../../data/r2-corpus.parquet
[prior]: https://huggingface.co/sed-i/ensomi-r2/blob/r2-1.0.1/prior.json
