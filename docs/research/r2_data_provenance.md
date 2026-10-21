# R2 data provenance: charts, mappers and music

Evidence snapshot: **2026-10-17**. This report traces which source charts,
metadata labels and audio each stage of **Ensomi R2 1.0 and 1.0.1** used:
timing, rows, arrangement, corpus prior, calibration, selection, evaluation and
previews. It does not audit R1 or any earlier Ensomi model. It establishes
attribution and stage membership, not rights clearance, permission to use the
sources or a complete review of how they were acquired. The
[R2 system report](r2_system.md#code-and-training-lineage) identifies the
releases, their training lineage and their
[published files](r2_system.md#files-and-licences); the
[tables](r2_data_provenance/README.md) give every stage's counts.

The released rows and arrangement models were trained from scratch on the same
**11,368 eligible training charts**, with **1,129 distinct `Creator` labels,
2,445 distinct `Artist` labels and 5,026 recorded audio-file hashes**, in
**4,167 groups** used to keep related sources within the same data partition;
neither groups nor audio hashes count songs. The same charts supply the corpus
prior, whose published file keeps one row of numbers for each training chart,
the published calibration and scope table, timing references and evaluation
material. BeatThis `final0` adds externally pretrained weights whose upstream
record of musicians is incomplete.

The frozen inventory of 22,066 charts, with 1,398 `Creator` and 3,384 `Artist`
labels, is not the training corpus. Counts give eligible or observed
membership, not a source's sampled exposure or contribution to an output, and
labels identify neither verified people nor verified rights holders.

## Permissions and release policy — 2026-10-21

The existing released models, `r2-1.0` and `r2-1.0.1`, remain available with
this disclosure of their data provenance. Future released models will use data
covered by permissive licences or by explicit permission suitable for the
intended training and release uses. The data review for such a release
considers, as applicable, inherited weights, upstream components such as
pretrained beat trackers, distillation from earlier models, and derived data
such as priors and calibration tables, not only the training charts. This
policy is separate from the 2026-10-17 evidence snapshot, which does not
establish that the existing corpus meets it.

Ensomi is exploring contributor enrollment (opt-in) and opt-out requests,
including a convenient way to verify that a decision comes from the relevant
chart or music rights holder. Neither the process nor its effect on existing
models has been established, and no removal of a source's learned influence is
claimed.

## Corrections and requests

To correct an attribution, ask how a chart or song was used, or say that you
are interested in enrollment or opt-out, open a
[data provenance issue](https://github.com/ensomi-labs/ensomi-model/issues/new?template=data-provenance.yml).
Issues are public: include no personal data and no audio. Opening an issue
does not by itself remove anything from a released model.

## Attribution labels

These are the ten most frequent exact labels among the eligible training
charts; every chart count is out of **11,368**. Ordering by charts favours
music with several mapped difficulties; the audio-hash column gives each artist
label's distinct recorded audio hashes.

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

Two label tables of every exact label in each stage, with its chart,
audio-hash and group counts and the stage's chart denominator, are
[pending](r2_data_provenance/README.md#label-tables); the per-chart records
behind them [remain private](#what-remains-private). `Creator` and `Artist`
are source-header claims: no aliases were merged or collaboration strings
split, `Creator` may omit a guest difficulty's author, `Artist` does not
enumerate composers, performers, remixers, sampled musicians or rights
holders, and “Various Artists” is one label. They count labels, not people.

## Mapper and artist style

R2 has no purpose of reproducing any particular mapper's or artist's style. In
the open code at release `r2-1.0.1` and at the two models' training commits,
neither model receives a mapper or artist name, ID or embedding: the rows model
reads log-Mel features, beat positions, bar phase, song progress, BPM, the
requested workload and section requests, and its own earlier figures
([inputs][style-rows-inputs]); R2 reads head times, the beat grid, the song
length and its own earlier decisions ([inputs][style-r2-inputs]). Both were
trained from scratch, not fine-tuned ([R2's trainer][style-r2-init],
[rows][rows-fit]), the rows trainer drawing training charts uniformly and R2 a
group uniformly, then a chart within it ([draw][style-r2-draw]), with no
weight, cap, stratum or filter keyed on a mapper or artist; artist and title
are read only to group charts of the same song, which keeps related sources in
one data partition and forms R2's draw unit, and `Creator` is not read for
grouping ([grouping][style-groups]). The released calibration is keyed by star
band and pattern subtype and the scope table by band, length, median gap and
chord, hand and jump shares, with no names, IDs or paths, though the scripts
that built them are not in this repository; the corpus prior takes the median
target of the 64 training charts nearest in four numeric features within a
star band ([prior][style-prior]), with no cap on one mapper's share, which was
not measured. No code evaluates, reports or stratifies results by mapper or
artist, but some panels, the listening selections and the four preview songs
were chosen outside the open code. Code shows the absence of a mechanism, not
intent, and a mapper with many eligible charts weighs more: by those charts in
the rows model's draws, and by their share of each group in R2's.

## Source collection, admission and identity

The audit starts from a local collection of osu!mania charts and records of
their audio. The [corpus builder][corpus-source] reads two roots of it, with
the acquisition records available for them, and freezes the inventory in
[`data/r2-corpus.parquet`][corpus]; this records origins, not a complete
acquisition history.

The [R2 admission rule][split-source] requires `eval_split == 'fit'`, ranked
or loved API status (a selection criterion, not evidence of permission) and
computed star rating **2 <= star <= 6**, a boundary that differs from the
released [request interval](r2_system.md#what-you-can-ask-for). Of 12,551
candidates, 12,531 were retained: 15 lacked audio, four failed chart-legality
checks (`illegal_source`) and one had a collision when nearby head rows were
merged, all technical exclusions. The cache has 11,368 `fit_train` and 1,163
`fit_dev` charts with disjoint groups; the corpus's own `calibration` and
`heldout` splits lie outside it and are not the later calibration and holdout
roles.

Three identity units stay separate:

- A **chart** is one distinct source `.osu` SHA-256, including its difficulty.
- An **audio hash** is a recorded SHA-256 of exact file bytes, not a
  composition or necessarily a distinct recording; this audit did not rehash
  audio.
- A **group** keeps related sources within the same data partition. The
  builder connects set directories and IDs, beatmap IDs, normalized
  artist/title, matching audio bytes and older R1 group IDs, so the 4,167
  training and 424 development groups are not counts of songs.

Every attribution join uses chart SHA-256; timing records that store paths
join through the frozen chart table. On the evidence date all 22,066 local
`.osu` files matched their frozen hashes and checked headers (`Creator`,
`Artist`, `Title`, `Version`, `BeatmapID`, `BeatmapSetID`), with no UTF-8
replacements, unmatched stage records or cache-to-corpus mismatches.

Metadata ambiguity remains even with exact joins:

| Population | Audio hashes with more than one `Artist` label | Raw artist/title pairs with an audio hash that span more than one hash |
| --- | ---: | ---: |
| Frozen corpus | 37 / 8,049 | 483 / 6,409 |
| `fit_train` | 15 / 5,026 | 287 / 4,287 |
| `fit_dev` | 1 / 497 | 29 / 433 |

These cannot tell spelling differences, aliases, collaborations, edits and
incorrect metadata apart. The inventory has 35 missing audio hashes, none in
the 25 cache-derived stages below. Every frozen chart has a nonempty
`Creator`, `Artist` and `Title`; one `Version` is empty. Nineteen training
charts lack a positive beatmap ID and six a positive set ID, which limits
lookup by osu! ID but not their hash identity.

## Stage-by-stage source populations

Counts are distinct source chart hashes, exact nonempty labels or recorded
audio hashes **within the named stage**, with variants, seeds and sections
deduplicated to charts. Stages overlap heavily, so their totals cannot be
added, and an audio hash identifies a chart's audio record, not that a
chart-only stage consumed audio.
[`stage_counts.csv`](r2_data_provenance/stage_counts.csv) adds each stage's
groups, source records and their unit, set IDs, missing values, and charts in
`fit_train` and `fit_dev`.

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

**Rows (FigureNet).** The released run, `full2-s1/model-5000.pt`, trained for
5,000 steps from a fresh model and optimizer ([training][rows-fit]); the
`pilot`, `full` and `full2` runs are separate initializations, not successive
fine-tunes, and the earlier rows network is a comparison, not an ancestor.
[Data preparation][rows-data] supplies standardized 128-bin log-Mel features,
the mapper's beat grid and per-beat figures derived from human head positions
(where taps or long notes begin). The 8,853-figure vocabulary, workload
normalization and star-to-workload mapping are fitted on `fit_train`;
out-of-vocabulary target beats are omitted from loss, every indexed chart
passes the minimum beat-length filter, and development targets train no
weights. The [released decoder][rows-decode] reads only the 128 Mel channels,
so BeatThis shapes the fitted inference grid, not FigureNet's input; trained
on human grids, the loss cannot measure errors of fitting a grid to audio at
inference.

**R2 arrangement.** The selected natural-phase run has `init_from=null`, and
its trainer constructs a new `R2Model`; the run's frozen copy of that trainer
is byte-identical to [`r2/train_ce.py` at commit `3954031`][r2-trainer], the
commit R2 was trained from. It uses supplied head rows and grids, joint lane
actions, taps, long-note heads and releases. Audio features never enter it,
though audio availability and duration affect cache eligibility and song-end
timing; with heads supplied, arrangement-only evaluation cannot test head
timing from sound. The released checkpoint, at 56,000,574 decisions of the
64M-decision run, was chosen by the corrected guard selection (the original
record selected none) among seven checkpoints from 16M to 64M decisions, on a
16-chart guard set wholly in `fit_dev` with source defect references from
`fit_train`. Neither released model inherits R1 weights; R1 contributes
architecture, chart-replay design, corpus grouping and partition ancestry.
Condition fine-tunes, preference arms and copy-model training are not in the
released weight path.

Rows training samples chart windows with replacement and R2 uses its corpus
sampler; with no per-chart draw history available, the indices show
eligibility, not whether or how often each chart was drawn.

### Numerical data, calibration and evaluation

The [corpus prior][prior-source] measures four head-row features and five
arrangement targets from each training chart, selects its method on `fit_dev`
and, as released, takes the median of the 64 nearest training rows within a
star band; its membership is reconstructed from cache roles and its empty
failure list. The [published prior][prior] keeps the four standardized
features and five targets of each of the 11,368 training charts, by star band.

Chart-level masks and long-note, difficulty and quad references use the 11,118
training charts with at least 256 head rows, excluding 250 shorter ones. Scope
calibration admits 132,053 windows from 11,117 charts, four sampled scopes at
each of 8, 16 and 32 seconds after length and head-count checks; the two
denominators differ by one chart. The published calibration and scope table
derive from these charts; the scope table keeps one row per scope, without
chart hashes or paths.

Lens D provides recognized **machine labels** and chart measurements for
stream and jack grade and subtype calibration: jumpstream uses positive stream
labels in headmass tercile 1, chordstream and chordjack the relevant positive
label in tercile 2. Lens V2, a separate partition, provides recognized **human
labels** for validation only; its 140 charts include 130 from `fit_train` and
ten from `fit_dev`, so it validates labels, not R2 on unseen charts.
Unrecognized cells are not treated as negatives. The difficulty pilot uses
supplied human head rows, so its measurements exclude audio-to-head errors.

Timing fitting uses mapper tempo, grids and head placements as references for
audio-derived beat and onset signals, fitting algorithm choices, not a neural
model. Tuning records contain 975 song indices over 962 audio hashes, and
development records 322 over 316; the indices are experiment identifiers, not
song counts. The final test selected 521 charts over 290 audio items; an audio
loader failure left 519 charts and 289 hashes, **all in R2 `fit_train`**: held
out from that timing experiment, not from R2's training.

End-to-end panels drive timing and rows from music, with source charts as
references; the 48-chart system panel's control checks use a 24-chart subset
at two seeds. Listening selections mix generated pairs, human-reference pairs
and full-song play slots. Membership, here and in the manifests below, records
selection or measurement, not every downstream use or that every item was
heard. Quality conclusions are in the
[system report](r2_system.md#what-has-been-checked).

### Additional calibration and selection populations

Twelve supplemental roles, 4,532 distinct charts all within the training and
development pools, cover the restoring-force experiment and its follow-up,
later decoder defaults and run manifests. The restoring force steers generated
arrangement statistics toward targets, with a gain setting its strength.

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

The gain comparison fixes source heads, grid and song end on 48 `fit_dev`
charts, including all sixteen difficulty-pilot charts; oracle variants also
receive the source's whole-song arrangement levels, and no audio features
enter. Donors supply target statistics; original mean donors and follow-up
candidate donors are `fit_train`, each candidate set with one chart outside
the 11,118-chart calibration subset. Human-pair null references, from both
pools, measure variation between source charts.

The experiment informs the decode-time gain in the
[system report](r2_system.md#r2); its donor recipes do not define the released
prior, and its source-grid results do not evaluate audio timing. The released
Python system fixes the held-lane share, heads-per-row and
all-busy-row gains at `(3, 3, 3)`
([`system/make.py` at `r2-1.0.1`](https://github.com/ensomi-labs/ensomi-model/blob/r2-1.0.1/src/ensomi_model/system/make.py#L54);
the same at `r2-1.0`) and disables the heads-per-row gain inside stream and
subtype scopes. Both experiments applied the raw coefficient three, unscaled
by a standard deviation and not estimated as a continuous optimum.

Quiet-rest selection compares source rests and rhythm with audio-driven rows
at two seeds, on one source chart for each audio item of the 96-audio,
231-chart rows panel. The difficulty-quantile population is reconstructed from
the 24 control-enabled entries of the system panel and the controls table, as
no per-quantile roster was recovered; it was measured with the earlier
quiet-rest setting, the final runs with the adopted one. These choices change
decoding, not weights. The natural-phase negative-log-likelihood (NLL)
validation manifest holds 64 windows from 64 `fit_dev` charts and the
free-generation manifest four `fit_dev` charts, both distinct from the guard
panel.

## Public preview source identities

The four released previews use the audio of these source charts, all in
`fit_dev`. A named source mapper is not the author of the generated
arrangement. The beatmap IDs identify the source beatmaps on osu!.

| Preview | Source `Artist` | Source `Title` | Source `Creator` | Source difficulty | Beatmap ID |
| --- | --- | --- | --- | --- | ---: |
| SaiGetsu | Otokaze | SaiGetsu (Cut Ver.) | Irone OSU | 4K Vibrant Seasons | 2894207 |
| Catch the Moment | LiSA | Catch the Moment | Maxus | Live the Moment | 1219748 |
| Hesitation Snow | fripSide | Hesitation Snow -version 2024- (Cut Ver.) | FerdiXIA- | Hard | 5483425 |
| Adventure | YOASOBI | Adventure | AelSan | Memorable Memories | 4661716 |

## External BeatThis training

Ensomi uses pretrained BeatThis `final0` for beat and downbeat inference
without fine-tuning it; the
[upstream model description](https://github.com/CPJKU/beat_this/tree/v1.1.0#available-models)
identifies it as the seed-0 main model trained on the usual training and
validation partitions, excluding GTZAN. A song processed at inference does not
thereby become BeatThis training data. The
[paper, section 4.1](https://arxiv.org/html/2407.21658v1#S4.SS1) reports 4,556
selected tracks; the pinned
[annotation release v1.0](https://github.com/CPJKU/beat_this_annotations/tree/c3c47fd37d3074d9f8119f18bbf460f909609f22)
contains 4,557 nominal candidate IDs in 15 `single.split` files, one per
collection:

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

These are upstream split entries, train and validation both being training
candidates for `final0`, not musicians, compositions, training excerpts or
examples verified as consumed. The
[upstream loader](https://github.com/CPJKU/beat_this/blob/v1.1.0/beat_this/dataset/dataset.py)
can omit candidates with missing augmented spectrograms or unsuitable
annotation shape, a possible but unestablished cause of the one-entry
difference; neither count is a verified ledger of what the original run
consumed. The published source list identifies no osu! mapping corpus.
Harmonix has rhythm-game authoring origins, which does not establish use of
released osu! or Rock Band gameplay charts; joining its
[metadata](https://github.com/urinieto/harmonixset/blob/64abeb509429e73d74559fb98e621dac866efea1/dataset/metadata.csv)
to its
[BeatThis split](https://github.com/CPJKU/beat_this_annotations/blob/c3c47fd37d3074d9f8119f18bbf460f909609f22/harmonix/single.split)
identifies Ciara's “1, 2 Step,” Lil Wayne's “6 Foot 7 Foot” and The Jackson 5's
“ABC” as nominal candidates, not as verified consumption. Completing the
musician roster needs dataset-specific metadata joins, reconciled loader
exclusions and separate attribution of recording artists, performers,
composers and annotators, which an upstream track ID or filename cannot give.

## Evidence scope and remaining gaps

The inventory resolves every chart identity in the 26 core and twelve
supplemental populations against the tracked
[`data/r2-corpus.parquet`][corpus] (SHA-256
`49ab9e1bbbdcf7a08bf86ee02181515dfa04bdc9efa3be25fe6bd4b0e6d9ee4b`, the hash the
R2 cache records), checking chart bytes and headers as of 2026-10-17 from
recorded source and configuration evidence, without loading checkpoints,
running inference or opening audio. The extraction recorded each input's
SHA-256; [`export_record.json`](r2_data_provenance/export_record.json) gives
those of the count tables behind the published files. Audio hashes are
recorded corpus values, not verified by this audit.

This covers the released weight inputs and the enumerated prior, calibration,
timing and evaluation populations, not every historical experiment: earlier
timing exploration and every historical listening screen are not reduced to
one manifest here, and they add no inherited weights. The optimizer draw
history, resolved identities, the complete BeatThis roster, source
permissions, chart quality and any source's causal contribution to model
parameters remain outside the evidence.

### What remains private

Per-item records derived from the private dataset are not published: the chart
inventory joining each chart's identity to its stages and to the headers read
during the audit, the per-stage chart hash lists, the R2 training cache and
the rows and prior indices, the timing, guard and Lens records, the panel and
listening selections, the supplemental roles' per-chart rows, the preview
sources' chart and audio hashes, and the conflicting strings behind the
integrity counts. Inspecting a source or repeating the extraction needs them
and the local chart collection. The frozen corpus table is tracked in this
repository. The BeatThis candidate table, the 15 public split files at the
pinned revision with each entry's dataset, track ID and split, is not
republished.

[corpus-source]: ../../src/ensomi_model/evaluation/corpus.py
[split-source]: ../../src/ensomi_model/r2/splits.py
[rows-fit]: ../../src/ensomi_model/audio_rows/fit.py
[rows-data]: ../../src/ensomi_model/audio_rows/beatdata.py
[rows-decode]: ../../src/ensomi_model/audio_rows/rowsgen.py
[prior-source]: ../../src/ensomi_model/system/prior.py
[r2-trainer]: https://github.com/ensomi-labs/ensomi-model/blob/39540317be42aa53104939c759e03610d64e6a3e/src/ensomi_model/r2/train_ce.py
[corpus]: ../../data/r2-corpus.parquet
[prior]: https://huggingface.co/sed-i/ensomi-r2/blob/r2-1.0.1/prior.json
[style-rows-inputs]: https://github.com/ensomi-labs/ensomi-model/blob/cf753aafa92530f82915c936b88eecd31ef6535d/src/ensomi_model/audio_rows/beatdata.py#L249-L256
[style-r2-inputs]: https://github.com/ensomi-labs/ensomi-model/blob/39540317be42aa53104939c759e03610d64e6a3e/src/ensomi_model/r2/features.py#L35-L40
[style-r2-init]: https://github.com/ensomi-labs/ensomi-model/blob/39540317be42aa53104939c759e03610d64e6a3e/src/ensomi_model/r2/train_ce.py#L188-L191
[style-r2-draw]: https://github.com/ensomi-labs/ensomi-model/blob/39540317be42aa53104939c759e03610d64e6a3e/src/ensomi_model/r2/data.py#L102-L105
[style-groups]: https://github.com/ensomi-labs/ensomi-model/blob/39540317be42aa53104939c759e03610d64e6a3e/src/ensomi_model/evaluation/corpus.py#L243-L258
[style-prior]: https://github.com/ensomi-labs/ensomi-model/blob/9e380a15f7c193476ff2063dfc10043e8484186a/src/ensomi_model/system/prior.py#L32-L35
