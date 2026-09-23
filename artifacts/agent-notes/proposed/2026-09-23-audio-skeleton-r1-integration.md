# Agent Note: Audio skeleton generation and R1 integration

Note ID: 2026-09-23-audio-skeleton-r1-integration
Status: proposed
Kind: research
Created: 2026-09-23
Updated: 2026-09-23
Product revision: 16209eaf9000867de72ba9c5f04989e1b1e990f6
Scope: Released R1 timing sensitivity; canonical audio-conditioned joint timing/action learning, native playability and inference
Related: 2026-09-20-r1-response-recovery-handoff

## Question and authority

Can an audio-conditioned model provide head times and release opportunities that
support playable generation with the released R1, adapting either model when
evidence warrants it? Complete source audio is available. The client protocol
supports incremental LN delivery; resolving a future tail before publishing its
head is not a requirement.

On 2026-09-23 the human owner explicitly authorized implementation, local
experiments, architecture iteration, training and joint optimization with R1 on
the available Mac, including overnight compute after small experiments. This
standing execution authority supersedes a separate approval before each local
experiment. It does not imply Note acceptance, remote publication, modification
of human annotations or a claim that a candidate is playable. Cards remain
proposed and results are exploratory until independently supported.

The existing orphan notes ref is attached to a dedicated worktree for research
persistence. Product implementation is on `codex/audio-skeleton`. The notes ref
is not merged into the product branch.

## Baseline and observations

Current research direction: fix the owner-confirmed repository music Mel frontend
and develop a simple learned audio encoder together with joint event timing and
R1-derived complete-row generation. The human owner rejected encoder information
sufficiency as a prerequisite/evaluation target: sufficiency depends on the
architecture and observed arrangements are not unique answers. Available direct
supervision is paired beatmaps/audio; no MERT-style pretraining dataset or acoustic
teacher labels are assumed. New long-term musical-relation memory is deferred.

R1 is temporary and may change; complete audio content beyond rhythm must
influence arrangement. Preserve ordinary subdivisions, Tech/high-fraction timing,
one musical cue elaborated into a long Jack/dump, independent LN release and
repetition with variation. Do not impose rigid chorus/section labels. Final
playability and expressive coverage take priority over reconstruction scores.
Use actual Beatmap Lens source/render inspection and human comparisons to identify
bad relationships; style presence is not a quality verdict.

The product research proposal is
`docs/research/audio_conditioned_choreography.md`. It distinguishes exact replay,
recent detail, audio/arrangement recurrence memory and provisional plans. Its
current exploration outcome is TEST for the small joint model family. The active
exploratory Card below specifies the first joint implementation and bounded run.
Existing pilot implementation and weights remain separate exploratory baselines.

- Release tag: `r1-restored-6.75m`, source
  `8f1310322ba1f64a1ca259893e76376daf36396e`.
- Model repository: `sed-i/pulsefield-r1-restored`, revision
  `0cf786c2018d769af8edf1afadc6cfd163eed5e4`.
- Checkpoint SHA-256:
  `4b3ec1561e33d0ebe2756cfe13571ec414fd5bb470b430f0c578545863115f70`.
- The 3,084,432-parameter restored model is distinct from the historical deleted
  response model. Historical quality measurements cannot serve as its baseline.
- Production seeds cover the complete row reaching 30 heads. H requires heads;
  R permits releases. Timing inputs include whole-schedule summaries. Start with
  a complete fixed schedule and preserve the supplied seed for causal isolation.
- Runtime: Apple M5, 10 CPU cores, 24 GiB unified memory, macOS 27.0,
  Python 3.10.20, Torch 2.11.0, BeatThis 1.1.0, MPS available.
- Available corpus: pinned catalog
  `artifacts/oracle-time-review/20260915-adfb1ee/catalog.json`, SHA-256
  `e31b7e8f4daa044503ef2b8411bc41727ba371eec804c9462a4608be8112ad28`,
  and admitted rows at `artifacts/oracle-time-continuation/full-cache-v1`.

## Architecture alternatives

The proposed skeleton model combines a local audio timing encoder with optional
frozen BeatThis contextual features. Separate heads predict required head times,
release-only opportunities and sub-frame offsets. A declared chart-level density
condition distinguishes different arrangements for the same music. Source-derived
conditioning used in validation must be labeled as supplied control information,
not audio-only prediction. A train-derived default condition is evaluated separately.

The first comparison is local audio features alone versus additional frozen
BeatThis features with matched training data and budgets. Fine-tuning the audio
encoder, autoregressive rhythmic planning and R1 adaptation remain evidence-driven
follow-ups. The sparse independent release target must not be replaced by a dense
grid without measuring the resulting R1 distribution shift.

Closest analogues: [BeatThis](https://arxiv.org/abs/2407.21658) provides contextual
beat representations, but predicts beats/downbeats rather than chart events;
[Dance Dance Convolution](https://proceedings.mlr.press/v70/donahue17a.html)
separates chart timing from placement and conditions timing on difficulty. The
combination is an adaptation; no new architecture or research novelty is claimed.

## Completed exploratory plan: audio-skeleton-r1-support, revision 1

### Identity and authority

- Owning Note: `2026-09-23-audio-skeleton-r1-integration`.
- Card ID: `audio-skeleton-r1-support`; revision: 1.
- Execution authority: explicit human authorization above; acceptance: none.

### Comparison

- Question: Does increasing non-head opportunities materially change native R1
  holds, independent of head timing and seed quality?
- Baseline source: the clean product revision above; baseline checkpoint as pinned.
- Baseline value: not previously measured for this release. Establish it using
  the same cohort and runner before interpreting paired interventions.
- Selection: deterministic source-only song-group selection, seed 20260923,
  validation strata by head-row density below 4, 4–8, and at least 8 per second,
  crossed with source LN-head fraction below/at least 0.1. Two distinct groups
  per available stratum, full charts lasting 60–360 seconds with a suffix.
  Freeze exact selected source/cache identities in a manifest before generation.
- Intervention: insert one midpoint non-head candidate in respectively one in
  four or every eligible post-seed original candidate gap. Do not change H,
  original R, the real terminal, seed rows, or crossing-LN endpoint timestamps.
  Remap crossing candidate indices. Only gaps at least 20 ms are eligible.
- Paired generation seeds: 17 and 23. Same released parameters and temperature 1.
- Primary diagnostic: per-chart suffix-born LN median-duration ratio relative
  to baseline, then median across paired charts where both have LNs.
- Triage threshold: more than 20% median shift, or at least three charts shifting
  over 40%, triggers candidate-distribution adaptation investigation. Thresholds
  are engineering triage choices, not playability or statistical significance.
- Guards: every output independently legal with H coverage, unchanged seed and
  known endpoints, and exact export/reparse; no failed case silently excluded.
- Additional measurements: LN quantity, release-only rows, concurrent-hold
  exposure, sharp same-lane gaps, full-chart generation cost and per-step latency.
- Qualitative check: inspect worst changed LN passages and density transitions;
  neither aggregate duration nor LN amount establishes retained organization.
- Bounds: 45 minutes for the small sensitivity suite, up to four CPU workers
  with one Torch thread each, 18 GiB aggregate process memory, 40 GiB free disk
  reserve. Outputs under fresh `artifacts/audio-skeleton/20260923-v1/` owners.
- Stop: changed input digest, illegal output, nonfinite state, explicit pause,
  memory/resource pressure or time limit. Preserve failed outputs and reasons.
- Confounders: inserted decisions change RNG consumption; paired seed alone
  does not couple every subsequent draw. Report multiple seeds and trajectory
  differences. Whole-schedule features also change; this is total sensitivity
  to added opportunities, not isolation of release hazard alone.
- Positive/negative interpretation: stability supports a subsequent predicted
  skeleton probe; instability supports adaptation or better candidate selection.
  Neither outcome establishes musical alignment or final playability.

## Next work and completion conditions

Record source commits, dataset/feature manifests and run results before training
comparisons. Preserve song groups and quarantine exact audio duplicates across
training and validation. TEST payloads remain unopened. Keep the original R1
fixed for timing comparisons, then evaluate targeted R1 adaptation if necessary.
Train a small learning check before a larger run. Evaluate native generated
charts, opening compatibility, playback buffer costs and player-facing quality
before claiming a final playable system.

## Result Log: R1 support sensitivity

Execution source: `135e181` on the product branch (resolve from Git for the full
OID). Manifest SHA-256:
`d416a1953bb4f0126c084457cd8d6c597c96533da9f40d8e3245949006af934c`.
The manifest fixes 48 TRAIN and 12 VAL song groups. Sensitivity used the 12 VAL
charts, three candidate conditions and seeds 17/23: 72 complete outputs.
All outputs passed independent mechanics and exact export/reparse. The raw owner
is `artifacts/audio-skeleton/20260923-v1/sensitivity/summary.json`.

Across chart/seed pairs, inserting at every fourth eligible gap gives median
suffix-born LN-duration ratio 0.72485 and LN-head-fraction increase 0.25407.
Inserting at every eligible gap gives ratio 0.375 and increase 0.69504. Baseline
median LN-head fraction is 0.16979. Thus candidate inflation changes both note
types and durations substantially; the frozen input bridge fails the declared
sensitivity triage threshold. This is a total schedule intervention and does not
isolate opportunity count from changed timing features or RNG consumption.
Generated organization still requires scoped visual inspection.

Baseline full-chart generation took 1.01–12.31 seconds per case with four CPU
workers, one Torch thread each, alongside local audio caching. Model load and
export are excluded. These are neither isolated latency nor end-to-end startup
measurements. The result motivates sparse learned release opportunities and a
subsequent matched R1 adaptation comparison, not a universal ban on dense grids.

## Completed exploratory plan: audio-skeleton-feature-pilot, revision 1

- Owning Note: `2026-09-23-audio-skeleton-r1-integration`.
- Card ID: `audio-skeleton-feature-pilot`; revision: 1; acceptance: none.
- Execution authority: the standing human authorization above. Exploratory.
- Question: Do frozen BeatThis representations improve chart-event timing beyond
  a local log-mel timing network under supplied coarse density controls?
- Closest analogues: BeatThis contextual audio features and Dance Dance
  Convolution's conditioned timing stage; joint R1 compatibility remains distinct.
- Baseline: clean product implementation, fixed corpus manifest above, newly
  initialized local encoder. Baseline values are measured by the paired pilot.
- Intervention: add the frozen `final0` BeatThis embedding and beat/downbeat
  logits through a zero-initialized projection. Both arms instantiate the same
  architecture and share initialization/sampling seed 172. Active capacity and
  information both change; this does not isolate representation from capacity.
- Architecture: 100 Hz log-mel detail path, width 128, separable bidirectional
  temporal convolutions, pooled slower context, coarse head/release-only density
  conditions, two ordered event slots per role per 10 ms frame, and sub-frame
  offsets. The additional slot's probability cannot exceed the first's.
- Target audit: a single slot would merge 41 head events in one VAL chart and
  two release-only events in another. Maximum observed per-role occupancy is
  two. Labels preserve exact source times; larger collisions fail explicitly.
- Controls: log1p head and release-only rows per audio second. Supplied source
  controls isolate timing under a requested quantity; a separate TRAIN-median
  condition is evaluated. Neither is called calibrated difficulty.
- TRAIN: 48 groups, one selected chart each. VAL: 12 groups, one from each of
  six density/LN strata for calibration and its paired group for assessment.
  Source selection does not establish prior R1 validation independence.
- Optimizer: AdamW, learning rate 0.0003, weight decay 0.01, gradient cap 1.
  Batch 12; 1,200 updates; uniform chart and time-center sampling; 16-second
  windows with 4-second halos and central 8-second supervised spans.
- Objective: weighted event BCE (12/48 for primary/additional H slots, 24/96 for
  release-only slots), plus positive-only smooth-L1 offsets. Padding/halos have
  no target loss. Slot presence is nested, while decoded times are ordered.
- Selection: calibrate each role's threshold in 0.1 increments from 0.1 to 0.9
  on the six calibration songs every 200 updates. Select the mean of H and
  release-only macro F1 at 20 ms. Event F1 averages only reference-positive
  charts; false positives on empty-reference charts are reported separately.
- The earlier four-chart learning check demonstrated falling loss and head
  learning, but its release macro score included empty-chart correctness and
  is not comparable to this corrected event metric. Keep its raw log unchanged.
- Primary comparison: paired assessment-song head/release-only F1 at 20 ms;
  also report 10/40/70 ms, precision/recall, counts and default-control behavior.
  Five percentage points is a useful pilot improvement, not a significance or
  playability threshold. Worsened precision, rest behavior or downstream R1
  organization can defeat an apparent timing-score gain.
- Budget: at most 10 minutes per small pilot and 30 minutes for feature caching,
  MPS training, two CPU Torch threads; stop with under 2 GiB available memory,
  nonfinite loss, explicit PAUSE, or time budget. Cache retains 40 GiB disk
  reserve. Larger overnight execution is authorized but follows these probes.
- Products: fresh run directories under
  `artifacts/audio-skeleton/20260923-v1/training/`, pinned feature index, resolved
  YAML, typed configuration, optimizer/RNG checkpoints and separate assessment.
  No existing run is overwritten. Resume uses a fresh run owner.
- Confounders: small single-seed cohort; source-density information; original
  pretrained audio overlap not audited; thresholds/checkpoints selected on six
  calibration groups. Assessment remains development evidence.
- Interpretation: improvement supports a larger grouped-data experiment and
  native R1 integration. Failure calls for rhythm/sequence modeling or audio
  adaptation after checking synchronization and label coverage. Neither outcome
  supplies a complete audio-only opener or final player-quality acceptance.

## Result Log: audio pilots and materialization

Local pilot source: `68a48aad2952ca153bdc205b6afaee4b5e2f46d5`.
Both variants trained for 1,200 updates with seed 172 on the same 48 TRAIN songs;
six VAL songs calibrated thresholds/checkpoint selection and six supplied
assessment. The local model selected update 600; the BeatThis-conditioned model
selected update 200. Models have 770,188 parameters, including an instantiated
but unused BeatThis projection in the local arm. The BeatThis encoder is frozen
and separately pretrained, so total model capacity and prior training differ.

With source-derived coarse density controls, assessment head F1 at 20 ms is
0.67713 for local audio and 0.73186 with BeatThis; release-only F1 is 0.03541 and
0.05130. With TRAIN-median controls, head F1 is 0.61941 and 0.69793; release-only
F1 is zero for both. These are descriptive, single-seed development results.
They measure reference timing agreement, not valid-alternative coverage or
playability. Rare release-only rows and different arrangements limit the proxy.

Local best weights SHA-256:
`704949a4078dd2c4cc77d557c0d315529a7f34f02a2272a041007afd4f549593`.
BeatThis `final0` checkpoint SHA-256:
`8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331`.
Native integration generated 48 cases per pilot: predicted extra releases with
source H versus fully predicted timing, source/default controls, and seeds17/23.
These initial outputs passed the model's internal mechanics/reparse checks.

The Lens parser then flagged fractional hitobject start/end timestamps. Its strict
preparation helper rejects nonzero diagnostics although the parser preserves the
fractional values. This exposed a materialization gap in the internal evidence;
it did not prove those values were illegal under the continuous-time V3 language.
Product commit `068988e670e174621f96627827dc28386b1e6775` quantizes proposed
native-export times before R1 and uses an integer true terminal. Earlier raw
outputs and metrics remain unchanged. The new owner is `integration-ms/`.
All 48 BeatThis integration cases completed there. Selected source/generated
charts subsequently passed canonical Lens preparation with zero diagnostics.

The integer-time sensitivity rerun has a separate owner:
`artifacts/audio-skeleton/20260923-ms-sensitivity/sensitivity/summary.json`.
All72 outputs pass internal mechanics/export/reparse. Every-fourth-gap insertion
gives median duration ratio0.68586 and LN-head-fraction delta0.25624; every-gap
gives0.375 and0.68595. Use this rerun for native-ms comparison rather than silently
replacing the first run. No whole-suite Lens-admission claim is made from the
selected-chart preparation.

## Result Log: target distribution and Lens inspection

The bounded TRAIN audit covers418 readable charts and149 exact-audio hashes from
128 selected musical groups; its seed-eligible, at-least60-second subset retains
416 charts and584 alternative-arrangement pairs. Median pairwise head F1@20ms is
0.785, all-release F1 is0.281 and release-only F1 is0.092. Among pairs with head
density ratio at most1.2, head F1 is0.941 but release-only F1 is0.169. Of release
rows,78.7% coincide with a head. These are pair-weighted descriptive measurements,
not population estimates or proof of audio/chart causal relationships.

The source minimal skeleton is the projection of an actual chart; unused serving
opportunities are not directly observed labels. A release-only target changes
when the selected head sequence changes. These findings motivate joint or
explicitly dependent release modeling rather than assuming two independent
audio detectors are sufficient. Detailed audit and limitations are under
`artifacts/audio-skeleton/20260923-v1/distribution-audit/`.

Actual Lens calls, returned source rows, rendered images and reviews are under
`artifacts/audio-skeleton/20260923-v1/lens-review/`. The valid inspection bundle
is `bundle-ms-v2`, based on a hash-checked human snapshot with204 examples and
current Lens tools from `22e5c84f5cacb8493bdab5f1d0fdc09c5373dc60`. Generated
notes were parsed with Lens's canonical source parser. No canonical annotations
were changed. Earlier preparation attempts are retained as failed artifacts.

SCREW generated SHA
`abb96b25861f6c828c8740e0113dc36aed64f764367cd2b635a4d339c855240d`,
90000–92000ms, has column1 release/repress at90369→90381 and90709→90721,
both12ms, with other columns available. These are specific agent-identified
counterfactual preference candidates. They are not human BAD labels or universal
minimum-gap rules. Case2 exposes closely spaced heads and higher chord burden;
cases3–6 and sampled early/mid/late windows include plausible alternative
arrangements. Source density differences alone do not reject an expressive dump.

Opened human contrasts include `human-d15c6a35c9ee8a86d981a9a4` (short LNs under
a longer hold; LN coordination absent), `human-03f7e300cf02f58f3dcbba66`
(independent overlapping LN coordination prominent), and the reviewer's
`human-682c8969424a86213ab1b271` (mixed-gap Tech supporting) and
`human-aab7a5db23744f5daae1dec1` (Jack prominent). Their style labels constrain
interpretation but do not provide good/bad or numerical demand targets. The
review report distinguishes complete row coverage from visually inspected pages.
No audio listening, human playtest or whole-chart quality acceptance occurred.

## Implementation stopped before the next research decision

A consequence-only correction draft was written but not tested, imported, trained
or committed. It is preserved at
`artifacts/audio-skeleton/20260923-v1/drafts/r1_adaptation-unselected.py`.
It only addresses equal-composition alternatives under the existing30ms,
two-onset machine preference; it cannot solve phrase structure or general
playability. The human owner's instruction to establish formalization and genuine
research trajectory precedes choosing this or a different intervention.

All pilot training and generation jobs have completed. No overnight training,
adaptation run, publishing, or recurring automation was started. Product code
remains a baseline on `codex/audio-skeleton`; no change was pushed remotely.

## Fixed frontend and revised immediate architecture

The selected source is `features/mel_base.py::MUSIC_MEL_CACHE_CONFIG` with the
repository waveform loading convention: mono 24 kHz, 128 bins, 10 ms hop, 40 ms
Hann window, FFT/window 960, 20–12,000 Hz, center=False, power2, norm1, natural
log and floor1e-5. The owner manually confirmed this representation. Frame i
uses [10i,10i+40) ms and is centered at10i+20 ms; reuse the canonical padding and
frame-count semantics rather than a nominally similar independent extractor.

The completed pilot used FFT1024, centered windows, log10, floor1e-10 and different
Mel/normalization settings. It is not a comparison on the owner-confirmed frontend.
Preserve its caches and model identities; the next experiment needs a new feature
owner. Reusing the general-music frontend does not adopt legacy mapper/timing
architecture.

The minimal proposed factorization predicts next event time, then a complete
nonempty row, from shared encoded Mel, chosen action history and exact gameplay
state. Head-only/release-only/combined roles are derived from the row. This avoids
an extra role sampler initially and gives actual holds a path into future timing.
A small frequency-preserving projection and residual temporal convolutions are
the initial audio encoder family; bidirectional audio context is allowed.

Teacher-forced time and row losses jointly train their shared encoder/history.
The row loss evaluated at source timestamps does not differentiate through the
time-head sample; do not claim direct downstream playability optimization from
that alone. Native joint rollouts and Lens inspection remain necessary. R1's
source-only future-schedule features and candidate-list-dependent support must be
replaced consistently, with useful weights retained as initialization. Train BOS
and short prefixes, preserve exact LN state and censored windows, and do not
fabricate future endpoint knowledge.

The immediate sequence is representation/state checks, a small joint learning
check on paired audio/charts, native joint generation with error attribution, then
one failure-driven correction. BeatThis, larger pretrained encoders, latent plans
and new long-term retrieval remain optional deferred work, not entrance gates.


## Experiment Card: joint-mel-hazard-rows-v1

Revision: 1. Owning Note: 2026-09-23-audio-skeleton-r1-integration. Acceptance:
none. Status: proposed. Execution uses the explicit standing local research and
training authority recorded above; this is an exploratory learning check.
Previous cards retain their historical results and do not control this run.

Question: can the shared canonical-Mel encoder and history-conditioned native-ms
hazard learn together with a complete-row decoder initialized from compatible R1
weights? The intervention replaces supplied R/H with a learned event distribution;
this is a new joint baseline, not an isolated estimate of encoder benefit.

The closest primitive is a marked temporal point process with discrete hazards
and conditional marks. Every native millisecond remains available, and a complete
row jointly owns head/release roles. The hazard reads candidate-local music and
actual history. Its survival product and carried exponential residual make
scheduler partitions immaterial to the waiting distribution. A gap mixture and
independent frame peaks are deferred: the former needs audio alignment machinery;
the latter discards chosen-action dependence. No claim of research novelty.

Clean implementation baseline: e2bca3e592800c1a49a1a82b31ae488fa7021e47. Record
the intervention commit in the result before execution. R1 checkpoint SHA remains
4b3ec1561e33d0ebe2756cfe13571ec414fd5bb470b430f0c578545863115f70.
The new model has 2,950,458 parameters, including a 463,392-parameter Mel encoder;
2,444,688 parameters copy from R1. Omit the seed residual, landmark memory,
candidate consequence and 906 future-query projection columns. Retain exact
state and the 511-row finite content encoder. No pretrained audio model or style
teacher is used.

Data: prepare the canonical frontend afresh at
artifacts/joint-audio/20260923-v1, based on the old pilot manifest
d416a1953bb4f0126c084457cd8d6c597c96533da9f40d8e3245949006af934c
and catalog e31b7e8f4daa044503ef2b8411bc41727ba371eec804c9462a4608be8112ad28.
Keep 48 TRAIN and 12 VAL base songs; admit up to two distinct TRAIN arrangements
per exact audio/group, with no TEST data. Hash-verified rows beyond the decoded
audio end fail rather than silently changing VAL. TRAIN normalization weights
unique audio once. The initial fit selects six deterministic TRAIN groups, one
per density/LN stratum, keeps their alternatives distinct, and samples a group
then a chart.

Procedure: run canonical preparation with the packaged joint_audio Hydra
`mode=prepare max_seconds=900`. After representation/integration tests pass, run
`mode=train run_name=memorize-v1 fixed_train_queries=32 train_groups=6 updates=300
validation_every=100 batch_size=8 cpu_threads=2 max_seconds=900`, with other
packaged defaults. Use seed 230923, shared/new learning rate 3e-4, inherited
learning rate 3e-5, AdamW decay .01 and gradient clip 1.
BOS/event-prefix/absolute-time/outro query probabilities are .08/.70/.17/.05;
the horizon is 4000 ms, with native 1 ms hazards and ten outputs per absolute
10 ms bin. Score exactly the same diagnostic queries before training and at
updates 100/200/300.

Primary diagnostic: mean joint NLL per fixed TRAIN query; separate event/survival
NLL and mean action NLL for noncensored targets. A reduction of at least 20% in
joint NLL is a learning-path gate, not statistical proof or playability. Neither
component may become nonfinite. Report VAL on four fixed queries for each of 12
songs descriptively; make no held-out quality claim from this unblinded cohort.
The baseline value is measured at update 0; no previous model has an equivalent
likelihood. Paired queries remove query-sampling variation, but one training seed
and the very small memorization pool remain strong confounders.

If the gate passes, generate native BOS samples from the best diagnostic
checkpoint on six TRAIN cases, then inspect exact event/hold behavior and Lens
source/render patterns. This diagnoses rollout mismatch before data scaling;
memorized-query success alone never authorizes a quality claim. Timings may be
very poor away from the fixed queries. A wider random-query learning run requires
an explicit subsequent Card revision and a result-grounded choice under the same
standing authority.

Guards: native rows strictly increase on integer clocks and are never all-empty;
there is no head on a held lane or release without a hold, and the true terminal
closes all held lanes. Censored query boundaries never force closure. Source
endpoints and future timing do not enter predictor inputs. Tests cover multiple
events within a frame, long rests, BOS at 0, crop/full audio parity, mirror
consistency, gradients and scheduler partition survival. Native output requires
canonical mechanics, export/reparse and Lens admission. Qualitative checks
preserve possible Tech, dump/Jack and independent releases; no global minimum-gap
or repetition ban is introduced.

Run bounds: Mac M5, 24 GiB, MPS, two CPU threads for training, up to 900 s per
preparation/fit, a fresh run directory, no overwrite/resume, at most 30,000 rows
per song and a 900 s generation bound. Stop on PAUSE, nonfinite loss/gradients,
available RAM below 2 GiB or free disk below 40 GiB. No network or pretrained
downloads. Save resolved Hydra, flat config, corpus and checkpoint pins,
parameter-transfer report, RNG states, update/resource logs, validation records
and explicit completion/stop cause. Raw artifacts stay local; code and the owning
Note are committed separately with no remote publication.

Environment deviation before the run: existing SciPy 1.15.3 binaries had malformed
Mach-O TLS zero-fill offsets rejected by macOS 27. Eight extensions received only
section-offset metadata repairs and ad-hoc signing. Every file-backed section
remained byte-identical; versions, lockfile and canonical frontend were unchanged.
Original binaries, script and receipt are preserved at
artifacts/joint-audio/20260923-v1/runtime/scipy-dyld-repair/receipt.json,
SHA 6dcf6178d89f22e0a563c2a9fc699c5b6a1c5f92a90b26807eed431825ee5ffe.
Canonical data tests and four real/complex PROPACK SVD smoke checks succeeded.
This is a local environment repair, not shipped model code.

Interpretation: a successful bounded fit warrants evaluating native errors and
broader paired training, not retaining this architecture by default. Failure to
fit directs gradient/representation analysis before scale. Good likelihood with
poor native outputs directs investigation of feedback and multimodal choices.
No sampled sequence is declared BAD solely for disagreeing with its source.

### Execution start: joint-mel-hazard-rows-v1

Clean intervention source: 09b919cdeab90e3856fee03a9198d59b4dc527af. Added only
the joint research owner, packaged schema, focused tests and scoped design-doc
update. Prior R1 and audio-pilot code were unchanged. Local checks cover 96 owner
tests, one package-layout test and 22 package-resource subtests. One test fixture
was corrected while the initial combined test process was running; its final
focused parity test passed. An MPS CPU/float64 conversion bug was found and fixed
before commit. These are contract/implementation checks, not generation quality
evidence.

Preparation launched with the recorded command and clean source. It writes
canonical features freshly; the old experimental Mel cache is not reused.

### Exploratory Result: joint-mel-hazard-rows-v1 memorization

Preparation completed in 20.99 s with 121 TRAIN charts across 48 groups and 12 VAL
charts across 12 groups. Manifest SHA:
4b995029a5344569d4506ff6b11249f61585d2bf7649285754340909bb06c21b.
Thirty alternative candidates had different audio bytes, five had unreadable
paired audio and one failed the inherited admitted-cache minimum-seed rule.
All 60 base songs were retained. New generation itself requires no source seed.

The six TRAIN groups contain 14 arrangements (3, 3, 3, 3, 1, 1). The seeded
32-query diagnostic happened to contain 32 events and no censored examples; this
limits its ability to demonstrate learned silence or long-rest survival.
Survival/state mechanics were tested separately.

The 300-update fit completed in 74.14 s. Fixed TRAIN query joint NLL fell from
9.6190 to 0.001238; time NLL from 5.87993 to 0.0009102; row NLL from 3.73911 to
0.0003279. Fixed VAL joint NLL rose from 10.4942 to 45.3490; time NLL from 7.1513
to 38.2366; event-row NLL from 3.5657 to 7.5866. Thus the gradient path and
representational memorization gate passed while generalization strongly worsened.
This is expected evidence of overfitting, not a playable candidate. Best and last
were both update 300, chosen by the TRAIN diagnostic score. Best checkpoint SHA:
cd935cb876b0716b34459425fcfd27314971c6ceaf1cc14cb038cb05e2bfa42c.
Observed MPS driver allocation was 1.31 GB, RSS approximately 0.98 GB and available
RAM approximately 7.2 GB. No resource guard or nonfinite failure occurred. Raw
freeze/evaluations/update logs and checkpoints remain under
artifacts/joint-audio/20260923-v1/training/memorize-v1.

Native six-TRAIN-case generation launched using the explicit checkpoint pin,
CPU with one thread, seed 17 + case index, a 900 s aggregate bound, 4 s query
chunks and no source rows. The first complete case, 0e5557107b9f, covered 278.23 s
in 35.04 s with 4083 rows and 4114 heads. Its 8 s coverage took 1.793 s; step P99
was 12.50 ms, including whole-song learned encoding but excluding decode, Mel and
cache verification. About 272 per 1000 heads followed a same-lane attack/release
within 20 ms. Lens inspection is pending to characterize the actual organization.
No threshold alone is a BAD label. This case is a diagnostic overfit rollout, not
a final quality result.

Current research recommendation: REFINE through randomized paired-chart training
if native inspection confirms sampling/coverage failure rather than a broken
state/probability contract. Do not enlarge the encoder before that comparison.

### Native completion and next bounded comparison

All six memorization native TRAIN cases completed in 140.60 s total on CPU with
one thread; every case passed exact mechanics/export/reparse. Cached-Mel coverage
of the first 8 s ranged from 0.554 to 2.169 s. Their same-lane <20 ms relation
counts ranged from 196 to 333 per 1000 heads. This consistent local pressure and
the held-out likelihood regression reject using the 32-query fit as a candidate.
Lens inspection runs separately on these actual outputs; no numerical relation
threshold is treated as universal playability.

## Experiment Card: joint-mel-hazard-rows-v1 (revision 2)

Owning Note and standing authority are unchanged; status is proposed, acceptance
none. Revision 1 is completed exploratory evidence above. Revision 2 keeps the
exact representation/model and pinned canonical corpus, changing only training
exposure from 32 fixed queries to fresh group → chart → query sampling across all
48 TRAIN groups and 121 arrangements. This is a necessary full-distribution
baseline after the memorization check, not evidence for a larger architecture.
Baseline source and intervention source are both
09b919cdeab90e3856fee03a9198d59b4dc527af; there are no code changes. Use the same
R1 checkpoint and initial random seed 230923. The same 12 VAL groups and 48 fixed
VAL queries have untrained baseline joint NLL 10.4942, time NLL 7.1513 and
event-row NLL 3.5657. This is descriptive, single-seed evidence.

Exact command:

```sh
uv run --extra mps python -m ensomi_model.research.joint_audio_continuation.hydra mode=train run_name=random-v1 fixed_train_queries=0 train_groups=0 updates=2400 validation_every=400 batch_size=16 cpu_threads=2 max_seconds=1200
```

All other schema defaults and the optimizer, objective and query mixture remain
fixed. Use fresh output training/random-v1 with no resume/overwrite. The bound is
1200 s, with unchanged guards for 2 GiB available RAM, 40 GiB free disk, nonfinite
values and PAUSE. Paired fixed 48 VAL queries compare update 0 and the selected
checkpoint; 48 fresh TRAIN probe queries are descriptive only.

Learning gate: VAL joint NLL improves at least 10% over update 0, with both time
and row NLL finite and neither more than 10% worse. This is a development learning
gate, not a model quality metric. A failure warrants investigating objective/data
fit before adding parameters. A pass proceeds to native BOS generation on the
same six TRAIN songs, using seed 17 + index, and six VAL songs, with identical
sampling, a 900 s aggregate bound per cohort, the pinned best checkpoint and CPU
with one thread. Preserve all output rows, timings, mechanics and source-render
context.

Primary adoption evidence remains native organization: inspect openings, dense
passages and independent-release passages using Lens, and compare the actual
local relationships to source/human examples. Long constant-spacing/Jack
organization, complex fractions and asynchronous releases must remain available;
global sparsity/regularity is not a goal. Generated density differences alone
remain ambiguous. Compare rates of suspicious local relationships descriptively
and review concrete examples rather than declaring a threshold-based win.

This comparison changes exposure and batch size, so it cannot isolate a single
optimizer effect or prove architectural superiority. It asks whether the chosen
small model begins to learn a transferable joint distribution from the available
paired charts. Even good source-conditioned NLL can coexist with bad generated
history. New architecture, memory and BeatThis remain deferred pending that
failure attribution. No human playability acceptance is implied.

### Lens review of the completed memorization outputs

Two of six outputs were reviewed through actual frozen Lens calls and viewed
time-proportional renders. Report:
artifacts/joint-audio/20260923-v1/lens-review/memorize-review.md;
identities/traces: memorize-review-identities.json. Bundles memorize-case1-v1 and
memorize-scars-v1 retain 204 historical human examples unchanged; each has 25
traced harness calls, with all 145 manifest files reverified. Generated and source
charts passed the canonical Lens bridge with 0 diagnostics.

Imaginary Waltz, generated SHA
50f9a7c1f7f515e816e75d7803af1e8eb5036ee7337b6bf4f227402c93d5d21a,
has 36 attack rows in 4600–4900 ms. Column 1 repeats at 4836 → 4846 → 4856;
columns 0/1 repeat at 8846 → 8849, with columns 2/3 at 8847. Scars, generated SHA
e9a40e97129fa21b4474c7ec9ccc84054f2e2dbc6e38faeb6de0671917702d5e,
repeats the same columns 0/1 chord at 2717 → 2719 and column 1 at 21325 → 21326.
These repeated 1–10 ms individual-key demands reject both inspected samples as
playable candidates. They are not merely dense, irregular, Jack-like or different
from the source. No assertion is made that the architecture inherently requires
this failure. Repeated 3/7 ms spacings and 10 ms recurrences are observations, not
a causal claim about binning.

Scars source has staggered LN/tap control absent from the sampled generated
scopes; this is missing observed organization, not a blanket requirement to copy
LN fraction. Human short-LN and independent-release examples were opened again
as guards against indiscriminate sparsification. No listening/playtesting or
whole-chart visual acceptance occurred. The root also viewed both the Imaginary
Waltz 300 ms zoom and its 8500–10000 ms dense image.

Revision 2 random-query training started from the same pinned R1 initialization.
At update 400, fixed VAL joint NLL fell from 10.4942 to 6.40574, time NLL from
7.15130 to 4.69706 and event-row NLL from 3.56572 to 1.82260; both components
improve. The 2400-update bounded run is ongoing. No model-size, architecture,
checkpoint-resume or dataset change was introduced.

### Exploratory Result: revision 2 randomized paired-chart fit

Completed 2400 updates in 989.40 s, with no resource/nonfinite stop. Best fixed VAL
query joint NLL was 6.07104 at update 2000 versus 10.49416 at initialization
(42.15% lower); time NLL fell from 7.15130 to 4.38610 and event-row NLL from 3.56572
to 1.79727. The paired development learning gate passes. At update 2400, VAL joint
NLL rose to 6.46048, so the selected checkpoint remains update 2000, SHA
52191e0095f0efc8bc0bc0f3f87765f6606e78d38188cefe32b5d4054542829f.
The fixed TRAIN probe at update 2000 has joint NLL 4.73108, time NLL 3.29967 and
event-row NLL 1.56154. One seed, 48 VAL queries and an unblinded 12-song development
cohort give no uncertainty estimate or general quality claim. Outputs remain at
training/random-v1. Native generation on the same six-song TRAIN cohort started
at generation/random-v1-train with this pinned checkpoint, CPU with one thread,
the original seed 17 + index and a 900 s aggregate budget.

The artifact-only short-gap diagnostic replayed 32 time-spaced anchors per stream
for the memorized Imaginary Waltz case. Source and generated prefixes are
separately replayed from BOS; there is no source priming or knowledge of future
endpoints. It completed on CPU with one thread in 3.51 s. Mean
P(next event <= 20 ms) was .3059 after source events and .4479 after generated
events, with medians .0010/.0265. At generated absolute clocks, the actual source
prefix gave mean .4407, similar to generated. The contexts differ in history,
recency/occupancy and selected anchor distribution; these are not causal effects.
The finding rejects the assumption that the memorization model is well calibrated
on all human histories and fails only after generated feedback. Full provenance:
diagnostics/memorize-v1-firstcase-time32.json; script: short_gap_hazards.py.
The CPU diagnostic ran for 3.51 s during MPS training; native latency measurement
was already complete and was not taken during this overlap.

A separate source/sampler audit found a localized startup/rest coverage weakness.
The 4000 ms event-prefix branch cannot reach 39 of 130988 TRAIN target rows
(.0298%): 18 late first events and 21 later long gaps. In the actual 38400 random
draw stream, late-start arrangements receive 19 positive and 525 censored
BOS-history queries; six late-start arrangements receive no positive opening
target. Only 26 of 39 long-gap targets are observed positively, with 46 positive
exposures total. Overall BOS exposure is mostly positive, so aggregate diagnostics
hide this subset. The 48-query VAL probe includes no positive long-gap transition.
This is a coverage weakness, not a lack of model support or future-input leakage.
A next comparison should cover event-ending and preceding censored windows
explicitly. The current baseline was not modified. See
diagnostics/query-coverage-random-v1.{json,md}.

### Native revision 2 status and LN interpretation correction

Both six-song cohorts completed: TRAIN in 104.14 s and VAL in 114.52 s, using CPU
with one thread and the pinned best checkpoint at update 2000. All 12 outputs
passed exact mechanics/export/reparse. In Imaginary Waltz, the combined same-lane
attack/release <20 ms diagnostic fell from 272 to 8.37 per 1000 heads, and Lens's
frozen 4.6–4.9 s/8.5–10 s scopes no longer show the earlier repetition storms. The
new output also has a recognizable 83–100 ms repeated-key figure at 23991–24344 ms.
Fresh scopes still contain 3–8 ms repeated TAP relations. Scars now contains
independently overlapping multi-lane LNs. Full VAL review is pending; there is no
final quality acceptance.

The combined short-gap diagnostic conflates two distinct relationships. A Lens
review initially called two Scars 10/13 ms release → head gaps failures;
calibration showed same-lane head → head intervals of 189 ms, approximately a
half-beat at 165 BPM. This is compatible with LN-jack/hold-to-tap organization and
does not establish an impossible reset. The original report/hash is preserved;
the corrected report downgrades these to preference/pressure hypotheses. No
universal release-gap filter or negative training label follows. Sustained
extremely short TAP repetition is separate. Retrieved 11 Jack-with-LN and 40
LN-coordination-with-LN cards did not contain a directly matching <=30 ms reset
in their narrow scopes; that is missing calibration, not evidence against the
generated pattern.

Coverage audit wording correction: the 19980 ms late-start YOASOBI VAL song has
four later-event queries, at 190313, 31813, 112979 and 189838 ms; none has BOS
history or its first-event target. The earlier claim of no query for the song was
too broad. Corrected receipt SHA
610e1f9d95dcbb99cd81dd34a91169e0ece51aeb882fc3f1e877dbdf720ee1ff
links the prior SHA and actual freeze/evaluation-0 records. No training behavior
changed.

## Bounded scheduler probe: query-chunks-v1

Proposed exploratory probe under standing authority, acceptance none. Use the
same clean source 09b919c, checkpoint
52191e0095f0efc8bc0bc0f3f87765f6606e78d38188cefe32b5d4054542829f,
and canonical manifest
4b995029a5344569d4506ff6b11249f61585d2bf7649285754340909bb06c21b.
The completed first TRAIN case with 4 s queries is the baseline. Change only
`timing_horizon_ms` to 500 during native generation, with
`run_name=random-v1-chunk500 generation_split=train generation_cases=1 device=cpu
cpu_threads=1`, seed 17 and `max_seconds=300`. No source, training or model changes.
Use fresh output with no overwrite.

Hypothesis: on dense output, shorter absolute-bin queries avoid scoring thousands
of unused future hazards after every event. Physical history and absolute-bin
features stay identical; the exponential residual carries across empty chunks.
Primary guard: the exact persisted row sequence/hash and osu hash equal the
baseline. Measure total elapsed time, 8 s coverage and step P99 descriptively,
using CPU with one thread in a normal desktop environment, with the same
checkpoint/source. No dedicated benchmark or multi-seed claim. Failure of
equality requires tracing numerical/partition effects before using this as an
inference optimization. There is no quality-preserving claim if rows differ.
This tests a scheduler cost lever, not a learned-model intervention or speculative
decoding. The same guards apply: 30000 rows, 2 GiB RAM, 40 GiB disk and PAUSE.

### Scheduler probe result

The 500 ms query rerun on Imaginary Waltz produced byte-identical 4630 rows and
.osu output compared with 4000 ms queries. Rows SHA:
a1a4142a9688040bf8a9ce05cf4eb8d6a408c79678fa40a650111ee335fc793e;
osu SHA: 4b4450a0a60b8cf7a0301c6e33f51ec115ce38a1c382ee4e700aa18fb1110b87.
Generation time fell from 38.222 to 9.386 s, cached-Mel coverage of the first 8 s
from .586 to .276 s, and step P99 from 8.942 to 2.184 ms. Scored bins fell from
1,857,223 to 237,341; scheduler steps changed from 4633 to 4659. This is one paired
case on a desktop CPU with one thread, not a dedicated benchmark. It supports
reducing speculative hazard computation while preserving this exact draw; it
does not improve chart quality. Raw result: generation/random-v1-chunk500.

### VAL inspection scope

ThisFffire shows plausible early/middle pulse, chords and LN organization in
viewed windows; its demanding late window needs demand, listening and player
calibration. GR4VITY has coherent local roles but duplicate-like taps at 86910
and 86912 on column 0 around a source anchor at 86911. The report and traces, with
30 calls per bundle, are at lens-review/random-val-review.md and
random-val-review-identities.json. No whole-chart pass, audio listening or player
trial is claimed.

## Experiment Card: joint-mel-hazard-rows-v1 (revision 3)

Proposed, acceptance none; standing execution authority is unchanged. Revision 2
is complete. Baseline source: 09b919cdeab90e3856fee03a9198d59b4dc527af;
checkpoint: 52191e0095f0efc8bc0bc0f3f87765f6606e78d38188cefe32b5d4054542829f.
Use the same 133-chart canonical corpus/normalization, R1 initialization and model
architecture. The intervention source will be recorded after focused checks and
a clean commit.

Question: does complete supervision of selected waiting intervals repair the
late-opening/long-rest exposure hole without worsening common local prediction?
This is a data/objective-coverage correction, not a larger encoder. The current
event-prefix sampler only observes the first 4 s of a selected longer wait.

For a selected BOS/event-prefix target, partition its actual wait into disjoint
<=4 s censored chunks and one final event-containing chunk. Retain physical
history; the cursor advances without rows. Sum timing NLL across chunks and one
row NLL, then normalize per logical example, not per chunk. Preserve absolute-time
and outro conditional 4 s queries. One deterministic initial coverage pass inserts
each of 39 TRAIN long-gap targets once: one example per update, replacing one slot
after normal random draws so the remaining RNG stream stays matched. Estimated
additional chunk work from full waits is 5.32% before this 39-example pass.
Microbatch expanded queries at `batch_size` to keep peak memory bounded;
accumulate gradients and step once per logical batch. No endpoint, target-time or
crop-choice label becomes an input feature.

Add full-gap likelihood probes for the two VAL long transitions, scored separately
from the unchanged 48 common fixed queries. Read the baseline checkpoint on these
probes before training. Full-gap loss is summed waiting likelihood, not per-ms F1
or a claim that the source chooses the only acceptable first attack. Native
audio-only outputs remain the quality target; timing imitation alone cannot prove
playability.

Run name: coverage-v1; 2400 updates, batch size 16, `full_wait_supervision=true`,
`coverage_pass=true`, `train_groups=0`, validation every 400 updates, seed 230923,
two CPU threads, MPS and `max_seconds=1500`. Other revision 2 defaults are
unchanged; initialize again from R1, rather than resuming the more-trained random
checkpoint. Record both code and sampling changes, configuration, exact coverage
and group/query-normalization checks. Use fresh output with no overwrite/resume;
the same guards apply for 2 GiB RAM, 40 GiB disk, nonfinite values and PAUSE.

Gate: the pass gives complete positive coverage of all 39 selected rare
transitions; the best checkpoint on the common 48 VAL queries has joint NLL no
more than 10% worse than 6.07104, with both components finite; and full-gap NLL
averaged over the two cases is better than the pinned baseline, without either
case becoming nonfinite. These very small targeted probes are diagnostic, not
population estimates. If positive, use pinned-checkpoint native generation on
six TRAIN songs and 12 VAL songs, including the late-start song, with 500 ms
inference chunks after the exact-draw probe and 900 s per cohort. Report cold/cache
scopes explicitly. Review plausible cases, startup, same-key tap clusters and LN
coordination via Lens; do not collapse distinct release/head relationships into
one BAD label. The likelihood gate does not imply acceptance.

Expected failure: rare-transition supervision improves while ancestral duplicate
taps remain. That directs a separate diagnostic of event hazards versus
conditional row probabilities at actual witnesses before a new architecture or
quality-filter branch. A nonoverlapping chunk partition and normalization per
logical example are the regression guards: overlapping prefix/tail windows would
double-count survival and change the objective incorrectly. Required tests
compare partitioned and unsplit hazard loss/gradients, coverage of 39 targets,
BOS at 0, long-held state and unchanged common-query identities. Optional style
controls and new long-term memory stay deferred.

### Revision 3 implementation and pinned baseline probes

Clean intervention 608f092e6cd534638e8e47432bcb98973b79a5a4 changes only sampling,
logical-example objective accumulation, focused tests/config and scoped docs.
Model, state, audio encoder and generation are unchanged. Twenty focused tests
pass, including actual `backward_logical` gradients against a dense reference
with microbatch sizes 1/2/4, nonoverlapping survival/gradient identity, zero-time
BOS, endpoint hiding, exact legacy RNG selection and TRAIN/VAL separation. Hydra
flag projection was checked. Fixed-query memorization with
`full_wait_supervision` is rejected rather than silently disabling it. Resource
check granularity remains logical updates, so clock/PAUSE limits can overrun by
one update. The longest pinned source wait, 44.82 s, expands to at most 12 queries
before microbatching. Normal positive-wait queries now end at the target instead
of scoring unused future hazards; crop/full parity preserves scored features.

The new deterministic coverage builder finds 39 TRAIN examples comprising 126
queries and two VAL full-gap examples comprising seven queries. The baseline
random-v1 checkpoint was scored through the new evaluation-only code with pinned
bytes; no training occurred during this probe. Full-gap baseline mean joint NLL
is 17.12721604, time NLL 14.10210943 and row NLL 3.02510661.
Case a0fc0cee42c3: cursor 112439 → target 116472, two queries, joint NLL 17.51861751.
Case df6f2787a3fc: cursor -1 → target 19980, five queries, joint NLL 16.73581457.
Receipt: diagnostics/random-v1-full-gap-baseline.json. The revision 3 run starts
from R1 with the predeclared 2400 updates, batch size 16, seed 230923,
`max_seconds=1500 full_wait_supervision=true coverage_pass=true`. No rejected
release-gap claims are used as training labels.

### Local native attribution and rejected jitter-only branch

The pinned random-v1 diagnostic replayed native histories once on CPU with one
thread in 2.94 s, preserving their actual rows. At GR4VITY 86912, the next-time
mass is .1224, rank 3 of 4000, and P(next <= 5 ms) = .7327. The sampled same-key
row has probability .01158 and rank 4; other single-lane alternatives have
probabilities .4440/.2873/.2486. Waltz 23474 follows an intervening row at 23470:
the event gap is 4 ms and the same-lane head gap is 8 ms. Its time mass .1249,
rank 3, and 5 ms CDF .6123 contrast with sampled-row probability .0774, rank 3,
with alternatives .4884/.3980. Waltz 183148 is an earlier time tail, with mass
.01258 and rank 16, and a very-low-probability row, .00467 and rank 5. No repeated
row is the mode; all lanes are free and old release clocks are seconds away.
See diagnostics/random-v1-duplicate-tap-attribution.json. High probability of a
nearby event is not itself a quality error: different-lane flams or other
elaboration can be valid. The witnessed repeated TAP choices need separate
treatment from tight LN tail gaps.

A proposed noise-in-demonstration analogue, DART
(proceedings.mlr.press/v78/laskey17a.html), suggested testing whether an early
generated head leaves an acoustic cue looking unconsumed. The bounded test kept
a source row identity consumed and moved that same complete row 1/3/5 ms earlier,
preserving legal history and unchanged future labels. Re-querying the original
source by shifted timestamp would be wrong: it would label the already-consumed
row as future again.

The CPU probe completed with one thread in .266 s. GR4VITY clean-prefix 5 ms CDF
remained 4–6e-7 under 0/1/3/5 ms shifts, compared with native .733. Waltz
clean-prefix CDF remained approximately 1e-6 versus native .612; probability
through the original anchor + 5 ms also stayed negligible. The Waltz source row
at 183132 was excluded because it mixes a head and LN close. Thus jitter alone
does not explain the actual native failure, and no jitter-augmentation trial was
launched. REFINE broader history/composition attribution. Receipt:
diagnostics/random-v1-source-history-shift-probe.json, SHA
4a08e0efc042e7a657d6eba1e5c5be62f4008f6478530466c595e4725357b5d6;
script SHA c9337167b7f86f115bd1a0028597135911f7abc7eb06b54ae24cce93b72aad5d.

Coverage-v1 remains the independent predeclared training intervention. At updates
400/800/1200, its common VAL joint NLL is 6.4687/6.2884/6.3425 and its two-case
full-gap NLL is 15.5913/15.4028/18.3083. Nonmonotonic targeted results preclude
declaring a late-start improvement from one checkpoint; selection and gates remain
unchanged. All 39 coverage examples were consumed by update 39. There is no
architecture/filter change.

### Revision 3 completion

Coverage-v1 completed 2400 updates in 1118.44 s with 38400 logical examples and
40554 physical queries (+5.61%), all 39 coverage examples consumed, and no
resource/nonfinite stop. Update 2000 was selected by the unchanged common VAL
joint NLL criterion: 6.09073456 versus baseline 6.07104107 (+0.32%). At that
checkpoint, targeted full-gap mean NLL was 14.62766263 versus baseline 17.12721604
(−14.59%); glacia changed from 17.51862 to 16.89078 and YOASOBI from 16.73581 to
12.36455. In YOASOBI, time NLL improved from 11.73759 to 6.24381 while row NLL
worsened from 4.99822 to 6.12074: timing improves without a parallel mark
improvement. These are two development cases, not statistical proof. Both
predeclared learning guards pass. Best SHA:
85f643077d127f9fe3e5256dc7b88512912d9ce8be39d6dbe164ef3ef4c9327e;
last SHA: 8edd5b723e5bb25673ffe53102e66fb6db2d6cb56ce4bfc4d56a38af6e2a61fd.
Native generation uses six TRAIN and 12 VAL songs, 500 ms queries, CPU with one
thread, seed 17 + index and 900 s per cohort; TRAIN launched at
generation/coverage-v1-train. This is the same 3M model, with no memory or encoder
expansion.

### History/exact-state sensitivity diagnostic

Artifact diagnostics/random-v1-history-exact-interventions.json records a 2.60 s
read-only analysis on CPU with one thread. At fixed audio/absolute clocks, with
native exact state fixed, replacing native encoded history with source history
lowers 5 ms CDF from .7327/.6123 to 1.74e-7/1.70e-7 for GR4/Waltz. These hybrid
inputs are nonphysical network interventions, not causally valid chart changes.
The GR4 source/source cell is properly high (.9451): cursor 86910 comes 299 ms
after its last row at 86611 and 1 ms before its existing next source event at
86911. That is not another failure.

Actual source/native event rates over the last 16 events are 3.57/20.16 Hz for
GR4 and 3.40/28.68 Hz for Waltz. Native 64-row histories are TAP-only; source
windows include LN starts/releases. The network responds strongly to these
different prefix organizations, not just jitter. This does not set a desired
difficulty, justify forcing native density to match the source, or label all
nearby different-lane events BAD. It narrows follow-up toward generated-history
state/composition and conditional row-tail choices. New memory, hard gap filters
and blind jitter training remain unselected.


### Complete native cohort and one positive prototype

Coverage-v1 generated six TRAIN charts in 28.608 s and all 12 VAL charts in
60.919 s on CPU with one thread and 500 ms queries. All 18 completed exact
mechanics and export/reparse. Cached-Mel first-8-second coverage ranged from
0.103 to 0.352 s; per-chart step P99 ranged from 2.395 to 3.063 ms. These are
research-generation timings, not client/network deadlines.

Lens still found duplicate-like TAP burdens in some samples. The data-only
coverage correction therefore does not establish reliable playability. Its
Scars sample has fewer LNs than random-v1; one stochastic sample does not prove
lost representational capability. See lens-review/coverage-review.md.

The exact YOASOBI Ano Yume o Nazotte output is a positive prototype candidate:
896 action events, 862 attack rows, 967 heads and 301 LNs. Lens read every action
and articulation page (14 pages per view), viewed systematic 2.5-second windows
every 25 seconds and targeted burst, long-hold, overlap, gap-return and ending
windows. No continuous all-pixels claim is made. It found coherent motion,
restrained chords, LN chains and independent held/released roles, with no
confirmed duplicate-like TAP burden in the complete event sequence. The root
also viewed the independent-LN and overlapping-obligation renders.

This supports handing off that chart for prototype playtesting, not accepting
the generator as reliable. Musical fit of the 7.975-second entry, the 4.841-second
no-head interval at 100.314–105.155, and repeated LN chains requires listening
and player feedback. Source entry at 19.980 seconds is one authored choice, not
the unique audio-only answer. No human playtest or audio-listening verdict has
been received. Report: lens-review/yoasobi-prototype-review.md; trace and bundle
pins: yoasobi-prototype-review-identity.json. Generated chart SHA:
cf9ff8c22d24ae4a805f768becd614c62ba3c7d9bba6a50b494e3580ec03758f.

### Exported weights and source-free inference

The inference-only checkpoint retains identical model tensors and normalization
but omits optimizer/RNG state. It is 11,862,079 bytes, SHA
29237d5bf25ed40fe1db4a8e022280ee834eae29521d62e3666462c110a71f49,
at delivery/joint-mel-r1-prototype-v1/model.pt. The adjacent README and
identity.json record provenance and usage. The exact reviewed chart and audio
are packaged in YOASOBI-Ano-Yume-o-Nazotte-joint-prototype.osz, SHA
12527120b2b6968774d8c220bec0239f79490837de3f087e841b3b52046f1150.
The archive preserves both file bytes; its source presentation header has
180 BPM, constant SV and OD8. Presentation metadata did not condition generation.

A fresh-Python-process profile on the same sample regenerated canonical Mel and
all rows byte-identically. Measured imports 0.689 s, setup/model/pins 0.081 s,
decode 0.403 s, Mel 0.110 s; first 8-second coverage 1.441 s, first 31 heads
1.515 s, full 242.666-second song 3.380 s. The first-head threshold includes a
two-head row at 20.805 seconds. OS disk caches were warm; interpreter/bootstrap
before the script, client and network are excluded. Receipt:
profiles/coverage-v1-yoasobi-cold.json. This is one sample, not a latency guarantee.

Product commit 78222bc803069afbfac59107cb8f447a2db533e6 adds infer_audio mode,
using only a pinned checkpoint and an audio file. It does not load a corpus,
source chart, seed or BPM. Fresh outputs contain canonical Mel, source/model
identity, step/startup profiles, audio and verified .osu. Source-free export
uses a declared 120 BPM editor/scroll placeholder, not an inferred beat grid.
The source-based reviewed archive remains unchanged. The commit also fixes the
source-free save_rollout case that copied audio without an AudioFilename header.
Twenty focused inference/config/generation tests passed, including a real short
WAV with no corpus or source owner, dispatcher projection, exact replay/export,
resource caps and native CPU/MPS partition behavior.

The real CLI was then run with the stripped checkpoint, seed 26, a fresh root
artifacts/joint-audio/standalone-proof and only the copied MP3. It reproduced all
896 row bytes exactly without a manifest in that root. Its exported 967 objects
also passed the strict Lens preparation bridge, which asserts zero diagnostics.
Receipts: standalone-proof/inference/yoasobi-v1/result.json and
profiles/source-free-lens-admission.json. The CLI profile starts after imports;
its 0.634-second first-8-second value is not interchangeable with the fresh-process
1.441-second value above.

Code and the owning proposed Note are locally committed; no remote push, new
release tag or canonical annotation edit occurred. The optional user playtest
question links the concrete archive and asks for time-local musical/physical
feedback. Work did not wait for that answer. The ultimate reliability goal
remains active: existing negative samples still require targeted correction;
new long-term musical memory and explicit style/difficulty controls are deferred.


## Head spacing audit and marked-process recovery branch

The previous goal turn made progress: commits, trained checkpoints, a source-free
inference path and native/Lens evidence changed the available baseline. The
current continuation revalidated clean product HEAD
78222bc803069afbfac59107cb8f447a2db533e6 and committed Note HEAD 047a6d5.
No training or generation process remained live at entry.

An action-specific audit separates consecutive same-lane head types from the
last release clock. The 121 TRAIN charts contain 179,541 heads and no TAP→TAP
interval at or below 20 ms. The 18 coverage-v1 outputs contain 52,265 heads,
157 TAP→TAP intervals at or below 20 ms, including 42 at or below 10 ms.
These counts locate a distribution mismatch; they do not assign a universal
human BAD threshold. Receipt: diagnostics/head-interval-audit-v1.json.

A deterministic broader TRAIN-only sample of 256 song groups, up to two charts
per group, admitted all 440 selected charts and 691,698 heads. It contains no
same-lane consecutive heads at or below 40 ms. This prompted a full audit of the
pinned catalog's TRAIN partition, without opening validation or TEST sources.
All 11,564 TRAIN charts passed their source/cache identity and byte checks,
covering 16,078,013 heads. None has consecutive same-lane heads at or below
20 ms. The minimum is 27 ms for TAP→TAP in Laur — A Lasting Promise [HS];
TAP→LN minimum is 30 ms, LN→TAP and LN→LN minima are 64 ms. There are 130
TAP→TAP and three TAP→LN intervals at or below 40 ms. This audit completed in
10.10 s with no exclusions. Receipt and exact selection/pins:
diagnostics/all-train-head-interval-audit-v1.json. It describes this corpus,
not a universal motor limit, and includes no playability labels.

A useful analogue is history-dependent point-process recovery: stimulus drive
and recent-event feedback can be modeled separately. Gerhard, Deger and Truccolo
(2017), https://doi.org/10.1371/journal.pcbi.1005390, demonstrate that fitted
point-process models may sample unrealistically high rates despite ordinary
fit checks; an absolute refractory bound alone may merely produce saturation
at its boundary. Their GLM/neuronal assumptions and stability theorem do not
transfer to our finite TCN or beatmap quality. The transferable warning is to
inspect native dynamics and boundary pileup, not just add a cutoff.
Paninski's cascade point-process work (2004),
https://www.cns.nyu.edu/~lcv/pubs/makeAbs.php?loc=Paninski04b,
provides the stimulus/history modeling analogy. Mark-dependent thinning is the
closest implementation primitive; our discrete probability calculation is
stated below rather than importing a Poisson assumption.

### Experiment Card: marked-head-spacing-v1

Revision: 1. Proposed; acceptance none. Execution uses the standing research and
local-experiment authority. Owning Note and lifecycle are unchanged. This is one
bounded decoder-prior probe, not an adopted architecture or a new model-size run.

Baseline: product source 78222bc803069afbfac59107cb8f447a2db533e6; frozen
coverage-v1 best checkpoint
85f643077d127f9fe3e5256dc7b88512912d9ce8be39d6dbe164ef3ef4c9327e;
canonical corpus manifest
4b995029a5344569d4506ff6b11249f61585d2bf7649285754340909bb06c21b.
Native baseline is the completed six-TRAIN/twelve-VAL cohort, seed 17 + case index,
500 ms queries, CPU with one thread. The observed 42 TAP→TAP intervals <=10 ms
across 52,265 generated heads are a diagnostic baseline, not the whole quality
metric. The reviewed YOASOBI positive sample is a specific regression guard.

Hypothesis: a direct same-key head-history factor can suppress unsupported
instant rearticulation while preserving close different-key events, all observed
TRAIN head spacings, and LN tail semantics. Use one scale tau=27 ms, derived from
the minimum observed TRAIN same-key head spacing. For a proposed row m at t,
let d_j be time since the last head on each newly pressed lane. Set

    a(m,t) = product_j min(1, (d_j / tau)^4).

A lane with no previous head contributes one. Only TAP/LN_START contribute;
CLOSE and EMPTY do not. This reads head-to-head time, never tail-to-head reset.
The exponent four is a fixed smooth-ramp hypothesis, not a fitted physiological
constant. All observed TRAIN event rows have acceptance exactly one at their
actual source prefixes. Every positive native head interval retains positive
mathematical support; there is no global event-spacing or beat-lattice rule.
Ordinary jacks, short LNs, asynchronous releases and fast distinct-lane figures
remain representable. This does not prove that every suppressed novel interval
would be undesirable.

For base hazard h_t and conditional complete-row distribution q_t(m), define

    P(new event m at t | actual history) = h_t q_t(m) a(m,t)
    P(no accepted event at t | actual history) = 1 - h_t sum_m q_t(m) a(m,t).

Implement by sampling a base proposal and accepting it with probability a.
Rejected proposals advance fixed-through time but never enter physical replay
or learned history. Draw a fresh base waiting threshold after rejection; retain
ordinary residual survival across empty scheduler chunks. Use an independent
acceptance RNG so accepted-only trajectories preserve the baseline RNG stream.
At the true terminal with open holds, force closure and normalize legal rows
with the factor; do not reject a required terminal close into an open-ended map.
Scale zero must be byte-equivalent to the baseline path. Proposal count is bounded
as well as committed rows, and capped runs retain partial evidence without
inventing endpoints.

This is a changed generative distribution, not neutral postprocessing. It uses
additional TRAIN-only spacing evidence while keeping all network weights fixed.
A row-only reranker cannot solve the case where every lane has just been pressed;
a global event cutoff would erase distinct-lane flams. The marked factor instead
changes time and lane choice jointly. It is a simple candidate to reject or
retain through evidence, not a claim that it solves all playability.

Implementation owner: joint_audio_continuation/head_spacing.py, optional typed
config and generator integration, focused probability/state tests and scoped
documentation. Default remains disabled. Required checks: product probability,
positive support, exact zero-scale identity, LN close/near-tail head separation,
no rejected-row history mutation, forced terminal closure, and query partition
invariance. Record a clean intervention commit before native runs.

Run head_spacing_ms=27 against the unchanged six TRAIN cases and twelve VAL
cases, same checkpoint/seeds/500 ms query settings, fresh generation/spacing-v1-*
outputs, at most 900 s per cohort. No training, parameter expansion or network
access. Existing 2 GiB available-RAM/40 GiB disk/PAUSE guards remain. One CPU
thread; model and audio preparation scopes stay explicit. Count proposals and
rejections, timing overhead, head/type rates, and head-interval histograms.

Decision gates: reduce <=10 ms consecutive TAP→TAP relations by at least 90%
without relying on source-density matching; no mechanical/export/Lens admission
failure; no persistent new pileup at the 27 ms knee; reviewed YOASOBI candidate
must remain byte-identical if its proposals all have >=27 ms same-key head ages.
Record all deviations, including composition/rate changes. Inspect complete
local episodes in the previously failing TRAIN and VAL scopes plus fresh
pressure sites, and positive ordinary Jack/LN/Tech-like figures, through Lens.
Do not call an improvement merely because counts fall. If rate saturation,
loss of organization, or little benefit occurs, reject the prior and investigate
history/demand modeling or training; do not tune a threshold grid until it passes.


### Result: marked-head-spacing-v1

The clean intervention was a5abc256a964521596b108248818a27c62b80ecf. It adds only
an optional decoder prior, typed configuration, execution accounting and focused
tests; all network weights and the canonical audio frontend remain unchanged.
Scale zero preserves the original row probability/RNG path. Tests cover active
CPU/MPS terminal normalization and partition invariance, head-vs-release clocks,
positive mathematical support, rejected-row exclusion from exact and learned
history, and proposal limits. Nine head-spacing tests pass; existing native
sampling, source-free inference and configuration tests also pass in the scoped
runs. A later tests/documentation commit records this evidence without changing
the generated run's implementation.

Both frozen cohorts completed: six TRAIN cases in 27.649 s and twelve VAL cases
in 59.489 s on CPU with one thread. All 18 pass exact replay, export/reparse and
the strict Lens bridge. There are 188 rejected proposals out of 40,889. Consecutive
same-lane TAP intervals <=10 ms fall from 42 to zero; <=20 ms fall from 157 to10.
Total heads are 50,901 versus 52,265. Per-chart head ratios have median0.9866 and
range0.8734–1.0368; this is not a uniform density reduction. No four-interval
same-lane run in the24–30 ms band was found. That selector tests an obvious form
of saturation, not global stability or a quality label. YOASOBI and glacia row
bytes remain identical to the baseline; the reviewed YOASOBI `.osu` is also
unchanged. Per-chart step P99 is2.138–2.183 ms in this desktop probe.

Eight scoped generated/source action sequences were read through all pages.
All11 requested generated image pages were viewed, including old and fresh
pressure scopes, Fffire's chord repetition and its LN-rich ending. The source
render pages were saved but not viewed in this pass; source actions/times and
complete endpoints were read. The TRAIN/VAL bundles have26/28 traced calls and
155/167 frozen files respectively, all byte-verified after inspection. Report:
lens-review/spacing-prior-review.md; identities:
lens-review/spacing-prior-review-identity.json; numerical comparison:
diagnostics/spacing-comparison-v1.json.

The former severe witnesses are absent from the new inspected scopes. Scars
now has a held column0 under81/89 ms column1 repetitions in the old four-key
chord-to-8ms-repeat scope. GR4VITY preserves separate54/64 ms LNs, release-only
rows and different-column heads two milliseconds apart; its ordinary72/83 ms
same-key figure survives. Fffire retains recurring two-/three-note chords near
200 ms and an accelerating chord passage. Its ending has staggered LN roles,
including a39 ms hold. The unchanged YOASOBI candidate provides a complete
positive-sequence regression guard, not a new independent success sample.

Limits remain visible: Waltz's opening retains15/21 ms same-key pairs, its fresh
scope has a19 ms repeat, and a fresh GR4VITY scope has another19 ms repeat.
These are local pressure questions, distinct from short LN tail gaps. No complete
musical-fit or player-demand judgment is available for the changed charts.
The predeclared diagnostic, mechanics and specific preservation guards pass,
but the full playability objective is not complete.

Decision: retain the prior as an explicit experimental option, still disabled
by default. Do not tune an exponent grid until the current witnesses disappear
or label every sub-threshold interval universally BAD. The next research branch
should investigate learning the corrected joint local distribution on native
histories while retaining source likelihood and expressive-pattern guards;
original source futures cannot simply be attached to altered generated prefixes.
The positive template supports local calibration, not a larger encoder, a new
long-term memory module or a claim of universal stability. No remote publication
or canonical annotation changes occurred.


## Experiment Card: native-joint-distillation-v1

Revision: 1. Proposed; acceptance none. Execution uses the standing local research,
implementation and experiment authority. No Note lifecycle transition is implied.

Question: can the existing shared Mel/history network learn a corrected joint
next-time/row law on its own histories, without an inference prior or added
parameters? The selected hypothesis is an exposure/objective gap, not insufficient
audio information. A frozen teacher is coverage-v1 plus the already inspected
27 ms marked head factor. This teacher carries a local corpus-informed preference,
not new human playability labels or unique musical ground truth.

The closest primitive is soft-target distillation (Hinton et al., 2015,
https://arxiv.org/abs/1503.02531). Here the target is the complete finite-window
next-event distribution, including survival/censoring, rather than a class label.
The point-process fit-versus-free-run warning from Gerhard et al. (2017), cited
above, motivates native-history evaluation; its GLM theorem does not transfer.
Naive jitter augmentation is not selected because the preceding controlled probe
failed to reproduce the native short-gap behavior. Larger encoders and new
long-term memory remain deferred.

Baseline source is clean fa64def92e1a4b57d0ac47e1792b5774c6390f3d. Checkpoint,
manifest and the six TRAIN/twelve VAL native cohorts retain the exact identities
in marked-head-spacing-v1. Baseline raw sampling has 42 TAP-to-TAP relations at
or below 10 ms, 157 at or below 20 ms, among 52,265 heads. The fixed 48-query
validation joint NLL is 6.09073456; these development observations are unblinded.

For a fixed actual generated prefix, enumerate the next 80 native milliseconds.
At each time t let base hazard be h_t, row distribution q_t, and the existing
head factor a_t(m). The corrected teacher has H_t=h_t sum_m q_t(m)a_t(m) and
Q_t(m)=q_t(m)a_t(m)/sum_m q_t(m)a_t(m). Its next-event mass is
S_t H_t Q_t(m), where S_t is survival through earlier milliseconds; the remaining
survival at the window end is a censor outcome. Minimize KL from this complete
distribution to the student's distribution. Do not omit censor mass, turn a
rejected proposal into a history row, or attach an original chart's future to a
changed generated prefix. True terminal closure remains forced; ordinary window
ends remain censored. Close factors are one and head factors use preceding heads,
not release clocks. Teacher tensors are detached.

Two paired continuations start from identical coverage-v1 weights with fresh
AdamW state: source-only control and source plus 32 times native-window KL.
Both use exactly the same source logical examples and 600 updates, batch 16,
full-wait supervision and the existing 39-transition coverage pass. Inherited
learning rate is 1e-5, new-module rate 1e-4, weight decay .01, gradient norm cap 1.
The coefficient 32 is fixed for this bounded test, not selected on validation or
native quality. The final update is the evaluated endpoint; no best-of-checkpoint
native selection. A small gradient/throughput preflight must establish finite
losses, actual gradients and feasible resource use before either main arm runs.

Native contexts come only from coverage-v1-train's first five chart identities.
The sixth, Scars Of FAUNA, is withheld from native supervision as a transfer
probe; its real source remains in ordinary TRAIN. Sample 640 fixed contexts,
half uniformly by song then event prefix, half from prefixes immediately before
an observed generated same-lane head interval below 27 ms. Draw with replacement
if this pressure pool is small; record unique counts and do not misrepresent
640 as independent failure examples. For the sixth song prepare 128 analogous
held-out contexts, reporting uniform and pressure strata separately. Source RNG
230924 and native RNG 230925 are independent. Native minibatch is four contexts
per update, with no paired source-future labels. Pin generated chart bytes and
admit them under new source/arrangement identities while preserving TRAIN audio.

Implementation adds a narrowly scoped native-window distribution/KL owner and
probability, causal-input, terminal and CPU/MPS tests. A run-local artifact runner
owns this exploratory training recipe; it must pin its own bytes, source OID,
inputs, context selections, targets and both run configurations. No packaged
inference or architecture defaults change. Commit implementation before main runs.

Primary learning gate: held-out native-window KL falls at least 50% relative to
the starting model, and below the source-only continuation, without validation
joint NLL worsening more than 5% relative to either the start or paired control.
This local gate does not by itself establish playability. Native gate: without
head_spacing, reduce <=10 ms TAP-to-TAP count by at least 75% relative to the raw
baseline and beat the paired source-only arm; all 18 cases complete mechanical,
export/reparse and strict Lens admission. Median per-chart head-count ratio to
baseline must remain in [.85,1.15], and total LN heads in [.7,1.3] of baseline.
These are regression alarms, not desired source-density targets. Inspect old
negative scopes, fresh pressure maxima, and positive LN/Jack/irregular figures
using Lens; broad thinning, rigid boundary pileup or loss of independent releases
fails the qualitative gate. Review the previously positive YOASOBI candidate as
a changed sample, not a byte-identity guard after weight updates.

Run on this Mac, Torch 2.11/MPS for learning, one CPU thread for native generation,
no network/data download. Per training arm at most 1,200 seconds, preflight and
teacher preparation together at most 600 seconds, native cohorts at most 900
seconds each. Preserve the 2 GiB available RAM, 40 GiB free disk and root PAUSE
file guards; stop on nonfinite gradients, illegal replay or a resource guard.
Fresh outputs live under artifacts/joint-audio/20260923-v1/training/native-distill-v1
and generation/native-distill-v1-{control,student}-{train,val}. No overwrite or
implicit resume. The exact artifact runner command and SHA must be recorded
before execution; a preflight failure is evidence and does not authorize silently
changing the intervention.

A positive result would retain the small architecture and motivate broader native
coverage. A negative result would reject this fixed distillation recipe, not
prove that Mel is insufficient or that a larger model is needed. Main confounders
are the locally designed teacher, only five native-supervision songs, repeated
pressure contexts, changed trajectories under paired random seeds, and previously
inspected validation songs. No conclusion substitutes for musical listening or
player feedback. More fitting of this teacher cannot independently discover
musical arrangement preferences absent from it.


### Native distillation preflight and Card revision 2

Implementation commit 3a5cea9a2739a8658905a155c612817f2d1d0fc2 adds only the
native-window distribution/KL owner and its tests. Twenty-two selected tests
pass, including six new CPU/MPS tests covering normalized marked/censor mass,
forced terminal closure, no future-row/LN-endpoint input leak, and gradients
through shared audio, history, timing and row modules. Architecture, parameter
count and inference remain unchanged.

Revision 1 preflight ran with artifact diagnostics/native_distill_v1.py, SHA
9c5ae38298049b71556b76d9b0a548b5609c312e70fda30a355dbc33b891542d,
command `uv run --extra mps python artifacts/joint-audio/20260923-v1/diagnostics/native_distill_v1.py prepare`.
It completed in 11.77 s and prepared normalized teacher targets. Source batch
joint NLL was 4.92971; source gradient norm 15.8977. On the four selected native
pressure contexts, KL was .220887 and the coefficient-32 gradient norm 408.001.
Peak observed driver allocation was 2.46 GB and available RAM 7.66 GB. Receipt:
training/native-distill-v1/preflight.json; targets SHA
da2e688dd7fdcb8cd5540506470495370446f0267b95cfb4e5baf4fc0b6cf2f4.
No optimizer update or main arm ran under revision 1.

This preflight changes a protected field before the paired experiment: Card
native-joint-distillation-v1 revision 2 uses native coefficient **1**, not 32.
The unweighted pressure-gradient norm is 12.7500, comparable to the source batch;
this avoids an initially 25.66-fold dominant correction gradient. This is a
TRAIN-gradient calibration, not validation or sample-quality tuning. Other
hypotheses, seeds, context selections, learning rates, update counts, metrics,
guards and resource bounds remain identical to revision 1. No coefficient sweep
is authorized by this amendment.

Fresh outputs change to training/native-distill-v2 and
 generation/native-distill-v2-{control,student}-{train,val}; revision 1 artifacts
remain intact. The revision 2 runner is diagnostics/native_distill_v2.py; pin its
printed SHA in the next preflight freeze. Commands are
`uv run --extra mps python artifacts/joint-audio/20260923-v1/diagnostics/native_distill_v2.py prepare`,
then the same command with `control` and `student`, serially. Both main arms
require a successful frozen preflight, exact runner bytes and a clean pinned
implementation commit. The Card and owning Note remain proposed, acceptance none.


### Result Log: native-joint-distillation-v1 revision 2

Card and Note remain proposed; acceptance none. Execution used standing local
research authority. Clean implementation source was
3a5cea9a2739a8658905a155c612817f2d1d0fc2. Revision 2 runner
`diagnostics/native_distill_v2.py` SHA
e942c57a36a94b6fc619649327762d0e2a1dfcb3d5525a114e4cb6729303933f
ran with `uv run --extra mps python` followed by its path and respectively
`prepare`, `control`, `student`. Its freeze precedes both main runs. The second
preflight passed in 11.08 s, source gradient norm 15.8977 and weighted native norm
12.7500; target bytes equal revision 1's frozen targets. No main run used the
superseded coefficient 32.

The paired arms completed all 600 updates in 246.899 and 303.218 seconds. Their
source-exposure hashes are identical:
c8c752c34029d9ddc2b3878790395b4e1ef8cae2b7733c26654068c0866be85e.
Each received 9600 logical source examples with full waits and all 39 coverage
examples. Final checkpoints, selected by the predeclared update rather than
native quality, are:

- control SHA 98c235199db7f6aba9482db65fe7df38d3596cb64e56d1276b4ced5226c2779a;
- student SHA 174cb1321ececa709660fc4ede4863b36285e62350f02e8fbe9663999625ddbf.

The TRAIN native pressure pool has 50/30/2/5/5 distinct prefixes in the five
training songs and seven in the withheld sixth song. Sampling yields 390 unique
training prefixes among 640 examples and 70 unique held-out prefixes among 128.
All six generated source/row digests also match their completed baseline run
receipts. New source/arrangement identities were admitted for these histories;
original chart futures did not label them. Driver allocation at final logging
was 3.45/3.46 GB, available RAM 5.98/6.32 GB. No resource or numerical guard fired.

| Metric | Starting coverage model | Source-only control | Native correction |
| --- | ---: | ---: | ---: |
| Fixed validation joint NLL | 6.090735 | 6.448922 | 6.436366 |
| Withheld native-window KL | .052978 | .201154 | .102891 |
| TRAIN pressure KL | .116273 | .256244 | .049185 |
| TRAIN uniform KL | .004125 | .100203 | .037188 |
| TAP-to-TAP <=10 ms count | 42 | 11 | 2 |
| TAP-to-TAP <=20 ms count | 157 | 19 | 13 |
| Total generated heads | 52265 | 28745 | 30470 |
| Median chart head ratio | 1 | .642161 | .687861 |
| Total LN heads | 1839 | 3202 | 4091 |

The student learns some TRAIN pressure correction but fails the withheld 50%
KL reduction and the 5% source-NLL guard. Its extreme-TAP count improves by95.2%,
but both the [.85,1.15] median head-ratio and [.7,1.3] LN-total guards fail. The
source-only arm also changes composition strongly. The positive count result
cannot be interpreted as an overall playability improvement or solely attributed
to the corrective preference: substantial continuation-training drift is present.

All four native cohorts completed with head_spacing_ms=0. The control TRAIN/VAL
runs took 19.698/31.126 s and the student runs 20.336/33.015 s. All36 charts pass
exact replay, export/reparse and strict Lens admission. Numerical comparison is
`diagnostics/native-distill-v2-comparison.json`, SHA
f41e873a4fb92c6b08d7176dcf7e6ce32ce05ea69d539c5da41b99c8e3dea3f9.
An additional TRAIN-fit diagnostic separates local learning from transfer and
is not a replacement metric: `native-distill-v2-training-kl.json`, SHA
cb009bf9d196793699b64bcc915cfe1c8f80f6244a404e7f6221f3f3c783f2e4.
The comparison writer initially failed to serialize a NumPy integer; converting
that descriptive count to a Python integer fixed reporting without changing
models, input/output charts or metric definitions. No failed run was hidden.

Lens review reads complete generated/source action pages for eight scopes and
views all16 generated render pages. There are22 student TRAIN and43 student VAL
harness calls; all four bundles remain byte-verified. Source render pages and
control charts were not visually reviewed; saved articulation output was not
read exhaustively. Exact coverage, hashes and judgments are in
`lens-review/native-distill-v2-review.md` (SHA
6aeb874d96e632e34a025f1789b49ed7b5c0ba8c17a9bde29ec10e7cc04a19d7)
and its identity JSON (SHA
fca8e42a26f8eed3ec8b076fd229b183242ecf2cc342b4a629387bf02a26e5bb).

The student retains a 9 ms same-column TAP pair at144844/144853 in Waltz and a
3 ms pair at230460/230463 in GR4VITY. The preceding 24 ms LN-tail gap in Waltz
is a different relation and was not labeled BAD. GR4VITY's distinct-lane one-ms
chord completion remains representable. Fffire retains repeated two-key groups,
accelerating movement and independent LN handoffs; YOASOBI's inspected three-second
passage has varied short LN chains and separate releases. Scars' old LN-rich
source scope instead becomes a legible tap/chord figure, which cannot establish
LN-coordination preservation. No source-density match or single gap threshold is
used as the quality definition.

### Additional diagnosis: silent-tail recovery

The strongest new regression is SCREW: both new models generate only eight rows
and ten TAP heads. The student's entire note range is7046–7820 ms, despite
completed coverage through the real121033 ms audio end. This is not a resource
cap or unresolved hold. The complete Lens action sequence and all four render
pages through its last note were inspected; the complete generator trace, not
the chart-range render, verifies the empty remaining audio interval.

A read-only, fixed-prefix network probe uses those exact eight student rows,
with no future commits, under all three checkpoints. From7821 to121033 ms,
integrated hazards are7.435406/5.094686/5.016541 for start/control/student, giving
no-further-event probabilities .000590/.006129/.006627. These are small
unconditional probabilities, not a claim that the model must always stop there.
After already surviving the first four seconds, conditional no-return probability
rises to .512111/.716767/.753637. This is one unblinded development history.
Receipt: `diagnostics/native-distill-v2-silence-probe.json`.

A separate instrumented native replay cloned the sampling RNG only for tracing.
The student's ninth waiting threshold is6.3415549965 and its remaining threshold
at true audio end is1.3250162417. Observed consumed hazard5.0165387548 agrees with
independent integration within2.3e-6. Four-second queries reproduce exactly the
eight rows from500 ms queries. Receipt:
`diagnostics/native-distill-v2-silence-sampler-probe.json`. This rules out a dropped
scheduler event for this case and locates the silence in the learned distribution
plus a rare waiting draw. A short80 ms correction window cannot distinguish
reasonable initial waiting from poor later recovery.

Evaluation recommendation: DROP this fixed distillation recipe as a candidate
upgrade; retain the tested objective and negative evidence. This is a research
recommendation, not a Note lifecycle transition. The current delivered prototype
and optional decoder prior stay unchanged. No coefficient grid, larger encoder,
long-term memory expansion, or new default follows from this result.

The next research question must include activation and return after rests, along
with preservation of arrangement composition. Live explanations include native
state/survival coverage and historical features suppressing exogenous audio drive;
this comparison does not distinguish them. A global arrangement-intent variable
is another possible source of composition consistency, but is not established
by this result and no new style labels are assumed. Follow-up should separate
these mechanisms using fixed-history probes before another larger training run.
The canonical Mel plus simple encoder and joint timing/row direction remains.
Playability is unfinished; musical listening and player feedback remain absent.

Implementation and curated evidence were locally committed. The latter adds a
self-contained objective, paired result and silent-tail finding to
`docs/research/audio_conditioned_choreography.md`. No remote push, new release,
or human-annotation modification occurred.


## Recovery conditioning probe

The preceding goal turn made progress: paired checkpoints, native/Lens evidence
and an independently verified silent-tail mechanism changed the next research
action. Product HEAD16209eaf9000867de72ba9c5f04989e1b1e990f6 and Note HEAD94ef006
were revalidated clean. No training job remains live.

A read-only exploratory probe will distinguish sensitivity to elapsed clocks
from sensitivity to encoded action history, before choosing a new architecture.
Use the pinned coverage-v1 checkpoint and its existing six TRAIN/twelve VAL native
charts. Select up to eight unique prefixes per chart near uniform audio-time
positions, requiring at least30 heads, no occupied lanes and12 seconds remaining.
No chart futures label these prefixes. Forecast the next12 seconds without
committing hypothetical rows, reporting survival through4 seconds and conditional
survival over the next8 seconds separately. These are conditional-risk probes,
not prevalence estimates of naturally occurring4-second gaps.

Over the latter8 seconds, compare true elapsed clocks to clocks after translating
the complete observed prefix forward by3800 ms. At the first scored time this
changes the last-event age from4001 to201 ms. Translation preserves occupancy,
action order and every content-token gap, while changing relation to the audio;
verify the content equality. Compare each clock condition with the actual encoded
history and with the learned BOS history vector. BOS-history/non-BOS-exact pairs
are explicitly nonphysical network interventions, never a proposed renderer reset.
The actual and translated full histories are valid alternative physical prefixes,
but translation carries no human musical-quality label.

Add the already named eight-row SCREW failure as a separate diagnostic, without
mixing it into the baseline cohort aggregate. Use identical future Mel encodings
across interventions. Record per-case values, unique-prefix counts, pins, runtime
and exclusions. No optimization, model-size change or inference-default change.
Bound execution to600 seconds, CPU one thread,2 GiB available RAM and40 GiB free
disk. Stop on source/hash/causal-state mismatch or nonfinite scores. Persist fresh
artifacts under diagnostics/recovery-conditioning-v1. This probe is exploratory,
acceptance none; it tests a mechanism rather than a quality gate.
