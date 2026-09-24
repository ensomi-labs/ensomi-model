# Agent Note: Audio skeleton generation and R1 integration

Note ID: 2026-09-23-audio-skeleton-r1-integration
Status: proposed
Kind: research
Created: 2026-09-23
Updated: 2026-09-24
Product revision: 7c316e6ea3d22aff1fd4761798f1b4aa41d47e89
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

The current integration contract is
`docs/research/audio_skeleton_information_contract.md`: direct full-audio input
to skeleton and rows, separate histories with minimal committed LN feedback,
and explicit candidate-action consequence evaluation. Its head-plan/release-clock
factorization is implemented as a research prototype, with playability still
under evaluation. The earlier flat joint architecture and completed fits are
diagnostic baselines. Earlier alternatives and Cards below
retain their original scope; the persistent-intent fit and extension of the
incomplete lineage sweep remain deferred. The broader research overview is
`docs/research/audio_conditioned_choreography.md`.

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
source H versus fully predicted timing, source/default controls, and seeds 17/23.
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


### Recovery probe result and external research question

The fixed coverage-v1 recovery panel completed in13.58 seconds on CPU with one
thread. All18 baseline charts contributed eight distinct eligible prefixes,
144 total, plus the separately identified failed student SCREW prefix. Every
translated content tensor was byte-equivalent to its original. No optimization
or new weights were produced. Fresh results are in
`diagnostics/recovery-conditioning-v1/result.json`; the consistent BOS supplement
is `bos-drive.json`. Their numerical summaries and all selected-prefix condition
values are now exported to a tracked evidence JSON named below.

For actual encoded histories, conditional on an uncommitted4-second wait, the
next8-second integrated hazard has median3.249924 and median no-event probability
.038777. Moving all past timestamps forward3800 ms, preserving content and
occupancy, changes these to16.846483 and4.8275e-8. All144 hazard ratios exceed2;
the median ratio is5.592885, and28 exceed10. This identifies sensitivity to elapsed
clocks while holding future audio and content fixed, not a musically approved
prefix transformation or evidence for a particular replacement architecture.

The nonphysical learned-BOS-history/old-exact-state condition strongly suppresses
hazard (median next8-second no-event probability .98134); it cannot justify a
physical reset. A separate **consistent** BOS condition, with both history and
exact past absent, has median integrated hazard4.871687 and no-event probability
.007664. Its hazard exceeds the aged condition in116/144 contexts, with median
ratio1.60695. It is a different conditioning event, not a way to erase committed
notes or holds. At the named failed student SCREW prefix, evaluated with the
coverage weights, consistent BOS has8-second hazard2.241745 versus aged .099481.

There are26/144 contexts with conditional no-event probability above.5 after the
hypothetical4-second wait. Crucially, the initial4-second survival probability
has median9.2106e-10 and exceeds.01 only2/144 times. These are edge-state probes,
not an estimate that18% of ordinary charts will stop. The earlier121-second
SCREW survival calculation is a different horizon and remains separately reported.

On2026-09-24 the user requested a self-contained question for a stronger external
model, specifying that it can access GitHub repositories but not local untracked
material. This redirects the immediate work to a concrete expert-review brief.
The user also clarified that both training and inference should use complete
audio, redline timing has little reference value, and final note placement is
the target. The current crop/full-encoding equivalence only covers the finite
local audio encoder; it does not mean the model uses whole-song musical context.
New full-audio conditioning must be consistent across training and inference.
Deferring long-term generated-chart memory does not forbid full-song audio context.
The real source-history/generated-history mismatch is a separate issue.

Product commit10a16fe64d65fa273d49921aeb6779976ee1117f adds
`docs/research/audio_joint_expert_question.md`, tracked numeric evidence and two
unchanged generated Lens PNGs under `docs/research/assets/audio_joint_expert_question/`,
a README entry and the clarified audio-information contract in the study document.
The brief states the model, masks, losses, query mixture, actual2000-step checkpoint
exposure (32000 logical/33862 physical queries), limited48-song training scope,
failed paired distillation, rare-tail caveat, runtime scope and open architecture
choices. No local artifact or note path is required to understand it. No music
files, full dataset, checkpoints or private absolute paths were published.

The GitHub-only delivery requirement was fulfilled by publishing the review
snapshot, including referenced tracked implementation, to the new branch
`codex/audio-joint-expert-review` in ensomi-labs/ensomi-model. Product main and the
agent-notes remote were not updated. The remote branch head equals the exact
commit above; GitHub API blob hashes for the brief, evidence, two figures and
model source equal local Git objects. Anonymous web access to the immutable
Markdown URL also succeeded:
https://github.com/ensomi-labs/ensomi-model/blob/10a16fe64d65fa273d49921aeb6779976ee1117f/docs/research/audio_joint_expert_question.md
No pull request or message to the external model was sent; the user will deliver
this link.

Before publication, the outgoing scope was checked against fetched origin/main
5c56e28bbf1ab92abaa0436b33c0d33a6c30eead. No outgoing commit or tree tracks
artifacts/agent-notes. A scoped credential-pattern scan covered71 outgoing code
history blobs and the new text evidence, with no findings; gitleaks was absent,
so this is not a claim of a full secret-scanner run. All15 local brief links
resolve, figures match their inspected PNG bytes and numerical identities were
checked. `uv run --extra mps --group dev pytest -q tests/research/audio_skeleton
tests/research/joint_audio_continuation tests/test_package_layout.py` passed141
tests and22 subtests in5.37 s; the packaged joint-audio `--cfg job` also resolved.
The final product worktree is clean. The Note remains proposed, acceptance none.
The broad playability goal remains active, with no live training process.


## Paired-audio expansion audit

The preceding goal turn delivered new recovery evidence and the published expert
brief, so it was progress. Product10a16fe and Note5048abd were revalidated clean;
no training remains live. The immediate independent work is data preparation,
without committing to a new architecture before the external feedback arrives.

Audit the existing pinned13216-entry TRAIN/VAL catalog and its admitted source
cache. Verify source/cache identities; locate each paired audio through the source
header and hash unique audio paths. Report exact-byte audio collisions across
splits and metadata groups, missing audio and source eligibility. Compute note
placement statistics relevant to coverage: event/head counts, head-row density,
LN fraction and durations, opening waits and later event/head gaps. These are
sampling descriptors, not Tech/Jack/LN-coordination labels. No TEST entries or
human labels are used, and no model generation or fit selects the new candidates.

The audit writes fresh metadata only under
`artifacts/joint-audio/20260924-data-audit-v1`, at most900 seconds, with one CPU
thread,2 GiB available memory and40 GiB free disk guards. Canonical Mel and decoded
waveforms are not computed during this first identity pass; equal encoded audio
bytes prove an identity, but unequal bytes do not prove different recordings.
A candidate expanded cohort will be selected only after the audit, preserving
named baseline identities where compatible with the stronger split boundary.
No checkpoint or published expert-review snapshot changes in this pass.


### Encoded-audio audit result and decoded-identity extension

The identity/placement audit completed in about20 seconds. It verified all13216
catalog source/cache byte identities. TRAIN has11136 paired eligible charts,
3081 metadata groups and3762 unique encoded-audio hashes; VAL has1589 charts,
427 groups and612 hashes.490 catalog charts lack adjacent local paired audio.
Eleven audio hashes span multiple metadata groups, including one spanning splits:
two TRAIN Bergentrueckung/ASGORE charts and a VAL compilation-pack ASGORE chart.
Its hash is4a8439c9dd010c42b540b8b32b1015e14d0edab41360a79fb90bdbbd700511c3.
No named baseline source is in that collision. The prior60-song cohort remains
unchanged. These counts exclude one further seed-ineligible TRAIN chart from
paired eligibility. TRAIN includes1261 paired charts with1641 later event waits
above4 seconds and1281 with openings above4 seconds. These are coverage facts,
not new human style labels or evidence that more data will necessarily fix quality.

Extend the audit by hashing canonical decoded, peak-normalized24 kHz waveforms
for every readable VAL audio identity plus all named baseline TRAIN identities.
This supplies a protected validation audio boundary before choosing more TRAIN
songs; no test audio or model output is used. Validate recomputed waveform hashes
against existing canonical assets wherever available. Record exact sample counts,
frontend identity, input hashes and decoding failures. Do not equate different
PCM hashes with different musical recordings; re-encodings/crops remain a limit.

The extension has a1800-second bound and the same2 GiB RAM/40 GiB disk/PAUSE
guards. It streams one waveform at a time and saves hashes/metadata, not audio.
Only after this pass will new TRAIN candidates be decoded and screened against
VAL. A proposed next corpus will aim at240 TRAIN groups and36 VAL groups, keeping
compatible baseline identities, separate same-audio arrangements and balanced
placement-density/LN strata. Actual decoded duration30–720 seconds is a compute
slice, not a product duration limit. The corpus and evaluation identity must be
frozen before any model fit or native quality selection. No new training or
architecture choice is authorized by a favorable data audit alone; the existing
user research authority remains the execution basis for subsequent scoped work.


## Expert feedback and the first discriminating experiment

The user supplied external-model feedback on2026-09-24, attachment SHA
6f4997ad9e900e85fa7c8815f467e60a0cc8cd116ea54383a77bfcb160ed109a.
Its technical recommendations are treated as research input, not acceptance of
an exact Experiment Card. The standing user authority covers implementation and
bounded experiments. The primary adopted order is: increase paired-song coverage
with the current model/objective; then separate full-song coarse audio from bounded
history modulation in a2x2 experiment; consider a persistent small latent only if
composition consistency remains a demonstrated bottleneck.

Two corrections are material. Hazard times conditional-row likelihood is a valid
chain factorization; absence of a gradient through a sampled time is not itself
a defective likelihood. A marked-intensity reparameterization need not enlarge
the support. Also, the144 recovery probes deliberately required>=30 heads and
free lanes, so they do not cover the10-head SCREW failure or held-note prefixes.
The next evaluation must retain BOS,1–29-head and mature states, with held/free
subgroups, and count failed early generations instead of conditioning them away.

### Expanded paired corpus

All660 requested VAL/core waveforms decoded successfully in about199 seconds,
and all60 existing canonical waveform hashes reproduced. No additional different-
file/same-PCM pair was found in that scope. A metadata reconciliation attaches all
catalog owners of those already decoded hashes: the known ASGORE cross-split
identity remains one; no original core TRAIN audio crosses VAL. The first decoder
receipt listed only requested VAL/core owner occurrences, so its zero cross-split
count did not include the ASGORE TRAIN owners. The reconciled receipt corrects
ownership without repeating or changing audio computation.

Deterministic selection with seed230924 retained all60 base charts and chose240
TRAIN plus36 VAL base groups, six density/LN strata with40/6 groups each. Each new
TRAIN PCM was checked against every decoded VAL identity;192 new TRAIN audios
were decoded. Selection took53.86 s and had two candidate exclusions. The selected
true decoded duration slice is30–720 seconds. This is a preparation slice, not a
runtime duration limit or an automatic style judgment.

Canonical preparation completed in95.86 s. Corpus root
`artifacts/joint-audio/20260924-expanded-v1` has585 separate TRAIN arrangements
from240 groups and36 VAL songs. Manifest SHA
cc60dd39920c626f3be5498ab8ba7aa342dd0956c0a548cc393d85a1b03f5173.
Selection SHA bbf550fdfbd1d2d7dc42a4233c448b536695db9cf52465e223a9133743cd138c.
All133 charts of the previous corpus are retained. Encoded and decoded audio
identities are disjoint across the selected splits, and the known global collision
is absent. The98 alternative exclusions are75 different-audio arrangements,
22 missing local audio and one insufficient-seed source. No source audio was
fabricated or time-padded to accommodate target notes.

The released R1 checkpoint has6750000 source-onset exposures and10338 nonempty
source-coverage bitsets; all corresponding catalog sources are TRAIN, with zero
overlap with the36 selected VAL source identities. Its response and release plans
both use the same pinned catalog/split and11563 source owners. Native recovery
pool ownership is being separately verified. Exact identities do not establish
perceptual/crop/speed de-duplication or that a song was never used in earlier
model selection.

### Experiment Card: paired-song-coverage-v1

Revision1. Proposed, acceptance none. Question: does increasing paired training
coverage improve held-out audio-conditioned likelihood and native generation
without changing the event representation or introducing a new model component?
This tests the data-coverage branch before the suggested audio/history2x2.

Baseline is the coverage-v1 **last2400-update** checkpoint, not its selected
best2000 checkpoint. SHA8edd5b723e5bb25673ffe53102e66fb6db2d6cb56ce4bfc4d56a38af6e2a61fd;
its source is608f092e6cd534638e8e47432bcb98973b79a5a4. Model, data, batching,
sampling and training files are byte-identical between that source and current
10a16fe64d65fa273d49921aeb6779976ee1117f, before the explicit normalizer option
below. Reuse that recorded control rather than rerun the same deterministic
training recipe. Its original48-song/121-arrangement corpus manifest is
4b995029a5344569d4506ff6b11249f61585d2bf7649285754340909bb06c21b.

The intervention changes supervised paired data to the frozen240-group/585-chart
corpus above. Start from the same released R1, torch/sample seed230923 and2.95M
architecture. Use the same2400 updates,batch16, new/inherited LR3e-4/3e-5,
AdamW weight decay .01, gradient clipping1, full waiting likelihood, query mixture
and once-per-long-transition coverage policy. Different numbers of long waits
can expand the same logical budget into different physical query counts; record
both counts and wall time. The causal change is data coverage and its source-
determined coverage examples, not a new loss or decoder rule.

Freeze the existing48-TRAIN audio normalizer for both arms, SHA
9cf461a0e825f974f0a80a364123c7afedf1683af76e60fe09b0fbe51c2c8287,
so a changed data-dependent standardization does not change the initial raw-Mel
function. Add an explicit pinned TRAIN-only normalization override to the
packaged training config, validate that its audio identities are a subset of
current TRAIN, and record the actual normalizer in the run freeze. Defaults retain
existing behavior. This is the only required training-owner change; add focused
config, ownership and consumption tests and commit before the run.

Evaluate the fixed final2400 endpoint. A common36-song validation panel uses four
baseline-policy queries per song with seed230926; report old12 and additional24
separately. Freeze the panel before evaluating either endpoint. Add true-source
BOS and1–29/mature held/free prefix panels as diagnostics with explicit missing
strata; they do not alter training. Report each arm's native full outputs from
BOS on the same36 VAL songs and six original TRAIN examples, seeds17 and19,
500 ms queries and no head-spacing prior. Retain early failures, capped outputs,
zero/low-head cases and missing first30 latency as outcomes, never exclusions.
Per-chart caps are30000 rows and90 seconds, whole-cohort cap900 seconds. Stop and
record resource/numerical/mechanical failures rather than invent endpoints.

Primary quantitative evidence is new24-song paired per-song joint NLL and native
early-generation/activity behavior. A>=3% reduction in new-panel mean NLL with
no>5% old-panel increase is a learning signal, not enough for adoption. Preserve
or improve the fraction of completed songs reaching30 heads; inspect full-song
head activity, local rates, LN-head share, occupied-lane time and duration
quantiles, including per-chart outliers. Flag median head-count ratios outside
[.75,1.25] or total LN-head ratios outside[.7,1.3] as composition regressions
requiring actual Lens adjudication, not automatic BAD labels. Report<=10/20 ms
TAP relations both as absolute counts and per eligible consecutive TAP-to-TAP
transition. Compare held/free and early/mature behavior without turning a
conditional tail probability into failure incidence.

Lens review must cover persistent/new failures and surviving expressive patterns;
source-density agreement and universal anti-repetition scores do not define
success. If source likelihood improves but native quality does not, shift priority
to the proposed full-audio/history experiment. If the larger corpus remains
underfit, this matched-budget stage cannot refute data scaling: record its curves
and define an explicit continuation before spending a larger training budget.
No latent, interval-objective change or unknown-history auxiliary enters this Card.

Run on this Mac with MPS, one CPU thread, FP32, at most3600 seconds for the new
training arm;2 GiB available RAM,40 GiB free disk and root PAUSE guards remain.
Fresh output is expanded-v1/training/coverage-240-v1; no overwrite or implicit
resume. Source-normalization provenance and the common evaluation freeze are
required before generation. Inference profiles exclude/include audio preprocessing
explicitly. No TEST or new architecture-default adoption follows automatically.

### Result Log: paired-song-coverage-v1

Card revision 1; accepted revision none. This is exploratory evidence under the
standing user authority. The clean training source was
f25508ef77a49fda3212652226a418211415b66c. The normalizer override is the only
training behavior extension; defaults preserve the older recipe. Native
evaluation used caee76b99d66530522c863012e1f968bba8993f0, adding diagnostics
without changing generation. Thirty-three focused config/model/sampling tests
passed; a one-update MPS preflight verified exact consumption of the old TRAIN
normalization buffers. Three diagnostic tests cover empty output, early held
states, separate TAP/release relations and the complete row crossing 30 heads.

The old control used two CPU threads and the expanded run used one, with MPS
for both. Reconstructed initial parameters and buffers under the two thread
settings have the same hash,
cf95aae8d6f5c21f0333ec2f6a896e245baad9de9edb682b4a8ec135476c3f7a.
This does not assert bitwise training-trajectory equality. Normalizer and
initialization receipts are in expanded-v1/evaluation-coverage-v1.

The new run completed all 2400 updates in 1385.83 seconds: 38,400 logical
examples, 40,235 physical queries, and all 136 coverage examples. The control
used 40,554 physical queries for the same logical budget. The fixed new last
checkpoint is a5e6c7434557b3402e2b06cebbd8e6e2779d24eeddd9c6cd88a9dca6b542fc90.
The final update was also the best validation update; the curve was still
improving. This is not a convergence claim. A TRAIN raw-support audit ran on
CPU during part of training, so wall time is not an isolated speed benchmark.

The pre-frozen common protocol hash is
9ff28d5c5f30effc13d8205c5e61967016b67457876a693c8cfbcb2545076d57.
It contains 144 common queries (four per VAL song) and 289 full waiting
examples across BOS, early free/held and mature free/held states. Seventeen
unavailable per-song strata are explicitly recorded. The original 48 common
queries are unchanged. These are true source prefixes, not altered generated
prefixes paired with the original future.

| Development panel | Small-data joint NLL | Expanded-data joint NLL | Change |
| --- | ---: | ---: | ---: |
| Original 12 songs, 48 queries | 6.3542 | 6.2128 | -2.23% |
| Additional 24 songs, 96 queries | 6.4852 | 5.4883 | -15.37% |

The combined query mean is 6.4415 versus 5.7298. The planned likelihood
threshold passes. Source-stage NLL also improves in most strata; original-panel
mature-free is essentially unchanged (6.0041 versus 6.0064). There is one
training seed and no independent training-replicate uncertainty estimate.

Both arms completed all 84 requested native outputs: six TRAIN and 36 VAL
songs at seed bases 17 and 19, preserving source-to-seed pairing. No decoder
head-spacing prior was used. Early and empty outputs were retained. Cohort
generation took about 257 seconds for the control and 257 seconds for the
expanded model, excluding audio decode/Mel. Full result comparison hash:
b1e360bfd8d830ed170730cabe91de164b2c633457f75521682588ab644d8da1.

| Native diagnostic, all 84 outputs | Small data | Expanded data |
| --- | ---: | ---: |
| Empty outputs | 2 | 0 |
| Outputs below 30 heads, including empty | 2 | 3 |
| Total heads | 138,688 | 135,146 |
| Total LN heads | 30,157 | 19,484 |
| Occupied lane-time / observed lane-time | .08075 | .06967 |
| Median per-chart median closed-LN duration, ms | 115.5 | 146.5 |
| Eligible consecutive TAP-to-TAP relations | 97,539 | 105,493 |
| TAP-to-TAP <=10 ms, absolute / per 1000 relations | 4 / .04101 | 1 / .00948 |
| TAP-to-TAP <=20 ms, absolute / per 1000 relations | 11 / .11278 | 4 / .03792 |

The median paired head-count ratio is .982 (82 finite ratios; two zero
control denominators are separate). The LN total ratio .646 violates the
predeclared composition review guard. It is not a universal LN-quality score.
Additional-24 outputs improve from two empty to none below 30, with median
head ratio 1.174. The original-12 panel instead gains two early failures and
has an LN total ratio .433. TRAIN gains one early failure. The global count
therefore conceals opposite subgroup changes.

#### Lens review and interpretation

The canonical Lens bridge admitted all 168 generated exports plus 42 distinct
source files with no skips or diagnostics. Both empty exports remain explicit
outcomes; no renderable section was invented for them. The frozen review bundle
hash is 3a0d435ed8ac07354c34f679f65768cd3c1eaa6df770621d87aae795ec53d9a6,
at expanded-v1/lens-review/coverage-v1. All 352 frozen member hashes were
reverified after inspection. The original 204 human examples and Foundation
remain unchanged. Admission is mechanical validity, not a quality verdict.

Complete paginated actions and articulation were retrieved for nine paired
windows. Time-proportional generated/source pages were visually inspected;
the repeated Dotabata source pages share the same source/window. Evidence is
under that review root's evidence directory. These are targeted failure and
expressiveness checks, not a blind whole-cohort or audio-listening study.

- Ju-Ju Yakiniku seed base 19 emits eight heads, ending at 2414 ms of
  117211 ms; SCREW seed base 17 emits ten, ending at 4142 of 121033 ms;
  glacia seed base 19 emits thirteen, ending at 2047 of 121913 ms.
  Their short opening sequences are mechanically valid. Their nearly empty
  remaining songs are not successful playable outputs.
- Paganini seed base 19 reaches 31 heads but stops at 7347 ms of 148618 ms.
  All three pages of its generated opening and source context were inspected.
  Passing first-30 does not establish sustained generation.
- GR4VITY G4ME seed base 19 contains lane-2 TAPs at 230445 and 230455 ms,
  inside continuing mixed-lane motion. This is an isolated same-key 10 ms
  reattack, distinct from a release-to-head relation or a deliberate long
  jack organization. The timing anomaly remains a local playability failure.
- Dotabata seed base 19 has no actions in the inspected 178154–182154 ms
  window, while the control and reference contain continued LN/tap motion.
  This does not prove every source rest must be filled, but the lower LN
  total cannot be credited as better local organization in this empty scope.
- Tsuikou seed base 19 retains staggered cross-lane LN starts/releases,
  synchronized closures and tap/LN combinations in 193684–197684 ms.
  The representation has not collapsed to taps only. Musical alignment and
  full-song quality remain unjudged.
- In my room recovers both previously empty generations. The inspected
  12904–16904 ms scope is active mixed single/chord motion with short LNs,
  substantially denser than its easy reference. Recovery is real; density
  mismatch alone is not BAD, nor does activity alone prove good mapping.

The evidence supports improved held-out conditional prediction from wider
paired coverage, but not adoption of the new weights as a playable replacement.
Native early stability fails its guard, and long silence occurs even after
30 heads. Composition changes and continued source-curve improvement leave
undertraining as a live alternative. They do not establish that more training
alone will fix the history-dependent generation failure.

Recommended outcome: REFINE. Prioritize the expert's local/global audio crossed
with original/bounded-history timing experiment at fixed data and interval
objective. Preserve this coverage comparison; do not reinterpret it as a
rejection of data scaling, or tune a decoder floor to force activity. No exact
Card acceptance, model adoption or remote publication is implied.

#### Completed input-support audits

R1 native recovery-pool ownership also resolves to TRAIN: 32 sources/runs and
92 queries, with no selected VAL identity. Cumulative checkpoint coverage,
response/release plans and pool ownership share the pinned source split.
Exact identity checks still do not guarantee absence of perceptual duplicates
or previous validation-based model selection.

The raw same-lane close+restart audit verified 11609 unique TRAIN source files
from the admission owner, including rejected cases. Of these, 11604 parsed
successfully, containing 16,078,030 objects; five failed with other raw
ambiguities. No same-lane same-millisecond LN close+restart relation was found
in the parsed set. No VAL/TEST payloads were read. This corpus audit supplies
no evidence for changing the current action alphabet before the next study;
it is not a universal statement about mania charts.

## Experiment Card: audio-context-history-paths-v1

Revision 2. Proposed; acceptance none. The standing user instruction authorizes
bounded implementation and experiments. This Card succeeds the completed
paired-song coverage comparison without changing its recorded result.

### Question, mechanism and alternatives

Can full-song audio context improve source-conditioned prediction and musical
organization, while a bounded time-history path prevents stale generated
prefixes from indefinitely suppressing continuation? Test these as two separate
factors. Do not assume that either component works or that correcting silence
establishes playable mapping.

The four cells cross local versus local-plus-global audio with the existing
timing fusion versus an audio/hold-only timing base plus bounded history
modulation. All cells use the same continuous-interval likelihood, examples,
initial R1, normalization and optimizer. No latent, missing-history auxiliary,
BeatThis, metrical grid or head-spacing decoder prior enters this experiment.
Interval training changes the earlier sampling objective; its local/fused cell
is therefore a newly trained control, not the old coverage checkpoint relabeled.

Closest primitives are bidirectional self-attention
([Transformer](https://arxiv.org/abs/1706.03762)) and time-evolving historical
influence in event processes
([Neural Hawkes](https://arxiv.org/abs/1612.09328)). The former supplies relations
between coarse audio positions; the latter motivates distinguishing persistent
physical obligations from decaying event influence. Neither paper establishes
this task-specific hold gate or playability. This is an adaptation/combination
experiment, with no new representation or novelty claim.

### Fixed implementation and comparison

Baseline product source is caee76b99d66530522c863012e1f968bba8993f0. Use the
240-group/585-arrangement TRAIN and 36-song development VAL corpus, manifest
cc60dd39920c626f3be5498ab8ba7aa342dd0956c0a548cc393d85a1b03f5173, and the old
TRAIN normalizer 9cf461a0e825f974f0a80a364123c7afedf1683af76e60fe09b0fbe51c2c8287.
Initialize all cells from released R1 4b3ec1561e33d0ebe2756cfe13571ec414fd5bb470b430f0c578545863115f70.
Model seed is 230928. Preserve identical common-module initialization; verify
this separately from architecture-specific parameters. All new parameters train
at 3e-4; inherited R1 parameters at 3e-5, AdamW .01 decay and norm clip 1.

Keep the 100 Hz, 96-dimensional local TCN and 511-row finite R1 history. The
global branch reduces normalized full-song Mel with learned depthwise temporal
weights over 50-frame cells, projects to 128 dimensions and uses two
bidirectional Transformer encoder layers, four heads and 512-wide feedforward
layers with dropout zero. Partial last cells count only real frames. Tokens
have fixed 500 ms cell-center coordinates; these are audio coordinates, not
output event slots. Positional sinusoids use elapsed audio seconds. The branch
is encoded once per song per optimizer microbatch and once per generated song;
candidate queries interpolate its cached representation. Both heads receive
global context through explicit projections; no chart future is an input.

For bounded timing, use logit h = b(F,G,S_hold) + 4*g*tanh(r(F,G,H,S)). The
base reads only audio, occupancy and active-LN age features. It cannot read
previous-row age, cumulative counts, time since the first row or content history.
The history path retains the full exact state and learned history. Set g=1
while any LN is active, exp(-elapsed_since_last_row/1000 ms) when all lanes
are free, and zero at genuine BOS. This is one preregistered bound/time scale,
not a parameter sweep. The property |logit h-b| <=4*g must hold numerically.
There is no hazard floor, forced nonterminal event, erased action history,
or forced query-boundary LN closure.

### Interval objective and exposure

Partition the real integer audio clock 0..T into disjoint intervals of at most
8000 milliseconds, including a possibly short final interval. At each sampled
interval, retain true pre-interval source state and the history envelope. Score
every event and every no-event millisecond exactly once, with updated true
history after each event. Sum timing and complete-row NLL before any weighting.
Reuse one causal-TCN sequence over the prefix plus interval. Padding, missing
audio and chunk boundaries cannot become events or context observations.

Optimize the mean over uniformly selected TRAIN groups, then separate
arrangements, of each chart's NLL per second. Select an interval uniformly
among that chart's J intervals and multiply its summed NLL by
1000*J/(T+1). Thus the expectation is full-chart NLL divided by its actual
integer-clock duration; short last intervals have the correct inclusion weight.
This intentionally weights songs/arrangements uniformly and chart time within
each song, rather than reproducing the former query mixture. Do not separately
average row and survival losses or merge alternative arrangements into labels.

Freeze a shared plan using seed 230929 before fitting: 1200 updates, two sampled
song groups per update, two independently selected arrangement/interval pairs
per song. All four cells see the same 4800 intervals, with actual milliseconds,
event rows and heads counted. Backpropagate each song microbatch before the
next, with one optimizer update after both; full-song audio is shared within
that song. A preflight of at most 32 updates per cell uses a separate fresh
destination and is not reused as fitted initialization. A structural or resource
failure stops scaling and requires a documented Card revision.

### Evaluation and decision

Freeze four uniformly drawn source intervals per VAL song with seed 230930;
report the per-song mean weighted NLL per second, original 12 and additional 24
separately. The BOS interval is an additional diagnostic, not mixed into that
population-weighted mean. Keep the previous BOS/early/mature held/free panel
as a source-state diagnostic where the shared evaluator supports full audio.
Select the final 1200 endpoint, not the most favorable checkpoint. A >=3%
improvement on the additional-24 mean without >5% original-panel regression
is conditional-learning evidence; actual values for the new interval control
are pending and must be reported even if worse than the old recipe.

Generate all original 42-song cases at seed bases 17 and 19, keeping the exact
index-to-seed mapping, 500 ms scheduler queries, 30,000-row and 90-second
per-case caps. Retain empty, early, capped and post-30 silent outputs. Report
absolute and eligible-transition-normalized <=10/20 ms TAP relations; full-song
and local activity, LN counts, occupancy, durations and actual startup latency.
Reaching 30 heads may not regress versus the interval control, and cannot by
itself establish success. Median paired head ratio outside [.75,1.25] or total
LN ratio outside [.7,1.3] triggers Lens adjudication. No forced activity or
universal anti-repetition metric may turn a guard failure into a success.

Inspect paired Lens actions/time pages for early failures, mature silence,
real long rests, active-LN release transitions, dense tap/chord motion and
repetition. Preserve valid irregular timing and jack/chordjack opportunities.
Source density is a context cue, not a target difficulty. Human playing/audio
assessment remains distinct from structural inspection.

For any global branch, evaluate correct, zeroed and half-song-shifted coarse
context while keeping local fine audio fixed. These are diagnostic perturbations,
not new model selection sweeps. A correct-context benefit must also appear on
development VAL and in the matched architecture comparison; output changes
alone do not demonstrate use of musical information.

Positive: a factor improves held-out likelihood and inspected native stability
without flattening real rests, valid repetition or LN coordination. Negative:
global context only helps TRAIN, or bounded timing fills rests/loses coherent
repetition/releases. Ambiguous: better likelihood with continued native failures,
composition guards breached, or nonconverged/capped runs. Recommend REFINE in
the absence of an accepted Card; do not adopt components solely from a numeric
gate. Undertraining remains a live alternative to a failed architectural premise.

### Execution bounds and reproducibility

Add scoped context-model, interval and training owners under joint_audio_continuation,
a package-local Hydra preset and focused tests. Preserve the existing model
and checkpoint defaults. Validate interval likelihood against independent old
single-query sums, dense/cache and crop/full equivalence, global padding/clock
semantics, hold-only base independence, bounded modulation, correct sampler
weights and config projection. Commit a clean implementation before preflight.

Command family: `python -m ensomi_model.research.joint_audio_continuation.context_hydra`
with explicit `global_audio={false,true}`, `bounded_timing={false,true}` and
unique run_name, under `uv run --extra mps`. Save resolved config, flat config,
source OID, manifest/normalizer/checkpoint/plan hashes, counts, resource/learning
curves and stop status. Main output is
`artifacts/joint-audio/20260924-expanded-v1/context-training/paths-<local|global>-<fused|bounded>-v1`.
Never overwrite; no implicit resume or reuse of a selected checkpoint.

Run serially on this Apple M5 Mac, MPS FP32 and one CPU thread. Bound each main
cell to 7200 seconds and the four-cell training queue to eight hours. Preflight
has a 900-second per-cell bound. Retain the existing PAUSE, 2 GiB available RAM
and 40 GiB free disk guards. Do not parallelize competing MPS training jobs.
Record actual end-to-end audio encode/startup and dense generation cost before
making realtime claims; warm Mel throughput is a separate measurement. No TEST
reads, remote publication or automatic model adoption is part of this Card.

### Context implementation and preflight memory diagnostic

Implementation source ce25a7b3e90d91225e22a9a63b65b9cd59bafcb3 adds the two
model switches, shared-history interval scoring, full-song-aware query scoring,
packaged Hydra and checkpoint dispatch. Local/fused retains 2,950,458 parameters;
local/bounded has 2,992,964, global/fused 3,402,938 and global/bounded 3,461,828.
The selected package/joint-model checks passed: 152 tests and 22 subtests.
Hydra job composition also passed. These checks do not establish learning or
playability. An initial launch was rejected before creating a run because a
trailing-blank-line check had prevented the source commit; the formatting issue
was fixed and the clean-source guard was respected on the actual preflight.

The shared 1200-update plan hash is
dee20d179a929fc0ca8f73c4001a7189a238fa2278d989a04a87ed8e5b432b3e.
It contains 4800 intervals covering 37,300,370 actual milliseconds and 180
validation interval records. Local/fused preflight completed 32 updates in
66.24 seconds, using 128 intervals, 1,003,105 ms, 7239 event rows and 9903 heads.
The two-song population probe fell from 98.3325 to 71.3007 NLL/second. This is
a small learning/mechanics check, not the main comparison or quality evidence.
Its checkpoint is c7edb68a53048eda03a41b224b8eb37dc457f92b1604402e80a61b8ee90be4bb.

Available memory fell from 8.47 GB at update 10 to 5.86 GB at update 32;
driver memory rose from 1.00 to 1.55 GB. Those counters alone do not attribute
the growth. Before scaling, compare fresh global/bounded processes with one
fixed planned microbatch repeated 32 times versus the first 32 variable planned
updates followed by the same 32 again. Record active MPS, driver, process RSS,
available memory and wall time at aligned post-update phases; then release
references, collect, synchronize and empty the cache. Each diagnostic is bounded
to 360 seconds, with a 2.5 GiB available-memory/PAUSE stop, no saved fitted
checkpoint and no model selection. Artifacts are context-memory-fixed-v1 and
context-memory-variable-v1 under the expanded corpus. This is an exploratory
resource investigation under the same implementation source; it does not change
the main Card's data, objective or model factors.

### Memory result and Card revision 2

Global/bounded preflight also completed 32 updates, in 90.81 seconds. It used
the exact same 128 intervals, clocks, events and heads; the common initial
module hash was identical, 3c62dfd7a0a04a9ca1cd997d9f6216187649dca3804f294ce7a807336eb45256.
Its two-song population probe fell from 98.3393 to 68.7119 NLL/second. Its
checkpoint is 153d39f344f6f75d3cdba5dfa3416bf20e91775e62503ad4fcb3c4ac75f548db.
This does not establish an architectural win at 32 updates on two songs.

The fresh fixed-input memory probe took 8.87 seconds for 32 updates. Active
MPS storage stayed at 167,217,664 bytes after warmup; driver storage stayed at
614,645,760 bytes. The variable-input cold cycle took 78.43 seconds for the
first 32 planned updates. Repeating those same 32 took 13.10 seconds. Active
storage remained exactly 167,217,664 bytes throughout both variable cycles;
driver storage reached 2.88 GB at the cold-cycle end and approximately 2.91 GB
after the warm cycle. Available memory fell in the cold cycle and then stayed
near 4.6 GB in the warm cycle. Cache release reduced driver storage to 1.29 GB,
but did not return process RSS to its initial level. No resource guard fired.

These observations locate substantial cost in first-seen input shapes and
retained runtime allocations; they do not fully attribute every RSS/driver
byte or prove a particular allocator mechanism. Apply behavior-preserving
padding buckets before scaling: local audio/history/timing axes at multiples
of 128, row-query axes at multiples of 64, and complete coarse audio at a
power of two in 50-frame cells. Masks and target counts exclude padding, and
the full-song context shift rotates real coarse tokens only. Record active
MPS and process RSS alongside existing counters. The model, objective, source
intervals, initial weights and main exposure budget remain unchanged.

Revision 2 permits this execution-shape change and fresh 32-update preflights
for all four cells under `paths-<local|global>-<fused|bounded>-preflight-v2`.
The two v1 preflight results remain evidence and are not overwritten or used
as initial weights. Recheck independent-query likelihood, causal indices,
padding/full-song equivalence and the native checkpoint path. Repeat the same
64-update variable/cold-warm memory probe with a fresh v2 output before main
training. Stop scaling if memory remains unbounded or the masked equivalence
checks fail. Main run names, input pins, shared protocol and 1200-update bound
remain those already specified; Card acceptance remains none.

### Masked buckets and four-cell preflight

Clean implementation b5c5ee33cacb3950b3c9b6a4a05d9d627c46aa64 adds only
masked execution buckets and memory counters. Thirteen focused context tests
passed, including independent unbucketed-query likelihood and native partition
parity. The frozen data protocol and all model parameter counts are unchanged.

The repeated variable-shape probe completed 64 updates in 21.79 seconds:
12.55 seconds cold and 9.23 warm, versus 78.43 and 13.10 previously. Active
MPS memory stayed at 191,561,216 bytes after warmup. Driver memory ended near
1.78 GB and fell to .48 GB after cache release; available memory remained near
9.76 GB at the end of both cycles. Process RSS changed from about 2.49 GB to
2.52 GB during the warm cycle. This bounds the observed execution problem;
longer main-run monitoring remains required.

All four revision-2 preflights completed with identical initial common-module
hash, source protocol and exposures: 128 intervals, 1,003,105 ms, 7239 rows,
9903 heads, 106826 real timing bins. Results are at context-queue-preflight-v2.

| Cell | Seconds | Two-song population NLL/s | Last checkpoint SHA-256 |
| --- | ---: | ---: | --- |
| Local/fused | 14.33 | 71.300747 | 993bd006e0074beeb518c673797d24cd641d31999c4ef42e4ac42771b73bdd4e |
| Global/fused | 15.90 | 70.921581 | b599e718dafb3f2ab0fbb8a687dbb575f32cf68a5d9085d0d46f381a8d43ebdb |
| Local/bounded | 14.48 | 69.336330 | 36a1680325426fe62aa3b8368ad1f2b30c1fdf8cab36a2e4a6c559e73185816a |
| Global/bounded | 16.10 | 68.711928 | 7a10bc623bfa96a45135bb71ec357a2c28292ade1f1150f44dd2f4134b8ca148 |

Local/fused and global/bounded NLL differ from the unbucketed 32-step probes
by less than .000002. These preflights establish bounded execution and a
learning signal, not an architecture ranking. Main training starts afresh
from R1 for all four cells, serially under the frozen 1200-update budget,
using run_context_queue.py and context-queue-main-v2. The queue checks common
initialization/exposure identity and stops after any incomplete cell.

One specific falsifier deserves attention in native review: a bounded
historical adjustment limits inhibition as well as facilitation. A strong
audio-base peak might require greater short-lag suppression to prevent a
second same-key hit. If recovery improves but <=10/20 ms TAP relations worsen,
inspect base, modulation and gate on those exact prefixes; do not count added
activity as success or change the bound after seeing the result. The original
short-transition diagnostics and Lens guards already cover this risk.

### Result Log: audio-context-history-paths-v1 revision 2

Accepted revision none; exploratory execution under the standing user
instruction. All four cells completed 1200 updates from clean
b5c5ee33cacb3950b3c9b6a4a05d9d627c46aa64 and the same frozen protocol.
Each consumed 4800 intervals, 37,300,370 ms, 248,836 complete event rows and
339,728 heads. The common initial-module hash matched. No resource guard
fired; available memory remained above approximately 8 GB in the recorded
main runs. The four fixed last checkpoints are:

| Cell | Checkpoint SHA-256 | Seconds | VAL population NLL/s |
| --- | --- | ---: | ---: |
| Local/fused | c5a9f5c775e5a228893e940143ca47abd3fab1a819b273cede2b53ac30f61762 | 359.26 | 41.91875 |
| Global/fused | 58135f737925107729c13e5e13221f321a62ca4ed70eff877a30f6c6be8d6234 | 393.92 | 42.13027 |
| Local/bounded | a0372bec2deb8c8ac9bf96028bfc91579d413b6a00df5c12793029a8d29d95bf | 367.46 | 41.83088 |
| Global/bounded | b7d56ea062b3ab0d4bdf27fcfb10835a94e376e8568761e37811391a3ebc31eb | 602.79 | 41.93044 |

The last training cell overlapped one CPU native-generation job. Later native
generation used two CPU processes, one thread each. No MPS training jobs
overlapped. Their realized durations are not an isolated architecture-speed
comparison; final startup/dense inference must be profiled separately.

On the additional 24 songs, population NLL/s is 37.69641, 37.85394,
37.73392 and 37.72944 in the same order. Original-12 values are 50.36342,
50.68295, 50.02479 and 50.33245. No intervention meets the preregistered 3%
held-out likelihood improvement threshold. All are close on this objective;
one training seed does not support a definitive ranking.

The old common-query panel gives additional-24 means 5.96186, 6.04052,
6.14301 and 6.20076, compared with 5.48827 for the earlier expanded-data
query-trained endpoint. Original-12 means are 5.89821, 5.97418, 6.02181
and 6.09964, compared with 6.21280 previously. Thus the new recipe does not
uniformly improve source conditional prediction. Source BOS waiting NLL is
also worse for bounded timing than the interval local/fused control. Interval
BOS NLL/s and complete first-event waiting NLL are different measurements.

#### Correct versus perturbed global context

Global/fused additional-24 population NLL/s is 37.85394 with correct context,
42.32608 when zeroed and 37.88364 when shifted by half a song. Global/bounded
values are 37.72944, 41.52112 and 37.74602. Original-panel shift changes are
also small. The decoder uses the global path, but these source-conditioned
probes provide little evidence that its time-specific alignment is useful.
Song-level conditioning is a plausible interpretation, not proof that full-song
musical relationships are unnecessary. Half-song shifts can also preserve
repeated material, and teacher-forced history can conceal native dependence.
The correct/masked/shifted records remain separate under evaluation-context-v1.

#### Native outcomes and composition

Every requested generation completed: 84 outputs per cell, 336 total, with
no empty or below-30-head output. Every final head occurs after 85% of its
audio duration; the maximum observed head-gap fraction is below .156 across
these outputs. This removes the gross early/post-30 silence observed in the
previous query-trained probes. It does not prove preservation of every proper
rest or universal stability.

| Cell | Heads | LN heads | LN fraction | Occupied lane-time fraction | TAP <=10/20 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Local/fused | 225579 | 92291 | .4091 | .2112 | 2 / 8 |
| Global/fused | 225817 | 110318 | .4885 | .2671 | 3 / 14 |
| Local/bounded | 199812 | 92161 | .4612 | .2287 | 2 / 6 |
| Global/bounded | 184509 | 102118 | .5535 | .2825 | 0 / 3 |

Against local/fused, median paired head ratios are 1.009, .895 and .814;
LN-total ratios are 1.195, .999 and 1.106. These pass the numerical whole-cohort
composition bounds but still change playing demand. Global/bounded has longer
LNs and greater occupancy. It is not a free improvement obtained from a
single short-TAP count.

Inspection exposed short TAP-to-LN head transitions that the TAP-only statistic
omits. Behavior-neutral diagnostics in b757da32ec081954c37c3945305a2e8b9d2cc1b9
now retain all four head-transition types, their eligible denominators, and
short LN duration counts separately from release-to-next-head gaps. Four focused
tests passed. Every existing generated row file was rechecked against its
recorded hash before deriving the additional report; baseline outputs were not
rewritten.

| Cell | All heads <=10/20 ms | <=20 per 1000 eligible head transitions |
| --- | ---: | ---: |
| Local/fused | 4 / 17 | .07547 |
| Global/fused | 9 / 35 | .15522 |
| Local/bounded | 4 / 11 | .05514 |
| Global/bounded | 1 / 4 | .02172 |

Global/bounded improves both inspected short-transition types: TAP-to-TAP
<=20 ms is .04857 per 1000 eligible TAP pairs versus .07237 for local/fused;
TAP-to-LN is .04868 versus .39885. The latter denominators are 20544 and
22565, so this difference is not explained solely by fewer TAP opportunities.
Counts remain small, and changed composition and one training seed limit the
causal quality conclusion.

#### Lens evidence

The canonical Lens bridge admitted all 336 outputs and 42 source files with
no skips or diagnostics. Full bundle hash:
1d18395220d0fad111983225dbd220c5b3ef0ab113aac3b5a96f1da4af8edea9, at
expanded-v1/lens-review/context-v1. An earlier local/fused-only bundle is
dea3095bd0cc7ed828b88e99d95f40d233803b5bc645de02712daf061e4d16d9.
Human records remain the unchanged frozen 204-example set. Admission is only
mechanical validity.

Complete paginated actions/articulation and time-proportional pages were
reviewed for nine local/fused windows and six global/bounded windows. Identical
reference pages were reused only after byte-hash equality checks. The inspected
scope includes early SCREW, middle Paganini, YOASOBI tap/LN transitions,
Tsuikou dense chords, late Dotabata LN coordination, repeated-chord candidates
and the located short-head failures. The generated/source evidence remains in
each bundle's evidence directory. This is targeted structural review, not
blind listening, player validation or a whole-song human label assignment.

Local/fused SCREW now continues through its formerly failed early region.
Paganini seed 19 continues across the song, though its inspected middle is
a much sparser tap interpretation than the reference. YOASOBI and Tsuikou
show mixed chord/tap movement and LN transitions; Dotabata's formerly empty
late window now contains sustained cross-column LN motion.

Local/fused also contains a verified 4 ms same-key TAP repeat in Airborne
Robots at 235816/235820 ms and a 7 ms TAP-to-LN head interval in Otherside at
78345/78352 ms. Global/bounded retains a 9 ms TAP-to-LN head interval in
666 Flags at 127633/127642 ms. These are repeated presses, not short gaps
after a hold release.

Expressive repetition remains available: local/fused I has seven consecutive
two-key TAP chords at 141855–142490 ms with 92–114 ms spacing; OVERDRIVERS
has four three-key chords at 27149–27669 ms. Global/bounded FORViDDEN ENERZY
has five repeated left-hand two-key chords at 71020–71672 ms, with 156–170 ms
spacing and surrounding changes in grouping. The latter timing follows the
reference's approximately 162 ms pulse while adding attacks inside longer
source holds. These inspected figures demonstrate surviving repeated-action
organization; they are not a prevalence estimate or a blanket Tech/dump claim.

Global/bounded YOASOBI has coordinated held starts and shared releases;
Paganini instead becomes an almost entirely LN interpretation in the inspected
middle, while Tsuikou becomes a tap/chord stream in its inspected dense window.
These are material style choices, not reference-density matches. Their musical
and player-level preference remains unresolved. Short LNs alone were not
classified BAD.

#### Evaluation and decision

Recommended outcome: REFINE. Retain the valid native-ms/complete-row object.
The larger interval recipe is a better native-stability candidate in this
bounded panel, but its improvement over the old recipe cannot be attributed
to interval sampling alone: the training seed, objective and event exposure
also changed substantially. The factorial comparison isolates the two model
switches within the new recipe; it does not identify that cross-recipe cause.

The two global branches fail the source-likelihood improvement gate, and their
time-specific use is unproven. Global/bounded is nevertheless a reasonable
next decoder candidate because its short-head rates improve within transition
types and inspected tap/chord/LN organization survives. Keep its LN bias and
lower activity explicit. Do not add a latent, increase parameters, adopt a V3
default or claim architectural victory from this result. The immediate remaining
concrete defect is rare excessively close same-key heads, including TAP-to-LN.

## Experiment Card: context-head-prior-v1

Revision 1. Proposed; acceptance none. Question: can the already implemented
marked head-age prior remove the remaining short repeated presses from the
global/bounded endpoint while preserving its complete native arrangement?
This is a fixed decoder comparison, not new training or a scale/exponent search.

Baseline is the 84 complete global/bounded outputs above, checkpoint
b7d56ea062b3ab0d4bdf27fcfb10835a94e376e8568761e37811391a3ebc31eb, trained at
b5c5ee33cacb3950b3c9b6a4a05d9d627c46aa64. Execution/diagnostic source is the
clean descendant b757da32ec081954c37c3945305a2e8b9d2cc1b9. The 240/36 corpus,
original 42-case order, seeds 17/19, native 500 ms queries and model bytes remain
fixed. The baseline has 184509 heads, 102118 LN heads, four same-key head
intervals <=20 ms (one <=10 ms), no early failure and all charts complete.

Set only head_spacing_ms=27, retaining the implemented exponent four. This
scale comes from the earlier pinned TRAIN head-spacing audit, not tuning on
these outputs. It is a soft decoder prior, not a universal physical minimum.
Each new head contributes min(1,(age_since_same_lane_head/27)^4); CLOSE and
first heads contribute one. Rejected rows advance the observation clock but
do not enter history/state. The separate acceptance RNG preserves the proposal
stream before the first rejection. Occupied true terminals reweight legal rows
and must close all outstanding LNs. Release-to-head gaps are not penalized
as if they were head-to-head gaps.

Audit every baseline trajectory before rerunning. With no penalized proposal
and no forced terminal, the same seed's entire proposal/state path is unchanged
by this prior; scale-one acceptance consumes no acceptance randomness. Forced
terminals are rerun even if their sampled heads had unit acceptance because
their float64 reweighting path can change draws. The frozen audit has SHA
adf58f147d991b77b5d9c6cb42ec7b991dabf99ef1420b5925c580f579fbbb04:
23 potentially affected cases (six with penalties, 18 forced terminals,
one overlap) and 61 provably unchanged paths. Rerun those 23 plus two unchanged
coupling checks (666 Flags seed base 17 and glacia seed base 17). Require exact
row-byte equality in the checks before reusing the remaining 59 outputs.
Every one of the 84 outcomes remains in evaluation. Reused timings are labeled
as baseline timings and cannot become fresh latency evidence.

Primary diagnostic: zero <=10 ms same-key head pairs and at most two <=20 ms,
with a reduced per-eligible-head rate. Preserve all complete/reached-30 outcomes;
median paired head ratio must lie in [.95,1.05] and total LN ratio in [.9,1.1].
Inspect every affected case with >20% head or LN change, and inspect representative
tap/chord/LN/repetition windows plus the formerly bad locations. Large changes
cannot be hidden by the many coupled-identical outputs. A short-event-count
win alone is insufficient. Do not retune the scale if a guard fails.

Use fresh outputs generation/context-gb-prior27-s<17|19>-<train|val>, under the
expanded corpus, and record input/script/source/row hashes plus reuse provenance.
Rerun on CPU with one thread, 90 seconds/30000 proposals per case, 900 seconds
per cohort, the existing 2 GiB RAM/40 GiB disk/PAUSE guards, and a 3600-second
overall bound. Never overwrite baseline or force nonterminal releases. No fit,
TEST access, external publication or model-default adoption is authorized by
this proposed Card. Standing user authority covers the bounded experiment.

### Result Log: context-head-prior-v1

Accepted revision none; Card revision 1. Clean execution source was
b757da32ec081954c37c3945305a2e8b9d2cc1b9, using the fixed global/bounded
checkpoint and audit above. Both unaffected coupling checks reproduced exact
row bytes. The experiment therefore ran 25 fresh trajectories and reused 59
proved-invariant trajectories, keeping all 84 outcomes. It finished in
145.86 seconds with no cap or resource stop. Output owner is
expanded-v1/generation/context-gb-prior27-s<17|19>-<train|val>.

All 84 outputs complete and reach 30 heads. Observed same-key head intervals
<=10 and <=20 ms are both zero, versus one and four before the prior. Total
heads change 184509 -> 184563; LN heads 102118 -> 102003. Median paired head
ratio is 1.0; LN total ratio .998874. Eighty timed-row files are unchanged.
The minimum final-head/audio-duration ratio remains .89042.

Only four outputs change: Imaginary Waltz seed base 17, Airborne Robots at
both seed bases, and 666 Flags seed base 19. Head-count changes range from
-2.66% to +1.87%; LN changes from -4.20% to +1.51%. No case crosses the
20% mandatory composition-review threshold. The 27 ms scale/exponent were
not changed after evaluation.

Lens admitted the complete refined panel with no skips or diagnostics; bundle
hash 45a4bd240a2dad254d308cf94d19a091c6c3a6387c22be289802562dff943233.
All 269 member hashes were verified after inspection. The two earlier context
bundles also retain all 269/521 member hashes. Complete actions/articulation
and time-proportional generated/source pages were inspected at the four changed
short-head locations: 39700–42200, 101900–104400, 126700–129000 and
213000–215500 ms. Repeated same-key spikes disappear, while the surrounding
LN or tap/chord activity remains. Later LN endpoints can change after a rejected
proposal even when the original head preceded it; this is a later close decision,
not a retroactive edit to a published head. Unchanged reviewed figures retain
exact row bytes, including the representative LN and repeated-chord passages.

Recommended outcome remains REFINE because the Card has no exact human
acceptance and structural inspection does not establish listening/player
quality. The numeric decoder guards pass and support exporting this fixed
model/recipe as a playtest candidate. Do not describe the result as a universal
minimum spacing, a learned removal of every BAD pattern or a product default.

### Incremental publication and actual startup observation

Source 9e1cb7d775371fec5e1f835433f7947d987ff5de adds an optional synchronous
GenerationUpdate consumer to rollout. Each immutable message contains a complete
row or no row, fixed-through coverage and true-end completion. Open LN heads are
published before their later CLOSE rows. Resource caps do not fabricate endings;
consumer errors propagate and consumer time is measured. The source-free runner
flushes events.jsonl during generation and writes a final stop record, rather
than requiring consumers to wait for whole-song export. This is a local research
stream, not a deployed client transport or crash-durable acknowledgement service.

Twenty-nine selected generation, audio-inference, prior and context-runner
tests passed, including unresolved-hold publication, empty coverage, caps,
callback sampling parity and source-free streaming output. Training/model
probabilities are unchanged. The stream enables startup to be measured at
actual reader availability rather than an internal timestamp alone.

An isolated parent process observed each fresh uv/Python worker's flushed file
at a 5 ms polling interval, including process launch, input/model verification,
audio decode, Mel and complete-song encoding. OS disk cache was warm. CPU used
one Torch thread on this Apple M5 Mac. No other training/generation job ran.

| Input | Audio duration, s | Readable fixed 8 s, s | Readable 30 heads, s | Step p99, ms |
| --- | ---: | ---: | ---: | ---: |
| I, native-panel peak 34 heads/s | 153.861 | 1.59049 | 1.35469 | 2.5022 |
| glass beach, longest native-panel input | 441.104 | 2.06029 | 2.07284 | 2.5120 |
| YOASOBI | 242.666 | 1.46674 | 1.52967 | 2.4865 |

All three new-audio executions reproduce the frozen prior-policy row bytes
exactly, and the incrementally published rows equal the final output rows.
The producer-trace playback simulation retains minimum coverage leads of
8.037, 8.080 and 8.123 seconds when playback starts at fixed-8 publication.
Those values exclude network/client renderer costs and are not deadline
guarantees for arbitrary hardware or audio.

A separate cached-Mel 100/500 ms horizon comparison also reproduces all three
row files exactly. Generation times at 500 -> 100 ms are 5.8016 -> 4.0936,
9.8969 -> 7.5040 and 5.5117 -> 4.1378 seconds. The dense-case p99 increases
3.25 -> 4.34 ms, while the other two decline. The result motivates query sizing
as an exact-clock scheduler optimization; it does not justify claiming uniform
latency gains or adding speculative decoding. The candidate recipe stays at
the evaluated 500 ms horizon.

### Playtest artifact

The candidate owner is
`artifacts/joint-audio/20260924-expanded-v1/delivery/context-r1-playtest-v2`.
The inference-only model is 13,912,295 bytes, SHA-256
1e86b79dd1144bca282a01fb798dc304094357bff5b80deeb099982749f5d48c.
It retains the context checkpoint format, model configuration, normalization
buffers and training provenance, without optimizer or training data. It loads
through the existing source-free entrypoint with explicit head_spacing_ms=27.

Four independently reparsed playtests package unchanged sampled note geometry:
YOASOBI (2105 heads/1599 LNs), I (2745/220), FORViDDEN ENERZY (1167/374), and
Dotabata (2526/1506). FORViDDEN is a TRAIN song; the others are development VAL.
They use constant-scroll/120-BPM presentation and OD 5, without source SV,
retiming or quantization. Audio hashes match the paired files. Each archive's
contents and CRC were checked. Metadata identifies a prototype, not a calibrated
difficulty or official source arrangement.

The combined 30,316,123-byte playtest ZIP has SHA-256
6a75b601a683331d20286356a1a454895fb9a0a5889c4805d850406ff66f3277.
Its ten members include weights, README, model card, four .osz files and
evaluation/profile manifests; the model bytes inside the ZIP were reverified.
Curated product evidence is in docs/research/audio_joint_playtest_v2.md.
No remote branch or prior expert-review link was updated.

The next research questions are composition/difficulty control and actual
musical/player judgment, plus whether useful position-specific audio context
can be learned without additional unnecessary complexity. Preserve the current
candidate as a fixed comparison point; do not turn these bounded successes into
a claim that every style or future song has been validated.

### Style coverage audit and byte-identical source aliases

At clean product 98013eeb3b4a867d751bd7705c2bd41aa92b5cd0, the frozen Lens
bundle contains 204 effective human examples from 130 source identities. None
of those exact source identities occurs in the 585 TRAIN/36 VAL paired corpus.
This is not evidence that its styles are absent: alternative difficulties in
the same song group remain separate targets. Lens mapset group identifiers
must be mapped through source SHA into the catalog's song groups; direct string
comparison between these namespaces is invalid.

The first five named sources checked (Who?, death piano, Prom Queen, Good Luck,
Babe!, and Airborne Robots) have byte-identical .osu files beside their audio in
dataset/. The pinned catalog instead selects their imported scoped-style source
copies, whose parent directory has no audio. The pairing code only searches
beside that chosen source path. This systematically excludes these examples
from paired training without proving any absence of audio assets.

A read-only audit of all 490 missing-pair entries recovered all 490 through
exact source SHA equality: 427 TRAIN and 63 VAL. No differing-audio ambiguity
was found. Of the 130 human sources, 113 are recovered catalog entries; the
remaining sources are not thereby admitted. Mapset metadata locates candidates
only and does not establish pairing. The matching source bytes and their local
AudioFilename reference establish the link. Raw owner:
expanded-v1/lens-review/context-gb-prior27-v1/style-audit-v1/paired-alias-audit.json.
Its script and input hashes are embedded in that report. No model was fitted.

Thirty recovered TRAIN arrangements are in existing selected groups and use
the exact already-selected audio bytes. Nine recovered VAL arrangements also
share selected audio. Five further TRAIN entries share a metadata group but
use different audio bytes and are not eligible for that simple repair. Across
the whole audit, 131 encoded audio identities were not in the earlier readable
pair inventory; any future admission must repeat split/decoded-audio checks.
The existing corpus, exclusions, checkpoints and evaluation results remain
immutable. An alias repair must verify source bytes, reject ambiguous audio,
record the paired source separately from the catalog path and retain split
ownership. It must not silently replace the original evidence.

### Experiment Card: style-anchored-native-audit-v1

Revision 1; proposed; Accepted revision none. Standing human authority covers
the local implementation and bounded diagnostic run; this is not Card acceptance.

Question: does the fixed candidate exhibit the timing/action relationships found
in human-positive Tech and chord-repetition episodes, and can currently missing
paired targets be restored without changing the event representation? The live
branches are inadequate paired target coverage, inability to select/persist a
particular arrangement, and inadequate local conditional prediction. This pass
can identify examples and data omissions; it cannot isolate all three causes.
The closest analogues remain the original complete marked-event likelihood and
the expert's finite-history conditional-generation proposal, not onset detection.
No novel representation or learning objective is claimed.

Baseline source is 98013eeb3b4a867d751bd7705c2bd41aa92b5cd0; fixed checkpoint
is inference artifact 1e86b79dd1144bca282a01fb798dc304094357bff5b80deeb099982749f5d48c.
Use 500 ms hazard queries and the explicit 27 ms soft head prior. Existing 84
outputs complete with no <=20 ms same-key head interval; that is a mechanical
baseline, not a style score. No aggregate Tech-preservation baseline exists.

The intervention is evaluation coverage only: add Who?, death piano, Prom Queen,
and Good Luck, Babe! audio-only BOS rollouts at seeds 17 and 19. Reuse the two
already frozen Airborne Robots outputs (actual seeds 33 and 35). The source
identities are, respectively, 07a5f9448345e50ed2e9282851b3423898327c488e1e8e28ceefe239f27be8f3,
9b422f0fbca360721bf1d9811095fbfe5b130bd3a7f8f739e4a4fbddb3b7a191,
bd453e8f29b360900104ac0e6a2f385b2f44d1a57e64949c70158bdc75351da6,
c4cad1f8b55f608be51e3a747f5305ac09ca42a7c9ab9808e498f503465e3fde,
and 5baa73d5a65dd3b6c9568e3901e12c7cf2c0817e014a1de76447cbda3cc63c42.
First four are catalog TRAIN diagnostic songs; Airborne is development VAL.
No TEST target is admitted. Source-free generation receives complete Mel only;
reference .osu data is used later for inspection and optional conditional scoring.

Review the complete human scopes and entry/exit contexts with frozen Lens
actions/articulation and every time-view page. Include high-confidence ANiMA
Tech-absent contrast human-ca3a981097a951aa6a179e51: rapid bursts alone do not
establish Tech. Compare concrete rhythm, attack-group motion and LN relations;
source matching is not the required output. Missing labels remain unreviewed.
No dump-positive claim is possible from this taxonomy or chart-only inspection.

Decision criterion: a documented generated episode must show a definite relation
before reporting that relation as represented; density, irregularity and LN count
alone cannot pass. If only generic activity is found, retain expressive coverage
as unresolved and test restored targets before adding model capacity or latent.
Source likelihood, if computed, is diagnostic only. Mechanical guards retain
every empty/capped output, actual head/LN composition and head-to-head transitions;
release-to-head is separate. Report any <=20 ms same-key head interval rather than
discarding the case. No threshold on a proxy metric determines playability.

Implementation is a pinned explicit source-alias input for paired preparation,
with tests for byte mismatch, escaped paths, differing-audio ambiguity, input
consumption and catalog identity preservation. It changes admission only when
explicitly enabled; existing model probabilities and corpus bytes stay fixed.
Record a clean implementation descendant before new model execution.

Run eight fresh CPU rollouts on Apple M5 with one Torch thread, each capped at
180 seconds/30000 proposals, with existing 2 GiB available RAM, 40 GiB disk and
PAUSE checks. Full audio uses the verified canonical Mel. Overall diagnostic
compute bound is 1800 seconds, no external network or parallel accelerator job.
Write a fresh style-audit-v1/generation directory; never overwrite prior results.
Record source/audio/model/row hashes and limits. No fit, new prior tuning,
reference-seeded generation, remote publication or human-label editing occurs.
Actual audio listening is unavailable to this model/tool context; visual and
event-structure review must not be represented as listening or player testing.

### Result Log: style-anchored-native-audit-v1

Card revision 1; accepted revision none. Clean implementation/execution source
de2ff6207158edf1e1cd71ae1697b4195ba73854 adds explicit pinned aliases to
preparation, retains canonical catalog identity, and verifies the paired copy
again when loading. The selected data/config tests passed (18 tests), and the
packaged Hydra inspection exposes both preparation-only fields. No model or
sampler parameter changed. Alias manifest SHA is
7ffaf7e1394aec6f3be6e35450ce64beca9f6245efc061e7e149f25a3cbbd49b.

The eight new BOS rollouts all complete, exceed 30 heads and have zero <=20 ms
same-key head intervals. They use the exact checkpoint and recipe in the Card.
Their head counts at seeds 17/19 are Who? 1604/1288, death piano 1964/1656,
Prom Queen 1914/1563, Good Luck, Babe! 1734/1439. Their whole-chart LN head
fractions are .756/.787, .862/.811, .597/.680 and .365/.190. Individual runs
take 3.24–6.35 seconds after imports/Git checks; these are not cold-start profiles.
All outputs remain in style-audit-v1/generation. Lens admits all eight; its frozen
bundle manifest SHA is 36b502144d6a2e1403e5783a1b5e6ca6116f1380320a1ae4906539af31a28a48.

All source/generated context action rows and every time-view page were inspected:
Who? 73000–87000, death piano 102953–107130, Prom Queen 75400–78700,
Good Luck, Babe! 127823–131852; Airborne 147208–150258 and the ANiMA absent
contrast 97000–98600 use the unchanged prior bundle. Generated and human
interpretations are not interchangeable labels. No human records were edited.

The pure-tap death piano episode contrasts with strongly LN-based output at
both seeds. The Prom Queen reference has recurring same-column presses inside
changing two-/three-key chords at roughly 208 ms spacing; both outputs replace
much of that organization with staggered holds. These crops do not supply a new
positive demonstration of the reference chord-repetition relation. Who? outputs
do contain interleaved holds and changes of press/release roles, but retain LN
activity through much of the source's later tap passage. Good Luck's reference
includes cross-column 8 ms and 32 ms staggered attacks; these are distinct from
same-column repeated heads and remain legal under the current representation
and per-lane prior. Neither output is a faithful reference realization, which
is not itself a failure: this model has no requested style/difficulty condition.
Its generic activity/irregularity is insufficient to establish Tech preservation.

Recommended outcome is REFINE. The scoped audit shows composition/selection
differences that short-head diagnostics cannot resolve. It does not show that
LN alternatives are necessarily bad, that all taps must be retained, that every
generated Tech episode is absent, or that a latent is necessary. Dump musical
intent remains untested. No actual listening/player verdict is available.

As a data-preparation follow-through, a new immutable corpus at
artifacts/joint-audio/20260924-alias-restored-v1 restores the 30 exact-audio TRAIN
alternatives. Its manifest SHA is
4ad9abfd0ae7798e0a85b96a5dbecdaefd2a81f78bf181438407442b964118e1.
All 585 old TRAIN charts, 240 TRAIN groups, 36 VAL charts, 276 audio/Mel assets
and normalization bytes remain; TRAIN now has 615 separate arrangements. Five
same-group/different-audio candidates are explicitly rejected. Canonical loading
verifies the complete result. This corpus has not been used for a fit.

### Experiment Card: style-source-row-probe-v1

Revision 1; proposed; accepted revision none. Standing user execution authority
covers this local diagnostic. Clean source is
de2ff6207158edf1e1cd71ae1697b4195ba73854; model/normalizer and the four TRAIN
audio identities are those frozen by style-anchored-native-audit-v1. No fit or
new generation is involved. The intervention is the observed history used for
diagnosis: real reference versus each separately generated chart's own prefix,
at that chart's own event times. This is not a paired causal prefix intervention
and never attaches reference future labels to an altered prefix.

Question: is the candidate's strong native LN composition also present in its
row distribution under genuine tap-heavy source prefixes? If source-conditioned
expected LN fractions are much lower than native-prefix fractions, simple lack
of tap support is an inadequate explanation. If both are high, conditional
prediction/data coverage becomes a more immediate bottleneck. These observations
cannot establish which architectural remedy is necessary. The closest analogue
is the expert's distinction between source likelihood and autoregressive rollout
reliability; this is measurement, not a novel objective.

Use every event in the four human scopes: Who? [75000,85000), death piano
[103453,106630), Prom Queen [75838,78338), Good Luck [128423,131252).
For each chart, replay its complete true prefix, project only R0 causal content
and exact state, and score the existing legal full-row distribution at the event
time with complete encoded audio. Expected heads and LN starts come from the
256-row probability table; report ratio of summed expected counts, alongside
observed counts and conditional row NLL. Do not average per-row ratios, use a
style label as a target, or interpret native self-likelihood as quality.

Run CPU one thread in a fresh source-row-probe-v1 artifact below style-audit-v1,
maximum 600 seconds, 2 GiB RAM/40 GiB disk guards. No random seed is needed for
deterministic inference. All selected events remain included; no checkpoint
selection. Stop on nonfinite legal scores, invalid source timing/occupancy,
changed input hashes or dense/full-audio versus canonical-crop row-score
disagreement above 1e-4 on one reference and native query per song. Record all
input/output hashes and sample counts. No positive numeric playability threshold
is claimed; report the direction/magnitude per episode and preserve ambiguous
cases. The existing native charts are fixed and cannot be worsened by the probe.

### Result Log: style-source-row-probe-v1

Card revision 1; accepted revision none. Execution used the clean pinned
de2ff6207158edf1e1cd71ae1697b4195ba73854 source, fixed inference checkpoint,
four exact audio/source identities and scopes above, and all 795 selected
source/native events. The deterministic CPU run took 31.03 seconds, hit no
resource cap and performed no fit or new sampling. Twelve full-audio/canonical-
crop checks have maximum legal log-probability difference 3.8147e-6, below the
1e-4 guard. There was no protected-field deviation. Output owner is
style-audit-v1/source-row-probe-v1/result.json, SHA-256
9305057f4284d73dfa9a3ca756a178818f025b43f30f76ba1fed47cdfc6106b6.

| Episode | Source events | Native events 17/19 | Expected LN/head source | Expected LN/head native 17/19 |
| --- | ---: | ---: | ---: | ---: |
| Who? | 163 | 168/171 | .65033 | .71969/.75002 |
| death piano | 20 | 75/51 | .00732 | .88828/.82263 |
| Prom Queen | 12 | 51/25 | .00874 | .85673/.75091 |
| Good Luck, Babe! | 22 | 22/15 | .00709 | .03579/.11446 |

Source conditional mean row NLL is respectively 1.35290, 1.48214, 2.11355 and
2.48810 nats/event. These are separate scoped source predictions, not a combined
quality score. Native self-likelihood is retained in the artifact only for
diagnosis. The source's actual LN fraction is 2/3 for Who? and zero otherwise.
The result contradicts a simple claim that the row decoder cannot express or
locally prefer tap-heavy arrangements. It does not identify a single causal
history feature: source/native event times, occupancy, content and clocks differ.
R1 transfer and TRAIN source exposure also prevent unseen-song claims.

Recommended outcome remains REFINE. A next discriminating probe should replay
a real tap-heavy prefix and then sample its continuation with full audio, without
reference future labels or a forced tap-only decoder. This can separate failure
to initiate an arrangement from inability to sustain it under free timing/actions.
Restored paired targets offer a separate data-only intervention; do not conflate
that experiment with adding persistent intent or increasing model capacity.
No new architecture extension, training run or lifecycle transition is adopted
by this result. Product evidence is summarized in the playtest candidate guide.
Both Lens bundles still match every frozen member hash (269 and 151 files).

### Experiment Card: observed-prefix-continuation-v1

Revision 1; proposed; accepted revision none. The previous goal turn made
material progress through the alias fix, eight native outputs and the 795-event
conditional probe. Standing user authority covers this diagnostic implementation
and execution. Clean baseline is 4b56f3ab72cd773838e212bd4d5b7e078ae267aa;
fixed model is inference SHA
1e86b79dd1144bca282a01fb798dc304094357bff5b80deeb099982749f5d48c.
Audio, source charts, canonical Mel, two BOS trajectories and their hashes are
frozen in style-audit-v1/generation/freeze.json and result.json. No weight changes,
fitting, TEST access, remote publication or human-label changes are included.

Question: can a real observed prefix maintain a tap-heavy arrangement when both
time and action generation are free? The selected branch is that initiation or
early arrangement selection contributes to native LN composition. The competing
branch predicts rapid return to LN-heavy output despite genuine source history.
The closest analogue is conditional autoregressive continuation and the expert's
distinction between valid likelihood and reliability near sampled trajectories.
This is a controlled prefix probe, not a new representation or training objective.

For each of Who?, death piano, Prom Queen and Good Luck, Babe!, use two exact
observed-coverage cutoffs. Early is the complete row crossing 30 reference heads;
mature is one millisecond before the previously reviewed human scope begins:

| Song | Early cutoff, ms / prefix rows | Mature cutoff, ms / prefix rows |
| --- | ---: | ---: |
| Who? | 5485 / 27 | 74999 / 745 |
| death piano | 3399 / 23 | 103452 / 894 |
| Prom Queen | 5915 / 23 | 75837 / 380 |
| Good Luck, Babe! | 5038 / 19 | 128422 / 1489 |

The early threshold is 30 heads, not 30 rows. Every row is retained atomically.
At each cutoff compare three observed prefixes: reference, frozen native seed17,
and frozen native seed19, all through the same absolute audio clock. Sample each
condition with fresh suffix RNG seeds 37 and 41: 48 continuations. The causal
intervention is the entire observed prefix, including exact clocks/occupancy,
not an isolated content-history feature. Prefixes are not edited and no source
suffix is used as a label. Only actual prior rows and known empty coverage are
supplied; unknown LN endpoints remain absent. Audio is complete in every arm.

Implement an explicit observed-prefix argument on the same rollout engine, with
exact replay of the full prefix and neural prefill bounded to the actual finite
receptive field (including the true predecessor gap). Keep default BOS sampling
unchanged. Prefix rows must be distinguished from new rows in receipts and
exports; callbacks publish only new rows. New-row caps and first-30 latency count
sampled rows/heads, not supplied ones. Startup for a diagnostic prefix means eight
seconds beyond the supplied clock and is not audio-only startup evidence.
Conditional waiting starts after observed coverage with a fresh survival draw;
this is conditional resampling, not bitwise RNG restoration of the old rollout.

Tests must cover finite-cache/full-replay parity, exact old clocks beyond the
neural window, first possible millisecond after known empty coverage, an open
hold crossing that coverage, caps without fabricated releases, invalid prefix
times/occupancy, unchanged BOS draws and callback scope. Rerun both original
death piano BOS seeds and require exact prior row hashes before the experiment.
Record a clean implementation commit before new model execution.

Primary diagnostic is actual LN heads / all generated heads in the first eight
seconds after each mature cutoff for death piano and Prom Queen. Also report
0–8 s, 8–32 s, later suffix and the exact human scope for every condition/song,
along with head/chord composition, first LN time and count, earliest eight-second
bin exceeding 50% LN, all head-transition bands, release-to-head gaps and caps.
Reference-prefix continuations <=10% LN at both seeds, while a same-song native
prefix condition is >=50%, support short-horizon conditional arrangement
maintenance. Reference-prefix >=50% at both seeds argues against initiation-only
explanations. A window needs at least ten generated heads for either diagnosis;
otherwise activity is insufficient, not a sparse-model win. Mixed seeds and
intermediate fractions are inconclusive. These thresholds classify a bounded
mechanism probe, never style, difficulty or BAD patterns.

Inspect all four critical mature reference-prefix episodes in Lens with entry/
exit context, plus the Who? LN reference condition and Good Luck control. Inspect
additional contrasts if needed to explain a mechanism change; keep uninspected
dimensions unreviewed. Valid LN alternatives are not automatically bad, and
source similarity is not a playability metric. No listening capability is claimed.

Use CPU one Torch thread on Apple M5, 500 ms hazard queries, fixed 27 ms prior,
30000 new proposals and 180 seconds per continuation, full true audio terminal.
Run sequentially with 2 GiB available RAM, 40 GiB disk and PAUSE guards; total
compute bound 1800 seconds. Fresh artifact owner is
artifacts/joint-audio/20260924-expanded-v1/observed-prefix-v1. Never overwrite the
BOS outputs or candidate. Record all prefix/source/audio/model/script/row hashes.
No training is authorized by this Card; a subsequent training intervention must
have its own bounded comparison. Missing fit coverage and small conditional
sample size remain confounders even if the diagnostic threshold is crossed.

### Result Log: observed-prefix-continuation-v1

Card revision 1; accepted revision none. Clean implementation/execution source
965c70aa267027ef486cd98f560b926a9e441bbd adds ObservedPrefix to the existing
rollout, replaying all exact facts and only the finite neural suffix. The true
predecessor gap survives truncation. Supplied rows do not consume generation
caps or first-30 counts and are not published again. Conditional exports identify
supplied versus sampled origin. The default source-free path remains BOS.
Forty-one selected tests passed, covering prefix/cache/replay behavior, native
generation, head prior, audio inference and existing cache/crop parity. The
original two death piano BOS trajectories reproduce exact row hashes before
the prefix comparison. No model probabilities or corpus inputs changed.

The experiment completes all 48 planned continuations in 170.42 seconds on CPU
one thread, with no resource stop, empty suffix or newly generated same-key head
interval <=20 ms. Full receipts, prefix files, row hashes, composition windows,
transition types and separate release-to-head gaps are retained under
artifacts/joint-audio/20260924-expanded-v1/observed-prefix-v1. The freeze hash is
9c06ea1cc304883d66d1b7700c064b9778e9445424933669d2401eef589eda06;
result hash is
9e2496baa80346e1628f42df17ad708d0d60afe8c6a90263c6a06064d73a3521.
No protected field changed. A scientific plot of all critical conditional
trajectories is prefix-ln-trajectories.png; it excludes supplied prefix heads.
No cold-start or source-free speed claim is taken from these conditional runs.

The primary mature-prefix diagnostic crosses the stated <=10% versus >=50%
threshold at both critical songs. In the first eight seconds, death piano's
reference prefixes yield 30 heads/1 LN and 43/0, while the four native-prefix
suffixes have 86.57–96.43% LN heads. Prom Queen's reference prefixes yield
77/0 and 101/0, while native-prefix suffixes have 67.86–89.42% LN heads. Every
critical window exceeds the minimum ten-head activity guard. This establishes
short-horizon conditional tap capacity under the tested complete states.

Persistence is conditional and variable. For death piano, early reference
prefixes already reach 14.77%/66.67% LN in the next eight seconds and
80.49%/94.53% in the following 24 seconds. Mature reference prefix suffixes
start near zero but reach 70.79%/35.82% LN during seconds 8–32. Prom Queen's
early reference suffix seeds diverge; mature reference suffixes remain almost
all tap through the rest of the song. Good Luck reference suffixes start with
zero LN but can become LN-heavy later. Who? is also variable: 37.5%/4.30% LN
in the first eight seconds despite the same reference prefix, while its reviewed
ten-second human scope contains 58/121 and 8/111 LN heads. A real short seed
alone therefore does not solve reliable style selection or later persistence.

Lens admits all 48 outputs. Frozen bundle hash is
a514745730fb3e6027627782bc5e338c123de71c63bf9864249ccc4e13cda479;
all 191 member hashes still match after inspection. Ten planned/contrast scopes
were read completely through action/articulation pages and all 28 time-view
pages: mature reference at both seeds for all four songs, plus native17/seed37
for death piano and Prom Queen. Entry context before the cutoff is explicitly
observed, not counted as generated evidence. No human annotation was changed.

In Prom Queen's generated scope, reference-conditioned seed41 produces repeated
[0,2] pairs at 75850/76046, then [0,1,3] at 76259 and [0,2,3] at 76468 ms.
Same-column recurrence continues inside changing chords; both seeds retain
26/31 heads versus the reference's 29. This is a scoped demonstration of the
conditional model's chord/repetition capacity, not an unconditional success.
Death piano instead has 14/15 all-single tap heads versus 23 source heads with
mixed chords; reduced LN fraction does not preserve the full Tech relation.
Who? seed37 later exhibits interleaved holds with independent release/press
roles, while seed41 spends much of the scope in tap motion. Generic variability
does not establish Tech, and none of these chart-only probes establish dump
musical intent, listening quality or player-specific difficulty.

Recommended outcome is REFINE because the Card remains unaccepted and the
conditional result is not the requested source-free system. Same-cutoff
reference/native comparisons intervene on the whole prefix, including occupancy
and clocks. Early/mature comparisons additionally change musical position and
cannot prove a history-length cause. Undertraining and paired-target coverage
remain viable explanations. Human prefix quality must not be reported as model
generation, nor can low LN output be assumed superior to an alternative LN chart.
Curated evidence is recorded at product 55f86faf9083da66d4d648623ac09ccf8f4797fb
in docs/research/audio_joint_playtest_v2.md.

### Research refinement: continued fitting versus persistent arrangement condition

The next comparison should keep the repaired 615-arrangement/240-song TRAIN
corpus, 36 VAL charts, full audio, finite history and exact state fixed in both
arms. One arm continues the current model; the other adds only a small persistent
categorical condition. This controls for further fitting on restored targets.
Relative to the old candidate, any shared improvement combines restored targets
and additional training; those two causes would not be independently identified.
Do not start this training until an exact proposed Card records initialization,
objective normalization, sampling plan, fixed endpoint, code-usage and native
quality guards. Standing implementation authority remains separate from Note
acceptance; no new run is claimed here.

The representation under consideration is

$$
p(Y\mid X)=\sum_{z=1}^{4}p_\psi(z\mid X)\,p_\theta(Y\mid X,z).
$$

One code is drawn once per whole output from an audio-only prior. Both hazard
and full-row heads read it throughout the run, retaining all physical history
and LN obligations. This is a two-bit arrangement choice, not four named styles,
a difficulty rating, a section planner or a future onset skeleton. It gives a
small explicit place for future control/readout work without requiring it now.
A neutral single-state control must retain the original conditional decoder.

An optional recognition network can read full audio plus a low-dimensional
summary of the reference arrangement during training. Only its four-way
categorical distribution crosses into decoding; reference statistics themselves
must not become inference inputs. Prefer exact enumeration of four states to
sampling/straight-through estimators. For a source-time interval sampled with
probability p(j|Y) and total integer-clock duration D=(T+1)/1000 seconds,

$$
\widehat{\mathcal L}=
\sum_z q_\phi(z\mid X,Y)\frac{\ell_j(z)}{p(j\mid Y)D}
+\frac{\mathrm{KL}(q_\phi(z\mid X,Y)\|p_\psi(z\mid X))}{D}.
$$

This preserves the existing per-chart-time normalization of reconstruction and
scales one whole-chart KL consistently. Do not charge a fresh whole-chart KL
at every event, omit survival, merge alternative charts, or assign a new z at
interval boundaries. A deterministic full-chart posterior summary permits
unbiased interval estimation of the reconstruction expectation, but its encoder
may still be a restricted approximation to the optimal posterior.

Closest analogues, rechecked from primary sources:

- [Sohn et al., conditional generative models (2015)](https://proceedings.neurips.cc/paper_files/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf)
  separates an input-only prior from a training recognition network that reads
  the target. The transferable mechanism is conditional multimodality and its
  variational objective. Their Gaussian/image decoder is not our categorical
  native-time autoregressive decoder; their reconstruction/prediction warning
  reinforces evaluating audio-prior samples separately from posterior outputs.
- [Roberts et al., MusicVAE (2018)](https://proceedings.mlr.press/v80/roberts18a/roberts18a.pdf)
  demonstrates latent organization of musical sequences and the risk of a strong
  autoregressive decoder ignoring it. Its bar hierarchy, reset boundaries and
  quantized outputs are not transferred. Our small whole-song choice is much
  narrower than its structural latent model.
- [He et al., lagging inference networks (2019)](https://arxiv.org/html/1901.05534)
  distinguishes model/inference collapse and shows why KL magnitude alone is
  insufficient evidence of useful latent information. Their optimization remedy
  is not adopted automatically. Code/posterior usage, correct-versus-permuted
  code predictions and audio-prior native behavior need explicit checks here.

Provisional outcome TEST for designing this two-arm comparison. No novel
objective claim is made: it is a bounded conditional-mixture adaptation to the
timed-row task. A decoder that ignores the code, only improves posterior
reconstruction, produces four fixed loops, or loses real LN/chord/Tech relations
fails the purpose. Better code separation or lower likelihood alone cannot win.
If the plain continuation resolves the native behavior equally well, additional
structure is unnecessary at this stage. The prefix study motivates the test;
it does not establish that four states are sufficient or required.

### Experiment Card: persistent-intent-comparison-v1

Revision 1; proposed; accepted revision none. Standing human authority covers
implementation, local tests and the bounded comparison. Clean baseline product
is 55f86faf9083da66d4d648623ac09ccf8f4797fb. Implementation is exploratory;
record its clean OID before execution. No publication or lifecycle adoption.

Question: does a persistent two-bit arrangement condition improve source-free
composition/expressive variety beyond additional fitting on repaired targets?
The live alternatives and primary-source analogues are recorded immediately
above. The single intervention is K=4 conditional latent versus the unmodified
K=1 decoder. Both initialize every common tensor, including normalization and
full-audio context, from candidate SHA
1e86b79dd1144bca282a01fb798dc304094357bff5b80deeb099982749f5d48c.
Both reset AdamW and train the same repaired 615 TRAIN arrangements/240 songs,
36 VAL targets, manifest SHA
4ad9abfd0ae7798e0a85b96a5dbecdaefd2a81f78bf181438407442b964118e1.
No TEST or alternative-arrangement union labels. Old candidate baseline is 84/84
complete native outputs, zero observed <=20 ms same-key heads, with the scoped
LN preference and conditional-prefix results above. New protocol likelihoods
are not yet measured; do not relabel older-query NLL as their baseline value.

The frozen plan intent-comparison-v1.json has SHA
34ef9751df4571f516ef2b670b3b0f4b398c212fdd16adcee1a029bd9168eb14:
1200 updates, two songs/update, two 8000 ms intervals/song, 4800 intervals,
37,253,258 scored milliseconds, 612 distinct TRAIN arrangements. Model seed
230941, plan seed 230942, VAL seed 230943. Canonical make_protocol and every
source interval identity are identical across arms. Epoch-like event exposure,
not only update count, must be reported; K=4 costs four decoder-head evaluations.

K=4 adds 4x128 centered code offsets to the existing coarse-context coordinates,
an audio prior 128->32->4 and recognition MLP 8->32->4. Code offsets initialize
to zero, prior logits to uniform; recognition starts from the fixed ordinary
random initialization. The initial four conditional decoders must therefore
match the control. The eight complete-reference descriptors are log head rate,
LN/head fraction, four attack-row chord-size fractions, occupied lane-time
fraction and log median LN duration. Recognition is deliberately restricted to
these descriptors; it is a valid but limited q(z|Y) family. Descriptors never
reach native generation. No code is named as a style, and no annotation labels
or BAD-pattern reward enter training.

Enumerate all four codes exactly. Share interval audio/history encoding, then
compute existing event/survival/row NLL for each code. Apply the per-second ELBO
above with beta=1 and KL/D, no annealing, free bits, inference inner loop or
scheduled sampling. A real chart has one q distribution irrespective of its
sampled interval. Inference samples one code from the full-audio prior with
separate seed generation_seed XOR 0x17C0, holds it for the song, and records
code/probabilities. Explicit fixed-code diagnostics are separate from prior
samples and posterior reconstruction. Full physical replay and legal support
remain unchanged. Default K=1 output must reproduce existing BOS row bytes.

Common audio/timing weights use learning rate 1e-4; inherited R1 modules
temporal/exact/fuse/joint/route_residual/release_residual use 1e-5; the three new
intent modules use 3e-4. AdamW weight decay .01, gradient clip 1. No best-model
selection: fixed final 1200 updates, evaluations at 0/300/600/900/1200. Keep
checkpoint audio normalization unchanged. A 32-update preflight per arm first
checks finite losses/gradients, nonzero code learning, identical exposures and
resource use; it uses fresh run names and does not select a learning recipe.
Stop on invalid causal inputs, nonfinite loss/gradients, source/hash mismatch,
or fewer than 2 GiB available RAM / 40 GiB free disk. After a passing preflight,
start fresh from the same initial candidate for main endpoints.

Evaluate the shared 180 fixed VAL intervals using expected reconstruction plus
whole-chart KL/D; label this a negative-ELBO estimate, not exact marginal NLL.
For the six canonical stratified VAL charts pinned in intent-comparison-summary,
sum each code's NLL over the complete source clock before one prior logsumexp;
never form a fresh mixture independently per interval. Report exact whole-chart
marginal NLL/s and posterior-versus-prior diagnostics. The K=4 likelihood guard
allows at most 3% regression versus the newly trained K=1 control; a bound gap
alone is not proof of true marginal regression, so consult exact whole-chart
values before rejecting on that ambiguous observation.

Native evaluation retains the existing 42-song/two-seed order for both endpoints
and the four-song/two-seed style panel. Add all four explicit codes at event seed
17 on the 12 canonical stratified VAL charts pinned in the summary. A useful
condition should produce a >=.20 LN-fraction range or >=1.5 head-count ratio
across codes on at least six of those 12 songs, and give >=.10 prior probability
to at least two codes on at least six. These are usage diagnostics, not quality
scores; fixed loops, sparse charts and loss of coordination cannot pass on code
separation alone. If the code is ignored or only target-posterior reconstruction
improves, reject this extension rather than enlarging the codebook automatically.

Native guard: preserve all complete/reached-30 outcomes in the original 84-case
panel, no <=10 ms same-key heads and at most two <=20 ms; report eligible rates
and transition types. The median paired head ratio versus K=1 must remain in
[.7,1.3]; inspect any case outside [.5,1.5] and all material LN composition
changes. Review full Lens contexts for Tech, chord repetition, tap flow and LN
coordination across the prior-sampled style panel and fixed-code contrasts.
Do not require reference replication or treat LN alternatives as bad by default.
The only affirmative decision is bounded usefulness of the added condition
after these native and qualitative checks; likelihood or code diversity alone
cannot establish playability. K=1 matching quality argues for the simpler model.

Use the packaged joint_audio_intent Hydra entrypoint, states=1 and states=4,
on this Apple M5 with explicit mps, one CPU thread, sequential training only.
Maximum 7200 seconds per main arm, 600 per preflight, 14400 overall training;
native CPU runs use 180 seconds/30000 proposals per chart, the fixed 500 ms
query and 27 ms prior, and 3600 seconds overall evaluation. Honor PAUSE files.
Fresh owner: repaired-v1/intent-training/{preflight|main}-k{1|4}-v1; no overwrite
or implicit resume. Freeze resolved config, plan, input/weight/source hashes,
state usage and actual compute. Stop and revise the Card for a behavioral
change; behavior-neutral memory engineering requires equivalence evidence.

### Implementation and revised execution priority

Product 60312912b121dc1de7f5dbc34b2cba9ecd4c5bd5 implements the persistent
intent comparison, shared interval encoding, fixed/ prior-sampled source-free
inference, typed Hydra configuration and objective tests. The intervention adds
5192 parameters, totaling 3467020. It preserves the original decoder at zero
code offsets and keeps full reference descriptors entirely on the recognition
side. The initial selected run passed 59 tests and 22 package subtests; the
subsequently added source-free/forbidden-recognition test passed with the seven
other intent tests. Shared-backbone gradients match separate code evaluations;
fixed-code future-label isolation and query-partition parity are tested.

No intent preflight or model training has started. The user then asked how much
long-form instability comes from R1-restored and whether its staged targets or
evaluation are faulty. This changes research priority: audit that attribution
before executing the latent comparison. The Card remains proposed revision 1;
there is no acceptance, model-quality claim or goal pause. The implemented
comparison is retained as a possible later experiment, not assumed necessary.

### Result Log: actual R1 lineage and transfer audit

Exploratory read-only audit; accepted revision none; no generation or fitting.
The released checkpoint's config locates the original stable asset owner
r1-restoration-20260920-v1/run. Its actual ledger, six checkpoints and 96 complete
native row files remain available. Their hashes and original evaluation pins
were checked rather than substituting historical candidate results. The owning
script/result is artifacts/joint-audio/20260924-r1-lineage-audit-v1; result SHA
09b58360f7f08b2c75002d17e3fcab04d8921e9e31d76d4a28e20bf99ae297a5.

Actual stage checkpoint identities:

| Stage | SHA-256 |
| --- | --- |
| Plain 4.5M | 816d9daf0eb387bb65a661eba348db5c90358a37d68739f71d3223073167c18e |
| Seed 5M | ad7844d800de879a1cb68d9c883a436bedf5f61d6fdd2aadf64a3aed69ccf394 |
| Memory 6M | 3262f210ff6f67837f10284a4e1c54ec408354c3e1c325fc0f95dc53dcca1a23 |
| Routing 6.25M | 9464045a711c1b10a04f7b3cfb76999d70042857ba9f898408f291a1c1ccdc1e |
| Release 6.5M | 8000beb0dee81b92f0252fcec876823afd60fe33a182fb344ae0bed02d03f991 |
| Response 6.75M | 4b3ec1561e33d0ebe2756cfe13571ec414fd5bb470b430f0c578545863115f70 |

The checkpoint training source is cdbc6870d9e6471a5dcb75d8a77f55c30544bf78;
the relevant restoration/bounded-model code has no diff between that source and
the release tag. The actual automated readout has eight VAL songs, seeds 17/23,
16 outputs/stage, with 181.727–277.537-second suffixes. Source likelihood uses
24 VAL windows and 6144 required onsets; the complete input also contains an
equally sized TRAIN readout. Source-native conditions include R/H timing, real
initial seed and its known crossing endpoints. They do not test audio-only BOS.

| Stage | VAL NLL/onset | Mean native LN/head | LN-fraction MAE versus reference | HH <30 ms | RH <30 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain | 1.771509 | .431218 | .3130 | 6 | 306 |
| Seed | 1.765174 | .318110 | .1739 | 3 | 87 |
| Memory | 1.727303 | .263538 | .1418 | 2 | 35 |
| Routing | 1.731063 | .274222 | .1333 | 2 | 68 |
| Release | 1.728967 | .276722 | .1472 | 2 | 67 |
| Response | 1.738584 | .192471 | .0724 | 5 | 7 |

Fractions/errors average the 16 case values (eight groups with two seeds).
Reference mean LN/head is .185998. These are composition diagnostics, not a
uniquely correct output requirement or playability labels. HH and RH count
individual new heads with a strict less-than gap; they can overlap and must not
be added as disjoint events. The final local improvement is mostly RH 67->7,
while HH changes 2->5. The response target explicitly uses the union of head
and release age below 30 ms plus an optimistic next-two-H continuation; it does
not model a complete player experience. A source CE arithmetic/label bug has
not been established by this audit.

Plain-model LN/head means rise .2723->.5396 across first/last required-onset
quarters, versus source .1378->.2111. The final policy gives .1913->.1667.
The longest three-unchanged-hold run rises 2->31 after head routing and returns
to 2 after release routing. These observations show local changes interacting
with later states; they do not imply every stage uniformly harms or improves
semantic organization. No fresh Lens/player judgment is added by the recount.

The current audio transfer copies 2444688 parameters, omits 523776 in seed,
landmark memory and row-consequence modules, and discards 115968 exact-input
weights (1435->529 input features). These counts are not causal contribution
fractions. All 227 resulting global/bounded initial state tensors from release
6.5M and response 6.75M are bit-identical at the same constructor seed; their
name-and-tensor SHA is 1b9ef122bd7ac55128ac3733e11cf07c31afb82780845cb636f1311e7bbbd3eb.
Thus the last response fine-tune has no direct parameter effect on the current
joint initialization. Its full-policy behavioral correction was omitted, so
this result does not make its removal harmless. Head/release residuals and the
earlier seed/memory-trained backbone do transfer.

The restoration worker advances on completed computation and mechanical native
execution; its final quality_status is explicitly
requires_longform_ln_tap_and_local_response_review. This is not a semantic
acceptance gate. Historical long-chart results on other checkpoint identities
cannot establish coverage for the actual restored bytes. A later independent
manual review, if present, needs its own evidence and is not inferred absent
merely from this ledger. The online model card was unavailable through the web
tool; no claim depends on reading it.

Recommended outcome REFINE. The missing causal comparison is full R1 versus
its transferred conditions, followed by equal-budget joint learning from distinct
lineage endpoints. Do not assign an additive R1/audio blame percentage from
different tasks. A useful first direct probe can start from release 6.5M and
zero only seed_residual.2.weight and long_memory.output.weight, preserving
source R/H, physical seed/known seed tails, all other weights and original
generation seeds. That measures neural-condition dependence under the original
task; it still does not remove future timing inputs or reproduce audio-only BOS.
Require baseline exact-row reproduction before using it. If later comparing
plain 4.5M, memory 6M and release6.5M joint initialization, assign learning rates
by the same module families in every arm, not by which tensors happened to copy.
Otherwise absent routing modules would silently receive a different rate.
Release 6.5M and response 6.75M are redundant transfer arms and should not both run.

The curated audit and interpretation limits are in
docs/research/r1_transfer_stability_audit.md at
0848d8d6898939d850cf3bd7bbf4332ab121dceb. Goal remains active; model-quality
completion is unproven. All model execution handles from prior probes are terminal.

### Experiment Card: r1-neural-condition-removal-v1

Revision 1; owning Note 2026-09-23-audio-skeleton-r1-integration; proposed;
accepted revision none. Standing local execution authority permits this
exploratory probe. The persistent-intent comparison remains deferred.

Question: does deleting the persistent seed and landmark-memory readouts of the
release-stage R1 substantially change its long-form allocation on the original
task? The selected mechanism is conditional co-adaptation: the inherited
backbone was fitted with neural readouts that the audio transfer omits. The
closest analogue is the existing original R1 seed/memory restoration itself;
this probe removes their outputs at inference without changing timing, physical
replay or fitting. Alternatives are remaining full-policy weaknesses and new
audio/timing-training dynamics. They cannot be separated by parameter counts.
The result can prioritize restoring equivalent conditioning versus testing
initialization lineage; it cannot choose a playable model by composition alone.

Clean runtime baseline: 0848d8d6898939d850cf3bd7bbf4332ab121dceb. Original
training/generation source: cdbc6870d9e6471a5dcb75d8a77f55c30544bf78.
Checkpoint: release 6.5M, SHA
8000beb0dee81b92f0252fcec876823afd60fe33a182fb344ae0bed02d03f991.
Evaluation JSON SHA
4cbb5992f3dd375316fb44a8909c476efa5270307b8603a0f7826ce84c50dd46.
Use all 16 original native VAL cases, eight songs paired at seeds 17/23.
The audit result SHA
09b58360f7f08b2c75002d17e3fcab04d8921e9e31d76d4a28e20bf99ae297a5
pins original rows and source identities. Baseline mean per-chart suffix LN/head
is .276721553, 36947 heads, 10875 LN heads, HH<30 ms 2 and RH<30 ms 67.
There are eight independent song groups, not 16 independent songs; no population
confidence is implied by this small screen.

Single intervention: create a clearly labeled inference-only diagnostic copy of
the release checkpoint, setting seed_residual.2.weight and
long_memory.output.weight to zero. Verify every other tensor is bit-identical.
Preserve the model configuration, all other weights, source R/H, observed seed
rows, known crossing seed endpoints, candidate support and RNG seeds. Retain
the modules and their state updates so only the residual readouts change. Do
not remove future timing features or simulate BOS. This joint removal measures
their combined contribution and cannot allocate separate seed/memory effects.
No product source or original asset changes are needed. Measurement code and
the diagnostic checkpoint belong to a fresh ignored artifact owner.

Primary diagnostic: absolute paired change in each chart's suffix LN/head,
reported per case and averaged over the 16 cases, with each song's two-seed
average also reported. A mean change >=.10 is a material reliance signal;
record signed changes and source differences without treating reference
composition as the unique correct answer. A mean below .10 is insufficient for
that selected material-effect threshold, not proof the omitted modules are
irrelevant. Also compare first/last required-onset-quarter fractions, head
counts, HH and RH gap counts separately, longest fixed-lane H and unchanged
three-hold H runs. These are inspection locators, not playability objectives.

Generation must finish and independently pass legal replay and osu! reparse in
all cases. A failure, nonfinite parameter/state, pin mismatch or baseline-parity
failure stops the probe and remains recorded. Native head-count ratios outside
[.7,1.3] or >=.10 LN-fraction changes require inspection. Review the two song
groups with largest paired mean composition change using Lens baseline/source/
ablated action traces and complete time pages around maximum local change;
also review any new longest three-hold run above 16 H. A discovered gap increase
alone does not classify BAD. Preserve Tech, repeated heads and LN articulation
when interpreting these views.

Procedure: verify pins and clean source, run val-08-s17 with unchanged release
weights through run_generation, require exact original rows.jsonl and
decisions.jsonl bytes, then run the 16 ablated cases through the same runner.
Use original GenerateConfig fields for each case, apart from fresh output,
diagnostic checkpoint and the explicit bounds below. Freeze input/config/script
and diagnostic-weight hashes before execution. Command:
.venv/bin/python artifacts/joint-audio/20260924-r1-condition-ablation-v1/run_probe.py.
Pair by case ID. Original baseline outputs may be reused only after parity.
No training, network requests, overwrite or implicit resume.

Environment: Apple M5, Python 3.10.20, Torch 2.11.0, CPU, one thread, sequential
runs. Each case <=600 seconds, total <=1800 seconds, candidate budget 8192,
physical footprint and RSS <=6 GiB, >=2 GiB available RAM, <=128 MiB swap
growth, >=40 GiB free disk, <=512 MiB output per case and <=4 GiB for the
artifact owner. Honor a PAUSE file in that owner. Fresh output owner:
artifacts/joint-audio/20260924-r1-condition-ablation-v1, with baseline and
ablated case subdirectories. Preserve interrupted outputs and stop; revised
procedures require a new Card revision/fresh destination.

Interpretation limits: inference removal is out of the fitted conditional
distribution and is not an optimally retrained smaller model. Original fixed
timing and real seeds make this a dependence probe rather than a reproduction
of audio-only generation. Diverged histories mediate the total rollout effect.
A large effect motivates an equivalent-conditioning or matched-refitting test;
a small effect directs attention to other transfer/task changes. No result
assigns an additive percentage of current audio-model instability to R1.

### Result Log: R1 neural-condition removal under original timing

Card r1-neural-condition-removal-v1 revision 1, proposed Note revision
5d02640392e35792e7e51e40ade52582ef157360; accepted revision none.
Exploratory execution under standing authority. Baseline/intervention runtime
source is the same clean 0848d8d6898939d850cf3bd7bbf4332ab121dceb.
The artifact script is pinned in freeze.json. All 30 bounded-model files match
the original training source after the package rename; no inference product
code change was needed. Only the two named readout tensors differ; 132 others
are exactly equal. The diagnostic checkpoint is not trained and has SHA
216a3f295acf063e5f0181ff7d2da3f0caee6b8e6c0da537fc8649127efb114f.

Original val-08-s17 rows and decisions reproduced byte-for-byte. All 16 paired
ablations then completed with legal replay and exact osu! reparse. The artifact
owner is artifacts/joint-audio/20260924-r1-condition-ablation-v1. Freeze SHA
48124074b54a682d7883870e7a6de47610f4cfb28a5b81bbb0e078f63323d602;
result SHA 72df5d4a8f09ad01e57d82c00977bf59fb3dd2db2ef6c0611955aba989aa039b;
baseline-parity SHA
5ebd2084b0aaeb0d7368ce40b8fd02998c46971c8e3b2aa1c6a37c45cf8a29e3.
No overwrite, resume, network access or model fitting. Command and seeds match
the Card. The CPU-one-thread probe took 62.71 seconds; peak sampled RSS
541.33 MiB, physical footprint 484.75 MiB, minimum available RAM 7.34 GiB,
zero sampled swap growth. Owner size including Lens evidence was 75.31 MB.
All recorded bounds passed; stop reason all_cases_completed. No live process
handles remain.

Mean per-chart LN/head changes .276721553 -> .267962399, signed mean
-.008759153 and mean absolute paired difference .068816914. This is below the
predeclared .10 material-effect threshold. Head totals change 36947 -> 38069;
LN-head totals 10875 -> 10803. Every head ratio lies within [.7,1.3]. HH<30 ms
changes 2 -> 6, RH<30 ms 67 -> 56; neither short-gap column defines BAD and
the two transition types remain separate. First/last onset-quarter mean LN
fractions are .21753/.29420 in baseline and .26430/.26337 under removal.
Maximum unchanged-three-hold run remains 2 H. No primary global LN increase
or new persistent three-hold trap is demonstrated on this original task.

Two cases have large whole-chart decreases: Youma Yakou seed 23, .37910 ->
.12309; tanasinn seed 17, .55266 -> .32196. Lens review covers both seeds of
these two song groups, including source/baseline/ablated charts. The bundle
manifest SHA is c195f7d5b4c9f0fe76019bba3f0cd7ec401edb3041aeb5b7f0c0839b0bf74d77;
all 153 members verified. Complete paginated actions/articulation were extracted
and all 48 time-view pages visually inspected. The post-hoc eight-second
max-difference locator, with one-second stride and >=8 heads per arm, adds
one-second entry/exit context. This is a stated inspection locator, not a new
success metric. Detailed scopes and judgment are in lens-review/inspection.md.

The changes are arrangement-mode changes, in both directions. Tanasinn seed 17
has an eight-second LN-head ratio 121/128 -> 10/138; seed 23 has 24/153 ->
109/134 despite only +2.48 percentage points in its whole-chart fraction.
Youma Yakou seed 17 changes consecutive/overlapping LN flow into taps around
long anchors, including a 3750 ms hold. Fewer LN heads need not mean lower
sustained occupancy. At seed 23 the more tap-heavy result adds larger chords,
so fewer LNs need not be easier or better. These views do not justify a blanket
BAD label or preference. No listening/player validation, Tech/dump preservation
claim or modification of human annotations is made.

Evaluation/Decision: REFINE. Combined readout removal can alter regional choices
substantially but does not meet the selected mean-effect threshold or show a
uniform deterioration. This weakens an unqualified explanation that omission of
these modules alone causes the current joint model's LN bias. It does not make
their removal harmless: this is out-of-distribution inference on source timing
with real seeds, not matched refitting, future-feature deletion or audio BOS.
The full restored policy's remaining limitations are still separate from the
joint model. No additive R1/audio blame percentage can be computed.

Durable findings and evaluation limits are committed in
docs/research/r1_transfer_stability_audit.md at
5f7439e208da94db4b2c7b2936dbc98aacc2a768. Documentation links and numeric
claims were checked against the pinned outputs; git diff --check passed. The
product change is prose only. Product and Note commits are local, not pushed.

Next research step remains a matched joint-learning initialization comparison
of plain 4.5M, memory 6M and release 6.5M to distinguish earlier backbone fitting
from later native correction weights. Keep the current global/bounded
architecture, full audio, repaired data, objective and sampling fixed. Before
implementing or running it, design a new proposed Card and make learning-rate
groups semantic and identical across arms: current context_training groups by
copied tensor names, which would otherwise give absent head/release residuals
a different rate. Do not silently change the deferred persistent-intent Card.
No new model training started in this probe. Ultimate playability remains
unproven, the goal is active, and this turn made material causal/evaluation
progress rather than delivering a new playable candidate.

### Experiment Card: r1-lineage-joint-learning-v1

Revision 1; proposed; accepted revision none. Owning Note
2026-09-23-audio-skeleton-r1-integration. Standing execution authority covers
implementation, short checks, bounded fitting and evaluation. The intent-code
trial remains deferred and unchanged.

Question: after equal joint audio/timing learning, do earlier backbone fitting
and later native-correction weights materially change source-free arrangement
behavior? Three initialization arms distinguish these two sources: plain 4.5M,
memory 6M and release 6.5M. All use the same existing global-audio/bounded-timing
model with 3461828 parameters. No architecture, latent, objective, support,
audio-input or decoder extension. The closest analogue is the actual R1 staged
lineage and its transfer audit: only the transferred tensors differ. The
alternative is that the shared joint task/data/optimization recipe dominates
the observed native behavior. This is attribution within a finite training
budget, not a universal ranking of pretraining or an additive blame estimate.

Clean baseline product OID: 5f7439e208da94db4b2c7b2936dbc98aacc2a768.
Initialization pins: plain
816d9daf0eb387bb65a661eba348db5c90358a37d68739f71d3223073167c18e,
memory 3262f210ff6f67837f10284a4e1c54ec408354c3e1c325fc0f95dc53dcca1a23,
release 8000beb0dee81b92f0252fcec876823afd60fe33a182fb344ae0bed02d03f991.
Original asset owner r1-restoration-20260920-v1/run; no original files change.
Response 6.75M is redundant with release after transfer and is not an arm.

Data: repaired 615 TRAIN arrangements/240 audio groups and unchanged 36 VAL,
manifest 4ad9abfd0ae7798e0a85b96a5dbecdaefd2a81f78bf181438407442b964118e1.
Normalization remains SHA
9cf461a0e825f974f0a80a364123c7afedf1683af76e60fe09b0fbe51c2c8287.
Full audio is available in all arms at training and inference; no TEST, target
future states at inference, arrangement unions or annotation pseudo-labels.
Use model seed 230941, sampling 230942 and VAL 230943. The existing shared plan
has SHA 34ef9751df4571f516ef2b670b3b0f4b398c212fdd16adcee1a029bd9168eb14,
4800 samples from disjoint clock intervals totaling 37253258 ms and 612 distinct
TRAIN arrangements. Recompute through canonical make_protocol and require
identical bytes under new protocol name r1-lineage-comparison-v1 before fitting.

Each main arm uses 1200 updates, two songs/update and two 8000 ms intervals/song,
the established group/arrangement/time per-second joint likelihood, and fresh
AdamW (.01 decay, gradient clip 1). Use 3e-5 for temporal/exact/fuse/joint/
route_residual/release_residual and 3e-4 for all audio/timing parameters. Assign
these rates by module family in every arm, including residuals absent from the
source checkpoint. The necessary context_training change replaces copied-name
grouping and records group names, rates, parameter ownership and initial hashes.
Share the module-family constant with initialize_from_r1. Verify grouping is
unchanged for a complete release transfer and equal across all three arms.
Other trainers and inference behavior are out of scope. Record a clean
intervention OID before any run.

The release arm is a newly fitted comparator; its endpoint value is pending.
The older 585-arrangement/other-seed model's 41.93044 VAL NLL/s and 84/84 native
completion are contextual evidence only, never a paired baseline. Primary
diagnostic: mean absolute per-case LN/head difference between release and each
other arm across the frozen 84 native cases. >=.10 is a material initialization
effect; below .10 is insufficient for this threshold, not equivalence. Report
signed differences, all cases, eight-second trajectories and song-group
aggregations so cancellations remain visible. Also report head-count ratios,
LN duration/occupancy, chord burden, HH/RH gaps separately and pause/end behavior.
These measurements select mechanisms and review locations, not playable winners.

All arms receive the same 180 VAL intervals (144 population and 36 BOS),
evaluated at updates 0/300/600/900/1200. Report fixed final NLL/s, timing/row
components during learning, exact interval/event/head exposure and compute.
No best-checkpoint selection or metric-driven learning recipe changes. A >3%
source-likelihood regression blocks promotion as an improved candidate until
its tradeoff is examined, but remains useful attribution evidence.

Native evaluation keeps the original six TRAIN plus 36 VAL identities, seed
bases 17/19 and original within-split index-to-seed mapping from
expanded-v1/generation/context-gb-prior27-s<17|19>-<train|val>/freeze.json and
result.json. Verify against the four pins in the frozen Lens manifest
45a4bd240a2dad254d308cf94d19a091c6c3a6387c22be289802562dff943233.
Add the unchanged Who/Death Piano/Prom Queen/Good Luck TRAIN audio panel at
seeds 17/19 from style-audit-v1/generation/freeze.json. Freeze the exact panel
and source/audio identities before fitting. Each chart starts from BOS, using
500 ms queries and the same 27 ms head-age prior; no reference timing or prefix.
CPU one thread, 30000 proposals and 90 seconds per chart; keep every capped,
empty, short or silent-tail result rather than dropping failures.

Native guards for a proposed improved candidate: 84/84 complete and >=30 heads,
no <=10 ms HH pairs and at most two <=20 ms; retain eligible transition rates,
median paired head ratio in [.7,1.3], inspect every ratio outside [.5,1.5].
Inspect material LN-mode changes and late silence, not only global descriptors.
Lens review covers all four human-reference contexts for both seeds per arm,
with full entry/exit action pages and time views. Compare Tech relations,
repeated chord/tap organization and independent LN presses/releases; missing
human labels remain unreviewed. Do not force reference replication or penalize
all LNs/repeats. A metric win alone cannot select a new playable candidate.

First run 32 updates per arm from fresh initialization, validation_songs=6,
evaluation at 0/32, <=600 seconds each. Verify finite losses/gradients, identical
consumed exposure, non-R1 initial hashes and semantic learning rates. A preflight
can check exported model loading but is not a quality endpoint. If all pass,
start main runs fresh, release then memory then plain, <=1800 seconds each,
<=7200 seconds total training including preflights. Generation <=1800 seconds
overall; freeze actual runtime and stop reason. Apple M5, explicit mps training,
one CPU thread, Python 3.10.20/Torch 2.11.0, sequential accelerator work only.
Require >=2 GiB available RAM, >=40 GiB free disk and <=5 GiB added artifacts.
Honor PAUSE in repaired corpus or the comparison owner. Stop on nonfinite
loss/gradient, pin mismatch, invalid causal inputs, unequal exposure, resource
or time cap, failed parity or parameter-ownership checks. Retain stopped runs;
no overwrite or implicit resume.

Use packaged ensomi_model.research.joint_audio_continuation.context_hydra with
the full pinned overrides saved in the comparison freeze and resolved Hydra
files. Fresh experiment owner:
artifacts/joint-audio/20260924-r1-joint-lineage-v1. Training outputs are
repaired-v1/context-training/lineage-{preflight|main}-{release|memory|plain}-v1;
native outputs stay under the comparison owner. The artifact launcher records
exact commands/script hash and checks return codes/bounds before advancing.
No remote push or lifecycle acceptance. Attribution with one training seed
does not establish robustness to initialization or a final playability result.

### Joint-lineage implementation and frozen inputs

Product 4db2335bec1996626f7eec526ecb3a7bb8f7ab9e implements semantic learning-rate
assignment only for the context trainer, shares the transfer module-family
constant, and records named optimizer groups and initial hashes. Architecture,
inference and other trainers are unchanged. The final focused command
uv run --extra mps --group dev pytest -q
tests/research/joint_audio_continuation/test_context_training.py
tests/research/joint_audio_continuation/test_model.py
tests/research/joint_audio_continuation/test_context_intervals.py
passed 25 tests. An initial test draft used a nonexistent output.weight name;
it was corrected to the existing score.2.weight without changing the intended
assertions. The checks cover missing/present routing checkpoints, equal rates,
unmodified audio initialization, actual trainer consumption, serialized group
ownership, transfer and interval behavior. git diff --check passed. Local
commit only; no publication.

Card r1-lineage-joint-learning-v1 revision 1 remains proposed at Note commit
da0170781870bda9255c5c7aec2c750e7b8353bd, accepted revision none. The baseline
to implementation diff contains only the declared grouping/receipt change,
its tests, shared constant and owning documentation. No unrelated intervention.

Fresh owner artifacts/joint-audio/20260924-r1-joint-lineage-v1 contains the
launcher run_comparison.py and freeze.json SHA
ab6456cd639a33275f5f9c03957ca5d7fa8995ae7e05d4b0308ea162bbf19aa3.
Canonical protocol recomputation exactly reproduces SHA
34ef9751df4571f516ef2b670b3b0f4b398c212fdd16adcee1a029bd9168eb14 under the
new r1-lineage-comparison-v1 name. All three real initializations have identical
1017140-parameter audio/timing group SHA
62a00e7262edabdac5f23c1f8bded5f122b281558c8f146ff19dd0b3f3542732.
Their complete group ownership/rates match; release grouping equals its copied
tensor set. The 84 native cases retain exact original source/seed mapping;
the eight style cases use the original audio pins. Inputs and all phase-specific
Hydra commands/configs are frozen before training.

The 32-step preflight phase has been launched sequentially on MPS with
caffeinate -i and the frozen launcher. No preflight outcome or main training
completion is yet claimed. Subsequent status must be verified from the live
process/tool handle and result files, not from this launch record alone.

### Result Log: joint-lineage preflights and main-run handoff

The three preflights completed on clean product
4db2335bec1996626f7eec526ecb3a7bb8f7ab9e under the frozen proposed Card. Result
SHA 755721dbfb423f211b85dd6939e08117701923ac8d157e408dbef955abcaaf97.
All consumed the same 32 updates, 128 intervals, 991062 ms, 6722 event rows,
9246 heads and 105185 timing bins. Total launcher time was 68.59 seconds.
All losses/gradient checks remained finite, group receipts matched their frozen
real initializations, and every saved model loaded with finite weights.

| Initialization | Preflight VAL NLL/s at 0 / 32 updates | Training seconds | Checkpoint SHA |
| --- | --- | ---: | --- |
| Release 6.5M | 63.29789 / 49.06779 | 22.97 | a0ed3b7b87527fc87bfbe9f87133d732cb18b69f267b3750584dd8c69a4c5760 |
| Memory 6M | 63.33745 / 49.11939 | 20.38 | 41363d8769c2333889786f327912e5f4f28c0943bcfefe24c01710e8c4f1e04b |
| Plain 4.5M | 59.45962 / 49.39479 | 19.49 | f84fa8d9a101d6487c6016b68ad179b35b67085b5505d860c647f24be4b3290f |

The preflight VAL subset contains 24 population plus six BOS intervals. These
values are sanity checks, not endpoint quality comparisons. Peak logged MPS
driver allocation was 1.573 GiB in every arm; minimum logged available RAM was
5.756/5.506/5.539 GiB respectively. No caps, resource stops or deviations.
Evaluation: passing engineering preflight permits the already-declared main
phase under standing execution authority; it does not establish a supported
research conclusion, Card acceptance or model quality.

Main training launched with the same frozen run_comparison.py main command,
wrapped by caffeinate -i. It resets each model and optimizer from its original
R1 checkpoint; no preflight state is reused. Order remains release, memory,
plain, each fixed at 1200 updates. Unified exec session 76752 owns this serial
launcher. At the last verified observation, its release child PID 90609 was
live at update 950 after 416.86 seconds. The 144-interval population readouts
were 62.24451 at initialization, 42.71762 at 300, 40.53602 at 600 and 40.09641
at 900. Do not substitute these incomplete-arm values for the final comparison.
Re-poll that session or verify the live child/result files before acting;
live.json or this note alone is not proof a process is still running.

Native evaluator evaluate_native.py is prepared in the comparison owner but
has not been launched. It requires all three complete main endpoints. Its SHA
is f78cc6cb4236a6e5cb3564260955054aef5cee90ffb08352180fb6a343f237eb.
It preserves all 84 native plus eight style cases/arm, audio-only BOS, fixed
query/prior and bounded outcomes. Complete outputs use independent mechanical
replay and exact export/reparse. Descriptors keep inclusive HH<=10/20/30/40 and
RH<=10/20/30/40 separate, record exact LN durations, held lane-time, chord sizes,
last-head position and eight-second trajectories. Empty outputs retain null
ratios and explicit missing counts instead of disappearing from the cohort.

Descriptor arithmetic was checked with a manual hold-crossing-window/terminal
fixture and an empty chart; head/release gap counts remain separate and summed
window occupancy equals 9998 held-lane milliseconds. Head and LN totals also
match six pinned older native outputs. description-check.json records these
checks; this is instrumentation verification, not new generation evidence.
The four style audio pins are unchanged. Who, Death Piano and Prom Queen audio
are absent from the selected joint-training corpus; Good Luck is present.
This describes joint-audio exposure only and makes no claim about R1's earlier
chart exposure or hidden TEST generalization.

Next actions: finish/poll the existing main launcher without restarting on an
observation timeout; verify identical final exposure and finite endpoints; run
the prepared native evaluator on the clean pinned source; inspect the declared
human-reference contexts and material changes using Lens; then evaluate the
lineage hypothesis before choosing the next training intervention. Keep the
intent comparison deferred. Product changes and Note records remain local;
the goal stays active and playability completion is unproven.

### Result Log: target composition within the fixed lineage-learning plan

Read-only exploratory audit, no Card intervention or training recipe change.
On the frozen repaired manifest and shared 1200-update plan, audit_targets.py
in the lineage comparison owner verifies full-chart LN start/release pairing
and recounts all sampled intervals with the actual per-second objective weight.
Target audit SHA
a682d79579d11631be6c2a79fa9386cbf96548ab1f82a118e4d376eb46f0f017.
No fitting or source/human annotation edits. An initial script draft had a
duplicate dictionary keyword and stopped before execution; the corrected
auditor's hash is recorded in its result.

The 615 TRAIN arrangements across 240 groups include 35 charts with no LN
heads. Under uniform group then uniform arrangement sampling, the probability
of choosing such a chart is 4.826%. Expected mean chart LN/head is 17.465%;
expected tap and LN-start rates are 7.68987/s and 1.57808/s, giving an LN share
of expected head-rate contributions of 17.027%. These are distinct aggregations.

The actual 4800 sampled intervals contain 287184 taps, 58429 LN starts and
58463 releases in 253512 event rows, totaling 37253258 ms and 345613 heads.
The time-weighted LN share of new heads is 16.886%, close to the sampling
expectation. Intervals have 1194 tap-only-head, 3347 mixed-head, 44 LN-only-head
and 215 no-new-head outcomes. Interval start/release counts may differ because
their boundaries cut holds; every full source has matching counts. Releases
are separate supervised actions, not extra LN-start labels.

These counts exactly match the completed release branch's recorded interval,
millisecond, event and head exposure. They do not support the simple explanation
that the fixed sampling plan mostly teaches LN heads or strongly inflates their
share relative to this corpus. They do not set a required fraction for every
native chart, establish a well-calibrated free-running distribution, or rule
out conditional optimization and feedback effects. Outcome REFINE; keep the
matched lineage comparison unchanged and inspect actual native action paths.

The release main arm has completed all 1200 updates in 555.17 seconds, checkpoint
SHA 02f6511fbd146a656b84e6aaf95821068d56f9dc16b3a0d36c51e0267a3be910;
fixed population VAL NLL/s is 40.06002. The same serial launcher session 76752
continued into memory, observed live as child PID 92516. Main comparison and
native quality are still incomplete. Revalidate the current handle/state before
continuing; no restart based on elapsed time or stale progress records.

prepare_lens.py and inspect_lens.py are now prepared in the comparison owner.
They await native-result.json, retain every native outcome in a case inventory,
copy the frozen human evidence unchanged, and request full actions/articulation
plus every time-view page for the 24 generated style contexts and four source
contexts. They have only been syntax checked; no new bundle or images have yet
been produced or judged. Material native guard cases still require additional
scoped review after their outcomes are known.

### Architecture steering and terminal lineage readout

The owner explicitly restated that NLL is a proxy and requested explanations
from model structure, module coupling, information flow, dependency, training
invariants and approximations. The owner also requires direct audio conditioning
of row materialization in addition to skeleton conditioning. The dependency was
then refined: skeleton uses its own previous skeleton, not the learned content
history of materialized rows, but may read actual LN occupancy because it affects
the skeleton. This last physical-state exception supersedes a strictly one-way
interpretation. Do not remove the audio-to-row path or all LN feedback.

The fixed boundary is therefore separate skeleton history and row-content
history, with an explicit minimal LN-state projection crossing from committed
rows to skeleton generation. Occupancy and active-LN start/age are candidate
physical fields; generic tap/chord layout, learned row embeddings, row counts
and past-head clocks must not enter by accident. At fixed audio/controls,
skeleton history and this LN projection, changing tap materialization should
leave the skeleton distribution unchanged. Row decoding may and should remain
sensitive to its own materialized history. Joint optimization through a shared
audio encoder is distinct from runtime row-content feedback.

The current global/bounded model violates this desired boundary: its timing
residual reads the row TCN and complete exact-state features. Its row decoder
does read local and global audio directly, but has only the current sampled
event time and past materialized context, not a separate future skeleton plan.
The existing models are now diagnostic baselines, not the presumed final
architecture. Keep the old persistent-intent trial deferred; do not enlarge
that architecture before resolving the changed dependency contract.

The main lineage fit completed all three endpoints with identical 4800
intervals, 37253258 ms, 253512 event rows, 345613 heads and 3954145 timing bins.
Main result SHA
21aede634fed0106535cd7bf77b4470d858306a9489307bcd0d72f801fda965a.
Release checkpoint 02f6511fbd146a656b84e6aaf95821068d56f9dc16b3a0d36c51e0267a3be910,
memory f872b270f380adba4142aa0d206fd1ea1973c7e0e3dcae1f8274de7acdaa3def,
plain c4fa67309a50473bb4400146229a6ce610be01c7ff011c3d530d931cfe9925f5.
Final source-conditioned population NLL/s is 40.06002 / 40.05814 / 40.40358;
this does not rank native playability. Total main launcher time 1691.02 seconds.

Readout supervisor run_readout.py was launched after verifying training parent
PID 90600 and its creation time. It waited for that actual parent to exit before
starting native evaluation. Unified training session 76752 and readout session
76923 are now terminal. Native generation hit the declared 1800-second overall
budget: 246 attempted, 245 complete, one partial plain case val-s19-13, and 30
unattempted. Release and memory each have all 92 native/style outcomes; plain
has 62 attempted native cases. The partial case stopped on the overall budget,
not its per-chart cap. Do not count this as a demonstrated model-generation
failure or drop unmatched cases. Native progress SHA
0908f003216ddd944b795a3b4fb4ffa61d9e890341136e7d64a21cad4a46eedf.
partial-readout-summary.json preserves the bounded outcome. No automatic
seed-variability analysis or full Lens extraction ran after the bounded stop.
readout-live.json is stale; both its child PID 96230 and supervisor have exited.
Do not restart or extend this old-architecture sweep implicitly.

A separate interim quality inspection used one complete Airborne Robots case,
seed 33, alongside incumbent v2, the source NM arrangement and the human-reviewed
AIRBORNE alternative of the exact same audio. Owner:
artifacts/joint-audio/20260924-r1-interim-playability-v1. Bundle-v2 manifest SHA
f8996225e8a66c03d5f75ae103451517312f8e9ab36beb11406fe6f527bf92e4.
All eight time-view pages of [147208,150258) were inspected. New release has
65 heads/16 LNs, incumbent 51/23, AIRBORNE 50/10 and NM 10/1. The new output
includes larger chords and a local LN block; a lower LN ratio does not by
itself make it better. Human Tech support belongs to AIRBORNE's narrower
[147608,149858) scope, not automatically to either generated chart. No listening
or player verdict is claimed, and this comparison changes multiple training
variables rather than isolating R1 lineage.

The initial interim bundle preparation stopped on an overly broad normalized
object equality check. Fresh parsing adds id/x/hitSound/sourceKind note metadata
absent from the frozen parquet representation; all 3814 frozen note fields,
source metadata, range and timing points match exactly. The second preparation
preserves the frozen human chart unchanged and records the fresh output in
bundle-v2; the first incomplete bundle remains as failed preparation evidence.

### Experiment Card: skeleton-input-noninterference-audit-v1

Revision 1; proposed; accepted revision none; exploratory diagnostic under
standing execution authority. Clean source
4db2335bec1996626f7eec526ecb3a7bb8f7ab9e, fixed release-initialized joint endpoint
02f6511fbd146a656b84e6aaf95821068d56f9dc16b3a0d36c51e0267a3be910.
Question: does the current timing path change when tap placement changes while
audio, skeleton times/roles and LN state remain identical? The desired new
interface forbids that dependency; the current graph explicitly permits it.
This test locates the path rather than proving why a chart is bad.

Use the byte-pinned completed Airborne release output in the interim inputs,
its exact audio/Mel and two prefix cutoffs 147207 and 148249 ms. Construct a
legal counterfactual by reassigning only preceding TAP heads to the earliest
available lanes, preserving the number of taps in every row and every LN
start/release action. Keep future rows unchanged and use none of their labels
for the probe. Verify all prefix event times/H-R roles and the complete LN
trajectory are identical. This changes row-content history and non-LN clocks,
not the allowed skeleton/LN inputs. It is never paired with original future
rows for a training loss.

Score the conditional next-event distribution over the following 500 ms with
complete audio and each legal prefix. Primary readout is maximum absolute
hazard-logit difference; >1e-5 demonstrates a locally active forbidden path.
Record selected CDF points, timing base, bounded residual and gate. The base
and gate must match within 1e-6 because their allowed inputs are identical;
failure stops interpretation as a content-only intervention. A zero difference
would show dormancy at these prefixes, not global independence.

Also verify manual row-score decomposition against the canonical decoder and
the algebraic invariant that head/release routing preserves kind probabilities
within a fixed head-mask/release-mask family. This is an implementation/property
check, not an alternative generator. Inspect direct local/global audio residual
contributions without treating their norms or zero-channel scores as quality.
No training, model mutation, generated rollout, reference-suffix supervision or
new performance claim. Fresh owner
artifacts/joint-audio/20260924-skeleton-input-audit-v1; freeze script/input pins
before scoring. CPU one thread, <=60 seconds, >=2 GiB available RAM, >=40 GiB
free disk, <=20 MiB outputs, no overwrite or resume. Record all failures.

### Result Log: skeleton-input-noninterference-audit-v1

#### Experiment and reproduction

Owning Note 2026-09-23-audio-skeleton-r1-integration; accepted revision none.
Card skeleton-input-noninterference-audit-v1 revision 1, proposed at note commit
06b4ff31b7d78cbf0cafa212703b8a308b27d46d. Baseline and instrumented execution use
clean product 4db2335bec1996626f7eec526ecb3a7bb8f7ab9e; no product implementation
change or model mutation. The frozen probe script SHA is
27d628a3183be044ea38f25b608d347baf68eef850ed5d8709d02a2727955a4e.
Procedure: execute probe.py from the fresh named owner, loading the fixed release
endpoint and completed Airborne Robots seed-33 output. No sampling or fitting
occurs in this diagnostic. The script verifies source cleanliness and all input
pins before encoding full audio and comparing each original/altered prefix.

Owner: artifacts/joint-audio/20260924-skeleton-input-audit-v1.
Freeze SHA 3769d59881b8a7c2a036293edbfd3c099194c09955f9915371dbe7d1c521be3e;
result SHA 2038a5f2be4bcb422496273affc9317212e4bdd0a69aa3b645781dbb1d0b861a.
Checkpoint 02f6511fbd146a656b84e6aaf95821068d56f9dc16b3a0d36c51e0267a3be910;
native rows 70fbdac5d811782ed0163e8b03391df531374f2e36b1234c5084a4406fbccd62;
audio 962f70f90431c99005f0644bd893329d0df5cbcbe79b56c9396ff9fb3bfc4f3d;
Mel 2ff88832f579faa64145d23c23a8baefcfa018feac6789ec1d10107602d142b7.
Apple M5, 24 GiB RAM, Torch 2.11.0, CPU one thread. Completed normally in
0.62359 seconds under the 60-second bound; no overwrite, resume or failed run.
The available-memory and free-disk guards passed before encoding and both
queries; numerical minima were not recorded. Post-run inspection confirms three
files totaling 14868 bytes in the owner, below the 20 MiB output bound.

#### Results and plan conformance

The two legal interventions preserve times, roles, per-row TAP counts and all
LN starts/releases. They relocate 1128 preceding tap rows at 147207 ms and
1136 at 148249 ms. The latter has columns 0/1 occupied; the former has no holds.
These large rearrangements test a functional dependency, not a natural quality
counterfactual. Next-event distributions are queried for 500 ms.

| Cutoff | Maximum timing-logit difference | Original / altered CDF at 50 ms |
| --- | ---: | ---: |
| 147207 ms | 0.4588027 | 0.6180910 / 0.5895400 |
| 148249 ms | 1.4190359 | 0.6135968 / 0.7308428 |

Both differences exceed the declared 1e-5 path-activity threshold. Base and
gate differences are exactly zero, within the 1e-6 guard. Row-score decomposition
matches the canonical decoder exactly. Across 54 legal head/release-mask
families, routing changes conditional tap/LN probabilities by at most
5.82194e-8, below the declared 1e-5 tolerance. Direct local/global row-audio
residuals are nonzero; their norms are not a quality or sufficiency metric.

No protected field changed. Source code, checkpoint, prefixes, intervention,
query horizon and deterministic comparisons match the Card. No rollout, future
label loss, listening or new player judgment was performed. The missing runtime
resource minima limit resource reporting, not the deterministic comparison.

#### Evaluation and decision

Observation: row-content/non-LN-clock changes reach the bounded timing residual
even with identical allowed skeleton/LN inputs. Interpretation: the existing
model has an active path excluded by the refined interface. This probe does
not separate the TCN from unrelated exact clocks, nor establish that either
caused a specific BAD chart. The allowed LN timing base is unchanged, so the
result does not support removing physical LN feedback.

Recommendation REFINE. Enforce the input boundary in the next architecture,
then evaluate native organization; do not scale the current coupled timing
model to compensate. Fixed-mask routing invariance is local and does not
guarantee preservation of complete sampled trajectories. No Note acceptance
or model adoption follows from this exploratory result.

### Frontier role retained in the next design

The human owner emphasized that frontier2's conceptual role matters to the
formulation. The original RowConsequence code remains in bounded R1, but neither
JointAudioModel nor ContextAudioModel instantiates or transfers it. The verified
bit-identical joint initialization from 6.5M and 6.75M therefore means the final
stage's direct correction is absent, not that it was behaviorally redundant.

Product b5a664cd3da5dd821bc5a6d5d149719cf76386dd established the skeleton/LN
information contract and recorded the probe. Product
e32e6e1df3a97f42af54d16b9b2292d05d832bcf adds the previously omitted consequence
path to that proposal and connects the transfer audit to it. These are scoped
documentation commits; no replacement module has been implemented or trained.

Three concepts stay distinct: the formulation's response function over legal
futures; the old learned candidate-row residual using passive next-H clocks and
a second-H time gap; and its optimistic two-H, one-tap-per-H, earliest-release,
30 ms machine-preference training objective. The last two approximate a small
part of the first. They do not define calibrated playability or a universal
anti-Jack/release rule.

The proposed row path evaluates candidate immediate post-states, reads direct
audio and generated head lookahead, and adds consequence-conditioned preference
before legal normalization. Release times not yet chosen remain unknown or are
queried as hypothetical conditional futures. They are never supplied as actual
reference tails. Candidate evaluation does not mutate committed state or
consume publication randomness. After choosing the row, only the allowed LN
projection returns to skeleton generation. Reweighting skeleton plans using
the full-history consequence score would change that contract and is not
silently included.

The initial residual can learn jointly from source-row likelihood without
claiming its score is a canonical demand quantity. Old weights require matching
feature meaning, not only matching shapes. The head stream's independence from
LN occupancy remains a stronger proposed factorization, not an owner-mandated
restriction on all future skeleton models. No numerical result establishes the
fraction of new-system errors attributable to the omitted frontier2 branch.

### Experiment Card: planned-head-release-consequence-v1

Revision 1; proposed; accepted revision none. Standing authority covers local
implementation, tests and the bounded runs below. Baseline product
e32e6e1df3a97f42af54d16b9b2292d05d832bcf. The prior turn made progress by fixing
the frontier omission in the documented graph and persisting the dependency
probe; it did not implement the new model.

Question: does an explicit head plan, LN-conditioned release clock and
candidate-action consequence branch yield a learnable native generator under
the required information boundary? The intervention is this factorization as a
whole, not an attribution study of its individual components. Closest analogues
are local R1's candidate-consequence residual and the current full-audio hazard
model. The change is their adaptation to a planned head stream with online
physical feedback; no new general-purpose learning algorithm is claimed.

Implement a native-ms autoregressive head stream using only audio and its own
head history, a release-only hazard using audio, previous H/R skeleton and
current LN state, and R1-derived row materialization with direct audio and
bounded forward head context. The head-only independence is a stronger pilot
bias than the owner-required LN projection contract. Retain mirror symmetry,
causal row history, explicit BOS and separate random streams. This pilot uses
16 forward heads, 127 prior heads and 63 prior H/R events for skeleton encoders,
plus the existing 511-row action history and canonical local/full-song audio.

Restore RowConsequence frontier2 parameters from the pinned 6.75M checkpoint
through an explicit transfer adapter. Native integer clocks provide a structural
earliest-release opportunity at current time plus one millisecond, including
an H clock when present. This is a possibility, never a predicted release time.
Reference suffix LN endpoints remain unavailable. Old weights face a changed
opportunity distribution and row conditioning; transfer is not policy parity.
No old 30 ms preference objective or inference head-spacing filter is used.

Define release hazard one at the next H minus one millisecond when all four
columns are held, and at true audio termination when holds remain. This is an
explicit deadline-atom distribution used in both likelihood and sampling, not
a silently truncated and renormalized unconstrained distribution. If the next
H is one millisecond away, exclude candidate rows leaving all columns held.
H rows require heads; release-only rows require nonempty releases and no heads.
Query chunks never become deadlines. Actual release times after a candidate
remain unknown. Hypothetical scores cannot reweight the head plan.

Before fitting, reject the implementation on any exact replay/support,
source clock-coverage, teacher/native probability, RNG/chunk-partition,
mirror, full-audio crop or information-noninterference test failure. Include
all-held release deadlines, terminal closure, sub-10-ms heads, LN crossings,
empty intervals and changed TAP layouts with identical LN/skeleton inputs.
The primary initial question is a functioning distribution with those
invariants, not an improved NLL or an automatic playability verdict.

Use the repaired 615-TRAIN/240-group plus 36-VAL corpus, manifest
4ad9abfd0ae7798e0a85b96a5dbecdaefd2a81f78bf181438407442b964118e1,
normalization 9cf461a0e825f974f0a80a364123c7afedf1683af76e60fe09b0fbe51c2c8287,
R1 4b3ec1561e33d0ebe2756cfe13571ec414fd5bb470b430f0c578545863115f70.
Reuse the lineage comparison's frozen exposure plan
34ef9751df4571f516ef2b670b3b0f4b398c212fdd16adcee1a029bd9168eb14:
1200 updates, two songs by two 8-second intervals, model seed 230941,
sample seed 230942, validation seed 230943. Same full audio in training and
inference; never combine alternative arrangements into union labels.
AdamW with inherited R1 and consequence modules at 3e-5, new audio/skeleton
modules at 3e-4, weight decay .01 and gradient clip 1. Mac MPS, CPU one thread.

First run 32 updates with six frozen VAL songs, at most 300 seconds. Require
finite likelihood/gradients, actual updates in both skeleton heads and the
consequence branch, a reloadable checkpoint and completed mechanically valid
native probes. Only then start a fresh 1200-update fit capped at 3600 seconds;
do not reuse preflight optimizer/model state or select a best-NLL checkpoint.
Retain fixed endpoint, source likelihood by factor and all native outcomes.
Stop on <2 GiB available RAM, <40 GiB free disk, nonfinite gradients, corrupt
export, per-chart >90 seconds or >30000 physical rows. No model scaling.

Fresh experiment owner artifacts/joint-audio/20260924-planned-head-release-v1;
fresh runs planned-preflight-v1 and planned-main-v1 under the repaired corpus's
planned-training directory. No overwrite or implicit resume. Record the clean
intervention source before running. Native screen: frozen Who, Death Piano,
Prom Queen and Good Luck style cases at seeds 17/19 plus Airborne at seed 33;
at most 900 seconds total, CPU one thread, no post-sampling head filter. Compare
with completed release-initialized lineage outputs where available and the
frozen human windows; differences combine architecture and initialization.

Inspect every generated style-context time page and actions/articulation via
Lens. Report missing/empty/incomplete cases, LN articulation and release-to-head
relationships separately from repeated presses, regularity/irregularity,
chord organization and sustained repeated figures. Record first-30-row/head
and 8-second readiness plus dense-window throughput. Source note counts or LN
fraction alone neither accept nor reject charts. If mechanically valid but
qualitatively poor, REFINE using located failures; even favorable evidence is
exploratory and does not prove universal playability or imply Note acceptance.

### Result Log: planned-head-release preflight and main dispatch

Card planned-head-release-consequence-v1 revision 1 remains proposed; accepted
revision none. Clean implementation product
e04b3490e60b4f7a7550f9abde20e6f551382dcc adds planned_audio_continuation and its
packaged Hydra entrypoint. Total parameters 4245188; copied R1 parameters
2472336 including all eight frontier2 tensors (27648 parameters). Seed and
landmark modules remain omitted, with the historical exact-projection slice
reported explicitly. No old preference objective or spacing filter is used.

Selected checks: uv run --extra mps --group dev pytest -q
tests/research/planned_audio_continuation tests/test_package_layout.py;
15 tests and 22 package subtests passed, including actual MPS forward/gradient
agreement. Checks cover an analytic joint law, interval partitioning, cached
native/teacher score parity, RNG/query partitions, mirroring, TAP-layout
noninterference, hidden future tails, actual frontier2 transfer and a two-update
CPU training/checkpoint/native round trip. Initial test drafts had a relative
fixture import error and a single-row fixture invalid for SourceChart's seed
contract; both were corrected without relaxing product invariants. Documentation
links/anchors and git diff --check passed. No remote push.

Preflight command: uv run --extra mps python -m
ensomi_model.research.planned_audio_continuation.hydra
run_name=planned-preflight-v1 updates=32 validation_every=32
validation_songs=6 max_seconds=300. Session 14106 exited successfully.
Fresh output under the repaired corpus: planned-training/planned-preflight-v1.
Completed 32 updates, 128 intervals, 991062 ms, 6722 physical rows, 6217 H rows,
505 release-only rows and 291439 occupied release-query clocks in 35.57792 s.
Freeze SHA f91f0bc5b3b16ff6da934eaef24dbdc1151f9dbd96f890bc4c8b690acf2e3455;
result SHA 89b14c196bc4e6b45afb3df95bc1042af633ed3539ccac40479b75ecaba2a2bf;
checkpoint SHA 3f00031489b01b398974ae2b2c982633ec825e1d07e02ac08de020ed4b09eb6d.
The exposure protocol matches
34ef9751df4571f516ef2b670b3b0f4b398c212fdd16adcee1a029bd9168eb14 exactly.

All tracked modules updated: head_temporal L2 1.75509, head_condition .38033,
timing .58635, release_clock .60039, skeleton_temporal 1.45231 and
row_consequence .03653. Population NLL/s on 24 source intervals changes
62.72237 to 50.43675; six BOS intervals change 51.34194 to 42.80964. These are
learning diagnostics. Logged MPS driver allocation was about 1.82 GB and update
receipts retained >5.7 GB available memory.

The nine fixed preflight native cases all completed and independently reparsed,
with no empty or incomplete outcomes and zero all-held deadline releases.
Native evaluation took 18.85482 s. First 30 physical rows took .1588-.2725 s;
8-second readiness took .1876-.3105 s, starting from cached canonical Mel and
excluding waveform/Mel preprocessing. Outputs have 505-1311 heads and LN
fractions .0381-.1877; neither sparsity nor lower LN fraction establishes quality.
Lens inspection is still pending here. Native result SHA
1f341902c32f18a3228fffe92bd42a4ae287d7f74356b6b76411e2e53c0af9ba.
Owner artifacts/joint-audio/20260924-planned-head-release-v1; cases.json pins
Who, Death Piano, Prom Queen and Good Luck at seeds 17/19 plus Airborne seed 33.
Native session 55602 exited successfully. No source chart is a generation input.

The declared mechanical/learning gates passed. A fresh main fit was dispatched
with caffeinate -i uv run --extra mps python -m
ensomi_model.research.planned_audio_continuation.hydra run_name=planned-main-v1.
Unified exec session 75685 was authoritatively polled through update 160 and
is still running. The run is 1200 updates capped at 3600 seconds, with all
36 VAL songs in source likelihood readouts. Output:
artifacts/joint-audio/20260924-alias-restored-v1/planned-training/planned-main-v1.
Revalidate this handle; never restart because a status file is stale or an
observation times out. Main quality is pending. Next: Lens, terminal main readout,
the same nine-case native screen, and all final generated style-context pages.
Recommendation remains REFINE pending these outcomes; no acceptance or adoption.

### Planned preflight Lens inspection

The preflight Lens bundle is frozen at manifest SHA
4e61afb5bf17a7ef70034671a6aa855d93c080c75b4433cd03f66011594f7380 under
artifacts/joint-audio/20260924-planned-head-release-v1/preflight-lens.
All 40 time-view pages were actually viewed: nine generated contexts and five
unchanged human reference contexts. All paginated actions/articulation were
read, including entering holds and complete endpoints. review-facts.json retains
the exact records and page pins; review.json contains scoped observations.
Extraction session 39107 completed successfully. Human evidence was copied
unchanged from the pinned earlier bundle; no label was transferred to a model
output automatically. No listening or player trial occurred.

Prom Queen source has a sustained 208/209 ms repeated chord pulse in the reviewed
context. Preflight seed 17 instead has seven isolated taps with intervals such
as 848/191/800/69 ms; seed 19 has fourteen attack rows, mostly singles with two
doubles and uneven short bursts. Neither demonstrates the source's definite
Jack organization. Who's 73-87 s generated windows have 48 heads/3 LNs and 50/0,
mostly sparse irregular taps; the three LNs in seed 17 are sequential, including
a 992 ms hold with no internal attacks. The source's interleaved short head/tail
figures are not demonstrated here. This is not a density-match acceptance rule.

Death Piano's generated contexts have 20/2 and 16/0 heads/LNs; seed 17 enters
with a column-2 hold ending at 102967 ms, then a column-0 902 ms hold over two
other attacks. Good Luck has 21/1 and 18/0, including one 764 ms column-3 hold
over five internal attacks. Airborne has eleven heads/two sequential LNs and
one double. These are scoped arrangement facts, not global quality verdicts.
The human Who source includes 29-30 ms LNs and Good Luck includes 8/32 ms
cross-column events, reinforcing the ban on universal minimum-duration/gap
repairs. The main remaining question is learned rhythmic organization, not
mere legality, lower LN fraction or a stronger hard spacing filter.

Main session 75685 was subsequently observed live through update 1190. The
fixed 1200-update endpoint and its native/Lens readout remain pending.

### Main terminal readout and head-starvation audit Card

Main session 75685 completed successfully: 1200 updates, 4800 intervals,
37253258 ms, 253512 event rows, 238904 H rows, 14608 release-only rows and
10684532 occupied release clocks in 991.63926 seconds. Checkpoint SHA
20cea8e5d00e554c93141af7403a47e7108a3ee8fbd2c2dd57b8c9c415aa6205;
freeze d4e085e8a4e452cbe81a6ba8bb4fdd8b7d8412bae220244cde81150c67563e11;
result 6a84e7017a08882b9de7c08ed9a72929f090ca16d44c6ff9984104116187c3fe.
All tracked modules updated. Population NLL/s 40.27903, BOS 27.22964; no model
selection was based on these. Logged available memory minimum 4626694144 bytes,
MPS driver maximum 3126149120 bytes. No training handle remains live.

Main native session 46746 also completed: all nine charts mechanically complete
and independently reparsed in 18.26422 s, zero all-held deadline releases and
zero <=20 ms same-lane head/head or release/next-head pairs in the descriptor.
Native result SHA 7f9ee93823ead44e1ce5a9d171d67a12fb236d35e6f074547229a7cdbcc4fab8.
These favorable mechanics hide a serious content failure. Who seed 19 has only
213 H rows and its last head is 56015 ms of 116368 ms audio. Good Luck seed 19
also has 213 H rows, last head 58555 of 218448 ms. Good Luck seed 17 has 441 H
rows, last head 125935 of 218448 ms; Death Piano seed 19 has 412 H rows, last
head 91236 of 177372 ms. Some positive human review windows lie wholly in the
generated silent tail. These outputs are not a playable-system success.

The initial main Lens extraction failed because its normalized note extent
ended before a fixed review window. Preserve that failed main-lens owner.
The main-lens-v2 preparation extends only generated visualization ranges to
known full-audio coverage, recording original note ranges in provenance.
It leaves every note and all human evidence unchanged and retains blank
windows rather than moving them to earlier populated excerpts. Session 37694
was launched for this extraction; its completion and page review still need
verification. This is an inspection-range correction, not new generation.

Experiment Card head-wait-survival-audit-v1, revision 1, proposed, accepted
revision none. Baseline clean product e04b3490e60b4f7a7550f9abde20e6f551382dcc
and the fixed main checkpoint above. No fitting or model/code mutation. The
question is whether long waits extinguish the head process independently of
row materialization. The head graph has no row input, but its own history and
elapsed-clock influence are unbounded. The closest local analogue is the
earlier bounded-history timing study; this audit tests the separated head path.

Use Who seed 19, Good Luck seeds 17/19 and Death Piano seed 19. Reproduce the
entire head stream alone from complete pinned audio, with the same head RNG
and 500 ms queries; assert every head time equals the saved native chart.
At the final head, integrate the model's remaining native-clock hazard through
actual audio end. Compare that mass with the next exponential threshold and
the sampler's retained residual. Report mass in <=500 ms, <=2 s, <=10 s and
the whole remaining song. If the independent head sampler does not reproduce
the saved times, stop the head-only attribution and investigate runtime parity.

For instantaneous queries 250/1000/5000/30000 ms after the last head, also
translate the same head-interval history so its final head is 100 ms before
the query. This is a legal counterfactual skeleton with identical interval
tokens and fixed current audio/time, changing only the recency clock. It is
not trained against the original suffix and is not a replacement generation
policy. Compare exact next-ms hazards and separate direct-audio and historical
contributions before the final timing nonlinearity. Norms alone do not prove
audio information sufficiency or chart quality.

Primary evidence: head-time equality, integrated mass/residual agreement within
1e-4, and the relative change in late instantaneous hazard under the recency
intervention. A post-2-second mass <.01 over >30 s of remaining audio, together
with >=100x recovery under the valid recency intervention, would locate a
long-wait suppression path. It would not prove a general infinite-horizon
defective distribution or that every silent span is musically wrong. If the
clock intervention fails, inspect encoded-history and direct-audio contributions
without asserting clock causality. CPU one thread, <=120 s, >=2 GiB available
RAM, >=40 GiB free disk, <=20 MiB fresh outputs, no overwrite or resume. Owner
artifacts/joint-audio/20260924-planned-head-release-v1/head-survival-audit.
The result changes whether the next intervention should add an audio-only head
base with a bounded, decaying head-history residual, rather than scaling model
capacity or applying a hard head-spacing rule. Outcome remains exploratory.

### Result Log: head waiting-time mechanism and complete main Lens review

Main Lens extraction session 37694 completed, with manifest
ef71d998ec207cc91ef195fed15fa97fa94cb8ab688ec55d7aea80d2cfa4b6a3 in main-lens-v2.
All 26 generated pages were actually viewed; all actions/articulation were read.
Fourteen reference PNGs were verified byte-identical to the already-viewed
preflight references. Five contexts are completely empty (Who 19, Death Piano
17/19, Good Luck 17/19). Airborne has no heads and only an entering LN from
142825 to 148361 ms, duration 5536 ms, with no internal attacks. Review SHA
cae16ff75750529968cc9c565ad416ee64ec298b5ab3f16ec896a39a76b185cf.

Partial organization did improve: Who 17 now has 58 heads/3 LNs in the fixed
context, with approximate 234/117 ms pulse relationships; Prom Queen has
20/0 and 16/0, with many approximate 208/416 ms placements and additional
chords compared with preflight. These observations do not rescue the silent
regions or establish requested style coverage, audio alignment or playability.
The endpoint is not promoted; no listening or player trial occurred.

Head-survival Card head-wait-survival-audit-v1 revision 1 remains proposed,
accepted revision none. First probe attempt failed before head reproduction
because constructing the model inside torch.inference_mode made parameters
lack version counters required by the temporal cache. Its freeze/failure and
original probe.py are retained. A behavior-neutral diagnostic retry uses
torch.no_grad for model construction, writes fresh attempt-002 outputs, and
keeps the same checkpoint, audio, cases, seeds, queries and comparisons.
Session 41010 failed; retry session 39190 completed in 2.43259 seconds.
Retry freeze SHA 625fa5365fd057ba3216b5fb4468c9160f5b8bbd950c84a74f093210438c88a7;
result dbc2e3143a5e3f48da09b031338e8e3a83313314cc0b87a15708691e0fb6266b.

All four head-only streams exactly reproduce every saved H timestamp, without
running R1 rows. Summed remaining hazard mass matches consumed exponential
budget within 3.82e-14. In Who 19 / Death 19 / Good Luck 17 / Good Luck 19,
total future masses are 3.36936 / 2.89453 / 1.85614 / 2.09168, below draws
4.56417 / 3.75976 / 4.07153 / 4.56417. Conditional no-more-head probability is
3.44% / 5.53% / 15.63% / 12.35% over 60.353 / 86.136 / 92.513 / 159.893 seconds
of remaining audio. At the +30-second query, translating the same interval
history to end 100 ms ago raises next-ms hazard 632 / 133 / 500 / 617 times.
Audio and interval tokens are fixed in those legal counterfactual skeletons.

The proposed stronger near-zero-tail criterion FAILED: post-two-second mass
is .736-1.862, not below .01. Do not call this an infinite-horizon absorbing
state or claim every later audio cue is ignored. The evidence instead locates
strong finite-horizon recency suppression and rules out a sampler, row-action
or LN-blockage explanation for these head-only silent tails. The selected four
failures do not estimate a population failure rate. The diagnostic norms do
not establish audio information sufficiency.

A separate descriptive TRAIN-clock audit uses the same 615 charts/240 groups
and exactly 37253258 sampled milliseconds. Result SHA
d2fc84af6bdf9b605b468517049059358812264af4e78bdf1b1b47e92d3f5c55; session 11987
completed. In the actual plan, 88.21% of >2-second head-age clocks occur after
the reference final H, rising to 94.53% for >5 seconds and 94.48% for >10 seconds.
Population group/arrangement/time counterparts are 87.17%, 90.38%, 85.17%.
No TRAIN chart ends its head stream before 60% of its paired audio. The four
reference arrangements contain 60/22/49/65 H rows in the five seconds after
the corresponding generated final head. Post-final reference time is not a
claim that the underlying waveform is silent.

Recommendation REFINE: the teacher distribution makes long head age a strong
ending cue, and own sampled long waits can activate that cue prematurely.
Bound the historical veto and let complete audio regain control after a long
wait, without a forced event or a spacing ban. The existing row/LN/skeleton
independence and frontier path remain. Candidate risks include forgetting
density/phase across rests and a weak audio base. All model and probe sessions
above are now terminal. No new recovery fit has yet started.

The reusable analysis is committed in docs/research/head_wait_recovery.md at
daa49334cef2b38e42cf16a30ef1cf0f1b8a2cee. The plotted hazard sums were visually
checked and retained as head-survival-audit/attempt-002/survival-mass.png and
.svg. The stronger near-zero-tail hypothesis remains explicitly falsified.

### Experiment Card: bounded-head-recovery-v1

Revision 1; proposed; accepted revision none. Clean baseline
daa49334cef2b38e42cf16a30ef1cf0f1b8a2cee is a documentation-only descendant of
the fitted source e04b3490e60b4f7a7550f9abde20e6f551382dcc. Baseline endpoint is
20cea8e5d00e554c93141af7403a47e7108a3ee8fbd2c2dd57b8c9c415aa6205, with the nine
native outputs and complete Lens review above. This is a single architecture
intervention motivated by the located long-wait suppression, not capacity search.

Add a linear audio-only head base over the existing local/full-song encoding
(2250 parameters at the current widths), and use the existing head predictor
as a bounded residual: head logit = base + 4 * exp(-head_age_ms/1000) * tanh(raw).
At BOS the historical gate is zero. Initialize the base near .006 hazard and
reset the residual output bias from the old rate logit to zero. Keep every
other module, initial shared parameter draw, R1/frontier2 transfer, dataset,
sampling plan, optimizer rate family and decoding rule matched. Preserve the
unbounded model as an explicit configuration for old checkpoint reproduction.
The new flag and bound/decay values must reach the model and be checkpointed.

Closest analogue: the earlier local bounded-history timing study, applied here
only to the head generator's own skeleton history. No row-content feedback is
reintroduced. The innovation claim is limited to a task-specific recovery
constraint. The musical base need not predict acoustic onsets and does not
force events: sustained cues and off-grid placements remain in native support.
The intervention adds no hard minimum gap, source seed, endpoint teacher input,
post-sampling correction or old 30 ms preference loss.

First verify the exact residual bound, zero BOS history influence, recovery to
the audio base after long waits, shared-parameter initialization, direct row
audio, mirror symmetry and teacher/cached-native/query-partition parity in both
configurations. CPU and MPS likelihood/gradient agreement remains required.
The head law must be identical during fitting and sampling. Reject any broken
support, replay, normalization, fixed-clock coverage or noninterference check.

Use the same repaired corpus/normalizer/R1 pins and frozen 1200-update plan as
planned-head-release-consequence-v1. Model seed 230941, sample 230942, validation
230943; two songs by two 8-second intervals, 3e-5 inherited R1/consequence rate,
3e-4 audio/skeleton rate, AdamW decay .01, clip 1, MPS, CPU one thread.
Fresh 32-update preflight with six VAL songs, <=300 seconds, then a fresh
1200-update fit <=3600 seconds only if finite gradients, actual module updates,
reload and mechanically complete native probes succeed. Never reuse preflight
state or select a best-NLL checkpoint. No added capacity beyond the base readout.

Fresh owner artifacts/joint-audio/20260924-head-recovery-v1. Training names
planned-bounded-head-preflight-v1 and planned-bounded-head-main-v1 under the
repaired corpus planned-training directory. Use bounded_head=true, head_bound=4,
head_decay_ms=1000. No overwrite or implicit resume. Record clean intervention
source before fitting. Keep >2 GiB available RAM and >40 GiB free disk; stop on
nonfinite values or broken export. Same nine audio/seed cases, no head-spacing
filter, CPU one thread, <=90 seconds and <=30000 physical rows per chart,
<=900 seconds total native evaluation. Retain every outcome.

Primary failure-recovery diagnostic: from each of the four frozen failed head
prefixes, evaluate the new next-H CDF through five seconds with unchanged full
audio. The reference has 60/22/49/65 heads in those respective five seconds;
require CDF >=.99 in all four as a narrow recovery gate. This is not a quality
acceptance test. Also inspect all nine original native windows for recurrence
of complete silence; any zero-head context must remain visible and be explained,
not dropped. Count additional activity only as recovery evidence.

Qualitative guards: all generated time pages/actions/articulation via Lens,
preservation or improvement of the learned Who/Prom pulse organization, no
new ungrounded flooding of rests, no collapse to a repeated single pattern,
and meaningful LN articulation/expressive coverage. Review long-form gaps and
startup/dense-window compute separately. Fewer short gaps, more heads and lower
NLL alone cannot pass. If recovery occurs but rhythm, rests or LN structure
degrade, REFINE the factorization rather than promote it. Losing persistent
density/phase across rests and a poorly learned audio base are the strongest
alternatives to an overall improvement. Standing execution authority applies;
the Card stays proposed and no successful run implies acceptance or adoption.

### Bounded head implementation, preflight and active fit

Clean intervention 88e56832287f492fdca81ea7194b90753526cc96 implements the Card:
linear audio-only base, bounded head-history residual and elapsed-time decay;
all settings reach the model and checkpoint. Total parameters 4247438, an
increase of exactly 2250. Shared initial tensors remain identical except the
intentionally centered timing output bias. Existing row, release, preview and
frontier2 paths remain. No spacing filter or forced head is introduced.

Selected command uv run --extra mps --group dev pytest -q
tests/research/planned_audio_continuation tests/test_package_layout.py passed
20 tests and 22 subtests, including both modes' CPU/MPS probability/gradient
agreement and teacher/cached-native/chunk invariance. New checks prove the
residual bound, zero BOS gate, convergence to the audio base after long waits,
and unchanged shared initialization. A real old-checkpoint regression on the
new source reproduced all 213 Who seed-19 H times exactly (session 19598,
terminal success). The unbounded checkpoint stays reproducible.

Bounded preflight session 62089 completed 32 updates in 35.17838 seconds with
the same 128 intervals, 991062 ms, 6722 rows, 6217 H rows, 505 release-only rows
and 291439 release clocks. All tracked modules updated; new head_base L2 .13040.
Output: repaired corpus planned-training/planned-bounded-head-preflight-v1.
Freeze SHA 1ddf137e70a55dab75c6122f813a6b88c9a6d60df3f4569cc5eb3d2d1f45ef03;
result f3179a60305cd2e5451e938aebb5a6c418b0738d78110dcc8bfc5dda5c530ef0;
checkpoint f3f67568f35e623b7cb31cadfa656f456af07494ff6cc3ab5ce86793583c9ad9.
Population NLL/s 50.58295 and BOS 41.41946 are diagnostics, not selection gates.

Bounded preflight native session 64070 also completed: all nine outputs legal
and independently reparsed in 18.80157 seconds, zero all-held deadline releases.
First 30 rows .1462-.2558 seconds; 8-second readiness .1709-.2957 seconds, from
cached canonical Mel. Native result SHA
974acae9d1c0e9a5a0f43c693978b3f007727e479df39cdc43e3bb6987a03d10.
This preflight is a mechanical/learning gate; no new Lens quality verdict is
claimed for it. Owner artifacts/joint-audio/20260924-head-recovery-v1.

A fresh 1200-update fit is RUNNING in unified exec session 62534, observed
authoritatively through update 1020. Command: caffeinate -i uv run --extra mps
python -m ensomi_model.research.planned_audio_continuation.hydra
bounded_head=true head_bound=4 head_decay_ms=1000
run_name=planned-bounded-head-main-v1. Output is the repaired corpus's
planned-training/planned-bounded-head-main-v1. No preflight state was reused.
Revalidate the handle before continuing; never restart from a stale file or
observation timeout. Keep product source at the pinned revision through readout.

After terminal success, verify result.json/1200 updates and run, from the model
repository with uv run --extra mps python, the prepared owner scripts in order:
probe_frozen_prefixes.py, evaluate_native.py main, inspect_lens.py main.
The first measures the declared >=.99 five-second CDF at all four fixed failed
head histories. All native outcomes and all final generated context pages still
require inspection even if the CDF passes; reject mere activity inflation.
No recovery result, final bounded-model quality verdict or new playtest delivery
is available yet. Cards remain proposed and the overall goal remains active.

### Bounded main completion and release-boundary investigation

Bounded main session 62534 is now terminal success: 1200 updates in 867.45819 s,
same 4800 intervals/37253258 ms/253512 event rows/238904 H/14608 release-only
rows/10684532 release clocks. Checkpoint
47d41844afc673788cb640c67a1d5e41b2b9ca75ae679298c10072200925bce6;
freeze 9d5dc5fc0f1ee10446149ecdd91babaf34361ae00fa3500fdd4b960ca9ec5df8;
result 17e0d55325fc343d7d16d23ee26a832be0b27ae1c37676870e7b32556d2f4adf.
Population NLL/s 40.55605 and BOS 27.40233 are slightly worse than the unbounded
endpoint; this is not a rejection criterion by itself. Logged available memory
minimum 4907761664 bytes; driver maximum 3117776896 bytes.

Recovery probe session 66131 completed in 2.08020 s. Result SHA
cd1148a1b6200e889945aef9e762a9de1d12657387352cd0d162231a79d04dc1.
Who 19 / Death 19 / Good Luck 17 / Good Luck 19 five-second CDF changes from
.91924/.69511/.70648/.63836 to essentially 1/.93522/1/1. The declared all-four
>=.99 gate FAILED on Death Piano; report partial recovery, not full success.
Death's audio-base-only CDF is .79334, while the other three exceed .99998.

Main native session 46460 completed all nine cases in 15.20279 s, with independent
export/reparse success. Native result SHA
29257b415498fcb432819057a3012e64de21f0a65b7a45d63c06c54688dc0273.
All last heads now reach near the song end (Death 17 ends at 167212 ms, close to
its reference's 167204 ms, against 177372 ms audio). Largest internal H gaps
are 1.07-2.73 s outside Death Piano; Death remains sparse with 7.407/10.493 s
internal gaps. First 30 rows take .1468-.2581 s from cached Mel; full-pipeline
preprocessing and dense-source stress are not established by this measurement.

Lens session 59531 completed; manifest
ceb8572f1d5ebe428b9dd375a40e5de16eae6991a5bce00fd5f81d14fb2a18c7.
All 26 new generated pages and all actions/articulation were inspected; 14
references are byte-identical to previously viewed pages. Every fixed window
now has heads. Who has 58/0 and 62/0 heads/LNs with repeated pulse organization;
Prom Queen has 23/0 and 19/4 with repeated chords or tap/LN exchanges. Good Luck
has 16/4 and 10/4, including a two-hold passage with distinct releases in seed
17. Death has only 2/1 and 4/2 with multi-second LNs, still lacking demonstrated
fine rhythmic expression. Airborne has 14/1 instead of its former headless
window. These are meaningful but incomplete improvements; no listening/player
claim or overall playability promotion follows.

One Airborne all-held deadline event remains: at 39901 ms row (2,0,0,2) joins
holds in columns 1/2, occupying all lanes. The next H is 40211 ms. A release-only
row at 40210 closes columns 0/1/2, followed one millisecond later by a TAP in
column 1; column 3 remains held until 40549. Full [38200,41900) actions,
articulation and both time pages were inspected in main-lens/release-witness.
This is a specific release/re-press concern, not an anti-LN-duration rule.
The frontier's earliest possible release is 39902, which does not forecast the
actual release law. No new fit is running.

Experiment Card all-held-wait-law-audit-v1 revision 1, proposed, accepted none.
Baseline clean source 88e56832287f492fdca81ea7194b90753526cc96 and the fixed
bounded main checkpoint above. Inspect the Airborne seed-33 witness only.
Reproduce the complete output with behavior-neutral observation of the release
sampler's RNG state and threshold at 39901 ms; require exact row equality.
Read the unchanged raw release logits over 39902-40210, with actual committed
LN state and generated future heads, without future actual release actions.

Compare two mathematical waiting laws on those same logits: the implemented
deadline atom, which moves all surviving probability to 40210, and the raw law
conditioned on a release before the required H. Report total feasible event
probability, last-clock mass under each law, and the sampled quantile under
the same underlying uniform draw. This changes no trained model or published
chart. If conditional normalization shifts the observed quantile earlier and
reduces last-clock mass by >=10x, it motivates a training/inference-consistent
normalization intervention; it does not prove that new complete charts are
better. If not, inspect row-consequence and release forecast coupling instead.

Also inspect the four same-head-mask tap/LN variants at the 39901 ms decision,
with/without the consequence residual at fixed row context. No target suffix
loss or universal comfort margin is introduced. CPU one thread, <=120 s,
>=2 GiB available RAM, >=40 GiB free disk, <=20 MiB fresh owner
artifacts/joint-audio/20260924-head-recovery-v1/release-law-audit;
no overwrite/resume. Stop interpretation on reproduction failure. The legal
next-H condition distinguishes this problem from the head stream: one release
is required when all lanes are held, whereas a new head is not required during
every musical rest. Conditional normalization preserves every legal native
release time and may avoid an artificial boundary atom without a spacing ban.

### Release-law result and current handoff

Release-law audit session 90901 completed in 4.84353 s, reproducing every
generated row exactly. Freeze SHA
1291fd1e66787885ddbef91c8f9d501f1141520db8bc68016240b384bbe4fe12;
result d9141ebcf077625dfd884941fac7ea70548ffbbb70e5390ed9f6e7a5d8e7a53b.
At the all-held 39901 ms prefix, raw release probability through 40210 is
.656448. The implemented deadline atom gives its last clock probability
.344447; conditioning the same raw event law on a feasible release reduces
that mass to .00136408, a 252.51x reduction. The same underlying uniform draw
.920926 selects 40148 ms under conditioning, leaving 63 ms before H instead
of one millisecond. Both predeclared diagnostic conditions are met, but no
alternative full chart was generated and no conditional-law fitting occurred.

At the same row context and fixed head mask 0/3 with no releases, frontier2
reduces the all-LN kind probability from .710357 to .654503; its full-row
probability is .058511. The module is active. Its earliest-release slack of
309 ms is a possibility and does not capture the release law's boundary mass.
Do not conclude that the module was absent, or replace every four-hold choice
with taps. The law-level artifact motivates feasibility conditioning before
inventing a universal duration or release/head threshold.

The bounded main review is saved at main-lens/review.json, SHA
045677dabdcf7b0947c6b2531c8bd66c8fdacf4cc09afb9fb8835d8f298039f6.
All 26 generated context pages, 14 byte-identical references and two additional
witness pages are accounted for, with complete actions/articulation. Runtime
profile SHA d7ad51b8e1be0fb29cdacb241964b997eeb6703457ed3201f7e811d7dbf78679
uses the densest forward two-second generated windows: 15-28 heads, 8-18 physical
rows and .0187-.0418 seconds of publication work. This does not establish worst
dense-source or simultaneous-load performance. No waveform/Mel startup cost or
player verdict is included.

Product ee5a96de5d6387aa6ef1fe5aee69cbd2a626a904 is documentation-only after
88e56832287f492fdca81ea7194b90753526cc96. It adds the bounded comparison to
docs/research/head_wait_recovery.md and the self-contained probability analysis
docs/research/release_wait_conditioning.md. Product and notes worktrees are
clean after the local commits; nothing was pushed. All training, native,
inspection and diagnostic sessions in this phase are terminal. Do not rerun
finished jobs or invoke old scripts against a changed source pin blindly.

Current outcome is REFINE, not overall success. The four-case head recovery
gate fails on Death Piano, source-style expressive coverage remains incomplete,
and the release waiting law still uses the deadline atom in actual code.
Next Design/implementation should replace that atom for all-held/next-H waits
with a normalized feasible first-release law, identically in training and
sampling. It must include future hypothetical raw hazards in its normalizer,
retain full audio and proper local halos, preserve interval-censor telescoping,
and never read future actual LN states/tails. No minimum-gap rule is needed.
First compare the fixed bounded checkpoint with unchanged head RNG/stream so
the waiting-law effect is isolated; then decide the smallest matched fine-tune.
Separate further investigation of piano audio-base activity and difficulty/
arrangement ambiguity from this release intervention. Keep the old persistent-
intent fit deferred rather than applying it to the superseded flat model.
No new Card or fit for conditional releases has yet been created or started.
The full playable, expressive audio-to-chart goal remains active.

### Frontier2 migration verification

The owner reiterated the conceptual importance of frontier2 in the formulation.
The earlier flat joint path omitted it; the planned path explicitly instantiates
RowConsequence(hidden, frontier2), restores its weights, adds its output before
legal-row normalization, and trains it at the inherited R1 rate. Both interval
training and cached native generation call the same score function.

A read-only inspection of bounded checkpoint
47d41844afc673788cb640c67a1d5e41b2b9ca75ae679298c10072200925bce6 confirms all
8 frontier2 tensors, 27648 parameters, all keys in the transfer receipt, output
weight L2 .31058186 and inherited optimizer rate .00003. The existing completed
1200-update result records frontier2 parameter-change L2 .38334670. This verifies
module presence and updates, not preserved gameplay semantics.

Focused tests for actual transfer and learning-rate ownership, row/audio/preview/
consequence gradients, and both head modes' cached-native/teacher score and RNG
partition agreement passed: 4 tests in 1.49 s. Command was uv run --extra mps
--group dev pytest -q with the corresponding test_training.py transfer test and
test_distribution.py gradient/cached-native tests. Sessions 80865 and 35136 are
terminal success. No training or new generation was started by this audit.

Product f4f3a2f639d2f5423d3b1d35b4ee2d74cdeba192 changes only the information
contract and planned-model guide: flat-model omissions are explicitly scoped,
and active frontier2 training and its finite consequence limits are documented.
No model parameters, probability law or training target changed. The documents
passed diff --check; the product commit is local and the worktree is clean.

Retain the candidate-action to exact post-state to future-response comparison
path as architecture work proceeds. Its current release feature is only the
earliest possible release, while the scheduler supplies a distribution over
actual releases. They are distinct information objects. The finite module also
only passively advances clocks to the next H and receives the second-H gap;
it is not a calibrated evaluator of all legal continuations and horizons.
The preceding release-law normalization investigation remains the next bounded
implementation. Any later release-forecast query for a hypothetical candidate
must use its LN projection, full audio and the chosen skeleton, never future
actual rows/tails, and must not mutate state or consume publication RNG. NLL
alone cannot establish that these consequences preserve playability. Cards
remain proposed and the overall goal remains active.

### Experiment Card: feasible-first-release-v1

Card revision 1; owning Note 2026-09-23-audio-skeleton-r1-integration remains
proposed; accepted revision none. Standing local implementation/execution
authority applies. The previous goal turn made progress through source and
checkpoint verification plus committed clarification. There is no live fit.

Question: does conditioning the first release on occurring before a required H
remove the observed last-clock artifact while retaining usable LN organization?
Selected mechanism: replace the all-held deadline atom with q(u) divided by
the raw probability of at least one release by H-1. This is the finite-deadline
conditional survival law derived in docs/research/release_wait_conditioning.md,
not a new event representation or learned comfort penalty. The closest baseline
is that document's reproduced fixed-state raw-hazard comparison; there is no
novelty claim. Larger frontier forecast features and new row penalties remain
separate hypotheses, because they would obscure this waiting-law intervention.

Clean baseline f4f3a2f639d2f5423d3b1d35b4ee2d74cdeba192; fixed bounded checkpoint
47d41844afc673788cb640c67a1d5e41b2b9ca75ae679298c10072200925bce6, trained at
88e56832287f492fdca81ea7194b90753526cc96. Use exactly the nine audio/seed cases
in artifacts/joint-audio/20260924-head-recovery-v1/cases.json, Who/Death Piano/
Prom Queen/Good Luck seeds 17/19 and Airborne seed 33. Baseline complete-native
result 29257b415498fcb432819057a3012e64de21f0a65b7a45d63c06c54688dc0273;
all nine complete/reparsed, zero same-lane head/head gaps <=20 ms, one same-lane
release/next-head gap <=20 ms, and one all-held deadline release. These are
descriptive counts for nine fixed outputs, not population estimates.

The single causal intervention is the conditional first-release law when all
four lanes are occupied and a next H is known. Apply the same law in training
and native sampling. Add a checkpointed model/training flag for paired off/on
comparison; add no trainable parameters. Keep partial occupancy, true audio-end
closure, row/frontier2 scoring, head planning and independent RNG unchanged.
Training normalizers query the hypothetical unchanged LN state through H-1,
with full audio and sufficient local halo, never actual future releases/rows.
Interval censoring must telescope. The conditional last hazard is certain,
but its probability comes from raw q/Z rather than moved survival mass.

Implement a shared probability primitive, interval normalizer inputs, native
scheduling, typed config projection and focused tests. Necessary instrumentation
may record conditional wait counts/lengths and guard outcomes without altering
draws. Test analytic probabilities and gradients including the last raw hazard,
extreme logits, interval partitioning, future-tail noninterference, actual
CPU/MPS agreement, teacher/cached-native agreement and RNG chunk invariance.
Preserve old-checkpoint behavior with the flag off. Commit a clean intervention
revision before running real-model comparisons and record exact script hashes.

First comparison uses frozen weights and no optimizer: reproduce all nine
baseline rows exactly, then enable the new law with the same audio and seeds.
Primary improvement threshold is reducing the panel's total same-lane
release/next-head <=20 ms count from one to zero. Guards: all nine complete and
independently reparse, identical H timestamp streams, no new head/head <=20 ms
events, per-chart LN fraction change <=.05 absolute, and cached-Mel first-30-row
and eight-second readiness each <=1 s. The count thresholds are diagnostic
readouts, not legality rules or general comfort definitions. Native support is
unchanged, including one-ms intervals and four-lane holds.

Inspect the fixed Lens comparison scopes, every new close-gap witness, and
changed full-occupancy episodes through the following two heads. Reuse prior
visual judgments only for verified byte-identical evidence. Keep every output,
including failures/empty scopes. A removed witness with passing guards supports
another bounded train-consistent experiment; it does not establish overall
playability or justify adopting inference-only changed weights. Regressions
reject immediate promotion; no changed all-held encounter is an inconclusive
case rather than an improvement. Same checkpoint weights were fitted under the
old law, so this isolates decoding structure and cannot assess trained quality.

Environment: this Apple M5/24-GiB Mac, Torch 2.11, uv --extra mps, CPU one thread
for native comparison; actual MPS for gradient checks. Native limit 900 s total,
90 s/chart, 30000 rows/chart; stop at <2 GiB available RAM or <40 GiB free disk.
Fresh owner artifacts/joint-audio/20260924-feasible-release-v1, <=250 MiB added
artifacts excluding unchanged linked audio; no overwrite or resume. Expected
commands are uv run --extra mps --group dev pytest -q on the planned package
and package-layout tests, then uv run --extra mps python <owner>/compare.py,
then its Lens inspection driver. No network or fitting is required. Stop
causal interpretation on baseline reproduction, H-stream or probability
invariant failure. A later fit requires its own bounded proposed Card.

#### Feasible-release implementation and comparison handoff

Clean intervention 12d80eb3ae488dbe8c13083975d3e60e702f4ef2 adds the declared
condition_full_holds flag, shared conditional-hazard primitive, hypothetical
normalizer inputs/audio extent, native complete-wait query and typed training
projection/checkpoint persistence. It adds no trainable parameters and leaves
frontier2, head and row score functions unchanged. The baseline-to-intervention
diff is confined to the planned package, its tests and owning research docs;
no note paths enter the product tree/history. Defaults reproduce old checkpoints.

Selected package and package-layout check passed 34 tests and 22 subtests in
4.71 s, including actual CPU/MPS gradients, probability normalization, interval
probability/gradient telescoping, future-tail exclusion, full hypothetical audio
coverage, both head modes' teacher/native/cache/RNG agreement and config runner
round trip. An added all-rare MPS case then passed alongside its mixed-rate case
(2 tests in .82 s); no previously passing implementation changed after the
package check. Initial diagnostics found MPS softplus(-40) rounded to zero,
making log-rate gradients nonfinite; logaddexp(x,0) preserved the tiny rate and
resolved this. One initial source fixture lacked the loader's required post-seed
H; it was corrected without changing any product/source admission rule.

The frozen comparison driver is prepared at
artifacts/joint-audio/20260924-feasible-release-v1/compare.py. It pins this clean
intervention, checkpoint 47d41844afc673788cb640c67a1d5e41b2b9ca75ae679298c10072200925bce6,
the prior nine-case result and unchanged cases/descriptor. It will reproduce
every baseline row, generate the conditional arm, verify equal head streams and
unchanged weights, independently export/reparse, and retain full-occupancy
episodes for Lens. Its freeze records script/input digests. No fit is authorized
by this Card; native comparison and qualitative review are pending at this entry.

#### Feasible-first-release result

Native comparison session 32687 completed in 30.97785 s. All nine baseline
charts were reproduced row-for-row; all conditional charts completed and
independently reparsed with exactly identical H timestamps and unchanged
weights. Freeze 69df640147b6712601bf887053bba3e20a0f80b5c914f5f83a6028544fac43d9;
result e992e57c53e848abffc9a6c1b16c20e8ccd55c2047bec95c31fce3be9eaad1c9.
The primary RH<=20-ms count falls 1->0, HH<=20 remains zero, and all numerical
guards pass. Seven complete charts are unchanged. Good Luck 17 changes three
release clocks by 4/1/1 ms, retaining heads/kinds/long holds. Airborne realizes
the fixed-state prediction: R 40148, H 40211, gap 63 ms instead of one; its LN
fraction falls .149038->.138172. First 30 rows .15529-.25790 s and eight-second
readiness .15294-.33375 s from cached Mel. No conditional-law fitting occurred.

Lens main preparation session 94383 completed. Main manifest
daa5c57c5bd63fbeca23017d26b845445b0f9f07152d9500d10ee4f36ec6481a;
paired episode manifest cbdbee7e42108cd0100ba47b607242b77f8eb989005e2b3ae5b033578c57045d.
Viewed 14 new pages: four fixed-context pages, eight paired episode pages and
two Good Luck exit pages. Reused 36 pages only after byte equality with the
previous viewed review. Complete actions/articulation and pagination chains are
accounted for in review 9ad0f8b47d917db39c1d0ab522c0cf2aaec088a1b759d862352eac9f3fd2ba74.
An initial review-file assembly incorrectly required every pagination page to
be terminal; it was corrected to verify the complete chain, without dropping
any page or changing evidence.

The local release/re-press witness improves. Airborne now has a 73-ms LN at
40211, independent tails at 40730/40758 after the next two-LN row, and a 41-ms LN
at 41720. Their durations alone do not establish BAD. Its later fixed context
retains 14 H timestamps but loses the previous one LN, becoming all taps; there
is no blanket claim of musical/style equivalence. Good Luck's existing sustained
right-hand holds and joint exit at 202267 remain. No additional mechanical
failure was identified in the inspected contexts; listening/player approval,
fine Tech, dump/chordjack coverage and sparse-piano recovery remain unresolved.

Exposure audit session 56179 completed in 3.76238 s. Audit
8ec4c6ba78dd4e4a49472411fce0f35567496ee135e82f0a7b8d4ef9338304eb counts 1377
full-hold waits before a known H in 108/615 TRAIN charts. Their true first-R to
next-H gaps have min/median/max 39/97/1872 ms; none <=20 ms. The fixed update
plan encounters 787 interval segments in 215 updates, ten in its first 32;
107210 scored full-hold ms and 205480 hypothetical ms, maximum wait 7107 ms.
This describes target/exposure structure, not a universal spacing requirement.

Outcome REFINE. The frozen intervention meets its diagnostic thresholds and
supports a matched learning check, not overall playability adoption. Product
7c316e6ea3d22aff1fd4761798f1b4aa41d47e89 is a clean documentation-only descendant
of the tested implementation, recording these findings. No active job remains
from the frozen comparison. No code or notes were pushed.

### Experiment Card: feasible-release-learning-v1

Revision 1, proposed, accepted revision none. Standing local execution authority
applies. Question: can joint training under the feasible first-release law retain
the mechanical gain without destabilizing learned head/row organization?
Closest analogue is the completed bounded-head joint fit, with the frozen-weight
law intervention above separating the decoding effect. This is a probability-law
adaptation, not a capacity increase or a new comfort target.

Clean source 7c316e6ea3d22aff1fd4761798f1b4aa41d47e89 (code unchanged from 12d80eb).
Comparator is the existing fresh 1200-update bounded fit at 88e56832287f492fdca81ea7194b90753526cc96,
checkpoint 47d41844afc673788cb640c67a1d5e41b2b9ca75ae679298c10072200925bce6.
Use exactly its 615 TRAIN/240-group corpus, pinned normalization, R1 initializer,
model seed 230941, sample seed 230942, validation seed 230943, and frozen
34ef9751df4571f516ef2b670b3b0f4b398c212fdd16adcee1a029bd9168eb14 protocol.
Keep bounded_head=true, head_bound=4, head_decay_ms=1000. The only causal change
is condition_full_holds=true in both fitting and generation. Frontier2 and all
audio/head/release/row modules train jointly with unchanged learning rates,
AdamW/clip, batch/exposure plan and parameter count. Source-head teacher forcing
and hypothetical normalizers use no actual future tails.

First run 32 fresh updates: uv run --extra mps python -m
ensomi_model.research.planned_audio_continuation.hydra bounded_head=true
condition_full_holds=true run_name=planned-feasible-release-preflight-v1
updates=32 validation_every=32 validation_songs=6 max_seconds=300.
Require complete finite training, nonzero updates to all tracked modules,
checkpoint/config round trip, and all nine native charts completing/reparsing
before the full fit. The exposure audit confirms ten conditional waits in this
preflight. Do not select a quality endpoint from its likelihood or samples.

If the mechanical/learning preflight passes, train a separate fresh 1200-update
model with the same command and run_name=planned-feasible-release-main-v1,
default full updates/validation and max_seconds=3600; no preflight weights or
optimizer reuse. This matches the existing baseline budget rather than adding
an unmatched continuation stage. Use MPS, CPU one thread, this M5/24-GiB Mac,
>=2 GiB available RAM, >=40 GiB disk; per-fit bounds 300/3600 s. Fresh output
directories under the repaired corpus's planned-training owner; <=150 MiB each,
no overwrite/resume. Record source/config/transfer/protocol/endpoint hashes and
actual exposure. Stop on resource guard, nonfinite loss/gradient or invariant
failure; a failed fit remains evidence and is not silently restarted.

Evaluate the fixed endpoint with the same nine audio/seed cases and complete
Lens scope inventory as the frozen comparison. CPU one thread, <=900 s native,
<=90 s and 30000 rows/chart. Write preflight/main results under a fresh fitting
subdirectory of artifacts/joint-audio/20260924-feasible-release-v1, with <=250 MiB
additional evaluation artifacts excluding unchanged audio. Record all native
outcomes, including failures. Inspect all new fixed-context pages and close-gap
or full-occupancy witnesses, including entering/exiting holds.

Primary mechanical threshold: zero same-lane release/next-head <=20-ms witnesses
in the nine main outputs, against one in the original fitted baseline. Guards:
all complete and independently reparse, zero new HH<=20-ms witnesses, all nine
fixed windows contain heads, last-H/audio >=.85 in each song, LN fraction change
<=.05 absolute per case versus the original bounded fit, and cached-Mel
first-30-row/eight-second readiness <=1 s. These thresholds screen regressions,
not define playability or restrict legal source patterns. Inspect LN handoffs,
repeated figures and rhythmic detail rather than treating counts as a player
verdict. H streams may differ after joint fitting; the fixed-weight invariance
claim must not be carried over to this comparison.

Report factor NLL only as fitting diagnostics, with its changed law explicit;
do not select/reject on a lower total NLL alone. Passing mechanical guards plus
bounded visual plausibility can retain this endpoint for subsequent architecture
work. Failure localizes whether learning, head activity, release/row interaction
or resource behavior regressed. Sparse-piano and broad expressive coverage remain
separate unresolved goals. A result with only proxy gains or uncertain visual
effects is REFINE and cannot promote the model as a final playable system.

#### Feasible-release learning preflight

Preflight session 41633 completed 32 updates in 38.30641 s, with the unchanged
128 intervals/991062 ms/6722 rows/6217 H/505 release-only rows/291439 release
clocks. All tracked modules updated, including release_clock L2 .5948875 and
frontier2 L2 .0363215. The checkpoint contains condition_full_holds=true and
the unchanged 4247438 parameter count. Freeze
240c6d0dcc73ddd36d646f4edcf6bb8e6c2257726a3295c357c9c9093da4b304;
result 65546a189b3c8b42607e6c075e50929aaab12d64a7c1ba62288229be546a5a84;
checkpoint 7921f94440327c42df1462f96580cfb4aec321950d7e45bfe67eb5e5a7c990bf.
Population NLL/s 50.54437 and BOS 41.42026 are fitting diagnostics only.

Preflight native session 81813 completed all nine exports/reparses in 20.53865 s;
result 2bad69c631b0a4e7840656c6051062f33ceb2ade33ba8d0b4a48780bd27c4efe,
under the feasible-release owner/fitting directory. This early checkpoint has
5 RH<=20 and 40 HH<=20 witnesses and is not a quality candidate. The declared
preflight gate is finite learning, correct config and complete/reparsed native
execution, which passed; the zero-close-gap main gate is unchanged. No Lens
quality verdict or endpoint selection is claimed for preflight.

The full comparison now uses a separate fresh initialization and the same frozen
1200-update plan, with run_name=planned-feasible-release-main-v1. Prepared
fitting/evaluate_native.py and fitting/inspect_lens.py pin clean source
7c316e6ea3d22aff1fd4761798f1b4aa41d47e89 and the declared corpus/config. Do not
reuse the preflight model or optimizer. Record the actual full-run handle after
dispatch and verify it directly before continuation; a path alone is not a
running process. Overall playability remains unresolved and the goal stays active.

#### Running full fit and a constrained frontier follow-up

Full fit dispatched in unified exec session 34618 with caffeinate -i and the
declared Hydra command. It is verified live through update 170, 140.17 s, with
about 5.09 GB available RAM and 1.98 GB MPS driver allocation. Freeze confirms
source 7c316e6ea3d22aff1fd4761798f1b4aa41d47e89, condition_full_holds=true,
4247438 parameters and the unchanged protocol hash. Do not treat this partial
observation as completion. Preserve this clean product revision until the
prepared main native/Lens drivers finish; re-poll the same session directly.

A possible later frontier interface should preserve the distinction between
first-release timing and per-lane availability. The skeleton predicts the next
nonempty release-only event. Its CDF does not say which held subset the row
decoder will release, and an H row can itself release other occupied columns.
No per-lane freedom probability can be claimed from the timing CDF alone.

For a fixed current replay, candidate rows induce at most 16 distinct post-LN
projections: each previously held lane either continues its known start or
closes, and each previously free lane either stays free or starts at the current
clock. TAP versus idle shares the same LN projection. This can bound candidate
release-time queries without evaluating 256 separate timing networks. The
pending event's time and H/R role are already known, so its hypothetical updated
skeleton history is shared by these queries. Queries must not mutate committed
state or consume publication RNG. Actual candidate row histories remain distinct
for any later release-subset forecast; the 16-way reduction does not apply to
full future-row materialization.

Use the actual conditional first-release CDF as a possible feature, not the raw
feasibility normalizer as a calibrated stress cost. For any fixed normalized
event masses f(u) and any Z in (0,1), raw masses q(u)=Z f(u), with hazards
q(u)/(1-sum_{v<u}q(v)), produce the same conditional law. Full-hold conditional
likelihood therefore does not identify Z. A feature called feasibility stress
would be arbitrary if it changed under that transformation while the generated
release distribution stayed identical. Partial-occupancy survival mass retains
its ordinary probabilistic meaning because no release is required before H.

This is a bounded representational observation and an open follow-up, not an
implemented augmentation or a new Card. Await the matched fit's actual failure
contexts before deciding between richer consequence inputs, release-subset
forecasting, or the independently unresolved piano/difficulty-conditioning
problem. Do not add a universal release/head spacing loss or enlarge capacity
solely because the finite frontier approximation is incomplete.

#### Feasible-release main completion and inspected result

Full fit session 34618 completed 1200 updates in 907.01029 s with the exact
baseline exposure (4800 intervals, 37253258 ms, 253512 rows, 238904 H rows,
14608 release-only rows, 10684532 occupied release clocks). Checkpoint
67b8fc8fbac5f7ca2debe99524e29ac002546f12cb8a3c4c9acf68d68d7f3839;
freeze 94759f22b74379a0da3b5087e9e30bf5f1a8047b46f7f3519a659a17d613402c;
result fdda13e0fe689859b573f7598f641d02fded911b50fe15abcf72f0d3500e1f8c.
All tracked modules updated; frontier2 L2 .382798, release_clock 6.118215.
Population NLL/s 40.46334 and BOS 27.35083 are diagnostics, not acceptance.
Logged available RAM minimum 4329127936 B, driver maximum 3202727936 B,
RSS maximum 3793125376 B; these are distinct measurements, not additive.

Native session 70129 completed all nine outputs/reparses in 16.17506 s.
Native freeze 260778f389a01123e5970fa6534b07f098cf699ab209f2902b7fad3db3bd31aa;
result accf33033b475afdb808d7128ea6c2b614815092aa01c293dd2232e74a893235.
Assessment ad1e152b9773bb8de2c6d01568bbf6950e4af5dc30b151c4941e92f9eb88d127:
RH<=20 and HH<=20 both zero, all fixed windows contain heads, all last-H/audio
ratios >=.85, and both startup measures <=1 s. The declared .05 absolute
LN-fraction guard FAILS in six cases: Who19, Death17, Prom19, Good17/19 and
Airborne. Do not reinterpret the gate as passed or select on lower NLL.

Lens session 25684 completed. Manifest
421f7b1ec087791f400d89d79c11b30e09c27fa2a25ef881ae2f7ee0b419be87;
review 8916408cbe5d430642961ab1943271e49e53c2701f8350c9c2dea3a2a994340d.
Viewed all 26 new fixed-context pages and 14 witness pages for all nine
full-occupancy episodes in seven merged contexts. The 14 human reference pages
were byte-identical to previously viewed evidence. Complete actions/articulation
were read and pagination checked. No training or native job remains live.

Who retains roughly 117/234/468-ms tap organization; seed19 includes six short
sequential holds in its fixed window. Prom Queen gives regular chord/single tap
figures near 208 ms; seed19 loses its former tap/LN arrangement. Death Piano
remains sparse: seed17 has one 3274-ms hold with no interior head and a 6900-ms
following-H gap; seed19 gives two new holds in the same fixed context. Fine
Tech/dump expressive coverage is not established.

Good17 moves from 825 to 1344 note heads while H-row count falls 671->626;
mean chord size rises 1.22951->2.14696. LN heads increase 204->227 even though
LN fraction falls .24727->.16890. This is not LN disappearance or a global
sparsification win. Recurring three-column grips and hold interactions are
visible, including independent releases while another column stays held.
An isolated 22-ms LN at 189468 has unresolved musical role; its duration alone
is not a universal BAD label. Airborne rises from 279 to 449 LN heads and has
one full-hold release 66 ms before its next H. Its fixed window contains 19
heads/four LNs and some separate hold/tap articulation. These changes are not
automatically bad arrangements, but the endpoint is not a clean overall gain.

Outcome REFINE. Keep the positive fixed-weight waiting-law result distinct
from joint-fit composition sensitivity. A shared seed does not preserve latent
arrangement choice after probabilities change, and H streams can change when
shared audio parameters are jointly trained. The next diagnostic should isolate
generated head-plan effects from the remaining conditional model before another
fit or a persistent-intent extension. Audio source inspection also confirms
load_audio_file already applies whole-song peak normalization; do not attribute
piano sparsity to an absent global-gain normalization without further evidence.

### Experiment Card: generated-head-plan-crossover-v1

Revision 1, proposed, accepted revision none. Standing local experiment authority
applies. Question: do the two largest composition shifts follow the generated
H plan or the conditional audio/release/row model at fixed audio and draw seed?
Closest analogue is the fixed-weight release intervention; the implemented
head-plan factorization permits a controlled modular crossover. This diagnoses
coupling and trajectory sensitivity; it does not estimate an additive fraction
of all BAD patterns attributable to R1.

Clean source 7c316e6ea3d22aff1fd4761798f1b4aa41d47e89. Conditional model A uses
bounded checkpoint 47d41844afc673788cb640c67a1d5e41b2b9ca75ae679298c10072200925bce6
with condition_full_holds=true; model B uses the trained conditional checkpoint
67b8fc8fbac5f7ca2debe99524e29ac002546f12cb8a3c4c9acf68d68d7f3839. Use the exact
Prom Queen seed19 and Good Luck seed17 audio/cases. Their H plans are taken
only from the two complete audio-generated endpoints, never source charts.
Diagonal A is the frozen conditional arm under the feasible-release owner;
diagonal B is the fitted main arm. Baselines: Prom LN fraction .180428/.020339;
Good mean chord size 1.229508/2.146965. Nine-case aggregate guards failed as
documented; these selected cases explain shifts, not overall failure rates.

For each case, evaluate A/B conditional weights crossed with A/B fixed H plans,
using the original row/release RNG seeds. The sampler's head planner is replaced
only by a complete immutable generated queue with identical preview/end semantics.
It must not expose any future rows, tails or occupancy. Audio encoding, release
law, row/frontier2 scores and legal support remain those of the selected model.
The conditional-weight factor includes audio encoder, release and row modules;
it is not a pure R1-only intervention. No parameters are changed or fitted.

Require exact full-row reproduction of both diagonal outputs and exact supplied
H timestamps for every arm before interpreting crossed arms. Record all row
and release outcomes, full-occupancy episodes and close-gap witnesses. Inspect
the same fixed Lens scopes for crossed outputs and any new mechanical witness.
Do not treat a lower LN fraction as an improvement by itself.

Primary readouts are Prom's LN fraction and Good's mean chord size. Compare each
crossed value with the two diagonal values by absolute distance. If both crossed
arms are closer to the diagonal sharing their H plan, prioritize plan-to-row
response. If both are closer to the diagonal sharing conditional weights,
prioritize the conditional audio/release/row policy. Otherwise classify the
result as interaction or sample-path ambiguity. Also report all four values,
heads/H rows/LN counts/held time and differences, not just the classification.
No global causal-percentage or style label follows from this two-case probe.

CPU one thread, <=240 s total, <=90 s and 30000 rows per chart, >=2 GiB available
RAM and >=40 GiB disk. Fresh owner
artifacts/joint-audio/20260924-feasible-release-v1/head-plan-crossover;
<=100 MiB excluding linked unchanged audio, no overwrite/resume. Eight rollouts
including the four diagonal reproduction controls, independent export/reparse,
then Lens inspection. Freeze the driver and input/checkpoint/plan hashes before
sampling. Stop causal interpretation on reproduction, H-stream, resource or
legal-completion failure. A mixed result calls for a common-state probability
or additional-seed diagnostic before adding latent capacity.
