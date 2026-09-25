# Agent Note: Audio skeleton generation and R1 integration

Note ID: 2026-09-23-audio-skeleton-r1-integration
Status: proposed
Kind: research
Created: 2026-09-23
Updated: 2026-09-25
Product revision: 4142ba4b01c338e76c9ce021491a5bcded68a636
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

#### Generated-head-plan crossover result

Session 79058 completed all eight rollouts in 9.77881 s. All four diagonal
controls reproduced their complete saved rows exactly; every crossed arm
preserved its supplied generated H timestamps and completed/reparsed. Freeze
1b60968c96ad0997d641384cf3181bedc9f4254af9c14fbb77efbe44f73d9310;
result 2b3b9315d384e5458f9a631384b9b830f3de0f328debd8ebec6d112144950284.
No parameter changed. Arm XY denotes conditional weights X and H plan Y.

Prom Queen19 LN fractions AA/AB/BA/BB = .180428/.028476/.171053/.020339.
Good Luck17 mean chord sizes = 1.229508/1.932907/1.214605/2.146965.
Both declared primary readouts follow H-plan origin under the predeclared
distance rule. Do not generalize this to every composition feature: Good BA
LN fraction is .088344 versus AA .247273 despite similar mean chord size.
The weight factor includes audio, release and row modules, not R1 alone.

The crossed Good outputs expose an HH19-ms and an RH11-ms witness even though
both corresponding diagonal outputs lack <=20-ms pairs. Initial Lens preparation
incorrectly asserted no new close-gap witness and stopped; its partial lens/
directory is preserved. Fresh lens-v2 includes all witnesses/hold episodes and
uses hard-linked immutable source evidence with separately written metadata.
All original baseline bundle bytes were reverified unchanged. New unshared
artifact bytes 74716714 remain inside the declared 100-MiB addition bound.

All 18 new crossover pages and complete actions/articulation were inspected.
Manifest d092202ee55be707c6bdba8fffaf62f849513c40bffc5553668e3133e36ca94c;
review bbaa35677281605ef264da1ff616bdd0e0c2a5670cbf8a869ea2f1d0c74ed06a.
The AB fixed Good scope has chordal holds with independent tails; BA is mostly
single taps there. Five AB full-hold episodes include independently closing
outro holds. These are not categorically bad simply because composition differs.

The two mechanical witnesses have different dependencies. AB commits taps in
0/1/3 at 47163, followed by mandatory H at 47166 and 47182. Five heads in four
lanes within 19 ms force a repeat after that first three-head choice; its
realized repeat is column 1. Retaining those three H clocks while reducing the
first chord to at most two heads can avoid that local repetition. BA closes a
column-2 LN in an H row at 187136 (also tapping 0/1), then taps column 2 at
187147. Column 3 remains available, so the latter re-press is avoidable at the
second row. Its tail belongs to an H row, not a release-only deadline; the
full-hold normalization does not directly schedule this failure.

Outcome REFINE: generated timing materially changes row organization in these
selected trajectories. This does not prove population degradation, an R1 error
percentage, a need for a latent, or permission to filter high-fraction H times.
Audit the actual candidate scores at the upstream commitment and the later
avoidable re-press before adding more model structure.

### Experiment Card: frontier-witness-score-audit-v1

Revision 1, proposed, accepted none. Source remains clean
7c316e6ea3d22aff1fd4761798f1b4aa41d47e89. Use only the two completed Good Luck17
crossed outputs and their exact checkpoint/audio/H-plan bytes from the crossover.
No fit or changed rollout is requested. Reproduce each full output with
behavior-neutral capture of current replay, row inputs and consequence scores.
Require exact full-row equality before interpreting captured distributions.

At AB time 47163, verify no active hold and all prior head/release clocks more
than 20 ms old, with the next two H at +3/+19 ms. For each candidate with k
heads, the minimum number of short repeated future heads is max(0,k+2-4):
the unused columns can serve the next required heads, while any excess must
reuse a lane. This is a local diagnostic under the declared 20-ms comparison,
not a legality restriction or a calibrated demand quantity. Also inspect AB
47182, where the committed past already makes every column recent, and BA
187147, where only column 3 can avoid a <=20-ms head/release relation.

On identical captured row contexts compare the complete probabilities with the
current frontier2 score, with that score removed, and with the released R1
frontier2 tensors evaluated on the same current inputs. The last comparison
isolates those tensors under the current representation; it does not reproduce
the old R1 policy or prove universal forgetting. All other logits/support stay
fixed. Pin released R1 SHA
4b3ec1561e33d0ebe2756cfe13571ec414fd5bb470b430f0c578545863115f70.

Report probability mass on positive diagnostic cost, expected diagnostic cost,
the sampled row probability and highest-probability safe alternatives. If the
current score leaves >=.05 risk mass at the upstream or avoidable decision,
treat it as a material local tail for subsequent response-alignment work. A
>=2x reduction versus score removal indicates active mitigation; increased
mass or a change under 10% is a weaker/misaligned response at that state. Mass
below .01 favors a rare-tail/additional-seed explanation before architecture
changes; intermediate values remain uncertain. The forced later row cannot
repair the upstream commitment and must not be misclassified as an independent
avoidable error.

CPU one thread, <=120 s, >=2 GiB RAM and >=40 GiB disk, fresh
artifacts/joint-audio/20260924-feasible-release-v1/head-plan-crossover/frontier-audit,
<=5 MiB outputs, no overwrite/resume. Freeze script/input hashes. No publication
RNG consumption by score comparisons, no actual future rows/tails as inputs,
no new training loss or minimum-gap decoding rule. This is a three-state local
diagnostic; it cannot by itself select a globally playable endpoint.

#### Frontier witness result and confirmed quality criterion

Score-audit session 79289 completed in 3.42639 s and reproduced all 674 AB and
676 BA rows exactly. Freeze
39ceff1aef2e040c142ef4bfc9a39f48f876cb0e462aa7cfc883a61ffa23524d;
result da94eadf75b24f04a294a9056972866c0f0a164a4f95118dec4a9297aa4fb104.
At 47163, risk mass without/current/released frontier2 on the same inputs is
.491618/.458580/.469805. Current frontier2 is active but reduces this local
mass by only 6.72%, below the declared 10% weak-mitigation comparison. The
chosen three-tap row has .233792 probability; the most probable zero-cost row
has .167541. Greedy decoding on this fixed prefix would also select a
positive-cost row. At 47182 every legal head has a short prior attack, so no
later choice repairs the earlier commitment.

At 187147, the combined head/release close-gap mass is .060178 without the
module, .051879 with current weights, and .051029 with released weights on the
same inputs. A safe column-3 TAP already has .940813 probability; the sampled
column-2 TAP has .046472. This is a remaining sampling tail, unlike the earlier
commitment where positive-cost choices collectively carry almost half the
probability. Restoring the released tensors alone does not solve either state
under the current inputs. The comparison does not reproduce the old R1 policy
or establish a general forgetting claim. Outcome REFINE, with no new fit.

On 2026-09-24 the human owner explicitly confirmed successive same-column
attacks strictly below 20 ms as an almost-certain bad pattern to avoid. Treat
this as a high-confidence negative and an independent generated-chart quality
failure. Attack includes TAP and LN head. Exactly 20 ms, cross-column attacks,
short LN duration, and release-to-head gaps are not classified by that statement.
Keep physical legality separate. The 19-ms witness is now confirmed within this
criterion; the separate RH11-ms witness remains a release/action concern.
The score audit's selected states contain no exactly-20-ms boundary, so its
specific HH findings are unchanged by strict versus inclusive notation.

Product 9a4f1fea5ec5ac37738b28a43ce2c0c9533f863a is documentation-only after the
tested 7c316e6 source. It records the strict quality criterion, matched fit,
head-plan crossover and candidate-score analysis in their curated owners.
Selected documentation diff checks passed; all model runs are terminal and
both worktrees are clean after local commits. Nothing was pushed. The global
playable/expressive goal remains active. Do not restart completed fits.

### Experiment Card: short-attack-response-policy-v1

Revision 1, proposed, accepted none. Standing local execution authority and the
new explicit human bad-pattern criterion apply. Clean baseline
9a4f1fea5ec5ac37738b28a43ce2c0c9533f863a, code unchanged from 7c316e6.
Question: can a small candidate-response selection policy avoid the confirmed
HH<20-ms failure while preserving tightly spaced cross-column H events and
already healthy generated decisions? This is a decoding-policy test over a
fixed learned proposal, not a new claim about its fitted likelihood.

Closest analogue is R1's optimistic response preference, but restrict this
intervention to the human-confirmed strict attack/attack criterion. Enumerate
the minimum HH<20-ms count over the current candidate plus required H events
strictly before current time +20 ms. Use one hypothetical TAP per future H,
exact post-candidate attack clocks/occupancy, and earliest legal unknown-LN
release opportunities. A 16-mask lane-use dynamic program suffices inside
this horizon, since any lane used twice there incurs a short repeat. Include
all previewed H events in the horizon rather than blindly stopping at two.
No future actual actions or tails enter. This optimistic release assumption
is a lower bound and is not a forecast of the release model's choices.

First sample the original complete-row proposal with its original row RNG.
If its response cost is already minimal among legal candidates, preserve it
exactly and consume no correction RNG. Otherwise select from the minimum-cost
candidates using the proposal probabilities, prioritizing minimal change in
head count, LN-start count, release count and lane actions, in that order.
Use a separate seeded correction RNG so healthy original decisions keep their
original draw stream. This explicitly defined correction kernel need not equal
the raw proposal distribution. No new parameters, temperature change, H filter,
LN-duration floor or generic anti-Jack objective is introduced.

If every legal candidate has positive minimum cost, choose only the minimum
and record the unresolved decision; do not claim successful avoidance. The
relaxed future can also underestimate risk when optional releases do not
materialize. Full-chart strict HH<20-ms witnesses remain the decisive quality
test. Keep the RH diagnostic separate rather than silently extending the
owner's threshold to releases.

Add an optional default-off runtime policy, a strict attack-gap diagnostic,
the finite response evaluator and focused invariance tests. Verify the
47163/47182 cases, exact-20-ms boundary, TAP/LN-head counting, cross-column
events, mirror symmetry, and that old checkpoints still reproduce with policy
off. Audit the known corpus: identify charts containing the strict bad pattern,
without deleting or relabeling them; for charts with no such pattern, verify
that each true row has zero response cost under its own true H plan. This
checks support preservation for real arrangements without using their suffix
actions as inputs to the candidate evaluator.

Freeze a clean intervention revision before model comparison. Use checkpoint
67b8fc8fbac5f7ca2debe99524e29ac002546f12cb8a3c4c9acf68d68d7f3839 on the fixed nine
native cases, plus the two Good Luck crossed arms from the completed probe.
Require exact reproduction with policy off, unchanged H timestamps with policy
on, and zero strict HH<20-ms pairs in all eleven policy outputs. The nine
standalone baseline outputs already have no close pairs; their unchanged rows
are a strong noninterference check because a valid realized suffix witnesses
a zero-cost relaxed continuation. Do not promote if this property fails.

Guards: all charts complete and independently reparse; no increase in aggregate
RH<=20-ms count above the eleven-case baseline's one; per-case total note-head
count >=.9 of baseline and absolute LN-fraction change <=.05; cached-Mel
first-30-row/eight-second readiness <=1 s. Inspect every changed output's fixed
Lens scopes, every correction episode and any remaining close-gap witness.
Correcting one chord must not be credited as success if it replaces the rest
of the chart with sparse or incoherent material. Original human evidence is
unchanged; no listening/player verdict is assumed.

CPU one thread, model-backed commands with uv --extra mps; <=600 s total native
comparison, <=90 s and 30000 rows/chart, >=2 GiB available RAM and >=40 GiB disk.
Corpus support audit <=300 s, no fit. Fresh owner
artifacts/joint-audio/20260924-short-attack-response-v1, <=200 MiB added artifacts
excluding unchanged linked evidence/audio; no overwrite/resume. Stop on source
reproduction, physical support, H-stream or resource failure. A successful
bounded policy comparison motivates response-aligned learning or scheduler
refinement; it does not complete the remaining musical/style/dense-runtime goal.

#### Short-attack response result

Clean intervention source 7ea2e82eebd9afbc97ec8db23cf361885fc8e0a1 implements
the default-off policy, strict diagnostic and finite response evaluator. The
selected planned-audio suite passed 40 tests in 5.95 s. Tests cover the witnessed
upstream/dead-end states, strict boundary, TAP/LN heads, exhaustive relaxed lane
assignments, mirror symmetry, minimal changes, independent correction RNG and
default-off/healthy-row noninterference. No model weights changed.

Corpus audit completed in 27.29748 s: all 615 TRAIN and 36 VAL charts have zero
strict HH<20 pairs, and all 683341 true physical rows have zero optimistic
response cost under their own H plans. No target was removed or relabeled.
Corpus result 5f9d45de29756e188b6b6568e0887060b1c1b9dd1e3ecb2d7e2409a0b73372c6;
per-chart details 1d8847473d9d124d74df49f497827b6de31dd805cbde526f137883a88dcbd4fc.

The 11 paired native cases completed in 38.13990 s. Every off output exactly
reproduced its frozen rows; all on H streams remain identical. All nine B native
outputs and the Good Luck B/A arm remain row-identical with zero corrections.
Good Luck A/B changes only one proposed decision through the kernel: at 47163,
TAP0/1/3 becomes TAP0/3. Subsequent TAP2 at 47166 and TAP1 at 47182 realize the
same H plan without the strict short repeat. All eleven corrected outputs have
zero strict HH pairs and unresolved minima. Head-count, LN-fraction and cached
startup guards pass; Good Luck A/B heads 1210->1257, LN fraction
.1942148760->.2155926810, mean chord 1.9329073482->2.0079872204.

The aggregate RH<=20 guard FAILS, increasing from one to two. In the changed
suffix, four holds end at 193538 and column 3 taps at 193542. This is a new
release-only event followed by H, over 146 s after the upstream correction.
The strict HH improvement does not establish an overall quality gain. RH is a
separate diagnostic, not an extension of the owner's confirmed HH threshold.
Keep the policy default off; do not repair this result by silently changing the
guard or adding another local threshold.

Native freeze 69db174bba2ced7f2fbf32657dd0841425e50146c2817117812a752b184da0f9;
result 1c5fcedf09f8c26d4e51015867d0eec7371e5c86718ef8c53c9d48c1a2cdb7ce.
Lens manifest a9f0b01c74129548ae775c4fdc590d7d53f3adfd6d665de2bbc05b926cab91fd;
review b708a90c48f08d58b9fbb299a9dc3c3eeeb525d93e99155da83b7cfe3c265c8a.
All eleven new time-proportional pages and complete action/articulation views
were inspected, covering the correction, fixed musical scope, late occupancy
episodes and new RH4 witness. The changed chart retains chords and independent
LN releases, including a 1838-ms LN spanning four later H rows. Listening and
player judgment remain unperformed. Parent human evidence was reverified and
unchanged. Result recommendation REFINE; no promotion or lifecycle transition.

#### System priority

On 2026-09-24 the human owner expanded the standing goal to prioritize the whole
real-time generation system and avoid getting trapped optimizing small flaws.
Conclude this bounded rule experiment here. Keep the learned audio/skeleton/row
pipeline as the main research object, with quality constraints at its commit
boundary. Investigate end-to-end readiness and worst workload before adding
speculative decoding, capacity or more isolated correction rules.

### Experiment Card: full-audio-playback-budget-v1

Revision 2, proposed, accepted none. Clean source
7ea2e82eebd9afbc97ec8db23cf361885fc8e0a1. This is behavior-neutral system
profiling, not a new fitted or decoded policy. Use conditional checkpoint
67b8fc8fbac5f7ca2debe99524e29ac002546f12cb8a3c4c9acf68d68d7f3839 with correction
off. Primary question: do full-audio startup or dense sequential execution
actually require a more complicated real-time scheduler? The closest local
analogue is the joint-audio new-file inference path; the planned model adds
full-song encoding, head lookahead and LN-conditioned release scheduling.

Run the frozen nine native cases with fresh waveform decoding and canonical
Mel computation (no Mel cache input). Require exact baseline rows and full
audio-duration equality. Add two explicitly labeled source-H workload probes,
chosen before execution from the already-pinned 651-chart corpus: maximum H
count in any half-open 8-s window, and maximum decoded song duration; ties by
source SHA. Dense source 15e5e1949e331bcb2530d8d3ef319849ecf1e419c850d13e3a638151c84d30c0
has 180 H in [53662,61662), duration 115509 ms. Long source
80d8b34bba60b4cb6ab9cd49340ab397a824e762cc11cc3d0bde8c277414d63d lasts 585495 ms,
with 4325 H rows. Source Hs test service workload, not autonomous timing quality;
no source actions/endpoints condition generation. Both use seed 230971.

Record import/model-load, decode, Mel, encode, first 30 physical rows, first
eight seconds of settled coverage, total generation, per-update coverage and
wall time, physical validity and strict HH diagnostics. Freshly decode each
native case rather than reusing a prior Mel array. A resident model is loaded
once; report its one-time startup separately. Existing cached-Mel startup is
the paired reference, not an end-to-end latency claim.

Evaluate a virtual player with a declared 2-s presentation lead. Start after
both 30 rows and eight seconds of settled coverage (or true completion for
short charts). A coverage update certifies all rows/no-events through its
clock; a last note timestamp alone is not that watermark. Check deadlines
immediately before every later update using the preceding coverage. Report
minimum slack, any misses, and maximum measured service seconds for advancing
an 8-s audio window. Replay the measured generation trace at 1x, 4x and 10x
wall-cost multipliers, with each policy's corresponding startup; also add one
virtual 1-s stall at the worst 8-s service window under 1x. Do not consume
model RNG or change content during this trace simulation. These are workload
experiments, not OS worst-case guarantees or measurements of the client.

A resident-model readiness above 3 s selects preprocessing/encoding work;
an 8-s service interval above 2 s, or a 4x trace deadline miss, selects compute
scheduling/throughput work. If neither occurs, prioritize musical distribution
and controls while keeping a simple buffered scheduler. Record failures rather
than tuning thresholds afterward. Physical completion and native row
reproduction are required; quality of new source-H outputs remains unreviewed
until inspected, and does not substitute for native generation quality.

CPU one thread on the current Mac, uv --extra mps, <=900 s total and <=90 s /
30000 rows per chart; >=2 GiB RAM, >=40 GiB disk. Fresh owner
artifacts/joint-audio/20260924-playback-budget-v1/run-v2, <=200 MiB new outputs, no
overwrite/resume. Save source/script/model/audio/manifest hashes and protocol
trace. Stop on resource failure, incomplete generation, changed bytes, invalid
reparse or native reproduction failure. No fitting, source label changes,
runtime fallback, speculative sampler or remote publication in this probe.

Revision 1 aborted in a synthetic instrumentation check before model loading
or any real case. Its synthetic trace unintentionally missed two deadlines
while the assertion expected one. Preserve probe.py and failure.json in the
parent owner; probe_v2.py uses a final update at the intended exact boundary
and a fresh run-v2 destination. The deadline calculation, real cohort, decision
thresholds and resource bounds are unchanged. This is not model evidence.

#### Full-audio playback result

The revision-2 probe completed all eleven cases in 37.24790 s at clean source
7ea2e82eebd9afbc97ec8db23cf361885fc8e0a1. Freeze
0a40f2a5f65800315a062c311c9365874f509bec3f6a30dac57146d1d6b2c913;
result d7a07c92412562e23d5b0cf9deef0751af0ca24fc842c98849cfd82f60078b2f.
Output bytes 52548605, below the 200 MiB bound. No weights changed; every
native baseline row sequence reproduced after fresh decoding and Mel.

Resident-model readiness, including input verification, waveform decoding,
Mel and generation of 30 rows plus eight seconds of settled coverage, is
.553436..1.051728 s for the nine native cases. One-time imports .879701 s and
model load .168325 s are separate. The dense source-H case is ready in .550150 s,
generates in 3.833720 s from Mel, and its slowest measured 8-s service interval
costs .332567 s. The 585495-ms source-H case is ready in 1.898674 s, generates
in 8.240929 s and has a .211049-s worst 8-s service interval. Native worst
8-s intervals range .093067 to .215081 s. Both threshold-trigger lists are empty.

All charts complete/reparse and have zero strict HH<20 pairs. Virtual deadline
misses are zero for every case under 1x, 4x and 10x generation cost, with scaled
startup readiness. They are also zero with a single 1-s virtual stall at each
trace's worst 8-s workload. Minimum slack is 5.998150 s at 1x and 5.981498 s
at 10x. This is trace replay, not actual OS jitter, client rendering or network
measurement. Source-H outputs are service probes and remain qualitatively
unreviewed; they are not evidence for autonomous skeleton quality. No actual
player/listening verdict occurred.

Outcome REFINE. A simple buffered scheduler has substantial headroom on this
cohort. Do not add a draft model or enlarge parameters solely for presumed
latency pressure. The result does not prove universal runtime capacity or
resolve musical sampling, composition consistency, control or release quality.
Product fdc6c13a51c8d8943fd44b3a632ee32bb4e6d33d adds only curated prose after
the tested code, including the dependency graph, settled-coverage deadline
equation, observed budgets, scope limits and research direction. Local links,
Markdown fences and diff checks passed. The earlier 40 selected model tests
remain valid because code did not change. No remote publication occurred.

#### Next system work and model branch

The runtime deliverable is one source-chart-free planned-model entrypoint
that reads audio, generates from BOS, emits complete rows plus settled-through
coverage, retains incremental LN state, and saves a reparsable playtest export.
It should expose actual end-to-end startup and use independent head/release/row
RNG. Keep the validated simple scheduling order; do not invent compute-triggered
sampling shortcuts or truncate full audio. Packaged settings must follow the
Hydra workflow. This entrypoint is implementation work, not proof of a playable
model. It is now implemented and verified as recorded below.

The selected next modeling branch is a shared interpretable arrangement
condition for head, release and row generation, rather than more isolated
decoding fixes. A descriptive TRAIN audit finds 184/240 exact audio assets have
multiple charts. Median within-audio ranges are 2.825697 H rows per audio second,
.364120 heads per H, and .093173 LN-head fraction. This motivates separating
audio from a sampled/requested arrangement choice, but does not prove that the
current autoregressive model lacks expressive capacity or that a latent will
improve generation. The audit changes no corpus labels or fitting inputs;
its output is arrangement-statistics.json in the playback owner, SHA
d3522363ec0efec4278b9398ec3192df056d61cbe142d8c83bed15b7a6b730cc.

Before a fit, write one bounded matched Experiment Card for the shared-condition
branch. Start with those three interpretable source statistics, not a universal
style or difficulty scalar. Preserve the full-audio path, separate charts,
direct audio-to-row condition, own-skeleton history, LN-only feedback and legal
native-ms support. Any chart-derived training condition must have an explicitly
learned audio-only prior or user-supplied counterpart at inference; hidden
reference statistics are not a native test. The prior must model joint plausible
profiles rather than three unrelated componentwise averages. Missing human
style annotations stay unreviewed. NLL is diagnostic only; compare realized
conditions, native musical scopes, complete-chart failures and model budgets.

Do not resume finished jobs. No live training or probes remain. The ultimate
playable, expressive system goal remains active; runtime success and the narrow
attack correction do not complete it. No Note acceptance or lifecycle transition
has occurred.

#### Source-chart-free streaming entrypoint verification

Implementation source 1c95f6914fd3fa390d8a46f1d267d5311def7078 adds the packaged
planned-audio inference schema/CLI, synchronous event callback, flushed JSONL
stream, full-audio preprocessing, BOS generation and independent osu! reparse.
It preserves the sampling implementation and checkpoint weights. Default
correction remains off. A row and inclusive settled-through clock are delivered
together. Open LN heads carry no future endpoint. Readiness requires 30 rows
and eight seconds of coverage, or true completion for shorter charts. Capped
generation keeps its prefix/open holds without inventing tails. Consumer errors
propagate with the prefix and error record preserved. No crash-resume promise,
client transport or playback engine is added.

Typed projection rejects unknown fields, and settings reach runtime consumers.
CLI help/config inspection is Torch-free. The selected command
uv run --extra mps --group dev pytest -q tests/research/planned_audio_continuation
tests/test_package_layout.py passed 47 tests and 22 subtests in 5.19 s. Tests
include real waveform/checkpoint loading without a corpus, native row parity,
stream replay, early readiness, partial LN preservation, callback failure and
preprocessing limits. Local Markdown links, fences and diff checks passed.

The actual CLI then reproduced all nine fixed conditional-release outputs,
with a separate process consuming stdout online and replaying exact LN state.
Every row sequence matches its frozen baseline; every stdout stream matches
events.jsonl byte for byte. Readiness arrives before process exit, in
1.398880..1.843109 s from subprocess launch, including imports and model loading.
Producer-side readiness is .545364 to .966192 s under its narrower profile.
The complete verification took 28.987641 s. No new chart-quality claim follows
from these identical outputs; earlier Lens inspection remains the evidence.

Owner artifacts/joint-audio/20260924-stream-entry-v1; freeze
752bfa3908a192a075af22a9c85233b0ff5d771313a2f0dd24ecf9b6c72af21a;
result 2eef5f7b1d4ab6b71e9420b665512990424524aa3d397db6f7e6cc6bc33922b4.
Nine osz previews package the unchanged generated chart and paired audio;
archive CRC and extracted byte equality were checked, with identities in
playtests.json. No listening or player verdict occurred. Product
9b998000ffbe8fdc540b10667d0f15b36ce820a4 only adds the verified CLI readout to
curated docs after the tested implementation. Both commits are local; no push.

### Experiment Card: shared-arrangement-profile-v1

Revision 3, proposed, accepted none. Standing implementation/training authority
applies. Clean baseline 9b998000ffbe8fdc540b10667d0f15b36ce820a4, with inference
code unchanged from verified 1c95f6914fd3fa390d8a46f1d267d5311def7078.
Implementation is complete at 10ddaa8daf7db1b5851d1f8c10733744da368b7e.
Preparation, both fit stages, main native evaluation and Lens review are
complete; the result remains exploratory and is not a model promotion.

Question: can one persistent, interpretable arrangement condition shared by
head, release and row factors improve controllable complete-chart generation,
while an audio-only prior selects that condition without reference leakage?
The intervention is a small supervised mixture over real arrangement profiles.
It is not another timing filter, a label that equates style with quality, or a
claim that source statistics alone exhaust musical intent.

Closest analogue: MuseMorphose computes symbolic-music attributes and uses
them to condition music generation (official repository
https://github.com/YatingMusic/MuseMorphose and author overview
https://slseanwu.github.io/site-musemorphose/). Transfer the computable-attribute
conditioning principle, not its MIDI/bar representation, VAE, beat quantization
or network size. CTRL (https://arxiv.org/abs/1909.05858) supplies the related
persistent-control-code analogue. Ensomi differs in native-ms timing hazards,
LN-state coupling, whole-row legality and full-audio conditioning of every
factor. This is an adaptation of conditional generation, not a novel generic
learning principle.

An unobserved four-state VAE remains an alternative, but it adds posterior
collapse and latent-interpretation questions before testing whether explicit
shared information helps. Independent componentwise regression is rejected
for this test because its mean profile can combine properties never jointly
requested in a source arrangement. More decoder thresholds do not answer the
selected common-condition question. Neither source multimodality nor the
previous crossover proves this intervention will improve playability.

#### Profile representation and probability law

Use the unchanged paired corpus and TRAIN-only normalization from earlier
Cards: manifest
4ad9abfd0ae7798e0a85b96a5dbecdaefd2a81f78bf181438407442b964118e1,
normalization
9cf461a0e825f974f0a80a364123c7afedf1683af76e60fe09b0fbe51c2c8287.
For each separate chart, measure H rows per complete-audio second, note heads
per H row, and LN-head fraction. Transform with log, log, and asin(sqrt(p)),
respectively; standardize using TRAIN-only weighted mean/std. Use the same
uniform-group/then-chart population weighting as the generation corpus. These
are global descriptive conditions, not local density quotas or calibrated
difficulty. Target rows and source release endpoints remain unchanged.

Freeze 16 joint profile representatives from actual TRAIN charts. Initialize
the first medoid at the weighted squared-distance optimum. Add each later
medoid by maximum weight times nearest-medoid squared distance. Assign charts
to nearest medoids, then replace each medoid with its cluster member nearest
the weighted centroid; stop on unchanged medoids or after 50 iterations.
Ties use source SHA order. Require 16 unique occupied clusters. Record raw
profiles, transforms/scales, medoid source SHAs, assignments, masses and
quantization error; no validation chart may set them. Each chart's fixed class
is its nearest representative. This finite support is an approximation to
continuous arrangement choice, not a restriction on the row/timing alphabet.

Add a zero-initialized, bias-free 3->224 profile projection to the encoded
audio condition supplied to all three factors. The same chosen profile is
constant through a chart and every sampled training interval. Full audio
continues to determine local variation; there are no imposed chorus/bar
boundaries. Add a 128->16 linear softmax prior on the valid-token mean of the
existing full-song coarse encoder. Initialize its bias from TRAIN profile
masses. Total new trainable parameters should be 2736 (guard <=5000).

The model defines p(k|A) p(H,R,rows|A,k). Training uses the chart's assigned k;
this is a hard, observed auxiliary assignment, not a per-window latent reset.
The existing importance-weighted event likelihood stays unchanged. Add the
profile cross-entropy once per chart in expectation, divided by
(duration_ms+1)/1000 to match the existing inclusive native-clock objective:
when intervals repeat a chart, average that term
across the interval samples rather than multiplying its population weight.
Prior and generator gradients may both reach the audio encoder. Report the
prior factor separately from conditional timing/row NLL. Neither their sum
nor an oracle-profile validation score establishes native quality.

Audio-only inference samples k once from the full-audio prior, with a separate
CPU RNG seeded by seed xor 0x61F9. Existing head/release/row draws retain their
independent streams. An explicit profile override selects a TRAIN representative
and is reported as a controlled request. Cache the same full-song encoding;
do not decode/encode audio twice. No target chart statistics, source H clocks,
row-content feedback to skeleton, or future actual tails enter native input.
The profile bank, normalization and prior must be checkpoint-contained, and
the source-chart-free streaming entrypoint must support the new family.

#### Matched fit and checks

Both arms start from weights of conditional checkpoint
67b8fc8fbac5f7ca2debe99524e29ac002546f12cb8a3c4c9acf68d68d7f3839 with fresh AdamW
state. Control arm continues the existing model without profile conditioning;
treatment adds the profile projection/prior. Copy every common tensor exactly.
Verify epoch-zero generator probabilities and fixed-seed rows match for every
profile because the projection is zero. Prior sampling must not perturb the
other RNG streams. Test valid full-audio pooling, TRAIN-only preparation,
fixed-chart assignments across intervals, CPU/MPS gradients, crop/native
agreement, separate loss accounting and checkpoint/stream round trips.

Use the same frozen 1200-update interval protocol
34ef9751df4571f516ef2b670b3b0f4b398c212fdd16adcee1a029bd9168eb14,
two songs times two 8-s intervals, seeds 230941/230942/230943, existing learning
rates 3e-5 inherited and 3e-4 other, weight decay .01 and gradient clip 1.
First run 32 updates per arm from the common checkpoint, <=300 s each, checking
finite loss/gradients, actual parameter updates and complete native generation.
Do not reuse preflight model/optimizer states. If these implementation/resource
checks pass, run the matched 1200-update fits, <=2400 s each. Fixed endpoint
selection; no best-NLL checkpoint selection or parameter/temperature sweep.

#### Native evaluation and interpretation

Use the nine fixed native cases and seeds in the preceding main-native result.
Evaluate both fitted arms with no requested profile. Also generate four
controlled profiles per case: the first four medoids in the fixed initialization
order, using their final representatives. No per-song cherry picking. Baseline
outputs ignore requested profiles and are reused for the matched comparison.
This gives 18 automatic and 36 controlled outputs. Profile requests are drawn
from TRAIN representatives, never the corresponding evaluation chart.

Primary metric: mean squared error between requested and generated three-vector
profiles, standardized with the frozen TRAIN transform. Require >=25% reduction
versus the continued baseline evaluated against the same 36 requests, and no
increase in any of the three component mean squared errors. Zero-head output
is a failure, not an omitted or finite-imputed metric. Report per-case values
and realized dimensions, rather than only a cohort mean. Prior calibration on
VAL is a separate diagnostic against the constant TRAIN profile-mass prior;
it cannot substitute for sampled output inspection.

Also require actual within-case control response. Center the four requested
vectors by their mean, and center the four generated vectors separately within
each fixed audio/seed case. For each descriptor, divide their centered MSE by
the requested variance. Require a reduction of at least 25% from the constant-
output error in all three dimensions. A constant output has exactly zero gain
under this test, even if its uncentered error is better than the baseline's.
Report the finite least-squares profile response matrix as a diagnostic, not
an infinitesimal Jacobian or proof of population controllability.

Guards: all outputs complete/reparse within 90 s and 30000 rows; zero strict
HH<20 pairs; cached-Mel 30-row/eight-second readiness <=1 s. On the nine automatic
outputs, preserve nonempty fixed musical scopes and last H at >=.85 of audio
duration, and do not increase aggregate RH<=20 over the matched continued
baseline. Requested low/high density can intentionally change head counts and
LN fractions, so the old composition-distance guard is inappropriate here.
Report every controlled-output RH witness and inspect its surrounding action
and LN episode without treating the HH criterion as an RH annotation.

Lens review must include all five fixed musical contexts for both automatic
arms, each of the four controlled profiles at seed17 for Who/Death Piano/Prom
Queen/Good Luck and seed33 for Airborne, and every remaining close-gap witness.
Reuse unchanged frozen human evidence. Inspect complete action/articulation
and time views. Check that improved aggregate descriptors have not replaced
musical organization with arbitrary extra notes, erased reference-compatible
Tech/Jack organization, or collapsed independent LN timing. Different requested
profiles need not reproduce every source style label: check expressive capacity
across requests and musical coherence at each request, rather than demanding
the same organization from every profile. Missing annotations stay unreviewed;
no listening/player verdict may be invented.

A positive bounded result shows usable shared conditioning with no measured
quality regression, not a completed final model. If descriptors are controlled
but native quality fails, retain the module only as a research direction and
locate the coupling failure before scaling capacity. If likelihood improves
but realized profiles do not, reject NLL as sufficient control evidence and
inspect whether the common condition is used. If the codebook approximation or
prior dominates failure, refine that identified component rather than silently
changing K or using evaluation-chart profiles.

MPS fitting / CPU-one-thread native generation, uv --extra mps, >=2 GiB available
RAM, >=40 GiB free disk. Native cohort budget <=1200 s. Fresh owner
artifacts/joint-audio/20260924-shared-profile-v1; <=1 GiB new outputs excluding
unchanged linked audio/evidence, no overwrite/resume. Packaged configs and
model checkpoints must carry every accepted setting and exact identities.
Freeze a clean intervention source before model runs. Stop on changed bytes,
implementation parity/support failure, nonfinite values, resource bounds or
incomplete native output. Local commits only; no remote publication.

Revision 2 aligns the added prior's normalization with the event objective's
inclusive 0..duration_ms clock. The descriptive H-rate control still uses
decoded duration. The correction was made before any real preparation/fit;
the unprofiled arm and all other protected fields are unchanged.

#### Implementation verification and preparation dispatch

Clean intervention 10ddaa8daf7db1b5851d1f8c10733744da368b7e adds deterministic
TRAIN profile preparation, the 2736-parameter prior/condition, explicit interval
conditioning, a once-per-chart prior factor, matched planned-weight initialization,
checkpoint-contained profile buffers and the streaming override. Default
unprofiled checkpoints retain their distribution. Only the declared model,
data preparation, fitting, inference and owning tests/docs changed.

The selected planned-audio plus package-layout checks passed 54 tests and
22 subtests in 5.99 s. New tests verify TRAIN-only medoids and validation
noninterference, exact zero-condition scores/samples, one full-audio encoding,
gradients through all three generator factors on CPU/MPS, prior padding/gradient
isolation, full-crop/native audio agreement, proper prior interval weighting,
warm initialization and profile checkpoint/stream round trips. A fixture first
named the existing audio_residual module incorrectly; correcting that fixture
allowed the intended gradient checks to run. No model change was made to weaken
those assertions. Prose links and diff checks passed.

Preparation was dispatched via uv run --extra mps python
artifacts/joint-audio/20260924-shared-profile-v1/prepare.py (session 62433).
It freezes the actual 16-profile bank and verifies every zero-condition profile
against the complete Good Luck17 baseline from checkpoint 67b8fc8f. It uses
one CPU thread, <=180 s, no fit, and checks pinned corpus/checkpoint/baseline
bytes. Results and fitting dispatch will be appended when that process finishes.

#### Preparation and preflight result

Preparation session 62433 completed successfully. All 16 zero-projection
profiles reproduce the full Good Luck17 baseline row sequence exactly, and
all common tensors remain equal. The 16 medoids converge in three iterations;
weighted squared-distance error is .5632440390, with unweighted .5/.9/max
quantiles .501764/1.238818/3.127955. Bank SHA
a8973b7f94fde90d9f3cc639eb63e781dd295435622ce79fd54fe244789e6f03;
preparation result 34f75712d4549677d486d4195318619c47651713f1c5c3fe427bd195885ae325.
The first four requested representatives are approximately
(4.749,1.370,.114), (7.032,2.141,0), (6.060,1.429,.734) and
(13.548,1.286,.0004), in H/s, heads/H and LN-head fraction. These are joint
source descriptors, not assigned style names.

Preflight fit session 50765 completed both fresh 32-update arms in 36.697 and
33.676 s. All tracked modules update; profile_condition L2=.057934 and
profile_prior L2=.141844. Checkpoints: base
a119318cfef886535cf122e9491695c03eb767aa83e03284c46f28a7e936700e;
conditioned 236957f2fdbe57ce71fe086e927f089b26b6c1afb176b7cbda8226a91646fcfe.
Fit result 99a8acd30584adbce99a3722bed9bb023eabd23b9f1eced7eb9254f8a691d107.

Preflight native session 59860 completed/reparsed all 18 automatic outputs in
62.81040 s, with zero strict HH<20 pairs. This is an operational pass, not a
quality pass: RH<=20 counts are 11 base and 22 conditioned, and Death Piano19
has an empty fixed musical scope in both arms. Do not call those charts an
improvement. The predeclared preflight gate checks implementation/resources
and completion; it does not choose an endpoint by early quality or NLL.
Native result 1cf943f3fb9dbed88ad470255bdf6d0aa9a0c19d00dcd6257e467ec9e629c91c.

#### Matched main fits and evaluation refinement

Main fit session 96094 is now terminal-success. It completed 1200 updates per
arm, with identical exposure counts and fresh optimizer state from the same
initial checkpoint. Base took 964.39089 s; conditioned took 920.73794 s. The
driver total was 1888.74076 s. Main freeze
7b515af00dbfaeba3f205f7b67c4ae84d8a98eac681036ff92ec1e7e60b8a074;
fit result 2580f089b24fb5188e31d986a41082689fa5a9f80ff3ace8be05f2b61d916602.
Final checkpoints: base
5f26b7b15d97fa2d6964cf77a4cadea016f5ace9dfd578ce2fae5dc7c8a0e121;
conditioned abc27f1d192869419e42729a6b9fcdfd1c507fd672e12c9a9c7c868a5082a2ef.

Population conditional NLL is 39.776696 base and 39.684357 conditioned; the
conditioned prior adds .017847 nats per integer-clock second, giving joint
39.702203. This is reference-profile evaluation, not marginal audio-only NLL
or a native quality verdict. Profile-condition update L2=.787393 and prior
L2=1.002683. All tracked inherited/skeleton modules also update.

Revision 3 strengthens the evaluation before any main native output was
generated or examined. Uncentered descriptor error alone can improve when a
constant-output model merely moves closer to the request mean. The added
within-case centered criterion rejects that case; a synthetic constant-output
check gives zero gain, and exact request tracking gives unit gain. The fits
were dispatched/frozen under revision 2. Training, models, cohort, requests,
resources and original guards are unchanged, so no refit is needed. Main
native_v3.py records revision 3; preserve native.py as the preflight script.
Assessment is in assess.py, and inspect_lens.py fixes all required contexts and
close-gap witnesses. The main generation/readout was pending at this revision-3 dispatch. The
completed result is appended below; no lifecycle transition has occurred.


### Result Log: shared-arrangement-profile-v1-main

Owning Note: 2026-09-23-audio-skeleton-r1-integration; accepted revision none.
Fits used Card revision2; native generation and assessment used revision3,
whose centered-response refinement preceded main native generation. Product
source stayed clean at 10ddaa8daf7db1b5851d1f8c10733744da368b7e throughout.
Curated findings are now in docs/research/shared_arrangement_profiles.md at
behavior-neutral documentation descendant d80de496293d344c4c803d30f05b0e4d957ba863.
The earlier preparation/fitting identities, environment, commands, bounds and
checkpoint pins remain unchanged.

Native command: uv run --extra mps python
artifacts/joint-audio/20260924-shared-profile-v1/native_v3.py main.
Session10194 completed all54 charts in117.824188s, with independent reparse and
unchanged weights. Eighteen automatic plus36 controlled outputs cover the nine
fixed audio/seed cases and first four TRAIN profiles. One CPU thread, no fitting,
correction off, per-chart90s/30000rows and total1200s bounds all satisfied.
Freeze639bd6213e450ef0da1d3d267c57b63700053762307536995323f5fe12d1671d;
native result de5544efff6ffdaeca0c0388636e60dcfe67dc2691d20466c10704be4f8abea4;
assessment fc434c7e9703b4bee3eeb35955fab4b10d627a80d308efbe11f06674e8299e97.
Artifacts remain under the same shared-profile owner; no overwrite/resume.

Mean squared standardized descriptor distance is13.25918683 base versus
7.58094113 conditioned, a42.82499197% reduction. Each component improves:
base(5.11699424,4.46312323,3.67906937), conditioned
(2.26431361,2.50042180,2.81620572). Centered gains are
(.31963112,.34375027,.26315300), all exceeding the declared.25 threshold.
The finite request-to-realization response matrix, with requested dimensions
as rows and realized dimensions as columns, is
(.36294153,.49060158,.05040684);
(.19960803,.46795938,.03961157);
(.07270315,-.06873112,.17674964).
This finite cohort has no population confidence claim. The LN-heavy request's
.734 target realizes only 0.03634..0.28228 LN-head fraction. VAL36 prior CE2.534448
versus constant TRAIN-mass CE2.588476 is a separate, modest diagnostic gain.

All54 have zero strict HH<20 pairs and pass completion/readiness checks.
All18 automatic fixed scopes are nonempty and last H exceeds.85 audio duration.
Profile0 DeathPiano19 has an empty fixed scope; this is outside the declared
automatic nonempty-scope guard and was not a reviewed musical success.
The automatic RH<=20 guard fails: base0 versus conditioned3. There are also
two controlled-output RH witnesses. No NLL or average-control improvement
cancels that failure.

Lens session47063 rendered118 pages in43scopes. Every time page was actually
viewed, and complete action/articulation tables were read across pagination:
957 physical rows and934 H rows in the review scopes. Parent reference evidence
remains unchanged. Manifest6899dff2ee9b00a7fbe9cb44fc3b4dd8400630cf368ecae4ab74862c597af195;
final review0a7fdd8562fc8885a034a998eecc1ba75ccd16802bd612a956c7747bbe703e1e.
The review explicitly retains uncertainty on acoustic fit, fine Tech and dump,
and records no listening/player verdict or new human annotation.

Observed organization: broad chord flow appears in GoodLuck/PromQueen under
profiles1/3, while Who remains largely single flow under the same conditions.
Independent LN tails persist. Low LN-head fraction can coexist with substantial
held-lane time: DeathPiano17's automatic scope has one new H but holds lasting
4389/6673/6968ms. Head fraction alone does not measure sustained occupancy.
Five close RH witnesses separate three dependency families. GoodLuck automatic
has full occupancy at197436 and a first release197518,18ms before nextH197536.
PromQueen automatic chooses TAP0 at84351 after R84345 although lane1 is rested.
Airborne automatic TAP0/3+CLOSE1/2 at133516 leaves no safe lane for H133519 when
both strict HH and RH diagnostics are considered. Profile3 PromQueen similarly
has TAP1/3+CLOSE0/2 at39201 before H39204. Profile1 GoodLuck closes three holds
at184037; only lane0 is rested at184055, so the chosen double requires RH18,
although a single-head alternative exists. These cannot all be attributed to
full-occupancy release scheduling.

The read-only corpus audit checks all651 paired charts,151003 same-column
release-to-next-head pairs: none<=20ms, minimum30ms. Source SHA/objects were
verified; source parser and admission retain literal starts/ends, with no
minimum-gap retiming. Corpus/admission selection remains a confounder. Audit
c27436b57c77ed4d06552794e7b969ba059e91141c780e77aa4441e3f1f0ff21.
Do not convert this observation into a universal RH threshold; only strict
same-column successive attacks<20ms have the owner's explicit bad-pattern rule.

Evaluation: partial controllability is observed, with weak calibration and
substantial cross-talk. Native failures remain. A shared offset may directly
change downstream preferences, or may change H plans that redirect row history;
these explanations are not separated by the current diagonal samples. Profile
quantization, limited training and sampled autoregressive divergence also remain.
Recommendation REFINE, no acceptance/adoption. All recorded jobs are terminal;
do not resume their fits or cohorts. The next bounded diagnostic separates
these information paths before altering model size or training budgets.

### Experiment Card: shared-profile-path-crossover-v1

Revision1, proposed, accepted none. Standing local implementation/execution
permission applies; no Note lifecycle change or remote publication. Owning Note
2026-09-23-audio-skeleton-r1-integration. This diagnostic supersedes no earlier
result and changes no checkpoint or training configuration.

Question: does the observed requested-H-rate to realized-chord-width response
primarily travel through the generated H plan, or through the profile condition
read directly by release/row factors? The analogous local experiment is the
fixed generated-plan crossover in docs/research/head_plan_row_response.md.
Its mechanism transfers exactly: separate a generated mediator from the
conditional decoder while preserving full audio and event RNGs. Here weights
are identical and only the two uses of one arrangement condition differ.
This is component intervention/attribution, not a new model or an estimate of
population causal mediation. Generic encoder scaling and a new RH filter do
not distinguish these alternatives, so defer them for this probe.

Clean baseline source d80de496293d344c4c803d30f05b0e4d957ba863 has model code
identical to10ddaa8daf7db1b5851d1f8c10733744da368b7e. Use conditioned checkpoint
abc27f1d192869419e42729a6b9fcdfd1c507fd672e12c9a9c7c868a5082a2ef, bank
a8973b7f94fde90d9f3cc639eb63e781dd295435622ce79fd54fe244789e6f03, and frozen
native result de5544efff6ffdaeca0c0388636e60dcfe67dc2691d20466c10704be4f8abea4.
Use all nine existing cases, seeds17/19 for the four musical cases and33 for
Airborne, with the same full audio/Mel identities. Baseline finite response
coefficient H-request to width=.4906015821, averaged over nine fixed cases;
this is not a population estimate. The prior is bypassed by explicit profiles.

Define F(i,j) as the three measured normalized descriptors after rendering the
previously generated H plan for profile i with downstream profile j. The single
intervention is replacing the H planner with a fixed plan of generated times;
release timing and row decisions still run natively from BOS with full audio,
the selected downstream code, LN-only skeleton feedback and exact replay.
No future reference rows, tails or LN state enter generation. Keep original
release/row RNG streams, temperature and correction-off policy. No weight update.
Only an artifact-owned diagnostic script changes; product behavior is untouched.

For each case use fixed reference profile0 and j=1,2,3. Reuse the three frozen
F(j,j) diagonal results. Generate F(0,0) once to verify complete-row parity, then
F(j,0) and F(0,j) for each j:63 total diagnostic outputs,54 new crossed samples.
Verify all36 frozen row-file hashes before extracting H plans; verify measured
values independently from those rows. An eager fixed-plan adapter must expose
exactly the same next-L preview and completion semantics as the native planner.
Nine F(0,0) controls must reproduce saved rows exactly. Stop before attribution
if any diagonal, plan identity, complete reparse or weight fingerprint fails.

For each j, total change T=F(j,j)-F(0,0). Define the plan contribution
P=.5*((F(j,0)-F(0,0))+(F(j,j)-F(0,j))) and downstream contribution
C=.5*((F(0,j)-F(0,0))+(F(j,j)-F(j,0))). Thus P+C=T exactly; each averages the
two possible intervention orders, sharing their interaction equally. Also
report the unallocated interaction F(j,j)-F(j,0)-F(0,j)+F(0,0), since strong
interaction weakens a single-path interpretation. These are finite sample
contrasts, not unique intrinsic module responsibility fractions.

Solve the full-rank3-by3 matrix of profile-code differences relative to0 for
T, P and C response matrices separately per case; average the nine matrices.
Primary diagnostic is the P/C split of the fixed positive total H-request to
width coefficient. Prioritize one path for architecture refinement only if it
accounts for at least two-thirds of the mean total and has that total's positive
sign in at least6/9 cases. If neither qualifies, both qualify through unusual
cancellation, or interaction makes attribution unstable, retain a coupled
explanation. Negative components must be reported as cancellation, not clipped
into misleading percentages. Report raw/profile-level effects and per-case
values. LN-fraction response decomposition is secondary; no quality promotion
can follow from the attribution metric.

Guards: complete/reparse, exact supplied H times, unchanged weights, finite
metrics and diagonal0 parity all required. Record strict HH<20 and RH<=20 in
every crossed output, including exact witnesses; these are outcome diagnostics,
not stop-on-first-failure selection. Reject a proposed deployment-quality claim
if close pairs remain. This uses precomputed H times, so measured runtime is
not end-to-end startup evidence. Musical check: inspect fixed Who/GoodLuck/Airborne
contexts for both directions of the profile0/profile3 cross (six scopes), plus
every close-gap witness, with complete Lens time/actions/articulation. Reuse
unchanged frozen human references; do not assign new human labels.

Command: uv run --extra mps python
artifacts/joint-audio/20260925-profile-path-crossover-v1/run.py.
One CPU thread, AppleM5/24GiB, Torch2.11/Python3.10.20; no network, fit or extra
accelerator process. Bounds: total600s, per chart90s/30000rows, >=2GiB available
RAM, >=40GiB disk, <=300MiB new outputs excluding linked original audio/Lens
references. Fresh owner at that command's directory; no overwrite/resume.
Freeze clean source, script/Note OIDs, input bytes and ordered arm list before
model execution. Stop on changed inputs, nonfinite metrics, replay/parity/plan
failure or resource bounds; keep partial evidence and mark attribution incomplete.

Main confounders: off-diagonal conditions intentionally disagree across modules
and may be off the training distribution; shared RNG seeds couple samples but
do not hold histories fixed; one seed pair and five audios limit generality;
profile0 and equal interaction allocation are attribution conventions. A
positive result chooses a smaller next architectural question, not a module
blame percentage. Negative/mixed findings reject the sufficient single-path
account; they do not show that the shared-condition architecture is useless.


### Result Log: shared-profile-path-crossover-v1

Exploratory Card revision 1 at Note commit
45768e31d757cc0157299c8893ce44e6080d8754; accepted revision none. Execution used
clean source d80de496293d344c4c803d30f05b0e4d957ba863 and the pinned conditioned
checkpoint/bank/native records, unchanged. Command exactly as specified in the
Card, session 77055, completed in 97.146975 s on one CPU thread. There was no fit,
checkpoint selection, overwrite or resume. Native outputs occupied 10939684
new bytes before Lens evidence. Source row files and model tensors remained
unchanged. All 63 outputs completed/reparsed, preserved supplied H times and
finite metrics; all nine profile0 diagonals reproduced full saved rows exactly.

Owner: artifacts/joint-audio/20260925-profile-path-crossover-v1.
Freeze e71c635982e8383af6ce0ce70ed626407676704e0f90b32f13bc26cb324065df;
run script a188d459323c2fb6f76682703c29fd8b8cff21b8f03a36aefbfa9b7b6506668e;
result 95964d01b5e12ed8c5a811151f2d22c99dfd35592e5eb201d55f6d78e6f85175.
No protected comparison field changed. Rendering used inspect_lens.py under
the same owner, script40bf12bf26286e0a1979406c030cb45cbbf23905645eaed7773168b342ec1615,
session27949, terminal-success. All source/reference bundle bytes were preserved.

The total finite response matrix reproduces the earlier diagonal calculation
to numerical tolerance. The request-difference matrix condition number is1.699.
For H-rate-request to realized width, mean total=.49060158, plan=.11076664,
downstream=.37983495. Downstream's two-order average contribution is77.4223%,
positive in9/9 cases; it meets the declared two-thirds and6/9 selection rule.
Plan contribution is22.5777%, positive in7/9 cases. Negative plan contributions
in GoodLuck19 and Airborne remain explicit rather than clipped into percentages.
The interaction coefficient for this readout is.10784596; individual orders
give downstream shares66.4311% and88.4135%. The average passes the fixed rule,
but its exact fraction is not order-invariant. This prioritizes examination of
downstream conditioning, not removal of the H planner or a general R1 blame share.

Width-request to width decomposes .46795938 total into .04401093 plan and
.42394845 downstream. LN-request to LN-fraction decomposes .17674964 into
-.04022233 plan and .21697197 downstream: plan changes partly cancel the
positive downstream response. PromQueen19 illustrates a large trajectory
interaction: H0/D2 realizes .57258 LN fraction versus .03634 on H2/D2. The limited
seeds, changed autoregressive histories and intentionally inconsistent component
conditions prevent a population inference from that case. Order-specific
matrices and per-case contrasts are retained in order-analysis.json/result.json.

Crossed outputs contain three strict HH pairs (2,4,11ms) and nine RH<=20 pairs.
They are outcome diagnostics, so all declared arms were retained; no first-bad-
sample stopping or favorable-output selection occurred. The nine identity
controls remain unchanged and have no such pairs. Crossed conditions do not
form a replacement playable model.

Lens review is complete for all15scopes:38 time-proportional pages, complete
paginated actions/articulation,356 physical rows and325 H rows. The six fixed
Who/GoodLuck/Airborne contexts and all12 gap witnesses are covered. Manifest
151dc270ba9309726bf1f5f25ed8f1f4d91d1947cc62a0185a62ad53a43c1f65;
review bea426a8ceb332793eccf4ec4787e203aea92355eb3557443d11412772687508.
No listening/player judgment or human annotation was added. Sparse-plan outputs
retain their large gaps; H3/D0 Who retains a long117-ms single-note passage.
Airborne H0/D3 carries chordal/independent LN material while H3/D0 is mainly
single taps. Aggregate descriptor changes do not exhaust those musical roles.

Strict HH witnesses:
- Airborne H0/D1 commits TAP0/1/2/3 at144014, then must realize H144025.
  The later11-ms repeat is unavoidable after that four-head commitment.
- PromQueen17 H0/D1 keeps LN1/2/3 active at105943,105945,105949. The only free
  column0 receives TAP,TAP,LN, creating2/4-ms successive attacks. LN1 closes
  only106041; LN2/3 spans4137/5170ms. Earlier release/row choices must reserve
  available columns for the known burst.

RH witnesses also separate earlier allocation from later choice. Airborne H0/D1
at30455 uses TAP2+CLOSE0/1/3 before H30469, leaving no column avoiding both
short-HH/RH; H0/D2 instead uses LN1/2+CLOSE0/3 and reheads3 after14ms. PromQueen19
at43587 uses LN1/3+CLOSE0/2 before H43598, again leaving only freshly released
choices. Other cases have rested alternatives: PromQueen19 RH20 at96056;
Airborne H0/D2 RH18 at74648 and RH4/RH9 at148209/150326; GoodLuck H3/D0 RH4 at116791.
Exact20 RH is not the strict HH criterion. Long holds and independent releases
remain present; their existence does not neutralize the failures.

Evaluation: the selected average attribution supports a bounded downstream-
coupling hypothesis. It does not establish that removing H-rate input from rows
would improve musical quality, or that the observed linear response is a pure
infinitesimal control derivative. The candidate-consequence traces identify a
separate availability gap: frontier2 exposes earliest possible R=now+1, while
the composed model may retain holds across several required heads. The exact
row support enforces physical legality, not the owner's strict HH criterion.
Training uses source replay/head previews and factor NLL, without evaluating
sampled continuations caused by all alternative current rows. Joint gradients
alone therefore provide no invariant protecting future playable lane capacity.
Curated analysis is committed at962d732e7ef3364d9f237fe4227576ec5f1ff6b7.

Recommendation REFINE. Keep these questions separate in the next Design:
(1) whether a factor-specific representation of arrangement demands reduces
unwanted direct response while preserving audio and actual H-preview information;
(2) how candidate evaluation represents future lane availability under the actual
release/row process. The playability priority is(2). A learned release response
may use full audio, own skeleton history and the hypothetical candidate LN
projection; no actual future tail or tap-layout feedback may enter the skeleton.
A raw conditional-release event-mass scale is not an identified pressure signal;
use normalized observable event probabilities if pursuing that branch.

Purely optimistic release timing cannot certify realized future availability.
A conservative certificate must also state its cost in expressive support:
assuming no future release-only event would exclude legitimate full-LN chords.
Do not silently adopt that assumption, a minimum LN duration, a generic onset
spacing filter or an RH rule inferred only from this admitted corpus. A bounded
joint continuation/forecast interface is a live alternative, with speculative
work confined to unpublished future rows. This is direction-setting analysis,
not a new Experiment Card or an implemented decoder change.

All native/fitting/render jobs are terminal. No remote push, Note acceptance,
lifecycle transition or final-model adoption occurred. The ultimate playable
system goal remains active; these two completed diagnostic stages should not
be rerun without a new discriminating question.


### Experiment Card: unpublished-continuation-screen-v1

Revision 1, proposed, accepted none. Standing local implementation and execution
permission applies. Owner: 2026-09-23-audio-skeleton-r1-integration. Clean baseline
962d732e7ef3364d9f237fe4227576ec5f1ff6b7, conditioned checkpoint
abc27f1d192869419e42729a6b9fcdfd1c507fd672e12c9a9c7c868a5082a2ef.

Question: can a bounded unpublished continuation, sampled jointly by the actual
release and row factors, avoid the reproduced short-pair failures without
changing H times or sacrificing LN/chord organization and realtime delivery?
The selected mechanism is finite-window conditional resampling, not a new tail
predictor. It uses the composed model's own future releases and rows, avoiding
the assumption that every physically possible early release will occur.

Closest analogues: NeuroLogic A*esque decoding uses estimated future constraint
satisfaction in autoregressive search (https://arxiv.org/abs/2112.08726). The
Alive Particle Filter handles indicator potentials with adaptive sampling cost
(https://arxiv.org/abs/1304.0151). Twisted SMC learns future-potential predictions
(https://arxiv.org/abs/2404.17546), which suggests a later amortized response if
sampling cost proves material. This first Ensomi test is a bounded single-window
rejection policy, not an implementation of those particle estimators, a novel
SMC theorem, or exact sampling from the globally constrained whole-chart law.
The difference is a four-lane event process with irreversible published rows,
mutable unpublished continuations, exact LN replay and separate H/R histories.

Alternative branches: a new marginal LN-tail predictor could be supervised from
beatmaps but may disagree with the actual release/row process; a conservative
no-future-release certificate could suppress legitimate full-LN chords. Defer
both until this test measures the benefit and cost of actual joint continuation.
A one-row optimistic correction already removed one HH witness while adding
an RH regression; it is not the intervention here.

Intervention: factor the native sampler into an in-memory branchable session
with one shared full-audio encoding and immutable model weights. Baseline
rollout remains the same single trajectory. An optional research buffered
rollout starts from BOS, proposes through an 8000-ms publication window and
screens a 20-ms future halo. Window cuts occur at completed scheduler steps,
so a step may exceed the nominal cut; only rows through cut+20ms enter the
halo check. Rows already published are immutable. LN heads can be published
without tails; only the known audio terminal forces closure.

A session fork owns independent CPU RNGs, mutable queues/lists and counters;
replay and neural caches may share immutable values. Retry restores the last
published session and changes only release/row RNGs, not head/profile RNGs.
When the prefix ends after observed empty time, a retry draws a fresh release
survival threshold conditional on that observed coverage. Keep original RNG
streams for the first proposal. A healthy first proposal must retain all rows
exactly; extra halo evaluation may not advance the accepted boundary's RNG.
No full row content enters the skeleton beyond the existing LN projection.

Screening potential: zero strict same-column HH<20 and, for this explicitly
bounded research policy, zero same-column release-to-next-head RH<=20. TAP and
LN heads count; exactly20 does not enter HH. The RH screen is a corpus-calibrated
experimental preference, not a universal human BAD annotation. All651 admitted
paired source charts satisfy both screens, while short LNs, close cross-column
H times and full-LN chords remain representable. Do not add a minimum duration,
onset gap, tap-layout feedback to skeleton or a timing-grid restriction.

Use at most four proposals per window, including the unchanged first proposal.
Accept the first screened proposal and publish only its window prefix; retain
its boundary state, so next-window first sampling reproduces the accepted halo.
If the attempt/time/row budget is exhausted, return the last published prefix,
with open holds and an explicit incomplete reason; do not publish the failing
speculation, fabricate closure or silently rerun the entire song. This is a
failure outcome to measure, not a production fallback. Incremental callbacks
must never expose rejected rows or claim speculative coverage as settled.

Baseline identities: shared-profile native result
 de5544efff6ffdaeca0c0388636e60dcfe67dc2691d20466c10704be4f8abea4 and path-crossover
result95964d01b5e12ed8c5a811151f2d22c99dfd35592e5eb201d55f6d78e6f85175.
The cohort has15 distinct case/conditioning pairs: the nine shared-profile
conditioned automatic cases; clean Who17 profile0; and five crossed cases
PromQueen17 H0/D1, PromQueen19 H0/D2, GoodLuck17 H3/D0, Airborne33 H0/D1 and
Airborne33 H0/D2. Baseline totals are three strict HH and twelve RH<=20 pairs.
All original full audios, seeds17/19/33, profiles and generated H plans are
pinned. Explicit fixed-H support is research instrumentation only and supplies
no future row/tail/state. Native automatic cases retain lazy H planning.

First verify the branchable engine reproduces saved rows for clean Who17,
PromQueen17 H0/D1 and Airborne33 H0/D1. Then screen those same three. If exact
baseline/clone parity, complete output and finite bounded resources fail, stop
before the other twelve cases and diagnose; do not silently increase attempts.
If that gate passes, complete the remaining paired baseline/screened cases.
No fitting, parameter sweep, NLL endpoint selection or temperature change.

Primary decision: all15 complete/reparse with zero strict HH and zero screened
RH, versus the baseline3/12, while preserving every H timestamp exactly.
Guards: first30-row/eight-second readiness<=1.5s from cached-Mel entry; each
accepted publication-window service<=2s; cohort total sampler wall time<=2x
matched baseline; per-case head counts within0.8..1.25 of baseline and absolute
LN-head-fraction drift<=.10. Report exact per-case values, attempts, rejected
work, max unpublished rows, stop reasons and coverage, not just totals.
The composition guards detect changes in the requested arrangement, not a
universal musical-quality score. A gate failure still yields an exploratory
REFINE result and cannot justify model promotion.

Lens: inspect the five fixed musical contexts across automatic cases, the five
crossed cases and clean Who17, plus accepted/rejected scopes for every resampled
window and any residual close pair. Read full actions, articulation and time
views, including entering holds; reuse unchanged frozen human references.
Different arrangements need not preserve a reference's tags, but check that
screening has not erased chords, independent LN releases or fine-time support.
No inferred human labels, listening verdict or automatic BAD taxonomy.

Implementation files: planned_audio_continuation/generation.py plus a small
buffered sampler module, focused tests and owning research docs. Leave packaged
CLI/config default behavior unchanged until the research policy passes; no Hydra
knob is added in this probe. Tests cover fork ownership/RNG restoration, halo
boundaries, exact20 HH versus RH, retry selection, budget exhaustion preserving
open holds, callback publication, one audio encoding and baseline distribution.

Freeze clean implementation source and script/Note/input identities before any
real-model run. Command: uv run --extra mps python
artifacts/joint-audio/20260925-unpublished-continuation-v1/run.py. One CPU thread,
AppleM5/24GiB, Torch2.11/Python3.10.20; no competing fit/network job. Per-chart90s
and30000rows, cohort900s, >=2GiB RAM, >=40GiB disk; <=500MiB new artifacts excluding
linked audio/reference evidence. Fresh output owner, no overwrite/resume. Stop
on changed bytes, nonfinite scores, parity/plan/physical-support violations or
resource limits. Local code and Note commits only; no remote publication.

Interpretation: a pass shows that a bounded scheduler can use actual joint
continuations to remove these screened failures at measured runtime cost. It
does not establish musical correctness or the global constrained posterior.
A quality/composition failure means conditional resampling changes too much;
a cost/attempt failure motivates a learned future-potential proposal or a
better joint event representation. Keeping H fixed can also expose infeasible
plans; do not conceal that case by deleting H events. Any subsequent model
learning or alternative policy requires its own discriminating Card.


#### Branchable sampler implementation and verification

Clean implementation452abc689055e2660bda5f62989cf70c95b0db8b extracts the native
step into ContinuationSession and adds optional buffered sampling. The CLI
continues to use single-trajectory rollout. Full-audio encoding is shared by
forks; separate RNGs and mutable queues preserve branch ownership. The extracted
step retains the original AST semantics after replacing state locals with
session fields. All original planned-model tests pass. Selected full check:
uv run --extra mps --group dev pytest -q tests/research/planned_audio_continuation
tests/test_package_layout.py:60 passed and22subtests in6.16s. After strengthening
the fork fixture to require actual continuation, the six new tests passed in1.06s.
They cover actual cache/RNG ownership, unchanged healthy rows, one encoding,
rejected halo isolation, open-LN preservation on exhaustion, strict20-ms boundary
semantics, interrupted-work accounting and callback exception propagation.
Formatting retained identical AST; scoped Markdown links/math and diff checks pass.

The implementation exposes an optional rejection observer for offline Lens
inspection. It retains an independent in-memory rejected fork; after sampler
measurement, the diagnostic driver can complete that fork with the same model
to resolve its generated LN objects for inspection. This evidence work remains
inside the total experiment resource bound, but is not charged as online sampler
service. It supplies no future source endpoints and cannot change published rows.

Distribution clarification: the ideal rejection law with independent base
proposals is only a reference. The first proposal of a later window reuses
randomness whose halo was screened previously, whereas retries use fresh R/row
randomness. This policy therefore depends on prior planning work as well as
published rows; it is not claimed to sample either the ideal local conditional
or the global constrained law exactly. No implementation, protected comparison
field or Card revision changes follow from making that approximation explicit.
No actual-model cohort has run yet; the next action is the pinned three-case gate.


### Result Log: unpublished-continuation-screen-v1-native

Card revision1, proposed, accepted none; execution Note pin
7b02e8d503c761578146315d3d09a3042bba4030. The clean implementation stayed at
452abc689055e2660bda5f62989cf70c95b0db8b throughout the real-model run. The command
specified in the Card ran as session65330 and is terminal-success. All15 paired
cases completed; the first three passed the predeclared gate before the remaining
twelve ran. All unbuffered trajectories reproduce saved row hashes exactly.
All buffered trajectories preserve every H time and independently reparse;
client callbacks replay to the same rows and LN state. Weight fingerprints and
all pinned source-row files remain unchanged.

Owner artifacts/joint-audio/20260925-unpublished-continuation-v1. Script
2749edf48984a18b473e949ef3bd9e0fcc95293a49833f13deb67d070ece8ba3;
freeze38a41df0f086ff2f779ce7e9ad8c3498567b11da40e7bd606fb39415a5343636;
gate f512a0cf2fcc3996426567b85eb6154195fd051cfa0eeb3ed20f4942eb9c2a20;
result3fff511d78ef4d0fd3400f33853d2e79219a69dbd034e67bee9d1263b5cb2ca9.
No overwrite, resume, parameter update or Card-field deviation occurred.
Total driver time95.514909s includes12 offline rejected-future completions.
Native artifacts before Lens occupy12957473 new bytes; linked audio is excluded.

Primary screened counts improve from HH3/RH12 to0/0 across all15complete outputs.
There are337accepted windows and349proposals, including12rejections in11windows;
maximum proposals in a window is3 under the cap4. Seven cases require no retry
and remain row-identical. Sampler time is34.689581s baseline versus35.952550s
buffered, ratio1.0364077. Maximum cached-Mel 30-row/eight-second readiness is
.3226423s; maximum validation-window service is.5366829s. Maximum unpublished
proposal size is136rows. These exclude waveform/Mel preprocessing and offline
evidence completion, and do not establish an OS/client contention guarantee.
All runtime, completion, H-preservation, note-count and physical guards pass.
Head-count ratios range.940092..1.068694.

Two LN-fraction guards fail, so the whole comparison is not a clean promotion.
PromQueen19 H0/D2 changes from497/868 LN heads (.572581) to326/816 (.399510),
delta-.173071. Held lane-time fraction falls.377863 to.290358. Airborne33 H0/D2
changes from1256/2871 (.437478) to829/2744 (.302114), delta-.135365; held lane-time
falls.226744 to.158898. The declared absolute bound is.10. Both still have many
LNs; do not turn this failure into an absence label or silently weaken the guard.

Read-only localization shows that the changes are not confined to rejected
windows. PromQueen's unchanged prefix has152LN starts in both outputs; inside
the two resampled windows57becomes64, while the remaining suffix outside those
windows changes288to110. Airborne's unchanged prefix has15in both; its resampled
window changes46to4 and the other suffix1195to810. Artifact
ln-drift-localization.json SHA384a53f1ba5e25e3fee54f79402374ca847f019593b0bd0ad36e20df411c3348.
This does not yet separate history-state effects from the continued RNG stream,
prove a population tap bias, or justify another seed/architecture sweep.

Lens rendering session46719 is terminal-success. The frozen plan has52scopes:
109new time pages and21reused scopes. Reuse requires exact combined action,
articulation, scope and entering-hold semantics, excluding only the source hash.
Twenty scopes reuse prior completed reviews; one reuses current scope10.
Parent human/reference files remain byte-identical. Plan SHA
2362673d720f7c22ccf29aa5db579d40f510226db3e0a3dd5a169ec10777dc7e;
render result2185790bafe7e07248ecc05fca7ad8da966eab80b7b0ae453fe3665339c00280.
Rendering is not a completed quality review.

Current actual inspection: newly rendered scopes10,11,12,13,30,35,37,40,
20pages total, with all actions/articulation read across pagination. Scope29
is now covered by its proven identity to scope10. Prior-reuse scopes retain
their pinned already-read evidence. The remaining23new scopes/89pages are
unreviewed. Exact progress/observations are in review-progress.json, currently
SHAeb50cfc086924e3647cd1fb351ad00496c3accd19db815cbc06be6b8c0357f2d.
Do not create a final review or claim all52scopes were inspected yet.

The inspected PromQueen17 repair is meaningful: both LN2/3 heads at104512 were
already published before the rejected window begins105324. Accepted R105429
closes those holds, and a new LN2 at105735 spans the three H times
105943/105945/105949. They use lanes1/0/3, preserving the2+4-ms cross-column
skeleton without the same-column repeats. The rejected future instead keeps
three columns occupied and emits TAP0/TAP0/LN0. Later accepted rows retain
staggered releases, overlapping LNs and repeated double grips. No prior tail
promise is rewritten, consistent with the client protocol's incremental holds.

In the two composition-regression fixed scopes, substantial LN organization
persists. Buffered PromQueen19 has a1243ms hold spanning4H, followed by
417/833/634ms overlapping holds with independent tails. The baseline has three
long entering holds and repeated lane0 taps. Buffered Airborne has749/819ms
holds spanning3H each with shorter released groups around them; the baseline
has more frequent LN starts and greater occupancy. No listening/player verdict
has been made, and the remaining scopes still matter for whole-chart assessment.
Curated bounded findings are at0de03c1fe71716d88affc4445fea9113f3a2ba93 in
docs/research/unpublished_continuation_screen.md.

Evaluation remains partial until Lens review finishes. Provisional recommendation
REFINE: finite joint planning removes the tested screened failures at low measured
cost, while two arrangement guards fail and broad musical quality is unproven.
The next immediate action is to finish the already-rendered Lens scopes, without
rerunning generation, fits or rendering. Then discriminate whether suffix drift
is principally carried by changed learned state or by altered randomness before
selecting a new intervention. A learned future-potential response and stable
persistent arrangement control remain live hypotheses, not implemented fixes.
No new Card, Note acceptance, remote push or final-model adoption has occurred.
All jobs are terminal; the ultimate playable-system goal remains active.


### Result Log: unpublished-continuation-screen-v1 completed scope review

Card unpublished-continuation-screen-v1 revision1 remains proposed, accepted
none. No new generation, training or rendering was needed to finish the declared
inspection. Product source65ef95c092e7d3024d99ed60e37c2c22ee041df9 publishes the
completed structural findings in docs/research/unpublished_continuation_screen.md.
The native comparison remains pinned to452abc689055e2660bda5f62989cf70c95b0db8b
and result3fff511d78ef4d0fd3400f33853d2e79219a69dbd034e67bee9d1263b5cb2ca9.

All31 newly rendered scopes/109time pages and their complete native-ms action
and articulation tables have now been read. Twenty scopes reuse prior completed
reviews by exact semantic hash; scope29 reuses current scope10. All52 declared
scopes are covered, but this is not whole-song visual inspection. Final artifact
review.json SHA7d41b477474a0ee0c0a95a367675fe51d33b743794be9568700ec675332c7267;
review-progress.json SHA91849c6a6a4b57b825741410d812367c208fc9c549900bf3b8ba82dbb2204c65.
Lens manifestee0e3a32a527aefaf2b0fe84bd68046f9af8f771f257068364f096b4cdb2919e
retains parent evidence. All12 rejected proposals are inspected, containing15
pair witnesses including newly arising RH conflicts after earlier resampling.
The original Airborne HH11 is avoided in a changed suffix, not by directly
rejecting that particular local episode.

Accepted scopes preserve close cross-column H bursts, broad chords, overlapping
LNs and independent subset releases. Airborne H0/D2's resampled LN relay becomes
mostly single/double taps (46to4LN starts); Prom19's edited windows remain
LN-rich (57to64), with most count loss later. Thus physical-screen success and
expression preservation remain separate. Neither aggregate LN fraction nor
presence alone describes the changed free-column/held-column organization.

One supplementary read of Airborne H0/D2 at81800..83900 adds one time page and
complete29physical-row/26H tables. LN0 at82889..82890 lasts1ms, far before
terminal271490 and without full occupation. It passes both screens and sits
within a short LN relay transitioning to taps/doubles. Its musical/player meaning
is unresolved. No minimum-duration rule or human BAD label is inferred.

The human-confirmed HH rule remains strict same-column successive attacks<20ms,
including TAP and LN heads. Exactly20, cross-column H gaps, LN durations and RH
pairs are separate. RH<=20 remains an experimental preference. No listening or
player test occurred. Evaluation recommendation REFINE: physical failures are
removed on the15 cases at low measured cost, but two composition guards fail.
The next discriminating question is the contribution of generated state versus
R/row randomness to suffix composition; no architecture change follows merely
from the count regression. This completes the previously partial review.


### Experiment Card: continuation-state-rng-v1

Revision1, proposed, accepted none. Owning Note2026-09-23-audio-skeleton-r1-integration.
Standing local implementation and execution authority applies. Clean baseline
65ef95c092e7d3024d99ed60e37c2c22ee041df9, a docs-only descendant of the screened
sampler implementation452abc689055e2660bda5f62989cf70c95b0db8b. Checkpoint remains
abc27f1d192869419e42729a6b9fcdfd1c507fd672e12c9a9c7c868a5082a2ef, no fitting.

Question: does the changed committed state after a successful screen materially
lower expected suffix LN composition, or was the observed long suffix change
primarily specific to the subsequent random trajectory? This distinguishes a
persistent state-conditioning problem from finite sampled composition variability
before modifying model structure or training. State means the entire coherent
replay plus row/skeleton caches, not only a neural cache; this experiment cannot
separate occupation from learned history.

Closest analogue: common random numbers for comparing stochastic simulators,
Glasserman and Yao1992, https://doi.org/10.1287/mnsc.38.6.884. Pair the same initial
R/row generators across alternative valid states. Their variance-reduction
conditions are not established here: action/release decisions alter draw counts,
so later event-level alignment is not guaranteed. The fixed-stream2x2 cross is
exact path accounting, not a population causal percentage. Fresh independent
stream seeds then estimate a conditional state effect at two selected anchors.
This is diagnostic simulation, not a new generation model or decoding policy.

Fixed slice: the two failed-composition cases from screened result
3fff511d78ef4d0fd3400f33853d2e79219a69dbd034e67bee9d1263b5cb2ca9:
PromQueen seed19 H0/D2, duration136569, anchor56907; Airborne seed33 H0/D2,
duration271490, anchor32420. Anchors are the first H after the last resampled
window's accepted cut (56906 and32419). They have278 and1934 subsequent H.
All H timestamps, full audio, downstream profile2 and checkpoint are fixed.
Whole-chart baseline-to-screened LN fraction changes were-.1730708412 and
-.1353645280; those two selected observations do not estimate population drift.
Audio/Mel identities and source rows come from the prior freeze.json, SHA
38a41df0f086ff2f779ce7e9ad8c3498567b11da40e7bd606fb39415a5343636.

One intervention: choose original versus screened coherent post-H state at the
same anchor, then independently choose subsequent R/row random streams. Reproduce
the two saved trajectories through the shared native and buffered engines while
a behavior-neutral observer captures post-H forks. Select the captured screened
fork by exact accepted-prefix row identity. At these event boundaries the release
residual isNone; the future fixed H queue is identical. Reset only resource
timers for continuation. No cached tensor, full-audio embedding, LN endpoint,
profile or row action is transplanted separately from its coherent history.

First perform four exact-stream cells per case: stateO/randomO, stateO/randomN,
stateN/randomO, stateN/randomN. Both diagonal cells must reproduce saved full rows
exactly before new streams are run. Capture uses an artifact-local subclass and
scoped module substitution around the existing sampler; no product API change
is required. All four cells use the ordinary unscreened suffix law. Because the
anchor follows the last rejection, the screened diagonal must still match the
saved screened chart exactly. A parity failure stops the experiment.

Then use16 fresh stream seeds239251..239266, paired across original/screened
state. Release and row generators use existing xor salts0x4E51 and0xA301.
Do not screen or select these32continuations per case. Total planned outputs:
72 complete suffix continuations, plus four reconstruction runs. Preserve original
prefixes in exports, so every chart reparses and entering LN obligations remain
meaningful. Seeds are not tuned and no best candidate is adopted.

Primary metric per case: arithmetic mean across16 paired draws of screened-state
minus original-state suffix LN-head fraction, counting heads strictly after the
anchor. Report both arm distributions, paired differences, standard error and a
percentile95% interval from10000 paired bootstrap resamples, analysis seed239267.
The interval concerns RNG variability conditional on these two frozen states,
not unseen songs or selection of prefixes. A persistent downward state effect
is provisionally indicated by mean<=-.10 and interval upper<0; practical
negligibility requires the entire interval inside[-.10,.10] and abs(mean)<.05.
Other outcomes are unresolved. Report cases separately, with no pooled universal
claim or automatic promotion. Original-stream Shapley-style two-order averages
separate state and stream contributions algebraically, including their interaction.

Diagnostics: suffix note count, heads/H, LN count, held lane-time, strictHH<20
and experimentalRH<=20, and composition in consecutive64-H blocks. Native timers,
full completion, H equality, exact input/weight hashes and independent reparse
are guards. Numeric/LN guards of the previous candidate are not a stopping rule
for these deliberately counterfactual diagnostic suffixes. Never publish their
rows to a player. Invalid replay, changed weights, non-finite output, a failed
parity/H check or resource cap stops further draws without increasing budgets.

Qualitative check: use the frozen Lens context to inspect original/screened
state continuations for each case's fresh seed with the largest absolute suffix
LN-fraction difference (tie chooses lowest seed). Read8seconds after the anchor
and8seconds beginning at the64-H block with the greatest arm difference in LN
fraction (tie earliest); merge overlaps and read all time pages and complete
action/articulation tables. Any new strictHH is reported automatically; inspect
at most the earliest witness per case from the exact-stream cross, separately
from the composition scopes. This is scoped diagnosis, not a quality pass for
72 new maps. No listening/player assertion is allowed.

Procedure: create artifact-local run.py and pin its SHA before execution; run
uv run --extra mps python artifacts/joint-audio/20260925-continuation-state-rng-v1/run.py.
Pin clean product and Note OIDs in freeze.json. CPU sampling uses one Torch
thread on AppleM5/24GiB, Python3.10.20/Torch2.11.0, with full cached canonical Mel
and unchanged model. Budget900s total sampler/driver,90s per continuation,
30000rows per chart,500MiB new artifacts. Network not needed for native sampling;
source verification already read the primary analogue. Existing process resource
checks apply. Lens preparation/review is separate bounded offline work.

Fresh owner artifacts/joint-audio/20260925-continuation-state-rng-v1; fail on an
existing freeze/result, no overwrite/resume and no rerun of the prior15-case
cohort. Behavior-neutral capture scripts are local artifacts; curated findings
will use a recoverable product source and summarize enough evidence for a fresh
clone. No parameter, training-budget, temperature, threshold or control sweep.

Confounders: two prefixes selected for observed composition failures; H fixed;
state comprises occupation, exact clocks and both neural histories; finite
samples and state-dependent random consumption; legacy training exposure and
musical quality unresolved. A negative mean tests these frozen states only.
A weak state effect with broad individual composition variation would weaken
persistent-history drift as the sufficient explanation, while still leaving
control reliability unresolved. An imprecise result returnsREFINE rather than
escalating compute automatically. All outcomes leave the playable-system goal
active and the Card proposed.


### Result Log: continuation-state-rng-v1

Card continuation-state-rng-v1 revision1, accepted none, remains proposed.
Exploratory run under standing local execution authority. Clean baseline and
intervention product source65ef95c092e7d3024d99ed60e37c2c22ee041df9; no product
code, model parameter, temperature or training objective changed. Execution
Note92ec31ec8b6a3cd23829a203691d24dfd2e9ca77. Artifact-local observer subclass
captures post-H forks without changing sampler behavior. Native session58734
and Lens rendering73459 are terminal-success; do not rerun them.

Command: uv run --extra mps python
artifacts/joint-audio/20260925-continuation-state-rng-v1/run.py.
Output owner artifacts/joint-audio/20260925-continuation-state-rng-v1 was fresh;
no overwrite/resume. AppleM5/24GiB, one Torch CPU thread, Python3.10.20,
Torch2.11.0, unchanged checkpointabc27f1d192869419e42729a6b9fcdfd1c507fd672e12c9a9c7c868a5082a2ef.
The same full audio/Mel bytes and H0 plans were verified against the parent
freeze. Shared encoded tensors agree exactly across original/screened states.
Original generation seeds19/33; fresh paired streams239251..239266; bootstrap
seed239267. No new audio or model training in this diagnostic.

Stable identities:
- run.py8bb30cf6e1d6d21807941aa9467652d19d0f423d1c180171a1b6d975f45b7470;
- freeze.json26a6d7234a98d389aa51eb4faec0dc1b1d98a176bd6f3daaf19fad1b8a50bda8;
- gates.json89dfba2655004fd346c6660cf922bb19e8c93c9ea87a5b3b0ac0b658c50acda9;
- result.jsonb0302e1769243c2212993ebe898f484a664c9500e7d455b29dc0c0657d900074;
- review.jsonbf0ee0cfcf0db05fb5d756e82f3fb69f252a8c59556daf9f7314bcd8c479d9b5.

Both original/screened reconstruction trajectories and all four diagonal suffix
controls reproduce their saved full row hashes exactly. Captures at56907 and
32420 have residualNone and identical futureH. Original-state and screened-state
forks remain unchanged by each continuation. All72 outputs complete, retain
exact prefix/H, independently reparse, and have no final open LN. Model tensors
and checkpoint bytes remain unchanged. Driver takes199.215619416s, suffix
sampling182.567487877s; all per-run90s and total900s bounds pass. Final owner
including Lens evidence is94,834,729bytes, within500MiB. Completion is the stop
reason; no failure artifact or protected-field deviation occurred. The result's
qualitative_review=pending is its immutable pre-review snapshot, superseded by
the separate completed review record below.

Fresh-stream mean suffix LN fractions, original versus screened state:
PromQueen .5626799795 vs.5220572853, paired mean-.0406226942, standard error
.0474746444, paired bootstrap95%[-.126691016,+.051046651]. Airborne .4977251388
vs.4159165248, paired mean-.0818086140, standard error.0261266686,
interval[-.129951062,-.030024727]. Prom does not establish a directional mean
effect here. Air supplies evidence of a negative conditional state effect but
misses the declared mean<=-.10 material-effect line. Neither satisfies the
practical-equivalence rule. Both planned classifications are unresolved; do not
change the thresholds after seeing the estimates.

The fixed saved-stream2x2 cells differ sharply from the fresh-stream expectation.
PromOO=.6343612335,ON=.6038543897,NO=.2170900693,NN=.2842377261; the two-order
state/stream terms are-.3684439139/+.0183204065. AirOO=.4701022817,
ON=.4368932039,NO=.4671562627,NN=.3373644704; terms-.0512373762/-.0815004351.
These are path-specific algebraic attributions, not unique causal responsibility
or a percentage of failure belonging to R1-restored. Prom's fresh original-state
fractions range.2678..8545 and screened-state.1829..7607; Air ranges.2685..6773
and.2125..5805. Substantial sampling variability remains under both fixed states.

Lens preparation command uses this owner's inspect_lens.py and the exact parent
manifestee0e3a32a527aefaf2b0fe84bd68046f9af8f771f257068364f096b4cdb2919e.
The selected contrast seeds areProm239252 andAir239258. All9scopes/33time pages
and complete action/articulation tables have been read, with entering holds.
Plan1cbd2e7bb3953f3cf59d57d090a907c48ab091a9ac3feedaa2806d70a8c95f31;
render0f43706be9fb55131aab29e38e9d42220e3f9171186f74d54164a5aaa9db9b4a;
manifest03b1d4aada4a3994c4f35d6c66ab122a93d55b0cd5a37a13c56eb4e84fe066a4.
Parent source/human evidence remains byte-identical. No listening/player test.

Actual organization supports the diagnostic distinction. Prom's chosen fresh
pair reverses the initial effect: screened-state suffix.6990LN vs original.3095.
At124814..132814, screened state has dense short-LN overlaps and independent
subsets, original state starts with broad tap chords; both retain later long
holds and the1ms cross-column126455/126456H pair. Air's lower-LN screened state
is locally more LN-heavy immediately after the anchor (.8039 vs.2963 in the8s
scope), but much less so at223054..231054 (.0870 vs.8696). In that later scope
both have exactly91H and115heads, with10 vs100LN starts. The tap-heavy arm still
has LN2 lasting610ms across six H; the other has independently released subsets
and longer holds amid short relays. A4ms LN and20ms LNpair remain unresolved
articulation questions, not automatic strictHH violations.

Across72 unscreened diagnostic suffixes there are two strictHH and71RH<=20 pairs.
The declared exact-cross witness is visually/table-confirmed: AirSO-UN uses
TAP0/1+CLOSE2 at252770 whileLN3 remains held, then LN0 at252778, a strict8ms
TAP-to-LN-head repeat. Lane2 is physically free but hasRH8, a separate criterion.
The other automatically reported HH is AirSO-U239266, lane2 at144014/144025
(11ms); it is not an additional declared Lens scope. Diagnostic suffixes were
unscreened by design; these counts do not overturn the earlier screened15-case
zeroHH/RH result. No diagnostic output is adopted for publication.

Architecture inspection identifies permitted propagation paths, without claiming
which causes the measured effect: exact LN state, lane clocks, cumulative
row/note counts, row TCN and H/R skeleton TCN all change coherently. The exact
query includes log1p(row_count)/10 and log1p(note_count)/10, so finite row-cache
length does not erase every older difference. Even without those counters,
autoregressive generated-row feedback can propagate a perturbation beyond the
cache horizon. Skeleton still sees only own history and minimal LN feedback;
no generic row-layout cache has been added to it. Future path isolation must
not splice an incoherent LN state into a supposedly valid replay.

Evaluation recommendation REFINE. The evidence weakens one sampled LN-fraction
change as a sufficient diagnosis of collapse, while preserving a real conditional
state effect in Airborne. It neither vindicates the whole model nor establishes
a need for larger memory. Current composition guard failures remain recorded;
variation is not a reason to silently declare them passed. Human-confirmed
strictHH<20 remains a separate publication-quality invariant, counting LN heads.
NLL, descriptor alignment and preservation of one sampled chart are distinct
from actual musical/player quality.

Curated analysis at product579c0a8add28377881a12ab263059fa0fc4ca938:
docs/research/continuation_state_dependence.md, linked from the screen study.
Scoped link/hash/count checks,72 reparses, diagonal controls, resource bounds
and git diff --check passed. No model behavior was edited, so no repeated
unrelated unit suite was run. Product and Note commits are local only.

Next research direction: broader whole-system evaluation should now use fresh
audio contexts under ordinary matched H/profile conditioning, alongside the
bounded unpublished screen, rather than extending seed searches on these two
synthetic component crosses. Select new cases from pinned existing source/audio
and human-context owners with explicit non-reuse and support coverage; formalize
a new bounded Card before generation. Inspect timing/row/LN organization and
publication failures together, retaining the strict human HH rule and separate
RH/articulation questions. Any targeted state-path intervention remains a live
branch if the same practical failure recurs. No new Card or run has started for
that stage. The ultimate playable-system goal remains active; all jobs here are
terminal and both completed studies must not be rerun without a new question.


### Experiment Card: fresh-audio-system-v1

Revision1, proposed, accepted none. Owner2026-09-23-audio-skeleton-r1-integration.
Standing local implementation, research and execution authority applies. Previous
goal turn completed a diagnostic, changed durable state and yielded a different
next action; it was progress, not a wait or repeated blocker.

Question: does the complete audio-file-to-published-row system remain functional
and musically plausible on new audio contexts, with varied arrangement requests
and actual bounded continuation screening? Earlier tests used repeated songs
or synthetic H/profile crosses. This comparison exercises ordinary matched
H/R/row conditioning, BOS startup, real decoding/Mel preprocessing, streaming
and export together. Do not substitute NLL, source copying or a count proxy for
playability or expressive organization.

Clean baseline579c0a8add28377881a12ab263059fa0fc4ca938. Unchanged shared-profile
checkpointabc27f1d192869419e42729a6b9fcdfd1c507fd672e12c9a9c7c868a5082a2ef.
Prior15-case paired result3fff511d78ef4d0fd3400f33853d2e79219a69dbd034e67bee9d1263b5cb2ca9
has complete baseline/screened charts, strictHH3to0, RH12to0, sampler cost+3.64%,
and two failed LN-composition guards. It is contextual evidence only. Baseline
values on this new panel are not yet observed and will be measured prospectively
with the frozen direct decoder, not inferred from the old15 cases.

Closest analogue remains bounded joint autoregressive continuation screening,
with the limitations explained in docs/research/unpublished_continuation_screen.md
and primary NeuroLogic A*esque/AliveParticleFilter/TwistedSMC references there.
This stage is system integration and evaluation coverage, not an estimator,
training objective or new model family. Defer state-cache surgery, additional
seed sweeps, bigger memory and fitting until this panel identifies a practical
failure that changes the next design decision.

Panel preparation ran without model generation using prepare_panel.py in
artifacts/joint-audio/20260925-fresh-audio-system-v1. Panel SHA
484d173628856a431e034b1dbdacab9173c40f3edf696d1b9a19721aeec70dc9.
Eight source/catalog validation groups: A Fool Moon Night (276990ms), Operation:
Zenithfall (357796), Hysteric Night Girl (301008), Goodbye (300000), Revenge
(278399), Take [Future] (144236), As It Was (167303), Yomi yori (498989).
All exact source bytes match recovered local aliases and audio references.
All groups, encoded hashes and decoded peak-normalized waveform hashes are
absent from the selected joint corpus; encoded hashes are absent from the
recent five-audio studies. Prior R1 exposure and perceptually identical alternate
recordings/crops are unknown. No target enters TRAIN and no TEST split is opened.

The source/human context is the frozen Lens bundle manifest
03b1d4aada4a3994c4f35d6c66ab122a93d55b0cd5a37a13c56eb4e84fe066a4,
using the existing catalog, alias audit and their verified byte identities.
The16 human records cover Tech, LN coordination, repeated chords, Trill and
negative contrasts. Only FoolMoon's prominent LN judgment has explicit High;
missing confidence remains unrecorded, absent labels apply only at their scope,
and no dump-positive label is invented. Reference scopes do not require the
model to copy that arrangement or force that tag on a new output.

One intervention: choose the existing direct decoder versus the existing
unpublished screen through the normal source-free audio entrypoint. Add one
boolean screen_unpublished to AudioInferenceConfig, defaultFalse, fully projected
by the existing packaged Hydra entrypoint. True selects the unchanged8s window,
20ms halo,4attempts and experimentalRH<=20 screen in buffering.rollout_buffered.
StrictHH<20 always counts TAP and LN heads. Reject simultaneous use of the older
correct_short_attacks option, whose optimistic row edits are a different policy.
Leave the default direct path and probabilities unchanged. No new window search,
release floor, target grid, generic row-to-skeleton feedback or model weights.

Likely product changes: inference_config.py, inference.py, their owning tests
and the planned-audio/unpublished-screen guides. Keep YAML package-owned, typed
validation, complete option consumption and Torch-free help. Focused checks must
cover selection of the real buffered engine, no rejected-row publication on
attempt exhaustion, honest capped output, conflicting-policy rejection and
Hydra projection. Existing engine invariants are already tested and need not be
reimplemented in adapter tests. Record a clean tested intervention OID before
running the panel; its baseline diff must contain only this declared adapter.

Comparison:8audio x3 arrangement requests x2 publication policies=48 outputs.
Requests are automatic prior, profile1 and profile2 (existing TRAIN medoids;
not derived from these targets). H/R/row use the same condition in every output.
Case seeds251701..251708 in panel order; paired policies use the same seed.
No fixed reference H, chart seed, redline, annotation or future LN endpoint enters
generation. Every call reads original full audio and recomputes canonical Mel,
loads the same checkpoint, then starts atBOS. All preprocessing digests must
match the panel. No control or seed is selected based on generated quality.

Run through infer_audio with typed settings and the same incremental consumer
interface as the CLI. Additionally run the actual packaged CLI for the longest
automatic screened case, consuming stdout online in a subprocess; require exact
rows and event ordering agreement with the API and file stream, while reporting
real process-start readiness separately. This duplicate is an integration
control, not another independent quality sample. Imports/setup and checkpoint
load are distinguished from producer-service and cached-Mel timing.

Incremental consumer: independently replay each accepted complete row and
coverage, track open LN heads, require no rollback or future endpoint in the
stream, and compare all emitted rows with final rows.jsonl. No emitted prefix
may contain strictHH<20 or screenedRH<=20 in the screened arm. Consumers receive
no speculative coverage, rejected row or fabricated LN tail. For complete outputs,
require independent export/reparse, complete audio clock and no open LN. Capped
outputs remain in all counts and retain honest prefixes, with no playable export.

Primary mechanical/system gate: all24 screened cases complete, zero strictHH
and zero experimentalRH, and all H times exactly match their paired direct
case. Report direct counts even if zero; if direct is already clean, row identity
is the expected control rather than evidence of an improvement. Readiness requires
30physical rows plus8s settled coverage, or true completion for short charts.
Producer readiness including fresh preprocessing<=4s; longest actualCLI readiness
from process start<=6s; each accepted-window service<=2s; total paired screened
sampler time<=2x direct. Report every failure rather than silently increasing
attempts/budgets or dropping an audio.

Playback accounting: once readiness is observed, simulate the consumer clock
at1x against arrival times for all emitted coverage. Also report4x service-time
scaling and one injected1s stall after readiness as offline sensitivity checks,
not real OS/network guarantees. Zero coverage starvation in the1x trace is a
system guard. Per-window rows/heads, rejections and maximal unpublished buffers
expose dense-part stress; all exported audio bytes must match the input.

Composition is assessed as distribution and organization, not preservation of
one sampled chart. Report H rate, heads/H, LN fraction, held lane-time, durations,
full-hold episodes and same-lane gaps across both policies and all requests.
Flag absLN-fraction drift>.10 or head ratios outside.8..1.25 for close inspection,
without relabeling prior failed guards or calling any flagged variant good.
These descriptors cannot certify difficulty, Tech, dump or player comfort.
No NLL gate and no model promotion follow from numerical mechanical success.

Qualitative scope is pinned before model outputs: all16 human review contexts,
merging overlaps within each source; compare corresponding generated scopes for
all3 requests and both policies, reusing only exact event/entry-hold/articulation
identity. Also inspect the densest8s interval of each automatic output (maxH,
tie earliest, merge overlaps), all windows with rejected proposals in screened
outputs and every strictHH witness from direct outputs with1s entry/exit context.
Report/render capped boundaries when present. Read every time page and complete
action/articulation table in the frozen resulting plan. New machine style claims
must compare actual relations to the frozen human context, not aggregate counts.
Reference timing metadata is an inspection aid only; generated120BPM headers do
not supply musical beat truth. No listening or player-testing claim is allowed
without a separate actual test. Dump acoustic elaboration remains unresolved
if only chart evidence is available.

Execution command: uv run --extra mps python
artifacts/joint-audio/20260925-fresh-audio-system-v1/run.py, with an artifact-local
pinned driver, clean product/Note OIDs and checkpoint/panel hashes in freeze.json.
AppleM5/24GiB, Python3.10.20/Torch2.11.0, one Torch CPU thread, max90s and30000rows
per output, max900s native/API/CLI driver and1GiB new artifacts. Existing2GiB
available-memory/40GiB-free-disk/PAUSE checks apply. Network unnecessary. Lens
render/review is separate offline work, still subject to storage guards. The
prepared panel/feature arrays may be reused only after exact hash verification;
all native output directories and freeze/results must be fresh, no overwrite
or resume. Old cohorts are not rerun.

Fail fast after the first two audios' automatic paired outputs: stop on identity,
publication, replay/H, resource, finite-output or completion failure before the
remaining requests/audios. A later quality or runtime guard failure remains a
measured result; continue other independent cases within bounds to diagnose
coverage unless input/replay integrity or global resources are compromised.
No retry-budget or model change inside this comparison. Unexpected errors save
an explicit failure and stop dependent work.

Positive evidence is a complete mechanically screened system with concrete
expressive organization on new contexts and bounded arrival latency, still not
universal playability. Failure localizes whether audio/head planning, R/row
materialization, condition control or publication budget needs revision. Sparse,
generic or acoustically unverified charts cannot pass an expressive claim solely
because they avoid short repeats. Confounders include the curated eight-audio
selection, one seed/request, soft descriptor controls, paired autoregressive
history changes, unknown oldR1 exposure and lack of player tests. Recommendation
remainsREFINE until actual results and scopes are assessed; Card acceptance and
remote publication are not implied.


### fresh-audio-system-v1 execution source

Clean intervention0b35eced5a8118da745a84dccee75e8b0e503bed implements only the
declared source-free inference adapter, semantic validation, owning tests and
documentation. Baseline-to-intervention diff has six paths,58insertions and
12deletions. No sampler probability, buffer policy, checkpoint or source data
changes. Card revision1 remains proposed, accepted none; standing run authority
applies. The only new setting is screen_unpublished=False by default, selecting
the existing8s/four-attempt/RH-screen policy whenTrue and rejecting simultaneous
optimistic row correction.

Focused verification: inference tests8passed in2.32s; buffer/package tests7passed
plus22subtests in.90s. The actual model-backed impossible dense-head fixture
exhausts four proposals and exposes no update or speculative row through the
entrypoint. Default direct-row parity, Hydra projection, conflict rejection and
Torch-free help are covered. Changed prose links and git diff --check pass.
No unrelated suite or CUDA coverage is claimed. Product commit is local only.

The artifact driver will pin this clean source and this Note's committed OID
before creating its fresh freeze.json. The first two automatic direct/screened
pairs are the gate. Existing prepared features are integrity references only;
every infer_audio call still decodes original audio and recomputes Mel. Consumer
primary starvation accounting uses zero extra presentation lead; an additional
2s visual-lead diagnostic is reported separately, not silently substituted for
the declared guard. No native output exists yet at this dispatch.


### Result Log: fresh-audio-system-v1 native complete, Lens in progress

Card revision1 remains proposed, accepted none. Clean source
0b35eced5a8118da745a84dccee75e8b0e503bed, execution Note
ad6aabec5f5ddca1be0a7b26ff47459311b81ba5, unchanged conditioned checkpoint
abc27f1d192869419e42729a6b9fcdfd1c507fd672e12c9a9c7c868a5082a2ef.
The prepared eight-audio panel remains484d173628856a431e034b1dbdacab9173c40f3edf696d1b9a19721aeec70dc9.
Native session1669 and Lens render57080 are terminal-success; no model run is
active. Do not rerun generation, preparation, tests or rendering to resume review.

Output owner artifacts/joint-audio/20260925-fresh-audio-system-v1.
Driver447b0d758d399d273efe3d395f46d9d7f8031e380764b56b54a5937b950e3af4;
freeze37bf355a871381aaa221b6444dc80cfdb94d6865f32774ac4c6b8a0edef0934b;
gate0329a592198e8704ad971346e741a660562c1e2812954fa4589a3b2199fd906f;
result9f3c4580adaa6239c44e04bb5d95a0ed921b8fd958b1eee26ccd8cedfa91e862.
The two-audio automatic preflight gate passed. All48API outputs (8audios x3
requests x2policies) complete and independently reparse. Every paired H sequence
matches exactly. Source-free calls decode the original full audio and recompute
canonical Mel; all PCM/Mel hashes match the panel. Consumer replay matches the
file and final rows, with no future endpoint, rollback or rejected-row leakage.

Direct24 outputs have2strictHH and8RH<=20 pairs; screened24 have0/0. There are
11rejected proposals, and18pairs require no retry and remain row-identical.
Total sampler time83.981923045s direct versus86.680394292s screened, ratio
1.032131572. Screened producer readiness including fresh preprocessing is
.576973416..1.575609500s; maximum accepted-window service.356228625s and maximum
unpublished proposal105rows. All declared system guards pass. No coverage
starvation occurs in recorded1x arrival traces,4x service scaling, the injected
1s stall, or the separate2s presentation-lead diagnostic. These are local/offline
trace checks, not real OS contention, network or player evidence.

The additional actualCLI run on498989ms Yomi yori reproduces its automatic
screened API rows exactly. Stdout is consumed as produced and equals events.jsonl
byte-for-byte. Process-start readiness2.920690583s, below6s. The entire driver,
including49calls, preprocessing, exports and checks, takes217.125090542s.
Native output uses171,065,586bytes, with copied audio verified and then replaced
by links to the same immutable input. Input/checkpoint bytes remain unchanged.
No training or new control selection occurred. No protected-field deviation.

One descriptive composition flag remains: Hysteric profile2 LN-head fraction
increases by.1886276894 after screening, with head ratio1.044622936. It is neither
a silent pass of an old guard nor automatic evidence of a bad chart. This sign
also differs from earlier LN reductions. Detailed organization must decide the
interpretation; numerical system success does not promote model quality.

The complete Lens plan has108scopes,75newly rendered scopes/209time pages and
33current semantic-reuse scopes. It includes all16human records through12merged
reference contexts, matching generated contexts for all requests/policies,
automatic dense windows, every resampled window and both direct strictHH
witnesses. Source/human evidence remains byte-identical. Rendering alone is not
inspection. At this record,23new scopes/54pages and all their complete action/
articulation tables are read: references46..57 and generated5..15 (allFoolMoon
andGoodbye scopes). The other52new scopes/155pages remain unreviewed. Reuse is
covered only when the referenced scope is read. Exact inventory and observations
are in review-progress.json; no final review.json exists.

The source examples demonstrate explicit relations: FoolMoon's repeated108/109ms
LN starts with217/435ms overlapping holds and staggered subsets; Goodbye's
21/22ms cross-column approaches into64/108ms independently released LNpairs;
Hysteric's39/58/78ms changing tap flow; Zenith's repeated four-LN chords with
staggered tails and later two-held-lane inner-column alternation. The source
negative cases distinguish a simple single-LN relay or single held lane from
LN coordination. Preserve human scope/confidence; no dump label is invented.

Initial generated inspection is mixed. FoolMoon profile1 retains dense broad
chords and fine cross-column bursts, while profile2 mostly produces single-LN
relay: every new hold in its first fixed scope ends at the next H with zero
interior H. Thus high LN fraction alone does not yield reference-like coordination.
Goodbye profile1 instead gives repeated broad LN chords, including full-occupation
entry and some independent tails, before switching to tap chords. Its profile2
scope is simpler with isolated/paired holds. Automatic Goodbye's dense8s has89H
of mainly70–100ms movement with chord accents, whereas its fixed source context
is much sparser and lacks the reference's fine LN-pair organization. These are
scoped observations, not a complete style verdict or listening/player result.

Recommendation remainsREFINE pending the remaining declared inspection. Immediate
next action: continue Lens at the saved inventory, then publish the full system
measurement with its actual expressive limitations. Do not select a larger model,
new seed sweep, duration floor or extra hard timing grid merely from the counts.
The ultimate playable-system goal remains active. Product/Note commits are local;
no remote publication or Note lifecycle change occurred.


### Result Log: fresh-audio-system-v1 completed inspection and evaluation

#### Experiment and reproduction

Owning Note 2026-09-23-audio-skeleton-r1-integration, Card fresh-audio-system-v1
revision 1, proposed, accepted revision none. Standing local research/execution
authority applies. This appends the completed inspection to the preceding
native result; it does not overwrite that result's pending-review snapshot.
Clean implementation 0b35eced5a8118da745a84dccee75e8b0e503bed and all run inputs,
seeds 251701–251708, checkpoint, commands and environment remain as recorded
above. No native generation, fitting, preparation or rendering was repeated.
All jobs remain terminal-success, with no active training job.

Output owner artifacts/joint-audio/20260925-fresh-audio-system-v1. Completed
review.json SHA 9302f23daed2ce010465437c1a3d90793754403035ac447121619fe91da5e737.
Plan 6c770c052b949f921f49d51d3e14afb3073faf1f1248c174325f131ed6578fdd;
render result 6330f8580595c34c7ba86c75187e5559e8cb7ac2727fc8081cd1001ae87dc1b6.
Native result remains 9f3c4580adaa6239c44e04bb5d95a0ed921b8fd958b1eee26ccd8cedfa91e862.
The new review records the observed tables/pages and is not a modification of
human labels or generated row data. Artifact storage after rendering was
279,784,142 bytes, below 1 GiB; native driver 217.125 s, below 900 s. Review adds only
small JSON records. No collision, overwrite, resume or budget escalation.

#### Results and conformance

All 75 new scopes and 209 time pages have now actually been viewed, with complete
physical-action/articulation tables read. All 33 reuse scopes match the semantic
hash of an inspected current scope, including entry holds and articulation.
All 108 declared scopes are covered. The plan and 16 human records are unchanged.
Selected-scope inspection is not whole-song visual review, listening or player
testing. No protected Card field changed and no planned scope was dropped.

All 48 API charts and the repeated actual CLI output complete/reparse. All 24
screened charts have identical H times to their direct pair, zero strict HH < 20 ms
and zero experimental RH <= 20 ms. Direct charts have 2 HH / 8 RH; 11 proposals are rejected,
and 18 pairs have no retry and exact row identity. Sampler time 83.982 to 86.680 s
(+3.21%). Screened producer readiness .577–1.576 s includes fresh preprocessing;
API consumer readiness .623–1.625 s also includes entry setup/delivery. Actual CLI
readiness 2.921 s includes process startup. Accepted-window service max .356 s,
unpublished proposal max 105 rows. All specified mechanical/runtime guards pass.
No covered trace starves; playback simulations are not actual client/network
or background-contention tests.

As It Was supplies a concrete strict HH repair: direct 92196 T1/2/3 followed by
92215 T2 becomes92196 T1/2/3 then92215 T0, retaining the 19-ms H interval. Four
consecutive 0/1/2 triples at 93237, 93410, 93584, 93755 ms remain. Other 3/7/17-ms
cross-column splits survive. This is not a blanket attack-spacing or anti-Jack
rule. Strict HH counts TAP and LN heads; exactly 20 ms is excluded. RH and LN
duration remain distinct from this confirmed human BAD condition.

Zenithfall automatic early material retains repeated broad LN chords with
independent tails: 14578-ms triple lasting 112/85/60 ms, 15755-ms quad lasting 78/78/222/222 ms. Generated
Hysteric scopes include overlapping LN roles spanning several H and staggered
subsets. Conversely, Fool Moon profile 2's first fixed scope remains a single-LN
relay, every new hold ending at the next H, not the source's overlapping
coordination. Take's source six-row lane 0 versus complete 2/3 Trill is not
established in corresponding generated scopes. These are observed mechanisms
and omissions, not a requirement to copy a unique source answer.

Whole-chart descriptors corroborate weak condition calibration. Profile 1 asks
for 7.032 H/s and 2.141 heads/H; realized H rate ranges 2.938–7.209 H/s, and screened
width ranges 1.032 (Revenge) to 2.590 (As It Was). Profile 2 asks for LN fraction .734;
screened realizations are .159, .242, .523, .247, .078, .144, .084, .074 in panel order.
The chart-level request does not require every local window to contain LNs;
the whole-chart deviation and local organization are separate observations.
Some poorly calibrated pairs need no retry, so the issue predates screening.

The single descriptive composition flag is still Hysteric profile 2: LN fraction
.334 to .523, delta +.1886, head ratio 1.045. Inspected later material has more
overlapping and independent LN roles, but also isolated 12/15/18-ms LN objects.
Neither more LN nor passing HH/RH establishes quality; no universal LN-duration
floor follows. Zenithfall's later 3480-ms held-lane role changes to isolated
relays after an earlier resampling. A single paired suffix cannot isolate
state persistence from changed RNG. All head ratios stay inside .8–1.25; the
composition flag is reported rather than silently promoted to a quality pass.

#### Evaluation and decision

Recommendation: REFINE. This exploratory Card has no accepted revision and does
not support a lifecycle transition or model adoption. The evidence establishes
low-cost mechanically screened publication on this frozen eight-audio panel, while
retaining concrete expressive mechanisms. It does not establish consistently
controlled or musically good charts, nor a human playability result.

The strongest remaining alternatives for poor control are insufficient learning
of the existing persistent condition, shortcut coupling through its shared
projection, and audio/corpus generalization. The earlier fixed-weight path
crosses identify a large direct H-condition contribution to row width, but
their out-of-support crosses do not prove that removing that path improves
ordinary generation. Likewise a local missing tag does not imply a representation
failure, and the three global descriptors do not define LN coordination/Tech.

Next discriminating work should compare an unchanged continuation fit against
one factor-specific condition-routing intervention under equal data/updates,
retaining direct audio and actual future-H preview for rows. Before choosing
that intervention, pin which cross-factor information is necessary for joint
arrangement and which shortcut the change is intended to remove. Do not replace
actual preview with a requested density scalar or prohibit legitimate density/
width interaction. Further training alone remains a live explanation; new
parameter count, more seed sweeps and duration clamps are not justified by this
result. A subsequent Design Card must freeze the exact intervention and guards
before implementation/run. Standing execution authority persists; exact Card
acceptance, quality adoption and remote publication are still absent.

Durable analysis is docs/research/fresh_audio_system_evaluation.md at product
e28435f10cfb154acfe8eb9f0507d2a6c242ff79, linked from the playback and screen
owners. Only those three documentation paths changed. Numeric/source identities,
scope coverage and relative links were checked; staged git diff --check passed.
The already-passing implementation tests were not rerun for a prose-only change.
No product behavior, checkpoint, default decoding policy or human annotation
changed in this completion pass. The ultimate playable-system goal is active.


### Experiment Card: profile-density-routing-v1

Revision 1; proposed; accepted revision none. Owning Note:
2026-09-23-audio-skeleton-r1-integration. Standing local implementation, fitting
and evaluation authority applies. The preceding goal turn completed the fresh
panel's inspection, persisted its evidence and changed the next research action;
it was progress, not a wait or a blocker.

Question: does directly supplying the requested global H rate to R/row factors
encourage unwanted width response, beyond what ordinary continued fitting fixes?
The selected intervention removes that one direct input while retaining actual
H preview, full audio, width/LN conditions, R's LN projection, row history and
frontier2. It does not remove legitimate dependence on actual H spacing or
force density/width to be statistically independent in generated charts.

For standardized profile z=(density,width,LN), head queries retain A+Wz. R/row
queries instead use A+W(0,width,LN). W remains the same shared 3-to-224 projection;
all learned tensors, parameter count and initial values are identical between
arms. A new boolean profile_head_rate_downstream defaults true; false selects
the intervention in training and native inference, including hypothetical
full-held release waits. No input is subtracted from an already-conditioned
floating-point tensor: construct the two views from the same unconditioned
audio encoding. Encode complete audio only once.

The approximation is explicit. A complete H sequence already determines its
global density, but each downstream query sees a finite preview/history.
Removing the request therefore may remove useful information about unseen
future density. H still reads all three attributes, so the proposed factorization
retains indirect effects through the generated plan. Changes in shared training
gradients are allowed; this is an architectural comparison, not a frozen-weight
estimate of one path's causal effect.

Closest analogues: MuseMorphose (https://slseanwu.github.io/site-musemorphose/)
uses computable rhythmic/polyphonic attributes with persistent decoder
conditioning; CTRL (https://arxiv.org/abs/1909.05858) conditions generation on
codes derived from observed data. FiLM (https://arxiv.org/abs/1709.07871) is an
alternative for stronger feature-wise conditioning, not evidence that this
particular path should be removed. This test adapts conditional-factor routing;
it is not a new representation or objective. It does not import bar grids,
VAE latents, pretrained audio labels or fixed section classes.

Live alternatives: (a) current conditioning is undertrained, predicting similar
gains from matched further fitting; (b) direct density input is an unwanted
shortcut, predicting better calibrated width and lower density-to-width response
after removing it; (c) density is useful to compensate finite preview, predicting
lost LN/row organization or control after removal. Stronger FiLM/deeper injection,
larger encoders and wider data changes are deferred because they would not
separate these explanations in one comparison.

Clean baseline product e28435f10cfb154acfe8eb9f0507d2a6c242ff79. Initial checkpoint
artifacts/joint-audio/20260924-alias-restored-v1/planned-training/shared-profile-cond-main-v1/last.pt,
SHA abc27f1d192869419e42729a6b9fcdfd1c507fd672e12c9a9c7c868a5082a2ef.
Both arms copy every source tensor, including learned profile-prior weights/bias
and profile buffers, then start a fresh optimizer. Extend the existing initializer
to allow same-bank profiled continuation and this declared mode change; reject
other architecture, bank, normalization or corpus mismatches. Do not reinitialize
the learned prior when validating the bank.

Data remains 615 TRAIN arrangements / 240 audio groups and 36 VAL charts in
manifest4ad9abfd0ae7798e0a85b96a5dbecdaefd2a81f78bf181438407442b964118e1.
Use frozen normalization9cf461a0e825f974f0a80a364123c7afedf1683af76e60fe09b0fbe51c2c8287
and existing 16-profile bank a8973b7f94fde90d9f3cc639eb63e781dd295435622ce79fd54fe244789e6f03.
No source/context from the fresh-audio panel enters training. All paired real
charts remain separate targets; no union labels or style-tag pseudo-labels.

Baseline evidence: the prior nine-case finite H-request-to-width coefficient
is .49060, including .37983 through downstream condition under the two-order
attribution convention. Those deliberate component crosses are out of training
support and cannot establish improvement from a new model. On the fresh eight
audios and requests 1/2, the current direct checkpoint's mean squared standardized
descriptor distance is 7.3404225 across 16 charts: components .8738584 H rate,
2.3704218 width, 4.0961423 LN fraction. This is descriptive small-cohort evidence,
not a population bound. The matched continuation endpoint and four-request
comparison values are not yet observed and will be measured prospectively.

Two serial training arms, shared=true and routed=false, each start from the
same pinned checkpoint. Preflight: 32 updates, first 32 entries of the same
1200-update plan, max 300 s/arm, six VAL charts at update 32. Main: 1200 updates,
max 2400 s/arm, all VAL charts every 300 updates. Main starts again from the
original checkpoint, never from the preflight endpoint. Fixed seed252801,
sample_seed252802, validation_seed230943; protocol_name=profile-density-routing-v1.
Otherwise keep original two songs/update, two 8-s intervals/song, learning rates
3e-4 new and3e-5 inherited, weight decay.01, clip1.0. Every trainable module
remains trainable; no endpoint selection or early stopping by NLL.

Test contracts before a clean implementation commit: original true-mode scores
and saved native row parity; false-mode downstream score/gradient invariance
to density at fixed audio/H/state while width/LN and H sensitivity remain;
training/full-native query agreement including full-held waits; fixed-generated-H
native row invariance across density-only test codes; unchanged copied tensors
and learned prior under warm start; strict bank/config rejection; checkpoint
load and typed Hydra projection. Use mps extras and dev group for pytest.

Preflight native: both endpoints generate the first 30 s of FoolMoon and Hysteric,
with complete original audio available, requests 1/2, direct decoding, seeds from
the frozen panel. Stop before main on nonfinite scores/gradients, missing module
updates, input/protocol mismatch, incomplete prefix-clock replay or export
failure. Short-attack/style defects are measured rather than silently changing
support. Also verify the unchanged initial true-mode model reproduces saved
full automatic FoolMoon rows, and switching only mode at initialization preserves
its H stream. No main run until these integrity checks pass.

Main native: the same eight-audio panel484d173628856a431e034b1dbdacab9173c40f3edf696d1b9a19721aeec70dc9,
seeds251701..251708, five requests (automatic and0/1/2/3), both endpoints:80
direct full-chart outputs. Canonical Mel may be reused after exact identity
verification; each model still encodes full audio. Source timing, rows, annotation
and LN tails never enter generation. Compare requested and realized standardized
descriptors on the32 explicit requests per arm, uniformly weighting audio and
request. Report each dimension, per-audio errors and centered finite response
matrices using the existing four-profile full-rank procedure. Empty/capped charts
are failures, not omitted metric rows. Record prior choices separately.

Primary positive signal: routed total descriptor squared error <=.85 times
matched shared error, and width-component error <=.85 times shared. Guard H/LN
component errors <=1.10 times shared. Mechanism signal: absolute finite
density-request-to-width coefficient <=.70 times shared, retaining at least
.80 of a positive shared own-width coefficient. A zero/negative shared
own-width coefficient makes that mechanism comparison unresolved. Report
centered control gain so a constant output cannot masquerade as control.
These thresholds guide the research branch, not playability certification.

For FoolMoon, Hysteric, Revenge and AsItWas, repeat automatic and requests1/2
through unchanged8s/20ms/four-attempt screening:24 additional outputs. All must
complete/reparse, preserve their own direct H sequence, and have zero strictHH
and experimentalRH. Compare raw defects separately. Require cached-Mel30-row/
8-s readiness and accepted-window service <=2s, and total routed direct sampler
time <=2x shared. Report all retry/composition changes and honest capped prefixes;
no increase in attempt budget or quality fallback inside the comparison.

Lens scope: reuse the same verified human references; inspect corresponding
screened musical contexts for all24 selected outputs (merging source-context
overlaps), automatic densest8s windows, and every direct strictHH witness from
the80 outputs with1s entry/exit context. Read all rendered time pages and full
action/articulation tables in the frozen resulting plan; reuse only exact
semantic identities. Evaluate repeated grips, fine cross-column timing,
independent overlapping LN roles and sustained occupancy, not count-derived
style labels. A control gain with structural loss is not a positive result.
No claim of listening/player quality, dump coverage or unreviewed styles.

Execution via uv run --extra mps python artifacts/joint-audio/20260925-profile-routing-v1/fit.py
preflight|main, dispatching the packaged planned_audio_continuation.hydra runner;
native.py preflight|main dispatches model-backed generation. Pin script hashes,
clean implementation/Note OIDs, exact commands and input hashes in new phase
freeze files before each run. Likely product owners: model/config/training,
interval and session queries, profile tests and shared-profile documentation.
No unrelated change may enter the implementation-to-baseline diff.

Apple M5/24GiB, Python3.10.20/Torch2.11.0, MPS fitting and one CPU thread;
CPU native generation. Bounds: serial fitting at most5400s in total, native
drivers at most1200s each and90s/30000rows per chart,2GiB new artifacts, existing
2GiB available-memory/40GiB-free-disk/PAUSE guards. New run names
profile-routing-{shared,routed}-{preflight,main}-v1 under the existing data root;
fresh owner artifacts/joint-audio/20260925-profile-routing-v1. No overwrite or
resume. Poll the actual live session; never restart from an observation timeout.

If matched training alone explains gains, retain that simpler explanation.
If routing improves calibration but loses LN organization, refine the conditional
independence approximation. If both remain poor, attribute neither capacity nor
data sufficiency from this test alone. Confounders include one training seed,
one native seed per audio, finite representative quantization/preview, correlated
TRAIN attributes and curated evaluation songs. Recommendation remains REFINE
until execution and declared review; Note acceptance and adoption are absent.


### Density-routing implementation and preflight dispatch

Card profile-density-routing-v1 revision 1 remains proposed, accepted none.
Clean intervention a2bce683642995627c17b154d5f83e8d774787f0 changes only the seven
declared model/config/interval/session/training/test/documentation paths against
baseline e28435f10cfb154acfe8eb9f0507d2a6c242ff79. It adds no parameter and preserves
the true-mode default. The generation encoder now returns raw audio and the
selected profile; session queries explicitly form the two declared views. A
same-bank profiled warm start copies the learned prior after bank validation,
without resetting it. No data, optimizer schedule, native screen or row support
changes. The baseline input tensors remain pinned by SHA.

Focused verification: profile tests 9 passed in4.43s, including CPU/MPS routed
gradient/score invariance, full-held waits, crop/native parity, fixed-H native
invariance and actual runner warm-start prior preservation. Distribution,
release conditioning, training, buffering, inference and packaging tests:
47 passed plus22subtests in5.32s. Staged git diff --check passed. No unrelated
suite or CUDA coverage is claimed. Local product commit only.

Artifact driver artifacts/joint-audio/20260925-profile-routing-v1/fit.py dispatches
the packaged runner with the exact Card settings. It will create fresh phase
freezes, record this committed Note OID and compare consumed counts across arms.
Preflight is32updates per arm; main remains gated by the subsequent integrity
native result. No fit or native output is claimed at this dispatch. Standing
local run authority continues; no approval or lifecycle transition is inferred.


### Result Log: density-routing preflight and main dispatch

Card profile-density-routing-v1 revision 1 remains proposed, accepted none.
Clean source a2bce683642995627c17b154d5f83e8d774787f0. Preflight fit session25677
and native session41128 are terminal-success. Both arms completed32updates,
128intervals,980652sampled ms,6720event rows,6300H rows,420release rows and
332688release clocks. Their exact protocol hash is
8f8bdd441dc4ba36452ca50915773729eb2d7cb9613affbf51a34eaf678af3c5.
Every tracked module changed with finite gradients; all source tensors were
copied, with no new tensor or reset learned prior. Shared took38.692s, routed
35.442s. Checkpoints respectively
bb64685b7ec28de7a35a3184e4d9b5625de84c4c6f79b43b6a5e80b01a6f7339 and
b2c7124b90d38e81770f4806f21ec0057f24c20f0899d2c502d9f76ebfec0440.

The original true-mode model exactly reproduces the saved full automatic
FoolMoon rows. Switching only its routing mode preserves the entire H stream.
All eight trained preflight outputs complete/reparse at the synthetic30s end,
while reading full original audio. This short-end test is not evidence about
full-song tails or quality. Shared Hysteric/profile2 has one strictHH: lane3
TAPs at19236 and19253ms, gap17ms. Other preflight charts have no strictHH/RH.
The recorded defect does not change the integrity gate or trigger a support
patch. No qualitative improvement is claimed from these eight draws.

Owner artifacts/joint-audio/20260925-profile-routing-v1. Fit script
810c9370f3c9a8d988d0c69c6b5b0b6090fe5cfaaebeb8a0c7d1372f56599b08;
native script e27cd0404d5ba44c2e0697671a2d929189b56ba67d02df9029998cbec787465a.
Fit freeze a552bbb1b0bfa1da593f1edeb2297f6e8597ab24e8bc56ba3c7508198324b109;
fit result6dd9fff37eee361492e74985e747f4e8a71cf66ca8b45f2e8a1a8d1d8ed28b40;
native freeze86b87d6956ac45f505649eb4b4bc5401803f16cdac4a4a04d73146df4d5702f4;
native result66c824c07bb3f0a9107414fac137da5d4fb62a39785b2997913caaf5e406fab6;
parity result6cdb0f0a8fbb2136a2a3d7208db584abd58ab7458d41d01d2d920295643bb747.
Native driver11.091s; new artifacts104669156bytes including both training
directories. All integrity checks pass; no Card field, data or model intervention
changed. Recommendation remainsREFINE until fixed main endpoints and inspection.

Proceed with the declared serial1200-update main fits, both restarting from
abc27f1d192869419e42729a6b9fcdfd1c507fd672e12c9a9c7c868a5082a2ef rather than
either preflight endpoint. The launcher may use caffeinate -i solely to prevent
idle sleep; packaged model commands and all scientific settings stay unchanged.
Record and poll the actual returned process handle. After both terminal fits,
run the frozen main native comparison and complete the declared Lens scope.
No main training or quality result is claimed in this dispatch record.


### Density-routing live fit, witness inspection and TRAIN exposure

Main fit session44544 is the live serial driver, launched with caffeinate -i.
Re-poll that actual handle before interpreting any saved progress file; never
restart from an observation timeout. Shared is terminal-success at1200updates,
906.859s, checkpoint7849e07a77febf12968b8197e844d9c26f8327ce789d0776117e02dc83de788f.
The same driver has started routed; last verified snapshot is150updates at
145.467s,5.166GB available RAM. No main native result exists yet. Main freeze
SHA f2dd8eebad4782c259d8a31f0daf340637ecdb8cbaf7f3999a1437eed3fdf2a2.
The source remains clean a2bce683642995627c17b154d5f83e8d774787f0. Keep it clean
while the serial runner is active; no weight or training-setting change.

Supplementary preflight Lens inspection is complete: one time page and all19
action/articulation rows in[18236,20253)ms, including entry state. Review owner
artifacts/joint-audio/20260925-profile-routing-v1/preflight-lens/review.json,
SHA a898536de2b0a823dd4f1e41784011220d3da86b54d660019735c22851327eb8.
There are no LNs or releases in this scope. Row19236 taps0/1/3; row19253 taps2/3.
Column2 is the only column not attacked within20ms, so a single head there is
locally feasible. The chosen double repeats column3 after17ms. This identifies
row choice under a feasible two-H sequence, not an LN-release failure or a
need to remove the H gap itself. Neighboring repeated1/2 grips120ms apart remain
separate from that failure. No style or listening/player verdict is inferred.

Two post-hoc descriptive audits clarify data limitations without changing the
Card's primary metric, thresholds, training plan or model. They use frozen
TRAIN profile assignments and the already-frozen exposure plan, never generated
outcomes to choose new settings. Exact prototype membership is sparse:

| Profile | TRAIN charts / groups | Prototype H/s, width, LN fraction | Weighted source mean H/s, width, LN fraction | Main sampled intervals |
| --- | --- | --- | --- | --- |
| 0 | 110 / 94 | 4.749,1.370,.114 | 4.891,1.369,.131 | 785 |
| 1 | 12 / 10 | 7.032,2.141,0 | 7.376,2.178,.0047 | 93 |
| 2 | 12 / 11 | 6.060,1.429,.734 | 6.607,1.454,.689 | 134 |
| 3 | 13 / 9 | 13.548,1.286,.0004 | 12.881,1.274,.0040 | 125 |

The full main plan has4800intervals. The first32updates had zero profile1
intervals and only two profile2 intervals from one chart, reinforcing that
preflight was an integrity check rather than a control-quality test. Across
all615 TRAIN charts,25charts/22groups have raw LN fraction at least.5. That
cutoff is descriptive, not a new style label or sampling rule. Other profiles
also supervise each dimension through the shared continuous projection; exact
class counts are not the complete supervision for LN or width.

The16-class, naturally weighted standardized quantization error is.56324.
Within-class source variation and centroid-to-representative bias are separately
recorded. This marginal TRAIN variation is not an irreducible generated-error
floor conditional on audio and cannot be directly compared to equally weighted
native requests without reweighting. A prototype is not an exact target shared
by every chart assigned to it. These facts support caution in attributing poor
control solely to architecture; they do not invalidate the frozen comparison.

Audit owners profile-target-audit.json (SHA85c249243db4328c37b050acfa0f92cb2212a2261e5c51c58198b71181a53d2f)
and exposure-audit.json (SHA09fd49265259c6862c4bc7462d777f83fa889d8da396f6b1e896704a1125bb82)
under artifacts/joint-audio/20260925-profile-routing-v1. Future balanced attribute
exposure or added source diversity is a live alternative if matched fits fail;
neither is introduced inside this run. No claim of an insufficient audio encoder
or need for larger parameter count follows from these counts.

After session44544 terminates successfully, verify main-fit-result.json and both
checkpoint/protocol hashes, then run uv run --extra mps python
artifacts/joint-audio/20260925-profile-routing-v1/native.py main. The driver is
already preflight-exercised and will generate the declared80direct plus24screened
charts, retain every failure and compute the frozen control comparison.
Then run inspect_lens.py main in the same owner and read every new page/table.
That driver has been exercised on the supplementary witness and now additionally
checks the parent plan against its completed review hash. Current script SHA
6a1b80375d5b3536f64c6e3551d4d8b52ceac2d9ebe49663aeeba5600d2dd9fb.
This is a behavior-neutral identity assertion; it does not change scope or
generation. Main native/review remain outstanding, as does model-quality adoption.
The previous turn and this turn both make concrete progress; the ultimate goal
remains active, with no blocker or requested pause and no remote publication.


### Result Log: density-routing main terminal result and partial Lens review

Card profile-density-routing-v1 revision 1 remains proposed, accepted none.
Clean implementation a2bce683642995627c17b154d5f83e8d774787f0. Main fit44544,
native95936 and Lens render3358 are all terminal-success. No job remains live.
Do not rerun fitting, generation, preparation or rendering to resume inspection.
The prior goal turn implemented and dispatched work; this turn verified its
completion, produced the fixed comparison and added mechanism evidence. Both
are progress, not a repeated wait or blocker.

#### Reproduction and quantitative results

Both main arms complete1200updates with the same4800intervals,37396141sampled
milliseconds,254883event rows,240283H rows,14600release rows and10678531release
clocks. Protocol8f8bdd441dc4ba36452ca50915773729eb2d7cb9613affbf51a34eaf678af3c5
is identical. Shared fitting906.859s, routed1041.464s; checkpoints:
shared7849e07a77febf12968b8197e844d9c26f8327ce789d0776117e02dc83de788f,
routed d2d943a8bb565559480121645128c70824d792b9830c013981780bbfb74a3f22.
All tracked modules change, inputs and bank stay fixed, and both models have
4250174parameters. No NLL endpoint selection or protected-field change occurs.
Reference-profile VAL conditional NLL/sec is40.1827 shared versus40.3493 routed;
these similar proxy values do not predict the following native difference.

Main owner artifacts/joint-audio/20260925-profile-routing-v1. Fit result
81180e0eac84541e11ad221c879940309b39d0cf82199303f41e84c3e80e77a3;
native freeze0ff9aa80aec5c0c38871f6a2da0d9494c6a646dabb1783a9c398f354c4bea4bf;
native result0097d705919e6203a000906cef464337f039d52b861c63ae561a9ece5f57d9d6.
All104 charts complete and independently reparse:80direct (eight audios,
automatic plus four explicit requests, two arms) and24screened (the four named
audios, automatic plus requests1/2, two arms). Driver609.107s, within1200s.
Native/fit artifacts350322694bytes before Lens; all artifacts490973895bytes
after rendering, below2GiB. Weights are unchanged throughout native generation.

Primary control comparison uses32 explicit charts per arm, uniformly over eight
audios and four requests. Squared standardized descriptor distance:

| Metric | Shared continuation | Routed continuation |
| --- | ---: | ---: |
| H-rate component MSE | 1.994592 | 3.610711 |
| Width component MSE | 2.257701 | 3.930808 |
| LN-fraction component MSE | 1.944378 | 3.000517 |
| Total | 6.196672 | 10.542035 |
| Centered H / width / LN control gain | .355 / .309 / .493 | .012 / .229 / .273 |
| Finite density-request to width coefficient | -.06638 | -.11051 |
| Finite own-width coefficient | .20757 | .10697 |

Routed error is higher on every one of the eight audios. The primary15% reduction,
H/LN non-regression guards and mechanism criterion all fail. Own-width response
approximately halves while the magnitude of density-to-width response grows.
The previous positive .49060 coefficient came from a different checkpoint/cohort;
do not treat it as this matched continuation's measured baseline.

A separate, exactly matched requests1/2 subset compares the initial checkpoint
with both final endpoints on the same eight audios/seeds. Total errors are
7.340423 initial,9.191414 shared,15.914283 routed. Component errors respectively
[.873858,2.370422,4.096142], [1.763494,3.983159,3.444761] and
[3.351010,6.819452,5.743821]. Thus ordinary continued fitting also worsens total
calibration on that matched subset, although its LN component improves. This
subset is contextual evidence, not a replacement primary metric. Owner
matched-initial-comparison.json, SHA86f1e66cafb722350c73253130792c656b0aba94ebf0b6b9fcc7afd87729e857.

Direct outputs contain8HH/10RH pairs shared and38HH/12RH routed. These are counts
over declared calls, including automatic/explicit exact duplicates, not80
independent samples. The screened subset's paired direct counts are only
1HH/2RH shared and13HH/3RH routed; all24 screened outputs have0/0 and identical
paired H sequences. Do not claim that screening removed all46 direct HH pairs
across the full80-chart cohort. Shared/routed need4/21 rejected proposals.
Maximum cached-Mel readiness is.493/1.120s; maximum accepted-window service
.813/1.375s. All specified screened system guards pass. Total direct sampler
time212.360/258.117s passes the2x guard. These are not fresh-process client or
listening measurements.

#### Post-hoc interpretation checks

assessment-audit.json verifies identity response, zero centered gain for a
constant output, explicit incomplete-chart failure and separate strictHH/RH
thresholds including LN heads. SHAde618d3fa8063b329aaf3575248c3cd31b3b7bc8956e8b292d1f8360fb58f26e.
No data, generator, frozen metric or threshold changes follow from that check.

ln-target-audit.json (SHA544dd30d37164190feaaa12aba758578148a746ffc656728fd8dc6b88c2cb967)
reads all twelve profile2 TRAIN targets with verified row hashes. Every target
contains overlapping holds, starts under existing holds and subset releases.
Time with at least two columns held ranges.233..736 of decoded duration;
the fraction of LNs spanning at least one interior H ranges.163..533.
This proves physical source support, not annotated salience, musical quality
or sufficiency of the small corpus. Simple relay output cannot be explained by
claiming the target representation contains no overlap.

hh-support-audit.json (SHA87c2c36bd7aae2b40166067bdd59b6f59129463989a577b38ffcca9553a66397)
separates H-time feasibility from actual prefix/row decisions. None of the80
direct plans has five H within a strictly sub20ms span. For one TAP per H in4K,
this condition is necessary by pigeonhole and sufficient by cyclic lane
assignment. It is only an existence proof for the strictHH constraint, not a
proposed chart or proof of feasibility with requested chords/LNs/RH preference.

The8 shared HH pairs occur in7rows: four rows have enough rested free columns
for their chosen head count, one exceeds that capacity, two have none. The38
routed pairs occur in36rows:25 have enough columns, eight exceed capacity,
three have none. Of the five no-rested-column rows across arms, four follow
recent taps/chords with no held LN; one shared Zenithfall row keeps lane1 held
while the other three were attacked12ms earlier. These classifications condition
on the actual prefix; changing one row can change later possibilities. They do
not establish how much of total quality is attributable to original R1 weights.

#### Lens status and provisional interpretation

Frozen main plan b68fbeae52c71a5075fab9b8e53596abf2046c7e533a4e9156ab5fa500afb10f
has91 scopes:84new scopes/180pages, six exact previously reviewed reference
scopes and one exact current reuse. Render result
1c43805a718ac514a6e0294bebffa017d513c880c0778d5622de43448fec7de4.
All16 human records were retrieved unchanged. Rendering is complete; inspection
is not. Actually viewed32pages/19new scopes, with complete action/articulation
tables:0,2..11,52..55,87..90. Current reuse1 is covered by0; parent references
56..61 are verified against completed parent review9302f23daed2ce010465437c1a3d90793754403035ac447121619fe91da5e737.
Remaining65new scopes/148pages must be read before a final cohort review.
Inventory and all observations are in review-progress.json; no final review exists.

The completed Revenge comparison shows a concrete structural difference.
Shared profile2 retains many independently staggered LN starts/tails, some
spanning other H, including a564ms hold over three H. Its13ms cross-column LN
starts survive screening. Routed profile2 at the same context has12 single
taps and no LN. Both wide-profile contexts instead contain only nine single
taps. Automatic variants retain isolated/relay holds, including routed holds
lasting1840/2380ms under other-column taps, without simultaneous independent
holds there. These observations do not assign a confident Tech label or claim
audio alignment; source tags do not require copying one exact arrangement.

Read HH witnesses distinguish layout from temporal/resource choices. Examples:
FoolMoon's16ms repeat has a free lane last attacked22ms earlier; other episodes
choose two/three heads when only one/two rested columns remain. Goodbye has a
TAP1 to LN_START1 after14ms. Its HH-rested columns2/3 were released only14ms
earlier, so a local substitute there would exchange HH for the experimental RH
problem; an earlier chord/release choice is needed to satisfy both. Such evidence
supports joint continuation checks without imposing a generic onset-spacing
filter or inventing a universal LN-duration floor.

Recommendation remainsREFINE. The routed endpoint fails the declared numerical
criteria and is not adopted; the shared endpoint is not automatically promoted
either. Similar teacher-forced NLL does not establish stable native control.
Removing an input at fixed weights did pass its local noninterference test, but
joint fitting still changes shared audio/condition gradients and H-model weights.
It therefore does not imply unchanged H plans after training. Actual H rates
change substantially across trained arms. Their separation into audio base,
head-history dynamics and materialization remains unmeasured; do not present
a self-excitation or R1-only explanation as confirmed.

Immediate action is to continue the declared Lens plan at the saved inventory,
then publish a self-contained result and choose the next discriminating question.
No next architecture, added hard constraint, larger model or new fit has been
selected from partial inspection. Existing checkpointabc27 and the default true
routing mode remain available unchanged. No test rerun, Note lifecycle change,
model adoption, remote publication or goal completion is implied.


### Result Log: density-routing completed inspection and evaluation

Card profile-density-routing-v1 revision 1 remains proposed, accepted revision
none. Standing local research/execution and persistence authority applies.
Implementation/execution source remains a2bce683642995627c17b154d5f83e8d774787f0.
Curated report and its shared-profile link are committed at product
9e6dbc61511ec9b1a45bc55148a344886e81d682 in
docs/research/profile_density_routing_evaluation.md and
docs/research/shared_arrangement_profiles.md. The report preserves its argument
without requiring this Note or local artifacts.

This completes the previously partial main review. All180 new time pages across
84new scopes were actually viewed, with complete native-ms action and LN
articulation tables. Six exact source-reference contexts reuse the fully
reviewed parent, and scope1 reuses current scope0 by identical actions, entering
holds and articulation. All91 planned scopes are covered. All16 human records
remain unchanged; source tags are scoped references, not required copies.
No listening, full-song visual review or player test is claimed.

Artifact owner artifacts/joint-audio/20260925-profile-routing-v1. Final
review.json SHA4e4d074e4f02b63b05d8d2d710700adf440e243aa5e036e5a1e244684967d659
contains the full frozen scope plan, actual per-scope page inventory, complete
table coverage, observations and limits. Main plan SHA remains
b68fbeae52c71a5075fab9b8e53596abf2046c7e533a4e9156ab5fa500afb10f;
main-native-result.json remains
0097d705919e6203a000906cef464337f039d52b861c63ae561a9ece5f57d9d6.
Its original qualitative_review=pending is a historical dispatch state; the
new review identity supplies completion rather than rewriting native evidence.
No training, generation, rendering or already-passing test was repeated.
No process remains live.

The negative routing result survives the complete review. Primary explicit
descriptor error is6.196672 shared versus10.542035 routed, higher on every audio
in the routed arm. Total/width improvement, H/LN bounds and mechanism criteria
all fail. Conditional VAL NLL/sec40.1827 versus40.3493 does not describe this
native difference. On the exactly matched original requests1/2 subset, total
errors7.340423 initial,9.191414 shared,15.914283 routed also defeat a claim that
ordinary further fitting is already a better endpoint.

The screen's paired subset remains distinct from the full direct cohort:
shared1HH/2RH and routed13HH/3RH become0/0 in all24 selected screened outputs,
with exact paired H preservation and4/21 rejected proposals. All104 native
outputs complete and independently reparse. Maximum cached-Mel readiness
.493/1.120s and window service.813/1.375s pass2s bounds; total direct sampler
212.360/258.117s passes2x. These exclude fresh waveform processing/process
startup and do not measure client/network latency.

#### Completed musical and failure evidence

Under explicit profile2, screened whole-chart H/s and LN fractions are:
FoolMoon3.661/.167 shared versus13.639/.042 routed;
Hysteric14.674/.101 versus16.209/.039;
Revenge2.920/.672 versus1.710/.184;
AsItWas6.001/.465 versus7.340/.032. Requested values are6.060H/s,
1.429heads/H and.734LN fraction. Local absence is interpreted alongside these
chart-level measurements, never as a standalone failure of a global request.

Shared FoolMoon scope78 has an entering6458ms LN3 supporting taps, overlapping
new holds and subset releases; routed scope43 has dense single TAP movement
and one51ms LN. Shared Hysteric scopes85/86 retain staggered multi-lane holds:
at243545 LN1 joins2/3 and survives changes in the other lanes, including common
0/3 release while1remains. Routed scopes50/51 are entirely TAP. Shared Revenge
scope90 has independently staggered holds, including564ms LN2 spanning3H;
routed scope55 has12single TAP rows. Shared AsItWas scope72 retains paired and
staggered holds with other-column taps, while routed scope38 has fewer such
relations, though436/845ms holds each span5H and a staggered pair still occurs.

Counterexamples prevent a total-capacity-loss claim. Routed automatic Hysteric
scope45 contains many staggered overlapping LNs and subset releases, including
374ms LN1 spanning5H. Routed AsItWas/profile1 scope37 has repeated outer doubles
and changing groups. Shared AsItWas/profile1 scope71 has recurring triples and
a late quad; automatic scope69 has substantial independent LN organization.
Both AsItWas automatic arms select profile13. All four selected automatic
audio pairs retain the same chosen profile identity across arms, so their
differences are not explained by a changed sampled profile index.

Fine cross-column timing survives. Routed Hysteric/profile2 at250802 has0/3,
then1at250809 and2at250819: four heads in17ms without a repeated lane. Numerous
1–19ms cross-column splits remain elsewhere. Isolated12–43ms LN durations are
separate articulation observations. Exactly20ms HH is excluded from the human
strict<20 criterion, but neither>=20 nor HH0 certifies comfort. No generic
onset-spacing rule, anti-Jack rule or universal LN-duration floor is inferred.

All direct HH witness contexts were reviewed. Across43bad rows,29have enough
rested free columns for their chosen head count,9exceed currently rested free
capacity and5have none. Four no-rested cases follow tap/chord allocation; the
fifth also has LN occupancy. The existence of one-TAP-per-H HH-safe assignment
for all80plans is only a weak temporal-support result, not a proposed good
chart or proof of simultaneous chord/LN/RH feasibility. Goodbye scope9 shows
why an HH-rested local substitute can still create a release-to-head conflict.
These classifications are conditional on actual sampled prefixes and cannot
attribute a fraction of overall failure to restored R1.

#### Evaluation, uncertainty and next question

Recommendation: REFINE. The input-removal hypothesis fails as an improvement
on this fixed cohort. Neither trained endpoint is adopted; the initialabc27
checkpoint and default true routing remain available. The implemented optional
flag retains reproducibility, without claiming a better policy. Card/Note
status remains proposed; no human acceptance or lifecycle transition occurred.

Confirmed information-flow distinction: runtime H does not consume tap layout
or row embeddings, while joint training can change shared audio/projection
parameters through R/row gradients. Fixed-score downstream invariance to the
removed coordinate therefore does not imply stable learned H. Source-history
NLL, bounded historical logit correction and legal-row replay supply different
properties; none is a native density, style or HH invariant. Frontier2 still
uses earliest possible release, not actual R forecast; the screen evaluates
actual unpublished joint continuations but only enforces its declared checks.

Remaining explanations include conditional-information loss under finite H
preview, changed shared representations/gradients, sampled-history amplification,
and limited/correlated training exposure. The raw data does contain overlapping
LN relations. Profiles1/2 each have12TRAIN charts and93/134main intervals;
the32-update preflight sees zero profile1 and only2profile2 intervals. Shared
continuous conditioning means this is not all relevant supervision. None of
these descriptive counts proves data sufficiency or identifies the dominant
cause; scaling or rebalancing is not selected on this evidence alone.

The next discriminating question is the entry point of training-induced drift:
full-audio H base, bounded sampled-history correction, or R/row response to the
changed H spacing. First inspect saved-trajectory factor contributions and
design a bounded fixed-input comparison at the original/shared/routed endpoints.
Preserve full audio and actual permitted state; cross-model conditions, if
used, are diagnostic and may be outside training support. Do not re-run the
already completed routing comparison or jump directly to another larger fit.
No next intervention or Experiment Card is accepted or executed by this log.

Verification for the curated report: numerical claims checked against frozen
results; exact result/review hashes and180page/84scope coverage verified;
relative Markdown links resolve; staged git diff --check passes. Product diff
is only the282-line documentation addition/link. The registered orphan notes
worktree and Markdown-only allowlist were verified; local commits only, no
remote push. The ultimate playable-system goal remains active.


### Experiment Card: head-factor-drift-v1

Revision 1; proposed; accepted revision none. Owner is this Note. Standing
local research, implementation and execution authority applies. Previous turn
completed180 Lens pages, persisted the negative routing result and changed the
next action; it was progress. Product baseline is clean
9e6dbc61511ec9b1a45bc55148a344886e81d682. No fit or full-chart generation is live.

Question: where does further fitting change native H counts: the audio/profile
base, the learned residual path at fixed history, or the history input induced
by the new trajectory? The purpose is to choose the next structural/training
intervention, not to optimize another count proxy or promote a crossed model.

Closest analogues: Brown et al. time-rescaling/counting-process analysis
(https://www.stat.cmu.edu/~kass/papers/rescaling.pdf); Haslinger et al. discrete
time correction (https://pubmed.ncbi.nlm.nih.gov/20608868/); Neural Hawkes event
excitation/inhibition (https://arxiv.org/abs/1612.09328). Transfer the distinction
between exogenous information, history modulation and the counting process.
Ensomi has a discrete native-ms Bernoulli law, an explicit additive logit base,
full-song audio and self-generated expressive targets. No continuous-time KS
claim, exponential-ISI assumption or Hawkes stability theorem is imported.
This is a diagnostic adaptation, not a new model or objective.

Branches: base-path drift predicts most fixed-history count-propensity change
comes from exchanging the audio base; learned-residual drift predicts the
opposite; history feedback predicts larger changes when replaying the other
endpoint's H sequence with weights fixed. Shared-gradient changes may affect
both base and residual paths; the residual also reads audio, so its name must
not be misreported as a pure head-TCN attribution. Mixed results preserve these
alternatives. Further blind fitting, bigger encoders and hard H-spacing changes
are deferred because they do not distinguish these mechanisms.

Use all eight previously frozen panel audios and explicit requests1/2, same
seeds251701..251708. Initial checkpointabc27f1d192869419e42729a6b9fcdfd1c507fd672e12c9a9c7c868a5082a2ef;
shared7849e07a77febf12968b8197e844d9c26f8327ce789d0776117e02dc83de788f;
routed d2d943a8bb565559480121645128c70824d792b9830c013981780bbfb74a3f22.
Paths stay under the existing20260924-alias-restored-v1/planned-training owner.
Bank a8973b7f94fde90d9f3cc639eb63e781dd295435622ce79fd54fe244789e6f03,
panel484d173628856a431e034b1dbdacab9173c40f3edf696d1b9a19721aeec70dc9.
Source native result owners are20260925-fresh-audio-system-v1/result.json
(9f3c4580adaa6239c44e04bb5d95a0ed921b8fd958b1eee26ccd8cedfa91e862)
and20260925-profile-routing-v1/main-native-result.json
(0097d705919e6203a000906cef464337f039d52b861c63ae561a9ece5f57d9d6).
Read only their48direct rows.jsonl streams after exact hash verification.
No source chart, redline, annotation or future LN tail enters H queries.

Baseline on16exactly matched requests is H-component descriptor error.873858
initial,1.763494 shared and3.351010 routed. For concrete count scale, initial
FoolMoon requests1/2 have1499/1078H; shared1045/1014 and routed4319/3778.
Initial Hysteric has2170/1297H; shared3922/4417 and routed5013/4879. These are
fixed sampled trajectories, not independent population estimates. Every case,
including improvements, is retained in the diagnostic.

One causal intervention exchanges only the additive H base logit between the
initial and a continued endpoint, holding residual output and the legal H
history fixed. Evaluate both exchange directions on both endpoints' actual H
histories. For each history h and base/residual source a,b, define
C_ab(h)=sum_t sigmoid(base_a(t)+residual_b(t|h_<t)) over t=0..duration inclusive.
Each source computes its own full-audio encoding and own head-history encoder;
only their scalar logit outputs are combined. These are fixed-history
conditional-probability sums, not expected counts of the freely evolving
crossed generator. No weights are copied between incompatible hidden spaces.

For initial0 and continued1, calculate on each h:
B(h)=.5[(C10-C00)+(C11-C01)],
R(h)=.5[(C01-C00)+(C11-C10)]. Average B and R over h0,h1.
History contribution D=.5[(C00(h1)-C00(h0))+(C11(h1)-C11(h0))].
Realization remainder E=(N1-C11(h1))-(N0-C00(h0)). Then
N1-N0=B+R+D+E exactly, within numerical tolerance. This symmetric accounting
reduces arbitrary path-order dependence; it is still conditional on two sampled
histories and cannot identify an intervention's native quality or root causes
inside a neural path.

Behavior-neutral readouts: per8s and whole-audio sums; own-path Bernoulli
variance sum p(1-p); mean/quantiles of base, residual and recency gate; and
sum sigmoid(base), the exact expected count for the history-free base law.
Distinguish this probability sum from sum softplus(logit), which governs
survival. No NLL selection or continuous-time goodness-of-fit significance.
Crossed base-only counts are not proposed skeletons and receive no style label.

Primary diagnostic: for each continued arm, compare |B|,|R|,|D| on cases with
|N1-N0|>=.1*N0. A component is a dominant lead only if its median share of
|B|+|R|+|D| is>=.60, it matches the signed count change in at least75% of eligible
cases, and median|E|/|N1-N0|<=.25. Fewer than4eligible cases, inconsistent signs
or no qualifying component means mixed/unresolved. Report all case values,
rates and absolute accounting, even outside this descriptive threshold.
Positive base lead focuses audio/profile calibration; residual lead focuses
learned clock/history modulation; history lead focuses autoregressive exposure.
No branch establishes sufficiency of a remedy, R1 fault share or playability.

Integrity checks before interpreting: verify all source/checkpoint/Mel/row/H
hashes; pure HeadPlanner replay must reproduce every saved H sequence exactly
with original seed xor0x17AB and500ms chunks. Dense-history and online-cache
reads agree at16uniformly spaced head positions per case with atol/rtol2e-5.
Native clock coverage uses canonical bin anchor10*floor(t/10)+9 but only heads
strictly before native t. Check BOS, H at0, multiple H in a10ms bin and final
audio clock with synthetic partition assertions. Every clock occurs exactly
once, every part finite, |residual|<=4*gate+1e-5, base+residual matches the
existing head_logits interface. Accounting error<=1e-6 heads and per8s sum
equals whole-audio sum within1e-6. Parameters and fixed inputs remain unchanged.
Fail on any integrity discrepancy, with partial diagnostic output preserved.

Qualitative evidence is the completed source trajectories' Lens review, not a
new chart produced here. It already distinguishes dense TAP flow, overlapping
LN relations and bad HH. This diagnostic cannot declare a musical improvement.
R/row fixed-plan interventions are deferred until H attribution is measured.

Execution: uv run --extra mps python
artifacts/joint-audio/20260925-head-factor-drift-v1/analyze.py.
One CPU thread on Apple M5/24GiB, Python3.10.20/Torch2.11.0; no optimizer or
network in the run. Fresh owner20260925-head-factor-drift-v1, max1200s total,
75s per H replay, at most30000H,256MiB new artifacts. Standard2GiB available
memory/40GiB free disk/PAUSE guards apply. Checksum the diagnostic script and
clean product/committed Note before writing a one-time freeze. No overwrite,
resume or automatic rerun on an observation timeout. Stop on nonfinite values,
identity mismatch, guard failure or exhausted cap. No product behavior changes
are required; instrumentation is a bounded derivative script with pinned bytes.
Recommendation remains REFINE until interpretation; no acceptance or adoption
is implied by execution authority.


### Head-factor diagnostic revision 2 and input-path correction

head-factor-drift-v1 revision2 remains proposed, accepted none. Attempt1 session
93089 is terminal failure before any factor measurement: the diagnostic assumed
rows.jsonl beside every result.json, while the original packaged audio-file
entrypoint stores its exact row export at chart/rows.jsonl. All48 intended row
streams exist and match the already-pinned rows_sha256 values after resolving
that owner-specific path. No source trajectory or baseline identity changes.
The failed analyze.py, freeze.json and failure.json remain preserved under
20260925-head-factor-drift-v1; no run is live and no result is inferred from it.

Revision2 changes only the original export path resolution and the fresh
attempt destination/procedure: uv run --extra mps python
artifacts/joint-audio/20260925-head-factor-drift-v1/analyze_v2.py,
outputs under20260925-head-factor-drift-v1/attempt2. Every other revision1 field,
equation, dataset/checkpoint, metric, threshold, guard and compute bound remains
unchanged. Resolve and verify all48 row paths before loading model-backed
queries. Preserve attempt1 source bytes; checksum the corrected script in the
new freeze. This is a declared procedural correction, not permission to change
the scientific comparison or silently overwrite a failed run.


### Result Log: head-factor-drift-v1 revision 2

Proposed, accepted none. Clean execution source9e6dbc61511ec9b1a45bc55148a344886e81d682;
execution Note e5c5dfc8aff62d96481d5fcc91d3222684309529. Command is the revision2
analyze_v2.py invocation above. Session17172 is terminal success in74.945380s,
with32paired endpoint comparisons and48own trajectories, one CPU thread.
All48H sequences are exactly reproduced and768selected dense/cache positions
pass2e-5 tolerances. Native partitions and all accounting identities pass1e-6H;
parameters/checkpoint/input identities remain unchanged. No fit or new row
chart was produced. The input-path failure in attempt1 is separately preserved;
no other protected field changed and no run remains live.

Owner artifacts/joint-audio/20260925-head-factor-drift-v1/attempt2.
Result4aaa7036b335c90ec75af4128f9e1ad3e9dc5031420b4a3fe0d73a819b74b1e9;
freeze301530989f1468e9f31b7667eafc1b27275d2079e1e23b6fe03d4d2dbabe5c9d;
script c99084123613c161d8f5f0cc038bcf9aed6efd4bdd6c7bc1e9e44979f2ff9b97.
Run bytes1351401 before the supplementary scientific figure. plot.py creates
head-factor-drift.png/.svg from the frozen result; the complete two-panel
figure was visually checked with common horizontal scales and labelled units.

Neither arm has a component meeting the declared dominance rule. Shared has
13eligible cases: median absolute B/R/D shares.218/.156/.495, Dsign agreement
10/13 and median relative realization remainder.017. Routed has15eligible:
.376/.111/.465, Dsign agreement14/15 and remainder.019. No60%median lead, so
do not rewrite the criterion to declare history the universal primary cause.
All32cases, including small changes and opposing contributions, remain recorded.

Selected profile2 accounting in heads, B/R/D/E:
shared Hysteric1297→4417:371.7/69.8/2674.6/3.9;
routed FoolMoon1078→3778:987.1/−27.6/1830.5/−90.0;
shared Take928→1258:644.9/−149.5/−161.1/−4.4;
shared Yomi2939→3590:1683.9/−10.2/−988.8/−34.0.
For Hysteric, base-only expected count changes1001.05→1089.60 while the own-path
median residual changes−1.747→+1.657. For Yomi, base-only expected count changes
1952.93→3323.68 and history partly offsets it. Conditional probability accounting
exposes feedback/coadaptation; crossed sums are not freely sampled model means,
independent causal percentages or playable outputs. The residual path also
reads audio, and the small paired remainder is not a seed-robustness result.

Recommendation REFINE. This gives evidence to avoid a universal base-only,
history-only or R1-only explanation. The bounded head correction protects a
long-wait property, not native count calibration; teacher likelihood identifies
the sum rather than an independently calibrated audio base. LN/row loss still
requires H-controlled evaluation, especially Revenge with lower density and
AsItWas with large LN loss relative to its smaller H change. No model, decoder
or threshold is adopted. Broader time-rescaled renewal/count-conditioned
representations remain hypotheses; Pillow's conditional-renewal analogue
https://proceedings.neurips.cc/paper/3740-time-rescaling-methods-for-the-estimation-and-assessment-of-non-poisson-neural-encoding-models.pdf
distinguishes real/rescaled history effects and warns that temporal-fit tests
alone do not establish stimulus-response quality. No such model is implemented.

Curated self-contained report docs/research/head_factor_drift.md and a link from
the routing report are committed atfd2ddfdbc66c1177167487085bccfbac818ddb39.
Selected documentation checks verify fixed evidence hashes, complete declared
accounting, local links and staged git diff --check. No passing model tests
were rerun for this prose-only product change. Goal remains active.


### Experiment Card: head-materializer-composition-v1

Revision1, proposed, accepted none. Owner is this Note; standing local research
and execution authority applies. Clean product baseline
fd2ddfdbc66c1177167487085bccfbac818ddb39. No training or new product architecture
is required for this bounded functional comparison.

Question: can the initial H process preserve its better calibration while the
continued shared R/row conditional supplies improved LN organization? One
intervention replaces the complete generated H plan seen by a fixed R/row model.
Compare both directions between initial I and shared continuation S:
H_I/B_S is the prospective composition; H_S/B_I is the reverse diagnostic.
B denotes release plus row materialization, each with its own audio encoding,
profile condition, LN projection, histories and frontier2. H_I/B_I and H_S/B_S
are the saved diagonal controls. No routed endpoint or synthetic profile cross
enters this comparison. No encoder activation is exchanged across checkpoints.

Closest local analogue is shared-profile-path-crossover-v1 in
docs/research/shared_arrangement_profiles.md, which holds generated H fixed while
crossing downstream profile codes. Here each entire conditional retains a
consistent profile and its own learned representation; only H source changes.
The probability factorization p(H|A,z)p(R,Y|H,A,z) already permits this functional
composition. This is an adaptation/test of modular conditional generation,
not a new probability primitive. Branches: useful complementary learning predicts
H_I/B_S improves native controls and LN relations; body degradation predicts
failure even on original H; H-sensitive materialization predicts the reverse
cross degrades with the changed plan. Shared RNG cannot keep diverging R/row
histories fixed, and crossed H/body pairs can still be outside training support.

Use all eight pinned panel audios and requests1/2, seeds251701..251708. Initial
checkpointabc27f1d192869419e42729a6b9fcdfd1c507fd672e12c9a9c7c868a5082a2ef,
shared7849e07a77febf12968b8197e844d9c26f8327ce789d0776117e02dc83de788f;
same bank a8973b7f94fde90d9f3cc639eb63e781dd295435622ce79fd54fe244789e6f03
and panel484d173628856a431e034b1dbdacab9173c40f3edf696d1b9a19721aeec70dc9.
Native source owners/hashes are the unchanged originals in head-factor-drift-v1.
Read plans from verified48-source evidence's initial/shared direct rows only,
using chart/rows.jsonl for I and rows.jsonl for S. Check H hashes/counts. The
preceding48exact pure-H replays establish their generator provenance. Native
materialization reads full canonical Mel; no chart layout, redlines, source
tail, style label or source-audio context enters a generation call.

Diagonal baseline on16explicit outputs: total standardized descriptor MSE
7.3404225633 I versus9.1914143851 S; components
[.8738583549,2.3704218356,4.0961423728] versus
[1.7634942568,3.9831588647,3.4447612636]. The primary metric compares H_I/B_S
with I on those same16 cases, uniform audio/request weights. Require total
error<=.85*7.3404225633 and LN error<=.85*4.0961423728; width guard
<=1.10*2.3704218356. H component must equal the original exactly apart from
floating reporting tolerance1e-10. This is a control signal, not playability.
Report all components, per-audio values, raw HH/RH and reverse-cross results;
retain incomplete/empty charts as failures, not dropped metric rows.

Preflight integrity: for FoolMoon and Hysteric request2, replay each diagonal
body with its own saved H plan (four calls). Require complete exact row-stream
identity and independent reparse, before the main cross. These are necessary
fixed-plan/online-preview parity checks, not additional independent quality
samples. Stop on a discrepancy; do not silently reinterpret end-of-plan flags.

Main:32direct crossed full charts, then32paired crossed screened charts using
the unchanged8s windows/20ms halo/four-proposal budget. All must complete,
independently reparse, preserve the supplied H sequence and maintain exact
incremental replay/coverage. Screened outputs require zero strict HH<20ms and
experimental RH<=20ms. TAP and LN heads both count; same-time cross-column
organization, exactly20ms HH and LN duration retain their separate meanings.
No added H spacing, minimum LN duration, attempt escalation or quality fallback.
Record cached-Mel body readiness/window service and all rejected windows.
Body-service bound2s/window; include full H-planning latency only in a later
integrated candidate test if this diagnostic passes. Do not compare cached-plan
readiness with a fresh-audio end-to-end claim.

Lens scope: verified human references and corresponding screened contexts on
FoolMoon, Hysteric, Revenge and AsItWas for both profiles and both crossings;
their densest8s H windows for each crossing/profile; all direct HH witnesses
with1s entry/exit context across32direct charts; every resampled publication
window across32screened charts. Merge overlaps per output. Read every rendered
time page and full native action/LN tables; reuse only exact actions, entering
holds and articulation from completed reviews. Assess repeated/changing grips,
fine timing and independent LN start/hold/release relations. No source tag is
mandatory for the alternative chart and no new listening or player claim.
Retain mechanical improvement with structural loss as a failed quality signal.

Positive signal: prospective composition meets primary/control/system bounds
and retains inspected organization, motivating a compact two-branch checkpoint
and integrated full-audio startup test. Negative: both crossings fail controls
or lose organization, motivating a training/representation change rather than
module recombination. Mixed result identifies per-factor tradeoffs and is
REFINE, never a declaration of the final playable system. Diagonal errors and
one seed per audio are small curated-cohort evidence, not population guarantees.

Command: uv run --extra mps python
artifacts/joint-audio/20260925-head-materializer-composition-v1/run.py.
CPU one thread, Apple M5/24GiB, Python3.10.20/Torch2.11.0. Bounds1200s native
driver,90s/30000rows per chart,2GiB new artifacts including Lens;2GiB available
memory/40GiB free disk/PAUSE guards. Fresh owner named above; no overwrite or
resume. Pin script/dependency hashes, clean source/committed Note and all inputs
before running. Stop on identity/nonfinite/replay mismatch or declared cap;
save truthful partials. No optimizer, remote publication, model adoption or
Note lifecycle transition is implied. Lens rendering/inspection follows native
completion, and no further fit starts before declared evaluation finishes.


### Result Log: composition native guard stop and first Lens evidence

Card head-materializer-composition-v1 revision1 remains proposed, accepted none.
Execution sourcefd2ddfdbc66c1177167487085bccfbac818ddb39; Note pin
62eb9852904fd6f42cc6d3539ce9464145db560e. The exact run.py command above executed
on CPU one thread, with no training or product behavior change. Four diagonal
fixed-H controls completed, independently reparsed and exactly matched their
saved row-stream hashes before any cross. Boundary checks also verified strict
HH<20, LN-head inclusion, safe cross-column separation and RH<=20.

Native session45026 is terminal failed at the declared planning-attempt guard
after211.678153s. It produced all32direct crosses,21complete screened crosses
and one capped screened prefix:54main calls attempted,53complete. Ten remaining
screened calls were not attempted. This is a conforming early stop, not a
complete64-chart comparison. Do not resume it, regenerate existing outputs,
increase the four-proposal budget or count absent cases as passes. Artifact
owner artifacts/joint-audio/20260925-head-materializer-composition-v1.

freeze.json SHA1f2366a214aea3fcb11a4c3311febff9a9af0be26ec8f89a5e88da9cf3f2f60e;
failure.json SHA818032b1a0170f93c9a0e3a3dc5a2173ed3610eb6ebc6b57c11197e3eac0bad7.
assess_terminal.py preserves terminal-result.json rather than overwriting the
failed run: SHA125a0c3f51570a73fb11e361db323ff3087c1c0b1051144c55b8ac75f5702fa1.
It verifies all54row hashes and unchanged checkpoint files. The end-of-driver
in-memory parameter fingerprint assertion was not reached and is explicitly
unclaimed. Complete outputs reparsed and maintained exact supplied H; the
capped prefix has no complete-chart osu! export.

The16prospective initialH/sharedB direct outputs have descriptor components
[.8738583548578978,2.69813674840344,3.8129303721235996], total7.384925475384938.
Against original7.3404225633, the required total/LN improvements and width
non-regression all fail; the H component remains exactly original as intended.
They contain0strictHH and16experimentalRH pairs. The16reverse sharedH/initialB
direct outputs have components[1.7634942567999312,2.9801342330093363,
5.462095174952044], total10.205723664761312, with28HH/20RH. Screened metrics
are incomplete for both crossings and are marked invalid, not calculated from
the successfully completed subset. No composition is adopted.

#### The actual scheduler failure

The capped call is reverse sharedH/initialB Take/profile1. Coverage stops at
113296ms of144236ms, with1476H published; generation took4.137662s and had10
rejections in total. Its last requested8s window is113296..121296; each proposal
reaches a121315ms cut and checks through121335ms. All four are rejected byHH:
attempt0 lane3 repeats115114..115119 (5ms);
attempt1 lanes2/3 repeat115105..115119 (14ms each);
attempt2 lane3 repeats114285..114294 (9ms);
attempt3 lanes2/3 repeat115105..115119 (14ms each).
These are not RH-only rejections. The failed window consumes.604306s; the
call's maximum window service is.795497s, below2s. This is failure to find an
acceptable continuation within the proposal count, not observed compute stress.
All published rows remain HH/RH-clean, but clean publication does not imply
continued playback coverage. Actual rejected full row trajectories were not
captured; only their exact pair records and proposal metadata are available.

#### Available-output Lens plan and current coverage

Renderer session39698 is terminal success. It uses the verified parent bundle
fa4fee5185909034db52f1f909f4845654f15d267c2411179e983efbb9b9cc81 and preserves
its bytes. Main plan1e8c08cba0a16c6bcc585dc1a3fe4913bc68d4ea329a8b9581a42c819ee8011f
has86scopes:80new scopes/258pages and six exact parent-reference reuses.
Render-result37e5d6fb2ba7c9728958a4655cfc6b5e243cf9b4a6937a5d653e6d66eefd7a7f;
new bundle58ea4ffe37669a6996db6c080d01a67ed7413c38d3c846909c131eca7f48606f.
New-owner storage188030180bytes after rendering, within2GiB. All16human records
were retrieved. No process remains live.

Because the declared native guard stopped the cohort, missing/partial outputs
cannot supply all original screened scopes. main-lens/unavailable.json
(131ce599675da2eff26560297a83e0b18875e7e3d62db479f1298463e7b85a11)
records the10unattempted calls and capped Take. Additional post-hoc direct
contexts/dense scopes are explicitly labelled when their screened output is
unavailable. They permit reading actual generated musical material, never
substitute for a successful screened case or change a primary/system threshold.
Every available resampled accepted window, selected screened context/dense
scope and directHHwitness remains in the inspection plan. The capped prefix
is not presented as a complete chart; its failed proposal pairs remain in the
terminal evidence. Completion of this available-output review cannot make the
original64-chart system comparison complete.

Actually viewed21new pages with full native action/LN tables at scopes3,16,35,48,75.
Tracker review-progress.json and in-session composition_lens_pages/tables/
observations store the exact inventory. Other237pages/75new scopes remain
unreviewed. Six parent-reference scopes27..32 have exact completed identities.
Do not rerender or rerun models to continue; read remaining frozen pages/tables.

The paired Hysteric long context gives positive and negative concrete evidence.
Prospective scope16 has paired/triple holds with staggered releases:802ms LN1
spans3H,1363ms LN1 spans4H and1408ms LN1 spans6H while other columns tap or start
and release holds. Numerous groups instead end before the next H, preserving
variation in relationships. Reverse scope75 is mainly moving TAP with one
142ms hold and a late staggered three-LN episode, so it retains some capacity
without the prospective output's LN organization. Fine cross-column intervals
remain; a21ms same-lane repeat is outside strictHH but is not certified comfort.

AsItWas/profile2 scopes3/35 are additional direct-only contexts. Both contain
no LN in the selected scope, despite retaining chord organization. Prospective
scope3 alternates full0/2 and1/3 double groups over five H at149–187ms and has
one quad; reverse scope35 has recurring broad triples and two quads. These are
not blanket bad-chart labels. The already reviewed shared diagonal with that
same shared H plan had independent LN organization: H count alone does not
describe the body/trajectory effect.

Scope48 is the complete reverse direct Take counterpart near the failed window,
not an actual rejected proposal. It has triple0/2/3 at115105, T1at115114 andT0at
115119, consuming all rested lanes and repeating0after14ms. No LN participates
in that conflict. Earlier chord size must change for that prefix; the9ms
0/3-to1/2 exchange nearby is HH-safe. This reinforces the need to reason about
future lane availability before commitment without banning fine H spacing.

Recommendation remains REFINE. Numerical composition criteria already fail,
and the reverse decoder has an observed guard stop. Complete the remaining
available-output Lens review before a final curated composition report or new
intervention. Current evidence preserves expressive local mechanisms while
showing unreliable global control and bounded-retry robustness. No listening,
player test, whole-cohort qualitative judgment, model promotion, remote push,
Note lifecycle change or goal completion is implied.


### Result Log: composition available-output review completed

Card head-materializer-composition-v1 revision1 remains proposed, accepted none.
The native run remains terminal stopped_on_guard; no model, training, renderer,
or passed behavioral check was rerun to complete this review. No live job remains.
The original64-call comparison is still incomplete:32direct and21screened
charts completed, one screened Take prefix was capped, and ten screened calls
were unattempted. Missing cases are not passes and the successful screened
subset is not substituted for either arm's aggregate metric.

All80new scopes and258new time pages in the frozen available-output plan have
now been inspected, with complete action/LN articulation tables. Six exact
source-reference reuses retain their completed parent identities. This includes
all28direct HH pairs, their surrounding organization, every available accepted
window requiring resampling, and the selected fixed/dense contexts. Post-hoc
direct fallbacks remain explicitly distinguished from unavailable screened
results. Actual rejected proposal trajectories were not saved; the direct Take
counterpart is never represented as one of those rejected samples.

Local owner artifacts/joint-audio/20260925-head-materializer-composition-v1.
The complete review.json has SHA-256
c7a1f1a764856b9038854e6f95d5c74f56c3e6e31183768b72387779f9591eaf.
finalize_review.py checks the recorded80scope/258page inventory against the
unchanged plan, complete table inventory, six parent review/plan/semantic
identities and terminal cohort accounting. Its all_declared_scopes_covered
field explicitly refers to main-lens/plan.json, the available-output scope;
original_full_cohort_complete remains false. The native terminal result is
unchanged; its historical qualitative_review=pending field is superseded for
available scopes by this separate review record, not rewritten in place.

The primary prospective comparison remains negative: total error7.384925
versus7.340423 original (+.6%), LN error3.812930 versus4.096142 (-6.9%, short
of15%), width2.698137 versus2.370422 (+13.8%, above10%guard). ExactHpreservation
passes. Reverse total10.205724, LN5.462095 and width2.980134 show a different
tradeoff. These are TRAIN-transformed/scaled descriptor errors, not musical
quality or a percentage of failure attributable to R1.

The visual review preserves positive evidence. Prospective Hysteric/profile2
has independently staggered holds spanning3/4/6H, paired/triple roles and subset
releases; Zenithfall has a2083ms hold spanning8H with other-column actions.
Reverse AsItWas/profile1 repeats triple1/2/3 six times at roughly160–190ms while
column0 holds through the first five H. Some short LN groups instead close
before the next H. Thus LN fraction, held-role organization and repeated-grip
structure are different observations. No generic anti-Jack or hold-duration
floor follows from this evidence.

Control remains unreliable: prospective FoolMoon/profile2 selected contexts and
peak are all TAP, as is the prospective AsItWas/profile2 fixed direct context.
The latter still has alternating0/2 and1/3 doubles and a quad. Reverse Hysteric
profile2 is largely TAP in inspected contexts and its148H/8speak, though other
scopes retain brief overlapping holds. The complete direct LN fractions vary
nonuniformly with H/materializer exchanges; denser H alone does not explain
all losses. These are scope-specific observations, not source-tag copying
requirements, listening judgments or a blanket bad-chart label.

The direct HH witnesses separate local wrong-lane selection, chosen chord sizes
exceeding rested capacity, and LN release interactions. In the direct Take
witness at115105/115114/115119, five attacks occur within14ms across four
columns. With H times preserved, an earlier chord must contain fewer heads;
reassigning all five heads cannot repair the conflict. Another Take witness
combines9ms HH and9ms RH because the only HH-rested lane has just released.
By contrast, many1–19ms cross-column splits and complementary double exchanges
survive screening. Exactly20ms same-column pairs are excluded from strictHH,
without being certified comfortable.

The actual reverse Take screened guard stop remains a liveness failure:
fourHH-rejected candidates in.604306s, stopped coverage113296/144236ms. It is
not an observed accelerator/CPU capacity failure. ZeroHH/RH in published rows
does not establish that playback can continue. Frontier2 is present, but its
now+1 release opportunity is not an actual release forecast or reservation of
future rested columns. This motivates jointly considering chord size, actual
LN occupation/release and upcoming H before commitment. A typed/cardinality
factor, finite-horizon constrained generation and changes to training feedback
remain hypotheses; none has been selected or validated by this comparison.

Curated report docs/research/head_materializer_composition.md and the updated
head_factor_drift.md are committed locally as
844de84ee7a2f7560939136710dbab6ce907e006. The report is self-contained and pins
the numerical/visual evidence without depending on this Note. Verification
checked source owners, local links/fragments, evidence hashes, cohort totals,
recorded review coverage and the staged diff. No runtime behavior changed, so
passing model tests were not rerun for the prose change. The earlier ad-hoc
reporting read initially assumed a parent realized_values field and was corrected
to the existing description.ln_fraction; it neither ran inference nor changed
evidence. Product and Note publication remain local only.

Recommendation remains REFINE: do not adopt either functional composition,
resume the capped comparison, increase its proposal budget, or claim the final
playable system. Available-output review is now complete, permitting the next
research choice. Prioritize a justified change to the generative coupling and
continuation contract, evaluated for requested organization and sustained
coverage as well as strictHH. No human acceptance, Note lifecycle transition,
listening/player claim, remote push or goal completion is implied.


### Idea Critique: separate action composition from column realization

Question: can a dedicated causal distribution over row action counts make
requested chord/LN organization more reliable while retaining R1's layout
relationships? The completed composition comparison rules out a useful
simple exchange of I/S conditionals on the tested panel. It also preserves
concrete positive LN/chord capacity. The next intervention must change the
learning/information structure, not repeat the default-off HH repair policy.

Let m(y)=(number of attacks, number of LN starts, number of releases).
The complete four-lane action alphabet induces35joint count triples, including
the terminal/no-op triple. For each physically legal mark m, normalize the
existing complete-row score only over legal rows having that mark; multiply
by a separately normalized mark probability. This is an exact finite
factorization, not independent per-lane Bernoulli sampling. It preserves every
currently legal row and permits taps/chords/holds/subset releases.

Closest analogues: Boulanger-Lewandowski et al., Modeling Temporal Dependencies
in High-Dimensional Sequences (2012), https://arxiv.org/abs/1206.6392, separates
temporal conditioning from a joint distribution over simultaneous polyphonic
content. Here the output alphabet is small enough for exact finite sums.
Compound Word Transformer (2021), https://arxiv.org/abs/2101.02402, models typed
musical attributes with distinct heads and grouped event representations.
Here the three count attributes form one joint categorical mark; timing stays
native-ms and no beat-grid or MIDI note-duration representation is imported.
These are architectural analogies, not evidence of gameplay quality. The
provisional novelty claim is only an adaptation of coarse/fine conditioning
and explicit normalization, not a new likelihood or new supervision source.

A competing branch is finite-horizon constrained inference. Twisted SMC,
https://arxiv.org/abs/2404.17546, uses learned future-potential estimates to guide
partial-sequence sampling. The relevant mechanism is retaining and reweighting
promising partial futures; it does not directly solve our uncalibrated profile
response and requires a defined gameplay potential. Defer that branch during
this learning comparison. The earlier optimistic HH-only correction reduced
one conflict but worsened a late RH count; its relaxed release assumption is
not an actual release forecast. Do not reinstate it as an undeclared second
intervention.

Selected branch: action composition should not inherit the entire R1 layout
history and cumulative exact-count state. Give it its own31-row finite history
of elapsed time and observed count triples, full encoded audio,16-H preview,
elapsed time since the last complete row, and sorted active LN ages. Sorting
makes this physical projection independent of column identity; hidden tails
are absent. R1 still reads full layout history, direct audio, exact state,
preview and frontier2 when distributing mass within the selected count group.
Temporal count memory allows recurring/changing widths; this is not an
independent-frame count model. H and R retain their existing input contracts.

Supporting signal: with H and its shared audio representation fixed in both
arms, the count/layout model improves requested width/LN response and keeps
inspected repetition/independent held roles. Falsifying signals include count
control no better than matched flat training, collapse to short isolated holds
or taps, more publication failures, or excessive runtime. An alternative
compatible with failure is inadequate rare-profile exposure or source/native
history mismatch; the factorization does not remove either automatically.

Outcome TEST, proceeding to the bounded Design below under existing local
implementation/execution authority. No acceptance, adoption or publication.

### Experiment Card: count-layout-materializer-v1

Revision1; proposed; accepted revision none. Owning Note
2026-09-23-audio-skeleton-r1-integration. Clean baseline
844de84ee7a2f7560939136710dbab6ce907e006. Standing user authority covers the
local code, tests and bounded fits; no separate experiment approval is needed.

Hypothesis: explicitly learning row-count composition with its own causal
count history, then normalizing R1 within each mark, improves native requested
width/LN organization without reducing the supported action vocabulary.
The one causal intervention is this count/layout conditional versus the
current flat conditional. Both arms freeze the full audio encoders, shared
profile projection/prior and all H modules to the originalI checkpoint, while
fitting R and row materialization. Freezing is common to both arms to separate
materialization learning from H drift; it is an experimental isolation, not an
adopted final rule that audio must remain frozen.

Model definition: row_factorization=flat or count_layout. In count_layout,
m(y) is the35-way joint triple (heads,LNstarts,releases). For each query, G_m
is the set of physically legal complete rows with that mark. Define
log p(y)=log q(m(y)|audio,count_history,preview,LN_ages,row_age)
+s_R1(y)-logsumexp_{v in G_m(y)}s_R1(v).
q is normalized only over nonempty G_m. This includes release-only and
terminal rows with their original support. The35-way head reads a32-wide,
four-level FiniteTemporal count encoder (31past rows); each past row token
contains existing time_features(delta) and the three counts divided by4.
Query state contains time_features of last-row age plus four sorted active
LN ages, with missing bits for absent holds. Concatenate that encoding, the
full conditioned audio and existing16-H preview into a128-wide GELU MLP
ending in35logits. No generic row embedding or cumulative head/LN totals enter
q. There is no new H feedback, lane-independent output factorization, duration
floor, spacing cutoff, source tail, or correction policy.

R1 retains its original scores, direct audio, exact state and frontier2 inside
G_m. The full256-way probability remains available and is sampled once with
the original row RNG; there is no extra mark-sampling RNG. Training sums
unchanged H/R likelihood and the exact joint row NLL; no auxiliary weighted
mark objective, pseudo labels or preference loss. Mark targets are deterministic
functions of each separate source chart. All existing rows remain representable.

Inputs: corpus manifest4ad9abfd0ae7798e0a85b96a5dbecdaefd2a81f78bf181438407442b964118e1,
615TRAIN arrangements/240audio groups and36VAL charts; existing canonical Mel
and normalization9cf461a0e825f974f0a80a364123c7afedf1683af76e60fe09b0fbe51c2c8287.
Original checkpointabc27f1d192869419e42729a6b9fcdfd1c507fd672e12c9a9c7c868a5082a2ef
at artifacts/joint-audio/20260924-alias-restored-v1/planned-training/shared-profile-cond-main-v1/last.pt.
Banka8973b7f94fde90d9f3cc639eb63e781dd295435622ce79fd54fe244789e6f03;
panel484d173628856a431e034b1dbdacab9173c40f3edf696d1b9a19721aeec70dc9
at artifacts/joint-audio/20260925-fresh-audio-system-v1/panel.json.
No TEST access, data deletion, human-label edits or additional annotation model.

Baseline observed originalI16explicit cases: H error.8738583548578978,
width2.3704218356354247,LN4.096142372828725; width+LN6.466564208464150.
Exact source record is fresh-audio result9f3c4580adaa6239c44e04bb5d95a0ed921b8fd958b1eee26ccd8cedfa91e862.
These are transformed/scaled descriptor squared errors averaged equally over
eight audios and two requests. The matched flat/materializer-only continuation
is not measured yet and will be reported regardless of its result.

Implementation scope: planned model/config/Hydra projection, a small count
representation/normalizer owner, training materializer-only parameter selection,
interval inputs and cached native count history, checkpoint initialization and
focused owner tests/docs. Existing defaults remain flat/all. Warm initialization
must copy every common tensor exactly and report only the new count modules;
same bank checks precede copying. Old default model scores/RNG behavior must
remain unchanged. Model-backed commands use uv --extra mps; pytest adds --group dev.

Before fitting, test exact normalization/group marginals, within-group row
ratios and invariance to an arbitrary common score shift within each group;
all-inactive groups must have finite relevant gradients. Test support,
mirror symmetry, count-history independence from tap layout, active-LN age
dependence, future-tail noninterference, BOS, dense/cached history agreement,
publication forks, CPU/MPS probabilities/gradients, complete checkpoint/Hydra
field round trips, and frozen H/audio parameter ownership. Tests must exercise
real training/native entry paths, not only a duplicated reference formula.

Both arms use seed253001, sample_seed253002, validation_seed230943,
two songs/update,two8sintervals/song, fresh optimizer and identical frozen
exposure plan. Learning rates3e-4 for new/noninherited materializer modules
and3e-5 for inherited R1 modules; weight_decay.01,max_grad_norm1.
Preflight32updates/arm with sixVALsongs and max_seconds300, fresh names
count-layout-flat-preflight-v1 and count-layout-factor-preflight-v1.
Require finite losses/gradients, exact frozen-tensor hashes and checkpoint
reload before main; do not judge native style from32updates.
Main1200fresh updates/arm, validation_every300, all fixedVAL records,
max_seconds1800/arm; start again from originalI, not preflight endpoints.
Names count-layout-flat-main-v1 and count-layout-factor-main-v1.
End-of-budget checkpoint is fixed; no NLL-based checkpoint selection.

Commands: uv run --extra mps python -m
ensomi_model.research.planned_audio_continuation.hydra with the pinned initial
checkpoint/bank paths and hashes, train_scope=materializer,
row_factorization=flat or count_layout, seeds/settings/names above.
Freeze exact resolved commands, clean intervention source and committed Note
before runs. CPU one thread for native generation, MPS for fitting on this
AppleM5/24GiB,Python3.10.20/Torch2.11.0. At least2GiB available RAM and40GiB free
disk; PAUSE/resource guards apply. No overwrite/resume, no remote actions.
Artifact owner artifacts/joint-audio/20260925-count-layout-materializer-v1;
new artifacts including both fits/native/Lens bounded by2GiB.

Native comparison: all eight panel audios, profiles1/2 and seeds251701..251708.
Generate32direct charts (16/arm), then32screened charts using the unchanged
8s/20ms-halo/four-proposal policy with RH screen enabled. H is generated by
each endpoint's unchanged own H/audio path; require exact originalI H hashes,
including screened outputs. Reparse every complete chart and replay every
published update/coverage. Capture actual rejected proposal witnesses with
on_rejected, bounded to the first four rejection records per chart, rather
than confusing them with a separate direct rollout. No more than90s/30000rows
per chart or1200s for the native driver. Stop on mismatch, nonfinite values,
declared resource/attempt limit; retain incomplete/absent cases as failures.

Primary: direct width+LN standardized MSE on16cases must be <=.85 times the
matched flat continuation and <=6.466564208464150 originalI. Each width/LN
component must be <=1.10 times matched flat; H component/hash exactly
originalI (report tolerance1e-10 for the numeric component).
All factor-arm screened charts must complete withHH<20 count0,RH<=20 count0,
independent reparse and unchanged H. Report all arms' direct/screened counts,
failed/empty calls and requested/realized values; no successful-subset averages.
Cached-Mel integrated H/body readiness <=2s and every window service <=2s;
separate fresh-audio startup if a candidate warrants later integrated testing.
NLL is diagnostic only, never a playability gate.

Qualitative scope is fixed before training and deliberately bounded:
screened windows FoolMoon167206..169945,Hysteric243000..247000,
Revenge239894..243706,AsItWas123000..127000 for both profiles and both arms;
each factor-arm profile2 chart's densest4s H window; factor-arm profile1
densest4s for those same four audios. Inspect all their time pages and full
actions/LN tables. Additionally inspect the four shortest directHH witnesses
per arm, tie-broken by audio order/request/time/column, with500ms context
before/after; full-chart counts still include every witness. Merge overlaps.
Use exact parent human-reference reuses only after identity verification.
A missing screened chart permits an explicitly labelled descriptive direct
fallback, never a replacement successful case. No requirement to render every
resampled8swindow; their complete numeric records remain available.

Judge repeated/changing grips, fine timing, independently held roles and subset
releases. A count improvement with widespread isolated-short-LN substitution
or loss of existing independent held/chord relations is not an improvement.
Do not require a generated alternative to copy its source tags, impose a
generic LN floor, or claim listening/player quality. A positive numerical/
system/qualitative result motivates an integrated candidate test; a negative
result defeats this marginal replacement under the frozen representation;
mixed evidence remains REFINE. Rare-profile exposure, one seed, source/native
history shift, coarse mark-memory length and the frozen shared encoder are
explicit confounders. The full real-time expressive-system goal remains active.


### Result Log: count/layout implementation and paired preflight

Card count-layout-materializer-v1 revision1 remains proposed, accepted none.
Clean intervention source4567d87732320f27c01b40b15f74846865219934 implements
the declared35-way count conditional,31-row count history, exact within-group
R1 normalization, source/native count inputs, checkpoint initialization and
materializer-only training ownership. Flat/all defaults are unchanged.
The adjacent parameter_counts docstring now correctly says all parameters,
because frozen parameters remain included in that existing counter.

Seven focused count tests pass in3.02s;63existing planned-audio owner tests
pass in7.90s. A subsequently strengthened nonzero count-history-gradient
assertion passes in the focused CPU/MPS test (1.84s). This is70unique tests,
not71. Coverage includes analytic group masses/ratios, inactive-group gradients,
common-group-score invariance, mirror/input boundaries, source/native cached
scores beyond31rows, query partitioning, fork ownership, actual training,
frozen parameters, Hydra projection, old checkpoint defaults and reload.
Three initial test failures were fixture problems: an insufficient random
legal pair on MPS, direct use of an unexpanded boundary tensor, and failure
to exclude padded teacher query rows. The corrected fixtures retain all real
queries and the original probability/gradient assertions. No model defect
was hidden by changing a tolerance or excluding a real event.

The flat model has4250174parameters,2926490trainable in this comparison.
Count/layout has4416513total,3092829trainable:166339additional parameters.
Both freeze1323684parameters. The separate mark conditional intentionally
removes R1's common cross-mark score biases, including frontier2's cross-mark
component; frontier2 continues to affect relative layouts within each mark.
It is therefore not a preservation claim for the whole former frontier2
policy. Native short-attack, sustained-coverage and structural checks remain
necessary. The sorted-LN projection and31-row mark memory are modeling biases,
not a proof that layout never affects desirable action composition.

Artifact owner artifacts/joint-audio/20260925-count-layout-materializer-v1.
Both32-update preflights completed on MPS, flat32.615221s and factor34.295448s.
Each consumed128identical intervals,996154ms,6956H,307R and268735R clocks.
Full fixed exposure protocol SHA14ecb11ce4ca3882c31f4c6be01d0cd8c2ad03b7db8bbbdde3c9aa8e9d84b440.
Original audio/H/profile frozen fingerprint
afa101b221ed85ffceb1d163917ef88a22a77f0dcda4037f9b9f5268622e4f58
is identical in both models before/after fitting and reload. Checkpoints:
flat3a7a6a082be14154afabc1b986d19e635124213d80772e0fe15102306a1f6c0b;
factoref3ef3c8a7a2fda7243c6607a6483f058a26f3ed5aa5179a71f113616b8cc3cb.
Preflight result85e3480e83480e586191978eb4d9212d574ed75ddde1819b9e0e70341338778b;
freeze def8cadd76b501c70077ca253e696920bfba822e4bf1c978614ee38646da13ec.
The six-VAL-song preflight is execution evidence, not native style evidence.

Both main fits restart from originalI with fresh optimizers, using the exact
recorded Hydra commands from train_stage.py/main-freeze.json. Execution Note
pin007d551eeeda0763cacc7762f3ac1f8d8a6076ef retains the unchanged Card.
Main freeze SHA0aa086fe441f52e51ec1092a31b10baa8741526ee3640edc5cd117b969d7f7a4.
The flat1200-update main fit completed in799.511985s, consuming4800intervals,
37392112ms,253263event rows,239124H,14139R and10692321R clocks. Its checkpoint
is da08044806e99eae6a6221c06ef8c91c02711361fae68c896c9af01bde61dea2;
result2557a1e05498568cd56bdc8c57198c11f830099aadf368fc5eaec1ebe31818d1.
Frozen parameter bytes and complete checkpoint reload pass.

At the last authoritative read, terminal session11773 is live and its factor
main fit has reached610updates at481.294s, with about4.68GB available RAM and
1.90GB MPS driver allocation. This is a progress snapshot, not a terminal
result. Resume by checking that same handle or its actual process state; never
restart from a stale progress file or an observation timeout. Wait for the
fixed1200-update endpoint or declared cap before native comparison. No other
training process or recurring automation was launched.

The source exposure audit finds3profile1 and3profile2 intervals in preflight,
and116/133 in the full4800-interval plan. Both profiles still have limited
exposure; the complete plan and normalization identities were compared.
An ad-hoc reporting read first looked for normalization_sha256 in result.json;
the authoritative field is in freeze.json. Correcting this reporting lookup
required no run restart or input modification.

Prepared native.py and inspect_lens.py have not executed. The native driver
requires both main endpoints, compares all64direct/screened calls, checks H
against originalI, records exact replay/coverage and captures at most the first
four actual rejected trajectories per chart in a separate rejected/ tree.
Keeping that tree outside the final export directory preserves save_rollout's
fresh-directory invariant. Its originalI descriptor baseline was independently
reproduced: components[.8738583548578978,2.3704218356354247,4.096142372828725],
body6.466564208464149. The Lens script implements the Card's bounded fixed/dense
scopes and four shortest directHH witnesses per arm, not every resampled8s
window. Both scripts are still unfrozen preparation; pin their bytes at execution.

Next action: finish observing session11773, verify both main frozen tensors,
exposure and checkpoint hashes; then run the prepared native comparison and
complete its declared Lens review. No NLL-based selection, new fitting branch,
resume of a capped case, automatic promotion, listening/player claim, remote
push, Note acceptance or goal completion is justified yet. Goal remains active.


### Result Log: count/layout main endpoints and native playability

Card count-layout-materializer-v1 revision1 remains proposed; accepted revision
none. This append supersedes the live-progress snapshot above with terminal
evidence, without changing the Card. Main driver11773, native driver45585 and
Lens renderer60873 have all terminated; do not poll, resume or restart them.
Standing local implementation/execution authority is unchanged.

#### Experiment and reproduction

Clean implementation source4567d87732320f27c01b40b15f74846865219934; baseline
844de84ee7a2f7560939136710dbab6ce907e006. The paired train_stage.py main command
and resolved Hydra commands are frozen in main-freeze.json
(0aa086fe441f52e51ec1092a31b10baa8741526ee3640edc5cd117b969d7f7a4).
Execution Note pin007d551eeeda0763cacc7762f3ac1f8d8a6076ef contains the same Card.
Both arms use the initial checkpoint, corpus, normalization, profile bank,
exposure and seeds specified in the Card; no TEST data or human-label changes.

Both main fits reach1200updates with identical4800intervals,37392112ms,
239124H and14139R. Flat takes799.511985s; factor910.983054s. Both preserve the
exact frozen audio/H/profile fingerprint and reload their saved checkpoints.
Flat checkpointda08044806e99eae6a6221c06ef8c91c02711361fae68c896c9af01bde61dea2;
factor41ab71640ce9571ac7f40d9f851a51e2ba54c10ba97fc6f9e54dbc54fac76185.
Main resultdebcbdf58f99e5359070465a8e77384874665008d7fb5a4f02542e33fc524908.
No NLL-selected endpoint or preflight continuation is used.

Native command: uv run --extra mps python
artifacts/joint-audio/20260925-count-layout-materializer-v1/native.py.
Execution Note pin851511b9ca2636463edd51fb6ec339250656c0d0; freeze
b7db758c974d266c055c3b4801f62d84e3a63bfff265cb91d0c434b3c994a8b8.
Source, pinned audio/Mel/checkpoint bytes and bank match. CPU one thread on
AppleM5/24GiB,Python3.10.20/Torch2.11.0; fitting used MPS.
Native ends normally under the declared planning_attempt_limit guard after
198.096102s. No proposal-budget increase or capped-case resume.

All32direct charts finish. Of32planned screened calls,9finish,1factor Hysteric
profile1 is capped,22are unattempted. All41complete outputs independently
reparse; emitted rows replay, coverage is monotone and H hashes/prefixes exactly
match originalI. Complete screened control aggregates are invalid in both arms.
The native result43d130ed6291abdebc3c994e1ad27b833954ef84deed5f9dfcc8422e4a60b5ff
retains its original qualitative_review=pending field; the separate completed
review below owns later observations. Do not mutate the frozen native result.

#### Results

The16direct cases per arm use equal case weights and the Card's transformed,
TRAIN-standardized squared descriptor errors:

| Arm | H | Width | LN | Width + LN |
| --- | ---: | ---: | ---: | ---: |
| OriginalI | .873858354858 | 2.370421835635 | 4.096142372829 | 6.466564208464 |
| Flat | .873858354858 | 2.563332941179 | 3.770444233532 | 6.333777174712 |
| Factor | .873858354858 | 1.301075614304 | 3.608154187841 | 4.909229802145 |

Numerical primary and component guards pass: body improves22.5% versus flat,
width49.2%,LN4.3%,Hunchanged. Five audios improve body; Hysteric/Take/Yomi worsen.
This is a small one-seed panel without a generalization confidence interval.
Profile2 requests LNfraction.734; actual flat→factor falls.712→.226 inHysteric,
.539→.219 inTake,.463→.084 inYomi. Better mean descriptor response does not
establish reliable independent control.

Flat direct has2HH/48RH pairs; factor24HH/16RH. HH means successive same-column
TAP/LN-head attacks strictly<20ms, exactly20excluded. RH<=20 is a separate
experimental preference, not an LN-duration floor. Factor's18bad rows classify
as17chosen-head-count>currentlyfree+HH-rested capacity and1layout choice.
Flat has1layout choice and1zero-capacity past commitment. These are selected
actions, not total risk mass or a causal percentage of R1-restored failures.

Factor Hysteric profile1 stops at243554/301008ms with1716H delivered.
Cached-Mel integrated H/body readiness.450761s; failed-window service.743592s.
Across attempted calls readiness<=.541770s and window service<=.875281s.
No fresh-audio startup or compute-saturation claim. Published prefixHH/RH0
does not establish generation liveness.

All four failed-window proposals repeat a column across248930→248935,5ms.
Actual first-three rejected trajectories were captured:
attempt0T0123→T023; attempt1T03→T123; attempt2T012→T01 whileLN3,started248824,
remains held until249047. The fourth has metadata only, because an earlier
window26RH rejection occupied one of the first-four capture slots.
Attempt0leaves zero rested columns; attempt1has two for three heads;
attempt2combines three recent attacks with one older hold. Current layout
cannot undo the earlier commitment. A possibleRnow+1 does not imply an actual
release before the upcomingH. ReleasingLN3 between the twoH could helpHHalone,
but the separateRHscreen would require earlier release or different counts.

#### Qualitative inspection and conformance

Lens command: uv run --extra mps python
artifacts/joint-audio/20260925-count-layout-materializer-v1/inspect_lens.py.
All32new scopes,60time-proportional pages and their complete native-ms actions/
LN articulation tables were inspected. Six exact parent human-reference scopes
reuse semantic/interval/plan/review identities; all16human records had been
retrieved. Parent bundle remains unchanged. Direct fallback scopes explicitly
describe missing screened results; they are not replacement successful cases.
The frozen plan did not require rendering every resampled8swindow.

Review owner artifact review.json:
f804febd5204ee69b7299666dc030c50586243092eee221428388f40ace5d4f0.
Plan a7c9fc62a7da8735ac51ff00979e9cef6c02f25dffee68b262117f51a1b5ffbc;
render ba6b50e4f8da72374525f4f1738ff4d420ce84c7a1ea06e6e073e1bec16094ac;
bundle0c2e7bc0a3ac16b8610f3b159e98c0defbe764d68250839db166a68b0dbdbdce.
At render completion new outputs including allfour fits used329677623bytes,
within2GiB. The review adds only a small JSON observation record.

HystericP2fixed243–247s changes flat's independently overlapping906/797ms
holds and staggered joins/subset tails into sequential single-held roles.
Factor still has458/469msLN with taps beneath, so this is lost layering, not
all-tap output. AsItWasP2fixed123–127s replaces352/338/348msLNhandoff withtaps.
Conversely, factor AsItWasP2peak has1356msLN spanning6H plus913msjoinedLN,
short handoffs above it, and laterquadLNsubsettails. FactorFoolMoonP2peak
has368msLN with82/166msjoinedroles and a subsetrelease/newhead. Independent
LN capacity survives; reliability is unproven.

Tap1–4width variation, ordinary repetitions and close cross-column events
remain. Take19/6ms andYomi10ms splits use disjoint columns; don't ban them by
spacing H. FoolMoon has a27ms same-column repeat in changingchords outside
the confirmedHHthreshold; comfortable difficulty remains unresolved.
No listening, player test or new human style approval. Generated alternatives
need not inherit source tags. Missing dimensions remain unreviewed.

Conformance: paired fits and guarded native execution followed the frozen
procedure. The screened cohort is incomplete by its declared stop, not patched
with successful subsets. No protected field changed. Preparatory failures:
plainpython renderer lackednumpy and exited before output creation; corrected
uv --extra mps invocation completed. Review identity lookup first used
parent main-lens/review.json instead of parent rootreview.json; corrected and
verified exact hashes before writing the new review. Neither required a
model rerun, parent mutation or altered analysis criterion.

#### Evaluation and decision

Outcome REFINE. Numerical body response improves, but the system guard fails
and paired LN organization is not reliably retained. Count/layout as fitted is
not adopted; neither row support nor descriptor accuracy is playability.

Mechanism: the within-mark normalizer cancels any score common to that mark.
It removes frontier2's ability to lower the marginal for an excessive count,
though frontier2 still influences layout ratios within the mark. Once q chooses
more heads than rested columns, every layout isHH-bad. This is a module-coupling
effect, not evidence that R1 weights simply forgot source patterns.

For a physicallylegal prefix with unique column attacks in(t−20,t), letCcount
recent TAP/LNheads andOcountcurrentlyheldLNaged>=20. Exactcurrentfree/rested
capacity isK=4−C−O. RecentLNheads alreadycountinC; olderholds aredisjoint.
Native instrumentation asserts the identity wheneverits prefix premiseholds.
No neural enforcement is implemented. The count inputs can carry this
anonymous capacity without generic tap-column feedback, but31-row learned
compression neednotretain it. The formula alone doesnot guaranteefutureR,
RHsafety, ergonomics or continued coverage.

Alternative explanations remain: new166339count parameters begin untrained,
frozen audio may favor prior flat readouts, rare profiles get116/133intervals,
coarse31-row memory may be inadequate, sorted LN ages erase hand identity, and
source/native history shift persists. Rneuralinputs areunchanged but global
gradientclipping couplesRupdate magnitudes to changedrowgradients; don't
attributeRdifferences exclusivelyto semanticcountlearning. VALrowNLL/s
9.880flat/11.135factor andRNLL2.028/2.034 arediagnostic,notcheckpointselection
or a sufficient reason toscale.

Durable self-contained report docs/research/count_layout_materializer_evaluation.md
and the linked planned-model guide are committed at
151f9c265a9c5eaaa43bc6736f41cdc626da5554. Documentation links/math/diff checks pass;
no product behavior changed after the70unique implementation tests.

Next causal question: compare explicit current-capacity information with
short-horizon joint continuation feasibility, keeping actual LN release
choices in the dependency. A current-row mask alone can still commit a quad
before anotherH or assume anRthat never occurs. Preserve learned variation,
LN roles, H timing, fixed compute and coverage checks. Design a new bounded
comparison before any new fit or modified decoder run. Standing execution
authority persists; Card acceptance, adoption, remote publication and goal
completion have not been granted or inferred. Goal remains active.


### Research branch: probability of a viable continuation

The completed count/layout comparison makes next-row legality an insufficient
research target. Separate three quantities: current clean-row mass; existence
of a clean future; and the learned probability of actually reaching that future.
Restoring a weak frontier residual, adding current capacity to q, and planning
actual R/row futures predict different observations on the same failure prefix.

Closest analogue: Park et al., Grammar-Aligned Decoding,
https://arxiv.org/abs/2405.21047 (full text v3). Their expected-future-validity
factor distinguishes a binary possible-completion mask from probability-weighted
conditioning. The transferable idea is to measure future constraint probability
before choosing a prefix; their grammar/code evaluations are not beatmap
quality evidence. Zhao et al., Twisted Sequential Monte Carlo,
https://arxiv.org/abs/2404.17546, supplies a subsequent family for approximating
future potentials. No twist network or SMC decoder is selected yet.

The application-specific difference is an H-conditioned asynchronous R/row
process with actual held state and publication deadlines. A future probability
must integrate sampled release times and row choices, not assume all held
columns can close at now+1. A hard HH test is distinct from the experimental RH
preference. Preserve both readings in the diagnostic. This is an adaptation of
conditional inference, not a new probabilistic objective or a claim of novelty.

Outcome TEST for a bounded conditional audit. First estimate whether immediate
clean-row conditioning leaves substantial future failure mass, and whether that
mass depends on current LN-start/count choices. Use existing weights before
selecting another fit, decoder or wider architecture. The preceding count-layout
Card is completed as an exploratory comparison; the following Card is the sole
active proposed experiment.

### Experiment Card: count-continuation-mass-v1

Revision3; owning Note2026-09-23-audio-skeleton-r1-integration; statusproposed;
accepted revision none. Standing local research/implementation/run authority
covers this bounded probe. Baseline clean product
151f9c265a9c5eaaa43bc6736f41cdc626da5554, behavior-identical to count implementation
4567d87732320f27c01b40b15f74846865219934.

Question: after a currently clean complete-row choice, how much base-model mass
actually survives the nearby H plan under HH alone and HH+RH, integrating the
true R/row sampler? Selected hypothesis: a substantial part of the factor arm's
failure is future count/occupation commitment rather than immediate bad layout.
The decision is whether the next intervention needs explicit future coordination
instead of only current-capacity information. This diagnostic does not adopt
a decoding distribution or establish whole-chart improvement.

Inputs: previous native result43d130ed6291abdebc3c994e1ad27b833954ef84deed5f9dfcc8422e4a60b5ff;
the exact same flat/factor checkpoints, profilebank, canonical Mel and panel
identities in the completed count-layout Card. No fitting or new data.
Fixed H is the factor direct profile1 plan, already verified identical to
originalI, for the selected audio. Four pinned factor-native prefix queries:
1. Hysteric rejected/factor/hysteric/profile1/1/rows.jsonl,
   SHAea00a750246faf81566c36e921ac41e3438366d43e9a9bd95e1892c7bdf0b242,
   rows strictly before248930; currentH248930, horizon248950.
2. Hysteric rejected/factor/hysteric/profile1/3/rows.jsonl,
   SHA4fdfb42b8379c04174c8c29b81f621aa08dac3db5e3a0fd5ab2b6e8d6e16d0aa,
   rows strictly before248824; currentH248824, horizon248944.
3. The same rejected file, rows strictly before248930; currentH248930,
   horizon248950. This prefix retainsLN3 from248824.
4. AsItWas native/factor/as-it-was/profile1/direct/rows.jsonl,
   SHAa701b6800f7256a0c5c670a13bca0c359107c4bdd3d1af02515bb90f4955eb27,
   rows strictly before148234; currentH148234,horizon148254.
All paths relative to artifacts/joint-audio/20260925-count-layout-materializer-v1.
The Hysteric direct plan file SHA is
01acdc4330865ffedc8177e06de018796800220119cdde5d4beb9fc3be2f92a3.
No hidden future tails enter a prefix projection. Case4's recorded5ms
cross-column split is a healthy control, not a statement that all its sampled
alternatives are healthy.

Paired fixed-prefix scoring uses each checkpoint's own full conditioned audio
and history encodings; the same literal prefix is forced into each model.
This is counterfactual for flat and is not a new whole-chart comparison.
The causal intervention is conditioning the first count triple while sampling
its layout from that model's first-row distribution, then allowing genuine
unmodified R/row continuation. All models, H, controls and observed prefix facts
stay fixed. Behavior-neutral instrumentation reconstructs exact replay/caches,
queries256row probabilities, enumerates35deterministic count groups and records
suffix outcomes. No product behavior/config/default is changed.

For each query compute exact raw group masses and Z0, total first-row probability
passing both immediateHH/RH. Restrict the first-row distribution to this clean
set. For every nonempty count group draw64independent first layouts proportional
to their conditional model probabilities, then fork/sample actual R/row paths
through the frozen horizon. No current/future HH correction or rejection retry.
Report HH-only and HH+RH survival per mark and
Zfuture=Z0*sum_m w_m*survival_m, withw_m the exact first clean mark marginal.
Report95% Clopper–Pearson group intervals with Bonferroni correction within each
query, and their weighted bounds forZfuture; no across-query population claim.
The unmeasured baseline conditional probabilities will be reported, not assigned
invented values. The existing whole-direct HH totals2flat/24factor and18factor
bad rows are prior motivation, not the probe's metric.

Primary signal: for at least one of the three failure queries, Z0>=.95 and the
upper95% bound for HH+RH Zfuture is at most.90*Z0. That supports future-conditioned
work beyond an immediate mask on this bounded prefix. If every failure query
has lower-boundZfuture>=.95*Z0, prioritize immediate-state or wider-history causes.
Other outcomes remain ambiguous/REFINE. Separately report HH-only so an RH-only
effect cannot be presented as explaining the confirmed HH fault.

Mechanical sanity check: for a second H strictly inside20ms, a column still
held after the first row cannot become attackable underRH<=20 by any laterR.
For these specific one-future-H windows, exact post-row free/clock availability
therefore proves some candidate futures impossible. Keep that proof separate
from zero successes in64samples. Case2 spans120ms and requires actualR modeling;
do not extrapolate the short-window proof to it.

Preflight before sampling: verify all input hashes, exactprefixreplay/Hmembership,
dense-vs-cached history reads with existing owner tolerances, normalized finite
row probabilities and row support. Each sampled suffix must re-replay and contain
exactly the planned H through its horizon; all weights remain byte-identical.
Selected-first-row checks must agree with close_pairs. A stop/mismatch/nonfinite
value leaves an incomplete diagnostic, no threshold or denominator adjustment.
Retain first success and first failure suffix per mark plus complete numeric
records; inspect these raw action relations for mechanical claims. This is not
a new whole-chart Lens/playability test. Existing completed Lens remains the
quality context; no new player/listening claim.

Command: uv run --extra mps python
artifacts/joint-audio/20260925-count-continuation-mass-v3/probe.py.
CPU one thread, AppleM5/24GiB,Python3.10.20/Torch2.11.0; no acceleratorfit.
Seeds derive from base253101 plus stable query index, arm, mark and repetition;
record the exact deterministic formula in the execution freeze. First-layout
and futureR/row generators use independent derived streams.
Max900s whole driver, max10s/128new rows per suffix, new artifacts<=200MiB,
available RAM>=2GiB/free disk>=40GiB; explicit PAUSE/resource guards.
Fresh owner above; no overwrite/resume, no networkduringrun and no source
chart/label edits. Freeze source, Note, script bytes, inputs and seeds before
execution. No automatic extension after a cap.

Confounders: four selected prefixes, factor-history bias, one small frozen panel,
finite Monte Carlo uncertainty, short horizons and an experimentalRHpreference.
Survival is a mechanical diagnostic, not style quality, whole-song liveness,
the final constrained posterior, or proof that another learned module is needed.
A positive result selects a subsequent bounded design; a negative result can
reject this local future-mass explanation. No result promotes an endpoint.


### Result Log: continuation-mass preflight failure

Card count-continuation-mass-v1 revision1, Note pin
681f04ecee2fc579bee97c38895541ca38d07e40, accepted none; product151f9c265a9c5eaaa43bc6736f41cdc626da5554.
The frozen probe terminal61910 exited1 after.563566s, before the first query
was recorded or any Monte Carlo samples were drawn. Error: Inference tensors
do not track version counter. Both endpoint parameter fingerprints remain
unchanged. Freeze e1fdcf2c0a098bad79613ea8d2fc8479e391d2f90f7c0f05c74214e0272f1e52.
Owner artifacts/joint-audio/20260925-count-continuation-mass-v1 is retained
with original script/freeze/result. No model-quality or primary-metric result.

Cause: the artifact driver decorated main with torch.inference_mode, so model
construction created inference parameters. FiniteTemporal.signature requires
their _version for cache ownership. Canonical native execution loads models
outside inference mode. The fix is no_grad for the driver while retaining
the existing native inference-mode methods; no model/sampling law changes.

Outcome REFINE, with a behavior-neutral instrumentation correction. Card
revision2 changes the fresh output owner/command to
artifacts/joint-audio/20260925-count-continuation-mass-v2 and pins corrected
script bytes. Question, checkpoints, literal prefixes, seeds,64samples/mark,
metrics/thresholds/guards and per-attempt900s/200MiB bounds remain unchanged.
The original capped/failed attempt is not resumed or overwritten. This is an
explicit new revision, not an extension of its runtime. No acceptance or
endpoint adoption. The corrected run remains separately authorized by the
standing local research task.


### Result Log: continuation-mass fixed-plan input preflight

Revision2 execution Note9a87fd74d7c2515d12223277823ecd94be5e6ce8; product/source,
models and hypotheses unchanged. Terminal16208 exited1 after.507283s with
Fixed H plan requires increasing native clocks within the audio. Zero recorded
queries and zero Monte Carlo samples; both model fingerprints unchanged.
Freeze d4f6a2c8feed4a15b9af0cfd2844b552a458ebb211a3a8f0f46a92178315e2f7.
Retain owner artifacts/joint-audio/20260925-count-continuation-mass-v2 unchanged.

Cause: CompleteRow normalizes clocks to float, while FixedHeadPlan deliberately
requires Python int native-ms clocks. The artifact reader extracted r.time_ms
without int conversion; canonical native.head_sequence converts explicitly.
All stored values are already exact integer milliseconds. Revision3 adds that
conversion after verifying integrality, changes the fresh owner/command to
artifacts/joint-audio/20260925-count-continuation-mass-v3, and retains revision2's
no_grad model construction. No timestamp rounding, different H plan, fit,
sampling-law change, endpoint/seed/metric/budget change or resumed attempt.
Outcome REFINE for instrumentation; no research measurement exists yet.


### Result Log: conditional continuation mass

Card count-continuation-mass-v1 revision3; accepted none; execution Note
bd63054818471bae5fa6ce64ae83a436c39dcbce. Clean run source
151f9c265a9c5eaaa43bc6736f41cdc626da5554, with model behavior at
4567d87732320f27c01b40b15f74846865219934. Both preceding setup attempts remain
terminal/retained and contribute zero queries. The successful driver terminal
94507 exited0 in51.381885s. All eight fixed queries and120nonempty clean count
groups completed64trials each, total7680suffixes. Do not resume or rerun this
completed handle.

Owner artifacts/joint-audio/20260925-count-continuation-mass-v3.
Probe SHA0b9c4dbf100599171036451548db703edfaea47ace916326ebf77d63e033c175;
freeze5059e03991736516301ac0f5de579a23d03965518c14f7bdb1f4331de21a19bf;
result77fee8403e7425b8c53e6081ebcbac81b6a8d8e77027a0c58667dd015e4b9075.
New owner uses9665613bytes. Same pinned checkpoints, literal prefixes, native
H plans, canonical Mel, bank and profile1; no fitting or network duringrun.
One CPU thread onAppleM5/24GiB,Python3.10.20/Torch2.11.0.
Command is the revision3 uv --extra mps probe.py command. Seed streams follow
the exact recorded SHA256 formula with base253101.

All input hashes, integer H membership, exact prefix replay, dense/cache reads,
native first-query log probabilities and forced-row cache buffers verify.
Every suffix replays and preserves every H through the declared horizon.
Both complete parameter fingerprints remain byte-identical. No guard is hit,
no subset is omitted, and no planned threshold or denominator is changed.

#### Probability results and uncertainty

First-row mass passing immediateHH/RH is within2e-7of1 for allqueries.
The small unit-sum error is float32 scoring/normalization roundoff; no material
first-row mass is removed. Reported futureHH-only and futureHH+RH use this same
joint-clean first-row condition. HHstrict<20 and RH<=20 retain their distinct
meanings. Weighted95% intervals use group Clopper–Pearson intervals with
Bonferroni correction within each query; they are not a panel-wide confidence
statement.

| Query | Arm | HH-only mass | HH+RH mass | Joint95% bounds | Exact joint-dead first mass |
| --- | --- | ---: | ---: | --- | ---: |
| Hysteric248930,nohold | flat | .993418 | .993418 | [.894121,.998514] | .001631 |
| Hysteric248930,nohold | factor | .149430 | .149430 | [.077968,.289565] | .038740 |
| Hysteric248824,earlier | flat | .952673 | .948870 | [.816474,.993129] | not proved for120ms |
| Hysteric248824,earlier | factor | .043125 | .041421 | [.005591,.167045] | not proved for120ms |
| Hysteric248930,LN3held | flat | .942215 | .771460 | [.648438,.839272] | .128283 |
| Hysteric248930,LN3held | factor | .047654 | .003706 | [.002074,.102362] | .618766 |
| AsItWas148234,split | flat | .999979 | .999979 | [.905819,.999998] | .000001284 |
| AsItWas148234,split | factor | .997016 | .997016 | [.902395,.998054] | .000175913 |

The three factor failurequeries meet Z0>=.95 andupperZfuture<=.90Z0.
Flat also meets the signal in the LN-entering case. The healthy AsItWas5ms
split remains high-survival for both models. Positive mechanical signal does
not establish an adopted decoder or whole-song liveness.

Without an enteringhold, factor mass.63364 selects threeTAPs; only2/64 ofthat
group's actualcontinuations areclean. TwoTAPs have mass.29481 and25/64clean.
Flat's corresponding masses.31627/.65631 yield63/64 and64/64clean.
A savedfactorfailure hasT123→T023 across5ms; a success withthe samefirstT123
usesT0next. Only.03874firstmass ismechanicallydead here, so mostestimated
failure comes from possible continuations that the samplerrarelychooses,
rather than currentcountchoices that alreadymake everyfuture impossible.

At the120msearlierquery, factorfirstoneTAP mass.56510 has1/64clean; twoTAPs
mass.39863 have5/64clean. Actualfailures still choose triples at248930/935.
Selecting anLN at248824 isnotnecessary for thefailure. Thisdoesnot justify
targeting onlyearlyLN release or makingallholds shorter.

WithLN3entering, firstthreeTAPsuseall remainingcolumns. Keeping thathold,
releasingitnow, or releasingitlater cannotprovide ajointHH/RHcleanhead5mslater.
Theexactjoint-dead mass.618766factor/.128283flat follows fromcurrentstate
andtheone-future-Hhorizon, notfromzeroobserved successes.
ForHHalone, a currentrelease canhelp: flatmark(3,0,1) has58/64HHclean but
0/64jointclean; itswitness tapscolumn3five millisecondsafterrelease.
Factor(3,0,1) has0/64HHclean asitsfuturecounts/layoutsstillovercommit.
Do not mislabel the61.9%combined impossibility as anHH-onlyproof.

#### Evaluation and next direction

Outcome REFINE. Observation: first-row currentvalidity hasalmostunitmass,
butreal futureR/row sampling leaves largefailuremass in thefactorHysteric
contexts. Existingconstraints/sourceNLL do notcoordinate sampledcountchoices.
The fulllegalrow vocabulary and activefrontier2 insideamark do notprevent
this two-row distributional mismatch.

Critical limit: onlythefirstrowwasconditioned. ThisprobehasNOTcompared
sequential currentcapacity masking at every subsequentrow againstlookahead.
A currentmask appliedlater mightrepair manypossible-but-unlikelyfutures.
Itwould still encounteranemptysupport aftersomeearliercommitments, including
theexactjoint-dead examples. The evidence favors a minimalcontrolledcomparison
ofthese two mechanisms before assuming a largeplanner/criticisnecessary.

Strong alternatives/confounders: factor-owned histories arecounterfactualforflat;
fourselectedprefixes are notrepresentativecohortstatistics;64trials/markgive
broadrare-eventbounds; a120mshorizonisnotlong-songstability; RHremainsanexplicit
experimental preference. Flat/factordifferin trainedcountandlayoutparameters,
so thisdoesnotisolateacausalpercentage forR1-restored. Actualfullprefix/horizon
mass estimates do not themselvesmeasure musical quality.

Raw successful/failed suffixwitnesses were inspected for theabove mechanical
relationships. No newfullchart orLensreview isclaimed; thecompleted60page
count-layoutLensreview remainsthequalitycontext. No listening/playerapproval.

Self-contained result docs/research/count_continuation_mass.md, linked fromthe
materializer evaluation, iscommitted at
b35565d6a3a58ccc5ebd0ae296e5a792e4a94318. Local link/math/diffchecks andnumeric
summary/group-count checks pass; no modelcode changed. Allrunprocesses are
terminal, no automation is active, and bothproduct/Notes commitsremainlocal.

Next work: Design one bounded comparison ofsequential currentHH/RH conditioning
versus a short feasible-continuation condition on the fixedfactor endpoint.
TreatR as the actual sampler; neverpretend anearliestpossibleRhasoccurred.
The short-window mechanical fact underRH is that a currentlyheld column cannot
be released after now and attack within20ms; use it only where exact.
OlderLNs mayrequireearlierreleases beyondthatwindow, so empty-support cases
muststop/record ratherthan emitbadrows or silentlyrelax thecriterion.
PreserveHtimes, allnormalrepetition/shortLNsupport, frozenruntimebounds and
whole-chart/LNorganizationchecks. NewCard/implementation/run stillpending;
do notstarta fit orretroactivelymodifycompleted outputs. Goalremainsactive.


### Research branch: current constraints versus short continuation support

The conditional mass audit is complete. A local row screen may repair many
possible-but-unlikely futures, but an earlier choice can also empty all later
clean support. Compare those two mechanisms in full native generation before
adding a learned critic, particle system or a new fit.

The short joint HH/RH constraint admits a simpler exact primitive than a
general search. Every H in(now,now+20] requires a different column from every
other H in that interval. A future release after now cannot make a held column
usable within the interval under RH<=20. For a current complete-row candidate,
advance its actual attack/release clocks and occupation; letA_j be the columns
free and sufficiently rested at the jth futureH. These sets grow monotonically
with time. A one-TAP-per-H continuation exists exactly when |A_j|>=j for allj.
This nested-set matching condition does not assume future releases, predict
their probability or enforce a generic H gap. Future extra heads/LNs are not
needed for the existence witness. Current newTAPs may be used again at exactly
now+20; current releases still cannot at that boundary.

The result is exact only for the joint experimental screen and the covered
20ms H horizon. HHalone permits some early-release solutions the RHscreen
excludes. The651admittedsourcecharts previously havezeroHH/RH, but that doesnot
establish auniversalhumanRHrule: earlier10/13msLN-jacktailgapsremainunresolved
preferencecases. KeepRHexplicit, measuretargetretention andinspectLNexpression.
An olderLNmayneedreleasebeforethis horizon; neitherlocalnorpreviewrowmaskcan
guarantee theunchangedRclockwilldoit. Empty-supportfailuresarepartoftheoutcome.

Closest analogue remains Grammar-Aligned Decoding, https://arxiv.org/abs/2405.21047:
binary possible-future filtering differs from future-probability weighting.
This test isolates theformer, simpler mechanism, not ASAp or theglobalconstrained
posterior. The existing 16-state optimisticHHresponse is a local reference
but assumes earliestpossibleR andusesminimal-editcorrection. Neither assumption
is reused. TwistedSMC/learnedforecast remainsdeferred unless simplerconstraints
fail their full-system/qualitycomparison. OutcomeTEST; nextactiveCardbelow.

### Experiment Card: current-preview-row-constraint-v1

Revision1; proposed; acceptednone. Owner2026-09-23-audio-skeleton-r1-integration.
Standinguserauthority permits localimplementation/tests/boundedruns andcommits.
Baselinecleanproductb35565d6a3a58ccc5ebd0ae296e5a792e4a94318. The preceding
continuation-mass Card iscompleted, notaccepted/adopted.

Question: does preserving a short feasible H continuation improve complete
bounded-publication generation beyond sequential current-row conditioning,
without losing controlledwidth/LNorganization? Onecausalintervention: add the
exact nested-availability condition for H in(now,now+20] to the same current
HH<20/RH<=20 row condition. Both arms use the original complete-row probability
conditioned on their allowed set after full count/layout normalization.
Do not pass the extra mask into model.planned_row_log_probs, which would
redefine group marginals differently. Sample once with the original row RNG;
no propose-then-minimal-edit correction and no new row RNG stream.

Implement optional Python research API row_constraint='none'|'current'|'preview',
defaultnone, in session/direct/bufferedrollout. It is not a packaged-inference
default or trainingconfigchange. Preview mode requires at least5Hlookahead,
enough to either cover the20ms horizon or prove more than4futureH impossible.
Currentmode applies only the immediate HH/RH predicate. Bothapply to all actual
H/R/terminalrowqueries; no imposedLN-durationfloor, Hspacing, grid, extra
skeletonfeedback or modifiedRhazard. The preview predicate tests actual post-row
clocks/occupation with the nested cardinality condition stated above.

The baseline none path must preserve existing model scores, rows and RNGs.
Do not combine either new mode with old correct_short_attacks. Bufferednewmodes
require the existing RHscreenenabled, so contradictory settings fail explicitly.
No-row support returns a distinct diagnostic with failingtime, currentLN/clocks,
preview and retained/excluded modelmass. Restore observedcursor beforethat step;
no badrow or unmaterializedH coverage may be published. Directrollout returns
the committed incompleteprefix. Bufferedrollout treats this as an ordinary
unpublished rejection, permits the same four attempts, and preserves old
resource/error/consumer behavior. Failed forks never alter publishedstate.
Capture decision statistics and actual rejectedprefixes, not inventedrowactions.

Productscope: smallpure constraint/conditioning owner, session/generation/
buffering parameters and diagnostics, focused tests, research guide. No trained
tensor, checkpointformat, Hydra schema, data preparation or defaultpolicychange.
Freeze a clean intervention OID and audit only this scope before modelruns.

Tests before corpus/native: exact strictHH and inclusiveRH boundaries; current
candidatepoststate; trueLNreleases withoutdurationfloor; closecross-columnH;
futureH atnow+20; multipleHnested matching against exhaustive column assignments;
mirror/columnpermutation consistency; terminal/no-future cases; insufficient
preview detection; explicit empty-support handling; same model probability
ratios within allowed rows; default parity; one draw per actualselectedrow;
native cache/fork ownership; rejectedfuture publication isolation; no claimed
coverage at a failed H; unchanged consumer/resource propagation. Exercise real
session and bufferedentrypaths, not only a copied reference mask.

Corpusguard: replay all615TRAIN+36VAL admitted charts from manifest
4ad9abfd0ae7798e0a85b96a5dbecdaefd2a81f78bf181438407442b964118e1 at
artifacts/joint-audio/20260924-alias-restored-v1. Build actualsourceHpreview and
require every target row passes both policies. Verify pinnedsource/cachebytes.
This checks the admitted corpus only, not a universalRHtaxonomy. NoTESTaccess,
labelmutation orsourcepreprocessingchange. Stop onanytargetexclusion/mismatch;
do not relaxcriteria, omitrows or continue withanunexplainedexclusion.

Nativecheckpoints: unchanged flatda08044806e99eae6a6221c06ef8c91c02711361fae68c896c9af01bde61dea2
andfactor41ab71640ce9571ac7f40d9f851a51e2ba54c10ba97fc6f9e54dbc54fac76185.
They retain the original fullaudio/H/profile frozen tensors. Bank
a8973b7f94fde90d9f3cc639eb63e781dd295435622ce79fd54fe244789e6f03;
panel484d173628856a431e034b1dbdacab9173c40f3edf696d1b9a19721aeec70dc9;
canonicalMel/normalization as in the completed count-layout comparison.
Parentnative43d130ed6291abdebc3c994e1ad27b833954ef84deed5f9dfcc8422e4a60b5ff
owns rawdescriptorbaselines and Hsequences. Exactcheckpoint/panelpaths are in
its native-freeze.json b7db758c974d266c055c3b4801f62d84e3a63bfff265cb91d0c434b3c994a8b8.

Run64screened calls:8audios×profiles1/2×flat/factor×current/preview.
Use unchanged seeds251701..251708 bypanel order, ownonlineHplanning, fullcachedMel,
8s window/20ms sampledhalo/fourattempts. Factor16casecohorts are primary; flat
is a declared diagnostic replication, not a pool for favorable averages.
Require every emitted Hprefix/fullsequence matches the originalI plan exactly.
Reparse every complete export and independentlyreplay all published rows and
settledcoverage. No directfallback replaces a missing constrained output.

Expected empty-support or four-attempt exhaustion is an outcome, retained as
incomplete, then continue the next independentcase. This deliberately differs
from the preceding driver that globallystopped on thefirstattemptlimit.
Nonfinite probabilities, identity/replay/Hmismatch, corpusfailure or global
resource/budgetguard stops thedriver. No retrylimitincrease or failedcaseresume.
Capture atmostfirstfourrejectionsperchart with actualprefixrows and diagnostics.

Primarycandidatecriterion: all16factorpreviewchartscomplete withHH0/RH0.
If factorcurrent hasfailures, preview mustreducefailedcases; ifbothcomplete,
previewmustreduce total rejectedproposals byatleast50% when current hasany.
Ifbothcompletewithzerorejections, prefer the simpler currentmechanism unless
otherdeclaredqualitycriteria distinguish them; no inventedstrictwin.
Report every cohort's failures, empty rows, retrycounts and coverage loss.

Controlguard: for a completefactorpreviewcohort, mean transformed/standardized
width+LNerror<=1.10*4.909229802145201 and each component<=1.10times its rawfactor
baseline(width1.3010756143039068,LN3.608154187841294). Hhashes/errorunchanged
(.8738583548578978). Reportcurrent/flatcompletecohorts andwholechartrealizedvalues
aswell; neveraverageonlysuccesses. Descriptorpassaloneisnotadoption.

Runtimeguards: cached-Mel integrated H/body first8s readiness<=2s andevery
publicationwindowservice<=2s. Atmost90s/30000rowspercase,1800swholedriver.
CPUonethread,AppleM5/24GiB,Python3.10.20/Torch2.11.0; uv run --extra mps for
modelcommands, --group dev forpytest. Corpusguard<=300s. AvailableRAM>=2GiB,
free disk>=40GiB, PAUSE honored. NewartifactsincludingLens<=2GiB, no network
duringruns. Fresh owner artifacts/joint-audio/20260925-current-preview-constraint-v1;
freeze exactscript/command/source/Note/inputs beforeeachstage. No overwrite/resume.

Qualitative plan: review availablefactorcurrent/preview charts in fixedcontexts
FoolMoon167206..169945,Hysteric243000..247000,Revenge239894..243706,
AsItWas123000..127000, bothprofiles. Also reviewfactorpreviewP2densest4s onall8
audios andfactorpreviewP1densest4s onthefourfixed-contextaudios. Addflatcurrent/
previewP2Hysteric243000..247000 andRevenge239894..243706 asdiagnosticpaired
contexts. Merge overlaps. Inspectalltimepages/fullactions/LNtables; exact
verifiedhumanreference reuses areallowed. Incomplete/unavailable scopes stay
explicitlymissing. No requirementtorenderallrejectionwindows; inspect actual
failure diagnostics and source relationships for causal claims.

Keep shortLN/chordgroups, sustainedindependentholds, subsetreleases, variable/
repeatedgrips andfinecross-columnonsets visible. A runtime/controlpass with
lostLNorganization or bland single-tap fallback isnotimprovement. No generic
anti-Jackfilter orrequiredcopyingofsourcestylelabels. No listening/playerclaim.
Negativeoutcome mayidentifyRtimingcommitment beyond20ms, source/nativehistory
shift or a failedcomparison; do not automaticallyscaleparameters orreinstate
optimisticreleaseassumptions. Positiveoutcome selectsanintegratedcandidate/
learning design, notfinalmodeladoption. Alloutcomes remainREFINE without
humanacceptance; thefullplayablerealtimesystem goalstaysactive.


### Result Log: current/preview constraint implementation

Card current-preview-row-constraint-v1 revision1 remains proposed, acceptednone.
Clean interventionac7fa3a59696a8cf23d3825a32a7d300bdba0fbb implements the declared
Python-only optional modes and post-composition conditioning. No weights,
training/Hydra/checkpoint schema or packaged default changed. None mode keeps
its previous score/RNG/row path; result metrics now explicitly name row_constraint
and record constraint decisions.

The pure96line owner uses nested eligibility counts, not a large search or
hypothetical release forecast. It includes exactlynow+20 for futureHH eligibility
while currentRH remainsblockedthroughthatboundary. Completepreview or coverage
through the horizon is sufficient; a truncatedpreviewofatleastfiveheads inside
20ms alreadyproves infeasibility. Sessionpreview mode requireslookahead>=5.

NoRowContinuation is distinct from a resource stop. Directsampling preserves
old observedcoverage and returns row_constraint_empty; buffered sampling rejects
onlytheunpublishedfork and uses its existing fourattempts. ActualRrandomness or
Hlookahead mayalreadyadvance inside a failedfork, so callers discardit rather
than resumeit. Rejected rows/unmaterializedHcoverage neverreach publication.

Seventy-nine uniqueplannedowner tests pass in8.76s. A subsequent numerical
review found that masking can leave only finite logweights below -1000; directly
exponentiating them would underflow despite nonempty support. The conditional
now subtracts its retained lognormalizer before sampling and records both
probability and log probability (null log mass only for empty support).
Allnine focused tests, including this finite-tiny-mass case, pass in1.15s.
This is79unique tests, not88; existingnone/defaultowner behavior didnotchange
after the numeric correction. Tests cover exhaustive orderedassignments,
permutations,20msboundaries,short/fullLNs,score-ratiolaw,actualentry/forkcache/RNG
behavior,failedHcoverage,retryacceptance and publicationisolation.

Scoped source/docs diff and documentationlinks checked; localcommitonly.
Prepared artifact corpus.py has notrun. It checks pinnedmetadata/source/cache
bytes,651TRAIN/VALcharts andeverytargetrow against both modes under actual
sourceHpreview. Stop on exclusion/mismatch without relaxing a criterion.
Set its clean source/Note pins beforeexecution; then prepare/freeze the native
64-casecomparison. No training or recurringautomation is running.


### Result Log: current/preview corpus and whole-chart comparison

Card current-preview-row-constraint-v1 revision1 remains proposed; acceptednone.
Clean implementationsourceac7fa3a59696a8cf23d3825a32a7d300bdba0fbb;
executionNote031087e22350167aa717c176a0b7f601c6ccbe95. Corpusdriver95531,
nativedriver69125 andLensrenderer16550 areallterminal-success. No training or
automation isactive; neverpoll/restartthese completedhandles.

Artifactowner artifacts/joint-audio/20260925-current-preview-constraint-v1.
Exactcommands are the frozen uv run --extra mps corpus.py, native.py and
inspect_lens.py entrypoints. Checkpoints/inputs/seeds/profile requests are the
Card's pinned values; model fingerprints and checkpointbytes remainunchanged.
HardwareAppleM5/24GiB,Python3.10.20/Torch2.11.0,oneCPUthread. No networkduringruns,
TESTaccess,labelchanges,overwrites,failedcaseresumes or increasedproposalbudget.

Corpusguard checks all651admittedTRAIN/VALcharts and683341targetrows, withzero
exclusions underbothcurrentandpreview. Source/cache/metadata identities verified.
It completesin36.592659s; freeze
b32a036ab4efc46d60a35c4f888e73eac9127454ac5fa2e85156dc6fb3dd99d3;
result43c7c01a2b43ba472073247b0be04f9520e3c793a67b0e25570483ad36e37790.
Thecorpus result isnot a universalhumanRHrule or all-mania supportclaim.

Native attempts all64declared cases in288.956843s:62complete and2expected
planning_attempt_limit outcomes. Unlike the previousglobal-stop comparison,
these independentfailures were retained and the remainingcases ran asdeclared.
Freeze17fe384034f4bb52f2fb7bd7f12e8c4e838bb8f7c91a0ace04595dafec0c6873;
result2dcbb708c1861d3d6d58bd2ec52c8bb744fe719fd01a6d964ce7b3a27f4034c6.
Allcompleteexports independentlyreparse; allpublishedrows/coverage replay,
HH/RHarezero, andfull/prefixHsequences exactlymatch the originalI plan.
No source-timing intervention orhiddenfuturetailswereintroduced.

| Cohort | Complete | Rejections | Maximum readiness s | Maximum window s |
| --- | ---: | ---: | ---: | ---: |
| flat/current | 15/16 | 19 | .489542 | .559525 |
| flat/preview | 15/16 | 19 | .477393 | .554887 |
| factor/current | 16/16 | 6 | .510240 | .344342 |
| factor/preview | 16/16 | 3 | .513053 | .336012 |

Readiness means cachedMel throughownonlineH/bodygeneration until8scoverage
and30completerows; it excludes freshwaveform/Mel/import/clientcosts.
Allrejections areemptyallowedrows, not badpublishedpairs. Factorpreviewpasses
thedeclared50%retry-reductioncriterion, fullcoverage andruntimebounds.
Three fewerproposalsinonesmallseededpanel isnotpopulationreliability.

Factorcurrent width/LNerrors1.3196517045994263/3.8221615840880805,
body5.141813288687507. Factorpreview1.30025379371665/3.820844378554703,
body5.121098172271353. Rawfactorbody4.909229802145201; previewincreases4.3%,
within10%bodyandper-componentguards. Herror.8738583548578978unchanged.
Bothflatcontrolaggregatesareinvalid dueonefailedcaseeach, withnosuccess-subset
substitution. Current/preview fullrowbytes match12/16factor and14/16flatcases;
factordifferences areHystericP1/P2,GoodbyeP1,YomiP1; flatdifferencesZenithfallP1
andGoodbyeP1. TheflatP2failedprefixesareidentical.

#### Completed Lens review

All38declaredscopes arecovered:24newgeneratedscopes/all48timepages,8exact
within-comparison generatedsemanticreuses,6verifiedparenthumanreference reuses.
Allactions/articulationpagination andsemantic hashes verified. No generated
scopeismissing inthisfixedqualityplan; thetwofailedflatZenithfallP2cases are
outsideitsflatHysteric/Revenge diagnosticcontexts andremainfailuresinthecohort.
All16humanrecordsretrieved. No listening/player test ornewhumanstylejudgment.

Review a2424ce15df942f121e2daff1de51a74bd5850107ce3c067afb67b3b7abca0a7;
planfe139ad1a670fc907feab3bd07001a1bc7b5e1da1fa6b9a8fc10fd1ddd99592a;
render2619b620eeaea5bd3a5a521cc4fdfe3054a476b07896e0297e6644034d41e9e4;
Lensfreezea0d9c911550f04eb4b872478e426bf5585f1ae4c3d91a7564ab9a7b752ca2968;
bundlebcfb37f4780cfa6891f8f15bdf053bf076884813a54c8e918b5e8ddc373cab4c.
Parentprofile-routingbundlemanifestfa4fee5185909034db52f1f909f4845654f15d267c2411179e983efbb9b9cc81
anditsfilebytesremainunchanged. Renderedowneruses167579667bytes, below2GiB.

Preview retainsindependentLNcapacity: AsItWasP2peak1356msLNspans6H while913msLN
joins/closes, withlaterhandoffs andquadLNsubsettail; AsItWasP1peakhas167/257/356ms
LN023subsettailsaroundtaps. FoolMoonP2 has368msLN with82/166msjoinedroles and
subsetrelease/newLN. HystericP1 has585msLNsupporting8H. ShortLNandfullLNchords,
variable1–4headgrips,ordinaryrepeats and5/6/10/19ms cross-columnHsplits survive.
This isnotuniformsingle-tapfallback. A27msFoolMoonsame-columnrepeat isoutside
theconfirmedHHpredicate withoutbeingcertifiedcomfortable.

Qualityremainsmixed. HystericP2current243–247s hasentering677msLN,824/585ms
overlap andlater179/458mssubsettails; previewhaslesslayeringandallTAPafter245532
inthatcrop. Flatcurrent/preview retainmanyindependentroles there.
YomiP2constrainedpeakisallTAP where rawfactorpeak hada406mshold; this isnot
anadditionalpreview-versus-current effect. Profile2wholeLNfractionsstillweak:
factorpreviewHysteric.164,Take.158,Yomi.094 versusrequested.734. Someprofile1
contexts retainunrequestedLNhandoffs. Numericalguardpassdoesnotfixcontrol or
proveconsistentqualityacrossaudios/seeds. Generatedalternatives neednotcopy
sourcehumanlabels; otherjudgmentsremainunreviewed.

#### Actual remaining failure mechanism

BothflatZenithfallP2policies stopat242310/357796ms with997H. Failedwindow
service.559525/.554887s disprovescomputeexhaustionfortheseobservedcases.
Preview'sfouractualrejectedRquerieshaveallfourcolumnsheld:
248976→H248983gap7;247570→H247590gap20;247580→H247590gap10;
247808→H247810gap2. All15physicalreleasesubsets havezerojointcleanfuture
support. CurrentallowsRbutfailsonthenextH; previewfailsattheRqueryitself.

Attempt0holds start0/1at248770,3at248840,2at248900. Atlasthead248900 the nextH
is83msaway, outsidethe20msrowpredicate. Atleastoneappropriateholdneedstoend
by248962(H−21)fortheexperimentalRHscreen, butactualRarrives248976.
ThephysicalfullholdwaitinglawrequiresRonlybyH−1. Post-eventrowfilteringcannot
repair a waitthat hasalreadyspentitsfeasible release interval.

Allthreefactorpreviewrejections showthesamefullheld mechanism: HystericP1
R217796→H217798(2ms),HystericP2R136299→H136313(14ms),TakeP2R1368→H1380(12ms).
Retriesrecover withinbudget, butdo notmake waiting-time support consistent with
rowconditioning. Capturedactualprefixes/metadataare retainedfor each; no
counterfactualdirecttrajectory ismisrepresented asarejectedproposal.

#### Evaluation and next question

Outcome REFINE: shortnestedmatchingisusefulboundedruntimeprogress, withno
newparameters, zero admittedtargetexclusions andcompletedfactorcohorts.
Noendpoint/default isadopted asafinalplayablemodel. WeakLNrequestrealization,
localorganizationcontrasts andflatreleasefailures remain. No protectedfield
changed andnodeclaredcasewasdropped; allscope/budgetconditionswereobserved.
Curated report docs/research/row_constraint_evaluation.md andlinkedguide are
committed atb7769afd9d7f0106f41b45d03103b719ddf1ca76. Scopedprose links/math/diff
checks pass; no newmodelcode afterthealreadyrecorded79ownerchecks.

NextDesign shouldconditiontheactualRwaitinglaw aswellasrowchoice oncontinued
feasibility, notmerelymove a newrowmask's horizon orassume releaseatnow+1.
A full-hold clearance deadline directlyaddresses theobservedremainingcases,
but older3-held/one-free failures alreadyexist in the path-crossover evidence
(PromQueen17H0/D1, attacks105943/105945/105949). A generalmechanicalwait predicate
would therefore address a morecomplete dependency than a full-held-only patch.

One live primitive: after a committedrow atnativeclockt, compute eachfree
column's earliestattack time from its actual HH/RHclocks. For a heldcolumn,
anexistence witness can releaseit at t+1, giving availability
max(last_attack+20,t+22) understrictHH/inclusiveRH. Assign oneTAP to each
previewedH using the earliest-ready column, thenadvance thatcolumn's readiness
toH+20. Withonlyinitialholdobligations andTAPfuturewitnesses, this tests finite
mechanicalfeasibility; itdoesnotpredictRorchoosemusicaltails. Waitingwithnoevent
movesheld-column readiness later. Thelastwaitthat preservesfeasibility yields
a releaseobligation, which mustbe enforcedbytheactualR/event-and-rowlaw and
recheckedafter everyrow. Anoptimisticwitnesswithoutthatcoupling repeats the
previousmistake.

This is anunimplemented researchdirection, not a newacceptedCard or an
establishedwhole-songguarantee. Provegreedy-readiness correctness/finite-preview
limits, simultaneousH+release semantics, exact21/22msboundaries, nativewaiting
partition/RNG invariants andsource-targetretention before a boundedcomparison.
KeepRh'sexperimentalstatus explicit andpreserve musicalLN/repetition freedom.
No training, modifiedRrun, extrapolatedsafetyclaim orgoalcompletion isjustified
bythenewdirectionalone. Thewholeplayablerealtimesystem objective remainsactive.


### Research refinement: separate the RH preference from confirmed playability

The coupled-wait direction remains open. Its implementation should not silently
breach the information contract: a general R deadline from exact TAP clocks
would change skeleton independence. A more restricted construction may use only
LN occupation and future H capacity, leaving exact lane clocks in row selection.
For F free columns, the first future H with more than F heads in a strict20ms
span identifies a need for an additional released column; under the experimental
RHscreen, that release would be due by H−21. Combining this LN-only deadline
with current/short-row checks and a positive remaining release clock deserves
a separate proof. No release policy or extra skeleton input has been implemented.

Before enforcing that preference more strongly, validate the preference itself.
HH<20 is user-confirmed BAD; RH<=20 is not. The earlier choreography report
explicitly preserves10/13ms LN-jack tail gaps with189ms head spacing as unresolved,
potentially valid articulation. Corpus compatibility does not settle that human
judgment. Also, the original unmasked flat ZenithfallP2 *did* complete under
windowHH/RHscreening (9rejections, rowsSHA
d3358839e81c9e4c146f67837b9c3cf6cd0f884a9bd9290bfd85527933efe010);
its direct chart hasHH0/RH20 androwsSHA
11e487d2e7d5ec08b3ac23a0934ae5d680ef67c4a2dc75c1a077d99700b51d0b.
Later localconditioning changes prefixes and can worsen this finite-budget
outcome. Do not label a base R law universally broken from the new failure alone.

An optional asynchronous user question asks whether RH<=20 should remain a hard
rule or only an inspection flag. It is not an execution approval or a blocker.
Compare both potentials under the existing unmodified sampler while that
preference is unresolved. This avoids designing the whole system around a proxy.

Closest analogue: Alshiekh et al., Safe Reinforcement Learning via Shielding,
https://arxiv.org/abs/1708.08611 (AAAI2018), separates policy proposals from a
specified allowed-action system. Its transferable point is enforcing a stated
property; it doesnotestablish that ourRHthreshold is a desirable beatmap property.
Grammar-Aligned Decoding, https://arxiv.org/abs/2405.21047, also motivates
separating a filtered distribution from the base model. This is an application
and evaluation-design question, not a new formal-control result.
OutcomeTEST for the preference ablation below; defer coupled-wait implementation.

### Experiment Card: release-gap-screen-sensitivity-v1

Revision1; proposed; acceptednone; owner2026-09-23-audio-skeleton-r1-integration.
Standing local research/run/commit authority applies. Baselinecleanproduct
b7769afd9d7f0106f41b45d03103b719ddf1ca76. Previous current-preview comparison
is completed; this is the sole active proposed Card.

Question: does adding the unconfirmed RH<=20 publication screen remove useful
LN articulation or introduce avoidable liveness/organization costs, compared
with enforcing only the confirmed HH<20 rule? The one causal intervention is
screen_release_heads=True versusFalse in rollout_buffered. Both use
row_constraint='none', originalRhazards/rowlaw, the unchanged8swindow/20mshalo/
fourattemptbudget, sameweights/fullcanonicalMel/onlineH/profile/seeds.
Do not mix the new current/preview row masks into this comparison.

Primarymodel is flatmaterializer checkpoint
da08044806e99eae6a6221c06ef8c91c02711361fae68c896c9af01bde61dea2.
The count/layout checkpoint
41ab71640ce9571ac7f40d9f851a51e2ba54c10ba97fc6f9e54dbc54fac76185
is a declared secondary comparison. No fitting, model-code change, newtraining
data or alteredsourceannotations. ExistingAPIalreadyprovidestheintervention.
Inputpaths/checkpoints are pinned in count-layout native-freeze
b7db758c974d266c055c3b4801f62d84e3a63bfff265cb91d0c434b3c994a8b8.
Panel484d173628856a431e034b1dbdacab9173c40f3edf696d1b9a19721aeec70dc9;
banka8973b7f94fde90d9f3cc639eb63e781dd295435622ce79fd54fe244789e6f03;
nativeprior43d130ed6291abdebc3c994e1ad27b833954ef84deed5f9dfcc8422e4a60b5ff.

The priorrawdirect16flatcaseshave2HH/48RH;16factorcases24HH/16RH.
PriorRH-on screenedcomparison is incompleteglobally, so it isnot a fullcohort
baseline. Newpairedcohorts are evaluated whether favorable ornot. SourceH
identities remain originalI; seeds251701..251708 byaudioorder, profiles1/2.
Run64buffered calls:8audios×2profiles×2models×RHoff/on. Handoff/export/replay
and Hhash/prefix checks must hold foralloutputs. Any HH in publishedcontent is
a hard failure. Report everyRHpair descriptively inbothtreatments; RHoff doesnot
relabel thatpair as GOOD. No completion or numericalvalue is invented for
anunattempted/failedchart.

Technicalcandidatecriterion: all16primaryflatRHoff casescomplete withHH0,
cachedMelownH/bodyreadiness<=2s through8scoverage and30rows, andeverywindow<=2s.
Compare failurecount, lostcoverage,rejections andwindowcostagainstRHon. A
positive reliability signal isstrictlyfewerfailedcases, oratleast50%fewer
rejectionswhenbothcomplete. Reportfullcohortcontrolerrors/realizedvalueswhere
complete, notsuccess-subsetaverages. Theyare diagnostics, not a replacement
for inspecting newlyadmittedRH articulation. No NLLselection or short-LNfloor.

Compatibilityguard: for any existingrawdirectparentchart withHH0, RHoff first
proposal should preserve its fullrowSHA andneedzero rejections. Verifythis for
all suchcompletecases, notjustonechosenexample. Weights/checkpoints/ownHremain
byte-identical. Expected fourattemptexhaustion isretained andnextindependentcase
continues; allidentity/replay/nonfinite/HHviolation/globalresource failuresstop
the driver. No extraattempts, cappedcaseresume or hiddenfallback.

Qualitativeplan: inspect fixedflatRHoff/on contexts forFoolMoon167206..169945,
Hysteric243000..247000,Revenge239894..243706,AsItWas123000..127000,
bothprofiles. Inspect everyRH<=20witness inallcompleteRHoffcharts (bothmodels)
with1000msbefore/after, mergingoverlappingwindows. Readalltimepages andcomplete
native-ms actions/LN articulation; enteringholds preservefullhead/endinformation.
Expand anindividual context onlyif needed toresolve a specific structural
judgment, recordingit as a qualitativefollow-up ratherthan changingnumerical
selection. Exactverifiedhuman-reference reusesareallowed; exactduplicates
withinthisreviewcaninherit anactuallyviewedscope. MissingRHoncharts staymissing.

Assess release→head inits actualorganization: priorheadspacing, LNduration,
independentheldroles, grouped/subsetreleases, repeatedgrips, howthequickrepress
fitsentry/exit, andwhether forcedfulloocupancy yields repetitiveorawkward
material. Findrelevanthumanexamplesif a GOOD/BAD judgment isdisputed. Existing
10/13ms LN-jack observations arecounterevidence to a universal ban, not proof
allRHshort gapsaregood. Unclearpreferencesstayunresolved; do not silently
convert source-style tags or a countthresholdinto humanBADlabels.
No listening/playerclaim withoutactualevidence.

Interpretation: improvedcoveragewithretainedplausibleLNrelations supports
treatingRH as a diagnostic while refining the model; clearstructuralbadpatterns
canjustify a morecontextual rule orretainedscreen. A neutral/mixedresult doesnot
justify imposingstrongerRdeadlines justtohitRH0. The usermayclarify theproperty
duringwork; thatsteers futureadoption/design, notretroactivechanges to frozen
comparisonresults. The goalremainsplayability andexpressiveness.

Freshowner artifacts/joint-audio/20260925-release-gap-screen-v1.
Command uv run --extra mps python
artifacts/joint-audio/20260925-release-gap-screen-v1/native.py, followed by its
frozenLensscript. CPU1thread,AppleM5/24GiB,Python3.10.20/Torch2.11.0.
Atmost1800sdriver,90s/30000rowspercase,2GiBnewartifactsincludingLens,
>=2GiBavailableRAM and>=40GiBfreedisk, PAUSE/resourceguards; no networkduringrun.
No overwrite/resume. Freeze source/Note/script/inputs/commands beforeeachstage.
Capture firstfouractualrejectsperchart whenneeded, preservingunpublishedstate.
AllrecommendationsremainREFINE pendingproperty/qualityinterpretation; no
automaticdefaultchange, remote publication orfinalmodeladoption.


## Target population correction: ranked 2–6 star 4K, 2026-09-25

The human owner fixes the first playable population to 2–6 star osu!mania,
including LN release burden. TAP and LN press are attacks. Same-column attacks
around 20 ms are a high-confidence problem; generic LN-jack examples do not
justify relaxing the release screen without examining this population.
Ranked natural arrangements provide the principal empirical reference for
what normal charts at the target difficulty can look like. They are not merely
optional style examples. The formulation's frontier concerns responses to
candidate continuations under a gameplay profile; the implemented three-field
arrangement profile is not that player-response specification.

Card release-gap-screen-sensitivity-v1 revision 1 is deferred, unexecuted.
No RH-off run or adoption occurred. Its proposed local output owner contains no
run evidence. The next active card is the population calibration below.
Existing generated results and their thresholds are retained as originally run.

### Experiment Card: ranked-2to6-action-reference-v1

#### Identity and Authority

- Owning Agent Note: 2026-09-23-audio-skeleton-r1-integration.
- Card ID: ranked-2to6-action-reference-v1; revision 1.
- Status: proposed; accepted revision: none. Standing local research authority
  permits execution; this census remains exploratory.

#### Question and Hypothesis

Do verified ranked 2–6 star 4K arrangements support the observed near-20-ms
same-column head/release relations, and what surrounding structures distinguish
ordinary expressive variation from generation failures? The selected hypothesis
is that near-20-ms same-column demands are absent or rare in this population,
whereas short cross-column head gaps can occur. The result determines whether
release filtering deserves relaxation and what frontier calibration must cover.

#### Analogues and Branch Selection

Use natural reference distributions conditioned on difficulty and LN share,
not scalar NLL or unconditioned global chart averages. The primary external
reference is the official osu!mania ranking criteria, inspected 2026-09-25:
https://osu.ppy.sh/wiki/en/Ranking_criteria/osu!mania . It distinguishes rhythm,
hold/release coordination and difficulty-dependent organization. Its guideline
categories are not numerical star bins or empirical motor response labels.
Alternative branches are genuine ranked exceptions, unmatched source versions,
and measurement/schema artifacts; exact-byte joins and Lens inspection separate
these before model changes. Novelty is none: this is population calibration.

#### Fixed Comparison

- Clean source: b7769afd9d7f0106f41b45d03103b719ddf1ca76.
- Index: artifacts/indexes/beatmap_index_4k.parquet, SHA-256
  2d2814c3b3ec47cd1247e8d50cef555e12ace7eae60a79944102926c142c2d8e.
- Read the index's exact dataset paths and per-set metadata.json. Primary cohort:
  metadata beatmap status ranked, mode_int 3, cs 4, convert false, snapshot
  difficulty_rating in inclusive [2,6], and original source MD5 equals checksum.
  Freeze every source SHA-256, metadata SHA-256, ID, star value and snapshot date.
- Preflight: 14,689 indexed charts, 4,200 set metadata paths; 9,536 eligible by
  metadata, 8,774 exact byte matches, 762 mismatches. Snapshot date 2026-08-05.
  This is identity discovery, not a measured action-gap baseline.
- No learned-model intervention, training or decoding. Measurement intervention:
  replace unconditioned anecdotal quality assessment with a defined corpus census.
- Do not use a timing-anomaly-filtered index or reject charts because they violate
  the tested threshold. Canonical raw source objects retain simultaneous tails
  and heads; report incompatible/overlapping events separately. Unparseable charts
  remain explicit missing evidence with their reasons.
- Source owners: scoped_style_modeling/replay.py for original objects,
  osu_core/difficulty.py for the versioned 20241007 SR calculation, and the current
  Beatmap Lens learning/harness implementation for independent inspection.

#### Evidence and Decision Rule

Primary outputs are chart counts and event counts for HH (consecutive same-column
attacks), HR (each LN head to its own release), and RH (an LN release to the next
same-column attack), separately at <20, ==20 and <=20 ms. Report denominators,
full cohort and star bins [2,3),[3,4),[4,5),[5,6], transition quantiles, source
witnesses, LN proportions, chord widths, occupied-lane/head interactions and
1/4-second peak attack densities. Cross-column head gaps are separate.

This is a census of the frozen local population, not a random sample of all
ranked maps; no iid-note confidence claim. Zero cases or <=0.1% affected charts
is evidence to retain the near-20-ms guard for this initial target, subject to
actual witness inspection. Higher prevalence triggers examination of every
affected chart's exact witness inventory and a bounded set of contexts before
any relaxation; prevalence alone cannot establish good playability. Missing
parses/identity mismatches limit coverage and cannot be counted as clean.

Qualitative check: use Lens on the shortest-gap witnesses of each relation
present, with at most three distinct charts per relation initially; if a relation
has no <=20-ms cases, inspect its nearest boundary examples. Add two controls
per star bin (one low-LN and one high-LN chart, selected by deterministic source
hash within each half of the LN-share distribution), inspecting the local peak
and its surrounding 4 seconds. Preserve complete LN endpoints/entering holds,
exact time spacing, source lines and chart context. Inspect all returned pages.
No new human gold, listening, player test or physiological inference.

As a secondary scope check, measure SR and the same relations in the 32 complete
raw count/layout panel outputs and 62 complete current/preview outputs, verifying
parent result and row hashes. Keep their non-SR-calibrated profile requests
explicit; do not claim matched difficulty controls or causal attribution. Failed
outputs remain incomplete and excluded from whole-chart SR comparisons with
counts reported. Versioned local SR is a diagnostic, not a learned player model.

#### Reproduction and Bounds

- Fresh owner: artifacts/joint-audio/20260925-ranked-2to6-reference-v1.
- Command: uv run --extra mps python
  artifacts/joint-audio/20260925-ranked-2to6-reference-v1/audit.py.
- Freeze the script, clean source and Note revision before reading action values.
  Deterministic sorted corpus traversal, no randomized sampling or fit.
- CPU on Apple M5 / 24 GiB; Python 3.10, explicit mps environment. At most 1,200 s
  for census plus 1,200 s for the generated comparison and Lens stage, <=2 GiB
  new files, >=2 GiB available RAM and >=40 GiB free disk. No network in the run.
- Fresh files only, no overwrite or resume. Stop on identity drift, resource or
  time limits; retain partial inventories without complete-cohort conclusions.
- Primary star definition is exact official metadata at the frozen snapshot;
  local index values are rounded and not the inclusion criterion. Recalculate
  selected Lens witnesses with the repository's named 20241007 calculator and
  report any star discrepancy, rather than silently reclassifying the cohort.
- Main confounders: corpus acquisition coverage, source mismatches, historic
  ranked exceptions, chart-level SR hiding local spikes, repeated song families,
  LN variation, and source chart preferences not being measured player responses.
- Positive result constrains the target and motivates calibrated continuation
  response learning. Negative or mixed evidence refines the property/context.
  Neither result adopts a model or proves whole-song playability.


### Result Log: ranked-2to6-action-reference-v1-complete

#### Experiment and Reproduction

Owning Note 2026-09-23-audio-skeleton-r1-integration; Card
ranked-2to6-action-reference-v1 revision 1. Proposed, accepted revision none;
standing local research execution authority applies. Recommendation REFINE.
Clean measurement source b7769afd9d7f0106f41b45d03103b719ddf1ca76; no learned
model intervention, fit, new generation, changed inputs or runtime policy.
Design Note revision ce01dc6b95ac1c2dc69b7ef82f992e17f21a41da.

Owner artifacts/joint-audio/20260925-ranked-2to6-reference-v1 contains frozen
scripts audit.py, generated.py and lens.py. All ran with uv run --extra mps
python SCRIPT on Apple M5 / 24 GiB, Python 3.10.20, CPU. Census terminal 30121
completed in 133.479541 s, generated audit terminal 33597 in 3.773038 s, and
Lens source renderer terminal 61416 in 2.949078 s. No resumptions or overwrites.
All jobs are terminal; there is no live training or renderer to poll.

Freeze identities:
- Census bd7685153cdf0aab98bd2dc70e7063a31775db49db919b4b1114a003b1227c28.
- Census result 393e813bc02ea0bf7e9b7cd287193846c21d0f688092ec3749c9a71900cf8770.
- Generated freeze c35f2dd64e3e2d3d34c00eee872f874ad160d23cf92b41107e6d2ffdbc775625.
- Generated result 00a120a758dbb837300ea967eca454db4300dca16559369a53f8d5d288c444ef.
- Source Lens freeze 855c78fa3879691974c286d1f93b19d08177db26e4395c4e650686485de9d729.
- Source Lens render 2907536b8d6192323a746f82b70c92a0ac48eb18484ff55b91eff620040126fd.
- Source review cda5a049dfc4e541841d13a56835f497e8738c0c67cd9a2a15b77a27120a25b5.
- Generated witness review 8614cc1d28ff12e56b12d9796cf6762aa32b501fc303d019ac4f2ad3304aa1f3.

The verified index has 14,689 charts. Metadata eligibility gives 9,536 ranked
native 4K charts with unrounded SR in inclusive [2,6]; 8,774 source MD5 values
match their per-beatmap metadata, covering 3,387 sets. The 762 mismatching
versions are excluded from primary ranked evidence, not classified as clean.
All metadata snapshots are dated 2026-08-05. Freeze records preserve each
source SHA-256 and metadata SHA-256. No timing-anomaly or short-gap cleaning.
All 8,774 source files parse, with no overlapping or coincident same-column
objects. Whole source object count is 12,953,523, including 2,272,555 LNs.

#### Results

| Relation | Instances | <=20 ms | Charts affected | Minimum |
| --- | ---: | ---: | ---: | ---: |
| Same-column HH | 12,918,427 | 0 | 0 | 37 ms |
| LN's own HR | 2,272,555 | 8 | 1 | 19 ms |
| LN release to next same-column H | 2,259,510 | 0 | 0 | 25 ms |
| Distinct adjacent H rows | 8,765,631 | 8,960 | 566 | 1 ms |

Four HR instances are 19 ms, four are exactly 20 ms, all in ranked beatmap
2012530 (4.08447 stars). Its two inspected passages have a 156/157 ms head
pulse and a repeated LN-duration progression from 19/20 to 39, 78, 104 ms,
mirrored ten seconds later. This is a rare source structure, not permission to
emit arbitrary ultrashort LNs. Zero HH/RH cases and one HR chart satisfy the
predeclared zero-or-at-most-0.1%-of-charts population criterion.

Star bins [2,3),[3,4),[4,5),[5,6] contain 3238/2906/2047/583 charts.
Minimum HH is 83/55/37/50 ms; minimum RH is 42/32/25/26 ms; minimum HR is
30/25/19/24 ms. Per-chart peak one-second attack count median/P95 is
12/16, 18/22, 24/28, 28/33. These are finite-population observations, not
proposed per-bin hard thresholds or independently measured player capacity.

Reused generated exports: all 32 raw plus 62 complete constrained charts,
with the two constrained flat failures retained as missing whole-chart outcomes.
Hashes and exact source-object-to-row equality verified for every complete
export. Source-selected SR recomputation agrees with official snapshot values
to 0.00000437 stars. Generated outcomes inside the 2–6-star interval:

| Cohort | Charts in interval / complete | HH<=20 | HR<=20 | RH<=20 |
| --- | ---: | ---: | ---: | ---: |
| Flat raw | 14/16 | 2 | 85 | 48 |
| Factor raw | 15/16 | 24 | 16 | 16 |
| Flat current | 13/15 | 1 | 64 | 0 |
| Flat preview | 13/15 | 1 | 58 | 0 |
| Factor current | 15/16 | 3 | 13 | 0 |
| Factor preview | 15/16 | 2 | 15 | 0 |

All constrained HH cases are exactly 20 ms, excluded by the original strict
<20 predicate. HR was not part of that predicate. Earlier reports are retained
with their exact original definition; they are not inclusive-20 or HR passes.
The original requests were arrangement descriptors, not calibrated SR requests.
These subset counts are descriptive, not matched difficulty causal comparisons.

#### Qualitative Evidence

Source Lens revision 22e5c84f5cacb8493bdab5f1d0fdc09c5373dc60 independently
matches all 59,011 notes across 17 sources. All 18 declared/follow-up scopes,
35 time pages and complete action/articulation tables were actually inspected.
The second lost-memory occurrence is the one extra source context. Raw source
readings cover regular chord alternation, irregular cross-column streams,
overlapping holds, independent subset tails, full-LN chords and development.
Astral Empire's 17 ms cross-column pairs are retained positive timing evidence.
I'm kidding releases the needed column 67 ms before its next head while the
other three tails coincide with that head. This is a concrete frontier relation.
No source annotation, human gold, listening or player test was added.

Five generated preview witness contexts were additionally inspected, with all
five PNG pages and complete numerical tables. Flat Yomi yori's 3 ms LN starts
alongside a 111 ms LN with two columns free and the next H 304 ms later.
Factor Hysteric has an isolated 14 ms LN with three columns free; factor
Zenithfall closes a 9 ms LN while the next heads use already-free columns.
These witnesses do not require an immediate capacity rescue. Factor Take's
96744 [03] -> 96764 [23] repeats column 3 at exactly 20 ms.

The first generated_lens.py helper failed before any image was rendered because
render_section returns a dictionary containing png, not a tuple. Its freeze and
partial chart/tables are retained. Fresh generated_lens_v2.py used the corrected
helper call and a separate generated-lens-v2 owner, completing the unchanged
five-scope selection. This is a renderer adapter failure, not a model/census
failure. No reuse of the partial attempt as complete evidence.

#### Plan Conformance and Evaluation

Primary numerical population, definitions, selection and thresholds did not
change. Added source context and generated-witness contexts are qualitative
follow-ups to the observed HR/boundary findings. First helper failure is
explicit; the fresh completed renderer replaces no data. No resource/time
limit was approached; new artifacts remained below 0.1 GiB. Product source
and checkpoints were unchanged throughout measurement.

Observation: the intended ranked population strongly separates same-column
action burden from short global skeleton gaps. Generated charts have HR
failures even at target whole-chart SR, and even without immediate LN capacity
pressure. Interpretation: RH relaxation is the wrong next direction for the
initial target. Include exactly 20 ms in the attack constraint; add LN's own
head-to-release relationship to the response specification.

The implementation explains an objective omission: response_costs penalizes
heads near preceding attacks/releases, but not an early LN release itself.
The consequence feature can expose release age without supplying a calibrated
response objective. A second verified algebraic limitation is that the
count/layout normalizer cancels consequence offsets constant across a count
group, so that route alone cannot adjust the group's total probability.
Neither fact identifies a numerical share of all failures due to R1.

The strongest remaining alternative is inadequate learned calibration despite
sufficient inputs, compounded by teacher-generated history mismatch. The
census does not by itself identify which learned weight path creates each
short hold. Global SR, ranked source normality, arrangement descriptors and
player responses remain distinct. The full corpus audit is not a blind
held-out metric and cannot later be reported as one after research reuse.

#### Decision and Handoff

REFINE: retain RH screening; release-gap-screen-sensitivity-v1 remains deferred
and unexecuted. Develop a coherent frontier response family that covers HH,
HR and RH, with difficulty/organization-conditioned ranked references and
on-policy future R behavior. Candidate effects must influence count/LN choices
as well as layout. Independent LN ages/occupation remain the only row-to-skeleton
feedback; direct audio still conditions both skeleton and row modules.

Do not implement a longer global H floor, remove LNs to pass checks, or impose
a universal long LN floor unsupported by real 21/22 ms source patterns. Do not
call an optimistic earliest-release witness a forecast of actual R selection.
The remaining late-R capacity failure and independent too-early release failure
need separate attribution while designing their shared constraint interface.
An integrated candidate must preserve observed ranked structures, meaningful
LN control and finite publication budget, rather than optimize just zero counts.

Durable report docs/research/ranked_2to6_action_reference.md and links from the
planned guide and row-constraint report are committed locally at product
bcb6f3a4198ac06884c6d3c37e8363a147eed678. Local link existence, exact aggregate
accounting, original strict/inclusive metric distinction and git diff --check
were verified. Only documentation changed in the product tree; no new runtime
test-suite claim, default change, training, remote push or Note transition.
The broader playable-generation goal remains active and incomplete.


### Experiment Card: joint-action-spacing-law-v1

#### Identity and Authority

Owning Note 2026-09-23-audio-skeleton-r1-integration. Card
joint-action-spacing-law-v1, revision 1, proposed, accepted revision none.
Standing local implementation/research authority applies. The preceding census
turn is progress: complete source and generated measurements changed the next
action; no completed process is treated as a pending wait.

#### Question and Hypothesis

Can a small common action-spacing law keep H generation, LN releases and row
selection mutually feasible from BOS to the audio end, while preserving ranked
structural support and the existing musical/control behavior? Use minimum gap
g=21 native ms for HH, own-LN HR and first RH. This implements the selected
initial target's inclusive-20 avoidance, not an individual physiological model
or a complete gameplay frontier.

The selected intervention is a common prefix-viability condition, used in both
likelihood and generation. Head skeleton history limits the fifth head in a
g-wide window; LN-only state and head preview determine necessary release
bounds; complete-row conditioning preserves a realizable future. Filtering
only the final row can lose continuation support. An unlikelihood penalty alone
cannot rescue a sampled R whose every physical row already violates the target.
This is why execution-law coherence precedes a larger fitting budget.

#### Analogues and Alternatives

The closest mechanism is a viability/shield constraint over a sequential policy,
as in Alshiekh et al., Safe Reinforcement Learning via Shielding,
https://arxiv.org/abs/1708.08611. The adaptation here is an exact small event
process with complete known audio and a generated H preview; it is not a learned
motor simulator. Novelty is an application-specific factorization and invariant,
not a new generic safe-learning algorithm.

Unlikelihood training (Welleck et al., https://arxiv.org/abs/1908.04319) offers a
later learned mass penalty for declared bad continuations. DAgger (Ross et al.,
https://proceedings.mlr.press/v15/ross11a) motivates learning on generated states,
but its queried expert actions are unavailable here. Do not claim either method
implemented merely by applying a mask or evaluating source histories. Graded
player responses, corpus-normality learning and on-policy fitting remain open.

#### Fixed Comparison

Clean baseline source bcb6f3a4198ac06884c6d3c37e8363a147eed678.
Checkpoint identities remain flat da08044806e99eae6a6221c06ef8c91c02711361fae68c896c9af01bde61dea2
and factor 41ab71640ce9571ac7f40d9f851a51e2ba54c10ba97fc6f9e54dbc54fac76185,
with the count-layout study's frozen paths. Panel
484d173628856a431e034b1dbdacab9173c40f3edf696d1b9a19721aeec70dc9 and bank
a8973b7f94fde90d9f3cc639eb63e781dd295435622ce79fd54fe244789e6f03 stay fixed.
Corpus/normalization identities are unchanged from the count-layout study.
No new weights or fitting in this card; likelihood parity must be implemented
and tested so a later joint fit trains the deployed distribution.

Baseline raw 32-chart results are pinned by
43d130ed6291abdebc3c994e1ad27b833954ef84deed5f9dfcc8422e4a60b5ff.
The additional HR and inclusive-HH measurements are pinned by census generated
result 00a120a758dbb837300ea967eca454db4300dca16559369a53f8d5d288c444ef.
The old current/preview policies are named historical comparators, not mixed
into the new joint law. Their two failed flat cases remain failures.

#### Candidate Law and Information Invariants

1. H uses only its own previous H times: after four heads, a next head cannot
   precede H[-4]+g. This necessary four-column capacity restriction preserves
   short cross-column timing and uses no row-content feedback.
2. For a post-row state at t, a free column's next head is no earlier than its
   actual attack/release clocks plus g. A held column can release no earlier
   than max(t+1,LN_start+g), and next attack g later. Open holds must still be
   closable by the true audio end. Earliest release is an existence witness,
   not a forecast of the learned R policy.
3. Check one-TAP-per-H realizability through t+2g using those availability
   times. After t+2g all original restrictions have expired; the H-capacity
   invariant supplies the later continuation. Require sufficient preview or
   a known complete H plan; nine heads suffice to cover the horizon under the
   global capacity bound. Verify this claim with an independent event oracle.
4. With F=4-held free columns, find the first future H whose preceding g-wide
   cluster contains F+1 heads; a needed release must happen by that H minus g.
   This deadline reads only LN projection and H preview. If it precedes the
   next H, normalize the first-R law over eligible clocks through that deadline
   and force its final clock. If it equals the next H, that H row may release
   other columns. The true audio end supplies the terminal closure bound.
5. Mask complete row probabilities after count/layout normalization. Enforce
   current HH/HR/RH and post-row viability; the surviving count-group masses
   can change. This does not pretend learned frontier2 itself regains a
   cross-count score canceled by its existing normalizer.
6. Use exactly these physical/profile conditions, time support and forced-event
   semantics in teacher likelihood and native execution. Default gap zero
   preserves all existing behavior and checkpoints. Reject incompatible fixed
   H plans and reference targets explicitly; never retime source notes.
7. Preserve direct full audio to both skeleton and row, separate H/R/row RNGs,
   committed LN-only skeleton feedback, true song-end semantics, and incremental
   LN publication. No full generic replay/tap history may enter the R deadline.

#### Evidence and Decision Rule

Primary: all 32 new native cases complete with zero HH/HR/RH <=20 ms in exact
published replay, under the original eight-second window and four-attempt
budget. No hidden fallback, extra retries, or successful-subset averages.
Support proof/tests do not substitute for real model sampling. Check whole-song
SR, every control component, realized LN fractions and the actual chart shapes.
Each complete cohort's mean width+LN error must be no more than 1.10 times its
own raw baseline; report missing/failed cohorts as incomplete. At least the
existing nontrivial LN layering and varied chord/cross-column forms must remain
in the inspected contexts; complete all-TAP or single-note simplification fails
expressive interpretation even if numerical bad counts become zero.

Runtime guard: cached-Mel own-H/body readiness through 8 s and 30 rows <=2 s,
and every publication window <=2 s, CPU one thread. Fresh audio startup retains
its separate earlier evidence and is not established by this cached-Mel run.
Record any changed H counts/times and their capacity reason; if the raw H plan
already satisfies the bound, require its exact same H hash under identical
weights/audio/profile/seed. All model parameter bytes must remain unchanged.

Qualitative panel: fixed Fool Moon 167206..169945, Hysteric 243000..247000,
Revenge 239894..243706, As It Was 123000..127000 for both requested profiles and
both models; additionally the densest four-second H window of each factor
profile-2 output and both Zenithfall profile-2 models. Merge overlaps, preserve
full entering LN endpoints, inspect every time page and complete action/LN
tables with Lens. No listening/player/gold claim without actual evidence.

#### Procedure and Bounds

Fresh owner artifacts/joint-audio/20260925-joint-spacing-law-v1. Implement the
kernel and focused tests first. Compare analytic feasibility/deadlines to an
independent exact event-sequence oracle on bounded grids (scaled g allowed),
including all columns held, mixed young/old holds, H+R rows, simultaneous subset
releases, dense H clusters, t+21/t+42 boundaries and true EOS. Check symmetry,
no-target-future use, LN-only deadline independence, finite positive support,
training/inference probability parity and chunk/branch/RNG invariance.

Verify source-target admission on the 651-chart prepared corpus and the 17
independently parsed ranked witnesses. Report all exclusions with exact source
relationships. The one ranked HR-exception chart is outside this profile's
support by design; no universal legal-chart claim. For an already profile-valid
source trajectory, excluding any target falsifies the claimed support law and
stops native work until corrected/revised.

Then run 32 unchanged/default raw replays to verify their frozen row hashes,
and 32 joint-law buffered cases on the same eight audios, requests 1/2 and seeds
251701..251708. Commands are uv run --extra mps python OWNER/corpus.py and
OWNER/native.py, then OWNER/inspect_lens.py; freeze each script and the clean
intervention source before its stage. Checkpoint migration adds only config,
no parameter tensors. Model/process config changes follow packaged Hydra
projection and consumption checks; runtime modules import no Hydra.

At most 1,800 s per corpus/native stage, 90 s/30,000 rows per case, <=2 GiB new
artifacts, >=2 GiB available RAM and >=40 GiB free disk on Apple M5/24 GiB.
No network during runs, no output overwrite/resume, observe PAUSE. Stop on
identity drift, nonfinite values, source-valid support exclusion, publication
mismatch, violated action bounds or resource/time guards. Expected native
attempt exhaustion is a retained case failure; continue independent cases and
report the complete planned denominator. Do not train or adopt automatically
from a successful mechanical comparison; use actual structure/control outcomes
to choose the subsequent joint fitting question.


#### Joint spacing implementation checkpoint

Card joint-action-spacing-law-v1 revision 1 remains proposed, accepted none.
Clean implementation source 7329b0ea93262717fb5d1475a308e6e562cf4a11 adds the
shared positive minimum_action_gap_ms law; default zero preserves the old law.
It changes no parameter tensors. Positive gap requires conditional release
waits and at least nine lookahead heads, with explicit exclusion of separate
row correction modes. Training/Hydra projection, warm-start metadata and saved
model settings carry the field. The inherited frontier2 native-opportunity
feature remains its documented optimistic approximation; the actual shared
spacing support is evaluated separately after full row-count composition.

Focused verification: 96 unique planned-owner tests passed in 10.29 s, command
uv run --extra mps --group dev pytest -q tests/research/planned_audio_continuation.
This includes CPU/MPS likelihood/gradient agreement, an independent exhaustive
native-clock release/TAP oracle on small grids, finite-horizon feasibility,
LN-only release deadline input, interval/gradient partition agreement, actual
native vs teacher row scores, chunk-invariant draws, rare-event conditioned
releases, forks, buffered publication, EOS, and actual fitting/checkpoint reload
with the setting. All test handles are terminal: 68245, 96623, 91844, 95068,
82377 and 35296. Do not restart or poll them as pending experiments.

Early verification failures were test-fixture issues: scripted publication
sessions lacked the newly required model config, row-score comparisons included
padded rows and a mismatched fixture Mel, and a two-row invalid-source fixture
lacked the legacy helper's required post-seed H. Fixtures now preserve the
actual tested behavior and check all real emitted rows; no assertion or
product safety condition was relaxed. Final 96-test run is the evidence count,
not the sum of repeated overlapping checks. Source support and native real-model
comparison remain unexecuted at this implementation checkpoint.

The baseline-to-implementation diff matches the declared single common law;
there is no fitting, alternate data, new audio encoder or changed checkpoint
weight in this step. docs/research/joint_action_spacing.md defines its bounded
existence argument, probability semantics and limits. Local diff and affected
relative links were checked. No remote push or lifecycle transition.


#### Completed joint spacing evidence and next exploratory prototype

Joint spacing source 7329b0e: 651 prepared charts admitted; all 32 new whole-song
cases completed without retries or HH/HR/RH <=20 ms. The 32 unchanged controls
reproduced their prior rows. Maximum cached-Mel readiness/window service was
0.523/0.281 s. Flat width+LN error improved from 6.334 to 5.672; factor changed
4.909 to 4.960. All H streams stayed unchanged. Lens covered 50 paired scopes
(28 fresh pages, 36 prior semantic reuses) and four additional boundary pages.
Rich LN layering survived, but 21-ms repeated chords and weak LN control remain.
Result owner: artifacts/joint-audio/20260925-joint-spacing-law-v1; review.json
ab7e169a505edf2c7c3d263ad9375be71351a9c50cdeba6f49449c78dfdb7e9e.
Decision REFINE; no playability or adoption claim. All runs are terminal.

Experiment Card typed-resource-plan-v1, revision 1; proposed, accepted none.
Standing user execution authority covers this exploratory local implementation
and fit. User requests faster architecture/training iteration with less repeated
verification. Baseline: clean 7329b0e, flat da080448 checkpoint, prepared corpus
4ad9abfd manifest. One structural intervention replaces separate H/R/count
selection with a typed time/TAP-count/new-LN-count/release-ID plan. An anonymous
four-resource state plans recovery and active holds; R1 consumes plan preview,
full audio and scoped controls to assign columns. The event representation is
an adaptation of timed note-on/off models such as Performance RNN, with native
1-ms timing and an explicit resource/geometry factorization, not a novelty claim.

Hypothesis: moving counts into the plan removes downstream count/time conflicts
while retaining real short cross-column events, LN coordination and chord forms.
HH>=37, RH>=25, HR>=21 ms is the initial empirical envelope, not a comfort model.
A prior scan admitted 650/651 prepared charts (614 TRAIN, 36 VAL); the excluded
TRAIN source c68d6450ce97b8149069b7069736c75624f100040e09198f6ebc5921a3f9c3ae
has three HH<37 ms. Controls have timestamped scopes, masked style fields,
whole-chart difficulty labels and scoped LN fractions. Difficulty labels are
weak local supervision, not local star ratings. Style controls require actual
scoped annotations; an untrained input slot is not successful control.

Bounded procedure: a small learning run, then at most 1,200 joint updates/one hour
on MPS, native CPU rollouts on existing audio with low/high and scoped controls,
and Lens inspection of dense/LN/transition contexts. TRAIN groups remain separate
from VAL; alternatives remain separate charts. Fresh owner
artifacts/joint-audio/20260925-typed-resource-plan-v1. Save checkpoint, losses and
actual outputs; no remote publication. Keep a failing/underfit pilot as evidence,
not permission to claim success. Primary practical check: completed trajectories
without support dead ends, preserved nontrivial LN/chord forms, and measured
control response in the intended direction. NLL is only a learning diagnostic.
Native runtime and 2-6-star coverage are reported; no source or style-quality
label may be inferred from a low loss. Stop on nonfinite loss or one-hour limit;
inspect observed failures before adding further checks or scaling. This is a
joint architecture pilot, not a controlled attribution of every new component.


Typed resource pilot source ca3dc65 completed 1,200 MPS updates in 412.8 s,
614 TRAIN/36 VAL charts, 3,895,879 parameters. All 15 native cases (three unseen
audios, five control modes) completed; all HH >=37 ms. Difficulty 3/5 at LN=.2
produced Hysteric 3.010/4.729 stars, Zenithfall 4.625/5.080, As It Was 2.574/2.972.
LN=.2/.7 at difficulty3 produced .195/.217, .143/.173 and .397/.458 respectively.
Scoped 32-second revisions preserved published rows. Style supervision overlaps
only 11 TRAIN charts. The 20-step Lens review inspected four pages and complete
action/articulation tables; it found preserved layered LNs plus uncomfortable
37-ms repeat under two held columns. Final-model Lens review is pending.

The fixed-history condition probe on three VAL prefixes changes requested LN
fraction .2 -> .7 but expected mark LN fraction changes only .013-.019.
Changing the request scope from full-song to 16 s changes this by <.005.
This identifies weak direct conditional sensitivity before free-running feedback,
not solely a rollout failure or scope-duration mismatch. NLL alone is inadequate.

Experiment Card typed-ln-control-base-v1 revision1, proposed, accepted none.
Under standing local authority, test one semantic parameterization change: factor
marks into (head count, release IDs) and conditional LN count, with a binomial
base at requested LN fraction and bounded learned within-group residual. Mask
that LN request from the residual inputs. The conditional mean LN count is then
monotone in the request at fixed audio/history/support; other mark-group masses
stay fixed. This does not guarantee exact scoped percentages after state feedback.
Compare the same 1200-step weights with/without this factor, then at most400
joint updates/10 minutes, same corpus and native controls. Primary: stronger
LN control in all three songs with preserved varied LN forms and completion;
reject a gain obtained by monotonous full-column holds. Inspect dense/transition
scopes with Lens. Keep all original evidence. Optional training scopes include
whole-song alongside 8/16/32 s to cover both supported request uses; record this
as an additional fitting change, not isolated evidence for the factor itself.
Fresh outputs under the typed-resource-plan owner. No adoption or remote push.


Explicit LN-base source 27b9526 completed 400 MPS updates in 146.95 s; frozen
1200-step weights were also tested without fitting (Hysteric LN=.2/.7 yielded
.069/.751, with stars2.999/4.641). The fitted 15-case native cohort completed
without dead ends and retained HH>=37, max cached-Mel startup/window .381/.344s.
At D=3, LN requests .2/.7 yield Zenithfall .085/.748, Hysteric .102/.791,
As It Was .171/.739. Active32s switched scopes yield .703/.835/.752, following
32s baseline scopes .100/.123/.204. Published prefixes remain unchanged.

Retain the new conditional LN-count mechanism, but REFINE the system: D=3,
LN=.2 now yields stars5.202/4.097/3.168 (Zenithfall/Hysteric/As It Was), so this
is not an adopted playability improvement. First-candidate Lens coverage is
7 scopes/14 pages; final high-LN review is3 scopes/6 pages, all numerical tables
read. Layering/subset tails/handoffs remain, but Hysteric introduces a21ms LN
near249284ms and Zenithfall has29ms double LNs near306221ms. Across the high-LN
Zenithfall output,173/3005 holds last<=40ms, median120ms. Source-relative local
release demand and difficulty coupling are unresolved. No listening/player test.

All run processes are terminal (73113 failed on empty-interval float64, fixed;
52381 smoke2,72703 native smoke,49047 joint fit,47764 native200,48384 native600,
21236 native1200,61914 fixed-condition probe,67660 unfit LN base,62105 final fit,
41839 final native). Four focused tests cover resource/geometry continuation,
scoped field semantics and conditional LN-group masses/monotonicity/gradients;
actual joint fitting supplies MPS evidence. No broad repeated test suite.
Standalone weights exported as candidate-typed-v1.pt and candidate-ln-control.pt
under artifacts/joint-audio/20260925-typed-resource-plan-v1, with configuration,
normalization buffers, style vocabulary, recovery profile, and source revision.

Next research question: a difficulty request must constrain the joint action
response of time/counts/releases/geometry, not only increase average note rate.
Investigate the observed short-LN additions and song-dependent density first.
The single previous-event clock can reset head phase on releases; this is a
hypothesis for testing, not an established cause. Explicit last-H clocks and
source-calibrated local attack+release burden are concrete next primitives.
Do not treat a larger minimum gap, more training or lower NLL alone as success.
Human style/audio overlap also needs expansion before style-control claims.
No ongoing process or overnight automation was left running; goal remains active.


Scoped demand investigation, proposed exploratory Card scoped-demand-reference-v1
revision1, accepted none. Previous turn made progress: typed planning, measured
scoped controls, completed candidates and Lens evidence. Current source929138b
is clean. Under standing authority, derive native action/release window facts
from byte-admitted ranked2-6 charts intersected with the existing TRAIN catalog;
VAL/TEST charts are not fitted into the reference. Use8s windows with4s hop,
including entering holds, releases and same-column intervals. Compare the three
native audio cohorts against conditioned distributions, not only their global
SR or head counts. A local star label is not assumed; all reference stars label
whole source charts. Emit measured features and reference quantiles under
artifacts/joint-audio/20260925-scoped-demand-v1. CPU, at most10minutes, no model
mutation, no overwrite; stop on parser/nonfinite failure. Decision: choose a
specific demand/time dependency to change from observed deviations. This is an
empirical chart-response reference, not a validated physiological simulator.
A separate-H history/clock and an empirical local demand budget are competing
next mechanisms; do not claim either established before a bounded comparison.
Analogues: Neural Hawkes Process (Mei/Eisner, arXiv1612.09328) for interacting
event intensities, Anticipatory Music Transformer (arXiv2306.08620) for temporal
conditioning on known controls. These analogues do not validate our demand law.


Scoped reference completed in47.55s:6,924 existing TRAIN-ranked charts and252,167
nonempty8s windows, uniform group/chart/window weighting. Whole-chart D2.5-3.5,
local LN>=.5 has median head/action rates7.25/11.875 per second and release-duration
median196ms; q95 within-window fraction of <=40ms releases is zero. Final typed
high-LN candidates have median action rates20.75/19.062/15.625 and median-release
durations120/128.25/175ms (Zenithfall/Hysteric/As It Was). This motivates treating
the extra release load jointly with head rhythm, not only controlling LN count.

Experiment Card head-owned-clock-v1 revision1; proposed, accepted none, standing
local authority. Clean baseline929138b; initial candidate-ln-control.pt (1600
updates). Selected intervention is explicit ownership of the head process:
separate63-head temporal memory and last-H clock; resource availability features
encode remaining wait only (already eligible is zero); H probability is factored
before conditional R-only probability, so an increased R logit cannot directly
renormalize H probability. R/marks keep complete typed history; LN occupancy
and recovery still condition H. Direct full audio and scoped controls feed both.
This is one coupled ownership redesign, not attribution to each subcomponent.
Motivation: mixed63-event history shrinks in musical time as releases increase,
and signed past availability deadlines expose arbitrary canonicalization age.

Compare800-update unchanged continuation with800-update ownership model under
identical sampled windows/seeds, same explicit LN factor and dataset, no encoder
scale-up. CPU/MPS; <=15minutes per arm. Native three audios D3/5,LN.2/.7 plus
scoped changes; fixed seeds and unchanged requests. Assess SR calibration,
source-relative action/tail/repetition profiles, completion, latency and selected
Lens peak/LN contexts. Keep variation and nontrivial LN forms; do not accept
silence or all-TAP collapse. NLL is diagnostic. Stop nonfinite learning; record
failures. Fresh owner20260925-head-owned-clock-v1. If the redesign does not
improve generated structure, do not scale it blindly: the remaining mechanism
is explicit demand-conditioned planning/selection, with the conservation law
releases = new LN heads + entering holds - exiting holds across a scope.


Matched800-update fits completed: baseline281.36s, head-owned342.50s. Both15-case
native cohorts complete. Static12-case star MAE1.235 ->.979, LN MAE.0516 ->.0675;
max window.374 ->.439s. The head-owned cohort has752 LN durations<=40ms versus
374 in the unchanged continuation. Do not adopt the redesign from its lower SR
error. Both over- and under-dense trajectories remain.

Post-hoc frontier attribution and fixed-program R1 rerendering completed on all
24 static cases. Original row laws reproduce exactly before intervention. The
counterfactual samples only minimum-cost legal rows, where cost counts heads
within80ms of prior same-column attack or release. This80ms probe is not a new
universal BAD label/hard generation profile. It preserves every typed event,
head/LN count and LN lifetime. Baseline HH<80:91->66, RH<80:989->523, HR<=40
unchanged374. Head-owned HH<80:191->137, RH<80:1187->684, HR<=40 unchanged752.
Largest absolute SR change is.070. Thus meaningful local response changes can
be nearly invisible to SR; row reassignment cannot repair fixed bad lifetimes.
This is a one-row preference, not a globally optimal response policy.

Next bounded decode probe: difficulty CFG with fixed strength2, using the
already-trained15% optional-control dropout, no new model fitting or parameters.
Mask only stars/value-known fields in the unconditional pass; audio/history/LN
and style controls stay fixed. Guide each normalized clock/mark/row law on its
shared legal support and renormalize. Compare the same static12 cases for each
of the two checkpoints against their existing strength1 output. Up to10minutes
CPU; source and exports remain local. Primary: requested-star error and response
contrast, with LN calibration, source-relative short-tail load and Lens structure
as guards. This tests whether a weak conditional signal can be used at decoding,
not whether NLL or CFG guarantees playability. No strength sweep on this panel.


CFG2 decode completed24/24 static cases, no fitting. Star MAE shared.936 versus
raw1.235; head-owned.836 versus raw.979. LN MAE.0661/.0687. Short-LN rates are
388/9954=3.90% and898/11481=7.82%; raw rates374/8738=4.28% and752/10264=7.33%.
Some low requests remain<2 stars. Maximum guided startup/window .548/.599s,
cached Mel only. Source analogy is Stay on topic with Classifier-Free Guidance,
arXiv2306.17806; categorical extrapolation is an adaptation, not quality proof.

Completed Lens coverage: attribution5scopes/10pages and CFG2 two scopes/4pages,
all complete action/articulation tables read. At212456ms row reassignment replaces
a25ms same-column RH with a different column. Layering remains but23-40ms LNs
are unchanged. The31521..31561 triple-LN group is40ms, followed by heads31661.
Hysteric243000..247000 contrasts four sparse head rows in shared versus six in
head-owned; neither this nor SR improvement certifies overall quality.

Lifetime slack-v2 caps at true audio end (v1 lacked that cap; results happen to
be identical here). Shared336/374 and owned679/752 short holds can reach80ms
while preserving all heads/columns and RH>=25;231/374 and456/752 can do so with
RH>=80. This is feasibility, not an instruction to stretch every hold. Artifacts
live in20260925-head-owned-clock-v1; curated evidence in
 docs/research/scoped_demand_frontier.md. All handles12178,99535,79050,13951,
92960,42926,40418,71291,8533,84903,31465 are terminal. No overnight process.

Decisions: do not adopt the larger head-owned branch on star error alone; do not
choose CFG as the complete control solution. Keep the fixed-program response
counterfactual as evidence that skeleton and geometry need different response
interfaces. Two model contracts now take priority: typed clock inherited
bounded_head configuration without implementing its bound/decay, and the bounded
LN base caps all-TAP probability in a fully feasible three-head/rho=.7 event at
about17%. The latter changes a scoped amount request into a local distribution
restriction and limits realistic pure-TAP passages. Current source confirms
both; their causal share in all failures is not yet established.

Next implementation direction: restore an audio/control/LN base plus bounded,
fading historical timing modulation; replace the per-head bounded-binomial cap
with a scope-aware amount mechanism that retains unconstrained local preferences.
Then expand paired ranked TRAIN coverage rather than continuing parameter-only
variants on614charts. The supplied expert response also favors broader paired
coverage and explicit audio/history separation, while retaining native-ms
hazards/complete rows and deferred LN ends. Do not adopt a fixed beat grid or an
upfront-endpoint object decoder just from these results. Audio-at-LN-birth,
future-head anticipation and response feedback into the planner remain concrete
hypotheses if lifetime errors survive the contract repairs and broader fitting.
Goal remains active; previous turn is progress, not completion.


Experiment Card typed-contract-repair-v1 revision1; proposed, accepted none.
Previous goal turn is progress. Clean baseline24096a1. Standing user authority
covers code, local data preparation and fitting. Repair two demonstrated
representation restrictions before scaling: (1) bounded/fading timing-history
modulation with a persistent direct audio/control/active-LN base; (2) replace
bounded per-head binomial residuals with unbounded learned local count preferences
and an explicit prior-odds tilt for the scoped LN request. The latter permits
locally near-certain pure TAP or LN groups even when the scope request differs.
It is an amount condition, not a guarantee of exact realized scope percentages.

Keep native-ms hazards, whole rows, deferred tails, shared canonical full audio
and the37/25/21 envelope. No fixed beat grid, MERT data or model-size increase
beyond the small base-condition branch. Head-owned variant is not adopted.
Use the pre-binomial1200-update checkpoint as a calibrated raw-score starting
point. Initial history bound4, decay1000ms; masks and active-LN state do not fade.
For LN tilt use logit(request)-logit(TRAIN reference fraction), preserving
head-count/release-mask group masses. Test the expressiveness counterexample,
clock decay, actual MPS fitting, native controls and Lens structure.

First a short learning check, then up to1200 updates/one hour on the650-chart
corpus. Compare against its saved starting candidate, not an inferred NLL target.
Primary remains plausible native structure/control response without bad recovery
or short-LN inflation; report failed and under-dense trajectories. In parallel
inventory the6,924 ranked TRAIN reference charts for actual paired audio. Prepare
broader data only within the existing group split; keep VAL/TEST outside fitting,
pin new inputs once at admission, and reuse canonical existing Mel assets.
Fresh owner artifacts/joint-audio/20260925-typed-contract-repair-v1. Current169GiB
free disk allows a bounded40GiB cache; keep>=40GiB free and stream chart states.
No network or remote publication. Stop on nonfinite fitting or resource bounds.

Repair implementation committed as4adcdd9393a603b3ef164645aeb0db6d6b1d9349.
The initial20-update smoke completed; the subsequent1200-update fit includes
the final distinction between an unspecified LN amount and a known reference
amount. It starts at the raw typed1200-update checkpoint, uses the frozen TRAIN
reference fraction.1703308179279884 and actual bounded-clock constructor flags.
The live fit owner is typed-contract-repair-v1/repair-main; process19977.

Broader canonical Mel preparation completed332.734s,6,923 TRAIN charts across
2,573 source groups, unchanged36VAL,2,665 total byte-distinct audio assets,
20.285GiB of new features. All2,438 new encodes succeeded; one source excluded
for the stated recovery envelope. Admission pins source/rows/audio once. The
larger TRAIN slice overlaps112 human-annotated charts and289 human cells,
compared with11 charts in the previous paired slice. Machine annotations are
not silently promoted to human supervision. No TEST opening or remote push.

Experiment Card ranked-typed-controls-v1 revision1; proposed, accepted none.
Standing local execution authority applies. Question: can broader paired
coverage teach the same small model a joint style/difficulty/LN response without
the short-LN inflation observed in the small-data candidates? Keep repaired
architecture, full-song Mel, frozen input normalization and LN prior, native-ms
support, exact physical resources and R1's direct audio plus skeleton preview.
Baseline is repair-main/step-1200 on the650-chart owner; native results pending.
This is a data/objective expansion, not an isolated parameter-count experiment.

Use the new ranked manifest, uniform source group -> chart ->8s interval for75%
of draws and an explicitly separate human-annotated interval objective for25%.
Correct the population interval weight to1000*interval_count/audio_duration;
the old pilot's inverse sampled-interval duration overweights short final cells.
Annotation-biased sampling remains declared and is not population NLL. Human
labels remain scoped, independently optional and absent only when assessed so.
Compute difficulty on actual source bytes with the same20241007 implementation
used for generated charts. Cache at most24 chart/program states; mmap Mel and
pad whole audio to32s shape buckets with exact real-prefix masking. No real
audio is cropped out of the global encoder. Record this weighting correction
alongside expanded data as a confounder of causal attribution.

Seed251927,2microbatches/update, body LR3e-5, other LR3e-4, fixed optimizer
restart. Run40 learning updates first, then at most6,000 updates/one hour after
the learning and native integration check; save every400. Fresh ranked-smoke
and ranked-main owners, no overwrite. Stop on nonfinite losses/gradients, the
wall-clock bound or an explicit STOP file. Save full constructor options,
optimizer and RNG alongside source/script/manifest/checkpoint identities.

Evaluate against the repaired small-data candidate on Zenithfall/Hysteric/As It
Was, fixed existing seeds: D3/5 x LN.2/.7, plus32s joint changes. Add a targeted
style-change probe on Hysteric using the same control interface, not a separate
style-only generator. Primary is native playability with meaningful conditional
contrast: compare whole-chart star error, local action/release demand, short-LN
frequency and complete Lens contexts jointly. Required regression guards:
all cases complete, no HH<37 or RH<25, unchanged published prefixes, no inflation
in the aggregate <=40ms LN fraction. A lower NLL or star error alone cannot
satisfy this experiment. Scope percentages are measured responses, not hard
quotas; style strength has its supplied multi-label ordinal semantics. Shared
scope clocks do not yet encode separate deadlines for overlapping partial
fields; this limits quota-like interpretations and is not hidden by the API.
Only after native improvement is visible use the remaining five audio-panel
cases as a broader generalization check. No adoption or claimed player testing.

Result log, accepted none: final repaired small-data fit completed1200 updates
in450.573s (19977 terminal). Checkpoint repair-main/step-1200.pt SHA256
1445432c483f4e8713d54d97377ebe1eb8c3fe9577f398f4790d7d6666ccfcf6.
The800-update integration case also completed (30379 terminal). Full repaired
native cohort15/15 completes (58988 terminal), static12 star MAE.738873,
LN-fraction MAE.074523,649/15120 LN lifetimes<=40ms (4.2929%). Switched32s
scopes yield.815/.750/.737, following32s.228/.139/.111. Four selected native
Lens contexts/eight pages and all action/articulation tables are read. An LN
88937..89106ms survives the89000ms condition expiry. Zenithfall still contains
26/29/30/32/34ms holds; the candidate is not accepted as playable.

Broader40-update smoke completed33.965s (28029 terminal), including36VAL
midpoint intervals. Training manifest SHA256
4cea2672387b6293a4da0846be479d8bd9c857e55535dc8143bd11d65b06d2c4;
driver SHA2561cc321e530d0578a49ed1cdb6ef63939d2413cf89f9c6865480c15a2ca0ab9ca.
Population weights, explicit25% annotation objective, fixed normalization,
full-audio padding/masks and24-chart LRU are exercised in actual MPS updates.
No extra model parameters were added. The6000-update/3600s ranked-main run is
LIVE process85912 under caffeinate, sourcec348ca288746560247ab09d2a053fdc34ca63c54
(documentation-only descendant of4adcdd9). At3000 updates/1417s it has seen3129
distinct TRAIN charts; peak RSS3.317GiB and MPS driver allocation5.768GiB are
overlapping counters, not additive. Continue this handle; do not restart it.

The800-update broader checkpoint completes15/15 native cases (52041 terminal).
Static12 star MAE.588019, LN MAE.091063,345/17309 short LNs (1.9932%). This is
mixed progress: fewer microholds overall and better star response, but worse
amount calibration. HystericD3/LN.7 short holds increase24->55 and median
duration170->127ms. The broader switched scopes reach.790/.825/.785, then
.144/.173/.156 in the following32s. Native cached-Mel max startup/window
.524/.397s, measured alongside training; no cold-audio or client claim.
Two new peak Lens contexts/four pages and complete tables are read. The near-
aligned Zenithfall peak preserves varied holds, subset tails and a165ms four-LN
group, but has a34ms insertion. Hysteric's peak contains68/47ms RH recovery in
an outer-column repeat and another34ms inner LN. Aggregate improvement is not
uniform chart-quality improvement. Reviews are under each native owner/lens.

Small-data style probe (71713 terminal) uses HystericD4/LN.2 and a32s style
override: baseline unspecified, Jack, Stream or Tech prominent. All four receive
the same update/scope-clock/RNG-invalidation procedure, including an empty-field
baseline span. Adjacent-head same-column fraction is.178/.276/.149/.133 across
the scope; these descriptors are not Foundation labels. Initial119000..123000ms
inspection is sparse and inconclusive. Follow-up chooses105000..109000ms by
maximizing the minimum head count across the four conditions, within the32s
scope (25/25/25/26 heads). All eight contexts/16pages and their tables are read.
Immediate structures remain very similar, with only isolated Jack repetitions
and altered LN layering in Tech. Reliable prominent-style control is unproven.
Human TRAIN positives are sparse: only5 prominent-Tech and8 prominent-LN cells
in the broader admitted cohort. Do not claim that112 charts provide dense labels
for every organization. No annotation mutation or player test occurred.

An exploratory fixed-prefix diagnostic completed11.344s (75661 terminal), no
fitting. At one middle8s interval per existing36VAL chart, fix direct full audio,
teacher history, exact resources, whole-song scope and requestedD3; vary only
rho.2->.7. Mean conditional H probability changes.004450->.004547 in repair and
.005208->.005278 at ranked800: +2.18%/+1.34%. Ranked800 direct-base H log-odds
change-.001466, while its bounded-modulation path changes+.020181 even with
history fixed, because that path also reads current controls. This is a direct
condition-path response, not itself an estimate of changed-history effects.
Native heads rise2102->3327 on Zenithfall and1900->2998 on Hysteric. The two
operating distributions differ; do not infer a causal percentage from the gap.
It motivates examining generated-state feedback, including releases, occupation,
and temporal phase, instead of only amplifying the existing LN input.

Remaining structural questions: a finite63-event planner with exact LN state
but no scope count ledger cannot distinguish prefixes with identical recent
events/occupation and different early-scope LN allocation. It can learn an
average response to rho, but cannot condition compensation on a forgotten
scope deficit. If explicit scope accounting is tested, it must derive only
from the skeleton, distinguish LN-head proportion from held time, preserve
crossing obligations, resolve the LN field's own scope, and roll back with the
unpublished planner. This is a candidate mechanism, not yet implemented or
shown to improve playability. Difficulty and style cannot be reduced to that
additive count ledger. The release conservation law and the weak direct-clock
response also motivate a joint demand-conditioned clock/frontier, rather than
a quota-only fix. Finish the broader bounded fit before selecting another
architecture intervention. Do not revive the rejected head-owned variant or
the restrictive binomial factor just from a scalar metric.

Curated control semantics and evidence committed asc348ca2 and4142ba4; model
implementation remains4adcdd9. No remote publication. Mid-training2400-update
native and matched style cohorts are running as49061 and97281; process85912
is the sole continuing trainer. All other handles named in this result log are
terminal. Final6000 native/control/Lens evaluation and any broader five-audio
generalization check remain required; the goal stays active.

The2400-update native and style processes49061/97281 are now terminal. All15
native cases complete. Static star MAE.488256, LN MAE.077713,256/12515 short
LNs (2.0455%). HystericD3/high-LN improves55->10 microholds and127->224ms
median duration versus800; requestedD3 yields3.563 stars. Zenithfall high-D/high-
LN still has154/2828 microholds and121ms median. Continue the bounded fit;
do not treat this intermediate as a final selection.

The2400 style comparison retainsD4 andrho.2. Its actual32s LN fractions are
.156 baseline,.064 Jack,.088 Stream,.081 Tech. Adjacent-head same-column
fractions.193/.295/.361/.309 do not demonstrate semantic selectivity. Four
matched105000..109000ms contexts/eight pages and every table are read. The
Stream condition repeats the outer pair and right-outer column; Jack adds a
four-note chord and an isolated repeated right pair; Tech remains close to
baseline. The same3/4ms cross-column stagger is present in every condition,
so it does not establish selective Tech control. This is evidence of imperfect
joint target adherence, not a demand that musical properties be statistically
independent. No whole32s style label is inferred from the4s visual inspection.

Primitive for a possible next amount-control intervention: for an active LN
scope defineH(t) as generated heads andL(t) as generated LN heads since the
scope began. B(t)=rho*H(t)-L(t) updates byrho*h-l for a new mark(h,l). These
facts come entirely from generated skeleton marks; no row embedding, TAP-column
history or target future is required. For a source-derived scope rho=L(end)/
H(end), B(end)=0 algebraically. Supplying B,H and remaining scope duration to
the predictor restores information that a63-event history can forget. It does
not imply enforcing an exact integer quota, forcing last-second corrections,
fixing style or treating NLL as a playability objective.

The closest analogy is desired-return-conditioned sequence modeling in
[Decision Transformer](https://arxiv.org/abs/2106.01345). Only the idea of an
explicit remaining objective transfers; this system has chart/audio examples,
not an offline-RL reward dataset, and no new Transformer is required for a
small exact counter. This is a provisional adaptation of a standard primitive,
not a novelty claim or an adopted model. Style and difficulty have non-additive
structure and should not be silently encoded as this same count discrepancy.

Scope ownership must be resolved before implementing that primitive: style-only
changes must leave LN accounting attached to its own active request; overriding
an LN amount must not create compensating debt for the excluded high-LN interval
after the earlier value resumes. A candidate is a new accounting episode on
effective LN-amount entry/return, while keeping earlier published choices and
held obligations intact. It differs from a strict quota over the original broad
interval. Prefix/scope accounting, training and native rollback must use the
same convention. No ledger, auxiliary loss or inference feedback controller
has been added during the ongoing data-only fit.

Trainer85912 remains live, last observed4200 updates/1979s,3860 distinct charts,
peak RSS3.317GiB and MPS driver5.791GiB. Finish at6000/3600s, inspect the final
native cohort, and then decide between unchanged learning, exact scope state,
and a demand-aware planner/frontier. Broader test audio and final style response
still require evaluation. All published evidence remains local.
