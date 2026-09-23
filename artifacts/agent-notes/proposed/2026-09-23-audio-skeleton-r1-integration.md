# Agent Note: Audio skeleton generation and R1 integration

Note ID: 2026-09-23-audio-skeleton-r1-integration
Status: proposed
Kind: research
Created: 2026-09-23
Updated: 2026-09-23
Product revision: e2bca3e592800c1a49a1a82b31ae488fa7021e47
Scope: Released R1 sensitivity to candidate timing; audio-conditioned skeleton learning and downstream integration
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
intervention commit in the result before execution. R1 checkpoint SHA remains
4b3ec1561e33d0ebe2756cfe13571ec414fd5bb470b430f0c578545863115f70.
The new model has 2,950,458 parameters, including a 463,392-parameter Mel encoder;
2,444,688 parameters copy from R1. Omit seed residual, landmark memory, candidate
consequence and 906 future-query projection columns. Retain exact state and the
511-row finite content encoder. No pretrained audio model or style teacher.

Data: canonical frontend fresh at artifacts/joint-audio/20260923-v1, based on
old pilot manifest d416a1953bb4f0126c084457cd8d6c597c96533da9f40d8e3245949006af934c
and catalog e31b7e8f4daa044503ef2b8411bc41727ba371eec804c9462a4608be8112ad28.
Keep 48 TRAIN and 12 VAL base songs, admit up to two distinct TRAIN arrangements
per exact audio/group, no TEST. Hash-verified rows beyond decoded audio end fail
rather than silently changing VAL. TRAIN normalization weights unique audio once.
The initial fit selects six deterministic TRAIN groups, one per density/LN
stratum, keeps their alternatives distinct, and samples group then chart.

Procedure: canonical prepare with the packaged joint_audio Hydra mode=prepare,
max_seconds=900. After representation/integration tests pass, run mode=train
run_name=memorize-v1 fixed_train_queries=32 train_groups=6 updates=300
validation_every=100 batch_size=8 cpu_threads=2 max_seconds=900, other packaged
defaults. Seed230923; shared/new lr3e-4, inherited lr3e-5, AdamW decay.01, clip1.
BOS/event-prefix/absolute-time/outro query probabilities .08/.70/.17/.05;
horizon4000ms, native1ms hazards, ten outputs per absolute10ms bin. Initial
untrained and updates100/200/300 score exactly the same diagnostic queries.

Primary diagnostic: mean joint NLL per fixed TRAIN query; separate event/survival
NLL and mean action NLL for noncensored targets. A reduction of at least20% in
joint NLL is a learning-path gate, not statistical proof or playability. Neither
component may become nonfinite. Report VAL on four fixed queries per each of12
songs descriptively; no held-out quality claim from this unblinded cohort.
Baseline value is measured at update0; no previous model has an equivalent
likelihood. Paired queries remove query-sampling variation, but one training seed
and very small memorization pool remain strong confounders.

If the gate passes, generate native BOS samples from the best diagnostic
checkpoint on six TRAIN cases, then inspect exact event/hold behavior and Lens
source/render patterns. This diagnoses rollout mismatch before data scaling;
memorized-query success alone never authorizes a quality claim. Timings may be
very poor away from the fixed queries. A wider random-query learning run requires
an explicit subsequent Card revision/result-grounded choice under the same
standing authority.

Guards: native rows strictly increase on integer clocks, never all-empty; no
head on held lane, no release without hold, true terminal closes all held lanes.
Censored query boundaries never force closure. Source endpoints/future timing do
not enter predictor inputs. Tests cover within-frame multiple events, long rests,
BOS0, crop/full audio parity, mirror, gradients and scheduler partition survival.
Native output requires canonical mechanics, export/reparse and Lens admission.
Qualitative checks preserve possible Tech, dump/Jack and independent releases;
no global minimum-gap or repetition ban is introduced.

Run bounds: Mac M5,24GiB,MPS, CPU2 training, up to900s per preparation/fit, fresh
run directory, no overwrite/resume, max30,000rows/song and900s generation bound.
Stop on PAUSE, nonfinite loss/gradients, availableRAM<2GiB or freedisk<40GiB.
No network/pretrained downloads. Save resolved Hydra, flat config, corpus and
checkpoint pins, parameter-transfer report, RNG states, update/resource logs,
validation records and explicit completion/stop cause. Raw artifacts stay local;
code and owning Note are committed separately with no remote publication.

Environment deviation before the run: existing SciPy1.15.3 binaries had malformed
Mach-O TLS zero-fill offsets rejected by macOS27. Eight extensions received only
section-offset metadata repairs and ad-hoc signing. Every file-backed section
remained byte-identical; versions/lockfile/canonical frontend unchanged. Original
binaries, script and receipt are preserved at
artifacts/joint-audio/20260923-v1/runtime/scipy-dyld-repair/receipt.json,
SHA6dcf6178d89f22e0a563c2a9fc699c5b6a1c5f92a90b26807eed431825ee5ffe.
Canonical data tests and four real/complex PROPACK SVD smoke checks succeeded.
This is a local environment repair, not shipped model code.

Interpretation: a successful bounded fit warrants evaluating native errors and
broader paired training, not retaining this architecture by default. Failure to
fit directs gradient/representation analysis before scale. Good likelihood with
poor native outputs directs investigation of feedback and multimodal choices.
No sampled sequence is declared BAD solely for disagreeing with its source.


### Execution start: joint-mel-hazard-rows-v1

Clean intervention source:09b919cdeab90e3856fee03a9198d59b4dc527af. Added only
the joint research owner, packaged schema, focused tests and scoped design-doc
update. Prior R1 and audio-pilot code unchanged. Local checks cover96 owner tests, one package-layout test
and22 package-resource subtests, with one test fixture corrected while the
initial combined test process was running; its final focused parity test passed.
An MPS CPU/float64 conversion bug was found and fixed before commit. These are
contract/implementation checks, not generation quality evidence.

Preparation launched with the recorded command and clean source. It writes
canonical features freshly; the old experimental Mel cache is not reused.


### Exploratory Result: joint-mel-hazard-rows-v1 memorization

Preparation completed in20.99s with121 TRAIN charts across48 groups and12 VAL
charts across12 groups. ManifestSHA
4b995029a5344569d4506ff6b11249f61585d2bf7649285754340909bb06c21b.
Thirty alternative candidates had different audio bytes, five had unreadable
paired audio and one failed the inherited admitted-cache minimum-seed rule.
All60 base songs were retained. New generation itself requires no source seed.

The six TRAIN groups contain14 arrangements (3,3,3,3,1,1). The seeded32-query
diagnostic happened to contain32 events and no censored examples; this limits
its ability to demonstrate learned silence/long-rest survival. Survival/state
mechanics were tested separately.

The300-update fit completed in74.14s. Fixed TRAIN query jointNLL9.6190→0.001238;
timeNLL5.87993→0.0009102; rowNLL3.73911→0.0003279. Fixed VAL jointNLL10.4942→
45.3490; timeNLL7.1513→38.2366; event-rowNLL3.5657→7.5866. Thus the gradient
path and representational memorization gate passed while generalization strongly
worsened. This is expected evidence of overfitting, not a playable candidate.
Best/last update300; chosen by TRAIN diagnostic score. Best checkpointSHA
cd935cb876b0716b34459425fcfd27314971c6ceaf1cc14cb038cb05e2bfa42c.
Observed MPS driver allocation1.31GB, RSS~0.98GB, availableRAM~7.2GB.
No resource guard or nonfinite failure. Raw freeze/evaluations/update logs and
checkpoints remain under artifacts/joint-audio/20260923-v1/training/memorize-v1.

Native six-TRAIN-case generation launched using the explicit checkpoint pin,
deviceCPU, one thread, seed17+case index,900s aggregate bound,4s query chunks,
no source rows. First complete case0e5557107b9f covered278.23s in35.04s with
4083rows/4114heads. Its8s coverage took1.793s, stepP99 was12.50ms, including
whole-song learned encoding but excluding decode/Mel/cache verification. About
272/1000heads followed a same-lane attack/release within20ms; Lens inspection
is pending to characterize actual organization. No threshold alone is a BAD
label. This case is a diagnostic overfit rollout, not a final quality result.

Current research recommendation: REFINE through randomized paired-chart training
if native inspection confirms sampling/coverage failure rather than a broken
state/probability contract. Do not enlarge the encoder before that comparison.


### Native completion and next bounded comparison

All six memorization native TRAIN cases completed in140.60s total, CPU1; every
case passed exact mechanics/export/reparse.8s cached-Mel coverage ranged0.554–
2.169s. Their same-lane <20ms relation counts ranged196–333 per1000heads. This
consistent local pressure and the held-out likelihood regression reject using
the32-query fit as a candidate. Lens inspection runs separately on these actual
outputs; no numerical relation threshold is treated as universal playability.

## Experiment Card: joint-mel-hazard-rows-v1 (revision 2)

Owning Note and standing authority unchanged; proposed, acceptance none.
Revision1 is completed exploratory evidence above. Revision2 keeps the exact
representation/model and pinned canonical corpus, changing only training exposure
from32fixedqueries to fresh group→chart→query sampling across all48 TRAIN groups
and121arrangements. This is a necessary full-distribution baseline after the
memorization check, not evidence for a larger architecture. Baseline source and
intervention source are both09b919cdeab90e3856fee03a9198d59b4dc527af; no code
changes. Same R1 checkpoint and initial random seed230923. Same12VAL groups and
48fixedVALqueries; their untrained baseline jointNLL10.4942 (time7.1513,
event-row3.5657), descriptive single-seed evidence.

Exact command: uv run --extra mps python -m
ensomi_model.research.joint_audio_continuation.hydra mode=train
run_name=random-v1 fixed_train_queries=0 train_groups=0 updates=2400
validation_every=400 batch_size=16 cpu_threads=2 max_seconds=1200.
All other schema defaults and optimizer/objective/query mixture remain fixed.
Fresh output training/random-v1; no resume/overwrite. Bound1200s, unchanged
2GiBavailableRAM/40GiBdisk/nonfinite/PAUSE guards. Paired fixed48VALqueries compare
update0 and selected checkpoint; fresh48TRAINprobe queries are descriptive only.

Learning gate: VAL jointNLL improves at least10% over update0, with both time and
rowNLL finite and neither >10% worse. This is a development learning gate, not
a model quality metric. A failure warrants investigating objective/data fit
before more parameters. A pass proceeds to native BOS generation on the same
six TRAIN songs (seed17+index) and six VAL songs, identical sampling and900s
aggregate bound percohort, pinned best checkpoint, CPU1. Preserve all output
rows, timings, mechanics and source-render context.

Primary adoption evidence remains native organization: inspect opening, dense
passages and independent-release passages using Lens, compare the actual local
relationships to source/human examples. Long constant-spacing/jack organization,
complex fractions and asynchronous releases must remain available; global
sparsity/regularity is not a goal. Generated density differences alone remain
ambiguous. Compare rates of suspicious local relationships descriptively and
review concrete examples rather than declaring a threshold-based win.

This comparison changes exposure and batch size, so it cannot isolate a single
optimizer effect or prove architectural superiority. It asks whether the chosen
small model begins to learn a transferable joint distribution from available
paired charts. Even good source-conditioned NLL can coexist with bad generated
history. New architecture, memory and BeatThis remain deferred pending that
failure attribution. No human playability acceptance implied.


### Lens review of the completed memorization outputs

Two of six outputs were reviewed through actual frozen Lens calls and viewed
time-proportional renders. Report:artifacts/joint-audio/20260923-v1/lens-review/
memorize-review.md; identities/traces:memorize-review-identities.json. Bundles
memorize-case1-v1 and memorize-scars-v1 retain204 historical human examples
unchanged; each has25 traced harness calls, all145 manifest files reverified.
Generated and source charts passed the canonical Lens bridge with0diagnostics.

Imaginary Waltz generatedSHA50f9a7c1f7f515e816e75d7803af1e8eb5036ee7337b6bf4f227402c93d5d21a
has36attackrows in4600–4900ms. Column1 repeats4836→4846→4856; columns0/1
repeat8846→8849, withcolumns2/3at8847. Scars generatedSHA
e9a40e97129fa21b4474c7ec9ccc84054f2e2dbc6e38faeb6de0671917702d5e
repeats the same columns0/1chord2717→2719 andcolumn1at21325→21326. These
repeated1–10ms individual-key demands reject both inspected samples as playable
candidates. They are not merely dense, irregular, Jack-like or different from
the source. No assertion is made that the architecture inherently requires this
failure. Repeated3/7ms spacings and10msrecurrences are observations, not a causal
claim about binning.

Scars source has staggered LN/tap control absent from the sampled generated
scopes; this is missing observed organization, not a blanket requirement to copy
LN fraction. Human short-LN and independent-release examples were opened again
as guards against indiscriminate sparsification. No listening/playtesting or
whole-chart visual acceptance occurred. The root also viewed both the Imaginary
Waltz300mszoom and its8500–10000ms dense image.

Revision2 random-query training started from the same pinned R1 initialization.
Atupdate400, fixedVALjointNLL10.4942→6.40574, time7.15130→4.69706 andevent-row
3.56572→1.82260; both components improve. The2400-update bounded run is ongoing.
No model-size, architecture, checkpoint-resume or dataset change was introduced.


### Exploratory Result: revision2 randomized paired-chart fit

Completed2400updates in989.40s, no resource/nonfinite stop. Best fixedVALquery
jointNLL was6.07104 atupdate2000 versus10.49416 atinitialization (42.15%lower);
timeNLL7.15130→4.38610 andevent-rowNLL3.56572→1.79727. The paired development
learning gate passes. Atupdate2400 VALjointNLL rose to6.46048, so the selected
checkpoint remains2000, SHA
52191e0095f0efc8bc0bc0f3f87765f6606e78d38188cefe32b5d4054542829f.
The fixedTRAINprobe at2000 has jointNLL4.73108, time3.29967, event-row1.56154.
One seed,48VALqueries and unblinded12song development cohort give no uncertainty
estimate or general quality claim. Outputs:training/random-v1. Native same-six
TRAINcohort generation started at generation/random-v1-train with this pinned
checkpoint, CPU1, original seed17+index and900s aggregate budget.

The artifact-only short-gap diagnostic replayed32time-spaced anchors per stream
for the memorized ImaginaryWaltzcase. Source and generated prefixes are separately
replayed fromBOS; no source priming or future endpoints. CPU1,3.51s complete;
mean P(next-event<=20ms) was.3059 after source events and.4479 after generated
events, with medians.0010/.0265. At generated absolute clocks, the actual source
prefix gave mean.4407, similar to generated. The contexts differ in history,
recency/occupancy and selected anchor distribution; these are not causal effects.
The finding rejects assuming that the memorization model is well-calibrated on
all human histories and fails only after generated feedback. Full provenance at
diagnostics/memorize-v1-firstcase-time32.json; scriptshort_gap_hazards.py.
The CPU diagnostic ran for3.51s during MPS training; native latency measurement
was already complete and was not taken during this overlap.

A separate source/sampler audit found a localized startup/rest coverage weakness.
The4000ms event-prefix branch cannot reach39/130988TRAINtargetrows (.0298%):
18late first events and21later long gaps. In the actual38400randomdraw stream,
late-start arrangements receive19positive and525censored BOShistoryqueries;
six late-start arrangements receive no positive opening target. Only26of39
long-gap targets are observed positively (46positive exposures total). Overall
BOSexposure is mostlypositive, so aggregate diagnostics hide this subset. The
48VALquery probe includes no positive long-gap transition. This is coverage
weakness, not lackofmodel support orfutureinputleakage. A next comparison should
cover event-ending and preceding censored windows explicitly. Current baseline
was not modified. See diagnostics/query-coverage-random-v1.{json,md}.
