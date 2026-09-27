# Agent Note: Controller semantics and an executable native qualification gate

Note ID: 2026-09-27-controller-semantics-and-native-qualification
Status: proposed
Kind: research
Created: 2026-09-27
Updated: 2026-09-27
Product revision: 14ee1534fbffcd6e809482f9ce748383358cb5a1
Scope: LN amount feedback ablation, recovery-profile corpus coverage, and a packaged native qualification runner
Related: 2026-09-27-playability-regression-evaluation, 2026-09-27-head-base-native-replay, 2026-09-27-audio-memory-joint-fit

## Priority and observations

Previous goal turn is progress: completed input/component probes and14 native
base-substitution outputs; curated measured failures and added reusable waiting
diagnostics. All previous handles are terminal. Full playability is not achieved.

The user supplies an expert critique prioritizing LNfeedback semantic alignment,
composed H/R/R1 support, current-policy state coverage and an executable quality
gate. Proceed with those priorities before further architecture expansion.
The feedback's citations are local document references, not independent code
verification. Current source confirms the concrete projected LNcontroller:
offset+= (rho*actual_heads-actual_LNstarts)/8, clipped to[-2,2]. This favors
prefix balance although a scoped total request allows nonuniform local texture.
Controller semantics are confirmed; causal responsibility for texture failures
requires the paired experiment below.

The five-H-in40ms example is already excluded by next_head_earliest: after
[0,10,20,30], the next H is at least60 with HH60. Existing spacing tests include
independent exhaustive native-clock release/TAP continuation search, separate
HH/RH/HR, current TAP recovery in row_release_window and missing-preview failure.
Do not pretend this support constraint is absent or claim a proof of every
expressive row pattern. Corpus support changes at60/50/50 and60/50/40 still need
explicit coverage figures, beyond the previously published20ms census.

Memory's matched384 fit has now actually failed native gates. No further size
increase or unchanged joint fit is selected. Fixed-prefix improvement is not
current-policy whole-song improvement. The qualification runner must execute
complete generation and preserve failures/witnesses and independent ranges.
Prototype/interface tests cannot supply a candidate promotion verdict.

## Deferred H architecture branch

Before this steering, a cardinality/location H factorization was considered:
for short elapsed-time windows, predict the number of H timestamps from full
audio/controls, then predict their native-ms locations using H history. Counts
would mean timing rows, never R1-owned simultaneous heads, columns or LN types.
This could remove event-by-event positive feedback into activity amount while
retaining timing-history dependence. It is not implemented and no fit is running.
Potential costs are window-edge artifacts, independently sampled activity,
latent count state across scope changes and expensive conditional normalizers.
It remains deferred until controller/qualification work supplies evidence.

Analogue search consulted the primary papers Professor Forcing
(https://arxiv.org/abs/1610.09038), ADD-THIN
(https://proceedings.neurips.cc/paper_files/paper/2023/hash/b1d9c7e7bd265d81aae8d74a7a6bd7f1-Abstract.html)
and HoTPP (https://arxiv.org/abs/2406.14341). They motivate comparing teacher/free
dynamics, whole-sequence event generation and elapsed-horizon forecasting,
respectively; none proves Ensomi needs those exact algorithms. The count/location
idea is an adaptation of elementary probability factorization, not a novelty claim.

## Executable qualification work

Promote the already used artifact qualifier's behavior into reusable product
code under research/gameplay_evaluation. Provide a typed Hydra startup boundary
and canonical packaged YAML. Runtime receives ordinary dataclasses/paths, not
Hydra objects. Supported checkpoint families initially controlled-audio/v1 and
controlled-audio-memory/v1; no default model or interface promotion.

The runner must verify pinned plan/checkpoint/audio/Mel bytes, generate from BOS,
apply declared live control changes, reparse exports, evaluate named scopes and
actual publication traces, and persist failed/incomplete cases. Numeric guards
are independent of pending Lens/semantic review. Successful computation is not
automatic playable qualification. Plans retain declared thresholds and required
review contexts; no aggregate score cancels a scope failure. The runner must
explicitly expose LNfeedback on/off and record the policy identity.

Use the runner for the LN ablation rather than producing another standalone
generation fork. Focused tests must exercise the actual model/export/evaluation
entrypoint and ensure a numeric failure or missing semantic review cannot become
a promoted result; config options must reach runtime. Avoid broad unrelated tests.

### Experiment Card: ln-feedback-scope-ablation-v1

Revision: 3
Accepted revision: none
Execution authority: user's ongoing research/experiment/local-commit request,
reinforced by the expert-feedback task. Exploratory, with no inferred acceptance,
Note lifecycle transition, adoption or remote publication.

Question: at fixed weights/audio/H, does projected prefix LNfeedback improve
whole-scope amount while harming local TAP/LN organization or recovery?
Baseline: actor128 SHA364c7b711bc39ab3384175b755658d2ef67a4c10b166f5eb85dc072d920e10e3
and memory384 SHA7efcbe4bd6da3a4ec9ead01f24fdda63803ea930ebe76bdc7781fbf080a97e81,
reported separately, each with its saved recovery profile. One intervention:
ln_feedback=None versus defaultLnAmountFeedback. Direct neural controls and
the model's analytic requested-rho tilt stay unchanged. No weight updates,
endpoint edits, R1 count/lane transfer, or activity adjustment.

Fixed cases: the four already inspected approximately4-star songs Max Burning,
Classic Pursuit, STYX HELIX, Blizzard Heights, at the existing273100/273110/
273120/273130 seeds and each seed+1; whole D4 and original source LNfraction,
styles unknown. Add the existing Zenithfall switch seed272304, D3/rho.2 before,
D4.5/rho.6 on[64000,96000), then restoredD3/rho.2. Two policies x9 cases x2
checkpoints=36 complete outputs. On/off H sequences must be exactly equal in
each pair; verify actual native H rather than force them to source times.
Pin a prepared JSON plan before the first run; original source scopes remain.

Observe per-scope absolute LNfraction error, held occupation, .25/.5/1s actual
recovery, TAP/LN-rate contrasts at.5/1/2/4/8/16s, source-anchored local type
organization and audio correspondence. No single contrast magnitude is rewarded.
Keep seeds, checkpoints and requested ranges separate. A replicated local
organization/recovery gain in at least two songs, alongside worse total amount,
supports a controller tradeoff. No such gain, or mixed directions, limits that
claim. Require matching actual H, complete legal exports and preserved published
prefixes. Report all original latency, difficulty and LN guards even on failure.
No inferred human labels or blanket removal of the controller follows.

Before native execution commit the runner and pin its OID/plan/hashes. Native
budget2400s across all36 cases,300s per case, CPU one thread,8GiB process bound;
fresh output per checkpoint/policy, no automatic overwrite/retry. Stop on explicit
STOP, input drift, nonfinite probability, support/export failure or resource cap.
Do not run corpus parsing or training during measured native timing.

## Ancillary recovery support census

Reuse the frozen8774-chart ranked2–6 population from
20260925-ranked-2to6-reference-v1/freeze.json, SHA
bd7685153cdf0aab98bd2dc70e7063a31775db49db919b4b1114a003b1227c28.
Count actual HH<60, RH<50 and HR<50/<40 events and affected charts, by star and
LNfraction strata; distinguish TAP-to-TAP, TAP-to-LN, LN-to-TAP, LN-to-LN and
hold durations. Keep examples and exact native times. These are support exclusions,
not BAD annotations or a new player-capacity law. Check source/metadata hashes;
report any unavailable/changed source without treating it as a clean chart.
CPU-only maximum1200s, fresh support-census output, no model/data mutation. Run
before timed generation. The existing four-H feasibility check and independent
execution oracle tests supply complementary implementation evidence.

The broader goal remains active: expressive playable full-audio H/R/R1 generation
with scoped controls and real-time publication. This work does not redefine success
as a runner passing, a sparse chart, or a better single proxy.


## Revision2: ancillary parser adapter correction

The support-census attempt is terminal/failed before its first chart because
iter_source_objects expects text lines, not a Path. Preserve that empty attempt
and failure receipt. Correct the adapter to UTF-8-sig source lines, consistent
with the existing census, and use fresh support-census-v2. The population,
thresholds, source hashes and1200s bound remain unchanged. This is not model
or quality evidence. Main LN ablation plan is unchanged and not yet run.
Corrected census script SHAf81b05deee4d0626fb22b3523d2a594c778e3438ddd4548ae16b20368ce7780d.


## Revision3: frozen qualification handoff and completed support census

Census60605 is terminal/complete:8774 sources/metadata hashes verified,51.31546s,
no missing/changed files. 60/50/50 affects558 charts (6.36percent),60/50/40 affects
420 (4.79percent). By2–3/3–4/4–5/5–6 stars the affected counts are9/72/281/196
versus5/33/207/175; denominators3238/2906/2047/583. At5–6 stars these are33.62
and30.02percent. HH<60:571events/38charts; RH<50:3743/329; HR<50:13930/403;
HR<40:2420/150. Local exclusions are not synonymous with whole-chart training
rejection or BAD labels. Existing10 spacing/oracle tests pass unchanged in1.02s.
Curated finite-support argument and coverage now live in joint_action_spacing.md.

Runner source226725c5838dab2ef7f381741079ea7c35bb4dc7 is committed. Seven focused
qualification tests pass (8.13s), including actual controlled/memory generation,
exports, live scope changes, numeric versus pending-review status, complete
continued evaluation after numeric failure, incomplete-case rejection, pressure
bounds/witnesses, profile/H-capacity consistency and an actual Hydra CLI run
returning2 on a numeric failure. Package-layout test also passed22 subtests.
Earlier runner fixtures incorrectly combined fixed H with a live update; the
runtime rejected it. Correct fixtures use a real seeded H generator. No failure
was waived, and the qualifier explicitly rejects unsupported fixed-H/live updates.

The frozen nine-case plan SHA is4ab74f18ece7b747a9410ca259f7e2a882c379744795228365f96ce99b621e02;
models ledger SHA99f48ecc0591f13887781101c49cdba773d5816a7b62f28071c12bb881863253.
Original actor/memory checkpoint bytes and every audio/Mel hash are verified.
Each of four songs has seeds0/1 as declared; diagnostic phrase scopes deliberately
have no invented local difficulty/LN target. The switch has separate actual
requested ranges. Each checkpoint keeps its saved recovery profile. Primary
intervention and interpretation criteria remain as Card1.

Use `uv run --extra mps python artifacts/joint-audio/20260927-controller-semantics-v1/run_ablation.py`.
The supervisor calls the packaged Hydra qualifier four times, sequentially, into
fresh actor-feedback-on/off and memory-feedback-on/off directories. Exit2 from a
complete numerically failed candidate is retained as data; incomplete execution
stops the study. Overall2400s, percase300s,8GiB and no concurrent workload remain.
State/config lives in ablation-run-v1. No fit is live, no architecture expansion
is implemented and no model is promoted. Card remains proposed, acceptance none.
Supervisor SHA39259dc5c7a6cc8ed553228cf9c405cce73ea046bf4274edc36c82bf0219e752.
