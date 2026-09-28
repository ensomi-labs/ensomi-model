# Agent Note: Clean joint proposal learning and initialization dependence

Note ID: 2026-09-28-clean-joint-proposal-learning
Status: proposed
Kind: research
Created: 2026-09-28
Updated: 2026-09-28
Product revision: ef42095a6e764b0374edbaa36b8ddf87c32364c9
Scope: Direct conditional joint proposal law, separated default/conditional sampling and matched initialization learning
Related: 2026-09-28-coordination-frontier-and-ranked-contrasts, 2026-09-28-head-audio-control-interaction

## Direction and rationale

The preceding goal turn made progress: completed matched-ranked observations,
an exact release-readout collision and a committed self-contained report.
The final playable system is not achieved. No old process is assumed running.
Current product HEAD and the two user-owned outstanding files were rechecked.

The next learning line tests whether the inherited proposal can be rehabilitated
under a broader coherent joint objective, compared with earlier and fresh
initialization. This does not attempt to identify physiological demand from
star labels. The richer continuation response remains a separate missing
function and must not be claimed from this experiment.

Use the existing full-audio H/R/R1 architecture, including direct row audio,
layout modulation and the optional H audio/control modulation. The row model
keeps cardinality, columns, TAP/LN and release subsets. A new optional direct
LN-conditioning mode removes the automatic logit(rho)-logit(reference) tilt:
the count network directly learns its complete conditional count law. Unknown
and known controls remain distinct, and explicit external ln_shift remains an
explicit separate policy input. Old checkpoint defaults preserve their laws.

This removes a strong analytic local allocation prior from the clean training
line. It is not a claim that the prior caused every failure; contextual_tilt
still included it in the earlier comparison. No scope allocation residual or
LN amount feedback is used in this line. Keep the same existing empirical
recovery preference in scoring/generation for now; removing it is a separate
policy experiment. Restored 60/25/21 support stays fixed.

Closest analogue is ordinary conditional maximum likelihood with missing
conditions and deliberate proposal measures; no novel estimator. The previous
FiLM mechanism supplies condition/history interaction, not a quality guarantee.
The principal causal comparison is initialization under the common new recipe.
Comparison with earlier published outputs changes law and data together and
must not be labeled an isolated recipe effect.

## Experiment Card: clean-joint-proposal-v1

Revision: 2
Accepted revision: none
Execution authority: the explicit continuing user goal authorizes local model
changes, bounded experiments, full local resources and suitable commits.
This is exploratory research, not human card acceptance or runtime adoption.

Baseline source is 0a74a198f73dec42753e0c92fb9c98ee298e4710.
The three initializations share the same 128-hidden architecture/config:

- inherited: scoped-LN progress checkpoint SHA
  8898c51714474f82171b570cd2c8867bb07e911a59dfbae91b2642687de6a4ca;
  explicitly omit the unused scope-allocation tensors;
- early: row-owned frontier-2500 checkpoint SHA
  0f1ddfa5b351988fca04246ec080be106855f49b50fb493370c2d5afee1febb8;
  load compatible tensors and initialize missing zero modulation matrices;
- fresh: constructor initialization, seed 280281; preserve the same input
  normalization convention as the factual Mel preparation.

All arms use direct LN conditioning, layout and H modulation, no hold-cue,
player-condition, count-prior or scope-allocation module. Record every imported,
omitted and initialized tensor. This is not bitwise pairing of the initial laws.
Existing frontier2 is an actor residual, not an independently calibrated cost.

Use ranked TRAIN and original human annotations with the existing held-out
song-group split. Exclude all five native-panel audio hashes. Freeze the source
program, seed and inputs before fitting; publish each immutable 64-draw chunk
before any arm consumes it. Verify source/rows/audio/Mel.
No source-label union or generated-prefix/factual-suffix substitution.

Separate training purposes: 50% natural equal-group then chart windows with
independent numeric missingness; 25% difficulty/LN-balanced windows with both
numeric requests known; 25% annotated scopes with at least one genuinely known
style retained and both numeric requests known. Population indices are uniform
over full audio, including empty windows. Numeric controls are 50% whole-song
and otherwise 16/32/64-second scopes with declared full-prefix proxy. Annotation
scopes are not extended. A source scope without any heads has undefined LN
fraction, which remains unobserved rather than being labeled zero or discarded.
Record actual branch/mask/exposure and rejection bias.
Natural sampling supplies unknown-LN defaults; balanced LN draws do not hide
their LN request. This specifies a learning measure, not an unbiased global
corpus likelihood claim.

Prepare 8192 eight-second draws in immutable chunks, paired across arms, batch two and 4096 updates
if the resource pilot passes. Train all audio/H/R/R1 parameters jointly with
AdamW, learning rate 1e-4, weight decay 1e-4, clip one; no old optimizer moments.
Every update re-encodes full-song coarse audio and the complete fine halo with
current weights. H/R waiting-time survival and actual deployed row likelihood
are all scored, weighted by elapsed seconds and the documented interval measure.
Learning rate is the same across initializations; budget curves, not a short
fresh-model endpoint, decide whether initialization remains a limitation.

Before full fitting, run a 32-update resource/learning pilot on the first 64
draws and verify intended gradients, finite losses, exact imports, checkpoint
resume and current probability-path agreement. A failed pilot stops without
automatic recipe change. Fresh model quality is not expected after 32 updates.

Retain fixed validation at initial, 32, 512, 2048 and 4096 updates. Validation
likelihood diagnoses fit but does not select a winner. Interim native evidence
at 512 and 2048 uses eight fixed witnesses: seed-zero Classic/Zenithfall D2/D6,
Blizzard/STYX seed one, Stream Zenithfall 271201 and live control switch.
Final native evidence uses the existing 28-case panel with separate D2/D4/D6,
LN/style and live-switch ranges;
initial/fresh early runs can stop at resource bounds and remain incomplete,
not be resampled until they look good. At the final endpoint compare all arms
and the existing profile-only reference. Inspect H floors, sustained responses,
LN timing/interaction and recurrence witnesses with Lens, including the new
ranked coordination references and their legitimate short-LN counterexamples.

The primary final-system signal is lower native D2/D4 error without worse D6,
LN amount or new sustained/coordination regressions. For a bounded positive
initialization result, require at least .20 improvement in D2 MAE and .10 in
D4 MAE over the other trained comparator, with D6 MAE no more than .10 worse,
LN MAE no more than .025 worse and no new numeric case failures. At least two
inspected organization improvements without a new severe witness are required.
These thresholds do not certify playability or establish canonical response
calibration. Lower NLL alone, earlier convergence alone, or uniform thinning
cannot select a production model.

On Apple M5/24 GiB, use explicit mps/render/dev extras and one CPU thread.
Pilot bound 600 seconds, 16 GiB sampled process footprint. Full fit bound ten
hours, 16 GiB sampled footprint and 24 GiB new checkpoint/output storage.
Workers are serial and retain optimizer/RNG state, with fixed milestones and
receipts. Each native arm is bounded to 1800 seconds, 180 seconds/case and
4 GiB footprint; no accelerator fitting overlaps native qualification.
Stop on owner STOP, nonfinite valid loss/gradient, source/hash drift, support
misalignment, failed receipt or resource bound. No overwrite/automatic retry.
Retain failures and revise the Card for a causal/procedure change.

Fresh owner artifacts/joint-audio/20260928-clean-joint-proposal-v1. Exact
scripts, source OID, inputs, model configurations, commands and seeds are
frozen before launch. User-owned AGENTS.md and the untracked architecture
walkthrough are excluded from edits. No benchmark worktree changes or remote
push. Any candidate still requires exact-loader realtime benchmark integration.

## Revision two: deterministic chunk preparation and implementation

Direct LN conditioning and 27 focused CPU/MPS checks are committed at
ef42095a6e764b0374edbaa36b8ddf87c32364c9. The mode adds no parameter tensors,
preserves old defaults and leaves explicit ln_shift distinct. The checks cover
neural request/context gradients, no automatic amount prior, checkpoint loading,
native/replay scope parity and mirror symmetry. Only the model, focused tests
and its research guide changed; user-owned files remain untouched.

Preparation v1 measured 256 draws in 67.07 seconds and 768 in 199.67 seconds,
projecting beyond its 1800-second bound. Its verified Python process 67606
under tool session 2051 was intentionally interrupted with SIGINT; exit 130
is confirmed. There had been no optimizer update. The original script/plan
and an interruption receipt remain. This was an engineering procedure change,
not an observation timeout or an inferred process failure.

Revision two raises the preparation bound to 3600 seconds and emits immutable
64-draw chunks atomically before any model consumes them. Source preparation
can overlap fitting; it reads no weights, losses or quality outcomes and keeps
the same seed/program choices. All three arms consume each identical chunk in
order. The complete ledger is also retained on producer completion. Pilot
quality does not choose later data. This replaces the original all-ledger-before-
fitting procedure; no claim of unchanged procedure is made.

Original preparation-plan SHA
b6d1970255978260aae23fc8d6fe07d141b21525f4b64a7f1b7140d2030f722c;
v2 preparation-plan SHA
70cd1eda1108a563376d462aaa98d54bc528860c7e9a65ceada8dd4795bc7593.
The first 64 draws are published in 16.21 seconds, SHA
fa5eef63c612a26245dd5dcd99d3e53776a97cb23cc31b45300dda29b2c5c3c5.
Initial validation contains the same 22 held-out windows as before.

Fit plan SHA
9190bcd465923d7460f8e2dac751b2ca4892a59a04450a2a005110a5b369273e
pins common.py, worker.py and validate.py. Each worker consumes exactly one
64-draw chunk for 32 updates, preserves optimizer/RNG checkpoint state and
records imports and gradients. Source guard compares executable paths with
the pinned revision, permitting later documentation-only commits.
Commands use uv run --extra mps --extra render --group dev python followed by
this owner's prepare-v2.py, validate.py 0 and worker.py 0 for the resource pilot.
No whole-run supervisor is launched before that pilot is inspected.

Initial H/R/row NLL per second:
inherited 32.01912594 / 1.31904570 / 11.42042167;
early 32.01912594 / 1.29347655 / 11.95515949;
fresh 45.55102973 / 1.70081769 / 33.27794954.
Macro row NLL is 1.57275295 / 1.64922791 / 4.32740172.
These verify finite distinct initial laws and factual replay, not a quality
ordering or evidence against the fresh initialization.

## Pilot completed; longer comparison authorized to execute

All three 4675633-parameter models finish the paired 32 updates, training all
audio/H/R/R1 paths. Fit plus checkpoint verification takes 113.12 seconds
(optimizer loop 108.45), peak sampled process footprint 4672623464 bytes.
All required gradient roots are nonzero; each saved checkpoint strictly reloads
with exact parameter equality. Current/early import 383/382 tensors; fresh
imports only the two shared Mel normalization buffers. Four scope-allocation
tensors are omitted only from current; missing modulation matrices are zero.

Pilot inherited/early/fresh checkpoint SHA:
82d5b0029324d3806cbdcb7e480743b2d9222316c34818f62221f1f1866971d8 /
8bd51abe79b44ae6e423513740c52ea0af8c82957ded5c2093788436a39a7a18 /
2d8644ac6a7eeec4c79a31f92e760685476d9c837033db522bed8a9dc7aec562.
The matching validation macro row NLL is 1.65113611/1.65161087/3.53933923;
H NLL per second is 31.70371923/31.72703899/45.36095056. Fresh improves
its row diagnostic; inherited row validation worsens despite better H NLL.
Do not describe all metrics as improving or infer playability from this pilot.

The fixed longer execution continues the saved optimizer states from step 32
to 4096, in serial 32-step workers. Source chunks were frozen before use;
producer can continue ahead without reading training results. Final source
ledger and chunk equality are checked after producer completion. Validation
and native runs pause optimizer work at the declared milestones. Numeric
failure is preserved, not resampled or auto-promoted. The full supervisor
has a ten-hour elapsed limit, per-child limits and a 24-GiB owner storage cap.
Native semantic/Lens review remains agent work and cannot be inferred from
the runner's complete flag.
