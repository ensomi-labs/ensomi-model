# Agent Note: Clean joint proposal learning and initialization dependence

Note ID: 2026-09-28-clean-joint-proposal-learning
Status: proposed
Kind: research
Created: 2026-09-28
Updated: 2026-09-28
Product revision: 0a74a198f73dec42753e0c92fb9c98ee298e4710
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

Revision: 1
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
song-group split. Exclude all five native-panel audio hashes. Prepare the
actual source/control ledger before fitting and verify source/rows/audio/Mel.
No source-label union or generated-prefix/factual-suffix substitution.

Separate training purposes: 50% natural equal-group then chart windows with
independent numeric missingness; 25% difficulty/LN-balanced windows with both
numeric requests known; 25% annotated scopes with at least one genuinely known
style retained and both numeric requests known. Population indices are uniform
over full audio, including empty windows. Numeric controls are 50% whole-song
and otherwise 16/32/64-second scopes with declared full-prefix proxy. Annotation
scopes are not extended. Record actual branch/mask/exposure and rejection bias.
Natural sampling supplies unknown-LN defaults; balanced LN draws do not hide
their LN request. This specifies a learning measure, not an unbiased global
corpus likelihood claim.

Prepare 8192 eight-second draws, paired across arms, batch two and 4096 updates
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

Retain fixed evaluations at initial, 512, 2048 and 4096 updates. Validation
likelihood diagnoses fit but does not select a winner. Native evidence uses the
existing 28-case panel with separate D2/D4/D6, LN/style and live-switch ranges;
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
