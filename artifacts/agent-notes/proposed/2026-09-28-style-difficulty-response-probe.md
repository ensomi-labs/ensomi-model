# Agent Note: Style-conditioned difficulty response and neural calibration

Note ID: 2026-09-28-style-difficulty-response-probe
Status: proposed
Kind: research
Created: 2026-09-28
Updated: 2026-09-28
Product revision: 0f5ec10b34c43fe21a6143d371a2f2bc36070e47
Scope: Fixed-model native style/difficulty coupling on two audios, without changing player reference
Related: 2026-09-27-full-row-history-observation-learning

## Motivation

The complete full-R1 fit improves ordinary Classic/STYX amount gates and reduces
both Stream seeds' attack excess, but four-star style requests on Zenithfall
produce approximately 5–6 stars. All three new style guards use that audio, so
audio-specific difficulty and style conditioning are confounded. The missing-view
arm is not a stable improvement and is not selected as this diagnostic parent.

This tests a systems-control hypothesis, not another decoder architecture:
whether the declared neural difficulty code remains a useful actuator, while
independent requested gameplay references remain fixed. Inverse response
calibration is an ordinary control primitive, not a novelty claim. No calibration
mapping is adopted from two audios.

## Experiment Card: style-difficulty-response-v1

Revision: 2
Accepted revision: none
Execution authority: standing user goal authorizes local model/system research,
native experiments and commits. No adoption or remote publication is inferred.

Frozen model: full-R1 step128, SHA
32afd48e5dbad6714401735e01200963c9d447e9f8ca92057bf6c7b67ab36406;
source 0f5ec10b34c43fe21a6143d371a2f2bc36070e47, a documentation-only descendant
of baseline 13eefbc67d3e7668910efc612e2f35ba50a30798. Keep weights, full audio,
recovery 60/50/50, user stars 4, unspecified LN and feedback OFF. Both seeds per
audio retain their existing frozen values. Compare three conditions on Zenithfall
and Classic: style unspecified; Stream prominent; Stream prominent with only the
neural star code offset by -1 (normalized-control coordinate -0.5).

Reuse the two completed unshifted Stream Zenithfall outputs. Generate the other
ten cases. Classic's previous ordinary cases had known LN fractions, so they
cannot serve as these unspecified-LN baselines. The neural offset is applied
explicitly at H, R and R1 logit calls only when the star-known bit is true.
Controls supplied to empirical recovery preference, scope checks and pressure
reference remain the user's 4-star request. Do not claim this is the same as
asking the entire system for 3 stars. Record the projection in case plans and
the outer run identity; no checkpoint weight is changed or re-exported.

Primary descriptive contrasts, by audio and seed: style-on minus style-off
achieved official whole-chart stars; shifted minus unshifted Stream stars;
H count, head count/composition, LN amount/occupation and attack excess under
the unchanged 4-star reference. A repeated style increment of at least .5 stars
favors style/control coupling as a contributor. A repeated neural-offset decrease
of at least .5 stars shows usable actuator response; it does not establish a
general calibration. For the two Zenithfall seeds, an absolute error at most .5
with no pressure increase is a promising diagnostic only, requiring scoped
Lens and style review before a quality interpretation. No pooling across audios
or seeds can hide an adverse result.

Confounders: removing style changes the requested distribution; a smaller star
code can trade density or style expression for difficulty; history and H are
allowed to change as mediated outcomes; two audios do not identify a universal
mapping or the entire 2–6-star curve. If both style-free and styled outputs remain
high, inspect audio/timing response rather than blaming style alone. If the
offset does not produce a meaningful/healthy response, do not preserve it by
loosening evaluation or adding shape filters.

Fresh owner artifacts/joint-audio/20260928-style-difficulty-response-v1.
Pin plans, script, checkpoint and source asset hashes before execution. Use the
packaged qualifier through an explicit process-local neural-control projection;
the standard qualifier does not silently implement this extra plan field.
CPU one thread, 900 seconds total, 180 per case, 4 GiB task-footprint bound.
Stop on STOP, source/hash drift, incomplete/illegal generation or resource bound.
Preserve all failures and outputs; no automatic retry or promotion. Existing
publication and below-20-ms checks remain. The realtime benchmark is unchanged.

## Revision 2 execution clarification

The source advance contains only the requested formal diagnosis document;
model, sampler, evaluator, tests and dependencies are unchanged. The projection
is an explicit artifact-owned wrapper, not a checkpoint or product modification.
prepare.py pins three plans and run.py/projection.py hashes before execution.
The original neural control tensor is cloned; only known-star coordinate zero
changes by offset/2 at H, R and R1 calls. Invocation statistics retain the
actual known input/output value ranges, and each case verifies original D4
controls and unchanged recovery in its evaluation identity.

Zenithfall seeds are 271200/271201; Classic seeds are 273110/273111. The three
generation arms contain four unspecified-style, two new unshifted Stream and
four shifted Stream cases. Two unshifted Zenithfall Stream cases are reused.
No hypothesis, metric, guard, budget or adoption condition changes. Acceptance
remains none; execution uses the standing user research authorization.

Frozen plan SHA:
625114fdfe3a8f43d1fe4606a1b21963de89f2d742c4af3a2afa5146340fb3b7.
Preparation verifies the checkpoint, both audio/Mel identities, reused exports,
the documentation-only source delta and actual qualifier plan validation.
Projection checks preserve unknown values and every non-star coordinate.
Execution command is uv run --extra mps python followed by the owner run.py.
