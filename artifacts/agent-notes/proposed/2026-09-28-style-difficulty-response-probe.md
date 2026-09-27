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

## Result Log: native-v1 and fixed-history row-code follow-up

Card style-difficulty-response-v1 revision 2 remains unaccepted/proposed.
All ten new cases complete normally in 139.8666s; two pinned cases are reused.
Observed peak task footprint is 656,065,952 bytes. No retry, weight change,
recovery change or benchmark run occurred. Output owner is the declared
20260928-style-difficulty-response-v1/native-v1, cases SHA
88ae0e1bacb38745c96a65f3edd830b4d9b951b35fdf873c32c934af60773709.
Input/output call-site ranges verify 0 to -.5 only for the shifted neural arm,
with original D4 controls and 60/50/50 recovery retained in case identities.

| Audio / seed | Unspecified style stars | Stream stars | Stream neural code -1 stars |
| --- | --- | --- | --- |
| Zenithfall 271200 | 4.918388 | 5.070924 | 4.388294 |
| Zenithfall 271201 | 5.206211 | 5.133506 | 4.586096 |
| Classic 273110 | 4.450135 | 4.365737 | 4.374811 |
| Classic 273111 | 4.452409 | 4.359785 | 4.004647 |

The repeated .5-star style-increment criterion is not met on either audio.
The neural-offset .5-star reduction criterion is met on Zenithfall only.
The promising two-seed Zenithfall quality condition is not met: seed 271200
attack excess rises .0110051 to .0126765 seconds, while seed 271201 remains
.586096 stars from the D4 request despite lower excess .0238275 to .0021776.
Do not adopt a universal correction, infer a full response curve, or shift the
player reference to preserve the result.

Twelve Lens pages were read: both unshifted/shifted Zenithfall 271201 at
[41094,49094) and Classic 273110 at [88589,93589). Repeated-column blocks
remain in Zenithfall despite lower stars/excess; Classic remains TAP/chord flow
without a general quality improvement. Review SHA
134384050320e1e6ed8ff1a74081546d9cb6a3d1c49c9ab43dc2049a663656ec.
These are agent observations, not human labels or listening/playtesting.

A separately recorded secondary read-only diagnosis changes only the current R1
code at the same factual generated prefix, H preview, exact state and D4 recovery.
It scores four unshifted Stream cases, Zenithfall [40000,48000) and Classic
[88000,96000), with 58/63/64/70 H queries. Within the one-TAP/no-release family,
mean absolute repeat-probability changes are .013523/.010015/.013796/.008423;
query maxima .032421/.029989/.029893/.023432. Expected heads decrease by
.039–.045 per query. The four-case record SHA is
165e72c6b706b63c77ca4de66e3502f19630ccd8a8001669e5c0fececc702ea5;
the run completes in 4.6504s under its declared 180s/4-GiB bound.

This secondary probe is exploratory and does not revise the primary Card.
It narrows the mechanism: large uniform immediate routing changes are not
observed in these fixed histories; the native intervention also changes timing,
composition and reached histories. It does not assign a causal percentage to H
or rule out amplification of small local row changes.

REFINE: preserve the neural difficulty input as an available control, but do not
apply the -1 correction as a quality fix. Broader full-R1 factual learning is
owned separately; future demand/selection work must distinguish relational
organization, not only scalar difficulty or excess thresholds. No checkpoint
promotion, Note lifecycle transition or remote publication occurs.

The durable numerical/relational synthesis is
docs/research/full_row_learning_and_difficulty_response.md at product
db4f89603942324b0fcd7ffe27ffd5c11f07b6d1. The whole playable-system goal remains
active. Current evidence supports preserving the broader arrangement candidate
while addressing scoped amount/difficulty coupling and independently evaluated
continuation responses; it does not justify another unchanged source-NLL extension
or a universal neural offset. All jobs and diagnostics for this record are terminal.
