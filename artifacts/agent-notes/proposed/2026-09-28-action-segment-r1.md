# Agent Note: Exact action-segment mixtures for R1

Note ID: 2026-09-28-action-segment-r1
Status: proposed
Kind: research
Created: 2026-09-28
Updated: 2026-09-28
Product revision: 33645c91437e39fe13ddfe977655e40828ab7a89
Scope: A replacement R/R1 decoder with shared segment choices and factual LN birth context
Related: 2026-09-28-persistent-arrangement-mixture; 2026-09-28-operational-response-frontier

## Decision and previous progress

The preceding goal turn made progress: five actual generation trials, fixed Lens
inspection, a failed persistent-scalar hypothesis, and joint R/R1 partial-trace
scoring changed the next action. No process remains live. The overall goal is
active and unmet. The user explicitly authorizes architecture changes, local
learning and native experiments; no Card acceptance or remote publication is
inferred.

Implement one concrete alternative: a private categorical plan shared across
an actual-time interval of at most4s; a short local-history nonlinear complete-row
decoder; and factual LN birth context preserved across interval boundaries.
Exact occupancy/clocks and full audio remain available. A longer factual-history
encoder informs the prior once per segment, rather than bypassing the plan at
every low-level decision. The old count/layout/consequence actor heads are
replaced, while independently calibrated continuation response remains external.

With four states, enumerate the conditional R/R1 likelihood for the entire
segment and optimize one logsumexp with the prior. This gives exact marginal
training without a recognition-network approximation or a fresh latent per row.
The conditional decoder includes both R survival/events and complete rows.
Shared audio/H are frozen during the first fit to isolate the materializer;
the final audio-to-chart goal still requires their joint improvement.

Local neural content resets at plan/control boundaries with a TRUNCATED marker;
physical state never resets. Each active LN retains its actual birth row and
origin audio, not an invented future endpoint. This first representation is
birth context, not persistence of its unobserved older latent code. Cross-boundary
hold behavior is therefore an explicit qualification question.

MusicVAE (https://proceedings.mlr.press/v80/roberts18a.html) and ACT
(https://tonyzhaozh.github.io/aloha/) motivate a shared subsequence choice.
Exact occupancy, nonquantized native time and discrete joint actions prevent
copying independent-subsequence or averaged-action execution directly. The
previous whole-song latent, missing-history and MMD failures remain evidence;
this is an adaptation with a different information path, not a novelty claim.

## Experiment Card: action-segment-prototype-v1

Revision: 3
Accepted revision: none

Baseline source33645c91437e39fe13ddfe977655e40828ab7a89 and joint80 checkpoint
b57934728a77abfef0d4bd1ea6387d7cb6f8582c6e450bdd890bfeb1d380ebd6.
Previous ordinary native outputs are unqualified. Compare one-state and
four-state versions of the same replacement decoder, with identical shared
initial weights, data, controls, optimizer exposure and native seeds. This
isolates the shared-plan mixture inside the new prototype; comparisons to the
old actor are compound architecture comparisons, not isolated latent effects.

Use factual TRAIN draws from the frozen8192-example source ledger SHA
a48c9cc55f60fd6353295513f060ac23fd7f5487cf798907eb3eca8b6d716862,
starting at draw5000. Select one segment inside each8s source interval, split
on the4s grid and actual control boundaries. Multiply the original sampling
weight by the number of possible pieces. Preserve full audio, source H and
every true R/row label and no-event clock. Actual raw source rows supply LN
birth context. No changed generated history gets a source suffix label.
Expose both original and style-known/LN-hidden query views with shared weight;
do not turn unannotated style-balanced draws into an unconditional prior.

First freeze concrete data/model/scripts into fresh owner
20260928-action-segment-r1-v1. Profile4updates, then a32-update learning pilot
per arm, four factual microbatches per update. Source/input identities and any
profile deviation must be recorded before further fitting. CPU2 or MPS with
explicit accelerator extra, at most1800s and16GiB task footprint for this pilot,
4GiB saved outputs, exclusive creation, no automatic restart after a terminal
failure. Stop on nonfinite loss/gradients, provenance drift, STOP, invalid exact
replay or resource bounds. A larger fit requires a recorded revision.

Prototype checks: exact segment marginal versus enumeration, complete-row
reflection/support, one plan per declared interval, R and row agreement on
that plan, fork/control-update ownership, and no invented closure or state
reset at plan boundaries. Conditional likelihood tests cover survival on an
open horizon; source-only NLL is not the quality criterion.

Record held-out factual segment likelihood, posterior/prior utilization, and
actual generated organization. Primary native witnesses are actual4star
STYX plus Kimi/Celestial held-role/TAP sources, with their audio excluded from
this fit. Use source H only as a separately reported diagnostic, never as an
audio-only score. The existing Zenithfall5.873star source-H case is not the
primary4star feasibility control. Exact cases/hashes/scopes are pinned before
generation; two independent seeds are required before any positive quality
claim. Keep explicit styles and default conditions separate.

Learning/profiling success permits a larger proposed fit, not model promotion.
A useful quality signal requires actual recognizable TAP/held-role organization
in both seeds, no new short-LN/jack/coordination failure, and separately reported
amount, difficulty, pressure and publication timing. Prior-generated output
must carry the improvement; posterior fitting or forced-code samples alone
do not qualify. The final goal remains complete native2–6star generation with
scoped controls, style diversity and realtime publication.

## Implementation and revision-two frozen pilot

Product7d31b1e797b3ee1a4d4aed551e03375dbb925818 implements the model, exact
segment marginal and branchable execution. There is no learned recognition
network: four conditional whole-segment R/R1 scores are enumerated. The old
count/layout/consequence actor heads are removed from this family. A shared
nonlinear complete-row decoder has two code-modulated layers. A short local
TCN resets with TRUNCATED, while the frozen longer encoder is read by the prior
once per plan. Active holds keep factual birth-row geometry and original audio.
The independent response mechanism remains external.

31affected CPU/MPS checks pass6.03s. They include exact sampling/conditional
likelihood across plan cuts, actual open holds across those cuts, gradients to
prior/decoder/birth context with no H/audio gradient, fork/control ownership,
and full birth-context reflection. Source review caught an absolute-coordinate
birth representation; encoding the birth row in each hand's relative frame
fixed it before fitting. No quality claim follows from these tests.

Revision2 makes the control-visibility partition explicit: hide an LN request
without revealing its old source extent as a reset clock. Each of the two
visibility views selects one piece from its own visible partition and receives
original_weight * partition_count / view_count. The shared8s source sample keeps
its total expected weight. The first four updates are the resource profile
inside the same32-update pilot, not four extra updates.

Preparation-v1 failed before data output due to a duplicate seeds keyword.
The failed script/plan and failure receipt remain. prepare_v2.py completed
with128draws,158views and126charts. Source data SHA
11d7bbae0badacf54900f9cec16a67f0379f30b4ccb3138d29e54966e8a3594d;
preparation plan093317b6dae62d28911a047f29b35973f841dafa841c33d43278a2bd66d5a129.
Actual4star cases are STYX[1800,7800), Kimi[80000,84500), Celestial[8800,19500),
D4 with their actual whole-chart LN fractions and two fixed seeds each. Their
audio bytes are excluded from this pilot, not declared unseen by the inherited
backbone. Case ledger5a6d27afd567324e4e3b63d49561b4d76dcdd4a3c2ad26233eb048d635e52b8d.

fit.py uses MPS, CPU2,32updates and4factual draws per update, both visibility
views where applicable. Paired shared parameters and code0 are identical at
initialization. New modules use3e-4, inherited materializer paths3e-5, release
flow1e-3; AdamW weight decay1e-4 and norm clipping1. Source learning uses raw
actor probabilities with no LN feedback or recovery energy to absorb. Frozen
audio/H/prior-history fingerprints are verified at saved steps4/16/32. Full
audio encodings are reused only because their entire producing path is frozen.
This is a compound comparison to the old actor and a matched1-vs4-state test.
Bounds remain1800s,16GiB footprint and4GiB output; exclusive pilot-v1, no
automatic restart. Fit plan.json pins code and every dependency.

## Completed pilot and revision-three native diagnostic

Fit plan0b931b6eccbfe7725e0e18c505a183551cb89c08c22b3646d6ae4c462a16dda4
ran session70868, now terminal exit0. Both arms completed32updates in146.176s:
158actual segments,3960target rows and160831native R-risk clocks. Frozen
fingerprints and strict checkpoint reloads passed. One-state checkpoint SHA
9c60831599bbda301d75c28c161630e1ec2fbdb0a654149608624443a14bb7d9;
four-state44dd528ced4424969488593c01bb691093296c0e186d94f00a96a482ab2ffc24.
This is a small fresh-decoder learning exposure, not convergence or playability.

The late four-state batches concentrate responsibility on code1: about.988
atstep28 and.969 atstep32. These are batch observations, not global code-use
estimates. Actual prior generation and conditional code scores are needed
before claiming useful multimodality or declaring permanent collapse. NLL
values on different random batches are not a matched learning curve.

Physical footprint rises from4.828GB atstep4 to16.668GB atstep32, peak
16.698GB(<16GiB bound); MPS driver reaches5.983GB. These overlapping counters
do not identify the owner of the growth. The MPS guide was read. Any larger
fit should use bounded process segments with complete optimizer/RNG recovery
until the growth is understood or removed. No unbounded monolithic job was
launched and no allocator attribution is claimed.

Revision3 adds a first actual raw-proposal probe: K1/K4 step32, the three
already pinned ordinary4star sources, first seed only, full BOS-to-audio-end
generation on source H. No quantity feedback, recovery preference, response
guidance or planner alters this actor diagnosis. Independently report pressure
after generation. This is not an audio-only result and cannot support a positive
quality claim without the second seed and actual native-H/accepted publication.

Native plan960514df82b10a9f5f3202cffab41485d29e504d8a3eaa372cb1923e3a81858e
pins native.py, both checkpoints, cases and the frozen action reference. CPU2,
180s per case, six runs at most1080s, fresh native-source-H-v1. Preserve incomplete
rows/open holds and all failures, no restart/overwrite. Measure actual whole
stars, LN amount, fixed-scope hold/TAP interactions, native-time response work,
plan usage and publication latency. Render fixed scopes if materialized.
All other Card conditions and no-promotion boundary remain.

## Completed native output and observer evidence

All six source-H runs completed and reparsed successfully, process14735 exit0.
K1 STYX/Kimi/Celestial stars are4.4671/4.0380/5.0449 and LN fractions
.2795/.2902/.2954. K4 values are4.6237/3.9726/5.0587 and .2451/.3026/.2966.
All85K4 plan selections are code1. No useful prior plan diversity is established;
the code-only long-history/future-audio path is ineffective in these observations.
The direct current audio/H paths remain. Only1of158fit views supervises the true
first source row, so startup exposure is a concrete recipe weakness. These are
hypotheses about current learning, not proof of representational impossibility.

The typed TAP observer is committed as3ea1827b9c638e87b6be89bfae5031ebc8110d9a.
Its87gameplay-evaluation checks passed5.73s before commit. It includes zero-TAP H
in group transitions and retains the origin of a single continuing held role;
no clock cutoff converts those counts into semantic Stream labels.

inspection-v1 rendered30Lens pages. Twelve were actually read: both Kimi source
pages and all generated Kimi/STYX pages for both arms. Remaining Celestial and
other source pages were not reviewed in this pass. K1 Kimi has10successive TAP H
under one held role, but mainly singles and repeated-finger subruns; source has
19interior TAP H/28heads with steady jump/single alternation. K4 Kimi keeps only
a2Hsingle-held body. Both STYX samples have11Hheld-role TAP bodies but short LN
clusters and weak surrounding organization. These partial roles do not qualify
the output. inspection-v1's unreviewed status must not be represented as a
complete review; this paragraph records exactly what was seen.

All six outputs contain21–25ms LNs. At <=40ms the counts are30/16/20 for K1 and
22/11/15 for K4, respectively; strict <40ms counts differ. This is the user's
priority regression. The native pass was raw proposal generation, not frontier
acceptance. Follow-up response analysis belongs to2026-09-28-response-blindspot.
No checkpoint, demo or benchmark branch was promoted.
