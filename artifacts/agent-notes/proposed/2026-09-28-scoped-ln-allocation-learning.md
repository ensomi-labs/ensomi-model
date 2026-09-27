# Agent Note: Learned scoped LN allocation with factual progress

Note ID: 2026-09-28-scoped-ln-allocation-learning
Status: proposed
Kind: research
Created: 2026-09-28
Updated: 2026-09-28
Product revision: 3591df29b824cf4340c5425da81c5338cd7c49aa
Scope: R1 scoped LN allocation observations, learned within-family preference, matched factual fitting and native evaluation
Related: 2026-09-28-broader-full-row-learning, 2026-09-27-ln-continuation-preference-diagnosis

## Question and mechanism

The broader full-R1 candidate improves Stream routing but overshoots explicit
LN requests. Exact replay lacks cumulative LN starts; the old clipped feedback
does not retain scope totals and imposes a prefix preference. Test a learned
R1 allocation function with factual progress, full-audio query, row history,
H preview and controls. This is control accounting, not demand state.

Each LN-bearing request retains its identity through partial overrides. Track
both all heads/LNs inside its declared interval and heads/LNs while it was the
effective LN owner. Announced future ownership supplies elapsed/remaining
owned time. These are observations, not new quotas or a change to evaluation
semantics. Style-only overrides do not reset them. Source observations include
only rows strictly before the current query.

The closest analogue is the conditional future potential for a scope outcome
derived in the formal report. This adapter is a small conditional log-odds
approximation, not an estimator proven to equal that potential. It does not
penalize intermediate prefixes. No external semantic labels or physiology are
assumed. A larger joint model and independent continuation responses remain
needed if this missing observation does not explain the control regression.

## Experiment Card: scoped-ln-allocation-v1

Revision: 2
Accepted revision: none
Execution authority: user's continuing goal explicitly authorizes local
research, implementation, experiments and suitable commits. This is exploratory;
no acceptance, adoption or push authority is inferred.

Baseline source: 073b6b68fa59b4be99be093cdf284f8c16e08492. Existing AGENTS.md
and the unrelated untracked architecture walkthrough are excluded. Before runs,
commit the bounded implementation and pin its OID and all script/input hashes.
Baseline checkpoint:
5206b1e4dcbff9820a1e84ecf1ba02aee22c60105cae93d8f648ab6458780ea5,
from 20260928-broader-full-row-learning-v1/fit-v1/496-512/full.pt.

Two arms share the same frozen baseline and zero-output-initialized 64-hidden
MLP. Both see current complete-audio features, row context, preview and controls.
The context arm receives zero accounting features; the progress arm receives
factual accounting. Only this input differs. The MLP learns a scalar LN-count
tilt after neural frontier normalization, conditionally normalized inside
(head count, release count) families. It preserves that neural family mass
and within-full-count layout odds. Subsequent recovery preference can alter
deployed family mass, so no stronger invariant is claimed.

The adapter is active only when LN fraction is requested. Unknown-LN neural
outputs must be exactly unchanged. LN amount feedback remains off in fitting
and evaluation. H/R, shared audio and existing R1 tensors remain frozen.
No shape mask or count ownership is transferred to H.

Training uses the first 512 draws of the existing frozen source plan
fb1d2634f8061c3fb55da2bcd369b9ea51872358dcafe50d8a85b082582c553f,
and the same 22 validation draws. The baseline has already trained on these
source windows: this is a matched new-observation study, not new-data coverage.
All five native-panel audio identities were excluded from that source fit.
Cache only frozen factual inputs and baseline probabilities/preferences; the
adapter and its normalized law are recomputed at current weights on every step.
Do not splice generated prefixes with source labels.

Train 256 updates, batch two, AdamW LR .0005, weight decay .0001, gradient cap
one, seed 280031, identical initialization and draw order. Loss is genuine-source
deployed row NLL per second, averaged over the two windows. Inspect the first
eight updates for finite loss/gradients and effective parameter changes; do not
select the endpoint by likelihood. Terminal validation remains diagnostic.

Primary native comparison: the unchanged six Classic/STYX/Blizzard D4 cases,
two seeds each, requested fractions .217153/.485281/.838046. Baseline outputs
are .360472/.437791, .564394/.696268, .968517/.976295 (full precision from frozen
cases). Require mean absolute fraction error improvement at least .04 against
baseline and .02 against the context arm to support scaling the progress
hypothesis. Six cases are mechanism evidence, not population confidence.

Run the existing twenty-case native plan for each endpoint, retaining all
scopes, seeds, reference inputs and numeric gates. Unknown-LN cases must match
baseline rows exactly, including four Stream cases. No new difficulty, pressure,
spacing or publication failures relative to baseline are allowed; existing
failures remain failures. The override retains its original scoped evaluation.
Any apparent amount gain needs Lens reading of both seeds in Classic
[88589,93589), STYX [1800,7800), Blizzard [40342,46342), plus newly reported
LN/recurrence witnesses. Normal short/long/overlapping roles remain admissible.
No candidate is promoted solely by the primary metric.

Use Apple M5, 24 GiB, Torch 2.11, explicit mps extra, one CPU thread.
Frozen factual inputs are extracted on CPU; only the small adapters fit on MPS.
Factual preparation/fit maximum 1800 seconds and 12 GiB process footprint;
native maximum 180 seconds per case and 1800 seconds across both arms, 4 GiB.
Output owner artifacts/joint-audio/20260928-scoped-ln-allocation-v1, fresh
subdirectories only. Record exact commands before execution. No overwrite or
unrecorded restart; a terminal failure remains in its directory. Stop on STOP,
source drift, nonfinite probabilities/loss, replay/probability disagreement,
budget violation or unknown-LN baseline mismatch.

Positive: progress improves native scope amount beyond the matched contextual
adapter while retaining organization and guards. Negative: matched context is
equally good, factual fitting improves but native remains bad, or amount gains
destroy organization. Ambiguous: small paired differences or opposite seed
effects require refinement, not longer unchanged training.

## Verification and next condition

Verify scoped ownership/resumption, declared versus effective counts, strict
pre-query observations, no future-tail use, immutable forks, default checkpoint
compatibility, zero initialization, unknown-condition identity, gradients and
native-versus-replay law. Report source NLL, native controls, response channels,
actual lens observations and runtime separately. The continuing goal remains
unachieved; this proposed Note does not authorize a lifecycle transition.

## Frozen implementation and execution identity

Product c0db49c142612750e04b586cfbd396a2e9036126 implements the two optional
allocation modes, immutable declared/effective accounting, source query helper,
native integration and checkpoint metadata. It changes no existing checkpoint
law by default. Announced-control snapshots are explicit: factual replay of an
earlier query cannot use a control update that had not yet been announced.

The 511-row observability gap is now a concrete legal-history construction:
old exact replay and 512 common final rows agree while earlier LN totals differ
by 64. This verifies missing factual information, not causation of quantity drift.
Twenty-five selected ownership, sampling, allocation and frontier tests pass in
12.70 seconds, including CPU/MPS; the subsequently added live-announcement test
passes separately in 1.56 seconds. No test establishes native quality.

Run owner: 20260928-scoped-ln-allocation-v1. Frozen plan SHA:
ba3d3ab72807a75dadc9ba6dc9aa904a93d4a43a42cf83b898d2d9107ac867f4.
Fit script SHA:
7e1912044a2769bed0689f5b9434e8044b47b28d0b79e40bfd332b44fd5d9444.
Evaluation script SHA:
b712ae448d12e29c28f78d1a4e59ad46cad957e19dcfb86fb573a1b384d05b82.
Commands are recorded in plan.json: uv run --extra mps python followed by the
owner-relative run.py and evaluate.py. The fit command has started; evaluation
has not. Completion requires the live process result, not this entry.

## Revision 2: bounded cache execution after a terminal memory stop

The revision-1 process terminated before any optimizer update. Its preserved
fit-v1/failure.json reports 174.703122 seconds and peak process footprint
12,904,709,504 bytes, exceeding the 12-GiB guard. The last progress record had
256 cached training windows; this is not a completed fit. No cause attribution
to active tensors versus driver/runtime storage is available from that counter.

Revision 2 uses CPU for frozen factual extraction, matching native inference,
and retains MPS for the two small adapter fits. Data, source OID, checkpoint,
initialization seed, arm difference, optimizer, objective and terminal evaluation
are unchanged. New fit-v2 and native-{context,progress}-v2 directories preserve
the failed attempt. The evaluator additionally checks unknown-LN object identity
immediately after each complete case and honors the study STOP between cases;
the qualifier's own STOP remains available during a case.

Revised frozen plan SHA:
8e9cab84b87c324ef07ddfa1d7cb7f8bca3a66eea99b2bda1d4382cb809b1c3a.
Fit script SHA:
cecb31e81a8a35cd5f18adfa7823d12816a2e9d07f364ac80018d71a7b8b67df.
Evaluation script SHA:
ece99dac4dca343a6b229b68910796bb81c812ce4db89d8d640af941c9f7979c.
Run commands use the same owner with run_v2.py and evaluate_v2.py, respectively.
The original plan/scripts remain unchanged. This is an execution refinement,
not a changed scientific arm or an accepted/adopted result.

## Result Log: matched allocation fit and native evaluation

Card revision 2, accepted revision none; exploratory execution under the user's
standing local authority. Fit/native executable source is
c0db49c142612750e04b586cfbd396a2e9036126. Both terminal adapters complete 256
updates on 512 windows, 425 charts and 392 groups. Actual draws comprise 383
population/129 human windows, 418 known-LN and 252 whole-song programs. There
are zero overlapping LN-bearing request programs. All inherited 4,600,369
parameters remain bitwise fixed; only 69,953 new parameters train per arm.

CPU factual extraction plus MPS adapter learning completes in 344.931965 seconds,
maximum sampled footprint 3,663,892,320 bytes. The old MPS-cache failure remains
preserved and is not counted as a completed fit. Macro validation nats/row on
the unchanged 22 windows is parent 1.5724699156, context 1.5709571901, progress
1.5711760974. These values do not select or qualify the checkpoints.

Context checkpoint SHA:
40fb120d6debf7ba4aeefcc3c108b3b7a7a96cd58d5075ea10fd9e5a2c184196.
Progress checkpoint SHA:
8898c51714474f82171b570cd2c8867bb07e911a59dfbae91b2642687de6a4ca.
Both live under the run owner's fit-v2 directory. They require ln_feedback=False
in the qualifier or None in ControlledSession; loading weights alone does not
change other entrypoints' older feedback defaults.

Forty complete native cases finish in 587.025155 seconds. Every H timeline
matches the parent. All eleven unknown-LN cases per arm match parent exported
objects exactly, retaining both the Stream gains and existing failures. No
previously passing numeric gate newly fails. Maximum startup/service in the
2-second qualifier are .841024/.356002 seconds for context and .810458/.373383
for progress; this is not the 30-row/8-second benchmark or full client path.

The six primary whole-song LN fractions are:

| Request / song | Parent s0/s1 | Context s0/s1 | Progress s0/s1 |
| --- | --- | --- | --- |
| Classic .217153 | .360472/.437791 | .297396/.393103 | .319927/.390197 |
| STYX .485281 | .564394/.696268 | .448020/.650460 | .448020/.538945 |
| Blizzard .838046 | .968517/.976295 | .962733/.943987 | .962733/.941320 |

Mean absolute errors are .1537959884/.1148766184/.0991172034. Progress improves
.0546787850 against parent but only .0157594151 against context, missing the
prespecified .02 arm comparison. Most relative gain comes from STYX seed 1;
Classic seed 0 is worse than context. The live D4.5/LN .6 override remains
.901186/.897163 in the two arms; its difficulty proxy worsens to
3.236606/3.285719 from 3.432769. Before/restored fragments remain diagnostic,
not independent quotas. No endpoint is promoted.

Native context/progress case SHA values:
f704c76da13fbe4937c5b64a45cec81e1c04dd1dec8f6df25652968138c9962e,
daecd5bfb7613ea48bda480b5da89d4a697a6f8350cf73cd16be3d2dc66049e6.
Fit result SHA:
12bb1ab59fb1e0ad3da32d57c909f43c3f72b68372f1aeb2c08bbe2d9825acdf.
Comparison-v2 SHA:
96b0cb99edc52f9dc6a4d3f46ab0fbb519e03ad87d414c5cf1833858e9ef1fb9.

## Learning coverage and local response evidence

Factual exposure includes 23,736 known-LN H queries with nonempty prefix.
Absolute cumulative fraction error exceeds .2 at 4.80% of those queries, but
only 28/11,398 (.246%) in the later half of a request. This is query-weighted
exposure with repeated draws retained, not native recovery coverage. Program
coverage SHA a3846de0f0af62ac55d8fd0dca441c033a02236b051dc6e49fe49644ba25be00;
progress exposure SHA
bc956bdf0dbd7e78b5b23bdd9bd8ae28167168a8bde0fdac1d63b12b41fd8ecf.

On 991 held-out factual H queries, the progress readout's derivative with
respect to observed LN fraction is negative everywhere, mean -.061022, later
half -.064412. The average first-order correction for a .2 surplus is only
about -.012 log-odds. This tests readout sensitivity while other inputs stay
fixed; it is not a valid-history intervention or a claim that the dynamics are
stable. Both local-family variance and closed-loop history changes matter.
Readout-response SHA:
9d1d4fb4bddcba723eb4fac1e61a227d615dd172a260dcf7d79dc34bb65a29ad.

## Lens observations and additional reusable evaluation

Forty main pages (34 distinct images) and eighteen additional witness/corpus
pages were actually viewed. Main contexts are Classic [88589,93589), STYX
[1800,7800), Blizzard [40342,46342), both seeds and both arms, plus sources.
Classic retains mixed organization. The inspected STYX and Blizzard episodes
are identical between arms per seed despite whole-amount differences; this is
also checked against action tables. Blizzard's source anchor/TAP roles remain
unrestored. High coverage or a different interpretation is not labelled BAD.
Large Classic duration-change witnesses include meaningful sustained roles and
sparser H passages; no duration-change penalty is adopted.

A new progress STYX seed-1 witness has eight TAPs and one LN press at column 2
from 85524 through 86497 ms, median/max HH 121/140 ms. There are no companion
heads, but all three other fingers are held throughout: origins 85035/85280/
85035 ms on columns 0/1/3. Attack excess is only .0000703125 seconds and full
stars 3.552520. At 85524, alternative row (0,3,1,0) can release column 1 while
tapping column 2; next H 85630 admits (0,1,0,0). Exact replay and short-support
checks confirm this alternative prefix, not its sampled probability or better
complete response. The run is not made inevitable by H.

The observer now includes other continuing holds/origins, simultaneous releases,
recurrent TAP/LN types and all per-column releases within first-to-last head
span. New counterexamples distinguish identical recurrence with three free
versus held fingers, and TAP versus LN press/release runs. These fields are
facts for future response comparisons, not new masks, style labels or calibrated
strain. Source 2cb03e6eb2bdd574f20043a5500eb2e64c2ed511 adds held context;
3591df29b824cf4340c5425da81c5338cd7c49aa adds articulation and the result report.
Final focused recurrence/qualification checks pass 17 tests in 11.55 seconds;
the earlier combined scope/state/evaluation check passes 32 in 13.42 seconds.

The known nine-head witness is not called an isolated TAP jack. Ranked retrieval
within top-eight witnesses of 1,972 metadata-band TRAIN charts first returns no
near-match with nearly three continuing holds. A broader query returns four
witnesses and three independently recomputed source contrasts. This is not an
exhaustive prevalence/absence result. One exploratory retrieval attempt lacked
the required explicit clock-rate argument; its failure is recorded, and the
corrected v3 uses rate 1.0 without changing the broader query.

The sources actually inspected are:

- Touhou EX Boss Rush!! +a Kouhen [Phantasm Stage], 4.160033: eight TAPs/one LN
  press over 225682–227103 ms, median HH 158 ms, up to two other held fingers.
- Bug Thief [Toaph's Swarm], 4.009425: nine LN presses over 100122–101406 ms,
  median HH 160.5 ms, 80/81-ms durations and changing held/accompaniment roles.
- Without Boundaries [Wandering through spiritual space...], 4.485903: fourteen
  LN presses over 115483–117038 ms, median HH 120 ms, one other sustained hold
  and no companion heads. Durations are 38,40,42,44,46,48,50,52,54,56,58,60,71,75 ms.

Six closes in the final example are hard-excluded by current HR=50. This is a
concrete target-support restriction, not proof that it caused the generated
STYX arrangement. Other windows from the map may remain usable; enlarging the
network cannot restore zero-supported transitions. Do not ban LN-jack or short
holds, nor infer a physiological threshold from this retrieval.

Witness facts SHA:
1c34ad5a9ec68b1f1c2d56ac44de8ea6cec7b57b8925474da4ea17ee109cc9e8.
Ranked articulation SHA:
424415b4bd7163ffd9485a1e646c308d7cddebac4815887d6db59ba2966c682c.
Lens reading SHA:
f316e657b96e26c5b621917939f3e84ed1da28ce3d4ec9e8e0fef167669d8b89.
Harness remains 22e5c84f5cacb8493bdab5f1d0fdc09c5373dc60. No listening,
playtesting or new human annotation claim is made.

## Evaluation and next decision

Recommendation: REFINE. Retain the explicit scope-state interface and matched
endpoints for research; do not extend this source-only adapter fit unchanged or
promote a candidate because average amount improved. The observed correction is
weak, training lacks overlapping LN programs and large late allocation errors,
and meaningful release/articulation is excluded by current support in a real
ranked reference.

Next work should address these concrete dependencies: audit release-support and
actual accepted exposure for normal 2–6-star articulation; learn control recovery
from current-policy histories and complete scoped outcomes without source suffix
substitution; preserve LN/coordination distinctions in continuation response and
selection. Scope accounting remains distinct from gameplay-demand state. Broad
R1/audio/H/R learning and scaling remain available within those clarified
objectives; all control scopes, styles and realtime requirements remain part of
the ultimate goal.

Curated owner is docs/research/scoped_ln_allocation.md, including three tracked
Lens figures; the main formal report and full-R1 report link it. Code/results and
this Note are local commits only. Unrelated AGENTS.md and the untracked architecture
walkthrough remain untouched. All fit, evaluation, retrieval, rendering and test
process handles are terminal. The full playable-system goal remains active and
unachieved. Note status stays proposed, with no acceptance or lifecycle transition.
