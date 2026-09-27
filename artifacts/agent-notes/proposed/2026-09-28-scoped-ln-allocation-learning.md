# Agent Note: Learned scoped LN allocation with factual progress

Note ID: 2026-09-28-scoped-ln-allocation-learning
Status: proposed
Kind: research
Created: 2026-09-28
Updated: 2026-09-28
Product revision: 073b6b68fa59b4be99be093cdf284f8c16e08492
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

Revision: 1
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
