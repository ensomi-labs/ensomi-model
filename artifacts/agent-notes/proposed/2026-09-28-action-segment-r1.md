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

Revision: 1
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
