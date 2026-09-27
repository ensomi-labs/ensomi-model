# Agent Note: Matched joint fitting of audio-history memory

Note ID: 2026-09-27-audio-memory-joint-fit
Status: proposed
Kind: research
Created: 2026-09-27
Updated: 2026-09-27
Product revision: baed4d720f3efc0c03ab8c2249bb96110497a586
Scope: Matched baseline/memory full-audio H/R/R1 fitting and fixed temporal, semantic-control and publication regression qualification
Related: 2026-09-27-audio-history-memory, 2026-09-27-playability-regression-evaluation, 2026-09-23-audio-skeleton-r1-integration

## Design and preparation

Previous goal turn is progress: implemented and verified memory architecture,
completed8+24 discarded joint integration/resource updates and reusable EVAL
with real historical failures and ranked positive exceptions. All those handles
are terminal. The next substantive task is actual matched quality fitting, not
another unchanged microbenchmark or rejected player-state fit.

Baseline is core2500, SHA0f1ddfa5b351988fca04246ec080be106855f49b50fb493370c2d5afee1febb8.
Compare ordinary controlled baseline continuation to AudioMemoryModel with the
same inherited tensors plus zero-output new modules. Both update audio/H/R/R1;
no frozen-audio encoding cache. Main learning will use384 updates with two source
microbatches (one population, one human-supported) per update. Inherited lr3e-5,
new memory/pyramid modules3e-4, AdamW decay1e-4, clip1. Final endpoint only; earlier
checkpoints serve recovery rather than selection by NLL. Integration weights are
not an initialization. A complete bounded Card will pin prepared inputs/gates
before either fit starts; this section authorizes only preparation under the
user's standing research instructions.

Preparation scope: ranked TRAIN charts with recomputed whole stars2–6 and complete
cached audio60–480s. Limiting to240s leaves5855 population charts and162 human
scopes; allowing480s gives6640/241 before final profile admission and additional
positive-reference exclusions. Human coverage remains sparse, particularly
prominent Tech; do not fabricate labels or claim full semantic coverage.
Exclude song groups/audio identities for the old fresh-audio panel, four genuine
style guards, four exposed near-four-star sources and inspected positive response
references. Population draws choose a difficulty band, then song group/chart,
then a uniform32s clock-partition interval. Human draws use the existing capped
inverse-square-root concept/assessment/difficulty-cell weights, then a32s window
containing its original annotation when possible. Original label scopes remain.

A proposal's group/branch or human selection is retained while retrying declared
profile-incompatible windows; every rejection is logged. Freeze the same accepted
768 examples and controls before either arm learns. Population importance retains
its clock-partition inclusion weight; human windows use per-second loss. Both
families have equal microbatch weight. Per-field controls use actual whole-chart
values by default, with independently sampled30% local difficulty and30% local
LN overrides on the scored window; local difficulty outside2–6 falls back to the
whole value, not clipping. No-head local LN fraction is undefined and likewise
falls back. Drop each control family independently with probability.15. Human
style assessments keep their original scopes/known bits. Shared control seeds
are frozen in prepared draws.

Preparation uses CPU source replay/collation only, not model fitting; maximum
1200 active seconds, fresh preparation output, no overwrite. It must preserve
actual endpoints and report profile exclusions. Before fitting, record accepted
coverage, input/script hashes, all reserved groups, exact evaluation cases/seeds,
per-range guards and the Mac runtime/memory stop. Reserved canonical validation
songs will be selected separately from the exposed developmental diagnostics;
no claim of being unseen by every inherited checkpoint is allowed without audit.
