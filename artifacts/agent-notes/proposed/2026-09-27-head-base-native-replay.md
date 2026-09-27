# Agent Note: Native replay with an unfitted H base and fitted memory policy

Note ID: 2026-09-27-head-base-native-replay
Status: proposed
Kind: research
Created: 2026-09-27
Updated: 2026-09-27
Product revision: fc641aa740a7528decbb0b5b522aa91039d06ff1
Scope: One H additive-path substitution in complete native rollouts, with reusable scoped regression evaluation
Related: 2026-09-27-head-control-hazard-probe, 2026-09-27-audio-memory-joint-fit, 2026-09-27-playability-regression-evaluation

### Experiment Card: head-base-native-replay-v1

Revision: 1
Accepted revision: none
Authority: exploratory execution under the user's ongoing research/experiment/
local-commit request; no implicit acceptance, adoption or remote publication.

The fixed-prefix probe identifies an H base contribution to faster next events:
replacing memory's base with the unfitted base increases restricted wait at least
20percent in8/12 generated contexts, median25percent. Restoring only the residual
does not recover the unfitted distribution. This does not yet explain complete
rollout counts because every new H changes subsequent history.

Question: does this additive path account for substantial sustained-pressure and
pacing regression under actual renewed generation? The native intervention uses
memory checkpoint7efcbe4bd6da3a4ec9ead01f24fdda63803ea930ebe76bdc7781fbf080a97e81
for its fitted H residual, H memory, R/R1, row memory and complete-audio context.
Only H's additive base is replaced with the output of core2500
0f1ddfa5b351988fca04246ec080be106855f49b50fb493370c2d5afee1febb8's own full-audio,
head-control and head-base layers at the same query clock and control vector.
No learned feature vector is transferred between encoders, no R1 content enters H,
and no parameter is updated. This is an explicit two-model diagnostic, not a
new checkpoint or proposed serving architecture.

The analogue is additive component substitution in the already implemented
bounded H law. It preserves the fitted policy's history feedback under a different
input base. Alternative explanations include residual dependence on generated
history, R1 response choices and common data/objective bias; unchanged complete
quality after substitution would limit the base-drift explanation.

The fixed14 cases are the existing9 Stream seeds, four ordinary development
conditions and the live control switch. Cases, audio hashes, controls, seed and
recovery60/50/40 are copied without alteration from the original qualification.
Plan SHA8c76ae534b46709ac89b81b1b98f6560b27149fa5c0282c4da887a220fb7ae83.
Paired native memory/unfitted outputs already exist. No reserved validation,
style guard or source-H case is promoted into a claimed native pass.

Primary diagnostic: Stream mean exact multiscale pressure J, memory baseline
.3823017792s over9 cases; call reduction substantial at<=50percent of this value.
Also require median paired H-count ratio<=.90 to support timing reduction as a
major contributor. These are diagnostic thresholds, not new adoption gates.
Report all9 values, whole-star MAE, maximum and durations of pressure episodes,
occupation/recovery/contrasts and conditional audio correspondence independently.
Actual first30 and2s service bounds remain2s; publication replay uses2s lookahead.
No below20ms attacks, incomplete output or mutated committed prefix is allowed.
The original stricter unfitted-pressure and star-MAE guards remain visible; a
diagnostic improvement cannot silently relax them or certify playable quality.

Switch scopes before[0,64000), override[64000,96000), restored[96000,357797)
remain separate with carried state, requested D3/4.5/3 and rho.2/.6/.2. Retain
the original difficulty error<=1 and LNfraction error<=.10 guards. Developmental
scope inspection uses the same previously viewed source passages, then all-scale
pressure witnesses as warranted. Never infer gain solely from amount or H count.

Procedure: use the existing qualifier's generate/inspect functions and native
MemorySession; a scoped wrapper replaces the H base output only. Include the
additional complete-audio base encoding in publication-start and audio accounting.
Record both model hashes and wrapper/source hashes in every evaluation identity.
Check additive formula equality on actual native queries, reparse exported osu
rows, then run the same temporal/publication/pressure evaluation. This probe must
not overwrite or edit the frozen384-fit qualification outputs.

Command `uv run --extra mps python artifacts/joint-audio/20260927-head-base-native-replay-v1/run.py`.
Fresh output run-v1, CPU one thread, Apple M5/24GiB; no accelerator fit concurrently.
Budget1800s,300s per case,8GiB process peak; one attempt, no automatic retry or
overwrite. Stop on owner STOP, input drift, nonfinite hazard, export/ownership
failure or resource limit. Reuse exact case seeds; helper checks seed274800.
Unrelated AGENTS.md and audio_architecture_walkthrough_zh.md edits remain excluded.

Interpretation: substantial pressure/H reductions with retained relief/control
guards support investigating base/residual training coupling. Unchanged/worse
native pressure despite local waiting effects rejects local component repair as
sufficient. Improvements in J with worse held occupation, rhythm or switch
response are a tradeoff requiring separate diagnosis, not an overall gain.
No default model/interface changes and no next fit are authorized by a pass alone;
the user's broader research goal remains the execution authority.


## Execution handoff

Wrapper SHA345b01fd92842305fc526caaa909b9ecb59abc1604209d7d80dfb6575ca7df5c; syntax compilation passes.
The original qualifier is pinned tobce8addba20e903581d5fae84f3a2d773494717abc2dda1fc33db44fc39ea8af.
All required library code is committed atfc641aa740a7528decbb0b5b522aa91039d06ff1;
the wrapper is experiment-owned instrumentation. Native queries1/50/500 compare
the substituted formula to captured base/residual factors. The extra full-audio
encoding is included in session start and audio timing. Start the bounded run;
no further fitting or model promotion.
