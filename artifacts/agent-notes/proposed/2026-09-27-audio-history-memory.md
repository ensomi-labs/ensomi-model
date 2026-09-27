# Agent Note: Joint audio and musically queried history memory

Note ID: 2026-09-27-audio-history-memory
Status: proposed
Kind: research
Created: 2026-09-27
Updated: 2026-09-27
Product revision: f65de370416255477f81993bfd594680ba40cbd6
Scope: Nonlinear multiscale full audio, elapsed-time history attention, native and teacher-scored H/R/R1 integration
Related: 2026-09-27-four-star-musical-phrasing, 2026-09-23-audio-skeleton-r1-integration

## Basis and authority

The preceding turn made progress: ten new complete matched generations and Lens
inspection identify concrete loss of rhythmic and TAP/LN texture contrasts. The
user explicitly asks to consider multiscale breathing, audio response and
attention-like historical understanding, and permits scaling when needed.
Implementation and bounded local experiments remain authorized. No remote push,
Note acceptance, runtime promotion or final-system completion is implied.

## Experiment Card: audio-history-memory-integration-v1

Revision: 1. Accepted revision: none. Exploratory implementation/profiling under
standing user authorization. One combined architecture intervention is evaluated
for correct information flow, trainability and feasible runtime before a main fit.
No fitted-quality claim or broad architecture adoption follows from this stage.

Hypothesis: a nonlinear multiscale audio path and current-context retrieval of
past music/arrangement pairs provide useful representational capacity missing
from the present late-fused compressed histories. This stage tests whether that
architecture can execute consistently and train within the Mac budget; the next
matched native-quality experiment must test musical benefit. Main alternatives
remain supervision/coverage and scope-control effects. The earlier report's
Music Transformer, Transformer Hawkes Process and MusicVAE analogues apply with
their stated limits. This is an adaptation, not a novelty claim.

Clean baseline f65de370416255477f81993bfd594680ba40cbd6, core2500 checkpoint
SHA 0f1ddfa5b351988fca04246ec080be106855f49b50fb493370c2d5afee1febb8.
Use a separate research model/session/checkpoint family that inherits the legal
row law and scoped-control contract. Leave default runtime selection unchanged.
Behavior-neutral hooks in the shared planner/session/interval scorer may expose
per-query history options; legacy paths must preserve their existing law.

Intervention: preserve the fine96/100Hz audio encoder and old global context;
add a zero-output-initialized nonlinear audio path from learned fine features.
At 500-ms cells, retain mean and maximum features; combine .5/2/8-second pooled
views, project to width256, apply three bidirectional four-head attention layers,
and project to the existing128 global coordinates. Full audio is available on
both paths. This branch augments cached full-song encoding and is jointly
trainable; no frozen-audio shortcut in the learning check.

H, R and R1 each gain a separate four-head, width128 query-memory read, binding
its own past TCN state to audio at that event. H memory has H timing only; R
has skeleton timing/roles and its permitted current LN projection; R1 has full
committed rows. Reads are conditioned on current audio, relevant query clocks/
preview/controls and hand context. Outputs are zero-initialized additions to
the appropriate local history. Shared mirrored weights preserve hand symmetry.
R1 still owns counts, lanes, TAP/LN and release subsets. Exact/player state and
response evaluation remain distinct.

Memory selection uses the latest known event state per absolute500-ms cell in
the preceding64seconds, at most129 cells including boundary cells. The fast
local TCN retains immediate order; sampled older states are an explicit lossy
approximation. Empty cells add no fake action. Relative real-time offsets and
current exact clocks carry age/silence. Every selection is capped by the actual
query's known-prefix event index, not only its 10-ms hazard-bin anchor; an anchor
may be later than the target event. This rule must prevent future-target leakage
inside the same bin/cell. Native rollback/forks preserve corresponding memory.

Training collation reconstructs full source-prefix temporal states under current
weights and attaches full-audio features at their actual times. It may initially
recompute local histories redundantly; optimize only after measured need. No
learned training cache may survive a parameter update. Existing interval survival,
conditioned release waits, complete-row support and true-end closure are retained.
A private interval or memory horizon never invents a release endpoint.

Checks: zero-initialized old-law equality; nonzero current-audio/history influence;
mirrored hand equivariance; known-prefix causality including two events in one
hazard bin; dense/cached teacher/native score agreement; empty memory and silence;
model checkpoint roundtrip; fork and control-suffix rollback; joint audio/memory
gradients. Existing affected owners must still pass. Primary integration bound:
row log-probability parity within3e-5 and finite gradients; no semantic quality
threshold at this stage. Native profiling must record actual parameter count,
full-audio encoding, first30 rows, two-second publication maxima and peak memory.

Data for bounded profiling: the four verified ranked audio assets in
20260927-four-star-phrasing-v1/plan.json (SHA9ed61800275342fa03e8282b54985c6d33930e2ac3666b124d9f177b333d24d9).
Teacher fixtures must satisfy the existing source-execution contract; any source
rejection is reported rather than silently relabeled. The integration learning
check is at most8 joint updates on declared valid source intervals, not a main
fit or quality candidate. Seeds273200–273207; discard its weights after profiling.
Same complete-song audio at inference and training. No target-derived external
encoder or borrowed generated style labels.

Environment: Apple M5/24GiB, uv --extra mps, --group dev for tests. Development
runtime bound30 active minutes for tests/profiling/smoke, separate from code and
analysis time. Stop on contract/causality failure, nonfinite gradients, sampled
footprint above18GiB, or native generation slower than audio duration. Fresh
artifacts/joint-audio/20260927-audio-history-memory-v1 outputs, no overwrite or
implicit resume. Model-backed runs require a clean intervention source commit
and recorded script/input hashes; tests can run during implementation. Exact
commands are recorded with their result before a main learning comparison is
selected. No large training run is authorized by a passing integration metric
alone; the standing goal permits designing and then executing a suitable follow-up.
