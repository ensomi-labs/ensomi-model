# Agent Note: Joint audio and musically queried history memory

Note ID: 2026-09-27-audio-history-memory
Status: proposed
Kind: research
Created: 2026-09-27
Updated: 2026-09-27
Product revision: 720457b8651d40095c2247b5992c3318b2cb5ced
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


## Implementation and live integration handoff

Previous turn classification: progress. Core intervention is committed at
b130dfb6b0e9d4617b8ec8a235b334287b4813e4; independent evaluation code is138a1f5,
and CPU/MPS audio-padding checks plus report are720457b8651d40095c2247b5992c3318b2cb5ced.
The product worktree is clean before profiling. The latter two commits do not
change the architecture comparison's native model law.

Implemented AudioMemoryModel/MemorySession, indexed half-second memory selection,
full-prefix teacher scorer and explicit checkpoint family. Native H/R logits and
row probabilities agree with current-weight teacher replay under nonzero memory;
tap layout/count cannot leak into H/R. Thirty-six affected checks pass in13.68s,
plus two nonzero audio-padding checks on CPU/MPS in4.03s. Eight evaluation-package
checks pass separately. Initial test collection needed the new test package's
__init__.py; that setup failure did not run model tests.

The full prototype has7,616,517 parameters versus core4,583,985. The first
preflight attempt80268 terminates before sampling because the observation driver
omitted PublicationLog's required callback argument. Retain its preflight folder.
Set callback=None, keep model/data/procedure unchanged, and run fresh preflight-v2.
Replacement handle77659 is live. Command:
uv run --extra mps python artifacts/joint-audio/20260927-audio-history-memory-v1/preflight.py.
Never restart this handle merely because an observation times out.

The declared learning draws are eight32-second intervals: chart i%4, interval
index1+i//4, seeds273200+i. Both native profiles use Max Burning at seed273200;
full-law equality is checked before learning. Inherited parameters lr3e-5, new
modules3e-4, clip1. Full audio/H/R/R1 update jointly, with native row preferences
replayed on factual prefixes. This is an integration smoke, not a quality fit;
weights are discarded for subsequent main fitting. The frozen config records
actual script/source/model/plan identities. Existing limits remain1800 active
seconds and18GiB sampled memory. No main quality fit has started.

User steering makes reusable EVAL a priority alongside the model. The separate
Note2026-09-27-playability-regression-evaluation owns those algorithms and real
historical calibration. Do not choose or promote a new model solely from NLL.

Preflight-v2 handle77659 is terminal after saving the baseline export: the
measurement driver merged duplicate generation_seconds keys from save_rollout.
This is an observation-record construction error, not a native model failure.
Retain that folder, merge saved fields with explicit report.update, and use fresh
preflight-v3. The model/source/data are unchanged. The preceding live-handle entry
is superseded; replacement status is recorded on its next authoritative poll.
