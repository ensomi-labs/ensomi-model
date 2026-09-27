# Agent Note: H timing sensitivity to control support and past skeletons

Note ID: 2026-09-27-head-control-hazard-probe
Status: proposed
Kind: research
Created: 2026-09-27
Updated: 2026-09-27
Product revision: f22e9366977b33029d571a5fe7ec7ffa9ba8a463
Scope: Fixed-prefix next-H distribution probes after failed matched joint fitting; no new fitting or model adoption
Related: 2026-09-27-audio-memory-joint-fit, 2026-09-27-playability-regression-evaluation

## Question and evidence

The 384-update memory candidate fails native Stream pressure and star-error
guards against its unfitted initialization. On one common Zenithfall seed,
H counts increase3582 to5514; the continued baseline increases to6390. Training
includes only6.368 scored seconds of prominent singleton Stream, repeated from
one3.184s source, while inference requests span minutes. Style boundary clocks
include unbounded asinh(seconds). These are verified observations, not a causal
identification of the density regression. Older H self-excitation, common fitting
selection bias and changed audio features remain alternatives.

### Experiment Card: head-control-hazard-probe-v1

Revision: 1
Owning Agent Note ID: 2026-09-27-head-control-hazard-probe
Accepted revision: none

Execution is exploratory under the user's standing request to research, implement,
experiment and commit locally. The Card remains proposed; no human acceptance,
SUPPORTED conclusion or remote publication is inferred. The working tree's
unrelated AGENTS.md and audio_architecture_walkthrough_zh.md changes are excluded.

#### Comparison and interventions

Baseline product OIDf22e9366977b33029d571a5fe7ec7ffa9ba8a463 preserves the qualified
model law. The common unfitted, continued baseline384 and memory384 checkpoints
are pinned by bytes in plan.json. They are0f1ddfa5b351988fca04246ec080be106855f49b50fb493370c2d5afee1febb8,
a6916403daf6c658f0952d1181703732156f4ac37a8e5b773007c5dc4e2c5c24 and
7efcbe4bd6da3a4ec9ead01f24fdda63803ea930ebe76bdc7781fbf080a97e81.

The probe uses15 fixed query contexts, not sampled future charts. Twelve are at
.15/.40/.65/.85 of the audio duration for failed-memory outputs
stream-zenithfall-0, stream-hysteric-2 and four-stream-classic-pursuit. The same
full H prefix, elapsed query clock and complete audio are supplied to each model.
Three more use actual human-source prefixes inside original all-five-style
prominent Stream annotations: distinct sources selected by sorted SHA from the
frozen draw ledger, requiring annotation duration>=4s. Source SHA prefixes are
1b54c6789fe5,1e1ab790ad09,4fd873b0a1e0. Their labels remain unchanged as evidence.
The complete plan SHA isd69ea7da517425120bffce6536aa0ce5d9f3350f5ccace8ac0f1284f0fccc02c.

Each counterfactual changes one input block relative to actual controls/history:

1. Shorten the active Stream request to a10s scope centered at the query. Its
   values/known mask remain active for the whole2s probe; numeric clocks stay.
2. Cap only known style clocks' asinh coordinate at asinh(5s), preserving all
   other coordinates. This is an explicit feature lesion, not a valid request.
3. Remove style requests, preserving unknown semantics; alternatively change
   Stream prominence1 to supporting0 on generated contexts.
4. Set only difficulty to2 or6 with its original range clocks.
5. Thin alternating older H events while preserving the latest four, last H,
   current observation time and four-H physical support. Recompute all H TCN
   and memory states from the edited prefix. This is an off-policy history probe.
6. For the memory model only, supply its null attention view while keeping the
   ordinary TCN history and every other input unchanged.

On human prefixes, compare the real full bundle with Stream-only and all-style-
unknown subsets, plus history thinning and the memory-only attention lesion.
Removing known labels changes the request and is not required to preserve the
output. No unspecified style is converted to absent. No combination of lesions,
gradient update, timing-temperature edit or R1 compensation is allowed.

The closest mechanism analogue is the repository's exact discrete-time survival
law in joint_audio_continuation/timing.py: one native-ms Bernoulli hazard for the
next event, evaluated along the no-event branch. These paired counterfactuals
adapt input ablation, not a new point-process objective. They require no external
audio pretraining or section labels. R1 and R release choices do not enter H.

#### Measurements and interpretation

Compute the exact first-event distribution over the next2000ms, including the
no-event censoring atom. Report restricted expected waiting time, survival at
20/50/100/250/500/1000/2000ms and median when reached. Paired total variation
includes the censoring atom; restricted Wasserstein timing distance measures
CDF displacement. Do not call this the expected H count in a2s rollout: after
an event the real H history would update. Record H base versus bounded residual
contributions to distinguish direct control-rate shifts from history pathways.

No reference value for these new probes is available; actual-condition values
are measured first under each frozen prefix and never selected for low/high risk.
For prioritizing mechanisms, call an effect material when the restricted mean
wait changes by at least20%. A consistent branch requires the expected direction
in at least8/12 generated contexts, reported per model; human subset effects are
descriptive3-source evidence with a2/3 material-sensitivity threshold. Preserve
individual contexts and absolute changes. These thresholds prioritize follow-up;
they neither score playability nor establish a physiological limit.

Clock lesions increasing waits consistently support a control-clock contribution
to rapid next-H predictions. Thinning increasing waits supports older-H positive
feedback conditional on unchanged last-four support. Null memory increasing waits
localizes an attention contribution, not the necessity of removing attention.
Small effects deprioritize a branch locally; conflicting directions or equal
effects before/after fitting remain ambiguous. Difficulty trends describe local
control sensitivity, not difficulty control of complete charts. Counterfactual
feature distribution and real semantics remain confounders.

#### Procedure and bounds

Ownerartifacts/joint-audio/20260927-head-control-hazard-probe-v1; preparation plan
exists with verified audio/Mel/source/checkpoint hashes. Add reusable waiting-law
diagnostics and focused probability tests, commit a recoverable source revision,
then run `uv run --extra mps python artifacts/joint-audio/20260927-head-control-hazard-probe-v1/probe.py`.
CPU one thread, Apple M5/24GiB, no fitting, no network, deterministic queries;
seed274700 is used only for implementation parity checks. Maximum900s and8GiB
process peak. Stop on nonfinite valid hazards, input-hash drift, unexpected
native/dense probability disagreement, resource bound or owner STOP file.

Fresh run-v1 output; refuse overwrite, no automatic retry/resume. Inputs and
probabilities are held constant across paired comparisons except the specified
block. Validate dense-prefix scoring against actual HeadPlanner/MemoryHeadPlanner
cache scoring at the first generated query for each checkpoint. Survival must
match sample_hazards semantics, censoring and invalid masks, with tests that
distinguish equal event mass but different waiting distributions. Keep actual
first-event probabilities separate from any generation-quality report.

At completion append input/source hashes, elapsed/resources, individual paired
results and limitations. Revise the proposed Card if its protected slice or
intervention changes. Select a follow-up based on this evidence before another
large fit; no checkpoint/default promotion follows from a positive probe.
