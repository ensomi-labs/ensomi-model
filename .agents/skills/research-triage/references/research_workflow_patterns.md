# Research Workflow Patterns

Use this reference to calibrate claim kind, analogue search, hypothesis branching, and evidence quality.

## Claim kind

These cases come from Agent4Sci (Zhong et al., arXiv 2610.28763), in which coding agents tried to improve published scientific ML models. Its authors label changes as insight or engineering case by case, without a stated rule; the readings below apply this skill's rule.

- **Same layer, different claim kinds.** Both PFDelta runs of one agent train on the evaluation metric, the power-balance loss: without the paper's skills as an auxiliary loss, with them as the primary loss, after recognizing that the loss follows from Kirchhoff's laws and needs no labels. As the paper describes them, only the second names a problem property, and that property makes out-of-metric predictions a card could test: the gain should not depend on labels, and the error against solved states should fall with the residual.
- **An engineering claim.** On Mars-Bench, the gain came mainly from a stronger backbone and recurred in every run; the paper groups it with engineering-driven gains. It names no problem property, and the generic explanation, here capacity, accounts for it. The paper's agents defaulted to such moves and miscalibrated their novelty; the technique analogue exposes this.
- **A gain the metric allows without solving the problem.** The original PFDelta evaluation ignores inference time, so calling the exact solver scores near-perfectly, and the run with skills also refines its predictions with the power-flow equations at inference. To the extent its gain comes from computation the metric does not charge for, it predicts nothing beyond the metric. Before reading a gain of either kind, check that the metric sees what the task needs, as guardrail 1 of `ensomi-research-guardrails` requires of any measure that decides; the paper's later three-day study does so by penalizing inference slower than the model's reference budget.
- **Failures.** The paper's experiment skill fixes an engineering failure (out of memory, NaN, divergence, a crash), pivots when the idea is sound but its implementation is not, and abandons an idea only for a diagnosed incompatibility with the problem's structure. This skill replaces that last condition with a testable one: a negative result counts against the hypothesis once the mechanism check passes.

## Analogue comparison

Compare each domain and technique analogue by mechanism rather than name:

- problem and prediction target;
- representation and model objective;
- data, supervision, and sampling;
- inference or decoding procedure;
- evaluation protocol and baseline;
- runtime and product constraints.

Record the remaining difference before assessing novelty.

Useful discovery sources include:

- `worldbench/awesome-ai-auto-research` for research-workflow taxonomy;
- `Future-House/paper-qa` and `assafelovic/gpt-researcher` for source-grounded evidence gathering;
- `microsoft/RD-Agent` for iterative research and development;
- `Embodied-Minds-Lab/BES` for branching candidates against decomposed goals;
- `InternScience/ResearchClawBench` for checklist-based evaluation;
- `SakanaAI/AI-Scientist` as an example of broad automated research loops.

Treat each project as an analogy, not authority for Ensomi architecture.

## Branch quality

A useful branch has a distinct mechanism and predicts an observation that separates it from competing branches. Reject a branch when it only renames another branch, cannot be falsified with available evidence, or requires a broad implementation before any local signal can be measured.

Use goal decomposition when the target metric is too sparse. Local subgoal checks can rank branches, but only the Card's decision threshold decides.

## Evidence grades

- **A:** reproducible local run, checked metric, or current repository trace.
- **B:** primary paper or repository with explicit data, commands, and evaluation.
- **C:** benchmark, leaderboard, or execution trace with inspectable criteria.
- **D:** survey, curated list, or secondary technical summary.
- **E:** README-only, marketing, popularity, or social claim.

Use the weakest grade needed to support a claim. Do not promote novelty or effectiveness beyond the available grade.

## Experiment pressure

A bounded experiment has a fixed baseline, one intervention, a named slice, a primary metric, a regression guard, a qualitative check, a mechanism check, and a runtime stop; for a research claim, the baseline is a matched baseline and the decision threshold also covers the out-of-metric prediction. Positive, negative, and ambiguous outcomes must lead to different research decisions.

Prefer the smallest comparison that can reject a branch. Full training is justified only when cheaper probes cannot preserve the question.
