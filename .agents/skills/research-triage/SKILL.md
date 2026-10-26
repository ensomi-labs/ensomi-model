---
name: research-triage
description: Guide Ensomi ML research through analogue search, hypothesis branching, bounded experiment design, and evidence-based result evaluation. Use for open or diffuse research questions, novelty assessment, experiment selection, an accepted Experiment Card, or interpretation of experimental results. Route established implementation, routine debugging, and causal analysis to their owning workflows.
---

# Ensomi Research Triage

Turn an open research question into a grounded direction, a proposed discriminating experiment, or a defensible interpretation, and state for each change whether it makes a research claim or an engineering claim. Keep the human owner responsible for research direction, acceptance, execution authority, and adoption.

## Choose the research mode

- **Explore:** the question, novelty, mechanism, or useful implementation family is unclear. Find analogues, branch plausible hypotheses, and identify evidence that separates them.
- **Design:** the direction is plausible but needs one bounded experiment. Select a branch and define the smallest test that can change the decision.
- **Evaluate:** results already exist. Check the planned comparison, guards, deviations, confounders, and scope before deciding what the evidence supports.

Implicit skill invocation selects the reasoning workflow; it does not authorize code changes, a run, card acceptance, or publication. Every mode may inspect the repository and author its scoped research output. When relay is on, the active research task authorizes recording that output in the notes as [Persist through relay](#persist-through-relay) describes; it never implies card acceptance. An explicit request to revise a protected Card field authorizes only the required revision increment and the `accepted -> proposed` reset; no other transition is implied. Change experiment code, implement instrumentation, or start a run only when the active task separately and clearly authorizes that action.

Use these transitions:

- Explore ends at `DROP`, `REFINE`, or `TEST`. `TEST` authorizes Design, not implementation or execution.
- Design produces a proposed Experiment Card. The card becomes accepted only when a human explicitly accepts that card ID and revision; with relay on, the acceptance is recorded in the notes as a decision with its private citation.
- Evaluate ends at `DROP`, `REFINE`, `REPEAT`, or `SUPPORTED`. `REPEAT` reruns the same accepted card; changing a protected field returns to Design. `SUPPORTED` is bounded evidence and awaits human direction—it does not authorize adoption, merge, or another run.

A request may span modes, but every new or revised card remains proposed until its exact revision is explicitly accepted.

## Ground the question

State the research question, current baseline, target improvement, and the decision the evidence should change. Then give each candidate change its **claim kind** under one rule, which every later step applies:

- A **research claim** explains the change by a **problem property**, a property of the domain, the data, or the process that generates the data, and that property makes an **out-of-metric prediction**: where the gain should and should not appear, what should transfer, or what an ablation should remove.
- Any other change makes an **engineering claim**, whatever layer it touches. Its gain has the **generic explanation**: more tuning, capacity, compute, schedule, ensembling, augmentation, or the like would produce it on a **matched baseline**, one given whatever of these the intervention adds.

A **mechanism check** shows that the implementation acts as the hypothesis says, for example that the model uses a new input (guardrail 3 of [`ensomi-research-guardrails`](../ensomi-research-guardrails/SKILL.md)). A gain supports a research claim only when the mechanism check passes and the out-of-metric prediction holds against a matched baseline; otherwise the generic explanation stands. A negative result refutes a hypothesis only after the mechanism check passes; until then it is an **engineering failure** (the run or code broke: fix it) or a **sound idea badly realized** (it ran, but the mechanism check failed: realize it another way). Engineering claims are legitimate; state the claim kind so that an engineering gain is not reported as a research claim.

Prefer primary papers, project repositories, benchmark definitions, and reproducible local traces. Treat surveys and curated lists as discovery aids. Use [research workflow patterns](references/research_workflow_patterns.md) when claim kind, analogue selection, or evidence grading needs calibration.

## Find and compare analogues

For each serious branch, name the closest **technique analogue**, the standard use of the technique the change applies, and for a research claim the closest **domain analogue**, the domain's account of the problem property or work in another field that uses the same property. Compare assumptions, data, objective, evaluation, and operational constraints. A nearby name or shared component is not enough; explain the mechanism that transfers and the difference that remains.

Then present a source-bounded novelty assessment, provisional until the human owner adopts it. Novelty lies in the problem property and its out-of-metric prediction, not in the component: a change that matches its technique analogue and names no problem property is an engineering claim, however new it is to the repository.

## Branch the reasoning

Generate only branches that predict meaningfully different observations; the generic explanation competes with every research branch. For each branch record:

- hypothesis, claim kind, and mechanism;
- for a research claim, the problem property and its out-of-metric prediction;
- domain and technique analogues;
- expected supporting and falsifying signals;
- smallest discriminating probe, which for a research claim tests the out-of-metric prediction against a matched baseline;
- cost, confounders, likely failure mode, and mechanism check.

Compare branches under the same baseline and decision criteria. Merge equivalent branches; a branch that names a problem property but no out-of-metric prediction is equivalent to the generic explanation and makes an engineering claim. Discard branches that no available observation can distinguish.

End exploration with one outcome:

- **DROP:** repository fit, or evidence from an implementation that passed a mechanism check, defeats the direction;
- **REFINE:** a smaller or different question is needed;
- **TEST:** one bounded experiment can decide between live branches.

## Design one bounded experiment

Use [the Experiment Card](templates/experiment_card.md). Give it an immutable card ID, an incrementing revision, and a status, `proposed` until a human accepts that exact revision. Define exactly:

- hypothesis, claim kind, selected branch, and decision the result can change; for a research claim, the problem property and out-of-metric prediction;
- domain and technique analogues and remaining difference;
- clean baseline source revision, configuration or checkpoint, dataset slice, baseline run or evidence identity, and baseline value with aggregation, sample count, and uncertainty; for a research claim, a matched baseline;
- one causal intervention, separate from behavior-neutral instrumentation;
- primary metric definition, direction, decision threshold (for a research claim, also on the out-of-metric prediction), regression guard bound, and qualitative check;
- exact procedure, paired or independent comparison design, commands, seeds, environment, runtime and resource bounds, fresh output destination, overwrite or resume policy, and early-stop condition;
- mechanism check, confounders, and positive, negative, and ambiguous interpretations.

Prefer a fail-fast slice over a full training run. The test must be capable of rejecting the selected branch. Any edit to the question, baseline identity or value, slice, causal intervention, metric, decision threshold, guard, procedure, comparison design, seeds, environment, resource bounds, output or resume policy, or stop condition increments the card revision, returns the card to `proposed`, and requires renewed acceptance.

## Preserve the execution handoff

The accepted Experiment Card controls implementation and execution. Before implementation or a run, compare its current protected fields with the `Accepted revision`; any difference requires a revised proposed Card rather than silent continuation. Before code edits, require the implementation worktree at the accepted clean baseline OID or an explicitly authorized clean descendant whose existing diff is behavior-neutral to the comparison. Otherwise stop or return to Design. An implementation agent may add only necessary behavior-neutral instrumentation; a change that can affect the causal comparison belongs in the intervention and requires a revised card.

Implementation and execution are separate permissions unless the active task clearly includes both. Before a run, record a clean intervention source OID and audit its baseline-to-intervention diff against the accepted card; stop on an unrelated change or undeclared causal difference. Verify the accepted card revision, exact baseline and inputs, commands, seeds, environment, fresh output destination, overwrite or resume policy, compute, storage, network, and runtime bounds. Stop on the recorded guard, kill condition, authorization limit, or unavailable prerequisite. Return to Design and reacquire acceptance when a protected field must change. A run-only request appends reproduction, measurements, and plan-conformance fields, leaving Evaluation and Decision pending.

## Evaluate results

Use [the Result Log](templates/result_log.md). Verify the accepted card revision and record clean baseline and intervention source OIDs, separate or paired run IDs and commands, environment and hardware, seeds, inputs, configuration and checkpoint identities, output artifact paths, budget consumed, stop reason, and planned-versus-actual deviations. An Evaluate request fills or revises the Evaluation and Decision fields from recorded evidence. Append each Result Log; never overwrite earlier run evidence.

Compare baseline and intervention values under the planned metric and guard. Separate observed measurements from interpretation, state the strongest alternative explanation still compatible with the data, and record the claim kind the evidence supports, claiming no more than the comparison carries (guardrail 8 of `ensomi-research-guardrails`):

- A gain supports a research claim, and `SUPPORTED` for one, only when the mechanism check passed and the out-of-metric prediction held against a matched baseline; otherwise it supports at most an engineering claim, and the generic explanation is the strongest alternative.
- A negative result supports `DROP` only when the mechanism check passed. Before that it is an engineering failure or a sound idea badly realized: recommend `REPEAT` when the fix changes no protected field, and `REFINE` otherwise.

A run with no accepted Card is exploratory: use `none` for unavailable acceptance or Card fields. When an accepted Card exists, append a dirty, unrecoverable, incomplete, or materially deviated run's exploratory Result Log beside it without changing the Card or its status. Never recommend `SUPPORTED` for these cases. Recommend `REPEAT` only for a conforming rerun of the same accepted Card; otherwise recommend `REFINE` and leave any Card revision to a subsequent Design task. Bound every conclusion to the tested source revisions, checkpoint, dataset slice, runtime, and measurement definition.

## Persist through relay

Relay is on when its session-start message names the notes worktree, or `relay status --repo <repo>` exits 0. Then every proposed or accepted card, executed run, direction-changing result, and cross-turn research task is recorded in the notes and follows relay's notes convention:

- A card and its Result Logs live in the notes material for the question they serve, or a new `artifacts/<topic>-<date>.md` when none fits, each under a stable anchor (`card-<id>`, `result-<id>`). Cite code as `<commit>:<path>` and run outputs where they live, by machine and absolute path.
- Evaluation lines are anchored observations and interpretations, kept distinct. A human's acceptance of a card revision, or direction after a result, is an anchored decision with its private citation; a question the human cannot settle yet goes under Open for the human.
- A result that changes a judgment updates the affected problem-map node and Current movement in `RESEARCH.md`: the entry point states the claim and links the material.
- Commit with `relay notes commit` when a card is proposed or revised, a run is recorded, or a judgment changes, and push as relay's skill says.

When relay is off, return cards and logs in the response and say that they are not persisted; do not write them anywhere else. Exploration-only brainstorming may stay in the response either way. Do not recommend `SUPPORTED` before the Result Log is recorded: committed in the notes when relay is on, or in the response when it is off.

Publish reusable conclusions through the repository prose workflow.

## Output resources

- Use [the Idea Critique](templates/idea_critique.md) for Explore.
- Use [the Experiment Card](templates/experiment_card.md) for Design.
- Use [the Result Log](templates/result_log.md) for Evaluate.

These are content-section templates, not standalone files. With relay on, embed each Experiment Card in its notes material and append each Result Log beneath it. Load only the resource needed for the selected mode.
