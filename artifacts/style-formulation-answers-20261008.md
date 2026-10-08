# Style formulation: the human's answers to q-style-20261008, and the full rewrite (2026-10-08)

Shareable. Main session `515ba54a` (worktree `~/wt/ensomi-model-formulation`, branch `docs/style-formulation` at `8da2bda`), one of the parallel discussion sessions; `RESEARCH.md` is left to session 6d71c094. The questions are [q-style-20261008](style-formulation-recheck-20261008.md#q-style-20261008), presented with the agent's recommendation first.

<a id="d-style-answers-20261008"></a>**The human's answers, 2026-10-08 ([private, local](private/human-inputs/515ba54a-4eb7-4951-92c7-3c986083259c.md#answer-1)).** Free text, read within the questions as presented. Interpretations are marked.

1. **ρ is the whole-chart identity (option a).** It includes chart-level level preferences, such as LN amount and length and chord density, where no target is set. Targets override them. A free property therefore follows the identity's preference.
2. **Head times: neither option as presented.**
   - `notation.md` is directionally correct, because the system ultimately generates head times.
   - Taking head times as given is a decomposition choice, and it is the current problem scope.
   - The formulation may add it, and must then state the priors, that is, why this decomposition is chosen.
   - Agent reading: the head times are not a new formulation input on a par with ρ. They are the given part of a stated decomposition of the full problem.
3. **Drop "validated" (option c).** Principle: **the formulation defines problems, not evaluation.** Agent reading: this applies to the whole formulation, so evaluation tables and protocols leave all four documents, including `gameplay-state.md`'s "Evaluation questions" on `main`. Requirements on the generation law stay.
4. **The draft enters `main` after the revision (option a).**
5. **"ρ after stage 2" no longer stands (option a).** The human's reason: without a chart identity the chart drifts away ([s-diagnose-answer](r2-diagnose-synthesis-20261008.md#s-diagnose-answer)). That makes "natural" ill-defined, and natural is required to define the conditions. This supersedes item 2 of [d-formulation-answers](r2-style-formulation-check.md#d-formulation-answers) on ordering.
6. **The written defaults, answered one by one:**
   - **q2 changed:** strength levels below the default are allowed.
   - **q3 kept:** strength applies to property targets of requests only.
   - **q7 replaced, the human's distinction:**
     - the prefix is committed history only;
     - the whole-chart identity is its own input, which can be extracted or calculated from some chart;
     - a random seed that controls sampling is a third thing, which a system may not have and which the formulation need not contain.

     Agent reading: "chart seed" goes as a term. The prefix is not identity evidence by definition. The seed-against-ρ conflict rule is unnecessary.
   - **q8 kept:** overlapping targets on different quantities are allowed.
   - **q9 not answered:** the overlap ban on the same style concept. Neither was the agent's reading that a request may be withdrawn or replaced before its start. Both stay as written and remain open.
7. **The Mapping scenario (local edit) is dropped from this update.** The human asked whether answering the formulation questions removes the problem.
   - <a id="a-local-edit-left"></a>Agent's answer, given to the human: only in part. A whole-chart identity and song-time scopes cover the steering side.
   - A window bounded by a fixed later chart still has no notation: legality joining a fixed future, long-note obligations at the right boundary, conditioning on the later chart.
   - Node `local-edit` is unchanged.

<a id="r-formulation-rewrite"></a>**Delegated 2026-10-08 at the human's request: a complete rewrite of the four formulation documents by one fresh Opus subagent.**
- Brief: `~/ensomi/.sync/cp/scratch/style-formulation-rewrite-20261008/brief.md`.
  - Binding decisions A1-A9 (the human's earlier answers already in the draft) and B1-B9 (the answers above).
  - A list of what the formulation keeps and what it excludes. Excluded: evaluation, research numbers, code and plan names, representation designs, the random seed, local edit, provenance narration.
  - Facts for the decomposition section, stated qualitatively, from [s-skeleton-achieves](r2-represent-skeleton-20261008.md#s-skeleton-achieves) and [c-ln-placement-mapper](r2-representation-20261008.md#c-ln-placement-mapper).
  - The human's named priors of the decomposition, from [d-represent-first](r2-representation-20261008.md#d-represent-first).
- Only `docs/formulation/*.md` is edited. Nothing is committed, and no mac jobs run.
- Report to `~/ensomi/.sync/cp/scratch/style-formulation-rewrite-20261008/report.md`.
- The human reviews the result before any commit.

**Effects proposed for `RESEARCH.md` (for the session that owns it):**
- Node `style`: decisions [d-style-answers-20261008](#d-style-answers-20261008); rewrite in progress ([r-formulation-rewrite](#r-formulation-rewrite)).
- Node `identity`: ρ is the whole-chart identity, including level preferences; natural is defined relative to it.
- Node `control`: identity precedes the conditions (answer 5).
- Movement item 3: the six questions are answered, except q9 and the withdraw-before-start reading.
