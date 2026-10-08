# Style formulation re-examined against the 2026-10-07/08 findings (2026-10-08)

Report of a fresh Opus subagent, returned 2026-10-08 about 04:55 UTC. Read-only: no edits, no commits, no mac jobs. Brief: `~/ensomi/.sync/cp/scratch/style-formulation-20261008/brief.md`. It is the first step of the parallel style branch ([d-style-parallel](entry-point-resynthesis-20261008.md#d-style-parallel)).
- "Draft" means `docs/formulation/style-conditions-and-control.md` at `8da2bda` (branch `docs/style-formulation`, worktree `~/wt/ensomi-model-formulation`).
- `main`'s formulation files have not changed since the branch point `178ea3c`.
- Tags: [R] read in the cited source; [I] inferred; [P] proposed.

<a id="v-style-recheck-checks"></a>**Main-thread checks (2026-10-08).**
- The 10-05 answer "ρ is built as its own stage after stage 2" is item 2 of [d-formulation-answers](r2-style-formulation-check.md#d-formulation-answers).
- `notation.md` on the branch has the generator place row times ("Every row time lies in W ∩ [0, T]").
- The worker did not re-verify numbers; it relied on notes the main thread had checked.
- The proposed wordings were not checked against the rest of the draft.

---

<a id="o-rho-identity"></a>**Headline [I]: the draft's ρ is close to `identity`, with two differences that matter.**
1. **ρ covers organisation only.** Identity also includes chart-level levels: LN amount and length, and chord density, which carries difficulty.
2. **The draft has no supplied rows.** Supplied rows fix part of identity and most of the difficulty.

**A consequence:** the draft's release and return tests compare with the generator's own natural continuation from the same history. So they cannot catch R2's failure, where the level random-walks.

## 1. Open questions re-examined (recommendations are proposals)

**q2, levels below the default.** Proposal: (a), not defined.
- **Why:** default adherence is unmeasured. No phase-N checkpoint was selected, and the checkpoint noise is large ([s-diagnose-answer](r2-diagnose-synthesis-20261008.md#s-diagnose-answer)).
- **Later:** weaker levels can be added without invalidating requests.

**q3, strength for style directives.** Proposal: (a), property targets only.
- **Why:** no style readout is validated. Lens F1 is 0.76 with relational representations, and only 127 of 2,888 sections carry a human label ([o-fable-lens](r2-represent-fable-20261008.md#o-fable-lens)).

**q7, a chart seed and a supplied ρ that disagree.** Proposal: (a) extended. The rows fix what they fix, ρ governs the rest, and the generation record notes any seed–ρ or rows–ρ disagreement.
- **Missing source of evidence:** the question misses the supplied rows as a third source. They imply at most 41 % of identity and most of the difficulty ([o-skeleton-identity](r2-represent-skeleton-20261008.md#o-skeleton-identity)).
- **Why (a):** option (b), a blend, needs an evidence weight in a validated identity representation, which does not exist.

**q8, different quantities overlapping with no priority.** Proposal: (a), allowed, with the shortfall reported.
- **Why:** with supplied rows, LN and difficulty are coupled, so jointly unattainable pairs will be common [I].
- **Against (b):** under plan v4's rule against unhonoured fields, requiring a priority would make every such pair invalid today, because no generator implements priority.
- **(c):** dropping priority contradicts the human's original "explicit priority", so it is the human's call only.

**q9, and the agent reading (same Lens concept at different levels on overlapping scopes).** Proposal: (a), all three cases rejected, with case (iii) marked provisional.
- **Case (i):** a ban needs no readout; allowing it would.
- **Case (ii):** moot today, because R2 takes one ν per property.
- **Case (iii):** a whole-reference directive acts as a scoped identity change, so it should be revisited once identity has a representation.

**Still unconfirmed, not affected by the findings:** a request may be withdrawn or replaced before its start.

**Supported unchanged [I]:**
- ρ is neither stored in nor recomputed from (H, g). The random-walk diagnosis shows what an identity derived from history does.
- When no baseline is supplied, sampling chooses the identity; this matches [o-x3-skeleton-theta](r2-collapse-20261007.md#o-x3-skeleton-theta).

## 2. The draft against the new findings

<a id="p-rho-identity"></a>**2.1 ρ and chart identity.**
- **Passages:** "Style and chart properties", "Generation inputs", "Chart seed, baseline and random seed", "Natural continuation".
- **The issue:**
  - In the draft, ρ is a style identity, and LN share and difficulty are free properties when not requested.
  - So the chart-level LN amount and chord density have no owner, and a generator whose levels drift is not clearly excluded.
  - LN level is mostly one mapper's convention ([c-ln-placement-mapper](r2-representation-20261008.md#c-ln-placement-mapper)).
  - The draft omits supplied rows as a source of ρ.
  - A ρ reused on other rows transfers only if it is stated relative to what those rows imply.
- **Proposed wording if the human chooses (a) in question 1:**
  - "A style identity also holds chart-level preferences for property levels, such as how much and how long to hold and how densely to chord, where no target is in effect. They are preferences, not targets: they carry no readout obligation, and a property target overrides them within its scope."
  - "ρ specifies the chart-level choice that the music, any supplied row times and the requests leave open, held across the song."
  - ρ is established "from the evidence of the chart seed and of supplied row times, when given, and by sampling otherwise".
- **If the human chooses (b):** the system establishes and records a whole-song target for each such property.

<a id="p-supplied-rows"></a>**2.2 What a request can change when the rows are supplied.**
- **Passages:** "Section difficulty and gameplay demand"; "Scoped requests"; `notation.md`, "Legal space and implementation support".
- **What the rows fix** ([s-skeleton-achieves](r2-represent-skeleton-20261008.md#s-skeleton-achieves)):
  - density, completely;
  - most of the difficulty (band accuracy 70 %, star R² 0.83), yet 69 % of near-identical-row pairs sit in different bands;
  - chord placement in part;
  - LN share weakly;
  - hand balance not at all.
- **Two consequences [I]:**
  - A met target can be the rows' doing.
  - `notation.md` hides that the supplied rows come from one chart, with its mapper's identity and difficulty.
- **Proposed wording if the human chooses (a) in question 2:** "A generator may receive supplied row times. They are not committed history, a baseline or a request. They fix some properties, such as density, and constrain others, such as section difficulty. A target is attainable only within what they leave open, and its readout is read against the value the rows alone imply under a declared rows-only reference."
- **Also proposed:**
  - The Evaluation row "Does a target hold?" gains "with supplied rows, also against the rows-only expectation".
  - The record gains the rows' source and the rows-implied value per target.

**2.3 Requested levels of the Lens concepts.**
- **Passages:** "Style observations and style directives"; the "Unspecified, zero and absent" table; "Evaluation"; `gameplay-state.md`, "Scope, context, and evidence".
- **Missing:**
  - a readout declaration for style directives (ν covers only properties);
  - a rule for scopes longer than a Lens section, whose median is 5.5 s;
  - a distinction between the machine labeller's levels and the human's.
- **Proposed wording:** "A style directive declares its readout as ν does for a property: the assessor (a human assessment under the pinned vocabulary, or a named recognizer with the human assessments it was validated against), its chart context, and, for a scope longer than the sections the vocabulary was calibrated on, how one assessment is formed for the scope. Unvalidated machine readouts characterise a directive's effect and do not establish it. A generator trained on machine-proposed assessments records their source."
- **Also proposed:** name the levels as "the assessment levels of the pinned vocabulary version" rather than fixing three.
- **For the human:** whether a long scope is judged once or aggregated.

<a id="p-identity-held-test"></a>**2.4 Evaluation and record sections that rest on unvalidated measures.**
- **"Trained and validated" default.** Plans v4 and v5 operationalise it with unvalidated guards.
  - [P] "Validated" means checked against the human's judgments, or against measures shown to agree with them. Until then the record marks the default "not validated".
- **Validation rule.** [P] Extend the validated-or-human rule from recognisability to every style, playability and identity readout.
- **Nulls.** [P] Every comparison names its null: random seeds, neighbouring checkpoints and, for identity, two human charts on the same rows by the same or by different mappers ([s-step1-answer](r2-representation-20261008.md#s-step1-answer)).
- **Missing tests [P]:**
  - **"Is identity held across the song?"** Identity readouts by song position under one ρ, against the within-chart drift of human charts.
  - **"Does an override leave the baseline?"** The gap between continuations after release and continuations without the request, under the same ρ and seeds. It should shrink to the seed spread. This is distributional, but it adds a convergence requirement, so the human decides.
- **Record additions:**
  - each style directive's readout, its assessor and validation status;
  - the source of supplied rows;
  - "ρ: none (not implemented)" as an allowed entry.

## 3. Remaining conflicts (not in the rethink's table)

1. **Row times.** `notation.md` on both `main` and the branch has the generator choose row times. R2 takes them from an existing chart, and the branch has no input for them. Question 2.
2. **Plan v5 ordering against the map.**
   - ρ is stage R, after phase C. This follows the human's 10-05 answer.
   - That answer rested on the premise "build a persistent vector only after CE alone holds a condition over a song".
   - The 10-08 diagnosis says CE alone does not hold the level, and the problem map now makes `control` depend on `identity`. Question 5.
3. **Default operating point.** Plan v5 selects with unvalidated guards, phase N selected nothing, and the draft's default is "trained and validated". Question 3.
4. **Lens levels and label sources.** Plan v5 decision 10 lists a machine labeller as a label source, while gameplay-state says machine proposals are not human judgments. See 2.3.
5. **Request scopes for style.** gameplay-state gives no semantics for an assessment over a caller's long scope. See 2.3.
6. **Mirror.** Hand balance belongs entirely to identity and is directional. [P] "ρ declares how it transforms under μ."
7. **Audio (minor).** The draft assumes audio ("adapts to the music"). R2 v2 excludes it, while chord placement points to song information ([o-step1-mapper](r2-represent-step1-20261008.md#o-step1-mapper)). The record should state when the audio input is unused.

## 4. Questions for the human (recommended option first)

<a id="q-style-20261008"></a>
1. **Is ρ the chart identity, including chart-level levels (LN amount and length, chord density) where no target is set?**
   - (a) yes: ρ holds them as preferences that targets override;
   - (b) no: ρ stays organisation, and the system sets and records whole-song property targets;
   - (c) a separate identity input beside ρ.
2. **Should the formulation name supplied row times as an input?**
   - (a) yes: optional, and not history, ρ or a request, with targets acting on what the rows leave open;
   - (b) no: it stays an R2 deviation, documented in research docs only.
3. **What makes the default operating point "validated"?**
   - (a) the human's screen, or measures validated against it, with the record marking the default unvalidated until then;
   - (b) the plan's selection guards as they are;
   - (c) drop "validated" from the definition.
4. **When does the draft enter `docs/formulation/` on `main`?**
   - (a) after the revision for questions 1-2 and the human's review;
   - (b) now, as `8da2bda`;
   - (c) when control work starts.
5. **Does the 10-05 answer "ρ after stage 2" still stand, given that CE alone does not hold the level?**
   - (a) no: identity and ρ come before the conditions;
   - (b) yes.
6. **Keep the written defaults?** These are q2 (a), q3 (a), q7 (a, with the rows fixing what they fix), q8 (a) and q9 (a), including the ban on overlapping directives for the same style concept.
   - (a) yes, all;
   - (b) change a named one.

## Not done

- Plan v4 was read only in §2.3, §5, §7, §9.6 and the amendments; plan v5 in §0-3.8, §6, §7, §9 and §10.
- The fable and history reports were read in summary only.
- The draft's rendering and links were not checked.
