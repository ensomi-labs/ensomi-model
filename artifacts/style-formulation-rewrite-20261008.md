# Formulation rewrite: returned and reviewed (2026-10-08)

Shareable. The task is [r-formulation-rewrite](style-formulation-answers-20261008.md#r-formulation-rewrite), applying [d-style-answers-20261008](style-formulation-answers-20261008.md#d-style-answers-20261008).

- One fresh Opus subagent did the work and returned it about 18 minutes later.
- The result is uncommitted in `~/wt/ensomi-model-formulation`, branch `docs/style-formulation` on top of `8da2bda`.
- `git diff --stat 8da2bda`: 4 files, +525 / −389. File lengths: `README.md` 75 lines, `notation.md` 516, `gameplay-state.md` 397, `style-conditions-and-control.md` 462.
- The harness refused the subagent's write of its report to the scratch path. The main thread saved the report here instead (section "Subagent report").

<a id="v-rewrite-review"></a>
## Main-thread review (bounded)

**Read in full:**
- `README.md`;
- `style-conditions-and-control.md`;
- the `notation.md` diff;
- the `gameplay-state.md` diff;
- a residual grep for baseline, seed, validat, evaluat, R1 and R2. The only hits are the difficulty evaluator of ν and the "invalidated" replay cache, both valid.

Links were not re-run; the subagent reports 61 internal and 34 inbound links resolving.

**Applied as decided:**
- ρ is the chart identity, with organisation and level preferences, overridden by targets and directives within scope.
- Natural continuation is defined relative to the identity, and the identity comes first in the definitions.
- Prefix, identity and sampling randomness are separate. "Chart seed" is gone, and so is the random seed.
- Strength applies to property targets only and has levels below the default. "Validated" is dropped.
- Overlaps on different quantities are allowed, with the shortfall declared.
- Both Evaluation sections and the generation record are removed, while the locality conditions and the monotone strength requirement stay as requirements.
- The decomposition section is added, and local edit is absent.

New and sound: "returning to the identity's level preference is not compensation", stated with an example.

<a id="o-rewrite-points"></a>**Points for the human [I]:**
1. **Research results inside the formulation.**
   - Two subsections of `notation.md` state corpus observations as facts, unqualified: "What given head times fix" and the corpus verdicts in "Assumptions".
   - One of them rests on 4 songs, direction only: "head times that an audio timing model generated lost the difficulty" ([o-skeleton-shift](r2-represent-skeleton-20261008.md#o-skeleton-shift)).
   - The README's own authority section says experimental results belong in research documentation.
   - Options:
     - (a) keep the reasons, the assumptions and the consequences in the formulation, and move the observations and verdicts to `docs/research/` with a link;
     - (b) keep them, worded with their strength;
     - (c) keep as is.
2. **The grid Γ restricts close-only rows to grid positions.** This is the subagent's choice, not the human's. The alternative is that Γ only supplies coordinates and close-only rows may fall at any legal time. R1 draws release candidates from source event times, not from a grid.
3. **The "Interpretable outcomes" bullet** says that the identity, the governing requests and the shortfalls accompany each generation. It is the subagent's choice, left over from the removed generation record. Keep it, or cut it to the shortfall clause.
4. **Still open from before:**
   - same-attribute style-directive overlap (q9);
   - withdrawing or replacing a request before its start;
   - with a prefix and no identity, whether the system may use the prefix (the text leaves this unfixed);
   - how an identity transforms under the mirror (left unfixed);
   - whether a level below the default sits above "free" in a defined way.
5. **Root `README.md` lines 93-97** still say "optional style and gameplay-demand controls". The line is accurate but does not mention the identity or the decomposition. The file is outside the brief's scope and was not edited.


<a id="r-formulation-pr"></a>**Committed and opened for review, 2026-10-08, at the human's request ([private, local](private/human-inputs/515ba54a-4eb7-4951-92c7-3c986083259c.md#prompt-1)).**
- The rewrite is commit `1a1b8ff` on `docs/style-formulation`, pushed to `origin`.
- PR [ensomi-labs/ensomi-model#17](https://github.com/ensomi-labs/ensomi-model/pull/17) into `main` (base `178ea3c`). Not merged.
- The PR description lists the five points above as open for review.
- Checks: 95 relative links resolve and `git diff --check` is clean. Documentation only.

## Subagent report

Verbatim, as returned:

> The four formulation files are rewritten and left uncommitted on `docs/style-formulation` at `8da2bda`. All links and anchors resolve. The report file was not written (harness refused subagent file writes there).

### 1. Per file: structure and main changes

**notation.md.** Structure: Chart object and time; Row language and long notes; Canonical hand-role coordinates; Committed history and exact replay; Legal continuations; Generation and optional controls; Provisional branches and prefix commit; Legal space and implementation support; Worked example; **Decomposed problem: given head times** (new, last), with subsections Head times and the decomposed legal set, Why this decomposition, What given head times fix, Assumptions, An implementation of the decomposition.
- "Chart seed σ" → "a supplied prefix (H₀, g₀)": committed history and nothing else, not a request, not by definition evidence for the identity. The random-seed sentence is removed.
- ρ bullet: the chart identity, the whole-chart choice the music, any given head times and the requests leave open, held across the song; always in effect; retained across calls; changes only by an explicit update at a committed boundary.
- Generation section: pointer that the decomposed problem adds head times and a grid; with 𝒰 empty, generation is natural continuation under the identity; requests may be unattainable under given head times.
- Legal space and implementation support: evaluation sentences dropped; it now says the decomposition changes the problem and is not a support limitation.
- New section: definitions (head row, close-only row, Heads(H̄)), inputs 𝒯_h and Γ (positions P_Γ, where close-only rows lie), the decomposed legal set, Heads(H) = 𝒯_h ∩ (0⁻, g], the decomposed generation law with X kept; two derived facts (empty continuation legal only if W has no head time; a head row needs a closed lane at s⁻); reasons, qualitative corpus observations, two consequences, three assumptions with their corpus status; an orientation paragraph on an implementation, marked as not part of the problem.
- All anchors kept.

**gameplay-state.md.** Structure unchanged except "Evaluation questions" removed.
- Schedule wording ("after the initial chart dataset is complete", "will be established") → "open".
- Checking wording made definitional: approximation claim relative to a declared continuation family, horizon range, response comparison and discrepancy; reproducibility distinct from sufficiency; sufficiency defined against independent target responses; response sufficiency defined by the target responses, not style recognition.
- "Supervised presence loss" clause rephrased without the training reference, proposition kept.
- One definitional fact salvaged from the removed probe paragraph, into the frontier section: the frontier's domain depends on occupancy; relating two frontiers needs continuations legal from both histories or a declared correspondence.
- Controls: "baseline" → "chart identity"; "a style directive carries no strength"; "No control requires…"; "Generation, and any readout computed on a branch, must preserve" branch isolation.
- The six required anchors kept.

**style-conditions-and-control.md.** Structure: Style and chart properties (Style observations and style directives; Section difficulty and gameplay demand); Chart identity (Identity and prefix); Natural continuation; Generation inputs; Chart properties and measurement semantics; Scoped requests (Activation, expiry and locality; Overlap and priority; Unspecified, zero and absent); Target strength; Release and return to natural; Overall target. Identity and natural continuation precede the request sections because the conditions are defined in terms of them.
- Chart identity: organization plus chart-level level preferences (LN amount and length, chord density); preferences, not targets; a property target overrides within scope; a style directive overrides the attributes it specifies; governs only what given head times leave open; sources: supplied, extracted or calculated from any chart, or established by the system; an input, not chart state; hand balance directional, transformation under μ unfixed.
- Identity and prefix: prefix is history only, need not agree with the identity; the prefix's statistics become neither targets nor preferences; minimal rule for a prefix without an identity; the single sampling-randomness sentence.
- Natural continuation: defined relative to the identity; a free property follows natural continuation under the identity, including its level preference, with no hold, compensation or pull; the dependency written out (without an identity the chart drifts and natural names no law, so the identity comes first in the definitions).
- Section difficulty rephrased definitionally; attainable only within what given head times leave open.
- Locality: fixed inputs include head times and grid; "together the conditions exclude any presence or announce signal outside the scope"; the generation-record reference dropped.
- Overlap: different quantities all apply; shortfall "declared with the generated continuation, not silently applied"; priority optional and only for that case; a property target's precedence over the identity's level preference stated as not a priority relation.
- Table: new Prefix and Head-times rows; strength "above or below the default"; ρ row is the chart identity.
- Target strength: property targets only; default "the one it is trained for", "validated" removed, the formulation does not fix how close the default comes; new "Lower levels" bullet (still a target, not a release); "Monotone" its own bullet; seeds replaced by "variation within the identity"; measurement paragraph, "holds temperature fixed" and "No level below the default" removed; "Lens prominence" → "Ordinal strength of an observation".
- Release and return to natural: returns to the identity's organizing and level preferences; returning to the level preference is not compensation; "a persisting consequence is not continued enforcement".
- Overall target: four bullets (diversity, variation, composable control, interpretable outcomes) and a capability paragraph.
- Removed: Generation record and Evaluation sections.

**README.md.** Target statement "follows a chart identity", plus a paragraph that generation is posed on the decomposed problem. Ownership table updated. "Research direction" → "Open questions" (response specification open; demand-state adequacy defined against target responses; style open-ended, no identity representation or strength scale fixed; the decomposition's assumptions hold only in part, the full problem remains the target). Authority: the decomposition is a declared problem scope; identity and request semantics are interface requirements; identity representations have research status; the principle "defines the problem, not how results are checked"; evaluation protocols belong in research documentation.

### 2. Decision trace

- A1 → style: Style and chart properties; Style observations and style directives.
- A2 → style: Scoped requests; notation: 𝒰 bullet.
- A3 → style: Scoped requests, q^property bullet.
- A4 → style: Natural continuation; Activation, expiry and locality; Release and return to natural.
- A5 → style: Activation, expiry and locality.
- A6 → style: Activation, expiry and locality; notation: 𝒰 bullet.
- A7 → style: Overlap and priority; gameplay-state: Controls.
- A8 → style: Target strength (Default bullet, distinct-from list).
- A9 → style: Chart identity; notation: Committed history, ρ bullet.
- B1 → style: Chart identity, Natural continuation, table, Release; notation: ρ bullet; all "baseline" wording replaced.
- B2 → style: Natural continuation, third paragraph.
- B3 → notation: Committed history; style: Chart identity, Identity and prefix.
- B4 → style: Target strength, table; gameplay-state: Controls.
- B5 → style: Overlap and priority; gameplay-state: Controls.
- B6 → style: Target strength, Default bullet.
- B7 → notation: Decomposed problem; README; style: Section difficulty, Chart identity, Generation inputs, locality inputs, table.
- B8 → all four files; section 3.
- B9 → not added.

### 3. Removed material (none moved)

- Style "Evaluation" table (9 questions) and the recognizability paragraph; the strength measurement paragraph; "default adherence is a measured property… reported for each property"; "a strength comparison holds temperature fixed". Could live in a research doc on control evaluation or beside the plan's receipts.
- "validated" (B6); "No level below the default is defined" and "adherence of a style directive is a quantity of the same kind" (replaced by B4).
- Generation record list (identity source label, chart and random seed, requests including withdrawn ones, per-target readout and deviation, crossing objects). Suits an implementation's receipt specification. Kept parts: see section 4, item 4.
- "Chart seed, baseline and random seed" section (seed as baseline evidence, sampling among identities a seed supports, seed-vs-baseline rule, random-seed definition). Superseded by B3.
- gameplay-state "Evaluation questions" (6-row table, the probe paragraph with its definitional part salvaged, the closing paragraph) and checking clauses ("full-history reference model…", "chart-only, demand-only, and joint recognition comparisons… must be declared", "examples exposed during calibration must remain grouped… for evaluation", "supporting response comparisons", "separately evaluated"). Could live in a research doc on recognition and demand evaluation.
- Schedule wording in gameplay-state and the README: notes.
- "supervised presence loss": training recipe, rephrased.
- notation: "must be evaluated against downstream chart quality…" and "pointwise timing recall alone…": a research doc on timing representations.
- README paragraph on the initial beatmap-lens dataset: duplicated gameplay-state.
- Style checking wording ("Meeting a difficulty target shows…", "Persistence alone does not prove…"): rephrased definitionally.
- Old "style identity" definition: replaced by the chart identity.

### 4. Agent choices and points for the owner

1. Same-attribute style overlaps (A7): kept as written, unconfirmed.
2. Withdraw or replace before start (A6): kept as written, unconfirmed.
3. Prefix supplied, no identity (B3): "the system establishes one as it does without a prefix; this formulation does not fix whether it takes the prefix into account."
4. Generation record: removed. Kept: shortfalls "declared with the generated continuation, not silently applied"; a system-established identity "can be supplied again"; agent choice, an "Interpretable outcomes" bullet that the identity in effect, the governing requests and every declared shortfall accompany each generation, and that the property query reads the result. Could be cut to the shortfall clause.
5. Identity under μ: left open.
6. Level below the default: "weighs the target less than the default balance does… may leave a larger deviation… still a target, not a release"; agent choice that the monotone requirement spans all levels; open whether a lower level sits above "free" in a defined way.
7. Γ restricts close-only rows to P_Γ (closes inside head rows unrestricted). Alternative: Γ only supplies coordinates. R1 in `docs/research/bounded_typed_continuation.md` draws release candidates from source event times, not a beat grid; P_Γ can hold that set, but "tempo and phase" describes the other implementation; "a declared set of release positions" is an alternative wording.
8. Audio X stays in the decomposed law; not using audio is an implementation property.
9. A supplied prefix must satisfy the head-time condition, not the grid condition.
10. Two derived legality facts in the decomposition.
11. "Ensomi's generation is posed on the decomposed problem" renders "current problem scope" without indexical wording; "one implementation studied in Ensomi's research" avoids confusion with R1.
12. Assumption verdicts are the agent's qualitative reading of the brief's facts: (1) partly supported, partly contradicted; (2) contradicted for choreography, control limited where head times fix a property; (3) supported for density and its profile and for most difficulty on human head times, not for identity beyond head times, song-informed placement, or difficulty on generated head times.
13. "Returning to the identity's level preference is not compensation" spelled out with an example.
14. A property target's precedence over the identity's level preference stated as not a priority relation.
15. "Style adherence" stays in the Controls list, as on main.
16. Salvaged frontier-domain paragraph in gameplay-state: drop if read as evaluation.
17. notation's "derived features for generation or evaluation" kept (names a use).

Not edited: root `README.md` lines 93-97.

### 5. Checks

- 61 relative links inside `docs/formulation/`, 0 failures (GitHub-slug checker `/tmp/sfr-check/links.py`, tested on a known-bad file first). 34 inbound links from the five named files, 0 failures; no reference to a removed anchor anywhere in the repository. Required anchors all present.
- Anchor changes against `8da2bda`: README `#research-direction` → `#open-questions`; gameplay-state removed `#evaluation-questions`; style removed `#chart-seed-baseline-and-random-seed`, `#evaluation`, `#generation-record`, `#identity-and-control-requirements`, `#natural-continuation-and-return-to-natural`; style added `#chart-identity`, `#identity-and-prefix`, `#natural-continuation`, `#release-and-return-to-natural`, `#overall-target`; notation added `#decomposed-problem-given-head-times`, `#head-times-and-the-decomposed-legal-set`, `#why-this-decomposition`, `#what-given-head-times-fix`, `#assumptions`, `#an-implementation-of-the-decomposition`.
- `git diff --check` clean; math uses only `$` and `$$`; leakage recall batteries leave only valid-context hits; no "seed", "baseline", "validated", run or code names, or numbers remain.

### 6. Not done

Report file not written (harness block); GitHub rendering not previewed; no independent review; the answers material and `artifacts/private/` not opened; numbers in the notes not re-verified (none used in the documents); root README not edited.
