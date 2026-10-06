# R2 condition plan v2 checked against the human's style formulation

Shareable. Written 2026-10-05 by the main thread (Claude, control plane). It compares [r2-condition-plan-v2](r2-condition-plan-v2.md) with the style and controllable-generation formulation the human gave the same day. The human said that formulation takes precedence over the plan and states the long-term problem ([private, local](private/human-inputs/aa6818ba-9ce4-4d72-b1de-e7d78a7ffbee.md#prompt-1)). Code cited at `r2/train` `7d9640a`, with paths relative to `src/ensomi_model/r2/`. Everything below is the agent's reading and proposal, and none of it has been reviewed. The formulation is not in `docs/formulation/`.

<a id="f-summary"></a>
## The formulation in the agent's words

- **Style and chart properties are different kinds of thing.** Style is how timing, actions, hand roles, motifs and transitions are organised. Its dimensions are open-ended: named, taken from references, or learned. Beatmap Lens concepts are only partial anchors. A chart property is a scoped measurement such as LN share or section difficulty. A property says what the result must measure. It does not say how the result is organised.
- **Baseline style ρ.** A reusable style specification, taken from a chart seed, references, explicit preferences or sampling. If none is given, the system sets one. It persists across continuation calls and local overrides until it is explicitly changed.
- **Requests.** A request is u = (S, style directive, property directive, η). Its scope S is in musical time and does not depend on inference chunking. A property directive is a target or a range. η sets the transition and composition policy. Overlapping requests compose by explicit priority, with a declared trade-off when they cannot all be met. A section target applies to the whole section, not to every smaller window inside it.
- **Readouts.** Properties_ν(chart, S) uses the same definitions as the targets. ν declares the counting and boundary conventions for LN share, and the evaluator, context and profile for difficulty. A readout measures the actual chart. It never echoes the requested value.
- **Unspecified, released, natural.** An unspecified attribute is free to vary in any way compatible with the baseline, the music, the history and the active targets. Unspecified is not the same as zero or absent. Natural means no style override; active property targets still apply. Releasing a directive returns generation to the baseline's organising preference, starting from the actual history after the override. The target is not the seed's statistics, not a population average and not a counterfactual trajectory. An expired directive stops being an objective, but its consequences may persist: a hold that ends late, a motif that completes. An expired LN target creates no obligation to compensate later. A local override does not rewrite the baseline.
- **Seed.** A chart seed is an optional committed prefix. It is evidence for ρ. Its LN share and difficulty become targets only if they are requested.
- **Overall target.** Diversity between identities, variation within an identity, composable control over organisation and properties, and readable outcomes.

<a id="c-conflicts"></a>
## Conflicts: plan v2 or the code must change

| # | Plan v2 / code | Formulation | Proposed change |
| --- | --- | --- | --- |
| C1 | LN share, difficulty and Lens style dimensions are three kinds of one condition track (§3.3, §3.8, stage 4). | Property directives and style directives differ in kind and in how they are released. Lens names are partial anchors. | Split the request schema into property directives and style directives. Lens dimensions become named-attribute style directives. Stage 4 is relabelled as that path. |
| C2 | Several definitions of each quantity. The LN-share label is computed inline (`conditions.py:43-52`), `labels.py:42` defines a second LN share, and the probes compute realised share themselves. Star comes from `labels.tiled_star`, described only by a version string (`labels.py:39`). No ν object exists, and the half-open [a, b) convention is stated only in a docstring. | Targets and readouts share one definition with a declared ν. | Add one `properties` module with an explicit ν. LN share: ownership by head time, half-open [a, b) in ms, one count per object. Difficulty: evaluator `compute_mania_star_rating_20241007`, context = the section's own objects tiled to 240 s, head ownership, untrimmed tails, at least 30 s, 4K at clock 1.0. Labels, frame counters, evaluation readouts and request validation all use it. A test checks that labels equal readouts on source charts. |
| C3 | Difficulty request: "the caller gives a span and a requested residual" (§4.1). | Targets use the same definition as queries. | The request carries an absolute section-difficulty target or range in ν. The system computes v_res = target − b(S) internally and flags requests outside the trained range as infeasible. Readouts report absolute difficulty and the residual. Long-term tension: b(S) needs the given skeleton, which holds in R2 but not once timing is generated. The residual stays an R2 parametrisation and the interface does not depend on it. |
| C4 | Natural behaviour and the stretch between conditioned parts are deferred with no default (Q1, Q2, §11). Guard (i) is "reported only". | Defaults are set: an unspecified or released property is free. Population-average style, seed statistics as a target, holding the last value, and compensation are all excluded. Transitions happen only under a declared η. | For training, base CE on unconditioned source rows is consistent, since the source is one realisation of free-property baseline continuation; Q1(a) holds for training. Q1(b) and Q1(c) are excluded for properties. Q2(a) is the default. Q2(b) becomes an explicit request. Q2(c) and Q2(d) become η policies. Guard (i) becomes binding at distribution level: natural LN share on a panel against the real continuations of the same prefixes, never per chart against the seed. Needs the human's confirmation, because the human deferred this. |
| C5 | Presence bit on every row once any interval of that kind exists (`features.py:291`); v2 keeps it as the default (Q-A). | A directive takes effect at activation and stops at expiry. Unspecified rows are free. | Default `none`: outside its scope a row sees exactly what it would see with no request. `announce` exists only as an η policy, with its own lesion test. T-P3b(i) then binds with no exception. |
| C6 | `validate_track` rejects overlapping intervals of one kind (`conditions.py:94-103`). | Overlapping requests compose by explicit priority. | Requests may overlap. A deterministic resolution by stated priority produces the effective, non-overlapping track the model reads, and the receipt logs it. An overlap with no priority is rejected, not resolved by a hidden rule. Training is unchanged, because source values never conflict. A trade-off across kinds is not trainable by CE and is reported, not claimed. |
| C7 | LN pieces are in beats (`conditions.py:20, 36-41`). Star cells are 30/60 s (`labels.py:37`); v2 proposes 30/60/120 s at 10 s offsets. | Scopes are in musical time and independent of chunking. | Depends on what "musical time" means (question 3). If it means beats or bars, difficulty scopes become bar spans with a 30 s floor, which changes the stage-0 relabel and Q-C. |
| C8 | Point values only. | A target or a range. | Encode (lo, hi) in the frame; a point target has lo = hi. Whether stage 1 trains ranges is question 4. |

<a id="g-gaps"></a>
## Gaps: the formulation requires it and v2 is silent

- <a id="g-baseline"></a>**G1 Baseline style ρ.** R2 has no ρ. Identity exists only in the committed history. Consequences:
  - Without a seed, nothing sets a baseline except what has been generated so far.
  - An override section enters the history and shifts the identity the history implies, so R2 cannot guarantee by construction that a local override leaves the baseline alone.
  - On return to natural, the closest R2 can come to the baseline is the history before the override.

  Evidence that the current model holds no identity across a song: seed persistence fades by 256-512 rows, the landmark memory is unused, and at 39.49M the variance across random seeds on one skeleton was twice the variance across skeletons ([a-r2-fable-judgment](r2-average-and-control.md#a-r2-fable-judgment)). The judgment's ordering was a persistent vector only after CE alone holds one condition over a song ([q-r2-after-judgment](r2-average-and-control.md#q-r2-after-judgment), item 5). The formulation now says what that vector means. Proposed: reserve ρ as an optional input in the interface, measure the behaviour it should have (G3), and build it as a separate stage after stage 2. Question 2.
- <a id="g-eta"></a>**G2 η policy.** Proposed default: immediate activation at the start of the scope and immediate release at its end, with consequences persisting and no compensation, plus explicit priority for overlaps. Announce, ramp and hold come later, each as a named input with a lesion test.
- <a id="g-diagnostics"></a>**G3 Diagnostics for what the formulation targets.** All are free-run probes, with no training:
  - (a) **Release without compensation.** An LN override far from the prefix's share, then free continuation. Compare post-span LN share and organisation statistics with a natural continuation from the same prefix and random seeds. Compensation means post-span share moves opposite to the override by more than 2 SE.
  - (b) **Identity kept under a property change.** Same prefix and random seeds, with and without the property request. Measure the change in organisation the property does not govern: chord-size histogram, the Lens query evidence (jack, stream, trill), hand-role balance.
  - (c) **Within-identity versus between-identity variation.** Variance across random seeds per prefix against variance across prefixes. Reported, with no claim.
  - (d) **Receipts.** Each generation lists every request with its ν, its target or range, the realised readout and the effective track after resolution.
- **G4 Style directives beyond named attributes** (references, relative edits, learned dimensions): out of R2's scope; listed so nothing built rules them out.

<a id="k-consistent"></a>
## Consistent with the formulation: keep

- **Governed-factor terms.** The LN term scores LN versus tap given the heads (§3.3). That is what a property directive should govern: the result, not the organisation.
- **Release ownership by the LN's head span** (§3.1). It matches a hold that ends after the control window, and it becomes part of ν (C2).
- **Span-local committed counters.** They end at the span end: the target is section-level and no obligation continues after it.
- **Unconditioned source rows.** Base CE on them, the inverse-probability weights and the natural manifest (§3.2, §5.1, §5.3).
- **Source values only.** Source decisions are never relabelled with values they lack (§5.1).
- **Span placement.** Spans are drawn on the song and windows are chosen around them, so scopes stay independent of chunking (§5.1).
- **Long spans** (runs, whole song; Q-B (a)): the formulation allows a scope of any section.
- **"Unreviewed = no interval" and explicit "absent"** (§3.8): this matches unspecified ≠ absent.

Questions the formulation does not touch: Q-D, Q-E, Q-F, Q-G (direction 2), Q-I, Q-J, Q-L and the infrastructure. Q-K gains one point: the vocabulary is open-ended.

<a id="p-v3-delta"></a>
## Proposed amendments for plan v3 (agent proposal, pending the questions)

- **Stage 0 adds:**
  - the `properties` module with ν (C2);
  - the request schema u = (S, style directive, property directive as target or range, η, priority), resolved into the effective track (C1, C6, G2);
  - absolute difficulty requests (C3);
  - presence default `none` (C5);
  - range encoding reserved (C8);
  - receipts with intended and realised values (G3d);
  - star relabel scopes per question 3;
  - G3a-c added to plan §6.
- **Stage 1:** arms unchanged, presence `none` in both, guard (i) binding at distribution level.
- **Stage 4:** the named-attribute style path.
- **New stage after stage 2:** the baseline ρ, unless question 2 is answered otherwise.
- **§11:** Q1 and Q2 leave the deferred list if the human confirms C4.

<a id="q-formulation"></a>
## Questions put to the human

1. Confirm the defaults read in C4 and C5: free on release, no hold, no compensation, no presence signal outside the scope, transitions only by explicit η.
2. Baseline ρ: reserve and diagnose now and build it after stage 2, or build it in this repair.
3. "Musical time" for scopes: beats or bars, or song time as opposed to chunk index.
4. Ranges: reserve the encoding and train point targets in stage 1, or train ranges from stage 1.

<a id="d-formulation-answers"></a>
## Human answers, 2026-10-05

Recorded from a selection among the agent's options ([private, local](private/human-inputs/aa6818ba-9ce4-4d72-b1de-e7d78a7ffbee.md#answer-1)). Each answer is read within the option it selected.

1. **Adopted: the defaults read in C4 and C5.** A released or unspecified property is free. No hold, no compensation, no population average. No presence or announce signal outside a scope. Transitions happen only under an explicit η. Effects: Q1 and Q2 leave plan v2's deferred list, which supersedes [d-natural-gap-deferred](r2-average-and-control.md#d-natural-gap-deferred) on these points; the presence default becomes `none`; natural-LN guard (i) becomes binding at panel-distribution level.
2. **Reserved: the baseline ρ.** An optional slot in the interface and the G3 diagnostics come now. ρ is built as its own stage after stage 2.
3. **Song time is enough.** Scopes are positions in the song, independent of inference chunks. Seconds-based difficulty cells stay. C7 is not a conflict.
4. **Ranges: not answered.** The question was unclear. Provisional plan default: the frame reserves (lo, hi), stage 1 trains point targets, and the question stays open in v3.

<a id="s-plan-v3"></a>
## Plan v3, 2026-10-05

A fresh Fable subagent wrote [r2-condition-plan-v3](r2-condition-plan-v3.md) (proposed; it supersedes v2 as the working plan). It adopts C1-C3, C8, G3 and G4 and amends C6 and G2:

- **C6 (overlap):** overlap is defined within one kind only; an LN scope and a difficulty scope on the same rows both apply.
- **G2 (η):** "hold" is not an η policy, because holding a value longer is just a longer scope.

It adds ten rows of its own. The main one, N1: the token conditioner (`features.py:308-330`) shows intervals that have not started yet, so it is an announce form and is not admissible as the default. It also separates random, training and chart seeds, and adds an explicit-zero against unspecified lesion test. In §12 the questions are rewritten in plain language with chart examples; Q-B, Q-E, Q-F, Q-R (ranges) and Q-G block stages.

The main thread reviewed the ruling table, §3, §7.7, stage R and §12. It spot-checked the token-conditioner claim and the separate LN-share definition in `report.py:46-48`, and found no verbatim copy of the private formulation. One point to revisit before stage R: its return-to-natural test compares with the organisation statistics before the override. The formulation does not tie the return to the seed's local statistics. The comparison should be distributional, against ρ-conditioned natural continuations from the same post-override history.

<a id="d-target-strength"></a>
## Decision: targets only, with an optional strength (human, 2026-10-06)

Source: [private, local](private/human-inputs/aa6818ba-9ce4-4d72-b1de-e7d78a7ffbee.md#prompt-2). This settles plan v3's Q-R and amends the formulation's "targets or ranges".

- A property directive is a target. It is never a range.
- The interface is to offer an optional strength for a target, for example LN share 0.2 at the default strength or a higher one. The human compares it to Lens prominence.
- The first implementation trains and serves exact targets only. Strength is defined in the formulation and not built yet.
- What strength means operationally is open; the formulation work is to propose it. It must stay distinct from priority (overlap resolution), from the target value and from sampling temperature.

Next, per the human: an Opus subagent writes the style formulation into `docs/formulation/` and resolves its conflicts with the existing documents (worktree `~/wt/ensomi-model-formulation`, branch `docs/style-formulation` from `main` `178ea3c`). A separate Opus subagent then revises the implementation plan from that result.

<a id="d-formulation-answers-2"></a>
## Human answers on the formulation draft, 2026-10-06

Recorded from a mix of option selections and free text ([private, local](private/human-inputs/aa6818ba-9ce4-4d72-b1de-e7d78a7ffbee.md#answer-2)); the questions are q1, q4, q5 and q6 of [style-formulation-rethink](style-formulation-rethink.md#q4).

- **Hold closes after the scope are natural.** When a hold starts inside a scope and is closed after it, the close does not see the request. ν may still count the hold in the scope, and the record lists it. For R2, this removes the release pointer's birth-role reading of an expired scope (plan v3 N5, T-O).
- **Same-property overlap is invalid input.** Two targets for one property on overlapping scopes are rejected; there is no composition and no priority between them. Agent reading, to confirm: the same holds for style directives on the same attribute.
- **Requests must precede their scope.** A request is valid only while the committed frontier is before its declared start. There is no mid-scope activation. Cancelling an active request was not addressed.
- **Default strength is the trained operating point.** It is the operating point the generator is trained and tuned for, balancing playability, style and control. It is not exact attainment and not maximal adherence; higher levels weight the target more.

The same Opus subagent is applying these to the draft. The plan revision follows.

<a id="d-plan-v4-answers"></a>
## Human answers on plan v4, 2026-10-06

Source: [private, local](private/human-inputs/aa6818ba-9ce4-4d72-b1de-e7d78a7ffbee.md#answer-3).

- **Q-B: yes.** Training LN scopes run past 64 beats, up to the whole song.
- **Q-F: Astra decides the learning-rate schedule.** It may change between training phases, informed by R1's staged recipe. The agent adds a constraint from the plan's matched-comparison rule: the schedule is fixed before arm comparisons and is identical across arms.
- **q10: no cancellation after a scope starts.** Before the start a request may still be withdrawn or replaced; that part is the agent's reading. The formulation now says so, in `docs/formulation/style-conditions-and-control.md` (renamed at the human's request, `8da2bda`).
- **Q-E: not answered.** The human asked what the LN-emphasis arm is and why it is needed.
- **Q-G: not understood as asked.** The human expects difficulty control to be weak with fixed head times, but says the response must still be tuned. Agent reading, to confirm: a training signal on the realised difficulty of generated sections (option B).

The human also asked whether the plan is clear enough for an agent to implement, with extra attention on its ML-specific parts. Answered afterwards ([private, local](private/human-inputs/aa6818ba-9ce4-4d72-b1de-e7d78a7ffbee.md#answer-4)):

- **Q-E:** the condition terms are on in the new recipe, with no separate arm.
- **Q-G: option (B).** A term on the realised star of generated sections, which is the OK for own-sample training for difficulty.
- **Process:** start implementing now. No build-time estimates and no AI-reviews-AI gates. Invent no training detail that was not discussed. The human checks before any actual run. Plan v4 carries these answers as amendments ([v4-amendments](r2-condition-plan-v4.md#v4-amendments)).

Side observation: `docs/formulation/notation.md` ("Generation and optional controls") and `gameplay-state.md` ("Controls") predate the formulation. Their generation formula has no ρ and no request set, and they say an absent style request permits "the learned natural style distribution". Whether the formulation enters `docs/formulation/` is the human's call.
