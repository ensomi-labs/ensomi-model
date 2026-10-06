# Style formulation in `docs/formulation/`: rethink and conflict resolution

Shareable. Written 2026-10-06 by an Opus subagent (Claude, control plane) for the main thread. It records how the human's style and controllable-generation formulation ([private, local](private/human-inputs/aa6818ba-9ce4-4d72-b1de-e7d78a7ffbee.md#prompt-1)) went into the formulation documents, with the human's later decisions applied ([private, local](private/human-inputs/aa6818ba-9ce4-4d72-b1de-e7d78a7ffbee.md#answer-1), [private, local](private/human-inputs/aa6818ba-9ce4-4d72-b1de-e7d78a7ffbee.md#prompt-2)): a property directive is a target only, with an optional strength; a released or unspecified property is free; generation sees nothing of a request outside its scope unless η says so; scopes are song time; ρ stays in the formulation. The first draft is committed on `docs/style-formulation` (from `main` `178ea3c`) as `c4e3d01`. The human then answered q1, q4, q5 and q6 ([private, local](private/human-inputs/aa6818ba-9ce4-4d72-b1de-e7d78a7ffbee.md#answer-2)). The revision that applies those answers is in the worktree `~/wt/ensomi-model-formulation`, uncommitted ([Revision after the answers](#revision-answer-2)). Nobody has reviewed either version. The conflict check this builds on is [r2-style-formulation-check](r2-style-formulation-check.md).

<a id="files"></a>
## Files changed

| File | Change |
| --- | --- |
| `docs/formulation/style.md` (new) | The human's formulation, rethought: style versus chart properties, ρ, 𝒰, ν, scopes, activation and locality, overlap and priority, unspecified/zero/absent, target strength, seeds, return to natural, identity requirements, the generation record, evaluation. |
| `docs/formulation/notation.md` | The generation formula takes ρ and 𝒰 in place of `c^style`, plus the property query. The chart seed is defined as an initial committed pair. (H, g) is the source of truth for the chart only, and ρ and 𝒰 are inputs that are not derived from it. Property targets are added to the list of things that leave legality alone. The worked example now says "Neither the baseline nor any request". |
| `docs/formulation/gameplay-state.md` | Ownership split with style.md. "Controls" rewritten: baseline instead of "learned natural style distribution", demand-request scope, strength among the separate meanings, priority composition. Lens "strength" renamed "ordinal strength". Pointers from style observations, presence and section difficulty to style.md. |
| `docs/formulation/README.md` | The target includes distinctive organization, diverse realizations and composable controls. Table row for style.md. A short paragraph on open-ended style. The authority section names request semantics as interface requirements and the calibration of strength as research status. |

Anchors kept: `notation.md#generation-and-optional-controls`, `#provisional-branches-and-prefix-commit`; `gameplay-state.md#controls`, `#style-observations`, `#community-observations`, `#presence-and-ordinal-strength`. A local script resolved every relative link and anchor in the worktree's Markdown. The two failures it found are pre-existing and outside the formulation (`docs/guides/technical_analysis_writing.md`).

<a id="revision-answer-2"></a>
### Revision after the human's answers (uncommitted, on top of `c4e3d01`)

| File | Change |
| --- | --- |
| `style.md` | "Activation, expiry and locality": the effective scope S ∩ (g_u, T] is gone. A request is valid only if the committed boundary g at which it is added precedes its start (g < a), and it is rejected otherwise. Targets concern S. Cancellation of an active request is kept, and a change is a new request that must start after the cancellation boundary. "Overlap and priority": same-quantity overlaps (the same property under any ν, or style directives on the same attribute) are invalid input and rejected. Priority remains only to order different-quantity directives that cannot all be met, and it is optional. "Target strength": the default is the generator's trained and validated operating point, which balances playability, style and control; the reference law q_τ is dropped; the effect is measured against the default and its costs include playability. Relative edits are now relative to generation without the directive. At most one target per property in a request. The generation record marks cancellations. The identity requirements say a higher strength may lower playability. |
| `notation.md` | The 𝒰 bullet states the validity rule. The pointer to "Overlap and priority" now says it defines which overlaps are invalid. |
| `gameplay-state.md` | "Controls": same-quantity overlaps are invalid; different-quantity overlaps compose under optional priorities. |
| `README.md` | The authority section adds request validity and gives a generator's default operating point research status. |

Locality condition 2 (q4) is unchanged, and nothing elsewhere contradicts it: the natural-continuation text and the generation record both treat a hold that ends after its scope as a consequence, not as enforcement. All internal links and anchors resolve.

<a id="conflicts"></a>
## Conflicts between the human's text and the existing documents

| # | Existing document | Human's text | Resolution |
| --- | --- | --- | --- |
| 1 | `notation.md` formula: `Y ~ p(· ∣ X, H, g, W, [c_W^style], [c_W^demand])` | `p(· ∣ X, H, g, W, ρ, 𝒰, [c_W^demand])` | Formula replaced; ρ, 𝒰 and demand explained in three bullets; property query added. |
| 2 | `gameplay-state.md` Controls: an absent style request "permits the learned natural style distribution" | Without a style override, generation follows the baseline; no population-average style | An absent style directive leaves generation conditioned on ρ. A population draw happens only when the system samples a baseline. |
| 3 | `notation.md`: "(H, g) is the source of truth"; summaries "do not require a separate committed control-memory object" | ρ is retained across continuation calls | (H, g) is the source of truth *for the chart*. ρ and 𝒰 are generation inputs retained across calls, neither stored in (H, g) nor derived from it. Recomputing ρ from the growing history would let overrides rewrite it. |
| 4 | Controls indexed by window: `c_W^style` | Scopes in song time, independent of inference chunking | Requests have song-time scopes S. `c_W^demand` is read as the demand requests whose scopes meet W. |
| 5 | Generation window W = (g, e]; Lens scopes [a, b) | "Musical-time scopes" (song time, per the human) | Request scopes are one interval [a, b), the Lens convention, and include T when b = T. A decision is in scope by its time. A request is valid only if it is added while g < a (answer-2), so W and S relate only through the time of each decision. |
| 6 | "Strength" in `gameplay-state.md` = Lens supporting/prominent | Strength of a property target (decision) | "Ordinal strength" for observations, "target strength" for requests. Each document says neither defines the other. |
| 7 | Controls: tendency, adherence, demand and temperature are separate; the ordinal scale defines no numerical control | Strength "similar to prominence" | Strength is an *adherence* level: ordered, with no numerical scale unless calibrated. The analogy with prominence is one of degree only. |
| 8 | Style observations (readouts) | Style directives with named attributes | Requested levels are present-supporting, present-prominent and absent. Unresolved and unreviewed are not request values. An unspecified concept is the request-side counterpart of unreviewed. |
| 9 | Demand request; "section or map difficulty require their own definitions" | Section difficulty as a chart property with ν | Difficulty is the readout of an evaluator that ν declares, not a demand coordinate. A met difficulty target says nothing about mapper-defined responses. Both kinds of request can apply to the same scope. |
| 10 | "Generation may seek a legal compromise or report"; "a declared compromise or infeasibility policy" | Explicit priorities, with declared trade-offs when requests are jointly unattainable | Priority sits in η. Same-quantity overlaps are invalid input (answer-2). An optional priority orders different-quantity directives that cannot all be met. Shortfalls are declared in the generation record. |
| 11 | `gameplay-state.md` owns "semantic controls" | A separate style and control document | gameplay-state owns demand requests and the gameplay semantics every control must respect. style.md owns ρ, directives and targets. |
| 12 | `notation.md` owns committed prefixes | The chart seed is defined in the style text | Definition and commit semantics in `notation.md`; the seed's role as style evidence in style.md. |
| 13 | README: "Style and gameplay-demand requests are optional controls" | A baseline is always in effect | Rewritten: the baseline is always in effect; directives, targets and demand requests are optional. |
| 14 | Worked example: "Neither a style request nor a demand request can make…" | New inputs | "Neither the baseline nor any request can make…". |
| 15 | Legality paragraph lists what affects choices but not legality | Property targets are new | Property targets added to that list. |
| 16 | The contract defines legality only | "Playable" | Playability is a quality judgment that does not change the legal set; it has no formal definition. |

<a id="wording"></a>
## Changes to the human's wording and meaning

<a id="wording-kept"></a>
### Wording changes that keep the meaning

- **Decisions applied.** "Targets or ranges" became targets with an optional strength. "Musical-time scopes" became song time, where beats and bars may state a scope that then resolves to song time. "An expired LN-share target creates no obligation to compensate" became "an expired target", because the decision makes every released property free.
- **"Foundation concepts"** is not defined in the documents. It became "named concepts of a versioned vocabulary, such as the style observation concepts of beatmap-lens".
- **"Control window"** collides with the generation window W. It became "scope".
- **ρ unbracketed in the formula** although "when omitted initially" it is established. Stated as: a baseline is always in effect, supplied or established.
- **Priority is missing from u = (S, q^style, q^property, η)** although "η specifies transition and composition behavior". Priority is placed in η.
- **Grammar.** "Neither persistence alone proves… nor expiry requires…" became two clauses.
- **"Representation and target behavior"** was renamed "Identity and control requirements". "Internal representations" became "Representations".
- **"Readouts expose the relationship between intended and realized results"** became a list of what each generation records (no schema).

<a id="wording-readings"></a>
### Meaning changes and reading choices

Each item is written in its conservative form in style.md. The items marked Q are open questions below.

1. **Chart seed and "natural" (wording with meaning).** The text says that without a seed "the chart is natural". In the same text, *natural* means that no directive is in effect, so a seeded chart with no directives is also natural. The aside was rewritten as: the seed conditions generation as committed history and as evidence for ρ, and is not a request. *Natural* keeps one meaning.
2. **Same-quantity overlap (answered, q5).** The draft's joint reading is gone. Two targets for the same property on overlapping scopes are invalid input and the request set is rejected; there is no composition and no priority between them. The revision extends the rule in two places, both open in Q9. It applies to the same property under any ν. It applies to two style directives that specify the same attribute: this is the main thread's reading, not confirmed by the human. As a consequence, a directive that specifies every attribute (a whole-reference directive) cannot overlap any other style directive. Adjacent half-open scopes do not overlap.
3. **Priority (revised).** One job is left for it: ordering overlapping requests whose directives on *different* quantities cannot all be met. It stays in η and is optional (Q8 open). Directives within one request have no order among them; a caller who wants an order puts them in separate requests on the same scope. This is a new choice. "Declared trade-offs" still reads as priorities fixed in advance plus a shortfall reported afterwards.
4. **Request validity (answered, q6).** A request is valid only if the committed boundary g at which it is added precedes its start a, and it is rejected otherwise. The effective scope S ∩ (g_u, T] is gone, and targets concern S. The revision adds a new choice: a transition interval that η declares before a must also start after g. At g = 0^- a scope may start at 0; after a seed, scopes start after g_0.
5. **Steering within an active scope is not compensation.** A section target concerns its whole scope across windows, so later rows may make up an earlier shortfall while the request is active. The no-compensation rule applies after expiry or release.
6. **Hold closing after the scope (answered, q4).** Unchanged. The close of a hold started inside S and decided after b does not see the request, even when ν counts the hold in S, and the generation record lists such objects.
7. **Locality written as two conditions on the law.** Before a, the law is the same as without the request (no announce). After b, the conditional law given the rows before b is the same (influence only through history). This is the formal version of "sees nothing", "no hold" and "no compensation".
8. **Relative edits (revised)** are relative to what generation would produce in S without that directive. The draft's second reference point, a lower-priority style directive on the same attribute, cannot occur any more.
9. **What a style directive overrides.** It overrides only the attributes it declares. A directive whose characteristics cannot be separated, such as a reference that describes the whole organization, specifies every attribute, so it cannot overlap another style directive.
10. **Seed plus supplied ρ (Q7).** The supplied baseline governs and the seed is history. The text's "instead" supports this, but the case is not stated.
11. **"Without either, the system establishes a baseline"** is read as "without a supplied baseline". The system then uses the seed's evidence when a seed is given and samples otherwise. Sampling also chooses among the identities a seed supports.
12. **Random seed and sampled baseline.** The random seed covers any sampling that establishes ρ. The established ρ is recorded and can be supplied again, so a recorded ρ with a new random seed gives another realization of the same identity.
13. **Demand requests.** Like every request they have song-time scopes. Until their interface is defined they have no precedence relative to 𝒰, and a joint shortfall is declared.
14. **One interval per scope.** S is one interval. A target over a non-contiguous section, such as "all choruses jointly", is not expressible. Several requests give one target per interval.
15. **A request carries at least one directive.**
16. **Undefined readouts.** ν declares when a readout is undefined, which is not zero. LN share 0 asks that every head be a tap, so a scope without heads does not meet it.
17. **ν declares more** than the text lists. It also declares the deviation measure, the readout's resolution and its behavior under the mirror μ (from `gameplay-state.md`'s symmetry rules).
18. **Baseline update** applies to decisions after the committed boundary at which it is made.
19. **Strength rules.** There is no zero strength, no level below the default (Q2 open), and no strength without a target. Strength is defined for property targets only (Q3 open).
20. **Default strength (answered, q1).** The default is the generator's primary operating point, the one it is trained and validated for, balancing playability, style and control. It is neither the target met exactly nor maximal adherence. The formulation states this as: default adherence is a measured property of a generator, reported for each property, with no value fixed by the formulation.
21. **Playability at higher strength (new choice).** The text's "local control and its withdrawal preserve continuity, playability, and committed obligations" now meets higher levels that may trade playability. Written as: a higher strength may lower playability; continuity and committed obligations hold at every level; legality is never traded.
22. **Cancellation of an active request (new choice, Q10).** Kept, because the human's text names cancellation and release apart from expiry. Cancelling inside the scope at g' ends governance at g', and the targets stop being objectives. The record keeps the committed part's readout marked as cancelled, not as adherence. A change is a cancellation plus a new request that must start after g', so neither governs the time in between.
23. **At most one target per property in a request**, which follows from the overlap rule.

<a id="strength"></a>
## Strength

<a id="strength-definition"></a>
### Definition (`style.md#target-strength`, revised after answer-2)

- **What it requests.** How strongly one property target governs generation, weighed against everything else the generator balances. Strength changes neither the value, ν, the scope, η nor the priority.
- **Default.** An unspecified strength is the default. The default is the generator's primary operating point: the one it is trained and validated for, balancing overall playability, style and control over every active directive. It is neither the target met exactly nor maximal adherence, so a readout may deviate where meeting the target would cost more of that balance. Default adherence is a measured property of a generator, reported for each property; the formulation does not fix it. For the exact-targets-first implementation, its trained and validated behaviour is the default.
- **Higher levels.** They weight the target more against that balance. They ask for a smaller deviation and accept a greater cost in playability, fidelity to the baseline, musical correspondence, the organization of unspecified attributes, and variation across seeds. They never trade legality or committed obligations. The requirement is monotone: with every other input fixed, raising the strength does not increase the expected deviation. When the default already reaches ν's resolution, there is nothing left to reduce.
- **Scale.** Levels are ordered, not equally spaced, not probabilities, not weights. A numerical scale needs its own calibration for each property.
- **Unspecified, zero, absent.** An unspecified strength is the default. There is no zero strength, because removing a target's effect is a release. A strength cannot be stated without a target. A zero target and a style "absent" are ordinary targets and can carry any strength the interface allows.
- **Relations.** Priority orders overlapping different-quantity directives that cannot all be met. Strength trades one target against the default balance, never against another directive, so a strong low-priority target still yields. η says where and when a request applies. Temperature changes the randomness of every decision. Lens prominence describes a realized section, and the analogy is one of ordered degree only. Strength does not make a section target hold in every smaller window.
- **Readout.** No single chart reveals strength. Measure against the default: pairs that differ only in one target's strength, under the same random seeds. The effect is the reduction of the deviation distribution beyond seed-to-seed variation. The cost is the change in playability judgments, style readouts against the baseline, musical correspondence, other targets' deviations, and within-identity variation. A gain obtained by collapsing variation shows in the last of these. When the default deviation is already at ν's resolution, no effect is expected.
- **Consequence.** The default deliberately balances, so even an ideal generator can leave a deviation there, and higher strength has a meaning that does not depend on a generator falling short. The draft's definition lacked this. Comparing strength across generators needs each one's reported default adherence.

<a id="strength-alternatives"></a>
### Alternatives set aside

| Alternative | Why not |
| --- | --- |
| Tolerance band: strength sets the allowed ∣readout − target∣ | A range under another name, which the decision rejected; it also turns a preference into an acceptance cliff |
| Guidance or logit scale on the condition | A mechanism, not a meaning: the same number gives different adherence in different models. It may implement strength. |
| Training loss weight | Implementation, and not a per-request control at inference |
| Probability of meeting the target | Needs a tolerance (as above) and a calibrated probability model |
| Strength as tie-breaker or priority between targets | Duplicates priority and would let a strong low-priority target override a high-priority one |
| Strength as how prominent the property is in the section's character | Confuses the value with the strength (LN 0.2 cannot be "more prominently 0.2") and blurs the style/property split |
| Strength as the share of smaller windows that meet the target | Contradicts section-level targets |
| Default as the plain conditional: the target met as closely as natural generation allows (the draft's proposal) | The human chose the trained operating point instead (answer-2). Under the conditional, an ideal generator gained nothing from higher strength. |
| A soft default with a fixed finite weight | Needs an anchor that the data does not supply. The chosen default is a soft default whose anchor is the generator's trained and validated operating point, not a fixed weight. |

<a id="open-questions"></a>
## Open questions for the human

<a id="q1"></a>
1. **What does default strength promise?** *Answered* ([private, local](private/human-inputs/aa6818ba-9ce4-4d72-b1de-e7d78a7ffbee.md#answer-2)): neither option. The default is the operating point the generator is trained and tuned for, balancing playability, style and control. `style.md#target-strength` is rewritten around it; see [Strength](#strength-definition).
<a id="q2"></a>
2. **Levels below the default?** Example: "LN about 0.2, loosely: don't bend the chart for it". (a) *Proposed:* not defined. Omit the target or accept the default. (b) Add weaker levels that sit between "free" and the default.
<a id="q3"></a>
3. **Strength for style directives?** Example: "Jack organization prominent in bars 33-48, strongly". (a) *Proposed:* strength is for property targets only, and style adherence stays an undefined interface. (b) The same adherence levels also apply to style directives.
<a id="q4"></a>
4. **A hold that starts inside a scope and ends after it: does its end still follow the request?** *Answered:* (a), no. After the scope the end is decided naturally, ν may still count the hold, and the record lists it. Locality condition 2 is unchanged.
<a id="q5"></a>
5. **Two targets for the same property on overlapping scopes.** *Answered:* neither option. Such a request set is never valid and is rejected. "Overlap and priority" is rewritten; the extension to style attributes and to different ν is Q9.
<a id="q6"></a>
6. **A request added after its scope has started.** *Answered:* neither option. A request is valid only while the committed frontier is before its declared start. The effective scope is removed. Cancelling or changing an active request is Q10.
<a id="q7"></a>
7. **A chart seed and a supplied baseline that disagree.** Example: the seed is a jack-heavy intro, and the supplied baseline is a stream style. (a) *Written:* the supplied baseline governs, and the seed is only history. (b) The baseline is a blend weighted by the seed's evidence.
<a id="q8"></a>
8. **Different properties overlapping without priority.** Example: LN 0.8 and difficulty 2.0 stars on the same section, with no priority given. (a) *Written:* allowed. If both cannot be met, the shortfall is reported with no rule for which yields. (b) Require a priority whenever any two requests overlap. This is now priority's only job, so (a) leaves it fully optional, and dropping priority altogether would be a third option.
<a id="q9"></a>
9. **What counts as "the same quantity" for the overlap ban?** Examples: (i) "Jack prominent 0-60 s" with "Jack absent 30-90 s"; (ii) LN share counted by heads over 0-60 s with LN share counted by rows over 30-90 s, which are two ν for one property; (iii) a whole-reference style directive over 0-120 s with "Stream prominent 60-90 s". (a) *Written:* all three are invalid and rejected. (b) Style directives on the same attribute may overlap, and only property targets are banned. (c) Only the same property under the same ν conflicts.
<a id="q10"></a>
10. **May an active request be cancelled or changed before its scope ends?** Example: "LN share 0.5 for 60-120 s", and at 90 s you want to stop it or switch it to 0.3. (a) *Written:* cancel at 90 s; the target stops being a goal and the 60-90 s readout is recorded as cancelled. A change is a new request that starts after 90 s. (b) No: cancelling or changing is valid only before the scope starts, like adding, so an active request runs to its end. (c) Cancelling is allowed, changing is not.

<a id="plan-points"></a>
## Points the plan revision must reconcile

From [r2-condition-plan-v3](r2-condition-plan-v3.md), read only for context, refreshed after answer-2:

- **§3.2 resolver.** It clips the lower-priority interval on a same-kind overlap. Same-kind overlaps are now invalid input: reject them, with no clipping and no priority between them. Q9 decides whether style directives on the same attribute and targets under different ν are rejected too.
- **Priority.** Its only remaining use is ordering different-kind directives that cannot all be met, and it is optional. Plan v3's "across kinds nothing is resolved; no trade-off is claimed or trained" is compatible as long as the receipt records any given priority and declares shortfalls. Q8 is still open.
- **N5 and T-O (§2.1, §4.4), answered q4.** A release factor of an LN born in S and decided after S's end must not read S's value: locality condition 2 is the default with no exception. The plan's ownership by birth for release factors goes. ν may still count the hold in S by its head, and the receipt lists it.
- **§3.2 mid-generation edits, `replace_interval`, §3.3 "next decision after the frontier", answered q6.** Adding a request whose start is at or before the committed frontier is rejected, not clipped. Cancelling inside a scope remains, pending Q10: the pre-frontier part keeps its value, the targets stop, and the receipt marks it cancelled. A change is a new request that starts after the frontier.
- **Default strength, answered q1.** The operating point the model is trained and validated for is the default by definition. Its adherence per property must be measured and reported (receipts, §7). The checkpoint-selection criteria (§7.6) are part of what "default" means, so the plan should state them as such. Higher levels stay reserved, with the default as the only level.
- **§3.7 (lo, hi) encoding.** Ranges are gone.
- **§3.2 "the whole song is [0, T)".** The formulation's whole-song scope also contains T.
- **Undefined readouts.** They are not zero; the plan already treats "unreadable" this way.
- **Presence `none`, and the token conditioner as an announce form (N1).** Both agree with locality condition 1.
- **Difficulty with "untrimmed tails".** The readout counts closes decided after the scope, which generation does not steer (q4).

<a id="not-checked"></a>
## Not checked

- Rendering of `\mathscr`, `\mathsf` and the display math on GitHub. No preview was run.
- The repository README (around line 95, "optional style and gameplay-demand controls") and the `docs/research/` files were not edited because they are outside the allowed scope. Their wording is still compatible.
- Beatmap-lens's own concept definitions. The requested levels rest on `gameplay-state.md`'s assessment table only.
- Plan v3 beyond §2-§4 and the R2 code. The points above are pointers, not a consistency check.
- The strength definition has not been tested against any model or generation.
- No independent review of the documents, of the draft or of the revision after answer-2.
- Whether any caller workflow needs a transition interval to start at or before g. The revision assumes it does not.
