# Style formulation in `docs/formulation/`: rethink and conflict resolution

Shareable. Written 2026-10-06 by an Opus subagent (Claude, control plane) for the main thread. It records how the human's style and controllable-generation formulation ([private, local](private/human-inputs/aa6818ba-9ce4-4d72-b1de-e7d78a7ffbee.md#prompt-1)) went into the formulation documents, with the human's later decisions applied ([private, local](private/human-inputs/aa6818ba-9ce4-4d72-b1de-e7d78a7ffbee.md#answer-1), [private, local](private/human-inputs/aa6818ba-9ce4-4d72-b1de-e7d78a7ffbee.md#prompt-2)): a property directive is a target only, with an optional strength; a released or unspecified property is free; generation sees nothing of a request outside its scope unless η says so; scopes are song time; ρ stays in the formulation. The edits are in the worktree `~/wt/ensomi-model-formulation`, branch `docs/style-formulation` from `main` `178ea3c`, uncommitted. Nobody has reviewed them yet. The conflict check this builds on is [r2-style-formulation-check](r2-style-formulation-check.md).

<a id="files"></a>
## Files changed

| File | Change |
| --- | --- |
| `docs/formulation/style.md` (new) | The human's formulation, rethought: style versus chart properties, ρ, 𝒰, ν, scopes, activation and locality, overlap and priority, unspecified/zero/absent, target strength, seeds, return to natural, identity requirements, the generation record, evaluation. |
| `docs/formulation/notation.md` | The generation formula takes ρ and 𝒰 in place of `c^style`, plus the property query. The chart seed is defined as an initial committed pair. (H, g) is the source of truth for the chart only, and ρ and 𝒰 are inputs that are not derived from it. Property targets are added to the list of things that leave legality alone. The worked example now says "Neither the baseline nor any request". |
| `docs/formulation/gameplay-state.md` | Ownership split with style.md. "Controls" rewritten: baseline instead of "learned natural style distribution", demand-request scope, strength among the separate meanings, priority composition. Lens "strength" renamed "ordinal strength". Pointers from style observations, presence and section difficulty to style.md. |
| `docs/formulation/README.md` | The target includes distinctive organization, diverse realizations and composable controls. Table row for style.md. A short paragraph on open-ended style. The authority section names request semantics as interface requirements and the calibration of strength as research status. |

Anchors kept: `notation.md#generation-and-optional-controls`, `#provisional-branches-and-prefix-commit`; `gameplay-state.md#controls`, `#style-observations`, `#community-observations`, `#presence-and-ordinal-strength`. A local script resolved every relative link and anchor in the worktree's Markdown. The two failures it found are pre-existing and outside the formulation (`docs/guides/technical_analysis_writing.md`).

<a id="conflicts"></a>
## Conflicts between the human's text and the existing documents

| # | Existing document | Human's text | Resolution |
| --- | --- | --- | --- |
| 1 | `notation.md` formula: `Y ~ p(· ∣ X, H, g, W, [c_W^style], [c_W^demand])` | `p(· ∣ X, H, g, W, ρ, 𝒰, [c_W^demand])` | Formula replaced; ρ, 𝒰 and demand explained in three bullets; property query added. |
| 2 | `gameplay-state.md` Controls: an absent style request "permits the learned natural style distribution" | Without a style override, generation follows the baseline; no population-average style | An absent style directive leaves generation conditioned on ρ. A population draw happens only when the system samples a baseline. |
| 3 | `notation.md`: "(H, g) is the source of truth"; summaries "do not require a separate committed control-memory object" | ρ is retained across continuation calls | (H, g) is the source of truth *for the chart*. ρ and 𝒰 are generation inputs retained across calls, neither stored in (H, g) nor derived from it. Recomputing ρ from the growing history would let overrides rewrite it. |
| 4 | Controls indexed by window: `c_W^style` | Scopes in song time, independent of inference chunking | Requests have song-time scopes S. `c_W^demand` is read as the demand requests whose scopes meet W. |
| 5 | Generation window W = (g, e]; Lens scopes [a, b) | "Musical-time scopes" (song time, per the human) | Request scopes are one interval [a, b), the Lens convention, and include T when b = T. A decision is in scope by its time. Effective scope S ∩ (g_u, T] for a request that enters at g_u. |
| 6 | "Strength" in `gameplay-state.md` = Lens supporting/prominent | Strength of a property target (decision) | "Ordinal strength" for observations, "target strength" for requests. Each document says neither defines the other. |
| 7 | Controls: tendency, adherence, demand and temperature are separate; the ordinal scale defines no numerical control | Strength "similar to prominence" | Strength is an *adherence* level: ordered, with no numerical scale unless calibrated. The analogy with prominence is one of degree only. |
| 8 | Style observations (readouts) | Style directives with named attributes | Requested levels are present-supporting, present-prominent and absent. Unresolved and unreviewed are not request values. An unspecified concept is the request-side counterpart of unreviewed. |
| 9 | Demand request; "section or map difficulty require their own definitions" | Section difficulty as a chart property with ν | Difficulty is the readout of an evaluator that ν declares, not a demand coordinate. A met difficulty target says nothing about mapper-defined responses. Both kinds of request can apply to the same scope. |
| 10 | "Generation may seek a legal compromise or report"; "a declared compromise or infeasibility policy" | Explicit priorities, with declared trade-offs when requests are jointly unattainable | Priority sits in η. Same-quantity overlaps need explicit, distinct priorities. Shortfalls are declared in the generation record. |
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
2. **Same-quantity overlap (Q5).** "Compose… with declared trade-offs when jointly unattainable" is read literally: both directives remain objectives, each over its own section, and priority decides only when they cannot both be met. Overlapping section targets can often be met together (LN 0.3 over [0, 60) and 0.5 over [30, 90)). The other reading, where the higher priority replaces the lower one inside the overlap, is not built in. A caller obtains it by stating the lower request's scope without the overlap.
3. **Explicit priority required only for the same quantity.** Same-quantity overlaps without distinct priorities are rejected. Overlaps of different quantities, such as LN and difficulty, may omit priority. When those cannot all be met, no precedence applies and the shortfall is declared. "Declared trade-offs" is read as priorities fixed in advance plus a shortfall reported afterwards, not a trade-off rule declared per request.
4. **Request added mid-scope (Q6).** "Activation applies an override to future decisions" is read as: a request entering at g_u governs and is measured over S ∩ (g_u, T], not over the committed part of S.
5. **Steering within an active scope is not compensation.** A section target concerns its whole effective scope across windows, so later rows may make up an earlier shortfall while the request is active. The no-compensation rule applies after expiry or release.
6. **Hold closing after the scope (Q4).** The decision "outside a request's scope, generation sees nothing of it unless η says so" is applied to every decision at or after b. That includes the close of a hold started inside S, even when ν assigns the hold to S (heads in S, untrimmed tails). The generation record lists such objects.
7. **Locality written as two conditions on the law.** Before a, the law is the same as without the request (no announce). After b, the conditional law given the rows before b is the same (influence only through history). This is the formal version of "sees nothing", "no hold" and "no compensation".
8. **Relative edits** are relative to the style that would otherwise govern S: the baseline, or a lower-priority overlapping style directive.
9. **What a style directive overrides.** It overrides only the attributes it declares. A directive whose characteristics cannot be separated, such as a reference that describes the whole organization, specifies every attribute, so any overlapping style directive is a same-quantity overlap.
10. **Seed plus supplied ρ (Q7).** The supplied baseline governs and the seed is history. The text's "instead" supports this, but the case is not stated.
11. **"Without either, the system establishes a baseline"** is read as "without a supplied baseline". The system then uses the seed's evidence when a seed is given and samples otherwise. Sampling also chooses among the identities a seed supports.
12. **Random seed and sampled baseline.** The random seed covers any sampling that establishes ρ. The established ρ is recorded and can be supplied again, so a recorded ρ with a new random seed gives another realization of the same identity.
13. **Demand requests.** Like every request they have song-time scopes. Until their interface is defined they have no precedence relative to 𝒰, and a joint shortfall is declared.
14. **One interval per scope.** S is one interval. A target over a non-contiguous section, such as "all choruses jointly", is not expressible. Several requests give one target per interval.
15. **A request carries at least one directive.**
16. **Undefined readouts.** ν declares when a readout is undefined, which is not zero. LN share 0 asks that every head be a tap, so a scope without heads does not meet it.
17. **ν declares more** than the text lists. It also declares the deviation measure, the readout's resolution and its behavior under the mirror μ (from `gameplay-state.md`'s symmetry rules).
18. **Baseline update** applies to decisions after the committed boundary at which it is made.
19. **Strength rules (Q1-Q3).** There is no zero strength, no level below the default, and no strength without a target. Strength is defined for property targets only.

<a id="strength"></a>
## Strength

<a id="strength-definition"></a>
### Proposed definition (`style.md#target-strength`)

- **What it requests.** One property target's adherence: how far generation may depart from the reference law q_τ (the same inputs without that target) to bring the readout under ν closer to the target value. Strength changes neither the value, ν, the scope, η nor the priority.
- **Default.** An unspecified strength is the default. At the default, the target is a plain statement about the result: produce what q_τ produces among results whose readout matches the target, at ν's resolution. Whatever deviation a generator leaves at the default is its shortfall, and it is reported. The default is also the only level the exact-targets-first implementation offers.
- **Higher levels.** They ask for a smaller deviation and permit more departure from q_τ: fidelity to the baseline, musical correspondence, the organization of unspecified attributes, and variation across seeds. The requirement is monotone: with every other input fixed, raising the strength does not increase the expected deviation. When the default already reaches ν's resolution, there is nothing left to reduce.
- **Scale.** Levels are ordered, not equally spaced, not probabilities, not weights. A numerical scale needs its own calibration for each property.
- **Unspecified, zero, absent.** An unspecified strength is the default. There is no zero strength, because removing a target's effect is a release. A strength cannot be stated without a target. A zero target and a style "absent" are ordinary targets and can carry any strength the interface allows.
- **Relations.** Priority orders conflicting overlapping requests; strength trades one target against q_τ, never against another directive, so a strong low-priority target still yields. η says where and when a request applies. Temperature changes the randomness of every decision. Lens prominence describes a realized section, and the analogy is one of ordered degree only. Strength does not make a section target hold in every smaller window.
- **Readout.** No single chart reveals strength. Generate pairs that differ only in one target's strength, under the same random seeds. The effect is the reduction of the deviation distribution beyond seed-to-seed variation. The cost is the change in style readouts against the baseline, musical correspondence, other targets' deviations, and within-identity variation. A gain obtained by collapsing variation shows in the last of these. When the default deviation is already at ν's resolution, no effect is expected.
- **Consequence.** An ideal generator that meets every default target at ν's resolution gains nothing from higher strength. Strength matters where a generator falls short, typically for targets that are atypical for the music, history and baseline. Q1 asks whether the human wants a softer default instead.

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
| A soft default below exact adherence, with a fixed finite weight | Needs an anchor for "default" that the data does not supply. Kept as option (b) of Q1. |

<a id="open-questions"></a>
## Open questions for the human

<a id="q1"></a>
1. **What does default strength promise?** Example: a jumpstream song, "LN share 0.2 for the chorus". (a) *Proposed:* default means "as close to 0.2 as the generator can manage while staying natural", and a higher strength only pushes harder where it falls short. (b) Default deliberately leaves room to drift, say to 0.15, when the music suggests fewer holds, and only a higher strength insists on 0.2. Under (b), "default" needs a fixed meaning for how much drift is allowed.
<a id="q2"></a>
2. **Levels below the default?** Example: "LN about 0.2, loosely: don't bend the chart for it". (a) *Proposed:* not defined. Omit the target or accept the default. (b) Add weaker levels that sit between "free" and the default.
<a id="q3"></a>
3. **Strength for style directives?** Example: "Jack organization prominent in bars 33-48, strongly". (a) *Proposed:* strength is for property targets only, and style adherence stays an undefined interface. (b) The same adherence levels also apply to style directives.
<a id="q4"></a>
4. **A hold that starts inside a scope and ends after it: does its end still follow the request?** Example: "LN share 0.5 for 60-90 s". A hold starts at 89.5 s and ends at 91 s, and the difficulty evaluator scores full hold lengths. (a) *Written:* no. After 90 s the end is decided naturally, the readout may still count the hold, and the record lists it. (b) Yes, whenever ν counts the hold in the scope. (c) Only when the request's η declares it.
<a id="q5"></a>
5. **Two targets for the same property on overlapping scopes.** Example: LN 0.3 for 0-60 s at priority 2, LN 0.5 for 30-90 s at priority 1. (a) *Written:* both remain goals for their own sections, which can often both be met (0-30 s low, 30-90 s at 0.5). Priority 2 wins only if they cannot both be met. (b) Priority 2 alone governs 30-60 s, and the 0.5 target then concerns 60-90 s only.
<a id="q6"></a>
6. **A request added after its scope has started.** Example: at 30 s you ask for LN 0.5 over 0-60 s, and 0-30 s is already committed at 0.1. (a) *Written:* the target concerns 30-60 s only. (b) It concerns all of 0-60 s, so 30-60 s must reach about 0.9 to make the section 0.5.
<a id="q7"></a>
7. **A chart seed and a supplied baseline that disagree.** Example: the seed is a jack-heavy intro, and the supplied baseline is a stream style. (a) *Written:* the supplied baseline governs, and the seed is only history. (b) The baseline is a blend weighted by the seed's evidence.
<a id="q8"></a>
8. **Different properties overlapping without priority.** Example: LN 0.8 and difficulty 2.0 stars on the same section, with no priority given. (a) *Written:* allowed. If both cannot be met, the shortfall is reported with no rule for which yields. (b) Require a priority whenever any two requests overlap.

<a id="plan-points"></a>
## Points the plan revision must reconcile

From [r2-condition-plan-v3](r2-condition-plan-v3.md), read only for context:

- §3.2 resolver clips the lower-priority interval on a same-kind overlap. Under the written default (Q5 a), same-quantity overlaps remain joint objectives. A plan that only clips should require callers to give non-overlapping same-kind scopes, or take Q5 (b) to the human.
- N5 and T-O (§2.1, §4.4): a release factor of an LN born in S and decided after S reads S's value. Under locality condition 2 (Q4 a) this is not admissible as the default. It is admissible as an explicit η, or if the human picks Q4 (b).
- §3.7 (lo, hi) encoding: ranges are gone. Strength is a separate, reserved, ordinal parameter of a target, with the default as its only level.
- Priority lives in η in the formulation; a separate field is an implementation choice.
- §3.2 "the whole song is [0, T)": the formulation's whole-song scope also contains T.
- Undefined readouts are not zero; the plan already treats "unreadable" this way.
- Presence `none` and the token conditioner as an announce form (N1) agree with locality condition 1.
- The difficulty readout's "untrimmed tails" makes it depend on decisions after the scope (Q4).

<a id="not-checked"></a>
## Not checked

- Rendering of `\mathscr`, `\mathsf` and the display math on GitHub. No preview was run.
- The repository README (around line 95, "optional style and gameplay-demand controls") and the `docs/research/` files were not edited because they are outside the allowed scope. Their wording is still compatible.
- Beatmap-lens's own concept definitions. The requested levels rest on `gameplay-state.md`'s assessment table only.
- Plan v3 beyond §2-§4 and the R2 code. The points above are pointers, not a consistency check.
- The strength definition has not been tested against any model or generation.
- No independent review of the documents.
