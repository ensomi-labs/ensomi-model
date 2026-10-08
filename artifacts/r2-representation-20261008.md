# Representations of chart sections first, then measures (opened 2026-10-08)

<a id="d-represent-first"></a>**Direction (human, 2026-10-08 about 03:00 UTC, [private, local](private/human-inputs/28c38740-5d34-43bb-ad36-ff8ec6d0299f.md#prompt-10)).**
- **The recurring failure is measurement.** We could not recognise the collapse the human judged blind (X0: [o-x0-scored](r2-bakeoff-night-20261007.md#o-x0-scored)), nor the five Lens style concepts across about 4,700 labelled sections (probe macro F1 0.577-0.587, the generator's state adding nothing: [s-style-module](r2-phasen-and-lens-20261006.md#s-style-module)).
- **Sequence-based modelling may fail hard in some aspects.**
- **First find better representations of chart sections, then derive measures from them.**
- **Agreed:** a whole-chart identity vector is probably needed, and per-step calibrated training causes problems ([s-diagnose-answer](r2-diagnose-synthesis-20261008.md#s-diagnose-answer)).
- **An agent blind spot:** R2's head rows are supplied. Behind that sit assumptions: raw audio decides the rows; the rows give enough for choreography and control; the separation of concerns lets R2 achieve some things without others. The agents never examined them.
- **The human asked for** a record, an analysis of the failures agents often make, and counterfactual thinking about the representation space and measures.
- **On hold:** the decode-controller plan [p-diagnose-plan](r2-diagnose-synthesis-20261008.md#p-diagnose-plan), not started.

<a id="a-represent-agree"></a>**Main-thread reading (inferred, for the human to check).**
- **Agreed, on the record so far.**
  - Every failed round optimised or judged with a quantity never shown to see what the human sees: the lineage's evaluators, R2's guards, round 1's G.
  - Those quantities were built from row symbols or window averages, which discard the relations that make up organisation: hand movement, pattern types, where choices fall relative to the rows.
- **Two conditions keep "representation first" from becoming open-ended.**
  1. A representation counts only if a simple probe on it recognises the labels we already hold: the Lens concepts and the X0 calls.
  2. It should describe the arrangement R2 chooses *given* the supplied rows, not the rows themselves.
- **What a validated representation becomes:** the measure, the decode target or critic, and the space for a chart-level identity.

<a id="r-represent-round"></a>**Delegated 2026-10-08 about 03:06 UTC, three fresh subagents in parallel, due within 75 minutes.** Context and briefs: `~/ensomi/.sync/cp/scratch/r2-represent/` (`context.md`, `brief-*.md`). Outputs on bings-mac: `artifacts/r2-represent-20261008/<worker>/`.
- **opus-history:** recurring agent failure patterns across the whole record; why the 2026-10-02 list did not stop them; at most five structural changes. Read-only.
- **fable-represent:** counterfactual analysis of the representation space and measures. The pilot probes window statistics against a structural (hand trajectory / pattern grammar) representation and a quickly learned section embedding, on the Lens sections and X0.
- **opus-skeleton:** what the supplied rows carry and assume; what the separation of concerns lets R2 achieve without what; a pilot on how much the rows determine; deployment shift; counterfactual.

<a id="r-represent-stopped"></a>**Stopped for the handoff, 2026-10-08 03:11 UTC.** The human closed the session about 5 minutes after the delegation. The three workers had written no file and started no mac job. They were stopped, and the round is to be relaunched unchanged from the same briefs ([handoff-20261008-represent](handoff-20261008-represent.md)).

<a id="r-represent-relaunch"></a>**Relaunched unchanged, 2026-10-08 03:14 UTC (main session 00e97a87), due about 04:30 UTC.** Same briefs, three fresh subagents in parallel (opus-history on Opus, fable-represent on Fable, opus-skeleton on Opus). Two facts were added to their prompts:
- **The mac venv has no scikit-learn, statsmodels or polars.** It has numpy 1.26.4, scipy 1.15.3, torch 2.11.0, pandas 2.3.3 and pyarrow 22.0.0 (job `20261008-031255-venv-check`). The probes are to be written with numpy, scipy or torch; no installs.
- **A correction to [r-represent-stopped](#r-represent-stopped):** the stopped launch had run two read-only checks on the mac, `20261008-030906-fr-envcheck` and `20261008-030938-osk-probe`. Both found sklearn missing and neither wrote outputs.
- **Updated 03:16 UTC:** the human allowed adding packages if necessary ([private, local](private/human-inputs/00e97a87-25ca-4fab-a49a-668fbf454be4.md#prompt-2)). scikit-learn 1.7.2 is now in the mac `.venv` (job `20261008-031535-venv-add-sklearn`; numpy and scipy unchanged; not in `pyproject.toml`, so `uv sync` would drop it). The two pilot workers were told.

<a id="o-represent-history"></a>**opus-history returned, 2026-10-08 about 03:25 UTC:** [r2-represent-history-20261008](r2-represent-history-20261008.md). Nine recurring patterns; the most damaging is deciding with measures never shown to see what the human sees ([s-history-worst](r2-represent-history-20261008.md#s-history-worst)). The 2026-10-02 list stopped reaching briefs after its link left `RESEARCH.md` on 10-06 (checked by the main thread: [v-history-checks](r2-represent-history-20261008.md#v-history-checks)). Five structural changes, proposed: [p-history-changes](r2-represent-history-20261008.md#p-history-changes). One caution for the synthesis: the labels used to choose a representation must be kept apart from those used to validate it.

<a id="o-represent-pilots"></a>**opus-skeleton and fable-represent returned, 2026-10-08 about 03:40-03:45 UTC.**
- Reports: [r2-represent-skeleton-20261008](r2-represent-skeleton-20261008.md) and [r2-represent-fable-20261008](r2-represent-fable-20261008.md).
- Both used well under their mac budgets; all jobs exited 0.
- The main thread checked the headline numbers against the mirrored outputs ([v-skeleton-checks](r2-represent-skeleton-20261008.md#v-skeleton-checks), [v-fable-checks](r2-represent-fable-20261008.md#v-fable-checks)).

## Synthesis (main thread, 2026-10-08)

<a id="s-represent-answer"></a>**The answer to the round's question.** Strength is marked per line.

1. **Unit of measurement: the arrangement given the rows, compared with a reference on the same rows.** Not window levels against the band. Two workers reached this on different data.
   - **Rows (opus-skeleton) [M]:**
     - The rows fix density and the between-part notes-per-second profile completely, and most of the difficulty: band 70 %, star R² 0.83.
     - They fix at most 41 % of chart identity and none of hand balance.
     - A window measure dominated by density cannot separate generated from real, because both sit on the same rows.
   - **X0 (fable-represent) [M, post hoc, mostly one chart]:** the human's marks localise to LN relations in excess of the source on the same rows. Rows with two or more lanes held give window AUC 0.99 against no-item windows and 0.83 within the yes-items. Without the LN columns, AUC is about 0.5.
2. **LN placement is where R2 is furthest from human mappers.** Two independent lines point there.
   - **Row level [M]:** two humans on near-identical rows agree on where LN heads fall at r 0.64. R2 agrees with its source at 0.08, which is what a rows-only sampler gives. Chords: 0.77 against 0.30.
   - **The human's marks:** they fall on LN relations (point 1).
   - **Inferred:** "weird LN distribution" is a placement failure, not only a level failure. The information that places LN heads is not in the local rows R2 reads.
3. **Relational representations recognise more than level representations [M; the labels are about 96 % machine labels].**
   - Lens macro F1 on 2,888 sections, chart-grouped cross-validation:

     | Representation | Macro F1 |
     |---|---:|
     | Window statistics | 0.677 |
     | Hand-built structure | 0.719 |
     | 7-minute contrastive embedding | 0.687 |
     | All three | 0.759 |

   - Trill gains 0.22 from structure; tech gains 0.10 from the embedding.
   - The earlier 0.587 used a different pool and is not comparable.
4. **X0 mixes two targets [M, 12 items].**
   - What the human calls collapse is LN-driven. It includes one real LN chart (x0-08).
   - What separates generated from real is a fingerprint: fast jacks, full jacks, hand imbalance and embedding dispersion, item AUC 0.81-0.94. These sit near chance against the window marks.
   - x0-09, judged collapsed with almost no LN, is not covered by the LN story.
   - A measure has to say which of the two targets it serves.
5. **The head-row premise, answered [M, from opus-skeleton].**
   - **R2 achieves, without audio or a timing model:**
     - the source's rhythm, density and section dynamics;
     - the rows-determined part of identity;
     - an approximate difficulty level.
   - **It cannot get:**
     - chart identity beyond the rows (59-80 % of between-chart variance; all of hand balance);
     - the row-level placement that mappers share.
   - **On timing-model rows** (4 songs, direction only), the difficulty leaked through human rows is lost and charts compress toward the middle.
6. <a id="c-band-reference"></a>**Corrections to earlier readings [I from M].**
   - **The F1 "band-biased start draw"** ([s-collapse-components](r2-collapse-synthesis-20261007.md#s-collapse-components)) was measured against the band mean, which mixes rows with arrangement. Against the rows-conditional expectation, R2's chart θ follows E[θ | rows] (r 0.48-0.64) and adds noise. A better reading: the start follows what the rows imply, and a real prefix adds the source's identity beyond the rows. X1's prefix result is consistent with both readings.
   - **Two results exceed the same-rows human ceiling, so they copy the source rather than hit a deployable target:**
     - the decode round's "identity r 0.94 given a target" ([s-diagnose-answer](r2-diagnose-synthesis-20261008.md#s-diagnose-answer));
     - B3-oracle (held r 0.81).

     The ceiling is the agreement between two humans on near-identical rows: chart-level ICC 0.58-0.68.
   - **m5, m6, the d0-env envelopes and the band-calibrated guards** are partly properties of the rows.

<a id="s-represent-gaps"></a>**What this does not yet establish.**
- **No null for the paired measures.** fable-represent scaled the paired differences by band SDs. A second human on the same rows also differs from the source. The ~2,240 same-song pairs on near-identical rows from opus-skeleton give that null, and nobody has applied it yet.
- **No measure is validated in the sense of guardrail 1.** The X0 AUCs were found after looking and rest mostly on x0-02. The Lens labels are mostly the Astra labeller's.
- **The source of the human-pair excess is unknown.** It could be audio, song structure, or set conventions of one mapper; the `.osu` Creator field would separate the last.

<a id="p-represent-next"></a>**Proposed next, for the human. Nothing started; no training.**
1. **Measurement on the mac, no human time.**
   - Build representation R (LN relations; timing-conditioned grammar with hands; learned residue) as one script from fable-represent's pilot.
   - Apply the same-rows human-pair null to the paired measures, and re-score X0 on that scale.
   - Split the human-pair placement agreement into same-mapper and different-mapper pairs.
   - Run a label-free test: can R tell R2 on the rows from a second human on the same rows?
2. **A short human screen, preregistered:** 10 blind pairs of 20-second clips, about 15 minutes ([p-fable-validation](r2-represent-fable-20261008.md#p-fable-validation), item 2). Include non-LN pairs ranked by the fingerprint block, so that x0-09's kind of complaint is tested too.
3. **Later, a human decision:** what the model needs.
   - An explicit chart identity latent: a rows part computed from the whole song, plus a free part.
   - Whether LN and chord placement need information outside the rows, which would reopen audio in R2's scope ([d-r2v2-scope](r2-v2-stage0-state.md#d-r2v2-scope)).
   - Step 1's mapper split bears on the second question.
