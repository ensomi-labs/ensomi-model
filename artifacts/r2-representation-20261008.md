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
