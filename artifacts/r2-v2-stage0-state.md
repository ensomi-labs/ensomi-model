# R2 v2 stage-0 code: state on 2026-10-06

Question (human, 2026-10-06, [private, local](private/human-inputs/dc997baa-3539-4368-a0b2-c72bd4cea5e8.md#prompt-1)): is R2 v2 ready, which ML-core decisions are open, and what is the current generation architecture. The human may bring in outside ML help on the open questions.

<a id="s-v2-stage0-state"></a>**Observation (main thread, from two fresh read-only subagents; key claims re-checked by the main thread).** The implementation worker launched after [d-plan-v4-answers](r2-style-formulation-check.md#d-plan-v4-answers) wrote the stage-0 code into the working tree of `r2/train` between 09:21 and 09:48 UTC. It is uncommitted on top of `7d9640a`, and no report from that worker survives.

- Tests: `pytest tests/r2 tests/evaluation` on the mac, job `20261006-104814`, gives 19 failed, 215 passed, 4 skipped. There are three causes:
  - `generate.py:108` passes `kind` twice to `dict()`. This fails 12 tests: every generation with a request, so T-A, T-C, T-L1, T-L2 and the record test never reach an assertion.
  - `features.py:392` calls the removed `_norm_value`. This fails 6 tests on the token-conditioner path.
  - The DPO path calls the aligned draw with an 8-row window. This fails 1 test (`StopIteration`, `conditions.py:307`).
- The earlier job `20261006-094149-r2v2-import` shows exit 0, but its log holds a collection ImportError that a pipe hid.
- Stage-0 items missing: 0.10 (DPO anchor on L_base), 0.21, 0.22, the b(S) property tests, the exposure-accounting test, and the baseline rows. No test covers `evaluate.py`, `select.py`, `baseline.py` or `proxy.py`.
- No stage-0 mac measurement has run (relabel, baseline fit, `draw_sim`, pilot). `n_bar*` are null in both arm configs, and the trainer refuses to start without them.
- Both arm configs carry lr 3e-4 with no schedule. That is v1's class default; the overnight v1 run used 1e-3, and Q-F gives the schedule to Astra.
- Model as built: 2,404,724 parameters (mac job `20261006-105046-r2-paramcount`). The TCN holds 82.8%. The token conditioner (102,144) is built but unused on the FiLM default.

Full evidence: [r2-v2-readiness-20261006](r2-v2-readiness-20261006.md) (stage-0 coverage, defaults, open decisions) and [r2-architecture-20261006](r2-architecture-20261006.md) (architecture with file:line, worked example run on the mac, v1 → v2 delta). Line numbers refer to the uncommitted tree and will move.

<a id="o-v2-open-ml"></a>**Open ML questions recorded by the readiness check (agent observations O1, O2 not yet in the plan).**

- Block stage 1:
  - the learning-rate schedule (Q-F, Astra);
  - restating the stage-1 claim and thresholds now that A1 also has λ = 1 (O2);
  - one condition manifest shared by both arms for the paired onset-NLL comparison; the code builds one per arm (O1).
- Later stages: μ_star and the B2 arm, the trimmed ν, Q-I for LN, ρ design, q3/q7/q9.
- Code-level, to check:
  - proxy and F3 seeds come from the draw RNG (O3);
  - A0 drops candidates before per-song selection (O4);
  - `star_conditions: auto` can switch star off silently (O5);
  - in-run records hash the operating point without the recipe (O6);
  - the cost of whole-song proxy scopes is unmeasured (O7).

**Explainer for the human:** [r2-architecture-20261006.html](r2-architecture-20261006.html), a self-contained local HTML page built from the two reports, with flow charts and a worked example you can step through. At the human's request it is not published anywhere ([private, local](private/human-inputs/dc997baa-3539-4368-a0b2-c72bd4cea5e8.md#prompt-2)).

**Status.** Nothing fixed, committed or run beyond the read-only checks above. The next step is the human's call.

<a id="q-decision-unit-rethink"></a>**Question (human, 2026-10-06, [private, local](private/human-inputs/dc997baa-3539-4368-a0b2-c72bd4cea5e8.md#prompt-3)): does the system match the human's model, and should the decision unit be rethought?**

The human's model is:
- each head row decides the LN releases in (t_{k−1}, t_k] and the new heads at t_k together;
- EOS is fixed at the song's end and only closes holds;
- an external module handles requests and their activation and expiry;
- the model sees only the current section's controls, never past ones, while it reads past notes;
- each section has at most one target per kind.

The human says the decision space, its order and its learnability may be rethought.

**Agent reading (main thread, checked against the code):**
- The code matches the decision unit as stated. Within a decision, one 625-way choice over lane codes is made first; the gap release times follow from the candidate grid, lane by lane, averaged over both mirror orders (`sampling.py:84-105`).
- The code also matches the request split. `request_set.py` validates requests, admits additions and withdrawals only before the start, refuses changes after it, and builds the intervals. The model reads only the visible intervals, and history tokens carry notes only.
- A request may carry both kinds on one scope; scopes of different kinds may also differ. v1, by contrast, read past controls (presence bit, birth role).
- Rule L is not an extra policy. It is what "current section" means for a decision that outputs times on both sides of a boundary: decision k emits gap releases that can fall before a.
- If the decision unit is redesigned so that no decision straddles a section boundary, rule L reduces to t ∈ [a, b). That would also replace the "split boundary decision" the plan keeps in reserve for D8 > 5%.

No redesign is decided. The architecture page's §3 and §6 were corrected for the two wordings that suggested a mismatch (notes `1fbb6cc`).

<a id="d-two-phase-recipe"></a>**Decision (human, 2026-10-06, [private, local](private/human-inputs/dc997baa-3539-4368-a0b2-c72bd4cea5e8.md#prompt-4)): two-phase training.**
- **Phase N:** learn natural chart structure first, by teacher-forced CE with no conditions.
- **Phase C:** then tune the conditions specifically, with the natural model either frozen or held close by a KL term. The human named both modes and has chosen neither.

This supersedes the Q-E answer (condition terms on from the start, A0/A1 arms) of [d-plan-v4-answers](r2-style-formulation-check.md#d-plan-v4-answers). A fresh Opus worker is writing plan v5 and changing the code on `r2/train`. Standing rules carry: no invented training details, no AI review gates, and no run before the human's code check.

Agent answers given to the human with the decision:
- The 256-decision window bounds only the scored decisions; history is encoded from row 0 (`model.py:232-250`).
- Window CE cannot see scope-level totals of the model's own choices, which matters for phase C on long scopes.
- Conditioning is one shared FiLM. After training it adds MLP(0) on every decision, so phase C needs an identity gate outside visible scopes.
- A per-kind adapter would make new kinds pluggable. It is not built.

<a id="d-r2v2-scope"></a>**Decision (human, 2026-10-06, [private, local](private/human-inputs/dc997baa-3539-4368-a0b2-c72bd4cea5e8.md#prompt-5)): R2 v2 scope and conditions.**
- **Out of scope for R2 v2:** music (audio) information, head-row (timing) decisions, and memory use.
- **Memory:** the landmark memory cannot be trained effectively now. Its intended purpose is to offer earlier similar patterns that new generation can reuse when needed, for example when a chorus returns. This adds to [a-r2-reading](r2-average-and-control.md#a-r2-reading), where CE on real histories gives the memory no pressure.
- **Conditions:** R2 v2 takes the five Beatmap Lens foundation concepts as condition inputs in addition to LN share and difficulty. This settles Q-K's attribute set (plan v4, line 916); the number of levels is open.
- **Capacity:** condition-control training needs more parameters than the current conditioning path (one FiLM, 42,112 parameters).

Open:
- whether the landmark read stays on or is switched off;
- the form and size of the larger conditioning path;
- the data source for Lens concept labels on the corpus.

The worker was told to put these in plan v5. It may make conditioning capacity configurable, but must not build Lens inputs.
