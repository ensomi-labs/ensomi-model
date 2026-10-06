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

**Explainer for the human** (private claude.ai page, built from the two reports): https://claude.ai/artifact/1C8NZMYVi1qkGoYn6nzPoE.

**Status.** Nothing fixed, committed or run beyond the read-only checks above. The next step is the human's call.
