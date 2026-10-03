# R2 ML design: independent review and runnable v1 spec

Copied into the notes 2026-10-03 by main session `cf834490` from `~/ensomi/.sync/cp/scratch/r2-design-review/review.md`; reviewer output, unchanged below this line. The agent's acceptance and overrides: [r2-implementation](r2-implementation.md#d-r2-v1-spec).

Reviewer: fresh Claude session (Fable), 2026-10-03, read-only on the repository. Inputs read: `artifacts/r2-ml-design-20261003/design.md` (rev 2) and `design-v1.md`, both probe outputs, `artifacts/ln-census-20261003/report.md`, the star-section study, R1 code (`research/bounded_typed_continuation/{model,temporal,long_memory,support,data,features}.py`), the oracle replay/export primitives, the evaluation package (`beats`, `redlines`, `legality`, `case`), the R1 research docs, and `data/r2-corpus.json`.

## Verdict in one paragraph

The design is correct where it matters: the factorisation, the 625-way joint action with hand-shared coupling, the two-orientation release mixture, the mirror identity, the causal graph, the splits and the DPO derivation all check out. It is over-built in three places that would eat tonight's hours without serving a stated intention: the three-level release pointer over every integer millisecond, the per-query 64-row look-ahead Transformer, and the full-prefix GRU memory on MPS. It narrows intention 5 (both condition forms compared) and makes the comparison it does keep nearly vacuous by giving the token form the same visibility as FiLM. It has one real correctness gap (sub-2 ms head gaps make the unrounded-midpoint fallback unexportable). About two thirds of the text is checks and receipts; roughly eight tests protect against the recorded failures, the rest can wait until the first checkpoint exists. The v1 spec below keeps the mathematics, reuses R1 modules for the encoder, head and streaming pointer, and should be buildable in a few hours with code that already exists in the tree.

## Verdict per part

| Part | Verdict | Reason |
| --- | --- | --- |
| Factorisation: one decision per head row = joint 4-lane action (5 codes, 625 logits, rank-16 hand coupling) × joint gap releases; headless EOS | **Keep** | Matches intentions 3 and 4 exactly. `JointHead` in R1 `model.py` is this construction and is proven mirror-equivariant. |
| Release support: anchors ∪ 1/32,1/24 grid ∪ every interior integer ms, hierarchical A/G/E pointer with 256 ms buckets | **Simplify** to a flat categorical over anchors ∪ grid (median 11, p99 49 candidates per gap), labels snapped to the nearest candidate in the same gap with the error logged | The ms fallback answers a coverage threshold Astra set itself (99/99.5 %), not one the human set. Census: on-row, midpoint, 1/8 or 1/4 before next row already cover 93.7 % of LN; off-grid E4 is 0.1 %. The hierarchy exists only to tame the fallback's cardinality. R1's `EndpointPointer.log_prob` already streams ragged candidate sets with checkpointing; reuse it. Door stays open: candidates carry a family one-hot, so an E family can be added later without changing the model class. |
| Two-orientation release mixture, placement relations | **Keep**, simplify relations to a fixed per-lane-slot vector | Exact, normalised, invariant; R1 `endpoint_log_probs` does exactly this. Sum-pooled relation MLP over V≤4 is replaced by 4 oriented lane slots × (Δms, Δbeat, same-birth, placed) with masks: same information, no extra module. |
| History: one token per head decision with release slots; TCN 8 levels / 511 rows | **Keep** | Correct causality (all releases in a token are ≤ its row time). `FiniteTemporal` exists. |
| GRU landmark memory over the full prefix, seed residual | **Simplify**: landmarks are TCN outputs at every 64th committed row (run the TCN over the full prefix once per window); drop the seed residual | A sequential GRU over 1k–9k rows per window is the one component that cannot be fast on MPS; TCN landmarks are parallel, causal and mirror-safe, and keep whole-chart reach. The seed residual duplicated the GRU's role in R1 and missed its own threshold there; in R2 the seed is simply committed history. |
| Look-ahead: 2-layer bidirectional Transformer over next 64 rows per query + density counts | **Simplify** to scalar features: next 16 input gaps (ms and beats through ψ) + the 6 density horizons, as R1's `TimingView` does with 16 | Q×64 tokens×2 layers per window is the most expensive block in the design for a timing-only signal whose decisive part is the next one or two gaps. Trigger to revisit: a reverse-direction `FiniteTemporal` over H (one pass per chart) if the look-ahead lesion shows the model is starved. |
| Conditions: interval tracks, tiled-star labels (≥30 s), FiLM and token forms behind one interface, dropout | **Fix** scope, keep mechanics | Intention 5 says build both and compare; rev 2 dropped the comparison. Also the "matched visibility" rule gives tokens the same ≤6 active frames FiLM sees, which makes the two forms nearly equivalent functions of the same 96 numbers. The meaningful token form attends over the whole announced track (past and future intervals, legitimate inputs by the design's own causal graph). Spec: FiLM trains overnight; token form is implemented and unit-tested tonight; the matched comparison runs tomorrow at equal exposure. |
| Training-time edit exercise (removal/reveal at window start) | **Defer** | The model only ever sees frames computed from the track at the query time; an interval that begins or ends mid-window is already produced by interval sampling. Keep `replace_interval` as a runtime API. |
| Star label cache: ≤9 tiled labels per chart, one CPU worker | **Keep**, 4 workers, run in parallel with implementation; launch without star if not ready | Human-fixed label. ~12.5k charts × 9 labels through the pure-Python calculator is roughly an hour single-threaded. |
| Song duration via ffprobe only | **Keep**, with a declared fallback | If ffprobe is missing or fails on the mac tonight, `api_total_length_s` (API audio length, 1 s resolution, release-independent) with a flag is better than no run. Needs the human's nod (below). |
| CE recipe: group-uniform draw, 256-row windows, seed strata, window-start strata, window-mean loss, AdamW 3e-4, 1×8 accumulation, 8M-exposure cosine | **Simplify** | Seed strata vanish with the seed residual. Cosine horizon must be set from measured throughput, not a fixed 8M, or a 12-hour stop lands mid-schedule. Effective batch 4 windows (1024 decisions) rather than 8 gives ~2× the optimizer steps at equal exposure. |
| MPS float32, 30-minute execution blocks, 4 GiB caps | **Simplify** | R1 measured CPU 2.8× faster than MPS on varying-shape pointer work and MPS footprint growth to 9.6 GiB. The pilot must time both; the faster device trains. One continuous process overnight (nobody else is scheduled), checkpoint every 250k exposures, memory guard from `oracle_time_continuation.runtime.ResourceGuard`. |
| Selection: earliest checkpoint within 2 SE of best fit_dev CE among guard-passing checkpoints | **Keep** the rule, run it in the morning | Nothing to automate tonight beyond logging. |
| DPO: cDPO soft labels, CE anchor, frozen reference, sums over decisions, KL monitor, source weighting, collector protocol, 3-round budget | **Keep** the loss and derivation; **defer** everything about real pairs | Intention 9: a DPO trainer tested on synthetic pairs. `dpo_loss` is ~30 lines on top of `sequence_log_prob`. Write it while the overnight run is going; it cannot run before a reference exists anyway. |
| Synthetic tests T1–T7 | **Simplify** to T1, T2, T5 (and T4 if time) | T1 fixes sign, scale and optimum; T2 catches per-release averaging; T5 checks the real network's gradient. T3/T6/T7 test musical fixtures and provenance rejection that have no consumer yet. |
| Sampling: temperature 1 full support, CPU float64 categoricals, named SHA-256 RNG streams, resumable generation state | **Simplify** | One `torch.Generator` per continuation seeded from an integer; log the seed. Resumable mid-chart generation is not needed for an overnight CE run. |
| Checks F/S/C/E/G/M1/M2, lesion matrix, 128-group native audit with 2000 bootstraps | **Simplify** to the eight pre-run tests below plus a per-checkpoint free-run report | The remaining checks become the evaluation of the first checkpoint, not a gate on starting. |

## Where I disagree with Astra, by consequence

1. **Full-millisecond release support and the three-level pointer.** This is the largest implementation item in the design and it buys exact representability of under 1 % of LN (E4 0.1 %, part of E5 0.5 %), measured against a threshold Astra declared itself. The human asked that candidates be described in ms and beats from three reference points and that one time be committed; nothing requires every integer to be in support. A flat softmax over anchors ∪ grid is what R1 already does over ragged candidate sets, so it is the zero-new-code path. Snapping labels to the nearest same-gap candidate quantises the training distribution by a few ms for a small minority of LN; the exact error distribution is logged per band and that log is the trigger for adding an E family later. There is a second cost Astra did not weigh: a 3-level masked hierarchy is the component most likely to hide a normalisation bug, and tonight no one will be awake to notice.

2. **The condition comparison.** Rev 1 had a matched 1M-exposure comparison; rev 2 removed it and chose FiLM. The human decided both forms are built and compared. More importantly, the "matched visibility" rule (tokens see only active intervals) removes the only structural reason tokens could win: seeing the announced future (an LN-heavy section coming in 20 s) and the past. Both forms then compute functions of the same ≤96 numbers and the comparison would most likely report "none". Spec: tokens attend over the whole announced track with relative offsets; FiLM uses active frames. The comparison is a real one and runs tomorrow at matched exposure.

3. **GRU memory and the look-ahead Transformer on MPS.** Astra kept R1's GRU because R1 had it and added a per-query Transformer because 64 rows felt right. Neither was costed against the machine. The GRU is sequential over the entire prefix (median 1015 rows, p99 5073) for every window; on MPS that is thousands of kernel launches per window, forward and backward. The Transformer is Q×64 tokens per window for a timing-only signal. TCN landmarks and scalar look-ahead features give the same reach at a fraction of the cost and are both existing patterns in the tree. R1's own measurement (CPU 2.8× faster than MPS with varying shapes) also says the device must be chosen by the pilot, not by assumption.

4. **Schedule by exposure count.** Cosine to 3e-5 at 8M exposures with a 12-hour stop means the LR is wherever the clock stops. Set the horizon after the pilot: `total = 0.9 × measured decisions/s × 3600 × hours_available`, and checkpoint on exposure count. A time-limited result is then a complete schedule, not a truncated one.

5. **Sequence DPO horizon.** 256 rows is long for the preferences the human named first (near-head releases, 40 ms holds): one preference bit over 256 joint decisions is weak credit assignment for a 3M-parameter model. The derivation is horizon-agnostic; make the horizon a parameter and start synthetic and first real pairs at 32–64 rows. Not a contradiction of intention 1, a recommendation.

6. **Over-defensiveness.** Receipts hashing every file, named RNG streams via SHA-256 of JSON, recording RNG algorithms, two-SE reporting rules in every table, 30-minute execution blocks, a provenance refusal in the pair loader before any pair source exists, a 14-row lesion matrix. None of these protects against a recorded failure. The recorded failures are release leakage, split leakage, claims outrunning what ran, free-running collapse, LN-share drift, 40 ms holds and near-head releases; the eight tests and the per-checkpoint free-run report below cover each one.

## Correctness findings

- **Sub-2 ms head gaps.** Source head rows 1 ms apart exist (gap p0 = 1 ms in the probe). The design keeps exact distinct rows and falls back to an unrounded midpoint when a gap has no interior integer. A release at t+0.5 ms cannot be written to `.osu` (integer times); rounding it lands on the same-lane head and is illegal. The all-four-held case before a 1 ms gap with a required head has no legal integer decision. Fix: merge head rows closer than 2 ms into the earlier time at ingestion (the census did this; log the count), assert every gap ≥ 2 ms and T ≥ t_K + 2 ms. Every gap then has an interior integer and every candidate set is exportable.
- **EOS gap.** Same requirement, T − t_K ≥ 2 ms; validate at cache time.
- **Joint head mirror**: verified against R1 `JointHead` (coupling `(M_L + M_Rᵀ)/2`, indices `a_L = 5a_0 + a_1`, `a_R = 5a_3 + a_2`): `J(MA; Ms) = J(A; s)` holds. Correct.
- **Release mixture**: `½Q→(U|s,A) + ½Q→(MU|Ms,MA)` is normalised and invariant since M is an involution; sampling an orientation fairly then sampling sequentially has exactly this marginal. Correct.
- **Hierarchical pointer normalisation**: correct as written (product of softmaxes over a partition). Moot if the flat pointer is adopted.
- **Condition dropout arithmetic** 0.328 for the one-interval-per-kind case: correct.
- **DPO**: KL-regularised optimum, Bradley–Terry substitution, partition-function cancellation across a shared start state, cDPO gradient `β(σ(βΔ) − q)∇Δ`, and the "fixed-N division is just β/N" remark are all right. The MC KL per decision is a valid monitor. No error found.
- **Causality**: history token for decision k−1 holds releases in (t_{k−2}, t_{k−1}], all ≤ t_{k−1} < t_k; query clocks use committed events; look-ahead is timing-only; condition counters use committed heads. Star labels are functions of source LN durations, but they enter only when the user requests that condition and are absent in natural mode; that is what intention 5 asks for, not a leak of choreography.
- **No release information in the inputs** (intention 1): grid from `musical_grid(red_lines, unique_head_times)`, T from audio, candidates target-independent. Holds, provided the cache builder never calls `Chart.musical_grid()`; the pre-run leakage test below checks it mechanically.

---

# Runnable v1 spec

Build under `src/ensomi_model/r2/`. Reuse: `research/bounded_typed_continuation/temporal.FiniteTemporal`, `model.JointHead`, `model.EndpointPointer` (streaming `log_prob`/`sample`), `research/oracle_time_continuation/replay.{ExactReplayState,commit}`, `schema.CompleteRow`, `runtime.{ResourceGuard,atomic_checkpoint}`, `evaluation.redlines.musical_grid`, `evaluation.beats.BeatGrid`, `evaluation.legality.violations`, `osu_core.difficulty.compute_mania_star_rating_20241007`. Where this spec is silent, `design.md` governs for the parts marked Keep.

## 1. Data pipeline (`cache.py`, `splits.py`, `labels.py`)

- Population: `data/r2-corpus.parquet` rows with `eval_split == 'fit'`, `api_status ∈ {ranked, loved}`, `2 ≤ star ≤ 6`, readable duration. Assert `eval_split == 'fit'` before opening any file. `fit_dev`: SHA-256(`r2-fit-dev-v1:` + group_id) first 8 bytes mod 10 == 0; rest `fit_train`. Persist the assignment.
- Per chart: parse objects (`osu_core.hitobjects`), head times = sorted distinct start times; **merge rows closer than 2 ms into the earlier time** (count logged); grid = `musical_grid(red_lines, head_times)`; T = ffprobe format duration ×1000 (fallback below); require T ≥ t_K + 2 and T > last source release (else exclude, reason logged). Source LN releases become labels: on a row → code 1/3/4 semantics; in a gap → snapped to the nearest candidate in the same gap (ties earlier); record `(original, snapped, error_ms)` per LN. Row/label round-trip test: expanding the decisions reproduces the source heads exactly and releases up to the logged snap error.
- Cache per chart: `head_ms[K] f64`, grid segments, T, decisions `actions[K,4] i8`, `gap_release_ms[K,4] f64|nan`, EOS releases `[4] f64|nan`, group_id, split role, star band. NumPy, one file per chart, index JSON with hashes.
- Labels: `ln_share(interval)` by prefix sums on head counts (online). `tiled_star(objects, [a,a+L), min 30 s, 240 s horizon, 1 ms seam rule)` exactly as design §3, precomputed for ≤9 intervals per chart (four consecutive 30 s cells, four 60 s cells, whole song), 4 worker processes, written to `labels/star.json` keyed by chart hash. If the cache is not complete at launch, star frames are absent for all charts in that run (flag `star_conditions=False` in the run config).

## 2. Decision, support, candidates (`state.py`, `support.py`, `candidates.py`)

- Codes per design §1 table. `action_support(state, k, eos) -> bool[625]`: local legality, ≥1 head at t_k, held lane needs a candidate to gap-release (always true after the 2 ms rule), EOS: held lanes code 2, free lanes 0. Vectorise like R1 `support.row_supports`.
- `advance(state, decision) -> (state', rows)`: expand into chronological `CompleteRow`s (gap releases sorted and coalesced, then the row at t_k) and `commit` each through the oracle replay; `advance(Ms, Md) == M advance(s, d)` by construction.
- `candidates(gap=(a,b), grid) -> times[C] f64, feats_static[C]`: strictly interior union of: rounded 1/32 and 1/24 grid positions per intersected segment; midpoint; b − 1/8 beat; b − 1/4 beat (global beat). Dedupe; flags for midpoint/eighth/quarter/grid; snap denominator one-hot (1,2,3,4,6,8,12,16,24,32). Version string `cand-v1`. Never read a source release here.
- Candidate views per lane ℓ at decision time: `(u−t_k, B(u)−B(t_k), u−t_{k−1}, B(u)−B(t_{k−1}), u−h_ℓ, B(u)−B(h_ℓ))` through ψ (12), fraction through gap (1), flags (4), snap one-hot (10), BPM/120 (1), phase sin/cos at 1,4,16 beats (6) = 34. Placement relations as 4 oriented lane slots × (Δms, Δbeat through ψ (4), same-birth (1), placed (1), on-row (1)) = 28, zero with mask when unplaced.

## 3. Features (`features.py`)

ψ_s(d) = [clip(d/s, −32, 32), sign(d)·log1p(|d|/s)], s = 1000 ms or 1 beat.

- History token `[L, 2, 100]`: as design §2 table (per oriented lane: action one-hot 5, prior-held 1, release-present 1, six release views 12; shared 24).
- Query `[Q, 2, 86 + 16·4]`: design's 86 (per oriented lane 15, shared 20, 6 density counts) plus next-16-gap scalars `ψ(t_{k+i} − t_{k+i−1})` in ms and beats (64), zero-masked past the end of H. No Transformer.
- Condition frames: 16 channels per (role ∈ {row, candidate, birth}) × (kind ∈ {ln, star}) as design §3; FiLM input is the flattened 96 with masks. Token form additionally gets the whole announced track: one token per interval with (kind, value, start−t and end−t through ψ in ms and beats, active bit, progress, committed counts if started) and a null token.

## 4. Model (`model.py`), ≈3M parameters

- `FiniteTemporal(100 → 128, 8 levels, k3, exp 4)` over the full committed prefix, both hands sharing weights (as R1). Landmarks: TCN outputs at committed rows j ≡ 0 mod 64, j < k; one cross-attention read (`LandmarkMemory.read` pattern with keys/values from TCN outputs, zero-init output).
- Exact-query MLP 150→128→128; fuse `LayerNorm(256) → 256→128→128`; add landmark read; conditioner; GELU everywhere; no dropout.
- `Conditioner(mode)`: `film`: MLP 96→128→256 → (γ, δ), `z + γ⊙LN(z) + δ`, zero-init last layer. `tokens`: MLP (≈20 inputs)→128→128 + kind embedding, null token, one 4-head cross-attention from `LN(z)`, zero-init output. Same weights on both hands. Both constructed; the run config picks one.
- `JointHead(128, vocabulary 5, rank 16)` from R1 with `left = 5a_0 + a_1`, `right = 5a_3 + a_2` over all 625 codes; mask → log-softmax.
- Release pointer: query = MLP(concat(z_L, z_R, 4×8 code embeddings, releasing lane's 15 query features, relation slots 28) → 128 → 128), conditioned by `Conditioner` with candidate/birth frames; candidate MLP 34→128→128; score = dot/√128 + linear bias. Use R1 `EndpointPointer.log_prob` streaming (candidate budget 8192) and `sample` (Gumbel-max on CPU RNG). Directed order 0,1,2,3 in the oriented frame; `release_log_prob = logaddexp(fwd, mirrored) − log 2`, as R1 `endpoint_log_probs`.
- `decision_log_prob(state, decision) = log P(A|s) + release_log_prob`; `sequence_log_prob` sums decisions. Print the parameter count per submodule at construction; stop above 4.5M.

## 5. CE recipe (`data.py`, `train_ce.py`)

- Draw: group uniform, chart uniform in group. Window start j: BOS with p = 0.125, `max(0, K−255)` with p = 0.125, else uniform in [0, K]. Score up to 256 decisions from j; include EOS iff the window reaches K. History = teacher-forced decisions before j (full prefix for TCN/landmarks). Loss = mean over the window's decisions; average windows in the batch.
- Conditions per window: LN intervals as design (lengths 8/16/32/64 beats, 1–4 kept, consecutive with p 0.5); star: whole song with p 0.10 else 1–3 cached intervals; dropout 0.20 all / 0.25 kind / 0.20 interval. No edit exercise.
- Optimiser: AdamW lr 3e-4, betas (0.9, 0.95), eps 1e-8, wd 0.01 on matrices, clip 1.0, float32. Effective batch 4 windows (microbatch 1, accumulate 4; raise microbatch if the pilot shows headroom). Warm-up 50k exposures, cosine to 3e-5 at `total_exposures` set from the pilot (below). Seeds 171 (weights), 471 (draws), 954 (validation).
- fit_dev manifest: 64 fixed windows (seed 954) across 64 groups; 4 fixed short charts (K ≤ 600) for free-running.

## 6. Tuning hour

1. 200-window pilot on MPS and on CPU (threads = performance cores), FiLM, identical draws: decisions/s, peak RSS, MPS footprint growth. Pick the faster device; if MPS footprint grows monotonically across varying shapes, use CPU.
2. `total_exposures = 0.9 × decisions/s × 3600 × hours_until_0800`. Checkpoint every 250k exposures.
3. If decisions/s < 150 on both devices: drop landmarks (`memory=none`) and re-measure; then halve the TCN to 6 levels (127 rows) if still short. Record each change.
4. Loss finite and decreasing over the pilot; LN-share of a 10-second free-run from the pilot checkpoint is finite and legal (sanity only).

## 7. Overnight run: logs and checkpoints

Every 250k exposures and at the end: checkpoint (model, optimiser, scheduler, RNG, draw cursor, config, cache hashes) via `atomic_checkpoint`, keep all. Log per 2k exposures: action NLL/decision, gap-release NLL/decision and per gap LN, EOS NLL, natural vs conditioned split, lr, decisions/s, RSS, MPS footprint. Per checkpoint: fit_dev CE on the manifest (same components); free-run the 4 fixed charts from BOS with seeds 954–956 (natural mode) and report: legality violations, every head present, EOS closed, LN share, hold length (share ≤40 ms, ≤60 ms), release 1–40 ms before another lane's head, on-row/midpoint/eighth/quarter/other release shares, chord-size histogram, open-hold occupancy, LN share first vs last third of the song. Stop automatically only on NaN loss, `ResourceGuard` (swap growth or RSS over 12 GiB), or the clock. Write `run.json` with device, decisions/s, exposures reached, wall time.

## 8. Generation and export (`sampling.py`, `export.py`)

`continue_chart(model, skeleton, prefix_decisions, track, seed, conditioner) -> decisions`: mask → sample action (CPU float64 categorical), fair orientation, sequential release draws with `EndpointPointer.sample`, `advance`; EOS step closes holds in (t_K, T). Export: rows → `ManiaHitObject` list → `.osu` text (reuse `osu_core.export` or the oracle `export_osu` with a presentation header) → `legality.violations` must be empty. Optional prefix = committed decisions replayed; a prefix that cannot be made legal under the skeleton is an error. Condition edits: `replace_interval(track, kind, a, b, value, frontier)` clips a to ≥ frontier, splits/coalesces; applies from the next decision.

## 9. DPO trainer, synthetic only (`train_dpo.py`, written during the run)

`dpo_loss(policy, reference, pairs, anchor_windows, beta=0.1, lambda_ce=0.2)` with `Δ = Σ_k [log p_θ − log p_0](y⁺) − same(y⁻)` over all decisions of each branch replayed on its own states, soft label q, `−q log σ(βΔ) − (1−q) log σ(−βΔ) + λ_CE·CE(anchor)`. Horizon is a parameter (default 64 for synthetic tests). Tests: T1 (two-outcome tabular adapter: one SGD step = ηβ(q−½), optimum logit(q)/β), T2 (shared nuisance factor has zero gradient; changing the number of probability-one factors changes nothing), T5 (tiny R2 model, θ = θ₀: first-order increment ηβ(q−½)‖∇Δ‖² within 1 % on CPU float64, both orientations, EOS included). T4 optional.

## 10. Tests that must pass before the run starts

1. **Normalisation**: on 8 tiny fixtures (0–4 held lanes, 2–5 candidates per gap, EOS), Σ over all legal decisions of exp(decision_log_prob) = 1 ± 1e-8 in float64, both orientations included.
2. **Mirror**: 32 random replayed states × up to 5 legal decisions, with and without conditions: `|log p(Md|Ms) − log p(d|s)| ≤ 1e-10` (CPU f64) and ≤ 3e-5 (device f32); support masks equal under the 625-permutation; `advance(Ms,Md) == M advance(s,d)`.
3. **Release leakage**: shift every source release of a chart by a random amount (keeping legality); every input tensor for decision k (history before k, query, look-ahead, condition frames in natural mode) is bit-identical; only the labels and the history tokens *after* the changed decisions differ. Also assert the cache never imports `Chart.musical_grid`.
4. **Causality**: mutate the label of decision k+1; log-probs for decisions ≤ k unchanged.
5. **Split**: cache builder raises on any non-`fit` row; fit_dev hash reproduces a stored manifest.
6. **Round-trip**: cache → decisions → `advance` → rows → `.osu` → parse reproduces the source heads exactly and releases within the logged snap error; legality clean.
7. **Train step**: 8 real windows, loss finite, gradients reach TCN, landmark read, exact MLP, conditioner (both modes), joint head, pointer; one optimiser step; save, load, second step identical to an uninterrupted run.
8. **Free-run smoke**: 200-step model generates a full short chart (zero seed and 100-row seed): legal, all heads present, EOS closes all holds; conditioner hand-swap identity holds for both modes.

## 11. Deferred, with triggers

| Item | Revisit when |
| --- | --- |
| Integer-ms release family and hierarchical pointer | Snap-error audit shows > 1 % of LN in any band moved by > 5 ms, or human review of generated charts finds grid-locked releases wrong |
| Reverse `FiniteTemporal` over H (look-ahead encoder) | Look-ahead lesion on the first checkpoint shows release placement insensitive to the next gaps, or 16 gaps measurably too short |
| GRU landmark memory, seed residual | TCN-landmark lesion shows no long-range effect and late-song drift persists |
| Token vs FiLM comparison | Tomorrow: matched run of the token form to the FiLM run's 2M checkpoint, same draws and seeds, fit_dev CE + LN/star following on 0.05/0.15/0.30 and ±0.5 star targets |
| Training-time condition edits | Runtime edit tests show the model mishandles a mid-chart change the interval sampler never produced |
| T3, T6, T7; pair collector; KL monitor; provenance refusal | A reference checkpoint is selected and real pairs are authorised |
| M1/M2 at 128 states/mode, lesion matrix, 128-group native audit | First checkpoint is selected; these become its evaluation |
| Mixed precision, larger model | Throughput measured and CE still improving at the end of the first run |

## 12. For the human (proceed on the recommendation, record it)

1. **Release support without the integer-ms fallback** (labels snapped, error logged). Recommend yes; the door stays open via the candidate family flag.
2. **FiLM trains overnight; token form (whole-track visibility) implemented tonight, compared tomorrow at matched exposure.** Recommend yes. The alternative, two half-night runs, leaves no model worth looking at in the morning.
3. **TCN landmarks instead of the GRU; no seed residual.** Recommend yes; both are recoverable as a measured comparison.
4. **Merge head rows closer than 2 ms at ingestion, and require T ≥ t_K + 2 ms.** Recommend yes; this is the census's convention and the only exportable choice.
5. **Device chosen by the pilot (CPU allowed).** Recommend yes; "M5" is the box, not a promise about MPS.
6. **If ffprobe fails tonight, use `api_total_length_s × 1000` with a flag** rather than not running. Recommend yes, with the ffprobe cache rebuilt tomorrow and the flagged charts re-cached.
7. **Schedule horizon from measured throughput, not 8M.** Recommend yes.
8. **DPO horizon as a parameter, synthetic tests at 64 rows.** Recommend yes.
