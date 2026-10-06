# R2 chart generator: architecture of the working tree

Repository `ensomi-model`, branch `r2/train`, HEAD `7d9640a` (R2 v1 plus the sequence-DPO trainer) with the uncommitted R2 v2 stage-0 change set on top. Paths are relative to `src/ensomi_model/r2/` unless they start with `src/`, `tests/` or `research/` (`research/` = `src/ensomi_model/research/`). Line numbers are working-tree lines.

What was run (read-only, on the mac): job `20261006-105046-r2-paramcount` (parameter counts), `20261006-105304-r2-example` (the worked example in §3.6, using the real code), `20261006-105335-r2-tokcheck` and `20261006-105623-r2-tokcheck2` (pytest on `tests/r2/test_conditions.py`, `tests/r2/test_locality.py` and one free-run test, with bytecode and pytest cache writes off). No repository file was edited.

---

## 1. Overview

R2 generates a 4-key osu!mania chart (taps and long notes, LNs) for a song whose **head skeleton** is given: the sorted times of the rows where at least one note head starts (`head_ms`, K rows), the BPM segments (a heads-only beat grid) and the song length T. It reads no audio and no source release times (`cache.py:1-9`, `features.py:3-9`). The unit of decision is the **head row**: decision k (k = 0..K-1) jointly chooses, for the 4 lanes, one of 5 codes (625 joint actions) and, for every lane that was holding an LN and closes inside the gap (t_{k-1}, t_k), the release time from a finite candidate set; one extra **EOS** decision k = K closes every remaining hold inside (t_{K-1}, T) (`common.py:7-18`). A row is legal only if it places at least one head (`state.py:49-57`). The network is a causal TCN over committed decisions with landmark read-outs, an exact-state query MLP, FiLM conditioning, a 625-way low-rank joint head and a release pointer averaged over the two mirror orientations (`model.py:1-9`), about 2.40 M parameters. Conditions (LN share, section difficulty) arrive as song-time-scoped intervals; rule L (`locality.py`) decides which decisions may read each interval. Training is teacher-forced cross-entropy on 256-decision windows drawn from a cached corpus, with condition tracks drawn from the source chart's own measured values, a fixed-divisor per-decision CE, inverse-probability weights on unconditioned decisions and condition-owned terms (`loss.py:1-20`, `train_ce.py:1-23`); a sequence-DPO trainer exists but needs real preference pairs that do not exist (`train_dpo.py:16-19`, `README.md:136-139`).

---

## 2. Inputs and outputs

### 2.1 Generation inputs

| Input | Type and unit | Where | Notes |
| --- | --- | --- | --- |
| Head times H | `head_ms` float64 [K], ms, strictly increasing | `features.py:36`, `sampling.py:47` | In the cache: distinct source head times, rows closer than 2 ms merged into the earlier one (`cache.py:83-91`, `common.py:38`). |
| BPM segments | `GridArrays` (offsets ms, canonical beat length ms, global beat origins, meters) | `common.py:69-111` | Built from the red lines and the head times only, never the release-aware grid (`cache.py:105-109`). "Beats" everywhere are **canonical global beats**: each segment's BPM is folded by powers of 2 into [80, 160) (`src/ensomi_model/evaluation/beats.py:34-47`, `common.py:79-92`). |
| Song length T | `song_ms` float, ms | `features.py:37` | Cache requires T >= t_{K-1} + 2 ms and T > last source release (`cache.py:101-104`). |
| Chart seed (optional) | `seed_actions` int [s,4], `seed_gap` float [s,4] (ms, NaN when absent) = the first s committed decisions; frontier g0 = t_{s-1} | `generate.py:114-131`, `generate.py:55-56` | Replay-checked before use (`sampling.py:64-66`). |
| Requests | `RequestSet` of `Request(id, scope (a,b) ms, property {ln_share or difficulty: Target(value)}, style=None, eta=Eta(), demand=None)`, each with its addition boundary g_u | `request_set.py:34-69`, `request_set.py:128-197` | Turned into the effective track (§6.1). The lower-level `continue_chart` takes the effective track directly (`sampling.py:10-12`). |
| Random seed | int, default 954 | `sampling.py:48`, `sampling.py:67` | One `torch.Generator` per continuation. |
| `stop`, `close_scope` | decision index; (a, b) | `sampling.py:48-53` | Partial generation; early stop once the scope's held LNs are closed (realised-response panels). |
| `baseline` | must be `None` | `sampling.py:35-37` | Reserved slot for the formulation's baseline style rho. |

### 2.2 Outputs

| Output | Form | Where |
| --- | --- | --- |
| Decisions | `actions` int [n,4] codes 0..4; `gap` float [n,4] gap-release ms, NaN when absent; n = K+1 for a full chart | `sampling.py:121` |
| Rows and objects | decisions replayed into chronological rows (gap closes sorted and coalesced, then the row at t_k), then hit objects (taps, holds) | `state.py:66-119`, `state.py:157-172`, `export.py:14-18` |
| `.osu` | minimal v14 file: `[TimingPoints]` from the notated grid segments, `[HitObjects]` x = 64 + 128 lane; written only if the legality checker finds no violation | `export.py:25-54` |
| Generation record (receipt) | JSON with fields `code, checkpoint, nu, evaluators, operating_point, decoding, rule_l, presence, random_seed, chart_seed, baseline, requests, effective_track, flags, targets, co_active, defects` (+ `baseline_model_sha256`, `operating_point_definition`, `complete`) | `generate.py:94-111`, `generate.py:134-136` |
| CLI files | `<sha16>-<seed>.npz` (actions, gap), `.osu` if complete, `.json` record, under `--out` | `generate.py:183-191` |

---

## 3. Decision unit and action space

### 3.1 What decision k produces

- Decision k < K: first the closes of the held lanes that release strictly inside (t_{k-1}, t_k), then the row at t_k (`state.py:66-101`; `locality.py:3-6`).
- Decision 0 has no gap: no lane can be held before it (`features.py:78-79` raises for a gap before decision 0).
- Decision K (EOS): only the closes inside (t_{K-1}, T); held lanes must use code 2 and free lanes code 0 (`common.py:17-18`, `state.py:52-54`). EOS never becomes a history token (`features.py:194`).
- Equal release times of different lanes coalesce into one physical row (`state.py:84`, `state.py:88-92`).

### 3.2 Per-lane codes (`common.py:7-15`)

| code | lane free before k | lane held before k |
| --- | --- | --- |
| 0 | nothing | keep holding |
| 1 | tap at t_k | release exactly at t_k (on the row) |
| 2 | LN head at t_k | release strictly inside (t_{k-1}, t_k), nothing at t_k |
| 3 | invalid | gap release, then tap at t_k |
| 4 | invalid | gap release, then LN head at t_k |

Joint action index = ((a0·5 + a1)·5 + a2)·5 + a3 over `ACTIONS` = all 5^4 = 625 tuples (`common.py:41-51`). `MIRROR_ACTION` maps an action to its lane-reversed action (`common.py:44`).

### 3.3 Legality mask (`state.py:49-57`)

Free lanes allow codes 0-2, held lanes 0-4; at least one lane must place a head (free: code 1 or 2; held: code 3 or 4). EOS allows exactly one action. Legal counts by number of held lanes (computed on the mac with `action_support_mask`): 0 held: 80; 1: 132; 2: 216; 3: 348; 4: 544; EOS: 1. Masked entries get -inf before the log-softmax (`model.py:254-257`). `advance` re-checks every decision through the oracle replay, so an illegal decision raises (`state.py:104-119`).

### 3.4 Release candidates (version `cand-v1`, `candidates.py:1-9`, `candidates.py:58-109`)

For a gap (a, b) with b - a >= 2 ms:

- the rounded (floor(t + 0.5)) positions of the 1/32 and 1/24 canonical-beat grids of every segment the gap intersects, strictly inside (a, b);
- the rounded midpoint;
- the rounded times 1/8 and 1/4 canonical global beat before b.

Union, sorted, unique. Flags per candidate: midpoint, eighth-before, quarter-before, on-grid (`candidates.py:90-93`). The set is never empty because the rounded midpoint is interior for gaps >= 2 ms (`candidates.py:7-8`). Size: a 500 ms gap at 120 BPM has 47 candidates, a 600 ms gap 60 (mac run, §3.6). Source releases are snapped to the nearest candidate, ties to the earlier (`candidates.py:112-118`, `cache.py:125-133`); a release exactly on a head row becomes code 1 instead (`cache.py:121-124`).

### 3.5 Joint chord releases, pointer order, mirror equivariance

- Several lanes may close in one gap. They are scored as a chain of **directed factors**, one per releasing lane, in lane order 0→3 in the forward orientation and 3→0 in the mirrored one (`features.py:437-439`). Each factor sees the releases already placed in its decision: lanes released on the row (code 1, at t_k, `on_row` = 1) from the start, and earlier factors' gap releases (`features.py:442-444`, `features.py:463-475`), through 28 placement-relation features per candidate (§4.5).
- The mirrored factor uses the swapped hand vectors, reversed codes and slot 3 - lane (`features.py:430-431`, `model.py:299-309`).
- Release likelihood per decision: log Q = logsumexp(Σ fwd factor log-probs, Σ mirrored factor log-probs) - log 2; 0 when the decision has no gap release (`model.py:289-297`, `model.py:4-9`).
- The joint head is the R1 `JointHead` with hand vocabularies left = 5·a0 + a1 and right = 5·a3 + a2 (`model.py:67-73`); the pair coupling is (M_left + M_rightᵀ)/2 (`research/bounded_typed_continuation/model.py:88-94`). Hand 0 lists lanes (0,1,2,3), hand 1 lists (3,2,1,0), so mirroring swaps the hands (`common.py:3-5`, `common.py:46`). Tests: `tests/r2/test_mirror.py` (mirror identity in CPU float64 and MPS float32; the directed pointer alone breaks it).
- In sampling, a fair coin picks the orientation, then lanes are sampled one at a time in that orientation's order (`sampling.py:93-105`), which samples the same two-orientation mixture [inferred].

### 3.6 Worked example (run through the real code on the mac)

Grid: one segment, offset 0 ms, 500 ms per beat (120 BPM), 4/4. Heads H = [1000, 1500, 2000] ms, T = 2600 ms, K = 3, so 4 decisions.

| k | t_k / gap | held before k | codes (lane 0..3) | gap release | action index | legal actions |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1000 / none | ---- | (2,0,1,0): LN head lane 0, tap lane 2 | - | 255 | 80 |
| 1 | 1500 / (1000,1500), C = 47 | lane 0 | (2,1,0,0): lane 0 closes in the gap, tap lane 1 | lane 0 at 1250 (candidate index 23, the midpoint) | 275 | 132 |
| 2 | 2000 / (1500,2000), C = 47 | ---- | (0,0,0,2): LN head lane 3 | - | 2 | 80 |
| 3 = EOS | T = 2600 / (2000,2600), C = 60 | lane 3 | (0,0,0,2) forced | lane 3 at 2250 | 2 | 1 |

Gap (1000, 1500): candidates 1016, 1021, 1031, 1042, 1047, 1063, … 1458, 1469, 1479, 1484; midpoint 1250, eighth-before 1438, quarter-before 1375. EOS gap (2000, 2600): midpoint 2300, eighth-before 2538, quarter-before 2475.

Replayed rows (action per lane; 1 tap, 2 LN start, 3 LN close): 1000 [2,0,1,0]; 1250 [3,0,0,0]; 1500 [0,1,0,0]; 2000 [0,0,0,2]; 2250 [0,0,0,3]. Objects: tap lane 2 at 1000; hold lane 0 1000→1250; tap lane 1 at 1500; hold lane 3 2000→2250.

Shapes: history tokens [3,2,100]; queries [4,2,150]; lane query [4,4,15]; hand vectors [4,2,128]; masked log-probs [4,625]; release factors: decision 1 (fwd slot 0, mirrored slot 3) and decision 3 (fwd slot 3, mirrored slot 0), each with candidate features [C,34] and relations [C,28]; candidate pairs scored = 2·47 + 2·60 = 214.

With an LN request interval [1250, T] value 0.5: V_0 = V_1 = ∅ (decision 1 has t_1 in scope but lane 0 is held and its earliest candidate 1016 < 1250, a masked onset), V_2 = V_3 = {the interval}. `locality.masked` reports onset 1, first visible 2, `onset_masked` true, `in_scope_outside_v` [1]. The LN frame of decision 2 at t = 2000 is [value 0.0 (= 2·0.5 - 1); ψ(a - t) ms -0.75, -0.5596; beats -1.5, -0.9163; ψ(b - t) ms 0.6, 0.47; beats 1.2, 0.7885; progress 0.5556; log1p heads 0.6931, log1p LNs 0, ratio 0, 0; log1p remaining rows 0.6931; active 1; lead-in 0].

---

## 4. State and features

All features are computed in float64 on CPU from committed decisions, then cast to the model dtype (`features.py:1-9`, `model.py:178-179`).

### 4.1 Raw state

- `R2State`: next decision index k, the exact replay state (open LN starts, last attack/release per lane), the birth row of each open LN, `finalized` (`state.py:38-46`).
- `derive(chart)` (`features.py:108-167`) gives, for each decision k, the state **before** k: `held` [n+1,4], held LN `start` ms, `birth` row, `release` time committed by k, `row_release`, `attack` (head placed), `ln_head`, `last_attack`, `last_release`, and prefix sums `cum_heads`, `cum_ln`, `cum_repeat` (heads whose lane also had a head on the previous row), `cum_held` (held lanes entering each row) (`features.py:91-105`).

### 4.2 Scalar transform

ψ_s(d) = [clip(d/s, -32, 32), sign(d)·log1p(|d|/s)]; ms use s = 1000, canonical beats s = 1; `psi_pair` concatenates both: 4 numbers per time difference (`common.py:54-62`). `phase(beats)` = sin/cos at periods 1, 4, 16 beats: 6 numbers (`candidates.py:24`, `candidates.py:34-40`).

### 4.3 History token [N,2,100] per head decision j (`features.py:193-210`)

Per lane (19, in each hand's lane order): code one-hot 5; held before j 1; released at j 1; views of the release time relative to t_j, t_{j-1} and the held start, in ms and beats (3 × 4 = 12) (`features.py:170-184`). Four lanes = 76. Shared (24): ψ(t_j - t_{j-1}) 4; ψ(t_j - t_0) 4; phase 6; one-hot of heads placed 5; one-hot of LN heads placed 5. Hand 0 orders lanes (0,1,2,3), hand 1 (3,2,1,0). The token carries no condition input.

### 4.4 Query [m,2,150] for decision k (`features.py:213-262`)

- Lane block (`lane_query`, 15 per lane, 60): held 1; LN age ψ (t_k - held start) 4; last-attack age 4; last-release age 4; no-attack-yet bit 1; no-release-yet bit 1.
- Shared 20: ψ gap to previous row 4; ψ remaining to T 4; t_k/T 1; k/K 1; first-decision bit 1; EOS bit 1; phase 6; BPM/120 1; meter/4 1.
- Density 6: log1p(head rows within 1, 2, 4, 8, 16, 32 beats after t_k) (zero at EOS).
- Lookahead 64: ψ of the next 16 inter-row gaps in ms and beats (zero past the last row) (`features.py:24-25`, `features.py:252-258`).

### 4.5 Release-factor inputs

- Lane features [15] of the releasing lane (the `lane_query` row) (`features.py:410`).
- Candidate features [C,34] (`candidates.py:22-23`, `candidates.py:52-55`, `candidates.py:105-108`): ψ(u - b) ms/beats 4; ψ(u - a) 4; ψ(u - held start) 4; fraction (u - a)/(b - a) 1; flags 4; snap one-hot over denominators 1, 2, 3, 4, 6, 8, 12, 16, 24, 32 within 1 ms, first match only, 10 (`candidates.py:20-21`, `candidates.py:96-103`); BPM/120 1; phase 6.
- Placement relations [C,28] = 4 oriented lane slots × 7: ψ(u - r) ms/beats 4; same birth row 1; placed 1; on the row 1 (`features.py:447-460`).
- Codes: the 4 oriented codes through an `Embedding(5, 8)`; a one-hot of the releasing slot 4 (`model.py:153-154`, `model.py:305-309`).

### 4.6 Condition inputs

FiLM frame [Q,2,17] per query time, one 17-vector per kind (0 = LN share, 1 = difficulty) (`features.py:328-369`): value 1 (`encode_value`: LN 2v - 1; absolute star v/4; residual star clip(v/0.5, -3, 3), `features.py:320-325`); ψ(a - t) ms/beats 4; ψ(b - t) 4; progress (t - a)/(b - a) 1; committed statistics 4 (LN: log1p heads, log1p LN heads, LN ratio, 0; difficulty: `star_proxies` = mean chord size/4, same-lane repeat rate, held-lane occupancy, LN share, `features.py:303-317`); log1p remaining head rows 1; active 1; reserved lead-in channel 1 (zero under `presence='none'`) (`features.py:29-31`). Counts use only decisions < k whose head time is in the scope (`features.py:292-300`). A kind with no visible interval active at the time is all zero (`features.py:332-334`), so "no request" (zeros) and "explicit 0" (value -1, active 1) differ (`tests/r2/test_locality.py::test_t_z_explicit_zero_differs_from_no_request`).

Token form [Q,NI,18] (token conditioner only): kind one-hot 2, value 1, ψ(a - t) 4, ψ(b - t) 4, active 1, progress 1, 4 counters, started 1 (`features.py:372-398`). It is currently broken (§10).

---

## 5. Model (`model.py`)

### 5.1 Configuration

`R2Config` defaults (`model.py:33-64`): hidden 128, TCN levels 8, expansion 4, joint-head rank 16, memory `landmarks`, landmark stride 64, conditioner `film`, code_dim 8, candidate_budget 8192 pairs, max_parameters 4,500,000, `rule_l` True, `birth_role` False, `presence` 'none', `star_value` 'absolute', `token_lead_in` False. The trainer passes only memory, levels, conditioner, hidden, expansion, rank, max_parameters, presence, rule_l, star_value (`train_ce.py:229-232`); stride, code_dim, candidate_budget, birth_role and token_lead_in stay at their defaults. Both v2 configs use hidden 128, levels 8 (implicit expansion 4, rank 16), FiLM, landmarks (`configs/ce_v2_a0.json`, `configs/ce_v2_a1.json`).

### 5.2 Modules and parameters (mac count with the default config)

| # | Module | Layers | Params |
| --- | --- | --- | --- |
| 1 | `temporal` (`FiniteTemporal`, `research/bounded_typed_continuation/temporal.py:86-120`) | Linear(100→128) per hand; 8 `CausalBlock`s, dilations 1, 2, 4, …, 128: LayerNorm → causal Conv1d(128→256, k=3) → tanh·sigmoid gate → Linear(128→128) residual; LayerNorm → Linear(128→512) → GELU → Linear(512→128) residual (`temporal.py:45-64`); learned BOS/TRUNCATED boundary [2,128]. Receptive field 1 + 2·255 = 511 tokens (`temporal.py:35-37`). Hands share weights (batched as 2B sequences, `temporal.py:115-120`). | 1,991,552 |
| 2 | `exact` | Linear(150→128) → GELU → Linear(128→128) | 35,840 |
| 3 | `fuse` | LayerNorm(256) → Linear(256→128) → GELU → Linear(128→128) | 49,920 |
| 4 | `lm_query`, `lm_key`, `lm_value`, `lm_out` | 3 × Linear(128→128); Linear(128→128, no bias), zero-initialised | 49,536 + 16,384 |
| 5 | `film` | MLP Linear(68→128) → GELU → Linear(128→256) (last layer zero-initialised) + LayerNorm(128); 68 = 2 roles × 2 kinds × 17 | 42,112 |
| 6 | `tokens` (`TokenConditioner`) | Linear(18→128) → GELU → Linear(128→128); kind Embedding(2,128); null token; LayerNorm; MultiheadAttention(128, 4 heads); output Linear (zero-init). Built always, used only when `conditioner='tokens'` | 102,144 |
| 7 | `joint` (`JointHead5`) | unary Linear(128→25) per hand; matrix Linear(128→256) per hand → 16×16; action Embedding(25,16) | 36,649 |
| 8 | `code_embedding` | Embedding(5,8) | 40 |
| 9 | `pointer_in` | Linear(307→128); 307 = 2·128 hands + 4·8 codes + 4 slot + 15 lane | 39,424 |
| 10 | `pointer_relation` | Linear(28→128, no bias) | 3,584 |
| 11 | `pointer_out` | Linear(128→128) | 16,512 |
| 12 | `candidate` | Linear(34→128) → GELU → Linear(128→128) | 20,992 |
| 13 | `candidate_bias` | Linear(34→1) | 35 |
| | **Total** | | **2,404,724** |

The constructor raises above `max_parameters` (`model.py:159-163`). The `tokens` module's 102,144 parameters count toward the total although the FiLM default never calls it. For comparison, v2 code with `birth_role=True` (FiLM input 102) has 2,409,076 (mac); the v1 model at `7d9640a` (FiLM input 6 × 16 = 96) has 2,408,308 by the same arithmetic [inferred from the code, not run].

### 5.3 Forward pass for a window of decisions ks (`model.py:226-287`)

1. History: tokens of decisions 0..N-1, N = min(max k, K) (`model.py:233-236`), through the TCN → `enc` [N,2,128] (`model.py:226-230`). The TCN always runs from token 0, even for a late window.
2. `before[k]` = enc[k-1], or the BOS boundary vector for k = 0 (`model.py:238-245`).
3. Landmarks: `marks` = enc at positions 0, 64, 128, … < N; decision k sees marks at positions < k (`model.py:246-248`). Read-out: single-head scaled dot-product attention per hand (hand-to-same-hand), masked to visible marks, zero when none is visible, through the zero-initialised `lm_out` (`model.py:183-192`).
4. Query: `exact(qf)` on [m,2,150] (`model.py:249`, `model.py:201`).
5. h = fuse(concat(before, exact)) [m,2,128]; h += landmark read (`model.py:200-204`).
6. Condition: z = h + γ ⊙ LayerNorm(h) + δ, (γ, δ) = FiLM-MLP(cond [m,68]), the same (γ, δ) for both hands (`model.py:76-89`, `model.py:194-198`). Row cond: role 0 = frames at t_k of V_k's intervals, role 1 (candidate) all zero (`model.py:214-224`).
7. Joint head: logits [m,625] = unary_left[l] + unary_right[r] + E_l (M_L + M_Rᵀ)/2 E_rᵀ / √16 with l = 5a0 + a1, r = 5a3 + a2; masked log-softmax (`model.py:254-257`, `research/bounded_typed_continuation/model.py:88-94`).
8. Release factors (both orientations) (`model.py:276-280`): base = pointer_in(concat(left hand, right hand [swapped if mirrored], code embeddings 32, slot one-hot 4, lane features 15)) [F,128] (`model.py:299-309`).
9. Per candidate pair: q = pointer_out(GELU(base[owner] + pointer_relation(rel))) → FiLM(q, pair cond [P,68]) where role 0 = row frame at t_k and role 1 = frame at the candidate time u; score = q · candidate(cand) / √128 + candidate_bias(cand) (`model.py:311-342`). Above 8192 pairs with gradients on, scored in checkpointed blocks (`model.py:344-358`).
10. Per-factor normaliser over its candidates (streaming max-shifted log-sum-exp) → factor log-prob of the target (`model.py:360-370`); orientation mixture → release [m] (`model.py:289-297`).
11. Outputs `WindowOut`: action [m], release [m], eos, gap_lns, `governed_ln` [m], `visible` [m,2], `logp` [m,625], `candidate_pairs` (`model.py:117-136`, `model.py:286-287`).

`governed_ln` = log P(a) - log Σ P(a') over legal a' with the same head mask and release types: per lane, free codes 1/2 merge and held codes 3/4 merge, i.e. the tap-versus-LN part of the row decision (`model.py:384-411`).

### 5.4 How conditions enter

| Path | Mechanism | Where |
| --- | --- | --- |
| Row decisions | FiLM (scale γ on LayerNorm(h) plus shift δ) on both hand vectors, after fuse and landmarks; input = frames at t_k | `model.py:200-204`, `model.py:214-224` |
| Release pointer | the same FiLM module on each pointer query; input = row frame at t_k + frame at the candidate time | `model.py:311-342` |
| Hand vectors into the pointer | the row-conditioned z feeds `pointer_in` | `model.py:299-309` |
| Masks | rule L chooses V_k (which intervals at all); the activity test a <= t < b (closed at T when b = T) zeroes a kind at a query time outside its interval | `locality.py:39-47`, `features.py:280-283`, `features.py:354-368` |
| History | none: history tokens hold decisions only; a condition reaches later decisions only through what was committed [inferred from `features.py:193-210`] | |
| Loss | condition terms score only decisions in V_k (ownership = visibility) | `loss.py:16-19`, `loss.py:41-59` |
| Switches (not default) | `presence='anywhere'` sets the lead-in channel whenever the full track has the kind (v1 reproduction); `birth_role=True` adds a third frame role at the held LN's start; `rule_l=False` shows the whole track; `conditioner='tokens'` cross-attends over the whole track (lead-in form) | `features.py:344-347`, `model.py:326-327`, `locality.py:44-45`, `model.py:92-114` |

### 5.5 Rule L (`locality.py:1-47`)

Decision k reads interval I = [a, b) (or [a, T]) if and only if every time it can produce lies in I:

- k < K: t_k ∈ I, and either no lane is held entering k or the earliest release candidate of gap k is >= a (`locality.py:29-32`);
- k = K (EOS): some lane is held, the earliest candidate >= a, and b = T or the latest candidate < b (`locality.py:33-36`).

Consequences stated in the code: no decision before a scope's onset reads the request; every decision that can produce a time at or after b reads nothing of it; V_k depends only on the skeleton, the scopes and the occupancy entering k, so it is the same in teacher forcing and sampling (`locality.py:11-15`). The onset decision is masked when a hold is open there and a candidate precedes a; the exit decision (first row at or after b) never reads I (`locality.py:69-102`). Version string `rule-L-v1` (`locality.py:23`). In the model, V_k is computed per decision for rows (`model.py:206-208`, `model.py:265`) and per factor for the pointer (`model.py:318-321`).

---

## 6. Generation loop

### 6.1 Request validation and the effective track (`request_set.py`)

1. Schema (`check_schema`, `request_set.py:87-121`): non-empty id; `style`, `demand`, `eta.priority` must be None and `eta.transitions` empty; at least one directive; 0 <= a < b <= T; property in {ln_share, difficulty}; one finite scalar `Target` (no ranges); ν must be the module's; `strength == 'default'`; LN target in [0, 1]; difficulty scope >= 30 s.
2. Add (`request_set.py:144-157`): unique id; addition boundary g_u < a (None = 0^-); two targets for the same property with intersecting scopes raise.
3. Withdraw/replace only while g < a; after the start a request can be neither cancelled nor changed (`request_set.py:159-175`). A continuation call may not resume from before a request's addition (`request_set.py:177-181`).
4. `effective_track` (`request_set.py:212-259`): one `Interval(kind, a, b, value)` per target (never merged), with provenance; flags: `ungovernable` (no head row in scope), `unattainable` (LN readout undefined), `extrapolation` (LN scope outside 8-512 beats unless whole song; difficulty length not within 1 ms of 30/60/120 s unless whole), residual outside ±1.5 star; residual mode replaces the value by target - b(S) and needs the frozen baseline; co-active pairs (different properties, intersecting scopes); `validate_track` (no overlap within a kind, values in range).

### 6.2 Properties and ν (`properties.py`)

ν = `r2-nu-v1` (`properties.py:36-55`), hashed into `NU_HASH`. Scope [a, b), or [a, T] when b = T; an object belongs to the scope of its head. LN share = LN heads / heads, undefined when the scope owns no head (`properties.py:118-137`). Difficulty = tiled star: the scope's own objects translated by -a and repeated with period b - a up to 240 s, an LN still held at the next copy's first head in its lane cut to end 1 ms before it, scored with `compute_mania_star_rating_20241007(objects, 4, clock_rate=1.0)`; undefined below 30 s, for an empty scope, a nonpositive cut, or a scope-headed hold not yet closed (`properties.py:146-188`). Labels, draws, frame counters, validation and readouts all read this module (`properties.py:1-6`).

### 6.3 Per-decision loop (`sampling.py:46-121`)

1. Refuse a baseline; model to eval; replay-check the prefix (`sampling.py:54-66`).
2. Seed `torch.Generator`; build the TCN online cache from the prefix's history tokens, saving a landmark every 64 tokens (`sampling.py:67-78`).
3. For k = s … stop-1 (`sampling.py:82-118`):
   - chart with decisions < k; held lanes;
   - hands: `before` = cache read, landmarks, query features, row condition (V_k by rule L) → z (`sampling.py:85-89`);
   - mask the support; masked log-softmax; Gumbel-max sample of the joint action on CPU float64 (`sampling.py:26-29`, `sampling.py:90-92`);
   - fair-coin orientation; lanes that close in the gap in that orientation's order (`sampling.py:93-94`);
   - per lane: build the factor with current placements, score all candidates, Gumbel-max pick u, add u to placements (`sampling.py:96-105`);
   - commit; with `close_scope`, stop at the first decision at or after the exit decision after which no LN headed in [a, b) is still held (`sampling.py:106-112`, `sampling.py:40-43`);
   - if k < K append history token k to the TCN cache; landmark when k % 64 == 0 (`sampling.py:113-118`).
4. Return actions[:n], gap[:n].

### 6.4 Decoding (the default operating point's decoding, `operating_point.py:17-18`)

Gumbel maximum on the masked log-softmax (CPU float64), temperature 1.0 (there is no temperature parameter in the code), fair-coin orientation, pointer one lane at a time in the orientation order, no truncation.

### 6.5 Record and export (`generate.py`)

`generate()` (`generate.py:114-131`): refuse baseline → `check_frontier(frontier_of(head_ms, s))` → `effective_track` → `continue_chart` → `prefix_objects` (open holds kept with end None, `properties.py:82-106`) → `generation_record`. Readouts (`generate.py:68-91`): per target, the ν readout from generated objects (never copied from the request), deviation (None if undefined, with reason), holds headed in scope and closed at or after b (`crossing`), rule-L masking (`first_visible`, `onset`, `exit`, `onset_masked`, `in_scope_outside_v`, `exit_closes`), `ungovernable`, and `realised_residual` in residual mode. Defects (`generate.py:59-65`): holds, holds <= 60 ms, releases 1-40 ms before another lane's head. The CLI (`generate.py:139-193`) works on a fit_dev cache chart by sha, adds `requests.json` at the chart seed's frontier or continues a previous record, and exports `.osu` only for a complete chart (`export.py:45-54` raises on any legality violation).

### 6.6 Checkpoint selection and operating point

- `select.py` (post-run, `select.py:70-99`): candidates = regular (non-safe) checkpoints after warm-up with full panels; primary = natural-manifest per-decision NLL averaged over the checkpoint and its two predecessors, with a song-group bootstrap (2,000 resamples, `select.py:42-52`); guards (`select.py:55-67`): legal (no illegal run, every head present), (i) prefix-panel LN-share difference within ±0.05, (ii) onset slope >= 0.7 and MAE <= 0.15, (iii) own-history calibration gap <= 0.05, (iv) holds <= 60 ms <= 0.5 % and 1-40 ms releases <= the source rate, (a1) identity after the exit decision. Pick the earliest passing candidate within 2 SE of the passing minimum; none passing → no selection. Writes `selection.json` and an adherence report (`select.py:102-111`, `select.py:114-136`).
- `operating_point.py`: the default operating point = hash of {recipe hash (config minus run_dir, device, threads, stop_at_unix, freerun, freerun_seeds), selection rule + `select.py` source hash, decoding, conditioning (rule L, presence, eta, conditioner, star value, baseline style 'none')} (`operating_point.py:20-65`). Every generation record names it (`generate.py:100-105`).

### 6.7 `proxy.py` (stage 2, training only)

`expected_proxies` returns the four `star_proxies` statistics of a scope where decisions that read the scope (rule L) enter through expected head, LN-head and same-lane-repeat counts under exp(log P(a)), and other decisions through their sampled action; held-lane occupancy is history without gradient (`proxy.py:1-8`, `proxy.py:33-63`).

---

## 7. Training

### 7.1 Corpus, splits, cache

- Population: corpus rows with `eval_split == 'fit'`, status ranked or loved, 2 <= star <= 6; fit_dev holds a song group when the first 8 bytes of SHA-256('r2-fit-dev-v1:' + group_id) are 0 mod 10, else fit_train (`splits.py:1-60`).
- Cache (`r2-cache-v1`, `cache.py:73-148`, `cache.py:198-246`): one `.npz` per chart: `head_ms` [K], `actions` int8 [K+1,4] (row K = EOS), `gap_release_ms` [K+1,4] snapped, `release_orig_ms`, `grid_segments` [S,3], `grid_bars`, `song_ms`. T from the audio container (`soundfile.info`). Exclusions with reasons (empty, illegal, merge collision, song too short, grid, lane conflict, round-trip failures). Round-trip checked (`cache.py:222-228`).
- `Corpus` (`data.py:54-112`): rows of one role; LRU of 128 charts; star cells from the v2 label file when star conditions are on; residual cells = label - b(S) (`data.py:90-100`).

### 7.2 Window and condition-track draw (`data.py:107-112`, `conditions.py:252-322`)

1. Song group uniform, chart uniform in the group (`data.py:102-105`).
2. Candidates: LN = a fresh partition of the song into 8/16/32/64-beat pieces (piece length drawn per piece); with p 0.10 the single candidate is the whole song; else with p 0.25 consecutive pieces merge into runs of U{2..8} pieces; value = the source's LN share over the piece's rows; pieces with no head dropped (`conditions.py:132-164`). Difficulty = one cell length from 30/60/120 s and one phase from 0/10/20 s, all cells of that tiling, or with p 0.10 the whole song; values from the label file (`conditions.py:167-178`). Star candidates only when star conditions are on.
3. Dropout first: all with p 0.20; each kind p 0.25; each candidate p 0.20 (`conditions.py:258-266`).
4. Onset per surviving candidate = the first decision that reads it under rule L on the source occupancy (`conditions.py:181-188`, `conditions.py:267-269`).
5. Start j:
   - `per_window` (A1): with p_align 0.6 and if some candidate has an onset: a kind uniformly, a lead u ~ U{0..32}, j = max(0, onset - u), one candidate chosen with weight 3 if it is an LN candidate whose value differs from the LN share of the 64 rows before j by z >= 2 binomial SE with n >= 20 heads, else weight 1 (`conditions.py:208-228`, `conditions.py:280-291`); otherwise the v1 start rule;
   - v1 start rule: j = 0 with p 0.125, max(0, K - 255) with p 0.125, else uniform on [0, K] (`conditions.py:191-205`).
6. Window = decisions [j, min(j + 256, K + 1)); EOS included exactly when reached (`conditions.py:294`, `data.py:5-7`).
7. Interval selection: `per_window`: per kind U{1..3} candidates intersecting the window's time span, the aligned one included, consecutive with p 0.5 (`conditions.py:300-318`); `per_song` (A0): U{1..4} LN and U{1..3} difficulty candidates per song, consecutive with p 0.5, start by the v1 rule (`conditions.py:271-278`).
8. Inverse-probability weight u = p_old(j) / ((1 - p_align) p_old(j) + p_align P_align(j)) when aligned draws are possible and `ipw` is on; it multiplies every scored decision with V_k empty (`conditions.py:295-297`, `loss.py:9-10`).
9. `validate_track` (`conditions.py:320`, `conditions.py:325-342`). A drawn track equals the effective track of requests all added at 0^- (`conditions.py:5-8`, `request_set.py:262-271`).

All parameters are `DrawConfig` fields (`conditions.py:48-97`); the draw hash covers them, ν and rule L (`conditions.py:92-97`).

### 7.3 Labels (`labels.py`)

`cells-v2`: for L ∈ {30, 60, 120} s and phase p ∈ {0, 10, 20} s, the cells [p + iL, p + (i+1)L) that end by T, plus the whole song [0, T] when T >= 30 s; value = Difficulty_ν of the cached decisions' objects, so a label equals the readout of the same chart; undefined → null, never used (`labels.py:1-13`, `labels.py:36-65`). File `labels/star-v2.json.gz`, refused if built under another ν (`labels.py:121-141`). LN-share values are computed online from row counts (`properties.ln_share_rows`, `conditions.py:124-129`).

### 7.4 Loss (`loss.py`)

Per scored decision j: ℓ_j = -(log P(A_j) + log Q(U_j)). For a window with weight u:

- L_base = (1/N̄) Σ_j w_j ℓ_j, w_j = u if V_j = ∅ else 1;
- L_ln = (1/N̄_ln) Σ over decisions reading an LN interval of -log P(tap-vs-LN split | head mask, release types) (`governed_ln`);
- L_star = (1/N̄_star) Σ over decisions reading a difficulty interval of ℓ_j;
- L = L_base + λ_ln L_ln + λ_star L_star (`loss.py:5-14`, `loss.py:47-68`).

Ownership: Ω_κ = decisions with an interval of kind κ in V_k (`loss.py:16-18`, `loss.py:41-44`). The divisors are fixed expected per-batch counts from `draw_sim` (N̄ = mean Σ w_j per batch of 4 windows; N̄_κ = mean count of Ω_κ factors: one action factor per head decision plus one per gap release), so each window is backpropagated on its own and the batch gradient is exact (`loss.py:18-20`, `draw_sim.py:2-11`). The trainer refuses to start unless `n_bar_key` matches the draw (`train_ce.py:313-317`); a missing N̄_κ becomes infinity, i.e. that term is zero (`train_ce.py:318-319`). EOS decisions are in L_base; their action term is 0 because only one action is legal [inferred from `state.py:52-54`].

### 7.5 Skeleton baseline b(S) (`baseline.py`, stage 2)

Quadratic ridge (λ = 1, unpenalised intercept) on 23 head-time features of the scope (duration, rows, density, occupied span, gap mean/sd/quantiles, short-gap shares, 1 s and 4 s count statistics) with all pairwise products: 1 + 23 + 276 = 300 columns, fitted on fit_train cells of all label lengths against Difficulty_ν (`baseline.py:1-14`, `baseline.py:33-70`, `baseline.py:128-163`). Skeleton-only and never fed to the model. A second linear ridge g maps three committed proxies (chord/4, repeat rate, LN share) to the residual Difficulty_ν - b(S) (`baseline.py:10-13`, `baseline.py:72-83`). Used for residual frame values, request flags, the relaxed-proxy term and the F3 measurement.

### 7.6 `draw_sim.py` (stage-0 measurement)

Draws windows from fit_train with a config's draw (default 2,000 draws), computes N̄, N̄_ln, N̄_star with standard errors and the `n_bar_key`, and checks D1 onset coverage >= 0.70, D2 slope of active intervals on K contains 0, D3 informative decisions >= 0.02 of scored heads, D4 active share LN/star >= 0.25, D6 natural effective sample size >= 0.50, D7 long-span (>64 beats) share >= 0.20, D8 in-scope outside-V share on 16-beat LN pieces <= 0.05 (`draw_sim.py:34-40`, `draw_sim.py:72-167`).

### 7.7 Optimiser, schedule, budget, arms

| Setting | Value | Where |
| --- | --- | --- |
| Optimiser | AdamW, β = (0.9, 0.95), ε 1e-8, weight decay 0.01 on tensors with ndim >= 2, 0 otherwise | `train_ce.py:324-328`, configs |
| Clip | global norm 1.0 | `train_ce.py:437` |
| Batch | 4 windows of up to 256 decisions per step (`accumulate`, `window`) | `train_ce.py:395-404`, configs |
| Exposure unit | scored head decisions (EOS excluded) | `train_ce.py:431`, `train_ce.py:442` |
| LR | linear warm-up to 3e-4 over 50,000 exposures, cosine to 3e-5 at `total_exposures`, unless `lr_schedule` lists phases (constant / linear / cosine) | `train_ce.py:162-188`, configs (`lr_schedule: null`) |
| Budget | 50,000,000 exposures; checkpoint every 4,388,000; full evaluation every 2nd checkpoint; G3(c) at the first full evaluation past 25 M and 50 M | `configs/ce_v2_a*.json`, `train_ce.py:105-107` |
| Seeds | weights 171, draws 471, validation 954, free-run 954-956 | configs |
| Arms | A0 = `per_song` draw, p_align 0, no IPW, λ_ln = λ_star = 0; A1 = `per_window` draw, p_align 0.6, IPW, λ_ln = λ_star = 1; every other key identical | `configs/ce_v2_a0.json`, `configs/ce_v2_a1.json` (only lines 52-57 differ), `tests/r2/test_draw.py::test_config_arms_differ_only_in_the_draw_and_the_terms` |
| Star conditions | `auto`: on iff `labels/star-v2_summary.json` says complete | `train_ce.py:215-218` |
| Robustness | non-finite loss or gradient → reload the latest checkpoint, skip the windows, halve the LR multiplier; third event stops with exit 3; resource guard (RSS 12 GiB, RSS growth 2 GiB between checkpoints, MPS 8 GiB, swap trip disabled) stops with exit 4 after a safe checkpoint; launcher runs from a frozen code copy, restarts at most 5 times in 6 h, reset by a new checkpoint | `train_ce.py:588-620`, `train_ce.py:60`, `train_ce.py:643-645`, `launch.py:1-16`, `launch.py:90-134` |

### 7.8 Stage-2 terms (off in both configs)

With `mu_star > 0` (requires `star_value='residual'` and star labels, `train_ce.py:320-321`): on windows with index % 4 == 0, sample the first difficulty scope a scored decision reads from the real prefix with the current model, re-score it teacher-forced, and add λ_star · μ_star · (g(E_θ[proxies]) - v_res)² (`train_ce.py:420-426`, `train_ce.py:450-479`); every 16 steps, a no-gradient "true F3" measurement on 4 samples run until the scope's LNs close, logged to `f3.jsonl` (`train_ce.py:444-445`, `train_ce.py:481-512`).

### 7.9 In-run evaluation cadence

At every checkpoint the teacher-forced evaluation runs; at every `full_eval_every`-th checkpoint the free-run panels run too (`train_ce.py:704-710`, `train_ce.py:520-535`). Failures are logged, never fatal (`train_ce.py:712-726`). Manifests are built once per run into `manifests.json` and rebuilt when their version hash changes (`data.py:226-238`).

### 7.10 DPO (`train_dpo.py`, unchanged since `7d9640a`)

Soft-label sequence DPO: R(y) = log π_θ(y|x) - log π_0(y|x) summed over the branch's decisions, loss = mean over pairs of -q log σ(βΔ) - (1-q) log σ(-βΔ) + λ_CE · CE(anchor windows), β 0.1, λ_CE 0.2, AdamW lr 1e-5, 50-step warm-up, 8 pairs and 8 anchor windows per update (`train_dpo.py:1-25`, `train_dpo.py:301-330`). Pairs come only from a file of real pairs; synthetic records are refused and the CLI refuses without `pairs_file` (`train_dpo.py:580-601`, `train_dpo.py:548-549`). Status: no real pair source exists (`README.md:136-139`), so it has not trained. The anchor CE is still the v1 window mean, `-total.mean()` (`train_dpo.py:190-191`).

---

## 8. Evaluation

### 8.1 `evaluate.py` (in-run, plan §9)

Teacher-forced, every checkpoint (`evaluate.py:340-354`):

- Natural manifest: per-decision NLL over 64 fit_dev windows by the v1 start rule with empty tracks; the selection primary (`evaluate.py:206-213`, `data.py:129-140`).
- Condition manifest (64 fit_dev windows by the configured draw, stratified: >= 24 windows with a scored LN onset, 24 star onset, 8 LN span > 64 beats, 8 value switch, 8 masked onset; `data.py:143-194`): per kind and stratum (onset = first 16 visible decisions, end = last 16, middle) the whole-decision NLL, the governed-split NLL (LN), the null contrast (NLL without the track minus with), informative-onset NLL (z >= 2), and counts of in-scope decisions outside V (`evaluate.py:224-265`).
- Train versus dev onset NLL (memorisation of up-weighted onsets) on a fit_train condition manifest (`evaluate.py:347-351`).
- Counterfactual: same state, interval value swapped (LN 0 vs 0.9; residual -0.5 vs +0.5, residual mode only): change of the expected LN fraction or expected heads, and the action KL, on onset and late decisions (`evaluate.py:272-306`).
- Representation probe: ridge from hand vectors at onset decisions to the frame value, 5-fold CV R² (`evaluate.py:308-338`).

Free-run panels, full evaluations (`evaluate.py:382-551`), every run recorded with its generation record to `records.jsonl`:

| Panel | What | Guard |
| --- | --- | --- |
| `natural_bos` | prefix panel (8 fixed recheck charts + 2 fit_dev charts per star band 2-5, 50-1,500 rows; `evaluate.py:150-165`) × seeds 954-956 from BOS: LN share minus source, drift (last third minus first third), chord-size JS | |
| `prefix_natural` | continue from the row nearest 1/3 of the song; generated minus real LN share of the continuation, chart-paired; SD ratio | (i) |mean| <= 0.05 |
| `onset` | 8 charts × LN scopes of 16/32 beats at 1/3 and 2/3 × targets 0.05/0.3/0.6/0.9 × 3 seeds, generated from the scope's onset until its LNs close: slope and MAE of readout on target, masked-onset share | (ii) slope >= 0.7, MAE <= 0.15 |
| `whole` | whole-song LN targets 0/0.1/0.3/0.6/0.9 | |
| `switch` | 0.1→0.6 and 0.6→0.1 at half song: DiD/2, per-half MAE | |
| `residual_star` (residual mode) | 30/60 s scopes, targets b(S) + {-0.5, -0.25, 0, 0.25, 0.5}: residual slope, MAE, and |Δ_tail| against the trimmed difficulty | |
| `calibration` | expected LN forecast over 64 decisions on real against own-sampled history from 96 start states per chart | (iii) |gap| <= 0.05 |
| `g3` | (a1) identity after the exit decision: rows and log-probs equal with and without the request; (a2) persistence after the scope; (f) explicit 0 versus no request; (b) organisation statistics kept under an LN request against seed-to-seed reference; (c) within/between variance over 8 seeds when due | (a1) |
| `legality`, `defects`, `source_defects` | illegal runs, missing heads, holds <= 60 ms rate, 1-40 ms release rate | (iv), legal |

### 8.2 `report.py` (v1, unchanged)

`chart_summary` of one free-run chart: violations, heads present, LN share, hold <= 40/60 ms shares, releases 1-40 ms before another head, release categories (on row, midpoint, eighth, quarter, other), chord histogram, open-hold occupancy at head rows, LN share of the first and last third (`report.py:14-63`). Used by the `--freerun-only` mode (`train_ce.py:537-561`).

### 8.3 `src/ensomi_model/evaluation/operators/` (untracked; not part of R2)

File times are 2026-10-03 and nothing in `r2/` or `tests/r2/` imports it. Its own docstring: "Experimental, corpus-calibrated descriptions of 4K chart organisation and demand", reported under `artifacts/eval-operators-20261003` (`operators/__init__.py:1-6`). Modules: `descriptions.py` (event descriptions on 16-canonical-beat windows: load channels per finger and hand, release load; organisation channels for transitions, rhythm residuals, runs, density, holds, recurrence; mirror handled jointly), `normal.py` (fit-only density cells and conditional two-sided JS divergence, bootstrap by song, conformal threshold), `gaps.py` (competing parametric gap densities), `injections.py` (prespecified legal corpus interventions with achieved dose), `analysis.py` (gap-density, context-permutation and star-information experiments), `experiment.py` (registered extraction, reference fitting, defect calibration), `provenance.py` (group selection, hashes), `stop_summary.py` (finalising after a registered stop). Tests: `tests/evaluation/test_operators_*.py` (untracked).

---

## 9. v1 → v2 delta (working tree against `7d9640a`)

Unchanged files: `common.py`, `state.py`, `candidates.py`, `cache.py`, `export.py`, `receipts.py`, `report.py`, `splits.py`, `train_dpo.py`, `configs/ce_v1.json`, `README.md`, `DEVIATIONS.md`.

| Module | v1 (`7d9640a`) | Working tree (v2 stage 0) |
| --- | --- | --- |
| `features.py` | `FRAME_DIM` 16: value, offsets 8, progress, 2 counts, ratio, remaining, active, **presence bit** set whenever the track has the kind; frames applied over the whole track; activity a <= t < b (open at T); kinds `ln`, `star`; `_norm_value` | `FRAME_DIM` 17 (`features.py:26`); frames take only V_k (`features.py:332-334`); reserved lead-in channel replaces presence (`presence='anywhere'` reproduces v1, `features.py:344-347`); activity closed at T (`features.py:280-283`); star frame stats = `star_proxies` (`features.py:303-317`); `encode_value` with residual (`features.py:320-325`); `Derived` gains `cum_repeat`, `cum_held` (`features.py:104-105`); `_norm_value` removed but still called (§10) |
| `model.py` | FiLM input 6 × 16 = 96 (row, candidate, **birth** roles); no rule L; `WindowOut(action, release, ks, eos, gap_lns)` | `R2Config` switches `rule_l`, `birth_role`, `presence`, `star_value`, `token_lead_in` (`model.py:45-49`); FiLM input roles × 2 × 17 = 68 by default (`model.py:76-84`); V_k per row and per factor (`model.py:206-224`, `model.py:311-329`); `governed_split` and `LN_GROUPS` (`model.py:384-411`); `WindowOut` adds `governed_ln`, `visible`, `logp`, `candidate_pairs`, `factors`; token conditioner raises unless `token_lead_in`; 2,408,308 → 2,404,724 params |
| `locality.py` | absent | new: rule L, `visible`, `masked`, `scope_decisions` |
| `conditions.py` | `draw_track`: fixed constants; LN = 1-4 of the partition pieces; star = whole song (p 0.10) or 1-3 cached 30/60 s cells; dropout **after** selection; `replace_interval` runtime edit that clipped and merged intervals | `DrawConfig` (all parameters, hash); LN runs of 2-8 pieces and whole-song candidates; star cells 30/60/120 s × 3 phases; dropout first; onset alignment with importance weights and IPW (`per_window`) or v1 selection (`per_song`); `validate_track` with [0, T] bound and residual values; `replace_interval` removed |
| `request_set.py` | absent | new: requests, request set (add / withdraw / replace before start), validation, effective track, flags, co-active pairs |
| `properties.py` | absent (LN share and tiled star lived in `labels.py`) | new: ν, readouts, tiling, undefined cases |
| `labels.py` | ≤ 4 consecutive 30 s cells + ≤ 4 consecutive 60 s cells (start by a hash of the inputs) + whole song [0, T); computed from the original `.osu`, empty cell = 0.0; `labels/star.json.gz` | `cells-v2`: 30/60/120 s × phases 0/10/20 s, all cells ending by T + whole [0, T]; from the cache representation through ν; undefined = null; `labels/star-v2.json.gz`; v1 loader kept |
| `data.py` | v1 start rule inline; one fit_dev manifest of 64 windows with drawn tracks + 4 free-run charts, rebuilt only when `star_conditions` changes | draw via `conditions.draw_window`; `Draw.weight`; natural manifest (empty tracks), stratified condition manifest, fit_train condition manifest, all versioned by hash; residual star cells |
| `loss.py` | absent; step loss = per-window mean NLL / batch size | new: fixed-divisor base CE with IPW plus λ-weighted condition terms owned by visibility |
| `baseline.py`, `draw_sim.py`, `proxy.py` | absent | new (stage 0 measurement, stage 2) |
| `sampling.py` | `continue_chart(model, H, T, grid, prefix, track, seed, stop)` | adds `baseline` (None only) and `close_scope`; documents the effective track and per-decision rule L |
| `generate.py`, `operating_point.py`, `select.py`, `evaluate.py` | absent (free runs only through `train_ce.freerun`) | new: request CLI and record; operating-point hash; frozen selection rule; plan §9 evaluator |
| `train_ce.py` | `TrainConfig` without condition or schedule keys; loss -mean per window; eval = fit_dev manifest CE + 12 free runs; logs `train.jsonl` in run dir | new keys `presence, rule_l, star_value, draw, lambda_ln, lambda_star, mu_star, proxy_every, f3_every, f3_samples, n_bar*, n_bar_key, lr_schedule, full_eval_every, g3c_exposures, rss_growth_limit_gib`; `loss.py`; N̄ key check; relaxed proxy + F3; `Evaluator`; segmented logs under `logs/`; RSS-growth limit; swap trip off; checkpoint payload adds draw hash, N̄ key, ν hash |
| `launch.py` | at most 5 restarts in total | at most 5 restarts in any 6 h, reset when a new regular checkpoint appears |
| `configs/` | `ce_v1.json` | + `ce_v2_a0.json`, `ce_v2_a1.json` (stage-1 arms) |
| tests | v1 suites | modified `helpers.py`, `test_conditions.py`, `test_freerun.py`, `test_train_step.py`; new `test_draw.py`, `test_locality.py`, `test_properties.py`, `test_requests.py`, `locality_fixture.py` |
| `.gitignore` | | adds `/.claude/skills/research-relay`, `/.claude/settings.local.json` |

---

## 10. Stubs, open switches and defects

### 10.1 Reserved or not implemented (raise if used)

| Item | Code |
| --- | --- |
| Baseline style ρ | `BASELINE_NONE = 'none (not implemented; identity carried by committed history only)'`; `'The baseline style rho is not implemented in R2; the baseline slot admits only None'` (`sampling.py:32-37`); `baseline_style='none (not implemented)'` (`operating_point.py:35-36`) |
| Strength | `'strength levels above the default are not built'` (`request_set.py:113-114`) |
| Style directives | `'Style directives are not built in R2 (stage 4); the style field must be None'` (`request_set.py:91-92`); "stage-4 style attributes are not framed" (`features.py:28`) |
| Gameplay demand | `'Gameplay-demand requests have no interface; the demand field must be None'` (`request_set.py:93-94`) |
| Priority, transitions (η) | `request_set.py:97-100` |
| Lead-in presence | `PRESENCE = ('none', 'anywhere')  # 'lead-in' is reserved and raises` (`features.py:29`, `model.py:57-58`) |
| Token conditioner | refused unless `token_lead_in` (`model.py:54-56`); the trainer cannot set `token_lead_in` (`train_ce.py:229-232`) |
| Range targets, other ν | `request_set.py:109-117` |
| LN own-sample term (stage 3) | not built; receipt records `mu_ln=0.0` (`train_ce.py:575`) |

### 10.2 Switches whose default is the plan's provisional choice

| Key | Default / config value | Code comment |
| --- | --- | --- |
| `rule_l` | True | "False: test power checks and v1 reproduction only" (`model.py:45`); "False only for pilot overhead measurement and power checks" (`train_ce.py:79`) |
| `birth_role` | False | "True: v1's birth-role frame; test power checks and v1 reproduction only" (`model.py:46`) |
| `presence` | 'none' | "'anywhere': v1's presence bit; test power checks and v1 reproduction only" (`model.py:47`) |
| `star_value` | 'absolute' in both configs | "'residual': difficulty frame value is target - b(S)" (`train_ce.py:80`); stage-2 arm |
| `mu_star`, `proxy_every` 4, `f3_every` 16, `f3_samples` 4 | 0 (off) | "stage-2 relaxed-proxy term (residual difficulty only)" (`train_ce.py:85-88`) |
| `n_bar`, `n_bar_ln`, `n_bar_star`, `n_bar_key` | **null in both v2 configs** | "from draw_sim, with the matching n_bar_key" (`train_ce.py:89-92`); the trainer refuses to start until `draw_sim` is run and these are filled (`train_ce.py:315-317`) |
| `lr`, `lr_min`, `lr_schedule` | 3e-4, 3e-5, null in both v2 configs | phases optional (`train_ce.py:16-19`). The plan amendment assigns the schedule to Astra (Q-F); the v1 overnight run used lr 1e-3 (`artifacts/r2-runs/r2-ce-overnight-20261004/config.json`). |
| `total_exposures` | 50,000,000 (configs); class default 8,000,000 "set from the pilot's throughput" (`train_ce.py:93`) | |
| `DrawConfig` defaults | `per_window`, p_align 0.6, lead ≤ 32, informative weight 3, z ≥ 2, n ≥ 20, per-window max 3, p_consecutive 0.5, p_whole 0.10, p_long 0.25, runs 2-8, dropout 0.20/0.25/0.20 | `conditions.py:48-71` ("all parameters in DrawConfig", `conditions.py:10`) |
| `star_conditions` | 'auto' | "auto: on iff the v2 label file is complete" (`train_ce.py:81`) |
| Request-flag thresholds | residual flag 1.5 star; LN trained 8-512 beats; star trained 30/60/120 s | `request_set.py:29-31` |
| `NO_SWAP_TRIP` | swap-growth trip disabled | "measures other processes; disabled (plan Q-D)" (`train_ce.py:60`) |
| Model capacity | hidden 128, levels 8, rank 16, stride 64 | fixed by config; the plan defers capacity re-tune and a landmark ablation |

### 10.3 Defects and inconsistencies found

1. **Token conditioner path raises `NameError`.** `features.tokens` calls `_norm_value(iv)` (`features.py:392`), which v2 removed from `features.py` (it existed at `7d9640a`, replaced by `encode_value`, `features.py:320-325`). Confirmed on the mac: `tests/r2/test_conditions.py::test_film_frames_hide_future_values_tokens_show_them`, `tests/r2/test_conditions.py::test_condition_counts_use_committed_heads_only` and `tests/r2/test_locality.py::test_t_p3b_token_conditioner_breaks_locality` fail with `NameError: name '_norm_value' is not defined` (3 failed, 38 passed in those two files plus the free-run hand-swap test). The default FiLM path does not call `tokens()`.
2. `README.md` and `DEVIATIONS.md` describe v1 and were not updated: label file `labels/star.json.gz`, `train.jsonl` in the run directory (now `logs/train-<segment>.jsonl`, `train_ce.py:343-353`), "at most five times" restarts, the presence bit and birth role (DEVIATIONS item 5). `properties.py:54` says the mirror behaviour of difficulty is documented in "DEVIATIONS.md"; that file has no such entry.
3. The DPO anchor CE is still the v1 window mean (`train_dpo.py:190-191`), not `L_base`; plan stage-0 item 0.10 asks for the anchor on `L_base`. DPO builds its anchor corpus with the default `DrawConfig` (the A1 aligned draw) and without a baseline (`train_dpo.py:508`, `train_dpo.py:529-531`) [inferred consequence: anchors of a residual-mode checkpoint would get absolute star values].
4. In-run generation records are built with `train_config=None` (`evaluate.py:371`), so their operating-point hash has `recipe: None`, while `select.py` hashes the run's config (`select.py:126`) [inferred: the two hashes differ for the same checkpoint].
5. Plan stage-0 item 0.22 (a tail-share measurement script) was not found as a separate script; the tail measurement exists only inside the residual-star panel (`evaluate.py:530-540`).
6. `generate.main` takes a fit_dev cache chart by sha (`generate.py:160-161`); there is no CLI for an arbitrary new skeleton, although `generate()` itself accepts any H, grid and T.

---

## 11. Diagram material

### (a) Generation data flow

Nodes:
- N1 `head_ms` H [K] (ms)
- N2 BPM segments → `GridArrays` (canonical beats)
- N3 song length T (ms)
- N4 chart seed: prefix decisions [s,4] + gap [s,4]
- N5 requests (scope, target) + addition boundary g_u
- N6 random seed
- N7 `check_schema` / `RequestSet.add` (validation)
- N8 `check_frontier`
- N9 `effective_track` (+ b(S) in residual mode) → track, provenance, flags, co-active
- N10 `continue_chart` per-decision loop
- N11 TCN online cache + landmarks
- N12 `R2Model` (hands, joint head, pointer)
- N13 rule L (V_k)
- N14 actions [n,4] + gap [n,4]
- N15 `prefix_objects` (objects, open holds)
- N16 `properties` readouts (ν)
- N17 `locality.masked` per target
- N18 generation record (JSON)
- N19 replay → rows → hit objects
- N20 legality check
- N21 `.osu` export
- N22 operating-point hash (recipe, selection rule, decoding, conditioning)

Edges: N5→N7; N7→N8; N4→N8; N8→N9; N1→N9; N2→N9; N9→N10 (track); N1→N10; N2→N10; N3→N10; N4→N10; N6→N10; N10↔N11; N10→N12; N13→N12; N9→N13; N12→N10 (log-probs → Gumbel samples); N10→N14; N14→N15; N15→N16; N9→N16 (targets); N14→N17; N16→N18; N17→N18; N9→N18; N22→N18; N4→N18 (chart-seed record); N14→N19; N19→N20; N20→N21 (only if legal and complete).

### (b) Model forward pass (one window, m decisions, N history tokens, F factors, P candidate pairs)

Nodes and shapes:
- F1 history tokens [N,2,100]
- F2 Linear in [N,2,128]
- F3 8 causal gated conv blocks (dilations 1…128) → enc [N,2,128]
- F4 before = enc[k-1] or BOS [m,2,128]
- F5 landmarks enc[0::64] [L,2,128]
- F6 query features [m,2,150]
- F7 exact MLP [m,2,128]
- F8 concat [m,2,256] → fuse MLP → h [m,2,128]
- F9 landmark attention (per hand, positions < k) → h + read [m,2,128]
- F10 rule L → V_k → row frames [m, 2 roles × 2 kinds × 17 = 68]
- F11 FiLM (γ, δ from 68→128→256) → z [m,2,128]
- F12 joint head: unaries [m,2,25], coupling [m,16,16] → logits [m,625]
- F13 legality mask [m,625] → masked log-softmax → log P(A) [m]; governed split [m]
- F14 factor context concat [F,307] → pointer_in → base [F,128]
- F15 relations [P,28] → pointer_relation [P,128]
- F16 q = pointer_out(GELU(base[owner] + rel)) [P,128]
- F17 pair frames (row at t_k, candidate at u) [P,68] → FiLM → q' [P,128]
- F18 candidate features [P,34] → candidate MLP [P,128]; bias [P]
- F19 score = q'·c/√128 + bias [P]
- F20 per-factor log-softmax → factor log-probs [F]
- F21 sum per (decision, orientation) [m,2] → logsumexp - log 2 → log Q [m]
- F22 total = log P(A) + log Q [m]

Edges: F1→F2→F3; F3→F4; F3→F5; F6→F7; F4→F8; F7→F8; F8→F9; F5→F9; F9→F11; F10→F11; F11→F12→F13; F11→F14 (hand vectors, swapped for mirrored factors); F14→F16; F15→F16; F16→F17; F10→F17 (same FiLM weights); F17→F19; F18→F19; F19→F20→F21; F13→F22; F21→F22.

### (c) Training step (one optimizer step)

Nodes:
- T1 draw RNG (seed 471) → song group → chart
- T2 condition candidates (LN pieces/runs/whole; star cells/whole)
- T3 dropout (all 0.20, kind 0.25, interval 0.20)
- T4 onsets under rule L (source occupancy)
- T5 start j (aligned with p 0.6 or v1 rule) → window [j, j+256)
- T6 interval selection → track; IPW weight u
- T7 `R2Model.window` (teacher forcing) → `WindowOut`
- T8 `window_terms`: base Σ w ℓ, LN governed Σ, star Σ
- T9 `window_loss` with fixed N̄, N̄_ln, N̄_star (from `draw_sim`)
- T10 (stage 2 only, μ_star > 0, 1 window in 4) relaxed-proxy term
- T11 backward per window
- T12 repeat for 4 windows
- T13 clip grad norm 1.0 → AdamW step; LR from exposures
- T14 exposures += head decisions
- T15 every 2,000 exposures: log; every 4,388,000: checkpoint + teacher-forced eval; every 2nd checkpoint: free-run panels
- T16 non-finite → reload, skip, halve LR (3 events → exit 3); resource guard → exit 4

Edges: T1→T2→T3→T4→T5→T6→T7→T8→T9→T11; T10→T11; T11→T12→T13→T14→T15; T7→T16; T13→T16.

### (d) One decision step (generation, decision k)

1. Build the chart with decisions 0..k-1; derive held lanes, LN starts, clocks.
2. Read the TCN cache (state after decision k-1, or BOS) and the landmarks at positions < k.
3. Compute query features at t_k (lane clocks, gap, remaining, phase, BPM, density, 16-gap lookahead).
4. Rule L: V_k from the effective track, the skeleton and the held lanes.
5. Build row frames at t_k for V_k's intervals active at t_k.
6. Model hands: fuse → landmark read → FiLM → z [1,2,128].
7. Joint head → 625 logits; apply the legality mask (EOS: one action).
8. Gumbel-max sample → codes (a0, a1, a2, a3).
9. Coin flip → orientation; list the lanes closing in gap (t_{k-1}, t_k) in that order.
10. Placements: held lanes with code 1 are placed at t_k.
11. For each closing lane: candidates of the gap; pointer scores with relations to placements and FiLM frames at each candidate time; Gumbel-max pick u; add u to placements.
12. Commit codes and release times (replay-checked later in export).
13. If `close_scope` and k >= exit decision and no scope-headed LN is held: stop.
14. If k < K: history token k → TCN cache append; landmark when k % 64 == 0.

---

## 12. Could not determine

- Whether the full `tests/r2` suite passes: only `test_conditions.py`, `test_locality.py` and one free-run test were run (3 failures, all the `NameError` above).
- No v2 training run, `draw_sim` result, relabel or baseline fit was looked for; the configs' N̄ values are null, so none of these has been fed into a config.
- The v1 parameter count (2,408,308) is arithmetic from the `7d9640a` code, not a run.
- The mac jobs imported the package with bytecode writes on in the first two jobs, which may have refreshed ignored `__pycache__` files on the mac; no tracked or untracked source file was touched.
