# R2 system bake-off: proposed plan (2026-10-07, for human review)

Main-thread synthesis of the design round. Sources:
- [r2-design-fable-20261007](r2-design-fable-20261007.md) and [r2-design-opus-20261007](r2-design-opus-20261007.md), each with a network-family addendum;
- the diagnosis [r2-collapse-synthesis-20261007](r2-collapse-synthesis-20261007.md);
- phase 0 so far: [o-x2-horizon](r2-collapse-20261007.md#o-x2-horizon), [r-x0-ready](r2-collapse-20261007.md#r-x0-ready). X1 and X3 are pending.

It answers the human's request ([d-system-comparison](r2-collapse-20261007.md#d-system-comparison)): tie the defects to architecture choices, and compare complete alternative systems on progress toward the working system. **Everything here is a proposal. Nothing is implemented or launched.**

<a id="s-arch-constraints"></a>
## Architecture choices and what they rule out (both design reports agree)

The causal chain, in Fable's words, which Opus independently supports:
- No chart-level input means the start regresses to the corpus mean (F1).
- The TCN and its row-local tokens give nowhere to keep chart identity (F2).
- Teacher-forced CE never penalises losing it, and will not teach a new anchor unless training forces it.
- Single-sample ancestral decoding never corrects an excursion.
- Selection on short-chart panels and NLL never notices.

| Choice | What it intrinsically rules out | Defect | Evidence |
| --- | --- | --- | --- |
| Causal dilated TCN, 511 rows, no memory | A convergent estimate of chart identity is **not representable** (a finite window cannot compute a running mean). The window that matters in practice is 64-128 rows. | F2 | Code; logits are identical when history beyond 511 rows changes; β_cum 0.51 on real prefixes against 0.23 on own |
| Row-local history tokens; cumulative statistics only in FiLM frames | No sufficient statistic of the chart so far on the natural path | F2 | Code |
| No chart-level variable (θ, band, latent) in phase N | Chart identity at the start is the corpus mean given the skeleton; between-chart variety is compressed | F1, compression | Opus D3, D4, D7 |
| Per-decision teacher-forced CE only | No gradient on own-history behaviour; anchoring is worth millinats | F2, absorption | NLL flat while regimes flip |
| Single ancestral sample, T = 1, every draw committed | Nothing rejects an excursion; temperature trades drift against composition | Absorption | Astra's T probe; stay rate 0.53-0.57 against 0.24 |
| Head times fixed | A too-intense regime on a sparse skeleton can only appear as chord and LN pile-up ("full 4 lane") | The form F1 takes in band 2 | Opus D2 |
| Long-note length as a chain of hazard decisions with no chart-level length | No per-chart length style | Regime C for length | `o-lnlen-hintfree` |
| Interval-local FiLM conditions on a frozen phase-N base | A chart-scope request fights a policy that re-estimates θ from its own window | Control defects | Code, C0 screen |
| Selection on NLL plus short (K ≤ 600), 16-chart panels | Cannot see F1 or F2; the panel covers 11% of decisions | Every defect, by omission | Opus D8, census |

**Not constraints:** the training window and start rule, exposure, the row head (probably), and phrase features beyond the 4-bar starts already seen.

<a id="s-network-families"></a>
## Network families (both addenda agree)

- **No encoder family fixes F2 on its own.**
  - The TCN makes a convergent anchor impossible. A GRU, a linear RNN or SSM (LRU/S4), attention, or a two-rate hierarchy makes it possible.
  - None makes it likely without recipe pressure: history dropout, on-policy training, or a held θ.
  - Precedent: R1's all-rows GRU and R2 v1's landmark attention were representable anchors and went unused.
- **Latency does not discriminate between families.**
  - The per-step growth (2.5 µs × k) comes from the sampler replaying the prefix (`with_decisions` / `derived()` every step), not from the network. An incremental state makes a step about 1.5 ms flat.
  - Per-step encoder costs: GRU 0.05 ms, linear RNN 0.20, transformer with KV cache 0.38-0.86, TCN 0.93 ms. The budget is ≥ 69 ms per row at p95 band 5.
- **Training cost** for 2,304 tokens, forward and backward: TCN 294 ms, GRU 201, full causal transformer 1,611 (5.5×).
- **The current window draw re-encodes about 5 positions per scored one.** Scoring whole charts would give an estimated 2-3× more exposures per Mac-hour for any family.
- **Chart-level representation.** Explicit θ comes first: no posterior collapse, and a direct control surface. A VAE/CVAE latent faces near-certain collapse under this objective (the decoder already predicts from the window). A k-means style token, or a residual z with free bits, is the add-on for unnamed style (regime C), only if it persists. About 4k song groups bound any chart-level model.
- **Generation paradigm is orthogonal to F1 and F2.**
  - Block-autoregressive generation is plan-then-realise.
  - Windowed masked refinement fits the mapper's local-edit scenario, but needs a legality redesign and from-scratch training; it is a separate track.
  - Whole-song non-autoregressive generation violates the standing causal constraint.

<a id="p-bakeoff-round1"></a>
## Proposed round 1 (one night, about 6-7.5 Mac-hours)

The arms are nested, so each difference isolates one change. All are warm-started from 56M with the same schedule (`ce_v2_n_lnlevel2_ft`: 200k warm-up, lr 1e-4 cosine to 3e-5) and equal training wall-clock of about 1.25 h each:

| Arm | System | Isolates |
| --- | --- | --- |
| B0 | Phase N 56M, no training | Reference |
| B1 | 56M + plain CE fine-tune | Effect of more fine-tuning (D8: regimes flip between checkpoints) |
| B2 | B1 + self-anchor: about 16 cumulative prefix statistics through a zero-initialised reader, plus history dropout (the encoded history truncated to 64-128 rows on 25-50% of windows) | B2 − B1: does a convergent self-statistic hold identity (F2)? |
| B3 | B2 + explicit chart vector θ (about 10-14 standardised whole-chart coordinates and band), dropped to unknown 30% of the time; at generation θ is drawn once from a skeleton-matched donor prior | B3 − B2: does a held, drawn chart identity fix the start (F1) and restore variety? Every planned control is a θ coordinate |
| B4 (optional) | B1 + a two-rate encoder: a zero-initialised slow stream (LRU or GRU) over 64-row block summaries, plus history dropout, with no hand-built statistics | B4 − B2: is a learned anchor as good as a hand-built one? The network-family arm. Only if two trainings run in parallel at ≥ 0.8× speed |

The two design reports differ on whether B3 should carry the anchor channels. Opus says yes, Fable says no. Nesting B3 on B2 is the main-thread choice: it keeps the differences interpretable, and B3's θ-unknown mode is a free second self-anchor.

**Enablers before the night (not arms; they change cost, not function):**
- an incremental sampler state, which removes the O(k) replay, roughly halves evaluation time, and is needed for real-time play anyway;
- the evaluation suite as one module (Opus C1-C4, Fable M1-M3, Astra adjacency, with P1-P4 below);
- the long-chart panel.

Whole-chart scoring is deferred to round 2. It would change batch composition and must apply to every arm or none.

**Panel.**
- fit_dev, one chart per song group, 25 groups per band. Band 2 uses K ≥ 1,000, because only 4 band-2 charts have K ≥ 1,500. Bands 3-5 use K ≥ 1,500. X0's items are excluded.
- Two modes: from the song start (BOS) with seeds 954-956, testing F1 and F2 together; and continuation from a real first-third prefix with 1 seed, testing F2 alone.
- B3 is judged in prior mode, the product. Oracle mode is a diagnostic only.

**Metrics and decision rule (Opus's gap closure, with Fable's primaries).**
- P1: identity hold at rows 1k+ in prefix mode.
- P2: the stay rate of degenerate stretches, and `bus4` exits in band 2.
- P3: the between-chart SD ratio for variety.
- P4: first-to-last-third correlation in BOS mode.
- Plus the adjacency gap (F3).
- **Gap closure** per measure and band: g = (m_sys − m_src)/(m_base − m_src). G is the mean.
- **Winner:** the guard-passing arm with the lowest G, at least 0.15 below B1, with a chart-bootstrap 90% CI on the difference excluding 0.
- **Guards:** legal export; NLL with θ unknown ≤ B0 + 0.02 nats; single-head share and 4-gram ratio within the source envelope; end-of-song exits ≤ 1.5× the source's; no head-lock run of 30+ rows.
- Once X0 validates a subset of measures (AUC ≥ 0.75 against the human's marks), only that subset counts.
- The winner then goes to a human blind screen of 8 long charts.

**Early signs (kill a failing arm and give its slot to B4):**
- at 2M exposures: the reader or slow-stream weight norm still near 0;
- at 2M-4M: own-prefix β_cum still ≤ 0.3 (B2);
- at 2M-4M: the obedience slope of realised to requested θ < 0.6 (B3).

**Expected outcomes (inferred) and what each means next:**
- **B3 < B2 < B1:** the chart variable is the lever. Controls become θ coordinates, starting with LN share and length. Round 2: plan-then-realise against B3, and an on-policy term on B3.
- **B2 ≈ B3, both below B1:** holding is the main fix. Keep the anchor and add θ only for controls.
- **B3 fixes variety and the start but not holding; B2 is null:** separate "the reader never learned" from "learned but overridden". Round 2: an on-policy objective on B3, with decode-time block selection as the training-free bound.
- **Nothing beats B1:** CE will not make R2 hold any input. Rollout objectives and decode-time selection become the main route.
- **B1 differs from B0 by more than 0.2 in G:** fine-tuning alone moves the regime. Any margin then needs a second training seed.

<a id="p-bakeoff-round2"></a>
## Round 2 candidates, by outcome (proposed)

- Plan-then-realise: per-4-bar section targets with a small planner.
- An on-policy layer (a DPO or moment term on its own continuations), only after X0 has validated the measures.
- A residual latent z, or a k-means style token, if regime C remains.
- Block reranking with a learned critic (the `selection` node).
- A from-scratch encoder-family comparison: TCN against LRU or a hierarchical encoder, with whole-chart scoring, at equal Mac-hours, about 3.5-4 h each.
- A 625-way head-residual screen, about 20 minutes.
- Separate tracks: windowed masked refinement for local edit (several Mac-nights from scratch). Audio stays out of R2 v2 scope.

**Revised 2026-10-07 before any result:** the gap-closure rule is two-sided and becomes a shortlist followed by a human blind comparison that includes B1; B3 in prefix mode takes θ from the prefix; causal wording above is to be read as corrected in [d-review-amendments](r2-bakeoff-night-20261007.md#d-review-amendments).
