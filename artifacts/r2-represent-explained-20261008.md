# The current representation, measures and evaluation, explained (2026-10-08)

The human asked for a thorough account before pushing forward ([private, local](private/human-inputs/00e97a87-25ca-4fab-a49a-668fbf454be4.md#prompt-6)). The main thread wrote it from the code and the reports.
- **Code (scratch, untracked):**
  - `~/ensomi/.sync/cp/scratch/r2-represent/fable-represent/pilot.py`: `decode`, `Chart`, `rep_a`, `rep_b`, `tokens`, `Encoder`, `train_encoder`, `embed`, `cv_probe`;
  - `~/ensomi/.sync/cp/scratch/r2-represent-step1/worker/represent.py`: the block grouping, `span_feats`, `block_norms`;
  - `null_pairs.py`, `x0_rescore.py` and `labelfree.py` in the same folder.
- **Reports:** [r2-represent-fable-20261008](r2-represent-fable-20261008.md), [r2-represent-step1-20261008](r2-represent-step1-20261008.md).

<a id="x-input"></a>
## 0. What goes in

**Per chart:**
- the head times, the "rows", which are supplied by an existing chart;
- one action per lane per row, which is the arrangement R2 chooses;
- the release times of holds.

`decode` turns the actions into four booleans per row and lane, plus a hold length:
- head: a note starts;
- LN head: a hold starts;
- held: the lane is held coming into this row;
- busy: head or held;
- hold length in ms, from `gap_release_ms`.

A section is a run of rows. The units used:
- a 64-row window at stride 32 (X0, the null);
- a Lens-labelled section (Lens probe);
- the whole chart, capped at 2,048 rows in the label-free test.

<a id="x-representation"></a>
## 1. Representation R: three blocks per section

<a id="x-block1"></a>
### Block 1, LN relations (18 numbers)

How holds sit against taps and against each other:

| Feature | What it counts |
|---|---|
| `held` | Share of lane-rows that are held |
| `ln2` | Rows with two or more lanes held |
| `bus4` | Rows with all four lanes busy |
| `lock` | Longest run of a lane being busy |
| `ln_share` | LN heads per head |
| `tap_held` | Rows with a tap while something is held |
| `ln_stag` | A new LN starts while another is held |
| `rel_head`, `rel_gap`, `rel_repress` | Release kinds: on a head row, inside a gap, release-and-repress |
| `rel_multi` | Simultaneous releases |
| `hold_lmed`, `hold_lcv` | Median and spread of hold length |
| `heldset_chg` | How often the set of held lanes changes |
| `f_ln`, `o_ln`, `s_ln` | LN share by gap class: fast (< 0.75 × the local median gap), on tempo (0.75-1.5×), slow (> 1.5×) |
| `cor_ln_dt` | Correlation of LN heads with log gap |

<a id="x-block2"></a>
### Block 2, pattern grammar with hands (45 numbers)

| Group | Features |
|---|---|
| Hands | Balance between left (lanes 1-2) and right (3-4); hand switch rate; longest one-hand run; rows using both hands; alternation inside each hand; lane entropy |
| Jacks | Full (same chord repeated); partial; longest jack run; jack episodes; minijacks; chordjacks; fast jacks (< 100 ms) |
| Trills | A/B/A alternation of disjoint lane groups; longest trill; cross-hand trills |
| Streams | Single notes changing lane; longest stream; stairs; direction changes; jumpstream; chord-size changes; mirror-symmetric pairs |
| Old window statistics kept | Chord rate (`nh`), ≥3- and 4-note rows, jack rate, pattern entropy, distinct 4-grams, repeat rate, loop period, max lane share, max hand share, longest lane lock |
| Timing-conditional choices | Chord size, jack rate and hand switching in each gap class; correlation of chord size with gap |

Two row properties were dropped from the blocks, log row count and median gap, because the rows fix them.

<a id="x-block3"></a>
### Block 3, the learned embedding (64 numbers): how it is made

1. **Tokens.** Each row becomes 14 numbers:
   - 4 bits for a head per lane;
   - 4 bits for an LN head per lane;
   - 4 bits for held per lane;
   - the log gap to the previous row;
   - the log of that gap relative to the window's median gap, clipped.
2. **Encoder.** Three 1-D convolutions over the rows, kernel 5, with dilations 1, 2 and 4, so each output sees about 29 rows. ReLU after each. Then an average over the rows, a linear layer, and L2 normalisation to a 64-d unit vector.
3. **Training, contrastive (InfoNCE), with no labels.**
   - **Batch:** 256 pairs. For each pair, draw a random chart and a random 32-row window, then a second 32-row window from the same chart within ±96 rows. Each side is lane-mirrored with probability 0.5, so mirror images count as the same.
   - **Score:** all 256 × 256 cosine similarities, divided by a temperature of 0.1, with cross-entropy for picking each window's true partner among the 256. Chance is ln 256 = 5.55.
   - **Optimiser:** Adam, lr 1e-3, trained for a fixed time.
   - **Data and runs:** 1,500 fit_train charts. The pilot trained 420 s (loss to 2.55). Step 1 refit it for 240 s on charts outside the human pairs and X0 (loss 5.30 → 2.75), because the pilot did not save its weights.
4. **Embedding a section.** Slide 32-row windows at stride 16, embed each, and average.
5. **What it learns.** Whatever tells "a nearby window of the same chart" apart from "a window of another chart": the chart's local habits.
   - It has no notion of good or bad, and no Lens or X0 label enters it.
   - **Control:** the same encoder untrained gives Lens F1 0.613; trained, 0.687.
   - **Caveat [I]:** the tokens include the gaps, and a positive pair shares its chart's rhythm. So part of what the encoder learns is the rows themselves: tempo and density.
     - In a paired comparison on the same rows, that part cancels.
     - In the Lens probe, it does not.

<a id="x-measures"></a>
## 2. Measures: candidates, none validated

<a id="x-paired"></a>**Paired with the source on the same rows.** Today the rows come from an existing chart, so a human arrangement of the same rows always exists.
- For each 64-row window, take the generated chart's features minus the source's features on the same time span, Δ.
- **Null units:** divide Δ by the 90th percentile of |Δ| between two human charts on near-identical rows (same band, head F1 ≥ 0.95, copies excluded).
- A flag at Δ > 1 unit means "beyond 90 % of human pairs".

| Candidate | Definition | Serves |
|---|---|---|
| `m-LNx` | One-sided Δ of `ln2` or `held` in null units; flag above 1 | In-place marks: windows |
| Block-2 chart norm | RMS of the per-feature Δ over block 2, at chart level, each feature divided by its human-pair SD | Non-LN complaints |
| Unpaired fingerprint | Fast-jack rate and full-jack rate per chart, with no source | Whole-song impressions |

**Not a quality measure:** the block distance d(arrangement, A), used only to test R (section 3, E3).

<a id="x-evaluation"></a>
## 3. Evaluation: how the representation and the measures are checked

| Check | Question | How | Result | Labels |
|---|---|---|---|---|
| E1, Lens probe | Can a simple classifier on R predict the five concept levels on songs it has not seen? | Logistic regression; 5-fold CV grouped by song group, so no difficulty of a test song is in training; macro F1 | R 0.759; old window statistics 0.677; majority 0.258 | About 96 % Astra labeller, 4 % human |
| E2, X0 | Do the measures flag where the human marked, and not on charts the human passed? | AUC and flag rates in null units, with and without x0-02 | One-sided LN flag: 73 % of marks, 0 % of passed charts; without x0-02, 3 of 14 marks | Human, but used both to choose and to check |
| E3, label-free | Does R tell R2 on A's rows from a second human B on the same rows? | AUC of d(gen, A) against d(B, A), 120 pairs, cluster bootstrap | Block 1 0.80; block 2 0.63, inside R2's seed spread; block 3 0.69 | None |
| E4, proposed | Do the measures agree with the human on new charts? | Blind clip pairs, with measures and thresholds fixed before the human judges | Not run | Human, held out |

<a id="x-patterns"></a>
## 4. Against the earlier bad measurement patterns

| Earlier pattern | What is done now | Still open |
|---|---|---|
| Deciding with measures never shown to see what the human sees | Each candidate is scored against the human's labels (E1, E2) before any use. Nothing selects, ranks, stops or trains on them. | X0 was used both to choose the scores and to check them, and the one-sided direction was chosen after seeing it. Only E4 separates design labels from validation labels. |
| Margins below noise; a selected checkpoint as the only baseline | Every flag is in human-pair null units. R2's seed spread is measured; block 2 sits inside it and is not used to separate. | The null is 92 % one mapper across two difficulties. The different-mapper null is 1.8× wider and rests on 47 pairs. |
| Ignoring fixed inputs (the supplied rows) | Every comparison is against a chart on the same rows, so the density and difficulty the rows fix cancel. Row properties are dropped from the blocks. | The embedding still sees the rows. With rows from audio there is no source chart, and an E[· given rows] reference is not built. |
| The band as the reference | Replaced by the same-rows reference. | — |
| Level statistics blind to organisation | Blocks 1 and 2 encode relations: holds against taps, hands, jack runs, trills. | — |
| Gaming the measure (`d0-phi` stripped LN to lower G) | The LN flag is one-sided. A two-sided "distance from source" would flag the LN-deficit charts the human passed (73 % of their windows). No measure is an objective or a selector yet. | A one-sided flag invites the opposite gaming: an LN-poor generator passes. It needs a second measure for the deficit, or the human's view on it. |
| Mixing targets | In-place marks are measured per window, whole-song impressions per chart; the human confirmed the split. | — |
| Overclaiming | Results are given with and without x0-02. Post-hoc choices and machine labels are stated. | — |

<a id="x-weak"></a>
## 5. Where the approach is weak

- **The X0 window evidence rests on x0-02.** The marks on x0-03, x0-09, x0-11 and x0-12 are not explained.
- **E1 measures agreement with a machine labeller,** not with the human.
- **The embedding** is small, briefly trained, and partly encodes the rows.
- **No measure is validated.**
- **Paired measures need a source chart on the same rows.** They exist only while the rows come from a human chart.
