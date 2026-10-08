# How charts are read, and what the Lens evidence notes offer the representation (2026-10-08)

Worker: perception (Opus, fresh context), brief `~/ensomi/.sync/cp/scratch/r2-perception/brief.md`, delegated per [r-perception](r2-representation-20261008.md#r-perception) at the human's request ([d-perception-first](r2-representation-20261008.md#d-perception-first)). Returned about 07:12 UTC. Measurement and reading only; about 6 minutes of mac time.

- **Full report** (literature with citations, every table, the human-facing questions): `~/ensomi/.sync/cp/scratch/r2-perception/report.md`. The harness refused the worker's own write, so the main thread saved it from the worker's final message, text unchanged.
- **Mirrored outputs**, `artifacts/r2-perception-20261008/`: `tables.md`, `summary.json`, `relative.md`, `relative.json`, `rationale-sample.md`, `human-decisions.md`. The last file holds the human's own Lens decision notes verbatim; it stays local and is not quoted here. Mac only: `claims.parquet`, `rationales.parquet`.
- **Scripts:** `~/ensomi/.sync/cp/scratch/r2-perception/worker/` (`extract.py`, `analyze.py`, `relative.py`, probes).
- **Final jobs:** `20261008-070239-pc-extract2`, `20261008-070536-pc-relative2` (exit 0). Superseded: `-065818-pc-extract`, `-065938-pc-analyze`, `-070438-pc-relative`.

Tags: [S] read in a source, [M] measured, [I] inferred, [P] proposed.

<a id="o-no-human-evidence"></a>
## The pool holds almost no human evidence [M]

- 594 of the 598 human cells are the human's decision on an agent proposal; 4 are direct-human.
- On the 591 cells that also carry a machine proposal, the note refs are identical in 98-100 % of cells per concept and the rationale text in 89.7 %. The pool records that no separate human note selection was made.
- The human's own words exist on 97 cells: 66 distinct short notes, one per section decision.
- **So every evidence span in the pool was chosen by the labeller agent under the frozen Foundation skill.** The evidence shows where that skill looks, not where the human looks.

<a id="o-evidence-shape"></a>
## Machine evidence: selective, localised, at mild boundaries [M]

- **Coverage:** 99.5 % of the 22,361 resolved cells carry note refs (median 18). A note ref is one source hit object (lane, head time, tail time); all 304,505 in-scope refs join a head row and lane in the R2 cache, and 99.4 % of LN refs match a replayed hold end.
- **Not selections:** two batches (corpus-500-original, manual-or-benchmark) reference every note in the scope. The October batches are the most selective (15-18 % of heads, about half the rows).
- **Shape (present claims, median):** jack 5.7 beats, stream 6.9, trill 1.75, tech 7.6, LN 7.3. Trill spans are short and contiguous (85 % one run); tech spans are long and fragmented (16 % one run).
- **Localisation** (mid-rank of the evidence span among all same-length spans of the scope, on the concept's own simple statistic; 0.5 = none): jack .851, trill .966, LN .846, stream .627, tech .599. It holds at matched density. Each concept's evidence also avoids the competing organization (jack evidence −.35 on disjoint alternation; stream evidence −.19 on lane sharing). For jack and trill the statistic is close to the skill's selection rule, so the high values are near-tautological.
- **Boundaries:** spans start on a grid beat 45 % of the time against a 21 % base (+.23 [.22, .25]); the gap before the start ranks .62 among the scope's gaps; rhythm-only novelty at the start .61. End boundaries are weaker. Mild boundaries, chosen by the machine.

<a id="o-episodes"></a>
## Labels read as episodes, not averages [M]

From `relative.md`; V1 has been used by the spaces check, so V1 numbers here describe.

| Concept (statistic) | V1 scope mean | V1 peak over 8 rows | V2 scope mean | V2 peak over 8 rows |
|---|---|---|---|---|
| jack (lane sharing) | .875 | .901 | .872 | .904 |
| stream (directional runs) | .709 | .740 | .827 | .827 |
| trill (A/B/A alternation) | .843 | .943 | .919 | .942 |
| tech (gap-ratio change) | .709 | .796 | .701 | .764 |
| LN (two lanes occupied) | .977 | .979 | .925 | .930 |

- The peak of a concept's statistic inside the scope recognises the label better than the scope mean, most for trill and tech. Intervals are unpaired, and the maximum grows with scope length (most scopes are about 10 s).
- **Chart-relative ranking recognises worse than absolute values for every concept** (LN .977 against .665 on V1). The human's notes judge difficulty and density against the whole chart; the concept labels are absolute. So the chart-relative reading belongs to whole-song intensity, not to pattern presence [I].

<a id="o-reading-model"></a>
## How a chart is read, and what can be modelled [S/I/P]

Sources: the beatmap-lens reading framework (from 139 expert judgments) and decisions 0004 and 0007; grouping and segmentation (GTTM, Frankland and Cohen 2004, LBDM, Pearce et al. 2010 on information-content boundaries); MIR structure analysis (Foote novelty, multi-level segmentation); point-set pattern discovery (SIA, SIATEC, COSIATEC; the MIREX repeated-pattern task); tension and EDM buildup studies; osu!mania ranking criteria and star rating. Citations are in the full report.

- **Nested units.** A chart is read as patterns or cells, then episodes or phrases, then passages.
- **Boundaries** come from proximity (larger gaps), metrical position, change, and the edges of repeats.
- **A unit is a dominant organization plus decorations.** Lens decision 0007 and the human's notes describe sections this way. Examples in the notes, paraphrased: jack-led on a stream skeleton; a lane that plays like a modifier; a section that would be trill alone but sits in a stream context.
- **Intensity is a trajectory** over those units. It splits under supplied rows into the rows' part and the arrangement's part, and R2 owns only the latter.
- **Expectation** (a fourth aspect): the Foundation defines tech as hard to anticipate through familiar patterns; information content under a corpus pattern model gives tech, boundaries and readability.
- **What can be checked against the human today:** only pattern presence, through the Lens labels. Boundaries and intensity can be computed from charts alone, but until the human marks some, they only characterise (guardrail 1).

<a id="p-perception-changes"></a>
## Proposed changes [P]

- **The unit comes first.** Replace fixed 8-beat windows by nested data-driven segments: pattern (1-2 beats), phrase (about 8 beats, fitted), passage (about 32 beats or detected). Boundary strength combines proximity, metrical weight, novelty, repeat edges and information-content peaks, with weights fitted label-free.
- **A pattern space (S9):** instance extractors (jack core, fixed A/B alternation, directional roll, LN interaction) that grow while their relation holds, plus a lexicon learned by compression. S3 becomes its first layer.
- **Whole song:** an ordered trajectory of segments carrying the rows' intensity, the arrangement's intensity given the rows, and the pattern mix.
- **Spaces fixes:**
  - Defer residualising S5 and S8 on S3. Within an episode, repetition, contour and alternation are competing readings of the same rows, and regression across sections would remove part of what is read as stream. Separate them by assigning each instance to its organization instead. Limiting S8 to absolute interval quantiles is harmless.
  - Keep S6's stable shrinkage; it matters more once units vary in length.
  - Keep S4's residualisation on held mass, though it will not move a human-facing result soon.
- **For the VQ-VAE refinement:**
  - Unit: segments, then instances, then a hierarchical 1-2-beat code with a phrase level.
  - Invariances: time translation (repeats share a code), decoration (one extra key outside the core moves the code little).
  - Objective: keep reconstruction given the rows and add a contrastive term between a chart's own repeats.
  - Evidence notes: agreement-with-labeller localisation against the simple-statistic scale, and a D-only supervised variant.

<a id="v-perception-checks"></a>
## Main-thread checks against the mirrored outputs (2026-10-08)

- `tables.md`: human cells by origin 167 / 23 / 404 agent-proposal (high, low, none) and 4 direct-human. The report says 166 and 405, an off-by-one with no effect. Identical refs .98-1.00 per concept; rationale identical on .897 of 591 cells; localisation rows for jack .851, trill .966, stream .627, tech .599 and LN .846; machine start boundaries on beat +.245 for jack and +.288 for trill. All match the report.
- `relative.md`: every scope-mean, peak and chart-relative AUC quoted above matches.
- Not checked: the shape table, the specificity matrix, the literature citations.

<a id="s-perception-reading"></a>
## Reading (main thread)

1. **The evidence notes cannot tell us where the human looks.** No human evidence span exists; human cells copy the labeller's spans. They tell us where the frozen skill looks, which is useful as a localisation target (agreement with the labeller), as examples of pattern instances, and as D-only supervision. Learning where the human looks needs new human selections.
2. **The strongest finding for the representation is the unit.** Labels attach to episodes inside a scope, not to section averages, and both the evidence spans and the literature point to nested segments with mild, measurable boundaries. This supports the human's earlier "fixed windows or something else" question with a direction: data-driven segments.
3. **Simple statistics set a bar.** One-line counts taken from the Foundation definitions recognise the human cells about as well as S3 does in absolute AUC (V2 jack .872, trill .919, LN .925, against S3's .900 and .888 for jack and trill). These counts are absolute, not increments over mass, so the comparison is indicative. A representation that cannot beat a one-line count per concept is not yet seeing more than the definition.
4. **Whole-song intensity is a separate target from pattern presence.** The human judges density and difficulty against the whole chart; the concept labels are absolute.
5. **The worker's advice to defer one approved fix contradicts the human's approval** and is put to the human.
