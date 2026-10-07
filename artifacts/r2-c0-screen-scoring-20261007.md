# R2 phase-C C0 blind screen: scoring against ras-v1

Report of a fresh Opus subagent (main session 022ac2c4, 2026-10-07 about 14:20 UTC), scoring the blind screen of [o-c0-screen](r2-phasec-c0-20261007.md#o-c0-screen) under the plan in [p-c0-judging](r2-phasec-c0-20261007.md#p-c0-judging). Copied here verbatim between the markers below. The harness refused the subagent's own write of `report.md`, so the report exists only here. Its scripts and data are on the control plane in `~/ensomi/.sync/cp/scratch/r2-c0-screen/`: `score.py`, `tables.py`, `scores.json`, `notes-coded.csv`, and `mac-copies/` (fresh `server.log` and `index.html`). Sheets and key: `artifacts/r2-phasec-c0-20261007/screen/` on bings-mac (`human-judgments.csv`, `judge-sheet.csv`, `KEY-sealed.json`). Mac job used: `20261007-135957-screen-blind-grep` (one read-only grep of Codex rollouts).

Main-thread check: the per-category agree / disagree / same counts for both judges, recomputed from the raw sheets and `KEY-sealed.json`, match section 1 exactly. The note-count confound (the higher-r side has more heads in 16/16 fixed, 16/16 natural_gap and 7/8 near_equal pairs with 1 tie) also matches. Not checked by the main thread: the note coding (the subagent's judgment, laid out in section 5), the hold and jack statistics, the blindness grep, and the human pace table. Summary and anchors: [o-c0-screen-scored](r2-phasec-c0-20261007.md#o-c0-screen-scored).

----- BEGIN report -----

# R2 phase-C C0 blind screen: scoring against ras-v1

Scored 2026-10-07 on the control plane, read-only on the replica `ensomi-model/artifacts/r2-phasec-c0-20261007/screen/`.

Files beside this report:
- `score.py` does the scoring (stdlib only).
- `tables.py` prints the tables below.
- `scores.json` holds all numbers and the per-pair rows.
- `notes-coded.csv` has 96 rows, one per judge and pair.
- `mac-copies/` has fresh copies of the review `server.log` and `index.html`, taken from the mac with `ens cat` because the local `server.log` replica is stale.

## Summary

- [measured] **Gaps above 0.25:** both judges agree with the direction of r on every pair with |Δlog r| > 0.25, 32 of 32.
  - That covers all 16 tilt pairs, all 8 constructed pairs and 9 natural pairs.
  - Pooled over all 48 pairs, the human agrees on 41 of 43 non-"same" answers (sign test p = 2e-10) and Astra on 39 of 42 (p = 6e-9).
- [measured] **Natural pairs (the only in-range test):** on the 16 phase-N pairs (|Δlog r| 0.175 to 0.459), the human agrees 15 of 16 and Astra 14 of 14, plus 2 "same".
  - Clustered by source chart, 8 of 8 charts agree for both judges (p = 0.0078).
  - Evidence strength is moderate: 16 pairs on 8 charts, with shared samples.
- [measured] **Near-equal controls** (|Δlog r| ≤ 0.03): the human says "same" 5 of 8 times and Astra 4 of 8; the other answers do not follow r.
- [measured] **Disagreements:** all 5 (human 2, Astra 3) are low-confidence. Every disagreement or "same" answer sits at |Δlog r| ≤ 0.2.
  - The human's two cite attack patterns: a minijack, and a long jack against a chordjack.
  - Astra's three are all near-equal pairs: 2 name holds as the main reason, 1 a jack-against-reading tradeoff.
- [measured] **Unplayable calls:** the human called 6 sides unplayable and 1 collapsed. All 7 are the higher-r side: 5 positive-tilt sides and 2 constructed one-lane jacks.
  - Unplayable sides therefore always produced agreement with r, never disagreement.
  - 14 of 16 positive-tilt sides have r above the largest natural 16 s r (2.81).
- [measured] **Confound:** in 43 of 48 pairs the higher-r side also has more notes, so the screen cannot tell r apart from a plain note count.
- [measured] **Blindness:** the six Astra judge transcripts contain no `KEY-sealed`, no key field names and no `private/candidates`. The human's browser fetched no Astra file.

## 1. Agreement with the direction of r

- "agree" means the judge's harder side is the higher-r side.
- p values are exact two-sided sign tests against 0.5, on the non-"same" answers.
- Pairs are not independent, so two clustered versions are also given, both by source chart (every reused sample stays within one chart):
  - a majority vote per chart, with ties and all-"same" charts dropped;
  - an exact sign-flip test on the per-chart (agree − disagree) totals.

[measured]

| Judge | Category | n | agree | disagree | same | agree / non-same | sign-test p | charts | chart majority agree:disagree (p) | cluster sign-flip p |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| human | fixed (tilt) | 16 | 16 | 0 | 0 | 16/16 | 3.1e-05 | 11 | 11:0 (0.00098) | 0.00098 |
| human | natural_gap | 16 | 15 | 1 | 0 | 15/16 | 0.00052 | 8 | 8:0 (0.0078) | 0.0078 |
| human | mechanical | 8 | 8 | 0 | 0 | 8/8 | 0.0078 | 8 | 8:0 (0.0078) | 0.0078 |
| human | near_equal | 8 | 2 | 1 | 5 | 2/3 | 1.000 | 8 | 2:1 (1.000) | 1.000 |
| human | pooled | 48 | 41 | 2 | 5 | 41/43 | 2.2e-10 | 17 | 16:0 (3.1e-05) | 3.1e-05 |
| human | natural_gap + near_equal | 24 | 17 | 2 | 5 | 17/19 | 0.00073 | 13 | 9:1 (0.021) | 0.016 |
| astra | fixed (tilt) | 16 | 16 | 0 | 0 | 16/16 | 3.1e-05 | 11 | 11:0 (0.00098) | 0.00098 |
| astra | natural_gap | 16 | 14 | 0 | 2 | 14/14 | 0.00012 | 8 | 8:0 (0.0078) | 0.0078 |
| astra | mechanical | 8 | 8 | 0 | 0 | 8/8 | 0.0078 | 8 | 8:0 (0.0078) | 0.0078 |
| astra | near_equal | 8 | 1 | 3 | 4 | 1/4 | 0.625 | 8 | 1:3 (0.625) | 0.625 |
| astra | pooled | 48 | 39 | 3 | 6 | 39/42 | 5.6e-09 | 17 | 15:1 (0.00052) | 0.00015 |
| astra | natural_gap + near_equal | 24 | 15 | 3 | 6 | 15/18 | 0.0075 | 13 | 7:2 (0.180) | 0.070 |

By confidence, as agree/disagree/same [measured]:

| Judge | Category | high | low |
|---|---|---|---|
| human | fixed | 16/0/0 | - |
| human | natural_gap | 14/0/0 | 1/1/0 |
| human | mechanical | 8/0/0 | - |
| human | near_equal | 0/0/2 | 2/1/3 |
| human | pooled | 38/0/2 | 3/2/3 |
| astra | fixed | 16/0/0 | - |
| astra | natural_gap | 10/0/0 | 4/0/2 |
| astra | mechanical | 8/0/0 | - |
| astra | near_equal | - | 1/3/4 |
| astra | pooled | 34/0/0 | 5/3/6 |

- [measured] No high-confidence answer from either judge contradicts r.
- [measured] Neither judge favours a side. The key has A higher in 26 pairs; the human answered A 26, B 17, same 5; Astra answered A 24, B 18, same 6.
- [measured] The fixed and mechanical categories only test gross direction:
  - Low-tilt sides sit at r 1.00 to 1.11, around or below the natural 16 s p5 of 1.105.
  - 14 of 16 high-tilt sides are above 2.8.
  - The constructed contrasts are singles against all-quad walls or against one-lane jacks.

## 2. Near-equal controls

[measured] How the extra columns were computed:
- Hold ms per s and releases come from the beatmap-lens parse of the exported `.osu`, inside the marked window.
- "Longest jack" is the longest run of consecutive head rows that all hit one column.

| Pair | chart | higher r | abs Δlog r | human | astra | notes A/B | hold ms per s A/B | releases A/B | longest jack A/B |
|---|---|---|---:|---|---|---|---|---|---|
| 001 | chart-004 | B | 0.009 | A/low (disagree) | A/low (disagree) | 61/62 | 329/146 | 8/5 | 2/2 |
| 004 | chart-013 | B | 0.022 | same/low | A/low (disagree) | 259/266 | 0/0 | 0/0 | 9/5 |
| 006 | chart-011 | B | 0.014 | B/low (agree) | same/low | 54/55 | 98/98 | 3/3 | 2/2 |
| 030 | chart-010 | A | 0.024 | A/low (agree) | same/low | 111/109 | 0/38 | 0/2 | 2/2 |
| 040 | chart-017 | A | 0.011 | same/high | B/low (disagree) | 173/171 | 84/312 | 9/23 | 2/2 |
| 042 | chart-001 | A | 0.001 | same/low | same/low | 111/111 | 227/262 | 10/13 | 2/2 |
| 046 | chart-003 | B | 0.018 | same/high | same/low | 341/348 | 0/0 | 0/0 | 1/1 |
| 048 | chart-006 | B | 0.029 | same/low | B/low (agree) | 248/258 | 77/24 | 6/2 | 2/3 |

- Human:
  - "same" on 5 of 8 (004, 040, 042, 046, 048);
  - otherwise 001 goes against r and 006 and 030 go with it.
- Astra:
  - "same" on 4 of 8 (006, 030, 042, 046);
  - otherwise 001, 004 and 040 go against r and 048 goes with it.
- [measured] Why Astra left "same" in its three disagreements:
  - In 001 and 040 it picked the side with more hold time (2.2x and 3.7x hold ms per s) and its note gives holds as the reason.
  - In 004 it picked the side with the longer jack, 9 rows against 5.
- [measured] In 001 the human also picked the hold-heavier side, but its note cites a "minijack spike".
- [measured] All Astra near-equal answers are low confidence. The human marked 040 and 046 "same/high".

## 3. Inter-judge agreement

Cohen's kappa over the labels {A, B, same} [measured]:

| Subset | n | raw agreement | Cohen kappa | pairs where labels differ (human, astra) |
|---|---:|---:|---:|---|
| pooled | 48 | 0.854 | 0.75 | 004 (same,A), 006 (B,same), 012 (A,same), 016 (A,same), 030 (A,same), 040 (same,B), 048 (same,B) |
| fixed | 16 | 1.000 | 1.00 | - |
| natural_gap | 16 | 0.875 | 0.78 | 012 (A,same), 016 (A,same) |
| mechanical | 8 | 1.000 | 1.00 | - |
| near_equal | 8 | 0.375 | -0.05 | 004 (same,A), 006 (B,same), 030 (A,same), 040 (same,B), 048 (same,B) |

- [measured] The judges never pick opposite sides; every difference is one of them saying "same".
- [measured] The pooled kappa comes from the easy categories. On the near-equal controls, agreement is at chance.

## 4. Agreement against the size of the r gap

By |Δlog r| band, as agree/disagree/same [measured]:

| abs Δlog r band | n | categories | human | astra |
|---|---:|---|---|---|
| ≤ 0.03 | 8 | near_equal 8 | 2/1/5 | 1/3/4 |
| 0.03 to 0.25 | 8 | natural_gap 7, mechanical 1 | 7/1/0 | 6/0/2 |
| 0.25 to 0.5 | 12 | natural_gap 9, mechanical 3 | 12/0/0 | 12/0/0 |
| > 0.5 | 20 | fixed 16, mechanical 4 | 20/0/0 | 20/0/0 |

- [measured] Terciles of the 48 pairs give the same picture:
  - lowest third (0.001 to 0.239): human 9/2/5, Astra 7/3/6;
  - middle third (0.250 to 1.108) and top third (1.131 to 1.371): 16/0/0 for both judges.
- [measured] Logistic fit of agree (against disagree or same) on |Δlog r|:

  | Judge | all 48 pairs: slope (Wald SE) | natural_gap + near_equal: slope (Wald SE) |
  |---|---:|---:|
  | human | 19.4 (6.9) | 18.2 (7.0) |
  | astra | 20.9 (7.1) | 19.4 (7.0) |

- The slope mostly reflects the step between the near-equal block and everything else, and gap is confounded with category. Read the band table rather than the slope.
- [measured] Inside natural_gap, every non-agreement from either judge sits at 0.184 or 0.199, the second and third smallest gaps in that category. Both judges agreed on the smallest gap (0.175).

## 5. What the notes cite

### Coding scheme

The codes are mine, and a note can carry several. The full 96-row table with complete note text is `notes-coded.csv`.

| Code | Values |
|---|---|
| attacks | jacks or same-lane repeats, chords, density or streams, trills, hand balance |
| holds | primary (main reason for the call); secondary (an added reason on the chosen side); counterweight (holds on the other side, weighed against the call); tradeoff (part of a "same" call); absent (the note says there are no holds); none |
| reading | driver; counterweight; neutral (named, not decisive); none |
| unplayable | a literal claim that a side cannot be played |
| other | degenerate; quality; unclear; denies_unplayable (Astra says no impossible overlap is visible); extreme |

"Holds cited" in the split below means primary, secondary, counterweight, tradeoff, or secondary+counterweight.

### Human (13 notes; 35 pairs have none)

| Pair | category | higher r | call | outcome | attacks (tags) | holds | reading | unplayable (side) | other | note |
|---|---|---|---|---|---|---|---|---|---|---|
| 001 | near_equal | B | A/low | disagree | 1 (jack: minijack spike) | none | none | 0 | quality (both good charts) | they're both good charts. maybe minijack spike makes A seem harder. |
| 002 | mechanical | B | B/high | agree | 0 | none | none | 1 (B) | | B is totally unplayable! |
| 003 | fixed | A | A/high | agree | 0 | none | none | 1 (A) | | A is not playable at all lol |
| 004 | near_equal | B | same/low | same | 0 | none | none | 0 | unclear | hard to judge |
| 005 | fixed | A | A/high | agree | 0 | none | none | 1 (A) | degenerate ("again") | The pattern degration occur again. A is not playable. |
| 006 | near_equal | B | B/low | agree | 1 (hand balance) | none | none | 0 | "balanced" coded as hand balance; could be readability | A is slightly more balanced. |
| 007 | natural_gap | B | B/high | agree | 1 (density, spike) | none | driver | 0 | | A is more easy to follow and the difficulty spike is smaller and less dense. |
| 009 | fixed | A | A/high | agree | 1 (chord: full 4-lane) | none | none | 1 (A) | | Obviously A is not playable with full 4-lane consequent attacks. |
| 010 | mechanical | B | B/high | agree | 0 | none | none | 1 (B, implied) | side not named | Not playable pattern. |
| 012 | natural_gap | A | A/high | agree | 1 (chord + jack, coordination) | none | none | 0 | | A has longer 3-chord jack mix and involve complicated finger control so I think it is harder. |
| 016 | natural_gap | B | A/low | disagree | 1 (long jack + coordination vs 3-1-3 chordjack) | none | none | 0 | | A is long jack with extra coordination demand while B's chordjack contains many 3-1-3s, so I think A is harder. |
| 025 | fixed | A | A/high | agree | 0 | none | none | 1 (A) | | A not playable lol. |
| 028 | fixed | A | A/high | agree | 1 (chord: collapse to 4 lanes) | none | none | 0 | degenerate (map collapse), side A implied | map collapse to full 4 lane. |

- Human totals [measured on my coding]: attacks 7, holds 0, reading 1, unplayable 6, degenerate 2 (005 is also unplayable), unclear 1.
- The human never mentions holds or releases.

### Astra (48 notes)

| Pair | category | higher r | call | outcome | attack tags | holds | reading | other | note (first 110 characters) |
|---|---|---|---|---|---|---|---|---|---|
| 001 | near_equal | B | A/low | disagree | chord (B repeated right-hand chords, counterweight) | primary | none |  | A has more sustained holds crossing other-lane taps and more independent release changes, especially late; B t... |
| 002 | mechanical | B | B/high | agree | jack (lane-1 bursts), recovery | none | none | denies_unplayable | B concentrates the flowing single notes into rapid lane-1 jack bursts with little finger recovery; A distribut... |
| 003 | fixed | A | A/high | agree | chord+jack (four-key chordjacks) | none | none | denies_unplayable, extreme | A sustains four-key chordjacks through nearly the whole window, including tighter bursts; B mainly alternates ... |
| 004 | near_equal | B | A/low | disagree | jack (long jack runs) vs chord shapes | none | counterweight |  | A more often sustains one finger through long jack runs while other chord members change, particularly in the ... |
| 005 | fixed | A | A/high | agree | chord+jack (full-width chordjacks), density | secondary | none | denies_unplayable, extreme | A builds into relentless full-width chordjacks, tight reattacks and short holds amid continuing chords; B is a... |
| 006 | near_equal | B | same/low | same | chord (doubles, chord shapes) | tradeoff | neutral |  | A repeats right-hand doubles and has early holds; B changes chord shapes more and places holds later among tap... |
| 007 | natural_gap | B | B/high | agree | chord+jack, recovery | secondary | none |  | B sustains substantially heavier changing doubles and triples, with repeated fingers and short hold-release tr... |
| 008 | fixed | B | B/high | agree | chord (four-key) | secondary | none |  | B combines recurring four-key chords with multi-finger holds and staggered releases or reattacks, while A most... |
| 009 | fixed | A | A/high | agree | chord+jack (full-chord jacks) | secondary | none |  | A has sustained full-chord jacks and extremely tight repeated chord presses with short LN releases near +2s; B... |
| 010 | mechanical | B | B/high | agree | jack (lane-1 jack) | absent | none |  | B concentrates the full rhythm into a sustained lane-1 jack, including the tighter early burst; A distributes ... |
| 011 | natural_gap | B | B/low | agree | chord+jack (chord reattacks) | primary | none | holds also as counterweight | B repeats paired short LNs, including adjacent-finger chord reattacks near +3s, and has greater chord press/re... |
| 012 | natural_gap | A | same/low | same | chord vs jack (lane-1 runs) tradeoff | none | none |  | A carries more recurring two- and three-note chords; B has longer uninterrupted narrow jacks, especially lane ... |
| 013 | natural_gap | A | A/high | agree | chord+jack (repeated full chords) | none | none |  | A sustains chord-plus-single flow and finishes with repeated full chords interleaved with pairs; B is predomin... |
| 014 | fixed | A | A/high | agree | chord+jack, density | secondary | none |  | A repeatedly presses nearly every lane, mixing full chords with short LN releases and immediate reattacks thro... |
| 015 | mechanical | B | B/high | agree | jack (lane-1 jack), density | none | none |  | B is an extremely rapid uninterrupted lane-1 jack for the entire window; A distributes the equally close attac... |
| 016 | natural_gap | B | same/low | same | jack (repeated-lane runs) vs chord tradeoff | none | none |  | A has longer narrow repeated-lane runs, notably lane 2 early and lane 1 later; B sustains more frequent multi-... |
| 017 | mechanical | A | A/high | agree | chord+jack (full-chord jack) | absent | none |  | A repeatedly strikes all four lanes, including extremely tight bursts; B distributes the same visible rhythmic... |
| 018 | natural_gap | A | A/high | agree | chord (doubles, triple) | none | none |  | A sustains changing and repeated doubles through the second half, with a late triple, where B is mostly repeat... |
| 019 | fixed | B | B/high | agree | chord+jack, density/stamina | none | none |  | B becomes a relentless stream of alternating doubles with close repeats and chord changes across the last thre... |
| 020 | mechanical | A | A/high | agree | chord+jack (full chords) | none | none |  | A turns the bursts and later continuous run into repeated full chords, including a very tight burst late in th... |
| 021 | fixed | B | B/high | agree | chord+jack (full-chord jacks), density | secondary | none |  | B combines sustained full-chord jacks with short holds and separate releases between attacks; A is spaced sing... |
| 022 | natural_gap | A | A/low | agree | chord (changing doubles) vs jack (repeated singles) | none | none |  | A carries more changing doubles through the middle of the window, adding simultaneous finger work; B has more ... |
| 023 | natural_gap | A | A/low | agree | chord (doubles, triples, quads) | counterweight | none |  | A has heavier changing doubles, triples and occasional full chords, especially in segment 3. B adds hold-and-t... |
| 024 | natural_gap | A | A/high | agree | chord+jack | none | none |  | A repeatedly interleaves doubles with singles and finishes with full chords alternating with doubles; B is alm... |
| 025 | fixed | A | A/high | agree | chord+jack (four-key chord jacks) | secondary | none | extreme, playability unverified | A sustains extremely tight four-key chord jacks across almost the whole window, with tiny LN releases mixed in... |
| 026 | natural_gap | B | B/high | agree | chord+jack (repeated chord fingers) | absent | none |  | B repeatedly requires two-key chords and occasional triples where A largely uses singles; repeated chord finge... |
| 027 | fixed | A | A/high | agree | chord+jack (hand chords, quads) | counterweight | none |  | A sustains alternating two-key hand chords, develops overlapping chord fingers, and ends with repeated quads. ... |
| 028 | fixed | A | A/high | agree | chord+jack (repeated large chords) | secondary+counterweight | none |  | A has prolonged tightly repeated large chords, brief LN chord releases and renewed attacks. B has single-note ... |
| 029 | natural_gap | A | A/high | agree | chord (jumps, hand chords), recovery | secondary | none |  | A has frequent jumps and alternating hand chords, then paired holds against opposite-hand taps. B is predomina... |
| 030 | near_equal | A | same/low | same | jack (repeats), density | tradeoff | none |  | Comparable single-note stream pace and brief repeats: A concentrates recurring rightmost-finger attacks, while... |
| 031 | mechanical | A | A/high | agree | chord+jack (quad repetition) | none | none |  | A repeats four-key chords throughout, including the faster passages; B spreads the same visible rhythm across ... |
| 032 | natural_gap | B | B/high | agree | chord+jack | secondary+counterweight | none |  | B combines more chord attacks and repeated chord fingers with frequent short holds and releases, especially th... |
| 033 | mechanical | B | B/high | agree | jack (lane-1 jacks) | none | none |  | B sustains fast lane-1 jacks throughout; A spreads the same flowing rhythm across fingers. Repeated-finger str... |
| 034 | fixed | B | B/high | agree | chord+jack (four-key chords) | secondary | none |  | B repeatedly strikes four-key chords with short LN releases mixed into the repetitions; A is mostly separated ... |
| 035 | natural_gap | B | B/high | agree | chord+jack (doubles, triples, repeated fingers) | secondary | none |  | B has more doubles and triples, repeated fingers and taps around changing holds, especially early and late; A ... |
| 036 | mechanical | A | A/high | agree | chord+jack (four-key chord jacks) | none | none | extreme | A sustains four-key chord jacks with tightly spaced bursts across both segments; B distributes single taps acr... |
| 037 | fixed | A | A/high | agree | chord+jack (full chords), recovery | secondary | none |  | A repeatedly presses and releases full chords, mixing short LNs with chord jacks; B is mainly isolated holds a... |
| 038 | fixed | A | A/high | agree | chord+jack (triple and quad repetitions) | secondary+counterweight | none |  | A develops from doubles into sustained triple and four-key repetitions with short LN changes; B stays mostly s... |
| 039 | natural_gap | B | B/high | agree | chord+jack (larger chords, repeated fingers) | counterweight | none |  | B has sustained larger chords and overlapping repeated fingers across both segments. A has some LN coordinatio... |
| 040 | near_equal | A | B/low | disagree | chord+jack (A doubles, repeats; counterweight) | primary | none |  | Close: B adds more changing holds with intervening taps and release coordination, especially in the middle; A ... |
| 041 | fixed | B | B/high | agree | chord+jack (four-key chord jacks), density | secondary | none | denies_unplayable | B has sustained rapid four-key chord jacks in the latter half, with short holds and releases mixed into repeat... |
| 042 | near_equal | A | same/low | same | chord (single/double flow) | tradeoff | none |  | Both have comparable single/double flow and short hold control. A carries more holds through the middle; B has... |
| 043 | fixed | A | A/high | agree | chord+jack, stamina | secondary | none |  | A sustains dense three/four-key repeated chords across the final three segments, after an early hold with othe... |
| 044 | natural_gap | B | B/low | agree | chord+jack (chording, anchor) vs jack (A single-column) | none | none |  | A has longer isolated single-column jacks; B maintains much more chording, including repeated chord fingers an... |
| 045 | fixed | A | A/high | agree | chord+jack, density | secondary | none |  | A has prolonged dense three/four-key chord jacks and rapid reattacks around short hold releases; B is predomin... |
| 046 | near_equal | B | same/low | same | density (rolling singles), jack (brief repeats) | absent | none |  | Both sustain similarly fast rolling single-note flow, occasional doubles, and brief same-column repeats. The l... |
| 047 | natural_gap | A | A/high | agree | chord+jack (doubles, repeated chord fingers) | counterweight | none |  | A sustains many more doubles and repeated chord fingers, culminating in alternating four-key and smaller chord... |
| 048 | near_equal | B | B/low | agree | chord+jack (same-hand doubles, repeated chords) | counterweight | none |  | B repeatedly uses same-hand doubles and short repeated-chord figures through the early and middle segments. A ... |

Astra totals [measured on my coding]:
- attacks: 48.
- holds: primary 3 (001, 011, 040), secondary 14, counterweight 5, tradeoff 3, secondary+counterweight 3, explicitly absent 4, none 16.
- reading: 2 (004 counterweight, 006 neutral).
- unplayable: 0.
  - Astra explicitly denies any visible impossible overlap in 002, 003, 005 and 041.
  - In 025 it writes "physically extreme ... playability unverified".

### Agreement split by what was cited

[measured, on my coding]

| Judge | subset | cited | n | agree | disagree | same | non-agreeing pairs |
|---|---|---|---:|---:|---:|---:|---|
| astra | all | holds cited | 28 | 23 | 2 | 3 | 001, 006, 030, 040, 042 (all near_equal) |
| astra | all | holds not cited | 20 | 16 | 1 | 3 | 004, 012, 016, 046 |
| astra | natural_gap + near_equal | holds cited | 14 | 9 | 2 | 3 | as above |
| astra | natural_gap + near_equal | holds not cited | 10 | 6 | 1 | 3 | as above |
| astra | all | reading cited | 1 | 0 | 1 | 0 | 004 |
| human | notes only | holds cited | 0 | - | - | - | - |
| human | notes only | reading cited | 1 | 1 | 0 | 0 | - |
| human | notes only | neither | 12 | 9 | 2 | 1 | 001, 004, 016 |

- [measured] Astra cites holds on 22 pairs with |Δlog r| ≥ 0.175, and agrees with r on all 22. Holds show up in non-agreements only where r is near-equal.
- [measured] Where holds were the primary reason: 011 agrees (natural_gap, gap 0.225); 001 and 040 disagree (both near-equal).
- [measured] The screen holds little LN, and both judges agreed on every LN-heavy pair:
  - 19 of 48 pairs have no LN head on either side.
  - 8 pairs have a side with LN-head share above 0.2: 008, 009, 011, 014, 032, 035, 037, 038.
  - 2 have a side above 0.5: 011 (0.61/0.70) and 037 (0.48/0.57).

### The two natural-gap non-agreements (both chart-013, 8 s window)

[measured] All sides share the same 67-row skeleton and carry no LN. sample-0024 appears in three pairs; its profile is r 2.216, 150 notes, rows by chord size 17 singles / 17 doubles / 33 triples, longest jack 5 rows.

| Pair | other side | other r | other notes, chords 1/2/3 | other longest jack | human | astra |
|---|---|---:|---|---:|---|---|
| 012 | sample-0058 | 1.816 | 122, 20/39/8 | 5 | sample-0024 harder (agree) | same |
| 016 | part3b-0197 | 1.844 | 122, 36/7/24 | 9 | part3b-0197 harder (disagree, low) | same |
| 039 | part3b-0188 | 1.860 | 125, 18/40/9 | 4 | sample-0024 harder (agree) | sample-0024 harder (agree) |

- The human's only natural-gap disagreement (016) goes to the side with the 9-row jack.
- Astra's near-equal disagreement in 004, also on chart-013, likewise goes to the side with the 9-row jack (9 against 5).
- In 044 the longer-jack side (12 against 9 rows) lost, for both judges, to a side with 44% more notes.

## 6. Unplayable and degenerate sides

[measured] Every side the human called unplayable or collapsed (7 pairs). Astra called none unplayable.

| Pair | side | literal "unplayable" | category | how made | r (other side) | LN-head share | notes/s | rows/s | mean chord | quad-row share | same-lane repeat rate | fastest same-lane reattack ms |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 002 | B | yes | mechanical | constructed-01-1, repeated-lane singles | 1.320 (1.019) | 0.00 | 9.4 | 9.4 | 1.00 | 0.00 | 1.00 | 80 |
| 003 | A | yes | fixed | part3a-0383, eta +2 | 3.791 (1.016) | 0.00 | 25.9 | 6.7 | 3.87 | 0.94 | 0.97 | 75 |
| 005 | A | yes, "pattern degradation again" | fixed | part3a-0221, eta +1 | 3.524 (1.028) | 0.01 | 36.4 | 10.6 | 3.45 | 0.71 | 0.88 | 32 |
| 009 | A | yes | fixed | part3a-0278, eta +2 | 3.750 (1.065) | 0.22 | 12.8 | 3.4 | 3.73 | 0.84 | 0.97 | 34 |
| 010 | B (implied) | yes | mechanical | constructed-00-1, repeated-lane singles | 1.295 (1.020) | 0.00 | 4.9 | 4.9 | 1.00 | 0.00 | 1.00 | 91 |
| 025 | A | yes | fixed | part3a-0076, eta +4 | 3.843 (1.046) | 0.02 | 72.4 | 18.7 | 3.88 | 0.94 | 0.97 | 53 |
| 028 | A | no, "map collapse to full 4 lane" | fixed | part3a-0186, eta +1 | 3.491 (1.108) | 0.13 | 33.9 | 9.8 | 3.46 | 0.68 | 0.86 | 47 |

- [measured] All seven are the higher-r side of their pair.
- [measured] pair-002 B is the constructed contrast in which every head in the window moves to lane 1 on the source skeleton.

For context, all 16 positive-tilt sides of the fixed category [measured]:

| Pair | side | eta | r | quad-row share | mean chord | notes/s | same-lane repeat rate | human note |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 003 | A | 2 | 3.79 | 0.94 | 3.87 | 25.9 | 0.97 | A is not playable at all lol |
| 005 | A | 1 | 3.52 | 0.71 | 3.45 | 36.4 | 0.88 | The pattern degration occur again. A is not playable. |
| 008 | B | 4 | 3.83 | 1.00 | 4.00 | 19.0 | 0.99 | - |
| 009 | A | 2 | 3.75 | 0.84 | 3.73 | 12.8 | 0.97 | Obviously A is not playable with full 4-lane consequent attacks. |
| 014 | A | 4 | 3.85 | 1.00 | 4.00 | 19.5 | 0.99 | - |
| 019 | B | 1 | 1.79 | 0.00 | 1.78 | 33.3 | 0.00 | - |
| 021 | B | 2 | 3.81 | 0.93 | 3.85 | 36.3 | 0.97 | - |
| 025 | A | 4 | 3.84 | 0.94 | 3.88 | 72.4 | 0.97 | A not playable lol. |
| 027 | A | 1 | 2.13 | 0.08 | 2.18 | 14.6 | 0.24 | - |
| 028 | A | 1 | 3.49 | 0.68 | 3.46 | 33.9 | 0.86 | map collapse to full 4 lane. |
| 034 | B | 4 | 3.78 | 0.95 | 3.87 | 23.0 | 0.98 | - |
| 037 | A | 4 | 3.87 | 1.00 | 4.00 | 19.8 | 0.99 | - |
| 038 | A | 1 | 3.17 | 0.51 | 3.09 | 18.4 | 0.76 | - |
| 041 | B | 1 | 2.90 | 0.37 | 2.80 | 16.1 | 0.72 | - |
| 043 | A | 1 | 3.20 | 0.56 | 3.06 | 32.3 | 0.80 | - |
| 045 | A | 1 | 3.20 | 0.56 | 3.15 | 29.7 | 0.79 | - |

- [measured] 13 of 16 have a quad-row share of at least 0.5.
- [measured] 14 of 16 have r above 2.806, the maximum over all 4,605 natural 16 s scopes from the 1,163 source charts in `part2.json`. The constructed all-quad sides score r 3.78 to 3.96, the same ceiling.
- [measured] The constructed one-lane jacks the human called unplayable score r 1.30 and 1.32. That is between the natural 8 s p25 (1.273) and median (1.427).
- [measured] The human wrote no note on 11 of these 16 sides. Among them are 008, 014, 034 and 037, with quad-row share 0.95 to 1.00.
- [inferred] The human largely stopped writing notes after pair-016, so a missing "unplayable" note says nothing about playability.

## 7. Blindness check

### What was checked

1. [measured] **Sheet consistency.**
   - `judge-sheet.csv` matches the six batch files on all 48 rows: verdict, confidence and note against the batch CSVs, and verdict, confidence and `decisive_comparison` against the evidence JSONs.
   - The sha256 of all 12 batch files matches `judging-report.json`.
   - The superseded partial pass (`agent-judgments-001-016.csv`, 6 pairs) is excluded from scoring.
2. [measured] **grep of `S/review/`** (every non-image file: evidence JSONs, CSVs, README, judging report, manifest, server log). It found none of:
   - `KEY-sealed` or `private/`;
   - key field names: `which_higher_r`, `ln_share`, `natural_gap`, `near_equal`, `mechanical`, `constructed`, `part3a`/`part3b`, `sample-0…`;
   - decimal numbers that look like r values.

   What it did find:
   - "sealed" appears only in the judges' statements that they did not open the sealed key.
   - "strain" and "workload" appear in four Astra notes as plain words ("repeated-finger strain", "jack strain", "chord workload").
   - `review/manifest.json`, which the judges read, contains only windows, file paths and segment images.
3. [measured] **No Astra judging job record.**
   - Neither `.sync/cp/jobs` nor `.sync/mac/jobs` has a job for the judging, so there is no brief.md, log or last.md.
   - The judging ran in a Codex thread started outside `ens astra`; the playback verification names the "Codex in-app browser".
   - The only related job is `20261007-061042-r2-phasec-c0`, which built the screen and wrote the key.
4. **One read-only search on the mac** (job `20261007-135957-screen-blind-grep`).
   - Scope: files in `~/.codex/sessions/2026/10/07` that contain `KEY-sealed` or `screen/review`, with per-file line counts for key strings. Mac local time is UTC+9.
   - [measured] **The six judges.** Six rollouts started between 22:08:22 and 22:21:53 JST and were last written between 13:10:47 and 13:24:09 UTC.
     - Each contains the batch-file name 5 or 6 times, and their end times match the six batch files (13:10:38 to 13:24:02 UTC).
     - Each has 0 lines with `KEY-sealed`, `which_higher_r`, `ln_share_difference`, `private/candidates` or `judge-sheet`.
     - [inferred] Listing `screen/` or reading the key would have put `KEY-sealed` into the rollout, because rollouts record commands and tool output (the batch-file hits show that).
   - [measured] **The likely orchestrator.** One rollout started at 21:48:27 JST and was last written at 13:50:53 UTC, about a minute after `human-judgments.csv` appeared.
     - It has 2 lines with `KEY-sealed`, 16 with `judge-sheet`, 9 with batch names, and 0 with `which_higher_r`, `ln_share_difference` or `private/candidates`.
     - It looks like the thread that built the review page, ran the judges and wrote `judge-sheet.csv`.
   - [measured] **Other hits.**
     - Rollouts that contain `which_higher_r` (06:10 to 07:48 UTC) belong to the screen-building job.
     - Other `KEY-sealed` hits (17:29, 18:01, 21:32 and 21:34 JST) are other threads. The last two ran at the same time as the judges but are not judge threads.
5. [measured] **What the human's browser fetched.** Source: the fresh `server.log`, 62,460 B; the local replica is a stale 9,534 B copy.
   - Requests: only the review page and its scripts, `review/manifest.json`, segment PNGs, audio, and `charts/pair-NNN-{A,B}.osu` (all 96 files, each exactly once).
   - There is no request for `review/astra-*` or the evidence files.
   - `serve.py` serves only `review/`, `charts/` and `audio/`. The key and `judge-sheet.csv` could not have been served, but `review/astra-*.csv` could have been.
   - The review page (`index.html`) fetches only `./manifest.json` and the `.osu` files.
6. [measured] **Timing** (mac index times, UTC).
   - Astra's batches were written from 13:10:38 to 13:24:02, and the final sheet at 13:24:50.
   - The human's session runs from 13:17:41 to 13:48:52 (last pair loaded). It began with a cold cache: 200 responses and a favicon request, unlike the Codex browser's 304s at 12:57 to 13:07.
   - `human-judgments.csv` appeared at 13:49:49.

### Verdict

- [measured] Nothing shows that any of the six judges listed, opened or was given the key or `private/`. [inferred] I treat Astra's sheet as blind.
- Not verified:
  - What the 2 `KEY-sealed` lines in the orchestrator rollout say. No key field names appear there.
  - Whether the judge prompts carried category hints worded without key field names; I did not grep for "tilt", "constructed" or "control".
  - Whether the human looked at Astra's files outside the browser.
  - The identity of the 21:50, 21:59 and 22:01 JST rollouts. They contain no key strings and are probably the review build, the superseded pass and the playback fix.

  One search was the budget. Reading the orchestrator's 2 lines would settle the first point.
- Could the judges see each other's answers?
  - Astra could not have seen the human's. They stayed in the human's browser storage until the export at 13:49:49, 25 minutes after Astra finished [measured].
  - The human judged while Astra's batch files were landing in `review/`, but the browser never fetched them [measured]. Viewing them outside the browser cannot be excluded. That would mostly affect kappa.
  - [inferred] Neither looks like copying: the judges never pick opposite sides, and all 7 label differences are one judge saying "same".
- [inferred] A judge that had read the key would hardly go against r on 3 of 4 near-equal answers.

## 8. Caveats

- **Dependence** [measured]: the 48 pairs come from 17 source charts.
  - 7 samples appear in more than one pair: sample-0024 (012, 016, 039), part3a-0388 (013, 024, 047), sample-0087 (013, 030), part3a-0381 (030, 047), part3b-0044 (018, 044), part3b-0071 (022, 026), sample-0029 (032, 035).
  - chart-010's 16 s window appears in 6 pairs and chart-013 in 4.
  - Use the chart-clustered tests; the pooled p values overstate the evidence.
- **Small n** [measured]: the informative categories have 16 natural pairs on 8 charts plus 8 controls. Every claim about holds, reading or jacks rests on 1 to 3 pairs.
- **Note-count confound** [measured]: the higher-r side has more notes in:
  - 16/16 fixed pairs;
  - 16/16 natural_gap pairs;
  - 7/8 near-equal pairs (1 tie);
  - among mechanical pairs, only the 4 chord constructions.

  More hold time sits on the higher-r side in only 16 of 48 pairs (20 ties).
- **Different information** [measured from evidence]:
  - Astra judged static images (240 px/s) with no audio and no playback. One judge (pairs 009 to 016) also read raw `.osu` rows for pair-009.
  - The human used synchronized 1x playback plus the segment images.
- **Human pace and fatigue** [measured]: 48 pairs in 32 minutes. "Dwell" below is the time from loading one pair to loading the next.

  | Pairs | dwell median | dwell shorter than the window | notes written | agree/disagree/same |
  |---|---:|---:|---:|---|
  | 001 to 016 | 40 s | 2 | 11 | 13/2/1 |
  | 017 to 032 | 22 s | 6 | 2 | 16/0/0 |
  | 033 to 048 | 22 s | 6 | 0 | 12/0/4 |

  - Of the 14 short-dwell pairs, 13 are fixed or mechanical; the other is 032. [inferred] Those were decided from the images without playing the window once at 1x after loading. Revisits are invisible because files were cached.
  - Agreement does not fall with position.
  - All 4 near-equal pairs in the last block got "same" (040 and 046 "same/high"), against 1 of 3 in the first block. Learning and fatigue cannot be told apart.
  - Low-confidence answers took a median 58 s against 20.5 s for high confidence. pair-016 took 390 s and pair-004 184 s.
- **Recognition** [inferred]: the note on pair-005, "pattern degration occur again", shows the human already recognised the tilt artefact. From then on, a collapsed side gives away the positive tilt. The direction scoring is unaffected, because those sides are obviously harder anyway.
- **Astra instability** [measured]: the superseded partial pass judged 6 of the same pairs. 5 verdicts match the final ones; pair-004 (near-equal) flipped from B (with r) to A (against r).
- **Replica oddities** [measured]:
  - The local `server.log` replica stopped updating at 13:24:51; the mac copy is 62 KB, last written 13:48:52.
  - The `.osu` files are not mirrored, so scope statistics come from `harness/charts/*.json`.
    - Each file's `sourceSha256` matches the side's `osu_sha256` in the key.
    - Parsed notes, rows, LN heads and same-lane repeats match the key wherever the key has them.
  - `parser-verification.json` reports 20 fractional LN end times.

## 9. What this implies for ras-v1 (inference, not measurement)

- [inferred] **Direction is good enough at useful step sizes.**
  - Inside the natural range, both judges order samples as r does once |Δlog r| is above about 0.2, and treat gaps of 0.03 or less as "same".
  - For phase C, a "harder" or "easier" step of at least about 20 to 25% in r should be perceived; a step of a few percent will not be.
  - That also gives the controller a natural deadband.
- [inferred] **This screen does not justify a hold or reading term yet.**
  - Holds swayed only Astra, only on near-equal pairs, and never on a pair with a real r gap. The human never cited holds.
  - The screen was LN-matched and mostly LN-light, so it had almost no power on LN-heavy material (8 pairs with a side above 0.2 LN share, 2 above 0.5).
  - A targeted LN screen, with attack r held fixed and hold load varied, would answer this better than adding a term blind.
- [inferred] **The real gap is a plausibility ceiling, not a missing term.**
  - r keeps rising as charts collapse into quad chordjack walls, and those walls are what the human called unplayable.
  - Positive tilt drove 14 of 16 sides beyond the natural maximum of r.
  - Phase C should clamp targets to the natural range, for example at most the natural p99 (about 2.3 at 8 to 16 s). It should also add a guard on quad-row share or a likelihood floor, rather than trust r above about 2.8.
- [inferred, weak] **Long single-lane jacks may be under-weighted against chord mass.**
  - Two calls (human on 016, Astra on 004) went against r to the side with the 9-row jack.
  - The one-lane jacks the human called unplayable score only r ≈ 1.3, below the natural median, while all-quad walls reach 3.9.
  - This is a hypothesis for a small jack-length screen, not a finding: 2 pairs, one chart.
- [inferred] **It is not yet shown that r beats a note count.**
  - Direction tracks note count in 43 of 48 pairs, so this screen cannot tell ras-v1 from "more notes on the same skeleton".
  - That is fine if phase C only needs direction on a fixed skeleton.
  - If r is meant to rank layouts with similar note counts, that needs pairs matched on note count.

----- END report -----
