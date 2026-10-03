# osu! player data: uses, caveats and limits of interpretation

Shareable. Written 2026-10-03 by main session `10fd41cc` (Claude, control plane), at the human's request to write down how osu! player data is to be used and what limits its interpretation ([private, local](private/human-inputs/10fd41cc-e5d5-4d85-8def-2bee49dada16.md#prompt-5)). What exists and under which terms: [s-play-data](operator-properties.md#s-play-data), with the worker's report [play-data-availability](play-data-availability.md). Everything below is the agent's `(proposed)` reading unless marked as the human's. Numbers come from one probed map and one replay unless stated; they illustrate, they do not estimate.

## What the data is

| Data | Granularity | Who it covers | Route |
| --- | --- | --- | --- |
| `failtimes`: 100 `fail` and 100 `exit` bins | per beatmap, fraction of map progress, lifetime totals | everyone who played the map, all mods, all clients (newer maps apparently sampled) | API beatmap lookups (50 ids per batch); data.ppy.sh dumps |
| `playcount`, `passcount` | per beatmap, lifetime totals | as above | API; every `metadata.json` on disk |
| Replays | per press and release, 1 ms | passed scores only: about the top 1,000 per map in stable; every player's best per mod combination in lazer | API score download, 10 per minute, client credentials |
| Best scores, attempt counts by player | per player and map | top 1,000 or 10,000 players, or a random 10,000, per mode | data.ppy.sh dumps |

## Uses

<a id="u-population-prior"></a>**U1. Test the human's population prior.** [h-exponential-population](operator-properties.md#h-exponential-population) says the share of players who manage a chart falls exponentially as difficulty climbs and gaps shrink. Across charts: pass rates against star and against the chart's demand at its peaks. The test needs attempts stratified by player skill (caveat C8); the dumps' per-player attempt counts for sampled populations are the closest source.

<a id="u-fail-location"></a>**U2. Where plays end, against local demand.** Within a chart, fail bins against the demand descriptions of the same span, conditional on the chart's star: which local organisation precedes the end of plays. This is the most direct observation of response at passage scale that the aggregates allow, and it is within the API terms.

<a id="u-response-from-play"></a>**U3. Response from play.** Replays give hit error, misses and release timing per note against the local pattern and the player's standing. That is the formulation's target response defined from play rather than from mapper evidence (`5c56e28:docs/formulation/gameplay-state.md`, "Defining the response from mapper evidence"). At scale it needs ppy's agreement.

<a id="u-check-demand"></a>**U4. Check the evaluator's demand side.** Whether the demand descriptions in which corpus charts of a star are normal also order where plays end: a check of the descriptions, never a fit of the evaluator, whose target stays the corpus ([d-evaluator-purpose](operator-properties.md#d-evaluator-purpose)).

<a id="u-control"></a>**U5. Later: difficulty control.** A difficulty request read through what players manage, not only through the star ([d-final-conditions](r2-ln-design.md#d-final-conditions)).

<a id="u-not"></a>**Not uses.** Player data does not define, fit or calibrate the normal-chart evaluator, does not supply its negatives, and does not stand in for the human's judgment: players pass charts the human rejects ([s-ranked-contains-rejected](lineage-review/synthesis.md#s-ranked-contains-rejected)), so playable is not good. No profiling of individual players.

## Caveats in the data

Fail data:

- <a id="c-lifetime-mixed"></a>**C1. Lifetime totals, everything mixed.** All mods together: rate mods change every gap in seconds, mirror and random change lanes, others change judgement or the health bar. All clients, and every era of the player population since the map was ranked (whether plays before ranking count is not known). No split by player or mod.
- <a id="c-bins"></a>**C2. Bins are fractions of map progress.** How progress is defined (first to last object, lead-in and breaks included or not) is not verified. On a 2 to 4 minute song one bin is 1.2 to 2.4 s, so a fail locates a passage, never a pattern.
- <a id="c-sampling"></a>**C3. Apparent sampling and missing fails.** On maps from about 2021 every bin is a multiple of 9, and fails plus exits plus passes fall well short of plays on the newest maps (ratio 0.42 to 0.77, against 0.83 to 0.99 on older ones). The worker reads this as 1-in-9 sampling and unrecorded lazer fails; ppy does not state either. A new 4K map with 6,040 plays had 7 sampled fails.
- <a id="c-exits"></a>**C4. Exits are not fails.** Exits include restarts for accuracy, bad starts and boredom; on the probed map they crowd the first bins. Keep exits apart and treat them as weak evidence of demand.
- <a id="c-health-lag"></a>**C5. The health bar lags.** A fail happens when health reaches zero, which integrates misses over the preceding seconds and depends on the chart's HP setting and on mods (no-fail plays never fail). The fail bin trails the demand that caused it; a lag model is needed before reading a bin as a location.
- <a id="c-version"></a>**C6. Version.** Counts belong to the beatmap id; a map updated before ranking, or a local file that differs, breaks the link. Match the checksum.
- <a id="c-snapshot"></a>**C7. Snapshot and scale.** Counts grow over time, so every value carries its fetch date. Popularity spans orders of magnitude (6,040 to 2.3 million plays among the probed mania maps), so precision varies by map.

Pass counts:

- <a id="c-selection"></a>**C8. Self-selection.** Players choose charts near their level. A hard chart attempted mostly by strong players can show a higher pass rate than an easy one attempted by beginners. Pass rate measures the audience as much as the chart.
- <a id="c-retries"></a>**C9. Plays, not players.** Play and pass counts include every retry and every pass by the same player.

Replays:

- <a id="c-survivorship"></a>**C10. Survivorship.** Only passed plays, and only each player's best per mod combination: errors near the limit and the plays that failed are not observed, and best-of-many understates typical error.
- <a id="c-clients"></a>**C11. Two populations.** Below the top 1,000 only lazer players appear; the population behind non-top replays differs from the stable-era population behind most of the aggregates.
- <a id="c-offset"></a>**C12. Offsets and setup.** Audio offset and hardware latency shift every press (the probe replay: mean -15 ms, SD 22.6 ms); a constant per replay must be removed before errors mean anything. Scroll speed, skin, key bindings and fingering are unknown, so lane to finger is an assumption.
- <a id="c-mods-replay"></a>**C13. Mods in replays.** Rate mods change the clock; mirror and random change lanes and must be mapped back.
- <a id="c-release-window"></a>**C14. Releases.** Release judgement is lenient, so early releases (the probe: mean -34 ms over 49 holds) may be strategy, not error.

Terms:

- <a id="c-terms"></a>**C15. Terms.** Mass collection of score or replay data needs ppy's agreement; the dump licence allows statistical analysis and asks for contact before production or public use; site content may not be redistributed. Per-player data stays aggregated and unpublished.

## Limits of interpretation

- <a id="l-population"></a>**L1.** Aggregates describe the people who played that map, under all mods, over its lifetime. They say nothing direct about a canonical player or a physiological limit.
- <a id="l-fail-where"></a>**L2.** A fail bin supports "plays end around this passage of this chart more than around others", lagged and smoothed by the health bar. It does not support "this pattern is unplayable".
- <a id="l-pass-not-difficulty"></a>**L3.** Pass rate is not difficulty without a model of who attempts. Across charts, pass rate against star confounds difficulty with audience; the population prior is tested only with attempts stratified by skill.
- <a id="l-replay-lower-bound"></a>**L4.** Replays support error against pattern for players who passed, at their best: a lower bound on difficulty near the limit.
- <a id="l-not-quality"></a>**L5.** Player data says what players manage, not what mappers or the human consider good. It informs response and difficulty, never the normal-chart target.
- <a id="l-not-causal"></a>**L6.** An association between local demand and fails is not a cause: fails also depend on chart length, HP and OD settings, earlier passages, and how many weak players a map attracts.

## Fetched fail data and the dump's player sample

<a id="s-failtimes-fetched"></a>**Observation, 2026-10-03, from a fresh worker of this session ([player-data-estimate-and-fetch](player-data-estimate-and-fetch.md)); the main thread re-read the manifest, coverage and README on the control plane.** Jobs `20261003-player-data-failtimes` and `20261003-player-data-postprocess` on bings-mac wrote `~/ensomi/ensomi-model/artifacts/player-data-20261003/failtimes/failtimes.parquet` (21,975 rows, SHA-256 `53d788e0…`; the table before a zero fill kept beside it), with `manifest.json` (script hashes, corpus hash `49ab9e1b…`, request log, dump identity) and `README.md`.

- Coverage: every ranked (19,147) and loved (2,632) corpus chart has fail and exit bins, play and pass counts. 104 rows have none: charts without an API match whose file id the API does not know. 91 corpus charts have no beatmap id.
- Sources: 21,584 rows from the data.ppy.sh dump `2026_09_01_performance_mania_random_10000.tar.bz2` (SHA-256 `2fa58ed1…`, counts as of 2026-09-01), 287 from 11 batch API requests (as of 2026-10-03). 51 rows lacked one array in the dump and got 100 zeros, as osu-web serves it.
- 1,326 rows have a source checksum different from our chart file: their counts describe another version of the chart.
- 13,590 of the 19,147 ranked rows (71%) have every non-zero bin divisible by 9 (the sampling of [C3](#c-sampling) covers most of the corpus, not only a few new maps); fails plus exits plus passes fall to a median 0.61 of plays at the newest ids.
- The dump is kept on the mac (`artifacts/player-data-20261003/dump/`, 448 MB; 328 GiB free).

<a id="s-dump-player-sample"></a>**Observation, same source. The dump holds a random sample of 10,000 mania players with skill, best scores and attempts on corpus maps.** Best scores with judgement counts, mods and pp: 1,236,864 stable scores on 20,963 corpus maps and 1,061,818 in the newer table on 21,293 (the two may overlap); attempts per player and map: 1,439,692 rows on 21,550 corpus maps, 8.1 million attempts; per-player pp, accuracy, play count, fail and quit counts. 372,751 of the newer scores carry a replay, a mean of 17.5 per corpus map: a sampling frame for replays if ppy agrees.

<a id="a-player-sample-use"></a>**Agent reading (proposed), 2026-10-03.** This weakens [C8](#c-selection): with players of known skill, pass and accuracy can be modelled against skill and chart, so [U1](#u-population-prior) becomes "the probability that a player of skill s passes chart m, and how it falls as the chart's demand rises", without any API harvesting. What remains: players still choose which charts to attempt; "random" is the dump's sample of mania players, whose definition is not stated; a best score shows a pass, not how many attempts preceded it; attempts include retries. New caveats: <a id="c-version-mismatch"></a>**C16**, 1,326 rows describe another chart version (the dump's `osu_files` could supply those versions; not opened); <a id="c-two-dates"></a>**C17**, dump rows and API rows describe different dates.

Replay estimate, for a later decision: about 10 KB per corpus chart; a pilot of 600 replays is 6 MB and an hour at 10 per minute, a core design of 4,500 is 45 MB and 7.5 h, a full one of 30,000 is 300 MB and 50 h. Any of them needs ppy's agreement first ([C15](#c-terms)).

## Fable review of the views, 2026-10-03

<a id="a-fable-views"></a>**Agent reading (proposed), 2026-10-03, of a fresh Fable subagent's memo ([fable-player-data-views](fable-player-data-views.md)), requested by the human ([private, local](private/human-inputs/4259953e-714c-4628-8911-eb1ec9fafdd8.md#prompt-1)). Nothing in it is a decision.** The main thread checked against job `20261003-player-data-views` (`artifacts/player-data-views-20261003/dump-views.json` in the code checkout, counts only): the score-table counts (882,842 imported stable and 178,976 lazer rows on corpus maps), the player sample's skill (median 174 pp, 26% at 1,000 pp or more), the rate pairs (29,672 NM with DT, 6,640 NM with HT), the mod counts, and the fail-histogram concentration (median normalised entropy 0.83 at 1 star to 0.73 at 7, against a uniform null of 0.94 to 0.99). It checked the health constants against `ppy/osu` master `ManiaHealthProcessor.cs` (fetched 2026-10-03): no drain, a miss costs (HP+1)×0.0075, a Perfect at HP 8 refills 1/45 of a miss before the multiplier. Stable's rule is closed source and was not checked.

What the agent would adopt:

1. One score view, not two: `scores` contains the stable bests as imports. The sample is weighted toward low skill, so it serves 2 to 6 stars and not a 6-star upper bound. This corrects the table above.
2. Aggregates identify where plays end and the population prior at chart scale. Every per-action part (the limit per relation, burst against sustained, τ, fatigue, tap against LN head, release, hand coordination) needs replays.
3. Even replays identify only a response function: miss probability and error spread against slack, relation and skill. A latent cost is defined only up to a monotone link. The response target should be that function, not a cost law ([p-identification-map](fable-player-data-views.md#p-identification-map)). This needs the human.
4. The health bar has a known kernel ([p-health-kernel](fable-player-data-views.md#p-health-kernel)). A fail sits inside or just after a cluster of misses, so the lag is bounded by the cluster ([C5](#c-health-lag) narrowed). HP and OD enter every fail-position analysis.
5. Rate mods and lazer's Hold Off, No Release, Mirror and Invert are within-player, within-chart contrasts already in the dump.
6. A skill-indexed frontier ([p-player-frontier](fable-player-data-views.md#p-player-frontier)) belongs to `response` and `control`. It never enters the evaluator's divergence. It widens [t-formulation-physiology](operator-properties.md#t-formulation-physiology).

Where the agent qualifies the memo:

- The pre-registered A1 ([p-prereg-a1](fable-player-data-views.md#p-prereg-a1)) excludes rate, LN and visibility-reducing mods from a clean pass but not the key-count mods: 7,456 lazer rows carry `4K` and about 200 carry other key counts on corpus maps. Those mods should be excluded, or the chart's key count verified.
- The skill prior `rank_score` is computed from the same players' best scores, so θ is partly fitted on the outcome. M0 and M1 share θ, so the increment Δ₁ is less exposed than the levels.
- The 150 ms run threshold inside a descriptor is a question under [d-no-cutoffs](operator-properties.md#d-no-cutoffs) that the memo itself puts to the human.
- The threshold of 0.003 nats per pair is a declared estimate, not derived.

<a id="d-response-function"></a>**Decisions, human, 2026-10-03 (selections among the agent's options, [private, local](private/human-inputs/4259953e-714c-4628-8911-eb1ec9fafdd8.md#answer-2)).**

- The response target is the response function: miss rate and error spread given slack, relation and skill. It replaces a cost law. The "exponential" prior becomes a claim about the shape of this function and about the share of players who manage a chart.
- A1 runs as pre-registered, with key-count mods excluded, after harness v0 finishes: first on the fit split, then once on calibration as confirmation.
- ppy is contacted for a replay pilot only after A1, and only if A1 shows a demand signal beyond star.

<a id="d-binding-threshold"></a>Later the same day ([private, local](private/human-inputs/4259953e-714c-4628-8911-eb1ec9fafdd8.md#prompt-3)):

- The binding is the formulation's canonical hand roles: lanes 1 and 2 left, 3 and 4 right, outer as the middle finger and inner as the index (`5c56e28:docs/formulation/notation.md`, "Canonical hand-role coordinates").
- The human is unsure of the 150 ms run threshold and reserves it: it stays a provisional, labelled parameter of A1, with its pre-registered sweep, to be refined later. Agent reading: a continuous run descriptor, such as run length as a function of gap or the peak of a strain state, would remove the threshold. This is not decided.
- The human set the direction this serves: a demand-response model of the hand-physiological gameplay state, seen from several views.

<a id="q-fable-views"></a>**Open for the human, from the memo, 2026-10-03.**

1. Run A1 as pre-registered, with key-count mods excluded, on the fit split, with one confirmatory run on calibration?
2. Define the response as P(miss | slack, relation, skill) plus the error distribution, in place of a cost law?
3. Which lane-to-finger binding is canonical? The memo proposes lanes 1 and 2 as the left middle and index fingers, and lanes 3 and 4 as the right index and middle fingers.
4. Contact ppy for a replay pilot after A1, as the memo proposes, or now, or not at all.

## Work

- Done 2026-10-03: corpus fail data with provenance and the estimate ([s-failtimes-fetched](#s-failtimes-fetched)).
<a id="d-skill-moves"></a>**Direction, human, 2026-10-03: a player's skill changes over time, so model the distribution of plays together with the maps played, not one fixed skill per player** ([private, local](private/human-inputs/4259953e-714c-4628-8911-eb1ec9fafdd8.md#prompt-4)). Agent reading: A1's skill term (log `rank_score` at the dump date as the prior mean of one latent skill per player, [p-prereg-a1](fable-player-data-views.md#p-prereg-a1)) needs amending before it runs. Skill becomes a state at a date, read from what the player plays and passes around that date. A1's attempts table holds undated lifetime counts. Successive monthly dumps can date attempts by differencing, if the same players recur. ppy's generator draws the random sample with `RAND(1)`, which suggests they do; this is not verified. The human also asked for a script that keeps adding player data. The agent reads that as the dump route only, not API harvesting.

<a id="w-dump-collector"></a>**Done, 2026-10-03:** a fresh Claude worker built the collector, code commit `2960027` (`scripts/player_data/`, `tests/player_data/`, 34 tests passing on the mac, job `20261003-142217-pdd-panel`). It collected every mania `random_10000` dump listed: 2026-04-01, 05-01, 07-13, 08-01 and 09-01. The SHA-256 of each is in `artifacts/player-data-dumps/<date>/manifest.json` on the mac. The licence text is unchanged; no tarball holds a licence file, so the site `LICENCE.txt` is the licence of record. The data takes 3.3 GB on the mac. A launchd agent `com.ensomi.player-dump-collector` runs daily at 13:15 UTC; to uninstall, run `sh scripts/player_data/launchd.sh uninstall` via `ens run`.

<a id="w-top-dumps"></a>**Running, 2026-10-03, at the human's instruction ([private, local](private/human-inputs/4259953e-714c-4628-8911-eb1ec9fafdd8.md#prompt-7)):** the six listed mania `top_10000` dumps (2026-04-01, 05-01, 06-01, 07-13, 08-01, 09-01, about 2 GB each) are downloading.

- The collector runs with `--include-top` as job `20261003-143207-player-top-collect` on bings-mac (dry run `20261003-143133-player-top-dryrun`). It downloads about 8 MB/s, one dump at a time, into `artifacts/player-data-dumps/<date>_top_10000/`.
- `top_1000` is a subset of `top_10000` of the same date and is not collected.
- The daily launchd agent still collects `random_10000` only: `launchd.sh` cannot pass `--include-top` without a small code edit, which has not been made.
- Whether the top samples follow the same players from month to month is to be measured by the panel report once the downloads finish.

<a id="s-not-a-panel"></a>**Observation, 2026-10-03: the monthly random dumps are not a panel.** Consecutive samples share 95 to 110 of 10,000 users. Non-consecutive pairs share 61 to 99, no user appears in all five, and 49,133 distinct users were seen in total. The main thread checked this in `artifacts/player-data-dumps/panel-report.json`, mirrored to the control plane. The fixed `RAND(1)` seed in ppy's script does not keep the same players. For the roughly 100 users per interval who repeat:

- attempt counts never decrease;
- new best scores are dated inside the interval in 91% to 100% of cases;
- the median pp change is 0;
- the summed per-map attempt increases run 2.2 to 5.1 times the change in the user's total playcount. This is unexplained, and it matters if attempts are used as exposure.

Agent reading of what this changes for [d-skill-moves](#d-skill-moves): differencing snapshots cannot date attempts for 99% of players. Monthly dumps add breadth, about 9,900 new players a month, not trajectories. Skill at a date has to come from each player's dated best scores within a dump, with survivorship bias: a best score survives only while it stays the best. A1's attempts stay undated lifetime counts. A top-by-pp dump may carry the same players from month to month; this is not measured. Six mania `top_10000` dumps are listed, about 2 GB each, including the only June 2026 mania dump. Collecting them is the human's decision.

- Done 2026-10-03: Fable review of the views ([a-fable-views](#a-fable-views)).
- Replays at scale wait for the human's decision to contact ppy; A1 waits for [q-fable-views](#q-fable-views).
