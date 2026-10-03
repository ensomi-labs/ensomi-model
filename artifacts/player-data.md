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

## Work

- Fail data for the corpus is being fetched with provenance by a fresh worker, together with an estimate of how much player data would be useful and its disk footprint (started 2026-10-03; output on bings-mac under `~/ensomi/ensomi-model/artifacts/player-data-20261003/`).
- Replays at scale wait for the human's decision to contact ppy.
