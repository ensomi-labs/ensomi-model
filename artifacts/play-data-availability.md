<!-- Provenance: written 2026-10-03 by a fresh general-purpose worker of main session 10fd41cc (Claude, control plane), from a brief asking what osu! player data is obtainable for 4K maps; copied unchanged from the session scratchpad. Probe jobs 20261003-play-data-probe and 20261003-play-data-probe2 ran on bings-mac (scripts in ~/ensomi/.sync/cp/jobs/<id>/; outputs in ~/ensomi/ensomi-model/artifacts/play-data-probe-20261003/, mirrored to the control plane except the replay file). The main thread re-read results.json and results-2.json for the failtimes shape, the replay download and its note matching, and the stable/lazer replay counts; they match the report. Code citations to ppy repositories were not re-checked. Agent reading: operator-properties.md#s-play-data. -->

# osu!mania 4K player data: what can be obtained, by which route, at what cost, under what terms

Date: 2026-10-03. Scope: replays and aggregate play statistics for ranked osu!mania 4K beatmaps.
Evidence labels: **[docs]** official documentation, **[code]** official source code (ppy repositories),
**[probe]** our live API calls on 2026-10-03, **[community]** third-party sources, **[inference]** my reading,
not stated by any source.

Code citations pin these commits:

| Repository | Commit |
| --- | --- |
| ppy/osu-web | `7d1c044b0acc34bdfb66b05d313f4ce109362820` (2026-10-02) |
| ppy/osu | `c834803ea9988159d67e1dc5d99cfaccada685d9` |
| ppy/osu-server-spectator | `1b73eb841825bdce679aa37a515415c42c77c3e1` |
| ppy/osu-queue-score-statistics | `7558b004f33cb79196b63682daa9862cb9c23faa` |
| ppy/osu-performance-datasets-generator | `1f50a36e7aa5352e69b67aa94f476421220c042e` |
| ppy/osu-infrastructure | `f6b8d24dccfc0b063741a7a9fd14c0ad0cdbda91` |

Headline: the owner expected that replays exist only for top players. That holds for scores set
in **stable** (top 1000 per beatmap). It does not hold for scores set in **lazer**. The server
keeps a replay for every passed lazer score on a ranked or loved beatmap while that score remains
the player's best for its mod combination. Client credentials can download those replays.
"Where players fail" exists as a 100-bin aggregate per beatmap, available both from the API and
in the monthly bulk dumps. It is coarse, mixes all players and all mods, and on newer maps
appears to be heavily sampled.

---

## 1. Aggregate per-beatmap data

**Points of Failure (`failtimes`).**

- The API exposes it. `failtimes` is a default include of the Get Beatmap, Lookup Beatmap and
  batch Get Beatmaps endpoints. [code] osu-web `app/Http/Controllers/BeatmapsController.php:26`
  sets `DEFAULT_API_INCLUDES = ['beatmapset.ratings', 'current_user_playcount', 'failtimes', 'max_combo', 'owners']`.
  The Get Beatmaps docblock (lines ~230-270) says "Includes `beatmapset` (with `ratings`), `failtimes`,
  `max_combo`, and `owners`" and caps a request at 50 ids. [docs] https://osu.ppy.sh/docs/index.html,
  sections "Get Beatmap" and "Get Beatmaps".
- Get Beatmapset (`GET /beatmapsets/{id}`) also includes `beatmaps.failtimes`. [code]
  `BeatmapsetsController.php:424,436`. [probe] All 3 beatmaps of set 1000065 came back with failtimes.
- Shape: `{"fail": int[100], "exit": int[100]}`. [docs] `resources/views/docs/_structures/beatmap.md:27-36`
  ("Array of length 100"). [code] Table `osu_beatmap_failtimes` is keyed by `(beatmap_id, type ∈ {fail, exit})`
  and has columns `p1..p100` (unsigned mediumint). See `app/Models/BeatmapFailtimes.php` and migration
  `database/migrations/2015_01_01_133338_create_beatmap_failtimes_table.php`.
- Meaning: lazer names `exit` "Retries". The docstrings read "Points of failure on a relative time
  scale (usually 0..100)" and "Points of retry on a relative time scale". [code] ppy/osu
  `osu.Game/Beatmaps/APIFailTimes.cs`. The exact binning (percent of total length, drain length,
  or last object time) is not documented anywhere I found. [inference] 100 equal fractions of map
  progress.
- **Mods are not separated.** The key has no mod or ruleset column. [code, as above] NoFail, DT and
  HT plays are mixed into one histogram. [inference]
- **Coverage.** The table accumulates over the beatmap's whole lifetime and has no time column.
  The writer is not in any public repository: osu-web only reads the table, and
  osu-queue-score-statistics (lazer processing) never writes it. Its `Beatmap` model throws
  `NotSupportedException` for `FailTimes`. [code] On stable, score submission carries a fail time
  `ft` and an exit flag `x`. [community] bancho.py `app/api/domains/osu.py:636-637`, commit
  `f10c03a3`.
- **Probe findings that qualify "all players":**
  - Bins on maps with beatmap id ≥ 3,083,501 are all multiples of 9 (31 of 31 such maps in the
    probe's 50-map sample). Older maps mix ordinary counts with multiples of 9; beatmap 2092272, for
    example, has bins 1 and 10. [probe] [inference] At some point (around 2021) recording switched
    to a 1-in-9 sample counted ×9. Effective sample size on a new map is therefore sum/9. One new 4K
    map with 6,040 plays has fail sum 63, which is 7 sampled fail events.
  - fails + exits + passcount ≈ playcount on older maps: ratios 0.83-0.99, and 0.987 for beatmap 2092272.
    On the newest maps the ratio drops to 0.42-0.77. [probe] [inference] lazer fails and quits are
    probably not recorded, while lazer passes do count toward `passcount`. lazer's
    `PlayCountProcessor` updates `osu_beatmaps.passcount` and playcount ([code]
    osu-queue-score-statistics `Processors/PlayCountProcessor.cs`), but nothing in the lazer pipeline
    writes failtimes.

**Other aggregates.**

- Per beatmap: `playcount` and `passcount` (all players, lifetime, no mod split) and `max_combo`.
  [code] [probe] The dataset's `metadata.json` files already contain these.
- `POST /beatmaps/{id}/attributes` with `ruleset=mania` returns only `star_rating` and `max_combo` for
  mania. [probe] Response: `{"star_rating": 4.4925, "max_combo": 4049}`.
- There is no endpoint for score or accuracy distributions. The nearest substitutes:
  - The beatmap leaderboard: at most 100 scores via the API ([code] `config/osu.php:52`
    `max_scores => 100`; `app/Libraries/Score/BeatmapScores.php:22` clamps `limit` to it) plus a
    `score_count` total ([probe] 7,437 on beatmap 2092272).
  - Per-score judgement counts and pp on every score object.
- The API's mod-filtered leaderboards need no osu!supporter. The check is
  `!empty($mods) && !is_api_request() && !$isSupporter` at `BeatmapsController.php:42`. Country and
  friend leaderboards need a supporter user token (`ScoreSearchParams::SUPPORTER_TYPES`). [code]

## 2. Replays

**Endpoint.** `GET /api/v2/scores/{score_id}/download`. A legacy form also exists:
`GET /api/v2/scores/{ruleset}/{legacy_score_id}/download`. [code] `routes/web.php:557-559`,
`ScoresController::download` at lines 46-110. The endpoint has no docblock and does not appear in the
public API documentation (as of the 2026-10-03 docs page). [docs, absence]

**Auth.**

- The controller exempts `download` from `auth` (lines 24-28) and requires only scope `public`. A
  user is required only for non-API web requests (lines 48-51). [code] So a client-credentials token
  (no user) with scope `public` is accepted. [probe] HTTP 200 and a 6,494-byte `application/x-osu-replay`.
- Downloads by a user token from a password client increment replay-watch counters. Client-credential
  downloads do not. [code] lines 70-98.
- API v1 `/api/get_replay` (API key, 10 requests per minute, "not intended for batch retrievals")
  still documented. [docs] https://github.com/ppy/osu-api/wiki, section "Get replay data".

**Rate limits.**

- Docs: "no more than 60 requests per minute". [docs] `resources/views/docs/info.md.blade.php:34`
- Server limits: 1,200 per minute global and 10 per minute for score downloads. [code]
  `config/osu.php:35-36`. [probe] Response headers `X-RateLimit-Limit: 1200` on API calls, `10` on
  the download, `60` on `/oauth/token`.

**Which scores have a stored replay.**

- **stable.** "Server replays are reserved for the top 1000 plays in the Global leaderboard of a
  difficulty/beatmap … the previous holder for #1000 position's server-side replay will be removed."
  [docs] https://osu.ppy.sh/wiki/en/Gameplay/Replay, section "Server".
  - [probe] On beatmap 2092272 all top 100 scores are stable scores with `has_replay=true`.
  - [probe] In a sample of 398 recent stable mania passes, 135 had replays (34%). Global ranks:
    945 (loved map) and 1,151 had replays; 2,510 had none. So a top-1000 cut-off holds roughly.
  - [inference] The rank 1,151 case may reflect `rank_global` counting lazer and stable scores together.
- **lazer.**
  - Passed scores on beatmaps with status Ranked through Loved are uploaded to S3 and marked
    `has_replay = 1`. [code] osu-server-spectator `Hubs/Spectator/SpectatorHub.cs:32-37,356-377`
    (status window; "Do nothing with scores on unranked beatmaps"); `Hubs/ScoreUploader.cs:64-67,114-118`
    (drops `!dbScore.passed`). [docs-ish] ppy/osu-infrastructure `score-submission.md` diagram: "Upload
    replay (ScoreUploader.Flush) … Mark has replay".
  - Failed plays get no replay. [probe] One user's 50 recent plays included 23 fails, none with a replay.
  - **Retention.** A lazer score stays `preserve=1` while it is the user's high for that beatmap and
    mod combination (by total score or pp), or is pinned, or belongs to multiplayer. Superseded scores
    are marked `preserve=0` and deleted with their replays after 2 days. [code]
    osu-queue-score-statistics `Commands/Maintenance/MarkNonPreservedScoresCommand.cs`
    (`CheckIsUserHigh`); `DeleteNonPreservedScoresCommand.cs:16` (`preserve_days = 2`, deletes replays).
  - [probe] 529 of 602 recent lazer mania passes had `has_replay=true` (88%). The cause of the
    remaining 12% is unknown; one of them was ranked #172.
- **Can non-top players' replays be obtained?** Yes, for lazer scores. [probe] Score 7617943299
  (lazer, 4K ranked map, `rank_global` = 23,674) downloaded with client credentials.

**Discovering non-top score ids.**

| Route | Coverage | Notes |
| --- | --- | --- |
| `GET /api/v2/scores?ruleset=mania` | Up to 1,000 of the most recent passed scores from all players, stable and lazer | Cursor `cursor_string`; a cursor older than 10,000,000 score ids is refused. [code] `ScoresController.php:116-190`, `config/osu.php:209`. [probe] 1,000 mania passes spanned about 6 minutes (~165 per minute), 602 lazer and 398 stable, 718 distinct users. |
| `GET /users/{id}/scores/{best\|recent\|firsts}` | One player's scores | `recent` accepts `include_fails`. |
| `GET /beatmaps/{id}/scores/users/{user}/all` | One player's scores on one beatmap | |
| data.ppy.sh `scores` table | 10k sampled users | Carries `has_replay`; see section 3. |

**Replay format (mania).**

- `.osr` header (mode, version, beatmap MD5, judgement counts, score, combo, mods, life bar,
  timestamp), then an LZMA stream of frames `w|x|y|z` where `w` is the delta in integer ms. [docs]
  https://osu.ppy.sh/wiki/en/Client/File_formats/osr_(file_format)
- For mania, `x` carries the held-column bitmask. [code] ppy/osu
  `osu.Game.Rulesets.Mania/Replays/ManiaReplayFrame.cs` (`FromLegacy`: `activeColumns = (int)MouseX`).
- lazer rounds frame times to integer ms ("stable could only parse integral values"). From version
  30000001 it appends compressed JSON score info (statistics, mods, pauses, client version). [code]
  `osu.Game/Scoring/Legacy/LegacyScoreEncoder.cs`
- **Per-column press and release times are recoverable at 1 ms resolution.** [probe] Replay 7617943299:
  - 6,886 frames, key states 0-15 (4 columns), median frame delta 17 ms, minimum 1 ms.
  - Presses per column 262/282/262/271, each with an equal number of releases.
  - Matched against the local chart (`dataset/0/1213119/... [Another].osu`), 1,020 of 1,022 notes paired
    within ±200 ms. Press offset mean −15.0 ms, SD 22.6 ms. Hold-release offset mean −34 ms over 49 holds.
  - The trailer carried lazer statistics and `pauses`.

## 3. Bulk dumps (data.ppy.sh)

**Files.**

- Listing at https://data.ppy.sh (fetched 2026-10-03): monthly `YYYY_MM_DD_performance_{osu,taiko,catch,mania}_{top_1000,top_10000,random_10000}.tar.bz2`
  from 2026-04-01 to 2026-09-01, plus `..._osu_files.tar.bz2`. June 2026 has only `mania_top_10000`
  for mania. Stray root files dated 2026-06-02: `osu_scores_mania_high.sql` (174 MB),
  `osu_user_beatmap_playcount.sql` (154 MB), `osu_user_stats_mania.sql`, `sample_users.sql`,
  `scores.sql` (3 KB).
- Sizes (HEAD, 2026-09-01 mania): `random_10000` 447,544,515 B; `top_1000` 522,670,193 B;
  `top_10000` 1,915,327,520 B; `osu_files` 1,431,349,366 B.

**Tables** ([code] `dump_sample_tables.sh` in ppy/osu-performance-datasets-generator):

- `sample_users`: top N users by `rank_score`, or N random users with `rank_score > 0` drawn with
  `RAND(1)`. Restricted and warned users removed.
- `osu_scores_mania_high`: stable best scores of sample users, with judgement counts, mods, pp and a
  `replay` flag ([code] osu-web `app/Models/Score/Best/Model.php:32`).
- `osu_user_stats_mania`.
- `osu_user_beatmap_playcount`: per user and beatmap play counts, i.e. attempts.
- `scores`: lazer-era table for sample users, filtered by `preserve = 1 AND ranked = 1 AND ruleset_id = 3`.
  It includes `has_replay`, `data` (statistics and mods), pp and timestamps.
- `osu_beatmapsets` and `osu_beatmaps` (playcount, passcount): all ranked, approved and loved maps,
  **all modes**.
- **`osu_beatmap_failtimes` for all ranked, approved and loved beatmaps.** This dump line has been in
  the script since the initial commit of 2022-06-29.
- `osu_beatmap_difficulty`, `osu_beatmap_difficulty_attribs` (mania only),
  `osu_difficulty_attribs`, `osu_beatmap_performance_blacklist`, `osu_counts`.

**Replays are not included.** No replay table or `.osr` files are dumped. [code]

**Licence** ([docs] https://data.ppy.sh/LICENCE.txt): "All data provided here is done so with the
intention of it being used for statistical analysis and testing osu! subsystems. Permission is NOT
implicitly granted to deploy this in production use of any kind. Should you wish to publicly
use/expose the data provided here, please contact me first at contact@ppy.sh." The generator README
says the scripts are "for internal use only. Please find the dumps on https://data.ppy.sh".

**Not verified.** I did not download a tarball, so the presence of `osu_beatmap_failtimes` inside the
current files is backed by the script, not by inspection.

## 4. Third-party and paid routes

- **Public mania replay datasets: none found.**
  - Hugging Face API searches (`osu`, `osr`, `replays`, `mania`, `osumania`) returned beatmap
    datasets only, for example `Tiger14n/osumaps19866` and `project-riz/osu-beatmaps`, and
    `NCYG/MusicGameGeneration_osu_mania` (charts). No replays.
  - Kaggle: the only replay dump found is "o!rdr osu standard replay dump"
    (kaggle.com/datasets/skihikingkevin/ordr-replay-dump), which is osu!standard. I did not search
    Kaggle beyond web search. [community]
- **Academic.**
  - eve-ning's *opal* (PyPI `opal-net`) predicts mania accuracy from "top 10K mania users data from
    https://data.ppy.sh", about 10M scores. It is score-level, not replays. [community]
  - Chart-generation papers on 4K (e.g. arXiv 2311.13687) use charts, not replays.
  - I found no paper that uses osu!mania replays.
- **Community collectors** (e.g. AutOsu) download osu!standard replays using a browser session
  cookie. [community] This is not a route to rely on.
- I did not investigate community score trackers (osu!track, Osekai, respektive's tools) in depth.
  None surfaced offering mania replays or fail data.
- **osu!supporter.** Perks are cosmetic plus "Extended leaderboards" (per-mod, country and friend
  leaderboards), osu!direct and higher limits. [docs] https://osu.ppy.sh/wiki/en/osu!supporter,
  section "Extended leaderboards". It grants no replay or bulk data access. The API already allows
  mod-filtered leaderboards without it (section 1).
  - Price: the wiki gives none ("All prices are in United States dollars"). osu-web carries a
    reference table marked "currently unused": USD 4 for 1 month, 8 for 2, 16 for 6, 26 for 12
    months. [code] `app/Models/SupporterTag.php:57-84`
- **Official or paid data access.** No published research or commercial data programme was found.
  The only stated contacts:
  - API terms: "If in doubt, ask (mailto:pe@ppy.sh) before serious long term use".
  - data.ppy.sh licence: contact@ppy.sh for public use or exposure.
  - The API is free ("Providing this API is done for free"). [docs] OAuth clients are created free in
    account settings. The project already has one.
  - **Nothing found costs money**, except osu!supporter, which does not help.

## 5. Terms

**API terms of use** ([docs] osu-web `resources/views/docs/info.md.blade.php:15-36`; live at
https://osu.ppy.sh/docs/index.html#terms-of-use):

- "Use the API for good. Don't overdo it. If in doubt, ask before serious long term use."
- Listed as incorrect or abusive:
  - "Using the API as if it is your database, re-requesting the same data every time it is needed"
  - "**Harvesting mass score/user/beatmap data (consider using data.ppy.sh instead if looking for
    seed or sample data)**"
  - "Using the API to try to gain a competitive advantage"
- "limit your usage to no more than 60 requests per minute … exceeding this specified limit may lead
  to your API tokens being revoked".

[inference] Bulk replay collection through `/scores/{id}/download`, or continuous `/scores` stream
harvesting, falls under "mass score data" and needs ppy's agreement first. Fetching failtimes for
some thousands of maps is borderline; the dump is the sanctioned route.

**Terms of Service** (last updated 8 March 2021; [docs] https://osu.ppy.sh/wiki/en/Legal/Terms,
section "PROPRIETARY RIGHTS TO CONTENT"): "User may not copy, reproduce, distribute, or create
derivative works from this Content without expressly being authorised to do so by the Service." The
ToS has no clause on the API, scraping, data mining or ML.

**Privacy policy** (updated 24 April 2026; [docs] https://osu.ppy.sh/wiki/en/Legal/Privacy, section
"On playing the game and submitting a score"): "When completing a game session (passing or failing a
beatmap), details on your performance will be automatically submitted … The scoring portion of this
submission includes game replay data and may be displayed publicly".

**Machine learning.** No source found mentions ML training or evaluation, explicitly or otherwise.

- Internal statistical modelling sits within the dump licence's stated intent ("statistical
  analysis").
- Shipping a model or evaluator as a product could count as "production use of any kind", which
  needs permission.
- Redistributing dumps, replays or derived per-player data needs permission (dump licence and ToS).
- [inference] Asking ppy (pe@ppy.sh for API volume, contact@ppy.sh for dump use) is the only route to
  clear ML use and replay volume.

## 6. Live probe

**Setup.**

- Credentials: `~/.config/ensomi/osu-api.env` on bings-mac with `OSU_CLIENT_ID` and
  `OSU_CLIENT_SECRET`; client-credentials grant with scope `public`. Source:
  `.sync/cp/jobs/20261002-114628-dataset-enrich/brief.md`. No osu! API code exists in the tracked
  repositories; the earlier acquisition script lived outside them. The scripts never print the
  values or the token.
- Existing `metadata.json` files ([probe] `ens cat` on two files): both carry per-beatmap
  `playcount`/`passcount`.
  - `dataset/0/1000065/metadata.json` (fetched 2026-08-05) has **no** `failtimes`. Its key set looks
    like a search-endpoint record despite `api_endpoint: .../beatmapsets/1000065`.
  - `dataset/0/1002446/metadata.json` (fetched 2026-10-02) has `failtimes` (100 + 100 bins) for every
    beatmap.
  - The index has 4,201 files dated 2026-08-05, 12 dated 2026-09-08 and 2,666 dated 2026-10-02.
    [inference] Roughly the 2026-10-02 generation carries failtimes, so failtimes for about 2/5 of
    the sets are already on disk. I checked only one file per generation.

**Jobs.** Both ran on the mac via `ens run ensomi-model`. Scripts are in `.sync/cp/jobs/<id>/`;
outputs are in `ensomi-model/artifacts/play-data-probe-20261003/`.

| Job id | Requests | What it did | Output |
| --- | --- | --- | --- |
| `20261003-play-data-probe` | 14 | Token. Get Beatmap / Lookup / batch Get Beatmaps / Get Beatmapset on beatmap 2092272 (Billx - Punishment (Dustvoxx Remix) [4K Equalizer], ranked). Difficulty attributes. Leaderboard top 100 (new format, `legacy_only=1`, old format). `/scores?ruleset=mania`. Batch lookup of 50 stream beatmaps. One score detail. One user's recent plays with fails. **One replay download.** | `results.json`, `requests.json`, `replay-7617943299.osr` (6,494 B, mac only; not mirrored) |
| `20261003-play-data-probe2` | 6 | Token. `/scores?ruleset=mania` broken down by stable or lazer × `has_replay`. Four score details for `rank_global` | `results-2.json` |

Total: 20 requests, all HTTP 200, no auth refusal. Results are in sections 1-2.

- Beatmap 2092272: playcount 145,297, passcount 23,716, fail sum 33,220, exit sum 86,477.
- The download was a non-top lazer score (global #23,674) and succeeded with client credentials.

**What failed or is limited.**

- I could not test whether a cursor-paged `/scores` history reaches back days, or any bulk behaviour.
  Both were out of scope by design.
- I did not try a second replay (stable top score) because of the one-replay limit. Its availability
  rests on `has_replay=true` plus the same code path.

## Summary table

| Data | Route | Granularity | Who it covers | Cost | Terms | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| Fail and exit histogram per beatmap | API `GET /beatmaps/{id}`, `/beatmaps?ids[]` (≤50), `/beatmaps/lookup`, `/beatmapsets/{id}` | 100 bins over relative map progress; no mod split; lifetime cumulative | Apparently stable plays only (inference); newer maps sampled 1-in-9 ×9 (inference) | Free; ≤60 req/min | API ToU; mass harvesting discouraged | code + docs + probe |
| Same, all ranked maps at once | data.ppy.sh `osu_beatmap_failtimes` in any monthly mania dump (smallest: random_10000, 448 MB) | As above | All ranked, approved and loved beatmaps, all modes | Free | Statistical analysis; production or public use needs contact@ppy.sh | code (script); tarball not inspected |
| playcount, passcount, max_combo, star rating | API (as above), `POST /beatmaps/{id}/attributes`; dumps `osu_beatmaps`, `osu_beatmap_difficulty_attribs` | Per beatmap, lifetime | All players (lazer passes included) | Free | As above | code + probe |
| Top scores with judgement counts and pp | API `GET /beatmaps/{id}/scores` (≤100, mod filter allowed) | Per score | Top 100 per beatmap (+ `score_count`) | Free | API ToU | code + probe |
| Best scores and attempt counts for sampled players | data.ppy.sh `osu_scores_mania_high`, `scores` (lazer), `osu_user_beatmap_playcount` | Per user × beatmap (best score; attempt count) | Top 1k, top 10k and random 10k mania players | Free (0.45-1.9 GB per dump) | Dump licence | code |
| Stream of recent passes | API `GET /scores?ruleset=mania` | Per score (stats, mods, pp, has_replay) | All players, passes only, ~165 mania passes per minute | Free | ToU: no mass harvesting | code + probe |
| Recent plays incl. fails for a player | API `GET /users/{id}/scores/recent?include_fails=1` | Per play, no replay for fails | Any player, recent window | Free | API ToU | code + probe |
| stable replays | API `GET /scores/{id}/download` (or `/scores/mania/{legacy_id}/download`), client credentials OK | Per-frame key bitmask, 1 ms | Top ~1000 scores per beatmap | Free; 10 downloads/min server limit | ToU: ask before bulk; ToS: no redistribution | docs (wiki) + code + probe (flags) |
| lazer replays | Same endpoint | Per-frame key bitmask, 1 ms, plus pauses and statistics trailer | Every player's best passed score per beatmap × mod combo on ranked or loved maps (~88% flagged); no fails | Free; 10/min | Same | code + probe (downloaded non-top #23,674) |
| Score ids with replay flag for sampled players | data.ppy.sh `scores.has_replay`, `osu_scores_mania_high.replay` → then the download endpoint | Per score | Sampled 10k users | Free | Dump licence + API ToU | code |
| Third-party mania replay datasets | none found | n/a | n/a | n/a | n/a | absence (HF API, web search) |
| Paid or official data access | none published; contacts pe@ppy.sh, contact@ppy.sh | n/a | n/a | No price stated anywhere | Permission by email | docs |
| osu!supporter | store | Country/friend leaderboards only | n/a | Code reference (unused): USD 4/month to 26/year | n/a | docs + code |

## Open gaps

1. **Failtimes semantics.** The bin definition, the writer (closed source), the 1-in-9 sampling and
   lazer exclusion are inferred from the data, not from any document. A ppy answer, or inspecting
   `osu_beatmap_failtimes` in a dump against `osu_beatmaps.playcount`, would settle the data side.
2. **Dump contents.** The September 2026 tarball has not been opened, so table presence and the
   columns of `scores` (including `has_replay`) are taken from the script and schema.
3. **Permission for ML use and replay volume.** No source addresses it. Only ppy can clear it. The
   limits are 10 downloads per minute on the server and 60 requests per minute in the ToU, and the
   ToU discourages bulk without asking.
4. **Scale of available 4K lazer replays.**
   - Unknown how many distinct 4K maps and players have preserved lazer replays.
   - lazer was 60% of recent mania passes. In the stream batch, 37 of 50 beatmaps were 4K ranked.
   - The 12% of lazer passes without replays is unexplained.
5. **Stable cut-off detail.** A stable score at global #1,151 had a replay. Whether the rule is top
   1000 at submission time, or `rank_global` counts differently, is unresolved.
6. **Failed and quit plays.** No replay or per-note data exists for them in either client. Only the
   aggregate histogram and per-play judgement counts in recent plays remain. Fatigue signals would
   have to come from passes (within-play timing drift) and from play sequences (timestamps).
   [inference]
7. **Kaggle and community trackers.** Searched only via web search; not exhaustive.
