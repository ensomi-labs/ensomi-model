<!-- Provenance: written 2026-10-03 by a fresh general-purpose worker of main session 10fd41cc, from a brief to estimate useful player data and fetch corpus fail data with provenance; copied unchanged from the session scratchpad. Jobs on bings-mac: 20261003-player-data-failtimes, -postprocess, -readme, -inspect2. Outputs: ~/ensomi/ensomi-model/artifacts/player-data-20261003/ (failtimes/ mirrored to the control plane except Parquet; dump/ mac only). The main thread re-read failtimes/manifest.json, coverage.json and README.md; counts, hashes and coverage match this report. Agent reading: player-data.md#s-failtimes-fetched. -->

# Player data: size estimate and fail-data fetch for the R2 corpus

Date: 2026-10-03. Worker of session 10fd41cc (Claude, control plane). All compute and downloads ran on
bings-mac through `ens run`. I downloaded no replays and contacted no one. I edited no tracked
files and made no commits.

Evidence labels:

- **[measured]**: this work's jobs.
- **[probe]**: the earlier probe jobs, `20261003-play-data-probe*`.
- **[code]**: ppy source at a pinned commit.
- **[inference]**: my reading; no source states it.

## Outcome

- **Fail data.** The table covers every ranked and loved chart in the corpus: 21,871 of the
  21,975 corpus charts that have a beatmap id.
- **Route.**
  - Mostly the data.ppy.sh dump `2026_09_01_performance_mania_random_10000.tar.bz2` (448 MB,
    kept on the mac).
  - The API covered the 388 ids the dump lacks: 11 requests, all HTTP 200.
- **Size.** The table is 6.2 MB as Parquet. Replays for a plausible response study come to
  roughly 6 MB (pilot), 45 MB (core) or 300 MB (full).
- **Wall time.** 1 h, 7.5 h or 50 h at 10 downloads per minute. Any of these needs ppy's
  agreement first.
- **Disk.** The mac has 328 GiB free.

---

## Task A: estimate

### A1. Fail data for the corpus [measured]

| Item | Value |
| --- | --- |
| Corpus charts | 22,066 |
| With a beatmap id | 21,975 rows (21,971 distinct ids): 21,853 with the corpus `api_beatmap_id`, 122 with only the `.osu` file `BeatmapID` |
| Without any id | 91 (all have corpus `api_status` None). Fail data is impossible for them |
| Ranked | 19,147 rows, all covered (18,997 dump, 150 API) |
| Loved | 2,632, all covered (2,584 dump, 48 API) |
| Other (graveyard 42, wip 25, pending 7) | 74, all covered by the API (the dump holds only ranked, approved, loved and qualified maps) |
| No API match in the corpus (file id only) | 122, of which 18 covered (3 dump, 15 API). The API returns nothing for the other 101 ids (104 rows) |
| Version check | 20,545 covered rows have the source checksum equal to the corpus chart's MD5. 1,326 do not, and their counts refer to another version of the chart |
| Parquet (zstd, 25 columns incl. 2 × 100 bins) | 6,207,525 B for 21,975 rows, about 282 B/row. Snappy: 8.57 MB |
| JSON, all columns, no indentation | 34.3 MB, about 1.56 KB/row. gzip: 6.7 MB |
| JSON, minimal (`beatmap_id`, `playcount`, `passcount`, `fail`, `exit`) | 18.2 MB for 21,824 rows, about 834 B/row |
| All ranked maps, all modes, if wanted | 234,868 beatmaps with failtimes in the dump. About 66 MB as Parquet at 282 B/row [inference]. The raw SQL member is 142.5 MB |

Assumptions:

- JSON sizes were measured in memory with `json.dumps`. No JSON copy of the table was written.
  A file over 4 MB would break the mirror.
- The id rule is `api_beatmap_id` when set, otherwise the file `BeatmapID` when it is > 0.

### A2. data.ppy.sh mania dumps

Sizes come from HTTP HEAD on 2026-10-03 [measured]. The listing's newest dump is 2026-09-01; no
2026-10-01 dump was posted yet.

| Dump date | random_10000 | top_1000 | top_10000 | osu_files (all modes) |
| --- | ---: | ---: | ---: | ---: |
| 2026-04-01 | 445.5 MB | 532.6 MB | 2,145.8 MB | 1,336.6 MB |
| 2026-05-01 | 443.8 MB | 541.2 MB | 2,182.1 MB | 1,352.7 MB |
| 2026-06-01 | none | none | 2,222.3 MB | none |
| 2026-07-13 | 473.4 MB | 528.3 MB | 2,062.2 MB | 1,406.1 MB |
| 2026-08-01 | 453.0 MB | 515.4 MB | 1,896.3 MB | 1,416.0 MB |
| 2026-09-01 | 447.5 MB (3.37 GB uncompressed) | 522.7 MB | 1,915.3 MB | 1,431.3 MB |

Stray root files dated 2026-06-02:

- `osu_scores_mania_high.sql`: 173.9 MB.
- `osu_user_beatmap_playcount.sql`: 154.4 MB.
- `osu_user_stats_mania.sql`: 1.8 MB.
- `sample_users.sql`: 0.27 MB.
- `scores.sql`: 3 KB.

Contents of the 2026-09-01 `random_10000` dump [measured]. Every member was parsed and counted
against the corpus ids:

| Table | Size | Rows | What matters here |
| --- | ---: | ---: | --- |
| `osu_beatmap_failtimes` | 142.5 MB | 466,828 (231,991 fail, 234,837 exit) | `(beatmap_id, type, p1..p100)` for 234,868 beatmaps, all modes. Covers 21,932 of the 21,949 4K mania maps in `osu_beatmaps` |
| `osu_beatmaps` | 56.6 MB | 235,061 | `checksum`, `last_update`, `playcount`, `passcount`, `approved`, `diff_size`. Mania: 26,020 ranked, 4,821 loved, 110 qualified |
| `osu_beatmapsets` | 46.2 MB | 60,659 | Set metadata |
| `osu_beatmap_difficulty`, `osu_beatmap_difficulty_attribs` | 1,030 + 1,176 MB | 20.8 M + 41.7 M | Star rating and attributes per beatmap × mods, all modes. Covers 21,600 corpus ids |
| `osu_scores_mania_high` (stable best scores) | 190.4 MB | 1,810,274 from 7,024 of the 10,000 sampled users | 1,236,864 on 20,963 corpus maps, with judgement counts, mods and pp. 284,743 of them carry the `replay` flag, spread over 20,960 corpus maps |
| `scores` (lazer-era table; preserve, ranked, mania) | 563.2 MB | 1,614,224 from 10,000 users | 1,061,818 on 21,293 corpus maps. 372,751 have `has_replay`, spread over 21,291 corpus maps; mean 17.5 per map |
| `osu_user_beatmap_playcount` | 160.8 MB | 7,789,903 | Attempts per user × map: 1,439,692 rows on 21,550 corpus maps, 8,148,321 attempts |
| `osu_user_stats_mania` | 1.8 MB | 10,000 | pp (`rank_score`), accuracy, playcount, `fail_count`, `exit_count`, play time |
| `sample_users` | 0.27 MB | 10,000 | The sample |

Notes on the table:

- The `scores` and `osu_scores_mania_high` counts may overlap, because `scores` can hold
  imported stable scores; I did not separate them.
- Replay flags describe 2026-09-01 and may be stale since [inference].

**Is keeping a dump worthwhile? Yes, the `random_10000` one, and it is kept.**

- At 448 MB it is 0.13% of free disk.
- It is the sanctioned source for failtimes, playcount and passcount of every ranked map.
- It is an unbiased random player sample with best scores, attempts per map and per-user
  fail and quit counts. That is enough for a score-level response model (accuracy and pass
  versus star rating by skill) without any API harvesting.
- Its `scores.has_replay` rows are a ready sampling frame of replay-bearing score ids on corpus
  maps.

The other dumps:

- `top_10000` (1.9 GB) adds strong players. It mainly helps hard charts, so I would skip it
  for normal charts.
- `osu_files` (1.43 GB) holds the ranked `.osu` files. It could supply the chart versions that
  the counts actually describe for the 1,326 checksum-mismatch rows [inference; not opened].
- Monthly re-downloads are only needed for a time series.

### A3. Replays (estimate only; no downloads)

Basis [probe]: one lazer replay, score 7617943299:

- 6,494 B with 6,886 frames spanning 98.4 s, on a 1,022-note chart.
- That is **3.96 KB per minute of chart** (66 B/s), **6.35 B per note** and 0.94 B per frame.
  Frames are LZMA-compressed inside the `.osr`.

Assumptions:

- lazer records a frame on every input change and otherwise at most 60 frames per second
  [code: ppy/osu `c834803e` `osu.Game/Rulesets/UI/ReplayRecorder.cs:33,80`]. Size therefore
  scales mainly with duration; the probe replay shows 70 frames/s.
- Stable replays were not measured. I assume they are within 2× per minute.
- Score metadata from the API carries no replay size, so extra leaderboard entries would not
  refine this. I fetched none.
- Per corpus chart: mean length 151.9 s gives **about 10 KB per replay**. The median (128 s) gives
  8.4 KB and the 90th percentile (246 s) gives 16 KB. The per-note basis (mean 1,500 notes)
  gives 9.5 KB.
- Compression: `.osr` frames are already LZMA, so zstd or xz on top saves under about 10%. A
  decoded per-note table (note, column, press and release offsets) is about 6–9 KB per replay
  in Parquet. In practice raw ≈ compressed.

Design for a response study:

- **Question.** How local demand relates to timing error and misses, conditional on skill.
- **Unit.** A passed play. Fails have no replay, and the failtimes histograms cover fails.
- **Map strata.** Star bands 1–6 (floor of star rating) over ranked or loved 4K corpus maps.
  Bands 0, 7 and 8 are thin and outside the normal-chart range.
- **Maps per band.** Chart-level effects are the target, so maps should be numerous: 30 per band
  in the core design. A band mean's standard error is then about 0.18 of the between-map SD
  (1/√30), which is enough for a mixed model with map random effects and a few chart
  covariates.
- **Skill strata.** 4 bands by user mania pp (quartiles of the dump sample).
- **Replays per map.** A map is passed by roughly 2–3 of the 4 bands, at 10 replays per band:
  about 25 per map.
- **Players.** Draw from a fixed panel (the dump's random users), so each player appears on
  several maps and player and map effects separate.
- **Availability.** The random sample has a mean of 17.5 replay-flagged lazer-table scores per
  corpus map. The core design therefore needs maps with at least 25 flagged scores (popular
  maps), or the `top_10000` dump, or per-map leaderboards (top 100 only) to reach 25 per map.

| Tier | Design | Replays | Raw ≈ compressed | Wall time at 10/min |
| --- | --- | ---: | ---: | ---: |
| Pilot | 6 bands × 5 maps × 20 | 600 | 6 MB | 1.0 h |
| Core | 6 bands × 30 maps × 25 | 4,500 | 45 MB | 7.5 h |
| Full | 6 bands × 100 maps × 50 | 30,000 | 300 MB | 50 h (2.1 days) |

- Discovery can come from the dump's `scores` table at zero API cost, so only downloads count
  against the server's 10 per minute limit.
- Some score ids from 2026-09-01 will have lost their replay since: superseded lazer scores are
  deleted after 2 days [probe report]. The share is unknown.
- **Terms.** Collection at any tier is "harvesting mass score data" under the API terms and
  needs ppy's agreement first (pe@ppy.sh). Storage is not a constraint.

### A4. Free disk on the mac [measured]

`df -h ~/ensomi` on bings-mac: `/dev/disk3s5` (`/System/Volumes/Data`) is 926 GiB in size with
557 GiB used and **328 GiB available** (63%). The job saw 352.2 GB free at start. Since then the
tarball (0.45 GB) and the table files (about 13 MB) were added.

---

## Task B: fail data with provenance

### Route and jobs

| Job id | Role | Result |
| --- | --- | --- |
| `20261003-player-data-failtimes` | Fetch: listing and licence check, dump download, one-pass tar/bz2 parse, API fallback and validation, table, comparisons, manifest | Exit 0 at 10:33:02–10:38:14Z |
| `20261003-player-data-postprocess` | No network. Fills absent type rows with zeros (osu-web semantics), recomputes coverage, adds the analyses in `postprocess.json`, adds a `postprocess` block to the manifest | Exit 0 |
| `20261003-player-data-readme` | Copies `README.md` into the output and records its hash in the manifest | Exit 0 |
| `20261003-player-data-inspect2` | Read-only corpus and disk inspection for Task A | Exit 0 |
| `20261003-player-data-inspect`, `-plan`, `-plan-cleanup` | Failed first inspect, dry run, cleanup (see Failures) | Not used for outputs |

**Route 1: data.ppy.sh**, 21,584 rows.

- The licence text matched the earlier report's quotes.
- Dump identity:

  | Field | Value |
  | --- | --- |
  | URL | `https://data.ppy.sh/2026_09_01_performance_mania_random_10000.tar.bz2` |
  | Size | 447,544,515 B |
  | Last-Modified | 2026-09-01 11:59:38 GMT |
  | ETag | `"c33c5f17f39b071954873eb9098b2afa-54"` |
  | SHA-256 | `2fa58ed17417deb9d395a36bdff97fa53d19681675622a32c7bd9f3c03cad81e` |
  | Failtimes snapshot | "Dump completed on 2026-09-01 5:51:26" |

- Downloaded at 10.8 MB/s and streamed once in 244 s.
- The tarball is kept on the mac at
  `~/ensomi/ensomi-model/artifacts/player-data-20261003/dump/2026_09_01_performance_mania_random_10000.tar.bz2`,
  since it is under 2 GB.

**Route 2: API** `GET /api/v2/beatmaps?ids[]=`, 287 rows.

- 11 requests: 1 token, 8 for the 388 ids the dump lacks, and 2 for a 100-id validation sample.
- All HTTP 200, spaced at least 2 s apart. No 429 and no errors.
- The cap of 600 was never approached.

### Coverage

By corpus `api_status`:

| Corpus `api_status` | Rows with id | With histograms | Dump | API | Checksum match |
| --- | ---: | ---: | ---: | ---: | ---: |
| ranked | 19,147 | 19,147 | 18,997 | 150 | 18,066 |
| loved | 2,632 | 2,632 | 2,584 | 48 | 2,398 |
| graveyard | 42 | 42 | 0 | 42 | 39 |
| wip | 25 | 25 | 0 | 25 | 22 |
| pending | 7 | 7 | 0 | 7 | 7 |
| None (no API match) | 122 | 18 | 3 | 15 | 13 |
| **Total** | **21,975** | **21,871** | **21,584** | **287** | **20,545** |

Ranked and loved by star band (floor of corpus `api_star`). Every row has histograms.

| Band | Rows | From API | Checksum match | All bins ×9 | Median playcount | Median fail sum | x9 rows with < 10 sampled fails |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 197 | 4 | 191 | 158 | 5,959 | 522 | 36 |
| 1 | 4,630 | 36 | 4,386 | 3,042 | 9,154 | 909 | 368 |
| 2 | 5,240 | 33 | 4,918 | 3,602 | 16,606 | 1,200 | 254 |
| 3 | 4,931 | 37 | 4,621 | 3,369 | 23,221 | 1,800 | 135 |
| 4 | 3,960 | 33 | 3,718 | 2,607 | 22,725 | 2,061 | 102 |
| 5 | 1,830 | 21 | 1,702 | 1,213 | 18,388 | 1,854 | 93 |
| 6 | 691 | 19 | 643 | 504 | 14,311 | 1,520 | 38 |
| 7 | 237 | 13 | 223 | 197 | 11,503 | 1,314 | 19 |
| 8 | 63 | 2 | 62 | 50 | 16,939 | 4,212 | 1 |

Why rows came from the API rather than the dump:

- **Ranked and loved rows (198).** About 181 of them are not in the dump's `osu_beatmaps`: they
  were ranked or loved after the 2026-09-01 snapshot. The other 17 are in `osu_beatmaps` but
  had no failtimes rows on 2026-09-01.
- **Other rows (89).** They are not ranked, so the dump does not include them.

**Missing:** 104 rows (101 distinct ids), all corpus charts with no API match and only a file
`BeatmapID`. Sending those ids to the batch API returned nothing [inference: deleted or unsubmitted
beatmaps; some are reused ids, e.g. 2165277 on three rate-edited "flowing 1.1x/1.2x/1.3x"
charts]. They are listed in `missing.csv`.

### Outputs

On the mac, under `~/ensomi/ensomi-model/artifacts/player-data-20261003/failtimes/`. Everything
except the Parquet files mirrors to the control plane at the same relative path. Every
JSON/CSV/MD file is under 100 KB.

| File | Bytes | Note |
| --- | ---: | --- |
| `failtimes.parquet` | 6,207,525 | The table, after post-processing. SHA-256 `53d788e0…6a31` |
| `failtimes.as-fetched.parquet` | 6,207,661 | The fetch job's output before the zero fill. SHA-256 `c784f582…9f88` |
| `manifest.json` | about 20 K | Provenance (see below) |
| `README.md` | 8,126 | What the table is, the route, checks and caveats |
| `requests.jsonl` | 4,104 | Per-request API log |
| `api-responses/NNN.json.gz` | 10 files | Raw API bodies (the token response is not saved) |
| `coverage.json` | 13,561 | By status × star band |
| `postprocess.json` | 9,296 | Zero fill, ×9 and accounting analyses, checksum mismatch, status drift |
| `metadata-comparison.json` | 14,809 | Comparison with `metadata.json` and the dump-vs-API validation |
| `validation-api-sample.json` | 99,114 | 100 dump-covered ids re-read from the API (not the source of record) |
| `dump-members.json` | 46,281 | Every dump member: size, columns, CREATE TABLE, rows, corpus overlap |
| `missing.csv` | 21,865 | The 104 missing rows with reasons |

`manifest.json` holds:

- Job id, script path and SHA-256 (fetch `141f8d18…85f1`, post-process `996ebc8a…c833`).
- The code checkout. On the mac, `git rev-parse HEAD` is `840fb095…`, which is .git drift. The
  control-plane identity at launch is `2044c2ba…`.
- The corpus SHA-256 `49ab9e1b…d9ee4b`.
- The routes, the dump identity and the licence text and check.
- The API request log, counts by source and status, start and end times, and output hashes.

Job scripts are in `~/ensomi/.sync/cp/jobs/<id>/` (gitignored).

### Comparisons

**Dump against today's API** (100 random dump-covered ids):

- 100/100 have the same checksum.
- Every API bin is at least the dump bin, and every API playcount is at least the dump playcount.
- 12 are identical. The median ratio of total events is 1.005.

**Dataset `metadata.json` (2026-10-02 generation) against this table:**

- 7,176 corpus rows had a 2026-10-02 snapshot with failtimes. 14,630 rows had older snapshots
  without failtimes; 18 had none.
- Dump rows (6,968): the metadata is at least the dump bin-wise in 6,964, with the same
  checksum; 1,094 are identical. The other 4 lack one type row in the dump; they were compared
  before the zero fill.
- API rows (208): today's API is at least the metadata bin-wise and in playcount for 208/208;
  56 are identical.
- **No disagreement beyond growth over time.**

### Data findings that affect use

These are in the README caveats.

- **Sampling ×9.**
  - From beatmap id 3.0 M on, the share of rows with every bin divisible by 9 rounds to 1.000
    (14,218 rows). It is 76% in 2.75–3.0 M and at most 1.5% in every bucket below.
  - This supports a 1-in-9 recording scheme counted ×9 from about 2021 [inference].
  - 1,046 ranked or loved rows have fewer than 10 sampled fails.
- **Accounting ratio**, (fail + exit + pass) / playcount:
  - The median is 0.98–0.99 for ids 1 M–3 M.
  - It falls to 0.89 at 4.25 M, 0.71 at 5.25 M and 0.61 above 5.75 M. This fits lazer fails and
    quits going unrecorded while lazer passes count [inference].
  - It is also 0.75–0.94 below 1 M.
- **Status drift.** 42 corpus-ranked rows were still `qualified` in the 2026-09-01 dump.

---

## Failures and unfinished work

- **`20261003-player-data-inspect` failed.** The script was named `inspect.py`, which shadowed
  Python's `inspect` module, so the import of pyarrow broke. I reran it as `-inspect2`; its
  output overwrote the bad `inspect/inspect.json`. The first job made no network calls.
- **The dry run wrote an oversized file.** `-plan` wrote `failtimes-plan/missing.csv` at 4.4 MB,
  over the mirror limit, into a mirrored subtree. `-plan-cleanup` deleted the directory about
  1 minute later. `ens status` shows all 9 sessions healthy. The fetch script now gzips
  `missing.csv` if it would exceed 3.5 MB.
- **The table was changed after the fetch.**
  - The post-process zero fill touched 47 rows (`fail` absent) and 4 rows (`exit` absent).
  - Before the fill, the fetch counted 21,824 rows with histograms. The 47 rows with only an
    `exit` row had `fail = null`.
  - The pre-fill table is kept, and the change is recorded in the manifest and in each row's
    `note`.
- **Mixed snapshot dates.** Dump rows are as of 2026-09-01 and API rows as of 2026-10-03. A
  single-date table needs either the 2026-10-01 dump (not posted as of 2026-10-03) or about 440
  API requests. The second is allowed under the 600 cap, but the terms prefer the dump. I did
  neither.
- **`requests.jsonl` stores only the first and last id of each request.**
  - The full id lists can be rebuilt: the ids missing from the dump, sorted and cut into
    chunks of 50.
  - The returned ids are in `api-responses/`; the missing ones are in `missing.csv`.
- **Dump overlap statistics are row counts.**
  - I did not separate the overlap of stable scores in `scores` and `osu_scores_mania_high`.
  - I did not check whether replay flags are still valid.
  - Stable replay size was not measured.
- **Not done, by design:** no replay downloads, no contact with ppy, no tracked-file edits, no
  commits. Committing anything, such as a note on the relay branch, is for the main thread to
  decide.
