# Player data from the data.ppy.sh dumps

`dump_collector.py` keeps a local, growing copy of the monthly osu!mania `random_10000`
performance dumps published at <https://data.ppy.sh>. `dump_panel.py` measures whether
consecutive dumps hold the same players and what their differences say about attempts and skill
over time. Both run on bings-mac; the control plane launches them with `ens run`.

## Rules the code enforces

- **Only data.ppy.sh is contacted**: the listing (once per run), `LICENCE.txt` (once per run
  with new work), one HEAD per new dump and the tarball download. Nothing goes to osu.ppy.sh,
  its API, or any replay endpoint.
- **Licence.** The dump licence allows statistical analysis and asks for contact before
  production or public use. The run compares `LICENCE.txt` and any licence file inside a
  tarball with the text recorded with the first dump (SHA-256 `b5c69b6c…6daa`). A different
  text stops the run before anything is parsed (exit 3). The 2026-09-01 tarball holds no licence
  file, so the site file is the licence of record; each manifest says which applied.
- **Privacy.** Per-player tables (user ids, usernames) are Parquet files on the mac. The meta
  mirror never copies Parquet or tarballs. The JSON, Markdown and log outputs, which do mirror
  to the control plane, hold aggregates only.
- **Politeness and disk.** One download at a time, resumable (`.part` file, `Range` with
  `If-Range`), size checked against HEAD, at most 5 attempts per run. A dump that keeps failing
  stops the run. A dump is not started if free disk would fall below 50 GB (`--min-free-gb`).
- **Idempotence.** A dump is collected once its `manifest.json` says `complete`. Failed dumps
  are retried on the next run; blocked dumps (format change, disk, hash mismatch) are skipped
  until `--retry-blocked`. A lock file keeps two runs from overlapping.

## Running it

From the control plane:

```sh
ens run ensomi-model -- .venv/bin/python scripts/player_data/dump_collector.py            # collect new dumps
ens run ensomi-model -- .venv/bin/python scripts/player_data/dump_collector.py --dry-run  # plan only
ens run ensomi-model -- .venv/bin/python scripts/player_data/dump_panel.py                # rebuild the report
ens run ensomi-model -- .venv/bin/python -m pytest -q tests/player_data
```

Options of `dump_collector.py`: `--only YYYY-MM-DD ...` restricts the dates; `--include-top`
also collects the mania `top_10000` dumps (about 2 GB each; off by default, stored as
`<date>_top_10000/` and ignored by the panel report); `--no-panel` skips the report rebuild
that otherwise follows a new dump; `--seed-dir DIR` names a directory of earlier downloads to
clone from instead of downloading (default `artifacts/player-data-20261003/dump`, where the
2026-09-01 tarball was first fetched; a seeded file must match its recorded SHA-256).

Exit codes: 0 done or nothing new, 1 a dump failed (retried next run), 2 blocked, 3 licence
changed.

## Outputs

Under `artifacts/player-data-dumps/` on the mac:

| Path | Content | Mirrored |
| --- | --- | --- |
| `<date>/<name>.tar.bz2` | The tarball as published | no |
| `<date>/<table>.parquet` | The seven tables below, zstd | no |
| `<date>/manifest.json` | URL, HEAD size, Last-Modified and ETag, tarball SHA-256 and how it was obtained, licence hashes, tar member list, per-table rows seen and kept, schema against the 2026-09-01 reference, mysqldump completion time, script hashes, job id, timings | yes |
| `collection.json` | One entry per dump (status, hashes, row counts, snapshot times, whether still listed), listed dumps not collected, disk use | yes |
| `panel-report.json`, `panel-report.md` | Panel measurements (`dump_panel.py`) | yes |
| `log/runs.jsonl` | One line per collector run | yes |
| `log/launchd.out.log`, `log/launchd.err.log` | Output of the scheduled runs | yes |

Tables, with the rows kept:

| Table | Rows kept | Notes |
| --- | --- | --- |
| `sample_users` | all | The sampled users, with username |
| `osu_user_stats_mania` | all | pp is `rank_score`; also playcount, fail and quit counts, play time |
| `scores` | `ruleset_id = 3` | Lazer-era best scores; adds `in_corpus` |
| `osu_scores_mania_high` | all | Stable best scores |
| `osu_user_beatmap_playcount` | all | Lifetime attempts per user and beatmap |
| `osu_beatmaps` | `playmode = 3` | |
| `osu_beatmap_failtimes` | mania beatmaps of the same dump | Filtered after `osu_beatmaps` is read, since it comes first in the tarball |

`in_corpus` marks beatmaps of `data/r2-corpus.parquet`, by `api_beatmap_id`, else the `.osu`
file `BeatmapID` when it is positive. The difficulty and attribute tables
(`osu_beatmap_difficulty`, `osu_beatmap_difficulty_attribs`, `osu_difficulty_attribs`) are not
extracted, nor are `osu_beatmapsets`, `osu_counts` and `osu_beatmap_performance_blacklist`; all
stay in the tarball. Column types: integers become int64, reals float64, timestamps
`timestamp[s, UTC]` (the dumps set `TIME_ZONE='+00:00'`; MySQL zero dates become null and are
counted in the manifest), everything else string, JSON included.

A table without a column the code depends on (`REQUIRED_COLUMNS` in `dump_collector.py`)
blocks the dump. Other column differences from the 2026-09-01 schema are recorded under
`schema_vs_reference` and do not block.

## Scheduled run (launchd)

`launchd.sh` writes `~/Library/LaunchAgents/com.ensomi.player-dump-collector.plist` with the
mac's absolute paths (the plist is never synced) and loads it. The agent runs the collector
daily at 21:15 local time (Asia/Shanghai, 13:15 UTC); data.ppy.sh posts dumps around 12:00 UTC
on the first of the month. A run missed while the mac sleeps happens at the next wake.

Install, check, start one run, or remove, on the mac or through `ens run`:

```sh
ens run ensomi-model -- sh scripts/player_data/launchd.sh install     # optional: install HOUR MINUTE
ens run ensomi-model -- sh scripts/player_data/launchd.sh status
ens run ensomi-model -- sh scripts/player_data/launchd.sh kick
ens run ensomi-model -- sh scripts/player_data/launchd.sh uninstall
```

`install` tries `launchctl bootstrap gui/<uid>` and falls back to `launchctl load -w`. If both
fail from a remote session, run `sh scripts/player_data/launchd.sh install` in a Terminal on
the mac. Uninstalling removes only the plist; collected data stays.

## Tests

`tests/player_data/` covers the INSERT parser (quoting, escapes, NULL against the string
`'NULL'`, the split fast path against the tokenizer), extraction of a small tarball shaped like
a real dump (filters, `in_corpus`, failtimes after beatmaps, missing tables and columns), the
licence comparison, and snapshot differencing (a count that rises, stays, falls, appears or
vanishes, and a user who leaves the sample).
