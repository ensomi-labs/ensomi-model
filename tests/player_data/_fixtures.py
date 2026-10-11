"""Small mysqldump members and tarballs shaped like the data.ppy.sh performance dumps."""

import importlib
import io
import sys
import tarfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts' / 'player_data'


def load(name):
    """Import a module from scripts/player_data (not a package)."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    return importlib.import_module(name)


HEADER = [
    '-- MySQL dump 10.13  Distrib 8.4.10, for Linux (x86_64)',
    '--',
    '-- Host: 127.0.0.1    Database: osu',
    '-- ------------------------------------------------------',
    '-- Server version\t8.4.9',
    "/*!40103 SET TIME_ZONE='+00:00' */;",
    '--',
    '-- Table structure for table `{t}`',
    '--',
]


def member(table, column_defs, insert_bodies, completed='2026-09-01  5:48:21'):
    """Bytes of one mysqldump member: header, CREATE TABLE, one INSERT per body, trailer."""
    lines = [h.format(t=table) for h in HEADER]
    lines.append(f'DROP TABLE IF EXISTS `{table}`;')
    lines.append(f'CREATE TABLE `{table}` (')
    lines += [f'  {c},' for c in column_defs]
    first = column_defs[0].split('`')[1]
    lines.append(f'  PRIMARY KEY (`{first}`),')
    lines.append(f'  KEY `k_{first}` (`{first}`)')
    lines.append(') ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 ROW_FORMAT=DYNAMIC;')
    lines.append(f'LOCK TABLES `{table}` WRITE;')
    lines.append(f'/*!40000 ALTER TABLE `{table}` DISABLE KEYS */;')
    for body in insert_bodies:
        lines.append(f'INSERT INTO `{table}` VALUES {body};')
    lines.append(f'/*!40000 ALTER TABLE `{table}` ENABLE KEYS */;')
    lines.append('UNLOCK TABLES;')
    lines.append(f'-- Dump completed on {completed}')
    return ('\n'.join(lines) + '\n').encode()


# Corpus ids used by the fixture: beatmap 101 is a corpus map, 102 and 103 are not.
CORPUS_IDS = {101}

MEMBERS = {
    'osu_beatmap_difficulty': member(
        'osu_beatmap_difficulty', ['`beatmap_id` mediumint unsigned NOT NULL', '`diff_unified` float NOT NULL'],
        ['(101,2.5),(102,3.1)']),
    'osu_beatmap_failtimes': member(
        'osu_beatmap_failtimes',
        ["`beatmap_id` mediumint NOT NULL", "`type` enum('fail','exit') NOT NULL",
         "`p1` mediumint unsigned NOT NULL DEFAULT '0'", "`p2` mediumint unsigned NOT NULL DEFAULT '0'"],
        ["(101,'fail',9,18),(101,'exit',0,9),(102,'fail',1,2)", "(103,'exit',4,5),(999,'fail',7,7)"],
        completed='2026-09-01  5:51:26'),
    'osu_beatmaps': member(
        'osu_beatmaps',
        ['`beatmap_id` mediumint unsigned NOT NULL AUTO_INCREMENT',
         '`version` varchar(80) CHARACTER SET latin1 NOT NULL DEFAULT \'\'',
         "`playmode` tinyint unsigned NOT NULL DEFAULT '0'", '`last_update` timestamp NOT NULL'],
        ["(101,'4K Hard (x, y)',3,'2020-01-02 03:04:05'),(102,'Insane',0,'2021-01-01 00:00:00'),"
         "(103,'Kyle\\'s 4K',3,'2022-05-06 07:08:09')"],
        completed='2026-09-01  5:51:15'),
    'osu_scores_mania_high': member(
        'osu_scores_mania_high',
        ['`score_id` bigint unsigned NOT NULL AUTO_INCREMENT', "`beatmap_id` mediumint unsigned NOT NULL DEFAULT '0'",
         '`user_id` int unsigned NOT NULL', "`rank` enum('A','B','C','D','S','SH','X','XH') NOT NULL",
         '`date` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP', '`pp` float DEFAULT NULL'],
        ["(5001,101,1,'S','2026-08-15 12:00:00',123.5),(5002,103,2,'A','2019-03-04 05:06:07',NULL)"],
        completed='2026-09-01  5:48:21'),
    'osu_user_beatmap_playcount': member(
        'osu_user_beatmap_playcount',
        ["`user_id` int unsigned NOT NULL DEFAULT '0'", '`beatmap_id` mediumint unsigned NOT NULL',
         '`playcount` smallint unsigned NOT NULL'],
        ['(1,101,5),(1,102,2),(2,103,7)', '(3,101,1)'],
        completed='2026-09-01  5:48:37'),
    'osu_user_stats_mania': member(
        'osu_user_stats_mania',
        ['`user_id` int unsigned NOT NULL', '`playcount` mediumint NOT NULL', '`rank_score` float unsigned NOT NULL',
         "`country_acronym` char(2) NOT NULL DEFAULT ''", '`last_played` timestamp NOT NULL'],
        ["(1,100,1234.56,'JP','2026-08-31 23:00:00'),(2,50,1e-05,'KR','0000-00-00 00:00:00'),"
         "(3,7,0,'US','2025-01-01 00:00:00')"],
        completed='2026-09-01  5:48:23'),
    'sample_users': member(
        'sample_users',
        ['`user_id` int NOT NULL', "`username` varchar(255) NOT NULL DEFAULT ''",
         "`user_warnings` tinyint NOT NULL DEFAULT '0'"],
        ["(1,'alpha',0),(2,'b\\'eta, (two)',0),(3,'NULL',1)"],
        completed='2026-09-01  5:46:12'),
    'scores': member(
        'scores',
        ['`id` bigint unsigned NOT NULL AUTO_INCREMENT', '`user_id` int unsigned NOT NULL',
         '`ruleset_id` smallint unsigned NOT NULL', '`beatmap_id` mediumint unsigned NOT NULL',
         '`data` json NOT NULL', '`pp` float unsigned DEFAULT NULL', '`legacy_score_id` bigint unsigned DEFAULT NULL',
         '`ended_at` timestamp NOT NULL'],
        ["(7001,1,3,101,'{\\\"mods\\\": [{\\\"acronym\\\": \\\"DT\\\"}], \\\"note\\\": \\\"a),(b\\\"}',200.25,NULL,"
         "'2026-08-20 01:02:03'),"
         "(7002,2,3,102,'{\\\"mods\\\": []}',NULL,99,'2026-07-01 00:00:00'),"
         "(7003,1,0,101,'{}',1.5,NULL,'2026-08-21 00:00:00')"],
        completed='2026-09-01  5:51:14'),
}

ORDER = ['osu_beatmap_difficulty', 'osu_beatmap_failtimes', 'osu_beatmaps', 'osu_scores_mania_high',
         'osu_user_beatmap_playcount', 'osu_user_stats_mania', 'sample_users', 'scores']


def write_tarball(path, members=None, prefix='2026_09_01_performance_mania_random_10000', extra=None):
    """A .tar.bz2 with the given members (default: all fixtures in the real dump's order)."""
    members = MEMBERS if members is None else members
    with tarfile.open(path, 'w:bz2') as tf:
        for name in [n for n in ORDER if n in members] + [n for n in members if n not in ORDER]:
            data = members[name]
            info = tarfile.TarInfo(f'{prefix}/{name}.sql')
            info.size = len(data)
            info.mtime = 1788241572
            tf.addfile(info, io.BytesIO(data))
        for name, data in (extra or {}).items():
            info = tarfile.TarInfo(f'{prefix}/{name}')
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path
