import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ensomi_model.evaluation.corpus import build_inventory, song_title, split_of

LANE_X = (64, 192, 320, 448)


def osu_text(*, title, artist='Artist', version='Hard', beatmap_id=0, set_id=0, audio='audio.mp3',
             timing=('0,500,4,2,0,80,1,0',), objects=()):
    lines = ['osu file format v14', '', '[General]', f'AudioFilename: {audio}', 'Mode: 3', '',
             '[Metadata]', f'Title:{title}', f'Artist:{artist}', 'Creator:mapper', f'Version:{version}',
             f'BeatmapID:{beatmap_id}', f'BeatmapSetID:{set_id}', '', '[Difficulty]', 'CircleSize:4',
             'OverallDifficulty:8', '', '[TimingPoints]', *timing, '', '[HitObjects]']
    for start, lane, end in objects:
        if end is None:
            lines.append(f'{LANE_X[lane]},192,{start},1,0,0:0:0:0:')
        else:
            lines.append(f'{LANE_X[lane]},192,{start},128,0,{end}:0:0:0:0:')
    return '\n'.join(lines) + '\n'


ON_GRID = [(t, (t // 250) % 4, None) for t in range(0, 8000, 250)] + [(8000, 1, 9000), (8500, 2, None)]


class SongTitleTests(unittest.TestCase):
    def test_strips_size_markers_only(self):
        self.assertEqual(song_title('Zero Centimeters (TV Size)'), 'zero centimeters')
        self.assertEqual(song_title('Song -TV ver.-'), 'song')
        self.assertEqual(song_title('Song [Short Version] (TV Size)'), 'song')
        self.assertEqual(song_title('Song (feat. Someone)'), 'song (feat. someone)')
        self.assertEqual(song_title('(TV Size)'), '(tv size)')


class SplitTests(unittest.TestCase):
    def test_r1_trained_groups_are_never_held_out(self):
        ids = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(4000)]
        self.assertFalse(any(split_of(g, True) == 'heldout' for g in ids))
        held = sum(split_of(g, False) == 'heldout' for g in ids) / len(ids)
        calibration = sum(split_of(g, False) == 'calibration' for g in ids) / len(ids)
        self.assertTrue(.12 < held < .18 and .12 < calibration < .18, (held, calibration))
        self.assertEqual([split_of(g, False) for g in ids[:50]], [split_of(g, False) for g in ids[:50]])


class InventoryTests(unittest.TestCase):
    def test_inventory_rows_groups_origins_and_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sets = root / 'dataset' / '0'
            files = {
                '111/a.osu': osu_text(title='Song', beatmap_id=11, set_id=111, objects=ON_GRID),
                '111/b.osu': osu_text(title='Song', version='Easy', beatmap_id=12, set_id=111,
                                      objects=ON_GRID[::2]),
                '222/c.osu': osu_text(title='Song (TV Size)', beatmap_id=21, set_id=222, objects=ON_GRID),
                '333/d.osu': osu_text(title='Other', beatmap_id=31, set_id=333, objects=ON_GRID),
                '444/e.osu': osu_text(title='Unrelated', artist='Someone', beatmap_id=41, set_id=444,
                                      timing=('13,250,4,2,0,80,1,0',), objects=ON_GRID),
                '555/f.osu': osu_text(title='Broken', artist='Nobody', beatmap_id=51, set_id=555,
                                      timing=('0,0,4,2,0,80,1,0',), objects=ON_GRID),
                '666/g.osu': osu_text(title='Gimmick', artist='Nobody', beatmap_id=61, set_id=666,
                                      timing=('0,500,4,2,0,80,1,0', '4000,1e-6,4,2,0,80,1,0',
                                              '4001,500,4,2,0,80,1,8'), objects=ON_GRID),
            }
            for rel, text in files.items():
                (sets / rel).parent.mkdir(parents=True, exist_ok=True)
                (sets / rel).write_text(text)
            for name, audio in (('111', b'song'), ('222', b'song-tv'), ('333', b'song'), ('444', b'other'),
                                ('555', b'broken'), ('666', b'gimmick')):
                (sets / name / 'audio.mp3').write_bytes(audio)
            md5 = {rel: hashlib.md5(text.encode()).hexdigest() for rel, text in files.items()}
            sha = {rel: hashlib.sha256(text.encode()).hexdigest() for rel, text in files.items()}
            (sets / '111' / 'metadata.json').write_text(json.dumps(dict(ranked_date='2020-01-01T00:00:00Z', beatmaps=[
                dict(id=11, checksum=md5['111/a.osu'], status='ranked', difficulty_rating=3.0, mode='mania', cs=4,
                     bpm=120, total_length=10, hit_length=9, top_tag_ids=[dict(tag_id=5, count=2)]),
                dict(id=12, checksum='0' * 32, status='loved', difficulty_rating=1.0, mode='mania', cs=4)])))
            acquisitions = root / 'dataset' / 'metadata' / 'acquisitions'
            acquisitions.mkdir(parents=True)
            (acquisitions / 'x.json').write_text(json.dumps(dict(acquisitionId='acq-test',
                                                                 sets=[dict(beatmapSetId=444)])))
            catalog = root / 'catalog.json'
            catalog.write_text(json.dumps([dict(source_sha256=sha['111/b.osu'], split='train', group_id='song:x')]))

            table, summary = build_inventory(root, r1_catalog=catalog, workers=1)
            rows = {Path(r['path']).relative_to('dataset/0').as_posix(): r for r in table.to_pylist()}

            self.assertEqual(set(rows), set(files))
            self.assertEqual(rows['444/e.osu']['origin'], 'acq-test')
            self.assertEqual(rows['111/a.osu']['origin'], 'base')
            self.assertEqual((rows['111/a.osu']['api_match'], rows['111/a.osu']['api_status']), ('checksum', 'ranked'))
            self.assertEqual(rows['111/a.osu']['api_tag_ids'], [5])
            self.assertEqual((rows['111/b.osu']['api_match'], rows['111/b.osu']['api_status']), ('beatmap_id', 'loved'))
            self.assertIsNone(rows['222/c.osu']['api_match'])
            a = rows['111/a.osu']
            self.assertEqual((a['n_taps'], a['n_lns'], a['dominant_fold'], a['canonical_bpm']), (33, 1, 0, 120.0))
            self.assertGreater(a['star'], 0)
            self.assertEqual((a['head_on_grid'], a['release_on_grid'], a['renotation_invariant']), (1.0, 1.0, True))
            self.assertLess(a['roundtrip_max_ms'], 1e-6)
            e = rows['444/e.osu']
            self.assertEqual((e['dominant_fold'], e['canonical_bpm']), (-1, 120.0))  # notated 240 folds to 120
            self.assertLess(e['head_on_grid'], 1.0)  # its grid starts 13 ms late
            group = {rel: r['group_id'] for rel, r in rows.items()}
            self.assertEqual(len({group['111/a.osu'], group['111/b.osu'], group['222/c.osu'], group['333/d.osu']}), 1)
            self.assertNotEqual(group['444/e.osu'], group['111/a.osu'])
            self.assertEqual(group['111/a.osu'], min(sha[k] for k in ('111/a.osu', '111/b.osu', '222/c.osu', '333/d.osu')))
            self.assertEqual(rows['111/b.osu']['r1_split'], 'train')
            self.assertNotEqual(rows['333/d.osu']['eval_split'], 'heldout')
            g = rows['666/g.osu']  # a 60,000,000 BPM line, then the grid resumes 1 ms later
            self.assertEqual((g['n_red_lines'], g['n_musical'], g['n_redundant'], g['n_expressive']), (3, 1, 1, 1))
            self.assertEqual(g['expressive_reasons'], 'implausible:1')
            self.assertEqual((g['head_on_grid'], g['n_bar_starts'], g['renotation_invariant']), (1.0, 1, True))
            broken = rows['555/f.osu']
            self.assertEqual(broken['error'], 'timing: No red line with a plausible BPM')
            self.assertIsNotNone(broken['star'])
            self.assertEqual(summary['files'], 7)
            self.assertEqual(summary['errors'], {'timing': 1})
            self.assertEqual(summary['r1_trained_in_heldout'], 0)


if __name__ == '__main__':
    unittest.main()
