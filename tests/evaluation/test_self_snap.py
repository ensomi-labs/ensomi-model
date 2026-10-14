import tempfile
import unittest
from pathlib import Path

from ensomi_model.evaluation.self_snap import check


def chart(red_lines, taps, holds):
    lines = ['osu file format v14', '[General]', 'Mode: 3', '[Difficulty]', 'CircleSize:4', '[TimingPoints]']
    lines += [f'{offset},{length},4,2,0,100,1,0' for offset, length in red_lines]
    lines += ['[HitObjects]'] + [f'64,192,{t},1,0,0:0:0:0:' for t in taps]
    lines += [f'192,192,{head},128,0,{release}:0:0:0:0:' for head, release in holds]
    return '\n'.join(lines) + '\n'


class SelfSnapTests(unittest.TestCase):
    def test_counts_notes_off_their_own_lines_and_rows_nearer_than_a_sixteenth(self):
        # Beats of 500 ms from 0, then of 400 ms from 2017.4, which no line of the first beat reaches.
        # 2017 and 2042 are the second segment's first two sixteenths, rounded to before its offset and
        # 25 ms apart: on its lines and a sixteenth of its beat apart, though not of the first segment's.
        # Off: the release at 1516 (a 1/32) and the tap at 2504. Near: 1031 (a 1/16) and 1042 (a 1/12).
        taps = [0, 125, 167, 1000, 1031, 1042, 2017, 2042, 2504]
        holds = [(250, 281), (1500, 1516)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'a.osu'
            path.write_text(chart([(0, 500), (2017.4, 400)], taps, holds))
            result = check(path)
        self.assertEqual((result['events'], result['head_rows']), (13, 11))
        self.assertEqual((result['off_snap'], result['first_off_snap_ms']), (2, 1516.0))
        self.assertEqual((result['close_rows'], result['first_close_rows_ms']), (1, [1031.0, 1042.0]))


if __name__ == '__main__':
    unittest.main()
