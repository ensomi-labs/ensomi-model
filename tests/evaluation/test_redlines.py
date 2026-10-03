import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from ensomi_model.evaluation.redlines import RedLine, classify, musical_grid, read_red_lines


def roles(lines, events):
    return [f'{r.role}:{r.reason}' for r in classify(lines, events)]


def every(step, start, stop):
    return list(np.arange(start, stop, step))


class ReadTests(unittest.TestCase):
    def test_reads_every_red_line_and_skips_green_ones(self):
        text = '\n'.join(['osu file format v14', '[TimingPoints]', '0,500,4,2,0,80,1,0',
                          '1000,-100,4,2,0,80,0,0', '2000,0,0,2,0,80,1,8', '3000,1e-300,4,2,0,80,1,9',
                          '4000,500', '', '[HitObjects]', '64,192,0,1,0,0:0:0:0:']) + '\n'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'a.osu'
            path.write_text(text)
            lines = read_red_lines(path)
        self.assertEqual([line.offset_ms for line in lines], [0.0, 2000.0, 3000.0, 4000.0])
        self.assertEqual(lines[1], RedLine(2000.0, 0.0, 4, True))  # meter 0 reads as 4, as in osu!
        self.assertTrue(lines[2].omit_first_barline)
        self.assertFalse(lines[1].plausible or lines[2].plausible)
        self.assertEqual(lines[3], RedLine(4000.0, 500.0, 4, False))


class ClassifyTests(unittest.TestCase):
    def test_superseded_and_implausible_lines(self):
        lines = [RedLine(0, 500), RedLine(0, 400), RedLine(1000, 1e-3), RedLine(2000, float('nan'))]
        self.assertEqual(roles(lines, every(400, 0, 8000)),
                         ['expressive:superseded', 'musical:first', 'expressive:implausible',
                          'expressive:implausible'])

    def test_empty_short_line_and_an_offset_line_the_events_do_not_need(self):
        lines = [RedLine(0, 500), RedLine(4000, 300), RedLine(4200, 500)]
        events = every(500, 0, 4000) + every(500, 4500, 9000)
        self.assertEqual(roles(lines, events), ['musical:first', 'expressive:empty', 'expressive:detour'])

    def test_lines_on_the_grid_in_force_are_redundant(self):
        lines = [RedLine(0, 500), RedLine(4000, 500, 3), RedLine(8000, 250)]  # 240 BPM folds onto 120
        self.assertEqual(roles(lines, every(500, 0, 16000)),
                         ['musical:first', 'redundant:same_grid', 'redundant:same_grid'])

    def test_bar_line_on_a_half_canonical_beat_moves_the_phase(self):
        # Notated 200 BPM is canonical 100: notated beat 9 is canonical beat 4.5.
        lines = [RedLine(0, 300, 4), RedLine(2700, 300, 6)]
        self.assertEqual(roles(lines, every(300, 0, 20000)), ['musical:first', 'musical:phase'])

    def test_bar_line_shorter_than_its_bar_is_dropped(self):
        lines = [RedLine(0, 300, 4), RedLine(2700, 300, 4), RedLine(3000, 300, 4)]
        self.assertEqual(roles(lines, every(300, 0, 20000)),
                         ['musical:first', 'expressive:short_bar', 'redundant:same_grid'])

    def test_scroll_speed_detour_returns_to_the_grid(self):
        # 180 BPM for 4 s while every note stays on the 120 BPM beats.
        lines = [RedLine(0, 500), RedLine(4000, 60000 / 180), RedLine(8000, 500)]
        self.assertEqual(roles(lines, every(500, 0, 20000)),
                         ['musical:first', 'expressive:detour', 'redundant:same_grid'])

    def test_tempo_changes_the_events_need_are_kept(self):
        lines = [RedLine(0, 500), RedLine(4000, 400)]
        self.assertEqual(roles(lines, every(500, 0, 4000) + every(400, 4000, 12000)),
                         ['musical:first', 'musical:tempo'])
        # A metric modulation: the 180 BPM passage would be triplets on the 120 grid.
        lines = [RedLine(0, 500), RedLine(4000, 60000 / 180), RedLine(8000, 500)]
        events = every(500, 0, 4000) + every(60000 / 180, 4000, 7999) + every(500, 8000, 12000)
        self.assertEqual(roles(lines, events), ['musical:first', 'musical:tempo', 'musical:tempo'])

    def test_roles_survive_renotating_the_whole_chart(self):
        cases = [([RedLine(0, 500), RedLine(4000, 60000 / 180), RedLine(8000, 500)], every(500, 0, 20000)),
                 ([RedLine(0, 300, 4), RedLine(2700, 300, 6)], every(300, 0, 20000)),
                 ([RedLine(0, 300, 4), RedLine(2700, 300, 4), RedLine(3000, 300, 4)], every(300, 0, 20000))]
        for lines, events in cases:
            for factor in (-1, 1):
                with self.subTest(lines=lines, factor=factor):
                    other = [replace(line, beat_length_ms=line.beat_length_ms / 2.0 ** factor) for line in lines]
                    self.assertEqual(roles(other, events), roles(lines, events))


class MusicalGridTests(unittest.TestCase):
    def test_segments_and_bar_starts(self):
        lines = [RedLine(0, 500, 4), RedLine(4000, 500, 3, True), RedLine(10000, 400, 4, True),
                 RedLine(12000, 400, 3), RedLine(20000, 1e-4)]
        events = every(500, 0, 10000) + every(400, 10000, 16000)
        grid, line_roles = musical_grid(lines, events)
        self.assertEqual([r.role for r in line_roles], ['musical', 'redundant', 'musical', 'redundant', 'expressive'])
        self.assertEqual([s.offset_ms for s in grid.segments], [0.0, 10000.0])
        # Omit-first-barline lines start no bar; the redundant 3/4 line without it does.
        self.assertEqual([(b.time_ms, b.meter) for b in grid.bar_starts], [(0.0, 4), (12000.0, 3)])
        with self.assertRaises(ValueError):
            musical_grid([RedLine(0, 0.0)], events)


if __name__ == '__main__':
    unittest.main()
