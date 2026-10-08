import unittest

from ensomi_model.evaluation.osu_text import osu_text, round_trip
from ensomi_model.evaluation.redlines import RedLine

from ._harness_charts import busy_chart, chart, hold, tap


class RoundTripTests(unittest.TestCase):
    def test_objects_and_red_lines_survive(self):
        c = busy_chart(4)
        c = c.__class__(c.objects, (RedLine(0.0, 500.0, 4), RedLine(4000.0, 1e-3, 4),
                                    RedLine(8000.0, 333.3333333333333, 3, True)))
        back = round_trip(c)
        self.assertEqual(sorted(back.objects, key=lambda o: (o.start_time_ms, o.lane)),
                         sorted(c.objects, key=lambda o: (o.start_time_ms, o.lane)))
        self.assertEqual(back.red_lines, c.red_lines)

    def test_times_are_written_as_integer_ms(self):
        back = round_trip(chart([tap(10.4, 0), hold(20.6, 70.2, 1)]))
        self.assertEqual([(o.start_time_ms, o.end_time_ms) for o in back.objects], [(10.0, 10.0), (21.0, 70.0)])

    def test_superseded_order_is_kept(self):
        lines = (RedLine(0.0, 1234.5, 3), RedLine(0.0, 500.0, 4))
        back = round_trip(chart([tap(0, 0)], lines=((0.0, 1234.5, 3), (0.0, 500.0, 4))))
        self.assertEqual(back.red_lines, lines)

    def test_extras_change_nothing_read(self):
        c = busy_chart(4)
        plain = round_trip(c)
        extra = round_trip(c, green_lines=[(500.0, 0.5), (0.0, 2.0)], hitsound_seed=3, order_seed=7)
        self.assertEqual(set(extra.objects), set(plain.objects))
        self.assertEqual(extra.red_lines, plain.red_lines)
        self.assertNotEqual(osu_text(c, order_seed=7), osu_text(c))


if __name__ == '__main__':
    unittest.main()
