import math
import unittest

from ensomi_model.evaluation.legality import (
    AFTER_SONG, BEFORE_SONG, EMPTY, LANE, NONFINITE, OVERLAP, RELEASE, TAP_LENGTH, is_legal, violations,
    violations_arrays)
from ensomi_model.osu_core.hitobjects import ManiaHitObject, ManiaHitObjectKind

from ._harness_charts import busy_chart, hold, tap


class LegalityTests(unittest.TestCase):
    def test_legal_chart(self):
        self.assertEqual(violations([tap(0, 0), tap(0, 1), hold(100, 300, 2), tap(301, 2), tap(200, 3)]), {})
        self.assertTrue(is_legal(busy_chart().objects))

    def test_overlaps_in_one_lane(self):
        self.assertEqual(violations([tap(0, 0), tap(0, 0)]), {OVERLAP: 1})
        self.assertEqual(violations([hold(0, 100, 1), tap(50, 1)]), {OVERLAP: 1})
        # A hold occupies its lane up to and including its release.
        self.assertEqual(violations([hold(0, 100, 1), tap(100, 1)]), {OVERLAP: 1})
        self.assertEqual(violations([hold(0, 100, 1), hold(101, 200, 1)]), {})
        # A long hold covers every later object it spans, not only the next one.
        self.assertEqual(violations([hold(0, 500, 3), tap(100, 3), tap(400, 3)]), {OVERLAP: 2})

    def test_release_lane_and_times(self):
        self.assertEqual(violations([hold(100, 100, 0)]), {RELEASE: 1})
        self.assertEqual(violations([hold(100, 50, 0)]), {RELEASE: 1})
        self.assertEqual(violations([tap(0, 4)]), {LANE: 1})
        self.assertEqual(violations([tap(0, -1)]), {LANE: 1})
        self.assertEqual(violations([tap(math.nan, 0)]), {NONFINITE: 1})
        self.assertEqual(violations([ManiaHitObject(0.0, 10.0, 0, ManiaHitObjectKind.TAP)]), {TAP_LENGTH: 1})
        self.assertEqual(violations([]), {EMPTY: 1})

    def test_columns_agree_with_objects(self):
        objs = [tap(0, 0), hold(0, 100, 1), tap(100, 1), tap(-5, 2), hold(50, 40, 3)]
        cols = ([o.start_time_ms for o in objs], [o.end_time_ms for o in objs], [o.lane for o in objs],
                [o.kind.value == 'HOLD' for o in objs])
        self.assertEqual(violations_arrays(*cols, song_span=(0, 90)), violations(objs, song_span=(0, 90)))
        self.assertEqual(violations_arrays([], [], [], []), {EMPTY: 1})

    def test_inside_the_song(self):
        objs = [tap(-5, 0), hold(100, 1200, 1)]
        self.assertEqual(violations(objs, song_span=(0, 1000)), {BEFORE_SONG: 1, AFTER_SONG: 1})
        self.assertEqual(violations(objs, song_span=(-5, 1200)), {})


if __name__ == '__main__':
    unittest.main()
