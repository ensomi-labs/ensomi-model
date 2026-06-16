import unittest
from pathlib import Path

from pulsefield_model.evals.c3_reduced_sidecar import (
    reduce_sidecar_payload,
    select_top_tokens_per_kind,
)


class C3ReducedSidecarTests(unittest.TestCase):
    def test_select_top_tokens_per_kind_remaps_by_kind_and_count(self) -> None:
        token_by_id = {
            1: "RAW|a",
            2: "RAW|b",
            3: "RAW|c",
            4: "REF|a",
            5: "REF|b",
            6: "RES|a",
            7: "OTHER|x",
        }
        counts = {
            1: 5,
            2: 9,
            3: 9,
            4: 3,
            5: 8,
            6: 4,
            7: 100,
        }

        remap = select_top_tokens_per_kind(counts, token_by_id, top_per_kind=2)

        self.assertEqual(
            remap,
            {
                2: 1,
                3: 2,
                5: 3,
                4: 4,
                6: 5,
            },
        )

    def test_reduce_sidecar_payload_filters_and_remaps_tokens(self) -> None:
        source_payload = {
            "contract": "source_contract",
            "token_vocab": {
                "RAW|a": 1,
                "RAW|b": 2,
                "REF|a": 3,
            },
            "windows": [
                {"beatmap_path": "a.osu", "window_start_ms": 0, "token_ids": [1, 2, 3, 2]},
                {"beatmap_path": "a.osu", "window_start_ms": 8000, "token_ids": [3]},
            ],
        }
        token_by_id = {1: "RAW|a", 2: "RAW|b", 3: "REF|a"}

        reduced, stats = reduce_sidecar_payload(
            source_payload,
            token_remap={2: 1, 3: 2},
            token_by_id=token_by_id,
            top_per_kind=1,
            source_path=Path("source.json"),
        )

        self.assertEqual(reduced["token_vocab"], {"RAW|b": 1, "REF|a": 2})
        self.assertEqual(reduced["windows"][0]["token_ids"], [1, 2, 1])
        self.assertEqual(reduced["windows"][1]["token_ids"], [2])
        self.assertEqual(stats["source_raw_token_count"], 5)
        self.assertEqual(stats["retained_raw_token_count"], 4)
        self.assertEqual(stats["source_positive_label_count"], 4)
        self.assertEqual(stats["retained_positive_label_count"], 3)


if __name__ == "__main__":
    unittest.main()
