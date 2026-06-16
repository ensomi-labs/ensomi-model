from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pulsefield_model.evals.c3_ordered_raw_field_grammar_probe import (
    decode_ordered_field_symbols,
    encode_ordered_field_symbols,
    normalize_token_vocab,
    raw_payload_parts,
    run_ordered_raw_field_grammar_probe,
)


class C3OrderedRawFieldGrammarProbeTests(unittest.TestCase):
    def test_encode_decode_preserves_raw_with_tail(self) -> None:
        token = "RAW|D:0:0:0:15:OE3,E0,E1,E2"

        symbols = encode_ordered_field_symbols(token)
        decoded = decode_ordered_field_symbols(symbols)

        self.assertEqual(symbols[0], "RAW_BEGIN")
        self.assertEqual(symbols[-1], "RAW_END")
        self.assertEqual(decoded, (token,))

    def test_encode_decode_preserves_ref_and_res_tokens(self) -> None:
        tokens = ("REF|skeleton|103|2", "RES|D:12:8:0:0")
        symbols = tuple(symbol for token in tokens for symbol in encode_ordered_field_symbols(token))

        decoded = decode_ordered_field_symbols(symbols)

        self.assertEqual(decoded, tokens)

    def test_raw_payload_parts_splits_payload_fields(self) -> None:
        self.assertEqual(
            raw_payload_parts("RAW|D:12:8:0:4"),
            ("D", "12", "8", "0", "4"),
        )

    def test_normalize_token_vocab_accepts_sidecar_shape(self) -> None:
        self.assertEqual(
            normalize_token_vocab({"RAW|D:1:2:3:4": 2, "REF|exact|1|1": 1}),
            {1: "REF|exact|1|1", 2: "RAW|D:1:2:3:4"},
        )

    def test_probe_reports_lossless_reconstruction_on_synthetic_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sidecar.json"
            payload = {
                "schema_version": 1,
                "contract": "test",
                "token_id_base": 1,
                "token_pad_id": 0,
                "mapper_window_ms": 8000,
                "timing_contract": "test",
                "window_anchor_mode": "exact_group",
                "token_vocab": {
                    "RAW|D:12:8:0:0": 1,
                    "RAW|D:6:4:0:0": 2,
                    "REF|exact|1|1": 3,
                    "RES|D:12:4:0:0": 4,
                },
                "windows": [
                    {"split": "train", "token_ids": [1, 2, 3], "boundary_risk_token_count": 0},
                    {"split": "test", "token_ids": [1, 4], "boundary_risk_token_count": 1},
                ],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")

            summary = run_ordered_raw_field_grammar_probe(sidecar_path=path)

            self.assertTrue(summary["pass_criteria"]["reconstruction_pass"])
            self.assertEqual(summary["reconstruction"]["mismatch_count"], 0)
            self.assertEqual(summary["dataset"]["token_count"], 5)
            self.assertGreater(summary["sequence"]["ordered_to_exact_token_ratio"], 1.0)
            self.assertLess(
                summary["label_space"]["raw_ordered_symbol_vocab_size"],
                summary["label_space"]["raw_exact_vocab_size"] + 10,
            )


if __name__ == "__main__":
    unittest.main()
