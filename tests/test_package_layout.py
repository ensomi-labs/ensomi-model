"""Import tests for the source package layout."""

import importlib
import unittest


class PackageLayoutTest(unittest.TestCase):
    def test_public_packages_import(self) -> None:
        modules = [
            "ensomi_model",
            "ensomi_model.osu_core",
            "ensomi_model.features",
            "ensomi_model.research",
            "ensomi_model.research.chart",
            "ensomi_model.research.oracle_time_continuation",
            "ensomi_model.research.bounded_typed_continuation",
            "ensomi_model.research.vacation_training",
            "ensomi_model.research.r1_restore",
        ]

        for module in modules:
            with self.subTest(module=module):
                importlib.import_module(module)


if __name__ == "__main__":
    unittest.main()
