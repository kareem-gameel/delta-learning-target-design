from __future__ import annotations

import unittest

import pandas as pd

from target_design.screening import compare_to_parent


class ScreeningTests(unittest.TestCase):
    def test_screening_classes(self) -> None:
        table = pd.DataFrame(
            {
                "target": ["parent", "good", "tradeoff", "large"],
                "target_iqr": [1.0, 0.6, 0.4, 1.2],
                "d_iqr": [2.0, 1.8, 3.0, 1.0],
            }
        )
        result = compare_to_parent(table, "parent").set_index("target")
        self.assertEqual(result.loc["parent", "screening_class"], "parent")
        self.assertEqual(result.loc["good", "screening_class"], "favoured")
        self.assertEqual(result.loc["tradeoff", "screening_class"], "tradeoff")
        self.assertEqual(result.loc["large", "screening_class"], "not_scale_reducing")


if __name__ == "__main__":
    unittest.main()
