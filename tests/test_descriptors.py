from __future__ import annotations

import unittest

import numpy as np

from target_design.descriptors.angular import build_angular_matrix
from target_design.descriptors.pre import build_pre_matrix, radial_grid


class DescriptorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            (
                "water",
                ("O", "H", "H"),
                np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
            )
        ]

    def test_paper_pre_dimensions(self) -> None:
        centers, widths = radial_grid()
        self.assertEqual(len(centers), 35)
        self.assertEqual(len(widths), 35)
        matrix, ids = build_pre_matrix(self.records)
        self.assertEqual(ids, ["water"])
        self.assertEqual(matrix.shape, (1, 34650))
        self.assertGreater(matrix.nnz, 0)

    def test_paper_angular_dimensions(self) -> None:
        matrix, ids = build_angular_matrix(self.records)
        self.assertEqual(ids, ["water"])
        self.assertEqual(matrix.shape, (1, 11088))
        self.assertEqual(matrix.sum(), 3.0)


if __name__ == "__main__":
    unittest.main()
