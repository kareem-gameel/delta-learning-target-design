from __future__ import annotations

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from target_design.roughness import (
    aggregate_undirected_edges,
    build_weighted_knn_graph,
    compute_graph_roughness,
    scale_features_by_population_std,
)


class RoughnessTests(unittest.TestCase):
    def test_duplicate_directed_edges_are_averaged(self) -> None:
        neighbours = np.array([[1], [0], [1]])
        directed_weights = np.array([[0.2], [0.6], [0.4]])
        edges, weights = aggregate_undirected_edges(3, neighbours, directed_weights)
        np.testing.assert_array_equal(edges, np.array([[0, 1], [1, 2]]))
        np.testing.assert_allclose(weights, np.array([0.4, 0.4]))

    def test_iqr_roughness_is_affine_scale_invariant(self) -> None:
        edges = np.array([[0, 1], [1, 2], [2, 3]])
        weights = np.array([1.0, 2.0, 1.0])
        target = np.array([0.0, 1.0, 3.0, 4.0])
        first = compute_graph_roughness(target, edges, weights)
        second = compute_graph_roughness(7.5 * target - 12.0, edges, weights)
        self.assertAlmostEqual(first.d_iqr, second.d_iqr, places=12)

    def test_sparse_scaling_drops_constant_channels(self) -> None:
        features = csr_matrix([[0.0, 1.0, 2.0], [1.0, 1.0, 4.0], [2.0, 1.0, 6.0]])
        scaled, keep = scale_features_by_population_std(features)
        np.testing.assert_array_equal(keep, np.array([True, False, True]))
        self.assertEqual(scaled.shape, (3, 2))

    def test_weighted_knn_graph_is_finite(self) -> None:
        features = csr_matrix(np.array([[1.0, 0.0], [0.9, 0.1], [0.1, 0.9], [0.0, 1.0]]))
        edges, weights, metadata = build_weighted_knn_graph(features, k=2)
        self.assertGreater(len(edges), 0)
        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue((weights > 0).all())
        self.assertEqual(metadata["k"], 2)


if __name__ == "__main__":
    unittest.main()
