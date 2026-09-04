import unittest

import numpy as np

from src.geometry_diagnostics import (
    classify_mask_provenance,
    diagnose_dual_radar_geometry,
)


def _geometry(azimuth1, azimuth2, valid1=True, valid2=True):
    return diagnose_dual_radar_geometry(
        np.array([azimuth1]), np.array([0.0]), np.array([1000.0]),
        np.array([valid1]),
        np.array([azimuth2]), np.array([0.0]), np.array([1000.0]),
        np.array([valid2]),
    )


class SyntheticGeometryTest(unittest.TestCase):
    def test_right_angle_has_best_horizontal_conditioning(self):
        right_angle = _geometry(0.0, 90.0)
        near_parallel = _geometry(0.0, 0.001)
        self.assertAlmostEqual(right_angle.horizontal_sigma_min[0], 1.0, places=12)
        self.assertAlmostEqual(right_angle.horizontal_sigma_max[0], 1.0, places=12)
        self.assertAlmostEqual(
            right_angle.horizontal_condition_number[0], 1.0, places=12
        )
        self.assertLess(
            near_parallel.horizontal_sigma_min[0],
            right_angle.horizontal_sigma_min[0],
        )
        self.assertGreater(
            near_parallel.horizontal_condition_number[0], 1.0e4
        )

    def test_parallel_and_antiparallel_are_singular(self):
        for second_azimuth in (0.0, 180.0):
            geometry = _geometry(0.0, second_azimuth)
            self.assertLessEqual(geometry.horizontal_sigma_min[0], 1.0e-8)
            self.assertTrue(np.isinf(geometry.horizontal_condition_number[0]))

    def test_symmetric_bca_maps_170_degrees_to_10(self):
        geometry = _geometry(0.0, 170.0)
        self.assertAlmostEqual(geometry.symmetric_bca_deg[0], 10.0, places=12)

    def test_single_radar_has_no_false_dual_geometry(self):
        geometry = _geometry(25.0, 100.0, valid1=True, valid2=False)
        self.assertFalse(geometry.dual_valid[0])
        self.assertEqual(geometry.n_valid_radars[0], 1)
        self.assertTrue(np.isnan(geometry.symmetric_bca_deg[0]))
        self.assertTrue(np.isnan(geometry.horizontal_sigma_min[0]))
        self.assertTrue(np.isnan(geometry.horizontal_condition_number[0]))
        self.assertTrue(np.isnan(geometry.direct_3d_sigma_min[0]))
        self.assertTrue(np.isnan(geometry.null_vertical_alignment[0]))

    def test_two_radar_three_dimensional_matrix_has_rank_at_most_two(self):
        geometry = _geometry(25.0, 105.0)
        azimuth = np.deg2rad([25.0, 105.0])
        observation_matrix = np.column_stack(
            (np.sin(azimuth), np.cos(azimuth), np.zeros(2))
        )
        singular_values = np.linalg.svd(observation_matrix, compute_uv=False)
        self.assertLessEqual(np.linalg.matrix_rank(observation_matrix), 2)
        self.assertAlmostEqual(
            geometry.direct_3d_sigma_max[0], singular_values[0], places=12
        )
        self.assertAlmostEqual(
            geometry.direct_3d_sigma_min[0], singular_values[1], places=12
        )
        null_vector = np.array([
            geometry.null_direction_east[0],
            geometry.null_direction_north[0],
            geometry.null_direction_up[0],
        ])
        np.testing.assert_allclose(observation_matrix @ null_vector, 0.0, atol=1e-12)
        self.assertAlmostEqual(geometry.null_vertical_alignment[0], 1.0, places=12)


class MaskProvenanceTest(unittest.TestCase):
    def test_artificial_failure_reasons_are_partitioned(self):
        radar1 = np.array([False, True, False, True, True, True])
        radar2 = np.array([False, False, True, True, True, True])
        bca = np.array([False, False, False, False, True, True])
        downstream = np.array([True, True, True, True, False, True])
        provenance = classify_mask_provenance(
            radar1, radar2, bca, downstream_valid=downstream
        )
        counts = provenance.counts()
        self.assertEqual(counts["no_radar"], 1)
        self.assertEqual(counts["radar1_only"], 1)
        self.assertEqual(counts["radar2_only"], 1)
        self.assertEqual(counts["dual_before_bca"], 3)
        self.assertEqual(counts["rejected_by_bca"], 1)
        self.assertEqual(counts["accepted_by_bca"], 2)
        self.assertEqual(counts["rejected_downstream"], 1)
        self.assertEqual(counts["final_valid"], 1)


if __name__ == "__main__":
    unittest.main()
