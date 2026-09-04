"""Tests for conservative mesocyclone candidate screening."""

from types import SimpleNamespace
import unittest

import numpy as np

from visualize_3d import assess_retrieval_quality, detect_mesocyclones


class MesocycloneDetectionTest(unittest.TestCase):
    def setUp(self):
        self.grid = SimpleNamespace(
            x=np.arange(-12.0, 15.0, 3.0),
            y=np.arange(-12.0, 15.0, 3.0),
            z=np.arange(1.0, 6.0, 1.0),
            dx=3.0, dy=3.0, dz=1.0,
            nx=9, ny=9, nz=5,
        )

    def test_requires_cyclonic_observed_reflective_vertical_column(self):
        shape = (9, 9, 5)
        vorticity = np.zeros(shape)
        valid = np.ones(shape, dtype=bool)
        reflectivity = np.full(shape, 35.0)

        # A valid cyclonic column, allowed to tilt one cell with height.
        for k in range(5):
            vorticity[4 + (k > 2), 4, k] = 0.006

        # Stronger distractors must be rejected: anticyclonic, no echo, no data.
        vorticity[1, 1, :] = -0.02
        vorticity[7, 7, :] = 0.02
        reflectivity[7, 7, :] = 5.0
        vorticity[1, 7, :] = 0.02
        valid[1, 7, :] = False

        candidates = detect_mesocyclones(
            vorticity, self.grid,
            valid_mask=valid,
            reflectivity=reflectivity,
        )
        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0][0], 0.0, delta=3.0)
        self.assertAlmostEqual(candidates[0][1], 0.0, delta=3.0)

    def test_weak_rotation_is_not_reported(self):
        shape = (9, 9, 5)
        candidates = detect_mesocyclones(
            np.full(shape, 0.003), self.grid,
            valid_mask=np.ones(shape, dtype=bool),
            reflectivity=np.full(shape, 40.0),
        )
        self.assertEqual(candidates, [])

    def test_horizontal_boundary_peak_is_rejected(self):
        shape = (9, 9, 5)
        vorticity = np.zeros(shape)
        vorticity[0, 4, :] = 0.02
        candidates = detect_mesocyclones(
            vorticity, self.grid,
            valid_mask=np.ones(shape, dtype=bool),
            reflectivity=np.full(shape, 40.0),
        )
        self.assertEqual(candidates, [])

    def test_weak_echo_rotation_is_rejected(self):
        shape = (9, 9, 5)
        vorticity = np.zeros(shape)
        vorticity[4, 4, :] = 0.01
        candidates = detect_mesocyclones(
            vorticity, self.grid,
            valid_mask=np.ones(shape, dtype=bool),
            reflectivity=np.full(shape, 25.0),
        )
        self.assertEqual(candidates, [])

    def test_unstable_vertical_wind_fails_quality_gate(self):
        shape = (9, 9, 5)
        data = {
            "w": np.full(shape, 20.0),
            "dual_doppler_mask": np.ones(shape, dtype=bool),
            "continuity_residual": np.full(shape, 0.006),
        }
        quality = assess_retrieval_quality(data)
        self.assertFalse(quality["ok"])
        self.assertEqual(len(quality["issues"]), 2)


if __name__ == "__main__":
    unittest.main()
