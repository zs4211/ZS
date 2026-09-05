import unittest

import numpy as np

from src.low_rank_diagnostics import (
    largest_true_cuboid,
    longest_true_run,
    masked_st_hosvd,
    mode_spectrum,
    relative_error,
    st_hosvd,
    tucker_parameter_count,
    vertical_vorticity_where_valid,
)


class TestLowRankDiagnostics(unittest.TestCase):
    def test_exact_low_rank_reconstruction(self):
        rng = np.random.default_rng(4)
        core = rng.normal(size=(2, 2, 1))
        factors = [rng.normal(size=(7, 2)), rng.normal(size=(6, 2)), rng.normal(size=(5, 1))]
        tensor = np.einsum("abc,ia,jb,kc->ijk", core, *factors)
        result = st_hosvd(tensor, (2, 2, 1))
        self.assertLess(relative_error(tensor, result.reconstruction), 1e-12)

    def test_spectrum_and_parameter_count(self):
        tensor = np.einsum("i,j,k->ijk", np.arange(1, 5), np.ones(3), np.ones(2))
        spectrum = mode_spectrum(tensor, 0)
        self.assertEqual(spectrum["rank_99"], 1)
        self.assertAlmostEqual(spectrum["effective_rank"], 1.0)
        self.assertEqual(tucker_parameter_count((4, 3, 2), (1, 1, 1)), 10)

    def test_largest_true_cuboid_is_complete(self):
        mask = np.zeros((4, 7, 8), dtype=bool)
        mask[1:4, 2:6, 3:8] = True
        mask[0, 0:2, 0:2] = True
        selection = largest_true_cuboid(mask)
        selected = mask[selection]
        self.assertTrue(np.all(selected))
        self.assertEqual(selected.size, 3 * 4 * 5)

    def test_masked_fit_does_not_treat_missing_as_zero(self):
        x = np.arange(6.0)[:, None, None]
        y = np.arange(5.0)[None, :, None]
        z = np.arange(3.0)[None, None, :]
        tensor = 10.0 + x + 2.0 * y + 0.5 * z
        mask = np.ones(tensor.shape, dtype=bool)
        mask[2:5, 1:4, :] = False
        result = masked_st_hosvd(tensor, mask, tensor.shape, iterations=2)
        self.assertLess(relative_error(tensor, result.reconstruction, mask), 1e-12)
        self.assertGreater(float(np.mean(result.reconstruction[~mask])), 1.0)

    def test_vertical_vorticity_uses_step3_differences_and_mask(self):
        x = np.linspace(-2000.0, 2000.0, 5)
        y = np.linspace(-3000.0, 3000.0, 7)
        z = np.arange(3.0)
        xx, yy = np.meshgrid(x, y, indexing="xy")
        u = np.broadcast_to(-2.0 * yy, (z.size,) + yy.shape).copy()
        v = np.broadcast_to(3.0 * xx, (z.size,) + xx.shape).copy()
        mask = np.ones_like(u, dtype=bool)
        vorticity, valid = vertical_vorticity_where_valid(u, v, x, y, mask)
        self.assertTrue(np.all(valid))
        np.testing.assert_allclose(vorticity, 5.0, atol=1e-12)
        mask[:, 3, 2] = False
        _, valid = vertical_vorticity_where_valid(u, v, x, y, mask)
        self.assertFalse(valid[:, 3, 2].any())
        self.assertEqual(longest_true_run([False, True, True, False, True]), 2)


if __name__ == "__main__":
    unittest.main()
