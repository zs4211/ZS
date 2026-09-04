import unittest

import numpy as np

from src.temporal_operator import (
    BilinearTranslationOperator,
    MotionCorrectedTimeOperator,
    estimate_translation_ncc,
)


class TranslationOperatorAnalyticTest(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(-6.0, 7.0)
        self.y = np.arange(-5.0, 6.0)

    def operator(self, dx, dy):
        return BilinearTranslationOperator(
            self.x, self.y, dx, dy, coordinate_unit="m"
        )

    def test_zero_shift_is_identity(self):
        rng = np.random.default_rng(1)
        field = rng.normal(size=(3, self.y.size, self.x.size))
        np.testing.assert_array_equal(self.operator(0.0, 0.0).apply(field), field)

    def test_integer_positive_x_and_negative_y_shift(self):
        field = np.zeros((self.y.size, self.x.size))
        field[5, 6] = 1.0
        shifted = self.operator(2.0, -3.0).apply(field)
        self.assertEqual(shifted[2, 8], 1.0)
        self.assertEqual(np.count_nonzero(shifted), 1)

    def test_subgrid_bilinear_weights(self):
        field = np.zeros((self.y.size, self.x.size))
        field[5, 6] = 1.0
        shifted = self.operator(0.5, 0.25).apply(field)
        self.assertAlmostEqual(shifted[5, 6], 0.375)
        self.assertAlmostEqual(shifted[5, 7], 0.375)
        self.assertAlmostEqual(shifted[6, 6], 0.125)
        self.assertAlmostEqual(shifted[6, 7], 0.125)
        self.assertAlmostEqual(float(np.sum(shifted)), 1.0)

    def test_all_x_y_direction_signs(self):
        field = np.zeros((self.y.size, self.x.size))
        field[5, 6] = 1.0
        for dx, dy in ((2, 1), (2, -1), (-2, 1), (-2, -1)):
            with self.subTest(dx=dx, dy=dy):
                shifted = self.operator(dx, dy).apply(field)
                self.assertEqual(shifted[5 + dy, 6 + dx], 1.0)

    def test_constant_is_preserved_only_on_valid_nonperiodic_overlap(self):
        field = np.ones((self.y.size, self.x.size))
        operator = self.operator(1.4, -0.7)
        valid = operator.transported_valid_mask(np.ones_like(field, dtype=bool))
        shifted = operator.apply(field)
        np.testing.assert_allclose(shifted[valid], 1.0, atol=0.0, rtol=0.0)
        self.assertTrue(np.any(~valid))
        self.assertTrue(np.any(shifted[~valid] < 1.0))

    def test_nonperiodic_boundary_does_not_wrap(self):
        field = np.zeros((self.y.size, self.x.size))
        field[5, -1] = 1.0
        shifted = self.operator(2.0, 0.0).apply(field)
        self.assertEqual(float(np.sum(shifted)), 0.0)
        self.assertEqual(shifted[5, 1], 0.0)

    def test_current_zero_extension_has_proven_negative_shift_equivalence(self):
        field = np.arange(self.y.size * self.x.size, dtype=float).reshape(
            self.y.size, self.x.size
        )
        operator = self.operator(1.4, -0.7)
        explicit_adjoint = operator.adjoint(field)
        negative_shift = self.operator(-1.4, 0.7).apply(field)
        # On a uniform grid with zero extension, matrix entries are samples of
        # an even linear-hat kernel: A_delta[i,j]=phi(i-j-delta). Therefore
        # A_delta.T[i,j]=phi(i-j+delta)=A_-delta[i,j]. The production adjoint
        # remains an explicit scatter rather than relying on this identity.
        np.testing.assert_allclose(explicit_adjoint, negative_shift, atol=1e-12, rtol=1e-14)


class TemporalOperatorAdjointTest(unittest.TestCase):
    def test_translation_random_adjoint(self):
        x = np.linspace(-9000.0, 9000.0, 19)
        y = np.linspace(-7000.0, 7000.0, 15)
        worst = 0.0
        for seed in range(10):
            rng = np.random.default_rng(seed)
            operator = BilinearTranslationOperator(
                x, y,
                rng.uniform(-3500.0, 3500.0),
                rng.uniform(-3500.0, 3500.0),
            )
            source = rng.normal(size=(3, 4, y.size, x.size))
            dual = rng.normal(size=source.shape)
            left = float(np.vdot(operator.apply(source), dual))
            right = float(np.vdot(source, operator.adjoint(dual)))
            scale = max(abs(left), abs(right), 1e-15)
            worst = max(worst, abs(left - right) / scale)
        self.assertLessEqual(worst, 1e-12)

    def test_complete_time_operator_random_adjoint(self):
        x = np.linspace(-6000.0, 6000.0, 13)
        y = np.linspace(-5000.0, 5000.0, 11)
        translations = [
            BilinearTranslationOperator(x, y, 1300.0, -800.0),
            BilinearTranslationOperator(x, y, -2100.0, 1700.0),
            BilinearTranslationOperator(x, y, 0.0, 900.0),
        ]
        operator = MotionCorrectedTimeOperator(translations)
        rng = np.random.default_rng(41)
        source = rng.normal(size=(4, 3, 2, y.size, x.size))
        dual = rng.normal(size=(3, 3, 2, y.size, x.size))
        left = float(np.vdot(operator.apply(source), dual))
        right = float(np.vdot(source, operator.adjoint(dual)))
        scale = max(abs(left), abs(right), 1e-15)
        self.assertLessEqual(abs(left - right) / scale, 1e-12)

    def test_time_mask_requires_translated_source_and_target(self):
        x = np.arange(6.0)
        y = np.arange(5.0)
        operator = MotionCorrectedTimeOperator([
            BilinearTranslationOperator(x, y, 1.0, 0.0)
        ])
        masks = np.ones((2, y.size, x.size), dtype=bool)
        masks[1, 2, 3] = False
        valid = operator.valid_pair_masks(masks)[0]
        self.assertFalse(valid[2, 0])
        self.assertFalse(valid[2, 3])
        self.assertTrue(valid[2, 2])


class MotionEstimationTest(unittest.TestCase):
    def test_moving_gaussian_vortex_displacement_recovery(self):
        x = np.arange(-40.0, 41.0)
        y = np.arange(-35.0, 36.0)
        xx, yy = np.meshgrid(x, y)
        radius2 = (xx / 8.0) ** 2 + (yy / 6.0) ** 2
        gaussian = 55.0 * np.exp(-0.5 * radius2)
        asymmetric_vortex = 8.0 * (xx / 8.0) * np.exp(-0.5 * radius2)
        source = gaussian + asymmetric_vortex
        true_dx = 2.35
        true_dy = -1.65
        target = BilinearTranslationOperator(
            x, y, true_dx, true_dy, coordinate_unit="m"
        ).apply(source)
        estimate = estimate_translation_ncc(
            source, target, x, y,
            max_displacement_m=8.0,
            reflectivity_floor_dbz=5.0,
        )
        self.assertAlmostEqual(estimate["displacement_x_m"], true_dx, delta=0.4)
        self.assertAlmostEqual(estimate["displacement_y_m"], true_dy, delta=0.4)
        self.assertGreater(estimate["peak_correlation"], 0.98)


if __name__ == "__main__":
    unittest.main()
