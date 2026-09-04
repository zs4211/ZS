import unittest

import numpy as np

from src.mass_continuity import (
    AnelasticMassContinuityOperator,
    exponential_density,
)


class MassContinuityAdjointTest(unittest.TestCase):
    def test_random_fields_match_exact_adjoint(self):
        for seed in range(5):
            rng = np.random.default_rng(seed)
            x = np.cumsum(rng.uniform(600.0, 1800.0, 7))
            y = np.cumsum(rng.uniform(700.0, 1700.0, 6))
            z = np.cumsum(rng.uniform(300.0, 900.0, 5))
            rho = np.exp(-(z - z[0]) / 8500.0) * rng.uniform(0.95, 1.05, z.size)
            operator = AnelasticMassContinuityOperator(x, y, z, rho)
            wind = rng.normal(size=(3, z.size, y.size, x.size))
            test_field = rng.normal(size=operator.shape)
            left = np.vdot(operator.apply(wind), test_field)
            right = np.vdot(wind, operator.adjoint(test_field))
            relative = abs(left - right) / max(abs(left), abs(right), 1.0e-15)
            self.assertLessEqual(relative, 1.0e-12)

    def test_random_output_mask_matches_masked_adjoint(self):
        rng = np.random.default_rng(20260904)
        coordinates = np.arange(6, dtype=float) * 1000.0
        operator = AnelasticMassContinuityOperator(
            coordinates, coordinates, coordinates,
            exponential_density(coordinates),
        )
        wind = rng.normal(size=(3, 6, 6, 6))
        test_field = rng.normal(size=operator.shape)
        output_mask = rng.random(operator.shape) > 0.35
        masked_test = np.where(output_mask, test_field, 0.0)
        left = np.vdot(np.where(output_mask, operator.apply(wind), 0.0), test_field)
        right = np.vdot(wind, operator.adjoint(masked_test))
        relative = abs(left - right) / max(abs(left), abs(right), 1.0e-15)
        self.assertLessEqual(relative, 1.0e-12)


class MassContinuityAnalyticTest(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(-3000.0, 3000.0, 7)
        self.y = np.linspace(-2500.0, 2500.0, 6)
        self.z = np.linspace(500.0, 5500.0, 5)
        self.rho = exponential_density(self.z, scale_height_m=9000.0)
        self.operator = AnelasticMassContinuityOperator(
            self.x, self.y, self.z, self.rho
        )
        self.grid_z, self.grid_y, self.grid_x = np.meshgrid(
            self.z, self.y, self.x, indexing="ij"
        )

    def test_uniform_horizontal_wind(self):
        wind = (
            np.full(self.operator.shape, 12.0),
            np.full(self.operator.shape, -7.0),
            np.zeros(self.operator.shape),
        )
        np.testing.assert_allclose(self.operator.apply(wind), 0.0, atol=1.0e-15)

    def test_analytic_nondivergent_horizontal_wind(self):
        # psi=x*y gives u=dpsi/dy=x and v=-dpsi/dx=-y.
        u = self.grid_x.copy()
        v = -self.grid_y.copy()
        w = np.zeros_like(u)
        np.testing.assert_allclose(self.operator.apply((u, v, w)), 0.0, atol=1.0e-15)

    def test_vertical_flux_form_analytic_field(self):
        # rho*w=constant is exactly anelastic non-divergent in this discretisation.
        w = np.broadcast_to((2.0 / self.rho)[:, None, None], self.operator.shape)
        zeros = np.zeros(self.operator.shape)
        np.testing.assert_allclose(
            self.operator.apply((zeros, zeros, w)), 0.0, atol=1.0e-15
        )
        # rho*w=z has derivative 1, including the one-sided physical boundaries.
        linear_flux_w = np.broadcast_to(
            (self.z / self.rho)[:, None, None], self.operator.shape
        )
        expected = np.broadcast_to((1.0 / self.rho)[:, None, None], self.operator.shape)
        np.testing.assert_allclose(
            self.operator.apply((zeros, zeros, linear_flux_w)), expected,
            rtol=2.0e-15, atol=2.0e-15,
        )

    def test_coordinate_unit_scaling(self):
        operator_km = AnelasticMassContinuityOperator(
            self.x / 1000.0, self.y / 1000.0, self.z / 1000.0,
            self.rho, coordinate_unit="km",
        )
        wind = (self.grid_x / 1000.0, self.grid_y / 2000.0, self.grid_z / 3000.0)
        np.testing.assert_allclose(
            self.operator.apply(wind), operator_km.apply(wind),
            rtol=0.0, atol=1.0e-15,
        )

    def test_one_sided_boundaries_are_explicit_and_linear_exact(self):
        rho_one = np.ones_like(self.z)
        operator = AnelasticMassContinuityOperator(self.x, self.y, self.z, rho_one)
        u = self.grid_x
        v = 2.0 * self.grid_y
        w = 3.0 * self.grid_z
        np.testing.assert_allclose(operator.apply((u, v, w)), 6.0, atol=1.0e-15)

    def test_artificial_physics_mask_excludes_cross_hole_stencils(self):
        rng = np.random.default_rng(42)
        wind = rng.normal(size=(3,) + self.operator.shape)
        base = np.ones(self.operator.shape, dtype=bool)
        hole = (2, 3, 3)
        base[hole] = False
        valid = self.operator.physical_valid_mask(wind, base)
        expected_invalid = {
            hole,
            (2, 3, 2), (2, 3, 4),
            (2, 2, 3), (2, 4, 3),
            (1, 3, 3), (3, 3, 3),
        }
        for index in expected_invalid:
            self.assertFalse(valid[index])
        residual_zero, mask_zero = self.operator.apply_where_valid(
            wind, base, undefined_fill=0.0
        )
        residual_large, mask_large = self.operator.apply_where_valid(
            wind, base, undefined_fill=1.0e9
        )
        np.testing.assert_array_equal(mask_zero, mask_large)
        np.testing.assert_allclose(
            residual_zero[valid], residual_large[valid], rtol=0.0, atol=0.0
        )
        self.assertTrue(np.isnan(residual_zero[~valid]).all())

    def test_zero_fill_creates_artificial_mask_edge_divergence(self):
        rho_one = np.ones_like(self.z)
        operator = AnelasticMassContinuityOperator(self.x, self.y, self.z, rho_one)
        u = np.full(self.operator.shape, np.nan)
        u[:, :, 2:5] = 10.0
        v = np.zeros(self.operator.shape)
        w = np.zeros(self.operator.shape)
        old_zero_fill = np.gradient(
            np.nan_to_num(u, nan=0.0), self.x[1] - self.x[0], axis=2
        )
        self.assertGreater(np.max(np.abs(old_zero_fill[:, :, (2, 4)])), 0.0)
        residual, valid = operator.apply_where_valid((u, v, w))
        self.assertTrue(np.isnan(residual[:, :, (2, 4)]).all())
        np.testing.assert_allclose(residual[valid], 0.0, atol=0.0)

    def test_density_profile_validation_and_callable_interface(self):
        callable_operator = AnelasticMassContinuityOperator(
            self.x, self.y, self.z,
            lambda z: exponential_density(z, scale_height_m=9000.0),
        )
        np.testing.assert_allclose(callable_operator.rho0, self.rho)
        bad_density = self.rho.copy()
        bad_density[2] = 0.0
        with self.assertRaises(ValueError):
            AnelasticMassContinuityOperator(self.x, self.y, self.z, bad_density)


if __name__ == "__main__":
    unittest.main()
