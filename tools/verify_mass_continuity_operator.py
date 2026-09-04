"""Numerically verify D-rho/D-rho-star and analytic Step 3 cases."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mass_continuity import AnelasticMassContinuityOperator, exponential_density


OUTPUT = ROOT / "artifacts" / "step3_continuity" / "operator_verification.json"


def relative_error(left: float, right: float) -> float:
    return float(abs(left - right) / max(abs(left), abs(right), 1.0e-15))


def main() -> None:
    adjoint_errors = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        x = np.cumsum(rng.uniform(500.0, 1700.0, 8))
        y = np.cumsum(rng.uniform(600.0, 1500.0, 7))
        z = np.cumsum(rng.uniform(250.0, 1000.0, 6))
        rho = np.exp(-(z - z[0]) / 9300.0) * rng.uniform(0.9, 1.1, z.size)
        operator = AnelasticMassContinuityOperator(x, y, z, rho)
        wind = rng.normal(size=(3,) + operator.shape)
        test = rng.normal(size=operator.shape)
        left = float(np.vdot(operator.apply(wind), test))
        right = float(np.vdot(wind, operator.adjoint(test)))
        adjoint_errors.append(relative_error(left, right))

    x = np.linspace(-3000.0, 3000.0, 7)
    y = np.linspace(-2500.0, 2500.0, 6)
    z = np.linspace(500.0, 5500.0, 5)
    rho = exponential_density(z, scale_height_m=9000.0)
    operator = AnelasticMassContinuityOperator(x, y, z, rho)
    grid_z, grid_y, grid_x = np.meshgrid(z, y, x, indexing="ij")
    zeros = np.zeros(operator.shape)
    uniform_error = float(np.max(np.abs(operator.apply((zeros + 12.0, zeros - 7.0, zeros)))))
    nondivergent_error = float(np.max(np.abs(operator.apply((grid_x, -grid_y, zeros)))))
    constant_flux_w = np.broadcast_to((2.0 / rho)[:, None, None], operator.shape)
    vertical_constant_flux_error = float(np.max(np.abs(
        operator.apply((zeros, zeros, constant_flux_w))
    )))
    linear_flux_w = np.broadcast_to((z / rho)[:, None, None], operator.shape)
    expected = np.broadcast_to((1.0 / rho)[:, None, None], operator.shape)
    vertical_linear_flux_error = float(np.max(np.abs(
        operator.apply((zeros, zeros, linear_flux_w)) - expected
    )))
    operator_km = AnelasticMassContinuityOperator(
        x / 1000.0, y / 1000.0, z / 1000.0, rho, coordinate_unit="km"
    )
    test_wind = (grid_x / 1000.0, grid_y / 2000.0, grid_z / 3000.0)
    unit_scaling_error = float(np.max(np.abs(
        operator.apply(test_wind) - operator_km.apply(test_wind)
    )))
    boundary_operator = AnelasticMassContinuityOperator(x, y, z, np.ones_like(z))
    boundary_linear_error = float(np.max(np.abs(
        boundary_operator.apply((grid_x, 2.0 * grid_y, 3.0 * grid_z)) - 6.0
    )))

    summary = {
        "adjoint": {
            "seeds": 20,
            "worst_relative_error": max(adjoint_errors),
            "mean_relative_error": float(np.mean(adjoint_errors)),
            "acceptance_limit": 1.0e-7,
        },
        "analytic_max_abs_error_s-1": {
            "uniform_horizontal": uniform_error,
            "nondivergent_horizontal": nondivergent_error,
            "vertical_constant_mass_flux": vertical_constant_flux_error,
            "vertical_linear_mass_flux": vertical_linear_flux_error,
            "metre_vs_kilometre_coordinates": unit_scaling_error,
            "one_sided_boundary_linear_field": boundary_linear_error,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

