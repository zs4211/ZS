"""Compare the independent D-rho operator with the frozen legacy residual."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mass_continuity import (
    AnelasticMassContinuityOperator,
    exponential_density,
)


BASELINE = ROOT / "output" / "wind3d_20260511070939Z_20260511070929Z.nc"
OUTPUT_DIR = ROOT / "artifacts" / "step3_continuity"
SCALE_HEIGHT_M = 10_000.0


def _stats(values: np.ndarray) -> dict:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {key: None for key in (
            "count", "mean", "rms", "abs_median", "abs_p95", "abs_p99", "abs_max"
        )}
    absolute = np.abs(finite)
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "rms": float(np.sqrt(np.mean(np.square(finite)))),
        "abs_median": float(np.median(absolute)),
        "abs_p95": float(np.percentile(absolute, 95)),
        "abs_p99": float(np.percentile(absolute, 99)),
        "abs_max": float(np.max(absolute)),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(BASELINE, decode_times=False) as source:
        dataset = source.load()
    x = np.asarray(dataset.x.values, dtype=np.float64)
    y = np.asarray(dataset.y.values, dtype=np.float64)
    z = np.asarray(dataset.z.values, dtype=np.float64)
    u = np.asarray(dataset.u.values, dtype=np.float64).squeeze()
    v = np.asarray(dataset.v.values, dtype=np.float64).squeeze()
    w = np.asarray(dataset.w.values, dtype=np.float64).squeeze()
    old_stored = np.asarray(
        dataset.continuity_residual.values, dtype=np.float64
    ).squeeze()
    dual_mask = np.asarray(dataset.dual_doppler_mask.values).squeeze().astype(bool)
    wind_finite = np.isfinite(u) & np.isfinite(v) & np.isfinite(w) & dual_mask

    rho = exponential_density(z, scale_height_m=SCALE_HEIGHT_M)
    operator = AnelasticMassContinuityOperator(x, y, z, rho)
    new_residual, physics_mask = operator.apply_where_valid(
        (u, v, w), wind_finite, undefined_fill=0.0
    )

    u_zero = np.where(np.isfinite(u), u, 0.0)
    v_zero = np.where(np.isfinite(v), v, 0.0)
    w_zero = np.where(np.isfinite(w), w, 0.0)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    dz = float(z[1] - z[0])
    old_recomputed = (
        np.gradient(u_zero, dx, axis=2)
        + np.gradient(v_zero, dy, axis=1)
        + np.gradient(w_zero, dz, axis=0)
        - w_zero / SCALE_HEIGHT_M
    )

    constant_density_operator = AnelasticMassContinuityOperator(
        x, y, z, np.ones_like(z)
    )
    expanded_same_difference = (
        constant_density_operator.apply((u_zero, v_zero, w_zero))
        - w_zero / SCALE_HEIGHT_M
    )
    common = physics_mask & np.isfinite(old_stored)
    mask_boundary = wind_finite & ~physics_mask

    component_medians = [
        float(np.nanmedian(component[wind_finite])) for component in (u, v, w)
    ]
    u_median = np.where(np.isfinite(u), u, component_medians[0])
    v_median = np.where(np.isfinite(v), v, component_medians[1])
    w_median = np.where(np.isfinite(w), w, component_medians[2])
    old_median_fill = (
        np.gradient(u_median, dx, axis=2)
        + np.gradient(v_median, dy, axis=1)
        + np.gradient(w_median, dz, axis=0)
        - w_median / SCALE_HEIGHT_M
    )
    fill_sensitivity = old_recomputed - old_median_fill

    old_match = old_recomputed - old_stored
    new_minus_old = new_residual - old_stored
    flux_minus_expanded = new_residual - expanded_same_difference
    derivative_scheme_delta = expanded_same_difference - old_recomputed
    correlation = None
    if np.count_nonzero(common) > 1:
        correlation = float(np.corrcoef(new_residual[common], old_stored[common])[0, 1])

    summary = {
        "case": {
            "baseline": str(BASELINE.relative_to(ROOT)),
            "grid_shape": list(operator.shape),
            "coordinate_unit": "m",
            "wind_unit": "m s-1",
            "residual_unit": "s-1",
        },
        "operator": {
            "definition": "Dx(u)+Dy(v)+rho0^-1 Dz(rho0*w)",
            "interior_difference": "centred two-point secant",
            "domain_edges": "first-order one-sided; no hard w boundary value",
            "rho0": "dimensionless exponential reference profile",
            "rho0_scale_height_m": SCALE_HEIGHT_M,
            "physics_mask": "finite/base-valid complete x,y,z derivative stencils",
        },
        "counts": {
            "domain": int(wind_finite.size),
            "dual_mask": int(np.count_nonzero(dual_mask)),
            "finite_wind": int(np.count_nonzero(wind_finite)),
            "physics_valid": int(np.count_nonzero(physics_mask)),
            "finite_but_stencil_undefined": int(np.count_nonzero(mask_boundary)),
            "physics_valid_fraction_of_finite_wind": float(
                np.count_nonzero(physics_mask) / np.count_nonzero(wind_finite)
            ),
        },
        "residual_statistics": {
            "new_Drho_physics_valid": _stats(new_residual[physics_mask]),
            "old_stored_all_finite_wind": _stats(old_stored[wind_finite]),
            "old_stored_on_common_physics_mask": _stats(old_stored[common]),
            "new_minus_old_on_common_physics_mask": _stats(new_minus_old[common]),
            "old_stored_on_mask_boundary": _stats(old_stored[mask_boundary]),
            "zero_vs_median_fill_sensitivity_on_mask_boundary": _stats(
                fill_sensitivity[mask_boundary]
            ),
        },
        "difference_attribution": {
            "old_recompute_minus_stored_on_finite": _stats(
                old_match[wind_finite & np.isfinite(old_stored)]
            ),
            "new_flux_minus_expanded_density_on_common": _stats(
                flux_minus_expanded[common]
            ),
            "new_difference_scheme_minus_numpy_gradient_on_common": _stats(
                derivative_scheme_delta[common]
            ),
            "new_old_correlation_on_common": correlation,
            "interpretation": (
                "On complete interior stencils the horizontal/vertical secants match "
                "numpy.gradient for this uniform grid. Remaining common-cell differences "
                "are the flux-form density discretisation. Finite centres without complete "
                "stencils are undefined by the new physics mask; the legacy value there "
                "depends on invalid/hidden neighbour values. Recomputing from the exported "
                "winds cannot exactly reproduce those boundary values because output masking "
                "discarded some neighbour values used by the in-memory legacy gradient."
            ),
        },
    }

    output_dataset = xr.Dataset(
        {
            "rho0": (("z",), rho.astype(np.float32)),
            "wind_finite_mask": (("z", "y", "x"), wind_finite.astype(np.uint8)),
            "physics_valid_mask": (("z", "y", "x"), physics_mask.astype(np.uint8)),
            "mask_boundary": (("z", "y", "x"), mask_boundary.astype(np.uint8)),
            "new_Drho": (("z", "y", "x"), new_residual.astype(np.float32)),
            "old_stored_continuity": (("z", "y", "x"), old_stored.astype(np.float32)),
            "old_zero_fill_recomputed": (("z", "y", "x"), old_recomputed.astype(np.float32)),
            "new_minus_old": (("z", "y", "x"), new_minus_old.astype(np.float32)),
            "old_fill_sensitivity": (("z", "y", "x"), fill_sensitivity.astype(np.float32)),
        },
        coords={"z": z, "y": y, "x": x},
        attrs={
            "scope": "Step 3 independent continuity diagnostic; no PyDDA modification",
            "rho0_source": "standard exponential approximation for comparison with legacy H=10 km",
        },
    )
    for coordinate in ("x", "y", "z"):
        output_dataset[coordinate].attrs["units"] = "m"
    for name in (
        "new_Drho", "old_stored_continuity", "old_zero_fill_recomputed",
        "new_minus_old", "old_fill_sensitivity",
    ):
        output_dataset[name].attrs["units"] = "s-1"
    encoding = {
        name: {"zlib": True, "complevel": 4, "shuffle": True}
        for name in output_dataset.data_vars
    }
    output_dataset.to_netcdf(
        OUTPUT_DIR / "continuity_comparison_20260511070939Z_20260511070929Z.nc",
        encoding=encoding,
    )
    (OUTPUT_DIR / "step3_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
