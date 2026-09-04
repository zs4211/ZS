"""Matrix-free discrete anelastic mass-continuity operator and adjoint.

Coordinates are cell-centre coordinates.  Interior derivatives use centred
two-point secants and physical domain edges use first-order one-sided
differences.  No boundary value (including w=0) is imposed by this operator.

The operator is

    D_rho(u, v, w) = Dx(u) + Dy(v) + rho^-1 Dz(rho w),

which is the flux-form discretisation of rho^-1 div(rho V).  ``rho`` depends
only on z and must be supplied explicitly as a positive array or callable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


Array = np.ndarray
DensityProfile = Sequence[float] | Array | Callable[[Array], Array]


def exponential_density(
    z_m: Array,
    *,
    scale_height_m: float = 10_000.0,
    reference_density: float = 1.0,
) -> Array:
    """Return a positive exponential reference-density profile.

    The absolute density scale cancels from ``D_rho``; ``reference_density``
    is retained so a physical sounding/model density can use the same API.
    """
    z = np.asarray(z_m, dtype=np.float64)
    if scale_height_m <= 0.0 or reference_density <= 0.0:
        raise ValueError("scale_height_m and reference_density must be positive")
    return reference_density * np.exp(-(z - z[0]) / float(scale_height_m))


def _coordinates_in_metres(values: Array, unit: str, name: str) -> Array:
    coordinates = np.asarray(values, dtype=np.float64)
    if coordinates.ndim != 1 or coordinates.size < 2:
        raise ValueError(f"{name} must be a one-dimensional array with at least two cells")
    if not np.all(np.isfinite(coordinates)) or not np.all(np.diff(coordinates) > 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing")
    factors = {"m": 1.0, "km": 1000.0}
    try:
        return coordinates * factors[unit]
    except KeyError as error:
        raise ValueError("coordinate_unit must be 'm' or 'km'") from error


def _derivative(values: Array, coordinates_m: Array, axis: int) -> Array:
    field = np.moveaxis(np.asarray(values, dtype=np.float64), axis, 0)
    if field.shape[0] != coordinates_m.size:
        raise ValueError("field shape does not match coordinate length")
    result = np.empty_like(field)
    lower_spacing = coordinates_m[1] - coordinates_m[0]
    upper_spacing = coordinates_m[-1] - coordinates_m[-2]
    result[0] = (field[1] - field[0]) / lower_spacing
    result[-1] = (field[-1] - field[-2]) / upper_spacing
    denominator = coordinates_m[2:] - coordinates_m[:-2]
    reshape = (denominator.size,) + (1,) * (field.ndim - 1)
    result[1:-1] = (field[2:] - field[:-2]) / denominator.reshape(reshape)
    return np.moveaxis(result, 0, axis)


def _derivative_adjoint(test_field: Array, coordinates_m: Array, axis: int) -> Array:
    dual = np.moveaxis(np.asarray(test_field, dtype=np.float64), axis, 0)
    if dual.shape[0] != coordinates_m.size:
        raise ValueError("test-field shape does not match coordinate length")
    result = np.zeros_like(dual)
    lower_spacing = coordinates_m[1] - coordinates_m[0]
    upper_spacing = coordinates_m[-1] - coordinates_m[-2]
    result[0] -= dual[0] / lower_spacing
    result[1] += dual[0] / lower_spacing
    result[-2] -= dual[-1] / upper_spacing
    result[-1] += dual[-1] / upper_spacing
    denominator = coordinates_m[2:] - coordinates_m[:-2]
    reshape = (denominator.size,) + (1,) * (dual.ndim - 1)
    scaled = dual[1:-1] / denominator.reshape(reshape)
    result[:-2] -= scaled
    result[2:] += scaled
    return np.moveaxis(result, 0, axis)


def _stencil_valid(mask: Array, axis: int) -> Array:
    source = np.moveaxis(np.asarray(mask, dtype=bool), axis, 0)
    valid = np.zeros_like(source)
    valid[0] = source[0] & source[1]
    valid[-1] = source[-2] & source[-1]
    valid[1:-1] = source[:-2] & source[1:-1] & source[2:]
    return np.moveaxis(valid, 0, axis)


@dataclass(frozen=True)
class AnelasticMassContinuityOperator:
    """Matrix-free ``D_rho`` with its exact Euclidean discrete adjoint."""

    x_m: Array
    y_m: Array
    z_m: Array
    rho0: Array

    def __init__(
        self,
        x: Array,
        y: Array,
        z: Array,
        rho0: DensityProfile,
        *,
        coordinate_unit: str = "m",
    ) -> None:
        x_m = _coordinates_in_metres(x, coordinate_unit, "x")
        y_m = _coordinates_in_metres(y, coordinate_unit, "y")
        z_m = _coordinates_in_metres(z, coordinate_unit, "z")
        density = rho0(z_m) if callable(rho0) else np.asarray(rho0, dtype=np.float64)
        density = np.asarray(density, dtype=np.float64)
        if density.shape != z_m.shape:
            raise ValueError("rho0 must return/provide one value per z level")
        if not np.all(np.isfinite(density)) or np.any(density <= 0.0):
            raise ValueError("rho0 must be finite and strictly positive")
        object.__setattr__(self, "x_m", x_m)
        object.__setattr__(self, "y_m", y_m)
        object.__setattr__(self, "z_m", z_m)
        object.__setattr__(self, "rho0", density)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.z_m.size, self.y_m.size, self.x_m.size

    def _wind(self, wind) -> tuple[Array, Array, Array]:
        if isinstance(wind, (tuple, list)) and len(wind) == 3:
            components = tuple(np.asarray(item, dtype=np.float64) for item in wind)
        else:
            array = np.asarray(wind, dtype=np.float64)
            if array.ndim != 4 or array.shape[0] != 3:
                raise ValueError("wind must be (u,v,w) or an array with shape (3,nz,ny,nx)")
            components = (array[0], array[1], array[2])
        if any(component.shape != self.shape for component in components):
            raise ValueError(f"each wind component must have shape {self.shape}")
        return components

    def apply(self, wind) -> Array:
        """Apply ``D_rho``; input winds are m/s and output is s^-1."""
        u, v, w = self._wind(wind)
        rho = self.rho0[:, None, None]
        return (
            _derivative(u, self.x_m, axis=2)
            + _derivative(v, self.y_m, axis=1)
            + _derivative(rho * w, self.z_m, axis=0) / rho
        )

    def adjoint(self, test_field: Array) -> Array:
        """Apply the exact transpose under the ordinary discrete dot product."""
        dual = np.asarray(test_field, dtype=np.float64)
        if dual.shape != self.shape:
            raise ValueError(f"test_field must have shape {self.shape}")
        rho = self.rho0[:, None, None]
        return np.stack((
            _derivative_adjoint(dual, self.x_m, axis=2),
            _derivative_adjoint(dual, self.y_m, axis=1),
            rho * _derivative_adjoint(dual / rho, self.z_m, axis=0),
        ))

    def physical_valid_mask(self, wind, base_mask: Array | None = None) -> Array:
        """Return cells whose complete D-rho stencil is physically defined.

        The mask is separate from ``apply``.  At holes it removes the hole and
        every output cell whose derivative stencil would touch that hole.
        External domain edges remain eligible through the documented one-sided
        stencils.  No undefined residual is replaced by a physical zero.
        """
        u, v, w = self._wind(wind)
        if base_mask is None:
            base = np.ones(self.shape, dtype=bool)
        else:
            base = np.broadcast_to(np.asarray(base_mask, dtype=bool), self.shape)
        x_valid = _stencil_valid(base & np.isfinite(u), axis=2)
        y_valid = _stencil_valid(base & np.isfinite(v), axis=1)
        z_valid = _stencil_valid(base & np.isfinite(w), axis=0)
        return x_valid & y_valid & z_valid

    def apply_where_valid(
        self,
        wind,
        base_mask: Array | None = None,
        *,
        undefined_fill: float = 0.0,
    ) -> tuple[Array, Array]:
        """Return ``(residual, mask)`` with invalid residuals represented by NaN.

        ``undefined_fill`` is computational only: the returned valid stencils
        never touch a filled input, which is tested independently.
        """
        u, v, w = self._wind(wind)
        mask = self.physical_valid_mask((u, v, w), base_mask)
        if base_mask is None:
            base = np.ones(self.shape, dtype=bool)
        else:
            base = np.broadcast_to(np.asarray(base_mask, dtype=bool), self.shape)
        filled = tuple(
            np.where(base & np.isfinite(component), component, undefined_fill)
            for component in (u, v, w)
        )
        residual = self.apply(filled)
        return np.where(mask, residual, np.nan), mask

