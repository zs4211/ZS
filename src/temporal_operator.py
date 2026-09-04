"""Matrix-free horizontal translation and motion-corrected time operators.

Positive displacement moves a feature toward increasing ENU x (east) or y
(north).  For example, ``S(dx, dy) f`` samples ``f(x-dx, y-dy)``.  Bilinear
interpolation uses zero extension outside the finite, non-periodic domain.
Physical validity is handled by a separate stencil mask, not by interpreting
the zero extension as observed data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


Array = np.ndarray


def _regular_spacing(values: Array, unit: str, name: str) -> tuple[Array, float]:
    coordinates = np.asarray(values, dtype=np.float64)
    if coordinates.ndim != 1 or coordinates.size < 2:
        raise ValueError(f"{name} must be one-dimensional with at least two cells")
    if not np.all(np.isfinite(coordinates)) or not np.all(np.diff(coordinates) > 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing")
    factors = {"m": 1.0, "km": 1000.0}
    try:
        coordinates_m = coordinates * factors[unit]
    except KeyError as error:
        raise ValueError("coordinate_unit must be 'm' or 'km'") from error
    differences = np.diff(coordinates_m)
    spacing = float(np.mean(differences))
    if not np.allclose(differences, spacing, rtol=1e-10, atol=1e-8):
        raise ValueError(f"{name} must be regularly spaced for bilinear translation")
    return coordinates_m, spacing


def _axis_stencil(length: int, shift_pixels: float) -> tuple[Array, Array, Array, Array]:
    """Return lower/upper source indices and interpolation weights per output."""
    source_coordinate = np.arange(length, dtype=np.float64) - shift_pixels
    nearest = np.rint(source_coordinate)
    source_coordinate = np.where(
        np.isclose(source_coordinate, nearest, rtol=0.0, atol=1e-12),
        nearest,
        source_coordinate,
    )
    lower = np.floor(source_coordinate).astype(np.int64)
    fraction = source_coordinate - lower
    upper = lower + 1
    lower_weight = 1.0 - fraction
    upper_weight = fraction
    return lower, upper, lower_weight, upper_weight


def _sample_axis(values: Array, shift_pixels: float, axis: int) -> Array:
    field = np.moveaxis(np.asarray(values, dtype=np.float64), axis, -1)
    length = field.shape[-1]
    lower, upper, lower_weight, upper_weight = _axis_stencil(length, shift_pixels)
    output = np.zeros_like(field)
    lower_valid = (lower >= 0) & (lower < length) & (lower_weight > 0.0)
    upper_valid = (upper >= 0) & (upper < length) & (upper_weight > 0.0)
    output[..., lower_valid] += (
        field[..., lower[lower_valid]] * lower_weight[lower_valid]
    )
    output[..., upper_valid] += (
        field[..., upper[upper_valid]] * upper_weight[upper_valid]
    )
    return np.moveaxis(output, -1, axis)


def _sample_axis_adjoint(values: Array, shift_pixels: float, axis: int) -> Array:
    dual = np.moveaxis(np.asarray(values, dtype=np.float64), axis, -1)
    length = dual.shape[-1]
    lower, upper, lower_weight, upper_weight = _axis_stencil(length, shift_pixels)
    output = np.zeros_like(dual)
    lower_valid = (lower >= 0) & (lower < length) & (lower_weight > 0.0)
    upper_valid = (upper >= 0) & (upper < length) & (upper_weight > 0.0)
    output[..., lower[lower_valid]] += (
        dual[..., lower_valid] * lower_weight[lower_valid]
    )
    output[..., upper[upper_valid]] += (
        dual[..., upper_valid] * upper_weight[upper_valid]
    )
    return np.moveaxis(output, -1, axis)


def _valid_axis(mask: Array, shift_pixels: float, axis: int) -> Array:
    source = np.moveaxis(np.asarray(mask, dtype=bool), axis, -1)
    length = source.shape[-1]
    lower, upper, lower_weight, upper_weight = _axis_stencil(length, shift_pixels)
    output = np.ones_like(source, dtype=bool)
    lower_required = lower_weight > 0.0
    upper_required = upper_weight > 0.0
    lower_inside = (lower >= 0) & (lower < length)
    upper_inside = (upper >= 0) & (upper < length)
    output[..., lower_required & ~lower_inside] = False
    output[..., upper_required & ~upper_inside] = False
    select = lower_required & lower_inside
    output[..., select] &= source[..., lower[select]]
    select = upper_required & upper_inside
    output[..., select] &= source[..., upper[select]]
    return np.moveaxis(output, -1, axis)


@dataclass(frozen=True)
class BilinearTranslationOperator:
    """Non-periodic bilinear horizontal shift with an exact discrete adjoint."""

    x_m: Array
    y_m: Array
    displacement_x_m: float
    displacement_y_m: float
    shift_x_pixels: float
    shift_y_pixels: float

    def __init__(
        self,
        x: Array,
        y: Array,
        displacement_x: float,
        displacement_y: float,
        *,
        coordinate_unit: str = "m",
    ) -> None:
        x_m, dx_m = _regular_spacing(x, coordinate_unit, "x")
        y_m, dy_m = _regular_spacing(y, coordinate_unit, "y")
        factor = {"m": 1.0, "km": 1000.0}[coordinate_unit]
        displacement_x_m = float(displacement_x) * factor
        displacement_y_m = float(displacement_y) * factor
        if not np.isfinite(displacement_x_m) or not np.isfinite(displacement_y_m):
            raise ValueError("displacements must be finite")
        object.__setattr__(self, "x_m", x_m)
        object.__setattr__(self, "y_m", y_m)
        object.__setattr__(self, "displacement_x_m", displacement_x_m)
        object.__setattr__(self, "displacement_y_m", displacement_y_m)
        object.__setattr__(self, "shift_x_pixels", displacement_x_m / dx_m)
        object.__setattr__(self, "shift_y_pixels", displacement_y_m / dy_m)

    @property
    def shape_yx(self) -> tuple[int, int]:
        return self.y_m.size, self.x_m.size

    def _validate(self, values: Array) -> Array:
        field = np.asarray(values, dtype=np.float64)
        if field.ndim < 2 or field.shape[-2:] != self.shape_yx:
            raise ValueError(f"last two dimensions must be {self.shape_yx}")
        return field

    def apply(self, values: Array) -> Array:
        """Apply ``S(dx,dy)`` to a scalar field or arbitrary leading batches."""
        field = self._validate(values)
        shifted_x = _sample_axis(field, self.shift_x_pixels, axis=-1)
        return _sample_axis(shifted_x, self.shift_y_pixels, axis=-2)

    def adjoint(self, test_field: Array) -> Array:
        """Apply the exact transpose of the implemented bilinear interpolation."""
        dual = self._validate(test_field)
        after_y = _sample_axis_adjoint(dual, self.shift_y_pixels, axis=-2)
        return _sample_axis_adjoint(after_y, self.shift_x_pixels, axis=-1)

    def transported_valid_mask(self, source_mask: Array) -> Array:
        """Mark outputs whose complete nonzero interpolation stencil is valid."""
        mask = np.asarray(source_mask, dtype=bool)
        if mask.ndim < 2 or mask.shape[-2:] != self.shape_yx:
            raise ValueError(f"last two mask dimensions must be {self.shape_yx}")
        after_x = _valid_axis(mask, self.shift_x_pixels, axis=-1)
        return _valid_axis(after_x, self.shift_y_pixels, axis=-2)

    def pair_valid_mask(self, source_mask: Array, target_mask: Array) -> Array:
        target = np.asarray(target_mask, dtype=bool)
        transported = self.transported_valid_mask(source_mask)
        if target.shape != transported.shape:
            raise ValueError("source and target masks must have equal shapes")
        return target & transported


class MotionCorrectedTimeOperator:
    """First time difference ``T_c U[t] = U[t+1] - S[t] U[t]``."""

    def __init__(self, translations: Sequence[BilinearTranslationOperator]) -> None:
        self.translations = tuple(translations)
        if not self.translations:
            raise ValueError("at least one translation is required")
        shapes = {translation.shape_yx for translation in self.translations}
        if len(shapes) != 1:
            raise ValueError("all translations must use the same horizontal grid")

    @property
    def n_times(self) -> int:
        return len(self.translations) + 1

    def _validate_series(self, values: Array, expected_time: int) -> Array:
        series = np.asarray(values, dtype=np.float64)
        if series.ndim < 3 or series.shape[0] != expected_time:
            raise ValueError(f"first dimension must have length {expected_time}")
        if series.shape[-2:] != self.translations[0].shape_yx:
            raise ValueError("series horizontal shape does not match translations")
        return series

    def apply(self, values: Array) -> Array:
        series = self._validate_series(values, self.n_times)
        return np.stack([
            series[index + 1] - translation.apply(series[index])
            for index, translation in enumerate(self.translations)
        ])

    def adjoint(self, test_field: Array) -> Array:
        dual = self._validate_series(test_field, self.n_times - 1)
        output = np.zeros((self.n_times,) + dual.shape[1:], dtype=np.float64)
        for index, translation in enumerate(self.translations):
            output[index] -= translation.adjoint(dual[index])
            output[index + 1] += dual[index]
        return output

    def valid_pair_masks(self, time_masks: Array) -> Array:
        masks = np.asarray(time_masks, dtype=bool)
        if masks.shape[0] != self.n_times:
            raise ValueError(f"first mask dimension must have length {self.n_times}")
        return np.stack([
            translation.pair_valid_mask(masks[index], masks[index + 1])
            for index, translation in enumerate(self.translations)
        ])


def estimate_translation_ncc(
    source: Array,
    target: Array,
    x: Array,
    y: Array,
    *,
    max_displacement_m: float = 30_000.0,
    reflectivity_floor_dbz: float = 20.0,
    refine_subpixel: bool = True,
) -> dict:
    """Estimate source-to-target translation by explicit normalized correlation.

    The correlated feature is positive reflectivity excess above the prescribed
    floor. Search candidates are integer grid displacements. A local parabolic
    peak interpolation optionally estimates a sub-grid displacement.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("source and target must be equal-shape two-dimensional arrays")
    x_m, dx_m = _regular_spacing(x, "m", "x")
    y_m, dy_m = _regular_spacing(y, "m", "y")
    if source.shape != (y_m.size, x_m.size):
        raise ValueError("reflectivity fields do not match x/y coordinates")
    max_x = int(np.floor(max_displacement_m / dx_m))
    max_y = int(np.floor(max_displacement_m / dy_m))
    x_shifts = np.arange(-max_x, max_x + 1, dtype=int)
    y_shifts = np.arange(-max_y, max_y + 1, dtype=int)
    scores = np.full((y_shifts.size, x_shifts.size), np.nan, dtype=np.float64)
    overlaps = np.full_like(scores, np.nan)
    finite_overlaps = np.full_like(scores, np.nan)

    source_feature = np.maximum(source - reflectivity_floor_dbz, 0.0)
    target_feature = np.maximum(target - reflectivity_floor_dbz, 0.0)
    finite_reference_count = min(
        int(np.count_nonzero(np.isfinite(source))),
        int(np.count_nonzero(np.isfinite(target))),
    )
    for row, shift_y in enumerate(y_shifts):
        if shift_y >= 0:
            source_y = slice(0, source.shape[0] - shift_y)
            target_y = slice(shift_y, target.shape[0])
        else:
            source_y = slice(-shift_y, source.shape[0])
            target_y = slice(0, target.shape[0] + shift_y)
        for column, shift_x in enumerate(x_shifts):
            if shift_x >= 0:
                source_x = slice(0, source.shape[1] - shift_x)
                target_x = slice(shift_x, target.shape[1])
            else:
                source_x = slice(-shift_x, source.shape[1])
                target_x = slice(0, target.shape[1] + shift_x)
            a = source_feature[source_y, source_x]
            b = target_feature[target_y, target_x]
            finite = (
                np.isfinite(source[source_y, source_x])
                & np.isfinite(target[target_y, target_x])
            )
            finite_overlaps[row, column] = (
                float(np.count_nonzero(finite) / finite_reference_count)
                if finite_reference_count else np.nan
            )
            union = finite & ((a > 0.0) | (b > 0.0))
            both = finite & (a > 0.0) & (b > 0.0)
            if np.count_nonzero(union) < 25:
                continue
            a_values = a[union]
            b_values = b[union]
            denominator = np.linalg.norm(a_values) * np.linalg.norm(b_values)
            if denominator <= 0.0:
                continue
            scores[row, column] = float(np.dot(a_values, b_values) / denominator)
            overlaps[row, column] = float(
                np.count_nonzero(both) / np.count_nonzero(union)
            )

    if not np.any(np.isfinite(scores)):
        raise RuntimeError("no valid normalized-correlation displacement candidate")
    peak_flat = int(np.nanargmax(scores))
    peak_row, peak_column = np.unravel_index(peak_flat, scores.shape)
    integer_y = int(y_shifts[peak_row])
    integer_x = int(x_shifts[peak_column])

    def parabolic_offset(left: float, centre: float, right: float) -> float:
        if not np.all(np.isfinite([left, centre, right])):
            return 0.0
        denominator = left - 2.0 * centre + right
        if denominator >= 0.0 or abs(denominator) < 1e-15:
            return 0.0
        return float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))

    sub_x = 0.0
    sub_y = 0.0
    if refine_subpixel and 0 < peak_column < scores.shape[1] - 1:
        sub_x = parabolic_offset(
            scores[peak_row, peak_column - 1], scores[peak_row, peak_column],
            scores[peak_row, peak_column + 1],
        )
    if refine_subpixel and 0 < peak_row < scores.shape[0] - 1:
        sub_y = parabolic_offset(
            scores[peak_row - 1, peak_column], scores[peak_row, peak_column],
            scores[peak_row + 1, peak_column],
        )

    excluded = scores.copy()
    excluded[
        max(0, peak_row - 1):min(scores.shape[0], peak_row + 2),
        max(0, peak_column - 1):min(scores.shape[1], peak_column + 2),
    ] = np.nan
    second_peak = float(np.nanmax(excluded)) if np.any(np.isfinite(excluded)) else np.nan
    peak = float(scores[peak_row, peak_column])
    overlap = float(overlaps[peak_row, peak_column])
    peak_margin = peak - second_peak if np.isfinite(second_peak) else np.nan
    at_search_boundary = bool(
        peak_row in (0, scores.shape[0] - 1)
        or peak_column in (0, scores.shape[1] - 1)
    )
    if peak >= 0.80 and overlap >= 0.45 and peak_margin >= 0.02 and not at_search_boundary:
        confidence = "high"
    elif peak >= 0.60 and overlap >= 0.25 and not at_search_boundary:
        confidence = "medium"
    else:
        confidence = "low"
    return {
        "displacement_x_m": float((integer_x + sub_x) * dx_m),
        "displacement_y_m": float((integer_y + sub_y) * dy_m),
        "integer_shift_x_cells": integer_x,
        "integer_shift_y_cells": integer_y,
        "subpixel_offset_x_cells": sub_x,
        "subpixel_offset_y_cells": sub_y,
        "peak_correlation": peak,
        "second_peak_correlation": second_peak,
        "peak_margin": float(peak_margin),
        "storm_overlap_fraction": overlap,
        "finite_valid_overlap_fraction": float(finite_overlaps[peak_row, peak_column]),
        "at_search_boundary": at_search_boundary,
        "confidence": confidence,
        "reflectivity_floor_dbz": float(reflectivity_floor_dbz),
        "max_displacement_m": float(max_displacement_m),
    }
