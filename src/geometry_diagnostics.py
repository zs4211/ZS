"""Dual-radar coverage and direct-observability diagnostics.

This module deliberately separates the two-dimensional horizontal inversion
geometry from the rank-deficient three-dimensional, two-radar observation
matrix. It computes diagnostics only; it does not change QC, gridding, BCA
thresholds, or the PyDDA solver.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from src.observation_operator import beam_direction_cosines


@dataclass(frozen=True)
class GeometryDiagnostics:
    radar1_valid: np.ndarray
    radar2_valid: np.ndarray
    n_valid_radars: np.ndarray
    dual_valid: np.ndarray
    symmetric_bca_deg: np.ndarray
    horizontal_sigma_min: np.ndarray
    horizontal_sigma_max: np.ndarray
    horizontal_inverse_condition: np.ndarray
    horizontal_condition_number: np.ndarray
    beam_elevation_radar1_deg: np.ndarray
    beam_elevation_radar2_deg: np.ndarray
    radar_range_radar1_m: np.ndarray
    radar_range_radar2_m: np.ndarray
    direct_3d_sigma_min: np.ndarray
    direct_3d_sigma_max: np.ndarray
    null_direction_east: np.ndarray
    null_direction_north: np.ndarray
    null_direction_up: np.ndarray
    null_vertical_alignment: np.ndarray
    final_bca_mask: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


def _masked_float(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return np.where(valid, np.asarray(values, dtype=np.float64), np.nan)


def diagnose_dual_radar_geometry(
    azimuth_radar1_deg: np.ndarray,
    elevation_radar1_deg: np.ndarray,
    range_radar1_m: np.ndarray,
    radar1_valid: np.ndarray,
    azimuth_radar2_deg: np.ndarray,
    elevation_radar2_deg: np.ndarray,
    range_radar2_m: np.ndarray,
    radar2_valid: np.ndarray,
    *,
    bca_threshold_deg: float = 20.0,
) -> GeometryDiagnostics:
    """Diagnose horizontal geometry and the observed 3-D row space.

    The two rows of the 3-D observation matrix can have rank at most two.
    Consequently, this routine reports its two row-space singular values and
    a null direction; it never forms or reports a misleading ordinary 3-D
    condition number for ``H.T @ H``.
    """
    arrays = np.broadcast_arrays(
        np.asarray(azimuth_radar1_deg, dtype=np.float64),
        np.asarray(elevation_radar1_deg, dtype=np.float64),
        np.asarray(range_radar1_m, dtype=np.float64),
        np.asarray(radar1_valid, dtype=bool),
        np.asarray(azimuth_radar2_deg, dtype=np.float64),
        np.asarray(elevation_radar2_deg, dtype=np.float64),
        np.asarray(range_radar2_m, dtype=np.float64),
        np.asarray(radar2_valid, dtype=bool),
    )
    az1, el1, range1, valid1, az2, el2, range2, valid2 = arrays
    finite_geometry1 = np.isfinite(az1) & np.isfinite(el1) & np.isfinite(range1)
    finite_geometry2 = np.isfinite(az2) & np.isfinite(el2) & np.isfinite(range2)
    valid1 = valid1 & finite_geometry1 & (range1 > 0.0)
    valid2 = valid2 & finite_geometry2 & (range2 > 0.0)
    dual = valid1 & valid2
    count = valid1.astype(np.uint8) + valid2.astype(np.uint8)

    east1, north1, up1 = beam_direction_cosines(az1, el1, degrees=True)
    east2, north2, up2 = beam_direction_cosines(az2, el2, degrees=True)

    horizontal_norm1 = np.hypot(east1, north1)
    horizontal_norm2 = np.hypot(east2, north2)
    horizontal_defined = dual & (horizontal_norm1 > 0.0) & (horizontal_norm2 > 0.0)
    horizontal_dot = (
        east1 * east2 + north1 * north2
    ) / np.where(horizontal_norm1 * horizontal_norm2 > 0.0,
                 horizontal_norm1 * horizontal_norm2, 1.0)
    symmetric_bca = np.rad2deg(
        np.arccos(np.clip(np.abs(horizontal_dot), 0.0, 1.0))
    )

    # Singular values of A_h from eigenvalues of A_h.T @ A_h.
    gram_ee = east1 * east1 + east2 * east2
    gram_nn = north1 * north1 + north2 * north2
    gram_en = east1 * north1 + east2 * north2
    trace = gram_ee + gram_nn
    discriminant = np.sqrt(
        np.maximum((gram_ee - gram_nn) ** 2 + 4.0 * gram_en * gram_en, 0.0)
    )
    eigen_max = np.maximum(0.5 * (trace + discriminant), 0.0)
    eigen_min = np.maximum(0.5 * (trace - discriminant), 0.0)
    horizontal_sigma_max = np.sqrt(eigen_max)
    horizontal_sigma_min = np.sqrt(eigen_min)
    inverse_condition = np.divide(
        horizontal_sigma_min,
        horizontal_sigma_max,
        out=np.full_like(horizontal_sigma_min, np.nan),
        where=horizontal_sigma_max > 0.0,
    )
    condition_number = np.divide(
        horizontal_sigma_max,
        horizontal_sigma_min,
        out=np.full_like(horizontal_sigma_max, np.inf),
        where=horizontal_sigma_min > np.finfo(np.float64).eps,
    )

    # The two singular values of the 2x3 H follow from H @ H.T. Each row is
    # a unit beam vector. A third singular value is identically absent/zero.
    beam_dot = east1 * east2 + north1 * north2 + up1 * up2
    absolute_beam_dot = np.clip(np.abs(beam_dot), 0.0, 1.0)
    direct_sigma_max = np.sqrt(1.0 + absolute_beam_dot)
    direct_sigma_min = np.sqrt(np.maximum(1.0 - absolute_beam_dot, 0.0))

    cross_east = north1 * up2 - up1 * north2
    cross_north = up1 * east2 - east1 * up2
    cross_up = east1 * north2 - north1 * east2
    cross_norm = np.sqrt(
        cross_east * cross_east
        + cross_north * cross_north
        + cross_up * cross_up
    )
    unique_null = dual & (cross_norm > 1.0e-12)
    null_east = np.divide(
        cross_east, cross_norm,
        out=np.full_like(cross_east, np.nan), where=unique_null,
    )
    null_north = np.divide(
        cross_north, cross_norm,
        out=np.full_like(cross_north, np.nan), where=unique_null,
    )
    null_up = np.divide(
        cross_up, cross_norm,
        out=np.full_like(cross_up, np.nan), where=unique_null,
    )

    bca = _masked_float(symmetric_bca, horizontal_defined)
    final_bca_mask = horizontal_defined & (bca >= float(bca_threshold_deg))
    return GeometryDiagnostics(
        radar1_valid=valid1,
        radar2_valid=valid2,
        n_valid_radars=count,
        dual_valid=dual,
        symmetric_bca_deg=bca,
        horizontal_sigma_min=_masked_float(horizontal_sigma_min, horizontal_defined),
        horizontal_sigma_max=_masked_float(horizontal_sigma_max, horizontal_defined),
        horizontal_inverse_condition=_masked_float(inverse_condition, horizontal_defined),
        horizontal_condition_number=_masked_float(condition_number, horizontal_defined),
        beam_elevation_radar1_deg=_masked_float(el1, finite_geometry1 & (range1 > 0.0)),
        beam_elevation_radar2_deg=_masked_float(el2, finite_geometry2 & (range2 > 0.0)),
        radar_range_radar1_m=_masked_float(range1, finite_geometry1 & (range1 > 0.0)),
        radar_range_radar2_m=_masked_float(range2, finite_geometry2 & (range2 > 0.0)),
        direct_3d_sigma_min=_masked_float(direct_sigma_min, dual),
        direct_3d_sigma_max=_masked_float(direct_sigma_max, dual),
        null_direction_east=_masked_float(null_east, unique_null),
        null_direction_north=_masked_float(null_north, unique_null),
        null_direction_up=_masked_float(null_up, unique_null),
        null_vertical_alignment=_masked_float(np.abs(null_up), unique_null),
        final_bca_mask=final_bca_mask,
    )


def diagnose_cartesian_grid_geometry(
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: np.ndarray,
    radar1_enu_m: tuple[float, float, float],
    radar2_enu_m: tuple[float, float, float],
    radar1_valid: np.ndarray,
    radar2_valid: np.ndarray,
    *,
    bca_threshold_deg: float = 20.0,
) -> GeometryDiagnostics:
    """Build beam angles/ranges from an ENU grid and diagnose its geometry."""
    grid_z, grid_y, grid_x = np.meshgrid(z_m, y_m, x_m, indexing="ij")

    def radar_geometry(radar_enu_m):
        east = grid_x - float(radar_enu_m[0])
        north = grid_y - float(radar_enu_m[1])
        up = grid_z - float(radar_enu_m[2])
        distance = np.sqrt(east * east + north * north + up * up)
        azimuth = (np.rad2deg(np.arctan2(east, north)) + 360.0) % 360.0
        elevation = np.rad2deg(
            np.arctan2(up, np.hypot(east, north))
        )
        return azimuth, elevation, distance

    azimuth1, elevation1, range1 = radar_geometry(radar1_enu_m)
    azimuth2, elevation2, range2 = radar_geometry(radar2_enu_m)
    return diagnose_dual_radar_geometry(
        azimuth1, elevation1, range1, radar1_valid,
        azimuth2, elevation2, range2, radar2_valid,
        bca_threshold_deg=bca_threshold_deg,
    )


@dataclass(frozen=True)
class MaskProvenance:
    no_radar: np.ndarray
    radar1_only: np.ndarray
    radar2_only: np.ndarray
    dual_before_bca: np.ndarray
    rejected_by_bca: np.ndarray
    accepted_by_bca: np.ndarray
    rejected_downstream: np.ndarray
    final_valid: np.ndarray

    def counts(self) -> dict[str, int]:
        return {
            item.name: int(np.count_nonzero(getattr(self, item.name)))
            for item in fields(self)
        }


def classify_mask_provenance(
    radar1_valid: np.ndarray,
    radar2_valid: np.ndarray,
    bca_accepted: np.ndarray,
    *,
    downstream_valid: np.ndarray | None = None,
) -> MaskProvenance:
    """Partition grid cells by observable mask-failure stage."""
    valid1, valid2, accepted = np.broadcast_arrays(
        np.asarray(radar1_valid, dtype=bool),
        np.asarray(radar2_valid, dtype=bool),
        np.asarray(bca_accepted, dtype=bool),
    )
    if downstream_valid is None:
        downstream = np.ones(valid1.shape, dtype=bool)
    else:
        downstream = np.broadcast_to(
            np.asarray(downstream_valid, dtype=bool), valid1.shape
        )
    dual = valid1 & valid2
    accepted_by_bca = dual & accepted
    return MaskProvenance(
        no_radar=~valid1 & ~valid2,
        radar1_only=valid1 & ~valid2,
        radar2_only=~valid1 & valid2,
        dual_before_bca=dual,
        rejected_by_bca=dual & ~accepted,
        accepted_by_bca=accepted_by_bca,
        rejected_downstream=accepted_by_bca & ~downstream,
        final_valid=accepted_by_bca & downstream,
    )
