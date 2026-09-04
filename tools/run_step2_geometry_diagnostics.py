"""Reproduce Step 2 coverage, mask-provenance, and geometry diagnostics.

The script stops before PyDDA optimization. It reuses the frozen baseline's
unchanged reader, QC, and gridding path, but computes geometry independently
from the Step 1 beam-vector convention.
"""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import xarray as xr
from scipy.ndimage import maximum_filter


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cinrad_products import load_mesocyclones
from src.geometry_diagnostics import (
    classify_mask_provenance,
    diagnose_cartesian_grid_geometry,
)
from src.gridder import wgs84_to_enu
from src.pipeline import RadarDataPipeline
from src.retrieval import WindRetrieval


STATION1 = "Z9539"
STATION2 = "Z9532"
TIMESTAMP1 = "20260511070939Z"
TIMESTAMP2 = "20260511070929Z"
BCA_THRESHOLD_DEG = 20.0
BASELINE_FILE = ROOT / "output" / f"wind3d_{TIMESTAMP1}_{TIMESTAMP2}.nc"
OUTPUT_DIR = ROOT / "artifacts" / "step2_geometry"
BCA_EDGES = np.array([0.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0])
RANGE_EDGES_KM = np.array([0.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0, 250.0, np.inf])


def _json_value(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _radar_gate_stats(radar, field_name: str) -> dict:
    data = np.ma.asarray(radar.fields[field_name]["data"])
    raw_values = np.ma.getdata(data).astype(np.float64)
    present = ~np.ma.getmaskarray(data) & np.isfinite(raw_values)
    return {
        "total_gates": int(data.size),
        "present_gates": int(np.count_nonzero(present)),
        "present_fraction": float(np.mean(present)),
        "range_qc_pass_gates": int(
            np.count_nonzero(present & (np.abs(raw_values) <= 100.0))
        ),
        "range_qc_fail_gates": int(
            np.count_nonzero(present & (np.abs(raw_values) > 100.0))
        ),
    }


def _sweep_rows(station: str, sweeps: list[dict]) -> list[dict]:
    rounded = [round(float(sweep["elevation"]), 2) for sweep in sweeps]
    counts = {elevation: rounded.count(elevation) for elevation in set(rounded)}
    rows = []
    for sweep, elevation in zip(sweeps, rounded):
        field_data = np.asarray(sweep["field_data"], dtype=np.float64)
        rows.append({
            "station": station,
            "scan_time_utc": sweep["scan_time"].isoformat(),
            "elevation_deg": elevation,
            "product_directory": sweep["elev_code"],
            "layer": sweep["layer"],
            "rays": int(field_data.shape[0]),
            "gates_per_ray": int(field_data.shape[1]),
            "valid_gate_fraction": float(np.mean(np.isfinite(field_data))),
            "same_elevation_sweep_count": int(counts[elevation]),
            "is_repeated_elevation": bool(counts[elevation] > 1),
        })
    return rows


def _radar_enu(grid, radar) -> tuple[float, float, float]:
    origin_lat = float(grid.origin_latitude["data"][0])
    origin_lon = float(grid.origin_longitude["data"][0])
    origin_alt_m = float(grid.origin_altitude["data"][0])
    east_km, north_km, up_km = wgs84_to_enu(
        np.array([float(radar.latitude["data"][0])]),
        np.array([float(radar.longitude["data"][0])]),
        np.array([float(radar.altitude["data"][0]) / 1000.0]),
        origin_lat,
        origin_lon,
        origin_alt_m / 1000.0,
    )
    return (
        float(east_km.item()) * 1000.0,
        float(north_km.item()) * 1000.0,
        float(up_km.item()) * 1000.0,
    )


def _finite_stat(values: np.ndarray, mask: np.ndarray, statistic: str) -> float:
    selected = np.asarray(values)[np.asarray(mask, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return float("nan")
    if statistic == "median":
        return float(np.median(selected))
    if statistic == "mean":
        return float(np.mean(selected))
    raise ValueError(statistic)


def _bca_rows_by_height(z_m, geometry) -> list[dict]:
    rows = []
    ny, nx = geometry.dual_valid.shape[1:]
    cells_per_level = ny * nx
    for level, height_m in enumerate(z_m):
        dual = geometry.dual_valid[level]
        bca = geometry.symmetric_bca_deg[level]
        dual_count = int(np.count_nonzero(dual))
        for lower, upper in zip(BCA_EDGES[:-1], BCA_EDGES[1:]):
            if upper == 90.0:
                in_bin = dual & (bca >= lower) & (bca <= upper)
            else:
                in_bin = dual & (bca >= lower) & (bca < upper)
            count = int(np.count_nonzero(in_bin))
            rows.append({
                "height_m": float(height_m),
                "bca_bin_deg": f"{lower:g}-{upper:g}",
                "gridpoint_count": count,
                "fraction_of_dual": count / dual_count if dual_count else np.nan,
                "fraction_of_level_domain": count / cells_per_level,
            })
    return rows


def _height_summary(z_m, geometry) -> list[dict]:
    rows = []
    cells_per_level = geometry.dual_valid.shape[1] * geometry.dual_valid.shape[2]
    for level, height_m in enumerate(z_m):
        dual = geometry.dual_valid[level]
        final = geometry.final_bca_mask[level]
        rows.append({
            "height_m": float(height_m),
            "radar1_fraction": float(np.mean(geometry.radar1_valid[level])),
            "radar2_fraction": float(np.mean(geometry.radar2_valid[level])),
            "dual_before_bca_fraction": float(np.mean(dual)),
            "bca_ge_20_fraction": float(np.mean(final)),
            "bca_rejected_fraction_of_dual": (
                1.0 - np.count_nonzero(final) / np.count_nonzero(dual)
                if np.count_nonzero(dual) else np.nan
            ),
            "dual_count": int(np.count_nonzero(dual)),
            "bca_ge_20_count": int(np.count_nonzero(final)),
            "level_cells": int(cells_per_level),
        })
    return rows


def _range_summary(geometry) -> list[dict]:
    rows = []
    maximum_range_km = np.maximum(
        geometry.radar_range_radar1_m,
        geometry.radar_range_radar2_m,
    ) / 1000.0
    for lower, upper in zip(RANGE_EDGES_KM[:-1], RANGE_EDGES_KM[1:]):
        in_range = (maximum_range_km >= lower) & (maximum_range_km < upper)
        dual = in_range & geometry.dual_valid
        final = in_range & geometry.final_bca_mask
        dual_count = int(np.count_nonzero(dual))
        rows.append({
            "max_radar_range_bin_km": f"{lower:g}-{upper:g}" if np.isfinite(upper) else f">={lower:g}",
            "domain_count": int(np.count_nonzero(in_range)),
            "dual_count": dual_count,
            "dual_fraction_in_range_bin": (
                dual_count / np.count_nonzero(in_range) if np.count_nonzero(in_range) else np.nan
            ),
            "bca_ge_20_count": int(np.count_nonzero(final)),
            "bca_ge_20_fraction_of_dual": (
                np.count_nonzero(final) / dual_count if dual_count else np.nan
            ),
            "median_bca_deg": _finite_stat(geometry.symmetric_bca_deg, dual, "median"),
            "median_horizontal_inverse_condition": _finite_stat(
                geometry.horizontal_inverse_condition, dual, "median"
            ),
            "median_null_vertical_alignment": _finite_stat(
                geometry.null_vertical_alignment, dual, "median"
            ),
        })
    return rows


def _nearest_polar_support(radar, field_name: str, azimuth_deg: float,
                           elevation_deg: float, range_m: float) -> dict:
    fixed_angles = np.asarray(radar.fixed_angle["data"], dtype=float)
    nearest_sweep = int(np.argmin(np.abs(fixed_angles - elevation_deg)))
    start = int(radar.sweep_start_ray_index["data"][nearest_sweep])
    end = int(radar.sweep_end_ray_index["data"][nearest_sweep]) + 1
    ray_azimuth = np.asarray(radar.azimuth["data"][start:end], dtype=float)
    gate_range = np.asarray(radar.range["data"], dtype=float)
    azimuth_delta = np.abs(
        (ray_azimuth - azimuth_deg + 180.0) % 360.0 - 180.0
    )
    ray_indices = np.flatnonzero(azimuth_delta <= 1.5)
    gate_indices = np.flatnonzero(np.abs(gate_range - range_m) <= 2000.0)
    field = np.ma.asarray(radar.fields[field_name]["data"])
    if ray_indices.size and gate_indices.size:
        local = field[np.ix_(start + ray_indices, gate_indices)]
        valid_count = int(np.ma.count(local))
        sample_count = int(local.size)
    else:
        valid_count = 0
        sample_count = 0
    return {
        "nearest_sweep_elevation_deg": float(fixed_angles[nearest_sweep]),
        "elevation_offset_deg": float(abs(fixed_angles[nearest_sweep] - elevation_deg)),
        "local_polar_samples": sample_count,
        "local_polar_valid_samples": valid_count,
        "within_recorded_range": bool(range_m <= float(gate_range.max()) + 2000.0),
    }


def _failure_reason(valid1: bool, valid2: bool, bca_ok: bool,
                    downstream_ok: bool, raw_support1: dict,
                    raw_support2: dict) -> str:
    if not valid1 and not valid2:
        outside = []
        if not raw_support1["within_recorded_range"]:
            outside.append("radar1")
        if not raw_support2["within_recorded_range"]:
            outside.append("radar2")
        if outside:
            return f"{'/'.join(outside)} target range exceeds recorded velocity range; neither radar is valid on the grid"
        if (raw_support1["local_polar_valid_samples"] == 0
                and raw_support2["local_polar_valid_samples"] == 0):
            return "neither radar has valid velocity in the sampled polar neighbourhood; no gridded observation"
        return "neither radar is valid after gridding; PyART did not preserve gate-to-grid contributor provenance"
    if valid1 and not valid2:
        if not raw_support2["within_recorded_range"]:
            return "radar2 target range exceeds recorded velocity range; gridded radar2 is invalid"
        if raw_support2["local_polar_valid_samples"] == 0:
            return "radar2 has no valid post-QC velocity in the sampled polar neighbourhood; gridded radar2 is invalid"
        return "radar2 has nearby valid polar data but no grid value; lost provenance prevents separating ROI/interpolation causes"
    if valid2 and not valid1:
        if not raw_support1["within_recorded_range"]:
            return "radar1 target range exceeds recorded velocity range; gridded radar1 is invalid"
        if raw_support1["local_polar_valid_samples"] == 0:
            return "radar1 has no valid post-QC velocity in the sampled polar neighbourhood; gridded radar1 is invalid"
        return "radar1 has nearby valid polar data but no grid value; lost provenance prevents separating ROI/interpolation causes"
    if not bca_ok:
        return "both radars are valid, but symmetric BCA<20 degrees is rejected by the current threshold"
    if not downstream_ok:
        return "coverage and BCA pass, but the current downstream mask rejects the point; finer provenance is unavailable"
    return "accepted by dual coverage, current BCA threshold, and downstream mask"


def _feature_rows(dataset, geometry, raw_radars, qc_radars,
                  radar_enu, velocity_field: str) -> list[dict]:
    x = np.asarray(dataset["x"].values, dtype=float)
    y = np.asarray(dataset["y"].values, dtype=float)
    z = np.asarray(dataset["z"].values, dtype=float)
    origin_lat = float(dataset["origin_latitude"].values.flat[0])
    origin_lon = float(dataset["origin_longitude"].values.flat[0])
    downstream = np.asarray(dataset["dual_doppler_mask"].values).squeeze().astype(bool)
    vorticity = np.asarray(dataset["vorticity"].values).squeeze()
    reflectivity = np.asarray(dataset["reflectivity"].values).squeeze()

    features = []
    for station, timestamp in ((STATION1, TIMESTAMP1), (STATION2, TIMESTAMP2)):
        for index, feature in enumerate(
            load_mesocyclones(station, timestamp, data_dir=str(ROOT / "data")), start=1
        ):
            east_km, north_km, _ = wgs84_to_enu(
                np.array([feature["lat"]]), np.array([feature["lon"]]), np.array([0.0]),
                origin_lat, origin_lon,
            )
            features.append({
                "feature_type": "business_M",
                "feature_id": f"{station}_M{index}",
                "source_station": station,
                "latitude_deg": feature["lat"],
                "longitude_deg": feature["lon"],
                "x_km": float(east_km.item()),
                "y_km": float(north_km.item()),
                "z_km": feature["height_m"] / 1000.0,
                "vorticity_s-1": np.nan,
                "reflectivity_dbz": np.nan,
            })

    eligible = downstream & np.isfinite(vorticity)
    local_maximum = vorticity == maximum_filter(
        np.where(eligible, vorticity, -np.inf),
        size=(3, 7, 7), mode="constant", cval=-np.inf,
    )
    candidate_indices = np.argwhere(eligible & local_maximum)
    candidate_indices = sorted(
        candidate_indices,
        key=lambda item: float(vorticity[tuple(item)]), reverse=True,
    )[:2]
    for rank, (iz, iy, ix) in enumerate(candidate_indices, start=1):
        east_km = x[ix] / 1000.0
        north_km = y[iy] / 1000.0
        # Small-domain inverse ENU approximation is sufficient for reporting.
        latitude = origin_lat + north_km / 111.0
        longitude = origin_lon + east_km / (
            111.0 * np.cos(np.deg2rad(latitude))
        )
        features.append({
            "feature_type": "retrieval_vortex_candidate",
            "feature_id": f"retrieval_C{rank}",
            "source_station": "Step0 baseline",
            "latitude_deg": latitude,
            "longitude_deg": longitude,
            "x_km": east_km,
            "y_km": north_km,
            "z_km": z[iz] / 1000.0,
            "vorticity_s-1": float(vorticity[iz, iy, ix]),
            "reflectivity_dbz": float(reflectivity[iz, iy, ix]),
        })

    rows = []
    for feature in features:
        ix = int(np.argmin(np.abs(x / 1000.0 - feature["x_km"])))
        iy = int(np.argmin(np.abs(y / 1000.0 - feature["y_km"])))
        iz = int(np.argmin(np.abs(z / 1000.0 - feature["z_km"])))
        point = (iz, iy, ix)
        valid1 = bool(geometry.radar1_valid[point])
        valid2 = bool(geometry.radar2_valid[point])
        bca_ok = bool(geometry.final_bca_mask[point])
        downstream_ok = bool(downstream[point])

        raw_support = []
        qc_support = []
        for radar_index in range(2):
            azimuth_deg = (
                np.rad2deg(np.arctan2(
                    x[ix] - radar_enu[radar_index][0],
                    y[iy] - radar_enu[radar_index][1],
                )) + 360.0
            ) % 360.0
            raw_support.append(_nearest_polar_support(
                raw_radars[radar_index], velocity_field,
                azimuth_deg,
                geometry.as_dict()[f"beam_elevation_radar{radar_index + 1}_deg"][point],
                geometry.as_dict()[f"radar_range_radar{radar_index + 1}_m"][point],
            ))
            qc_support.append(_nearest_polar_support(
                qc_radars[radar_index], velocity_field,
                azimuth_deg,
                geometry.as_dict()[f"beam_elevation_radar{radar_index + 1}_deg"][point],
                geometry.as_dict()[f"radar_range_radar{radar_index + 1}_m"][point],
            ))

        horizontal_distance = np.sqrt(
            (x[None, :] / 1000.0 - feature["x_km"]) ** 2
            + (y[:, None] / 1000.0 - feature["y_km"]) ** 2
        )
        horizontal_neighbourhood = horizontal_distance <= 5.0
        level_neighbourhood = np.zeros(len(z), dtype=bool)
        level_neighbourhood[max(0, iz - 1):min(len(z), iz + 2)] = True
        neighbourhood = level_neighbourhood[:, None, None] & horizontal_neighbourhood[None]
        neighbourhood_dual = neighbourhood & geometry.dual_valid

        row = dict(feature)
        row.update({
            "nearest_grid_x_km": float(x[ix] / 1000.0),
            "nearest_grid_y_km": float(y[iy] / 1000.0),
            "nearest_grid_z_km": float(z[iz] / 1000.0),
            "radar1_range_km": float(geometry.radar_range_radar1_m[point] / 1000.0),
            "radar2_range_km": float(geometry.radar_range_radar2_m[point] / 1000.0),
            "radar1_valid": valid1,
            "radar2_valid": valid2,
            "radar1_elevation_deg": float(geometry.beam_elevation_radar1_deg[point]),
            "radar2_elevation_deg": float(geometry.beam_elevation_radar2_deg[point]),
            "symmetric_bca_deg": float(geometry.symmetric_bca_deg[point]),
            "horizontal_sigma_min": float(geometry.horizontal_sigma_min[point]),
            "horizontal_sigma_max": float(geometry.horizontal_sigma_max[point]),
            "horizontal_inverse_condition": float(geometry.horizontal_inverse_condition[point]),
            "horizontal_condition_number": float(geometry.horizontal_condition_number[point]),
            "direct_3d_sigma_min": float(geometry.direct_3d_sigma_min[point]),
            "direct_3d_sigma_max": float(geometry.direct_3d_sigma_max[point]),
            "null_direction_east": float(geometry.null_direction_east[point]),
            "null_direction_north": float(geometry.null_direction_north[point]),
            "null_direction_up": float(geometry.null_direction_up[point]),
            "null_vertical_alignment": float(geometry.null_vertical_alignment[point]),
            "bca_ge_20": bca_ok,
            "current_pydda_effective_dual": downstream_ok,
            "neighbourhood_radar1_fraction": float(np.mean(geometry.radar1_valid[neighbourhood])),
            "neighbourhood_radar2_fraction": float(np.mean(geometry.radar2_valid[neighbourhood])),
            "neighbourhood_dual_fraction": float(np.mean(geometry.dual_valid[neighbourhood])),
            "neighbourhood_bca_ge_20_fraction": float(np.mean(geometry.final_bca_mask[neighbourhood])),
            "neighbourhood_median_bca_deg": _finite_stat(
                geometry.symmetric_bca_deg, neighbourhood_dual, "median"
            ),
            "radar1_raw_local_valid_samples": raw_support[0]["local_polar_valid_samples"],
            "radar2_raw_local_valid_samples": raw_support[1]["local_polar_valid_samples"],
            "radar1_qc_local_valid_samples": qc_support[0]["local_polar_valid_samples"],
            "radar2_qc_local_valid_samples": qc_support[1]["local_polar_valid_samples"],
            "radar1_within_recorded_velocity_range": qc_support[0]["within_recorded_range"],
            "radar2_within_recorded_velocity_range": qc_support[1]["within_recorded_range"],
            "radar1_nearest_sweep_elevation_deg": qc_support[0]["nearest_sweep_elevation_deg"],
            "radar2_nearest_sweep_elevation_deg": qc_support[1]["nearest_sweep_elevation_deg"],
            "radar1_nearest_sweep_offset_deg": qc_support[0]["elevation_offset_deg"],
            "radar2_nearest_sweep_offset_deg": qc_support[1]["elevation_offset_deg"],
            "failure_or_acceptance_reason": _failure_reason(
                valid1, valid2, bca_ok, downstream_ok,
                qc_support[0], qc_support[1],
            ),
        })
        rows.append({key: _json_value(value) for key, value in row.items()})
    return rows


def _write_netcdf(path: Path, x, y, z, geometry, attrs: dict) -> None:
    variables = {}
    for name, values in geometry.as_dict().items():
        array = np.asarray(values)
        if array.dtype == bool:
            array = array.astype(np.uint8)
        elif np.issubdtype(array.dtype, np.floating):
            array = array.astype(np.float32)
        variables[name] = (("z", "y", "x"), array)
    dataset = xr.Dataset(
        variables,
        coords={"z": ("z", z), "y": ("y", y), "x": ("x", x)},
        attrs=attrs,
    )
    for coordinate in ("x", "y", "z"):
        dataset[coordinate].attrs["units"] = "m"
    dataset["symmetric_bca_deg"].attrs["units"] = "degree"
    dataset["horizontal_condition_number"].attrs["note"] = (
        "Condition number of horizontal 2x2 A_h only; no 3-D H.T W H condition number is reported"
    )
    dataset["null_vertical_alignment"].attrs["definition"] = "abs(n dot e_z)"
    encoding = {
        name: {"zlib": True, "complevel": 4, "shuffle": True}
        for name in dataset.data_vars
    }
    dataset.to_netcdf(path, encoding=encoding)


def _make_figures(height_rows: list[dict], geometry, x, y, z) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    heights = np.array([row["height_m"] for row in height_rows]) / 1000.0
    fig, axis = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
    axis.plot([row["radar1_fraction"] * 100 for row in height_rows], heights, label=STATION1)
    axis.plot([row["radar2_fraction"] * 100 for row in height_rows], heights, label=STATION2)
    axis.plot([row["dual_before_bca_fraction"] * 100 for row in height_rows], heights, label="Dual before BCA")
    axis.plot([row["bca_ge_20_fraction"] * 100 for row in height_rows], heights, label="Dual and BCA>=20°")
    axis.set_xlabel("Fraction of horizontal grid (%)")
    axis.set_ylabel("Height (km)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    axis.set_title("Coverage and current geometry mask by height")
    fig.savefig(OUTPUT_DIR / "coverage_by_height.png", dpi=180)
    plt.close(fig)

    level = int(np.argmin(np.abs(np.asarray(z) - 4000.0)))
    classification = np.zeros(geometry.dual_valid.shape[1:], dtype=np.uint8)
    classification[geometry.radar1_valid[level] ^ geometry.radar2_valid[level]] = 1
    classification[geometry.dual_valid[level] & ~geometry.final_bca_mask[level]] = 2
    classification[geometry.final_bca_mask[level]] = 3
    cmap = ListedColormap(["#e6e8eb", "#d9a441", "#d9534f", "#2f7d6d"])
    fig, axis = plt.subplots(figsize=(6.6, 5.5), constrained_layout=True)
    image = axis.pcolormesh(x / 1000.0, y / 1000.0, classification, cmap=cmap,
                            vmin=-0.5, vmax=3.5, shading="nearest")
    colorbar = fig.colorbar(image, ax=axis, ticks=[0, 1, 2, 3])
    colorbar.ax.set_yticklabels(["No observation", "Single radar", "Dual / BCA<20°", "Dual / BCA>=20°"])
    axis.set_xlabel("East (km)")
    axis.set_ylabel("North (km)")
    axis.set_title(f"Observation classes at z={z[level]/1000.0:.2f} km")
    axis.set_aspect("equal")
    fig.savefig(OUTPUT_DIR / "coverage_classes_4km.png", dpi=180)
    plt.close(fig)


def main() -> None:
    if not BASELINE_FILE.is_file():
        raise FileNotFoundError(BASELINE_FILE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = RadarDataPipeline(ROOT / "data")

    station_inputs = []
    sweep_rows = []
    raw_gate_stats = {}
    qc_gate_stats = {}
    reflectivity_stats = {}
    for station, timestamp in ((STATION1, TIMESTAMP1), (STATION2, TIMESTAMP2)):
        velocity_sweeps = pipeline.build_volume(station, timestamp, "velocity", "high")
        reflectivity_sweeps = pipeline.build_volume(station, timestamp, "reflectivity", "high")
        sweep_rows.extend(_sweep_rows(station, velocity_sweeps))
        raw_velocity_radar = pipeline.sweeps_to_pyart_radar(velocity_sweeps)
        raw_reflectivity_radar = pipeline.sweeps_to_pyart_radar(reflectivity_sweeps)
        velocity_field = next(iter(raw_velocity_radar.fields))
        reflectivity_field = next(iter(raw_reflectivity_radar.fields))
        raw_gate_stats[station] = _radar_gate_stats(raw_velocity_radar, velocity_field)
        raw_velocity_copy = deepcopy(raw_velocity_radar)
        qc_velocity_radar = pipeline.quality_control(raw_velocity_radar)
        qc_reflectivity_radar = pipeline.quality_control(raw_reflectivity_radar)
        qc_gate_stats[station] = _radar_gate_stats(qc_velocity_radar, velocity_field)
        reflectivity_stats[station] = {
            "raw_present_gates": int(np.ma.count(
                pipeline.sweeps_to_pyart_radar(reflectivity_sweeps).fields[reflectivity_field]["data"]
            )),
            "qc_present_gates": int(np.ma.count(qc_reflectivity_radar.fields[reflectivity_field]["data"])),
        }
        station_inputs.append((
            raw_velocity_copy, qc_velocity_radar, qc_reflectivity_radar,
        ))

    with xr.open_dataset(BASELINE_FILE, decode_times=False) as baseline:
        x = np.asarray(baseline["x"].values, dtype=float)
        y = np.asarray(baseline["y"].values, dtype=float)
        z = np.asarray(baseline["z"].values, dtype=float)
        baseline_loaded = baseline.load()

    retrieval = WindRetrieval(
        station_inputs[0][1], station_inputs[1][1],
        reflectivity_radar1=station_inputs[0][2],
        reflectivity_radar2=station_inputs[1][2],
        vel_field=next(iter(station_inputs[0][1].fields)),
        grid_shape=(len(z), len(y), len(x)),
        grid_limits=((float(z[0]), float(z[-1])),
                     (float(y[0]), float(y[-1])),
                     (float(x[0]), float(x[-1]))),
        station1=STATION1,
        station2=STATION2,
        data_dir=str(ROOT / "data"),
    )
    grid1, grid2 = retrieval.grid_radars()
    velocity_field = retrieval.vel_field
    radar1_valid = ~np.ma.getmaskarray(grid1.fields[velocity_field]["data"])
    radar2_valid = ~np.ma.getmaskarray(grid2.fields[velocity_field]["data"])
    radar_enu = (
        _radar_enu(grid1, station_inputs[0][1]),
        _radar_enu(grid1, station_inputs[1][1]),
    )
    geometry = diagnose_cartesian_grid_geometry(
        x, y, z,
        radar_enu[0], radar_enu[1],
        radar1_valid, radar2_valid,
        bca_threshold_deg=BCA_THRESHOLD_DEG,
    )
    baseline_final = np.asarray(
        baseline_loaded["dual_doppler_mask"].values
    ).squeeze().astype(bool)
    provenance = classify_mask_provenance(
        radar1_valid, radar2_valid, geometry.final_bca_mask,
        downstream_valid=baseline_final,
    )

    total = geometry.dual_valid.size
    counts = provenance.counts()
    summary = {
        "case": {
            "station1": STATION1,
            "station2": STATION2,
            "timestamp1": TIMESTAMP1,
            "timestamp2": TIMESTAMP2,
            "baseline_file": str(BASELINE_FILE.relative_to(ROOT)),
            "grid_shape": list(geometry.dual_valid.shape),
            "total_gridpoints": total,
            "bca_threshold_deg": BCA_THRESHOLD_DEG,
        },
        "polar_gate_provenance": {
            "raw_velocity": raw_gate_stats,
            "after_unchanged_qc": qc_gate_stats,
            "reflectivity": reflectivity_stats,
            "limitation": (
                "PyART grid output does not retain source-gate contributor IDs. "
                "Raw/QC gate failures can be counted globally and sampled locally, "
                "but a missing grid cell cannot always be uniquely assigned to one gate-level rule."
            ),
        },
        "grid_coverage": {
            "radar1_count": int(np.count_nonzero(radar1_valid)),
            "radar1_fraction": float(np.mean(radar1_valid)),
            "radar2_count": int(np.count_nonzero(radar2_valid)),
            "radar2_fraction": float(np.mean(radar2_valid)),
            "dual_before_bca_count": counts["dual_before_bca"],
            "dual_before_bca_fraction": counts["dual_before_bca"] / total,
            "rejected_by_bca_count": counts["rejected_by_bca"],
            "rejected_by_bca_fraction_of_dual": (
                counts["rejected_by_bca"] / counts["dual_before_bca"]
                if counts["dual_before_bca"] else np.nan
            ),
            "bca_ge_20_count": counts["accepted_by_bca"],
            "bca_ge_20_fraction": counts["accepted_by_bca"] / total,
            "baseline_final_count": int(np.count_nonzero(baseline_final)),
            "baseline_final_fraction": float(np.mean(baseline_final)),
            "rejected_downstream_count": counts["rejected_downstream"],
            "diagnostic_vs_baseline_disagreement_count": int(np.count_nonzero(
                geometry.final_bca_mask ^ baseline_final
            )),
        },
        "observation_classes": {
            "dual_good_geometry_bca_ge_20": counts["accepted_by_bca"],
            "dual_poor_geometry_bca_lt_20": counts["rejected_by_bca"],
            "single_radar": counts["radar1_only"] + counts["radar2_only"],
            "no_observation": counts["no_radar"],
        },
        "mask_provenance_counts": counts,
        "repeated_sweeps": {
            station: {
                "repeated_elevation_groups": len({
                    row["elevation_deg"] for row in sweep_rows
                    if row["station"] == station and row["is_repeated_elevation"]
                }),
                "repeated_sweep_records": sum(
                    row["station"] == station and row["is_repeated_elevation"]
                    for row in sweep_rows
                ),
            }
            for station in (STATION1, STATION2)
        },
    }

    height_rows = _height_summary(z, geometry)
    bca_rows = _bca_rows_by_height(z, geometry)
    range_rows = _range_summary(geometry)
    feature_rows = _feature_rows(
        baseline_loaded, geometry,
        (station_inputs[0][0], station_inputs[1][0]),
        (station_inputs[0][1], station_inputs[1][1]),
        radar_enu,
        velocity_field,
    )

    _write_netcdf(
        OUTPUT_DIR / f"geometry_diagnostics_{TIMESTAMP1}_{TIMESTAMP2}.nc",
        x, y, z, geometry,
        attrs={
            "diagnostic_scope": "Step 2 coverage, mask provenance, and direct observability only",
            "station1": STATION1,
            "station2": STATION2,
            "timestamp1": TIMESTAMP1,
            "timestamp2": TIMESTAMP2,
            "bca_threshold_deg": BCA_THRESHOLD_DEG,
            "three_dimensional_rank_note": (
                "The 2x3 two-radar H has rank at most 2; no ordinary 3-D condition number is reported"
            ),
        },
    )
    _write_csv(OUTPUT_DIR / "coverage_by_height.csv", height_rows)
    _write_csv(OUTPUT_DIR / "bca_bins_by_height.csv", bca_rows)
    _write_csv(OUTPUT_DIR / "range_geometry_summary.csv", range_rows)
    _write_csv(OUTPUT_DIR / "feature_neighbourhood_diagnostics.csv", feature_rows)
    _write_csv(OUTPUT_DIR / "repeated_sweep_diagnostics.csv", sweep_rows)
    with (OUTPUT_DIR / "step2_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, default=_json_value)
    _make_figures(height_rows, geometry, x, y, z)

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_value))
    print("\nFEATURES")
    for row in feature_rows:
        print(json.dumps(row, ensure_ascii=False, default=_json_value))


if __name__ == "__main__":
    main()
