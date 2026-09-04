"""Build Step 4 time windows, estimate storm motion, and diagnose Tc.

This script reuses the frozen reader, QC, and Cartesian gridding settings only
to recover observation masks and gridded reflectivity for all six volumes.  It
does not call the dual-Doppler initialization or the PyDDA solver.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import re
import sys

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.geometry_diagnostics import diagnose_cartesian_grid_geometry
from src.gridder import wgs84_to_enu
from src.pipeline import RadarDataPipeline
from src.retrieval import WindRetrieval
from src.temporal_operator import (
    BilinearTranslationOperator,
    estimate_translation_ncc,
)


STATION1 = "Z9539"
STATION2 = "Z9532"
BCA_THRESHOLD_DEG = 20.0
COMPOSITE_Z_MIN_M = 2000.0
COMPOSITE_Z_MAX_M = 6000.0
REFLECTIVITY_FLOOR_DBZ = 20.0
MAX_DISPLACEMENT_M = 30_000.0
OUTPUT_DIR = ROOT / "artifacts" / "step4_temporal"
FILE_PATTERN = re.compile(r"wind3d_(\d{14}Z)_(\d{14}Z)\.nc$")


def parse_time(timestamp: str) -> datetime:
    return datetime.strptime(timestamp, "%Y%m%d%H%M%SZ").replace(tzinfo=timezone.utc)


def isoformat_z(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def radar_enu(grid, radar) -> tuple[float, float, float]:
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


def masked_float(values) -> np.ndarray:
    return np.asarray(np.ma.filled(np.ma.asarray(values), np.nan), dtype=np.float64)


def load_baselines() -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    files = sorted((ROOT / "output").glob("wind3d_20260511*.nc"))
    records = []
    reference_coordinates = None
    for path in files:
        match = FILE_PATTERN.match(path.name)
        if not match:
            continue
        timestamp1, timestamp2 = match.groups()
        with xr.open_dataset(path, decode_times=False) as dataset:
            x = np.asarray(dataset.x.values, dtype=np.float64)
            y = np.asarray(dataset.y.values, dtype=np.float64)
            z = np.asarray(dataset.z.values, dtype=np.float64)
            coordinates = (x, y, z)
            if reference_coordinates is None:
                reference_coordinates = coordinates
            else:
                for current, reference in zip(coordinates, reference_coordinates):
                    if not np.array_equal(current, reference):
                        raise RuntimeError("baseline wind files do not share an exact grid")
            wind = np.stack([
                np.asarray(dataset[name].values, dtype=np.float64).squeeze()
                for name in ("u", "v", "w")
            ])
            variable_names = set(dataset.data_vars)
        time1 = parse_time(timestamp1)
        time2 = parse_time(timestamp2)
        midpoint = time1 + (time2 - time1) / 2
        records.append({
            "path": path,
            "timestamp1": timestamp1,
            "timestamp2": timestamp2,
            "time1": time1,
            "time2": time2,
            "midpoint": midpoint,
            "pair_offset_seconds": abs((time1 - time2).total_seconds()),
            "wind": wind,
            "baseline_finite": np.all(np.isfinite(wind), axis=0),
            "saved_mask_variables": bool(
                {"radar_count", "dual_doppler_mask", "reflectivity"}
                <= variable_names
            ),
        })
    if len(records) != 6:
        raise RuntimeError(f"expected six current continuous baselines, found {len(records)}")
    records.sort(key=lambda item: item["midpoint"])
    assert reference_coordinates is not None
    return records, *reference_coordinates


def recover_observations(records: list[dict], x: np.ndarray, y: np.ndarray,
                         z: np.ndarray) -> None:
    pipeline = RadarDataPipeline(ROOT / "data")
    z_composite = (z >= COMPOSITE_Z_MIN_M) & (z <= COMPOSITE_Z_MAX_M)
    if not np.any(z_composite):
        raise RuntimeError("reflectivity composite height interval has no grid levels")
    for index, record in enumerate(records):
        print(
            f"[{index + 1}/{len(records)}] recover observations "
            f"{record['timestamp1']} / {record['timestamp2']}"
        )
        radars = []
        reflectivity_radars = []
        for station, timestamp in (
            (STATION1, record["timestamp1"]),
            (STATION2, record["timestamp2"]),
        ):
            velocity_sweeps = pipeline.build_volume(
                station, timestamp, "velocity", "high"
            )
            reflectivity_sweeps = pipeline.build_volume(
                station, timestamp, "reflectivity", "high"
            )
            if not velocity_sweeps or not reflectivity_sweeps:
                raise RuntimeError(f"missing volume for {station} {timestamp}")
            radars.append(pipeline.quality_control(
                pipeline.sweeps_to_pyart_radar(velocity_sweeps)
            ))
            reflectivity_radars.append(pipeline.quality_control(
                pipeline.sweeps_to_pyart_radar(reflectivity_sweeps)
            ))

        retrieval = WindRetrieval(
            radars[0], radars[1],
            reflectivity_radar1=reflectivity_radars[0],
            reflectivity_radar2=reflectivity_radars[1],
            vel_field=next(iter(radars[0].fields)),
            grid_shape=(z.size, y.size, x.size),
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
        geometry = diagnose_cartesian_grid_geometry(
            x, y, z,
            radar_enu(grid1, radars[0]),
            radar_enu(grid1, radars[1]),
            radar1_valid,
            radar2_valid,
            bca_threshold_deg=BCA_THRESHOLD_DEG,
        )
        reflectivity1 = masked_float(grid1.fields["reflectivity"]["data"])
        reflectivity2 = masked_float(grid2.fields["reflectivity"]["data"])
        with np.errstate(all="ignore"):
            reflectivity = np.nanmax(
                np.stack((reflectivity1, reflectivity2)), axis=0
            )
            composite = np.nanmax(reflectivity[z_composite], axis=0)
        record.update({
            "radar1_valid": radar1_valid,
            "radar2_valid": radar2_valid,
            # Omega has an explicit radar sample dimension.  Spatial any/dual
            # masks below are derived diagnostics, not replacements for it.
            "observation_samples": np.stack((radar1_valid, radar2_valid)),
            "observation_any": radar1_valid | radar2_valid,
            "observation_dual": radar1_valid & radar2_valid,
            "current_effective": geometry.final_bca_mask,
            "reflectivity_composite": composite,
        })
        del retrieval, grid1, grid2, radars, reflectivity_radars
        gc.collect()


def stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def estimate_motion(records: list[dict], x: np.ndarray, y: np.ndarray) -> list[dict]:
    motions = []
    for index in range(len(records) - 1):
        source = records[index]
        target = records[index + 1]
        estimate = estimate_translation_ncc(
            source["reflectivity_composite"],
            target["reflectivity_composite"],
            x, y,
            max_displacement_m=MAX_DISPLACEMENT_M,
            reflectivity_floor_dbz=REFLECTIVITY_FLOOR_DBZ,
        )
        interval_seconds = (target["midpoint"] - source["midpoint"]).total_seconds()
        estimate.update({
            "source_index": index,
            "target_index": index + 1,
            "source_midpoint_utc": isoformat_z(source["midpoint"]),
            "target_midpoint_utc": isoformat_z(target["midpoint"]),
            "interval_seconds": float(interval_seconds),
            "storm_motion_east_m_s": estimate["displacement_x_m"] / interval_seconds,
            "storm_motion_north_m_s": estimate["displacement_y_m"] / interval_seconds,
            "storm_speed_m_s": float(np.hypot(
                estimate["displacement_x_m"], estimate["displacement_y_m"]
            ) / interval_seconds),
        })
        motions.append(estimate)
    return motions


def window_rows(records: list[dict], mask_name: str) -> tuple[list[dict], list[dict]]:
    rows = []
    height_rows = []
    for length in (3, 4, 5):
        for start in range(len(records) - length + 1):
            window = records[start:start + length]
            masks = np.stack([record[mask_name] for record in window])
            union = np.any(masks, axis=0)
            intersection = np.all(masks, axis=0)
            row = {
                "mask": mask_name,
                "K": length,
                "start_index": start,
                "end_index": start + length - 1,
                "start_midpoint_utc": isoformat_z(window[0]["midpoint"]),
                "end_midpoint_utc": isoformat_z(window[-1]["midpoint"]),
                "per_time_counts": ";".join(
                    str(int(np.count_nonzero(mask))) for mask in masks
                ),
                "union_count": int(np.count_nonzero(union)),
                "intersection_count": int(np.count_nonzero(intersection)),
                "intersection_fraction_of_union": (
                    float(np.count_nonzero(intersection) / np.count_nonzero(union))
                    if np.any(union) else np.nan
                ),
                "intersection_fraction_of_domain": float(np.mean(intersection)),
            }
            rows.append(row)
            has_radar_dimension = union.ndim == 4
            for level, height_m in enumerate(records[0]["z"]):
                union_at_height = union[:, level] if has_radar_dimension else union[level]
                intersection_at_height = (
                    intersection[:, level] if has_radar_dimension else intersection[level]
                )
                height_rows.append({
                    "mask": mask_name,
                    "K": length,
                    "start_index": start,
                    "height_m": float(height_m),
                    "union_count": int(np.count_nonzero(union_at_height)),
                    "intersection_count": int(np.count_nonzero(intersection_at_height)),
                    "intersection_fraction_of_level": float(np.mean(intersection_at_height)),
                })
    return rows, height_rows


def norm_row(values: np.ndarray, mask: np.ndarray, prefix: str) -> dict:
    selected = values[:, mask]
    output = {}
    for component, name in enumerate(("u", "v", "w")):
        component_values = selected[component]
        output[f"{prefix}_{name}_l2"] = float(np.linalg.norm(component_values))
        output[f"{prefix}_{name}_rmse"] = float(np.sqrt(np.mean(component_values ** 2)))
    output[f"{prefix}_vector_l2"] = float(np.linalg.norm(selected))
    output[f"{prefix}_vector_rmse"] = float(
        np.sqrt(np.mean(np.sum(selected ** 2, axis=0)))
    )
    return output


def temporal_consistency(records: list[dict], motions: list[dict],
                         x: np.ndarray, y: np.ndarray) -> list[dict]:
    rows = []
    for index, motion in enumerate(motions):
        translation = BilinearTranslationOperator(
            x, y, motion["displacement_x_m"], motion["displacement_y_m"]
        )
        source_wind = records[index]["wind"]
        target_wind = records[index + 1]["wind"]
        source_mask = records[index]["current_effective"] & records[index]["baseline_finite"]
        target_mask = records[index + 1]["current_effective"] & records[index + 1]["baseline_finite"]
        shifted_source = translation.apply(np.where(source_mask[None], source_wind, 0.0))
        corrected_valid = translation.pair_valid_mask(source_mask, target_mask)
        fair_mask = corrected_valid & source_mask & target_mask
        if not np.any(fair_mask):
            raise RuntimeError(f"no fair common wind cells for pair {index}")
        uncorrected = target_wind - source_wind
        corrected = target_wind - shifted_source
        row = {
            "pair_index": index,
            "source_midpoint_utc": motion["source_midpoint_utc"],
            "target_midpoint_utc": motion["target_midpoint_utc"],
            "fair_common_count": int(np.count_nonzero(fair_mask)),
        }
        row.update(norm_row(uncorrected, fair_mask, "uncorrected"))
        row.update(norm_row(corrected, fair_mask, "corrected"))
        for name in ("u", "v", "w", "vector"):
            before = row[f"uncorrected_{name}_l2"]
            after = row[f"corrected_{name}_l2"]
            row[f"{name}_relative_reduction"] = (
                float(1.0 - after / before) if before > 0.0 else np.nan
            )
        rows.append(row)
    return rows


def singular_values(matrix: np.ndarray) -> tuple[list[float], float, float]:
    values = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
    normalized = values / values[0] if values[0] > 0.0 else values
    energy = values ** 2
    rank1_residual = float(np.sum(energy[1:]) / np.sum(energy)) if np.sum(energy) else 0.0
    second_ratio = float(normalized[1]) if normalized.size > 1 else 0.0
    return [float(item) for item in normalized], second_ratio, rank1_residual


def time_mode_diagnostics(records: list[dict], motions: list[dict],
                          x: np.ndarray, y: np.ndarray) -> list[dict]:
    rows = []
    for length in (3, 4, 5):
        for start in range(len(records) - length + 1):
            window = records[start:start + length]
            aligned_winds = [window[0]["wind"]]
            aligned_masks = [
                window[0]["current_effective"] & window[0]["baseline_finite"]
            ]
            cumulative_x = 0.0
            cumulative_y = 0.0
            for local_index in range(1, length):
                motion = motions[start + local_index - 1]
                cumulative_x += motion["displacement_x_m"]
                cumulative_y += motion["displacement_y_m"]
                align = BilinearTranslationOperator(x, y, -cumulative_x, -cumulative_y)
                raw_mask = (
                    window[local_index]["current_effective"]
                    & window[local_index]["baseline_finite"]
                )
                aligned_winds.append(align.apply(
                    np.where(raw_mask[None], window[local_index]["wind"], 0.0)
                ))
                aligned_masks.append(align.transported_valid_mask(raw_mask))
            raw_masks = [
                item["current_effective"] & item["baseline_finite"]
                for item in window
            ]
            fair_mask = np.logical_and.reduce(raw_masks + aligned_masks)
            count = int(np.count_nonzero(fair_mask))
            if count == 0:
                rows.append({
                    "K": length,
                    "start_index": start,
                    "common_count": 0,
                    "uncorrected_normalized_singular_values": "",
                    "corrected_normalized_singular_values": "",
                    "uncorrected_s2_over_s1": np.nan,
                    "corrected_s2_over_s1": np.nan,
                    "uncorrected_rank1_residual_energy": np.nan,
                    "corrected_rank1_residual_energy": np.nan,
                })
                continue
            raw_matrix = np.stack([
                item["wind"][:, fair_mask].reshape(-1) for item in window
            ])
            corrected_matrix = np.stack([
                field[:, fair_mask].reshape(-1) for field in aligned_winds
            ])
            raw_sv, raw_second, raw_residual = singular_values(raw_matrix)
            corrected_sv, corrected_second, corrected_residual = singular_values(corrected_matrix)
            rows.append({
                "K": length,
                "start_index": start,
                "common_count": count,
                "uncorrected_normalized_singular_values": ";".join(f"{v:.10g}" for v in raw_sv),
                "corrected_normalized_singular_values": ";".join(f"{v:.10g}" for v in corrected_sv),
                "uncorrected_s2_over_s1": raw_second,
                "corrected_s2_over_s1": corrected_second,
                "uncorrected_rank1_residual_energy": raw_residual,
                "corrected_rank1_residual_energy": corrected_residual,
            })
    return rows


def save_netcdf(records: list[dict], motions: list[dict], x: np.ndarray,
                y: np.ndarray, z: np.ndarray) -> None:
    midpoint_seconds = np.array([
        record["midpoint"].timestamp() for record in records
    ], dtype=np.float64)
    dataset = xr.Dataset(
        data_vars={
            "radar1_valid": (("time", "z", "y", "x"), np.stack([
                item["radar1_valid"] for item in records
            ]).astype(np.uint8)),
            "radar2_valid": (("time", "z", "y", "x"), np.stack([
                item["radar2_valid"] for item in records
            ]).astype(np.uint8)),
            "observation_any": (("time", "z", "y", "x"), np.stack([
                item["observation_any"] for item in records
            ]).astype(np.uint8)),
            "observation_dual": (("time", "z", "y", "x"), np.stack([
                item["observation_dual"] for item in records
            ]).astype(np.uint8)),
            "omega_effective_bca_ge_20": (("time", "z", "y", "x"), np.stack([
                item["current_effective"] for item in records
            ]).astype(np.uint8)),
            "baseline_finite_wind": (("time", "z", "y", "x"), np.stack([
                item["baseline_finite"] for item in records
            ]).astype(np.uint8)),
            "reflectivity_composite": (("time", "y", "x"), np.stack([
                item["reflectivity_composite"] for item in records
            ]).astype(np.float32)),
            "motion_dx": (("pair",), np.array([
                item["displacement_x_m"] for item in motions
            ], dtype=np.float64)),
            "motion_dy": (("pair",), np.array([
                item["displacement_y_m"] for item in motions
            ], dtype=np.float64)),
            "motion_peak_correlation": (("pair",), np.array([
                item["peak_correlation"] for item in motions
            ], dtype=np.float64)),
            "motion_overlap_fraction": (("pair",), np.array([
                item["storm_overlap_fraction"] for item in motions
            ], dtype=np.float64)),
            "motion_finite_valid_overlap_fraction": (("pair",), np.array([
                item["finite_valid_overlap_fraction"] for item in motions
            ], dtype=np.float64)),
        },
        coords={
            "time": midpoint_seconds,
            "pair": np.arange(len(motions)),
            "z": z,
            "y": y,
            "x": x,
        },
        attrs={
            "scope": "Step 4 diagnostics only; no PyDDA solve",
            "time_definition": "mean of two radar nominal volume timestamps",
            "within_volume_asynchrony": (
                "unknown: source products contain no per-ray timestamps; no synthetic ray times used"
            ),
            "reflectivity_composite_height_m": (
                f"{COMPOSITE_Z_MIN_M:g}-{COMPOSITE_Z_MAX_M:g}"
            ),
            "reflectivity_floor_dbz": REFLECTIVITY_FLOOR_DBZ,
            "motion_method": "explicit normalized cross-correlation with local parabolic refinement",
            "physics_domain": "not defined in Step 4",
        },
    )
    dataset.time.attrs["units"] = "seconds since 1970-01-01T00:00:00Z"
    for coordinate in ("x", "y", "z"):
        dataset[coordinate].attrs["units"] = "m"
    dataset.motion_dx.attrs["units"] = "m east"
    dataset.motion_dy.attrs["units"] = "m north"
    encoding = {
        name: {"zlib": True, "complevel": 4}
        for name in dataset.data_vars
        if dataset[name].ndim >= 2
    }
    dataset.to_netcdf(OUTPUT_DIR / "step4_masks_and_motion.nc", encoding=encoding)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records, x, y, z = load_baselines()
    for record in records:
        record["z"] = z
    recover_observations(records, x, y, z)
    motions = estimate_motion(records, x, y)

    window_summary_rows = []
    height_rows = []
    for mask_name in (
        "observation_samples", "observation_any", "observation_dual",
        "current_effective", "baseline_finite",
    ):
        rows, by_height = window_rows(records, mask_name)
        window_summary_rows.extend(rows)
        height_rows.extend(by_height)
    consistency_rows = temporal_consistency(records, motions, x, y)
    spectrum_rows = time_mode_diagnostics(records, motions, x, y)

    time_rows = []
    for index, record in enumerate(records):
        time_rows.append({
            "time_index": index,
            "radar1_nominal_utc": isoformat_z(record["time1"]),
            "radar2_nominal_utc": isoformat_z(record["time2"]),
            "pair_offset_seconds": record["pair_offset_seconds"],
            "volume_midpoint_utc": isoformat_z(record["midpoint"]),
            "interval_from_previous_midpoint_seconds": (
                (record["midpoint"] - records[index - 1]["midpoint"]).total_seconds()
                if index else None
            ),
            "radar1_observation_count": int(np.count_nonzero(record["radar1_valid"])),
            "radar2_observation_count": int(np.count_nonzero(record["radar2_valid"])),
            "observation_sample_count": int(np.count_nonzero(record["observation_samples"])),
            "observation_any_count": int(np.count_nonzero(record["observation_any"])),
            "observation_dual_count": int(np.count_nonzero(record["observation_dual"])),
            "current_effective_count": int(np.count_nonzero(record["current_effective"])),
            "baseline_finite_wind_count": int(np.count_nonzero(record["baseline_finite"])),
            "baseline_saved_mask_variables": record["saved_mask_variables"],
        })

    write_csv(OUTPUT_DIR / "volume_times_and_masks.csv", time_rows)
    write_csv(OUTPUT_DIR / "storm_motion.csv", motions)
    write_csv(OUTPUT_DIR / "window_mask_summary.csv", window_summary_rows)
    write_csv(OUTPUT_DIR / "window_common_by_height.csv", height_rows)
    write_csv(OUTPUT_DIR / "temporal_consistency.csv", consistency_rows)
    write_csv(OUTPUT_DIR / "time_mode_spectrum.csv", spectrum_rows)
    save_netcdf(records, motions, x, y, z)

    motion_speeds = np.array([item["storm_speed_m_s"] for item in motions])
    vector_reductions = np.array([
        item["vector_relative_reduction"] for item in consistency_rows
    ])
    spectrum_improvement = np.array([
        item["uncorrected_rank1_residual_energy"]
        - item["corrected_rank1_residual_energy"]
        for item in spectrum_rows
    ])
    observation_windows = [
        row for row in window_summary_rows if row["mask"] == "observation_samples"
    ]
    effective_windows = [
        row for row in window_summary_rows if row["mask"] == "current_effective"
    ]
    summary = {
        "scope": "Step 4 only; Step 5 rank selection/HOSVD not performed",
        "case": {
            "stations": [STATION1, STATION2],
            "n_volume_pairs": len(records),
            "grid_shape": [int(z.size), int(y.size), int(x.size)],
            "grid_spacing_m": [float(np.diff(z).mean()), float(np.diff(y).mean()), float(np.diff(x).mean())],
            "nominal_time_limitation": (
                "No per-ray sampling timestamps are present. Midpoints use only the two nominal "
                "volume timestamps; within-volume asynchrony cannot be corrected exactly and no "
                "synthetic ray timestamps were constructed."
            ),
        },
        "definitions": {
            "observation_samples": (
                "Omega_t: post-QC gridded radial-velocity samples with retained radar dimension; "
                "|Omega_t| counts radar-gridpoint samples"
            ),
            "observation_any": "post-QC gridded radial velocity from at least one radar",
            "observation_dual": "post-QC gridded radial velocity from both radars",
            "current_effective": (
                "dual observation and frozen symmetric BCA >= 20 degree criterion; this is not "
                "the full radar-indexed observation mask"
            ),
            "baseline_finite_wind": "finite u/v/w serialized in the existing baseline file",
            "physics_domain": "not defined in Step 4 and not inferred from any fill value",
            "computational_fill": (
                "zero is used only under a complete transported stencil mask during operator "
                "evaluation; invalid outputs remain excluded"
            ),
        },
        "motion_configuration": {
            "source": "maximum gridded observed reflectivity from both radars",
            "height_interval_m": [COMPOSITE_Z_MIN_M, COMPOSITE_Z_MAX_M],
            "reflectivity_floor_dbz": REFLECTIVITY_FLOOR_DBZ,
            "max_displacement_m": MAX_DISPLACEMENT_M,
            "method": "explicit normalized cross-correlation plus local parabolic sub-grid refinement",
            "does_not_use_retrieved_vortex_centres": True,
        },
        "volume_records": time_rows,
        "motion_records": motions,
        "motion_aggregate": {
            "speed_m_s": stats(motion_speeds),
            "east_m_s": stats(np.array([item["storm_motion_east_m_s"] for item in motions])),
            "north_m_s": stats(np.array([item["storm_motion_north_m_s"] for item in motions])),
            "confidence_counts": {
                level: sum(item["confidence"] == level for item in motions)
                for level in ("high", "medium", "low")
            },
        },
        "temporal_consistency": {
            "pair_records": consistency_rows,
            "vector_relative_reduction": stats(vector_reductions),
            "pairs_improved": int(np.count_nonzero(vector_reductions > 0.0)),
            "pair_count": len(consistency_rows),
        },
        "window_observation_samples": observation_windows,
        "window_current_effective": effective_windows,
        "time_mode_spectrum": {
            "diagnostic_only": True,
            "window_records": spectrum_rows,
            "rank1_residual_energy_reduction": stats(spectrum_improvement),
            "windows_with_stronger_decay": int(np.count_nonzero(spectrum_improvement > 0.0)),
            "window_count": len(spectrum_rows),
        },
        "baseline_mask_limitation": (
            "Only the first of six existing wind files stores radar/mask variables. The later five "
            "serialize finite wind values over the full grid, so finite-wind masks there are not "
            "observation masks. Observation masks were independently reconstructed from the frozen "
            "post-QC gridding path without running PyDDA."
        ),
    }
    with (OUTPUT_DIR / "step4_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
