"""Run the frozen Step 5 Tucker low-rank feasibility experiment.

No retrieval is run or changed here.  The script consumes only the six
uniform Step 5 baseline files and the frozen Step 4 motion estimates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.low_rank_diagnostics import (  # noqa: E402
    largest_true_cuboid,
    longest_true_run,
    masked_st_hosvd,
    mode_spectrum,
    relative_error,
    st_hosvd,
    tucker_parameter_count,
    vertical_vorticity_where_valid,
)
from src.mass_continuity import AnelasticMassContinuityOperator, exponential_density  # noqa: E402
from src.temporal_operator import BilinearTranslationOperator  # noqa: E402


BASELINE_DIR = ROOT / "artifacts" / "step5_low_rank" / "uniform_baseline"
STEP4_SUMMARY = ROOT / "artifacts" / "step4_temporal" / "step4_summary.json"
OUTPUT_DIR = ROOT / "artifacts" / "step5_low_rank" / "diagnostics"
MODES = ("x", "y", "z", "time", "component")
COMPONENTS = ("u", "v", "w")
WINDOW_LENGTHS = (3, 4, 5)
MISSING_COMPLETION_ITERATIONS = 6


def _commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rank_grid(shape: tuple[int, ...]) -> list[tuple[str, tuple[int, ...]]]:
    """Predeclared rank rules; they are not adapted after inspecting errors."""
    nx, ny, nz, nt, nc = shape

    def fraction(size: int, value: float, minimum: int = 2) -> int:
        return min(size, max(1 if size == 1 else minimum, int(math.ceil(value * size))))

    candidates = [
        ("aggressive_temporal", (fraction(nx, .125), fraction(ny, .125), fraction(nz, .25), 1, nc)),
        ("compact_temporal", (fraction(nx, .25), fraction(ny, .25), fraction(nz, .50), min(2, nt), nc)),
        ("moderate_temporal", (fraction(nx, .50), fraction(ny, .50), fraction(nz, .75), min(3, nt), nc)),
        ("compact_rt_full", (fraction(nx, .25), fraction(ny, .25), fraction(nz, .50), nt, nc)),
        ("moderate_rt_full", (fraction(nx, .50), fraction(ny, .50), fraction(nz, .75), nt, nc)),
    ]
    return candidates


def _slices(selection: tuple[slice, slice, slice]) -> dict[str, list[int]]:
    z, y, x = selection
    return {"z": [z.start, z.stop], "y": [y.start, y.stop], "x": [x.start, x.stop]}


def _bbox(mask_zyx: np.ndarray) -> tuple[slice, slice, slice]:
    indices = np.argwhere(mask_zyx)
    if indices.size == 0:
        raise ValueError("empty mask has no bounding box")
    low = indices.min(axis=0)
    high = indices.max(axis=0) + 1
    return tuple(slice(int(a), int(b)) for a, b in zip(low, high))  # type: ignore[return-value]


def _tensor(fields_tzyxc: np.ndarray) -> np.ndarray:
    return np.transpose(fields_tzyxc, (3, 2, 1, 0, 4))


def _series(tensor_xyztc: np.ndarray) -> np.ndarray:
    return np.transpose(tensor_xyztc, (3, 2, 1, 0, 4))


def _load() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict], dict]:
    manifest = json.loads((BASELINE_DIR / "manifest.json").read_text(encoding="utf-8"))
    completed = manifest["completed"]
    if len(completed) != 6:
        raise RuntimeError(f"expected six uniform baselines, found {len(completed)}")
    fields, physics, finite = [], [], []
    coordinates = None
    required = {
        "observation_mask_radar1", "observation_mask_radar2",
        "dual_radar_observation_mask", "bca_ge_20_mask",
        "baseline_finite_wind_mask", "physics_mask", "u", "v", "w",
    }
    for record in completed:
        path = ROOT / record["file"]
        with xr.open_dataset(path) as dataset:
            missing = required - set(dataset.variables)
            if missing:
                raise RuntimeError(f"{path.name} missing variables {sorted(missing)}")
            if dataset.attrs.get("uniform_baseline_schema") != "step5-v1":
                raise RuntimeError(f"{path.name} is not a uniform Step 5 baseline")
            if dataset.attrs.get("retrieval_configuration_json") != json.dumps(
                manifest["configuration"], sort_keys=True
            ):
                raise RuntimeError(f"{path.name} configuration differs from manifest")
            item = np.stack([
                np.asarray(dataset[name].isel(time=0).values, dtype=np.float64)
                for name in COMPONENTS
            ], axis=-1)
            fields.append(item)
            physics.append(np.asarray(dataset["physics_mask"].isel(time=0).values, dtype=bool))
            finite.append(np.asarray(dataset["baseline_finite_wind_mask"].isel(time=0).values, dtype=bool))
            xyz = tuple(np.asarray(dataset[name].values, dtype=np.float64) for name in ("x", "y", "z"))
            if coordinates is None:
                coordinates = xyz
            elif any(not np.array_equal(a, b) for a, b in zip(coordinates, xyz)):
                raise RuntimeError("uniform baseline coordinates differ")
    assert coordinates is not None
    motion = json.loads(STEP4_SUMMARY.read_text(encoding="utf-8"))
    return np.stack(fields), np.stack(physics), np.stack(finite), coordinates, completed, motion


def _storm_relative(
    fields: np.ndarray,
    masks: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    motion_records: list[dict],
    start: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Transport every window member backward to the first member's frame."""
    shifted_fields = []
    shifted_masks = []
    cumulative_x = cumulative_y = 0.0
    for relative in range(fields.shape[0]):
        if relative:
            record = motion_records[start + relative - 1]
            cumulative_x += float(record["displacement_x_m"])
            cumulative_y += float(record["displacement_y_m"])
        if relative == 0:
            shifted_fields.append(np.where(masks[relative, ..., None], fields[relative], np.nan))
            shifted_masks.append(masks[relative].copy())
            continue
        operator = BilinearTranslationOperator(x, y, -cumulative_x, -cumulative_y)
        source_mask = masks[relative]
        transported_mask = operator.transported_valid_mask(source_mask)
        computational = np.where(source_mask[..., None], fields[relative], 0.0)
        transported = np.stack(
            [operator.apply(computational[..., component]) for component in range(3)],
            axis=-1,
        )
        shifted_fields.append(np.where(transported_mask[..., None], transported, np.nan))
        shifted_masks.append(transported_mask)
    return np.stack(shifted_fields), np.stack(shifted_masks)


def _safe_ratio(value: float, reference: float) -> float:
    return float(value / reference) if np.isfinite(reference) and reference != 0.0 else float("nan")


def _vortex_metrics(
    reference_tzyxc: np.ndarray,
    estimate_tzyxc: np.ndarray,
    valid_tzyx: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> dict[str, float]:
    if min(len(x), len(y), len(z)) < 2 or not np.any(valid_tzyx):
        return {name: float("nan") for name in (
            "reference_zeta_max_s-1", "estimate_zeta_max_s-1", "zeta_preservation",
            "reference_circulation_m2_s-1", "estimate_circulation_m2_s-1",
            "circulation_preservation", "reference_cyclonic_circulation_m2_s-1",
            "estimate_cyclonic_circulation_m2_s-1", "cyclonic_circulation_preservation",
            "circulation_region_coverage", "centre_shift_m", "reference_column_depth_m",
            "estimate_column_depth_m", "column_depth_preservation",
        )}
    reference_zeta = []
    estimate_zeta = []
    for time in range(reference_tzyxc.shape[0]):
        ref, _ = vertical_vorticity_where_valid(
            reference_tzyxc[time, ..., 0], reference_tzyxc[time, ..., 1], x, y,
            valid_tzyx[time],
        )
        est, _ = vertical_vorticity_where_valid(
            estimate_tzyxc[time, ..., 0], estimate_tzyxc[time, ..., 1], x, y,
            valid_tzyx[time],
        )
        reference_zeta.append(ref)
        estimate_zeta.append(est)
    ref_zeta = np.stack(reference_zeta)
    est_zeta = np.stack(estimate_zeta)
    if not np.any(np.isfinite(ref_zeta)):
        return _vortex_metrics(reference_tzyxc, estimate_tzyxc, np.zeros_like(valid_tzyx), x, y, z)
    ref_index = np.unravel_index(int(np.nanargmax(ref_zeta)), ref_zeta.shape)
    time0, z0, y0, x0 = ref_index
    ref_peak = float(ref_zeta[ref_index])
    xx, yy = np.meshgrid(x, y, indexing="xy")
    horizontal_distance = np.hypot(xx - x[x0], yy - y[y0])
    local = (
        valid_tzyx
        & (horizontal_distance[None, None, ...] <= 30_000.0)
        & (np.abs(z[None, :, None, None] - z[z0]) <= 3_000.0)
        & (np.arange(reference_tzyxc.shape[0])[:, None, None, None] == time0)
    )
    local_est = np.where(local, est_zeta, np.nan)
    est_index = (
        np.unravel_index(int(np.nanargmax(local_est)), local_est.shape)
        if np.any(np.isfinite(local_est)) else ref_index
    )
    est_peak = float(est_zeta[est_index])
    centre_shift = float(np.sqrt(
        (x[est_index[3]] - x[x0]) ** 2
        + (y[est_index[2]] - y[y0]) ** 2
        + (z[est_index[1]] - z[z0]) ** 2
    ))
    circulation_region = valid_tzyx[time0, z0] & (horizontal_distance <= 15_000.0)
    dx = float(np.mean(np.diff(x)))
    dy = float(np.mean(np.diff(y)))
    ref_circulation = float(np.nansum(ref_zeta[time0, z0][circulation_region]) * dx * dy)
    est_circulation = float(np.nansum(est_zeta[time0, z0][circulation_region]) * dx * dy)
    ref_cyclonic = float(np.nansum(np.maximum(ref_zeta[time0, z0][circulation_region], 0.0)) * dx * dy)
    est_cyclonic = float(np.nansum(np.maximum(est_zeta[time0, z0][circulation_region], 0.0)) * dx * dy)
    circle_count = int(np.count_nonzero(horizontal_distance <= 15_000.0))
    region_coverage = float(np.count_nonzero(circulation_region) / circle_count) if circle_count else float("nan")

    def column_depth(zeta: np.ndarray) -> float:
        active = []
        for level in range(z.size):
            region = valid_tzyx[time0, level] & (horizontal_distance <= 15_000.0)
            peak = float(np.nanmax(zeta[time0, level][region])) if np.any(region) else -np.inf
            active.append(peak >= 0.25 * ref_peak)
        levels = longest_true_run(active)
        dz = float(np.mean(np.diff(z)))
        return float(levels * dz)

    ref_depth = column_depth(ref_zeta)
    est_depth = column_depth(est_zeta)
    return {
        "reference_zeta_max_s-1": ref_peak,
        "estimate_zeta_max_s-1": est_peak,
        "zeta_preservation": _safe_ratio(est_peak, ref_peak),
        "reference_circulation_m2_s-1": ref_circulation,
        "estimate_circulation_m2_s-1": est_circulation,
        "circulation_preservation": _safe_ratio(est_circulation, ref_circulation),
        "reference_cyclonic_circulation_m2_s-1": ref_cyclonic,
        "estimate_cyclonic_circulation_m2_s-1": est_cyclonic,
        "cyclonic_circulation_preservation": _safe_ratio(est_cyclonic, ref_cyclonic),
        "circulation_region_coverage": region_coverage,
        "centre_shift_m": centre_shift,
        "reference_column_depth_m": ref_depth,
        "estimate_column_depth_m": est_depth,
        "column_depth_preservation": _safe_ratio(est_depth, ref_depth),
    }


def _metrics(
    reference: np.ndarray,
    estimate: np.ndarray,
    valid_tzyx: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> dict[str, float]:
    valid_components = valid_tzyx[..., None]
    result = {"relative_frobenius_error": relative_error(reference, estimate, valid_components)}
    for index, name in enumerate(COMPONENTS):
        valid = valid_tzyx & np.isfinite(reference[..., index]) & np.isfinite(estimate[..., index])
        result[f"rmse_{name}_m_s"] = float(np.sqrt(np.mean(
            (estimate[..., index][valid] - reference[..., index][valid]) ** 2
        ))) if np.any(valid) else float("nan")
        rms = float(np.sqrt(np.mean(reference[..., index][valid] ** 2))) if np.any(valid) else float("nan")
        result[f"nrmse_{name}"] = _safe_ratio(result[f"rmse_{name}_m_s"], rms)
    speed_ref = np.linalg.norm(reference, axis=-1)
    speed_est = np.linalg.norm(estimate, axis=-1)
    result["reference_max_speed_m_s"] = float(np.nanmax(np.where(valid_tzyx, speed_ref, np.nan)))
    result["estimate_max_speed_m_s"] = float(np.nanmax(np.where(valid_tzyx, speed_est, np.nan)))
    result["max_speed_preservation"] = _safe_ratio(
        result["estimate_max_speed_m_s"], result["reference_max_speed_m_s"]
    )
    result.update(_vortex_metrics(reference, estimate, valid_tzyx, x, y, z))
    return result


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(name for name in row if name not in fieldnames)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields, physics, finite, coordinates, baseline_records, step4 = _load()
    x, y, z = coordinates
    spectrum_rows: list[dict] = []
    rank_rows: list[dict] = []
    height_rows: list[dict] = []
    domains: list[dict] = []
    for length in WINDOW_LENGTHS:
        for start in range(7 - length):
            stop = start + length
            window_id = f"K{length}_T{start + 1}-{stop}"
            raw_fields = fields[start:stop]
            raw_masks = physics[start:stop]
            storm_fields, storm_masks = _storm_relative(
                raw_fields, raw_masks, x, y, step4["motion_records"], start
            )
            fair_mask = np.all(raw_masks, axis=0) & np.all(storm_masks, axis=0)
            selection = largest_true_cuboid(fair_mask)
            domains.append({
                "window": window_id,
                "K": length,
                "start_index": start,
                "fair_common_count": int(np.count_nonzero(fair_mask)),
                "dense_box": _slices(selection),
                "dense_box_shape_zyx": list(fair_mask[selection].shape),
                "dense_box_samples_per_time": int(fair_mask[selection].size),
                "x_range_m": [float(x[selection[2].start]), float(x[selection[2].stop - 1])],
                "y_range_m": [float(y[selection[1].start]), float(y[selection[1].stop - 1])],
                "z_range_m": [float(z[selection[0].start]), float(z[selection[0].stop - 1])],
            })
            representations = {
                "eulerian": (raw_fields, raw_masks),
                "storm_relative": (storm_fields, storm_masks),
            }
            for representation, (rep_fields, rep_masks) in representations.items():
                dense_series = rep_fields[(slice(None),) + selection + (slice(None),)]
                dense_tensor = _tensor(dense_series)
                if not np.all(np.isfinite(dense_tensor)):
                    raise RuntimeError(f"{window_id} {representation} dense box is not complete")
                for mode, mode_name in enumerate(MODES):
                    spectrum = mode_spectrum(dense_tensor, mode)
                    for index, (singular, cumulative) in enumerate(zip(
                        spectrum["singular_values"], spectrum["cumulative_energy"]
                    )):
                        spectrum_rows.append({
                            "window": window_id, "K": length, "representation": representation,
                            "domain": "A_common_dense", "mode": mode_name,
                            "index": index + 1, "singular_value": float(singular),
                            "normalized_singular_value": float(singular / spectrum["singular_values"][0])
                            if spectrum["singular_values"][0] else 0.0,
                            "cumulative_energy": float(cumulative),
                            "effective_rank": float(spectrum["effective_rank"]),
                            "rank_90": int(spectrum["rank_90"]),
                            "rank_95": int(spectrum["rank_95"]),
                            "rank_99": int(spectrum["rank_99"]),
                        })
                dense_x = x[selection[2]]
                dense_y = y[selection[1]]
                dense_z = z[selection[0]]
                dense_valid = np.ones(dense_series.shape[:-1], dtype=bool)
                for rank_name, ranks in _rank_grid(dense_tensor.shape):
                    approximation = st_hosvd(dense_tensor, ranks)
                    estimate = _series(approximation.reconstruction)
                    metrics = _metrics(dense_series, estimate, dense_valid, dense_x, dense_y, dense_z)
                    parameters = tucker_parameter_count(dense_tensor.shape, ranks)
                    row = {
                        "window": window_id, "K": length, "representation": representation,
                        "domain": "A_common_dense", "rank_name": rank_name,
                        "ranks": json.dumps(ranks), "shape": json.dumps(dense_tensor.shape),
                        "raw_dof": int(dense_tensor.size), "tucker_parameters": parameters,
                        "parameter_fraction": float(parameters / dense_tensor.size),
                        "compression_factor": float(dense_tensor.size / parameters),
                        "completion_iterations": 0,
                    }
                    row.update(metrics)
                    rank_rows.append(row)
                    for level, z_value in enumerate(dense_z):
                        for component, name in enumerate(COMPONENTS):
                            difference = estimate[:, level, ..., component] - dense_series[:, level, ..., component]
                            height_rows.append({
                                "window": window_id, "representation": representation,
                                "domain": "A_common_dense", "rank_name": rank_name,
                                "z_m": float(z_value), "component": name,
                                "rmse_m_s": float(np.sqrt(np.mean(difference * difference))),
                            })

                # B: missing-aware representation over the union bounding box.
                union = np.any(rep_masks, axis=0)
                box = _bbox(union)
                missing_series = rep_fields[(slice(None),) + box + (slice(None),)]
                missing_mask = rep_masks[(slice(None),) + box]
                missing_tensor = _tensor(missing_series)
                tensor_mask = _tensor(missing_mask[..., None])
                grid_indices = np.indices(missing_tensor.shape[:-1])
                holdout_pattern = (
                    grid_indices[0] + 2 * grid_indices[1] + 3 * grid_indices[2]
                    + 5 * grid_indices[3]
                ) % 10 == 0
                holdout_mask = tensor_mask & holdout_pattern[..., None]
                fit_mask = tensor_mask & ~holdout_mask
                bx, by, bz = x[box[2]], y[box[1]], z[box[0]]
                local_operator = AnelasticMassContinuityOperator(
                    bx, by, bz, exponential_density
                )
                diagnostic_masks = []
                for time in range(length):
                    components = tuple(missing_series[time, ..., index] for index in range(3))
                    diagnostic_masks.append(local_operator.physical_valid_mask(
                        tuple(np.where(np.isfinite(item), item, np.nan) for item in components),
                        base_mask=missing_mask[time],
                    ))
                diagnostic_masks = np.stack(diagnostic_masks)
                for rank_name, ranks in _rank_grid(missing_tensor.shape):
                    approximation = masked_st_hosvd(
                        missing_tensor, fit_mask, ranks,
                        iterations=MISSING_COMPLETION_ITERATIONS,
                    )
                    estimate = _series(approximation.reconstruction)
                    metrics = _metrics(
                        missing_series, estimate, diagnostic_masks, bx, by, bz
                    )
                    parameters = tucker_parameter_count(missing_tensor.shape, ranks)
                    row = {
                        "window": window_id, "K": length, "representation": representation,
                        "domain": "B_structured_missing", "rank_name": rank_name,
                        "ranks": json.dumps(ranks), "shape": json.dumps(missing_tensor.shape),
                        "raw_dof": int(missing_tensor.size), "tucker_parameters": parameters,
                        "parameter_fraction": float(parameters / missing_tensor.size),
                        "compression_factor": float(missing_tensor.size / parameters),
                        "completion_iterations": MISSING_COMPLETION_ITERATIONS,
                        "observed_fraction": float(np.count_nonzero(tensor_mask) / tensor_mask.size),
                        "observed_scalar_count": int(np.count_nonzero(tensor_mask) * 3),
                        "fit_scalar_count": int(np.count_nonzero(fit_mask) * 3),
                        "holdout_scalar_count": int(np.count_nonzero(holdout_mask) * 3),
                        "holdout_fraction_of_observed": float(
                            np.count_nonzero(holdout_mask) / np.count_nonzero(tensor_mask)
                        ),
                        "parameters_per_observed_scalar": float(
                            parameters / (np.count_nonzero(tensor_mask) * 3)
                        ),
                        "parameters_per_fit_scalar": float(
                            parameters / (np.count_nonzero(fit_mask) * 3)
                        ),
                        "holdout_relative_frobenius_error": relative_error(
                            missing_tensor, approximation.reconstruction, holdout_mask
                        ),
                        "missing_truth_available": False,
                    }
                    holdout_series = _series(holdout_mask)
                    for component, name in enumerate(COMPONENTS):
                        component_valid = holdout_series[..., 0]
                        difference = (
                            estimate[..., component] - missing_series[..., component]
                        )
                        row[f"holdout_rmse_{name}_m_s"] = (
                            float(np.sqrt(np.mean(difference[component_valid] ** 2)))
                            if np.any(component_valid) else float("nan")
                        )
                    row.update(metrics)
                    rank_rows.append(row)
                    for level, z_value in enumerate(bz):
                        for component, name in enumerate(COMPONENTS):
                            valid = diagnostic_masks[:, level]
                            difference = (
                                estimate[:, level, ..., component]
                                - missing_series[:, level, ..., component]
                            )
                            height_rows.append({
                                "window": window_id, "representation": representation,
                                "domain": "B_structured_missing", "rank_name": rank_name,
                                "z_m": float(z_value), "component": name,
                                "rmse_m_s": float(np.sqrt(np.mean(difference[valid] ** 2)))
                                if np.any(valid) else float("nan"),
                            })
    _write_csv(output_dir / "mode_spectra.csv", spectrum_rows)
    _write_csv(output_dir / "rank_reconstruction_metrics.csv", rank_rows)
    _write_csv(output_dir / "height_rmse.csv", height_rows)
    summary = {
        "scope": "Step 5 feasibility only; no Tucker manifold or optimisation",
        "code_commit": _commit(),
        "input_uniform_baseline_commit": json.loads(
            (BASELINE_DIR / "manifest.json").read_text(encoding="utf-8")
        )["code_commit"],
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (
                ROOT / "src" / "low_rank_diagnostics.py",
                ROOT / "tools" / "run_step5_uniform_baselines.py",
                ROOT / "tools" / "plot_step5_low_rank_results.py",
                Path(__file__).resolve(),
            )
        },
        "uniform_baseline_sha256": {
            record["file"]: _sha256(ROOT / record["file"])
            for record in baseline_records
        },
        "n_uniform_baselines": len(baseline_records),
        "windows": {"K3": 4, "K4": 3, "K5": 2, "total": 9},
        "tensor_order": ["x", "y", "z", "time", "component"],
        "component_rank_fixed": 3,
        "rank_rule": {
            "aggressive_temporal": ["ceil(0.125 nx),min>=2", "ceil(0.125 ny),min>=2", "ceil(0.25 nz),min>=2", 1, 3],
            "compact_temporal": ["ceil(0.25 nx),min>=2", "ceil(0.25 ny),min>=2", "ceil(0.5 nz),min>=2", "min(2,K)", 3],
            "moderate_temporal": ["ceil(0.5 nx),min>=2", "ceil(0.5 ny),min>=2", "ceil(0.75 nz),min>=2", "min(3,K)", 3],
            "compact_rt_full": ["same compact spatial", "rt=K", "rc=3"],
            "moderate_rt_full": ["same moderate spatial", "rt=K", "rc=3"],
        },
        "missing_domain_method": {
            "initialisation": "per-component observed mean; global observed mean fallback; never zero spectrum",
            "iterations": MISSING_COMPLETION_ITERATIONS,
            "holdout": "fixed deterministic 10% lattice of available spatial-time cells; all 3 components held out together",
            "interpretation": "available-sample representation plus holdout diagnostic only; no truly missing wind truth available",
        },
        "motion_source": "frozen Step 4 NCC estimates; no retuning",
        "dense_domains": domains,
        "output_tables": ["mode_spectra.csv", "rank_reconstruction_metrics.csv", "height_rmse.csv"],
    }
    (output_dir / "step5_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    summary = run(args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
