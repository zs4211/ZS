"""Re-run six frozen PyDDA baselines and standardize mask semantics for Step 5.

The scientific retrieval configuration is copied verbatim from the frozen
Step 0 NetCDF metadata.  This tool adds output provenance and independently
derived masks after retrieval; it does not alter the solver, QC, filtering, or
BCA implementation.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
import subprocess
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
from src.pipeline import RadarDataPipeline
from src.retrieval import WindRetrieval


STATION1 = "Z9539"
STATION2 = "Z9532"
TIME_PAIRS = (
    ("20260511070939Z", "20260511070929Z"),
    ("20260511071440Z", "20260511071502Z"),
    ("20260511071941Z", "20260511072036Z"),
    ("20260511072514Z", "20260511072609Z"),
    ("20260511073015Z", "20260511073142Z"),
    ("20260511073515Z", "20260511073715Z"),
)
OUTPUT_DIR = ROOT / "artifacts" / "step5_low_rank" / "uniform_baseline"
CONFIG = {
    "grid_shape": (20, 201, 201),
    "grid_limits_m": (
        (500.0, 15_000.0),
        (-150_000.0, 150_000.0),
        (-150_000.0, 150_000.0),
    ),
    "Co": 100.0,
    "Cb": 0.01,
    "Cm": 1000.0,
    "Cx": 1e-4,
    "Cy": 1e-4,
    "Cz": 1e-4,
    "wind_tol": 0.5,
    "max_iterations": 150,
    "filter_window": 15,
    "filter_order": 3,
    "min_cross_beam_angle_deg": 20.0,
    "dual_radar_iterations": 0,
    "engine": "scipy",
}


def code_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def add_uniform_schema(path: Path, retrieval: WindRetrieval, commit: str,
                       timestamp1: str, timestamp2: str) -> dict:
    grid1, grid2 = retrieval.grid1, retrieval.grid2
    field = retrieval.vel_field
    radar1_valid = ~np.ma.getmaskarray(grid1.fields[field]["data"])
    radar2_valid = ~np.ma.getmaskarray(grid2.fields[field]["data"])
    dual_observation = radar1_valid & radar2_valid

    with xr.open_dataset(path, decode_times=False) as opened:
        dataset = opened.load()
    x = np.asarray(dataset.x.values, dtype=np.float64)
    y = np.asarray(dataset.y.values, dtype=np.float64)
    z = np.asarray(dataset.z.values, dtype=np.float64)
    wind = tuple(
        np.asarray(dataset[name].values, dtype=np.float64).squeeze()
        for name in ("u", "v", "w")
    )
    finite_wind = np.logical_and.reduce([np.isfinite(item) for item in wind])
    cross_beam_angle = np.asarray(
        dataset["cross_beam_angle"].values, dtype=np.float64
    ).squeeze()
    bca_mask = (
        dual_observation
        & np.isfinite(cross_beam_angle)
        & (cross_beam_angle >= CONFIG["min_cross_beam_angle_deg"])
    )
    continuity = AnelasticMassContinuityOperator(
        x, y, z,
        lambda z_m: exponential_density(z_m, scale_height_m=10_000.0),
    )
    physics_mask = continuity.physical_valid_mask(
        wind, base_mask=bca_mask & finite_wind
    )

    dimensions = ("time", "z", "y", "x")
    additions = {
        "observation_mask_radar1": radar1_valid,
        "observation_mask_radar2": radar2_valid,
        "dual_radar_observation_mask": dual_observation,
        "bca_ge_20_mask": bca_mask,
        "baseline_finite_wind_mask": finite_wind,
        "physics_mask": physics_mask,
    }
    for name, values in additions.items():
        dataset[name] = (dimensions, values[None].astype(np.uint8))
        dataset[name].attrs.update({
            "units": "1",
            "flag_values": np.array([0, 1], dtype=np.uint8),
            "flag_meanings": "invalid valid",
        })
    dataset["observation_mask_radar1"].attrs["semantics"] = (
        "post-QC gridded radial velocity available from Z9539"
    )
    dataset["observation_mask_radar2"].attrs["semantics"] = (
        "post-QC gridded radial velocity available from Z9532"
    )
    dataset["dual_radar_observation_mask"].attrs["semantics"] = (
        "both post-QC gridded radial velocities available; no BCA threshold"
    )
    dataset["bca_ge_20_mask"].attrs["semantics"] = (
        "dual_radar_observation_mask and frozen symmetric BCA >= 20 degrees"
    )
    dataset["baseline_finite_wind_mask"].attrs["semantics"] = (
        "u, v, and w are all finite in this uniformly serialized baseline"
    )
    dataset["physics_mask"].attrs["semantics"] = (
        "Step 3 complete D_rho stencil inside bca_ge_20_mask and finite wind; "
        "not a zero-fill mask"
    )
    dataset.attrs.update({
        "code_commit": commit,
        "frozen_parent_tag": "step4-temporal",
        "uniform_baseline_schema": "step5-v1",
        "input_station1": STATION1,
        "input_station2": STATION2,
        "input_timestamp1": timestamp1,
        "input_timestamp2": timestamp2,
        "retrieval_configuration_json": json.dumps(CONFIG, sort_keys=True),
        "physics_mask_density_profile": "rho0 proportional to exp(-z/10000m)",
        "mask_semantics_note": (
            "observation, dual-radar, BCA, finite-wind, and physics masks are "
            "distinct; missing values are not physical zero wind"
        ),
    })

    temporary = path.with_suffix(".uniform.tmp.nc")
    for variable in dataset.variables:
        dataset[variable].encoding = {}
    encoding = {
        name: {"zlib": True, "complevel": 4}
        for name in dataset.data_vars
        if dataset[name].ndim >= 2
    }
    dataset.to_netcdf(temporary, encoding=encoding)
    temporary.replace(path)
    return {
        "file": str(path.relative_to(ROOT)),
        "timestamp1": timestamp1,
        "timestamp2": timestamp2,
        "code_commit": commit,
        "radar1_observation_count": int(np.count_nonzero(radar1_valid)),
        "radar2_observation_count": int(np.count_nonzero(radar2_valid)),
        "dual_observation_count": int(np.count_nonzero(dual_observation)),
        "bca_ge_20_count": int(np.count_nonzero(bca_mask)),
        "finite_wind_count": int(np.count_nonzero(finite_wind)),
        "physics_mask_count": int(np.count_nonzero(physics_mask)),
        "radial_velocity_rmse_radar1": dataset.attrs.get(
            "radial_velocity_rmse_radar1"
        ),
        "radial_velocity_rmse_radar2": dataset.attrs.get(
            "radial_velocity_rmse_radar2"
        ),
    }


def output_is_complete(path: Path, commit: str) -> bool:
    if not path.is_file():
        return False
    try:
        with xr.open_dataset(path, decode_times=False) as dataset:
            required = {
                "u", "v", "w", "reflectivity", "observation_mask_radar1",
                "observation_mask_radar2", "dual_radar_observation_mask",
                "bca_ge_20_mask", "baseline_finite_wind_mask", "physics_mask",
            }
            return (
                required <= set(dataset.data_vars)
                and dataset.attrs.get("code_commit") == commit
                and dataset.attrs.get("retrieval_configuration_json")
                == json.dumps(CONFIG, sort_keys=True)
            )
    except Exception:
        return False


def run_one(pipeline: RadarDataPipeline, timestamp1: str, timestamp2: str,
            commit: str) -> dict:
    output_path = OUTPUT_DIR / f"wind3d_{timestamp1}_{timestamp2}.nc"
    if output_is_complete(output_path, commit):
        print(f"[resume] verified existing {output_path.name}")
        with xr.open_dataset(output_path, decode_times=False) as dataset:
            return {
                "file": str(output_path.relative_to(ROOT)),
                "timestamp1": timestamp1,
                "timestamp2": timestamp2,
                "code_commit": commit,
                "radar1_observation_count": int(dataset.observation_mask_radar1.sum()),
                "radar2_observation_count": int(dataset.observation_mask_radar2.sum()),
                "dual_observation_count": int(dataset.dual_radar_observation_mask.sum()),
                "bca_ge_20_count": int(dataset.bca_ge_20_mask.sum()),
                "finite_wind_count": int(dataset.baseline_finite_wind_mask.sum()),
                "physics_mask_count": int(dataset.physics_mask.sum()),
                "radial_velocity_rmse_radar1": dataset.attrs.get(
                    "radial_velocity_rmse_radar1"
                ),
                "radial_velocity_rmse_radar2": dataset.attrs.get(
                    "radial_velocity_rmse_radar2"
                ),
            }

    velocity_radars = []
    reflectivity_radars = []
    for station, timestamp in ((STATION1, timestamp1), (STATION2, timestamp2)):
        velocity_sweeps = pipeline.build_volume(
            station, timestamp, "velocity", "high"
        )
        reflectivity_sweeps = pipeline.build_volume(
            station, timestamp, "reflectivity", "high"
        )
        if not velocity_sweeps or not reflectivity_sweeps:
            raise RuntimeError(f"missing input volume {station} {timestamp}")
        velocity_radars.append(pipeline.quality_control(
            pipeline.sweeps_to_pyart_radar(velocity_sweeps)
        ))
        reflectivity_radars.append(pipeline.quality_control(
            pipeline.sweeps_to_pyart_radar(reflectivity_sweeps)
        ))

    retrieval = WindRetrieval(
        velocity_radars[0], velocity_radars[1],
        reflectivity_radar1=reflectivity_radars[0],
        reflectivity_radar2=reflectivity_radars[1],
        vel_field=next(iter(velocity_radars[0].fields)),
        grid_shape=CONFIG["grid_shape"],
        grid_limits=CONFIG["grid_limits_m"],
        station1=STATION1,
        station2=STATION2,
        data_dir=str(ROOT / "data"),
    )
    initial = retrieval.make_initial_wind(
        min_cross_beam_angle_deg=CONFIG["min_cross_beam_angle_deg"],
        max_iterations=CONFIG["dual_radar_iterations"],
    )
    retrieval.retrieve(
        u_init=initial[0], v_init=initial[1], w_init=initial[2],
        Co=CONFIG["Co"], Cb=CONFIG["Cb"], Cm=CONFIG["Cm"],
        Cx=CONFIG["Cx"], Cy=CONFIG["Cy"], Cz=CONFIG["Cz"],
        wind_tol=CONFIG["wind_tol"],
        max_iterations=CONFIG["max_iterations"],
        filter_window=CONFIG["filter_window"],
        filter_order=CONFIG["filter_order"],
        min_bca=CONFIG["min_cross_beam_angle_deg"],
        engine=CONFIG["engine"],
    )
    retrieval.save(str(output_path))
    result = add_uniform_schema(
        output_path, retrieval, commit, timestamp1, timestamp2
    )
    del retrieval, initial, velocity_radars, reflectivity_radars
    gc.collect()
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    commit = code_commit()
    pipeline = RadarDataPipeline(ROOT / "data")
    records = []
    for index, pair in enumerate(TIME_PAIRS):
        print(f"\n=== uniform baseline {index + 1}/{len(TIME_PAIRS)}: {pair} ===")
        records.append(run_one(pipeline, pair[0], pair[1], commit))
        with (OUTPUT_DIR / "manifest.json").open("w", encoding="utf-8") as stream:
            json.dump({
                "code_commit": commit,
                "frozen_parent_tag": "step4-temporal",
                "configuration": CONFIG,
                "completed": records,
            }, stream, indent=2)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
