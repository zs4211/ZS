"""Audit radar conventions and compare the independent H with PyDDA.

This diagnostic reads local CINRAD data and an existing retrieval result. It
does not run or modify the PyDDA solver.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
from pathlib import Path
import sys

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.observation_operator import RadarObservationOperator


def _first_data_file(directory: Path, pattern: str = "*") -> Path:
    files = sorted(
        path for path in directory.glob(pattern) if path.name != "ProductIndex"
    )
    if not files:
        raise FileNotFoundError(f"No radar data files found in {directory}")
    return files[0]


def _load_pydda_angles_module():
    """Load PyDDA's angle implementation without importing its sample-data tests."""
    package_spec = importlib.util.find_spec("pydda")
    if package_spec is None or not package_spec.submodule_search_locations:
        raise RuntimeError("PyDDA installation not found")
    module_path = (
        Path(next(iter(package_spec.submodule_search_locations)))
        / "retrieval"
        / "angles.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "installed_pydda_angles", module_path
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module, module_path


def audit_actual_cinrad_data(station: str = "Z9539") -> None:
    """Print cardinal-axis, elevation, and empirical velocity-sign checks."""
    from cinrad.io import read_auto
    import pyart

    from src.pipeline import RadarDataPipeline

    velocity_file = _first_data_file(ROOT / "data" / station / "PPI" / "026")
    velocity_product = read_auto(str(velocity_file))
    dataset = velocity_product.get_data()
    azimuth = np.asarray(dataset["azimuth"].values, dtype=np.float64)
    distance = np.asarray(dataset["distance"].values, dtype=np.float64)
    longitude = np.asarray(dataset["longitude"].values, dtype=np.float64)
    latitude = np.asarray(dataset["latitude"].values, dtype=np.float64)
    height = np.asarray(dataset["height"].values, dtype=np.float64)

    print("[actual-data conventions]")
    print(f"file={velocity_file.relative_to(ROOT)}")
    print(
        "pycinrad_version="
        f"{importlib.metadata.version('cinrad')}; "
        f"azimuth_range_rad=[{azimuth.min():.9f}, {azimuth.max():.9f}]"
    )
    print(f"pycinrad_vc_attrs={dict(dataset[velocity_product.pname].attrs)}")
    pyart_azimuth_metadata = pyart.config.get_metadata("azimuth")
    pyart_elevation_metadata = pyart.config.get_metadata("elevation")
    pyart_velocity_metadata = pyart.config.get_metadata("velocity")
    print(
        f"pyart_version={pyart.__version__}; "
        f"azimuth_long_name={pyart_azimuth_metadata['long_name']}; "
        f"azimuth_comment={pyart_azimuth_metadata['comment']}; "
        f"elevation_long_name={pyart_elevation_metadata['long_name']}; "
        f"velocity_standard_name={pyart_velocity_metadata['standard_name']}"
    )

    timestamp = velocity_file.name.split("_")[1]
    pipeline = RadarDataPipeline(ROOT / "data")
    actual_sweeps = pipeline.build_volume(
        station, timestamp, product_type="velocity", resolution="high"
    )
    actual_pyart_radar = pipeline.sweeps_to_pyart_radar(actual_sweeps)
    actual_field_name = next(iter(actual_pyart_radar.fields))
    print(
        f"actual_pyart_azimuth_units={actual_pyart_radar.azimuth['units']}; "
        f"actual_pyart_elevation_units={actual_pyart_radar.elevation['units']}; "
        f"actual_pyart_velocity_standard_name="
        f"{actual_pyart_radar.fields[actual_field_name]['standard_name']}"
    )
    range_index = int(np.argmin(np.abs(distance - 50.0)))
    site_lon = float(dataset.attrs["site_longitude"])
    site_lat = float(dataset.attrs["site_latitude"])
    cardinal_offsets = {}
    for target_degrees in (0.0, 90.0, 180.0, 270.0):
        target_radians = np.deg2rad(target_degrees)
        angular_distance = np.abs(
            (azimuth - target_radians + np.pi) % (2.0 * np.pi) - np.pi
        )
        ray_index = int(np.argmin(angular_distance))
        cardinal_offsets[target_degrees] = (
            longitude[ray_index, range_index] - site_lon,
            latitude[ray_index, range_index] - site_lat,
        )
        print(
            f"az_target_deg={target_degrees:.0f}; "
            f"az_actual_deg={np.rad2deg(azimuth[ray_index]):.5f}; "
            f"delta_lon_deg={longitude[ray_index, range_index] - site_lon:.8f}; "
            f"delta_lat_deg={latitude[ray_index, range_index] - site_lat:.8f}"
        )
    print(
        "positive_elevation_height_rise="
        f"{height[0, -1] - height[0, 0]:.9f} "
        f"(native height units)"
    )
    assert cardinal_offsets[0.0][1] > 0.0
    assert cardinal_offsets[90.0][0] > 0.0
    assert cardinal_offsets[180.0][1] < 0.0
    assert cardinal_offsets[270.0][0] < 0.0
    assert height[0, -1] > height[0, 0]
    assert pyart_velocity_metadata["standard_name"].endswith(
        "away_from_instrument"
    )

    # Independent empirical sign check: compare the first azimuthal harmonic
    # of Vc with the simultaneous VWP wind direction. A positive dot product
    # supports positive radial velocity along the downwind (outward) beam.
    vwp_file = _first_data_file(
        ROOT / "data" / station / "VWP" / "048", f"{station}_{timestamp}_VWP_*"
    )
    vwp = read_auto(str(vwp_file)).get_data()
    vwp_speed = np.asarray(vwp["wind_speed"].values).squeeze()
    vwp_direction = np.asarray(vwp["wind_direction"].values).squeeze()
    valid_profile = np.flatnonzero(np.isfinite(vwp_speed) & (vwp_speed > 0.0))
    profile_index = int(valid_profile[0])
    profile_height_m = float(vwp["height"].values[profile_index])
    direction_from_deg = float(vwp_direction[profile_index])
    speed = float(vwp_speed[profile_index])
    expected_u = -speed * np.sin(np.deg2rad(direction_from_deg))
    expected_v = -speed * np.cos(np.deg2rad(direction_from_deg))

    height_m = height * 1000.0 if np.nanmax(height) < 100.0 else height
    radial_velocity = np.asarray(dataset[velocity_product.pname].values, dtype=float)
    sin_azimuth = np.broadcast_to(np.sin(azimuth)[:, None], radial_velocity.shape)
    cos_azimuth = np.broadcast_to(np.cos(azimuth)[:, None], radial_velocity.shape)
    fit_mask = (
        np.isfinite(radial_velocity)
        & (height_m >= profile_height_m - 150.0)
        & (height_m <= profile_height_m + 200.0)
    )
    design = np.column_stack(
        (
            sin_azimuth[fit_mask],
            cos_azimuth[fit_mask],
            np.ones(int(fit_mask.sum())),
        )
    )
    fitted_u, fitted_v, fitted_offset = np.linalg.lstsq(
        design, radial_velocity[fit_mask], rcond=None
    )[0]
    fitted_toward = (
        np.rad2deg(np.arctan2(fitted_u, fitted_v)) + 360.0
    ) % 360.0
    vwp_toward = (direction_from_deg + 180.0) % 360.0
    sign_dot_product = fitted_u * expected_u + fitted_v * expected_v
    print(
        "velocity_sign_check="
        f"samples:{fit_mask.sum()}, "
        f"vwp_toward_deg:{vwp_toward:.5f}, "
        f"vc_harmonic_toward_deg:{fitted_toward:.5f}, "
        f"same_sign_dot:{sign_dot_product:.9f}, "
        f"fitted_offset_mps:{fitted_offset:.9f}"
    )
    direction_difference = abs(
        (fitted_toward - vwp_toward + 180.0) % 360.0 - 180.0
    )
    assert sign_dot_product > 0.0
    assert direction_difference < 15.0


def compare_with_pydda() -> None:
    """Compare H and PyDDA's installed azimuth/elevation projection per grid cell."""
    from cinrad.io import read_auto
    from pyart.core.transforms import cartesian_to_geographic

    angles, angles_path = _load_pydda_angles_module()
    output_file = sorted((ROOT / "output").glob("wind3d_*.nc"))[-1]
    with xr.open_dataset(output_file, decode_times=False) as dataset:
        u = np.asarray(dataset["u"].values.squeeze(), dtype=np.float64)
        v = np.asarray(dataset["v"].values.squeeze(), dtype=np.float64)
        w = np.asarray(dataset["w"].values.squeeze(), dtype=np.float64)
        wind = np.stack((u, v, w), axis=0)
        x = np.asarray(dataset["x"].values, dtype=np.float64)
        y = np.asarray(dataset["y"].values, dtype=np.float64)
        z = np.asarray(dataset["z"].values, dtype=np.float64)
        origin_lat = float(dataset["origin_latitude"].values.flat[0])
        origin_lon = float(dataset["origin_longitude"].values.flat[0])
        projection = dict(dataset["projection"].attrs)
        projection["lat_0"] = origin_lat
        projection["lon_0"] = origin_lon
        station_codes = (dataset.attrs["station1"], dataset.attrs["station2"])

    point_x, point_y = np.meshgrid(x, y, indexing="xy")
    point_lon, point_lat = cartesian_to_geographic(
        point_x, point_y, projection
    )

    filename_parts = output_file.stem.split("_")
    timestamps = (filename_parts[-2], filename_parts[-1])
    all_absolute_differences = []

    print("[PyDDA forward comparison]")
    print(
        f"output={output_file.relative_to(ROOT)}; "
        f"pydda_version={importlib.metadata.version('pydda')}; "
        f"angles_source={angles_path}"
    )
    for station, timestamp in zip(station_codes, timestamps):
        radar_file = _first_data_file(
            ROOT / "data" / station / "PPI" / "026",
            f"{station}_{timestamp}_PPI_*_026",
        )
        radar = read_auto(str(radar_file))
        radar_lat = float(radar.stationlat)
        radar_lon = float(radar.stationlon)
        radar_alt = float(radar.radarheight)

        azimuth_2d = angles.gc_bear_array(
            radar_lat, radar_lon, point_lat, point_lon
        )
        ground_range_2d = angles.gc_dist(
            radar_lat, radar_lon, point_lat, point_lon
        )
        azimuth = np.broadcast_to(azimuth_2d, u.shape)
        elevation = np.empty_like(u)
        for level, height_m in enumerate(z):
            _, elevation[level] = angles.rsl_get_slantr_and_elev(
                ground_range_2d, (height_m - radar_alt) / 1000.0
            )

        valid = (
            np.isfinite(wind).all(axis=0)
            & np.isfinite(azimuth)
            & np.isfinite(elevation)
        )
        operator = RadarObservationOperator.from_azimuth_elevation(
            azimuth, elevation, valid_mask=valid, degrees=True
        )
        independent = operator.forward(wind)

        # Installed PyDDA 2.4.1 uses this wind-only part of its observation
        # forward model. Fall speed is a separate known affine correction.
        azimuth_rad = np.deg2rad(azimuth)
        elevation_rad = np.deg2rad(elevation)
        pydda_reference = (
            np.cos(elevation_rad) * np.sin(azimuth_rad) * u
            + np.cos(elevation_rad) * np.cos(azimuth_rad) * v
            + np.sin(elevation_rad) * w
        )
        difference = np.abs(independent[valid] - pydda_reference[valid])
        all_absolute_differences.append(difference)
        print(
            f"station={station}; compared_cells={difference.size}; "
            f"max_abs_discrepancy_mps={difference.max():.17g}; "
            f"mean_abs_discrepancy_mps={difference.mean():.17g}"
        )

    combined = np.concatenate(all_absolute_differences)
    print(
        f"combined_cells={combined.size}; "
        f"max_abs_discrepancy_mps={combined.max():.17g}; "
        f"mean_abs_discrepancy_mps={combined.mean():.17g}"
    )
    print(
        "difference_scope=wind-only linear H; PyDDA terminal-fall-speed term "
        "-sin(elevation)*abs(wt) is an affine observation correction"
    )
    assert combined.max() <= 1.0e-12


def audit_adjoint() -> None:
    """Report the worst weighted adjoint error over deterministic random cases."""
    worst_relative_error = 0.0
    for seed in range(10):
        rng = np.random.default_rng(seed)
        shape = (5, 7, 9)
        azimuth = rng.uniform(0.0, 360.0, shape)
        elevation = rng.uniform(-3.0, 35.0, shape)
        mask = rng.random(shape) > 0.31
        weights = rng.lognormal(mean=0.0, sigma=1.0, size=shape)
        weights[rng.random(shape) < 0.13] = 0.0
        wind = rng.normal(size=(3,) + shape)
        radial_test = rng.normal(size=shape)
        operator = RadarObservationOperator.from_azimuth_elevation(
            azimuth, elevation, valid_mask=mask
        )
        left = np.vdot(operator.weighted_forward(wind, weights), radial_test)
        right = np.vdot(
            wind, operator.weighted_adjoint(radial_test, weights)
        )
        denominator = max(abs(left), abs(right), np.finfo(float).eps)
        worst_relative_error = max(
            worst_relative_error, float(abs(left - right) / denominator)
        )
    print(
        "[adjoint test]\n"
        f"random_cases=10; worst_relative_error={worst_relative_error:.17g}; "
        "acceptance_limit=1e-7"
    )
    assert worst_relative_error <= 1.0e-7


if __name__ == "__main__":
    audit_actual_cinrad_data()
    audit_adjoint()
    compare_with_pydda()
