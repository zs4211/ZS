"""Matrix-free Doppler-radar wind observation operator.

The convention is the one used by Py-ART/CF-Radial and PyDDA:

* ``u`` is positive eastward, ``v`` northward, and ``w`` upward;
* azimuth is measured clockwise from true north;
* elevation is positive above the local horizontal plane; and
* radial velocity is positive away from the radar.

For a beam at azimuth ``alpha`` and elevation ``epsilon``, the wind-only
radial velocity is

``Vr = cos(epsilon) sin(alpha) u
    + cos(epsilon) cos(alpha) v
    + sin(epsilon) w``.

Hydrometeor terminal fall speed is an affine observation correction and is
deliberately not part of this linear operator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def beam_direction_cosines(
    azimuth: np.ndarray,
    elevation: np.ndarray,
    *,
    degrees: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return east, north, and up components of the outward beam unit vector."""
    azimuth_array, elevation_array = np.broadcast_arrays(
        np.asarray(azimuth, dtype=np.float64),
        np.asarray(elevation, dtype=np.float64),
    )
    if degrees:
        azimuth_array = np.deg2rad(azimuth_array)
        elevation_array = np.deg2rad(elevation_array)

    cos_elevation = np.cos(elevation_array)
    return (
        cos_elevation * np.sin(azimuth_array),
        cos_elevation * np.cos(azimuth_array),
        np.sin(elevation_array),
    )


@dataclass(frozen=True)
class RadarObservationOperator:
    """Matrix-free masked observation operator for one radar.

    ``forward`` implements :math:`P_\\Omega H`, while ``adjoint`` implements
    :math:`H^* P_\\Omega`. The weighted variants implement
    :math:`W^{1/2}P_\\Omega H` and its exact Euclidean adjoint. Reliability
    weights therefore retain their usual quadratic-objective meaning.
    """

    beam_east: np.ndarray
    beam_north: np.ndarray
    beam_up: np.ndarray
    valid_mask: np.ndarray

    @classmethod
    def from_azimuth_elevation(
        cls,
        azimuth: np.ndarray,
        elevation: np.ndarray,
        *,
        valid_mask: np.ndarray | None = None,
        degrees: bool = True,
    ) -> "RadarObservationOperator":
        """Construct the operator without assembling an observation matrix."""
        beam_east, beam_north, beam_up = beam_direction_cosines(
            azimuth, elevation, degrees=degrees
        )
        geometry_valid = (
            np.isfinite(beam_east)
            & np.isfinite(beam_north)
            & np.isfinite(beam_up)
        )
        if valid_mask is None:
            mask = geometry_valid
        else:
            mask = np.broadcast_to(
                np.asarray(valid_mask, dtype=bool), beam_east.shape
            ).copy()
            mask &= geometry_valid

        # Invalid geometry is represented by a zero row rather than NaN so
        # that masked values cannot leak through multiplication by zero.
        return cls(
            beam_east=np.where(mask, beam_east, 0.0),
            beam_north=np.where(mask, beam_north, 0.0),
            beam_up=np.where(mask, beam_up, 0.0),
            valid_mask=mask,
        )

    @property
    def observation_shape(self) -> tuple[int, ...]:
        """Shape of this radar's gridded observation space."""
        return self.valid_mask.shape

    def _wind_components(
        self, wind: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(wind, tuple):
            if len(wind) != 3:
                raise ValueError("wind tuple must contain exactly (u, v, w)")
            components = tuple(np.asarray(item, dtype=np.float64) for item in wind)
        else:
            wind_array = np.asarray(wind, dtype=np.float64)
            expected_shape = (3,) + self.observation_shape
            if wind_array.shape != expected_shape:
                raise ValueError(
                    f"wind must have shape {expected_shape}, got {wind_array.shape}"
                )
            components = (wind_array[0], wind_array[1], wind_array[2])

        try:
            return tuple(
                np.broadcast_to(component, self.observation_shape)
                for component in components
            )
        except ValueError as exc:
            raise ValueError(
                f"wind components must broadcast to {self.observation_shape}"
            ) from exc

    def forward(
        self, wind: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> np.ndarray:
        """Project an east/north/up wind field onto the outward radar beam."""
        u, v, w = self._wind_components(wind)
        radial_velocity = (
            self.beam_east * u
            + self.beam_north * v
            + self.beam_up * w
        )
        return np.where(self.valid_mask, radial_velocity, 0.0)

    def adjoint(self, radial: np.ndarray) -> np.ndarray:
        """Apply the exact Euclidean adjoint of :meth:`forward`."""
        radial_array = np.broadcast_to(
            np.asarray(radial, dtype=np.float64), self.observation_shape
        )
        radial_array = np.where(self.valid_mask, radial_array, 0.0)
        return np.stack(
            (
                self.beam_east * radial_array,
                self.beam_north * radial_array,
                self.beam_up * radial_array,
            ),
            axis=0,
        )

    def _sqrt_weights(self, weights: np.ndarray) -> np.ndarray:
        weight_array = np.broadcast_to(
            np.asarray(weights, dtype=np.float64), self.observation_shape
        )
        active_weights = weight_array[self.valid_mask]
        if not np.all(np.isfinite(active_weights)):
            raise ValueError("weights must be finite on valid observations")
        if np.any(active_weights < 0.0):
            raise ValueError("weights must be non-negative")
        return np.where(self.valid_mask, np.sqrt(np.maximum(weight_array, 0.0)), 0.0)

    def weighted_forward(
        self,
        wind: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray],
        weights: np.ndarray,
    ) -> np.ndarray:
        """Apply :math:`W^{1/2}P_\\Omega H` without forming a matrix."""
        return self._sqrt_weights(weights) * self.forward(wind)

    def weighted_adjoint(
        self, radial: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Apply :math:`H^*P_\\Omega W^{1/2}` exactly."""
        return self.adjoint(self._sqrt_weights(weights) * radial)
