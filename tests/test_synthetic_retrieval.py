"""Closed-loop tests for the traditional dual-Doppler initializer."""

from types import SimpleNamespace
import unittest

import numpy as np

from src.dual_radar_retrieval import dual_radar_retrieve
from src.gridder import wgs84_to_enu


def _radar(lat, lon, altitude_m=0.0):
    return SimpleNamespace(
        latitude={"data": np.array([lat])},
        longitude={"data": np.array([lon])},
        altitude={"data": np.array([altitude_m])},
    )


def _grid(x, y, z, radial_velocity, origin_lat=35.0, origin_lon=118.0):
    return SimpleNamespace(
        x={"data": x}, y={"data": y}, z={"data": z},
        origin_latitude={"data": np.array([origin_lat])},
        origin_longitude={"data": np.array([origin_lon])},
        origin_altitude={"data": np.array([0.0])},
        fields={"Vc": {"data": np.ma.array(radial_velocity, mask=False)}},
    )


class SyntheticRetrievalTest(unittest.TestCase):
    def test_known_solid_body_vortex_is_recovered(self):
        """Project a known vortex to two radars and retrieve it again."""
        origin_lat, origin_lon = 35.0, 118.0
        radar1 = _radar(origin_lat, origin_lon - 0.55)
        radar2 = _radar(origin_lat, origin_lon + 0.55)

        x = np.linspace(-40_000.0, 40_000.0, 17)
        y = np.linspace(-40_000.0, 40_000.0, 17)
        z = np.linspace(500.0, 2_500.0, 5)
        zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

        # Analytic, horizontally non-divergent solid-body rotation plus background.
        omega = 2.0e-4
        u_true = 8.0 - omega * yy
        v_true = 3.0 + omega * xx
        w_true = np.zeros_like(u_true)

        def project(radar):
            e, n, up = wgs84_to_enu(
                np.array([radar.latitude["data"][0]]),
                np.array([radar.longitude["data"][0]]),
                np.array([0.0]), origin_lat, origin_lon, 0.0,
            )
            dx = xx - e.item() * 1000.0
            dy = yy - n.item() * 1000.0
            dz = zz - up.item() * 1000.0
            distance = np.sqrt(dx * dx + dy * dy + dz * dz)
            return (u_true * dx / distance + v_true * dy / distance
                    + w_true * dz / distance)

        grid1 = _grid(x, y, z, project(radar1), origin_lat, origin_lon)
        grid2 = _grid(x, y, z, project(radar2), origin_lat, origin_lon)

        u, v, w, valid, angle = dual_radar_retrieve(
            grid1, grid2, radar1, radar2,
            min_cross_beam_angle_deg=20.0,
            max_iterations=0,
        )

        self.assertGreater(valid.sum(), 300)
        self.assertTrue(np.all(angle[valid] >= 20.0))
        self.assertLess(np.max(np.abs(u[valid] - u_true[valid])), 1.0e-4)
        self.assertLess(np.max(np.abs(v[valid] - v_true[valid])), 1.0e-4)
        self.assertLess(np.max(np.abs(w[valid])), 1.0e-5)


if __name__ == "__main__":
    unittest.main()
