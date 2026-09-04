"""Regression test for compatible PROJ database selection."""

import unittest

from src.proj_setup import configure_proj


class ProjSetupTest(unittest.TestCase):
    def test_epsg_transform_is_available(self):
        data_dir = configure_proj()
        from pyproj import Transformer

        x, y = Transformer.from_crs(
            4326, 3857, always_xy=True
        ).transform(120.0, 36.0)

        self.assertTrue((data_dir / "proj.db").is_file())
        self.assertAlmostEqual(x, 13358338.895, places=2)
        self.assertAlmostEqual(y, 4300621.372, places=2)


if __name__ == "__main__":
    unittest.main()
