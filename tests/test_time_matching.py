"""Regression tests for temporal consistency safeguards."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.cinrad_products import load_mesocyclones
from src.pipeline import RadarDataPipeline


class TimeMatchingTest(unittest.TestCase):
    @staticmethod
    def _touch_product(root, station, timestamp, product="026"):
        directory = Path(root) / station / "PPI" / product
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{station}_{timestamp}_PPI_01_{product}").touch()

    def test_default_two_minute_limit_and_one_to_one_matching(self):
        with TemporaryDirectory() as temp_dir:
            self._touch_product(temp_dir, "A", "20240101000000Z")
            self._touch_product(temp_dir, "A", "20240101001000Z")
            self._touch_product(temp_dir, "B", "20240101000130Z")
            self._touch_product(temp_dir, "B", "20240101001300Z")

            matches = RadarDataPipeline.match_times(
                "A", "B", Path(temp_dir),
                product_type="velocity", resolution="high",
            )

        self.assertEqual(
            matches,
            [("20240101000000Z", "20240101000130Z", 1.5)],
        )

    def test_missing_station_directories_return_no_matches(self):
        with TemporaryDirectory() as temp_dir:
            matches = RadarDataPipeline.match_times(
                "A", "B", Path(temp_dir),
                product_type="velocity", resolution="high",
            )
        self.assertEqual(matches, [])

    def test_mesocyclone_product_outside_tolerance_is_rejected(self):
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir) / "A" / "M" / "060"
            directory.mkdir(parents=True)
            (directory / "A_20240101000000Z_M_00_060").touch()

            result = load_mesocyclones(
                "A", "20240201000000Z",
                data_dir=temp_dir,
                tol_seconds=600,
            )

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
