import unittest
from tests.helpers import TestHelpers
import numpy as np
import datetime as dt
from rbamlib.utils import idx
from rbamlib.utils import parse_datetime
from rbamlib.utils import storm_idx
from rbamlib.utils import fixfill


class TestUtils(unittest.TestCase):

    def setUp(self):
        """Set up common test variables."""
        self.arr = np.array([1, 2, 3, 4, 5])

    def tearDown(self):
        """Clean up after tests if necessary."""
        pass

    def test_idx(self):
        """Test idx function for various cases."""
        # Test without tolerance
        self.assertEqual(idx(self.arr, 3.1), 2, "Nearest index without tolerance is incorrect.")
        self.assertEqual(idx(self.arr, 4.9), 4, "Nearest index without tolerance is incorrect.")

        # Test with tolerance
        self.assertTrue(np.isnan(idx(self.arr, 3.1, tol=0.05)), "Index should be NaN when outside tolerance.")
        self.assertEqual(idx(self.arr, 3.1, tol=0.2), 2, "Index within tolerance is incorrect.")

        # Edge cases
        self.assertEqual(idx(self.arr, 1), 0, "Index for exact match is incorrect.")
        self.assertEqual(idx(self.arr, 5), 4, "Index for exact match is incorrect.")
        self.assertTrue(np.isnan(idx(self.arr, 0, tol=0.5)),
                        "Index should be NaN when value is out of range with tolerance.")

    def test_parse_datetime(self):
        """Test parse_datetime function with various formats."""
        self.assertEqual(parse_datetime("2023100112"), dt.datetime(2023, 10, 1, 12),
                         "Failed to parse YYYYMMDDHH format.")
        self.assertEqual(parse_datetime("2023-10-01"), dt.datetime(2023, 10, 1),
                         "Failed to parse YYYY-MM-DD format.")
        self.assertEqual(parse_datetime("20231001"), dt.datetime(2023, 10, 1),
                         "Failed to parse YYYYMMDD format.")
        self.assertEqual(parse_datetime("20231101"), dt.datetime(2023, 11, 1),
                         "Failed to parse YYYYMMDD format.")
        self.assertEqual(parse_datetime("20231001T12:00"), dt.datetime(2023, 10, 1, 12),
                         "Failed to parse ISO-like format.")
        self.assertEqual(parse_datetime("2023-10-01T12:00"), dt.datetime(2023, 10, 1, 12),
                         "Failed to parse ISO-like format.")
        self.assertEqual(parse_datetime("2023-10-01 12:30"), dt.datetime(2023, 10, 1, 12, 30),
                         "Failed to parse standard format.")
        self.assertEqual(parse_datetime("01-10-2023"), dt.datetime(2023, 10, 1),
                         "Failed to parse European format.")
        self.assertEqual(parse_datetime("Oct 01, 2023"), dt.datetime(2023, 10, 1),
                         "Failed to parse human-readable format.")

        # Test passing a datetime object
        dt_obj = dt.datetime(2023, 10, 1, 14)
        self.assertEqual(parse_datetime(dt_obj), dt_obj, "Failed to handle already datetime object.")

        # Test invalid format
        with self.assertRaises(ValueError):
            parse_datetime("invalid-date")

    def test_storm_idx(self):
        """Test storm_idx function using minimum nad onset methods."""
        time = [dt.datetime(2023, 1, 1) + dt.timedelta(hours=i) for i in range(10)]
        dst = np.array([5, 3, -10, -45, -50, -20, 1, 2, -42, -10])
        indices = storm_idx(time, dst, threshold=-40, method='minimum', gap_hours=2.0)
        self.assertEqual(list(indices), [4, 8], "Minimum Dst indices incorrect.")

        dst = np.array([5, 3, -10, -45, -50, -20, 1, 2, -42, -10])
        indices = storm_idx(time, dst, threshold=-40, method='onset')
        self.assertEqual(list(indices), [1, 7], "Onset Dst indices incorrect.")

    def test_fixfill_nan(self):
        """Test fixfill function with NaN replacement."""
        time = [dt.datetime(2023, 1, 1, 0, 0) + dt.timedelta(minutes=5*i) for i in range(6)]
        data = np.array([1.0, 2.0, 999, 4.0, 999, 6.0])
        cleaned = fixfill(time, data, fillval=999, method='nan')
        expected = np.array([1.0, 2.0, np.nan, 4.0, np.nan, 6.0])
        np.testing.assert_array_equal(np.isnan(cleaned), np.isnan(expected), "NaN fill values incorrect.")

    def test_fixfill_interp(self):
        """Test fixfill function with interpolation."""
        time = [dt.datetime(2023, 1, 1, 0, 0) + dt.timedelta(minutes=5*i) for i in range(6)]
        data = np.array([1.0, 2.0, 999, 4.0, 999, 6.0])
        interpolated = fixfill(time, data, fillval=999, method='interp')
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        np.testing.assert_almost_equal(interpolated, expected, decimal=1, err_msg="Interpolation incorrect.")

    def test_fixfill_nan_fillval_mode(self):
        """Test fixfill function with NaN replacement."""
        time = [dt.datetime(2023, 1, 1, 0, 0) + dt.timedelta(minutes=5*i) for i in range(6)]
        data = np.array([1.0, 2.0, 999, 4.0, 999, 6.0])
        cleaned = fixfill(time, data, fillval=99, method='nan', fillval_mode='gt')
        expected = np.array([1.0, 2.0, np.nan, 4.0, np.nan, 6.0])
        np.testing.assert_array_equal(np.isnan(cleaned), np.isnan(expected), "NaN fill values incorrect.")


if __name__ == '__main__':
    unittest.main()
