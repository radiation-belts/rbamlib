import unittest
from tests.helpers import TestHelpers
import numpy as np
import datetime as dt
from rbamlib.utils import idx
from rbamlib.utils import parse_datetime


class TestUtils(unittest.TestCase):
    # TODO: add test for fixfill

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

if __name__ == '__main__':
    unittest.main()