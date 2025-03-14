import unittest
from tests.helpers import TestHelpers
import numpy as np
from rbamlib.utils import idx


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


if __name__ == '__main__':
    unittest.main()