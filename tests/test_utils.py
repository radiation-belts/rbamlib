import unittest
from tests.helpers import TestHelpers
import numpy as np
from rbamlib.utils import idx


class TestUtils(unittest.TestCase, TestHelpers):

    def test_idx(self):
        """Test idx function for various cases."""
        arr = np.array([1, 2, 3, 4, 5])

        # Test without tolerance
        assert idx(arr, 3.1) == 2, "Nearest index without tolerance is incorrect."
        assert idx(arr, 4.9) == 4, "Nearest index without tolerance is incorrect."

        # Test with tolerance
        assert np.isnan(idx(arr, 3.1, tol=0.05)), "Index should be NaN when outside tolerance."
        assert idx(arr, 3.1, tol=0.2) == 2, "Index within tolerance is incorrect."

        # Edge cases
        assert idx(arr, 1) == 0, "Index for exact match is incorrect."
        assert idx(arr, 5) == 4, "Index for exact match is incorrect."
        assert np.isnan(idx(arr, 0, tol=0.5)), "Index should be NaN when value is out of range with tolerance."


if __name__ == '__main__':
    unittest.main()
