import unittest
import numpy as np
from rbamlib.models.lpp import CA1992


class TestModelsLpp(unittest.TestCase):
    def test_CA1992_values(self):
        """Test the CA1992 function with known values."""
        time = np.array([0, 1, 2])  # Example time in days
        kp = np.array([1, 2, 3])  # Example Kp-index values
        expected_output = np.array([5.14, 4.68, 4.22])  # Expected plasmapause locations

        # Call the CA1992 function
        result = CA1992(time, kp)

        # Assert that the result is as expected
        np.testing.assert_almost_equal(result, expected_output, decimal=2,
                                       err_msg="CA1992 did not return expected values.")


if __name__ == '__main__':
    unittest.main()
