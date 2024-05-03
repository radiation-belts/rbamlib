import unittest
import numpy as np
from rbamlib.models.dip import B, B0

class TestDip(unittest.TestCase):
    def test_B_values(self):
        """Test the dipole magnetic field."""
        r = np.array([1, 2, 3, 3])  # Example r
        mlat = np.array([0.1, 0.2, 0.3, 0])  # Example mlat, mlat =0 should redirect to B0
        expected_output = np.array([0.3166, 0.0412, 0.0130, 0.0116])  # Expected values

        # Call the B function
        result = B(r, mlat)

        # Assert that the result is as expected
        np.testing.assert_almost_equal(result, expected_output, decimal=4,
                                       err_msg="B did not return expected values.")

    def test_B0_values(self):
        """Test the dipole magnetic field at the equator."""
        r = np.array([1, 2, 3])  # Example r
        expected_output = np.array([0.312, 0.0390, 0.0116])  # Expected B0 locations

        # Call the B0 function
        result = B0(r)

        # Assert that the result is as expected
        np.testing.assert_almost_equal(result, expected_output, decimal=4,
                                       err_msg="B0 did not return expected values.")


if __name__ == '__main__':
    unittest.main()
