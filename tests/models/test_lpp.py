import unittest
import numpy as np
from rbamlib.models.lpp import CA1992
from rbamlib.models.lpp import OBM2003


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

    def test_ORM2003_values(self):
        """Test the OBM2003 function with known values for different index types."""

        # Test case for 'Kp' index type
        time = np.array([0, 1, 2])  # Example time in days
        kp = np.array([1, 2, 3])  # Example Kp-index values
        expected_output_kp = np.array([np.nan, 5.47, 5.04])  # Expected plasmapause locations for Kp

        result_kp = OBM2003(time, kp, 'kp')

        # First, check for NaN in the first index if expected
        self.assertTrue(np.isnan(result_kp[0]), "First value should be NaN due to lack of data in previous 24 hours.")

        # Check remaining values excluding NaN (from the second index onwards)
        np.testing.assert_almost_equal(result_kp[1:], expected_output_kp[1:], decimal=2,
                                       err_msg="ORM2003 did not return expected values for Kp index.")

        # Test case for 'Ae' index type
        ae = np.array([100, 200, 300])  # Example Ae-index values
        expected_output_ae = np.array([6.6800, 5.8191, 5.3154])  # Expected plasmapause locations for Ae

        result_ae = OBM2003(time, ae, 'ae')
        np.testing.assert_almost_equal(result_ae, expected_output_ae, decimal=2,
                                       err_msg="OBM2003 did not return expected values for Ae index.")

        # Test case for 'Dst' index type
        dst = np.array([-10, -20, -30])  # Example Dst-index values
        expected_output_dst = np.array([4.7300, 4.2574, 3.9809])  # Expected plasmapause locations for Dst

        result_dst = OBM2003(time, dst, 'dst')
        np.testing.assert_almost_equal(result_dst, expected_output_dst, decimal=2,
                                       err_msg="OBM2003 did not return expected values for Dst index.")


if __name__ == '__main__':
    unittest.main()
