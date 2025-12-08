import unittest
import warnings
import numpy as np
from rbamlib.models.dip import B, B0, T, Y

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
        expected_output = np.array([0.312, 0.0390, 0.0116])  # Expected B0

        # Call the B0 function
        result = B0(r)

        # Assert that the result is as expected
        np.testing.assert_almost_equal(result, expected_output, decimal=4,
                                       err_msg="B0 did not return expected values.")

    # TODO: Add tests for other planet
    # TODO: Add tests for B0 == constant. at r == 1

    def test_T_values(self):
        """Test the T. Based on table 1 from Schulz & Lanzerotti (1974)"""
        al = np.deg2rad([0, 5.34, 34.38, 90])  # Example alpha
        expected_output = np.array([1.380, 1.253, 0.959, 0.740])  # Expected T

        # Call the T function
        result = T(al)

        # Assert that the result is as expected
        np.testing.assert_almost_equal(result, expected_output, decimal=3,
                                       err_msg="T did not return expected values.")

    def test_Y_values(self):
        """
        Test the Y. Based on table 1 from Schulz & Lanzerotti (1974).
        """

        al = np.deg2rad([5.34, 34.38, 90])  # Example alpha
        expected_output = np.array([2.091, 0.756, 0.000])  # Expected Y

        # Call the Y function
        result = Y(al)

        # Assert that the result is as expected
        np.testing.assert_almost_equal(result, expected_output, decimal=3,
                                       err_msg="T did not return expected values.")

    def test_Y_0_values(self):
        """
        Test the Y function at alpha=0.
        Table 1 provides a value for α = 0 even though log(sin(α)) is undefined,
        with the relation Y(0) = 2T(0).
        This test checks that:
          - the output is not NaN,
          - the output is not 0,
          - the output satisfies the relationship Y(0)=2T(0).
        """
        # α=0 in radians.
        alpha_zero = [0, 0]
        
        # Call Y(0).
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = Y(alpha_zero)
            self.assertFalse(w, f"Warning '{w}' was triggered")
        
        # Verify that the output is not NaN.
        self.assertFalse(np.isnan(result).any(), "Y(0) returned NaN.")
        
        # Verify that the output is not (unexpectedly) 0.
        self.assertFalse(np.allclose(result, 0),
                         "Y(0) returned 0, but a nonzero value was expected.")

        # Verify that Y(0) satisfies the relationship Y(0) = 2T(0).
        np.testing.assert_allclose(Y(0), 2 * T(0), rtol=1e-3,
                                   err_msg="Y(0) did not equal 2*T(0).")

 

if __name__ == '__main__':
    unittest.main()
