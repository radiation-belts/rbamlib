import unittest
import warnings
import numpy as np
from rbamlib.models.dip import B, B0, T, Y, al_lc

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

    def test_B0_other_planets(self):
        """Test B0 function for Jupiter and Saturn planets"""
        # Test Jupiter
        # B0_Jupiter = 4.28 G (from constants.py)
        # At r=1: B = 4.28 / 1^3 = 4.28 G
        # At r=2: B = 4.28 / 2^3 = 0.535 G
        result_jupiter_r1 = B0(1.0, planet='Jupiter')
        np.testing.assert_almost_equal(result_jupiter_r1, 4.28, decimal=4,
                                       err_msg="B0(1, Jupiter) should return 4.28 G")

        result_jupiter_r2 = B0(2.0, planet='Jupiter')
        np.testing.assert_almost_equal(result_jupiter_r2, 0.535, decimal=4,
                                       err_msg="B0(2, Jupiter) should return 0.535 G")

        # Test Saturn
        # B0_Saturn = 0.215 G (from constants.py)
        # At r=1: B = 0.215 / 1^3 = 0.215 G
        # At r=2: B = 0.215 / 2^3 = 0.0269 G (rounded to 0.0269)
        result_saturn_r1 = B0(1.0, planet='Saturn')
        np.testing.assert_almost_equal(result_saturn_r1, 0.215, decimal=4,
                                       err_msg="B0(1, Saturn) should return 0.215 G")

        result_saturn_r2 = B0(2.0, planet='Saturn')
        np.testing.assert_almost_equal(result_saturn_r2, 0.026875, decimal=4,
                                       err_msg="B0(2, Saturn) should return 0.0269 G")

    def test_B_other_planets(self):
        """Test B function for Jupiter and Saturn with magnetic latitude"""
        # Test Jupiter with mlat=0 (should match B0)
        result_jupiter_mlat0 = B(2.0, mlat=0, planet='Jupiter')
        expected_jupiter_mlat0 = 4.28 / 2**3  # 0.535 G
        np.testing.assert_almost_equal(result_jupiter_mlat0, expected_jupiter_mlat0, decimal=4,
                                       err_msg="B(2, 0, Jupiter) should match B0(2, Jupiter)")

        # Test Jupiter with non-zero mlat
        # B = B0 / r^3 * sqrt(1 + 3*sin(mlat)^2)
        # For r=2, mlat=0.1 rad: B = 4.28 / 8 * sqrt(1 + 3*sin(0.1)^2) = 0.535 * 1.0149 ≈ 0.543
        result_jupiter_mlat = B(2.0, mlat=0.1, planet='Jupiter')
        expected_jupiter_mlat = 4.28 / 8 * np.sqrt(1 + 3 * np.sin(0.1)**2)
        np.testing.assert_almost_equal(result_jupiter_mlat, expected_jupiter_mlat, decimal=4,
                                       err_msg="B(2, 0.1, Jupiter) calculation incorrect")

        # Test Saturn with mlat=0 (should match B0)
        result_saturn_mlat0 = B(2.0, mlat=0, planet='Saturn')
        expected_saturn_mlat0 = 0.215 / 2**3  # 0.026875 G
        np.testing.assert_almost_equal(result_saturn_mlat0, expected_saturn_mlat0, decimal=4,
                                       err_msg="B(2, 0, Saturn) should match B0(2, Saturn)")

        # Test Saturn with non-zero mlat
        result_saturn_mlat = B(3.0, mlat=0.2, planet='Saturn')
        expected_saturn_mlat = 0.215 / 27 * np.sqrt(1 + 3 * np.sin(0.2)**2)
        np.testing.assert_almost_equal(result_saturn_mlat, expected_saturn_mlat, decimal=4,
                                       err_msg="B(3, 0.2, Saturn) calculation incorrect")

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

    def test_al_lc_single_value(self):
        """Single L-shell value returns correct angle (float)."""
        res = al_lc(4.0)  # default planet='Earth'
        self.assertIsInstance(res, float, "Result should be float for scalar input")
        self.assertAlmostEqual(res, np.deg2rad(5.341843503512352), places=6,
                               msg="Single value result mismatch")

    def test_al_lc_array_values(self):
        """Vectorized L-shell input returns correct ndarray of angles."""
        res = al_lc(np.array([2.0, 3.0, 4.0, 6.0]))
        self.assertIsInstance(res, np.ndarray, "Result should be ndarray for array input")
        np.testing.assert_allclose(res, np.deg2rad(np.array([16.33008612,  8.40853781,  5.34184366,  2.85139895])),
                                   rtol=0, atol=1e-6,
                                   err_msg="Array values result mismatch")

if __name__ == '__main__':
    unittest.main()
