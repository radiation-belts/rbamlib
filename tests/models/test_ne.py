import unittest
from tests.helpers import TestHelpers
import numpy as np
from rbamlib.models.ne import D2012
from rbamlib.models.ne import D2002



class TestNe(unittest.TestCase, TestHelpers):

    def setUp(self):
        """Set up test parameters."""
        self.r = np.array([2.0, 3.0, 4.0])  # Geocentric distances (Re)
        self.ne_eq = np.array([100.0, 50.0, 10.0])  # Equatorial densities (cm^-3)
        self.L = np.array([4.0, 5.0, 6.0])  # McIlwain L-shell values

        # Precomputed expected results (double precision)
        self.expected_ne_L = np.array([263.90158215, 111.00553015,  29.88452790])
        self.expected_ne_given = np.array([400.0, 179.30478455, 33.75])
        self.expected_ne_Rmax = np.array([263.90158215, 111.00553015,  29.88452790])


    def test_D2012(self):
        self.AssertBlank(D2012)

    def test_D2002_computed_alpha_from_L(self):
        """D2002 with alpha computed from L (precomputed expected)."""
        ne = D2002(self.r, self.ne_eq, L=self.L)
        np.testing.assert_almost_equal(
            ne, self.expected_ne_L, decimal=6,
            err_msg="D2002 output with computed alpha (L) is incorrect."
        )

    def test_D2002_with_given_alpha(self):
        """D2002 with user-supplied alpha (precomputed expected)."""
        alpha = np.array([2.0, 2.5, 3.0])
        ne = D2002(self.r, self.ne_eq, L=self.L, alpha=alpha)
        np.testing.assert_almost_equal(
            ne, self.expected_ne_given, decimal=6,
            err_msg="D2002 output with provided alpha is incorrect."
        )

    def test_D2002_with_Rmax(self):
        """D2002 with Rmax provided (precomputed expected)."""
        Rmax = np.array([4.0, 5.0, 6.0])
        ne = D2002(self.r, self.ne_eq, Rmax=self.Rmax)
        np.testing.assert_almost_equal(
            ne, self.expected_ne_Rmax, decimal=6,
            err_msg="D2002 output with Rmax is incorrect."
        )

    def test_D2002_missing_L_and_Rmax(self):
        """Test D2002 raises ValueError if neither L nor Rmax is provided."""
        with self.assertRaises(ValueError, msg="Missing L and Rmax did not raise ValueError."):
            D2002(self.r, self.ne_eq)



if __name__ == '__main__':
    unittest.main()
