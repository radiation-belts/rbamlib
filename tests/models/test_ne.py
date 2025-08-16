import unittest
from tests.helpers import TestHelpers
from numpy.testing import assert_allclose
import numpy as np
from rbamlib.models.ne import D2006
from rbamlib.models.ne import D2002
from rbamlib.models.ne import CA1992
from rbamlib.models.ne.CA1992 import _CA1992_ne_plasmasphere, _CA1992_ne_trough, _CA1992_ne_plasmapause, _CA1992_Lppo_solve
from rbamlib.models.ne import S2001
from rbamlib.models.ne.S2001 import _S2001_trough, _S2001_plasmasphere, _S2001_LT


class TestNe(unittest.TestCase, TestHelpers):

    def setUp(self):
        """Set up test parameters."""
        self.r = np.array([2.0, 3.0, 4.0])  # Geocentric distances (Re)
        self.ne_eq = np.array([100.0, 50.0, 10.0])  # Equatorial densities (cm^-3)
        self.L = np.array([4.0, 5.0, 6.0])  # McIlwain L-shell values
        self.Rmax = np.array([4.0, 5.0, 6.0])
        self.MLT = np.array([0.0, 6.0, 12.0, 15.0])
        self.Lpp, self.ne_Lpp = 4.0, 100.0
        self.kp = 2.0  # single Kp index

        self.LT_S2001 = 17.18

        # Precomputed expected results (double precision)
        self.expected_ne_L = np.array([263.90158215, 111.00553015,  29.88452790])
        self.expected_ne_given = np.array([400.0, 179.30478455, 33.75])
        self.expected_ne_Rmax = np.array([263.90158215, 111.00553015,  29.88452790])

        self.expected_ne_base = np.array([442.89420720, 214.68415894, 104.06387654], dtype=float)
        self.expected_ne_plasmapause = np.array([100.0, 1.0e-08, 1.0e-16], dtype=float)

        # expected (double precision), precomputed once
        self.expected_ne_trough = np.array([
            [11.50939425, 15.02501925, 31.43126925, 39.63439425],
            [ 4.40932395,  5.69729910, 11.70784982, 14.71312519],
            [ 2.15671706,  2.72372857,  5.36978231,  6.69280917]
        ], dtype=float)
        self.expected_ne_trough_col12 = np.array([31.43126925, 11.70784982, 5.36978231], dtype=float)
        self.expected_ne_pp = np.array([
            [1.00000000e+02, 1.00000000e+02, 1.00000000e+02, 1.00000000e+02],
            [1.00000000e-08, 1.00000000e-08, 5.62341325e-05, 5.45559478e-04],
            [1.00000000e-18, 1.00000000e-18, 3.16227766e-11, 2.97635144e-09]
        ], dtype=float)

        # Precomputed expected results
        self.expected_ps = np.array([1390.0 * (3.0 / 4.0) ** 4.83,
                                     1390.0 * (3.0 / 5.0) ** 4.83,
                                     1390.0 * (3.0 / 6.0) ** 4.83])
        self.expected_tr = np.array([[33.66927732, 27.31695897, 44.79947268, 51.596379],
                                     [11.56481927, 12.07296984, 20.57598073, 22.08293666],
                                     [4.96361821, 6.21342052, 10.53638179, 10.80679523]]
                                    )

        # Precomputed expected values (all from equations in Sheeley+2001)
        self.expected_LT_S2001 = 17.18

        # Plasmasphere densities for L=[4,5,6]
        self.expected_plasma_S2001 = np.array([346.386297, 117.892273,  48.869728])

        # Trough densities for L=[4,5,6], LT=17.18
        self.expected_trough_S2001 = np.array([52.063769, 20.935897,  9.844892])

        # Masked case for L outside [3,7]
        self.L_mask = np.array([2.5, 4.0, 6.0, 8.0])
        self.expected_masked_S2001 = np.array([np.nan, 346.386297, 48.869728, np.nan])

        # Branching case: plasmasphere for L<=4.5, trough for L>4.5
        self.expected_branch_S2001 = np.array([346.386297, 20.935897,  9.844892])

    # def test_D2006(self):
    #     self.AssertBlank(D2006)

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
        ne = D2002(self.r, self.ne_eq, Rmax=self.Rmax)
        np.testing.assert_almost_equal(
            ne, self.expected_ne_Rmax, decimal=6,
            err_msg="D2002 output with Rmax is incorrect."
        )

    def test_D2002_missing_L_and_Rmax(self):
        """Test D2002 raises ValueError if neither L nor Rmax is provided."""
        with self.assertRaises(ValueError, msg="Missing L and Rmax did not raise ValueError."):
            D2002(self.r, self.ne_eq)

    def test__CA1992_ne_plasmasphere_base(self):
        """_CA1992_ne_plasmasphere returns CA1992 base plasmasphere values."""
        ne = _CA1992_ne_plasmasphere(self.L)
        np.testing.assert_allclose(
            ne, self.expected_ne_base, rtol=1e-8, atol=0.0,
            err_msg="_CA1992_ne_plasmasphere base output does not match expected CA1992 relation."
        )

    def test__CA1992_ne_trough_vectorized(self):
        ne = _CA1992_ne_trough(self.L, self.MLT)
        np.testing.assert_almost_equal(ne, self.expected_ne_trough, decimal=8,
                                       err_msg="_CA1992_ne_trough (L×MLT) mismatch.")

    def test__CA1992_ne_trough_scalar_mlt(self):
        ne = _CA1992_ne_trough(self.L, 12.0)
        np.testing.assert_almost_equal(ne, self.expected_ne_trough_col12, decimal=8,
                                       err_msg="_CA1992_ne_trough (scalar MLT) mismatch.")

    def test__CA1992_ne_plasmapause_vectorized(self):
        ne = _CA1992_ne_plasmapause(self.L, self.Lpp, self.ne_Lpp, self.MLT)
        np.testing.assert_almost_equal(ne, self.expected_ne_pp, decimal=10,
                                       err_msg="_CA1992_ne_plasmapause (L×MLT) mismatch.")

    def test__CA1992_ne_plasmapause(self):
        """Precomputed plasmapause density profile."""
        ne = _CA1992_ne_plasmapause(L=self.L, Lpp = self.L[0], ne_Lpp = 100., MLT=self.MLT[0])
        np.testing.assert_almost_equal(
            ne, self.expected_ne_plasmapause, decimal=6,
            err_msg="_CA1992_ne_plasmapause output mismatch."
        )

    def test__CA1992_Lppo_solve_simple(self):
        """_CA1992_Lppo_solve finds a crossing with tiny grid."""
        Lpp = 5.0
        MLT = 12.0
        Lppo = _CA1992_Lppo_solve(Lpp, MLT, ngrid=3)
        self.assertIsInstance(Lppo, float, "Lppo must be float.")
        self.assertGreaterEqual(Lppo, Lpp, "Lppo must be >= Lpp.")
        self.assertLessEqual(Lppo, 8.0, "Lppo must be <= Lmax.")

    def test_CA1992_base_only(self):
        """CA1992 base plasmasphere, no mods."""
        ne = CA1992(self.L)
        np.testing.assert_almost_equal(
            ne, self.expected_ne_base, decimal=6,
            err_msg="CA1992 base plasmasphere is incorrect."
        )

    def test_CA1992_with_doy_and_R13(self):
        """CA1992 seasonal + solar terms."""
        ne_base = CA1992(L = 4.0)
        ne_mod = CA1992(L = 4.0, doy=100, R13=50)
        self.assertNotEqual(ne_base, ne_mod, "CA1992 did not apply doy/R13 terms.")


    def test_CA1992_with_Lpp_piecewise(self):
        """CA1992 with plasmapause, piecewise split."""
        ne = CA1992(self.L, Lpp=4.5, MLT=12.0, Lppo=False)
        self.assertEqual(ne.shape, self.L.shape, "CA1992 output shape mismatch.")
        self.assertGreater(ne[0], ne[-1], "Plasmasphere should exceed trough density.")


    def test_CA1992_with_Lpp_and_Lppo(self):
        """CA1992 with plasmapause + outer plasmapause solver."""
        L = np.arange(4, 5, 0.01)
        ne_noLppo = CA1992(L, Lpp=4.5, MLT=12.0, Lppo=False)
        ne_withLppo = CA1992(L, Lpp=4.5, MLT=12.0, Lppo=True)
        self.assertTrue(
            np.any(ne_noLppo != ne_withLppo),
            "Lppo=True must alter densities compared to simple piecewise."
        )

    def test__S2001_LT(self):
        result = _S2001_LT(self.kp)
        assert_allclose(result, self.expected_LT_S2001, rtol=1e-6)

    def test__S2001_plasmasphere(self):
        result = _S2001_plasmasphere(self.L)
        assert_allclose(result, self.expected_plasma_S2001, rtol=1e-6)

        # Also check main S2001 in plasmasphere mode (Lpp=None)
        result_main = S2001(self.L, Lpp=None)
        assert_allclose(result_main, self.expected_plasma_S2001, rtol=1e-6)

    def test__S2001_trough(self):
        result = _S2001_trough(self.L, self.expected_LT_S2001)
        assert_allclose(result, self.expected_trough_S2001, rtol=1e-6)

        # Also check main S2001 in trough mode (force Lpp < 4)
        result_main = S2001(self.L, Lpp=3.5, kp=self.kp)
        assert_allclose(result_main, self.expected_trough_S2001, rtol=1e-6)

    def test_S2001_mask_and_branch(self):
        # Masked case
        result_mask = S2001(self.L_mask, Lpp=None)
        assert_allclose(result_mask, self.expected_masked_S2001, rtol=1e-6, equal_nan=True)

        # Branching case: plasmasphere for L<=Lpp, trough for L>Lpp
        result_branch = S2001(self.L, Lpp=self.Lpp, kp=self.kp)
        assert_allclose(result_branch, self.expected_branch_S2001, rtol=1e-6)

if __name__ == '__main__':
    unittest.main()
