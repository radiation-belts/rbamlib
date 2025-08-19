import unittest
from tests.helpers import TestHelpers
from numpy.testing import assert_allclose
import numpy as np
from rbamlib.models.ne import D2002
from rbamlib.models.ne import CA1992
from rbamlib.models.ne.CA1992 import CA1992_plasmasphere, CA1992_trough, CA1992_plasmapause, _CA1992_Lppo_solve
from rbamlib.models.ne import S2001
from rbamlib.models.ne.S2001 import S2001_trough, S2001_plasmasphere, _S2001_LT

from rbamlib.models.ne import D2004
from rbamlib.models.ne.D2004 import D2004_plasmasphere, D2004_trough

from rbamlib.models.ne import D2006


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

        self.a_default = (3.78, -0.324, 3.77, -3.45) # (a1, a2, a3, a4)
        self.R13_default = 13.0
        self.R13_none = None

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

        # ---- Precomputed EXPECTED (all single-line arrays) ----
        # Plasmasphere, Eq. (5): log10 ne = a1 + a2 L + (0.00127 R13 - 0.0635) * exp(-(L-2)/1.5)
        # Using default a1,a2 and R13=None (i.e., omit sunspot term)
        self.expected_plasmasphere_D2004_R13_none = np.array(
            [304.78949895855994, 144.54397706628906, 68.54882264667571])

        # Plasmasphere with default R13=13.0
        self.expected_plasmasphere_D2004_R13_13 = np.array([296.2194494842024, 142.44283051418385, 68.03540472271488])

        # Plasmatrough, Eq. (11): ne = a3 * L ** a4, default a3,a4
        self.expected_trough_D2004_default = np.array([0.03156708217353685, 0.014618231911687338, 0.007793268998202528])

        # Branching (default R13=13.0): L<=Lpp -> plasmasphere, L>Lpp -> trough
        # For L = [4,5,6], Lpp=4.5 -> [PS(4), TR(5), TR(6)]
        self.expected_branch_D2004_default = np.array([296.2194494842024, 0.014618231911687338, 0.007793268998202528])

        # Branching with R13=None (omit sunspot term)
        self.expected_branch_D2004_R13_none = np.array([304.78949895855994, 0.014618231911687338, 0.007793268998202528])

        # Custom coefficients to verify override path
        self.a_custom = (3.70, -0.300, 4.10, -3.60)
        # Plasmasphere with custom (a1,a2) and R13=13.0
        self.expected_plasmasphere_D2004_custom = np.array([307.3360961592984, 156.1854578457947, 78.83788640535525])
        # Trough with custom (a3,a4)
        self.expected_trough_D2004_custom = np.array([0.02788481915669601, 0.01248796791022128, 0.006477981944714957])

        # ---- Precomputed expected values ----
        self.expected_equatorial_D2006 = np.array([
            [217.17008433, 122.12371302, 217.17008433, 326.26966370],  # L=4
            [158.05200153,  88.87917199, 158.05200153, 237.45247207],  # L=5
            [199.89411112, 112.40871935, 199.89411112, 300.31477223],  # L=6
        ])

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
        """CA1992_plasmasphere returns CA1992 base plasmasphere values."""
        ne = CA1992_plasmasphere(self.L)
        np.testing.assert_allclose(
            ne, self.expected_ne_base, rtol=1e-8, atol=0.0,
            err_msg="CA1992_plasmasphere base output does not match expected CA1992 relation."
        )

    def test__CA1992_ne_trough_vectorized(self):
        ne = CA1992_trough(self.L, self.MLT)
        np.testing.assert_almost_equal(ne, self.expected_ne_trough, decimal=8,
                                       err_msg="CA1992_trough (L×MLT) mismatch.")

    def test__CA1992_ne_trough_scalar_mlt(self):
        ne = CA1992_trough(self.L, 12.0)
        np.testing.assert_almost_equal(ne, self.expected_ne_trough_col12, decimal=8,
                                       err_msg="CA1992_trough (scalar MLT) mismatch.")

    def test__CA1992_ne_plasmapause_vectorized(self):
        ne = CA1992_plasmapause(self.L, self.Lpp, self.ne_Lpp, self.MLT)
        np.testing.assert_almost_equal(ne, self.expected_ne_pp, decimal=10,
                                       err_msg="CA1992_plasmapause (L×MLT) mismatch.")

    def test__CA1992_ne_plasmapause(self):
        """Precomputed plasmapause density profile."""
        ne = CA1992_plasmapause(L=self.L, Lpp = self.L[0], ne_Lpp = 100., MLT=self.MLT[0])
        np.testing.assert_almost_equal(
            ne, self.expected_ne_plasmapause, decimal=6,
            err_msg="CA1992_plasmapause output mismatch."
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
        result = S2001_plasmasphere(self.L)
        assert_allclose(result, self.expected_plasma_S2001, rtol=1e-6)

        # Also check main S2001 in plasmasphere mode (Lpp=None)
        result_main = S2001(self.L, Lpp=None)
        assert_allclose(result_main, self.expected_plasma_S2001, rtol=1e-6)

    def test__S2001_trough(self):
        result = S2001_trough(self.L, self.expected_LT_S2001)
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


    # ------------------ D2004 ------------------

    def test_D2004_plasmasphere_R13_none_helpers_and_main(self):
        """Eq. (5) without sunspot term (R13=None): helper == main."""
        a1, a2, _, _ = self.a_default
        result_helper = D2004_plasmasphere(self.L, a1, a2, self.R13_none)
        assert_allclose(result_helper, self.expected_plasmasphere_D2004_R13_none, rtol=1e-6)

        result_main = D2004(self.L, Lpp=None, R13=self.R13_none, a=self.a_default)
        assert_allclose(result_main, self.expected_plasmasphere_D2004_R13_none, rtol=1e-6)

    def test_D2004_plasmasphere_R13_default_helpers_and_main(self):
        """Eq. (5) with default sunspot term (R13=13.0): helper == main."""
        a1, a2, _, _ = self.a_default
        result_helper = D2004_plasmasphere(self.L, a1, a2, self.R13_default)
        assert_allclose(result_helper, self.expected_plasmasphere_D2004_R13_13, rtol=1e-6)

        result_main = D2004(self.L, Lpp=None, R13=self.R13_default, a=self.a_default)
        assert_allclose(result_main, self.expected_plasmasphere_D2004_R13_13, rtol=1e-6)

    def test_D2004_trough_helper(self):
        """Eq. (11) trough helper with default (a3,a4)."""
        _, _, a3, a4 = self.a_default
        result_helper = D2004_trough(self.L, a3, a4)
        assert_allclose(result_helper, self.expected_trough_D2004_default, rtol=1e-6)

    def test_D2004_branch_default(self):
        """Main D2004 with branching (R13=13.0, Lpp=4.5)."""
        result = D2004(self.L, Lpp=self.Lpp, R13=self.R13_default, a=self.a_default)
        assert_allclose(result, self.expected_branch_D2004_default, rtol=1e-6)

    def test_D2004_branch_no_R13(self):
        """Main D2004 with branching (R13=None, Lpp=4.5)."""
        result = D2004(self.L, Lpp=self.Lpp, R13=self.R13_none, a=self.a_default)
        assert_allclose(result, self.expected_branch_D2004_R13_none, rtol=1e-6)

    def test_D2004_custom_coeffs(self):
        """Custom coefficient override for both plasmasphere and trough."""
        a1, a2, a3, a4 = self.a_custom
        # Plasmasphere helper with custom (a1,a2) and R13=13
        res_ps = D2004_plasmasphere(self.L, a1, a2, self.R13_default)
        assert_allclose(res_ps, self.expected_plasmasphere_D2004_custom, rtol=1e-6)
        # Trough helper with custom (a3,a4)
        res_tr = D2004_trough(self.L, a3, a4)
        assert_allclose(res_tr, self.expected_trough_D2004_custom, rtol=1e-6)



    def test_D2006_equatorial_multiMLT(self):
        """Equatorial density for multiple MLT values."""
        result = D2006(self.L[:, None], MLT=self.MLT[None, :], r=None)
        assert_allclose(result, self.expected_equatorial_D2006, rtol=1e-3)

    def test_D2006_with_fieldline_extension(self):
        """Check D2006 + D2002 field-aligned extension (r=L → same as equatorial)."""
        # Broadcast L and MLT to grid
        result = D2006(self.L[:, None], MLT=self.MLT[None, :], r=self.L[:, None])
        assert_allclose(result, self.expected_equatorial_D2006, rtol=1e-3)

if __name__ == '__main__':
    unittest.main()
