import unittest
import numpy as np
from tests.helpers import TestHelpers
from rbamlib.models.tau.chorus import W2024, G2012
from rbamlib.models.tau.chorus.G2012 import tau_day_1_10keV, tau_day_10_100keV, tau_day_100_400keV, tau_day_400_2000keV, tau_night_1_10keV, tau_night_10_100keV


class TestChorusG2012(unittest.TestCase, TestHelpers):
    """Unit tests for the Gu et al. (2012) chorus lifetime parameterization (G2012)."""

    def test_scalar_day_keV_bin_and_kp_scaling(self):
        """
        Scalar input on the DAYSIDE in the 1–10 keV bin:
        - lifetime should be finite and positive,
        - lifetime should DECREASE with increasing Kp (Eq. 15 scaling).
        """
        phi = 0.0            # MLT = 12 -> dayside
        L = 6.0
        en = 0.005           # 5 keV in MeV
        t_kp1 = G2012(phi, L, en, 1.0)
        t_kp5 = G2012(phi, L, en, 5.0)

        # Scalar checks
        self.assertTrue(np.isscalar(t_kp1), "Result should be scalar for scalar inputs")
        self.assertTrue(np.isfinite(t_kp1) and t_kp1 > 0.0, "Lifetime must be finite and positive")
        self.assertTrue(np.isfinite(t_kp5) and t_kp5 > 0.0, "Lifetime must be finite and positive")

        # Kp monotonicity: lifetimes decrease as Kp increases
        self.assertLess(t_kp5, t_kp1, "Lifetime should decrease with increasing Kp (dayside)")

    def test_array_bins_dayside(self):
        """
        Array input spanning the DAYSIDE bins:
        - 1–10 keV (0.005 MeV),
        - 10–100 keV (0.05 MeV),
        - 0.1–0.4 MeV (0.2 MeV),
        - 0.4–2.0 MeV (1.0 MeV).
        All should return finite values.
        """
        mlt = 12.0            # MLT = 12 -> dayside
        L = 4.0
        en = np.array([0.005, 0.05, 0.2, 1.0])  # MeV
        kp = 2.0
        t = G2012(mlt, L, en, kp)

        self.assertIsInstance(t, np.ndarray, "Array input should return ndarray")
        self.assertEqual(t.shape, en.shape, "Output shape should match input shape")
        self.assertTrue(np.all(np.isfinite(t) & (t > 0.0)), "All dayside bin outputs should be finite and positive")

    def test_nightside_nan_above_0p1MeV(self):
        """
        On the NIGHTSIDE, Gu (2012) parameterizations cover up to 0.1 MeV.
        Energies > 0.1 MeV should produce NaN (not parameterized).
        """
        mlt = 0         # MLT = 0 nightside
        L = 4.0
        en_valid = np.array([0.005, 0.05])      # <= 0.1 MeV (nightside bins)
        en_invalid = np.array([0.5, 1.0])       # > 0.1 MeV (not parameterized)
        kp = 2.0

        t_valid = G2012(mlt, L, en_valid, kp)
        t_invalid = G2012(mlt, L, en_invalid, kp)

        self.assertTrue(np.all(np.isfinite(t_valid) & (t_valid > 0.0)),
                        "Nightside lifetimes <= 0.1 MeV must be finite and positive")
        self.assertTrue(np.all(np.isnan(t_invalid)),
                        "Nightside lifetimes > 0.1 MeV must be NaN (not parameterized)")

    def test_kp_scaling_monotonic(self):
        """
        Confirm monotonic decrease of lifetime with increasing Kp on DAYSIDE for a mid-energy point.
        """
        mlt = 12.0            # dayside
        L = 6.0
        en = 0.2             # MeV (0.1–0.4 MeV bin)
        t1 = G2012(mlt, L, en, 1.0)
        t3 = G2012(mlt, L, en, 3.0)
        t5 = G2012(mlt, L, en, 5.0)

        self.assertTrue(t1 > t3 > t5,
                        "Expected t(Kp=1) > t(Kp=3) > t(Kp=5) due to Kp scaling (Eq. 15)")

    def test_broadcasting_shapes(self):
        """
        Broadcasting consistency across arrays for phi, L, and en on DAYSIDE.
        """
        mlt = np.array([12.0, 12.0, 12.0])     # dayside
        L = np.array([4.0, 5.0, 6.0])
        en = np.array([0.005, 0.05, 0.2])   # MeV (three different bins)
        kp = 2.0

        t = G2012(mlt, L, en, kp)
        self.assertEqual(t.shape, en.shape, "Output should broadcast to input shape")
        self.assertTrue(np.all(np.isfinite(t) & (t > 0.0)), "Broadcasted outputs should be finite and positive")

    def test_G2012_functions(self):        
        """
        Test polynomials of G2012
        Based on Figure 1
        """
        self.assetBetween(np.log10(tau_night_1_10keV(4, 0.005)), -1.8, -1., "tau_night_1_10keV failed.")
        self.assetBetween(np.log10(tau_night_10_100keV(9, 0.09)), -1, 0, "tau_night_10_100keV failed.")
        self.assetBetween(np.log10(tau_day_1_10keV(4, 0.005)), -1.8, -1., "tau_day_1_10keV failed.")
        self.assetBetween(np.log10(tau_day_10_100keV(4, 0.05)), -1.8, -1.2, "tau_day_10_100keV failed.")
        self.assetBetween(np.log10(tau_day_100_400keV(4, 0.35)), -0.8, -0.2, "tau_day_100_400keV failed.")
        self.assetBetween(np.log10(tau_day_400_2000keV(4, 1.8)), -0.2, 0.4, "tau_day_400_2000keV failed.")
        
        
        
        



    def test_W2024(self):
        self.AssertBlank(W2024)


if __name__ == '__main__':
    unittest.main()
