import unittest
import numpy as np
import warnings
import os
import tempfile
import shutil
import scipy.io
from tests.helpers import TestHelpers
from rbamlib.models.tau.chorus import W2024, G2012
from rbamlib.models.tau.chorus.G2012 import tau_day_1_10keV, tau_day_10_100keV, tau_day_100_400keV, tau_day_400_2000keV, tau_night_1_10keV, tau_night_10_100keV

if 'RUN_LIVE_TESTS' not in os.environ:
    os.environ['RUN_LIVE_TESTS'] = 'False'


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
        
        
class TestChorusW2024(unittest.TestCase, TestHelpers):
    """Unit tests for Wang et al. (2024) chorus lifetime model (W2024)."""

    @staticmethod
    def _create_synthetic_data_file(filepath):
        """Create a synthetic W2024 data file for testing."""
        # Generate grid of points covering key parameter ranges
        kp_vals = [1, 2, 3, 4, 5]
        mlt_vals = [0, 6, 12, 18]
        L_vals = np.linspace(3, 7, 5)
        log10_E_vals = np.linspace(-3, 0.28, 5)

        # Create all combinations
        points = []
        for kp in kp_vals:
            for mlt in mlt_vals:
                for L in L_vals:
                    for log10_E in log10_E_vals:
                        # Synthetic lifetime calculation
                        # Formula: log10_tau = base + L_coeff*L + Kp_coeff*Kp + E_coeff*log10_E
                        log10_tau_albert = 5.0 + 2.0*L + 0.5*kp + 1.0*log10_E
                        log10_tau_lc = 4.0 + 1.8*L + 0.4*kp + 0.9*log10_E

                        # Columns: [Kp, MLT, col2, col3, L, log10_E, log10_tau_albert, log10_tau_lc]
                        points.append([kp, mlt, 0.0, 0.0, L, log10_E, log10_tau_albert, log10_tau_lc])

        data_array = np.array(points)
        scipy.io.savemat(filepath, {'life_time_all_MLT': data_array})

    @classmethod
    def setUpClass(cls):
        """Create a temporary directory and synthetic data file for testing."""
        # Create temporary directory (shared for synthetic and downloaded data)
        cls.temp_dir = tempfile.mkdtemp()

        # Create synthetic data file in temp directory
        cls.synthetic_data_file = os.path.join(cls.temp_dir, 'synthetic_W2024_data.mat')
        cls._create_synthetic_data_file(cls.synthetic_data_file)

    @classmethod
    def tearDownClass(cls):
        """Clean up the temporary directory and all files (synthetic + downloaded)."""
        if hasattr(cls, 'temp_dir') and os.path.isdir(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_scalar_inputs(self):
        """Test scalar inputs using synthetic data file"""
        tau = W2024(L=5.0, en=1.0, kp=3.0, mlt=12.0, data_file=self.synthetic_data_file)
        self.assertTrue(np.isscalar(tau))
        self.assertTrue(np.isfinite(tau) and tau > 0)

    def test_array_inputs(self):
        """Test array broadcasting using synthetic data file"""
        L = np.array([4.0, 5.0, 6.0])
        tau = W2024(L, en=1.0, kp=3.0, mlt=12.0, data_file=self.synthetic_data_file)
        self.assertEqual(tau.shape, L.shape)
        self.assertTrue(np.all(np.isfinite(tau) & (tau > 0)))

    def test_method_parameter(self):
        """Test different methods using synthetic data file"""
        tau_albert = W2024(L=5.0, en=1.0, kp=3.0, mlt=12.0, method='albert', data_file=self.synthetic_data_file)
        tau_lc = W2024(L=5.0, en=1.0, kp=3.0, mlt=12.0, method='lc', data_file=self.synthetic_data_file)
        # Methods should give different but similar results
        self.assertNotEqual(tau_albert, tau_lc)
        self.assertTrue(abs(tau_albert - tau_lc) / tau_albert < 2.0)  # Within factor of 2

    def test_invalid_method(self):
        """Test invalid method raises error"""
        with self.assertRaises(ValueError):
            W2024(L=5.0, en=1.0, kp=3.0, mlt=12.0, method='invalid')

    def test_kp_warning(self):
        """Test Kp > 6 issues warning using synthetic data file"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            W2024(L=5.0, en=1.0, kp=7.0, mlt=12.0, data_file=self.synthetic_data_file)
            self.assertTrue(any("Kp > 6" in str(warning.message) for warning in w))

    def test_synthetic_data_correctness(self):
        """Test that synthetic data follows the expected formulas"""
        # Test Kp dependence: lifetime should increase with increasing Kp (simplified for testing)
        tau_kp2 = W2024(5.0, 1.0, 2.0, 12.0, method='albert', data_file=self.synthetic_data_file)
        tau_kp5 = W2024(5.0, 1.0, 5.0, 12.0, method='albert', data_file=self.synthetic_data_file)
        self.assertGreater(tau_kp5, tau_kp2,
                          "Synthetic: Lifetime should increase with increasing Kp (formula has +0.5*kp)")

        # Test L dependence: lifetime should increase with L
        tau_L4 = W2024(4.0, 0.5, 3.0, 12.0, method='albert', data_file=self.synthetic_data_file)
        tau_L6 = W2024(6.0, 0.5, 3.0, 12.0, method='albert', data_file=self.synthetic_data_file)
        self.assertGreater(tau_L6, tau_L4,
                          "Synthetic: Lifetime should increase with L-shell (formula has +2.0*L)")

        # Test energy dependence: lifetime should increase with energy
        tau_en_low = W2024(5.0, 0.1, 3.0, 12.0, method='albert', data_file=self.synthetic_data_file)
        tau_en_high = W2024(5.0, 1.0, 3.0, 12.0, method='albert', data_file=self.synthetic_data_file)
        self.assertGreater(tau_en_high, tau_en_low,
                          "Synthetic: Lifetime should increase with energy (formula has +1.0*log10_E)")

        # Test method differences: albert should give larger values than lc
        tau_albert = W2024(5.0, 1.0, 3.0, 12.0, method='albert', data_file=self.synthetic_data_file)
        tau_lc = W2024(5.0, 1.0, 3.0, 12.0, method='lc', data_file=self.synthetic_data_file)
        self.assertGreater(tau_albert, tau_lc,
                          "Synthetic: Albert method should give larger values (5.0 base vs 4.0)")

    @unittest.skipUnless(os.getenv('RUN_LIVE_TESTS', 'False').lower() == 'true',
                         "Skipping live test unless enabled.")
    def test_download_data_live(self):
        """Test that W2024 can find/use data with auto_download enabled."""
        # This test verifies the auto_download mechanism works
        # If data exists in working directory or temp directory, it should be found
        # If not, it should be downloaded
        tau = W2024(L=5.0, en=1.0, kp=3.0, mlt=12.0,
                   data_folder=self.temp_dir, auto_download=True)

        # Verify tau is valid (proves data was found/downloaded and used)
        self.assertTrue(np.isfinite(tau) and tau > 0,
                       "W2024 with auto_download should produce valid lifetime")

        # Verify that W2024 works correctly with the available data
        tau2 = W2024(L=6.0, en=0.5, kp=4.0, mlt=6.0,
                    data_folder=self.temp_dir, auto_download=True)
        self.assertTrue(np.isfinite(tau2) and tau2 > 0,
                       "W2024 should work for multiple calls with auto_download")

    @unittest.skipUnless(os.getenv('RUN_LIVE_TESTS', 'False').lower() == 'true',
                         "Skipping live test unless enabled.")
    def test_data_points_live(self):
        """Test W2024 model output against known reference values using real data."""
        # Use en=50keV (0.05 MeV) and kp=5 for all tests
        en = 0.05  # 50 keV
        kp = 5

        # Ensure real data is downloaded (only once, first call)
        W2024(5.0, en, kp, 12.0, method='albert',
              data_folder=self.temp_dir, auto_download=True)

        # Test point 1: At mlt=12, tau should decrease with L for both models
        tau_mlt12_L4_albert = W2024(4.0, en, kp, 12.0, method='albert', data_folder=self.temp_dir)
        tau_mlt12_L6_albert = W2024(6.0, en, kp, 12.0, method='albert', data_folder=self.temp_dir)
        self.assertGreater(tau_mlt12_L4_albert, tau_mlt12_L6_albert,
                          "Real data: At MLT=12, tau should decrease with increasing L (albert)")

        tau_mlt12_L4_lc = W2024(4.0, en, kp, 12.0, method='lc', data_folder=self.temp_dir)
        tau_mlt12_L6_lc = W2024(6.0, en, kp, 12.0, method='lc', data_folder=self.temp_dir)
        self.assertGreater(tau_mlt12_L4_lc, tau_mlt12_L6_lc,
                          "Real data: At MLT=12, tau should decrease with increasing L (lc)")

        # Test point 2: At mlt=6, L=5, lc should be smaller than albert
        tau_mlt6_L5_albert = W2024(5.0, en, kp, 6.0, method='albert', data_folder=self.temp_dir)
        tau_mlt6_L5_lc = W2024(5.0, en, kp, 6.0, method='lc', data_folder=self.temp_dir)
        self.assertGreater(tau_mlt6_L5_albert, tau_mlt6_L5_lc,
                          "Real data: At MLT=6, L=5, albert should be greater than lc")


if __name__ == '__main__':
    unittest.main()
