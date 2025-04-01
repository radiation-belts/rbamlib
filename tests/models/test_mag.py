import unittest
import numpy as np
from tests.helpers import TestHelpers
import datetime as dt
from rbamlib.models.mag import TS2005_S
from rbamlib.models.mag import TS2005_W


class TestNe(unittest.TestCase, TestHelpers):
    def test_TS2005_S(self):
        """Test TS2005_S source function computation."""
        Nsw = np.array([1.0, 5.0])
        Vsw = np.array([1.0, 400.0])
        Bz  = np.array([1.0, -5.0])

        # Expected Bs: [0.0, 5.0]
        S = TS2005_S(Nsw, Vsw, Bz)

        # First row should be all zeros since Bs[0] = 0
        self.assertTrue(np.allclose(S[0], 0.0), "First output row should be zero due to Bs = 0")

        # Second row should return all ones due to normalization
        self.assertTrue(np.allclose(S[1], 1.0), "Second output row should be ones for normalized input")

    def test_TS2005_S_exponents(self):
        """Test that hard-coded exponents apply correctly when inputs are powers of normalization constants."""
        lambda_ = np.array([0.39, 0.46, 0.39, 0.42, 0.41, 1.29])
        beta_   = np.array([0.80, 0.18, 2.32, 1.25, 1.60, 2.40])
        gamma_  = np.array([0.87, 0.67, 1.32, 1.29, 0.69, 0.53])

        S_expected = 8.0

        for k in range(6):
            Nsw = 5 * 2 ** (1 / lambda_[k])
            Vsw = 400 * 2 ** (1 / beta_[k])
            Bz = -5 * 2 ** (1 / gamma_[k])  # Ensure Bs > 0

            S = TS2005_S(Nsw, Vsw, Bz)
            self.assertAlmostEqual(S[0, k], S_expected, places=2,
                                   msg=f"S[0, {k}] should be {S_expected} when each term is scaled to 2")

    def test_TS2005_W_basic(self):
        """Test TS2005_W output for known constant S and time values."""
        # Create dummy time: hourly resolution
        time = [dt.datetime(2023, 1, 1, hour=h) for h in range(4)]

        # Set S such that all values are 1.0 for simplicity
        S = np.ones((4, 6))

        # Compute W with no storm_onsets (continuous integration)
        W = TS2005_W(time, S, storm_onsets=None)

        # Check shape
        self.assertEqual(W.shape, (4, 6), "W output shape should be (4, 6)")

        # Values should increase with time index (since integration window grows)
        for k in range(6):
            self.assertTrue(np.all(np.diff(W[:, k]) > 0), f"W[:, {k}] should increase with time")

    def test_TS2005_W_with_storm_onsets(self):
        """Test TS2005_W with a storm onset that resets integration."""
        time = [dt.datetime(2023, 1, 1, hour=h) for h in range(6)]
        S = np.ones((6, 6))
        storm_onsets = [0, 3]  # Two storms: [0-2], [3-5]

        W = TS2005_W(time, S, storm_onsets=storm_onsets, fill_value=-1)

        # Check that first segment is non-negative and increasing
        for k in range(6):
            self.assertTrue(np.all(W[:3, k] > 0), f"W first segment should be > 0 for k={k}")
            self.assertTrue(np.all(np.diff(W[:3, k]) > 0), f"W first segment increasing for k={k}")

        # Second segment should also be increasing from a reset
        for k in range(6):
            self.assertTrue(np.all(W[3:, k] > 0), f"W second segment should be > 0 for k={k}")
            self.assertTrue(np.all(np.diff(W[3:, k]) > 0), f"W second segment increasing for k={k}")

if __name__ == '__main__':
    unittest.main()
