import unittest
import numpy as np
from tests.helpers import TestHelpers
from rbamlib.models.dll import BA2000, O2014, A2016, L2016


class TestDll(unittest.TestCase, TestHelpers):

    def setUp(self):
        """Set up test parameters."""
        self.L = np.array([4.0, 5.0, 6.0])  # Example L values
        self.kp = np.array([2, 2, 2])  # Example Kp values
        self.mu = np.array([500, 500, 500])  # Example mu values (MeV/G)

    def test_BA2000_values(self):
        """Test BA2000 values for all diffusion coefficient types."""
        # Test electromagnetic diffusion coefficient
        dllm = BA2000(self.L, self.kp, dll_type='M')
        expected_dllm = np.array([0.00510035, 0.0475007, 0.29411184]) # From matlab
        np.testing.assert_almost_equal(dllm, expected_dllm, decimal=3,
                                       err_msg="Electromagnetic diffusion coefficients are incorrect.")

        # Test electrostatic diffusion coefficient
        dlle = BA2000(self.L, self.kp, mu=self.mu, dll_type='E')
        expected_dlle = np.array([8.4195e-03, 4.3609e-02, 1.7471e-01]) # From matlab
        np.testing.assert_almost_equal(dlle, expected_dlle, decimal=3,
                                       err_msg="Electromagnetic diffusion coefficients are incorrect.")

        # Test both diffusion coefficients
        dllm_both, dlle_both = BA2000(self.L, self.kp, mu=self.mu, dll_type='ME')
        np.testing.assert_almost_equal(dllm_both, expected_dllm, decimal=3,
                                       err_msg="Electromagnetic coefficients from 'ME' type are incorrect.")
        np.testing.assert_almost_equal(dlle_both, expected_dlle, decimal=3,
                                       err_msg="Electrostatic coefficients from 'ME' type are incorrect.")

        # Test error for missing mu
        with self.assertRaises(ValueError, msg="Missing 'mu' did not raise ValueError for electrostatic diffusion."):
            BA2000(self.L, self.kp, dll_type='E')

    def test_O2014(self):
        self.AssertBlank(O2014)

    def test_L2016(self):
        self.AssertBlank(L2016)

    def test_A2016(self):
        self.AssertBlank(A2016)


if __name__ == '__main__':
    unittest.main()
