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
                                       err_msg="Electrostatic diffusion coefficients are incorrect.")

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
        """Test O2014 values for all diffusion coefficient types."""

        # Test electromagnetic diffusion coefficient
        dllb = O2014(self.L, self.kp, dll_type='B')
        expected_dllb = np.array([0.000037062773886,   0.000473061957052,   0.003746864541156])  # From matlab
        np.testing.assert_almost_equal(dllb, expected_dllb, decimal=3,
                                       err_msg="Magnetic diffusion coefficients are incorrect.")

        # Test electrostatic diffusion coefficient
        dlle = O2014(self.L, self.kp, dll_type='E')
        expected_dlle = np.array([0.005455237955669,  0.034298393384386,  0.168795651219601])  # From matlab
        np.testing.assert_almost_equal(dlle, expected_dlle, decimal=3,
                                       err_msg="Electric diffusion coefficients are incorrect.")

        # Test both diffusion coefficients
        dllb_both, dlle_both = O2014(self.L, self.kp,  dll_type='BE')
        np.testing.assert_almost_equal(dllb_both, expected_dllb, decimal=3,
                                       err_msg="Magnetic coefficients from 'BE' type are incorrect.")
        np.testing.assert_almost_equal(dlle_both, expected_dlle, decimal=3,
                                       err_msg="Electric coefficients from 'BE' type are incorrect.")


    def test_L2016(self):
        def test_L2016(self):
            """Test L2016 values for electric diffusion coefficient."""

            # Test electrostatic diffusion coefficient
            dlle = L2016(self.L, self.kp, self.mu, dll_type='E')
            expected_dlle = np.array([0.007862489414963, 0.048828301802163, 0.217115554389266])  # From matlab
            np.testing.assert_almost_equal(dlle, expected_dlle, decimal=3,
                                           err_msg="Electric diffusion coefficients are incorrect.")

    def test_A2016(self):
        """Test A2016 values for all diffusion coefficient types."""

        # Test electromagnetic diffusion coefficient
        dllm = A2016(self.L, self.kp, dll_type='M')
        expected_dllm = np.array([0.028631587258395, 0.121815859719923, 0.518277367768120]) * 1.0e-03  # From matlab
        np.testing.assert_almost_equal(dllm, expected_dllm, decimal=3,
                                       err_msg="Magnetic diffusion coefficients are incorrect.")

        # Test electrostatic diffusion coefficient
        dlle = A2016(self.L, self.kp, dll_type='E')
        expected_dlle = np.array([0.000513120421795, 0.005348174453228, 0.055743191592555])  # From matlab
        np.testing.assert_almost_equal(dlle, expected_dlle, decimal=3,
                                       err_msg="Electric diffusion coefficients are incorrect.")

        # Test both diffusion coefficients
        dllm_both, dlle_both = A2016(self.L, self.kp, dll_type='ME')
        np.testing.assert_almost_equal(dllm_both, expected_dllm, decimal=3,
                                       err_msg="Magnetic coefficients from 'ME' type are incorrect.")
        np.testing.assert_almost_equal(dlle_both, expected_dlle, decimal=3,
                                       err_msg="Electric coefficients from 'ME' type are incorrect.")


if __name__ == '__main__':
    unittest.main()
