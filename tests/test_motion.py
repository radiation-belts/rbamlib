import unittest
import numpy as np
from rbamlib.models.dip import B
from rbamlib.motion import f_gyro, omega_gyro, T_gyro

class TestUtils(unittest.TestCase):

    def setUp(self):
        """Set up common test variables."""
        # 1 / 1e10 - is a scaling  for testing. Not real units!!
        self.B = np.array([1e4, 0.5e4]) / 1e10 # 1 and 0.5 T in Gauss 
        self.f = np.array([2.8e10, 2.8e10/2]) / 1e10 # Hz
        self.T_gyro = np.array([0.35e-10, 0.71e-10]) * 1e10 # s
        # 27990582811, 27990582811/2

        self.arr = np.array([1, 2, 3, 4, 5])

    def tearDown(self):
        """Clean up after tests if necessary."""
        pass

    def test_f_gyro(self):
        result = f_gyro(self.B)
        np.testing.assert_array_almost_equal(result, self.f, decimal=1, err_msg="test_f_gyro failes")

    def test_omega_gyro(self):
        result = omega_gyro(self.B)
        np.testing.assert_array_almost_equal(result, 2 * np.pi * self.f , decimal=1, err_msg="test_omega_gyro failed")

    def test_T_gyro(self):
        result = T_gyro(self.B)
        np.testing.assert_array_almost_equal(result, self.T_gyro,
                                              decimal=1, err_msg="test_T_gyro failes")

        B0 = 0.3**2 # L=2 with B0=0.3
        result = T_gyro(B0, en=np.array([0.01, 0.1, 1]))
        # Physics of Earth Radiaion Belts Table
        np.testing.assert_array_almost_equal(result, np.array([9.71e-6, 11.4e-6, 28.1e-6]),
                                              decimal=1, err_msg="test_T_gyro with energy failes")


if __name__ == '__main__':
    unittest.main()
