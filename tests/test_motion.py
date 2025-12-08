import unittest
import numpy as np
from rbamlib.models.dip import B
from rbamlib.motion import f_gyro, omega_gyro, T_gyro
from rbamlib.motion import f_bounce, omega_bounce, T_bounce
from rbamlib.motion import f_drift, omega_drift, T_drift


class TestUtils(unittest.TestCase):

    def setUp(self):
        """Set up common test variables."""
        # 1 / 1e10 - is a scaling  for testing. Not real units!!
        self.B = np.array([1e4, 0.5e4]) / 1e10 # 1 and 0.5 T in Gauss 
        self.f = np.array([2.8e10, 2.8e10/2]) / 1e10 # Hz
        self.T_gyro = np.array([0.35e-10, 0.71e-10]) * 1e10 # s
        self.en = np.array([0.01, 0.1, 1])
        # 27990582811, 27990582811/2

        self.arr = np.array([1, 2, 3, 4, 5])

    def tearDown(self):
        """Clean up after tests if necessary."""
        pass

    def test_f_gyro(self):
        result = f_gyro(B=self.B)
        np.testing.assert_array_almost_equal(result, self.f, decimal=1, err_msg="test_f_gyro failes")

    def test_omega_gyro(self):
        result = omega_gyro(B=self.B)
        np.testing.assert_array_almost_equal(result, 2 * np.pi * self.f , decimal=1, err_msg="test_omega_gyro failed")

    def test_T_gyro(self):
        result = T_gyro(B=self.B)
        np.testing.assert_array_almost_equal(result, self.T_gyro,
                                              decimal=1, err_msg="test_T_gyro failes")

        B0 = 0.3**2 # L=2 with B0=0.3
        result = T_gyro(B=B0, en=self.en)
        # Physics of Earth Radiaion Belts Table 2.2
        np.testing.assert_array_almost_equal(result, np.array([9.71e-6, 11.4e-6, 28.1e-6]),
                                              decimal=1, err_msg="test_T_gyro with energy failes")
        
    def test_T_gyro_L(self):        
        L = 2 # L=2 
        result = T_gyro(L=2, en=self.en)
        # Physics of Earth Radiaion Belts Table 2.2
        np.testing.assert_array_almost_equal(result, np.array([9.71e-6, 11.4e-6, 28.1e-6]),
                                              decimal=1, err_msg="test_T_gyro_L with energy failes")

    def test_f_bounce(self):
        result = f_bounce(L=4, al=np.deg2rad(80), en=self.en[0])
        np.testing.assert_array_almost_equal(result, 1/1.30,
                                              decimal=1, err_msg="test_f_bounce failes")
        
    def test_omega_bounce(self):
        result = omega_bounce(L=4, al=np.deg2rad(80), en=self.en[0])        
        np.testing.assert_array_almost_equal(result, 2 * np.pi / 1.30,
                                              decimal=1, err_msg="test_omega_bounce failes")

    def test_T_bounce(self):
        result = T_bounce(L=4, al=np.deg2rad(80), en=self.en)
        # Physics of Earth Radiaion Belts Table 2.2
        # The value for 0.01 keV is likely incorrect, it is calculated without relativistic assumtion
        np.testing.assert_array_almost_equal(result, np.array([1.30, 0.46, 0.27]),
                                              decimal=2, err_msg="test_T_gyro failes")

    def test_T_drift(self):
        result = T_drift(L=4, al=np.pi/2, en=np.array([0.001, 0.01])) / 60 / 60 # Convert to h
        # Quantitatve aspectes of Magnetrophseric Phyusics
        # Lyons and Williams, table 2.2 (only approximate non relativistic case)
        np.testing.assert_array_almost_equal(result, np.array([184, 18.4]),
                                              decimal=0, err_msg="test_T_drift failes")

    def test_omega_drift(self):
        result = omega_drift(L=4, al=np.pi/2, en=0.01)
        np.testing.assert_array_almost_equal(result, 1 / (18.4 * 60 * 60) / (2 * np.pi),
                                              decimal=0, err_msg="test_T_drift failes")

    def test_f_drift(self):
        result = f_drift(L=4, al=np.pi/2, en=0.01)
        np.testing.assert_array_almost_equal(result, 1 / (18.4 * 60 * 60),
                                              decimal=0, err_msg="test_T_drift failes")        

if __name__ == '__main__':
    unittest.main()
