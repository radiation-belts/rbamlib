import unittest
import numpy as np
from tests.helpers import TestHelpers
from rbamlib.models.elec import VS1975
from rbamlib.models.elec.VS1975 import VS1975_Phi_conv, VS1975_Phi_corr, VS1975_E_corr, VS1975_E_conv, VS1975_parameters_conv, VS1975_parameters_corr


class TestElec(unittest.TestCase, TestHelpers):

    def test_micro_VS1975(self):
        """
            Micro test
        """
        E = VS1975(1, phi=0, C=0, Omega=0)

        self.assertEqual(E['r'], 0)
        self.assertEqual(E['phi'], 0)

        # # Test with an unknown planet, expecting an error
        with self.assertRaises(ValueError):
            VS1975(1, phi=0, planet='Unknown', C=0, Omega=0)

        E = VS1975(1, phi=np.array([0, 0]), C=0, Omega=0)

        np.testing.assert_equal(E['r'], np.array([0, 0]))
        np.testing.assert_equal(E['phi'], np.array([0, 0]))

    def test_VS1975_Phi_corr(self):

        Omega, B0, R0 = VS1975_parameters_corr()
        Phi = VS1975_Phi_corr(1, np.pi/2, Omega, B0, R0)

        # Potential value is estimated from. -94.4 is due to more acurate Earth parameters
        # Maynard, N. C., and A. J. Chen (1975), Isolated cold plasma regions: Observations and their relation to possible production mechanisms, J. Geophys. Res., 80(7), 1009–1013, doi:10.1029/JA080i007p01009. 
        self.assertAlmostEqual(Phi, -94.4e3, delta=1e3)

    def test_VS1975_Phi_conv_simple(self):
        
        Phi = VS1975_Phi_conv(r=1, phi=np.pi/2, theta=np.pi/2, gamma=1, C=1, r0=1)
        self.assertEqual(Phi, 1)

    # def test_VS1975(self):
    #     E = VS1975(r=1, phi=0)

    #     self.assertEqual(E['r'], 1.5e-3)
    #     self.assertEqual(E['phi'], 0)

    # def test_VS1975_E_corr(self):
    #     E = VS1975_E_corr(r=1, phi=0)

    #     self.assertEqual(E['r'], 1.5e-3)
    #     self.assertEqual(E['phi'], 0)

    # def test_VS1975_E_conv(self):
    #     E = VS1975_E_conv(r=5, phi=0)

    #     self.assertEqual(E['r'], 1.5e-3)
    #     self.assertEqual(E['phi'], 0)
    

if __name__ == '__main__':
    unittest.main()
