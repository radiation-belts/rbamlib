import unittest
import numpy as np
from tests.helpers import TestHelpers
from rbamlib.models.elec import VS1975



class TestElec(unittest.TestCase, TestHelpers):

    def test_VS1975(self):
        """Micro test
        """
        E = VS1975(1, phi=0, C=0, Omega=0)

        self.assertEqual(E['r'], 0)
        self.assertEqual(E['phi'], 0)

        E = VS1975(1, phi=np.array([0, 0]), C=0, Omega=0)

        np.testing.assert_equal(E['r'], np.array([0, 0]))
        np.testing.assert_equal(E['phi'], np.array([0, 0]))



if __name__ == '__main__':
    unittest.main()
