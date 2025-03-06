import unittest
import rbamlib.constants as c

class TestConstants(unittest.TestCase):
    def test01_content(self):
        """ Test if module constant include corresponding variables """
        self.assertTrue(hasattr(c, 'MC2'))
        self.assertTrue(hasattr(c, 'B0_Earth'))
        self.assertTrue(hasattr(c, 'B0_Saturn'))
        self.assertTrue(hasattr(c, 'B0_Jupiter'))
        self.assertTrue(hasattr(c, 'T0'))
        self.assertTrue(hasattr(c, 'T1'))
        self.assertTrue(hasattr(c, 'R_Earth'))
        self.assertTrue(hasattr(c, 'R_Saturn'))
        self.assertTrue(hasattr(c, 'R_Jupiter'))
        self.assertTrue(hasattr(c, 'c'))
        self.assertTrue(hasattr(c, 'q'))

    def test02_MC2(self):
        """ Test MC2"""
        self.assertAlmostEqual(c.MC2, 0.511, 3)

    def test03_B0_Earth(self):
        """ Test B0 Earth"""
        self.assertAlmostEqual(c.B0_Earth, 0.3, 1)

    def test03_B0_Saturn(self):
        """ Test B0 Earth"""
        self.assertAlmostEqual(c.B0_Saturn, 0.2, 1)

    def test03_B0_Jupiter(self):
        """ Test B0 Earth"""
        self.assertAlmostEqual(c.B0_Jupiter, 4.28, 2)

    def test04_T0(self):
        """ Test T0"""
        self.assertAlmostEqual(c.T0, 1.3802, 4)

    def test04_T1(self):
        """ Test T1"""
        self.assertAlmostEqual(c.T1, 0.7405, 4)

    def test05_R_Earth(self):
        """ Test R_Earth cm"""
        self.assertAlmostEqual(c.R_Earth, 12756/2*1e5, delta=1e6)

    def test05_R_Saturn(self):
        """ Test R_Saturn cm"""
        self.assertAlmostEqual(c.R_Saturn, 120536/2*1e5, delta=1e7)

    def test05_R_Jupiter(self):
        """ Test R_Jupiter cm"""
        self.assertAlmostEqual(c.R_Jupiter, 142984/2*1e5, delta=1e7)

    def test06_c(self):
        """ Test speed of light in cm/s"""
        self.assertAlmostEqual(c.c, 2.9e10, delta=1e9)

    def test06_q(self):
        """ Test charge in CGS"""
        self.assertAlmostEqual(c.q, 4.8e-10, delta=1e-10)


if __name__ == '__main__':
    unittest.main()
