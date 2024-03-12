import unittest
import rbamlib.constants as c

class TestConstants(unittest.TestCase):
    def test01_content(self):
        """ Test if module constant include variable MC2"""
        self.assertTrue(hasattr(c, 'MC2'))

    def test02_MC2(self):
        """ Test MC2"""
        self.assertAlmostEqual(c.MC2, 0.511, 3)


if __name__ == '__main__':
    unittest.main()
