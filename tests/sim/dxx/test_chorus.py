import unittest
from tests.helpers import TestHelpers
from rbamlib.sim.dxx.chorus import S2011, Z2019


class TestChorus(unittest.TestCase, TestHelpers):

    def test_S2011(self):
        self.AssertBlank(S2011)

    def test_Z2019(self):
        self.AssertBlank(Z2019)


if __name__ == '__main__':
    unittest.main()
