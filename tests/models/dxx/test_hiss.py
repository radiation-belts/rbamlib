import unittest
from tests.helpers import TestHelpers
from rbamlib.models.dxx.hiss import S2011, Z2019, S2015


class TestHiss(unittest.TestCase, TestHelpers):

    def test_S2011(self):
        self.AssertBlank(S2011)

    def test_Z2019(self):
        self.AssertBlank(Z2019)

    def test_S2015(self):
        self.AssertBlank(S2015)


if __name__ == '__main__':
    unittest.main()
