import unittest
from tests.helpers import TestHelpers
from rbamlib.models.dll import BA2000, O2014, A2016, L2016


class TestDll(unittest.TestCase, TestHelpers):

    def test_BA2000(self):
        self.AssertBlank(BA2000)

    def test_O2014(self):
        self.AssertBlank(O2014)

    def test_L2016(self):
        self.AssertBlank(L2016)

    def test_A2016(self):
        self.AssertBlank(A2016)


if __name__ == '__main__':
    unittest.main()
