import unittest
from tests.helpers import TestHelpers
from rbamlib.models.dxx.vlf import S2011


class TestVlf(unittest.TestCase, TestHelpers):

    def test_S2011(self):
        self.AssertBlank(S2011)


if __name__ == '__main__':
    unittest.main()
