import unittest
from tests.helpers import TestHelpers
from rbamlib.models.dxx.lightning import S2011


class TestLightning(unittest.TestCase, TestHelpers):

    def test_S2011(self):
        self.AssertBlank(S2011)


if __name__ == '__main__':
    unittest.main()
