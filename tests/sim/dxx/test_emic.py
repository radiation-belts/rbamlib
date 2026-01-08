import unittest
from tests.helpers import TestHelpers
from rbamlib.sim.dxx.emic import D2017


class TestEmic(unittest.TestCase, TestHelpers):

    def test_D2017(self):
        self.AssertBlank(D2017)

if __name__ == '__main__':
    unittest.main()
