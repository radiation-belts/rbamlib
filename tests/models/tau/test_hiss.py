import unittest
from tests.helpers import TestHelpers
from rbamlib.models.tau.hiss import O2016


class TestHiss(unittest.TestCase, TestHelpers):

    def test_O2016(self):
        self.AssertBlank(O2016)

if __name__ == '__main__':
    unittest.main()
