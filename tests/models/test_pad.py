import unittest
from tests.helpers import TestHelpers
from rbamlib.models.pad import S2022


class TestPad(unittest.TestCase, TestHelpers):

    def test_S2022(self):
        self.AssertBlank(S2022)


if __name__ == '__main__':
    unittest.main()
