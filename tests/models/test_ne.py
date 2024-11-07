import unittest
from tests.helpers import TestHelpers
from rbamlib.models.ne import D2012


class TestNe(unittest.TestCase, TestHelpers):

    def test_D2012(self):
        self.AssertBlank(D2012)


if __name__ == '__main__':
    unittest.main()
