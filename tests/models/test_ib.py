import unittest
from tests.helpers import TestHelpers
from rbamlib.models.ib import W2024


class TestIb(unittest.TestCase, TestHelpers):

    def test_W2024(self):
        self.AssertBlank(W2024)


if __name__ == '__main__':
    unittest.main()
