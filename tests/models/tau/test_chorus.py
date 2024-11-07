import unittest
from tests.helpers import TestHelpers
from rbamlib.models.tau.chorus import W2024, G2012


class TestChorus(unittest.TestCase, TestHelpers):

    def test_G2012(self):
        self.AssertBlank(G2012)

    def test_W2024(self):
        self.AssertBlank(W2024)


if __name__ == '__main__':
    unittest.main()
