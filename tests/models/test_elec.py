import unittest
from tests.helpers import TestHelpers
from rbamlib.models.elec import VS1975


class TestElec(unittest.TestCase, TestHelpers):

    def test_VS1975(self):
        self.AssertBlank(VS1975)


if __name__ == '__main__':
    unittest.main()
