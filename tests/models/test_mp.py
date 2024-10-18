import unittest
import numpy as np
from rbamlib.models.mp import S1998


class TestModelsMp(unittest.TestCase):
    def test_S1998_magnetopause_with_expected(self):
        """Test the S1998 function with specific Dp, Bz, P values and compare to expected results."""

        # Test case parameters
        Pdyn = np.array([1, 5, 50])  # Dynamic pressure in nPa
        Bz = np.array([-10, 0, 10])  # IMF Bz in nT
        Phy = np.radians([0, 30, 60, 90])  # Angles 0, 30, 60, 90 degrees in radians

        # Expected results for the combinations of Pdyn and Bz (for each angle)
        # Populate these with the actual expected values
        expected_results = {
            (1, -10): [10.000000,10.487723,12.184506,16.097091],
            (1, 0): [11.400000,11.872570,13.492548,17.109930],
            (1, 10): [11.530000,11.924161,13.255607,16.134988],
            (5, -10): [7.836019,8.233713,9.622806,12.853780],
            (5, 0): [8.933061,9.318345,10.643587,13.624720],
            (5, 10): [9.034929,9.356242,10.444650,12.812795],
            (50, -10): [5.528156,5.933282,7.413406,11.210653],
            (50, 0): [6.302098,6.693958,8.094348,11.518111],
            (50, 10): [6.373964,6.700243,7.840893,10.499079],
        }

        # Loop through all combinations of Pdyn and Bz
        for i, (pd, bz) in enumerate(zip(Pdyn, Bz)):
            # Call the S1998 function
            result = S1998(Phy, np.array([bz]), np.array([pd]))

            # Flatten the result to compare with 1D expected values
            result_flat = result.flatten()

            # Fetch expected values for this combination of Pdyn and Bz
            expected = expected_results[(pd, bz)]

            # Assert that the result matches the expected values (up to 2 decimal places)
            np.testing.assert_almost_equal(result_flat, expected, decimal=2,
                                           err_msg=f"Failed for Pdyn={pd}, Bz={bz}")


if __name__ == '__main__':
    unittest.main()
