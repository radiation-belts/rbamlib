import unittest
import numpy as np
from rbamlib.models.tau import tau_lc
from rbamlib.motion.bounce import T_bounce
import rbamlib.models.dip as dip

class TestMotion(unittest.TestCase):
    def test_single_value(self):
        """Test taulc_Dip with single input values."""
        # Expected values
        expected_taulc = T_bounce(4.0, np.deg2rad(5), 1) / 4  # Actual expected
        result = tau_lc(4.0, np.deg2rad(5), 1)
        self.assertAlmostEqual(result, expected_taulc, places=6, msg=f"Single value {result} is incorrect")

    def test_array_input(self):
        """Test taulc_Dip with array input values."""
        L_array = np.array([4.0, 5.0])
        en_array = np.array([1.0, 2.0])
        al_array = np.radians([5, 2])
        
        # Expected values for the array case (example values)
        expected_array = T_bounce(L_array, al_array, en_array) / 4 # Actual expected

        result = tau_lc(L_array, al_array, en_array)
        np.testing.assert_almost_equal(result, expected_array, decimal=6,
                                       err_msg="Array input output incorrect")

    def test_nan_outside_loss_cone(self):
        """tau_lc should return NaN on/above the loss-cone boundary."""
        L = 4.0
        en = 1.0
        # Compute the canonical loss-cone threshold from the dip model
        alpha_lc = dip.al_lc(L)

        # Boundary: alpha == alpha_lc -> NaN
        res_boundary = tau_lc(L, alpha_lc, en)
        self.assertTrue(np.isnan(res_boundary),
                        "tau_lc should be NaN at the loss-cone boundary (alpha == alpha_lc)")

        # Outside: alpha slightly above alpha_lc -> NaN
        res_outside = tau_lc(L, alpha_lc + 1e-4, en)
        self.assertTrue(np.isnan(res_outside),
                        "tau_lc should be NaN for alpha >= alpha_lc (outside loss cone)")

        # Mixed array: first inside, second outside
        al_array = np.array([alpha_lc * 0.5, alpha_lc + 1e-3])
        res_array = tau_lc(L, al_array, en)
        self.assertFalse(np.isnan(res_array[0]),
                        "First entry (inside loss cone) should be finite")
        self.assertTrue(np.isnan(res_array[1]),
                        "Second entry (outside loss cone) should be NaN")



    def test_custom_al_lc(self):
        """tau_lc must honor a user-supplied loss-cone angle (al_lc)."""
        L = 4.0
        en = 1.0

        # Choose a custom loss-cone angle smaller than the model's to make the test decisive
        alpha_lc_model = dip.al_lc(L)
        alpha_lc_custom = alpha_lc_model * 0.5  # tighter loss cone

        # Inside relative to custom threshold -> finite
        alpha_inside = alpha_lc_custom * 0.8
        res_inside = tau_lc(L, alpha_inside, en, al_lc=alpha_lc_custom)
        # Expect finite: equals T_bounce/4 by design
        expected_inside = T_bounce(L, alpha_inside, en) / 4.0
        self.assertAlmostEqual(res_inside, expected_inside, places=6,
                            msg="With custom al_lc, inside value should match T_bounce/4")

        # Outside relative to custom threshold -> NaN
        alpha_outside = alpha_lc_custom + 1e-4
        res_outside = tau_lc(L, alpha_outside, en, al_lc=alpha_lc_custom)
        self.assertTrue(np.isnan(res_outside),
                        "With custom al_lc, alpha >= al_lc must yield NaN")


if __name__ == '__main__':
    unittest.main()