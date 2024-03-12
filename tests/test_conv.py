import unittest
import numpy as np
from rbamlib.conv import en2pc, pc2en

class TestConv(unittest.TestCase):

    def setUp(self):
        # Input values
        self.in_float = 0.1
        self.in_1d = np.array([0.1, 1., 2.])
        self.in_2d = np.array([[0.01, 0.1], [1., 2.]])

        # Expected values
        self.res_pc_float = 0.3350
        self.res_pc_1d = np.array([0.3350, 1.4220, 2.4585])
        self.res_pc_2d = np.array([[0.1016, 0.3350], [1.4220, 2.4585]])

        self.res_en_float = 0.0097
        self.res_en_1d = np.array([0.0097, 0.6120, 1.5532])
        self.res_en_2d = np.array([[0.0001,  0.0097], [0.6120, 1.5532]])


    def assertSingleValue(self, function, input, expected):
        """Assert function works for single float values"""
        result = function(input)
        self.assertIsInstance(result, float, "Result should be a float for single value input")
        self.assertAlmostEqual(result, expected, places=4, msg="Single value output incorrect")

    def assert1DArray(self, function, input, expected):
        """Assert function works for 1D numpy arrays"""
        result = function(np.array(input))
        np.testing.assert_array_almost_equal(result, expected, decimal=4, err_msg="1D array output incorrect")
        self.assertIsInstance(result, np.ndarray, "Result should be a 1D numpy array")
        self.assertEqual(result.ndim, 1, "Result should be a 1D numpy array")

    def assert2DArray(self, function, input, expected):
        """Assert function works for 2D numpy arrays"""
        result = function(np.array(input))
        np.testing.assert_array_almost_equal(result, expected, decimal=4, err_msg="2D array output incorrect")
        self.assertIsInstance(result, np.ndarray, "Result should be a 2D numpy array")
        self.assertEqual(result.ndim, 2, "Result should be a 2D numpy array")

    def assertRoundTrip(self, func1, func2, input):
        res1 = func1(input)
        res2 = func2(res1)
        self.assertAlmostEqual(input, res2,9, "Round trip conversion failed")

    # Tests for en2pc
    def test_en2pc_single_value(self):
        self.assertSingleValue(en2pc, self.in_float, self.res_pc_float)  # Replace expected_value with the correct one

    def test_en2pc_1D_array(self):
        self.assert1DArray(en2pc, self.in_1d, self.res_pc_1d)  # Replace expected_values

    def test_en2pc_2D_array(self):
        self.assert2DArray(en2pc, self.in_2d, self.res_pc_2d)  # Replace expected_values

    # Tests for pc2en
    def test_pc2en_single_value(self):
        self.assertSingleValue(pc2en, self.in_float, self.res_en_float)  # Replace expected_value with the correct one

    def test_pc2en_1D_array(self):
        self.assert1DArray(pc2en, self.in_1d, self.res_en_1d)  # Replace expected_values

    def test_pc2en_2D_array(self):
        self.assert2DArray(pc2en, self.in_2d, self.res_en_2d)  # Replace expected_values

    def test_round_trip_pc2en_en2pc(self):
        self.assertRoundTrip(pc2en, en2pc, self.in_float)

    def test_round_trip_en2pc_pc2en(self):
        self.assertRoundTrip(en2pc, pc2en, self.in_float)

if __name__ == '__main__':
    unittest.main()

