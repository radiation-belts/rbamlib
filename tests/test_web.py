import unittest
from unittest import TestCase
from unittest.mock import patch
import numpy as np

from rbamlib.web import omni


class TestWeb(TestCase):
    def setUp(self):
        """Set up common test variables."""
        self.start_date = '2023-01-01'
        self.end_date = '2023-01-02'

        self.mock_response_kp = """<HTML><HEAD><TITLE>OMNIWeb Results</TITLE></HEAD><BODY>
<B>Listing for omni2 data from 20230101 to 20230102</B><hr><pre>Selected parameters:
1 Kp index
YEAR DOY HR  1
2023   1  0 23
2023   1  1 23
2023   2 23 30
</pre><hr></BODY></HTML>"""

        self.mock_response_kp_dst = """<HTML><HEAD><TITLE>OMNIWeb Results</TITLE></HEAD><BODY>
<B>Listing for omni2 data from 20230101 to 20230102</B><hr><pre>Selected parameters:
1 Kp index
2 Dst-index, nT
YEAR DOY HR  1    2
2023   1  0 23   -23
2023   1  1 23   -21
2023   1  2 23   -23
2023   2 23 30    -9
</pre><hr></BODY></HTML>"""

    def tearDown(self):
        """Clean up resources after tests if needed."""
        pass

    @unittest.skip('Skipping live url request')
    def test_omni_live(self):
        """Test omni function with a real internet connection for Kp only."""
        time, kp = omni(self.start_date, self.end_date, {'Kp'})

        self.assertIsInstance(time, np.ndarray, "Output time should be a numpy array.")
        self.assertIsInstance(kp, np.ndarray, "Output Kp should be a numpy array.")
        self.assertEqual(len(time), len(kp), "Time and Kp data lengths should match.")

        print("Live omni function test passed.")

    def test_omni_mock(self):
        """Test omni function without using the internet by mocking the request for Kp only."""
        with patch('requests.get') as mock_get:
            mock_get.return_value.text = self.mock_response_kp
            time, kp = omni(self.start_date, self.end_date, {'Kp'})

            self.assertIsInstance(time, np.ndarray, "Output time should be a numpy array.")
            self.assertIsInstance(kp, np.ndarray, "Output Kp should be a numpy array.")
            self.assertEqual(len(time), len(kp), "Time and Kp data lengths should match.")

    def test_omni_mock_kp_dst(self):
        """Test omni function without using the internet by mocking the request for Kp and Dst."""
        with patch('requests.get') as mock_get:
            mock_get.return_value.text = self.mock_response_kp_dst
            time, kp, dst = omni(self.start_date, self.end_date, {38, 'Dst'})

            self.assertIsInstance(time, np.ndarray, "Output time should be a numpy array.")
            self.assertIsInstance(kp, np.ndarray, "Output Kp should be a numpy array.")
            self.assertIsInstance(dst, np.ndarray, "Output Dst should be a numpy array.")
            self.assertEqual(len(time), len(kp), "Time and Kp data lengths should match.")
            self.assertEqual(len(time), len(dst), "Time and Dst data lengths should match.")


if __name__ == '__main__':
    unittest.main()
