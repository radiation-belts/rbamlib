import unittest
from unittest import TestCase
from unittest.mock import patch
import numpy as np
import os

from rbamlib.web import omni

if 'RUN_LIVE_TESTS' not in os.environ:
    os.environ['RUN_LIVE_TESTS'] = 'False'

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

        self.mock_response_ae_al = """<HTML><HEAD><TITLE>OMNIWeb Results</TITLE></HEAD><BODY>
<B>Listing for omni_5min data from 20230101 to 20230102</B><hr><pre>Selected parameters:
1 AE-index, nT
2 AL-index, nT
YYYY DOY HR MN    1     2
2023   1  0  0     9    -5
2023   1  0  5     9    -5
2023   1  0 10    10    -6
2023   1  0 15    12    -8
2023   1  0 20    10    -8
2023   1  0 25    12    -7
2023   2 23 30    35   -22
2023   2 23 35    30   -18
2023   2 23 40    37   -25
2023   2 23 45    30   -17
2023   2 23 50    30   -16
2023   2 23 55    30   -16
</pre><hr></BODY></HTML>"""

    def tearDown(self):
        """Clean up resources after tests if needed."""
        pass

    @unittest.skipUnless(os.getenv('RUN_LIVE_TESTS', 'False').lower() == 'true', "Skipping live test unless enabled.")
    def test_omni_live(self):
        """Test LRO omni function with a real internet connection for Kp only."""
        time, kp = omni(self.start_date, self.end_date, {'Kp'})

        self.assertIsInstance(time, np.ndarray, "Output time should be a numpy array.")
        self.assertIsInstance(kp, np.ndarray, "Output Kp should be a numpy array.")
        self.assertEqual(len(time), len(kp), "Time and Kp data lengths should match.")

        print("Live omni low function test passed.")

    @unittest.skipUnless(os.getenv('RUN_LIVE_TESTS', 'False').lower() == 'true', "Skipping live test unless enabled.")
    def test_omni_live_high_res(self):
        """Test LRO omni function with a real internet connection for AE index at high resolution."""
        time, ae = omni(self.start_date, self.end_date, {'AE'}, resolution='5min')

        self.assertIsInstance(time, np.ndarray, "Output time should be a numpy array.")
        self.assertIsInstance(ae, np.ndarray, "Output AE should be a numpy array.")
        self.assertEqual(len(time), len(ae), "Time and AE data lengths should match.")

        print("Live high-resolution omni function test passed.")

    def test_omni_mock(self):
        """Test omnilow function without using the internet by mocking the request for Kp only."""
        with patch('requests.get') as mock_get:
            mock_get.return_value.text = self.mock_response_kp
            time, kp = omni(self.start_date, self.end_date, {'Kp'})

            self.assertIsInstance(time, np.ndarray, "Output time should be a numpy array.")
            self.assertIsInstance(kp, np.ndarray, "Output Kp should be a numpy array.")
            self.assertEqual(len(time), len(kp), "Time and Kp data lengths should match.")

    def test_omni_mock_kp_dst(self):
        """Test omnilow function without using the internet by mocking the request for Kp and Dst."""
        with patch('requests.get') as mock_get:
            mock_get.return_value.text = self.mock_response_kp_dst
            time, kp, dst = omni(self.start_date, self.end_date, {38, 'Dst'})

            self.assertIsInstance(time, np.ndarray, "Output time should be a numpy array.")
            self.assertIsInstance(kp, np.ndarray, "Output Kp should be a numpy array.")
            self.assertIsInstance(dst, np.ndarray, "Output Dst should be a numpy array.")
            self.assertEqual(len(time), len(kp), "Time and Kp data lengths should match.")
            self.assertEqual(len(time), len(dst), "Time and Dst data lengths should match.")

    def test_omni_mock_high_res(self):
        """Test omni function without using the internet for high-resolution AE and AL data."""
        with patch('requests.get') as mock_get:
            mock_get.return_value.text = self.mock_response_ae_al
            time, ae, al = omni(self.start_date, self.end_date, {'AE', 'AL'}, resolution='5min')

            self.assertIsInstance(time, np.ndarray, "Output time should be a numpy array.")
            self.assertIsInstance(ae, np.ndarray, "Output AE should be a numpy array.")
            self.assertIsInstance(al, np.ndarray, "Output AL should be a numpy array.")
            self.assertEqual(len(time), len(ae), "Time and AE data lengths should match.")
            self.assertEqual(len(time), len(al), "Time and AL data lengths should match.")

if __name__ == '__main__':
    unittest.main()
