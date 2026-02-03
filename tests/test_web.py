import unittest
from unittest import TestCase
from unittest.mock import patch
import numpy as np
import os

from rbamlib.web import omni, download_unzip

if 'RUN_LIVE_TESTS' not in os.environ:
    os.environ['RUN_LIVE_TESTS'] = 'False'

class TestWeb(TestCase):

    def setUp(self):
        """Set up common test variables."""
        self.start_date = '2023-10-01'
        self.end_date = '2023-10-02'

        self.mock_response_kp = """<HTML><HEAD><TITLE>OMNIWeb Results</TITLE></HEAD><BODY>
<B>Listing for omni2 data from 20231001 to 20231002</B><hr><pre>Selected parameters:
1 Kp index
YEAR DOY HR  1
2023  274  0 23
2023  274  1 23
2023  275 23 30
</pre><hr></BODY></HTML>"""

        self.mock_response_kp_dst = """<HTML><HEAD><TITLE>OMNIWeb Results</TITLE></HEAD><BODY>
<B>Listing for omni2 data from 20231001 to 20231002</B><hr><pre>Selected parameters:
1 Kp index
2 Dst-index, nT
YEAR DOY HR  1    2
2023  274  0 23   -23
2023  274  1 23   -21
2023  274  2 23   -23
2023  275 23 30    -9
</pre><hr></BODY></HTML>"""

        self.mock_response_dst_kp = """<HTML><HEAD><TITLE>OMNIWeb Results</TITLE></HEAD><BODY>
<B>Listing for omni2 data from 20231001 to 20231002</B><hr><pre>Selected parameters:
1 Dst-index, nT
2 Kp index
YEAR DOY HR  1    2
2023  274  0 -23  23
2023  274  1 -21  23
2023  274  2 -23  23
2023  275 23  -9  30
</pre><hr></BODY></HTML>"""

        self.mock_response_ae_al = """<HTML><HEAD><TITLE>OMNIWeb Results</TITLE></HEAD><BODY>
<B>Listing for omni_5min data from 20231001 to 20231002</B><hr><pre>Selected parameters:
1 AE-index, nT
2 SymH-index, nT
YYYY DOY HR MN    1     2
2023  274  0  0   636   -55
2023  274  0  5   460   -54
2023  274  0 10   386   -53
2023  274  0 15   380   -52
2023  274  1  0   354   -45
2023  274  1  5   304   -43
2023  274  1 10   288   -41
2023  274  1 15   297   -40
</pre><hr></BODY></HTML>"""

    def tearDown(self):
        """Clean up resources after tests if needed."""
        pass

    def _setup_mock_urlopen(self, mock_urlopen, response_text):
        """Helper to configure urllib.request.urlopen mock with response text."""
        mock_urlopen.return_value.__enter__.return_value.read.return_value = response_text.encode('utf-8')

    @unittest.skipUnless(os.getenv('RUN_LIVE_TESTS', 'False').lower() == 'true', "Skipping live test unless enabled.")
    def test_omni_live(self):
        """Test LRO omni function with a real internet connection for Kp only."""
        time, kp = omni(self.start_date, self.end_date, ['Kp'])

        self.assertIsInstance(time, np.ndarray, "Output time should be a numpy array.")
        self.assertIsInstance(kp, np.ndarray, "Output Kp should be a numpy array.")
        self.assertEqual(len(time), len(kp), "Time and Kp data lengths should match.")

        print("Live omni low function test passed.")

    @unittest.skipUnless(os.getenv('RUN_LIVE_TESTS', 'False').lower() == 'true', "Skipping live test unless enabled.")
    def test_omni_live_high_res(self):
        """Test LRO omni function with a real internet connection for AE index at high resolution."""
        time, ae = omni(self.start_date, self.end_date, ['AE'], resolution='5min')

        self.assertIsInstance(time, np.ndarray, "Output time should be a numpy array.")
        self.assertIsInstance(ae, np.ndarray, "Output AE should be a numpy array.")
        self.assertEqual(len(time), len(ae), "Time and AE data lengths should match.")

        print("Live high-resolution omni function test passed.")

    def test_omni_mock(self):
        """Test omnilow function without using the internet by mocking the request for Kp only."""
        with patch('urllib.request.urlopen') as mock_urlopen:
            self._setup_mock_urlopen(mock_urlopen, self.mock_response_kp)
            time, kp = omni(self.start_date, self.end_date, ['Kp'])

            self.assertIsInstance(time, np.ndarray, "Output time should be a numpy array.")
            self.assertIsInstance(kp, np.ndarray, "Output Kp should be a numpy array.")
            self.assertEqual(len(time), len(kp), "Time and Kp data lengths should match.")

    def test_omni_mock_smallcase(self):
        """Test omnilow function without using the internet by mocking the request for Kp only."""
        with patch('urllib.request.urlopen') as mock_urlopen:
            self._setup_mock_urlopen(mock_urlopen, self.mock_response_kp)
            time, kp = omni(self.start_date, self.end_date, ['kp'])
            self.assertIsInstance(kp, np.ndarray, "Output Kp should be a numpy array.")

    def test_omni_mock_kp_dst(self):
        """Test omnilow function without using the internet by mocking the request for Kp and Dst."""
        with patch('urllib.request.urlopen') as mock_urlopen:
            self._setup_mock_urlopen(mock_urlopen, self.mock_response_kp_dst)
            time, kp, dst = omni(self.start_date, self.end_date, [38, 'Dst'])

            self.assertIsInstance(time, np.ndarray, "Output time should be a numpy array.")
            self.assertIsInstance(kp, np.ndarray, "Output Kp should be a numpy array.")
            self.assertIsInstance(dst, np.ndarray, "Output Dst should be a numpy array.")
            self.assertEqual(len(time), len(kp), "Time and Kp data lengths should match.")
            self.assertEqual(len(time), len(dst), "Time and Dst data lengths should match.")

    def test_omni_mock_high_res(self):
        """Test omni function without using the internet for high-resolution AE and AL data."""
        with patch('urllib.request.urlopen') as mock_urlopen:
            self._setup_mock_urlopen(mock_urlopen, self.mock_response_ae_al)
            time, ae, symh = omni(self.start_date, self.end_date, ['AE', 'SYM_H'], resolution='5min')

            self.assertIsInstance(time, np.ndarray, "Output time should be a numpy array.")
            self.assertIsInstance(ae, np.ndarray, "Output AE should be a numpy array.")
            self.assertIsInstance(symh, np.ndarray, "Output SymH should be a numpy array.")
            self.assertEqual(len(time), len(ae), "Time and AE data lengths should match.")
            self.assertEqual(len(time), len(symh), "Time and SymH data lengths should match.")

    def test_omni_parameter_order(self):
        """Test that omni returns data in the order of requested variables"""
        with patch('urllib.request.urlopen') as mock_urlopen:
            # Request Kp, Dst - response has Kp in column 1, Dst in column 2
            self._setup_mock_urlopen(mock_urlopen, self.mock_response_kp_dst)
            _, kp_a, dst_a = omni(self.start_date, self.end_date, ['Kp', 'Dst'])

            # Request Dst, Kp - response has Dst in column 1, Kp in column 2
            self._setup_mock_urlopen(mock_urlopen, self.mock_response_dst_kp)
            _, dst_b, kp_b = omni(self.start_date, self.end_date, ['Dst', 'Kp'])

            # Verify kp_a and kp_b are identical, dst_a and dst_b are identical
            np.testing.assert_array_equal(kp_a, kp_b, err_msg="Kp values should be identical")
            np.testing.assert_array_equal(dst_a, dst_b, err_msg="Dst values should be identical")

    def test_omni_parameter_case_variations(self):
        """Test that omni handles different case variations of parameter names"""
        with patch('urllib.request.urlopen') as mock_urlopen:
            self._setup_mock_urlopen(mock_urlopen, self.mock_response_kp)

            # Test lowercase 'kp'
            time, kp_lower = omni(self.start_date, self.end_date, ['kp'])
            self.assertIsInstance(kp_lower, np.ndarray, "Lowercase 'kp' should work")
            self.assertEqual(len(time), len(kp_lower), "Data lengths should match")

            # Test uppercase 'KP'
            time, kp_upper = omni(self.start_date, self.end_date, ['KP'])
            self.assertIsInstance(kp_upper, np.ndarray, "Uppercase 'KP' should work")
            self.assertEqual(len(time), len(kp_upper), "Data lengths should match")

            # Test mixed case 'Kp' (standard)
            time, kp_mixed = omni(self.start_date, self.end_date, ['Kp'])
            self.assertIsInstance(kp_mixed, np.ndarray, "Mixed case 'Kp' should work")
            self.assertEqual(len(time), len(kp_mixed), "Data lengths should match")

        # Test with Dst parameter case variations
        with patch('urllib.request.urlopen') as mock_urlopen:
            self._setup_mock_urlopen(mock_urlopen, self.mock_response_kp_dst)

            # Test lowercase 'dst'
            time, kp, dst_lower = omni(self.start_date, self.end_date, ['Kp', 'dst'])
            self.assertIsInstance(dst_lower, np.ndarray, "Lowercase 'dst' should work")

            # Test uppercase 'DST'
            time, kp, dst_upper = omni(self.start_date, self.end_date, ['Kp', 'DST'])
            self.assertIsInstance(dst_upper, np.ndarray, "Uppercase 'DST' should work")

    def test_omni_same_day_different_hours(self):
        """Test that omni distinguishes between different hours on the same day"""
        with patch('urllib.request.urlopen') as mock_urlopen:
            self._setup_mock_urlopen(mock_urlopen, self.mock_response_ae_al)

            # First call: start from beginning of day (hour 0)
            time1, ae1, symh1 = omni(self.start_date, self.end_date, ['AE', 'SYM_H'], resolution='5min')

            # Second call: start from hour 1 (filters out hour 0 data)
            time2, ae2, symh2 = omni(self.start_date + ' 01:00', self.end_date, ['AE', 'SYM_H'], resolution='5min')

            # First call should have more data points (includes hour 0 and hour 1)
            self.assertGreater(len(time1), len(time2), "First call should have more data points")
            # Verify first call includes data from hour 0
            np.testing.assert_array_equal(ae1[:6], np.array([636., 460., 386., 380., 354., 304.]), err_msg="First call AE values incorrect")
            np.testing.assert_array_equal(symh1[:6], np.array([-55., -54., -53., -52., -45., -43.]), err_msg="First call SYM_H values incorrect")
            # Verify second call starts from hour 1 (skips hour 0)
            np.testing.assert_array_equal(ae2[:4], np.array([354., 304., 288., 297.]), err_msg="Second call AE values incorrect")
            np.testing.assert_array_equal(symh2[:4], np.array([-45., -43., -41., -40.]), err_msg="Second call SYM_H values incorrect")

    def test_download_unzip_single_file(self):
        """Test download_unzip extracting a single file from a zip."""
        with patch('urllib.request.urlretrieve') as mock_retrieve, \
             patch('zipfile.ZipFile') as mock_zipfile, \
             patch('tempfile.NamedTemporaryFile') as mock_tempfile, \
             patch('os.makedirs'), \
             patch('os.rename'), \
             patch('os.unlink'):

            # Setup mocks
            mock_temp = mock_tempfile.return_value.__enter__.return_value
            mock_temp.name = '/tmp/test.zip'

            mock_zip = mock_zipfile.return_value.__enter__.return_value
            mock_zip.namelist.return_value = ['data/test_file.mat']
            mock_zip.extract.return_value = '/target/data/test_file.mat'

            # Call function
            result = download_unzip(
                'https://example.com/test.zip',
                target_folder='/target',
                filename_in_zip='test_file.mat'
            )

            # Verify
            self.assertIsInstance(result, str)
            self.assertTrue(result.endswith('test_file.mat'))

    def test_download_unzip_all_files(self):
        """Test download_unzip extracting all files from a zip."""
        with patch('urllib.request.urlretrieve') as mock_retrieve, \
             patch('zipfile.ZipFile') as mock_zipfile, \
             patch('tempfile.NamedTemporaryFile') as mock_tempfile, \
             patch('os.makedirs'), \
             patch('os.unlink'):

            # Setup mocks
            mock_temp = mock_tempfile.return_value.__enter__.return_value
            mock_temp.name = '/tmp/test.zip'

            mock_zip = mock_zipfile.return_value.__enter__.return_value
            mock_zip.namelist.return_value = [
                'file1.txt',
                'subdir/',
                'subdir/file2.dat',
                'file3.csv'
            ]

            # Call function with no filename_in_zip
            result = download_unzip(
                'https://example.com/test.zip',
                target_folder='/target'
            )

            # Verify
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 3)  # 3 files, directory excluded
            mock_zip.extractall.assert_called_once_with('/target')

    def test_download_unzip_default_folder(self):
        """Test download_unzip with default target folder (cwd)."""
        with patch('urllib.request.urlretrieve'), \
             patch('zipfile.ZipFile') as mock_zipfile, \
             patch('tempfile.NamedTemporaryFile') as mock_tempfile, \
             patch('os.makedirs'), \
             patch('os.getcwd', return_value='/current/dir'), \
             patch('os.unlink'):

            mock_temp = mock_tempfile.return_value.__enter__.return_value
            mock_temp.name = '/tmp/test.zip'

            mock_zip = mock_zipfile.return_value.__enter__.return_value
            mock_zip.namelist.return_value = ['file.txt']

            # Call without target_folder
            result = download_unzip('https://example.com/test.zip')

            # Verify extractall was called with cwd
            self.assertIsInstance(result, list)
            mock_zip.extractall.assert_called_once_with('/current/dir')

    def test_download_unzip_url_passed_correctly(self):
        """Test that download_unzip passes the correct URL to urlretrieve."""
        with patch('urllib.request.urlretrieve') as mock_retrieve, \
             patch('zipfile.ZipFile') as mock_zipfile, \
             patch('tempfile.NamedTemporaryFile') as mock_tempfile, \
             patch('os.makedirs'), \
             patch('os.unlink'):

            mock_temp = mock_tempfile.return_value.__enter__.return_value
            mock_temp.name = '/tmp/test.zip'

            mock_zip = mock_zipfile.return_value.__enter__.return_value
            mock_zip.namelist.return_value = ['file.txt']

            test_url = 'https://example.com/my_data.zip'
            download_unzip(test_url, target_folder='/target')

            # Verify urlretrieve was called with the correct URL
            mock_retrieve.assert_called_once_with(test_url, '/tmp/test.zip')

if __name__ == '__main__':
    unittest.main()
