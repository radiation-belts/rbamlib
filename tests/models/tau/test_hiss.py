import unittest
import numpy as np
from tests.helpers import TestHelpers
from rbamlib.models.tau.hiss import O2016


class TestHiss(unittest.TestCase, TestHelpers):

    def test_Figure1(self):
        """Test O2016 with single input values. based on approximate value from Figure 1 (https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2015JA021878)"""        
        MLT = np.arange(0, 24)        
        kp = np.arange(0,7)

        mlt_g, kp_g = np.meshgrid(MLT, kp, indexing='xy')
                
        result = O2016(mlt=mlt_g, L=4.5, en=0.1, kp=kp_g) / (60 * 60 * 24)
        ave = np.mean(result.flatten())
        self.assertAlmostEqual(np.log10(ave), 0, places=0, msg=f"Single value {np.log10(ave)} is incorrect")

        result = O2016(mlt=mlt_g, L=5.5, en=10, kp=kp_g) / (60 * 60 * 24)
        ave = np.mean(result.flatten())
        self.assertAlmostEqual(np.log10(ave), 3, places=0, msg=f"Single value {np.log10(ave)} is incorrect")

        result = O2016(mlt=mlt_g, L=2.5, en=0.01, kp=kp_g) / (60 * 60 * 24)        
        self.assertTrue(np.isnan(result.flatten()).all(), msg=f"Should be all NaN")

if __name__ == '__main__':
    unittest.main()
