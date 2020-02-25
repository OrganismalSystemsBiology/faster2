# -*- coding: utf-8 -*-
import unittest
import os
import sys
sys.path.append('../')
import eeg_tools as et
import numpy as np

DATA_ROOT = 'G:/EEG/experiment-data'
FASTER_FOLDER = 'FASTER_20191101_EEG_2019-015'
STAGE_FOLDER = 'data/auto.stage'
TRANSMITTER_ID = '13'

class TestDSI_TEXT_Reader(unittest.TestCase):
    def setUp(self):
        self.dsi_reader = et.DSI_TXT_Reader(os.path.join(
            DATA_ROOT, 
            FASTER_FOLDER, 
            'data/dsi.txt'), 
            f'{TRANSMITTER_ID}', 'EEG', sample_freq=128)

    
    def test_read_epochs(self):
        data = self.dsi_reader.read_epochs(1, 451)
        self.assertEqual(len(data), 128*8*451)


class TestFunctions(unittest.TestCase):
    def test_read_stages(self):
        res = et.read_stages(os.path.join(
                DATA_ROOT, 
                FASTER_FOLDER, 
                'data/auto.stage'), 
                f'{TRANSMITTER_ID}')
        
        exp = np.array(['WAKE', 'NREM', 'REM', 'WAKE'])
        res = res[np.r_[0,400,498,32399]]
        np.testing.assert_array_equal(res, exp)

    def test_patch_nan(self):
        """test one nan in an array is pached
        """
        exp = np.array([0,1,2,3,0,4,5])

        t = np.array([0,1,2,3,np.nan,4,5])
        et.patch_nan(t)

        np.testing.assert_array_equal(t, exp)

    def test_patch_nan_percent(self):
        """test if it returns the correct ratio of nan
        """
        exp = 1/7

        t = np.array([0,1,2,3,np.nan,4,5])
        ans = et.patch_nan(t)

        assert exp == ans


    def test_patch_nan_not_patchable(self):
        """test if nothing happens when the given array is non patchable, 
        i.e. the more than a half elements of the given array are nan.
        """
        t = np.array([0, np.nan, np.nan, np.nan, np.nan, 4, 5])
        exp = np.copy(t)

        et.patch_nan(t)

        np.testing.assert_array_equal(t, exp)

if __name__ == "__main__":
    unittest.main()