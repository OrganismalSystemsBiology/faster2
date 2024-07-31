# -*- coding: utf-8 -*-
import unittest
import os
import sys
sys.path.append('../')
import faster2lib.dsi_tools as dt
from datetime import datetime

DATA_ROOT = 'data/'
FASTER_FOLDER = 'FASTER2_20200206_EEG_2019-023'
STAGE_FOLDER = 'data/auto.stage'
TRANSMITTER_ID = 'ID33572'

class TestDSI_TEXT_Reader(unittest.TestCase):
    def setUp(self):
        self.dsi_reader = dt.DSI_TXT_Reader(os.path.join(
            DATA_ROOT, 
            FASTER_FOLDER, 
            'data/dsi.txt'), 
            f'{TRANSMITTER_ID}', 'EEG', sample_freq=100)

    
    def test_read_epochs(self):
        data = self.dsi_reader.read_epochs(1, 451)
        self.assertEqual(len(data), 100*8*451)

    def test_read_epochs_by_datetime(self):
        data = self.dsi_reader.read_epochs_by_datetime( datetime(2020,2,7,8,0,0), datetime(2020,2,7,10,0,0)) 
        self.assertEqual(len(data), 2*60*60*100)


if __name__ == "__main__":
    unittest.main()