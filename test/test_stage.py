# -*- coding: utf-8 -*-
import unittest
import numpy as np
import sys
import pickle
sys.path.append('../')
import stage


class  TestStage(unittest.TestCase):
    """Test class for stage.py
    """
    def test_psd_100Hz(self):
        exp = np.load('data/Power_EEG_0thRow_ID46770_FASTER2_20200206_EEG_2019-023.npy')
        # this is from 0th row of EEG at ID46770 from FASTER2_20200206_EEG_2019-023
        y = np.load('data/Voltage_EEG_0thRow_ID46770_FASTER2_20200206_EEG_2019-023.npy')
        ans = stage.psd(y, 256, 100)

        np.testing.assert_array_almost_equal(exp, ans)

    
    def test_psd_128Hz(self):
        exp = np.load('data/Power_EEG_0thRow_09ch_FASTER_20191101_EEG_2019-015.npy')
        # this is from 0th row of EEG at 09 channel from FASTER_20191101_EEG_2019-015
        y = np.load('data/Voltage_EEG_0thRow_09ch_FASTER_20191101_EEG_2019-015.npy')
        ans = stage.psd(y, 327, 128)

        np.testing.assert_array_almost_equal(exp, ans)


    def test_psd_500Hz(self):
        exp = np.load('data/Power_EEG_0thRow_ID45073_FASTER_161006_Wake17.npy')
        # this is from 0th row of EEG at ID45073 channel from FASTER_161006_Wake17
        y = np.load('data/Voltage_EEG_0thRow_ID45073_FASTER_161006_Wake17.npy')
        ans = stage.psd(y, 1280, 500)

        np.testing.assert_array_almost_equal(exp, ans)


    def test_plot_scatter2D(self):
        points = np.load('data/Points2D_LowFreq-HighFreq_Axes_ID45764_FASTER2_20200206_EEG_2019-023.npy')
        pred   = np.load('data/Pred_LowFreq-HighFreq_Axes_ID45764_FASTER2_20200206_EEG_2019-023.npy')
        means  = np.load('data/Remodel_means_LowFreq-HighFreq_Axes_ID45764_FASTER2_20200206_EEG_2019-023.npy')
        covars = np.load('data/Remodel_covers_LowFreq-HighFreq_Axes_ID45764_FASTER2_20200206_EEG_2019-023.npy')

        fig = stage.plot_scatter2D(
            points, pred, means, covars, [stage.COLOR_NREM, stage.COLOR_WAKE], stage.XLABEL, stage.YLABEL)

        exp = (stage.SCATTER_PLOT_FIG_WIDTH, stage.SCATTER_PLOT_FIG_HEIGHT)
        ans = (fig.get_figwidth(), fig.get_figheight())

        assert exp == ans


    def test_pickle_voltage_matrices(self):
        dummy_eeg = np.zeros((32400, 800))
        dummy_emg  = np.ones((32400, 800))

        stage.pickle_voltage_matrices(dummy_eeg, dummy_emg, '../test/data', 'ID-test')
        
        with open('../test/data/pkl/ID-test_EEG.pkl', 'rb') as pkl:
            res_eeg = pickle.load(pkl)

        with open('../test/data/pkl/ID-test_EMG.pkl', 'rb') as pkl:
            res_emg = pickle.load(pkl)

        assert (res_eeg.shape == (32400, 800)) & (res_emg.shape == (32400, 800))


    def test_spectrum_normalize(self):
        exp = np.load('data/NormPSD_EEG_0-9thRow_ID37963_FASTER_20191101_EEG_2019-013.npy')
        test_vm = np.load('data/Voltage_EEG_0-9thRow_ID37963_FASTER_20191101_EEG_2019-013.npy')

        ans = stage.spectrum_normalize(test_vm, 256, 100)

        np.testing.assert_array_almost_equal(exp, ans['psd'])

if __name__ == "__main__":
    unittest.main()
