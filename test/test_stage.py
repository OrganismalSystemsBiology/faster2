# -*- coding: utf-8 -*-
import unittest
import numpy as np
import sys
sys.path.append('../')
import stage
import datetime
import pickle

class  TestStage(unittest.TestCase):
    """Test class for stage.py
    """

    def test_read_mouse_info(self):
        mouse_info = stage.read_mouse_info("../data")

        exp = np.array(["ID46770",
                        "ID45764",
                        "ID47313",
                        "ID37963",
                        "ID33572",
                        "ID47481",
                        "ID47479",
                        "ID45791"]
                       )

        res = mouse_info['Device label']

        np.testing.assert_array_equal(res, exp)

    
    def test_read_exp_info(self):
        exp_info = stage.read_exp_info("../data")

        exp = np.array(['EEG_2019-023', 'SSS_A-E', '2020/2/7 08:00:00', '2020/2/10 08:00:00', '100'])
        res = np.array([
            exp_info['Experiment label'].values[0],
            exp_info['Rack label'].values[0],
            exp_info['Start datetime'].values[0],
            exp_info['End datetime'].values[0],
            exp_info['Sampling freq'].values[0]])

        np.testing.assert_array_equal(res, exp)


    def test_interpret_datetimestr_with_slashes(self):
        exp = datetime.datetime(2018, 12, 12)
        ans = stage.interpret_datetimestr('2018/12/12')
        assert exp == ans


    def test_interpret_datetimestr_with_hyphens(self):
        exp = datetime.datetime(2018,2,4)
        ans = stage.interpret_datetimestr('2018-2-4')
        assert exp == ans


    def test_interpret_datetimestr_with_slash_and_colon(self):
        exp = datetime.datetime(2018,2,12, 12,29,31)
        ans = stage.interpret_datetimestr('2018/2/12 12:29:31')
        assert exp == ans
        

    def test_interpret_datetimestr_without_delimiters(self):
        exp = datetime.datetime(2018,2,2)
        ans = stage.interpret_datetimestr('20180202')
        assert exp == ans


    def test_interpret_datetimestr_without_delimiters2(self):
        exp = datetime.datetime(2018,2,12,7,59,59)
        ans = stage.interpret_datetimestr('20180212075959')
        assert exp == ans


    def test_read_voltage_matrices_dsi(self):
        exp = [[450, 800], [450, 800]]

        (eeg_vm, emg_vm, _) = stage.read_voltage_matrices('../data/dsi_test', 'ID33572', 100, 8, 450)
        ans = [eeg_vm.shape, emg_vm.shape]
        
        np.testing.assert_array_equal(exp, ans)


    def test_psd_100Hz(self):
        exp = np.load('Power_EEG_0thRow_ID46770_FASTER2_20200206_EEG_2019-023.npy')
        # this is from 0th row of EEG at ID46770 from FASTER2_20200206_EEG_2019-023
        y = np.load('Voltage_EEG_0thRow_ID46770_FASTER2_20200206_EEG_2019-023.npy')
        ans = stage.psd(y, 256, 100)

        np.testing.assert_array_equal(exp, ans)

    
    def test_psd_128Hz(self):
        exp = np.load('Power_EEG_0thRow_09ch_FASTER_20191101_EEG_2019-015.npy')
        # this is from 0th row of EEG at 09 channel from FASTER_20191101_EEG_2019-015
        y = np.load('Voltage_EEG_0thRow_09ch_FASTER_20191101_EEG_2019-015.npy')
        ans = stage.psd(y, 327, 128)

        np.testing.assert_array_equal(exp, ans)


    def test_psd_500Hz(self):
        exp = np.load('Power_EEG_0thRow_ID45073_FASTER_161006_Wake17.npy')
        # this is from 0th row of EEG at ID45073 channel from FASTER_161006_Wake17
        y = np.load('Voltage_EEG_0thRow_ID45073_FASTER_161006_Wake17.npy')
        ans = stage.psd(y, 1280, 500)

        np.testing.assert_array_equal(exp, ans)


    def test_plot_scatter2D(self):
        points = np.load('Points2D_LowFreq-HighFreq_Axes_ID45764_FASTER2_20200206_EEG_2019-023.npy')
        pred   = np.load('Pred_LowFreq-HighFreq_Axes_ID45764_FASTER2_20200206_EEG_2019-023.npy')
        means  = np.load('Remodel_means_LowFreq-HighFreq_Axes_ID45764_FASTER2_20200206_EEG_2019-023.npy')
        covars = np.load('Remodel_covers_LowFreq-HighFreq_Axes_ID45764_FASTER2_20200206_EEG_2019-023.npy')

        fig = stage.plot_scatter2D(
            points, pred, means, covars, [stage.COLOR_NREM, stage.COLOR_WAKE], stage.XLABEL, stage.YLABEL)

        exp = (stage.SCATTER_PLOT_FIG_WIDTH, stage.SCATTER_PLOT_FIG_HEIGHT)
        ans = (fig.get_figwidth(), fig.get_figheight())

        assert exp == ans

    
    def test_shrink_rem_cluster_case1(self):

        m = np.array([[-2.75317264e+01,  7.72000698e+00,  6.10634993e+01],
                      [-2.73355234e+01,  3.70791857e+01, -3.76312561e+01],
                      [1.94492513e+01, -2.18150968e+01,  5.94370116e-16]])
        c = np.array([[ 215.50327258,   19.76270819,  170.27740157],
                      [19.76270819,  212.28787468,  149.06614634],
                      [170.27740157,  149.06614634, 1821.99592244]])
        exp = np.array([[198.90607922,   5.34295722,  78.45686422],
                        [5.34295722, 196.56756494,  68.85410475],
                        [78.45686422,  68.85410475, 935.79826227]])

        ans = stage.shrink_rem_cluster(m[0], c)

        np.testing.assert_array_almost_equal(exp, ans)

    def test_shrink_rem_cluster_no_shrink(self):

        m = np.array([[142.86587753, -42.10978493,  88.71408145],
                      [-42.10978493, 129.2552841,  80.03791474],
                      [88.71408145,  80.03791474, 933.6907858]])
        c = np.array([[ 215.50327258,   19.76270819,  170.27740157],
                      [19.76270819,  212.28787468,  149.06614634],
                      [170.27740157,  149.06614634, 1821.99592244]])
        ans = stage.shrink_rem_cluster(m[0], c).size
        exp = 0
        assert exp == ans


    def test_pickle_voltage_matrices(self):
        dummy_eeg = np.zeros((32400, 800))
        dummy_emg  = np.ones((32400, 800))

        stage.pickle_voltage_matrices(dummy_eeg, dummy_emg, '../data', 'ID-test')
        
        with open('../data/pkl/ID-test_EEG.pkl', 'rb') as pkl:
            res_eeg = pickle.load(pkl)

        with open('../data/pkl/ID-test_EMG.pkl', 'rb') as pkl:
            res_emg = pickle.load(pkl)

        assert (res_eeg.shape == (32400, 800)) & (res_emg.shape == (32400, 800))


    def test_read_voltage_matrices_pickle(self):
        exp = [[32400, 800], [32400, 800]]

        (eeg_vm, emg_vm, _) = stage.read_voltage_matrices('../data', 'ID-test', 100, 8, 32400)
        ans = [eeg_vm.shape, emg_vm.shape]
        
        np.testing.assert_array_equal(exp, ans)


    def test_read_voltage_matrices_edf(self):
        exp = [[31, 1024], [31, 1024]]

        (eeg_vm, emg_vm, _) = stage.read_voltage_matrices('../data/edf_test', '09', 128, 8, 31)
        ans = [eeg_vm.shape, emg_vm.shape]
        
        np.testing.assert_array_equal(exp, ans)


    def test_read_voltage_matrices_edf_with_startdatetime(self):
        exp = [[28, 1024], [28, 1024]]

        # 2020/3/12/ 14:31:00 (20 sec offset from the meas_date) 
        # This start datetime allows extract up to 28 epochs from the test EDF file
        start_datetime = datetime.datetime(2020, 3, 12, 14, 31, 00) 
        (eeg_vm, emg_vm, _) = stage.read_voltage_matrices('../data/edf_test', '09', 128, 8, 28, start_datetime)
        ans = [eeg_vm.shape, emg_vm.shape]
        
        np.testing.assert_array_equal(exp, ans)


    def test_spectrum_normalize(self):
        exp = np.load('NormPSD_EEG_0-9thRow_ID37963_FASTER_20191101_EEG_2019-013.npy')
        test_vm = np.load('Voltage_EEG_0-9thRow_ID37963_FASTER_20191101_EEG_2019-013.npy')

        ans = stage.spectrum_normalize(test_vm, 256, 100)

        np.testing.assert_array_equal(exp, ans)

if __name__ == "__main__":
    unittest.main()
