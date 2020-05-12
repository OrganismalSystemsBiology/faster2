# -*- coding: utf-8 -*-
import unittest
import numpy as np
import pandas as pd
import sys
import datetime
import pickle
sys.path.append('../')
import plot_summary as ps
import stage
import eeg_tools as et
import os

class  TestFunctions(unittest.TestCase):
    """Test class for plot_summary.py
    """
    @classmethod
    def setUpClass(cls):
        # dummy stage calls of one day (24 hours)
        cls.stage_call_dummy = np.array(
            ['REM']*450 + ['NREM']*0   + ['WAKE']*0   + ['UNKNOWN']*0 +  # 0
            ['REM']*0   + ['NREM']*450 + ['WAKE']*0   + ['UNKNOWN']*0 +  # 1
            ['REM']*0   + ['NREM']*0   + ['WAKE']*450 + ['UNKNOWN']*0 +  # 2
            ['REM']*440 + ['NREM']*0   + ['WAKE']*0   + ['UNKNOWN']*10+  # 3
            ['REM']*450 + ['NREM']*0   + ['WAKE']*0   + ['UNKNOWN']*0 +  # 4
            ['REM']*450 + ['NREM']*0   + ['WAKE']*0   + ['UNKNOWN']*0 +  # 5
            ['REM']*450 + ['NREM']*0   + ['WAKE']*0   + ['UNKNOWN']*0 +  # 6
            ['REM']*450 + ['NREM']*0   + ['WAKE']*0   + ['UNKNOWN']*0 +  # 7
            ['REM']*450 + ['NREM']*0   + ['WAKE']*0   + ['UNKNOWN']*0 +  # 8
            ['REM']*450 + ['NREM']*0   + ['WAKE']*0   + ['UNKNOWN']*0 +  # 9
            ['REM']*450 + ['NREM']*0   + ['WAKE']*0   + ['UNKNOWN']*0 +  #10
            ['REM']*450 + ['NREM']*0   + ['WAKE']*0   + ['UNKNOWN']*0 +  #11
            ['REM']*450 + ['NREM']*0   + ['WAKE']*0   + ['UNKNOWN']*0 +  #12
            ['REM']*450 + ['NREM']*0   + ['WAKE']*0   + ['UNKNOWN']*0 +  #13
            ['REM']*450 + ['NREM']*0   + ['WAKE']*0   + ['UNKNOWN']*0 +  #14
            ['REM']*450 + ['NREM']*0   + ['WAKE']*0   + ['UNKNOWN']*0 +  #15
            ['REM']*450 + ['NREM']*0   + ['WAKE']*0   + ['UNKNOWN']*0 +  #16
            ['REM']*450 + ['NREM']*0   + ['WAKE']*0   + ['UNKNOWN']*0 +  #17
            ['REM']*450 + ['NREM']*0   + ['WAKE']*0   + ['UNKNOWN']*0 +  #18
            ['REM']*450 + ['NREM']*0   + ['WAKE']*0   + ['UNKNOWN']*0 +  #19
            ['REM']*450 + ['NREM']*0   + ['WAKE']*0   + ['UNKNOWN']*0 +  #20
            ['REM']*450 + ['NREM']*0   + ['WAKE']*0   + ['UNKNOWN']*0 +  #21
            ['REM']*450 + ['NREM']*0   + ['WAKE']*0   + ['UNKNOWN']*0 +  #22
            ['REM']*450 + ['NREM']*0   + ['WAKE']*0   + ['UNKNOWN']*0    #23
        )

        # Sample EEG voltage matrix
        faster_dir = '../test/data/FASTER2_20200206_EEG_2019-023'
        data_dir = os.path.join(faster_dir, 'data')
        result_dir = os.path.join(faster_dir, 'result')
        device_id = 'ID33572'
        EPOCH_LEN_SEC = 8
        epoch_num = 1800
        start_datetime = "2020/2/7 08:00:00"
        cls.sample_freq = 100
        cls.n_fft = int(256 * cls.sample_freq/100) # assures frequency bins compatibe among different sampleling frequencies

        (cls.eeg_vm_org, _, _) = stage.read_voltage_matrices(
            data_dir, device_id, cls.sample_freq, EPOCH_LEN_SEC, epoch_num, start_datetime)

        # Sample stage call
        stage_call = et.read_stages(result_dir, device_id, 'faster2')
        cls.stage_call = stage_call[0:1800]

    def test_collect_mouse_info(self):
        faster_dir_list = ['../test/data/FASTER2_20200206_EEG_2019-023',
                           '../test/data/FASTER2_20200213_EEG_2019-024']

        exp = np.array(['ID46770', 'ID45764', 'ID47313', 'ID37963',
                        'ID33572', 'ID47481', 'ID47479', 'ID45791',
                        'ID47567', 'ID46890', 'ID42046', 'ID51292',
                        'ID47395', 'ID46501', 'ID47248', 'ID47348']
                       )
        mouse_info_collected = ps.collect_mouse_info_df(faster_dir_list)
        mouse_info_df = mouse_info_collected['mouse_info']

        res = mouse_info_df['Device label']

        np.testing.assert_array_equal(res, exp)

    
    def test_stagetime_in_a_day(self):
        nrem_call = ['NREM']*700
        rem_call  = ['REM']*30
        wake_call = ['WAKE']*700
        unknown_call = ['UNKNOWN']*10
        stage_call = np.array(nrem_call + rem_call + wake_call + unknown_call)

        exp = (30, 700, 700, 10)

        ans = ps.stagetime_in_a_day(stage_call)

        np.testing.assert_array_equal(exp, ans)


    def test_stagetime_profile(self):

        exp = np.array([
            [60, 0, 0, 60*440/450, 60, 60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60],
            [0, 60, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0, 0, 60,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        ])

        ans = ps.stagetime_profile(self.stage_call_dummy)

        np.testing.assert_array_equal(exp, ans)


    def test_stagetime_circadian_profile(self):
        exp = np.array([[
                [60, 0, 0, 60*440/450, 60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60],
                [0, 60, 0,          0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0,  0, 60,         0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ],
            [
                [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            ]])
        
        ans = ps.stagetime_circadian_profile(self.stage_call_dummy)

        np.testing.assert_array_equal(exp, ans)


    def test_transmat_from_stages(self):
        exp = np.array([
            [0., 1., 0.],
            [0., 1., 0.],
            [0.5, 0., 0.5]])

        stage_calls = np.array(['NREM', 'NREM', 'REM','WAKE','WAKE'])
        ans = ps.transmat_from_stages(stage_calls)

        np.testing.assert_array_equal(ans, exp)


    def test_swtrans_from_stages(self):
        exp = np.array([1/3, 1/2])

        stage_calls = np.array(['NREM', 'NREM', 'REM','WAKE','WAKE', 'NREM'])
        ans = ps.swtrans_from_stages(stage_calls)

        np.testing.assert_array_equal(ans, exp)


    def test_make_summary_stats(self):
        faster_dir_list = ['../test/data/FASTER2_20200206_EEG_2019-023',
                            '../test/data/FASTER2_20200213_EEG_2019-024']

        exp = np.array(['ID46770', 'ID45764', 'ID47313', 'ID37963',
                        'ID33572', 'ID47481', 'ID45791',
                        'ID47567', 'ID46890', 'ID42046', 'ID51292',
                        'ID47395', 'ID46501', 'ID47248', 'ID47348']
                       )
        mouse_info_collected = ps.collect_mouse_info_df(faster_dir_list)
        mouse_info_df = mouse_info_collected['mouse_info']
        epoch_num = mouse_info_collected['epoch_num']
        
        stagetime_df = ps.make_summary_stats(mouse_info_df, slice(0, epoch_num, None), 'faster2')['stagetime']
        ans = stagetime_df['Device label']

        np.testing.assert_array_equal(ans, exp)


    def test_log_psd_inv(self):
        exp_psd = stage.psd(self.eeg_vm_org[0,:], self.n_fft, self.sample_freq)
        
        snorm_psd_dict = stage.spectrum_normalize(self.eeg_vm_org, 256, 100)
        snorm_psd = snorm_psd_dict['psd'][0]
        nm = snorm_psd_dict['mean']
        nf = snorm_psd_dict['norm_fac']
        ans_psd = ps.log_psd_inv(snorm_psd, nf, nm)

        np.testing.assert_array_almost_equal(exp_psd, ans_psd)


    def test_conv_PSD_from_snorm_PSD(self):
        psd_mat = np.apply_along_axis(lambda y: stage.psd(y, self.n_fft, self.sample_freq), 1, self.eeg_vm_org)
        exp = psd_mat

        snorm = stage.spectrum_normalize(self.eeg_vm_org, self.n_fft, self.sample_freq)
        ans = ps.conv_PSD_from_snorm_PSD(snorm)

        np.testing.assert_array_almost_equal(exp, ans)
       
 
    def test_make_psd_stats(self):
        # conventional PSD from voltage matrix
        conv_psd = np.apply_along_axis(lambda y: stage.psd(y, self.n_fft, self.sample_freq), 1, self.eeg_vm_org)

        bidx_unknown = (self.stage_call == 'UNKNOWN')
        bidx_rem = (self.stage_call == 'REM') & (~bidx_unknown)
        bidx_nrem = (self.stage_call == 'NREM') & (~bidx_unknown)
        bidx_wake = (self.stage_call == 'WAKE') & (~bidx_unknown)
        exp_psd_mean_rem = np.apply_along_axis(np.mean, 0, conv_psd[bidx_rem, :])
        exp_psd_mean_nrem = np.apply_along_axis(np.mean, 0, conv_psd[bidx_nrem, :])
        exp_psd_mean_wake = np.apply_along_axis(np.mean, 0, conv_psd[bidx_wake, :])

        # a small mouse_info_df
        mif = pd.DataFrame({'Device label':['ID33572'], 
                    'Stats report':['Yes'], 
                    'Mouse group':['T287D'], 
                    'Mouse ID':'AAV0837_1',
                    'Experiment label': ['FASTER2_20200206_EEG_2019-023'], 
                    'FASTER_DIR':['../test/data/FASTER2_20200206_EEG_2019-023/']})

        # TEST
        df = ps.make_psd_profile(mif, 100, slice(0,1800,None), 'faster2')
        ans_psd_mean_rem = df.iloc[0][5:].tolist()
        ans_psd_mean_nrem = df.iloc[1][5:].tolist()
        ans_psd_mean_wake = df.iloc[2][5:].tolist()

        np.testing.assert_array_almost_equal(exp_psd_mean_rem, ans_psd_mean_rem)
        np.testing.assert_array_almost_equal(exp_psd_mean_nrem, ans_psd_mean_nrem)
        np.testing.assert_array_almost_equal(exp_psd_mean_wake, ans_psd_mean_wake)


if __name__ == "__main__":
    unittest.main()
