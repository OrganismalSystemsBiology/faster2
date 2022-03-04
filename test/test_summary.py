# -*- coding: utf-8 -*-
import unittest
import numpy as np
import pandas as pd
import sys
import datetime
import pickle
sys.path.append('../')
import summary as ps
import stage
import faster2lib.eeg_tools as et
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

    
    def test_test_two_samples_1(self):
        # case 1: (x,y)~normal dist, eqaul variance between x,y
        # rnorm(20)
        x = np.array([-2.11474504, 0.07514748,  1.08321138, 0.59016378,
                      -0.04564552, -2.62057918, -1.32851937, 1.71542292,
                      0.99186626, 0.32985441, -0.00738640, 0.67014891,
                      -0.07817505, 0.12347018, -0.04558761, -0.62279079,
                      1.05188492, 1.00611101,  1.30288529, -0.37993559])
        # rnorm(20)
        y = np.array([1.19839898, 1.47494754, -0.86085112, -1.74540238,
                      0.06592349, -1.32656423,  0.14253285, -0.23655643,
                      -2.41977403, 0.10170696, -1.34227134, -0.04263858,
                      -0.95247291, 1.27389145,  0.08989706, 0.91369910,
                      1.41455185, 0.37031156, -1.38824560, -0.48298429])
        exp = 0.4444081 # calculated in R by t.test(x,y, var.equal=T)$p.value
        ans = ps.test_two_sample(x,y)
        np.testing.assert_almost_equal(exp, ans['p_value'])
        np.testing.assert_equal("Student's t-test", ans['method'])
        np.testing.assert_equal("", ans['stars'])


    def test_test_two_samples_2(self):
        # case 2: (x,y)~normal dist, NOT eqaul variance between x,y
        # rnorm(20)
        x = np.array([-2.11474504, 0.07514748,  1.08321138, 0.59016378,
                      -0.04564552, -2.62057918, -1.32851937, 1.71542292,
                      0.99186626, 0.32985441, -0.00738640, 0.67014891,
                      -0.07817505, 0.12347018, -0.04558761, -0.62279079,
                      1.05188492, 1.00611101,  1.30288529, -0.37993559])
        # rnorm(20, sd=5)
        y = np.array([3.7625258, 2.9305043, -0.5908568, 1.4358666, -6.2935640,
                      3.1244294, 5.6344052, 7.3062716, -0.4502659,  0.3271298,
                      -0.8690113, -2.8706936, -1.9556485, 13.9742048, -0.7297829,
                      -2.4053616, 2.5806931, 6.8943658, 1.5190817,  6.1774943])

        exp = 0.08442955 # calculated in R by t.test(x,y)$p.value
        ans = ps.test_two_sample(x,y)
        np.testing.assert_almost_equal(exp, ans['p_value'])
        np.testing.assert_equal("Welch's t-test", ans['method'])
        np.testing.assert_equal("", ans['stars'])


    def test_test_two_samples_3(self):
        # case 3: x obeys normal dist. but y does not. Equal variance of (x,y)
        # rnorm(20)
        x = np.array([-2.11474504, 0.07514748,  1.08321138, 0.59016378,
                      -0.04564552, -2.62057918, -1.32851937, 1.71542292,
                      0.99186626, 0.32985441, -0.00738640, 0.67014891,
                      -0.07817505, 0.12347018, -0.04558761, -0.62279079,
                      1.05188492, 1.00611101,  1.30288529, -0.37993559])
        # rnorm(20) but this FAILS Shapiro test
        y = np.array([0.60874031, 0.27950822, 0.51005652, 0.19058967,
                      0.96623791 - 1.09896277 - 0.78564818, 0.95238928,
                      -0.39770532, 1.12683608, 0.51255756, 0.62453564,
                      -0.07906373, 0.67780482, 0.78982522 - 0.04997242,
                      -0.06509925 - 2.11578914, 0.70092554 - 1.08215608])

        exp = 0.9734088 # calculated in R by wilcox.test(x,y,exact=F, correct=F)$p.value
        ans = ps.test_two_sample(x,y)
        np.testing.assert_almost_equal(exp, ans['p_value'])
        np.testing.assert_equal("Wilcoxon test", ans['method'])
        np.testing.assert_equal("", ans['stars'])


    def test_test_two_samples_4(self):
        # case 4: In the case of data length < 3, normality cannot be tested.
        #         Therefore, Wilcoxon test should be selected.
        x = np.array([0.1596474, 0.1850897, 0.1764961, 0.2131736, 0.1999569,
                      0.1996548, 0.2321302, 0.1890020, 0.1666172, 0.1775435,
                      0.1846986])
        y = np.array([np.nan, np.nan, np.nan, 0.3164970, 0.1842207])

        exp = 0.4297953 # calculated in R by wilcox.test(x,y,exact=F, correct=F)$p.value
        ans = ps.test_two_sample(x,y)

        np.testing.assert_almost_equal(exp, ans['p_value'])
        np.testing.assert_equal("Wilcoxon test", ans['method'])
        np.testing.assert_equal("", ans['stars'])


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
       

    def test_make_target_psd_info(self):
        # a small mouse_info_df
        mif = pd.DataFrame({'Device label':['190815.M3no4'], 
                    'Stats report':['Yes'], 
                    'Mouse group':['M3'], 
                    'Mouse ID':'M3.3',
                    'Experiment label': ['FASTER2_UT_20180814_M3Dq'], 
                    'FASTER_DIR':['../test/data/FASTER2_UT_20180814_M3Dq/']})

        # TEST
        start_datetime = datetime.datetime(2019, 8, 15, 8, 0)
        psd_info_list = ps.make_target_psd_info(mif, slice(100,1800,None), self.sample_freq, 'faster2_1800', start_datetime)
        psd_info = psd_info_list[0]
        ans_conv_psd = psd_info['norm']

        # check 1: The length of stage_call is full length (not the targeted length) i.e. 1800
        np.testing.assert_equal(len(psd_info_list[0]['stage_call']), 1800)
        
        # check 2: Unknown epoch is only at index:186
        np.testing.assert_array_equal(np.where(psd_info_list[0]['bidx_unknown']), np.array([[186]]))
        
        # check 3: Because epoch_range is specified, index 0:100 should be all false in bidx_[rem, nrem, wake]
        stage_mat = np.array([psd_info_list[0]['bidx_rem'], psd_info_list[0]['bidx_nrem'], psd_info_list[0]['bidx_wake']])
        true_if_any_stage_found = np.apply_along_axis(lambda y: np.any(y), 0, stage_mat)
        np.testing.assert_equal(np.any(true_if_any_stage_found[0:100]), False)
        
        # check 4: Within the range (index:100-1800), if the all of bidx_[rem, nrem, wake] of an epoch are false, then the epoch is not a good epoch
        idx = np.where(~true_if_any_stage_found[100:])[0] + 100 + 8
        np.testing.assert_array_equal(np.array([158,  159,  162,  164,  174,  175,  176,  177,  178,  179,  180,
                                                181,  182,  183,  184,  185,  186,  187,  188,  189,  190,  191,
                                                192,  193,  194,  195,  196,  197,  198,  199,  203,  204,  205,
                                                206,  208,  210,  212,  218,  219,  221,  222,  223,  224,  229,
                                                230,  231,  232,  233,  234,  235,  238,  239,  240,  245,  246,
                                                289,  290,  293,  314,  335,  346,  347,  350,  362,  363,  364,
                                                371,  383,  389,  397,  401,  402,  403,  404,  412,  428,  470,
                                                475,  480,  647,  754,  757,  762,  776,  795,  833,  835,  838,
                                                839,  862,  866,  868,  871,  872,  890,  898,  900,  952,  959,
                                                960,  969,  970,  976,  993, 1018, 1020, 1021, 1034, 1035, 1040,
                                                1041, 1042, 1043, 1053, 1054, 1056, 1070, 1076, 1112, 1115, 1121,
                                                1122, 1123, 1143, 1213, 1377, 1378, 1379, 1380, 1390, 1394, 1415,
                                                1416, 1417, 1457, 1458, 1463, 1499, 1500, 1501, 1502, 1508, 1509,
                                                1511, 1515, 1516, 1517, 1518, 1519, 1523, 1528, 1529, 1531, 1534,
                                                1536, 1537, 1538, 1539, 1543, 1562, 1563, 1565, 1566, 1567, 1577,
                                                1624, 1755]), idx) # row numbers in the excel file
        # check 5: 'NOT of bidx_target' is the 'NOT of the good epochs (no staged epoch in the check 4)'
        np.testing.assert_array_equal(np.where(~psd_info_list[0]['bidx_target'])[0][100:]+8, idx)
        
        # check 6: loaded conv_psd's shape should be 1800,129 even there is one unknown
        np.testing.assert_equal(ans_conv_psd.shape, (1800, 129))


    def test_make_psd_domain(self):
        psd_profile_df = pd.read_csv("../test/data/FASTER2_20200306_EEG_2019-025/summary/psd_profile.csv")
        
        # test
        ans_df = ps.make_psd_domain(psd_profile_df)

        ans_slow = ans_df.iloc[0]['Slow']
        ans_delta_wo_slow = ans_df.iloc[0]['Delta w/o slow']
        ans_delta = ans_df.iloc[0]['Delta']
        ans_theta = ans_df.iloc[0]['Theta']

        # calculated by Excel
        exp_slow = 0.020354406
        exp_delta_wo_slow = 0.048376081
        exp_delta = 0.040733806
        exp_theta = 0.038871448

        np.testing.assert_almost_equal(ans_slow, exp_slow)
        np.testing.assert_almost_equal(ans_delta_wo_slow, exp_delta_wo_slow)
        np.testing.assert_almost_equal(ans_delta, exp_delta)
        np.testing.assert_almost_equal(ans_theta, exp_theta)

    
    def test_make_psd_stats(self):
        psd_profile_df = pd.read_csv("../test/data/FASTER2_20200306_EEG_2019-025/summary/psd_profile.csv")
        psd_domain_df = ps.make_psd_domain(psd_profile_df)

        # test
        ans_df = ps.make_psd_stats(psd_domain_df)
        ans_n = ans_df.iloc[0]['N']
        ans_mean = ans_df.iloc[0]['Mean']
        ans_sd = ans_df.iloc[0]['SD']
        ans_pvalue = ans_df.iloc[12]['Pvalue']
        ans_stars = ans_df.iloc[12]['Stars']
        ans_method = ans_df.iloc[12]['Method']

        # calculated by a different code
        exp_n = 3
        exp_mean = 0.016761628477375
        exp_sd = 0.00275619660337911
        exp_pvalue = 0.431024979258898
        exp_stars = ''
        exp_method = "Student's t-test"

        np.testing.assert_equal(ans_n, exp_n)
        np.testing.assert_almost_equal(ans_mean, exp_mean)
        np.testing.assert_almost_equal(ans_sd, exp_sd)
        np.testing.assert_almost_equal(ans_pvalue, exp_pvalue)
        np.testing.assert_equal(ans_stars, exp_stars)
        np.testing.assert_equal(ans_method, exp_method)

if __name__ == "__main__":
    unittest.main()
