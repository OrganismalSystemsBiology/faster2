# -*- coding: utf-8 -*-
import unittest
import numpy as np
import sys
import datetime
import pickle
sys.path.append('../')
import plot_summary as ps
import stage

class  TestFunctions(unittest.TestCase):
    """Test class for plot_summary.py
    """
    def setUp(self):
        # dummy stage calls of one day (24 hours)
        self.stage_call = np.array(
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

        ans = ps.stagetime_profile(self.stage_call)

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
        
        ans = ps.stagetime_circadian_profile(self.stage_call)

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
        
        stagetime_df = ps.make_summary_stats(mouse_info_df, epoch_num)['stagetime']
        ans = stagetime_df['Device label']

        np.testing.assert_array_equal(ans, exp)


if __name__ == "__main__":
    unittest.main()