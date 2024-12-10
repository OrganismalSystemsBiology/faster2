# -*- coding: utf-8 -*-
import unittest
import os
import sys
sys.path.append('../')
import faster2lib.eeg_tools as et
import numpy as np
import datetime
from pathlib import Path

DATA_ROOT = 'data/'
FASTER_FOLDER = 'FASTER2_20200206_EEG_2019-023'
TRANSMITTER_ID = 'ID33572'

class TestFunctions(unittest.TestCase):

    def test_DataSet_read_vm_dafault(self):
        summary_dir = Path(r"../test/data/FASTER2_20240629_EEG_2024-033\summary_033_034")
        new_root_dir = r'../test/data/' # rewrite the root directory saved in the mouse_info_collected

        # DataSet required basic information about the measurement
        mouse_info_collected = et.load_collected_mouse_info(summary_dir)
        # basic parameters of the summary
        mouse_info_df = mouse_info_collected['mouse_info']
        epoch_num = mouse_info_collected['epoch_num']
        epoch_len_sec = mouse_info_collected['epoch_len_sec']
        stage_ext = mouse_info_collected['stage_ext'] # this is None for this test data
        sample_freq = mouse_info_collected['sample_freq']
        epoch_range_summarised_str = mouse_info_collected['epoch_range'] # this is empty for this test data

        # construct dataset with default (without epoch_range, stage_ext)
        dataset = et.DataSet(mouse_info_df, sample_freq, epoch_len_sec,
                    epoch_num, None, 3, stage_ext, new_root_dir)
        
        # Test 1: read EEG voltage matrix
        eeg_vm, emg_vm = dataset.load_vm(0)
        np.testing.assert_array_equal((1800, 1024), eeg_vm.shape)
        np.testing.assert_array_equal((1800, 1024), emg_vm.shape)


        # Test 2: read stages
        stages = dataset.load_stage(0)
        np.testing.assert_equal(np.sum(stages=='WAKE'), 660) # counted in another script


    def test_DataSet_read_vm_options(self):
        summary_dir = Path(r"../test/data/FASTER2_20240629_EEG_2024-033\summary_033_034")
        new_root_dir = r'../test/data/' # rewrite the root directory saved in the mouse_info_collected

        # DataSet required basic information about the measurement
        mouse_info_collected = et.load_collected_mouse_info(summary_dir)
        # basic parameters of the summary
        mouse_info_df = mouse_info_collected['mouse_info']
        epoch_num = mouse_info_collected['epoch_num']
        epoch_len_sec = mouse_info_collected['epoch_len_sec']
        stage_ext = mouse_info_collected['stage_ext'] # this is None for this test data
        sample_freq = mouse_info_collected['sample_freq']
        epoch_range_summarised_str = mouse_info_collected['epoch_range'] # this is empty for this test data

        # construct dataset with default witch custom epoch_range, stage_ext
        dataset = et.DataSet(mouse_info_df, sample_freq, epoch_len_sec,
                    epoch_num, slice(0, 100), 3, 'modified', new_root_dir)
        
        # Test 1: read EEG voltage matrix
        eeg_vm, emg_vm = dataset.load_vm(0)
        np.testing.assert_array_equal((100, 1024), eeg_vm.shape)
        np.testing.assert_array_equal((100, 1024), emg_vm.shape)

        # Test 2: read stages (check the contents)
        stages = dataset.load_stage(0)
        np.testing.assert_equal(np.sum(stages=='UNKNOWN'), 5) # counted in another script
    
        # Test 3: read stages (check the length)
        stages = dataset.load_stage(0)
        np.testing.assert_equal(len(stages), 100)

    def test_read_voltage_matrices_dsi(self):
        exp = [[450, 800], [450, 800]]

        (eeg_vm, emg_vm, _) = et.read_voltage_matrices(
            '../test/data/FASTER2_20200206_EEG_2019-023/data', 'ID33572', 100, 8, 450)
        ans = [eeg_vm.shape, emg_vm.shape]
        
        np.testing.assert_array_equal(exp, ans)


    def test_read_voltage_matrices_dsi_by_startdatetime(self):
        exp = [[450, 800], [450, 800]]

        start_datetime = datetime.datetime(2020, 2 , 7, 8, 0, 0)
        (eeg_vm, emg_vm, _) = et.read_voltage_matrices(
            '../test/data/FASTER2_20200206_EEG_2019-023/data', 'ID33572', 100, 8, 450, start_datetime)
        ans = [eeg_vm.shape, emg_vm.shape]
        
        np.testing.assert_array_equal(exp, ans)

    def test_read_voltage_matrices_pickle(self):
        exp = [[32400, 800], [32400, 800]]

        (eeg_vm, emg_vm, _) = et.read_voltage_matrices('../test/data', 'ID-test', 100, 8, 32400)
        ans = [eeg_vm.shape, emg_vm.shape]
        
        np.testing.assert_array_equal(exp, ans)


    def test_read_voltage_matrices_edf(self):
        exp = [[31, 1024], [31, 1024]]

        (eeg_vm, emg_vm, _) = et.read_voltage_matrices('../test/data/edf_test', '09', 128, 8, 31)
        ans = [eeg_vm.shape, emg_vm.shape]
        
        np.testing.assert_array_equal(exp, ans)


    def test_read_voltage_matrices_edf_with_startdatetime(self):
        exp = [[28, 1024], [28, 1024]]

        # 2020/3/12/ 14:31:00 (20 sec offset from the meas_date) 
        # This start datetime allows extract up to 28 epochs from the test EDF file
        start_datetime = datetime.datetime(2020, 3, 12, 14, 31, 00) 
        (eeg_vm, emg_vm, _) = et.read_voltage_matrices('../test/data/edf_test', '09', 128, 8, 28, start_datetime)
        ans = [eeg_vm.shape, emg_vm.shape]
        
        np.testing.assert_array_equal(exp, ans)


    def test_read_stages(self):
        res = et.read_stages(os.path.join(
                DATA_ROOT, 
                FASTER_FOLDER, 
                'data/auto.stage'), 
                f'{TRANSMITTER_ID}')
        
        exp = np.array(['NREM', 'WAKE', 'REM', 'UNKNOWN'])
        res = res[np.r_[0,30,8470,4190]]
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


    def test_read_mouse_info(self):
        mouse_info = et.read_mouse_info("../test/data/FASTER2_20200206_EEG_2019-023/data")

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


    def test_read_mouse_info_with_sub_ext(self):
        mouse_info = et.read_mouse_info("../test/data/FASTER2_20200206_EEG_2019-023/data", "part_of_other_exp")

        exp = np.array(["ID46770",
                        "ID45764",
                        "ID47313",
                        "ID37963"]
                       )

        res = mouse_info['Device label']

        np.testing.assert_array_equal(res, exp)

    
    def test_read_exp_info(self):
        exp_info = et.read_exp_info("../test/data/FASTER2_20200206_EEG_2019-023/data")

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
        ans = et.interpret_datetimestr('2018/12/12')
        assert exp == ans


    def test_interpret_datetimestr_with_hyphens(self):
        exp = datetime.datetime(2018,2,4)
        ans = et.interpret_datetimestr('2018-2-4')
        assert exp == ans


    def test_interpret_datetimestr_with_slash_and_colon(self):
        exp = datetime.datetime(2018,2,12, 12,29,31)
        ans = et.interpret_datetimestr('2018/2/12 12:29:31')
        assert exp == ans
        

    def test_interpret_datetimestr_without_delimiters(self):
        exp = datetime.datetime(2018,2,2)
        ans = et.interpret_datetimestr('20180202')
        assert exp == ans


    def test_interpret_datetimestr_without_delimiters2(self):
        exp = datetime.datetime(2018,2,12,7,59,59)
        ans = et.interpret_datetimestr('20180212075959')
        assert exp == ans



if __name__ == "__main__":
    unittest.main()