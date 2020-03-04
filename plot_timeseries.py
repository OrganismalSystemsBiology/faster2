# -*- coding: utf-8 -*-
import os
import argparse


import numpy as np
import pandas as pd

import stage
import timeseries_graph as tg

import concurrent.futures
from datetime import datetime

EPOCH_LEN_SEC = 8

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="path to the directory of input data")
    parser.add_argument("result_dir", help="path to the directory of results")

    args = parser.parse_args()

    data_dir = args.data_dir
    exp_info_df = stage.read_exp_info(data_dir)
    (epoch_num, sample_freq, exp_label, rack_label, start_datetime, end_datetime) = stage.interpret_exp_info(exp_info_df)

    stage_dir = os.path.join(args.result_dir)

    mouse_info_df = stage.read_mouse_info(data_dir)
    for i, r in mouse_info_df.iterrows():
        device_id = r[0]
        note = r[4]

        (eeg_vm_org, emg_vm_org, flag) = stage.read_voltage_matrices(
            data_dir, device_id, epoch_num, sample_freq, EPOCH_LEN_SEC)

        eeg_vm_norm = (eeg_vm_org - np.nanmean(eeg_vm_org))/np.nanstd(eeg_vm_org)
        emg_vm_norm = (emg_vm_org - np.nanmean(emg_vm_org))/np.nanstd(emg_vm_org)

        stage_filepath = os.path.join(stage_dir, f'{device_id}.faster2.stage.csv')
        stage_df = pd.read_csv(stage_filepath, skiprows=7, header=None, engine='python')
        tp = tg.Timeseries_plot(eeg_vm_norm, emg_vm_norm, stage_df, device_id, start_datetime, sample_freq)

        dt_now = datetime.now()
        print(f'started: {dt_now}')
        
        plot_dir = os.path.join(args.result_dir, 'figure', 'voltage', device_id)
        os.makedirs(plot_dir, exist_ok=True)
        os.chdir(plot_dir)
        for i in range(1, int(np.ceil(eeg_vm_norm.shape[0]/45)+1)):
            tp.plot_timeseries_a_page(i)
        elapsed_time = (datetime.now() - dt_now)
        print(f'ended: {elapsed_time.total_seconds()}')


    #res[0].savefig(f'{transmitterID}.{epoch_nums[0]:06}.png', pad_inches=0, bbox_inches='tight', dpi=100)
